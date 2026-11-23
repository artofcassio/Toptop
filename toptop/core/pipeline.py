"""Pipeline completo da retopologia (passos 1 a 9 do plano), sem Maya no meio.

``run_pipeline`` recebe a malha que o motor vai ver (talvez já reduzida pelo
Maya) e a malha de referência (o original), e faz, nesta ordem:

1. **validar e limpar** as duas (``cleanup.clean_mesh``); malha non-manifold
   aborta com ``PipelineError`` trazendo as arestas culpadas;
2. **separar as cascas** da malha do motor;
3. **redução**: quem reduz é a camada do Maya (``maya_retopologize``); aqui só
   fica a anotação;
4. **creases -> features**, com as bordas abertas como feature guia (sem
   crease nenhuma, as quinas acima de ``suggest_angle`` viram crease forte);
5. **simetria**: fica para a Fase 2 (a opção só gera uma nota);
6. **motor** em cada casca, com o número de quads dividido pela área. Casca
   com menos de ``MIN_ENGINE_TRIANGLES`` triângulos não vai para o motor (cada
   triângulo vira 3 quads, sem mudar a forma); casca plana roda no QuadWild
   sem o remesh (com ele o motor trava em superfície plana); se o motor falha
   numa casca, as outras seguem e a que falhou também vira quads direto;
7. **reprojeção** de cada casca na casca correspondente do original
   (``reproject.reproject``), para uma peça não grudar na vizinha;
8. **pós-processo**: hard edges = arestas do resultado sobre creases fortes;
   creases para o Film = todos os loops de feature (forte e guia, com os
   valores que o limiar escolhido lê como forte e guia);
9. **juntar** as cascas e **medir** contra a referência limpa.

As funções ``maya_*`` no fim do arquivo são a camada fina que lê a malha do
Maya, reduz uma cópia quando precisa e cria o objeto novo; elas importam
``maya`` dentro de cada função.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, TypeVar

import numpy as np

from ..engine import ENGINE_NAMES, get_engine
from ..engine.base import Density, Engine, EngineCancelled, EngineError, EngineUnavailable
from .cleanup import (
    DEFAULT_TRIANGLE_LIMIT,
    DEFAULT_WELD_TOL_REL,
    CleanResult,
    Shell,
    _corner_next,
    clean_mesh,
    compact_vertices,
    num_triangles,
    reduction_target,
    split_shells,
)
from .creases import (
    DEFAULT_STRONG_THRESHOLD,
    DEFAULT_SUGGEST_ANGLE,
    GUIDE_VALUE,
    STRONG_VALUE,
    FeatureSet,
    check_input,
    suggest_creases,
)
from .geom import point_grid
from .mesh import Mesh, face_centers
from .quality import QualityReport, Targets, evaluate, face_counts, heat_colors, match_features, vertex_deviation
from .reproject import KIND_CORNER, KIND_CURVE, KIND_SURFACE, KIND_UNTOUCHED, ReprojectResult
from .reproject import reproject as reproject_mesh

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]
_T = TypeVar("_T")

MODES = ("game", "film")
DETAIL_MIN = 1
DETAIL_MAX = 10
DETAIL_DEFAULT = 5
SYMMETRY_AXES = ("x", "y", "z")

#: lado do quad no nível de detalhe 1, como fração da diagonal
BASE_EDGE_REL = 0.1
#: cada nível de detalhe multiplica o lado do quad por este fator
EDGE_STEP = 0.75
#: o Film pede esta vezes mais quads que o Game
FILM_FACTOR = 2.5
#: mínimo de quads pedido ao motor por casca
MIN_SHELL_QUADS = 24
#: casca com menos triângulos que isto não vai para o motor (vira quads direto)
MIN_ENGINE_TRIANGLES = 8
#: casca com todos os vértices a até esta fração da diagonal de um plano é plana
FLAT_TOL_REL = 1e-4
#: o que aconteceu com cada casca (``PipelineResult.shell_status``)
SHELL_ENGINE = "motor"
SHELL_DIRECT = "direto"
SHELL_FAILED = "falhou"
#: metas por casca que também valem para o resultado geral (polos, proporção e
#: tempo dependem do tamanho da casca e ficam só no relatório dela)
SHELL_REQUIRED_CHECKS = ("quads", "crease_coverage", "feature_fidelity")
#: de quanto em quanto tempo (s) a interface é atualizada durante uma etapa longa
PUMP_INTERVAL_S = 0.1
#: passadas da reprojeção (a segunda acha mais vértices de loop)
REPROJECT_ITERATIONS = 2
#: amostras do Hausdorff no relatório geral (cada casca recebe a sua parte)
REPORT_SAMPLES = 50_000
MIN_SHELL_SAMPLES = 2_000
#: maior tolerância de solda aceita (acima disso a solda fica lenta e destrói detalhe)
MAX_WELD_TOL_REL = 1e-2

_NO_PROBLEM_MESSAGES = ("Malha limpa: nada a corrigir.", "Entrada sem problemas de crease.")


# --------------------------------------------------------------------------
# Opções, erro e resultado
# --------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    """Inteiro de verdade (``bool`` não conta)."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_number(value: Any) -> bool:
    """Número real finito (``bool`` não conta)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return math.isfinite(float(value))


def _check_mode(mode: Any) -> None:
    """Levanta ``ValueError`` se o modo não é ``game`` nem ``film``."""
    if mode not in MODES:
        raise ValueError(f"modo inválido: {mode!r} (opções: {', '.join(MODES)})")


def _check_detail(detail: Any) -> None:
    """Levanta ``ValueError`` se o detalhe não é inteiro de 1 a 10."""
    if not _is_int(detail) or not DETAIL_MIN <= int(detail) <= DETAIL_MAX:
        raise ValueError(f"detalhe inválido: {detail!r} (inteiro de {DETAIL_MIN} a {DETAIL_MAX})")


@dataclass
class PipelineOptions:
    """Tudo o que o usuário escolhe para uma execução.

    ``mode``: ``game`` (hard edges nas creases fortes) ou ``film`` (2,5x mais
    quads, tudo suave). ``detail`` (1 a 10) escolhe a densidade, a não ser que
    ``target_quads`` seja dado. ``strong_threshold``: crease a partir do qual a
    aresta é forte. ``suggest_angle``: sem nenhuma crease marcada, as quinas
    acima deste ângulo viram crease forte (180 desliga). ``reduce_limit``:
    acima deste número de triângulos, o Maya reduz uma cópia antes do motor.
    ``engine``: nome do motor ou uma instância pronta (``engine_options`` só
    vale para o nome). ``symmetry`` ("x", "y" ou "z") fica para a Fase 2.
    """

    mode: str = "game"
    detail: int = DETAIL_DEFAULT
    target_quads: int | None = None
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD
    hard_edges: bool = True
    suggest_angle: float = DEFAULT_SUGGEST_ANGLE
    reduce_limit: int = DEFAULT_TRIANGLE_LIMIT
    weld_tol_rel: float = DEFAULT_WELD_TOL_REL
    reproject: bool = True
    engine: "str | Engine" = "quadwild"
    engine_options: dict = field(default_factory=dict)
    timeout_s: float = 600.0
    symmetry: str | None = None

    def validate(self) -> None:
        """Confere os valores; levanta ``ValueError`` explicando o primeiro problema."""
        _check_mode(self.mode)
        _check_detail(self.detail)
        if self.target_quads is not None and (not _is_int(self.target_quads) or self.target_quads < 1):
            raise ValueError(f"target_quads inválido: {self.target_quads!r} (inteiro >= 1 ou None)")
        if not _is_number(self.strong_threshold) or not self.strong_threshold > 0:
            raise ValueError(f"limiar de crease forte inválido: {self.strong_threshold!r} (número > 0)")
        if not isinstance(self.hard_edges, (bool, np.bool_)):
            raise ValueError(f"hard_edges deve ser verdadeiro ou falso, não {self.hard_edges!r}")
        if not _is_number(self.suggest_angle) or not 0 < self.suggest_angle <= 180:
            raise ValueError(f"ângulo de sugestão inválido: {self.suggest_angle!r} (entre 0 e 180 graus)")
        if not _is_int(self.reduce_limit) or self.reduce_limit < 1:
            raise ValueError(f"limite de redução inválido: {self.reduce_limit!r} (inteiro >= 1)")
        if not _is_number(self.weld_tol_rel) or not 0 <= self.weld_tol_rel <= MAX_WELD_TOL_REL:
            raise ValueError(
                f"tolerância de solda inválida: {self.weld_tol_rel!r} (entre 0 e {MAX_WELD_TOL_REL:g})"
            )
        if not isinstance(self.reproject, (bool, np.bool_)):
            raise ValueError(f"reproject deve ser verdadeiro ou falso, não {self.reproject!r}")
        if isinstance(self.engine, str):
            if self.engine not in ENGINE_NAMES:
                raise ValueError(f"motor desconhecido: {self.engine!r} (opções: {', '.join(ENGINE_NAMES)})")
        elif not isinstance(self.engine, Engine):
            raise ValueError(f"engine deve ser o nome de um motor ou um Engine, não {self.engine!r}")
        if not isinstance(self.engine_options, dict):
            raise ValueError("engine_options deve ser um dicionário")
        if not _is_number(self.timeout_s) or not self.timeout_s > 0:
            raise ValueError(f"tempo limite inválido: {self.timeout_s!r} (segundos > 0)")
        if self.symmetry is not None and self.symmetry not in SYMMETRY_AXES:
            raise ValueError(f"eixo de simetria inválido: {self.symmetry!r} (x, y, z ou None)")


class PipelineError(RuntimeError):
    """Falha do pipeline antes do motor (por exemplo, malha non-manifold).

    ``edges`` (K, 2) e ``vertices`` são os culpados, na numeração da malha de
    entrada indicada por ``stage``: ``validar`` = referência (o original),
    ``reduzir`` = malha entregue ao motor (a versão reduzida).
    """

    def __init__(
        self,
        message: str,
        edges: np.ndarray | None = None,
        stage: str = "",
        vertices: np.ndarray | None = None,
    ) -> None:
        super().__init__(message)
        self.edges = (
            np.zeros((0, 2), dtype=np.int64) if edges is None else np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        )
        self.vertices = (
            np.zeros(0, dtype=np.int64) if vertices is None else np.asarray(vertices, dtype=np.int64).reshape(-1)
        )
        self.stage = stage


@dataclass(eq=False)
class PipelineResult:
    """Malha final (todas as cascas juntas) e o que vai para o Maya.

    ``hard_edges`` e ``crease_edges`` são pares de vértices de ``mesh``;
    ``report`` mede ``mesh`` contra a referência limpa e ``shell_reports`` traz
    um relatório por casca, contra a casca correspondente da referência.
    ``report.passed`` também exige que nenhuma casca tenha falhado no motor e
    que cada casca do motor passe em ``SHELL_REQUIRED_CHECKS``.
    ``seconds`` é o tempo total; ``engine_seconds``, só o do motor.
    """

    mesh: Mesh
    hard_edges: np.ndarray
    crease_edges: FeatureSet
    report: QualityReport
    shell_reports: list[QualityReport] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0
    engine_seconds: float = 0.0
    #: total de quads pedido ao motor e a parte de cada casca (0 = não foi para o motor)
    target_quads: int = 0
    shell_targets: list[int] = field(default_factory=list)
    #: ``SHELL_ENGINE``, ``SHELL_DIRECT`` (pequena demais: quads direto) ou
    #: ``SHELL_FAILED`` (o motor falhou: quads direto) para cada casca
    shell_status: list[str] = field(default_factory=list)
    #: distância de cada vértice do resultado até a superfície original (mapa de calor)
    vertex_deviation: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: distância que vira vermelho no mapa de calor: a meta de desvio da forma
    heat_scale: float = 0.0


# --------------------------------------------------------------------------
# Densidade
# --------------------------------------------------------------------------


def surface_area(mesh: Mesh) -> float:
    """Área total das faces (Newell), sem montar a topologia."""
    if mesh.num_faces == 0:
        return 0.0
    p = mesh.points[mesh.connects]
    p = p - p.mean(axis=0)
    q = p[_corner_next(mesh)]
    vec = np.add.reduceat(np.cross(p, q), mesh.offsets[:-1], axis=0)
    return float(0.5 * np.linalg.norm(vec, axis=1).sum())


def estimate_from(area: float, diagonal: float, detail: int, mode: str = "game") -> int:
    """Número de quads para um nível de detalhe (1 a 10), dados a área e a diagonal.

    Lado do quad L = diagonal x 0,1 x 0,75^(detalhe - 1); quads = área / L²;
    o Film pede 2,5 vezes mais. Nunca menos que 24.
    """
    _check_detail(detail)
    _check_mode(mode)
    area, diagonal = float(area), float(diagonal)
    if not diagonal > 0 or not area > 0 or not math.isfinite(area) or not math.isfinite(diagonal):
        return MIN_SHELL_QUADS
    edge = diagonal * BASE_EDGE_REL * EDGE_STEP ** (int(detail) - 1)
    quads = area / (edge * edge)
    if mode == "film":
        quads *= FILM_FACTOR
    return max(MIN_SHELL_QUADS, int(round(quads)))


def estimate_quads(mesh: Mesh, detail: int, mode: str = "game") -> int:
    """``estimate_from`` com a área e a diagonal da malha. Rápido: não monta a topologia."""
    _check_detail(detail)
    _check_mode(mode)
    return estimate_from(surface_area(mesh), mesh.diagonal(), detail, mode)


def shell_targets(total: int, areas: np.ndarray, minimum: int = MIN_SHELL_QUADS) -> list[int]:
    """Divide ``total`` quads entre as cascas pela área (cada uma com pelo menos ``minimum``).

    Sem área nenhuma, divide em partes iguais.
    """
    areas = np.asarray(areas, dtype=np.float64).reshape(-1)
    if areas.size == 0:
        return []
    weights = np.where(np.isfinite(areas) & (areas > 0), areas, 0.0)
    if not weights.sum() > 0:
        weights = np.ones(areas.size)
    share = float(total) * weights / weights.sum()
    return [max(int(minimum), int(round(s))) for s in share.tolist()]


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def _strong_value(strong_threshold: float) -> float:
    """Valor de crease que conta como forte com este limiar."""
    return max(STRONG_VALUE, float(strong_threshold))


def _guide_value(strong_threshold: float) -> float:
    """Valor de crease que conta como guia com este limiar."""
    return min(GUIDE_VALUE, 0.5 * float(strong_threshold))


def transfer_features(
    target: Mesh,
    source: Mesh,
    features: FeatureSet,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
    match_tol_rel: float = 0.01,
) -> FeatureSet:
    """Leva as features de ``source`` para as arestas de ``target`` que correm em cima delas.

    Serve para passar as creases do original para a versão reduzida (ou o
    contrário) quando os índices dos vértices não batem. Usa
    ``quality.match_features``; cada aresta achada sai como forte ou guia
    conforme a crease que ela segue.
    """
    if len(features) == 0 or target.num_faces == 0 or source.num_points == 0:
        return FeatureSet.empty()
    valid = ((features.edges >= 0) & (features.edges < source.num_points)).all(axis=1)
    feats = FeatureSet(features.edges[valid], features.crease[valid])
    diagonal = source.diagonal()
    if len(feats) == 0 or not diagonal > 0:
        return FeatureSet.empty()
    match = match_features(
        target, source.points, feats, diagonal, strong_threshold, match_tol_rel=match_tol_rel
    )
    if not match.applicable or len(match.matched_edges) == 0:
        return FeatureSet.empty()
    values = np.where(match.matched_strong, _strong_value(strong_threshold), _guide_value(strong_threshold))
    return FeatureSet(match.matched_edges, values)


def _suggested(mesh: Mesh, options: PipelineOptions, notes: list[str]) -> FeatureSet:
    """Quinas acima do ângulo de sugestão como creases fortes (com a nota para o usuário)."""
    pairs = suggest_creases(mesh, float(options.suggest_angle))
    if len(pairs):
        notes.append(
            f"Nenhuma crease marcada: {len(pairs)} aresta(s) com quina acima de "
            f"{options.suggest_angle:g}° viraram crease forte (sugestão automática)."
        )
    else:
        notes.append(
            f"Nenhuma crease marcada e nenhuma quina acima de {options.suggest_angle:g}°: "
            "só as bordas abertas (se houver) viram feature."
        )
    return FeatureSet(pairs, np.full(len(pairs), _strong_value(options.strong_threshold)))


# --------------------------------------------------------------------------
# Auxiliares
# --------------------------------------------------------------------------


def _report(progress: ProgressFn | None, fraction: float, message: str) -> None:
    """Chama o callback de progresso, se houver."""
    if progress is not None:
        progress(float(min(max(fraction, 0.0), 1.0)), message)


def _check_cancel(cancel: CancelFn | None) -> None:
    """Levanta ``EngineCancelled`` se o usuário pediu para parar."""
    if cancel is not None and cancel():
        raise EngineCancelled("execução cancelada pelo usuário")


def resolve_engine(options: PipelineOptions) -> Engine:
    """O motor das opções: a instância dada ou ``get_engine(nome, timeout_s=..., **engine_options)``."""
    if isinstance(options.engine, Engine):
        return options.engine
    extra = dict(options.engine_options)
    extra.setdefault("timeout_s", float(options.timeout_s))
    return get_engine(options.engine, **extra)


def _input_edges(input_mesh: Mesh, vertex_map: np.ndarray, clean_pairs: np.ndarray) -> np.ndarray:
    """Arestas da malha de entrada que viraram os pares dados da malha limpa."""
    clean_pairs = np.asarray(clean_pairs, dtype=np.int64).reshape(-1, 2)
    if len(clean_pairs) == 0 or input_mesh.num_faces == 0:
        return np.zeros((0, 2), dtype=np.int64)
    edges = input_mesh.topology.edges
    mapped = vertex_map[edges]
    valid = (mapped >= 0).all(axis=1)
    base = int(max(vertex_map.max(), clean_pairs.max())) + 1
    keys = mapped.min(axis=1) * base + mapped.max(axis=1)
    wanted = clean_pairs.min(axis=1) * base + clean_pairs.max(axis=1)
    return edges[valid & np.isin(keys, wanted)].copy()


def _require_manifold(input_mesh: Mesh, clean: CleanResult, stage: str, what: str) -> None:
    """Levanta ``PipelineError`` com as arestas e vértices culpados se a malha é non-manifold."""
    if clean.ok:
        return
    edges = _input_edges(input_mesh, clean.vertex_map, clean.nonmanifold_edges)
    vertices = np.flatnonzero(np.isin(clean.vertex_map, clean.nonmanifold_vertices) & (clean.vertex_map >= 0))
    parts = []
    if len(clean.nonmanifold_edges):
        parts.append(f"{len(clean.nonmanifold_edges)} aresta(s) com mais de 2 faces")
    if len(clean.nonmanifold_vertices):
        parts.append(f"{len(clean.nonmanifold_vertices)} vértice(s) ligando faces só pela ponta")
    message = f"{what} é non-manifold ({' e '.join(parts)}). Corrija a malha antes de retopologizar."
    if stage == "reduzir":
        message += (
            " Aumente o limite de redução na janela (Limite de redução) para o Maya reduzir menos;"
            " por script, passe PipelineOptions(reduce_limit=...)."
        )
    raise PipelineError(message, edges=edges, stage=stage, vertices=vertices)


def _clean(
    mesh: Mesh, features: FeatureSet | None, options: PipelineOptions, stage: str, what: str
) -> CleanResult:
    """Limpa uma cópia e aborta se ela não tem faces ou é non-manifold."""
    if mesh.num_faces == 0:
        raise PipelineError(f"{what} não tem faces.", stage=stage)
    clean = clean_mesh(
        mesh,
        features,
        weld_tol_rel=float(options.weld_tol_rel),
        triangulate=True,
        border_value=_guide_value(options.strong_threshold),
    )
    if clean.mesh.num_faces == 0:
        raise PipelineError(f"{what} não tem nenhuma face válida depois da limpeza.", stage=stage)
    _require_manifold(mesh, clean, stage, what)
    return clean


def _merge(meshes: list[Mesh], name: str) -> tuple[Mesh, np.ndarray]:
    """Junta as malhas num só objeto; devolve também o primeiro vértice de cada uma."""
    sizes = np.array([m.num_points for m in meshes], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    if not meshes:
        return Mesh(np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), name), offsets
    merged = Mesh(
        np.concatenate([m.points for m in meshes]),
        np.concatenate([m.counts for m in meshes]),
        np.concatenate([m.connects + offsets[i] for i, m in enumerate(meshes)]),
        name,
    )
    return merged, offsets


def _reference_shell_index(engine_shells: list[Shell], ref_shells: list[Shell], ref_mesh: Mesh) -> list[int]:
    """Casca da referência que corresponde a cada casca do motor (voto dos vértices mais próximos)."""
    if len(ref_shells) == 1:
        return [0] * len(engine_shells)
    label = np.full(ref_mesh.num_points, -1, dtype=np.int64)
    for j, shell in enumerate(ref_shells):
        label[shell.vertex_ids] = j
    used = np.flatnonzero(label >= 0)
    grid = point_grid(ref_mesh.points[used])
    out = []
    for shell in engine_shells:
        _, idx, _, _ = grid.query(shell.mesh.points)
        votes = np.bincount(label[used[idx[idx >= 0]]], minlength=len(ref_shells))
        out.append(int(np.argmax(votes)))
    return out


def _run_engine(
    engine: Engine,
    shell: Shell,
    target: int,
    label: str,
    cancel: CancelFn | None,
    progress: ProgressFn | None,
) -> Mesh:
    """Roda o motor numa casca; erros saem como ``EngineError`` com o nome da casca."""
    try:
        result = engine.remesh(shell.mesh, shell.features, Density(int(target)), cancel=cancel, progress=progress)
    except EngineCancelled:
        raise
    except EngineError as error:
        message = str(error).split("\nPasta de diagnóstico")[0]
        try:
            renamed = type(error)(f"{label}: {message}", getattr(error, "workdir", None), getattr(error, "log", ""))
        except Exception:  # subclasse com outro construtor: mantém o erro original
            raise error
        raise renamed from error
    except Exception as error:
        raise EngineError(f"{label}: falha no motor {engine.name or type(engine).__name__}: {error}") from error
    if not isinstance(result, Mesh) or result.num_faces == 0:
        raise EngineError(f"{label}: o motor {engine.name or type(engine).__name__} devolveu uma malha vazia")
    return result


def _run_shell(
    engine: Engine,
    shell: Shell,
    target: int,
    label: str,
    cancel: CancelFn | None,
    progress: ProgressFn | None,
) -> Mesh:
    """Roda o motor numa casca; um motor que roda dentro do Maya fica sem desfazer e sem eventos.

    A malha temporária do motor do Maya não pode entrar na fila de desfazer.
    Com o desfazer desligado, nada pode processar eventos da interface (o
    usuário mexeria na cena sem o desfazer gravar), então esse motor roda sem
    ``progress`` e sem ``cancel``, que processam eventos na janela.
    """
    if not getattr(engine, "requires_maya", False):
        return _run_engine(engine, shell, target, label, cancel, progress)
    with _maya_undo_suspended():
        return _run_engine(engine, shell, target, label, None, None)


def _pumped(
    work: Callable[[], _T],
    progress: ProgressFn | None,
    cancel: CancelFn | None,
    fraction: float,
    message: str,
) -> _T:
    """Roda uma etapa longa (só NumPy, sem Maya) sem congelar a interface.

    Sem ``progress``, roda direto. Com ele, a etapa roda numa thread de
    trabalho enquanto esta chama ``progress`` a cada ``PUMP_INTERVAL_S`` (é aí
    que a janela processa eventos) e confere o cancelamento; cancelada, a
    função levanta ``EngineCancelled`` na hora e o resultado da thread é
    descartado.
    """
    if progress is None:
        return work()
    _report(progress, fraction, message)
    box: dict[str, Any] = {}

    def run() -> None:
        """Corpo da thread: guarda o resultado ou o erro."""
        try:
            box["value"] = work()
        except BaseException as error:  # repassado na thread principal
            box["error"] = error

    thread = threading.Thread(target=run, name="toptop-pipeline", daemon=True)
    thread.start()
    while True:
        thread.join(PUMP_INTERVAL_S)
        if not thread.is_alive():
            break
        _report(progress, fraction, message)
        _check_cancel(cancel)
    if "error" in box:
        raise box["error"]
    return box["value"]


def split_into_quads(mesh: Mesh) -> Mesh:
    """Cada face de n lados vira n quads: vértice, meio da aresta, centro da face, meio da aresta.

    Subdivisão linear, sem mover nada: a forma é a mesma, cada aresta original
    vira duas arestas em linha (uma crease continua sendo um loop) e o
    resultado é 100% quads, com o mesmo sentido das faces. Serve para a casca
    pequena demais para o motor e para a casca em que o motor falhou.
    """
    if mesh.num_faces == 0:
        empty = np.zeros(0, dtype=np.int64)
        return Mesh(np.zeros((0, 3)), empty, empty.copy(), mesh.name)
    topo = mesh.topology
    n = mesh.num_points
    edges = topo.edges
    mids = 0.5 * (mesh.points[edges[:, 0]] + mesh.points[edges[:, 1]])
    points = np.concatenate([mesh.points, mids, face_centers(mesh)])
    nxt = topo.corner_next
    prev = np.empty_like(nxt)
    prev[nxt] = np.arange(len(nxt), dtype=np.int64)
    quads = np.stack(
        [mesh.connects, n + topo.corner_edge, n + len(edges) + topo.corner_face, n + topo.corner_edge[prev]],
        axis=1,
    )
    out = Mesh(points, np.full(len(quads), 4, dtype=np.int64), quads.reshape(-1), mesh.name)
    return compact_vertices(out)[0]


def reverse_winding(mesh: Mesh) -> Mesh:
    """Cópia com todas as faces percorridas ao contrário (as normais apontam para o outro lado).

    Vértices e ordem das faces não mudam; as arestas também são as mesmas.
    """
    if mesh.num_faces == 0:
        return Mesh(mesh.points, mesh.counts, mesh.connects, mesh.name)
    start = np.repeat(mesh.offsets[:-1], mesh.counts)
    local = np.arange(mesh.connects.size, dtype=np.int64) - start
    target = start + np.repeat(mesh.counts, mesh.counts) - 1 - local
    connects = np.empty_like(mesh.connects)
    connects[target] = mesh.connects
    return Mesh(mesh.points, mesh.counts, connects, mesh.name)


def is_flat(mesh: Mesh, tol_rel: float = FLAT_TOL_REL) -> bool:
    """Verdadeiro se todos os vértices usados estão a até ``tol_rel * diagonal`` de um mesmo plano."""
    if mesh.num_faces == 0:
        return True
    pts = mesh.points[np.unique(mesh.connects)]
    diagonal = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    if len(pts) < 4 or not diagonal > 0:
        return True
    centered = pts - pts.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    return float(np.abs(centered @ vectors[:, 0]).max()) <= float(tol_rel) * diagonal


def _flat_shell_engine(engine: Engine) -> Engine:
    """Motor para uma casca plana: o QuadWild roda numa cópia sem o remesh; os outros, como estão.

    O remesh do QuadWild tira a escala da curvatura, que é zero numa
    superfície plana, e não termina (até o tempo limite). Sem ele, o motor
    trabalha direto nos triângulos da casca e termina em menos de um segundo.
    """
    from ..engine.quadwild import QuadWildEngine

    if isinstance(engine, QuadWildEngine) and engine.do_remesh:
        clone = copy.copy(engine)
        clone.do_remesh = False
        return clone
    return engine


def _reference_shells(
    shell_list: list[Shell], same: bool, ref_mesh: Mesh, ref_feats: FeatureSet
) -> tuple[list[Shell], list[int]]:
    """Cascas da referência e, para cada casca do motor, o índice da correspondente."""
    if same:
        return shell_list, list(range(len(shell_list)))
    if len(shell_list) == 1:
        whole = Shell(
            ref_mesh,
            ref_feats,
            np.arange(ref_mesh.num_points, dtype=np.int64),
            np.arange(ref_mesh.num_faces, dtype=np.int64),
        )
        return [whole], [0]
    ref_shells = split_shells(ref_mesh, ref_feats)
    return ref_shells, _reference_shell_index(shell_list, ref_shells, ref_mesh)


def _reprojection_notes(reps: list[ReprojectResult]) -> list[str]:
    """Resumo da reprojeção (com uma casca só, as mensagens dela mesma)."""
    if len(reps) == 1:
        return list(reps[0].messages)
    counts = {k: sum(int((r.kind == k).sum()) for r in reps) for k in (KIND_SURFACE, KIND_CURVE, KIND_CORNER)}
    notes = [
        f"Reprojeção (cada casca na casca correspondente do original): {counts[KIND_SURFACE]} vértice(s) "
        f"na superfície, {counts[KIND_CURVE]} em curvas de feature e {counts[KIND_CORNER]} em cantos."
    ]
    missing = sum(int(r.corners_missing) for r in reps)
    if missing:
        notes.append(
            f"{missing} canto(s) de feature sem vértice de loop por perto; a quina pode sair arredondada ou cortada."
        )
    far = sum(int((r.mesh.topology.used_vertex & (r.kind == KIND_UNTOUCHED)).sum()) for r in reps)
    if far:
        notes.append(
            f"{far} vértice(s) ficaram parados: o destino passava do deslocamento máximo da reprojeção."
        )
    return notes


def _unmeasured_report(piece: Mesh, name: str) -> QualityReport:
    """Relatório de uma casca convertida direto em quads: só as contagens (ela não é medida sozinha)."""
    counts = face_counts(piece)
    report = QualityReport(
        name=name,
        engine="direto",
        vertices=int(np.unique(piece.connects).size),
        faces=int(counts["faces"]),
        quads=int(counts["quads"]),
        triangles=int(counts["triangles"]),
        ngons=int(counts["ngons"]),
        quad_ratio=float(counts["quad_ratio"]),
        shells=1,
    )
    report.notes.append("Casca convertida direto em quads, sem o motor; ela entra na medida do resultado geral.")
    return report


def _list_shells(numbers: list[int], limit: int = 10) -> str:
    """Números das cascas para uma nota ("1, 4, 7" ou "1, 2, ... e mais 30")."""
    shown = ", ".join(str(k) for k in numbers[:limit])
    return shown if len(numbers) <= limit else f"{shown} e mais {len(numbers) - limit}"


def _one_line(error: BaseException) -> str:
    """Mensagem do erro numa linha só."""
    return " ".join(str(error).split())


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def run_pipeline(
    mesh: Mesh,
    features: FeatureSet | None,
    options: PipelineOptions,
    *,
    reference: Mesh | None = None,
    reference_features: FeatureSet | None = None,
    progress: ProgressFn | None = None,
    cancel: CancelFn | None = None,
) -> PipelineResult:
    """Roda os passos 1 a 9 e devolve a malha de quads com o relatório.

    ``mesh``/``features``: o que o motor vê (talvez a versão reduzida pelo
    Maya). ``reference``/``reference_features``: o original, usado na
    reprojeção e na medição; por padrão, ``mesh``/``features``. Features
    ``None`` = nenhuma crease marcada: as do outro lado são transferidas por
    geometria e, sem nenhuma, as quinas acima de ``suggest_angle`` viram crease
    forte; ``FeatureSet.empty()`` = sem creases de propósito. Nenhuma malha de
    entrada é alterada. Cancelar levanta ``EngineCancelled``; malha inválida
    levanta ``PipelineError``. Se o motor falha numa casca, as outras seguem
    (a que falhou vira quads direto e o relatório sai reprovado); só quando
    nenhuma casca passou pelo motor o primeiro ``EngineError`` (com o nome da
    casca) é levantado. Com ``progress``, as etapas longas rodam numa thread
    de trabalho e ``progress`` é chamado a cada ``PUMP_INTERVAL_S``.
    """
    start = time.perf_counter()
    options.validate()
    notes: list[str] = []
    threshold = float(options.strong_threshold)
    same = reference is None or reference is mesh
    ref_input = mesh if reference is None else reference
    if same and reference_features is None:
        reference_features = features

    # 1. validar e limpar (a referência primeiro: os culpados saem na numeração do original)
    ref_clean = _pumped(
        lambda: _clean(ref_input, reference_features, options, "validar", "A malha"),
        progress,
        cancel,
        0.0,
        "Validando e limpando a malha",
    )
    if same:
        eng_clean = ref_clean
    else:
        _check_cancel(cancel)
        eng_clean = _pumped(
            lambda: _clean(mesh, features, options, "reduzir", "A versão reduzida da malha"),
            progress,
            cancel,
            0.02,
            "Validando e limpando a versão reduzida",
        )
    notes.extend(m for m in ref_clean.messages if m not in _NO_PROBLEM_MESSAGES)
    if not same:
        notes.extend(f"Versão reduzida: {m}" for m in eng_clean.messages if m not in _NO_PROBLEM_MESSAGES)
    _check_cancel(cancel)

    # 3. redução: só a anotação (quem reduz é o Maya)
    if not same:
        notes.append(
            f"O motor recebeu uma versão reduzida ({num_triangles(eng_clean.mesh)} triângulos; "
            f"o original tem {num_triangles(ref_clean.mesh)})."
        )
    elif reduction_target(ref_clean.mesh, options.reduce_limit) is not None:
        notes.append(
            f"A malha tem {num_triangles(ref_clean.mesh)} triângulos, acima do limite de redução "
            f"({options.reduce_limit}); no Maya uma cópia é reduzida antes do motor, aqui ela foi inteira."
        )

    # 4. creases -> features (as bordas abertas já vieram como guia da limpeza)
    _report(progress, 0.04, "Preparando as creases")
    ref_feats = ref_clean.features
    eng_feats = eng_clean.features
    if same:
        if reference_features is None:
            ref_feats = eng_feats = ref_feats.merged(_suggested(ref_clean.mesh, options, notes))
    else:
        if reference_features is None:
            if features is not None:
                moved = transfer_features(ref_clean.mesh, eng_clean.mesh, eng_feats, threshold)
                ref_feats = ref_feats.merged(moved)
                notes.append(
                    f"As creases do original foram transferidas da versão reduzida ({len(moved)} aresta(s))."
                )
            else:
                ref_feats = ref_feats.merged(_suggested(ref_clean.mesh, options, notes))
        if features is None:
            moved = transfer_features(eng_clean.mesh, ref_clean.mesh, ref_feats, threshold)
            eng_feats = eng_feats.merged(moved)
            notes.append(
                f"As creases da versão reduzida foram transferidas do original ({len(moved)} aresta(s))."
            )
    check = check_input(ref_clean.mesh, ref_feats, float(options.suggest_angle), threshold)
    notes.extend(m for m in check.messages if m not in _NO_PROBLEM_MESSAGES)

    # 5. simetria: Fase 2
    if options.symmetry is not None:
        notes.append(f"Simetria fica para a Fase 2: o eixo {options.symmetry.upper()} foi ignorado.")

    # 2. cascas (com as features já completas) e a casca correspondente da referência
    shell_list = split_shells(eng_clean.mesh, eng_feats)
    if not shell_list:
        raise PipelineError("A malha não tem nenhuma casca com faces.", stage="validar")
    count = len(shell_list)
    if count > 1:
        notes.append(f"{count} cascas separadas: o motor roda uma vez em cada uma.")
    ref_shells, ref_index = _reference_shells(shell_list, same, ref_clean.mesh, ref_feats)
    areas = np.array([surface_area(s.mesh) for s in shell_list])
    total = int(options.target_quads) if options.target_quads is not None else estimate_quads(
        ref_clean.mesh, int(options.detail), options.mode
    )
    targets = shell_targets(total, areas)
    small = [s.mesh.num_faces < MIN_ENGINE_TRIANGLES for s in shell_list]
    flat = [not small[k] and is_flat(s.mesh) for k, s in enumerate(shell_list)]
    engine_targets = [t for t, tiny in zip(targets, small) if not tiny]
    if engine_targets and sum(engine_targets) > total * 1.05:
        notes.append(
            f"Com o mínimo de {MIN_SHELL_QUADS} quads por casca, o motor recebeu {sum(engine_targets)} quads "
            f"no total (pedido: {total})."
        )
    tiny_ids = [k + 1 for k in range(count) if small[k]]
    if tiny_ids:
        notes.append(
            f"{len(tiny_ids)} casca(s) com menos de {MIN_ENGINE_TRIANGLES} triângulos não foram para o motor: "
            f"cada triângulo virou 3 quads, sem mudar a forma (casca(s) {_list_shells(tiny_ids)})."
        )

    # 6. motor em cada casca
    engine = resolve_engine(options)
    engine_label = engine.name or type(engine).__name__
    flat_engine = _flat_shell_engine(engine) if any(flat) else engine
    if flat_engine is not engine:
        notes.append(
            f"{sum(flat)} casca(s) plana(s): o QuadWild rodou nelas sem o remesh, que trava em superfície plana "
            f"(casca(s) {_list_shells([k + 1 for k in range(count) if flat[k]])})."
        )
    lo, hi = 0.08, 0.80
    weights = np.where(small, 0.0, np.asarray(targets, dtype=np.float64))
    if not weights.sum() > 0:
        weights = np.ones(count)
    bounds = lo + (hi - lo) * np.concatenate([[0.0], np.cumsum(weights) / weights.sum()])
    results: list[Mesh] = []
    shell_seconds: list[float] = []
    status: list[str] = []
    failures: list[EngineError] = []
    for k, shell in enumerate(shell_list):
        _check_cancel(cancel)
        label = f"Casca {k + 1} de {count} ({shell.mesh.num_faces} triângulos)"
        if small[k]:
            results.append(split_into_quads(shell.mesh))
            shell_seconds.append(0.0)
            status.append(SHELL_DIRECT)
            continue
        a, b = float(bounds[k]), float(bounds[k + 1])

        def sub_progress(fraction: float, message: str, a: float = a, b: float = b, label: str = label) -> None:
            _report(progress, a + (b - a) * min(max(float(fraction), 0.0), 1.0), f"{label}: {message}")

        used = flat_engine if flat[k] else engine
        waiting = " (não dá para cancelar)" if getattr(used, "requires_maya", False) else ""
        _report(progress, a, f"{label}: motor {engine_label}, {targets[k]} quads{waiting}")
        before = getattr(used, "last_run", None)
        t0 = time.perf_counter()
        try:
            out = _run_shell(used, shell, targets[k], label, cancel, sub_progress if progress else None)
        except (EngineCancelled, EngineUnavailable):
            raise
        except EngineError as error:
            shell_seconds.append(time.perf_counter() - t0)
            failures.append(error)
            results.append(split_into_quads(shell.mesh))
            status.append(SHELL_FAILED)
            notes.append(
                f"{_one_line(error)} -> a casca foi convertida direto em quads (cada triângulo em 3); "
                "o resto seguiu normalmente."
            )
            continue
        shell_seconds.append(time.perf_counter() - t0)
        run = getattr(used, "last_run", None)
        details = getattr(run, "details", None) if run is not before else None
        if isinstance(details, dict) and details.get("warning"):
            notes.append(f"{label}: {details['warning']}")
        results.append(Mesh(out.points, out.counts, out.connects, mesh.name))
        status.append(SHELL_ENGINE)
    if failures and SHELL_ENGINE not in status:
        raise failures[0]
    engine_seconds = float(sum(shell_seconds))
    _check_cancel(cancel)

    # 7. reprojetar cada casca na casca correspondente do original (cada uma só
    # enxerga a própria superfície: peças com folga pequena não grudam uma na outra).
    # A casca convertida direto já está em cima do original, a não ser que tenha
    # vindo da versão reduzida.
    pieces = list(results)
    if options.reproject:
        reps: list[ReprojectResult] = []
        todo = [k for k in range(count) if status[k] == SHELL_ENGINE or not same]
        todo_faces = max(sum(results[k].num_faces for k in todo), 1)
        done_faces = 0
        for k in todo:
            _check_cancel(cancel)
            ref_shell = ref_shells[ref_index[k]]
            rep = _pumped(
                lambda out=results[k], ref_shell=ref_shell: reproject_mesh(
                    out, ref_shell.mesh, ref_shell.features, iterations=REPROJECT_ITERATIONS
                ),
                progress,
                cancel,
                0.80 + 0.08 * done_faces / todo_faces,
                "Reprojetando na superfície original" + (f" (casca {k + 1} de {count})" if count > 1 else ""),
            )
            reps.append(rep)
            pieces[k] = rep.mesh
            done_faces += results[k].num_faces
        if reps:
            notes.extend(_reprojection_notes(reps))
    _check_cancel(cancel)

    # 9. juntar e medir; 8. pós-processo a partir dos loops achados na medição.
    # Vértice sem face sai antes: o Maya o descartaria ao criar o objeto e as
    # hard edges, creases e cores (por índice) cairiam nos vértices errados.
    pieces = [compact_vertices(piece)[0] for piece in pieces]
    merged, _ = _merge(pieces, ref_input.name or mesh.name)
    measured: dict[str, Any] = {}
    report = _pumped(
        lambda: evaluate(
            merged,
            ref_clean.mesh,
            ref_feats,
            time_s=engine_seconds,
            strong_threshold=threshold,
            samples=REPORT_SAMPLES,
            name=merged.name,
            engine=engine_label,
            details=measured,
        ),
        progress,
        cancel,
        0.90,
        "Medindo a qualidade",
    )
    deviation = _pumped(
        lambda: vertex_deviation(merged, ref_clean.mesh),
        progress,
        cancel,
        0.94,
        "Calculando o mapa de calor",
    )
    match = measured.get("match")
    if match is not None and match.applicable and len(match.matched_edges):
        matched = match.matched_edges
        strong = np.asarray(match.matched_strong, dtype=bool)
        hard_edges = FeatureSet(matched[strong]).edges
        crease_edges = FeatureSet(
            matched, np.where(strong, _strong_value(threshold), _guide_value(threshold))
        )
    else:
        hard_edges = np.zeros((0, 2), dtype=np.int64)
        crease_edges = FeatureSet.empty()

    if count == 1:
        shell_reports = [report]
    else:
        share = areas / areas.sum() if areas.sum() > 0 else np.full(count, 1.0 / count)
        shell_reports = []
        for k, piece in enumerate(pieces):
            _check_cancel(cancel)
            if status[k] != SHELL_ENGINE:  # forma exata do original: a medida geral basta
                shell_reports.append(_unmeasured_report(piece, f"{merged.name} casca {k + 1}"))
                continue
            ref_shell = ref_shells[ref_index[k]]
            samples = int(np.clip(REPORT_SAMPLES * share[k], MIN_SHELL_SAMPLES, REPORT_SAMPLES))
            shell_reports.append(
                _pumped(
                    lambda piece=piece, ref_shell=ref_shell, samples=samples, k=k: evaluate(
                        piece,
                        ref_shell.mesh,
                        ref_shell.features,
                        time_s=shell_seconds[k],
                        strong_threshold=threshold,
                        samples=samples,
                        name=f"{merged.name} casca {k + 1}",
                        engine=engine_label,
                    ),
                    progress,
                    cancel,
                    0.95 + 0.05 * k / count,
                    f"Medindo a casca {k + 1} de {count}",
                )
            )
        # uma casca errada não pode se esconder na média do resultado geral
        bad = [
            k + 1
            for k in range(count)
            if status[k] == SHELL_ENGINE
            and any(shell_reports[k].checks.get(key) is False for key in SHELL_REQUIRED_CHECKS)
        ]
        if bad:
            report.passed = False
            report.notes.append(
                f"Reprovado: a(s) casca(s) {_list_shells(bad)} não passaram em quads, cobertura das creases "
                "ou fidelidade das quinas (veja o relatório de cada casca)."
            )
    failed_ids = [k + 1 for k in range(count) if status[k] == SHELL_FAILED]
    if failed_ids:
        report.passed = False
        report.notes.append(
            f"Reprovado: o motor falhou na(s) casca(s) {_list_shells(failed_ids)}, que viraram quads direto."
        )

    _report(progress, 1.0, "Concluído")
    return PipelineResult(
        mesh=merged,
        hard_edges=hard_edges,
        crease_edges=crease_edges,
        report=report,
        shell_reports=shell_reports,
        notes=notes,
        seconds=time.perf_counter() - start,
        engine_seconds=engine_seconds,
        target_quads=total,
        shell_targets=[0 if tiny else int(t) for t, tiny in zip(targets, small)],
        shell_status=status,
        vertex_deviation=deviation,
        heat_scale=Targets().hausdorff_max * ref_clean.mesh.diagonal(),
    )


def heatmap_colors(result: PipelineResult) -> np.ndarray:
    """Cores RGB (0-1) por vértice do resultado: azul = no original, vermelho = na meta de desvio ou acima."""
    deviation = np.asarray(result.vertex_deviation, dtype=np.float64)
    if len(deviation) != result.mesh.num_points:
        raise ValueError("o resultado não tem o desvio por vértice (rode o pipeline de novo)")
    scale = result.heat_scale if result.heat_scale > 0 else float(deviation.max(initial=0.0)) or 1.0
    return heat_colors(deviation, scale)


# --------------------------------------------------------------------------
# Camada Maya (só roda dentro do Maya / mayapy)
# --------------------------------------------------------------------------


@contextmanager
def _maya_undo_suspended() -> Iterator[None]:
    """Não grava desfazer do trabalho temporário (cópia reduzida, malhas do motor).

    Tudo o que é criado aqui dentro também é apagado aqui dentro; a fila de
    desfazer não é esvaziada e volta ao estado anterior no fim. Nada aqui
    dentro pode processar eventos da interface: o usuário mexeria na cena sem
    o desfazer gravar.
    """
    from maya import cmds

    was_on = bool(cmds.undoInfo(query=True, state=True))
    if was_on:
        cmds.undoInfo(stateWithoutFlush=False)
    try:
        yield
    finally:
        if was_on:
            cmds.undoInfo(stateWithoutFlush=True)


def _maya_mirrored(transform: str) -> bool:
    """Verdadeiro se a matriz de mundo do objeto espelha (determinante negativo, ex.: scaleX = -1).

    Lida em espaço de mundo, a malha espelhada fica com as faces no sentido
    errado (o Maya compensa isso na hora de desenhar o original).
    """
    from maya import cmds

    matrix = np.asarray(cmds.xform(transform, query=True, worldSpace=True, matrix=True), dtype=np.float64)
    return bool(np.linalg.det(matrix.reshape(4, 4)[:3, :3]) < 0)


def _maya_resolve(uuid: str | None, fallback: str) -> str | None:
    """Caminho atual do objeto (pelo UUID, que sobrevive a renomear e mudar de pai), ou ``None`` se sumiu."""
    from maya import cmds

    if uuid:
        found = cmds.ls(uuid, long=True) or []
        if found:
            return found[0]
    return fallback if cmds.objExists(fallback) else None


def _maya_reduced_copy(transform: str, short: str, target: int) -> tuple[Mesh, FeatureSet]:
    """Duplica o objeto, reduz a cópia para ``target`` triângulos, lê malha e creases e apaga a cópia.

    Tudo sem gravar desfazer e sem processar eventos; a seleção do usuário
    volta a ser a de antes. Falha do Reduce vira ``PipelineError`` (etapa ``reduzir``).
    """
    from maya import cmds

    from .. import maya_bridge
    from .cleanup import maya_reduce
    from .creases import maya_read_creases

    saved = cmds.ls(selection=True, long=True) or []
    copy_name = None
    with _maya_undo_suspended():
        try:
            copy_name = cmds.duplicate(transform, name=short + "_toptopReduzida", returnRootsOnly=True)[0]
            try:
                maya_reduce(maya_bridge.shape_of(copy_name), target, crease_weight=1.0, border_weight=1.0)
            except Exception as error:
                raise PipelineError(
                    f"O Reduce do Maya falhou ao reduzir uma cópia para {target} triângulos: "
                    f"{_one_line(error)}. Aumente o limite de redução na janela (Limite de redução) "
                    "para o Maya reduzir menos, ou limpe a malha.",
                    stage="reduzir",
                ) from error
            mesh = maya_bridge.mesh_from_maya(copy_name, world=True)
            feats = maya_read_creases(copy_name)
        finally:
            if copy_name is not None and cmds.objExists(copy_name):
                cmds.delete(copy_name)
            try:
                if saved:
                    cmds.select(saved, replace=True)
                else:
                    cmds.select(clear=True)
            except Exception:  # devolver a seleção é cortesia: nunca esconde o erro de verdade
                pass
    return mesh, feats


def _maya_apply_creases(new: str, feats: FeatureSet) -> int:
    """Grava as creases (cada valor de uma vez) nas arestas do objeto novo; devolve quantas arestas."""
    from maya import cmds

    from .. import maya_bridge

    done = 0
    for value in np.unique(feats.crease).tolist():
        pairs = feats.edges[feats.crease == value]
        components = maya_bridge.edge_components(new, pairs)
        if components:
            cmds.polyCrease(components, value=float(value))
            done += len(pairs)
    return done


def maya_retopologize(
    name: str,
    options: PipelineOptions,
    progress: ProgressFn | None = None,
    cancel: CancelFn | None = None,
) -> tuple[str, PipelineResult]:
    """Retopologiza um objeto do Maya e cria ``<nome>_toptop`` ao lado dele.

    Lê a malha (espaço de mundo; objeto espelhado tem as faces desviradas) e
    as creases. Acima de ``reduce_limit`` triângulos, valida o original (a
    malha non-manifold sai com os culpados, antes do Reduce, que não aceita
    non-manifold) e reduz uma cópia, sem gravar desfazer. ``run_pipeline``
    roda com o desfazer ligado, usando o original como referência. Depois, num
    único passo de desfazer, cria o objeto novo, aplica as hard edges (Game
    com ``hard_edges``; senão tudo suave), grava as creases no Film, copia o
    material e o seleciona; se algo falhar aí, o objeto novo é apagado. O
    original nunca é alterado. Devolve (transform novo, resultado).
    """
    from maya import cmds

    from .. import maya_bridge
    from .creases import maya_read_creases

    options.validate()
    transform = maya_bridge.transform_of(name)
    short = transform.split("|")[-1]
    uuid = (cmds.ls(transform, uuid=True) or [None])[0]
    _report(progress, 0.0, "Lendo a malha do Maya")
    mirrored = _maya_mirrored(transform)
    original = maya_bridge.mesh_from_maya(transform, world=True)
    if mirrored:
        original = reverse_winding(original)
    original_feats = maya_read_creases(transform)
    target = reduction_target(original, options.reduce_limit)
    base = 0.0 if target is None else 0.1
    _check_cancel(cancel)

    def pipeline_progress(fraction: float, message: str) -> None:
        _report(progress, base + (1.0 - base) * fraction, message)

    if target is None:
        mesh, feats = original, original_feats
        reference, reference_feats = None, None
    else:
        # o Reduce não aceita malha non-manifold: os culpados saem do original, antes dele
        _pumped(
            lambda: _clean(original, original_feats if len(original_feats) else None, options, "validar", "A malha"),
            progress,
            cancel,
            0.01,
            "Validando a malha",
        )
        _check_cancel(cancel)
        _report(progress, 0.02, f"Reduzindo uma cópia para {target} triângulos")
        mesh, feats = _maya_reduced_copy(transform, short, target)
        if mirrored:
            mesh = reverse_winding(mesh)
        reference = original
        reference_feats = original_feats if len(original_feats) else None
    _check_cancel(cancel)
    result = run_pipeline(
        mesh,
        feats if len(feats) else None,
        options,
        reference=reference,
        reference_features=reference_feats,
        progress=pipeline_progress if progress else None,
        cancel=cancel,
    )
    if mirrored:
        result.notes.append("O objeto está espelhado (escala negativa): as faces foram desviradas na leitura.")

    threshold = float(options.strong_threshold)
    with maya_bridge.undo_chunk("TopTop: retopologia"):
        new = maya_bridge.create_mesh(result.mesh, short + "_toptop")
        try:
            if options.hard_edges and options.mode == "game":
                hard = result.hard_edges
            else:
                hard = np.zeros((0, 2), dtype=np.int64)
            maya_bridge.set_hard_edges(new, hard)
            if options.mode == "film" and len(result.crease_edges):
                written = _maya_apply_creases(new, result.crease_edges)
                strong = int(result.crease_edges.strong_mask(threshold).sum())
                result.notes.append(
                    f"Creases gravadas para o Film: {written} aresta(s) de loop de feature "
                    f"({strong} forte(s) com {_strong_value(threshold):g}, "
                    f"{len(result.crease_edges) - strong} guia(s) com {_guide_value(threshold):g})."
                )
            source = _maya_resolve(uuid, transform)
            if source is None:
                result.notes.append("O original sumiu durante a execução: o material não foi copiado.")
            else:
                maya_bridge.copy_material(source, new)
            cmds.select(new, replace=True)
        except Exception:
            # nada pela metade na cena: o objeto novo sai junto com o erro
            if cmds.objExists(new):
                cmds.delete(new)
            raise
    return new, result


def maya_estimate_inputs(name: str) -> tuple[float, float]:
    """(área, diagonal) de um objeto do Maya em espaço de mundo: tudo o que a estimativa precisa.

    A janela guarda isto por objeto e calcula qualquer nível e modo com
    ``estimate_from``, sem ler a malha de novo. Roda a cada troca de seleção,
    por isso pede área e caixa ao próprio Maya (rápido em qualquer tamanho)
    em vez de copiar a malha para o Python; a estimativa só usa a razão
    área / diagonal², então a unidade da cena não importa.
    """
    from maya import cmds

    from .. import maya_bridge

    shape = maya_bridge.shape_of(name)
    try:
        area = cmds.polyEvaluate(shape, worldArea=True)
        box = cmds.exactWorldBoundingBox(shape)
        if isinstance(area, (int, float)) and box and len(box) == 6:
            low, high = np.asarray(box[:3], dtype=np.float64), np.asarray(box[3:], dtype=np.float64)
            return float(area), float(np.linalg.norm(high - low))
    except RuntimeError:
        pass
    # o polyEvaluate devolve texto quando não consegue contar: lê a malha
    mesh = maya_bridge.mesh_from_maya(name, world=True)
    return surface_area(mesh), mesh.diagonal()


def maya_estimate(name: str, detail: int, mode: str = "game") -> int:
    """``estimate_quads`` direto num objeto do Maya (malha em espaço de mundo)."""
    area, diagonal = maya_estimate_inputs(name)
    return estimate_from(area, diagonal, detail, mode)


def maya_show_heatmap(name: str, result: PipelineResult) -> None:
    """Pinta o objeto criado pela retopologia com o mapa de calor de desvio.

    As cores vão num color set próprio (``toptopDesvio``) e a exibição de cor
    de vértice é ligada; ``maya_hide_heatmap`` desliga. As cores são gravadas
    pela API e não entram no desfazer: para escondê-las, use
    ``maya_hide_heatmap`` (o botão da janela alterna entre os dois).
    """
    from .. import maya_bridge

    maya_bridge.set_vertex_colors(name, heatmap_colors(result))


def maya_hide_heatmap(name: str) -> None:
    """Desliga a exibição das cores de vértice (o color set continua no objeto)."""
    from maya import cmds

    from .. import maya_bridge

    cmds.setAttr(maya_bridge.shape_of(name) + ".displayColors", 0)
