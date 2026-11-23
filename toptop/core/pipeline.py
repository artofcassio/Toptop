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
6. **motor** em cada casca, com o número de quads dividido pela área;
7. **reprojeção** do resultado na referência inteira (``reproject.reproject``);
8. **pós-processo**: hard edges = arestas do resultado sobre creases fortes;
   creases para o Film = todos os loops de feature (3.0 forte, 1.0 guia);
9. **juntar** as cascas e **medir** contra a referência limpa.

As funções ``maya_*`` no fim do arquivo são a camada fina que lê a malha do
Maya, reduz uma cópia quando precisa e cria o objeto novo; elas importam
``maya`` dentro de cada função.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np

from ..engine import ENGINE_NAMES, get_engine
from ..engine.base import Density, Engine, EngineCancelled, EngineError
from .cleanup import (
    DEFAULT_TRIANGLE_LIMIT,
    DEFAULT_WELD_TOL_REL,
    CleanResult,
    Shell,
    _corner_next,
    clean_mesh,
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
from .mesh import Mesh
from .quality import QualityReport, evaluate, match_features
from .reproject import reproject as reproject_mesh

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]

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
    #: total de quads pedido ao motor e a parte de cada casca
    target_quads: int = 0
    shell_targets: list[int] = field(default_factory=list)


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


def estimate_quads(mesh: Mesh, detail: int, mode: str = "game") -> int:
    """Número de quads para um nível de detalhe (1 a 10).

    Lado do quad L = diagonal x 0,1 x 0,75^(detalhe - 1); quads = área / L²;
    o Film pede 2,5 vezes mais. Nunca menos que 24. Rápido: não monta a
    topologia da malha.
    """
    _check_detail(detail)
    _check_mode(mode)
    diagonal = mesh.diagonal()
    area = surface_area(mesh)
    if not diagonal > 0 or not area > 0:
        return MIN_SHELL_QUADS
    edge = diagonal * BASE_EDGE_REL * EDGE_STEP ** (int(detail) - 1)
    quads = area / (edge * edge)
    if mode == "film":
        quads *= FILM_FACTOR
    return max(MIN_SHELL_QUADS, int(round(quads)))


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
        message += " Tente aumentar o limite de redução para o Maya reduzir menos."
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
    entrada é alterada. Cancelar levanta ``EngineCancelled``; falha do motor
    levanta ``EngineError`` com o nome da casca; malha inválida levanta
    ``PipelineError``.
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
    _report(progress, 0.0, "Validando e limpando a malha")
    ref_clean = _clean(ref_input, reference_features, options, "validar", "A malha")
    if same:
        eng_clean = ref_clean
    else:
        _check_cancel(cancel)
        eng_clean = _clean(mesh, features, options, "reduzir", "A versão reduzida da malha")
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

    # 2. cascas (com as features já completas)
    shell_list = split_shells(eng_clean.mesh, eng_feats)
    if not shell_list:
        raise PipelineError("A malha não tem nenhuma casca com faces.", stage="validar")
    if len(shell_list) > 1:
        notes.append(f"{len(shell_list)} cascas separadas: o motor roda uma vez em cada uma.")
    areas = np.array([surface_area(s.mesh) for s in shell_list])
    total = int(options.target_quads) if options.target_quads is not None else estimate_quads(
        ref_clean.mesh, int(options.detail), options.mode
    )
    targets = shell_targets(total, areas)
    if sum(targets) > total * 1.05:
        notes.append(
            f"Com o mínimo de {MIN_SHELL_QUADS} quads por casca, o motor recebeu {sum(targets)} quads "
            f"no total (pedido: {total})."
        )

    # 6. motor em cada casca
    engine = resolve_engine(options)
    engine_label = engine.name or type(engine).__name__
    lo, hi = 0.08, 0.80
    weights = np.asarray(targets, dtype=np.float64)
    bounds = lo + (hi - lo) * np.concatenate([[0.0], np.cumsum(weights) / weights.sum()])
    results: list[Mesh] = []
    shell_seconds: list[float] = []
    count = len(shell_list)
    for k, shell in enumerate(shell_list):
        _check_cancel(cancel)
        label = f"Casca {k + 1} de {count} ({shell.mesh.num_faces} triângulos)"
        a, b = float(bounds[k]), float(bounds[k + 1])

        def sub_progress(fraction: float, message: str, a: float = a, b: float = b, label: str = label) -> None:
            _report(progress, a + (b - a) * min(max(float(fraction), 0.0), 1.0), f"{label}: {message}")

        _report(progress, a, f"{label}: motor {engine_label}, {targets[k]} quads")
        before = getattr(engine, "last_run", None)
        t0 = time.perf_counter()
        out = _run_engine(engine, shell, targets[k], label, cancel, sub_progress if progress else None)
        shell_seconds.append(time.perf_counter() - t0)
        run = getattr(engine, "last_run", None)
        details = getattr(run, "details", None) if run is not before else None
        if isinstance(details, dict) and details.get("warning"):
            notes.append(f"{label}: {details['warning']}")
        results.append(Mesh(out.points, out.counts, out.connects, mesh.name))
    engine_seconds = float(sum(shell_seconds))
    _check_cancel(cancel)

    # 9a. juntar as cascas; 7. reprojetar tudo de uma vez na referência inteira
    # (é o mesmo que reprojetar cada casca, mas a grade da referência é montada uma vez)
    merged, offsets = _merge(results, ref_input.name or mesh.name)
    if options.reproject:
        _report(progress, 0.82, "Reprojetando na superfície original")
        rep = reproject_mesh(merged, ref_clean.mesh, ref_feats, iterations=REPROJECT_ITERATIONS)
        merged = rep.mesh
        notes.extend(rep.messages)
    _check_cancel(cancel)

    # 9b. medir; 8. pós-processo a partir dos loops achados na medição
    _report(progress, 0.90, "Medindo a qualidade")
    measured: dict[str, Any] = {}
    report = evaluate(
        merged,
        ref_clean.mesh,
        ref_feats,
        time_s=engine_seconds,
        strong_threshold=threshold,
        samples=REPORT_SAMPLES,
        name=merged.name,
        engine=engine_label,
        details=measured,
    )
    match = measured.get("match")
    if match is not None and match.applicable and len(match.matched_edges):
        matched = match.matched_edges
        strong = np.asarray(match.matched_strong, dtype=bool)
        hard_edges = FeatureSet(matched[strong]).edges
        crease_edges = FeatureSet(matched, np.where(strong, STRONG_VALUE, GUIDE_VALUE))
    else:
        hard_edges = np.zeros((0, 2), dtype=np.int64)
        crease_edges = FeatureSet.empty()

    _report(progress, 0.95, "Medindo cada casca")
    if count == 1:
        shell_reports = [report]
    else:
        ref_shells = shell_list if same else split_shells(ref_clean.mesh, ref_feats)
        if same:
            ref_index = list(range(count))
        else:
            ref_index = _reference_shell_index(shell_list, ref_shells, ref_clean.mesh)
        share = areas / areas.sum() if areas.sum() > 0 else np.full(count, 1.0 / count)
        shell_reports = []
        for k, out in enumerate(results):
            _check_cancel(cancel)
            piece = Mesh(merged.points[offsets[k] : offsets[k + 1]], out.counts, out.connects, merged.name)
            ref_shell = ref_shells[ref_index[k]]
            samples = int(np.clip(REPORT_SAMPLES * share[k], MIN_SHELL_SAMPLES, REPORT_SAMPLES))
            shell_reports.append(
                evaluate(
                    piece,
                    ref_shell.mesh,
                    ref_shell.features,
                    time_s=shell_seconds[k],
                    strong_threshold=threshold,
                    samples=samples,
                    name=f"{merged.name} casca {k + 1}",
                    engine=engine_label,
                )
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
        shell_targets=list(targets),
    )


# --------------------------------------------------------------------------
# Camada Maya (só roda dentro do Maya / mayapy)
# --------------------------------------------------------------------------


@contextmanager
def _maya_undo_suspended() -> Iterator[None]:
    """Não grava desfazer do trabalho temporário (cópia reduzida, malhas do motor).

    Tudo o que é criado aqui dentro também é apagado aqui dentro; a fila de
    desfazer não é esvaziada e volta ao estado anterior no fim.
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


def maya_retopologize(
    name: str,
    options: PipelineOptions,
    progress: ProgressFn | None = None,
    cancel: CancelFn | None = None,
) -> tuple[str, PipelineResult]:
    """Retopologiza um objeto do Maya e cria ``<nome>_toptop`` ao lado dele.

    Lê a malha (espaço de mundo) e as creases; acima de ``reduce_limit``
    triângulos, duplica o objeto, reduz a cópia, lê e apaga a cópia. Roda
    ``run_pipeline`` com o original como referência e, num único passo de
    desfazer, cria o objeto novo, aplica as hard edges (Game com
    ``hard_edges``; senão tudo suave), copia o material e o seleciona. O
    original nunca é alterado. Devolve (transform novo, resultado).
    """
    from maya import cmds

    from .. import maya_bridge
    from .cleanup import maya_reduce
    from .creases import maya_read_creases

    options.validate()
    transform = maya_bridge.transform_of(name)
    short = transform.split("|")[-1]
    _report(progress, 0.0, "Lendo a malha do Maya")
    original = maya_bridge.mesh_from_maya(transform, world=True)
    original_feats = maya_read_creases(transform)
    target = reduction_target(original, options.reduce_limit)
    base = 0.0 if target is None else 0.1
    _check_cancel(cancel)

    def pipeline_progress(fraction: float, message: str) -> None:
        _report(progress, base + (1.0 - base) * fraction, message)

    with _maya_undo_suspended():
        if target is None:
            mesh, feats = original, original_feats
            reference, reference_feats = None, None
        else:
            _report(progress, 0.02, f"Reduzindo uma cópia para {target} triângulos")
            copy = cmds.duplicate(transform, name=short + "_toptopReduzida", returnRootsOnly=True)[0]
            try:
                maya_reduce(maya_bridge.shape_of(copy), target, crease_weight=1.0, border_weight=1.0)
                mesh = maya_bridge.mesh_from_maya(copy, world=True)
                feats = maya_read_creases(copy)
            finally:
                if cmds.objExists(copy):
                    cmds.delete(copy)
            reference = original
            reference_feats = original_feats if len(original_feats) else None
        result = run_pipeline(
            mesh,
            feats if len(feats) else None,
            options,
            reference=reference,
            reference_features=reference_feats,
            progress=pipeline_progress if progress else None,
            cancel=cancel,
        )

    with maya_bridge.undo_chunk("TopTop: retopologia"):
        new = maya_bridge.create_mesh(result.mesh, short + "_toptop")
        if options.hard_edges and options.mode == "game":
            hard = result.hard_edges
        else:
            hard = np.zeros((0, 2), dtype=np.int64)
        maya_bridge.set_hard_edges(new, hard)
        maya_bridge.copy_material(transform, new)
        cmds.select(new, replace=True)
    return new, result


def maya_estimate(name: str, detail: int, mode: str = "game") -> int:
    """``estimate_quads`` direto num objeto do Maya (malha em espaço de mundo)."""
    from .. import maya_bridge

    return estimate_quads(maya_bridge.mesh_from_maya(name, world=True), detail, mode)
