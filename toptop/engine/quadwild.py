"""Adaptador do QuadWild Bi-MDF (executável externo, GPL3).

Binários prontos: https://github.com/cgg-bern/quadwild-bimdf/releases (o zip do
Windows traz ``quadwild.exe``, ``quad_from_patches.exe`` e a pasta ``config``).
O motor roda como **processo separado**; nada dele é ligado a este código.

Linha de comando (conferida no código-fonte, ``quadwild/quadwild.cpp`` e
``components/quad_from_patches/main.cpp``):

1. ``quadwild <malha.obj> 2 <prep.txt> <malha.sharp>``: remesh, campo e traçado;
   grava ``<malha>_rem_p0.obj`` e companhia ao lado da malha. Os argumentos
   depois do número são reconhecidos pela extensão.
2. ``quad_from_patches <malha>_rem_p0.obj <n> <main.txt> [stats.json]``: grava
   ``<malha>_rem_p0_<n>_quadrangulation.obj`` e ``..._quadrangulation_smooth.obj``.

Formato do ``.sharp``: primeira linha com a quantidade; depois uma linha
``tipo,face,aresta`` por feature, em que ``tipo`` é 0 (côncava) ou 1 (convexa),
``face`` é o índice do triângulo no OBJ (a partir de 0) e ``aresta`` é 0, 1 ou 2
(a aresta que vai do vértice ``aresta`` ao seguinte dentro do triângulo).

Detalhes que este adaptador resolve:

* os dois programas acham os arquivos por pedaços do caminho (``.obj``,
  ``.txt``, ``.sharp``) e leem configs com caminhos relativos; por isso tudo
  roda com a pasta temporária como diretório atual e só nomes simples vão na
  linha de comando (caminhos com acento ou espaço deixam de ser problema);
* quando se passa um ``.sharp``, o motor não marca as bordas abertas sozinho;
  elas são acrescentadas aqui;
* a densidade é o ``scaleFact`` do segundo passo: o lado do quad é a aresta
  média da malha remeshada vezes ``scaleFact``. O valor é calculado para o
  número de quads pedido e corrigido com uma segunda quadrangulação se o
  resultado sair fora da folga.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time

import numpy as np

from ..core.creases import FeatureSet
from ..core.mesh import Mesh, face_normals_areas, triangle_mesh
from ..io.obj import read_obj, write_obj
from .base import (
    CancelFn,
    Density,
    Engine,
    EngineError,
    EngineUnavailable,
    ProgressFn,
    RunInfo,
    ScaleTransform,
    as_feature_pairs,
    read_log_tail,
    run_process,
)

INSTALL_ENV = "TOPTOP_QUADWILD_DIR"
RELEASES_URL = "https://github.com/cgg-bern/quadwild-bimdf/releases"
_EXE = ".exe" if os.name == "nt" else ""
QUADWILD_EXE = "quadwild" + _EXE
PATCHES_EXE = "quad_from_patches" + _EXE

# config/main_config/flow.txt do release v0.0.2. O leitor do motor é posicional:
# a ordem e os nomes das linhas não podem mudar.
FLOW_TEMPLATE = """\
alpha 0.005
ilpMethod 1
timeLimit 200
gapLimit 0.0
callbackTimeLimit 8 3.00 5.000 10.0 20.0 30.0 60.0 90.0 120.0
callbackGapLimit 8 0.005 0.02 0.05 0.10 0.15 0.20 0.25 0.3
minimumGap 0.4
isometry 1
regularityQuadrilaterals 1
regularityNonQuadrilaterals 1
regularityNonQuadrilateralsWeight 0.9
alignSingularities 1
alignSingularitiesWeight 0.1
repeatLosingConstraintsIterations 1
repeatLosingConstraintsQuads 0
repeatLosingConstraintsNonQuads 0
repeatLosingConstraintsAlign 1
hardParityConstraint 1
scaleFact 1
fixedChartClusters 0
useFlowSolver 1
flow_config_filename "config/main_config/flow_virtual_simple.json"
satsuma_config_filename "config/satsuma/default.json"
"""


def _repo_root() -> str:
    """Raiz do repositório (pasta que contém ``toptop``)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_install(install_dir: str | None = None) -> str | None:
    """Pasta com os dois executáveis do QuadWild, ou ``None``.

    Procura, nesta ordem: o argumento, a variável ``TOPTOP_QUADWILD_DIR``,
    ``third_party/quadwild-bimdf`` na raiz do repositório e ``~/quadwild-bimdf``;
    em cada uma, também uma subpasta de primeiro nível (zip extraído "para uma
    pasta").
    """
    roots = [
        install_dir,
        os.environ.get(INSTALL_ENV),
        os.path.join(_repo_root(), "third_party", "quadwild-bimdf"),
        os.path.join(os.path.expanduser("~"), "quadwild-bimdf"),
    ]
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        candidates = [root]
        try:
            candidates += sorted(
                os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
            )
        except OSError:
            pass
        for folder in candidates:
            if os.path.isfile(os.path.join(folder, QUADWILD_EXE)) and os.path.isfile(
                os.path.join(folder, PATCHES_EXE)
            ):
                return os.path.abspath(folder)
    return None


def clean_triangles(mesh: Mesh) -> Mesh:
    """Triangula e tira os triângulos com vértice repetido (o motor recusa o OBJ)."""
    tri = triangle_mesh(mesh)
    t = tri.connects.reshape(-1, 3)
    good = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 2] != t[:, 0])
    if good.all():
        return tri
    t = t[good]
    return Mesh(tri.points, np.full(len(t), 3, dtype=np.int64), t.reshape(-1), tri.name)


def sharp_records(tri_mesh: Mesh, pairs: np.ndarray, include_boundary: bool = True) -> np.ndarray:
    """Linhas do ``.sharp`` como array (K, 3): tipo, face, aresta dentro da face.

    ``tri_mesh`` tem de ser só triângulos, na mesma ordem em que vai para o OBJ.
    Pares que não são aresta da malha são ignorados; com ``include_boundary`` as
    bordas abertas entram como convexas, como o próprio motor faz quando detecta
    as features sozinho.
    """
    if tri_mesh.num_faces and int(tri_mesh.counts.max()) != 3:
        raise ValueError("sharp_records espera uma malha só de triângulos")
    topo = tri_mesh.topology
    ids = topo.edge_ids(pairs)
    ids = ids[ids >= 0]
    if include_boundary:
        ids = np.concatenate([ids, np.flatnonzero(topo.boundary_edge)])
    ids = np.unique(ids)
    if ids.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    corner = topo.edge_corners[ids, 0]
    face = topo.corner_face[corner]
    local = corner - tri_mesh.offsets[face]
    other = topo.edge_faces[ids, 1]
    manifold = topo.edge_face_count[ids] == 2

    normals, _ = face_normals_areas(tri_mesh)
    points = tri_mesh.points
    direction = points[tri_mesh.connects[topo.corner_next[corner]]] - points[tri_mesh.connects[corner]]
    cross = np.cross(normals[face], normals[np.maximum(other, 0)])
    # mesma conta do motor (IsConcaveEdge): côncava se (N0 x N1) . aresta < 0
    concave = manifold & (np.einsum("ij,ij->i", cross, direction) < 0)
    kind = np.where(concave, 0, 1)
    return np.stack([kind, face, local], axis=1).astype(np.int64)


def format_sharp(records: np.ndarray) -> str:
    """Texto do arquivo ``.sharp``."""
    lines = [str(len(records))]
    lines.extend("%d,%d,%d" % (k, f, e) for k, f, e in np.asarray(records).tolist())
    return "\n".join(lines) + "\n"


def prep_config_text(do_remesh: bool = True, sharp_angle: float = 35.0, alpha: float = 0.01) -> str:
    """Config do primeiro passo (``config/prep_config/basic_setup.txt``).

    ``sharp_angle`` só vale quando nenhum ``.sharp`` é passado; ``alpha`` e
    ``scaleFact`` são lidos mas só usados no passo 3 embutido, que não é usado.
    """
    return (
        f"do_remesh {1 if do_remesh else 0}\n"
        f"sharp_feature_thr {sharp_angle:g}\n"
        f"alpha {alpha:g}\n"
        "scaleFact 1\n"
    )


def main_config_text(template: str, scale_fact: float) -> str:
    """Config do segundo passo: o modelo com a linha ``scaleFact`` trocada."""
    lines = template.splitlines()
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith("scaleFact"):
            lines[i] = f"scaleFact {scale_fact:.6g}"
            found = True
    if not found:
        raise EngineError("o modelo de config do QuadWild não tem a linha 'scaleFact'")
    return "\n".join(lines) + "\n"


def scale_fact_for(remeshed: Mesh, target_quads: int) -> float:
    """``scaleFact`` que dá aproximadamente ``target_quads`` quads.

    O motor usa lado do quad = (média das arestas dos triângulos remeshados) x
    ``scaleFact``; um quad desse lado cobre lado ao quadrado de área.
    """
    points = remeshed.points[remeshed.connects]
    nxt = points[remeshed.topology.corner_next]
    lengths = np.linalg.norm(nxt - points, axis=1)
    average = float(lengths.mean()) if lengths.size else 0.0
    area = float(face_normals_areas(remeshed)[1].sum())
    if average <= 0 or area <= 0 or target_quads <= 0:
        return 1.0
    return float(np.clip(math.sqrt(area / target_quads) / average, 0.02, 200.0))


class QuadWildEngine(Engine):
    """QuadWild Bi-MDF como processo filho, com pasta temporária e tempo limite."""

    name = "quadwild"
    label = "QuadWild Bi-MDF"

    def __init__(
        self,
        install_dir: str | None = None,
        timeout_s: float = 600.0,
        keep_workdir: bool = False,
        main_config: str = "flow",
        do_remesh: bool = True,
        sharp_angle: float = 35.0,
        use_smooth: bool = True,
        density_passes: int = 2,
        normalized_diagonal: float = 100.0,
        engine_detects_features: bool = False,
        workdir_root: str | None = None,
    ) -> None:
        """Parâmetros.

        ``main_config``: nome de um arquivo em ``config/main_config`` do motor,
        sem ``.txt`` (``flow`` ou ``flow_noalign`` são os recomendados pelo
        README do release). ``do_remesh``: o motor refaz a triangulação antes;
        melhora robustez e tempo, mas pode simplificar demais (desligue se
        perder detalhe). ``use_smooth``: usa a saída suavizada do motor.
        ``density_passes``: quantas quadrangulações no máximo para acertar a
        densidade. ``engine_detects_features``: ignora as creases dadas e deixa
        o motor detectar por ``sharp_angle`` (só para comparação).
        """
        super().__init__(timeout_s)
        self.install_dir = install_dir
        self.keep_workdir = keep_workdir
        self.main_config = main_config
        self.do_remesh = do_remesh
        self.sharp_angle = sharp_angle
        self.use_smooth = use_smooth
        self.density_passes = max(1, int(density_passes))
        self.normalized_diagonal = normalized_diagonal
        self.engine_detects_features = engine_detects_features
        self.workdir_root = workdir_root

    def availability(self) -> tuple[bool, str]:
        """Procura os executáveis; a mensagem diz onde colocá-los se faltarem."""
        folder = find_install(self.install_dir)
        if folder:
            return True, folder
        return False, (
            f"QuadWild não encontrado. Baixe o zip do seu sistema em {RELEASES_URL}, extraia em "
            f"{os.path.join(_repo_root(), 'third_party', 'quadwild-bimdf')} "
            f"(ou aponte a variável de ambiente {INSTALL_ENV} para a pasta com {QUADWILD_EXE})."
        )

    def _template(self, install: str) -> str:
        """Texto do config principal escolhido (do motor instalado, se existir)."""
        path = os.path.join(install, "config", "main_config", self.main_config + ".txt")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read()
        if self.main_config == "flow":
            return FLOW_TEMPLATE
        raise EngineError(f"config do QuadWild não encontrado: {path}")

    @staticmethod
    def _read_smooth(path: str, raw: Mesh) -> Mesh | None:
        """Saída suavizada, se existir e tiver a mesma topologia da não suavizada."""
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return None
        try:
            smooth = read_obj(path)
        except ValueError:
            return None
        same = (
            smooth.num_points == raw.num_points
            and np.array_equal(smooth.counts, raw.counts)
            and np.array_equal(smooth.connects, raw.connects)
            and bool(np.isfinite(smooth.points).all())
        )
        return smooth if same else None

    def remesh(
        self,
        mesh: Mesh,
        feature_edges: "FeatureSet | np.ndarray | None",
        density: "Density | int",
        *,
        cancel: CancelFn | None = None,
        progress: ProgressFn | None = None,
    ) -> Mesh:
        """Exporta, roda os dois passos do motor e devolve a malha de quads."""
        density = Density.coerce(density)
        pairs = as_feature_pairs(feature_edges)
        install = find_install(self.install_dir)
        if install is None:
            raise EngineUnavailable(self.availability()[1])

        def report(fraction: float, message: str) -> None:
            if progress is not None:
                progress(fraction, message)

        start = time.perf_counter()
        deadline = time.monotonic() + self.timeout_s
        workdir = tempfile.mkdtemp(prefix="toptop_quadwild_", dir=self.workdir_root)
        log_path = os.path.join(workdir, "log.txt")
        info = RunInfo(workdir=workdir)
        self.last_run = info

        def run(exe: str, args: list[str]) -> int:
            remaining = deadline - time.monotonic()
            return run_process([os.path.join(install, exe)] + args, workdir, log_path, max(remaining, 0.0), cancel)

        def need(filename: str, stage: str) -> str:
            path = os.path.join(workdir, filename)
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                raise EngineError(f"o QuadWild não gerou {filename} ({stage})")
            return path

        try:
            report(0.02, "Exportando a malha")
            tri = clean_triangles(mesh)
            if tri.num_faces == 0:
                raise EngineError("a malha de entrada não tem triângulos válidos")
            transform = ScaleTransform.for_mesh(tri, self.normalized_diagonal)
            tri = tri.with_points(transform.apply(tri.points))
            write_obj(os.path.join(workdir, "mesh.obj"), tri)
            args = ["mesh.obj", "2", "prep.txt"]
            if not self.engine_detects_features:
                records = sharp_records(tri, pairs)
                with open(os.path.join(workdir, "mesh.sharp"), "w", encoding="ascii", newline="\n") as handle:
                    handle.write(format_sharp(records))
                args.append("mesh.sharp")
                info.details["sharp_features"] = int(len(records))
            with open(os.path.join(workdir, "prep.txt"), "w", encoding="ascii", newline="\n") as handle:
                handle.write(prep_config_text(self.do_remesh, self.sharp_angle))
            config_src = os.path.join(install, "config")
            if os.path.isdir(config_src):
                shutil.copytree(config_src, os.path.join(workdir, "config"))
            template = self._template(install)

            report(0.05, "QuadWild: remesh, campo e traçado")
            info.details["exit_trace"] = run(QUADWILD_EXE, args)
            for suffix in (".patch", ".corners", ".feature", ".c_feature"):
                need("mesh_rem_p0" + suffix, "passo 1: traçado")
            remeshed = read_obj(need("mesh_rem_p0.obj", "passo 1: traçado"))
            if remeshed.num_faces == 0:
                raise EngineError("o QuadWild devolveu uma malha remeshada vazia (passo 1)")
            info.details["remeshed_triangles"] = remeshed.num_faces

            scale = scale_fact_for(remeshed, density.target_quads)
            result: Mesh | None = None
            for attempt in range(self.density_passes):
                report(0.6 + 0.35 * attempt / self.density_passes, "QuadWild: quadrangulação")
                with open(os.path.join(workdir, "main.txt"), "w", encoding="ascii", newline="\n") as handle:
                    handle.write(main_config_text(template, scale))
                info.details["exit_quads"] = run(
                    PATCHES_EXE, ["mesh_rem_p0.obj", str(attempt), "main.txt", f"stats_{attempt}.json"]
                )
                stem = f"mesh_rem_p0_{attempt}_quadrangulation"
                result = read_obj(need(stem + ".obj", "passo 2: quadrangulação"))
                if result.num_faces == 0:
                    raise EngineError("o QuadWild devolveu uma malha de quads vazia (passo 2)")
                if self.use_smooth:
                    # a versão suavizada tem a mesma topologia; se não bater, o
                    # arquivo ficou incompleto e vale a versão sem suavização
                    smooth = self._read_smooth(os.path.join(workdir, stem + "_smooth.obj"), result)
                    if smooth is None:
                        info.details["smooth_missing"] = True
                    else:
                        result = smooth
                if info.details["exit_quads"] != 0:
                    # visto no binário Linux v0.0.2: falha de segmentação ao sair,
                    # depois de gravar todos os arquivos
                    info.details["warning"] = (
                        f"quad_from_patches saiu com código {info.details['exit_quads']} "
                        "mas gravou a saída completa"
                    )
                info.details["scale_fact"] = scale
                info.details["passes"] = attempt + 1
                if density.accepts(result.num_faces):
                    break
                scale = float(np.clip(scale * math.sqrt(result.num_faces / density.target_quads), 0.02, 200.0))

            assert result is not None
            result = Mesh(transform.invert(result.points), result.counts, result.connects, mesh.name)
        except EngineError as error:
            info.seconds = time.perf_counter() - start
            info.log = read_log_tail(log_path)
            raise type(error)(str(error).split("\nPasta de diagnóstico")[0], workdir, info.log) from None
        except Exception as error:  # leitura de arquivo, disco cheio etc.
            info.seconds = time.perf_counter() - start
            info.log = read_log_tail(log_path)
            raise EngineError(f"falha no adaptador do QuadWild: {error}", workdir, info.log) from error

        info.seconds = time.perf_counter() - start
        info.log = read_log_tail(log_path)
        if self.keep_workdir:
            info.details["workdir_kept"] = True
        else:
            shutil.rmtree(workdir, ignore_errors=True)
            info.workdir = None
        report(1.0, "QuadWild concluído")
        return result
