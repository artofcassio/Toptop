"""Métricas de qualidade da retopologia (seção 11 do plano) em NumPy puro.

Tudo aqui trabalha sobre ``Mesh`` (vértices + faces) e não depende do Maya nem
do motor. O uso típico é::

    report = evaluate(malha_nova, malha_original, features, time_s=segundos)
    print(format_report(report))

Convenções:

* distâncias "relativas" são divididas pela diagonal da caixa envolvente da
  malha **original**;
* um valor ``None`` significa "não se aplica" (por exemplo, cobertura de
  creases num modelo sem creases) e não reprova o modelo.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .creases import (
    DEFAULT_CORNER_ANGLE,
    DEFAULT_STRONG_THRESHOLD,
    FeatureSet,
    boundary_edges,
    feature_corners,
)
from .geom import PrimitiveGrid, point_grid, sample_segments, sample_triangles
from .mesh import Mesh, face_centers, shells, triangulate


@dataclass
class Targets:
    """Metas iniciais da seção 11 (recalibradas na Fase 0)."""

    quad_ratio_min: float = 1.0
    crease_coverage_min: float = 0.99
    feature_fidelity_max: float = 0.001
    hausdorff_max: float = 0.005
    aspect_median_max: float = 1.5
    aspect_p95_max: float = 3.0
    pole_ratio_max: float = 0.05
    poles_on_strong_crease_max: int = 0
    time_max_s: float = 120.0
    time_reference_triangles: int = 100_000
    timeout_s: float = 600.0

    def time_limit(self, ref_triangles: int) -> float:
        """Meta de tempo para uma entrada com ``ref_triangles`` triângulos.

        A meta do plano vale para 100 mil triângulos; entradas maiores ganham
        tempo proporcional, sem passar do tempo limite.
        """
        scale = max(1.0, ref_triangles / float(self.time_reference_triangles))
        return min(self.time_max_s * scale, self.timeout_s)


# --------------------------------------------------------------------------
# Métricas simples
# --------------------------------------------------------------------------


def face_counts(mesh: Mesh) -> dict[str, float]:
    """Quantos triângulos, quads e ngons a malha tem e a fração de quads."""
    counts = mesh.counts
    faces = int(len(counts))
    quads = int((counts == 4).sum())
    return {
        "faces": faces,
        "quads": quads,
        "triangles": int((counts == 3).sum()),
        "ngons": int((counts > 4).sum()),
        "quad_ratio": quads / faces if faces else 0.0,
    }


def quad_aspect_ratios(mesh: Mesh) -> np.ndarray:
    """Lado maior dividido pelo lado menor de cada quad (inf se um lado é zero)."""
    quad = np.flatnonzero(mesh.counts == 4)
    if quad.size == 0:
        return np.zeros(0)
    q = mesh.connects[mesh.offsets[quad][:, None] + np.arange(4)]
    p = mesh.points[q]
    sides = np.linalg.norm(np.roll(p, -1, axis=1) - p, axis=2)
    longest = sides.max(axis=1)
    shortest = sides.min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(shortest > 0, longest / np.where(shortest > 0, shortest, 1.0), np.inf)
    return ratio


def pole_vertices(mesh: Mesh) -> np.ndarray:
    """Vértices internos (fora de borda aberta) com valência diferente de 4."""
    topo = mesh.topology
    interior = topo.used_vertex & ~topo.boundary_vertex
    return np.flatnonzero(interior & (topo.vertex_valence != 4)).astype(np.int64)


# --------------------------------------------------------------------------
# Features: cobertura, fidelidade, polos sobre crease
# --------------------------------------------------------------------------


@dataclass
class FeatureMatch:
    """Correspondência entre as creases originais e os edge loops da malha nova."""

    applicable: bool
    feature_length: float = 0.0
    coverage: float | None = None
    coverage_strong: float | None = None
    vertex_dev: float | None = None
    corner_dev: float | None = None
    chord_dev: float | None = None
    corners_total: int = 0
    corners_matched: int = 0
    matched_edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    matched_strong: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    loop_vertex: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    strong_loop_vertex: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    near_corner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    uncovered_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


def match_features(
    new_mesh: Mesh,
    ref_points: np.ndarray,
    features: FeatureSet,
    diagonal: float,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
    match_tol_rel: float = 0.01,
    align_deg: float = 30.0,
    sample_align_deg: float = 60.0,
    spacing_rel: float = 0.002,
    corner_angle_deg: float = DEFAULT_CORNER_ANGLE,
    max_samples: int = 400_000,
) -> FeatureMatch:
    """Acha, na malha nova, as arestas que seguem as creases da original.

    Uma aresta nova conta como "loop de feature" quando as duas pontas e o meio
    estão a menos de ``match_tol_rel`` (fração da diagonal) da rede de creases e
    ela está alinhada (até ``align_deg``) com a crease mais próxima. A partir
    daí saem:

    * ``coverage``: fração do comprimento de crease coberta por um loop desses,
      contando só o loop que tem essa crease como a mais próxima e corre na
      direção dela, até ``sample_align_deg`` (``coverage_strong`` olha só as
      creases fortes);
    * ``vertex_dev``: maior distância dos vértices desses loops até a crease;
    * ``corner_dev``: maior distância de um canto original até o vértice novo
      mais próximo (pega quina arredondada ou cortada);
    * ``chord_dev``: maior distância do loop até a crease, medida ao longo das
      arestas (inclui a flecha das cordas em creases curvas; só informativo).
    """
    num_new = new_mesh.num_points
    empty_mask = np.zeros(num_new, dtype=bool)
    ref_points = np.asarray(ref_points, dtype=np.float64)
    if len(features) == 0 or diagonal <= 0:
        return FeatureMatch(False, loop_vertex=empty_mask, strong_loop_vertex=empty_mask.copy(), near_corner=empty_mask.copy())

    seg_a = ref_points[features.edges[:, 0]]
    seg_b = ref_points[features.edges[:, 1]]
    seg_len = np.linalg.norm(seg_b - seg_a, axis=1)
    keep = seg_len > 0
    seg_a, seg_b, seg_len = seg_a[keep], seg_b[keep], seg_len[keep]
    feat_edges = features.edges[keep]
    strong = features.strong_mask(strong_threshold)[keep]
    total_len = float(seg_len.sum())
    if total_len <= 0:
        return FeatureMatch(False, loop_vertex=empty_mask, strong_loop_vertex=empty_mask.copy(), near_corner=empty_mask.copy())

    tol = match_tol_rel * diagonal
    ref_grid = PrimitiveGrid(np.stack([seg_a, seg_b], axis=1))
    verts = new_mesh.points
    topo = new_mesh.topology

    # 1. arestas novas que estão em cima da rede de creases
    dist_v, _, _, _ = ref_grid.query(verts, max_dist=tol)
    on_v = dist_v <= tol
    new_edges = topo.edges[~topo.degenerate_edge]
    cand = new_edges[on_v[new_edges[:, 0]] & on_v[new_edges[:, 1]]]
    matched = np.zeros((0, 2), dtype=np.int64)
    matched_strong = np.zeros(0, dtype=bool)
    if len(cand):
        pa, pb = verts[cand[:, 0]], verts[cand[:, 1]]
        dist_m, seg_m, _, _ = ref_grid.query(0.5 * (pa + pb), max_dist=tol)
        near = seg_m >= 0
        seg_safe = np.maximum(seg_m, 0)
        edge_dir = pb - pa
        edge_len = np.linalg.norm(edge_dir, axis=1)
        ref_dir = (seg_b - seg_a)[seg_safe]
        denom = edge_len * seg_len[seg_safe]
        cosine = np.abs(np.einsum("ij,ij->i", edge_dir, ref_dir)) / np.where(denom > 0, denom, 1.0)
        ok = near & (dist_m <= tol) & (edge_len > 0) & (cosine >= math.cos(math.radians(align_deg)))
        matched = cand[ok]
        matched_strong = strong[seg_safe[ok]]

    loop_vertex = np.zeros(num_new, dtype=bool)
    loop_vertex[matched.reshape(-1)] = True
    strong_loop_vertex = np.zeros(num_new, dtype=bool)
    strong_loop_vertex[matched[matched_strong].reshape(-1)] = True

    # 2. cobertura: cada crease é dividida em trechos de tamanho ``spacing``; um
    # trecho está coberto quando alguma amostra de um loop novo tem ESSA crease
    # como a mais próxima (em chapa fina, o loop de cima não cobre a crease de
    # baixo) e o loop corre na direção dela (a borda onde a crease termina
    # cruza, não cobre)
    spacing = max(spacing_rel * diagonal, total_len / max_samples)
    buckets = np.maximum(1, np.ceil(seg_len / spacing)).astype(np.int64)
    bucket_start = np.cumsum(buckets) - buckets
    bucket_seg = np.repeat(np.arange(len(buckets)), buckets)
    bucket_len = (seg_len / buckets)[bucket_seg]
    covered = np.zeros(int(buckets.sum()), dtype=bool)
    chord_dev: float | None = None
    if len(matched):
        new_a, new_b = verts[matched[:, 0]], verts[matched[:, 1]]
        loop_pts, loop_edge, _ = sample_segments(new_a, new_b, 0.5 * spacing)
        dist_l, seg_l, foot, _ = ref_grid.query(loop_pts, max_dist=tol)
        seg_safe = np.maximum(seg_l, 0)
        loop_dir = (new_b - new_a)[loop_edge]
        ref_dir = (seg_b - seg_a)[seg_safe]
        denom = np.linalg.norm(loop_dir, axis=1) * seg_len[seg_safe]
        cosine = np.abs(np.einsum("ij,ij->i", loop_dir, ref_dir)) / np.where(denom > 0, denom, 1.0)
        credit = (seg_l >= 0) & (cosine >= math.cos(math.radians(sample_align_deg)))
        if credit.any():
            seg_c = seg_safe[credit]
            along = np.linalg.norm(foot[credit] - seg_a[seg_c], axis=1) / seg_len[seg_c]
            local = np.minimum((along * buckets[seg_c]).astype(np.int64), buckets[seg_c] - 1)
            covered[bucket_start[seg_c] + local] = True
            chord_dev = float(dist_l[credit].max())
    coverage = float(bucket_len[covered].sum() / total_len)
    bucket_strong = strong[bucket_seg]
    strong_len = float(bucket_len[bucket_strong].sum())
    coverage_strong = (
        float(bucket_len[covered & bucket_strong].sum() / strong_len) if strong_len > 0 else None
    )
    missing = np.flatnonzero(~covered)
    miss_seg = bucket_seg[missing]
    miss_t = ((missing - bucket_start[miss_seg] + 0.5) / buckets[miss_seg])[:, None]
    uncovered_points = seg_a[miss_seg] + (seg_b - seg_a)[miss_seg] * miss_t

    vertex_dev = float(dist_v[loop_vertex].max()) if loop_vertex.any() else None

    # 3. cantos: junções, pontas e viradas bruscas da rede de creases
    corners = feature_corners(ref_points, feat_edges, corner_angle_deg)
    near_corner = np.zeros(num_new, dtype=bool)
    corner_dev: float | None = None
    corners_matched = 0
    used = np.flatnonzero(topo.used_vertex)
    if corners.size and used.size:
        vgrid = point_grid(verts[used])
        dist_c, idx_c, _, _ = vgrid.query(ref_points[corners])
        corner_dev = float(dist_c.max())
        hit = dist_c <= tol
        corners_matched = int(hit.sum())
        near_corner[used[idx_c[hit]]] = True
        cgrid = point_grid(ref_points[corners], cell=tol)
        dist_n, _, _, _ = cgrid.query(verts, max_dist=tol)
        near_corner |= dist_n <= tol

    return FeatureMatch(
        applicable=True,
        feature_length=total_len,
        coverage=coverage,
        coverage_strong=coverage_strong,
        vertex_dev=vertex_dev,
        corner_dev=corner_dev,
        chord_dev=chord_dev,
        corners_total=int(corners.size),
        corners_matched=corners_matched,
        matched_edges=matched,
        matched_strong=matched_strong,
        loop_vertex=loop_vertex,
        strong_loop_vertex=strong_loop_vertex,
        near_corner=near_corner,
        uncovered_points=uncovered_points,
    )


# --------------------------------------------------------------------------
# Desvio de forma (Hausdorff) e mapa de calor
# --------------------------------------------------------------------------


def _triangle_points(mesh: Mesh) -> np.ndarray:
    """Triângulos da malha como array (T, 3, 3), sem os de área zero."""
    tris, _ = triangulate(mesh)
    pts = mesh.points[tris]
    if len(pts) == 0:
        return pts.reshape(0, 3, 3)
    area2 = np.linalg.norm(np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]), axis=1)
    return pts[area2 > 0]


def _capped(points: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """No máximo ``cap`` pontos, escolhidos ao acaso quando há mais."""
    if len(points) <= cap:
        return points
    return points[rng.choice(len(points), size=cap, replace=False)]


def surface_deviation(
    new_mesh: Mesh, ref_mesh: Mesh, samples: int = 50_000, seed: int = 0
) -> dict[str, float | None]:
    """Distância entre as duas superfícies nos dois sentidos (unidades da malha).

    Sentido nova -> original: vértices, centros de face e amostras por área da
    malha nova contra os triângulos da original. Sentido original -> nova: o
    inverso, que acusa partes faltando e quinas arredondadas. ``hausdorff`` é o
    maior dos dois máximos. É uma estimativa por amostragem (limite inferior do
    Hausdorff exato), determinística para a mesma ``seed``.
    """
    rng = np.random.default_rng(seed)
    ref_tri = _triangle_points(ref_mesh)
    new_tri = _triangle_points(new_mesh)
    out: dict[str, float | None] = {
        "new_to_ref_max": None,
        "new_to_ref_mean": None,
        "ref_to_new_max": None,
        "ref_to_new_mean": None,
        "hausdorff": None,
    }
    if len(ref_tri) == 0 or len(new_tri) == 0:
        return out

    new_pts = np.concatenate(
        [
            _capped(new_mesh.points[new_mesh.topology.used_vertex], samples, rng),
            _capped(face_centers(new_mesh), samples, rng),
            sample_triangles(new_tri, samples, rng)[0],
        ]
    )
    dist_new, _, _, _ = PrimitiveGrid(ref_tri).query(new_pts)
    ref_pts = np.concatenate(
        [
            _capped(ref_mesh.points[ref_mesh.topology.used_vertex], samples, rng),
            sample_triangles(ref_tri, samples, rng)[0],
        ]
    )
    dist_ref, _, _, _ = PrimitiveGrid(new_tri).query(ref_pts)
    out["new_to_ref_max"] = float(dist_new.max())
    out["new_to_ref_mean"] = float(dist_new.mean())
    out["ref_to_new_max"] = float(dist_ref.max())
    out["ref_to_new_mean"] = float(dist_ref.mean())
    out["hausdorff"] = max(out["new_to_ref_max"], out["ref_to_new_max"])
    return out


def vertex_deviation(new_mesh: Mesh, ref_mesh: Mesh) -> np.ndarray:
    """Distância de cada vértice da malha nova até a superfície original."""
    ref_tri = _triangle_points(ref_mesh)
    if len(ref_tri) == 0 or new_mesh.num_points == 0:
        return np.zeros(new_mesh.num_points)
    dist, _, _, _ = PrimitiveGrid(ref_tri).query(new_mesh.points)
    return dist


def heat_colors(values: np.ndarray, vmax: float) -> np.ndarray:
    """Cores RGB (0-1) do mapa de calor: azul (0), verde, amarelo, vermelho (>= vmax)."""
    stops = np.array([[0.0, 0.2, 1.0], [0.0, 0.9, 0.3], [1.0, 0.9, 0.0], [1.0, 0.0, 0.0]])
    t = np.clip(np.asarray(values, dtype=np.float64) / vmax if vmax > 0 else np.zeros(len(values)), 0.0, 1.0)
    pos = t * (len(stops) - 1)
    low = np.minimum(pos.astype(np.int64), len(stops) - 2)
    frac = (pos - low)[:, None]
    return stops[low] * (1.0 - frac) + stops[low + 1] * frac


# --------------------------------------------------------------------------
# Relatório
# --------------------------------------------------------------------------

CHECK_LABELS = {
    "quads": "Quads",
    "crease_coverage": "Cobertura das creases",
    "feature_fidelity": "Fidelidade das quinas",
    "shape_deviation": "Desvio da forma",
    "aspect_ratio": "Proporção dos quads",
    "poles": "Polos",
    "time": "Tempo",
}


@dataclass
class QualityReport:
    """Todas as métricas de uma execução, prontas para CSV/JSON."""

    name: str = ""
    engine: str = ""
    vertices: int = 0
    faces: int = 0
    quads: int = 0
    triangles: int = 0
    ngons: int = 0
    quad_ratio: float = 0.0
    shells: int = 0
    boundary_edges: int = 0
    nonmanifold_edges: int = 0
    feature_length: float | None = None
    crease_coverage: float | None = None
    crease_coverage_strong: float | None = None
    feature_fidelity: float | None = None
    feature_vertex_dev: float | None = None
    feature_corner_dev: float | None = None
    feature_chord_dev: float | None = None
    corners_total: int = 0
    corners_matched: int = 0
    hausdorff: float | None = None
    dev_new_to_ref_max: float | None = None
    dev_new_to_ref_mean: float | None = None
    dev_ref_to_new_max: float | None = None
    dev_ref_to_new_mean: float | None = None
    aspect_median: float | None = None
    aspect_p95: float | None = None
    aspect_max: float | None = None
    poles: int = 0
    pole_ratio: float = 0.0
    poles_on_strong_crease: int | None = None
    time_s: float | None = None
    ref_triangles: int = 0
    diagonal: float = 0.0
    checks: dict[str, bool | None] = field(default_factory=dict)
    passed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Dicionário só com tipos do JSON (inf e NaN viram ``None``)."""
        return _json_safe(asdict(self))


def _json_safe(value: Any) -> Any:
    """Converte tipos do NumPy e números não finitos para algo que o JSON aceita."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def check_targets(report: QualityReport, targets: Targets | None = None) -> dict[str, bool | None]:
    """Compara o relatório com as metas; ``None`` = métrica não se aplica."""
    t = targets or Targets()

    def at_most(value: float | None, limit: float) -> bool | None:
        return None if value is None else bool(value <= limit)

    checks: dict[str, bool | None] = {}
    checks["quads"] = bool(report.faces > 0 and report.quad_ratio >= t.quad_ratio_min)
    checks["crease_coverage"] = (
        None if report.crease_coverage is None else bool(report.crease_coverage >= t.crease_coverage_min)
    )
    checks["feature_fidelity"] = at_most(report.feature_fidelity, t.feature_fidelity_max)
    checks["shape_deviation"] = at_most(report.hausdorff, t.hausdorff_max)
    if report.aspect_median is None or report.aspect_p95 is None:
        checks["aspect_ratio"] = None if report.faces else False
    else:
        checks["aspect_ratio"] = bool(
            report.aspect_median <= t.aspect_median_max and report.aspect_p95 <= t.aspect_p95_max
        )
    poles_ok = report.pole_ratio <= t.pole_ratio_max
    if report.poles_on_strong_crease is not None:
        poles_ok = poles_ok and report.poles_on_strong_crease <= t.poles_on_strong_crease_max
    checks["poles"] = bool(poles_ok)
    checks["time"] = at_most(report.time_s, t.time_limit(report.ref_triangles))
    return checks


def evaluate(
    new_mesh: Mesh,
    ref_mesh: Mesh,
    features: FeatureSet | None = None,
    *,
    time_s: float | None = None,
    targets: Targets | None = None,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
    include_boundary: bool = True,
    samples: int = 50_000,
    seed: int = 0,
    match_tol_rel: float = 0.01,
    name: str = "",
    engine: str = "",
) -> QualityReport:
    """Mede a malha nova contra a original e devolve o relatório completo.

    ``features`` são as creases da malha **original** (pares de vértices dela).
    Com ``include_boundary``, as bordas abertas da original entram como feature
    de nível guia, como no pipeline.
    """
    targets = targets or Targets()
    report = QualityReport(name=name or new_mesh.name, engine=engine, time_s=time_s)
    diagonal = ref_mesh.diagonal()
    report.diagonal = diagonal
    report.ref_triangles = int((ref_mesh.counts - 2).sum())

    counts = face_counts(new_mesh)
    report.faces = int(counts["faces"])
    report.quads = int(counts["quads"])
    report.triangles = int(counts["triangles"])
    report.ngons = int(counts["ngons"])
    report.quad_ratio = float(counts["quad_ratio"])
    if report.faces == 0:
        report.notes.append("A malha nova não tem faces.")
        report.checks = check_targets(report, targets)
        report.checks.update({"shape_deviation": False, "aspect_ratio": False, "poles": False})
        report.passed = False
        return report

    topo = new_mesh.topology
    report.vertices = int(topo.used_vertex.sum())
    report.shells = shells(new_mesh)[1]
    report.boundary_edges = int(topo.boundary_edge.sum())
    report.nonmanifold_edges = int(topo.nonmanifold_edge.sum())
    if report.nonmanifold_edges:
        report.notes.append(f"{report.nonmanifold_edges} aresta(s) non-manifold na malha nova.")

    aspect = quad_aspect_ratios(new_mesh)
    if aspect.size:
        report.aspect_median = float(np.median(aspect))
        report.aspect_p95 = float(np.percentile(aspect, 95))
        report.aspect_max = float(aspect.max())

    poles = pole_vertices(new_mesh)
    report.poles = int(poles.size)
    report.pole_ratio = report.poles / report.vertices if report.vertices else 0.0

    feats = features if features is not None else FeatureSet.empty()
    if include_boundary:
        border = boundary_edges(ref_mesh)
        if len(border):
            # borda aberta é feature de nível guia (sem hard edge)
            feats = feats.merged(FeatureSet(border, np.full(len(border), strong_threshold * 0.5)))
    if diagonal > 0:
        match = match_features(
            new_mesh, ref_mesh.points, feats, diagonal, strong_threshold, match_tol_rel=match_tol_rel
        )
        if match.applicable:
            report.feature_length = match.feature_length
            report.crease_coverage = match.coverage
            report.crease_coverage_strong = match.coverage_strong
            report.corners_total = match.corners_total
            report.corners_matched = match.corners_matched
            parts = [d for d in (match.vertex_dev, match.corner_dev) if d is not None]
            report.feature_vertex_dev = None if match.vertex_dev is None else match.vertex_dev / diagonal
            report.feature_corner_dev = None if match.corner_dev is None else match.corner_dev / diagonal
            report.feature_chord_dev = None if match.chord_dev is None else match.chord_dev / diagonal
            if match.vertex_dev is None:
                # nenhum loop encontrado: a distância é, no mínimo, a tolerância de busca
                parts.append(match_tol_rel * diagonal)
                report.notes.append(
                    "Nenhum edge loop foi encontrado sobre as creases; a fidelidade é um limite inferior."
                )
            report.feature_fidelity = max(parts) / diagonal
            on_strong = match.strong_loop_vertex[poles] & ~match.near_corner[poles]
            report.poles_on_strong_crease = int(on_strong.sum())

        dev = surface_deviation(new_mesh, ref_mesh, samples=samples, seed=seed)
        if dev["hausdorff"] is not None:
            report.hausdorff = dev["hausdorff"] / diagonal
            report.dev_new_to_ref_max = dev["new_to_ref_max"] / diagonal
            report.dev_new_to_ref_mean = dev["new_to_ref_mean"] / diagonal
            report.dev_ref_to_new_max = dev["ref_to_new_max"] / diagonal
            report.dev_ref_to_new_mean = dev["ref_to_new_mean"] / diagonal
    else:
        report.notes.append("A malha original tem diagonal zero; distâncias não foram medidas.")

    report.checks = check_targets(report, targets)
    report.passed = all(v is not False for v in report.checks.values())
    return report


def _pct(value: float | None, digits: int = 3) -> str:
    """Formata uma fração como porcentagem, ou 'n/a'."""
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return "inf"
    return f"{value * 100:.{digits}f}%"


def _num(value: float | None, digits: int = 2) -> str:
    """Formata um número, ou 'n/a'."""
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return "inf"
    return f"{value:.{digits}f}"


def format_report(report: QualityReport, targets: Targets | None = None) -> str:
    """Relatório de texto, uma linha por métrica, com a meta ao lado."""
    t = targets or Targets()
    checks = report.checks or check_targets(report, t)

    def mark(key: str) -> str:
        value = checks.get(key)
        return "n/a  " if value is None else ("OK   " if value else "FALHA")

    title = " / ".join(x for x in (report.name, report.engine) if x) or "TopTop"
    lines = [
        f"== {title} ==",
        f"{report.faces} faces ({report.quads} quads, {report.triangles} tris, {report.ngons} ngons), "
        f"{report.vertices} vértices, {report.shells} casca(s)",
        f"[{mark('quads')}] Quads: {_pct(report.quad_ratio, 1)}  (meta {_pct(t.quad_ratio_min, 0)})",
        f"[{mark('crease_coverage')}] Cobertura das creases: {_pct(report.crease_coverage, 1)}"
        f"  (meta >= {_pct(t.crease_coverage_min, 0)}; fortes: {_pct(report.crease_coverage_strong, 1)})",
        f"[{mark('feature_fidelity')}] Fidelidade das quinas: {_pct(report.feature_fidelity)}"
        f"  (meta <= {_pct(t.feature_fidelity_max, 1)}; vértices {_pct(report.feature_vertex_dev)},"
        f" cantos {_pct(report.feature_corner_dev)} [{report.corners_matched}/{report.corners_total}],"
        f" corda {_pct(report.feature_chord_dev)})",
        f"[{mark('shape_deviation')}] Desvio da forma: {_pct(report.hausdorff)}"
        f"  (meta <= {_pct(t.hausdorff_max, 1)}; nova->orig {_pct(report.dev_new_to_ref_max)},"
        f" orig->nova {_pct(report.dev_ref_to_new_max)})",
        f"[{mark('aspect_ratio')}] Proporção dos quads: mediana {_num(report.aspect_median)},"
        f" 95% {_num(report.aspect_p95)}, máx {_num(report.aspect_max)}"
        f"  (meta mediana <= {t.aspect_median_max:g}, 95% <= {t.aspect_p95_max:g})",
        f"[{mark('poles')}] Polos: {report.poles} ({_pct(report.pole_ratio, 1)} dos vértices;"
        f" meta <= {_pct(t.pole_ratio_max, 0)}); sobre crease forte fora dos cantos:"
        f" {'n/a' if report.poles_on_strong_crease is None else report.poles_on_strong_crease}",
        f"[{mark('time')}] Tempo: {'n/a' if report.time_s is None else f'{report.time_s:.1f} s'}"
        f"  (meta <= {t.time_limit(report.ref_triangles):.0f} s para {report.ref_triangles} triângulos)",
        f"Resultado: {'APROVADO' if report.passed else 'REPROVADO'}",
    ]
    lines.extend(f"Nota: {n}" for n in report.notes)
    return "\n".join(lines)


# Métricas comparadas com a referência salva: (sentido, folga absoluta)
_BASELINE_METRICS: dict[str, tuple[str, float]] = {
    "quad_ratio": ("higher", 0.0),
    "crease_coverage": ("higher", 0.0),
    "feature_fidelity": ("lower", 1e-5),
    "hausdorff": ("lower", 1e-5),
    "aspect_median": ("lower", 1e-3),
    "aspect_p95": ("lower", 1e-3),
    "pole_ratio": ("lower", 1e-4),
    "poles_on_strong_crease": ("lower", 0.0),
    "time_s": ("lower", 1.0),
}


def compare_to_baseline(
    current: dict[str, Any], baseline: dict[str, Any], tolerance: float = 0.10
) -> list[str]:
    """Lista as métricas que pioraram mais que ``tolerance`` em relação à referência.

    Lista vazia = mudança aprovada (regra de autoverificação da seção 10).
    """
    regressions: list[str] = []
    for key, (direction, slack) in _BASELINE_METRICS.items():
        base, cur = baseline.get(key), current.get(key)
        if base is None:
            continue
        if cur is None:
            regressions.append(f"{key}: a referência tinha {base} e agora não foi medido")
            continue
        if direction == "lower":
            worse = cur > base * (1.0 + tolerance) + slack
        else:
            worse = cur < base * (1.0 - tolerance) - slack
        if worse:
            regressions.append(f"{key}: {base:.6g} -> {cur:.6g}")
    return regressions
