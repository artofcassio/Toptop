"""Reprojeção: leva os vértices da malha nova para a superfície original (passo 7).

Depois do motor, os vértices novos ficam perto, mas não em cima, da superfície
original. Aqui cada vértice vai para o lugar certo:

* vértice de edge loop que passa num **canto** da rede de features (junção,
  ponta ou virada brusca) vai exatamente para o canto (tipo 2);
* os outros vértices de **loop de feature** vão para o ponto mais próximo da
  curva de crease que eles seguem (tipo 1);
* o resto vai para o ponto mais próximo da **superfície** original (tipo 0).

Nenhum vértice anda mais que ``max_move_rel`` da diagonal: quem precisaria
andar mais fica onde está (tipo -1), assim como os vértices sem face. A malha
de entrada nunca é alterada; o resultado é uma ``Mesh`` nova com as mesmas
faces. Tudo em NumPy puro, sem Maya.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .creases import (
    DEFAULT_CORNER_ANGLE,
    DEFAULT_STRONG_THRESHOLD,
    FeatureSet,
    boundary_edges,
    feature_corners,
)
from .geom import PrimitiveGrid, closest_on_segments, point_grid
from .mesh import Mesh
from .quality import _triangle_points, match_features

#: vértice sem face ou cujo destino passava do deslocamento máximo
KIND_UNTOUCHED = -1
#: projetado na superfície original
KIND_SURFACE = 0
#: projetado numa curva de feature (crease ou borda aberta)
KIND_CURVE = 1
#: encaixado exatamente num canto da rede de features
KIND_CORNER = 2

#: tolerância de alinhamento entre o loop e o segmento de crease escolhido
ALIGN_DEG = 30.0

_OFFSETS = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.int64
)
_PAIR_CHUNK = 500_000
_MAX_PIECES = 500_000
_MAX_CELLS_PER_AXIS = 100_000.0


@dataclass(eq=False)
class ReprojectResult:
    """Malha reprojetada e o que aconteceu com cada vértice.

    ``kind`` usa ``KIND_UNTOUCHED`` (-1), ``KIND_SURFACE`` (0), ``KIND_CURVE``
    (1) e ``KIND_CORNER`` (2); ``moved`` é a distância que cada vértice andou.
    """

    mesh: Mesh
    kind: np.ndarray
    moved: np.ndarray
    max_move: float
    corners_snapped: int = 0
    corners_missing: int = 0
    messages: list[str] = field(default_factory=list)


def feature_network(
    ref_mesh: Mesh, features: FeatureSet | None = None, include_boundary: bool = True
) -> tuple[FeatureSet, int]:
    """Rede de features da reprojeção: creases válidas mais as bordas abertas.

    Bordas entram como crease guia (como em ``quality.evaluate``); arestas de
    comprimento zero saem. Devolve a rede e quantas arestas foram descartadas
    por apontar para vértices que não existem na malha original.
    """
    feats = features if features is not None else FeatureSet.empty()
    n = ref_mesh.num_points
    dropped = 0
    if len(feats):
        valid = ((feats.edges >= 0) & (feats.edges < n)).all(axis=1)
        dropped = int((~valid).sum())
        if dropped:
            feats = FeatureSet(feats.edges[valid], feats.crease[valid])
    if include_boundary:
        border = boundary_edges(ref_mesh)
        if len(border):
            guide = np.full(len(border), DEFAULT_STRONG_THRESHOLD * 0.5)
            feats = feats.merged(FeatureSet(border, guide))
    if len(feats):
        pts = ref_mesh.points
        length = np.linalg.norm(pts[feats.edges[:, 1]] - pts[feats.edges[:, 0]], axis=1)
        keep = length > 0
        if not keep.all():
            feats = FeatureSet(feats.edges[keep], feats.crease[keep])
    return feats, dropped


def _snap_corners(
    corner_pts: np.ndarray, cand_pts: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Casa cantos com vértices candidatos, dos pares mais curtos para os mais longos.

    Cada canto pega no máximo um vértice e cada vértice no máximo um canto; num
    conflito ganha o par mais curto e o perdedor tenta o próximo candidato.
    Devolve (índice do canto, índice do candidato) de cada par formado.
    """
    empty = np.zeros(0, dtype=np.int64)
    if len(corner_pts) == 0 or len(cand_pts) == 0 or radius <= 0:
        return empty, empty.copy()
    corner_free = np.ones(len(corner_pts), dtype=bool)
    cand_free = np.ones(len(cand_pts), dtype=bool)
    out_c: list[np.ndarray] = []
    out_v: list[np.ndarray] = []
    while True:
        fc = np.flatnonzero(corner_free)
        fv = np.flatnonzero(cand_free)
        if fc.size == 0 or fv.size == 0:
            break
        dist, idx, _, _ = point_grid(cand_pts[fv]).query(corner_pts[fc], max_dist=radius)
        ok = idx >= 0
        # sem candidato agora, sem candidato depois (a lista só diminui)
        corner_free[fc[~ok]] = False
        if not ok.any():
            break
        c, v, d = fc[ok], fv[idx[ok]], dist[ok]
        order = np.lexsort((d, v))
        _, first = np.unique(v[order], return_index=True)
        win = order[first]
        out_c.append(c[win])
        out_v.append(v[win])
        corner_free[c[win]] = False
        cand_free[v[win]] = False
    if not out_c:
        return empty, empty.copy()
    return np.concatenate(out_c), np.concatenate(out_v)


def _direction_table(
    points: np.ndarray, matched: np.ndarray, row_of: np.ndarray, num_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """Direções unitárias das arestas de loop de cada linha, em formato CSR.

    ``row_of[vértice]`` diz a linha de cada vértice (-1 = fora da tabela).
    Devolve ``(start, dirs)``: as direções da linha i são ``dirs[start[i]:start[i+1]]``.
    """
    matched = np.asarray(matched, dtype=np.int64).reshape(-1, 2)
    vec = points[matched[:, 1]] - points[matched[:, 0]]
    length = np.linalg.norm(vec, axis=1)
    ok = length > 0
    unit = vec[ok] / length[ok, None]
    ends = np.concatenate([matched[ok, 0], matched[ok, 1]])
    dirs = np.concatenate([unit, unit])
    rows = row_of[ends]
    keep = rows >= 0
    rows, dirs = rows[keep], dirs[keep]
    order = np.argsort(rows, kind="stable")
    count = np.bincount(rows, minlength=num_rows)
    start = np.concatenate([[0], np.cumsum(count)]).astype(np.int64)
    return start, dirs[order]


def _max_alignment(
    seg_dir: np.ndarray, owner: np.ndarray, start: np.ndarray, dirs: np.ndarray
) -> np.ndarray:
    """Maior |cosseno| entre cada direção de segmento e as direções da linha dona."""
    owner = np.asarray(owner, dtype=np.int64)
    out = np.zeros(len(owner))
    k = start[owner + 1] - start[owner]
    total = int(k.sum())
    if total == 0:
        return out
    rep = np.repeat(np.arange(len(owner), dtype=np.int64), k)
    first = np.cumsum(k) - k
    idx = np.arange(total, dtype=np.int64) - np.repeat(first, k) + np.repeat(start[owner], k)
    cosine = np.abs(np.einsum("ij,ij->i", seg_dir[rep], dirs[idx]))
    has = k > 0
    out[has] = np.maximum.reduceat(cosine, first[has])
    return out


def _closest_aligned(
    query: np.ndarray,
    owner: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    seg_dir: np.ndarray,
    radius: float,
    start: np.ndarray,
    dirs: np.ndarray,
    cos_min: float,
) -> np.ndarray:
    """Segmento alinhado mais próximo de cada ponto, a no máximo ``radius``.

    Alinhado = faz no máximo acos(``cos_min``) com alguma direção da linha
    ``owner[i]`` da tabela ``(start, dirs)``. Busca por grade uniforme de
    células com todos os pares ponto-segmento dentro do raio. Devolve o índice
    do segmento, ou -1 quando nenhum segmento alinhado está perto.
    """
    m = len(query)
    best = np.full(m, -1, dtype=np.int64)
    if m == 0 or len(seg_a) == 0 or not radius > 0:
        return best
    seg_len = np.linalg.norm(seg_b - seg_a, axis=1)
    piece = max(radius, float(seg_len.sum()) / _MAX_PIECES)
    n = np.maximum(1, np.ceil(seg_len / piece)).astype(np.int64)
    parent = np.repeat(np.arange(len(seg_a), dtype=np.int64), n)
    local = np.arange(int(n.sum()), dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
    t0 = (local / n[parent])[:, None]
    t1 = ((local + 1) / n[parent])[:, None]
    span = (seg_b - seg_a)[parent]
    pa = seg_a[parent] + span * t0
    pb = seg_a[parent] + span * t1
    mid = 0.5 * (pa + pb)

    # com célula >= raio + meio pedaço, todo pedaço a menos de ``radius`` do
    # ponto tem o meio numa das 27 células em volta da célula do ponto
    lo, hi = mid.min(axis=0), mid.max(axis=0)
    cell = max(1.001 * (radius + 0.5 * piece), float((hi - lo).max()) / _MAX_CELLS_PER_AXIS)
    origin = lo - cell
    ic = np.floor((mid - origin) / cell).astype(np.int64)
    dims = ic.max(axis=0) + 2
    keys = (ic[:, 0] * dims[1] + ic[:, 1]) * dims[2] + ic[:, 2]
    order = np.argsort(keys, kind="stable")
    ukeys, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
    last = len(ukeys) - 1

    qc = np.floor((query - origin) / cell).astype(np.int64)
    c = qc[:, None, :] + _OFFSETS[None, :, :]
    valid = ((c >= 0) & (c < dims)).all(axis=2)
    key = (c[..., 0] * dims[1] + c[..., 1]) * dims[2] + c[..., 2]
    pos = np.minimum(np.searchsorted(ukeys, key), last)
    found = valid & (ukeys[pos] == key)
    cnt = np.where(found, counts[pos], 0)
    st = starts[pos]
    per_pt = cnt.sum(axis=1)
    cum = np.cumsum(per_pt)
    r2 = radius * radius
    s = 0
    while s < m:
        base = int(cum[s - 1]) if s else 0
        e = max(int(np.searchsorted(cum, base + _PAIR_CHUNK, side="right")), s + 1)
        npairs = int(cum[e - 1]) - base
        if npairs:
            nn = cnt[s:e].reshape(-1)
            ss = st[s:e].reshape(-1)
            nz = nn > 0
            nn, ss = nn[nz], ss[nz]
            flat = np.arange(npairs, dtype=np.int64) - np.repeat(np.cumsum(nn) - nn, nn) + np.repeat(ss, nn)
            pid = order[flat]
            pt = np.repeat(np.arange(s, e, dtype=np.int64), per_pt[s:e])
            d2, _ = closest_on_segments(query[pt], pa[pid], pb[pid])
            near = d2 <= r2
            pt, seg, d2 = pt[near], parent[pid[near]], d2[near]
            if pt.size:
                aligned = _max_alignment(seg_dir[seg], owner[pt], start, dirs) >= cos_min
                pt, seg, d2 = pt[aligned], seg[aligned], d2[aligned]
            if pt.size:
                o = np.lexsort((d2, pt))
                _, first = np.unique(pt[o], return_index=True)
                best[pt[o[first]]] = seg[o[first]]
        s = e
    return best


def _curve_targets(
    points: np.ndarray,
    loop_idx: np.ndarray,
    matched: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    seg_grid: PrimitiveGrid,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Segmento de crease e ponto mais próximo para cada vértice de loop.

    O segmento precisa estar alinhado (até ``ALIGN_DEG``) com alguma aresta de
    loop do vértice, para ele não pular para outra crease que passa perto (numa
    junção, por exemplo); sem nenhum alinhado a menos de ``tol``, vale o mais
    próximo. Devolve ``(segmento, ponto)``, com segmento -1 quando não há
    nenhum segmento a menos de ``tol``.
    """
    m = len(loop_idx)
    foot = np.full((m, 3), np.nan)
    if m == 0:
        return np.zeros(0, dtype=np.int64), foot
    query = points[loop_idx]
    _, seg, _, _ = seg_grid.query(query, max_dist=tol)
    seg = seg.astype(np.int64)
    row_of = np.full(len(points), -1, dtype=np.int64)
    row_of[loop_idx] = np.arange(m, dtype=np.int64)
    start, dirs = _direction_table(points, matched, row_of, m)
    seg_vec = seg_b - seg_a
    seg_dir = seg_vec / np.linalg.norm(seg_vec, axis=1)[:, None]
    cos_min = math.cos(math.radians(ALIGN_DEG))

    has = np.flatnonzero(seg >= 0)
    misaligned = has[_max_alignment(seg_dir[seg[has]], has, start, dirs) < cos_min]
    if misaligned.size:
        better = _closest_aligned(
            query[misaligned], misaligned, seg_a, seg_b, seg_dir, tol, start, dirs, cos_min
        )
        ok = better >= 0
        seg[misaligned[ok]] = better[ok]
    has = seg >= 0
    if has.any():
        foot[has] = closest_on_segments(query[has], seg_a[seg[has]], seg_b[seg[has]])[1]
    return seg, foot


def reproject(
    new_mesh: Mesh,
    ref_mesh: Mesh,
    features: FeatureSet | None = None,
    *,
    include_boundary: bool = True,
    match_tol_rel: float = 0.01,
    corner_angle_deg: float = DEFAULT_CORNER_ANGLE,
    max_move_rel: float = 0.05,
    iterations: int = 1,
) -> ReprojectResult:
    """Leva os vértices de ``new_mesh`` para a superfície, as creases e os cantos de ``ref_mesh``.

    ``features`` são as creases da malha **original** (pares de vértices dela);
    com ``include_boundary`` as bordas abertas da original também contam. Um
    vértice novo é "de loop" quando ``quality.match_features`` acha uma aresta
    dele correndo sobre a rede (tolerância ``match_tol_rel`` da diagonal). Cada
    canto (``creases.feature_corners``) puxa o vértice de loop mais próximo a
    menos dessa tolerância; os outros vértices de loop vão para a crease que
    seguem e o resto vai para a superfície. Com ``iterations`` > 1 o processo
    se repete, recalculando quem é de loop. Nenhum vértice termina a mais de
    ``max_move_rel`` da diagonal da posição de entrada.
    """
    num = new_mesh.num_points
    original = new_mesh.points
    points = original.copy()
    kind = np.full(num, KIND_UNTOUCHED, dtype=np.int8)
    diagonal = ref_mesh.diagonal()
    max_move = float(max_move_rel) * diagonal
    tol = float(match_tol_rel) * diagonal
    messages: list[str] = []

    def result(snapped: int = 0, missing: int = 0) -> ReprojectResult:
        """Monta o resultado com as posições atuais."""
        moved = np.linalg.norm(points - original, axis=1)
        return ReprojectResult(
            new_mesh.with_points(points), kind, moved, max_move, snapped, missing, messages
        )

    ref_tri = _triangle_points(ref_mesh) if ref_mesh.num_faces else np.zeros((0, 3, 3))
    if not diagonal > 0 or len(ref_tri) == 0:
        messages.append(
            "A malha original não tem superfície (sem faces ou diagonal zero); nada foi reprojetado."
        )
        return result()
    if new_mesh.num_faces == 0:
        messages.append("A malha nova não tem faces; nada foi reprojetado.")
        return result()

    used = new_mesh.topology.used_vertex
    surface = PrimitiveGrid(ref_tri)
    feats, dropped = feature_network(ref_mesh, features, include_boundary)
    if dropped:
        messages.append(
            f"{dropped} aresta(s) de feature apontam para vértices que não existem na malha "
            "original e foram ignoradas."
        )
    seg_a = ref_mesh.points[feats.edges[:, 0]]
    seg_b = ref_mesh.points[feats.edges[:, 1]]
    seg_grid = PrimitiveGrid(np.stack([seg_a, seg_b], axis=1)) if len(feats) else None
    if len(feats):
        corners = feature_corners(ref_mesh.points, feats.edges, corner_angle_deg)
    else:
        corners = np.zeros(0, dtype=np.int64)
    corner_pts = ref_mesh.points[corners]
    corner_of = np.full(num, -1, dtype=np.int64)

    for it in range(max(1, int(iterations))):
        current = new_mesh if it == 0 else new_mesh.with_points(points)
        target = points.copy()
        tkind = np.full(num, KIND_UNTOUCHED, dtype=np.int8)
        tcorner = np.full(num, -1, dtype=np.int64)

        if seg_grid is not None and tol > 0:
            match = match_features(
                current,
                ref_mesh.points,
                feats,
                diagonal,
                match_tol_rel=match_tol_rel,
                corner_angle_deg=corner_angle_deg,
            )
            loop_idx = np.flatnonzero(match.loop_vertex & used)

            # 1. cantos: cada canto puxa o vértice de loop mais próximo
            if corners.size and loop_idx.size:
                near_d, _, _, _ = point_grid(corner_pts, cell=tol).query(points[loop_idx], max_dist=tol)
                cand = loop_idx[near_d <= tol]
                ci, vi = _snap_corners(corner_pts, points[cand], tol)
                hit = cand[vi]
                target[hit] = corner_pts[ci]
                tkind[hit] = KIND_CORNER
                tcorner[hit] = ci

            # 2. outros vértices de loop: ponto mais próximo da crease que seguem
            rest = loop_idx[tkind[loop_idx] == KIND_UNTOUCHED]
            seg, foot = _curve_targets(points, rest, match.matched_edges, seg_a, seg_b, seg_grid, tol)
            ok = seg >= 0
            target[rest[ok]] = foot[ok]
            tkind[rest[ok]] = KIND_CURVE

        # 3. o resto vai para a superfície
        rest = np.flatnonzero(used & (tkind == KIND_UNTOUCHED))
        if rest.size:
            _, tri, closest, _ = surface.query(points[rest], max_dist=max_move)
            ok = tri >= 0
            target[rest[ok]] = closest[ok]
            tkind[rest[ok]] = KIND_SURFACE

        # 4. ninguém passa do deslocamento máximo (contado desde a entrada)
        step = np.linalg.norm(target - original, axis=1)
        apply = used & (tkind != KIND_UNTOUCHED) & (step <= max_move)
        points[apply] = target[apply]
        kind[apply] = tkind[apply]
        corner_of[apply] = tcorner[apply]

    snapped = int(np.unique(corner_of[kind == KIND_CORNER]).size)
    missing = int(corners.size) - snapped
    counts = {k: int((kind == k).sum()) for k in (KIND_SURFACE, KIND_CURVE, KIND_CORNER)}
    messages.append(
        f"Reprojeção: {counts[KIND_SURFACE]} vértice(s) na superfície, {counts[KIND_CURVE]} em "
        f"curvas de feature e {counts[KIND_CORNER]} em cantos."
    )
    if missing:
        messages.append(
            f"{missing} canto(s) de feature sem vértice de loop a menos de {tol:.4g}; "
            "a quina pode sair arredondada ou cortada."
        )
    far = int((used & (kind == KIND_UNTOUCHED)).sum())
    if far:
        messages.append(
            f"{far} vértice(s) ficaram parados: o destino estava a mais de {max_move:.4g} "
            f"({max_move_rel:.1%} da diagonal)."
        )
    return result(snapped, missing)
