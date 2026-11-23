"""Validar, limpar, separar cascas e reduzir a malha de entrada (passos 1 a 3 do plano).

1. **Validar e limpar**: tudo trabalha numa cópia; solda vértices duplicados,
   remove faces degeneradas, de área zero e repetidas, aponta o que é
   non-manifold e triangula. Bordas abertas viram feature guia sozinhas.
2. **Separar cascas**: cada casca conectada é processada sozinha.
3. **Reduzir se preciso**: acima do limite de triângulos, ``maya_reduce`` roda o
   ``polyReduce`` novo preservando creases, hard edges e bordas.

Tudo é NumPy puro e testado sem Maya, menos ``maya_reduce``, que importa
``maya`` dentro da função.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .creases import GUIDE_VALUE, FeatureSet
from .mesh import Mesh, connected_labels, face_normals_areas, shells
from .mesh import triangulate as triangulate_faces

DEFAULT_WELD_TOL_REL = 1e-6
DEFAULT_AREA_TOL_REL = 1e-12
DEFAULT_TRIANGLE_LIMIT = 100_000

# limite de células por eixo da grade de solda: 2**20 por eixo cabe em int64
_MAX_CELLS_PER_AXIS = 2**20 - 4
# metade dos 26 vizinhos de uma célula (a outra metade é simétrica)
_HALF_NEIGHBORS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
)


# --------------------------------------------------------------------------
# Auxiliares
# --------------------------------------------------------------------------


def _corner_next(mesh: Mesh) -> np.ndarray:
    """Canto seguinte de cada canto dentro da mesma face (sem montar a topologia)."""
    num = mesh.connects.size
    nxt = np.arange(1, num + 1, dtype=np.int64)
    if num:
        off = mesh.offsets
        nxt[off[1:] - 1] = off[:-1]
    return nxt


def _used_mask(mesh: Mesh) -> np.ndarray:
    """Vértices referenciados por pelo menos uma face."""
    used = np.zeros(mesh.num_points, dtype=bool)
    used[mesh.connects] = True
    return used


def _group_starts(sorted_values: np.ndarray) -> np.ndarray:
    """Início de cada sequência de valores iguais num array já ordenado."""
    if sorted_values.size == 0:
        return np.zeros(0, dtype=np.int64)
    new = np.ones(len(sorted_values), dtype=bool)
    new[1:] = sorted_values[1:] != sorted_values[:-1]
    return np.flatnonzero(new)


def num_triangles(mesh: Mesh) -> int:
    """Número de triângulos da malha triangulada (soma de lados - 2 de cada face)."""
    return int((mesh.counts - 2).sum())


def compact_vertices(mesh: Mesh) -> tuple[Mesh, np.ndarray]:
    """Descarta os vértices sem face, mantendo a ordem.

    Devolve a malha nova e o novo índice de cada vértice antigo (-1 = descartado).
    """
    used = _used_mask(mesh)
    new_index = np.full(mesh.num_points, -1, dtype=np.int64)
    new_index[used] = np.arange(int(used.sum()), dtype=np.int64)
    out = Mesh(mesh.points[used], mesh.counts.copy(), new_index[mesh.connects], mesh.name)
    return out, new_index


# --------------------------------------------------------------------------
# Solda
# --------------------------------------------------------------------------


def _exact_groups(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Agrupa pontos com coordenadas idênticas.

    Devolve o grupo de cada ponto e o primeiro ponto (menor índice) de cada
    grupo; os grupos saem na ordem da primeira ocorrência.
    """
    num = len(points)
    if num == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    ordered = points[order]
    new = np.ones(num, dtype=bool)
    new[1:] = (ordered[1:] != ordered[:-1]).any(axis=1)
    sorted_group = np.cumsum(new) - 1
    # lexsort é estável: o primeiro de cada grupo é o de menor índice
    first = order[new]
    rank = np.argsort(first, kind="stable")
    relabel = np.empty(len(rank), dtype=np.int64)
    relabel[rank] = np.arange(len(rank), dtype=np.int64)
    group = np.empty(num, dtype=np.int64)
    group[order] = relabel[sorted_group]
    return group, first[rank]


def _block_pairs(
    start_a: np.ndarray, size_a: np.ndarray, start_b: np.ndarray, size_b: np.ndarray, same: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Todos os pares entre os blocos ``a`` e ``b`` (posições na ordem por célula).

    Com ``same`` os blocos são a mesma célula e só saem os pares i < j.
    """
    sizes = size_a * size_b
    total = int(sizes.sum())
    rep = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    local = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    width = size_b[rep]
    i = start_a[rep] + local // width
    j = start_b[rep] + local % width
    if same:
        keep = i < j
        i, j = i[keep], j[keep]
    return i, j


def close_pairs(points: np.ndarray, tol: float) -> np.ndarray:
    """Pares (i, j), i < j, de pontos a distância <= ``tol`` (grade uniforme vetorizada).

    Cada célula tem lado >= ``tol``; um par próximo está na mesma célula ou numa
    das 26 vizinhas (inclusive na diagonal), então pares que atravessam a divisa
    entre células também são achados. Os pontos devem ser finitos. Pontos
    idênticos em grande número numa célula geram muitos candidatos: agrupe-os
    antes (``weld_vertices`` faz isso).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    num = len(points)
    if num < 2 or not tol >= 0:
        return np.zeros((0, 2), dtype=np.int64)
    lo = points.min(axis=0)
    extent = float((points.max(axis=0) - lo).max())
    cell = max(float(tol) * (1.0 + 1e-6), extent / _MAX_CELLS_PER_AXIS)
    if cell <= 0:
        cell = 1.0
    ijk = np.floor((points - lo) / cell).astype(np.int64) + 1
    dims = ijk.max(axis=0) + 2
    stride = np.array([dims[1] * dims[2], dims[2], 1], dtype=np.int64)
    keys = (ijk * stride).sum(axis=1)

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    start = _group_starts(sorted_keys)
    cells = sorted_keys[start]
    size = np.diff(np.append(start, num)).astype(np.int64)

    tol2 = float(tol) * float(tol)
    found: list[np.ndarray] = []

    def keep_close(i: np.ndarray, j: np.ndarray) -> None:
        """Guarda (em índices originais) só os candidatos que estão mesmo perto."""
        a, b = order[i], order[j]
        diff = points[a] - points[b]
        near = np.einsum("ij,ij->i", diff, diff) <= tol2
        if near.any():
            found.append(np.stack([a[near], b[near]], axis=1))

    multi = np.flatnonzero(size > 1)
    if multi.size:
        keep_close(*_block_pairs(start[multi], size[multi], start[multi], size[multi], same=True))
    for dx, dy, dz in _HALF_NEIGHBORS:
        target = cells + (dx * stride[0] + dy * stride[1] + dz)
        pos = np.minimum(np.searchsorted(cells, target), len(cells) - 1)
        hit = np.flatnonzero(cells[pos] == target)
        if hit.size:
            other = pos[hit]
            keep_close(*_block_pairs(start[hit], size[hit], start[other], size[other], same=False))
    if not found:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.concatenate(found)
    a, b = pairs[:, 0], pairs[:, 1]
    return np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1).astype(np.int64)


def weld_vertices(mesh: Mesh, tol_rel: float = DEFAULT_WELD_TOL_REL) -> tuple[Mesh, np.ndarray]:
    """Solda os vértices mais próximos que ``tol_rel * diagonal``.

    A relação é transitiva (uma cadeia de vértices próximos vira um só). Cada
    grupo fica na posição do seu vértice de menor índice. Vértices sem face são
    descartados e a ordem de primeira ocorrência é mantida, então uma malha já
    soldada volta com a mesma numeração. Devolve a malha nova e o novo índice de
    cada vértice antigo (-1 se ele não era usado por nenhuma face).
    """
    vertex_map = np.full(mesh.num_points, -1, dtype=np.int64)
    used_ids = np.flatnonzero(_used_mask(mesh))
    pts = mesh.points[used_ids]
    group, first = _exact_groups(pts)
    reps = pts[first]

    finite = np.flatnonzero(np.isfinite(reps).all(axis=1))
    pairs = np.zeros((0, 2), dtype=np.int64)
    if len(finite) > 1 and tol_rel > 0:
        fin = reps[finite]
        tol = float(tol_rel) * float(np.linalg.norm(fin.max(axis=0) - fin.min(axis=0)))
        pairs = finite[close_pairs(fin, tol)]
    roots = connected_labels(len(reps), pairs)
    kept_roots, cluster = np.unique(roots, return_inverse=True)
    cluster = cluster.reshape(-1).astype(np.int64)
    vertex_map[used_ids] = cluster[group]
    out = Mesh(reps[kept_roots], mesh.counts.copy(), vertex_map[mesh.connects], mesh.name)
    return out, vertex_map


# --------------------------------------------------------------------------
# Faces degeneradas e repetidas
# --------------------------------------------------------------------------


def _duplicate_faces(mesh: Mesh, candidates: np.ndarray) -> np.ndarray:
    """Faces candidatas com o mesmo conjunto de vértices de uma candidata anterior."""
    dup = np.zeros(mesh.num_faces, dtype=bool)
    faces = np.flatnonzero(candidates)
    if faces.size < 2:
        return dup
    off = mesh.offsets[:-1]
    faces = faces[np.argsort(mesh.counts[faces], kind="stable")]
    sizes = mesh.counts[faces]
    starts = _group_starts(sizes)
    ends = np.append(starts[1:], len(faces))
    for s, e in zip(starts.tolist(), ends.tolist()):
        if e - s < 2:
            continue
        group = faces[s:e]
        k = int(sizes[s])
        rows = np.sort(mesh.connects[off[group][:, None] + np.arange(k)], axis=1)
        order = np.lexsort(rows.T[::-1])  # estável: a primeira face de cada grupo vem antes
        ordered = rows[order]
        same = np.zeros(len(group), dtype=bool)
        same[1:] = (ordered[1:] == ordered[:-1]).all(axis=1)
        dup[group[order[same]]] = True
    return dup


def _remove_degenerate(mesh: Mesh, area_tol_rel: float) -> tuple[Mesh, np.ndarray, tuple[int, int, int]]:
    """Versão de ``remove_degenerate_faces`` que também conta cada motivo.

    Os contadores são (menos de 3 vértices distintos, área zero, duplicadas).
    """
    nf = mesh.num_faces
    if nf == 0:
        return Mesh(mesh.points, mesh.counts, mesh.connects, mesh.name), np.ones(0, dtype=bool), (0, 0, 0)
    connects = mesh.connects
    corner_face = np.repeat(np.arange(nf, dtype=np.int64), mesh.counts)

    # 1. vértice repetido em sequência (inclusive o último igual ao primeiro)
    keep_corner = connects != connects[_corner_next(mesh)]
    new_counts = np.bincount(corner_face[keep_corner], minlength=nf).astype(np.int64)
    alive = new_counts >= 3
    # faces com 4+ cantos ainda podem repetir vértice fora de sequência (a, b, a, b)
    big = alive & (new_counts >= 4)
    if big.any():
        sel = keep_corner & big[corner_face]
        base = max(mesh.num_points, 1)
        keys = np.unique(corner_face[sel] * base + connects[sel])
        distinct = np.bincount(keys // base, minlength=nf)
        alive &= ~big | (distinct >= 3)
    collapsed = int((~alive).sum())

    mid = Mesh(mesh.points, new_counts[alive], connects[keep_corner & alive[corner_face]], mesh.name)
    # 2. área zero (Newell), relativa à diagonal da malha de entrada
    _, areas = face_normals_areas(mid)
    diag = mesh.diagonal()
    flat = areas <= float(area_tol_rel) * diag * diag
    # 3. duplicadas: mesmo conjunto de vértices, qualquer rotação ou sentido
    dup = _duplicate_faces(mid, ~flat)

    keep_mid = ~flat & ~dup
    keep = np.zeros(nf, dtype=bool)
    keep[np.flatnonzero(alive)[keep_mid]] = True
    out = Mesh(mesh.points, mid.counts[keep_mid], mid.connects[np.repeat(keep_mid, mid.counts)], mesh.name)
    return out, keep, (collapsed, int(flat.sum()), int(dup.sum()))


def remove_degenerate_faces(
    mesh: Mesh, area_tol_rel: float = DEFAULT_AREA_TOL_REL
) -> tuple[Mesh, np.ndarray]:
    """Remove faces degeneradas, de área zero e duplicadas.

    Vértices repetidos em sequência dentro de uma face são tirados; caem as
    faces que ficam com menos de 3 vértices distintos, as de área <=
    ``area_tol_rel * diagonal**2`` e as repetidas (mesmo conjunto de vértices,
    em qualquer rotação ou sentido; fica a primeira). Os vértices não mudam.
    Devolve a malha nova e a máscara das faces de entrada que ficaram
    (verdadeira também para as que só foram encurtadas).
    """
    out, keep, _ = _remove_degenerate(mesh, area_tol_rel)
    return out, keep


# --------------------------------------------------------------------------
# Non-manifold
# --------------------------------------------------------------------------


def fan_count(mesh: Mesh) -> np.ndarray:
    """Número de leques de faces em volta de cada vértice.

    Duas faces no mesmo vértice estão no mesmo leque quando dá para ir de uma
    à outra passando só por arestas que tocam esse vértice. Vértice sem face
    tem 0; vértice normal (inclusive de borda) tem 1.
    """
    con = mesh.connects
    num_corners = con.size
    if num_corners == 0:
        return np.zeros(mesh.num_points, dtype=np.int64)
    topo = mesh.topology
    nxt = topo.corner_next
    corner_edge = topo.corner_edge
    order = np.argsort(corner_edge, kind="stable")
    a, b = order[:-1], order[1:]
    same_edge = corner_edge[a] == corner_edge[b]
    a, b = a[same_edge], b[same_edge]
    # canto de b em cada ponta da aresta, conforme o sentido da meia-aresta
    same_dir = con[a] == con[b]
    b_at_start = np.where(same_dir, b, nxt[b])
    b_at_end = np.where(same_dir, nxt[b], b)
    pairs = np.concatenate([np.stack([a, b_at_start], axis=1), np.stack([nxt[a], b_at_end], axis=1)])
    labels = connected_labels(num_corners, pairs)
    keys = np.unique(con * num_corners + labels)
    return np.bincount(keys // num_corners, minlength=mesh.num_points).astype(np.int64)


def nonmanifold_report(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Arestas e vértices non-manifold.

    Devolve (K, 2) pares de vértices das arestas com mais de 2 faces e os
    índices dos vértices "gravata-borboleta" (faces em volta formam mais de um
    leque). Vértice de borda com um leque só não conta.
    """
    topo = mesh.topology
    edges = topo.edges[topo.nonmanifold_edge].copy()
    return edges, np.flatnonzero(fan_count(mesh) > 1).astype(np.int64)


# --------------------------------------------------------------------------
# Limpeza completa
# --------------------------------------------------------------------------


@dataclass(eq=False)
class CleanResult:
    """Malha limpa, features renumeradas e o relatório do que foi feito."""

    mesh: Mesh
    features: FeatureSet
    vertex_map: np.ndarray
    welded_vertices: int
    removed_faces: int
    nonmanifold_edges: np.ndarray
    nonmanifold_vertices: np.ndarray
    boundary_edges: np.ndarray
    messages: list[str] = field(default_factory=list)
    #: face de entrada de onde veio cada face da malha limpa
    face_origin: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def ok(self) -> bool:
        """Verdadeiro quando não sobrou nada non-manifold."""
        return len(self.nonmanifold_edges) == 0 and len(self.nonmanifold_vertices) == 0


def _remap_features(
    features: FeatureSet, vertex_map: np.ndarray, mesh: Mesh
) -> tuple[FeatureSet, int, int]:
    """Renumera as features para a malha limpa.

    Devolve as features que são arestas da malha limpa, quantas sumiram na
    solda (as duas pontas viraram um vértice só) e quantas não são arestas.
    """
    if len(features) == 0:
        return FeatureSet.empty(), 0, 0
    edges = features.edges
    in_range = (edges >= 0).all(axis=1) & (edges < len(vertex_map)).all(axis=1)
    mapped = np.full(edges.shape, -1, dtype=np.int64)
    mapped[in_range] = vertex_map[edges[in_range]]
    valid = (mapped >= 0).all(axis=1)
    collapsed = valid & (mapped[:, 0] == mapped[:, 1])
    candidate = valid & ~collapsed
    on_mesh = np.zeros(len(edges), dtype=bool)
    on_mesh[candidate] = mesh.topology.edge_ids(mapped[candidate]) >= 0
    kept = FeatureSet(mapped[on_mesh], features.crease[on_mesh])
    return kept, int(collapsed.sum()), int((~collapsed & ~on_mesh).sum())


def clean_mesh(
    mesh: Mesh,
    features: FeatureSet | None = None,
    *,
    weld_tol_rel: float = DEFAULT_WELD_TOL_REL,
    triangulate: bool = True,
    border_value: float = GUIDE_VALUE,
    area_tol_rel: float = DEFAULT_AREA_TOL_REL,
) -> CleanResult:
    """Limpa uma cópia da malha: solda, tira faces ruins, triangula e aponta non-manifold.

    As features (na numeração da malha de entrada) são renumeradas; as que
    somem na solda ou não são arestas da malha limpa caem e são contadas nas
    mensagens. As bordas abertas entram como feature guia com crease
    ``border_value`` (0 desliga). ``weld_tol_rel`` e ``area_tol_rel`` são as
    tolerâncias de ``weld_vertices`` e ``remove_degenerate_faces``. A malha de
    entrada não é alterada.
    """
    messages: list[str] = []
    welded, weld_map = weld_vertices(mesh, weld_tol_rel)
    used_count = int((weld_map >= 0).sum())
    unused = mesh.num_points - used_count
    welded_count = used_count - welded.num_points

    tidy, keep, (collapsed, flat, dup) = _remove_degenerate(welded, area_tol_rel)
    face_origin = np.flatnonzero(keep).astype(np.int64)
    if triangulate and tidy.num_faces:
        tris, tri_face = triangulate_faces(tidy)
        tidy = Mesh(tidy.points, np.full(len(tris), 3, dtype=np.int64), tris.reshape(-1), tidy.name)
        face_origin = face_origin[tri_face]
    clean, compact_map = compact_vertices(tidy)
    vertex_map = np.full(mesh.num_points, -1, dtype=np.int64)
    has = weld_map >= 0
    vertex_map[has] = compact_map[weld_map[has]]

    feats, lost_weld, not_edges = _remap_features(
        features if features is not None else FeatureSet.empty(), vertex_map, clean
    )
    nm_edges, nm_vertices = nonmanifold_report(clean)
    topo = clean.topology
    borders = topo.edges[topo.boundary_edge].copy()
    if len(borders) and border_value > 0:
        feats = feats.merged(FeatureSet(borders, np.full(len(borders), float(border_value))))

    if unused:
        messages.append(f"{unused} vértice(s) sem face descartado(s).")
    if welded_count:
        messages.append(f"{welded_count} vértice(s) duplicado(s) soldado(s).")
    if collapsed:
        messages.append(f"{collapsed} face(s) com menos de 3 vértices distintos removida(s).")
    if flat:
        messages.append(f"{flat} face(s) de área zero removida(s).")
    if dup:
        messages.append(f"{dup} face(s) duplicada(s) removida(s).")
    if lost_weld:
        messages.append(f"{lost_weld} aresta(s) de feature sumiram na solda (as duas pontas viraram um vértice só).")
    if not_edges:
        messages.append(f"{not_edges} aresta(s) de feature não são arestas da malha limpa e foram ignoradas.")
    if len(nm_edges):
        messages.append(f"{len(nm_edges)} aresta(s) non-manifold (mais de 2 faces).")
    if len(nm_vertices):
        messages.append(f"{len(nm_vertices)} vértice(s) non-manifold (faces ligadas só pelo vértice).")
    if len(borders):
        if border_value > 0:
            messages.append(f"{len(borders)} aresta(s) de borda aberta viraram feature guia (crease {border_value:g}).")
        else:
            messages.append(f"{len(borders)} aresta(s) de borda aberta.")
    if not messages:
        messages.append("Malha limpa: nada a corrigir.")

    return CleanResult(
        mesh=clean,
        features=feats,
        vertex_map=vertex_map,
        welded_vertices=int(welded_count),
        removed_faces=int((~keep).sum()),
        nonmanifold_edges=nm_edges,
        nonmanifold_vertices=nm_vertices,
        boundary_edges=borders,
        messages=messages,
        face_origin=face_origin,
    )


# --------------------------------------------------------------------------
# Cascas
# --------------------------------------------------------------------------


@dataclass(eq=False)
class Shell:
    """Uma casca conectada, com as features renumeradas para ela."""

    mesh: Mesh
    features: FeatureSet
    #: índice, na malha de entrada, de cada vértice da casca
    vertex_ids: np.ndarray
    #: índice, na malha de entrada, de cada face da casca
    face_ids: np.ndarray


def _segments(labels: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Ordem estável por rótulo e o início de cada rótulo nessa ordem (tamanho count + 1)."""
    order = np.argsort(labels, kind="stable")
    start = np.concatenate([[0], np.cumsum(np.bincount(labels, minlength=count))]).astype(np.int64)
    return order.astype(np.int64), start


def split_shells(mesh: Mesh, features: FeatureSet | None = None) -> list[Shell]:
    """Separa as cascas conectadas, da maior (em faces) para a menor.

    Cada casca traz só os vértices que usa (na ordem original), as faces na
    ordem original e as features cujas duas pontas estão nela.
    """
    labels, count = shells(mesh)
    if count == 0:
        return []
    features = features if features is not None else FeatureSet.empty()
    n = mesh.num_points

    face_order, face_start = _segments(labels, count)
    corner_label = np.repeat(labels, mesh.counts)
    corner_order, corner_start = _segments(corner_label, count)
    vert_label = np.full(n, -1, dtype=np.int64)
    vert_label[mesh.connects] = corner_label
    used = np.flatnonzero(vert_label >= 0)
    sub_order, vert_start = _segments(vert_label[used], count)
    vert_order = used[sub_order]
    local = np.full(n, -1, dtype=np.int64)
    local[vert_order] = np.arange(len(vert_order), dtype=np.int64) - np.repeat(
        vert_start[:-1], np.diff(vert_start)
    )
    local_connects = local[mesh.connects[corner_order]]

    edges = features.edges
    in_range = (edges >= 0).all(axis=1) & (edges < n).all(axis=1) if len(edges) else np.zeros(0, dtype=bool)
    edge_label = np.full(len(edges), -1, dtype=np.int64)
    if len(edges):
        pair_label = vert_label[np.where(in_range[:, None], edges, 0)]
        same = in_range & (pair_label[:, 0] == pair_label[:, 1]) & (pair_label[:, 0] >= 0)
        edge_label[same] = pair_label[same, 0]
    chosen = np.flatnonzero(edge_label >= 0)
    sub_feat, feat_start = _segments(edge_label[chosen], count)
    feat_order = chosen[sub_feat]

    out: list[Shell] = []
    for k in range(count):
        face_ids = face_order[face_start[k] : face_start[k + 1]]
        vertex_ids = vert_order[vert_start[k] : vert_start[k + 1]]
        sub = Mesh(
            mesh.points[vertex_ids],
            mesh.counts[face_ids],
            local_connects[corner_start[k] : corner_start[k + 1]],
            mesh.name,
        )
        sel = feat_order[feat_start[k] : feat_start[k + 1]]
        feats = FeatureSet(local[edges[sel]], features.crease[sel]) if sel.size else FeatureSet.empty()
        out.append(Shell(sub, feats, vertex_ids, face_ids))
    out.sort(key=lambda s: -s.mesh.num_faces)
    return out


# --------------------------------------------------------------------------
# Redução
# --------------------------------------------------------------------------


def reduction_target(mesh: Mesh, limit_triangles: int = DEFAULT_TRIANGLE_LIMIT) -> int | None:
    """Alvo de triângulos para reduzir, ou ``None`` se a malha já está no limite."""
    if num_triangles(mesh) <= int(limit_triangles):
        return None
    return int(limit_triangles)


def maya_reduce(
    name: str,
    triangle_count: int,
    *,
    crease_weight: float | None = None,
    border_weight: float | None = None,
) -> None:
    """Reduz NO LUGAR a malha ``name`` do Maya até ~``triangle_count`` triângulos.

    Usa o algoritmo novo do ``polyReduce`` (``version=1``) parando por número de
    triângulos e preservando creases, hard edges, bordas e a topologia; depois
    apaga o histórico. ``crease_weight`` e ``border_weight`` (0 a 1; 1 preserva
    exatamente) só são passados quando dados; senão vale o padrão do Maya (0,5).
    Chame só na cópia de trabalho, nunca no original, e dentro do
    ``undo_chunk`` da operação. Os índices de vértice mudam: releia malha e
    creases depois.
    """
    from maya import cmds

    flags: dict[str, float | int | bool] = {
        "version": 1,
        "termination": 2,
        "triangleCount": int(triangle_count),
        "keepCreaseEdge": True,
        "keepHardEdge": True,
        "keepBorder": True,
        "preserveTopology": True,
        "constructionHistory": False,
    }
    if crease_weight is not None:
        flags["keepCreaseEdgeWeight"] = float(crease_weight)
    if border_weight is not None:
        flags["keepBorderWeight"] = float(border_weight)
    cmds.polyReduce(name, **flags)
    cmds.delete(name, constructionHistory=True)
