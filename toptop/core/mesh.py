"""Malha poligonal em arrays NumPy e a topologia derivada dela (sem Maya).

O formato é o mesmo de ``MFnMesh.getVertices()``: um array com o número de
vértices de cada face (``counts``) e um array plano com os índices
(``connects``). Isso permite trocar dados com o Maya sem laços em Python e
calcular quase tudo de forma vetorizada.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Sequence

import numpy as np


@dataclass(eq=False)
class Mesh:
    """Malha poligonal imutável por convenção (não altere os arrays no lugar)."""

    points: np.ndarray
    counts: np.ndarray
    connects: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(self.points, dtype=np.float64).reshape(-1, 3)
        self.counts = np.ascontiguousarray(self.counts, dtype=np.int64).reshape(-1)
        self.connects = np.ascontiguousarray(self.connects, dtype=np.int64).reshape(-1)
        if int(self.counts.sum()) != self.connects.size:
            raise ValueError("counts e connects não batem")
        if self.counts.size and int(self.counts.min()) < 3:
            raise ValueError("face com menos de 3 vértices")
        if self.connects.size and (
            int(self.connects.min()) < 0 or int(self.connects.max()) >= len(self.points)
        ):
            raise ValueError("índice de vértice fora do intervalo")

    @classmethod
    def from_faces(
        cls, points: Iterable[Sequence[float]], faces: Sequence[Sequence[int]], name: str = ""
    ) -> "Mesh":
        """Cria a malha a partir de uma lista de faces (listas de índices)."""
        counts = np.fromiter((len(f) for f in faces), dtype=np.int64, count=len(faces))
        connects = np.fromiter(
            (v for f in faces for v in f), dtype=np.int64, count=int(counts.sum())
        )
        return cls(np.asarray(points, dtype=np.float64), counts, connects, name)

    @property
    def num_points(self) -> int:
        """Número de vértices."""
        return len(self.points)

    @property
    def num_faces(self) -> int:
        """Número de faces."""
        return len(self.counts)

    @cached_property
    def offsets(self) -> np.ndarray:
        """Posição inicial de cada face em ``connects`` (tamanho F + 1)."""
        return np.concatenate([[0], np.cumsum(self.counts)]).astype(np.int64)

    @cached_property
    def topology(self) -> "Topology":
        """Topologia derivada (arestas, valências, bordas), calculada uma vez."""
        return Topology(self)

    def faces(self) -> list[tuple[int, ...]]:
        """Faces como lista de tuplas (prático para testes e malhas pequenas)."""
        off = self.offsets
        con = self.connects.tolist()
        return [tuple(con[off[i] : off[i + 1]]) for i in range(self.num_faces)]

    def with_points(self, points: np.ndarray) -> "Mesh":
        """Cópia com os mesmos polígonos e novas posições."""
        return Mesh(points, self.counts, self.connects, self.name)

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """Cantos mínimo e máximo da caixa envolvente dos vértices usados."""
        if self.connects.size == 0:
            zero = np.zeros(3)
            return zero, zero.copy()
        used = self.points[np.unique(self.connects)]
        return used.min(axis=0), used.max(axis=0)

    def diagonal(self) -> float:
        """Diagonal da caixa envolvente (a régua de todas as métricas relativas)."""
        lo, hi = self.bbox()
        return float(np.linalg.norm(hi - lo))


class Topology:
    """Arestas únicas, faces vizinhas, valências e bordas de uma ``Mesh``."""

    def __init__(self, mesh: Mesh) -> None:
        n = mesh.num_points
        counts, connects, offsets = mesh.counts, mesh.connects, mesh.offsets
        num_corners = connects.size

        #: face de cada canto (posição em ``connects``)
        self.corner_face = np.repeat(np.arange(mesh.num_faces, dtype=np.int64), counts)
        nxt = np.arange(1, num_corners + 1, dtype=np.int64)
        if num_corners:
            nxt[offsets[1:] - 1] = offsets[:-1]
        #: canto seguinte dentro da mesma face
        self.corner_next = nxt

        src = connects
        dst = connects[nxt] if num_corners else connects
        lo = np.minimum(src, dst)
        hi = np.maximum(src, dst)
        keys = lo * max(n, 1) + hi
        ukeys, he_edge, edge_count = np.unique(keys, return_inverse=True, return_counts=True)

        self._n = max(n, 1)
        self._keys = ukeys
        #: aresta de cada canto (a que vai do canto ao canto seguinte)
        self.corner_edge = he_edge.reshape(-1).astype(np.int64)
        #: (E, 2) pares de vértices, menor índice primeiro
        self.edges = np.stack([ukeys // self._n, ukeys % self._n], axis=1).astype(np.int64)
        #: número de faces em cada aresta (1 = borda, 2 = normal, >2 = non-manifold)
        self.edge_face_count = edge_count.astype(np.int64)

        num_edges = len(ukeys)
        order = np.argsort(self.corner_edge, kind="stable")
        first = np.searchsorted(self.corner_edge[order], np.arange(num_edges))
        edge_faces = np.full((num_edges, 2), -1, dtype=np.int64)
        edge_corners = np.full((num_edges, 2), -1, dtype=np.int64)
        if num_edges:
            edge_corners[:, 0] = order[first]
            edge_faces[:, 0] = self.corner_face[order[first]]
            two = edge_count >= 2
            edge_corners[two, 1] = order[first[two] + 1]
            edge_faces[two, 1] = self.corner_face[order[first[two] + 1]]
        #: (E, 2) as duas primeiras faces de cada aresta (-1 quando não existe)
        self.edge_faces = edge_faces
        #: (E, 2) um canto de cada uma dessas faces (o canto de onde a aresta sai)
        self.edge_corners = edge_corners

        #: aresta degenerada (os dois vértices iguais); só aparece em malha suja
        self.degenerate_edge = self.edges[:, 0] == self.edges[:, 1]
        self.boundary_edge = (self.edge_face_count == 1) & ~self.degenerate_edge
        self.nonmanifold_edge = self.edge_face_count > 2

        valid = self.edges[~self.degenerate_edge]
        #: número de arestas em cada vértice
        self.vertex_valence = np.bincount(valid.reshape(-1), minlength=n).astype(np.int64)
        boundary_vertex = np.zeros(n, dtype=bool)
        boundary_vertex[self.edges[self.boundary_edge].reshape(-1)] = True
        self.boundary_vertex = boundary_vertex
        used = np.zeros(n, dtype=bool)
        used[connects] = True
        #: vértice referenciado por pelo menos uma face
        self.used_vertex = used

    @property
    def num_edges(self) -> int:
        """Número de arestas únicas."""
        return len(self.edges)

    def edge_ids(self, pairs: np.ndarray) -> np.ndarray:
        """Índice da aresta de cada par de vértices; -1 se o par não é aresta."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if len(self._keys) == 0 or len(pairs) == 0:
            return np.full(len(pairs), -1, dtype=np.int64)
        lo = pairs.min(axis=1)
        hi = pairs.max(axis=1)
        keys = lo * self._n + hi
        pos = np.minimum(np.searchsorted(self._keys, keys), len(self._keys) - 1)
        in_range = (lo >= 0) & (hi < self._n)
        return np.where(in_range & (self._keys[pos] == keys), pos, -1).astype(np.int64)


def face_normals_areas(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Normal unitária e área de cada face pelo método de Newell."""
    if mesh.num_faces == 0:
        return np.zeros((0, 3)), np.zeros(0)
    center = mesh.points[np.unique(mesh.connects)].mean(axis=0)
    p = mesh.points[mesh.connects] - center
    q = p[mesh.topology.corner_next]
    vec = np.add.reduceat(np.cross(p, q), mesh.offsets[:-1], axis=0)
    length = np.linalg.norm(vec, axis=1)
    normals = vec / np.where(length > 0, length, 1.0)[:, None]
    return normals, 0.5 * length


def face_centers(mesh: Mesh) -> np.ndarray:
    """Média dos vértices de cada face."""
    if mesh.num_faces == 0:
        return np.zeros((0, 3))
    total = np.add.reduceat(mesh.points[mesh.connects], mesh.offsets[:-1], axis=0)
    return total / mesh.counts[:, None]


def edge_lengths(mesh: Mesh) -> np.ndarray:
    """Comprimento de cada aresta única."""
    e = mesh.topology.edges
    return np.linalg.norm(mesh.points[e[:, 1]] - mesh.points[e[:, 0]], axis=1)


def _ear_clip(pts: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Triangula um polígono simples (possivelmente côncavo) cortando orelhas."""
    n = len(ids)
    centered = pts - pts.mean(axis=0)
    normal = np.cross(centered, np.roll(centered, -1, axis=0)).sum(axis=0)
    axis = int(np.argmax(np.abs(normal)))
    sign = 1.0 if normal[axis] >= 0 else -1.0
    u, v = ((1, 2), (2, 0), (0, 1))[axis]
    x = centered[:, u].tolist()
    y = centered[:, v].tolist()
    scale = float(np.abs(centered).max()) or 1.0
    eps = 1e-12 * scale * scale
    same_tol = 1e-9 * scale

    def cross(o: int, a: int, b: int) -> float:
        return sign * ((x[a] - x[o]) * (y[b] - y[o]) - (y[a] - y[o]) * (x[b] - x[o]))

    def same(i: int, j: int) -> bool:
        return abs(x[i] - x[j]) <= same_tol and abs(y[i] - y[j]) <= same_tol

    def find_ear(idx: list[int], strict: bool) -> int:
        """Posição em ``idx`` de uma orelha válida, ou -1.

        Com ``strict`` falso, um vértice encostado na borda do triângulo também
        bloqueia a orelha (evita cortar por cima de vértices colineares); com
        ``strict`` verdadeiro só bloqueia o que está estritamente dentro.
        """
        m = len(idx)
        for i in range(m):
            a, b, c = idx[i - 1], idx[i], idx[(i + 1) % m]
            if cross(a, b, c) <= eps:
                continue  # canto reflexo ou degenerado: não é orelha
            blocked = False
            for j in idx:
                if j == a or j == b or j == c:
                    continue
                if strict:
                    inside = cross(a, b, j) > eps and cross(b, c, j) > eps and cross(c, a, j) > eps
                else:
                    inside = (
                        cross(a, b, j) >= -eps
                        and cross(b, c, j) >= -eps
                        and cross(c, a, j) >= -eps
                        and not (same(j, a) or same(j, b) or same(j, c))
                    )
                if inside:
                    blocked = True
                    break
            if not blocked:
                return i
        return -1

    idx = list(range(n))
    out: list[tuple[int, int, int]] = []
    while len(idx) > 3:
        i = find_ear(idx, strict=False)
        if i < 0:
            i = find_ear(idx, strict=True)
        if i < 0:
            break  # polígono degenerado: o resto vira leque
        out.append((idx[i - 1], idx[i], idx[(i + 1) % len(idx)]))
        idx.pop(i)
    for k in range(1, len(idx) - 1):
        out.append((idx[0], idx[k], idx[k + 1]))
    return ids[np.asarray(out, dtype=np.int64)]


def triangulate(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Triangula a malha; devolve (T, 3) índices e a face de origem de cada triângulo.

    Triângulos passam direto, quads são divididos pela diagonal que não inverte
    nenhum dos dois triângulos e ngons são cortados por orelhas. Os triângulos
    saem agrupados por face, na ordem das faces.
    """
    counts, connects = mesh.counts, mesh.connects
    off = mesh.offsets[:-1]
    pts = mesh.points
    ntri = counts - 2
    toff = np.concatenate([[0], np.cumsum(ntri)]).astype(np.int64)
    tris = np.empty((int(toff[-1]), 3), dtype=np.int64)
    tri_face = np.repeat(np.arange(len(counts), dtype=np.int64), ntri)

    m3 = np.flatnonzero(counts == 3)
    if m3.size:
        tris[toff[m3]] = connects[off[m3][:, None] + np.arange(3)]

    m4 = np.flatnonzero(counts == 4)
    if m4.size:
        q = connects[off[m4][:, None] + np.arange(4)]
        p = pts[q]
        normal = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])

        def signed(i: int, j: int, k: int) -> np.ndarray:
            return np.einsum("ij,ij->i", np.cross(p[:, j] - p[:, i], p[:, k] - p[:, i]), normal)

        score_02 = np.minimum(signed(0, 1, 2), signed(0, 2, 3))
        score_13 = np.minimum(signed(0, 1, 3), signed(1, 2, 3))
        use_13 = (score_13 > score_02)[:, None]
        tris[toff[m4]] = np.where(use_13, q[:, [0, 1, 3]], q[:, [0, 1, 2]])
        tris[toff[m4] + 1] = np.where(use_13, q[:, [1, 2, 3]], q[:, [0, 2, 3]])

    for f in np.flatnonzero(counts > 4).tolist():
        poly = connects[off[f] : off[f] + counts[f]]
        tris[toff[f] : toff[f + 1]] = _ear_clip(pts[poly], poly)
    return tris, tri_face


def triangle_mesh(mesh: Mesh) -> Mesh:
    """Versão só com triângulos da malha (mesmos vértices, mesmos índices)."""
    tris, _ = triangulate(mesh)
    return Mesh(mesh.points, np.full(len(tris), 3, dtype=np.int64), tris.reshape(-1), mesh.name)


def connected_labels(num_nodes: int, edges: np.ndarray) -> np.ndarray:
    """Rótulo do componente conexo de cada nó de um grafo (união vetorizada)."""
    parent = np.arange(num_nodes, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return parent
    a, b = edges[:, 0], edges[:, 1]
    while True:
        pa, pb = parent[a], parent[b]
        lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
        differ = lo != hi
        if not differ.any():
            break
        np.minimum.at(parent, hi[differ], lo[differ])
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped
    return parent


def shells(mesh: Mesh) -> tuple[np.ndarray, int]:
    """Casca (componente conexo) de cada face e o número de cascas."""
    if mesh.num_faces == 0:
        return np.zeros(0, dtype=np.int64), 0
    roots = connected_labels(mesh.num_points, mesh.topology.edges)
    face_root = roots[mesh.connects[mesh.offsets[:-1]]]
    _, labels = np.unique(face_root, return_inverse=True)
    labels = labels.reshape(-1).astype(np.int64)
    return labels, int(labels.max()) + 1


def extract_faces(mesh: Mesh, face_mask: np.ndarray) -> tuple[Mesh, np.ndarray]:
    """Submalha com as faces escolhidas; devolve também o índice antigo de cada vértice."""
    face_mask = np.asarray(face_mask)
    if face_mask.dtype != bool:
        chosen = np.zeros(mesh.num_faces, dtype=bool)
        chosen[face_mask] = True
        face_mask = chosen
    corner_mask = np.repeat(face_mask, mesh.counts)
    old_ids, new_connects = np.unique(mesh.connects[corner_mask], return_inverse=True)
    sub = Mesh(mesh.points[old_ids], mesh.counts[face_mask], new_connects.reshape(-1), mesh.name)
    return sub, old_ids


def subdivide_triangles(mesh: Mesh, levels: int = 1) -> tuple[Mesh, np.ndarray]:
    """Subdivisão linear (sem suavizar): cada triângulo vira 4, a forma não muda.

    Devolve a malha nova e, para cada vértice, um par de vértices originais da
    aresta onde ele nasceu (o próprio índice, duas vezes, para os que já
    existiam; aproximado a partir do segundo nível). Os vértices antigos mantêm
    o índice.
    """
    tris = triangle_mesh(mesh).connects.reshape(-1, 3)
    points = mesh.points
    origin = np.stack([np.arange(len(points)), np.arange(len(points))], axis=1).astype(np.int64)
    for _ in range(max(0, int(levels))):
        n = len(points)
        pairs = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
        lo, hi = pairs.min(axis=1), pairs.max(axis=1)
        keys, inverse = np.unique(lo * n + hi, return_inverse=True)
        inverse = inverse.reshape(-1)
        ea, eb = keys // n, keys % n
        points = np.concatenate([points, 0.5 * (points[ea] + points[eb])])
        origin = np.concatenate([origin, np.stack([origin[ea, 0], origin[eb, 1]], axis=1)])
        m = len(tris)
        mid = n + inverse
        m01, m12, m20 = mid[:m], mid[m : 2 * m], mid[2 * m :]
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        tris = np.concatenate(
            [
                np.stack([a, m01, m20], axis=1),
                np.stack([m01, b, m12], axis=1),
                np.stack([m20, m12, c], axis=1),
                np.stack([m01, m12, m20], axis=1),
            ]
        )
    out = Mesh(points, np.full(len(tris), 3, dtype=np.int64), tris.reshape(-1), mesh.name)
    return out, origin
