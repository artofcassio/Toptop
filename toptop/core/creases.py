"""Creases: leitura, dois níveis, sugestão por ângulo e "Verificar entrada".

A lógica geométrica é NumPy puro e testada sem Maya. As funções ``maya_*`` no
fim do arquivo são a camada fina que lê e grava creases numa malha do Maya.

Dois níveis (seção 6 do plano): crease >= limiar é **forte** (edge loop exato,
hard edge, corte de UV, support loops no Film); 0 < crease < limiar é **guia**
(edge loop exato, sem hard edge). Para o motor, os dois são feature obrigatória.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh import Mesh, face_normals_areas

DEFAULT_STRONG_THRESHOLD = 2.0
DEFAULT_SUGGEST_ANGLE = 30.0
DEFAULT_CORNER_ANGLE = 50.0
STRONG_VALUE = 3.0
GUIDE_VALUE = 1.0


def _normalize_pairs(edges: np.ndarray) -> np.ndarray:
    """Pares (K, 2) com o menor índice primeiro."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return np.stack([edges.min(axis=1), edges.max(axis=1)], axis=1) if len(edges) else edges


@dataclass(eq=False)
class FeatureSet:
    """Arestas de feature como pares de vértices, cada uma com seu valor de crease."""

    edges: np.ndarray
    crease: np.ndarray | None = None

    def __post_init__(self) -> None:
        edges = _normalize_pairs(self.edges)
        if self.crease is None:
            crease = np.full(len(edges), STRONG_VALUE)
        else:
            crease = np.asarray(self.crease, dtype=np.float64).reshape(-1)
            if len(crease) != len(edges):
                raise ValueError("edges e crease com tamanhos diferentes")
        keep = (edges[:, 0] != edges[:, 1]) & (crease > 0) if len(edges) else np.zeros(0, dtype=bool)
        edges, crease = edges[keep], crease[keep]
        if len(edges):
            base = int(edges.max()) + 1
            keys = edges[:, 0] * base + edges[:, 1]
            ukeys, inverse = np.unique(keys, return_inverse=True)
            merged = np.zeros(len(ukeys))
            np.maximum.at(merged, inverse.reshape(-1), crease)
            edges = np.stack([ukeys // base, ukeys % base], axis=1)
            crease = merged
        self.edges = edges.astype(np.int64).reshape(-1, 2)
        self.crease = crease.astype(np.float64)

    @classmethod
    def empty(cls) -> "FeatureSet":
        """Conjunto sem nenhuma aresta."""
        return cls(np.zeros((0, 2), dtype=np.int64), np.zeros(0))

    def __len__(self) -> int:
        return len(self.edges)

    def strong_mask(self, threshold: float = DEFAULT_STRONG_THRESHOLD) -> np.ndarray:
        """Arestas de crease forte (valor >= limiar)."""
        return self.crease >= threshold

    def guide_mask(self, threshold: float = DEFAULT_STRONG_THRESHOLD) -> np.ndarray:
        """Arestas de crease guia (0 < valor < limiar)."""
        return (self.crease > 0) & (self.crease < threshold)

    def merged(self, other: "FeatureSet") -> "FeatureSet":
        """União dos dois conjuntos; em aresta repetida vale o maior crease."""
        return FeatureSet(
            np.concatenate([self.edges, other.edges]), np.concatenate([self.crease, other.crease])
        )

    def remapped(self, new_index: np.ndarray) -> "FeatureSet":
        """Renumera os vértices (``new_index[antigo] = novo``; -1 descarta a aresta)."""
        new_index = np.asarray(new_index, dtype=np.int64)
        if len(self.edges) == 0:
            return FeatureSet.empty()
        mapped = new_index[self.edges]
        keep = (mapped >= 0).all(axis=1)
        return FeatureSet(mapped[keep], self.crease[keep])


def dihedral_angles(mesh: Mesh) -> np.ndarray:
    """Ângulo em graus entre as normais das duas faces de cada aresta.

    0 é superfície plana; arestas de borda, non-manifold ou com face degenerada
    ficam com NaN. Faces com orientação invertida entre si são detectadas pelo
    sentido das meias-arestas e corrigidas.
    """
    topo = mesh.topology
    normals, areas = face_normals_areas(mesh)
    out = np.full(topo.num_edges, np.nan)
    two = (topo.edge_face_count == 2) & ~topo.degenerate_edge
    if not two.any():
        return out
    f0 = topo.edge_faces[two, 0]
    f1 = topo.edge_faces[two, 1]
    cosine = np.einsum("ij,ij->i", normals[f0], normals[f1])
    c0 = topo.edge_corners[two, 0]
    c1 = topo.edge_corners[two, 1]
    flipped = mesh.connects[c0] == mesh.connects[c1]
    cosine = np.where(flipped, -cosine, cosine)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    angle[(areas[f0] <= 0) | (areas[f1] <= 0)] = np.nan
    out[two] = angle
    return out


def suggest_creases(
    mesh: Mesh, angle_deg: float = DEFAULT_SUGGEST_ANGLE, include_boundary: bool = False
) -> np.ndarray:
    """Pares de vértices das arestas com ângulo diedral acima de ``angle_deg``."""
    topo = mesh.topology
    angle = dihedral_angles(mesh)
    with np.errstate(invalid="ignore"):
        chosen = angle > angle_deg
    if include_boundary:
        chosen = chosen | topo.boundary_edge
    return topo.edges[chosen]


def boundary_edges(mesh: Mesh) -> np.ndarray:
    """Pares de vértices das arestas de borda aberta (viram feature sozinhas)."""
    topo = mesh.topology
    return topo.edges[topo.boundary_edge]


def feature_degree(num_points: int, edges: np.ndarray) -> np.ndarray:
    """Número de arestas de feature em cada vértice."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return np.bincount(edges.reshape(-1), minlength=num_points)


def feature_corners(
    points: np.ndarray, edges: np.ndarray, angle_deg: float = DEFAULT_CORNER_ANGLE
) -> np.ndarray:
    """Vértices de canto da rede de features.

    É canto o vértice onde chegam 1 ou 3+ arestas de feature (ponta ou junção)
    e o vértice de 2 arestas em que a curva vira mais que ``angle_deg``.
    """
    edges = _normalize_pairs(edges)
    if len(edges) == 0:
        return np.zeros(0, dtype=np.int64)
    points = np.asarray(points, dtype=np.float64)
    deg = feature_degree(len(points), edges)
    corner = (deg > 0) & (deg != 2)
    two = np.flatnonzero(deg == 2)
    if two.size:
        src = np.concatenate([edges[:, 0], edges[:, 1]])
        dst = np.concatenate([edges[:, 1], edges[:, 0]])
        order = np.argsort(src, kind="stable")
        src, dst = src[order], dst[order]
        start = np.searchsorted(src, two)
        before = points[two] - points[dst[start]]
        after = points[dst[start + 1]] - points[two]
        norm = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
        cosine = np.einsum("ij,ij->i", before, after) / np.where(norm > 0, norm, 1.0)
        turn = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        corner[two[(turn > angle_deg) & (norm > 0)]] = True
    return np.flatnonzero(corner).astype(np.int64)


def feature_chains(edges: np.ndarray, stops: np.ndarray | None = None) -> list[np.ndarray]:
    """Parte a rede de features em cadeias de vértices.

    Cada cadeia vai de um nó (vértice de grau diferente de 2, ou listado em
    ``stops``) até o próximo; laços fechados sem nó saem com o primeiro vértice
    repetido no fim.
    """
    edges = _normalize_pairs(edges)
    if len(edges) == 0:
        return []
    neighbors: dict[int, list[tuple[int, int]]] = {}
    for i, (a, b) in enumerate(edges.tolist()):
        neighbors.setdefault(a, []).append((b, i))
        neighbors.setdefault(b, []).append((a, i))
    stop_set = set(np.asarray(stops, dtype=np.int64).tolist()) if stops is not None else set()
    nodes = {v for v, nb in neighbors.items() if len(nb) != 2} | (stop_set & set(neighbors))
    used = np.zeros(len(edges), dtype=bool)
    chains: list[np.ndarray] = []

    def walk(start: int, nxt: int, edge: int) -> list[int]:
        path = [start, nxt]
        used[edge] = True
        cur = nxt
        while cur not in nodes and cur != start:
            step = [(v, e) for v, e in neighbors[cur] if not used[e]]
            if not step:
                break
            cur, e = step[0]
            used[e] = True
            path.append(cur)
        return path

    for node in sorted(nodes):
        for nxt, e in neighbors[node]:
            if not used[e]:
                chains.append(np.asarray(walk(node, nxt, e), dtype=np.int64))
    for i in np.flatnonzero(~used).tolist():
        if not used[i]:
            a, b = edges[i].tolist()
            chains.append(np.asarray(walk(a, b, i), dtype=np.int64))
    return chains


@dataclass
class InputCheck:
    """Resultado do "Verificar entrada" (seção 6 do plano)."""

    dangling_vertices: np.ndarray
    dangling_chains: list[np.ndarray]
    marked_ratio: float
    marked_ratio_excessive: bool
    unmarked_sharp_edges: np.ndarray
    not_mesh_edges: np.ndarray
    num_strong: int
    num_guide: int
    messages: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Verdadeiro quando nenhum dos três problemas apareceu."""
        return not (
            len(self.dangling_vertices)
            or self.marked_ratio_excessive
            or len(self.unmarked_sharp_edges)
            or len(self.not_mesh_edges)
        )


def check_input(
    mesh: Mesh,
    features: FeatureSet,
    angle_deg: float = DEFAULT_SUGGEST_ANGLE,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
    max_marked_ratio: float = 0.25,
    min_edges_for_ratio: int = 200,
) -> InputCheck:
    """Aponta os problemas de crease antes de rodar o motor.

    1. Cadeias de crease que terminam no meio de uma superfície (sem fechar,
       sem chegar a outra crease e sem chegar a uma borda aberta).
    2. Proporção excessiva de arestas marcadas.
    3. Quinas vivas acima do ângulo de sugestão que ficaram sem crease.
    """
    topo = mesh.topology
    ids = topo.edge_ids(features.edges)
    on_mesh = ids >= 0
    not_mesh = features.edges[~on_mesh]
    feat_edges = features.edges[on_mesh]

    network = np.concatenate([feat_edges, topo.edges[topo.boundary_edge]])
    network = np.unique(_normalize_pairs(network), axis=0) if len(network) else network
    deg = feature_degree(mesh.num_points, network)
    feat_vertex = np.zeros(mesh.num_points, dtype=bool)
    feat_vertex[feat_edges.reshape(-1)] = True
    dangling = np.flatnonzero((deg == 1) & feat_vertex)
    dangling_set = set(dangling.tolist())
    chains = [
        c for c in feature_chains(network) if int(c[0]) in dangling_set or int(c[-1]) in dangling_set
    ]

    real_edges = int((~topo.degenerate_edge).sum())
    ratio = float(len(feat_edges)) / real_edges if real_edges else 0.0
    excessive = real_edges >= min_edges_for_ratio and ratio > max_marked_ratio

    marked = np.zeros(topo.num_edges, dtype=bool)
    marked[ids[on_mesh]] = True
    with np.errstate(invalid="ignore"):
        sharp = dihedral_angles(mesh) > angle_deg
    unmarked = topo.edges[sharp & ~marked]

    strong = features.strong_mask(strong_threshold)
    messages: list[str] = []
    if len(not_mesh):
        messages.append(
            f"{len(not_mesh)} aresta(s) de feature não são arestas reais da malha e serão ignoradas."
        )
    if len(dangling):
        messages.append(
            f"{len(dangling)} ponta(s) de crease terminam no meio de uma superfície "
            "(a cadeia não fecha, não chega a outra crease nem a uma borda)."
        )
    if excessive:
        messages.append(
            f"{ratio:.0%} das arestas estão marcadas (limite {max_marked_ratio:.0%}); "
            "confira se o ângulo de sugestão não ficou baixo demais."
        )
    if len(unmarked):
        messages.append(
            f"{len(unmarked)} aresta(s) com quina acima de {angle_deg:g}° estão sem crease."
        )
    if not messages:
        messages.append("Entrada sem problemas de crease.")
    return InputCheck(
        dangling_vertices=dangling,
        dangling_chains=chains,
        marked_ratio=ratio,
        marked_ratio_excessive=bool(excessive),
        unmarked_sharp_edges=unmarked,
        not_mesh_edges=not_mesh,
        num_strong=int(strong.sum()),
        num_guide=int((~strong).sum()),
        messages=messages,
    )


# --------------------------------------------------------------------------
# Camada Maya (só roda dentro do Maya / mayapy)
# --------------------------------------------------------------------------


def maya_read_creases(mesh_name: str) -> FeatureSet:
    """Lê o crease por aresta de uma malha do Maya como ``FeatureSet``."""
    from .. import maya_bridge

    fn = maya_bridge.mesh_fn(mesh_name)
    try:
        edge_ids, values = fn.getCreaseEdges()
    except RuntimeError:  # algumas versões falham (kFailure) em vez de devolver listas vazias
        return FeatureSet.empty()
    ids = np.asarray(list(edge_ids), dtype=np.int64)
    values = np.asarray(list(values), dtype=np.float64)
    keep = values > 0  # crease 0 gravado à mão não é feature
    if not keep.any():
        return FeatureSet.empty()
    return FeatureSet(maya_bridge.edge_vertex_pairs(mesh_name, ids[keep]), values[keep])


def maya_read_hard_edges(mesh_name: str) -> np.ndarray:
    """Pares de vértices das hard edges já existentes (ponto de partida da sugestão)."""
    from .. import maya_bridge

    fn = maya_bridge.mesh_fn(mesh_name)
    pairs = maya_bridge.edge_vertex_pairs(mesh_name)
    hard = np.fromiter(
        (not fn.isEdgeSmooth(i) for i in range(fn.numEdges)), dtype=bool, count=fn.numEdges
    )
    return pairs[hard]


def maya_suggest(mesh_name: str, angle_deg: float = DEFAULT_SUGGEST_ANGLE, select: bool = True) -> np.ndarray:
    """Sugere creases por ângulo e (opcionalmente) seleciona as arestas na viewport.

    Nada é gravado: o usuário ajusta a seleção e confirma com ``maya_apply``.
    """
    from .. import maya_bridge

    mesh = maya_bridge.mesh_from_maya(mesh_name)
    pairs = suggest_creases(mesh, angle_deg)
    if select:
        maya_bridge.select_edges(mesh_name, pairs)
    return pairs


def maya_apply(mesh_name: str, pairs: np.ndarray, value: float = STRONG_VALUE) -> int:
    """Grava ``value`` como crease nas arestas dadas (um passo de desfazer)."""
    from maya import cmds

    from .. import maya_bridge

    components = maya_bridge.edge_components(mesh_name, pairs)
    if not components:
        return 0
    with maya_bridge.undo_chunk("TopTop: gravar creases"):
        cmds.polyCrease(components, value=float(value))
    return len(pairs)


def maya_apply_selection(value: float = STRONG_VALUE) -> int:
    """Grava ``value`` como crease nas arestas selecionadas no momento."""
    from maya import cmds

    edges = cmds.filterExpand(selectionMask=32) or []
    if not edges:
        return 0
    cmds.undoInfo(openChunk=True, chunkName="TopTop: gravar creases")
    try:
        cmds.polyCrease(edges, value=float(value))
    finally:
        cmds.undoInfo(closeChunk=True)
    return len(edges)


def maya_check(
    mesh_name: str,
    angle_deg: float = DEFAULT_SUGGEST_ANGLE,
    strong_threshold: float = DEFAULT_STRONG_THRESHOLD,
) -> InputCheck:
    """ "Verificar entrada" direto numa malha do Maya."""
    from .. import maya_bridge

    mesh = maya_bridge.mesh_from_maya(mesh_name)
    return check_input(mesh, maya_read_creases(mesh_name), angle_deg, strong_threshold)
