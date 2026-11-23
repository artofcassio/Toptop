"""Ponte entre as malhas do Maya e os arrays do TopTop.

Este módulo importa ``maya`` no topo: só pode ser importado dentro do Maya ou
do ``mayapy``. O resto do pacote o importa dentro das funções.

Regra de desfazer: os nós são sempre criados por ``cmds`` (o transform) e só a
geometria entra pela API, debaixo desse transform. Assim um único chunk de
undo (``undo_chunk``) remove tudo o que foi criado. Limitação conhecida:
refazer (redo) recria o transform sem a geometria.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import maya.api.OpenMaya as om
import numpy as np
from maya import cmds

from .core.mesh import Mesh


@contextmanager
def undo_chunk(name: str = "TopTop") -> Iterator[None]:
    """Agrupa tudo o que acontecer dentro num único passo de desfazer."""
    cmds.undoInfo(openChunk=True, chunkName=name)
    try:
        yield
    finally:
        cmds.undoInfo(closeChunk=True)


def is_batch() -> bool:
    """Verdadeiro no mayapy / Maya sem interface."""
    return bool(cmds.about(batch=True))


def shape_of(name: str) -> str:
    """Caminho completo do shape de malha de um transform (ou do próprio shape)."""
    if not cmds.objExists(name):
        raise ValueError(f"objeto não encontrado: {name}")
    if cmds.nodeType(name) == "mesh":
        return cmds.ls(name, long=True)[0]
    shapes = cmds.listRelatives(name, shapes=True, noIntermediate=True, type="mesh", fullPath=True) or []
    if not shapes:
        raise ValueError(f"{name} não é uma malha poligonal")
    return shapes[0]


def transform_of(name: str) -> str:
    """Caminho completo do transform de uma malha (dado o transform ou o shape)."""
    shape = shape_of(name)
    return cmds.listRelatives(shape, parent=True, fullPath=True)[0]


def mesh_fn(name: str) -> om.MFnMesh:
    """``MFnMesh`` com caminho DAG (necessário para ler em espaço de mundo)."""
    selection = om.MSelectionList()
    selection.add(shape_of(name))
    return om.MFnMesh(selection.getDagPath(0))


def mesh_from_maya(name: str, world: bool = True) -> Mesh:
    """Lê vértices e faces de uma malha do Maya (em espaço de mundo por padrão)."""
    fn = mesh_fn(name)
    space = om.MSpace.kWorld if world else om.MSpace.kObject
    points = np.array(fn.getPoints(space), dtype=np.float64).reshape(-1, 4)[:, :3]
    counts, connects = fn.getVertices()
    short = name.split("|")[-1]
    return Mesh(points, np.array(counts, dtype=np.int64), np.array(connects, dtype=np.int64), short)


def edge_vertex_pairs(name: str, ids=None) -> np.ndarray:
    """(E, 2) vértices de cada aresta, na numeração de arestas do Maya.

    Com ``ids``, só dessas arestas (bem mais rápido numa malha grande, já que
    a API responde uma aresta por chamada).
    """
    fn = mesh_fn(name)
    wanted = range(fn.numEdges) if ids is None else [int(i) for i in ids]
    pairs = np.empty((len(wanted), 2), dtype=np.int64)
    get = fn.getEdgeVertices
    for row, i in enumerate(wanted):
        pairs[row] = get(i)
    return pairs


def edge_ids(name: str, pairs: np.ndarray) -> np.ndarray:
    """Índice de aresta do Maya para cada par de vértices (-1 se não é aresta)."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) == 0:
        return np.zeros(0, dtype=np.int64)
    all_pairs = edge_vertex_pairs(name)
    if len(all_pairs) == 0:
        return np.full(len(pairs), -1, dtype=np.int64)
    base = int(max(all_pairs.max(), pairs.max())) + 1
    keys = all_pairs.min(axis=1) * base + all_pairs.max(axis=1)
    order = np.argsort(keys)
    wanted = pairs.min(axis=1) * base + pairs.max(axis=1)
    pos = np.minimum(np.searchsorted(keys[order], wanted), len(keys) - 1)
    return np.where(keys[order][pos] == wanted, order[pos], -1).astype(np.int64)


def edge_components(name: str, pairs: np.ndarray) -> list[str]:
    """Componentes de aresta (``shape.e[a:b]``) dos pares dados, em faixas compactas."""
    ids = edge_ids(name, pairs)
    ids = np.unique(ids[ids >= 0])
    if ids.size == 0:
        return []
    shape = shape_of(name)
    breaks = np.flatnonzero(np.diff(ids) != 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [ids.size - 1]])
    return [f"{shape}.e[{ids[a]}:{ids[b]}]" for a, b in zip(starts.tolist(), ends.tolist())]


def select_edges(name: str, pairs: np.ndarray) -> int:
    """Seleciona na viewport as arestas dadas; devolve quantas faixas selecionou."""
    components = edge_components(name, pairs)
    if components:
        cmds.select(components, replace=True)
    else:
        cmds.select(clear=True)
    return len(components)


def create_mesh(mesh: Mesh, name: str = "toptop_mesh") -> str:
    """Cria um objeto novo no Maya com a geometria de ``mesh``; devolve o transform.

    Os vértices e as faces ficam na mesma ordem dos arrays: hard edges,
    creases e cores gravadas depois usam esses índices. Por isso a malha não
    pode ter vértice sem face (o Maya o descartaria e a numeração andaria);
    se o Maya mudar a contagem mesmo assim, o objeto é apagado e o erro sobe.
    """
    transform = cmds.createNode("transform", name=name, skipSelect=True)
    selection = om.MSelectionList()
    selection.add(transform)
    parent = selection.getDependNode(0)
    points = om.MPointArray([om.MPoint(x, y, z) for x, y, z in mesh.points.tolist()])
    try:
        fn = om.MFnMesh()
        fn.create(points, mesh.counts.tolist(), mesh.connects.tolist(), parent=parent)
        if fn.numVertices != mesh.num_points or fn.numPolygons != mesh.num_faces:
            raise RuntimeError(
                f"o Maya criou {fn.numVertices} vértices e {fn.numPolygons} faces, mas a malha tem "
                f"{mesh.num_points} e {mesh.num_faces}: a numeração mudou (vértice sem face ou face inválida)"
            )
    except Exception:
        cmds.delete(transform)
        raise
    transform = cmds.ls(transform, long=True)[0]
    shape = cmds.listRelatives(transform, shapes=True, fullPath=True)[0]
    cmds.rename(shape, transform.split("|")[-1] + "Shape")
    cmds.sets(transform, edit=True, forceElement="initialShadingGroup")
    return transform


def set_hard_edges(name: str, pairs: np.ndarray) -> None:
    """Deixa duras só as arestas dadas; todas as outras ficam suaves."""
    shape = shape_of(name)
    cmds.polySoftEdge(shape, angle=180, constructionHistory=False)
    components = edge_components(name, pairs)
    if components:
        cmds.polySoftEdge(components, angle=0, constructionHistory=False)


def copy_material(source: str, target: str) -> str:
    """Aplica em ``target`` o primeiro material de ``source``; devolve o shading group."""
    groups = cmds.listConnections(shape_of(source), type="shadingEngine") or []
    group = groups[0] if groups else "initialShadingGroup"
    cmds.sets(target, edit=True, forceElement=group)
    return group


def selected_mesh() -> str | None:
    """Transform da primeira malha selecionada (aceita seleção de componentes)."""
    for node in cmds.ls(selection=True, long=True, objectsOnly=True) or []:
        try:
            return transform_of(node)
        except ValueError:
            continue
    return None


def set_vertex_colors(name: str, colors: np.ndarray, color_set: str = "toptopDesvio") -> None:
    """Pinta os vértices (N, 3 em 0-1) e liga a exibição de cor: o mapa de calor."""
    shape = shape_of(name)
    if color_set not in (cmds.polyColorSet(shape, query=True, allColorSets=True) or []):
        cmds.polyColorSet(shape, create=True, colorSet=color_set, representation="RGB")
    cmds.polyColorSet(shape, currentColorSet=True, colorSet=color_set)
    fn = mesh_fn(name)
    array = om.MColorArray([om.MColor((float(r), float(g), float(b), 1.0)) for r, g, b in np.asarray(colors).tolist()])
    fn.setVertexColors(array, list(range(len(array))))
    cmds.setAttr(shape + ".displayColors", 1)
