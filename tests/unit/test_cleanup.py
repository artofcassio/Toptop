"""Testes de toptop.core.cleanup (solda, faces ruins, non-manifold, cascas, redução)."""

import sys
import time
import types

import numpy as np
import pytest

import shapes
from toptop.core.cleanup import (
    CleanResult,
    Shell,
    clean_mesh,
    close_pairs,
    compact_vertices,
    fan_count,
    maya_reduce,
    nonmanifold_report,
    num_triangles,
    reduction_target,
    remove_degenerate_faces,
    split_shells,
    weld_vertices,
)
from toptop.core.creases import FeatureSet
from toptop.core.mesh import Mesh


def unwelded(mesh: Mesh) -> Mesh:
    """Cópia em que cada canto de cada face tem o seu próprio vértice."""
    return Mesh(mesh.points[mesh.connects], mesh.counts, np.arange(mesh.connects.size), mesh.name)


def pair_set(edges: np.ndarray) -> set[tuple[int, int]]:
    """Pares como conjunto de tuplas ordenadas."""
    return {(min(a, b), max(a, b)) for a, b in np.asarray(edges).reshape(-1, 2).tolist()}


def snapshot(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cópia dos arrays para conferir que nada foi alterado no lugar."""
    return mesh.points.copy(), mesh.counts.copy(), mesh.connects.copy()


def assert_unchanged(mesh: Mesh, snap: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Os arrays da malha continuam iguais à cópia."""
    assert np.array_equal(mesh.points, snap[0])
    assert np.array_equal(mesh.counts, snap[1])
    assert np.array_equal(mesh.connects, snap[2])


# --------------------------------------------------------------------------
# Solda
# --------------------------------------------------------------------------


def test_weld_unwelded_cube_becomes_closed_manifold():
    cube = shapes.cube(1)
    raw = unwelded(cube)
    snap = snapshot(raw)
    assert raw.num_points == 24 and raw.topology.boundary_edge.all()
    welded, vmap = weld_vertices(raw)
    assert_unchanged(raw, snap)
    assert welded.num_points == 8 and welded.num_faces == 6
    topo = welded.topology
    assert topo.num_edges == 12
    assert not topo.boundary_edge.any() and not topo.nonmanifold_edge.any()
    assert (topo.vertex_valence == 3).all()
    # cada vértice novo está na posição dos antigos que ele juntou
    assert vmap.shape == (24,) and (vmap >= 0).all()
    assert np.allclose(welded.points[vmap], raw.points)
    # ordem de primeira ocorrência
    first = [int(vmap[i]) for i in range(24)]
    seen = []
    for v in first:
        if v not in seen:
            seen.append(v)
    assert seen == list(range(8))


def test_weld_keeps_numbering_of_clean_mesh_and_drops_unused():
    cube = shapes.cube(2)
    points = np.concatenate([cube.points, [[5.0, 5.0, 5.0]]])  # vértice solto no fim
    mesh = Mesh(points, cube.counts, cube.connects, "cubo")
    welded, vmap = weld_vertices(mesh)
    assert welded.num_points == cube.num_points
    assert vmap[:-1].tolist() == list(range(cube.num_points)) and vmap[-1] == -1
    assert np.array_equal(welded.connects, cube.connects)
    assert np.array_equal(welded.points, cube.points)
    assert welded.name == "cubo"


def test_weld_unused_vertex_in_the_middle():
    points = [[0, 0, 0], [9, 9, 9], [1, 0, 0], [0, 1, 0]]
    mesh = Mesh.from_faces(points, [(0, 2, 3)])
    welded, vmap = weld_vertices(mesh)
    assert vmap.tolist() == [0, -1, 1, 2]
    assert welded.faces() == [(0, 1, 2)]


def two_triangles(offset: float) -> Mesh:
    """Dois triângulos lado a lado; a aresta comum duplicada e deslocada de ``offset`` em z."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, offset],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, offset],
        ]
    )
    return Mesh.from_faces(points, [(0, 1, 2), (3, 4, 5)])


@pytest.mark.parametrize("factor, expected", [(0.0, 4), (0.5, 4), (0.99, 4), (1.5, 6), (3.0, 6)])
def test_weld_tolerance(factor, expected):
    tol_rel = 1e-4
    mesh = two_triangles(0.0)
    tol = tol_rel * mesh.diagonal()
    mesh = two_triangles(factor * tol)
    welded, vmap = weld_vertices(mesh, tol_rel)
    assert welded.num_points == expected
    if expected == 4:
        assert vmap[3] == vmap[1] and vmap[5] == vmap[2]
        assert int(welded.topology.boundary_edge.sum()) == 4


def test_weld_is_transitive_chain():
    tol_rel = 1e-3
    base = shapes.grid_plane(1, 1)
    tol = tol_rel * base.diagonal()
    # cadeia de 3 cópias do canto 0, cada uma a 0,8 tol da anterior (pontas a 1,6 tol)
    extra = np.array([[0.8 * tol, 0, 0], [1.6 * tol, 0, 0]])
    points = np.concatenate([base.points, extra])
    mesh = Mesh(points, [4, 3, 3], np.concatenate([base.connects, [4, 1, 2], [5, 1, 3]]))
    welded, vmap = weld_vertices(mesh, tol_rel)
    assert vmap[4] == vmap[0] == vmap[5]
    assert welded.num_points == 4


def test_weld_pairs_straddling_cell_boundaries():
    # par que atravessa a divisa de célula nos três eixos ao mesmo tempo
    tol_rel = 1e-3
    corners = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    tol = tol_rel * np.sqrt(3.0)
    edge = 100 * tol * (1.0 + 1e-6)
    pair = np.array([[edge - 0.2 * tol] * 3, [edge + 0.2 * tol] * 3])
    pts = np.concatenate([corners, pair, [[0.5, 0.2, 0.9]]])
    mesh = Mesh.from_faces(pts, [(0, 1, 2), (3, 2, 5), (4, 5, 1)])
    welded, vmap = weld_vertices(mesh, tol_rel)
    assert vmap[3] == vmap[4]
    assert welded.num_points == 5

    # aleatório: milhares de pares em posições quaisquer, cada um a 0,9 tol
    rng = np.random.default_rng(7)
    grid = np.stack(np.meshgrid(*[np.arange(20)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
    base = grid * 0.05 + rng.uniform(-0.01, 0.01, size=grid.shape)
    lo, hi = base.min(axis=0), base.max(axis=0)
    tol_rel = 1e-5
    tol = tol_rel * np.linalg.norm(hi - lo)
    direction = rng.normal(size=base.shape)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    copies = base + 0.9 * tol * direction
    # as cópias não podem aumentar a caixa (a tolerância depende da diagonal)
    inside = ((copies >= lo) & (copies <= hi)).all(axis=1)
    m = len(base)
    tri = np.arange(m - m % 3).reshape(-1, 3)
    faces = np.concatenate([tri, tri + m])
    mesh = Mesh(np.concatenate([base, copies]), np.full(len(faces), 3), faces.reshape(-1))
    welded, vmap = weld_vertices(mesh, tol_rel)
    used = np.zeros(2 * m, dtype=bool)
    used[faces.reshape(-1)] = True
    ids = np.flatnonzero(used[:m] & inside)
    assert np.array_equal(vmap[ids], vmap[ids + m])
    assert welded.num_points == int(used[:m].sum())


def test_weld_outside_tolerance_random_pairs_stay_apart():
    rng = np.random.default_rng(3)
    base = rng.uniform(0, 1, size=(3000, 3))
    tol_rel = 1e-6
    tol = tol_rel * np.linalg.norm(base.max(axis=0) - base.min(axis=0))
    direction = rng.normal(size=base.shape)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    copies = base + 1.5 * tol * direction
    faces = np.concatenate([np.arange(3000).reshape(-1, 3), np.arange(3000, 6000).reshape(-1, 3)])
    mesh = Mesh(np.concatenate([base, copies]), np.full(len(faces), 3), faces.reshape(-1))
    welded, _ = weld_vertices(mesh, tol_rel)
    assert welded.num_points == 6000


def test_weld_exact_duplicates_with_zero_tolerance():
    raw = unwelded(shapes.cube(1))
    welded, _ = weld_vertices(raw, 0.0)
    assert welded.num_points == 8


def test_close_pairs_matches_brute_force():
    rng = np.random.default_rng(11)
    pts = rng.uniform(0, 1, size=(400, 3))
    pts = np.concatenate([pts, pts[:50] + rng.normal(scale=0.01, size=(50, 3))])
    tol = 0.03
    got = pair_set(close_pairs(pts, tol))
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    i, j = np.nonzero(np.triu(dist <= tol, k=1))
    assert got == pair_set(np.stack([i, j], axis=1))
    assert close_pairs(pts[:1], tol).shape == (0, 2)


def test_weld_empty_mesh():
    mesh = Mesh(np.zeros((3, 3)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    welded, vmap = weld_vertices(mesh)
    assert welded.num_points == 0 and welded.num_faces == 0
    assert vmap.tolist() == [-1, -1, -1]


def test_weld_speed_300k_vertices():
    plane = shapes.grid_plane(270, 270, 10.0, 10.0)
    raw = unwelded(plane)
    assert raw.num_points == 291_600
    rng = np.random.default_rng(5)
    tol = 1e-6 * raw.diagonal()
    jitter = rng.normal(size=raw.points.shape)
    jitter *= (0.3 * tol / np.linalg.norm(jitter, axis=1))[:, None]
    raw = raw.with_points(raw.points + jitter)
    start = time.perf_counter()
    welded, _ = weld_vertices(raw)
    elapsed = time.perf_counter() - start
    assert welded.num_points == 271 * 271
    assert elapsed < 10.0
    assert not welded.topology.nonmanifold_edge.any()


# --------------------------------------------------------------------------
# Faces degeneradas e duplicadas
# --------------------------------------------------------------------------


def test_remove_degenerate_and_duplicate_faces():
    points = [
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 2
        [0, 1, 0],  # 3
        [2, 0, 0],  # 4 (colinear com 0 e 1)
        [2, 1, 0],  # 5
    ]
    faces = [
        (0, 1, 2, 3),  # 0 boa
        (1, 4, 4, 5, 2),  # 1 repetição em sequência: vira o quad (1, 4, 5, 2)
        (0, 0, 1),  # 2 só 2 vértices distintos
        (0, 1, 0, 1),  # 3 só 2 vértices distintos fora de sequência
        (0, 1, 4),  # 4 área zero (colinear)
        (2, 1, 0, 3),  # 5 a face 0 rodada e invertida
        (3, 0, 1, 2),  # 6 a face 0 rodada
        (1, 4, 5, 2),  # 7 igual à face 1 depois de encurtada
        (3, 2, 2, 3),  # 8 colapsa
        (1, 5, 2),  # 9 boa (mesmos vértices da face 1, mas outro número de lados)
        (0, 1, 2, 3, 0),  # 10 último igual ao primeiro: vira a face 0 (duplicada)
    ]
    mesh = Mesh.from_faces(points, faces)
    snap = snapshot(mesh)
    out, keep = remove_degenerate_faces(mesh)
    assert_unchanged(mesh, snap)
    assert keep.tolist() == [True, True, False, False, False, False, False, False, False, True, False]
    assert out.faces() == [(0, 1, 2, 3), (1, 4, 5, 2), (1, 5, 2)]
    assert out.num_points == mesh.num_points


def test_remove_degenerate_tiny_area_threshold():
    eps = 1e-9
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, eps, 0]]
    mesh = Mesh.from_faces(points, [(0, 1, 2), (0, 1, 3)])
    # área 0,5e-9 > 1e-12 * diag^2: fica
    _, keep = remove_degenerate_faces(mesh)
    assert keep.tolist() == [True, True]
    _, keep = remove_degenerate_faces(mesh, area_tol_rel=1e-6)
    assert keep.tolist() == [True, False]


def test_remove_degenerate_empty_and_clean():
    cube = shapes.cube(2)
    out, keep = remove_degenerate_faces(cube)
    assert keep.all() and np.array_equal(out.connects, cube.connects)
    empty = Mesh(np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    out, keep = remove_degenerate_faces(empty)
    assert out.num_faces == 0 and keep.size == 0


# --------------------------------------------------------------------------
# Non-manifold
# --------------------------------------------------------------------------


def test_bowtie_vertex():
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
    mesh = Mesh.from_faces(points, [(0, 1, 2), (0, 3, 4)])
    edges, verts = nonmanifold_report(mesh)
    assert edges.shape == (0, 2)
    assert verts.tolist() == [0]
    assert fan_count(mesh).tolist() == [2, 1, 1, 1, 1]


def test_two_closed_cones_touching_at_apex():
    # duas pirâmides fechadas que só se tocam no ápice (vértice 0)
    points = [[0, 0, 0]]
    faces = []
    for sign in (1.0, -1.0):
        base = len(points)
        points += [[1, sign, 1], [-1, sign, 1], [-1, sign, -1], [1, sign, -1]]
        ring = [base, base + 1, base + 2, base + 3]
        for k in range(4):
            a, b = ring[k], ring[(k + 1) % 4]
            faces.append((0, a, b) if sign > 0 else (0, b, a))
        faces.append(tuple(reversed(ring)) if sign > 0 else tuple(ring))
    mesh = Mesh.from_faces(points, faces)
    assert not mesh.topology.boundary_edge.any()
    edges, verts = nonmanifold_report(mesh)
    assert len(edges) == 0 and verts.tolist() == [0]


def test_edge_with_three_faces():
    points = [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0], [0.5, 0, 1]]
    mesh = Mesh.from_faces(points, [(0, 1, 2), (1, 0, 3), (0, 1, 4)])
    edges, verts = nonmanifold_report(mesh)
    assert edges.tolist() == [[0, 1]]
    assert verts.size == 0  # as faces se ligam pela própria aresta
    result = clean_mesh(mesh)
    assert not result.ok
    assert result.nonmanifold_edges.tolist() == [[0, 1]]
    assert any("non-manifold" in m for m in result.messages)


def test_manifold_shapes_have_no_reports():
    for mesh in (shapes.cube(3), shapes.torus(), shapes.grid_plane(3, 2), shapes.cylinder(8, 2, caps="fan")):
        edges, verts = nonmanifold_report(mesh)
        assert len(edges) == 0 and len(verts) == 0
    # vértice de borda de um plano: um leque só
    assert (fan_count(shapes.grid_plane(2, 2)) == 1).all()


def test_flipped_neighbor_still_single_fan():
    plane = shapes.grid_plane(2, 2)
    faces = plane.faces()
    faces[0] = tuple(reversed(faces[0]))
    mesh = Mesh.from_faces(plane.points, faces)
    assert (fan_count(mesh) == 1).all()


# --------------------------------------------------------------------------
# Limpeza completa
# --------------------------------------------------------------------------


def test_clean_mesh_unwelded_cube_with_features():
    cube = shapes.cube(2)
    raw = unwelded(cube)
    # features dadas na numeração da malha suja: cada aresta do cubo aparece
    # duas vezes (uma em cada face), com valores diferentes
    clean_feat = shapes.cube_feature_edges(cube)
    corner_of = {}
    for c, v in enumerate(cube.connects.tolist()):
        corner_of.setdefault(v, []).append(c)
    pairs, values = [], []
    for a, b in clean_feat.tolist():
        for ca in corner_of[a]:
            for cb in corner_of[b]:
                if raw.topology.edge_ids(np.array([[ca, cb]]))[0] >= 0:
                    pairs.append((ca, cb))
                    values.append(3.0 if len(pairs) % 2 else 1.5)
    feats = FeatureSet(np.array(pairs), np.array(values))
    snap = snapshot(raw)
    result = clean_mesh(raw, feats, triangulate=False)
    assert_unchanged(raw, snap)
    assert isinstance(result, CleanResult)
    assert result.ok
    assert result.mesh.num_points == cube.num_points and result.mesh.num_faces == 24
    assert result.welded_vertices == raw.num_points - cube.num_points
    assert result.removed_faces == 0
    assert len(result.boundary_edges) == 0
    expected = shapes.cube_feature_edges(result.mesh)
    assert pair_set(result.features.edges) == pair_set(expected)
    assert (result.features.crease == 3.0).all()  # vale o maior dos valores repetidos
    assert any("soldado" in m for m in result.messages)
    assert np.array_equal(result.face_origin, np.arange(24))

    tri = clean_mesh(raw, feats)
    assert (tri.mesh.counts == 3).all() and tri.mesh.num_faces == 48
    assert pair_set(tri.features.edges) == pair_set(expected)
    assert np.array_equal(np.bincount(tri.face_origin), np.full(24, 2))
    assert tri.ok


def test_clean_mesh_feature_edges_lost_in_weld_or_not_edges():
    tol_rel = 1e-4
    mesh = two_triangles(0.0)
    tol = tol_rel * mesh.diagonal()
    mesh = two_triangles(0.3 * tol)
    feats = FeatureSet(
        np.array([[1, 3], [0, 4], [1, 2], [3, 5], [0, 99]]), np.array([3.0, 3.0, 1.0, 3.0, 3.0])
    )
    result = clean_mesh(mesh, feats, weld_tol_rel=tol_rel, triangulate=False, border_value=0.0)
    # [1,3] soldou num ponto só; [0,4] é a diagonal (não é aresta); [0,99] não existe
    assert pair_set(result.features.edges) == {(1, 2)}
    assert result.features.crease.tolist() == [3.0]  # [1,2] e [3,5] viraram a mesma aresta
    assert any("sumiram na solda" in m for m in result.messages)
    assert any("2 aresta(s) de feature não são arestas" in m for m in result.messages)


def test_clean_mesh_open_cylinder_borders_become_guides():
    cyl = shapes.cylinder(12, 2)
    strong = np.array([[0, 1], [12, 13]])  # uma aresta de borda e uma do anel do meio
    result = clean_mesh(cyl, FeatureSet(strong, np.array([3.0, 3.0])))
    assert result.ok
    assert len(result.boundary_edges) == 24
    assert pair_set(result.boundary_edges) == pair_set(cyl.topology.edges[cyl.topology.boundary_edge])
    feats = result.features
    assert len(feats) == 25
    by_pair = dict(zip(map(tuple, feats.edges.tolist()), feats.crease.tolist()))
    assert by_pair[(0, 1)] == 3.0 and by_pair[(12, 13)] == 3.0
    border_values = [by_pair[p] for p in pair_set(result.boundary_edges) if p != (0, 1)]
    assert border_values == [1.0] * 23
    assert int(feats.guide_mask().sum()) == 23
    assert any("borda aberta" in m for m in result.messages)

    custom = clean_mesh(cyl, border_value=0.5)
    assert np.allclose(custom.features.crease, 0.5) and len(custom.features) == 24
    off = clean_mesh(cyl, border_value=0.0)
    assert len(off.features) == 0 and len(off.boundary_edges) == 24


def test_clean_mesh_removes_bad_faces_and_compacts():
    cube = shapes.cube(1)
    points = np.concatenate([cube.points, [[7.0, 7.0, 7.0]]])  # vértice 8 só em face ruim
    faces = cube.faces() + [(0, 8, 8), cube.faces()[2][::-1]]
    mesh = Mesh.from_faces(points, faces, "sujo")
    result = clean_mesh(mesh, triangulate=False)
    assert result.removed_faces == 2
    assert result.mesh.num_points == 8 and result.mesh.num_faces == 6
    assert result.vertex_map[8] == -1
    assert result.ok and len(result.boundary_edges) == 0
    assert result.mesh.name == "sujo"
    text = "\n".join(result.messages)
    assert "menos de 3 vértices" in text and "duplicada" in text


def test_clean_mesh_already_clean():
    torus = shapes.torus()
    result = clean_mesh(torus, triangulate=False)
    assert result.messages == ["Malha limpa: nada a corrigir."]
    assert np.array_equal(result.mesh.connects, torus.connects)
    assert result.vertex_map.tolist() == list(range(torus.num_points))
    assert result.mesh.points is not torus.points


# --------------------------------------------------------------------------
# Cascas
# --------------------------------------------------------------------------


def test_split_shells_from_combine():
    cube = shapes.cube(2)
    torus = shapes.translated(shapes.torus(), (5.0, 0.0, 0.0))
    plane = shapes.translated(shapes.grid_plane(2, 2), (0.0, 5.0, 0.0))
    mesh = shapes.combine(cube, torus, plane)
    cube_feats = shapes.cube_feature_edges(cube)
    torus_feat = np.array([[0, 1], [1, 2]]) + cube.num_points
    feats = FeatureSet(
        np.concatenate([cube_feats, torus_feat]), np.concatenate([np.full(len(cube_feats), 3.0), [1.0, 1.0]])
    )
    out = split_shells(mesh, feats)
    assert [type(s) for s in out] == [Shell] * 3
    assert [s.mesh.num_faces for s in out] == [torus.num_faces, cube.num_faces, plane.num_faces]
    for shell in out:
        # vértices e faces apontam de volta para a malha de entrada
        assert np.array_equal(shell.mesh.points, mesh.points[shell.vertex_ids])
        all_faces = mesh.faces()
        back = [tuple(shell.vertex_ids[list(f)].tolist()) for f in shell.mesh.faces()]
        assert back == [all_faces[i] for i in shell.face_ids.tolist()]
        assert np.all(np.diff(shell.vertex_ids) > 0) and np.all(np.diff(shell.face_ids) > 0)
    torus_shell, cube_shell, plane_shell = out
    assert pair_set(cube_shell.vertex_ids[cube_shell.features.edges]) == pair_set(cube_feats)
    assert (cube_shell.features.crease == 3.0).all()
    assert pair_set(torus_shell.vertex_ids[torus_shell.features.edges]) == pair_set(torus_feat)
    assert len(plane_shell.features) == 0
    assert split_shells(mesh, None)[0].mesh.num_faces == torus.num_faces


def test_split_shells_single_and_empty():
    cube = shapes.cube(1)
    out = split_shells(cube, FeatureSet(np.array([[0, 1]]), np.array([3.0])))
    assert len(out) == 1
    assert np.array_equal(out[0].mesh.connects, cube.connects)
    assert out[0].features.edges.tolist() == [[0, 1]]
    empty = Mesh(np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    assert split_shells(empty, FeatureSet.empty()) == []


# --------------------------------------------------------------------------
# Redução
# --------------------------------------------------------------------------


def test_reduction_target():
    cube = shapes.cube(1)
    assert num_triangles(cube) == 12
    assert reduction_target(cube) is None
    assert reduction_target(cube, limit_triangles=12) is None
    assert reduction_target(cube, limit_triangles=10) == 10
    ngon = shapes.cylinder(8, 1, caps="ngon")  # 8 quads + 2 octógonos = 16 + 12 triângulos
    assert num_triangles(ngon) == 28
    assert reduction_target(ngon, 27) == 27


def test_compact_vertices():
    mesh = Mesh.from_faces([[0, 0, 0], [5, 5, 5], [1, 0, 0], [0, 1, 0]], [(0, 2, 3)])
    out, new_index = compact_vertices(mesh)
    assert new_index.tolist() == [0, -1, 1, 2]
    assert out.faces() == [(0, 1, 2)]


def test_maya_reduce_flags(monkeypatch):
    calls = []
    cmds = types.ModuleType("maya.cmds")
    cmds.polyReduce = lambda *args, **flags: calls.append(("polyReduce", args, flags))
    cmds.delete = lambda *args, **flags: calls.append(("delete", args, flags))
    maya = types.ModuleType("maya")
    maya.cmds = cmds
    monkeypatch.setitem(sys.modules, "maya", maya)
    monkeypatch.setitem(sys.modules, "maya.cmds", cmds)
    maya_reduce("|copia", 100_000.0)
    assert calls[0] == (
        "polyReduce",
        ("|copia",),
        {
            "version": 1,
            "termination": 2,
            "triangleCount": 100_000,
            "keepCreaseEdge": True,
            "keepHardEdge": True,
            "keepBorder": True,
            "preserveTopology": True,
            "constructionHistory": False,
        },
    )
    assert calls[1] == ("delete", ("|copia",), {"constructionHistory": True})
    calls.clear()
    maya_reduce("copia", 5000, crease_weight=1.0, border_weight=1)
    flags = calls[0][2]
    assert flags["keepCreaseEdgeWeight"] == 1.0 and flags["keepBorderWeight"] == 1.0
    assert flags["triangleCount"] == 5000


def test_clean_mesh_empty():
    empty = Mesh(np.zeros((2, 3)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    result = clean_mesh(empty, FeatureSet(np.array([[0, 1]]), np.array([3.0])))
    assert result.mesh.num_points == 0 and result.mesh.num_faces == 0
    assert len(result.features) == 0 and result.ok
    assert result.vertex_map.tolist() == [-1, -1]
