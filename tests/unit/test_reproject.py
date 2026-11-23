"""Testes da reprojeção (toptop.core.reproject) em malhas sintéticas."""

import math

import numpy as np
import pytest

import shapes
from toptop.core.creases import FeatureSet
from toptop.core.mesh import Mesh, triangle_mesh
from toptop.core.quality import evaluate, vertex_deviation
from toptop.core.reproject import (
    KIND_CORNER,
    KIND_CURVE,
    KIND_SURFACE,
    KIND_UNTOUCHED,
    ReprojectResult,
    _snap_corners,
    feature_network,
    reproject,
)


def _cube_reference() -> tuple[Mesh, FeatureSet]:
    """Cubo unitário triangulado e as 12 arestas dele como feature."""
    return triangle_mesh(shapes.cube(1)), FeatureSet(shapes.cube_feature_edges(shapes.cube(1)))


def _noisy_cube(n: int = 8, scale: float = 1.01, noise: float = 0.002, seed: int = 1) -> Mesh:
    """Cubo n x n inflado em torno do centro e com ruído pequeno."""
    cube = shapes.cube(n)
    rng = np.random.default_rng(seed)
    pts = 0.5 + (cube.points - 0.5) * scale + rng.uniform(-noise, noise, cube.points.shape)
    return cube.with_points(pts)


def _on_side(values: np.ndarray) -> np.ndarray:
    """Coordenadas exatamente em 0 ou 1."""
    return (values == 0.0) | (values == 1.0)


def _plane(xs: list[float], ys: np.ndarray, z: float = 0.0) -> Mesh:
    """Grade de quads com linhas nas coordenadas dadas."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)
    ny = len(ys)
    faces = []
    for i in range(len(xs) - 1):
        for j in range(ny - 1):
            a, b = i * ny + j, (i + 1) * ny + j
            faces.append((a, b, b + 1, a + 1))
    return Mesh.from_faces(pts, faces, "grade")


def _vertex_at(mesh: Mesh, xyz: tuple[float, float, float]) -> int:
    """Índice do vértice numa posição dada."""
    d = np.linalg.norm(mesh.points - np.asarray(xyz, dtype=float), axis=1)
    i = int(np.argmin(d))
    assert d[i] < 1e-12
    return i


def test_noisy_cube_goes_back_to_surface_edges_and_corners():
    ref, feats = _cube_reference()
    new = _noisy_cube()
    res = reproject(new, ref, feats)
    assert isinstance(res, ReprojectResult)
    out = res.mesh.points
    assert res.mesh is not new and res.mesh.num_faces == new.num_faces
    assert np.array_equal(res.mesh.connects, new.connects)

    # todo vértice em cima da superfície do cubo
    assert vertex_deviation(res.mesh, ref).max() < 1e-9

    grid = shapes.cube(8).points
    sides = _on_side(grid).sum(axis=1)
    corner, edge, face = sides == 3, sides == 2, sides == 1
    assert corner.sum() == 8 and edge.sum() == 84 and face.sum() == 294
    # cantos exatos, vértices de loop exatamente sobre as arestas do cubo
    assert np.array_equal(out[corner], grid[corner])
    assert (res.kind[corner] == KIND_CORNER).all()
    assert (_on_side(out[edge]).sum(axis=1) >= 2).all()
    assert (res.kind[edge] == KIND_CURVE).all()
    assert (res.kind[face] == KIND_SURFACE).all()
    assert res.corners_snapped == 8 and res.corners_missing == 0
    assert res.max_move == pytest.approx(0.05 * math.sqrt(3.0))
    assert np.allclose(res.moved, np.linalg.norm(out - new.points, axis=1))
    assert res.moved.max() <= res.max_move
    assert res.kind.dtype == np.int8 and res.kind.shape == (new.num_points,)

    before = evaluate(new, ref, feats)
    after = evaluate(res.mesh, ref, feats)
    assert before.feature_fidelity > 1e-3 and before.hausdorff > 1e-3
    assert after.feature_fidelity == pytest.approx(0.0, abs=1e-9)
    assert after.hausdorff == pytest.approx(0.0, abs=1e-9)
    assert after.crease_coverage == pytest.approx(1.0)
    assert after.corners_matched == 8


def test_cylinder_pulled_back_to_radius():
    ref = triangle_mesh(shapes.cylinder(64, rings=1, radius=1.0, height=2.0))
    coarse = shapes.cylinder(16, rings=4, radius=1.0, height=2.0)
    angle = math.radians(7.0)  # vértices entre os da referência
    rot = np.array(
        [[math.cos(angle), 0.0, -math.sin(angle)], [0.0, 1.0, 0.0], [math.sin(angle), 0.0, math.cos(angle)]]
    )
    pts = coarse.points @ rot.T
    pts[:, [0, 2]] *= 1.05
    new = coarse.with_points(pts)

    chord = math.cos(math.pi / 64)  # raio mínimo da superfície de 64 lados
    for iterations in (1, 2):
        res = reproject(new, ref, iterations=iterations)
        out = res.mesh.points
        radius = np.hypot(out[:, 0], out[:, 2])
        assert radius.min() >= chord - 1e-9 and radius.max() <= 1.0 + 1e-9
        assert np.allclose(out[:, 1], new.points[:, 1], atol=1e-9)
        assert vertex_deviation(res.mesh, ref).max() < 1e-9
        assert (res.kind >= KIND_SURFACE).all()
        assert res.corners_missing == 0  # aros redondos não têm cantos

    # na 2ª passada os aros (bordas abertas) já estão perto e viram loops
    rim = (new.points[:, 1] == 0.0) | (new.points[:, 1] == 2.0)
    first = reproject(new, ref)
    second = reproject(new, ref, iterations=2)
    assert (first.kind[rim] == KIND_SURFACE).all()
    assert (second.kind[rim] == KIND_CURVE).all()
    assert (second.kind[~rim] == KIND_SURFACE).all()
    assert reproject(new, ref, include_boundary=False, iterations=2).kind[rim].max() == KIND_SURFACE


def test_vertex_near_crease_but_not_on_loop_goes_to_surface():
    # placa com duas creases retas: x = 0.25 e x = 0.75
    ref = triangle_mesh(shapes.grid_plane(4, 4))
    p = ref.points
    edges = ref.topology.edges
    on_line = np.isclose(p[:, 0], 0.25) | np.isclose(p[:, 0], 0.75)
    same_x = np.isclose(p[edges[:, 0], 0], p[edges[:, 1], 0])
    feats = FeatureSet(edges[on_line[edges].all(axis=1) & same_x])
    assert len(feats) == 8

    # coluna x = 0.753 corre sobre a crease x = 0.75; nenhuma coluna segue a x = 0.25
    new = _plane([0.0, 0.2, 0.4, 0.6, 0.753, 1.0], np.linspace(0.0, 1.0, 6), z=0.004)
    pts = new.points.copy()
    near = _vertex_at(new, (0.2, 0.4, 0.004))
    pts[near, 0] = 0.24  # a 0.011 da crease x = 0.25 (dentro da tolerância), fora de loop
    new = new.with_points(pts)
    loop = np.flatnonzero(np.isclose(pts[:, 0], 0.753) & (pts[:, 1] > 0) & (pts[:, 1] < 1))
    assert len(loop) == 4

    res = reproject(new, ref, feats)
    out = res.mesh.points
    assert res.kind[near] == KIND_SURFACE
    assert out[near] == pytest.approx([0.24, 0.4, 0.0], abs=1e-12)
    assert (res.kind[loop] == KIND_CURVE).all()
    assert np.allclose(out[loop, 0], 0.75, atol=1e-12) and np.allclose(out[loop, 2], 0.0, atol=1e-12)
    # 4 cantos da placa + as pontas da crease x = 0.75; as da x = 0.25 ficam sem vértice
    assert res.corners_snapped == 6 and res.corners_missing == 2
    assert out[_vertex_at(new, (0.753, 0.0, 0.004))] == pytest.approx([0.75, 0.0, 0.0], abs=1e-12)
    assert out[_vertex_at(new, (0.0, 0.0, 0.004))] == pytest.approx([0.0, 0.0, 0.0], abs=0.0)
    assert np.abs(out[:, 2]).max() < 1e-12


def test_loop_vertex_keeps_to_the_crease_it_follows():
    # crease A horizontal (y = 0.5) e crease B saindo do meio dela a 40 graus
    end = (0.5 + 0.5 * math.cos(math.radians(40)), 0.5 + 0.5 * math.sin(math.radians(40)))
    pts = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0.5), (0.5, 0.5), (1, 0.5), end]
    tris = [(0, 1, 5), (0, 5, 4), (1, 6, 5), (4, 5, 3), (5, 7, 3), (5, 6, 7), (6, 2, 7), (7, 2, 3)]
    ref = Mesh.from_faces([(x, y, 0.0) for x, y in pts], tris, "v")
    feats = FeatureSet([(4, 5), (5, 6), (5, 7)])

    # faixa de quads logo acima da crease A; o vértice x = 0.52 está mais perto
    # da crease B (0.0052) do que da A (0.01), mas o loop dele corre ao longo de A
    xs = [0.0] + [0.12 + 0.1 * i for i in range(9)] + [1.0]
    strip_pts = [(x, 0.3, 0.0) for x in xs] + [(x, 0.51, 0.0) for x in xs]
    k = len(xs)
    faces = [(i, i + 1, k + i + 1, k + i) for i in range(k - 1)]
    new = Mesh.from_faces(strip_pts, faces, "faixa")
    v = _vertex_at(new, (0.52, 0.51, 0.0))

    res = reproject(new, ref, feats, include_boundary=False)
    out = res.mesh.points
    assert res.kind[v] == KIND_CURVE
    assert out[v] == pytest.approx([0.52, 0.5, 0.0], abs=1e-12)
    top = np.arange(k, 2 * k)
    assert (res.kind[top] >= KIND_CURVE).all() and np.allclose(out[top, 1], 0.5, atol=1e-12)
    assert res.kind[k] == KIND_CORNER and res.kind[2 * k - 1] == KIND_CORNER
    assert (res.kind[:k] == KIND_SURFACE).all()
    # nenhum vértice novo perto da junção nem da ponta solta da crease B
    assert res.corners_snapped == 2 and res.corners_missing == 2
    assert any("canto" in m for m in res.messages)


def test_max_move_leaves_far_vertices_untouched():
    ref, feats = _cube_reference()
    cube = shapes.cube(8)
    pts = np.concatenate([cube.points, [[5.0, 5.0, 5.0]]])  # um vértice solto, sem face
    top = _vertex_at(cube, (0.5, 0.5, 1.0))
    pts[top, 2] = 1.3
    new = Mesh(pts, cube.counts, cube.connects, "cubo")

    res = reproject(new, ref, feats)
    assert res.kind[top] == KIND_UNTOUCHED and res.kind[-1] == KIND_UNTOUCHED
    assert np.array_equal(res.mesh.points[top], pts[top])
    assert np.array_equal(res.mesh.points[-1], pts[-1])
    assert res.moved[top] == 0.0 and res.moved[-1] == 0.0
    assert (res.kind[:-1] >= KIND_SURFACE).sum() == cube.num_points - 1
    assert any("1 vértice(s) ficaram parados" in m for m in res.messages)

    loose = reproject(new, ref, feats, max_move_rel=0.5)
    assert loose.kind[top] == KIND_SURFACE and loose.kind[-1] == KIND_UNTOUCHED
    assert loose.mesh.points[top] == pytest.approx([0.5, 0.5, 1.0], abs=1e-12)
    assert loose.moved[top] == pytest.approx(0.3)


def test_inputs_are_not_modified():
    ref, feats = _cube_reference()
    new = _noisy_cube(n=4)
    saved = [a.copy() for a in (new.points, new.counts, new.connects, ref.points, ref.connects)]
    saved_feats = (feats.edges.copy(), feats.crease.copy())
    res = reproject(new, ref, feats, iterations=2)
    for before, after in zip(saved, (new.points, new.counts, new.connects, ref.points, ref.connects)):
        assert np.array_equal(before, after)
    assert np.array_equal(saved_feats[0], feats.edges) and np.array_equal(saved_feats[1], feats.crease)
    assert res.mesh.points is not new.points
    assert not np.shares_memory(res.mesh.points, new.points)
    assert not np.array_equal(res.mesh.points, new.points)


def test_without_features_everything_goes_to_surface():
    ref, _ = _cube_reference()
    new = _noisy_cube(n=4)
    res = reproject(new, ref)  # cubo fechado: nenhuma borda, nenhuma crease
    assert (res.kind == KIND_SURFACE).all()
    assert res.corners_snapped == 0 and res.corners_missing == 0
    assert vertex_deviation(res.mesh, ref).max() < 1e-9


def test_degenerate_inputs():
    new = _noisy_cube(n=2)
    empty_ref = Mesh(np.zeros((3, 3)), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    res = reproject(new, empty_ref)
    assert (res.kind == KIND_UNTOUCHED).all() and np.array_equal(res.mesh.points, new.points)
    assert res.messages and (res.moved == 0).all()

    ref, feats = _cube_reference()
    bad = FeatureSet(np.concatenate([feats.edges, [[0, 99]]]))
    res = reproject(new, ref, bad)
    assert any("ignoradas" in m for m in res.messages)
    assert res.corners_snapped == 8


def test_feature_network_adds_borders_and_drops_bad_edges():
    plane = triangle_mesh(shapes.grid_plane(2, 2))
    net, dropped = feature_network(plane, FeatureSet([(0, 4), (0, 50)]))
    assert dropped == 1 and len(net) == 8 + 1
    assert net.strong_mask().sum() == 1  # bordas entram como guia
    assert len(feature_network(plane, None, include_boundary=False)[0]) == 0


def test_snap_corners_resolves_conflicts_by_distance():
    corners = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    cands = np.array([[0.004, 0.0, 0.0], [0.02, 0.0, 0.0], [0.9, 0.0, 0.0]])
    ci, vi = _snap_corners(corners, cands, 0.05)
    assert dict(zip(ci.tolist(), vi.tolist())) == {0: 0, 1: 1}
    ci, vi = _snap_corners(corners, cands[:1], 0.05)
    assert ci.tolist() == [0] and vi.tolist() == [0]
    ci, vi = _snap_corners(corners, cands[2:], 0.05)
    assert ci.size == 0 and vi.size == 0
