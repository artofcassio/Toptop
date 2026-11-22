"""Testes de toptop.core.geom."""

import numpy as np
import pytest

import shapes
from toptop.core.geom import (
    PrimitiveGrid,
    closest_on_segments,
    closest_on_triangles,
    point_grid,
    sample_segments,
    sample_triangles,
)
from toptop.core.mesh import triangulate


def _brute_triangles(points, tris):
    best = np.full(len(points), np.inf)
    arg = np.zeros(len(points), dtype=int)
    for i, tri in enumerate(tris):
        t = np.broadcast_to(tri, (len(points), 3, 3))
        d2, _ = closest_on_triangles(points, t[:, 0], t[:, 1], t[:, 2])
        better = d2 < best
        best[better] = d2[better]
        arg[better] = i
    return np.sqrt(best), arg


def test_closest_on_triangle_regions():
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    c = np.array([[0.0, 1.0, 0.0]])
    cases = [
        ((0.25, 0.25, 2.0), (0.25, 0.25, 0.0)),  # interior
        ((-1.0, -1.0, 0.0), (0.0, 0.0, 0.0)),  # vértice A
        ((2.0, -0.5, 0.0), (1.0, 0.0, 0.0)),  # vértice B
        ((-0.5, 3.0, 0.0), (0.0, 1.0, 0.0)),  # vértice C
        ((0.5, -1.0, 0.0), (0.5, 0.0, 0.0)),  # aresta AB
        ((-1.0, 0.5, 0.0), (0.0, 0.5, 0.0)),  # aresta AC
        ((1.0, 1.0, 0.0), (0.5, 0.5, 0.0)),  # aresta BC
    ]
    for p, expected in cases:
        d2, q = closest_on_triangles(np.array([p]), a, b, c)
        assert q[0] == pytest.approx(expected)
        assert d2[0] == pytest.approx(np.sum((np.array(p) - np.array(expected)) ** 2))


def test_closest_on_triangle_matches_dense_sampling():
    rng = np.random.default_rng(3)
    tri = rng.normal(size=(40, 3, 3))
    pts = rng.normal(size=(40, 3)) * 2.0
    d2, q = closest_on_triangles(pts, tri[:, 0], tri[:, 1], tri[:, 2])
    # amostragem densa do triângulo dá um limite superior muito próximo
    u, v = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 1, 60))
    keep = u + v <= 1.0
    u, v = u[keep], v[keep]
    for i in range(len(tri)):
        s = tri[i, 0] + np.outer(u, tri[i, 1] - tri[i, 0]) + np.outer(v, tri[i, 2] - tri[i, 0])
        sampled = np.min(np.linalg.norm(s - pts[i], axis=1))
        assert np.sqrt(d2[i]) <= sampled + 1e-12
        assert np.sqrt(d2[i]) >= sampled - 0.08
    assert np.linalg.norm(pts - q, axis=1) == pytest.approx(np.sqrt(d2))


def test_closest_on_segment():
    a = np.array([[0.0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]])
    b = np.array([[2.0, 0, 0], [2, 0, 0], [2, 0, 0], [1, 1, 1]])
    p = np.array([[1.0, 3, 0], [-1, 0, 0], [5, 4, 0], [1, 1, 3]])
    d2, q = closest_on_segments(p, a, b)
    assert np.sqrt(d2) == pytest.approx([3.0, 1.0, 5.0, 2.0])
    assert q[:3] == pytest.approx(np.array([[1, 0, 0], [0, 0, 0], [2, 0, 0]]))


def test_grid_matches_brute_force_on_random_soup():
    rng = np.random.default_rng(7)
    centers = rng.uniform(-1, 1, size=(300, 1, 3))
    tris = centers + rng.normal(scale=0.08, size=(300, 3, 3))
    tris[:5] = centers[:5] + rng.normal(scale=1.5, size=(5, 3, 3))  # alguns triângulos enormes
    near = tris[rng.integers(0, 300, 400)].mean(axis=1) + rng.normal(scale=0.02, size=(400, 3))
    far = rng.uniform(-6, 6, size=(200, 3))
    pts = np.concatenate([near, far])
    grid = PrimitiveGrid(tris)
    dist, index, closest, exact = grid.query(pts)
    brute, _ = _brute_triangles(pts, tris)
    assert exact.all()
    assert dist == pytest.approx(brute, abs=1e-10)
    assert np.linalg.norm(pts - closest, axis=1) == pytest.approx(dist, abs=1e-10)
    # o índice devolvido é do array original, e a distância até ele bate
    t = tris[index]
    d2, _ = closest_on_triangles(pts, t[:, 0], t[:, 1], t[:, 2])
    assert np.sqrt(d2) == pytest.approx(brute, abs=1e-10)


def test_grid_on_cad_like_mesh_with_long_thin_triangles():
    m = shapes.cylinder(64, rings=1, radius=1.0, height=40.0, caps="fan")
    tris, _ = triangulate(m)
    tri_pts = m.points[tris]
    rng = np.random.default_rng(11)
    pts = np.stack(
        [rng.uniform(-1.5, 1.5, 500), rng.uniform(-2, 42, 500), rng.uniform(-1.5, 1.5, 500)], axis=1
    )
    dist, _, _, exact = PrimitiveGrid(tri_pts).query(pts)
    brute, _ = _brute_triangles(pts, tri_pts)
    assert exact.all()
    assert dist == pytest.approx(brute, abs=1e-9)


def test_grid_max_dist_cutoff():
    tris = np.array([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    pts = np.array([[0.2, 0.2, 0.05], [0.2, 0.2, 0.5], [9.0, 9.0, 9.0]])
    dist, index, closest, _ = PrimitiveGrid(tris).query(pts, max_dist=0.1)
    assert dist[0] == pytest.approx(0.05) and index[0] == 0
    assert np.isinf(dist[1]) and index[1] == -1 and np.isnan(closest[1]).all()
    assert np.isinf(dist[2]) and index[2] == -1


def test_segment_grid_and_point_grid():
    rng = np.random.default_rng(5)
    a = rng.uniform(-1, 1, size=(200, 3))
    b = a + rng.normal(scale=0.2, size=(200, 3))
    pts = rng.uniform(-2, 2, size=(300, 3))
    dist, index, _, _ = PrimitiveGrid(np.stack([a, b], axis=1)).query(pts)
    brute = np.sqrt(
        np.min([closest_on_segments(pts, np.broadcast_to(x, pts.shape), np.broadcast_to(y, pts.shape))[0] for x, y in zip(a, b)], axis=0)
    )
    assert dist == pytest.approx(brute, abs=1e-10)

    cloud = rng.uniform(-1, 1, size=(500, 3))
    dist, index, _, _ = point_grid(cloud).query(pts)
    full = np.linalg.norm(pts[:, None, :] - cloud[None, :, :], axis=2)
    assert dist == pytest.approx(full.min(axis=1), abs=1e-10)
    assert (index == full.argmin(axis=1)).all()


def test_empty_grid_and_empty_query():
    grid = PrimitiveGrid(np.zeros((0, 3, 3)))
    dist, index, _, _ = grid.query(np.zeros((2, 3)))
    assert np.isinf(dist).all() and (index == -1).all()
    dist, index, _, _ = PrimitiveGrid(np.zeros((1, 3, 3))).query(np.zeros((0, 3)))
    assert len(dist) == 0


def test_budget_exhaustion_returns_bounds_not_garbage():
    rng = np.random.default_rng(2)
    tris = rng.uniform(0, 1, size=(400, 1, 3)) + rng.normal(scale=0.02, size=(400, 3, 3))
    pts = rng.uniform(40, 41, size=(50, 3))  # muito longe de tudo
    dist, _, _, exact = PrimitiveGrid(tris).query(pts, pair_budget=2000)
    brute, _ = _brute_triangles(pts, tris)
    assert (dist[exact] == pytest.approx(brute[exact], abs=1e-9))
    assert not exact.all()
    assert np.isfinite(dist).all()


def test_sampling():
    rng = np.random.default_rng(0)
    tris = np.array([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 5], [3, 0, 5], [0, 3, 5]]])
    pts, tri = sample_triangles(tris, 5000, rng)
    assert len(pts) == 5000
    assert np.mean(tri == 1) == pytest.approx(0.9, abs=0.01)  # proporcional à área
    assert (pts[tri == 0][:, 2] == 0).all() and (pts[:, :2].sum(axis=1)[tri == 0] <= 1 + 1e-12).all()

    a = np.array([[0.0, 0, 0], [0, 0, 0]])
    b = np.array([[1.0, 0, 0], [0, 0.25, 0]])
    spts, seg, weight = sample_segments(a, b, 0.1)
    assert weight.sum() == pytest.approx(1.25)
    assert (seg == 0).sum() == 10 and (seg == 1).sum() == 3
    assert spts[seg == 0][:, 0] == pytest.approx(np.arange(10) / 10 + 0.05)
