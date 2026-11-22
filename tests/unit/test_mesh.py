"""Testes de toptop.core.mesh."""

import numpy as np
import pytest

import shapes
from toptop.core.mesh import (
    Mesh,
    connected_labels,
    edge_lengths,
    extract_faces,
    face_centers,
    face_normals_areas,
    shells,
    triangle_mesh,
    triangulate,
)


def test_cube_topology():
    m = shapes.cube(1)
    topo = m.topology
    assert m.num_points == 8 and m.num_faces == 6
    assert topo.num_edges == 12
    assert (topo.edge_face_count == 2).all()
    assert (topo.vertex_valence == 3).all()
    assert not topo.boundary_vertex.any()
    assert m.diagonal() == pytest.approx(np.sqrt(3.0))


def test_subdivided_cube_is_closed_and_welded():
    m = shapes.cube(4)
    topo = m.topology
    assert m.num_points == 6 * 16 + 2  # V - E + F = 2
    assert m.num_points - topo.num_edges + m.num_faces == 2
    assert not topo.boundary_edge.any() and not topo.nonmanifold_edge.any()
    normals, areas = face_normals_areas(m)
    assert areas.sum() == pytest.approx(6.0)
    # normais para fora: apontam para longe do centro
    outward = np.einsum("ij,ij->i", normals, face_centers(m) - 0.5)
    assert (outward > 0).all()


def test_plane_boundary():
    m = shapes.grid_plane(3, 2)
    topo = m.topology
    assert int(topo.boundary_edge.sum()) == 2 * (3 + 2)
    assert int(topo.boundary_vertex.sum()) == 10
    assert topo.vertex_valence.max() == 4
    assert edge_lengths(m).min() == pytest.approx(1.0 / 3.0)


def test_edge_ids_lookup():
    m = shapes.cube(1)
    topo = m.topology
    ids = topo.edge_ids(topo.edges[::-1, ::-1])
    assert ids.tolist() == list(range(topo.num_edges))[::-1]
    assert topo.edge_ids(np.array([[0, 0], [0, 99], [-1, 2]])).tolist() == [-1, -1, -1]


def test_nonmanifold_edge_detected():
    # três triângulos compartilhando a mesma aresta
    pts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
    m = Mesh.from_faces(pts, [(0, 1, 2), (1, 0, 3), (0, 1, 4)])
    assert int(m.topology.nonmanifold_edge.sum()) == 1


def test_invalid_mesh_rejected():
    with pytest.raises(ValueError):
        Mesh.from_faces([(0, 0, 0), (1, 0, 0)], [(0, 1)])
    with pytest.raises(ValueError):
        Mesh.from_faces([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 5)])


def test_triangulate_keeps_area_and_orientation():
    m = shapes.cylinder(12, rings=2, caps="ngon")
    tris, tri_face = triangulate(m)
    assert len(tris) == int((m.counts - 2).sum())
    assert (np.diff(tri_face) >= 0).all()
    t = triangle_mesh(m)
    _, poly_area = face_normals_areas(m)
    _, tri_area = face_normals_areas(t)
    assert tri_area.sum() == pytest.approx(poly_area.sum())
    assert np.bincount(tri_face, weights=tri_area) == pytest.approx(poly_area)


def test_triangulate_concave_ngon():
    # "L" plano: o leque a partir do vértice 0 sairia do polígono
    pts = [(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (1, 2, 0), (0, 2, 0)]
    for start in range(6):
        order = [(start + i) % 6 for i in range(6)]
        m = Mesh.from_faces(pts, [tuple(order)])
        t = triangle_mesh(m)
        normals, areas = face_normals_areas(t)
        assert areas.sum() == pytest.approx(3.0)
        assert (normals[:, 2] > 0.99).all()


def test_triangulate_concave_quad_uses_valid_diagonal():
    # quad em "ponta de flecha" com o vértice reflexo em cada posição
    base = [(0, 0, 0), (2, 0, 0), (0.5, 0.5, 0), (0, 2, 0)]
    for shift in range(4):
        pts = base[shift:] + base[:shift]
        t = triangle_mesh(Mesh.from_faces(pts, [(0, 1, 2, 3)]))
        normals, areas = face_normals_areas(t)
        assert (normals[:, 2] > 0.99).all()
        assert areas.sum() == pytest.approx(1.0)


def test_shells_and_extract():
    m = shapes.combine(shapes.cube(2), shapes.translated(shapes.cube(1), (5, 0, 0)), shapes.torus())
    labels, count = shells(m)
    assert count == 3
    sub, old = extract_faces(m, labels == labels[-1])
    assert sub.num_faces == shapes.torus().num_faces
    assert np.allclose(sub.points, m.points[old])
    assert shells(sub)[1] == 1


def test_connected_labels_long_chain():
    n = 5000
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    rng = np.random.default_rng(1)
    labels = connected_labels(n, edges[rng.permutation(n - 1)])
    assert (labels == 0).all()
