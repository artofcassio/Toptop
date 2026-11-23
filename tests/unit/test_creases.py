"""Testes da lógica pura de toptop.core.creases."""

import numpy as np
import pytest

import shapes
from toptop.core.creases import (
    FeatureSet,
    boundary_edges,
    check_input,
    dihedral_angles,
    feature_chains,
    feature_corners,
    suggest_creases,
)
from toptop.core.mesh import Mesh, triangle_mesh


def test_dihedral_cube_and_plane():
    m = shapes.cube(2)
    angle = dihedral_angles(m)
    feat = m.topology.edge_ids(shapes.cube_feature_edges(m))
    on_edge = np.zeros(m.topology.num_edges, dtype=bool)
    on_edge[feat] = True
    assert angle[on_edge] == pytest.approx(90.0)
    assert angle[~on_edge] == pytest.approx(0.0, abs=1e-6)

    plane = shapes.grid_plane(3, 3)
    angle = dihedral_angles(plane)
    assert np.isnan(angle[plane.topology.boundary_edge]).all()
    assert angle[~plane.topology.boundary_edge] == pytest.approx(0.0, abs=1e-6)


def test_dihedral_ignores_flipped_winding():
    plane = shapes.grid_plane(2, 1)
    faces = plane.faces()
    flipped = Mesh.from_faces(plane.points, [faces[0], tuple(reversed(faces[1]))])
    inner = ~flipped.topology.boundary_edge
    assert dihedral_angles(flipped)[inner] == pytest.approx(0.0, abs=1e-6)


def test_suggest_by_angle():
    m = shapes.cylinder(16, rings=2, caps="ngon")  # lados a 22,5 graus, tampas a 90
    at30 = suggest_creases(m, 30.0)
    assert len(at30) == 32
    assert set(np.unique(m.points[at30.ravel()][:, 1]).tolist()) == {0.0, 2.0}
    assert len(suggest_creases(m, 20.0)) == 32 + 16 * 2
    assert len(suggest_creases(shapes.cube(3), 30.0)) == 12 * 3
    # a triangulação não cria quinas novas
    assert len(suggest_creases(triangle_mesh(shapes.cube(3)), 30.0)) == 36
    open_cyl = shapes.cylinder(16)
    assert len(suggest_creases(open_cyl, 30.0)) == 0
    assert len(suggest_creases(open_cyl, 30.0, include_boundary=True)) == 32
    assert len(boundary_edges(open_cyl)) == 32


def test_feature_set_normalizes_and_merges():
    fs = FeatureSet(np.array([[3, 1], [1, 3], [2, 2], [4, 5], [6, 7]]), np.array([1.0, 3.0, 5.0, 0.5, 0.0]))
    assert fs.edges.tolist() == [[1, 3], [4, 5]]
    assert fs.crease.tolist() == [3.0, 0.5]
    assert fs.strong_mask().tolist() == [True, False]
    assert fs.guide_mask().tolist() == [False, True]
    assert fs.strong_mask(threshold=4.0).tolist() == [False, False]
    merged = fs.merged(FeatureSet(np.array([[5, 4], [8, 9]]), np.array([2.5, 1.0])))
    assert merged.edges.tolist() == [[1, 3], [4, 5], [8, 9]]
    assert merged.crease.tolist() == [3.0, 2.5, 1.0]
    remap = np.full(10, -1)
    remap[[1, 3, 4]] = [0, 1, 2]
    assert merged.remapped(remap).edges.tolist() == [[0, 1]]
    assert len(FeatureSet.empty()) == 0
    assert len(FeatureSet(np.array([[0, 1]]))) == 1  # sem valor: crease forte padrão


def test_feature_corners():
    m = shapes.cube(3)
    corners = feature_corners(m.points, shapes.cube_feature_edges(m))
    assert len(corners) == 8
    assert (np.isin(m.points[corners], [0.0, 1.0])).all()

    for sides, expected in ((16, 0), (8, 0), (6, 6), (4, 4)):
        c = shapes.cylinder(sides)
        rim = c.topology.edges[(c.points[c.topology.edges][:, :, 1] == 0).all(axis=1)]
        assert len(rim) == sides
        assert len(feature_corners(c.points, rim)) == expected

    pts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0.2, 0]])
    chain = np.array([[0, 1], [1, 2], [2, 3]])
    assert feature_corners(pts, chain).tolist() == [0, 3]  # só as pontas
    assert feature_corners(pts, chain, angle_deg=5.0).tolist() == [0, 2, 3]


def test_feature_chains():
    m = shapes.cube(3)
    chains = feature_chains(shapes.cube_feature_edges(m))
    assert len(chains) == 12 and all(len(c) == 4 for c in chains)
    c = shapes.cylinder(12)
    rim = c.topology.edges[(c.points[c.topology.edges][:, :, 1] == 0).all(axis=1)]
    loops = feature_chains(rim)
    assert len(loops) == 1 and len(loops[0]) == 13 and loops[0][0] == loops[0][-1]
    assert feature_chains(np.zeros((0, 2), dtype=int)) == []


def _plane_line(plane: Mesh, x: float, y_max: float) -> np.ndarray:
    """Arestas do plano sobre a reta x = const, até y_max."""
    e = plane.topology.edges
    p = plane.points[e]
    on = (np.abs(p[:, :, 0] - x) < 1e-9).all(axis=1) & (p[:, :, 1] <= y_max + 1e-9).all(axis=1)
    return e[on]


def test_check_input_clean_cube():
    m = shapes.cube(3)
    result = check_input(m, FeatureSet(shapes.cube_feature_edges(m)))
    assert result.ok and result.num_strong == 36 and result.num_guide == 0
    assert result.messages == ["Entrada sem problemas de crease."]


def test_check_input_dangling_chain():
    plane = shapes.grid_plane(8, 8)
    # crease que sai da borda e para no meio da superfície
    half = check_input(plane, FeatureSet(_plane_line(plane, 0.5, 0.5), None))
    assert len(half.dangling_vertices) == 1 and not half.ok
    assert plane.points[half.dangling_vertices[0]] == pytest.approx([0.5, 0.5, 0.0])
    assert len(half.dangling_chains) == 1 and len(half.dangling_chains[0]) == 5
    # a mesma crease atravessando de borda a borda está certa
    full = check_input(plane, FeatureSet(_plane_line(plane, 0.5, 1.0), None))
    assert len(full.dangling_vertices) == 0 and full.ok


def test_check_input_unmarked_sharp_and_bad_edges():
    m = shapes.cube(2)
    feat = shapes.cube_feature_edges(m)
    partial = FeatureSet(np.concatenate([feat[:20], [[0, 25]]]), None)
    result = check_input(m, partial)
    assert len(result.unmarked_sharp_edges) == 4
    assert result.not_mesh_edges.tolist() == [[0, 25]]
    assert not result.ok and len(result.messages) >= 2


def test_check_input_marked_ratio():
    plane = shapes.grid_plane(12, 12)
    every = FeatureSet(plane.topology.edges, np.full(plane.topology.num_edges, 1.0))
    result = check_input(plane, every)
    assert result.marked_ratio == pytest.approx(1.0) and result.marked_ratio_excessive
    assert result.num_guide == plane.topology.num_edges
    # em malha minúscula a proporção não é avaliada (um cubo tem 100% de quinas)
    cube = shapes.cube(1)
    assert not check_input(cube, FeatureSet(cube.topology.edges)).marked_ratio_excessive
