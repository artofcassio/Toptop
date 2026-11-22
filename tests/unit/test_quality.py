"""Testes das métricas (toptop.core.quality) em malhas sintéticas."""

import json
import math

import numpy as np
import pytest

import shapes
from toptop.core.creases import FeatureSet
from toptop.core.mesh import Mesh, triangle_mesh
from toptop.core.quality import (
    Targets,
    check_targets,
    compare_to_baseline,
    evaluate,
    face_counts,
    format_report,
    heat_colors,
    pole_vertices,
    quad_aspect_ratios,
    surface_deviation,
    vertex_deviation,
)


def _cube_reference():
    ref = triangle_mesh(shapes.cube(1))
    return ref, FeatureSet(shapes.cube_feature_edges(shapes.cube(1)))


def test_face_counts_and_aspect():
    cyl = shapes.cylinder(12, rings=3, caps="fan")
    counts = face_counts(cyl)
    assert counts["quads"] == 36 and counts["triangles"] == 24 and counts["ngons"] == 0
    assert counts["quad_ratio"] == pytest.approx(0.6)
    assert face_counts(shapes.cylinder(12, caps="ngon"))["ngons"] == 2
    assert quad_aspect_ratios(shapes.grid_plane(4, 4, sx=2.0, sy=1.0)) == pytest.approx(2.0)
    assert quad_aspect_ratios(shapes.cube(3)) == pytest.approx(1.0)
    assert len(quad_aspect_ratios(triangle_mesh(shapes.cube(1)))) == 0


def test_poles():
    assert len(pole_vertices(shapes.torus())) == 0
    assert len(pole_vertices(shapes.grid_plane(5, 5))) == 0  # bordas não contam
    cube = shapes.cube(4)
    poles = pole_vertices(cube)
    assert len(poles) == 8 and (cube.topology.vertex_valence[poles] == 3).all()
    fan = shapes.cylinder(12, caps="fan")
    assert sorted(fan.topology.vertex_valence[pole_vertices(fan)].tolist())[-2:] == [12, 12]


def test_perfect_cube_passes_everything():
    ref, feats = _cube_reference()
    report = evaluate(shapes.cube(8), ref, feats, time_s=1.5, name="cubo", engine="teste")
    assert report.faces == 384 and report.quad_ratio == 1.0 and report.shells == 1
    assert report.crease_coverage == pytest.approx(1.0)
    assert report.crease_coverage_strong == pytest.approx(1.0)
    assert report.feature_fidelity == pytest.approx(0.0, abs=1e-12)
    assert report.corners_total == 8 and report.corners_matched == 8
    assert report.hausdorff == pytest.approx(0.0, abs=1e-12)
    assert report.aspect_median == pytest.approx(1.0) and report.aspect_p95 == pytest.approx(1.0)
    assert report.poles == 8 and report.poles_on_strong_crease == 0
    assert report.pole_ratio == pytest.approx(8 / 386)
    assert report.ref_triangles == 12 and report.diagonal == pytest.approx(math.sqrt(3))
    assert all(report.checks.values()) and report.passed
    text = format_report(report)
    assert "APROVADO" in text and "cubo / teste" in text and "FALHA" not in text
    json.dumps(report.to_dict())


def test_coarse_cube_fails_only_pole_ratio():
    ref, feats = _cube_reference()
    report = evaluate(shapes.cube(2), ref, feats)
    assert report.pole_ratio == pytest.approx(8 / 26)
    failed = [k for k, v in report.checks.items() if v is False]
    assert failed == ["poles"] and not report.passed
    assert report.checks["time"] is None  # tempo não informado não reprova


def test_triangles_fail_the_quad_check():
    ref, feats = _cube_reference()
    report = evaluate(triangle_mesh(shapes.cube(4)), ref, feats)
    assert report.quads == 0 and report.triangles == 192
    assert report.checks["quads"] is False and report.checks["aspect_ratio"] is None
    assert report.hausdorff == pytest.approx(0.0, abs=1e-12)


def test_hausdorff_between_parallel_planes():
    ref = shapes.grid_plane(2, 2)
    new = shapes.grid_plane(7, 5, z=0.01)
    dev = surface_deviation(new, ref)
    assert dev["hausdorff"] == pytest.approx(0.01)
    assert dev["new_to_ref_mean"] == pytest.approx(0.01)
    report = evaluate(new, ref)
    assert report.hausdorff == pytest.approx(0.01 / math.sqrt(2))
    assert report.checks["shape_deviation"] is False
    assert vertex_deviation(new, ref) == pytest.approx(0.01)


def test_missing_part_shows_in_reverse_direction():
    ref = shapes.combine(shapes.cube(1), shapes.translated(shapes.cube(1), (3, 0, 0)))
    dev = surface_deviation(shapes.cube(6), ref)
    assert dev["new_to_ref_max"] == pytest.approx(0.0, abs=1e-12)
    assert dev["ref_to_new_max"] == pytest.approx(3.0)  # canto mais distante do cubo que sumiu
    assert dev["hausdorff"] == dev["ref_to_new_max"]


def _plane_with_crease():
    """Plano 1x1 com uma crease reta em x = 0,5 (aresta entre os dois quads)."""
    ref = shapes.grid_plane(2, 1)
    e = ref.topology.edges
    mid = e[(np.abs(ref.points[e][:, :, 0] - 0.5) < 1e-9).all(axis=1)]
    return ref, mid


def test_crease_coverage_needs_an_edge_loop():
    ref, mid = _plane_with_crease()
    feats = FeatureSet(mid, np.array([3.0]))
    with_loop = evaluate(shapes.grid_plane(4, 4), ref, feats)
    assert with_loop.crease_coverage == pytest.approx(1.0)
    assert with_loop.crease_coverage_strong == pytest.approx(1.0)
    assert with_loop.feature_length == pytest.approx(5.0)  # 4 de borda + 1 de crease
    without = evaluate(shapes.grid_plane(3, 3), ref, feats)
    assert without.crease_coverage_strong == pytest.approx(0.0)
    assert without.crease_coverage == pytest.approx(4.0 / 5.0, abs=1e-3)
    assert without.checks["crease_coverage"] is False
    # sem borda automática só sobra a crease
    alone = evaluate(shapes.grid_plane(3, 3), ref, feats, include_boundary=False)
    assert alone.crease_coverage == pytest.approx(0.0)
    # sem loop nenhum, a fidelidade vira um limite inferior: aqui, a distância
    # das pontas da crease (0,5; 0) ao vértice novo mais próximo (1/3; 0)
    assert alone.feature_fidelity == pytest.approx((0.5 - 1 / 3) / math.sqrt(2))
    assert any("limite inferior" in n for n in alone.notes)
    assert alone.checks["feature_fidelity"] is False


def test_guide_crease_is_not_strong():
    ref, mid = _plane_with_crease()
    report = evaluate(shapes.grid_plane(4, 4), ref, FeatureSet(mid, np.array([1.0])))
    assert report.crease_coverage == pytest.approx(1.0)
    assert report.crease_coverage_strong is None


def _grid_with_pole_on_crease():
    grid = shapes.grid_plane(4, 4)
    faces = grid.faces()
    quad = (12, 17, 18, 13)  # quad com canto no vértice (0,5; 0,5)
    assert quad in faces and grid.points[12] == pytest.approx([0.5, 0.5, 0.0])
    faces.remove(quad)
    faces += [(12, 17, 18), (12, 18, 13)]
    return Mesh.from_faces(grid.points, faces)


def test_pole_on_strong_crease_is_counted():
    ref, mid = _plane_with_crease()
    new = _grid_with_pole_on_crease()
    strong = evaluate(new, ref, FeatureSet(mid, np.array([3.0])))
    assert strong.triangles == 2 and strong.poles == 2  # valências 5 e 5 (12 e 18)
    assert strong.poles_on_strong_crease == 1
    assert strong.checks["poles"] is False
    guide = evaluate(new, ref, FeatureSet(mid, np.array([1.0])))
    assert guide.poles_on_strong_crease == 0


def test_pole_at_corner_is_allowed():
    ref, feats = _cube_reference()
    report = evaluate(shapes.cube(8), ref, feats)
    assert report.poles == 8 and report.poles_on_strong_crease == 0


def test_cut_corner_hurts_fidelity():
    ref, feats = _cube_reference()
    new = shapes.cube(8)
    pts = new.points.copy()
    corner = int(np.flatnonzero((pts == 1.0).all(axis=1))[0])
    pts[corner] -= 0.05  # quina empurrada para dentro
    report = evaluate(new.with_points(pts), ref, feats)
    assert report.corners_matched == 7 and report.corners_total == 8
    assert report.feature_corner_dev == pytest.approx(0.05 * math.sqrt(3) / math.sqrt(3))
    assert report.feature_fidelity == pytest.approx(0.05)
    assert report.checks["feature_fidelity"] is False
    assert report.dev_ref_to_new_max > 0.01


def test_coarse_circle_keeps_vertex_fidelity_but_has_chord_error():
    ref = shapes.cylinder(64, caps="ngon")
    rim = ref.topology.edges[np.isin(ref.points[ref.topology.edges][:, :, 1], [0.0, 2.0]).all(axis=1)]
    rim = rim[np.linalg.norm(ref.points[rim[:, 0]] - ref.points[rim[:, 1]], axis=1) < 0.2]
    assert len(rim) == 128
    report = evaluate(shapes.cylinder(16, rings=2, caps="ngon"), ref, FeatureSet(rim))
    # flecha: o vértice original no meio do arco fica a 1 - cos(pi/16) da corda nova
    sagitta = (1.0 - math.cos(math.pi / 16)) / report.diagonal
    assert report.crease_coverage == pytest.approx(1.0)
    assert report.feature_fidelity == pytest.approx(0.0, abs=1e-9)
    assert report.feature_chord_dev == pytest.approx(sagitta, rel=0.03)
    assert report.hausdorff == pytest.approx(sagitta, rel=1e-6)
    assert report.corners_total == 0


def test_stretched_quads_fail_aspect():
    ref = shapes.grid_plane(1, 1, sx=2.0, sy=1.0)
    report = evaluate(shapes.grid_plane(4, 4, sx=2.0, sy=1.0), ref)
    assert report.aspect_median == pytest.approx(2.0)
    assert report.checks["aspect_ratio"] is False


def test_empty_result_is_a_failure():
    ref, feats = _cube_reference()
    empty = Mesh(np.zeros((0, 3)), np.zeros(0, dtype=int), np.zeros(0, dtype=int))
    report = evaluate(empty, ref, feats)
    assert not report.passed and report.checks["quads"] is False
    assert "REPROVADO" in format_report(report)


def test_evaluation_is_deterministic():
    ref = triangle_mesh(shapes.torus(48, 24))
    new = shapes.torus(20, 10)
    a = evaluate(new, ref).to_dict()
    b = evaluate(new, ref).to_dict()
    assert a == b
    assert a["crease_coverage"] is None and a["checks"]["crease_coverage"] is None
    assert 0.0 < a["hausdorff"] < 0.05


def test_time_target_scales_with_input_size():
    t = Targets()
    assert t.time_limit(10_000) == 120.0
    assert t.time_limit(100_000) == 120.0
    assert t.time_limit(250_000) == 300.0
    assert t.time_limit(1_000_000) == 600.0
    ref, feats = _cube_reference()
    slow = evaluate(shapes.cube(8), ref, feats, time_s=500.0)
    assert slow.checks["time"] is False and not slow.passed
    assert check_targets(slow, Targets(time_max_s=900.0, timeout_s=900.0))["time"] is True


def test_compare_to_baseline():
    base = {"quad_ratio": 1.0, "crease_coverage": 0.995, "feature_fidelity": 0.0005, "hausdorff": 0.004,
            "aspect_median": 1.2, "aspect_p95": 2.0, "pole_ratio": 0.03, "poles_on_strong_crease": 0, "time_s": 40.0}
    assert compare_to_baseline(dict(base), base) == []
    assert compare_to_baseline({**base, "hausdorff": 0.00435, "time_s": 44.5}, base) == []  # dentro de 10%
    worse = compare_to_baseline({**base, "hausdorff": 0.0046, "quad_ratio": 0.8, "poles_on_strong_crease": 1}, base)
    assert [w.split(":")[0] for w in worse] == ["quad_ratio", "hausdorff", "poles_on_strong_crease"]
    assert compare_to_baseline({**base, "hausdorff": None}, base)[0].startswith("hausdorff")
    assert compare_to_baseline(base, {}) == []  # sem referência nada reprova


def test_heat_colors():
    colors = heat_colors(np.array([0.0, 0.5, 1.0, 9.0]), vmax=1.0)
    assert colors[0] == pytest.approx([0.0, 0.2, 1.0])
    assert colors[2] == pytest.approx([1.0, 0.0, 0.0]) and colors[3] == pytest.approx([1.0, 0.0, 0.0])
    assert colors.shape == (4, 3) and (colors >= 0).all() and (colors <= 1).all()


def _plane_with_lines(xs):
    """Plano 1x1 com uma coluna de quads entre cada par de x; devolve também as linhas internas."""
    xs = np.asarray(xs, dtype=float)
    pts = [(x, y, 0.0) for x in xs for y in (0.0, 1.0)]
    faces = [(2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1) for i in range(len(xs) - 1)]
    inner = np.array([[2 * i, 2 * i + 1] for i in range(1, len(xs) - 1)])
    return Mesh.from_faces(pts, faces), inner


def test_thin_wall_each_crease_needs_its_own_loop():
    """Duas creases mais próximas que a tolerância: um loop só cobre a mais próxima."""
    ref, inner = _plane_with_lines([0.0, 0.5, 0.504, 1.0])  # 0,28% da diagonal entre as duas
    feats = FeatureSet(inner)
    both, _ = _plane_with_lines([0.0, 0.25, 0.5, 0.504, 0.75, 1.0])
    assert evaluate(both, ref, feats).crease_coverage_strong == pytest.approx(1.0)
    one, _ = _plane_with_lines([0.0, 0.25, 0.5, 0.75, 1.0])
    report = evaluate(one, ref, feats)
    assert report.crease_coverage_strong == pytest.approx(0.5)
    assert report.checks["crease_coverage"] is False
