"""Testes da interface dos motores e do adaptador do QuadWild.

O motor de verdade não roda aqui: dois scripts Python fazem o papel de
``quadwild`` e ``quad_from_patches`` e conferem a linha de comando e os
arquivos que o adaptador escreve. Com ``TOPTOP_QUADWILD_DIR`` apontando para os
binários reais, o último teste roda o motor de verdade.
"""

import os
import sys
import time

import numpy as np
import pytest

import shapes
from fake_quadwild import fake_install as _fake_install
from toptop.core.creases import FeatureSet, suggest_creases
from toptop.core.mesh import Mesh, triangle_mesh
from toptop.core.quality import evaluate
from toptop.engine import quadwild as qw
from toptop.engine.base import (
    Density,
    EngineCancelled,
    EngineError,
    EngineTimeout,
    EngineUnavailable,
    ScaleTransform,
    as_feature_pairs,
    run_process,
)

posix_only = pytest.mark.skipif(os.name == "nt", reason="os motores falsos são scripts com shebang")


# ---------------------------------------------------------------- base


def test_density_and_feature_pairs():
    assert Density.coerce(1999.6).target_quads == 2000
    d = Density(1000, tolerance=0.1)
    assert Density.coerce(d) is d and d.accepts(1100) and d.accepts(900) and not d.accepts(1101)
    assert as_feature_pairs(None).shape == (0, 2)
    assert as_feature_pairs([[3, 1], [1, 3], [2, 2]]).tolist() == [[1, 3]]
    fs = FeatureSet(np.array([[0, 1]]))
    assert as_feature_pairs(fs) is fs.edges


def test_scale_transform_round_trip():
    mesh = shapes.translated(shapes.cube(2, size=0.004), (1000.0, -3.0, 7.0))
    t = ScaleTransform.for_mesh(mesh, target_diagonal=100.0)
    scaled = mesh.with_points(t.apply(mesh.points))
    assert scaled.diagonal() == pytest.approx(100.0)
    lo, hi = scaled.bbox()
    assert lo + hi == pytest.approx(np.zeros(3), abs=1e-6)  # centro na origem
    assert t.invert(scaled.points) == pytest.approx(mesh.points, rel=1e-12)
    flat = Mesh.from_faces([(1, 1, 1)] * 3, [(0, 1, 2)])
    assert ScaleTransform.for_mesh(flat).factor == 1.0  # diagonal zero não divide por zero


def test_run_process_timeout_and_cancel(tmp_path):
    log = str(tmp_path / "log.txt")
    ok = run_process([sys.executable, "-c", "print('oi')"], str(tmp_path), log, 30)
    assert ok == 0 and "oi" in open(log).read()
    assert run_process([sys.executable, "-c", "import sys; sys.exit(3)"], str(tmp_path), log, 30) == 3
    start = time.monotonic()
    with pytest.raises(EngineTimeout):
        run_process([sys.executable, "-c", "import time; time.sleep(30)"], str(tmp_path), log, 0.5)
    with pytest.raises(EngineCancelled):
        run_process([sys.executable, "-c", "import time; time.sleep(30)"], str(tmp_path), log, 30, cancel=lambda: True)
    assert time.monotonic() - start < 10
    with pytest.raises(EngineError):
        run_process([str(tmp_path / "nao_existe")], str(tmp_path), log, 5)


# ---------------------------------------------------------------- arquivos do QuadWild


def test_sharp_records_point_at_the_right_edges():
    tri = triangle_mesh(shapes.cube(2))
    pairs = suggest_creases(tri, 30.0)
    rec = qw.sharp_records(tri, pairs)
    assert len(rec) == 24 and (rec[:, 0] == 1).all()  # cubo: tudo convexo
    faces = tri.connects.reshape(-1, 3)
    a = faces[rec[:, 1], rec[:, 2]]
    b = faces[rec[:, 1], (rec[:, 2] + 1) % 3]
    found = np.sort(np.stack([a, b], axis=1), axis=1)
    assert sorted(map(tuple, found.tolist())) == sorted(map(tuple, pairs.tolist()))
    assert qw.format_sharp(rec).splitlines()[0] == "24"
    assert all(line.count(",") == 2 for line in qw.format_sharp(rec).splitlines()[1:])


def test_sharp_records_concave_boundary_and_bad_input():
    tri = triangle_mesh(shapes.cube(2))
    inside_out = Mesh(tri.points, tri.counts, tri.connects.reshape(-1, 3)[:, ::-1].reshape(-1))
    rec = qw.sharp_records(inside_out, suggest_creases(inside_out, 30.0))
    assert len(rec) == 24 and (rec[:, 0] == 0).all()  # normais para dentro: tudo côncavo

    plane = triangle_mesh(shapes.grid_plane(2, 2))
    border_only = qw.sharp_records(plane, np.zeros((0, 2), dtype=int))
    assert len(border_only) == 8 and (border_only[:, 0] == 1).all()
    assert len(qw.sharp_records(plane, np.zeros((0, 2), dtype=int), include_boundary=False)) == 0
    assert len(qw.sharp_records(plane, np.array([[0, 8], [0, 99]]), include_boundary=False)) == 0
    assert qw.format_sharp(np.zeros((0, 3), dtype=int)) == "0\n"
    with pytest.raises(ValueError):
        qw.sharp_records(shapes.cube(1), np.zeros((0, 2), dtype=int))


def test_config_texts():
    assert qw.prep_config_text(True, 35.0) == "do_remesh 1\nsharp_feature_thr 35\nalpha 0.01\nscaleFact 1\n"
    assert qw.prep_config_text(False, -1).startswith("do_remesh 0\nsharp_feature_thr -1\n")
    text = qw.main_config_text(qw.FLOW_TEMPLATE.replace("\n", "\r\n"), 2.5)
    lines = text.splitlines()
    original = qw.FLOW_TEMPLATE.splitlines()
    assert len(lines) == len(original) == 23 and "\r" not in text
    assert [l.split()[0] for l in lines] == [l.split()[0] for l in original]  # mesma ordem
    assert lines[18] == "scaleFact 2.5"
    assert [l for i, l in enumerate(lines) if i != 18] == [l for i, l in enumerate(original) if i != 18]
    with pytest.raises(EngineError):
        qw.main_config_text("alpha 1\n", 1.0)


def test_scale_fact_and_clean_triangles():
    tri = triangle_mesh(shapes.grid_plane(10, 10))  # área 1, cateto 0,1
    avg = (0.1 + 0.1 + 0.1 * np.sqrt(2)) / 3
    assert qw.scale_fact_for(tri, 400) == pytest.approx(0.05 / avg)
    assert qw.scale_fact_for(tri, 100) == pytest.approx(2 * qw.scale_fact_for(tri, 400))
    dirty = Mesh.from_faces(shapes.cube(1).points, [(0, 1, 2), (0, 0, 1), (1, 2, 2), (0, 1, 2, 3)])
    assert qw.clean_triangles(dirty).num_faces == 3


def test_find_install(tmp_path, monkeypatch):
    monkeypatch.delenv(qw.INSTALL_ENV, raising=False)
    monkeypatch.setattr(qw, "_repo_root", lambda: str(tmp_path / "repo"))
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / "home"))
    assert qw.find_install() is None
    engine = qw.QuadWildEngine()
    ok, message = engine.availability()
    assert not ok and "releases" in message and qw.INSTALL_ENV in message
    with pytest.raises(EngineUnavailable):
        engine.remesh(shapes.cube(1), None, 100)
    if os.name != "nt":
        nested = _fake_install(str(tmp_path / "baixados" / "windows-binaries"))
        assert qw.find_install(str(tmp_path / "baixados")) == nested  # zip extraído numa subpasta
        monkeypatch.setenv(qw.INSTALL_ENV, nested)
        assert qw.find_install() == nested and qw.QuadWildEngine().is_available()


@posix_only
def test_remesh_with_fake_engine(tmp_path):
    install = _fake_install(str(tmp_path / "qw"))
    work = tmp_path / "trabalho com espaço"
    work.mkdir()
    mesh = shapes.translated(shapes.cube(3, size=0.02), (50.0, 0.0, -9.0))
    feats = FeatureSet(suggest_creases(mesh, 30.0))
    steps = []
    engine = qw.QuadWildEngine(install_dir=install, workdir_root=str(work))
    result = engine.remesh(mesh, feats, Density(100), progress=lambda f, m: steps.append((f, m)))
    info = engine.last_run
    assert result.num_faces == 100 and (result.counts == 4).all()
    assert info.details["passes"] == 2 and info.details["sharp_features"] == 36
    assert info.details["remeshed_triangles"] == 108
    assert info.workdir is None and os.listdir(work) == []  # pasta temporária removida
    # escala restaurada: o resultado volta para a caixa da malha original
    lo, hi = mesh.bbox()
    rlo, rhi = result.bbox()
    assert rlo[:2] == pytest.approx(lo[:2]) and rhi == pytest.approx(hi)
    assert steps[0][0] < steps[-1][0] == 1.0 and "fake quad_from_patches" in info.log
    assert mesh.points[0].tolist() == shapes.translated(shapes.cube(3, size=0.02), (50.0, 0.0, -9.0)).points[0].tolist()


@posix_only
def test_density_within_tolerance_needs_one_pass(tmp_path):
    # 42 quads com scaleFact 1: o primeiro scaleFact calculado já dá ~100 quads
    install = _fake_install(str(tmp_path / "qw"), quads_at_scale_one=42)
    engine = qw.QuadWildEngine(install_dir=install, workdir_root=str(tmp_path), keep_workdir=True)
    result = engine.remesh(triangle_mesh(shapes.cube(3)), None, 100)
    assert engine.last_run.details["passes"] == 1 and Density(100).accepts(result.num_faces)
    assert engine.last_run.workdir and os.path.isfile(os.path.join(engine.last_run.workdir, "main.txt"))


@posix_only
@pytest.mark.parametrize("mode,expected", [("no_smooth", True), ("half_smooth", True), ("crash_at_exit", False)])
def test_incomplete_smooth_and_crash_at_exit(tmp_path, mode, expected):
    install = _fake_install(str(tmp_path / "qw"), mode=mode, quads_at_scale_one=27)
    engine = qw.QuadWildEngine(install_dir=install, workdir_root=str(tmp_path))
    result = engine.remesh(shapes.cube(3), None, 100)
    assert result.num_faces > 0 and (result.counts == 4).all()
    assert engine.last_run.details.get("smooth_missing", False) is expected
    assert ("warning" in engine.last_run.details) == (mode == "crash_at_exit")


@posix_only
def test_failure_keeps_the_work_folder(tmp_path):
    install = _fake_install(str(tmp_path / "qw"), mode="fail")
    engine = qw.QuadWildEngine(install_dir=install, workdir_root=str(tmp_path))
    with pytest.raises(EngineError) as raised:
        engine.remesh(shapes.cube(3), suggest_creases(shapes.cube(3)), 100)
    error = raised.value
    assert error.workdir and os.path.isdir(error.workdir)
    assert sorted(os.listdir(error.workdir))[:1] == ["config"]
    assert {"mesh.obj", "mesh.sharp", "prep.txt", "log.txt"} <= set(os.listdir(error.workdir))
    assert "mesh_rem_p0" in str(error) and error.workdir in str(error)
    assert "Wrong Sharp Feature File" in error.log
    assert engine.last_run.workdir == error.workdir


@posix_only
def test_timeout_and_cancel(tmp_path):
    install = _fake_install(str(tmp_path / "qw"), mode="hang")
    engine = qw.QuadWildEngine(install_dir=install, workdir_root=str(tmp_path), timeout_s=0.6)
    start = time.monotonic()
    with pytest.raises(EngineTimeout) as raised:
        engine.remesh(shapes.cube(2), None, 50)
    assert raised.value.workdir and os.path.isdir(raised.value.workdir)
    engine = qw.QuadWildEngine(install_dir=install, workdir_root=str(tmp_path), timeout_s=60)
    with pytest.raises(EngineCancelled):
        engine.remesh(shapes.cube(2), None, 50, cancel=lambda: time.monotonic() - start > 1.0)
    assert time.monotonic() - start < 20


@posix_only
def test_engine_can_detect_its_own_features(tmp_path):
    install = _fake_install(str(tmp_path / "qw"), quads_at_scale_one=27)
    engine = qw.QuadWildEngine(install_dir=install, workdir_root=str(tmp_path), engine_detects_features=True, keep_workdir=True)
    engine.remesh(shapes.cube(3), suggest_creases(shapes.cube(3)), 100)
    assert "mesh.sharp" not in os.listdir(engine.last_run.workdir)
    assert "sharp_features" not in engine.last_run.details


@pytest.mark.skipif(
    not (os.environ.get(qw.INSTALL_ENV) and qw.find_install()), reason="binários reais do QuadWild não instalados"
)
def test_real_quadwild_on_a_cube(tmp_path):
    ref = triangle_mesh(shapes.cube(6))
    feats = FeatureSet(suggest_creases(ref, 30.0))
    engine = qw.QuadWildEngine(workdir_root=str(tmp_path), timeout_s=300)
    result = engine.remesh(ref, feats, 600)
    report = evaluate(result, ref, feats, time_s=engine.last_run.seconds)
    assert report.quad_ratio == 1.0 and Density(600).accepts(report.quads)
    assert report.crease_coverage > 0.99 and report.hausdorff < 0.005
    assert report.corners_matched == 8
