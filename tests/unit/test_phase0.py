"""Testes da lógica da Fase 0 e da captura de wireframe (sem Maya)."""

import csv
import json
import os
import sys

import numpy as np
import pytest

import shapes
from fake_quadwild import fake_install
from toptop.core import phase0
from toptop.core.creases import FeatureSet, suggest_creases
from toptop.core.mesh import triangle_mesh
from toptop.core.render import BACKGROUND, POLE_LOW, STRONG, read_png, render_wireframe, write_png
from toptop.io.features import features_path, write_features
from toptop.io.obj import write_obj

posix_only = pytest.mark.skipif(os.name == "nt", reason="o motor falso é um script com shebang")


def _models(folder, names=("cubo",)):
    specs = []
    for name in names:
        tri = triangle_mesh(shapes.cube(3))
        obj = str(folder / (name + ".obj"))
        write_obj(obj, tri)
        write_features(features_path(obj), FeatureSet(suggest_creases(tri, 30.0)))
        specs.append(phase0.ModelSpec(name=name, obj=obj, features=features_path(obj), target_quads=96, title="Cubo"))
    return specs


def test_manifest_round_trip_and_obj_fallback(tmp_path):
    specs = _models(tmp_path, ("b_modelo", "a_modelo"))
    assert [s.name for s in phase0.load_models(str(tmp_path))] == ["a_modelo", "b_modelo"]  # sem manifesto: *.obj
    assert phase0.load_models(str(tmp_path))[0].target_quads == phase0.DEFAULT_TARGET_QUADS
    path = phase0.write_manifest(str(tmp_path), specs, extra={"b_modelo": {"triangles": 108}})
    data = json.load(open(path, encoding="utf-8"))
    assert data["models"][0]["obj"] == "b_modelo.obj" and data["models"][0]["triangles"] == 108
    loaded = phase0.load_models(str(tmp_path))
    assert [s.name for s in loaded] == ["b_modelo", "a_modelo"]  # ordem do manifesto
    assert loaded[0].target_quads == 96 and loaded[0].title == "Cubo"
    assert os.path.isfile(loaded[0].obj) and os.path.isfile(loaded[0].features)


def test_png_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(13, 7, 3), dtype=np.uint8)
    path = str(tmp_path / "x.png")
    write_png(path, image)
    assert np.array_equal(read_png(path), image)
    assert open(path, "rb").read(8) == b"\x89PNG\r\n\x1a\n"


def test_render_marks_loops_and_poles(tmp_path):
    mesh = shapes.cube(4)
    path = render_wireframe(
        mesh, str(tmp_path / "cubo.png"), panel=(320, 240), strong_edges=shapes.cube_feature_edges(mesh)
    )
    image = read_png(path)
    assert image.shape == (240, 640, 3)  # dois quadros lado a lado
    flat = image.reshape(-1, 3).astype(int)
    background = (np.abs(flat - np.array(BACKGROUND)).sum(axis=1) < 6).mean()
    assert 0.2 < background < 0.9
    assert (np.abs(flat - np.array(STRONG)).sum(axis=1) < 60).sum() > 200  # loops em laranja
    assert (np.abs(flat - np.array(POLE_LOW)).sum(axis=1) < 60).sum() > 10  # cantos de valência 3
    plain = read_png(render_wireframe(mesh, str(tmp_path / "liso.png"), panel=(320, 240), show_poles=False))
    assert (np.abs(plain.reshape(-1, 3).astype(int) - np.array(STRONG)).sum(axis=1) < 60).sum() == 0


@posix_only
def test_run_one_with_fake_engine(tmp_path):
    spec = _models(tmp_path)[0]
    out = tmp_path / "saida"
    install = fake_install(str(tmp_path / "qw"), quads_at_scale_one=42)
    options = {"install_dir": install, "workdir_root": str(tmp_path)}
    row = phase0.run_one(spec, "quadwild", str(out), engine_options=options, samples=2000)
    assert row["status"] == phase0.STATUS_OK and row["model"] == "cubo" and row["engine"] == "quadwild"
    assert row["quads"] == row["faces"] and row["quad_ratio"] == 1.0 and row["time_s"] > 0
    assert os.path.isfile(row["mesh"]) and os.path.isfile(row["image"])
    assert row["check_quads"] is True and row["target_quads"] == 96
    # o motor falso devolve só um plano: a medição acusa o que falta
    assert row["passed"] is False and row["check_shape_deviation"] is False
    assert set(row) == set(phase0.CSV_COLUMNS)

    failing = fake_install(str(tmp_path / "qw_ruim"), mode="fail")
    bad = phase0.run_one(spec, "quadwild", str(out), engine_options={"install_dir": failing, "workdir_root": str(tmp_path)})
    assert bad["status"] == phase0.STATUS_ERROR and "Pasta de diagnóstico" in bad["message"] and not bad["passed"]
    slow = fake_install(str(tmp_path / "qw_lento"), mode="hang")
    late = phase0.run_one(spec, "quadwild", str(out), timeout_s=0.5, engine_options={"install_dir": slow, "workdir_root": str(tmp_path)})
    assert late["status"] == phase0.STATUS_TIMEOUT and late["time_s"] >= 0.5


def test_unavailable_engine_and_manual_result(tmp_path, monkeypatch):
    spec = _models(tmp_path)[0]
    out = str(tmp_path / "saida")
    missing = phase0.run_one(spec, "maya_retopo", out)
    assert missing["status"] == phase0.STATUS_UNAVAILABLE and "manual" in missing["message"]
    manual = phase0.manual_result_path(out, "quadremesher", spec.name)
    assert manual.endswith(os.path.join("manual", "quadremesher", "cubo.obj"))
    os.makedirs(os.path.dirname(manual))
    write_obj(manual, shapes.cube(8))
    open(os.path.splitext(manual)[0] + ".time.txt", "w").write("12,5\n")
    row = phase0.run_one(spec, "quadremesher", out, manual_only=True, samples=2000)
    assert row["status"] == phase0.STATUS_MANUAL and row["passed"] is True
    assert row["time_s"] == 12.5 and row["quads"] == 384 and row["crease_coverage"] == pytest.approx(1.0)
    unreadable = phase0.run_one(phase0.ModelSpec("x", str(tmp_path / "nao_existe.obj"), ""), "quadwild", out)
    assert unreadable["status"] == phase0.STATUS_ERROR and "ler o modelo" in unreadable["message"]


def _row(model, engine, status="ok", passed=True, fidelity=0.0005, time_s=10.0):
    row = {key: None for key in phase0.CSV_COLUMNS}
    row.update(model=model, engine=engine, status=status, passed=passed, feature_fidelity=fidelity, time_s=time_s,
               quads=1000, quad_ratio=1.0, crease_coverage=0.995, hausdorff=0.002, aspect_median=1.1,
               pole_ratio=0.02, message="linha 1\nlinha 2")
    return row


def test_gate_picks_the_engine_that_passes_more_models():
    rows = []
    for i in range(10):
        rows.append(_row(f"m{i}", "a", passed=i < 9, fidelity=0.0008))  # 9 de 10
        rows.append(_row(f"m{i}", "b", passed=i < 8, fidelity=0.0001))  # 8 de 10, mais fiel
        rows.append(_row(f"m{i}", "c", passed=i < 7))  # 7 de 10: não qualifica
        rows.append(_row(f"m{i}", "d", status="indisponivel", passed=False, fidelity=None, time_s=None))
    result = phase0.gate(rows)
    assert result["min_passed"] == 8 and result["models"] == 10
    assert result["ranking"] == ["a", "b"] and result["chosen"] == "a"
    assert result["engines"]["c"]["qualifies"] is False and result["engines"]["c"]["failed_models"] == ["m7", "m8", "m9"]
    assert result["engines"]["d"]["attempted"] == 0 and not result["engines"]["d"]["qualifies"]
    assert result["engines"]["a"]["total_time_s"] == 100.0


def test_gate_tie_breaks_by_fidelity_then_time():
    rows = [_row(f"m{i}", e, fidelity=f, time_s=t) for i in range(10) for e, f, t in (("lento", 0.0002, 50), ("rapido", 0.0002, 5), ("infiel", 0.0009, 1))]
    assert phase0.gate(rows)["ranking"] == ["rapido", "lento", "infiel"]
    few = [_row(f"m{i}", "a", passed=i < 2) for i in range(3)]
    assert phase0.gate(few)["min_passed"] == 3 and phase0.gate(few)["chosen"] is None  # 80% de 3 arredonda para 3
    assert "Nenhum motor passou" in phase0.summary_text(few)
    assert phase0.gate([])["chosen"] is None


def test_csv_json_and_summary(tmp_path):
    rows = [_row("m1", "a"), _row("m1", "b", status="erro", passed=False, fidelity=None, time_s=None)]
    path = phase0.write_csv(rows, str(tmp_path / "r.csv"))
    table = list(csv.reader(open(path, encoding="utf-8")))
    assert table[0] == phase0.CSV_COLUMNS and len(table) == 3
    first = dict(zip(table[0], table[1]))
    assert first["passed"] == "1" and first["feature_fidelity"] == "0.0005" and first["message"] == "linha 1 | linha 2"
    assert dict(zip(table[0], table[2]))["time_s"] == ""
    br = list(csv.reader(open(phase0.write_csv(rows, str(tmp_path / "br.csv"), excel_br=True), encoding="utf-8-sig"), delimiter=";"))
    assert dict(zip(br[0], br[1]))["feature_fidelity"] == "0,0005"
    data = json.load(open(phase0.write_json(rows, str(tmp_path / "r.json")), encoding="utf-8"))
    assert data["gate"]["chosen"] == "a" and len(data["results"]) == 2 and data["targets"]["hausdorff_max"] == 0.005
    text = phase0.summary_text(rows)
    assert "PASSOU" in text and "QUALIFICA" in text and "Motor escolhido pelas métricas: a" in text


def test_exit_now_ends_the_process_with_the_code():
    import subprocess

    script = "import sys; sys.path.insert(0, %r); from toptop.core.phase0 import exit_now; print('ok'); exit_now(7)"
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    done = subprocess.run([sys.executable, "-c", script % root], capture_output=True, text=True, timeout=60)
    assert done.returncode == 7 and done.stdout.strip() == "ok"
