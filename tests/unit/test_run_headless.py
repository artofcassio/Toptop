"""Testes da autoverificação (tests/run_headless.py)."""

import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def rh():
    spec = importlib.util.spec_from_file_location("run_headless", os.path.join(ROOT, "tests", "run_headless.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(model, status="ok", **metrics):
    base = {"quad_ratio": 1.0, "crease_coverage": 1.0, "feature_fidelity": 0.0004, "hausdorff": 0.002,
            "aspect_median": 1.1, "aspect_p95": 1.6, "pole_ratio": 0.02, "poles_on_strong_crease": 0, "time_s": 10.0}
    base.update(metrics)
    return {"model": model, "status": status, "passed": True, "metrics": base}


def test_baseline_round_trip_and_comparison(rh, tmp_path):
    path = str(tmp_path / "base" / "headless.json")
    assert rh.load_baseline(path) == {}
    entries = [_entry("a"), _entry("b"), _entry("c", status="erro")]
    assert rh.save_baseline(path, entries, {"models": {"velho": {"hausdorff": 0.1}}}, "quadwild") == 2
    base = rh.load_baseline(path)
    assert set(base["models"]) == {"a", "b", "velho"} and base["engine"] == "quadwild"

    same, broken = rh.check_against_baseline([_entry("a"), _entry("b")], base, 0.10)
    assert same == {} and broken == []
    worse, broken = rh.check_against_baseline(
        [_entry("a", hausdorff=0.0025), _entry("b", status="erro"), _entry("novo", hausdorff=9.0)], base, 0.10
    )
    assert list(worse) == ["a"] and worse["a"][0].startswith("hausdorff")
    assert broken == ["b"]  # tinha referência e não rodou; "novo" não tem referência


def test_main_without_models_and_with_maya_only_engine(rh, tmp_path, capsys):
    assert rh.main(["--models-dir", str(tmp_path), "--out", str(tmp_path / "o")]) == 2
    assert "Nenhum modelo" in capsys.readouterr().out


@pytest.mark.skipif(not os.environ.get("TOPTOP_QUADWILD_DIR"), reason="binários reais do QuadWild não instalados")
def test_selftest_detects_a_regression(rh, tmp_path):
    out = str(tmp_path / "saida")
    assert rh.main(["--selftest", "--models", "cubo", "--out", out, "--save-baseline", "--no-images"]) == 0
    baseline_path = os.path.join(out, "selftest_baseline.json")
    data = json.load(open(baseline_path, encoding="utf-8"))
    data["models"]["cubo"]["aspect_median"] = 0.5  # referência melhor do que o motor consegue
    json.dump(data, open(baseline_path, "w", encoding="utf-8"))
    assert rh.main(["--selftest", "--models", "cubo", "--out", out, "--no-images"]) == 1
    results = json.load(open(os.path.join(out, "headless_results.json"), encoding="utf-8"))
    assert list(results["regressions"]) == ["cubo"]


def test_engine_comes_from_the_baseline_and_mismatch_is_refused(rh, tmp_path, capsys, monkeypatch):
    """Sem --engine, vale o motor da referência; com outro motor, a comparação é recusada."""
    models = tmp_path / "modelos"
    models.mkdir()
    (models / "models.json").write_text(json.dumps({"models": [{"name": "m"}]}), encoding="utf-8")
    path = str(tmp_path / "headless.json")
    rh.save_baseline(path, [_entry("m")], {}, "maya_retopo")
    seen = []

    def fake_run_model(spec, args):
        seen.append(args.engine)
        return {**_entry(spec.name), "seconds": 1.0, "message": "", "notes": []}

    monkeypatch.setattr(rh, "run_model", fake_run_model)
    monkeypatch.setattr(rh.run_phase0, "have_maya", lambda: True)
    monkeypatch.setattr(rh.run_phase0, "init_maya", lambda: None)
    common = ["--models-dir", str(models), "--out", str(tmp_path / "o"), "--baseline", path]

    assert rh.main(common) == 0 and seen == ["maya_retopo"]
    assert rh.main(common + ["--engine", "quadwild"]) == 2 and len(seen) == 1
    assert "gravada com o motor maya_retopo" in capsys.readouterr().out

    assert rh.main(common + ["--engine", "quadwild", "--save-baseline"]) == 0 and seen[-1] == "quadwild"
    saved = rh.load_baseline(path)
    assert saved["engine"] == "quadwild" and set(saved["models"]) == {"m"}
    assert "será substituída" in capsys.readouterr().out
