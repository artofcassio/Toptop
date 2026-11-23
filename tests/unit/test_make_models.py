"""Testes da lógica de tests/make_test_models.py que não depende do Maya."""

import importlib.util
import json
import os
import sys
import types

import numpy as np
import pytest

import shapes

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def gen():
    spec = importlib.util.spec_from_file_location("make_test_models", os.path.join(ROOT, "tests", "make_test_models.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fake_bridge(monkeypatch):
    """maya_bridge falso: a malha do 'nó' vem de um dicionário; componentes viram pares."""
    meshes = {}
    bridge = types.ModuleType("toptop.maya_bridge")
    bridge.mesh_from_maya = lambda node, world=True: meshes[node]
    bridge.edge_components = lambda node, pairs: [f"{node}.e{tuple(p)}" for p in np.asarray(pairs).tolist()]
    import toptop

    monkeypatch.setitem(sys.modules, "toptop.maya_bridge", bridge)
    monkeypatch.setattr(toptop, "maya_bridge", bridge, raising=False)
    return meshes


def _box(sx, sy, sz):
    m = shapes.cube(1)
    return m.with_points((m.points - 0.5) * np.array([sx, sy, sz]))


def test_edges_where_picks_only_the_vertical_corner_edges(gen, fake_bridge):
    fake_bridge["bloco"] = _box(12, 6, 8)
    pick = lambda p: (np.abs(np.abs(p[:, 0]) - 6) < 1e-4) & (np.abs(np.abs(p[:, 2]) - 4) < 1e-4)
    vertical = gen.edges_where("bloco", pick, axis=1)
    assert len(vertical) == 4  # antes: as 12 arestas (todos os cantos passam no filtro por vértice)
    top = gen.edges_where("bloco", lambda p: np.abs(p[:, 1] - 3) < 1e-4, min_angle=60.0)
    assert len(top) == 4
    with pytest.raises(RuntimeError):
        gen.edges_where("bloco", lambda p: p[:, 1] > 100)  # nunca devolve lista vazia


def test_manifest_keeps_previous_entries_and_extras(gen, tmp_path):
    out = str(tmp_path)
    previous = {
        "10_quinas_rasas": {"name": "10_quinas_rasas", "obj": "10_quinas_rasas.obj", "features": "10_quinas_rasas.feat",
                            "target_quads": 1500, "suggest_angle": 15.0},
        "05_simetrica": {"name": "05_simetrica", "obj": "05_simetrica.obj", "features": "05_simetrica.feat",
                         "target_quads": 3000, "symmetry": "x"},
        "meu_modelo": {"name": "meu_modelo", "obj": "meu_modelo.obj", "features": "meu_modelo.feat", "target_quads": 800},
    }
    gen._write_manifest(out, previous)
    items = gen._previous_items(out)
    assert items["10_quinas_rasas"]["suggest_angle"] == 15.0 and items["05_simetrica"]["symmetry"] == "x"
    data = json.load(open(os.path.join(out, "models.json"), encoding="utf-8"))
    assert [m["name"] for m in data["models"]] == ["05_simetrica", "10_quinas_rasas", "meu_modelo"]


def test_previous_items_without_manifest_uses_obj_files(gen, tmp_path):
    from toptop.io.obj import write_obj

    write_obj(str(tmp_path / "02_cilindros_canos.obj"), shapes.cube(1))
    items = gen._previous_items(str(tmp_path))
    assert list(items) == ["02_cilindros_canos"]
    assert items["02_cilindros_canos"]["obj"] == "02_cilindros_canos.obj"
    assert items["02_cilindros_canos"]["features"] == "02_cilindros_canos.feat"


def test_ten_models_listed(gen):
    assert [name[:2] for name, _, _ in gen.MODELS] == [f"{i:02d}" for i in range(1, 11)]


class _FakeCmds:
    """cmds falso que registra as chamadas, na ordem."""

    def __init__(self, edges=0):
        self.calls, self.edges = [], edges

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args))
            if name == "polyCBoolOp":
                return ["resultado"]
            if name == "polyEvaluate":
                return self.edges
            if name == "objExists":
                return False
            return None

        return call


def test_boolean_evaluates_both_operands_first(gen, monkeypatch):
    fake = _FakeCmds()
    monkeypatch.setattr(gen, "_cmds", lambda: fake)
    assert gen.boolean("corpo", "furo", gen.DIFFERENCE, "r") == "resultado"
    names = [name for name, _ in fake.calls]
    first_bool = names.index("polyCBoolOp")
    evaluated = {args[0] for name, args in fake.calls[:first_bool] if name == "polyEvaluate"}
    assert evaluated == {"corpo", "furo"}  # Maya 2026: sem isso a booleana sai errada


def test_check_edges_match_rejects_repeated_edges(gen, fake_bridge, monkeypatch):
    fake_bridge["placa"] = _box(2, 1, 1)
    monkeypatch.setattr(gen, "_cmds", lambda: _FakeCmds(edges=12))
    gen.check_edges_match("placa")  # cubo: 12 arestas nos dois
    monkeypatch.setattr(gen, "_cmds", lambda: _FakeCmds(edges=15))
    with pytest.raises(RuntimeError):
        gen.check_edges_match("placa")
