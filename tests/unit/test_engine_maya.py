"""Testes dos adaptadores que rodam dentro do Maya, com um Maya de mentira.

Não substituem rodar no Maya de verdade: conferem o fluxo (flags montados,
limpeza da malha temporária, erros) sem depender dele.
"""

import sys
import types

import numpy as np
import pytest

import shapes
from toptop.engine import ENGINE_NAMES, get_engine
from toptop.engine.base import Density, EngineError, EngineUnavailable
from toptop.engine.maya_retopo import MayaRetopoEngine, supported_flags
from toptop.engine.quadremesher import QuadRemesherEngine

HELP_2025 = """
Synopsis: polyRetopo [flags] [String...]
Flags:
   -e -edit
   -q -query
   -a -anisotropy                  Float
 -ang -angle                       Float
 -eba -edgesByAngle                on|off
  -ch -constructionHistory         on|off
 -fun -faceUniformity              Float
  -pm -preprocessMesh              on|off
 -phe -preserveHardEdges           on|off
 -rpo -replaceOriginal             on|off
 -tfc -targetFaceCount             Int
 -tft -targetFaceCountTolerance    Int
 -trg -topologyRegularity          Float
"""
HELP_2023 = "\n".join(l for l in HELP_2025.splitlines() if "edgesByAngle" not in l and "preprocessMesh" not in l and "-angle" not in l)


class FakeMaya:
    """Registra as chamadas feitas ao ``maya.cmds`` e ao ``maya_bridge``."""

    def __init__(self, monkeypatch, help_text=HELP_2025, batch=True, fail=False):
        self.calls = []
        self.fail = fail
        self.result = shapes.cube(3)
        self.roots = ["|pCube1"]  # objetos na raiz da cena (cmds.ls(assemblies=True))
        self.created = {}  # objeto -> malha, para um polyRetopo que cria objeto novo
        cmds = types.ModuleType("maya.cmds")
        cmds.help = lambda name: help_text
        cmds.about = lambda **kw: batch
        cmds.objExists = lambda name: True
        cmds.delete = lambda *a, **kw: self.calls.append(("delete", a))
        cmds.ls = lambda **kw: list(self.roots)
        cmds.polyRetopo = self._retopo
        maya = types.ModuleType("maya")
        maya.cmds = cmds
        bridge = types.ModuleType("toptop.maya_bridge")
        bridge.create_mesh = self._create
        bridge.set_hard_edges = lambda name, pairs: self.calls.append(("hard", len(pairs)))
        bridge.mesh_from_maya = lambda name, world=True: self.created[name] if name in self.created else self.result
        monkeypatch.setitem(sys.modules, "maya", maya)
        monkeypatch.setitem(sys.modules, "maya.cmds", cmds)
        monkeypatch.setitem(sys.modules, "toptop.maya_bridge", bridge)
        import toptop

        monkeypatch.setattr(toptop, "maya_bridge", bridge, raising=False)

    def _create(self, mesh, name):
        self.calls.append(("create", mesh.num_faces, name))
        self.roots.append("|tmp")
        return "|tmp"

    def _retopo(self, node, **flags):
        self.calls.append(("retopo", node, flags))
        if self.fail:
            raise RuntimeError("Retopologize failed")
        self.roots.extend(self.created)


def test_registry_and_unavailable_outside_maya():
    assert ENGINE_NAMES == ("quadwild", "maya_retopo", "quadremesher")
    for name in ("maya_retopo", "quadremesher"):
        engine = get_engine(name)
        assert engine.name == name and engine.requires_maya
        ok, reason = engine.availability()
        assert not ok and "Maya" in reason
        with pytest.raises(EngineUnavailable):
            engine.remesh(shapes.cube(1), None, 100)
    with pytest.raises(ValueError):
        get_engine("instant_meshes")


def test_supported_flags_parser():
    flags = supported_flags(HELP_2025)
    assert {"targetFaceCount", "preserveHardEdges", "edgesByAngle", "preprocessMesh", "edit"} <= flags
    assert "edgesByAngle" not in supported_flags(HELP_2023)
    assert supported_flags("") == set() and supported_flags(None) == set()


def test_retopo_flags_follow_the_maya_version(monkeypatch):
    fake = FakeMaya(monkeypatch)
    mesh = shapes.cube(2)
    pairs = shapes.cube_feature_edges(mesh)
    engine = MayaRetopoEngine(face_uniformity=0.5, extra_flags={"anisotropy": 0.3})
    assert engine.is_available()
    result = engine.remesh(mesh, pairs, Density(1200, tolerance=0.1))
    assert result.num_faces == fake.result.num_faces
    assert [c[0] for c in fake.calls] == ["create", "hard", "retopo", "delete"]
    assert fake.calls[1] == ("hard", 24)
    flags = fake.calls[2][2]
    assert flags == {
        "targetFaceCount": 1200,
        "targetFaceCountTolerance": 10,
        "preserveHardEdges": True,
        "edgesByAngle": False,
        "preprocessMesh": True,
        "faceUniformity": 0.5,
        "replaceOriginal": True,
        "constructionHistory": False,
        "anisotropy": 0.3,
    }
    assert engine.last_run.details["hard_edges"] == 24 and engine.last_run.seconds >= 0

    old = FakeMaya(monkeypatch, help_text=HELP_2023)
    MayaRetopoEngine().remesh(mesh, None, 500)
    flags = old.calls[2][2]
    assert "edgesByAngle" not in flags and "preprocessMesh" not in flags
    assert flags["preserveHardEdges"] is False  # sem creases não há o que preservar
    assert flags["targetFaceCountTolerance"] == 15


def test_retopo_failure_still_deletes_the_temporary_mesh(monkeypatch):
    fake = FakeMaya(monkeypatch, fail=True)
    with pytest.raises(EngineError, match="polyRetopo falhou"):
        MayaRetopoEngine().remesh(shapes.cube(2), None, 300)
    assert fake.calls[-1][0] == "delete"
    fake = FakeMaya(monkeypatch)
    fake.result = shapes.cube(1).__class__(np.zeros((0, 3)), [], [])
    with pytest.raises(EngineError, match="vazia"):
        MayaRetopoEngine().remesh(shapes.cube(2), None, 300)


def test_quadremesher_is_manual_in_batch_and_explains_missing_names(monkeypatch):
    FakeMaya(monkeypatch, batch=True)
    ok, reason = QuadRemesherEngine().availability()
    assert not ok and "manual" in reason
    FakeMaya(monkeypatch, batch=False)
    ok, reason = QuadRemesherEngine(module_name="modulo_que_nao_existe_qr").availability()
    assert not ok and "não encontrado" in reason
    plugin = types.ModuleType("fake_qr_plugin")
    plugin.doRemeshNow = lambda: None
    monkeypatch.setitem(sys.modules, "fake_qr_plugin", plugin)
    engine = QuadRemesherEngine(module_name="fake_qr_plugin", class_name="Nada", method_name="nada")
    with pytest.raises(EngineError, match="doRemeshNow"):
        engine._entry_point()
    assert QuadRemesherEngine(module_name="fake_qr_plugin", method_name="doRemeshNow")._entry_point() is plugin.doRemeshNow


def test_retopo_reads_a_new_object_when_replace_original_is_ignored(monkeypatch):
    """Se o polyRetopo deixar a temporária intacta e criar outro objeto, o resultado vem dele e ele é apagado."""
    mesh = shapes.cube(2)
    fake = FakeMaya(monkeypatch)
    fake.result = mesh  # a temporária continua igual à entrada
    fake.created = {"|tmpRetopo": shapes.cube(4)}
    result = MayaRetopoEngine().remesh(mesh, None, 300)
    assert result.num_faces == shapes.cube(4).num_faces
    deleted = [c[1][0] for c in fake.calls if c[0] == "delete"]
    assert deleted == ["|tmp", "|tmpRetopo"]  # nada do motor fica na cena; o que já existia não é tocado

    fake = FakeMaya(monkeypatch)
    fake.result = mesh
    with pytest.raises(EngineError, match="sem mudar a malha"):
        MayaRetopoEngine().remesh(mesh, None, 300)
    assert [c[1][0] for c in fake.calls if c[0] == "delete"] == ["|tmp"]
