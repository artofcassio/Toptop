"""Testes do ``maya_bridge`` e da leitura de creases com uma API do Maya de mentira.

Conferem o fluxo e as proteções (numeração dos vértices, malha sem creases),
não o comportamento real do Maya.
"""

import importlib
import sys
import types

import numpy as np
import pytest

import shapes


class FakeFnMesh:
    """O mínimo do ``MFnMesh`` que o ``maya_bridge`` usa."""

    store: dict = {}

    def __init__(self, path=None):
        self.path = path

    # criação
    def create(self, points, counts, connects, parent=None):
        drop = FakeFnMesh.drop_unused
        used = sorted(set(connects)) if drop else range(len(points))
        FakeFnMesh.store["shape"] = {"points": [points[i] for i in used], "counts": list(counts)}
        self.path = "shape"
        return "shape"

    @property
    def numVertices(self):
        return len(self.store[self.path]["points"])

    @property
    def numPolygons(self):
        return len(self.store[self.path]["counts"])

    # leitura
    @property
    def numEdges(self):
        return len(self.store[self.path]["edges"])

    def getEdgeVertices(self, i):
        FakeFnMesh.edge_calls += 1
        return self.store[self.path]["edges"][i]

    def getCreaseEdges(self):
        data = self.store[self.path]
        if data.get("creases") is None:
            raise RuntimeError("(kFailure): Unexpected Internal Failure")
        ids, values = zip(*data["creases"]) if data["creases"] else ((), ())
        return list(ids), list(values)


@pytest.fixture
def bridge(monkeypatch):
    calls = []
    om = types.ModuleType("maya.api.OpenMaya")
    om.MPoint = lambda x, y, z: (x, y, z)
    om.MPointArray = list

    class Selection:
        def add(self, name):
            self.name = name

        def getDependNode(self, i):
            return self.name

        def getDagPath(self, i):
            return "shape"

    om.MSelectionList = Selection
    om.MFnMesh = FakeFnMesh
    FakeFnMesh.store = {}
    FakeFnMesh.drop_unused = False
    FakeFnMesh.edge_calls = 0
    cmds = types.ModuleType("maya.cmds")
    cmds.createNode = lambda kind, name, skipSelect: calls.append(("createNode", name)) or name
    cmds.delete = lambda *a, **kw: calls.append(("delete", a))
    cmds.ls = lambda name, long=False: ["|" + name.lstrip("|")]
    cmds.listRelatives = lambda *a, **kw: ["|obj|shape"]
    cmds.rename = lambda *a: calls.append(("rename", a))
    cmds.sets = lambda *a, **kw: calls.append(("sets", a))
    cmds.objExists = lambda name: True
    cmds.nodeType = lambda name: "mesh"
    maya = types.ModuleType("maya")
    maya.cmds = cmds
    api = types.ModuleType("maya.api")
    api.OpenMaya = om
    for name, module in (("maya", maya), ("maya.cmds", cmds), ("maya.api", api), ("maya.api.OpenMaya", om)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "toptop.maya_bridge", raising=False)
    module = importlib.import_module("toptop.maya_bridge")
    yield module, calls
    sys.modules.pop("toptop.maya_bridge", None)
    import toptop

    if hasattr(toptop, "maya_bridge"):
        delattr(toptop, "maya_bridge")


def test_create_mesh_refuses_a_renumbered_mesh(bridge):
    module, calls = bridge
    mesh = shapes.cube(2)
    assert module.create_mesh(mesh, "obj") == "|obj"
    assert ("delete", ("obj",)) not in calls

    # o Maya descarta vértice sem face: a numeração mudaria, então o objeto sai e o erro sobe
    FakeFnMesh.drop_unused = True
    loose = type(mesh)(np.vstack([mesh.points, [[9.0, 9.0, 9.0]]]), mesh.counts, mesh.connects)
    with pytest.raises(RuntimeError, match="numeração mudou"):
        module.create_mesh(loose, "obj")
    assert calls[-1] == ("delete", ("obj",))


def test_read_creases_fetches_only_the_creased_edges(bridge):
    module, _ = bridge
    from toptop.core.creases import maya_read_creases

    edges = [(i, i + 1) for i in range(1000)]
    FakeFnMesh.store["shape"] = {"edges": edges, "creases": [(5, 3.0), (7, 0.0), (900, 1.0)]}
    feats = maya_read_creases("obj")
    assert feats.edges.tolist() == [[5, 6], [900, 901]] and feats.crease.tolist() == [3.0, 1.0]
    assert FakeFnMesh.edge_calls == 2  # não percorre as 1000 arestas

    FakeFnMesh.store["shape"]["creases"] = None  # versão que falha em vez de devolver vazio
    assert len(maya_read_creases("obj")) == 0
    FakeFnMesh.store["shape"]["creases"] = []
    assert len(maya_read_creases("obj")) == 0
    assert module.edge_vertex_pairs("obj").shape == (1000, 2)


def test_pipeline_output_has_no_loose_vertices():
    """O resultado do pipeline chega compacto ao Maya (nenhum vértice sem face)."""
    from toptop.core.pipeline import PipelineOptions, run_pipeline
    from toptop.engine.base import Engine

    class LooseEngine(Engine):
        name = "solto"
        label = "solto"

        def availability(self):
            return True, ""

        def remesh(self, mesh, feature_edges, density, *, cancel=None, progress=None):
            out = shapes.cube(6, size=1.0)
            out = type(out)(out.points - out.points.mean(axis=0) + mesh.points.mean(axis=0), out.counts, out.connects)
            return type(out)(np.vstack([[[50.0, 50.0, 50.0]], out.points]), out.counts, out.connects + 1)

    mesh = shapes.cube(4)
    result = run_pipeline(mesh, None, PipelineOptions(engine=LooseEngine(), target_quads=200, reproject=False))
    used = np.zeros(result.mesh.num_points, dtype=bool)
    used[result.mesh.connects] = True
    assert used.all()
    assert len(result.vertex_deviation) == result.mesh.num_points


def test_estimate_inputs_ask_maya_instead_of_reading_the_mesh(bridge, monkeypatch):
    """A estimativa (a cada troca de seleção) usa polyEvaluate + caixa; só lê a malha se o Maya não responder."""
    module, _ = bridge
    from toptop.core import pipeline

    cmds = sys.modules["maya.cmds"]
    cmds.polyEvaluate = lambda shape, worldArea: 24.0
    cmds.exactWorldBoundingBox = lambda shape: [0.0, 0.0, 0.0, 2.0, 2.0, 1.0]
    cmds.currentUnit = lambda **kw: "cm"
    monkeypatch.setattr(module, "mesh_from_maya", lambda *a, **kw: pytest.fail("leu a malha inteira"))
    assert pipeline.maya_estimate_inputs("obj") == (24.0, 3.0)
    # Maya 2026: a área vem em cm² e a caixa na unidade da cena (cena em mm: 30 mm = 3 cm)
    cmds.polyEvaluate = lambda shape, worldArea: 24.0
    cmds.exactWorldBoundingBox = lambda shape: [0.0, 0.0, 0.0, 20.0, 20.0, 10.0]
    cmds.currentUnit = lambda **kw: "mm"
    assert pipeline.maya_estimate_inputs("obj") == pytest.approx((24.0, 3.0))
    cmds.currentUnit = lambda **kw: "cm"

    cmds.polyEvaluate = lambda shape, worldArea: "Nothing counted : no polygonal object is selected."
    cube = shapes.cube(1, size=2.0)
    monkeypatch.setattr(module, "mesh_from_maya", lambda *a, **kw: cube)
    area, diagonal = pipeline.maya_estimate_inputs("obj")
    assert area == pytest.approx(24.0) and diagonal == pytest.approx(cube.diagonal())
