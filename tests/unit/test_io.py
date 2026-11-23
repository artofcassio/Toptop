"""Testes de toptop.io (OBJ e arquivo de features)."""

import numpy as np
import pytest

import shapes
from toptop.core.creases import FeatureSet
from toptop.core.mesh import triangle_mesh
from toptop.io.features import (
    FeatureFileError,
    features_path,
    format_features,
    parse_features,
    read_features,
    write_features,
)
from toptop.io.obj import ObjError, format_obj, parse_obj, read_obj, write_obj

TRICKY = """\
# comentario com acento: malha de teste\r
mtllib nada.mtl\r
o Peca_1
v 0 0 0
v 1 0 0 0.5 0.5 0.5
v 1 1 0 1.0
\tv\t0 1 0
vt 0 0
vt 1 0
vn 0 0 1
vp 0.1 0.2
g grupo_a
usemtl aco
s 1
f 1/1/1 2/2/1 3//1
f 1 3 \\
4
o Peca_2
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
v 0.5 0.5 2e0
f -5 -4 -3 -2
f -1/1 -2/2 -3
l 1 2
p 1
f 5 6
f 9
"""


def test_parse_tricky_obj():
    m = parse_obj(TRICKY, "peca")
    assert m.name == "peca" and m.num_points == 9
    assert m.faces() == [(0, 1, 2), (0, 2, 3), (4, 5, 6, 7), (8, 7, 6)]
    assert m.points[1].tolist() == [1.0, 0.0, 0.0]
    assert m.points[8].tolist() == [0.5, 0.5, 2.0]


def test_obj_errors_name_the_line():
    with pytest.raises(ObjError, match="linha 2"):
        parse_obj("v 0 0 0\nv 1 x 0\n")
    with pytest.raises(ObjError, match="linha 4"):
        parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 7\n")
    with pytest.raises(ObjError, match="linha 2"):
        parse_obj("v 0 0 0\nf 0 1 1\n")
    with pytest.raises(ObjError, match="linha 2"):
        parse_obj("v 0 0 0\nf -2 1 1\n")
    with pytest.raises(ObjError):
        parse_obj("v 0 0\n")


def test_empty_obj():
    m = parse_obj("# nada\n\n")
    assert m.num_points == 0 and m.num_faces == 0


@pytest.mark.parametrize(
    "mesh",
    [shapes.cube(3), shapes.cylinder(9, rings=2, caps="ngon"), triangle_mesh(shapes.torus(7, 5)), shapes.cylinder(5, caps="fan")],
    ids=["quads", "ngons", "triangulos", "misto"],
)
def test_obj_round_trip(tmp_path, mesh):
    path = tmp_path / "pasta com espaço" / "peça.obj"
    path.parent.mkdir()
    write_obj(path, mesh, precision=17, comment="linha 1\nlinha 2")
    back = read_obj(path)
    assert back.name == "peça"
    assert np.array_equal(back.points, mesh.points)
    assert np.array_equal(back.counts, mesh.counts) and np.array_equal(back.connects, mesh.connects)
    assert path.read_bytes().count(b"\r") == 0


def test_obj_default_precision_and_vertex_order():
    rng = np.random.default_rng(4)
    mesh = shapes.cube(2)
    mesh = mesh.with_points(mesh.points * 1234.5678 + rng.normal(size=mesh.points.shape))
    back = parse_obj(format_obj(mesh))
    assert np.allclose(back.points, mesh.points, rtol=1e-11, atol=0)
    assert back.faces() == mesh.faces()
    text = format_obj(triangle_mesh(shapes.cube(1)))
    assert text.count("\nf ") == 12 and "/" not in text


def test_features_round_trip(tmp_path):
    fs = FeatureSet(np.array([[5, 2], [7, 8], [0, 1]]), np.array([3.0, 1.0, 2.5]))
    path = tmp_path / "peca.feat"
    write_features(path, fs, comment="modelo de teste")
    text = path.read_text()
    assert text.startswith("# toptop-features 1\n") and "# modelo de teste" in text
    back = read_features(path, num_points=9)
    assert back.edges.tolist() == fs.edges.tolist() == [[0, 1], [2, 5], [7, 8]]
    assert back.crease.tolist() == [2.5, 3.0, 1.0]
    assert back.strong_mask().tolist() == [True, True, False]
    assert features_path(tmp_path / "peca.obj") == str(path)
    assert len(read_features(tmp_path / "nao_existe.feat")) == 0


def test_parse_features_variants_and_errors():
    fs = parse_features("# cabecalho\n\n3 4\n  5, 6, 1.5  # guia\n4 3 0.25\n")
    assert fs.edges.tolist() == [[3, 4], [5, 6]]
    assert fs.crease.tolist() == [3.0, 1.5]  # sem valor = forte; repetida = maior valor
    assert len(parse_features("")) == 0
    with pytest.raises(FeatureFileError, match="linha 2"):
        parse_features("0 1\n2\n")
    with pytest.raises(FeatureFileError, match="linha 1"):
        parse_features("0 um 2\n")
    with pytest.raises(FeatureFileError, match="linha 1"):
        parse_features("0 12\n", num_points=12)
    with pytest.raises(FeatureFileError):
        parse_features("-1 2\n")
    assert format_features(FeatureSet.empty()).count("\n") == 2


def test_obj_and_features_stay_in_sync(tmp_path):
    """O par OBJ + .feat descreve as mesmas arestas depois de gravar e ler."""
    mesh = shapes.cube(3)
    feats = FeatureSet(shapes.cube_feature_edges(mesh))
    write_obj(tmp_path / "cubo.obj", mesh)
    write_features(features_path(tmp_path / "cubo.obj"), feats)
    back = read_obj(tmp_path / "cubo.obj")
    back_feats = read_features(features_path(tmp_path / "cubo.obj"), back.num_points)
    assert (back.topology.edge_ids(back_feats.edges) >= 0).all()
    assert np.array_equal(shapes.cube_feature_edges(back), back_feats.edges)
