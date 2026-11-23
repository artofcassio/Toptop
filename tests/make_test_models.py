"""Gera proceduralmente os 10 modelos de teste da Fase 0, com creases marcados.

Uso (Windows, na raiz do repositório):

    mayapy tests\\make_test_models.py
    mayapy tests\\make_test_models.py --only 01,04 --out tests\\models

Também roda dentro do Maya: ``import make_test_models; make_test_models.main([])``.

Para cada modelo grava em ``tests/models``:

* ``<nome>.mb``: a cena com a peça (ngons das booleanas preservados), creases
  gravados nas arestas (3 = forte, 1 = guia) e hard edges nas creases fortes;
* ``<nome>.obj``: a mesma peça triangulada (a superfície de referência);
* ``<nome>.feat``: as arestas de feature (pares de vértices do OBJ + crease);
* ``models.json``: a lista dos modelos com a densidade pedida a todos os motores.

As creases fortes saem do ângulo diedral (30 graus; 15 no modelo de quinas
rasas) e as guias são linhas cortadas de propósito em superfícies lisas.
Este script NÃO foi executado na sessão que o escreveu (não havia Maya): se um
modelo falhar, os outros continuam e o erro aparece no resumo final.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

STRONG = 3.0
GUIDE = 1.0
UNION, DIFFERENCE, INTERSECTION = 1, 2, 3


# --------------------------------------------------------------------------
# Ajudantes de construção (maya.cmds)
# --------------------------------------------------------------------------


def _cmds():
    from maya import cmds

    return cmds


def box(name, size, center=(0.0, 0.0, 0.0), subdivisions=(1, 1, 1), rotate=None):
    """Caixa de lados ``size`` centrada em ``center``."""
    cmds = _cmds()
    node = cmds.polyCube(w=size[0], h=size[1], d=size[2], sx=subdivisions[0], sy=subdivisions[1],
                         sz=subdivisions[2], ch=False, n=name)[0]  # fmt: skip
    if rotate:
        cmds.rotate(rotate[0], rotate[1], rotate[2], node)
    cmds.move(center[0], center[1], center[2], node)
    return node


def cylinder(name, radius, height, sides, center=(0.0, 0.0, 0.0), axis=(0, 1, 0)):
    """Cilindro com tampas em ngon (como sai de um CAD), eixo ``axis``."""
    cmds = _cmds()
    node = cmds.polyCylinder(r=radius, h=height, sx=sides, sy=1, sc=0, ax=axis, ch=False, n=name)[0]
    cmds.move(center[0], center[1], center[2], node)
    return node


def boolean(a, b, operation, name):
    """Booleana sem histórico; apaga o que sobrar dos operandos."""
    cmds = _cmds()
    result = cmds.polyCBoolOp(a, b, op=operation, ch=False, n=name)[0]
    cmds.delete(result, ch=True)
    for leftover in (a, b):
        if leftover != result and cmds.objExists(leftover):
            cmds.delete(leftover)
    return result


def cut(body, cutters, name):
    """Subtrai os cortadores do corpo, um de cada vez (mais estável que juntá-los)."""
    for i, cutter in enumerate(cutters):
        body = boolean(body, cutter, DIFFERENCE, f"{name}_c{i}")
    return _cmds().rename(body, name)


def unite(parts, name):
    """Junta as peças num objeto só, sem soldar (uma casca por peça)."""
    cmds = _cmds()
    result = cmds.polyUnite(parts, ch=False, n=name)[0]
    for part in parts:
        if cmds.objExists(part):
            cmds.delete(part)
    return result


def _normals_point_out(node):
    """Verdadeiro se o volume assinado da malha é positivo (normais para fora)."""
    import numpy as np

    from toptop import maya_bridge
    from toptop.core.mesh import triangle_mesh

    tri = triangle_mesh(maya_bridge.mesh_from_maya(node))
    p = tri.points[tri.connects.reshape(-1, 3)]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()) > 0


def finish(node):
    """Solda vértices coincidentes, apaga histórico e congela as transformações."""
    cmds = _cmds()
    cmds.polyMergeVertex(node, d=1e-5, ch=False)
    cmds.delete(node, ch=True)
    cmds.makeIdentity(node, apply=True, t=True, r=True, s=True, n=False)
    cmds.xform(node, centerPivots=True)
    return node


def edges_where(node, predicate, min_angle=None):
    """Componentes das arestas cujas duas pontas satisfazem ``predicate(pontos)``."""
    from toptop import maya_bridge
    from toptop.core.creases import dihedral_angles

    mesh = maya_bridge.mesh_from_maya(node)
    edges = mesh.topology.edges
    keep = predicate(mesh.points[edges[:, 0]]) & predicate(mesh.points[edges[:, 1]])
    if min_angle is not None:
        import numpy as np

        with np.errstate(invalid="ignore"):
            keep &= dihedral_angles(mesh) > min_angle
    return maya_bridge.edge_components(node, edges[keep])


def mark_creases(node, angle=30.0, guide_planes=()):
    """Grava os creases: forte onde o diedro passa de ``angle``; guia nos planos dados.

    ``guide_planes`` são pares (eixo, valor): arestas contidas nesse plano que
    não são quina viram crease guia (uma divisão de painel em superfície lisa).
    Devolve (número de arestas fortes, número de guias).
    """
    import numpy as np

    from toptop import maya_bridge
    from toptop.core.creases import dihedral_angles

    cmds = _cmds()
    mesh = maya_bridge.mesh_from_maya(node)
    edges = mesh.topology.edges
    with np.errstate(invalid="ignore"):
        strong = dihedral_angles(mesh) > angle
    guide = np.zeros(len(edges), dtype=bool)
    for axis, value in guide_planes:
        coord = mesh.points[edges][:, :, axis]
        guide |= (np.abs(coord - value) < 1e-4).all(axis=1)
    guide &= ~strong
    cmds.polySoftEdge(node, a=180, ch=False)
    strong_components = maya_bridge.edge_components(node, edges[strong])
    if strong_components:
        cmds.polyCrease(strong_components, value=STRONG)
        cmds.polySoftEdge(strong_components, a=0, ch=False)
    guide_components = maya_bridge.edge_components(node, edges[guide])
    if guide_components:
        cmds.polyCrease(guide_components, value=GUIDE)
    cmds.delete(node, ch=True)
    return int(strong.sum()), int(guide.sum())


def from_arrays(name, points, faces):
    """Cria um objeto do Maya a partir de vértices e faces (listas do Python)."""
    from toptop import maya_bridge
    from toptop.core.mesh import Mesh

    return maya_bridge.create_mesh(Mesh.from_faces(points, faces), name)


def plane_edge_count(node, axis, value):
    """Quantas arestas da peça estão inteiras dentro do plano ``eixo = valor``."""
    import numpy as np

    from toptop import maya_bridge

    mesh = maya_bridge.mesh_from_maya(node)
    coord = mesh.points[mesh.topology.edges][:, :, axis]
    return int((np.abs(coord - value) < 1e-4).all(axis=1).sum())


def ensure_plane_edges(node, axis, value):
    """Garante arestas no plano ``eixo = valor`` (a futura crease guia).

    Os modelos já são construídos com uma divisão nesse plano; se a booleana a
    tiver apagado, a peça é cortada com ``polyCut``. A orientação padrão do
    plano de corte não foi verificada, então as três rotações são tentadas até
    aparecerem arestas no plano (um corte na orientação errada só acrescenta
    arestas, sem mudar a forma).
    """
    cmds = _cmds()
    if plane_edge_count(node, axis, value):
        return
    center = [0.0, 0.0, 0.0]
    center[axis] = value
    for rotation in ((0, 90, 0), (90, 0, 0), (0, 0, 0)):
        cmds.polyCut(node, pc=center, ro=rotation, ch=False)
        cmds.delete(node, ch=True)
        if plane_edge_count(node, axis, value):
            return
    print(f"  AVISO: não consegui criar arestas no plano {'XYZ'[axis]} = {value}; a crease guia ficará vazia.")


# --------------------------------------------------------------------------
# Os dez modelos (seção 11 do plano)
# --------------------------------------------------------------------------


def model_01_placa_furos():
    """Placa com furos passantes, um furo grande e um rebaixo quadrado."""
    # sx=2: a caixa já nasce com um loop em x = 0, a linha de painel (crease guia)
    body = box("placa", (20, 1.2, 12), subdivisions=(2, 1, 1))
    cutters = [cylinder(f"furo{i}", 0.8, 3, 20, (-7.5 + 3 * i, 0, -3.5)) for i in range(6)]
    cutters.append(cylinder("furo_grande", 2.5, 3, 32, (-4, 0, 2)))
    cutters.append(box("rebaixo", (4, 1.0, 3), (5, 0.6, 2)))
    node = finish(cut(body, cutters, "placa_furos"))
    ensure_plane_edges(node, 0, 0.0)
    return node, {"guide_planes": [(0, 0.0)], "target_quads": 3000}


def model_02_cilindros_canos():
    """Cano em T com flange: cilindros que se cruzam, por fora e por dentro."""
    outer = cylinder("cano", 3.0, 14, 32)
    flange = cylinder("flange", 4.5, 0.8, 32, (0, -6.7, 0))  # -7,1 a -6,3: sem face coplanar com o cano
    branch = cylinder("ramal", 1.6, 6, 24, (4, 1.5, 0), axis=(1, 0, 0))
    body = boolean(boolean(outer, flange, UNION, "cano_flange"), branch, UNION, "cano_ramal")
    bores = [
        cylinder("furo_cano", 2.4, 16, 32),
        cylinder("furo_ramal", 1.1, 8, 24, (4, 1.5, 0), axis=(1, 0, 0)),
    ]
    return finish(cut(body, bores, "cilindros_canos")), {"target_quads": 4000}


def model_03_booleana_ngons():
    """Bloco com rebaixo sextavado, rasgo inclinado e um ressalto: sobra cheio de ngons."""
    body = box("bloco", (12, 6, 10))
    boss = cylinder("ressalto", 2.0, 3, 16, (6.5, 0, 0), axis=(1, 0, 0))
    body = boolean(body, boss, UNION, "bloco_ressalto")
    cutters = [
        cylinder("sextavado", 2.2, 4, 6, (-2.5, 2.0, -1.5)),
        box("rasgo", (14, 2.5, 1.6), (0, 2.6, 3.0), rotate=(0, 30, 0)),
    ]
    return finish(cut(body, cutters, "booleana_ngons")), {"target_quads": 2500}


def model_04_chanfros_fillets():
    """Bloco com as 4 quinas verticais arredondadas (fillet) e o topo chanfrado."""
    import numpy as np

    cmds = _cmds()
    node = box("chanfros_fillets", (12, 6, 8))
    vertical = edges_where(node, lambda p: (np.abs(np.abs(p[:, 0]) - 6) < 1e-4) & (np.abs(np.abs(p[:, 2]) - 4) < 1e-4))
    cmds.polyBevel3(vertical, offset=1.2, offsetAsFraction=False, segments=6, depth=1, chamfer=True, ch=False)
    cmds.delete(node, ch=True)
    top = edges_where(node, lambda p: np.abs(p[:, 1] - 3) < 1e-4, min_angle=60.0)
    cmds.polyBevel3(top, offset=0.6, offsetAsFraction=False, segments=1, depth=1, chamfer=True, ch=False)
    return finish(node), {"target_quads": 2500}


def model_05_simetrica():
    """Suporte simétrico em X: furos, ressaltos e um rasgo central espelhados."""
    body = box("suporte", (16, 3, 8))
    for side in (-1, 1):
        boss = cylinder(f"ressalto{side}", 1.6, 2.0, 20, (5 * side, 2.0, -1.8))
        body = boolean(body, boss, UNION, f"suporte_r{side}")
    cutters = [cylinder(f"furo{side}", 1.1, 6, 20, (5 * side, 0, 1.8)) for side in (-1, 1)]
    cutters.append(box("rasgo", (3, 1.6, 10), (0, 1.0, 0)))
    return finish(cut(body, cutters, "simetrica")), {"target_quads": 3000, "symmetry": "x"}


def model_06_chapa_fina():
    """Chapa fina em L (0,15 de espessura) com três furos: creases opostas quase coladas."""
    t = 0.15
    profile = [(-5.0, 0.0), (5.0, 0.0), (5.0, t), (-5.0 + t, t), (-5.0 + t, 6.0), (-5.0, 6.0)]  # (z, y)
    points = [(x, y, z) for x in (-10.0, 10.0) for z, y in profile]
    n = len(profile)
    faces = [tuple(range(n)), tuple(n + i for i in reversed(range(n)))]  # tampas em L (ngon côncavo)
    faces += [(i, n + i, n + (i + 1) % n, (i + 1) % n) for i in range(n)]
    body = from_arrays("chapa_l", points, faces)
    if not _normals_point_out(body):
        _cmds().polyNormal(body, normalMode=0, userNormalMode=0, ch=False)  # inverte
    cutters = [cylinder(f"furo{i}", 1.0, 2, 24, (-6 + 6 * i, 0, 1.5)) for i in range(3)]
    return finish(cut(body, cutters, "chapa_fina")), {"target_quads": 3000}


def model_07_varias_cascas():
    """Peça com seis cascas soltas no mesmo objeto: caixa furada, tampa e quatro pés."""
    body = cut(box("caixa", (10, 4, 8)), [cylinder("furo", 1.5, 6, 24, (0, 0, 0))], "caixa_furada")
    lid = cylinder("tampa", 3.0, 1.0, 32, (0, 4.0, 0))
    feet = [box(f"pe{i}", (1.5, 1.0, 1.5), (3.5 * sx, -3.0, 2.5 * sz)) for i, (sx, sz) in
            enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)))]  # fmt: skip
    return finish(unite([body, lid] + feet, "varias_cascas")), {"target_quads": 3500}


def model_08_pecas_repetidas():
    """Seis parafusos sextavados idênticos lado a lado (cascas repetidas)."""
    cmds = _cmds()
    head = cylinder("cabeca", 1.2, 0.8, 6, (0, 1.9, 0))
    shank = cylinder("haste", 0.6, 3.2, 16, (0, 0.1, 0))  # vai até 1,7: entra na cabeça (1,5 a 2,3)
    bolt = finish(boolean(head, shank, UNION, "parafuso"))
    copies = [bolt]
    for i in range(1, 6):
        copy = cmds.duplicate(bolt, rr=True, n=f"parafuso{i}")[0]
        cmds.move(3.5 * i, 0, 0, copy)
        copies.append(copy)
    return finish(unite(copies, "pecas_repetidas")), {"target_quads": 4000}


def model_09_500k_faces():
    """Bloco com fillets e furos, subdividido sem mudar a forma até ~500 mil triângulos.

    O bevel é feito na caixa grossa (com malha densa o offset passaria das faces
    vizinhas); a densidade vem depois, por subdivisão linear dos triângulos.
    """
    import math

    import numpy as np

    from toptop import maya_bridge
    from toptop.core.mesh import subdivide_triangles, triangle_mesh

    cmds = _cmds()
    body = box("denso_base", (20, 12, 16))
    vertical = edges_where(body, lambda p: (np.abs(np.abs(p[:, 0]) - 10) < 1e-4) & (np.abs(np.abs(p[:, 2]) - 8) < 1e-4))
    cmds.polyBevel3(vertical, offset=2.0, offsetAsFraction=False, segments=8, depth=1, chamfer=True, ch=False)
    cmds.delete(body, ch=True)
    cutters = [cylinder(f"furo{i}", 1.5, 16, 32, (-5 + 10 * i, 0, 0)) for i in range(2)]
    coarse = finish(cut(body, cutters, "denso_grosso"))
    tri = triangle_mesh(maya_bridge.mesh_from_maya(coarse))
    levels = max(0, round(math.log(500_000 / max(tri.num_faces, 1), 4)))
    dense, _ = subdivide_triangles(tri, levels)
    cmds.delete(coarse)
    print(f"  {tri.num_faces} triângulos x 4^{levels} = {dense.num_faces}")
    return maya_bridge.create_mesh(dense, "denso_500k"), {"target_quads": 6000}


def model_10_quinas_rasas():
    """Placa com vincos de 20 graus no topo: quinas abaixo do ângulo de sugestão."""
    import numpy as np

    from toptop import maya_bridge

    cmds = _cmds()
    node = box("quinas_rasas", (20, 2, 12), subdivisions=(4, 1, 1))
    points = maya_bridge.mesh_from_maya(node).points
    raised = np.flatnonzero((points[:, 1] > 0.5) & (np.abs(np.abs(points[:, 0]) - 5) < 1e-4))
    cmds.move(0, 0.9, 0, [f"{node}.vtx[{i}]" for i in raised.tolist()], relative=True)
    return finish(node), {"target_quads": 1500, "angle": 15.0}


MODELS = [
    ("01_placa_furos", "Placa com furos", model_01_placa_furos),
    ("02_cilindros_canos", "Cilindros e canos", model_02_cilindros_canos),
    ("03_booleana_ngons", "Booleana com ngons", model_03_booleana_ngons),
    ("04_chanfros_fillets", "Peça com chanfros e fillets", model_04_chanfros_fillets),
    ("05_simetrica", "Peça simétrica", model_05_simetrica),
    ("06_chapa_fina", "Chapa fina", model_06_chapa_fina),
    ("07_varias_cascas", "Peça com várias cascas", model_07_varias_cascas),
    ("08_pecas_repetidas", "Peças repetidas (parafusos)", model_08_pecas_repetidas),
    ("09_500k_faces", "Peça de 500 mil faces", model_09_500k_faces),
    ("10_quinas_rasas", "Peça com quinas abaixo de 30 graus", model_10_quinas_rasas),
]


# --------------------------------------------------------------------------
# Gravação
# --------------------------------------------------------------------------


def export_model(node, name, title, options, out_dir):
    """Marca os creases, salva a cena e exporta OBJ triangulado + features."""
    from toptop import maya_bridge
    from toptop.core import phase0
    from toptop.core.creases import maya_read_creases
    from toptop.io.features import features_path, write_features
    from toptop.io.obj import write_obj

    cmds = _cmds()
    node = cmds.rename(node, name)
    strong, guide = mark_creases(node, options.get("angle", 30.0), options.get("guide_planes", ()))

    scene = os.path.join(out_dir, name + ".mb")
    cmds.select(node, replace=True)
    cmds.file(rename=scene)
    cmds.file(save=True, type="mayaBinary", force=True)

    features = maya_read_creases(node)
    copy = cmds.duplicate(node, rr=True, n=name + "_tri")[0]
    cmds.polyTriangulate(copy, ch=False)
    triangles = maya_bridge.mesh_from_maya(copy)
    cmds.delete(copy)
    on_mesh = triangles.topology.edge_ids(features.edges) >= 0
    if not on_mesh.all():
        print(f"  AVISO: {int((~on_mesh).sum())} crease(s) de {name} não são arestas da malha triangulada.")
    obj = os.path.join(out_dir, name + ".obj")
    write_obj(obj, triangles, comment=f"TopTop modelo de teste: {title}")
    write_features(features_path(obj), features, comment=f"{title}: crease >= 2 forte, < 2 guia")

    spec = phase0.ModelSpec(name=name, obj=obj, features=features_path(obj),
                            target_quads=int(options.get("target_quads", 3000)), title=title)  # fmt: skip
    extra = {
        "scene": name + ".mb",
        "triangles": triangles.num_faces,
        "vertices": triangles.num_points,
        "strong_edges": int(features.strong_mask().sum()),
        "guide_edges": int(features.guide_mask().sum()),
        "suggest_angle": options.get("angle", 30.0),
        "symmetry": options.get("symmetry"),
    }
    print(f"  {name}: {triangles.num_faces} triângulos, {strong} creases fortes, {guide} guias")
    return spec, extra


def main(argv=None):
    """Gera os modelos pedidos e o ``models.json``."""
    parser = argparse.ArgumentParser(description="Gera os 10 modelos de teste da Fase 0.")
    parser.add_argument("--out", default=os.path.join(REPO, "tests", "models"), help="pasta de saída")
    parser.add_argument("--only", default="", help="só os modelos cujo nome contém um destes trechos (vírgula)")
    args = parser.parse_args(argv)

    from toptop.core import phase0

    cmds = _cmds()
    os.makedirs(args.out, exist_ok=True)
    wanted = [part.strip() for part in args.only.split(",") if part.strip()]
    specs, extras, failures = [], {}, []
    # mantém no manifesto os modelos já gerados que não estão sendo refeitos agora
    previous = {spec.name: spec for spec in phase0.load_models(args.out)} if os.path.isfile(
        os.path.join(args.out, phase0.MANIFEST)) else {}  # fmt: skip
    for name, title, builder in MODELS:
        if wanted and not any(part in name for part in wanted):
            if name in previous:
                specs.append(previous[name])
            continue
        print(f"Gerando {name} ({title}) ...")
        try:
            cmds.file(new=True, force=True)
            node, options = builder()
            spec, extra = export_model(node, name, title, options, args.out)
            specs.append(spec)
            extras[name] = extra
        except Exception as error:
            failures.append((name, f"{error}"))
            print(f"  FALHOU: {error}")
            traceback.print_exc()
    manifest = phase0.write_manifest(args.out, specs, extras)
    print()
    print(f"{len(specs)} modelo(s) em {args.out}; manifesto: {manifest}")
    for name, message in failures:
        print(f"  não gerado: {name}: {message}")
    return 1 if failures else 0


if __name__ == "__main__":
    import maya.standalone

    maya.standalone.initialize(name="python")
    code = main()
    sys.stdout.flush()
    os._exit(code)
