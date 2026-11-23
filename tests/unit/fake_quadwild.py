"""Instalação falsa do QuadWild para os testes (dois scripts Python com shebang)."""

import os
import stat
import sys
import textwrap

from toptop.engine import quadwild as qw


def fake_install(folder, mode="ok", quads_at_scale_one=400):
    """Cria uma instalação falsa do QuadWild em ``folder``."""
    os.makedirs(os.path.join(folder, "config", "main_config"))
    os.makedirs(os.path.join(folder, "config", "satsuma"))
    with open(os.path.join(folder, "config", "main_config", "flow.txt"), "w") as f:
        f.write(qw.FLOW_TEMPLATE)
    with open(os.path.join(folder, "config", "satsuma", "default.json"), "w") as f:
        f.write("{}")
    shebang = "#!" + sys.executable + "\n"
    trace = shebang + textwrap.dedent(
        f"""
        import os, sys, time
        MODE = {mode!r}
        args = sys.argv[1:]
        assert args[:3] == ["mesh.obj", "2", "prep.txt"], args
        assert all(os.sep not in a and "/" not in a for a in args), "só nomes simples"
        assert open("prep.txt").read().split()[::2] == ["do_remesh", "sharp_feature_thr", "alpha", "scaleFact"]
        faces = [l for l in open("mesh.obj") if l.startswith("f ")]
        assert all(len(l.split()) == 4 for l in faces), "só triângulos"
        if "mesh.sharp" in args:
            lines = open("mesh.sharp").read().split()
            assert int(lines[0]) == len(lines) - 1
            for line in lines[1:]:
                kind, face, edge = map(int, line.split(","))
                assert kind in (0, 1) and 0 <= face < len(faces) and 0 <= edge < 3
        print("fake quadwild", MODE, len(faces))
        if MODE == "hang":
            time.sleep(60)
        if MODE == "fail":
            print("ERROR: Wrong Sharp Feature File")
            sys.exit(0)
        open("mesh_rem_p0.obj", "w").write(open("mesh.obj").read())
        for ext in (".patch", ".corners", ".feature", ".c_feature"):
            open("mesh_rem_p0" + ext, "w").write("1\\n")
        if MODE == "hang_after_output":
            time.sleep(60)  # como um crash preso atrás da caixa de erro do Windows
        """
    )
    patches = shebang + textwrap.dedent(
        f"""
        import os, sys
        MODE = {mode!r}
        mesh, num, config = sys.argv[1:4]
        assert mesh == "mesh_rem_p0.obj" and config == "main.txt"
        lines = open(config).read().splitlines()
        assert len(lines) == 23 and lines[0].startswith("alpha ") and lines[20] == "useFlowSolver 1"
        assert os.path.isfile(lines[22].split('"')[1]), "config do satsuma relativo à pasta atual"
        scale = float(lines[18].split()[1])
        pts = [list(map(float, l.split()[1:4])) for l in open(mesh) if l.startswith("v ")]
        lo = [min(p[i] for p in pts) for i in range(3)]
        hi = [max(p[i] for p in pts) for i in range(3)]
        n = max(1, round(({quads_at_scale_one} / scale ** 2) ** 0.5))
        out = []
        for i in range(n + 1):
            for j in range(n + 1):
                out.append("v %r %r %r" % (lo[0] + (hi[0] - lo[0]) * i / n, lo[1] + (hi[1] - lo[1]) * j / n, hi[2]))
        for i in range(n):
            for j in range(n):
                a = i * (n + 1) + j + 1
                out.append("f %d %d %d %d" % (a, a + n + 1, a + n + 2, a + 1))
        text = "\\n".join(out) + "\\n"
        stem = "mesh_rem_p0_%s_quadrangulation" % num
        open(stem + ".obj", "w").write(text)
        if MODE == "half_smooth":
            text = "\\n".join(out[: len(out) - 3]) + "\\n"
        if MODE != "no_smooth":
            open(stem + "_smooth.obj", "w").write(text)
        open("stats_%s.json" % num, "w").write("{{}}")
        print("fake quad_from_patches", scale, n * n)
        if MODE == "crash_at_exit":
            os._exit(139)
        if MODE == "hang_after_output":
            import time
            time.sleep(60)
        """
    )
    for name, body in ((qw.QUADWILD_EXE, trace), (qw.PATCHES_EXE, patches)):
        path = os.path.join(folder, name)
        with open(path, "w") as f:
            f.write(body)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return folder
