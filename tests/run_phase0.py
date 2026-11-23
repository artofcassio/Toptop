"""Fase 0: roda cada modelo de teste em cada motor e grava as métricas.

Uso no Windows (na raiz do repositório):

    mayapy tests\\run_phase0.py
    mayapy tests\\run_phase0.py --engines quadwild,maya_retopo --models 01,04
    python tests\\run_phase0.py --selftest          (só QuadWild; não precisa do Maya)

Para cada modelo x motor: roda o motor com a mesma densidade (``target_quads``
do ``models.json``), mede tudo com ``toptop.core.quality``, grava o resultado
em OBJ e uma captura de wireframe em PNG, e no fim escreve
``phase0_results.csv`` e ``phase0_results.json`` na pasta de saída.

* QuadWild: executável externo; não precisa do Maya.
* Retopologize: cada execução roda num ``mayapy`` separado, para que um
  travamento ou tempo limite não derrube o lote (``--in-process`` desliga isso).
* Quad Remesher: nunca é chamado daqui (a licença não cobre lote). Rode-o à mão
  no Maya e grave cada resultado com ``toptop.engine.quadremesher.collect_manual``
  em ``<saida>/manual/quadremesher/<modelo>.obj``; aqui ele só é medido.

Motor indisponível é pulado com aviso, não com erro.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from toptop.core import phase0  # noqa: E402
from toptop.core.creases import FeatureSet, suggest_creases  # noqa: E402
from toptop.core.mesh import triangle_mesh  # noqa: E402
from toptop.engine import ENGINE_NAMES  # noqa: E402
from toptop.io.features import features_path, write_features  # noqa: E402
from toptop.io.obj import write_obj  # noqa: E402

MAYA_ENGINES = ("maya_retopo",)
MANUAL_ENGINES = ("quadremesher",)
WORKER_MARGIN_S = 600.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Fase 0 do TopTop: compara os motores nos modelos de teste.")
    parser.add_argument("--models-dir", default=os.path.join(REPO, "tests", "models"), help="pasta com models.json, .obj e .feat")
    parser.add_argument("--out", default=os.path.join(REPO, "tests", "output", "phase0"), help="pasta de saída")
    parser.add_argument("--engines", default=",".join(ENGINE_NAMES), help="motores, separados por vírgula")
    parser.add_argument("--models", default="", help="filtro de modelos: trechos do nome separados por vírgula")
    parser.add_argument("--timeout", type=float, default=600.0, help="tempo limite do motor, em segundos")
    parser.add_argument("--samples", type=int, default=50_000, help="amostras por sentido no desvio de forma")
    parser.add_argument("--quadwild-dir", default=None, help="pasta com quadwild(.exe) e quad_from_patches(.exe)")
    parser.add_argument("--quadwild-config", default="flow", help="config principal do QuadWild (flow, flow_noalign, ...)")
    parser.add_argument("--quadwild-no-remesh", action="store_true",
                        help="desliga o pré-remesh do QuadWild (use se ele apagar detalhe da peça)")
    parser.add_argument("--append", action="store_true",
                        help="mantém as linhas de motor x modelo da execução anterior que não forem refeitas agora")
    parser.add_argument("--in-process", action="store_true", help="roda os motores do Maya neste processo, sem isolar")
    parser.add_argument("--no-images", action="store_true", help="não grava as capturas de wireframe")
    parser.add_argument("--excel-br", action="store_true", help="CSV com ponto e vírgula e vírgula decimal")
    parser.add_argument("--selftest", action="store_true", help="roda em 3 modelos sintéticos gerados aqui (sem Maya)")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def have_maya() -> bool:
    """Verdadeiro se este Python é o do Maya (``maya`` importável)."""
    try:
        return importlib.util.find_spec("maya") is not None
    except (ImportError, ValueError):
        return False


def init_maya() -> None:
    """Inicia o Maya sem interface (uma vez por processo)."""
    import maya.standalone

    if not getattr(init_maya, "done", False):
        maya.standalone.initialize(name="python")
        init_maya.done = True  # type: ignore[attr-defined]


def mayapy_executable() -> str | None:
    """Caminho do mayapy: este próprio Python ou o mayapy ao lado do maya.exe."""
    exe = sys.executable or ""
    base = os.path.basename(exe).lower()
    if base.startswith("mayapy"):
        return exe
    for name in ("mayapy.exe", "mayapy"):
        candidate = os.path.join(os.path.dirname(exe), name)
        if os.path.isfile(candidate):
            return candidate
    return None


def engine_options(engine: str, args: argparse.Namespace) -> dict:
    """Opções de construção de cada motor."""
    if engine == "quadwild":
        return {
            "install_dir": args.quadwild_dir,
            "main_config": args.quadwild_config,
            "do_remesh": not getattr(args, "quadwild_no_remesh", False),
        }
    return {}


def previous_rows(out_dir: str, redo: set[tuple[str, str]]) -> list[dict]:
    """Linhas da execução anterior (``phase0_results.json``) que não serão refeitas agora."""
    path = os.path.join(out_dir, "phase0_results.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            rows = json.load(handle).get("results", [])
    except (OSError, ValueError):
        print(f"AVISO: não consegui ler {path}; --append ignorado.")
        return []
    return [row for row in rows if (row.get("model"), row.get("engine")) not in redo]


def run_job(spec: phase0.ModelSpec, engine: str, args: argparse.Namespace) -> dict:
    """Roda um modelo num motor neste processo."""
    return phase0.run_one(
        spec,
        engine,
        args.out,
        timeout_s=args.timeout,
        engine_options=engine_options(engine, args),
        images=not args.no_images,
        samples=args.samples,
        manual_only=engine in MANUAL_ENGINES,
    )


def worker_main(job_path: str) -> None:
    """Processo filho: inicia o Maya, roda um trabalho e grava a linha em JSON."""
    with open(job_path, "r", encoding="utf-8") as handle:
        job = json.load(handle)
    args = argparse.Namespace(**job["args"])
    spec = phase0.ModelSpec(**job["spec"])
    try:
        init_maya()
        row = run_job(spec, job["engine"], args)
    except Exception as error:  # qualquer falha vira linha de erro, não um processo morto
        row = {key: None for key in phase0.CSV_COLUMNS}
        row.update(model=spec.name, engine=job["engine"], status=phase0.STATUS_ERROR, passed=False,
                   target_quads=spec.target_quads, message=f"falha no processo do Maya: {error}")  # fmt: skip
    with open(job["result"], "w", encoding="utf-8") as handle:
        json.dump(row, handle)
    sys.stdout.flush()
    os._exit(0)  # o encerramento normal do Maya standalone às vezes trava


def run_in_worker(spec: phase0.ModelSpec, engine: str, args: argparse.Namespace) -> dict:
    """Roda um trabalho num mayapy separado, com tempo limite."""
    row = {key: None for key in phase0.CSV_COLUMNS}
    row.update(model=spec.name, engine=engine, status=phase0.STATUS_ERROR, passed=False,
               target_quads=spec.target_quads, message="")  # fmt: skip
    mayapy = mayapy_executable()
    if mayapy is None:
        row.update(status=phase0.STATUS_UNAVAILABLE, message="mayapy não encontrado")
        return row
    folder = os.path.join(args.out, "_jobs")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, f"{spec.name}__{engine}")
    job = {"spec": spec.__dict__, "engine": engine, "result": stem + ".result.json",
           "args": {k: v for k, v in vars(args).items() if k != "worker"}}  # fmt: skip
    for path in (job["result"],):
        if os.path.exists(path):
            os.remove(path)
    with open(stem + ".job.json", "w", encoding="utf-8") as handle:
        json.dump(job, handle)
    start = time.perf_counter()
    with open(stem + ".log.txt", "wb") as log:
        proc = subprocess.Popen(
            [mayapy, os.path.abspath(__file__), "--worker", stem + ".job.json"],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=REPO,
        )  # fmt: skip
        try:
            proc.wait(timeout=args.timeout + WORKER_MARGIN_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            row.update(status=phase0.STATUS_TIMEOUT, time_s=time.perf_counter() - start,
                       message=f"processo do Maya encerrado depois de {args.timeout + WORKER_MARGIN_S:.0f} s")  # fmt: skip
            return row
    if os.path.isfile(job["result"]):
        with open(job["result"], "r", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("status") == phase0.STATUS_OK and (result.get("time_s") or 0) > args.timeout:
            result.update(status=phase0.STATUS_TIMEOUT, passed=False,
                          message=f"terminou, mas passou do tempo limite de {args.timeout:.0f} s")  # fmt: skip
        return result
    tail = b""
    try:
        with open(stem + ".log.txt", "rb") as handle:
            tail = handle.read()[-1500:]
    except OSError:
        pass
    row["message"] = f"o mayapy saiu com código {proc.returncode} sem resultado: {tail.decode('utf-8', 'replace')}"
    return row


def selftest_models(folder: str) -> str:
    """Gera 3 modelos sintéticos simples (cubo, cilindro, bloco) para testar a instalação."""
    sys.path.insert(0, os.path.join(REPO, "tests", "unit"))
    import shapes

    os.makedirs(folder, exist_ok=True)
    block = shapes.cube(8)
    block = block.with_points(block.points * (4.0, 0.6, 2.5))
    sources = [
        ("cubo", shapes.cube(6), 600),
        ("cilindro", shapes.cylinder(48, rings=6, radius=1.0, height=3.0, caps="fan"), 800),
        ("bloco", block, 900),
    ]
    specs = []
    for name, source, target in sources:
        tri = triangle_mesh(source)
        obj = os.path.join(folder, name + ".obj")
        write_obj(obj, tri)
        write_features(features_path(obj), FeatureSet(suggest_creases(tri, 30.0)))
        specs.append(phase0.ModelSpec(name=name, obj=obj, features=features_path(obj), target_quads=target))
    phase0.write_manifest(folder, specs)
    return folder


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada."""
    args = parse_args(argv)
    if args.worker:
        worker_main(args.worker)
        return 0
    os.makedirs(args.out, exist_ok=True)
    if args.selftest:
        args.models_dir = selftest_models(os.path.join(args.out, "selftest_models"))
    engines = [name.strip() for name in args.engines.split(",") if name.strip()]
    unknown = [name for name in engines if name not in ENGINE_NAMES]
    if unknown:
        print(f"Motor desconhecido: {', '.join(unknown)}. Opções: {', '.join(ENGINE_NAMES)}")
        return 2
    specs = phase0.load_models(args.models_dir)
    wanted = [part.strip() for part in args.models.split(",") if part.strip()]
    if wanted:
        specs = [spec for spec in specs if any(part in spec.name for part in wanted)]
    if not specs:
        print(f"Nenhum modelo em {args.models_dir}. Rode antes: mayapy tests/make_test_models.py")
        return 2

    maya = have_maya()
    print(f"TopTop Fase 0: {len(specs)} modelo(s) x {len(engines)} motor(es); saída em {args.out}")
    if not maya and any(name in MAYA_ENGINES for name in engines):
        print("AVISO: este Python não é o mayapy; os motores do Maya serão pulados.")

    rows: list[dict] = []
    if args.append:
        rows = previous_rows(args.out, {(spec.name, engine) for spec in specs for engine in engines})
        if rows:
            print(f"--append: mantendo {len(rows)} linha(s) da execução anterior.")
    total = len(specs) * len(engines)
    count = 0
    for spec in specs:
        if not args.no_images:
            try:
                phase0.render_input(spec, args.out)
            except Exception as error:
                print(f"  (sem imagem da entrada de {spec.name}: {error})")
        for engine in engines:
            count += 1
            print(f"[{count}/{total}] {spec.name} x {engine} ...", flush=True)
            start = time.perf_counter()
            if engine in MAYA_ENGINES and maya and not args.in_process:
                row = run_in_worker(spec, engine, args)
            else:
                if engine in MAYA_ENGINES and maya:
                    init_maya()
                row = run_job(spec, engine, args)
            rows.append(row)
            verdict = "PASSOU" if row.get("passed") else row["status"]
            print(f"    {verdict}  ({time.perf_counter() - start:.1f} s no total)"
                  + (f"  {str(row['message']).splitlines()[0][:150]}" if row.get("message") else ""), flush=True)  # fmt: skip
            # grava a cada passo: se o lote for interrompido, o que já rodou fica
            phase0.write_csv(rows, os.path.join(args.out, "phase0_results.csv"), args.excel_br)
            phase0.write_json(rows, os.path.join(args.out, "phase0_results.json"))

    print()
    print(phase0.summary_text(rows))
    print()
    print(f"Tabela: {os.path.join(args.out, 'phase0_results.csv')}")
    print(f"Dados:  {os.path.join(args.out, 'phase0_results.json')}")
    print(f"Wireframes e malhas: {args.out}{os.sep}<motor>{os.sep}<modelo>.png / .obj")
    return 0


if __name__ == "__main__":
    code = main()
    if have_maya() and getattr(init_maya, "done", False):
        sys.stdout.flush()
        os._exit(code)
    sys.exit(code)
