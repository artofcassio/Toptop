"""Autoverificação do TopTop (seção 10 do plano): pipeline completo nos modelos de teste.

Uso (na raiz do repositório):

    mayapy tests\\run_headless.py                    (compara com a referência salva)
    mayapy tests\\run_headless.py --save-baseline    (grava as métricas atuais como referência)
    python tests\\run_headless.py --selftest         (3 peças sintéticas, só QuadWild, sem Maya)

Para cada modelo de ``tests/models`` (``models.json`` + ``.obj`` + ``.feat``),
roda ``toptop.core.pipeline.run_pipeline`` com a densidade do manifesto,
mede tudo, grava ``<saida>/headless_results.json`` e uma captura de wireframe
por modelo (``<saida>/<modelo>.png``: laranja = hard edges nas creases
fortes, magenta = trecho de crease sem loop).

Regra de aprovação de uma mudança: nenhuma métrica pode piorar mais que
``--tolerance`` (10%) em relação à referência (``tests/baseline/headless.json``).
Saída com código 1 quando alguma piora; com ``--strict``, também quando um
modelo não cumpre as metas. Julgar a topologia nas capturas continua sendo
tarefa sua.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
for _path in (REPO, TESTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_phase0  # noqa: E402  (selftest, Maya e console compartilhados)
from toptop.core import phase0  # noqa: E402
from toptop.core.pipeline import PipelineOptions, run_pipeline  # noqa: E402
from toptop.core.quality import compare_to_baseline, format_report  # noqa: E402
from toptop.core.render import render_wireframe  # noqa: E402
from toptop.engine import ENGINE_NAMES, get_engine  # noqa: E402
from toptop.io.features import read_features  # noqa: E402
from toptop.io.obj import read_obj, write_obj  # noqa: E402

DEFAULT_BASELINE = os.path.join(TESTS, "baseline", "headless.json")
RESULTS_NAME = "headless_results.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description="Autoverificação do TopTop: pipeline completo + comparação com a referência.")
    parser.add_argument("--models-dir", default=os.path.join(TESTS, "models"), help="pasta com models.json, .obj e .feat")
    parser.add_argument("--out", default=os.path.join(TESTS, "output", "headless"), help="pasta de saída")
    parser.add_argument("--engine", default="quadwild", choices=ENGINE_NAMES, help="motor do pipeline")
    parser.add_argument("--mode", default="game", choices=("game", "film"), help="modo do pipeline")
    parser.add_argument("--models", default="", help="filtro de modelos: trechos do nome separados por vírgula")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, help="arquivo de referência das métricas")
    parser.add_argument("--save-baseline", action="store_true", help="grava as métricas desta execução como referência")
    parser.add_argument("--tolerance", type=float, default=0.10, help="piora máxima aceita (fração)")
    parser.add_argument("--strict", action="store_true", help="também reprova quando um modelo não cumpre as metas")
    parser.add_argument("--timeout", type=float, default=600.0, help="tempo limite do motor por casca, em segundos")
    parser.add_argument("--quadwild-dir", default=None, help="pasta com quadwild(.exe) e quad_from_patches(.exe)")
    parser.add_argument("--no-images", action="store_true", help="não grava as capturas de wireframe")
    parser.add_argument("--selftest", action="store_true", help="roda em 3 modelos sintéticos gerados aqui")
    return parser.parse_args(argv)


def run_model(spec: phase0.ModelSpec, args: argparse.Namespace) -> dict:
    """Roda o pipeline num modelo e devolve as métricas (nunca levanta exceção)."""
    entry: dict = {"model": spec.name, "status": "erro", "passed": False, "metrics": {}, "notes": [], "message": ""}
    try:
        mesh = read_obj(spec.obj)
        feats = read_features(spec.features, mesh.num_points)
        engine_options = {"install_dir": args.quadwild_dir} if args.engine == "quadwild" else {}
        engine = get_engine(args.engine, timeout_s=args.timeout, **engine_options)
        available, reason = engine.availability()
        if not available:
            entry.update(status="indisponivel", message=reason)
            return entry
        options = PipelineOptions(mode=args.mode, target_quads=spec.target_quads, engine=engine, timeout_s=args.timeout)
        result = run_pipeline(mesh, feats if len(feats) else None, options)
    except Exception as error:  # um modelo com problema não derruba o lote
        entry["message"] = f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=4)}"
        return entry
    entry.update(
        status="ok",
        passed=bool(result.report.passed),
        metrics=result.report.to_dict(),
        notes=list(result.notes),
        seconds=result.seconds,
        shell_status=list(result.shell_status),
    )
    entry["metrics"]["time_s"] = result.engine_seconds
    folder = args.out
    write_obj(os.path.join(folder, spec.name + ".obj"), result.mesh, comment=f"{spec.name} / pipeline / {args.engine}")
    if not args.no_images:
        try:
            render_wireframe(result.mesh, os.path.join(folder, spec.name + ".png"), strong_edges=result.hard_edges)
        except Exception as error:
            entry["notes"].append(f"sem captura: {error}")
    entry["report_text"] = format_report(result.report)
    return entry


def check_against_baseline(
    entries: list[dict], baseline: dict, tolerance: float
) -> tuple[dict[str, list[str]], list[str]]:
    """Pioras por modelo (só modelos com referência) e modelos da referência que agora falharam."""
    regressions: dict[str, list[str]] = {}
    broken: list[str] = []
    reference = baseline.get("models", {})
    for entry in entries:
        base = reference.get(entry["model"])
        if base is None:
            continue
        if entry["status"] != "ok":
            broken.append(entry["model"])
            continue
        worse = compare_to_baseline(entry["metrics"], base, tolerance)
        if worse:
            regressions[entry["model"]] = worse
    return regressions, broken


def load_baseline(path: str) -> dict:
    """Referência gravada (vazia se ainda não existe)."""
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_baseline(path: str, entries: list[dict], previous: dict, engine: str) -> int:
    """Grava as métricas dos modelos que rodaram; mantém os outros da referência anterior."""
    models = dict(previous.get("models", {}))
    count = 0
    for entry in entries:
        if entry["status"] == "ok":
            models[entry["model"]] = entry["metrics"]
            count += 1
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"version": 1, "engine": engine, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "models": models}, handle, indent=2, ensure_ascii=False)  # fmt: skip
    return count


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada; devolve 1 quando a mudança deve ser reprovada."""
    run_phase0.safe_console()
    args = parse_args(argv)
    args.out = os.path.abspath(args.out)
    args.models_dir = os.path.abspath(args.models_dir)
    if args.quadwild_dir:
        args.quadwild_dir = os.path.abspath(args.quadwild_dir)
    os.makedirs(args.out, exist_ok=True)
    if args.selftest:
        args.models_dir = run_phase0.selftest_models(os.path.join(args.out, "selftest_models"))
        if args.baseline == DEFAULT_BASELINE:
            args.baseline = os.path.join(args.out, "selftest_baseline.json")
    specs = phase0.load_models(args.models_dir)
    wanted = [part.strip() for part in args.models.split(",") if part.strip()]
    if wanted:
        specs = [spec for spec in specs if any(part in spec.name for part in wanted)]
    if not specs:
        print(f"Nenhum modelo em {args.models_dir}. Rode antes: mayapy tests/make_test_models.py")
        return 2
    if get_engine(args.engine).requires_maya:
        if not run_phase0.have_maya():
            print(f"O motor {args.engine} só roda no mayapy.")
            return 2
        run_phase0.init_maya()

    print(f"TopTop autoverificação: {len(specs)} modelo(s), motor {args.engine}, modo {args.mode}")
    entries = []
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec.name} ...", flush=True)
        entry = run_model(spec, args)
        entries.append(entry)
        if entry["status"] == "ok":
            print(f"    {'APROVADO' if entry['passed'] else 'REPROVADO nas metas'}  ({entry['seconds']:.1f} s)")
        else:
            print(f"    {entry['status']}: {entry['message'].splitlines()[0] if entry['message'] else ''}")

    baseline = load_baseline(args.baseline)
    regressions, broken = check_against_baseline(entries, baseline, args.tolerance)
    results_path = os.path.join(args.out, RESULTS_NAME)
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "engine": args.engine, "mode": args.mode,
                   "baseline": args.baseline, "tolerance": args.tolerance, "regressions": regressions,
                   "broken": broken, "results": entries}, handle, indent=2, ensure_ascii=False)  # fmt: skip

    print()
    failed_targets = [e["model"] for e in entries if e["status"] == "ok" and not e["passed"]]
    not_run = [e["model"] for e in entries if e["status"] != "ok"]
    if not baseline:
        print(f"Sem referência em {args.baseline}: nada a comparar (use --save-baseline para criar).")
    for model, items in regressions.items():
        print(f"PIOROU {model}: " + "; ".join(items))
    for model in broken:
        print(f"PIOROU {model}: tinha referência e agora não rodou")
    if failed_targets:
        print("Não cumprem as metas: " + ", ".join(failed_targets))
    if not_run:
        print("Não rodaram: " + ", ".join(not_run))
    if args.save_baseline:
        count = save_baseline(args.baseline, entries, baseline, args.engine)
        print(f"Referência gravada com {count} modelo(s): {args.baseline}")
    reject = bool(regressions or broken or (args.strict and (failed_targets or not_run)))
    print(f"Resultado: {'REPROVADO' if reject else 'APROVADO'}  (detalhes em {results_path})")
    return 1 if reject else 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        if run_phase0.have_maya() and getattr(run_phase0.init_maya, "done", False):
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
    sys.exit(code)
