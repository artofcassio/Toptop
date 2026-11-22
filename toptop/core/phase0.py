"""Fase 0: rodar modelo x motor, medir, gravar CSV/JSON e decidir o portão.

Tudo aqui roda sem Maya. ``tests/run_phase0.py`` é só a linha de comando por
cima deste módulo (e quem cuida de iniciar o Maya quando o motor precisa).

Protocolo (seção 3 do plano): os mesmos modelos, as mesmas creases, densidade
equivalente (o mesmo número de quads pedido a todos os motores). Segue o motor
que cumprir as metas em pelo menos 8 de 10 modelos; quem passar em mais modelos
ganha, com desempate pela fidelidade das quinas e depois pelo tempo.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any

from ..engine import get_engine
from ..engine.base import EngineCancelled, EngineError, EngineTimeout
from ..io.features import features_path, read_features
from ..io.obj import read_obj, write_obj
from .creases import DEFAULT_STRONG_THRESHOLD
from .mesh import Mesh
from .quality import QualityReport, Targets, evaluate
from .render import render_wireframe

STATUS_OK = "ok"
STATUS_MANUAL = "manual"
STATUS_ERROR = "erro"
STATUS_TIMEOUT = "tempo_limite"
STATUS_UNAVAILABLE = "indisponivel"
#: situações em que o motor foi de fato executado (contam para a robustez)
ATTEMPTED = (STATUS_OK, STATUS_MANUAL, STATUS_ERROR, STATUS_TIMEOUT)
FINISHED = (STATUS_OK, STATUS_MANUAL)

MANIFEST = "models.json"
DEFAULT_TARGET_QUADS = 3000
MAX_INPUT_RENDER_TRIANGLES = 300_000

CSV_COLUMNS = [
    "model", "engine", "status", "passed", "time_s", "target_quads", "faces", "quads", "quad_ratio",
    "crease_coverage", "crease_coverage_strong", "feature_fidelity", "feature_vertex_dev",
    "feature_corner_dev", "feature_chord_dev", "corners_matched", "corners_total", "hausdorff",
    "dev_new_to_ref_max", "dev_ref_to_new_max", "aspect_median", "aspect_p95", "aspect_max", "poles",
    "pole_ratio", "poles_on_strong_crease", "check_quads", "check_crease_coverage",
    "check_feature_fidelity", "check_shape_deviation", "check_aspect_ratio", "check_poles", "check_time",
    "vertices", "triangles", "ngons", "shells", "boundary_edges", "nonmanifold_edges", "ref_triangles",
    "diagonal", "image", "mesh", "message",
]  # fmt: skip


@dataclass
class ModelSpec:
    """Um modelo de teste: o OBJ, as creases e a densidade pedida."""

    name: str
    obj: str
    features: str
    target_quads: int = DEFAULT_TARGET_QUADS
    title: str = ""
    notes: str = ""


def load_models(models_dir: str) -> list[ModelSpec]:
    """Modelos de ``models_dir``: pelo ``models.json`` ou, sem ele, todo ``*.obj``."""
    manifest = os.path.join(models_dir, MANIFEST)
    specs: list[ModelSpec] = []
    if os.path.isfile(manifest):
        with open(manifest, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        for item in data.get("models", []):
            obj = os.path.join(models_dir, item.get("obj", item["name"] + ".obj"))
            feat = os.path.join(models_dir, item["features"]) if item.get("features") else features_path(obj)
            specs.append(
                ModelSpec(
                    name=item["name"],
                    obj=obj,
                    features=feat,
                    target_quads=int(item.get("target_quads", DEFAULT_TARGET_QUADS)),
                    title=item.get("title", ""),
                    notes=item.get("notes", ""),
                )
            )
        return specs
    for obj in sorted(glob.glob(os.path.join(models_dir, "*.obj"))):
        name = os.path.splitext(os.path.basename(obj))[0]
        specs.append(ModelSpec(name=name, obj=obj, features=features_path(obj)))
    return specs


def write_manifest(models_dir: str, specs: list[ModelSpec], extra: dict[str, dict[str, Any]] | None = None) -> str:
    """Grava o ``models.json`` (caminhos relativos à pasta dos modelos)."""
    items = []
    for spec in specs:
        item = asdict(spec)
        item["obj"] = os.path.relpath(spec.obj, models_dir).replace(os.sep, "/")
        item["features"] = os.path.relpath(spec.features, models_dir).replace(os.sep, "/")
        item.update((extra or {}).get(spec.name, {}))
        items.append(item)
    path = os.path.join(models_dir, MANIFEST)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"version": 1, "models": items}, handle, indent=2, ensure_ascii=False)
    return path


def manual_result_path(out_dir: str, engine: str, model: str) -> str:
    """Onde fica o OBJ de um resultado feito à mão (por exemplo, Quad Remesher)."""
    return os.path.join(out_dir, "manual", engine, model + ".obj")


def _manual_seconds(obj_path: str) -> float | None:
    """Tempo anotado ao lado de um resultado manual (``<modelo>.time.txt``), se houver."""
    path = os.path.splitext(obj_path)[0] + ".time.txt"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return float(handle.read().strip().replace(",", "."))
    except (OSError, ValueError):
        return None


def _render(mesh: Mesh, path: str, match: Any) -> str:
    """Captura de wireframe com os loops de crease e os trechos sem loop."""
    if match is not None and match.applicable:
        return render_wireframe(
            mesh,
            path,
            strong_edges=match.matched_edges[match.matched_strong],
            guide_edges=match.matched_edges[~match.matched_strong],
            missing_points=match.uncovered_points,
        )
    return render_wireframe(mesh, path)


def render_input(spec: ModelSpec, out_dir: str, strong_threshold: float = DEFAULT_STRONG_THRESHOLD) -> str | None:
    """Imagem do modelo de entrada com as creases (laranja = forte, azul = guia)."""
    mesh = read_obj(spec.obj)
    if int((mesh.counts - 2).sum()) > MAX_INPUT_RENDER_TRIANGLES:
        return None
    feats = read_features(spec.features, mesh.num_points)
    strong = feats.strong_mask(strong_threshold)
    folder = os.path.join(out_dir, "_entrada")
    os.makedirs(folder, exist_ok=True)
    return render_wireframe(
        mesh, os.path.join(folder, spec.name + ".png"),
        strong_edges=feats.edges[strong], guide_edges=feats.edges[~strong], show_poles=False,
    )  # fmt: skip


def run_one(
    spec: ModelSpec,
    engine_name: str,
    out_dir: str,
    *,
    timeout_s: float = 600.0,
    engine_options: dict[str, Any] | None = None,
    images: bool = True,
    samples: int = 50_000,
    targets: Targets | None = None,
    manual_only: bool = False,
) -> dict[str, Any]:
    """Roda um motor num modelo, mede e devolve a linha de resultado.

    Nunca levanta exceção: falha, tempo limite e motor indisponível viram
    ``status`` e ``message`` na linha. O resultado do motor é gravado em
    ``<out_dir>/<motor>/<modelo>.obj`` e a captura em ``.png`` ao lado. Com
    ``manual_only`` o motor nunca é chamado: só é medido o resultado feito à
    mão, se existir (é o caso do Quad Remesher, cuja licença não cobre lote).
    """
    row: dict[str, Any] = {key: None for key in CSV_COLUMNS}
    row.update(model=spec.name, engine=engine_name, status=STATUS_ERROR, passed=False,
               target_quads=spec.target_quads, message="")  # fmt: skip
    folder = os.path.join(out_dir, engine_name)
    try:
        os.makedirs(folder, exist_ok=True)
        mesh = read_obj(spec.obj)
        feats = read_features(spec.features, mesh.num_points)
    except Exception as error:
        row["message"] = f"não foi possível ler o modelo: {error}"
        return row

    result: Mesh | None = None
    seconds: float | None = None
    try:
        engine = get_engine(engine_name, timeout_s=timeout_s, **(engine_options or {}))
        if manual_only:
            available, reason = False, "motor de uso manual: rode pela janela do Maya e grave o resultado"
        else:
            available, reason = engine.availability()
    except Exception as error:
        row["message"] = f"não foi possível criar o motor: {error}"
        return row

    manual = manual_result_path(out_dir, engine_name, spec.name)
    if available:
        try:
            result = engine.remesh(mesh, feats, spec.target_quads)
            seconds = engine.last_run.seconds
            row["status"] = STATUS_OK
            row["message"] = str(engine.last_run.details.get("warning", ""))
        except EngineTimeout as error:
            row.update(status=STATUS_TIMEOUT, message=str(error), time_s=engine.last_run.seconds)
            return row
        except (EngineError, EngineCancelled) as error:
            row.update(status=STATUS_ERROR, message=str(error), time_s=engine.last_run.seconds)
            return row
        except Exception as error:  # erro de programação no adaptador: não derruba o lote
            row.update(status=STATUS_ERROR, message=f"{error}\n{traceback.format_exc(limit=3)}")
            return row
    elif os.path.isfile(manual):
        try:
            result = read_obj(manual)
        except Exception as error:
            row["message"] = f"resultado manual ilegível ({manual}): {error}"
            return row
        seconds = _manual_seconds(manual)
        row.update(status=STATUS_MANUAL, message=f"resultado manual lido de {manual}")
    else:
        row.update(status=STATUS_UNAVAILABLE, message=f"{reason} (resultado manual esperado em {manual})")
        return row

    try:
        mesh_path = os.path.join(folder, spec.name + ".obj")
        write_obj(mesh_path, result, comment=f"{spec.name} / {engine_name}")
        details: dict[str, Any] = {}
        report = evaluate(
            result, mesh, feats, time_s=seconds, targets=targets, samples=samples,
            name=spec.name, engine=engine_name, details=details,
        )  # fmt: skip
        fill_row(row, report)
        row["mesh"] = mesh_path
        if images:
            row["image"] = _render(result, os.path.join(folder, spec.name + ".png"), details.get("match"))
    except Exception as error:
        row.update(status=STATUS_ERROR, passed=False,
                   message=f"falha ao medir o resultado: {error}\n{traceback.format_exc(limit=3)}")  # fmt: skip
    return row


def fill_row(row: dict[str, Any], report: QualityReport) -> None:
    """Copia as métricas do relatório para a linha de resultado."""
    data = report.to_dict()
    for key in CSV_COLUMNS:
        if key in data and key not in ("message",):
            row[key] = data[key]
    for key, value in data["checks"].items():
        row["check_" + key] = value
    row["passed"] = bool(report.passed) and row["status"] in FINISHED
    notes = "; ".join(report.notes)
    if notes:
        row["message"] = (row["message"] + "; " if row["message"] else "") + notes


def _cell(value: Any, decimal: str) -> str:
    """Valor de uma célula do CSV."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        text = repr(round(value, 9)) if math.isfinite(value) else ""
        return text.replace(".", decimal)
    return str(value).replace("\r", " ").replace("\n", " | ")


def write_csv(rows: list[dict[str, Any]], path: str, excel_br: bool = False) -> str:
    """Grava as linhas em CSV. ``excel_br``: ponto e vírgula e vírgula decimal."""
    with open(path, "w", encoding="utf-8-sig" if excel_br else "utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";" if excel_br else ",")
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow([_cell(row.get(key), "," if excel_br else ".") for key in CSV_COLUMNS])
    return path


def gate(rows: list[dict[str, Any]], min_passed: int | None = None) -> dict[str, Any]:
    """Resumo por motor e o motor escolhido pelo portão da Fase 0.

    ``min_passed`` padrão: 8 de 10 (ou 80% arredondado para cima, se a rodada
    não tiver 10 modelos).
    """
    models = sorted({row["model"] for row in rows})
    if min_passed is None:
        min_passed = 8 if len(models) == 10 else max(1, math.ceil(0.8 * len(models)))
    engines: dict[str, dict[str, Any]] = {}
    for name in sorted({row["engine"] for row in rows}):
        mine = [row for row in rows if row["engine"] == name]
        attempted = [row for row in mine if row["status"] in ATTEMPTED]
        finished = [row for row in mine if row["status"] in FINISHED]
        fidelity = [row["feature_fidelity"] for row in finished if row.get("feature_fidelity") is not None]
        times = [row["time_s"] for row in finished if row.get("time_s") is not None]
        passed = sum(1 for row in mine if row.get("passed"))
        engines[name] = {
            "models": len(mine),
            "attempted": len(attempted),
            "finished": len(finished),
            "passed": passed,
            "failed_models": sorted(row["model"] for row in attempted if not row.get("passed")),
            "mean_feature_fidelity": sum(fidelity) / len(fidelity) if fidelity else None,
            "total_time_s": sum(times) if times else None,
            "qualifies": bool(attempted) and passed >= min_passed,
        }
    ranking = sorted(
        (name for name, info in engines.items() if info["qualifies"]),
        key=lambda name: (
            -engines[name]["passed"],
            engines[name]["mean_feature_fidelity"] if engines[name]["mean_feature_fidelity"] is not None else math.inf,
            engines[name]["total_time_s"] if engines[name]["total_time_s"] is not None else math.inf,
            name,
        ),
    )
    return {
        "models": len(models),
        "min_passed": min_passed,
        "engines": engines,
        "ranking": ranking,
        "chosen": ranking[0] if ranking else None,
    }


def write_json(rows: list[dict[str, Any]], path: str, extra: dict[str, Any] | None = None) -> str:
    """Grava linhas + resumo do portão em JSON."""
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "targets": asdict(Targets()),
        "gate": gate(rows),
        "results": rows,
    }
    payload.update(extra or {})
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def _fmt(value: Any, scale: float = 1.0, digits: int = 2, suffix: str = "") -> str:
    """Número curto para a tabela de resumo."""
    if value is None:
        return "-"
    return f"{value * scale:.{digits}f}{suffix}"


def summary_text(rows: list[dict[str, Any]]) -> str:
    """Tabela de resultados e decisão do portão, em texto."""
    header = (
        f"{'modelo':<24}{'motor':<14}{'status':<14}{'quads':>7}{'%quad':>7}{'cobert':>8}"
        f"{'fidel%':>8}{'desvio%':>9}{'prop':>6}{'polos%':>8}{'tempo':>8}  metas"
    )
    lines = [header, "-" * len(header)]
    for row in sorted(rows, key=lambda r: (r["model"], r["engine"])):
        lines.append(
            f"{row['model']:<24}{row['engine']:<14}{row['status']:<14}"
            f"{row['quads'] if row.get('quads') is not None else '-':>7}"
            f"{_fmt(row.get('quad_ratio'), 100, 0):>7}{_fmt(row.get('crease_coverage'), 100, 1):>8}"
            f"{_fmt(row.get('feature_fidelity'), 100, 3):>8}{_fmt(row.get('hausdorff'), 100, 3):>9}"
            f"{_fmt(row.get('aspect_median')):>6}{_fmt(row.get('pole_ratio'), 100, 1):>8}"
            f"{_fmt(row.get('time_s'), 1, 1, 's'):>8}  {'PASSOU' if row.get('passed') else 'não'}"
        )
    summary = gate(rows)
    lines.append("")
    lines.append(f"Portão: metas cumpridas em pelo menos {summary['min_passed']} de {summary['models']} modelos")
    for name, info in summary["engines"].items():
        lines.append(
            f"  {name:<14} passou em {info['passed']}/{info['models']}"
            f" (rodou {info['attempted']}, terminou {info['finished']})"
            f"  fidelidade média {_fmt(info['mean_feature_fidelity'], 100, 3, '%')}"
            f"  tempo total {_fmt(info['total_time_s'], 1, 1, ' s')}"
            f"  {'QUALIFICA' if info['qualifies'] else 'não qualifica'}"
        )
    if summary["chosen"]:
        lines.append(f"Motor escolhido pelas métricas: {summary['chosen']} (falta a sua nota visual dos wireframes)")
    else:
        lines.append("Nenhum motor passou no portão: pelo plano, o projeto para e é reavaliado.")
    return "\n".join(lines)
