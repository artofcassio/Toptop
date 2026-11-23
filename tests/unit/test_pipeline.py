"""Testes do pipeline completo (toptop.core.pipeline) com um motor falso.

O motor falso devolve, para cada casca, uma grade de quads do tamanho da caixa
envolvente dela (um pouco inflada e com ruído). A reprojeção tem de levar isso
de volta à superfície, às arestas e aos cantos da malha original.
"""

import os
import sys
import types
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pytest

import shapes
from toptop.core import creases as creases_module
from toptop.core import pipeline as pl
from toptop.core.creases import FeatureSet
from toptop.core.mesh import Mesh, shells, triangle_mesh
from toptop.core.pipeline import (
    DETAIL_DEFAULT,
    DETAIL_MAX,
    DETAIL_MIN,
    MIN_SHELL_QUADS,
    MODES,
    PipelineError,
    PipelineOptions,
    PipelineResult,
    estimate_quads,
    maya_estimate,
    maya_retopologize,
    resolve_engine,
    run_pipeline,
    shell_targets,
    surface_area,
    transfer_features,
)
from toptop.core.quality import QualityReport
from toptop.engine import quadwild as qw
from toptop.engine.base import (
    Density,
    Engine,
    EngineCancelled,
    EngineError,
    EngineTimeout,
    RunInfo,
    as_feature_pairs,
)


# --------------------------------------------------------------------------
# Motor falso e auxiliares
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fewer_samples(monkeypatch):
    """Menos amostras no Hausdorff do relatório: o teste fica rápido e a medida continua valendo."""
    monkeypatch.setattr(pl, "REPORT_SAMPLES", 5_000)
    monkeypatch.setattr(pl, "MIN_SHELL_SAMPLES", 1_000)


def _box_grid(lo: np.ndarray, hi: np.ndarray, target: int) -> Mesh:
    """Grade de quads que cobre a caixa (lo, hi); caixa achatada vira plano."""
    extent = hi - lo
    diag = float(np.linalg.norm(extent))
    flat = extent <= 1e-9 * max(diag, 1e-300)
    if flat.any():
        w = int(np.flatnonzero(flat)[0])
        u, v = [a for a in range(3) if a != w]
        n = max(1, int(round(np.sqrt(target))))
        plane = shapes.grid_plane(n, n)
        pts = np.zeros((plane.num_points, 3))
        pts[:, u] = lo[u] + plane.points[:, 0] * extent[u]
        pts[:, v] = lo[v] + plane.points[:, 1] * extent[v]
        pts[:, w] = lo[w]
        return plane.with_points(pts)
    n = max(1, int(round(np.sqrt(target / 6.0))))
    box = shapes.cube(n)
    return box.with_points(lo + box.points * extent)


class FakeEngine(Engine):
    """Motor de mentira: grade de quads na caixa de cada casca, inflada e com ruído."""

    name = "falso"
    label = "Motor falso"

    def __init__(
        self, inflate: float = 1.01, jitter: float = 0.001, fail: Exception | None = None, seed: int = 3
    ) -> None:
        super().__init__(timeout_s=60.0)
        self.inflate = inflate
        self.jitter = jitter
        self.fail = fail
        self.rng = np.random.default_rng(seed)
        self.calls: list[dict] = []

    def availability(self) -> tuple[bool, str]:
        """Sempre disponível."""
        return True, "motor falso"

    def remesh(self, mesh, feature_edges, density, *, cancel=None, progress=None) -> Mesh:
        """Registra a chamada e devolve a grade da caixa envolvente."""
        density = Density.coerce(density)
        lo, hi = mesh.bbox()
        self.calls.append(
            {
                "faces": mesh.num_faces,
                "target": density.target_quads,
                "features": len(as_feature_pairs(feature_edges)),
                "size": float(np.linalg.norm(hi - lo)),
                "mesh": mesh,
            }
        )
        if progress is not None:
            progress(0.5, "calculando")
        if cancel is not None and cancel():
            raise EngineCancelled("execução cancelada pelo usuário")
        if self.fail is not None:
            raise self.fail
        grid = _box_grid(lo, hi, density.target_quads)
        center = 0.5 * (lo + hi)
        diag = float(np.linalg.norm(hi - lo))
        pts = center + (grid.points - center) * self.inflate
        pts = pts + self.rng.uniform(-self.jitter, self.jitter, pts.shape) * diag
        self.last_run = RunInfo(seconds=0.0)
        return Mesh(pts, grid.counts, grid.connects, mesh.name)


def _cube_input(size: float = 1.0, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[Mesh, FeatureSet]:
    """Cubo triangulado (12 triângulos) e as 12 arestas dele como crease forte."""
    box = shapes.cube(1, size)
    feats = FeatureSet(shapes.cube_feature_edges(box, size))
    return shapes.translated(triangle_mesh(box), offset), feats


def _snapshot(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cópia dos arrays para conferir que nada foi alterado no lugar."""
    return mesh.points.copy(), mesh.counts.copy(), mesh.connects.copy()


def _unchanged(mesh: Mesh, snap: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
    """Os arrays continuam iguais à cópia."""
    return (
        np.array_equal(mesh.points, snap[0])
        and np.array_equal(mesh.counts, snap[1])
        and np.array_equal(mesh.connects, snap[2])
    )


def _length(mesh: Mesh, pairs: np.ndarray) -> float:
    """Comprimento total das arestas dadas."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return float(np.linalg.norm(mesh.points[pairs[:, 1]] - mesh.points[pairs[:, 0]], axis=1).sum())


def _on_box_edges(points: np.ndarray, lo: np.ndarray, hi: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Pontos sobre as 12 arestas da caixa (duas coordenadas na face da caixa)."""
    side = (np.abs(points - lo) <= tol) | (np.abs(points - hi) <= tol)
    return side.sum(axis=1) >= 2


# --------------------------------------------------------------------------
# Densidade e opções
# --------------------------------------------------------------------------


def test_constants():
    assert MODES == ("game", "film")
    assert (DETAIL_MIN, DETAIL_MAX, DETAIL_DEFAULT) == (1, 10, 5)
    opts = PipelineOptions()
    assert opts.mode == "game" and opts.detail == 5 and opts.target_quads is None
    assert opts.strong_threshold == 2.0 and opts.hard_edges and opts.suggest_angle == 30.0
    assert opts.reduce_limit == 100_000 and opts.weld_tol_rel == 1e-6 and opts.reproject
    assert opts.engine == "quadwild" and opts.engine_options == {} and opts.timeout_s == 600.0
    assert opts.symmetry is None
    assert PipelineOptions().engine_options is not opts.engine_options
    opts.validate()


def test_estimate_quads_formula_monotonic_film_and_minimum():
    box = shapes.cube(3)
    diag = np.sqrt(3.0)
    for detail in (1, 5, 10):
        edge = diag * 0.1 * 0.75 ** (detail - 1)
        assert estimate_quads(box, detail) == round(6.0 / edge**2)
    game = [estimate_quads(box, d, "game") for d in range(DETAIL_MIN, DETAIL_MAX + 1)]
    film = [estimate_quads(box, d, "film") for d in range(DETAIL_MIN, DETAIL_MAX + 1)]
    assert all(a < b for a, b in zip(game, game[1:]))
    assert all(f > g for f, g in zip(film, game))
    assert film[4] == pytest.approx(2.5 * game[4], rel=1e-3)
    # tira longa e fina: área minúscula para a diagonal -> mínimo de 24
    strip = shapes.grid_plane(10, 1, sx=10.0, sy=0.01)
    assert estimate_quads(strip, 1) == MIN_SHELL_QUADS
    assert estimate_quads(Mesh(np.zeros((0, 3)), [], []), 5) == MIN_SHELL_QUADS
    assert surface_area(box) == pytest.approx(6.0)
    assert surface_area(triangle_mesh(box)) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        estimate_quads(box, 0)
    with pytest.raises(ValueError):
        estimate_quads(box, 5, "cinema")


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "cinema"},
        {"detail": 0},
        {"detail": 11},
        {"detail": 2.5},
        {"detail": True},
        {"target_quads": 0},
        {"target_quads": 10.5},
        {"strong_threshold": 0.0},
        {"strong_threshold": float("nan")},
        {"suggest_angle": 0.0},
        {"suggest_angle": 200.0},
        {"reduce_limit": 0},
        {"weld_tol_rel": -1e-6},
        {"weld_tol_rel": 0.5},
        {"timeout_s": 0},
        {"symmetry": "w"},
        {"engine": "motor_que_nao_existe"},
        {"engine": 42},
        {"engine_options": None},
        {"hard_edges": "sim"},
        {"reproject": 1},
    ],
)
def test_options_validate_rejects_bad_values(changes):
    with pytest.raises(ValueError):
        PipelineOptions(**changes).validate()


def test_options_validate_accepts_good_values():
    PipelineOptions(mode="film", detail=10, target_quads=24, symmetry="z", engine=FakeEngine()).validate()
    PipelineOptions(detail=np.int64(1), suggest_angle=180, weld_tol_rel=0, engine="maya_retopo").validate()


def test_shell_targets_split_by_area_with_minimum():
    assert shell_targets(1500, np.array([6.0, 24.0])) == [300, 1200]
    assert shell_targets(100, np.array([1.0, 1000.0])) == [MIN_SHELL_QUADS, 100]
    assert shell_targets(100, np.array([0.0, 0.0])) == [50, 50]
    assert shell_targets(100, np.zeros(0)) == []


def test_resolve_engine_by_name_or_instance():
    engine = resolve_engine(PipelineOptions(engine="quadwild", timeout_s=12.0, engine_options={"keep_workdir": True}))
    assert isinstance(engine, qw.QuadWildEngine)
    assert engine.timeout_s == 12.0 and engine.keep_workdir
    engine = resolve_engine(PipelineOptions(timeout_s=12.0, engine_options={"timeout_s": 7.0}))
    assert engine.timeout_s == 7.0
    fake = FakeEngine()
    assert resolve_engine(PipelineOptions(engine=fake)) is fake


# --------------------------------------------------------------------------
# Pipeline com o motor falso
# --------------------------------------------------------------------------


def test_cube_end_to_end():
    mesh, feats = _cube_input()
    snap, edges_before = _snapshot(mesh), feats.edges.copy()
    engine = FakeEngine()
    steps: list[tuple[float, str]] = []
    result = run_pipeline(
        mesh, feats, PipelineOptions(engine=engine, target_quads=600), progress=lambda f, m: steps.append((f, m))
    )
    assert isinstance(result, PipelineResult)
    assert _unchanged(mesh, snap) and np.array_equal(feats.edges, edges_before)

    # o motor viu a malha limpa com as 12 creases e o alvo pedido
    assert len(engine.calls) == 1
    assert engine.calls[0]["target"] == 600 and engine.calls[0]["features"] == 12
    assert result.target_quads == 600 and result.shell_targets == [600]

    out = result.mesh
    assert out.num_faces == 600 and bool((out.counts == 4).all())
    assert out.name == mesh.name

    report = result.report
    assert isinstance(report, QualityReport)
    assert report.quad_ratio == 1.0 and report.shells == 1
    assert report.hausdorff < 1e-9
    assert report.feature_fidelity < 1e-9
    assert report.crease_coverage > 0.999 and report.corners_matched == 8
    assert report.time_s == pytest.approx(result.engine_seconds)
    assert report.passed, report.checks
    assert result.shell_reports == [report]
    assert result.seconds >= result.engine_seconds >= 0.0

    # hard edges: exatamente as arestas novas sobre as 12 arestas do cubo
    hard = result.hard_edges
    assert hard.shape == (120, 2)
    lo, hi = np.zeros(3), np.ones(3)
    assert _on_box_edges(out.points[hard.reshape(-1)], lo, hi).all()
    assert _length(out, hard) == pytest.approx(12.0, abs=1e-9)
    assert (out.topology.edge_ids(hard) >= 0).all()

    # creases para o Film: todos os loops de feature, 3.0 nas fortes
    assert len(result.crease_edges) == 120
    assert np.all(result.crease_edges.crease == 3.0)

    # progresso: crescente, entre 0 e 1, termina em 1
    fractions = [f for f, _ in steps]
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))
    assert any("Casca 1 de 1" in m and "calculando" in m for _, m in steps)


def test_without_reprojection_the_shape_is_off():
    mesh, feats = _cube_input()
    result = run_pipeline(mesh, feats, PipelineOptions(engine=FakeEngine(), target_quads=600, reproject=False))
    assert result.report.hausdorff > 1e-3
    assert not any(n.startswith("Reprojeção") for n in result.notes)


def test_multi_shell_runs_engine_per_shell_split_by_area_and_merges():
    small, small_feats = _cube_input()
    big, big_feats = _cube_input(2.0, (3.0, 0.0, 0.0))
    mesh = shapes.combine(small, big)
    feats = small_feats.merged(FeatureSet(big_feats.edges + small.num_points))
    engine = FakeEngine()
    result = run_pipeline(mesh, feats, PipelineOptions(engine=engine, target_quads=1500))

    assert len(engine.calls) == 2
    by_size = sorted((c["size"], c["target"]) for c in engine.calls)
    assert [t for _, t in by_size] == [300, 1200]
    assert sorted(result.shell_targets) == [300, 1200]
    assert all(c["features"] == 12 for c in engine.calls)

    out = result.mesh
    assert shells(out)[1] == 2 and bool((out.counts == 4).all())
    assert out.num_faces == 6 * 7**2 + 6 * 14**2  # cubos 7x7 e 14x14 do motor falso
    assert any("2 cascas" in n for n in result.notes)

    report = result.report
    assert report.shells == 2 and report.hausdorff < 1e-9
    assert report.corners_matched == 16 and report.passed, report.checks
    assert _length(out, result.hard_edges) == pytest.approx(12.0 + 24.0, abs=1e-9)

    assert len(result.shell_reports) == 2
    for shell_report in result.shell_reports:
        assert shell_report.ref_triangles == 12 and shell_report.shells == 1
        assert shell_report.passed, shell_report.checks
        assert shell_report.corners_matched == 8
    diagonals = sorted(r.diagonal for r in result.shell_reports)
    assert diagonals == pytest.approx([np.sqrt(3.0), 2.0 * np.sqrt(3.0)])
    assert sum(r.time_s for r in result.shell_reports) == pytest.approx(result.engine_seconds)


def test_detail_and_film_mode_set_the_target():
    mesh, feats = _cube_input()
    game = FakeEngine()
    run_pipeline(mesh, feats, PipelineOptions(engine=game, detail=3))
    film = FakeEngine()
    result = run_pipeline(mesh, feats, PipelineOptions(engine=film, detail=3, mode="film", symmetry="x"))
    assert game.calls[0]["target"] == estimate_quads(mesh, 3, "game")
    assert film.calls[0]["target"] == estimate_quads(mesh, 3, "film") > game.calls[0]["target"]
    assert any("Simetria fica para a Fase 2" in n for n in result.notes)
    # hard edges são calculadas também no Film (a camada do Maya decide se aplica)
    assert len(result.hard_edges) > 0


def test_features_none_suggests_creases_by_angle():
    mesh, _ = _cube_input()
    engine = FakeEngine()
    result = run_pipeline(mesh, None, PipelineOptions(engine=engine, target_quads=600))
    assert engine.calls[0]["features"] == 12
    assert any("sugestão automática" in n for n in result.notes)
    assert _length(result.mesh, result.hard_edges) == pytest.approx(12.0, abs=1e-9)

    # FeatureSet vazio = sem creases de propósito (com aviso das quinas); 180° desliga a sugestão
    for feats, angle, warned in ((FeatureSet.empty(), 30.0, True), (None, 180.0, False)):
        engine = FakeEngine()
        result = run_pipeline(mesh, feats, PipelineOptions(engine=engine, target_quads=600, suggest_angle=angle))
        assert engine.calls[0]["features"] == 0
        assert len(result.hard_edges) == 0 and len(result.crease_edges) == 0
        assert any("estão sem crease" in n for n in result.notes) == warned
        assert any("nenhuma quina acima de 180°" in n for n in result.notes) == (angle == 180.0)


def test_open_borders_are_guide_features_without_hard_edges():
    plane = shapes.grid_plane(4, 4)
    engine = FakeEngine()
    result = run_pipeline(plane, FeatureSet.empty(), PipelineOptions(engine=engine, target_quads=100))
    assert engine.calls[0]["features"] == 16  # as 16 arestas de borda, como guia
    assert len(result.hard_edges) == 0
    feats = result.crease_edges
    assert len(feats) == 40 and np.all(feats.crease == 1.0)
    assert _length(result.mesh, feats.edges) == pytest.approx(4.0, abs=1e-9)
    assert result.report.hausdorff < 1e-9 and result.report.crease_coverage > 0.999


def test_weld_before_the_engine():
    mesh, feats = _cube_input()
    # cada canto com o seu próprio vértice: sem solda, tudo seria borda
    raw = Mesh(mesh.points[mesh.connects], mesh.counts, np.arange(mesh.connects.size), mesh.name)
    engine = FakeEngine()
    result = run_pipeline(raw, None, PipelineOptions(engine=engine, target_quads=600))
    assert engine.calls[0]["mesh"].num_points == 8
    assert engine.calls[0]["features"] == 12
    assert any("soldado" in n for n in result.notes)
    assert result.report.boundary_edges == 0 and result.report.passed


def test_nonmanifold_edge_raises_with_input_edges():
    pts = [(9, 9, 9), (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
    mesh = Mesh.from_faces(pts, [(1, 2, 3), (2, 1, 4), (1, 2, 5)])
    engine = FakeEngine()
    with pytest.raises(PipelineError) as info:
        run_pipeline(mesh, None, PipelineOptions(engine=engine))
    error = info.value
    assert isinstance(error, RuntimeError)
    assert error.stage == "validar" and "non-manifold" in str(error)
    assert error.edges.tolist() == [[1, 2]]
    assert not engine.calls


def test_bowtie_vertex_raises_with_vertices():
    pts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
    mesh = Mesh.from_faces(pts, [(0, 1, 2), (0, 3, 4)])
    with pytest.raises(PipelineError) as info:
        run_pipeline(mesh, None, PipelineOptions(engine=FakeEngine()))
    assert info.value.vertices.tolist() == [0] and info.value.edges.shape == (0, 2)


def test_empty_mesh_raises():
    with pytest.raises(PipelineError) as info:
        run_pipeline(Mesh(np.zeros((0, 3)), [], []), None, PipelineOptions(engine=FakeEngine()))
    assert info.value.stage == "validar"
    assert PipelineError("x").edges.shape == (0, 2) and PipelineError("x").stage == ""


def test_bad_options_raise_before_anything():
    mesh, feats = _cube_input()
    engine = FakeEngine()
    with pytest.raises(ValueError):
        run_pipeline(mesh, feats, PipelineOptions(engine=engine, detail=0))
    assert not engine.calls


def test_cancel_before_and_inside_the_engine():
    mesh, feats = _cube_input()
    engine = FakeEngine()
    with pytest.raises(EngineCancelled):
        run_pipeline(mesh, feats, PipelineOptions(engine=engine, target_quads=600), cancel=lambda: True)
    assert not engine.calls

    asked = []

    def cancel_later() -> bool:
        asked.append(1)
        return len(asked) > 3  # deixa passar as checagens antes do motor

    engine = FakeEngine()
    with pytest.raises(EngineCancelled):
        run_pipeline(mesh, feats, PipelineOptions(engine=engine, target_quads=600), cancel=cancel_later)
    assert len(engine.calls) == 1


def test_engine_errors_name_the_shell():
    small, small_feats = _cube_input()
    big, big_feats = _cube_input(2.0, (3.0, 0.0, 0.0))
    mesh = shapes.combine(small, big)
    engine = FakeEngine(fail=EngineTimeout("tempo limite de 60 s excedido", "/tmp/pasta", "log"))
    with pytest.raises(EngineTimeout) as info:
        run_pipeline(mesh, None, PipelineOptions(engine=engine, target_quads=600))
    message = str(info.value)
    assert "Casca 1 de 2" in message and "tempo limite" in message
    assert message.count("Pasta de diagnóstico") == 1 and info.value.workdir == "/tmp/pasta"

    engine = FakeEngine(fail=ZeroDivisionError("divisão por zero"))
    with pytest.raises(EngineError) as info:
        run_pipeline(small, small_feats, PipelineOptions(engine=engine, target_quads=600))
    assert "Casca 1 de 1" in str(info.value) and "divisão por zero" in str(info.value)

    class EmptyEngine(FakeEngine):
        def remesh(self, mesh, feature_edges, density, *, cancel=None, progress=None):
            return Mesh(np.zeros((0, 3)), [], [])

    with pytest.raises(EngineError, match="vazia"):
        run_pipeline(small, small_feats, PipelineOptions(engine=EmptyEngine(), target_quads=600))


def test_separate_reference_transfers_creases_both_ways():
    reference = triangle_mesh(shapes.cube(4))
    ref_feats = FeatureSet(shapes.cube_feature_edges(reference))
    reduced = triangle_mesh(shapes.cube(2))
    reduced_feats = FeatureSet(shapes.cube_feature_edges(reduced))
    assert (len(ref_feats), len(reduced_feats)) == (48, 24)
    snaps = _snapshot(reference), _snapshot(reduced)

    # creases só no original: vão para a versão reduzida
    engine = FakeEngine()
    options = PipelineOptions(engine=engine, target_quads=600)
    result = run_pipeline(reduced, None, options, reference=reference, reference_features=ref_feats)
    assert engine.calls[0]["faces"] == reduced.num_faces and engine.calls[0]["features"] == 24
    assert result.report.ref_triangles == reference.num_faces
    assert any("transferidas do original" in n for n in result.notes)
    assert any("versão reduzida" in n for n in result.notes)
    assert result.report.passed and result.report.feature_length == pytest.approx(12.0)

    # creases só na versão reduzida: vão para o original
    result = run_pipeline(
        reduced, reduced_feats, PipelineOptions(engine=FakeEngine(), target_quads=600), reference=reference
    )
    assert result.report.feature_length == pytest.approx(12.0)
    assert result.report.crease_coverage > 0.999
    assert _length(result.mesh, result.hard_edges) == pytest.approx(12.0, abs=1e-9)

    # nenhuma crease: sugestão no original, transferida para a reduzida
    engine = FakeEngine()
    result = run_pipeline(reduced, None, PipelineOptions(engine=engine, target_quads=600), reference=reference)
    assert engine.calls[0]["features"] == 24 and result.report.feature_length == pytest.approx(12.0)
    assert _unchanged(reference, snaps[0]) and _unchanged(reduced, snaps[1])


def test_nonmanifold_reduced_version_is_reported_as_such():
    reference, ref_feats = _cube_input()
    pts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
    bad = Mesh.from_faces(pts, [(0, 1, 2), (1, 0, 3), (0, 1, 4)])
    with pytest.raises(PipelineError) as info:
        run_pipeline(bad, None, PipelineOptions(engine=FakeEngine()), reference=reference, reference_features=ref_feats)
    assert info.value.stage == "reduzir" and info.value.edges.tolist() == [[0, 1]]


def test_transfer_features_between_densities():
    dense = triangle_mesh(shapes.cube(4))
    coarse = triangle_mesh(shapes.cube(2))
    feats = FeatureSet(shapes.cube_feature_edges(dense), np.full(48, 1.0))
    moved = transfer_features(coarse, dense, feats)
    assert len(moved) == 24 and np.all(moved.crease == 1.0)
    assert _length(coarse, moved.edges) == pytest.approx(12.0)
    strong = transfer_features(coarse, dense, FeatureSet(feats.edges, np.full(48, 6.0)), strong_threshold=5.0)
    assert len(strong) == 24 and np.all(strong.crease >= 5.0)
    # crease 3.0 fica abaixo do limiar 5.0: sai como guia (abaixo de 2.5)
    guide = transfer_features(coarse, dense, FeatureSet(feats.edges), strong_threshold=5.0)
    assert len(guide) == 24 and np.all(guide.crease < 5.0)
    assert len(transfer_features(coarse, dense, FeatureSet.empty())) == 0
    assert len(transfer_features(coarse, dense, FeatureSet(np.array([[0, 999]])))) == 0


# --------------------------------------------------------------------------
# Camada Maya com um Maya de mentira
# --------------------------------------------------------------------------


class FakeMayaScene:
    """Registra as chamadas ao ``maya.cmds`` e ao ``maya_bridge`` de mentira."""

    def __init__(
        self,
        monkeypatch,
        original: Mesh,
        feats: FeatureSet,
        reduced: Mesh | None = None,
        reduced_feats: FeatureSet | None = None,
    ) -> None:
        self.calls: list[tuple] = []
        cmds = types.ModuleType("maya.cmds")

        def undo_info(**flags):
            if flags.get("query"):
                return True
            self.calls.append(("undoInfo", flags))
            return None

        cmds.undoInfo = undo_info
        cmds.duplicate = lambda node, **kw: self.calls.append(("duplicate", node, kw)) or ["|copia"]
        cmds.objExists = lambda name: True
        cmds.delete = lambda *a, **kw: self.calls.append(("delete", a, kw))
        cmds.select = lambda *a, **kw: self.calls.append(("select", a, kw))
        cmds.polyReduce = lambda name, **kw: self.calls.append(("polyReduce", name, kw))
        maya = types.ModuleType("maya")
        maya.cmds = cmds

        bridge = types.ModuleType("toptop.maya_bridge")
        bridge.transform_of = lambda name: "|" + name.split("|")[-1]
        bridge.shape_of = lambda name: name + "|" + name.split("|")[-1] + "Shape"

        def read(name, world=True):
            self.calls.append(("read", name, world))
            return reduced if "copia" in name else original

        @contextmanager
        def undo_chunk(name: str = "TopTop") -> Iterator[None]:
            self.calls.append(("chunk", name))
            try:
                yield
            finally:
                self.calls.append(("chunk_end",))

        bridge.mesh_from_maya = read
        bridge.undo_chunk = undo_chunk
        bridge.create_mesh = lambda mesh, name: self.calls.append(("create", mesh.num_faces, name)) or "|" + name
        bridge.set_hard_edges = lambda name, pairs: self.calls.append(("hard", name, len(pairs)))
        bridge.copy_material = lambda source, target: self.calls.append(("material", source, target))
        monkeypatch.setitem(sys.modules, "maya", maya)
        monkeypatch.setitem(sys.modules, "maya.cmds", cmds)
        monkeypatch.setitem(sys.modules, "toptop.maya_bridge", bridge)
        import toptop

        monkeypatch.setattr(toptop, "maya_bridge", bridge, raising=False)
        monkeypatch.setattr(
            creases_module,
            "maya_read_creases",
            lambda name: (reduced_feats if "copia" in name else feats),
        )

    def names(self) -> list[str]:
        """Só o nome de cada chamada, na ordem."""
        return [c[0] for c in self.calls]


def test_maya_retopologize_without_reduction(monkeypatch):
    mesh, feats = _cube_input()
    scene = FakeMayaScene(monkeypatch, mesh, feats)
    new, result = maya_retopologize("|grupo|cubo", PipelineOptions(engine=FakeEngine(), target_quads=600))
    assert new == "|cubo_toptop" and isinstance(result, PipelineResult)
    assert scene.names() == [
        "read", "undoInfo", "undoInfo", "chunk", "create", "hard", "material", "select", "chunk_end",
    ]  # fmt: skip
    assert scene.calls[0] == ("read", "|cubo", True)
    assert scene.calls[1][1] == {"stateWithoutFlush": False} and scene.calls[2][1] == {"stateWithoutFlush": True}
    assert scene.calls[4] == ("create", 600, "cubo_toptop")
    assert scene.calls[5] == ("hard", "|cubo_toptop", 120)
    assert scene.calls[6] == ("material", "|cubo", "|cubo_toptop")
    assert scene.calls[7][1] == ("|cubo_toptop",) and scene.calls[7][2] == {"replace": True}


def test_maya_retopologize_reduces_a_copy_and_film_is_all_soft(monkeypatch):
    original = triangle_mesh(shapes.cube(4))
    feats = FeatureSet(shapes.cube_feature_edges(original))
    reduced = triangle_mesh(shapes.cube(2))
    scene = FakeMayaScene(monkeypatch, original, feats, reduced, FeatureSet.empty())
    engine = FakeEngine()
    options = PipelineOptions(engine=engine, target_quads=600, reduce_limit=50, mode="film")
    steps: list[float] = []
    new, result = maya_retopologize("cubo", options, progress=lambda f, m: steps.append(f))
    names = scene.names()
    assert names[:2] == ["read", "undoInfo"]
    assert names[2:7] == ["duplicate", "polyReduce", "delete", "read", "delete"]
    assert scene.calls[2][2] == {"name": "cubo_toptopReduzida", "returnRootsOnly": True}
    reduce_name, flags = scene.calls[3][1], scene.calls[3][2]
    assert reduce_name == "|copia|copiaShape"
    assert flags["triangleCount"] == 50 and flags["keepCreaseEdgeWeight"] == 1.0
    assert flags["keepBorderWeight"] == 1.0 and flags["constructionHistory"] is False
    assert scene.calls[5] == ("read", "|copia", True)  # leitura da cópia reduzida
    assert scene.calls[6][1] == ("|copia",) and scene.calls[6][2] == {}  # cópia apagada
    assert names[7:] == ["undoInfo", "chunk", "create", "hard", "material", "select", "chunk_end"]
    assert scene.calls[10] == ("hard", "|cubo_toptop", 0)  # Film: tudo suave
    # o motor viu a cópia reduzida, com as creases transferidas do original
    assert engine.calls[0]["faces"] == reduced.num_faces and engine.calls[0]["features"] == 24
    assert result.report.ref_triangles == original.num_faces
    assert steps[-1] == 1.0 and all(b >= a for a, b in zip(steps, steps[1:]))


def test_maya_retopologize_cancel_creates_nothing(monkeypatch):
    mesh, feats = _cube_input()
    scene = FakeMayaScene(monkeypatch, mesh, feats)
    with pytest.raises(EngineCancelled):
        maya_retopologize("cubo", PipelineOptions(engine=FakeEngine()), cancel=lambda: True)
    assert scene.names() == ["read"]

    # cancelado já dentro do pipeline: a gravação de desfazer volta a ligar
    scene = FakeMayaScene(monkeypatch, mesh, feats)
    asked: list[int] = []

    def cancel_second_time() -> bool:
        asked.append(1)
        return len(asked) > 1

    with pytest.raises(EngineCancelled):
        maya_retopologize("cubo", PipelineOptions(engine=FakeEngine()), cancel=cancel_second_time)
    assert scene.names() == ["read", "undoInfo", "undoInfo"]
    assert scene.calls[-1][1] == {"stateWithoutFlush": True}


def test_maya_estimate(monkeypatch):
    mesh, feats = _cube_input()
    FakeMayaScene(monkeypatch, mesh, feats)
    assert maya_estimate("cubo", 4, "film") == estimate_quads(mesh, 4, "film")


# --------------------------------------------------------------------------
# Integração com o QuadWild de verdade
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.environ.get(qw.INSTALL_ENV) and qw.find_install()), reason="binários reais do QuadWild não instalados"
)
def test_real_quadwild_pipeline_on_a_cube(tmp_path):
    mesh = triangle_mesh(shapes.cube(6))
    feats = FeatureSet(shapes.cube_feature_edges(mesh))
    options = PipelineOptions(target_quads=600, timeout_s=300.0, engine_options={"workdir_root": str(tmp_path)})
    result = run_pipeline(mesh, feats, options)
    report = result.report
    assert report.quad_ratio == 1.0 and Density(600).accepts(report.quads)
    assert report.crease_coverage > 0.99 and report.corners_matched == 8
    assert report.hausdorff < 1e-3 and report.feature_fidelity < 1e-6
    assert result.engine_seconds > 0 and report.time_s == pytest.approx(result.engine_seconds)
    lo, hi = np.zeros(3), np.ones(3)
    assert _on_box_edges(result.mesh.points[result.hard_edges.reshape(-1)], lo, hi).all()
    assert _length(result.mesh, result.hard_edges) == pytest.approx(12.0, rel=1e-6)
