"""Testes da janela (toptop.ui.window) e da prateleira (toptop.shelf) sem Maya.

A janela é montada de verdade com PySide6 em modo ``offscreen``; os módulos do
Maya e o ``toptop.maya_bridge`` são de mentira (``sys.modules``) e as funções
do TopTop chamadas pelos botões são trocadas por versões que registram as
chamadas. Sem PySide6 (ou sem as bibliotecas do Qt), os testes da janela são
pulados; os da prateleira e das funções puras rodam sempre.
"""

import importlib
import os
import subprocess
import sys
import types
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pytest

import shapes
from toptop.core import creases as creases_module
from toptop.core import pipeline as pl
from toptop.core.creases import FeatureSet, check_input
from toptop.core.pipeline import DETAIL_DEFAULT, DETAIL_MAX, DETAIL_MIN, PipelineError, PipelineResult
from toptop.core.quality import QualityReport, format_report
from toptop.engine.base import EngineCancelled, EngineError, EngineTimeout

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = "|pCube1"
OTHER = "|pSphere1"


# --------------------------------------------------------------------------
# Qt e Maya de mentira
# --------------------------------------------------------------------------


def _qt():
    """Importa o PySide6 em modo offscreen ou pula o teste."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets


@pytest.fixture(scope="module")
def qapp():
    """Uma ``QApplication`` para o módulo inteiro."""
    QtWidgets = _qt()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


class FakeMaya:
    """Maya de mentira: seleção, callbacks, ``cmds`` e ``maya_bridge`` que registram chamadas."""

    def __init__(self, monkeypatch, main_window=None) -> None:
        self.calls: list[tuple] = []
        self.selected: str | None = TARGET
        self.callbacks: dict[int, tuple[str, object]] = {}
        self.removed: list[int] = []
        self._next_id = 100

        cmds = types.ModuleType("maya.cmds")
        cmds.warning = lambda message: self.calls.append(("warning", message))
        cmds.select = lambda *a, **kw: self.calls.append(("select", a, kw))
        maya = types.ModuleType("maya")
        maya.cmds = cmds

        om = types.ModuleType("maya.api.OpenMaya")

        def add_event_callback(event, function, client_data=None):
            self._next_id += 1
            self.callbacks[self._next_id] = (event, function)
            return self._next_id

        def remove_callback(cid):
            if cid not in self.callbacks:
                raise RuntimeError("callback já removido")
            del self.callbacks[cid]
            self.removed.append(cid)

        om.MEventMessage = types.SimpleNamespace(addEventCallback=add_event_callback)
        om.MMessage = types.SimpleNamespace(removeCallback=remove_callback)
        api = types.ModuleType("maya.api")
        api.OpenMaya = om
        maya.api = api

        omui = types.ModuleType("maya.OpenMayaUI")
        if main_window is not None:
            from shiboken6 import getCppPointer

            pointer = int(getCppPointer(main_window)[0])
        else:
            pointer = None
        omui.MQtUtil = types.SimpleNamespace(mainWindow=lambda: pointer)
        maya.OpenMayaUI = omui

        bridge = types.ModuleType("toptop.maya_bridge")
        bridge.selected_mesh = lambda: self.selected
        bridge.shape_of = lambda name: name + "|" + name.split("|")[-1] + "Shape"

        def select_edges(name, pairs):
            pairs = np.asarray(pairs).reshape(-1, 2)
            self.calls.append(("select_edges", name, pairs.copy()))
            return len(pairs)

        bridge.select_edges = select_edges

        @contextmanager
        def undo_chunk(name: str = "TopTop") -> Iterator[None]:
            yield

        bridge.undo_chunk = undo_chunk

        for name, module in (
            ("maya", maya),
            ("maya.cmds", cmds),
            ("maya.api", api),
            ("maya.api.OpenMaya", om),
            ("maya.OpenMayaUI", omui),
            ("toptop.maya_bridge", bridge),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        import toptop

        monkeypatch.setattr(toptop, "maya_bridge", bridge, raising=False)

    def fire_selection(self, selected: str | None) -> None:
        """Muda a seleção e chama os callbacks ``SelectionChanged`` como o Maya faria."""
        self.selected = selected
        for event, function in list(self.callbacks.values()):
            if event == "SelectionChanged":
                function(None)

    def named(self, name: str) -> list[tuple]:
        """Só as chamadas com este nome."""
        return [c for c in self.calls if c[0] == name]


ENGINES = [
    ("quadwild", "QuadWild Bi-MDF", True, "/opt/quadwild"),
    ("maya_retopo", "Maya Retopologize", True, "comando polyRetopo"),
    ("quadremesher", "Quad Remesher", False, "módulo do Quad Remesher não encontrado"),
]


@pytest.fixture
def ui(qapp, monkeypatch):
    """(módulo da janela, Maya de mentira, estimativas pedidas) com motores e estimativa falsos."""
    window_module = importlib.import_module("toptop.ui.window")
    scene = FakeMaya(monkeypatch)
    estimates: list[tuple] = []

    def fake_estimate(name, detail, mode="game"):
        estimates.append((name, detail, mode))
        return 1000 * detail * (2 if mode == "film" else 1) + (7 if name == OTHER else 0)

    monkeypatch.setattr(window_module, "engine_choices", lambda: list(ENGINES))
    monkeypatch.setattr(pl, "maya_estimate", fake_estimate)
    monkeypatch.setattr(window_module, "_window", None)
    yield window_module, scene, estimates
    window_module.close()
    for widget in qapp.topLevelWidgets():
        if widget.objectName() == window_module.OBJECT_NAME:
            widget.close()
            widget.deleteLater()
    _flush_deletes(qapp)


def _flush_deletes(qapp) -> None:
    """Processa os ``deleteLater`` pendentes (fora de um laço de eventos eles não rodam sozinhos)."""
    from PySide6 import QtCore

    qapp.processEvents()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


@pytest.fixture
def win(ui):
    """Janela montada (sem mostrar) e o Maya de mentira."""
    window_module, scene, estimates = ui
    window = window_module.TopTopWindow()
    window.update_estimate()
    yield window, scene, estimates
    from shiboken6 import isValid

    if isValid(window) and not window.running:
        window.close()


def _result(**kw) -> PipelineResult:
    """``PipelineResult`` pequeno com um relatório aprovado."""
    report = QualityReport(name="pCube1_toptop", engine="quadwild", faces=600, quads=600, quad_ratio=1.0, passed=True)
    values = dict(
        mesh=shapes.cube(10),
        hard_edges=np.array([[0, 1], [1, 2], [2, 3]]),
        crease_edges=FeatureSet.empty(),
        report=report,
        notes=["Limpeza: 2 vértice(s) soldado(s).", "Motor: aviso de teste"],
        seconds=4.25,
        engine_seconds=3.5,
        target_quads=600,
        shell_targets=[600],
    )
    values.update(kw)
    return PipelineResult(**values)


# --------------------------------------------------------------------------
# Funções puras (sem Qt)
# --------------------------------------------------------------------------


def test_modules_import_without_maya():
    """``toptop.shelf`` e ``toptop.ui.window`` não importam o Maya ao carregar."""
    code = (
        "import sys, os\n"
        "os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n"
        "import toptop.shelf\n"
        "try:\n"
        "    import toptop.ui.window\n"
        "except ImportError as exc:\n"
        "    assert 'PySide6' in str(exc) or 'lib' in str(exc), exc\n"
        "assert not any(m == 'maya' or m.startswith('maya.') for m in sys.modules), sorted(sys.modules)\n"
        "assert 'toptop.maya_bridge' not in sys.modules\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")


def test_crease_values_follow_threshold():
    """No limiar padrão grava 3.0 / 1.0; com outro limiar, o forte continua forte e o guia, guia."""
    _qt()
    from toptop.ui.window import guide_crease_value, strong_crease_value

    assert strong_crease_value(2.0) == creases_module.STRONG_VALUE
    assert guide_crease_value(2.0) == creases_module.GUIDE_VALUE
    for threshold in (0.5, 1.0, 2.0, 3.0, 4.5, 10.0):
        feats = FeatureSet(np.array([[0, 1], [1, 2]]), np.array([strong_crease_value(threshold), guide_crease_value(threshold)]))
        assert feats.strong_mask(threshold).tolist() == [True, False]
        assert feats.guide_mask(threshold).tolist() == [False, True]


def test_check_problem_edges_and_text():
    """Cadeias soltas viram pares de vértices seguidos; quinas sem crease entram também."""
    _qt()
    from toptop.ui.window import chain_pairs, check_problem_edges, format_check

    assert chain_pairs([np.array([4, 5, 6]), np.array([9])]).tolist() == [[4, 5], [5, 6]]
    assert chain_pairs([]).shape == (0, 2)

    cube = shapes.cube(2)
    sharp = creases_module.suggest_creases(cube, 30.0)
    # só um pedaço de uma quina marcado: duas pontas soltas e o resto das quinas sem crease
    check = check_input(cube, FeatureSet(sharp[:1]))
    edges = check_problem_edges(check)
    assert len(edges) == len(sharp)
    assert set(map(tuple, np.sort(edges, axis=1).tolist())) == set(map(tuple, np.sort(sharp, axis=1).tolist()))
    text = format_check(TARGET, check, 2.0, len(edges))
    assert "pCube1" in text and "ponta(s) de crease" in text and "sem crease" in text
    assert f"{len(edges)} aresta(s) com problema selecionada(s)" in text

    clean = check_input(cube, FeatureSet(sharp))
    assert len(check_problem_edges(clean)) == 0
    assert "sem problemas" in format_check(TARGET, clean, 2.0, 0)


def test_format_result_has_report_and_notes():
    """O relatório final traz ``format_report``, o objeto criado, tempos, hard edges e notas."""
    _qt()
    from toptop.ui.window import format_int, format_result

    assert format_int(1234567) == "1.234.567"
    result = _result()
    game = format_result("|pCube1_toptop", result, pl.PipelineOptions())
    assert game.startswith(format_report(result.report))
    assert "Objeto criado: pCube1_toptop" in game
    assert "Hard edges: 3 aresta(s)" in game
    assert "- Motor: aviso de teste" in game and "3.5 s no motor" in game
    film = format_result("|pCube1_toptop", result, pl.PipelineOptions(mode="film"))
    assert "Hard edges: nenhuma" in film
    shells = _result(shell_reports=[QualityReport(quads=10, passed=True), QualityReport(quads=20, passed=False)])
    text = format_result("|x", shells, pl.PipelineOptions())
    assert "1: 10 quads, APROVADA" in text and "2: 20 quads, REPROVADA" in text


def test_engine_choices_lists_all_engines(monkeypatch):
    """Todos os motores aparecem; um adaptador que quebra vira indisponível, sem exceção."""
    _qt()
    from toptop.engine import ENGINE_NAMES
    from toptop.ui import window as window_module

    choices = window_module.engine_choices()
    assert [c[0] for c in choices] == list(ENGINE_NAMES)
    assert all(isinstance(c[2], bool) and c[3] for c in choices)
    assert choices[0][1] == "QuadWild Bi-MDF"

    def broken(name, **kw):
        raise RuntimeError("quebrado")

    monkeypatch.setattr(window_module, "get_engine", broken)
    choices = window_module.engine_choices()
    assert all(not ok and "quebrado" in why for _, _, ok, why in choices)


# --------------------------------------------------------------------------
# Janela
# --------------------------------------------------------------------------


def test_window_controls_and_defaults(win):
    """Controles da seção 9, valores padrão e os das próximas fases desligados."""
    window, scene, _ = win
    from toptop.ui import window as window_module

    assert window.objectName() == "TopTopWindow"
    assert window.target == TARGET
    assert window.target_label.text() == "pCube1"
    assert window.mode_game.isChecked() and not window.mode_film.isChecked()
    assert window.mode() == "game"
    assert (window.detail_slider.minimum(), window.detail_slider.maximum()) == (DETAIL_MIN, DETAIL_MAX)
    assert window.detail_slider.value() == DETAIL_DEFAULT
    assert window.detail_value.text() == str(DETAIL_DEFAULT)
    assert window.estimate_label.text() == "≈ 5.000 quads"
    assert window.angle_spin.value() == 30.0
    assert window.threshold_spin.value() == 2.0
    assert window.hard_edges_check.isChecked() and window.hard_edges_check.isEnabled()
    for button in (window.suggest_button, window.strong_button, window.guide_button, window.check_button):
        assert button.isEnabled()
    assert window.run_button.isEnabled() and not window.cancel_button.isEnabled()
    assert window.progress_bar.value() == 0 and window.report() == ""

    phase2 = [window.symmetry_check, window.symmetry_axis, window.weighted_normals_check]
    phase3 = [window.support_slider, window.transfer_check, window.circles_check, window.uvs_check]
    for widget in phase2:
        assert not widget.isEnabled() and widget.toolTip() == "Fase 2"
    for widget in phase3:
        assert not widget.isEnabled() and widget.toolTip() == "Fase 3"
    assert [window.symmetry_axis.itemText(i) for i in range(3)] == ["X", "Y", "Z"]
    assert (window.support_slider.minimum(), window.support_slider.maximum(), window.support_slider.value()) == (0, 10, 2)
    assert window.symmetry_check.text() == "Simetria"
    assert window.weighted_normals_check.text() == "Normais ponderadas"
    assert window.transfer_check.text() == "Transferir creases"
    assert window.circles_check.text() == "Ajustar círculos para múltiplos de 4"
    assert window.uvs_check.text() == "Gerar UVs"

    combo = window.engine_combo
    assert combo.count() == 3 and combo.currentData() == "quadwild"
    model = combo.model()
    assert model.item(0).isEnabled() and model.item(1).isEnabled() and not model.item(2).isEnabled()
    assert "(indisponível)" in combo.itemText(2)
    assert combo.itemData(2, window_module.QtCore.Qt.ItemDataRole.ToolTipRole) == ENGINES[2][3]

    options = window.options()
    assert (options.mode, options.detail, options.strong_threshold, options.hard_edges) == ("game", 5, 2.0, True)
    assert (options.suggest_angle, options.engine, options.symmetry) == (30.0, "quadwild", None)
    options.validate()

    # um callback SelectionChanged ligado
    assert [event for event, _ in scene.callbacks.values()] == ["SelectionChanged"]


def test_no_available_engine_disables_run(ui, monkeypatch):
    """Sem nenhum motor disponível, Retopologizar fica desligado com o motivo na dica."""
    window_module, _, _ = ui
    monkeypatch.setattr(window_module, "engine_choices", lambda: [("quadwild", "QuadWild", False, "não instalado")])
    window = window_module.TopTopWindow()
    try:
        assert not window.run_button.isEnabled()
        assert "não instalado" in window.run_button.toolTip()
        window.retopologize()
        assert "Motor indisponível" in window.report()
    finally:
        window.close()


def test_target_follows_selection_and_estimate(win):
    """O alvo segue a seleção; a estimativa muda com alvo, nível e modo, com cache."""
    window, scene, estimates = win
    assert estimates == [(TARGET, 5, "game")]

    scene.fire_selection(OTHER)
    assert window.target == OTHER and window.target_label.text() == "pSphere1"
    window.update_estimate()
    assert estimates[-1] == (OTHER, 5, "game")
    assert window.estimate_label.text() == "≈ 5.007 quads"

    window.detail_slider.setValue(8)
    assert window.detail_value.text() == "8"
    window.update_estimate()
    assert estimates[-1] == (OTHER, 8, "game") and window.estimate_label.text() == "≈ 8.007 quads"

    window.mode_film.setChecked(True)
    assert window.mode() == "film"
    assert not window.hard_edges_check.isEnabled()
    window.update_estimate()
    assert estimates[-1] == (OTHER, 8, "film") and window.estimate_label.text() == "≈ 16.007 quads"

    count = len(estimates)
    window.mode_game.setChecked(True)
    window.update_estimate()
    assert len(estimates) == count  # (OTHER, 8, game) já estava no cache
    assert window.hard_edges_check.isEnabled()

    scene.fire_selection(None)
    assert window.target is None and window.target_label.text() == "(selecione uma malha)"
    window.update_estimate()
    assert window.estimate_label.text() == ""
    assert not window.run_button.isEnabled()
    assert not window.suggest_button.isEnabled() and not window.check_button.isEnabled()
    assert window.strong_button.isEnabled()  # grava na seleção de arestas, com ou sem alvo


def test_estimate_timer_coalesces_changes(win, qapp):
    """Mexer no slider várias vezes seguidas faz uma leitura só, depois da pausa."""
    from PySide6 import QtTest

    window, _, estimates = win
    count = len(estimates)
    for value in (6, 7, 8, 9):
        window.detail_slider.setValue(value)
    assert len(estimates) == count
    QtTest.QTest.qWait(3 * window._estimate_timer.interval())
    assert estimates[count:] == [(TARGET, 9, "game")]


def test_estimate_error_shows_question_mark(win, monkeypatch):
    """Se a estimativa falhar, o rótulo mostra '?' e a janela segue funcionando."""
    window, _, _ = win

    def boom(*a, **kw):
        raise ValueError("objeto sumiu")

    monkeypatch.setattr(pl, "maya_estimate", boom)
    window.detail_slider.setValue(2)
    window.update_estimate()
    assert window.estimate_label.text() == "≈ ? quads"
    assert "objeto sumiu" in window.estimate_label.toolTip()


def test_suggest_selects_edges_without_writing(win, monkeypatch):
    """Sugerir chama ``creases.maya_suggest`` com o ângulo e seleciona as arestas achadas."""
    window, scene, _ = win
    calls = []
    pairs = np.array([[0, 1], [1, 2], [2, 3]])

    def fake_suggest(name, angle_deg=30.0, select=True):
        calls.append((name, angle_deg, select))
        return pairs

    monkeypatch.setattr(creases_module, "maya_suggest", fake_suggest)
    monkeypatch.setattr(creases_module, "maya_apply_selection", lambda value=3.0: pytest.fail("não pode gravar"))
    window.angle_spin.setValue(42.5)
    window.suggest_button.click()
    assert calls == [(TARGET, 42.5, False)]
    selected = scene.named("select_edges")
    assert len(selected) == 1 and selected[0][1] == TARGET and selected[0][2].tolist() == pairs.tolist()
    assert "3 aresta(s) com quina acima de 42.5°" in window.report()
    assert "Nada foi gravado" in window.report()

    # nada acima do ângulo: não mexe na seleção (o alvo continua selecionado)
    pairs = np.zeros((0, 2), dtype=np.int64)
    window.suggest_button.click()
    assert len(scene.named("select_edges")) == 1
    assert "Nenhuma aresta" in window.report()

    def broken(*a, **kw):
        raise RuntimeError("falhou a leitura")

    monkeypatch.setattr(creases_module, "maya_suggest", broken)
    window.suggest_button.click()
    assert "falhou a leitura" in window.report()
    assert any("falhou a leitura" in c[1] for c in scene.named("warning"))


def test_record_strong_and_guide(win, monkeypatch):
    """Os botões gravam 3.0 / 1.0 na seleção (com o limiar padrão) e dizem quantas arestas."""
    window, _, _ = win
    values = []
    answer = {"count": 12}

    def fake_apply(value=3.0):
        values.append(value)
        return answer["count"]

    monkeypatch.setattr(creases_module, "maya_apply_selection", fake_apply)
    window.strong_button.click()
    assert values == [creases_module.STRONG_VALUE]
    assert "Crease 3 (forte) gravada em 12 aresta(s)" in window.report()
    window.guide_button.click()
    assert values[-1] == creases_module.GUIDE_VALUE
    assert "Crease 1 (guia) gravada em 12 aresta(s)" in window.report()

    answer["count"] = 0
    window.strong_button.click()
    assert "Nenhuma aresta selecionada" in window.report()

    # limiar acima de 3: o forte grava o próprio limiar (senão viraria guia)
    window.threshold_spin.setValue(4.0)
    window.strong_button.click()
    assert values[-1] == 4.0


def test_check_input_reports_and_selects(win, monkeypatch):
    """Verificar entrada mostra as mensagens de ``maya_check`` e seleciona as arestas com problema."""
    window, scene, _ = win
    cube = shapes.cube(2)
    sharp = creases_module.suggest_creases(cube, 30.0)
    check = check_input(cube, FeatureSet(sharp[:1]))
    calls = []

    def fake_check(name, angle_deg=30.0, strong_threshold=2.0):
        calls.append((name, angle_deg, strong_threshold))
        return check

    monkeypatch.setattr(creases_module, "maya_check", fake_check)
    window.threshold_spin.setValue(2.5)
    window.check_button.click()
    assert calls == [(TARGET, 30.0, 2.5)]
    report = window.report()
    for message in check.messages:
        assert message in report
    selected = scene.named("select_edges")
    assert len(selected) == 1 and len(selected[0][2]) == len(sharp)

    # entrada limpa: não seleciona nada
    check = check_input(cube, FeatureSet(sharp))
    window.check_button.click()
    assert len(scene.named("select_edges")) == 1
    assert "Entrada sem problemas de crease." in window.report()


def test_retopologize_success(win, monkeypatch):
    """Retopologizar passa as opções da janela, atualiza a barra e mostra relatório + notas."""
    window, scene, _ = win
    seen = {}
    result = _result()

    def fake_retopo(name, options, progress=None, cancel=None):
        seen["name"], seen["options"] = name, options
        seen["run_enabled"] = window.run_button.isEnabled()
        seen["cancel_enabled"] = window.cancel_button.isEnabled()
        seen["check_enabled"] = window.check_button.isEnabled()
        values = []
        for fraction, message in ((0.0, "Lendo a malha do Maya"), (0.4, "Casca 1 de 1: motor"), (1.0, "Pronto")):
            progress(fraction, message)
            values.append(window.progress_bar.value())
            assert cancel() is False
        seen["values"] = values
        seen["status"] = window.status_label.text()
        # o Maya seleciona o objeto novo no fim: a janela só relê a seleção depois
        scene.fire_selection("|pCube1_toptop")
        seen["target_during"] = window.target
        return "|pCube1_toptop", result

    monkeypatch.setattr(pl, "maya_retopologize", fake_retopo)
    window.mode_film.setChecked(True)
    window.detail_slider.setValue(7)
    window.threshold_spin.setValue(1.5)
    window.angle_spin.setValue(35.0)
    window.engine_combo.setCurrentIndex(1)
    window.run_button.click()

    options = seen["options"]
    assert seen["name"] == TARGET
    assert (options.mode, options.detail, options.strong_threshold) == ("film", 7, 1.5)
    assert (options.suggest_angle, options.engine, options.hard_edges) == (35.0, "maya_retopo", True)
    assert seen["run_enabled"] is False and seen["cancel_enabled"] is True and seen["check_enabled"] is False
    assert seen["values"] == [0, 400, 1000]
    assert seen["status"] == "Pronto"
    assert seen["target_during"] == TARGET

    report = window.report()
    assert report.startswith(format_report(result.report))
    assert "Objeto criado: pCube1_toptop" in report
    assert "- Limpeza: 2 vértice(s) soldado(s)." in report and "- Motor: aviso de teste" in report
    assert window.last_result is result and window.last_new == "|pCube1_toptop"
    assert window.progress_bar.value() == 1000
    assert window.run_button.isEnabled() and not window.cancel_button.isEnabled()
    assert window.target == "|pCube1_toptop"  # depois da execução, segue a seleção de novo
    assert not window.running


def test_retopologize_cancel_button(win, monkeypatch):
    """O clique em Cancelar (processado pelos callbacks) faz ``cancel()`` devolver verdadeiro."""
    from PySide6 import QtCore

    window, _, _ = win
    seen = {"polls": 0}

    def fake_retopo(name, options, progress=None, cancel=None):
        progress(0.1, "Rodando o motor")
        # o usuário clica em Cancelar; o clique só é tratado quando a janela processa eventos
        QtCore.QTimer.singleShot(0, window.cancel_button.click)
        while not cancel():
            seen["polls"] += 1
            assert seen["polls"] < 1000, "o clique em Cancelar nunca chegou"
        seen["cancel_enabled_after"] = window.cancel_button.isEnabled()
        raise EngineCancelled("execução cancelada pelo usuário")

    monkeypatch.setattr(pl, "maya_retopologize", fake_retopo)
    window.run_button.click()
    assert seen["cancel_enabled_after"] is False
    assert "cancelada pelo usuário" in window.report()
    assert window.run_button.isEnabled() and not window.cancel_button.isEnabled()
    assert not window.running and window._cancel_requested is False


def test_retopologize_pipeline_error_selects_culprits(win, monkeypatch):
    """Erro de malha non-manifold: mensagem no relatório, ``cmds.warning`` e arestas selecionadas."""
    window, scene, _ = win
    edges = np.array([[1, 2], [2, 5]])

    def fake_retopo(name, options, progress=None, cancel=None):
        raise PipelineError("Original é non-manifold (2 aresta(s) com mais de 2 faces).", edges=edges, stage="validar")

    monkeypatch.setattr(pl, "maya_retopologize", fake_retopo)
    window.run_button.click()
    report = window.report()
    assert "non-manifold" in report and "selecionados na viewport" in report
    selected = scene.named("select_edges")
    assert selected[-1][1] == TARGET and selected[-1][2].tolist() == edges.tolist()
    assert any("non-manifold" in c[1] for c in scene.named("warning"))
    assert window.run_button.isEnabled()

    # vértice em gravata: seleciona os vértices
    def bowtie(name, options, progress=None, cancel=None):
        raise PipelineError("vértice ligando faces só pela ponta", stage="validar", vertices=np.array([7, 8, 9, 12]))

    monkeypatch.setattr(pl, "maya_retopologize", bowtie)
    window.run_button.click()
    select = scene.named("select")[-1]
    assert select[1][0] == ["|pCube1|pCube1Shape.vtx[7:9]", "|pCube1|pCube1Shape.vtx[12:12]"]
    assert select[2] == {"replace": True}

    # erro na malha reduzida: a cópia já foi apagada, nada para selecionar
    count = len(scene.named("select_edges"))

    def reduced(name, options, progress=None, cancel=None):
        raise PipelineError("A malha reduzida é non-manifold.", edges=edges, stage="reduzir")

    monkeypatch.setattr(pl, "maya_retopologize", reduced)
    window.run_button.click()
    assert len(scene.named("select_edges")) == count
    assert "reduzida" in window.report()


@pytest.mark.parametrize(
    "error, expected",
    [
        (EngineTimeout("o motor passou de 600 s", workdir="C:/tmp/qw"), "Erro no motor: o motor passou de 600 s"),
        (EngineError("Casca 1 de 2 (12 triângulos): quadwild falhou"), "Casca 1 de 2"),
        (ZeroDivisionError("divisão por zero"), "Erro inesperado: divisão por zero"),
    ],
)
def test_retopologize_errors_never_escape(win, monkeypatch, error, expected):
    """Erros do motor e exceções inesperadas vão para o relatório, nunca para fora da janela."""
    window, scene, _ = win

    def fake_retopo(name, options, progress=None, cancel=None):
        progress(0.3, "motor")
        raise error

    monkeypatch.setattr(pl, "maya_retopologize", fake_retopo)
    window.run_button.click()
    assert expected in window.report()
    assert scene.named("warning")
    assert window.run_button.isEnabled() and not window.running
    if isinstance(error, ZeroDivisionError):
        assert "Traceback" in window.report()
    if isinstance(error, EngineTimeout):
        assert "C:/tmp/qw" in window.report()


def test_close_removes_callback(win, qapp):
    """Fechar a janela remove o callback ``SelectionChanged``; durante a execução, só cancela."""
    window, scene, _ = win
    (cid,) = scene.callbacks
    window._running = True
    window.close()
    assert window.isVisible() is False and cid in scene.callbacks  # fechamento ignorado
    assert window._cancel_requested is True
    window._running = False
    window._cancel_requested = False
    window.close()
    assert scene.removed == [cid] and not scene.callbacks
    _flush_deletes(qapp)
    # o Maya ainda chamaria o callback antigo: nada quebra
    scene.callbacks[cid] = ("SelectionChanged", window._on_selection_changed)
    scene.fire_selection(OTHER)


def test_show_is_singleton_and_parented(ui, qapp, monkeypatch):
    """``show()`` cria uma janela filha da janela principal do Maya e reaproveita a aberta."""
    from PySide6 import QtWidgets
    from shiboken6 import isValid

    window_module, _, _ = ui
    main = QtWidgets.QMainWindow()
    FakeMaya(monkeypatch, main_window=main)
    try:
        first = window_module.show()
        assert first.isVisible() and first.objectName() == "TopTopWindow"
        assert first.parentWidget() is main or first.parentWidget() == main
        second = window_module.show()
        assert second is first
        names = [w for w in qapp.topLevelWidgets() if w.objectName() == "TopTopWindow"]
        assert len(names) == 1

        # fechada, a próxima chamada cria outra (mesmo antes do deleteLater rodar)
        first.close()
        third = window_module.show()
        assert third is not first and third.isVisible()
        _flush_deletes(qapp)
        assert not isValid(first) and isValid(third)
        assert window_module.show() is third
        window_module.close()
        _flush_deletes(qapp)
        assert not isValid(third)
    finally:
        main.deleteLater()
        _flush_deletes(qapp)


def test_show_replaces_window_from_reloaded_module(ui, qapp):
    """Uma janela de uma versão antiga do módulo (reload) é fechada e trocada."""
    from PySide6 import QtWidgets

    window_module, _, _ = ui
    stale = QtWidgets.QWidget()
    stale.setObjectName("TopTopWindow")
    stale.setAttribute(window_module.QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
    stale.show()
    window = window_module.show()
    assert window is not stale and isinstance(window, window_module.TopTopWindow)
    assert not stale.isVisible()


# --------------------------------------------------------------------------
# Prateleira
# --------------------------------------------------------------------------


class FakeShelfMaya:
    """``maya.cmds`` e ``maya.mel`` de mentira para ``toptop.shelf``."""

    def __init__(self, monkeypatch, batch=False, shelf_exists=False, add_returns="TopTop") -> None:
        self.calls: list[tuple] = []
        self.shelves = {"TopTop"} if shelf_exists else set()
        self.buttons: dict[str, dict] = {}
        self.children: dict[str, list[str]] = {}
        cmds = types.ModuleType("maya.cmds")
        mel = types.ModuleType("maya.mel")

        def record(name):
            def call(*a, **kw):
                self.calls.append((name, a, kw))
                return getattr(self, "_" + name)(*a, **kw)

            return call

        for name in ("about", "shelfTabLayout", "shelfLayout", "shelfButton", "saveAllShelves"):
            setattr(cmds, name, record(name))

        def mel_eval(code):
            self.calls.append(("mel", code))
            if "$gShelfTopLevel" in code:
                return "ShelfLayout"
            if code.startswith("addNewShelfTab"):
                self.shelves.add("TopTop")
                return add_returns
            raise AssertionError(code)

        mel.eval = mel_eval
        self.batch = batch
        maya = types.ModuleType("maya")
        maya.cmds, maya.mel = cmds, mel
        monkeypatch.setitem(sys.modules, "maya", maya)
        monkeypatch.setitem(sys.modules, "maya.cmds", cmds)
        monkeypatch.setitem(sys.modules, "maya.mel", mel)

    def _about(self, **kw):
        return self.batch

    def _shelfTabLayout(self, name, **kw):
        return True if kw.get("exists") else None

    def _shelfLayout(self, name, **kw):
        if kw.get("exists"):
            return name in self.shelves
        if kw.get("query"):
            return list(self.children.get(name, [])) or None
        self.shelves.add(name)
        return name

    def _shelfButton(self, *a, **kw):
        if kw.get("exists"):
            return a[0] in self.buttons
        if kw.get("query"):
            return self.buttons[a[0]].get("docTag")
        if kw.get("edit"):
            self.buttons[a[0]].update({k: v for k, v in kw.items() if k != "edit"})
            return None
        name = f"shelfButton{len(self.buttons) + 1}"
        self.buttons[name] = dict(kw)
        self.children.setdefault(kw["parent"], []).append(name)
        return name

    def _saveAllShelves(self, *a, **kw):
        return None

    def names(self) -> list[str]:
        """Só o nome de cada chamada, na ordem."""
        return [c[0] for c in self.calls]


def test_button_command_adds_repo_and_opens_window(monkeypatch):
    """O código do botão põe o repositório no ``sys.path`` (uma vez) e chama ``show()``."""
    import toptop.shelf as shelf

    assert shelf.repo_root() == REPO_ROOT
    root = r"C:\Users\eu\Toptop's repo"
    code = shelf.button_command(root)
    calls = []
    fake_window = types.ModuleType("toptop.ui.window")
    fake_window.show = lambda: calls.append("show")
    monkeypatch.setitem(sys.modules, "toptop.ui.window", fake_window)
    import toptop.ui

    monkeypatch.setattr(toptop.ui, "window", fake_window, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    exec(compile(code, "<shelf>", "exec"), {})
    exec(compile(code, "<shelf>", "exec"), {})
    assert calls == ["show", "show"]
    assert sys.path.count(root) == 1 and sys.path[0] == root


def test_shelf_install_creates_and_reuses(monkeypatch, capsys):
    """``install`` cria a aba e o botão; rodar de novo atualiza o mesmo botão."""
    import toptop.shelf as shelf

    fake = FakeShelfMaya(monkeypatch)
    button = shelf.install(root="C:/repo")
    assert button == "shelfButton1"
    assert ("mel", 'addNewShelfTab "TopTop"') in fake.calls
    flags = fake.buttons[button]
    assert flags["parent"] == "TopTop" and flags["docTag"] == shelf.BUTTON_TAG
    assert flags["sourceType"] == "python" and flags["label"] == "TopTop"
    assert "'C:/repo'" in flags["command"] and "toptop.ui.window.show()" in flags["command"]
    assert len(flags["imageOverlayLabel"]) <= 6
    assert ("shelfTabLayout", ("ShelfLayout",), {"edit": True, "selectTab": "TopTop"}) in fake.calls
    assert ("saveAllShelves", ("ShelfLayout",), {}) in fake.calls
    out = capsys.readouterr().out
    assert "TopTop: botão criado" in out and "C:/repo" in out

    fake.calls.clear()
    again = shelf.install(root="D:/novo")
    assert again == button and len(fake.buttons) == 1
    assert not any(c[0] == "mel" and "addNewShelfTab" in c[1] for c in fake.calls)
    assert "'D:/novo'" in fake.buttons[button]["command"]
    assert "botão atualizado" in capsys.readouterr().out


def test_shelf_install_fallback_and_batch(monkeypatch, capsys):
    """Sem ``addNewShelfTab`` útil, cria a aba com ``shelfLayout``; sem interface, só avisa."""
    import toptop.shelf as shelf

    fake = FakeShelfMaya(monkeypatch, add_returns=None)
    fake_add = sys.modules["maya.mel"].eval

    def mel_eval(code):
        if code.startswith("addNewShelfTab"):
            fake.calls.append(("mel", code))
            return None  # não criou nada
        return fake_add(code)

    sys.modules["maya.mel"].eval = mel_eval
    shelf.install("Minha", root="C:/repo")
    assert ("shelfLayout", ("Minha",), {"parent": "ShelfLayout"}) in fake.calls
    assert fake.buttons["shelfButton1"]["parent"] == "Minha"

    batch = FakeShelfMaya(monkeypatch, batch=True)
    assert shelf.install() == ""
    assert batch.names() == ["about"]
    assert "sem interface" in capsys.readouterr().out


def test_shelf_uses_only_existing_flags():
    """Cada flag de ``cmds`` usado em shelf.py e window.py existe no Maya 2025 (se a referência existir)."""
    import ast
    import pickle

    path = os.environ.get("TOPTOP_MAYA_CMDS_PKL", "")
    if not path or not os.path.exists(path):
        pytest.skip("referência dos comandos do Maya não disponível (TOPTOP_MAYA_CMDS_PKL)")
    with open(path, "rb") as fh:
        reference = pickle.load(fh)[0]
    for module in ("toptop/shelf.py", "toptop/ui/window.py"):
        with open(os.path.join(REPO_ROOT, module), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "cmds"
            ):
                command = node.func.attr
                assert command in reference, f"{module}: cmds.{command} não existe"
                flags = reference[command]["flags"]
                for kw in node.keywords:
                    if kw.arg is not None:
                        assert kw.arg in flags, f"{module}: cmds.{command}(-{kw.arg}) não existe"
    import toptop.shelf as shelf

    flags = reference["shelfButton"]["flags"]
    for name in list(shelf.button_flags("pass")) + ["docTag"]:
        assert {"create", "edit"} <= set(flags[name]["modes"]), name
    assert "query" in flags["docTag"]["modes"] and "query" in reference["shelfLayout"]["flags"]["childArray"]["modes"]
    assert "edit" in reference["shelfTabLayout"]["flags"]["selectTab"]["modes"]
