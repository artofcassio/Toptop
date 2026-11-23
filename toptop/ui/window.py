"""Janela do TopTop no Maya (PySide6), aberta pelo botão da prateleira.

A janela segue a seleção (o "objeto alvo" é a primeira malha selecionada) e
só calcula quando um botão é clicado; a única conta automática é a estimativa
de quads ao lado do nível de detalhe: a malha é lida uma vez por objeto (só a
área e a diagonal ficam em cache) e qualquer nível ou modo sai dessa leitura.
Os botões releem a seleção antes de agir, então renomear ou mudar o objeto de
pai não deixa a janela com um caminho velho.

Uso, dentro do Maya::

    import toptop.ui.window
    toptop.ui.window.show()

O módulo importa só PySide6 no topo; ``maya`` e ``toptop.maya_bridge`` são
importados dentro das funções, então ele também carrega fora do Maya (testes).
As opções das Fases 2 e 3 já aparecem, desligadas, com a fase na dica.
"""

from __future__ import annotations

import functools
import traceback
from typing import Any

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..core import creases, pipeline, quality
from ..core.cleanup import DEFAULT_TRIANGLE_LIMIT
from ..core.creases import GUIDE_VALUE, STRONG_VALUE, InputCheck
from ..core.pipeline import (
    DETAIL_DEFAULT,
    DETAIL_MAX,
    DETAIL_MIN,
    SHELL_DIRECT,
    SHELL_FAILED,
    PipelineError,
    PipelineOptions,
    PipelineResult,
)
from ..engine import ENGINE_NAMES, get_engine
from ..engine.base import EngineCancelled, EngineError

OBJECT_NAME = "TopTopWindow"
WINDOW_TITLE = "TopTop: retopologia hard surface"
NO_TARGET_TEXT = "(selecione uma malha)"
PHASE2_TIP = "Fase 2"
PHASE3_TIP = "Fase 3"
#: espera (ms) antes de recalcular a estimativa, para não ler a malha a cada passo do slider
ESTIMATE_DELAY_MS = 150
#: o progresso vai de 0 a este valor na barra
PROGRESS_STEPS = 1000
#: faixa e passo do limite de redução (triângulos)
REDUCE_LIMIT_RANGE = (1_000, 10_000_000)
REDUCE_LIMIT_STEP = 10_000
#: motores que a janela lista, mas nunca deixa escolher, com o motivo
MANUAL_ONLY_ENGINES = {
    "quadremesher": (
        "a licença do Quad Remesher não cobre uso automatizado (a retopologia o chamaria sozinha, "
        "uma vez por casca); use o fluxo manual: toptop.engine.quadremesher.prepare_manual e "
        "collect_manual (veja HANDOFF_NEXT.md)"
    ),
}

#: instância aberta (a janela é única)
_window: TopTopWindow | None = None


# --------------------------------------------------------------------------
# Funções puras (testáveis sem Maya e sem janela)
# --------------------------------------------------------------------------


def format_int(value: int) -> str:
    """Inteiro com ponto separando os milhares (1.234.567)."""
    return f"{int(value):,}".replace(",", ".")


def strong_crease_value(threshold: float) -> float:
    """Crease gravado por "Gravar como forte": 3.0, ou o limiar se ele passar de 3."""
    return max(STRONG_VALUE, float(threshold))


def guide_crease_value(threshold: float) -> float:
    """Crease gravado por "Gravar como guia": 1.0, ou metade do limiar se ele for <= 2."""
    return min(GUIDE_VALUE, 0.5 * float(threshold))


def engine_choices() -> list[tuple[str, str, bool, str]]:
    """(nome, rótulo, disponível?, explicação) de cada motor de ``ENGINE_NAMES``.

    Os motores de ``MANUAL_ONLY_ENGINES`` aparecem sempre indisponíveis, com o motivo.
    """
    out = []
    for name in ENGINE_NAMES:
        try:
            engine = get_engine(name)
            label = engine.label or name
            ok, why = engine.availability()
        except Exception as exc:  # um adaptador quebrado não pode derrubar a janela
            label, ok, why = name, False, f"erro ao carregar o motor: {exc}"
        if name in MANUAL_ONLY_ENGINES:
            ok, why = False, MANUAL_ONLY_ENGINES[name]
        out.append((name, label, bool(ok), str(why)))
    return out


def chain_pairs(chains: list[np.ndarray]) -> np.ndarray:
    """Arestas (K, 2) das cadeias de vértices dadas (vértices seguidos de cada cadeia)."""
    parts = [
        np.stack([c[:-1], c[1:]], axis=1)
        for c in (np.asarray(c, dtype=np.int64).reshape(-1) for c in chains)
        if len(c) > 1
    ]
    if not parts:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(parts).astype(np.int64)


def check_problem_edges(check: InputCheck) -> np.ndarray:
    """Arestas para selecionar depois do "Verificar entrada": pontas soltas e quinas sem crease."""
    pairs = np.concatenate(
        [chain_pairs(check.dangling_chains), np.asarray(check.unmarked_sharp_edges, dtype=np.int64).reshape(-1, 2)]
    )
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0).reshape(-1, 2)


def format_check(target: str, check: InputCheck, threshold: float, selected: int) -> str:
    """Texto do "Verificar entrada" para a área de relatório."""
    lines = [
        f"== Verificar entrada: {target.split('|')[-1]} ==",
        (
            f"Creases: {check.num_strong} forte(s), {check.num_guide} guia(s) (limiar {threshold:g});"
            f" {check.marked_ratio:.1%} das arestas marcadas."
        ),
    ]
    lines.extend(f"- {m}" for m in check.messages)
    if selected:
        lines.append(f"{selected} aresta(s) com problema selecionada(s) na viewport.")
    lines.append("Resultado: " + ("sem problemas" if check.ok else "confira os avisos acima"))
    return "\n".join(lines)


def format_result(new: str, result: PipelineResult, options: PipelineOptions) -> str:
    """Relatório final: ``quality.format_report`` mais objeto criado, cascas, creases, tempos e notas."""
    lines = [quality.format_report(result.report)]
    if len(result.shell_reports) > 1:
        lines.append("")
        lines.append("Cascas:")
        status = list(result.shell_status) + [""] * (len(result.shell_reports) - len(result.shell_status))
        for k, (rep, how) in enumerate(zip(result.shell_reports, status), 1):
            if how == SHELL_DIRECT:
                state = "convertida direto (pequena demais para o motor)"
            elif how == SHELL_FAILED:
                state = "o motor falhou: convertida direto"
            else:
                state = "APROVADA" if rep.passed else "REPROVADA"
            lines.append(f"  {k}: {format_int(rep.quads)} quads, {state}")
    lines.append("")
    lines.append(f"Objeto criado: {new.split('|')[-1]}")
    if options.mode == "game" and options.hard_edges:
        lines.append(f"Hard edges: {format_int(len(result.hard_edges))} aresta(s) sobre creases fortes")
    else:
        lines.append("Hard edges: nenhuma (tudo suave)")
    if options.mode == "film":
        strong = int(result.crease_edges.strong_mask(float(options.strong_threshold)).sum())
        lines.append(
            f"Creases: {format_int(len(result.crease_edges))} aresta(s) de loop de feature"
            f" ({format_int(strong)} forte(s), {format_int(len(result.crease_edges) - strong)} guia(s))"
        )
    lines.append(
        f"Tempo: {result.seconds:.1f} s no total, {result.engine_seconds:.1f} s no motor"
        f" (pedido: {format_int(result.target_quads)} quads)"
    )
    if result.notes:
        lines.append("Notas:")
        lines.extend(f"- {n}" for n in result.notes)
    return "\n".join(lines)


def _vertex_components(shape: str, ids: np.ndarray) -> list[str]:
    """Componentes de vértice (``shape.vtx[a:b]``) em faixas compactas."""
    ids = np.unique(np.asarray(ids, dtype=np.int64).reshape(-1))
    ids = ids[ids >= 0]
    if ids.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(ids) != 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [ids.size - 1]])
    return [f"{shape}.vtx[{ids[a]}:{ids[b]}]" for a, b in zip(starts.tolist(), ends.tolist())]


def _remove_callbacks(ids: list) -> None:
    """Remove os callbacks do Maya da lista (e esvazia a lista); ignora os que já saíram."""
    if not ids:
        return
    try:
        import maya.api.OpenMaya as om
    except ImportError:
        ids.clear()
        return
    while ids:
        cid = ids.pop()
        try:
            om.MMessage.removeCallback(cid)
        except Exception:
            pass


def _warn(message: str) -> None:
    """``cmds.warning`` com a primeira linha da mensagem (sem Maya, não faz nada)."""
    try:
        from maya import cmds
    except ImportError:
        return
    first = message.strip().splitlines()[0] if message.strip() else "erro"
    cmds.warning(f"TopTop: {first}")


def maya_main_window() -> QtWidgets.QWidget | None:
    """Janela principal do Maya como ``QWidget`` (``None`` fora do Maya)."""
    try:
        from maya import OpenMayaUI
        from shiboken6 import wrapInstance
    except ImportError:
        return None
    ptr = OpenMayaUI.MQtUtil.mainWindow()
    if ptr is None:
        return None
    return wrapInstance(int(ptr), QtWidgets.QWidget)


# --------------------------------------------------------------------------
# Janela
# --------------------------------------------------------------------------


def _phase(widget: QtWidgets.QWidget, tip: str) -> QtWidgets.QWidget:
    """Desliga um controle de fase futura e põe a fase como dica."""
    widget.setEnabled(False)
    widget.setToolTip(tip)
    return widget


class TopTopWindow(QtWidgets.QWidget):
    """Janela principal: alvo, densidade, creases, motor, ações e relatório."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(OBJECT_NAME)
        self.setWindowTitle(WINDOW_TITLE)
        self.setWindowFlags(QtCore.Qt.WindowType.Window)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(540, 800)

        self._target: str | None = None
        self._running = False
        self._cancel_requested = False
        self._target_dirty = False
        self._closing = False
        #: (área, diagonal) de cada objeto já lido; qualquer nível e modo sai daqui
        self._estimate_inputs: dict[str, tuple[float, float]] = {}
        self._engines: list[tuple[str, str, bool, str]] = []
        self._callbacks: list = []
        self.last_result: PipelineResult | None = None
        self.last_new: str | None = None

        self._estimate_timer = QtCore.QTimer(self)
        self._estimate_timer.setSingleShot(True)
        self._estimate_timer.setInterval(ESTIMATE_DELAY_MS)
        self._estimate_timer.timeout.connect(self.update_estimate)

        self._build()
        self._fill_engines()
        self._install_selection_callback()
        # remove o callback mesmo se a janela for destruída sem closeEvent (Maya fechando)
        self.destroyed.connect(functools.partial(_remove_callbacks, self._callbacks))
        self.refresh_target()

    # ---------------------------------------------------------------- montagem

    def _build(self) -> None:
        """Cria os controles (seção 9 do plano)."""
        root = QtWidgets.QVBoxLayout(self)

        # Objeto alvo
        target_row = QtWidgets.QHBoxLayout()
        target_row.addWidget(QtWidgets.QLabel("Objeto alvo:"))
        self.target_label = QtWidgets.QLabel(NO_TARGET_TEXT)
        self.target_label.setObjectName("toptopTarget")
        font = self.target_label.font()
        font.setBold(True)
        self.target_label.setFont(font)
        self.target_label.setToolTip("Segue a seleção: a primeira malha selecionada é a retopologizada.")
        target_row.addWidget(self.target_label, 1)
        root.addLayout(target_row)

        # Densidade
        density = QtWidgets.QGroupBox("Densidade")
        form = QtWidgets.QFormLayout(density)
        mode_row = QtWidgets.QHBoxLayout()
        self.mode_game = QtWidgets.QRadioButton("Game")
        self.mode_game.setToolTip("Hard edges nas creases fortes; densidade do nível de detalhe.")
        self.mode_film = QtWidgets.QRadioButton("Film")
        self.mode_film.setToolTip("2,5x mais quads e tudo suave (para subdivisão).")
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.addButton(self.mode_game)
        self.mode_group.addButton(self.mode_film)
        self.mode_game.setChecked(True)
        mode_row.addWidget(self.mode_game)
        mode_row.addWidget(self.mode_film)
        mode_row.addStretch(1)
        form.addRow("Modo:", mode_row)

        detail_row = QtWidgets.QHBoxLayout()
        self.detail_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.detail_slider.setObjectName("toptopDetail")
        self.detail_slider.setRange(DETAIL_MIN, DETAIL_MAX)
        self.detail_slider.setValue(DETAIL_DEFAULT)
        self.detail_slider.setPageStep(1)
        self.detail_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.detail_slider.setTickInterval(1)
        self.detail_slider.setToolTip("1 = quads grandes, 10 = quads pequenos.")
        self.detail_value = QtWidgets.QLabel(str(DETAIL_DEFAULT))
        self.detail_value.setMinimumWidth(18)
        self.estimate_label = QtWidgets.QLabel("")
        self.estimate_label.setObjectName("toptopEstimate")
        self.estimate_label.setMinimumWidth(120)
        self.estimate_label.setToolTip("Estimativa de quads para o objeto alvo neste nível e modo.")
        detail_row.addWidget(self.detail_slider, 1)
        detail_row.addWidget(self.detail_value)
        detail_row.addWidget(self.estimate_label)
        form.addRow("Nível de detalhe:", detail_row)

        self.reduce_spin = QtWidgets.QSpinBox()
        self.reduce_spin.setObjectName("toptopReduceLimit")
        self.reduce_spin.setRange(*REDUCE_LIMIT_RANGE)
        self.reduce_spin.setSingleStep(REDUCE_LIMIT_STEP)
        self.reduce_spin.setValue(DEFAULT_TRIANGLE_LIMIT)
        self.reduce_spin.setGroupSeparatorShown(True)
        self.reduce_spin.setSuffix(" triângulos")
        self.reduce_spin.setToolTip(
            "Acima deste número de triângulos, o Maya reduz uma cópia antes do motor (o original não muda). "
            "Aumente para o Maya reduzir menos."
        )
        form.addRow("Limite de redução:", self.reduce_spin)
        root.addWidget(density)

        # Creases
        crease_box = QtWidgets.QGroupBox("Creases")
        cform = QtWidgets.QFormLayout(crease_box)
        suggest_row = QtWidgets.QHBoxLayout()
        self.suggest_button = QtWidgets.QPushButton("Sugerir creases")
        self.suggest_button.setToolTip(
            "Seleciona as arestas com quina acima do ângulo. Nada é gravado: ajuste a seleção e grave."
        )
        self.angle_spin = QtWidgets.QDoubleSpinBox()
        self.angle_spin.setObjectName("toptopAngle")
        self.angle_spin.setRange(1.0, 180.0)
        self.angle_spin.setDecimals(1)
        self.angle_spin.setSingleStep(5.0)
        self.angle_spin.setValue(float(creases.DEFAULT_SUGGEST_ANGLE))
        self.angle_spin.setSuffix("°")
        self.angle_spin.setToolTip(
            "Ângulo da sugestão e do Verificar entrada; sem nenhuma crease marcada, "
            "a retopologia usa as quinas acima dele."
        )
        suggest_row.addWidget(self.suggest_button)
        suggest_row.addWidget(QtWidgets.QLabel("Ângulo:"))
        suggest_row.addWidget(self.angle_spin)
        suggest_row.addStretch(1)
        cform.addRow(suggest_row)

        record_row = QtWidgets.QHBoxLayout()
        self.strong_button = QtWidgets.QPushButton("Gravar como forte")
        self.strong_button.setToolTip("Grava crease forte nas arestas selecionadas (vira edge loop e hard edge).")
        self.guide_button = QtWidgets.QPushButton("Gravar como guia")
        self.guide_button.setToolTip("Grava crease guia nas arestas selecionadas (vira edge loop, fica suave).")
        record_row.addWidget(self.strong_button)
        record_row.addWidget(self.guide_button)
        cform.addRow(record_row)

        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setObjectName("toptopThreshold")
        self.threshold_spin.setRange(0.1, 10.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(float(creases.DEFAULT_STRONG_THRESHOLD))
        self.threshold_spin.setToolTip("Crease a partir deste valor é forte; abaixo, guia.")
        cform.addRow("Limiar de crease forte:", self.threshold_spin)

        self.hard_edges_check = QtWidgets.QCheckBox("Hard edges nas creases fortes")
        self.hard_edges_check.setChecked(True)
        cform.addRow(self.hard_edges_check)
        root.addWidget(crease_box)

        # Fases 2 e 3 (presentes, desligadas)
        later = QtWidgets.QGroupBox("Próximas fases")
        lform = QtWidgets.QFormLayout(later)
        sym_row = QtWidgets.QHBoxLayout()
        self.symmetry_check = _phase(QtWidgets.QCheckBox("Simetria"), PHASE2_TIP)
        self.symmetry_axis = QtWidgets.QComboBox()
        self.symmetry_axis.addItems(["X", "Y", "Z"])
        _phase(self.symmetry_axis, PHASE2_TIP)
        sym_row.addWidget(self.symmetry_check)
        sym_row.addWidget(self.symmetry_axis)
        sym_row.addStretch(1)
        lform.addRow(sym_row)
        self.weighted_normals_check = _phase(QtWidgets.QCheckBox("Normais ponderadas"), PHASE2_TIP)
        lform.addRow(self.weighted_normals_check)
        support_row = QtWidgets.QHBoxLayout()
        self.support_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.support_slider.setRange(0, 10)
        self.support_slider.setValue(2)
        _phase(self.support_slider, PHASE3_TIP)
        self.support_value = _phase(QtWidgets.QLabel("2"), PHASE3_TIP)
        support_row.addWidget(self.support_slider, 1)
        support_row.addWidget(self.support_value)
        self.support_label = _phase(QtWidgets.QLabel("Distância do support loop:"), PHASE3_TIP)
        lform.addRow(self.support_label, support_row)
        self.transfer_check = _phase(QtWidgets.QCheckBox("Transferir creases"), PHASE3_TIP)
        lform.addRow(self.transfer_check)
        self.circles_check = _phase(QtWidgets.QCheckBox("Ajustar círculos para múltiplos de 4"), PHASE3_TIP)
        lform.addRow(self.circles_check)
        self.uvs_check = _phase(QtWidgets.QCheckBox("Gerar UVs"), PHASE3_TIP)
        lform.addRow(self.uvs_check)
        root.addWidget(later)

        # Motor
        engine_row = QtWidgets.QHBoxLayout()
        engine_row.addWidget(QtWidgets.QLabel("Motor:"))
        self.engine_combo = QtWidgets.QComboBox()
        self.engine_combo.setObjectName("toptopEngine")
        engine_row.addWidget(self.engine_combo, 1)
        root.addLayout(engine_row)

        # Ações
        buttons = QtWidgets.QHBoxLayout()
        self.check_button = QtWidgets.QPushButton("Verificar entrada")
        self.check_button.setToolTip("Aponta pontas de crease soltas e quinas sem crease (e as seleciona).")
        self.run_button = QtWidgets.QPushButton("Retopologizar")
        self.run_button.setObjectName("toptopRun")
        self.cancel_button = QtWidgets.QPushButton("Cancelar")
        self.cancel_button.setEnabled(False)
        self.heatmap_button = QtWidgets.QPushButton("Mapa de calor")
        self.heatmap_button.setObjectName("toptopHeatmap")
        self.heatmap_button.setCheckable(True)
        self.heatmap_button.setToolTip(
            "Pinta o resultado pelo desvio até o original: azul = em cima da superfície, "
            "vermelho = na meta de desvio (0,5% da diagonal) ou acima. Clique de novo para esconder."
        )
        buttons.addWidget(self.check_button)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.heatmap_button)
        root.addLayout(buttons)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, PROGRESS_STEPS)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setObjectName("toptopStatus")
        root.addWidget(self.status_label)

        self.report_text = QtWidgets.QPlainTextEdit()
        self.report_text.setObjectName("toptopReport")
        self.report_text.setReadOnly(True)
        self.report_text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.report_text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.report_text.setPlaceholderText("O relatório aparece aqui.")
        root.addWidget(self.report_text, 1)

        # Sinais
        self.mode_game.toggled.connect(self._mode_changed)
        self.detail_slider.valueChanged.connect(self._detail_changed)
        self.suggest_button.clicked.connect(self.suggest)
        self.strong_button.clicked.connect(self.record_strong)
        self.guide_button.clicked.connect(self.record_guide)
        self.check_button.clicked.connect(self.check_input)
        self.run_button.clicked.connect(self.retopologize)
        self.cancel_button.clicked.connect(self.request_cancel)
        self.heatmap_button.clicked.connect(self.toggle_heatmap)
        self.engine_combo.currentIndexChanged.connect(self._update_buttons)

    def _fill_engines(self) -> None:
        """Lista os motores no combo; os indisponíveis ficam desligados com o motivo na dica."""
        self._engines = engine_choices()
        self.engine_combo.blockSignals(True)
        self.engine_combo.clear()
        model = self.engine_combo.model()
        first_ok = -1
        for i, (name, label, ok, why) in enumerate(self._engines):
            text = label if ok else f"{label} (indisponível)"
            self.engine_combo.addItem(text, name)
            self.engine_combo.setItemData(i, why, QtCore.Qt.ItemDataRole.ToolTipRole)
            if isinstance(model, QtGui.QStandardItemModel):
                model.item(i).setEnabled(ok)
            if ok and first_ok < 0:
                first_ok = i
        self.engine_combo.setCurrentIndex(max(first_ok, 0) if self._engines else -1)
        self.engine_combo.blockSignals(False)
        self._update_buttons()

    def _install_selection_callback(self) -> None:
        """Liga o callback ``SelectionChanged`` do Maya (sem Maya, não faz nada)."""
        try:
            import maya.api.OpenMaya as om
        except ImportError:
            return
        cid = om.MEventMessage.addEventCallback("SelectionChanged", self._on_selection_changed)
        self._callbacks.append(cid)

    # ------------------------------------------------------------- estado

    @property
    def target(self) -> str | None:
        """Objeto alvo atual (transform, caminho completo) ou ``None``."""
        return self._target

    @property
    def running(self) -> bool:
        """Verdadeiro durante a retopologia."""
        return self._running

    def selected_engine(self) -> tuple[str, bool, str]:
        """(nome, disponível?, explicação) do motor escolhido no combo."""
        i = self.engine_combo.currentIndex()
        if i < 0 or i >= len(self._engines):
            return "", False, "nenhum motor na lista"
        name, _, ok, why = self._engines[i]
        return name, ok, why

    def mode(self) -> str:
        """``game`` ou ``film``."""
        return "film" if self.mode_film.isChecked() else "game"

    def options(self) -> PipelineOptions:
        """``PipelineOptions`` com os valores dos controles."""
        name, _, _ = self.selected_engine()
        return PipelineOptions(
            mode=self.mode(),
            detail=int(self.detail_slider.value()),
            strong_threshold=float(self.threshold_spin.value()),
            hard_edges=bool(self.hard_edges_check.isChecked()),
            suggest_angle=float(self.angle_spin.value()),
            reduce_limit=int(self.reduce_spin.value()),
            engine=name or "quadwild",
        )

    def report(self) -> str:
        """Texto atual da área de relatório."""
        return self.report_text.toPlainText()

    def _set_report(self, text: str) -> None:
        """Troca o texto da área de relatório."""
        self.report_text.setPlainText(text)

    def _error(self, text: str) -> None:
        """Mostra um erro no relatório e como ``cmds.warning``."""
        self._set_report(text)
        _warn(text)

    def _update_buttons(self, *_: Any) -> None:
        """Liga e desliga os botões conforme o alvo, o motor e a execução."""
        has_target = self._target is not None
        idle = not self._running
        _, engine_ok, engine_why = self.selected_engine()
        self.suggest_button.setEnabled(idle and has_target)
        self.check_button.setEnabled(idle and has_target)
        self.strong_button.setEnabled(idle)
        self.guide_button.setEnabled(idle)
        self.run_button.setEnabled(idle and has_target and engine_ok)
        self.cancel_button.setEnabled(self._running)
        self.heatmap_button.setEnabled(idle and self.last_result is not None and self.last_new is not None)
        self.engine_combo.setEnabled(idle)
        if not engine_ok:
            self.run_button.setToolTip(f"Motor indisponível: {engine_why}")
        elif not has_target:
            self.run_button.setToolTip("Selecione uma malha.")
        else:
            self.run_button.setToolTip("Cria um objeto novo retopologizado; o original não muda.")
        self.engine_combo.setToolTip(engine_why)

    # ------------------------------------------------------- seleção e estimativa

    def _on_selection_changed(self, *_: Any) -> None:
        """Callback do Maya: a seleção mudou."""
        try:
            from shiboken6 import isValid
        except ImportError:  # pragma: no cover - PySide6 sempre traz o shiboken6
            isValid = None
        if isValid is not None and not isValid(self):
            return
        self.refresh_target()

    def refresh_target(self) -> None:
        """Relê a seleção e atualiza o rótulo do alvo (adiado enquanto a retopologia roda)."""
        if self._running:
            self._target_dirty = True
            return
        self._target_dirty = False
        try:
            from .. import maya_bridge

            target = maya_bridge.selected_mesh()
        except Exception:
            target = None
        changed = target != self._target
        self._target = target
        if target is None:
            self.target_label.setText(NO_TARGET_TEXT)
            self.target_label.setToolTip("Selecione uma malha poligonal.")
        else:
            self.target_label.setText(target.split("|")[-1])
            self.target_label.setToolTip(target)
        if changed:
            self._estimate_inputs.clear()
            self.schedule_estimate()
        self._update_buttons()

    def schedule_estimate(self) -> None:
        """Recalcula a estimativa daqui a pouco (junta mudanças seguidas)."""
        self._estimate_timer.start()

    def update_estimate(self) -> None:
        """Mostra a estimativa de quads: lê área e diagonal uma vez por alvo (``pipeline.maya_estimate_inputs``)."""
        self._estimate_timer.stop()
        if self._running:
            return
        target = self._target
        if target is None:
            self.estimate_label.setText("")
            return
        try:
            if target not in self._estimate_inputs:
                self._estimate_inputs[target] = pipeline.maya_estimate_inputs(target)
            area, diagonal = self._estimate_inputs[target]
            quads = int(pipeline.estimate_from(area, diagonal, int(self.detail_slider.value()), self.mode()))
        except Exception as exc:
            self.estimate_label.setText("≈ ? quads")
            self.estimate_label.setToolTip(f"Não deu para estimar: {exc}")
            return
        self.estimate_label.setText(f"≈ {format_int(quads)} quads")
        self.estimate_label.setToolTip("Estimativa de quads para o objeto alvo neste nível e modo.")

    def _detail_changed(self, value: int) -> None:
        """Slider de detalhe mexeu."""
        self.detail_value.setText(str(int(value)))
        self.schedule_estimate()

    def _mode_changed(self, *_: Any) -> None:
        """Game/Film mudou: estimativa nova; no Film não há hard edges."""
        film = self.mode() == "film"
        self.hard_edges_check.setEnabled(not film)
        self.hard_edges_check.setToolTip(
            "No Film tudo fica suave." if film else "Arestas sobre creases fortes ficam duras."
        )
        self.schedule_estimate()

    # -------------------------------------------------------------- ações

    def suggest(self) -> None:
        """"Sugerir creases": seleciona as quinas acima do ângulo; não grava nada."""
        self.refresh_target()  # renomear ou mudar de pai não dispara SelectionChanged
        target = self._target
        if target is None:
            self._set_report("Selecione uma malha para sugerir creases.")
            return
        angle = float(self.angle_spin.value())
        try:
            pairs = creases.maya_suggest(target, angle, select=False)
            if len(pairs):
                from .. import maya_bridge

                maya_bridge.select_edges(target, pairs)
        except Exception as exc:
            self._error(f"Erro ao sugerir creases: {exc}")
            return
        if len(pairs):
            self._set_report(
                f"{len(pairs)} aresta(s) com quina acima de {angle:g}° selecionada(s) em "
                f"{target.split('|')[-1]}.\nAjuste a seleção e use \"Gravar como forte\" ou "
                "\"Gravar como guia\". Nada foi gravado."
            )
        else:
            self._set_report(f"Nenhuma aresta com quina acima de {angle:g}° em {target.split('|')[-1]}.")

    def _record(self, value: float, kind: str) -> None:
        """Grava ``value`` como crease nas arestas selecionadas."""
        try:
            count = int(creases.maya_apply_selection(value))
        except Exception as exc:
            self._error(f"Erro ao gravar creases: {exc}")
            return
        if count == 0:
            self._set_report(
                "Nenhuma aresta selecionada: selecione arestas (ou use \"Sugerir creases\") antes de gravar."
            )
        else:
            self._set_report(f"Crease {value:g} ({kind}) gravada em {count} aresta(s).")

    def record_strong(self) -> None:
        """"Gravar como forte" na seleção (um passo de desfazer)."""
        self._record(strong_crease_value(self.threshold_spin.value()), "forte")

    def record_guide(self) -> None:
        """"Gravar como guia" na seleção (um passo de desfazer)."""
        self._record(guide_crease_value(self.threshold_spin.value()), "guia")

    def check_input(self) -> None:
        """"Verificar entrada": mostra os avisos e seleciona as arestas com problema."""
        self.refresh_target()  # renomear ou mudar de pai não dispara SelectionChanged
        target = self._target
        if target is None:
            self._set_report("Selecione uma malha para verificar.")
            return
        threshold = float(self.threshold_spin.value())
        try:
            check = creases.maya_check(target, float(self.angle_spin.value()), threshold)
            problems = check_problem_edges(check)
            if len(problems):
                from .. import maya_bridge

                maya_bridge.select_edges(target, problems)
        except Exception as exc:
            self._error(f"Erro ao verificar a entrada: {exc}")
            return
        self._set_report(format_check(target, check, threshold, len(problems)))

    def request_cancel(self) -> None:
        """Botão Cancelar: pede para parar no próximo ponto de checagem."""
        if self._running:
            self._cancel_requested = True
            self.status_label.setText("Cancelando...")
            self.cancel_button.setEnabled(False)

    def _progress(self, fraction: float, message: str) -> None:
        """Callback de progresso do pipeline: atualiza a barra e deixa a interface respirar."""
        self.progress_bar.setValue(round(min(max(float(fraction), 0.0), 1.0) * PROGRESS_STEPS))
        if not self._cancel_requested:
            self.status_label.setText(str(message))
        QtWidgets.QApplication.processEvents()

    def _is_cancelled(self) -> bool:
        """Callback de cancelamento (também processa eventos, para o Cancelar ser ouvido)."""
        QtWidgets.QApplication.processEvents()
        return self._cancel_requested

    def _select_problem(self, target: str, exc: PipelineError) -> int:
        """Seleciona no original as arestas (ou vértices) apontados pelo erro."""
        if exc.stage != "validar":
            return 0
        from .. import maya_bridge

        if len(exc.edges):
            return maya_bridge.select_edges(target, exc.edges)
        if len(exc.vertices):
            from maya import cmds

            components = _vertex_components(maya_bridge.shape_of(target), exc.vertices)
            if components:
                cmds.select(components, replace=True)
            return len(components)
        return 0

    def retopologize(self) -> None:
        """"Retopologizar": roda ``pipeline.maya_retopologize`` no alvo e mostra o relatório."""
        if self._running:
            return
        self.refresh_target()  # renomear ou mudar de pai não dispara SelectionChanged
        target = self._target
        if target is None:
            return
        name, ok, why = self.selected_engine()
        if not ok:
            self._error(f"Motor indisponível: {why}")
            return
        options = self.options()
        self._running = True
        self._cancel_requested = False
        self._update_buttons()
        self.progress_bar.setValue(0)
        self.status_label.setText("Começando...")
        self._set_report(f"Retopologizando {target.split('|')[-1]} com {name}...")
        QtWidgets.QApplication.processEvents()
        try:
            new, result = pipeline.maya_retopologize(
                target, options, progress=self._progress, cancel=self._is_cancelled
            )
        except EngineCancelled:
            self.status_label.setText("Cancelado.")
            self._set_report("Retopologia cancelada pelo usuário. Nada foi criado.")
        except PipelineError as exc:
            self.status_label.setText("Erro na entrada.")
            text = f"Erro: {exc}"
            try:
                selected = self._select_problem(target, exc)
            except Exception:
                selected = 0
            if selected:
                text += "\nOs culpados estão selecionados na viewport."
            self._error(text)
        except EngineError as exc:
            self.status_label.setText("Erro no motor.")
            self._error(f"Erro no motor: {exc}")
        except Exception as exc:
            self.status_label.setText("Erro inesperado.")
            self._error(f"Erro inesperado: {exc}\n\n{traceback.format_exc()}")
        else:
            self.last_new, self.last_result = new, result
            self.heatmap_button.setChecked(False)
            self.progress_bar.setValue(PROGRESS_STEPS)
            self.status_label.setText("Pronto.")
            self._set_report(format_result(new, result, options))
        finally:
            self._running = False
            self._cancel_requested = False
            self._update_buttons()
        if self._target_dirty:
            self.refresh_target()
        self.schedule_estimate()  # o slider pode ter mexido durante a execução

    def toggle_heatmap(self) -> None:
        """"Mapa de calor": pinta (ou despinta) o último objeto criado pelo desvio até o original."""
        new, result = self.last_new, self.last_result
        if new is None or result is None:
            self.heatmap_button.setChecked(False)
            return
        show = self.heatmap_button.isChecked()
        try:
            if show:
                pipeline.maya_show_heatmap(new, result)
                self.status_label.setText(
                    "Mapa de calor: azul = sobre o original, vermelho = na meta de desvio ou acima."
                )
            else:
                pipeline.maya_hide_heatmap(new)
                self.status_label.setText("Mapa de calor escondido.")
        except Exception as exc:
            self.heatmap_button.setChecked(False)
            self._error(f"Não foi possível mostrar o mapa de calor em {new.split('|')[-1]}: {exc}")

    # ------------------------------------------------------------ fechamento

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Fechar durante a retopologia só pede o cancelamento; senão solta o callback do Maya."""
        if self._running:
            self.request_cancel()
            self._set_report("Cancelando a retopologia; feche a janela quando ela parar.")
            event.ignore()
            return
        self._closing = True
        self._estimate_timer.stop()
        _remove_callbacks(self._callbacks)
        super().closeEvent(event)


# --------------------------------------------------------------------------
# Abrir (janela única)
# --------------------------------------------------------------------------


def _is_alive(widget: QtWidgets.QWidget | None) -> bool:
    """Verdadeiro se o objeto C++ do widget ainda existe e a janela não foi fechada."""
    if widget is None or getattr(widget, "_closing", False):
        return False
    try:
        from shiboken6 import isValid
    except ImportError:  # pragma: no cover
        return True
    return bool(isValid(widget))


def find_window() -> TopTopWindow | None:
    """A janela do TopTop já aberta, se houver (fecha versões de um módulo recarregado)."""
    if _is_alive(_window) and isinstance(_window, TopTopWindow):
        return _window
    app = QtWidgets.QApplication.instance()
    if app is None:
        return None
    for widget in app.topLevelWidgets():
        if widget.objectName() != OBJECT_NAME or not _is_alive(widget):
            continue
        if isinstance(widget, TopTopWindow):
            return widget
        widget.close()  # sobra de um reload do módulo
        widget.deleteLater()
    return None


def show() -> TopTopWindow:
    """Abre a janela do TopTop (ou traz para a frente a que já está aberta)."""
    global _window
    if QtWidgets.QApplication.instance() is None:
        raise RuntimeError("A janela do TopTop precisa do Maya com interface (não há QApplication).")
    window = find_window()
    if window is None:
        window = TopTopWindow(maya_main_window())
    else:
        window.refresh_target()
    _window = window
    window.show()
    window.raise_()
    window.activateWindow()
    return window


def close() -> None:
    """Fecha a janela do TopTop, se estiver aberta."""
    global _window
    window = find_window()
    if window is not None:
        window.close()
    _window = None
