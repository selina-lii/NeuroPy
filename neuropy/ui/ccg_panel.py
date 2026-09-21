"""CCG main view panel."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.Qt.QtCore import Qt, Signal, QObject, QRectF, QTimer
from pyqtgraph.Qt.QtWidgets import (
    QApplication,
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QScrollArea, QFrame, QLabel, QPushButton, QCheckBox,
    QRadioButton, QButtonGroup, QSpinBox, QDoubleSpinBox,
    QComboBox, QInputDialog, QLineEdit, QSizePolicy, QMenu,
    QToolButton, QSlider, QGroupBox, QFileDialog, QMessageBox,
)
from pyqtgraph.Qt.QtGui import QAction, QActionGroup
from neuropy.analyses.ccg_transforms import NormalizeBy, CCGNorm, ConnectionStrength
from neuropy.analyses.jitter import compute_jbsi, JitterConfig
from neuropy.analyses import correlations
from neuropy.analyses.ms_connectivity import Key, EranConv, _multiple_correction
from neuropy.plotting.ccg import (
    RenderContext, JitterOverlay, TitleConfig, PlotStyle,
    test_window_bin_mask, test_window_span_ms, render_ccg_png,
    ACG_REF_COLOR, ACG_TGT_COLOR, WF_COLOR,
)
from neuropy.analyses.ccg_transforms import (_fill_waveform, lag_window_bins,
                                             peak_waveform_on_lag_axis)
from neuropy.ui.ui_common import qt_dark_mode, LRUCache
from neuropy.ui.utils import (chip_button, CycleButton, FlowLayout, CollapsibleSection,
                              ArrowChipBar, MetricInput, SliderWithInput,
                              has_primary_modifier, small_font_pt,
                              widget_row, radio_group, set_checked_quietly,
                              apply_plot_chrome, plot_pen)

if TYPE_CHECKING:
    from neuropy.ui.app_state import AppState

pg.setConfigOptions(antialias=True)

CH_PER_SHANK = 16
_Y_PAD = 0.05   # shared by every overlay: equal padding puts every zero on one line
PVAL_COLOR = '#e74c3c'

_CHIP_STYLE = (
    "QPushButton { border: 1px solid #bbb; border-radius: 3px; "
    "padding: 1px 6px; background: #e8e8e8; }"
    "QPushButton[sig=true] { background: #90EE90; }"
    "QPushButton[active=true] { background: #4a7fd4; color: white; }"
    "QPushButton[active=true][sig=true] { background: #4CAF50; color: white; }"
    "QPushButton[stacked=true] { background: #7fb87f; color: white; }"
    "QPushButton[selected=true] { border: 2px solid #4a7fd4; }"
)


class SegmentBar(QWidget):
    """Scrollable segment chip row with lo|hi and CS chips."""

    def __init__(self, nav: 'AppState', parent=None):
        super().__init__(parent)
        self.nav = nav
        self._chips: dict[int, QPushButton] = {}   # seg_idx → chip widget
        self._selected: set[int] = set()            # multi-select, display-only
        self._build()
        nav.segment_changed.connect(self._refresh)
        nav.stacked_segments_changed.connect(self._refresh)
        nav.resolution_changed.connect(self._on_lo_hi_btn_changed)
        nav.key_changed.connect(self.rebuild)
        nav.custom_segs_changed.connect(self.rebuild)
        nav.pair_changed.connect(self._on_pair_sig_changed)
        nav.sig_threshold_changed.connect(self._on_pair_sig_changed)
        nav.cs_overlay_changed.connect(self._on_cs_overlay_changed)

    def refresh_font(self):
        self._seg_lbl.setStyleSheet(f"font-size: {small_font_pt()}pt;")

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(3)

        self._seg_lbl = QLabel("Segments:")
        self._seg_lbl.setStyleSheet(f"font-size: {small_font_pt()}pt;")
        root.addWidget(self._seg_lbl)

        self._chip_bar = ArrowChipBar(
            self, on_left=lambda: self._nav_step(-1),
            on_right=lambda: self._nav_step(+1))
        root.addWidget(self._chip_bar, stretch=1)   # only the chips absorb spare width

        self._lo_hi_btn = chip_button("lo|hi", checkable=True)
        self._lo_hi_btn.toggled.connect(
            lambda on: self.nav.set_resolution("lo_hi" if on else "lo"))
        self._cs_btn = chip_button("CS", checkable=True)
        self._cs_btn.toggled.connect(self.nav.set_cs_overlay)
        root.addLayout(widget_row(None, self._lo_hi_btn, self._cs_btn,
                                  stretch=False))

        self.rebuild()

    def rebuild(self):
        self._chip_bar.clear()
        self._chips.clear()
        self._selected.clear()
        nav = self.nav
        labels = nav.segment_names()
        for i, name in enumerate(labels):
            self._add_chip(name, i, bold=(i == 0))
            if i == 0 and len(labels) > 1:
                self._chip_bar.add_widget(self._vline())
        self._refresh()
        self._on_pair_sig_changed()

    def _add_chip(self, label: str, seg_idx: int, bold: bool = False):
        btn = QPushButton(label)
        btn.setCheckable(False)
        btn.setFlat(False)
        f = btn.font(); f.setBold(bold); btn.setFont(f)
        btn.setStyleSheet(_CHIP_STYLE)
        btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        btn.clicked.connect(lambda _checked, i=seg_idx: self._on_chip_click(i))
        btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        btn.customContextMenuRequested.connect(
            lambda pos, i=seg_idx, b=btn: self._show_chip_menu(i, b.mapToGlobal(pos)))
        self._chip_bar.add_widget(btn)
        self._chips[seg_idx] = btn

    def _on_chip_click(self, seg_idx: int):
        if has_primary_modifier(QtWidgets.QApplication.keyboardModifiers()):
            if seg_idx in self._selected:
                self._selected.remove(seg_idx)
            else:
                self._selected.add(seg_idx)
            self._refresh()
        else:
            self.nav.clear_stacked_segments()   # a plain click is a single-segment view
            self.nav.set_current_segment(self.nav.segment_name(seg_idx))

    def _show_chip_menu(self, seg_idx: int, global_pos):
        nav = self.nav
        menu = QMenu(self)
        labels = ([nav.segment_name(i) for i in sorted(self._selected)]
                  if self._selected else [nav.segment_name(seg_idx)])
        is_stacked = all(l in nav.stacked_segments for l in labels)
        verb = "Unstack" if is_stacked else "Stack"
        menu.addAction(f"{verb} segment" + (f"s ({len(labels)})" if len(labels) > 1 else ""),
                       lambda: self._stack_labels(labels))
        menu.addSeparator()
        transpose = menu.addAction("Transpose rows/columns", nav.toggle_stacked_transposed)
        transpose.setCheckable(True)
        transpose.setChecked(nav.stacked_transposed)
        if nav.stacked_segments:
            menu.addAction(f"Clear stacked ({len(nav.stacked_segments)})",
                           nav.clear_stacked_segments)
        menu.exec(global_pos)

    def _stack_labels(self, labels: list):
        self._selected.clear()   # multi-select is transient: consumed by the stack action
        self.nav.toggle_stacked_segments(labels)

    def _nav_step(self, step: int):
        """◀/▶ navigate to the prev/next segment (cyclic over All + real + custom)."""
        nav = self.nav
        names = nav.segment_names()
        if not names:
            return
        cur = nav.segment_index(nav.current_segment)
        nav.set_current_segment(nav.segment_name((cur + step) % len(names)))

    def _refresh(self, *_):
        nav = self.nav
        if len(self._chips) != len(nav.segment_names()):  # lazy-load attached a segment
            self.rebuild()
            return
        active_idx = nav.segment_index(nav.current_segment)
        stacked = nav.stacked_segments
        inds = nav.current_pair_inds
        ref, tgt = (int(inds[0]), int(inds[1])) if inds is not None else (None, None)
        for idx, btn in self._chips.items():
            is_stacked = nav.segment_name(idx) in stacked
            sig = nav.is_significant(ref, tgt, idx) if ref is not None else False
            btn.setProperty("active",  idx == active_idx and not is_stacked)
            btn.setProperty("stacked", is_stacked)
            btn.setProperty("selected", idx in self._selected)
            btn.setProperty("sig",     sig)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _on_pair_sig_changed(self, *_):
        self._refresh()

    def _on_lo_hi_btn_changed(self, res: str):
        set_checked_quietly(self._lo_hi_btn, res == "lo_hi")

    def _on_cs_overlay_changed(self, active: bool):
        set_checked_quietly(self._cs_btn, active)

    @staticmethod
    def _vline() -> QFrame:
        f = QFrame(); f.setFrameShape(QFrame.Shape.VLine); f.setFixedWidth(1)
        return f


class NormSection(CollapsibleSection):
    # Backend: nav.active_norms / nav.set_active_norms (AppState)
    #          consumed by CCGNorm.apply() in ccg_transforms.py

    norms_changed   = Signal(object)   # set[NormalizeBy] → nav.set_active_norms
    scale_changed   = Signal(object)   # str | None      → nav.set_same_scale_mode
    apply_requested = Signal()

    _NORM_OPTIONS = [
        (NormalizeBy.REF_FRATE,    "Ref f-rate"),
        (NormalizeBy.TARGET_FRATE, "Tgt f-rate"),
        (NormalizeBy.TIME_SPAN,    "Time (hr)"),
        (NormalizeBy.TIME_SECOND,  "Time (sec)"),
        (NormalizeBy.TOTAL_AREA,   "CCG total area"),
        (NormalizeBy.BASELINE,     "Subtract baseline"),
    ]
    _SCALE_OPTIONS = [('pair', "Same scale (pair)"), ('session', "Same scale (session)"),
                      ('visible', "Same scale (visible)")]

    def __init__(self, nav: 'AppState', parent=None, expanded: bool = True):
        super().__init__("Normalization", expanded=expanded, parent=parent)
        self.nav = nav
        self._norm_btns: dict[NormalizeBy, QPushButton] = {}
        self._scale_btns: dict[str, QPushButton] = {}
        self._build()
        nav.norms_changed.connect(self._on_nav_norms_changed)
        nav.scale_mode_changed.connect(self._on_nav_scale_changed)
        self.norms_changed.connect(nav.set_active_norms)
        self.scale_changed.connect(nav.set_same_scale_mode)

    def _build(self):
        chips = QWidget()
        chip_layout = FlowLayout(chips)
        scales = QWidget()
        scale_layout = FlowLayout(scales)
        for w in (chips, scales):
            w.setStyleSheet('QWidget { border: none; }')
        for norm, label in self._NORM_OPTIONS:
            button = chip_button(label, checkable=True)
            button.toggled.connect(lambda _checked: self._emit_norms())
            self._norm_btns[norm] = button
            # the baseline chip leads the scale row: both change what the y axis means
            (scale_layout if norm is NormalizeBy.BASELINE else chip_layout).addWidget(button)
        for mode, label in self._SCALE_OPTIONS:
            button = chip_button(label, checkable=True)
            button.toggled.connect(
                lambda checked, m=mode: self.scale_changed.emit(m if checked else None))
            self._scale_btns[mode] = button
            scale_layout.addWidget(button)
        self.body_layout.addWidget(chips)
        self.body_layout.addWidget(scales)

        apply_btn = chip_button("Apply to data…", checkable=False)
        apply_btn.clicked.connect(self.apply_requested)
        apply_row = QHBoxLayout()
        apply_row.addStretch()
        apply_row.addWidget(apply_btn)
        self.body_layout.addLayout(apply_row)

    def _emit_norms(self):
        self.norms_changed.emit({norm for norm, btn in self._norm_btns.items()
                                 if btn.isChecked()})

    def _on_nav_norms_changed(self, active: set):
        for norm, button in self._norm_btns.items():
            set_checked_quietly(button, norm in active)

    def _on_nav_scale_changed(self, mode):
        for scale_mode, button in self._scale_btns.items():
            set_checked_quietly(button, mode == scale_mode)


class TailRow(QWidget):
    """One tail interval in ms; blank or ±inf means the window edge."""

    changed = Signal()
    add_requested = Signal()
    delete_requested = Signal(object)   # self

    def __init__(self, start_ms, end_ms, deletable: bool, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.start = QLineEdit('' if start_ms is None else f"{start_ms:g}")
        self.end = QLineEdit('' if end_ms is None else f"{end_ms:g}")
        for box in (self.start, self.end):
            box.setFixedWidth(44)
            box.setPlaceholderText("inf")
            box.editingFinished.connect(self.changed)
        for w in (QLabel("Tail:"), self.start, QLabel("–"), self.end, QLabel("ms")):
            row.addWidget(w)
        self.add_btn = chip_button("+")
        self.add_btn.clicked.connect(self.add_requested)
        row.addWidget(self.add_btn)
        if deletable:
            del_btn = chip_button("−")
            del_btn.clicked.connect(lambda: self.delete_requested.emit(self))
            row.addWidget(del_btn)
        row.addStretch()

    @property
    def interval(self) -> tuple:
        """(start, end) in seconds, None where the bound is open."""
        def parse(box):
            try:
                v = float(box.text())
            except ValueError:
                return None
            return None if np.isinf(v) else v / 1000.0
        return parse(self.start), parse(self.end)


class ExtendRow(QWidget):
    """One extend view: enable toggle, window + bin size, and add/delete."""

    changed = Signal()
    add_requested = Signal()
    delete_requested = Signal(object)   # self

    def __init__(self, make_spin, deletable: bool, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.extend_check = QCheckBox("Extend:")
        self.extend_check.clicked.connect(self.changed)
        self._ms_spin = make_spin((5, 10, 20, 50, 100, 200, 500, 1000), "50")
        bin_opts = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        self._bin_spin = make_spin(bin_opts, "1.0")
        for spin in (self._ms_spin, self._bin_spin):
            spin.currentTextChanged.connect(lambda _: self.changed.emit())
        for w in (self.extend_check, self._ms_spin,
                  QLabel("ms  resolution:"), self._bin_spin, QLabel("ms")):
            row.addWidget(w)
        self.add_btn = chip_button("+")
        self.add_btn.clicked.connect(self.add_requested)
        row.addWidget(self.add_btn)
        if deletable:
            del_btn = chip_button("−")
            del_btn.clicked.connect(lambda: self.delete_requested.emit(self))
            row.addWidget(del_btn)
        row.addStretch()

    @property
    def enabled(self) -> bool:
        return self.extend_check.isChecked()

    @property
    def extend_ms(self) -> int:
        try: return int(self._ms_spin.currentText())
        except ValueError: return 50

    @property
    def extend_bin_ms(self) -> float:
        try: return float(self._bin_spin.currentText().split()[0])
        except (ValueError, IndexError): return 1.0

    def set_sample_bin(self, sampling_rate: float):
        """Offer the sample period as the finest bin, labelled as such."""
        ms = 1000.0 / sampling_rate
        if self._bin_spin.findData(ms) < 0:
            self._bin_spin.insertItem(0, f"{ms:.4g} (sample)", ms)


class CorrelogramSection(CollapsibleSection):

    style_changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Correlogram", parent=parent)
        self._build()

    def _build(self):
        self.body_layout.addLayout(self._trace_row())
        self.body_layout.addLayout(self._scale_row())
        self.body_layout.addLayout(self._extend_rows_box())
        self.jitter_line_btn = CycleButton("jitter", start_hidden=True)
        self.body_layout.addLayout(widget_row(self.jitter_line_btn))
        for btn in (self.ccg_btn, self.baseline_btn, self.ref_btn, self.tgt_btn,
                    self.ref_wf_btn, self.autoscale_btn, self.deconv_ref_btn,
                    self.deconv_tgt_btn, self.jitter_line_btn):
            btn.clicked.connect(self.style_changed)

    def _scale_entry(self) -> 'SliderWithInput':
        """A 0.01-1.5x y-scale slider that redraws on change."""
        widget = SliderWithInput(1, 150, 100, scale=0.01)
        widget.value_changed.connect(lambda _: self.style_changed.emit())
        return widget

    def _trace_row(self) -> QHBoxLayout:
        self.ccg_btn = CycleButton("CCG")
        self.baseline_btn = CycleButton("baseline")
        self.ref_btn = CycleButton("ref", start_hidden=True)
        self.tgt_btn = CycleButton("tgt", start_hidden=True)
        self.ref_wf_btn = chip_button("ref waveform", checkable=True)
        return widget_row(self.ccg_btn, self.baseline_btn, None,
                          "Show ACG", self.ref_btn, self.tgt_btn, None,
                          self.ref_wf_btn)

    def _scale_row(self) -> QHBoxLayout:
        self.autoscale_btn = chip_button("Autoscale", checkable=True)
        self._ref_scale_widget = self._scale_entry()
        self._tgt_scale_widget = self._scale_entry()
        self.deconv_ref_btn = chip_button("ref", checkable=True)
        self.deconv_tgt_btn = chip_button("tgt", checkable=True)
        self.wf_pad_slider = SliderWithInput(0, 400, 5, scale=0.01)
        self.wf_pad_slider.value_changed.connect(lambda _: self.style_changed.emit())
        return widget_row(self.autoscale_btn,
                          "ref:", self._ref_scale_widget,
                          "tgt:", self._tgt_scale_widget, None,
                          "Deconvolve", self.deconv_ref_btn, self.deconv_tgt_btn,
                          "wf pad:", self.wf_pad_slider)

    def _extend_rows_box(self) -> QVBoxLayout:
        self._extend_rows: list = []
        self._sampling_rate = None
        self._extend_box = QVBoxLayout()
        self._extend_box.setContentsMargins(0, 0, 0, 0)
        self._add_extend_row()
        return self._extend_box

    def set_sampling_rate(self, rate: float):
        """Session clock: every extend row offers its sample period as the finest bin."""
        self._sampling_rate = rate
        for row in self._extend_rows:
            row.set_sample_bin(rate)

    def _add_extend_row(self) -> 'ExtendRow':
        """First row is permanent, so only later ones offer delete."""
        row = ExtendRow(self.make_spin, deletable=bool(self._extend_rows))
        if self._sampling_rate:
            row.set_sample_bin(self._sampling_rate)
        row.changed.connect(self.style_changed)
        row.add_requested.connect(self._on_add_extend_btn)
        row.delete_requested.connect(self._on_delete_extend_btn)
        self._extend_rows.append(row)
        self._extend_box.addWidget(row)
        return row

    def _on_add_extend_btn(self):
        self._add_extend_row().extend_check.setChecked(True)
        self.style_changed.emit()

    def _on_delete_extend_btn(self, row: 'ExtendRow'):
        self._extend_rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self.style_changed.emit()

    @property
    def extend_views(self) -> list:
        """Enabled extend rows, in display order."""
        return [r for r in self._extend_rows if r.enabled]

    def extend_state(self) -> list:
        """Every extend row (enabled or not), for persisting across sessions."""
        return [{'enabled': r.enabled, 'ms': r.extend_ms, 'bin_ms': r.extend_bin_ms}
                for r in self._extend_rows]

    def restore_extend_state(self, state: list):
        if not state:
            return
        while len(self._extend_rows) > len(state):
            self._on_delete_extend_btn(self._extend_rows[-1])
        while len(self._extend_rows) < len(state):
            self._add_extend_row()
        for row, saved in zip(self._extend_rows, state):
            row._ms_spin.setCurrentText(str(saved['ms']))
            row._bin_spin.setCurrentText(str(saved['bin_ms']))
            row.extend_check.setChecked(bool(saved['enabled']))

    @property
    def acg_yscale_ref(self) -> float:
        return max(0.01, self._ref_scale_widget.value)

    @property
    def acg_yscale_tgt(self) -> float:
        return max(0.01, self._tgt_scale_widget.value)



_BASELINE_EXPLANATIONS = {
    'conv':   "Conv: Convolution smoothed null baseline",
    'tailed': "Tailed: ACG deconvolution, tail-bin baseline",
    'global': "Global: max bin outside of test window as baseline",
    'jitter': "Jitter: Bootstrapped baseline from surrogate data using interval jitter",
}


class BaselineCSSection(CollapsibleSection):
    metric_changed    = Signal(str)    # 'STG' | 'JBSI'
    baseline_changed  = Signal(str)    # 'conv' | 'tailed' | 'global' | 'jitter'
    sig_changed       = Signal()

    def __init__(self, nav: 'AppState', parent=None, expanded: bool = True):
        super().__init__("Baseline & Connection Strength", expanded=expanded, parent=parent)
        self.nav        = nav
        self.jitter_mgr = None  # set via set_jitter_mgr()
        self._build()
        nav.cs_overlay_changed.connect(self._on_cs_overlay_changed)
        nav.cs_params_changed.connect(self._on_cs_params_changed)
        nav.display_changed.connect(self._on_display_changed)
        self.baseline_changed.connect(lambda m: nav.set_cs_params(m, nav.cs_metric))
        self.metric_changed.connect(lambda m: nav.set_cs_params(nav.baseline_method, m))
        self._on_display_changed()

    def _on_display_changed(self, _=None):
        """Mirror nav: chips and the lag windows conf owns."""
        for btn, flag in ((self.nonneg_btn, 'cs_nonneg'), (self.p_btn, 'show_p'),
                          (self.pc_btn, 'show_pc'),
                          (self.test_window_btn, 'show_test_window'),
                          (self.tail_window_btn, 'show_tail_window')):
            set_checked_quietly(btn, getattr(self.nav, flag))
        conf = self.nav.cd.conf
        for box, sec in ((self.win_start, conf.min_lag), (self.win_end, conf.max_lag)):
            if not box.input.hasFocus():       # never fight the field being typed in
                box.input.setText(f"{sec * 1000:g}")

    def refresh_font(self):
        self._explanation.setStyleSheet(f"color: #666; font-size: {small_font_pt()}pt;")

    def set_jitter_mgr(self, jctrl):
        self.jitter_mgr = jctrl

    def _build(self):
        layout = self.body_layout

        self.cs_show_check = QCheckBox("Show CS overlay")
        self.cs_show_check.toggled.connect(self.nav.set_cs_overlay)
        self._metric_group, metric_rbs = radio_group(
            [('STG', 'STG'), ('JBSI', 'JBSI')], selected='STG', parent=self,
            on_click=lambda btn: self.metric_changed.emit(btn.text()))
        self.win_start = MetricInput("Window:", ("ms",), default="1", input_width=44)
        self.win_end = MetricInput("–", ("ms",), default="3", input_width=44)
        for box in (self.win_start, self.win_end):
            box.input.editingFinished.connect(self._on_window_input)
        layout.addLayout(widget_row(self.cs_show_check, "Measure:",
                                    *metric_rbs.values(), None,
                                    self.win_start, self.win_end))

        self._cs_label = QLabel("CS: —|—")
        self.nonneg_btn = chip_button("non-negative", checkable=True)
        self.nonneg_btn.toggled.connect(
            lambda on: self.nav.set_display_flag('cs_nonneg', on))
        layout.addLayout(widget_row(self._cs_label, self.nonneg_btn))

        self._baseline_group, self._baseline_rbs = radio_group(
            [('conv', 'Conv'), ('tailed', 'Tailed'),
             ('global', 'Global'), ('jitter', 'Jitter')],
            selected='conv', parent=self, on_click=self._on_baseline_clicked)
        self.test_window_btn = chip_button("Test window", checkable=True, checked=True)
        self.test_window_btn.toggled.connect(
            lambda on: self.nav.set_display_flag('show_test_window', on))
        layout.addLayout(widget_row("Baseline:", *self._baseline_rbs.values(),
                                    None, self.test_window_btn))

        self.p_btn = chip_button("p", checkable=True, checked=True)
        self.pc_btn = chip_button("p-corrected", checkable=True, checked=True)
        for btn, flag in ((self.p_btn, 'show_p'), (self.pc_btn, 'show_pc')):
            btn.toggled.connect(
                lambda on, f=flag: self.nav.set_display_flag(f, on))
        layout.addLayout(widget_row(self.p_btn, self.pc_btn))

        self._explanation = QLabel(_BASELINE_EXPLANATIONS['conv'])
        self._explanation.setStyleSheet(f"color: #666; font-size: {small_font_pt()}pt;")
        self._explanation.setWordWrap(True)
        layout.addWidget(self._explanation)

        self.tail_window_btn = chip_button("Tail window", checkable=True)
        self.tail_window_btn.toggled.connect(
            lambda on: self.nav.set_display_flag('show_tail_window', on))
        self.tail_source_combo = QComboBox()
        self.tail_source_combo.addItems(['bins', 'conv', 'jitter'])
        self.tail_source_combo.setMinimumWidth(80)
        self._tail_head = widget_row("Average of:", self.tail_source_combo,
                                     None, self.tail_window_btn)
        layout.addLayout(self._tail_head)
        self._tail_rows: list = []
        self._tail_box = QVBoxLayout()
        self._tail_box.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._tail_box)
        for start, end in self.nav.cd.conf.tail_intervals:
            self._add_tail_row(start, end)
        self.tail_source_combo.currentTextChanged.connect(lambda _: self._on_tail_input())

        self._update_pval_row_visibility('conv')

    def _add_tail_row(self, start=None, end=None) -> 'TailRow':
        row = TailRow(None if start is None else start * 1000.0,
                      None if end is None else end * 1000.0,
                      deletable=bool(self._tail_rows))
        row.changed.connect(self._on_tail_input)
        row.add_requested.connect(lambda: (self._add_tail_row(), self._on_tail_input()))
        row.delete_requested.connect(self._on_delete_tail_btn)
        self._tail_rows.append(row)
        self._tail_box.addWidget(row)
        row.setVisible(self.nav.baseline_method == 'tailed')
        return row

    def _on_delete_tail_btn(self, row: 'TailRow'):
        self._tail_rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self._on_tail_input()

    def _on_tail_input(self):
        """Store the typed tail intervals on the config, then redraw everything reading them."""
        self.nav.cd.set_tail_window([r.interval for r in self._tail_rows],
                                    self.tail_source_combo.currentText())
        self.nav.display_changed.emit(None)

    def _on_cs_overlay_changed(self, active: bool):
        set_checked_quietly(self.cs_show_check, active)

    def _on_cs_params_changed(self, baseline_method: str, cs_metric: str):
        for method, button in self._baseline_rbs.items():
            set_checked_quietly(button, method == baseline_method)
        for button in self._metric_group.buttons():
            set_checked_quietly(button, button.text() == cs_metric)
        self._explanation.setText(_BASELINE_EXPLANATIONS.get(baseline_method, ""))
        self._update_pval_row_visibility(baseline_method)

    def _on_baseline_clicked(self, btn: 'QRadioButton'):
        method = btn.text().lower()
        self._explanation.setText(_BASELINE_EXPLANATIONS.get(method, ""))
        self._update_pval_row_visibility(method)
        self.baseline_changed.emit(method)

    def _update_pval_row_visibility(self, method: str):
        show = method in ('conv', 'jitter')
        self.p_btn.setVisible(show)
        self.pc_btn.setVisible(show)
        for w in (self.tail_window_btn, self.tail_source_combo, *self._tail_rows):
            w.setVisible(method == 'tailed')

    def set_jitter_baseline_enabled(self, enabled: bool):
        rb = self._baseline_rbs.get('jitter')
        if rb is not None:
            rb.setEnabled(enabled)
            if not enabled and rb.isChecked():
                self._baseline_rbs['conv'].setChecked(True)
                self._on_baseline_clicked(self._baseline_rbs['conv'])

    # baseline_method and cs_metric live in nav.baseline_method / nav.cs_metric

    CHIPS = ('p_btn', 'pc_btn', 'test_window_btn')   # persisted significance chips

    def chip_state(self) -> dict:
        """Which significance chips are on, for persisting across sessions."""
        return {name: getattr(self, name).isChecked() for name in self.CHIPS}

    def restore_chip_state(self, state: dict):
        for name, on in state.items():
            getattr(self, name).setChecked(bool(on))

    def set_cs(self, *values, names=('lo', 'hi'), segment=None):
        """CS label as name|name|… = value|value|…, one column per view."""
        self.set_cs_rows([(segment, values)], names=names)

    def set_cs_rows(self, rows, names=('lo', 'hi')):
        """One line per (segment, values) row; the segment name is dropped when alone."""
        def _fmt(v):
            return "—" if v is None else f"{float(v):.3f}"
        lines = []
        for segment, values in rows:
            labels = list(names) + [f"ext{i + 1}" for i in range(len(values) - len(names))]
            head = f"CS {segment}" if segment and len(rows) > 1 else "CS"
            lines.append(f"{head} ({'|'.join(labels[:len(values)])} = "
                         f"{'|'.join(_fmt(v) for v in values)})")
        self._cs_label.setText('\n'.join(lines))

    def sync_window(self):
        """Show the dataset's test window in the input boxes."""
        start, end = self.nav.cd.significance_window
        self.win_start.set_value(round(start * 1000, 4))
        self.win_end.set_value(round(end * 1000, 4))

    def _on_window_input(self):
        """Move the test window to the typed lags, then redraw everything reading it."""
        start_ms, _ = self.win_start.value()
        end_ms, _ = self.win_end.value()
        try:
            self.nav.cd.set_significance_window(start_ms / 1000.0, end_ms / 1000.0)
        except ValueError as exc:
            QMessageBox.warning(self, "Test window", str(exc))
            self.sync_window()
            return
        self.nav.display_changed.emit(None)

    def update_display(self):
        """Recompute and display the CS value of every view of the current pair."""
        nav = self.nav
        inds = nav.current_pair_inds
        if inds is None:
            self.set_cs(None, None)
            return
        ref, tgt = int(inds[0]), int(inds[1])
        extend = [None if ctx is None else ctx.cs_value
                  for ctx in self._extend_contexts()]

        def _row(seg_label):
            seg_idx = nav.segment_index(seg_label)
            values = [CCGContextBuilder._cs_value(
                          nav, self.jitter_mgr, seg_idx, ref, tgt, resolution,
                          nonneg=nav.cs_nonneg)
                      for resolution in ('lowres', 'highres')]
            return seg_label, values + extend   # extend views report CS at their own window/bin

        segments = list(nav.stacked_segments) or [nav.current_segment]
        self.set_cs_rows([_row(s) for s in segments])

    def _extend_contexts(self) -> list:
        """The context each enabled extend row last rendered."""
        panel = self.nav.root.mainview
        return [CCGContextBuilder.build_extend_context(self.nav, panel, ext_view=r)
                for r in self._extend_views()]

    def _extend_views(self) -> list:
        """Extend rows currently drawn, so each gets its own CS column."""
        return self.nav.root.mainview.corr_section.extend_views


class JitterSection(CollapsibleSection):
    # Backend: JitterManager (neuropy/ui/jitter_ui.py)
    # Injected via set_jitter_mgr() after construction.

    jitter_done = Signal()   # emitted when poll completes → CorrelogramPanel rerenders

    def __init__(self, nav: 'AppState', parent=None):
        super().__init__("Jitter", parent=parent)
        self.nav   = nav
        self._jctrl = None
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(200)
        self._poll_timer.timeout.connect(self._poll)
        self._build()

    def set_jitter_mgr(self, jctrl):
        self._jctrl = jctrl

    _OUTLINE_STYLE = ("QPushButton { border: 1px solid palette(mid); border-radius: 3px; "
                      "padding: 2px 8px; } "
                      "QPushButton:hover { border-color: palette(highlight); }")

    def _build(self):
        self._n_spin = self.make_spin((10, 20, 50, 100, 200, 500, 1000), "100")
        self._run_btn = self._outline_button("Run Jitter", self._run)
        self.lo_btn = chip_button("lo", checkable=True, checked=True)
        self.hi_btn = chip_button("hi", checkable=True, checked=False)
        self.body_layout.addLayout(widget_row(
            "n=", self._n_spin, self._run_btn,
            self._outline_button("Clear", self._clear),
            self._outline_button("Save", self._save),
            "Resolution:", self.lo_btn, self.hi_btn, spacing=4))

    def _outline_button(self, label: str, slot) -> QPushButton:
        button = QPushButton(label)
        button.setStyleSheet(self._OUTLINE_STYLE)
        button.clicked.connect(slot)
        return button

    @property
    def n_jitter(self) -> int:
        try: return max(1, int(self._n_spin.currentText()))
        except ValueError: return 100

    def set_running(self, running: bool):
        self._run_btn.setText("Running…" if running else "Run Jitter")
        self._run_btn.setEnabled(not running)

    def _run(self):
        jctrl = self._jctrl
        inds  = self.nav.current_pair_inds
        data  = self.nav.ccg_data
        if jctrl is None or inds is None or data is None:
            return
        ref, tgt  = int(inds[0]), int(inds[1])
        nav       = self.nav
        run_hi = self.hi_btn.isChecked()
        run_lo = self.lo_btn.isChecked() or not run_hi
        jctrl.run_jitter(ref, tgt, self.n_jitter, run_lo=run_lo, run_hi=run_hi)
        self.set_running(True)
        self._poll_timer.start()

    def _poll(self):
        jctrl = self._jctrl
        if jctrl is None:
            self._poll_timer.stop()
            return
        if not jctrl.jitter_worker.is_running():
            self._poll_timer.stop()
            self.set_running(False)
            self.jitter_done.emit()

    def _clear(self):
        jctrl = self._jctrl
        inds  = self.nav.current_pair_inds
        if jctrl is None or inds is None:
            return
        ref, tgt = int(inds[0]), int(inds[1])
        jctrl.clear(ref, tgt)
        self.jitter_done.emit()

    def _save(self):
        jctrl = self._jctrl
        if jctrl is None:
            return
        jctrl.on_save()


class SpikeAttributionSection(CollapsibleSection):

    set_requested    = Signal(float, str)   # (bin_value, unit: 'ms' | '#')
    enable_toggled   = Signal(bool)

    def __init__(self, parent=None):
        super().__init__("Spike Attribution", parent=parent)
        self._build()

    def _build(self):
        self.enable_btn = chip_button("Enable", checkable=True)
        self.enable_btn.toggled.connect(self._on_enable)

        self._bin_metric = MetricInput("Bin:", ("ms", "#"), default="0",
                                       suggestions=(0, 1, 2, 5, 10), unit_width=45)
        self._bin_metric.setEnabled(False)
        self._bin_metric.unit_combo.setToolTip(
            "ms: lag in milliseconds\n"
            "#: ±i-th bin relative to 0 ms bin")
        self._bin_metric.input.returnPressed.connect(self._on_set)

        self._set_btn = QPushButton("Set")
        self._set_btn.setEnabled(False)
        self._set_btn.clicked.connect(self._on_set)

        self.body_layout.addLayout(widget_row(
            self.enable_btn, self._bin_metric, self._set_btn, spacing=4))

    def _on_enable(self, checked: bool):
        self._bin_metric.setEnabled(checked)
        self._set_btn.setEnabled(checked)
        self.enable_toggled.emit(checked)

    def _on_set(self):
        if not self.enable_btn.isChecked():
            return
        try:
            val = float(self._bin_metric.input.text())
            self.set_requested.emit(val, self._bin_metric.unit_combo.currentText())
        except ValueError:
            p = self.window()
            lp = p.pairs_view.spike_pairs if p else None
            if lp is not None:
                lp._spike_pairs_count.set('Invalid bin')

    @property
    def is_enabled(self) -> bool:
        return self.enable_btn.isChecked()


@dataclass
class Subplot:
    """One plot and every overlay drawn on it, kept together so they can't misalign."""
    widget: 'pg.PlotWidget'
    plot: 'pg.PlotItem'
    pval_vb: 'pg.ViewBox'
    pval_axis: 'pg.AxisItem'
    wf_vb: 'pg.ViewBox'
    acg_axes: list          # [(view_box, axis) ref, (view_box, axis) tgt]
    readout: 'pg.TextItem'
    title: 'pg.LabelItem'
    pval_items: list = field(default_factory=list)
    ctx: 'RenderContext | None' = None


@dataclass(frozen=True)
class LagAxis:
    """The x geometry of one plot, in milliseconds: bin centers and bar edges."""
    centers: np.ndarray
    edges: np.ndarray
    bin_ms: float
    window_ms: float

    @classmethod
    def of(cls, ctx: 'RenderContext', n_bins: int) -> 'LagAxis':
        bin_ms = ctx.bin_size_eff * 1000.0
        window_ms = ctx.window_size_eff * 1000.0
        half = window_ms / 2
        return cls(
            centers=np.linspace(-half, half, n_bins),
            edges=np.linspace(-half - bin_ms / 2, half + bin_ms / 2, n_bins + 1),
            bin_ms=bin_ms, window_ms=window_ms,
        )

    @staticmethod
    def has_test_window(ctx: 'RenderContext') -> bool:
        return ctx.min_lag_plot is not None and ctx.max_lag_plot is not None


@dataclass(frozen=True)
class TraceToggle:
    """Whether one trace is drawn, and whether as a line rather than bars."""
    show: bool
    line: bool

    @classmethod
    def read(cls, btn: 'CycleButton') -> 'TraceToggle':
        return cls(show=btn.show, line=btn.line)


@dataclass(frozen=True)
class AcgToggle(TraceToggle):
    """A trace toggle plus the ACG-only y-scale and deconvolution switches."""
    yscale: float
    deconv: bool

    @classmethod
    def read_acg(cls, btn: 'CycleButton', yscale: float,
                 deconv_btn) -> 'AcgToggle':
        return cls(show=btn.show, line=btn.line, yscale=yscale,
                   deconv=deconv_btn.isChecked())


@dataclass(frozen=True)
class DisplayToggles:
    """Every display switch a RenderContext depends on, read once from the widgets.

    Frozen so it doubles as the extend cache key: what is drawn and what is
    cached can no longer disagree.
    """
    ccg: TraceToggle
    baseline: TraceToggle
    acg_ref: AcgToggle
    acg_tgt: AcgToggle
    line_jitter: bool
    acg_match_ccg: bool
    show_pval: bool
    show_pval_corrected: bool
    show_test_window: bool
    show_tail_window: bool
    show_ref_waveform: bool
    wf_y_pad: float

    @classmethod
    def read(cls, cor: 'CorrelogramSection', cs: 'BaselineCSSection') -> 'DisplayToggles':
        """Snapshot the correlogram and baseline sections' current switches."""
        return cls(
            ccg=TraceToggle.read(cor.ccg_btn),
            baseline=TraceToggle.read(cor.baseline_btn),
            acg_ref=AcgToggle.read_acg(cor.ref_btn, cor.acg_yscale_ref,
                                       cor.deconv_ref_btn),
            acg_tgt=AcgToggle.read_acg(cor.tgt_btn, cor.acg_yscale_tgt,
                                       cor.deconv_tgt_btn),
            line_jitter=cor.jitter_line_btn.line,
            acg_match_ccg=cor.autoscale_btn.isChecked(),
            show_ref_waveform=cor.ref_wf_btn.isChecked(),
            wf_y_pad=cor.wf_pad_slider.value,
            show_pval=cs.nav.show_p,
            show_pval_corrected=cs.nav.show_pc,
            show_test_window=cs.nav.show_test_window,
            show_tail_window=cs.nav.show_tail_window,
        )


@dataclass
class CCGSource:
    """The traces one plot is drawn from, before display toggles and normalization.

    The two build paths differ only in how they fill this: the stored path slices
    a precomputed array, the extend path recomputes at a new window and bin size.
    """
    ccg: np.ndarray
    baseline: np.ndarray | None
    pval: np.ndarray | None
    pval_corrected: np.ndarray | None
    acg_ref: np.ndarray | None
    acg_tgt: np.ndarray | None
    conf: object
    bin_size: float
    duration: float
    neurons: object
    time_hours: float | None
    refit_after_deconv: bool


class CCGContextBuilder:
    """Assemble a RenderContext from AppState + CorrelogramPanel sections.

    build_context()        — stored CCG for the current pair and segment
    build_extend_context() — CCG recomputed at a user-specified window and bin

    Both fill a CCGSource, then hand it to _finish(), which applies the display
    toggles, deconvolution, and normalization that they share.
    """

    def __init__(self, nav, panel):
        self.nav = nav
        self.panel = panel
        self.cor = panel.corr_section
        self.cs = panel.cs_section
        self.toggles = DisplayToggles.read(self.cor, self.cs)

    @property
    def dark_mode(self) -> bool:
        """True when the plot should be drawn on a dark background."""
        theme = self.panel._theme_fn() if self.panel._theme_fn is not None else None
        return qt_dark_mode() if theme is None else theme.dark

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_pair(nav):
        """Return (ref, tgt, pair_key); in all-session mode the pair names its own session."""
        ref, tgt = int(nav.current_pair_inds[0]), int(nav.current_pair_inds[1])
        pair_key = nav.key
        if nav.session_any_mode:
            handles = nav.cross_session_handles or []
            idx = nav.current_pair_idx
            if idx < len(handles):
                pair_key = handles[idx][0]
                ref, tgt = int(handles[idx][1]), int(handles[idx][2])
        return ref, tgt, pair_key

    @staticmethod
    def _resolve_data(nav, pair_key, hi_res_override):
        """Return (nd-key, CCGData) to render from; the key resolves segment labels."""
        nd = pair_key.nd()
        hi = nd.change(resolution='highres')
        lo = nd.change(resolution='lowres')
        if hi_res_override is True:
            return (hi, nav.cd.ccg_for(hi)) if nav.cd.ccg_for(hi) else (lo, nav.cd.ccg_for(lo))
        if hi_res_override is False:
            return lo, nav.cd.ccg_for(lo)
        if nav.resolution in ("hi", "lo_hi"):
            return hi, nav.cd.ccg_for(hi)
        return lo, nav.cd.ccg_for(lo)

    @staticmethod
    def _firing_rates(nav, ref, tgt):
        """(fr_ref, fr_tgt); an appended window's own rates when it declares them."""
        key = nav.get_complete_key()
        src = nav.cd.source_config(key, key.segment) if key.segment else None
        seg_fr = src.firing_rates if src is not None else None
        if seg_fr is None:
            seg_fr = nav.neurons.firing_rate
        return float(seg_fr[ref]), float(seg_fr[tgt])

    @staticmethod
    def _time_hours_for_seg(nav, seg_idx) -> float | None:
        """Recording hours for dim0 *seg_idx* — TIME norm divisor (same resolver as batch)."""
        key = nav.get_complete_key()
        return nav.cd.time_hours_for(key, nav.cd.segment_name(key, seg_idx))

    @staticmethod
    def _cs_value(nav, jitter_mgr, seg_idx, ref, tgt, resolution, *, nonneg):
        """Connection strength for one pair/segment, or None if that resolution isn't loaded."""
        key = nav.get_complete_key().change(resolution=resolution, ref=ref, tgt=tgt,
                                            segment=nav.segment_name(seg_idx))
        slices = nav.cd.pair_slices(key)
        if slices is None:
            return None
        ccg_raw, null_raw, _pval, _pvc, _qval = slices
        metric = nav.cs_metric
        method = nav.baseline_method
        cached = jitter_mgr.get_result(ref, tgt, 'lo' if resolution == 'lowres' else 'hi')
        j_avg = cached[0] if (metric == 'JBSI' and cached is not None) else None
        if method == 'jitter' and cached is not None:
            null_raw = cached[0]
        fr_ref = fr_tgt = None
        if metric == 'JBSI':
            fr_ref, fr_tgt = CCGContextBuilder._firing_rates(nav, ref, tgt)
        return ConnectionStrength.conn_strength(
            ccg_raw, null_raw, ref, tgt, nav.cd.ccg_for(key).conf,
            metric=metric, method=method, active_norms=nav.active_norms,
            neurons=nav.neurons,
            custom_time_hours=CCGContextBuilder._time_hours_for_seg(nav, seg_idx),
            fr_ref=fr_ref, fr_tgt=fr_tgt, j_avg=j_avg, nonneg=nonneg,
            excitability=nav.key.excitability)

    @staticmethod
    def _neuron_meta(neurons, ref: int, tgt: int):
        """Return (type_ref, type_tgt, shank_ref, shank_tgt) for the pair."""
        if neurons is None:
            return None, None, None, None
        types = neurons.neuron_type
        shanks = neurons.shank_ids   # optional: a dataset may bind no shank column
        return (str(types[ref]), str(types[tgt]),
                None if shanks is None else int(shanks[ref]),
                None if shanks is None else int(shanks[tgt]))

    @staticmethod
    def _jitter_overlay(panel, nav, ref: int, tgt: int) -> JitterOverlay:
        """Cached jitter result for the current resolution; empty when there is none."""
        jitter_mgr = panel.jitter_mgr
        if jitter_mgr is None:
            return JitterOverlay()
        res_key = 'hi' if nav.resolution in ("hi", "lo_hi") else 'lo'
        result = jitter_mgr.get_result(ref, tgt, res_key)
        if result is None:
            return JitterOverlay()
        avg, pval, _bins, lo, hi = result
        return JitterOverlay(j_ccg=avg, j_pval=pval, j_ccg_lo=lo, j_ccg_hi=hi)

    @staticmethod
    def _bin_size(conf, n_bins: int) -> float:
        """Seconds per bin, recovered from the window so a mutated conf can't mislead."""
        duration = conf.duration or 1.0
        return duration / (n_bins - 1) if n_bins > 1 else conf.bin_size

    @staticmethod
    def _refit_conv(ccg, conf, bs: float, excitability: str):
        """(baseline, pval, pval_corrected) fitted to `ccg` by hollow convolution.

        W is in bins; _conv widens a gaussian W to ~6*sigma and its reflect-padding
        must fit the array, hence the cap."""
        W = max(1, int(min(conf.conv_window / bs, (len(ccg) - 1) / 3)))
        pvals, pred, qvals = EranConv._conv(ccg, W=W, wintype="gauss")
        p_raw = (qvals if excitability == 'I' else pvals)[0]
        _, p_corr = _multiple_correction(p_raw, conf.alpha,
                                         method=conf.multiple_correction)
        return pred[0], p_raw, p_corr

    @classmethod
    def _same_scale_ylim(cls, nav, panel, data, ref: int, tgt: int, neurons):
        """(0, ymax) shared across a pair's segments, or across every pair in the session."""
        mode = nav.same_scale_mode
        if mode is None or data.ccg is None:
            return None
        if mode == 'visible':
            return 'visible'   # only render() sees every plot on screen; it resolves this
        # the rendered array's own bin count, not nav.resolution: 'lo_hi' draws both
        cache_key = (mode, frozenset(nav.active_norms), data.ccg.shape[-1], str(nav.key),
                     (ref, tgt) if mode == 'pair' else None)
        hit = panel._same_scale_cache.get(cache_key)
        if hit is not None:
            return hit

        if mode == 'pair':
            pairs = [(ref, tgt)]
        else:
            pairs = [(int(r), int(t)) for r, t in nav.all_pairs_np]

        top = 0.0
        for seg in range(data.ccg.shape[0]):
            time_hours = cls._time_hours_for_seg(nav, seg)
            for r, t in pairs:
                normed, _ = CCGNorm.apply(data.ccg[seg, r, t, :], None, r, t,
                                          nav.active_norms, neurons=neurons,
                                          custom_time_hours=time_hours)
                top = max(top, float(np.nanmax(normed)))

        ylim = (0.0, top * 1.1) if top > 0 else (0.0, 1.0)
        panel._same_scale_cache.put(cache_key, ylim)
        return ylim

    # ── public entry points ────────────────────────────────────────────

    def for_current(self, seg_label=None, hi_res_override=None,
                    pair_override=None) -> 'RenderContext | None':
        """RenderContext for the stored CCG of the current pair and segment.

        seg_label:       segment to render; None = nav.current_segment
        hi_res_override: True=hi, False=lo, None=follow nav.resolution
        pair_override:   (pair_key, ref, tgt) to render a pinned "Show Together"
                         pair instead of the current one
        """
        nav = self.nav
        if pair_override is not None:
            pair_key = pair_override[0]
            ref, tgt = int(pair_override[1]), int(pair_override[2])
        elif nav.current_pair_inds is None:
            return None
        else:
            ref, tgt, pair_key = self._resolve_pair(nav)
        data_key, data = self._resolve_data(nav, pair_key, hi_res_override)
        if data is None or data.ccg is None:
            return None

        seg_label = seg_label or nav.current_segment
        seg_idx = self.nav.cd.segment_index(data_key, seg_label)
        arr = data.ccg          # segment_index may have grown dim0 by lazy-loading
        ccg, baseline, pval, pval_corrected, _qval = data.pair(seg_idx, ref, tgt)
        if ccg is None or len(ccg) == 0:
            return None

        conf = data.conf
        neurons = nav.cd.nd.neurons_for(pair_key)   # not nav.neurons: differs in all-session mode
        source = CCGSource(
            ccg=ccg, baseline=baseline, pval=pval, pval_corrected=pval_corrected,
            acg_ref=arr[seg_idx, ref, ref, :], acg_tgt=arr[seg_idx, tgt, tgt, :],
            conf=conf, bin_size=self._bin_size(conf, len(ccg)),
            duration=conf.duration or 1.0, neurons=neurons,
            time_hours=self._time_hours_for_seg(nav, seg_idx),
            refit_after_deconv=True,
        )
        return self._finish(
            source, ref, tgt,
            seg_display=seg_label, sess_label=str(pair_key.session or ''),
            jitter=self._jitter_overlay(self.panel, nav, ref, tgt),
            show_test_window=self.toggles.show_test_window,
            cs_overlay=nav.cs_overlay_active,
            is_significant=nav.is_significant(ref, tgt, seg_idx),
            ylim_override=self._same_scale_ylim(nav, self.panel, data, ref, tgt, neurons),
            cs_annotation_lines=(self._cs_annotation_lines(nav, self.cs, ref, tgt, seg_idx)
                                 if nav.cs_overlay_active else []),
        )

    def for_extend(self, ext_view: 'ExtendRow',
                   seg_label=None) -> 'RenderContext | None':
        """RenderContext for the pair recomputed at the extend row's window and bin.

        Nothing is stored at this window/bin, so the baseline and p-values are
        always fitted here. Cached per (view, segment, window, bin, norms, toggles).
        """
        nav = self.nav
        if ext_view is None or not ext_view.enabled or nav.current_pair_inds is None:
            return None
        view = nav.get_complete_key()
        ref, tgt = view.ref, view.tgt
        if ref == tgt or nav.neurons is None:
            return None

        extend_ms = max(5, ext_view.extend_ms)
        # extend recomputes from spike times: the sample period bounds it, not what is stored
        extend_bin_ms = max(ext_view.extend_bin_ms, 1000.0 / nav.neurons.sampling_rate)
        duration, bin_size = extend_ms / 1000.0, extend_bin_ms / 1000.0
        seg_label = seg_label or nav.current_segment

        cache_key = (str(view), seg_label, extend_ms, extend_bin_ms,
                     frozenset(nav.active_norms), nav.key.excitability, self.toggles,
                     nav.baseline_method, tuple(nav.cd.conf.tail_intervals),
                     nav.cd.conf.tail_source, nav.cs_overlay_active)
        hit = self.panel._extend_cache.get(cache_key)
        if hit is not None:
            return hit

        conf = nav.ccg_data.conf if nav.ccg_data is not None else nav.cd.conf
        full = self._compute_extend_ccg(nav, ref, tgt, duration, bin_size, conf, seg_label)
        if full is None:
            return None

        source = CCGSource(
            ccg=full[0, 1, :], baseline=None, pval=None, pval_corrected=None,
            acg_ref=full[0, 0, :], acg_tgt=full[1, 1, :],
            conf=conf, bin_size=bin_size, duration=duration, neurons=nav.neurons,
            time_hours=self._time_hours_for_seg(nav, nav.segment_index(seg_label)),
            refit_after_deconv=False,
        )
        ctx = self._finish(
            source, ref, tgt,
            seg_display=f'{seg_label} (extend {extend_ms}ms @ {extend_bin_ms:.4f}ms/bin)',
            sess_label=str(view.session or ''),
            jitter=JitterOverlay(), show_test_window=nav.show_test_window,
            cs_overlay=nav.cs_overlay_active,
            is_significant=False,
            base_window_ms=(conf.duration or 0.0) * 1000.0, extend_on=True,
        )
        self.panel._extend_cache.put(cache_key, ctx)
        return ctx

    # ── the shared tail ────────────────────────────────────────────────

    def _cs_at(self, ccg, baseline, conf, bin_size_eff) -> float | None:
        """CS over the test window, in bins of *this* view — extend views rebin it."""
        if baseline is None or not len(ccg):
            return None
        return ConnectionStrength.conn_strength(
            ccg, None, 0, 0, conf, baseline=baseline, bin_size_eff=bin_size_eff,
            excitability=self.nav.key.excitability)

    def _finish(self, src: CCGSource, ref: int, tgt: int, *,
                seg_display, sess_label, jitter, show_test_window, cs_overlay,
                is_significant, cs_annotation_lines=None, ylim_override=None,
                base_window_ms=None, extend_on=False) -> RenderContext:
        """Apply deconvolution, display toggles and normalization, then pack a context."""
        tg = self.toggles
        wf_ms, wf_amp = (peak_waveform_on_lag_axis(src.neurons, ref, CH_PER_SHANK)
                         if tg.show_ref_waveform else (None, None))
        ccg = self._deconvolve(src, ref, tgt)

        if src.baseline is None or ccg is not src.ccg:
            # no stored fit, or the deconvolved CCG invalidated the stored one
            baseline, pval, pval_corrected = self._refit_conv(
                ccg, src.conf, src.bin_size, self.nav.key.excitability)
        else:
            baseline, pval, pval_corrected = src.baseline, src.pval, src.pval_corrected

        # only conv reuses the fit above; other methods fit themselves in the dispatcher
        baseline = (baseline if tg.baseline.show
                    and self.nav.baseline_method == 'conv' else None)
        pval = pval if tg.show_pval else None
        pval_corrected = pval_corrected if tg.show_pval_corrected else None
        acg_ref = src.acg_ref if tg.acg_ref.show else None
        acg_tgt = src.acg_tgt if tg.acg_tgt.show else None

        ccg, baseline = self._normalize(src, ccg, baseline, ref, tgt)
        acg_ref = self._normalize_acg(src, acg_ref, ref)
        acg_tgt = self._normalize_acg(src, acg_tgt, tgt)

        nt_ref, nt_tgt, sh_ref, sh_tgt = self._neuron_meta(src.neurons, ref, tgt)
        bin_size_eff = (src.duration / (len(ccg) - 1) if len(ccg) > 1
                        else src.bin_size)
        if tg.baseline.show or cs_overlay:
            baseline = ConnectionStrength.baseline(
                ccg, baseline, src.conf, self.nav.baseline_method,
                bin_size_eff=bin_size_eff)
        return RenderContext(
            ccg=ccg, bin_size_eff=bin_size_eff, window_size_eff=src.duration,
            alpha=self.nav.active_sig_threshold, seg_id_display=seg_display,
            inds=(ref, tgt), jitter=jitter, dark_mode=self.dark_mode,
            title=TitleConfig(
                title_session_label=sess_label,
                title_show_session=True, title_show_type=True,
                title_show_inds=True, title_show_seg=True,
                title_show_shanks=(sh_ref is not None),
            ),
            style=PlotStyle(),
            ccg_null_plot=baseline if tg.baseline.show else None,
            pval=pval, pval_corrected=pval_corrected,
            acg_ref=acg_ref, acg_tgt=acg_tgt,
            wf_peak_ms=wf_ms, wf_peak_amp=wf_amp,
            cs_baseline_arg=baseline if (baseline is not None and cs_overlay) else None,
            cs_value=self._cs_at(ccg, baseline, src.conf, bin_size_eff),
            norm_info=None,
            wf_y_pad=tg.wf_y_pad,
            extend_on=extend_on,
            cs_annotation_lines=cs_annotation_lines or [],
            min_lag_plot=src.conf.min_lag if show_test_window else None,
            max_lag_plot=src.conf.max_lag if show_test_window else None,
            cs_window=(src.conf.min_lag, src.conf.max_lag),
            tail_plot=(src.conf.tail_intervals if self.toggles.show_tail_window else None),
            neuron_type=(nt_ref, nt_tgt) if (nt_ref or nt_tgt) else None,
            shank_ids=(sh_ref, sh_tgt) if sh_ref is not None else None,
            show_ccg=tg.ccg.show, line_ccg=tg.ccg.line,
            line_baseline=tg.baseline.line,
            line_ref=tg.acg_ref.line if acg_ref is not None else False,
            line_tgt=tg.acg_tgt.line if acg_tgt is not None else False,
            line_jitter=tg.line_jitter,
            acg_yscale_ref=tg.acg_ref.yscale, acg_yscale_tgt=tg.acg_tgt.yscale,
            acg_match_ccg=tg.acg_match_ccg,
            ylim=ylim_override or self._ylim(ccg, baseline),
            is_significant_pair=is_significant,
            base_window_ms=base_window_ms,
        )

    def _deconvolve(self, src: CCGSource, ref: int, tgt: int) -> np.ndarray:
        """CCG with the requested autocorrelograms divided out; src.ccg if none are."""
        return CCGNorm.deconv_for_pair(
            src.ccg, src.neurons, ref, tgt,
            acg_ref=src.acg_ref if self.toggles.acg_ref.deconv else None,
            acg_tgt=src.acg_tgt if self.toggles.acg_tgt.deconv else None)

    def _normalize(self, src: CCGSource, ccg, baseline, ref: int, tgt: int):
        """(ccg, baseline) under the active normalizations."""
        return CCGNorm.apply(ccg, baseline, ref, tgt, self.nav.active_norms,
                             neurons=src.neurons, custom_time_hours=src.time_hours)

    def _normalize_acg(self, src: CCGSource, acg, neuron: int):
        """One ACG under the active normalizations; BASELINE never applies to an ACG."""
        if acg is None:
            return None
        return CCGNorm.apply(acg, None, neuron, neuron,
                             self.nav.active_norms - {NormalizeBy.BASELINE},
                             neurons=src.neurons, custom_time_hours=src.time_hours)[0]

    @staticmethod
    def _ylim(ccg, baseline):
        """(0, top) covering the CCG and its baseline, or None when both are flat."""
        ccg_top = float(np.nanmax(ccg)) if len(ccg) else 0.0
        base_top = (float(np.nanmax(baseline))
                    if baseline is not None and len(baseline) else 0.0)
        top = max(ccg_top, base_top) * 1.05
        return (0.0, top) if top > 0 else None

    # ── backwards-compatible entry points ──────────────────────────────

    @classmethod
    def build_context(cls, nav, panel, seg_label=None, hi_res_override=None,
                      pair_override=None) -> 'RenderContext | None':
        return cls(nav, panel).for_current(seg_label, hi_res_override, pair_override)

    @classmethod
    def build_extend_context(cls, nav, panel, seg_label=None,
                             ext_view: 'ExtendRow' = None) -> 'RenderContext | None':
        return cls(nav, panel).for_extend(ext_view, seg_label)

    @staticmethod
    def _cs_annotation_lines(nav, cs_section: 'BaselineCSSection',
                              ref: int, tgt: int, seg_idx: int) -> list:
        """Connection-strength lines stamped onto an exported PNG, one per loaded resolution."""
        nonneg = nav.cs_nonneg
        lines = []
        for resolution, label in (('lowres', 'lo-res'), ('highres', 'hi-res')):
            value = CCGContextBuilder._cs_value(nav, cs_section.jitter_mgr, seg_idx,
                                                ref, tgt, resolution, nonneg=nonneg)
            if value is not None:
                lines.append(f"{nav.cs_metric} ({nav.baseline_method}) {label}: {value:.4f}")
        return lines

    @staticmethod
    def _compute_extend_ccg(nav, ref: int, tgt: int, duration: float,
                            bin_size: float, conf, seg_label: str):
        """Recompute the pair at this window and bin as [2,2,bins], ACGs on the diagonal."""
        pair_neurons = nav.neurons.neuron_slice(neuron_inds=np.array([ref, tgt]))
        # an appended window carries its own extent; 'full' spans the session
        source = (nav.cd.source_config(
                      nav.get_complete_key().change(resolution=nav.data_resolution),
                      seg_label)
                  if seg_label else None)
        windowed = (source is not None
                    and not isinstance(source.t0, str)
                    and not isinstance(source.t1, str))
        kwargs = dict(neuron_inds=np.array([0, 1]), bin_size=bin_size,
                      window_size=duration, symmetrize=conf.symmetrize_ccg,
                      use_acceleration=conf.use_acceleration)
        try:
            if windowed:
                extent = np.array([[float(source.t0)], [float(source.t1)]])
                full = correlations.spike_correlations(
                    pair_neurons, start_end_times=extent, **kwargs)[0]
            else:
                full = correlations.spike_correlations(pair_neurons, **kwargs)
        except (ValueError, IndexError, MemoryError) as exc:
            # a window/bin combination the correlator rejects must not kill the GUI
            print(f"[CCGPanel] extend compute failed: {exc}", flush=True)
            return None
        full = np.asarray(full, dtype=float)
        return full if full.size > 0 else None


class CCGPlotWidget(QWidget):
    """CCG plot: bars, baseline, ACGs, test window and p-values, one _draw_ per layer."""

    context_menu_requested = Signal(object)   # QPoint

    _MIN_PLOT_H = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self._resize_render_pending = False
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Container of per-row HBoxes (rows stack vertically; each row scrolls
        # horizontally if it holds multiple resolution/extend variants)
        self._plot_container = QWidget()
        self._plot_container.setMinimumHeight(self._MIN_PLOT_H)
        self._plot_grid = QVBoxLayout(self._plot_container)
        self._plot_grid.setContentsMargins(0, 0, 0, 0)
        self._plot_grid.setSpacing(2)
        self._plot_scroll = QScrollArea()
        self._plot_scroll.setWidgetResizable(True)
        self._plot_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._plot_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._plot_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._plot_scroll.setWidget(self._plot_container)
        outer.addWidget(self._plot_scroll, stretch=1)

        self._row_layouts: list = []   # QHBoxLayout, one per stacked row
        self._subplots: list[Subplot] = []

        self._last_rows: list | None = None
        self._rebuild_subplots([1])

    def showEvent(self, event):
        super().showEvent(event)
        self._schedule_resize_render()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_resize_render()

    def _schedule_resize_render(self):
        if self._last_rows is None:
            return
        if self._resize_render_pending:
            return
        self._resize_render_pending = True
        QTimer.singleShot(0, self._render_after_resize)

    def _render_after_resize(self):
        self._resize_render_pending = False
        if self._last_rows is not None and self._plot_scroll.viewport().height() > 20:
            self.render(self._last_rows)

    def _rebuild_subplots(self, row_lengths: list):
        """Discard every subplot and lay out fresh ones, row_lengths[i] per row."""
        for sub in self._subplots:
            scene = sub.widget.scene()
            for item in ([sub.pval_vb, sub.wf_vb, sub.title]
                         + [vb for vb, _ in sub.acg_axes]):
                scene.removeItem(item)          # scene-level, so the widget never owned them
            for axis in [sub.pval_axis] + [a for _, a in sub.acg_axes]:
                sub.plot.layout.removeItem(axis)
            sub.widget.setParent(None)
            sub.widget.deleteLater()
        for row_layout in self._row_layouts:
            self._plot_grid.removeItem(row_layout)
        self._row_layouts.clear()
        self._subplots.clear()

        for row_len in row_lengths:
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(2)
            for _ in range(row_len):
                slot = len(self._subplots)
                sub = self._make_subplot(slot)
                row_layout.addWidget(sub.widget)
                self._subplots.append(sub)
            self._plot_grid.addLayout(row_layout)
            self._row_layouts.append(row_layout)

    def _make_subplot(self, slot: int) -> 'Subplot':
        """One plot with its p-value and ACG overlays wired up."""
        pw = pg.PlotWidget()
        pw.setMinimumSize(280, self._MIN_PLOT_H)
        pw.setBackground('w')
        pw.showGrid(x=False, y=True, alpha=0.3)
        pw.setMouseEnabled(x=False, y=False)
        pw.getViewBox().setMouseEnabled(x=False, y=False)
        pw.scene().sigMouseClicked.connect(
            lambda ev, k=slot: self._on_mouse_click(k, ev))
        pw.scene().sigMouseMoved.connect(
            lambda pos, k=slot: self._show_bin_readout(self._time_at(k, pos)))

        p = pw.getPlotItem()
        # the title's text width is a layout minimum that would hold the plot
        # wider than its column; uncapped it overflows onto the next plot
        p.titleLabel.setMaximumWidth(1)   # pinned: its text width is a column minimum
        p.setTitle(' ', size='9pt')       # reserves the strip the scene-level title draws in

        pval_vb = self._add_overlay_viewbox(pw)
        pval_axis = pg.AxisItem('right')
        pval_axis.linkToView(pval_vb)
        pval_axis.setPen(pg.mkPen(PVAL_COLOR))
        pval_axis.setTextPen(pg.mkPen(PVAL_COLOR))
        pval_axis.setLabel('p', color=PVAL_COLOR)
        p.layout.addItem(pval_axis, 2, 5)
        pval_axis.hide()
        p.layout.setColumnMinimumWidth(3, 14)   # gap: keeps the outer axis clear of the inner label
        acg_axes = []
        for col, color, name in ((2, ACG_REF_COLOR, 'ACG ref'),
                                 (4, ACG_TGT_COLOR, 'ACG tgt')):
            view_box = self._add_overlay_viewbox(pw)
            view_box.setXLink(p.vb)
            axis = pg.AxisItem('right')
            axis.linkToView(view_box)
            axis.setPen(pg.mkPen(color))
            axis.setTextPen(pg.mkPen(color))
            axis.setLabel(name, color=color)
            p.layout.addItem(axis, 2, col)
            axis.hide()
            acg_axes.append((view_box, axis))

        wf_vb = self._add_overlay_viewbox(pw)
        overlays = [pval_vb, wf_vb] + [vb for vb, _ in acg_axes]
        syncing = [False]

        def sync_geometry():
            if syncing[0]:      # setGeometry re-emits sigResized
                return
            syncing[0] = True
            try:
                rect = p.vb.sceneBoundingRect()
                for view_box in overlays:
                    view_box.setGeometry(rect)
                # centred over the plot body: a scene item, so it sets no column minimum
                title.setGeometry(QRectF(rect.x(), rect.y() - 20, rect.width(), 20))
            finally:
                syncing[0] = False

        p.vb.sigResized.connect(sync_geometry)

        title = pg.LabelItem('', justify='center')
        pw.scene().addItem(title)

        # on the ViewBox: p.clear() would drop it from the PlotItem
        readout = pg.TextItem(anchor=(0, 1))
        readout.setZValue(p.vb.zValue() + 3)   # above every overlay
        readout.hide()
        p.vb.addItem(readout, ignoreBounds=True)
        return Subplot(widget=pw, plot=p, pval_vb=pval_vb, pval_axis=pval_axis,
                       wf_vb=wf_vb, acg_axes=acg_axes, readout=readout, title=title)

    @staticmethod
    def _add_overlay_viewbox(pw) -> pg.ViewBox:
        """A scene-level view box for an overlay: no zoom, no context menu."""
        view_box = pg.ViewBox()
        view_box.setMouseEnabled(x=False, y=False)
        view_box.setMenuEnabled(False)
        pw.scene().addItem(view_box)
        return view_box

    def _on_mouse_click(self, i: int, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.context_menu_requested.emit(event.screenPos().toPoint())
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._show_bin_readout(self._time_at(i, event.scenePos()))

    def _time_at(self, slot: int, scene_pos) -> float | None:
        """Lag in ms under a point in subplot *slot*'s own scene, or None if off the axes."""
        sub = self._subplots[slot]
        if sub.ctx is None or not sub.plot.vb.sceneBoundingRect().contains(scene_pos):
            return None
        return sub.plot.vb.mapSceneToView(scene_pos).x()

    @staticmethod
    def _bin_of(ctx, t_ms: float) -> int | None:
        """Bin holding lag *t_ms*, or None when that lag is off this plot's window."""
        bs = ctx.bin_size_eff * 1000.0
        n  = len(np.asarray(ctx.ccg))
        if not bs or n == 0:
            return None
        b = int(round((t_ms + ctx.window_size_eff * 1000.0 / 2) / bs))
        return b if 0 <= b < n else None

    def _show_bin_readout(self, t_ms: float | None) -> None:
        """Label the same lag on every subplot, each in its own bins, for comparison."""
        for sub in self._subplots:
            ctx = sub.ctx
            bin_idx = None if (ctx is None or t_ms is None) else self._bin_of(ctx, t_ms)
            if bin_idx is None:
                sub.readout.hide()
                continue
            ccg = np.asarray(ctx.ccg, dtype=float)
            x = -ctx.window_size_eff * 1000.0 / 2 + bin_idx * ctx.bin_size_eff * 1000.0
            lines = [f"time: {x:g} ms", f"count: {ccg[bin_idx]:g}"]
            baseline = ctx.ccg_null_plot
            if baseline is not None and bin_idx < len(baseline):
                lines.append(f"baseline: {float(baseline[bin_idx]):g}")
            sub.readout.setText('\n'.join(lines),
                                color='#dddddd' if ctx.dark_mode else '#222222')
            sub.readout.fill = pg.mkBrush(*((30, 30, 30, 220) if ctx.dark_mode
                                            else (255, 255, 255, 225)))
            sub.readout.setPos(x, ccg[bin_idx])
            sub.readout.show()

    def render(self, rows) -> None:
        """Draw from rows of RenderContexts: each row is a list of contexts laid
        out side by side; rows themselves stack vertically."""
        if rows and not isinstance(rows[0], list):
            rows = [rows]
        rows = [[c for c in row if c is not None] for row in rows]
        rows = [row for row in rows if row]
        self._last_rows = rows
        row_lengths = [len(row) for row in rows] or [1]
        if [rl.count() for rl in self._row_layouts] != row_lengths:
            self._rebuild_subplots(row_lengths)
        contexts = [c for row in rows for c in row]
        # 'visible' is resolved here: only render sees every plot on screen at once
        tops: dict = {}
        for c in contexts:
            if c.ylim == 'visible' and c.ccg is not None and len(c.ccg):
                tops[len(c.ccg)] = max(tops.get(len(c.ccg), 0.0), float(np.nanmax(c.ccg)))
        for c in contexts:
            if c.ylim == 'visible':      # lo and hi rows scale apart; empty traces autoscale
                top = tops.get(len(c.ccg) if c.ccg is not None else -1, 0.0)
                c.ylim = (0.0, top * 1.1) if top > 0 else None
        dark = contexts[0].dark_mode if contexts else qt_dark_mode()
        plot_bg = '#1e1e1e' if dark else 'w'
        for i, sub in enumerate(self._subplots):
            sub.widget.setBackground(plot_bg)
            self._render_one(sub, contexts[i] if i < len(contexts) else None)
        self._show_bin_readout(None)   # stale labels point at the previous data
        vp = self._plot_scroll.viewport()
        max_row_len = max(row_lengths)
        plot_w = max(320 * max_row_len, vp.width())
        plot_h = max(self._MIN_PLOT_H * len(row_lengths), vp.height())
        self._plot_container.setMinimumSize(plot_w, plot_h)

    def _render_one(self, sub: 'Subplot', ctx) -> None:
        """Draw one context onto a subplot: CCG, then every overlay it asks for."""
        p, pval_vb, pval_items, acg_axes = (sub.plot, sub.pval_vb,
                                            sub.pval_items, sub.acg_axes)
        p.clear()
        for view_box, axis in acg_axes:
            view_box.clear()
            axis.hide()
        for item in pval_items:
            pval_vb.removeItem(item)
        pval_items.clear()
        sub.ctx = ctx
        if ctx is None:
            return

        ccg = np.asarray(ctx.ccg, dtype=float)
        if len(ccg) == 0:
            return
        apply_plot_chrome(p, ctx.dark_mode)
        geom = LagAxis.of(ctx, len(ccg))

        self._draw_ccg(p, ctx, ccg, geom)
        self._draw_baseline(p, ctx, geom)
        self._draw_connection_strength(p, ctx, ccg, geom)
        self._draw_test_window(p, ctx, geom)
        self._draw_tail_window(p, ctx, geom)
        self._draw_base_window_edges(p, ctx, geom)
        self._draw_acgs(p, ctx, geom, acg_axes)
        self._draw_ref_waveform(sub, ctx, geom)
        self._draw_jitter(p, ctx, geom)
        self._draw_pvalues(p, ctx, geom, pval_vb, pval_items, sub.pval_axis)

        if ctx.ylim is not None:
            p.setYRange(*ctx.ylim, padding=_Y_PAD)
        else:
            p.enableAutoRange(axis='xy', enable=True)
            p.autoRange()
        fg = '#dddddd' if ctx.dark_mode else '#333333'
        sub.title.setText(self._make_title(ctx), size='9pt', color=fg)
        p.setLabel('bottom', 'Lag (ms)')

    @staticmethod
    def _draw_ccg(p, ctx, ccg, geom: 'LagAxis') -> None:
        if not ctx.show_ccg:
            return
        color = '#7aafff' if ctx.dark_mode else '#4a7fd4'
        if ctx.line_ccg:
            p.addItem(pg.PlotDataItem(geom.edges, ccg, stepMode='center',
                                      pen=plot_pen(color), fillLevel=None))
        else:
            p.addItem(pg.BarGraphItem(x=geom.centers, height=ccg,
                                      width=geom.bin_ms, brush=color, pen=None))

    @staticmethod
    def _draw_baseline(p, ctx, geom: 'LagAxis') -> None:
        if ctx.ccg_null_plot is None:
            return
        null = np.asarray(ctx.ccg_null_plot, dtype=float)
        color = '#cc6666' if ctx.dark_mode else '#e88'
        if ctx.line_baseline:
            p.addItem(pg.PlotDataItem(geom.edges[:len(null) + 1], null, stepMode='center',
                                      pen=plot_pen(color, Qt.PenStyle.DashLine)))
        else:
            p.addItem(pg.BarGraphItem(x=geom.centers[:len(null)], height=null,
                                      width=geom.bin_ms,
                                      brush=pg.mkBrush(color + '88'), pen=None))

    @staticmethod
    def _draw_connection_strength(p, ctx, ccg, geom: 'LagAxis') -> None:
        """Shade the CCG's excess over its baseline inside the test window."""
        if ctx.cs_baseline_arg is None or ctx.cs_window is None:
            return
        baseline = np.asarray(ctx.cs_baseline_arg, dtype=float)
        if len(baseline) != len(ccg):
            return
        mask = test_window_bin_mask(geom.centers, ctx.cs_window[0],
                                    ctx.cs_window[1], geom.bin_ms)
        bottoms = baseline[mask]
        color = '#3ecf6e' if ctx.dark_mode else '#1a6b2e'
        p.addItem(pg.BarGraphItem(x=geom.centers[mask],
                                  height=np.maximum(ccg[mask] - bottoms, 0),
                                  y0=bottoms, width=geom.bin_ms,
                                  brush=pg.mkBrush(color), pen=None))

    @staticmethod
    def _draw_tail_window(p, ctx, geom: 'LagAxis') -> None:
        """Shade each configured tail interval; a None edge reaches the window edge."""
        if not ctx.tail_plot:
            return
        half = geom.window_ms / 2 + geom.bin_ms
        brush = (150, 110, 60, 55) if ctx.dark_mode else (255, 220, 170, 90)
        for start, end in ctx.tail_plot:
            lo = -half if start is None else start * 1000.0
            hi = half if end is None else end * 1000.0
            p.addItem(pg.LinearRegionItem(values=[lo, hi], brush=pg.mkBrush(*brush),
                                          pen=pg.mkPen(None), movable=False))

    @staticmethod
    def _draw_test_window(p, ctx, geom: 'LagAxis') -> None:
        if not geom.has_test_window(ctx):
            return
        span_lo, span_hi = test_window_span_ms(ctx.min_lag_plot, ctx.max_lag_plot,
                                               geom.bin_ms)
        brush = (80, 100, 140, 60) if ctx.dark_mode else (200, 220, 255, 60)
        p.addItem(pg.LinearRegionItem(values=[span_lo, span_hi],
                                      brush=pg.mkBrush(*brush),
                                      pen=pg.mkPen(None), movable=False))

    @staticmethod
    def _draw_base_window_edges(p, ctx, geom: 'LagAxis') -> None:
        """In extend mode, mark where the un-extended window ended."""
        if not ctx.base_window_ms or ctx.base_window_ms >= geom.window_ms:
            return
        for edge in (-ctx.base_window_ms / 2, ctx.base_window_ms / 2):
            p.addItem(pg.InfiniteLine(pos=edge, angle=90,
                                      pen=plot_pen('#e74c3c', Qt.PenStyle.DashLine)))

    @staticmethod
    def _draw_acgs(p, ctx, geom: 'LagAxis', acg_axes) -> None:
        """Each ACG on its own right-hand axis, since their y-scales differ from the CCG's."""
        overlays = (
            (acg_axes[0], ACG_REF_COLOR, ctx.acg_ref, ctx.acg_yscale_ref, ctx.line_ref),
            (acg_axes[1], ACG_TGT_COLOR, ctx.acg_tgt, ctx.acg_yscale_tgt, ctx.line_tgt),
        )
        for (view_box, axis), color, data, scale, as_line in overlays:
            if data is None:
                continue
            acg = np.asarray(data, dtype=float)
            if as_line:
                view_box.addItem(pg.PlotDataItem(geom.edges[:len(acg) + 1], acg,
                                                 stepMode='center', pen=plot_pen(color)))
            else:
                view_box.addItem(pg.BarGraphItem(x=geom.centers[:len(acg)], height=acg,
                                                 width=geom.bin_ms,
                                                 brush=pg.mkBrush(color + '66'), pen=None))
            peak = float(np.nanmax(acg)) if len(acg) else 0.0
            top = peak * 1.1 if peak > 0 else 1.0
            view_box.setYRange(0, top if ctx.acg_match_ccg else top / max(scale, 0.01),
                               padding=_Y_PAD)
            view_box.setGeometry(p.vb.sceneBoundingRect())
            view_box.setZValue(1)   # scene sibling of PlotItem (z=0), which holds the CCG
            axis.show()

    @staticmethod
    def _draw_ref_waveform(sub: 'Subplot', ctx, geom: 'LagAxis') -> None:
        """Reference peak-channel waveform over the CCG, on the same lag axis."""
        view_box = sub.wf_vb
        view_box.clear()
        if ctx.wf_peak_ms is None or ctx.wf_peak_amp is None:
            view_box.hide()
            return
        view_box.setGeometry(sub.plot.vb.sceneBoundingRect())
        view_box.setXLink(sub.plot.vb)
        view_box.setZValue(3)   # above the p-value box (2) and the ACGs (1)
        view_box.addItem(pg.PlotDataItem(ctx.wf_peak_ms, ctx.wf_peak_amp,
                                         pen=plot_pen(WF_COLOR)))
        # zero on the shared line: same height fraction as every (0, top) overlay
        amp = np.asarray(ctx.wf_peak_amp, dtype=float)
        pad = max(ctx.wf_y_pad, 1e-6)
        frac = pad / (1.0 + 2.0 * pad)
        top = max(float(np.nanmax(np.abs(amp))), 1e-12)
        view_box.setYRange(-top * frac / (1.0 - frac), top, padding=0)
        view_box.show()

    @staticmethod
    def _draw_jitter(p, ctx, geom: 'LagAxis') -> None:
        jitter = ctx.jitter
        if jitter.j_ccg is None:
            return
        xs = geom.centers[:len(jitter.j_ccg)]
        if jitter.j_ccg_lo is not None and jitter.j_ccg_hi is not None:
            lo = pg.PlotDataItem(xs, jitter.j_ccg_lo, pen=None)
            hi = pg.PlotDataItem(xs, jitter.j_ccg_hi, pen=None)
            p.addItem(lo)
            p.addItem(hi)
            p.addItem(pg.FillBetweenItem(lo, hi, brush=pg.mkBrush(180, 160, 210, 100)))
        style = Qt.PenStyle.DashLine if ctx.line_jitter else Qt.PenStyle.SolidLine
        p.plot(xs, jitter.j_ccg, pen=plot_pen('#9b59b6', style))
        if jitter.j_pval is not None:
            p.plot(xs, jitter.j_pval, pen=plot_pen('#6c3483', Qt.PenStyle.DotLine))

    @staticmethod
    def _draw_pvalues(p, ctx, geom: 'LagAxis', pval_vb, pval_items, pval_axis=None) -> None:
        """P-values on their own 0-1 view box, above the CCG and the ACGs."""
        pval_vb.setGeometry(p.vb.sceneBoundingRect())
        pval_vb.setZValue(2)
        if ctx.pval is None and ctx.pval_corrected is None:
            if pval_axis is not None:
                pval_axis.hide()
            return
        if pval_axis is not None:
            pval_axis.show()
        pval_vb.setYRange(0, 1, padding=_Y_PAD)   # not linked to the right axis: pyqtgraph recurses
        traces = ((ctx.pval, PVAL_COLOR, Qt.PenStyle.DotLine),
                  (ctx.pval_corrected, '#922b21', Qt.PenStyle.DashLine))
        for values, color, style in traces:
            if values is None:
                continue
            item = pg.PlotDataItem(geom.centers[:len(values)], values,
                                   pen=plot_pen(color, style))
            pval_vb.addItem(item)
            pval_items.append(item)
        alpha_line = pg.InfiniteLine(pos=ctx.alpha, angle=0,
                                     pen=plot_pen(PVAL_COLOR, Qt.PenStyle.DotLine))
        pval_vb.addItem(alpha_line)
        pval_items.append(alpha_line)

    @staticmethod
    def _make_title(ctx) -> str:
        parts = []
        if ctx.title.title_show_session and ctx.title.title_session_label:
            parts.append(ctx.title.title_session_label)
        if ctx.seg_id_display and ctx.title.title_show_seg:
            parts.append(f"{ctx.seg_id_display}:")
        if ctx.title.title_show_shanks and ctx.shank_ids is not None:
            sh = ' '.join(str(x) for x in ctx.shank_ids)
            parts.append(f"shank=({sh})")
        if ctx.title.title_show_inds and ctx.inds is not None:
            ind = ' '.join(str(x) for x in ctx.inds)
            parts.append(f"inds=({ind})")
        if ctx.neuron_type and ctx.title.title_show_type:
            a, b = ctx.neuron_type
            parts.append(f"{a}->{b}")
        return ', '.join(parts)


class WaveformPanelQt(QWidget):
    """Probe waveform display for current pair. Shown/hidden by Ctrl+E."""

    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            self._fig    = Figure(figsize=(3, 5), tight_layout=True)
            self._canvas = FigureCanvasQTAgg(self._fig)
            layout.addWidget(self._canvas)
            self._ok = True
        except Exception:
            self._ok = False

    def render(self, neurons, ref: int, tgt: int = None):
        """Both neurons of a pair, or just *ref* when tgt is None."""
        if not self._ok:
            return
        self._fig.clear()
        if neurons is None:
            self._canvas.draw()
            return
        waveforms = neurons.waveforms
        shank_ids = neurons.shank_ids
        if waveforms is None:
            self._canvas.draw()
            return
        ref_shank = int(shank_ids[ref]) if shank_ids is not None else 0
        tgt_shank = None if tgt is None else (
            int(shank_ids[tgt]) if shank_ids is not None else 0)
        # the plotter indexes one shank's 16 rows; waveforms span the whole probe
        ref_wf    = _fill_waveform(waveforms[ref], ref_shank, CH_PER_SHANK, None)
        tgt_wf    = None if tgt is None else _fill_waveform(
            waveforms[tgt], tgt_shank, CH_PER_SHANK, None)
        ax = self._fig.add_subplot(111)
        try:
            from neuropy.plotting.probe import plot_waveform_on_channel
            plot_waveform_on_channel(ref_wf, ref_shank, tgt_wf, tgt_shank,
                                     ax=ax, ch_per_shank=CH_PER_SHANK)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
        self._canvas.draw()


class CorrelogramPanel(QWidget):
    """Center panel: CCG plot + toolbox.

    Wires AppState signals to render pipeline.
    Per-panel display config lives on the section sub-widgets.
    """

    # Emitted whenever a plot redraw is needed (parent connects to its refresh)
    plot_update_requested = Signal()

    def __init__(self, nav: 'AppState', parent=None):
        super().__init__(parent)
        self.nav = nav
        self._theme_fn = None
        self._extend_cache: LRUCache = LRUCache(32)   # several extend views × segments
        self._same_scale_cache: LRUCache = LRUCache(4)
        self._build()
        self._connect_nav()
        self._connect_sections()

    def refresh_font(self):
        self.seg_bar.refresh_font()
        self.cs_section.refresh_font()

    def set_jitter_mgr(self, jctrl):
        self.jitter_mgr = jctrl
        self.jitter_section.set_jitter_mgr(jctrl)
        self.cs_section.set_jitter_mgr(jctrl)

    def _build(self):
        self.jitter_mgr = None   # set externally via set_jitter_mgr()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_plot_area())
        splitter.addWidget(self._build_toolbox())
        splitter.setSizes([600, 250])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        for i in range(splitter.count()):
            splitter.setCollapsible(i, False)
        root.addWidget(splitter, stretch=1)

        self.plot_widget.context_menu_requested.connect(self._show_context_menu)

    def _build_plot_area(self) -> QSplitter:
        """CCG plot beside the waveform panel, which starts hidden."""
        self.plot_widget = CCGPlotWidget()
        self._wf_panel = WaveformPanelQt()
        self._wf_panel.setVisible(False)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.plot_widget)
        splitter.addWidget(self._wf_panel)
        splitter.setSizes([700, 300])
        return splitter

    def _build_toolbox(self) -> QScrollArea:
        """Every control section, stacked and scrollable."""
        self.seg_bar = SegmentBar(self.nav)
        self.norm_section = NormSection(self.nav)
        self.corr_section = CorrelogramSection()
        self.cs_section = BaselineCSSection(self.nav)
        self.jitter_section = JitterSection(self.nav)
        self.sa_section = SpikeAttributionSection()

        toolbox = QWidget()
        layout = QVBoxLayout(toolbox)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(3)
        for section in (self.seg_bar, self.norm_section, self.corr_section,
                        self.cs_section, self.jitter_section, self.sa_section):
            layout.addWidget(self._hline())
            layout.addWidget(section)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(toolbox)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setFixedHeight(1)
        return line

    @staticmethod
    def _render_fail_reason(nav) -> str:
        """Why build_context returned None — for terminal diagnostics."""
        n = len(nav.all_pairs_np)
        idx = nav.current_pair_idx
        inds = nav.current_pair_inds
        if n == 0:
            return (f"no pairs in sel_data (nav.key={nav.key}) — "
                    "plot needs at least one pair in lists")
        if inds is None:
            return f"current_pair_inds None (idx={idx}, n_pairs={n})"
        ref, tgt = int(inds[0]), int(inds[1])
        data = nav.ccg_data
        if data is None:
            return f"pair=({ref},{tgt}) but cd.ccg_for({nav.key.nd()}) is None"
        arr = data.ccg
        if arr is None:
            return f"pair=({ref},{tgt}) but ccg array is None"
        seg_idx = nav.segment_index(nav.current_segment)
        if seg_idx < arr.shape[0] or seg_idx == nav.n_segments:
            pass  # All-segment sum path — build_context handles
        return (f"pair=({ref},{tgt}) ccg shape={arr.shape} seg={nav.current_segment!r} "
                f"but slice failed — check segment index")

    def _together_handle(self, entry):
        """Normalize a together_pairs entry → (pair_key, ref, tgt).

        Entries are (Key, ref, tgt) in all-session mode, else (ref, tgt) in the
        current session.
        """
        if len(entry) == 3:
            return entry[0], int(entry[1]), int(entry[2])
        return self.nav.key, int(entry[0]), int(entry[1])

    def request_render(self):
        nav = self.nav
        cor = self.corr_section
        if nav.current_pair_inds is not None:
            cor.set_sampling_rate(nav.neurons.sampling_rate)
        segs = list(nav.stacked_segments) or [nav.current_segment]
        # Row axis = view kind (lo / hi / extend); column axis = segment. Transposed: swap.
        builders = ([lambda s, hi=hi: CCGContextBuilder.build_context(nav, self, seg_label=s, hi_res_override=hi)
                     for hi in ([False, True] if nav.resolution == "lo_hi" else [None])]
                    + [lambda s, v=v: CCGContextBuilder.build_extend_context(
                           nav, self, seg_label=s, ext_view=v)
                       for v in cor.extend_views])
        rows = [[build(seg) for seg in segs] for build in builders]
        if nav.stacked_transposed:
            rows = [list(r) for r in zip(*rows)]
        rows = [[c for c in row if c is not None] for row in rows]
        # "Show Together": overlay each pinned pair's CCG on top of the current view, own row each.
        for entry in nav.together_pairs:
            pk, r, t = self._together_handle(entry)
            tctx = CCGContextBuilder.build_context(nav, self, pair_override=(pk, r, t))
            if tctx is not None:
                rows.append([tctx])
        rows = [r for r in rows if r]
        if not rows:
            print(f"[CCGPanel] RENDER FAILED: {self._render_fail_reason(nav)}",
                  flush=True)
        self.plot_widget.render(rows)
        self.cs_section.update_display()
        if hasattr(self, '_wf_panel') and self._wf_panel.isVisible():
            if nav.current_pair_inds is not None:
                # Pair's own session neurons (correct across sessions in all-session mode).
                ref, tgt, pair_key = CCGContextBuilder._resolve_pair(nav)
                self._wf_panel.render(nav.cd.nd.neurons_for(pair_key), ref, tgt)

    def _connect_nav(self):
        nav = self.nav
        for sig in (nav.key_changed, nav.pair_changed, nav.segment_changed,
                    nav.resolution_changed, nav.norms_changed,
                    nav.stacked_segments_changed, nav.cs_overlay_changed,
                    nav.sig_threshold_changed, nav.scale_mode_changed,
                    nav.display_changed):
            sig.connect(lambda _: self.plot_update_requested.emit())
        nav.cs_params_changed.connect(lambda *_: self.plot_update_requested.emit())
        nav.pair_changed.connect(self._update_jitter_baseline_state)
        self.plot_update_requested.connect(self.request_render)

    def _connect_sections(self):
        nav = self.nav

        self.norm_section.apply_requested.connect(self._on_apply_norms)
        self.cs_section.sig_changed.connect(lambda: self.plot_update_requested.emit())
        self.corr_section.style_changed.connect(self.plot_update_requested)
        self.corr_section.autoscale_btn.toggled.connect(self.plot_update_requested)
        self.jitter_section.jitter_done.connect(self.request_render)
        self.sa_section.set_requested.connect(self._on_spike_attr_set)
        self.sa_section.enable_toggled.connect(self._on_spike_attr_enable)

    def _update_jitter_baseline_state(self, *_):
        inds = self.nav.current_pair_inds
        has_jitter = (inds is not None and self.jitter_mgr is not None and
                      self.jitter_mgr.has_result(int(inds[0]), int(inds[1])))
        self.cs_section.set_jitter_baseline_enabled(has_jitter)

    def refresh_spike_attr_if_enabled(self):
        if not self.sa_section.is_enabled:
            return
        self._on_spike_attr_set(*self.sa_section._bin_metric.value())

    def _on_spike_attr_enable(self, enabled: bool):
        if not enabled:
            self.window().pairs_view.spike_pairs.clear()

    def _on_apply_norms(self):
        import copy
        norms = self.nav.active_norms
        if not norms:
            QMessageBox.information(self, "Apply to data", "No normalizations selected.")
            return
        data = self.nav.ccg_data
        if data is None:
            return
        name, ok = QInputDialog.getText(self, "Apply to data", "Name for normalized dataset:")
        if not ok or not name.strip():
            return
        nav = self.nav
        arr, null = data.ccg, data.ccg_null
        n_seg, n_ref, n_tgt, _ = arr.shape
        new_ccg  = np.empty_like(arr,  dtype=float)
        new_null = np.empty_like(null, dtype=float) if null is not None else None
        for seg in range(n_seg):
            for r in range(n_ref):
                for t in range(n_tgt):
                    c, cn = CCGNorm.apply(
                        arr[seg, r, t], null[seg, r, t] if null is not None else None,
                        r, t, norms,
                        neurons=nav.neurons,
                        custom_time_hours=CCGContextBuilder._time_hours_for_seg(nav, seg))
                    new_ccg[seg, r, t] = c
                    if new_null is not None and cn is not None:
                        new_null[seg, r, t] = cn
        new_data = copy.copy(data)
        new_data.ccg = new_ccg
        new_data.ccg_null = new_null
        new_data.pval = new_data.qval = None
        QMessageBox.information(self, "Done",
                                f"Normalized CCG '{name.strip()}' applied in memory.")

    def _on_spike_attr_set(self, bin_val: float, unit: str):
        p = self.window()
        fn = p._on_spike_attribution_set
        if fn is not None:
            fn(bin_val, unit)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Left:
            self.nav.set_current_segment(
                self.nav.segment_name(
                    (self.nav.segment_index(self.nav.current_segment) - 1)
                    % len(self.nav.available_segments())))
        elif key == Qt.Key.Key_Right:
            self.nav.set_current_segment(
                self.nav.segment_name(
                    (self.nav.segment_index(self.nav.current_segment) + 1)
                    % len(self.nav.available_segments())))
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_R:
            self.nav.set_resolution("lo" if self.nav.resolution == "hi" else "hi")
        else:
            super().keyPressEvent(event)

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction("Export view as PNG…", self._export_png)
        menu.addSeparator()
        view_menu = menu.addMenu("View values in terminal")
        for label, key in self._VIEWABLE:
            view_menu.addAction(label, lambda k=key: self._view_values(k))
        menu.exec(pos)

    def _export_png(self):
        ctx = CCGContextBuilder.build_context(self.nav, self)
        if ctx is None:
            return
        path, _ = QFileDialog.getSaveFileName(   # savefig picks the format from the suffix
            self, "Export view", "", "PNG files (*.png);;PDF files (*.pdf)")
        if not path:
            return
        try:
            render_ccg_png(ctx, path)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    # (menu label, RenderContext field)
    _VIEWABLE = [("CCG", 'ccg'), ("Reference ACG", 'acg_ref'),
                 ("Target ACG", 'acg_tgt'), ("Baseline", 'ccg_null_plot'),
                 ("P-values", 'pval')]

    def _view_values(self, field: str):
        """Print one context array to the terminal, from the plot's context menu."""
        ctx = CCGContextBuilder.build_context(self.nav, self)
        if ctx is None:
            return
        values = getattr(ctx, field)
        if values is None:
            print(f"[CCG] {field}: None")
        else:
            print(f"[CCG] {field}: shape={np.asarray(values).shape}\n{values}")
