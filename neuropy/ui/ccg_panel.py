"""CCG main view panel."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.Qt.QtCore import Qt, Signal, QObject, QTimer
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
from neuropy.analyses.ms_connectivity import _CCG_RESOLUTION, Key
from neuropy.plotting.ccg import (
    RenderContext, JitterOverlay, TitleConfig, PlotStyle,
    test_window_bin_mask, test_window_span_ms, render_ccg_png,
    ACG_REF_COLOR, ACG_TGT_COLOR,
)
from neuropy.ui.ui_common import qt_dark_mode, LRUCache
from neuropy.ui.utils import chip_button, CycleButton, FlowLayout, CollapsibleSection, ArrowChipBar, SliderWithInput, has_primary_modifier, small_font_pt

if TYPE_CHECKING:
    from neuropy.ui.app_state import AppState

pg.setConfigOptions(antialias=True)

_LINE_W = 2   # pen width for every line-mode overlay (CCG, baseline, ACG, jitter, p-values)


def _pen(color, style=None):
    """Overlay pen at _LINE_W. Round cap/join keeps dashes even on horizontal runs."""
    p = pg.mkPen(color, width=_LINE_W)
    p.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    if style is not None:
        p.setStyle(style)
    return p

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
        root.addWidget(self._chip_bar, stretch=1)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
        root.addWidget(sep)

        self._lo_hi_btn = chip_button("lo|hi", checkable=True)
        self._lo_hi_btn.toggled.connect(
            lambda on: self.nav.set_resolution("lo_hi" if on else "lo"))
        root.addWidget(self._lo_hi_btn)

        self._cs_btn = chip_button("CS", checkable=True)
        self._cs_btn.toggled.connect(self.nav.set_cs_overlay)
        root.addWidget(self._cs_btn)

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
        if nav.stacked_segments:
            menu.addSeparator()
            menu.addAction("Transpose stacked", nav.toggle_stacked_transposed)
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
        self._lo_hi_btn.blockSignals(True)
        self._lo_hi_btn.setChecked(res == "lo_hi")
        self._lo_hi_btn.blockSignals(False)

    def _on_cs_overlay_changed(self, active: bool):
        self._cs_btn.blockSignals(True)
        self._cs_btn.setChecked(active)
        self._cs_btn.blockSignals(False)

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
        (None,                     "Same scale (pair)"),
        (None,                     "Same scale (session)"),
    ]

    def __init__(self, nav: 'AppState', parent=None):
        super().__init__("Normalization", parent=parent)
        self.nav               = nav
        self._norm_btns: dict[NormalizeBy, QPushButton] = {}
        self._pair_scale_btn    = None
        self._session_scale_btn = None
        self._build()
        nav.norms_changed.connect(self._on_nav_norms_changed)
        nav.scale_mode_changed.connect(self._on_nav_scale_changed)
        self.norms_changed.connect(nav.set_active_norms)
        self.scale_changed.connect(nav.set_same_scale_mode)

    def _build(self):
        wrap = QWidget()
        wrap_layout = FlowLayout(wrap)

        for nm, label in self._NORM_OPTIONS:
            btn = chip_button(label, checkable=True)
            if nm is not None:
                self._norm_btns[nm] = btn
                btn.toggled.connect(lambda _checked, nm=nm: self._emit_norms())
            elif label == "Same scale (pair)":
                self._pair_scale_btn = btn
                btn.toggled.connect(lambda chk: self.scale_changed.emit('pair' if chk else None))
            else:
                self._session_scale_btn = btn
                btn.toggled.connect(lambda chk: self.scale_changed.emit('session' if chk else None))
            wrap_layout.addWidget(btn)

        self.body_layout.addWidget(wrap)

        apply_row = QHBoxLayout()
        apply_row.addStretch()
        apply_btn = chip_button("Apply to data…", checkable=False)
        apply_btn.clicked.connect(self.apply_requested)
        apply_row.addWidget(apply_btn)
        self.body_layout.addLayout(apply_row)

    def _emit_norms(self):
        self.norms_changed.emit({nm for nm, btn in self._norm_btns.items() if btn.isChecked()})

    def _on_nav_norms_changed(self, active: set):
        for nm, btn in self._norm_btns.items():
            btn.blockSignals(True)
            btn.setChecked(nm in active)
            btn.blockSignals(False)

    def _on_nav_scale_changed(self, mode):
        for btn, val in ((self._pair_scale_btn, 'pair'), (self._session_scale_btn, 'session')):
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(mode == val)
                btn.blockSignals(False)


class CorrelogramSection(CollapsibleSection):

    style_changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Correlogram", parent=parent)
        self._ref_scale_widget = None   # reserved for future per-ACG y-scale wiring
        self._tgt_scale_widget = None
        self._build()

    def _build(self):
        for row in (self._row1(), self._row2(), self._row3(), self._row4()):
            self.body_layout.addLayout(row)
        for btn in (self.ccg_btn, self.baseline_btn, self.ref_btn, self.tgt_btn,
                    self.ref_wf_btn, self.autoscale_btn, self.deconv_ref_btn,
                    self.deconv_tgt_btn, self.extend_check, self.jitter_line_btn):
            btn.clicked.connect(self.style_changed)

    @staticmethod
    def _vline() -> QFrame:
        f = QFrame(); f.setFrameShape(QFrame.Shape.VLine)
        return f

    def _scale_entry(self, name: str) -> 'SliderWithInput':
        w = SliderWithInput(1, 150, 100, scale=0.01)
        w.value_changed.connect(lambda _: self.style_changed.emit())
        setattr(self, f"_{name}_scale_widget", w)
        return w

    def _row1(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.ccg_btn, self.baseline_btn = CycleButton("CCG"), CycleButton("baseline")
        for btn in (self.ccg_btn, self.baseline_btn):
            row.addWidget(btn)
        row.addWidget(self._vline())
        row.addWidget(QLabel("Show ACG"))
        self.ref_btn = CycleButton("ref", start_hidden=True)
        self.tgt_btn = CycleButton("tgt", start_hidden=True)
        for btn in (self.ref_btn, self.tgt_btn):
            row.addWidget(btn)
        row.addWidget(self._vline())
        self.ref_wf_btn = chip_button("ref waveform", checkable=True)
        row.addWidget(self.ref_wf_btn)
        row.addStretch()
        return row

    def _row2(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.autoscale_btn = chip_button("Autoscale", checkable=True)
        row.addWidget(self.autoscale_btn)
        for name in ("ref", "tgt"):
            w = self._scale_entry(name)
            row.addWidget(QLabel(f"{name}:"))
            row.addWidget(w)
        row.addWidget(self._vline())
        row.addWidget(QLabel("Deconvolve"))
        self.deconv_ref_btn = chip_button("ref", checkable=True)
        self.deconv_tgt_btn = chip_button("tgt", checkable=True)
        for btn in (self.deconv_ref_btn, self.deconv_tgt_btn):
            row.addWidget(btn)
        row.addStretch()
        return row

    def _row3(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.extend_check = QCheckBox("Extend:")
        self._extend_ms_spin = self.make_spin((5, 10, 20, 50, 100, 200, 500, 1000), "50")
        self._extend_ms_spin.currentTextChanged.connect(lambda _: self.style_changed.emit())
        _min_bin_ms = _CCG_RESOLUTION['highres'] * 1000
        _bin_opts = sorted({round(_min_bin_ms, 4), 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0})
        self._extend_bin_spin = self.make_spin(_bin_opts, "1.0")
        self._extend_bin_spin.currentTextChanged.connect(lambda _: self.style_changed.emit())
        for w in (self.extend_check, self._extend_ms_spin,
                  QLabel("ms  resolution:"), self._extend_bin_spin, QLabel("ms")):
            row.addWidget(w)
        row.addStretch()
        return row

    def _row4(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.jitter_line_btn = CycleButton("jitter", start_hidden=True)
        row.addWidget(self.jitter_line_btn)
        row.addStretch()
        return row

    @property
    def acg_yscale_ref(self) -> float:
        w = self._ref_scale_widget
        return max(0.01, w.value) if w is not None else 1.0

    @property
    def acg_yscale_tgt(self) -> float:
        w = self._tgt_scale_widget
        return max(0.01, w.value) if w is not None else 1.0

    @property
    def extend_ms(self) -> int:
        try: return int(self._extend_ms_spin.currentText())
        except ValueError: return 50

    @property
    def extend_bin_ms(self) -> float:
        try: return float(self._extend_bin_spin.currentText())
        except ValueError: return 1.0


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

    def __init__(self, nav: 'AppState', parent=None):
        super().__init__("Baseline & Connection Strength", parent=parent)
        self.nav        = nav
        self.jitter_mgr = None  # set via set_jitter_mgr()
        self._build()
        nav.cs_overlay_changed.connect(self._on_cs_overlay_changed)
        nav.cs_params_changed.connect(self._on_cs_params_changed)
        self.baseline_changed.connect(lambda m: nav.set_cs_params(m, nav.cs_metric))
        self.metric_changed.connect(lambda m: nav.set_cs_params(nav.baseline_method, m))

    def refresh_font(self):
        self._explanation.setStyleSheet(f"color: #666; font-size: {small_font_pt()}pt;")

    def set_jitter_mgr(self, jctrl):
        self.jitter_mgr = jctrl

    def _build(self):
        layout = self.body_layout

        # Row 1: Show CS overlay checkbox | Measure: STG / JBSI
        row1 = QHBoxLayout()
        self.cs_show_check = QCheckBox("Show CS overlay")
        self.cs_show_check.toggled.connect(self.nav.set_cs_overlay)
        row1.addWidget(self.cs_show_check)
        row1.addWidget(QLabel("Measure:"))
        self._metric_group = QButtonGroup(self)
        for val in ("STG", "JBSI"):
            rb = QRadioButton(val)
            if val == "STG":
                rb.setChecked(True)
            self._metric_group.addButton(rb)
            row1.addWidget(rb)
        self._metric_group.buttonClicked.connect(
            lambda btn: self.metric_changed.emit(btn.text()))
        row1.addStretch()
        layout.addLayout(row1)

        # Row 2: CS: lo|hi label | non-negative chip
        row2 = QHBoxLayout()
        self._cs_label = QLabel("CS: —|—")
        row2.addWidget(self._cs_label)
        self.nonneg_btn = chip_button("non-negative", checkable=True)
        self.nonneg_btn.toggled.connect(self.sig_changed)
        row2.addWidget(self.nonneg_btn)
        row2.addStretch()
        layout.addLayout(row2)

        # Row 3: Baseline radio | test window chip | Adaptive button
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Baseline:"))
        self._baseline_group = QButtonGroup(self)
        self._baseline_rbs: dict[str, QRadioButton] = {}
        for val, lbl in [('conv','Conv'),('tailed','Tailed'),
                         ('global','Global'),('jitter','Jitter')]:
            rb = QRadioButton(lbl)
            if val == 'conv':
                rb.setChecked(True)
            self._baseline_group.addButton(rb)
            self._baseline_rbs[val] = rb
            row3.addWidget(rb)

        _vsep = QFrame(); _vsep.setFrameShape(QFrame.Shape.VLine)
        row3.addWidget(_vsep)

        self.test_window_btn = chip_button("Test window", checkable=True, checked=True)
        self.test_window_btn.toggled.connect(self.sig_changed)
        row3.addWidget(self.test_window_btn)

        row3.addStretch()
        layout.addLayout(row3)
        self._baseline_group.buttonClicked.connect(self._on_baseline_clicked)

        # Row 4: p / p-corrected chips (shown when Conv or Jitter)
        row4 = QHBoxLayout()
        self.p_btn  = chip_button("p",           checkable=True, checked=True)
        self.pc_btn = chip_button("p-corrected", checkable=True, checked=True)
        self.p_btn.toggled.connect(self.sig_changed)
        self.pc_btn.toggled.connect(self.sig_changed)
        row4.addWidget(self.p_btn)
        row4.addWidget(self.pc_btn)
        row4.addStretch()
        layout.addLayout(row4)

        # Explanation label (updates on baseline change)
        self._explanation = QLabel(_BASELINE_EXPLANATIONS['conv'])
        self._explanation.setStyleSheet(f"color: #666; font-size: {small_font_pt()}pt;")
        self._explanation.setWordWrap(True)
        layout.addWidget(self._explanation)

        self._update_pval_row_visibility('conv')

    def _on_cs_overlay_changed(self, active: bool):
        self.cs_show_check.blockSignals(True)
        self.cs_show_check.setChecked(active)
        self.cs_show_check.blockSignals(False)

    def _on_cs_params_changed(self, baseline_method: str, cs_metric: str):
        for val, rb in self._baseline_rbs.items():
            rb.blockSignals(True)
            rb.setChecked(val == baseline_method)
            rb.blockSignals(False)
        for btn in self._metric_group.buttons():
            btn.blockSignals(True)
            btn.setChecked(btn.text() == cs_metric)
            btn.blockSignals(False)
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

    def set_jitter_baseline_enabled(self, enabled: bool):
        rb = self._baseline_rbs.get('jitter')
        if rb is not None:
            rb.setEnabled(enabled)
            if not enabled and rb.isChecked():
                self._baseline_rbs['conv'].setChecked(True)
                self._on_baseline_clicked(self._baseline_rbs['conv'])

    # baseline_method and cs_metric live in nav.baseline_method / nav.cs_metric

    def set_cs(self, lo_val, hi_val):
        """Format lo|hi connection-strength values into the CS label."""
        def _fmt(v):
            if v is None: return "—"
            try: return f"{float(v):.3f}"
            except Exception: return "—"
        self._cs_label.setText(f"CS: {_fmt(lo_val)}|{_fmt(hi_val)}")

    def update_display(self):
        """Recompute and display lo|hi CS values for the current pair."""
        nav      = self.nav
        inds     = nav.current_pair_inds
        if inds is None:
            self.set_cs(None, None)
            return
        ref, tgt = int(inds[0]), int(inds[1])
        seg_idx  = nav.segment_index(nav.current_segment)

        def _cs_for(resolution: str) -> float | None:
            return CCGContextBuilder._cs_value(
                nav, self.jitter_mgr, seg_idx, ref, tgt, resolution,
                nonneg=self.nonneg_btn.isChecked())

        # Highres is always loaded at startup, so always show both resolutions
        # (hi renders "—" only when that session genuinely lacks highres data).
        self.set_cs(_cs_for('lowres'), _cs_for('highres'))


class JitterSection(CollapsibleSection):
    # Backend: JitterManager (neuropy/analyses/jitter.py)
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

    def _build(self):
        layout = QHBoxLayout()
        self.body_layout.addLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(QLabel("n="))
        self._n_spin = self.make_spin((10, 20, 50, 100, 200, 500, 1000), "100")
        layout.addWidget(self._n_spin)

        _btn_outline = ("QPushButton { border: 1px solid palette(mid); border-radius: 3px; "
                        "padding: 2px 8px; } "
                        "QPushButton:hover { border-color: palette(highlight); }")
        self._run_btn = QPushButton("Run Jitter")
        self._run_btn.setStyleSheet(_btn_outline)
        self._run_btn.clicked.connect(self._run)
        layout.addWidget(self._run_btn)

        for label, slot in [("Clear", self._clear), ("Save", self._save)]:
            b = QPushButton(label); b.setStyleSheet(_btn_outline)
            b.clicked.connect(slot); layout.addWidget(b)

        layout.addWidget(QLabel("Resolution:"))
        self.lo_btn = chip_button("lo", checkable=True, checked=True)
        self.hi_btn = chip_button("hi", checkable=True, checked=False)
        layout.addWidget(self.lo_btn)
        layout.addWidget(self.hi_btn)
        layout.addStretch()

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
        layout = QHBoxLayout()
        self.body_layout.addLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.enable_btn = chip_button("Enable", checkable=True)
        self.enable_btn.toggled.connect(self._on_enable)
        layout.addWidget(self.enable_btn)

        layout.addWidget(QLabel("Bin:"))
        self._bin_input = QLineEdit("0")
        self._bin_input.setFixedWidth(50)
        self._bin_input.setEnabled(False)
        layout.addWidget(self._bin_input)

        self._unit_combo = QComboBox()
        self._unit_combo.addItem("ms", "ms")
        self._unit_combo.addItem("#",  "#")
        self._unit_combo.setFixedWidth(45)
        self._unit_combo.setEnabled(False)
        self._unit_combo.setToolTip(
            "ms: lag in milliseconds\n"
            "#: ±i-th bin relative to 0 ms bin")
        layout.addWidget(self._unit_combo)

        self._set_btn = QPushButton("Set")
        self._set_btn.setEnabled(False)
        self._set_btn.clicked.connect(self._on_set)
        layout.addWidget(self._set_btn)

        layout.addStretch()

        self._bin_input.returnPressed.connect(self._on_set)

    def _on_enable(self, checked: bool):
        self._bin_input.setEnabled(checked)
        self._unit_combo.setEnabled(checked)
        self._set_btn.setEnabled(checked)
        self.enable_toggled.emit(checked)

    def _on_set(self):
        if not self.enable_btn.isChecked():
            return
        try:
            val  = float(self._bin_input.text())
            unit = self._unit_combo.currentData()
            self.set_requested.emit(val, unit)
        except ValueError:
            p = self.window()
            lp = p.pairs_view.spike_pairs if p else None
            if lp is not None:
                lp._spike_pairs_count.set('Invalid bin')

    @property
    def is_enabled(self) -> bool:
        return self.enable_btn.isChecked()


class CCGContextBuilder:
    """Assemble a RenderContext from AppState + CorrelogramPanel sections.

    Entry points:
      build_context()        — normal CCG view (regular + custom segments)
      build_extend_context() — recomputed CCG at user-specified window/bin size

    All data slicing, pair resolution, and jitter lookup are extracted into
    private static helpers so each entry point reads as a straight narrative.

    wf_peak_ms / wf_peak_amp are always None — waveform data not yet available
    in CCGData; fields are reserved for future wiring.
    """

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _dark_mode(panel) -> bool:
        fn = panel._theme_fn
        if fn is not None:
            try:
                theme = fn()
                if theme is not None:
                    return theme.dark
            except Exception:
                pass
        return qt_dark_mode()

    @staticmethod
    def _resolve_pair(nav):
        """Return (ref, tgt, pair_key, sess_label), accounting for cross-session mode."""
        ref, tgt = int(nav.current_pair_inds[0]), int(nav.current_pair_inds[1])
        pair_key   = nav.key
        sess_label = str(nav.key.session or '')
        if nav.session_any_mode:
            hl  = nav.cross_session_handles or []
            idx = nav.current_pair_idx
            if idx < len(hl):
                pair_key   = hl[idx][0]
                ref, tgt   = int(hl[idx][1]), int(hl[idx][2])
                sess_label = str(pair_key.session or '')
        return ref, tgt, pair_key, sess_label

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
    def _firing_rates(nav, seg_idx, ref, tgt):
        """(fr_ref, fr_tgt) for a pair. Prefer per-segment rates for an appended
        window; fall back to the neurons object's whole-session firing_rate."""
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
        """Connection strength for one pair/segment at a given resolution.

        Fetches the raw CCG (+ null / firing rates / segment extent), resolves the
        cached lo-res jitter result, and hands the whole pipeline to
        ConnectionStrength.compute. Returns None when that resolution isn't loaded.
        """
        key = nav.get_complete_key().change(resolution=resolution, ref=ref, tgt=tgt)
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
            fr_ref, fr_tgt = CCGContextBuilder._firing_rates(nav, seg_idx, ref, tgt)
        return ConnectionStrength.compute(
            ccg_raw, null_raw, ref, tgt, nav.cd.ccg_for(key).conf,
            metric=metric, method=method, active_norms=nav.active_norms,
            neurons=nav.neurons,
            custom_time_hours=CCGContextBuilder._time_hours_for_seg(nav, seg_idx),
            fr_ref=fr_ref, fr_tgt=fr_tgt, j_avg=j_avg, nonneg=nonneg,
            excitability=nav.key.excitability)

    @staticmethod
    def _neuron_meta(neurons, ref: int, tgt: int):
        """Return (nt_ref, nt_tgt, sh_ref, sh_tgt) from the pair's neurons object.

        Caller passes the pair's own session neurons (resolved via pair_key), so
        ref/tgt always index this array in range.
        """
        if neurons is None:
            return None, None, None, None
        types = neurons.neuron_type
        si    = neurons.shank_ids
        return str(types[ref]), str(types[tgt]), int(si[ref]), int(si[tgt])

    @staticmethod
    def _jitter_overlay(panel, nav, ref: int, tgt: int) -> JitterOverlay:
        """Look up cached jitter result for the current resolution."""
        jctrl = panel.jitter_mgr
        if jctrl is None:
            return JitterOverlay()
        res_key = 'hi' if nav.resolution in ("hi", "lo_hi") else 'lo'
        res = jctrl.get_result(ref, tgt, res_key)
        if res is None:
            return JitterOverlay()
        j_avg, j_pval, _j_bins = res
        return JitterOverlay(j_ccg=j_avg, j_pval=j_pval)

    @staticmethod
    def _bin_size(conf, n_bins: int) -> float:
        dur = conf.duration or 1.0
        return dur / (n_bins - 1) if n_bins > 1 else conf.bin_size

    @classmethod
    def _same_scale_ylim(cls, nav, panel, data, ref: int, tgt: int, neurons):
        """(0, ymax) shared across a pair's segments, or across every pair in the session."""
        mode = nav.same_scale_mode
        if mode is None or data.ccg is None:
            return None
        ck = (mode, frozenset(nav.active_norms), nav.resolution, str(nav.key),
              (ref, tgt) if mode == 'pair' else None)
        hit = panel._same_scale_cache.get(ck)
        if hit is None:
            pairs = ([(ref, tgt)] if mode == 'pair'
                     else [(int(r), int(t)) for r, t in nav.all_pairs_np])
            top = max((float(np.nanmax(CCGNorm.apply(
                          data.ccg[seg, r, t, :], None, r, t, nav.active_norms,
                          neurons=neurons,
                          custom_time_hours=cls._time_hours_for_seg(nav, seg))[0]))
                       for r, t in pairs for seg in range(data.ccg.shape[0])), default=0.0)
            hit = (0.0, top * 1.1) if top > 0 else (0.0, 1.0)
            panel._same_scale_cache.put(ck, hit)
        return hit

    # ── public entry points ────────────────────────────────────────────

    @classmethod
    def build_context(cls, nav, panel,
                      seg_label=None, hi_res_override=None,
                      pair_override=None) -> 'RenderContext | None':
        """Build RenderContext for the current pair/segment.

        seg_label:       segment to render; None = nav.current_segment
        hi_res_override: True=hi, False=lo, None=follow nav.resolution
        pair_override:   (pair_key, ref, tgt) to render a specific pair instead of the
                         current one — used to overlay "Show Together" pinned pairs.
        """
        if pair_override is not None:
            pair_key, ref, tgt = pair_override[0], int(pair_override[1]), int(pair_override[2])
            sess_label = str(pair_key.session or '')
        else:
            if nav.current_pair_inds is None:
                return None
            ref, tgt, pair_key, sess_label = cls._resolve_pair(nav)
        data_key, data = cls._resolve_data(nav, pair_key, hi_res_override)
        if data is None:
            return None

        seg_label = seg_label or nav.current_segment
        seg_idx = nav.cd.segment_index(data_key, seg_label)   # lazy-loads the label into this resolution
        arr = data.ccg                                        # after load: dim0 may have grown
        if arr is None:
            return None

        cor  = panel.corr_section
        cs   = panel.cs_section
        dark = cls._dark_mode(panel)
        conf = data.conf
        show_tw = cs.test_window_btn.isChecked()

        # Every segment (whole-session 'full' at 0, appended windows after) lives on dim0
        # of this one array — the pair's traces come from data.pair(); the ACG diagonals
        # (ref-ref, tgt-tgt) are different pairs, sliced directly.
        ccg, null, pval, pvc, _qval = data.pair(seg_idx, ref, tgt)
        if ccg is None or len(ccg) == 0:
            return None

        bsz = cls._bin_size(conf, len(ccg))
        dur = conf.duration or 1.0

        null  = null if cor.baseline_btn.show else None
        pval  = pval if cs.p_btn.isChecked() else None
        pvc   = pvc  if cs.pc_btn.isChecked() else None
        acg_r = arr[seg_idx, ref, ref, :] if cor.ref_btn.show else None
        acg_t = arr[seg_idx, tgt, tgt, :] if cor.tgt_btn.show else None

        _time_hrs = cls._time_hours_for_seg(nav, seg_idx)
        # Pair's own session neurons — correct in all-session mode where nav.neurons
        # is the current (possibly different) session.
        pair_neurons = nav.cd.nd.neurons_for(pair_key)
        ccg, null = CCGNorm.apply(
            ccg, null, ref, tgt, nav.active_norms,
            neurons=pair_neurons,
            custom_time_hours=_time_hrs)
        if acg_r is not None:
            acg_r, _ = CCGNorm.apply(acg_r, None, ref, ref,
                                     nav.active_norms - {NormalizeBy.BASELINE},
                                     neurons=pair_neurons,
                                     custom_time_hours=_time_hrs)
        if acg_t is not None:
            acg_t, _ = CCGNorm.apply(acg_t, None, tgt, tgt,
                                     nav.active_norms - {NormalizeBy.BASELINE},
                                     neurons=pair_neurons,
                                     custom_time_hours=_time_hrs)

        nt_ref, nt_tgt, sh_ref, sh_tgt = cls._neuron_meta(pair_neurons, ref, tgt)
        seg_display = seg_label

        return cls._make_context(
            ccg=ccg, bsz=bsz, dur=dur, conf=conf, nav=nav,
            ref=ref, tgt=tgt, seg_display=seg_display, sess_label=sess_label,
            jitter=cls._jitter_overlay(panel, nav, ref, tgt), dark=dark,
            null=null, pval=pval, pval_corrected=pvc,
            acg_ref=acg_r, acg_tgt=acg_t,
            nt_ref=nt_ref, nt_tgt=nt_tgt, sh_ref=sh_ref, sh_tgt=sh_tgt,
            show_tw=show_tw, cor=cor, cs_overlay=nav.cs_overlay_active,
            is_significant=nav.is_significant(ref, tgt, seg_idx),
            ylim_override=cls._same_scale_ylim(nav, panel, data, ref, tgt, pair_neurons),
            cs_annotation_lines=(cls._cs_annotation_lines(nav, cs, ref, tgt, seg_idx)
                                 if nav.cs_overlay_active else []),
        )

    @classmethod
    def build_extend_context(cls, nav, panel, seg_label=None) -> 'RenderContext | None':
        """Recompute CCG at user-specified window + bin size (extend mode).

        seg_label: segment to extend; None = nav.current_segment.
        Result is cached by (key, ref, tgt, seg, extend_ms, bin_ms).
        """
        cor = panel.corr_section
        if not cor.extend_check.isChecked():
            return None
        if nav.current_pair_inds is None:
            return None
        view = nav.get_complete_key()
        ref, tgt = view.ref, view.tgt
        if ref == tgt:
            return None

        extend_ms     = max(5, cor.extend_ms)
        min_bin_ms    = _CCG_RESOLUTION['highres'] * 1000
        extend_bin_ms = max(cor.extend_bin_ms, min_bin_ms)
        dur, bs       = extend_ms / 1000.0, extend_bin_ms / 1000.0

        seg_label = seg_label or nav.current_segment
        cache_key = (str(view), seg_label, extend_ms, extend_bin_ms,
                     frozenset(nav.active_norms))
        hit = panel._extend_cache.get(cache_key)
        if hit is not None:
            return hit

        if nav.neurons is None:
            return None
        conf = (nav.ccg_data.conf if nav.ccg_data is not None else nav.cd.conf)
        ccg_slice = cls._compute_extend_ccg(nav, ref, tgt, dur, bs, conf, seg_label)
        if ccg_slice is None:
            return None
        ccg_slice, _ = CCGNorm.apply(ccg_slice, None, ref, tgt,
                                     nav.active_norms - {NormalizeBy.BASELINE},
                                     neurons=nav.neurons,
                                     custom_time_hours=cls._time_hours_for_seg(
                                         nav, nav.segment_index(seg_label)))

        bsz  = dur / (len(ccg_slice) - 1) if len(ccg_slice) > 1 else bs
        dark = cls._dark_mode(panel)
        ctx = cls._make_context(
            ccg=ccg_slice, bsz=bsz, dur=dur, conf=conf, nav=nav,
            ref=ref, tgt=tgt,
            seg_display=f'{seg_label} (extend {extend_ms}ms @ {extend_bin_ms:.4f}ms/bin)',
            sess_label=str(nav.key.session or ''),
            jitter=JitterOverlay(), dark=dark,
            null=None, pval=None, pval_corrected=None,
            acg_ref=None, acg_tgt=None,
            nt_ref=None, nt_tgt=None, sh_ref=None, sh_tgt=None,
            show_tw=False, cor=cor, cs_overlay=False,
            is_significant=False,
        )
        panel._extend_cache.put(cache_key, ctx)
        return ctx

    # ── private builders ───────────────────────────────────────────────

    @staticmethod
    def _make_context(*, ccg, bsz, dur, conf, nav, ref, tgt,
                       seg_display, sess_label, jitter, dark,
                       null, pval, pval_corrected,
                       acg_ref, acg_tgt,
                       nt_ref, nt_tgt, sh_ref, sh_tgt,
                       show_tw, cor, cs_overlay, is_significant,
                       cs_annotation_lines=None, ylim_override=None) -> RenderContext:
        min_lag = conf.min_lag if show_tw else None
        max_lag = conf.max_lag if show_tw else None
        if null is not None:
            null = ConnectionStrength.baseline(ccg, null, conf, nav.baseline_method)
        _ccg_top  = float(np.nanmax(ccg))  if len(ccg)  else 0.0
        _null_top = float(np.nanmax(null)) if null is not None and len(null) else 0.0
        _ylim_top = max(_ccg_top, _null_top) * 1.05
        _ylim = ylim_override or ((0.0, _ylim_top) if _ylim_top > 0 else None)
        return RenderContext(
            ccg=ccg, bin_size_eff=bsz, window_size_eff=dur,
            alpha=nav.active_sig_threshold, seg_id_display=seg_display,
            inds=(ref, tgt), jitter=jitter, dark_mode=dark,
            title=TitleConfig(
                title_session_label=sess_label,
                title_show_session=True, title_show_type=True,
                title_show_inds=True, title_show_seg=True,
                title_show_shanks=(sh_ref is not None),
            ),
            style=PlotStyle(),
            # CCG data
            ccg_null_plot=null,
            pval=pval,
            pval_corrected=pval_corrected,
            # ACG overlays (None when not shown)
            acg_ref=acg_ref, acg_tgt=acg_tgt,
            # Waveforms: not yet wired — data not available in CCGData
            wf_peak_ms=None, wf_peak_amp=None,
            cs_baseline_arg=null if (null is not None and cs_overlay) else None,
            norm_info=None,
            extend_on=cor.extend_check.isChecked(),
            cs_annotation_lines=cs_annotation_lines or [],
            min_lag_plot=min_lag, max_lag_plot=max_lag,
            neuron_type=(nt_ref, nt_tgt) if (nt_ref or nt_tgt) else None,
            shank_ids=(sh_ref, sh_tgt) if sh_ref is not None else None,
            # Display style from CorrelogramSection
            show_ccg=cor.ccg_btn.show, line_ccg=cor.ccg_btn.line,
            line_baseline=cor.baseline_btn.line,
            line_ref=cor.ref_btn.line if acg_ref is not None else False,
            line_tgt=cor.tgt_btn.line if acg_tgt is not None else False,
            line_jitter=cor.jitter_line_btn.line,
            acg_yscale_ref=cor.acg_yscale_ref, acg_yscale_tgt=cor.acg_yscale_tgt,
            acg_match_ccg=cor.autoscale_btn.isChecked(),
            ylim=_ylim,
            is_significant_pair=is_significant,
        )

    @staticmethod
    def _cs_annotation_lines(nav, cs_section: 'BaselineCSSection',
                              ref: int, tgt: int, seg_idx: int) -> list:
        """Return formatted CS annotation strings for PNG export."""
        metric = nav.cs_metric
        method = nav.baseline_method
        nonneg = cs_section.nonneg_btn.isChecked()

        lo = CCGContextBuilder._cs_value(nav, cs_section.jitter_mgr, seg_idx,
                                         ref, tgt, 'lowres', nonneg=nonneg)
        hi = CCGContextBuilder._cs_value(nav, cs_section.jitter_mgr, seg_idx,
                                         ref, tgt, 'highres', nonneg=nonneg)

        lines = []
        if lo is not None:
            lines.append(f"{metric} ({method}) lo-res: {lo:.4f}")
        if hi is not None:
            lines.append(f"{metric} ({method}) hi-res: {hi:.4f}")
        return lines

    @staticmethod
    def _compute_extend_ccg(nav, ref: int, tgt: int, dur: float, bs: float, conf, seg_label: str):
        """Recompute CCG for ref/tgt at given window/bin. Returns 1-D array or None."""
        neurons = nav.neurons
        neurons_sub = neurons.neuron_slice(neuron_inds=np.array([ref, tgt]))
        # An appended window carries its own extent (source config); 'full' spans the session.
        src = nav.cd.source_config(nav.key, seg_label) if seg_label else None
        kwargs  = dict(
            bin_size=bs, window_size=dur,
            symmetrize=conf.symmetrize_ccg,
            use_acceleration=conf.use_acceleration,
        )
        try:
            if src is not None and not isinstance(src.t0, str) and not isinstance(src.t1, str):
                full = correlations.spike_correlations(
                    neurons_sub, neuron_inds=np.array([0, 1]),
                    start_end_times=np.array([[float(src.t0)], [float(src.t1)]]), **kwargs)
                slc = full[0, 0, 1, :]
            else:
                full = correlations.spike_correlations(
                    neurons_sub, ref_neuron_inds=np.array([0]),
                    neuron_inds=np.array([1]), **kwargs)
                slc = full[0, 0, :]
            slc = np.asarray(slc, dtype=float)
            return slc if slc.size > 0 else None
        except Exception as exc:
            print(f"[CCGPanel] extend compute failed: {exc}", flush=True)
            return None


class CCGPlotWidget(QWidget):
    """pyqtgraph-based CCG plot: bars + baseline + ACG + test window + p-value.

    Rendering is driven by RenderContext from CCGContextBuilder.build_context().
    """

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

        self._row_layouts:  list = []   # list[QHBoxLayout], one per stacked row
        self._plot_widgets: list = []   # list[pg.PlotWidget], flat
        self._plots:        list = []   # list[pg.PlotItem], flat
        self._pval_vbs:     list = []   # list[pg.ViewBox], flat
        self._pval_items_per: list = [] # list[list], flat
        self._acgs:         list = []   # list[[(vb, axis) ref, (vb, axis) tgt]], flat

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
        # Remove old rows/widgets
        for pw in self._plot_widgets:
            pw.setParent(None)
            pw.deleteLater()
        for row_layout in self._row_layouts:
            self._plot_grid.removeItem(row_layout)
        self._row_layouts.clear()
        self._plot_widgets.clear()
        self._plots.clear()
        self._pval_vbs.clear()
        self._pval_items_per.clear()
        self._acgs.clear()

        for row_len in row_lengths:
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(2)
            for _ in range(row_len):
                pw = pg.PlotWidget()
                pw.setMinimumSize(280, self._MIN_PLOT_H)
                pw.setBackground('w')
                pw.showGrid(x=False, y=True, alpha=0.3)
                pw.setMouseEnabled(x=False, y=False)
                pw.getViewBox().setMouseEnabled(x=False, y=False)
                pw.scene().sigMouseClicked.connect(self._on_mouse_click)
                p = pw.getPlotItem()
                vb = pg.ViewBox()
                vb.setMouseEnabled(x=False, y=False)   # no pinch/wheel/drag zoom on p-val overlay
                vb.setMenuEnabled(False)
                pw.scene().addItem(vb)
                # ACG ref/tgt each get their own right-hand y-axis, stepped outward
                acg = []
                p.layout.setColumnMinimumWidth(3, 14)   # gap: keeps the outer axis clear of the inner label
                for col, color, name in ((2, ACG_REF_COLOR, 'ACG ref'),
                                         (4, ACG_TGT_COLOR, 'ACG tgt')):
                    avb, ax = pg.ViewBox(), pg.AxisItem('right')
                    avb.setMouseEnabled(x=False, y=False)
                    avb.setMenuEnabled(False)
                    avb.setXLink(p.vb)
                    pw.scene().addItem(avb)
                    ax.linkToView(avb)
                    ax.setPen(pg.mkPen(color)); ax.setTextPen(pg.mkPen(color))
                    ax.setLabel(name, color=color)
                    p.layout.addItem(ax, 2, col); ax.hide()
                    acg.append((avb, ax))
                _guard = [False]
                def _sync_geom(vb=vb, p=p, g=_guard, acg=acg):
                    if g[0]: return
                    g[0] = True
                    try:
                        rect = p.vb.sceneBoundingRect()
                        for v in (vb, acg[0][0], acg[1][0]):
                            v.setGeometry(rect)
                    finally: g[0] = False
                p.vb.sigResized.connect(_sync_geom)
                row_layout.addWidget(pw)
                self._plot_widgets.append(pw)
                self._plots.append(p)
                self._pval_vbs.append(vb)
                self._pval_items_per.append([])
                self._acgs.append(acg)
            self._plot_grid.addLayout(row_layout)
            self._row_layouts.append(row_layout)

    def _on_mouse_click(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.context_menu_requested.emit(event.screenPos().toPoint())

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
        valid = [c for row in rows for c in row]
        dark = valid[0].dark_mode if valid else qt_dark_mode()
        plot_bg = '#1e1e1e' if dark else 'w'
        for pw in self._plot_widgets:
            pw.setBackground(plot_bg)
        for i, ctx in enumerate(valid):
            self._render_one(self._plots[i], self._pval_vbs[i],
                             self._pval_items_per[i], ctx, self._acgs[i])
        for i in range(len(valid), len(self._plots)):
            self._render_one(self._plots[i], self._pval_vbs[i],
                             self._pval_items_per[i], None, self._acgs[i])
        vp = self._plot_scroll.viewport()
        max_row_len = max(row_lengths)
        plot_w = max(320 * max_row_len, vp.width())
        plot_h = max(self._MIN_PLOT_H * len(row_lengths), vp.height())
        self._plot_container.setMinimumSize(plot_w, plot_h)

    @staticmethod
    def _apply_plot_chrome(p, dark: bool) -> None:
        fg = '#dddddd' if dark else '#333333'
        muted = '#888888' if dark else '#666666'
        p.showGrid(x=False, y=True, alpha=0.25 if dark else 0.3)
        for axis_name in ('bottom', 'left'):
            axis = p.getAxis(axis_name)
            if axis is not None:
                axis.setPen(pg.mkPen(muted))
                axis.setTextPen(pg.mkPen(fg))
        try:
            p.titleLabel.item.setAttr('color', fg)
        except Exception:
            pass

    def _render_one(self, p, pval_vb, pval_items, ctx, acg_axes) -> None:
        p.clear()
        for avb, ax in acg_axes:
            avb.clear()
            ax.hide()
        if ctx is None:
            return
        self._apply_plot_chrome(p, ctx.dark_mode)
        ccg = np.asarray(ctx.ccg, dtype=float)
        n   = len(ccg)
        if n == 0:
            return
        bs    = ctx.bin_size_eff * 1000.0
        ws    = ctx.window_size_eff * 1000.0
        xs    = np.linspace(-ws / 2, ws / 2, n)
        x_edges = np.linspace(-ws / 2 - bs / 2, ws / 2 + bs / 2, n + 1)
        width = bs

        # CCG bars
        if ctx.show_ccg:
            color = '#4a7fd4' if not ctx.dark_mode else '#7aafff'
            if ctx.line_ccg:
                item = pg.PlotDataItem(x_edges, ccg, stepMode='center',
                                       pen=_pen(color), fillLevel=None)
                p.addItem(item)
            else:
                p.addItem(pg.BarGraphItem(x=xs, height=ccg, width=width,
                                          brush=color, pen=None))

        # Baseline
        if ctx.ccg_null_plot is not None:
            null = np.asarray(ctx.ccg_null_plot, dtype=float)
            color = '#e88' if not ctx.dark_mode else '#cc6666'
            if ctx.line_baseline:
                item = pg.PlotDataItem(x_edges[:len(null)+1], null, stepMode='center',
                                       pen=_pen(color, Qt.PenStyle.DashLine))
                p.addItem(item)
            else:
                p.addItem(pg.BarGraphItem(x=xs[:len(null)], height=null,
                                          width=width, brush=pg.mkBrush(color + '88'),
                                          pen=None))

        # Connection strength overlay (CCG excess above baseline within test window)
        if (ctx.cs_baseline_arg is not None
                and ctx.min_lag_plot is not None and ctx.max_lag_plot is not None):
            bl = np.asarray(ctx.cs_baseline_arg, dtype=float)
            if len(bl) == len(ccg):
                mask = test_window_bin_mask(
                    xs, ctx.min_lag_plot, ctx.max_lag_plot, width)
                bottoms = bl[mask]
                heights = np.maximum(ccg[mask] - bottoms, 0)
                cs_color = '#3ecf6e' if ctx.dark_mode else '#1a6b2e'
                p.addItem(pg.BarGraphItem(
                    x=xs[mask], height=heights, y0=bottoms, width=width,
                    brush=pg.mkBrush(cs_color), pen=None))

        # Test window (span matches CS bin selection geometry)
        if ctx.min_lag_plot is not None and ctx.max_lag_plot is not None:
            span_lo, span_hi = test_window_span_ms(
                ctx.min_lag_plot, ctx.max_lag_plot, width)
            tw_brush = (80, 100, 140, 60) if ctx.dark_mode else (200, 220, 255, 60)
            p.addItem(pg.LinearRegionItem(
                values=[span_lo, span_hi],
                brush=pg.mkBrush(*tw_brush),
                pen=pg.mkPen(None), movable=False))

        # ACGs — each on its own right-hand axis so overlays stay readable
        for (avb, ax), color, data, scale, as_line in (
                (acg_axes[0], ACG_REF_COLOR, ctx.acg_ref, ctx.acg_yscale_ref, ctx.line_ref),
                (acg_axes[1], ACG_TGT_COLOR, ctx.acg_tgt, ctx.acg_yscale_tgt, ctx.line_tgt)):
            if data is None:
                continue
            a  = np.asarray(data, dtype=float)
            na = len(a)
            if as_line:
                avb.addItem(pg.PlotDataItem(x_edges[:na+1], a, stepMode='center',
                                            pen=_pen(color)))
            else:
                avb.addItem(pg.BarGraphItem(x=xs[:na], height=a, width=width,
                                            brush=pg.mkBrush(color + '66'), pen=None))
            top = float(np.nanmax(a)) if na else 0.0
            top = top * 1.1 if top > 0 else 1.0
            avb.setYRange(0, top if ctx.acg_match_ccg else top / max(scale, 0.01),
                          padding=0)
            avb.setGeometry(p.vb.sceneBoundingRect())
            ax.show()

        # Jitter overlay
        j = ctx.jitter
        if j.j_ccg is not None:
            jx = xs[:len(j.j_ccg)]
            if j.j_ccg_lo is not None and j.j_ccg_hi is not None:
                lo_c = pg.PlotDataItem(jx, j.j_ccg_lo, pen=None)
                hi_c = pg.PlotDataItem(jx, j.j_ccg_hi, pen=None)
                p.addItem(lo_c); p.addItem(hi_c)
                p.addItem(pg.FillBetweenItem(lo_c, hi_c,
                                             brush=pg.mkBrush(180, 160, 210, 100)))
            p.plot(jx, j.j_ccg,
                   pen=_pen('#9b59b6', (Qt.PenStyle.DashLine if ctx.line_jitter
                                        else Qt.PenStyle.SolidLine)))
            if j.j_pval is not None:
                p.plot(jx, j.j_pval,
                       pen=_pen('#6c3483', Qt.PenStyle.DotLine))

        # P-value lines on right axis
        for item in pval_items:
            pval_vb.removeItem(item)
        pval_items.clear()
        pval_vb.setGeometry(p.vb.sceneBoundingRect())

        has_pval = ctx.pval is not None or ctx.pval_corrected is not None
        # right axis not used (no linkToView — avoids pyqtgraph recursion bug)
        if has_pval:
            pval_vb.setYRange(0, 1, padding=0)
            if ctx.pval is not None:
                c = pg.PlotDataItem(xs[:len(ctx.pval)], ctx.pval,
                                    pen=_pen('#e74c3c', Qt.PenStyle.DotLine))
                pval_vb.addItem(c); pval_items.append(c)
            if ctx.pval_corrected is not None:
                c = pg.PlotDataItem(xs[:len(ctx.pval_corrected)], ctx.pval_corrected,
                                    pen=_pen('#922b21', Qt.PenStyle.DashLine))
                pval_vb.addItem(c); pval_items.append(c)
            inf = pg.InfiniteLine(pos=ctx.alpha, angle=0,
                                  pen=_pen('#e74c3c', Qt.PenStyle.DotLine))
            pval_vb.addItem(inf); pval_items.append(inf)

        if ctx.ylim is not None:
            p.setYRange(*ctx.ylim, padding=0.05)   # headroom so overlays aren't clipped at the top edge
        else:
            p.enableAutoRange(axis='xy', enable=True)
            p.autoRange()

        p.setTitle(self._make_title(ctx), size='9pt')
        p.setLabel('bottom', 'Lag (ms)')

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

    def render(self, neurons, ref: int, tgt: int):
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
        ref_wf    = waveforms[ref]
        tgt_wf    = waveforms[tgt]
        ref_shank = int(shank_ids[ref]) if shank_ids is not None else 0
        tgt_shank = int(shank_ids[tgt]) if shank_ids is not None else 0
        ax = self._fig.add_subplot(111)
        try:
            from neuropy.plotting.probe import plot_waveform_on_channel
            plot_waveform_on_channel(ref_wf, ref_shank, tgt_wf, tgt_shank,
                                     ax=ax, ch_per_shank=16)
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
        self._extend_cache: LRUCache = LRUCache(8)
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

        # Top: CCG plot + optional waveform panel side by side
        plot_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.plot_widget = CCGPlotWidget()
        plot_splitter.addWidget(self.plot_widget)
        self._wf_panel = WaveformPanelQt()
        self._wf_panel.setVisible(False)
        plot_splitter.addWidget(self._wf_panel)
        plot_splitter.setSizes([700, 300])
        # Vertical splitter: CCG plot (top) | toolbox (bottom, draggable)
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.addWidget(plot_splitter)

        # Bottom: scrollable toolbox
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        toolbox = QWidget()
        tb_layout = QVBoxLayout(toolbox)
        tb_layout.setContentsMargins(2, 2, 2, 2)
        tb_layout.setSpacing(3)

        self.seg_bar      = SegmentBar(self.nav)
        self.norm_section = NormSection(self.nav)
        self.corr_section = CorrelogramSection()
        self.cs_section   = BaselineCSSection(self.nav)
        self.jitter_section = JitterSection(self.nav)
        self.sa_section   = SpikeAttributionSection()

        def _sep():
            f = QFrame(); f.setFrameShape(QFrame.Shape.HLine)
            f.setFrameShadow(QFrame.Shadow.Sunken); f.setFixedHeight(1)
            return f
        for w in (self.seg_bar, self.norm_section, self.corr_section,
                  self.cs_section, self.jitter_section, self.sa_section):
            tb_layout.addWidget(_sep())
            tb_layout.addWidget(w)
        tb_layout.addStretch()

        scroll.setWidget(toolbox)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        v_splitter.addWidget(scroll)
        v_splitter.setSizes([600, 250])
        v_splitter.setStretchFactor(0, 3)
        v_splitter.setStretchFactor(1, 1)
        for i in range(v_splitter.count()):
            v_splitter.setCollapsible(i, False)
        root.addWidget(v_splitter, stretch=1)

        # Context menu
        self.plot_widget.context_menu_requested.connect(self._show_context_menu)

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
        segs = list(nav.stacked_segments) or [nav.current_segment]
        # Row axis = view kind (lo / hi / extend); column axis = segment. Transposed: swap.
        builders = ([lambda s, hi=hi: CCGContextBuilder.build_context(nav, self, seg_label=s, hi_res_override=hi)
                     for hi in ([False, True] if nav.resolution == "lo_hi" else [None])]
                    + ([lambda s: CCGContextBuilder.build_extend_context(nav, self, seg_label=s)]
                       if cor.extend_check.isChecked() else []))
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
                ref, tgt, pair_key, _ = CCGContextBuilder._resolve_pair(nav)
                self._wf_panel.render(nav.cd.nd.neurons_for(pair_key), ref, tgt)

    def _connect_nav(self):
        nav = self.nav
        for sig in (nav.key_changed, nav.pair_changed, nav.segment_changed,
                    nav.resolution_changed, nav.norms_changed,
                    nav.stacked_segments_changed, nav.cs_overlay_changed,
                    nav.sig_threshold_changed, nav.scale_mode_changed):
            sig.connect(lambda _: self.plot_update_requested.emit())
        nav.cs_params_changed.connect(lambda *_: self.plot_update_requested.emit())
        nav.pair_changed.connect(self._update_jitter_baseline_state)
        self.plot_update_requested.connect(self.request_render)

    def _connect_sections(self):
        nav = self.nav

        self.norm_section.apply_requested.connect(self._on_apply_norms)
        self.cs_section.sig_changed.connect(lambda: self.plot_update_requested.emit())
        self.corr_section.style_changed.connect(self.plot_update_requested)
        self.corr_section.ref_wf_btn.toggled.connect(self._wf_panel.setVisible)
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
        self._on_spike_attr_set(
            float(self.sa_section._bin_input.text()),
            self.sa_section._unit_combo.currentData())

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
        for label, key in [
            ("CCG",           'ccg'),
            ("Reference ACG", 'acg_ref'),
            ("Target ACG",    'acg_tgt'),
            ("Baseline",      'baseline'),
            ("P-values",      'pval'),
        ]:
            view_menu.addAction(label, lambda k=key: self._view_values(k))
        menu.exec(pos)

    def _export_png(self):
        ctx = CCGContextBuilder.build_context(self.nav, self)
        if ctx is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", "", "PNG files (*.png)")
        if not path:
            return
        try:
            render_ccg_png(ctx, path)
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _view_values(self, item: str):
        ctx = CCGContextBuilder.build_context(self.nav, self)
        if ctx is None:
            return
        val = ctx.ccg if item == 'ccg' else ctx.acg_ref if item == 'acg_ref' else ctx.acg_tgt if item == 'acg_tgt' else ctx.ccg_null_plot if item == 'baseline' else ctx.pval if item == 'pval' else None
        if val is None:
            print(f"[CCG] {item}: None")
        else:
            import numpy as np
            print(f"[CCG] {item}: shape={np.asarray(val).shape}\n{val}")
