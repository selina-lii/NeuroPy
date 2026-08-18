"""CCGReviewUI — Qt root window for CCG Manual Review."""
from __future__ import annotations
import os
import json

from pathlib import Path as _Path
from typing import TYPE_CHECKING
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.Qt.QtCore import Qt, QTimer, QThread, QObject, Signal
from pyqtgraph.Qt.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QStatusBar, QLabel, QApplication,
    QProgressDialog, QMessageBox, QFileDialog, QTabWidget, QPushButton, QScrollArea,
)
from pyqtgraph.Qt.QtGui import QKeySequence, QCloseEvent, QAction, QShortcut
from neuropy.analyses.neurons_dataset import Key
from neuropy.analyses.utils import JsonSavable
from neuropy.ui.app_state import AppState, _ALL_SEGS
from neuropy.ui.pair_selection_panel import PairSelectionPanelContainer
from neuropy.ui.ui_common import UITheme, qt_dark_mode
from neuropy.ui.utils import GroupHotkeysBar, Tunable
from neuropy.ui.dialogs import (
    QuickSaveDialog, ManageGroupsDialog, CustomCCGManageDialog,
)
from neuropy.ui.stats_tests import StatsTestPanel
from neuropy.ui.jitter_ui import JitterManager
from neuropy.ui.time_slider import CustomCCGManager, TimeSliderPanel
from neuropy.ui.menubar import ReviewMenuBar, IndexBar
from neuropy.ui.ccg_panel import CorrelogramPanel
from neuropy.ui.neuron_network import NetworkPanel
from neuropy.analyses.ms_connectivity import CCGDataset, ProjectConfig, open_project
from neuropy.ui.all_session_mode import AllSessionMode
from neuropy.ui.classifier_ui import ClassifierDialog
from neuropy.analyses.spike_attribution import compute_spike_pairs

if TYPE_CHECKING:
    pass


class _CCGLoadWorker(QObject):
    """Runs one blocking cd.get_ccg off the Qt main thread (numpy/file IO releases the GIL,
    so the event loop stays responsive). Emits done/error back to the main thread."""
    done  = Signal()
    error = Signal(str)

    def __init__(self, cd, nd_key, resolution):
        super().__init__()
        self._cd = cd
        self._nd_key = nd_key
        self._res = resolution

    def run(self):
        try:
            self._cd.get_ccg(self._nd_key.change(resolution=self._res))
        except Exception as e:
            self.error.emit(str(e))
            return
        self.done.emit()


class UISettings(JsonSavable):
    """Persisted UI/Settings-dialog values."""

    min_font_size = Tunable(12)
    save_ui_on_close = Tunable(True)
    autosave_sel_on = Tunable(True)
    autosave_sel_interval = Tunable(1)        # number + unit kept as two scalars,
    autosave_sel_unit = Tunable('hour')       # so neither can drift into a bad shape
    autosave_grp_on = Tunable(True)
    autosave_grp_interval = Tunable(1)
    autosave_grp_unit = Tunable('hour')
    # Fewest tagged pairs a label needs before the classifier will train on it.
    classifier_min_count = Tunable(20)

    def __init__(self, **kwargs):
        super().__init__()
        Tunable.apply_defaults(self)
        self.list_cursor_follows_action: bool = kwargs.pop('list_cursor_follows_action', True)
        self.ccg_memory_limit_gb: float = kwargs.pop('ccg_memory_limit_gb', 4.0)
        for name, value in kwargs.items():
            setattr(self, name, value)


class UIStates(JsonSavable):
    """The single persisted UI snapshot for CCGReviewUI. Global."""

    def save_path(self) -> str:
        return str(_Path(__file__).resolve().parents[2] / 'data' / 'ui_state_qt')

    def __init__(self):
        super().__init__()
        self.settings = UISettings()
        self.session = ''
        self.type_label = ''
        self.session_any_mode = False
        self.resolution = 'lo'
        self.current_segment = _ALL_SEGS
        self.splitter_sizes: list = []
        self.panel_sizes: dict = {}       # panel attr -> pre-collapse width
        self.collapsed_panels: list = []  # panel attrs currently hidden
        self.panel_state: dict = {}       # PairSelectionPanelContainer state

    def __setstate__(self, state: dict) -> None:
        """settings is one nested object, not a dict of them — rebuild it here."""
        settings = dict(state.pop('settings', {}))
        super().__setstate__(state)
        self.settings = UISettings(**settings)


class BottomStatusBar:
    """Owns the stats QLabel in the window status bar. Refreshes on nav signals."""

    def __init__(self, nav: 'AppState', status_bar):
        self.nav = nav
        self.label = QLabel("")
        status_bar.addWidget(self.label)
        self._pair_info_label = QLabel("")
        status_bar.addPermanentWidget(self._pair_info_label)
        nav.pair_changed.connect(self.refresh)
        nav.key_changed.connect(self.refresh)
        nav.selection_changed.connect(self.refresh)

    def refresh(self, *_):
        self.label.setText(self._str())

    @staticmethod
    def _counts_str(ref_t, tgt_t, n_ref, n_tgt) -> str:
        return (f"  {ref_t}: {n_ref}" if ref_t == tgt_t
                else f"  ref({ref_t}): {n_ref}  tgt({tgt_t}): {n_tgt}")
    
    def _sig_str(self, n_poss: int, n_sig: int, n_sel: int) -> str:
        if n_poss:
            return f"Significant: {n_sig}/{n_poss} Selected: {n_sel}/{n_poss}"
        return f"Significant: {n_sig}"

    def _frate_str(self, frates, ref, tgt) -> str:
        return f"  |  FR: ref={frates[ref]:.1f}Hz  tgt={frates[tgt]:.1f}Hz"

    def _str(self) -> str:
        nav = self.nav
        key = nav.key
        ref_t, tgt_t = key.conn_type
        
        n_sig = len(nav.all_pairs_np)
        n_sel = len(nav.active_selections.selected)

        if nav.session_any_mode:
            n_ref = n_tgt = n_poss = 0
            for nk in nav.real_nd_keys():
                a, b, p = nav.cd.nd.neurons_for(nk).pair_count(ref_t, tgt_t)
                n_ref += a
                n_tgt += b
                n_poss += p
        else:
            n_ref, n_tgt, n_poss = nav.neurons.pair_count(ref_t, tgt_t)

        ct_prefix = f"{key.type_label()} | "
        s = ct_prefix
        s += self._sig_str(n_poss, n_sig, n_sel)
        s += self._counts_str(ref_t, tgt_t, n_ref, n_tgt)

        inds = nav.current_pair_inds
        if inds is not None:
            ref, tgt = int(inds[0]), int(inds[1])
            try:
                s += self._frate_str(nav.neurons.firing_rate, ref, tgt)
            except Exception:
                pass
        return s

class CCGReviewUI(QMainWindow):
    """Qt root window for CCG Manual Review.

    Owns AppState. Panels are created in _build_layout() and receive
    nav as their primary interface.
    """
    nav: AppState

    def __init__(self, cd: 'CCGDataset', key=None):
        super().__init__()
        self._loading_thread = self._loading_worker = None   # in-flight lazy-load (see _ensure_loaded)
        self.ui_states = UIStates()   # one snapshot for everything
        try:
            self.ui_states.load()
        except Exception as e:
            pass   # first run — keep defaults
        
        if key is None:                               # restore last-used, else default
            s = self.ui_states
            key = cd.find(s.session, type_label=s.type_label, strict=False) or cd.find('')
        else:
            key = cd.find(key if isinstance(key, str) else key.session)

        # Bootstrap: nav (hence self.cd) doesn't exist yet, so probe the local cd directly.
        check = cd.ccg_for(key)
        if check is None or check.ccg is None or check.ccg.ndim < 4:
            cd.get_ccg(key)

        self.theme = UITheme.from_dark(qt_dark_mode())

        self.nav = AppState(cd, key)
        self.nav.root = self
        self._project_dir = _Path(cd.save_path).name   # the combo's idea of where we are
        # A ProcessData project's sessions come from the caller and cannot be re-read from its
        # header, so keep each project's nd to switch back to it.
        self._nd_by_project = {self._project_dir: cd.nd}

        self.prediction_store = None      # set by ClassifierDialog after a run
        self.classifier_dialog = None     # kept alive so it stays non-modal

        self.jitter_mgr = JitterManager(self.nav, cd)
        self.jitter_mgr.completed.connect(self._on_jitter_completed)
        self.jitter_mgr.failed.connect(self._on_jitter_failed)

        self.custom_mgr = CustomCCGManager(self)
        self.all_sess_mgr = AllSessionMode(self.nav, self.cd)
        self.nav.pair_changed.connect(self._on_pair_changed)
        self.nav.segment_changed.connect(self._on_segment_changed)
        self.nav.key_changed.connect(self._on_key_changed)
        self.nav.session_mode_changed.connect(self._on_session_mode_changed)
        self.nav.cross_session_handles_changed.connect(
            lambda: self.mainview.request_render() if self.nav.session_any_mode else None)
        self.nav.groups.changed.connect(self._on_groups_changed)

        self._build_layout()
        self._bind_shortcuts()

        QTimer.singleShot(100, self._initial_draw)

    @property
    def settings(self) -> UISettings:
        return self.ui_states.settings
    
    @property
    def cd(self) -> CCGDataset:
        return self.nav.cd

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_theme(redraw=True)

    def _refresh_theme(self, *, redraw: bool = False):
        self.theme = UITheme.from_dark(qt_dark_mode())
        self.hotkeys_bar.refresh()
        self.stats_panel.apply_theme()
        if redraw:
            self.mainview.request_render()
            self.neuron_network.draw()

    def _build_layout(self):
        self._apply_min_font_size(self.settings.min_font_size)
        self.setWindowTitle("CCG Manual Review")
        self.resize(1800, 950)

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(2)

        self.index_bar = IndexBar(self)
        self.index_bar.attach(root_layout)

        self.hotkeys_bar = GroupHotkeysBar(self)
        root_layout.addWidget(self.hotkeys_bar)

        self.v_splitter_widget = QSplitter(Qt.Orientation.Vertical)
        v_splitter = self.v_splitter_widget
        root_layout.addWidget(v_splitter, stretch=1)

        # Time slider (above main panel per spec)
        self.time_slider = TimeSliderPanel(self.nav, self.cd)
        self.time_slider.save_requested.connect(lambda: CustomCCGManageDialog.show(self.custom_mgr, self.time_slider, parent=self))
        self.time_slider.load_requested.connect(lambda: CustomCCGManageDialog.show(self.custom_mgr, self.time_slider, select_mode=True, parent=self))
        self.time_slider.queue_ccg_requested.connect(self._on_queue_ccg)
        time_slider_scroll = QScrollArea()
        time_slider_scroll.setWidgetResizable(True)
        time_slider_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        time_slider_scroll.setWidget(self.time_slider)
        v_splitter.addWidget(time_slider_scroll)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        v_splitter.addWidget(splitter)
        v_splitter.setSizes([180, 720])
        v_splitter.setHandleWidth(6)
        for i in range(v_splitter.count()):
            v_splitter.setCollapsible(i, True)

        # pairs_view: pair selection
        self.left_frame = QWidget()
        left_frame = self.left_frame
        self.pairs_view = PairSelectionPanelContainer(
            left_frame, self.nav.sel_data, self.nav,
            self.ui_states.panel_state)
        lft_layout = QVBoxLayout(left_frame)
        lft_layout.setContentsMargins(0, 0, 0, 0)
        _lft_title = QLabel("Pair Selection"); _lft_title.setStyleSheet("font-weight:bold;padding:2px 4px;")
        lft_layout.addWidget(_lft_title)
        lft_layout.addWidget(self.pairs_view)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setWidget(left_frame)
        splitter.addWidget(left_scroll)

        # Center: CCG panel
        self.mainview = CorrelogramPanel(self.nav)
        self.mainview.set_jitter_mgr(self.jitter_mgr)
        self.mainview._theme_fn = lambda: self.theme
        self.jitter_mgr.status_changed.connect(
            lambda text: self.mainview.jitter_section.set_running(bool(text)))
        center_scroll = QScrollArea()
        center_scroll.setWidgetResizable(True)
        center_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        center_scroll.setWidget(self.mainview)
        splitter.addWidget(center_scroll)

        # Right: probe network
        self.right_frame = QWidget()
        right_frame = self.right_frame
        right_outer = QVBoxLayout(right_frame)
        right_outer.setContentsMargins(0, 0, 0, 0)
        _rgt_title = QLabel("Neuron Network")
        _rgt_title.setStyleSheet("font-weight:bold;padding:2px 4px;")
        right_outer.addWidget(_rgt_title)
        net_scroll = QScrollArea()
        net_scroll.setWidgetResizable(True)
        net_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        net_container = QWidget()
        net_scroll.setWidget(net_container)
        right_outer.addWidget(net_scroll)
        self.neuron_network = NetworkPanel(net_container, self.nav)
        splitter.addWidget(right_frame)

        splitter.setSizes([340, 1040, 320])
        for i in range(splitter.count()):
            splitter.setCollapsible(i, True)
        self._splitter = splitter
        self._panel_splitter_map = {
            'pairs_view':    (splitter, 0, left_frame),
            'mainview':     (splitter, 1, self.mainview),
            'neuron_network': (splitter, 2, self.right_frame),
            'time_slider':   (v_splitter, 0, self.time_slider),
            'hotkeys_bar':   (None, None, self.hotkeys_bar),
        }

        ReviewMenuBar(self).build()
        self.status_bar = BottomStatusBar(self.nav, self.statusBar())
        self.stats_panel = StatsTestPanel(self.nav)
        self.stats_panel.setWindowTitle("Stats Tests")
        self.stats_panel.resize(1100, 750)

    def _switch_project(self, project_dir: str):
        config_name = project_dir[len('project_'):]
        header = ProjectConfig(name=config_name)
        if os.path.isfile(header.save_path() + '.json'):
            header.load()
        # A ProcessData project (or a pre-header one) names no source to scan: its sessions came
        # from the caller, so reuse the nd it was opened with — never the current project's.
        if header.scannable:
            _neurons, cd, _sd = open_project(config_name)
        else:
            nd = self._nd_by_project.get(project_dir, self.cd.nd)
            cd = CCGDataset(self.cd.conf.copy(name=config_name), nd)
            cd.load()
        self._adopt_project(cd)
        print(f"[CCGReviewUI] project → {project_dir} (config={config_name})",
              flush=True)

    def _adopt_project(self, new_cd: 'CCGDataset'):
        """Point every panel at *new_cd* and reload the current session through it."""
        self.nav.set_cd(new_cd)   # self.cd is a read-only property → nav.cd
        self._project_dir = _Path(new_cd.save_path).name   # cd owns the path; combo reads this
        self._nd_by_project[self._project_dir] = new_cd.nd
        self.jitter_mgr.cd = new_cd
        self.time_slider.cd = new_cd

        # a freshly built project has different sessions entirely -> fall back to its first;
        # nd keys come first because an uncomputed project has no ptr entries yet
        tk = (new_cd.find(str(self.nav.key.session), strict=False)
              or next(iter(self.nav.real_nd_keys()), None))
        if tk is not None:
            self.nav.set_key(tk)   # before any refresh: the old key names no session in new_cd

        self.nav.reset_selection_for_project(new_cd)
        os.makedirs(new_cd.custom_dir, exist_ok=True)

        if tk is not None:
            self._ensure_loaded(tk.nd(), 'lowres', lambda: self._switch_session(tk))
        self.index_bar.populate_session_combo()   # the new project owns different sessions
        self.index_bar.sync()

        self.pairs_view._autoload_session_latest(restore_groups=True)
        self.nav.apply_sel_for_key(self.nav.key)

        self.pairs_view.pair_selection.refresh_lists()
        self._post_load_refresh()

    def _panel_target(self, attr: str):
        """Widget whose visibility represents panel *attr* (splitter frame, else the panel)."""
        entry = self._panel_splitter_map.get(attr)
        return (entry[2] if entry else None) or getattr(self, attr, None)

    def _toggle_panel(self, attr: str):
        if attr == '_waveforms_panel':
            self._toggle_waveforms()   # waveforms live in mainview, not a standalone panel
            return
        target = self._panel_target(attr)
        if target is not None:
            self._set_panel_visible(attr, not target.isVisible())

    def _set_panel_visible(self, attr: str, visible: bool):
        """Show/hide a splitter panel, remembering its pre-collapse width on ui_states."""
        target = self._panel_target(attr)
        if target is None:
            return
        entry = self._panel_splitter_map.get(attr)
        if entry and entry[0] is not None:            # lives in a splitter
            spl, idx, _ = entry
            sizes = list(spl.sizes())
            if visible:
                sizes[idx] = self.ui_states.panel_sizes.get(attr, 300)
            else:
                if sizes[idx] > 0:
                    self.ui_states.panel_sizes[attr] = sizes[idx]
                sizes[idx] = 0
            spl.setSizes(sizes)
        target.setVisible(visible)
        panel = getattr(self, attr, None)
        if isinstance(panel, QWidget) and panel is not target:
            panel.setVisible(visible)
        act = self._panel_actions.get(attr)
        if act:
            act.setChecked(visible)

    def _show_stats_panel(self):
        self.stats_panel.show()
        self.stats_panel.raise_()

    def _run_classifier(self):
        ClassifierDialog.show_for(self)

    def _apply_min_font_size(self, size: int) -> None:
        app = QApplication.instance()
        f = app.font()
        f.setPointSize(size)
        app.setFont(f)
        if hasattr(self, 'pairs_view'):
            self.pairs_view.pair_selection.refresh_font()
        if hasattr(self, 'hotkeys_bar'):
            self.hotkeys_bar.refresh()
        if hasattr(self, 'time_slider'):
            self.time_slider._update_legend()
        if hasattr(self, 'neuron_network'):
            self.neuron_network.refresh_font()
        if hasattr(self, 'mainview'):
            self.mainview.refresh_font()

    def _show_transient_banner(self, message: str, duration_ms: int = 3500) -> None:
        """Brief status-bar message (e.g. unassigned group hotkey)."""
        self.statusBar().showMessage(message, duration_ms)

    # pairs_view / Right       Navigate segments
    # Del / Backspace    Move current pair to Deleted (in list focus)
    # 1-9, 0             Group hotkeys (assigned via Manage Groups)
    # Ctrl+1-0           Same (when list does not have keyboard focus)
    def _bind_shortcuts(self):
        def _sc(seq, fn):
            QShortcut(QKeySequence(seq), self).activated.connect(fn)
        _sc("pairs_view",           lambda: self._change_segment(-1))
        _sc("Right",          lambda: self._change_segment(1))
        _sc("Ctrl+R",         self._toggle_resolution)
        _sc("Ctrl+E",         self._toggle_waveforms)
        _sc("Ctrl+F",         self.pairs_view.pair_selection.search_bar.toggle)
        _sc("Ctrl+S",         lambda: QuickSaveDialog.show(self.pairs_view, self))
        _sc("Ctrl+B",         self.pairs_view.pair_selection._bookmark_toggle_current)
        _sc("Ctrl+Z",         self._undo)
        _sc("Ctrl+Y",         self._redo)
        _sc("Ctrl+Shift+Z",   self._redo)

    def _on_jitter_completed(self, ref, tgt, _res_key, _seg_key):
        inds = self.nav.current_pair_inds
        if inds is not None and int(inds[0]) == ref and int(inds[1]) == tgt:
            self.jitter_mgr.mark_viewed(ref, tgt)
            self.mainview.request_render()
        self.mainview.jitter_section.set_running(False)
        self.status_bar.refresh()

    def _on_jitter_failed(self, msg: str):
        self.mainview.jitter_section.set_running(False)
        QMessageBox.critical(self, "Jitter error", msg)

    def _on_spike_attribution_set(self, bin_val: float, unit: str):
        nav = self.nav
        if not self.mainview.sa_section.is_enabled:
            return
        inds = nav.current_pair_inds
        if inds is None:
            return
        ref, tgt = int(inds[0]), int(inds[1])
        # Bound to the appended window's extent (source config); 'full' = whole session (None).
        label = nav.current_segment
        src = self.cd.source_config(nav.key, label) if label and label != _ALL_SEGS else None
        t0 = float(src.t0) if src is not None and not isinstance(src.t0, str) else None
        t1 = float(src.t1) if src is not None and not isinstance(src.t1, str) else None
        pairs = compute_spike_pairs(
            nav.neurons, ref, tgt, bin_val, unit, nav.ccg_data, t0, t1)
        self.pairs_view.spike_pairs.populate(pairs)
        if pairs:
            self.pairs_view.spike_pairs.activate()

    def _on_pair_changed(self):
        self.status_bar.refresh()
        self.mainview.request_render()
        self.neuron_network.follow_current_pair()

    def _on_segment_changed(self):
        self.mainview.refresh_spike_attr_if_enabled()
        self.mainview.request_render()

    def _on_groups_changed(self):
        self.hotkeys_bar.refresh()
        self.neuron_network.refresh_group_buttons()

    def _on_key_changed(self):
        self.index_bar.sync()
        inds = self.nav.all_pairs_np
        if len(inds) > 0:
            self.nav.set_current_pair(min(self.nav.current_pair_idx, len(inds) - 1))
        self.pairs_view.pair_selection.refresh_lists()
        self.jitter_mgr.load_from_cd()
        self.nav.groups.changed.emit()
        self.neuron_network.refresh_ct_buttons()
        self.neuron_network.draw()
        self.status_bar.refresh()
        self.mainview.request_render()

    def _post_load_refresh(self):
        self.nav.groups.changed.emit()
        self.pairs_view.pair_selection.refresh_lists()
        self.neuron_network.draw()
        inds = self.nav.all_pairs_np
        if len(inds) > 0:
            idx = min(self.nav.current_pair_idx, len(inds) - 1)
            if idx != self.nav.current_pair_idx:
                self.nav.set_current_pair(idx)
            else:
                self._on_pair_changed()
        self.mainview.request_render()
        self.status_bar.refresh()

    def _initial_draw(self):
        print(f"[CCGReviewUI] ccg_ui={__file__}", flush=True)
        try:
            self.pairs_view._autoload_session_latest(restore_groups=True)
            self.nav.apply_sel_for_key(self.nav.key)
            self.pairs_view.pair_selection.refresh_lists()
            if self.time_slider is not None:
                self.time_slider.reload_themes()
            self._post_load_refresh()
            self.neuron_network.refresh_shank_buttons()
            self.status_bar.refresh()
            # Restore saved view (segment / resolution / splitter / collapse) before first draw.
            s = self.ui_states
            self.nav.set_current_segment(_ALL_SEGS)   # segment is per-session view state, never restored
            if s.resolution == 'hi':
                self._ensure_loaded(self.nav.key.nd(), 'highres',
                                    lambda: self.nav.set_resolution('hi'))
            if s.splitter_sizes:
                self._splitter.setSizes(s.splitter_sizes)
            for attr in list(s.collapsed_panels):
                self._set_panel_visible(attr, False)
            self.mainview.request_render()
            # Restore all-session mode after the single-session baseline is set up
            # (_enter_all_session_mode preserves the current key's type_label).
            if s.session_any_mode:
                self._enter_all_session_mode()
        except Exception:
            raise

    def _toggle_resolution(self):
        nd_key = self.nav.key.nd()
        if self.nav.resolution == "lo":
            self._ensure_loaded(nd_key, 'highres', lambda: self.nav.set_resolution("hi"))
        else:
            self.nav.set_resolution("lo")

    def _toggle_waveforms(self):
        btn = self.mainview.corr_section.ref_wf_btn
        btn.setChecked(not btn.isChecked())
        act = self._panel_actions.get('_waveforms_panel')
        if act:
            act.setChecked(btn.isChecked())
        if btn.isChecked():
            self.mainview.request_render()

    def _on_queue_ccg(self, spec: 'CCGBatchRequest'):
        # scope lives in spec; backend expands over sessions/splits
        n = self.custom_mgr._queue_custom_ccgs(spec)
        if n:
            if self.time_slider is not None:   # show status before the (possibly blocking) launch
                self.time_slider._status_lbl.setText(f"Queued {n} custom CCG task(s)")
                QApplication.processEvents()
            self.custom_mgr.worker._custom_ccg_start_next()
        else:
            if self.time_slider is not None:
                self.time_slider._status_lbl.setText("Nothing queued (check scope/session)")

    def _manage_groups(self):
        ManageGroupsDialog.show(
            self.nav.sel_data, self.pairs_view.pair_selection,
            pairs_by_conn_type_fn=self.nav._pairs_by_conn_type, parent=self)

    def _undo(self):
        self.pairs_view.pair_selection.undo()

    def _redo(self):
        self.pairs_view.pair_selection.redo()

    def _save_ui_state(self):
        # Refresh the nav-derived fields on ui_states from live state, then persist the whole thing.
        s = self.ui_states
        s.splitter_sizes = self._splitter.sizes()
        s.resolution = self.nav.resolution
        s.current_segment = self.nav.current_segment
        s.session = str(self.nav.key.session)
        s.type_label = self.nav.key.type_label()
        s.session_any_mode = bool(self.nav.session_any_mode)
        s.collapsed_panels = [a for a in self._panel_splitter_map
                              if not self._panel_target(a).isVisible()]
        s.save()   # settings / panel_sizes already live on s

    def closeEvent(self, event: 'QCloseEvent'):
        self.nav.closing.emit()
        if self.settings.save_ui_on_close:
            self._save_ui_state()
        super().closeEvent(event)

    def _enter_all_session_mode(self):
        type_keys = self.nav.available_type_keys_any()
        if not type_keys:
            QMessageBox.warning(self, "All sessions", "No connection types in dataset.")
            return
        prev_lbl = self.nav.key.type_label()
        type_labels = [k.type_label() for k in type_keys]
        new_key = type_keys[type_labels.index(prev_lbl)] if prev_lbl in type_labels else type_keys[0]
        self.all_sess_mgr.load_groups()
        self._ensure_loaded(new_key.nd(), 'lowres', lambda: self._switch_session(new_key))
        self.nav.set_session_any_mode(True)
        self.index_bar.sync()

    def _exit_all_session_mode(self):
        self.all_sess_mgr.flush_deleted_to_stores()
        self.nav.set_session_any_mode(False)
        self.index_bar.sync()

    def _on_session_mode_changed(self, any_mode: bool):
        if any_mode:
            self.nav.any_expanded_group_tags = (
                set(self.nav.groups.header_names())
                | {str(k.session) for k in self.nav.real_nd_keys()})
            self.pairs_view.pair_selection._sort_btns.select('tag')
            self.all_sess_mgr.rebuild_universe()
            self.nav.set_current_pair(0)
            self.nav.set_current_segment(_ALL_SEGS)
        else:
            self.nav.any_expanded_group_tags.clear()
            self.nav.set_cross_session_handles([])
            self.nav.apply_sel_for_key(self.nav.key)
        np = self.neuron_network
        if any_mode:
            np._net_any_sessions_cache = np._net_any_sessions()
            np._net_any_idx = 0
        np.refresh_ct_buttons()
        np.draw()
        self.time_slider._on_session_mode_changed(any_mode)
        self._post_load_refresh()

    def _ccg_ready(self, nd_key, resolution: str = 'lowres') -> bool:
        data = self.cd.ccg_for(nd_key.change(resolution=resolution))
        return data is not None and data.ccg is not None and data.ccg.ndim >= 4

    def _ensure_loaded(self, nd_key, resolution: str, on_loaded):
        if self._ccg_ready(nd_key, resolution):
            on_loaded()
            return
        # Reentrancy guard: a load already in flight → ignore (avoids overlapping cd mutation
        # from rapid session switches).
        if self._loading_thread is not None:
            return
        # Busy dialog kept hidden; a 2s single-shot reveals it only if the load outlasts 2s.
        dlg = QProgressDialog("Loading…", None, 0, 0, self)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setCancelButton(None)
        show_timer = QTimer(self)
        show_timer.setSingleShot(True)
        show_timer.timeout.connect(dlg.show)
        show_timer.start(2000)

        thread = QThread(self)
        worker = _CCGLoadWorker(self.cd, nd_key, resolution)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        self._loading_thread, self._loading_worker = thread, worker

        def _teardown():
            show_timer.stop()
            dlg.close()
            thread.quit()
            thread.wait()
            worker.deleteLater()
            thread.deleteLater()
            self._loading_thread = self._loading_worker = None

        def _on_done():
            _teardown()
            on_loaded()

        def _on_error(msg: str):
            _teardown()
            QMessageBox.critical(self, "Load failed", f"Could not load CCG data:\n{msg}")

        worker.done.connect(_on_done)
        worker.error.connect(_on_error)
        thread.start()

    def _ensure_sessions_loaded(self, nd_keys, resolution: str):
        """Batch-load CCG for many sessions (one progress dialog). No eviction, so
        every session stays resident for immediate cross-session metric capture."""
        pending = [k for k in nd_keys if not self._ccg_ready(k, resolution)]
        if not pending:
            return
        dlg = QProgressDialog("Loading sessions…", None, 0, len(pending), self)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.show()
        for i, k in enumerate(pending):
            dlg.setValue(i)
            QApplication.processEvents()
            self.cd.get_ccg(k.change(resolution=resolution))
        dlg.close()

    def _change_segment(self, delta: int):
        all_names = self.nav.segment_names()
        if not all_names:
            return
        cur = self.nav.current_segment
        idx = all_names.index(cur) if cur in all_names else 0
        self.nav.set_current_segment(all_names[(idx + delta) % len(all_names)])
        self.mainview.request_render()

    def _switch_session(self, new_key) -> bool:
        """Switch to a new Key. Returns False if data unavailable."""
        if self.nav.session_any_mode:
            return self._switch_type_any(new_key)

        new_session = str(new_key.session)
        session_changed = str(self.nav.key.session or '') != new_session

        # On a session change, load that session's saved selection from disk (window owns the
        # IO decision; nav runs it at the right point in the transition).
        load = None
        if session_changed:
            path = os.path.join(self.cd.selections_dir, f"{new_session}.json")
            if os.path.isfile(path):
                load = lambda: self.pairs_view._load_selection_from_file(
                    path, restore_groups=True, _skip_redraw=True)

        self.nav.switch_key(new_key, load_selection=load)

        self.index_bar.sync()
        self._post_load_refresh()
        return True

    def _switch_type_any(self, new_key) -> bool:
        """All-session mode: rebuild the cross-session universe for a new conn type."""
        self.nav.set_key(new_key)
        self.index_bar.sync()
        self.all_sess_mgr.rebuild_universe()
        self.nav.set_current_pair(0)
        self.neuron_network.refresh_ct_buttons()
        self.neuron_network.draw()
        self._post_load_refresh()
        return True

    @classmethod
    def launch(cls, cd: 'CCGDataset', key=None) -> 'CCGReviewUI':
        """Create and show the Qt review UI.

        key may be a full Key, a session-string query, or None.
        When omitted the last-used session is restored from ui_state_qt.json.
        """
        QApplication.instance() or QApplication([])
        win = cls(cd, key) # __init__
        win.show()
        return win
