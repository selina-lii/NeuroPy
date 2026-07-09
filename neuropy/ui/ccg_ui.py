"""CCGReviewUI — Qt root window for CCG Manual Review."""
from __future__ import annotations
import os
import json

from dataclasses import dataclass
from pathlib import Path as _Path
from typing import TYPE_CHECKING
from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.Qt.QtCore import Qt, QTimer
from pyqtgraph.Qt.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QStatusBar, QLabel, QApplication,
    QProgressDialog, QMessageBox, QFileDialog, QTabWidget, QPushButton, QScrollArea,
)
from pyqtgraph.Qt.QtGui import QKeySequence, QCloseEvent, QAction, QShortcut, QPalette, QColor
from neuropy.analyses.neurons_dataset import Key
from neuropy.ui.app_state import AppState, _ALL_SESSION_MARKER, _ALL_SEGS
from neuropy.ui.pair_selection_panel import (
    PairSelectionPanelContainer, SelectionData,
)
from neuropy.ui.ui_common import (
    UITheme, atomic_write_json, qt_dark_mode,
)
from neuropy.ui.utils import GroupHotkeysBar
from neuropy.ui.dialogs import (
    QuickSaveDialog, LoadSelectionDialog, ManageGroupsDialog,
    CreateGroupDialog,
    VersionSaveDialog, VersionLoadDialog, CustomCCGManageDialog,
)
from neuropy.ui.stats_tests import StatsTestPanelQt
from neuropy.ui.jitter_ui import JitterControllerQt, JitterQueueDialog
from neuropy.ui.time_slider import CustomCCGManager, TimeSliderPanelQt
from neuropy.ui.menubar import ReviewMenuBar, IndexBar
from neuropy.ui.ccg_panel import CorrelogramPanel
from neuropy.ui.neuron_network import NetworkPanel
from neuropy.analyses.ms_connectivity import CCGDataset, CCGSourceConfig
from neuropy.ui.all_session_mode import AllSessionMode
from neuropy.analyses.spike_attribution import compute_spike_pairs
from neuropy.analyses.ccg_transforms import CCGNorm
from neuropy.analyses.epoch_filter import EpochFilter

if TYPE_CHECKING:
    pass

@dataclass
class SavePaths:
    data_root: _Path
    project_dir: str

    @classmethod
    def from_config(cls, config_name: str) -> 'SavePaths':
        root = _Path(__file__).resolve().parents[2] / 'data'
        return cls(data_root=root, project_dir=f'project_{config_name}')

    @property
    def selections_dir(self) -> str:
        return str(self.data_root / self.project_dir / 'selections')

    @property
    def custom_ccg_dir(self) -> str:
        return str(self.data_root / self.project_dir / 'custom_ccg')

    @property
    def ui_state_file(self) -> str:
        return str(self.data_root / 'ui_state_qt.json')

    def for_project(self, project_dir: str) -> 'SavePaths':
        return SavePaths(data_root=self.data_root, project_dir=project_dir)


@dataclass
class UISettings:
    min_font_size: int = 12
    save_ui_on_close: bool = True
    list_cursor_follows_action: bool = True
    autosave_sel_on: bool = True
    autosave_sel_interval: int = 1
    autosave_sel_unit: str = 'hour'
    autosave_grp_on: bool = True
    autosave_grp_interval: int = 1
    autosave_grp_unit: str = 'hour'
    ccg_memory_limit_gb: float = 4.0


class BottomStatusBar:
    """Owns the stats QLabel in the window status bar. Refreshes on nav signals."""

    def __init__(self, nav: 'AppState', status_bar):
        self._nav = nav
        self.label = QLabel("")
        status_bar.addWidget(self.label)
        self._pair_info_label = QLabel("")
        status_bar.addPermanentWidget(self._pair_info_label)
        nav.pair_changed.connect(self.refresh)
        nav.key_changed.connect(self.refresh)
        nav.selection_changed.connect(self.refresh)

    def refresh(self, *_):
        self.label.setText(self._stats_str())

    @staticmethod
    def _counts_for(neurons, ref_t, tgt_t):
        """(n_ref, n_tgt, n_poss) for one session's neurons; None if type absent."""
        try:
            n_ref = neurons.get_neuron_type(ref_t).n_neurons
            n_tgt = neurons.get_neuron_type(tgt_t).n_neurons
        except Exception:
            return None
        n_poss = n_ref * (n_ref - 1) if ref_t == tgt_t else n_ref * n_tgt
        return n_ref, n_tgt, n_poss

    @staticmethod
    def _counts_str(ref_t, tgt_t, n_ref, n_tgt) -> str:
        return (f"  {ref_t}: {n_ref}" if ref_t == tgt_t
                else f"  ref({ref_t}): {n_ref}  tgt({tgt_t}): {n_tgt}")

    def _stats_str(self) -> str:
        nav = self._nav
        n_sig = len(nav.all_inds)
        n_sel = len(nav.active_selections.selected)
        key = nav.key
        if key is None:
            return f"Significant: {n_sig}  Selected: {n_sel}"
        if nav.session_any_mode:
            return self._stats_str_any(nav, key, n_sig, n_sel)

        ct_prefix = f"{key.type_label()} | "
        s = f"Significant: {n_sig}  Selected: {n_sel}"
        if key.conn_type:
            neurons = nav.neurons
            if neurons is not None:
                ref_t, tgt_t = key.conn_type
                counts = self._counts_for(neurons, ref_t, tgt_t)
                if counts is not None:
                    n_ref, n_tgt, n_poss = counts
                    s = f"Significant: {n_sig}/{n_poss}  Selected: {n_sel}/{n_poss}"
                    s += self._counts_str(ref_t, tgt_t, n_ref, n_tgt)
                inds = nav.current_pair_inds
                if inds is not None:
                    ref, tgt = int(inds[0]), int(inds[1])
                    try:
                        s += f"  |  FR: ref={neurons.firing_rate[ref]:.1f}Hz  tgt={neurons.firing_rate[tgt]:.1f}Hz"
                    except Exception:
                        pass
        return ct_prefix + s

    def _stats_str_any(self, nav, key, n_sig, n_sel) -> str:
        """Pooled cross-session stats for all-session mode."""
        s = f"Significant: {n_sig}  Selected: {n_sel}"
        if key.conn_type:
            ref_t, tgt_t = key.conn_type
            n_ref = n_tgt = n_poss = 0
            for nk in nav.real_nd_keys():
                neurons = nav.cd.nd.data.get(nk)
                if neurons is None:
                    continue
                counts = self._counts_for(neurons, ref_t, tgt_t)
                if counts is None:
                    continue
                a, b, p = counts
                n_ref += a
                n_tgt += b
                n_poss += p
            if n_poss:
                s = f"Significant: {n_sig}/{n_poss}  Selected: {n_sel}/{n_poss}"
                s += self._counts_str(ref_t, tgt_t, n_ref, n_tgt)
        return f"All sessions · {key.type_label()} | " + s


class CCGReviewUI(QMainWindow):
    """Qt root window for CCG Manual Review.

    Owns AppState. Panels are created in _build_layout() and receive
    nav as their primary interface.
    """

    def __init__(self, cd: 'CCGDataset', key: Key):
        super().__init__()

        self._cd  = cd
        self._key = key
        self.theme = UITheme.from_dark(qt_dark_mode())

        self._nav = AppState(cd, key)
        self._nav.root = self

        self.jitter_controller = JitterControllerQt(self._nav, cd)
        self.jitter_controller.jitter_completed.connect(self._on_jitter_completed)
        self.jitter_controller.jitter_failed.connect(self._on_jitter_failed)

        self.paths = SavePaths.from_config(cd.conf.name)
        self._nav.sd.save_dir = self.paths.selections_dir

        self._saved_panel_sizes: dict[str, int] = {}

        self._build_layout()
        self._custom_mgr = CustomCCGManager(self)
        self.any_session = AllSessionMode(self._nav, self._cd, self.paths)
        self._bind_shortcuts()

        self._nav.pair_changed.connect(self._on_pair_changed)
        self._nav.segment_changed.connect(self._on_segment_changed)
        self._nav.key_changed.connect(self._on_key_changed)
        self._nav.session_mode_changed.connect(self._on_session_mode_changed)
        self._nav.cross_session_handles_changed.connect(
            lambda: self.request_redraw() if self._nav.session_any_mode else None)
        self._nav.groups.changed.connect(self._on_groups_changed)

        QTimer.singleShot(100, self._initial_draw)

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_theme(redraw=True)

    def _refresh_theme(self, *, redraw: bool = False):
        self.theme = UITheme.from_dark(qt_dark_mode())
        if redraw:
            self.request_redraw()
            self.neuron_network.draw()

    def toggle_dark_mode(self):
        app = QApplication.instance()
        if qt_dark_mode():
            app.setPalette(app.style().standardPalette())
        else:
            p = QPalette()
            dark = QColor(30, 30, 30)
            p.setColor(QPalette.ColorRole.Window,          dark)
            p.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
            p.setColor(QPalette.ColorRole.Base,            QColor(42, 42, 42))
            p.setColor(QPalette.ColorRole.AlternateBase,   dark)
            p.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
            p.setColor(QPalette.ColorRole.Button,          QColor(53, 53, 53))
            p.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
            p.setColor(QPalette.ColorRole.Highlight,       QColor(42, 130, 218))
            p.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
            app.setPalette(p)
        self._refresh_theme(redraw=True)
        self.hotkeys_bar._update_dark_btn()
        self.hotkeys_bar.refresh()

    def _build_layout(self):
        self.settings = UISettings()
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
        self.time_slider = TimeSliderPanelQt(self._nav, self._cd)
        self.time_slider.save_requested.connect(lambda: CustomCCGManageDialog.show(self._custom_mgr, self.time_slider, parent=self))
        self.time_slider.load_requested.connect(lambda: CustomCCGManageDialog.show(self._custom_mgr, self.time_slider, select_mode=True, parent=self))
        self.time_slider.ccg_enqueue_requested.connect(self._on_ccg_enqueue)
        v_splitter.addWidget(self.time_slider)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        v_splitter.addWidget(splitter)
        v_splitter.setSizes([180, 720])
        v_splitter.setHandleWidth(6)

        # pairs_view: pair selection
        self.left_frame = QWidget()
        left_frame = self.left_frame
        self.pairs_view = PairSelectionPanelContainer(
            left_frame, self._nav.sel_data, self._nav,
            self._load_ui_state().get('panel_state', {}))
        lft_layout = QVBoxLayout(left_frame)
        lft_layout.setContentsMargins(0, 0, 0, 0)
        _lft_title = QLabel("Pair Selection"); _lft_title.setStyleSheet("font-weight:bold;padding:2px 4px;")
        lft_layout.addWidget(_lft_title)
        lft_layout.addWidget(self.pairs_view)
        splitter.addWidget(left_frame)

        # Center: CCG panel
        self.mainview = CorrelogramPanel(self._nav)
        self.mainview.set_jitter_ctrl(self.jitter_controller)
        self.mainview._theme_fn = lambda: self.theme
        self.jitter_controller.status_changed.connect(
            lambda text: self.mainview.jitter_section.set_running(bool(text)))
        splitter.addWidget(self.mainview)

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
        self.neuron_network = NetworkPanel(net_container, self._nav)
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
        self.status_bar = BottomStatusBar(self._nav, self.statusBar())
        self.stats_panel = StatsTestPanelQt(self._nav, self._cd)
        self.stats_panel.setWindowTitle("Stats Tests")
        self.stats_panel.resize(1100, 750)

    def _switch_project(self, project_dir: str):
        config_name = project_dir[len('project_'):]
        new_conf = self._cd.conf.copy(name=config_name)
        new_cd = CCGDataset(new_conf, self._cd.nd)

        self.cd = new_cd
        self._nav.set_cd(new_cd)
        self.jitter_controller._cd = new_cd
        self.time_slider._cd = new_cd

        self.paths = self.paths.for_project(project_dir)
        self._nav.reset_selection_for_project(new_cd, self.paths.selections_dir)
        os.makedirs(self.paths.custom_ccg_dir, exist_ok=True)

        sess = str(self._nav.key.session)
        tk = new_cd.find(sess)
        if tk is not None:
            self._ensure_loaded(tk.nd(), 'lowres', lambda: self._switch_session(tk))
        self.index_bar.sync()

        self.pairs_view._autoload_session_latest(restore_groups=True)
        self._nav.apply_sel_for_key(self._nav.key)

        self.pairs_view.pair_selection.refresh_lists()
        self._post_load_refresh()
        print(f"[CCGReviewUI] project → {project_dir} (config={config_name})",
              flush=True)

    def _toggle_panel(self, attr: str):
        if attr == '_waveforms_panel':
            self._toggle_waveforms()   # waveforms live in mainview, not a standalone panel
            return
        panel = getattr(self, attr, None)
        if panel is None:
            return
        visible = not panel.isVisible()
        entry = self._panel_splitter_map.get(attr)
        if entry is not None:
            spl, idx, frame = entry
            if spl is not None and idx is not None:
                sizes = list(spl.sizes())
                if not visible:
                    if sizes[idx] > 0:
                        self._saved_panel_sizes[attr] = sizes[idx]
                    sizes[idx] = 0
                    spl.setSizes(sizes)
                else:
                    sizes[idx] = self._saved_panel_sizes.get(attr, 300)
                    spl.setSizes(sizes)
                    if frame is not None:
                        frame.setVisible(True)
        panel.setVisible(visible)
        act = self._panel_actions.get(attr)
        if act:
            act.setChecked(visible)

    def _show_stats_panel(self):
        self.stats_panel.show()
        self.stats_panel.raise_()

    def _run_classifier(self):
        QMessageBox.information(self, "Classify", "Classifier not implemented yet.")

    def _apply_min_font_size(self, size: int) -> None:
        app = QApplication.instance()
        f = app.font()
        if f.pointSize() < size:
            f.setPointSize(size)
            app.setFont(f)

    def show_transient_banner(self, message: str, duration_ms: int = 3500) -> None:
        """Brief status-bar message (e.g. unassigned group hotkey)."""
        self.statusBar().showMessage(message, duration_ms)

    # fmt: off
    # Complete shortcut reference — update here when adding/removing shortcuts.
    # pairs_view / Right       Navigate segments
    # Ctrl+R             Toggle hi/lo resolution
    # Ctrl+E             Toggle waveform panel
    # Ctrl+F             Toggle search bar
    # Ctrl+S             Save selection
    # Ctrl+B             Toggle bookmark on current pair
    # Ctrl+Z             Undo
    # Ctrl+Y / Ctrl+Shift+Z   Redo
    # Del / Backspace    Move current pair to Deleted (in list focus)
    # 1-9, 0             Group hotkeys (assigned via Manage Groups)
    # Ctrl+1-0           Same (when list does not have keyboard focus)
    # fmt: on
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
        inds = self._nav.current_pair_inds
        if inds is not None and int(inds[0]) == ref and int(inds[1]) == tgt:
            self.jitter_controller.mark_viewed(ref, tgt)
            self.request_redraw()
        self.mainview.jitter_section.set_running(False)
        self.status_bar.refresh()

    def _on_jitter_failed(self, msg: str):
        self.mainview.jitter_section.set_running(False)
        QMessageBox.critical(self, "Jitter error", msg)

    def _on_spike_attribution_set(self, bin_val: float, unit: str):
        nav = self._nav
        if not self.mainview.sa_section.is_enabled:
            return
        inds = nav.current_pair_inds
        if inds is None:
            return
        ref, tgt = int(inds[0]), int(inds[1])
        pairs = compute_spike_pairs(
            nav.neurons, ref, tgt, bin_val, unit,
            nav.ccg_data, nav.seg_idx(nav.current_segment),
            nav.n_segments, self._cd.edge_times_for(nav.key))
        self.pairs_view.spike_pairs.populate(pairs)
        if pairs:
            self.pairs_view.spike_pairs.activate()

    def _on_pair_changed(self, _idx: int):
        self.status_bar.refresh()
        self.request_redraw()
        self.neuron_network.draw()   # current-pair edge must track the selected pair

    def _on_segment_changed(self, _name: str):
        self.mainview.refresh_spike_attr_if_enabled()
        self.request_redraw()

    def _on_groups_changed(self) -> None:
        self.hotkeys_bar.refresh()
        self.neuron_network.refresh_group_buttons()

    def _on_key_changed(self, _key):
        self.index_bar.sync()
        inds = self._nav.all_inds
        if len(inds) > 0:
            self._nav.set_current_pair(min(self._nav.current_pair_idx, len(inds) - 1))
        self.pairs_view.pair_selection.refresh_lists()
        self.jitter_controller.load_from_cd()
        self._nav.groups.changed.emit()
        self.neuron_network.refresh_ct_buttons()
        self.neuron_network.draw()
        self.status_bar.refresh()
        self.request_redraw()

    def request_redraw(self):
        """Ask CorrelogramPanel to re-render the current pair."""
        self.mainview.request_render()

    def _post_load_refresh(self):
        self._nav.groups.changed.emit()
        self.pairs_view.pair_selection.refresh_lists()
        self.neuron_network.draw()
        inds = self._nav.all_inds
        if len(inds) > 0:
            idx = min(self._nav.current_pair_idx, len(inds) - 1)
            if idx != self._nav.current_pair_idx:
                self._nav.set_current_pair(idx)
            else:
                self._on_pair_changed(idx)
        self.request_redraw()
        self.status_bar.refresh()

    def _initial_draw(self):
        print(f"[CCGReviewUI] ccg_ui={__file__}", flush=True)
        try:
            self.pairs_view._autoload_session_latest(restore_groups=True)
            self._nav.apply_sel_for_key(self._nav.key)
            self.pairs_view.pair_selection.refresh_lists()
            if self.time_slider is not None:
                self.time_slider.reload_themes()
            self._post_load_refresh()
            self.neuron_network.refresh_shank_buttons()
            self.status_bar.refresh()
            self.request_redraw()
        except Exception:
            raise

    def _toggle_resolution(self):
        nd_key = self._nav.key.nd()
        if self._nav.resolution == "lo":
            self._ensure_loaded(nd_key, 'highres', lambda: self._nav.set_resolution("hi"))
        else:
            self._nav.set_resolution("lo")

    def _toggle_waveforms(self):
        btn = self.mainview.corr_section.ref_wf_btn
        btn.setChecked(not btn.isChecked())
        act = self._panel_actions.get('_waveforms_panel')
        if act:
            act.setChecked(btn.isChecked())
        if btn.isChecked():
            self.request_redraw()

    def _on_ccg_enqueue(self, spec):
        if isinstance(spec, dict):
            spec = CCGSourceConfig.deserialize(spec, default_session=str(self._nav.key.session))
        all_sessions = {str(k.session) for k in self._nav.real_nd_keys()}
        picked = [str(s) for s in (spec.sessions or []) if str(s).lower() != 'all']
        scope_all = str(spec.scope or '').lower() == 'all'
        if not picked and not scope_all:
            picked = [str(spec.scope or self._nav.key.session)]
        for_all = scope_all or (bool(all_sessions) and bool(picked)
                                and set(picked) >= all_sessions)
        target = None if for_all else picked
        print(f"[enqueue] spec={spec.name!r} picked={picked} for_all={for_all}")
        n = self._custom_mgr._queue_custom_ccgs(
            spec, for_all=for_all, auto_save=True, target_sessions=target)
        print(f"[enqueue] queued={n}")
        if n:
            self._custom_mgr.worker._custom_ccg_start_next()
            if self.time_slider is not None:
                self.time_slider._status_lbl.setText(f"Queued {n} custom CCG task(s)")
        else:
            if self.time_slider is not None:
                self.time_slider._status_lbl.setText("Nothing queued (check scope/session)")

    def _manage_groups(self):
        ManageGroupsDialog.show(
            self._nav.sel_data, self.pairs_view.pair_selection,
            pairs_by_conn_type_fn=self._nav._pairs_by_conn_type, parent=self)

    def _undo(self):
        self.pairs_view.pair_selection.undo()

    def _redo(self):
        self.pairs_view.pair_selection.redo()

    def _load_ui_state(self) -> dict:
        try:
            with open(self.paths.ui_state_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_ui_state(self):
        state = {
            'splitter_sizes': self._splitter.sizes(),
            'resolution': self._nav.resolution,
            'current_segment': self._nav.current_segment,
        }
        os.makedirs(os.path.dirname(self.paths.ui_state_file), exist_ok=True)
        with open(self.paths.ui_state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def closeEvent(self, event: 'QCloseEvent'):
        self._nav.closing.emit()
        self._save_ui_state()
        super().closeEvent(event)

    def _enter_all_session_mode(self):
        type_keys = self._nav.available_type_keys_any()
        if not type_keys:
            QMessageBox.warning(self, "All sessions", "No connection types in dataset.")
            return
        prev_lbl = self._nav.key.type_label()
        type_labels = [k.type_label() for k in type_keys]
        new_key = type_keys[type_labels.index(prev_lbl)] if prev_lbl in type_labels else type_keys[0]
        self.any_session.load_groups()
        self._ensure_loaded(new_key.nd(), 'lowres', lambda: self._switch_session(new_key))
        self._nav.set_session_any_mode(True)
        self.index_bar.sync()

    def _exit_all_session_mode(self):
        self.any_session.flush_deleted_to_stores()
        self._nav.set_session_any_mode(False)
        self.index_bar.sync()

    def _on_session_mode_changed(self, any_mode: bool):
        if any_mode:
            self._nav.any_expanded_group_tags = (
                set(self._nav.groups.header_names())
                | {str(k.session) for k in self._nav.real_nd_keys()})
            self.pairs_view.pair_selection._sort_btns.select('tag')
            self.any_session.load_deleted_aggregate()
            self.any_session.rebuild_pair_handles()
            self.any_session.sync_selection_from_universe()
            self._nav.set_current_pair(0)
            self._nav.set_current_segment(_ALL_SEGS)
        else:
            self._nav.any_expanded_group_tags.clear()
            self._nav.set_cross_session_handles([])
            self._nav.apply_sel_for_key(self._nav.key)
        np = self.neuron_network
        if any_mode:
            np._net_any_sessions_cache = np._net_any_sessions()
            np._net_any_idx = 0
        np.refresh_ct_buttons()
        np.draw()
        self.time_slider._on_session_mode_changed(any_mode)
        self._post_load_refresh()

    def _ccg_ready(self, nd_key, resolution: str = 'lowres') -> bool:
        data = self._cd.ccg_for(nd_key, resolution)
        return data is not None and data.ccg is not None and data.ccg.ndim >= 4

    def _ensure_loaded(self, nd_key, resolution: str, on_loaded):
        if self._ccg_ready(nd_key, resolution):
            on_loaded()
            return
        dlg = QProgressDialog("Loading…", None, 0, 0, self)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.show()
        QApplication.processEvents()
        self._cd.get_ccg(resolution=resolution, nd_key=nd_key)
        self._cd._check_memory_and_evict(self.settings.ccg_memory_limit_gb)
        dlg.close()
        on_loaded()

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
            self._cd.get_ccg(resolution=resolution, nd_key=k)
        dlg.close()

    def _seg_name(self, idx: int) -> str:
        if idx == self._nav.n_segments:
            return _ALL_SEGS
        cs_list = self._custom_segments
        if idx > self._nav.n_segments:
            ci = idx - self._nav.n_segments - 1
            if 0 <= ci < len(cs_list):
                return cs_list[ci].src_conf.name
            return _ALL_SEGS
        names = self._cd.segment_names_for(self._nav.key)
        return names[idx] if 0 <= idx < len(names) else _ALL_SEGS

    def _change_segment(self, delta: int):
        all_names = self._nav.all_segment_names()
        if not all_names:
            return
        cur = self._nav.current_segment
        idx = all_names.index(cur) if cur in all_names else 0
        self._nav.set_current_segment(all_names[(idx + delta) % len(all_names)])
        self.request_redraw()

    def _switch_session(self, new_key) -> bool:
        """Switch to a new Key. Returns False if data unavailable."""
        if self._nav.session_any_mode:
            return self._switch_type_any(new_key)

        prev_key = self._nav.key
        prev_session = prev_key.session

        if new_key == prev_key:
            self._nav.apply_sel_for_key(new_key)
            self._post_load_refresh()
            return True

        # prev_key's bucket already holds its selected/deleted state (buckets persist).
        _prev_seg = self._nav.current_segment
        new_session = str(new_key.session)
        session_changed = str(prev_session or '') != new_session
        sess_sel_path = os.path.join(self.paths.selections_dir, f"{new_session}.json")

        if not session_changed:
            self._nav.apply_sel_for_key(new_key)
        self._nav.set_key(new_key)
        self.index_bar.sync()

        if session_changed and os.path.isfile(sess_sel_path):
            self.pairs_view._load_selection_from_file(
                sess_sel_path, restore_groups=True, _skip_redraw=True)
        elif session_changed:
            self._nav.apply_sel_for_key(new_key)

        if _prev_seg in self._nav.all_segment_names():
            self._nav.set_current_segment(_prev_seg)
        self._nav.clamp_segment()

        self._nav.set_current_pair(0)
        self._nav.set_active_norms(set())

        if session_changed:
            self._custom_segments = self._custom_mgr._by_session.setdefault(new_session, [])
            self._nav._custom_seg_index = {
                cd.src_conf.name: cd
                for cd in self._custom_segments
                if cd.src_conf is not None
            }
            self._nav.custom_segs_changed.emit()
        self._post_load_refresh()
        return True

    def _switch_type_any(self, new_key) -> bool:
        """All-session mode: rebuild the cross-session universe for a new conn type."""
        self._nav.set_key(new_key)
        self.index_bar.sync()
        self.any_session.load_deleted_aggregate()
        self.any_session.rebuild_pair_handles()
        self.any_session.sync_selection_from_universe()
        self._nav.set_current_pair(0)
        self.neuron_network.refresh_ct_buttons()
        self.neuron_network.draw()
        self._post_load_refresh()
        return True

    def default_launch_key(cd: 'CCGDataset', session_query: str) -> Key:
        return cd.find(session_query)

    @classmethod
    def launch(cls, cd: 'CCGDataset', key: Key) -> 'CCGReviewUI':
        """Create and show the Qt review UI.
        """
        app = QApplication.instance() or QApplication([])
        sess = str(key.session)
        key = cls.default_launch_key(cd, sess)
        nd_key = key.nd()
        data = cd.ccg_for(nd_key, 'lowres')
        if data is None or data.ccg is None or data.ccg.ndim < 4:
            cd.get_ccg(nd_key=nd_key)
        win = cls(cd, key)
        win.show()
        return win
