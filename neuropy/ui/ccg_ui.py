"""
CCG Manual Review UI

Interactive GUI for reviewing and selecting significant CCG pairs.

Layout (3-column PanedWindow):
  Left   — pair selection lists + alpha control
  Center — CCG plot with normalization toggles, segment navigation, optional
            waveforms sub-panel
  Right  — probe network (neuron positions + connection edges)

Top bar:  menubar (Panels / Groups / Selections menus) + tool strip
Time slider: full-width panel above the 3-column area, hidden by default.
Bottom bar: pair statistics + Save / Cancel buttons.

Keyboard shortcuts
------------------
  ←  /  →      previous / next segment
  Ctrl+R        toggle lo-res ↔ hi-res
  Ctrl+L        toggle bar ↔ line for visible traces (CCG, baseline, ACGs)
  Ctrl+E        toggle waveforms sub-panel
  Ctrl+S        save selection + export groups
  1..0          assign / jump to group 1–10
"""

import io
import re
import glob as _glob
import shutil
import time as _time
import traceback
from collections import defaultdict as _defaultdict
from copy import deepcopy
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import os
import json
import datetime
import collections
import subprocess
import sys
import threading
import multiprocessing as _mp
from pathlib import Path as _Path
from PIL import Image, ImageDraw, ImageFont
from neuropy.plotting import ccg as plot_ccg
from neuropy.plotting.probe import plot_waveform_on_channel, plot_probe
from neuropy.plotting._jitter_worker import jitter_worker
from neuropy.analyses.jitter import JitterManager, _MAX_JITTER_QUEUE as _JITTER_QUEUE_MAX
from neuropy.analyses.ccg_classifier import (
    CCGTemplateClassifier, GroupTemplate, PeakRule,
    CCGClassifier, CCGClusterClassifier, ClassifyResult,
)
try:
    from neuropy.analyses.ccg_classifier import ccgconfig_to_main_template as _ccgconfig_to_main_template
except ImportError:
    _ccgconfig_to_main_template = None
# Connectivity strength
import copy as _copy
from neuropy.analyses.correlations import spike_correlations
from neuropy.analyses.ms_connectivity import (
    EranConv, CCGConfig, CCGData, _CCG_RESOLUTION, deconv_autocorr,
    NormalizeBy, apply_norms_to_ccg, compute_pair_conn_strength_1d,
    CCGPanelData, compute_ccg_panel_data,
)
from neuropy.analyses.neurons_dataset import Key
from neuropy.core.neurons import Neurons
import imageio
from neuropy.ui.ccg_network_panel import NetworkPanel
from neuropy.ui.pair_selection_panel import LeftPanelContainer, SelectionData

try:
    from neuropy.core.epoch import Epoch as _Epoch
except ImportError:
    _Epoch = None

# Sentinel value for the virtual "All segments" view
_ALL_SEGS = "All"
_ADMITTED_GROUP = "__admitted__"
_SPECIAL_PREFIX = "__special_"
# Virtual session entry: union of all sessions (lazy-loaded per group tag).
_ALL_SESSION_MARKER = object()
# Backward-compat alias for older code paths.
_ANY_SESSION_MARKER = _ALL_SESSION_MARKER
_AVAIL_GROUP_HDR = "__avail_group_hdr__"

# maximum number of queued jitters (including the currently running one)
_MAX_JITTER_QUEUE = 50



class CCGReviewUI:
    """
    GUI for reviewing and manually selecting CCG pairs.

    Parameters
    ----------
    cd : CCGDataset
        Dataset to review.  If ``cd._ccg_highres`` is populated
        (via ``cd.load_highres()``), a resolution toggle button is shown.
    key : Key
        Identifies which CCGPointer to review.
    """

    def __init__(self, cd, key):
        self.cd = cd

        self.key = key
        self.ccg_pointer = self.cd.data.get(key)
        self.ccg_data = self.cd._ccg.get(key.nd()) if getattr(self.cd, '_ccg', None) else None

        # Neurons (normalization + network + waveforms)
        self.neurons = (self.cd.nd.data[key.nd()]
                        if getattr(self.cd, 'nd', None) is not None
                           and self.ccg_pointer is not None
                        else None)

        # SelectionData owns all group/tag state; CCGReviewUI keeps same-object aliases
        # so existing code (undo, load, classify, etc.) that writes self._groups etc.
        # continues to work. Call _sync_sel_data() after any bulk reassignment.
        self._sel_data = SelectionData()
        self._groups         = self._sel_data._groups         # same dict object
        self._group_hotkeys  = self._sel_data._group_hotkeys  # same dict object
        self._group_notes    = self._sel_data._group_notes    # same dict object
        self._group_registry = self._sel_data._group_registry # same dict object
        self._next_group_id  = self._sel_data._next_group_id  # int — re-synced by _sync_sel_data
        self._pair_tags      = self._sel_data._pair_tags      # same dict object

        # Pair / segment state  (all_inds is a @property — see below)
        # These are set to empty defaults when data is not yet loaded; they are
        # refreshed in _finish_initial_draw() after lazy loading completes.
        if self.ccg_pointer is not None:
            self.n_segments = self.ccg_pointer.n_segments
            self.segment_names = list(self.ccg_pointer.edge_times['label'].values)
        else:
            self.n_segments = 0
            self.segment_names = []
        self.current_segment = self.n_segments   # default = All (n_segments sentinel)
        self.current_pair_idx = 0

        # Manual selection state
        if (self.ccg_pointer is not None
                and hasattr(self.ccg_pointer, 'manually_selected_inds')
                and self.ccg_pointer.manually_selected_inds is not None):
            self.selected_inds = set(
                map(tuple, self.ccg_pointer.manually_selected_inds))
        else:
            self.selected_inds = set()
        self.unselected_inds = set(map(tuple, self.all_inds)) - self.selected_inds
        self._sel_data.selected_inds   = self.selected_inds
        self._sel_data.unselected_inds = self.unselected_inds

        # Undo/redo stack for pair selection changes
        self._undo_stack: list = []  # list of (selected_inds_copy, unselected_inds_copy, deleted_inds_copy)
        self._redo_stack: list = []
        self._UNDO_LIMIT = 30

        # Session-only list markers (cleared on quit / Selections menu)
        self._bookmarked_pairs: set[tuple[int, int]] = set()

        # Active config
        self.active_alpha = getattr(getattr(self.cd, 'conf', None), 'alpha', 0.05)
        self.active_norms: set = set()
        self.norm_vars: dict = {}
        self.active_segment_filter = None

        # Resolution state
        self._highres_mode = False           # True when showing _ccg_highres
        self._sbs_mode = False               # True when showing lo/hi side-by-side
        # Per-item line/outline toggle — initialized after Tk root exists
        self._line_ccg_var = None
        self._line_baseline_var = None
        self._baseline_show_var = None
        self._baseline_style_btn = None
        self._line_ref_var = None
        self._line_tgt_var = None

        # Same-scale state
        self._same_scale_mode: str = None    # None | 'pair' | 'session'
        self._pair_scale_cache: dict = {}    # (ref, tgt) -> (ymin, ymax)
        self._session_scale_cache = None     # (ymin, ymax)

        # Jitter state — LRU cache with max size to avoid memory overflow
        self._JITTER_CACHE_MAX = 500         # max cached pairs before LRU eviction
        self._jitter_cache = collections.OrderedDict()  # (ref,tgt,res) -> (j_avg, j_pval, j_pval_bins)
        self._jitter_proc: _mp.Process = None            # currently running jitter process
        self._jitter_result_queue: _mp.Queue = None      # result queue from running jitter process
        self._jitter_pending: collections.deque = collections.deque()
        self._jitter_poll_id = None                      # after() id for polling
        self._jitter_unviewed: set = set()               # (ref, tgt) pairs with unviewed results
        self._jitter_mgr: 'JitterManager' = None         # lazy init after data loads
        # Custom CCG has its own independent queue/thread — never blocked by jitter
        self._custom_ccg_pending: collections.deque = collections.deque()
        self._custom_ccg_thread: threading.Thread = None
        self._custom_ccg_thread_result: list = []
        self._custom_ccg_poll_id = None
        # Multi-chunk time-slider splits: track batches so we can prompt to save all
        self._split_batch_next_id: int = 1
        self._split_batch_counts: dict[int, int] = {}
        self._split_batch_chunk_names: dict[int, list[str]] = {}

        # Extend-window CCG (tentative feature): cache per (pair, seg, res, ms)
        self._extend_cache: dict = {}

        # Display-only computed values for the current visual state (e.g., ACG deconvolution)
        self._display_pair_temp: dict = {}

        # Significance display toggles live in BooleanVars (created after Tk root).
        # Use _sig(name) helper to read them.

        # Double-click debounce
        self._select_after: int = None       # after() id for deferred pair update

        # Probe-network neuron/pair focus
        self._focused_neuron: int = None     # neuron index to highlight (None = off)
        self._focused_pair: tuple = None     # (ref, tgt) pair to highlight (None = off)
        self._net_show_arrows: bool = True   # show/hide connection arrows
        self._net_hide_unconnected: bool = False  # hide neurons with no connections
        self._net_group_filter_vars: dict = {}   # group_name -> BooleanVar (multi-select)
        self._net_grp_items: list = []           # (widget, is_separator) for wrapping layout

        # Pair list state
        self.deleted_inds: set = set()           # spurious/deleted pairs (shown grayed at bottom of Available)
        # Per connection-type (str(Key)) deleted sets — survives type switches + accurate saves
        self._pair_deleted_store: dict = {}

        # Versioned selections save dir
        self._sel_save_dir = str(
            _Path(__file__).resolve().parents[2] / "data" / "selections")
        os.makedirs(self._sel_save_dir, exist_ok=True)
        # Custom CCG cache dir
        self._ccg_cache_dir = str(
            _Path(__file__).resolve().parents[2] / "data" / "custom_ccg")
        os.makedirs(self._ccg_cache_dir, exist_ok=True)
        self._custom_ccg_suggestions_path = os.path.join(
            self._ccg_cache_dir, "suggested_custom_ccgs.json")
        # CCG classifier dir
        self._clf_dir = str(
            _Path(__file__).resolve().parents[2] / "data" / "ccg_classifier")
        os.makedirs(self._clf_dir, exist_ok=True)
        # Single shared templates file (session/conn_type agnostic)
        self._templates_path = os.path.join(self._clf_dir, "templates.json")
        # Speculated labels: {(ref,tgt) → ClassifyResult} — never touches _groups
        self._speculated_groups: dict = {}
        # Template classifier: {group_name → GroupTemplate} — editable via Classify menu
        self._templates: dict = {}
        self._templates_smooth_ms: float = 2.0
        # Set of template names enabled for auto-classify (None = all enabled)
        self._active_templates: set = set()
        # Auto-load templates from the shared file if it exists
        self._autoload_templates()

        # UI settings (persisted in ui_state.json alongside panel state)
        _raw_ui_state = self._load_ui_state()
        self._ui_state_cache = _raw_ui_state   # reused by setup_panels_menu
        _settings_defaults = {'max_show_together': 5}
        self._settings: dict = {**_settings_defaults,
                                 **_raw_ui_state.get('settings', {})}
        # Cache configuration — the ONE display state that is saved to disk PNG cache.
        # Any other display state is rendered in real-time (not cached).
        # None = no config set (legacy behavior: cache all states).
        self._cache_config: dict | None = _raw_ui_state.get('cache_config', None)

        # "Show together" — list of (ref, tgt) pairs to display stacked
        self._together_pairs: list = []

        # Panel visibility state
        self._panel_vars: dict = {}          # populated in setup_panels_menu
        self._waveforms_visible = False

        # Time-slider / custom-segment state
        self._slider_t_start: float = None
        self._slider_t_end: float = None
        self._slider_dragging: str = None    # 'start' | 'end' | None
        self._ts_epoch_bounds = []           # [(t0_sec, t1_sec, label), …]
        self._ts_total_sec: float = 0.0
        self._ts_active_label: str = None    # label filter for overlapping themes (None = show all)
        self._ts_segment_name: str = ""      # name for the next custom segment
        # Custom segments: each entry =
        #   {'name':str, 't0':float, 't1':float,
        #    'ccg': [1,N,N,bins], 'ccg_null', 'pval', 'pval_corrected',
        #    (optional hi-res): 'ccg_hi', 'ccg_null_hi', 'pval_hi', 'pval_corrected_hi'}
        # Per-session buckets (All-session mode); ``_custom_segments`` aliases the active one.
        self._custom_segments_by_session: dict[str, list] = {}
        self._custom_segments: list = []
        self._bind_custom_segments_to_session(str(self.key.session))
        # Multi-select segment chips for stacked display
        self._stacked_segments: set[int] = set()
        self._stats_panel = None          # StatsTestPanel singleton
        self._sel_collapsed_headers: set = set()  # header texts that are currently collapsed
        # Virtual "All" session: union across dataset (pairs are (sess, ref, tgt) triples).
        self._session_any_mode: bool = False
        self._any_expanded_group_tags: set[str] = set()
        # Ordered (Key, ref, tgt) for navigation + plot — built in refresh_lists.
        self._any_pair_handle_list: list[tuple] = []
        self._png_sess_slug: str = ""
        self._custom_ccg_inventory_sig: tuple = ()

        # Time slider theme state
        self._ts_themes: dict = {}            # name -> Epoch object
        self._ts_current_theme: str = 'segments'  # default = CCG segment edges

        # Zoom/selection state (selection tool cursors, independent of custom-window cursors)
        self._ts_zoom_start: float = None
        self._ts_zoom_end: float = None
        self._ts_zoom_dragging: str = None   # 'start' | 'end' | None

        # Spike attribution state
        self._sa_enabled = False              # toggle state
        self._sa_bin_ms: float = 0.0          # target bin lag in ms
        self._sa_spike_pairs: list = []       # [(ref_spike_t, tgt_spike_t), ...]
        self._sa_selected_idx: int = -1       # selected spike pair index
        self._sa_raster_window: float = 0.050 # ±50 ms raster window (seconds)

        # PNG cache
        self.tmp_dir = str(
            _Path(__file__).resolve().parents[2] / "images" / "tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self._pregen_cancel = False
        self._pregen_thread: threading.Thread = None
        self._pregen_proc: subprocess.Popen | None = None
        self._pregen_poll_id = None   # after() id for subprocess status polling
        self._pregen_priority: str | None = None   # 'auto' or 'user'
        self._auto_pregen_enabled: bool = _raw_ui_state.get('auto_pregen_enabled', False)

        # Build UI — use Toplevel if a Tk root already exists (avoids
        # multiple Tk() instances which cause event loop conflicts).
        self._owns_mainloop = False
        try:
            existing = tk._default_root  # noqa: access internal
        except AttributeError:
            existing = None
        if existing is not None and existing.winfo_exists():
            self.root = tk.Toplevel(existing)
        else:
            self.root = tk.Tk()
            self._owns_mainloop = True
        self.root.title("CCG Manual Review")
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._set_window_icon()
        # Per-item line/outline toggle (False = filled bar, True = step outline)
        self._line_ccg_var = tk.BooleanVar(master=self.root, value=False)
        # Baseline (conv null): False = shaded bars, True = dashed step line
        self._line_baseline_var = tk.BooleanVar(master=self.root, value=False)
        self._baseline_show_var = tk.BooleanVar(master=self.root, value=True)
        self._line_ref_var = tk.BooleanVar(master=self.root, value=True)
        self._line_tgt_var = tk.BooleanVar(master=self.root, value=True)
        self._line_peak_wf_var = tk.BooleanVar(master=self.root, value=True)
        self._peak_wf_var = tk.BooleanVar(master=self.root, value=False)
        self._line_jitter_var = tk.BooleanVar(master=self.root, value=False)
        # Connectivity strength toggles
        self._conn_str_show_var = tk.BooleanVar(master=self.root, value=False)
        self._conn_str_nonneg_var = tk.BooleanVar(master=self.root, value=False)
        self._conn_str_metric_var = tk.StringVar(
            master=self.root,
            value=self._ui_state_cache.get('conn_str_metric', 'STG'))
        self._conn_str_method_var = tk.StringVar(
            master=self.root,
            value=self._ui_state_cache.get('conn_str_method', 'conv'))
        self._conn_strength_cache: dict = {}
        self._adaptive_tw_var = tk.BooleanVar(master=self.root, value=False)
        # Jitter run resolution toggles (used by GUI run-jitter button)
        self._jitter_run_lo_var = tk.BooleanVar(
            master=self.root,
            value=bool(self._ui_state_cache.get('jitter_run_lo', True)))
        self._jitter_run_hi_var = tk.BooleanVar(
            master=self.root,
            value=bool(self._ui_state_cache.get('jitter_run_hi', False)))
        # Main template — built once at startup and injected into _templates
        self._main_template = None
        self._build_main_template()
        # Heartbeat: keep event loop responsive even when Jupyter cell finishes
        self._heartbeat_id = None
        self._closing = False
        self._start_heartbeat()
        self.setup_ui()
        # Apply any persisted font sizing constraints after widgets exist
        self._apply_min_font_size()
        self._emit_custom_ccg_inventory_event()

    # ------------------------------------------------------------------
    # Derived state
    # ------------------------------------------------------------------

    @property
    def _res_key(self):
        """Current resolution key for cache keying ('hi' or 'lo')."""
        return 'hi' if getattr(self, '_highres_mode', False) else 'lo'

    @property
    def _session_all_mode(self) -> bool:
        return bool(getattr(self, '_session_any_mode', False))

    @_session_all_mode.setter
    def _session_all_mode(self, value: bool):
        self._session_any_mode = bool(value)

    @property
    def _all_in1_expanded_group_tags(self) -> set:
        return getattr(self, '_any_expanded_group_tags', set())

    @_all_in1_expanded_group_tags.setter
    def _all_in1_expanded_group_tags(self, value: set):
        self._any_expanded_group_tags = set(value)

    @property
    def _all_in1_pair_handle_list(self) -> list:
        return getattr(self, '_any_pair_handle_list', [])

    @_all_in1_pair_handle_list.setter
    def _all_in1_pair_handle_list(self, value: list):
        self._any_pair_handle_list = list(value)

    @property
    def all_inds(self):
        """Significant pairs + manually admitted pairs, as Nx2 numpy array.

        Autocorrelograms (ref == tgt) are always excluded.
        Returns an empty (0,2) array when data is not yet loaded.
        """
        if getattr(self, '_session_any_mode', False):
            hl = getattr(self, '_any_pair_handle_list', None) or []
            if not hl:
                return np.empty((0, 2), dtype=int)
            return np.array([[int(r), int(t)] for _, r, t in hl], dtype=int)
        if self.ccg_pointer is None:
            return np.empty((0, 2), dtype=int)
        base = self.ccg_pointer.inds2
        # Filter out self-pairs (autocorrelograms)
        mask = base[:, 0] != base[:, 1]
        base = base[mask]
        admitted = self._group_pairs(_ADMITTED_GROUP)
        if not admitted:
            return base
        base_set = set(map(tuple, base))
        extra = sorted(set(tuple(p) for p in admitted if p[0] != p[1]) - base_set)
        if not extra:
            return base
        return np.vstack([base, np.array(extra, dtype=base.dtype)])

    def _all_inds_set_for_ptr(self, ptr) -> set:
        """set of (ref, tgt) for a CCGPointer — same rules as all_inds, without self.ccg_pointer."""
        if ptr is None or ptr.inds2 is None:
            return set()
        base = ptr.inds2
        base = base[base[:, 0] != base[:, 1]]
        s = set(map(tuple, base))
        admitted = self._group_pairs(_ADMITTED_GROUP)
        if not admitted:
            return s
        base_set = set(s)
        extra = sorted(set(tuple(p) for p in admitted if p[0] != p[1]) - base_set)
        if not extra:
            return s
        return s | set(extra)

    # ------------------------------------------------------------------
    # Per-session group helpers
    # ------------------------------------------------------------------

    def _set_window_icon(self):
        """Create a 'CCG' text icon and set it as the window icon."""
        try:
            size = 64
            img = Image.new('RGBA', (size, size), (30, 40, 80, 255))
            draw = ImageDraw.Draw(img)
            # Try to use a bold font; fall back to default
            try:
                font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 26)
            except Exception:
                try:
                    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
                except Exception:
                    font = ImageFont.load_default()
            text = 'CCG'
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]),
                      text, fill=(255, 220, 80, 255), font=font)
            # Convert to tk.PhotoImage via PNG bytes
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            photo = tk.PhotoImage(data=buf.read())
            self.root.iconphoto(True, photo)
            self._icon_photo = photo   # keep reference
        except Exception:
            pass   # icon is optional — never break startup

    def _current_session_str(self):
        return getattr(self.key, 'session', 'sess')

    def _autoload_templates(self):
        """Load templates from the shared templates.json if it exists."""
        try:
            loaded = CCGTemplateClassifier.load_templates_from_file(
                self._templates_path)
            if loaded:
                self._templates.update(loaded)
                print(f"[CCGReviewUI] loaded {len(loaded)} template(s) "
                      f"from {self._templates_path}")
            meta = CCGTemplateClassifier.load_file_metadata(self._templates_path)
            if meta.get('smooth_ms') is not None:
                self._templates_smooth_ms = meta['smooth_ms']
            if meta.get('classify_with') is not None:
                self._active_templates = set(meta['classify_with'])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Group registry helpers (v4.0 schema)
    # ------------------------------------------------------------------

    def _ensure_group_registered(self, name: str) -> int:
        """Return the int ID for group name, creating an entry if new."""
        for gid, g in self._group_registry.items():
            if g['name'] == name:
                return gid
        gid = self._next_group_id
        self._group_registry[gid] = {
            'name': name,
            'hotkey': self._group_hotkeys.get(name),
            'notes': self._group_notes.get(name, ''),
        }
        self._next_group_id += 1
        return gid

    def _group_id_for(self, name: str) -> int | None:
        """Return int ID for group name, or None if not registered."""
        for gid, g in self._group_registry.items():
            if g['name'] == name:
                return gid
        return None

    def _sync_registry_from_groups(self):
        """Ensure every group in self._groups has a registry entry."""
        for name in list(self._groups.keys()):
            self._ensure_group_registered(name)
        # Sync hotkeys/notes into registry
        for gid, g in self._group_registry.items():
            name = g['name']
            g['hotkey'] = self._group_hotkeys.get(name)
            g['notes'] = self._group_notes.get(name, '')

    def _atomic_write_json(self, path: str, data: dict):
        """Write JSON atomically via a .tmp file, preventing partial writes."""
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=self._json_default)
        os.replace(tmp, path)

    def _group_pairs(self, gname, session=None):
        """Return pairs set for group in the given session (default: current)."""
        g = self._groups.get(gname, {})
        if isinstance(g, set):
            return g  # legacy flat format
        sess = session or self._current_session_str()
        return g.get(sess, set())

    def _group_pairs_all_sessions(self, gname):
        """Return all pairs across all sessions for a group."""
        g = self._groups.get(gname, {})
        if isinstance(g, set):
            return g
        all_pairs = set()
        for pairs in g.values():
            all_pairs |= pairs
        return all_pairs

    def _group_add_pair(self, gname, pair, session=None):
        sess = session or self._current_session_str()
        self._groups.setdefault(gname, {}).setdefault(sess, set()).add(pair)

    def _group_discard_pair(self, gname, pair, session=None):
        sess = session or self._current_session_str()
        print(f"[CCGReviewUI] group_discard: {gname!r} -= {pair} @ session={sess!r}")
        g = self._groups.get(gname, {})
        if isinstance(g, set):
            g.discard(pair)
        elif sess in g:
            g[sess].discard(pair)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def setup_ui(self):
        self.root.geometry("1800x950")

        # ── Menubar ────────────────────────────────────────────────────
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        self.setup_panels_menu(menubar)
        self.setup_groups_menu(menubar)
        self.setup_classify_menu(menubar)
        self.setup_file_menu(menubar)
        self.setup_modules_menu(menubar)
        self.setup_settings_menu(menubar)
        self.setup_help_menu(menubar)

        # ── Tool-strip row ─────────────────────────────────────────────
        self.setup_menu()

        # ── Group hotkeys bar (below tool-strip, hidden by default) ────
        self.setup_group_hotkeys_bar()

        # ── Bottom bar (packed before main so it gets space first) ─────
        self.setup_bottom_panel()

        # ── Main area ──────────────────────────────────────────────────
        self._main_frame = ttk.Frame(self.root)
        self._main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        # Time slider (full-width, hidden by default; packed before paned)
        self.setup_time_slider_panel(self._main_frame)

        # Three-column PanedWindow
        self._paned = ttk.PanedWindow(self._main_frame, orient=tk.HORIZONTAL)
        self._paned.pack(fill=tk.BOTH, expand=True)

        self._left_frame = ttk.Frame(self._paned, width=350)
        self._center_frame = ttk.Frame(self._paned)
        self._right_frame = ttk.Frame(self._paned, width=340)

        self._paned.add(self._left_frame, weight=0)
        self._paned.add(self._center_frame, weight=1)
        self._paned.add(self._right_frame, weight=0)

        self.setup_left_panel(self._left_frame)
        self.setup_center_panel(self._center_frame)
        self.setup_network_panel(self._right_frame)

        # Apply saved panel states — panels are added above with their defaults;
        # call _toggle_panel_impl for any panel whose saved state differs so the
        # actual visibility matches what was persisted.
        _panel_defaults = {
            'Pair Selection': True, 'CCG': True, 'Probe Network': True,
            'Waveforms': False, 'Time Slider': False, 'Group Hotkeys': True,
        }
        for _pname, _pdefault in _panel_defaults.items():
            _saved_val = self._panel_vars[_pname].get()
            if _saved_val != _pdefault:
                self._toggle_panel_impl(_pname)

        # Keyboard bindings (Control for Linux/Windows; Command for macOS)
        self.root.bind('<Left>',      lambda e: self.change_segment(-1))
        self.root.bind('<Right>',     lambda e: self.change_segment(1))
        for _key in ('<Control-r>', '<Command-r>'):
            self.root.bind(_key, lambda e: self._toggle_resolution())
        for _key in ('<Control-e>', '<Command-e>'):
            self.root.bind(_key, lambda e: self._on_ctrl_e())
        for _key in ('<Control-l>', '<Command-l>'):
            self.root.bind(_key, lambda e: self._toggle_plot_style())
        for _key in ('<Control-s>', '<Command-s>'):
            self.root.bind(_key, lambda e: self._quick_save())
        for _key in ('<Control-f>', '<Command-f>'):
            self.root.bind(_key, lambda e: self.left_container.left_panel._search_toggle())
        for _key in ('<Control-b>', '<Command-b>'):
            self.root.bind(_key, lambda e: self._bookmark_toggle_current())
        for _key in ('<Control-z>', '<Command-z>'):
            self.root.bind(_key, self._undo)
        for _key in ('<Control-y>', '<Command-y>',
                      '<Control-Shift-z>', '<Command-Shift-z>',
                      '<Control-Shift-Z>', '<Command-Shift-Z>'):
            self.root.bind(_key, self._redo)
        for _del_key in ('<Control-Delete>', '<Control-BackSpace>',
                         '<Command-Delete>', '<Command-BackSpace>',
                         '<Meta-Delete>',    '<Meta-BackSpace>'):
            self.root.bind(_del_key, lambda e: self._on_delete_pair())
        # Digit keys: bare digit tags group + advances; Shift+digit tags group only
        # (no advance), so holding Shift and pressing multiple digits assigns all
        # those groups to the current pair.
        _SHIFT_DIGIT = {
            'exclam': '1', 'at': '2', 'numbersign': '3', 'dollar': '4', 'percent': '5',
            'asciicircum': '6', 'ampersand': '7', 'asterisk': '8', 'parenleft': '9',
            'parenright': '0',
        }

        def _global_key_handler(e):
            if isinstance(e.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox,
                                     tk.Text)):
                return
            # Don't interpret modified keystrokes (Ctrl/Cmd/Alt) as group hotkeys.
            # This prevents collisions with Ctrl+B (bookmark), Ctrl+S, etc.
            st = getattr(e, 'state', 0) or 0
            if st & 0x0004 or st & 0x0008 or st & 0x00020000:
                return
            ks = e.keysym
            if ks in ('Delete', 'BackSpace'):
                self._on_delete_pair()
                return
            if ks in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0'):
                self._group_hotkey_handler(ks, advance=True)
            elif ks in ('KP_1', 'KP_2', 'KP_3', 'KP_4', 'KP_5',
                        'KP_6', 'KP_7', 'KP_8', 'KP_9'):
                self._group_hotkey_handler(ks[-1], advance=True)
            elif ks == 'KP_0':
                self._group_hotkey_handler('0', advance=True)
            elif ks in _SHIFT_DIGIT:
                # Shift+digit → no advance
                self._group_hotkey_handler(_SHIFT_DIGIT[ks], advance=False)
            elif len(ks) == 1 and ks.islower():
                # Bare letter → assign + advance
                self._group_hotkey_handler(ks, advance=True)
            elif len(ks) == 1 and ks.isupper():
                # Shift+letter → assign, no advance
                self._group_hotkey_handler(ks.lower(), advance=False)

        self.root.bind('<KeyPress>', _global_key_handler)
        # When holding Shift to apply multiple tags without advancing, advance
        # once when Shift is released.
        self._shift_tag_pending_advance = False
        self.root.bind('<KeyRelease-Shift_L>', lambda e: self._on_shift_release_advance())
        self.root.bind('<KeyRelease-Shift_R>', lambda e: self._on_shift_release_advance())

        # Restore segment and sash position after the window is realized
        self.root.after(200, self._restore_deferred_ui_state)

    def _dbg_log(self, hypothesis_id: str, location: str, message: str, data: dict):
        """Debug-mode NDJSON logger (session cde521)."""
        # #region agent log
        try:
            import json as _json
            import os as _os
            import time as _t
            payload = {
                "sessionId": "cde521",
                "runId": "pre-fix",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(_t.time() * 1000),
            }
            _path = "/Users/selinl/Documents/ms_synchrony/NeuroPy/.cursor/debug-cde521.log"
            _os.makedirs(_os.path.dirname(_path), exist_ok=True)
            with open(_path, "a", encoding="utf-8") as _f:
                _f.write(_json.dumps(payload) + "\n")
        except Exception as _ex:
            try:
                print(f"[DBG_LOG] failed: {location}: {_ex}")
            except Exception:
                pass
        # #endregion agent log

    def _on_shift_release_advance(self):
        """Advance one pair after a multi-hotkey (Shift-held) tagging sequence."""
        if not getattr(self, '_shift_tag_pending_advance', False):
            return
        self._shift_tag_pending_advance = False
        try:
            if self.current_pair_idx >= len(self.all_inds):
                return
            next_idx = min(self.current_pair_idx + 1, len(self.all_inds) - 1)
            if next_idx != self.current_pair_idx:
                self.current_pair_idx = next_idx
                self._select_pair_in_list(tuple(self.all_inds[next_idx]))
                self.update_plot()
                self._draw_network()
        except Exception:
            pass

    def _restore_deferred_ui_state(self):
        """Restore state that can only be applied after the window is mapped."""
        s = self._ui_state_cache

        # Restore loaded custom CCGs first (so segment indices exist before we restore current_segment)
        try:
            self._restore_loaded_custom_ccgs_from_state()
        except Exception:
            pass

        # Restore sash position
        pane = getattr(self, '_pair_list_pane', None)
        if pane is not None:
            sp = s.get('sash_pos')
            if sp is not None:
                try:
                    pane.sashpos(0, int(sp))
                except Exception:
                    pass

        # Restore current segment (only if valid for the loaded session)
        saved_seg = s.get('current_segment')
        if saved_seg is not None:
            try:
                total = self._n_total_segments()
                if 0 <= int(saved_seg) < total:
                    self.current_segment = int(saved_seg)
                    self._update_segment_label()
            except Exception:
                pass

        # Restore core display vars (baseline solid/outline, test window, etc.)
        disp = s.get('display_vars', None)
        if isinstance(disp, dict):
            try:
                for attr, val in disp.items():
                    v = getattr(self, attr, None)
                    if v is None:
                        continue
                    try:
                        v.set(val)
                    except Exception:
                        pass
            except Exception:
                pass

        # Restore resolution / side-by-side mode (these are not Tk variables)
        try:
            self._highres_mode = bool(s.get('highres_mode', getattr(self, '_highres_mode', False)))
        except Exception:
            pass
        try:
            self._sbs_mode = bool(s.get('sbs_mode', getattr(self, '_sbs_mode', False)))
        except Exception:
            pass
        # Ensure ccg_data matches restored resolution
        try:
            nd_key = self.key.nd()
            if (getattr(self, '_highres_mode', False)
                    and hasattr(self.cd, '_ccg_highres')
                    and self.cd._ccg_highres.get(nd_key) is not None):
                self.ccg_data = self.cd._ccg_highres.get(nd_key)
            else:
                self.ccg_data = self.cd._ccg.get(nd_key) if getattr(self.cd, '_ccg', None) else self.ccg_data
        except Exception:
            pass

        # Re-sync active_norms from restored norm_vars (they were set from saved state)
        if hasattr(self, 'norm_vars'):
            self.active_norms = {nm for nm, var in self.norm_vars.items() if var.get()}

        # Finally, re-render so main-panel button states are reflected visually.
        try:
            self.update_plot()
        except Exception:
            pass

    def _restore_loaded_custom_ccgs_from_state(self):
        """Reload custom CCG .npz files listed in ui_state.json (additively)."""
        paths = self._ui_state_cache.get('loaded_custom_ccgs', []) or []
        if not paths:
            return

        session = str(self.key.session)

        added = []
        added_active_view = False
        for p in paths:
            if not isinstance(p, str) or not p:
                continue
            try:
                base = os.path.basename(p)
                file_sess = base.split("__", 1)[0] if "__" in base else session
                if not os.path.exists(p):
                    continue
                npz = np.load(p, allow_pickle=False)
                cs = dict(
                    name=str(npz['name_']),
                    t0=float(npz['t0_']),
                    t1=float(npz['t1_']),
                    ccg=npz['ccg'],
                    ccg_null=npz['ccg_null'],
                    pval=npz['pval'],
                    pval_corrected=npz['pval_corrected'],
                    compute_sec=(float(npz['compute_sec_'])
                                 if 'compute_sec_' in npz else float('nan')),
                    active_duration=(float(npz['active_duration_'])
                                     if 'active_duration_' in npz else float('nan')),
                    total_time_hours=(float(npz['total_time_hours_'])
                                      if 'total_time_hours_' in npz else None),
                    filter_state=(json.loads(str(npz['filter_state_']))
                                  if 'filter_state_' in npz else {}),
                    metadata=(json.loads(str(npz['metadata_']))
                              if 'metadata_' in npz else {}),
                    src_path=p,
                    **(({'firing_rates': npz['firing_rates']})
                       if 'firing_rates' in npz else {}),
                )
                for k in ('ccg_hi', 'ccg_null_hi', 'pval_hi', 'pval_corrected_hi'):
                    if k in npz:
                        cs[k] = npz[k]
                lst = self._custom_segments_by_session.setdefault(file_sess, [])
                self._upsert_custom_segment_by_name(lst, cs)
                if lst is self._custom_segments:
                    added_active_view = True
                added.append(cs['name'])
            except Exception as ex:
                print(f"[CCGReviewUI] restore custom CCG failed: {p}: {ex}")

        if added and added_active_view:
            try:
                self._build_sig_chips()
                self._update_segment_label()
                self.update_plot()
            except Exception:
                pass

    # ── Menubar ────────────────────────────────────────────────────────

    def _ui_state_path(self):
        return os.path.join(self._sel_save_dir, 'ui_state.json')

    def _load_ui_state(self) -> dict:
        """Return full saved UI state dict, or {} if not found/invalid."""
        try:
            with open(self._ui_state_path(), 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_ui_state(self):
        """Persist full app state (panel visibility, norms, segment, sort, etc.) to ui_state.json."""
        try:
            key_dict = {f: getattr(self.key, f) for f in self.key.__dataclass_fields__}
            if key_dict.get('conn_type') is not None:
                key_dict['conn_type'] = list(key_dict['conn_type'])

            # ── Normalization ──
            active_norms = sorted(nm.name for nm in getattr(self, 'active_norms', set()))

            # ── Sort options ──
            sort_sel  = getattr(self, '_sort_selected_var', None)
            sort_tag  = getattr(self, '_sort_by_tag_var',   None)
            sort_mean = getattr(self, '_sort_by_mean_var',  None)
            sort_minp = getattr(self, '_sort_by_min_p_var', None)

            # ── Scale options ──
            pair_scale = getattr(self, '_pair_scale_var', None)
            sess_scale = getattr(self, '_sess_scale_var', None)

            # ── Sash position (Available / Selected splitter) ──
            sash_pos = None
            pane = getattr(self, '_pair_list_pane', None)
            if pane is not None:
                try:
                    sash_pos = pane.sashpos(0)
                except Exception:
                    pass

            state = {
                'panels':              {n: v.get() for n, v in self._panel_vars.items()},
                'settings':            self._settings,
                'cache_config':        self._cache_config,
                'auto_pregen_enabled': self._auto_pregen_enabled,
                'last_key':            key_dict,
                # ── new full-snapshot fields ──
                'active_norms':        active_norms,
                'conn_str_method':     (self._conn_str_method_var.get()
                                        if hasattr(self, '_conn_str_method_var') else 'conv'),
                'conn_str_metric':     (self._conn_str_metric_var.get()
                                        if hasattr(self, '_conn_str_metric_var') else 'STG'),
                'jitter_run_lo':       (self._jitter_run_lo_var.get()
                                        if hasattr(self, '_jitter_run_lo_var') else True),
                'jitter_run_hi':       (self._jitter_run_hi_var.get()
                                        if hasattr(self, '_jitter_run_hi_var') else False),
                'current_segment':     self.current_segment,
                'sort_selected':       sort_sel.get()  if sort_sel  else False,
                'sort_by_tag':         sort_tag.get()  if sort_tag  else False,
                'sort_by_mean':        sort_mean.get() if sort_mean else False,
                'sort_by_min_p':       sort_minp.get() if sort_minp else False,
                'pair_scale':          pair_scale.get() if pair_scale else False,
                'sess_scale':          sess_scale.get() if sess_scale else False,
                'sash_pos':            sash_pos,
                # Core display toggles (needed for consistent relaunch rendering)
                'display_vars':        self._current_display_config(),
                'highres_mode':        bool(getattr(self, '_highres_mode', False)),
                'sbs_mode':            bool(getattr(self, '_sbs_mode', False)),
                # Loaded custom CCGs (saved .npz paths) for this session/conn_type
                'loaded_custom_ccgs':  [
                    cs.get('src_path')
                    for lst in getattr(self, '_custom_segments_by_session', {}).values()
                    for cs in lst
                    if isinstance(cs, dict) and cs.get('src_path')
                ],
            }
            with open(self._ui_state_path(), 'w') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def setup_panels_menu(self, menubar):
        """Panels menu with checkbuttons for each panel."""
        panels_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Panels", menu=panels_menu)
        self._panels_menu = panels_menu
        panel_defaults = [
            ('Pair Selection',  True),
            ('CCG',             True),
            ('Probe Network',   True),
            ('Waveforms',       False),
            ('Time Slider',     False),
            ('Group Hotkeys',   True),
        ]
        saved = self._ui_state_cache.get('panels', None)
        for name, default in panel_defaults:
            value = saved.get(name, default) if saved is not None else default
            var = tk.BooleanVar(value=value)
            self._panel_vars[name] = var
            panels_menu.add_checkbutton(
                label=name, variable=var,
                command=lambda n=name: self._toggle_panel(n))
        panels_menu.add_separator()
        ts_menu = tk.Menu(panels_menu, tearoff=0)
        ts_menu.add_command(
            label="Refresh suggested custom CCGs",
            command=self._refresh_custom_ccg_suggestions,
        )
        ts_menu.add_command(
            label="Generate suggested custom CCGs",
            command=self._generate_suggested_custom_ccgs,
        )
        panels_menu.add_cascade(label="Time Slider Actions", menu=ts_menu)

    def setup_groups_menu(self, menubar):
        """Groups menu: create / manage pair groups."""
        self._groups_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Groups", menu=self._groups_menu)
        self._groups_menu.add_command(label="Create group…",
                                      command=self._create_group_dialog)
        self._groups_menu.add_command(label="Create special group…",
                                      command=self._create_special_group_dialog)
        self._groups_menu.add_command(label="Manage groups…",
                                      command=self._manage_groups_dialog)
        self._groups_menu.add_command(label="Merge groups…",
                                      command=self._merge_groups_dialog)
        self._groups_menu.add_command(label="Export groups…",
                                      command=self._export_groups)
        self._groups_menu.add_command(label="Import groups…",
                                      command=self._import_groups)
        self._groups_menu.add_separator()
        self._groups_menu.add_command(label="Pair tags…",
                                      command=self._pair_tags_dialog)
        self._groups_menu.add_separator()
        # Dynamic group entries added in _rebuild_groups_menu()

    def setup_file_menu(self, menubar):
        """Selections menu: save / load selection versions."""
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Selections", menu=file_menu)
        file_menu.add_command(label="Save selection…",
                              command=self._quick_save)
        file_menu.add_command(label="Load selection…",
                              command=self._load_selection_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Export as PNG…",
                              command=lambda: self._export_current_view('png'))
        file_menu.add_command(label="Export as PDF…",
                              command=lambda: self._export_current_view('pdf'))
        file_menu.add_separator()
        file_menu.add_command(label="Clear bookmarks",
                              command=self._clear_bookmarks)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self._selections_menu_close)

    def _selections_menu_close(self):
        self._bookmarked_pairs.clear()
        self.root.destroy()

    def setup_settings_menu(self, menubar):
        """Settings menu — opens the settings dialog."""
        m = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=m)
        m.add_command(label="Settings…", command=self._settings_dialog)

    def setup_classify_menu(self, menubar):
        """Classify menu for template-based classification of CCG pairs."""
        m = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Classify", menu=m)
        m.add_command(label="Edit templates…", command=self._template_editor_dialog)
        m.add_separator()
        m.add_command(label="Auto-classify…",  command=self._auto_classify_dialog)
        m.add_command(label="Clear speculated", command=self._clear_speculated)

    def setup_modules_menu(self, menubar):
        """Modules menu — Stats Tests, Jitter and Simulation sub-menus."""
        modules_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Modules", menu=modules_menu)

        # Stats Tests — prepended first
        modules_menu.add_command(label="Stats Tests…", command=self._open_stats_panel)

        # Jitter sub-menu
        jitter_menu = tk.Menu(modules_menu, tearoff=0)
        modules_menu.add_cascade(label="Jitter", menu=jitter_menu)
        jitter_menu.add_command(label="View queue…", command=self._jitter_queue_dialog)
        jitter_menu.add_command(label="Clear queue",  command=self._jitter_clear_queue)

        # Simulation sub-menu
        sim_menu = tk.Menu(modules_menu, tearoff=0)
        modules_menu.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(label="New simulation…", command=self._simulation_dialog)

        # Tooltip-style help on hover
        self._modules_menu = modules_menu
        self._menu_tooltips = {
            'Stats Tests…': "Run statistical comparisons (t-tests) on connection strengths across groups/segments",
            'Jitter': "Significance test for a pairwise connection using surrogate data",
            'Simulation': "Simulate CCG of two random neurons with designated properties",
        }
        self._menu_tooltip_win = None
        modules_menu.bind('<<MenuSelect>>', self._on_modules_menu_hover)

    def _open_stats_panel(self):
        """Open (or raise) the Stats Tests panel."""
        from neuropy.ui.stats_panel import StatsTestPanel  # noqa: PLC0415
        if self._stats_panel is None or not self._stats_panel.root.winfo_exists():
            self._stats_panel = StatsTestPanel(self)
        else:
            self._stats_panel.root.lift()
            try:
                self._stats_panel.refresh_session_dropdowns()
            except Exception:
                pass

    def _on_modules_menu_hover(self, event):
        """Show tooltip for hovered Modules menu item."""
        menu = self._modules_menu
        try:
            idx = menu.index('active')
            if idx is not None:
                label = menu.entrycget(idx, 'label')
                tip = self._menu_tooltips.get(label)
                if tip:
                    if self._menu_tooltip_win is not None:
                        self._menu_tooltip_win.destroy()
                    tw = tk.Toplevel(self.root)
                    tw.wm_overrideredirect(True)
                    tw.wm_attributes('-topmost', True)
                    x = self.root.winfo_pointerx() + 16
                    y = self.root.winfo_pointery() + 10
                    tw.wm_geometry(f"+{x}+{y}")
                    lbl = tk.Label(tw, text=tip, background='#ffffe0',
                                   relief=tk.SOLID, borderwidth=1,
                                   font=('TkDefaultFont', 9), padx=6, pady=3)
                    lbl.pack()
                    self._menu_tooltip_win = tw
                    return
        except (tk.TclError, ValueError):
            pass
        if self._menu_tooltip_win is not None:
            self._menu_tooltip_win.destroy()
            self._menu_tooltip_win = None

    def _simulation_dialog(self):
        """Open the neuron-pair CCG simulation dialog."""
        win = tk.Toplevel(self.root)
        win.title("CCG Simulation")
        win.geometry("620x780")
        win.transient(self.root)

        # Vertical PanedWindow: top = params, bottom = CCG result
        pw = tk.PanedWindow(win, orient=tk.VERTICAL,
                            sashrelief=tk.RAISED, sashwidth=5, bg='#CCCCCC')
        pw.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ── Top pane: parameters ──
        param_frame = ttk.Frame(pw, padding=6)
        pw.add(param_frame, stretch='never')

        # Name + duration row
        top = ttk.Frame(param_frame)
        top.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(top, text="Name:").pack(side=tk.LEFT)
        sim_name_var = tk.StringVar(value="sim1")
        ttk.Entry(top, textvariable=sim_name_var, width=20).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(top, text="Duration:").pack(side=tk.LEFT)
        sim_dur_var = tk.StringVar(value="60")
        ttk.Entry(top, textvariable=sim_dur_var, width=8).pack(side=tk.LEFT, padx=(4, 4))
        sim_dur_unit = tk.StringVar(value="s")
        ttk.OptionMenu(top, sim_dur_unit, "s", "ms", "s", "min", "hour").pack(side=tk.LEFT)

        # Common parameters row
        common = ttk.Frame(param_frame)
        common.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(common, text="Noise (gauss \u03c3):").pack(side=tk.LEFT)
        sim_noise_var = tk.StringVar(value="0.0")
        ttk.Entry(common, textvariable=sim_noise_var, width=7).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(common, text="Excess synchrony (%):").pack(side=tk.LEFT)
        sim_sync_var = tk.StringVar(value="0")
        ttk.Entry(common, textvariable=sim_sync_var, width=7).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(common, text="Synaptic delay (ms):").pack(side=tk.LEFT)
        sim_delay_var = tk.StringVar(value="1.5")
        ttk.Entry(common, textvariable=sim_delay_var, width=7).pack(side=tk.LEFT, padx=(2, 0))

        # Two-column neuron panels
        cols = ttk.Frame(param_frame)
        cols.pack(fill=tk.X, pady=(0, 6))
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)

        sim_vars = {}
        for col_idx, (role, title) in enumerate([('ref', 'Ref neuron'), ('tgt', 'Tgt neuron')]):
            panel = ttk.LabelFrame(cols, text=title, padding=8)
            panel.grid(row=0, column=col_idx, sticky='nsew', padx=(0 if col_idx == 0 else 4, 0))

            v = {}
            ttk.Label(panel, text="Nickname:").pack(anchor='w')
            v['nickname'] = tk.StringVar(value=role)
            ttk.Entry(panel, textvariable=v['nickname'], width=16).pack(anchor='w', pady=(0, 6))

            ttk.Label(panel, text="Type:").pack(anchor='w')
            v['type'] = tk.StringVar(value="E")
            type_frame = ttk.Frame(panel)
            type_frame.pack(anchor='w', pady=(0, 6))
            for t in ("E", "I", "any"):
                ttk.Radiobutton(type_frame, text=t, variable=v['type'], value=t).pack(side=tk.LEFT, padx=(0, 6))

            ttk.Label(panel, text="Firing rate (Hz):").pack(anchor='w')
            v['firing_rate'] = tk.StringVar(value="5.0")
            ttk.Entry(panel, textvariable=v['firing_rate'], width=10).pack(anchor='w', pady=(0, 6))

            ttk.Label(panel, text="Burst config:", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', pady=(4, 2))

            ttk.Label(panel, text="Burst rate (%):").pack(anchor='w')
            v['burst_rate'] = tk.StringVar(value="0")
            ttk.Entry(panel, textvariable=v['burst_rate'], width=10).pack(anchor='w', pady=(0, 4))

            ttk.Label(panel, text="Number of spikes/burst:").pack(anchor='w')
            v['n_bursts'] = tk.StringVar(value="3")
            ttk.Entry(panel, textvariable=v['n_bursts'], width=10).pack(anchor='w', pady=(0, 4))

            ttk.Label(panel, text="Burst interval (ms):").pack(anchor='w')
            v['burst_interval'] = tk.StringVar(value="5.0")
            ttk.Entry(panel, textvariable=v['burst_interval'], width=10).pack(anchor='w', pady=(0, 4))

            sim_vars[role] = v

        # ── Bottom pane: CCG result + buttons ──
        bottom_frame = ttk.Frame(pw, padding=6)
        pw.add(bottom_frame, stretch='always')

        # State dict to hold simulation results for redraws
        sim_state = {}

        sim_fig = Figure(figsize=(6, 3.5))
        sim_ax = sim_fig.add_subplot(111)
        sim_ax.set_title("(no simulation run yet)", fontsize=10)
        sim_canvas = FigureCanvasTkAgg(sim_fig, master=bottom_frame)

        sim_res_label_var = tk.StringVar(value="Res: lowres")

        # Buttons row at top of bottom pane
        btn_frame = ttk.Frame(bottom_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(btn_frame, text="Compute CCG", command=lambda: self._run_simulation(
            win, sim_name_var, sim_dur_var, sim_dur_unit,
            sim_noise_var, sim_sync_var, sim_delay_var,
            sim_vars, sim_fig, sim_ax, sim_canvas, sim_state,
            sim_res_label_var)).pack(side=tk.LEFT)
        ttk.Button(
            btn_frame, textvariable=sim_res_label_var,
            command=lambda: self._sim_toggle_resolution(
                sim_state, sim_fig, sim_ax, sim_canvas, sim_res_label_var),
            width=16).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.RIGHT)

        sim_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def _sim_ctrl_r(_e=None):
            self._sim_toggle_resolution(
                sim_state, sim_fig, sim_ax, sim_canvas, sim_res_label_var)
            return 'break'

        # One bind on the Toplevel: bindtags deliver Key-* here for any focused
        # descendant (plot canvas, buttons, …) without double-firing.
        win.bind('<Control-r>', _sim_ctrl_r)
        win.bind('<Command-r>', _sim_ctrl_r)

    @staticmethod
    def _sim_generate_train(duration_s, firing_rate, noise_std,
                            burst_rate_pct, n_per_burst, burst_interval_ms):
        """Generate a Poisson spike train with optional bursting.

        Parameters
        ----------
        duration_s : float  — total duration in seconds
        firing_rate : float — mean rate in Hz
        noise_std : float   — Gaussian jitter (seconds) added to each spike
        burst_rate_pct : float — percentage of spikes that trigger a burst
        n_per_burst : int   — number of extra spikes per burst
        burst_interval_ms : float — inter-spike interval within a burst (ms)

        Returns
        -------
        np.ndarray of spike times in seconds, sorted
        """
        rng = np.random.default_rng()
        # Base Poisson process
        n_expected = int(firing_rate * duration_s * 1.2) + 100
        isis = rng.exponential(1.0 / firing_rate, size=n_expected)
        times = np.cumsum(isis)
        times = times[times < duration_s]

        # Gaussian noise
        if noise_std > 0:
            times = times + rng.normal(0, noise_std, size=len(times))

        # Bursting: for each spike with probability burst_rate_pct/100,
        # add n_per_burst extra spikes at burst_interval_ms intervals
        if burst_rate_pct > 0 and n_per_burst > 0:
            burst_mask = rng.random(len(times)) < (burst_rate_pct / 100.0)
            burst_origins = times[burst_mask]
            extra = []
            isi_s = burst_interval_ms / 1000.0
            for b in burst_origins:
                for k in range(1, n_per_burst + 1):
                    extra.append(b + k * isi_s)
            if extra:
                times = np.concatenate([times, np.array(extra)])

        times = np.sort(times)
        times = times[(times >= 0) & (times < duration_s)]
        return times

    def _sim_compute_correlogram(self, sim_neurons, bin_size, duration, conf):
        """Return (ccg_1d, null_1d, pval_1d, bin_size_eff) for one bin size."""
        from neuropy.analyses.correlations import np_spike_correlations
        ccg_raw = np_spike_correlations(
            sim_neurons,
            neuron_inds=np.array([0, 1]),
            bin_size=bin_size,
            window_size=duration,
            symmetrize=True,
        )
        ccg_1d = ccg_raw[0, 1, :]
        pvals_all, pred_all, _q = EranConv._conv(
            ccg_1d, W=conf.conv_window_bins, wintype="gauss")
        null_1d = pred_all[0]
        pval_1d = pvals_all[0]
        n_bins = len(ccg_1d)
        bin_size_eff = duration / (n_bins - 1) if n_bins > 1 else bin_size
        return ccg_1d, null_1d, pval_1d, bin_size_eff

    def _sim_redraw_plot(self, fig, ax, canvas, sim_state):
        """Redraw simulator CCG from sim_state at current lo/hi resolution."""
        if 'ccg_lo' not in sim_state:
            return
        hi = sim_state.get('highres', False)
        if hi:
            ccg = sim_state['ccg_hi']
            null_1d = sim_state['null_hi']
            bse = sim_state['bin_size_eff_hi']
        else:
            ccg = sim_state['ccg_lo']
            null_1d = sim_state['null_lo']
            bse = sim_state['bin_size_eff_lo']
        conf = sim_state['conf']
        params = sim_state['params']
        name = sim_state['name']
        ax.clear()
        ref_nick = params['ref']['nickname']
        tgt_nick = params['tgt']['nickname']
        res_tag = 'hi' if hi else 'lo'
        plot_ccg.plot_ccg_panel(
            ax=ax, ccg=ccg,
            ids=(ref_nick, tgt_nick), inds=(0, 1),
            neuron_type=(params['ref'].get('type'), params['tgt'].get('type')),
            window_size=conf.duration, bin_size=bse,
            ccg_null=null_1d,
            segment_id=f"sim: {name} ({res_tag}-res)",
            min_lag=conf.min_lag,
            max_lag=conf.max_lag,
            line_baseline=True,
        )
        fig.tight_layout()
        canvas.draw()

    def _sim_toggle_resolution(self, sim_state, fig, ax, canvas, label_var):
        """Toggle simulator between low- and high-res CCG (button or Ctrl/Cmd+R)."""
        cur_hi = 'highres' in label_var.get()
        label_var.set('Res: lowres' if cur_hi else 'Res: highres')
        if 'ccg_lo' not in sim_state:
            return 'break'
        sim_state['highres'] = not cur_hi
        self._sim_redraw_plot(fig, ax, canvas, sim_state)
        return 'break'

    def _run_simulation(self, win, name_var, dur_var, unit_var,
                        noise_var, sync_var, delay_var,
                        sim_vars, fig, ax, canvas, sim_state, sim_res_label_var):
        """Simulate spike trains, compute CCG + EranConv, and plot result."""
        try:
            # Parse parameters
            dur_raw = float(dur_var.get())
            unit = unit_var.get()
            dur_s = {'ms': dur_raw / 1000.0, 's': dur_raw,
                     'min': dur_raw * 60.0, 'hour': dur_raw * 3600.0}[unit]
            noise_std = float(noise_var.get())
            sync_pct = float(sync_var.get())
            delay_ms = float(delay_var.get())
            delay_s = delay_ms / 1000.0

            params = {}
            for role in ('ref', 'tgt'):
                v = sim_vars[role]
                params[role] = {
                    'nickname': v['nickname'].get(),
                    'type': v['type'].get(),
                    'rate': float(v['firing_rate'].get()),
                    'burst_rate': float(v['burst_rate'].get()),
                    'n_bursts': int(v['n_bursts'].get()),
                    'burst_interval': float(v['burst_interval'].get()),
                }
        except (ValueError, KeyError) as e:
            messagebox.showerror("Simulation", f"Invalid parameter: {e}", parent=win)
            return

        # Generate spike trains
        ref_train = self._sim_generate_train(
            dur_s, params['ref']['rate'], noise_std,
            params['ref']['burst_rate'], params['ref']['n_bursts'],
            params['ref']['burst_interval'])
        tgt_train = self._sim_generate_train(
            dur_s, params['tgt']['rate'], noise_std,
            params['tgt']['burst_rate'], params['tgt']['n_bursts'],
            params['tgt']['burst_interval'])

        # Excess synchrony: for each ref spike, with probability sync_pct/100,
        # add an extra spike to tgt at ref_time + synaptic_delay
        if sync_pct > 0:
            rng = np.random.default_rng()
            mask = rng.random(len(ref_train)) < (sync_pct / 100.0)
            extra_tgt = ref_train[mask] + delay_s
            extra_tgt = extra_tgt[(extra_tgt >= 0) & (extra_tgt < dur_s)]
            tgt_train = np.sort(np.concatenate([tgt_train, extra_tgt]))

        # CCG config matching main UI (duration / conv window); bin sizes from presets
        conf = self.ccg_data.conf if self.ccg_data is not None else CCGConfig()
        duration = conf.duration if conf.duration else 20e-3
        bin_lo = _CCG_RESOLUTION['lowres']
        bin_hi = _CCG_RESOLUTION['highres']

        # Minimal Neurons object — only spiketrains needed by np_spike_correlations
        sim_neurons = Neurons(
            spiketrains=np.array([ref_train, tgt_train], dtype=object),
            t_stop=dur_s,
            t_start=0.0,
            sampling_rate=30000,
            neuron_ids=np.array([0, 1]),
        )

        ccg_lo, null_lo, pval_lo, bse_lo = self._sim_compute_correlogram(
            sim_neurons, bin_lo, duration, conf)
        ccg_hi, null_hi, pval_hi, bse_hi = self._sim_compute_correlogram(
            sim_neurons, bin_hi, duration, conf)

        sim_state['ccg_lo'] = ccg_lo
        sim_state['null_lo'] = null_lo
        sim_state['pval_lo'] = pval_lo
        sim_state['bin_size_eff_lo'] = bse_lo
        sim_state['ccg_hi'] = ccg_hi
        sim_state['null_hi'] = null_hi
        sim_state['pval_hi'] = pval_hi
        sim_state['bin_size_eff_hi'] = bse_hi
        sim_state['conf'] = conf
        sim_state['params'] = params
        sim_state['name'] = name_var.get()
        sim_state['highres'] = 'highres' in sim_res_label_var.get()

        self._sim_redraw_plot(fig, ax, canvas, sim_state)

    def _jitter_queue_dialog(self):
        """Show a dialog listing all queued (and running) jitter/CCG tasks."""
        win = tk.Toplevel(self.root)
        win.title("Jitter Queue")
        win.geometry("460x340")
        win.transient(self.root)

        frame = ttk.Frame(win, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Queued tasks", font=('TkDefaultFont', 11, 'bold')).pack(anchor='w')

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        lb = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                        selectmode=tk.EXTENDED, font=('TkFixedFont', 10))
        scrollbar.config(command=lb.yview)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _refresh():
            lb.delete(0, tk.END)
            for i, task in enumerate(self._jitter_pending):
                ref, tgt = task[1], task[2]
                n = task[3]
                res = task[4]
                seg = task[6] if len(task) > 6 else None
                seg_s = f" seg{seg}" if seg is not None else ""
                status = "▶ RUNNING" if i == 0 and self._is_task_running() else "  queued"
                lb.insert(tk.END, f"{status}  jitter [{ref},{tgt}] n={n} {res}{seg_s}")
            for i, task in enumerate(self._custom_ccg_pending):
                name = (task.get('name') if isinstance(task, dict) else task[3])
                status = "▶ RUNNING" if i == 0 and self._custom_ccg_is_running() else "  queued"
                lb.insert(tk.END, f"{status}  custom CCG '{name}'")
            if lb.size() == 0:
                lb.insert(tk.END, "  (empty)")

        def _delete_selected():
            sel = lb.curselection()
            if not sel:
                return
            n_jitter = len(self._jitter_pending)
            n_ccg    = len(self._custom_ccg_pending)
            running_jitter = self._is_task_running()
            running_ccg    = self._custom_ccg_is_running()
            jitter_to_remove = []
            ccg_to_remove    = []
            for s in sel:
                if s < n_jitter:
                    if s == 0 and running_jitter:
                        continue
                    jitter_to_remove.append(s)
                else:
                    ccg_idx = s - n_jitter
                    if ccg_idx == 0 and running_ccg:
                        continue
                    ccg_to_remove.append(ccg_idx)
            if jitter_to_remove:
                pending = list(self._jitter_pending)
                for idx in sorted(jitter_to_remove, reverse=True):
                    pending.pop(idx)
                self._jitter_pending.clear()
                self._jitter_pending.extend(pending)
                self._update_jitter_btn_text()
            if ccg_to_remove:
                pending = list(self._custom_ccg_pending)
                for idx in sorted(ccg_to_remove, reverse=True):
                    removed = pending.pop(idx)
                    self._on_split_batch_task_done(removed)
                self._custom_ccg_pending.clear()
                self._custom_ccg_pending.extend(pending)
            _refresh()

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btn_frame, text="Delete selected", command=_delete_selected).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Refresh", command=_refresh).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.RIGHT)

        _refresh()

    def _jitter_clear_queue(self):
        """Remove all pending (non-running) tasks from jitter and custom CCG queues."""
        n_removed = 0
        if self._is_task_running() and self._jitter_pending:
            first = self._jitter_pending[0]
            n_removed += len(self._jitter_pending) - 1
            self._jitter_pending.clear()
            self._jitter_pending.append(first)
        else:
            n_removed += len(self._jitter_pending)
            self._jitter_pending.clear()
        if self._custom_ccg_is_running() and self._custom_ccg_pending:
            first = self._custom_ccg_pending[0]
            for task in list(self._custom_ccg_pending)[1:]:
                self._on_split_batch_task_done(task)
            n_removed += len(self._custom_ccg_pending) - 1
            self._custom_ccg_pending.clear()
            self._custom_ccg_pending.append(first)
        else:
            for task in list(self._custom_ccg_pending):
                self._on_split_batch_task_done(task)
            n_removed += len(self._custom_ccg_pending)
            self._custom_ccg_pending.clear()
        self._update_jitter_btn_text()
        if n_removed == 0:
            messagebox.showinfo("Jitter", "Queue is empty.")
        else:
            messagebox.showinfo("Jitter", f"Removed {n_removed} queued task(s).")

    def setup_help_menu(self, menubar):
        """Help menu with hotkey reference and project website."""
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Hotkeys", command=self._show_hotkeys_dialog)
        help_menu.add_command(label="Website",
                              command=lambda: __import__('webbrowser').open(
                                  'https://github.com/selina-lii/NeuroPy'))

    def _settings_dialog(self):
        """VSCode-style settings dialog with a left nav and right content area."""
        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.geometry("620x420")
        win.resizable(True, True)
        win.transient(self.root)
        win.protocol("WM_DELETE_WINDOW", win.destroy)
        win.bind('<Escape>', lambda e: win.destroy())

        # ── Derive colors from the main window theme ───────────────────
        style = ttk.Style(win)
        _raw_bg = style.lookup('TFrame', 'background') or self.root.cget('bg')
        try:
            _rgb = win.winfo_rgb(_raw_bg)
            _lum = (0.299 * _rgb[0] + 0.587 * _rgb[1] + 0.114 * _rgb[2]) / 65535
        except Exception:
            _lum = 1.0   # assume light if can't detect
        _dark = _lum < 0.4
        _CONT_BG  = '#1e1e1e'   if _dark else 'white'
        _NAV_BG   = '#252526'   if _dark else '#f3f3f3'
        _NAV_SEL  = '#37373d'   if _dark else '#dce8f5'
        _FG       = '#cccccc'   if _dark else '#111111'
        _FG_DIM   = '#888888'   if _dark else '#666666'
        _SUM_BG   = '#2d2d2d'   if _dark else '#f8f8f8'
        _HDR_FONT = ('Arial', 13, 'bold')
        _LBL_FONT = ('Arial', 10)

        # ── Bottom bar — packed FIRST so it anchors to bottom ──────────
        bot = tk.Frame(win, bg=_NAV_BG)
        ttk.Separator(win).pack(side=tk.BOTTOM, fill=tk.X)
        bot.pack(side=tk.BOTTOM, fill=tk.X)

        def _apply():
            try:
                v = int(_max_tog_var.get())
                if 2 <= v <= 20:
                    self._settings['max_show_together'] = v
                    self._save_ui_state()
            except (ValueError, tk.TclError):
                pass
            try:
                fs = int(_min_font_var.get())
                if 6 <= fs <= 32:
                    self._settings['min_font_size'] = fs
                    self._save_ui_state()
                    self._apply_min_font_size()
            except (ValueError, tk.TclError):
                pass
            win.destroy()

        ttk.Button(bot, text="Save", command=_apply).pack(
            side=tk.RIGHT, padx=8, pady=6)
        ttk.Button(bot, text="Cancel", command=win.destroy).pack(
            side=tk.RIGHT, padx=0, pady=6)

        # ── Left sidebar (section list) ────────────────────────────────
        sidebar = tk.Frame(win, bg=_NAV_BG, width=160)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        ttk.Separator(win, orient='vertical').pack(side=tk.LEFT, fill=tk.Y)

        # ── Right content area (scrollable) ───────────────────────────
        cont_outer = tk.Frame(win, bg=_CONT_BG)
        cont_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cont_canvas = tk.Canvas(cont_outer, bg=_CONT_BG, highlightthickness=0)
        cont_scroll = ttk.Scrollbar(cont_outer, command=cont_canvas.yview)
        cont_canvas.configure(yscrollcommand=cont_scroll.set)
        cont_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        cont_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        content = tk.Frame(cont_canvas, bg=_CONT_BG)
        cwin = cont_canvas.create_window((0, 0), window=content, anchor='nw')
        content.bind('<Configure>', lambda e: (
            cont_canvas.configure(scrollregion=cont_canvas.bbox('all')),
            cont_canvas.itemconfig(cwin, width=cont_canvas.winfo_width()),
        ))
        cont_canvas.bind('<Configure>', lambda e:
            cont_canvas.itemconfig(cwin, width=e.width))

        # ── Registry of sections ───────────────────────────────────────
        _section_frames = {}
        _nav_buttons    = {}

        def _show_section(name):
            for f in _section_frames.values():
                f.pack_forget()
            _section_frames[name].pack(fill=tk.BOTH, expand=True, padx=20, pady=16)
            for n, b in _nav_buttons.items():
                b.config(bg=_NAV_SEL if n == name else _NAV_BG, relief='flat')

        def _add_section(name):
            frame = tk.Frame(content, bg=_CONT_BG)
            _section_frames[name] = frame
            btn = tk.Button(sidebar, text=name, anchor='w', bd=0,
                            bg=_NAV_BG, fg=_FG, activebackground=_NAV_SEL,
                            activeforeground=_FG, font=_LBL_FONT,
                            padx=12, pady=6,
                            command=lambda n=name: _show_section(n))
            btn.pack(fill=tk.X)
            _nav_buttons[name] = btn
            return frame

        # ── Display section ────────────────────────────────────────────
        disp = _add_section('Display')
        tk.Label(disp, text="Display", bg=_CONT_BG, fg=_FG,
                 font=_HDR_FONT, anchor='w').pack(fill=tk.X, pady=(0, 12))
        ttk.Separator(disp).pack(fill=tk.X, pady=(0, 10))

        row = tk.Frame(disp, bg=_CONT_BG)
        row.pack(fill=tk.X, pady=6)
        tk.Label(row, text="Max pairs in 'Show Together':", bg=_CONT_BG,
                 fg=_FG, font=_LBL_FONT).pack(side=tk.LEFT)
        _max_tog_var = tk.IntVar(value=self._settings.get('max_show_together', 5))
        ttk.Spinbox(row, from_=2, to=20, textvariable=_max_tog_var,
                    width=5).pack(side=tk.LEFT, padx=10)
        tk.Label(row, text="(2–20)", bg=_CONT_BG,
                 fg=_FG_DIM, font=('Arial', 9)).pack(side=tk.LEFT)

        row = tk.Frame(disp, bg=_CONT_BG)
        row.pack(fill=tk.X, pady=6)
        tk.Label(row, text="Minimum font size:", bg=_CONT_BG,
                 fg=_FG, font=_LBL_FONT).pack(side=tk.LEFT)
        _min_font_var = tk.IntVar(value=int(self._settings.get('min_font_size', 9) or 9))
        ttk.Spinbox(row, from_=6, to=32, textvariable=_min_font_var,
                    width=5).pack(side=tk.LEFT, padx=10)
        tk.Label(row, text="(6–32)", bg=_CONT_BG,
                 fg=_FG_DIM, font=('Arial', 9)).pack(side=tk.LEFT)

        # ── Cache Configuration section ────────────────────────────────
        cache_sec = _add_section('Cache Config')
        tk.Label(cache_sec, text="Cache Configuration", bg=_CONT_BG, fg=_FG,
                 font=_HDR_FONT, anchor='w').pack(fill=tk.X, pady=(0, 4))
        ttk.Separator(cache_sec).pack(fill=tk.X, pady=(0, 10))
        _cache_config_help = tk.Label(cache_sec,
                 text="Only one display configuration is saved to the PNG disk cache.\n"
                      "Any other configuration is rendered in real-time (not cached).\n"
                      "Set up the significance panel as desired, then capture it here.",
                 bg=_CONT_BG, fg=_FG_DIM, font=('Arial', 9),
                 justify='left', wraplength=380)
        _cache_config_help.pack(anchor='w', pady=(0, 10))

        _cache_summary_var = tk.StringVar()

        def _refresh_cache_summary():
            cfg = self._cache_config
            if cfg is None:
                _cache_summary_var.set("No configuration set  (legacy mode: all states cached)")
            else:
                lines = []
                on_sigs = [k.replace('_sig_', '').replace('_var', '')
                           for k in self._CACHE_CONFIG_ATTRS
                           if k.startswith('_sig_') and cfg.get(k)]
                lines.append(f"Sig overlays: {', '.join(on_sigs) or 'none'}")
                on_lines = [k.replace('_line_', '').replace('_var', '')
                            for k in self._CACHE_CONFIG_ATTRS
                            if k.startswith('_line_') and cfg.get(k)]
                lines.append(f"Line styles:  {', '.join(on_lines) or 'none'}")
                norms = cfg.get('active_norms') or []
                lines.append(f"Norms:        {', '.join(norms) or 'raw'}")
                lines.append(f"Alpha:        {cfg.get('active_alpha', '—')}")
                _cache_summary_var.set('\n'.join(lines))

        _refresh_cache_summary()
        tk.Label(cache_sec, textvariable=_cache_summary_var,
                 bg=_SUM_BG, fg=_FG, relief='groove', font=('Courier', 9),
                 justify='left', anchor='nw', padx=8, pady=6).pack(fill=tk.X, pady=(0, 10))

        cbtn_row = tk.Frame(cache_sec, bg=_CONT_BG)
        cbtn_row.pack(anchor='w')

        def _capture():
            self._cache_config = self._current_display_config()
            self._save_ui_state()
            _refresh_cache_summary()
            self._clear_all_png_cache()

        def _clear_cache_config():
            self._cache_config = None
            self._save_ui_state()
            _refresh_cache_summary()

        ttk.Button(cbtn_row, text="Capture current settings",
                   command=_capture).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(cbtn_row, text="Clear",
                   command=_clear_cache_config).pack(side=tk.LEFT)

        # ── Cache Management section ───────────────────────────────────
        cache_mgmt = _add_section('Cache')
        tk.Label(cache_mgmt, text="Cache", bg=_CONT_BG, fg=_FG,
                 font=_HDR_FONT, anchor='w').pack(fill=tk.X, pady=(0, 4))
        ttk.Separator(cache_mgmt).pack(fill=tk.X, pady=(0, 10))

        # -- Disk usage row
        _disk_var = tk.StringVar(value="–")
        disk_row = tk.Frame(cache_mgmt, bg=_CONT_BG)
        disk_row.pack(fill=tk.X, pady=(0, 6))
        tk.Label(disk_row, text="Cache folder:", bg=_CONT_BG, fg=_FG,
                 font=_LBL_FONT).pack(side=tk.LEFT)
        tk.Label(disk_row, text=self.tmp_dir, bg=_CONT_BG, fg=_FG_DIM,
                 font=('Arial', 9)).pack(side=tk.LEFT, padx=(6, 12))
        tk.Label(disk_row, textvariable=_disk_var, bg=_CONT_BG, fg=_FG,
                 font=_LBL_FONT).pack(side=tk.LEFT)

        def _compute_disk_size():
            total = 0
            n_files = 0
            try:
                for fn in os.listdir(self.tmp_dir):
                    if fn.endswith('.png'):
                        try:
                            total += os.path.getsize(
                                os.path.join(self.tmp_dir, fn))
                            n_files += 1
                        except OSError:
                            pass
            except OSError:
                pass
            if total < 1024 ** 2:
                sz = f"{total / 1024:.1f} KB"
            elif total < 1024 ** 3:
                sz = f"{total / 1024**2:.1f} MB"
            else:
                sz = f"{total / 1024**3:.2f} GB"
            return f"{n_files} PNGs  ·  {sz}"

        # -- Completeness tree
        tree_frame = tk.Frame(cache_mgmt, bg=_CONT_BG)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        cols = ('type', 'pairs', 'segs', 'pngs', 'pct')
        tree = ttk.Treeview(tree_frame, columns=cols, show='tree headings',
                            height=10, selectmode='none')
        tree.heading('#0', text='Session')
        tree.heading('type', text='Type')
        tree.heading('pairs', text='Pairs')
        tree.heading('segs', text='Segs')
        tree.heading('pngs', text='PNGs')
        tree.heading('pct', text='%')
        tree.column('#0', width=160, stretch=True)
        tree.column('type', width=120, stretch=True)
        tree.column('pairs', width=50, anchor='e', stretch=False)
        tree.column('segs', width=45, anchor='e', stretch=False)
        tree.column('pngs', width=80, anchor='e', stretch=False)
        tree.column('pct', width=55, anchor='e', stretch=False)
        vsb = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Color tags for completeness rows
        tree.tag_configure('full',    background='#d4edda')  # green
        tree.tag_configure('partial', background='')
        tree.tag_configure('low',     background='#fff3cd')  # amber
        tree.tag_configure('empty',   background='#f8d7da')  # red

        def _pct_tag(pct):
            if pct >= 100:
                return 'full'
            if pct >= 50:
                return 'partial'
            if pct > 0:
                return 'low'
            return 'empty'

        def _count_pngs_for_pair(ref, tgt, n_segs, seg_names, cfg_now, has_hi):
            """Count existing PNGs on disk for this pair using _png_path-style names."""
            count = 0
            for seg in list(range(n_segs)) + [n_segs]:
                seg_nm = ('All_segments' if seg == n_segs
                          else seg_names[seg] if seg < len(seg_names)
                          else str(seg))
                norms = cfg_now.get('active_norms') or []
                norm_key = '_'.join(sorted(norms)) if norms else 'raw'
                alpha = cfg_now.get('active_alpha', 0.001)
                for hires in ([False, True] if has_hi else [False]):
                    res_key = '_hi' if hires else '_lo'
                    # alpha_key: only present when pval_corrected exists;
                    # check with and without for robustness
                    for ak in (f'_a{alpha:.3f}', ''):
                        fn = (f"pair_{ref}_{tgt}_{seg_nm}_{norm_key}"
                              f"{ak}{res_key}.png")
                        if os.path.isfile(os.path.join(self.tmp_dir, fn)):
                            count += 1
                            break  # found one — count once per (pair,seg,res)
            return count

        def _populate_tree_async():
            """Populate cache completeness table without freezing Settings UI."""
            for iid in tree.get_children():
                tree.delete(iid)
            _disk_var.set("…")

            cfg_now = self._cache_config or self._current_display_config()
            nd_sessions = {}   # nd_key_str → list of (type_key, ptr)
            for tk_, ptr in self.cd.data.items():
                nd_str = str(tk_.nd())
                nd_sessions.setdefault(nd_str, []).append((tk_, ptr))

            loaded_nd = {str(k.nd()) for k in self.cd.data.keys()
                         if getattr(self.cd, '_ccg', {}).get(k.nd()) is not None}

            def _worker():
                disk_txt = _compute_disk_size()
                rows = []
                for nd_str, entries in sorted(nd_sessions.items()):
                    is_loaded = nd_str in loaded_nd
                    # nd header row
                    rows.append(('__nd__', nd_str, None))
                    for type_key, ptr in sorted(entries, key=lambda x: str(x[0])):
                        type_lbl = self._type_label(type_key)
                        if not is_loaded:
                            rows.append(('__type__', nd_str, (type_lbl, '–', '–', '–', '–', 'partial')))
                            continue
                        pairs_arr = ptr.inds2
                        n_pairs = len(pairs_arr)
                        n_segs = ptr.n_segments
                        seg_names = list(ptr.edge_times['label'].values)
                        has_hi = (hasattr(self.cd, '_ccg_highres') and
                                  self.cd._ccg_highres.get(type_key.nd()) is not None)
                        res_mult = 2 if has_hi else 1
                        expected = n_pairs * (n_segs + 1) * res_mult
                        actual = 0
                        for inds in pairs_arr:
                            ref2, tgt2 = int(inds[0]), int(inds[1])
                            actual += _count_pngs_for_pair(
                                ref2, tgt2, n_segs, seg_names, cfg_now, has_hi)
                        pct = int(100 * actual / expected) if expected > 0 else 0
                        pct_str = f"{pct}%" if expected > 0 else '–'
                        tag = _pct_tag(pct)
                        rows.append(('__type__', nd_str, (type_lbl, n_pairs, n_segs,
                                                         f"{actual}/{expected}", pct_str, tag)))
                return disk_txt, rows

            def _render(disk_txt, rows):
                if not win.winfo_exists():
                    return
                _disk_var.set(disk_txt)
                nodes = {}
                for kind, nd_str, payload in rows:
                    if kind == '__nd__':
                        nodes[nd_str] = tree.insert('', 'end', text=nd_str,
                                                    values=('', '', '', '', ''), open=True)
                    else:
                        node = nodes.get(nd_str)
                        if node is None:
                            continue
                        type_lbl, n_pairs, n_segs, pngs, pct_str, tag = payload
                        tree.insert(node, 'end',
                                    values=(type_lbl, n_pairs, n_segs, pngs, pct_str),
                                    tags=(tag,))

            import threading as _threading
            def _run():
                try:
                    disk_txt, rows = _worker()
                except Exception:
                    disk_txt, rows = "err", []
                win.after(0, lambda: _render(disk_txt, rows))
            _threading.Thread(target=_run, daemon=True).start()

        # -- Auto pre-gen toggle
        auto_row = tk.Frame(cache_mgmt, bg=_CONT_BG)
        auto_row.pack(anchor='w', pady=(0, 4))
        _auto_var = tk.BooleanVar(value=self._auto_pregen_enabled)
        def _on_auto_toggle():
            self._auto_pregen_enabled = _auto_var.get()
            self._save_ui_state()
        ttk.Checkbutton(auto_row, text="Auto-generate cache on session switch",
                        variable=_auto_var, command=_on_auto_toggle).pack(side=tk.LEFT)

        # -- Button row
        btn_row2 = tk.Frame(cache_mgmt, bg=_CONT_BG)
        btn_row2.pack(anchor='w', pady=(0, 8))

        _pregen_status_var = tk.StringVar(value="Idle")
        ttk.Button(btn_row2, text="⚡ Run Pre-gen",
                   command=lambda: self._start_pregen_with_defaults(
                       status_var=_pregen_status_var)).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(btn_row2, textvariable=_pregen_status_var,
                 bg=_CONT_BG, fg=_FG_DIM, font=('Arial', 9)).pack(side=tk.LEFT)
        ttk.Button(btn_row2, text="🗑 Clear all PNGs",
                   command=lambda: (self._clear_all_png_cache(),
                                    _populate_tree_async())).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(btn_row2, text="↻ Refresh",
                   command=_populate_tree_async).pack(side=tk.LEFT, padx=(8, 0))

        # Wire Cache nav button to also populate tree when clicked
        def _show_cache_section():
            _show_section('Cache')
            win.after(10, _populate_tree_async)
        _nav_buttons['Cache'].config(command=_show_cache_section)

        # Keep cache section wrapping/columns responsive to window size
        def _on_resize(_e=None):
            try:
                w = max(int(cont_canvas.winfo_width()), 320)
                _cache_config_help.config(wraplength=max(w - 80, 220))
            except Exception:
                pass
            try:
                tfw = max(int(tree_frame.winfo_width()), 320)
                # Allocate remaining space to the stretchy columns (#0 and 'type')
                fixed = 50 + 45 + 80 + 55 + 20
                rem = max(tfw - fixed, 200)
                tree.column('#0', width=int(rem * 0.55))
                tree.column('type', width=int(rem * 0.45))
            except Exception:
                pass
        win.bind('<Configure>', _on_resize)

        _show_section('Display')

    def _min_font_size(self) -> int:
        """Minimum allowed font size for UI labels/lists."""
        try:
            v = int(self._settings.get('min_font_size', 9))
        except Exception:
            v = 9
        return max(6, min(32, v))

    def _apply_min_font_size(self):
        """Clamp selected UI fonts to at least the configured minimum size."""
        import tkinter.font as _tkfont

        min_fs = self._min_font_size()

        # Pair selection list bodies
        for lb_name in ('unselected_list', 'selected_list'):
            lb = getattr(self, lb_name, None)
            if lb is None:
                continue
            try:
                f = _tkfont.Font(font=lb.cget('font'))
                if f.cget('size') < min_fs:
                    f.configure(size=min_fs)
                lb.config(font=f)
            except Exception:
                pass

        # Plot title (UI label above the plot; not the matplotlib title)
        lbl = getattr(self, '_plot_title_label', None)
        if lbl is not None:
            try:
                f = _tkfont.Font(font=lbl.cget('font'))
                if f.cget('size') < min_fs:
                    f.configure(size=min_fs)
                lbl.config(font=f)
            except Exception:
                pass

        # Hotkey groups chips + bar text (uses mixed widgets)
        try:
            bar = getattr(self, '_hotkeys_bar', None)
            if bar is not None and bar.winfo_exists():
                for w in bar.winfo_children():
                    try:
                        f = _tkfont.Font(font=w.cget('font'))
                        if f.cget('size') < min_fs:
                            f.configure(size=min_fs)
                        w.config(font=f)
                    except Exception:
                        pass
        except Exception:
            pass

    def _show_hotkeys_dialog(self):
        """Show a dialog listing all keyboard shortcuts."""
        hotkeys_text = (
            "Ctrl+E    Toggle between waveform and CCG\n"
            "Ctrl+L    Toggle all histograms outline/filled\n"
            "           (right-click CCG for per-item control)\n"
            "Ctrl+R    Toggle resolution (hi / lo)\n"
            "Ctrl+S    Save selection\n"
            "Ctrl+B    Toggle bookmark on current pair (pin + highlight in lists)\n"
            "Ctrl+Z    Undo\n"
            "Ctrl+Y    Redo\n"
            "\n"
            "1..0          Assign group + advance cursor\n"
            "Shift+1..0    Assign group(s) to current pair (no advance)\n"
            "Ctrl+Delete / Ctrl+Backspace   Move current pair to Deleted\n"
            "Left/Right Arrow   Change segment"
        )
        messagebox.showinfo("Keyboard Shortcuts", hotkeys_text)

    # ── Tool-strip row ─────────────────────────────────────────────────

    def setup_menu(self):
        menu_frame = ttk.Frame(self.root, relief=tk.RAISED, borderwidth=2)
        menu_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Session
        ttk.Label(menu_frame, text="Session:",
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(8, 2))
        nd_keys = self._all_nd_keys()
        self._nd_keys_list = nd_keys
        session_labels = [self._session_label(k) for k in nd_keys]
        self._session_var = tk.StringVar(value=self._session_label(self.key.nd()))
        self._session_combo = ttk.Combobox(
            menu_frame, textvariable=self._session_var,
            values=session_labels, width=22, state='readonly')
        self._session_combo.pack(side=tk.LEFT, padx=2)
        self._session_combo.bind('<<ComboboxSelected>>', self._on_session_change)

        # Type — if the key has no conn_type, fall back to E pyr→pyr or first available
        ttk.Label(menu_frame, text="Type:",
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(10, 2))
        type_keys = self._available_type_keys(self.key.nd())
        self._type_keys_list = type_keys
        type_labels = [self._type_label(k) for k in type_keys]
        if not getattr(self.key, 'conn_type', None) and type_keys:
            # Key has no conn_type — pick E pyr→pyr, else first available
            preferred = [k for k in type_keys
                         if getattr(k, 'excitability', None) == 'E'
                         and getattr(k, 'conn_type', None) == ('pyr', 'pyr')]
            self.key = (preferred or type_keys)[0]
            if self.ccg_pointer is None:
                self.ccg_pointer = self.cd.data.get(self.key)
        self._type_var = tk.StringVar(value=self._type_label(self.key))
        self._type_combo = ttk.Combobox(
            menu_frame, textvariable=self._type_var,
            values=type_labels, width=18, state='readonly')
        self._type_combo.pack(side=tk.LEFT, padx=2)
        self._type_combo.bind('<<ComboboxSelected>>', self._on_type_change)

        # Pre-gen button — warms PNG cache with canonical display defaults
        self._pregen_btn = ttk.Button(
            menu_frame, text="⚡ Pre-gen", width=10,
            command=self._start_pregen_with_defaults)
        self._pregen_btn.pack(side=tk.RIGHT, padx=(2, 8))

    # ── Group hotkeys bar ──────────────────────────────────────────────

    def setup_group_hotkeys_bar(self):
        """Horizontal bar showing Ctrl+1…0 → group-name mappings."""
        self._hotkeys_bar = ttk.Frame(self.root, relief=tk.GROOVE, borderwidth=1)
        self._hotkeys_bar_labels: list[tk.Label] = []
        self._hotkeys_bar_offset = 0  # scroll offset (number of chips hidden on the left)
        self._refresh_hotkeys_bar()
        # Pack immediately if default is visible
        if self._panel_vars.get('Group Hotkeys', tk.BooleanVar()).get():
            self._hotkeys_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 2))

    def _refresh_hotkeys_bar(self):
        """Rebuild the labels inside the hotkeys bar."""
        for w in self._hotkeys_bar.winfo_children():
            w.destroy()
        self._hotkeys_bar_labels.clear()

        ttk.Label(self._hotkeys_bar, text="Groups:",
                  font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(6, 4))

        # Fixed "Del" key label for deleted toggle
        del_lbl = tk.Label(self._hotkeys_bar, text="Del/⌫: deleted",
                           font=('Courier', 9), padx=6, pady=1,
                           relief=tk.RIDGE, borderwidth=1, fg='#888888')
        del_lbl.pack(side=tk.LEFT, padx=2, pady=2)

        # Order: digits 1–9, 0 first, then letters a–z
        digit_order = [str(i) for i in range(1, 10)] + ['0']
        letter_order = list('abcdefghijklmnopqrstuvwxyz')
        slot_order = digit_order + letter_order
        # Invert: hotkey_str → group_name
        hk_to_group = {v: k for k, v in self._group_hotkeys.items()}

        # Build ordered list of (key_str, gname) chips
        all_chips = [(k, hk_to_group[k]) for k in slot_order if k in hk_to_group]

        if not all_chips:
            ttk.Label(self._hotkeys_bar, text="(no hotkeys assigned)",
                      font=('Arial', 9), foreground='#888').pack(
                side=tk.LEFT, padx=4)
            return

        # Clamp scroll offset
        offset = getattr(self, '_hotkeys_bar_offset', 0)
        offset = max(0, min(offset, len(all_chips) - 1))
        self._hotkeys_bar_offset = offset

        # Right scroll arrow — packed RIGHT first so it stays anchored at the edge
        right_btn = tk.Label(self._hotkeys_bar, text='▶', font=('Arial', 9),
                             padx=4, pady=1, cursor='hand2',
                             fg='#555' if offset < len(all_chips) - 1 else '#ccc')
        right_btn.pack(side=tk.RIGHT, padx=(1, 4))
        right_btn.bind('<Button-1>', lambda e: self._scroll_hotkeys_bar(1))

        # Left scroll arrow
        left_btn = tk.Label(self._hotkeys_bar, text='◀', font=('Arial', 9),
                            padx=4, pady=1, cursor='hand2',
                            fg='#555' if offset > 0 else '#ccc')
        left_btn.pack(side=tk.LEFT, padx=(0, 1))
        left_btn.bind('<Button-1>', lambda e: self._scroll_hotkeys_bar(-1))

        for key_str, gname in all_chips[offset:]:
            display = f"{key_str}: {gname}"
            lbl = tk.Label(self._hotkeys_bar, text=display,
                           font=('Courier', 9), padx=6, pady=1,
                           relief=tk.RIDGE, borderwidth=1)
            lbl.pack(side=tk.LEFT, padx=2, pady=2)
            lbl.bind('<Button-1>',
                     lambda e, g=gname: self._select_group(g))
            lbl.bind('<Double-Button-1>',
                     lambda e, g=gname: self._group_chip_double_click(g))
            self._hotkeys_bar_labels.append(lbl)

    def _scroll_hotkeys_bar(self, direction: int):
        """Scroll the hotkeys bar left (-1) or right (+1) by one chip."""
        digit_order = [str(i) for i in range(1, 10)] + ['0']
        slot_order = digit_order + list('abcdefghijklmnopqrstuvwxyz')
        hk_to_group = {v: k for k, v in self._group_hotkeys.items()}
        n_chips = sum(1 for k in slot_order if k in hk_to_group)
        self._hotkeys_bar_offset = max(0, min(
            getattr(self, '_hotkeys_bar_offset', 0) + direction, n_chips - 1))
        self._refresh_hotkeys_bar()

    def _group_chip_double_click(self, group_name: str):
        """Draw a random example pair from group_name; flash chip red if empty."""
        import random as _random
        pairs = self._group_pairs(group_name)
        if not pairs:
            # Flash the chip label red for 0.3 s
            for lbl in self._hotkeys_bar_labels:
                if group_name in lbl.cget('text'):
                    orig_fg = lbl.cget('fg')
                    lbl.config(fg='red')
                    self.root.after(300, lambda l=lbl, c=orig_fg: l.config(fg=c))
            return
        chosen = _random.choice(sorted(pairs))
        self.current_pair_idx = self.get_pair_index(chosen)
        self.update_plot()
        self._draw_network()

    # ── Left panel ─────────────────────────────────────────────────────

    def setup_left_panel(self, parent):
        # Delegate to LeftPanelContainer (pair_selection_panel.py)
        self.left_container = LeftPanelContainer(
            parent, self._sel_data, self, self._ui_state_cache)
        self.left_container.widget.pack(fill=tk.BOTH, expand=True)

        # ── Backward-compat aliases so existing CCGReviewUI code keeps working ──
        lp = self.left_container.left_panel
        sp = self.left_container.spike_pairs

        self._left_notebook    = self.left_container.widget
        self.unselected_list   = lp.unselected_list
        self.selected_list     = lp.selected_list
        self._avail_label_var  = lp._avail_label    # LeftPanel uses _avail_label (no _var suffix)
        self._sel_label_var    = lp._sel_label
        self._clear_spec_btn   = lp._clear_spec_btn
        self._pair_list_pane   = lp._pair_list_pane

        # Sort vars — LeftPanel drops the _var suffix; alias under old names
        self._sort_selected_var = lp._sort_selected
        self._sort_by_tag_var   = lp._sort_by_tag
        self._sort_by_mean_var  = lp._sort_by_mean
        self._sort_by_min_p_var = lp._sort_by_min_p

        # Search — old methods (_search_show/_hide/_go/_update) are dead code;
        # Ctrl+F is rebound below.  Keep minimal aliases for any residual refs.
        self._search_frame   = lp.search_bar._frame
        self._search_entry   = lp.search_bar._entry
        self._search_var     = lp.search_bar._var
        self._search_matches = lp.search_bar._matches

        # Spike pairs panel aliases (old code used self._sa_*)
        self._sa_tab       = sp._tab
        self._sa_listbox   = sp._spike_pairs_listbox
        self._sa_count_var = sp._spike_pairs_count
        self._sa_tab_index = sp._spike_pairs_tab_index

    # ── Center panel ───────────────────────────────────────────────────

    def setup_center_panel(self, parent):
        self.plot_title_var = tk.StringVar(value=self.get_plot_title())
        self._plot_title_label = ttk.Label(
            parent, textvariable=self.plot_title_var, font=('Arial', 11, 'bold'))
        self._plot_title_label.pack(side=tk.TOP)

        # Vertical PanedWindow: top = CCG figure, bottom = control panels
        pw = tk.PanedWindow(parent, orient=tk.VERTICAL,
                            sashrelief=tk.RAISED, sashwidth=5,
                            bg='#CCCCCC')
        pw.pack(fill=tk.BOTH, expand=True)

        # Top pane: horizontal split — CCG (left) | Waveforms (right, toggleable)
        plot_frame = ttk.Frame(pw)
        self._plot_pw = tk.PanedWindow(plot_frame, orient=tk.HORIZONTAL,
                                       sashrelief=tk.RAISED, sashwidth=4,
                                       bg='#CCCCCC')
        self._plot_pw.pack(fill=tk.BOTH, expand=True)
        ccg_inner = ttk.Frame(self._plot_pw)
        self.fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=ccg_inner)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.get_tk_widget().bind('<Button-2>', self._ccg_context_menu)
        self.canvas.get_tk_widget().bind('<Button-3>', self._ccg_context_menu)
        self._plot_pw.add(ccg_inner, stretch='always')
        pw.add(plot_frame, minsize=100, stretch='always')

        # Bottom pane: scrollable control panels
        scroll_outer = ttk.Frame(pw)
        _ctrl_canvas = tk.Canvas(scroll_outer, highlightthickness=0)
        _ctrl_scroll = ttk.Scrollbar(scroll_outer, orient='vertical',
                                     command=_ctrl_canvas.yview)
        _ctrl_canvas.configure(yscrollcommand=_ctrl_scroll.set)
        _ctrl_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        _ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ctrl_frame = ttk.Frame(_ctrl_canvas)
        _ctrl_win = _ctrl_canvas.create_window((0, 0), window=ctrl_frame, anchor='nw')

        def _ctrl_resize(event):
            _ctrl_canvas.configure(scrollregion=_ctrl_canvas.bbox('all'))
        def _ctrl_canvas_width(event):
            _ctrl_canvas.itemconfigure(_ctrl_win, width=event.width)
        ctrl_frame.bind('<Configure>', _ctrl_resize)
        _ctrl_canvas.bind('<Configure>', _ctrl_canvas_width)

        # Mousewheel scrolling over the control area
        def _ctrl_scroll_wheel(event):
            if event.delta:
                _ctrl_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
            elif event.num == 4:
                _ctrl_canvas.yview_scroll(-1, 'units')
            elif event.num == 5:
                _ctrl_canvas.yview_scroll(1, 'units')
        _ctrl_canvas.bind('<MouseWheel>', _ctrl_scroll_wheel)
        _ctrl_canvas.bind('<Button-4>', _ctrl_scroll_wheel)
        _ctrl_canvas.bind('<Button-5>', _ctrl_scroll_wheel)

        self.setup_baseline_cs_panel(ctrl_frame)
        self.setup_norm_panel(ctrl_frame)
        self.setup_waveforms_panel(ctrl_frame)   # hidden by default
        # Keep these at the end (per request)
        self.setup_spike_attrib_panel(ctrl_frame)
        self.setup_jitter_panel(ctrl_frame)

        # Significance chips
        self.sig_frame = ttk.Frame(ctrl_frame)
        self.sig_frame.pack(side=tk.BOTTOM, pady=2, fill=tk.X)
        self._build_sig_chips()

        pw.add(scroll_outer, minsize=30, stretch='never')

        # Hidden segment state (combo removed; segment chips handle navigation)
        _seg_var_init = (_ALL_SEGS if self.current_segment >= self.n_segments
                         else self.segment_names[self.current_segment])
        self.segment_var = tk.StringVar(value=_seg_var_init)
        self.segment_combo = ttk.Combobox(
            parent, textvariable=self.segment_var,
            values=self.segment_names + [_ALL_SEGS], width=14,
            state='readonly')
        self.segment_combo.bind('<<ComboboxSelected>>', self._on_segment_change)

        self.root.after(100, self._deferred_initial_draw)

    def setup_baseline_cs_panel(self, parent):
        """Unified 'Baseline & Connection Strength' panel."""
        cs_frame, self._cs_fold_var = self._make_collapsible_panel(
            parent, "Baseline & Connection Strength")
        self._sig_inner_frame = cs_frame

        # ── Row 1: CS overlay toggle + metric selector ────────────────
        row1 = ttk.Frame(cs_frame)
        row1.pack(fill=tk.X, anchor='w')
        ttk.Checkbutton(row1, text="Show CS overlay",
                        variable=self._conn_str_show_var,
                        command=lambda: (self._on_conn_str_toggle(),
                                        self._build_sig_chips())).pack(
            side=tk.LEFT, padx=(0, 6))
        ttk.Label(row1, text="Measure:").pack(side=tk.LEFT, padx=(8, 2))
        self._cs_metric_rbs = {}
        for val in ("STG", "JBSI"):
            rb = ttk.Radiobutton(
                row1, text=val, value=val, variable=self._conn_str_metric_var,
                command=lambda: (self._clear_all_png_cache(),
                                 self.update_plot(),
                                 self._update_conn_str_label())
            )
            rb.pack(side=tk.LEFT, padx=(0, 6))
            self._cs_metric_rbs[val] = rb
        self._update_conn_str_metric_availability()

        # ── Row 2: CS value label + non-negative toggle ────────────────
        row_cs = ttk.Frame(cs_frame)
        row_cs.pack(fill=tk.X, anchor='w', pady=(2, 0))
        self._conn_str_label = ttk.Label(row_cs, text="CS: \u2014")
        self._conn_str_label.pack(side=tk.LEFT)
        ttk.Checkbutton(
            row_cs,
            text="non-negative",
            variable=self._conn_str_nonneg_var,
            command=lambda: (self._clear_all_png_cache(),
                             self.update_plot(),
                             self._update_conn_str_label(),
                             getattr(self, '_stats_panel', None)
                             and self._stats_panel.on_parent_display_option_changed()),
        ).pack(side=tk.LEFT, padx=(10, 0))

        # ── Row 3: Baseline radio + test-window checkbox ──────────────
        row2 = ttk.Frame(cs_frame)
        row2.pack(fill=tk.X, anchor='w', pady=(2, 0))
        ttk.Label(row2, text="Baseline:").pack(side=tk.LEFT)
        self._global_rb = None
        self._jitter_rb = None
        for val, lbl in [('conv', 'Conv'), ('tailed', 'Tailed'),
                         ('global', 'Global'), ('jitter', 'Jitter')]:
            rb = ttk.Radiobutton(row2, text=lbl, variable=self._conn_str_method_var,
                                 value=val, command=self._on_baseline_method_change)
            rb.pack(side=tk.LEFT, padx=3)
            if val == 'global':
                self._global_rb = rb
                rb.state(['disabled'])  # enabled when test window is on + CCG data exists
            if val == 'jitter':
                self._jitter_rb = rb
                rb.state(['disabled'])  # enabled when jitter data is cached for this pair
        ttk.Separator(row2, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        self._sig_test_window_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="test window",
                        variable=self._sig_test_window_var,
                        command=self._on_sig_toggle).pack(side=tk.LEFT, padx=2)
        ttk.Separator(row2, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        self._adaptive_tw_btn = ttk.Checkbutton(
            row2, text="adaptive test window",
            variable=self._adaptive_tw_var,
            command=self._on_adaptive_tw_toggle)
        self._adaptive_tw_btn.pack(side=tk.LEFT, padx=2)
        self._adaptive_tw_btn.state(['disabled'])

        # ── Row 3: conditional p-value buttons + description ─────────
        self._cs_pval_row = ttk.Frame(cs_frame)
        self._cs_pval_row.pack(fill=tk.X, anchor='w', pady=(2, 0))
        # BooleanVars created once; widgets rebuilt by _rebuild_cs_pval_row
        self._sig_conv_p_var  = tk.BooleanVar(value=False)
        self._sig_conv_pc_var = tk.BooleanVar(value=False)
        self._sig_jitter_pc_var = tk.BooleanVar(value=False)
        self._sig_jitter_pc_cb = None  # set in _rebuild_cs_pval_row
        self._rebuild_cs_pval_row()

        # ── Correlograms row ──────────────────────────────────────────
        acg_outer = ttk.Frame(parent)
        acg_outer.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 0))
        acg_hdr = ttk.Frame(acg_outer)
        acg_hdr.pack(fill=tk.X)
        self._acg_fold_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(acg_hdr, text="Correlograms",
                        variable=self._acg_fold_var,
                        command=lambda: self._toggle_fold(
                            self._acg_fold_var, acg_frame, 'Correlograms', '▸ Correlograms',
                            acg_hdr)).pack(side=tk.LEFT)
        acg_frame = ttk.Frame(acg_outer, padding=4)
        acg_frame.pack(fill=tk.X)
        self._acg_inner_frame = acg_frame

        # CCG tri-state: ■ solid → □ outline → X hidden → ■ ...
        self._ccg_show_var = tk.BooleanVar(value=True)
        self._ccg_style_btn = tk.Label(
            acg_frame, text="■ CCG", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._ccg_style_btn.pack(side=tk.LEFT, padx=2)
        self._ccg_style_btn.bind('<Button-1>',
            lambda e: self._cycle_style('ccg'))

        self._baseline_style_btn = tk.Label(
            acg_frame, text="■ baseline", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._baseline_style_btn.pack(side=tk.LEFT, padx=2)
        self._baseline_style_btn.bind('<Button-1>',
            lambda e: self._cycle_style('baseline'))

        ttk.Separator(acg_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        ttk.Label(acg_frame, text="ACG", font=('Arial', 9, 'bold')).pack(
            side=tk.LEFT, padx=(0, 4))

        # Ref/Tgt ACG tri-state buttons (default: hidden X)
        self._acg_ref_var = tk.BooleanVar(value=False)
        self._ref_style_btn = tk.Label(
            acg_frame, text="□ ref", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._ref_style_btn.pack(side=tk.LEFT, padx=2)
        self._ref_style_btn.bind('<Button-1>',
            lambda e: self._cycle_style_acg('ref'))
        self._acg_tgt_var = tk.BooleanVar(value=False)
        self._tgt_style_btn = tk.Label(
            acg_frame, text="□ tgt", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._tgt_style_btn.pack(side=tk.LEFT, padx=2)
        self._tgt_style_btn.bind('<Button-1>',
            lambda e: self._cycle_style_acg('tgt'))

        self._peak_wf_style_btn = tk.Label(
            acg_frame, text="X ref peak", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._peak_wf_style_btn.pack(side=tk.LEFT, padx=2)
        self._peak_wf_style_btn.bind('<Button-1>',
            lambda e: self._cycle_style_acg('peak_wf'))

        ttk.Label(acg_frame, text="ACG deconvolution",
                  font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(10, 4))
        # Non-mutually-exclusive toggles: can deconvolve ref, tgt, or both.
        self._acg_deconv_ref_var = tk.BooleanVar(value=False)
        self._acg_deconv_tgt_var = tk.BooleanVar(value=False)
        self._deconv_ref_btn = tk.Label(
            acg_frame, text="□ ref", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._deconv_ref_btn.pack(side=tk.LEFT, padx=2)
        self._deconv_ref_btn.bind('<Button-1>',
            lambda e: self._toggle_acg_deconv('ref'))
        self._deconv_tgt_btn = tk.Label(
            acg_frame, text="□ tgt", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._deconv_tgt_btn.pack(side=tk.LEFT, padx=2)
        self._deconv_tgt_btn.bind('<Button-1>',
            lambda e: self._toggle_acg_deconv('tgt'))

        ttk.Separator(acg_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        # Ref ACG Y scale
        ttk.Label(acg_frame, text="ref Y:", font=('Arial', 8)).pack(
            side=tk.LEFT, padx=(0, 1))
        self._acg_yscale_ref_var = tk.DoubleVar(value=1.0)
        ttk.Scale(acg_frame, from_=0.1, to=1.5,
                  variable=self._acg_yscale_ref_var,
                  orient=tk.HORIZONTAL, length=50,
                  command=lambda v: self._on_acg_scale_change()
                  ).pack(side=tk.LEFT, padx=1)
        self._acg_scale_ref_entry = ttk.Entry(acg_frame, width=5,
                                               font=('Courier', 8))
        self._acg_scale_ref_entry.insert(0, "1.0")
        self._acg_scale_ref_entry.pack(side=tk.LEFT, padx=(0, 1))
        self._acg_scale_ref_entry.bind('<Return>',
            lambda e: self._on_acg_entry_submit('ref'))
        self._acg_scale_ref_entry.bind('<FocusOut>',
            lambda e: self._on_acg_entry_submit('ref'))

        # Tgt ACG Y scale
        ttk.Label(acg_frame, text="tgt Y:", font=('Arial', 8)).pack(
            side=tk.LEFT, padx=(0, 1))
        self._acg_yscale_tgt_var = tk.DoubleVar(value=1.0)
        ttk.Scale(acg_frame, from_=0.1, to=1.5,
                  variable=self._acg_yscale_tgt_var,
                  orient=tk.HORIZONTAL, length=50,
                  command=lambda v: self._on_acg_scale_change()
                  ).pack(side=tk.LEFT, padx=1)
        self._acg_scale_tgt_entry = ttk.Entry(acg_frame, width=5,
                                               font=('Courier', 8))
        self._acg_scale_tgt_entry.insert(0, "1.0")
        self._acg_scale_tgt_entry.pack(side=tk.LEFT, padx=(0, 1))
        self._acg_scale_tgt_entry.bind('<Return>',
            lambda e: self._on_acg_entry_submit('tgt'))
        self._acg_scale_tgt_entry.bind('<FocusOut>',
            lambda e: self._on_acg_entry_submit('tgt'))

        self._acg_match_ccg_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(acg_frame, text="Match ACG to CCG scale",
                        variable=self._acg_match_ccg_var,
                        command=self._on_sig_toggle).pack(side=tk.LEFT, padx=4)

        # ── New line: Extend-window toggle (recompute this pair only) ──
        extend_row = ttk.Frame(acg_outer)
        extend_row.pack(fill=tk.X, anchor='w', pady=(2, 0))
        self._extend_enable_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            extend_row, text="Extend",
            variable=self._extend_enable_var,
            command=self._on_extend_toggle,
        ).pack(side=tk.LEFT, padx=(2, 4))
        ttk.Label(extend_row, text="ms:").pack(side=tk.LEFT)
        self._extend_ms_var = tk.IntVar(value=50)
        self._extend_ms_spin = ttk.Spinbox(
            extend_row, from_=1, to=5000, increment=1, width=6,
            textvariable=self._extend_ms_var,
            command=self._on_extend_ms_commit,
        )
        self._extend_ms_spin.pack(side=tk.LEFT, padx=(2, 6))
        self._extend_ms_spin.bind('<Return>', lambda e: self._on_extend_ms_commit())
        self._extend_ms_spin.bind('<FocusOut>', lambda e: self._on_extend_ms_commit())

        ttk.Label(extend_row, text="resolution (ms):").pack(side=tk.LEFT, padx=(8, 2))
        self._extend_bin_ms_var = tk.IntVar(value=1)
        self._extend_bin_spin = ttk.Spinbox(
            extend_row, from_=1, to=50, increment=1, width=4,
            textvariable=self._extend_bin_ms_var,
            command=self._on_extend_ms_commit,
        )
        self._extend_bin_spin.pack(side=tk.LEFT, padx=(2, 6))
        self._extend_bin_spin.bind('<Return>', lambda e: self._on_extend_ms_commit())
        self._extend_bin_spin.bind('<FocusOut>', lambda e: self._on_extend_ms_commit())

        self._update_style_btns()
        self._update_acg_deconv_btns()

    def _toggle_acg_deconv(self, which: str):
        """Toggle ACG deconvolution view for ref/tgt (can enable both)."""
        if which == 'ref' and hasattr(self, '_acg_deconv_ref_var'):
            self._acg_deconv_ref_var.set(not bool(self._acg_deconv_ref_var.get()))
        elif which == 'tgt' and hasattr(self, '_acg_deconv_tgt_var'):
            self._acg_deconv_tgt_var.set(not bool(self._acg_deconv_tgt_var.get()))
        self._update_acg_deconv_btns()
        self._clear_all_png_cache()
        self.update_plot()

    def _update_acg_deconv_btns(self):
        if not hasattr(self, '_acg_deconv_ref_var') or not hasattr(self, '_acg_deconv_tgt_var'):
            return
        cur_ref = bool(self._acg_deconv_ref_var.get())
        cur_tgt = bool(self._acg_deconv_tgt_var.get())
        btn = getattr(self, '_deconv_ref_btn', None)
        if btn is not None:
            btn.config(text=('■ ' if cur_ref else '□ ') + 'ref')
        btn = getattr(self, '_deconv_tgt_btn', None)
        if btn is not None:
            btn.config(text=('■ ' if cur_tgt else '□ ') + 'tgt')

    def _on_extend_toggle(self):
        """Toggle extend-window rendering for the current pair."""
        # Changing extend mode invalidates PNG + computed-ccg caches.
        try:
            self._extend_cache.clear()
        except Exception:
            pass
        self._clear_all_png_cache()
        self.update_plot()

    def _on_extend_ms_commit(self):
        """Commit extend-ms value and refresh if Extend is enabled."""
        if not hasattr(self, '_extend_ms_var'):
            return
        try:
            ms = int(self._extend_ms_var.get())
        except Exception:
            ms = 50
        ms = max(1, min(5000, ms))
        try:
            self._extend_ms_var.set(ms)
        except Exception:
            pass
        # Extend bin resolution (ms)
        if hasattr(self, "_extend_bin_ms_var"):
            try:
                bms = int(self._extend_bin_ms_var.get())
            except Exception:
                bms = 1
            bms = max(1, min(50, bms))
            try:
                self._extend_bin_ms_var.set(bms)
            except Exception:
                pass
        if bool(getattr(self, '_extend_enable_var', None) and self._extend_enable_var.get()):
            try:
                self._extend_cache.clear()
            except Exception:
                pass
            self._clear_all_png_cache()
            self.update_plot()

    def _on_acg_scale_change(self):
        """Correlogram Y-scale slider changed — sync entry boxes."""
        if not hasattr(self, '_acg_yscale_ref_var') or not hasattr(self, '_acg_yscale_tgt_var'):
            return  # guard during widget init
        ref_val = self._acg_yscale_ref_var.get()
        tgt_val = self._acg_yscale_tgt_var.get()
        # Update entry boxes to reflect slider value
        if hasattr(self, '_acg_scale_ref_entry'):
            self._acg_scale_ref_entry.delete(0, tk.END)
            self._acg_scale_ref_entry.insert(0, f"{ref_val:.1f}")
        if hasattr(self, '_acg_scale_tgt_entry'):
            self._acg_scale_tgt_entry.delete(0, tk.END)
            self._acg_scale_tgt_entry.insert(0, f"{tgt_val:.1f}")
        if (self._acg_ref_var.get() or self._acg_tgt_var.get()
                or self._peak_wf_var.get()):
            self._clear_all_png_cache()
            self.update_plot()

    def _on_acg_entry_submit(self, which):
        """User typed a value in the ACG Y-scale entry box."""
        if which == 'ref':
            entry = self._acg_scale_ref_entry
            var = self._acg_yscale_ref_var
        else:
            entry = self._acg_scale_tgt_entry
            var = self._acg_yscale_tgt_var
        try:
            val = float(entry.get())
            if val < 0.01:
                val = 0.01
            var.set(val)
            entry.delete(0, tk.END)
            entry.insert(0, f"{val:.1f}")
            if (self._acg_ref_var.get() or self._acg_tgt_var.get()
                    or self._peak_wf_var.get()):
                self._clear_all_png_cache()
                self.update_plot()
        except ValueError:
            # Reset entry to current slider value
            entry.delete(0, tk.END)
            entry.insert(0, f"{var.get():.1f}")

    def _acg_var_get(self, name, default=None):
        """Safely read a Tk variable that may not exist yet."""
        v = getattr(self, name, None)
        return v.get() if v is not None else default

    def _sig(self, name):
        """Read a significance/display toggle BooleanVar by short name."""
        _map = {
            'conv_p':      '_sig_conv_p_var',
            'conv_pc':     '_sig_conv_pc_var',
            'test_window': '_sig_test_window_var',
            'jitter_pc':   '_sig_jitter_pc_var',
        }
        var = getattr(self, _map[name], None)
        return var.get() if var is not None else False

    # All attr names whose values form the "cache configuration"
    _CACHE_CONFIG_ATTRS = (
        '_sig_conv_p_var', '_sig_conv_pc_var', '_sig_test_window_var',
        '_sig_jitter_pc_var',
        '_conn_str_method_var', '_conn_str_show_var', '_adaptive_tw_var',
        '_acg_ref_var', '_acg_tgt_var',
        '_acg_deconv_ref_var', '_acg_deconv_tgt_var',
        '_extend_enable_var', '_extend_ms_var', '_extend_bin_ms_var',
        '_peak_wf_var',
        '_acg_yscale_ref_var', '_acg_yscale_tgt_var',
        '_acg_match_ccg_var', '_ccg_show_var', '_baseline_show_var',
        '_line_ccg_var', '_line_baseline_var',
        '_line_ref_var', '_line_tgt_var', '_line_peak_wf_var', '_line_jitter_var',
    )

    def _current_display_config(self) -> dict:
        """Snapshot of every display-state var that affects PNG rendering."""
        cfg = {}
        for attr in self._CACHE_CONFIG_ATTRS:
            v = getattr(self, attr, None)
            cfg[attr] = v.get() if v is not None else None
        cfg['active_norms'] = sorted(n.name for n in self.active_norms)
        cfg['active_alpha'] = self.active_alpha
        return cfg

    def _display_matches_cache_config(self) -> bool:
        """True when the current display state matches the saved cache configuration."""
        if self._cache_config is None:
            return False
        cur = self._current_display_config()
        return cur == self._cache_config

    def _on_sig_toggle(self):
        """Clear PNG cache and redraw (vars are read live via _sig())."""
        self._clear_all_png_cache()
        self.update_plot()

    def _update_jitter_sig_buttons(self):
        """Enable/disable jitter significance buttons based on cache.

        When the current pair has cached jitter data, auto-enable the
        display toggles so the overlay is visible immediately.
        """
        inds = self.all_inds[self.current_pair_idx] if self.current_pair_idx < len(self.all_inds) else None
        has_jitter = False
        if inds is not None:
            has_jitter = self._jitter_cache.get(
                (int(inds[0]), int(inds[1]), 'lo', self._jitter_seg())) is not None
        if not has_jitter:
            self._line_jitter_var.set(False)
        # p-corrected checkbutton (only exists in UI when method='jitter')
        cb = getattr(self, '_sig_jitter_pc_cb', None)
        if cb is not None:
            try:
                cb.state(['!disabled'] if has_jitter else ['disabled'])
            except Exception:
                pass
        # Jitter radio button: gray out when no jitter data for this pair
        jrb = getattr(self, '_jitter_rb', None)
        if jrb is not None:
            try:
                jrb.state(['!disabled'] if has_jitter else ['disabled'])
            except Exception:
                pass
            if not has_jitter and self._conn_str_method_var.get() == 'jitter':
                self._conn_str_method_var.set('conv')
                self._rebuild_cs_pval_row()
        self._update_global_baseline_availability()
        self._update_adaptive_tw_availability()

    def setup_norm_panel(self, parent):
        norm_inner, self._norm_fold_var = self._make_collapsible_panel(parent, "Normalization")

        saved_norms = self._ui_state_cache.get('active_norms', [])
        options = [
            (NormalizeBy.REF_FRATE,    "Ref f-rate"),
            (NormalizeBy.TARGET_FRATE, "Tgt f-rate"),
            (NormalizeBy.TIME_SPAN,    "Time (hr)"),
            (NormalizeBy.TIME_SECOND,  "Time (sec)"),
            (NormalizeBy.TOTAL_AREA,   "CCG total area"),
            (NormalizeBy.BASELINE,     "Subtract baseline"),
        ]

        # Row 1 (top): wrapping normalization toggles only
        top_row = ttk.Frame(norm_inner)
        top_row.pack(fill=tk.X)
        # Wrapping body — checkbuttons are repositioned via _rewrap_norm_checkbuttons
        norm_body = tk.Frame(top_row, height=22)
        norm_body.pack(side=tk.LEFT, fill=tk.X, expand=True)
        norm_body.pack_propagate(False)
        self._norm_body = norm_body
        self._norm_checkbuttons: list = []   # [(widget, NormalizeBy), ...]

        for nm, label in options:
            if self.neurons is None and nm in (
                    NormalizeBy.REF_FRATE, NormalizeBy.TARGET_FRATE):
                continue
            var = tk.BooleanVar(value=(nm.name in saved_norms))
            self.norm_vars[nm] = var
            cb = ttk.Checkbutton(norm_body, text=label, variable=var,
                                 command=self._on_norm_toggle)
            self._norm_checkbuttons.append((cb, nm))

        norm_body.bind('<Configure>',
                       lambda e: self.root.after_idle(self._rewrap_norm_checkbuttons))

        # Row 2 (bottom): scale toggles + Normalize All (no wrapping)
        bottom_row = ttk.Frame(norm_inner)
        bottom_row.pack(fill=tk.X, pady=(2, 0))
        self._pair_scale_var = tk.BooleanVar(
            value=self._ui_state_cache.get('pair_scale', False))
        ttk.Checkbutton(bottom_row, text="Same scale (pair)",
                        variable=self._pair_scale_var,
                        command=self._on_pair_scale_toggle).pack(side=tk.LEFT, padx=4)
        self._sess_scale_var = tk.BooleanVar(
            value=self._ui_state_cache.get('sess_scale', False))
        ttk.Checkbutton(bottom_row, text="Same scale (session)",
                        variable=self._sess_scale_var,
                        command=self._on_session_scale_toggle).pack(side=tk.LEFT, padx=4)
        ttk.Button(bottom_row, text="Normalize all…",
                   command=self._finalize_normalization).pack(side=tk.RIGHT, padx=6)

        # Trigger initial layout after window is realized
        self.root.after(150, self._rewrap_norm_checkbuttons)

    def _rewrap_norm_checkbuttons(self):
        """Reposition normalization checkboxes into wrapping rows."""
        if not hasattr(self, '_norm_body') or not hasattr(self, '_norm_checkbuttons'):
            return
        body = self._norm_body
        body.update_idletasks()
        avail_w = body.winfo_width()
        if avail_w <= 1:
            body.after(100, self._rewrap_norm_checkbuttons)
            return

        PAD_X, PAD_Y = 2, 1
        x, y, row_h = PAD_X, PAD_Y, 0
        for cb, _ in self._norm_checkbuttons:
            if not cb.winfo_exists():
                continue
            cb.update_idletasks()
            ww = cb.winfo_reqwidth()
            wh = cb.winfo_reqheight()
            if x + ww > avail_w - PAD_X and x > PAD_X:
                y += row_h + PAD_Y
                x, row_h = PAD_X, 0
            cb.place(x=x, y=y)
            x += ww + PAD_X
            row_h = max(row_h, wh)

        total_h = y + row_h + PAD_Y * 2
        body.configure(height=max(total_h, 22))

    def setup_jitter_panel(self, parent):
        """Jitter controls: run jitter on demand for the current pair."""
        jitter_inner, self._jitter_fold_var = self._make_collapsible_panel(parent, "Jitter")
        ttk.Label(jitter_inner, text="n=").pack(side=tk.LEFT)
        self._njitter_var = tk.IntVar(value=100)
        ttk.Spinbox(jitter_inner, from_=10, to=5000, increment=50,
                    textvariable=self._njitter_var, width=6).pack(
            side=tk.LEFT, padx=2)
        self._jitter_btn_text = tk.StringVar(value="Run Jitter")
        ttk.Button(jitter_inner, textvariable=self._jitter_btn_text,
                   command=self._on_run_jitter).pack(side=tk.LEFT, padx=6)
        ttk.Button(jitter_inner, text="Clear",
                   command=self._on_clear_jitter).pack(side=tk.LEFT)
        ttk.Button(jitter_inner, text="Save",
                   command=self._on_save_jitter).pack(side=tk.LEFT, padx=(4, 0))
        # Resolution selector (to the right of Save)
        ttk.Label(jitter_inner, text="Resolution:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Checkbutton(jitter_inner, text="lo", variable=self._jitter_run_lo_var).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Checkbutton(jitter_inner, text="hi", variable=self._jitter_run_hi_var).pack(
            side=tk.LEFT, padx=(0, 0))

    def setup_spike_attrib_panel(self, parent):
        """Spike attribution controls: toggle on row 1, bin+set+back on row 2."""
        sa_inner, self._sa_fold_var = self._make_collapsible_panel(parent, "Spike Attribution")

        # Row 1: toggle only
        row1 = ttk.Frame(sa_inner)
        row1.pack(fill=tk.X)
        self._sa_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row1, text="Allow spike attribution",
                        variable=self._sa_enabled_var,
                        command=self._on_sa_toggle).pack(side=tk.LEFT)

        # Row 2: bin input, Set, Back to CCG
        row2 = ttk.Frame(sa_inner)
        row2.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(row2, text="Bin (ms):").pack(side=tk.LEFT)
        self._sa_bin_var = tk.StringVar(value="0")
        self._sa_bin_entry = ttk.Entry(row2, textvariable=self._sa_bin_var,
                                       width=6, state='disabled')
        self._sa_bin_entry.pack(side=tk.LEFT, padx=2)
        self._sa_bin_entry.bind('<Return>', lambda _: self._on_sa_set())
        self._sa_set_btn = ttk.Button(row2, text="Set",
                                      command=self._on_sa_set,
                                      state='disabled')
        self._sa_set_btn.pack(side=tk.LEFT, padx=4)

    # ------------------------------------------------------------------
    # Spike Attribution logic
    # ------------------------------------------------------------------

    def _on_sa_toggle(self):
        """Toggle unlock: enable/disable the bin entry and Set button."""
        enabled = self._sa_enabled_var.get()
        self._sa_enabled = enabled
        state = 'normal' if enabled else 'disabled'
        self._sa_bin_entry.config(state=state)
        self._sa_set_btn.config(state=state)
        if not enabled:
            self._sa_spike_pairs = []
            self._sa_selected_idx = -1
            if hasattr(self, 'left_container'):
                self.left_container.spike_pairs.clear()
            self.update_plot()

    def _on_sa_set(self):
        """Query spike pairs for the current CCG pair + bin offset."""
        if not self._sa_enabled or self.neurons is None:
            return
        if self.current_pair_idx >= len(self.all_inds):
            return
        try:
            self._sa_bin_ms = float(self._sa_bin_var.get())
        except ValueError:
            if hasattr(self, 'left_container'):
                self.left_container.spike_pairs._spike_pairs_count.set("Invalid bin")
            return
        inds = self.all_inds[self.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        self._compute_spike_pairs(ref, tgt, self._sa_bin_ms)
        if hasattr(self, 'left_container'):
            self.left_container.spike_pairs.activate()

    def _exit_spike_attribution_view(self):
        """Exit spike attribution raster and restore normal CCG view."""
        if self._sa_selected_idx < 0:
            return  # not in raster view
        self._sa_selected_idx = -1
        self.update_plot()

    def _compute_spike_pairs(self, ref: int, tgt: int, bin_ms: float):
        """Find all spike pairs contributing to the given CCG time bin.

        For a CCG bin at lag ``bin_ms``, a spike pair (ref_t, tgt_t) contributes
        when ``tgt_t - ref_t`` falls within the bin's half-open interval.
        """
        conf = self.ccg_data.conf if self.ccg_data is not None else None
        if conf is None:
            return

        # Infer bin width from current resolution
        n_bins = self.ccg_data.ccg.shape[-1] if self.ccg_data.ccg is not None else conf.nbins
        bin_size = conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size

        # Bin center in seconds; bin edges
        lag_sec = bin_ms / 1000.0
        bin_lo = lag_sec - bin_size / 2.0
        bin_hi = lag_sec + bin_size / 2.0

        # Get spike trains
        ref_spikes = self.neurons.spiketrains[ref]
        tgt_spikes = self.neurons.spiketrains[tgt]

        # Optionally restrict to current segment's time window
        seg = self.current_segment
        if seg < self.n_segments:
            et = self.ccg_pointer.edge_times
            t0 = float(et.iloc[seg]['start']) if 'start' in et.columns else None
            t1 = float(et.iloc[seg]['stop']) if 'stop' in et.columns else None
            if t0 is not None and t1 is not None:
                ref_spikes = ref_spikes[(ref_spikes >= t0) & (ref_spikes <= t1)]
                tgt_spikes = tgt_spikes[(tgt_spikes >= t0) & (tgt_spikes <= t1)]

        # Find all (ref_spike_time, tgt_spike_time) where lag is in [bin_lo, bin_hi)
        pairs = []
        tgt_sorted = np.sort(tgt_spikes)
        for rt in ref_spikes:
            # Target spikes at absolute time rt + bin_lo to rt + bin_hi
            lo = rt + bin_lo
            hi = rt + bin_hi
            idx_lo = np.searchsorted(tgt_sorted, lo, side='left')
            idx_hi = np.searchsorted(tgt_sorted, hi, side='right')
            for j in range(idx_lo, idx_hi):
                pairs.append((float(rt), float(tgt_sorted[j])))

        self._sa_spike_pairs = pairs
        self._sa_selected_idx = -1

        if hasattr(self, 'left_container'):
            self.left_container.spike_pairs.populate(pairs)
        else:
            # Pre-container fallback
            self._sa_listbox.delete(0, tk.END)
            for i, (rt, tt) in enumerate(pairs):
                lag_ms = (tt - rt) * 1000.0
                self._sa_listbox.insert(
                    tk.END,
                    f"{i+1:>5}  ref {rt:10.4f}  tgt {tt:10.4f}  lag {lag_ms:+6.2f}ms")
            self._sa_count_var.set(f"{len(pairs)} spike pairs")

    def _on_sa_pair_click(self, _event=None):
        """Handle click on a spike pair — show raster in center panel."""
        sel = self._sa_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self._sa_spike_pairs):
            return
        self._sa_selected_idx = idx
        self._draw_sa_raster(idx)

    def _draw_sa_raster(self, idx: int):
        """Draw a 2-row raster of ref/tgt spike trains around the selected pair."""
        ref_t, tgt_t = self._sa_spike_pairs[idx]
        inds = self.all_inds[self.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])

        # Time window centered on the ref spike
        center = ref_t
        win = self._sa_raster_window  # ±50ms in seconds
        t0 = center - win
        t1 = center + win

        ref_spikes = self.neurons.spiketrains[ref]
        tgt_spikes = self.neurons.spiketrains[tgt]
        ref_win = ref_spikes[(ref_spikes >= t0) & (ref_spikes <= t1)]
        tgt_win = tgt_spikes[(tgt_spikes >= t0) & (tgt_spikes <= t1)]

        ref_label = f"Ref {self._shank_label(ref)}"
        tgt_label = f"Tgt {self._shank_label(tgt)}"

        self.fig.clear()
        ax_ref = self.fig.add_subplot(211)
        ax_tgt = self.fig.add_subplot(212, sharex=ax_ref)

        # Ref raster
        if len(ref_win):
            ax_ref.eventplot([ref_win - center], lineoffsets=0,
                             linelengths=0.8, colors='#1565C0')
        # Highlight the selected ref spike
        ax_ref.axvline(0, color='#E53935', lw=1.5, ls='--', alpha=0.7)
        ax_ref.set_ylabel(ref_label, fontsize=9)
        ax_ref.set_yticks([])
        ax_ref.set_title(
            f"Spike pair #{idx+1}: ref={ref_t:.4f}s  tgt={tgt_t:.4f}s  "
            f"lag={(tgt_t - ref_t)*1000:.2f}ms",
            fontsize=9)

        # Tgt raster
        if len(tgt_win):
            ax_tgt.eventplot([tgt_win - center], lineoffsets=0,
                             linelengths=0.8, colors='#2E7D32')
        # Highlight the selected tgt spike
        tgt_offset = tgt_t - center
        ax_tgt.axvline(tgt_offset, color='#E53935', lw=1.5, ls='--', alpha=0.7)
        ax_tgt.set_ylabel(tgt_label, fontsize=9)
        ax_tgt.set_yticks([])
        ax_tgt.set_xlabel("Time relative to ref spike (s)", fontsize=9)
        ax_tgt.set_xlim(-win, win)

        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_spike_pairs_raster(self, idx: int, spike_pairs: list):
        """Called from SpikePairsPanel when a spike pair is clicked.

        Stores the pairs list on self for _draw_sa_raster to index into,
        then delegates to the existing drawing logic.
        """
        self._sa_spike_pairs = spike_pairs
        self._sa_selected_idx = idx
        self._draw_sa_raster(idx)

    def setup_waveforms_panel(self, parent):
        """Waveform pane inside the CCG horizontal split — hidden by default."""
        self.wave_frame = ttk.LabelFrame(self._plot_pw, text="Waveforms")
        # Not added to _plot_pw yet — toggled via Panels menu / Ctrl+E
        self.wave_fig = Figure(figsize=(4, 5), tight_layout=True)
        self.wave_ax = self.wave_fig.add_subplot(111)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_fig, master=self.wave_frame)
        self.wave_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    _CS_METHOD_DESCRIPTIONS = {
        'conv':   "Conv baseline: smoothed null (EranConv)",
        'tailed': "Tailed: ACG deconvolution, tail-bin baseline",
        'global': "Global: max bin outside test window as baseline",
        'jitter': "Jitter: surrogate spike baseline",
    }

    def _rebuild_cs_pval_row(self):
        """Repopulate row 3 of the Baseline & CS panel based on current method."""
        for w in self._cs_pval_row.winfo_children():
            w.destroy()
        method = self._conn_str_method_var.get()
        if method == 'conv':
            ttk.Checkbutton(self._cs_pval_row, text="p",
                            variable=self._sig_conv_p_var,
                            command=self._on_sig_toggle).pack(side=tk.LEFT, padx=2)
            ttk.Checkbutton(self._cs_pval_row, text="p-corrected",
                            variable=self._sig_conv_pc_var,
                            command=self._on_sig_toggle).pack(side=tk.LEFT, padx=2)
        elif method == 'jitter':
            self._sig_jitter_pc_cb = ttk.Checkbutton(
                self._cs_pval_row, text="p-corrected",
                variable=self._sig_jitter_pc_var,
                command=self._on_sig_toggle)
            self._sig_jitter_pc_cb.pack(side=tk.LEFT, padx=2)
            self._sig_jitter_pc_cb.state(['disabled'])
        ttk.Label(self._cs_pval_row,
                  text=self._CS_METHOD_DESCRIPTIONS.get(method, ''),
                  font=('Arial', 8), foreground='#555').pack(side=tk.LEFT, padx=6)

    def _on_baseline_method_change(self):
        """Called when the Baseline radio button changes."""
        self._rebuild_cs_pval_row()
        self._conn_strength_cache.clear()
        self._clear_all_png_cache()
        self.update_plot()
        self._update_conn_str_label()

    # ── Right panel ────────────────────────────────────────────────────

    def setup_network_panel(self, parent):
        self.network_panel = NetworkPanel(parent, self)

    # ── Time slider panel ──────────────────────────────────────────────

    def setup_time_slider_panel(self, parent):
        """Full-width time-window selector — hidden by default."""
        self.time_slider_frame = ttk.LabelFrame(
            parent, text="Time Slider - Behavioral Epochs")
        # Not packed — shown when 'Time Slider' panel is enabled

        # Theme selector row
        theme_row = ttk.Frame(self.time_slider_frame)
        theme_row.pack(fill=tk.X, padx=4, pady=(2, 0))
        ttk.Label(theme_row, text="Theme:").pack(side=tk.LEFT)
        self._ts_theme_var = tk.StringVar(value='segments')
        self._ts_theme_combo = ttk.Combobox(
            theme_row, textvariable=self._ts_theme_var,
            values=['segments'], width=16, state='readonly')
        self._ts_theme_combo.pack(side=tk.LEFT, padx=4)
        self._ts_theme_combo.bind('<<ComboboxSelected>>', self._on_ts_theme_change)
        self._ts_theme_info_var = tk.StringVar(value="")
        ttk.Label(theme_row, textvariable=self._ts_theme_info_var,
                  font=('Courier', 8), foreground='#666').pack(
            side=tk.LEFT, padx=6)

        # ── Overlap label selector (inline, hidden until multi-label theme) ──
        self._ts_overlap_row = ttk.Frame(theme_row)
        # Not packed initially — shown inline when theme has multiple labels
        ttk.Label(self._ts_overlap_row, text="Show:").pack(side=tk.LEFT, padx=(0, 2))
        self._ts_label_var = tk.StringVar(value='All')
        self._ts_label_combo = ttk.Combobox(
            self._ts_overlap_row, textvariable=self._ts_label_var,
            values=['All'], width=12, state='readonly')
        self._ts_label_combo.pack(side=tk.LEFT)
        self._ts_label_combo.bind('<<ComboboxSelected>>', self._on_ts_label_change)
        ttk.Button(self._ts_overlap_row, text="All",
                   command=self._on_ts_label_reset).pack(side=tk.LEFT, padx=2)

        # ── Tool bar (right side of theme row) ──
        toolbar = ttk.Frame(theme_row)
        toolbar.pack(side=tk.RIGHT, padx=(0, 4))
        ttk.Button(toolbar, text="💾", width=2,
                   command=self._ts_save_custom_ccg).pack(side=tk.LEFT, padx=1)
        ttk.Button(toolbar, text="📂", width=2,
                   command=self._ts_load_custom_ccg).pack(side=tk.LEFT, padx=1)
        ttk.Label(toolbar, text="|", foreground='#BBB').pack(side=tk.LEFT, padx=2)
        self._ts_snap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Snap",
                        variable=self._ts_snap_var).pack(side=tk.LEFT, padx=2)
        self._ts_tool_var = tk.StringVar(value='none')
        self._ts_selection_btn = ttk.Checkbutton(
            toolbar, text="Zoom-in", variable=self._ts_tool_var,
            onvalue='selection', offvalue='none',
            command=self._on_ts_tool_change)
        self._ts_selection_btn.pack(side=tk.LEFT, padx=2)
        self._ts_lock_var = tk.BooleanVar(value=False)
        self._ts_lock_btn = ttk.Checkbutton(
            toolbar, text="\U0001F512 Lock", variable=self._ts_lock_var,
            command=self._on_ts_tool_change)
        self._ts_lock_btn.pack(side=tk.LEFT, padx=2)

        # ── Legend row (below theme row, populated by _ts_update_legend) ──
        self._ts_legend_frame = ttk.Frame(self.time_slider_frame)
        # Packed dynamically by _ts_update_legend

        # ── Main canvas ──
        self._ts_main_canvas_frame = ttk.Frame(self.time_slider_frame)
        top = self._ts_main_canvas_frame
        top.pack(fill=tk.X, padx=4, pady=(2, 0))

        self.ts_canvas = tk.Canvas(top, height=56, bg='#F5F5F5', cursor='crosshair')
        self.ts_canvas.pack(fill=tk.X, expand=True)
        self.ts_canvas.bind('<Configure>',      self._ts_redraw)
        self.ts_canvas.bind('<Button-1>',        self._ts_mouse_press)
        self.ts_canvas.bind('<B1-Motion>',       self._ts_mouse_drag)
        self.ts_canvas.bind('<ButtonRelease-1>', self._ts_mouse_release)

        # ── CCG time range bar (below main canvas, always visible) ──
        self._ts_ccg_ctrl = ttk.Frame(self.time_slider_frame)
        ccg_ctrl = self._ts_ccg_ctrl
        ccg_ctrl.pack(fill=tk.X, padx=4, pady=(2, 0))
        ttk.Label(ccg_ctrl, text="CCG time range",
                  font=('Arial', 8, 'bold'), foreground='#444').pack(
            side=tk.LEFT, padx=(0, 6))
        def _bind_resolve(entry, var):
            def _resolve(e=None):
                raw = var.get().strip().lower()
                if raw in ('start', 'end'):
                    var.set(raw)  # keep sentinel as-is
                    return
                try:
                    sec = self._ts_hms_to_sec(raw)
                    var.set(self._ts_sec_to_hms(sec))
                except (ValueError, IndexError):
                    pass
            entry.bind('<Return>', _resolve)
            entry.bind('<FocusOut>', _resolve)

        ttk.Label(ccg_ctrl, text="Start:").pack(side=tk.LEFT)
        self._ts_start_var = tk.StringVar(value="00:00:00")
        _e = ttk.Entry(ccg_ctrl, textvariable=self._ts_start_var, width=10)
        _e.pack(side=tk.LEFT, padx=2)
        _bind_resolve(_e, self._ts_start_var)
        ttk.Label(ccg_ctrl, text="End:").pack(side=tk.LEFT, padx=(6, 0))
        self._ts_end_var = tk.StringVar(value="00:00:00")
        _e = ttk.Entry(ccg_ctrl, textvariable=self._ts_end_var, width=10)
        _e.pack(side=tk.LEFT, padx=2)
        _bind_resolve(_e, self._ts_end_var)
        ttk.Button(ccg_ctrl, text="Set",
                   command=self._on_time_slider_set).pack(side=tk.LEFT, padx=4)
        ttk.Label(ccg_ctrl, text="Name:").pack(side=tk.LEFT, padx=(8, 0))
        self._ts_name_var = tk.StringVar(value="")
        ttk.Entry(ccg_ctrl, textvariable=self._ts_name_var,
                  width=14).pack(side=tk.LEFT, padx=2)
        ttk.Label(ccg_ctrl, text="Splits:").pack(side=tk.LEFT, padx=(6, 0))
        self._ts_splits_var = tk.IntVar(value=1)
        ttk.Spinbox(ccg_ctrl, from_=1, to=99, textvariable=self._ts_splits_var,
                    width=3).pack(side=tk.LEFT, padx=2)
        ttk.Label(ccg_ctrl, text="Overlap(s):").pack(side=tk.LEFT, padx=(4, 0))
        self._ts_overlap_sec_var = tk.StringVar(value="0")
        ttk.Entry(ccg_ctrl, textvariable=self._ts_overlap_sec_var,
                  width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(ccg_ctrl, text="Clear",
                   command=self._on_time_slider_clear).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(ccg_ctrl, text="Apply to Multiple Sessions",
                   command=self._on_time_slider_apply_multiple_sessions).pack(side=tk.LEFT, padx=(2, 2))
        self._ts_status_var = tk.StringVar(value="")
        ttk.Label(ccg_ctrl, textvariable=self._ts_status_var,
                  font=('Courier', 8), foreground='#555').pack(
            side=tk.LEFT, padx=8)

        # ── Zoom detail canvas (hidden until zoom is active) ──
        self._ts_zoom_frame = ttk.Frame(self.time_slider_frame)
        # Not packed initially — shown when zoom region is set
        # Radiating lines canvas (thin strip between main and zoom)
        self._ts_radiate_canvas = tk.Canvas(
            self._ts_zoom_frame, height=16, bg='#FEFEFE',
            highlightthickness=0)
        self._ts_radiate_canvas.pack(fill=tk.X, expand=True, side=tk.TOP)
        self._ts_zoom_canvas = tk.Canvas(
            self._ts_zoom_frame, height=56, bg='#FAFAFA', cursor='crosshair')
        self._ts_zoom_canvas.pack(fill=tk.X, expand=True)
        self._ts_zoom_canvas.bind('<Configure>', self._ts_zoom_redraw)

        # ── Zoom time range bar (below zoom canvas, shown with zoom) ──
        self._ts_zoom_ctrl = ttk.Frame(self._ts_zoom_frame)
        self._ts_zoom_ctrl.pack(fill=tk.X, padx=4, pady=(2, 0))
        ttk.Label(self._ts_zoom_ctrl, text="Zoom range",
                  font=('Arial', 8, 'bold'), foreground='#444').pack(
            side=tk.LEFT, padx=(0, 6))
        ttk.Label(self._ts_zoom_ctrl, text="Start:").pack(side=tk.LEFT)
        self._ts_zoom_start_var = tk.StringVar(value="00:00:00")
        _ze = ttk.Entry(self._ts_zoom_ctrl, textvariable=self._ts_zoom_start_var,
                        width=10)
        _ze.pack(side=tk.LEFT, padx=2)
        _bind_resolve(_ze, self._ts_zoom_start_var)
        ttk.Label(self._ts_zoom_ctrl, text="End:").pack(
            side=tk.LEFT, padx=(6, 0))
        self._ts_zoom_end_var = tk.StringVar(value="00:00:00")
        _ze = ttk.Entry(self._ts_zoom_ctrl, textvariable=self._ts_zoom_end_var,
                        width=10)
        _ze.pack(side=tk.LEFT, padx=2)
        _bind_resolve(_ze, self._ts_zoom_end_var)
        ttk.Button(self._ts_zoom_ctrl, text="Set",
                   command=self._on_zoom_range_set).pack(
            side=tk.LEFT, padx=4)

    # ── Bottom bar ─────────────────────────────────────────────────────

    def setup_bottom_panel(self):
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)
        ttk.Button(bottom_frame, text="Save Selections",
                   command=self._quick_save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bottom_frame, text="Quit",
                   command=self._on_close).pack(side=tk.RIGHT, padx=5)
        self.stats_var = tk.StringVar(value=self._compute_stats_str())
        ttk.Label(bottom_frame, textvariable=self.stats_var,
                  font=('Courier', 9)).pack(side=tk.LEFT, padx=8)
        self._pair_info_var = tk.StringVar(value="")
        ttk.Label(bottom_frame, textvariable=self._pair_info_var,
                  font=('Courier', 9)).pack(side=tk.LEFT, padx=8)

    # ------------------------------------------------------------------
    # Panel toggle helpers
    # ------------------------------------------------------------------

    def _toggle_panel(self, name):
        """Show or hide a panel based on its BooleanVar."""
        print(f"[CCGReviewUI] _toggle_panel({name!r})")
        try:
            self._toggle_panel_impl(name)
            self._save_ui_state()
        except Exception as ex:
            print(f"[CCGReviewUI] _toggle_panel error for {name!r}: {ex}")
            traceback.print_exc()

    def _toggle_panel_impl(self, name):
        """Inner implementation of panel toggling."""
        show = self._panel_vars[name].get()
        if name in ('Pair Selection', 'CCG', 'Probe Network'):
            frame_map = {
                'Pair Selection': self._left_frame,
                'CCG':            self._center_frame,
                'Probe Network':  self._right_frame,
            }
            order   = ['Pair Selection', 'CCG', 'Probe Network']
            weights = {'Pair Selection': 0, 'CCG': 1, 'Probe Network': 0}
            frame = frame_map[name]
            if show:
                pos = sum(1 for n in order[:order.index(name)]
                          if self._panel_vars[n].get())
                current_panes = self._paned.panes()
                if str(frame) in current_panes:
                    pass  # already managed, nothing to do
                elif pos < len(current_panes):
                    self._paned.insert(pos, frame, weight=weights[name])
                else:
                    self._paned.add(frame, weight=weights[name])
            else:
                if str(frame) in self._paned.panes():
                    self._paned.forget(frame)
        elif name == 'Waveforms':
            self._toggle_waveforms_panel()
        elif name == 'Time Slider':
            self._toggle_time_slider()
        elif name == 'Group Hotkeys':
            if show:
                self._refresh_hotkeys_bar()
                # Pack after the tool-strip (setup_menu frame), before the main area
                self._hotkeys_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 2),
                                       before=self._main_frame)
            else:
                self._hotkeys_bar.pack_forget()

    def _toggle_waveforms_panel(self):
        self._waveforms_visible = self._panel_vars['Waveforms'].get()
        if self._waveforms_visible:
            self._plot_pw.add(self.wave_frame, stretch='never', width=280)
            self._draw_waveforms()
        else:
            try:
                self._plot_pw.forget(self.wave_frame)
            except tk.TclError:
                pass

    def _toggle_time_slider(self):
        show = self._panel_vars['Time Slider'].get()
        print(f"[CCGReviewUI] _toggle_time_slider show={show}")
        if show:
            try:
                self.time_slider_frame.pack(
                    in_=self._main_frame, side=tk.TOP,
                    fill=tk.X, before=self._paned, pady=(0, 4))
                self._ts_discover_themes()
                self._ts_init_times()
                print(f"[CCGReviewUI]   epoch_bounds={len(self._ts_epoch_bounds)} "
                      f"total_sec={self._ts_total_sec:.1f}")
                self._ts_redraw()
            except Exception as ex:
                print(f"[CCGReviewUI]   ERROR in _toggle_time_slider: {ex}")
                traceback.print_exc()
        else:
            self.time_slider_frame.pack_forget()

    def _toggle_resolution(self):
        """Toggle between low-res ``_ccg`` and high-res ``_ccg_highres``."""
        if not (hasattr(self.cd, '_ccg_highres') and self.cd._ccg_highres):
            return
        # Resolution change invalidates scale caches (jitter cache keyed by res)
        self._pair_scale_cache.clear()
        self._session_scale_cache = None
        self._highres_mode = not self._highres_mode
        mode_label = 'highres' if self._highres_mode else 'lowres'
        if hasattr(self, '_res_btn_text'):
            self._res_btn_text.set(f"Res: {mode_label}")
        nd_key = self.key.nd()
        if self._highres_mode:
            self.ccg_data = self.cd._ccg_highres.get(nd_key)
        else:
            self.ccg_data = self.cd._ccg.get(nd_key)
        self._clear_all_png_cache()
        self._build_sig_chips()
        self.update_plot()

    def _make_collapsible_panel(self, parent, title, expanded=True):
        """
        Create a foldable panel section. Returns (inner_frame, fold_var).
        Caller stores fold_var as self._<name>_fold_var and populates inner_frame.
        """
        outer = ttk.Frame(parent)
        outer.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 0))
        hdr = ttk.Frame(outer)
        hdr.pack(fill=tk.X)
        fold_var = tk.BooleanVar(value=expanded)
        inner = ttk.Frame(outer, padding=4)
        if expanded:
            inner.pack(fill=tk.X)
        ttk.Checkbutton(hdr, text=title, variable=fold_var,
                        command=lambda: self._toggle_fold(
                            fold_var, inner, title, f'\u25b8 {title}', hdr)).pack(side=tk.LEFT)
        return inner, fold_var

    def _toggle_fold(self, var, inner_frame, text_open, text_closed, hdr_frame):
        """Show/hide the inner frame of a foldable section."""
        cb = hdr_frame.winfo_children()[0]  # the Checkbutton
        if var.get():
            inner_frame.pack(fill=tk.X)
            cb.config(text=text_open)
        else:
            inner_frame.pack_forget()
            cb.config(text=text_closed)

    _STYLE_VARS = {
        'ccg': ('_line_ccg_var', '_ccg_show_var'),
        'baseline': ('_line_baseline_var', '_baseline_show_var'),
        'ref': ('_line_ref_var', '_acg_ref_var'),
        'tgt': ('_line_tgt_var', '_acg_tgt_var'),
        'peak_wf': ('_line_peak_wf_var', '_peak_wf_var'),
    }

    def _get_style_vars(self, item):
        la, sa = self._STYLE_VARS[item]
        return getattr(self, la), getattr(self, sa)

    def _cycle_style(self, item):
        """Tri-state cycle: ■ solid → □ outline → X hidden → ■ solid."""
        line_var, show_var = self._get_style_vars(item)
        if show_var.get() and not line_var.get():   # solid → outline
            line_var.set(True)
        elif show_var.get() and line_var.get():      # outline → hidden
            show_var.set(False); line_var.set(False)
        else:                                         # hidden → solid
            show_var.set(True); line_var.set(False)
        self._update_style_btns(); self._clear_all_png_cache(); self.update_plot()

    def _cycle_style_acg(self, item):
        """Tri-state cycle: X hidden → □ outline → ■ solid → X hidden."""
        line_var, show_var = self._get_style_vars(item)
        if not show_var.get():                       # hidden → outline
            show_var.set(True); line_var.set(True)
        elif line_var.get():                          # outline → solid
            line_var.set(False)
        else:                                         # solid → hidden
            show_var.set(False); line_var.set(False)
        self._update_style_btns(); self._clear_all_png_cache(); self.update_plot()

    def _update_style_btns(self):
        """Refresh tri-state button labels: ■ name / □ name / X name."""
        for item, btn_attr, name in [
            ('ccg', '_ccg_style_btn', 'CCG'),
            ('baseline', '_baseline_style_btn', 'baseline'),
            ('ref', '_ref_style_btn', 'ref'),
            ('tgt', '_tgt_style_btn', 'tgt'),
            ('peak_wf', '_peak_wf_style_btn', 'ref peak'),
        ]:
            btn = getattr(self, btn_attr, None)
            if not btn:
                continue
            line_var, show_var = self._get_style_vars(item)
            if not show_var.get():
                btn.config(text=f"X {name}")
            elif line_var.get():
                btn.config(text=f"□ {name}")
            else:
                btn.config(text=f"■ {name}")

    def _toggle_plot_style(self):
        """Toggle all visible histogram items between filled and outline (Ctrl+L)."""
        visible_lines = [self._get_style_vars(item)[0]
                         for item in ('ccg', 'baseline', 'ref', 'tgt', 'peak_wf')
                         if self._get_style_vars(item)[1].get()]
        any_line = any(v.get() for v in visible_lines)
        for v in visible_lines:
            v.set(not any_line)
        self._update_style_btns(); self._clear_all_png_cache(); self.update_plot()

    def _ccg_context_menu(self, event):
        """Right-click context menu on the CCG plot canvas."""
        menu = tk.Menu(self.root, tearoff=0)
        # Export actions
        menu.add_command(label="Export view as PNG…",
                         command=lambda: self._export_current_view('png'))
        menu.add_command(label="Export view as PDF…",
                         command=lambda: self._export_current_view('pdf'))
        menu.add_separator()
        menu.add_command(label="View CCG values",
                         command=lambda: self._view_values('ccg'))
        menu.add_command(label="View ref ACG values",
                         command=lambda: self._view_values('acg_ref'))
        menu.add_command(label="View tgt ACG values",
                         command=lambda: self._view_values('acg_tgt'))
        menu.add_command(label="View baseline values",
                         command=lambda: self._view_values('baseline'))
        menu.add_command(label="View p-values",
                         command=lambda: self._view_values('pval'))
        menu.tk_popup(event.x_root, event.y_root)

    def _export_current_view(self, fmt: str):
        """Export the currently displayed CCG view (including stacked/sbs) to PNG/PDF."""
        fmt = (fmt or '').lower().strip()
        if fmt not in ('png', 'pdf'):
            return
        if self.ccg_data is None:
            messagebox.showwarning("Export", "No plot to export.")
            return
        selected_pairs = self._selected_pairs_from_lists()

        def _strip_any_session_pair(p):
            if p is None:
                return None
            if getattr(self, '_session_any_mode', False) and len(p) == 3:
                nk = self._nd_key_for_session_str(str(p[0]))
                if nk is not None:
                    ckey = self._type_key_for_nd(nk)
                    if ckey is not None:
                        self._bind_context_to_type_key(ckey)
                return int(p[1]), int(p[2])
            return int(p[0]), int(p[1])

        if getattr(self, '_session_any_mode', False):
            selected_pairs = [_strip_any_session_pair(p) for p in selected_pairs]

        # If user didn't explicitly select rows, default to current pair for "Export current".
        # (Also drives preview in the dialog.)
        preview_pair = selected_pairs[0] if selected_pairs else self._selected_pair_from_lists()
        preview_pair = _strip_any_session_pair(preview_pair)
        if preview_pair is None:
            inds = self._current_inds()
            if inds is not None:
                preview_pair = (int(inds[0]), int(inds[1]))
        opt = self._export_options_dialog(fmt=fmt, preview_pair=preview_pair, selected_pairs=selected_pairs)
        if opt is None:
            return
        # Multi-export actions should go straight to a folder picker (no save-as).
        if opt.get('_action') in ('all', 'bookmarked', 'groups', 'all_groups', 'all_sessions_selected'):
            self._export_pairs_from_opt(fmt=fmt, opt=opt)
            return
        # Suggest a filename from session/type/shank/pair/segment
        inds = self._current_inds()
        if inds is not None:
            ref, tgt = int(inds[0]), int(inds[1])
        else:
            ref = tgt = None
        seg = self.current_segment
        if seg == self.n_segments:
            seg_tag = "All"
        elif self._is_custom_segment(seg):
            seg_tag = "custom"
        else:
            seg_tag = f"seg{seg}"
        sess = str(getattr(getattr(self.key, 'nd', lambda: self.key)(), 'session', None) or getattr(self.key, 'session', 'sess'))
        exc = getattr(self.key, 'excitability', None)
        ct = getattr(self.key, 'conn_type', None)
        if isinstance(ct, (tuple, list)) and len(ct) >= 2:
            _map = {'pyr': 'PYR', 'inter': 'INT', 'int': 'INT'}
            a = _map.get(str(ct[0]).lower(), str(ct[0]).upper())
            b = _map.get(str(ct[1]).lower(), str(ct[1]).upper())
            ct_str = f"{a}-{b}"
        else:
            ct_str = str(ct) if ct is not None else "any"
        type_str = f"{exc}_{ct_str}" if exc is not None else ct_str
        sh = ''
        shank_ids = getattr(self.neurons, 'shank_ids', None)
        if shank_ids is not None and ref is not None and tgt is not None:
            try:
                sh = f"_sh{int(shank_ids[ref])}-{int(shank_ids[tgt])}"
            except Exception:
                sh = ''
        base = (f"{sess}_{type_str}{sh}_ccg_{ref}_{tgt}_{seg_tag}"
                if ref is not None else f"{sess}_{type_str}_ccg_{seg_tag}")
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title=f"Export as {fmt.upper()}",
            defaultextension=f".{fmt}",
            initialfile=f"{base}.{fmt}",
            filetypes=[(fmt.upper(), f"*.{fmt}"), ("All files", "*.*")],
        )

        if not path:
            return
        try:
            self._export_one_view_to_path(path=path, fmt=fmt, opt=opt)
        except Exception as exc:
            messagebox.showerror("Export failed", f"Could not export:\n\n{exc}")

    def _export_one_view_to_path(self, path: str, fmt: str, opt: dict):
        """Export current view to a specific file path using overrides."""
        old = getattr(self, '_export_overrides', None)
        self._export_overrides = opt
        try:
            self.update_plot()
            self.canvas.draw()
            self.fig.savefig(path, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
        finally:
            self._export_overrides = old
            self.update_plot()

    def _export_pairs_with_handles(self, fmt: str, opt: dict,
                                     items: list[tuple], folder: str) -> None:
        """Core export loop: render each (tk_, ptr, ref, tgt) into *folder*.

        *items* is a list of 4-tuples (tk_, ptr, ref, tgt) where tk_/ptr are
        the exact Key/pointer objects from cd.data — no string lookup needed.
        """
        export_segs = (opt or {}).get('export_segments', None)
        if not export_segs:
            export_segs = ["Current"]
        # normalize
        export_segs = [str(s) for s in export_segs if str(s).strip()] or ["Current"]
        subfolder_by = list((opt or {}).get('subfolder_by') or [])

        _ct_map = {'pyr': 'PYR', 'inter': 'INT', 'int': 'INT'}

        old_state = {
            'key': self.key,
            'ccg_pointer': self.ccg_pointer,
            'ccg_data': self.ccg_data,
            'neurons': self.neurons,
            'n_segments': getattr(self, 'n_segments', 0),
            'segment_names': getattr(self, 'segment_names', []),
            'current_pair_idx': int(getattr(self, 'current_pair_idx', 0)),
            'current_segment': int(getattr(self, 'current_segment', 0)),
        }

        n_ok = 0
        n_fail = 0
        fail_msgs = []
        try:
            for tk_, ptr, ref, tgt in items:
                # Point UI at this key/pointer directly — no lookup required.
                self.key = tk_
                self.ccg_pointer = ptr
                nd_key = tk_.nd()
                sess = str(getattr(tk_, 'session', getattr(nd_key, 'session', '')))
                self.neurons = (self.cd.nd.data.get(nd_key)
                                if getattr(self.cd, 'nd', None) is not None else None)
                self.n_segments = ptr.n_segments
                self.segment_names = list(ptr.edge_times['label'].values)
                # Resolve segment indices for this pointer.
                # - Current: use UI's current_segment (clamped to this ptr)
                # - All: use ptr.n_segments sentinel
                # - Label: map label->index for this ptr; if missing, skip that segment
                seg_indices: list[int] = []
                for export_seg in export_segs:
                    if export_seg == 'All':
                        seg_indices.append(int(ptr.n_segments))
                        continue
                    if export_seg == 'Current':
                        seg_idx = int(old_state.get('current_segment', 0))
                        if seg_idx < 0:
                            seg_idx = 0
                        if seg_idx > int(ptr.n_segments):
                            seg_idx = int(ptr.n_segments)
                        seg_indices.append(int(seg_idx))
                        continue
                    try:
                        seg_indices.append(int(list(ptr.edge_times['label'].values).index(export_seg)))
                    except Exception:
                        n_fail += 1
                        fail_msgs.append(f"({ref},{tgt}) [{sess}] missing segment '{export_seg}'")
                        continue
                # de-dupe while preserving order
                _seen_seg = set()
                seg_indices = [s for s in seg_indices if not (s in _seen_seg or _seen_seg.add(s))]
                if not seg_indices:
                    continue
                if (getattr(self, '_highres_mode', False)
                        and hasattr(self.cd, '_ccg_highres')
                        and self.cd._ccg_highres.get(nd_key) is not None):
                    self.ccg_data = self.cd._ccg_highres.get(nd_key)
                else:
                    self.ccg_data = (self.cd._ccg.get(nd_key)
                                     if getattr(self.cd, '_ccg', None) else self.ccg_data)
                try:
                    self.current_pair_idx = self.get_pair_index((ref, tgt))
                except Exception:
                    self.current_pair_idx = 0

                exc = getattr(tk_, 'excitability', None)
                ct = getattr(tk_, 'conn_type', None)
                if isinstance(ct, (tuple, list)) and len(ct) >= 2:
                    a = _ct_map.get(str(ct[0]).lower(), str(ct[0]).upper())
                    b = _ct_map.get(str(ct[1]).lower(), str(ct[1]).upper())
                    ct_str = f"{a}-{b}"
                else:
                    ct_str = str(ct) if ct is not None else "any"
                type_str = f"{exc}_{ct_str}" if exc is not None else ct_str
                sh = ''
                shank_ids = getattr(self.neurons, 'shank_ids', None)
                if shank_ids is not None:
                    try:
                        sh = f"_sh{int(shank_ids[ref])}-{int(shank_ids[tgt])}"
                    except Exception:
                        pass

                def _animal_from_session(s: str) -> str:
                    try:
                        return str(s).split('_')[0]
                    except Exception:
                        return str(s)

                def _ei_folder(x) -> str:
                    s = str(x or '').strip()
                    if not s:
                        return "EI"
                    sl = s.lower()
                    if sl.startswith('e'):
                        return "E"
                    if sl.startswith('i'):
                        return "I"
                    # common: 'exc'/'inh'
                    if 'exc' in sl:
                        return "E"
                    if 'inh' in sl:
                        return "I"
                    return s

                def _conn_type_folder(ct_) -> str:
                    if isinstance(ct_, (tuple, list)) and len(ct_) >= 2:
                        a = _ct_map.get(str(ct_[0]).lower(), str(ct_[0]).upper())
                        b = _ct_map.get(str(ct_[1]).lower(), str(ct_[1]).upper())
                        return f"{a}-{b}"
                    return str(ct_ or "any")

                # Build subfolder path parts in the chosen order
                parts = []
                for k in subfolder_by:
                    if k == "conn type":
                        parts.append(_conn_type_folder(ct))
                    elif k == "excitatory/inhibitory":
                        parts.append(_ei_folder(exc))
                    elif k == "session":
                        parts.append(sess)
                    elif k == "animal":
                        parts.append(_animal_from_session(sess))

                base_dir = os.path.join(folder, *parts) if parts else folder
                try:
                    os.makedirs(base_dir, exist_ok=True)
                except Exception:
                    base_dir = folder

                # Export one file per selected segment
                for seg_idx in seg_indices:
                    self.current_segment = int(seg_idx)
                    seg = int(self.current_segment)
                    if seg == int(ptr.n_segments):
                        seg_tag = "All"
                    elif self._is_custom_segment(seg):
                        seg_tag = "custom"
                    else:
                        seg_tag = f"seg{seg}"
                    fname = f"{sess}_{type_str}{sh}_ccg_{ref}_{tgt}_{seg_tag}.{fmt}"
                    out_path = os.path.join(base_dir, fname)
                    try:
                        self._export_one_view_to_path(path=out_path, fmt=fmt, opt=opt)
                        n_ok += 1
                    except Exception as ex:
                        n_fail += 1
                        fail_msgs.append(f"({ref},{tgt}) seg={seg_tag}: {ex}")
        finally:
            self.key = old_state['key']
            self.ccg_pointer = old_state['ccg_pointer']
            self.ccg_data = old_state['ccg_data']
            self.neurons = old_state['neurons']
            self.n_segments = old_state['n_segments']
            self.segment_names = old_state['segment_names']
            self.current_pair_idx = old_state['current_pair_idx']
            self.current_segment = old_state.get('current_segment', self.current_segment)
            try:
                self.update_plot()
            except Exception:
                pass

        if n_fail == 0:
            messagebox.showinfo("Export", f"Exported {n_ok} file(s) to:\n\n{folder}")
        else:
            msg = f"Exported {n_ok} file(s) to:\n\n{folder}\n\nFailed: {n_fail}"
            if fail_msgs:
                msg += "\n\n" + "\n".join(fail_msgs[:12])
                if len(fail_msgs) > 12:
                    msg += f"\n… ({len(fail_msgs) - 12} more)"
            messagebox.showwarning("Export", msg)

    def _export_all_selected_pairs(self, fmt: str, opt: dict):
        """Export pairs listed in opt['_selected_pairs'] (current-session subset)."""
        pairs_in = list(opt.get('_selected_pairs') or [])
        if not pairs_in:
            messagebox.showinfo("Export", "No pairs selected.")
            return
        folder = filedialog.askdirectory(
            parent=self.root, title=f"Export {len(pairs_in)} views to folder")
        if not folder:
            return

        # Build (tk_, ptr, ref, tgt) from current session only
        items = []
        for it in pairs_in:
            pair = tuple(it['pair']) if isinstance(it, dict) else tuple(it)
            try:
                ref, tgt = int(pair[0]), int(pair[1])
            except Exception:
                continue
            items.append((self.key, self.ccg_pointer, ref, tgt))

        self._export_pairs_with_handles(fmt=fmt, opt=opt, items=items, folder=folder)

    def _collect_all_sessions_selected(self) -> list[tuple]:
        """Return (tk_, ptr, ref, tgt) for every selected pair in every session/type."""
        # Flush live selection into the current pointer before iterating.
        if self.ccg_pointer is not None:
            if getattr(self, '_session_any_mode', False):
                self._flush_any_selections_to_pointers()
            else:
                self.ccg_pointer.manually_selected_inds = (
                    np.array(sorted(self.selected_inds), dtype=int)
                    if getattr(self, 'selected_inds', None) else None
                )
        items = []
        for tk_, ptr in self.cd.data.items():
            sel = getattr(ptr, 'manually_selected_inds', None)
            if sel is None or len(sel) == 0:
                continue
            for pair in sel:
                try:
                    items.append((tk_, ptr, int(pair[0]), int(pair[1])))
                except Exception:
                    continue
        # Sort: session str → ref → tgt for stable output ordering
        items.sort(key=lambda x: (str(getattr(x[0], 'session', '')), x[2], x[3]))
        return items

    def _pair_handle_map(self) -> dict[tuple, list[tuple]]:
        """Build {(session_str, ref, tgt): [(tk_, ptr), ...]} from all cd.data entries.

        A pair can legitimately appear in multiple conn-type keys for the same
        session.  We keep all of them so the caller can pick the right one.
        """
        m: dict[tuple, list] = {}
        for tk_, ptr in self.cd.data.items():
            if ptr is None:
                continue
            sess = str(getattr(tk_, 'session', getattr(tk_.nd(), 'session', '')))
            inds = getattr(ptr, 'inds2', None)
            if inds is None:
                continue
            for pair in inds:
                try:
                    k = (sess, int(pair[0]), int(pair[1]))
                    m.setdefault(k, []).append((tk_, ptr))
                except Exception:
                    pass
        return m

    def _export_pairs_from_opt(self, fmt: str, opt: dict):
        """Resolve (tk_, ptr, ref, tgt) handles and export to a chosen folder."""
        action = opt.get('_action')

        if action == 'all_sessions_selected':
            # All selected pairs across every session/type — handles come straight
            # from cd.data, no lookup needed.
            items = self._collect_all_sessions_selected()

        elif action == 'all':
            # Pairs explicitly highlighted in the current-session listbox.
            raw = list(opt.get('_selected_pairs') or [])
            items = []
            for it in raw:
                pair = tuple(it['pair']) if isinstance(it, dict) else tuple(it)
                try:
                    items.append((self.key, self.ccg_pointer,
                                  int(pair[0]), int(pair[1])))
                except Exception:
                    pass

        elif action == 'bookmarked':
            # Bookmarks are per-session (current session only).
            items = []
            for p in sorted(getattr(self, '_bookmarked_pairs', set()) or set()):
                try:
                    items.append((self.key, self.ccg_pointer,
                                  int(p[0]), int(p[1])))
                except Exception:
                    pass

        elif action in ('groups', 'all_groups'):
            # Group data is stored as {session_str: [[ref,tgt], ...]}.
            # IMPORTANT: we must NOT “guess” a handle for (session, pair). Instead,
            # we scan each (tk_, ptr).inds2 and include it only if that pair is
            # explicitly in the chosen group(s) for that session.
            if action == 'all_groups':
                gnames = [g for g in (self._groups or {}) if g and not str(g).startswith('__')]
            else:
                gnames = list(opt.get('_selected_groups') or [])
                if not gnames:
                    messagebox.showinfo("Export", "No groups selected.")
                    return

            # Build desired pairs per session from group definitions
            want_by_sess: dict[str, set[tuple[int, int]]] = {}
            for g in gnames:
                try:
                    sd = self._groups.get(g, {})
                    pairs_by_sess = sd if isinstance(sd, dict) else {
                        self._current_session_str(): list(sd)}
                    for sess, pp in pairs_by_sess.items():
                        ss = str(sess)
                        s = want_by_sess.setdefault(ss, set())
                        for p in pp:
                            try:
                                s.add((int(p[0]), int(p[1])))
                            except Exception:
                                continue
                except Exception:
                    continue

            raw_items: list[tuple] = []
            seen: set[tuple] = set()
            found_by_sess: dict[str, set[tuple[int, int]]] = {}
            for tk_, ptr in self.cd.data.items():
                if ptr is None:
                    continue
                try:
                    sess = str(getattr(tk_, 'session', tk_.nd().session))
                except Exception:
                    sess = ''
                want = want_by_sess.get(sess)
                if not want:
                    continue
                inds = getattr(ptr, 'inds2', None)
                if inds is None:
                    continue
                for pair in inds:
                    try:
                        ref, tgt = int(pair[0]), int(pair[1])
                    except Exception:
                        continue
                    if (ref, tgt) not in want:
                        continue
                    k = (id(tk_), ref, tgt)
                    if k in seen:
                        continue
                    seen.add(k)
                    raw_items.append((tk_, ptr, ref, tgt))
                    found_by_sess.setdefault(sess, set()).add((ref, tgt))

            items = sorted(
                raw_items,
                key=lambda x: (str(getattr(x[0], 'session', '')), x[2], x[3])
            )
            # Warn about any group pairs that are not present in loaded data for that session.
            missing = []
            for sess, want in want_by_sess.items():
                found = found_by_sess.get(sess, set())
                for p in sorted(want):
                    if p not in found:
                        missing.append((sess, p))
            if missing:
                preview = "\n".join([f"{s}: {p}" for s, p in missing[:15]])
                more = "" if len(missing) <= 15 else f"\n… ({len(missing) - 15} more)"
                messagebox.showwarning(
                    "Export",
                    "Some pairs in the selected group(s) were not found in the loaded data and will be skipped.\n\n"
                    + preview + more,
                    parent=self.root
                )
        else:
            items = []

        if not items:
            messagebox.showinfo("Export", "No pairs to export.")
            return

        folder = filedialog.askdirectory(
            parent=self.root, title=f"Export {len(items)} view(s) to folder")
        if not folder:
            return

        self._export_pairs_with_handles(fmt=fmt, opt=opt, items=items, folder=folder)

    def _export_options_dialog(self, fmt: str, preview_pair=None, selected_pairs=None):
        """Return export overrides dict or None if cancelled.

        When multiple pairs are selected in the lists, this dialog also offers
        an 'Export all selected pairs…' option. Preview (if enabled) always
        shows only the first selected pair.
        """
        win = tk.Toplevel(self.root)
        win.title("Export options")
        win.transient(self.root)
        win.grab_set()
        win.resizable(True, True)

        # ------------------------------
        # Group selection (export scope)
        # ------------------------------
        # Available groups exclude internal/special/private entries.
        all_groups = sorted(
            g for g in (self._groups.keys() if getattr(self, '_groups', None) else [])
            if g and not str(g).startswith('__')
        )
        avail_groups = [g for g in all_groups if not str(g).startswith(_SPECIAL_PREFIX)]
        selected_groups: list[str] = []

        # Export defaults persisted in ui_state.json via self._settings
        _exp_def = (self._settings.get('export_defaults', {}) if isinstance(getattr(self, '_settings', None), dict) else {}) or {}
        ccg_var = tk.StringVar(value=str(_exp_def.get('ccg_color', "steelblue")))
        base_var = tk.StringVar(value=str(_exp_def.get('baseline_color', "orange")))
        minfs_var = tk.StringVar(value=str(_exp_def.get('min_text_size', "8")))
        show_prev_var = tk.BooleanVar(value=False)
        ccg_a_var = tk.StringVar(value=str(_exp_def.get('ccg_alpha', "0.5")))
        base_a_var = tk.StringVar(value=str(_exp_def.get('baseline_alpha', "0.3")))
        show_legend_var = tk.BooleanVar(value=bool(_exp_def.get('show_legend', True)))
        xticks_var = tk.StringVar(value=str(_exp_def.get('xticks_raw', "")))
        mirror_ticks_var = tk.BooleanVar(value=bool(_exp_def.get('mirror_xticks', True)))
        # Segment export selection (multi): union of segment labels across loaded types for this session.
        # Stored as a list of strings in export_defaults['export_segments'].
        seg_export_default = list(_exp_def.get('export_segments') or []) if isinstance(_exp_def, dict) else []
        if not seg_export_default:
            seg_export_default = ["Current"]
        try:
            nd_key = self.key.nd() if getattr(self, 'key', None) is not None else None
            type_keys = self._available_type_keys(nd_key) if nd_key is not None else []
            seg_union: set[str] = set()
            for tk_ in type_keys:
                ptr = self.cd.data.get(tk_)
                try:
                    labels = list(ptr.edge_times['label'].values) if ptr is not None else []
                except Exception:
                    labels = []
                for lab in labels:
                    if lab is None:
                        continue
                    s = str(lab).strip()
                    if s:
                        seg_union.add(s)
            seg_export_choices = ["Current", "All"] + sorted(seg_union)
        except Exception:
            seg_export_choices = ["Current", "All"]

        # Subfolder hierarchy selection (multi + order)
        subfolder_default = list(_exp_def.get('subfolder_by') or []) if isinstance(_exp_def, dict) else []
        _subfolder_choices = ["conn type", "excitatory/inhibitory", "session", "animal"]
        # sanitize defaults
        subfolder_default = [x for x in subfolder_default if x in _subfolder_choices]

        # Scrollable body (export options can grow tall)
        outer = ttk.Frame(win)
        outer.grid(row=0, column=0, sticky="nsew")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        frm = ttk.Frame(canvas, padding=10)
        win_id = canvas.create_window((0, 0), window=frm, anchor='nw')

        def _on_frame_configure(_event=None):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass

        def _on_canvas_configure(event):
            # Keep inner frame width matched to canvas width
            try:
                canvas.itemconfigure(win_id, width=event.width)
            except Exception:
                pass

        frm.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        frm.columnconfigure(1, weight=1)

        # Group selection UI (above plot configs)
        grp_frame = ttk.LabelFrame(frm, text="Groups")
        grp_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        grp_frame.columnconfigure(0, weight=1)
        grp_frame.columnconfigure(2, weight=1)

        ttk.Label(grp_frame, text="Available:").grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
        ttk.Label(grp_frame, text="Selected:").grid(row=0, column=2, sticky="w", padx=6, pady=(6, 2))

        avail_lb = tk.Listbox(grp_frame, height=6, exportselection=False)
        sel_lb = tk.Listbox(grp_frame, height=6, exportselection=False)
        for g in avail_groups:
            avail_lb.insert(tk.END, g)
        avail_lb.grid(row=1, column=0, sticky="ew", padx=6, pady=4)
        sel_lb.grid(row=1, column=2, sticky="ew", padx=6, pady=4)

        mid = ttk.Frame(grp_frame)
        mid.grid(row=1, column=1, sticky="ns", padx=6)

        selected_tags_var = tk.StringVar(value="")
        tags_lbl = ttk.Label(grp_frame, textvariable=selected_tags_var, foreground="#555555", wraplength=560)
        tags_lbl.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(2, 6))

        def _refresh_group_tags():
            # "recorded below selection list": show selected group names (tags)
            if selected_groups:
                selected_tags_var.set("Selected group tags: " + ", ".join(selected_groups))
            else:
                selected_tags_var.set("")

        def _add_groups():
            sel = list(avail_lb.curselection())
            if not sel:
                return
            # Add in displayed order
            names = [avail_lb.get(i) for i in sel]
            # Remove from bottom-up to keep indices valid
            for i in sorted(sel, reverse=True):
                avail_lb.delete(i)
            for g in names:
                if g not in selected_groups:
                    selected_groups.append(g)
                    sel_lb.insert(tk.END, g)
            _refresh_group_tags()

        def _remove_groups():
            sel = list(sel_lb.curselection())
            if not sel:
                return
            names = [sel_lb.get(i) for i in sel]
            for i in sorted(sel, reverse=True):
                sel_lb.delete(i)
            for g in names:
                try:
                    selected_groups.remove(g)
                except ValueError:
                    pass
            # Return to available list (keep sorted)
            cur = list(avail_lb.get(0, tk.END))
            cur.extend(names)
            cur = sorted(set(cur))
            avail_lb.delete(0, tk.END)
            for g in cur:
                avail_lb.insert(tk.END, g)
            _refresh_group_tags()

        ttk.Button(mid, text="Add →", command=_add_groups).pack(pady=(10, 6))
        ttk.Button(mid, text="← Remove", command=_remove_groups).pack()
        avail_lb.bind("<Double-Button-1>", lambda e: _add_groups())
        sel_lb.bind("<Double-Button-1>", lambda e: _remove_groups())

        # Plot config fields (start after group section)
        row0 = 1
        ttk.Label(frm, text="CCG color (name or #hex):").grid(row=row0 + 0, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=ccg_var, width=22).grid(row=row0 + 0, column=1, sticky="ew", padx=(8, 0), pady=4)

        ttk.Label(frm, text="Baseline color (name or #hex):").grid(row=row0 + 1, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=base_var, width=22).grid(row=row0 + 1, column=1, sticky="ew", padx=(8, 0), pady=4)

        ttk.Label(frm, text="Min text size (pt):").grid(row=row0 + 2, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=minfs_var, width=10).grid(row=row0 + 2, column=1, sticky="w", padx=(8, 0), pady=4)

        ttk.Label(frm, text="CCG alpha (0–1):").grid(row=row0 + 3, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=ccg_a_var, width=10).grid(row=row0 + 3, column=1, sticky="w", padx=(8, 0), pady=4)

        ttk.Label(frm, text="Baseline alpha (0–1):").grid(row=row0 + 4, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=base_a_var, width=10).grid(row=row0 + 4, column=1, sticky="w", padx=(8, 0), pady=4)

        ttk.Checkbutton(frm, text="Show legend", variable=show_legend_var).grid(
            row=row0 + 5, column=0, sticky="w", pady=(6, 0))

        ttk.Label(frm, text="X ticks (ms, comma-separated):").grid(row=row0 + 6, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=xticks_var, width=28).grid(row=row0 + 6, column=1, sticky="ew", padx=(8, 0), pady=4)
        ttk.Checkbutton(frm, text="Mirror to negative ticks", variable=mirror_ticks_var).grid(
            row=row0 + 7, column=0, sticky="w", pady=(0, 4))

        # Segments selection UI (same pattern as Groups)
        seg_frame = ttk.LabelFrame(frm, text="Segments to export")
        seg_frame.grid(row=row0 + 8, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        seg_frame.columnconfigure(0, weight=1)
        seg_frame.columnconfigure(2, weight=1)
        ttk.Label(seg_frame, text="Available:").grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
        ttk.Label(seg_frame, text="Selected:").grid(row=0, column=2, sticky="w", padx=6, pady=(6, 2))

        seg_avail_lb = tk.Listbox(seg_frame, height=6, exportselection=False)
        seg_sel_lb = tk.Listbox(seg_frame, height=6, exportselection=False)
        for s in seg_export_choices:
            seg_avail_lb.insert(tk.END, s)
        # preload selected segments
        selected_segments: list[str] = []
        for s in seg_export_default:
            if s in seg_export_choices and s not in selected_segments:
                selected_segments.append(s)
                seg_sel_lb.insert(tk.END, s)
        # remove preselected from available
        if selected_segments:
            cur = [seg_avail_lb.get(i) for i in range(seg_avail_lb.size())]
            seg_avail_lb.delete(0, tk.END)
            for s in cur:
                if s not in selected_segments:
                    seg_avail_lb.insert(tk.END, s)

        seg_avail_lb.grid(row=1, column=0, sticky="ew", padx=6, pady=4)
        seg_sel_lb.grid(row=1, column=2, sticky="ew", padx=6, pady=4)

        seg_mid = ttk.Frame(seg_frame)
        seg_mid.grid(row=1, column=1, sticky="ns", padx=6)

        def _add_segments():
            sel = list(seg_avail_lb.curselection())
            if not sel:
                return
            names = [seg_avail_lb.get(i) for i in sel]
            for i in sorted(sel, reverse=True):
                seg_avail_lb.delete(i)
            for s in names:
                if s not in selected_segments:
                    selected_segments.append(s)
                    seg_sel_lb.insert(tk.END, s)

        def _remove_segments():
            sel = list(seg_sel_lb.curselection())
            if not sel:
                return
            names = [seg_sel_lb.get(i) for i in sel]
            for i in sorted(sel, reverse=True):
                seg_sel_lb.delete(i)
            for s in names:
                try:
                    selected_segments.remove(s)
                except ValueError:
                    pass
            cur = list(seg_avail_lb.get(0, tk.END))
            cur.extend(names)
            cur = sorted(set(cur), key=lambda x: (0 if x == "Current" else 1 if x == "All" else 2, x))
            seg_avail_lb.delete(0, tk.END)
            for s in cur:
                if s not in selected_segments:
                    seg_avail_lb.insert(tk.END, s)

        ttk.Button(seg_mid, text="Add →", command=_add_segments).pack(pady=(10, 6))
        ttk.Button(seg_mid, text="← Remove", command=_remove_segments).pack()
        seg_avail_lb.bind("<Double-Button-1>", lambda e: _add_segments())
        seg_sel_lb.bind("<Double-Button-1>", lambda e: _remove_segments())

        # Subfolder hierarchy UI (dual list; selected list is reorderable by drag)
        sf_frame = ttk.LabelFrame(frm, text="Subfolder by (optional)")
        sf_frame.grid(row=row0 + 9, column=0, columnspan=2, sticky="ew", pady=(6, 6))
        sf_frame.columnconfigure(0, weight=1)
        sf_frame.columnconfigure(2, weight=1)
        ttk.Label(sf_frame, text="Available:").grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
        ttk.Label(sf_frame, text="Selected (drag to reorder):").grid(row=0, column=2, sticky="w", padx=6, pady=(6, 2))

        sf_avail_lb = tk.Listbox(sf_frame, height=4, exportselection=False)
        sf_sel_lb = tk.Listbox(sf_frame, height=4, exportselection=False)
        selected_subfolders: list[str] = list(subfolder_default)
        for s in _subfolder_choices:
            if s not in selected_subfolders:
                sf_avail_lb.insert(tk.END, s)
        for s in selected_subfolders:
            sf_sel_lb.insert(tk.END, s)

        sf_avail_lb.grid(row=1, column=0, sticky="ew", padx=6, pady=4)
        sf_sel_lb.grid(row=1, column=2, sticky="ew", padx=6, pady=4)

        sf_mid = ttk.Frame(sf_frame)
        sf_mid.grid(row=1, column=1, sticky="ns", padx=6)

        def _add_subfolders():
            sel = list(sf_avail_lb.curselection())
            if not sel:
                return
            names = [sf_avail_lb.get(i) for i in sel]
            for i in sorted(sel, reverse=True):
                sf_avail_lb.delete(i)
            for s in names:
                if s not in selected_subfolders:
                    selected_subfolders.append(s)
                    sf_sel_lb.insert(tk.END, s)

        def _remove_subfolders():
            sel = list(sf_sel_lb.curselection())
            if not sel:
                return
            names = [sf_sel_lb.get(i) for i in sel]
            for i in sorted(sel, reverse=True):
                sf_sel_lb.delete(i)
            for s in names:
                try:
                    selected_subfolders.remove(s)
                except ValueError:
                    pass
            cur = list(sf_avail_lb.get(0, tk.END))
            cur.extend(names)
            cur = sorted(set(cur), key=lambda x: _subfolder_choices.index(x) if x in _subfolder_choices else 999)
            sf_avail_lb.delete(0, tk.END)
            for s in cur:
                if s not in selected_subfolders:
                    sf_avail_lb.insert(tk.END, s)

        ttk.Button(sf_mid, text="Add →", command=_add_subfolders).pack(pady=(10, 6))
        ttk.Button(sf_mid, text="← Remove", command=_remove_subfolders).pack()
        sf_avail_lb.bind("<Double-Button-1>", lambda e: _add_subfolders())
        sf_sel_lb.bind("<Double-Button-1>", lambda e: _remove_subfolders())

        # Drag-reorder for selected subfolder listbox
        _drag_state = {'i': None}
        def _sf_on_press(e):
            try:
                _drag_state['i'] = sf_sel_lb.nearest(e.y)
            except Exception:
                _drag_state['i'] = None
        def _sf_on_drag(e):
            try:
                i0 = _drag_state.get('i')
                if i0 is None:
                    return
                i1 = sf_sel_lb.nearest(e.y)
                if i1 == i0:
                    return
                item = sf_sel_lb.get(i0)
                sf_sel_lb.delete(i0)
                sf_sel_lb.insert(i1, item)
                # keep backing list in sync
                try:
                    selected_subfolders.pop(i0)
                    selected_subfolders.insert(i1, item)
                except Exception:
                    pass
                _drag_state['i'] = i1
                sf_sel_lb.selection_clear(0, tk.END)
                sf_sel_lb.selection_set(i1)
            except Exception:
                pass
        sf_sel_lb.bind("<Button-1>", _sf_on_press)
        sf_sel_lb.bind("<B1-Motion>", _sf_on_drag)

        out = {}
        action = {'mode': 'current'}  # 'current' | 'all'

        def _collect_opts() -> dict:
            o: dict = {}
            o['ccg_color'] = (ccg_var.get() or '').strip() or None
            o['baseline_color'] = (base_var.get() or '').strip() or None
            try:
                v = float(minfs_var.get())
                o['min_text_size'] = v if np.isfinite(v) and v > 0 else None
            except Exception:
                o['min_text_size'] = None
            try:
                v = float(ccg_a_var.get())
                o['ccg_alpha'] = v if np.isfinite(v) else None
            except Exception:
                o['ccg_alpha'] = None
            try:
                v = float(base_a_var.get())
                o['baseline_alpha'] = v if np.isfinite(v) else None
            except Exception:
                o['baseline_alpha'] = None
            o['show_legend'] = bool(show_legend_var.get())
            o['mirror_xticks'] = bool(mirror_ticks_var.get())
            raw = (xticks_var.get() or '').strip()
            if raw:
                vals = []
                for part in raw.replace(';', ',').split(','):
                    s = part.strip()
                    if not s:
                        continue
                    try:
                        vals.append(float(s))
                    except Exception:
                        continue
                o['xticks_ms'] = vals
            else:
                o['xticks_ms'] = None
            # Store raw for defaults (preserves user's formatting)
            o['_xticks_raw'] = raw
            o['export_segments'] = list(selected_segments) if selected_segments else ["Current"]
            o['subfolder_by'] = list(selected_subfolders)
            return o

        def _ok():
            out.update(_collect_opts())
            win.destroy()

        def _cancel():
            out.clear()
            win.destroy()

        def _save_export_defaults():
            """Persist current export settings as defaults for next time."""
            opts = _collect_opts()
            try:
                self._settings.setdefault('export_defaults', {})
                self._settings['export_defaults'] = {
                    'ccg_color': opts.get('ccg_color') or "steelblue",
                    'baseline_color': opts.get('baseline_color') or "orange",
                    'min_text_size': opts.get('min_text_size') if opts.get('min_text_size') is not None else "8",
                    'ccg_alpha': opts.get('ccg_alpha') if opts.get('ccg_alpha') is not None else "0.5",
                    'baseline_alpha': opts.get('baseline_alpha') if opts.get('baseline_alpha') is not None else "0.3",
                    'show_legend': bool(opts.get('show_legend', True)),
                    'mirror_xticks': bool(opts.get('mirror_xticks', True)),
                    'xticks_raw': opts.get('_xticks_raw', ''),
                    'export_segments': opts.get('export_segments', ['Current']),
                    'subfolder_by': opts.get('subfolder_by', []),
                }
                # Save UI state without touching selections
                self._save_all_state(selection_name=None, silent=True)
            except Exception:
                pass

        # Preview area (optional)
        prev_holder = ttk.Frame(frm)
        prev_holder.grid(row=row0 + 12, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        prev_holder.columnconfigure(0, weight=1)
        prev_holder.rowconfigure(0, weight=1)
        prev_label = ttk.Label(prev_holder, text="", anchor="center")
        prev_label.grid(row=0, column=0, sticky="nsew")
        prev_img = {'obj': None}

        def _render_preview():
            if not show_prev_var.get():
                prev_label.configure(text="")
                prev_label.configure(image="")
                prev_img['obj'] = None
                return
            if preview_pair is None:
                prev_label.configure(text="(no pair selected)")
                return
            try:
                # Build a PNG using the same rendering path (but without mutating UI state).
                # Preview uses the current segment/resolution mode and first selected pair.
                ref, tgt = int(preview_pair[0]), int(preview_pair[1])
                seg = int(self.current_segment)
                highres = bool(getattr(self, '_highres_mode', False))
                # Apply overrides temporarily so preview reflects export styling.
                tmp_over = {
                    'ccg_color': (ccg_var.get() or '').strip() or None,
                    'baseline_color': (base_var.get() or '').strip() or None,
                    'min_text_size': float(minfs_var.get()) if str(minfs_var.get()).strip() else None,
                    'ccg_alpha': float(ccg_a_var.get()) if str(ccg_a_var.get()).strip() else None,
                    'baseline_alpha': float(base_a_var.get()) if str(base_a_var.get()).strip() else None,
                    'show_legend': bool(show_legend_var.get()),
                    'mirror_xticks': bool(mirror_ticks_var.get()),
                    'xticks_ms': ([(float(x.strip())) for x in (xticks_var.get() or '').split(',')
                                   if x.strip()] if (xticks_var.get() or '').strip() else None),
                }
                old = getattr(self, '_export_overrides', None)
                self._export_overrides = tmp_over
                try:
                    png_path = self._render_png((ref, tgt), seg, highres=highres)
                finally:
                    self._export_overrides = old

                try:
                    from PIL import Image, ImageTk
                    im = Image.open(png_path)
                    max_w = 700
                    max_h = 450
                    w, h = im.size
                    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
                    if scale < 1.0:
                        im = im.resize((int(w * scale), int(h * scale)))
                    tk_im = ImageTk.PhotoImage(im)
                    prev_label.configure(image=tk_im, text="")
                    prev_img['obj'] = tk_im  # keep reference
                except Exception:
                    prev_label.configure(text=f"Preview ready: {os.path.basename(png_path)}")
            except Exception as ex:
                prev_label.configure(text=f"Preview failed: {ex}")

        ttk.Checkbutton(frm, text="Show preview", variable=show_prev_var,
                        command=_render_preview).grid(row=row0 + 11, column=0, sticky="w", pady=(8, 0))
        # Re-render preview when override fields change (only if enabled)
        for _v in (ccg_var, base_var, minfs_var, ccg_a_var, base_a_var, xticks_var):
            try:
                _v.trace_add('write', lambda *a: _render_preview())
            except Exception:
                pass
        try:
            show_legend_var.trace_add('write', lambda *a: _render_preview())
        except Exception:
            pass
        try:
            mirror_ticks_var.trace_add('write', lambda *a: _render_preview())
        except Exception:
            pass

        btns = ttk.Frame(frm)
        btns.grid(row=row0 + 13, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="Cancel", command=_cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btns, text="Save export settings", command=_save_export_defaults).pack(
            side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btns, text=f"Export current as {fmt.upper()}", command=_ok).pack(side=tk.RIGHT)

        # If multiple pairs explicitly selected in current list, also offer that
        if selected_pairs and len(selected_pairs) > 1:
            def _export_all():
                action['mode'] = 'all'
                _ok()
            ttk.Button(btns, text=f"Export all selected ({len(selected_pairs)})…",
                       command=_export_all).pack(side=tk.RIGHT, padx=(6, 6))

        # Export bookmarked pairs
        n_bm = len(getattr(self, '_bookmarked_pairs', set()) or set())
        if n_bm > 0:
            def _export_bookmarked():
                action['mode'] = 'bookmarked'
                _ok()
            ttk.Button(btns, text=f"Export bookmarked ({n_bm})…",
                       command=_export_bookmarked).pack(side=tk.RIGHT, padx=(6, 6))

        # Export pairs from groups — show buttons whenever any groups exist
        any_groups = bool(all_groups)  # all_groups includes special groups (filtered only __-prefix)
        if any_groups:
            def _export_groups():
                action['mode'] = 'groups'
                _ok()
            ttk.Button(btns, text="Export selected group(s)…",
                       command=_export_groups).pack(side=tk.RIGHT, padx=(6, 6))

        win.protocol("WM_DELETE_WINDOW", _cancel)
        win.bind("<Escape>", lambda e: _cancel())
        win.wait_window()

        if not out:
            return None
        out['_action'] = action.get('mode', 'current')
        out['_selected_pairs'] = list(selected_pairs) if selected_pairs else []
        out['_preview_pair'] = preview_pair
        out['_fmt'] = fmt
        out['_selected_groups'] = list(selected_groups)
        return out

    def _view_values(self, item):
        """Print values of a CCG data item to cell output."""
        if self.current_pair_idx >= len(self.all_inds):
            print("[ViewValues] No pair selected"); return
        inds = self.all_inds[self.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        segment = self.current_segment
        cd = self.ccg_data
        if cd is None:
            print("[ViewValues] No CCG data available"); return

        # If ACG deconvolution is active, inform the user that displayed values
        # are derived (deconvolved) and should be interpreted accordingly.
        try:
            dref = bool(getattr(self, "_acg_deconv_ref_var", None) and self._acg_deconv_ref_var.get())
            dtgt = bool(getattr(self, "_acg_deconv_tgt_var", None) and self._acg_deconv_tgt_var.get())
            if dref or dtgt:
                mode = ("ref+tgt" if (dref and dtgt) else "ref" if dref else "tgt")
                print(f"[ViewValues] NOTE: ACG deconvolution is ON ({mode}); values reflect the displayed (deconvolved) CCG.")
        except Exception:
            pass

        is_all = (segment == self.n_segments)
        is_custom = self._is_custom_segment(segment)

        def _get(arr, r, c):
            """Extract values from ccg-shaped array [seg, r, c, bins]."""
            if arr is None:
                return None
            if is_all:
                return np.sum(arr[:, r, c, :], axis=0)
            elif is_custom:
                ci = self._custom_seg_index(segment)
                cs = self._custom_segments[ci]
                src = cs.get('ccg_hi') if self._highres_mode and 'ccg_hi' in cs else cs.get('ccg')
                return src[0, r, c, :] if src is not None else None
            else:
                return arr[segment, r, c, :]

        items = {
            # These two are display-derived (norms/baseline + possible deconvolution)
            'ccg':     (None,        ref, tgt, f"CCG [{ref},{tgt}]"),
            'baseline':(None,        ref, tgt, f"Baseline [{ref},{tgt}]"),
            'pval':    (cd.pval,     ref, tgt, f"P-value [{ref},{tgt}]"),
            'acg_ref': (cd.ccg,      ref, ref, f"ACG ref [{ref}]"),
            'acg_tgt': (cd.ccg,      tgt, tgt, f"ACG tgt [{tgt}]"),
        }
        if item not in items:
            print(f"[ViewValues] Unknown item: {item}"); return
        arr, r, c, label = items[item]
        vals = None
        if item in ("ccg", "baseline"):
            try:
                resk = 'hi' if bool(self._highres_mode) else 'lo'
                tmp = self._display_pair_temp.get((ref, tgt, int(segment), resk))
                if tmp is None:
                    # Force a render to populate display temp cache, then re-read.
                    self._render_png((ref, tgt), segment, highres=self._highres_mode)
                    tmp = self._display_pair_temp.get((ref, tgt, int(segment), resk))
                if tmp is not None:
                    vals = tmp.get("ccg") if item == "ccg" else tmp.get("baseline_1d")
            except Exception:
                vals = None
        else:
            vals = _get(arr, r, c)

        if vals is not None:
            print(f"\n{label} seg={segment}:")
            print(f"  values: {vals}")
            print(f"  min={np.min(vals):.4f}  max={np.max(vals):.4f}  "
                  f"mean={np.mean(vals):.4f}  sum={np.sum(vals):.4f}")
        else:
            print(f"[ViewValues] No data for {item}")

    def _on_ctrl_e(self):
        var = self._panel_vars.get('Waveforms')
        if var:
            var.set(not var.get())
            self._toggle_panel('Waveforms')

    # ------------------------------------------------------------------
    # Significance helpers
    # ------------------------------------------------------------------

    def _is_significant(self, ref: int, tgt: int, seg: int) -> bool:
        """Return True if (ref, tgt) is significant in segment *seg*.

        When seg == self.n_segments ("All"), returns True if
        the pair is significant in ANY real segment.

        Significance is always derived from the **low-res** CCGData so that
        segment chips stay green when the user switches to high-res mode.
        Priority: jitter (segment-aware) → pval_corrected + active_alpha
        → stored significant array.
        """
        # Custom segments: use stored pval_corrected
        if self._is_custom_segment(seg):
            ci = self._custom_seg_index(seg)
            cs_list = getattr(self, '_custom_segments', [])
            if 0 <= ci < len(cs_list):
                cs = cs_list[ci]
                pc = cs.get('pval_corrected')
                if pc is not None:
                    conf = self.ccg_data.conf
                    lb = getattr(conf, 'min_lag_bin', None)
                    ub = getattr(conf, 'max_lag_bin', None)
                    if lb is not None and ub is not None:
                        try:
                            return bool(pc[0, ref, tgt, lb:ub].min()
                                        <= self.active_alpha)
                        except (IndexError, ValueError):
                            pass
            return False

        if seg == self.n_segments:
            return any(self._is_significant(ref, tgt, s)
                       for s in range(self.n_segments))

        j = self.cd._jitter.get(self.key) if hasattr(self.cd, '_jitter') else None
        if j is not None:
            inds = j.ccg_pointer.inds
            if inds is not None:
                if getattr(j.ccg_pointer, 'stored_by_segment', False):
                    mask = ((inds[:, 0] == seg) &
                            (inds[:, -2] == ref) & (inds[:, -1] == tgt))
                else:
                    mask = (inds[:, -2] == ref) & (inds[:, -1] == tgt)
                if mask.any():
                    return bool(j.j_sig[mask].any())
                return False

        # Always use low-res data for significance (high-res may lack pval arrays)
        cd = self.cd._ccg.get(self.key.nd()) if hasattr(self.cd, '_ccg') else None
        if cd is None:
            cd = self.ccg_data   # fallback
        if cd is not None and cd.pval_corrected is not None:
            conf = cd.conf
            lb = getattr(conf, 'min_lag_bin', None)
            ub = getattr(conf, 'max_lag_bin', None)
            if lb is not None and ub is not None:
                try:
                    return bool(
                        cd.pval_corrected[seg, ref, tgt, lb:ub].min()
                        <= self.active_alpha)
                except (IndexError, ValueError):
                    pass
        if cd is not None and cd.significant is not None:
            try:
                return bool(cd.significant[seg, ref, tgt])
            except (IndexError, ValueError):
                pass
        return False

    # ------------------------------------------------------------------
    # Alpha / normalization callbacks
    # ------------------------------------------------------------------

    def _on_alpha_change(self, val):
        self.active_alpha = round(float(val), 4)
        self._alpha_label.set(f"{self.active_alpha:.3f}")
        if hasattr(self.cd, 'conf') and self.cd.conf is not None:
            self.cd.conf.alpha = self.active_alpha
        if self.current_pair_idx < len(self.all_inds):
            self._update_sig_indicators(self.all_inds[self.current_pair_idx])

    def _on_norm_toggle(self):
        self.active_norms = {nm for nm, var in self.norm_vars.items()
                             if var.get()}
        # Norms change the y-axis values AND the CS (which is based on normalized CCG)
        self._pair_scale_cache.clear()
        self._session_scale_cache = None
        self._conn_strength_cache.clear()
        if self.current_pair_idx < len(self.all_inds):
            inds = self.all_inds[self.current_pair_idx]
            for seg in range(self.n_segments + 1):
                p = self._png_path(inds, seg)
                if os.path.exists(p):
                    os.remove(p)
        self.update_plot()

    # ------------------------------------------------------------------
    # Same-scale helpers (Task 1)
    # ------------------------------------------------------------------

    def _effective_bin_size(self) -> float:
        """Infer true bin_size from CCG array shape (robust to conf mutation)."""
        conf = self.ccg_data.conf
        n_bins = self.ccg_data.ccg.shape[-1]
        return conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size

    def _accumulate_ylim(self, ccg_raw, ccg_null_raw, ref, tgt, seg, is_custom, ymin, ymax, *, custom_time_h=None):
        """Accumulate y-axis limits for one trace using the active normalizations.

        Uses the same normalization helper as rendering, so pair-scale never clips
        when the user changes normalization settings.
        """
        ccg, ccg_null = apply_norms_to_ccg(
            ccg_raw,
            ccg_null_raw,
            ref, tgt, seg,
            self.active_norms,
            self.neurons,
            self.cd.nd,
            self.key.nd(),
            self.n_segments,
            bool(is_custom),
            custom_time_hours=custom_time_h,
        )
        ymin = min(ymin, float(np.nanmin(ccg)))
        ymax = max(ymax, float(np.nanmax(ccg)))
        if ccg_null is not None:
            ymax = max(ymax, float(np.nanmax(ccg_null)))
        return ymin, ymax

    def _compute_pair_scale(self, ref: int, tgt: int):
        """Return (ymin, ymax) unified across all segments for this pair.

        Includes:
        - all real segments
        - the summed 'All' view
        - all loaded custom segments (and their hi-res versions when available)

        Computed per-resolution and cached under (ref, tgt, res_key).
        """
        nd_key = self.key.nd()
        res_key = getattr(self, "_res_key", "lo")
        if res_key == "hi":
            cd = (self.cd._ccg_highres.get(nd_key)
                  if hasattr(self.cd, "_ccg_highres") else None)
            if cd is None:
                cd = self.ccg_data
        else:
            cd = (self.cd._ccg.get(nd_key)
                  if hasattr(self.cd, "_ccg") else self.ccg_data)
        if cd is None:
            self._pair_scale_cache[(ref, tgt, res_key)] = None
            return None

        ymin, ymax = 0.0, 0.0

        # Real segments
        for seg in range(self.n_segments):
            try:
                ccg_raw = cd.ccg[seg, ref, tgt, :]
                ccg_null_raw = (cd.ccg_null[seg, ref, tgt, :]
                                if getattr(cd, "ccg_null", None) is not None else None)
                ymin, ymax = self._accumulate_ylim(
                    ccg_raw, ccg_null_raw, ref, tgt, seg, False, ymin, ymax
                )
            except Exception:
                continue

        # All segments summed
        try:
            all_seg = self.n_segments
            ccg_raw = np.sum(cd.ccg[:, ref, tgt, :], axis=0)
            ccg_null_raw = (np.sum(cd.ccg_null[:, ref, tgt, :], axis=0)
                            if getattr(cd, "ccg_null", None) is not None else None)
            ymin, ymax = self._accumulate_ylim(
                ccg_raw, ccg_null_raw, ref, tgt, all_seg, False, ymin, ymax
            )
        except Exception:
            pass

        # Custom segments (include both lo/hi versions depending on res_key)
        cs_list = getattr(self, "_custom_segments", []) or []
        for ci, cs in enumerate(cs_list):
            if not isinstance(cs, dict):
                continue
            key_ccg = "ccg_hi" if (res_key == "hi" and cs.get("ccg_hi") is not None) else "ccg"
            key_null = "ccg_null_hi" if (res_key == "hi" and cs.get("ccg_null_hi") is not None) else "ccg_null"
            try:
                src = cs.get(key_ccg)
                if src is None:
                    continue
                ccg_raw = src[0, ref, tgt, :]
                null_src = cs.get(key_null)
                ccg_null_raw = null_src[0, ref, tgt, :] if null_src is not None else None
                seg_idx = self.n_segments + ci  # custom segment index in UI space
                ymin, ymax = self._accumulate_ylim(
                    ccg_raw, ccg_null_raw, ref, tgt, seg_idx, True, ymin, ymax,
                    custom_time_h=cs.get("total_time_hours"),
                )
            except Exception:
                continue

        result = (ymin, ymax * 1.1 if ymax > 0 else 1.0)
        self._pair_scale_cache[(ref, tgt, res_key)] = result
        return result

    def _compute_session_scale(self):
        """Return (ymin, ymax) unified across all pairs and segments in this key."""
        ymin, ymax = 0.0, 0.0
        nd_key = self.key.nd()
        res_key = getattr(self, "_res_key", "lo")
        if res_key == "hi":
            cd = (self.cd._ccg_highres.get(nd_key)
                  if hasattr(self.cd, "_ccg_highres") else None)
            if cd is None:
                cd = self.ccg_data
        else:
            cd = (self.cd._ccg.get(nd_key)
                  if hasattr(self.cd, "_ccg") else self.ccg_data)
        all_seg = self.n_segments
        for ref_tgt in self.all_inds:
            ref, tgt = int(ref_tgt[0]), int(ref_tgt[1])
            for seg in range(self.n_segments):
                try:
                    ccg_raw = cd.ccg[seg, ref, tgt, :]
                    ccg_null_raw = (cd.ccg_null[seg, ref, tgt, :]
                                    if getattr(cd, "ccg_null", None) is not None else None)
                    ymin, ymax = self._accumulate_ylim(
                        ccg_raw, ccg_null_raw, ref, tgt, seg, False, ymin, ymax
                    )
                except Exception:
                    pass
            ccg_raw = np.sum(cd.ccg[:, ref, tgt, :], axis=0)
            ccg_null_raw = (np.sum(cd.ccg_null[:, ref, tgt, :], axis=0)
                            if cd.ccg_null is not None else None)
            try:
                ymin, ymax = self._accumulate_ylim(
                    ccg_raw, ccg_null_raw, ref, tgt, all_seg, False, ymin, ymax
                )
            except Exception:
                pass
        return (ymin, ymax * 1.1 if ymax > 0 else 1.0)

    def _get_current_scale_ylim(self, ref: int, tgt: int):
        """Return (ymin, ymax) for the active scale mode, or None.
        Pair scale is cached per (ref, tgt, resolution) so hi-res and lo-res
        get independent y-axes instead of sharing the lo-res maximum."""
        if self._same_scale_mode == 'pair':
            cache_key = (ref, tgt, self._res_key)
            if cache_key not in self._pair_scale_cache:
                self._compute_pair_scale(ref, tgt)   # populates cache
            return self._pair_scale_cache[cache_key]
        if self._same_scale_mode == 'session':
            if self._session_scale_cache is None:
                self._session_scale_cache = self._compute_session_scale()
            return self._session_scale_cache
        return None

    def _on_pair_scale_toggle(self):
        if self._pair_scale_var.get():
            self._same_scale_mode = 'pair'
            self._sess_scale_var.set(False)   # mutual exclusion
        else:
            self._same_scale_mode = None
        self._pair_scale_cache.clear()
        self._clear_all_png_cache()
        self.update_plot()

    def _on_session_scale_toggle(self):
        if self._sess_scale_var.get():
            self._same_scale_mode = 'session'
            self._pair_scale_var.set(False)   # mutual exclusion
            self._session_scale_cache = None  # force recompute
        else:
            self._same_scale_mode = None
        self._clear_all_png_cache()
        self.update_plot()

    # ------------------------------------------------------------------
    # On-demand jitter (Task 2)
    # ------------------------------------------------------------------

    def _on_run_jitter(self):
        if self.current_pair_idx >= len(self.all_inds):
            return
        if self.neurons is None:
            messagebox.showerror("Jitter", "No neuron data attached.")
            return
        inds = self.all_inds[self.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        njitter = int(self._njitter_var.get())
        # User-chosen resolutions
        run_lo = bool(getattr(self, '_jitter_run_lo_var', tk.BooleanVar(value=True)).get())
        run_hi = bool(getattr(self, '_jitter_run_hi_var', tk.BooleanVar(value=False)).get())
        if not run_lo and not run_hi:
            messagebox.showwarning("Jitter", "Select at least one resolution (lo and/or hi).")
            return

        # Count running + queued tasks
        running = 1 if (self._jitter_proc is not None
                        and self._jitter_proc.is_alive()) else 0
        total = running + len(self._jitter_pending)
        if total >= _MAX_JITTER_QUEUE:
            messagebox.showwarning(
                "Jitter", f"Queue full ({total}/{_MAX_JITTER_QUEUE}).\n"
                          "Wait for running jitters to complete.")
            return

        # Determine segment scope (None = whole session, int = specific segment)
        seg_arg = self._jitter_seg()
        if seg_arg is not None:
            et = self.ccg_pointer.edge_times
            jitter_t0 = float(et.iloc[seg_arg]['start'])
            jitter_t1 = float(et.iloc[seg_arg]['stop'])
        else:
            jitter_t0, jitter_t1 = None, None

        # Enqueue one task per requested resolution (lo/hi)
        nd_key = self.key.nd()
        tasks = []
        if run_lo:
            lo_ccg_data = self.cd._ccg.get(nd_key) if hasattr(self.cd, '_ccg') else self.ccg_data
            lo_conf = lo_ccg_data.conf
            lo_n_bins = lo_ccg_data.ccg.shape[-1]
            bin_size_eff = lo_conf.duration / (lo_n_bins - 1) if lo_n_bins > 1 else lo_conf.bin_size
            tasks.append(('jitter', ref, tgt, njitter, 'lo', bin_size_eff, seg_arg, jitter_t0, jitter_t1))
        if run_hi:
            hi_ccg_data = None
            if hasattr(self.cd, '_ccg_highres'):
                hi_ccg_data = self.cd._ccg_highres.get(nd_key)
            if hi_ccg_data is None:
                messagebox.showwarning("Jitter", "High-res CCG not loaded; cannot run hi jitter.")
            else:
                hi_conf = hi_ccg_data.conf
                hi_n_bins = hi_ccg_data.ccg.shape[-1]
                bin_size_eff = hi_conf.duration / (hi_n_bins - 1) if hi_n_bins > 1 else hi_conf.bin_size
                tasks.append(('jitter', ref, tgt, njitter, 'hi', bin_size_eff, seg_arg, jitter_t0, jitter_t1))

        for t in tasks:
            self._jitter_pending.append(t)
        self._dbg_log(
            "H3",
            "ccg_ui.py:_on_run_jitter:enqueue",
            "Enqueued jitter task(s)",
            {
                "pair": [int(ref), int(tgt)],
                "res_keys": [x[4] for x in tasks],
                "seg_arg": seg_arg,
                "njitter": int(njitter),
                "current_segment": int(self.current_segment),
            },
        )
        self._update_jitter_btn_text()
        # Kick off processing if nothing is running
        self._jitter_start_next()

    def _is_task_running(self):
        return self._jitter_proc is not None and self._jitter_proc.is_alive()

    def _custom_ccg_is_running(self):
        return self._custom_ccg_thread is not None and self._custom_ccg_thread.is_alive()

    def _jitter_start_next(self):
        """Start the next queued jitter task if none is running."""
        if self._is_task_running():
            return
        if not self._jitter_pending:
            self._update_jitter_btn_text()
            return
        task = self._jitter_pending[0]
        _, ref, tgt, njitter, res_key, bin_size_eff = task[:6]
        seg_arg   = task[6] if len(task) > 6 else None
        jitter_t0 = task[7] if len(task) > 7 else None
        jitter_t1 = task[8] if len(task) > 8 else None
        nd_key = self.key.nd()
        if res_key == 'hi':
            ccg_data_eff = (self.cd._ccg_highres.get(nd_key)
                            if hasattr(self.cd, '_ccg_highres') else None)
        else:
            ccg_data_eff = (self.cd._ccg.get(nd_key)
                            if hasattr(self.cd, '_ccg') else self.ccg_data)
        if ccg_data_eff is None:
            # Drop this task and proceed
            try:
                self._jitter_pending.popleft()
            except Exception:
                pass
            self._jitter_start_next()
            return
        self._jitter_result_queue = _mp.Queue()
        self._jitter_proc = _mp.Process(
            target=jitter_worker,
            args=(self._jitter_result_queue, self.key, self.neurons,
                  ccg_data_eff, self.ccg_pointer.edge_times,
                  ref, tgt, njitter, bin_size_eff),
            kwargs={'segment': seg_arg, 't0': jitter_t0, 't1': jitter_t1},
            daemon=True,
        )
        self._jitter_proc.start()
        self._update_jitter_btn_text()
        if self._jitter_poll_id is None:
            self._jitter_poll_id = self.root.after(300, self._poll_jitter)

    def _custom_ccg_start_next(self):
        """Start the next queued custom CCG task if none is running."""
        if self._custom_ccg_is_running():
            return
        if not self._custom_ccg_pending:
            return
        task = self._custom_ccg_pending[0]
        if isinstance(task, dict):
            t0 = float(task.get('t0', 0.0))
            t1 = float(task.get('t1', 0.0))
            name = str(task.get('name', 'custom'))
            intervals = task.get('intervals')
            active_duration = task.get('active_duration')
            filter_state = task.get('filter_state', {})
            key_for_task = task.get('key', self.key)
            metadata = task.get('metadata', {})
        else:
            _, t0, t1, name, intervals, active_duration, filter_state = task
            key_for_task = self.key
            metadata = {}
        self._custom_ccg_thread_result.clear()
        _t_start = _time.monotonic()
        nd_key = key_for_task.nd()
        ccg_data_obj = self.cd._ccg.get(nd_key) if hasattr(self.cd, '_ccg') else self.ccg_data
        neurons_obj = (self.cd.nd.data[nd_key]
                       if getattr(self.cd, 'nd', None) is not None else None)
        if ccg_data_obj is None or neurons_obj is None:
            try:
                self.cd.get_ccg()
            except Exception as ex:
                messagebox.showerror("Custom CCG", f"Session load failed for {key_for_task.session}:\n{ex}")
                try:
                    failed = self._custom_ccg_pending.popleft()
                    self._on_split_batch_task_done(failed)
                except Exception:
                    pass
                self._custom_ccg_start_next()
                return
            ccg_data_obj = self.cd._ccg.get(nd_key) if hasattr(self.cd, '_ccg') else self.ccg_data
            neurons_obj = (self.cd.nd.data[nd_key]
                           if getattr(self.cd, 'nd', None) is not None else None)
            if ccg_data_obj is None or neurons_obj is None:
                print(f"[CustomCCG] missing session data after load: {key_for_task.session}")
                try:
                    failed = self._custom_ccg_pending.popleft()
                    self._on_split_batch_task_done(failed)
                except Exception:
                    pass
                self._custom_ccg_start_next()
                return

        def _ccg_worker(_t0=t0, _t1=t1, _name=name, _intervals=intervals,
                        _ad=active_duration, _fs=filter_state, _key=key_for_task,
                        _meta=metadata, _ccg_data=ccg_data_obj, _neurons=neurons_obj):
            try:
                neurons_override = (
                    self._ts_apply_brain_state_intervals(_intervals, _t0, _t1, neurons_obj=_neurons)
                    if _intervals is not None else None)
                result = self._compute_custom_segment(
                    _t0, _t1, _name,
                    neurons_override=neurons_override, active_duration=_ad,
                    key_override=_key, neurons_obj=_neurons, ccg_data_obj=_ccg_data,
                    metadata=_meta)
                if result is not None:
                    result['filter_state'] = _fs
                    result['compute_sec'] = _time.monotonic() - _t_start
                    result['_task_session'] = str(_key.session)
                self._custom_ccg_thread_result.append(
                    result if result is not None else {'error': 'compute returned None'})
            except Exception as ex:
                self._custom_ccg_thread_result.append({'error': str(ex)})

        self._custom_ccg_thread = threading.Thread(target=_ccg_worker, daemon=True)
        self._custom_ccg_thread.start()
        if self._custom_ccg_poll_id is None:
            self._custom_ccg_poll_id = self.root.after(300, self._poll_custom_ccg)

    def _update_jitter_btn_text(self):
        running = self._is_task_running()
        queued = len(self._jitter_pending)
        if running and self._jitter_pending:
            task = self._jitter_pending[0]
            ref, tgt = task[1], task[2]
            seg_arg = task[6] if len(task) > 6 else None
            if seg_arg is None:
                seg_name = _ALL_SEGS
            else:
                try:
                    seg_name = str(self.segment_names[int(seg_arg)])
                except Exception:
                    seg_name = f"seg{seg_arg}"
            label = f"Jitter [{ref},{tgt}] {seg_name}…"
            if queued > 1:
                self._jitter_btn_text.set(f"{label} +{queued - 1} queued")
            else:
                self._jitter_btn_text.set(label)
        else:
            self._jitter_btn_text.set("Run Jitter")

    def _poll_jitter(self):
        """Poll the running jitter process; collect result and start next."""
        if self._is_task_running():
            self._jitter_poll_id = self.root.after(300, self._poll_jitter)
            return
        self._jitter_poll_id = None

        completed = self._jitter_pending.popleft() if self._jitter_pending else None
        result = None
        try:
            if (self._jitter_result_queue is not None
                    and not self._jitter_result_queue.empty()):
                result = self._jitter_result_queue.get_nowait()
        except Exception:
            pass
        if self._jitter_proc is not None:
            self._jitter_proc.join(timeout=1)
            self._jitter_proc = None
        self._jitter_result_queue = None

        if result is not None and not result.get('error') and result.get('j_avg') is not None:
            res_key = completed[4] if completed else 'lo'
            seg_arg = completed[6] if completed and len(completed) > 6 else None
            cache_key = (result['ref'], result['tgt'], res_key, seg_arg)
            jitter_val = (
                result.get('j_avg'),
                result.get('j_pval'),
                result.get('j_pval_bins'),
                result.get('j_lo'),
                result.get('j_hi'),
            )
            self._jitter_cache_put(cache_key, jitter_val)
            nd_key = self.key.nd()
            if hasattr(self.cd, '_jitter_results'):
                if nd_key not in self.cd._jitter_results:
                    self.cd._jitter_results[nd_key] = {}
                self.cd._jitter_results[nd_key][cache_key] = jitter_val
            completed_pair = (result['ref'], result['tgt'])
            self._jitter_unviewed.add(completed_pair)
            self._apply_jitter_list_colors(pair=completed_pair)
            if self._mark_jitter_viewed():
                self._update_jitter_sig_buttons()
                self.update_plot()
            self._update_conn_str_metric_availability()
            self.root.bell()
        elif result is not None and result.get('error'):
            messagebox.showerror("Jitter", f"Jitter failed:\n{result['error']}")

        self._jitter_start_next()

    def _poll_custom_ccg(self):
        """Poll the running custom CCG thread; collect result and start next."""
        if self._custom_ccg_is_running():
            self._custom_ccg_poll_id = self.root.after(300, self._poll_custom_ccg)
            return
        self._custom_ccg_poll_id = None

        completed_task = self._custom_ccg_pending.popleft() if self._custom_ccg_pending else None
        if self._custom_ccg_thread is not None:
            self._custom_ccg_thread.join(timeout=1)
            self._custom_ccg_thread = None
        result = self._custom_ccg_thread_result[0] if self._custom_ccg_thread_result else None
        self._custom_ccg_thread_result.clear()

        if result is not None and not result.get('error'):
            if isinstance(completed_task, dict) and completed_task.get('auto_save'):
                key_for_save = completed_task.get('key', self.key)
                _sess_save = str(key_for_save.session)
                self._purge_timestamped_custom_ccg_npz(_sess_save, str(result['name']))
                fname = self._ccg_cache_filename_for_key(result['name'], key_for_save)
                path = os.path.join(self._ccg_cache_dir, fname)
                arrays = dict(
                    name_=np.array(result['name']),
                    t0_=np.array(result['t0']),
                    t1_=np.array(result['t1']),
                    ccg=result['ccg'],
                    ccg_null=result['ccg_null'],
                    pval=result['pval'],
                    pval_corrected=result['pval_corrected'],
                    compute_sec_=np.array(result.get('compute_sec', float('nan'))),
                    active_duration_=np.array(result.get('active_duration', float('nan'))),
                    total_time_hours_=np.array(result.get('total_time_hours', float('nan'))),
                    filter_state_=np.array(json.dumps(result.get('filter_state', {}))),
                    metadata_=np.array(json.dumps(result.get('metadata', {}))),
                    **({'firing_rates': result['firing_rates']}
                       if 'firing_rates' in result else {}),
                )
                for k in ('ccg_hi', 'ccg_null_hi', 'pval_hi', 'pval_corrected_hi'):
                    if k in result:
                        arrays[k] = result[k]
                np.savez_compressed(path, **arrays)
                result['src_path'] = path
                self._emit_custom_ccg_inventory_event()
            should_load = (not isinstance(completed_task, dict)
                            or bool(completed_task.get('load_into_ui', True)))
            _tk_done = (completed_task.get('key', self.key)
                        if isinstance(completed_task, dict) else self.key)
            _lsess = str(result.get('_task_session', getattr(_tk_done, 'session', '')))
            _lst = self._custom_segments_by_session.setdefault(_lsess, [])
            idx, _did_append = self._upsert_custom_segment_by_name(_lst, result)
            if should_load and self._custom_segments is _lst:
                self._build_sig_chips()
                self.current_segment = self.n_segments + 1 + idx
                self._clamp_current_segment_for_session()
                self._update_segment_label()
                self.update_plot()
            if hasattr(self, '_ts_status_var'):
                self._ts_status_var.set(f"Done: {result.get('name', '')}")
            self.root.bell()
        elif result is not None and result.get('error'):
            messagebox.showerror("Custom CCG", f"Computation failed:\n{result['error']}")

        if completed_task is not None:
            self._on_split_batch_task_done(completed_task)

        self._custom_ccg_start_next()

    def _on_save_jitter(self):
        """Save all jitter results currently in cd._jitter_results to disk."""
        if not hasattr(self.cd, 'save_jitter'):
            messagebox.showerror("Save Jitter", "CCGDataset does not support jitter persistence.")
            return
        try:
            self.cd.save_jitter()
            total = sum(len(v) for v in self.cd._jitter_results.values())
            messagebox.showinfo("Save Jitter", f"Saved {total} pair(s).")
        except Exception as exc:
            messagebox.showerror("Save Jitter", f"Save failed:\n{exc}")

    def _load_jitter_from_cd(self):
        """Populate _jitter_cache from cd._jitter_results for the current session."""
        if not hasattr(self.cd, '_jitter_results'):
            return
        nd_key = self.key.nd()
        pairs = self.cd._jitter_results.get(nd_key, {})
        for cache_key, val in pairs.items():
            # Promote old 3-tuple keys (ref, tgt, res_key) → (ref, tgt, res_key, None)
            if len(cache_key) == 3:
                cache_key = cache_key + (None,)
            self._jitter_cache_put(cache_key, val)
        if pairs:
            self._update_jitter_sig_buttons()
            self._apply_jitter_list_colors()

    def _jitter_seg(self, seg=None):
        """Return the segment key for jitter cache lookups.

        Returns None for 'All' and custom segments (whole-session data).
        Returns the integer segment index for specific real segments.
        """
        if seg is None:
            seg = self.current_segment
        if seg == self.n_segments or self._is_custom_segment(seg):
            return None
        return int(seg)

    def _jitter_cache_put(self, key, value):
        """Insert into jitter cache with LRU eviction when full."""
        if key in self._jitter_cache:
            self._jitter_cache.move_to_end(key)
            self._jitter_cache[key] = value
        else:
            self._jitter_cache[key] = value
            while len(self._jitter_cache) > self._JITTER_CACHE_MAX:
                self._jitter_cache.popitem(last=False)  # evict oldest

    def _on_clear_jitter(self):
        if self.current_pair_idx >= len(self.all_inds):
            return
        inds = self.all_inds[self.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        # Clear both resolutions for this pair/segment
        segk = self._jitter_seg()
        for rk in ('lo', 'hi'):
            cache_key = (ref, tgt, rk, segk)
            self._jitter_cache.pop(cache_key, None)
        if hasattr(self.cd, '_jitter_results'):
            nd_key = self.key.nd()
            if nd_key in self.cd._jitter_results:
                for rk in ('lo', 'hi'):
                    self.cd._jitter_results[nd_key].pop((ref, tgt, rk, segk), None)
        self._jitter_unviewed.discard((ref, tgt))
        self._clear_all_png_cache()
        self._update_jitter_sig_buttons()
        self._apply_jitter_list_colors()
        self.update_plot()

    # Jitter list highlight colors (fg set explicitly for dark-mode visibility)
    _JITTER_UNVIEWED_BG = '#FFEE58'   # bright yellow — ready, not yet viewed
    _JITTER_UNVIEWED_FG = '#333333'
    _JITTER_VIEWED_BG   = '#FFF9C4'   # muted light yellow — has jitter (viewed)
    _JITTER_VIEWED_FG   = '#333333'

    def _pair_coords_for_jitter(self, inds) -> tuple[int, int] | None:
        """``(ref, tgt)`` for jitter coloring / cache keys from a selection triple or pair."""
        if not isinstance(inds, tuple) or len(inds) < 2:
            return None
        try:
            if len(inds) >= 3 and (
                    isinstance(inds[0], Key) or isinstance(inds[0], str)):
                return int(inds[1]), int(inds[2])
            return int(inds[0]), int(inds[1])
        except (TypeError, ValueError):
            return None

    def _apply_jitter_list_colors(self, pair=None):
        """Color pair list items based on jitter cache state.

        Parameters
        ----------
        pair : tuple or None
            If given as (ref, tgt), update only that pair's item.
            If None, update all items in both listboxes.
        """
        def _apply_row(listbox, idx, ref, tgt):
            p = (int(ref), int(tgt))
            if pair is not None and p != pair:
                return False
            has_any_res = any(
                k[0] == p[0] and k[1] == p[1] for k in self._jitter_cache)
            try:
                if p in self._jitter_unviewed:
                    listbox.itemconfig(idx, background=self._JITTER_UNVIEWED_BG,
                                       foreground=self._JITTER_UNVIEWED_FG)
                elif has_any_res:
                    listbox.itemconfig(idx, background=self._JITTER_VIEWED_BG,
                                       foreground=self._JITTER_VIEWED_FG)
                else:
                    listbox.itemconfig(idx, background='', foreground='')
            except tk.TclError:
                return False
            return True

        if not hasattr(self, 'unselected_list'):
            return  # called during LeftPanel construction before bridge aliases exist

        if getattr(self, '_session_any_mode', False):
            # Selected rows are ``_sel_list_pairs`` (headers = None); indices are
            # ``(Key, r, t)``. ``selected_inds`` is triples — do not sort/map by idx.
            try:
                n_items = int(self.selected_list.size())
            except Exception:
                n_items = 0
            sel_map = getattr(self, '_sel_list_pairs', None) or []
            for idx, entry in enumerate(sel_map):
                if idx >= n_items or entry is None:
                    continue
                rt = self._pair_coords_for_jitter(entry)
                if rt is None:
                    continue
                if _apply_row(self.selected_list, idx, *rt) and pair is not None:
                    self._reapply_bookmark_list_styles()
                    return
            self._reapply_bookmark_list_styles()
            return

        for listbox, inds_set in [(self.unselected_list, self.unselected_inds),
                                  (self.selected_list, self.selected_inds)]:
            sorted_items = sorted(inds_set)
            try:
                n_items = int(listbox.size())
            except Exception:
                n_items = 0
            for idx, inds in enumerate(sorted_items):
                if idx >= n_items:
                    break
                rt = self._pair_coords_for_jitter(inds)
                if rt is None:
                    continue
                if _apply_row(listbox, idx, *rt) and pair is not None:
                    self._reapply_bookmark_list_styles()
                    return
        self._reapply_bookmark_list_styles()

    def _mark_jitter_viewed(self) -> bool:
        """Mark current pair's jitter as viewed; auto-enable overlay.

        Does NOT call update_plot() — the caller is responsible for that.
        Returns True if a pair was newly marked as viewed.
        """
        if self.current_pair_idx >= len(self.all_inds):
            return False
        inds = self.all_inds[self.current_pair_idx]
        pair = (int(inds[0]), int(inds[1]))
        if pair in self._jitter_unviewed:
            self._jitter_unviewed.discard(pair)
            self._apply_jitter_list_colors(pair=pair)
            # Auto-enable jitter overlay only if current segment has jitter
            seg_key = self._jitter_seg()
            if self._jitter_cache.get((pair[0], pair[1], 'lo', seg_key)) is not None:
                self._sig_jitter_pc_var.set(True)
                self._line_jitter_var.set(False)
            return True
        return False

    def _finalize_normalization(self):
        if not self.active_norms:
            messagebox.showinfo("Normalize all", "No normalizations are toggled on.")
            return
        if self.neurons is None and any(
                nm in (NormalizeBy.REF_FRATE, NormalizeBy.TARGET_FRATE)
                for nm in self.active_norms):
            messagebox.showerror(
                "Normalize all",
                "Neuron data is unavailable — cannot normalize by firing rate.")
            return
        norm_names = ', '.join(nm.name for nm in self.active_norms)
        if not messagebox.askyesno(
                "Normalize all",
                f"Rewrite current normalization into the stored CCG arrays?\n\n"
                "This permanently modifies the in-memory dataset (ccg/ccg_null) and cannot be undone.\n"
                "Use this only if you want the data itself to become normalized, not just the display."):
            return
        cd = self.ccg_data
        nd_key = self.key.nd()
        edge_times = (self.cd.nd.edge_times[nd_key]
                      if self.cd.nd is not None else None)
        frates = (self.cd.nd.segment_firing_rates[nd_key]
                  if self.cd.nd is not None else None)
        for seg in range(cd.n_segment):
            for nm in self.active_norms:
                if nm == NormalizeBy.REF_FRATE and frates is not None:
                    fr = frates[seg]
                    cd.ccg[seg] /= np.maximum(fr[np.newaxis, :, np.newaxis], 1e-12)
                    if cd.ccg_null is not None:
                        cd.ccg_null[seg] /= np.maximum(
                            fr[np.newaxis, :, np.newaxis], 1e-12)
                elif nm == NormalizeBy.TARGET_FRATE and frates is not None:
                    fr = frates[seg]
                    cd.ccg[seg] /= np.maximum(fr[:, np.newaxis, np.newaxis], 1e-12)
                    if cd.ccg_null is not None:
                        cd.ccg_null[seg] /= np.maximum(
                            fr[:, np.newaxis, np.newaxis], 1e-12)
                elif nm == NormalizeBy.TIME_SPAN and edge_times is not None:
                    et = float(edge_times.iloc[seg]['effective_time_hours'])
                    cd.ccg[seg] /= max(et, 1e-12)
                    if cd.ccg_null is not None:
                        cd.ccg_null[seg] /= max(et, 1e-12)
        existing = list(getattr(cd.conf, 'normalize_methods', []) or [])
        for nm in self.active_norms:
            if nm not in existing:
                existing.append(nm)
        cd.conf.normalize_methods = existing
        self._clear_all_png_cache()
        for var in self.norm_vars.values():
            var.set(False)
        self.active_norms = set()
        self.update_plot()
        messagebox.showinfo("Normalize all",
                            "Applied normalization to stored CCG data. Don't forget to Save Selections.")

    def _clear_all_png_cache(self):
        self._pregen_cancel = True  # stop any in-progress pre-generation
        self._terminate_pregen_proc()
        for f in os.listdir(self.tmp_dir):
            if f.endswith('.png'):
                try:
                    os.remove(os.path.join(self.tmp_dir, f))
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # PNG pre-generation  (runs as an independent subprocess)
    # ------------------------------------------------------------------

    def _terminate_pregen_proc(self):
        """Terminate a running pre-gen subprocess, if any."""
        if self._pregen_poll_id is not None:
            try:
                self.root.after_cancel(self._pregen_poll_id)
            except Exception:
                pass
            self._pregen_poll_id = None
        proc = getattr(self, '_pregen_proc', None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        self._pregen_proc = None

    def _pregen_job_payload(self, cfg: dict) -> dict:
        """Build the JSON job dict for the pre-gen subprocess."""
        nd_key = self.key.nd()
        has_highres = (hasattr(self.cd, '_ccg_highres')
                       and self.cd._ccg_highres.get(nd_key) is not None)
        ccg_lo_path = os.path.expanduser(self.cd._ccgdata_path()) + '.hkl'
        ccg_hi_path = (os.path.expanduser(self.cd.highres_save_path()) + '.hkl'
                       if has_highres else None)

        # Neurons data for norms / shank labels
        neurons_fr    = None
        neurons_shank = None
        edge_times    = None
        if self.neurons is not None:
            fr = getattr(self.neurons, 'firing_rate', None)
            if fr is not None:
                neurons_fr = [float(x) for x in fr]
            sh = getattr(self.neurons, 'shank_ids', None)
            if sh is not None:
                neurons_shank = [int(x) for x in sh]
        # edge_times for TIME_SPAN norm
        et_df = getattr(self.ccg_pointer, 'edge_times', None)
        if et_df is not None:
            try:
                edge_times = [float(et_df.iloc[s]['effective_time_hours'])
                              for s in range(self.n_segments)]
            except Exception:
                edge_times = None

        return {
            'nd_key':          str(nd_key),
            'ccg_lo_path':     ccg_lo_path,
            'ccg_hi_path':     ccg_hi_path,
            'has_highres':     has_highres,
            'n_segments':      self.n_segments,
            'segment_names':   self.segment_names,
            'pairs':           [list(map(int, p)) for p in self.all_inds],
            'tmp_dir':         self.tmp_dir,
            'cache_config':    cfg,
            'neurons_firing_rate': neurons_fr,
            'neurons_shank_ids':   neurons_shank,
            'edge_times':      edge_times,
        }

    def _launch_pregen_subprocess(self, cfg: dict, status_var=None, priority: str = 'user'):
        """Write job file and launch pregen.py as an independent subprocess.

        priority='user': preempts any running auto task; runs immediately.
        priority='auto': skipped silently if any pregen is already running.
        """
        if priority == 'auto':
            # Never interrupt a running task (auto or user) for a background auto-gen
            if self._pregen_proc is not None and self._pregen_proc.poll() is None:
                return
        else:
            # User-requested: preempt any running auto task transparently
            if self._pregen_priority == 'auto':
                self._terminate_pregen_proc()
            elif self._pregen_proc is not None and self._pregen_proc.poll() is None:
                # Another user task already running — terminate it first
                self._terminate_pregen_proc()

        self._pregen_priority = priority
        job = self._pregen_job_payload(cfg)
        job_path = os.path.join(self.tmp_dir, '_pregen_job.json')
        with open(job_path, 'w', encoding='utf-8') as fh:
            json.dump(job, fh)

        script = str(_Path(__file__).parent / 'pregen.py')
        self._pregen_proc = subprocess.Popen([sys.executable, script, job_path])
        if priority == 'user':
            print(f"[CCGReviewUI] pre-gen subprocess started (pid {self._pregen_proc.pid})")
        if status_var is not None:
            status_var.set("Generating…")
        self._pregen_poll_id = self.root.after(
            1000, self._poll_pregen_proc, status_var)

    def _poll_pregen_proc(self, status_var=None):
        """Poll the pre-gen subprocess for completion; update status_var when done."""
        self._pregen_poll_id = None
        proc = getattr(self, '_pregen_proc', None)
        if proc is None:
            return
        if proc.poll() is None:
            # Still running — schedule next poll
            self._pregen_poll_id = self.root.after(
                1000, self._poll_pregen_proc, status_var)
        else:
            # Finished
            if status_var is not None:
                status_var.set("Idle")
            if self._pregen_priority == 'user':
                print(f"[CCGReviewUI] pre-gen subprocess finished "
                      f"(exit {proc.returncode})")
            self._pregen_proc = None
            self._pregen_priority = None

    def _pregen_png_cache(self):
        """Launch background pre-gen subprocess for all pairs × segments."""
        self._pregen_cancel = True   # keep for on-demand render cancellation
        self._pregen_cancel = False
        if self._cache_config is not None:
            cfg = dict(self._cache_config)
        else:
            cfg = self._current_display_config()
        self._launch_pregen_subprocess(cfg)

    def _start_pregen_with_defaults(self, status_var=None):
        """Launch pre-gen subprocess using the saved cache configuration (user-requested)."""
        if self._cache_config is None:
            messagebox.showinfo(
                "Pre-gen",
                "No cache configuration set.\n\n"
                "Go to Settings → Cache Configuration and click\n"
                "\"Capture current settings\" to define the one\n"
                "display state that will be saved to disk cache.",
                parent=self.root)
            return
        self._launch_pregen_subprocess(dict(self._cache_config), status_var=status_var,
                                       priority='user')

    # ------------------------------------------------------------------
    # Segment filter
    # ------------------------------------------------------------------

    def _apply_seg_filter(self):
        name = self.seg_filter_var.get().strip()
        if not name:
            self.active_segment_filter = None
            self._draw_network()
            return
        try:
            idx = self.segment_names.index(name)
        except ValueError:
            matches = [i for i, n in enumerate(self.segment_names)
                       if n.startswith(name)]
            if len(matches) == 1:
                idx = matches[0]
            else:
                messagebox.showwarning(
                    "Segment filter",
                    f"'{name}' not found:\n" + ', '.join(self.segment_names))
                return
        self.active_segment_filter = idx
        self.current_segment = idx
        self.segment_var.set(self.segment_names[idx])
        self.plot_title_var.set(self.get_plot_title())
        self.update_plot()
        self._draw_network()

    def _clear_seg_filter(self):
        self.seg_filter_var.set('')
        self.active_segment_filter = None
        self._draw_network()

    # ------------------------------------------------------------------
    # Sig-chip builder
    # ------------------------------------------------------------------

    def _toggle_sbs_mode(self):
        """Toggle lo/hi side-by-side comparison view."""
        self._sbs_mode = not self._sbs_mode
        self._build_sig_chips()
        self.update_plot()

    def _build_sig_chips(self):
        for widget in self.sig_frame.winfo_children():
            widget.destroy()
        self.seg_sig_labels = []
        # Conn-strength chip (right-aligned, to the left of lo|hi)
        cs_on = self._conn_str_show_var.get()
        cs_btn = tk.Label(
            self.sig_frame, text="CS",
            relief=tk.SUNKEN if cs_on else tk.RAISED,
            font=('Arial', 8, 'bold'),
            bg='#B0C4FF' if cs_on else '#E0E0E0',
            padx=4, pady=2, cursor='hand2')
        cs_btn.pack(side=tk.RIGHT, padx=(2, 2))
        cs_btn.bind('<Button-1>', lambda e: (
            self._conn_str_show_var.set(not self._conn_str_show_var.get()),
            self._on_conn_str_toggle(),
            self._build_sig_chips(),
        ))
        # Side-by-side toggle button (right-aligned)
        sbs_bg = '#B0C4FF' if self._sbs_mode else '#E0E0E0'
        sbs_relief = tk.SUNKEN if self._sbs_mode else tk.RAISED
        sbs_btn = tk.Label(
            self.sig_frame, text="lo|hi",
            relief=sbs_relief, font=('Arial', 8, 'bold'),
            bg=sbs_bg, padx=4, pady=2, cursor='hand2')
        sbs_btn.pack(side=tk.RIGHT, padx=(2, 6))
        sbs_btn.bind('<Button-1>', lambda e: self._toggle_sbs_mode())
        ttk.Label(self.sig_frame, text="Segments:").pack(side=tk.LEFT, padx=(4, 2))
        # "All" chip first (visual), appended to seg_sig_labels after real segs (index = n_segments)
        lbl_all = tk.Label(
            self.sig_frame, text="All",
            relief=tk.RAISED, font=('Arial', 8, 'bold'),
            bg='#E0E0E0', padx=4, pady=2, cursor='hand2')
        lbl_all.pack(side=tk.LEFT, padx=(2, 0))
        # Use current ``self.n_segments`` at click time (Any-mode session hops
        # may not rebuild chips immediately; "All" is always ``n_segments``).
        lbl_all.bind(
            '<Button-1>',
            lambda e: self._on_segment_chip_primary_click(self.n_segments))
        for _seq in ('<Control-Button-1>', '<Command-Button-1>'):
            lbl_all.bind(
                _seq,
                lambda e: (self._toggle_segment_chip_multi(self.n_segments), 'break')[1])
        lbl_all.bind(
            '<Button-2>',
            lambda e: self._segment_chip_ctx_menu(e, self.n_segments))
        lbl_all.bind(
            '<Button-3>',
            lambda e: self._segment_chip_ctx_menu(e, self.n_segments))
        # Separator between All and individual segments
        tk.Frame(self.sig_frame, width=1, bg='#AAAAAA').pack(
            side=tk.LEFT, fill=tk.Y, padx=3, pady=2)
        # Real segment chips (indices 0..n_segments-1 in seg_sig_labels)
        for i, name in enumerate(self.segment_names):
            lbl = tk.Label(
                self.sig_frame, text=name,
                relief=tk.RAISED, font=('Arial', 8),
                bg='#E0E0E0', padx=4, pady=2)
            lbl.pack(side=tk.LEFT, padx=2)
            lbl.bind('<Button-1>', lambda e, idx=i: self._on_segment_chip_primary_click(idx))
            for _seq in ('<Control-Button-1>', '<Command-Button-1>'):
                lbl.bind(_seq, lambda e, idx=i: (self._toggle_segment_chip_multi(idx), 'break')[1])
            lbl.bind('<Button-2>', lambda e, idx=i: self._segment_chip_ctx_menu(e, idx))
            lbl.bind('<Button-3>', lambda e, idx=i: self._segment_chip_ctx_menu(e, idx))
            self.seg_sig_labels.append(lbl)
        # Append All chip at index n_segments (after real segs for correct index lookup)
        self.seg_sig_labels.append(lbl_all)
        # Custom segment chips
        for ci, cs in enumerate(getattr(self, '_custom_segments', [])):
            seg_idx = self.n_segments + 1 + ci
            lbl_cust = tk.Label(
                self.sig_frame, text=cs['name'],
                relief=tk.SUNKEN, font=('Arial', 8, 'italic'),
                bg='#FFF9C4', fg='#5D4037', padx=4, pady=2)
            lbl_cust.pack(side=tk.LEFT, padx=(4, 2))
            lbl_cust.bind('<Button-1>',
                          lambda e, idx=seg_idx: self._on_segment_chip_primary_click(idx))
            for _seq in ('<Control-Button-1>', '<Command-Button-1>'):
                lbl_cust.bind(_seq, lambda e, idx=seg_idx: (self._toggle_segment_chip_multi(idx), 'break')[1])
            lbl_cust.bind('<Button-2>',
                          lambda e, idx=seg_idx: self._segment_chip_ctx_menu(e, idx))
            lbl_cust.bind('<Button-3>',
                          lambda e, idx=seg_idx: self._segment_chip_ctx_menu(e, idx))
            lbl_cust.bind('<Double-Button-1>',
                          lambda e, idx=ci: self._remove_custom_segment(idx))
            self.seg_sig_labels.append(lbl_cust)

    def _on_segment_chip_primary_click(self, idx: int):
        """Normal click: clear multi-selection and jump to idx."""
        try:
            self._stacked_segments.clear()
        except Exception:
            self._stacked_segments = set()
        self._jump_to_segment(idx)

    def _toggle_segment_chip_multi(self, idx: int):
        """Ctrl/Cmd-click: toggle idx in the multi-selection without changing current_segment."""
        sel = getattr(self, '_stacked_segments', set()) or set()
        if idx in sel:
            sel.discard(idx)
        else:
            sel.add(idx)
        self._stacked_segments = sel
        inds = self._current_inds()
        if inds is not None:
            self._update_sig_indicators(inds)

    def _segment_chip_ctx_menu(self, event, idx: int):
        """Right-click menu for segment chips."""
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(
            label="Stack selected segments",
            command=self._show_stacked_segments,
        )
        menu.add_command(
            label="Select only this segment",
            command=lambda i=idx: self._select_only_segment(i),
        )
        menu.add_command(
            label="Clear segment selection",
            command=self._clear_segment_selection,
        )
        menu.tk_popup(event.x_root, event.y_root)

    def _select_only_segment(self, idx: int):
        self._stacked_segments = {int(idx)}
        inds = self._current_inds()
        if inds is not None:
            self._update_sig_indicators(inds)

    def _clear_segment_selection(self):
        try:
            self._stacked_segments.clear()
        except Exception:
            self._stacked_segments = set()
        inds = self._current_inds()
        if inds is not None:
            self._update_sig_indicators(inds)

    def _show_stacked_segments(self):
        """Trigger stacked-segments view for the current pair."""
        if not getattr(self, '_stacked_segments', None):
            return
        self.update_plot()

    # ------------------------------------------------------------------
    # Key / dropdown helpers
    # ------------------------------------------------------------------

    def _all_nd_keys(self) -> list:
        """Return unique nd_keys (one per session) across the dataset.

        Prefers ``cd._ccg`` (keyed by nd_keys directly) so that ALL sessions
        are represented even when some have no significant pairs in ``cd.data``.
        Falls back to enumerating ``cd.data`` if ``_ccg`` is unavailable.
        """
        seen, seen_str = [], set()
        # Primary source: _ccg is keyed by nd_keys, one per session
        ccg_source = getattr(self.cd, '_ccg', None) or {}
        for nk in ccg_source.keys():
            s = str(nk)
            if s not in seen_str:
                seen.append(nk)
                seen_str.add(s)
        # Secondary: pick up any sessions present only in cd.data
        for k in self.cd.data.keys():
            nk = k.nd()
            s = str(nk)
            if s not in seen_str:
                seen.append(nk)
                seen_str.add(s)
        # Tertiary: if neither _ccg nor cd.data are populated (lazy-load case),
        # enumerate from the neuron-dataset edge_times which is always available
        if not seen:
            for nk in getattr(getattr(self.cd, 'nd', None), 'edge_times', {}).keys():
                s = str(nk)
                if s not in seen_str:
                    seen.append(nk)
                    seen_str.add(s)
        return [_ALL_SESSION_MARKER] + seen

    def _session_label(self, nd_key) -> str:
        if nd_key is _ALL_SESSION_MARKER:
            return 'All'
        return str(nd_key.session) if nd_key.session else str(nd_key)

    def _real_nd_keys_ordered(self) -> list:
        """Session nd_keys only (excludes the synthetic ``any`` marker)."""
        keys = self._all_nd_keys()
        return keys[1:] if keys and keys[0] is _ALL_SESSION_MARKER else keys

    def _sanitize_sess_slug(self, sess: str) -> str:
        s = re.sub(r'[^\w.\-]+', '_', str(sess))[:48]
        return s or 'sess'

    def _type_key_for_nd(self, nd_key):
        """Pick a data Key matching *nd_key* and the current type label (any-mode)."""
        lbl = self._type_label(self.key)
        matches = [k for k in self.cd.data.keys()
                   if k.nd() == nd_key and self._type_label(k) == lbl]
        return matches[0] if matches else None

    def _nd_key_for_session_str(self, sess_str: str):
        for nk in self._real_nd_keys_ordered():
            if str(getattr(nk, 'session', nk)) == sess_str:
                return nk
        return None

    def _any_conn_type_sort_key(self, key):
        """Order connection types: E before I; pyr→pyr before pyr→inter, etc."""
        exc = getattr(key, 'excitability', None)
        ep = 0 if str(exc).upper() == 'E' else 1 if str(exc).upper() == 'I' else 2
        ct = getattr(key, 'conn_type', None)

        def _cell_rank(cell) -> tuple:
            s = str(cell).lower()
            if s in ('pyr', 'pyramidal'):
                return (0, s)
            if s in ('inter', 'int', 'interneuron'):
                return (1, s)
            return (2, s)

        if isinstance(ct, (tuple, list)) and len(ct) >= 2:
            a, b = _cell_rank(ct[0]), _cell_rank(ct[1])
            ct_key = (a, b)
        else:
            ct_key = ((99, ''), (99, ''))
        epoch = str(getattr(key, 'epoch', None) or '')
        return (ep, ct_key, epoch, self._type_label(key))

    def _available_type_keys_any(self) -> list:
        """One representative Key per distinct type label across all sessions."""
        by_lbl: dict = {}
        keys_sorted = sorted(
            self.cd.data.keys(),
            key=lambda k: (str(getattr(k, 'session', '')), self._type_label(k)))
        for k in keys_sorted:
            lbl = self._type_label(k)
            by_lbl.setdefault(lbl, k)
        return sorted(by_lbl.values(), key=self._any_conn_type_sort_key)

    def _any_group_header_names(self) -> list[str]:
        """Sorted user group names (tags) for ``any``-mode list sections."""

        def _sort_key(n: str):
            try:
                return (0, int(n), '')
            except (ValueError, TypeError):
                return (1, 0, n)

        names = [
            g for g in self._groups
            if not g.startswith('__') and not g.startswith(_SPECIAL_PREFIX)
        ]
        return sorted(names, key=_sort_key)

    def _any_triples_in_group(self, gname: str) -> set[tuple]:
        """All (session, ref, tgt) in *gname* for the current connection type."""
        lbl = self._type_label(self.key)
        out: set[tuple] = set()
        for k in self.cd.data.keys():
            if self._type_label(k) != lbl:
                continue
            sess = str(k.session)
            valid = self._all_inds_set_for_ptr(self.cd.data.get(k))
            if not valid:
                continue
            for pair in self._group_pairs(gname, session=sess):
                r, t = int(pair[0]), int(pair[1])
                if (r, t) in valid:
                    out.add((sess, r, t))
        return out

    def _any_nd_keys_for_group(self, gname: str) -> list:
        """Neuron-dataset keys that have ≥1 pair in *gname* (current type)."""
        lbl = self._type_label(self.key)
        seen, seen_id = [], set()
        for nk in self._real_nd_keys_ordered():
            ckey = self._type_key_for_nd(nk)
            if ckey is None or self._type_label(ckey) != lbl:
                continue
            sess = str(ckey.session)
            ptr = self.cd.data.get(ckey)
            valid = self._all_inds_set_for_ptr(ptr)
            if any((int(a), int(b)) in valid
                   for a, b in self._group_pairs(gname, session=sess)):
                nid = id(nk)
                if nid not in seen_id:
                    seen.append(nk)
                    seen_id.add(nid)
        return seen

    def _any_iter_pairs_for_group(self, gname: str):
        """Yield ``(ckey, r, t)`` for *gname* by scanning sessions (expanded tag only)."""
        if gname not in self._any_expanded_group_tags:
            return
        lbl = self._type_label(self.key)
        dead = self.deleted_inds
        for nk in self._real_nd_keys_ordered():
            ckey = self._type_key_for_nd(nk)
            if ckey is None or self._type_label(ckey) != lbl:
                continue
            sess = str(ckey.session)
            ptr = self.cd.data.get(ckey)
            valid = self._all_inds_set_for_ptr(ptr)
            pairs = self._group_pairs(gname, session=sess)
            if not pairs:
                continue
            for r, t in sorted((int(a), int(b)) for a, b in pairs):
                if r == t:
                    continue
                if (r, t) not in valid:
                    continue
                if (sess, r, t) in dead:
                    continue
                yield ckey, r, t

    def _any_rebuild_pair_handles(self):
        """Rebuild ``_any_pair_handle_list`` in tag header order × session order."""
        if not getattr(self, '_session_any_mode', False):
            self._any_pair_handle_list = []
            return
        handles: list[tuple] = []
        for gname in self._any_group_header_names():
            handles.extend(self._any_iter_pairs_for_group(gname))
        self._any_pair_handle_list = handles

    def _any_sync_selection_from_universe(self):
        """Any mode: all pairs in expanded tags are selected; Available stays empty."""
        hl = getattr(self, '_any_pair_handle_list', None) or []
        self.selected_inds = {
            (str(ckey.session), int(r), int(t)) for ckey, r, t in hl
        }
        self.unselected_inds = set()

    def _any_load_deleted_aggregate(self):
        lbl = self._type_label(self.key)
        deleted: set = set()
        for k in self.cd.data.keys():
            if self._type_label(k) != lbl:
                continue
            ptr = self.cd.data.get(k)
            valid = self._all_inds_set_for_ptr(ptr)
            raw = set(self._pair_deleted_store.get(str(k), set())) & valid
            sess = str(k.session)
            for r, c in raw:
                deleted.add((sess, int(r), int(c)))
        self.deleted_inds = deleted

    def _flush_any_selections_to_pointers(self):
        if not getattr(self, '_session_any_mode', False):
            return
        lbl = self._type_label(self.key)
        by_sess: dict[str, list[tuple[int, int]]] = _defaultdict(list)
        for trip in self.selected_inds:
            by_sess[trip[0]].append((int(trip[1]), int(trip[2])))
        for k in self.cd.data.keys():
            if self._type_label(k) != lbl:
                continue
            sess = str(k.session)
            arr = by_sess.get(sess)
            ptr = self.cd.data[k]
            ptr.manually_selected_inds = (
                np.array(sorted(arr), dtype=int) if arr else None)

    def _flush_any_deleted_to_stores(self):
        if not getattr(self, '_session_any_mode', False):
            return
        lbl = self._type_label(self.key)
        by_key: dict[str, set] = _defaultdict(set)
        for trip in self.deleted_inds:
            s, r, t = trip[0], int(trip[1]), int(trip[2])
            for k in self.cd.data.keys():
                if self._type_label(k) != lbl or str(k.session) != s:
                    continue
                by_key[str(k)].add((r, t))
        for ks, pairs in by_key.items():
            self._pair_deleted_store[ks] = set(pairs)

    def _enter_all_session_mode(self):
        """Switch UI to virtual ``All`` session (collapsed group tags; lazy expand)."""
        self._session_all_mode = True
        self._any_expanded_group_tags = set()
        self._any_pair_handle_list = []
        self._png_sess_slug = ''
        prev_lbl = self._type_label(self.key)
        self._type_keys_list = self._available_type_keys_any()
        type_labels = [self._type_label(k) for k in self._type_keys_list]
        self._type_combo['values'] = type_labels
        if not self._type_keys_list:
            messagebox.showwarning("All sessions", "No connection types in dataset.")
            self._session_all_mode = False
            try:
                self._session_var.set(self._session_label(self.key.nd()))
            except Exception:
                pass
            return
        if prev_lbl in type_labels:
            self.key = self._type_keys_list[type_labels.index(prev_lbl)]
        else:
            self.key = self._type_keys_list[0]
            self._type_var.set(type_labels[0])
        self._bind_context_to_type_key(self.key)
        self._any_load_deleted_aggregate()
        self.current_pair_idx = 0
        self.current_segment = self.n_segments
        self.segment_combo['values'] = self.segment_names + [_ALL_SEGS]
        self.segment_var.set(_ALL_SEGS)
        self._bind_custom_segments_to_session(str(self.key.session))

    def _enter_any_session_mode(self):
        """Backward-compat wrapper for older call sites."""
        self._enter_all_session_mode()

    def _exit_all_session_mode(self):
        """Leave ``All`` mode (flush pointers/stores first if entering from multi-save)."""
        self._flush_any_selections_to_pointers()
        self._flush_any_deleted_to_stores()
        self._session_all_mode = False
        self._any_expanded_group_tags = set()
        self._any_pair_handle_list = []
        self._png_sess_slug = ''
        # ``selected_inds`` was (session, ref, tgt) triples; single-session code expects
        # (ref, tgt) only — reload from the bound pointer before _switch_key / autosave.
        try:
            ptr = self.cd.data.get(self.key) if getattr(self, 'key', None) is not None else None
            if ptr is not None and getattr(ptr, 'manually_selected_inds', None) is not None:
                self.selected_inds = set(map(tuple, ptr.manually_selected_inds))
            else:
                self.selected_inds = set()
            _avail = set(map(tuple, self.all_inds))
            self.deleted_inds = (
                set(self._pair_deleted_store.get(str(self.key), set())) & _avail)
            self.unselected_inds = _avail - self.selected_inds - self.deleted_inds
        except Exception:
            pass
        try:
            self._bind_custom_segments_to_session(str(self.key.session))
        except Exception:
            pass

    def _exit_any_session_mode(self):
        """Backward-compat wrapper for older call sites."""
        self._exit_all_session_mode()

    def _bind_context_to_type_key(self, tk):
        """Point ccg_pointer / ccg_data / neurons at *tk* without touching triple sets."""
        ptr = self.cd.data.get(tk)
        nd_key = tk.nd()
        self.key = tk
        self.ccg_pointer = ptr
        if ptr is None:
            self.ccg_data = None
            self.neurons = None
            self.n_segments = 0
            self.segment_names = []
            return
        if (getattr(self, '_highres_mode', False)
                and hasattr(self.cd, '_ccg_highres')
                and self.cd._ccg_highres.get(nd_key) is not None):
            self.ccg_data = self.cd._ccg_highres[nd_key]
        else:
            self.ccg_data = self.cd._ccg.get(nd_key)
        self.neurons = (self.cd.nd.data[nd_key]
                        if getattr(self.cd, 'nd', None) is not None else None)
        self.n_segments = ptr.n_segments
        self.segment_names = list(ptr.edge_times['label'].values)

    def _autosave_all_sessions_for_current_type(self):
        """Write selection JSON 'latest' once per physical session (any-mode)."""
        self._flush_any_selections_to_pointers()
        self._flush_any_deleted_to_stores()
        lbl = self._type_label(self.key)
        saved_sess: set[str] = set()
        old_key = self.key
        old_ptr = self.ccg_pointer
        old_cd = self.ccg_data
        old_neurons = self.neurons
        old_ns = self.n_segments
        old_sn = list(self.segment_names) if self.segment_names else []
        try:
            for nk in self._real_nd_keys_ordered():
                ckey = self._type_key_for_nd(nk)
                if ckey is None or self._type_label(ckey) != lbl:
                    continue
                sess = str(ckey.session)
                if sess in saved_sess:
                    continue
                saved_sess.add(sess)
                self._bind_context_to_type_key(ckey)
                try:
                    self._save_selection_version('latest')
                except Exception as exc:
                    print(f"[CCGReviewUI] any-session save failed for {sess}: {exc}")
        finally:
            self.key = old_key
            self.ccg_pointer = old_ptr
            self.ccg_data = old_cd
            self.neurons = old_neurons
            self.n_segments = old_ns
            self.segment_names = old_sn
            # The save loop binds every session; restoring ``old_*`` can leave
            # ``ccg_*`` on session A while ``current_pair_idx`` still points at
            # a pair row for session B → IndexError in ``_resolve_segment_data``.
            if getattr(self, '_session_any_mode', False):
                idx = self.current_pair_idx
                hl = getattr(self, '_any_pair_handle_list', None) or []
                if (getattr(self, '_focused_pair', None) is None
                        and 0 <= idx < len(hl)):
                    self._sync_any_plot_context(idx)

    def _sync_any_plot_context(self, row_idx: int):
        """Bind ``ccg_*`` to the ``Key`` for ``_any_pair_handle_list[row_idx]``."""
        if not getattr(self, '_session_any_mode', False):
            return
        hl = getattr(self, '_any_pair_handle_list', None) or []
        if row_idx < 0 or row_idx >= len(hl):
            return
        ckey, _r, _t = hl[row_idx]
        sess = str(ckey.session)
        self._png_sess_slug = self._sanitize_sess_slug(sess)
        prev_sess = str(getattr(self.key, 'session', '') or '')
        if (self.key == ckey and self.ccg_data is not None
                and getattr(self.ccg_pointer, 'inds2', None) is not None):
            self._bind_custom_segments_to_session(sess)
            self._clamp_current_segment_for_session()
            try:
                self._update_segment_label()
            except Exception:
                pass
            return
        if prev_sess != sess:
            # Custom CCG data are session-specific: show that session's segment chips.
            self._bind_custom_segments_to_session(sess)
        self._bind_context_to_type_key(ckey)
        self._clamp_current_segment_for_session()
        try:
            if getattr(self, 'segment_combo', None) is not None:
                self.segment_combo['values'] = self.segment_names + [_ALL_SEGS]
            self._build_sig_chips()
            self._load_jitter_from_cd()
            self._update_segment_label()
        except Exception:
            pass

    def _toggle_any_avail_group(self, gname: str):
        """Expand/collapse a group tag (Any mode); load CCG for involved sessions."""
        if gname in self._any_expanded_group_tags:
            self._any_expanded_group_tags.discard(gname)
            self.refresh_lists()
            return

        nds = self._any_nd_keys_for_group(gname)

        def _finish_expand():
            self._any_expanded_group_tags.add(gname)
            self.refresh_lists()

        if not nds:
            _finish_expand()
            return

        def _chain(idx: int):
            if idx >= len(nds):
                _finish_expand()
                return
            self._ensure_session_loaded(nds[idx], on_loaded=lambda: _chain(idx + 1))

        _chain(0)

    def _pair_sess_rt(self, inds) -> tuple[str, tuple[int, int]]:
        """Session string + (ref,tgt) for group/tag lookups."""
        if getattr(self, '_session_any_mode', False):
            if isinstance(inds[0], Key):
                return str(inds[0].session), (int(inds[1]), int(inds[2]))
            return str(inds[0]), (int(inds[1]), int(inds[2]))
        return self._current_session_str(), (int(inds[0]), int(inds[1]))

    def _pair_row_selected_trip(self, inds):
        """Selected-list row → ``selected_inds`` key (sess, r, t) in any-mode."""
        if (getattr(self, '_session_any_mode', False) and isinstance(inds, tuple)
                and len(inds) >= 3 and isinstance(inds[0], Key)):
            return (str(inds[0].session), int(inds[1]), int(inds[2]))
        return inds

    def _pair_in_group(self, pair, group_name: str) -> bool:
        sess, p2 = self._pair_sess_rt(pair)
        return p2 in self._group_pairs(group_name, session=sess)

    def _available_type_keys(self, nd_key) -> list:
        if nd_key is _ALL_SESSION_MARKER:
            return self._available_type_keys_any()
        nd_session = nd_key.session
        return [k for k in self.cd.data.keys() if k.nd().session == nd_session]

    def _type_label(self, key) -> str:
        parts = []
        if getattr(key, 'excitability', None):
            parts.append(key.excitability)
        if getattr(key, 'conn_type', None):
            ref, tgt = key.conn_type
            _map = {'pyr': 'PYR', 'inter': 'INT', 'int': 'INT'}
            ref_lbl = _map.get(str(ref).lower(), str(ref).upper())
            tgt_lbl = _map.get(str(tgt).lower(), str(tgt).upper())
            parts.append(f"{ref_lbl}→{tgt_lbl}")
        if getattr(key, 'epoch', None):
            parts.append(f"[{key.epoch}]")
        return ' '.join(parts) if parts else str(key)

    def _switch_key(self, new_key) -> bool:
        if getattr(self, '_session_any_mode', False):
            return False
        # Persist in-session selections to the current pointer before switching,
        # so they survive type/session changes and can be restored on return.
        prev_key = self.key
        prev_session = getattr(prev_key, 'session', None) if prev_key is not None else None

        if self.ccg_pointer is not None:
            self.ccg_pointer.manually_selected_inds = (
                np.array(sorted(self.selected_inds), dtype=int)
                if self.selected_inds else None
            )

        ptr = self.cd.data.get(new_key)
        if ptr is None or ptr.inds is None:
            messagebox.showwarning("Switch key", f"No data for key:\n{new_key}")
            return False
        # Persist deleted pairs for the key we are leaving (per conn type)
        if prev_key is not None:
            self._pair_deleted_store[str(prev_key)] = set(self.deleted_inds)

        self.key = new_key
        self.ccg_pointer = ptr
        nd_key = new_key.nd()
        if (getattr(self, '_highres_mode', False)
                and hasattr(self.cd, '_ccg_highres')
                and self.cd._ccg_highres.get(nd_key) is not None):
            self.ccg_data = self.cd._ccg_highres[nd_key]
        else:
            self._highres_mode = False   # reset if highres not available
            self.ccg_data = self.cd._ccg.get(nd_key)
        self.neurons = (self.cd.nd.data[new_key.nd()]
                        if getattr(self.cd, 'nd', None) is not None else None)
        # all_inds is a @property — no assignment needed
        self.n_segments = self.ccg_pointer.n_segments
        self.segment_names = list(self.ccg_pointer.edge_times['label'].values)
        self.current_segment = 0
        self.current_pair_idx = 0
        if (hasattr(self.ccg_pointer, 'manually_selected_inds') and
                self.ccg_pointer.manually_selected_inds is not None):
            self.selected_inds = set(
                map(tuple, self.ccg_pointer.manually_selected_inds))
        else:
            self.selected_inds = set()
        _avail = set(map(tuple, self.all_inds))
        self.deleted_inds = (
            set(self._pair_deleted_store.get(str(new_key), set())) & _avail)
        self.unselected_inds = _avail - self.selected_inds - self.deleted_inds
        self.active_norms = set()
        for var in self.norm_vars.values():
            var.set(False)
        self.active_segment_filter = None
        # Clear custom segments only when the session changes; retain them across type switches
        new_session = getattr(new_key, 'session', None)
        self._switch_key_session_changed = (
            str(prev_session or '') != str(new_session or ''))
        if self._switch_key_session_changed:
            self._custom_segments.clear()
            self._bind_custom_segments_to_session(str(new_session))
        # Update probe network ct checkboxes to match new key
        new_ct = getattr(new_key, 'conn_type', None)
        for ct, var in getattr(self, '_net_ct_vars', {}).items():
            var.set(ct == new_ct)
        return True

    def _refresh_after_key_switch(self):
        self.segment_combo['values'] = self.segment_names + [_ALL_SEGS]
        self.segment_var.set(self.segment_names[0])
        self._build_sig_chips()
        self.refresh_lists()
        self.plot_title_var.set(self.get_plot_title())
        self._load_jitter_from_cd()
        self.update_plot()
        if self._focused_neuron is not None:
            self._update_focus_info(self._focused_neuron)
        self._refresh_net_shank_buttons()
        self._draw_network()
        if self._auto_pregen_enabled and self._cache_config is not None:
            self._launch_pregen_subprocess(dict(self._cache_config), priority='auto')
        _sess_ch = getattr(self, '_switch_key_session_changed', False)
        if _sess_ch:
            self._ts_refresh_epochs_for_current_key()
        else:
            self._ts_reinit_times_for_current_key()
            self._ts_refresh_union_if_all_sessions_mode()
        self._switch_key_session_changed = False

    def _ts_reinit_times_for_current_key(self):
        """Refresh time-slider bounds from the current CCG pointer / theme (same session)."""
        if getattr(self, '_ts_theme_combo', None) is None:
            return
        self._ts_init_times()
        self._ts_redraw()

    def _custom_ccg_has_unsaved(self) -> bool:
        """True if any in-memory custom segment has no on-disk .npz (or file missing)."""
        buckets = getattr(self, '_custom_segments_by_session', None) or {}
        seq = buckets.values() if buckets else [getattr(self, '_custom_segments', [])]
        for lst in seq:
            for cs in lst:
                if not isinstance(cs, dict):
                    continue
                p = cs.get('src_path')
                if not p or not os.path.isfile(str(p)):
                    return True
        return False

    def _maybe_prompt_save_custom_ccgs_before_session_switch(self) -> bool:
        """Return True to proceed switching session; False to cancel (revert session combo)."""
        if not self._custom_ccg_has_unsaved():
            return True
        r = messagebox.askyesnocancel(
            "Unsaved custom CCGs",
            "Custom CCG segments belong only to the current session. They are not valid "
            "after you switch sessions.\n\n"
            "You have one or more custom segments that are not saved to a .npz file.\n\n"
            "Save them now?\n"
            "  Yes — open the save dialog\n"
            "  No — discard those segments and switch\n"
            "  Cancel — stay on this session")
        if r is None:
            return False
        if r:
            self._ts_save_custom_ccg()
            if self._custom_ccg_has_unsaved():
                messagebox.showwarning(
                    "Custom CCGs not saved",
                    "Some custom segments are still unsaved. Session switch was cancelled.")
                return False
            return True
        return True

    def _ts_refresh_epochs_for_current_key(self):
        """Re-discover behavioral Epoch objects for the current session and refresh bounds."""
        combo = getattr(self, '_ts_theme_combo', None)
        if combo is None:
            return
        self._ts_discover_themes()
        vals = tuple(combo.cget('values') or ())
        if vals and self._ts_theme_var.get() not in vals:
            self._ts_theme_var.set('segments')
        self._on_ts_theme_change()

    # ------------------------------------------------------------------
    # Dropdown callbacks
    # ------------------------------------------------------------------

    def _on_session_change(self, event):
        idx = self._session_combo.current()
        if idx < 0 or idx >= len(self._nd_keys_list):
            return
        nd_key = self._nd_keys_list[idx]
        cur_any = getattr(self, '_session_any_mode', False)
        new_any = nd_key is _ALL_SESSION_MARKER

        if new_any and cur_any:
            return

        if not new_any and not cur_any:
            cur_nd = self.key.nd()
            cur_sess = getattr(cur_nd, 'session', None)
            new_sess = getattr(nd_key, 'session', None)
            if (cur_sess is not None and new_sess is not None
                    and str(cur_sess) == str(new_sess)):
                return

        self._autosave_current()
        if not self._maybe_prompt_save_custom_ccgs_before_session_switch():
            cur_nd = self.key.nd()
            self._session_var.set(
                self._session_label(_ALL_SESSION_MARKER) if cur_any
                else self._session_label(cur_nd))
            return
        try:
            self._stacked_segments.clear()
        except Exception:
            self._stacked_segments = set()

        if new_any:
            def _do_enter_any():
                self._enter_all_session_mode()
                if not getattr(self, '_session_all_mode', False):
                    self._session_var.set(self._session_label(self.key.nd()))
                    return
                self._refresh_after_key_switch()
                self.refresh_lists()
                self._draw_network()

            _do_enter_any()
            return

        # Leaving ``any`` → concrete session
        if cur_any:
            try:
                self._autosave_all_sessions_for_current_type()
            except Exception as exc:
                print(f"[CCGReviewUI] multi-session autosave: {exc}")
            self._exit_all_session_mode()

        def _do_switch():
            type_keys = self._available_type_keys(nd_key)
            self._type_keys_list = type_keys
            type_labels = [self._type_label(k) for k in type_keys]
            self._type_combo['values'] = type_labels
            if not type_keys:
                return
            current_lbl = self._type_label(self.key)
            if current_lbl in type_labels:
                new_key = type_keys[type_labels.index(current_lbl)]
            else:
                new_key = type_keys[0]
                self._type_var.set(type_labels[0])
            if self._switch_key(new_key):
                self._refresh_after_key_switch()
                self._autoload_session_latest()
                self.refresh_lists()
                self._draw_network()

        self._ensure_session_loaded(nd_key, on_loaded=_do_switch)

    def _switch_type_any(self, new_key):
        """Change connection type while in virtual ``any`` session."""
        self._flush_any_selections_to_pointers()
        try:
            self._autosave_all_sessions_for_current_type()
        except Exception as exc:
            print(f"[CCGReviewUI] any-session type switch autosave: {exc}")
        self._bind_context_to_type_key(new_key)
        self._type_var.set(self._type_label(new_key))
        self._any_load_deleted_aggregate()
        self._any_expanded_group_tags = set()
        self._refresh_after_key_switch()
        self.refresh_lists()
        self._draw_network()

    def _on_type_change(self, event):
        idx = self._type_combo.current()
        if idx < 0 or idx >= len(self._type_keys_list):
            return
        new_key = self._type_keys_list[idx]
        if getattr(self, '_session_any_mode', False):
            if str(new_key) == str(self.key):
                return
            self._switch_type_any(new_key)
            return
        if self._switch_key(new_key):
            self._refresh_after_key_switch()

    def _on_segment_change(self, event):
        name = self.segment_var.get()
        if name == _ALL_SEGS:
            self.current_segment = self.n_segments
        elif name in self.segment_names:
            self.current_segment = self.segment_names.index(name)
        self.plot_title_var.set(self.get_plot_title())
        self._exit_spike_attribution_view()
        self.update_plot()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _compute_stats_str(self) -> str:
        n_sig = len(self.all_inds)
        n_sel = len(self.selected_inds)
        n_poss = None
        n_ref = None
        n_tgt = None
        ref_t = tgt_t = None
        if (self.neurons is not None and
                self.key is not None and
                self.key.conn_type is not None):
            ref_t, tgt_t = self.key.conn_type
            try:
                n_ref = self.neurons.get_neuron_type(ref_t).n_neurons
                n_tgt = self.neurons.get_neuron_type(tgt_t).n_neurons
                # Exclude self-self when ref and target types are the same
                n_poss = n_ref * (n_ref - 1) if ref_t == tgt_t else n_ref * n_tgt
            except Exception:
                pass

        s = f"Sig: {n_sig:4d}  Sel: {n_sel:4d}"
        # Show per-type neuron counts
        if n_ref is not None:
            if ref_t == tgt_t:
                s += f"  {ref_t}: {n_ref}"
            else:
                s += f"  {ref_t}: {n_ref}  {tgt_t}: {n_tgt}"
        if n_poss is not None and n_poss > 0:
            sel_over_sig  = n_sel / max(n_sig,  1)
            sel_over_poss = n_sel / n_poss
            sig_over_poss = n_sig / n_poss
            s += (f"  Poss: {n_poss:4d}"
                  f"  Sel/Sig: {sel_over_sig:.0%}"
                  f"  Sel/Poss: {sel_over_poss:.0%}"
                  f"  Sig/Poss: {sig_over_poss:.0%}")
        return s

    def _refresh_stats(self):
        if hasattr(self, 'stats_var'):
            self.stats_var.set(self._compute_stats_str())

    def _update_pair_info(self, inds):
        """Update per-pair info (firing rates) in the stats bar."""
        if not hasattr(self, '_pair_info_var'):
            return
        if self.neurons is None:
            self._pair_info_var.set("")
            return
        ref, tgt = int(inds[0]), int(inds[1])
        fr = getattr(self.neurons, 'firing_rate', None)
        if fr is None:
            self._pair_info_var.set("")
            return
        ref_fr = float(fr[ref])
        tgt_fr = float(fr[tgt])

        # Segment-level firing rate
        seg_str = ""
        seg = self.current_segment
        nd_key = self.key.nd() if self.key else None
        if self._is_custom_segment(seg):
            ci = self._custom_seg_index(seg)
            cs = self._custom_segments[ci]
            cs_fr = cs.get('firing_rates')
            if cs_fr is not None and ref < len(cs_fr) and tgt < len(cs_fr):
                sr = float(cs_fr[ref])
                st = float(cs_fr[tgt])
                seg_str = f"  Seg FR: ref={sr:.1f}  tgt={st:.1f}"
        else:
            seg_frates = None
            if (nd_key is not None and self.cd.nd is not None
                    and seg != self.n_segments):
                seg_frates = self.cd.nd.segment_firing_rates.get(nd_key)
            if seg_frates is not None and seg < seg_frates.shape[0]:
                sr = float(seg_frates[seg, ref])
                st = float(seg_frates[seg, tgt])
                seg_str = f"  Seg FR: ref={sr:.1f}  tgt={st:.1f}"

        self._pair_info_var.set(
            f"FR: ref={ref_fr:.1f}Hz  tgt={tgt_fr:.1f}Hz{seg_str}")

    # ------------------------------------------------------------------
    # Undo / redo for pair selection
    # ------------------------------------------------------------------

    def _flush_deleted_to_store(self):
        """Copy current deleted_inds into _pair_deleted_store for the active Key."""
        if getattr(self, '_session_any_mode', False):
            self._flush_any_deleted_to_stores()
            return
        if getattr(self, 'key', None) is not None:
            self._pair_deleted_store[str(self.key)] = set(self.deleted_inds)

    def _push_undo(self):
        """Snapshot current selection AND group state before a mutation."""
        self._undo_stack.append((
            set(self.selected_inds),
            set(self.unselected_inds),
            set(self.deleted_inds),
            _copy.deepcopy(self._groups),
        ))
        if len(self._undo_stack) > self._UNDO_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    # Highlight color for undo/redo indicators (matches CCG baseline orange)
    _UNDO_HIGHLIGHT = '#ff7f0e'

    def _sync_sel_data(self):
        """Re-sync _sel_data from CCGReviewUI attrs after any bulk reassignment.

        Called at the top of refresh_lists (which runs after every state change)
        and explicitly after undo/redo and load operations that create new objects.
        """
        if not hasattr(self, '_sel_data'):
            return
        self._sel_data.selected_inds   = self.selected_inds
        self._sel_data.unselected_inds = self.unselected_inds
        self._sel_data._groups         = self._groups
        self._sel_data._pair_tags      = self._pair_tags
        self._sel_data._group_hotkeys  = self._group_hotkeys
        self._sel_data._group_notes    = self._group_notes
        self._sel_data._group_registry = self._group_registry
        self._sel_data._next_group_id  = self._next_group_id

    def _undo(self, event=None):
        if not self._undo_stack:
            return
        cur = (set(self.selected_inds), set(self.unselected_inds),
               set(self.deleted_inds), _copy.deepcopy(self._groups))
        self._redo_stack.append(cur)
        state = self._undo_stack.pop()
        self.selected_inds = state[0]
        self.unselected_inds = state[1]
        self.deleted_inds = state[2] if len(state) > 2 else set()
        if len(state) > 3:
            self._groups = state[3]
            self._sync_sel_data()  # re-sync before _rebuild reads _sel_data._groups
            self._rebuild_groups_menu()
        changed = (cur[0] ^ state[0]) | (cur[1] ^ state[1])
        self.refresh_lists()
        self._highlight_changed_pairs(changed)
        self._flush_deleted_to_store()
        self.update_plot()
        self._draw_network()
        self._refresh_stats()

    def _redo(self, event=None):
        if not self._redo_stack:
            return
        cur = (set(self.selected_inds), set(self.unselected_inds),
               set(self.deleted_inds), _copy.deepcopy(self._groups))
        self._undo_stack.append(cur)
        state = self._redo_stack.pop()
        self.selected_inds = state[0]
        self.unselected_inds = state[1]
        self.deleted_inds = state[2] if len(state) > 2 else set()
        if len(state) > 3:
            self._groups = state[3]
            self._sync_sel_data()
            self._rebuild_groups_menu()
        changed = (cur[0] ^ state[0]) | (cur[1] ^ state[1])
        self.refresh_lists()
        self._highlight_changed_pairs(changed)
        self._flush_deleted_to_store()
        self.update_plot()
        self._draw_network()
        self._refresh_stats()

    def _highlight_changed_pairs(self, changed_pairs):
        """Highlight pairs that moved during undo/redo with baseline color.

        The highlight clears on the next arbitrary click anywhere in the UI.
        """
        if not changed_pairs:
            return
        avail_map = getattr(self, '_avail_list_pairs', None)
        unsel_items = (((i, e[0]) for i, e in enumerate(avail_map) if e is not None)
                       if avail_map else enumerate(sorted(self.unselected_inds)))
        for idx, inds in unsel_items:
            if inds in changed_pairs:
                self.unselected_list.itemconfig(idx, background=self._UNDO_HIGHLIGHT,
                                                foreground='white')
        sel_map = getattr(self, '_sel_list_pairs', None)
        sel_items = (((i, e) for i, e in enumerate(sel_map) if e is not None)
                     if sel_map else enumerate(sorted(self.selected_inds)))
        for idx, inds in sel_items:
            if inds in changed_pairs:
                self.selected_list.itemconfig(idx, background=self._UNDO_HIGHLIGHT,
                                              foreground='white')
        # Clear highlight on next click anywhere
        def _clear_highlight(e=None):
            self._clear_undo_highlight()
            self.root.unbind('<Button-1>', bind_id)
        bind_id = self.root.bind('<Button-1>', _clear_highlight, add='+')

    def _clear_undo_highlight(self):
        """Remove undo/redo highlight from all list items."""
        for listbox in (self.unselected_list, self.selected_list):
            for idx in range(listbox.size()):
                listbox.itemconfig(idx, background='', foreground='')

    # ------------------------------------------------------------------
    # Pair lists
    # ------------------------------------------------------------------

    def _on_sort_by_group_toggle(self):
        if self._sort_selected_var.get():
            self._sort_by_tag_var.set(False)
            self._sort_by_mean_var.set(False)
            self._sort_by_min_p_var.set(False)
        self.refresh_lists()

    def _on_sort_by_tag_toggle(self):
        if self._sort_by_tag_var.get():
            self._sort_selected_var.set(False)
            self._sort_by_mean_var.set(False)
            self._sort_by_min_p_var.set(False)
        self.refresh_lists()

    def _on_sort_by_mean_toggle(self):
        if self._sort_by_mean_var.get():
            self._sort_selected_var.set(False)
            self._sort_by_tag_var.set(False)
            self._sort_by_min_p_var.set(False)
        self.refresh_lists()

    def _on_sort_by_min_p_toggle(self):
        if self._sort_by_min_p_var.get():
            self._sort_selected_var.set(False)
            self._sort_by_tag_var.set(False)
            self._sort_by_mean_var.set(False)
        self.refresh_lists()

    def _pair_mean_ccg(self, inds):
        """Return the mean CCG value for a pair at the current segment."""
        if self.ccg_data is None:
            return 0.0
        ref, tgt = int(inds[0]), int(inds[1])
        try:
            seg = min(self.current_segment, self.ccg_data.ccg.shape[0] - 1)
            return float(np.mean(self.ccg_data.ccg[seg, ref, tgt, :]))
        except (IndexError, KeyError):
            return 0.0

    def _pair_min_pval(self, inds):
        """Minimum EranConv p-value in the test-window bins at the current segment.

        Sort key: smaller = more significant. Pairs without p-value data sort as 1.0.
        """
        if self.ccg_data is None or self.ccg_data.pval is None:
            return 1.0
        ref, tgt = int(inds[0]), int(inds[1])
        try:
            seg = min(self.current_segment, self.ccg_data.pval.shape[0] - 1)
            arr = self.ccg_data.pval[seg, ref, tgt, :]
            conf = self.ccg_data.conf
            lo = int(conf.min_lag_bin)
            hi = int(conf.max_lag_bin)
            sl = arr[lo:hi]
            if sl.size == 0:
                return 1.0
            m = float(np.nanmin(sl))
            if not np.isfinite(m):
                return 1.0
            return m
        except (IndexError, KeyError, TypeError, ValueError):
            return 1.0

    def refresh_lists(self):
        # Sync SelectionData from CCGReviewUI attrs, then delegate to LeftPanel.
        # All list-rebuild logic lives in LeftPanel.refresh_lists(); keeping this
        # method on CCGReviewUI so the 20+ call-sites here don't need to change.
        self._sync_sel_data()
        if hasattr(self, 'left_container'):
            self.left_container.left_panel.refresh_lists()
            return
        # ── Pre-container fallback (should not be reached after setup_ui) ──
        try:
            _unsel_scroll_top = self.unselected_list.yview()[0]
        except Exception:
            _unsel_scroll_top = None
        try:
            _sel_scroll_top = self.selected_list.yview()[0]
        except Exception:
            _sel_scroll_top = None

        self.unselected_list.delete(0, tk.END)
        self.selected_list.delete(0, tk.END)

        if getattr(self, '_session_any_mode', False):
            self._any_rebuild_pair_handles()
            self._any_sync_selection_from_universe()

        # Build gray-out set based on active focus mode
        gray_out = None
        fn = self._focused_neuron
        fp = self._focused_pair
        if fn is not None:
            focus_connected = {(ref, tgt) for ref, tgt in
                               map(tuple, self.all_inds)
                               if ref == fn or tgt == fn}
            gray_out = lambda inds: inds not in focus_connected
        elif fp is not None:
            gray_out = lambda inds: inds != fp

        # Same-channel / same-shank graying
        hide_same_channel = (hasattr(self, 'network_panel')
                             and self.network_panel._net_hide_same_channel_var.get())
        hide_same_shank = (hasattr(self, 'network_panel')
                           and self.network_panel._net_hide_same_shank_var.get())
        peak_channels = (getattr(self.neurons, 'peak_channels', None)
                         if self.neurons is not None else None)
        shank_ids = (getattr(self.neurons, 'shank_ids', None)
                     if self.neurons is not None else None)

        def _should_gray(inds):
            if getattr(self, '_session_any_mode', False):
                ref_i, tgt_i = int(inds[1]), int(inds[2])
                inds2 = (ref_i, tgt_i)
            else:
                ref_i, tgt_i = int(inds[0]), int(inds[1])
                inds2 = (ref_i, tgt_i)
            if gray_out is not None and gray_out(inds2):
                return True
            if hide_same_shank and shank_ids is not None:
                if int(shank_ids[ref_i]) == int(shank_ids[tgt_i]):
                    return True
            elif hide_same_channel and peak_channels is not None:
                if int(peak_channels[ref_i]) == int(peak_channels[tgt_i]):
                    return True
            return False

        def _spec_group_sort_key(gname):
            """Sort group names numerically then alphabetically."""
            try:
                return (0, int(gname), '')
            except (ValueError, TypeError):
                return (1, 0, gname)

        speculated = getattr(self, '_speculated_groups', {})
        if getattr(self, '_session_any_mode', False):
            speculated = {}
        # Parallel to unselected_list items: None = header/separator, (inds, pred_group) = pair
        self._avail_list_pairs = []   # parallel to unselected_list items; None = header/separator

        # ── Predicted group sections (pairs over threshold) — shown first ─
        if speculated:
            spec_in_avail = {p: r for p, r in speculated.items()
                             if p in self.unselected_inds}
            if spec_in_avail:
                buckets = _defaultdict(list)
                for pair, r in spec_in_avail.items():
                    if r.action == 'assign' and r.groups:
                        for gname in r.groups:
                            buckets[gname].append(pair)

                if buckets:
                    sorted_gnames = sorted(buckets.keys(), key=_spec_group_sort_key)
                    for gname in sorted_gnames:
                        pairs_in_g = sorted(set(buckets[gname]))
                        hdr_idx = self.unselected_list.size()
                        self.unselected_list.insert(tk.END,
                            f"── predicted: {gname} ({len(pairs_in_g)}) ──")
                        self.unselected_list.itemconfig(
                            hdr_idx, foreground='#555555',
                            selectforeground='#555555')
                        self._avail_list_pairs.append(None)
                        for inds in pairs_in_g:
                            label = f"{self._bookmark_label_prefix(inds)}[{inds[0]}, {inds[1]}]"
                            existing_grp = self._pair_group_label(inds)
                            if existing_grp:
                                label += f" {existing_grp}"
                            self.unselected_list.insert(tk.END, label)
                            self._avail_list_pairs.append((inds, gname))
                            if _should_gray(inds):
                                item_idx = self.unselected_list.size() - 1
                                self.unselected_list.itemconfig(
                                    item_idx, foreground='#AAAAAA')

        # ── Divider between predicted and normal sections ────────────────
        if speculated and self.unselected_inds:
            spec_in_avail_any = any(
                r.action == 'assign' and r.groups
                for r in speculated.values()
                if r.pair in self.unselected_inds
            )
            if spec_in_avail_any:
                div_idx = self.unselected_list.size()
                self.unselected_list.insert(tk.END, "─" * 18)
                self.unselected_list.itemconfig(
                    div_idx, foreground='#CCCCCC',
                    selectforeground='#CCCCCC',
                    selectbackground='#F5F5F5')
                self._avail_list_pairs.append(None)

        # ── All unselected pairs (normal list) ───────────────────────────
        _sort_mean_active = (getattr(self, '_sort_by_mean_var', None)
                             and self._sort_by_mean_var.get()
                             and not getattr(self, '_session_any_mode', False))
        _sort_minp_active = (getattr(self, '_sort_by_min_p_var', None)
                             and self._sort_by_min_p_var.get()
                             and not getattr(self, '_session_any_mode', False))
        if getattr(self, '_session_any_mode', False):
            # Available intentionally empty — pairs live under tags in Selected only.
            pass
        elif _sort_mean_active:
            _unsorted_avail = sorted(self.unselected_inds,
                                     key=self._pair_mean_ccg, reverse=True)
            for inds in _unsorted_avail:
                ref_i, tgt_i = int(inds[0]), int(inds[1])
                if ref_i == tgt_i:
                    continue
                label = f"{self._bookmark_label_prefix(inds)}[{inds[0]}, {inds[1]}]"
                grp = self._pair_group_label(inds)
                if grp:
                    label += f" {grp}"
                self.unselected_list.insert(tk.END, label)
                self._avail_list_pairs.append((inds, None))
                item_idx = self.unselected_list.size() - 1
                if self._main_template is not None and not self._pair_passes_main(ref_i, tgt_i, self.current_segment):
                    self.unselected_list.itemconfig(item_idx, foreground='gray')
                elif _should_gray(inds):
                    self.unselected_list.itemconfig(item_idx, foreground='#AAAAAA')
        elif _sort_minp_active:
            _unsorted_avail = sorted(self.unselected_inds,
                                     key=self._pair_min_pval)
            for inds in _unsorted_avail:
                ref_i, tgt_i = int(inds[0]), int(inds[1])
                if ref_i == tgt_i:
                    continue
                label = f"{self._bookmark_label_prefix(inds)}[{inds[0]}, {inds[1]}]"
                grp = self._pair_group_label(inds)
                if grp:
                    label += f" {grp}"
                self.unselected_list.insert(tk.END, label)
                self._avail_list_pairs.append((inds, None))
                item_idx = self.unselected_list.size() - 1
                if self._main_template is not None and not self._pair_passes_main(ref_i, tgt_i, self.current_segment):
                    self.unselected_list.itemconfig(item_idx, foreground='gray')
                elif _should_gray(inds):
                    self.unselected_list.itemconfig(item_idx, foreground='#AAAAAA')
        else:
            _unsorted_avail = sorted(self.unselected_inds)
            for inds in _unsorted_avail:
                ref_i, tgt_i = int(inds[0]), int(inds[1])
                if ref_i == tgt_i:
                    continue
                label = f"{self._bookmark_label_prefix(inds)}[{inds[0]}, {inds[1]}]"
                grp = self._pair_group_label(inds)
                if grp:
                    label += f" {grp}"
                self.unselected_list.insert(tk.END, label)
                self._avail_list_pairs.append((inds, None))
                item_idx = self.unselected_list.size() - 1
                if self._main_template is not None and not self._pair_passes_main(ref_i, tgt_i, self.current_segment):
                    self.unselected_list.itemconfig(item_idx, foreground='gray')
                elif _should_gray(inds):
                    self.unselected_list.itemconfig(item_idx, foreground='#AAAAAA')

        # ── Deleted (spurious) section at bottom of Available ──────────
        if self.deleted_inds and not getattr(self, '_session_any_mode', False):
            sep_idx = self.unselected_list.size()
            self.unselected_list.insert(tk.END, "── deleted ──")
            self.unselected_list.itemconfig(sep_idx, foreground='#999999',
                                            selectforeground='#999999',
                                            selectbackground='#E8E8E8')
            self._avail_list_pairs.append(None)
            for inds in sorted(self.deleted_inds):
                if getattr(self, '_session_any_mode', False):
                    label = (f"{self._bookmark_label_prefix(inds)}"
                             f"{inds[0]} [{inds[1]}, {inds[2]}]")
                else:
                    label = f"{self._bookmark_label_prefix(inds)}[{inds[0]}, {inds[1]}]"
                grp = self._pair_group_label(inds)
                if grp:
                    label += f" {grp}"
                self.unselected_list.insert(tk.END, label)
                idx = self.unselected_list.size() - 1
                self.unselected_list.itemconfig(idx, foreground='#BBBBBB',
                                                selectforeground='#BBBBBB',
                                                selectbackground='#F0F0F0')
                self._avail_list_pairs.append((inds, 'deleted'))

        # ── Selected list ────────────────────────────────────────────────
        self._sel_list_pairs = []       # parallel index: None = separator, inds = pair
        self._sel_list_header_keys = [] # parallel index: header text for headers, None for pairs

        def _pair_group_combo(inds):
            """Sorted tuple of non-internal group names this pair belongs to."""
            sess, pair = self._pair_sess_rt(inds)
            pair = tuple(int(x) for x in pair)
            return tuple(sorted(
                g for g in self._groups
                if not g.startswith('__')
                and pair in self._group_pairs(g, session=sess)
            ))

        def _insert_sel_pair(inds):
            if getattr(self, '_session_any_mode', False):
                sess_lbl = (str(inds[0].session) if isinstance(inds[0], Key)
                            else str(inds[0]))
                label = (f"{self._bookmark_label_prefix(inds)}"
                         f"{sess_lbl} [{inds[1]}, {inds[2]}]")
            else:
                label = f"{self._bookmark_label_prefix(inds)}[{inds[0]}, {inds[1]}]"
            grp = self._pair_group_label(inds)
            if grp:
                label += f" {grp}"
            self.selected_list.insert(tk.END, label)
            self._sel_list_pairs.append(inds)
            self._sel_list_header_keys.append(None)
            if _should_gray(inds):
                self.selected_list.itemconfig(
                    self.selected_list.size() - 1, foreground='#AAAAAA')

        _collapsed = getattr(self, '_sel_collapsed_headers', set())

        def _insert_sel_header(text, count):
            """Insert a group header row; returns True if the header is collapsed."""
            is_collapsed = text in _collapsed
            display = f"── {text} ({count}) ──" + (" >>" if is_collapsed else "")
            hdr_idx = self.selected_list.size()
            self.selected_list.insert(tk.END, display)
            self.selected_list.itemconfig(
                hdr_idx,
                foreground='#444444',
                selectforeground='#444444',
                background='#CCCCCC',
                selectbackground='#BBBBBB',
            )
            self._sel_list_pairs.append(None)
            self._sel_list_header_keys.append(text)
            return is_collapsed

        sort_group = (getattr(self, '_sort_selected_var', None)
                      and self._sort_selected_var.get()
                      and not getattr(self, '_session_any_mode', False))
        sort_tag   = (getattr(self, '_sort_by_tag_var', None)
                      and self._sort_by_tag_var.get()
                      and not getattr(self, '_session_any_mode', False))
        sort_mean  = (getattr(self, '_sort_by_mean_var', None)
                      and self._sort_by_mean_var.get()
                      and not getattr(self, '_session_any_mode', False))
        sort_minp  = (getattr(self, '_sort_by_min_p_var', None)
                      and self._sort_by_min_p_var.get()
                      and not getattr(self, '_session_any_mode', False))

        if getattr(self, '_session_any_mode', False):
            for gname in self._any_group_header_names():
                trips_g = self._any_triples_in_group(gname)
                n_tag = len(trips_g - self.deleted_inds)
                exp = gname in self._any_expanded_group_tags
                hdr = (f"── {gname} ({n_tag}) ──"
                       + (" >>" if not exp else ""))
                hdr_idx = self.selected_list.size()
                self.selected_list.insert(tk.END, hdr)
                self.selected_list.itemconfig(
                    hdr_idx,
                    foreground='#444444',
                    selectforeground='#444444',
                    background='#CCCCCC',
                    selectbackground='#BBBBBB',
                )
                self._sel_list_pairs.append(None)
                self._sel_list_header_keys.append(gname)
                if exp:
                    for row in self._any_iter_pairs_for_group(gname):
                        _insert_sel_pair(row)
        elif sort_mean:
            # Sort by mean CCG value (descending) at current segment
            for inds in sorted(self.selected_inds,
                               key=self._pair_mean_ccg, reverse=True):
                _insert_sel_pair(inds)

        elif sort_minp:
            # Sort by minimum p in test window (ascending — most significant first)
            for inds in sorted(self.selected_inds, key=self._pair_min_pval):
                _insert_sel_pair(inds)

        elif sort_group:
            buckets = _defaultdict(list)
            for inds in sorted(self.selected_inds):
                buckets[_pair_group_combo(inds)].append(inds)

            def _combo_sort_key(combo):
                return (1, []) if not combo else (0, list(combo))

            for combo in sorted(buckets.keys(), key=_combo_sort_key):
                pairs_in_combo = buckets[combo]
                hdr_text = ', '.join(combo) if combo else '(untagged)'
                collapsed = _insert_sel_header(hdr_text, len(pairs_in_combo))
                if not collapsed:
                    for inds in pairs_in_combo:
                        _insert_sel_pair(inds)

        elif sort_tag:
            # Each pair appears once under every tag it belongs to
            tag_buckets: dict = _defaultdict(list)  # tag_name -> [inds, ...]
            untagged = []
            non_internal = [g for g in self._groups
                            if not g.startswith('__') and not g.startswith(_SPECIAL_PREFIX)]
            for inds in sorted(self.selected_inds):
                _s, _p = self._pair_sess_rt(inds)
                _p = tuple(int(x) for x in _p)
                tags = [g for g in non_internal
                        if _p in self._group_pairs(g, session=_s)]
                if tags:
                    for t in tags:
                        tag_buckets[t].append(inds)
                else:
                    untagged.append(inds)
            for tag in sorted(tag_buckets.keys()):
                collapsed = _insert_sel_header(tag, len(tag_buckets[tag]))
                if not collapsed:
                    for inds in tag_buckets[tag]:
                        _insert_sel_pair(inds)
            if untagged:
                collapsed = _insert_sel_header('(untagged)', len(untagged))
                if not collapsed:
                    for inds in untagged:
                        _insert_sel_pair(inds)

        else:
            for inds in sorted(self.selected_inds):
                _insert_sel_pair(inds)

        n_spec = len({p for p in getattr(self, '_speculated_groups', {})
                      if p in self.unselected_inds})
        if getattr(self, '_session_any_mode', False):
            self._avail_label_var.set("Available (0) — Any mode")
        else:
            avail_parts = [f"Available ({len(self.unselected_inds)}"]
            if n_spec:
                avail_parts.append(f", {n_spec} predicted")
            if self.deleted_inds:
                avail_parts.append(f", {len(self.deleted_inds)} deleted")
            avail_parts.append(")")
            self._avail_label_var.set(''.join(avail_parts))
        # Show/hide the "✕ predictions" button
        if hasattr(self, '_clear_spec_btn'):
            if n_spec and not getattr(self, '_session_any_mode', False):
                self._clear_spec_btn.pack(side=tk.RIGHT, padx=2)
            else:
                self._clear_spec_btn.pack_forget()
        # Selected count should reflect current conn type
        try:
            if getattr(self, '_session_any_mode', False):
                self._sel_label_var.set(f"Selected ({len(self.selected_inds)})")
            else:
                sess = self._current_session_str()
                ct_lbl = self._conn_type_label(getattr(self.key, 'conn_type', None))
                n_ct = len(self._filter_pairs_to_conn_types(sess, self.selected_inds, {ct_lbl}))
                self._sel_label_var.set(f"Selected ({n_ct})")
        except Exception:
            self._sel_label_var.set(f"Selected ({len(self.selected_inds)})")
        if hasattr(self, '_select_all_btn'):
            if getattr(self, '_session_any_mode', False):
                self._select_all_btn.config(
                    text="Select All", state='disabled')
            else:
                self._select_all_btn.config(
                    text="Deselect All" if not self.unselected_inds else "Select All",
                    state='normal')
        self._apply_jitter_list_colors()
        self._refresh_stats()
        # Re-apply search highlights on top of any other coloring
        if getattr(self, '_search_var', None) and self._search_var.get():
            self._search_update()
        else:
            self._reapply_bookmark_list_styles()

        # Restore scroll positions
        try:
            if _unsel_scroll_top is not None:
                self.unselected_list.yview_moveto(_unsel_scroll_top)
        except Exception:
            pass
        try:
            if _sel_scroll_top is not None:
                self.selected_list.yview_moveto(_sel_scroll_top)
        except Exception:
            pass

    def move_to_selected(self, event=None):
        """Move the item under the cursor (double-click) or current pair (keyboard)."""
        if getattr(self, '_session_any_mode', False):
            return
        # Cancel any pending single-click deferred update
        if self._select_after is not None:
            self.root.after_cancel(self._select_after)
            self._select_after = None
        # Determine which item to move
        if event is not None:
            # Double-click: use position under cursor — independent of selection state
            idx = self.unselected_list.nearest(event.y)
        else:
            # Keyboard / context menu: use currently highlighted item
            sel = self.unselected_list.curselection()
            idx = sel[-1] if sel else None
        if idx is None or idx < 0:
            return
        avail_map = getattr(self, '_avail_list_pairs', None)
        if avail_map is not None:
            if idx >= len(avail_map) or avail_map[idx] is None:
                return
            entry = avail_map[idx]
            inds, pred_group = entry
        else:
            sorted_unsel = sorted(self.unselected_inds)
            if idx >= len(sorted_unsel):
                return
            inds, pred_group = sorted_unsel[idx], None
        # Preserve scroll position so the list doesn't jump to the top
        scroll_top = self.unselected_list.yview()[0]
        if inds in self.selected_inds:
            # Already selected — just attach the predicted group tag if any
            if pred_group is not None:
                if getattr(self, '_session_any_mode', False):
                    self._group_add_pair(
                        pred_group, (inds[1], inds[2]), session=inds[0])
                else:
                    self._group_add_pair(pred_group, inds)
                self.refresh_lists()
            return
        self._push_undo()
        self.unselected_inds.discard(inds)
        self.selected_inds.add(inds)
        if pred_group is not None:
            if getattr(self, '_session_any_mode', False):
                self._group_add_pair(
                    pred_group, (inds[1], inds[2]), session=inds[0])
            else:
                self._group_add_pair(pred_group, inds)
        self.refresh_lists()
        self.unselected_list.yview_moveto(scroll_top)
        self._highlight_changed_pairs({inds})
        self.current_pair_idx = self.get_pair_index(inds)
        self.update_plot()
        self._draw_network()

    def move_to_unselected(self, event=None):
        """Move the item under the cursor (double-click) or current pair (keyboard)."""
        if self._select_after is not None:
            self.root.after_cancel(self._select_after)
            self._select_after = None
        if event is not None:
            idx = self.selected_list.nearest(event.y)
        else:
            sel = self.selected_list.curselection()
            idx = sel[-1] if sel else None
        if idx is None or idx < 0:
            return
        sel_map = getattr(self, '_sel_list_pairs', None)
        if sel_map is not None:
            if idx >= len(sel_map) or sel_map[idx] is None:
                # Check if it's a collapsible group header
                hdr_keys = getattr(self, '_sel_list_header_keys', None)
                if hdr_keys is not None and idx < len(hdr_keys) and hdr_keys[idx] is not None:
                    hkey = hdr_keys[idx]
                    if getattr(self, '_session_any_mode', False):
                        if hkey in self._any_expanded_group_tags:
                            self._any_expanded_group_tags.discard(hkey)
                            scroll_top = self.selected_list.yview()[0]
                            self.refresh_lists()
                            self.selected_list.yview_moveto(scroll_top)
                        else:
                            self._toggle_any_avail_group(hkey)
                        return
                    collapsed = getattr(self, '_sel_collapsed_headers', set())
                    if hkey in collapsed:
                        collapsed.discard(hkey)
                    else:
                        collapsed.add(hkey)
                    scroll_top = self.selected_list.yview()[0]
                    self.refresh_lists()
                    self.selected_list.yview_moveto(scroll_top)
                return
            inds = sel_map[idx]
        else:
            sorted_sel = sorted(self.selected_inds)
            if idx >= len(sorted_sel):
                return
            inds = sorted_sel[idx]
        if getattr(self, '_session_any_mode', False):
            return
        # Preserve scroll position so the list doesn't jump to the top
        scroll_top = self.selected_list.yview()[0]
        self._push_undo()
        self.selected_inds.discard(inds)
        self.unselected_inds.add(inds)
        self.refresh_lists()
        self.selected_list.yview_moveto(scroll_top)
        self._highlight_changed_pairs({inds})
        self.current_pair_idx = self.get_pair_index(inds)
        self.update_plot()
        self._draw_network()

    def _move_current_pair(self):
        """Hotkey 'm': toggle the current pair between Available and Selected,
        then advance the cursor to the next pair so the user keeps their place."""
        if getattr(self, '_session_any_mode', False):
            return
        if self.current_pair_idx >= len(self.all_inds):
            return
        row = self.all_inds[self.current_pair_idx]
        inds = tuple(row)
        self._push_undo()
        if inds in self.unselected_inds:
            self.unselected_inds.discard(inds)
            self.selected_inds.add(inds)
        else:
            self.selected_inds.discard(inds)
            self.unselected_inds.add(inds)
        self._highlight_changed_pairs({inds})
        # Advance to next pair (clamp at end)
        next_idx = min(self.current_pair_idx + 1, len(self.all_inds) - 1)
        self.current_pair_idx = next_idx
        self.refresh_lists()
        self._select_pair_in_list(self._pair_at_all_inds_idx(next_idx))
        self.update_plot()
        self._draw_network()

    def _select_pair_in_list(self, inds):
        """Set listbox cursor to the given pair and scroll it into view."""
        if inds is None:
            return
        if inds in self.unselected_inds:
            listbox = self.unselected_list
            avail_map = getattr(self, '_avail_list_pairs', None)
            if avail_map:
                pos = next((i for i, e in enumerate(avail_map)
                            if e is not None and e[0] == inds), None)
                if pos is None:
                    return
            else:
                pos = sorted(self.unselected_inds).index(inds)
        elif inds in self.selected_inds:
            listbox = self.selected_list
            sel_map = getattr(self, '_sel_list_pairs', None)
            if sel_map:
                if (getattr(self, '_session_any_mode', False)
                        and isinstance(inds, tuple) and len(inds) == 3):
                    sess, r, t = str(inds[0]), int(inds[1]), int(inds[2])
                    pos = next(
                        (i for i, e in enumerate(sel_map)
                         if e is not None and len(e) >= 3
                         and isinstance(e[0], Key)
                         and str(e[0].session) == sess
                         and int(e[1]) == r and int(e[2]) == t),
                        None,
                    )
                else:
                    pos = next((i for i, e in enumerate(sel_map) if e == inds), None)
                if pos is None:
                    return
            else:
                pos = sorted(self.selected_inds).index(inds)
        else:
            return
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(pos)
        listbox.activate(pos)
        listbox.see(pos)

    def on_pair_select(self, event):
        widget = event.widget
        # Keep a single active selection across the two pair listboxes:
        # clicking in one clears selection in the other (Finder-like).
        try:
            if widget is self.unselected_list:
                self.selected_list.selection_clear(0, tk.END)
            elif widget is self.selected_list:
                self.unselected_list.selection_clear(0, tk.END)
        except Exception:
            pass
        # In EXTENDED mode curselection() reflects the full multi-selection at ButtonRelease
        sel = widget.curselection()
        idx = sel[-1] if sel else widget.nearest(event.y)
        if idx < 0 or idx >= widget.size():
            return
        inds_from_map = None
        if widget is self.unselected_list:
            mp = getattr(self, '_avail_list_pairs', None)
            if mp and idx < len(mp):
                e = mp[idx]
                if e is not None:
                    if e[0] == _AVAIL_GROUP_HDR:
                        return
                    inds_from_map = e[0]
        elif widget is self.selected_list:
            mp = getattr(self, '_sel_list_pairs', None)
            if mp and idx < len(mp) and mp[idx] is not None:
                inds_from_map = mp[idx]

        if inds_from_map is not None:
            try:
                self.current_pair_idx = self.get_pair_index(inds_from_map)
            except (ValueError, TypeError):
                return
        else:
            item = widget.get(idx)
            # Allow optional bookmark pin prefix before the "[ref, tgt]" token.
            m = re.match(
                r'^\s*(?:\U0001F4CC\s*)?(?:[^\[]+\s+)?\[\s*(\d+)\s*,\s*(\d+)\s*\]',
                item)
            if not m:
                return  # header / separator
            inds = (int(m.group(1)), int(m.group(2)))
            try:
                self.current_pair_idx = self.get_pair_index(inds)
            except (ValueError, TypeError):
                return
        # When Control is held (toggle-select), skip the expensive plot redraw so
        # the user can build a multi-selection without triggering redraws on each click.
        # Shift (range-select) still navigates to the last item as normal.
        _ctrl = event.state & 0x4   # Control key (macOS: Ctrl; also covers Ctrl+Shift)
        _cmd  = event.state & 0x8   # Command/Meta key on macOS
        if _ctrl or _cmd:
            return
        # Debounce: defer the heavy update so double-click can fire first
        if self._select_after is not None:
            self.root.after_cancel(self._select_after)
        self._select_after = self.root.after(180, self._do_pair_select_update)

    def _pair_list_toggle_select(self, event, listbox, kind: str):
        """Cmd/Ctrl+click on a pair row: toggle selection without clearing others."""
        # Keep a single active selection across listboxes.
        try:
            if listbox is self.unselected_list:
                self.selected_list.selection_clear(0, tk.END)
            elif listbox is self.selected_list:
                self.unselected_list.selection_clear(0, tk.END)
        except Exception:
            pass
        try:
            idx = int(listbox.nearest(event.y))
        except Exception:
            return 'break'
        if idx < 0 or idx >= listbox.size():
            return 'break'
        # Toggle this row in/out of selection
        try:
            included = listbox.selection_includes(idx)
        except Exception:
            included = False
        if included:
            listbox.selection_clear(idx)
        else:
            listbox.selection_set(idx)
        listbox.activate(idx)
        listbox.see(idx)

        # Update current_pair_idx to the clicked pair and show it immediately.
        inds = None
        if kind == 'avail':
            mp = getattr(self, '_avail_list_pairs', None)
            if mp and idx < len(mp) and mp[idx] is not None:
                ent = mp[idx]
                if ent[0] == _AVAIL_GROUP_HDR:
                    return 'break'
                inds = ent[0]
        else:
            mp = getattr(self, '_sel_list_pairs', None)
            if mp and idx < len(mp) and mp[idx] is not None:
                inds = mp[idx]
        if inds is not None:
            try:
                self.current_pair_idx = self.get_pair_index(inds)
            except Exception:
                pass
        # Ensure the clicked pair becomes the active preview "on top",
        # without collapsing the multi-selection.
        try:
            self._exit_spike_attribution_view()
        except Exception:
            pass
        try:
            self._focused_pair = None
        except Exception:
            pass
        try:
            self.update_plot()
            self._draw_network()
        except Exception:
            pass
        return 'break'

    def _do_pair_select_update(self):
        """Execute the deferred pair-select update (after debounce timeout)."""
        self._select_after = None
        self._exit_spike_attribution_view()
        self._mark_jitter_viewed()
        # Clear focus pair so the clicked pair's CCG is displayed
        self._focused_pair = None
        if hasattr(self, 'network_panel'):
            self.network_panel._focus_pair_var.set("")
            self.network_panel._focus_pair_info_var.set("")
        self.update_plot()
        self._draw_network()

    def _on_arrow_key(self, event):
        """Up/Down arrow key in a pair list: update CCG to the newly selected pair."""
        self.on_pair_select(event)

    # ── Pair-list search (Ctrl+F) ──────────────────────────────────────

    def _search_show(self):
        """Show the search bar and focus the entry."""
        if not hasattr(self, '_search_frame'):
            return
        self._search_frame.pack(fill=tk.X, pady=(3, 0))
        self._search_entry.focus_set()
        self._search_entry.select_range(0, tk.END)

    def _search_hide(self):
        """Clear search and hide the bar."""
        self._search_clear()
        if hasattr(self, '_search_frame'):
            self._search_frame.pack_forget()
        self.root.focus_set()

    _SEARCH_MATCH_BG = '#fff099'   # all matches
    _BOOKMARK_LIST_BG = '#ffcdd2'
    _BOOKMARK_LIST_FG = '#b71c1c'
    _BOOKMARK_LIST_SELBG = '#ef9a9a'
    _BOOKMARK_LIST_SELFG = '#4a0000'

    def _bookmark_label_prefix(self, inds) -> str:
        if getattr(self, '_session_any_mode', False):
            sess = (str(inds[0].session) if isinstance(inds[0], Key)
                    else str(inds[0]))
            t = (sess, int(inds[1]), int(inds[2]))
            return '\U0001F4CC ' if t in self._bookmarked_pairs else ''
        t = tuple(int(x) for x in inds[:2])
        return '\U0001F4CC ' if t in self._bookmarked_pairs else ''

    def _reapply_bookmark_list_styles(self):
        """Red row styling for bookmarked pairs (after jitter / search coloring)."""
        bm = getattr(self, '_bookmarked_pairs', None) or set()
        if not bm:
            return
        av = getattr(self, '_avail_list_pairs', None)
        if av:
            for i, entry in enumerate(av):
                if entry is None:
                    continue
                raw = entry[0]
                if raw == _AVAIL_GROUP_HDR:
                    continue
                if getattr(self, '_session_any_mode', False):
                    sess = (str(raw[0].session) if isinstance(raw[0], Key)
                            else str(raw[0]))
                    ti = (sess, int(raw[1]), int(raw[2]))
                else:
                    ti = tuple(int(x) for x in raw[:2])
                if ti not in bm:
                    continue
                try:
                    self.unselected_list.itemconfig(
                        i,
                        background=self._BOOKMARK_LIST_BG,
                        foreground=self._BOOKMARK_LIST_FG,
                        selectbackground=self._BOOKMARK_LIST_SELBG,
                        selectforeground=self._BOOKMARK_LIST_SELFG,
                    )
                except tk.TclError:
                    pass
        sel = getattr(self, '_sel_list_pairs', None)
        if sel:
            for i, inds in enumerate(sel):
                if inds is None:
                    continue
                if getattr(self, '_session_any_mode', False):
                    sess = (str(inds[0].session) if isinstance(inds[0], Key)
                            else str(inds[0]))
                    ti = (sess, int(inds[1]), int(inds[2]))
                else:
                    ti = tuple(int(x) for x in inds[:2])
                if ti not in bm:
                    continue
                try:
                    self.selected_list.itemconfig(
                        i,
                        background=self._BOOKMARK_LIST_BG,
                        foreground=self._BOOKMARK_LIST_FG,
                        selectbackground=self._BOOKMARK_LIST_SELBG,
                        selectforeground=self._BOOKMARK_LIST_SELFG,
                    )
                except tk.TclError:
                    pass

    def _bookmark_toggle_current(self, event=None):
        """Add/remove bookmark for the currently selected pair (list selection wins)."""
        inds = self._selected_pair_from_lists()
        if inds is None:
            if self.current_pair_idx >= len(self.all_inds):
                return
            row = self.all_inds[self.current_pair_idx]
            if getattr(self, '_session_any_mode', False):
                hl = getattr(self, '_any_pair_handle_list', None) or []
                ci = self.current_pair_idx
                if 0 <= ci < len(hl):
                    ck, r, t = hl[ci]
                    inds = (str(ck.session), int(r), int(t))
                else:
                    return
            else:
                inds = tuple(int(x) for x in row[:2])
        if inds in self._bookmarked_pairs:
            self._bookmarked_pairs.discard(inds)
        else:
            self._bookmarked_pairs.add(inds)
        self.refresh_lists()

    def _selected_pair_from_lists(self):
        """Return (ref,tgt) or (sess,ref,tgt) in any-mode; else None."""
        for lb, mp in [
            (getattr(self, 'unselected_list', None), getattr(self, '_avail_list_pairs', None)),
            (getattr(self, 'selected_list', None), getattr(self, '_sel_list_pairs', None)),
        ]:
            if lb is None:
                continue
            try:
                sel = list(lb.curselection())
            except Exception:
                sel = []
            if not sel:
                continue
            i = sel[-1]
            if mp is None:
                continue
            try:
                entry = mp[i]
            except Exception:
                continue
            if entry is None:
                continue
            if isinstance(entry, tuple) and entry and entry[0] == _AVAIL_GROUP_HDR:
                continue
            if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], (tuple, list, np.ndarray)):
                # avail map entry: (inds, tag)
                inds = entry[0]
            else:
                inds = entry
            try:
                if getattr(self, '_session_any_mode', False):
                    sess = (str(inds[0].session) if isinstance(inds[0], Key)
                            else str(inds[0]))
                    return (sess, int(inds[1]), int(inds[2]))
                return (int(inds[0]), int(inds[1]))
            except Exception:
                continue
        return None

    def _selected_pairs_from_lists(self) -> list[tuple[int, int]]:
        """Return all explicitly selected (ref,tgt) pairs across both lists (deduped)."""
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for lb, mp in [
            (getattr(self, 'unselected_list', None), getattr(self, '_avail_list_pairs', None)),
            (getattr(self, 'selected_list', None), getattr(self, '_sel_list_pairs', None)),
        ]:
            if lb is None or mp is None:
                continue
            try:
                sel = list(lb.curselection())
            except Exception:
                sel = []
            for i in sel:
                try:
                    entry = mp[i]
                except Exception:
                    continue
                if entry is None:
                    continue
                if isinstance(entry, tuple) and entry and entry[0] == _AVAIL_GROUP_HDR:
                    continue
                if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], (tuple, list, np.ndarray)):
                    inds = entry[0]
                else:
                    inds = entry
                try:
                    if getattr(self, '_session_any_mode', False):
                        sess = (str(inds[0].session) if isinstance(inds[0], Key)
                                else str(inds[0]))
                        pair = (sess, int(inds[1]), int(inds[2]))
                    else:
                        pair = (int(inds[0]), int(inds[1]))
                except Exception:
                    continue
                if pair in seen:
                    continue
                seen.add(pair)
                out.append(pair)
        out.sort(key=lambda x: (str(x[0]), x[1], x[2]) if len(x) == 3
                               else ('', x[0], x[1]))
        return out

    def _clear_bookmarks(self):
        """Clear all bookmark markers (Selections menu)."""
        if not self._bookmarked_pairs:
            return
        self._bookmarked_pairs.clear()
        self.refresh_lists()

    def _search_update(self):
        """Rebuild match list and highlight all matches in both listboxes."""
        query = getattr(self, '_search_var', None)
        if query is None:
            return
        q = query.get().lower()
        self._search_matches = []
        self._search_cur = -1

        # Clear previous highlights by re-applying normal list colours
        self._apply_search_highlights(clear=True)

        if not q:
            self._search_count_var.set('')
            return

        for lb in (self.unselected_list, self.selected_list):
            for i in range(lb.size()):
                if q in lb.get(i).lower():
                    self._search_matches.append((lb, i))

        n = len(self._search_matches)
        if n == 0:
            self._search_count_var.set('0/0')
            return

        self._search_cur = 0
        self._apply_search_highlights()
        self._search_scroll_to_current()
        self._search_count_var.set(f'1/{n}')

    def _search_go(self, delta: int):
        """Move to next (+1) or previous (-1) match."""
        if not self._search_matches:
            return
        n = len(self._search_matches)
        self._search_cur = (self._search_cur + delta) % n
        self._apply_search_highlights()
        self._search_scroll_to_current()
        self._search_count_var.set(f'{self._search_cur + 1}/{n}')

    def _search_clear(self):
        # Clear highlights before resetting state — the trace on _search_var
        # fires _search_update() synchronously, which would wipe _search_matches
        # before we get a chance to iterate them.
        self._apply_search_highlights(clear=True)
        self._search_matches = []
        self._search_cur = -1
        self._search_count_var.set('')
        if hasattr(self, '_search_var'):
            self._search_var.set('')

    def _apply_search_highlights(self, clear: bool = False):
        """Apply/remove search highlight colours on matched rows."""
        if clear:
            # Restore default bg for all previously highlighted rows
            for lb, i in getattr(self, '_search_matches', []):
                try:
                    lb.itemconfig(i, background='', selectbackground='')
                except tk.TclError:
                    pass
            self._reapply_bookmark_list_styles()
            return
        for lb, i in self._search_matches:
            try:
                lb.itemconfig(i, background=self._SEARCH_MATCH_BG,
                              selectbackground=self._SEARCH_MATCH_BG)
            except tk.TclError:
                pass
        self._reapply_bookmark_list_styles()

    def _search_scroll_to_current(self):
        if not self._search_matches or self._search_cur < 0:
            return
        lb, i = self._search_matches[self._search_cur]
        lb.see(i)

    def _select_all(self):
        """Toggle between Select All and Deselect All."""
        if getattr(self, '_session_any_mode', False):
            return
        if self.unselected_inds:
            for inds in list(self.unselected_inds):
                self.selected_inds.add(inds)
            self.unselected_inds.clear()
        else:
            for inds in list(self.selected_inds):
                self.unselected_inds.add(inds)
            self.selected_inds.clear()
        self.refresh_lists()
        self._draw_network()

    def _ctx_menu(self, event, widget, action):
        """Right-click context menu for the pair lists.

        Supports multi-selection: if multiple items are selected (via
        Ctrl+click or Shift+click), the group-tag action applies to all.
        """
        # If right-clicked item is not already part of the selection,
        # replace selection with just that item (standard behavior).
        click_idx = widget.nearest(event.y)
        if click_idx not in widget.curselection():
            widget.selection_clear(0, tk.END)
            widget.selection_set(click_idx)
            widget.activate(click_idx)

        # Build list of selected pairs — for 'add', check if in deleted section
        if action == 'add':
            avail_map = getattr(self, '_avail_list_pairs', None)
            sel_indices = list(widget.curselection())
            pairs, deleted_pairs = [], []
            for i in sel_indices:
                if avail_map is not None:
                    if i >= len(avail_map) or avail_map[i] is None:
                        continue
                    inds, tag = avail_map[i]
                    if tag == 'deleted':
                        deleted_pairs.append(inds)
                    else:
                        pairs.append(inds)
                else:
                    # fallback: no predicted pairs present
                    sorted_unsel = sorted(self.unselected_inds)
                    sorted_del   = sorted(self.deleted_inds)
                    sep_idx = len(sorted_unsel) if self.deleted_inds else None
                    if i < len(sorted_unsel):
                        pairs.append(sorted_unsel[i])
                    elif sep_idx is not None and i > sep_idx:
                        di = i - sep_idx - 1
                        if 0 <= di < len(sorted_del):
                            deleted_pairs.append(sorted_del[di])
        else:
            sel_map = getattr(self, '_sel_list_pairs', None)
            if sel_map is not None:
                pairs = [sel_map[i] for i in widget.curselection()
                         if i < len(sel_map) and sel_map[i] is not None]
            else:
                ss = sorted(self.selected_inds)
                pairs = [ss[i] for i in widget.curselection() if i < len(ss)]
            deleted_pairs = []
        n = len(pairs)
        nd = len(deleted_pairs)

        menu = tk.Menu(self.root, tearoff=0)
        _any = getattr(self, '_session_any_mode', False)
        if action == 'add':
            if not _any:
                if pairs:
                    menu.add_command(
                        label=f"Move to Selected ({n})" if n > 1 else "Move to Selected",
                        command=lambda pp=pairs: self._ctx_move_multi_to_selected(pp))
                    menu.add_command(
                        label=f"Move to Deleted ({n})" if n > 1 else "Move to Deleted",
                        command=lambda pp=pairs: self._ctx_delete_pairs(pp))
                if deleted_pairs:
                    menu.add_command(
                        label=f"Restore to Available ({nd})" if nd > 1 else "Restore to Available",
                        command=lambda pp=deleted_pairs: self._ctx_restore_from_deleted(pp))
                if not pairs and not deleted_pairs:
                    menu.add_command(label="(nothing selected)", state='disabled')
                menu.add_command(label="Select All", command=self._select_all)
            elif not pairs and not deleted_pairs:
                menu.add_command(label="(nothing selected)", state='disabled')
        else:
            if not _any:
                menu.add_command(
                    label=f"Move to Available ({n})" if n > 1 else "Move to Available",
                    command=lambda pp=pairs: self._ctx_move_multi_to_unselected(pp))
                menu.add_command(
                    label=f"Move to Deleted ({n})" if n > 1 else "Move to Deleted",
                    command=lambda pp=pairs: self._ctx_delete_from_selected(pp))
            elif pairs:
                menu.add_command(
                    label=f"Move to Deleted ({n})" if n > 1 else "Move to Deleted",
                    command=lambda pp=pairs: self._ctx_delete_from_selected(pp))

        # Group tag submenu
        menu.add_separator()
        grp_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Group tag", menu=grp_menu)
        grp_menu.add_command(label="Create new group…",
                             command=self._create_group_dialog)
        if self._groups:
            grp_menu.add_separator()
        special_items = []
        for gname in sorted(self._groups):
            if gname.startswith(_SPECIAL_PREFIX):
                special_items.append(gname)
                continue
            if gname.startswith('__'):
                continue
            if pairs:
                all_in = all(self._pair_in_group(p, gname) for p in pairs)
                label = f"{'✓ ' if all_in else ''}  {gname}"
                grp_menu.add_command(
                    label=label,
                    command=lambda g=gname, pp=pairs: self._toggle_pairs_group(pp, g))
        # Special groups as sub-cascade
        if special_items:
            sp_menu = tk.Menu(grp_menu, tearoff=0)
            grp_menu.add_cascade(label="Special", menu=sp_menu)
            for gname in special_items:
                display = gname[len(_SPECIAL_PREFIX):]
                if pairs:
                    all_in = all(self._pair_in_group(p, gname) for p in pairs)
                    label = f"{'✓ ' if all_in else ''}  {display}"
                    sp_menu.add_command(
                        label=label,
                        command=lambda g=gname, pp=pairs: self._toggle_pairs_group(pp, g))

        # "Show together" — pin/unpin pairs in stacked view
        if pairs:
            menu.add_separator()
            max_tog = self._settings.get('max_show_together', 5)
            all_pinned = all(tuple(p) in [tuple(x) for x in self._together_pairs]
                             for p in pairs)
            tog_label = ("Remove from 'Show Together'"
                         if all_pinned else "Show Together")
            menu.add_command(label=tog_label,
                             command=lambda pp=pairs: self._toggle_together(pp))
            if self._together_pairs:
                menu.add_command(
                    label=f"Clear 'Show Together' ({len(self._together_pairs)} pairs)",
                    command=self._clear_together)

        # Pair tags (single pair only)
        if n == 1:
            menu.add_separator()
            p = pairs[0]
            _sess, _rt = self._pair_sess_rt(p)
            has_tags = _rt in self._pair_tags
            menu.add_command(
                label=f"{'✓ ' if has_tags else ''}Pair tags…",
                command=self._pair_tags_dialog)

        # Export (single or multi-selection). Export dialog shows preview of first selected pair.
        menu.add_separator()
        menu.add_command(label="Export view as PNG…", command=lambda: self._export_current_view('png'))
        menu.add_command(label="Export view as PDF…", command=lambda: self._export_current_view('pdf'))
        menu.tk_popup(event.x_root, event.y_root)

    def _toggle_together(self, pairs):
        """Pin or unpin pairs from the 'Show Together' stacked view."""
        max_tog = self._settings.get('max_show_together', 5)
        tog_tuples = [tuple(p) for p in self._together_pairs]
        for p in pairs:
            pt = tuple(p)
            if pt in tog_tuples:
                self._together_pairs = [x for x in self._together_pairs
                                        if tuple(x) != pt]
                tog_tuples = [tuple(x) for x in self._together_pairs]
            else:
                if len(self._together_pairs) < max_tog:
                    self._together_pairs.append(pt)
                    tog_tuples.append(pt)
        self.update_plot()

    def _clear_together(self):
        """Remove all pairs from the 'Show Together' pool."""
        self._together_pairs.clear()
        self.update_plot()

    def _ctx_move_to_selected(self, pair):
        """Context-menu: move a specific pair from Available → Selected."""
        if pair is None: return
        self._ctx_move_multi_to_selected([pair])
        self.current_pair_idx = self.get_pair_index(pair)
        self.update_plot()

    def _ctx_move_multi_to_selected(self, pairs):
        """Context-menu: move multiple pairs from Available → Selected."""
        if getattr(self, '_session_any_mode', False):
            return
        if not pairs: return
        scroll_top = self.unselected_list.yview()[0]
        self._push_undo()
        for p in pairs:
            self.unselected_inds.discard(p)
            self.selected_inds.add(p)
        self.refresh_lists()
        self.unselected_list.yview_moveto(scroll_top)
        self._highlight_changed_pairs(set(pairs))
        self._draw_network()

    def _ctx_move_to_unselected(self, pair):
        """Context-menu: move a specific pair from Selected → Available."""
        if pair is None: return
        self._ctx_move_multi_to_unselected([pair])
        self.current_pair_idx = self.get_pair_index(pair)
        self.update_plot()

    def _ctx_move_multi_to_unselected(self, pairs):
        """Context-menu: move multiple pairs from Selected → Available."""
        if getattr(self, '_session_any_mode', False):
            return
        if not pairs: return
        scroll_top = self.selected_list.yview()[0]
        self._push_undo()
        for p in pairs:
            self.selected_inds.discard(p)
            self.unselected_inds.add(p)
        self.refresh_lists()
        self.selected_list.yview_moveto(scroll_top)
        self._highlight_changed_pairs(set(pairs))
        self._draw_network()

    def _on_delete_pair(self, event=None):
        """Delete key: toggle current pair in/out of the Deleted section.
        When triggered from the selected list, moves the pair from selected→deleted."""
        if self.current_pair_idx >= len(self.all_inds):
            return
        if getattr(self, '_session_any_mode', False):
            trip = self._pair_at_all_inds_idx(self.current_pair_idx)
            if trip is None:
                return
            if trip not in self.selected_inds:
                return
            scroll_top = self.selected_list.yview()[0]
            self._push_undo()
            self.selected_inds.discard(trip)
            self.deleted_inds.add(trip)
            hl_old = list(getattr(self, '_any_pair_handle_list', ()) or ())
            next_trip = None
            for i in range(self.current_pair_idx + 1, len(hl_old)):
                ck, r, t = hl_old[i]
                tr = (str(ck.session), int(r), int(t))
                if tr in self.selected_inds:
                    next_trip = tr
                    break
            if next_trip is None:
                for i in range(self.current_pair_idx):
                    ck, r, t = hl_old[i]
                    tr = (str(ck.session), int(r), int(t))
                    if tr in self.selected_inds:
                        next_trip = tr
                        break
            self.refresh_lists()
            if next_trip is not None:
                self.current_pair_idx = self.get_pair_index(next_trip)
            elif len(self.all_inds):
                self.current_pair_idx = min(
                    self.current_pair_idx, len(self.all_inds) - 1)
            else:
                self.current_pair_idx = 0
            self.selected_list.yview_moveto(scroll_top)
            if len(self.all_inds):
                self._select_pair_in_list(
                    self._pair_at_all_inds_idx(self.current_pair_idx))
            self._flush_deleted_to_store()
            self._draw_network()
            self.update_plot()
            return

        inds = tuple(int(x) for x in self.all_inds[self.current_pair_idx])

        if inds in self.selected_inds:
            # Allow deletion from selected list: move selected→deleted
            scroll_top = self.selected_list.yview()[0]
            self._push_undo()
            self.selected_inds.discard(inds)
            self.deleted_inds.add(inds)
            # Advance cursor to next selected pair, then any available pair
            n = len(self.all_inds)
            next_idx = None
            for i in range(self.current_pair_idx + 1, n):
                if tuple(self.all_inds[i]) in self.selected_inds:
                    next_idx = i; break
            if next_idx is None:
                for i in range(self.current_pair_idx):
                    if tuple(self.all_inds[i]) in self.selected_inds:
                        next_idx = i; break
            if next_idx is None:
                for i in range(self.current_pair_idx + 1, n):
                    if tuple(self.all_inds[i]) in self.unselected_inds:
                        next_idx = i; break
            if next_idx is not None:
                self.current_pair_idx = next_idx
            self.refresh_lists()
            self.selected_list.yview_moveto(scroll_top)
            if next_idx is not None:
                self._select_pair_in_list(tuple(self.all_inds[next_idx]))
            self._flush_deleted_to_store()
            self._draw_network()
            self.update_plot()
            return

        scroll_top = self.unselected_list.yview()[0]
        self._push_undo()
        if inds in self.deleted_inds:
            # Restore: move back to Available
            self.deleted_inds.discard(inds)
            self.unselected_inds.add(inds)
            self.refresh_lists()
            self.unselected_list.yview_moveto(scroll_top)
            self._select_pair_in_list(inds)
        else:
            # Delete: move from Available into Deleted
            self.unselected_inds.discard(inds)
            self.deleted_inds.add(inds)
            # Advance cursor to next pair in Available (forward then wrap)
            n = len(self.all_inds)
            next_idx = None
            for i in range(self.current_pair_idx + 1, n):
                if tuple(self.all_inds[i]) in self.unselected_inds:
                    next_idx = i
                    break
            if next_idx is None:
                for i in range(self.current_pair_idx):
                    if tuple(self.all_inds[i]) in self.unselected_inds:
                        next_idx = i
                        break
            if next_idx is not None:
                self.current_pair_idx = next_idx
            self.refresh_lists()
            self.unselected_list.yview_moveto(scroll_top)
            if next_idx is not None:
                self._select_pair_in_list(tuple(self.all_inds[next_idx]))
        self._flush_deleted_to_store()
        self._draw_network()
        self.update_plot()

    def _ctx_restore_from_deleted(self, pairs):
        """Context-menu: restore pairs from deleted section back to Available."""
        if not pairs:
            return
        scroll_top = self.unselected_list.yview()[0]
        self._push_undo()
        for p in pairs:
            self.deleted_inds.discard(p)
            self.unselected_inds.add(p)
        self.refresh_lists()
        self.unselected_list.yview_moveto(scroll_top)
        self._highlight_changed_pairs(set(pairs))
        self._flush_deleted_to_store()

    def _ctx_delete_pairs(self, pairs):
        """Context-menu: move pairs from Available to deleted section."""
        if not pairs:
            return
        scroll_top = self.unselected_list.yview()[0]
        self._push_undo()
        for p in pairs:
            self.unselected_inds.discard(p)
            self.deleted_inds.add(p)
        self.refresh_lists()
        self.unselected_list.yview_moveto(scroll_top)
        self._flush_deleted_to_store()

    def _ctx_delete_from_selected(self, pairs):
        """Context-menu: move pairs from Selected to deleted section (keep pair tags)."""
        if not pairs:
            return
        scroll_top = self.selected_list.yview()[0]
        self._push_undo()
        for p in pairs:
            trip = self._pair_row_selected_trip(p)
            self.selected_inds.discard(trip)
            self.deleted_inds.add(trip)
        self.refresh_lists()
        self.selected_list.yview_moveto(scroll_top)
        self._flush_deleted_to_store()

    def _pair_at_all_inds_idx(self, idx: int):
        """Canonical pair key matching list rows / selection sets."""
        if getattr(self, '_session_any_mode', False):
            hl = getattr(self, '_any_pair_handle_list', None) or []
            if idx < 0 or idx >= len(hl):
                return None
            ck, r, t = hl[idx]
            return (str(ck.session), int(r), int(t))
        row = self.all_inds[idx]
        return tuple(int(x) for x in row)

    def get_pair_index(self, inds):
        if getattr(self, '_session_any_mode', False):
            hl = getattr(self, '_any_pair_handle_list', None) or []
            if len(inds) >= 3:
                if isinstance(inds[0], Key):
                    ck, r, t = inds[0], int(inds[1]), int(inds[2])
                    for i, (k2, r2, t2) in enumerate(hl):
                        if k2 == ck and r2 == r and t2 == t:
                            return i
                    return 0
                sess, r, t = str(inds[0]), int(inds[1]), int(inds[2])
                for i, (k2, r2, t2) in enumerate(hl):
                    if str(k2.session) == sess and r2 == r and t2 == t:
                        return i
                return 0
            r, t = int(inds[0]), int(inds[1])
            for i, (k2, r2, t2) in enumerate(hl):
                if k2 == self.key and r2 == r and t2 == t:
                    return i
            return 0
        for i, pair in enumerate(self.all_inds):
            if tuple(pair) == tuple(inds):
                return i
        return 0

    # ------------------------------------------------------------------
    # Segment navigation
    # ------------------------------------------------------------------

    def _n_total_segments(self):
        """Total navigable segments: real + All + custom."""
        return self.n_segments + 1 + len(self._custom_segments)

    def _clamp_current_segment_for_session(self):
        """Keep ``current_segment`` within the loaded session (fixes Any-mode session hops).

        After binding another session's ``ccg_pointer``/``n_segments``/``_custom_segments``,
        a prior custom-segment index can point past ``_custom_segments`` → IndexError in
        ``_png_path`` / plot title.
        """
        ns = int(self.n_segments) if self.n_segments is not None else 0
        cs_list = getattr(self, '_custom_segments', []) or []
        if ns <= 0:
            self.current_segment = 0
            return
        seg = int(self.current_segment)
        # Valid ids: ``0..ns-1`` (real), ``ns`` (All), ``ns+1 .. ns+len(cs_list)`` (custom)
        max_id = ns + len(cs_list)
        if seg < 0:
            self.current_segment = 0
        elif seg > max_id:
            self.current_segment = ns
        elif seg > ns:
            ci = seg - ns - 1
            if ci < 0 or ci >= len(cs_list):
                self.current_segment = ns

    def _is_custom_segment(self, seg=None):
        if seg is None:
            seg = self.current_segment
        return seg > self.n_segments

    def _custom_seg_index(self, seg=None):
        """Return the index into _custom_segments for the given segment id."""
        if seg is None:
            seg = self.current_segment
        return seg - self.n_segments - 1

    def change_segment(self, delta):
        total = self._n_total_segments()
        self.current_segment = (self.current_segment + delta) % total
        self._clamp_current_segment_for_session()
        self._update_segment_label()
        self._exit_spike_attribution_view()
        self.update_plot()

    def _jump_to_segment(self, idx):
        self.current_segment = idx
        self._clamp_current_segment_for_session()
        self._update_segment_label()
        self._exit_spike_attribution_view()
        # Update chip highlight immediately (otherwise it only updates on pair change)
        try:
            inds = self._current_inds()
            if inds is not None:
                self._update_sig_indicators(inds)
        except Exception:
            pass
        self.update_plot()

    def _update_segment_label(self):
        seg = int(self.current_segment)
        if seg == self.n_segments:
            self.segment_var.set(_ALL_SEGS)
        elif self._is_custom_segment(seg):
            ci = self._custom_seg_index(seg)
            cs_list = getattr(self, '_custom_segments', []) or []
            if 0 <= ci < len(cs_list):
                self.segment_var.set(cs_list[ci]['name'])
            else:
                self.segment_var.set('custom')
        else:
            if 0 <= seg < len(self.segment_names):
                self.segment_var.set(self.segment_names[seg])
            else:
                self.segment_var.set(str(seg))
        self.plot_title_var.set(self.get_plot_title())

    def _remove_custom_segment(self, ci):
        """Remove a custom segment by its index in _custom_segments."""
        if 0 <= ci < len(self._custom_segments):
            self._custom_segments.pop(ci)
            # If we were viewing the removed (or a later) custom segment, reset
            if self.current_segment > self.n_segments:
                self.current_segment = min(self.current_segment,
                                           self._n_total_segments() - 1)
            self._build_sig_chips()
            self._update_segment_label()
            self.update_plot()

    def _update_sig_indicators(self, inds):
        """Color each segment chip green/gray based on significance."""
        if not hasattr(self, 'seg_sig_labels'):
            return
        ref, tgt = int(inds[0]), int(inds[1])
        cd = self.ccg_data
        conf = cd.conf if cd is not None else None
        lb = getattr(conf, 'min_lag_bin', None)
        ub = getattr(conf, 'max_lag_bin', None)

        selected = getattr(self, '_stacked_segments', set()) or set()
        for chip_idx, lbl in enumerate(self.seg_sig_labels):
            # Custom segment chips (after All chip)
            if chip_idx > self.n_segments:
                ci = chip_idx - self.n_segments - 1
                cs_list = getattr(self, '_custom_segments', [])
                if 0 <= ci < len(cs_list):
                    cs = cs_list[ci]
                    pc = cs.get('pval_corrected')
                    if (pc is not None and lb is not None and ub is not None
                            and ref < pc.shape[1] and tgt < pc.shape[2]):
                        sig = bool(pc[0, ref, tgt, lb:ub].min()
                                   <= self.active_alpha)
                    else:
                        sig = False
                    active = (self.current_segment == chip_idx)
                    bg = ('#4CAF50' if active else '#90EE90') if sig \
                        else ('#FFD54F' if active else '#FFF9C4')
                else:
                    bg = '#FFF9C4'
                if chip_idx in selected:
                    lbl.config(bg='#BBDEFB', relief=tk.SUNKEN)
                else:
                    lbl.config(bg=bg, relief=tk.SUNKEN)
                continue

            is_all_chip = (chip_idx == self.n_segments)
            if is_all_chip:
                sig = any(self._is_significant(ref, tgt, s)
                          for s in range(self.n_segments))
                active = (self.current_segment == self.n_segments)
            else:
                seg = chip_idx
                if (cd is not None and cd.pval_corrected is not None and
                        lb is not None and ub is not None and
                        seg < cd.pval_corrected.shape[0]):
                    sig = bool(
                        cd.pval_corrected[seg, ref, tgt, lb:ub].min()
                        <= self.active_alpha)
                else:
                    sig = self._is_significant(ref, tgt, seg)
                active = (seg == self.current_segment)

            if sig:
                bg = '#4CAF50' if active else '#90EE90'
            else:
                bg = '#9E9E9E' if active else '#E0E0E0'
            if chip_idx in selected:
                lbl.config(bg='#BBDEFB', relief=tk.SUNKEN)
            else:
                lbl.config(bg=bg, relief=tk.RAISED if not active else tk.SUNKEN)

    def _shank_label(self, idx: int) -> str:
        """Return the shank number for neuron at position idx, or str(idx) as fallback."""
        shank_ids = getattr(self.neurons, 'shank_ids', None)
        if shank_ids is not None:
            try:
                return str(int(shank_ids[idx]))
            except Exception:
                pass
        return str(idx)

    def _pair_label(self, inds) -> str:
        """Short display label for a (ref, tgt) pair using shank numbers."""
        return f"{self._shank_label(inds[0])}→{self._shank_label(inds[1])}"

    def get_plot_title(self):
        if self.current_pair_idx < len(self.all_inds):
            inds = self.all_inds[self.current_pair_idx]
            if self._is_custom_segment():
                ci = self._custom_seg_index()
                seg_label = self._custom_segments[ci]['name']
            elif self.current_segment == self.n_segments:
                seg_label = _ALL_SEGS
            else:
                seg_label = self.segment_names[self.current_segment]
            sess = self.key.session if self.key else ''
            def _fmt_ct(x):
                s = str(x)
                low = s.lower()
                if low == 'pyr':
                    return 'PYR'
                if low in ('inter', 'int'):
                    return 'INT'
                return s.upper()
            ct = (f"{_fmt_ct(self.key.conn_type[0])}-{_fmt_ct(self.key.conn_type[1])}"
                  if self.key and self.key.conn_type else '')
            pair_str = (f"{self._pair_label(inds)} "
                        f"(inds [{inds[0]}, {inds[1]}])")
            return f"{sess} | {ct} | {pair_str} — {seg_label}"
        return "No pair selected"

    # ------------------------------------------------------------------
    # PNG rendering
    # ------------------------------------------------------------------

    def _png_path(self, inds, segment, _render_cfg=None, _hires_override=None) -> str:
        """Return the disk-cache path for a PNG.

        When a cache configuration is set and the current display state matches it,
        the path encodes only (pair, segment, norm, alpha, res, scale, jitter) —
        no sig-state suffix — since there is only one cacheable configuration.

        When no cache configuration is set (legacy mode), the full sig state is
        encoded so different display configs each get their own cache file.

        When a cache configuration is set but the current display state does NOT
        match it, returns a short ``_rt_`` real-time path (always overwritten,
        never reused across pairs).
        """
        if self._is_custom_segment(segment):
            ci = self._custom_seg_index(segment)
            cs_list = getattr(self, '_custom_segments', []) or []
            if 0 <= ci < len(cs_list):
                cs = cs_list[ci]
                seg_name = f"custom{ci}_{cs['name']}_{cs['t0']:.2f}_{cs['t1']:.2f}"
                seg_name = seg_name.replace(' ', '_').replace(':', '-')
            else:
                seg_name = _ALL_SEGS.replace(' ', '_')
        elif segment == self.n_segments:
            seg_name = _ALL_SEGS.replace(' ', '_')
        else:
            sn = getattr(self, 'segment_names', []) or []
            if 0 <= segment < len(sn):
                seg_name = sn[segment]
            else:
                seg_name = _ALL_SEGS.replace(' ', '_')
        _norms = (_render_cfg.get('active_norms') if _render_cfg else None)
        norm_key = ('_'.join(sorted(_norms)) if _norms
                    else ('_'.join(sorted(n.name for n in self.active_norms))
                          if self.active_norms else 'raw'))
        _alpha = (_render_cfg.get('active_alpha') if _render_cfg else None)
        alpha_key = ''
        if self.ccg_data is not None and self.ccg_data.pval_corrected is not None:
            alpha_key = f'_a{(_alpha if _alpha is not None else self.active_alpha):.3f}'
        _hires = _hires_override if _hires_override is not None else getattr(self, '_highres_mode', False)
        res_key = '_hi' if _hires else '_lo'
        scale_key = {'pair': '_ssp', 'session': '_sss'}.get(
            getattr(self, '_same_scale_mode', None), '')
        _jrk = 'hi' if _hires else 'lo'
        j_key = '_j' if self._jitter_cache.get(
            (int(inds[0]), int(inds[1]), _jrk, self._jitter_seg(segment))) is not None else ''

        # Cache configuration determines path style
        if self._cache_config is not None:
            if _render_cfg is not None or self._display_matches_cache_config():
                # Canonical cached path — no sig encoding (only one config)
                _sp = (f"{self._png_sess_slug}_" if getattr(self, '_session_any_mode', False)
                       and getattr(self, '_png_sess_slug', '') else '')
                return os.path.join(
                    self.tmp_dir,
                    f"{_sp}pair_{int(inds[0])}_{int(inds[1])}_{seg_name}_{norm_key}"
                    f"{alpha_key}{res_key}{scale_key}{j_key}.png")
            else:
                # Real-time path — one file per (pair, seg, res), always overwritten
                _sp = (f"{self._png_sess_slug}_" if getattr(self, '_session_any_mode', False)
                       and getattr(self, '_png_sess_slug', '') else '')
                return os.path.join(
                    self.tmp_dir,
                    f"{_sp}_rt_{int(inds[0])}_{int(inds[1])}_{seg_name}{res_key}.png")

        # Legacy mode (no cache config): encode full sig state
        sig_key = ''
        _m = self._conn_str_method_var.get()
        sig_bits = (
            (_m if _m != 'conv' else '') +
            ('cs' if self._conn_str_show_var.get() else '') +
            ('p' if self._sig('conv_p') else '') +
            ('c' if self._sig('conv_pc') else '') +
            ('tw' if self._sig('test_window') else '') +
            ('jc' if self._sig('jitter_pc') else '') +
            ('ar' if self._acg_var_get('_acg_ref_var', False) else '') +
            ('at' if self._acg_var_get('_acg_tgt_var', False) else '') +
            ('dcr' if bool(self._acg_var_get('_acg_deconv_ref_var', False)) else '') +
            ('dct' if bool(self._acg_var_get('_acg_deconv_tgt_var', False)) else '') +
            ('wp' if _hires and self._acg_var_get('_peak_wf_var', False) else '') +
            (f'asr{self._acg_var_get("_acg_yscale_ref_var", 1.0):.1f}'
             if self._acg_var_get('_acg_ref_var', False)
                or (_hires and self._acg_var_get('_peak_wf_var', False)) else '') +
            (f'ast{self._acg_var_get("_acg_yscale_tgt_var", 1.0):.1f}'
             if self._acg_var_get('_acg_tgt_var', False) else '') +
            ('am' if self._acg_var_get('_acg_match_ccg_var', False) else '') +
            ('nc' if not self._acg_var_get('_ccg_show_var', True) else '') +
            ('nb' if not self._acg_var_get('_baseline_show_var', True) else '') +
            ('lc' if self._line_ccg_var.get() else '') +
            ('lb' if self._line_baseline_var.get() else '') +
            ('lr' if self._line_ref_var.get() else '') +
            ('lt' if self._line_tgt_var.get() else '') +
            ('lp' if (_hires and self._acg_var_get('_peak_wf_var', False)
                      and self._line_peak_wf_var.get()) else '') +
            ('lj' if self._line_jitter_var.get() else ''))
        if sig_bits:
            sig_key = f'_s{sig_bits}'
        _sp = (f"{self._png_sess_slug}_" if getattr(self, '_session_any_mode', False)
               and getattr(self, '_png_sess_slug', '') else '')
        return os.path.join(
            self.tmp_dir,
            f"{_sp}pair_{int(inds[0])}_{int(inds[1])}_{seg_name}_{norm_key}"
            f"{alpha_key}{res_key}{scale_key}{j_key}{sig_key}.png")


    def _ccg_pair_in_bounds(self, ref: int, tgt: int, cd=None) -> bool:
        """True if *(ref, tgt)* index the ref/tgt axes of *cd.ccg*."""
        cd = cd if cd is not None else self.ccg_data
        if cd is None or not hasattr(cd, 'ccg') or cd.ccg is None:
            return False
        try:
            sh = cd.ccg.shape
            if len(sh) < 4:
                return False
            r, t = int(ref), int(tgt)
            return 0 <= r < sh[1] and 0 <= t < sh[2]
        except (TypeError, ValueError, IndexError):
            return False

    def _resolve_segment_data(self, ref, tgt, segment, highres=None, include_pval=True, include_acg=False, _cd=None):
        """
        Retrieve raw CCG arrays for (ref, tgt, segment), handling custom/all-segs/normal.
        highres defaults to self._highres_mode when None.
        Returns dict with keys: ccg_raw, ccg_null_raw, seg_label,
        and optionally pval, pval_corrected (include_pval=True),
        acg_ref, acg_tgt (include_acg=True).
        Pass _cd to override self.ccg_data (e.g. from background threads).
        """
        if highres is None:
            highres = self._highres_mode
        cd = _cd if _cd is not None else self.ccg_data
        is_custom = self._is_custom_segment(segment)
        is_all = (segment == self.n_segments)
        result = {}

        if is_custom:
            ci = self._custom_seg_index(segment)
            cs = self._custom_segments[ci]
            hi = highres and 'ccg_hi' in cs
            key_ccg = 'ccg_hi' if hi else 'ccg'
            key_null = 'ccg_null_hi' if hi else 'ccg_null'
            result['ccg_raw'] = cs[key_ccg][0, ref, tgt, :]
            raw_null = cs.get(key_null)
            result['ccg_null_raw'] = raw_null[0, ref, tgt, :] if raw_null is not None else None
            result['seg_label'] = cs['name']
            if include_pval:
                key_p = 'pval_hi' if hi else 'pval'
                key_pc = 'pval_corrected_hi' if hi else 'pval_corrected'
                p = cs.get(key_p)
                pc = cs.get(key_pc)
                result['pval'] = p[0, ref, tgt, :] if p is not None else None
                result['pval_corrected'] = pc[0, ref, tgt, :] if pc is not None else None
            if include_acg:
                src = cs[key_ccg]
                result['acg_ref'] = src[0, ref, ref, :]
                result['acg_tgt'] = src[0, tgt, tgt, :]
        elif is_all:
            result['ccg_raw'] = np.sum(cd.ccg[:, ref, tgt, :], axis=0)
            result['ccg_null_raw'] = (np.sum(cd.ccg_null[:, ref, tgt, :], axis=0)
                                       if cd.ccg_null is not None else None)
            result['seg_label'] = _ALL_SEGS
            if include_pval:
                result['pval'] = None
                result['pval_corrected'] = None
            if include_acg:
                result['acg_ref'] = np.sum(cd.ccg[:, ref, ref, :], axis=0)
                result['acg_tgt'] = np.sum(cd.ccg[:, tgt, tgt, :], axis=0)
        else:
            result['ccg_raw'] = cd.ccg[segment, ref, tgt, :]
            result['ccg_null_raw'] = (cd.ccg_null[segment, ref, tgt, :]
                                       if cd.ccg_null is not None else None)
            result['seg_label'] = self.segment_names[segment]
            if include_pval:
                result['pval'] = cd.pval[segment, ref, tgt, :] if cd.pval is not None else None
                result['pval_corrected'] = (cd.pval_corrected[segment, ref, tgt, :]
                                             if cd.pval_corrected is not None else None)
            if include_acg:
                result['acg_ref'] = cd.ccg[segment, ref, ref, :]
                result['acg_tgt'] = cd.ccg[segment, tgt, tgt, :]
        return result

    def _resolve_extended_ccg(self, ref: int, tgt: int, segment: int, highres: bool,
                              extend_ms: int, bin_size_eff: float, cd):
        """Compute an extended-window raw CCG/null/pval for the current pair only.

        extend_ms is interpreted as the half-width in milliseconds (so total window is 2*extend_ms).
        Returns None when extension can't be computed (e.g., missing neurons).
        """
        if self.neurons is None:
            return None
        try:
            extend_ms = int(extend_ms)
        except Exception:
            return None
        if extend_ms <= 0:
            return None

        resk = 'hi' if bool(highres) else 'lo'
        key = (int(ref), int(tgt), int(segment), resk, int(extend_ms), float(bin_size_eff))
        if key in self._extend_cache:
            return self._extend_cache[key]

        # Segment time slicing (match jitter behavior)
        neurons_eff = self.neurons
        seg_label_eff = None
        if self._is_custom_segment(segment):
            # Custom segment: recompute on the custom window (and its saved brain-state filter).
            try:
                ci = self._custom_seg_index(segment)
                cs = self._custom_segments[ci]
                seg_label_eff = str(cs.get('name', 'custom'))
                t0 = float(cs['t0'])
                t1 = float(cs['t1'])
                neurons_eff = self.neurons.time_slice(t_start=t0, t_stop=t1)
                fs = cs.get('filter_state') or {}
                labels = (fs.get('labels') if isinstance(fs, dict) else None) or {}
                # If labels are present and not all ON, reconstruct intervals.
                if labels and not all(bool(v) for v in labels.values()):
                    active_labels = {lbl for lbl, on in labels.items() if bool(on)}
                    if active_labels:
                        none_active = 'NONE' in active_labels
                        real_labels = active_labels - {'NONE'}
                        intervals = []
                        for s, e, lbl in getattr(self, '_ts_epoch_bounds', []) or []:
                            if lbl in real_labels:
                                s_clipped, e_clipped = max(float(s), t0), min(float(e), t1)
                                if e_clipped > s_clipped:
                                    intervals.append((s_clipped, e_clipped))
                        if none_active:
                            epoch_times = sorted(
                                (max(float(s), t0), min(float(e), t1))
                                for s, e, _ in (getattr(self, '_ts_epoch_bounds', []) or [])
                                if min(float(e), t1) > max(float(s), t0))
                            cursor = t0
                            for es, ee in epoch_times:
                                if es > cursor:
                                    intervals.append((cursor, es))
                                cursor = max(cursor, ee)
                            if cursor < t1:
                                intervals.append((cursor, t1))
                        if intervals:
                            neurons_eff = self._ts_apply_brain_state_intervals(intervals, t0, t1)
            except Exception:
                neurons_eff = self.neurons
        elif segment != self.n_segments:
            try:
                et = self.ccg_pointer.edge_times
                t0 = float(et.iloc[int(segment)]['start'])
                t1 = float(et.iloc[int(segment)]['stop'])
                neurons_eff = self.neurons.time_slice(t_start=t0, t_stop=t1)
                seg_label_eff = str(self.segment_names[int(segment)])
            except Exception:
                neurons_eff = self.neurons
        else:
            seg_label_eff = _ALL_SEGS

        try:
            from neuropy.analyses import correlations
        except Exception:
            return None

        win_s = (2.0 * float(extend_ms)) / 1000.0
        # Sanity check: window must be at least 3 bins wide to be meaningful.
        if win_s < 3.0 * float(bin_size_eff):
            return None
        try:
            ccg_raw_mat = correlations.spike_correlations(
                neurons=neurons_eff,
                neuron_inds=np.array([int(ref), int(tgt)]),
                bin_size=float(bin_size_eff),
                window_size=float(win_s),
                use_acceleration=cd.conf.use_acceleration,
                symmetrize=cd.conf.symmetrize_ccg,
                edge_times=None,
            )
            ccg_1d = np.asarray(ccg_raw_mat[0, 1, :], dtype=float)
            if len(ccg_1d) < 3:
                return None
        except Exception:
            return None

        # Null + pvals from convolution method.
        # Use bin_size_eff (not cd.conf.conv_window_bins) so W scales correctly
        # when the extend window / resolution differ from the stored config.
        try:
            W = max(1, int(round(cd.conf.conv_window / float(bin_size_eff))))
            pvals_all, pred_all, _q = EranConv._conv(
                ccg_1d, W=W, wintype="gauss")
            ccg_null_1d = np.asarray(pred_all[0], dtype=float)
            pval_1d = np.asarray(pvals_all[0], dtype=float)
        except Exception:
            ccg_null_1d = None
            pval_1d = None

        if seg_label_eff is None:
            seg_label_eff = _ALL_SEGS if segment == self.n_segments else f"seg{int(segment)}"

        n_bins = len(ccg_1d)
        actual_bin_size = win_s / (n_bins - 1) if n_bins > 1 else float(bin_size_eff)
        out = {
            "ccg_raw": ccg_1d,
            "ccg_null_raw": ccg_null_1d,
            "pval": pval_1d,
            "pval_corrected": None,
            "seg_label": seg_label_eff,
            "extended": True,
            "window_size_s": float(win_s),
            "bin_size_eff": float(actual_bin_size),
        }
        self._extend_cache[key] = out
        return out

    def _get_or_render_png(self, inds, segment):
        """Return PNG path, using disk cache when display matches cache configuration.

        When a cache configuration is set and the current display state does NOT
        match it, the plot is always re-rendered (real-time, no disk-cache reuse).
        """
        is_rt = (self._cache_config is not None
                 and not self._display_matches_cache_config())
        p = self._png_path(inds, segment)
        if is_rt or not os.path.exists(p):
            return self._render_png(inds, segment)
        return p

    def _render_png_with_res(self, inds, segment, highres: bool) -> str:
        """Render a PNG at a specific resolution without changing persistent state."""
        nd_key = self.key.nd()
        if highres:
            data = getattr(self.cd, '_ccg_highres', {}).get(nd_key)
        else:
            data = getattr(self.cd, '_ccg', {}).get(nd_key)
        if data is None:
            data = self.ccg_data
        old_mode = self._highres_mode
        old_data = self.ccg_data
        self._highres_mode = highres
        self.ccg_data = data
        try:
            path = self._render_png(inds, segment, highres=highres)
        finally:
            self._highres_mode = old_mode
            self.ccg_data = old_data
        return path

    def _render_png(self, inds, segment, highres=None,
                    _render_cfg=None, _ccg_data_override=None) -> str:
        def _rsig(name):
            _map = {'conv_p': '_sig_conv_p_var', 'conv_pc': '_sig_conv_pc_var',
                    'test_window': '_sig_test_window_var',
                    'jitter_pc': '_sig_jitter_pc_var'}
            return bool(_render_cfg[_map[name]]) if _render_cfg else self._sig(name)

        def _rline(attr):
            return bool(_render_cfg[attr]) if _render_cfg else (
                getattr(self, attr).get() if getattr(self, attr, None) else False)

        def _racg(attr, default=None):
            return _render_cfg.get(attr, default) if _render_cfg else self._acg_var_get(attr, default)

        if highres is None:
            highres = self._highres_mode
        ref, tgt = int(inds[0]), int(inds[1])
        cd = _ccg_data_override if _ccg_data_override is not None else self.ccg_data
        conf = cd.conf

        d = self._resolve_segment_data(ref, tgt, segment, highres=highres,
                                        include_pval=True, include_acg=False, _cd=cd)
        ccg_raw = d['ccg_raw']
        ccg_null_raw = d['ccg_null_raw']
        pval_arg = d['pval']
        pval_c_arg = d['pval_corrected']
        seg_label = d['seg_label']
        window_size_eff = float(conf.duration)
        # Bin size for this resolution inferred from stored arrays (not conf.bin_size)
        try:
            _n0 = int(len(ccg_raw)) if ccg_raw is not None else 0
            bin_size_eff0 = float(conf.duration) / (_n0 - 1) if _n0 > 1 else float(conf.bin_size)
        except Exception:
            bin_size_eff0 = float(conf.bin_size)

        # ── Extend-window view (recompute this pair only) ────────────────
        _extend_on = bool(_racg('_extend_enable_var', False))
        _extend_ms = _racg('_extend_ms_var', 0)
        _extend_bin_ms = _racg('_extend_bin_ms_var', None)
        if _extend_on:
            try:
                if _extend_bin_ms is not None:
                    bin_size_eff_ext = max(1.0, float(_extend_bin_ms)) / 1000.0
                else:
                    bin_size_eff_ext = float(bin_size_eff0)
                ext = self._resolve_extended_ccg(
                    ref, tgt, segment, bool(highres),
                    int(_extend_ms),
                    float(bin_size_eff_ext),
                    cd,
                )
                if ext is not None:
                    ccg_raw = ext["ccg_raw"]
                    ccg_null_raw = ext["ccg_null_raw"]  # None if EranConv failed — never mix with old null
                    pval_arg = ext.get("pval")
                    pval_c_arg = None
                    seg_label = ext.get("seg_label", seg_label)
                    window_size_eff = float(ext["window_size_s"])
                    # Use the bin size actually produced by spike_correlations,
                    # not conf.bin_size — they may differ when extend resolution differs.
                    bin_size_eff0 = float(ext["bin_size_eff"])
            except Exception:
                pass

        if _render_cfg is not None and NormalizeBy is not None:
            _norms = {n for n in NormalizeBy if n.name in (_render_cfg.get('active_norms') or [])}
        else:
            _norms = self.active_norms
        _alpha = _render_cfg.get('active_alpha', self.active_alpha) if _render_cfg else self.active_alpha

        _custom_time_h = None
        if self._is_custom_segment(segment):
            _ci = self._custom_seg_index(segment)
            _custom_time_h = self._custom_segments[_ci].get('total_time_hours')

        norm_info = (', '.join(nm.name for nm in _norms)
                     if _norms and NormalizeBy is not None else None)

        # ── Baseline method & display flags ───────────────────────────
        method = (_render_cfg.get('_conn_str_method_var', 'conv')
                  if _render_cfg else self._conn_str_method_var.get())
        cs_show = (_render_cfg.get('_conn_str_show_var', False)
                   if _render_cfg else self._conn_str_show_var.get())

        # Effective test-window lags — resolved once here, forwarded to both
        # compute_ccg_panel_data and plot_ccg_panel (no independent calls).
        eff_min_lag, eff_max_lag = self._effective_lags(ref, tgt)
        _tw_active = _rsig('test_window')

        # ── ACG data (needed for 'tailed' CS and/or ACG overlays) ────
        show_acg_ref = _racg('_acg_ref_var', False)
        show_acg_tgt = _racg('_acg_tgt_var', False)
        _dref = bool(_racg('_acg_deconv_ref_var', False))
        _dtgt = bool(_racg('_acg_deconv_tgt_var', False))
        _need_acg = (method == 'tailed') or show_acg_ref or show_acg_tgt or _dref or _dtgt
        acg_ref_raw = acg_tgt_raw = None
        nspks_ref = nspks_tgt = 1.0
        if _need_acg:
            d_acg = self._resolve_segment_data(ref, tgt, segment,
                                               include_pval=False, include_acg=True, _cd=cd)
            acg_ref_raw = d_acg['acg_ref'].copy().astype(float)
            acg_tgt_raw = d_acg['acg_tgt'].copy().astype(float)
            nspks_ref = max(float(np.sum(acg_ref_raw)), 1.0)
            nspks_tgt = max(float(np.sum(acg_tgt_raw)), 1.0)

        # ── ACG deconvolution view (display-only): apply BEFORE CS/baseline ──
        # When enabled, the effective CCG/null are deconvolved and *then* all
        # downstream baseline + CS computations are performed on that signal.
        _deconv_active = (bool(_dref) or bool(_dtgt))
        if _deconv_active and acg_ref_raw is not None and acg_tgt_raw is not None:
            def _deconv_1d(x):
                if x is None:
                    return None
                if _dref and _dtgt:
                    return deconv_autocorr(
                        x.copy().astype(float),
                        acg_ref_raw, nspks_ref,
                        acg_tgt_raw, nspks_tgt,
                    )
                if _dref:
                    z = np.zeros_like(acg_tgt_raw, dtype=float)
                    return deconv_autocorr(x.copy().astype(float),
                                           acg_ref_raw, nspks_ref,
                                           z, 1.0)
                z = np.zeros_like(acg_ref_raw, dtype=float)
                return deconv_autocorr(x.copy().astype(float),
                                       z, 1.0,
                                       acg_tgt_raw, nspks_tgt)

            try:
                ccg_raw = _deconv_1d(ccg_raw)
                ccg_null_raw = _deconv_1d(ccg_null_raw)
            except Exception:
                pass

        # ── Single authoritative normalization + CS computation ───────
        # compute_ccg_panel_data applies norms (without BASELINE), computes
        # CS/baseline_1d on the pre-subtraction signal, then applies BASELINE
        # norm using the method's own baseline.  This is the only place where
        # any of these operations happen for rendering.
        if method in ('conv', 'tailed', 'global'):
            panel: CCGPanelData = compute_ccg_panel_data(
                ccg_raw, ccg_null_raw, conf, method,
                _norms, ref, tgt, segment, self.n_segments,
                self._is_custom_segment(segment), _custom_time_h,
                eff_min_lag, eff_max_lag,
                neurons=self.neurons, nd=self.cd.nd, nd_key=self.key.nd(),
                acg_ref=acg_ref_raw if method == 'tailed' else None,
                acg_tgt=acg_tgt_raw if method == 'tailed' else None,
                nspks_ref=nspks_ref, nspks_tgt=nspks_tgt,
            )
            ccg        = panel.ccg
            show_null  = panel.ccg_null
            baseline_1d = panel.baseline_1d
            cs_val      = panel.cs_val
            if not _deconv_active:
                # Populate cache so _update_conn_str_label can read the active method
                # without triggering a full _compute_pair_conn_strength call.
                self._conn_strength_cache[
                    self._cs_cache_key(ref, tgt, segment, method, highres,
                                       eff_min_lag, eff_max_lag)
                ] = (cs_val, baseline_1d)
        else:
            # 'jitter' — no normalization via compute_ccg_panel_data;
            # use raw conv null for display.
            norms_no_bl = _norms - {NormalizeBy.BASELINE}
            ccg, ccg_null = apply_norms_to_ccg(
                ccg_raw, ccg_null_raw, ref, tgt, segment,
                norms_no_bl, self.neurons, self.cd.nd,
                self.key.nd(), self.n_segments,
                self._is_custom_segment(segment),
                custom_time_hours=_custom_time_h)
            show_null   = ccg_null
            baseline_1d = None
            cs_val      = None

        n_bins = len(ccg)
        bin_size_eff = window_size_eff / (n_bins - 1) if n_bins > 1 else conf.bin_size
        # Store display-only computed result for later "View values" / CS label reads
        try:
            resk = 'hi' if bool(highres) else 'lo'
            self._display_pair_temp[(ref, tgt, int(segment), resk)] = {
                "deconv_ref": bool(_dref),
                "deconv_tgt": bool(_dtgt),
                "method": str(method),
                "ccg": ccg,
                "ccg_null": show_null,
                "baseline_1d": baseline_1d,
                "cs_val": cs_val,
                "bin_size_eff": float(bin_size_eff),
                "min_lag": float(eff_min_lag),
                "max_lag": float(eff_max_lag),
            }
        except Exception:
            pass

        # ── Jitter data ───────────────────────────────────────────────
        resk = 'hi' if bool(highres) else 'lo'
        _jseg = self._jitter_seg(segment)
        j_data = self._jitter_cache.get((ref, tgt, resk, _jseg))
        # Fallback: if viewing a specific real segment but only whole-session
        # jitter exists (seg=None), use it.
        if j_data is None and _jseg is not None:
            j_data = self._jitter_cache.get((ref, tgt, resk, None))
        j_ccg_arg = j_data[0] if j_data is not None else None
        j_pval_bins_arg = j_data[2] if j_data is not None and len(j_data) > 2 else None
        j_ccg_lo_arg = j_data[3] if j_data is not None and len(j_data) > 3 else None
        j_ccg_hi_arg = j_data[4] if j_data is not None and len(j_data) > 4 else None
        self._dbg_log(
            "H1",
            "ccg_ui.py:_render_png:jitter_lookup",
            "Jitter lookup before plot",
            {
                "highres": bool(highres),
                "method": str(method),
                "segment": int(segment),
                "j_key": [int(ref), int(tgt), "lo", self._jitter_seg(segment)],
                "len_ccg": int(len(ccg)) if ccg is not None else None,
                "len_j_ccg": int(len(j_ccg_arg)) if j_ccg_arg is not None else None,
                "len_j_pval": int(len(j_pval_bins_arg)) if j_pval_bins_arg is not None else None,
            },
        )

        # ── p-value overlays (method-dependent) ───────────────────────
        show_pval = show_pval_c = None
        show_j_ccg = show_j_pval = None
        show_j_ccg_lo = show_j_ccg_hi = None
        if method == 'conv':
            show_pval = pval_arg if _rsig('conv_p') else None
            show_pval_c = pval_c_arg if _rsig('conv_pc') else None
            if _rsig('conv_p') and pval_arg is None and _render_cfg is None:
                is_custom = self._is_custom_segment(segment)
                is_all = (segment == self.n_segments)
                reason = ('all-segments view' if is_all
                          else 'custom segment' if is_custom else 'cd.pval is None')
                print(f"[CCGReviewUI] p-value ON but unavailable ({ref},{tgt}) seg={segment}: {reason}")
        elif method == 'jitter':
            # Jitter overlays are only defined for low-res; for high-res panels do nothing.
            if highres:
                show_j_ccg = None
                show_j_pval = None
                show_j_ccg_lo = None
                show_j_ccg_hi = None
            else:
                show_j_ccg = j_ccg_arg
                show_j_pval = j_pval_bins_arg if _rsig('jitter_pc') else None
                show_j_ccg_lo = j_ccg_lo_arg
                show_j_ccg_hi = j_ccg_hi_arg
            self._dbg_log(
                "H2",
                "ccg_ui.py:_render_png:jitter_show",
                "Jitter overlay selection",
                {
                    "highres": bool(highres),
                    "show_j_ccg": bool(show_j_ccg is not None),
                    "show_j_pval": bool(show_j_pval is not None),
                    "len_ccg": int(len(ccg)) if ccg is not None else None,
                    "len_show_j_ccg": int(len(show_j_ccg)) if show_j_ccg is not None else None,
                },
            )

        def _fmt_cs_val(v):
            if v is None: return "n/a"
            x = float(v)
            return f"{x:.2f}" if abs(x) >= 1000 else f"{x:.3g}"

        # CS overlay: green bars showing excess above baseline in the test window.
        # baseline_1d is always derived from the pre-BASELINE-subtraction CCG.
        # When BASELINE norm is active panel.ccg is already baseline-subtracted
        # (sits at ~0), so the overlay bottom must also be zero — green bars then
        # show everything above zero within the window.
        # When BASELINE norm is off, use baseline_1d directly as the bar bottom.
        if cs_show:
            _has_baseline_norm = NormalizeBy.BASELINE in _norms
            if _has_baseline_norm and baseline_1d is not None:
                cs_baseline_arg = np.zeros(len(panel.ccg))
            else:
                cs_baseline_arg = baseline_1d
        else:
            cs_baseline_arg = None

        # Title should not include CS (user-requested). Keep CS visible via the CS label in the UI.
        seg_id_display = seg_label

        # ── ACG overlays ──────────────────────────────────────────────
        acg_ref = acg_tgt = None
        if show_acg_ref and acg_ref_raw is not None:
            acg_ref = acg_ref_raw
        if show_acg_tgt and acg_tgt_raw is not None:
            acg_tgt = acg_tgt_raw

        # Ref mean waveform on peak channel — only when rendering hi-res PNG
        wf_peak_ms = wf_peak_amp = None
        if (
            bool(highres)
            and _racg('_peak_wf_var', False)
            and self.neurons is not None
        ):
            wf_all = getattr(self.neurons, 'waveforms', None)
            pc = getattr(self.neurons, 'peak_channels', None)
            sids = getattr(self.neurons, 'shank_ids', None)
            if wf_all is not None and pc is not None and sids is not None:
                nd_conf = getattr(getattr(self.cd, 'nd', None), '_conf', None)
                ch_ps = int(getattr(nd_conf, 'ch_per_shank', 16) or 16)
                recinfo = getattr(nd_conf, 'recinfo', None) if nd_conf else None
                skipped = getattr(recinfo, 'skipped_channels', None) if recinfo else None
                discarded = (
                    None if skipped is None else np.asarray(skipped, dtype=int))
                try:
                    peak_ch = int(pc[ref])
                    rs = int(sids[ref])
                except (IndexError, TypeError, ValueError):
                    peak_ch = None
                disc_set = (
                    set(int(x) for x in discarded.ravel())
                    if discarded is not None and discarded.size else set())
                if peak_ch is not None and peak_ch not in disc_set:
                    local_idx = peak_ch - ch_ps * rs
                    if 0 <= local_idx < ch_ps:

                        def _filled_wf(shank_id, wf_neuron):
                            if wf_neuron.ndim == 1:
                                return np.tile(wf_neuron, (ch_ps, 1))
                            sid = int(shank_id)
                            channel_ids = ch_ps * sid + np.arange(ch_ps)
                            if discarded is None:
                                mask = np.ones(ch_ps, dtype=bool)
                                start = int(ch_ps * sid)
                                length = int(np.sum(mask))
                            else:
                                mask = ~np.isin(channel_ids, discarded)
                                start = int(
                                    ch_ps * sid
                                    - np.sum(discarded < ch_ps * sid))
                                length = int(np.sum(mask))
                            clean = np.full((ch_ps, wf_neuron.shape[-1]), np.nan)
                            clean[mask] = wf_neuron[start:start + length]
                            return clean

                        ref_full = _filled_wf(rs, wf_all[ref])
                        tr = ref_full[local_idx]
                        if np.any(np.isfinite(tr)):
                            fs = float(
                                getattr(self.neurons, 'sampling_rate', 1) or 30000.0)
                            if not np.isfinite(fs) or fs <= 0:
                                fs = 30000.0
                            n = int(tr.shape[0])
                            ctr = n // 2
                            wf_peak_ms = (np.arange(n, dtype=float) - ctr) / fs * 1000.0
                            wf_peak_amp = np.asarray(tr, dtype=float)

        ids = (self._shank_label(ref), self._shank_label(tgt))
        png_path = self._png_path(inds, segment, _render_cfg=_render_cfg,
                                  _hires_override=highres)

        fig = Figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        _show_bl = bool(_racg('_baseline_show_var', True))
        ccg_null_plot = show_null if _show_bl else None

        nt = (self.neurons.neuron_type[ref], self.neurons.neuron_type[tgt])
        ov = getattr(self, '_export_overrides', None) or {}
        _min_lag_plot = eff_min_lag if (_tw_active or cs_show) else None
        _max_lag_plot = eff_max_lag if (_tw_active or cs_show) else None
        plot_ccg.plot_ccg_panel(
            ax=ax, ccg=ccg, ids=ids, inds=inds,
            neuron_type=nt,
            window_size=window_size_eff, bin_size=bin_size_eff,
            pval=show_pval, pval_corrected=show_pval_c,
            alpha=_alpha, ccg_null=ccg_null_plot,
            j_ccg=show_j_ccg, j_pval=show_j_pval,
            segment_id=seg_id_display,
            is_significant_pair=self._is_significant(ref, tgt, segment),
            min_lag=_min_lag_plot,
            max_lag=_max_lag_plot,
            normalize_info=norm_info,
            acg_ref=acg_ref, acg_tgt=acg_tgt,
            acg_yscale_ref=_racg('_acg_yscale_ref_var', 1.0),
            acg_yscale_tgt=_racg('_acg_yscale_tgt_var', 1.0),
            acg_match_ccg=_racg('_acg_match_ccg_var', False),
            show_ccg=_racg('_ccg_show_var', True),
            line_ccg=_rline('_line_ccg_var'),
            line_baseline=_rline('_line_baseline_var'),
            line_ref=_rline('_line_ref_var'),
            line_tgt=_rline('_line_tgt_var'),
            line_jitter=_rline('_line_jitter_var'),
            conn_strength_baseline=cs_baseline_arg,
        )

        # Extend view: override x ticks to keep labels readable on long windows.
        if bool(_racg('_extend_enable_var', False)):
            try:
                half_ms = float(window_size_eff) * 1000.0 / 2.0
                # Choose a "nice" step such that we draw ~<=11 major ticks.
                nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
                step = nice[-1]
                for s in nice:
                    if (2 * half_ms / s) <= 10.5:
                        step = s
                        break
                start = -np.floor(half_ms / step) * step
                ticks = np.arange(start, half_ms + 0.5 * step, step)
                # Always include 0
                if not np.any(np.isclose(ticks, 0.0)):
                    ticks = np.sort(np.append(ticks, 0.0))
                ax.set_xticks(ticks)
            except Exception:
                pass

        # Jitter confidence band (95% interval) drawn here (keeps plotting/ccg.py stable).
        if show_j_ccg_lo is not None and show_j_ccg_hi is not None:
            try:
                jlo = np.asarray(show_j_ccg_lo, dtype=float)
                jhi = np.asarray(show_j_ccg_hi, dtype=float)
                if len(jlo) == len(ccg) and len(jhi) == len(ccg):
                    bins_s = np.arange(-window_size_eff / 2, window_size_eff / 2 + bin_size_eff, bin_size_eff)
                    bins = bins_s * 1000.0
                    bin_w = bin_size_eff * 1000.0
                    edges = np.append(bins - bin_w / 2, bins[-1] + bin_w / 2)
                    x_step = np.repeat(edges, 2)[1:-1]
                    for arr in (jlo, jhi):
                        ax.plot(
                            x_step,
                            np.repeat(arr, 2),
                            color="#C62828",
                            linewidth=1.15,
                            alpha=0.9,
                            linestyle=(0, (4, 3)),
                            zorder=4,
                        )
            except Exception:
                pass
        ylim = self._get_current_scale_ylim(ref, tgt)
        if ylim is not None:
            ax.set_ylim(ylim)

        fig.savefig(png_path, dpi=100, bbox_inches='tight')
        matplotlib.pyplot.close(fig)
        return png_path

    # ------------------------------------------------------------------
    # Plot update
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # .history/ backup and periodic autosave
    # ------------------------------------------------------------------

    _HISTORY_SUBDIR = '.history'
    _AUTOSAVE_INTERVAL_MS = 15 * 60 * 1000  # 15 minutes

    def _history_dir(self) -> str:
        return os.path.join(self._sel_save_dir, self._HISTORY_SUBDIR)

    def _git_commit_paths(self, paths: list, msg: str):
        """Stage and commit one or more files to git (background thread)."""
        import threading, subprocess
        repo = os.path.abspath(os.path.join(self._sel_save_dir, '..', '..'))
        rels = [os.path.relpath(p, repo) for p in paths]
        def _run():
            try:
                for rel in rels:
                    subprocess.run(['git', 'add', rel], cwd=repo,
                                   capture_output=True)
                subprocess.run(['git', 'commit', '--no-gpg-sign', '-m', msg],
                               cwd=repo, capture_output=True)
            except Exception as exc:
                print(f"[CCGReviewUI] git commit failed: {exc}")
        threading.Thread(target=_run, daemon=True).start()

    def _save_to_history(self, data: dict, suffix: str) -> str:
        """Write data dict to .history/{session}__{ts}{suffix}.json and git-commit."""
        hdir = self._history_dir()
        os.makedirs(hdir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        sess = getattr(self.key, 'session', 'sess')
        fname = f"{sess}__{ts}{suffix}.json"
        path = os.path.join(hdir, fname)
        self._atomic_write_json(path, data)
        self._git_commit_paths([path], f'[history] {fname}')
        return path

    def _save_autosnapshot(self):
        """Periodic 15-min autosave to .history/ as .autosaved.json."""
        if self.ccg_pointer is None or getattr(self, '_closing', False):
            return
        try:
            data = self._build_save_dict(
                datetime.datetime.now().isoformat(), 'autosaved')
            self._save_to_history(data, '.autosaved')
            print(f"[CCGReviewUI] autosnapshot saved")
        except Exception as exc:
            print(f"[CCGReviewUI] autosnapshot failed: {exc}")

    def _schedule_autosnapshot(self):
        def _do():
            self._save_autosnapshot()
            self.root.after(self._AUTOSAVE_INTERVAL_MS, _do)
        self.root.after(self._AUTOSAVE_INTERVAL_MS, _do)

    def _purge_history(self):
        """Delete .history/ files older than 3 days and commit the deletion."""
        hdir = self._history_dir()
        if not os.path.isdir(hdir):
            return
        import subprocess
        cutoff = datetime.datetime.now() - datetime.timedelta(days=3)
        removed = []
        for fname in os.listdir(hdir):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(hdir, fname)
            try:
                if datetime.datetime.fromtimestamp(os.path.getmtime(fpath)) < cutoff:
                    os.remove(fpath)
                    removed.append(fpath)
            except OSError:
                pass
        if removed:
            repo = os.path.abspath(os.path.join(self._sel_save_dir, '..', '..'))
            for p in removed:
                subprocess.run(['git', 'rm', '--cached', '-f',
                                os.path.relpath(p, repo)],
                               cwd=repo, capture_output=True)
            subprocess.run(['git', 'commit', '--no-gpg-sign', '-m',
                            f'[history] purge {len(removed)} files older than 3 days'],
                           cwd=repo, capture_output=True)
            print(f"[CCGReviewUI] purged {len(removed)} history files older than 3 days")

    def _purge_tmp_png_cache(self, days: int = 3):
        """Delete cached PNGs in tmp_dir older than *days* days."""
        if not getattr(self, 'tmp_dir', None) or not os.path.isdir(self.tmp_dir):
            return
        try:
            days = int(days)
        except Exception:
            days = 3
        cutoff = datetime.datetime.now() - datetime.timedelta(days=max(0, days))
        for fn in os.listdir(self.tmp_dir):
            if not fn.endswith('.png'):
                continue
            path = os.path.join(self.tmp_dir, fn)
            try:
                if datetime.datetime.fromtimestamp(os.path.getmtime(path)) < cutoff:
                    os.remove(path)
            except OSError:
                pass

    def _autosave_current(self):
        """Silently save current session's selections + groups as 'latest'.

        Called before any operation that would overwrite self._groups or
        self.selected_inds (session switch, GUI close).
        """
        if getattr(self, '_session_any_mode', False):
            try:
                self._autosave_all_sessions_for_current_type()
            except Exception as exc:
                print(f"[CCGReviewUI] any-session autosave failed: {exc}")
            try:
                if getattr(self, '_groups', None):
                    self._save_groups_export()
            except Exception:
                traceback.print_exc()
            try:
                self._save_ui_state()
            except Exception:
                traceback.print_exc()
            return
        if self.ccg_pointer is None:
            print("[CCGReviewUI] autosave skipped: ccg_pointer is None (data not yet loaded)")
            return
        ok = self._save_all_state('latest', silent=True)
        if not ok:
            print("[CCGReviewUI] autosave failed")

    def _save_groups_export(self):
        """Write groups_export.json (v4.0): registry + cross-session pair assignments.

        CRITICAL: Only updates the CURRENT session's pair assignments; all other
        sessions' data is read from the existing file and preserved.  This prevents
        one session's save from corrupting another session's group assignments.
        """
        self._sync_registry_from_groups()
        export_path = os.path.join(self._sel_save_dir, 'groups_export.json')
        cur_sess = self._current_session_str()

        # Load existing file to preserve other sessions' data
        existing_pairs: dict = {}  # gid_str → {sess → [[r,t],...]}
        if os.path.isfile(export_path):
            try:
                with open(export_path, encoding='utf-8') as f:
                    existing = json.load(f)
                if existing.get('version', '3.x') >= '4.0':
                    existing_pairs = existing.get('group_pairs', {})
                else:
                    # v3.x: migrate pair assignments into existing_pairs keyed by int IDs
                    for gname, val in existing.get('groups', {}).items():
                        gid = self._ensure_group_registered(gname)
                        gid_str = str(gid)
                        existing_pairs.setdefault(gid_str, {})
                        if isinstance(val, dict):
                            for sess, pp in val.items():
                                if sess != cur_sess:  # only preserve OTHER sessions
                                    existing_pairs[gid_str][sess] = pp
            except Exception as exc:
                print(f"[CCGReviewUI] _save_groups_export: failed to read existing: {exc}")

        # Build new group_pairs: preserve other sessions, update current session only
        group_pairs: dict = {}
        for gid, g_entry in self._group_registry.items():
            gname = g_entry['name']
            gid_str = str(gid)
            g_pairs: dict = {}
            # Copy other sessions from existing file
            for sess, pp in existing_pairs.get(gid_str, {}).items():
                if sess != cur_sess:
                    g_pairs[sess] = pp
            # Write current session
            cur_pairs = sorted(self._group_pairs(gname, cur_sess))
            if cur_pairs:
                g_pairs[cur_sess] = [[int(r), int(c)] for r, c in cur_pairs]
            if g_pairs:
                group_pairs[gid_str] = g_pairs

        data = {
            'version': '4.0',
            'group_registry': {str(k): v for k, v in self._group_registry.items()},
            'next_id': self._next_group_id,
            'group_pairs': group_pairs,
        }
        self._atomic_write_json(export_path, data)

    def _autoload_session_latest(self, restore_groups: bool = False):
        """Load the 'latest' selection file for the current session, if it exists.

        By default only restores pair selections — groups are shared across
        sessions and should not be overwritten on session switch.  Pass
        restore_groups=True on first launch to seed groups from the file.

        When restore_groups=True, groups are loaded from the central
        ``groups_export.json`` (written on every save) rather than from
        the per-session ``__latest.json`` — since per-session files only
        contain a snapshot of groups at the time *that* session was saved,
        which can be stale for other sessions' group entries.
        """
        latest_path = self._sel_version_path('latest')
        if not os.path.isfile(latest_path):
            # Even without a session file, try to load groups
            if restore_groups:
                self._load_groups_from_export()
            return
        try:
            # Always load pair selections from the session-specific file;
            # never load groups from it (they may be stale).
            self._load_selection_from_file(latest_path,
                                           restore_groups=False,
                                           _skip_redraw=True)
        except Exception as exc:
            print(f"[CCGReviewUI] failed to autoload latest: {exc}")
        if restore_groups:
            self._load_groups_from_export()

    def _load_groups_from_export(self):
        """Load group registry + all-session pair assignments from groups_export.json.

        Handles both v4.0 (registry + group_pairs) and v3.x (legacy string-name groups).
        Falls back to per-session __latest.json if the export file doesn't exist yet.
        """
        export_path = os.path.join(self._sel_save_dir, 'groups_export.json')
        if not os.path.isfile(export_path):
            latest_path = self._sel_version_path('latest')
            if os.path.isfile(latest_path):
                try:
                    with open(latest_path, encoding='utf-8') as f:
                        data = json.load(f)
                    self._restore_groups_from_data(data)
                except Exception as exc:
                    print(f"[CCGReviewUI] failed to load groups from session file: {exc}")
            return
        try:
            with open(export_path, encoding='utf-8') as f:
                data = json.load(f)
            version = data.get('version', '3.x')
            if version >= '4.0':
                self._load_groups_v4(data)
            else:
                # v3.x → migrate
                self._restore_groups_from_data(data, restore_hotkeys=True)
                self._sync_registry_from_groups()
                # Rewrite as v4.0 immediately so future loads use new format
                try:
                    self._save_groups_export()
                    print("[CCGReviewUI] groups_export.json migrated to v4.0")
                except Exception as exc:
                    print(f"[CCGReviewUI] migration save failed: {exc}")
            n_groups = len(self._groups)
            n_pairs = sum(len(p) for sd in self._groups.values()
                          if isinstance(sd, dict) for p in sd.values())
            print(f"[CCGReviewUI] groups loaded: {n_groups} groups, "
                  f"{n_pairs} pair-session entries")
            # Refresh any UI that depends on group/hotkey state.
            try:
                self._rebuild_groups_menu()
            except Exception:
                pass
            try:
                if hasattr(self, '_hotkeys_bar'):
                    self._refresh_hotkeys_bar()
            except Exception:
                pass
        except Exception as exc:
            print(f"[CCGReviewUI] failed to load groups_export.json: {exc}")

    def _load_groups_v4(self, data: dict):
        """Populate _group_registry, _groups, _group_hotkeys, _group_notes from v4.0 data."""
        registry = data.get('group_registry', {})
        self._group_registry = {}
        for k, v in registry.items():
            self._group_registry[int(k)] = v
        self._next_group_id = data.get('next_id', max(self._group_registry.keys(), default=0) + 1)

        # Rebuild _group_hotkeys and _group_notes from registry
        self._group_hotkeys = {}
        self._group_notes = {}
        for gid, g in self._group_registry.items():
            name = g['name']
            if g.get('hotkey'):
                self._group_hotkeys[name] = g['hotkey']
            if g.get('notes'):
                self._group_notes[name] = g['notes']

        # Populate _groups from group_pairs
        group_pairs = data.get('group_pairs', {})
        for gid_str, sessions_dict in group_pairs.items():
            try:
                gid = int(gid_str)
            except ValueError:
                continue
            gname = self._group_registry.get(gid, {}).get('name')
            if not gname:
                continue
            self._groups.setdefault(gname, {})
            for sess, pairs in sessions_dict.items():
                self._groups[gname][sess] = set(
                    tuple(int(v) for v in p) for p in pairs)
        self._groups.setdefault(_ADMITTED_GROUP, {})
        self._sync_sel_data()

    def _merge_groups_from_session_files(self, export_path: str):
        """Merge group definitions from all per-session __latest.json files.

        Adds any group names (+ their pair assignments) that exist in per-session
        files but are missing from the already-loaded self._groups.  Does NOT
        overwrite existing entries in self._groups (export file is authoritative
        for groups that already exist there).  Saves the export file if anything
        was added.
        """
        import glob as _glob
        save_dir = os.path.dirname(export_path)
        added = False
        for fpath in sorted(_glob.glob(os.path.join(save_dir, '*__latest.json'))):
            try:
                with open(fpath, encoding='utf-8') as fh:
                    fdata = json.load(fh)
            except Exception:
                continue
            file_session = fdata.get('session',
                                     os.path.basename(fpath).replace('__latest.json', ''))
            for g, val in fdata.get('groups', {}).items():
                if isinstance(val, list):
                    pairs = set(tuple(int(v) for v in p) for p in val)
                    sess_pairs = {file_session: pairs}
                elif isinstance(val, dict):
                    sess_pairs = {s: set(tuple(int(v) for v in p) for p in pp)
                                  for s, pp in val.items()}
                else:
                    continue
                if g not in self._groups:
                    self._groups[g] = sess_pairs
                    added = True
                else:
                    for sess, pairs in sess_pairs.items():
                        if sess not in self._groups[g] and pairs:
                            self._groups[g][sess] = pairs
                            added = True
        if added:
            self._save_groups_export()
            print("[CCGReviewUI] groups_export.json updated with groups from per-session files")

    def _restore_groups_from_data(self, data: dict, restore_hotkeys: bool = False):
        """Merge group data from a dict into self._groups (v3.x legacy path).

        CRITICAL: Merges — never overwrites existing entries — so loading a
        per-session file cannot erase groups belonging to other sessions.
        Pass restore_hotkeys=True only when loading from groups_export.json.
        """
        raw_groups = data.get('groups', {})
        file_session = data.get('session', self._current_session_str())
        for g, val in raw_groups.items():
            if isinstance(val, list):
                pairs = {file_session: set(tuple(int(v) for v in p) for p in val)}
            elif isinstance(val, dict):
                pairs = {sess: set(tuple(int(v) for v in p) for p in pp)
                         for sess, pp in val.items()}
            else:
                pairs = {}
            if g not in self._groups:
                self._groups[g] = pairs
            else:
                # Merge: only add sessions that don't already exist
                for sess, sp in pairs.items():
                    if sess not in self._groups[g]:
                        self._groups[g][sess] = sp
        self._groups.setdefault(_ADMITTED_GROUP, {})
        if restore_hotkeys:
            hk = data.get('hotkeys', {})
            self._group_hotkeys.update(hk)
        notes = data.get('notes', {})
        for k, v in notes.items():
            if k not in self._group_notes:
                self._group_notes[k] = v
        self._rebuild_groups_menu()

    def _ensure_session_loaded(self, nd_key, on_loaded):
        """Call on_loaded() immediately if data for nd_key is present.

        Otherwise show a modal "Loading dataset…" dialog, run cd.get_ccg()
        in a background thread, then dismiss the dialog and call on_loaded().
        """
        if nd_key in getattr(self.cd, '_ccg', {}):
            on_loaded()
            return
        # Build a non-closeable modal dialog
        dlg = tk.Toplevel(self.root)
        dlg.title("Loading")
        dlg.geometry("300x80")
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.protocol('WM_DELETE_WINDOW', lambda: None)  # prevent manual close
        ttk.Label(dlg, text="Loading dataset…", anchor='center').pack(
            expand=True, fill='both', padx=20, pady=20)
        self.root.update_idletasks()

        result = {}

        def _worker():
            try:
                self.cd.get_ccg()
            except Exception as ex:
                result['error'] = str(ex)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        def _poll():
            if t.is_alive():
                self.root.after(200, _poll)
            else:
                dlg.destroy()
                if 'error' in result:
                    messagebox.showerror("Load error", result['error'],
                                         parent=self.root)
                else:
                    on_loaded()

        self.root.after(200, _poll)

    def _finish_initial_draw(self):
        """Complete initialisation after data has been loaded; then draw."""
        nd_key = self.key.nd()
        ptr = self.cd.data.get(self.key)
        if ptr is None:
            # Key may not exist yet; pick first available key for this session
            type_keys = self._available_type_keys(nd_key)
            if type_keys:
                self.key = type_keys[0]
                ptr = self.cd.data.get(self.key)
        if ptr is None:
            messagebox.showerror("Load error",
                                 f"No data found for session {nd_key}",
                                 parent=self.root)
            return
        self.ccg_pointer = ptr
        self.ccg_data = self.cd._ccg.get(nd_key)
        self.neurons = (self.cd.nd.data[nd_key]
                        if getattr(self.cd, 'nd', None) is not None else None)
        self.n_segments = self.ccg_pointer.n_segments
        self.segment_names = list(self.ccg_pointer.edge_times['label'].values)
        # Restore selection state from pointer if available
        if (hasattr(self.ccg_pointer, 'manually_selected_inds')
                and self.ccg_pointer.manually_selected_inds is not None):
            self.selected_inds = set(
                map(tuple, self.ccg_pointer.manually_selected_inds))
        else:
            self.selected_inds = set()
        self._pair_deleted_store.clear()
        self.deleted_inds = set()
        self.unselected_inds = set(map(tuple, self.all_inds)) - self.selected_inds
        # Load persisted jitter results (try disk first, then in-memory)
        if hasattr(self.cd, 'load_jitter') and not self.cd._jitter_results:
            self.cd.load_jitter()
        self._load_jitter_from_cd()
        self._post_load_refresh()
        # Startup: purge old history + tmp PNG cache, then start 15-min autosnapshot timer
        try:
            self._purge_history()
            self._purge_tmp_png_cache(days=3)
        except Exception as exc:
            print(f"[CCGReviewUI] history purge failed: {exc}")
        self._schedule_autosnapshot()

    def _post_load_refresh(self):
        """Unified UI refresh called after any load (initial, autoload, manual).

        Keeps all three code paths (launch, autoload-latest, load-selection)
        visually in sync.
        """
        self.refresh_lists()
        self._build_sig_chips()
        self._update_segment_label()
        self._refresh_net_shank_buttons()
        self._rebuild_groups_menu()   # also refreshes hotkeys bar chips
        self._update_conn_str_metric_availability()
        self.update_plot()
        self._draw_network()

    # ------------------------------------------------------------------
    # Connectivity Strength
    # ------------------------------------------------------------------

    def _build_main_template(self):
        """Build the 'Main' template from the current CCGConfig and inject into _templates."""
        try:
            if self.ccg_data is not None and _ccgconfig_to_main_template is not None:
                self._main_template = _ccgconfig_to_main_template(self.ccg_data.conf)
                self._templates['Main'] = self._main_template
        except Exception as e:
            print(f"[CCGReviewUI] Could not build Main template: {e}")
            self._main_template = None

    def _pair_passes_main(self, ref: int, tgt: int, seg) -> bool:
        """
        Returns True if this pair passes the primary 'Main' template criterion.
        Uses EranConv p-value: any bin < alpha within [min_lag_bin, max_lag_bin].
        Falls back to True if no p-value data available.
        """
        cd = self.ccg_data
        if cd is None or cd.pval is None:
            return True  # no data to filter on
        is_custom = self._is_custom_segment(seg)
        is_all = (seg == self.n_segments)
        try:
            conf = cd.conf
            lo = conf.min_lag_bin
            hi = conf.max_lag_bin
            alpha = conf.alpha
            if is_custom or is_all:
                return True  # skip graying for special segments
            pval = cd.pval[seg, ref, tgt, lo:hi]
            return bool(np.any(pval < alpha))
        except Exception:
            return True


    def _on_conn_str_toggle(self):
        """CS overlay toggled — PNG must be re-rendered with/without green bars."""
        self._clear_all_png_cache()
        self.update_plot()
        self._update_conn_str_label()

    def _update_conn_str_label(self):
        """Always show CS value regardless of overlay toggle."""
        if not hasattr(self, '_conn_str_label'):
            return
        method = self._conn_str_method_var.get()
        try:
            inds = self._current_inds()
            if inds is None:
                self._conn_str_label.config(text="CS: \u2014")
                return
            ref, tgt = int(inds[0]), int(inds[1])
            seg = self.current_segment
            eff_min_lag, eff_max_lag = self._effective_lags(ref, tgt)
            metric = (self._conn_str_metric_var.get()
                      if hasattr(self, '_conn_str_metric_var') else 'STG')

            def _fmt_cs(v):
                if v is None: return "n/a"
                x = float(v)
                if getattr(self, '_conn_str_nonneg_var', None) is not None and self._conn_str_nonneg_var.get():
                    x = max(x, 0.0)
                return f"{x:.2f}" if abs(x) >= 1000 else f"{x:.3g}"

            def _get_cs(hr):
                if metric == 'STG':
                    # If ACG deconvolution is active, CS/baseline are display-derived
                    # and should be recomputed on the deconvolved signal.
                    if (hasattr(self, '_acg_deconv_ref_var') and hasattr(self, '_acg_deconv_tgt_var')
                            and (self._acg_deconv_ref_var.get() or self._acg_deconv_tgt_var.get())):
                        try:
                            seg_eff = self.current_segment
                            resk = 'hi' if bool(hr) else 'lo'
                            tmp = self._display_pair_temp.get((ref, tgt, int(seg_eff), resk))
                            if tmp is not None and tmp.get("method") == method:
                                return _fmt_cs(tmp.get("cs_val"))
                        except Exception:
                            pass
                    k = self._cs_cache_key(ref, tgt, seg, method, hr,
                                           eff_min_lag, eff_max_lag)
                    if k not in self._conn_strength_cache:
                        self._compute_pair_conn_strength(ref, tgt, seg, highres=hr)
                    v, _ = self._conn_strength_cache.get(k, (None, None))
                    return _fmt_cs(v)

                # JBSI: requires jitter mean CCG (low-res only).
                if hr:
                    return "n/a"
                try:
                    from neuropy.analyses.jitter import compute_jbsi, JitterConfig
                except Exception:
                    return "n/a"
                # Jitter cache key uses low-res only
                _jseg = self._jitter_seg(seg)
                jk = (ref, tgt, 'lo', _jseg)
                entry = self._jitter_cache.get(jk)
                if entry is None and _jseg is not None:
                    entry = self._jitter_cache.get((ref, tgt, 'lo', None))
                if entry is None:
                    return "n/a"
                j_avg = entry[0]
                if j_avg is None:
                    return "n/a"
                # Real ccg (raw counts) for this segment (low-res)
                d = self._resolve_segment_data(ref, tgt, seg, highres=False, include_pval=False)
                real_ccg = d.get('ccg_raw')
                if real_ccg is None:
                    return "n/a"

                # Firing rates: segment FR if available, otherwise whole-session
                fr_ref = fr_tgt = None
                nd_key = self.key.nd() if self.key else None
                if (nd_key is not None and self.cd.nd is not None
                        and seg != self.n_segments and not self._is_custom_segment(seg)):
                    seg_fr = self.cd.nd.segment_firing_rates.get(nd_key)
                    if seg_fr is not None and seg < seg_fr.shape[0]:
                        fr_ref = float(seg_fr[seg, ref])
                        fr_tgt = float(seg_fr[seg, tgt])
                if fr_ref is None or fr_tgt is None:
                    fr = getattr(self.neurons, 'firing_rate', None)
                    if fr is not None:
                        fr_ref = float(fr[ref])
                        fr_tgt = float(fr[tgt])
                if fr_ref is None or fr_tgt is None:
                    return "n/a"

                # Match worker defaults: jscale from default JitterConfig
                try:
                    lo_cd = self.cd._ccg.get(self.key.nd()) if hasattr(self.cd, '_ccg') else self.ccg_data
                    jscale = float(JitterConfig(ccg=lo_cd.conf, njitter=1).jscale)
                except Exception:
                    jscale = 5e-3

                jbsi = compute_jbsi(
                    real_ccg=real_ccg,
                    j_ccg_avg=j_avg,
                    fr_ref=fr_ref,
                    fr_tgt=fr_tgt,
                    bin_size=float(self.ccg_data.conf.bin_size),
                    jscale=jscale,
                )
                # Scalar = sum in effective test window
                n_bins = len(jbsi)
                bin_size_eff = self.ccg_data.conf.duration / (n_bins - 1) if n_bins > 1 else self.ccg_data.conf.bin_size
                center = n_bins // 2
                lo = max(0, center + int(eff_min_lag / bin_size_eff))
                hi_bin = min(n_bins, center + int(eff_max_lag / bin_size_eff) + 1)
                return _fmt_cs(float(np.sum(jbsi[lo:hi_bin])))

            if self._sbs_mode:
                lo = _get_cs(False)
                hi = _get_cs(True)
                nn = "  non-neg" if self._conn_str_nonneg_var.get() else ""
                self._conn_str_label.config(text=f"CS: lo|hi = {lo}|{hi}{nn}")
            else:
                nn = "  non-neg" if self._conn_str_nonneg_var.get() else ""
                self._conn_str_label.config(text=f"CS: {_get_cs(self._highres_mode)}{nn}")
        except Exception:
            self._conn_str_label.config(text="CS: err")

    def _update_conn_str_metric_availability(self):
        """Enable/disable CS metric radio buttons (e.g., JBSI needs jitter)."""
        rbs = getattr(self, "_cs_metric_rbs", None) or {}
        jb = rbs.get("JBSI")
        if jb is None:
            return
        inds = self._current_inds()
        if inds is None:
            has_j = False
        else:
            ref, tgt = int(inds[0]), int(inds[1])
            seg = self.current_segment
            # JBSI depends on low-res jitter mean; accept segment-specific or whole-session
            k = (ref, tgt, "lo", self._jitter_seg(seg))
            has_j = (k in self._jitter_cache) or ((ref, tgt, "lo", None) in self._jitter_cache)
        try:
            jb.state(["!disabled"] if has_j else ["disabled"])
        except Exception:
            pass
        if not has_j and getattr(self, "_conn_str_metric_var", None) is not None:
            if str(self._conn_str_metric_var.get()) == "JBSI":
                self._conn_str_metric_var.set("STG")

    def _current_inds(self):
        """Return current (ref, tgt) inds or None."""
        if self._focused_pair is not None:
            return np.array(self._focused_pair)
        if self.current_pair_idx < len(self.all_inds):
            return self.all_inds[self.current_pair_idx]
        return None

    def _cs_cache_key(self, ref: int, tgt: int, seg, method: str, highres: bool,
                      eff_min_lag, eff_max_lag) -> tuple:
        """Return the canonical cache key for a CS result.

        Effective lags are included so that switching the adaptive test window
        on/off (which changes eff_min_lag/eff_max_lag) always produces a
        distinct key even without clearing the cache.
        """
        return (ref, tgt, seg, method, highres, eff_min_lag, eff_max_lag)

    def _compute_pair_conn_strength(self, ref: int, tgt: int, seg, highres: bool = False):
        """Compute CS for all applicable methods and populate _conn_strength_cache.

        Normalises the CCG once (without NormalizeBy.BASELINE — that is a
        display-only operation handled by compute_ccg_panel_data / _render_png)
        and computes conv / global / tailed on the same normalised signal.
        Cache keys include effective test-window lags so stale entries from
        a different adaptive-TW state are never reused.
        """
        cd = self.ccg_data
        conf = cd.conf

        # ── Resolve raw arrays ────────────────────────────────────────────
        d = self._resolve_segment_data(ref, tgt, seg, highres=highres,
                                        include_pval=False, include_acg=False)
        ccg_raw = d['ccg_raw']
        ccg_null_raw = d['ccg_null_raw']

        _custom_time_h = None
        if self._is_custom_segment(seg):
            _ci = self._custom_seg_index(seg)
            _custom_time_h = self._custom_segments[_ci].get('total_time_hours')

        # ── Single normalisation pass (without BASELINE) ──────────────────
        # BASELINE norm is a display transform only; CS scalars must be
        # computed on the pre-subtraction signal so that global/tailed
        # baselines reflect the actual signal amplitude.
        norms_no_bl = self.active_norms - {NormalizeBy.BASELINE}
        ccg, ccg_null = apply_norms_to_ccg(
            ccg_raw, ccg_null_raw, ref, tgt, seg,
            norms_no_bl, self.neurons, self.cd.nd,
            self.key.nd(), self.n_segments, self._is_custom_segment(seg),
            custom_time_hours=_custom_time_h)

        # ── Effective test-window lags (adaptive TW aware) ────────────────
        eff_min_lag, eff_max_lag = self._effective_lags(ref, tgt)
        _kw = dict(min_lag_override=eff_min_lag, max_lag_override=eff_max_lag)

        # ── conv ─────────────────────────────────────────────────────────
        cs, bl = compute_pair_conn_strength_1d(ccg, ccg_null, conf, 'conv', **_kw)
        self._conn_strength_cache[
            self._cs_cache_key(ref, tgt, seg, 'conv', highres, eff_min_lag, eff_max_lag)
        ] = (cs, bl)

        # ── global (max outside detection window) ─────────────────────────
        cs, bl = compute_pair_conn_strength_1d(ccg, ccg_null, conf, 'global', **_kw)
        self._conn_strength_cache[
            self._cs_cache_key(ref, tgt, seg, 'global', highres, eff_min_lag, eff_max_lag)
        ] = (cs, bl)

        # ── tailed (ACG deconvolution + tail baseline) ────────────────────
        try:
            d_acg = self._resolve_segment_data(ref, tgt, seg, highres=highres,
                                                include_pval=False, include_acg=True)
            acg_ref = d_acg['acg_ref'].copy().astype(float)
            acg_tgt = d_acg['acg_tgt'].copy().astype(float)
            nspks_ref = max(float(np.sum(acg_ref)), 1.0)
            nspks_tgt = max(float(np.sum(acg_tgt)), 1.0)
            cs, bl = compute_pair_conn_strength_1d(
                ccg, ccg_null, conf, 'tailed',
                acg_ref=acg_ref, acg_tgt=acg_tgt,
                nspks_ref=nspks_ref, nspks_tgt=nspks_tgt, **_kw)
            self._conn_strength_cache[
                self._cs_cache_key(ref, tgt, seg, 'tailed', highres, eff_min_lag, eff_max_lag)
            ] = (cs, bl)
        except Exception as e:
            print(f"[CCGReviewUI] Tailed CS failed for ({ref},{tgt}): {e}")
            self._conn_strength_cache[
                self._cs_cache_key(ref, tgt, seg, 'tailed', highres, eff_min_lag, eff_max_lag)
            ] = (None, None)

        method = self._conn_str_method_var.get()
        return self._conn_strength_cache.get(
            self._cs_cache_key(ref, tgt, seg, method, highres, eff_min_lag, eff_max_lag),
            (None, None))

    def _update_global_baseline_availability(self):
        """Enable/disable the Global radio button.

        Global is always available whenever CCG data exists.
        """
        rb = getattr(self, '_global_rb', None)
        if rb is None:
            return
        available = (self.ccg_data is not None)
        try:
            rb.state(['!disabled'] if available else ['disabled'])
        except Exception:
            pass
        if not available and self._conn_str_method_var.get() == 'global':
            self._conn_str_method_var.set('conv')
            self._rebuild_cs_pval_row()

    # ── Adaptive test window ───────────────────────────────────────────

    _ADAPTIVE_TW_GROUPS = ('msconn', 'widems', '2peakms')
    _ADAPTIVE_TW_MIN_LAG = -1e-3   # -1 ms
    _ADAPTIVE_TW_MAX_LAG =  1e-3   #  1 ms

    def _pair_qualifies_for_adaptive_tw(self, ref: int, tgt: int) -> bool:
        pair = (ref, tgt)
        return any(pair in self._group_pairs(g) for g in self._ADAPTIVE_TW_GROUPS)

    def _effective_lags(self, ref: int, tgt: int):
        """Return (min_lag, max_lag) accounting for adaptive test window."""
        if (self._adaptive_tw_var.get()
                and self._pair_qualifies_for_adaptive_tw(ref, tgt)):
            return self._ADAPTIVE_TW_MIN_LAG, self._ADAPTIVE_TW_MAX_LAG
        conf = self.ccg_data.conf if self.ccg_data else None
        if conf is None:
            return None, None
        return conf.min_lag, conf.max_lag

    def _update_adaptive_tw_availability(self):
        """Enable/disable the adaptive TW button based on current pair's group tags."""
        btn = getattr(self, '_adaptive_tw_btn', None)
        if btn is None:
            return
        inds = self._current_inds()
        qualifies = (inds is not None
                     and self._pair_qualifies_for_adaptive_tw(int(inds[0]), int(inds[1])))
        try:
            btn.state(['!disabled'] if qualifies else ['disabled'])
        except Exception:
            pass
        if not qualifies and self._adaptive_tw_var.get():
            self._adaptive_tw_var.set(False)

    def _on_adaptive_tw_toggle(self):
        """Adaptive test window toggled — clear caches and re-render."""
        self._conn_strength_cache.clear()
        self._clear_all_png_cache()
        self.update_plot()
        self._update_conn_str_label()
        self._draw_network()

    def _deferred_initial_draw(self):
        # Load groups/hotkeys first (independent of session data).
        self._load_groups_from_export()

        def _after_initial_draw():
            self._finish_initial_draw()
            # Autoload selections + pair tags AFTER _finish_initial_draw so the
            # reset in that method doesn't clobber what we loaded.
            self._autoload_session_latest(restore_groups=False)
            self.refresh_lists()
            self._draw_network()

        self._ensure_session_loaded(self.key.nd(), on_loaded=_after_initial_draw)

    def update_plot(self):
        try:
            if getattr(self, '_session_any_mode', False):
                hl = getattr(self, '_any_pair_handle_list', None) or []
                if self._focused_pair is None and self.current_pair_idx < len(hl):
                    row_ck = hl[self.current_pair_idx][0]
                    if self.key != row_ck:
                        self._sync_any_plot_context(self.current_pair_idx)
                if self._focused_pair is None:
                    if self.current_pair_idx < len(self.all_inds):
                        self._sync_any_plot_context(self.current_pair_idx)
                self._clamp_current_segment_for_session()
            # Data not yet loaded — nothing to render
            if self.ccg_data is None or self.ccg_pointer is None:
                return

            # In All/Any mode, Time Slider can be enabled even though a single
            # behavioral timeline can't represent multiple sessions at once.
            # Keep rendering the CCG plot (the time slider UI is separate).

            # If spike attribution raster is active, keep showing it
            if self._sa_selected_idx >= 0 and self._sa_spike_pairs:
                return

            # Focus-pair override: show focused pair's CCG directly
            if self._focused_pair is not None:
                inds = np.array(self._focused_pair)
            elif self.current_pair_idx >= len(self.all_inds):
                return
            else:
                inds = self.all_inds[self.current_pair_idx]
            # Safety: keep rendered pair indices aligned to currently bound ccg_data.
            r0, t0 = int(inds[0]), int(inds[1])
            if not self._ccg_pair_in_bounds(r0, t0):
                if (getattr(self, '_session_any_mode', False)
                        and self.current_pair_idx < len(self.all_inds)):
                    self._sync_any_plot_context(self.current_pair_idx)
                    self._clamp_current_segment_for_session()
                    if self.ccg_data is None or self.current_pair_idx >= len(self.all_inds):
                        return
                    inds = self.all_inds[self.current_pair_idx]
                    r0, t0 = int(inds[0]), int(inds[1])
                if not self._ccg_pair_in_bounds(r0, t0):
                    print(f"[CCGReviewUI] skipping out-of-bounds pair ({r0}, {t0}) "
                          f"for ccg shape={getattr(getattr(self.ccg_data, 'ccg', None), 'shape', None)}")
                    return
            sbs = self._sbs_mode

            if self._together_pairs and len(self._together_pairs) >= 2:
                # Stacked view: one row per pinned pair, 1 or 2 columns (sbs only)
                n_tog = len(self._together_pairs)
                n_cols = 2 if sbs else 1
                col_titles = (['Lo-res', 'Hi-res'] if sbs else [''])

                self.fig.clear()
                axes_grid = self.fig.subplots(n_tog, n_cols, squeeze=False)
                for row_i, tp in enumerate(self._together_pairs):
                    tp_arr = np.array(tp)
                    seg = self.current_segment
                    if sbs:
                        pngs = [
                            self._render_png_with_res(tp_arr, seg, highres=False),
                            self._render_png_with_res(tp_arr, seg, highres=True),
                        ]
                    else:
                        pngs = [self._get_or_render_png(tp_arr, seg)]
                    for ax, png, col_title in zip(axes_grid[row_i], pngs, col_titles):
                        ax.imshow(mpimg.imread(png))
                        ax.axis('off')
                        if col_title:
                            ax.set_title(col_title, fontsize=8, pad=1)
                self.fig.tight_layout(pad=0.05)

            elif getattr(self, '_stacked_segments', None):
                segs = sorted(int(s) for s in self._stacked_segments)
                if not segs:
                    return
                nd_key = self.key.nd()
                has_hi = (hasattr(self.cd, '_ccg_highres')
                          and self.cd._ccg_highres.get(nd_key) is not None)
                n_cols = 2 if (sbs and has_hi) else 1
                col_titles = (['Lo-res', 'Hi-res'] if n_cols == 2 else [''])

                def _seg_label(si: int) -> str:
                    if si == self.n_segments:
                        return _ALL_SEGS
                    if self._is_custom_segment(si):
                        ci = self._custom_seg_index(si)
                        cs_list = getattr(self, '_custom_segments', [])
                        if 0 <= ci < len(cs_list):
                            return cs_list[ci].get('name', f'custom {ci}')
                        return 'custom'
                    if 0 <= si < len(self.segment_names):
                        return str(self.segment_names[si])
                    return str(si)

                self.fig.clear()
                axes_grid = self.fig.subplots(len(segs), n_cols, squeeze=False)
                for row_i, seg in enumerate(segs):
                    if n_cols == 2:
                        pngs = [
                            self._render_png_with_res(inds, seg, highres=False),
                            self._render_png_with_res(inds, seg, highres=True),
                        ]
                    else:
                        pngs = [self._render_png_with_res(inds, seg, highres=False)]
                    for ax, png, col_title in zip(axes_grid[row_i], pngs, col_titles):
                        ax.imshow(mpimg.imread(png))
                        ax.axis('off')
                        t = _seg_label(seg)
                        if col_title:
                            ax.set_title(f"{t} · {col_title}", fontsize=8, pad=1)
                        else:
                            ax.set_title(t, fontsize=8, pad=1)
                self.fig.tight_layout(pad=0.05)

            elif not sbs:
                # 1x1 — single view (CS overlaid directly on this plot)
                png_path = self._get_or_render_png(inds, self.current_segment)
                img = mpimg.imread(png_path)
                self.fig.clear()
                ax = self.fig.add_subplot(111)
                ax.imshow(img)
                ax.axis('off')
                self.fig.tight_layout(pad=0)

            else:
                # 1x2 — lo | hi side-by-side (CS overlaid on each)
                png_lo = self._render_png_with_res(inds, self.current_segment, highres=False)
                png_hi = self._render_png_with_res(inds, self.current_segment, highres=True)
                img_lo = mpimg.imread(png_lo)
                img_hi = mpimg.imread(png_hi)
                self.fig.clear()
                ax1, ax2 = self.fig.subplots(1, 2)
                ax1.imshow(img_lo); ax1.axis('off'); ax1.set_title('Lo-res', fontsize=9, pad=2)
                ax2.imshow(img_hi); ax2.axis('off'); ax2.set_title('Hi-res', fontsize=9, pad=2)
                self.fig.tight_layout(pad=0.3)

            self.canvas.draw()

            self.plot_title_var.set(self.get_plot_title())
            self._update_sig_indicators(inds)
            self._update_jitter_sig_buttons()
            self._update_global_baseline_availability()
            self._update_pair_info(inds)
            self._draw_waveforms()
            self._update_conn_str_label()

        except Exception as e:
            print(f"Error updating plot: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Waveforms
    # ------------------------------------------------------------------

    def _draw_waveforms(self):
        if not self._waveforms_visible or self.neurons is None:
            return
        if self.current_pair_idx >= len(self.all_inds):
            return
        inds = self.all_inds[self.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        wf = getattr(self.neurons, 'waveforms', None)
        shank_ids = getattr(self.neurons, 'shank_ids', None)
        self.wave_ax.clear()
        if wf is None or shank_ids is None:
            self.wave_ax.text(0.5, 0.5, "No waveform data",
                              ha='center', va='center',
                              transform=self.wave_ax.transAxes, fontsize=8)
            self.wave_ax.axis('off')
        else:
            # Resolve probe geometry from NeuronsDatasetConfig so discarded
            # channels are mapped correctly (mirrors plot_ccg_figure logic).
            nd_conf = getattr(getattr(self.cd, 'nd', None), '_conf', None)
            ch_per_shank = getattr(nd_conf, 'ch_per_shank', 16)
            recinfo = getattr(nd_conf, 'recinfo', None)
            skipped = getattr(recinfo, "skipped_channels", None)
            discarded = None if skipped is None else np.asarray(skipped, dtype=int)
            
            def get_filled_waveform(shank_id, wf_neuron):
                """Map compact waveform to full (ch_per_shank, n_samples) grid."""
                if wf_neuron.ndim == 1:
                    return np.tile(wf_neuron, (ch_per_shank, 1))
                channel_ids = ch_per_shank * shank_id + np.arange(ch_per_shank)
                mask = ~np.isin(channel_ids, discarded)
                start = int(ch_per_shank * shank_id
                            - np.sum(discarded < ch_per_shank * shank_id))
                length = int(np.sum(mask))
                clean = np.full((ch_per_shank, wf_neuron.shape[-1]), np.nan)
                clean[mask] = wf_neuron[start:start + length]
                return clean

            ref_shank = int(shank_ids[ref])
            tgt_shank = int(shank_ids[tgt])
            ref_waveform = get_filled_waveform(ref_shank, wf[ref])
            tgt_waveform = get_filled_waveform(tgt_shank, wf[tgt])
            color = 'green' if ref_shank != tgt_shank else 'orange'
            peak_g = None
            pk = getattr(self.neurons, 'peak_channels', None)
            if pk is not None:
                try:
                    peak_g = int(pk[ref])
                except (TypeError, IndexError, ValueError):
                    peak_g = None
            plot_waveform_on_channel(
                ref_waveform=ref_waveform,
                ref_shank=ref_shank,
                target_waveform=tgt_waveform,
                target_shank=tgt_shank,
                color=color,
                amplitude_limit=True,
                ax=self.wave_ax,
                ch_per_shank=ch_per_shank,
                discarded_channels=discarded,
                peak_channel_global=peak_g,
            )
        self.wave_canvas.draw()

    # ------------------------------------------------------------------
    # Time slider
    # ------------------------------------------------------------------

    def _ts_discover_themes(self):
        """Discover available Epoch objects from the session for theme switching."""
        self._ts_themes = {}
        self._ts_theme_label_union_all_sessions = {}
        # Known Epoch attribute names on session objects (ProcessData)
        _EPOCH_ATTRS = [
            'paradigm', 'brainstates', 'theta', 'theta_epochs',
            'ripple', 'sw', 'spindle', 'pbe', 'off_epochs',
            'micro_arousals', 'artifact', 'handling',
            'maze1_run', 'maze2_run', 'maze_run', 'remaze_run',
        ]
        sessions = getattr(getattr(self.cd, 'nd', None), '_sessions', None)
        if not sessions:
            return
        if not isinstance(sessions, (list, tuple)):
            sessions = [sessions]
        # Use the first session that matches our current key
        session_name = getattr(self.key, 'session', None)
        session = None
        for s in sessions:
            nd = getattr(self.cd, 'nd', None)
            if nd is not None:
                sname = nd._short_session_name(s)
                if sname == session_name:
                    session = s
                    break
        if session is None:
            print(f"[TimeSlider] no matching session object for {session_name}")
            return
        for attr in _EPOCH_ATTRS:
            obj = getattr(session, attr, None)
            if obj is not None and _Epoch is not None and isinstance(obj, _Epoch):
                if obj.n_epochs > 0:
                    self._ts_themes[attr] = obj
        # Update combobox values
        theme_names = ['segments'] + sorted(self._ts_themes.keys())
        self._ts_theme_combo['values'] = theme_names
        n_themes = len(self._ts_themes)
        self._ts_theme_info_var.set(
            f"{n_themes} theme{'s' if n_themes != 1 else ''} available")
        self._ts_rebuild_theme_label_union_for_all_sessions()

    def _ts_all_process_data_sessions(self):
        """ProcessData objects for every loaded session (for cross-session label union)."""
        nd = getattr(self.cd, 'nd', None)
        if nd is None:
            return []
        raw = getattr(nd, '_sessions', None)
        if raw is None:
            raw = []
        elif not isinstance(raw, (list, tuple)):
            raw = [raw]
        objs = [s for s in raw if s is not None]
        if objs:
            return objs
        out = []
        seen = set()
        for nk in self._real_nd_keys_ordered():
            s = self._session_obj_for_nd_key(nk)
            if s is None:
                continue
            sid = id(s)
            if sid in seen:
                continue
            seen.add(sid)
            out.append(s)
        return out

    def _ts_refresh_union_if_all_sessions_mode(self):
        """Recompute All-session label union without resetting the time-slider theme/handles."""
        if not getattr(self, '_session_any_mode', False):
            return
        if getattr(self, '_ts_theme_combo', None) is None:
            return
        self._ts_rebuild_theme_label_union_for_all_sessions()
        self._ts_label_colors = None
        self._ts_update_overlap_ui()
        self._ts_redraw()

    def _ts_rebuild_theme_label_union_for_all_sessions(self):
        """When Session=All, map each theme to sorted unique labels seen on any session."""
        self._ts_theme_label_union_all_sessions = {}
        if not getattr(self, '_session_any_mode', False):
            return
        session_objs = self._ts_all_process_data_sessions()
        if not session_objs:
            return
        for attr in self._ts_themes.keys():
            acc = set()
            for s in session_objs:
                obj = getattr(s, attr, None)
                if obj is None or _Epoch is None or not isinstance(obj, _Epoch):
                    continue
                if obj.n_epochs <= 0:
                    continue
                for lbl in obj.labels:
                    s = str(lbl).strip()
                    if s:
                        acc.add(s)
            if acc:
                self._ts_theme_label_union_all_sessions[attr] = sorted(acc)
        # segments: union of segment labels across every loaded CCG pointer
        seg = set()
        data = getattr(self.cd, 'data', None)
        if data:
            for ptr in data.values():
                if ptr is None:
                    continue
                try:
                    et = ptr.edge_times
                except Exception:
                    continue
                cols = getattr(et, 'columns', None)
                if cols is None or 'label' not in cols:
                    continue
                for v in et['label'].values:
                    s = str(v).strip()
                    if s:
                        seg.add(s)
        if seg:
            self._ts_theme_label_union_all_sessions['segments'] = sorted(seg)

    def _on_ts_theme_change(self, _event=None):
        """Handle theme combobox selection — repopulate epoch bounds."""
        theme = self._ts_theme_var.get()
        self._ts_current_theme = theme
        # Reset handles
        self._slider_t_start = None
        self._slider_t_end = None
        if hasattr(self, '_ts_start_var'):
            self._ts_start_var.set("00:00:00")
        if hasattr(self, '_ts_end_var'):
            self._ts_end_var.set("00:00:00")
        # Reset zoom
        self._ts_zoom_start = None
        self._ts_zoom_end = None
        self._ts_zoom_start_var.set("00:00:00")
        self._ts_zoom_end_var.set("00:00:00")
        self._ts_zoom_frame.pack_forget()
        # Reset label color cache for new theme
        self._ts_label_colors = None
        self._ts_init_times()
        self._ts_redraw()

    def _ts_collect_theme_ui_labels(self) -> list[str]:
        """Non-blank labels for overlap + legend; Session=All includes union (also stripped)."""
        theme = getattr(self, '_ts_current_theme', 'segments')
        acc = {str(lb).strip() for _, _, lb in self._ts_epoch_bounds if str(lb).strip()}
        if getattr(self, '_session_any_mode', False):
            extra = (getattr(self, '_ts_theme_label_union_all_sessions', None)
                     or {}).get(theme, ())
            acc |= {str(x).strip() for x in (extra or ()) if str(x).strip()}
        out = sorted(acc)
        if not out and theme != 'segments' and theme in getattr(self, '_ts_themes', {}):
            return [theme]
        return out

    def _ts_update_overlap_ui(self):
        """Update the label-filter dropdown for the current theme."""
        combo = getattr(self, '_ts_label_combo', None)
        row = getattr(self, '_ts_overlap_row', None)
        if combo is None or row is None:
            return
        theme = getattr(self, '_ts_current_theme', 'segments')
        all_labels = self._ts_collect_theme_ui_labels()
        if len(all_labels) > 1:
            sorted_labels = ['All'] + all_labels + ['NONE']
            combo['values'] = sorted_labels
            self._ts_label_var.set('All')
        else:
            # Single-label theme (e.g. ripple): show theme name + NONE
            display_name = theme if theme != 'segments' else all_labels[0] if all_labels else 'segments'
            combo['values'] = [display_name, 'NONE']
            self._ts_label_var.set(display_name)
        row.pack(side=tk.LEFT, padx=(8, 0))
        self._ts_active_label = None
        # Reset label color cache so it rebuilds for new theme
        self._ts_label_colors = None
        self._ts_update_legend()

    def _ts_init_times(self):
        """Populate epoch bounds from the selected theme or edge_times."""
        theme = getattr(self, '_ts_current_theme', 'segments')

        if theme != 'segments' and theme in self._ts_themes:
            # Use Epoch object directly
            epoch = self._ts_themes[theme]
            labs = [str(x).strip() for x in epoch.labels]
            self._ts_epoch_bounds = [
                (float(s), float(e), lb)
                for s, e, lb in zip(epoch.starts, epoch.stops, labs)]
            unique_nonblank = {lb for lb in labs if lb}
            # No usable labels (e.g. ripple): collapse to theme name for UI/chips
            if len(unique_nonblank) == 0:
                self._ts_epoch_bounds = [
                    (s, e, theme) for s, e, _ in self._ts_epoch_bounds]
            elif len(unique_nonblank) <= 1:
                self._ts_epoch_bounds = [
                    (s, e, theme) for s, e, _ in self._ts_epoch_bounds]
            self._ts_total_sec = (float(epoch.stops.max())
                                  if len(epoch.stops) else 1.0)
            self._ts_update_overlap_ui()
            return

        # Default: use CCG segment edge_times
        et = self.ccg_pointer.edge_times
        cols = et.columns.tolist()

        # Find start/stop columns by common name conventions
        def _find_col(candidates):
            for c in candidates:
                if c in cols:
                    return c
            return None

        start_col = _find_col(['start', 't_start', 'start_time', 'start_s'])
        stop_col  = _find_col(['stop',  't_end',   'end_time',   'stop_s', 'end'])

        self._ts_epoch_bounds = []
        if start_col and stop_col:
            for _, row in et.iterrows():
                t0 = float(row[start_col])
                t1 = float(row[stop_col])
                self._ts_epoch_bounds.append((t0, t1, str(row['label'])))
            self._ts_total_sec = (
                max((b[1] for b in self._ts_epoch_bounds), default=1.0)
                if self._ts_epoch_bounds else 1.0)
        else:
            # Fall back: reconstruct from cumulative effective_time_hours
            t = 0.0
            for _, row in et.iterrows():
                dur = float(row['effective_time_hours']) * 3600.0
                self._ts_epoch_bounds.append((t, t + dur, str(row['label'])))
                t += dur
            self._ts_total_sec = t if t > 0 else 1.0
        self._ts_update_overlap_ui()

    def _ts_t_to_x(self, t: float) -> int:
        w = max(self.ts_canvas.winfo_width(), 20)
        return int((t / max(self._ts_total_sec, 1)) * (w - 20) + 10)

    def _ts_x_to_t(self, x: int) -> float:
        w = max(self.ts_canvas.winfo_width(), 20)
        return max(0.0, min(self._ts_total_sec,
                            (x - 10) / max(w - 20, 1) * self._ts_total_sec))

    def _on_ts_tool_change(self):
        """Handle toolbar selection / lock toggle."""
        locked = self._ts_lock_var.get()
        selection = self._ts_tool_var.get() == 'selection'
        if locked:
            self.ts_canvas.config(cursor='arrow')
        elif selection:
            self.ts_canvas.config(cursor='plus')
        else:
            self.ts_canvas.config(cursor='crosshair')
        # Hide zoom panel when selection is off
        if not selection:
            self._ts_zoom_frame.pack_forget()
            self._ts_zoom_start = None
            self._ts_zoom_end = None
        self._ts_redraw()

    # ── Drawing ───────────────────────────────────────────────────────

    _TS_COLORS = ['#BBDEFB', '#C8E6C9', '#FFF9C4', '#FFE0B2', '#E1BEE7',
                  '#F8BBD0', '#D7CCC8', '#B2EBF2', '#DCEDC8', '#F0F4C3']

    _TS_NONE_COLOR = '#E0E0E0'

    def _ts_label_color_map(self):
        """Return {label: color} mapping — consistent color per unique label."""
        if not hasattr(self, '_ts_label_colors') or self._ts_label_colors is None:
            self._ts_label_colors = {}
        # Rebuild if labels changed
        all_labels = self._ts_collect_theme_ui_labels()
        expected = set(all_labels) | {'NONE'}
        if expected != set(self._ts_label_colors.keys()):
            self._ts_label_colors = {
                lbl: self._TS_COLORS[i % len(self._TS_COLORS)]
                for i, lbl in enumerate(all_labels)
            }
            self._ts_label_colors['NONE'] = self._TS_NONE_COLOR
        return self._ts_label_colors

    def _ts_draw_epochs(self, canvas, t_to_x, bounds, h):
        """Draw epoch rectangles on a canvas using given t_to_x mapping."""
        cmap = self._ts_label_color_map()
        y_bot = h - 16  # leave room for time axis
        for t0, t1, lbl in bounds:
            x0, x1 = t_to_x(t0), t_to_x(t1)
            color = cmap.get(lbl, '#E0E0E0')
            canvas.create_rectangle(x0, 6, x1, y_bot,
                                    fill=color, outline='#90A4AE')
            if x1 - x0 > 22:
                canvas.create_text((x0 + x1) // 2, (6 + y_bot) // 2,
                                   text=lbl, font=('Arial', 7), fill='#333')

    def _ts_update_legend(self):
        """Populate legend row with toggle-able swatches for each unique label."""
        frame = self._ts_legend_frame
        for w in frame.winfo_children():
            w.destroy()
        cmap = self._ts_label_color_map()
        self._ts_legend_toggles = {}
        for lbl, color in cmap.items():
            var = tk.BooleanVar(value=True)
            self._ts_legend_toggles[lbl] = var
            # Combined swatch+label as a single clickable button
            btn = tk.Frame(frame, cursor='hand2')
            btn.pack(side=tk.LEFT, padx=(4, 6), pady=1)
            swatch = tk.Frame(btn, width=12, height=10, bg=color,
                              highlightbackground='#90A4AE', highlightthickness=1)
            swatch.pack(side=tk.LEFT, padx=(0, 2))
            swatch.pack_propagate(False)
            lbl_w = tk.Label(btn, text=lbl, font=('Arial', 7),
                             fg='#333', relief=tk.RAISED, bd=1, padx=2)
            lbl_w.pack(side=tk.LEFT)

            def _toggle(v=var, s=swatch, l=lbl_w, c=color):
                v.set(not v.get())
                if v.get():
                    s.config(bg=c)
                    l.config(fg='#333', relief=tk.RAISED)
                else:
                    s.config(bg='#D0D0D0')
                    l.config(fg='#AAA', relief=tk.SUNKEN)
            for w in (swatch, lbl_w, btn):
                w.bind('<Button-1>', lambda e, t=_toggle: t())
        frame.pack(fill=tk.X, padx=4, pady=(1, 0),
                   before=self._ts_main_canvas_frame)

    def _ts_draw_time_axis(self, canvas, t_to_x, t_min, t_max, h):
        """Draw time axis tick marks and labels at the bottom of a canvas."""
        w = canvas.winfo_width()
        if w < 40:
            return
        span = t_max - t_min
        if span <= 0:
            return
        # Choose tick interval: aim for ~6-10 ticks
        raw = span / 8.0
        # Round to nice intervals (in seconds)
        nice_intervals = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600,
                          900, 1800, 3600, 7200, 14400, 28800]
        tick_step = nice_intervals[-1]
        for ni in nice_intervals:
            if ni >= raw:
                tick_step = ni
                break
        # Draw ticks
        y_tick = h - 2
        y_label = h - 1
        t = (int(t_min / tick_step) + 1) * tick_step if t_min % tick_step != 0 else t_min
        # Always draw first and last
        endpoints = {t_min, t_max}
        drawn_x = []
        for tp in sorted(endpoints):
            x = t_to_x(tp)
            if 5 <= x <= w - 5:
                canvas.create_line(x, y_tick - 4, x, y_tick, fill='#888')
                canvas.create_text(x, y_label, text=self._ts_sec_to_hms(tp),
                                   font=('Courier', 6), fill='#666', anchor='s')
                drawn_x.append(x)
        while t <= t_max:
            x = t_to_x(t)
            # Skip if too close to an already-drawn label
            if 5 <= x <= w - 5 and all(abs(x - dx) > 38 for dx in drawn_x):
                canvas.create_line(x, y_tick - 4, x, y_tick, fill='#888')
                canvas.create_text(x, y_label, text=self._ts_sec_to_hms(t),
                                   font=('Courier', 6), fill='#666', anchor='s')
                drawn_x.append(x)
            t += tick_step

    def _ts_draw_handles(self, canvas, t_to_x, h, t_start, t_end,
                         color_start='#1565C0', color_end='#B71C1C'):
        """Draw cursor handles and selection range on a canvas."""
        for t, color in [(t_start, color_start), (t_end, color_end)]:
            if t is not None:
                x = t_to_x(t)
                canvas.create_line(x, 2, x, h - 2, fill=color, width=2)
                canvas.create_polygon(x - 5, 2, x + 5, 2, x, 10,
                                      fill=color, outline='')
        if t_start is not None and t_end is not None:
            x0, x1 = t_to_x(t_start), t_to_x(t_end)
            canvas.create_rectangle(x0, 8, x1, h - 8,
                                    fill='', outline=color_start,
                                    width=2, dash=(4, 2))

    _TS_NONE = 'NONE'  # sentinel for gap intervals

    def _on_ts_label_change(self, _event=None):
        """Handle overlap label combobox selection."""
        val = self._ts_label_var.get()
        if val == 'All':
            self._ts_active_label = None
        elif val == self._TS_NONE:
            self._ts_active_label = self._TS_NONE
        else:
            # Could be a real label or the theme display name (single-label themes)
            all_labels = self._ts_collect_theme_ui_labels()
            if val in all_labels:
                self._ts_active_label = val
            else:
                # Theme display name for single-label theme → show all intervals
                self._ts_active_label = None
        self._ts_redraw()

    def _on_ts_label_reset(self):
        """Revert to showing all labels."""
        self._ts_active_label = None
        combo = getattr(self, '_ts_label_combo', None)
        if combo:
            vals = list(combo['values'])
            self._ts_label_var.set(vals[0] if vals else 'All')
        else:
            self._ts_label_var.set('All')
        self._ts_redraw()

    def _ts_visible_bounds(self):
        """Return epoch bounds filtered by active label (or all if None).
        NONE returns gaps between all epoch intervals."""
        if self._ts_active_label is None:
            return self._ts_epoch_bounds
        if self._ts_active_label == self._TS_NONE:
            return self._ts_compute_gaps()
        return [(t0, t1, lbl) for t0, t1, lbl in self._ts_epoch_bounds
                if lbl == self._ts_active_label]

    def _ts_compute_gaps(self):
        """Compute time gaps between all epoch intervals."""
        if not self._ts_epoch_bounds:
            return [(0, self._ts_total_sec, 'NONE')]
        # Merge overlapping intervals first
        sorted_bounds = sorted(self._ts_epoch_bounds, key=lambda x: x[0])
        merged = []
        cur_start, cur_end = sorted_bounds[0][0], sorted_bounds[0][1]
        for t0, t1, _ in sorted_bounds[1:]:
            if t0 <= cur_end:
                cur_end = max(cur_end, t1)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = t0, t1
        merged.append((cur_start, cur_end))
        # Build gaps
        gaps = []
        if merged[0][0] > 0:
            gaps.append((0, merged[0][0], 'NONE'))
        for i in range(len(merged) - 1):
            gap_start = merged[i][1]
            gap_end = merged[i + 1][0]
            if gap_end > gap_start:
                gaps.append((gap_start, gap_end, 'NONE'))
        if merged[-1][1] < self._ts_total_sec:
            gaps.append((merged[-1][1], self._ts_total_sec, 'NONE'))
        return gaps

    def _ts_redraw(self, event=None):
        c = self.ts_canvas
        c.delete('all')
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 20:
            return

        if getattr(self, '_session_any_mode', False):
            c.create_text(
                w // 2, h // 2,
                text="All sessions view — no single behavioral timeline to display.\n"
                     "Use 'Set' + 'Apply to Multiple Sessions' to compute custom CCGs for sessions.",
                anchor='center', font=('Arial', 9), fill='#555', justify='center')
            return

        if not self._ts_epoch_bounds:
            return

        self._ts_draw_epochs(c, self._ts_t_to_x, self._ts_visible_bounds(), h)
        self._ts_draw_time_axis(c, self._ts_t_to_x, 0, self._ts_total_sec, h)

        # Draw select-mode handles (custom window cursors)
        self._ts_draw_handles(c, self._ts_t_to_x, h,
                              self._slider_t_start, self._slider_t_end)

        # Draw selection/zoom handles (orange) when selection tool active
        if self._ts_tool_var.get() == 'selection':
            self._ts_draw_handles(c, self._ts_t_to_x, h,
                                  self._ts_zoom_start, self._ts_zoom_end,
                                  color_start='#E65100', color_end='#BF360C')
            # Shade zoom region
            if self._ts_zoom_start is not None and self._ts_zoom_end is not None:
                zx0 = self._ts_t_to_x(self._ts_zoom_start)
                zx1 = self._ts_t_to_x(self._ts_zoom_end)
                c.create_rectangle(zx0, 4, zx1, h - 4,
                                   fill='#FFF3E0', outline='#E65100',
                                   width=1, stipple='gray25')
            self._ts_zoom_redraw()
            self._ts_draw_radiate_lines()

    # ── Zoom detail canvas ────────────────────────────────────────────

    def _ts_zoom_t_to_x(self, t: float) -> int:
        """Map time to x within the zoom canvas, using zoom region bounds."""
        w = max(self._ts_zoom_canvas.winfo_width(), 20)
        z0 = self._ts_zoom_start if self._ts_zoom_start is not None else 0
        z1 = self._ts_zoom_end if self._ts_zoom_end is not None else self._ts_total_sec
        span = max(z1 - z0, 1e-6)
        return int(((t - z0) / span) * (w - 20) + 10)

    def _ts_zoom_redraw(self, event=None):
        """Redraw the zoomed-in detail canvas."""
        zc = self._ts_zoom_canvas
        zc.delete('all')
        if self._ts_zoom_start is None or self._ts_zoom_end is None:
            return
        w = zc.winfo_width()
        h = zc.winfo_height()
        if w < 20:
            return

        z0, z1 = self._ts_zoom_start, self._ts_zoom_end

        # Filter epoch bounds that overlap the zoom region (respects active label)
        zoomed_bounds = [
            (max(t0, z0), min(t1, z1), lbl)
            for t0, t1, lbl in self._ts_visible_bounds()
            if t1 > z0 and t0 < z1
        ]
        self._ts_draw_epochs(zc, self._ts_zoom_t_to_x, zoomed_bounds, h)
        self._ts_draw_time_axis(zc, self._ts_zoom_t_to_x, z0, z1, h)

        # Draw select-mode handles within zoom view
        self._ts_draw_handles(zc, self._ts_zoom_t_to_x, h,
                              self._slider_t_start, self._slider_t_end)

    def _ts_draw_radiate_lines(self):
        """Draw radiating lines connecting zoom region on main canvas to zoom canvas."""
        rc = self._ts_radiate_canvas
        rc.delete('all')
        if self._ts_zoom_start is None or self._ts_zoom_end is None:
            return
        w = rc.winfo_width()
        rh = rc.winfo_height()
        if w < 20:
            return

        # Top points: zoom region edges on main canvas
        zx0_main = self._ts_t_to_x(self._ts_zoom_start)
        zx1_main = self._ts_t_to_x(self._ts_zoom_end)

        # Bottom points: full width of zoom canvas
        zx0_zoom = 10
        zx1_zoom = max(self._ts_zoom_canvas.winfo_width(), 20) - 10

        # Draw radiating lines
        rc.create_line(zx0_main, 0, zx0_zoom, rh,
                       fill='#E65100', width=1, dash=(3, 3))
        rc.create_line(zx1_main, 0, zx1_zoom, rh,
                       fill='#E65100', width=1, dash=(3, 3))

    # ── Mouse interaction ─────────────────────────────────────────────

    def _ts_mouse_press(self, event):
        if self._ts_lock_var.get():
            return  # all cursor interaction disabled
        selection = self._ts_tool_var.get() == 'selection'
        if selection:
            self._ts_zoom_mouse_press(event)
            return
        # Default: custom CCG window tool
        if self._slider_t_start is None:
            self._slider_dragging = 'start'
        elif self._slider_t_end is None:
            self._slider_dragging = 'end'
        else:
            xs = self._ts_t_to_x(self._slider_t_start)
            xe = self._ts_t_to_x(self._slider_t_end)
            self._slider_dragging = ('start'
                                     if abs(event.x - xs) <= abs(event.x - xe)
                                     else 'end')
        self._ts_update_handle(event.x)

    def _ts_mouse_drag(self, event):
        if self._ts_lock_var.get():
            return
        if self._ts_tool_var.get() == 'selection':
            self._ts_zoom_mouse_drag(event)
            return
        self._ts_update_handle(event.x)

    def _ts_mouse_release(self, event):
        if self._ts_lock_var.get():
            return
        if self._ts_tool_var.get() == 'selection':
            self._ts_zoom_mouse_release(event)
            return
        self._ts_update_handle(event.x, snap=True)
        self._slider_dragging = None

    def _ts_update_handle(self, canvas_x: int, snap: bool = False):
        t = self._ts_x_to_t(canvas_x)
        if snap and self._ts_snap_var.get() and self._ts_epoch_bounds:
            bounds_t = [b for (t0, t1, _) in self._ts_epoch_bounds
                        for b in (t0, t1)]
            for bt in bounds_t:
                if abs(self._ts_t_to_x(bt) - canvas_x) <= 25:
                    t = bt
                    break
        if self._slider_dragging == 'start':
            cap = self._slider_t_end if self._slider_t_end is not None else self._ts_total_sec
            self._slider_t_start = min(t, cap)
            self._ts_start_var.set(self._ts_sec_to_hms(self._slider_t_start))
        elif self._slider_dragging == 'end':
            floor = self._slider_t_start if self._slider_t_start is not None else 0.0
            self._slider_t_end = max(t, floor)
            self._ts_end_var.set(self._ts_sec_to_hms(self._slider_t_end))
        self._ts_redraw()

    # ── Zoom tool mouse handlers ──────────────────────────────────────

    def _ts_zoom_mouse_press(self, event):
        if self._ts_zoom_start is None:
            self._ts_zoom_dragging = 'start'
        elif self._ts_zoom_end is None:
            self._ts_zoom_dragging = 'end'
        else:
            xs = self._ts_t_to_x(self._ts_zoom_start)
            xe = self._ts_t_to_x(self._ts_zoom_end)
            self._ts_zoom_dragging = ('start'
                                      if abs(event.x - xs) <= abs(event.x - xe)
                                      else 'end')
        self._ts_zoom_update(event.x)

    def _ts_zoom_mouse_drag(self, event):
        self._ts_zoom_update(event.x)

    def _ts_zoom_mouse_release(self, event):
        self._ts_zoom_update(event.x, snap=True)
        self._ts_zoom_dragging = None
        # Show zoom canvas once both ends are placed
        if self._ts_zoom_start is not None and self._ts_zoom_end is not None:
            self._ts_zoom_frame.pack(fill=tk.X, padx=4, pady=(0, 0),
                                     after=self._ts_ccg_ctrl)
            self._ts_zoom_redraw()
            self._ts_draw_radiate_lines()

    def _ts_zoom_update(self, canvas_x: int, snap: bool = False):
        t = self._ts_x_to_t(canvas_x)
        if snap and self._ts_snap_var.get() and self._ts_epoch_bounds:
            bounds_t = [b for (t0, t1, _) in self._ts_epoch_bounds
                        for b in (t0, t1)]
            for bt in bounds_t:
                if abs(self._ts_t_to_x(bt) - canvas_x) <= 25:
                    t = bt
                    break
        if self._ts_zoom_dragging == 'start':
            cap = self._ts_zoom_end if self._ts_zoom_end is not None else self._ts_total_sec
            self._ts_zoom_start = min(t, cap)
            self._ts_zoom_start_var.set(self._ts_sec_to_hms(self._ts_zoom_start))
        elif self._ts_zoom_dragging == 'end':
            floor = self._ts_zoom_start if self._ts_zoom_start is not None else 0.0
            self._ts_zoom_end = max(t, floor)
            self._ts_zoom_end_var.set(self._ts_sec_to_hms(self._ts_zoom_end))
        self._ts_redraw()

    def _on_zoom_range_set(self):
        """Set zoom range from the zoom time entry boxes."""
        try:
            t0 = self._ts_hms_to_sec(self._ts_zoom_start_var.get())
            t1 = self._ts_hms_to_sec(self._ts_zoom_end_var.get())
        except (ValueError, IndexError):
            return
        if t1 <= t0:
            return
        self._ts_zoom_start = max(0.0, min(t0, self._ts_total_sec))
        self._ts_zoom_end = max(0.0, min(t1, self._ts_total_sec))
        self._ts_zoom_start_var.set(self._ts_sec_to_hms(self._ts_zoom_start))
        self._ts_zoom_end_var.set(self._ts_sec_to_hms(self._ts_zoom_end))
        self._ts_redraw()

    def _ts_hms_to_sec(self, hms: str) -> float:
        s = hms.strip().lower()
        if s == 'start':
            return 0.0
        if s == 'end':
            return float(getattr(self, '_ts_total_sec', 0))
        parts = s.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])

    @staticmethod
    def _ts_sec_to_hms(sec) -> str:
        if sec is None:
            return "end"
        sec = int(sec)
        return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"

    @staticmethod
    def _resolve_ts_time(val, t_start: float, t_end: float) -> float:
        """Resolve a t0/t1 spec value — float, or sentinel string 'start'/'end'."""
        if val is None or (isinstance(val, str) and val.lower() == 'end'):
            return t_end
        if isinstance(val, str) and val.lower() == 'start':
            return t_start
        return float(val)

    @staticmethod
    def _split_time_range(t0: float, t1: float, n_splits: int, overlap_sec: float,
                          base_name: str) -> list:
        """Return list of (chunk_t0, chunk_t1, chunk_name) tuples.

        With n_splits=1 and overlap_sec=0, returns a single chunk (the original range).
        overlap_sec lets adjacent chunks overlap by that many seconds.
        """
        n_splits = max(1, int(n_splits))
        overlap_sec = max(0.0, float(overlap_sec))
        if n_splits == 1 and overlap_sec == 0.0:
            return [(t0, t1, base_name)]
        total = t1 - t0
        if total <= 0:
            return [(t0, t1, base_name)]
        if n_splits == 1:
            return [(t0, t1, base_name)]
        # chunk_len such that: chunk_len + (n-1)*(chunk_len - overlap) = total
        # => chunk_len * n - (n-1)*overlap = total
        # => chunk_len = (total + (n-1)*overlap) / n
        chunk_len = (total + (n_splits - 1) * overlap_sec) / n_splits
        stride = chunk_len - overlap_sec
        if stride <= 0:
            stride = total / n_splits
            chunk_len = stride
        chunks = []
        for i in range(n_splits):
            cs = t0 + i * stride
            ce = min(cs + chunk_len, t1)
            suffix = f"{i + 1}"
            chunks.append((cs, ce, base_name + suffix))
        return chunks

    def _on_time_slider_set(self):
        spec = self._build_custom_spec(for_all=False)
        if spec is None:
            return
        filter_state = spec.get('filter_state', {})
        # Resolve sentinels for current session (use min/max times, not first/last table row)
        seg_bounds = self._segment_bounds_for_key(self.key)
        t_sess_start, t_sess_end = self._segment_bounds_time_extent(seg_bounds)
        if not seg_bounds:
            t_sess_end = float(getattr(self, '_ts_total_sec', 0.0))
        t0 = self._resolve_ts_time(spec['t0'], t_sess_start, t_sess_end)
        t1 = self._resolve_ts_time(spec['t1'], t_sess_start, t_sess_end)
        lone = self._single_exclusive_segment_filter_label(filter_state)
        if lone is not None:
            span = self._union_span_for_segment_label(self.key, lone)
            if span is not None:
                t0, t1 = span[0], span[1]
        self._slider_t_start = t0
        self._slider_t_end = t1
        n_splits = max(1, int(spec.get('n_splits') or 1))
        overlap_sec = max(0.0, float(spec.get('overlap_sec') or 0.0))
        chunks = self._split_time_range(t0, t1, n_splits, overlap_sec, spec['name'])
        split_bid = None
        if n_splits > 1:
            split_bid = self._split_batch_next_id
            self._split_batch_next_id += 1
        split_names: list[str] = []
        queued = 0
        for chunk_t0, chunk_t1, chunk_name in chunks:
            bs_result = self._ts_brain_state_intervals(chunk_t0, chunk_t1)
            if bs_result is False:
                continue
            intervals, active_duration = bs_result
            metadata = {
                'name': chunk_name,
                'theme': filter_state.get('theme', 'segments'),
                'labels': filter_state.get('labels', {}),
                'scope': spec.get('scope', self._current_session_str()),
                'session': str(self.key.session),
                'timing': {'t0': chunk_t0, 't1': chunk_t1},
            }
            ok = self._enqueue_custom_ccg_task(
                key=self.key,
                t0=chunk_t0,
                t1=chunk_t1,
                name=chunk_name,
                intervals=intervals,
                active_duration=active_duration,
                filter_state=filter_state,
                metadata=metadata,
                auto_save=False,
                load_into_ui=True,
                split_batch_id=split_bid,
            )
            if ok:
                queued += 1
                if split_bid is not None:
                    split_names.append(chunk_name)
        if split_bid is not None and split_names:
            self._split_batch_counts[split_bid] = len(split_names)
            self._split_batch_chunk_names[split_bid] = split_names
        if queued:
            self._record_custom_ccg_suggestion(spec)
            label = spec['name'] if n_splits == 1 else f"{spec['name']} ({queued} chunks)"
            self._ts_status_var.set(f"Queued: {label}")
            self._custom_ccg_start_next()

    def _on_time_slider_apply_multiple_sessions(self):
        spec = self._build_custom_spec(for_all=False)
        if spec is None:
            return
        all_nd_keys = self._real_nd_keys_ordered()
        if not all_nd_keys:
            messagebox.showinfo("Sessions", "No sessions available.")
            return
        session_labels = [str(nk.session) for nk in all_nd_keys]
        current_sess = str(self.key.session) if not getattr(self, '_session_any_mode', False) else None
        selected = self._pick_sessions_dialog(
            title="Apply custom CCG to sessions",
            sessions=session_labels,
            current_session=current_sess,
        )
        if not selected:
            return
        is_all = len(selected) == len(session_labels)
        spec['sessions'] = selected
        spec['scope'] = 'All' if is_all else (
            selected[0] if len(selected) == 1 else ', '.join(selected[:2]) + ('…' if len(selected) > 2 else ''))
        self._record_custom_ccg_suggestion(spec)
        queued = self._queue_custom_ccg_for_spec(
            spec, for_all=is_all, auto_save=True,
            target_sessions=None if is_all else selected,
        )
        if queued:
            sess_label = "all sessions" if is_all else f"{len(selected)} session(s)"
            self._ts_status_var.set(f"Queued {queued} task(s) for {sess_label}")
            self._custom_ccg_start_next()
        else:
            self._ts_status_var.set("No missing custom CCGs for selected sessions")

    def _pick_sessions_dialog(self, title: str, sessions: list[str],
                              current_session: str | None = None) -> list[str] | None:
        """Open a session picker dialog; returns selected session strings or None on cancel."""
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("400x340")
        win.resizable(True, True)
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Select sessions  (Shift / Ctrl+click for multi-select):").pack(
            anchor='w', padx=8, pady=(8, 2))

        list_frame = ttk.Frame(win)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))
        lb = tk.Listbox(list_frame, selectmode=tk.EXTENDED, exportselection=False, height=14,
                        activestyle='dotbox')
        vsb = ttk.Scrollbar(list_frame, orient='vertical', command=lb.yview)
        lb.configure(yscrollcommand=vsb.set)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        for i, sess in enumerate(sessions):
            lb.insert(tk.END, sess)
            if current_session and sess == current_session:
                lb.itemconfigure(i, foreground='#1E5FBB')
        lb.select_set(0, tk.END)  # pre-select all

        result: list[str] | None = [None]

        def _ok():
            sels = lb.curselection()
            result[0] = [sessions[i] for i in sels]
            win.destroy()

        def _cancel():
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(btn_frame, text="Select All",
                   command=lambda: lb.select_set(0, tk.END)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Select None",
                   command=lambda: lb.selection_clear(0, tk.END)).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Cancel", command=_cancel).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="Apply", command=_ok).pack(side=tk.RIGHT, padx=4)

        win.bind('<Return>', lambda e: _ok())
        win.bind('<Escape>', lambda e: _cancel())
        win.wait_window()
        return result[0]

    def _generate_suggested_custom_ccgs(self):
        specs = self._load_custom_ccg_suggestions()
        if not specs:
            messagebox.showinfo(
                "Suggested custom CCGs",
                "No suggested custom CCG entries found. Use 'Refresh suggested custom CCGs' first.")
            return
        win = tk.Toplevel(self.root)
        win.title("Suggested custom CCGs")
        win.geometry("620x360")
        win.transient(self.root)
        win.grab_set()
        ttk.Label(
            win,
            text="Generate custom CCGs from availability list:"
        ).pack(anchor='w', padx=8, pady=(8, 4))
        lb = tk.Listbox(win, selectmode=tk.MULTIPLE, height=12)
        lb.pack(fill=tk.BOTH, expand=True, padx=8)
        def _fmt_t(v):
            if isinstance(v, str) and v.lower() in ('start', 'end'):
                return v
            try:
                return self._ts_sec_to_hms(float(v))
            except Exception:
                return str(v)

        for i, spec in enumerate(specs):
            name = str(spec.get('name', 'custom'))
            t0 = _fmt_t(spec.get('t0', 0.0))
            t1 = _fmt_t(spec.get('t1', 0.0))
            scope = str(spec.get('scope', 'By session'))
            n_have = len(spec.get('sessions', []) or [])
            n_total = max(1, len(self._real_nd_keys_ordered()))
            if scope == 'All':
                label = f"[{name} | {t0}-{t1}] for ALL ({n_have}/{n_total})"
            else:
                label = f"[{name} | {t0}-{t1}] for {scope} ({n_have}/{n_total})"
            lb.insert(tk.END, label)
            lb.select_set(i)

        def _run(selected_idxs):
            queued = 0
            for idx in selected_idxs:
                spec = specs[int(idx)]
                queued += self._queue_custom_ccg_for_spec(
                    spec, for_all=(str(spec.get('scope', '')).lower() == 'all'),
                    auto_save=True)
            if queued:
                self._ts_status_var.set(f"Queued {queued} suggested custom CCG task(s)")
                self._custom_ccg_start_next()
            else:
                self._ts_status_var.set("All suggested custom CCGs already exist")
            win.destroy()

        btns = ttk.Frame(win)
        btns.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(btns, text="Generate selected",
                   command=lambda: _run(lb.curselection())).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Generate all",
                   command=lambda: _run(range(len(specs)))).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.RIGHT)

    def _on_time_slider_clear(self):
        self._custom_segments.clear()
        if hasattr(self, '_ts_status_var'):
            self._ts_status_var.set("")
        # Reset time selection
        self._slider_t_start = None
        self._slider_t_end = None
        if hasattr(self, '_ts_start_var'):
            self._ts_start_var.set("00:00:00")
        if hasattr(self, '_ts_end_var'):
            self._ts_end_var.set("00:00:00")
        # Reset zoom
        self._ts_zoom_start = None
        self._ts_zoom_end = None
        self._ts_zoom_start_var.set("00:00:00")
        self._ts_zoom_end_var.set("00:00:00")
        self._ts_zoom_frame.pack_forget()
        # Reset to first real segment
        self.current_segment = 0
        self._build_sig_chips()
        self._update_segment_label()
        self.update_plot()

    def _session_obj_for_nd_key(self, nd_key):
        sessions = getattr(getattr(self.cd, 'nd', None), '_sessions', None)
        if sessions is None:
            return None
        if not isinstance(sessions, (list, tuple)):
            sessions = [sessions]
        nd = getattr(self.cd, 'nd', None)
        for s in sessions:
            try:
                if nd is not None and nd._short_session_name(s) == getattr(nd_key, 'session', None):
                    return s
            except Exception:
                continue
        return None

    def _segment_bounds_for_key(self, key) -> list:
        """Return segment bounds from edge_times for a key (always uses 'segments' theme)."""
        ptr = self.cd.data.get(key)
        if ptr is None or getattr(ptr, 'edge_times', None) is None:
            return []
        out = []
        et = ptr.edge_times
        cols = et.columns.tolist()
        start_col = next((c for c in ['start', 't_start', 'start_time', 'start_s'] if c in cols), None)
        stop_col = next((c for c in ['stop', 't_end', 'end_time', 'stop_s', 'end'] if c in cols), None)
        if start_col and stop_col:
            for _, row in et.iterrows():
                out.append((float(row[start_col]), float(row[stop_col]), str(row['label'])))
        else:
            t = 0.0
            for _, row in et.iterrows():
                dur = float(row['effective_time_hours']) * 3600.0
                out.append((t, t + dur, str(row['label'])))
                t += dur
        return out

    @staticmethod
    def _segment_bounds_time_extent(seg_bounds: list) -> tuple[float, float]:
        """Min start / max end over segment rows (table order may differ from wall-clock order)."""
        if not seg_bounds:
            return (0.0, 0.0)
        return (min(b[0] for b in seg_bounds), max(b[1] for b in seg_bounds))

    def _union_span_for_segment_label(self, key, label: str) -> tuple[float, float] | None:
        """Absolute [t0, t1] covering all edge_times rows whose label matches *label*."""
        bounds = self._segment_bounds_for_key(key)
        if not bounds:
            return None
        want = str(label)
        spans = [(s, e) for s, e, lbl in bounds if str(lbl) == want]
        if not spans:
            return None
        return (min(s for s, _ in spans), max(e for _, e in spans))

    def _single_exclusive_segment_filter_label(self, filter_state: dict) -> str | None:
        """If theme is ``segments`` and exactly one behavioral label is ON, return that label."""
        fs = filter_state or {}
        if str(fs.get('theme', 'segments')) != 'segments':
            return None
        labels = fs.get('labels') or {}
        if not labels or all(bool(v) for v in labels.values()):
            return None
        active = [str(k) for k, v in labels.items() if v and str(k).upper() != 'NONE']
        if len(active) != 1:
            return None
        return active[0]

    @staticmethod
    def _suppress_legacy_post_split_suggestion_name(name: str) -> bool:
        """Strip legacy auto-generated post-window suggestion names from lists/UI."""
        n = re.sub(r'[\s_]+', '', str(name).strip().lower())
        return n in ('post1', 'post2', 'post1st', 'post2nd', 'postfirst', 'postsecond')

    def _theme_bounds_for_key(self, key):
        theme = str(getattr(self, '_ts_current_theme', 'segments'))
        if theme == 'segments':
            ptr = self.cd.data.get(key)
            if ptr is None or getattr(ptr, 'edge_times', None) is None:
                return []
            out = []
            et = ptr.edge_times
            cols = et.columns.tolist()
            def _find_col(candidates):
                for c in candidates:
                    if c in cols:
                        return c
                return None
            start_col = _find_col(['start', 't_start', 'start_time', 'start_s'])
            stop_col = _find_col(['stop', 't_end', 'end_time', 'stop_s', 'end'])
            if start_col and stop_col:
                for _, row in et.iterrows():
                    out.append((float(row[start_col]), float(row[stop_col]), str(row['label'])))
            else:
                t = 0.0
                for _, row in et.iterrows():
                    dur = float(row['effective_time_hours']) * 3600.0
                    out.append((t, t + dur, str(row['label'])))
                    t += dur
            return out
        sess_obj = self._session_obj_for_nd_key(key.nd())
        if sess_obj is None:
            return None
        epoch = getattr(sess_obj, theme, None)
        if _Epoch is None or epoch is None or not isinstance(epoch, _Epoch) or epoch.n_epochs <= 0:
            return None
        out = []
        labs = [str(x).strip() for x in epoch.labels]
        for s, e, lb in zip(epoch.starts, epoch.stops, labs):
            out.append((float(s), float(e), lb))
        unique_nonblank = {lb for lb in labs if lb}
        if len(unique_nonblank) == 0:
            out = [(s, e, theme) for s, e, _ in out]
        elif len(unique_nonblank) <= 1:
            out = [(s, e, theme) for s, e, _ in out]
        return out

    def _intervals_for_spec_on_key(self, spec: dict, key):
        bounds = self._theme_bounds_for_key(key)
        if bounds is None:
            print(f"[CustomCCG] missing behavioral epochs: {key.session}")
            return None
        # Resolve sentinels per-session: 'start' → bounds start, 'end' → bounds end
        seg_bounds = self._segment_bounds_for_key(key)
        t_sess_start, t_sess_end = self._segment_bounds_time_extent(seg_bounds)
        if not seg_bounds:
            t_sess_end = float(getattr(self, '_ts_total_sec', 0.0))
        t0 = self._resolve_ts_time(spec.get('t0', 0.0), t_sess_start, t_sess_end)
        t1 = self._resolve_ts_time(spec.get('t1', t_sess_end), t_sess_start, t_sess_end)
        lone = self._single_exclusive_segment_filter_label(spec.get('filter_state', {}))
        if lone is not None:
            span = self._union_span_for_segment_label(key, lone)
            if span is not None:
                t0, t1 = span[0], span[1]
        labels = ((spec.get('filter_state') or {}).get('labels') or {})
        if not labels or all(bool(v) for v in labels.values()):
            return (None, t1 - t0)
        active_labels = {str(lbl) for lbl, v in labels.items() if bool(v)}
        if not active_labels:
            print(f"[CustomCCG] no active labels: {key.session}")
            return False
        available_labels = {str(lbl) for _, _, lbl in bounds}
        none_active = 'NONE' in active_labels
        # Only labels that are both toggled ON and present on this session contribute
        # intervals. Selected labels absent here are ignored (narrower coverage).
        required_real = (active_labels - {'NONE'}) & available_labels
        if not required_real and not none_active:
            print(f"[CustomCCG] no active labels: {key.session}")
            return False
        real_labels = required_real
        intervals = []
        for s, e, lbl in bounds:
            if lbl in real_labels:
                ss, ee = max(s, t0), min(e, t1)
                if ee > ss:
                    intervals.append((ss, ee))
        if none_active:
            epoch_times = sorted(
                (max(s, t0), min(e, t1)) for s, e, _ in bounds if min(e, t1) > max(s, t0)
            )
            cursor = t0
            for es, ee in epoch_times:
                if es > cursor:
                    intervals.append((cursor, es))
                cursor = max(cursor, ee)
            if cursor < t1:
                intervals.append((cursor, t1))
        # Keep comparable label constraints even if the selected range has no overlap.
        # Empty intervals are valid and produce an empty restricted spike selection.
        return (intervals, float(sum(e - s for s, e in intervals)))

    def _custom_npz_spec(self, path: str) -> dict | None:
        try:
            npz = np.load(path, allow_pickle=False)
            return {
                'name': str(npz['name_']),
                't0': float(npz['t0_']),
                't1': float(npz['t1_']),
                'filter_state': (json.loads(str(npz['filter_state_']))
                                 if 'filter_state_' in npz else {}),
            }
        except Exception:
            return None

    def _custom_ccg_name_session_coverage(self) -> tuple[dict[str, set[str]], int]:
        """Logical custom CCG name -> sessions that have an npz with that name (full cache scan)."""
        pattern = os.path.join(self._ccg_cache_dir, "*.npz")
        by_name: dict[str, set[str]] = {}
        for p in sorted(_glob.glob(pattern)):
            base = os.path.basename(p)
            sess = base.split("__", 1)[0] if "__" in base else ""
            if not sess:
                continue
            spec = self._custom_npz_spec(p)
            if not spec:
                continue
            nm = str(spec.get('name', '')).strip()
            if not nm or self._suppress_legacy_post_split_suggestion_name(nm):
                continue
            by_name.setdefault(nm, set()).add(sess)
        n_tot = len(self._real_nd_keys_ordered())
        return by_name, n_tot

    def _custom_segment_disk_session(self, cs: dict) -> str:
        """Session string for a loaded/saved custom segment (metadata or npz filename)."""
        md = cs.get('metadata') or {}
        if md.get('session') is not None:
            return str(md['session'])
        sp = cs.get('src_path')
        if sp:
            bn = os.path.basename(sp)
            if "__" in bn:
                return bn.split("__", 1)[0]
        return str(self.key.session)

    def _bind_custom_segments_to_session(self, sess: str):
        """Point ``_custom_segments`` at the in-memory list for session *sess*."""
        self._custom_segments = self._custom_segments_by_session.setdefault(str(sess), [])

    def _key_for_custom_segment_save(self, cs: dict):
        """``Key`` for npz filenames: segment's session + same connection-type label as UI."""
        want_sess = self._custom_segment_disk_session(cs)
        cur_lbl = self._type_label(self.key)
        for k in self.cd.data.keys():
            if str(k.session) == want_sess and self._type_label(k) == cur_lbl:
                return k
        for k in self.cd.data.keys():
            if str(k.session) == want_sess:
                return k
        return self.key

    def _enqueue_custom_ccg_task(self, *, key, t0, t1, name, intervals,
                                 active_duration, filter_state, metadata,
                                 auto_save: bool, load_into_ui: bool,
                                 split_batch_id: int | None = None) -> bool:
        running = 1 if self._custom_ccg_is_running() else 0
        total = running + len(self._custom_ccg_pending)
        if total >= _MAX_JITTER_QUEUE:
            messagebox.showwarning(
                "Task queue full",
                f"Custom CCG queue full ({total}/{_MAX_JITTER_QUEUE}). "
                "Wait for running tasks to complete.")
            return False
        self._custom_ccg_pending.append({
            'kind': 'custom_ccg',
            'key': key,
            't0': float(t0),
            't1': float(t1),
            'name': str(name),
            'intervals': intervals,
            'active_duration': active_duration,
            'filter_state': filter_state or {},
            'metadata': metadata or {},
            'auto_save': bool(auto_save),
            'load_into_ui': bool(load_into_ui),
            'split_batch_id': split_batch_id,
        })
        return True

    def _iter_type_keys_for_all_sessions(self):
        lbl = self._type_label(self.key)
        out = []
        for nk in self._real_nd_keys_ordered():
            tk_ = self._type_key_for_nd(nk)
            if tk_ is not None and self._type_label(tk_) == lbl:
                out.append(tk_)
        return out

    def _queue_custom_ccg_for_spec(self, spec: dict, *, for_all: bool, auto_save: bool,
                                    target_sessions: list | None = None) -> int:
        """Enqueue custom CCG tasks for the given spec.

        target_sessions: explicit list of session strings to target (from picker dialog).
        for_all: if True (and target_sessions is None), targets all sessions.
        """
        queued = 0
        if target_sessions is not None:
            sess_set = set(str(s) for s in target_sessions)
            targets = []
            for nk in self._real_nd_keys_ordered():
                if str(nk.session) in sess_set:
                    tk_ = self._type_key_for_nd(nk)
                    if tk_ is not None:
                        targets.append(tk_)
        elif for_all:
            targets = self._iter_type_keys_for_all_sessions()
        else:
            requested_sessions = set(str(s) for s in (spec.get('sessions') or []) if s != 'All')
            if requested_sessions:
                targets = []
                for nk in self._real_nd_keys_ordered():
                    if str(nk.session) in requested_sessions:
                        tk_ = self._type_key_for_nd(nk)
                        if tk_ is not None:
                            targets.append(tk_)
            else:
                targets = [self.key]
        n_splits = max(1, int(spec.get('n_splits') or 1))
        overlap_sec = max(0.0, float(spec.get('overlap_sec') or 0.0))
        scope_label = 'All' if (for_all and target_sessions is None) else str(spec.get('scope', ''))
        for tk_ in targets:
            # Resolve sentinels per-session
            seg_bounds = self._segment_bounds_for_key(tk_)
            t_sess_start, t_sess_end = self._segment_bounds_time_extent(seg_bounds)
            if not seg_bounds:
                t_sess_end = 0.0
            t0_r = self._resolve_ts_time(spec.get('t0', 0.0), t_sess_start, t_sess_end)
            t1_r = self._resolve_ts_time(spec.get('t1', t_sess_end), t_sess_start, t_sess_end)
            lone = self._single_exclusive_segment_filter_label(spec.get('filter_state', {}))
            if lone is not None:
                span = self._union_span_for_segment_label(tk_, lone)
                if span is not None:
                    t0_r, t1_r = span[0], span[1]
            chunks = self._split_time_range(t0_r, t1_r, n_splits, overlap_sec, str(spec['name']))
            split_bid = None
            if len(chunks) > 1 and str(tk_.session) == str(self.key.session):
                split_bid = self._split_batch_next_id
                self._split_batch_next_id += 1
            split_names: list[str] = []
            for chunk_t0, chunk_t1, chunk_name in chunks:
                chunk_spec = dict(spec, name=chunk_name, t0=chunk_t0, t1=chunk_t1)
                iv = self._intervals_for_spec_on_key(chunk_spec, tk_)
                if iv is None or iv is False:
                    continue
                intervals, active_duration = iv
                metadata = {
                    'name': chunk_name,
                    'theme': (spec.get('filter_state') or {}).get('theme', 'segments'),
                    'labels': (spec.get('filter_state') or {}).get('labels', {}),
                    'scope': scope_label,
                    'session': str(tk_.session),
                    'timing': {'t0': chunk_t0, 't1': chunk_t1},
                }
                ok = self._enqueue_custom_ccg_task(
                    key=tk_,
                    t0=chunk_t0,
                    t1=chunk_t1,
                    name=chunk_name,
                    intervals=intervals,
                    active_duration=active_duration,
                    filter_state=spec.get('filter_state') or {},
                    metadata=metadata,
                    auto_save=auto_save,
                    load_into_ui=(str(tk_.session) == str(self.key.session)),
                    split_batch_id=split_bid,
                )
                if ok:
                    queued += 1
                    if split_bid is not None:
                        split_names.append(chunk_name)
            if split_bid is not None and split_names:
                self._split_batch_counts[split_bid] = len(split_names)
                self._split_batch_chunk_names[split_bid] = split_names
        return queued

    # ------------------------------------------------------------------
    # Custom-window CCG
    # ------------------------------------------------------------------

    def _ts_brain_state_intervals(self, t0, t1):
        """Validate brain-state toggles and return the active intervals (main-thread, fast).

        Returns:
            (None, t1-t0)        — no restriction (all toggles on or no toggles)
            (intervals, active_sec) — filtered intervals list with total active seconds
            False                 — abort (no active labels or no intervals in range)
        """
        toggles = getattr(self, '_ts_legend_toggles', {})
        if not toggles or all(v.get() for v in toggles.values()):
            return (None, t1 - t0)
        active_labels = {lbl for lbl, v in toggles.items() if v.get()}
        if not active_labels:
            messagebox.showwarning("Brain-state filter",
                                   "All epoch labels are toggled off. "
                                   "Enable at least one label to compute CCG.")
            return False
        if self.neurons is None:
            return (None, t1 - t0)
        available_labels = {lbl for _, _, lbl in self._ts_epoch_bounds}
        none_active = 'NONE' in active_labels
        real_active = active_labels - {'NONE'}
        real_labels = real_active & available_labels
        intervals = []
        for s, e, lbl in self._ts_epoch_bounds:
            if lbl in real_labels:
                s_clipped, e_clipped = max(s, t0), min(e, t1)
                if e_clipped > s_clipped:
                    intervals.append((s_clipped, e_clipped))
        if none_active:
            epoch_times = sorted(
                (max(s, t0), min(e, t1))
                for s, e, _ in self._ts_epoch_bounds
                if min(e, t1) > max(s, t0))
            cursor = t0
            for es, ee in epoch_times:
                if es > cursor:
                    intervals.append((cursor, es))
                cursor = max(cursor, ee)
            if cursor < t1:
                intervals.append((cursor, t1))
        if not intervals:
            messagebox.showwarning("Brain-state filter",
                                   "No active epoch intervals in the selected time range.")
            return False
        active_sec = sum(e - s for s, e in intervals)
        return (intervals, active_sec)

    def _ts_apply_brain_state_intervals(self, intervals, t0, t1, neurons_obj=None):
        """Filter self.neurons to the given intervals and return a new Neurons object.
        Called in the background worker thread — deepcopy stays off the main thread.
        """
        source_neurons = neurons_obj if neurons_obj is not None else self.neurons
        neurons = deepcopy(source_neurons)
        filtered_trains = []
        for st in neurons.spiketrains:
            mask = np.zeros(len(st), dtype=bool)
            for s, e in intervals:
                mask |= (st >= s) & (st <= e)
            filtered_trains.append(st[mask])
        return Neurons(
            spiketrains=filtered_trains,
            t_stop=t1, t_start=t0,
            sampling_rate=neurons.sampling_rate,
            neuron_ids=neurons.neuron_ids,
            neuron_type=neurons.neuron_type,
            waveforms=neurons.waveforms,
            waveforms_amplitude=neurons.waveforms_amplitude,
            peak_channels=getattr(neurons, 'peak_channels', None),
            shank_ids=getattr(neurons, 'shank_ids', None),
            metadata=neurons.metadata,
        )

    def _compute_custom_segment(self, t0: float, t1: float, name: str,
                                neurons_override=None, active_duration=None,
                                key_override=None, neurons_obj=None,
                                ccg_data_obj=None, metadata=None):
        """Compute full CCG pipeline for a custom time window.

        Runs spike_correlations → EranConv._conv → multiple_correction
        at **both** low-res (1 ms) and high-res (0.1 ms) bin sizes so
        that Ctrl+R resolution toggle works on custom segments too.

        Returns a dict with keys: name, t0, t1, ccg, ccg_null, pval,
        pval_corrected (low-res), firing_rates, and optionally ccg_hi,
        ccg_null_hi, pval_hi, pval_corrected_hi — or None on failure.
        """
        key_eff = key_override or self.key
        neurons_eff = neurons_obj if neurons_obj is not None else self.neurons
        cd_eff = ccg_data_obj if ccg_data_obj is not None else self.ccg_data
        if neurons_eff is None:
            messagebox.showerror("Custom CCG", "No neuron data available.")
            return None
        try:

            neurons_slice = (neurons_override if neurons_override is not None
                            else neurons_eff.time_slice(t0, t1))
            if active_duration is None:
                active_duration = t1 - t0
            conf = cd_eff.conf
            n_neurons = neurons_eff.n_neurons
            neuron_inds = np.arange(n_neurons)
            method = conf.multiple_correction if conf.multiple_correction is not None else 'bonferroni'
            ei = getattr(key_eff, 'excitability', 'E')

            def _run_pipeline(bin_size, label):
                print(f"[CustomSegment] computing {label} CCG for {name} "
                      f"({t1-t0:.1f}s, {n_neurons} neurons, "
                      f"bin={bin_size*1e3:.2f}ms) ...")
                ccg = spike_correlations(
                    neurons=neurons_slice,
                    neuron_inds=neuron_inds,
                    bin_size=bin_size,
                    window_size=conf.duration,
                    symmetrize=conf.symmetrize_ccg,
                    use_acceleration=conf.use_acceleration,
                )
                ccg = ccg[np.newaxis, ...]
                # Compute W from the actual bin_size used, not from conf
                # (conf.bin_size may be mutated to high-res)
                W = conf.conv_window / bin_size
                pvals, pred, qvals = EranConv._conv(
                    ccg, W=W, wintype="gauss",
                    hollow_frac=None)
                p_raw = pvals if ei == 'E' else qvals
                _, pval_corrected = EranConv.multiple_correction(
                    p_raw, conf.alpha, method=method)
                print(f"[CustomSegment] {label} done. shape={ccg.shape}")
                return ccg, pred, p_raw, pval_corrected

            # Low-res (always)
            lo_bs = _CCG_RESOLUTION['lowres']
            ccg_lo, pred_lo, pval_lo, pvalc_lo = _run_pipeline(lo_bs, 'lowres')

            # Firing rates over the active duration
            firing_rates = np.array(
                [len(st) for st in neurons_slice.spiketrains],
                dtype=float) / max(active_duration, 1e-9)

            result = {
                'name':              name,
                't0':                t0,
                't1':                t1,
                'ccg':               ccg_lo,
                'ccg_null':          pred_lo,
                'pval':              pval_lo,
                'pval_corrected':    pvalc_lo,
                'firing_rates':      firing_rates,
                'active_duration':   active_duration,
                # total_time_hours: active recording time in hours — needed for
                # TIME_SPAN normalisation on custom segments.
                'total_time_hours':  active_duration / 3600.0,
                'metadata':          metadata or {},
            }

            # High-res (only if highres data is available for this session)
            has_highres = (hasattr(self.cd, '_ccg_highres') and
                           bool(self.cd._ccg_highres))
            if has_highres:
                hi_bs = _CCG_RESOLUTION['highres']
                ccg_hi, pred_hi, pval_hi, pvalc_hi = _run_pipeline(
                    hi_bs, 'highres')
                result['ccg_hi'] = ccg_hi
                result['ccg_null_hi'] = pred_hi
                result['pval_hi'] = pval_hi
                result['pval_corrected_hi'] = pvalc_hi

            return result
        except Exception as ex:
            print(f"[CustomSegment] ERROR: {ex}")
            traceback.print_exc()
            messagebox.showerror("Custom CCG",
                                 f"Error computing CCG:\n{ex}")
            return None

    # ------------------------------------------------------------------
    # Probe network
    # ------------------------------------------------------------------

    def _get_neuron_positions(self, x_scale=1.0, y_scale=1.0):
        """Return (x, y, peak_channels) or None.

        Maps each neuron to its peak channel's (x, y) on the ProbeGroup,
        scaled by x_scale / y_scale to match plot_probe rendering.
        """
        neurons = self.neurons
        if neurons is None or neurons.peak_channels is None:
            return None

        pgs = getattr(getattr(self.cd, 'nd', None), 'probegroups', {})
        nd_key = self.key.nd()
        pg = pgs.get(nd_key)
        if pg is None:
            print(f"[ProbeNetwork] No ProbeGroup for key={nd_key}. Available keys: {list(pgs.keys())}")
            return None

        peak_ch = np.asarray(neurons.peak_channels)
        pg_df = pg.to_dataframe()
        # channel_id -> (x, y) lookup (raw probe coordinates)
        ch_to_xy = {int(row['channel_id']): (float(row['x']), float(row['y']))
                     for _, row in pg_df.iterrows()}

        x = np.zeros(len(peak_ch))
        y = np.zeros(len(peak_ch))
        n_miss = 0
        for i, ch in enumerate(peak_ch):
            xy = ch_to_xy.get(int(ch))
            if xy is not None:
                x[i] = xy[0] * x_scale
                y[i] = xy[1] * y_scale
            else:
                n_miss += 1
        if n_miss:
            print(f"[ProbeNetwork] WARNING: {n_miss}/{len(peak_ch)} neurons "
                  f"have peak_channels not found in ProbeGroup.channel_id. "
                  f"peak_ch sample={peak_ch[:5]}, "
                  f"pg_ch sample={list(ch_to_xy.keys())[:5]}")
        return x, y, peak_ch

    def _pairs_for_segment_filter(self):
        if self.active_segment_filter is None:
            return set(map(tuple, self.all_inds))
        pt = self.ccg_pointer
        if pt.stored_by_segment:
            seg_i = self.active_segment_filter
            mask = pt.inds[:, 0] == seg_i
            return set(map(tuple, pt.inds[mask, -2:]))
        return set(map(tuple, self.all_inds))

    def _draw_network(self):
        if not self._panel_vars.get('Probe Network', tk.BooleanVar(value=True)).get():
            return
        self.network_panel.draw()

    def _draw_network_impl(self):
        # Delegated — implementation lives in NetworkPanel._draw_impl
        self.network_panel.draw()

    def _on_network_pick(self, event):
        # Delegated to NetworkPanel
        self.network_panel._on_network_pick(event)

    def _on_net_scroll(self, event):
        # Delegated to NetworkPanel
        self.network_panel._on_net_scroll(event)

    # ------------------------------------------------------------------
    # Neuron focus (Part II.1)
    # ------------------------------------------------------------------

    def _on_neuron_focus(self):
        val = self.network_panel._focus_var.get().strip()
        print(f"[CCGReviewUI] _on_neuron_focus called, val={val!r}")
        if not val:
            self._on_neuron_focus_clear()
            return
        try:
            nid = int(val)
        except ValueError:
            messagebox.showerror("Neuron focus", f"Invalid neuron id: {val!r}")
            return
        if self.neurons is not None:
            if nid < 0 or nid >= self.neurons.n_neurons:
                messagebox.showerror("Neuron focus",
                                     f"Neuron {nid} out of range "
                                     f"[0, {self.neurons.n_neurons-1}]")
                return
        self._focused_neuron = nid
        self._focused_pair = None
        self.network_panel._focus_pair_var.set("")
        self.network_panel._focus_pair_info_var.set("")
        self._update_focus_info(nid)
        self.refresh_lists()
        self._draw_network()

    def _update_focus_info(self, nid):
        """Update focus info label with current-type and total connection counts."""
        # Current type counts
        cur_out = sum(1 for r, t in map(tuple, self.all_inds) if r == nid)
        cur_in  = sum(1 for r, t in map(tuple, self.all_inds) if t == nid)
        # Total across all types
        tot_out, tot_in = 0, 0
        for tk_ in self._available_type_keys(self.key.nd()):
            pt = self.cd.data.get(tk_)
            if pt is None or pt.inds is None:
                continue
            arr = pt.inds[:, -2:]
            tot_out += sum(1 for r, t in set(map(tuple, arr)) if r == nid)
            tot_in  += sum(1 for r, t in set(map(tuple, arr)) if t == nid)
        ct_label = self._type_label(self.key)
        self.network_panel._focus_info_var.set(
            f"{ct_label}: in={cur_in} out={cur_out}  |  all: in={tot_in} out={tot_out}")

    def _on_neuron_focus_clear(self):
        self._focused_neuron = None
        self.network_panel._focus_var.set("")
        self.network_panel._focus_info_var.set("")
        self.refresh_lists()
        self._draw_network()

    def _on_pair_focus(self):
        """Set focus to a specific (ref, tgt) pair. Clears neuron focus."""
        val = self.network_panel._focus_pair_var.get().strip()
        if not val:
            self._on_pair_focus_clear()
            return
        try:
            parts = val.replace(' ', '').split(',')
            ref, tgt = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            messagebox.showerror("Pair focus",
                                 f"Invalid pair format: {val!r}\nUse ref,tgt (e.g. 1,170)")
            return
        # Validate neuron indices exist
        if self.neurons is not None:
            n = self.neurons.n_neurons
            if ref < 0 or ref >= n or tgt < 0 or tgt >= n:
                messagebox.showerror("Pair focus",
                                     f"Neuron index out of range [0, {n-1}]")
                return
        # Check if pair exists in any type
        pair = (ref, tgt)
        pair_exists = False
        for tk_ in self._available_type_keys(self.key.nd()):
            pt = self.cd.data.get(tk_)
            if pt is None or pt.inds is None:
                continue
            if pair in set(map(tuple, pt.inds2)):
                pair_exists = True
                break
        if not pair_exists:
            self._show_temp_warning(f"Pair ({ref},{tgt}) not significant — showing position")
        self._focused_pair = pair
        self._focused_neuron = None
        self.network_panel._focus_var.set("")
        self.network_panel._focus_info_var.set("")
        # Update pair info
        self._update_pair_focus_info(pair, pair_exists)
        self.refresh_lists()
        self._draw_network()
        self.update_plot()

    def _update_pair_focus_info(self, pair, exists):
        """Update the pair focus info label and 'Add to available' button."""
        ref, tgt = pair
        if self.neurons is not None:
            nt = self.neurons.neuron_type
            ref_type = nt[ref] if nt is not None and ref < len(nt) else '?'
            tgt_type = nt[tgt] if nt is not None and tgt < len(nt) else '?'
            # Check if already in all_inds (sig or admitted)
            in_available = pair in set(map(tuple, self.all_inds))
            status = "sig" if exists else ("admitted" if in_available else "not sig")
            self.network_panel._focus_pair_info_var.set(
                f"{ref}({ref_type})→{tgt}({tgt_type}) [{status}]")
        else:
            in_available = pair in set(map(tuple, self.all_inds))
            self.network_panel._focus_pair_info_var.set(f"{ref}→{tgt}")
        # Enable "Add to available" only for pairs not already available
        if hasattr(self, 'network_panel'):
            self.network_panel._add_pair_btn.config(
                state=tk.NORMAL if not in_available else tk.DISABLED)

    def _on_pair_focus_clear(self):
        self._focused_pair = None
        self.network_panel._focus_pair_var.set("")
        self.network_panel._focus_pair_info_var.set("")
        if hasattr(self, 'network_panel'):
            self.network_panel._add_pair_btn.config(state=tk.DISABLED)
        self.refresh_lists()
        self._draw_network()
        self.update_plot()

    def _on_add_focused_pair(self):
        """Add the currently focused non-significant pair to available pairs."""
        pair = self._focused_pair
        if pair is None:
            return
        ref, tgt = pair
        # Validate indices are within CCG data shape
        cd = self.ccg_data
        if cd is None or cd.ccg is None:
            messagebox.showerror("Add pair", "No CCG data loaded.")
            return
        if ref >= cd.ccg.shape[1] or tgt >= cd.ccg.shape[2]:
            messagebox.showerror("Add pair",
                                 f"Pair ({ref},{tgt}) out of CCG data range "
                                 f"({cd.ccg.shape[1]}x{cd.ccg.shape[2]}).")
            return
        # Add to admitted group
        self._push_undo()
        self._group_add_pair(_ADMITTED_GROUP, pair)
        # Add to unselected (now a valid available pair)
        self.unselected_inds.add(pair)
        # Navigate to the pair
        self.current_pair_idx = self.get_pair_index(pair)
        # Clear focus and update UI
        self._focused_pair = None
        self.network_panel._focus_pair_var.set("")
        self._update_pair_focus_info(pair, exists=False)
        self.network_panel._add_pair_btn.config(state=tk.DISABLED)
        self.refresh_lists()
        self._draw_network()
        self.update_plot()

    def _on_net_toggle_arrows(self):
        self.network_panel._on_toggle_arrows()

    def _on_net_toggle_hide(self):
        self.network_panel._on_toggle_hide()

    def _on_net_toggle_hide_same_channel(self):
        self.network_panel._on_toggle_hide_same_channel()

    def _on_net_toggle_hide_same_shank(self):
        self.network_panel._on_toggle_hide_same_shank()

    def _on_net_group_toggle(self, group_name):
        self.network_panel._on_group_toggle(group_name)

    def _on_net_group_clear(self):
        self.network_panel._on_group_clear()

    def _on_net_save_selections_to_group(self):
        self.network_panel._on_save_selections_to_group()

    def _refresh_net_group_buttons(self):
        self.network_panel.refresh_group_buttons()

    def _rewrap_group_buttons(self):
        self.network_panel.rewrap_group_buttons()

    def _refresh_net_shank_buttons(self):
        self.network_panel.refresh_shank_buttons()

    def _on_net_zoom(self, _=None):
        self.network_panel._on_zoom()

    # ------------------------------------------------------------------
    # Modified _draw_network (neuron focus support)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Group helpers (Part II.2)
    # ------------------------------------------------------------------

    def _pair_group_label(self, inds) -> str:
        """Return a short label like '[G1,G2]' for the groups this pair belongs to."""
        sess, pair = self._pair_sess_rt(inds)
        pair = tuple(int(x) for x in pair)
        labels = []
        for gname in self._groups:
            if pair not in self._group_pairs(gname, session=sess):
                continue
            if gname.startswith(_SPECIAL_PREFIX):
                labels.append('*' + gname[len(_SPECIAL_PREFIX):])
            elif not gname.startswith('__'):
                labels.append(gname)
        pt = self._pair_tags.get(pair, {})
        tag_mark = '~' if (pt.get('tags') or pt.get('notes', '').strip()) else ''
        group_str = f"[{','.join(labels)}]" if labels else ""
        return tag_mark + group_str

    def _toggle_pair_group(self, pair, group_name):
        if group_name not in self._groups:
            self._groups[group_name] = {}
        sess, p2 = self._pair_sess_rt(pair)
        if p2 in self._group_pairs(group_name, session=sess):
            self._group_discard_pair(group_name, p2, session=sess)
        else:
            self._group_add_pair(group_name, p2, session=sess)
        # Preserve scroll positions so the list doesn't jump to the top
        unsel_scroll = self.unselected_list.yview()[0]
        sel_scroll = self.selected_list.yview()[0]
        self.refresh_lists()
        self.unselected_list.yview_moveto(unsel_scroll)
        self.selected_list.yview_moveto(sel_scroll)

    def _toggle_pairs_group(self, pairs, group_name):
        """Toggle multiple pairs in/out of a group.

        If ALL pairs are already in the group, remove them all.
        Otherwise, add all pairs to the group.
        """
        if group_name not in self._groups:
            self._groups[group_name] = {}
        all_in = all(self._pair_in_group(p, group_name) for p in pairs)
        # action = "REMOVE" if all_in else "ADD"
        if all_in:
            for p in pairs:
                s2, p2 = self._pair_sess_rt(p)
                self._group_discard_pair(group_name, p2, session=s2)
        else:
            for p in pairs:
                s2, p2 = self._pair_sess_rt(p)
                self._group_add_pair(group_name, p2, session=s2)
        unsel_scroll = self.unselected_list.yview()[0]
        sel_scroll = self.selected_list.yview()[0]
        self.refresh_lists()
        self.unselected_list.yview_moveto(unsel_scroll)
        self.selected_list.yview_moveto(sel_scroll)

    def _create_group_dialog(self):
        name = simpledialog.askstring(
            "Create group", "Group name:", parent=self.root)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if name in self._groups:
            messagebox.showinfo("Create group", f"Group '{name}' already exists.")
            return
        self._groups[name] = {}
        self._rebuild_groups_menu()
        self.refresh_lists()

    def _create_special_group_dialog(self):
        name = simpledialog.askstring(
            "Create special group", "Special group name:", parent=self.root)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        full_name = _SPECIAL_PREFIX + name
        if full_name in self._groups:
            messagebox.showinfo("Create special group",
                                f"Special group '{name}' already exists.")
            return
        self._groups[full_name] = {}
        self._rebuild_groups_menu()
        self.refresh_lists()

    def _pair_tags_dialog(self):
        """Dialog to view/edit tags and notes for the current pair."""
        if self.current_pair_idx >= len(self.all_inds):
            messagebox.showinfo("Pair tags", "No pair selected.")
            return
        inds = tuple(self.all_inds[self.current_pair_idx])
        ref, tgt = int(inds[0]), int(inds[1])
        tag_data = self._pair_tags.get((ref, tgt), {})

        win = tk.Toplevel(self.root)
        win.title(f"Pair Tags — [{ref}, {tgt}]")
        win.geometry("400x350")
        win.transient(self.root)
        win.grab_set()

        # Tags (comma-separated)
        ttk.Label(win, text="Tags (comma-separated):").pack(
            anchor='w', padx=8, pady=(8, 0))
        tags_var = tk.StringVar(value=', '.join(tag_data.get('tags', [])))
        ttk.Entry(win, textvariable=tags_var, width=50).pack(
            fill=tk.X, padx=8, pady=2)

        # Notes
        ttk.Label(win, text="Notes:").pack(anchor='w', padx=8, pady=(8, 0))
        notes_text = tk.Text(win, height=12, width=50, font=('Arial', 9),
                             wrap=tk.WORD)
        notes_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)
        notes_text.insert('1.0', tag_data.get('notes', ''))

        def _save():
            tags = [t.strip() for t in tags_var.get().split(',') if t.strip()]
            notes = notes_text.get('1.0', 'end-1c')
            if tags or notes:
                # Preserve 'groups' field — that is managed separately via group toggles
                existing_groups = self._pair_tags.get((ref, tgt), {}).get('groups', [])
                entry = {'tags': tags, 'notes': notes}
                if existing_groups:
                    entry['groups'] = existing_groups
                self._pair_tags[(ref, tgt)] = entry
            elif (ref, tgt) in self._pair_tags:
                del self._pair_tags[(ref, tgt)]
            self.refresh_lists()
            self._select_pair_in_list((ref, tgt))
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(btn_frame, text="Save", command=_save).pack(
            side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(
            side=tk.RIGHT)

    def _pairs_by_conn_type(self, session_str, pairs):
        """Group pairs by their conn_type for display. Returns {label: [pairs]}."""
        # Build pair → conn_type mapping from cd.data keys for this session
        pair_ct = {}
        for key in self.cd.data:
            if str(key.session) != session_str and str(key.nd().session) != session_str:
                continue
            ct = getattr(key, 'conn_type', None)
            ct_label = (f"{ct[0]}\u2192{ct[1]}" if ct else "unknown")
            pt = self.cd.data[key]
            if pt is None or not hasattr(pt, 'inds2') or pt.inds2 is None:
                continue
            for ref, tgt in map(tuple, pt.inds2):
                pair_ct[(ref, tgt)] = ct_label
        # Fallback: look up pairs not found in cd.data from saved JSON selections.
        # Selection keys encode conn_type: 'sess_{S}.ex_{E}.type_{ref_type}-{tgt_type}'
        needs_lookup = [p for p in pairs if tuple(p) not in pair_ct]
        if needs_lookup:
            fb = self._json_pair_ct_fallback(session_str)
            for ref, tgt in map(tuple, needs_lookup):
                if (ref, tgt) in fb:
                    pair_ct[(ref, tgt)] = fb[(ref, tgt)]
        # Group the requested pairs
        result = collections.OrderedDict()
        for pair in sorted(pairs):
            ct_label = pair_ct.get(tuple(pair), "unknown")
            result.setdefault(ct_label, []).append(pair)
        return result

    def _conn_type_label(self, ct) -> str:
        """Return conn-type label matching _pairs_by_conn_type keys (e.g. 'pyr→inter')."""
        if ct is None:
            return "unknown"
        try:
            _map = {'pyr': 'PYR', 'inter': 'INT', 'int': 'INT'}
            a = _map.get(str(ct[0]).lower(), str(ct[0]).upper())
            b = _map.get(str(ct[1]).lower(), str(ct[1]).upper())
            return f"{a}→{b}"
        except Exception:
            return str(ct)

    def _filter_pairs_to_conn_types(self, session_str: str, pairs, allowed_labels: set[str]):
        """Return subset of pairs whose conn_type label is in allowed_labels."""
        if not pairs or not allowed_labels:
            return []
        grouped = self._pairs_by_conn_type(session_str, pairs)
        out = []
        for lbl in allowed_labels:
            out.extend(grouped.get(lbl, []))
        return out

    def _json_pair_ct_fallback(self, session_str):
        """Build (ref,tgt)->conn_type_label from JSON selection files for a session.
        Result is cached per session string."""
        cache_attr = '_json_ct_cache'
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
        cache = getattr(self, cache_attr)
        if session_str in cache:
            return cache[session_str]
        result = {}
        try:
            for fname in os.listdir(self._sel_save_dir):
                if not fname.endswith('.json'):
                    continue
                fpath = os.path.join(self._sel_save_dir, fname)
                try:
                    with open(fpath, encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    continue
                for sel_key, sel_pairs in data.get('selections', {}).items():
                    # key format: sess_{session}.ex_{E/I}.type_{pyr/inter}-{pyr/inter}
                    if '.type_' not in sel_key or '.ex_' not in sel_key:
                        continue
                    sess = sel_key.split('.ex_')[0].replace('sess_', '', 1)
                    if sess != session_str:
                        continue
                    raw_ct = sel_key.split('.type_')[1]  # e.g. 'pyr-pyr'
                    parts = raw_ct.split('-', 1)
                    if len(parts) == 2:
                        ct_label = self._conn_type_label((parts[0], parts[1]))
                    else:
                        ct_label = raw_ct
                    for p in sel_pairs:
                        if len(p) >= 2:
                            result[(int(p[0]), int(p[1]))] = ct_label
        except Exception:
            pass
        cache[session_str] = result
        return result

    def _manage_groups_dialog(self):
        """Pop-up window to rename groups, view pairs, assign hotkeys."""
        if not self._groups:
            messagebox.showinfo("Manage groups", "No groups yet. Create one first.")
            return
        win = tk.Toplevel(self.root)
        win.title("Manage Groups")
        win.geometry("480x420")
        win.transient(self.root)
        win.grab_set()

        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        def _add_group_tab(nb, gname, is_special=False):
            display = gname[len(_SPECIAL_PREFIX):] if is_special else gname
            frame = ttk.Frame(nb)
            nb.add(frame, text=display)

            top = ttk.Frame(frame)
            top.pack(fill=tk.X, padx=6, pady=4)
            ttk.Label(top, text="Name:").pack(side=tk.LEFT)
            name_var = tk.StringVar(value=display)
            ttk.Entry(top, textvariable=name_var, width=18).pack(
                side=tk.LEFT, padx=4)
            def _do_rename(old=gname, nv=name_var, sp=is_special):
                new = nv.get().strip()
                if sp:
                    new = _SPECIAL_PREFIX + new
                self._rename_group(old, new, win)
            ttk.Button(top, text="Rename", command=_do_rename).pack(side=tk.LEFT)

            if not is_special:
                hk_frame = ttk.Frame(frame)
                hk_frame.pack(fill=tk.X, padx=6, pady=2)
                ttk.Label(hk_frame, text="Hotkey (1–9/0/a–z):").pack(side=tk.LEFT)
                hk_var = tk.StringVar(value=self._group_hotkeys.get(gname, ''))
                hk_entry = ttk.Entry(hk_frame, textvariable=hk_var, width=6)
                hk_entry.pack(side=tk.LEFT, padx=4)
                ttk.Button(hk_frame, text="Set",
                           command=lambda g=gname, hv=hk_var: self._set_group_hotkey(g, hv.get())).pack(
                    side=tk.LEFT)

            # Notes — larger for special groups
            ttk.Label(frame, text="Discussion notes:" if is_special else "Notes:"
                      ).pack(anchor='w', padx=6, pady=(4, 0))
            notes_h = 10 if is_special else 3
            notes_text = tk.Text(frame, height=notes_h, width=40,
                                 font=('Arial', 9), wrap=tk.WORD)
            notes_text.pack(fill=tk.BOTH if is_special else tk.X,
                            expand=is_special, padx=6, pady=2)
            notes_text.insert('1.0', self._group_notes.get(gname, ''))
            notes_text.bind('<KeyRelease>',
                            lambda e, g=gname, t=notes_text:
                            self._group_notes.__setitem__(g, t.get('1.0', 'end-1c')))

            ttk.Label(frame, text="Pairs in group:").pack(anchor='w', padx=6)
            lb_frame = ttk.Frame(frame)
            lb_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
            sb = ttk.Scrollbar(lb_frame)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            lb = tk.Listbox(lb_frame, yscrollcommand=sb.set, font=('Courier', 9))
            lb.pack(fill=tk.BOTH, expand=True)
            sb.config(command=lb.yview)
            g = self._groups.get(gname, {})
            if isinstance(g, dict):
                for sess in sorted(g):
                    pairs = g[sess]
                    if not pairs:
                        continue
                    # Group pairs by conn_type
                    ct_map = self._pairs_by_conn_type(sess, pairs)
                    for ct_label, ct_pairs in ct_map.items():
                        lb.insert(tk.END, f"── {sess} | {ct_label} ──")
                        lb.itemconfig(lb.size() - 1, foreground='#666')
                        for pair in sorted(ct_pairs):
                            lb.insert(tk.END,
                                      f"  [{pair[0]:3d}, {pair[1]:3d}]")
            else:
                for pair in sorted(g):
                    lb.insert(tk.END, f"[{pair[0]:3d}, {pair[1]:3d}]")
            btn_row = ttk.Frame(frame)
            btn_row.pack(pady=4)
            # Convert button: toggle between regular ↔ special
            if is_special:
                conv_label = "Convert to group"
                def _do_convert(g=gname, d=display):
                    self._rename_group(g, d, win)   # strip prefix → regular group
            else:
                conv_label = "Convert to special group"
                def _do_convert(g=gname, d=display):
                    self._rename_group(g, _SPECIAL_PREFIX + d, win)
            ttk.Button(btn_row, text=conv_label,
                       command=_do_convert).pack(side=tk.LEFT, padx=4)
            ttk.Button(btn_row, text=f"Delete group '{display}'",
                       command=lambda g=gname: self._delete_group(g, win)).pack(
                side=tk.LEFT, padx=4)

        # Regular groups
        for gname in sorted(self._groups):
            if gname.startswith('__'):
                continue
            _add_group_tab(nb, gname, is_special=False)

        # Special groups — own notebook section
        special_names = sorted(g for g in self._groups if g.startswith(_SPECIAL_PREFIX))
        if special_names:
            special_frame = ttk.Frame(nb)
            nb.add(special_frame, text="Special")
            snb = ttk.Notebook(special_frame)
            snb.pack(fill=tk.BOTH, expand=True)
            for gname in special_names:
                _add_group_tab(snb, gname, is_special=True)
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=4)


    def _rename_group(self, old_name, new_name, win=None):
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return
        if new_name in self._groups:
            messagebox.showwarning("Rename", f"'{new_name}' already exists.")
            return

        # Preserve group identity in the v4 registry: rename the registry entry
        # instead of orphaning it (which can drop assignments on save).
        try:
            gid = self._group_id_for(old_name)
            if gid is not None and gid in self._group_registry:
                self._group_registry[gid]['name'] = new_name
        except Exception:
            pass

        self._groups[new_name] = self._groups.pop(old_name)
        if old_name in self._group_hotkeys:
            if new_name.startswith(_SPECIAL_PREFIX):
                # Special groups don't use hotkeys — drop it on conversion
                self._group_hotkeys.pop(old_name)
                try:
                    gid = self._group_id_for(new_name)
                    if gid is not None and gid in self._group_registry:
                        self._group_registry[gid]['hotkey'] = None
                except Exception:
                    pass
            else:
                self._group_hotkeys[new_name] = self._group_hotkeys.pop(old_name)
        if old_name in self._group_notes:
            self._group_notes[new_name] = self._group_notes.pop(old_name)
        self._rebuild_groups_menu()
        self.refresh_lists()
        if win:
            win.destroy()
            self._manage_groups_dialog()

    def _delete_group(self, name, win=None):
        if not messagebox.askyesno("Delete group",
                                   f"Delete group '{name}'?"):
            return
        self._groups.pop(name, None)
        self._group_hotkeys.pop(name, None)
        self._group_notes.pop(name, None)
        self._rebuild_groups_menu()
        self.refresh_lists()
        if win:
            win.destroy()
            # Reopen if there are remaining groups
            if self._groups:
                self._manage_groups_dialog()

    # ------------------------------------------------------------------
    # Auto-classification
    def _clear_speculated(self):
        """Discard all pending speculated predictions and refresh the list."""
        self._speculated_groups.clear()
        self.refresh_lists()

    # ------------------------------------------------------------------
    # Template editor
    # ------------------------------------------------------------------

    def _template_editor_dialog(self, preselect: str = None):
        """
        Dialog for creating / editing per-group CCG shape templates.

        Rule types: peak, trough, baseline_low (counts in lag range expected
        to be below a fraction of the peak).  Each rule can also be restricted
        to specific conn_types.  A live Gaussian preview updates as rules change.
        """
        win = tk.Toplevel(self.root)
        win.title("CCG Template Editor")
        win.geometry("1100x820")
        win.resizable(True, True)

        def _on_close():
            win.destroy()
            self.root.focus_set()   # return keyboard focus to main window

        win.protocol("WM_DELETE_WINDOW", _on_close)

        # Determine CCG half-window in ms (for preview x-range)
        _x_half = 10.0
        try:
            _conf = (getattr(self.ccg_data, 'conf', None)
                     or getattr(self.ccg_data, '_conf', None))
            if _conf is not None:
                _x_half = float(_conf.duration) * 500.0   # s → half-ms
        except Exception:
            pass

        # Available conn_types for filter column
        _all_ct = ['(all)', 'Excitatory', 'Inhibitory']
        try:
            for k in self.cd.data:
                ct = k.conn_type
                s = ('-'.join(ct) if isinstance(ct, tuple) else str(ct)) if ct else '?'
                if s not in _all_ct:
                    _all_ct.append(s)
        except Exception:
            pass

        # ── Top bar ──────────────────────────────────────────────────────
        top = ttk.Frame(win)
        top.pack(fill=tk.X, padx=8, pady=4)

        ttk.Label(top, text="Group:").pack(side=tk.LEFT)
        group_var = tk.StringVar()
        group_cb  = ttk.Combobox(top, textvariable=group_var, state='readonly',
                                 width=18)
        group_cb.pack(side=tk.LEFT, padx=(4, 8))

        ttk.Label(top, text="Smooth (ms):").pack(side=tk.LEFT)
        smooth_var = tk.DoubleVar(value=self._templates_smooth_ms)
        ttk.Spinbox(top, textvariable=smooth_var, from_=0.0, to=10.0,
                    increment=0.5, width=6).pack(side=tk.LEFT, padx=(2, 8))

        # ── Main pane: left=rules, right=preview ─────────────────────────
        pane = ttk.PanedWindow(win, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # ── Left: rules table ────────────────────────────────────────────
        left = ttk.Frame(pane)
        pane.add(left, weight=1)

        tbl_frame = ttk.LabelFrame(left, text="Rules")
        tbl_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 2))

        _RULE_TYPES = ['peak', 'trough', 'baseline_low', 'not in',
                       'trough-peak', 'peak-trough', 'min_bincount', 'min_bincount_avg']
        _RES_OPTIONS = ['both', 'hi', 'lo']
        col_hdrs   = ("", "Type", "Lag min", "Lag max",
                      "W min", "W max", "Smooth", "Conn type", "Res", "Req", "Min cnt")
        col_widths = (2,   8,     7,        7,
                      7,    7,     5,       9,          6,    4,    7)
        for c, (lbl, w) in enumerate(zip(col_hdrs, col_widths)):
            ttk.Label(tbl_frame, text=lbl,
                      font=('TkDefaultFont', 9, 'bold'),
                      width=w).grid(row=0, column=c, padx=2, pady=2, sticky='w')

        rule_rows: list = []
        _row_counter = [0]   # monotonic — never reused so add_btn never collides

        _preview_after = [None]

        def _schedule_preview(*_):
            if _preview_after[0] is not None:
                try:
                    win.after_cancel(_preview_after[0])
                except Exception:
                    pass
            _preview_after[0] = win.after(120, _update_preview)

        def _add_rule_row(rule_type='peak', lag_min=0.0, lag_max=5.0,
                          w_min='', w_max='', smooth_ms='', conn_type='(all)',
                          resolution='both', required=True, ref_group='',
                          min_count=''):
            _row_counter[0] += 1
            r = _row_counter[0]
            row = {
                'type':       tk.StringVar(value=rule_type),
                'lag_min':    tk.StringVar(value=str(lag_min)),
                'lag_max':    tk.StringVar(value=str(lag_max)),
                'w_min':      tk.StringVar(value='' if w_min == '' else str(w_min)),
                'w_max':      tk.StringVar(value='' if w_max == '' else str(w_max)),
                'smooth_ms':  tk.StringVar(value='' if smooth_ms == '' else str(smooth_ms)),
                'conn_type':  tk.StringVar(value=conn_type),
                'resolution': tk.StringVar(value=resolution),
                'required':   tk.BooleanVar(value=required),
                'ref_group':  tk.StringVar(value=ref_group),
                'min_count':  tk.StringVar(value='' if min_count == '' else str(min_count)),
                '_row':       r,
            }

            def _on_type_change(*_, rr=row):
                rtype = rr['type'].get()
                is_bl       = rtype == 'baseline_low'
                is_notin    = rtype == 'not in'
                is_mincount = rtype in ('min_bincount', 'min_bincount_avg')
                # Lag entries: hidden for 'not in'
                for e in rr.get('_lag_entries', []):
                    e.grid_remove() if is_notin else e.grid()
                # Width entries: hidden for 'not in' and 'min_bincount'; disabled for baseline_low
                for e in rr.get('_width_entries', []):
                    if is_notin or is_mincount:
                        e.grid_remove()
                    else:
                        e.grid()
                        e.config(state='disabled' if is_bl else 'normal')
                # Smooth entry: hidden for 'not in' only
                sm = rr.get('_smooth_entry')
                if sm:
                    sm.grid_remove() if is_notin else sm.grid()
                # Ref group combobox: only for 'not in'
                ref_cb = rr.get('_ref_cb')
                if ref_cb:
                    cur = group_var.get()
                    known = sorted({g for g in list(self._templates.keys())
                                    + list(self._groups.keys() if self._groups else [])
                                    if g != cur})
                    ref_cb['values'] = known
                    ref_cb.grid() if is_notin else ref_cb.grid_remove()
                # Min count entry: only for 'min_bincount' rules
                mc_e = rr.get('_min_count_entry')
                if mc_e:
                    mc_e.grid() if is_mincount else mc_e.grid_remove()
                # Res combobox: hidden for 'not in'
                res_cb = rr.get('_res_cb')
                if res_cb:
                    res_cb.grid_remove() if is_notin else res_cb.grid()
                _schedule_preview()

            def _del(rr=row):
                for w in tbl_frame.grid_slaves():
                    if w.grid_info().get('row') == rr['_row']:
                        w.destroy()
                if rr in rule_rows:
                    rule_rows.remove(rr)
                try:
                    _place_add_btn()
                except Exception:
                    pass
                _schedule_preview()

            del_btn = ttk.Button(tbl_frame, text='×', width=2, command=_del)
            del_btn.grid(row=r, column=0, padx=2)
            row['_del_btn'] = del_btn

            ttk.Combobox(tbl_frame, textvariable=row['type'],
                         values=_RULE_TYPES, state='readonly',
                         width=9).grid(row=r, column=1, padx=2, pady=1)
            row['type'].trace_add('write', _on_type_change)

            lag_entries = []
            _lag_limits = {'lag_min': -_x_half, 'lag_max': _x_half}
            for c, key in enumerate(['lag_min', 'lag_max'], start=2):
                e = ttk.Entry(tbl_frame, textvariable=row[key], width=7)
                e.grid(row=r, column=c, padx=2, pady=1, sticky='ew')
                row[key].trace_add('write', _schedule_preview)
                def _fill_limit(_, rr=row, k=key):
                    rr[k].set(str(_lag_limits[k]))
                    _schedule_preview()
                    return 'break'
                e.bind('<Double-Button-3>', _fill_limit)
                e.bind('<Double-Button-2>', _fill_limit)
                lag_entries.append(e)
            row['_lag_entries'] = lag_entries

            width_entries = []
            for c, key in enumerate(['w_min', 'w_max'], start=4):
                e = ttk.Entry(tbl_frame, textvariable=row[key], width=7)
                e.grid(row=r, column=c, padx=2, pady=1, sticky='ew')
                row[key].trace_add('write', _schedule_preview)
                width_entries.append(e)
            row['_width_entries'] = width_entries

            # Per-rule smooth (ms) — blank means use global default
            sm_e = ttk.Entry(tbl_frame, textvariable=row['smooth_ms'], width=5)
            sm_e.grid(row=r, column=6, padx=2, pady=1, sticky='ew')
            row['smooth_ms'].trace_add('write', _schedule_preview)
            row['_smooth_entry'] = sm_e

            # 'not in' group selector — spans columns 2-5, hidden by default
            ref_cb = ttk.Combobox(tbl_frame, textvariable=row['ref_group'],
                                  values=[], state='readonly', width=20)
            ref_cb.grid(row=r, column=2, columnspan=4, padx=2, pady=1, sticky='ew')
            ref_cb.grid_remove()
            row['_ref_cb'] = ref_cb
            row['ref_group'].trace_add('write', _schedule_preview)

            ttk.Combobox(tbl_frame, textvariable=row['conn_type'],
                         values=_all_ct, state='readonly',
                         width=9).grid(row=r, column=7, padx=2, pady=1)
            row['conn_type'].trace_add('write', _schedule_preview)

            res_cb = ttk.Combobox(tbl_frame, textvariable=row['resolution'],
                                  values=_RES_OPTIONS, state='readonly', width=5)
            res_cb.grid(row=r, column=8, padx=2, pady=1)
            row['resolution'].trace_add('write', _schedule_preview)
            row['_res_cb'] = res_cb

            ttk.Checkbutton(tbl_frame, variable=row['required']).grid(
                row=r, column=9, padx=2, pady=1)
            row['required'].trace_add('write', _schedule_preview)

            # Min count entry (column 10) — shown only for min_bincount rules
            mc_e = ttk.Entry(tbl_frame, textvariable=row['min_count'], width=7)
            mc_e.grid(row=r, column=10, padx=2, pady=1, sticky='ew')
            mc_e.grid_remove()   # hidden by default; _on_type_change shows it
            row['min_count'].trace_add('write', _schedule_preview)
            row['_min_count_entry'] = mc_e
            rule_rows.append(row)
            _on_type_change()   # set initial disabled state

        # Add rule button
        add_btn = ttk.Button(tbl_frame, text='+ Add rule')
        _is_main = [False]

        def _place_add_btn():
            if _is_main[0]:
                add_btn.grid_remove()
                return
            max_r = max((rr['_row'] for rr in rule_rows), default=0)
            add_btn.grid(row=max_r + 1, column=0, columnspan=5,
                         sticky='w', padx=4, pady=4)

        def _add_and_reposition(**kw):
            _add_rule_row(**kw)
            _place_add_btn()
            _schedule_preview()
        add_btn.config(command=_add_and_reposition)
        _place_add_btn()   # show button immediately even with no rules yet

        # ── Extra constraints ────────────────────────────────────────────
        extra = ttk.Frame(left)
        extra.pack(fill=tk.X, pady=2)

        baseline_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(extra, text="Expect low baseline before t=0",
                        variable=baseline_var).pack(side=tk.LEFT, padx=4)
        baseline_var.trace_add('write', _schedule_preview)

        ttk.Label(extra, text="  n_peaks:").pack(side=tk.LEFT)
        npmin_var = tk.StringVar(value='')
        npmax_var = tk.StringVar(value='')
        ttk.Label(extra, text="min").pack(side=tk.LEFT, padx=(4, 0))
        ttk.Entry(extra, textvariable=npmin_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(extra, text="max").pack(side=tk.LEFT, padx=(4, 0))
        ttk.Entry(extra, textvariable=npmax_var, width=4).pack(side=tk.LEFT, padx=2)

        # ── Test / score readout (per-rule breakdown) ─────────────────────
        test_txt = tk.Text(left, height=5, width=52, font=('Courier', 8),
                           state=tk.DISABLED, relief=tk.FLAT,
                           bg='#F0F4FA', fg='#1A237E', wrap=tk.WORD)
        test_txt.pack(fill=tk.X, padx=4, pady=2)
        test_txt.tag_config('pass', foreground='#2E7D32')   # green
        test_txt.tag_config('fail', foreground='#C62828')   # red
        test_txt.tag_config('head', foreground='#1565C0', font=('Courier', 8, 'bold'))

        def _test_set(text, tag=None):
            test_txt.config(state=tk.NORMAL)
            test_txt.delete('1.0', tk.END)
            test_txt.insert(tk.END, text, tag or '')
            test_txt.config(state=tk.DISABLED)

        def _test_append(text, tag=None):
            test_txt.config(state=tk.NORMAL)
            test_txt.insert(tk.END, text, tag or '')
            test_txt.config(state=tk.DISABLED)

        # ── Right: live preview canvas ────────────────────────────────────
        right = ttk.LabelFrame(pane, text="Preview (Gaussian approximation)")
        pane.add(right, weight=2)

        _fig  = Figure(figsize=(4.8, 3.8), dpi=90, facecolor='#F8F8F8')
        _ax   = _fig.add_subplot(111)
        _canvas = FigureCanvasTkAgg(_fig, master=right)
        _canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Preview update ────────────────────────────────────────────────
        _COLORS = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800',
                   '#9C27B0', '#00BCD4', '#795548', '#607D8B']

        def _update_preview(*_):
            _ax.clear()
            x = np.linspace(-_x_half, _x_half, 600)
            composite = np.zeros_like(x)

            _ax.axhline(0, color='#AAAAAA', linewidth=0.8, linestyle='-')
            _ax.axvline(0, color='#CCCCCC', linewidth=0.6)

            has_rules = False
            for i, row in enumerate(rule_rows):
                col = _COLORS[i % len(_COLORS)]
                req = row['required'].get()
                alpha_band = 0.10
                alpha_line = 0.9 if req else 0.45
                ls = '-' if req else '--'
                lbl_suffix = '' if req else ' (opt)'
                rtype = row['type'].get()

                try:
                    lmin = float(row['lag_min'].get())
                    lmax = float(row['lag_max'].get())
                except ValueError:
                    continue

                center = (lmin + lmax) / 2.0

                if rtype == 'not in':
                    # 'not in' is a logic gate — not a visual shape, skip preview
                    continue

                if rtype in ('trough-peak', 'peak-trough'):
                    # Biphasic: draw a trough+peak (or peak+trough) side by side
                    span    = lmax - lmin
                    c_left  = lmin + span * 0.25
                    c_right = lmin + span * 0.75
                    wmax_s  = row['w_max'].get().strip()
                    wmin_s  = row['w_min'].get().strip()
                    w_val   = (float(wmax_s) if wmax_s else
                               (float(wmin_s) * 1.5 if wmin_s else span * 0.35))
                    sigma_b = max(w_val / 2.3548, 1e-3)
                    if rtype == 'trough-peak':
                        pol_l, pol_r = -1, 1
                    else:
                        pol_l, pol_r = 1, -1
                    g = (pol_l * np.exp(-0.5 * ((x - c_left)  / sigma_b) ** 2) +
                         pol_r * np.exp(-0.5 * ((x - c_right) / sigma_b) ** 2))
                    _ax.axvspan(lmin, lmax, alpha=alpha_band, color=col, linewidth=0)
                    ct_lbl  = row['conn_type'].get()
                    res_lbl = row['resolution'].get()
                    ct_str  = f' [{ct_lbl}]' if ct_lbl != '(all)' else ''
                    res_str = f' ({res_lbl})' if res_lbl and res_lbl != 'both' else ''
                    label   = f"R{i+1} {rtype}{lbl_suffix}{ct_str}{res_str}"
                    _ax.plot(x, g, color=col, linewidth=1.8, linestyle=ls, label=label)
                    composite += g
                    has_rules = True
                    continue

                if rtype == 'baseline_low':
                    # Shaded suppressed region — flat at ~0.1
                    _ax.axvspan(lmin, lmax, alpha=0.18, color=col, linewidth=0)
                    _ax.annotate(
                        f'R{i+1} baseline_low{lbl_suffix}',
                        xy=(center, 0.08), fontsize=7, color=col,
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))
                    has_rules = True
                    continue

                wmax_s = row['w_max'].get().strip()
                wmin_s = row['w_min'].get().strip()
                wmax = float(wmax_s) if wmax_s else None
                wmin = float(wmin_s) if wmin_s else None

                # Choose display FWHM
                if wmax is not None:
                    fwhm = wmax
                elif wmin is not None:
                    fwhm = wmin * 1.5
                else:
                    fwhm = max(0.3, (lmax - lmin) * 0.75)
                sigma = max(fwhm / 2.3548, 1e-3)

                polarity = 1 if rtype == 'peak' else -1
                g = polarity * np.exp(-0.5 * ((x - center) / sigma) ** 2)
                composite += g

                # Lag range band
                _ax.axvspan(lmin, lmax, alpha=alpha_band, color=col, linewidth=0)
                # Width constraint markers (dashed verticals at ±fwhm/2)
                if wmax is not None:
                    for side in (-1, 1):
                        _ax.axvline(center + side * wmax / 2,
                                    color=col, linewidth=0.8, linestyle=':',
                                    alpha=0.7)
                # Gaussian curve
                ct_lbl  = row['conn_type'].get()
                res_lbl = row.get('resolution') and row['resolution'].get()
                ct_str  = f' [{ct_lbl}]' if ct_lbl != '(all)' else ''
                res_str = f' ({res_lbl})' if res_lbl and res_lbl != 'both' else ''
                label = f"R{i+1} {rtype}{lbl_suffix}{ct_str}{res_str}"
                _ax.plot(x, g, color=col, linewidth=1.8, linestyle=ls, label=label)
                # Peak marker
                _ax.plot(center, polarity,
                         marker='v' if polarity == -1 else '^',
                         color=col, markersize=7,
                         markeredgecolor='#333', markeredgewidth=0.6)
                has_rules = True

            # 'not in' exclusion rules — list as text below title
            excl = [rr['ref_group'].get() for rr in rule_rows
                    if rr['type'].get() == 'not in' and rr['ref_group'].get()]
            if excl:
                _ax.set_title(
                    (_ax.get_title() or '') + f"\n✗ excl: {', '.join(excl)}",
                    fontsize=9, fontweight='bold')

            # Composite envelope
            if sum(1 for rr in rule_rows
                   if rr['type'].get() not in ('baseline_low', 'not in')) > 1:
                _ax.plot(x, composite, color='#222', linewidth=2.2,
                         alpha=0.5, linestyle=':', label='sum')

            # Baseline-before-zero constraint
            if baseline_var.get():
                _ax.axvspan(-_x_half, 0, alpha=0.06, color='#F44336')
                _ax.text(-_x_half * 0.92, 1.05, '← low baseline',
                         color='#F44336', fontsize=7)

            # Axes styling
            _ax.set_xlim(-_x_half, _x_half)
            cur_ylim = _ax.get_ylim()
            _ax.set_ylim(min(-0.55, cur_ylim[0] - 0.05),
                         max(1.25, cur_ylim[1] + 0.05))
            _ax.set_xlabel("Lag (ms)", fontsize=8)
            _ax.set_ylabel("Normalized amplitude", fontsize=8)
            _ax.tick_params(labelsize=7)
            gname = group_var.get()
            _ax.set_title(f"Template: {gname}" if gname else "Template preview",
                          fontsize=9, fontweight='bold')
            if has_rules:
                _ax.legend(fontsize=7, loc='upper right',
                           framealpha=0.8, handlelength=1.6)
            _fig.tight_layout(pad=0.8)
            _canvas.draw_idle()

        # ── Helpers ──────────────────────────────────────────────────────
        def _current_template():
            def _f(s):
                """Parse a string to float; return None if blank or unparseable."""
                v = str(s).strip()
                if not v:
                    return None
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None

            _POL = {'peak': 1, 'trough': -1, 'baseline_low': 0, 'not in': 2,
                    'trough-peak': 3, 'peak-trough': 4,
                    'min_bincount': 5, 'min_bincount_avg': 6}

            rules = []
            for row in rule_rows:
                rtype = row['type'].get()
                ct_s  = row['conn_type'].get()
                res_s = row['resolution'].get()
                req   = bool(row['required'].get())
                pol   = _POL.get(rtype, 1)
                if rtype == 'not in':
                    rg = row['ref_group'].get().strip()
                    if rg:
                        rules.append(PeakRule(lag_min=0.0, lag_max=0.0,
                                              polarity=2, required=req,
                                              resolution=res_s, conn_type=ct_s,
                                              ref_group=rg))
                    continue
                # lag_min / lag_max: use 0.0 as fallback so the rule is never silently dropped
                lmin = _f(row['lag_min'].get()) if _f(row['lag_min'].get()) is not None else 0.0
                lmax = _f(row['lag_max'].get()) if _f(row['lag_max'].get()) is not None else 0.0
                rules.append(PeakRule(
                    lag_min=lmin, lag_max=lmax,
                    width_min=_f(row['w_min'].get()),
                    width_max=_f(row['w_max'].get()),
                    polarity=pol, required=req,
                    resolution=res_s, conn_type=ct_s,
                    smooth_ms=_f(row['smooth_ms'].get()),
                    min_count=_f(row['min_count'].get()),
                ))
            npmin_s = npmin_var.get().strip()
            npmax_s = npmax_var.get().strip()
            return GroupTemplate(
                name=group_var.get(),
                peak_rules=rules,
                n_peaks_min=int(npmin_s) if npmin_s else None,
                n_peaks_max=int(npmax_s) if npmax_s else None,
                baseline_low_before_zero=baseline_var.get(),
            )

        def _load_group(name):
            _is_main[0] = (name == 'Main')
            for rr in list(rule_rows):
                for w in tbl_frame.grid_slaves():
                    if w.grid_info().get('row') == rr['_row']:
                        w.destroy()
            rule_rows.clear()
            tmpl = self._templates.get(name)
            if tmpl is not None:
                for pr in tmpl.peak_rules:
                    if pr.polarity == 2:
                        _add_rule_row(
                            rule_type='not in',
                            conn_type=getattr(pr, 'conn_type', '(all)'),
                            resolution=getattr(pr, 'resolution', 'both'),
                            required=pr.required,
                            ref_group=getattr(pr, 'ref_group', ''),
                        )
                        continue
                    rtype = {0: 'baseline_low', 1: 'peak', -1: 'trough',
                             3: 'trough-peak', 4: 'peak-trough',
                             5: 'min_bincount', 6: 'min_bincount_avg'}.get(pr.polarity, 'trough')
                    _add_rule_row(
                        rule_type=rtype,
                        lag_min=pr.lag_min, lag_max=pr.lag_max,
                        w_min='' if pr.width_min is None else pr.width_min,
                        w_max='' if pr.width_max is None else pr.width_max,
                        smooth_ms='' if getattr(pr, 'smooth_ms', None) is None
                                  else pr.smooth_ms,
                        conn_type=getattr(pr, 'conn_type', '(all)'),
                        resolution=getattr(pr, 'resolution', 'both'),
                        required=pr.required,
                        min_count='' if getattr(pr, 'min_count', None) is None
                                  else pr.min_count,
                    )
                baseline_var.set(tmpl.baseline_low_before_zero)
                npmin_var.set('' if tmpl.n_peaks_min is None else str(tmpl.n_peaks_min))
                npmax_var.set('' if tmpl.n_peaks_max is None else str(tmpl.n_peaks_max))
            else:
                baseline_var.set(False)
                npmin_var.set('')
                npmax_var.set('')
            _place_add_btn()
            if _is_main[0]:
                for rr in rule_rows:
                    if rr.get('_del_btn'):
                        rr['_del_btn'].config(state='disabled')
            _update_preview()

        def _on_group_change(*_):
            if group_var.get() == _SEP:
                return
            _load_group(group_var.get())
            _update_preview()

        group_var.trace_add('write', lambda *_: _update_preview())

        def _save_current():
            name = group_var.get()
            if not name or name == 'Main':
                return
            self._templates[name] = _current_template()
            self._templates_smooth_ms = smooth_var.get()

        def _test_on_current():
            _save_current()
            if not self._templates:
                _test_set("No templates defined.")
                return
            if self.ccg_data is None:
                _test_set("No CCG data loaded.")
                return
            conf = getattr(self.ccg_data, 'conf', None) or getattr(
                self.ccg_data, '_conf', None)
            if conf is None:
                _test_set("Cannot read conf.")
                return
            nd = self.key.nd()
            cd = (self.cd._ccg_highres.get(nd)
                  if (hasattr(self.cd, '_ccg_highres')
                      and self.cd._ccg_highres
                      and self.cd._ccg_highres.get(nd) is not None)
                  else self.ccg_data)
            try:
                ct = self.key.conn_type
                ct_str = ('-'.join(ct) if isinstance(ct, (list, tuple))
                          else str(ct)) if ct else '(all)'
            except Exception:
                ct_str = '(all)'
            clf = CCGTemplateClassifier(cd, conf, smooth_ms=smooth_var.get(),
                                        conn_type_str=ct_str)
            clf.load_templates(self._templates)
            inds = self.all_inds[self.current_pair_idx]
            ref, tgt = int(inds[0]), int(inds[1])
            detail = clf.score_pair_detail(ref, tgt)

            test_txt.config(state=tk.NORMAL)
            test_txt.delete('1.0', tk.END)
            for gname, gdata in sorted(detail.items(),
                                       key=lambda x: -x[1]['score']):
                score = gdata['score']
                test_txt.insert(tk.END, f"{gname}: {score:.2f}\n", ('head',))
                for r in gdata['rules']:
                    sym  = '✓' if r['matched'] else '✗'
                    rtag = 'pass' if r['matched'] else 'fail'
                    req  = '(req)' if r['required'] else '(opt)'
                    test_txt.insert(
                        tk.END,
                        f"  {sym} {req} {r['label']}  sim={r['sim']:.2f}\n",
                        rtag,
                    )
            test_txt.config(state=tk.DISABLED)

            # Overlay the smoothed CCG on the preview canvas (normalized 0–1)
            try:
                _, ccg_smooth = clf.smooth_ccg(ref, tgt)
                nb  = len(ccg_smooth)
                bs  = conf.duration / (nb - 1) if nb > 1 else getattr(conf, 'bin_size', 0.001)
                cb  = (nb - 1) // 2
                lag_ms = (np.arange(nb) - cb) * bs * 1e3
                vmin = ccg_smooth.min()
                vmax = ccg_smooth.max()
                ccg_norm = (ccg_smooth - vmin) / (vmax - vmin + 1e-12)
                # Scale to y-range of current preview (peaks at ≈1 → shift to top quarter)
                ccg_scaled = ccg_norm * 1.0  # maps 0→0, 1→1 in normalized units
                _update_preview()   # redraw Gaussians first
                _ccg_line, = _ax.plot(lag_ms, ccg_scaled, color='#1A237E',
                                      linewidth=1.5, alpha=0.75, zorder=5,
                                      label=f'CCG smooth ({ref}→{tgt})')
                _ax.legend(fontsize=7, loc='upper right', framealpha=0.8, handlelength=1.6)
                _canvas.draw_idle()

                def _clear_ccg_overlay():
                    try:
                        _ccg_line.remove()
                        _canvas.draw_idle()
                    except Exception:
                        pass
                win.after(3000, _clear_ccg_overlay)
            except Exception:
                pass

        def _save_to_file():
            _save_current()
            if not self._templates:
                messagebox.showwarning("Save templates",
                                       "No templates to save.", parent=win)
                return
            path = self._templates_path
            try:
                CCGTemplateClassifier.save_templates_to_file(
                    path, self._templates,
                    smooth_ms=smooth_var.get(),
                    classify_with=sorted(self._active_templates) if self._active_templates else None)
            except Exception as exc:
                messagebox.showerror("Save templates",
                                     f"Save failed:\n{exc}", parent=win)
                return
            messagebox.showinfo("Save templates", f"Saved to:\n{path}", parent=win)
            _rebuild_toggle_bar()

        def _load_from_file():
            path = filedialog.askopenfilename(
                parent=win, title="Load templates",
                initialdir=self._clf_dir,
                filetypes=[("JSON", "*.json"), ("All", "*")])
            if not path:
                return
            loaded = CCGTemplateClassifier.load_templates_from_file(path)
            if not loaded:
                messagebox.showwarning("Load templates",
                                       "No templates found in file.", parent=win)
                return
            self._templates.update(loaded)
            meta = CCGTemplateClassifier.load_file_metadata(path)
            if meta.get('smooth_ms') is not None:
                smooth_var.set(meta['smooth_ms'])
                self._templates_smooth_ms = meta['smooth_ms']
            if meta.get('classify_with') is not None:
                self._active_templates = set(meta['classify_with'])
                _rebuild_toggle_bar()
            _refresh_groups()
            if group_var.get() in loaded:
                _load_group(group_var.get())
            messagebox.showinfo("Load templates",
                                f"Loaded {len(loaded)} template(s).", parent=win)

        _SEP = '─────────────'

        def _refresh_groups():
            all_tmpl = list(self._templates.keys())
            main_names = ['Main'] if 'Main' in all_tmpl else []
            secondary_tmpl = [n for n in all_tmpl if n != 'Main']
            other_names = sorted(n for n in (self._groups or {})
                             if n not in all_tmpl and not n.startswith('__'))
            names = main_names[:]
            if main_names and (secondary_tmpl or other_names):
                names.append(_SEP)
            names.extend(secondary_tmpl)
            if secondary_tmpl and other_names:
                names.append(_SEP)
            elif not secondary_tmpl and other_names and main_names:
                pass  # separator already added after Main
            names.extend(other_names)
            group_cb['values'] = names
            cur = group_var.get()
            if cur not in names and names:
                first = next((n for n in names if n != _SEP), '')
                group_var.set(first)
                if first:
                    _load_group(first)

        group_cb.bind('<<ComboboxSelected>>', _on_group_change)

        # ── Active-for-classify toggle bar ───────────────────────────────
        # Toggle buttons (one per template) control which templates are used
        # by Auto-classify.  State lives in self._active_templates (empty = all).
        act_frame = ttk.LabelFrame(win, text="Classify with:")
        act_frame.pack(fill=tk.X, padx=8, pady=(0, 2))

        _toggle_btns: dict = {}   # name → tk.Button

        def _rebuild_toggle_bar():
            for w in act_frame.winfo_children():
                w.destroy()
            _toggle_btns.clear()
            for name in sorted(self._templates):
                if name == 'Main':
                    continue
                is_active = (not self._active_templates or name in self._active_templates)

                btn = tk.Button(
                    act_frame,
                    text=f"{name}*" if is_active else name,
                    relief='sunken' if is_active else 'raised'
                )

                def _toggle(n=name):
                    print("clicked:", n)
                    print("before:", self._active_templates)

                    if not self._active_templates:
                        self._active_templates = set(self._templates) - {n}
                    elif n in self._active_templates:
                        self._active_templates.discard(n)
                        if not self._active_templates:
                            self._active_templates = set()
                    else:
                        self._active_templates.add(n)
                    _rebuild_toggle_bar()

                btn.config(command=_toggle)
                btn.pack(side=tk.LEFT)
                _toggle_btns[name] = btn
        _rebuild_toggle_bar()

        # ── Bottom buttons ────────────────────────────────────────────────
        bot = ttk.Frame(win)
        bot.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(bot, text="Test on current pair",
                   command=_test_on_current).pack(side=tk.LEFT, padx=4)
        ttk.Button(bot, text="Save to file…",
                   command=_save_to_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(bot, text="Load from file…",
                   command=_load_from_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(bot, text="Close", command=_on_close).pack(side=tk.RIGHT, padx=4)

        _refresh_groups()
        if preselect and preselect in group_cb['values']:
            group_var.set(preselect)
            _load_group(preselect)
        _rebuild_toggle_bar()
        _place_add_btn()
        _update_preview()

    # ------------------------------------------------------------------

    def _auto_classify_dialog(self):
        """Settings dialog for template-based auto-classification."""
        if self.ccg_data is None:
            messagebox.showinfo("Auto-classify", "No CCG data loaded.",
                                parent=self.root)
            return
        if not self._templates:
            if messagebox.askyesno(
                    "Auto-classify",
                    "No templates defined yet.\n\nOpen the Template Editor first "
                    "(Classify > Edit templates…) to define peak rules for each group."
                    "\n\nOpen Template Editor now?",
                    parent=self.root):
                self._template_editor_dialog()
            return

        win = tk.Toplevel(self.root)
        win.title("Auto-classify pairs")
        win.resizable(False, False)
        win.grab_set()
        pad = {'padx': 8, 'pady': 4}

        # Scope
        scope_var = tk.StringVar(value='current')
        ttk.Label(win, text="Scope:").grid(row=0, column=0, sticky='w', **pad)
        ttk.Radiobutton(win, text="Current conn-type only",
                        variable=scope_var, value='current').grid(
            row=0, column=1, sticky='w', **pad)
        ttk.Radiobutton(win, text="All available conn-types",
                        variable=scope_var, value='all').grid(
            row=1, column=1, sticky='w', **pad)

        # Pairs to classify
        target_var = tk.StringVar(value='unlabeled')
        ttk.Label(win, text="Classify:").grid(row=2, column=0, sticky='w', **pad)
        ttk.Radiobutton(win, text="Unlabeled pairs only",
                        variable=target_var, value='unlabeled').grid(
            row=2, column=1, sticky='w', **pad)
        ttk.Radiobutton(win, text="All pairs (read-only preview)",
                        variable=target_var, value='all').grid(
            row=3, column=1, sticky='w', **pad)

        # Smoothing — mirrors Template Editor setting (read-only info)
        active = sorted(self._active_templates) if self._active_templates else sorted(self._templates)
        ttk.Label(win,
                  text=f"Smooth: {self._templates_smooth_ms:.1f} ms  |  "
                       f"Templates: {', '.join(active) or '(none)'}",
                  foreground='#336699').grid(
            row=4, column=0, columnspan=3, sticky='w', **pad)
        ttk.Label(win,
                  text="(Smoothing and active templates are set in Classify > Edit templates…)",
                  foreground='gray').grid(row=5, column=0, columnspan=3, sticky='w', **pad)

        def _run():
            win.destroy()
            self._run_auto_classify(
                scope=scope_var.get(),
                target=target_var.get(),
                smooth_ms=self._templates_smooth_ms,
            )

        btn_frame = ttk.Frame(win)
        btn_frame.grid(row=6, column=0, columnspan=3, pady=8)
        ttk.Button(btn_frame, text="Run", command=_run).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(side=tk.LEFT)

    def _visualize_cluster_embedding(self, scope='current', smooth_sigma=1.0):
        """
        Fit CCGClusterClassifier and show a matplotlib PCA scatter of the
        training set, coloured by group label.  Opens a non-blocking figure
        window — does not modify any state.
        """
        if self.ccg_data is None:
            return

        keys_to_run = ([self.key] if scope == 'current'
                       else self._available_type_keys(self.key.nd()))

        for tk_ in keys_to_run:
            nd_key = tk_.nd()
            cd = (self.cd._ccg_highres.get(nd_key)
                  if (hasattr(self.cd, '_ccg_highres')
                      and self.cd._ccg_highres
                      and self.cd._ccg_highres.get(nd_key) is not None)
                  else self.cd._ccg.get(nd_key))
            if cd is None:
                continue
            conf = getattr(cd, 'conf', None) or getattr(cd, '_conf', None)
            if conf is None:
                conf = getattr(self.cd, 'conf', None)
            if conf is None:
                continue

            ptr = self.cd.data.get(tk_)
            if ptr is None or ptr.inds is None:
                continue
            local_pairs = set(map(tuple, ptr.inds2))

            labeled_pairs: dict = {}
            for gname, sess_dict in self._groups.items():
                if gname.startswith('__'):
                    continue
                all_in_group: set = set()
                if isinstance(sess_dict, set):
                    all_in_group = sess_dict
                elif isinstance(sess_dict, dict):
                    for sp in sess_dict.values():
                        all_in_group.update(sp)
                resolved = all_in_group & local_pairs
                if resolved:
                    labeled_pairs[gname] = resolved
            deleted_local = self.deleted_inds & local_pairs

            if not labeled_pairs and not deleted_local:
                continue

            all_labeled: set = set()
            for s in labeled_pairs.values():
                all_labeled.update(s)
            all_labeled.update(deleted_local)
            unlabeled = [p for p in sorted(local_pairs) if p not in all_labeled]

            ct_str = (('-'.join(tk_.conn_type) if isinstance(tk_.conn_type, tuple)
                       else str(tk_.conn_type)) if tk_.conn_type else 'unknown')

            clf = CCGClusterClassifier(cd, conf, smooth_sigma=smooth_sigma)
            counts = clf.fit(labeled_pairs, deleted_local,
                             all_pairs=sorted(local_pairs))
            if not counts:
                continue

            sess = self._current_session_str()
            fig = clf.plot_embedding(
                labeled_pairs=labeled_pairs,
                deleted_pairs=deleted_local,
                unlabeled_pairs=unlabeled,
                title=f"{sess}  |  {ct_str}  |  smooth={smooth_sigma:.1f}",
            )
            fig.canvas.manager.set_window_title(f"Cluster embedding — {ct_str}")
            plt.show(block=False)

    def _run_auto_classify(self, scope, target, smooth_ms=2.0):
        """Run template classifier and open preview dialog."""
        ts   = datetime.datetime.now().strftime('%y%m%d-%H-%M')
        sess = self._current_session_str()

        keys_to_run = ([self.key] if scope == 'current'
                       else self._available_type_keys(self.key.nd()))

        all_results = []   # list of (key, list[ClassifyResult])

        for tk_ in keys_to_run:
            nd_key = tk_.nd()
            # Prefer high-res CCGData for peak shape; fall back to low-res
            cd = (self.cd._ccg_highres.get(nd_key)
                  if (hasattr(self.cd, '_ccg_highres')
                      and self.cd._ccg_highres
                      and self.cd._ccg_highres.get(nd_key) is not None)
                  else self.cd._ccg.get(nd_key))
            if cd is None:
                continue
            conf = getattr(cd, 'conf', None) or getattr(cd, '_conf', None)
            if conf is None:
                conf = getattr(self.cd, 'conf', None)
            if conf is None:
                continue

            ct_str = (('-'.join(tk_.conn_type) if isinstance(tk_.conn_type, tuple)
                       else str(tk_.conn_type)) if tk_.conn_type else 'unknown')

            ptr = self.cd.data.get(tk_)
            if ptr is None or ptr.inds is None:
                continue
            local_pairs = set(map(tuple, ptr.inds2))

            # Determine pairs to classify
            if target == 'unlabeled':
                all_labeled_local: set = set()
                for gname, sess_dict in self._groups.items():
                    if gname.startswith('__'):
                        continue
                    grp: set = (sess_dict if isinstance(sess_dict, set)
                                else set().union(*sess_dict.values())
                                if sess_dict else set())
                    all_labeled_local.update(grp & local_pairs)
                all_labeled_local.update(self.deleted_inds & local_pairs)
                pairs_to_clf = [p for p in sorted(local_pairs)
                                if p not in all_labeled_local]
            else:
                pairs_to_clf = sorted(local_pairs)

            if not pairs_to_clf:
                continue

            # Build classifier — same parameters as "Test on current pair":
            # uses stored smooth_ms and conn_type_str so rule filtering matches.
            clf = CCGTemplateClassifier(cd, conf, smooth_ms=smooth_ms,
                                        conn_type_str=ct_str)
            # Only load templates that are toggled on (empty set = all)
            active_tmpls = ({k: v for k, v in self._templates.items()
                             if k in self._active_templates}
                            if self._active_templates else self._templates)
            clf.load_templates(active_tmpls)

            # Score each pair using score_pair_detail (same as template panel).
            # Assign to every group whose score >= 0.5; action='review' otherwise.
            _THRESH = 0.5
            results = []
            for pair in pairs_to_clf:
                ref, tgt = int(pair[0]), int(pair[1])
                detail = clf.score_pair_detail(ref, tgt)
                matched = [g for g, d in detail.items() if d['score'] > _THRESH]
                confidences = {g: d['score'] for g, d in detail.items()}
                if matched:
                    action = 'assign'
                    groups = sorted(matched, key=lambda g: -confidences[g])
                else:
                    action = 'review'
                    groups = []
                results.append(ClassifyResult(
                    pair=pair, action=action, groups=groups,
                    confidences=confidences, trash_confidence=0.0))

            # Save speculated JSON
            spec_path = os.path.join(
                self._clf_dir,
                f"{sess}__{ct_str}__speculated__{ts}.json")
            CCGClassifier.save_speculated(spec_path, results, sess, ct_str)

            all_results.append((tk_, results))

        if not all_results:
            messagebox.showinfo("Auto-classify",
                                "No pairs to classify.", parent=self.root)
            return

        # Store in _speculated_groups — does NOT touch _groups or deleted_inds
        for _, results in all_results:
            for r in results:
                self._speculated_groups[r.pair] = r

        self.refresh_lists()

    def _merge_groups_dialog(self):
        """Dialog to merge two or more groups into one."""
        if len(self._groups) < 2:
            messagebox.showinfo("Merge groups",
                                "Need at least 2 groups to merge.")
            return
        win = tk.Toplevel(self.root)
        win.title("Merge Groups")
        win.geometry("340x320")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Select groups to merge:",
                  font=('Arial', 10, 'bold')).pack(pady=(8, 4))

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10)
        check_vars = {}
        for gname in sorted(self._groups):
            if gname.startswith('__'):
                continue
            var = tk.BooleanVar(value=False)
            check_vars[gname] = var
            ttk.Checkbutton(frame, text=gname, variable=var).pack(anchor='w')

        name_frame = ttk.Frame(win)
        name_frame.pack(fill=tk.X, padx=10, pady=(8, 4))
        ttk.Label(name_frame, text="Merged group name:").pack(side=tk.LEFT)
        name_entry = ttk.Entry(name_frame, width=20)
        name_entry.pack(side=tk.LEFT, padx=4)

        def do_merge():
            selected = [g for g, v in check_vars.items() if v.get()]
            if len(selected) < 2:
                messagebox.showwarning("Merge", "Select at least 2 groups.")
                return
            target = name_entry.get().strip() or selected[0]
            if not messagebox.askokcancel(
                    "Merge groups",
                    f"This will merge {len(selected)} groups into '{target}'.\n"
                    "This cannot be undone. Proceed?"):
                return
            merged = {}
            for g in selected:
                g_data = self._groups.get(g, {})
                if isinstance(g_data, set):
                    # Legacy flat format
                    sess = self._current_session_str()
                    merged.setdefault(sess, set()).update(g_data)
                else:
                    for sess, pairs in g_data.items():
                        merged.setdefault(sess, set()).update(pairs)
                if g != target:
                    self._groups.pop(g, None)
                    self._group_hotkeys.pop(g, None)
                    self._group_notes.pop(g, None)
            self._groups[target] = merged
            self._rebuild_groups_menu()
            self.refresh_lists()
            win.destroy()

        ttk.Button(win, text="Merge", command=do_merge).pack(pady=8)

    def _set_group_hotkey(self, group_name, key_str):
        """Assign hotkey: single digit 1–9/0 or single letter a–z."""
        key_str = key_str.strip().lower()
        if not key_str:
            self._group_hotkeys.pop(group_name, None)
            self._rebuild_groups_menu()
            return
        valid_digits = [str(i) for i in range(1, 10)] + ['0']
        if key_str not in valid_digits and not (len(key_str) == 1 and key_str.isalpha()):
            messagebox.showwarning("Hotkey", "Enter a digit 1–9/0 or a single letter a–z.")
            return
        # Remove from any other group that had this key
        for g, k in list(self._group_hotkeys.items()):
            if k == key_str and g != group_name:
                del self._group_hotkeys[g]
        self._group_hotkeys[group_name] = key_str
        self._rebuild_groups_menu()

    def _rebuild_groups_menu(self):
        """Refresh the dynamic part of the Groups menu."""
        if hasattr(self, 'left_container'):
            self.left_container.left_panel._rebuild_groups_menu()
            return
        # Pre-container fallback (during early setup before left_container exists)
        if not hasattr(self, '_groups_menu'):
            return
        try:
            while self._groups_menu.index('end') >= 8:
                self._groups_menu.delete(8)
        except tk.TclError:
            pass
        self._groups_menu.add_separator()
        current_pairs = set(map(tuple, self.all_inds)) if len(self.all_inds) else set()
        special_groups = []
        for gname in sorted(self._groups):
            if gname.startswith(_SPECIAL_PREFIX):
                special_groups.append(gname)
                continue
            if gname.startswith('__'):
                continue
            hk = self._group_hotkeys.get(gname, '')
            label = gname + (f" [{hk}]" if hk else "")
            self._groups_menu.add_command(
                label=label,
                command=lambda g=gname: self._select_group(g))
        if special_groups:
            special_menu = tk.Menu(self._groups_menu, tearoff=0)
            for gname in special_groups:
                display = gname[len(_SPECIAL_PREFIX):]
                n = len(self._group_pairs(gname) & current_pairs)
                special_menu.add_command(
                    label=f"{display} ({n})",
                    command=lambda g=gname: self._select_group(g))
            self._groups_menu.add_cascade(label="Special", menu=special_menu)
        self._refresh_net_group_buttons()
        if (hasattr(self, '_hotkeys_bar') and
                self._panel_vars.get('Group Hotkeys', tk.BooleanVar()).get()):
            self._refresh_hotkeys_bar()

    def _select_group(self, group_name):
        """Navigate to the first pair in the group."""
        pairs = self._group_pairs(group_name)
        if not pairs:
            return
        first = sorted(pairs)[0]
        self.current_pair_idx = self.get_pair_index(first)
        self.update_plot()
        self._draw_network()

    def _group_hotkey_handler(self, key_str: str, advance: bool = True):
        """Toggles the current pair in/out of the group assigned to key_str.

        key_str is a single character: '1'-'9', '0', or 'a'-'z'.
        advance=False (Shift held): tag without moving the cursor.
        """
        current_pair = self._selected_pair_from_lists()
        if current_pair is None:
            if self.current_pair_idx >= len(self.all_inds):
                return
            if getattr(self, '_session_any_mode', False):
                trip = self._pair_at_all_inds_idx(self.current_pair_idx)
                if trip is None:
                    return
                current_pair = trip
            else:
                row = self.all_inds[self.current_pair_idx]
                current_pair = tuple(int(x) for x in row)
        for gname, k in self._group_hotkeys.items():
            if k != key_str:
                continue

            if not advance:
                # Ctrl held: only the current pair, no listbox multi-select
                highlighted = [current_pair]
                # If the user is holding Shift to apply multiple tags, advance
                # once when Shift is released.
                self._shift_tag_pending_advance = True
            else:
                # ── Collect all highlighted pairs from both listboxes ────────
                avail_map = getattr(self, '_avail_list_pairs', None)
                highlighted = []
                for i in self.unselected_list.curselection():
                    if avail_map and i < len(avail_map):
                        entry = avail_map[i]
                        if entry is None or entry[0] == _AVAIL_GROUP_HDR:
                            continue
                        if entry[1] != 'deleted':
                            highlighted.append(entry[0])
                    elif not avail_map:
                        su = sorted(self.unselected_inds)
                        if i < len(su):
                            highlighted.append(su[i])
                sel_map = getattr(self, '_sel_list_pairs', None)
                for i in self.selected_list.curselection():
                    if sel_map and i < len(sel_map):
                        inds = sel_map[i]
                        if inds is not None:
                            highlighted.append(inds)
                    elif not sel_map:
                        ss = sorted(self.selected_inds)
                        if i < len(ss):
                            highlighted.append(ss[i])
                # If nothing is explicitly selected, fall back to current pair.
                if not highlighted:
                    highlighted = [current_pair]

            multi = len(highlighted) > 1
            changed = set()
            self._push_undo()

            for pair in highlighted:
                was_in_group = self._pair_in_group(pair, gname)
                self._toggle_pair_group(pair, gname)
                if getattr(self, '_session_any_mode', False):
                    changed.add(pair)
                    continue
                if not was_in_group:
                    # Gained a tag → move to selected
                    if pair in self.unselected_inds:
                        self.unselected_inds.discard(pair)
                        self.selected_inds.add(pair)
                        changed.add(pair)
                else:
                    # Lost a tag → move back to available if no tags remain
                    if pair in self.selected_inds:
                        has_groups = any(
                            self._pair_in_group(pair, g)
                            for g in self._groups
                            if not g.startswith('__')
                        )
                        if not has_groups:
                            self.selected_inds.discard(pair)
                            self.unselected_inds.add(pair)
                            changed.add(pair)

            self.refresh_lists()
            self._highlight_changed_pairs(changed or {current_pair})
            if advance:
                next_idx = min(self.current_pair_idx + 1, len(self.all_inds) - 1)
                self.current_pair_idx = next_idx
                self._select_pair_in_list(self._pair_at_all_inds_idx(next_idx))
            else:
                # No advance: keep cursor on current pair after list rebuild
                self._select_pair_in_list(current_pair)
            self.update_plot()
            self._draw_network()
            return
        # No group assigned to this hotkey — show temporary warning
        self._show_temp_warning(f"No group assigned to Ctrl+{key_str}")

    def _show_temp_warning(self, msg: str, duration_ms: int = 2000):
        """Show a temporary warning label at the top of the window that auto-disappears."""
        lbl = tk.Label(self.root, text=msg, bg='#FFF3CD', fg='#856404',
                       font=('Arial', 10, 'bold'), padx=8, pady=4)
        lbl.place(relx=0.5, y=4, anchor='n')
        self.root.after(duration_ms, lbl.destroy)

    # ------------------------------------------------------------------
    # Group export / import
    # ------------------------------------------------------------------

    def _export_groups(self):
        """Export all group definitions to a standalone JSON file."""
        if not self._groups:
            messagebox.showinfo("Export groups", "No groups to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export groups",
            defaultextension='.json',
            filetypes=[('JSON files', '*.json')],
            initialfile='groups_export.json',
            initialdir=self._sel_save_dir,
        )
        if not path:
            return
        groups = {}
        for g, sessions_dict in self._groups.items():
            if isinstance(sessions_dict, set):
                groups[g] = {self._current_session_str():
                             [[int(r), int(c)] for r, c in sorted(sessions_dict)]}
            else:
                groups[g] = {sess: [[int(r), int(c)] for r, c in sorted(pairs)]
                             for sess, pairs in sessions_dict.items() if pairs}
        data = {
            'groups': groups,
            'hotkeys': dict(self._group_hotkeys),
            'notes': dict(self._group_notes),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_default)
        print(f"[CCGReviewUI] groups exported → {path}")

    def _import_groups(self):
        """Import group definitions from a JSON file, merging with existing."""
        path = filedialog.askopenfilename(
            title="Import groups",
            filetypes=[('JSON files', '*.json')],
            initialdir=self._sel_save_dir,
        )
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            messagebox.showerror("Import groups", f"Failed to read file:\n{exc}")
            return
        imported_groups = data.get('groups', {})
        for gname, val in imported_groups.items():
            if isinstance(val, list):
                # Flat format → current session
                sess = self._current_session_str()
                for p in val:
                    self._group_add_pair(gname, tuple(int(v) for v in p), sess)
            elif isinstance(val, dict):
                # Per-session format
                for sess, pairs in val.items():
                    for p in pairs:
                        self._group_add_pair(gname, tuple(int(v) for v in p), sess)
            else:
                self._groups.setdefault(gname, {})
        for gname, hk in data.get('hotkeys', {}).items():
            if gname not in self._group_hotkeys:
                self._group_hotkeys[gname] = hk
        for gname, note in data.get('notes', {}).items():
            if gname not in self._group_notes:
                self._group_notes[gname] = note
        self._rebuild_groups_menu()
        self.refresh_lists()
        print(f"[CCGReviewUI] groups imported from {path}")

    # ------------------------------------------------------------------
    # Versioning helpers (Part II.3)
    # ------------------------------------------------------------------

    def _sel_version_path(self, name: str) -> str:
        safe = name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        session_tag = getattr(self.key, 'session', 'sess')
        return os.path.join(self._sel_save_dir, f"{session_tag}__{safe}.json")

    @staticmethod
    def _json_default(obj):
        """JSON encoder that converts numpy integer/float types to Python scalars."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

    def _build_save_dict(self, saved_at: str, name: str = '') -> dict:
        """Build the serializable dict for a session save (v4.0 format).

        Flushes current type's selections to the pointer, then collects all
        type keys + pair_tags (including group membership) + deleted pairs.
        Does NOT write any files.
        """
        if self.ccg_pointer is None:
            raise RuntimeError("Cannot save: CCG data not yet loaded")
        self._enforce_label_selection_integrity_live()
        # Flush current type's selections to pointer
        if getattr(self, '_session_any_mode', False):
            self._flush_any_selections_to_pointers()
        else:
            self.ccg_pointer.manually_selected_inds = (
                np.array(sorted(self.selected_inds), dtype=int)
                if self.selected_inds else None
            )
        # Collect selections for every type key in this session
        type_keys = self._available_type_keys(self.key.nd())
        selections_by_type = {}
        for tk_ in type_keys:
            ptr = self.cd.data.get(tk_)
            if ptr is None:
                continue
            sel = getattr(ptr, 'manually_selected_inds', None)
            selections_by_type[str(tk_)] = (
                [[int(r), int(c)] for r, c in sorted(map(tuple, sel))]
                if sel is not None and len(sel) > 0 else []
            )
        # Serialize pair_tags: include group membership (names) for this session.
        # NOTE: We intentionally store group NAMES (not numeric IDs) in session save
        # files. IDs remain internal to groups_export.json only.
        cur_sess = self._current_session_str()
        pair_tags_ser: dict = {}
        # Collect all pairs that have either tags/notes OR group membership
        all_annotated = set(self._pair_tags.keys())
        for gname, sd in self._groups.items():
            if isinstance(sd, dict):
                all_annotated |= sd.get(cur_sess, set())
        for pair in sorted(all_annotated):
            r, t = int(pair[0]), int(pair[1])
            existing = dict(self._pair_tags.get(pair, {}))
            # Compute group names for this pair in the current session
            group_names: list[str] = []
            for gname, sd in self._groups.items():
                if gname.startswith('__'):
                    continue
                sp = sd.get(cur_sess, set()) if isinstance(sd, dict) else sd
                if pair in sp:
                    group_names.append(str(gname))
            if group_names or existing.get('notes') or existing.get('tags'):
                entry: dict = {}
                if group_names:
                    entry['groups'] = sorted(set(group_names))
                if existing.get('notes'):
                    entry['notes'] = existing['notes']
                if existing.get('tags'):
                    entry['tags'] = existing['tags']
                pair_tags_ser[f"{r},{t}"] = entry
        self._flush_deleted_to_store()
        deleted_by_type = {}
        for tk_ in type_keys:
            ptr = self.cd.data.get(tk_)
            valid = self._all_inds_set_for_ptr(ptr)
            raw = set(self._pair_deleted_store.get(str(tk_), set())) & valid
            deleted_by_type[str(tk_)] = [[int(r), int(c)] for r, c in sorted(raw)]
        # Legacy single-list field = current key only (older readers)
        deleted_ser = deleted_by_type.get(str(self.key), [])
        return {
            'version': '4.0',
            'name': name,
            'saved_at': saved_at,
            'session': getattr(self.key, 'session', 'sess'),
            'selections': selections_by_type,
            'pair_tags': pair_tags_ser,
            'deleted': deleted_ser,
            'deleted_by_type': deleted_by_type,
        }

    @staticmethod
    def _pair_tag_has_labels(entry: dict) -> bool:
        if not isinstance(entry, dict):
            return False
        groups = entry.get('groups', []) or []
        tags = entry.get('tags', []) or []
        notes = str(entry.get('notes', '') or '').strip()
        return bool(groups or tags or notes)

    def _enforce_label_selection_integrity_live(self):
        """If a pair has labels/tags/notes, force it into selected."""
        if getattr(self, '_session_all_mode', False):
            # all-in-one view selections are triples; pair-tags are session-local pairs.
            return
        tagged_pairs = {
            tuple(map(int, p))
            for p, entry in getattr(self, '_pair_tags', {}).items()
            if self._pair_tag_has_labels(entry)
        }
        if not tagged_pairs:
            return
        avail = set(map(tuple, self.all_inds))
        to_select = tagged_pairs & avail
        if not to_select:
            return
        self.selected_inds |= to_select
        self.unselected_inds -= to_select
        self.deleted_inds -= to_select

    def _enforce_label_selection_integrity_file(
            self, selections_by_type: dict, pair_tags: dict, type_keys: list):
        """Normalize loaded file so pair_tags-labeled pairs are selected in some type."""
        if not isinstance(selections_by_type, dict):
            return selections_by_type
        tagged_pairs = set()
        for key_str, entry in (pair_tags or {}).items():
            if not self._pair_tag_has_labels(entry):
                continue
            parts = str(key_str).split(',')
            if len(parts) != 2:
                continue
            try:
                tagged_pairs.add((int(parts[0]), int(parts[1])))
            except Exception:
                continue
        if not tagged_pairs:
            return selections_by_type
        union_selected = set()
        for v in selections_by_type.values():
            for p in (v or []):
                try:
                    union_selected.add((int(p[0]), int(p[1])))
                except Exception:
                    pass
        missing = sorted(tagged_pairs - union_selected)
        if not missing:
            return selections_by_type
        key_by_str = {str(k): k for k in type_keys}
        for pair in missing:
            candidates = []
            for kstr, tk_ in key_by_str.items():
                ptr = self.cd.data.get(tk_)
                valid = self._all_inds_set_for_ptr(ptr)
                if pair in valid:
                    candidates.append(kstr)
            if not candidates:
                # fall back to the most-populated key in the file
                candidates = sorted(
                    selections_by_type.keys(),
                    key=lambda kk: len(selections_by_type.get(kk, []) or []),
                    reverse=True,
                )
            if not candidates:
                continue
            chosen = candidates[0]
            cur = selections_by_type.get(chosen, []) or []
            if pair not in {(int(x[0]), int(x[1])) for x in cur if isinstance(x, (list, tuple)) and len(x) >= 2}:
                cur.append([int(pair[0]), int(pair[1])])
                selections_by_type[chosen] = cur
        return selections_by_type

    def _save_selection_version(self, name: str) -> str:
        """Persist selections + pair annotations to a JSON file.

        Writes to the named version path and copies to .history/ for recovery.
        Uses atomic write to prevent partial saves from corrupting the file.
        """
        saved_at = datetime.datetime.now().isoformat()
        data = self._build_save_dict(saved_at, name)
        path = self._sel_version_path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._atomic_write_json(path, data)
        n_tags = len(data['pair_tags'])
        print(f"[CCGReviewUI] saved → {os.path.basename(path)}  "
              f"({n_tags} pair_tags, {len(data['selections'])} type keys)")
        # Always copy to .history/ so every save is recoverable
        try:
            self._save_to_history(data, '')
        except Exception as exc:
            print(f"[CCGReviewUI] history copy failed: {exc}")
        return path

    def _list_selection_versions(self) -> list:
        """Return list of (name, path, saved_at, is_valid, is_history) tuples.

        Main versions (in data/selections/) are listed first; then .history/
        entries (backups + autosaves) follow, newest first, marked is_history=True.
        """
        session_tag = getattr(self.key, 'session', 'sess')
        prefix = session_tag + '__'
        versions = []

        def _read_entry(fpath, fname, is_hist):
            is_autosaved = fname.endswith('.autosaved.json')
            try:
                with open(fpath, encoding='utf-8') as f:
                    meta = json.load(f)
                raw_name = meta.get('name', fname)
                if is_hist:
                    kind = '[autosaved]' if is_autosaved else '[backup]'
                    display = f"{kind} {meta.get('saved_at', '')[:16]}"
                else:
                    display = raw_name
                return (display, fpath, meta.get('saved_at', ''), True, is_hist)
            except Exception:
                return (fname, fpath, '⚠ corrupted', False, is_hist)

        if os.path.isdir(self._sel_save_dir):
            for fname in sorted(os.listdir(self._sel_save_dir)):
                if not fname.startswith(prefix) or not fname.endswith('.json'):
                    continue
                versions.append(_read_entry(
                    os.path.join(self._sel_save_dir, fname), fname, False))

        hdir = self._history_dir()
        hist_entries = []
        if os.path.isdir(hdir):
            for fname in os.listdir(hdir):
                if not fname.startswith(prefix) or not fname.endswith('.json'):
                    continue
                fpath = os.path.join(hdir, fname)
                hist_entries.append(_read_entry(fpath, fname, True))
        # Sort history newest first
        hist_entries.sort(key=lambda e: e[2], reverse=True)
        versions.extend(hist_entries)
        return versions

    def _load_selection_from_file(self, path: str, restore_groups: bool = True,
                                   _skip_redraw: bool = False):
        """Load selection from a JSON file (v1.0, v3.x, or v4.0).

        If restore_groups is False, only pair selections are loaded — groups,
        hotkeys, and notes are left untouched.  Used by autoload on session
        switch (groups are shared across sessions and loaded from groups_export).
        """
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSON parse error in {os.path.basename(path)}.\n\n"
                f"The file is likely corrupted from a previous save attempt.\n"
                f"Technical detail: {exc}\n\n"
                f"Please delete this file and re-save your selection."
            ) from exc

        version = data.get('version', '1.0')
        if version >= '2.0' and 'selections' in data:
            selections_by_type = data.get('selections', {})
            type_keys = self._available_type_keys(self.key.nd())
            selections_by_type = self._enforce_label_selection_integrity_file(
                selections_by_type, data.get('pair_tags', {}), type_keys)
            for tk_ in type_keys:
                ptr = self.cd.data.get(tk_)
                if ptr is None:
                    continue
                pairs = selections_by_type.get(str(tk_), [])
                if pairs:
                    ptr.manually_selected_inds = np.array(
                        [[int(r), int(c)] for r, c in pairs], dtype=int)
                else:
                    ptr.manually_selected_inds = None
            cur_sel = selections_by_type.get(str(self.key), [])
            selected = set(tuple(int(v) for v in p) for p in cur_sel)
        else:
            # v1.0 backward compat
            file_key = data.get('key', '')
            if file_key == str(self.key):
                selected = set(tuple(int(v) for v in p)
                               for p in data.get('selected', []))
            else:
                selected = set()

        self._push_undo()
        current_available = set(map(tuple, self.all_inds))
        missing = selected - current_available
        if missing and restore_groups:
            action = self._show_missing_pairs_dialog(missing)
            if action == 'cancel':
                return
            elif action == 'partial':
                selected = selected & current_available
            elif action == 'admit_all':
                for pair in missing:
                    self._group_add_pair(_ADMITTED_GROUP, pair)
                current_available = set(map(tuple, self.all_inds))
        elif missing:
            selected = selected & current_available

        self.selected_inds = selected
        # Per–conn-type deleted sets (v4.0+ optional deleted_by_type; else legacy flat list)
        type_keys_ld = self._available_type_keys(self.key.nd())
        tkey_strs = [str(tk) for tk in type_keys_ld]
        dbtype = data.get('deleted_by_type')
        self._pair_deleted_store = {}
        if isinstance(dbtype, dict) and dbtype:
            for k_str, plist in dbtype.items():
                self._pair_deleted_store[k_str] = {
                    tuple(int(v) for v in p) for p in plist}
        else:
            raw_deleted = data.get('deleted', []) or []
            if raw_deleted:
                st = {tuple(int(v) for v in p) for p in raw_deleted}
                for k_str in tkey_strs:
                    self._pair_deleted_store[k_str] = set(st)
        self.deleted_inds = (
            set(self._pair_deleted_store.get(str(self.key), set())) & current_available
        )
        self.unselected_inds = current_available - selected - self.deleted_inds

        if restore_groups and version < '4.0':
            # v3.x: groups stored in the session file itself (legacy)
            self._restore_groups_from_data(data)

        # Load pair_tags for this session — always reset to avoid stale cross-session tags
        self._pair_tags = {}
        raw_tags = data.get('pair_tags', {})
        cur_sess = self._current_session_str()
        for key_str, tdata in raw_tags.items():
            parts = key_str.split(',')
            if len(parts) != 2:
                continue
            pair = (int(parts[0]), int(parts[1]))
            entry = dict(tdata) if isinstance(tdata, dict) else {'notes': str(tdata)}
            self._pair_tags[pair] = entry
            # Reconstruct current-session group membership from pair_tags.groups.
            # Supports both legacy numeric IDs (v4.0 earlier) and current name-based storage.
            if version >= '4.0' and 'groups' in entry:
                for gitem in entry['groups']:
                    gname = None
                    if isinstance(gitem, (str,)):
                        gname = gitem
                    else:
                        try:
                            gid = int(gitem)
                            g = self._group_registry.get(gid, {})
                            gname = g.get('name') if g else None
                        except Exception:
                            gname = None
                    if gname:
                        self._groups.setdefault(str(gname), {}).setdefault(
                            cur_sess, set()).add(pair)
            # v3.x: also handle old 'tags' key as plain string list (keep as-is)
        self._enforce_label_selection_integrity_live()
        self._sync_sel_data()
        if not _skip_redraw:
            self._post_load_refresh()

    def _show_missing_pairs_dialog(self, missing: set) -> str:
        """Dialog when loaded selection has pairs not in current available set.

        Returns 'partial', 'admit_all', or 'cancel'.
        """
        win = tk.Toplevel(self.root)
        win.title("Missing Pairs")
        win.geometry("450x320")
        win.transient(self.root)
        win.grab_set()

        result = {'action': 'cancel'}
        n = len(missing)

        ttk.Label(win, text=f"{n} selected pair(s) are no longer in available pairs:",
                  font=('Arial', 10, 'bold')).pack(pady=(8, 4))
        ttk.Label(win,
                  text="These pairs may have lost significance after CCG/epoch changes.",
                  font=('Arial', 9), foreground='#666').pack(pady=(0, 4))

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        lb = tk.Listbox(frame, font=('Courier', 9))
        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=lb.yview)
        lb.config(yscrollcommand=sb.set)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        for ref, tgt in sorted(missing):
            lb.insert(tk.END, f"  ({ref:3d}, {tgt:3d})")

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=8)

        def partial():
            result['action'] = 'partial'
            win.destroy()

        def admit_all():
            result['action'] = 'admit_all'
            win.destroy()

        def cancel():
            result['action'] = 'cancel'
            win.destroy()

        ttk.Button(btn_frame, text="Keep only available",
                   command=partial).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Admit all missing",
                   command=admit_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Cancel",
                   command=cancel).pack(side=tk.RIGHT, padx=4)

        win.protocol('WM_DELETE_WINDOW', cancel)
        win.wait_window()
        return result['action']

    def _load_selection_dialog(self):
        """Show a dialog listing all saved versions; user picks one to load."""
        versions = self._list_selection_versions()
        if not versions:
            messagebox.showinfo("Load selection",
                                "No saved selections found for this key.")
            return
        win = tk.Toplevel(self.root)
        win.title("Load Selection")
        win.geometry("620x340")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Select a version to load:",
                  font=('Arial', 10, 'bold')).pack(pady=(8, 4))
        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        sb = ttk.Scrollbar(frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        lb = tk.Listbox(frame, yscrollcommand=sb.set, font=('Courier', 9),
                        selectmode=tk.BROWSE)
        lb.pack(fill=tk.BOTH, expand=True)
        sb.config(command=lb.yview)
        for name, path, saved_at, is_valid, is_history in versions:
            pfx = '   ' if is_valid else '⚠  '
            lb.insert(tk.END, f"{pfx}{name:30s}  {saved_at[:19]}")
            if not is_valid:
                lb.itemconfig(lb.size() - 1, foreground='#CC4444')
            elif is_history:
                lb.itemconfig(lb.size() - 1, foreground='#999999')

        def do_load():
            sel = lb.curselection()
            if not sel:
                return
            name, path, saved_at, is_valid, is_history = versions[sel[0]]
            if not is_valid:
                if not messagebox.askyesno(
                        "Corrupted file",
                        f"'{name}' appears to be corrupted.\n"
                        "Delete it and continue?"):
                    return
                try:
                    os.remove(path)
                except OSError as ex:
                    messagebox.showerror("Delete failed", str(ex))
                win.destroy()
                self._load_selection_dialog()   # reopen with updated list
                return
            try:
                self._load_selection_from_file(path)
                win.destroy()
            except Exception as ex:
                messagebox.showerror("Load selection",
                                     f"Failed to load:\n{ex}")

        def do_delete(event=None):
            sel = lb.curselection()
            if not sel:
                # Select item under cursor for right-click
                idx = lb.nearest(event.y) if event else None
                if idx is not None:
                    lb.selection_clear(0, tk.END)
                    lb.selection_set(idx)
                    sel = (idx,)
                else:
                    return
            name, path, saved_at, is_valid, is_history = versions[sel[0]]
            if not messagebox.askyesno(
                    "Delete selection",
                    f"Move '{name}' to deleted folder?",
                    parent=win):
                return
            deleted_dir = os.path.join(self._sel_save_dir, 'deleted')
            os.makedirs(deleted_dir, exist_ok=True)
            try:
                shutil.move(path, os.path.join(deleted_dir, os.path.basename(path)))
            except OSError as ex:
                messagebox.showerror("Delete failed", str(ex), parent=win)
                return
            win.destroy()
            self._load_selection_dialog()  # reopen with updated list

        def _ctx_menu_load(event):
            menu = tk.Menu(win, tearoff=0)
            menu.add_command(label="Delete", command=lambda: do_delete(event))
            menu.tk_popup(event.x_root, event.y_root)

        lb.bind('<Button-2>', _ctx_menu_load)
        lb.bind('<Button-3>', _ctx_menu_load)
        lb.bind('<Double-Button-1>', lambda e: do_load())
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=6)
        ttk.Button(btn_frame, text="Load", command=do_load).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Cancel",
                   command=win.destroy).pack(side=tk.LEFT, padx=6)
        ttk.Label(btn_frame, text="gray = backup/autosave  ⚠ = corrupted",
                  font=('Arial', 8), foreground='#888888').pack(
            side=tk.RIGHT, padx=6)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_all_state(self, selection_name: str | None = None, *, silent: bool = True) -> bool:
        """Single saving pathway for selections + groups + ui_state.

        - If selection_name is provided: writes that selection version.
        - Always attempts to write groups export (if any groups exist).
        - Always writes ui_state.json (panel + display button state, resolution, etc.).
        """
        if selection_name is not None:
            try:
                self._save_selection_version(selection_name)
            except Exception as exc:
                traceback.print_exc()
                if not silent:
                    messagebox.showerror("Save error",
                                         f"Failed to save selection:\n{exc}",
                                         parent=self.root)
                return False
        try:
            if getattr(self, '_groups', None):
                self._save_groups_export()
        except Exception:
            # never block save on groups export
            traceback.print_exc()
        try:
            self._save_ui_state()
        except Exception:
            traceback.print_exc()
        return True

    def _do_save(self, name: str):
        """Core save logic: persist all types' selections + groups."""
        if not self._save_all_state(name, silent=False):
            return

        # Count total selections across all types
        type_keys = self._available_type_keys(self.key.nd())
        total = sum(
            len(self.cd.data[tk_].manually_selected_inds)
            for tk_ in type_keys
            if self.cd.data.get(tk_) is not None
            and getattr(self.cd.data[tk_], 'manually_selected_inds', None) is not None
        )

        # Groups were exported via _save_all_state; keep message for UI feedback.
        groups_msg = f"\nGroups exported ({len(self._groups)} groups)." if self._groups else ""

        messagebox.showinfo(
            "Saved",
            f"Saved {total} pairs across {len(type_keys)} types as '{name}'.{groups_msg}",
            parent=self.root)

    def _quick_save(self):
        """Ctrl+S / Save button: custom dialog with name entry + Latest button."""
        default_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

        win = tk.Toplevel(self.root)
        win.title("Save selection")
        win.geometry("360x130")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Version name:").pack(pady=(10, 2))
        name_var = tk.StringVar(value=default_name)
        entry = ttk.Entry(win, textvariable=name_var, width=32)
        entry.pack(padx=10)
        entry.select_range(0, tk.END)
        entry.focus_set()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=10)

        def _save_named():
            name = name_var.get().strip() or default_name
            win.destroy()
            self._do_save(name)

        def _save_latest():
            win.destroy()
            self._do_save('latest')

        ttk.Button(btn_frame, text="Save", command=_save_named).pack(
            side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Save as Latest", command=_save_latest).pack(
            side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(
            side=tk.LEFT, padx=6)

        entry.bind('<Return>', lambda e: _save_named())

    def save_gifs(self):
        """Generate per-pair animated GIFs cycling over segments.

        Kept separate from save_selections so it can be called explicitly
        after all PNGs have been rendered at a consistent figure size.
        Each frame is resized to the shape of the first frame so imageio
        can stack them into a GIF.
        """
        selected_array = np.array(sorted(self.selected_inds))
        if self.n_segments <= 1 or len(selected_array) == 0:
            print("Nothing to GIF (need >1 segment and ≥1 selected pair).")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_folder = os.path.join(self.tmp_dir, f"review_{timestamp}")
        os.makedirs(gif_folder, exist_ok=True)
        print(f"Generating GIFs for {len(selected_array)} selected pairs...")
        for inds in selected_array:
            raw_frames = []
            for seg in range(self.n_segments):
                png_path = self._png_path(inds, seg)
                if not os.path.exists(png_path):
                    png_path = self._render_png(inds, seg)
                raw_frames.append(mpimg.imread(png_path))
            # Normalise to uint8 and resize to first frame's shape
            h, w = raw_frames[0].shape[:2]
            frames_u8 = []
            for f in raw_frames:
                arr = (f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
                if arr.shape[:2] != (h, w):
                    arr = np.array(
                        Image.fromarray(arr).resize((w, h), Image.LANCZOS))
                frames_u8.append(arr)
            gif_path = os.path.join(gif_folder,
                                    f"pair_{inds[0]}_{inds[1]}.gif")
            imageio.mimsave(gif_path, frames_u8, duration=0.8)
        print(f"GIFs saved to: {gif_folder}")
        return gif_folder

    def _start_heartbeat(self):
        """Periodic no-op to keep the Tk event loop alive in Jupyter."""
        def _beat():
            if self._closing:
                return
            try:
                if self.root.winfo_exists():
                    self._heartbeat_id = self.root.after(2000, _beat)
            except tk.TclError:
                pass
        self._heartbeat_id = self.root.after(2000, _beat)

    def _on_close(self):
        """Close the app.

        - Always prompts to save on exit (no change-detection).
        - Unsaved custom CCG segments prompt for permission to save.
        - Bookmarks are session-only and are cleared.
        """
        self._closing = True
        if self._heartbeat_id is not None:
            try:
                self.root.after_cancel(self._heartbeat_id)
            except tk.TclError:
                pass
            self._heartbeat_id = None
        if self._jitter_poll_id is not None:
            try:
                self.root.after_cancel(self._jitter_poll_id)
            except tk.TclError:
                pass
            self._jitter_poll_id = None
        if self._jitter_proc is not None and self._jitter_proc.is_alive():
            self._jitter_proc.terminate()
        self._jitter_pending.clear()
        # Custom segments: ask permission to save if any are unsaved
        if self._custom_ccg_has_unsaved():
            r = messagebox.askyesnocancel(
                "Unsaved custom CCGs",
                "You have one or more custom segments that are not saved to a .npz file.\n\n"
                "Save them before quitting?\n"
                "  Yes — open the save dialog\n"
                "  No — quit without saving them\n"
                "  Cancel — don't quit",
                default=messagebox.YES,
            )
            if r is None:
                self._closing = False
                self._start_heartbeat()
                return
            if r:
                self._ts_save_custom_ccg()
                if self._custom_ccg_has_unsaved():
                    messagebox.showwarning(
                        "Custom CCGs not saved",
                        "Some custom segments are still unsaved. Quit was cancelled.",
                    )
                    self._closing = False
                    self._start_heartbeat()
                    return

        # Always ask to save on exit (Save / Don't Save / Cancel)
        r = messagebox.askyesnocancel(
            "Quit",
            "Save before quitting?\n\n"
            "  Yes — save selections/groups + UI state\n"
            "  No — quit without saving\n"
            "  Cancel — don't quit",
            default=messagebox.YES,
        )
        if r is None:
            self._closing = False
            self._start_heartbeat()
            return
        if r:
            # Unified save pathway
            self._autosave_current()
        self._bookmarked_pairs.clear()
        self.root.destroy()

    def run(self):
        if self._owns_mainloop:
            self.root.mainloop()
        else:
            # Another Tk root owns the mainloop; just wait for this window
            self.root.wait_window(self.root)

    # ------------------------------------------------------------------ #
    #  Custom CCG cache — save / load                                       #
    # ------------------------------------------------------------------ #

    def _ccg_cache_filename_for_key(self, seg_name: str, key=None) -> str:
        """Stable cache filename per (session, segment name); recomputes overwrite the same file."""
        key = key or self.key
        session = str(key.session)
        safe = re.sub(r'[^A-Za-z0-9_\-]', '_', str(seg_name).replace(' ', '_'))
        return f"{session}__{safe}.npz"

    def _purge_timestamped_custom_ccg_npz(self, session: str, seg_name: str):
        """Remove legacy ``session__name__timestamp.npz`` files for this logical segment name."""
        safe = re.sub(r'[^A-Za-z0-9_\-]', '_', str(seg_name).replace(' ', '_'))
        patt = os.path.join(self._ccg_cache_dir, f"{session}__{safe}__*.npz")
        for p in _glob.glob(patt):
            try:
                os.remove(p)
            except OSError:
                pass

    def _upsert_custom_segment_by_name(self, lst: list, result: dict) -> tuple[int, bool]:
        """Replace an existing in-memory custom segment with the same name, else append.

        Returns (index_in_lst, did_append).
        """
        nm = str(result.get('name', ''))
        for i, existing in enumerate(lst):
            if str(existing.get('name', '')) == nm:
                lst[i] = result
                return i, False
        lst.append(result)
        return len(lst) - 1, True

    def _ccg_cache_filename(self, seg_name: str) -> str:
        return self._ccg_cache_filename_for_key(seg_name, key=self.key)

    def _ccg_cache_prefix_for_key(self, key=None) -> str:
        """Prefix that all cache files for a session share."""
        key = key or self.key
        return f"{str(key.session)}__"

    def _ccg_cache_prefix(self) -> str:
        return self._ccg_cache_prefix_for_key(key=self.key)

    def _current_filter_state(self) -> dict:
        toggles = getattr(self, '_ts_legend_toggles', {})
        return {
            'theme': getattr(self, '_ts_current_theme', 'segments'),
            'labels': {str(lbl): bool(v.get()) for lbl, v in toggles.items()},
        }

    def _build_custom_spec(self, *, for_all: bool, for_session: str | None = None):
        raw_t0 = self._ts_start_var.get().strip().lower() if hasattr(self, '_ts_start_var') else "00:00:00"
        raw_t1 = self._ts_end_var.get().strip().lower() if hasattr(self, '_ts_end_var') else "00:00:00"
        # Preserve sentinel strings; otherwise parse to float
        if raw_t0 == 'start':
            t0 = 'start'
        else:
            try:
                t0 = self._ts_hms_to_sec(raw_t0)
            except (ValueError, IndexError):
                messagebox.showerror("Time window", "Invalid start time. Use HH:MM:SS or 'start'.")
                return None
        if raw_t1 == 'end':
            t1 = 'end'
        else:
            try:
                t1 = self._ts_hms_to_sec(raw_t1)
            except (ValueError, IndexError):
                messagebox.showerror("Time window", "Invalid end time. Use HH:MM:SS or 'end'.")
                return None
        # Validate: if both are numeric, require t1 > t0
        if not isinstance(t0, str) and not isinstance(t1, str):
            if float(t1) <= float(t0):
                messagebox.showerror("Time window", "End time must be after start time.")
                return None
        try:
            n_splits = max(1, int(getattr(self, '_ts_splits_var', None) and self._ts_splits_var.get() or 1))
        except (ValueError, TypeError):
            n_splits = 1
        try:
            overlap_sec = max(0.0, float(getattr(self, '_ts_overlap_sec_var', None) and self._ts_overlap_sec_var.get() or 0))
        except (ValueError, TypeError):
            overlap_sec = 0.0
        name = self._ts_name_var.get().strip() if hasattr(self, '_ts_name_var') else ""
        if not name:
            t0_str = t0 if isinstance(t0, str) else self._ts_sec_to_hms(t0)
            t1_str = t1 if isinstance(t1, str) else self._ts_sec_to_hms(t1)
            name = f"{t0_str}–{t1_str}"
        spec = {
            'name': name,
            't0': t0,
            't1': t1,
            'filter_state': self._current_filter_state(),
            'scope': 'All' if for_all else str(for_session or self._current_session_str()),
            'created_from_session': str(self._current_session_str()),
            'sessions': ['All'] if for_all else [str(for_session or self._current_session_str())],
            'n_splits': n_splits,
            'overlap_sec': overlap_sec,
        }
        return spec

    @staticmethod
    def _custom_spec_key(spec: dict) -> tuple:
        fs = spec.get('filter_state', {}) or {}
        labels = fs.get('labels', {}) or {}
        t0_raw = spec.get('t0', 0.0)
        t1_raw = spec.get('t1', 0.0)
        t0_key = str(t0_raw) if isinstance(t0_raw, str) else float(t0_raw)
        t1_key = str(t1_raw) if isinstance(t1_raw, str) else float(t1_raw)
        return (
            str(spec.get('name', '')),
            t0_key,
            t1_key,
            str(fs.get('theme', 'segments')),
            tuple(sorted((str(k), bool(v)) for k, v in labels.items())),
        )

    def _normalize_custom_spec(self, spec: dict) -> dict:
        fs = spec.get('filter_state', {}) or {}
        labels = fs.get('labels', {}) or {}
        sessions = spec.get('sessions', []) or []
        sessions = sorted(str(s) for s in sessions if s is not None)
        t0_raw = spec.get('t0', 0.0)
        t1_raw = spec.get('t1', 0.0)
        t0 = str(t0_raw) if isinstance(t0_raw, str) and t0_raw.lower() in ('start', 'end') else float(t0_raw)
        t1 = str(t1_raw) if isinstance(t1_raw, str) and t1_raw.lower() in ('start', 'end') else float(t1_raw)
        return {
            'name': str(spec.get('name', '')),
            't0': t0,
            't1': t1,
            'filter_state': {
                'theme': str(fs.get('theme', 'segments')),
                'labels': {str(k): bool(v) for k, v in labels.items()},
            },
            'scope': str(spec.get('scope', self._current_session_str())),
            'created_from_session': str(spec.get('created_from_session', self._current_session_str())),
            'sessions': sessions,
            'n_splits': int(spec.get('n_splits') or 1),
            'overlap_sec': float(spec.get('overlap_sec') or 0.0),
        }

    def _load_custom_ccg_suggestions(self) -> list[dict]:
        path = self._custom_ccg_suggestions_path
        if not os.path.isfile(path):
            return []
        try:
            with open(path, encoding='utf-8') as f:
                raw = json.load(f)
            out = [self._normalize_custom_spec(x) for x in (raw.get('items', []) or [])
                   if isinstance(x, dict)]
            return [x for x in out
                    if not self._suppress_legacy_post_split_suggestion_name(x.get('name', ''))]
        except Exception as ex:
            print(f"[CustomCCG] suggestion list load failed: {ex}")
            return []

    def _save_custom_ccg_suggestions(self, specs: list[dict]):
        payload = {
            'version': 1,
            'items': [self._normalize_custom_spec(s) for s in specs],
        }
        self._atomic_write_json(self._custom_ccg_suggestions_path, payload)

    def _record_custom_ccg_suggestion(self, spec: dict):
        norm = self._normalize_custom_spec(spec)
        if self._suppress_legacy_post_split_suggestion_name(norm.get('name', '')):
            return
        key = (self._custom_spec_key(norm), norm.get('scope', ''))
        specs = self._load_custom_ccg_suggestions()
        existing = {(self._custom_spec_key(s), s.get('scope', '')) for s in specs}
        if key in existing:
            return
        specs.append(norm)
        self._save_custom_ccg_suggestions(specs)

    def _available_custom_ccg_specs(self) -> dict[tuple, dict]:
        pattern = os.path.join(self._ccg_cache_dir, "*.npz")
        by_key: dict[tuple, dict] = {}
        for p in sorted(_glob.glob(pattern)):
            try:
                npz = np.load(p, allow_pickle=False)
                base = os.path.basename(p)
                session = str(base.split("__", 1)[0]) if "__" in base else ""
                nm = str(npz['name_'])
                if self._suppress_legacy_post_split_suggestion_name(nm):
                    continue
                spec = {
                    'name': nm,
                    't0': float(npz['t0_']),
                    't1': float(npz['t1_']),
                    'filter_state': (json.loads(str(npz['filter_state_']))
                                     if 'filter_state_' in npz else {}),
                    'scope': session,
                    'created_from_session': session,
                    'sessions': [session],
                }
            except Exception:
                continue
            k = self._custom_spec_key(spec)
            if k not in by_key:
                by_key[k] = self._normalize_custom_spec(spec)
            else:
                cur = set(by_key[k].get('sessions', []))
                cur.add(session)
                by_key[k]['sessions'] = sorted(cur)
        all_sessions = sorted(str(nk.session) for nk in self._real_nd_keys_ordered())
        for entry in by_key.values():
            sess = entry.get('sessions', [])
            if all_sessions and sorted(sess) == all_sessions:
                entry['scope'] = 'All'
            elif len(sess) == 1:
                entry['scope'] = sess[0]
            else:
                entry['scope'] = 'By session'
        return by_key

    def _custom_ccg_inventory_signature(self) -> tuple:
        avail = self._available_custom_ccg_specs()
        rows = []
        for key, spec in avail.items():
            rows.append((key, tuple(spec.get('sessions', [])), str(spec.get('scope', ''))))
        return tuple(sorted(rows))

    def _emit_custom_ccg_inventory_event(self):
        """Event-driven sync point for custom CCG availability changes."""
        sig = self._custom_ccg_inventory_signature()
        if sig != getattr(self, '_custom_ccg_inventory_sig', tuple()):
            self._custom_ccg_inventory_sig = sig
            self._refresh_custom_ccg_suggestions(silent=True)

    def _refresh_custom_ccg_suggestions(self, silent: bool = False):
        """Rebuild suggestion list from saved custom CCG npz metadata."""
        specs = sorted(
            self._available_custom_ccg_specs().values(),
            key=lambda x: (x['name'], x['t0'], x['scope'])
        )
        specs = [s for s in specs
                 if not self._suppress_legacy_post_split_suggestion_name(s.get('name', ''))]
        self._save_custom_ccg_suggestions(specs)
        if not silent:
            messagebox.showinfo("Custom CCG suggestions",
                                f"Updated suggestion list with {len(specs)} item(s).")

    def _on_split_batch_task_done(self, task):
        """Decrement split-batch counter (compute finished, load failed, or queue removed)."""
        if not isinstance(task, dict):
            return
        bid = task.get('split_batch_id')
        if bid is None:
            return
        counts = getattr(self, '_split_batch_counts', None) or {}
        if bid not in counts:
            return
        counts[bid] -= 1
        if counts[bid] > 0:
            return
        del counts[bid]
        names = list((getattr(self, '_split_batch_chunk_names', None) or {}).pop(bid, []))
        self.root.after(100, lambda n=names: self._prompt_save_split_batch_custom_ccgs(n))

    def _prompt_save_split_batch_custom_ccgs(self, names: list[str]):
        """After all tasks in a time-slider split batch finish, offer to save unsaved chunks."""
        name_set = set(names)
        if not name_set:
            return
        cs_list = getattr(self, '_custom_segments', []) or []
        indices = [i for i, cs in enumerate(cs_list)
                   if cs.get('name') in name_set and not cs.get('src_path')]
        if not indices:
            return
        n = len(indices)
        if messagebox.askyesno(
                "Save split windows",
                f"{n} split window(s) finished computing but are not saved to disk yet.\n\n"
                "Save them as .npz files now? (You can reload them later from the cache.)"):
            self._save_custom_segments_at_indices(indices)

    def _save_custom_segment_objects(self, segments: list) -> list[str]:
        """Write custom segment dicts to npz (correct session prefix per segment)."""
        saved: list[str] = []
        for cs in segments:
            if not isinstance(cs, dict):
                continue
            save_key = self._key_for_custom_segment_save(cs)
            _sess_w = str(save_key.session)
            self._purge_timestamped_custom_ccg_npz(_sess_w, str(cs['name']))
            fname = self._ccg_cache_filename_for_key(cs['name'], key=save_key)
            path = os.path.join(self._ccg_cache_dir, fname)
            arrays = dict(
                name_=np.array(cs['name']),
                t0_=np.array(cs['t0']),
                t1_=np.array(cs['t1']),
                ccg=cs['ccg'],
                ccg_null=cs['ccg_null'],
                pval=cs['pval'],
                pval_corrected=cs['pval_corrected'],
                compute_sec_=np.array(cs.get('compute_sec', float('nan'))),
                active_duration_=np.array(cs.get('active_duration', float('nan'))),
                total_time_hours_=np.array(cs.get('total_time_hours', float('nan'))),
                filter_state_=np.array(json.dumps(cs.get('filter_state', {}))),
                metadata_=np.array(json.dumps(cs.get('metadata', {}))),
                **({'firing_rates': cs['firing_rates']}
                   if 'firing_rates' in cs else {}),
            )
            for k in ('ccg_hi', 'ccg_null_hi', 'pval_hi', 'pval_corrected_hi'):
                if k in cs:
                    arrays[k] = cs[k]
            np.savez_compressed(path, **arrays)
            cs['src_path'] = path
            saved.append(str(cs['name']))
        if saved:
            messagebox.showinfo(
                "Saved",
                "Saved custom CCG segment(s):\n" + "\n".join(f"  • {n}" for n in saved))
            if hasattr(self, '_ts_status_var'):
                self._ts_status_var.set(f"Saved: {', '.join(saved)}")
            self._emit_custom_ccg_inventory_event()
            self._save_ui_state()
        return saved

    def _save_custom_segments_at_indices(self, indices: list[int]) -> list[str]:
        """Write selected custom segments (indices into active ``_custom_segments``) to npz."""
        objs = []
        for ci in sorted(set(indices)):
            if 0 <= ci < len(self._custom_segments):
                objs.append(self._custom_segments[ci])
        return self._save_custom_segment_objects(objs)

    def _ts_save_custom_ccg(self):
        """Save one or more current custom segments to disk."""
        buckets = getattr(self, '_custom_segments_by_session', None) or {}
        if not any(lst for lst in buckets.values()):
            messagebox.showinfo("Save custom CCG", "No custom segments to save.")
            return

        any_mode = getattr(self, '_session_any_mode', False)
        total_sess = max(1, len(self._real_nd_keys_ordered()))

        if any_mode:
            name_to_unsaved: dict[str, list] = collections.defaultdict(list)
            for lst in buckets.values():
                for cs in lst:
                    if not isinstance(cs, dict) or cs.get('src_path'):
                        continue
                    nm = str(cs.get('name', '')).strip() or '(unnamed)'
                    name_to_unsaved[nm].append(cs)
            if not name_to_unsaved:
                messagebox.showinfo(
                    "Save custom CCG",
                    "No unsaved custom segments. "
                    "(Everything in memory already has a .npz on disk.)")
                return
            win = tk.Toplevel(self.root)
            win.title("Save custom CCG — All sessions")
            win.geometry("440x300")
            win.transient(self.root)
            win.grab_set()
            ttk.Label(
                win,
                text="Unsaved segments (grouped by name). Select rows to save:",
            ).pack(anchor='w', padx=8, pady=(8, 2))
            lb = tk.Listbox(win, selectmode=tk.MULTIPLE, height=10)
            lb.pack(fill=tk.BOTH, expand=True, padx=8)
            row_cs: list[list] = []
            for nm in sorted(name_to_unsaved.keys(), key=lambda s: s.lower()):
                cs_list = name_to_unsaved[nm]
                sess_set = {self._custom_segment_disk_session(c) for c in cs_list}
                n_u = len(sess_set)
                lb.insert(tk.END, f"{nm} ({n_u}/{total_sess} sessions unsaved)")
                row_cs.append(cs_list)
            lb.select_set(0, tk.END)
            chosen: list[int] = []

            def _ok():
                chosen.extend(lb.curselection())
                win.destroy()

            btn_f = ttk.Frame(win)
            btn_f.pack(fill=tk.X, padx=8, pady=6)
            ttk.Button(btn_f, text="Save selected", command=_ok).pack(
                side=tk.RIGHT, padx=4)
            ttk.Button(btn_f, text="Cancel",
                       command=win.destroy).pack(side=tk.RIGHT)
            win.wait_window(win)
            if not chosen:
                return
            to_save_cs: list = []
            for r in chosen:
                to_save_cs.extend(row_cs[int(r)])
            self._save_custom_segment_objects(to_save_cs)
            return

        if len(self._custom_segments) == 1:
            to_save = [0]
        else:
            win = tk.Toplevel(self.root)
            win.title("Save custom CCG")
            win.geometry("340x260")
            win.transient(self.root)
            win.grab_set()
            ttk.Label(win, text="Select segments to save:").pack(
                anchor='w', padx=8, pady=(8, 2))
            lb = tk.Listbox(win, selectmode=tk.MULTIPLE, height=8)
            lb.pack(fill=tk.BOTH, expand=True, padx=8)
            for cs in self._custom_segments:
                lb.insert(tk.END,
                          f"{cs['name']}  ({self._ts_sec_to_hms(cs['t0'])}–"
                          f"{self._ts_sec_to_hms(cs['t1'])})")
            lb.select_set(0, tk.END)
            chosen = []

            def _ok():
                chosen.extend(lb.curselection())
                win.destroy()

            btn_f = ttk.Frame(win)
            btn_f.pack(fill=tk.X, padx=8, pady=6)
            ttk.Button(btn_f, text="Save selected", command=_ok).pack(
                side=tk.RIGHT, padx=4)
            ttk.Button(btn_f, text="Cancel",
                       command=win.destroy).pack(side=tk.RIGHT)
            win.wait_window(win)
            if not chosen:
                return
            to_save = list(chosen)

        self._save_custom_segments_at_indices(to_save)

    def _archive_stale_custom_ccgs(self):
        """Move saved custom CCG files that pre-date the total_time_hours field to _trash/.
        Returns (n_archived, trash_dir) so the caller can notify the user."""
        prefix = self._ccg_cache_prefix()
        pattern = os.path.join(self._ccg_cache_dir, f"{prefix}*.npz")
        trash_dir = os.path.join(self._ccg_cache_dir, '_trash')
        os.makedirs(trash_dir, exist_ok=True)
        archived = []
        for p in _glob.glob(pattern):
            try:
                m = np.load(p, allow_pickle=False)
                if 'total_time_hours_' not in m:
                    dest = os.path.join(trash_dir, os.path.basename(p))
                    shutil.move(p, dest)
                    archived.append(os.path.basename(p))
            except Exception:
                pass
        return len(archived), trash_dir

    def _ts_load_custom_ccg(self):
        """Scan cache dir for saved custom segments and load selected ones additively."""
        # Archive stale files (those missing total_time_hours) before showing dialog
        n_archived, trash_dir = self._archive_stale_custom_ccgs()
        if n_archived:
            messagebox.showinfo(
                "Stale custom CCGs archived",
                f"{n_archived} old custom CCG file(s) were moved to:\n  {trash_dir}\n\n"
                "These files lack the 'total_time' field required for correct Time Span "
                "normalisation and cannot be loaded. They are preserved in the trash folder.")
        prefix = self._ccg_cache_prefix()
        if getattr(self, '_session_all_mode', False):
            pattern = os.path.join(self._ccg_cache_dir, "*.npz")
        else:
            pattern = os.path.join(self._ccg_cache_dir, f"{prefix}*.npz")
        paths = sorted(_glob.glob(pattern))
        if not paths:
            messagebox.showinfo(
                "Load custom CCG",
                f"No saved custom CCGs found for\n{prefix[:-2]}")
            return

        win = tk.Toplevel(self.root)
        win.title("Load custom CCG")
        win.geometry("560x380")
        win.transient(self.root)
        win.grab_set()
        ttk.Label(win, text="Select segments to load (click ▶ to expand details):").pack(
            anchor='w', padx=8, pady=(8, 2))

        tree_frame = ttk.Frame(win)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8)
        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        tv = ttk.Treeview(tree_frame, selectmode='extended',
                          yscrollcommand=vsb.set, show='tree')
        vsb.config(command=tv.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tv.column('#0', width=520, stretch=True)

        # tag: detail rows are not selectable, shown in grey
        tv.tag_configure('detail', foreground='#666')
        tv.tag_configure('file',   foreground='#000')
        tv.tag_configure('group',  foreground='#333', font=('TkDefaultFont', 9, 'bold'))

        name_cov, _n_total_sessions = self._custom_ccg_name_session_coverage()
        _n_sess_denom = max(1, _n_total_sessions)

        def _dur_str(dur):
            if dur != dur:  return ""          # nan
            if dur >= 60:   return f"  ⏱ {dur/60:.1f}min"
            return f"  ⏱ {dur:.0f}s"

        def _parse_meta(p, *, child_session_line: bool = False):
            """Return (summary_line, [bullet_strings]) from npz metadata."""
            base = os.path.basename(p)
            file_sess = base.split("__", 1)[0] if "__" in base else ""
            parts = base.replace('.npz', '').rsplit('__', 1)
            date_str = parts[1] if len(parts) > 1 else ''
            safe_name = parts[0].replace('_', ' ') if parts else base
            bullets = []
            t0, t1 = None, None
            try:
                m = np.load(p, allow_pickle=False)
                if 'name_' in m:
                    safe_name = str(m['name_'])
                t0 = float(m['t0_']) if 't0_' in m else None
                t1 = float(m['t1_']) if 't1_' in m else None
                if t0 is not None and t1 is not None:
                    bullets.append(
                        f"Time range: {self._ts_sec_to_hms(t0)} – {self._ts_sec_to_hms(t1)}")
                dur = float(m['compute_sec_']) if 'compute_sec_' in m else float('nan')
                if dur == dur:
                    bullets.append(f"Compute time: {_dur_str(dur).strip()}")
                act = float(m['active_duration_']) if 'active_duration_' in m else float('nan')
                if act == act:
                    bullets.append(f"Active duration: {act:.1f}s")
                if 'filter_state_' in m:
                    fs = json.loads(str(m['filter_state_']))
                    theme = fs.get('theme', '')
                    labels = fs.get('labels', {})
                    if theme:
                        bullets.append(f"Theme: {theme}")
                    if labels:
                        on  = [l for l, v in labels.items() if v]
                        off = [l for l, v in labels.items() if not v]
                        if on:
                            bullets.append(f"Active labels: {', '.join(on)}")
                        if off:
                            bullets.append(f"Inactive labels: {', '.join(off)}")
                if 'metadata_' in m:
                    md = json.loads(str(m['metadata_']))
                    src = md.get('session')
                    if src:
                        bullets.append(f"Derived from session: {src}")
                has_fr = 'firing_rates' in m
                has_hi = 'ccg_hi' in m
                flags = []
                if has_fr: flags.append("firing rates")
                if has_hi: flags.append("hi-res CCG")
                if flags:
                    bullets.append(f"Contains: {', '.join(flags)}")
            except Exception:
                pass
            if child_session_line:
                tr = ""
                if t0 is not None and t1 is not None:
                    tr = (f"{self._ts_sec_to_hms(t0)}–{self._ts_sec_to_hms(t1)}  ·  ")
                summary = f"{file_sess or '?'}  ·  {tr}[{date_str}]"
            else:
                summary = f"{safe_name}  [{date_str}]"
            return summary, bullets

        # file_meta maps iid → path
        file_meta = {}

        def _populate(path_list):
            for top in tv.get_children():
                tv.delete(top)
            file_meta.clear()
            _groups: dict[str, list[str]] = collections.defaultdict(list)
            for p in path_list:
                spec = self._custom_npz_spec(p) or {}
                nm = str(spec.get('name', '')).strip()
                if not nm:
                    nm = os.path.basename(p).replace('.npz', '')
                if self._suppress_legacy_post_split_suggestion_name(nm):
                    continue
                _groups[nm].append(p)
            for gname in sorted(_groups.keys(), key=lambda s: s.lower()):
                plist = sorted(
                    _groups[gname],
                    key=lambda pp: (os.path.basename(pp).split('__', 1)[0], pp),
                )
                n_have = len(name_cov.get(gname, set()))
                parent_text = (
                    f"▶ {gname}  ({n_have}/{_n_sess_denom} sessions)")
                pid = tv.insert('', tk.END, text=parent_text,
                                tags=('group',), open=True)
                for p in plist:
                    summary, bullets = _parse_meta(p, child_session_line=True)
                    iid = tv.insert(pid, tk.END, text=f"☐  {summary}",
                                    tags=('file',), open=False)
                    file_meta[iid] = p
                    for b in bullets:
                        tv.insert(iid, tk.END, text=f"    • {b}", tags=('detail',))

        _populate(paths)

        # Clicking a file row toggles its checkbox marker (selection via Treeview selection)
        # Prevent detail rows from being selected
        def _on_click(event):
            iid = tv.identify_row(event.y)
            if iid:
                tags = tv.item(iid, 'tags')
                if 'detail' in tags or 'group' in tags:
                    tv.selection_remove(iid)

        tv.bind('<ButtonRelease-1>', _on_click)

        def _refresh_list():
            new_paths = sorted(_glob.glob(pattern))
            _populate(new_paths)
            if not file_meta:
                win.destroy()

        def _selected_file_iids():
            return [iid for iid in tv.selection()
                    if 'file' in tv.item(iid, 'tags')]

        def _delete():
            sel = _selected_file_iids()
            if not sel:
                return
            names = [tv.item(iid, 'text').lstrip('☐ ').split('  [')[0] for iid in sel]
            if not messagebox.askyesno(
                    "Delete files",
                    f"Permanently delete {len(sel)} file(s)?\n"
                    + "\n".join(f"  • {n}" for n in names)):
                return
            for iid in sel:
                try:
                    os.remove(file_meta[iid])
                except Exception as ex:
                    print(f"[LoadCustomCCG] delete failed: {ex}")
            _refresh_list()
            self._emit_custom_ccg_inventory_event()

        def _ok():
            sel = _selected_file_iids()
            if not sel:
                win.destroy()
                return
            added = []
            touched_view = False
            last_idx = 0
            for iid in sel:
                p = file_meta[iid]
                try:
                    npz = np.load(p, allow_pickle=False)
                    cs = dict(
                        name=str(npz['name_']),
                        t0=float(npz['t0_']),
                        t1=float(npz['t1_']),
                        ccg=npz['ccg'],
                        ccg_null=npz['ccg_null'],
                        pval=npz['pval'],
                        pval_corrected=npz['pval_corrected'],
                        compute_sec=(float(npz['compute_sec_'])
                                     if 'compute_sec_' in npz else float('nan')),
                        active_duration=(float(npz['active_duration_'])
                                         if 'active_duration_' in npz else float('nan')),
                        total_time_hours=(float(npz['total_time_hours_'])
                                          if 'total_time_hours_' in npz else None),
                        filter_state=(json.loads(str(npz['filter_state_']))
                                      if 'filter_state_' in npz else {}),
                        metadata=(json.loads(str(npz['metadata_']))
                                  if 'metadata_' in npz else {}),
                        src_path=p,
                        **(({'firing_rates': npz['firing_rates']})
                           if 'firing_rates' in npz else {}),
                    )
                    for k in ('ccg_hi', 'ccg_null_hi', 'pval_hi', 'pval_corrected_hi'):
                        if k in npz:
                            cs[k] = npz[k]
                    bn = os.path.basename(p)
                    file_sess = bn.split("__", 1)[0] if "__" in bn else str(self.key.session)
                    lst = self._custom_segments_by_session.setdefault(file_sess, [])
                    last_idx, _ = self._upsert_custom_segment_by_name(lst, cs)
                    if lst is self._custom_segments:
                        touched_view = True
                    added.append(cs['name'])
                except Exception as ex:
                    print(f"[LoadCustomCCG] failed to load {p}: {ex}")
            win.destroy()
            if added and touched_view:
                self._build_sig_chips()
                self.current_segment = self.n_segments + 1 + last_idx
                self._clamp_current_segment_for_session()
                self._update_segment_label()
                self.update_plot()
                if hasattr(self, '_ts_status_var'):
                    self._ts_status_var.set(f"Loaded: {', '.join(added)}")
                self._save_ui_state()
            elif added:
                if hasattr(self, '_ts_status_var'):
                    self._ts_status_var.set(
                        f"Loaded {len(added)} segment(s) for other session(s); "
                        "switch pairs to that session to view chips.")
                self._save_ui_state()

        btn_f = ttk.Frame(win)
        btn_f.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(btn_f, text="Load selected", command=_ok).pack(
            side=tk.RIGHT, padx=4)
        ttk.Button(btn_f, text="Delete selected",
                   command=_delete).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_f, text="Cancel",
                   command=win.destroy).pack(side=tk.RIGHT)


# ---------------------------------------------------------------------------

def _load_last_key() -> 'Key | None':
    """Return the Key saved in ui_state.json from the last session, or None."""
    ui_state_path = os.path.join(
        str(_Path(__file__).resolve().parents[2] / "data" / "selections"),
        'ui_state.json')
    try:
        with open(ui_state_path, 'r') as f:
            state = json.load(f)
        kd = state.get('last_key')
        if not kd:
            return None
        if kd.get('conn_type') is not None:
            kd['conn_type'] = tuple(kd['conn_type'])
        return Key(**kd)
    except Exception:
        return None


def launch_ccg_review(cd, key=None):
    """
    Launch CCG review UI.

    Parameters
    ----------
    cd : CCGDataset
        Dataset to review.  Call ``cd.load_highres()`` beforehand to enable
        the resolution toggle button.
    key : Key, optional
        Key identifying which CCGPointer to review.  If omitted, the last
        session opened is restored from ``ui_state.json``; if no saved state
        exists, the first key in ``cd.data`` is used.

    Examples
    --------
    >>> ui = launch_ccg_review(cd, key)
    >>> cd.load_highres(); ui = launch_ccg_review(cd, key)
    """
    if key is None:
        key = _load_last_key()
        if key is None or key not in cd.data:
            key = next(iter(cd.data))
    ui = CCGReviewUI(cd, key)
    ui.run()
    return ui
