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
  Ctrl+L        toggle bar ↔ line plot style
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
from neuropy.analyses.ms_connectivity import EranConv, CCGConfig, CCGData, _CCG_RESOLUTION, deconv_autocorr, NormalizeBy, apply_norms_to_ccg
from neuropy.core.neurons import Neurons
import imageio
from neuropy.ui.ccg_network_panel import NetworkPanel

try:
    from neuropy.core.epoch import Epoch as _Epoch
except ImportError:
    _Epoch = None

# Sentinel value for the virtual "All segments" view
_ALL_SEGS = "All segments"
_ADMITTED_GROUP = "__admitted__"
_SPECIAL_PREFIX = "__special_"

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

        # Group state  {group_name -> {session_str -> set((ref,tgt))}}
        # Initialized early because all_inds @property reads _groups
        self._groups: dict = {}
        self._groups.setdefault(_ADMITTED_GROUP, {})
        self._group_hotkeys: dict = {}       # group_name -> hotkey str e.g. 'Control-1'
        self._group_notes: dict = {}         # group_name -> notes string
        # Pair tags  {(ref, tgt): {"notes": str, "tags": [str, ...]}}
        self._pair_tags: dict = {}

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

        # Undo/redo stack for pair selection changes
        self._undo_stack: list = []  # list of (selected_inds_copy, unselected_inds_copy, deleted_inds_copy)
        self._redo_stack: list = []
        self._UNDO_LIMIT = 30

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
        self._jitter_thread: threading.Thread = None     # currently running custom-CCG thread
        self._custom_ccg_thread_result: list = []        # [result_dict] set by thread on finish
        self._jitter_pending: collections.deque = collections.deque()
        self._jitter_poll_id = None                      # after() id for polling
        self._jitter_unviewed: set = set()               # (ref, tgt) pairs with unviewed results
        self._jitter_mgr: 'JitterManager' = None         # lazy init after data loads

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

        # Versioned selections save dir
        self._sel_save_dir = str(
            _Path(__file__).resolve().parents[2] / "data" / "selections")
        os.makedirs(self._sel_save_dir, exist_ok=True)
        # Custom CCG cache dir
        self._ccg_cache_dir = str(
            _Path(__file__).resolve().parents[2] / "data" / "custom_ccg")
        os.makedirs(self._ccg_cache_dir, exist_ok=True)
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
        self._custom_segments: list = []

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
        self._line_baseline_var = tk.BooleanVar(master=self.root, value=True)
        self._line_ref_var = tk.BooleanVar(master=self.root, value=True)
        self._line_tgt_var = tk.BooleanVar(master=self.root, value=True)
        self._line_jitter_var = tk.BooleanVar(master=self.root, value=False)
        # Connectivity strength toggles
        self._conn_str_show_var = tk.BooleanVar(master=self.root, value=False)
        self._conn_str_method_var = tk.StringVar(master=self.root, value='conv')
        self._conn_strength_cache: dict = {}
        # Main template — built once at startup and injected into _templates
        self._main_template = None
        self._build_main_template()
        # Heartbeat: keep event loop responsive even when Jupyter cell finishes
        self._heartbeat_id = None
        self._closing = False
        self._start_heartbeat()
        self.setup_ui()

    # ------------------------------------------------------------------
    # Derived state
    # ------------------------------------------------------------------

    @property
    def _res_key(self):
        """Current resolution key for cache keying ('hi' or 'lo')."""
        return 'hi' if getattr(self, '_highres_mode', False) else 'lo'

    @property
    def all_inds(self):
        """Significant pairs + manually admitted pairs, as Nx2 numpy array.

        Autocorrelograms (ref == tgt) are always excluded.
        Returns an empty (0,2) array when data is not yet loaded.
        """
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
        extra = sorted((p for p in admitted if p[0] != p[1]) - base_set)
        if not extra:
            return base
        return np.vstack([base, np.array(extra, dtype=base.dtype)])

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
        print(f"[CCGReviewUI] group_add: {gname!r} += {pair} @ session={sess!r}")
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
            self.root.bind(_key, lambda e: self._search_show())
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
        """Persist panel visibility, settings, and cache config to ui_state.json."""
        try:
            state = {
                'panels':       {n: v.get() for n, v in self._panel_vars.items()},
                'settings':     self._settings,
                'cache_config': self._cache_config,
            }
            with open(self._ui_state_path(), 'w') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def setup_panels_menu(self, menubar):
        """Panels menu with checkbuttons for each panel."""
        panels_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Panels", menu=panels_menu)
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
        file_menu.add_command(label="Close", command=self.root.destroy)

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
        """Modules menu — Jitter and Simulation sub-menus."""
        modules_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Modules", menu=modules_menu)

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
            'Jitter': "Significance test for a pairwise connection using surrogate data",
            'Simulation': "Simulate CCG of two random neurons with designated properties",
        }
        self._menu_tooltip_win = None
        modules_menu.bind('<<MenuSelect>>', self._on_modules_menu_hover)

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

        # Buttons row at top of bottom pane
        btn_frame = ttk.Frame(bottom_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(btn_frame, text="Compute CCG", command=lambda: self._run_simulation(
            win, sim_name_var, sim_dur_var, sim_dur_unit,
            sim_noise_var, sim_sync_var, sim_delay_var,
            sim_vars, sim_fig, sim_ax, sim_canvas, sim_state)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.RIGHT)

        # CCG canvas
        sim_fig = Figure(figsize=(6, 3.5))
        sim_ax = sim_fig.add_subplot(111)
        sim_ax.set_title("(no simulation run yet)", fontsize=10)
        sim_canvas = FigureCanvasTkAgg(sim_fig, master=bottom_frame)
        sim_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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

    def _run_simulation(self, win, name_var, dur_var, unit_var,
                        noise_var, sync_var, delay_var,
                        sim_vars, fig, ax, canvas, sim_state):
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

        # CCG config matching main UI
        conf = self.ccg_data.conf if self.ccg_data is not None else CCGConfig()
        bin_size = conf.bin_size if conf.bin_size else 1e-3
        duration = conf.duration if conf.duration else 20e-3

        # Minimal Neurons object — only spiketrains needed by np_spike_correlations
        sim_neurons = Neurons(
            spiketrains=np.array([ref_train, tgt_train], dtype=object),
            t_stop=dur_s,
            t_start=0.0,
            sampling_rate=30000,
            neuron_ids=np.array([0, 1]),
        )

        from neuropy.analyses.correlations import np_spike_correlations
        ccg_raw = np_spike_correlations(
            sim_neurons,
            neuron_inds=np.array([0, 1]),
            bin_size=bin_size,
            window_size=duration,
            symmetrize=True,
        )
        # Shape: [2, 2, n_bins] → extract ref→tgt 1-D CCG
        ccg_1d = ccg_raw[0, 1, :]

        # Run convolution predictor directly (avoids EranConv's neuron-type indexing)
        pvals_all, pred_all, qvals_all = EranConv._conv(
            ccg_1d, W=conf.conv_window_bins, wintype="gauss")
        null_1d = pred_all[0]   # _conv wraps 1D input in a batch dim
        pval_1d = pvals_all[0]

        n_bins = len(ccg_1d)
        bin_size_eff = duration / (n_bins - 1) if n_bins > 1 else bin_size

        # Store in sim_state for future redraws
        sim_state['ccg'] = ccg_1d
        sim_state['null'] = null_1d
        sim_state['pval'] = pval_1d
        sim_state['conf'] = conf
        sim_state['bin_size_eff'] = bin_size_eff
        sim_state['params'] = params
        sim_state['name'] = name_var.get()

        # Plot
        ax.clear()
        ref_nick = params['ref']['nickname']
        tgt_nick = params['tgt']['nickname']
        plot_ccg.plot_ccg_panel(
            ax=ax, ccg=ccg_1d,
            ids=(ref_nick, tgt_nick), inds=(0, 1),
            window_size=duration, bin_size=bin_size_eff,
            ccg_null=null_1d,
            segment_id=f"sim: {name_var.get()}",
            min_lag=conf.min_lag,
            max_lag=conf.max_lag,
            line_baseline=True,
        )
        fig.tight_layout()
        canvas.draw()

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
                if task[0] == 'jitter':
                    ref, tgt = task[1], task[2]
                    n = task[3]
                    res = task[4]
                    seg = task[6] if len(task) > 6 else None
                    seg_s = f" seg{seg}" if seg is not None else ""
                    status = "▶ RUNNING" if i == 0 and self._is_task_running() else f"  queued"
                    lb.insert(tk.END, f"{status}  jitter [{ref},{tgt}] n={n} {res}{seg_s}")
                elif task[0] == 'custom_ccg':
                    name = task[3]
                    status = "▶ RUNNING" if i == 0 and self._is_task_running() else f"  queued"
                    lb.insert(tk.END, f"{status}  custom CCG '{name}'")
            if lb.size() == 0:
                lb.insert(tk.END, "  (empty)")

        def _delete_selected():
            sel = lb.curselection()
            if not sel:
                return
            # Cannot delete index 0 if it is currently running
            running = self._is_task_running()
            to_remove = []
            for s in sorted(sel, reverse=True):
                if s == 0 and running:
                    continue  # skip the running task
                if s < len(self._jitter_pending):
                    to_remove.append(s)
            # Remove from deque (convert to list, delete, rebuild)
            pending = list(self._jitter_pending)
            for idx in sorted(to_remove, reverse=True):
                pending.pop(idx)
            self._jitter_pending.clear()
            self._jitter_pending.extend(pending)
            self._update_jitter_btn_text()
            _refresh()

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btn_frame, text="Delete selected", command=_delete_selected).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Refresh", command=_refresh).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.RIGHT)

        _refresh()

    def _jitter_clear_queue(self):
        """Remove all pending (non-running) tasks from the queue."""
        if not self._jitter_pending:
            messagebox.showinfo("Jitter", "Queue is empty.")
            return
        running = self._is_task_running()
        if running:
            # Keep only the first (running) task
            first = self._jitter_pending[0]
            n_removed = len(self._jitter_pending) - 1
            self._jitter_pending.clear()
            self._jitter_pending.append(first)
        else:
            n_removed = len(self._jitter_pending)
            self._jitter_pending.clear()
        self._update_jitter_btn_text()
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

        # ── Cache Configuration section ────────────────────────────────
        cache_sec = _add_section('Cache Config')
        tk.Label(cache_sec, text="Cache Configuration", bg=_CONT_BG, fg=_FG,
                 font=_HDR_FONT, anchor='w').pack(fill=tk.X, pady=(0, 4))
        ttk.Separator(cache_sec).pack(fill=tk.X, pady=(0, 10))
        tk.Label(cache_sec,
                 text="Only one display configuration is saved to the PNG disk cache.\n"
                      "Any other configuration is rendered in real-time (not cached).\n"
                      "Set up the significance panel as desired, then capture it here.",
                 bg=_CONT_BG, fg=_FG_DIM, font=('Arial', 9),
                 justify='left', wraplength=380).pack(anchor='w', pady=(0, 10))

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

        _show_section('Display')

    def _show_hotkeys_dialog(self):
        """Show a dialog listing all keyboard shortcuts."""
        hotkeys_text = (
            "Ctrl+E    Toggle between waveform and CCG\n"
            "Ctrl+L    Toggle all histograms outline/filled\n"
            "           (right-click CCG for per-item control)\n"
            "Ctrl+R    Toggle resolution (hi / lo)\n"
            "Ctrl+S    Save selection\n"
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

        for key_str in slot_order:
            gname = hk_to_group.get(key_str)
            if gname is None:
                continue
            display = f"{key_str}: {gname}"
            lbl = tk.Label(self._hotkeys_bar, text=display,
                           font=('Courier', 9), padx=6, pady=1,
                           relief=tk.RIDGE, borderwidth=1)
            lbl.pack(side=tk.LEFT, padx=2, pady=2)
            # Click label → jump to group
            lbl.bind('<Button-1>',
                     lambda e, g=gname: self._select_group(g))
            self._hotkeys_bar_labels.append(lbl)

        if not hk_to_group:
            ttk.Label(self._hotkeys_bar, text="(no hotkeys assigned)",
                      font=('Arial', 9), foreground='#888').pack(
                side=tk.LEFT, padx=4)

    # ── Left panel ─────────────────────────────────────────────────────

    def setup_left_panel(self, parent):
        # Tabbed notebook: Pair Selection | Spike Pairs
        self._left_notebook = ttk.Notebook(parent)
        self._left_notebook.pack(fill=tk.BOTH, expand=True)

        # ── Tab 1: Pair Selection ─────────────────────────────────────
        pair_tab = ttk.Frame(self._left_notebook)
        self._left_notebook.add(pair_tab, text="Pair Selection")

        columns_frame = ttk.Frame(pair_tab)
        columns_frame.pack(fill=tk.BOTH, expand=True, pady=6)

        # Unselected list
        unsel_frame = ttk.Frame(columns_frame)
        unsel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))
        self._avail_label_var = tk.StringVar(
            value=f"Available ({len(self.unselected_inds)})")
        _avail_hdr = ttk.Frame(unsel_frame)
        _avail_hdr.pack(fill=tk.X)
        ttk.Label(_avail_hdr, textvariable=self._avail_label_var,
                  font=('Arial', 10)).pack(side=tk.LEFT)
        self._clear_spec_btn = ttk.Button(
            _avail_hdr, text="✕ predictions",
            command=self._clear_speculated, width=12)
        # shown only when speculated is active — packed lazily in refresh_lists
        unsel_scroll = ttk.Scrollbar(unsel_frame)
        unsel_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.unselected_list = tk.Listbox(
            unsel_frame, yscrollcommand=unsel_scroll.set,
            selectmode=tk.EXTENDED, font=('Courier', 9), activestyle='none',
            width=1)
        self.unselected_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        unsel_scroll.config(command=self.unselected_list.yview)
        # Single click → navigate (debounced); double-click → move
        self.unselected_list.bind('<ButtonRelease-1>', self.on_pair_select)
        self.unselected_list.bind('<Double-Button-1>', self.move_to_selected)
        self.unselected_list.bind('<Return>',          self.move_to_selected)
        # Bind right-click (Button-3) and macOS two-finger/ctrl-click (Button-2)
        self.unselected_list.bind('<Button-3>',
            lambda e: self._ctx_menu(e, self.unselected_list, 'add'))
        self.unselected_list.bind('<Button-2>',
            lambda e: self._ctx_menu(e, self.unselected_list, 'add'))
        self.unselected_list.bind('<KeyRelease-Up>',   self._on_arrow_key)
        self.unselected_list.bind('<KeyRelease-Down>', self._on_arrow_key)
        # Freeze horizontal scrolling — pass Left/Right to segment navigation
        self.unselected_list.bind('<Left>',  lambda e: (self.change_segment(-1), 'break')[1])
        self.unselected_list.bind('<Right>', lambda e: (self.change_segment(1),  'break')[1])

        # Selected list
        sel_frame = ttk.Frame(columns_frame)
        sel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 0))
        self._sel_label_var = tk.StringVar(
            value=f"Selected ({len(self.selected_inds)})")
        ttk.Label(sel_frame, textvariable=self._sel_label_var,
                  font=('Arial', 10)).pack()
        sel_scroll = ttk.Scrollbar(sel_frame)
        sel_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.selected_list = tk.Listbox(
            sel_frame, yscrollcommand=sel_scroll.set,
            selectmode=tk.EXTENDED, font=('Courier', 9), activestyle='none')
        self.selected_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sel_scroll.config(command=self.selected_list.yview)
        self.selected_list.bind('<ButtonRelease-1>', self.on_pair_select)
        self.selected_list.bind('<Double-Button-1>', self.move_to_unselected)
        self.selected_list.bind('<Return>',          self.move_to_unselected)
        self.selected_list.bind('<Button-3>',
            lambda e: self._ctx_menu(e, self.selected_list, 'remove'))
        self.selected_list.bind('<Button-2>',
            lambda e: self._ctx_menu(e, self.selected_list, 'remove'))
        self.selected_list.bind('<KeyRelease-Up>',   self._on_arrow_key)
        self.selected_list.bind('<KeyRelease-Down>', self._on_arrow_key)
        # Freeze horizontal scrolling — pass Left/Right to segment navigation
        self.selected_list.bind('<Left>',  lambda e: (self.change_segment(-1), 'break')[1])
        self.selected_list.bind('<Right>', lambda e: (self.change_segment(1),  'break')[1])

        self.refresh_lists()

        # Buttons row: Select All/Deselect All / resolution toggle
        btn_frame = ttk.Frame(pair_tab)
        btn_frame.pack(fill=tk.X, pady=(4, 0))
        self._select_all_btn = ttk.Button(btn_frame, text="Select All",
                                          command=self._select_all)
        self._select_all_btn.pack(side=tk.LEFT, padx=(0, 4))

        # Sort selected list by group combo / individual tag / mean CCG
        self._sort_selected_var = tk.BooleanVar(value=False)
        self._sort_by_tag_var   = tk.BooleanVar(value=False)
        self._sort_by_mean_var  = tk.BooleanVar(value=False)
        ttk.Checkbutton(btn_frame, text="Sort by group",
                        variable=self._sort_selected_var,
                        command=self._on_sort_by_group_toggle).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(btn_frame, text="Sort by tag",
                        variable=self._sort_by_tag_var,
                        command=self._on_sort_by_tag_toggle).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(btn_frame, text="Sort by mean",
                        variable=self._sort_by_mean_var,
                        command=self._on_sort_by_mean_toggle).pack(side=tk.LEFT, padx=2)

        # ── Search bar (hidden; shown via Ctrl+F) ─────────────────────
        search_frame = ttk.Frame(pair_tab)
        # not packed — shown on demand
        ttk.Label(search_frame, text="🔍").pack(side=tk.LEFT, padx=(0, 2))
        self._search_var = tk.StringVar()
        self._search_entry = ttk.Entry(search_frame, textvariable=self._search_var)
        self._search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._search_count_var = tk.StringVar(value="")
        ttk.Label(search_frame, textvariable=self._search_count_var,
                  width=6, anchor='e').pack(side=tk.LEFT, padx=(3, 0))
        ttk.Button(search_frame, text="▲", width=2,
                   command=lambda: self._search_go(-1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(search_frame, text="▼", width=2,
                   command=lambda: self._search_go(1)).pack(side=tk.LEFT, padx=1)
        ttk.Button(search_frame, text="✕", width=2,
                   command=self._search_clear).pack(side=tk.LEFT, padx=(1, 0))
        self._search_var.trace_add('write', lambda *_: self._search_update())
        self._search_entry.bind('<Return>',        lambda e: self._search_go(1))
        self._search_entry.bind('<Shift-Return>',  lambda e: self._search_go(-1))
        self._search_entry.bind('<Escape>',        lambda e: self._search_hide())
        self._search_frame = search_frame
        self._search_matches: list = []   # [(listbox, idx), ...]
        self._search_cur: int = -1

        # ── Tab 2: Spike Pairs (spike attribution) ────────────────────
        sa_tab = ttk.Frame(self._left_notebook)
        self._left_notebook.add(sa_tab, text="Spike Pairs", state='disabled')
        self._sa_tab = sa_tab
        self._sa_tab_index = 1

        self._sa_count_var = tk.StringVar(value="")
        ttk.Label(sa_tab, textvariable=self._sa_count_var,
                  font=('Courier', 9)).pack(side=tk.TOP, anchor='w', padx=4, pady=2)

        sa_scroll = ttk.Scrollbar(sa_tab)
        sa_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._sa_listbox = tk.Listbox(
            sa_tab, yscrollcommand=sa_scroll.set,
            selectmode=tk.BROWSE, font=('Courier', 9), activestyle='none')
        self._sa_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sa_scroll.config(command=self._sa_listbox.yview)
        self._sa_listbox.bind('<<ListboxSelect>>', self._on_sa_pair_click)

    # ── Center panel ───────────────────────────────────────────────────

    def setup_center_panel(self, parent):
        self.plot_title_var = tk.StringVar(value=self.get_plot_title())
        ttk.Label(parent, textvariable=self.plot_title_var,
                  font=('Arial', 11, 'bold')).pack(side=tk.TOP)

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

        self.setup_sig_display_panel(ctrl_frame)
        self.setup_norm_panel(ctrl_frame)
        self.setup_jitter_panel(ctrl_frame)
        self.setup_spike_attrib_panel(ctrl_frame)
        self.setup_conn_strength_panel(ctrl_frame)
        self.setup_waveforms_panel(ctrl_frame)   # hidden by default

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

    def setup_sig_display_panel(self, parent):
        """Significance display toggles: conv baseline/p/p-corrected, jitter."""
        sig_frame, self._sig_fold_var = self._make_collapsible_panel(parent, "Significance")
        self._sig_inner_frame = sig_frame

        # Convolution group
        conv_lbl = ttk.Label(sig_frame, text="Convolution:", font=('Arial', 8))
        conv_lbl.pack(side=tk.LEFT, padx=(0, 1))
        self._sig_conv_baseline_var = tk.BooleanVar(value=True)
        self._baseline_style_btn = tk.Label(
            sig_frame, text="■ baseline", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._baseline_style_btn.pack(side=tk.LEFT, padx=2)
        self._baseline_style_btn.bind('<Button-1>',
            lambda e: self._cycle_baseline('conv'))
        self._sig_conv_p_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sig_frame, text="p",
                        variable=self._sig_conv_p_var,
                        command=self._on_sig_toggle).pack(side=tk.LEFT, padx=2)
        self._sig_conv_pc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sig_frame, text="p-corrected",
                        variable=self._sig_conv_pc_var,
                        command=self._on_sig_toggle).pack(side=tk.LEFT, padx=2)
        self._sig_test_window_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sig_frame, text="test window",
                        variable=self._sig_test_window_var,
                        command=self._on_sig_toggle).pack(side=tk.LEFT, padx=2)

        # Tailed Baseline — exclusive with Convolution baseline
        ttk.Separator(sig_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=2)
        ttk.Label(sig_frame, text="Tailed Baseline:", font=('Arial', 8)).pack(
            side=tk.LEFT, padx=(0, 1))
        self._sig_tailed_baseline_var = tk.BooleanVar(value=False)
        self._tailed_baseline_style_btn = tk.Label(
            sig_frame, text="X baseline", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2')
        self._tailed_baseline_style_btn.pack(side=tk.LEFT, padx=2)
        self._tailed_baseline_style_btn.bind('<Button-1>',
            lambda e: self._cycle_baseline('tailed'))

        sep = ttk.Separator(sig_frame, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        # Jitter group
        j_lbl = ttk.Label(sig_frame, text="Jitter:", font=('Arial', 8))
        j_lbl.pack(side=tk.LEFT, padx=(0, 1))
        self._sig_jitter_p_var = tk.BooleanVar(value=False)
        self._jitter_style_btn = tk.Label(
            sig_frame, text="X jitter", font=('TkDefaultFont', 9),
            relief='raised', bd=1, padx=2, cursor='hand2',
            state=tk.DISABLED, fg='gray')
        self._jitter_style_btn.pack(side=tk.LEFT, padx=2)
        self._jitter_style_btn.bind('<Button-1>',
            lambda e: self._cycle_style('jitter')
            if str(self._jitter_style_btn['state']) != 'disabled' else None)
        self._sig_jitter_pc_var = tk.BooleanVar(value=False)
        self._sig_jitter_pc_cb = ttk.Checkbutton(
            sig_frame, text="p-corrected",
            variable=self._sig_jitter_pc_var,
            command=self._on_sig_toggle)
        self._sig_jitter_pc_cb.pack(side=tk.LEFT, padx=2)
        # Disable jitter buttons initially (enabled when jitter data exists)
        self._sig_jitter_pc_cb.state(['disabled'])

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

        ttk.Separator(acg_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

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
        ttk.Checkbutton(acg_frame, text="Match CCG scale",
                        variable=self._acg_match_ccg_var,
                        command=self._on_sig_toggle).pack(side=tk.LEFT, padx=4)

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
        if self._acg_ref_var.get() or self._acg_tgt_var.get():
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
            if self._acg_ref_var.get() or self._acg_tgt_var.get():
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
        """Read a significance toggle BooleanVar by short name."""
        _map = {
            'conv_baseline':    '_sig_conv_baseline_var',
            'tailed_baseline':  '_sig_tailed_baseline_var',
            'conv_p':           '_sig_conv_p_var',
            'conv_pc':          '_sig_conv_pc_var',
            'test_window':      '_sig_test_window_var',
            'jitter_p':         '_sig_jitter_p_var',
            'jitter_pc':        '_sig_jitter_pc_var',
        }
        var = getattr(self, _map[name], None)
        return var.get() if var is not None else False

    # All attr names whose values form the "cache configuration"
    _CACHE_CONFIG_ATTRS = (
        '_sig_conv_baseline_var', '_sig_tailed_baseline_var',
        '_sig_conv_p_var', '_sig_conv_pc_var', '_sig_test_window_var',
        '_sig_jitter_p_var', '_sig_jitter_pc_var',
        '_acg_ref_var', '_acg_tgt_var',
        '_acg_yscale_ref_var', '_acg_yscale_tgt_var',
        '_acg_match_ccg_var', '_ccg_show_var',
        '_line_ccg_var', '_line_baseline_var',
        '_line_ref_var', '_line_tgt_var', '_line_jitter_var',
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
        # Tri-state jitter label button
        btn = getattr(self, '_jitter_style_btn', None)
        if btn:
            if has_jitter:
                btn.config(state=tk.NORMAL, fg='black')
            else:
                btn.config(state=tk.DISABLED, fg='gray')
                self._sig_jitter_p_var.set(False)
                self._line_jitter_var.set(False)
            self._update_style_btns()
        # p-corrected checkbutton
        if hasattr(self, '_sig_jitter_pc_cb'):
            state = ['!disabled'] if has_jitter else ['disabled']
            self._sig_jitter_pc_cb.state(state)

    def setup_norm_panel(self, parent):
        norm_inner, self._norm_fold_var = self._make_collapsible_panel(parent, "Normalization")

        options = [
            (NormalizeBy.REF_FRATE,    "Ref firing rate"),
            (NormalizeBy.TARGET_FRATE, "Tgt firing rate"),
            (NormalizeBy.TIME_SPAN,    "Time span"),
        ]
        btn_frame = ttk.Frame(norm_inner)
        btn_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        for nm, label in options:
            if self.neurons is None and nm in (
                    NormalizeBy.REF_FRATE, NormalizeBy.TARGET_FRATE):
                continue
            var = tk.BooleanVar(value=False)
            self.norm_vars[nm] = var
            cb = ttk.Checkbutton(btn_frame, text=label, variable=var,
                                 command=self._on_norm_toggle)
            cb.pack(side=tk.LEFT, padx=4)
        # Scale checkbuttons
        scale_frame = ttk.Frame(norm_inner)
        scale_frame.pack(side=tk.LEFT, padx=(12, 0))
        self._pair_scale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scale_frame, text="Same scale (pair)",
                        variable=self._pair_scale_var,
                        command=self._on_pair_scale_toggle).pack(side=tk.LEFT, padx=4)
        self._sess_scale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(scale_frame, text="Same scale (session)",
                        variable=self._sess_scale_var,
                        command=self._on_session_scale_toggle).pack(side=tk.LEFT, padx=4)

        ttk.Button(norm_inner, text="Normalize All",
                   command=self._finalize_normalization).pack(
            side=tk.RIGHT, padx=6)

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
            # Disable tab, clear spike pairs, restore CCG
            self._left_notebook.tab(self._sa_tab_index, state='disabled')
            self._left_notebook.select(0)  # back to Pair Selection
            self._sa_spike_pairs = []
            self._sa_selected_idx = -1
            self._sa_count_var.set("")
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
            self._sa_count_var.set("Invalid bin")
            return
        inds = self.all_inds[self.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        self._compute_spike_pairs(ref, tgt, self._sa_bin_ms)
        # Enable and switch to Spike Pairs tab
        self._left_notebook.tab(self._sa_tab_index, state='normal')
        self._left_notebook.select(self._sa_tab_index)

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

        # Populate listbox
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

    def setup_waveforms_panel(self, parent):
        """Waveform pane inside the CCG horizontal split — hidden by default."""
        self.wave_frame = ttk.LabelFrame(self._plot_pw, text="Waveforms")
        # Not added to _plot_pw yet — toggled via Panels menu / Ctrl+E
        self.wave_fig = Figure(figsize=(4, 5), tight_layout=True)
        self.wave_ax = self.wave_fig.add_subplot(111)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_fig, master=self.wave_frame)
        self.wave_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_conn_strength_panel(self, parent):
        """Foldable 'Connection Strength' section in the control scrollbox."""
        cs_inner, self._cs_fold_var = self._make_collapsible_panel(parent, "Connection Strength", expanded=False)

        self._conn_str_label = ttk.Label(cs_inner, text="CS: \u2014")
        self._conn_str_label.pack(side=tk.LEFT, pady=1)

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
                try:
                    sec = self._ts_hms_to_sec(var.get())
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
        ttk.Button(ccg_ctrl, text="Clear",
                   command=self._on_time_slider_clear).pack(side=tk.LEFT, padx=(8, 2))
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
        'ccg':             ('_line_ccg_var',      '_ccg_show_var'),
        'baseline':        ('_line_baseline_var', '_sig_conv_baseline_var'),
        'tailed_baseline': ('_line_baseline_var', '_sig_tailed_baseline_var'),
        'ref':             ('_line_ref_var',      '_acg_ref_var'),
        'tgt':             ('_line_tgt_var',      '_acg_tgt_var'),
        'jitter':          ('_line_jitter_var',   '_sig_jitter_p_var'),
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

    def _cycle_baseline(self, which):
        """Exclusive tri-state for conv/tailed baseline buttons.

        Activating one always deactivates the other and updates
        _conn_str_method_var to the corresponding method.
        """
        line_var = self._line_baseline_var
        if which == 'conv':
            show_var = self._sig_conv_baseline_var
            other_var = getattr(self, '_sig_tailed_baseline_var', None)
            method = 'conv'
        else:
            show_var = getattr(self, '_sig_tailed_baseline_var', None)
            if show_var is None:
                return
            other_var = self._sig_conv_baseline_var
            method = 'tailed'

        if show_var.get() and not line_var.get():    # solid → outline
            line_var.set(True)
        elif show_var.get() and line_var.get():       # outline → hidden
            show_var.set(False)
            line_var.set(False)
        else:                                          # hidden → solid
            show_var.set(True)
            line_var.set(False)
            if other_var is not None:
                other_var.set(False)
            self._conn_str_method_var.set(method)

        self._update_style_btns()
        self._clear_all_png_cache()
        self.update_plot()
        self._update_conn_str_label()

    def _update_style_btns(self):
        """Refresh tri-state button labels: ■ name / □ name / X name."""
        for item, btn_attr, name in [
            ('ccg',             '_ccg_style_btn',             'CCG'),
            ('baseline',        '_baseline_style_btn',        'baseline'),
            ('tailed_baseline', '_tailed_baseline_style_btn', 'baseline'),
            ('ref',             '_ref_style_btn',             'ref'),
            ('tgt',             '_tgt_style_btn',             'tgt'),
            ('jitter',          '_jitter_style_btn',          'jitter'),
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
                         for item in ('ccg', 'baseline', 'ref', 'tgt')
                         if self._get_style_vars(item)[1].get()]
        any_line = any(v.get() for v in visible_lines)
        for v in visible_lines:
            v.set(not any_line)
        self._update_style_btns(); self._clear_all_png_cache(); self.update_plot()

    def _ccg_context_menu(self, event):
        """Right-click context menu on the CCG plot canvas."""
        menu = tk.Menu(self.root, tearoff=0)
        # Pair actions
        if self.current_pair_idx < len(self.all_inds):
            inds = tuple(self.all_inds[self.current_pair_idx])
            ref, tgt = int(inds[0]), int(inds[1])
            in_deleted = inds in self.deleted_inds
            menu.add_command(
                label=f"Move [{ref}, {tgt}] to Deleted" if not in_deleted
                      else f"Restore [{ref}, {tgt}] from Deleted",
                command=(self._on_delete_pair if not in_deleted
                         else lambda: self._ctx_restore_from_deleted([inds])))
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
            'ccg':     (cd.ccg,      ref, tgt, f"CCG [{ref},{tgt}]"),
            'baseline':(cd.ccg_null, ref, tgt, f"Baseline [{ref},{tgt}]"),
            'pval':    (cd.pval,     ref, tgt, f"P-value [{ref},{tgt}]"),
            'acg_ref': (cd.ccg,      ref, ref, f"ACG ref [{ref}]"),
            'acg_tgt': (cd.ccg,      tgt, tgt, f"ACG tgt [{tgt}]"),
        }
        if item not in items:
            print(f"[ViewValues] Unknown item: {item}"); return
        arr, r, c, label = items[item]
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

        When seg == self.n_segments ("All segments"), returns True if
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
        # Norms change the y-axis values → invalidate scale caches
        self._pair_scale_cache.clear()
        self._session_scale_cache = None
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

    def _accumulate_ylim(self, ref, tgt, seg, ymin, ymax):
        """Accumulate y-axis limits for one (ref, tgt, seg) using normalized CCG."""
        cd = self.ccg_data
        ccg, ccg_null = apply_norms_to_ccg(
            cd.ccg[seg, ref, tgt, :],
            cd.ccg_null[seg, ref, tgt, :] if cd.ccg_null is not None else None,
            ref, tgt, seg, self.active_norms, self.neurons, self.cd.nd,
            self.key.nd(), self.n_segments, self._is_custom_segment(seg))
        ymin = min(ymin, float(ccg.min()))
        ymax = max(ymax, float(ccg.max()))
        if ccg_null is not None:
            ymax = max(ymax, float(ccg_null.max()))
        return ymin, ymax

    def _compute_pair_scale(self, ref: int, tgt: int):
        """Return (ymin, ymax) unified across all segments for this pair."""
        ymin, ymax = 0.0, 0.0
        for seg in range(self.n_segments):
            ymin, ymax = self._accumulate_ylim(ref, tgt, seg, ymin, ymax)
        return (ymin, ymax * 1.1 if ymax > 0 else 1.0)

    def _compute_session_scale(self):
        """Return (ymin, ymax) unified across all pairs and segments in this key."""
        ymin, ymax = 0.0, 0.0
        for ref_tgt in self.all_inds:
            ref, tgt = int(ref_tgt[0]), int(ref_tgt[1])
            for seg in range(self.n_segments):
                ymin, ymax = self._accumulate_ylim(ref, tgt, seg, ymin, ymax)
        return (ymin, ymax * 1.1 if ymax > 0 else 1.0)

    def _get_current_scale_ylim(self, ref: int, tgt: int):
        """Return (ymin, ymax) for the active scale mode, or None."""
        if self._same_scale_mode == 'pair':
            if (ref, tgt) not in self._pair_scale_cache:
                self._pair_scale_cache[(ref, tgt)] = self._compute_pair_scale(ref, tgt)
            return self._pair_scale_cache[(ref, tgt)]
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
        # Jitter always uses low-res CCG data
        res_key = 'lo'
        lo_ccg_data = self.cd._ccg.get(self.key.nd()) if hasattr(self.cd, '_ccg') else self.ccg_data
        lo_conf = lo_ccg_data.conf
        lo_n_bins = lo_ccg_data.ccg.shape[-1]
        bin_size_eff = lo_conf.duration / (lo_n_bins - 1) if lo_n_bins > 1 else lo_conf.bin_size

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

        # Enqueue the task
        self._jitter_pending.append(
            ('jitter', ref, tgt, njitter, res_key, bin_size_eff, seg_arg, jitter_t0, jitter_t1))
        self._update_jitter_btn_text()
        # Kick off processing if nothing is running
        self._jitter_start_next()

    def _is_task_running(self):
        return ((self._jitter_proc is not None and self._jitter_proc.is_alive()) or
                (self._jitter_thread is not None and self._jitter_thread.is_alive()))

    def _jitter_start_next(self):
        """Start the next queued task (jitter or custom CCG) if none is running."""
        if self._is_task_running():
            return
        if not self._jitter_pending:
            self._update_jitter_btn_text()
            return
        task = self._jitter_pending[0]

        if task[0] == 'jitter':
            _, ref, tgt, njitter, res_key, bin_size_eff = task[:6]
            seg_arg   = task[6] if len(task) > 6 else None
            jitter_t0 = task[7] if len(task) > 7 else None
            jitter_t1 = task[8] if len(task) > 8 else None
            # Always use low-res CCG data for jitter computation
            lo_ccg_data = (self.cd._ccg.get(self.key.nd())
                           if hasattr(self.cd, '_ccg') else self.ccg_data)
            self._jitter_result_queue = _mp.Queue()
            self._jitter_proc = _mp.Process(
                target=jitter_worker,
                args=(self._jitter_result_queue, self.key, self.neurons,
                      lo_ccg_data, self.ccg_pointer.edge_times,
                      ref, tgt, njitter, bin_size_eff),
                kwargs={'segment': seg_arg, 't0': jitter_t0, 't1': jitter_t1},
                daemon=True,
            )
            self._jitter_proc.start()

        elif task[0] == 'custom_ccg':
            _, t0, t1, name, neurons_override, active_duration, filter_state = task
            self._custom_ccg_thread_result.clear()
            _t_start = _time.monotonic()

            def _ccg_worker(_t0=t0, _t1=t1, _name=name, _no=neurons_override,
                            _ad=active_duration, _fs=filter_state):
                try:
                    result = self._compute_custom_segment(
                        _t0, _t1, _name,
                        neurons_override=_no, active_duration=_ad)
                    if result is not None:
                        result['filter_state'] = _fs
                        result['compute_sec'] = _time.monotonic() - _t_start
                    self._custom_ccg_thread_result.append(
                        result if result is not None else {'error': 'compute returned None'})
                except Exception as ex:
                    self._custom_ccg_thread_result.append({'error': str(ex)})

            self._jitter_thread = threading.Thread(
                target=_ccg_worker, daemon=True)
            self._jitter_thread.start()

        self._update_jitter_btn_text()
        if self._jitter_poll_id is None:
            self._jitter_poll_id = self.root.after(300, self._poll_jitter)

    def _update_jitter_btn_text(self):
        running = self._is_task_running()
        queued = len(self._jitter_pending)
        if running and self._jitter_pending:
            task = self._jitter_pending[0]
            if task[0] == 'jitter':
                ref, tgt = task[1], task[2]
                seg_arg = task[6] if len(task) > 6 else None
                seg_suffix = (f" seg{seg_arg}" if seg_arg is not None else "")
                label = f"Jitter [{ref},{tgt}]{seg_suffix}…"
            else:
                label = f"CCG '{task[3]}'…"
            if queued > 1:
                self._jitter_btn_text.set(f"{label} +{queued - 1} queued")
            else:
                self._jitter_btn_text.set(label)
        else:
            self._jitter_btn_text.set("Run Jitter")

    def _poll_jitter(self):
        """Poll background task (jitter or custom CCG); collect result and start next."""
        if self._is_task_running():
            self._jitter_poll_id = self.root.after(300, self._poll_jitter)
            return
        self._jitter_poll_id = None

        completed = self._jitter_pending.popleft() if self._jitter_pending else None
        task_type = completed[0] if completed else None

        if task_type == 'jitter':
            # Read result from mp.Queue
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
                seg_arg = completed[6] if len(completed) > 6 else None
                cache_key = (result['ref'], result['tgt'], res_key, seg_arg)
                jitter_val = (result['j_avg'], result['j_pval'], result['j_pval_bins'])
                self._jitter_cache_put(cache_key, jitter_val)
                # Persist in cd._jitter_results
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
                self.root.bell()
            elif result is not None and result.get('error'):
                messagebox.showerror("Jitter",
                                     f"Jitter failed:\n{result['error']}")

        elif task_type == 'custom_ccg':
            if self._jitter_thread is not None:
                self._jitter_thread.join(timeout=1)
                self._jitter_thread = None
            result = self._custom_ccg_thread_result[0] if self._custom_ccg_thread_result else None
            self._custom_ccg_thread_result.clear()

            if result is not None and not result.get('error'):
                self._custom_segments.append(result)
                self._build_sig_chips()
                self.current_segment = self.n_segments + len(self._custom_segments)
                self._update_segment_label()
                self.update_plot()
                if hasattr(self, '_ts_status_var'):
                    self._ts_status_var.set(f"Done: {result.get('name', '')}")
                self.root.bell()
            elif result is not None and result.get('error'):
                messagebox.showerror("Custom CCG",
                                     f"Computation failed:\n{result['error']}")

        # Start next in queue
        self._jitter_start_next()
        if self._jitter_proc is not None and self._jitter_proc.is_alive():
            self._jitter_poll_id = self.root.after(300, self._poll_jitter)

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

        Returns None for 'All segments' and custom segments (whole-session data).
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
        cache_key = (ref, tgt, 'lo', self._jitter_seg())
        self._jitter_cache.pop(cache_key, None)
        if hasattr(self.cd, '_jitter_results'):
            nd_key = self.key.nd()
            if nd_key in self.cd._jitter_results:
                self.cd._jitter_results[nd_key].pop(cache_key, None)
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

    def _apply_jitter_list_colors(self, pair=None):
        """Color pair list items based on jitter cache state.

        Parameters
        ----------
        pair : tuple or None
            If given as (ref, tgt), update only that pair's item.
            If None, update all items in both listboxes.
        """
        for listbox, inds_set in [(self.unselected_list, self.unselected_inds),
                                  (self.selected_list, self.selected_inds)]:
            sorted_items = sorted(inds_set)
            for idx, inds in enumerate(sorted_items):
                ref, tgt = int(inds[0]), int(inds[1])
                p = (ref, tgt)
                if pair is not None and p != pair:
                    continue
                has_any_res = any(
                    k[0] == ref and k[1] == tgt for k in self._jitter_cache)
                if p in self._jitter_unviewed:
                    listbox.itemconfig(idx, background=self._JITTER_UNVIEWED_BG,
                                       foreground=self._JITTER_UNVIEWED_FG)
                elif has_any_res:
                    listbox.itemconfig(idx, background=self._JITTER_VIEWED_BG,
                                       foreground=self._JITTER_VIEWED_FG)
                else:
                    listbox.itemconfig(idx, background='', foreground='')
                if pair is not None:
                    return  # found and updated the single pair

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
                self._sig_jitter_p_var.set(True)
                self._sig_jitter_pc_var.set(True)
                self._line_jitter_var.set(False)
            return True
        return False

    def _finalize_normalization(self):
        if not self.active_norms:
            messagebox.showinfo("Normalize All", "No normalizations are toggled on.")
            return
        if self.neurons is None and any(
                nm in (NormalizeBy.REF_FRATE, NormalizeBy.TARGET_FRATE)
                for nm in self.active_norms):
            messagebox.showerror(
                "Normalize All",
                "Neuron data is unavailable — cannot normalize by firing rate.")
            return
        norm_names = ', '.join(nm.name for nm in self.active_norms)
        if not messagebox.askyesno(
                "Normalize All",
                f"Apply {norm_names} normalization to the stored CCG data?\n\n"
                "This modifies the data in place and cannot be undone."):
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
        messagebox.showinfo("Normalize All",
                            "Normalization applied. Don't forget to Save Selections.")

    def _clear_all_png_cache(self):
        self._pregen_cancel = True  # stop any in-progress pre-generation
        for f in os.listdir(self.tmp_dir):
            if f.endswith('.png'):
                try:
                    os.remove(os.path.join(self.tmp_dir, f))
                except OSError:
                    pass
        # Restart pre-generation with current display settings
        self.root.after(50, self._pregen_png_cache)

    # ------------------------------------------------------------------
    # Idle-time PNG pre-generation
    # ------------------------------------------------------------------

    def _pregen_png_cache(self):
        """Schedule background PNG pre-generation during Tk idle time.

        Renders all pairs × segments (lo-res and hi-res if available) one
        at a time via ``root.after_idle``, so the UI stays responsive while
        the cache is populated.  Any cache miss during browsing will still
        render on demand — this just warms the cache proactively.
        """
        # Cancel any previous pre-generation queue
        self._pregen_cancel = True

        self._pregen_cancel = False
        pairs = [tuple(row) for row in self.all_inds]
        n_segs = self.n_segments
        has_highres = (hasattr(self.cd, '_ccg_highres')
                       and self.cd._ccg_highres.get(self.key.nd()) is not None)

        # Build a flat work list of (inds_array, segment, highres) tuples
        work = []
        for inds_tuple in pairs:
            inds = np.array(inds_tuple)
            for seg in list(range(n_segs)) + [n_segs]:
                for hires in ([False, True] if has_highres else [False]):
                    work.append((inds, seg, hires))

        cfg = dict(self._cache_config) if self._cache_config is not None \
              else self._current_display_config()
        self._pregen_thread = threading.Thread(
            target=self._pregen_thread_worker,
            args=(work, cfg),
            daemon=True)
        self._pregen_thread.start()

    def _start_pregen_with_defaults(self):
        """Pre-generate PNG cache using the saved cache configuration in a background thread.

        If no cache configuration has been set, prompts the user to set one first.
        """
        if self._cache_config is None:
            messagebox.showinfo(
                "Pre-gen",
                "No cache configuration set.\n\n"
                "Go to Settings → Cache Configuration and click\n"
                "\"Capture current settings\" to define the one\n"
                "display state that will be saved to disk cache.",
                parent=self.root)
            return
        # Cancel any running thread
        self._pregen_cancel = True
        if getattr(self, '_pregen_thread', None) and self._pregen_thread.is_alive():
            self._pregen_thread.join(timeout=0.1)
        self._pregen_cancel = False

        cfg = dict(self._cache_config)
        has_highres = (hasattr(self.cd, '_ccg_highres')
                       and self.cd._ccg_highres.get(self.key.nd()) is not None)
        pairs = [tuple(row) for row in self.all_inds]
        n_segs = self.n_segments
        work = []
        for inds_tuple in pairs:
            inds = np.array(inds_tuple)
            for seg in list(range(n_segs)) + [n_segs]:
                for hires in ([False, True] if has_highres else [False]):
                    work.append((inds, seg, hires))

        self._pregen_thread = threading.Thread(
            target=self._pregen_thread_worker,
            args=(work, cfg),
            daemon=True)
        self._pregen_thread.start()

    def _pregen_thread_worker(self, work, cfg):
        """Background thread: render all (pair, seg, res) PNGs using the given display config."""
        nd_key = self.key.nd()
        lo_data = getattr(self.cd, '_ccg', {}).get(nd_key) or self.ccg_data
        hi_data = getattr(self.cd, '_ccg_highres', {}).get(nd_key)

        n_total = len(work)
        n_done = 0
        n_skipped = 0
        print(f"[Pre-gen] Starting: {n_total} items (lo+hi pairs × segments)")
        for inds, seg, hires in work:
            if self._pregen_cancel:
                print(f"[Pre-gen] Cancelled after {n_done} rendered, {n_skipped} skipped")
                return
            try:
                p = self._png_path(inds, seg, _render_cfg=cfg, _hires_override=hires)
                if os.path.exists(p):
                    n_skipped += 1
                    continue
                data = (hi_data if hires and hi_data is not None else lo_data)
                if data is None:
                    continue
                self._render_png(inds, seg, highres=hires,
                                 _render_cfg=cfg, _ccg_data_override=data)
                n_done += 1
            except Exception as e:
                import traceback
                print(f"[Pre-gen] ERROR on pair ({int(inds[0])},{int(inds[1])}) seg={seg} hires={hires}: {e}")
                traceback.print_exc()
        print(f"[Pre-gen] Done: {n_done} rendered, {n_skipped} skipped (already existed)")

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
            bg='#E0E0E0', padx=4, pady=2)
        lbl_all.pack(side=tk.LEFT, padx=(2, 0))
        lbl_all.bind('<Button-1>',
                     lambda e: self._jump_to_segment(self.n_segments))
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
            lbl.bind('<Button-1>', lambda e, idx=i: self._jump_to_segment(idx))
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
                          lambda e, idx=seg_idx: self._jump_to_segment(idx))
            lbl_cust.bind('<Double-Button-1>',
                          lambda e, idx=ci: self._remove_custom_segment(idx))
            self.seg_sig_labels.append(lbl_cust)

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
        return seen

    def _session_label(self, nd_key) -> str:
        return str(nd_key.session) if nd_key.session else str(nd_key)

    def _available_type_keys(self, nd_key) -> list:
        nd_session = nd_key.session
        return [k for k in self.cd.data.keys() if k.nd().session == nd_session]

    def _type_label(self, key) -> str:
        parts = []
        if getattr(key, 'excitability', None):
            parts.append(key.excitability)
        if getattr(key, 'conn_type', None):
            ref, tgt = key.conn_type
            parts.append(f"{ref}→{tgt}")
        if getattr(key, 'epoch', None):
            parts.append(f"[{key.epoch}]")
        return ' '.join(parts) if parts else str(key)

    def _switch_key(self, new_key) -> bool:
        # Persist in-session selections to the current pointer before switching,
        # so they survive type/session changes and can be restored on return.
        if self.ccg_pointer is not None:
            self.ccg_pointer.manually_selected_inds = (
                np.array(sorted(self.selected_inds), dtype=int)
                if self.selected_inds else None
            )

        ptr = self.cd.data.get(new_key)
        if ptr is None or ptr.inds is None:
            messagebox.showwarning("Switch key", f"No data for key:\n{new_key}")
            return False
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
        self.deleted_inds = set()
        self.unselected_inds = set(map(tuple, self.all_inds)) - self.selected_inds
        self.active_norms = set()
        for var in self.norm_vars.values():
            var.set(False)
        self.active_segment_filter = None
        # Clear custom segments only when the session changes; retain them across type switches
        old_session = getattr(self.key, 'session', None) if hasattr(self, 'key') else None
        new_session = getattr(new_key, 'session', None)
        if old_session != new_session:
            self._custom_segments.clear()
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
        # Pre-generate all CCG PNGs in background for fast pair switching
        self._pregen_png_cache()

    # ------------------------------------------------------------------
    # Dropdown callbacks
    # ------------------------------------------------------------------

    def _on_session_change(self, event):
        idx = self._session_combo.current()
        if idx < 0 or idx >= len(self._nd_keys_list):
            return
        # Auto-save current session before switching away
        self._autosave_current()
        nd_key = self._nd_keys_list[idx]

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

        self._ensure_session_loaded(nd_key, on_loaded=_do_switch)

    def _on_type_change(self, event):
        idx = self._type_combo.current()
        if idx < 0 or idx >= len(self._type_keys_list):
            return
        new_key = self._type_keys_list[idx]
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
            self._rebuild_groups_menu()
        changed = (cur[0] ^ state[0]) | (cur[1] ^ state[1])
        self.refresh_lists()
        self._highlight_changed_pairs(changed)
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
            self._rebuild_groups_menu()
        changed = (cur[0] ^ state[0]) | (cur[1] ^ state[1])
        self.refresh_lists()
        self._highlight_changed_pairs(changed)
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
        self.refresh_lists()

    def _on_sort_by_tag_toggle(self):
        if self._sort_by_tag_var.get():
            self._sort_selected_var.set(False)
            self._sort_by_mean_var.set(False)
        self.refresh_lists()

    def _on_sort_by_mean_toggle(self):
        if self._sort_by_mean_var.get():
            self._sort_selected_var.set(False)
            self._sort_by_tag_var.set(False)
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

    def refresh_lists(self):
        self.unselected_list.delete(0, tk.END)
        self.selected_list.delete(0, tk.END)

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
            if gray_out is not None and gray_out(inds):
                return True
            if hide_same_shank and shank_ids is not None:
                if int(shank_ids[inds[0]]) == int(shank_ids[inds[1]]):
                    return True
            elif hide_same_channel and peak_channels is not None:
                if int(peak_channels[inds[0]]) == int(peak_channels[inds[1]]):
                    return True
            return False

        def _spec_group_sort_key(gname):
            """Sort group names numerically then alphabetically."""
            try:
                return (0, int(gname), '')
            except (ValueError, TypeError):
                return (1, 0, gname)

        speculated = getattr(self, '_speculated_groups', {})
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
                            label = f"[{inds[0]}, {inds[1]}]"
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
                             and self._sort_by_mean_var.get())
        _unsorted_avail = (sorted(self.unselected_inds,
                                  key=self._pair_mean_ccg, reverse=True)
                           if _sort_mean_active
                           else sorted(self.unselected_inds))
        for inds in _unsorted_avail:
            ref_i, tgt_i = int(inds[0]), int(inds[1])
            # Skip self-pairs (ACGs) — they should not appear in the pair list
            if ref_i == tgt_i:
                continue
            label = f"[{inds[0]}, {inds[1]}]"
            grp = self._pair_group_label(inds)
            if grp:
                label += f" {grp}"
            self.unselected_list.insert(tk.END, label)
            self._avail_list_pairs.append((inds, None))
            item_idx = self.unselected_list.size() - 1
            # Gray out pairs that don't pass the Main template criterion
            if self._main_template is not None and not self._pair_passes_main(ref_i, tgt_i, self.current_segment):
                self.unselected_list.itemconfig(item_idx, foreground='gray')
            elif _should_gray(inds):
                idx = self.unselected_list.size() - 1
                self.unselected_list.itemconfig(idx, foreground='#AAAAAA')

        # ── Deleted (spurious) section at bottom of Available ──────────
        if self.deleted_inds:
            sep_idx = self.unselected_list.size()
            self.unselected_list.insert(tk.END, "── deleted ──")
            self.unselected_list.itemconfig(sep_idx, foreground='#999999',
                                            selectforeground='#999999',
                                            selectbackground='#E8E8E8')
            self._avail_list_pairs.append(None)
            for inds in sorted(self.deleted_inds):
                label = f"[{inds[0]}, {inds[1]}]"
                self.unselected_list.insert(tk.END, label)
                idx = self.unselected_list.size() - 1
                self.unselected_list.itemconfig(idx, foreground='#BBBBBB',
                                                selectforeground='#BBBBBB',
                                                selectbackground='#F0F0F0')
                self._avail_list_pairs.append((inds, 'deleted'))

        # ── Selected list ────────────────────────────────────────────────
        self._sel_list_pairs = []   # parallel index: None = separator, inds = pair

        def _pair_group_combo(inds):
            """Sorted tuple of non-internal group names this pair belongs to."""
            return tuple(sorted(
                g for g in self._groups
                if not g.startswith('__') and inds in self._group_pairs(g)
            ))

        def _insert_sel_pair(inds):
            label = f"[{inds[0]}, {inds[1]}]"
            grp = self._pair_group_label(inds)
            if grp:
                label += f" {grp}"
            self.selected_list.insert(tk.END, label)
            self._sel_list_pairs.append(inds)
            if _should_gray(inds):
                self.selected_list.itemconfig(
                    self.selected_list.size() - 1, foreground='#AAAAAA')

        def _insert_sel_header(text, count):
            hdr_idx = self.selected_list.size()
            self.selected_list.insert(tk.END, f"── {text} ({count}) ──")
            self.selected_list.itemconfig(
                hdr_idx, foreground='#555555', selectforeground='#555555')
            self._sel_list_pairs.append(None)

        sort_group = getattr(self, '_sort_selected_var', None) and self._sort_selected_var.get()
        sort_tag   = getattr(self, '_sort_by_tag_var', None)   and self._sort_by_tag_var.get()
        sort_mean  = getattr(self, '_sort_by_mean_var', None)  and self._sort_by_mean_var.get()

        if sort_mean:
            # Sort by mean CCG value (descending) at current segment
            for inds in sorted(self.selected_inds,
                               key=self._pair_mean_ccg, reverse=True):
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
                _insert_sel_header(hdr_text, len(pairs_in_combo))
                for inds in pairs_in_combo:
                    _insert_sel_pair(inds)

        elif sort_tag:
            # Each pair appears once under every tag it belongs to
            tag_buckets: dict = _defaultdict(list)  # tag_name -> [inds, ...]
            untagged = []
            non_internal = [g for g in self._groups
                            if not g.startswith('__') and not g.startswith(_SPECIAL_PREFIX)]
            for inds in sorted(self.selected_inds):
                tags = [g for g in non_internal if inds in self._group_pairs(g)]
                if tags:
                    for t in tags:
                        tag_buckets[t].append(inds)
                else:
                    untagged.append(inds)
            for tag in sorted(tag_buckets.keys()):
                _insert_sel_header(tag, len(tag_buckets[tag]))
                for inds in tag_buckets[tag]:
                    _insert_sel_pair(inds)
            if untagged:
                _insert_sel_header('(untagged)', len(untagged))
                for inds in untagged:
                    _insert_sel_pair(inds)

        else:
            for inds in sorted(self.selected_inds):
                _insert_sel_pair(inds)

        n_spec = len({p for p in getattr(self, '_speculated_groups', {})
                      if p in self.unselected_inds})
        avail_parts = [f"Available ({len(self.unselected_inds)}"]
        if n_spec:
            avail_parts.append(f", {n_spec} predicted")
        if self.deleted_inds:
            avail_parts.append(f", {len(self.deleted_inds)} deleted")
        avail_parts.append(")")
        self._avail_label_var.set(''.join(avail_parts))
        # Show/hide the "✕ predictions" button
        if hasattr(self, '_clear_spec_btn'):
            if n_spec:
                self._clear_spec_btn.pack(side=tk.RIGHT, padx=2)
            else:
                self._clear_spec_btn.pack_forget()
        self._sel_label_var.set(f"Selected ({len(self.selected_inds)})")
        if hasattr(self, '_select_all_btn'):
            self._select_all_btn.config(
                text="Deselect All" if not self.unselected_inds else "Select All")
        self._apply_jitter_list_colors()
        self._refresh_stats()
        # Re-apply search highlights on top of any other coloring
        if getattr(self, '_search_var', None) and self._search_var.get():
            self._search_update()

    def move_to_selected(self, event=None):
        """Move the item under the cursor (double-click) or current pair (keyboard)."""
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
            inds, pred_group = avail_map[idx]
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
                self._group_add_pair(pred_group, inds)
                self.refresh_lists()
            return
        self._push_undo()
        self.unselected_inds.discard(inds)
        self.selected_inds.add(inds)
        if pred_group is not None:
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
                return
            inds = sel_map[idx]
        else:
            sorted_sel = sorted(self.selected_inds)
            if idx >= len(sorted_sel):
                return
            inds = sorted_sel[idx]
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
        if self.current_pair_idx >= len(self.all_inds):
            return
        inds = tuple(self.all_inds[self.current_pair_idx])
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
        self._select_pair_in_list(tuple(self.all_inds[next_idx]))
        self.update_plot()
        self._draw_network()

    def _select_pair_in_list(self, inds):
        """Set listbox cursor to the given pair and scroll it into view."""
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
        # In BROWSE mode curselection() is reliably set by the time ButtonRelease fires
        sel = widget.curselection()
        idx = sel[-1] if sel else widget.nearest(event.y)
        if idx < 0 or idx >= widget.size():
            return
        item = widget.get(idx)
        m = re.match(r'^\[\s*(\d+)\s*,\s*(\d+)\s*\]', item)
        if not m:
            return  # header / separator
        inds = (int(m.group(1)), int(m.group(2)))
        try:
            self.current_pair_idx = self.get_pair_index(inds)
        except (ValueError, TypeError):
            return
        # Debounce: defer the heavy update so double-click can fire first
        if self._select_after is not None:
            self.root.after_cancel(self._select_after)
        self._select_after = self.root.after(180, self._do_pair_select_update)

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

    _SEARCH_MATCH_BG   = '#fff099'   # all matches
    _SEARCH_CURRENT_BG = '#ff9900'   # current match

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
        if hasattr(self, '_search_var'):
            self._search_var.set('')
        self._search_matches = []
        self._search_cur = -1
        self._apply_search_highlights(clear=True)
        self._search_count_var.set('')

    def _apply_search_highlights(self, clear: bool = False):
        """Apply/remove search highlight colours on matched rows."""
        if clear:
            # Restore default bg for all previously highlighted rows
            for lb, i in getattr(self, '_search_matches', []):
                try:
                    lb.itemconfig(i, background='', selectbackground='')
                except tk.TclError:
                    pass
            return
        cur = self._search_cur
        for j, (lb, i) in enumerate(self._search_matches):
            bg = self._SEARCH_CURRENT_BG if j == cur else self._SEARCH_MATCH_BG
            try:
                lb.itemconfig(i, background=bg, selectbackground=bg)
            except tk.TclError:
                pass

    def _search_scroll_to_current(self):
        if not self._search_matches or self._search_cur < 0:
            return
        lb, i = self._search_matches[self._search_cur]
        lb.see(i)

    def _select_all(self):
        """Toggle between Select All and Deselect All."""
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
        if action == 'add':
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
        else:
            menu.add_command(
                label=f"Move to Available ({n})" if n > 1 else "Move to Available",
                command=lambda pp=pairs: self._ctx_move_multi_to_unselected(pp))

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
                all_in = all(p in self._group_pairs(gname) for p in pairs)
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
                    all_in = all(p in self._group_pairs(gname) for p in pairs)
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
            has_tags = p in self._pair_tags
            menu.add_command(
                label=f"{'✓ ' if has_tags else ''}Pair tags…",
                command=self._pair_tags_dialog)
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
        """Delete key: toggle current pair in/out of the Deleted section."""
        if self.current_pair_idx >= len(self.all_inds):
            return
        inds = tuple(self.all_inds[self.current_pair_idx])
        # Selected pairs cannot be deleted
        if inds in self.selected_inds:
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

    def get_pair_index(self, inds):
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
        self._update_segment_label()
        self._exit_spike_attribution_view()
        self.update_plot()

    def _jump_to_segment(self, idx):
        self.current_segment = idx
        self._update_segment_label()
        self._exit_spike_attribution_view()
        self.update_plot()

    def _update_segment_label(self):
        seg = self.current_segment
        if seg == self.n_segments:
            self.segment_var.set(_ALL_SEGS)
        elif self._is_custom_segment(seg):
            ci = self._custom_seg_index(seg)
            self.segment_var.set(self._custom_segments[ci]['name'])
        else:
            self.segment_var.set(self.segment_names[seg])
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
                lbl.config(bg=bg)
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
            lbl.config(bg=bg)

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
            ct = (f"{self.key.conn_type[0]}-{self.key.conn_type[1]}"
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
            cs = self._custom_segments[ci]
            seg_name = f"custom{ci}_{cs['name']}_{cs['t0']:.2f}_{cs['t1']:.2f}"
            seg_name = seg_name.replace(' ', '_').replace(':', '-')
        elif segment == self.n_segments:
            seg_name = _ALL_SEGS.replace(' ', '_')
        else:
            seg_name = self.segment_names[segment]
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
        j_key = '_j' if self._jitter_cache.get(
            (int(inds[0]), int(inds[1]), 'lo', self._jitter_seg(segment))) is not None else ''

        # Cache configuration determines path style
        if self._cache_config is not None:
            if _render_cfg is not None or self._display_matches_cache_config():
                # Canonical cached path — no sig encoding (only one config)
                return os.path.join(
                    self.tmp_dir,
                    f"pair_{int(inds[0])}_{int(inds[1])}_{seg_name}_{norm_key}"
                    f"{alpha_key}{res_key}{scale_key}{j_key}.png")
            else:
                # Real-time path — one file per (pair, seg, res), always overwritten
                return os.path.join(
                    self.tmp_dir,
                    f"_rt_{int(inds[0])}_{int(inds[1])}_{seg_name}{res_key}.png")

        # Legacy mode (no cache config): encode full sig state
        sig_key = ''
        sig_bits = (
            ('b' if self._sig('conv_baseline') else '') +
            ('tb' if self._sig('tailed_baseline') else '') +
            ('p' if self._sig('conv_p') else '') +
            ('c' if self._sig('conv_pc') else '') +
            ('tw' if self._sig('test_window') else '') +
            ('jp' if self._sig('jitter_p') else '') +
            ('jc' if self._sig('jitter_pc') else '') +
            ('ar' if self._acg_var_get('_acg_ref_var', False) else '') +
            ('at' if self._acg_var_get('_acg_tgt_var', False) else '') +
            (f'asr{self._acg_var_get("_acg_yscale_ref_var", 1.0):.1f}'
             if self._acg_var_get('_acg_ref_var', False) else '') +
            (f'ast{self._acg_var_get("_acg_yscale_tgt_var", 1.0):.1f}'
             if self._acg_var_get('_acg_tgt_var', False) else '') +
            ('am' if self._acg_var_get('_acg_match_ccg_var', False) else '') +
            ('nc' if not self._acg_var_get('_ccg_show_var', True) else '') +
            ('lc' if self._line_ccg_var.get() else '') +
            ('lb' if self._line_baseline_var.get() else '') +
            ('lr' if self._line_ref_var.get() else '') +
            ('lt' if self._line_tgt_var.get() else '') +
            ('lj' if self._line_jitter_var.get() else ''))
        if sig_bits:
            sig_key = f'_s{sig_bits}'
        return os.path.join(
            self.tmp_dir,
            f"pair_{int(inds[0])}_{int(inds[1])}_{seg_name}_{norm_key}"
            f"{alpha_key}{res_key}{scale_key}{j_key}{sig_key}.png")


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

    def _render_png_with_res(self, inds, segment, highres: bool, conn_strength: bool = False) -> str:
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
            path = self._render_png(inds, segment, conn_strength=conn_strength, highres=highres)
        finally:
            self._highres_mode = old_mode
            self.ccg_data = old_data
        return path

    def _render_png(self, inds, segment, conn_strength=False, highres=None,
                    _render_cfg=None, _ccg_data_override=None) -> str:
        # Helpers that read from _render_cfg when provided (thread-safe: no Tk access)
        def _rsig(name):
            _map = {'conv_baseline': '_sig_conv_baseline_var',
                    'tailed_baseline': '_sig_tailed_baseline_var',
                    'conv_p': '_sig_conv_p_var', 'conv_pc': '_sig_conv_pc_var',
                    'test_window': '_sig_test_window_var',
                    'jitter_p': '_sig_jitter_p_var', 'jitter_pc': '_sig_jitter_pc_var'}
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

        d = self._resolve_segment_data(ref, tgt, segment, highres=highres, include_pval=True, include_acg=False, _cd=cd)
        ccg_raw = d['ccg_raw']
        ccg_null_raw = d['ccg_null_raw']
        pval_arg = d['pval']
        pval_c_arg = d['pval_corrected']
        seg_label = d['seg_label']

        if _render_cfg is not None and NormalizeBy is not None:
            _norms = {n for n in NormalizeBy if n.name in (_render_cfg.get('active_norms') or [])}
        else:
            _norms = self.active_norms
        _alpha = _render_cfg.get('active_alpha', self.active_alpha) if _render_cfg else self.active_alpha

        ccg, ccg_null = apply_norms_to_ccg(
            ccg_raw, ccg_null_raw, ref, tgt, segment,
            _norms, self.neurons, self.cd.nd,
            self.key.nd(), self.n_segments, self._is_custom_segment(segment))

        norm_info = (', '.join(nm.name for nm in _norms)
                     if _norms and NormalizeBy is not None else None)

        # Infer bin_size from the actual CCG length so rendering is
        # correct even if conf.bin_size was mutated (e.g. by load_highres).
        n_bins = len(ccg)
        bin_size_eff = conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size

        # Conn-strength baseline retrieval (only when conn_strength=True)
        if conn_strength:
            method = self._conn_str_method_var.get()
            cs_key = (ref, tgt, segment, method, highres)
            if cs_key not in self._conn_strength_cache:
                self._compute_pair_conn_strength(ref, tgt, segment, highres=highres)
            cs_val, baseline_1d = self._conn_strength_cache.get(cs_key, (None, None))
        else:
            cs_val, baseline_1d = None, None

        # Jitter overlay for this pair (if computed)
        # Cache stores (j_avg, j_pval_scalar, j_pval_bins); key includes segment
        j_data = self._jitter_cache.get((ref, tgt, 'lo', self._jitter_seg(segment)))
        j_ccg_arg = j_data[0] if j_data is not None else None
        j_pval_bins_arg = j_data[2] if j_data is not None and len(j_data) > 2 else None

        # Apply significance display toggles
        show_null = ccg_null if _rsig('conv_baseline') else None
        if show_null is None and _rsig('tailed_baseline'):
            k = (ref, tgt, segment, 'tailed', highres)
            if k not in self._conn_strength_cache:
                self._compute_pair_conn_strength(ref, tgt, segment, highres=highres)
            _, bl = self._conn_strength_cache.get(k, (None, None))
            if bl is not None:
                show_null = bl
        show_pval = pval_arg if _rsig('conv_p') else None
        show_pval_c = pval_c_arg if _rsig('conv_pc') else None
        if _rsig('conv_p') and pval_arg is None and _render_cfg is None:
            is_custom = self._is_custom_segment(segment)
            is_all = (segment == self.n_segments)
            reason = ('all-segments view (no per-segment pval)' if is_all
                      else 'custom segment (pval not stored)' if is_custom
                      else 'cd.pval is None')
            print(f"[CCGReviewUI] p-value ON but unavailable for ({ref},{tgt}) seg={segment}: {reason}")
        show_j_ccg = j_ccg_arg if _rsig('jitter_p') else None
        show_j_pval = j_pval_bins_arg if _rsig('jitter_pc') else None

        # Auto-correlogram overlays (diagonal of CCG matrix)
        acg_ref = acg_tgt = None
        show_acg_ref = _racg('_acg_ref_var', False)
        show_acg_tgt = _racg('_acg_tgt_var', False)
        if show_acg_ref or show_acg_tgt:
            d_acg = self._resolve_segment_data(ref, tgt, segment, include_pval=False, include_acg=True, _cd=cd)
            if show_acg_ref:
                acg_ref = d_acg['acg_ref']
            if show_acg_tgt:
                acg_tgt = d_acg['acg_tgt']

        # In conn-strength mode: suppress pval/jitter/ACG overlays
        if conn_strength:
            show_pval = show_pval_c = show_j_ccg = show_j_pval = None
            acg_ref = acg_tgt = None

        # Use shank numbers for the figure title (display only — inds are the functional keys)
        ids = (self._shank_label(ref), self._shank_label(tgt))

        # Build segment label (with CS value when in conn-strength mode)
        seg_id_display = (f"{seg_label} [CS:{cs_val:.2f}]"
                          if conn_strength and cs_val is not None else seg_label)

        if conn_strength:
            method = self._conn_str_method_var.get()
            res_tag = 'hi' if highres else 'lo'
            png_path = os.path.join(
                self.tmp_dir,
                f"cs_{ref}_{tgt}_{segment}_{method}_{res_tag}.png"
            )
        else:
            png_path = self._png_path(inds, segment, _render_cfg=_render_cfg,
                                      _hires_override=highres)

        fig = Figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        plot_ccg.plot_ccg_panel(
            ax=ax, ccg=ccg, ids=ids, inds=inds,
            window_size=conf.duration, bin_size=bin_size_eff,
            pval=show_pval, pval_corrected=show_pval_c,
            alpha=_alpha, ccg_null=show_null,
            j_ccg=show_j_ccg, j_pval=show_j_pval,
            segment_id=seg_id_display,
            is_significant_pair=self._is_significant(ref, tgt, segment),
            min_lag=conf.min_lag if _rsig('test_window') else None,
            max_lag=conf.max_lag if _rsig('test_window') else None,
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
            conn_strength_baseline=baseline_1d,
        )
        # Same-scale y-axis override (only in normal mode)
        if not conn_strength:
            ylim = self._get_current_scale_ylim(ref, tgt)
            if ylim is not None:
                ax.set_ylim(ylim)

        fig.savefig(png_path, dpi=100, bbox_inches='tight')
        matplotlib.pyplot.close(fig)
        return png_path

    # ------------------------------------------------------------------
    # Plot update
    # ------------------------------------------------------------------

    def _autosave_current(self):
        """Silently save current session's selections + groups as 'latest'.

        Called before any operation that would overwrite self._groups or
        self.selected_inds (session switch, GUI close).
        """
        if self.ccg_pointer is None:
            print("[CCGReviewUI] autosave skipped: ccg_pointer is None (data not yet loaded)")
            return
        try:
            self._save_selection_version('latest')
            self._save_groups_export()
        except Exception as exc:
            print(f"[CCGReviewUI] autosave failed: {exc}")
            traceback.print_exc()

    def _save_groups_export(self):
        """Write the central groups_export.json with the current in-memory groups."""
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
        path = os.path.join(self._sel_save_dir, 'groups_export.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_default)

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
                                           restore_groups=False)
        except Exception as exc:
            print(f"[CCGReviewUI] failed to autoload latest: {exc}")
        if restore_groups:
            self._load_groups_from_export()

    def _load_groups_from_export(self):
        """Load groups, hotkeys, and notes from the central groups_export.json."""
        export_path = os.path.join(self._sel_save_dir, 'groups_export.json')
        if not os.path.isfile(export_path):
            # Fall back to per-session file if export doesn't exist yet
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
            self._restore_groups_from_data(data)
            n_groups = len(self._groups)
            n_pairs = sum(len(p) for sd in self._groups.values()
                          if isinstance(sd, dict) for p in sd.values())
            print(f"[CCGReviewUI] groups loaded from {export_path} "
                  f"({n_groups} groups, {n_pairs} pair-session entries)")
        except Exception as exc:
            print(f"[CCGReviewUI] failed to load groups_export.json: {exc}")

    def _restore_groups_from_data(self, data: dict):
        """Restore self._groups, _group_hotkeys, _group_notes from a dict."""
        raw_groups = data.get('groups', {})
        file_session = data.get('session', self._current_session_str())
        self._groups = {}
        for g, val in raw_groups.items():
            if isinstance(val, list):
                self._groups[g] = {file_session: set(
                    tuple(int(v) for v in p) for p in val)}
            elif isinstance(val, dict):
                self._groups[g] = {sess: set(
                    tuple(int(v) for v in p) for p in pairs)
                    for sess, pairs in val.items()}
            else:
                self._groups[g] = {}
        self._groups.setdefault(_ADMITTED_GROUP, {})
        self._group_hotkeys = data.get('hotkeys', {})
        self._group_notes = data.get('notes', {})
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
        self.deleted_inds = set()
        self.unselected_inds = set(map(tuple, self.all_inds)) - self.selected_inds
        self.refresh_lists()
        self._build_sig_chips()
        self._update_segment_label()
        self._refresh_net_shank_buttons()
        # Load persisted jitter results (try disk first, then in-memory)
        if hasattr(self.cd, 'load_jitter') and not self.cd._jitter_results:
            self.cd.load_jitter()
        self._load_jitter_from_cd()
        self.update_plot()

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
        self._conn_strength_cache.clear()
        self._clear_conn_str_png_cache()
        self.update_plot()
        self._update_conn_str_label()

    def _clear_conn_str_png_cache(self):
        """Remove cached conn-strength PNGs."""
        import glob as _glob2
        for p in _glob2.glob(os.path.join(self.tmp_dir, 'cs_*.png')):
            try:
                os.remove(p)
            except OSError:
                pass

    def _update_conn_str_label(self):
        if not hasattr(self, '_conn_str_label'):
            return
        if not self._conn_str_show_var.get():
            self._conn_str_label.config(text="CS: \u2014")
            return
        try:
            inds = self._current_inds()
            if inds is None:
                self._conn_str_label.config(text="CS: \u2014")
                return
            ref, tgt = int(inds[0]), int(inds[1])
            seg = self.current_segment
            method = self._conn_str_method_var.get()

            def _get_cs(hr):
                k = (ref, tgt, seg, method, hr)
                if k not in self._conn_strength_cache:
                    self._compute_pair_conn_strength(ref, tgt, seg, highres=hr)
                v, _ = self._conn_strength_cache.get(k, (None, None))
                return f"{float(v):.2f}" if v is not None else "n/a"

            if self._sbs_mode:
                lo = _get_cs(False)
                hi = _get_cs(True)
                self._conn_str_label.config(text=f"CS: {lo}|{hi}")
            else:
                self._conn_str_label.config(text=f"CS: {_get_cs(self._highres_mode)}")
        except Exception:
            self._conn_str_label.config(text="CS: err")

    def _current_inds(self):
        """Return current (ref, tgt) inds or None."""
        if self._focused_pair is not None:
            return np.array(self._focused_pair)
        if self.current_pair_idx < len(self.all_inds):
            return self.all_inds[self.current_pair_idx]
        return None

    def _compute_pair_conn_strength(self, ref: int, tgt: int, seg, highres: bool = False):
        """
        Compute connectivity strength for both methods and cache results.
        Returns (cs_scalar, baseline_1d) for the currently selected method.
        """
        cd = self.ccg_data
        conf = cd.conf

        # --- Retrieve raw CCG and null ---
        d = self._resolve_segment_data(ref, tgt, seg, highres=highres, include_pval=False, include_acg=False)
        ccg_raw = d['ccg_raw']
        ccg_null_raw = d['ccg_null_raw']

        ccg, ccg_null = apply_norms_to_ccg(
            ccg_raw, ccg_null_raw, ref, tgt, seg,
            self.active_norms, self.neurons, self.cd.nd,
            self.key.nd(), self.n_segments, self._is_custom_segment(seg))

        n_bins = len(ccg)
        # Recompute lag bin indices from the actual bin size of the data being
        # processed.  conf.min_lag_bin / max_lag_bin are calibrated for the
        # low-res bin size; using them unchanged for high-res CCG (more bins)
        # gives the wrong (too-narrow or negative) test window.
        bin_size_eff = conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size
        center = n_bins // 2
        lo = max(0, center + int(conf.min_lag / bin_size_eff))
        hi_bin = min(n_bins, center + int(conf.max_lag / bin_size_eff) + 1)

        # --- CONV method (convolution baseline) ---
        if ccg_null is not None:
            baseline_conv = ccg_null.copy()
            cs_conv = float(np.sum((ccg - ccg_null)[lo:hi_bin]))
        else:
            baseline_conv = np.zeros_like(ccg)
            cs_conv = float(np.sum(ccg[lo:hi_bin]))
        self._conn_strength_cache[(ref, tgt, seg, 'conv', highres)] = (cs_conv, baseline_conv)

        # --- TAILED method (ACG deconvolution) ---
        try:
            d_acg = self._resolve_segment_data(ref, tgt, seg, highres=highres, include_pval=False, include_acg=True)
            acg_ref = d_acg['acg_ref'].copy().astype(float)
            acg_tgt = d_acg['acg_tgt'].copy().astype(float)

            nspks_ref = max(float(np.sum(acg_ref)), 1.0)
            nspks_tgt = max(float(np.sum(acg_tgt)), 1.0)

            dcccg = deconv_autocorr(ccg.copy().astype(float),
                                    acg_ref, nspks_ref,
                                    acg_tgt, nspks_tgt)

            # Tail baseline: mean of bins with |t| > 11 ms from center
            center = n_bins // 2
            hw = max(1, int(11e-3 / (conf.duration / (n_bins - 1))))
            l_idx = center - hw
            r_idx = center + hw + 1
            if l_idx > 0 and r_idx < n_bins:
                tail = np.concatenate([dcccg[:l_idx], dcccg[r_idx:]])
            else:
                edge = max(1, n_bins // 10)
                tail = np.concatenate([dcccg[:edge], dcccg[-edge:]])
            baseline_val = float(np.mean(tail))
            baseline_tl = np.full(n_bins, baseline_val)
            cs_tl = float(np.sum((dcccg - baseline_val)[lo:hi_bin]))
            self._conn_strength_cache[(ref, tgt, seg, 'tailed', highres)] = (cs_tl, baseline_tl)
        except Exception as e:
            print(f"[CCGReviewUI] Tailed conn strength failed for ({ref},{tgt}): {e}")
            self._conn_strength_cache[(ref, tgt, seg, 'tailed', highres)] = (None, None)

        method = self._conn_str_method_var.get()
        return self._conn_strength_cache.get((ref, tgt, seg, method, highres), (None, None))

    def _deferred_initial_draw(self):
        # On first launch, restore groups from file (subsequent session
        # switches will keep groups intact via restore_groups=False)
        self._autoload_session_latest(restore_groups=True)
        self._ensure_session_loaded(self.key.nd(), on_loaded=self._finish_initial_draw)

    def update_plot(self):
        try:
            # Data not yet loaded — nothing to render
            if self.ccg_data is None or self.ccg_pointer is None:
                return

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
            sbs = self._sbs_mode
            cstr = self._conn_str_show_var.get()

            if self._together_pairs and len(self._together_pairs) >= 2:
                # Stacked view: one row per pinned pair, columns mirror the
                # single-pair mode (1, 2, or 4 cols depending on sbs/cstr).
                n_tog = len(self._together_pairs)
                if not sbs and not cstr:
                    n_cols = 1
                    col_titles = ['']
                elif sbs and not cstr:
                    n_cols = 2
                    col_titles = ['Lo-res', 'Hi-res']
                elif not sbs and cstr:
                    n_cols = 2
                    col_titles = ['CCG', 'Conn Strength']
                else:
                    n_cols = 4
                    col_titles = ['Lo-res', 'Hi-res', 'CS (lo)', 'CS (hi)']

                self.fig.clear()
                axes_grid = self.fig.subplots(n_tog, n_cols,
                                              squeeze=False)  # always (n_tog, n_cols)
                for row_i, tp in enumerate(self._together_pairs):
                    tp_arr = np.array(tp)
                    seg = self.current_segment
                    if not sbs and not cstr:
                        pngs = [self._get_or_render_png(tp_arr, seg)]
                    elif sbs and not cstr:
                        pngs = [
                            self._render_png_with_res(tp_arr, seg, highres=False),
                            self._render_png_with_res(tp_arr, seg, highres=True),
                        ]
                    elif not sbs and cstr:
                        pngs = [
                            self._get_or_render_png(tp_arr, seg),
                            self._render_png_with_res(tp_arr, seg,
                                highres=self._highres_mode, conn_strength=True),
                        ]
                    else:
                        pngs = [
                            self._render_png_with_res(tp_arr, seg, highres=False),
                            self._render_png_with_res(tp_arr, seg, highres=True),
                            self._render_png_with_res(tp_arr, seg, highres=False, conn_strength=True),
                            self._render_png_with_res(tp_arr, seg, highres=True,  conn_strength=True),
                        ]
                    for ax, png, col_title in zip(axes_grid[row_i], pngs, col_titles):
                        ax.imshow(mpimg.imread(png))
                        ax.axis('off')
                        if col_title:
                            ax.set_title(col_title, fontsize=8, pad=1)
                self.fig.tight_layout(pad=0.05)

            elif not sbs and not cstr:
                # 1x1 — normal single view
                png_path = self._get_or_render_png(inds, self.current_segment)
                img = mpimg.imread(png_path)
                self.fig.clear()
                ax = self.fig.add_subplot(111)
                ax.imshow(img)
                ax.axis('off')
                self.fig.tight_layout(pad=0)

            elif sbs and not cstr:
                # 1x2 — lo | hi (existing SBS logic)
                png_lo = self._render_png_with_res(inds, self.current_segment, highres=False)
                png_hi = self._render_png_with_res(inds, self.current_segment, highres=True)
                img_lo = mpimg.imread(png_lo)
                img_hi = mpimg.imread(png_hi)
                self.fig.clear()
                ax1, ax2 = self.fig.subplots(1, 2)
                ax1.imshow(img_lo); ax1.axis('off'); ax1.set_title('Lo-res', fontsize=9, pad=2)
                ax2.imshow(img_hi); ax2.axis('off'); ax2.set_title('Hi-res', fontsize=9, pad=2)
                self.fig.tight_layout(pad=0.3)

            elif not sbs and cstr:
                # 1x2 — CCG | conn str (both at current resolution)
                png_ccg = self._get_or_render_png(inds, self.current_segment)
                png_cs = self._render_png_with_res(inds, self.current_segment, highres=self._highres_mode, conn_strength=True)
                img_ccg = mpimg.imread(png_ccg)
                img_cs = mpimg.imread(png_cs)
                self.fig.clear()
                ax1, ax2 = self.fig.subplots(1, 2)
                ax1.imshow(img_ccg); ax1.axis('off'); ax1.set_title('CCG', fontsize=9, pad=2)
                ax2.imshow(img_cs); ax2.axis('off'); ax2.set_title('Conn Strength', fontsize=9, pad=2)
                self.fig.tight_layout(pad=0.3)

            else:
                # 2x2 — lo | hi (row 1); lo_cs | hi_cs (row 2)
                png_lo = self._render_png_with_res(inds, self.current_segment, highres=False)
                png_hi = self._render_png_with_res(inds, self.current_segment, highres=True)
                png_lo_cs = self._render_png_with_res(inds, self.current_segment, highres=False, conn_strength=True)
                png_hi_cs = self._render_png_with_res(inds, self.current_segment, highres=True, conn_strength=True)
                imgs = [mpimg.imread(p) for p in [png_lo, png_hi, png_lo_cs, png_hi_cs]]
                titles = ['Lo-res', 'Hi-res', 'CS (lo)', 'CS (hi)']
                self.fig.clear()
                axes = self.fig.subplots(2, 2)
                for ax, img, title in zip(axes.flat, imgs, titles):
                    ax.imshow(img); ax.axis('off'); ax.set_title(title, fontsize=9, pad=2)
                self.fig.tight_layout(pad=0.3)

            self.canvas.draw()

            self.plot_title_var.set(self.get_plot_title())
            self._update_sig_indicators(inds)
            self._update_jitter_sig_buttons()
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
            plot_waveform_on_channel(
                ref_waveform=ref_waveform,
                ref_shank=ref_shank,
                target_waveform=tgt_waveform,
                target_shank=tgt_shank,
                color=color,
                amplitude_limit=True,
                ax=self.wave_ax,
            )
        self.wave_canvas.draw()

    # ------------------------------------------------------------------
    # Time slider
    # ------------------------------------------------------------------

    def _ts_discover_themes(self):
        """Discover available Epoch objects from the session for theme switching."""
        self._ts_themes = {}
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
        if session is None and sessions:
            session = sessions[0]
        if session is None:
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

    def _on_ts_theme_change(self, _event=None):
        """Handle theme combobox selection — repopulate epoch bounds."""
        theme = self._ts_theme_var.get()
        self._ts_current_theme = theme
        # Reset handles
        self._slider_t_start = None
        self._slider_t_end = None
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

    def _ts_update_overlap_ui(self):
        """Update the label-filter dropdown for the current theme."""
        combo = getattr(self, '_ts_label_combo', None)
        row = getattr(self, '_ts_overlap_row', None)
        if combo is None or row is None:
            return
        all_labels = sorted(set(lbl for _, _, lbl in self._ts_epoch_bounds))
        theme = getattr(self, '_ts_current_theme', 'segments')
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
            self._ts_epoch_bounds = []
            for start, stop, label in zip(epoch.starts, epoch.stops, epoch.labels):
                self._ts_epoch_bounds.append((float(start), float(stop), str(label)))
            # For binary/single-label themes, use the theme name as the label
            unique_labels = set(lbl for _, _, lbl in self._ts_epoch_bounds)
            if len(unique_labels) <= 1:
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
            self._ts_total_sec = (self._ts_epoch_bounds[-1][1]
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
        all_labels = sorted(set(lbl for _, _, lbl in self._ts_epoch_bounds))
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
            all_labels = sorted(set(lbl for _, _, lbl in self._ts_epoch_bounds))
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
        if w < 20 or not self._ts_epoch_bounds:
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
    def _ts_sec_to_hms(sec: float) -> str:
        sec = int(sec)
        return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"

    def _on_time_slider_set(self):
        try:
            t0 = self._ts_hms_to_sec(self._ts_start_var.get())
            t1 = self._ts_hms_to_sec(self._ts_end_var.get())
        except (ValueError, IndexError):
            messagebox.showerror("Time window",
                                 "Invalid time format. Use HH:MM:SS.")
            return
        if t1 <= t0:
            messagebox.showerror("Time window",
                                 "End time must be after start time.")
            return
        self._slider_t_start = t0
        self._slider_t_end = t1
        # Use entered name, falling back to the time-range string
        name = getattr(self, '_ts_name_var', None)
        name = name.get().strip() if name is not None else ""
        if not name:
            name = f"{self._ts_sec_to_hms(t0)}–{self._ts_sec_to_hms(t1)}"

        # Capture brain-state filter state before computing
        theme = getattr(self, '_ts_current_theme', 'segments')
        toggles = getattr(self, '_ts_legend_toggles', {})
        filter_state = {
            'theme': theme,
            'labels': {lbl: v.get() for lbl, v in toggles.items()},
        }

        # Filter by active legend toggles (brain-state restriction)
        neurons_override = self._ts_brain_state_slice(t0, t1)
        if neurons_override is False:
            return  # user has no active labels — abort
        active_duration = (getattr(self, '_ts_brain_state_active_sec', t1 - t0)
                           if neurons_override is not None else t1 - t0)

        # Enqueue custom CCG computation (runs in background thread, same queue as jitter)
        running = 1 if self._is_task_running() else 0
        total = running + len(self._jitter_pending)
        if total >= _MAX_JITTER_QUEUE:
            messagebox.showwarning(
                "Task queue full",
                f"Queue full ({total}/{_MAX_JITTER_QUEUE}). "
                "Wait for running tasks to complete.")
            return
        self._jitter_pending.append(
            ('custom_ccg', t0, t1, name, neurons_override, active_duration, filter_state))
        self._ts_status_var.set(f"Queued: {name}")
        self._jitter_start_next()

    def _on_time_slider_clear(self):
        self._custom_segments.clear()
        if hasattr(self, '_ts_status_var'):
            self._ts_status_var.set("")
        # Reset time selection
        self._slider_t_start = None
        self._slider_t_end = None
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

    # ------------------------------------------------------------------
    # Custom-window CCG
    # ------------------------------------------------------------------

    def _ts_brain_state_slice(self, t0, t1):
        """Return a Neurons object restricted to active legend-toggle epochs,
        or None if all toggles are on (no restriction needed)."""
        toggles = getattr(self, '_ts_legend_toggles', {})
        if not toggles:
            return None
        # Check if all toggles are on — no restriction needed
        if all(v.get() for v in toggles.values()):
            return None
        # Check if NO toggles are on — abort computation entirely
        active_labels = {lbl for lbl, v in toggles.items() if v.get()}
        if not active_labels:
            messagebox.showwarning("Brain-state filter",
                                   "All epoch labels are toggled off. "
                                   "Enable at least one label to compute CCG.")
            return False  # sentinel: abort, do not compute
        if self.neurons is None:
            return None
        # Collect intervals for active labels within [t0, t1]
        # NONE label represents gap intervals between named epochs
        none_active = 'NONE' in active_labels
        real_labels = active_labels - {'NONE'}
        intervals = []
        # Collect named-epoch intervals
        named_covered = []
        for s, e, lbl in self._ts_epoch_bounds:
            if lbl in real_labels:
                s_clipped = max(s, t0)
                e_clipped = min(e, t1)
                if e_clipped > s_clipped:
                    intervals.append((s_clipped, e_clipped))
                    named_covered.append((s, e))
        # If NONE is active, also include gaps between named epochs
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
            return False  # sentinel: abort, do not compute
        self._ts_brain_state_active_sec = sum(e - s for s, e in intervals)
        # Filter spike trains to only include spikes within active intervals
        neurons = deepcopy(self.neurons)
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
                                neurons_override=None, active_duration=None):
        """Compute full CCG pipeline for a custom time window.

        Runs spike_correlations → EranConv._conv → multiple_correction
        at **both** low-res (1 ms) and high-res (0.1 ms) bin sizes so
        that Ctrl+R resolution toggle works on custom segments too.

        Returns a dict with keys: name, t0, t1, ccg, ccg_null, pval,
        pval_corrected (low-res), firing_rates, and optionally ccg_hi,
        ccg_null_hi, pval_hi, pval_corrected_hi — or None on failure.
        """
        if self.neurons is None:
            messagebox.showerror("Custom CCG", "No neuron data available.")
            return None
        try:

            neurons_slice = (neurons_override if neurons_override is not None
                             else self.neurons.time_slice(t0, t1))
            if active_duration is None:
                active_duration = t1 - t0
            conf = self.ccg_data.conf
            n_neurons = self.neurons.n_neurons
            neuron_inds = np.arange(n_neurons)
            method = conf.multiple_correction if conf.multiple_correction is not None else 'bonferroni'
            ei = getattr(self.key, 'excitability', 'E')

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
                'name':           name,
                't0':             t0,
                't1':             t1,
                'ccg':            ccg_lo,
                'ccg_null':       pred_lo,
                'pval':           pval_lo,
                'pval_corrected': pvalc_lo,
                'firing_rates':   firing_rates,
                'active_duration': active_duration,
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
        pair = tuple(inds)
        labels = []
        for gname in self._groups:
            if pair not in self._group_pairs(gname):
                continue
            if gname.startswith(_SPECIAL_PREFIX):
                labels.append('*' + gname[len(_SPECIAL_PREFIX):])
            elif not gname.startswith('__'):
                labels.append(gname)
        tag_mark = '~' if pair in self._pair_tags else ''
        group_str = f"[{','.join(labels)}]" if labels else ""
        return tag_mark + group_str

    def _toggle_pair_group(self, pair, group_name):
        if group_name not in self._groups:
            self._groups[group_name] = {}
        if pair in self._group_pairs(group_name):
            self._group_discard_pair(group_name, pair)
        else:
            self._group_add_pair(group_name, pair)
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
        cur_pairs = self._group_pairs(group_name)
        all_in = all(p in cur_pairs for p in pairs)
        action = "REMOVE" if all_in else "ADD"
        print(f"[CCGReviewUI] toggle_group: {action} {pairs} {'from' if all_in else 'to'} "
              f"{group_name!r} (session={self._current_session_str()!r}, "
              f"cur_pairs_count={len(cur_pairs)})")
        if all_in:
            for p in pairs:
                self._group_discard_pair(group_name, p)
        else:
            for p in pairs:
                self._group_add_pair(group_name, p)
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
                self._pair_tags[(ref, tgt)] = {'tags': tags, 'notes': notes}
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
                        ct_label = f"{parts[0]}\u2192{parts[1]}"
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
        self._groups[new_name] = self._groups.pop(old_name)
        if old_name in self._group_hotkeys:
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
        if not hasattr(self, '_groups_menu'):
            return
        # Remove dynamic entries (index 8+); preserve static items 0–7:
        # 0=Create, 1=Create special, 2=Manage, 3=Merge, 4=Export, 5=Import,
        # 6=sep, 7=Pair tags
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
                continue  # hide internal groups like __admitted__
            hk = self._group_hotkeys.get(gname, '')
            label = gname + (f" [{hk}]" if hk else "")
            self._groups_menu.add_command(
                label=label,
                command=lambda g=gname: self._select_group(g))
        # Special groups submenu
        if special_groups:
            special_menu = tk.Menu(self._groups_menu, tearoff=0)
            for gname in special_groups:
                display = gname[len(_SPECIAL_PREFIX):]
                n = len(self._group_pairs(gname) & current_pairs)
                special_menu.add_command(
                    label=f"{display} ({n})",
                    command=lambda g=gname: self._select_group(g))
            self._groups_menu.add_cascade(label="Special", menu=special_menu)
        # Also refresh the probe-network group toggle buttons and hotkeys bar
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
        if self.current_pair_idx >= len(self.all_inds):
            return
        for gname, k in self._group_hotkeys.items():
            if k != key_str:
                continue

            current_pair = tuple(self.all_inds[self.current_pair_idx])
            if not advance:
                # Ctrl held: only the current pair, no listbox multi-select
                highlighted = [current_pair]
            else:
                # ── Collect all highlighted pairs from both listboxes ────────
                avail_map = getattr(self, '_avail_list_pairs', None)
                highlighted = []
                for i in self.unselected_list.curselection():
                    if avail_map and i < len(avail_map):
                        entry = avail_map[i]
                        if entry is not None and entry[1] != 'deleted':
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
                if current_pair not in highlighted:
                    highlighted.append(current_pair)

            multi = len(highlighted) > 1
            changed = set()
            self._push_undo()

            for pair in highlighted:
                was_in_group = pair in self._group_pairs(gname)
                self._toggle_pair_group(pair, gname)
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
                            pair in self._group_pairs(g)
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
                self._select_pair_in_list(tuple(self.all_inds[next_idx]))
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

    def _save_selection_version(self, name: str):
        """Persist selections for ALL types in the session to a single JSON."""
        if self.ccg_pointer is None:
            raise RuntimeError("Cannot save: CCG data not yet loaded (ccg_pointer is None)")
        # Flush current type's selections to its pointer
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
            if sel is not None and len(sel) > 0:
                selections_by_type[str(tk_)] = [[int(r), int(c)]
                                                 for r, c in sorted(map(tuple, sel))]
            else:
                selections_by_type[str(tk_)] = []

        groups = {}
        # Debug: dump raw in-memory group state before serialization
        cur_sess = self._current_session_str()
        print(f"[CCGReviewUI] _save: current_session={cur_sess!r}, "
              f"n_groups={len(self._groups)}")
        for g, sd in self._groups.items():
            if isinstance(sd, dict):
                sess_with_pairs = {s: len(p) for s, p in sd.items() if p}
                if sess_with_pairs:
                    print(f"[CCGReviewUI] _save raw: {g!r} → {sess_with_pairs}")
            elif isinstance(sd, set) and sd:
                print(f"[CCGReviewUI] _save raw: {g!r} → legacy set({len(sd)})")
        for g, sessions_dict in self._groups.items():
            if isinstance(sessions_dict, set):
                # Legacy flat → wrap in current session
                groups[g] = {self._current_session_str():
                             [[int(r), int(c)] for r, c in sorted(sessions_dict)]}
            else:
                groups[g] = {sess: [[int(r), int(c)] for r, c in sorted(pairs)]
                             for sess, pairs in sessions_dict.items() if pairs}
        # Serialize pair tags: key "(ref,tgt)" → {tags, notes}
        pair_tags_ser = {}
        for (r, t), tdata in self._pair_tags.items():
            pair_tags_ser[f"{int(r)},{int(t)}"] = tdata
        deleted_ser = [[int(r), int(c)] for r, c in sorted(self.deleted_inds)]
        data = {
            'version': '3.2',
            'name': name,
            'saved_at': datetime.datetime.now().isoformat(),
            'session': getattr(self.key, 'session', 'sess'),
            'selections': selections_by_type,
            'groups': groups,
            'hotkeys': dict(self._group_hotkeys),
            'notes': dict(self._group_notes),
            'pair_tags': pair_tags_ser,
            'deleted': deleted_ser,
        }
        path = self._sel_version_path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Diagnostic: report what groups and sessions are being written
        for g, gdata in groups.items():
            if gdata:
                sess_counts = {s: len(p) for s, p in gdata.items() if p}
                if sess_counts:
                    print(f"[CCGReviewUI] save '{g}': {sess_counts}")
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_default)
        print(f"[CCGReviewUI] selection saved → {path}  ({len(groups)} groups, "
              f"{sum(len(p) for v in groups.values() for p in v.values())} total pair-session entries)")
        return path

    def _list_selection_versions(self) -> list:
        """Return list of (name, path, saved_at, is_valid) for all matching versions."""
        session_tag = getattr(self.key, 'session', 'sess')
        prefix = session_tag + '__'
        versions = []
        if not os.path.isdir(self._sel_save_dir):
            return versions
        all_files = sorted(os.listdir(self._sel_save_dir))
        print(f"[list_versions] session_tag={session_tag!r}, prefix={prefix!r}, "
              f"files={[f for f in all_files if f.endswith('.json')]}")
        for fname in all_files:
            if not fname.startswith(prefix) or not fname.endswith('.json'):
                continue
            path = os.path.join(self._sel_save_dir, fname)
            try:
                with open(path, encoding='utf-8') as f:
                    meta = json.load(f)
                versions.append((meta.get('name', fname), path,
                                 meta.get('saved_at', ''), True))
            except Exception:
                versions.append((fname, path, '⚠ corrupted', False))
        return versions

    def _load_selection_from_file(self, path: str, restore_groups: bool = True):
        """Load selection from a JSON file (v1.0 or v2.0).

        If restore_groups is False, only pair selections are loaded — groups,
        hotkeys, and notes are left untouched.  Used by autoload on session
        switch (groups are shared across sessions).
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
            # v2.0: session-level — restore each type's selections
            selections_by_type = data.get('selections', {})
            type_keys = self._available_type_keys(self.key.nd())
            total_loaded = 0
            for tk_ in type_keys:
                ptr = self.cd.data.get(tk_)
                if ptr is None:
                    continue
                pairs = selections_by_type.get(str(tk_), [])
                if pairs:
                    ptr.manually_selected_inds = np.array(
                        [[int(r), int(c)] for r, c in pairs], dtype=int)
                    total_loaded += len(pairs)
                else:
                    ptr.manually_selected_inds = None
            # Apply current type's selections to UI
            cur_sel = selections_by_type.get(str(self.key), [])
            selected = set(tuple(int(v) for v in p) for p in cur_sel)
        else:
            # v1.0 backward compat: single-type selections
            # Only apply selections if the file's key matches the current type
            file_key = data.get('key', '')
            if file_key == str(self.key):
                selected = set(tuple(int(v) for v in p)
                               for p in data.get('selected', []))
            else:
                selected = set()

        self._push_undo()
        # Check for pairs that are no longer available (epoch/CCG change)
        current_available = set(map(tuple, self.all_inds))
        missing = selected - current_available
        if missing and restore_groups:
            # Only show dialog on explicit load (not autoload)
            action = self._show_missing_pairs_dialog(missing)
            if action == 'cancel':
                return
            elif action == 'partial':
                selected = selected & current_available
            elif action == 'admit_all':
                for pair in missing:
                    self._group_add_pair(_ADMITTED_GROUP, pair)
                # Recompute current_available now that admitted pairs are added
                current_available = set(map(tuple, self.all_inds))
        elif missing:
            # Autoload: silently keep only available pairs
            selected = selected & current_available

        self.selected_inds = selected
        # Restore deleted pairs (v3.2+), keeping only currently available pairs
        raw_deleted = data.get('deleted', [])
        self.deleted_inds = set(tuple(int(v) for v in p) for p in raw_deleted
                                ) & current_available
        self.unselected_inds = current_available - selected - self.deleted_inds
        if restore_groups:
            self._restore_groups_from_data(data)
        # Pair tags (v3.1+) — always reset to avoid stale tags from previous session
        self._pair_tags = {}
        raw_tags = data.get('pair_tags', {})
        if raw_tags:
            for key_str, tdata in raw_tags.items():
                parts = key_str.split(',')
                if len(parts) == 2:
                    self._pair_tags[(int(parts[0]), int(parts[1]))] = tdata
        self.refresh_lists()
        self._draw_network()

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
        for name, path, saved_at, is_valid in versions:
            prefix = '   ' if is_valid else '⚠  '
            lb.insert(tk.END, f"{prefix}{name:28s}  {saved_at[:19]}")
            if not is_valid:
                lb.itemconfig(lb.size() - 1, foreground='#CC4444')

        def do_load():
            sel = lb.curselection()
            if not sel:
                return
            name, path, saved_at, is_valid = versions[sel[0]]
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
            name, path, saved_at, is_valid = versions[sel[0]]
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
        ttk.Label(btn_frame, text="⚠ = corrupted file (can be deleted)",
                  font=('Arial', 8), foreground='#CC4444').pack(
            side=tk.RIGHT, padx=6)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _do_save(self, name: str):
        """Core save logic: persist all types' selections + groups."""
        try:
            self._save_selection_version(name)
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Save error",
                                 f"Failed to save selection:\n{exc}")
            return

        # Count total selections across all types
        type_keys = self._available_type_keys(self.key.nd())
        total = sum(
            len(self.cd.data[tk_].manually_selected_inds)
            for tk_ in type_keys
            if self.cd.data.get(tk_) is not None
            and getattr(self.cd.data[tk_], 'manually_selected_inds', None) is not None
        )

        # Auto-export groups alongside the selection
        groups_msg = ""
        if self._groups:
            self._save_groups_export()
            groups_msg = f"\nGroups exported ({len(self._groups)} groups)."

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
        """Prompt user before closing — optionally skip autosave."""
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
        answer = messagebox.askyesnocancel(
            "Quit",
            "Save current selections before quitting?",
            default=messagebox.YES)
        if answer is None:
            # Cancel — resume normal operation
            self._closing = False
            self._start_heartbeat()
            return
        if answer:
            self._autosave_current()
        self._save_ui_state()
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

    def _ccg_cache_filename(self, seg_name: str) -> str:
        """Build a cache filename for a custom segment belonging to the current key."""
        session = str(self.key.session)
        ct = self.key.conn_type
        ct_str = "-".join(ct) if isinstance(ct, (tuple, list)) else str(ct)
        safe = re.sub(r'[^A-Za-z0-9_\-]', '_', seg_name.replace(' ', '_'))
        ts = datetime.datetime.now().strftime('%y%m%d-%H-%M')
        return f"{session}__{ct_str}__{safe}__{ts}.npz"

    def _ccg_cache_prefix(self) -> str:
        """Prefix that all cache files for the current session share."""
        return f"{str(self.key.session)}__"

    def _ts_save_custom_ccg(self):
        """Save one or more current custom segments to disk."""
        if not self._custom_segments:
            messagebox.showinfo("Save custom CCG", "No custom segments to save.")
            return

        if len(self._custom_segments) == 1:
            to_save = [0]
        else:
            # Let user pick which segments to save
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
            # Default: select all
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

        saved = []
        for ci in to_save:
            cs = self._custom_segments[ci]
            fname = self._ccg_cache_filename(cs['name'])
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
                filter_state_=np.array(
                    __import__('json').dumps(cs.get('filter_state', {}))),
                **({'firing_rates': cs['firing_rates']}
                   if 'firing_rates' in cs else {}),
            )
            for k in ('ccg_hi', 'ccg_null_hi', 'pval_hi', 'pval_corrected_hi'):
                if k in cs:
                    arrays[k] = cs[k]
            np.savez_compressed(path, **arrays)
            saved.append(cs['name'])
        if saved:
            messagebox.showinfo(
                "Saved",
                "Saved custom CCG segment(s):\n" + "\n".join(f"  • {n}" for n in saved))
            if hasattr(self, '_ts_status_var'):
                self._ts_status_var.set(f"Saved: {', '.join(saved)}")

    def _ts_load_custom_ccg(self):
        """Scan cache dir for saved custom segments and load selected ones additively."""
        prefix = self._ccg_cache_prefix()
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

        def _dur_str(dur):
            if dur != dur:  return ""          # nan
            if dur >= 60:   return f"  ⏱ {dur/60:.1f}min"
            return f"  ⏱ {dur:.0f}s"

        def _parse_meta(p):
            """Return (summary_line, [bullet_strings], path) from npz metadata."""
            base = os.path.basename(p)
            parts = base[len(prefix):].rsplit('__', 1)
            safe_name = parts[0].replace('_', ' ') if parts else base
            date_str = parts[1].replace('.npz', '') if len(parts) > 1 else ''
            bullets = []
            try:
                m = np.load(p, allow_pickle=False)
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
                has_fr = 'firing_rates' in m
                has_hi = 'ccg_hi' in m
                flags = []
                if has_fr: flags.append("firing rates")
                if has_hi: flags.append("hi-res CCG")
                if flags:
                    bullets.append(f"Contains: {', '.join(flags)}")
            except Exception:
                pass
            summary = f"{safe_name}  [{date_str}]"
            return summary, bullets

        # file_meta maps iid → path
        file_meta = {}

        def _populate(path_list):
            for iid in list(file_meta.keys()):
                if tv.exists(iid):
                    tv.delete(iid)
            file_meta.clear()
            for p in path_list:
                summary, bullets = _parse_meta(p)
                iid = tv.insert('', tk.END, text=f"☐  {summary}",
                                tags=('file',), open=False)
                file_meta[iid] = p
                for b in bullets:
                    tv.insert(iid, tk.END, text=f"    • {b}", tags=('detail',))

        _populate(paths)

        # Clicking a file row toggles its checkbox marker (selection via Treeview selection)
        # Prevent detail rows from being selected
        def _on_click(event):
            iid = tv.identify_row(event.y)
            if iid and 'detail' in tv.item(iid, 'tags'):
                # Don't select detail rows
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

        def _ok():
            sel = _selected_file_iids()
            if not sel:
                win.destroy()
                return
            existing = {(cs['name'], cs['t0'], cs['t1'])
                        for cs in self._custom_segments}
            added = []
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
                        filter_state=(json.loads(str(npz['filter_state_']))
                                      if 'filter_state_' in npz else {}),
                        **(({'firing_rates': npz['firing_rates']})
                           if 'firing_rates' in npz else {}),
                    )
                    for k in ('ccg_hi', 'ccg_null_hi', 'pval_hi', 'pval_corrected_hi'):
                        if k in npz:
                            cs[k] = npz[k]
                    key = (cs['name'], cs['t0'], cs['t1'])
                    if key not in existing:
                        self._custom_segments.append(cs)
                        existing.add(key)
                        added.append(cs['name'])
                except Exception as ex:
                    print(f"[LoadCustomCCG] failed to load {p}: {ex}")
            win.destroy()
            if added:
                self._build_sig_chips()
                self.current_segment = (self.n_segments
                                        + len(self._custom_segments))
                self._update_segment_label()
                self.update_plot()
                if hasattr(self, '_ts_status_var'):
                    self._ts_status_var.set(f"Loaded: {', '.join(added)}")

        btn_f = ttk.Frame(win)
        btn_f.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(btn_f, text="Load selected", command=_ok).pack(
            side=tk.RIGHT, padx=4)
        ttk.Button(btn_f, text="Delete selected",
                   command=_delete).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_f, text="Cancel",
                   command=win.destroy).pack(side=tk.RIGHT)


# ---------------------------------------------------------------------------

def launch_ccg_review(cd, key):
    """
    Launch CCG review UI.

    Parameters
    ----------
    cd : CCGDataset
        Dataset to review.  Call ``cd.load_highres()`` beforehand to enable
        the resolution toggle button.
    key : Key
        Key identifying which CCGPointer to review.

    Examples
    --------
    >>> ui = launch_ccg_review(cd, key)
    >>> cd.load_highres(); ui = launch_ccg_review(cd, key)
    """
    ui = CCGReviewUI(cd, key)
    ui.run()
    return ui
