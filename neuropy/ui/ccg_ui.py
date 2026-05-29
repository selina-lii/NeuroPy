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
  Ctrl+E        toggle waveforms sub-panel
  Ctrl+S        save selection + export groups
  1..0          assign / jump to group 1–10
"""

import io
import random
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

# winfo_containing raises KeyError('popdown') when a ttk Combobox dropdown is
# open during scroll. Patch captures the original C-level call; sentinel on
# tkinter module prevents double-wrap on autoreload.
try:
    import tkinter as _tk_patch
    if not getattr(_tk_patch, '_ccg_winfo_patched', False):
        _orig_winfo_containing = _tk_patch.Misc.winfo_containing
        def _winfo_containing_safe(self, rootX, rootY, displayof=0,
                                   _orig=_orig_winfo_containing):
            try:
                return _orig(self, rootX, rootY, displayof)
            except (KeyError, Exception):
                return None
        _tk_patch.Misc.winfo_containing = _winfo_containing_safe
        _tk_patch._ccg_winfo_patched = True
        del _orig_winfo_containing
except Exception:
    pass
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
from neuropy.analyses.jitter import compute_jbsi, JitterConfig
import copy as _copy
from neuropy.analyses import correlations
from neuropy.analyses.correlations import np_spike_correlations, sim_generate_train
from neuropy.analyses import custom_ccg as _custom_ccg_mod
from neuropy.analyses.ms_connectivity import (
    NormalizeBy, apply_norms_to_ccg, compute_pair_conn_strength_1d,
    EranConv, CCGConfig, _CCG_RESOLUTION,
)
from neuropy.analyses.neurons_dataset import Key
from neuropy.core.neurons import Neurons
from neuropy.core.epoch import Epoch
from neuropy.ui.probe_network import NetworkPanel
from neuropy.ui.custom_ccg_manager import CustomCCGManager
from neuropy.ui.time_slider import TimeSliderPanel
from neuropy.ui.ccg_renderer import CCGRenderEngine
from neuropy.ui.pair_selection_panel import LeftPanelContainer, SelectionData
from neuropy.ui.ccg_mainview import CenterPanelContainer
from neuropy.ui.jitter_ui import JitterController, JitterWorker, JitterQueueDialog
from neuropy.ui.utils import (BackgroundTaskRunner, GenericUI, LRUCache,
                              get_png_filename, purge_png_dir,
                              _SEPARATOR_ROW, is_separator_row,
                              _SPECIAL_PREFIX, is_special_group,
                              _admitted_pairs, _admit_pair,
                              ArrowScroller)
from neuropy.ui.selection_migration import SelectionMigration
from neuropy.ui.dialogs import (
    MergeGroupsDialog, ManageGroupsDialog,
    SimulationDialog, SettingsDialog, ExportOptionsDialog,
)

# Sentinel value for the virtual "All segments" view
_ALL_SEGS = "All"
# Virtual session entry: union of all sessions (lazy-loaded per group tag).
_ALL_SESSION_MARKER = object()
# TODO: make this configurable per dataset — currently hardcoded for 8-session NSD/SD design
_SESS_PER_BLOCK = 8   # sessions 1..N: idx<=_SESS_PER_BLOCK → NSD, idx>_SESS_PER_BLOCK → SD
# maximum number of queued jitters (including the currently running one)
_MAX_JITTER_QUEUE = 50



class SettingsManager:
    """Settings persistence and font-size logic for CCGReviewUI."""

    _CACHE_CONFIG_ATTRS = (
        '_sig_conv_p', '_sig_conv_pc', '_sig_test_window',
        '_sig_jitter_pc',
        '_conn_str_method', '_conn_str_show', '_adaptive_tw',
        '_acg_ref', '_acg_tgt',
        '_acg_deconv_ref', '_acg_deconv_tgt',
        '_extend_enable', '_extend_ms', '_extend_bin_ms',
        '_peak_wf',
        '_acg_yscale_ref', '_acg_yscale_tgt',
        '_acg_match_ccg', '_ccg_show', '_baseline_show',
        '_line_ccg', '_line_baseline',
        '_line_ref', '_line_tgt', '_line_peak_wf', '_line_jitter',
    )

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        raw = self.load_ui_state()
        ui._ui_state_cache = raw
        ui._settings: dict = {**{'max_show_together': 5}, **raw.get('settings', {})}
        ui._cache_config = raw.get('cache_config', None)
        ui._auto_pregen_enabled = raw.get('auto_pregen_enabled', False)

    def load_ui_state(self) -> dict:
        try:
            with open(self._ui._ui_state_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def save_ui_state(self):
        ui = self._ui
        try:
            key_dict = {f: getattr(ui.key, f) for f in ui.key.__dataclass_fields__}
            if key_dict.get('conn_type') is not None:
                key_dict['conn_type'] = list(key_dict['conn_type'])

            active_norms = sorted(nm.name for nm in getattr(ui, 'active_norms', set()))

            sort_sel  = getattr(ui, '_sort_selected_var', None)
            sort_tag  = getattr(ui, '_sort_by_tag_var',   None)
            sort_mean = getattr(ui, '_sort_by_mean_var',  None)
            sort_minp = getattr(ui, '_sort_by_min_p_var', None)

            pair_scale = getattr(ui, '_pair_scale', None)
            sess_scale = getattr(ui, '_sess_scale', None)

            sash_pos = None
            pane = getattr(ui, '_pair_list_pane', None)
            if pane is not None:
                try:
                    sash_pos = pane.sashpos(0)
                except Exception:
                    pass

            state = {
                'panels':              {n: v.get() for n, v in ui._panel_vars.items()},
                'settings':            ui._settings,
                'cache_config':        ui._cache_config,
                'auto_pregen_enabled': ui._auto_pregen_enabled,
                'last_key':            key_dict,
                'active_norms':        active_norms,
                'conn_str_method':     (ui.center_container.baseline_panel._conn_str_method.get()
                                        if hasattr(ui, '_conn_str_method') else 'conv'),
                'conn_str_metric':     (ui.center_container.cs_panel._conn_str_metric.get()
                                        if hasattr(ui, '_conn_str_metric') else 'STG'),
                'jitter_run_lo':       (ui.center_container.jitter_panel._jitter_run_lo.get()
                                        if hasattr(ui, '_jitter_run_lo') else True),
                'jitter_run_hi':       (ui.center_container.jitter_panel._jitter_run_hi.get()
                                        if hasattr(ui, '_jitter_run_hi') else False),
                'current_segment':     ui.current_segment,
                'sort_selected':       sort_sel.get()  if sort_sel  else False,
                'sort_by_tag':         sort_tag.get()  if sort_tag  else False,
                'sort_by_mean':        sort_mean.get() if sort_mean else False,
                'sort_by_min_p':       sort_minp.get() if sort_minp else False,
                'pair_scale':          pair_scale.get() if pair_scale else False,
                'sess_scale':          sess_scale.get() if sess_scale else False,
                'sash_pos':            sash_pos,
                'display_vars':        ui._settings_mgr._current_display_config(),
                'highres_mode':        bool(getattr(ui, '_highres_mode', False)),
                'lo_hi_mode':            bool(getattr(ui, '_lo_hi_mode', False)),
                'loaded_custom_ccgs':  [
                    cs.get('src_path')
                    for lst in getattr(ui, '_custom_segments_by_session', {}).values()
                    for cs in lst
                    if isinstance(cs, dict) and cs.get('src_path')
                ],
            }
            with open(ui._ui_state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def min_font_size(self) -> int:
        return max(6, min(32, int(self._ui._settings.get('min_font_size', 9) or 9)))

    def apply_min_font_size(self):
        ui = self._ui
        import tkinter.font as _tkfont

        try:
            min_fs = max(6, min(32, int(ui._settings.get('min_font_size', 9) or 9)))
        except Exception:
            min_fs = 9

        for lb_name in ('unselected_list', 'selected_list'):
            lb = getattr(ui, lb_name, None)
            if lb is None:
                continue
            try:
                f = _tkfont.Font(font=lb.cget('font'))
                if f.cget('size') < min_fs:
                    f.configure(size=min_fs)
                lb.config(font=f)
            except Exception:
                pass

        lbl = getattr(ui, '_plot_title_label', None)
        if lbl is not None:
            try:
                f = _tkfont.Font(font=lbl.cget('font'))
                if f.cget('size') < min_fs:
                    f.configure(size=min_fs)
                lbl.config(font=f)
            except Exception:
                pass

        try:
            bar = getattr(ui, '_hotkeys_bar', None)
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

        try:
            ui.center_container.seg_chips_panel.rebuild()
        except Exception:
            pass

        try:
            ui.time_slider._update_legend()
        except Exception:
            pass

        try:
            sp = getattr(ui, '_stats_panel', None)
            if sp is not None and sp.root.winfo_exists():
                txt = getattr(sp, '_result_text', None)
                if txt is not None:
                    f = _tkfont.Font(font=txt.cget('font'))
                    if f.cget('size') < min_fs:
                        f.configure(size=max(9, min_fs))
                        txt.config(font=f)
        except Exception:
            pass

class ExportManager:
    """Export methods for CCGReviewUI."""

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui

    def _export_current_view(self):
        """Export the currently displayed CCG view (including stacked/sbs) to PNG."""
        ui = self._ui
        fmt = 'png'
        if ui.ccg_data is None:
            messagebox.showwarning("Export", "No plot to export.")
            return
        selected_pairs = ui._selected_pairs_from_lists()

        def _strip_any_session_pair(p):
            if p is None:
                return None
            if getattr(ui, '_session_any_mode', False) and len(p) == 3:
                nk = ui._nd_key_for_session_str(str(p[0]))
                if nk is not None:
                    ckey = ui._type_key_for_nd(nk)
                    if ckey is not None:
                        ui._bind_context_to_type_key(ckey)
                return int(p[1]), int(p[2])
            return int(p[0]), int(p[1])

        if getattr(ui, '_session_any_mode', False):
            selected_pairs = [_strip_any_session_pair(p) for p in selected_pairs]

        # If user didn't explicitly select rows, default to current pair for "Export current".
        # (Also drives preview in the dialog.)
        preview_pair = selected_pairs[0] if selected_pairs else ui._selected_pair_from_lists()
        preview_pair = _strip_any_session_pair(preview_pair)
        if preview_pair is None:
            inds = ui._current_inds()
            if inds is not None:
                preview_pair = (int(inds[0]), int(inds[1]))
        opt = ExportOptionsDialog.show(ui, fmt=fmt, preview_pair=preview_pair, selected_pairs=selected_pairs)
        if opt is None:
            return
        # Multi-export actions should go straight to a folder picker (no save-as).
        if opt.get('_action') in ('all', 'bookmarked', 'groups', 'all_groups', 'all_sessions_selected'):
            ui._export_mgr._export_pairs_from_opt(fmt=fmt, opt=opt)
            return
        # Suggest a filename from session/type/shank/pair/segment
        inds = ui._current_inds()
        if inds is not None:
            ref, tgt = int(inds[0]), int(inds[1])
        else:
            ref = tgt = None
        seg = ui.current_segment
        if seg == ui.n_segments:
            seg_tag = "All"
        elif ui._is_custom_segment(seg):
            seg_tag = "custom"
        else:
            seg_tag = f"seg{seg}"
        sess = str(getattr(getattr(ui.key, 'nd', lambda: ui.key)(), 'session', None) or getattr(ui.key, 'session', 'sess'))
        exc = getattr(ui.key, 'excitability', None)
        ct = getattr(ui.key, 'conn_type', None)
        if isinstance(ct, (tuple, list)) and len(ct) >= 2:
            _map = {'pyr': 'PYR', 'inter': 'INT', 'int': 'INT'}
            a = _map.get(str(ct[0]).lower(), str(ct[0]).upper())
            b = _map.get(str(ct[1]).lower(), str(ct[1]).upper())
            ct_str = f"{a}-{b}"
        else:
            ct_str = str(ct) if ct is not None else "any"
        type_str = f"{exc}_{ct_str}" if exc is not None else ct_str
        sh = ''
        shank_ids = getattr(ui.neurons, 'shank_ids', None)
        if shank_ids is not None and ref is not None and tgt is not None:
            try:
                sh = f"_sh{int(shank_ids[ref])}-{int(shank_ids[tgt])}"
            except Exception:
                sh = ''
        base = (f"{sess}_{type_str}{sh}_ccg_{ref}_{tgt}_{seg_tag}"
                if ref is not None else f"{sess}_{type_str}_ccg_{seg_tag}")
        path = filedialog.asksaveasfilename(
            parent=ui.root,
            title=f"Export as {fmt.upper()}",
            defaultextension=f".{fmt}",
            initialfile=f"{base}.{fmt}",
            filetypes=[(fmt.upper(), f"*.{fmt}"), ("All files", "*.*")],
        )

        if not path:
            return
        try:
            ui._export_mgr._export_one_view_to_path(path=path, fmt=fmt, opt=opt)
        except Exception as exc:
            messagebox.showerror("Export failed", f"Could not export:\n\n{exc}")

    def _export_one_view_to_path(self, path: str, fmt: str, opt: dict):
        """Export current view to a specific file path using overrides."""
        ui = self._ui
        old = getattr(ui, '_export_overrides', None)
        ui._export_overrides = opt
        try:
            if (not getattr(ui, '_stacked_segments', None) and
                    not getattr(ui, '_together_pairs', None) and
                    not getattr(ui, '_lo_hi_mode', False)):
                # Single-pair view: render directly to path, bypassing the viewer cache
                inds = ui.all_inds[ui.current_pair_idx]
                seg  = int(ui.current_segment)
                hr   = bool(getattr(ui, '_highres_mode', False))
                ctx  = ui._render_engine.build_context(inds, seg, hr, None, None)
                dpi  = 300 if fmt == 'png' else None
                ui._render_engine.write_png(ctx, path, dpi=dpi)
            else:
                # Multi-view (stacked/SBS): save composite figure as-is
                ui.update_plot()
                ui.canvas.draw()
                ui.fig.savefig(path, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
        finally:
            ui._export_overrides = old

    def _export_pairs_with_handles(self, fmt: str, opt: dict,
                                     items: list[tuple], folder: str) -> None:
        """Core export loop: render each (tk_, ptr, ref, tgt) into *folder*.

        *items* is a list of 4-tuples (tk_, ptr, ref, tgt) where tk_/ptr are
        the exact Key/pointer objects from cd.data — no string lookup needed.
        """
        ui = self._ui
        export_segs = (opt or {}).get('export_segments', None)
        if not export_segs:
            export_segs = ["Current"]
        # normalize
        export_segs = [str(s) for s in export_segs if str(s).strip()] or ["Current"]
        subfolder_by = list((opt or {}).get('subfolder_by') or [])

        _ct_map = {'pyr': 'PYR', 'inter': 'INT', 'int': 'INT'}

        do_lores   = bool((opt or {}).get('export_lores', True))
        do_hires   = bool((opt or {}).get('export_hires', False))
        # Fall back to lo-res if neither is selected
        if not do_lores and not do_hires:
            do_lores = True
        _cs_by_sess = getattr(ui, '_custom_segments_by_session', None) or {}

        old_state = {
            'key': ui.key,
            'ccg_ptr': ui.ccg_ptr,
            'ccg_data': ui.ccg_data,
            'neurons': ui.neurons,
            'n_segments': getattr(ui, 'n_segments', 0),
            'segment_names': ui.ccg_ptr.segment_names,
            'current_pair_idx': int(getattr(ui, 'current_pair_idx', 0)),
            'current_segment': int(getattr(ui, 'current_segment', 0)),
            '_custom_segments': getattr(ui, '_custom_segments', []),
            '_highres_mode': getattr(ui, '_highres_mode', False),
        }

        n_ok = 0
        n_fail = 0
        fail_msgs = []
        try:
            for tk_, ptr, ref, tgt in items:
                # Point UI at this key/pointer directly — no lookup required.
                ui.key = tk_
                ui.ccg_ptr = ptr
                nd_key = tk_.nd()
                sess = str(getattr(tk_, 'session', getattr(nd_key, 'session', '')))
                ui.neurons = (ui.cd.nd.data.get(nd_key)
                                if getattr(ui.cd, 'nd', None) is not None else None)
                ui.n_segments = ptr.n_segments
                # Swap custom segments to this session's list so custom indices resolve correctly
                ui._custom_segments = _cs_by_sess.get(sess, []) or []
                # Resolve segment indices for this pointer.
                # - Current: use UI's current_segment (clamped to this ptr)
                # - All: use ptr.n_segments sentinel
                # - Named: first try regular segment label, then try custom segment name (silent skip)
                _reg_labels = list(ptr.edge_times['label'].values)
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
                    # Try regular segment label
                    try:
                        seg_indices.append(int(_reg_labels.index(export_seg)))
                        continue
                    except ValueError:
                        pass
                    # Try custom segment name for this session (silent skip if absent)
                    _ci = next((i for i, _cs in enumerate(ui._custom_segments)
                                if isinstance(_cs, dict) and _cs.get('name') == export_seg), None)
                    if _ci is not None:
                        seg_indices.append(int(ptr.n_segments) + 1 + _ci)
                    # else: silently skip — this session doesn't have that custom segment
                # de-dupe while preserving order
                _seen_seg = set()
                seg_indices = [s for s in seg_indices if not (s in _seen_seg or _seen_seg.add(s))]
                if not seg_indices:
                    continue
                if (getattr(ui, '_highres_mode', False)
                        and hasattr(ui.cd, '_ccg_highres')
                        and ui.cd._ccg_highres.get(nd_key) is not None):
                    ui.ccg_data = ui.cd._ccg_highres.get(nd_key)
                else:
                    ui.ccg_data = (ui.cd._ccg.get(nd_key)
                                     if getattr(ui.cd, '_ccg', None) else ui.ccg_data)
                try:
                    ui.current_pair_idx = ui.get_pair_index((ref, tgt))
                except Exception:
                    ui.current_pair_idx = 0

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
                shank_ids = getattr(ui.neurons, 'shank_ids', None)
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

                def _has_hires():
                    return (hasattr(ui.cd, '_ccg_highres')
                            and ui.cd._ccg_highres.get(nd_key) is not None)

                def _render_one(seg, highres: bool, path: str):
                    ui._highres_mode = highres
                    if highres and _has_hires():
                        ui.ccg_data = ui.cd._ccg_highres.get(nd_key)
                    else:
                        ui.ccg_data = (ui.cd._ccg.get(nd_key)
                                         if getattr(ui.cd, '_ccg', None) else ui.ccg_data)
                    old_eo = getattr(ui, '_export_overrides', None)
                    ui._export_overrides = opt
                    try:
                        ctx = ui._render_engine.build_context(
                            np.array([ref, tgt]), seg, highres, None, None)
                        dpi = 300 if fmt == 'png' else None
                        ui._render_engine.write_png(ctx, path, dpi=dpi)
                    finally:
                        ui._export_overrides = old_eo

                # Export one file per selected segment (render directly; bypass viewer cache)
                for seg_idx in seg_indices:
                    ui.current_segment = int(seg_idx)
                    seg = int(ui.current_segment)
                    if seg == int(ptr.n_segments):
                        seg_tag = "All"
                    elif ui._is_custom_segment(seg):
                        ci_tag = ui._custom_seg_index(seg)
                        cs_name = (ui._custom_segments[ci_tag].get('name', f'custom{ci_tag}')
                                   if ci_tag < len(ui._custom_segments) else f'custom{ci_tag}')
                        seg_tag = re.sub(r'[^A-Za-z0-9_\-]', '_', str(cs_name))
                    else:
                        seg_tag = f"seg{seg}"
                    base_name = f"{sess}_{type_str}{sh}_ccg_{ref}_{tgt}_{seg_tag}"
                    if not bool((opt or {}).get('title_show_norm_details', True)):
                        _active_norms = sorted(nm.name for nm in getattr(ui, 'active_norms', set()))
                        if _active_norms:
                            base_name += '_' + '_'.join(_active_norms)
                    try:
                        if do_lores and do_hires and _has_hires():
                            # Render both resolutions and combine side by side
                            import tempfile as _tf
                            with _tf.NamedTemporaryFile(suffix='.png', delete=False) as _tlo:
                                lo_path = _tlo.name
                            with _tf.NamedTemporaryFile(suffix='.png', delete=False) as _thi:
                                hi_path = _thi.name
                            try:
                                _render_one(seg, False, lo_path)
                                _render_one(seg, True,  hi_path)
                                _lo_img = Image.open(lo_path)
                                _hi_img = Image.open(hi_path)
                                _h = max(_lo_img.height, _hi_img.height)
                                _combined = Image.new('RGB',
                                                      (_lo_img.width + _hi_img.width, _h),
                                                      (255, 255, 255))
                                _combined.paste(_lo_img, (0, 0))
                                _combined.paste(_hi_img, (_lo_img.width, 0))
                                out_path = os.path.join(base_dir, f"{base_name}_lohires.{fmt}")
                                _combined.save(out_path, dpi=(300, 300))
                            finally:
                                for _p in (lo_path, hi_path):
                                    try:
                                        os.remove(_p)
                                    except OSError:
                                        pass
                        elif do_hires and _has_hires():
                            out_path = os.path.join(base_dir, f"{base_name}_hires.{fmt}")
                            _render_one(seg, True, out_path)
                        else:
                            out_path = os.path.join(base_dir, f"{base_name}.{fmt}")
                            _render_one(seg, False, out_path)
                        n_ok += 1
                    except Exception as ex:
                        n_fail += 1
                        fail_msgs.append(f"({ref},{tgt}) seg={seg_tag}: {ex}")
        finally:
            ui.key = old_state['key']
            ui.ccg_ptr = old_state['ccg_ptr']
            ui.ccg_data = old_state['ccg_data']
            ui.neurons = old_state['neurons']
            ui.n_segments = old_state['n_segments']
            ui.current_pair_idx = old_state['current_pair_idx']
            ui.current_segment = old_state.get('current_segment', ui.current_segment)
            ui._custom_segments = old_state.get('_custom_segments', ui._custom_segments)
            ui._highres_mode = old_state.get('_highres_mode', ui._highres_mode)
            try:
                ui.update_plot()
            except Exception:
                pass

        if n_fail == 0:
            messagebox.showinfo("Export", f"Exported {n_ok} file(s) to:\n\n{folder}",
                                parent=ui.root)
        else:
            msg = f"Exported {n_ok} file(s) to:\n\n{folder}\n\nFailed: {n_fail}"
            if fail_msgs:
                msg += "\n\n" + "\n".join(fail_msgs[:12])
                if len(fail_msgs) > 12:
                    msg += f"\n… ({len(fail_msgs) - 12} more)"
            messagebox.showwarning("Export", msg, parent=ui.root)

    def _export_all_selected_pairs(self, fmt: str, opt: dict):
        """Export pairs listed in opt['_selected_pairs'] (current-session subset)."""
        ui = self._ui
        pairs_in = list(opt.get('_selected_pairs') or [])
        if not pairs_in:
            messagebox.showinfo("Export", "No pairs selected.")
            return
        folder = filedialog.askdirectory(
            parent=ui.root, title=f"Export {len(pairs_in)} views to folder")
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
            items.append((ui.key, ui.ccg_ptr, ref, tgt))

        ui._export_mgr._export_pairs_with_handles(fmt=fmt, opt=opt, items=items, folder=folder)

    def _collect_all_sessions_selected(self) -> list[tuple]:
        """Return (tk_, ptr, ref, tgt) for every selected pair in every session/type."""
        ui = self._ui
        # Flush live selection into the current pointer before iterating.
        if ui.ccg_ptr is not None:
            if getattr(ui, '_session_any_mode', False):
                ui._sess_mgr._flush_any_selections_to_pointers()
            else:
                ui.ccg_ptr.selected_inds = (
                    np.array(sorted(ui.selected_inds), dtype=int)
                    if getattr(ui, 'selected_inds', None) else None
                )
        items = []
        for tk_, ptr in ui.cd.data.items():
            sel = getattr(ptr, 'selected_inds', None)
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
        ui = self._ui
        m: dict[tuple, list] = {}
        for tk_, ptr in ui.cd.data.items():
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
        ui = self._ui
        action = opt.get('_action')

        if action == 'all_sessions_selected':
            # All selected pairs across every session/type — handles come straight
            # from cd.data, no lookup needed.
            items = ui._export_mgr._collect_all_sessions_selected()

        elif action == 'all':
            # Pairs explicitly highlighted in the current-session listbox.
            raw = list(opt.get('_selected_pairs') or [])
            items = []
            for it in raw:
                pair = tuple(it['pair']) if isinstance(it, dict) else tuple(it)
                try:
                    items.append((ui.key, ui.ccg_ptr,
                                  int(pair[0]), int(pair[1])))
                except Exception:
                    pass

        elif action == 'bookmarked':
            # Bookmarks are per-session (current session only).
            items = []
            for p in sorted(getattr(ui, '_bookmarked_pairs', set()) or set()):
                try:
                    items.append((ui.key, ui.ccg_ptr,
                                  int(p[0]), int(p[1])))
                except Exception:
                    pass

        elif action in ('groups', 'all_groups'):
            # Group data is stored as {session_str: [[ref,tgt], ...]}.
            # IMPORTANT: we must NOT “guess” a handle for (session, pair). Instead,
            # we scan each (tk_, ptr).inds2 and include it only if that pair is
            # explicitly in the chosen group(s) for that session.
            if action == 'all_groups':
                gnames = [g for g in (ui._sel_data._groups or {}) if g and not str(g).startswith('__')]
            else:
                gnames = list(opt.get('_selected_groups') or [])
                if not gnames:
                    messagebox.showinfo("Export", "No groups selected.")
                    return

            # Build desired pairs per session from group definitions
            want_by_sess: dict[str, set[tuple[int, int]]] = {}
            for g in gnames:
                try:
                    sd = ui._sel_data._groups.get(g, {})
                    pairs_by_sess = sd if isinstance(sd, dict) else {
                        ui._current_session_str(): list(sd)}
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
            for tk_, ptr in ui.cd.data.items():
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
                    parent=ui.root
                )
        else:
            items = []

        if not items:
            messagebox.showinfo("Export", "No pairs to export.")
            return

        folder = filedialog.askdirectory(
            parent=ui.root, title=f"Export {len(items)} view(s) to folder")
        if not folder:
            return

        ui._export_mgr._export_pairs_with_handles(fmt=fmt, opt=opt, items=items, folder=folder)

class GroupManager:
    """Group management, hotkeys, and related UI for CCGReviewUI."""

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui

    def _ensure_group_registered(self, name: str) -> int:
        """Return the int ID for group name, creating an entry if new."""
        ui = self._ui
        for gid, g in ui._sel_data._group_registry.items():
            if g['name'] == name:
                return gid
        gid = ui._sel_data._next_group_id
        ui._sel_data._group_registry[gid] = {
            'name': name,
            'hotkey': ui._sel_data._group_hotkeys.get(name),
            'notes': ui._sel_data._group_notes.get(name, ''),
        }
        ui._sel_data._next_group_id += 1
        return gid

    def _group_id_for(self, name: str) -> int | None:
        """Return int ID for group name, or None if not registered."""
        ui = self._ui
        for gid, g in ui._sel_data._group_registry.items():
            if g['name'] == name:
                return gid
        return None

    def _sync_registry_from_groups(self):
        """Ensure every group in ui._sel_data._groups has a registry entry."""
        ui = self._ui
        for name in list(ui._sel_data._groups.keys()):
            ui._group_mgr._ensure_group_registered(name)
        # Sync hotkeys/notes into registry
        for gid, g in ui._sel_data._group_registry.items():
            name = g['name']
            g['hotkey'] = ui._sel_data._group_hotkeys.get(name)
            g['notes'] = ui._sel_data._group_notes.get(name, '')

    def _group_pairs(self, gname, session=None):
        """Return pairs set for group in the given session (default: current)."""
        ui = self._ui
        g = ui._sel_data._groups.get(gname, {})
        if isinstance(g, set):
            return g  # legacy flat format
        sess = session or ui._current_session_str()
        return g.get(sess, set())

    def _group_pairs_all_sessions(self, gname):
        """Return all pairs across all sessions for a group."""
        ui = self._ui
        g = ui._sel_data._groups.get(gname, {})
        if isinstance(g, set):
            return g
        all_pairs = set()
        for pairs in g.values():
            all_pairs |= pairs
        return all_pairs

    def _group_add_pair(self, gname, pair, session=None):
        ui = self._ui
        sess = session or ui._current_session_str()
        ui._sel_data._groups.setdefault(gname, {}).setdefault(sess, set()).add(pair)

    def _group_discard_pair(self, gname, pair, session=None):
        ui = self._ui
        sess = session or ui._current_session_str()
        print(f"[CCGReviewUI] group_discard: {gname!r} -= {pair} @ session={sess!r}")
        g = ui._sel_data._groups.get(gname, {})
        if isinstance(g, set):
            g.discard(pair)
        elif sess in g:
            g[sess].discard(pair)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _any_group_header_names(self) -> list[str]:
        """Sorted user group names (tags) for ``any``-mode list sections."""
        ui = self._ui

        def _sort_key(n: str):
            try:
                return (0, int(n), '')
            except (ValueError, TypeError):
                return (1, 0, n)

        names = [
            g for g in ui._sel_data._groups
            if not g.startswith('__')
        ]
        return sorted(names, key=_sort_key)

    def _any_triples_in_group(self, gname: str) -> set[tuple]:
        """All (session, ref, tgt) in *gname* for the current connection type."""
        ui = self._ui
        lbl = ui._type_label(ui.key)
        out: set[tuple] = set()
        for k in ui.cd.data.keys():
            if ui._type_label(k) != lbl:
                continue
            sess = str(k.session)
            valid = ui._all_inds_set_for_ptr(ui.cd.data.get(k))
            if not valid:
                continue
            for pair in ui._group_pairs(gname, session=sess):
                r, t = int(pair[0]), int(pair[1])
                if (r, t) in valid:
                    out.add((sess, r, t))
        return out

    def _any_nd_keys_for_group(self, gname: str) -> list:
        """Neuron-dataset keys that have ≥1 pair in *gname* (current type)."""
        ui = self._ui
        lbl = ui._type_label(ui.key)
        seen, seen_id = [], set()
        for nk in ui._real_nd_keys_ordered():
            ckey = ui._type_key_for_nd(nk)
            if ckey is None or ui._type_label(ckey) != lbl:
                continue
            sess = str(ckey.session)
            ptr = ui.cd.data.get(ckey)
            valid = ui._all_inds_set_for_ptr(ptr)
            if any((int(a), int(b)) in valid
                   for a, b in ui._group_pairs(gname, session=sess)):
                nid = id(nk)
                if nid not in seen_id:
                    seen.append(nk)
                    seen_id.add(nid)
        return seen

    def _any_iter_pairs_for_group(self, gname: str):
        """Yield ``(ckey, r, t)`` for *gname* by scanning sessions (expanded tag only)."""
        ui = self._ui
        if gname not in ui._any_expanded_group_tags:
            return
        lbl = ui._type_label(ui.key)
        dead = ui.deleted_inds
        for nk in ui._real_nd_keys_ordered():
            ckey = ui._type_key_for_nd(nk)
            if ckey is None or ui._type_label(ckey) != lbl:
                continue
            sess = str(ckey.session)
            ptr = ui.cd.data.get(ckey)
            valid = ui._all_inds_set_for_ptr(ptr)
            pairs = ui._group_pairs(gname, session=sess)
            if not pairs:
                continue
            # Precompute CCG bounds to skip stale pairs (pointer may outlive data)
            _cd = ui.cd._ccg.get(ckey.nd()) if hasattr(ui.cd, '_ccg') else None
            _ccg_sh = (_cd.ccg.shape if _cd is not None and hasattr(_cd, 'ccg')
                       and _cd.ccg is not None else None)
            for r, t in sorted((int(a), int(b)) for a, b in pairs):
                if r == t:
                    continue
                if (r, t) not in valid:
                    continue
                if (sess, r, t) in dead:
                    continue
                if (_ccg_sh is not None and len(_ccg_sh) >= 4
                        and (r >= _ccg_sh[1] or t >= _ccg_sh[2])):
                    continue
                yield ckey, r, t

    def _toggle_any_avail_group(self, gname: str):
        """Expand/collapse a group tag (Any mode); load CCG for involved sessions."""
        ui = self._ui
        if gname in ui._any_expanded_group_tags:
            ui._any_expanded_group_tags.discard(gname)
            ui.refresh_lists()
            return

        nds = ui._any_nd_keys_for_group(gname)

        def _finish_expand():
            ui._any_expanded_group_tags.add(gname)
            ui.refresh_lists()

        if not nds:
            _finish_expand()
            return

        def _chain(idx: int):
            if idx >= len(nds):
                _finish_expand()
                return
            ui._ensure_session_loaded(nds[idx], on_loaded=lambda: _chain(idx + 1))

        _chain(0)

    def _pair_in_group(self, pair, group_name: str) -> bool:
        ui = self._ui
        sess, p2 = ui._pair_sess_rt(pair)
        return p2 in ui._group_pairs(group_name, session=sess)

    def _save_groups_export(self):
        """Write groups_export.json (v4.0): registry + cross-session pair assignments.

        In any-mode: all sessions' data lives in _sel_data._groups in memory — write
        all of it directly.
        In single-session mode: merge-on-save — preserve other sessions from the
        existing file and update only the current session's pairs.
        """
        ui = self._ui
        ui._group_mgr._sync_registry_from_groups()
        export_path = os.path.join(ui._sel_save_dir, 'groups_export.json')
        cur_sess = ui._current_session_str()
        any_mode = getattr(ui, '_session_any_mode', False)

        # In single-session mode: load existing file to preserve other sessions' data
        existing_pairs: dict = {}  # gid_str → {sess → [[r,t],...]}
        if not any_mode and os.path.isfile(export_path):
            try:
                with open(export_path, encoding='utf-8') as f:
                    existing = json.load(f)
                if existing.get('version', '3.x') >= '4.0':
                    existing_pairs = existing.get('group_pairs', {})
                else:
                    # v3.x: migrate pair assignments into existing_pairs keyed by int IDs
                    for gname, val in existing.get('groups', {}).items():
                        gid = ui._group_mgr._ensure_group_registered(gname)
                        gid_str = str(gid)
                        existing_pairs.setdefault(gid_str, {})
                        if isinstance(val, dict):
                            for sess, pp in val.items():
                                if sess != cur_sess:
                                    existing_pairs[gid_str][sess] = pp
            except Exception as exc:
                print(f"[CCGReviewUI] _save_groups_export: failed to read existing: {exc}")

        # Build new group_pairs
        group_pairs: dict = {}
        for gid, g_entry in ui._sel_data._group_registry.items():
            gname = g_entry['name']
            gid_str = str(gid)
            g_pairs: dict = {}
            if any_mode:
                # All sessions' assignments are in memory — write them all
                g = ui._sel_data._groups.get(gname, {})
                if isinstance(g, dict):
                    for sess, pairs in g.items():
                        sorted_p = sorted(pairs) if pairs else []
                        if sorted_p:
                            g_pairs[sess] = [[int(r), int(c)] for r, c in sorted_p]
                elif isinstance(g, set):
                    sorted_p = sorted(g)
                    if sorted_p:
                        g_pairs[cur_sess] = [[int(r), int(c)] for r, c in sorted_p]
            else:
                # Single-session: preserve other sessions, update current session only
                for sess, pp in existing_pairs.get(gid_str, {}).items():
                    if sess != cur_sess:
                        g_pairs[sess] = pp
                cur_pairs = sorted(ui._group_pairs(gname, cur_sess))
                if cur_pairs:
                    g_pairs[cur_sess] = [[int(r), int(c)] for r, c in cur_pairs]
            if g_pairs:
                group_pairs[gid_str] = g_pairs

        data = {
            'version': '4.0',
            'group_registry': {str(k): v for k, v in ui._sel_data._group_registry.items()},
            'next_id': ui._sel_data._next_group_id,
            'group_pairs': group_pairs,
        }
        ui._atomic_write_json(export_path, data)

    def _load_groups_from_export(self):
        """Load group registry + all-session pair assignments from groups_export.json.

        Handles both v4.0 (registry + group_pairs) and v3.x (legacy string-name groups).
        Falls back to per-session __latest.json if the export file doesn't exist yet.
        """
        ui = self._ui
        export_path = os.path.join(ui._sel_save_dir, 'groups_export.json')
        if not os.path.isfile(export_path):
            latest_path = ui._sel_mgr._sel_version_path('latest')
            if os.path.isfile(latest_path):
                try:
                    with open(latest_path, encoding='utf-8') as f:
                        data = json.load(f)
                    ui._group_mgr._restore_groups_from_data(data)
                except Exception as exc:
                    print(f"[CCGReviewUI] failed to load groups from session file: {exc}")
            return
        try:
            with open(export_path, encoding='utf-8') as f:
                data = json.load(f)
            version = data.get('version', '3.x')
            if 'group_registry' in data or str(version) >= '4.0':
                ui._group_mgr._load_groups_v4(data)
            else:
                # v3.x → migrate
                ui._group_mgr._restore_groups_from_data(data, restore_hotkeys=True)
                ui._group_mgr._sync_registry_from_groups()
                # Rewrite as v4.0 immediately so future loads use new format
                try:
                    ui._save_groups_export()
                    print("[CCGReviewUI] groups_export.json migrated to v4.0")
                except Exception as exc:
                    print(f"[CCGReviewUI] migration save failed: {exc}")
            n_groups = len(ui._sel_data._groups)
            n_pairs = sum(len(p) for sd in ui._sel_data._groups.values()
                          if isinstance(sd, dict) for p in sd.values())
            print(f"[CCGReviewUI] groups loaded: {n_groups} groups, "
                  f"{n_pairs} pair-session entries")
            # Refresh any UI that depends on group/hotkey state.
            try:
                ui._rebuild_groups_menu()
            except Exception:
                pass
            try:
                if hasattr(ui, '_hotkeys_bar'):
                    ui._refresh_hotkeys_bar()
            except Exception:
                pass
        except Exception as exc:
            print(f"[CCGReviewUI] failed to load groups_export.json: {exc}")

    def _load_groups_v4(self, data: dict):
        """Populate _group_registry, _groups, _group_hotkeys, _group_notes from v4.0 data."""
        ui = self._ui
        registry = data.get('group_registry', {})
        ui._sel_data._group_registry = {}
        for k, v in registry.items():
            ui._sel_data._group_registry[int(k)] = v
        ui._sel_data._next_group_id = data.get('next_id', max(ui._sel_data._group_registry.keys(), default=0) + 1)

        # Rebuild _group_hotkeys and _group_notes from registry
        ui._sel_data._group_hotkeys = {}
        ui._sel_data._group_notes = {}
        for gid, g in ui._sel_data._group_registry.items():
            name = g['name']
            if g.get('hotkey'):
                ui._sel_data._group_hotkeys[name] = g['hotkey']
            if g.get('notes'):
                ui._sel_data._group_notes[name] = g['notes']

        # Populate _groups from group_pairs
        group_pairs = data.get('group_pairs', {})
        for gid_str, sessions_dict in group_pairs.items():
            try:
                gid = int(gid_str)
            except ValueError:
                continue
            gname = ui._sel_data._group_registry.get(gid, {}).get('name')
            if not gname:
                continue
            ui._sel_data._groups.setdefault(gname, {})
            for sess, pairs in sessions_dict.items():
                ui._sel_data._groups[gname][sess] = set(
                    tuple(int(v) for v in p) for p in pairs)

    def _merge_groups_from_session_files(self, export_path: str):
        """Merge group definitions from all per-session __latest.json files.

        Adds any group names (+ their pair assignments) that exist in per-session
        files but are missing from the already-loaded ui._sel_data._groups.  Does NOT
        overwrite existing entries in ui._sel_data._groups (export file is authoritative
        for groups that already exist there).  Saves the export file if anything
        was added.
        """
        ui = self._ui
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
                if g not in ui._sel_data._groups:
                    ui._sel_data._groups[g] = sess_pairs
                    added = True
                else:
                    for sess, pairs in sess_pairs.items():
                        if sess not in ui._sel_data._groups[g] and pairs:
                            ui._sel_data._groups[g][sess] = pairs
                            added = True
        if added:
            ui._save_groups_export()
            print("[CCGReviewUI] groups_export.json updated with groups from per-session files")

    def _restore_groups_from_data(self, data: dict, restore_hotkeys: bool = False):
        """Merge group data from a dict into ui._sel_data._groups (v3.x legacy path).

        CRITICAL: Merges — never overwrites existing entries — so loading a
        per-session file cannot erase groups belonging to other sessions.
        Pass restore_hotkeys=True only when loading from groups_export.json.
        """
        ui = self._ui
        raw_groups = data.get('groups', {})
        file_session = data.get('session', ui._current_session_str())
        for g, val in raw_groups.items():
            if isinstance(val, list):
                pairs = {file_session: set(tuple(int(v) for v in p) for p in val)}
            elif isinstance(val, dict):
                pairs = {sess: set(tuple(int(v) for v in p) for p in pp)
                         for sess, pp in val.items()}
            else:
                pairs = {}
            if g not in ui._sel_data._groups:
                ui._sel_data._groups[g] = pairs
            else:
                # Merge: only add sessions that don't already exist
                for sess, sp in pairs.items():
                    if sess not in ui._sel_data._groups[g]:
                        ui._sel_data._groups[g][sess] = sp
        if restore_hotkeys:
            hk = data.get('hotkeys', {})
            ui._sel_data._group_hotkeys.update(hk)
        notes = data.get('notes', {})
        for k, v in notes.items():
            if k not in ui._sel_data._group_notes:
                ui._sel_data._group_notes[k] = v
        ui._rebuild_groups_menu()

    def _pair_group_label(self, inds) -> str:
        """Return a short label like '[G1,G2]' for the groups this pair belongs to."""
        ui = self._ui
        sess, pair = ui._pair_sess_rt(inds)
        pair = tuple(int(x) for x in pair)
        labels = []
        for gname in ui._sel_data._groups:
            if pair not in ui._group_pairs(gname, session=sess):
                continue
            if is_special_group(gname):
                labels.append('*' + gname[len(_SPECIAL_PREFIX):])
            elif not gname.startswith('__'):
                labels.append(gname)
        pt = ui._sel_data._pair_tags.get(pair, {})
        tag_mark = '~' if (pt.get('tags') or pt.get('notes', '').strip()) else ''
        group_str = f"[{','.join(labels)}]" if labels else ""
        return tag_mark + group_str

    def _toggle_pair_group(self, pair, group_name):
        ui = self._ui
        if group_name not in ui._sel_data._groups:
            ui._sel_data._groups[group_name] = {}
        sess, p2 = ui._pair_sess_rt(pair)
        if p2 in ui._group_pairs(group_name, session=sess):
            ui._group_discard_pair(group_name, p2, session=sess)
        else:
            ui._group_mgr._group_add_pair(group_name, p2, session=sess)
        # Preserve scroll positions so the list doesn't jump to the top
        unsel_scroll = ui.unselected_list.yview()[0]
        sel_scroll = ui.selected_list.yview()[0]
        ui.refresh_lists()
        ui.unselected_list.yview_moveto(unsel_scroll)
        ui.selected_list.yview_moveto(sel_scroll)

    def _toggle_pairs_group(self, pairs, group_name):
        """Toggle multiple pairs in/out of a group.

        If ALL pairs are already in the group, remove them all.
        Otherwise, add all pairs to the group.
        """
        ui = self._ui
        if group_name not in ui._sel_data._groups:
            ui._sel_data._groups[group_name] = {}
        all_in = all(ui._pair_in_group(p, group_name) for p in pairs)
        # action = "REMOVE" if all_in else "ADD"
        if all_in:
            for p in pairs:
                s2, p2 = ui._pair_sess_rt(p)
                ui._group_discard_pair(group_name, p2, session=s2)
        else:
            for p in pairs:
                s2, p2 = ui._pair_sess_rt(p)
                ui._group_mgr._group_add_pair(group_name, p2, session=s2)
        unsel_scroll = ui.unselected_list.yview()[0]
        sel_scroll = ui.selected_list.yview()[0]
        ui.refresh_lists()
        ui.unselected_list.yview_moveto(unsel_scroll)
        ui.selected_list.yview_moveto(sel_scroll)

    def _create_group_dialog(self):
        ui = self._ui
        name = simpledialog.askstring(
            "Create group", "Group name:", parent=ui.root)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if name in ui._sel_data._groups:
            messagebox.showinfo("Create group", f"Group '{name}' already exists.")
            return
        ui._sel_data._groups[name] = {}
        ui._rebuild_groups_menu()
        ui.refresh_lists()

    def _create_special_group_dialog(self):
        ui = self._ui
        name = simpledialog.askstring(
            "Create special group", "Special group name:", parent=ui.root)
        if not name or not (name := name.strip()):
            return
        full_name = _SPECIAL_PREFIX + name
        if full_name in ui._sel_data._groups:
            messagebox.showinfo("Create special group",
                                f"Special group '{name}' already exists.")
            return
        ui._sel_data._groups[full_name] = {}
        ui._rebuild_groups_menu()
        ui.refresh_lists()

    def _rename_group(self, old_name, new_name, win=None):
        ui = self._ui
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return
        if new_name in ui._sel_data._groups:
            messagebox.showwarning("Rename", f"'{new_name}' already exists.")
            return

        # Preserve group identity in the v4 registry: rename the registry entry
        # instead of orphaning it (which can drop assignments on save).
        try:
            gid = ui._group_id_for(old_name)
            if gid is not None and gid in ui._sel_data._group_registry:
                ui._sel_data._group_registry[gid]['name'] = new_name
        except Exception:
            pass

        ui._sel_data._groups[new_name] = ui._sel_data._groups.pop(old_name)
        if old_name in ui._sel_data._group_hotkeys:
            if is_special_group(new_name):
                ui._sel_data._group_hotkeys.pop(old_name)
                try:
                    gid = ui._group_id_for(new_name)
                    if gid is not None and gid in ui._sel_data._group_registry:
                        ui._sel_data._group_registry[gid]['hotkey'] = None
                except Exception:
                    pass
            else:
                ui._sel_data._group_hotkeys[new_name] = ui._sel_data._group_hotkeys.pop(old_name)
        if old_name in ui._sel_data._group_notes:
            ui._sel_data._group_notes[new_name] = ui._sel_data._group_notes.pop(old_name)
        ui._rebuild_groups_menu()
        ui.refresh_lists()
        if win:
            win.destroy()
            ui._manage_groups_dialog()

    def _delete_group(self, name, win=None):
        ui = self._ui
        if not messagebox.askyesno("Delete group",
                                   f"Delete group '{name}'?"):
            return
        ui._sel_data._groups.pop(name, None)
        ui._sel_data._group_hotkeys.pop(name, None)
        ui._sel_data._group_notes.pop(name, None)
        ui._rebuild_groups_menu()
        ui.refresh_lists()
        if win:
            win.destroy()
            # Reopen if there are remaining groups
            if ui._sel_data._groups:
                ui._manage_groups_dialog()

    # ------------------------------------------------------------------
    # Auto-classification

    def _set_group_hotkey(self, group_name, key_str):
        """Assign hotkey: single digit 1–9/0 or single letter a–z."""
        ui = self._ui
        key_str = key_str.strip().lower()
        if not key_str:
            ui._sel_data._group_hotkeys.pop(group_name, None)
            ui._rebuild_groups_menu()
            return
        valid_digits = [str(i) for i in range(1, 10)] + ['0']
        if key_str not in valid_digits and not (len(key_str) == 1 and key_str.isalpha()):
            messagebox.showwarning("Hotkey", "Enter a digit 1–9/0 or a single letter a–z.")
            return
        # Remove from any other group that had this key
        for g, k in list(ui._sel_data._group_hotkeys.items()):
            if k == key_str and g != group_name:
                del ui._sel_data._group_hotkeys[g]
        ui._sel_data._group_hotkeys[group_name] = key_str
        ui._rebuild_groups_menu()

    def _rebuild_groups_menu(self):
        """Refresh the dynamic part of the Groups menu."""
        ui = self._ui
        if hasattr(ui, 'left_container'):
            ui.left_container.left_panel._rebuild_groups_menu()
            return
        # Pre-container fallback (during early setup before left_container exists)
        if not hasattr(ui, '_groups_menu'):
            return
        try:
            while ui._groups_menu.index('end') >= 7:
                ui._groups_menu.delete(7)
        except tk.TclError:
            pass
        ui.network_panel.refresh_group_buttons()
        if (hasattr(ui, '_hotkeys_bar') and
                ui._panel_vars.get('Group Hotkeys', tk.BooleanVar()).get()):
            ui._refresh_hotkeys_bar()

    def _select_group(self, group_name):
        """Navigate to the first pair in the group."""
        ui = self._ui
        pairs = ui._group_pairs(group_name)
        if not pairs:
            return
        first = sorted(pairs)[0]
        ui.current_pair_idx = ui.get_pair_index(first)
        ui.update_plot()
        ui.network_panel.draw()

    def _group_hotkey_handler(self, key_str: str, advance: bool = True):
        """Toggles the current pair in/out of the group assigned to key_str.

        key_str is a single character: '1'-'9', '0', or 'a'-'z'.
        advance=False (Shift held): tag without moving the cursor.
        """
        ui = self._ui
        current_pair = ui._selected_pair_from_lists()
        if current_pair is None:
            if ui.current_pair_idx >= len(ui.all_inds):
                return
            if getattr(ui, '_session_any_mode', False):
                trip = ui._pair_at_all_inds_idx(ui.current_pair_idx)
                if trip is None:
                    return
                current_pair = trip
            else:
                row = ui.all_inds[ui.current_pair_idx]
                current_pair = tuple(int(x) for x in row)
        for gname, k in ui._sel_data._group_hotkeys.items():
            if k != key_str:
                continue

            if not advance:
                # Ctrl held: only the current pair, no listbox multi-select
                highlighted = [current_pair]
                # If the user is holding Shift to apply multiple tags, advance
                # once when Shift is released.
                ui._shift_tag_pending_advance = True
            else:
                # ── Collect all highlighted pairs from both listboxes ────────
                _lp = getattr(ui.left_container, 'left_panel', None)
                avail_map = getattr(_lp, '_avail_list_pairs', None)
                highlighted = []
                for i in ui.unselected_list.curselection():
                    if avail_map and i < len(avail_map):
                        entry = avail_map[i]
                        if entry is None or is_separator_row(entry):
                            continue
                        if entry[1] != 'deleted':
                            highlighted.append(entry[0])
                    elif not avail_map:
                        su = sorted(ui.unselected_inds)
                        if i < len(su):
                            highlighted.append(su[i])
                sel_map = getattr(_lp, '_sel_list_pairs', None)
                for i in ui.selected_list.curselection():
                    if sel_map and i < len(sel_map):
                        inds = sel_map[i]
                        if inds is not None:
                            highlighted.append(inds)
                    elif not sel_map:
                        ss = sorted(ui.selected_inds)
                        if i < len(ss):
                            highlighted.append(ss[i])
                # If nothing is explicitly selected, fall back to current pair.
                if not highlighted:
                    highlighted = [current_pair]

            multi = len(highlighted) > 1
            changed = set()
            ui._push_undo()

            for pair in highlighted:
                was_in_group = ui._pair_in_group(pair, gname)
                ui._toggle_pair_group(pair, gname)
                if getattr(ui, '_session_any_mode', False):
                    changed.add(pair)
                    continue
                if not was_in_group:
                    # Gained a tag → move to selected
                    if pair in ui.unselected_inds:
                        ui._sel_data.set_state(pair, 'sel')
                        changed.add(pair)
                else:
                    # Lost a tag → move back to available if no tags remain
                    if pair in ui.selected_inds:
                        has_groups = any(
                            ui._pair_in_group(pair, g)
                            for g in ui._sel_data._groups
                            if not g.startswith('__')
                        )
                        if not has_groups:
                            ui._sel_data.set_state(pair, 'unsel')
                            changed.add(pair)

            ui.refresh_lists()
            ui._highlight_changed_pairs(changed or {current_pair})
            if advance:
                next_idx = min(ui.current_pair_idx + 1, len(ui.all_inds) - 1)
                ui.current_pair_idx = next_idx
                ui._select_pair_in_list(ui._pair_at_all_inds_idx(next_idx))
            else:
                # No advance: keep cursor on current pair after list rebuild
                ui._select_pair_in_list(current_pair)
            ui.update_plot()
            ui.network_panel.draw()
            return
        # No group assigned to this hotkey — show temporary warning
        ui._show_temp_warning(f"No group assigned to Ctrl+{key_str}")

    def _export_groups(self):
        """Export all group definitions to a standalone JSON file."""
        ui = self._ui
        if not ui._sel_data._groups:
            messagebox.showinfo("Export groups", "No groups to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export groups",
            defaultextension='.json',
            filetypes=[('JSON files', '*.json')],
            initialfile='groups_export.json',
            initialdir=ui._sel_save_dir,
        )
        if not path:
            return
        groups = {}
        for g, sessions_dict in ui._sel_data._groups.items():
            if isinstance(sessions_dict, set):
                groups[g] = {ui._current_session_str():
                             [[int(r), int(c)] for r, c in sorted(sessions_dict)]}
            else:
                groups[g] = {sess: [[int(r), int(c)] for r, c in sorted(pairs)]
                             for sess, pairs in sessions_dict.items() if pairs}
        data = {
            'groups': groups,
            'hotkeys': dict(ui._sel_data._group_hotkeys),
            'notes': dict(ui._sel_data._group_notes),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=SelectionPersistenceManager._json_default)
        print(f"[CCGReviewUI] groups exported → {path}")

    def _import_groups(self):
        """Import group definitions from a JSON file, merging with existing."""
        ui = self._ui
        path = filedialog.askopenfilename(
            title="Import groups",
            filetypes=[('JSON files', '*.json')],
            initialdir=ui._sel_save_dir,
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
                sess = ui._current_session_str()
                for p in val:
                    ui._group_mgr._group_add_pair(gname, tuple(int(v) for v in p), sess)
            elif isinstance(val, dict):
                # Per-session format
                for sess, pairs in val.items():
                    for p in pairs:
                        ui._group_mgr._group_add_pair(gname, tuple(int(v) for v in p), sess)
            else:
                ui._sel_data._groups.setdefault(gname, {})
        for gname, hk in data.get('hotkeys', {}).items():
            if gname not in ui._sel_data._group_hotkeys:
                ui._sel_data._group_hotkeys[gname] = hk
        for gname, note in data.get('notes', {}).items():
            if gname not in ui._sel_data._group_notes:
                ui._sel_data._group_notes[gname] = note
        ui._rebuild_groups_menu()
        ui.refresh_lists()
        print(f"[CCGReviewUI] groups imported from {path}")

    # ------------------------------------------------------------------
    # Versioning helpers (Part II.3)
    # ------------------------------------------------------------------

    def _show_hotkeys_dialog(self):
        """Show a dialog listing all keyboard shortcuts."""
        ui = self._ui
        hotkeys_text = (
            "Ctrl+E    Toggle between waveform and CCG\n"
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

    def setup_groups_menu(self, menubar):
        """Groups menu: create / manage pair groups."""
        ui = self._ui
        ui._groups_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Groups", menu=ui._groups_menu)
        ui._groups_menu.add_command(label="Create group…",
                                      command=ui._create_group_dialog)
        ui._groups_menu.add_command(label="Create special group…",
                                      command=ui._create_special_group_dialog)
        ui._groups_menu.add_command(label="Manage groups…",
                                      command=ui._manage_groups_dialog)
        ui._groups_menu.add_command(label="Merge groups…",
                                      command=lambda: MergeGroupsDialog.show(ui))
        ui._groups_menu.add_command(label="Export groups…",
                                      command=ui._group_mgr._export_groups)
        ui._groups_menu.add_command(label="Import groups…",
                                      command=ui._group_mgr._import_groups)
        ui._groups_menu.add_separator()
        # Dynamic group entries added in _rebuild_groups_menu()

    def setup_group_hotkeys_bar(self):
        """Horizontal bar showing Ctrl+1…0 → group-name mappings."""
        ui = self._ui
        ui._hotkeys_bar = ttk.Frame(ui.root, relief=tk.GROOVE, borderwidth=1)
        ui._hotkeys_bar_labels: list[tk.Label] = []
        ui._hk_scroller = ArrowScroller()
        self._refresh_hotkeys_bar()
        # Pack immediately if default is visible
        if ui._panel_vars.get('Group Hotkeys', tk.BooleanVar()).get():
            ui._hotkeys_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 2))

    def _refresh_hotkeys_bar(self):
        """Rebuild the labels inside the hotkeys bar."""
        ui = self._ui
        for w in ui._hotkeys_bar.winfo_children():
            w.destroy()
        ui._hotkeys_bar_labels.clear()

        ttk.Label(ui._hotkeys_bar, text="Groups:",
                  font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(6, 4))

        del_lbl = tk.Label(ui._hotkeys_bar, text="Del/⌫: deleted",
                           font=('Courier', 9), padx=6, pady=1,
                           relief=tk.RIDGE, borderwidth=1, fg='#888888')
        del_lbl.pack(side=tk.LEFT, padx=2, pady=2)

        # Order: digits 1–9, 0 first, then letters a–z
        digit_order = [str(i) for i in range(1, 10)] + ['0']
        letter_order = list('abcdefghijklmnopqrstuvwxyz')
        slot_order = digit_order + letter_order
        # Invert: hotkey_str → group_name
        hk_to_group = {v: k for k, v in ui._sel_data._group_hotkeys.items()}

        # Build ordered list of (key_str, gname) chips
        all_chips = [(k, hk_to_group[k]) for k in slot_order if k in hk_to_group]

        if not all_chips:
            ttk.Label(ui._hotkeys_bar, text="(no hotkeys assigned)",
                      font=('Arial', 9), foreground='#888').pack(
                side=tk.LEFT, padx=4)
            return

        n_chips = len(all_chips)
        ui._hk_scroller.install(ui._hotkeys_bar, n_chips,
                                lambda _: self._refresh_hotkeys_bar())
        offset = ui._hk_scroller.offset

        for key_str, gname in all_chips[offset:]:
            display = f"{key_str}: {gname}"
            lbl = tk.Label(ui._hotkeys_bar, text=display,
                           font=('Courier', 9), padx=6, pady=1,
                           relief=tk.RIDGE, borderwidth=1)
            lbl.pack(side=tk.LEFT, padx=2, pady=2)
            lbl.bind('<Button-1>',
                     lambda e, g=gname: ui._select_group(g))
            lbl.bind('<Double-Button-1>',
                     lambda e, g=gname: self._group_chip_double_click(g))
            ui._hotkeys_bar_labels.append(lbl)

    def _group_chip_double_click(self, group_name: str):
        """Draw a random example pair from group_name; flash chip red if empty."""
        ui = self._ui
        pairs = ui._group_pairs(group_name)
        if not pairs:
            # Flash the chip label red for 0.3 s
            for lbl in ui._hotkeys_bar_labels:
                if group_name in lbl.cget('text'):
                    orig_fg = lbl.cget('fg')
                    lbl.config(fg='red')
                    ui.root.after(300, lambda l=lbl, c=orig_fg: l.config(fg=c))
            return
        chosen = random.choice(sorted(pairs))
        ui.current_pair_idx = ui.get_pair_index(chosen)
        ui.update_plot()
        ui.network_panel.draw()

    # ── Left panel ─────────────────────────────────────────────────────

class SelectionPersistenceManager:
    """Selection save/load, undo/redo, history, and bookmarks for CCGReviewUI."""

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        ui._pair_deleted_store: dict = {}

    def setup_file_menu(self, menubar):
        """Selections menu: save / load selection versions."""
        ui = self._ui
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Selections", menu=file_menu)
        file_menu.add_command(label="Save selection…",
                              command=ui._sel_mgr._quick_save)
        file_menu.add_command(label="Load selection…",
                              command=ui._sel_mgr._load_selection_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Export as PNG…",
                              command=ui._export_current_view)
        file_menu.add_separator()
        file_menu.add_command(label="Clear bookmarks",
                              command=ui._clear_bookmarks)

    def _selections_menu_close(self):
        ui = self._ui
        ui._bookmarked_pairs.clear()
        ui.root.destroy()

    def _push_undo(self):
        """Snapshot current selection AND group state before a mutation."""
        ui = self._ui
        ui._undo_stack.append((
            set(ui.selected_inds),
            set(ui.unselected_inds),
            set(ui.deleted_inds),
            _copy.deepcopy(ui._sel_data._groups),
        ))
        if len(ui._undo_stack) > ui._UNDO_LIMIT:
            ui._undo_stack.pop(0)
        ui._redo_stack.clear()

    # Highlight color for undo/redo indicators (matches CCG baseline orange)
    _UNDO_HIGHLIGHT = '#ff7f0e'

    def _undo(self, event=None):
        ui = self._ui
        if not ui._undo_stack:
            return
        cur = (set(ui.selected_inds), set(ui.unselected_inds),
               set(ui.deleted_inds), _copy.deepcopy(ui._sel_data._groups))
        ui._redo_stack.append(cur)
        state = ui._undo_stack.pop()
        ui._sel_data.populate(state[0] | state[1] | (state[2] if len(state) > 2 else set()),
                              selected=state[0], deleted=state[2] if len(state) > 2 else set())
        if len(state) > 3:
            ui._sel_data._groups = state[3]
            ui._rebuild_groups_menu()
        changed = (cur[0] ^ state[0]) | (cur[1] ^ state[1])
        ui.refresh_lists()
        ui._highlight_changed_pairs(changed)
        ui._flush_deleted_to_store()
        ui.update_plot()
        ui.network_panel.draw()
        ui._refresh_stats()

    def _redo(self, event=None):
        ui = self._ui
        if not ui._redo_stack:
            return
        cur = (set(ui.selected_inds), set(ui.unselected_inds),
               set(ui.deleted_inds), _copy.deepcopy(ui._sel_data._groups))
        ui._undo_stack.append(cur)
        state = ui._redo_stack.pop()
        ui._sel_data.populate(state[0] | state[1] | (state[2] if len(state) > 2 else set()),
                              selected=state[0], deleted=state[2] if len(state) > 2 else set())
        if len(state) > 3:
            ui._sel_data._groups = state[3]
            ui._rebuild_groups_menu()
        changed = (cur[0] ^ state[0]) | (cur[1] ^ state[1])
        ui.refresh_lists()
        ui._highlight_changed_pairs(changed)
        ui._flush_deleted_to_store()
        ui.update_plot()
        ui.network_panel.draw()
        ui._refresh_stats()

    def _clear_undo_highlight(self):
        """Remove undo/redo highlight from all list items."""
        ui = self._ui
        for listbox in (ui.unselected_list, ui.selected_list):
            for idx in range(listbox.size()):
                listbox.itemconfig(idx, background='', foreground='')

    # ------------------------------------------------------------------
    # Pair lists — bridges to LeftPanel
    # ------------------------------------------------------------------

    def _reapply_bookmark_list_styles(self):
        ui = self._ui
        if hasattr(ui, 'left_container'):
            ui.left_container.left_panel._reapply_bookmark_list_styles()

    def _bookmark_toggle_current(self, event=None):
        ui = self._ui
        if hasattr(ui, 'left_container'):
            ui.left_container.left_panel._bookmark_toggle_current(event)

    def _clear_bookmarks(self):
        ui = self._ui
        if hasattr(ui, 'left_container'):
            ui.left_container.left_panel._clear_bookmarks()

    def _history_dir(self) -> str:
        ui = self._ui
        return os.path.join(ui._sel_save_dir, ui._HISTORY_SUBDIR)

    def _save_to_history(self, data: dict, suffix: str) -> str:
        """Write data dict to .history/{session}__{ts}{suffix}.json and git-commit."""
        ui = self._ui
        hdir = ui._sel_mgr._history_dir()
        os.makedirs(hdir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        sess = getattr(ui.key, 'session', 'sess')
        fname = f"{sess}__{ts}{suffix}.json"
        path = os.path.join(hdir, fname)
        ui._atomic_write_json(path, data)
        ui._git_commit_paths([path], f'[history] {fname}')
        return path

    def _purge_history(self):
        """Delete .history/ files older than 3 days and commit the deletion."""
        ui = self._ui
        hdir = ui._sel_mgr._history_dir()
        if not os.path.isdir(hdir):
            return
        cutoff = datetime.datetime.now() - datetime.timedelta(days=7)
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
            repo = os.path.abspath(os.path.join(ui._sel_save_dir, '..', '..'))
            for p in removed:
                subprocess.run(['git', 'rm', '--cached', '-f',
                                os.path.relpath(p, repo)],
                               cwd=repo, capture_output=True)
            subprocess.run(['git', 'commit', '--no-gpg-sign', '-m',
                            f'[history] purge {len(removed)} files older than 7 days'],
                           cwd=repo, capture_output=True)
            print(f"[CCGReviewUI] purged {len(removed)} history files older than 7 days")

    def _save_autosnapshot(self):
        """Periodic 15-min autosave to .history/ as .autosaved.json."""
        ui = self._ui
        if ui.ccg_ptr is None or getattr(ui, '_closing', False):
            return
        try:
            data = self._build_save_dict(
                datetime.datetime.now().isoformat(), 'autosaved')
            ui._sel_mgr._save_to_history(data, '.autosaved')
            print(f"[CCGReviewUI] autosnapshot saved")
        except Exception as exc:
            print(f"[CCGReviewUI] autosnapshot failed: {exc}")

    def _schedule_autosnapshot(self):
        ui = self._ui
        def _do():
            ui._sel_mgr._save_autosnapshot()
            ui.root.after(ui._AUTOSAVE_INTERVAL_MS, _do)
        ui.root.after(ui._AUTOSAVE_INTERVAL_MS, _do)

    def _autoload_session_latest(self, restore_groups: bool = False):
        """Load the latest selection file for the current session.

        By default only restores pair selections. Pass restore_groups=True on
        first launch to also load groups from groups_export.json (not the
        per-session file, which may have stale group entries).
        """
        ui = self._ui
        latest_path = self._sel_version_path('latest')
        if not os.path.isfile(latest_path):
            # Even without a session file, try to load groups
            if restore_groups:
                ui._group_mgr._load_groups_from_export()
            return
        try:
            # Always load pair selections from the session-specific file;
            # never load groups from it (they may be stale).
            ui._load_selection_from_file(latest_path,
                                           restore_groups=False,
                                           _skip_redraw=True)
        except Exception as exc:
            print(f"[CCGReviewUI] failed to autoload latest: {exc}")
        if restore_groups:
            ui._group_mgr._load_groups_from_export()

    def _autosave_current(self):
        """Silently save current state to a fixed-name action-autosave file.

        Writes to .history/{session}-last_action.autosaved.json (overwrite, no
        new file per call).  Does NOT touch __latest.json — that is only written
        on explicit user save.
        """
        ui = self._ui
        if getattr(ui, '_session_any_mode', False):
            try:
                ui._autosave_all_sessions_for_current_type()
            except Exception as exc:
                print(f"[CCGReviewUI] any-session autosave failed: {exc}")
            try:
                if ui._sel_data._groups:
                    ui._save_groups_export()
            except Exception:
                traceback.print_exc()
            try:
                ui._save_ui_state()
            except Exception:
                traceback.print_exc()
            return
        if ui.ccg_ptr is None:
            return
        ui._sel_mgr._autosave_to_history_fixed()

    def _autosave_to_history_fixed(self):
        """Write selections to .history/{session}-last_action.autosaved.json (overwrite)."""
        ui = self._ui
        try:
            data = self._build_save_dict(datetime.datetime.now().isoformat(), 'action_autosave')
            sess = getattr(ui.key, 'session', 'sess')
            hdir = ui._sel_mgr._history_dir()
            os.makedirs(hdir, exist_ok=True)
            path = os.path.join(hdir, f"{sess}-last_action.autosaved.json")
            ui._atomic_write_json(path, data)
        except Exception as exc:
            print(f"[CCGReviewUI] action autosave failed: {exc}")

    def _autosave_all_sessions_for_current_type(self):
        """Write fixed action-autosave file for each physical session (any-mode).

        Writes to .history/{session}-last_action.autosaved.json per session
        (overwrite).  Does NOT touch __latest.json.
        """
        ui = self._ui
        if getattr(ui, '_session_any_mode', False):
            ui._sess_mgr._flush_any_selections_to_pointers()
        ui._sess_mgr._flush_any_deleted_to_stores()
        lbl = ui._type_label(ui.key)
        saved_sess: set[str] = set()
        old_key = ui.key
        old_ptr = ui.ccg_ptr
        old_cd = ui.ccg_data
        old_neurons = ui.neurons
        old_ns = ui.n_segments
        try:
            for nk in ui._real_nd_keys_ordered():
                ckey = ui._type_key_for_nd(nk)
                if ckey is None or ui._type_label(ckey) != lbl:
                    continue
                sess = str(ckey.session)
                if sess in saved_sess:
                    continue
                saved_sess.add(sess)
                ui._bind_context_to_type_key(ckey)
                try:
                    ui._sel_mgr._autosave_to_history_fixed()
                except Exception as exc:
                    print(f"[CCGReviewUI] any-session autosave failed for {sess}: {exc}")
        finally:
            ui.key = old_key
            ui.ccg_ptr = old_ptr
            ui.ccg_data = old_cd
            ui.neurons = old_neurons
            ui.n_segments = old_ns
            # The save loop binds every session; restoring ``old_*`` can leave
            # ``ccg_*`` on session A while ``current_pair_idx`` still points at
            # a pair row for session B → IndexError in ``_resolve_segment_data``.
            if getattr(ui, '_session_any_mode', False):
                idx = ui.current_pair_idx
                hl = getattr(ui, '_any_pair_handle_list', None) or []
                if (getattr(ui.network_panel, '_focused_pair', None) is None
                        and 0 <= idx < len(hl)):
                    ui._sync_any_plot_context(idx)

    def _sel_version_path(self, name: str) -> str:
        ui = self._ui
        safe = name.replace('/', '_').replace('\\', '_').replace(' ', '_')
        session_tag = getattr(ui.key, 'session', 'sess')
        return os.path.join(ui._sel_save_dir, f"{session_tag}__{safe}.json")

    @staticmethod
    def _json_default(obj):
        """JSON encoder that converts numpy integer/float types to Python scalars."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')

    def _enforce_label_selection_integrity_live(self):
        """If a pair has labels/tags/notes, force it into selected."""
        ui = self._ui
        if getattr(ui, '_session_all_mode', False):
            # all-in-one view selections are triples; pair-tags are session-local pairs.
            return
        tagged_pairs = {
            tuple(map(int, p))
            for p, entry in getattr(ui, '_pair_tags', {}).items()
            if SelectionPersistenceManager._pair_tag_has_labels(entry)
        }
        if not tagged_pairs:
            return
        avail = set(map(tuple, ui.all_inds))
        to_select = tagged_pairs & avail
        if not to_select:
            return
        for p in to_select:
            ui._sel_data.set_state(p, 'sel')

    def _enforce_label_selection_integrity_file(
            self, selections_by_type: dict, pair_tags: dict, type_keys: list):
        """Normalize loaded file so pair_tags-labeled pairs are selected in some type."""
        ui = self._ui
        if not isinstance(selections_by_type, dict):
            return selections_by_type
        tagged_pairs = set()
        for key_str, entry in (pair_tags or {}).items():
            if not SelectionPersistenceManager._pair_tag_has_labels(entry):
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
                ptr = ui.cd.data.get(tk_)
                valid = ui._all_inds_set_for_ptr(ptr)
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
        ui = self._ui
        saved_at = datetime.datetime.now().isoformat()
        data = self._build_save_dict(saved_at, name)
        path = self._sel_version_path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ui._atomic_write_json(path, data)
        n_tags = len(data['pair_tags'])
        print(f"[CCGReviewUI] saved → {os.path.basename(path)}  "
              f"({n_tags} pair_tags, {len(data['selections'])} type keys)")
        try:
            ui._sel_mgr._save_to_history(data, '')
        except Exception as exc:
            print(f"[CCGReviewUI] history copy failed: {exc}")
        return path

    def _list_selection_versions(self) -> list:
        """Return list of (name, path, saved_at, is_valid, is_history) tuples.

        Main versions (in data/selections/) are listed first; then .history/
        entries (backups + autosaves) follow, newest first, marked is_history=True.
        """
        ui = self._ui
        session_tag = getattr(ui.key, 'session', 'sess')
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

        if os.path.isdir(ui._sel_save_dir):
            for fname in sorted(os.listdir(ui._sel_save_dir)):
                if not fname.startswith(prefix) or not fname.endswith('.json'):
                    continue
                versions.append(_read_entry(
                    os.path.join(ui._sel_save_dir, fname), fname, False))

        hdir = ui._sel_mgr._history_dir()
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
        ui = self._ui
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

        if data.get('neuron_ids') is not None:
            m = SelectionMigration(data['neuron_ids'], ui.neurons.neuron_ids)
            if not m.is_identity:
                data = m.apply(data)

        selections_by_type = data.get('selections', {})
        type_keys = ui._available_type_keys(ui.key.nd())
        selections_by_type = self._enforce_label_selection_integrity_file(
            selections_by_type, data.get('pair_tags', {}), type_keys)
        saved_keys = set(selections_by_type.keys())
        for tk_ in type_keys:
            ptr = ui.cd.data.get(tk_)
            if ptr is None:
                continue
            pairs = selections_by_type.get(str(tk_), [])
            if not pairs and saved_keys:
                print(f"[load] key mismatch: current={str(tk_)!r}, saved keys={sorted(saved_keys)}")
            if pairs:
                ptr.selected_inds = np.array(
                    [[int(r), int(c)] for r, c in pairs], dtype=int)
            else:
                ptr.selected_inds = None
        cur_sel = selections_by_type.get(str(ui.key), [])
        print(f"[load] ui.key={str(ui.key)!r}, cur_sel count={len(cur_sel)}, saved keys={sorted(selections_by_type.keys())}")
        selected = set(tuple(int(v) for v in p) for p in cur_sel)

        ui._push_undo()
        current_available = set(map(tuple, ui.all_inds))
        print(f"[load] available={len(current_available)}, selected={len(selected)}, overlap={len(selected & current_available)}")
        missing = selected - current_available
        if missing and restore_groups:
            action = self._show_missing_pairs_dialog(missing)
            if action == 'cancel':
                return
            elif action == 'partial':
                selected = selected & current_available
            elif action == 'admit_all':
                for pair in missing:
                    _admit_pair(ui._sel_data, pair)
                current_available = set(map(tuple, ui.all_inds))
        elif missing:
            _cd = ui.cd._ccg.get(ui.key.nd()) if hasattr(ui.cd, '_ccg') else ui.ccg_data
            _n = (_cd.ccg.shape[1] if _cd is not None
                  and getattr(_cd, 'ccg', None) is not None else None)
            for pair in missing:
                if _n is not None and (pair[0] >= _n or pair[1] >= _n):
                    continue
                _admit_pair(ui._sel_data, pair)
            current_available = set(map(tuple, ui.all_inds))

        ui._pair_deleted_store = {}
        dbtype = data.get('deleted_by_type') or {}
        for k_str, plist in dbtype.items():
            ui._pair_deleted_store[k_str] = {
                tuple(int(v) for v in p) for p in plist}
        deleted = set(ui._pair_deleted_store.get(str(ui.key), set())) & current_available
        ui._sel_data.populate(current_available, selected=selected, deleted=deleted)

        # Load pair_tags for this session — always reset to avoid stale cross-session tags
        ui._sel_data._pair_tags = {}
        raw_tags = data.get('pair_tags', {})
        cur_sess = ui._current_session_str()
        for key_str, tdata in raw_tags.items():
            parts = key_str.split(',')
            if len(parts) != 2:
                continue
            pair = (int(parts[0]), int(parts[1]))
            entry = dict(tdata) if isinstance(tdata, dict) else {'notes': str(tdata)}
            ui._sel_data._pair_tags[pair] = entry
            if 'groups' in entry:
                for gitem in entry['groups']:
                    gname = gitem if isinstance(gitem, str) else None
                    if gname:
                        ui._sel_data._groups.setdefault(str(gname), {}).setdefault(
                            cur_sess, set()).add(pair)
        self._enforce_label_selection_integrity_live()
        if not _skip_redraw:
            ui._post_load_refresh()

    def _load_selection_dialog(self):
        """Show a dialog listing all saved versions; user picks one to load."""
        ui = self._ui
        versions = self._list_selection_versions()
        if not versions:
            messagebox.showinfo("Load selection",
                                "No saved selections found for this key.")
            return
        win = tk.Toplevel(ui.root)
        win.title("Load Selection")
        win.geometry("620x340")
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
                ui._load_selection_dialog()   # reopen with updated list
                return
            try:
                ui._load_selection_from_file(path)
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
            deleted_dir = os.path.join(ui._sel_save_dir, 'deleted')
            os.makedirs(deleted_dir, exist_ok=True)
            try:
                shutil.move(path, os.path.join(deleted_dir, os.path.basename(path)))
            except OSError as ex:
                messagebox.showerror("Delete failed", str(ex), parent=win)
                return
            win.destroy()
            ui._load_selection_dialog()  # reopen with updated list

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

    def _do_save(self, name: str):
        """Core save logic: persist all types' selections + groups."""
        ui = self._ui
        if not ui._sel_mgr._save_all_state(name, silent=False):
            return

        # Count total selections across all types
        type_keys = ui._available_type_keys(ui.key.nd())
        total = sum(
            len(ui.cd.data[tk_].selected_inds)
            for tk_ in type_keys
            if ui.cd.data.get(tk_) is not None
            and getattr(ui.cd.data[tk_], 'selected_inds', None) is not None
        )

        # Groups were exported via _save_all_state; keep message for UI feedback.
        groups_msg = f"\nGroups exported ({len(ui._sel_data._groups)} groups)." if ui._sel_data._groups else ""

        messagebox.showinfo(
            "Saved",
            f"Saved {total} pairs across {len(type_keys)} types as '{name}'.{groups_msg}",
            parent=ui.root)

    def _quick_save(self):
        """Ctrl+S / Save button: custom dialog with name entry + Latest button."""
        ui = self._ui
        default_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

        win = tk.Toplevel(ui.root)
        win.title("Save selection")
        win.geometry("360x130")
        win.transient(ui.root)
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

    def _show_temp_warning(self, msg: str, duration_ms: int = 2000):
        """Show a temporary warning label at the top of the window that auto-disappears."""
        ui = self._ui
        lbl = tk.Label(ui.root, text=msg, bg='#FFF3CD', fg='#856404',
                       font=('Arial', 10, 'bold'), padx=8, pady=4)
        lbl.place(relx=0.5, y=4, anchor='n')
        ui.root.after(duration_ms, lbl.destroy)

    def _build_save_dict(self, saved_at: str, name: str = '') -> dict:
        """Build the serializable dict for a session save (v4.0 format).

        Flushes current type's selections to the pointer, then collects all
        type keys + pair_tags (including group membership) + deleted pairs.
        Does NOT write any files.
        """
        ui = self._ui
        if ui.ccg_ptr is None:
            raise RuntimeError("Cannot save: CCG data not yet loaded")
        self._enforce_label_selection_integrity_live()
        # Flush current type's selections to pointer
        if getattr(ui, '_session_any_mode', False):
            ui._sess_mgr._flush_any_selections_to_pointers()
        else:
            ui.ccg_ptr.selected_inds = (
                np.array(sorted(ui.selected_inds), dtype=int)
                if ui.selected_inds else None
            )
        # Collect selections for every type key in this session
        type_keys = ui._available_type_keys(ui.key.nd())
        selections_by_type = {}
        for tk_ in type_keys:
            ptr = ui.cd.data.get(tk_)
            if ptr is None:
                continue
            sel = getattr(ptr, 'selected_inds', None)
            selections_by_type[str(tk_)] = (
                [[int(r), int(c)] for r, c in sorted(map(tuple, sel))]
                if sel is not None and len(sel) > 0 else []
            )
        # Serialize pair_tags: include group membership (names) for this session.
        # NOTE: We intentionally store group NAMES (not numeric IDs) in session save
        # files. IDs remain internal to groups_export.json only.
        cur_sess = ui._current_session_str()
        pair_tags_ser: dict = {}
        # Collect all pairs that have either tags/notes OR group membership
        all_annotated = set(ui._sel_data._pair_tags.keys())
        for gname, sd in ui._sel_data._groups.items():
            if isinstance(sd, dict):
                all_annotated |= sd.get(cur_sess, set())
        for pair in sorted(all_annotated):
            r, t = int(pair[0]), int(pair[1])
            existing = dict(ui._sel_data._pair_tags.get(pair, {}))
            # Compute group names for this pair in the current session
            group_names: list[str] = []
            for gname, sd in ui._sel_data._groups.items():
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
        ui._flush_deleted_to_store()
        deleted_by_type = {}
        for tk_ in type_keys:
            ptr = ui.cd.data.get(tk_)
            valid = ui._all_inds_set_for_ptr(ptr)
            raw = set(ui._pair_deleted_store.get(str(tk_), set())) & valid
            deleted_by_type[str(tk_)] = [[int(r), int(c)] for r, c in sorted(raw)]
        return {
            'name': name,
            'saved_at': saved_at,
            'session': getattr(ui.key, 'session', 'sess'),
            'nd_key': str(ui.key.nd()),
            'neuron_ids': [int(x) for x in ui.neurons.neuron_ids],
            'selections': selections_by_type,
            'pair_tags': pair_tags_ser,
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

    def _show_missing_pairs_dialog(self, missing: set) -> str:
        """Dialog when loaded selection has pairs not in current available set.

        Returns 'partial', 'admit_all', or 'cancel'.
        """
        ui = self._ui
        win = tk.Toplevel(ui.root)
        win.title("Missing Pairs")
        win.geometry("450x320")
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
        win.update_idletasks()
        win.lift()
        win.focus_force()
        win.wait_window()
        return result['action']

    def _save_all_state(self, selection_name: str | None = None, *, silent: bool = True) -> bool:
        """Single saving pathway for selections + groups + ui_state.

        - If selection_name is provided: writes that selection version.
        - Always attempts to write groups export (if any groups exist).
        - Always writes ui_state.json (panel + display button state, resolution, etc.).
        """
        ui = self._ui
        if selection_name is not None:
            try:
                self._save_selection_version(selection_name)
            except Exception as exc:
                traceback.print_exc()
                if not silent:
                    messagebox.showerror("Save error",
                                         f"Failed to save selection:\n{exc}",
                                         parent=ui.root)
                return False
        try:
            if ui._sel_data._groups:
                ui._save_groups_export()
        except Exception:
            # never block save on groups export
            traceback.print_exc()
        try:
            ui._save_ui_state()
        except Exception:
            traceback.print_exc()
        return True

    def run(self):
        ui = self._ui
        if ui._owns_mainloop:
            ui.root.mainloop()
        else:
            # Another Tk root owns the mainloop; just wait for this window
            ui.root.wait_window(ui.root)


    def _current_filter_state(self) -> dict:
        ui = self._ui
        toggles = getattr(ui.time_slider, '_ts_legend_toggles', {})
        return {
            'theme': ui.time_slider._current_theme,
            'labels': {str(lbl): bool(v.get()) for lbl, v in toggles.items()},
        }

class MultiSessionManager:
    """Multi-session navigation, scale, and data binding for CCGReviewUI."""

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        ui._session_any_mode: bool = False
        ui._any_expanded_group_tags: set = set()
        ui._any_pair_handle_list: list = []
        ui._png_sess_slug: str = ""

    def _compute_session_scale(self):
        """Return (ymin, ymax) unified across all pairs and segments in this key."""
        ui = self._ui
        ymin, ymax = 0.0, 0.0
        nd_key = ui.key.nd()
        res_key = getattr(ui, "_res_key", "lo")
        if res_key == "hi":
            cd = (ui.cd._ccg_highres.get(nd_key)
                  if hasattr(ui.cd, "_ccg_highres") else None)
            if cd is None:
                cd = ui.ccg_data
        else:
            cd = (ui.cd._ccg.get(nd_key)
                  if hasattr(ui.cd, "_ccg") else ui.ccg_data)
        all_seg = ui.n_segments
        for ref_tgt in ui.all_inds:
            ref, tgt = int(ref_tgt[0]), int(ref_tgt[1])
            for seg in range(ui.n_segments):
                try:
                    ccg_raw = cd.ccg[seg, ref, tgt, :]
                    ccg_null_raw = (cd.ccg_null[seg, ref, tgt, :]
                                    if getattr(cd, "ccg_null", None) is not None else None)
                    ymin, ymax = ui._accumulate_ylim(
                        ccg_raw, ccg_null_raw, ref, tgt, seg, False, ymin, ymax
                    )
                except Exception:
                    pass
            ccg_raw = np.sum(cd.ccg[:, ref, tgt, :], axis=0)
            ccg_null_raw = (np.sum(cd.ccg_null[:, ref, tgt, :], axis=0)
                            if cd.ccg_null is not None else None)
            try:
                ymin, ymax = ui._accumulate_ylim(
                    ccg_raw, ccg_null_raw, ref, tgt, all_seg, False, ymin, ymax
                )
            except Exception:
                pass
        return (ymin, ymax * 1.1 if ymax > 0 else 1.0)

    def _on_session_scale_toggle(self):
        ui = self._ui
        if ui.center_container.norm_panel._sess_scale.get():
            ui._same_scale_mode = 'session'
            ui.center_container.norm_panel._pair_scale.set(False)   # mutual exclusion
            ui._session_scale_cache = None  # force recompute
        else:
            ui._same_scale_mode = None
        ui._refresh()

    # ------------------------------------------------------------------
    # On-demand jitter
    # ------------------------------------------------------------------

    def _sanitize_sess_slug(self, sess: str) -> str:
        s = re.sub(r'[^\w.\-]+', '_', str(sess))[:48]
        return s or 'sess'

    def _all_nd_keys(self) -> list:
        """Return unique nd_keys (one per session) across the dataset.

        Prefers ``cd._ccg`` (keyed by nd_keys directly) so that ALL sessions
        are represented even when some have no significant pairs in ``cd.data``.
        Falls back to enumerating ``cd.data`` if ``_ccg`` is unavailable.
        """
        ui = self._ui
        seen, seen_str = [], set()
        # Primary source: _ccg is keyed by nd_keys, one per session
        ccg_source = getattr(ui.cd, '_ccg', None) or {}
        for nk in ccg_source.keys():
            s = str(nk)
            if s not in seen_str:
                seen.append(nk)
                seen_str.add(s)
        # Secondary: pick up any sessions present only in cd.data
        for k in ui.cd.data.keys():
            nk = k.nd()
            s = str(nk)
            if s not in seen_str:
                seen.append(nk)
                seen_str.add(s)
        # Tertiary: if neither _ccg nor cd.data are populated (lazy-load case),
        # enumerate from the neuron-dataset edge_times which is always available
        if not seen:
            for nk in getattr(getattr(ui.cd, 'nd', None), 'edge_times', {}).keys():
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
        ui = self._ui
        keys = self._all_nd_keys()
        return keys[1:] if keys and keys[0] is _ALL_SESSION_MARKER else keys

    def _type_key_for_nd(self, nd_key):
        """Pick a data Key matching *nd_key* and the current type label (any-mode)."""
        ui = self._ui
        lbl = ui._type_label(ui.key)
        matches = [k for k in ui.cd.data.keys()
                   if k.nd() == nd_key and ui._type_label(k) == lbl]
        return matches[0] if matches else None

    def _nd_key_for_session_str(self, sess_str: str):
        ui = self._ui
        for nk in ui._real_nd_keys_ordered():
            if str(getattr(nk, 'session', nk)) == sess_str:
                return nk
        return None

    def _any_conn_type_sort_key(self, key):
        """Order connection types: E before I; pyr→pyr before pyr→inter, etc."""
        ui = self._ui
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
        return (ep, ct_key, epoch, ui._type_label(key))

    def _available_type_keys_any(self) -> list:
        """One representative Key per distinct type label across all sessions."""
        ui = self._ui
        by_lbl: dict = {}
        keys_sorted = sorted(
            ui.cd.data.keys(),
            key=lambda k: (str(getattr(k, 'session', '')), ui._type_label(k)))
        for k in keys_sorted:
            lbl = ui._type_label(k)
            by_lbl.setdefault(lbl, k)
        return sorted(by_lbl.values(), key=self._any_conn_type_sort_key)

    def _any_rebuild_pair_handles(self):
        """Rebuild ``_any_pair_handle_list`` in tag header order × session order."""
        ui = self._ui
        if not getattr(ui, '_session_any_mode', False):
            ui._any_pair_handle_list = []
            return
        handles: list[tuple] = []
        for gname in ui._any_group_header_names():
            handles.extend(ui._any_iter_pairs_for_group(gname))
        ui._any_pair_handle_list = handles

    def _any_load_deleted_aggregate(self):
        ui = self._ui
        lbl = ui._type_label(ui.key)
        deleted: set = set()
        for k in ui.cd.data.keys():
            if ui._type_label(k) != lbl:
                continue
            ptr = ui.cd.data.get(k)
            valid = ui._all_inds_set_for_ptr(ptr)
            raw = set(ui._pair_deleted_store.get(str(k), set())) & valid
            sess = str(k.session)
            for r, c in raw:
                deleted.add((sess, int(r), int(c)))
        for p in deleted:
            ui._sel_data.set_state(p, 'del')

    def _flush_any_deleted_to_stores(self):
        ui = self._ui
        if not getattr(ui, '_session_any_mode', False):
            return
        lbl = ui._type_label(ui.key)
        by_key: dict[str, set] = _defaultdict(set)
        for trip in ui.deleted_inds:
            s, r, t = trip[0], int(trip[1]), int(trip[2])
            for k in ui.cd.data.keys():
                if ui._type_label(k) != lbl or str(k.session) != s:
                    continue
                by_key[str(k)].add((r, t))
        for ks, pairs in by_key.items():
            ui._pair_deleted_store[ks] = set(pairs)

    def _enter_all_session_mode(self):
        """Switch UI to virtual ``All`` session (collapsed group tags; lazy expand)."""
        ui = self._ui
        ui._session_all_mode = True
        ui._any_expanded_group_tags = set()
        ui._any_pair_handle_list = []
        ui._png_sess_slug = ''
        prev_lbl = ui._type_label(ui.key)
        ui._type_keys_list = self._available_type_keys_any()
        type_labels = [ui._type_label(k) for k in ui._type_keys_list]
        ui._type_combo['values'] = type_labels
        if not ui._type_keys_list:
            messagebox.showwarning("All sessions", "No connection types in dataset.")
            ui._session_all_mode = False
            try:
                ui._session_var.set(ui._session_label(ui.key.nd()))
            except Exception:
                pass
            return
        if prev_lbl in type_labels:
            ui.key = ui._type_keys_list[type_labels.index(prev_lbl)]
        else:
            ui.key = ui._type_keys_list[0]
            ui._type_var.set(type_labels[0])
        ui._bind_context_to_type_key(ui.key)
        self._any_load_deleted_aggregate()
        ui.current_pair_idx = 0
        ui.current_segment = ui.n_segments
        ui.segment_combo['values'] = ui.ccg_ptr.segment_names + [_ALL_SEGS]
        ui.segment_var.set(_ALL_SEGS)
        ui._bind_custom_segments_to_session(str(ui.key.session))
        # Default sort: sort-by-tag when entering All mode
        try:
            lp = ui.left_container.left_panel
            if not lp._sort_by_tag.get():
                lp._sort_by_tag.set(True)
                lp._sort_selected.set(False)
                lp._sort_by_mean.set(False)
                lp._sort_by_min_p.set(False)
        except Exception:
            pass

    def _exit_all_session_mode(self):
        """Leave ``All`` mode (flush pointers/stores first if entering from multi-save)."""
        ui = self._ui
        self._flush_any_selections_to_pointers()
        self._flush_any_deleted_to_stores()
        ui._session_all_mode = False
        ui._any_expanded_group_tags = set()
        ui._any_pair_handle_list = []
        ui._png_sess_slug = ''
        # ``selected_inds`` was (session, ref, tgt) triples; single-session code expects
        # (ref, tgt) only — reload from the bound pointer before _switch_key / autosave.
        try:
            ptr = ui.cd.data.get(ui.key) if getattr(ui, 'key', None) is not None else None
            _avail = set(map(tuple, ui.all_inds))
            _sel = (set(map(tuple, ptr.selected_inds))
                    if ptr is not None and getattr(ptr, 'selected_inds', None) is not None
                    else set())
            _del = set(ui._pair_deleted_store.get(str(ui.key), set())) & _avail
            ui._sel_data.populate(_avail, selected=_sel, deleted=_del)
        except Exception:
            pass
        try:
            ui._bind_custom_segments_to_session(str(ui.key.session))
        except Exception:
            pass

    def _bind_context_to_type_key(self, tk):
        """Point ccg_pointer / ccg_data / neurons at *tk* without touching triple sets."""
        ui = self._ui
        ptr = ui.cd.data.get(tk)
        nd_key = tk.nd()
        ui.key = tk
        ui.ccg_ptr = ptr
        if ptr is None:
            ui.ccg_data = None
            ui.neurons = None
            ui.n_segments = 0
            return
        if (getattr(ui, '_highres_mode', False)
                and hasattr(ui.cd, '_ccg_highres')
                and ui.cd._ccg_highres.get(nd_key) is not None):
            ui.ccg_data = ui.cd._ccg_highres[nd_key]
        else:
            ui.ccg_data = ui.cd._ccg.get(nd_key)
        try:
            ui.neurons = (ui.cd.nd.data[nd_key]
                            if getattr(ui.cd, 'nd', None) is not None else None)
        except KeyError:
            ui.neurons = None
        ui.n_segments = ptr.n_segments

    def _sync_any_plot_context(self, row_idx: int):
        """Bind ``ccg_*`` to the ``Key`` for ``_any_pair_handle_list[row_idx]``."""
        ui = self._ui
        if not getattr(ui, '_session_any_mode', False):
            return
        hl = getattr(ui, '_any_pair_handle_list', None) or []
        if row_idx < 0 or row_idx >= len(hl):
            return
        ckey, _r, _t = hl[row_idx]
        sess = str(ckey.session)
        ui._png_sess_slug = self._sanitize_sess_slug(sess)
        prev_sess = str(getattr(ui.key, 'session', '') or '')
        if (ui.key == ckey and ui.ccg_data is not None
                and getattr(ui.ccg_ptr, 'inds2', None) is not None):
            ui._bind_custom_segments_to_session(sess)
            ui._clamp_current_segment_for_session()
            try:
                ui._update_segment_label()
            except Exception:
                pass
            return
        # Save current segment name before switching so it can be restored by name
        _saved_seg_name = None
        try:
            seg = int(ui.current_segment)
            if seg == ui.n_segments:
                _saved_seg_name = _ALL_SEGS
            elif 0 <= seg < len(ui.ccg_ptr.segment_names):
                _saved_seg_name = ui.ccg_ptr.segment_names[seg]
        except Exception:
            pass

        if prev_sess != sess:
            # Custom CCG data are session-specific: show that session's segment chips.
            ui._bind_custom_segments_to_session(sess)
        ui._bind_context_to_type_key(ckey)
        ui._clamp_current_segment_for_session()

        # Restore segment by name so switching pairs doesn't reset the segment
        if _saved_seg_name is not None:
            try:
                if _saved_seg_name == _ALL_SEGS:
                    ui.current_segment = ui.n_segments
                elif _saved_seg_name in ui.ccg_ptr.segment_names:
                    ui.current_segment = ui.ccg_ptr.segment_names.index(_saved_seg_name)
                ui._clamp_current_segment_for_session()
            except Exception:
                pass

        try:
            if getattr(ui, 'segment_combo', None) is not None:
                ui.segment_combo['values'] = ui.ccg_ptr.segment_names + [_ALL_SEGS]
            ui._build_sig_chips()
            ui.jitter_controller.load_from_cd()
            ui._update_segment_label()
        except Exception:
            pass

    def _available_type_keys(self, nd_key) -> list:
        ui = self._ui
        if nd_key is _ALL_SESSION_MARKER:
            return self._available_type_keys_any()
        nd_session = nd_key.session
        return [k for k in ui.cd.data.keys() if k.nd().session == nd_session]

    def _on_session_change(self, event):
        ui = self._ui
        idx = ui._session_combo.current()
        if idx < 0 or idx >= len(ui._nd_keys_list):
            return
        nd_key = ui._nd_keys_list[idx]
        cur_any = getattr(ui, '_session_any_mode', False)
        new_any = nd_key is _ALL_SESSION_MARKER

        if new_any and cur_any:
            return

        if not new_any and not cur_any:
            cur_nd = ui.key.nd()
            cur_sess = getattr(cur_nd, 'session', None)
            new_sess = getattr(nd_key, 'session', None)
            if (cur_sess is not None and new_sess is not None
                    and str(cur_sess) == str(new_sess)):
                return

        ui._sel_mgr._autosave_current()

        def _revert_session_combo():
            cur_nd = ui.key.nd()
            ui._session_var.set(
                ui._session_label(_ALL_SESSION_MARKER) if cur_any
                else ui._session_label(cur_nd))

        if ui._custom_mgr._custom_ccg_has_unsaved():
            r = messagebox.askyesnocancel(
                "Unsaved custom CCGs",
                "Unsaved custom segments will be lost when switching sessions.\n\n"
                "Save them now? (No = discard, Cancel = stay)")
            if r is None:
                _revert_session_combo()
                return
            if r:
                ui.time_slider._save_custom_ccg()
                if ui._custom_mgr._custom_ccg_has_unsaved():
                    messagebox.showwarning(
                        "Custom CCGs not saved",
                        "Some custom segments are still unsaved. Session switch was cancelled.")
                    _revert_session_combo()
                    return
        try:
            ui._stacked_segments.clear()
        except Exception:
            ui._stacked_segments = set()

        if new_any:
            ui._enter_all_session_mode()
            if not getattr(ui, '_session_all_mode', False):
                ui._session_var.set(ui._session_label(ui.key.nd()))
                return
            ui._refresh_after_key_switch()
            ui.refresh_lists()
            ui.network_panel.draw()
            return

        # Leaving ``any`` → concrete session
        if cur_any:
            try:
                ui._autosave_all_sessions_for_current_type()
            except Exception as exc:
                print(f"[CCGReviewUI] multi-session autosave: {exc}")
            try:
                if ui._sel_data._groups:
                    ui._save_groups_export()
            except Exception:
                traceback.print_exc()
            ui._exit_all_session_mode()

        def _do_switch():
            type_keys = ui._available_type_keys(nd_key)
            ui._type_keys_list = type_keys
            type_labels = [ui._type_label(k) for k in type_keys]
            ui._type_combo['values'] = type_labels
            if not type_keys:
                return
            current_lbl = ui._type_label(ui.key)
            if current_lbl in type_labels:
                new_key = type_keys[type_labels.index(current_lbl)]
            else:
                new_key = type_keys[0]
                ui._type_var.set(type_labels[0])
            if ui._switch_key(new_key):
                ui._refresh_after_key_switch()
                ui._autoload_session_latest()
                ui.refresh_lists()
                ui.network_panel.draw()

        ui._ensure_session_loaded(nd_key, on_loaded=_do_switch)

    def _ensure_session_loaded(self, nd_key, on_loaded):
        """Call on_loaded() immediately if raw CCG data for nd_key is present.

        Otherwise show a modal "Loading dataset…" dialog, run cd.get_ccg(nd_key)
        in a background thread, then dismiss and call on_loaded().
        """
        ui = self._ui
        if nd_key in getattr(ui.cd, '_ccg', {}):
            getattr(ui.cd, '_touch_session', lambda k: None)(nd_key)
            on_loaded()
            return
        dlg = tk.Toplevel(ui.root)
        dlg.title("Loading")
        dlg.geometry("300x80")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.protocol('WM_DELETE_WINDOW', lambda: None)
        ttk.Label(dlg, text="Loading dataset…", anchor='center').pack(
            expand=True, fill='both', padx=20, pady=20)
        ui.root.update_idletasks()

        result = {}

        def _worker():
            try:
                ui.cd.get_ccg(nd_key=nd_key)
                limit = getattr(ui, '_ccg_memory_limit_gb', 8.0)
                getattr(ui.cd, '_check_memory_and_evict', lambda l: None)(limit)
            except Exception as ex:
                result['error'] = str(ex)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        def _poll():
            if t.is_alive():
                ui.root.after(200, _poll)
            else:
                dlg.destroy()
                if 'error' in result:
                    messagebox.showerror("Load error", result['error'],
                                         parent=ui.root)
                else:
                    on_loaded()

        ui.root.after(200, _poll)

    def _ensure_highres_loaded(self, nd_key, on_loaded):
        """Call on_loaded() if high-res CCG for nd_key is in memory.

        Otherwise trigger cd.load_highres(nd_key) in a background thread with
        a modal "Loading hi-res…" dialog, then call on_loaded().
        """
        ui = self._ui
        if getattr(ui.cd, '_ccg_highres', {}).get(nd_key) is not None:
            on_loaded()
            return
        dlg = tk.Toplevel(ui.root)
        dlg.title("Loading")
        dlg.geometry("300x80")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.protocol('WM_DELETE_WINDOW', lambda: None)
        ttk.Label(dlg, text="Loading hi-res CCG…", anchor='center').pack(
            expand=True, fill='both', padx=20, pady=20)
        ui.root.update_idletasks()

        result = {}

        def _worker():
            try:
                ui.cd.load_highres(nd_key=nd_key)
                limit = getattr(ui, '_ccg_memory_limit_gb', 8.0)
                getattr(ui.cd, '_check_memory_and_evict', lambda l: None)(limit)
            except Exception as ex:
                result['error'] = str(ex)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        def _poll():
            if t.is_alive():
                ui.root.after(200, _poll)
            else:
                dlg.destroy()
                if 'error' in result:
                    messagebox.showerror("Hi-res load error", result['error'],
                                         parent=ui.root)
                else:
                    on_loaded()

        ui.root.after(200, _poll)

    def _session_obj_for_nd_key(self, nd_key):
        ui = self._ui
        sessions = getattr(getattr(ui.cd, 'nd', None), '_sessions', None)
        if sessions is None:
            return None
        if not isinstance(sessions, (list, tuple)):
            sessions = [sessions]
        nd = getattr(ui.cd, 'nd', None)
        for s in sessions:
            try:
                if nd is not None and nd._short_session_name(s) == getattr(nd_key, 'session', None):
                    return s
            except Exception:
                continue
        return None

    def _session_wall_clock_extent_for_key(self, key) -> tuple[float, float]:
        """Wall-clock [start, stop] for *key*'s session (same time base as ``Neurons`` / edge_times)."""
        ui = self._ui
        sb = ui._segment_bounds_for_key(key)
        if sb:
            return ui._segment_bounds_time_extent(sb)
        ptr = ui.cd.data.get(key) if getattr(ui, 'cd', None) is not None else None
        if ptr is None or getattr(ptr, 'edge_times', None) is None:
            return (0.0, max(1.0, float(ui.time_slider._total_sec)))
        et = ptr.edge_times
        t = 0.0
        for _, row in et.iterrows():
            dur = float(row['effective_time_hours']) * 3600.0
            t += dur
        return (0.0, t if t > 0 else 1.0)

    def _iter_type_keys_for_all_sessions(self):
        ui = self._ui
        lbl = ui._type_label(ui.key)
        out = []
        for nk in ui._real_nd_keys_ordered():
            tk_ = ui._type_key_for_nd(nk)
            if tk_ is not None and ui._type_label(tk_) == lbl:
                out.append(tk_)
        return out

    def _any_sync_selection_from_universe(self):
        """Any mode: all pairs in expanded tags are selected; Available stays empty."""
        ui = self._ui
        hl = getattr(ui, '_any_pair_handle_list', None) or []
        sel = {(str(ckey.session), int(r), int(t)) for ckey, r, t in hl}
        ui._sel_data.populate(sel, selected=sel, deleted=set())

    def _flush_any_selections_to_pointers(self):
        ui = self._ui
        if not getattr(ui, '_session_any_mode', False):
            return
        lbl = ui._type_label(ui.key)
        by_sess: dict[str, list[tuple[int, int]]] = _defaultdict(list)
        for trip in ui.selected_inds:
            if len(trip) < 3:
                continue
            by_sess[trip[0]].append((int(trip[1]), int(trip[2])))
        for k in ui.cd.data.keys():
            if ui._type_label(k) != lbl:
                continue
            sess = str(k.session)
            arr = by_sess.get(sess)
            ptr = ui.cd.data[k]
            ptr.selected_inds = (
                np.array(sorted(arr), dtype=int) if arr else None)

class PNGCacheManager:
    """PNG cache path, clearing, and rendering for CCGReviewUI."""

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        ui.tmp_dir = str(_Path(__file__).resolve().parents[2] / "images" / "tmp")
        os.makedirs(ui.tmp_dir, exist_ok=True)

    def _clear_all_png_cache(self):
        ui = self._ui
        ui._pregen_cancel = True  # stop any in-progress pre-generation
        ui._terminate_pregen_proc()
        for _fn in os.listdir(ui.tmp_dir):
            if _fn.endswith('.png'):
                try: os.remove(os.path.join(ui.tmp_dir, _fn))
                except OSError: pass

    # ------------------------------------------------------------------
    # PNG pre-generation  (runs as an independent subprocess)
    # ------------------------------------------------------------------

    def _png_path(self, inds, segment, _render_cfg=None, _hires_override=None) -> str:
        """Return the disk-cache path for a PNG.

        Cache config set + state matches → path encodes (pair, seg, norm, alpha, res, scale, jitter), no sig suffix.
        Cache config set + state mismatch → short ``_rt_`` path (always overwritten, never reused).
        No cache config → full sig state encoded (legacy, one file per display config).
        """
        ui = self._ui
        if ui._is_custom_segment(segment):
            ci = ui._custom_seg_index(segment)
            cs_list = getattr(ui, '_custom_segments', []) or []
            if 0 <= ci < len(cs_list):
                cs = cs_list[ci]
                seg_name = f"custom{ci}_{cs['name']}_{cs['t0']:.2f}_{cs['t1']:.2f}"
                seg_name = seg_name.replace(' ', '_').replace(':', '-')
            else:
                seg_name = _ALL_SEGS.replace(' ', '_')
        elif segment == ui.n_segments:
            seg_name = _ALL_SEGS.replace(' ', '_')
        else:
            sn = ui.ccg_ptr.segment_names if ui.ccg_ptr is not None else []
            if 0 <= segment < len(sn):
                seg_name = sn[segment]
            else:
                seg_name = _ALL_SEGS.replace(' ', '_')
        _norms = (_render_cfg.get('active_norms') if _render_cfg else None)
        norm_key = ('_'.join(sorted(_norms)) if _norms
                    else ('_'.join(sorted(n.name for n in ui.active_norms))
                          if ui.active_norms else 'raw'))
        _alpha = (_render_cfg.get('active_alpha') if _render_cfg else None)
        alpha_key = ''
        if ui.ccg_data is not None and ui.ccg_data.pval_corrected is not None:
            alpha_key = f'_a{(_alpha if _alpha is not None else ui.active_alpha):.3f}'
        _hires = _hires_override if _hires_override is not None else getattr(ui, '_highres_mode', False)
        res_key = '_hi' if _hires else '_lo'
        scale_key = {'pair': '_ssp', 'session': '_sss'}.get(
            getattr(ui, '_same_scale_mode', None), '')
        _jrk = 'hi' if _hires else 'lo'
        j_key = '_j' if ui._jitter_cache.get(
            (int(inds[0]), int(inds[1]), _jrk, ui._jitter_seg(segment))) is not None else ''

        # Cache configuration determines path style
        _dk = '_dk' if ui.theme.dark else ''
        _ext_key = (f'_ext{int(ui._acg_var_get("_extend_ms", 0))}'
                    if ui._acg_var_get('_extend_enable', False) else '')
        if ui._cache_config is not None:
            if _render_cfg is not None or ui._display_matches_cache_config():
                # Canonical cached path — no sig encoding (only one config)
                _sp = (f"{ui._png_sess_slug}_" if getattr(ui, '_session_any_mode', False)
                       and getattr(ui, '_png_sess_slug', '') else '')
                return os.path.join(
                    ui.tmp_dir,
                    get_png_filename(
                        int(inds[0]), int(inds[1]), seg_name, norm_key,
                        alpha_key, res_key, scale_key, j_key, _ext_key, _dk, _sp))
            else:
                # Real-time path — one file per (pair, seg, res), always overwritten
                _sp = (f"{ui._png_sess_slug}_" if getattr(ui, '_session_any_mode', False)
                       and getattr(ui, '_png_sess_slug', '') else '')
                return os.path.join(
                    ui.tmp_dir,
                    f"{_sp}_rt_{int(inds[0])}_{int(inds[1])}_{seg_name}{res_key}{_dk}.png")

        # Legacy mode (no cache config): encode full sig state
        sig_key = ''
        _m = ui.center_container.baseline_panel._conn_str_method.get()
        sig_bits = (
            (_m if _m != 'conv' else '') +
            ('cs' if ui.center_container.cs_panel._conn_str_show.get() else '') +
            ('p' if ui._sig('conv_p') else '') +
            ('c' if ui._sig('conv_pc') else '') +
            ('tw' if ui._sig('test_window') else '') +
            ('jc' if ui._sig('jitter_pc') else '') +
            ('ar' if ui._acg_var_get('_acg_ref', False) else '') +
            ('at' if ui._acg_var_get('_acg_tgt', False) else '') +
            ('dcr' if bool(ui._acg_var_get('_acg_deconv_ref', False)) else '') +
            ('dct' if bool(ui._acg_var_get('_acg_deconv_tgt', False)) else '') +
            ('wp' if _hires and ui._acg_var_get('_peak_wf', False) else '') +
            (f'asr{ui._acg_var_get("_acg_yscale_ref", 1.0):.1f}'
             if ui._acg_var_get('_acg_ref', False)
                or (_hires and ui._acg_var_get('_peak_wf', False)) else '') +
            (f'ast{ui._acg_var_get("_acg_yscale_tgt", 1.0):.1f}'
             if ui._acg_var_get('_acg_tgt', False) else '') +
            ('am' if ui._acg_var_get('_acg_match_ccg', False) else '') +
            ('nc' if not ui._acg_var_get('_ccg_show', True) else '') +
            ('nb' if not ui._acg_var_get('_baseline_show', True) else '') +
            ('lc' if ui.center_container.correlogram_panel._line_ccg.get() else '') +
            ('lb' if ui.center_container.correlogram_panel._line_baseline.get() else '') +
            ('lr' if ui.center_container.correlogram_panel._line_ref.get() else '') +
            ('lt' if ui.center_container.correlogram_panel._line_tgt.get() else '') +
            ('lp' if (_hires and ui._acg_var_get('_peak_wf', False)
                      and ui.center_container.correlogram_panel._line_peak_wf.get()) else '') +
            ('lj' if ui.center_container.correlogram_panel._line_jitter.get() else '') +
            (f'ext{int(ui._acg_var_get("_extend_ms", 0))}'
             if ui._acg_var_get('_extend_enable', False) else ''))
        if sig_bits:
            sig_key = f'_s{sig_bits}'
        _sp = (f"{ui._png_sess_slug}_" if getattr(ui, '_session_any_mode', False)
               and getattr(ui, '_png_sess_slug', '') else '')
        return os.path.join(
            ui.tmp_dir,
            f"{_sp}pair_{int(inds[0])}_{int(inds[1])}_{seg_name}_{norm_key}"
            f"{alpha_key}{res_key}{scale_key}{j_key}{sig_key}{_dk}.png")

    def _render_png_with_res(self, inds, segment, highres: bool) -> str:
        """Render a PNG at a specific resolution without changing persistent state."""
        ui = self._ui
        nd_key = ui.key.nd()
        if highres:
            data = getattr(ui.cd, '_ccg_highres', {}).get(nd_key)
        else:
            data = getattr(ui.cd, '_ccg', {}).get(nd_key)
        if data is None:
            data = ui.ccg_data
        old_mode = ui._highres_mode
        old_data = ui.ccg_data
        ui._highres_mode = highres
        ui.ccg_data = data
        try:
            path = ui._plot_mgr._render_png(inds, segment, highres=highres)
        finally:
            ui._highres_mode = old_mode
            ui.ccg_data = old_data
        return path

    def _purge_tmp_png_cache(self, days: int = 3, max_gb: float = 1.5):
        """Delete PNGs in tmp_dir older than *days* days; also evict oldest if total > max_gb."""
        ui = self._ui
        purge_png_dir(getattr(ui, 'tmp_dir', ''), days=days, max_gb=max_gb)

class PregenController:
    """Pregeneration subprocess management for CCGReviewUI."""

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        self._runner = BackgroundTaskRunner(max_queue=1, use_result_queue=False)
        ui._pregen_cancel = False
        ui._pregen_priority = None

    def _terminate_pregen_proc(self):
        """Terminate a running pre-gen subprocess, if any."""
        self._runner.terminate()

    def _pregen_job_payload(self, cfg: dict) -> dict:
        """Build the JSON job dict for the pre-gen subprocess."""
        ui = self._ui
        nd_key = ui.key.nd()
        has_highres = (hasattr(ui.cd, '_ccg_highres')
                       and ui.cd._ccg_highres.get(nd_key) is not None)
        ccg_lo_path = os.path.expanduser(ui.cd._ccgdata_path()) + '.hkl'
        ccg_hi_path = (os.path.expanduser(ui.cd.highres_save_path()) + '.hkl'
                       if has_highres else None)

        neurons_fr    = None
        neurons_shank = None
        edge_times    = None
        if ui.neurons is not None:
            fr = getattr(ui.neurons, 'firing_rate', None)
            if fr is not None:
                neurons_fr = [float(x) for x in fr]
            neurons_shank = [int(x) for x in ui.neurons.shank_ids]
        # edge_times for TIME_SPAN norm
        et_df = getattr(ui.ccg_ptr, 'edge_times', None)
        if et_df is not None:
            try:
                edge_times = [float(et_df.iloc[s]['effective_time_hours'])
                              for s in range(ui.n_segments)]
            except Exception:
                edge_times = None

        return {
            'nd_key':          str(nd_key),
            'ccg_lo_path':     ccg_lo_path,
            'ccg_hi_path':     ccg_hi_path,
            'has_highres':     has_highres,
            'n_segments':      ui.n_segments,
            'segment_names':   ui.ccg_ptr.segment_names,
            'pairs':           [list(map(int, p)) for p in ui.all_inds],
            'tmp_dir':         ui.tmp_dir,
            'cache_config':    cfg,
            'neurons_firing_rate': neurons_fr,
            'neurons_shank_ids':   neurons_shank,
            'edge_times':      edge_times,
        }

    def _launch_pregen_subprocess(self, cfg: dict, status_var=None, priority: str = 'user'):
        """Write job file and launch pre_generate_images.py as an independent subprocess.

        priority='user': preempts any running auto task; runs immediately.
        priority='auto': skipped silently if any pregen is already running.
        """
        ui = self._ui
        if priority == 'auto':
            if self._runner.is_running():
                return
        else:
            if ui._pregen_priority == 'auto':
                self._terminate_pregen_proc()
            elif self._runner.is_running():
                self._terminate_pregen_proc()

        ui._pregen_priority = priority
        job = self._pregen_job_payload(cfg)
        job_path = os.path.join(ui.tmp_dir, '_pregen_job.json')
        with open(job_path, 'w', encoding='utf-8') as fh:
            json.dump(job, fh)

        script = str(_Path(__file__).parent / 'pre_generate_images.py')
        proc = subprocess.Popen([sys.executable, script, job_path])
        if priority == 'user':
            print(f"[CCGReviewUI] pre-gen subprocess started (pid {proc.pid})")
        if status_var is not None:
            status_var.set("Generating…")
        self._runner._proc = proc
        self._runner.start_polling(
            ui.root, interval_ms=1000,
            on_done=lambda _task, _result: self._on_pregen_done(status_var))

    def _on_pregen_done(self, status_var):
        """Called by BackgroundTaskRunner when pre-gen subprocess exits."""
        ui = self._ui
        if status_var is not None:
            status_var.set("Idle")
        if ui._pregen_priority == 'user':
            print("[CCGReviewUI] pre-gen subprocess finished")
        ui._pregen_priority = None

    def _pregen_png_cache(self):
        """Launch background pre-gen subprocess for all pairs × segments."""
        ui = self._ui
        ui._pregen_cancel = True   # keep for on-demand render cancellation
        ui._pregen_cancel = False
        if ui._cache_config is not None:
            cfg = dict(ui._cache_config)
        else:
            cfg = ui._settings_mgr._current_display_config()
        ui._pregen_ctrl._launch_pregen_subprocess(cfg)

    def _start_pregen_with_defaults(self, status_var=None):
        """Launch pre-gen subprocess using the saved cache configuration (user-requested)."""
        ui = self._ui
        if ui._cache_config is None:
            messagebox.showinfo(
                "Pre-gen",
                "No cache configuration set.\n\n"
                "Go to Settings → Cache Configuration and click\n"
                "\"Capture current settings\" to define the one\n"
                "display state that will be saved to disk cache.",
                parent=ui.root)
            return
        ui._pregen_ctrl._launch_pregen_subprocess(dict(ui._cache_config), status_var=status_var,
                                                   priority='user')

    # ------------------------------------------------------------------
    # Segment filter
    # ------------------------------------------------------------------

class ConnectionStrengthManager:
    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        ui._conn_strength_cache: LRUCache = LRUCache(256)

    # ── Class-level constants ────────────────────────────────
    _CS_METHOD_DESCRIPTIONS = {
        'conv':   "Conv baseline: smoothed null (EranConv)",
        'tailed': "Tailed: ACG deconvolution, tail-bin baseline",
        'global': "Global: max bin outside test window as baseline",
        'jitter': "Jitter: surrogate spike baseline",
    }

    _ADAPTIVE_TW_GROUPS = ('msconn', 'widems', '2peakms')
    _ADAPTIVE_TW_MIN_LAG = -1e-3   # -1 ms
    _ADAPTIVE_TW_MAX_LAG =  1e-3   #  1 ms

    def _rebuild_cs_pval_row(self):
        """Repopulate row 3 of the Baseline & CS panel based on current method."""
        panel = self._ui.center_container.baseline_panel
        for w in panel._cs_pval_row.winfo_children():
            w.destroy()
        method = panel._conn_str_method.get()
        if method == 'conv':
            ttk.Checkbutton(panel._cs_pval_row, text="p",
                            variable=panel._sig_conv_p,
                            command=self._ui._on_sig_toggle).pack(side=tk.LEFT, padx=2)
            ttk.Checkbutton(panel._cs_pval_row, text="p-corrected",
                            variable=panel._sig_conv_pc,
                            command=self._ui._on_sig_toggle).pack(side=tk.LEFT, padx=2)
        elif method == 'jitter':
            cb = ttk.Checkbutton(panel._cs_pval_row, text="p-corrected",
                                 variable=panel._sig_jitter_pc,
                                 command=self._ui._on_sig_toggle)
            cb.pack(side=tk.LEFT, padx=2)
            cb.state(['disabled'])
            self._ui._sig_jitter_pc_cb = cb
        ttk.Label(panel._cs_pval_row,
                  text=self.__class__._CS_METHOD_DESCRIPTIONS.get(method, ''),
                  font=('Arial', 8), foreground='#555').pack(side=tk.LEFT, padx=6)

    def _refresh_conn_str(self, *, clear_cache=False, rebuild_pval_row=False, redraw_network=False):
        """Common refresh path for conn-strength setting changes."""
        ui = self._ui
        if rebuild_pval_row:
            ui._cs_mgr._rebuild_cs_pval_row()
        if clear_cache:
            ui._conn_strength_cache.clear()
        ui._refresh()
        ui._update_conn_str_label()
        if redraw_network:
            ui.network_panel.draw()

    def _on_baseline_method_change(self):
        """Called when the Baseline radio button changes."""
        self._refresh_conn_str(clear_cache=True, rebuild_pval_row=True)

    def _on_conn_str_toggle(self):
        """CS overlay toggled — PNG must be re-rendered with/without green bars."""
        self._refresh_conn_str()

    def _on_adaptive_tw_toggle(self):
        """Adaptive test window toggled — clear caches and re-render."""
        self._refresh_conn_str(clear_cache=True, redraw_network=True)

    def _fmt_cs_value(self, v):
        """Format a CS float for display; respects non-negative clamp toggle."""
        if v is None:
            return "n/a"
        x = float(v)
        if self._ui.center_container.cs_panel._conn_str_nonneg.get():
            x = max(x, 0.0)
        return f"{x:.2f}" if abs(x) >= 1000 else f"{x:.3g}"

    def _get_stg_cs(self, ref, tgt, seg, hr, method, eff_min_lag, eff_max_lag):
        """Return formatted STG conn-strength string for (ref, tgt, seg, hr)."""
        ui = self._ui
        # ACG deconvolution active → CS/baseline are display-derived
        if (hasattr(ui, '_acg_deconv_ref') and hasattr(ui, '_acg_deconv_tgt')
                and (ui.center_container.correlogram_panel._acg_deconv_ref.get()
                     or ui.center_container.correlogram_panel._acg_deconv_tgt.get())):
            try:
                resk = 'hi' if bool(hr) else 'lo'
                tmp = ui._display_pair_temp.get((ref, tgt, int(seg), resk))
                if tmp is not None and tmp.get("method") == method:
                    return self._fmt_cs_value(tmp.get("cs_val"))
            except Exception:
                pass
        return self._fmt_cs_value(self._get_cs_val(ref, tgt, seg, method, hr))

    def _get_jbsi_cs(self, ref, tgt, seg, hr, nd_key, eff_min_lag, eff_max_lag):
        """Return formatted JBSI conn-strength string for (ref, tgt, seg, hr)."""
        ui = self._ui
        if hr:
            cd_res = (ui.cd._ccg_highres.get(nd_key)
                      if hasattr(ui.cd, '_ccg_highres') else None)
            if cd_res is None:
                cd_res = ui.cd._ccg.get(nd_key) if hasattr(ui.cd, '_ccg') else None
        else:
            cd_res = ui.cd._ccg.get(nd_key) if hasattr(ui.cd, '_ccg') else None
        if cd_res is None:
            cd_res = ui.ccg_data
        if cd_res is None:
            return "n/a"

        d = ui._resolve_segment_data(ref, tgt, seg, highres=hr,
                                     include_pval=False, _cd=cd_res)
        real_ccg = d.get('ccg_raw')
        if real_ccg is None:
            return "n/a"

        # Jitter mean baseline: lo-res only; zero-fill when not available
        j_avg = None
        if not hr:
            _jseg = ui._jitter_seg(seg)
            entry = ui._jitter_cache.get((ref, tgt, 'lo', _jseg))
            if entry is None and _jseg is not None:
                entry = ui._jitter_cache.get((ref, tgt, 'lo', None))
            if entry is not None:
                j_avg = entry[0]
        if j_avg is None:
            j_avg = np.zeros_like(real_ccg, dtype=float)

        # Firing rates: segment-specific if available, else whole-session
        # n1 inside compute_jbsi = min(fr_ref, fr_tgt) — the lower-firing neuron
        fr_ref = fr_tgt = None
        if (nd_key is not None and ui.cd.nd is not None
                and seg != ui.n_segments and not ui._is_custom_segment(seg)):
            seg_fr = ui.cd.nd.segment_firing_rates.get(nd_key)
            if (seg_fr is not None and seg < seg_fr.shape[0]
                    and ref < seg_fr.shape[1] and tgt < seg_fr.shape[1]):
                fr_ref = float(seg_fr[seg, ref])
                fr_tgt = float(seg_fr[seg, tgt])
        if fr_ref is None or fr_tgt is None:
            fr = getattr(ui.neurons, 'firing_rate', None)
            if fr is not None and ref < len(fr) and tgt < len(fr):
                fr_ref = float(fr[ref])
                fr_tgt = float(fr[tgt])
        if fr_ref is None or fr_tgt is None:
            return "n/a"

        try:
            lo_cd = ui.cd._ccg.get(nd_key) if hasattr(ui.cd, '_ccg') else ui.ccg_data
            jscale = float(JitterConfig(ccg=lo_cd.conf, njitter=1).jscale)
        except Exception:
            jscale = 5e-3

        bin_size_res = float(cd_res.conf.bin_size)
        jbsi = compute_jbsi(
            real_ccg=real_ccg,
            j_ccg_avg=j_avg,
            fr_ref=fr_ref,
            fr_tgt=fr_tgt,
            bin_size=bin_size_res,
            jscale=jscale,
        )
        n_bins = len(jbsi)
        bin_size_eff = cd_res.conf.duration / (n_bins - 1) if n_bins > 1 else bin_size_res
        center = n_bins // 2
        lo = max(0, center + int(eff_min_lag / bin_size_eff))
        hi_bin = min(n_bins, center + int(eff_max_lag / bin_size_eff))
        return self._fmt_cs_value(float(np.sum(jbsi[lo:hi_bin])))

    def _get_cs(self, ref, tgt, seg, hr, method, metric, nd_key, eff_min_lag, eff_max_lag):
        """Dispatch to STG or JBSI CS computation; return formatted string."""
        if metric == 'STG':
            return self._get_stg_cs(ref, tgt, seg, hr, method, eff_min_lag, eff_max_lag)
        return self._get_jbsi_cs(ref, tgt, seg, hr, nd_key, eff_min_lag, eff_max_lag)

    def _update_conn_str_label(self):
        """Always show CS value regardless of overlay toggle."""
        ui = self._ui
        if not hasattr(ui, 'center_container'):
            return
        method = ui.center_container.baseline_panel._conn_str_method.get()
        try:
            inds = ui._current_inds()
            if inds is None:
                ui.center_container.cs_panel._conn_str_label.config(text="CS: —")
                return
            ref, tgt = int(inds[0]), int(inds[1])
            seg = ui.current_segment
            eff_min_lag, eff_max_lag = ui._effective_lags(ref, tgt)
            metric = ui.center_container.cs_panel._conn_str_metric.get()
            nd_key = ui.key.nd() if ui.key else None

            def _cs(hr):
                return self._get_cs(ref, tgt, seg, hr, method, metric, nd_key,
                                    eff_min_lag, eff_max_lag)

            if ui._lo_hi_mode:
                ui.center_container.cs_panel._conn_str_label.config(
                    text=f"CS: lo|hi = {_cs(False)}|{_cs(True)}")
            else:
                ui.center_container.cs_panel._conn_str_label.config(
                    text=f"CS: {_cs(ui._highres_mode)}")
        except Exception:
            ui.center_container.cs_panel._conn_str_label.config(text="CS: err")

    def _get_cs_val(self, ref: int, tgt: int, seg, method: str, highres: bool):
        """Return raw CS value (float or None) from cache, computing if absent."""
        ui = self._ui
        eff_min_lag, eff_max_lag = ui._effective_lags(ref, tgt)
        k = ui._cs_cache_key(ref, tgt, seg, method, highres, eff_min_lag, eff_max_lag)
        if k not in ui._conn_strength_cache:
            ui._compute_pair_conn_strength(ref, tgt, seg, highres=highres)
        _cs_entry = ui._conn_strength_cache.get(k)
        v, _ = _cs_entry if _cs_entry is not None else (None, None)
        return v
    def _cs_annotation_lines(self, ref: int, tgt: int, seg, highres: bool,
                              print_stg: bool, print_jbsi: bool) -> list:
        """Return ["STG: val"] and/or ["JBSI: val"] for export annotation."""
        if not print_stg and not print_jbsi:
            return []
        ui = self._ui
        method = ui.center_container.baseline_panel._conn_str_method.get()
        def _fmt(v):
            if v is None: return "n/a"
            x = float(v)
            return f"{x:.2f}" if abs(x) >= 1000 else f"{x:.3g}"
        lines = []
        try:
            if print_stg:
                try: lines.append(f"STG: {_fmt(self._get_cs_val(ref, tgt, seg, method, highres))}")
                except Exception: lines.append("STG: n/a")
            if print_jbsi:
                try: lines.append(f"JBSI: {_fmt(self._get_cs_val(ref, tgt, seg, 'JBSI', highres))}")
                except Exception: lines.append("JBSI: n/a")
        except Exception:
            pass
        return lines

    def _update_conn_str_metric_availability(self):
        """Enable/disable CS metric radio buttons (e.g., JBSI needs jitter)."""
        ui = self._ui
        if not hasattr(ui, 'center_container'):
            return
        rbs = ui.center_container.cs_panel._cs_metric_rbs
        jb = rbs.get("JBSI")
        if jb is None:
            return
        # JBSI is always computable (uses j_avg=0 when no jitter data available)
        try:
            jb.state(["!disabled"])
        except Exception:
            pass

    def _cs_cache_key(self, ref: int, tgt: int, seg, method: str, highres: bool,
                      eff_min_lag, eff_max_lag) -> tuple:
        """Return the canonical cache key for a CS result.

        Effective lags are included so that switching the adaptive test window
        on/off (which changes eff_min_lag/eff_max_lag) always produces a
        distinct key even without clearing the cache.
        """
        return (ref, tgt, seg, method, highres, eff_min_lag, eff_max_lag)

    def _compute_pair_conn_strength(self, ref: int, tgt: int, seg, highres: bool = False):
        """Compute CS for all methods and populate _conn_strength_cache.

        Normalises once (no BASELINE — display-only), runs conv/global/tailed.
        Cache keys include TW lags to avoid stale adaptive-TW hits.
        """
        ui = self._ui
        cd = ui.ccg_data
        conf = cd.conf

        # ── Resolve raw arrays ────────────────────────────────────────────
        d = ui._resolve_segment_data(ref, tgt, seg, highres=highres,
                                        include_pval=False, include_acg=False)
        ccg_raw = d['ccg_raw']
        ccg_null_raw = d['ccg_null_raw']

        _custom_time_h = None
        if ui._is_custom_segment(seg):
            _ci = ui._custom_seg_index(seg)
            _custom_time_h = ui._custom_segments[_ci].get('total_time_hours')

        # ── Single normalisation pass (without BASELINE) ──────────────────
        # BASELINE norm is a display transform only; CS scalars must be
        # computed on the pre-subtraction signal so that global/tailed
        # baselines reflect the actual signal amplitude.
        norms_no_bl = ui.active_norms - {NormalizeBy.BASELINE}
        ccg, ccg_null = apply_norms_to_ccg(
            ccg_raw, ccg_null_raw, ref, tgt, seg,
            norms_no_bl, ui.neurons, ui.cd.nd,
            ui.key.nd(), ui.n_segments, ui._is_custom_segment(seg),
            custom_time_hours=_custom_time_h)

        # ── Effective test-window lags (adaptive TW aware) ────────────────
        eff_min_lag, eff_max_lag = ui._effective_lags(ref, tgt)
        _kw = dict(min_lag_override=eff_min_lag, max_lag_override=eff_max_lag)

        # ── conv ─────────────────────────────────────────────────────────
        cs, bl = compute_pair_conn_strength_1d(ccg, ccg_null, conf, 'conv', **_kw)
        ui._conn_strength_cache.put(
            ui._cs_cache_key(ref, tgt, seg, 'conv', highres, eff_min_lag, eff_max_lag),
            (cs, bl))

        # ── global (max outside detection window) ─────────────────────────
        cs, bl = compute_pair_conn_strength_1d(ccg, ccg_null, conf, 'global', **_kw)
        ui._conn_strength_cache.put(
            ui._cs_cache_key(ref, tgt, seg, 'global', highres, eff_min_lag, eff_max_lag),
            (cs, bl))

        # ── tailed (ACG deconvolution + tail baseline) ────────────────────
        try:
            d_acg = ui._resolve_segment_data(ref, tgt, seg, highres=highres,
                                                include_pval=False, include_acg=True)
            acg_ref = d_acg['acg_ref'].copy().astype(float)
            acg_tgt = d_acg['acg_tgt'].copy().astype(float)
            nspks_ref = max(float(np.sum(acg_ref)), 1.0)
            nspks_tgt = max(float(np.sum(acg_tgt)), 1.0)
            cs, bl = compute_pair_conn_strength_1d(
                ccg, ccg_null, conf, 'tailed',
                acg_ref=acg_ref, acg_tgt=acg_tgt,
                nspks_ref=nspks_ref, nspks_tgt=nspks_tgt, **_kw)
            ui._conn_strength_cache.put(
                ui._cs_cache_key(ref, tgt, seg, 'tailed', highres, eff_min_lag, eff_max_lag),
                (cs, bl))
        except Exception as e:
            print(f"[CCGReviewUI] Tailed CS failed for ({ref},{tgt}): {e}")
            ui._conn_strength_cache.put(
                ui._cs_cache_key(ref, tgt, seg, 'tailed', highres, eff_min_lag, eff_max_lag),
                (None, None))

        # ── JBSI ─────────────────────────────────────────────────────────
        try:
            nd_key = ui.key.nd() if ui.key else None
            if highres:
                cd_res = (ui.cd._ccg_highres.get(nd_key)
                          if hasattr(ui.cd, '_ccg_highres') else None) or ui.ccg_data
            else:
                cd_res = (ui.cd._ccg.get(nd_key)
                          if hasattr(ui.cd, '_ccg') else None) or ui.ccg_data
            if cd_res is not None:
                d_jbsi = ui._resolve_segment_data(ref, tgt, seg, highres=highres,
                                                    include_pval=False, _cd=cd_res)
                real_ccg = d_jbsi.get('ccg_raw')
                if real_ccg is not None:
                    fr_ref = fr_tgt = None
                    if (nd_key is not None and ui.cd.nd is not None
                            and seg != ui.n_segments and not ui._is_custom_segment(seg)):
                        seg_fr = ui.cd.nd.segment_firing_rates.get(nd_key)
                        if (seg_fr is not None and seg < seg_fr.shape[0]
                                and ref < seg_fr.shape[1] and tgt < seg_fr.shape[1]):
                            fr_ref = float(seg_fr[seg, ref])
                            fr_tgt = float(seg_fr[seg, tgt])
                    if fr_ref is None or fr_tgt is None:
                        fr = getattr(ui.neurons, 'firing_rate', None)
                        if fr is not None and ref < len(fr) and tgt < len(fr):
                            fr_ref = float(fr[ref])
                            fr_tgt = float(fr[tgt])
                    if fr_ref is not None and fr_tgt is not None:
                        try:
                            lo_cd = (ui.cd._ccg.get(nd_key)
                                     if hasattr(ui.cd, '_ccg') else None) or ui.ccg_data
                            jscale = float(JitterConfig(ccg=lo_cd.conf, njitter=1).jscale)
                        except Exception:
                            jscale = 5e-3
                        j_avg = np.zeros_like(real_ccg, dtype=float)
                        bin_size_res = float(cd_res.conf.bin_size)
                        jbsi_arr = compute_jbsi(
                            real_ccg=real_ccg,
                            j_ccg_avg=j_avg,
                            fr_ref=fr_ref,
                            fr_tgt=fr_tgt,
                            bin_size=bin_size_res,
                            jscale=jscale,
                        )
                        n_bins = len(jbsi_arr)
                        bin_size_eff = (cd_res.conf.duration / (n_bins - 1)
                                        if n_bins > 1 else bin_size_res)
                        center = n_bins // 2
                        _lo = (max(0, center + int(eff_min_lag / bin_size_eff))
                               if eff_min_lag is not None else 0)
                        _hi = (min(n_bins, center + int(eff_max_lag / bin_size_eff))
                               if eff_max_lag is not None else n_bins)
                        jbsi_cs = float(np.sum(jbsi_arr[_lo:_hi]))
                        ui._conn_strength_cache.put(
                            ui._cs_cache_key(ref, tgt, seg, 'JBSI', highres,
                                               eff_min_lag, eff_max_lag),
                            (jbsi_cs, None))
        except Exception as _jbsi_err:
            print(f"[CCGReviewUI] JBSI CS failed for ({ref},{tgt}): {_jbsi_err}")
            ui._conn_strength_cache.put(
                ui._cs_cache_key(ref, tgt, seg, 'JBSI', highres, eff_min_lag, eff_max_lag),
                (None, None))

        method = ui.center_container.baseline_panel._conn_str_method.get()
        _final = ui._conn_strength_cache.get(
            ui._cs_cache_key(ref, tgt, seg, method, highres, eff_min_lag, eff_max_lag))
        return _final if _final is not None else (None, None)

    def _update_global_baseline_availability(self):
        """Enable/disable the Global radio button.

        Global is always available whenever CCG data exists.
        """
        ui = self._ui
        rb = ui.center_container.baseline_panel._global_rb
        if rb is None:
            return
        available = (ui.ccg_data is not None)
        try:
            rb.state(['!disabled'] if available else ['disabled'])
        except Exception:
            pass
        if not available and ui.center_container.baseline_panel._conn_str_method.get() == 'global':
            ui.center_container.baseline_panel._conn_str_method.set('conv')
            ui._cs_mgr._rebuild_cs_pval_row()

    def _pair_qualifies_for_adaptive_tw(self, ref: int, tgt: int) -> bool:
        ui = self._ui
        pair = (ref, tgt)
        return any(pair in ui._group_pairs(g) for g in self.__class__._ADAPTIVE_TW_GROUPS)

    def _effective_lags(self, ref: int, tgt: int):
        """Return (min_lag, max_lag) accounting for adaptive test window."""
        ui = self._ui
        _eo = getattr(ui, '_export_overrides', None)
        _adaptive = (_eo.get('adaptive_tw_export', False) if _eo
                     else ui.center_container.baseline_panel._adaptive_tw.get())
        if (_adaptive and ui._pair_qualifies_for_adaptive_tw(ref, tgt)):
            return self.__class__._ADAPTIVE_TW_MIN_LAG, self.__class__._ADAPTIVE_TW_MAX_LAG
        conf = ui.ccg_data.conf if ui.ccg_data else None
        if conf is None:
            return None, None
        return conf.min_lag, conf.max_lag

    def _update_adaptive_tw_availability(self):
        """Enable/disable the adaptive TW button based on current pair's group tags."""
        ui = self._ui
        btn = ui.center_container.baseline_panel._adaptive_tw_btn
        if btn is None:
            return
        inds = ui._current_inds()
        qualifies = (inds is not None
                     and ui._pair_qualifies_for_adaptive_tw(int(inds[0]), int(inds[1])))
        try:
            btn.state(['!disabled'] if qualifies else ['disabled'])
        except Exception:
            pass
        if not qualifies and ui.center_container.baseline_panel._adaptive_tw.get():
            ui.center_container.baseline_panel._adaptive_tw.set(False)

class PairAnalysisManager:
    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui

    def _on_sig_toggle(self):
        """Clear PNG cache and redraw (vars are read live via _sig())."""
        ui = self._ui
        ui._refresh()

    def _is_significant(self, ref: int, tgt: int, seg: int) -> bool:
        """Return True if (ref, tgt) is significant in segment *seg*.

        When seg == ui.n_segments ("All"), returns True if
        the pair is significant in ANY real segment.

        Significance is always derived from the **low-res** CCGData so that
        segment chips stay green when the user switches to high-res mode.
        Priority: jitter (segment-aware) → pval_corrected + active_alpha
        → stored significant array.
        """
        ui = self._ui
        # Custom segments: use stored pval_corrected
        if ui._is_custom_segment(seg):
            ci = ui._custom_seg_index(seg)
            cs_list = getattr(ui, '_custom_segments', [])
            if 0 <= ci < len(cs_list):
                cs = cs_list[ci]
                pc = cs.get('pval_corrected')
                if pc is not None:
                    conf = ui.ccg_data.conf
                    lb = getattr(conf, 'min_lag_bin', None)
                    ub = getattr(conf, 'max_lag_bin', None)
                    if lb is not None and ub is not None:
                        try:
                            return bool(pc[0, ref, tgt, lb:ub].min()
                                        <= ui.active_alpha)
                        except (IndexError, ValueError):
                            pass
            return False

        if seg == ui.n_segments:
            return any(ui._is_significant(ref, tgt, s)
                       for s in range(ui.n_segments))

        j = ui.cd._jitter.get(ui.key) if hasattr(ui.cd, '_jitter') else None
        if j is not None:
            inds = j.ccg_ptr.inds
            if inds is not None:
                if getattr(j.ccg_ptr, 'stored_by_segment', False):
                    mask = ((inds[:, 0] == seg) &
                            (inds[:, -2] == ref) & (inds[:, -1] == tgt))
                else:
                    mask = (inds[:, -2] == ref) & (inds[:, -1] == tgt)
                if mask.any():
                    return bool(j.j_sig[mask].any())
                return False

        # Always use low-res data for significance (high-res may lack pval arrays)
        cd = ui.cd._ccg.get(ui.key.nd()) if hasattr(ui.cd, '_ccg') else None
        if cd is None:
            cd = ui.ccg_data   # fallback
        if cd is not None and cd.pval_corrected is not None:
            conf = cd.conf
            lb = getattr(conf, 'min_lag_bin', None)
            ub = getattr(conf, 'max_lag_bin', None)
            if lb is not None and ub is not None:
                try:
                    return bool(
                        cd.pval_corrected[seg, ref, tgt, lb:ub].min()
                        <= ui.active_alpha)
                except (IndexError, ValueError):
                    pass
        if cd is not None and cd.significant is not None:
            try:
                return bool(cd.significant[seg, ref, tgt])
            except (IndexError, ValueError):
                pass
        return False

    def _compute_pair_scale(self, ref: int, tgt: int):
        """Return (ymin, ymax) unified across all segments for this pair.

        Includes:
        - all real segments
        - the summed 'All' view
        - all loaded custom segments (and their hi-res versions when available)

        Computed per-resolution and cached under (ref, tgt, res_key).
        """
        ui = self._ui
        nd_key = ui.key.nd()
        res_key = getattr(ui, "_res_key", "lo")
        if res_key == "hi":
            cd = (ui.cd._ccg_highres.get(nd_key)
                  if hasattr(ui.cd, "_ccg_highres") else None)
            if cd is None:
                cd = ui.ccg_data
        else:
            cd = (ui.cd._ccg.get(nd_key)
                  if hasattr(ui.cd, "_ccg") else ui.ccg_data)
        if cd is None:
            ui._pair_scale_cache.pop((ref, tgt, res_key))
            return None

        ymin, ymax = 0.0, 0.0

        # Real segments
        for seg in range(ui.n_segments):
            try:
                ccg_raw = cd.ccg[seg, ref, tgt, :]
                ccg_null_raw = (cd.ccg_null[seg, ref, tgt, :]
                                if getattr(cd, "ccg_null", None) is not None else None)
                ymin, ymax = ui._accumulate_ylim(
                    ccg_raw, ccg_null_raw, ref, tgt, seg, False, ymin, ymax
                )
            except Exception:
                continue

        # All segments summed
        try:
            all_seg = ui.n_segments
            ccg_raw = np.sum(cd.ccg[:, ref, tgt, :], axis=0)
            ccg_null_raw = (np.sum(cd.ccg_null[:, ref, tgt, :], axis=0)
                            if getattr(cd, "ccg_null", None) is not None else None)
            ymin, ymax = ui._accumulate_ylim(
                ccg_raw, ccg_null_raw, ref, tgt, all_seg, False, ymin, ymax
            )
        except Exception:
            pass

        # Custom segments (include both lo/hi versions depending on res_key)
        cs_list = getattr(ui, "_custom_segments", []) or []
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
                seg_idx = ui.n_segments + ci  # custom segment index in UI space
                ymin, ymax = ui._accumulate_ylim(
                    ccg_raw, ccg_null_raw, ref, tgt, seg_idx, True, ymin, ymax,
                    custom_time_h=cs.get("total_time_hours"),
                )
            except Exception:
                continue

        result = (ymin, ymax * 1.1 if ymax > 0 else 1.0)
        ui._pair_scale_cache.put((ref, tgt, res_key), result)
        return result

    def _finalize_normalization(self):
        ui = self._ui
        if not ui.active_norms:
            messagebox.showinfo("Normalize all", "No normalizations are toggled on.")
            return
        if ui.neurons is None and any(
                nm in (NormalizeBy.REF_FRATE, NormalizeBy.TARGET_FRATE)
                for nm in ui.active_norms):
            messagebox.showerror(
                "Normalize all",
                "Neuron data is unavailable — cannot normalize by firing rate.")
            return
        if not messagebox.askyesno(
                "Normalize all",
                f"Rewrite current normalization into the stored CCG arrays?\n\n"
                "This permanently modifies the in-memory dataset (ccg/ccg_null) and cannot be undone.\n"
                "Use this only if you want the data itself to become normalized, not just the display."):
            return
        cd = ui.ccg_data
        nd_key = ui.key.nd()
        edge_times = (ui.cd.nd.edge_times[nd_key]
                      if ui.cd.nd is not None else None)
        frates = (ui.cd.nd.segment_firing_rates[nd_key]
                  if ui.cd.nd is not None else None)
        for seg in range(cd.n_segment):
            for nm in ui.active_norms:
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
        for nm in ui.active_norms:
            if nm not in existing:
                existing.append(nm)
        cd.conf.normalize_methods = existing
        ui._clear_all_png_cache()
        for var in ui.norm_vars.values():
            var.set(False)
        ui.active_norms = set()
        ui.update_plot()
        messagebox.showinfo("Normalize all",
                            "Applied normalization to stored CCG data. Don't forget to Save Selections.")

    def _compute_stats_str(self) -> str:
        ui = self._ui
        n_sig = len(ui.all_inds)
        n_sel = len(ui.selected_inds)
        n_poss = None
        n_ref = None
        n_tgt = None
        ref_t = tgt_t = None
        if (ui.neurons is not None and
                ui.key is not None and
                ui.key.conn_type is not None):
            ref_t, tgt_t = ui.key.conn_type
            try:
                n_ref = ui.neurons.get_neuron_type(ref_t).n_neurons
                n_tgt = ui.neurons.get_neuron_type(tgt_t).n_neurons
                # Exclude self-self when ref and target types are the same
                n_poss = n_ref * (n_ref - 1) if ref_t == tgt_t else n_ref * n_tgt
            except Exception:
                pass

        if n_poss is not None and n_poss > 0:
            s = (f"Significant: {n_sig}/{n_poss}"
                 f"  Selected: {n_sel}/{n_poss}")
        else:
            s = f"Significant: {n_sig}  Selected: {n_sel}"
        if n_ref is not None:
            if ref_t == tgt_t:
                s += f"  {ref_t}: {n_ref}"
            else:
                s += f"  ref({ref_t}): {n_ref}  tgt({tgt_t}): {n_tgt}"
        return s

    def _update_sig_indicators(self, inds):
        """Color each segment chip green/gray based on significance."""
        ui = self._ui
        if not hasattr(ui, 'seg_sig_labels'):
            return
        ref, tgt = int(inds[0]), int(inds[1])
        cd = ui.ccg_data
        conf = cd.conf if cd is not None else None
        lb = getattr(conf, 'min_lag_bin', None)
        ub = getattr(conf, 'max_lag_bin', None)

        selected = getattr(ui, '_stacked_segments', set()) or set()
        _labels = ui.seg_sig_labels
        _items = _labels.items() if isinstance(_labels, dict) else enumerate(_labels)
        for chip_idx, lbl in _items:
            # Custom segment chips (after All chip)
            if chip_idx > ui.n_segments:
                ci = chip_idx - ui.n_segments - 1
                cs_list = getattr(ui, '_custom_segments', [])
                if 0 <= ci < len(cs_list):
                    cs = cs_list[ci]
                    pc = cs.get('pval_corrected')
                    if (pc is not None and lb is not None and ub is not None
                            and ref < pc.shape[1] and tgt < pc.shape[2]):
                        sig = bool(pc[0, ref, tgt, lb:ub].min()
                                   <= ui.active_alpha)
                    else:
                        sig = False
                    active = (ui.current_segment == chip_idx)
                    bg = ('#4CAF50' if active else '#90EE90') if sig \
                        else ('#FFD54F' if active else '#FFF9C4')
                else:
                    bg = '#FFF9C4'
                if chip_idx in selected:
                    lbl.config(bg='#BBDEFB', relief=tk.SUNKEN)
                else:
                    lbl.config(bg=bg, relief=tk.SUNKEN)
                continue

            is_all_chip = (chip_idx == ui.n_segments)
            if is_all_chip:
                sig = any(ui._is_significant(ref, tgt, s)
                          for s in range(ui.n_segments))
                active = (ui.current_segment == ui.n_segments)
            else:
                seg = chip_idx
                if (cd is not None and cd.pval_corrected is not None and
                        lb is not None and ub is not None and
                        seg < cd.pval_corrected.shape[0]):
                    sig = bool(
                        cd.pval_corrected[seg, ref, tgt, lb:ub].min()
                        <= ui.active_alpha)
                else:
                    sig = ui._is_significant(ref, tgt, seg)
                active = (seg == ui.current_segment)

            if sig:
                bg = '#4CAF50' if active else '#90EE90'
            else:
                bg = '#9E9E9E' if active else '#E0E0E0'
            if chip_idx in selected:
                lbl.config(bg='#BBDEFB', relief=tk.SUNKEN)
            else:
                lbl.config(bg=bg, relief=tk.RAISED if not active else tk.SUNKEN)

    def _resolve_segment_data(self, ref, tgt, segment, highres=None, include_pval=True, include_acg=False, _cd=None):
        """
        Retrieve raw CCG arrays for (ref, tgt, segment), handling custom/all-segs/normal.
        highres defaults to ui._highres_mode when None.
        Returns dict with keys: ccg_raw, ccg_null_raw, seg_label,
        and optionally pval, pval_corrected (include_pval=True),
        acg_ref, acg_tgt (include_acg=True).
        Pass _cd to override ui.ccg_data (e.g. from background threads).
        """
        ui = self._ui
        if highres is None:
            highres = ui._highres_mode
        cd = _cd if _cd is not None else ui.ccg_data
        is_custom = ui._is_custom_segment(segment)
        is_all = (segment == ui.n_segments)
        result = {}

        if is_custom:
            ci = ui._custom_seg_index(segment)
            cs = ui._custom_segments[ci]
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
            result['seg_label'] = ui.ccg_ptr.segment_names[segment]
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
        ui = self._ui
        if ui.neurons is None:
            return None
        try:
            extend_ms = int(extend_ms)
        except Exception:
            return None
        if extend_ms <= 0:
            return None

        resk = 'hi' if bool(highres) else 'lo'
        _sess_k = str(getattr(ui.key, 'session', '')) if ui.key is not None else ''
        key = (int(ref), int(tgt), int(segment), resk, int(extend_ms), float(bin_size_eff), _sess_k)
        if key in ui._extend_cache:
            return ui._extend_cache[key]

        # Segment time slicing (match jitter behavior)
        neurons_eff = ui.neurons
        seg_label_eff = None
        if ui._is_custom_segment(segment):
            # Custom segment: recompute on the custom window (and its saved brain-state filter).
            try:
                ci = ui._custom_seg_index(segment)
                cs = ui._custom_segments[ci]
                seg_label_eff = str(cs.get('name', 'custom'))
                t0 = float(cs['t0'])
                t1 = float(cs['t1'])
                neurons_eff = ui.neurons.time_slice(t_start=t0, t_stop=t1)
                fs = cs.get('filter_state') or {}
                labels = (fs.get('labels') if isinstance(fs, dict) else None) or {}
                # If labels are present and not all ON, reconstruct intervals.
                if labels and not all(bool(v) for v in labels.values()):
                    active_labels = {lbl for lbl, on in labels.items() if bool(on)}
                    if active_labels:
                        none_active = 'NONE' in active_labels
                        real_labels = active_labels - {'NONE'}
                        intervals = []
                        for s, e, lbl in ui.time_slider._epoch_bounds or []:
                            if lbl in real_labels:
                                s_clipped, e_clipped = max(float(s), t0), min(float(e), t1)
                                if e_clipped > s_clipped:
                                    intervals.append((s_clipped, e_clipped))
                        if none_active:
                            epoch_times = sorted(
                                (max(float(s), t0), min(float(e), t1))
                                for s, e, _ in (ui.time_slider._epoch_bounds or [])
                                if min(float(e), t1) > max(float(s), t0))
                            cursor = t0
                            for es, ee in epoch_times:
                                if es > cursor:
                                    intervals.append((cursor, es))
                                cursor = max(cursor, ee)
                            if cursor < t1:
                                intervals.append((cursor, t1))
                        if intervals:
                            neurons_eff = ui.time_slider._apply_brain_state_intervals(intervals, t0, t1)
            except Exception:
                neurons_eff = ui.neurons
        elif segment != ui.n_segments:
            try:
                et = ui.ccg_ptr.edge_times
                t0 = float(et.iloc[int(segment)]['start'])
                t1 = float(et.iloc[int(segment)]['stop'])
                neurons_eff = ui.neurons.time_slice(t_start=t0, t_stop=t1)
                seg_label_eff = str(ui.ccg_ptr.segment_names[int(segment)])
            except Exception:
                neurons_eff = ui.neurons
        else:
            seg_label_eff = _ALL_SEGS

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
            seg_label_eff = _ALL_SEGS if segment == ui.n_segments else f"seg{int(segment)}"

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
        ui._extend_cache[key] = out
        return out

    def _draw_waveforms(self):
        ui = self._ui
        if not ui._waveforms_visible or ui.neurons is None:
            return
        if ui.current_pair_idx >= len(ui.all_inds):
            return
        inds = ui.all_inds[ui.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        wf = getattr(ui.neurons, 'waveforms', None)
        shank_ids = getattr(ui.neurons, 'shank_ids', None)
        ui.wave_ax.clear()
        if wf is None or shank_ids is None:
            ui.wave_ax.text(0.5, 0.5, "No waveform data",
                              ha='center', va='center',
                              transform=ui.wave_ax.transAxes, fontsize=8)
            ui.wave_ax.axis('off')
        else:
            # Resolve probe geometry from NeuronsDatasetConfig so discarded
            # channels are mapped correctly (mirrors plot_ccg_figure logic).
            nd_conf = getattr(getattr(ui.cd, 'nd', None), '_conf', None)
            ch_per_shank = getattr(nd_conf, 'ch_per_shank', 16)
            recinfo = getattr(nd_conf, 'recinfo', None)
            skipped = getattr(recinfo, "skipped_channels", None)
            discarded = None if skipped is None else np.asarray(skipped, dtype=int)

            def get_filled_waveform(shank_id, wf_neuron):
                """Map compact waveform to full (ch_per_shank, n_samples) grid."""
                if wf_neuron.ndim == 1:
                    return np.tile(wf_neuron, (ch_per_shank, 1))
                if discarded is None or len(discarded) == 0:
                    return wf_neuron
                channel_ids = ch_per_shank * shank_id + np.arange(ch_per_shank)
                mask = ~np.isin(channel_ids, discarded)
                start = int(ch_per_shank * shank_id
                            - np.sum(discarded < ch_per_shank * shank_id))
                length = int(np.sum(mask))
                if length == 0 or start + length > wf_neuron.shape[0]:
                    return wf_neuron
                clean = np.full((ch_per_shank, wf_neuron.shape[-1]), np.nan)
                clean[mask] = wf_neuron[start:start + length]
                return clean

            ref_shank = int(shank_ids[ref])
            tgt_shank = int(shank_ids[tgt])
            ref_waveform = get_filled_waveform(ref_shank, wf[ref])
            tgt_waveform = get_filled_waveform(tgt_shank, wf[tgt])
            color = 'green' if ref_shank != tgt_shank else 'orange'
            peak_g = None
            pk = getattr(ui.neurons, 'peak_channels', None)
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
                ax=ui.wave_ax,
                ch_per_shank=ch_per_shank,
                discarded_channels=discarded,
                peak_channel_global=peak_g,
            )
        if ui.theme.dark:
            _bg, _fg = '#2b2b2b', 'white'
            ui.wave_fig.set_facecolor(_bg)
            ui.wave_ax.set_facecolor(_bg)
            ui.wave_ax.tick_params(colors=_fg)
            ui.wave_ax.xaxis.label.set_color(_fg)
            ui.wave_ax.yaxis.label.set_color(_fg)
            ui.wave_ax.title.set_color(_fg)
            for sp in ui.wave_ax.spines.values():
                sp.set_edgecolor('#666666')
        ui.wave_canvas.draw()

    def _theme_bounds_for_key(self, key):
        ui = self._ui
        theme = ui.time_slider._current_theme
        if theme == 'segments':
            ptr = ui.cd.data.get(key)
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
        sess_obj = ui._session_obj_for_nd_key(key.nd())
        if sess_obj is None:
            return None
        epoch = getattr(sess_obj, theme, None)
        if Epoch is None or epoch is None or not isinstance(epoch, Epoch) or epoch.n_epochs <= 0:
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


    def _pair_passes_main(self, ref: int, tgt: int, seg) -> bool:
        """
        Returns True if this pair passes the primary 'Main' template criterion.
        Uses EranConv p-value: any bin < alpha within [min_lag_bin, max_lag_bin].
        Falls back to True if no p-value data available.
        """
        ui = self._ui
        cd = ui.ccg_data
        if cd is None or cd.pval is None:
            return True  # no data to filter on
        is_custom = ui._is_custom_segment(seg)
        is_all = (seg == ui.n_segments)
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

    def _view_values(self, item):
        """Print values of a CCG data item to cell output."""
        ui = self._ui
        if ui.current_pair_idx >= len(ui.all_inds):
            print("[ViewValues] No pair selected"); return
        inds = ui.all_inds[ui.current_pair_idx]
        ref, tgt = int(inds[0]), int(inds[1])
        segment = ui.current_segment
        cd = ui.ccg_data
        if cd is None:
            print("[ViewValues] No CCG data available"); return

        # If ACG deconvolution is active, inform the user that displayed values
        # are derived (deconvolved) and should be interpreted accordingly.
        try:
            corr = ui.center_container.correlogram_panel
            dref = bool(corr._acg_deconv_ref.get())
            dtgt = bool(corr._acg_deconv_tgt.get())
            if dref or dtgt:
                mode = ("ref+tgt" if (dref and dtgt) else "ref" if dref else "tgt")
                print(f"[ViewValues] NOTE: ACG deconvolution is ON ({mode}); values reflect the displayed (deconvolved) CCG.")
        except Exception:
            pass

        is_all = (segment == ui.n_segments)
        is_custom = ui._is_custom_segment(segment)

        def _get(arr, r, c):
            """Extract values from ccg-shaped array [seg, r, c, bins]."""
            if arr is None:
                return None
            if is_all:
                return np.sum(arr[:, r, c, :], axis=0)
            elif is_custom:
                ci = ui._custom_seg_index(segment)
                cs = ui._custom_segments[ci]
                src = cs.get('ccg_hi') if ui._highres_mode and 'ccg_hi' in cs else cs.get('ccg')
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
                resk = 'hi' if bool(ui._highres_mode) else 'lo'
                tmp = ui._display_pair_temp.get((ref, tgt, int(segment), resk))
                if tmp is None:
                    # Force a render to populate display temp cache, then re-read.
                    ui._plot_mgr._render_png((ref, tgt), segment, highres=ui._highres_mode)
                    tmp = ui._display_pair_temp.get((ref, tgt, int(segment), resk))
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

class SimulationManager:
    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui

    def _sim_compute_correlogram(self, sim_neurons, bin_size, duration, conf):
        """Return (ccg_1d, null_1d, pval_1d, bin_size_eff) for one bin size."""
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
            shank_ids=(ref_nick, tgt_nick), inds=(0, 1),
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
        ui = self._ui
        cur_hi = 'highres' in label_var.get()
        label_var.set('Res: lowres' if cur_hi else 'Res: highres')
        if 'ccg_lo' not in sim_state:
            return 'break'
        sim_state['highres'] = not cur_hi
        ui._sim_redraw_plot(fig, ax, canvas, sim_state)
        return 'break'

    def _run_simulation(self, win, name_var, dur_var, unit_var,
                        noise_var, sync_var, delay_var,
                        sim_vars, fig, ax, canvas, sim_state, sim_res_label_var):
        """Simulate spike trains, compute CCG + EranConv, and plot result."""
        ui = self._ui
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
        ref_train = sim_generate_train(
            dur_s, params['ref']['rate'], noise_std,
            params['ref']['burst_rate'], params['ref']['n_bursts'],
            params['ref']['burst_interval'])
        tgt_train = sim_generate_train(
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
        conf = ui.ccg_data.conf if ui.ccg_data is not None else CCGConfig()
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

        for res, bin_size in (('lo', bin_lo), ('hi', bin_hi)):
            ccg, null, pval, bse = ui._sim_compute_correlogram(
                sim_neurons, bin_size, duration, conf)
            sim_state[f'ccg_{res}']          = ccg
            sim_state[f'null_{res}']         = null
            sim_state[f'pval_{res}']         = pval
            sim_state[f'bin_size_eff_{res}'] = bse
        sim_state['conf']    = conf
        sim_state['params']  = params
        sim_state['name']    = name_var.get()
        sim_state['highres'] = 'highres' in sim_res_label_var.get()

        ui._sim_redraw_plot(fig, ax, canvas, sim_state)

class UISetupManager:
    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        ui._panel_vars: dict = {}
        ui._waveforms_visible = False
        ui._stats_panel = None
        ui._sel_collapsed_headers: set = set()

    def setup_ui(self):
        ui = self._ui
        ui.root.geometry("1800x950")

        # ── Menubar ────────────────────────────────────────────────────
        menubar = tk.Menu(ui.root)
        ui.root.config(menu=menubar)
        ui._setup_mgr.setup_panels_menu(menubar)
        ui._group_mgr.setup_groups_menu(menubar)
        ui._sel_mgr.setup_file_menu(menubar)
        self.setup_modules_menu(menubar)
        ui.setup_settings_menu(menubar)
        ui.setup_help_menu(menubar)

        # ── Tool-strip row ─────────────────────────────────────────────
        ui._setup_mgr.setup_menu()

        # ── Group hotkeys bar (below tool-strip, hidden by default) ────
        ui._group_mgr.setup_group_hotkeys_bar()

        # ── Bottom bar (packed before main so it gets space first) ─────
        ui._setup_mgr.setup_bottom_panel()

        # ── Main area ──────────────────────────────────────────────────
        ui._main_frame = ttk.Frame(ui.root)
        ui._main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        # Time slider (full-width, hidden by default; packed before paned)
        ui.time_slider = TimeSliderPanel(ui._main_frame, ui)

        # Three-column PanedWindow
        ui._paned = ttk.PanedWindow(ui._main_frame, orient=tk.HORIZONTAL)
        ui._paned.pack(fill=tk.BOTH, expand=True)

        ui._left_frame = ttk.Frame(ui._paned, width=350)
        ui._center_frame = ttk.Frame(ui._paned)
        ui._right_frame = ttk.Frame(ui._paned, width=340)

        ui._paned.add(ui._left_frame, weight=0)
        ui._paned.add(ui._center_frame, weight=1)
        ui._paned.add(ui._right_frame, weight=0)

        ui._setup_mgr.setup_left_panel(ui._left_frame)
        self.setup_center_panel(ui._center_frame)
        ui._setup_mgr.setup_network_panel(ui._right_frame)

        # Apply saved panel states — panels are added above with their defaults;
        # call _toggle_panel_impl for any panel whose saved state differs so the
        # actual visibility matches what was persisted.
        _panel_defaults = {
            'Pair Selection': True, 'CCG': True, 'Probe Network': True,
            'Waveforms': False, 'Time Slider': False, 'Group Hotkeys': True,
        }
        for _pname, _pdefault in _panel_defaults.items():
            _saved_val = ui._panel_vars[_pname].get()
            if _saved_val != _pdefault:
                ui._setup_mgr._toggle_panel_impl(_pname)

        # Keyboard bindings (Control for Linux/Windows; Command for macOS)
        ui.root.bind('<Left>',      lambda e: ui.change_segment(-1))
        ui.root.bind('<Right>',     lambda e: ui.change_segment(1))
        for _key in ('<Control-r>', '<Command-r>'):
            ui.root.bind(_key, lambda e: ui._toggle_resolution())
        for _key in ('<Control-e>', '<Command-e>'):
            ui.root.bind(_key, lambda e: ui._on_ctrl_e())
        for _key in ('<Control-s>', '<Command-s>'):
            ui.root.bind_all(_key, lambda e: ui._sel_mgr._quick_save())
        for _key in ('<Control-f>', '<Command-f>'):
            ui.root.bind(_key, lambda e: ui.left_container.left_panel._search_toggle())
        for _key in ('<Control-b>', '<Command-b>'):
            ui.root.bind(_key, lambda e: ui._bookmark_toggle_current())
        for _key in ('<Control-z>', '<Command-z>'):
            ui.root.bind(_key, ui._undo)
        for _key in ('<Control-y>', '<Command-y>',
                      '<Control-Shift-z>', '<Command-Shift-z>',
                      '<Control-Shift-Z>', '<Command-Shift-Z>'):
            ui.root.bind(_key, ui._redo)
        for _del_key in ('<Control-Delete>', '<Control-BackSpace>',
                         '<Command-Delete>', '<Command-BackSpace>',
                         '<Meta-Delete>',    '<Meta-BackSpace>'):
            ui.root.bind(_del_key, lambda e: ui.left_container.left_panel._on_delete_pair())
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
                ui.left_container.left_panel._on_delete_pair()
                return
            if ks in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0'):
                ui._group_mgr._group_hotkey_handler(ks, advance=True)
            elif ks in ('KP_1', 'KP_2', 'KP_3', 'KP_4', 'KP_5',
                        'KP_6', 'KP_7', 'KP_8', 'KP_9'):
                ui._group_mgr._group_hotkey_handler(ks[-1], advance=True)
            elif ks == 'KP_0':
                ui._group_mgr._group_hotkey_handler('0', advance=True)
            elif ks in _SHIFT_DIGIT:
                # Shift+digit → no advance
                ui._group_mgr._group_hotkey_handler(_SHIFT_DIGIT[ks], advance=False)
            elif len(ks) == 1 and ks.islower():
                # Bare letter → assign + advance
                ui._group_mgr._group_hotkey_handler(ks, advance=True)
            elif len(ks) == 1 and ks.isupper():
                # Shift+letter → assign, no advance
                ui._group_mgr._group_hotkey_handler(ks.lower(), advance=False)

        ui.root.bind('<KeyPress>', _global_key_handler)
        # When holding Shift to apply multiple tags without advancing, advance
        # once when Shift is released.
        ui._shift_tag_pending_advance = False
        ui.root.bind('<KeyRelease-Shift_L>', lambda e: ui._on_shift_release_advance())
        ui.root.bind('<KeyRelease-Shift_R>', lambda e: ui._on_shift_release_advance())

        # Restore segment and sash position after the window is realized
        ui.root.after(200, ui._setup_mgr._restore_deferred_ui_state)

    def _restore_deferred_ui_state(self):
        """Restore state that can only be applied after the window is mapped."""
        ui = self._ui
        s = ui._ui_state_cache

        # Restore loaded custom CCGs first (so segment indices exist before we restore current_segment)
        try:
            ui._custom_mgr._restore_loaded_custom_ccgs_from_state()
        except Exception:
            pass

        # Restore sash position
        pane = getattr(ui, '_pair_list_pane', None)
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
                total = ui._n_total_segments()
                if 0 <= int(saved_seg) < total:
                    ui.current_segment = int(saved_seg)
                    ui._update_segment_label()
            except Exception:
                pass

        # Restore core display vars (baseline solid/outline, test window, etc.)
        disp = s.get('display_vars', None)
        if isinstance(disp, dict) and hasattr(ui, 'center_container'):
            try:
                cc = ui.center_container
                panels = [getattr(cc, p, None) for p in
                          ('correlogram_panel', 'baseline_panel', 'cs_panel',
                           'jitter_panel', 'acg_panel')]
                panels = [p for p in panels if p is not None]
                for attr, val in disp.items():
                    if val is None:
                        continue
                    # Search panels first (where most display vars live), then self
                    v = None
                    for panel in panels:
                        v = getattr(panel, attr, None)
                        if v is not None:
                            break
                    if v is None:
                        v = getattr(ui, attr, None)
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
            ui._highres_mode = bool(s.get('highres_mode', getattr(ui, '_highres_mode', False)))
        except Exception:
            pass
        try:
            ui._lo_hi_mode = bool(s.get('lo_hi_mode', getattr(ui, '_lo_hi_mode', False)))
        except Exception:
            pass
        # Ensure ccg_data matches restored resolution
        try:
            nd_key = ui.key.nd()
            if (getattr(ui, '_highres_mode', False)
                    and hasattr(ui.cd, '_ccg_highres')
                    and ui.cd._ccg_highres.get(nd_key) is not None):
                ui.ccg_data = ui.cd._ccg_highres.get(nd_key)
            else:
                ui.ccg_data = ui.cd._ccg.get(nd_key) if getattr(ui.cd, '_ccg', None) else ui.ccg_data
        except Exception:
            pass

        # Re-sync active_norms from restored norm_vars (they were set from saved state)
        if hasattr(ui, 'norm_vars'):
            ui.active_norms = {nm for nm, var in ui.norm_vars.items() if var.get()}

        # Finally, re-render so main-panel button states are reflected visually.
        try:
            ui.update_plot()
        except Exception:
            pass

    def setup_menu(self):
        ui = self._ui
        menu_frame = ttk.Frame(ui.root, relief=tk.RAISED, borderwidth=2)
        menu_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Session
        ttk.Label(menu_frame, text="Session:",
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(8, 2))
        nd_keys = ui._sess_mgr._all_nd_keys()
        ui._nd_keys_list = nd_keys
        session_labels = [ui._session_label(k) for k in nd_keys]
        ui._session_var = tk.StringVar(value=ui._session_label(ui.key.nd()))
        ui._session_combo = ttk.Combobox(
            menu_frame, textvariable=ui._session_var,
            values=session_labels, width=22, state='readonly')
        ui._session_combo.pack(side=tk.LEFT, padx=2)
        ui._session_combo.bind('<<ComboboxSelected>>', ui._sess_mgr._on_session_change)

        # Type — if the key has no conn_type, fall back to E pyr→pyr or first available
        ttk.Label(menu_frame, text="Type:",
                  font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(10, 2))
        type_keys = ui._available_type_keys(ui.key.nd())
        ui._type_keys_list = type_keys
        type_labels = [ui._type_label(k) for k in type_keys]
        if not getattr(ui.key, 'conn_type', None) and type_keys:
            # Key has no conn_type — pick E pyr→pyr, else first available
            preferred = [k for k in type_keys
                         if getattr(k, 'excitability', None) == 'E'
                         and getattr(k, 'conn_type', None) == ('pyr', 'pyr')]
            ui.key = (preferred or type_keys)[0]
            if ui.ccg_ptr is None:
                ui.ccg_ptr = ui.cd.data.get(ui.key)
        ui._type_var = tk.StringVar(value=ui._type_label(ui.key))
        ui._type_combo = ttk.Combobox(
            menu_frame, textvariable=ui._type_var,
            values=type_labels, width=18, state='readonly')
        ui._type_combo.pack(side=tk.LEFT, padx=2)
        ui._type_combo.bind('<<ComboboxSelected>>', ui._on_type_change)

        # Pre-gen button — warms PNG cache with canonical display defaults
        ui._pregen_btn = ttk.Button(
            menu_frame, text="⚡ Pre-gen", width=10,
            command=ui._pregen_ctrl._start_pregen_with_defaults)
        ui._pregen_btn.pack(side=tk.RIGHT, padx=(2, 8))


    def setup_left_panel(self, parent):
        ui = self._ui
        # Delegate to LeftPanelContainer (pair_selection_panel.py)
        ui.left_container = LeftPanelContainer(
            parent, ui._sel_data, ui, ui._ui_state_cache)
        ui.left_container.widget.pack(fill=tk.BOTH, expand=True)

        # ── Backward-compat aliases so existing CCGReviewUI code keeps working ──
        lp = ui.left_container.left_panel
        sp = ui.left_container.spike_pairs

        ui._left_notebook    = ui.left_container.widget
        ui.unselected_list   = lp.unselected_list
        ui.selected_list     = lp.selected_list
        ui._avail_label_var  = lp._avail_label    # LeftPanel uses _avail_label (no _var suffix)
        ui._sel_label_var    = lp._sel_label
        ui._clear_spec_btn   = lp._clear_spec_btn
        ui._pair_list_pane   = lp._pair_list_pane

        # Sort vars — LeftPanel drops the _var suffix; alias under old names
        ui._sort_selected_var = lp._sort_selected
        ui._sort_by_tag_var   = lp._sort_by_tag
        ui._sort_by_mean_var  = lp._sort_by_mean
        ui._sort_by_min_p_var = lp._sort_by_min_p

        # Search — old methods (_search_show/_hide/_go/_update) are dead code;
        # Ctrl+F is rebound below.  Keep minimal aliases for any residual refs.
        ui._search_frame   = lp.search_bar._frame
        ui._search_entry   = lp.search_bar._entry
        ui._search_var     = lp.search_bar._var
        ui._search_matches = lp.search_bar._matches

        # Spike pairs panel aliases (old code used ui._sa_*)
        ui._sa_tab       = sp._tab
        ui._sa_listbox   = sp._spike_pairs_listbox
        ui._sa_count_var = sp._spike_pairs_count
        ui._sa_tab_index = sp._spike_pairs_tab_index

    def setup_center_panel(self, parent):
        ui = self._ui
        ui.plot_title_var = tk.StringVar(value=ui._plot_mgr.get_plot_title())
        ui._plot_title_label = ttk.Label(
            parent, textvariable=ui.plot_title_var, font=('Arial', 11, 'bold'))
        ui._plot_title_label.pack(side=tk.TOP)

        # Vertical PanedWindow: top = CCG figure, bottom = control panels
        pw = tk.PanedWindow(parent, orient=tk.VERTICAL,
                            sashrelief=tk.RAISED, sashwidth=5,
                            bg='#CCCCCC')
        pw.pack(fill=tk.BOTH, expand=True)

        # Top pane: plot frame (CCGPlotPanel packs into here)
        plot_frame = ttk.Frame(pw)
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

        # Build all center panels via CenterPanelContainer
        ui.center_container = CenterPanelContainer(plot_frame, ctrl_frame, ui)
        cp = ui.center_container

        ui.fig      = cp.plot_panel.fig
        ui.canvas   = cp.plot_panel.canvas
        ui._plot_pw = cp.plot_panel._plot_pw

        # Initialise p-value row and CS metric availability
        ui._sig_jitter_pc_cb = None
        ui._cs_mgr._rebuild_cs_pval_row()
        ui._update_conn_str_metric_availability()

        # Waveforms panel (uses ui._plot_pw set above)
        ui.setup_waveforms_panel(ctrl_frame)

        # Significance chips — owned by SegmentChipsPanel (via CenterPanelContainer)
        ui.sig_frame = ui.center_container.seg_chips_panel.frame
        ui.center_container.seg_chips_panel.rebuild()

        pw.add(scroll_outer, minsize=30, stretch='never')

        # Hidden segment state (combo removed; segment chips handle navigation)
        _seg_var_init = (_ALL_SEGS if ui.current_segment >= ui.n_segments
                         else ui.ccg_ptr.segment_names[ui.current_segment])
        ui.segment_var = tk.StringVar(value=_seg_var_init)
        ui.segment_combo = ttk.Combobox(
            parent, textvariable=ui.segment_var,
            values=ui.ccg_ptr.segment_names + [_ALL_SEGS], width=14,
            state='readonly')
        ui.segment_combo.bind('<<ComboboxSelected>>', ui._on_segment_change)

        ui._render_engine = CCGRenderEngine(ui)
        ui.root.after(100, ui._deferred_initial_draw)

    def setup_network_panel(self, parent):
        ui = self._ui
        ui.network_panel = NetworkPanel(parent, ui)

    def setup_bottom_panel(self):
        ui = self._ui
        bottom_frame = ttk.Frame(ui.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)
        ttk.Button(bottom_frame, text="Save Selections",
                   command=ui._sel_mgr._quick_save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bottom_frame, text="Quit",
                   command=ui._on_close).pack(side=tk.RIGHT, padx=5)
        ui.stats_var = tk.StringVar(value=ui._compute_stats_str())
        ttk.Label(bottom_frame, textvariable=ui.stats_var,
                  font=('Courier', 9)).pack(side=tk.LEFT, padx=8)
        ui._pair_info_var = tk.StringVar(value="")
        ttk.Label(bottom_frame, textvariable=ui._pair_info_var,
                  font=('Courier', 9)).pack(side=tk.LEFT, padx=8)

    def _toggle_panel(self, name):
        """Show or hide a panel based on its BooleanVar."""
        ui = self._ui
        try:
            ui._setup_mgr._toggle_panel_impl(name)
            ui._save_ui_state()
        except Exception as ex:
            print(f"[CCGReviewUI] _toggle_panel error for {name!r}: {ex}")
            traceback.print_exc()

    def _toggle_panel_impl(self, name):
        """Inner implementation of panel toggling."""
        ui = self._ui
        show = ui._panel_vars[name].get()
        if name in ('Pair Selection', 'CCG', 'Probe Network'):
            frame_map = {
                'Pair Selection': ui._left_frame,
                'CCG':            ui._center_frame,
                'Probe Network':  ui._right_frame,
            }
            order   = ['Pair Selection', 'CCG', 'Probe Network']
            weights = {'Pair Selection': 0, 'CCG': 1, 'Probe Network': 0}
            frame = frame_map[name]
            if show:
                pos = sum(1 for n in order[:order.index(name)]
                          if ui._panel_vars[n].get())
                current_panes = ui._paned.panes()
                if str(frame) in current_panes:
                    pass  # already managed, nothing to do
                elif pos < len(current_panes):
                    ui._paned.insert(pos, frame, weight=weights[name])
                else:
                    ui._paned.add(frame, weight=weights[name])
            else:
                if str(frame) in ui._paned.panes():
                    ui._paned.forget(frame)
        elif name == 'Waveforms':
            ui._toggle_waveforms_panel()
        elif name == 'Time Slider':
            ui.time_slider.toggle()
        elif name == 'Group Hotkeys':
            if show:
                ui._refresh_hotkeys_bar()
                # Pack after the tool-strip (setup_menu frame), before the main area
                ui._hotkeys_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 2),
                                       before=ui._main_frame)
            else:
                ui._hotkeys_bar.pack_forget()

    def _toggle_waveforms_panel(self):
        ui = self._ui
        ui._waveforms_visible = ui._panel_vars['Waveforms'].get()
        if ui._waveforms_visible:
            ui._plot_pw.add(ui.wave_frame, stretch='never', width=280)
            ui._draw_waveforms()
        else:
            try:
                ui._plot_pw.forget(ui.wave_frame)
            except tk.TclError:
                pass

    def _finish_initial_draw(self):
        """Complete initialisation after data has been loaded; then draw."""
        ui = self._ui
        nd_key = ui.key.nd()
        ptr = ui.cd.data.get(ui.key)
        if ptr is None:
            # Key may not exist yet; pick first available key for this session
            type_keys = ui._available_type_keys(nd_key)
            if type_keys:
                ui.key = type_keys[0]
                ptr = ui.cd.data.get(ui.key)
        if ptr is None:
            messagebox.showerror("Load error",
                                 f"No data found for session {nd_key}",
                                 parent=ui.root)
            return
        ui.ccg_ptr = ptr
        ui.ccg_data = ui.cd._ccg.get(nd_key)
        try:
            ui.neurons = (ui.cd.nd.data[nd_key]
                            if getattr(ui.cd, 'nd', None) is not None else None)
        except KeyError:
            ui.neurons = None
        ui.n_segments = ui.ccg_ptr.n_segments
        # Restore selection state from pointer if available
        _sel = (set(map(tuple, ui.ccg_ptr.selected_inds))
                if hasattr(ui.ccg_ptr, 'selected_inds')
                and ui.ccg_ptr.selected_inds is not None else set())
        ui._pair_deleted_store.clear()
        ui._sel_data.populate(map(tuple, ui.all_inds), selected=_sel, deleted=set())
        # Load persisted jitter results for current session only (lazy — others load on demand)
        _jitter_nd = ui.key.nd()
        if (hasattr(ui.cd, 'load_jitter')
                and _jitter_nd not in getattr(ui.cd, '_jitter_results', {})):
            ui.cd.load_jitter(nd_key=_jitter_nd)
        ui.jitter_controller.load_from_cd()
        ui._post_load_refresh()
        # Startup: purge old history + tmp PNG cache, then start 15-min autosnapshot timer
        try:
            ui._sel_mgr._purge_history()
            ui._png_mgr._purge_tmp_png_cache(days=3)
        except Exception as exc:
            print(f"[CCGReviewUI] history purge failed: {exc}")
        ui._sel_mgr._schedule_autosnapshot()

    def setup_panels_menu(self, menubar):
        """Panels menu with checkbuttons for each panel."""
        ui = self._ui
        panels_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Panels", menu=panels_menu)
        ui._panels_menu = panels_menu
        panel_defaults = [
            ('Pair Selection',  True),
            ('CCG',             True),
            ('Probe Network',   True),
            ('Waveforms',       False),
            ('Time Slider',     False),
            ('Group Hotkeys',   True),
        ]
        saved = ui._ui_state_cache.get('panels', None)
        for name, default in panel_defaults:
            value = saved.get(name, default) if saved is not None else default
            var = tk.BooleanVar(value=value)
            ui._panel_vars[name] = var
            panels_menu.add_checkbutton(
                label=name, variable=var,
                command=lambda n=name: ui._setup_mgr._toggle_panel(n))
        panels_menu.add_separator()
        ts_menu = tk.Menu(panels_menu, tearoff=0)
        ts_menu.add_command(
            label="Refresh suggested custom CCGs",
            command=ui._custom_mgr._refresh_custom_ccg_suggestions,
        )
        ts_menu.add_command(
            label="Generate suggested custom CCGs",
            command=ui._custom_mgr._generate_suggested_custom_ccgs,
        )
        panels_menu.add_cascade(label="Time Slider Actions", menu=ts_menu)

    def setup_modules_menu(self, menubar):
        """Modules menu — Stats Tests, Jitter and Simulation sub-menus."""
        ui = self._ui
        modules_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Modules", menu=modules_menu)

        # Stats Tests — prepended first
        modules_menu.add_command(label="Stats Tests…", command=ui._open_stats_panel)

        # Jitter sub-menu
        jitter_menu = tk.Menu(modules_menu, tearoff=0)
        modules_menu.add_cascade(label="Jitter", menu=jitter_menu)
        jitter_menu.add_command(label="View queue…", command=ui._jitter_queue_dialog)
        jitter_menu.add_command(label="Clear queue",  command=ui._jitter_clear_queue)

        # Simulation sub-menu
        sim_menu = tk.Menu(modules_menu, tearoff=0)
        modules_menu.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(label="New simulation…", command=lambda: SimulationDialog.show(ui))

        # Tooltip-style help on hover
        ui._modules_menu = modules_menu
        ui._menu_tooltips = {
            'Stats Tests…': "Run statistical comparisons (t-tests) on connection strengths across groups/segments",
            'Jitter': "Significance test for a pairwise connection using surrogate data",
            'Simulation': "Simulate CCG of two random neurons with designated properties",
        }
        ui._menu_tooltip_win = None
        modules_menu.bind('<<MenuSelect>>', self._on_modules_menu_hover)

    def _on_modules_menu_hover(self, event):
        """Show tooltip for hovered Modules menu item."""
        ui = self._ui
        menu = ui._modules_menu
        try:
            idx = menu.index('active')
            if idx is not None:
                label = menu.entrycget(idx, 'label')
                tip = ui._menu_tooltips.get(label)
                if tip:
                    if ui._menu_tooltip_win is not None:
                        ui._menu_tooltip_win.destroy()
                    tw = tk.Toplevel(ui.root)
                    tw.wm_overrideredirect(True)
                    tw.wm_attributes('-topmost', True)
                    x = ui.root.winfo_pointerx() + 16
                    y = ui.root.winfo_pointery() + 10
                    tw.wm_geometry(f"+{x}+{y}")
                    lbl = tk.Label(tw, text=tip, background='#ffffe0',
                                   relief=tk.SOLID, borderwidth=1,
                                   font=('TkDefaultFont', 9), padx=6, pady=3)
                    lbl.pack()
                    ui._menu_tooltip_win = tw
                    return
        except (tk.TclError, ValueError):
            pass
        if ui._menu_tooltip_win is not None:
            ui._menu_tooltip_win.destroy()
            ui._menu_tooltip_win = None

    def _ccg_context_menu(self, event):
        """Right-click context menu on the CCG plot canvas."""
        ui = self._ui
        menu = tk.Menu(ui.root, tearoff=0)
        # Export actions
        menu.add_command(label="Export view as PNG…",
                         command=ui._export_current_view)
        menu.add_separator()
        for _label, _key in [("View CCG values",      'ccg'),
                              ("View ref ACG values",  'acg_ref'),
                              ("View tgt ACG values",  'acg_tgt'),
                              ("View baseline values", 'baseline'),
                              ("View p-values",        'pval')]:
            menu.add_command(label=_label, command=lambda k=_key: ui._view_values(k))
        menu.tk_popup(event.x_root, event.y_root)

    def _on_close(self):
        """Close the app.

        - Always prompts to save on exit (no change-detection).
        - Unsaved custom CCG segments prompt for permission to save.
        - Bookmarks are session-only and are cleared.
        """
        ui = self._ui
        ui._closing = True
        if ui._heartbeat_id is not None:
            try:
                ui.root.after_cancel(ui._heartbeat_id)
            except tk.TclError:
                pass
            ui._heartbeat_id = None
        if ui.jitter_controller._poll_id is not None:
            try:
                ui.root.after_cancel(ui.jitter_controller._poll_id)
            except tk.TclError:
                pass
            ui.jitter_controller._poll_id = None
        if ui.jitter_worker.is_running():
            ui.jitter_worker._runner.terminate()
        ui._jitter_pending.clear()
        # Custom segments: ask permission to save if any are unsaved
        if ui._custom_mgr._custom_ccg_has_unsaved():
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
                ui._closing = False
                ui._start_heartbeat()
                return
            if r:
                # On exit always save every session — no dialog
                _all_exit_segs = [
                    cs
                    for lst in (getattr(ui, '_custom_segments_by_session', None) or {}).values()
                    for cs in lst
                    if isinstance(cs, dict)
                ]
                if _all_exit_segs:
                    try:
                        ui._custom_mgr._save_custom_segment_objects(_all_exit_segs, show_saved_message=False)
                    except Exception as _exc:
                        print(f"[CCGReviewUI] on-exit custom CCG save error: {_exc}")
                if ui._custom_mgr._custom_ccg_has_unsaved():
                    messagebox.showwarning(
                        "Custom CCGs not saved",
                        "Some custom segments are still unsaved. Quit was cancelled.",
                    )
                    ui._closing = False
                    ui._start_heartbeat()
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
            ui._closing = False
            ui._start_heartbeat()
            return
        if r:
            # Unified save pathway
            ui._sel_mgr._autosave_current()
        ui._bookmarked_pairs.clear()
        ui.root.destroy()

    def _current_session_str(self):
        ui = self._ui
        return getattr(ui.key, 'session', 'sess')

class PlotManager:
    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        ui._together_pairs: list = []

    def update_plot(self):
        ui = self._ui
        try:
            if getattr(ui, '_session_any_mode', False):
                hl = getattr(ui, '_any_pair_handle_list', None) or []
                if ui.network_panel._focused_pair is None and ui.current_pair_idx < len(hl):
                    row_ck = hl[ui.current_pair_idx][0]
                    if ui.key != row_ck:
                        ui._sync_any_plot_context(ui.current_pair_idx)
                if ui.network_panel._focused_pair is None:
                    if ui.current_pair_idx < len(ui.all_inds):
                        ui._sync_any_plot_context(ui.current_pair_idx)
                ui._clamp_current_segment_for_session()
            # Data not yet loaded — nothing to render
            if ui.ccg_data is None or ui.ccg_ptr is None:
                return

            # In All/Any mode, Time Slider can be enabled even though a single
            # behavioral timeline can't represent multiple sessions at once.
            # Keep rendering the CCG plot (the time slider UI is separate).

            # If spike attribution raster is active, keep showing it
            sa = ui.center_container.spike_attribution_panel
            if sa._selected_idx >= 0 and sa._spike_pairs:
                return

            # Focus-pair override: show focused pair's CCG directly
            if ui.network_panel._focused_pair is not None:
                inds = np.array(ui.network_panel._focused_pair)
            elif ui.current_pair_idx >= len(ui.all_inds):
                return
            else:
                inds = ui.all_inds[ui.current_pair_idx]
            # Safety: keep rendered pair indices aligned to currently bound ccg_data.
            r0, t0 = int(inds[0]), int(inds[1])
            if not ui._ccg_pair_in_bounds(r0, t0):
                if (getattr(ui, '_session_any_mode', False)
                        and ui.current_pair_idx < len(ui.all_inds)):
                    ui._sync_any_plot_context(ui.current_pair_idx)
                    ui._clamp_current_segment_for_session()
                    if ui.ccg_data is None or ui.current_pair_idx >= len(ui.all_inds):
                        return
                    inds = ui.all_inds[ui.current_pair_idx]
                    r0, t0 = int(inds[0]), int(inds[1])
                if not ui._ccg_pair_in_bounds(r0, t0):
                    print(f"[CCGReviewUI] skipping out-of-bounds pair ({r0}, {t0}) "
                          f"for ccg shape={getattr(getattr(ui.ccg_data, 'ccg', None), 'shape', None)}")
                    return
            sbs = ui._lo_hi_mode
            extend_on = bool(ui._acg_var_get('_extend_enable', False))

            def _pngs_for_seg(arr, seg, highres=None):
                """Return [(png_path, col_title), ...] for one segment slot.

                sbs off, extend off  → [Normal]
                sbs off, extend on   → [Normal, Ext]
                sbs on,  extend off  → [Lo, Hi]  (or [Normal] if no hi-res)
                sbs on,  extend on   → [Lo, Hi, Ext]  (Ext = hi-res if avail)
                """
                slots = []
                if sbs:
                    nd_key = ui.key.nd()
                    has_hi = (hasattr(ui.cd, '_ccg_highres')
                              and ui.cd._ccg_highres.get(nd_key) is not None)
                    if has_hi:
                        p_lo = ui._png_mgr._render_png_with_res(arr, seg, highres=False)
                        p_hi = ui._png_mgr._render_png_with_res(arr, seg, highres=True)
                        slots.append((p_lo, 'Lo'))
                        slots.append((p_hi, 'Hi'))
                    else:
                        p = ui._get_or_render_png(arr, seg)
                        slots.append((p, ''))
                    if extend_on:
                        hi_for_ext = has_hi
                        p_ext = ui._plot_mgr._render_png(arr, seg, highres=hi_for_ext)
                        slots.append((p_ext, 'Ext'))
                else:
                    hi = highres if highres is not None else False
                    p = ui._get_or_render_png(arr, seg)
                    slots.append((p, ''))
                    if extend_on:
                        p_ext = ui._plot_mgr._render_png(arr, seg, highres=hi)
                        slots.append((p_ext, 'Ext'))
                return slots

            if ui._together_pairs and len(ui._together_pairs) >= 2:
                n_tog = len(ui._together_pairs)
                seg = ui.current_segment
                slots0 = _pngs_for_seg(np.array(ui._together_pairs[0]), seg)
                n_cols = len(slots0)
                ui.fig.clear()
                axes_grid = ui.fig.subplots(n_tog, n_cols, squeeze=False)
                for row_i, tp in enumerate(ui._together_pairs):
                    tp_arr = np.array(tp)
                    slots = _pngs_for_seg(tp_arr, seg)
                    for ax, (png, col_title) in zip(axes_grid[row_i], slots):
                        ax.imshow(mpimg.imread(png))
                        ax.axis('off')
                        if col_title:
                            ax.set_title(col_title, fontsize=8, pad=1)
                ui.fig.tight_layout(pad=0.05)

            elif getattr(ui, '_stacked_segments', None):
                all_idx = ui.n_segments
                segs_raw = [int(s) for s in ui._stacked_segments]
                has_all = all_idx in segs_raw
                segs = ([all_idx] if has_all else []) + sorted(
                    s for s in segs_raw if s != all_idx)
                if not segs:
                    return

                def _seg_label(si: int) -> str:
                    if si == ui.n_segments:
                        return _ALL_SEGS
                    if ui._is_custom_segment(si):
                        ci = ui._custom_seg_index(si)
                        cs_list = getattr(ui, '_custom_segments', [])
                        if 0 <= ci < len(cs_list):
                            return cs_list[ci].get('name', f'custom {ci}')
                        return 'custom'
                    if 0 <= si < len(ui.ccg_ptr.segment_names):
                        return str(ui.ccg_ptr.segment_names[si])
                    return str(si)

                slots0 = _pngs_for_seg(inds, segs[0])
                n_rows = len(slots0)   # rows = res/ext slots
                n_seg_cols = len(segs)
                ui.fig.clear()
                axes_grid = ui.fig.subplots(n_rows, n_seg_cols, squeeze=False)
                for col_i, seg in enumerate(segs):
                    seg_lbl = _seg_label(seg)
                    slots = _pngs_for_seg(inds, seg)
                    for row_i, (png, col_title) in enumerate(slots):
                        ax = axes_grid[row_i][col_i]
                        ax.imshow(mpimg.imread(png))
                        ax.axis('off')
                        t = f"{seg_lbl} · {col_title}" if col_title else seg_lbl
                        ax.set_title(t, fontsize=8, pad=1)
                ui.fig.tight_layout(pad=0.05)
                if has_all and n_seg_cols >= 2:
                    ax0_bb = axes_grid[0][0].get_position()
                    ax1_bb = axes_grid[0][1].get_position()
                    x_mid = (ax0_bb.x1 + ax1_bb.x0) / 2
                    _col = 'white' if ui.theme.dark else '#888888'
                    ui.fig.add_artist(Line2D(
                        [x_mid, x_mid], [0.02, 0.98],
                        transform=ui.fig.transFigure,
                        color=_col, lw=1.0, zorder=10))

            else:
                # Single segment: 1×n_slots
                seg = ui.current_segment
                slots = _pngs_for_seg(inds, seg)
                ui.fig.clear()
                if len(slots) == 1:
                    ax = ui.fig.add_subplot(111)
                    ax.imshow(mpimg.imread(slots[0][0]))
                    ax.axis('off')
                    ui.fig.tight_layout(pad=0)
                else:
                    axes = ui.fig.subplots(1, len(slots))
                    for ax, (png, col_title) in zip(axes, slots):
                        ax.imshow(mpimg.imread(png))
                        ax.axis('off')
                        if col_title:
                            ax.set_title(col_title, fontsize=9, pad=2)
                    ui.fig.tight_layout(pad=0.3)

            if ui.theme.dark:
                ui.fig.set_facecolor('#2b2b2b')
                for _a in ui.fig.get_axes():
                    _a.set_facecolor('#2b2b2b')
                    _a.title.set_color('white')
            ui.canvas.draw()

            ui.plot_title_var.set(ui._plot_mgr.get_plot_title())
            ui._update_sig_indicators(inds)
            ui._update_jitter_sig_buttons()
            ui._update_global_baseline_availability()
            ui._update_pair_info(inds)
            ui._draw_waveforms()
            ui._update_conn_str_label()

        except Exception as e:
            print(f"Error updating plot: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Waveforms
    # ------------------------------------------------------------------

    def get_plot_title(self):
        ui = self._ui
        if ui.current_pair_idx < len(ui.all_inds):
            inds = ui.all_inds[ui.current_pair_idx]
            if ui._is_custom_segment():
                ci = ui._custom_seg_index()
                seg_label = ui._custom_segments[ci]['name']
            elif ui.current_segment == ui.n_segments:
                seg_label = _ALL_SEGS
            else:
                seg_label = ui.ccg_ptr.segment_names[ui.current_segment]
            sess = ui.key.session if ui.key else ''
            def _fmt_ct(x):
                s = str(x)
                low = s.lower()
                if low == 'pyr':
                    return 'PYR'
                if low in ('inter', 'int'):
                    return 'INT'
                return s.upper()
            ct = (f"{_fmt_ct(ui.key.conn_type[0])}-{_fmt_ct(ui.key.conn_type[1])}"
                  if ui.key and ui.key.conn_type else '')
            _plabel = (ui.network_panel._pair_label(inds)
                       if hasattr(self, 'network_panel') else
                       f"{inds[0]}→{inds[1]}")
            pair_str = f"{_plabel} (inds [{inds[0]}, {inds[1]}])"
            return f"{sess} | {ct} | {pair_str} — {seg_label}"
        return "No pair selected"

    # ------------------------------------------------------------------
    # PNG rendering
    # ------------------------------------------------------------------

    def get_pair_index(self, inds):
        ui = self._ui
        if getattr(ui, '_session_any_mode', False):
            hl = getattr(ui, '_any_pair_handle_list', None) or []
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
                if r2 == r and t2 == t:
                    return i
            return 0
        for i, pair in enumerate(ui.all_inds):
            if tuple(pair) == tuple(inds):
                return i
        return 0

    # ------------------------------------------------------------------
    # Segment navigation
    # ------------------------------------------------------------------

    def _update_pair_info(self, inds):
        """Update per-pair info (firing rates) in the stats bar."""
        ui = self._ui
        if not hasattr(ui, '_pair_info_var'):
            return
        if ui.neurons is None:
            ui._pair_info_var.set("")
            return
        ref, tgt = int(inds[0]), int(inds[1])
        fr = getattr(ui.neurons, 'firing_rate', None)
        if fr is None:
            ui._pair_info_var.set("")
            return
        if ref >= len(fr) or tgt >= len(fr):
            ui._pair_info_var.set("")
            return
        ref_fr = float(fr[ref])
        tgt_fr = float(fr[tgt])

        # Segment-level firing rate
        seg_str = ""
        seg = ui.current_segment
        nd_key = ui.key.nd() if ui.key else None
        if ui._is_custom_segment(seg):
            ci = ui._custom_seg_index(seg)
            cs = ui._custom_segments[ci]
            cs_fr = cs.get('firing_rates')
            if cs_fr is not None and ref < len(cs_fr) and tgt < len(cs_fr):
                sr = float(cs_fr[ref])
                st = float(cs_fr[tgt])
                seg_str = f"  Seg FR: ref={sr:.1f} Hz  tgt={st:.1f} Hz"
        else:
            seg_frates = None
            if (nd_key is not None and ui.cd.nd is not None
                    and seg != ui.n_segments):
                seg_frates = ui.cd.nd.segment_firing_rates.get(nd_key)
            if (seg_frates is not None and seg < seg_frates.shape[0]
                    and ref < seg_frates.shape[1] and tgt < seg_frates.shape[1]):
                sr = float(seg_frates[seg, ref])
                st = float(seg_frates[seg, tgt])
                seg_str = f"  Seg FR: ref={sr:.1f} Hz  tgt={st:.1f} Hz"

        ui._pair_info_var.set(
            f"FR: ref={ref_fr:.1f}Hz  tgt={tgt_fr:.1f}Hz{seg_str}")

    # ------------------------------------------------------------------
    # Undo / redo for pair selection
    # ------------------------------------------------------------------

    def _accumulate_ylim(self, ccg_raw, ccg_null_raw, ref, tgt, seg, is_custom, ymin, ymax, *, custom_time_h=None):
        """Accumulate y-axis limits for one trace using the active normalizations.

        Uses the same normalization helper as rendering, so pair-scale never clips
        when the user changes normalization settings.
        """
        ui = self._ui
        ccg, ccg_null = apply_norms_to_ccg(
            ccg_raw,
            ccg_null_raw,
            ref, tgt, seg,
            ui.active_norms,
            ui.neurons,
            ui.cd.nd,
            ui.key.nd(),
            ui.n_segments,
            bool(is_custom),
            custom_time_hours=custom_time_h,
        )
        ymin = min(ymin, float(np.nanmin(ccg)))
        ymax = max(ymax, float(np.nanmax(ccg)))
        if ccg_null is not None:
            ymax = max(ymax, float(np.nanmax(ccg_null)))
        return ymin, ymax

    def _render_png(self, inds, segment, highres=None,
                    _render_cfg=None, _ccg_data_override=None) -> str:
        """Thin coordinator: resolve state, build render context, write PNG."""
        ui = self._ui
        if highres is None:
            highres = ui._highres_mode
        ctx = ui._render_engine.build_context(
            inds, segment, highres, _render_cfg, _ccg_data_override)
        png_path = ui._png_path(inds, segment,
                                   _render_cfg=_render_cfg, _hires_override=highres)
        ui._render_engine.write_png(ctx, png_path)
        return png_path

    # ------------------------------------------------------------------
    # Plot update
    def _highlight_changed_pairs(self, changed_pairs):
        """Highlight pairs that moved during undo/redo with baseline color.

        The highlight clears on the next arbitrary click anywhere in the UI.
        """
        ui = self._ui
        if not changed_pairs:
            return
        avail_map = getattr(self, '_avail_list_pairs', None)
        unsel_items = (((i, e[0]) for i, e in enumerate(avail_map) if e is not None)
                       if avail_map else enumerate(sorted(ui.unselected_inds)))
        for idx, inds in unsel_items:
            if inds in changed_pairs:
                ui.unselected_list.itemconfig(idx, background=ui._sel_mgr._UNDO_HIGHLIGHT,
                                                foreground='white')
        sel_map = getattr(self, '_sel_list_pairs', None)
        sel_items = (((i, e) for i, e in enumerate(sel_map) if e is not None)
                     if sel_map else enumerate(sorted(ui.selected_inds)))
        for idx, inds in sel_items:
            if inds in changed_pairs:
                ui.selected_list.itemconfig(idx, background=ui._sel_mgr._UNDO_HIGHLIGHT,
                                              foreground='white')
        # Clear highlight on next click anywhere
        def _clear_highlight(e=None):
            ui._clear_undo_highlight()
            ui.root.unbind('<Button-1>', bind_id)
        bind_id = ui.root.bind('<Button-1>', _clear_highlight, add='+')

class MiscManager:
    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        ui.active_alpha = getattr(getattr(ui.cd, 'conf', None), 'alpha', 0.05)
        ui.active_norms: set = set()
        ui.norm_vars: dict = {}
        ui.active_segment_filter = None
        ui._highres_mode = False
        ui._lo_hi_mode = False
        ui._ccg_memory_limit_gb = float(
            (ui._settings if hasattr(ui, '_settings') else {}).get('ccg_memory_limit_gb', 8.0))
        ui._same_scale_mode: str = None
        ui._pair_scale_cache: LRUCache = LRUCache(512)
        ui._session_scale_cache = None
        ui.jitter_controller = JitterController(ui)
        ui.jitter_worker = ui.jitter_controller.jitter_worker
        ui._jitter_cache = ui.jitter_worker._cache
        ui._jitter_pending = ui.jitter_worker._runner._pending
        ui._jitter_unviewed = ui.jitter_worker.unviewed

    def _json_pair_ct_fallback(self, session_str):
        """Build (ref,tgt)->conn_type_label from JSON selection files for a session.
        Result is cached per session string."""
        ui = self._ui
        cache_attr = '_json_ct_cache'
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
        cache = getattr(self, cache_attr)
        if session_str in cache:
            return cache[session_str]
        result = {}
        try:
            for fname in os.listdir(ui._sel_save_dir):
                if not fname.endswith('.json'):
                    continue
                fpath = os.path.join(ui._sel_save_dir, fname)
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
                        ct_label = ui._conn_type_label((parts[0], parts[1]))
                    else:
                        ct_label = raw_ct
                    for p in sel_pairs:
                        if len(p) >= 2:
                            result[(int(p[0]), int(p[1]))] = ct_label
        except Exception:
            pass
        cache[session_str] = result
        return result

    def _on_add_focused_pair(self):
        """Add the currently focused non-significant pair to available pairs."""
        ui = self._ui
        pair = ui.network_panel._focused_pair
        if pair is None:
            return
        ref, tgt = pair
        # Validate indices are within CCG data shape
        cd = ui.ccg_data
        if cd is None or cd.ccg is None:
            messagebox.showerror("Add pair", "No CCG data loaded.")
            return
        if ref >= cd.ccg.shape[1] or tgt >= cd.ccg.shape[2]:
            messagebox.showerror("Add pair",
                                 f"Pair ({ref},{tgt}) out of CCG data range "
                                 f"({cd.ccg.shape[1]}x{cd.ccg.shape[2]}).")
            return
        ui._push_undo()
        _admit_pair(ui._sel_data, pair)
        ui._sel_data.set_state(pair, 'unsel')
        # Navigate to the pair
        ui.current_pair_idx = ui.get_pair_index(pair)
        # Clear focus and update UI
        ui.network_panel._focused_pair = None
        ui.network_panel._focus_pair_var.set("")
        ui.network_panel._update_pair_focus_info(pair, exists=False)
        ui.network_panel._add_pair_btn.config(state=tk.DISABLED)
        ui.refresh_lists()
        ui.network_panel.draw()
        ui.update_plot()

    # ------------------------------------------------------------------
    # Group helpers (Part II.2)
    # ------------------------------------------------------------------

    def _pairs_by_conn_type(self, session_str, pairs):
        """Group pairs by their conn_type for display. Returns {label: [pairs]}."""
        ui = self._ui
        # Build pair → conn_type mapping from cd.data keys for this session
        pair_ct = {}
        for key in ui.cd.data:
            if str(key.session) != session_str and str(key.nd().session) != session_str:
                continue
            ct = getattr(key, 'conn_type', None)
            ct_label = ui._conn_type_label(ct)
            pt = ui.cd.data[key]
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


class CCGReviewUI(GenericUI):
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

    _HISTORY_SUBDIR = '.history'
    _AUTOSAVE_INTERVAL_MS = 30 * 60 * 1000  # 30 minutes

    def __init__(self, cd, key):
        super().__init__()
        self.cd = cd
        self.key = key
        self.ccg_ptr = self.cd.data.get(key)
        # If raw CCG arrays for this session aren't loaded yet, load them now (fast: sidecar file)
        nd_key = key.nd()
        if nd_key not in cd._ccg:
            cd.get_ccg(nd_key=nd_key)
        self.ccg_data = self.cd._ccg.get(nd_key)
        self.neurons = self.cd.nd.data[nd_key]

        self._sel_data = SelectionData()

        # Dirs needed by managers at construction time
        self._sel_save_dir = str(_Path(__file__).resolve().parents[2] / "data" / "selections")
        os.makedirs(self._sel_save_dir, exist_ok=True)

        # Instantiate manager singletons early — methods may be called via @property
        # accessors (e.g. all_inds → _group_mgr) before the later setup block runs.
        self._settings_mgr = SettingsManager(self)
        self._export_mgr = ExportManager(self)
        self._custom_mgr = CustomCCGManager(self)
        self._group_mgr = GroupManager(self)
        self._sel_mgr = SelectionPersistenceManager(self)
        self._sess_mgr = MultiSessionManager(self)
        self._png_mgr = PNGCacheManager(self)
        self._pregen_ctrl = PregenController(self)
        self._cs_mgr = ConnectionStrengthManager(self)
        self._pair_mgr = PairAnalysisManager(self)
        self._sim_mgr = SimulationManager(self)
        self._setup_mgr = UISetupManager(self)
        self._plot_mgr = PlotManager(self)
        self._misc_mgr = MiscManager(self)

        # Pair / segment state  (all_inds is a @property — see below)
        # These are set to empty defaults when data is not yet loaded; they are
        # refreshed in _finish_initial_draw() after lazy loading completes.
        self.n_segments = self.ccg_ptr.n_segments
        self.current_segment = self.n_segments   # default = All (n_segments sentinel)
        self.current_pair_idx = 0

        # Manual selection state
        ptr = self.ccg_ptr
        _sel = getattr(ptr, 'selected_inds', None)
        self._sel_data.selected_inds   = set(map(tuple, _sel)) if _sel is not None else set()
        self._sel_data.unselected_inds = ptr.unselected_inds

        # Session-only list markers (cleared on quit / Selections menu)
        self._bookmarked_pairs: set[tuple[int, int]] = set()

        # Display-only computed values for the current visual state (e.g., ACG deconvolution)
        self._display_pair_temp: dict = {}

        # Significance display toggles live in BooleanVars (created after Tk root).
        # Use _sig(name) helper to read them.

        # Double-click debounce
        self._select_after: int = None       # after() id for deferred pair update

        self._setup_window()
        self._setup_mgr.setup_ui()
        # Apply any persisted font sizing constraints after widgets exist
        self._settings_mgr.apply_min_font_size()
        self._emit_inventory_event()

    # ------------------------------------------------------------------
    # Derived state
    # ------------------------------------------------------------------

    @property #TODO is this outdated? moved to ccg manager?
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
        if self.ccg_ptr is None:
            return np.empty((0, 2), dtype=int)
        base = self.ccg_ptr.inds2
        # Filter out self-pairs (autocorrelograms)
        mask = base[:, 0] != base[:, 1]
        base = base[mask]
        admitted = _admitted_pairs(self._sel_data)
        if not admitted:
            return base
        base_set = set(map(tuple, base))
        extra = sorted(set(tuple(p) for p in admitted if p[0] != p[1]) - base_set)
        if not extra:
            return base
        return np.vstack([base, np.array(extra, dtype=base.dtype)])

    def _all_inds_set_for_ptr(self, ptr) -> set:
        """set of (ref, tgt) for a CCGPointer — same rules as all_inds, without self.ccg_ptr."""
        if ptr is None or ptr.inds2 is None:
            return set()
        base = ptr.inds2
        base = base[base[:, 0] != base[:, 1]]
        s = set(map(tuple, base))
        admitted = _admitted_pairs(self._sel_data)
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
        return self._setup_mgr._current_session_str()


    def _refresh_hotkeys_bar(self):
        self._group_mgr._refresh_hotkeys_bar()

    def _load_selection_from_file(self, path: str, restore_groups: bool = True, **kw):
        return self._sel_mgr._load_selection_from_file(path, restore_groups=restore_groups, **kw)

    def get_plot_title(self) -> str:
        return self._plot_mgr.get_plot_title()

    # ------------------------------------------------------------------
    # Group registry helpers (v4.0 schema)
    # ------------------------------------------------------------------




    def _atomic_write_json(self, path: str, data: dict):
        """Write JSON atomically via a .tmp file, preventing partial writes."""
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=SelectionPersistenceManager._json_default)
        os.replace(tmp, path)

    def _group_pairs(self, gname, session=None):
        return self._group_mgr._group_pairs(gname, session)

    def _group_pairs_all_sessions(self, gname):
        return self._group_mgr._group_pairs_all_sessions(gname)

    def _group_add_pair(self, gname, pair, session=None):
        self._group_mgr._group_add_pair(gname, pair, session)


    def _dbg_log(self, hypothesis_id: str, location: str, message: str, data: dict):
        """Debug-mode NDJSON logger (session cde521)."""
        # #region agent log
        try:
            payload = {
                "sessionId": "cde521",
                "runId": "pre-fix",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(_time.time() * 1000),
            }
            _path = "/Users/selinl/Documents/ms_synchrony/NeuroPy/.cursor/debug-cde521.log"
            os.makedirs(os.path.dirname(_path), exist_ok=True)
            with open(_path, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(payload) + "\n")
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
                self.network_panel.draw()
        except Exception:
            pass

    def _ui_state_path(self):
        return os.path.join(self._sel_save_dir, 'ui_state.json')

    def _save_ui_state(self):
        self._settings_mgr.save_ui_state()

    def setup_settings_menu(self, menubar):
        """Settings menu — opens the settings dialog."""
        m = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=m)
        m.add_command(label="Settings…", command=lambda: SettingsDialog.show(self))

    def _open_stats_panel(self):
        """Open (or raise) the Stats Tests panel."""
        from neuropy.ui.stats_tests import StatsTestPanel  # noqa: PLC0415
        if self._stats_panel is None or not self._stats_panel.root.winfo_exists():
            self._stats_panel = StatsTestPanel(self)
        else:
            self._stats_panel.root.lift()
            try:
                self._stats_panel.refresh_session_dropdowns()
            except Exception:
                pass

    def _jitter_queue_dialog(self):
        JitterQueueDialog.show(self)

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
                self._on_chunk_done(task)
            n_removed += len(self._custom_ccg_pending) - 1
            self._custom_ccg_pending.clear()
            self._custom_ccg_pending.append(first)
        else:
            for task in list(self._custom_ccg_pending):
                self._on_chunk_done(task)
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
        help_menu.add_command(label="Hotkeys", command=self._group_mgr._show_hotkeys_dialog)
        help_menu.add_command(label="Website",
                              command=lambda: __import__('webbrowser').open(
                                  'https://github.com/selina-lii/NeuroPy'))

    def _min_font_size(self) -> int:
        return self._settings_mgr.min_font_size()

    # ── Center panel ───────────────────────────────────────────────────


    def _acg_var_get(self, name, default=None):
        """Safely read a correlogram-panel Tk variable by attribute name."""
        v = getattr(self.center_container.correlogram_panel, name, None)
        return v.get() if v is not None else default

    def _sig(self, name):
        """Read a significance/display toggle BooleanVar by short name."""
        bp = self.center_container.baseline_panel
        if name == 'conv_p':      return bool(bp._sig_conv_p.get())
        if name == 'conv_pc':     return bool(bp._sig_conv_pc.get())
        if name == 'test_window': return bool(bp._sig_test_window.get())
        if name == 'jitter_pc':   return bool(bp._sig_jitter_pc.get())
        return False

    def _current_display_config(self) -> dict:
        """Snapshot of every display-state var that affects PNG rendering."""
        if not hasattr(self, 'center_container'):
            return {}
        cc = self.center_container
        panels = (cc.correlogram_panel, cc.baseline_panel, cc.cs_panel)
        cfg = {}
        for attr in SettingsManager._CACHE_CONFIG_ATTRS:
            for panel in panels:
                v = getattr(panel, attr, None)
                if v is not None:
                    cfg[attr] = v.get()
                    break
            else:
                cfg[attr] = None
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
        self._pair_mgr._on_sig_toggle()

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
            self.center_container.correlogram_panel._line_jitter.set(False)
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
            if not has_jitter and self.center_container.baseline_panel._conn_str_method.get() == 'jitter':
                self.center_container.baseline_panel._conn_str_method.set('conv')
                self._cs_mgr._rebuild_cs_pval_row()
        self._update_global_baseline_availability()
        self._update_adaptive_tw_availability()

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

    def _on_baseline_method_change(self):
        self._cs_mgr._on_baseline_method_change()

    # ── Right panel ────────────────────────────────────────────────────

    def _draw_network(self):
        self.network_panel.draw()

    # ── Time slider panel ──────────────────────────────────────────────

    # ── Bottom bar ─────────────────────────────────────────────────────

    def _toggle_waveforms_panel(self):
        self._setup_mgr._toggle_waveforms_panel()

    def _toggle_resolution(self):
        """Toggle between low-res ``_ccg`` and high-res ``_ccg_highres``.

        If high-res data for the current session is not yet in memory, triggers
        a background load via _ensure_highres_loaded before switching.
        """
        nd_key = self.key.nd()
        # Toggling back to low-res never needs a load
        if self._highres_mode:
            self._apply_resolution_toggle(nd_key)
            return
        # Switching to high-res: ensure data is available first
        hi_available = (hasattr(self.cd, '_ccg_highres')
                        and self.cd._ccg_highres.get(nd_key) is not None)
        if hi_available:
            self._apply_resolution_toggle(nd_key)
        else:
            self._sess_mgr._ensure_highres_loaded(nd_key, on_loaded=lambda: self._apply_resolution_toggle(nd_key))

    def _apply_resolution_toggle(self, nd_key):
        """Apply resolution switch after data is confirmed available."""
        self._pair_scale_cache.clear()
        self._session_scale_cache = None
        self._highres_mode = not self._highres_mode
        mode_label = 'highres' if self._highres_mode else 'lowres'
        if hasattr(self, '_res_btn_text'):
            self._res_btn_text.set(f"Res: {mode_label}")
        if self._highres_mode:
            self.ccg_data = self.cd._ccg_highres.get(nd_key)
        else:
            self.ccg_data = self.cd._ccg.get(nd_key)
        self._clear_all_png_cache()
        self._build_sig_chips()
        if (self.ccg_data is not None and self.current_pair_idx < len(self.all_inds)):
            r0, t0 = int(self.all_inds[self.current_pair_idx][0]), int(self.all_inds[self.current_pair_idx][1])
            if not self._ccg_pair_in_bounds(r0, t0):
                for cand_idx in range(len(self.all_inds)):
                    cr, ct = int(self.all_inds[cand_idx][0]), int(self.all_inds[cand_idx][1])
                    if self._ccg_pair_in_bounds(cr, ct):
                        self.current_pair_idx = cand_idx
                        break
        self.update_plot()

    def _ccg_context_menu(self, event):
        self._setup_mgr._ccg_context_menu(event)

    def _export_current_view(self):
        self._export_mgr._export_current_view()

    def _on_ctrl_e(self):
        var = self._panel_vars.get('Waveforms')
        if var:
            var.set(not var.get())
            self._setup_mgr._toggle_panel('Waveforms')

    # ------------------------------------------------------------------
    # Significance helpers
    # ------------------------------------------------------------------

    def _is_significant(self, ref: int, tgt: int, seg: int) -> bool:
        return self._pair_mgr._is_significant(ref, tgt, seg)

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
    # Same-scale helpers
    # ------------------------------------------------------------------

    def _effective_bin_size(self) -> float:
        """Infer true bin_size from CCG array shape (robust to conf mutation)."""
        conf = self.ccg_data.conf
        n_bins = self.ccg_data.ccg.shape[-1]
        return conf.duration / (n_bins - 1) if n_bins > 1 else conf.bin_size

    def _get_current_scale_ylim(self, ref: int, tgt: int):
        """Return (ymin, ymax) for the active scale mode, or None.
        Pair scale is cached per (ref, tgt, resolution) so hi-res and lo-res
        get independent y-axes instead of sharing the lo-res maximum."""
        if self._same_scale_mode == 'pair':
            cache_key = (ref, tgt, self._res_key)
            if cache_key not in self._pair_scale_cache:
                self._pair_mgr._compute_pair_scale(ref, tgt)   # populates cache
            return self._pair_scale_cache.get(cache_key)
        if self._same_scale_mode == 'session':
            if self._session_scale_cache is None:
                self._session_scale_cache = self._sess_mgr._compute_session_scale()
            return self._session_scale_cache
        return None

    def _on_pair_scale_toggle(self):
        if self.center_container.norm_panel._pair_scale.get():
            self._same_scale_mode = 'pair'
            self.center_container.norm_panel._sess_scale.set(False)   # mutual exclusion
        else:
            self._same_scale_mode = None
        self._pair_scale_cache.clear()
        self._refresh()

    def _on_session_scale_toggle(self):
        self._sess_mgr._on_session_scale_toggle()

    def _on_run_jitter(self):
        self.jitter_controller.on_run_jitter()

    def _is_task_running(self):
        return self.jitter_controller.is_task_running()

    def _custom_ccg_is_running(self):
        return self._custom_mgr._runner.is_running()

    def _jitter_start_next(self):
        self.jitter_controller.start_next()

    def _custom_ccg_start_next(self):
        return self._custom_mgr._custom_ccg_start_next()

    def _update_jitter_btn_text(self):
        self.jitter_controller.update_btn_text()

    def _on_save_jitter(self):
        self.jitter_controller.on_save()

    def _load_jitter_from_cd(self):
        self.jitter_controller.load_from_cd()

    def _jitter_seg(self, seg=None):
        return self.jitter_controller.seg(seg)

    def _on_clear_jitter(self):
        self.jitter_controller.on_clear()

    def _apply_jitter_list_colors(self, pair=None):
        self.jitter_controller.apply_list_colors(pair)

    def _mark_jitter_viewed(self) -> bool:
        return self.jitter_controller.mark_viewed()

    def _finalize_normalization(self):
        self._pair_mgr._finalize_normalization()

    def _clear_all_png_cache(self):
        self._png_mgr._clear_all_png_cache()

    def _refresh(self, clear_extend: bool = False) -> None:
        """Invalidate PNG cache and redraw current pair."""
        if clear_extend:
            self._extend_cache.clear()
        self._refresh()

    def _terminate_pregen_proc(self):
        self._pregen_ctrl._terminate_pregen_proc()

    def _clear_seg_filter(self):
        self.seg_filter_var.set('')
        self.active_segment_filter = None
        self.network_panel.draw()

    def _build_sig_chips(self):
        if hasattr(self, 'center_container'):
            self.center_container.seg_chips_panel.rebuild()

    # ------------------------------------------------------------------
    # Key / dropdown helpers
    # ------------------------------------------------------------------

    def _session_label(self, nd_key) -> str:
        return self._sess_mgr._session_label(nd_key)

    def _real_nd_keys_ordered(self) -> list:
        return self._sess_mgr._real_nd_keys_ordered()

    def _sess_title_label(self, sess: str) -> str:
        """Return 'session=NSD 2' style label for use in plot titles."""
        ordered = [str(getattr(nk, 'session', nk)) for nk in self._real_nd_keys_ordered()]
        try:
            idx = ordered.index(sess) + 1   # 1-based
        except ValueError:
            return f"session={sess}"
        block_label = "SD" if idx > _SESS_PER_BLOCK else "NSD"
        block_num   = ((idx - 1) % _SESS_PER_BLOCK) + 1
        return f"session={block_label}{block_num}"

    def _type_key_for_nd(self, nd_key):
        return self._sess_mgr._type_key_for_nd(nd_key)

    def _nd_key_for_session_str(self, sess_str: str):
        return self._sess_mgr._nd_key_for_session_str(sess_str)

    def _any_group_header_names(self) -> list[str]:
        return self._group_mgr._any_group_header_names()

    def _any_triples_in_group(self, gname: str) -> set[tuple]:
        return self._group_mgr._any_triples_in_group(gname)

    def _any_nd_keys_for_group(self, gname: str) -> list:
        return self._group_mgr._any_nd_keys_for_group(gname)

    def _any_iter_pairs_for_group(self, gname: str):
        return self._group_mgr._any_iter_pairs_for_group(gname)

    def _any_rebuild_pair_handles(self):
        return self._sess_mgr._any_rebuild_pair_handles()

    def _any_sync_selection_from_universe(self):
        self._sess_mgr._any_sync_selection_from_universe()

    def _enter_all_session_mode(self):
        return self._sess_mgr._enter_all_session_mode()

    def _exit_all_session_mode(self):
        self._sess_mgr._exit_all_session_mode()

    def _bind_context_to_type_key(self, tk):
        return self._sess_mgr._bind_context_to_type_key(tk)

    def _autosave_all_sessions_for_current_type(self):
        self._sel_mgr._autosave_all_sessions_for_current_type()

    def _sync_any_plot_context(self, row_idx: int):
        return self._sess_mgr._sync_any_plot_context(row_idx)

    def _toggle_any_avail_group(self, gname: str):
        return self._group_mgr._toggle_any_avail_group(gname)

    def _pair_sess_rt(self, inds) -> tuple[str, tuple[int, int]]:
        """Session string + (ref,tgt) for group/tag lookups."""
        if getattr(self, '_session_any_mode', False):
            if isinstance(inds[0], Key):
                return str(inds[0].session), (int(inds[1]), int(inds[2]))
            return str(inds[0]), (int(inds[1]), int(inds[2]))
        return self._current_session_str(), (int(inds[0]), int(inds[1]))

    def _pair_in_group(self, pair, group_name: str) -> bool:
        return self._group_mgr._pair_in_group(pair, group_name)

    def _available_type_keys(self, nd_key) -> list:
        return self._sess_mgr._available_type_keys(nd_key)

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

        if self.ccg_ptr is not None:
            self.ccg_ptr.selected_inds = (
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

        old_ptr = self.ccg_ptr
        _prev_seg_label = (old_ptr.segment_names[self.current_segment]
                           if old_ptr is not None
                              and 0 <= self.current_segment < old_ptr.n_segments
                           else None)
        self.key = new_key
        self.ccg_ptr = ptr
        nd_key = new_key.nd()
        if (getattr(self, '_highres_mode', False)
                and hasattr(self.cd, '_ccg_highres')
                and self.cd._ccg_highres.get(nd_key) is not None):
            self.ccg_data = self.cd._ccg_highres[nd_key]
        else:
            self._highres_mode = False   # reset if highres not available
            self.ccg_data = self.cd._ccg.get(nd_key)
        try:
            self.neurons = (self.cd.nd.data[new_key.nd()]
                            if getattr(self.cd, 'nd', None) is not None else None)
        except KeyError:
            self.neurons = None
        # all_inds is a @property — no assignment needed
        self.n_segments = self.ccg_ptr.n_segments
        new_names = self.ccg_ptr.segment_names
        if _prev_seg_label is not None and _prev_seg_label in new_names:
            self.current_segment = new_names.index(_prev_seg_label)
        else:
            self.current_segment = 0
        self.current_pair_idx = 0
        if (hasattr(self.ccg_ptr, 'selected_inds') and
                self.ccg_ptr.selected_inds is not None):
            self.selected_inds = set(
                map(tuple, self.ccg_ptr.selected_inds))
        else:
            self.selected_inds = set()
        _avail = set(map(tuple, self.all_inds))
        self.deleted_inds = (
            set(self._pair_deleted_store.get(str(new_key), set())) & _avail)
        self.unselected_inds = _avail - self.selected_inds - self.deleted_inds
        self._sel_data.selected_inds   = self.selected_inds
        self._sel_data.unselected_inds = self.unselected_inds
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
        self.segment_combo['values'] = self.ccg_ptr.segment_names + [_ALL_SEGS]
        self.segment_var.set(self.ccg_ptr.segment_names[0])
        self._build_sig_chips()
        self.refresh_lists()
        self.plot_title_var.set(self.get_plot_title())
        # Lazy-load jitter results for the new session if not yet in memory
        _nd = self.key.nd()
        if (hasattr(self.cd, 'load_jitter')
                and _nd not in getattr(self.cd, '_jitter_results', {})):
            self.cd.load_jitter(nd_key=_nd)
        self.jitter_controller.load_from_cd()
        self.update_plot()
        if self.network_panel._focused_neuron is not None:
            self.network_panel._update_focus_info(self.network_panel._focused_neuron)
        self.network_panel.refresh_shank_buttons()
        self.network_panel.draw()
        if self._auto_pregen_enabled and self._cache_config is not None:
            self._pregen_ctrl._launch_pregen_subprocess(dict(self._cache_config), priority='auto')
        _sess_ch = getattr(self, '_switch_key_session_changed', False)
        self.time_slider.on_key_changed()
        if not _sess_ch:
            self.time_slider.on_session_mode_changed()
        self._switch_key_session_changed = False

    def _switch_type_any(self, new_key):
        """Change connection type while in virtual ``any`` session."""
        self._flush_any_selections_to_pointers()
        try:
            self._autosave_all_sessions_for_current_type()
        except Exception as exc:
            print(f"[CCGReviewUI] any-session type switch autosave: {exc}")
        try:
            if self._sel_data._groups:
                self._save_groups_export()
        except Exception:
            traceback.print_exc()
        self._bind_context_to_type_key(new_key)
        self._type_var.set(self._type_label(new_key))
        self._any_load_deleted_aggregate()
        self._any_expanded_group_tags = set()
        self._refresh_after_key_switch()
        self.refresh_lists()
        self.network_panel.draw()

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

    def _exit_spike_attribution_view(self):
        self.center_container.spike_attribution_panel._exit_spike_attribution_view()

    def _on_segment_change(self, event):
        name = self.segment_var.get()
        if name == _ALL_SEGS:
            self.current_segment = self.n_segments
        elif name in self.ccg_ptr.segment_names:
            self.current_segment = self.ccg_ptr.segment_names.index(name)
        self.plot_title_var.set(self.get_plot_title())
        self.center_container.spike_attribution_panel._exit_spike_attribution_view()
        self.update_plot()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _compute_stats_str(self) -> str:
        return self._pair_mgr._compute_stats_str()

    def _refresh_stats(self):
        if hasattr(self, 'stats_var'):
            self.stats_var.set(self._compute_stats_str())

    def _update_pair_info(self, inds):
        return self._plot_mgr._update_pair_info(inds)

    def _flush_any_selections_to_pointers(self):
        self._sess_mgr._flush_any_selections_to_pointers()

    def _any_load_deleted_aggregate(self):
        self._sess_mgr._any_load_deleted_aggregate()

    def _flush_any_deleted_to_stores(self):
        self._sess_mgr._flush_any_deleted_to_stores()

    def _flush_deleted_to_store(self):
        """Copy current deleted_inds into _pair_deleted_store for the active Key."""
        if getattr(self, '_session_any_mode', False):
            self._flush_any_deleted_to_stores()
            return
        if getattr(self, 'key', None) is not None:
            self._pair_deleted_store[str(self.key)] = set(self.deleted_inds)

    def _push_undo(self):
        self._sel_mgr._push_undo()

    @property
    def selected_inds(self):
        return self._sel_data.selected_inds

    @selected_inds.setter
    def selected_inds(self, v):
        self._sel_data.selected_inds = v

    @property
    def unselected_inds(self):
        return self._sel_data.unselected_inds

    @unselected_inds.setter
    def unselected_inds(self, v):
        self._sel_data.unselected_inds = v

    @property
    def deleted_inds(self):
        return self._sel_data.deleted_inds

    @deleted_inds.setter
    def deleted_inds(self, v):
        self._sel_data.deleted_inds = v

    def _undo(self, event=None):
        return self._sel_mgr._undo(event)

    def _redo(self, event=None):
        return self._sel_mgr._redo(event)

    def _highlight_changed_pairs(self, changed_pairs):
        return self._plot_mgr._highlight_changed_pairs(changed_pairs)

    def _clear_undo_highlight(self):
        self._sel_mgr._clear_undo_highlight()

    def refresh_lists(self):
        if hasattr(self, 'left_container'):
            self.left_container.left_panel.refresh_lists()

    def _select_pair_in_list(self, inds):
        if hasattr(self, 'left_container'):
            self.left_container.left_panel._select_pair_in_list(inds)

    def _reapply_bookmark_list_styles(self):
        self._sel_mgr._reapply_bookmark_list_styles()

    def _selected_pair_from_lists(self):
        if hasattr(self, 'left_container'):
            return self.left_container.left_panel._selected_pair_from_lists()
        return None

    def _selected_pairs_from_lists(self):
        if hasattr(self, 'left_container'):
            return self.left_container.left_panel._selected_pairs_from_lists()
        return []

    def _pair_at_all_inds_idx(self, idx: int):
        if hasattr(self, 'left_container'):
            return self.left_container.left_panel._pair_at_all_inds_idx(idx)
        row = self.all_inds[idx]
        return tuple(int(x) for x in row)

    def _bookmark_toggle_current(self, event=None):
        self._sel_mgr._bookmark_toggle_current(event)

    def _clear_bookmarks(self):
        self._sel_mgr._clear_bookmarks()

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

    def get_pair_index(self, inds):
        return self._plot_mgr.get_pair_index(inds)

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
        return self._custom_mgr._is_custom_segment(seg)

    def _custom_seg_index(self, seg=None):
        return self._custom_mgr._custom_seg_index(seg)

    def change_segment(self, delta):
        total = self._n_total_segments()
        self.current_segment = (self.current_segment + delta) % total
        self._clamp_current_segment_for_session()
        self._update_segment_label()
        self.center_container.spike_attribution_panel._exit_spike_attribution_view()
        self.update_plot()

    def _jump_to_segment(self, idx):
        self.current_segment = idx
        self._clamp_current_segment_for_session()
        self._update_segment_label()
        self.center_container.spike_attribution_panel._exit_spike_attribution_view()
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
            if 0 <= seg < len(self.ccg_ptr.segment_names):
                self.segment_var.set(self.ccg_ptr.segment_names[seg])
            else:
                self.segment_var.set(str(seg))
        self.plot_title_var.set(self.get_plot_title())

    def _remove_custom_segment(self, ci):
        self._custom_mgr._remove_custom_segment(ci)

    def _update_sig_indicators(self, inds):
        self._pair_mgr._update_sig_indicators(inds)

    def _png_path(self, inds, segment, _render_cfg=None, _hires_override=None) -> str:
        return self._png_mgr._png_path(inds, segment, _render_cfg, _hires_override)

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
        return self._pair_mgr._resolve_segment_data(ref, tgt, segment, highres, include_pval, include_acg, _cd)

    def _resolve_extended_ccg(self, ref: int, tgt: int, segment: int, highres: bool,
                              extend_ms: int, bin_size_eff: float, cd):
        return self._pair_mgr._resolve_extended_ccg(ref, tgt, segment, highres, extend_ms, bin_size_eff, cd)

    def _get_or_render_png(self, inds, segment):
        """Return PNG path, using disk cache when display matches cache configuration.

        When a cache configuration is set and the current display state does NOT
        match it, the plot is always re-rendered (real-time, no disk-cache reuse).
        """
        is_rt = (self._cache_config is not None
                 and not self._display_matches_cache_config())
        p = self._png_path(inds, segment)
        if is_rt or not os.path.exists(p):
            return self._plot_mgr._render_png(inds, segment)
        return p

    def _git_commit_paths(self, paths: list, msg: str):
        """Stage and commit one or more files to git (background thread)."""
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

    def _save_groups_export(self):
        self._group_mgr._save_groups_export()

    def _autoload_session_latest(self, restore_groups: bool = False):
        return self._sel_mgr._autoload_session_latest(restore_groups)

    def _ensure_session_loaded(self, nd_key, on_loaded):
        return self._sess_mgr._ensure_session_loaded(nd_key, on_loaded)

    def _post_load_refresh(self):
        """Unified UI refresh called after any load (initial, autoload, manual).

        Keeps all three code paths (launch, autoload-latest, load-selection)
        visually in sync.
        """
        self.refresh_lists()
        self._build_sig_chips()
        self._update_segment_label()
        self.network_panel.refresh_shank_buttons()
        self._rebuild_groups_menu()   # also refreshes hotkeys bar chips
        self._update_conn_str_metric_availability()
        self.update_plot()
        self.network_panel.draw()

    # ------------------------------------------------------------------
    # Connectivity Strength
    # ------------------------------------------------------------------

    def _pair_passes_main(self, ref: int, tgt: int, seg) -> bool:
        return self._pair_mgr._pair_passes_main(ref, tgt, seg)
    def _view_values(self, item):
        return self._pair_mgr._view_values(item)

    def _on_conn_str_toggle(self):
        self._cs_mgr._on_conn_str_toggle()

    def _update_conn_str_label(self):
        self._cs_mgr._update_conn_str_label()

    def _cs_annotation_lines(self, ref: int, tgt: int, seg, highres: bool,
                              print_stg: bool, print_jbsi: bool) -> list:
        return self._cs_mgr._cs_annotation_lines(ref, tgt, seg, highres, print_stg, print_jbsi)

    def _update_conn_str_metric_availability(self):
        self._cs_mgr._update_conn_str_metric_availability()

    def _current_inds(self):
        """Return current (ref, tgt) inds or None."""
        if self.network_panel._focused_pair is not None:
            return np.array(self.network_panel._focused_pair)
        if self.current_pair_idx < len(self.all_inds):
            return self.all_inds[self.current_pair_idx]
        return None

    def _get_cs_val(self, ref: int, tgt: int, seg, method: str, highres: bool):
        return self._cs_mgr._get_cs_val(ref, tgt, seg, method, highres)

    def _cs_cache_key(self, ref: int, tgt: int, seg, method: str, highres: bool,
                      eff_min_lag, eff_max_lag) -> tuple:
        return self._cs_mgr._cs_cache_key(ref, tgt, seg, method, highres, eff_min_lag, eff_max_lag)

    def _compute_pair_conn_strength(self, ref: int, tgt: int, seg, highres: bool = False):
        return self._cs_mgr._compute_pair_conn_strength(ref, tgt, seg, highres)

    def _update_global_baseline_availability(self):
        self._cs_mgr._update_global_baseline_availability()

    # ── Adaptive test window ───────────────────────────────────────────

    _ADAPTIVE_TW_GROUPS = ('msconn', 'widems', '2peakms')
    _ADAPTIVE_TW_MIN_LAG = -1e-3   # -1 ms
    _ADAPTIVE_TW_MAX_LAG =  1e-3   #  1 ms

    def _pair_qualifies_for_adaptive_tw(self, ref: int, tgt: int) -> bool:
        return self._cs_mgr._pair_qualifies_for_adaptive_tw(ref, tgt)

    def _effective_lags(self, ref: int, tgt: int):
        return self._cs_mgr._effective_lags(ref, tgt)

    def _update_adaptive_tw_availability(self):
        self._cs_mgr._update_adaptive_tw_availability()

    def _on_adaptive_tw_toggle(self):
        self._cs_mgr._on_adaptive_tw_toggle()

    def _deferred_initial_draw(self):
        # Load groups/hotkeys first (independent of session data).
        self._group_mgr._load_groups_from_export()

        def _after_initial_draw():
            self._setup_mgr._finish_initial_draw()
            # Autoload selections + pair tags AFTER _finish_initial_draw so the
            # reset in that method doesn't clobber what we loaded.
            self._autoload_session_latest(restore_groups=False)
            self.refresh_lists()
            self.network_panel.draw()

        self._ensure_session_loaded(self.key.nd(), on_loaded=_after_initial_draw)

    def update_plot(self):
        return self._plot_mgr.update_plot()

    def _draw_waveforms(self):
        return self._pair_mgr._draw_waveforms()

    # ------------------------------------------------------------------
    # Time slider
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_ts_time(val, t_start: float, t_end: float) -> float:
        """Resolve a t0/t1 spec value — float, or sentinel string 'start'/'end'.

        Numeric values are clamped to ``[min(t_start,t_end), max(...)]`` so a window
        built on one session's timeline does not run out-of-range on another session.
        """
        lo, hi = (t_start, t_end) if t_end >= t_start else (t_end, t_start)
        if hi <= lo:
            hi = lo + 1e-6
        if val is None or (isinstance(val, str) and str(val).strip().lower() == 'end'):
            return hi
        if isinstance(val, str) and str(val).strip().lower() == 'start':
            return lo
        x = float(val)
        return min(max(x, lo), hi)

    def _pick_sessions_dialog(self, title: str, sessions: list[str],
                              current_session: str | None = None) -> list[str] | None:
        """Open a session picker dialog; returns selected session strings or None on cancel."""
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("400x340")
        win.resizable(True, True)
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
        win.update_idletasks()
        win.lift()
        win.focus_force()
        win.wait_window()
        return result[0]

    def _generate_suggested_custom_ccgs(self):
        return self._custom_mgr._generate_suggested_custom_ccgs()

    def _session_obj_for_nd_key(self, nd_key):
        return self._sess_mgr._session_obj_for_nd_key(nd_key)

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

    def _session_wall_clock_extent_for_key(self, key) -> tuple[float, float]:
        return self._sess_mgr._session_wall_clock_extent_for_key(key)

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
    def _is_legacy_name(name: str) -> bool:
        """Strip legacy auto-generated post-window suggestion names from lists/UI."""
        n = re.sub(r'[\s_]+', '', str(name).strip().lower())
        return n in ('post1', 'post2', 'post1st', 'post2nd', 'postfirst', 'postsecond')

    def _theme_bounds_for_key(self, key):
        return self._pair_mgr._theme_bounds_for_key(key)

    def _intervals_for_spec_on_key(self, spec: dict, key):
        bounds = self._theme_bounds_for_key(key)
        if bounds is None:
            print(f"[CustomCCG] missing behavioral epochs: {key.session}")
            return None
        # Resolve 'start'/'end' sentinels to session bounds.
        # Do NOT re-expand to epoch union span — caller already resolved epoch →
        # per-chunk t0/t1; re-expanding would collapse all chunks to the same interval.
        t_sess_start, t_sess_end = self._session_wall_clock_extent_for_key(key)
        t0 = self._resolve_ts_time(spec.get('t0', 0.0), t_sess_start, t_sess_end)
        t1 = self._resolve_ts_time(spec.get('t1', t_sess_end), t_sess_start, t_sess_end)
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
        return self._custom_mgr._custom_npz_spec(path)

    def _custom_ccg_name_session_coverage(self) -> tuple[dict[str, set[str]], int]:
        return self._custom_mgr._custom_ccg_name_session_coverage()

    def _custom_segment_disk_session(self, cs: dict) -> str:
        return self._custom_mgr._custom_segment_disk_session(cs)

    def _bind_custom_segments_to_session(self, sess: str):
        self._custom_mgr._bind_custom_segments_to_session(sess)

    def _key_for_custom_segment_save(self, cs: dict):
        return self._custom_mgr._key_for_custom_segment_save(cs)

    def _enqueue_custom_ccg_task(self, *, key, t0, t1, name, intervals,
                                 active_duration, filter_state, metadata,
                                 auto_save: bool, load_into_ui: bool,
                                 batch_id: int | None = None) -> bool:
        return self._custom_mgr._enqueue_custom_ccg_task(
            key=key, t0=t0, t1=t1, name=name, intervals=intervals,
            active_duration=active_duration, filter_state=filter_state,
            metadata=metadata, auto_save=auto_save, load_into_ui=load_into_ui,
            batch_id=batch_id)

    def _iter_type_keys_for_all_sessions(self):
        return self._sess_mgr._iter_type_keys_for_all_sessions()

    def _queue_custom_ccg_for_spec(self, spec: dict, *, for_all: bool, auto_save: bool,
                                    target_sessions: list | None = None) -> int:
        return self._custom_mgr._queue_custom_ccg_for_spec(
            spec, for_all=for_all, auto_save=auto_save, target_sessions=target_sessions)

    def _compute_custom_segment(self, t0: float, t1: float, name: str,
                                neurons_override=None, active_duration=None,
                                key_override=None, neurons_obj=None,
                                ccg_data_obj=None, metadata=None):
        return self._custom_mgr._compute_custom_segment(t0, t1, name, neurons_override, active_duration, key_override, neurons_obj, ccg_data_obj, metadata)

    def _on_add_focused_pair(self):
        return self._misc_mgr._on_add_focused_pair()

    def _pair_group_label(self, inds) -> str:
        return self._group_mgr._pair_group_label(inds)

    def _toggle_pair_group(self, pair, group_name):
        self._group_mgr._toggle_pair_group(pair, group_name)

    def _toggle_pairs_group(self, pairs, group_name):
        self._group_mgr._toggle_pairs_group(pairs, group_name)

    def _create_group_dialog(self):
        return self._group_mgr._create_group_dialog()

    def _create_special_group_dialog(self):
        return self._group_mgr._create_special_group_dialog()

    def _pair_tags_dialog(self):
        """Dialog to view/edit tags and notes for the current pair."""
        if self.current_pair_idx >= len(self.all_inds):
            messagebox.showinfo("Pair tags", "No pair selected.")
            return
        inds = tuple(self.all_inds[self.current_pair_idx])
        ref, tgt = int(inds[0]), int(inds[1])
        tag_data = self._sel_data._pair_tags.get((ref, tgt), {})

        win = tk.Toplevel(self.root)
        win.title(f"Pair Tags — [{ref}, {tgt}]")
        win.geometry("400x350")
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
                existing_groups = self._sel_data._pair_tags.get((ref, tgt), {}).get('groups', [])
                entry = {'tags': tags, 'notes': notes}
                if existing_groups:
                    entry['groups'] = existing_groups
                self._sel_data._pair_tags[(ref, tgt)] = entry
            elif (ref, tgt) in self._sel_data._pair_tags:
                del self._sel_data._pair_tags[(ref, tgt)]
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
        return self._misc_mgr._pairs_by_conn_type(session_str, pairs)

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

    def _manage_groups_dialog(self):
        ManageGroupsDialog.show(self)

    def _rename_group(self, old_name, new_name, win=None):
        return self._group_mgr._rename_group(old_name, new_name, win)

    def _delete_group(self, name, win=None):
        return self._group_mgr._delete_group(name, win)

    # ------------------------------------------------------------------
    # Template editor
    # ------------------------------------------------------------------

    def _set_group_hotkey(self, group_name, key_str):
        return self._group_mgr._set_group_hotkey(group_name, key_str)

    def _rebuild_groups_menu(self):
        return self._group_mgr._rebuild_groups_menu()

    def _select_group(self, group_name):
        return self._group_mgr._select_group(group_name)

    def _show_temp_warning(self, msg: str, duration_ms: int = 2000):
        return self._sel_mgr._show_temp_warning(msg, duration_ms)

    def _upsert_custom_segment_by_name(self, lst: list, result: dict) -> tuple:
        return self._custom_mgr._upsert_custom_segment_by_name(lst, result)

    def _on_chunk_done(self, task):
        return self._custom_mgr._on_chunk_done(task)

    # ── Property shims: attrs moved to CustomCCGManager ─────────────────────

    @property
    def _custom_segments(self) -> list:
        return self._custom_mgr._active_list

    @_custom_segments.setter
    def _custom_segments(self, v):
        self._custom_mgr._by_session[self._custom_mgr._active_sess] = v

    @property
    def _custom_segments_by_session(self) -> dict:
        return self._custom_mgr._by_session

    @property
    def _extend_cache(self) -> dict:
        return self._custom_mgr._extend_cache

    @_extend_cache.setter
    def _extend_cache(self, v):
        self._custom_mgr._extend_cache = v

    @property
    def _stacked_segments(self) -> set:
        return self._custom_mgr._stacked_segments

    @_stacked_segments.setter
    def _stacked_segments(self, v):
        self._custom_mgr._stacked_segments = v

    @property
    def _custom_ccg_inventory_sig(self) -> tuple:
        return self._custom_mgr._inventory_sig

    @_custom_ccg_inventory_sig.setter
    def _custom_ccg_inventory_sig(self, v):
        self._custom_mgr._inventory_sig = v

    @property
    def _ccg_cache_dir(self) -> str:
        return self._custom_mgr.cache_dir

    @property
    def _custom_ccg_suggestions_path(self) -> str:
        return self._custom_mgr.suggestions_path

    def _emit_inventory_event(self):
        return self._custom_mgr._emit_inventory_event()

    def _bind_custom_segments_to_session(self, sess: str):
        return self._custom_mgr._bind_custom_segments_to_session(sess)

    def _archive_stale_custom_ccgs(self):
        return self._custom_mgr._archive_stale_custom_ccgs()

    def _build_custom_spec(self, *, for_all: bool, for_session: str | None = None):
        return self._custom_mgr._build_custom_spec(for_all=for_all, for_session=for_session)

    def _record_custom_ccg_suggestion(self, spec: dict):
        return self._custom_mgr._record_custom_ccg_suggestion(spec)

    def _emit_custom_ccg_inventory_event(self):
        return self._custom_mgr._emit_inventory_event()

    def _on_close(self):
        return self._setup_mgr._on_close()

    def run(self):
        return self._sel_mgr.run()

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
