"""Dialog helper classes for CCGReviewUI.

Each class wraps one dialog interaction. Call <ClassName>.show(ui) from
CCGReviewUI methods.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from neuropy.ui.utils import _SPECIAL_PREFIX, is_special_group

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from neuropy.ui.ccg_ui import CCGReviewUI

class MergeGroupsDialog:
    @classmethod
    def show(cls, ui: "CCGReviewUI") -> None:
        if len(ui._sel_data._groups) < 2:
            messagebox.showinfo("Merge groups", "Need at least 2 groups to merge.")
            return
        cls(ui).win.wait_window()

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        self.win = tk.Toplevel(ui.root)
        self.win.title("Merge Groups")
        self.win.geometry("340x320")
        self.win.grab_set()
        self._build()

    def _build(self):
        ui = self._ui
        win = self.win
        ttk.Label(win, text="Select groups to merge:",
                  font=('Arial', 10, 'bold')).pack(pady=(8, 4))
        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10)
        check_vars = {}
        for gname in sorted(ui._sel_data._groups):
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
                g_data = ui._sel_data._groups.get(g, {})
                if isinstance(g_data, set):
                    sess = ui._setup_mgr._current_session_str()
                    merged.setdefault(sess, set()).update(g_data)
                else:
                    for sess, pairs in g_data.items():
                        merged.setdefault(sess, set()).update(pairs)
                if g != target:
                    ui._sel_data._groups.pop(g, None)
                    ui._sel_data._group_hotkeys.pop(g, None)
                    ui._sel_data._group_notes.pop(g, None)
            ui._sel_data._groups[target] = merged
            ui._group_mgr._rebuild_groups_menu()
            ui.refresh_lists()
            win.destroy()

        ttk.Button(win, text="Merge", command=do_merge).pack(pady=8)

class ManageGroupsDialog:
    @classmethod
    def show(cls, ui: "CCGReviewUI") -> None:
        if not ui._sel_data._groups:
            messagebox.showinfo("Manage groups", "No groups yet. Create one first.")
            return
        cls(ui).win.wait_window()

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        self.win = tk.Toplevel(ui.root)
        self.win.title("Manage Groups")
        self.win.geometry("480x420")
        self.win.grab_set()
        self._build()

    def _add_group_tab(self, nb, gname, is_special=False):
        ui = self._ui
        win = self.win
        display = gname[len(_SPECIAL_PREFIX):] if is_special else gname
        frame = ttk.Frame(nb)
        nb.add(frame, text=display)
        top = ttk.Frame(frame)
        top.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(top, text="Name:").pack(side=tk.LEFT)
        name_var = tk.StringVar(value=display)
        ttk.Entry(top, textvariable=name_var, width=18).pack(side=tk.LEFT, padx=4)

        def _do_rename(old=gname, nv=name_var, sp=is_special):
            new = nv.get().strip()
            if sp:
                new = _SPECIAL_PREFIX + new
            ui._group_mgr._rename_group(old, new, win)

        ttk.Button(top, text="Rename", command=_do_rename).pack(side=tk.LEFT)
        if not is_special:
            hk_frame = ttk.Frame(frame)
            hk_frame.pack(fill=tk.X, padx=6, pady=2)
            ttk.Label(hk_frame, text="Hotkey (1–9/0/a–z):").pack(side=tk.LEFT)
            hk_var = tk.StringVar(value=ui._sel_data._group_hotkeys.get(gname, ''))
            ttk.Entry(hk_frame, textvariable=hk_var, width=6).pack(side=tk.LEFT, padx=4)
            ttk.Button(hk_frame, text="Set",
                       command=lambda g=gname, hv=hk_var: ui._group_mgr._set_group_hotkey(g, hv.get())).pack(
                side=tk.LEFT)
        ttk.Label(frame, text="Discussion notes:" if is_special else "Notes:"
                  ).pack(anchor='w', padx=6, pady=(4, 0))
        notes_h = 10 if is_special else 3
        notes_text = tk.Text(frame, height=notes_h, width=40,
                             font=('Arial', 9), wrap=tk.WORD)
        notes_text.pack(fill=tk.BOTH if is_special else tk.X,
                        expand=is_special, padx=6, pady=2)
        notes_text.insert('1.0', ui._sel_data._group_notes.get(gname, ''))
        notes_text.bind('<KeyRelease>',
                        lambda e, g=gname, t=notes_text:
                        ui._sel_data._group_notes.__setitem__(g, t.get('1.0', 'end-1c')))
        ttk.Label(frame, text="Pairs in group:").pack(anchor='w', padx=6)
        lb_frame = ttk.Frame(frame)
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        sb = ttk.Scrollbar(lb_frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        lb = tk.Listbox(lb_frame, yscrollcommand=sb.set, font=('Courier', 9))
        lb.pack(fill=tk.BOTH, expand=True)
        sb.config(command=lb.yview)
        g = ui._sel_data._groups.get(gname, {})
        if isinstance(g, dict):
            for sess in sorted(g):
                pairs = g[sess]
                if not pairs:
                    continue
                ct_map = ui._misc_mgr._pairs_by_conn_type(sess, pairs)
                for ct_label, ct_pairs in ct_map.items():
                    lb.insert(tk.END, f"── {sess} | {ct_label} ──")
                    lb.itemconfig(lb.size() - 1, foreground='#666')
                    for pair in sorted(ct_pairs):
                        lb.insert(tk.END, f"  [{pair[0]:3d}, {pair[1]:3d}]")
        else:
            for pair in sorted(g):
                lb.insert(tk.END, f"[{pair[0]:3d}, {pair[1]:3d}]")
        btn_row = ttk.Frame(frame)
        btn_row.pack(pady=4)
        if is_special:
            conv_label = "Convert to group"
            def _do_convert(g=gname, d=display):
                ui._group_mgr._rename_group(g, d, win)
        else:
            conv_label = "Convert to special group"
            def _do_convert(g=gname, d=display):
                ui._group_mgr._rename_group(g, _SPECIAL_PREFIX + d, win)
        ttk.Button(btn_row, text=conv_label, command=_do_convert).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text=f"Delete group '{display}'",
                   command=lambda g=gname: ui._group_mgr._delete_group(g, win)).pack(side=tk.LEFT, padx=4)

    def _build(self):
        nb = ttk.Notebook(self.win)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        for gname in sorted(self._ui._sel_data._groups):
            if gname.startswith('__'):
                continue
            self._add_group_tab(nb, gname, is_special=False)
        special_names = sorted(g for g in self._ui._sel_data._groups
                               if g.startswith(_SPECIAL_PREFIX))
        if special_names:
            special_frame = ttk.Frame(nb)
            nb.add(special_frame, text="Special")
            snb = ttk.Notebook(special_frame)
            snb.pack(fill=tk.BOTH, expand=True)
            for gname in special_names:
                self._add_group_tab(snb, gname, is_special=True)
        ttk.Button(self.win, text="Close", command=self.win.destroy).pack(pady=4)

class SimulationDialog:
    @classmethod
    def show(cls, ui: "CCGReviewUI") -> None:
        cls(ui)  # non-modal — no wait_window

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        self._sim_state: dict = {}
        self.win = tk.Toplevel(ui.root)
        self.win.title("CCG Simulation")
        self.win.geometry("620x780")
        self._build()

    def _build(self):
        ui = self._ui
        win = self.win
        # Button row anchored to bottom — pack before PanedWindow so it's always visible
        sim_res_label_var = tk.StringVar(value="Res: lowres")
        btn_frame = ttk.Frame(win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(0, 6))

        pw = tk.PanedWindow(win, orient=tk.VERTICAL,
                            sashrelief=tk.RAISED, sashwidth=5, bg='#CCCCCC')
        pw.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))
        param_frame = ttk.Frame(pw, padding=6)
        pw.add(param_frame, stretch='never')
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
        common = ttk.Frame(param_frame)
        common.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(common, text="Noise (gauss σ):").pack(side=tk.LEFT)
        sim_noise_var = tk.StringVar(value="0.0")
        ttk.Entry(common, textvariable=sim_noise_var, width=7).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(common, text="Excess synchrony (%):").pack(side=tk.LEFT)
        sim_sync_var = tk.StringVar(value="0")
        ttk.Entry(common, textvariable=sim_sync_var, width=7).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(common, text="Synaptic delay (ms):").pack(side=tk.LEFT)
        sim_delay_var = tk.StringVar(value="1.5")
        ttk.Entry(common, textvariable=sim_delay_var, width=7).pack(side=tk.LEFT, padx=(2, 0))
        cols = ttk.Frame(param_frame)
        cols.pack(fill=tk.X, pady=(0, 6))
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)
        sim_vars = {}
        for col_idx, (role, title) in enumerate([('ref', 'Ref neuron'), ('tgt', 'Tgt neuron')]):
            panel = ttk.LabelFrame(cols, text=title, padding=8)
            panel.grid(row=0, column=col_idx, sticky='nsew',
                       padx=(0 if col_idx == 0 else 4, 0))
            v = {}
            ttk.Label(panel, text="Nickname:").pack(anchor='w')
            v['nickname'] = tk.StringVar(value=role)
            ttk.Entry(panel, textvariable=v['nickname'], width=16).pack(anchor='w', pady=(0, 6))
            ttk.Label(panel, text="Type:").pack(anchor='w')
            v['type'] = tk.StringVar(value="E")
            type_frame = ttk.Frame(panel)
            type_frame.pack(anchor='w', pady=(0, 6))
            for t in ("E", "I", "any"):
                ttk.Radiobutton(type_frame, text=t, variable=v['type'],
                                value=t).pack(side=tk.LEFT, padx=(0, 6))
            ttk.Label(panel, text="Firing rate (Hz):").pack(anchor='w')
            v['firing_rate'] = tk.StringVar(value="5.0")
            ttk.Entry(panel, textvariable=v['firing_rate'], width=10).pack(anchor='w', pady=(0, 6))
            ttk.Label(panel, text="Burst config:",
                      font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', pady=(4, 2))
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
        bottom_frame = ttk.Frame(pw, padding=6)
        pw.add(bottom_frame, stretch='always')
        sim_fig = Figure(figsize=(6, 3.5))
        sim_ax = sim_fig.add_subplot(111)
        sim_ax.set_title("(no simulation run yet)", fontsize=10)
        sim_canvas = FigureCanvasTkAgg(sim_fig, master=bottom_frame)
        ttk.Button(btn_frame, text="Compute CCG", command=lambda: ui._run_simulation(
            win, sim_name_var, sim_dur_var, sim_dur_unit,
            sim_noise_var, sim_sync_var, sim_delay_var,
            sim_vars, sim_fig, sim_ax, sim_canvas, self._sim_state,
            sim_res_label_var)).pack(side=tk.LEFT)
        ttk.Button(
            btn_frame, textvariable=sim_res_label_var,
            command=lambda: ui._sim_toggle_resolution(
                self._sim_state, sim_fig, sim_ax, sim_canvas, sim_res_label_var),
            width=16).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.RIGHT)
        sim_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def _sim_ctrl_r(_e=None):
            ui._sim_toggle_resolution(
                self._sim_state, sim_fig, sim_ax, sim_canvas, sim_res_label_var)
            return 'break'

        win.bind('<Control-r>', _sim_ctrl_r)
        win.bind('<Command-r>', _sim_ctrl_r)

class SettingsDialog:
    @classmethod
    def show(cls, ui: "CCGReviewUI") -> None:
        cls(ui).win.wait_window()

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        self.win = tk.Toplevel(ui.root)
        self.win.title("Settings")
        self.win.geometry("620x420")
        self.win.resizable(True, True)
        self.win.protocol("WM_DELETE_WINDOW", self.win.destroy)
        self.win.bind('<Escape>', lambda e: self.win.destroy())
        self._build()

    def _build(self):
        ui = self._ui
        win = self.win
        style = ttk.Style(win)
        _raw_bg = style.lookup('TFrame', 'background') or ui.root.cget('bg')
        try:
            _rgb = win.winfo_rgb(_raw_bg)
            _lum = (0.299 * _rgb[0] + 0.587 * _rgb[1] + 0.114 * _rgb[2]) / 65535
        except Exception:
            _lum = 1.0
        _dark = _lum < 0.4
        _CONT_BG  = '#1e1e1e'   if _dark else 'white'
        _NAV_BG   = '#252526'   if _dark else '#f3f3f3'
        _NAV_SEL  = '#37373d'   if _dark else '#dce8f5'
        _FG       = '#cccccc'   if _dark else '#111111'
        _FG_DIM   = '#888888'   if _dark else '#666666'
        _SUM_BG   = '#2d2d2d'   if _dark else '#f8f8f8'
        _HDR_FONT = ('Arial', 13, 'bold')
        _LBL_FONT = ('Arial', 10)

        bot = tk.Frame(win, bg=_NAV_BG)
        ttk.Separator(win).pack(side=tk.BOTTOM, fill=tk.X)
        bot.pack(side=tk.BOTTOM, fill=tk.X)

        def _apply():
            try:
                v = int(_max_tog_var.get())
                if 2 <= v <= 20:
                    ui._settings['max_show_together'] = v
                    ui._settings_mgr.save_ui_state()
            except (ValueError, tk.TclError):
                pass
            try:
                fs = int(_min_font_var.get())
                if 6 <= fs <= 32:
                    ui._settings['min_font_size'] = fs
                    ui._settings_mgr.save_ui_state()
                    ui._settings_mgr.apply_min_font_size()
            except (ValueError, tk.TclError):
                pass
            win.destroy()

        ttk.Button(bot, text="Save", command=_apply).pack(side=tk.RIGHT, padx=8, pady=6)
        ttk.Button(bot, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=0, pady=6)

        sidebar = tk.Frame(win, bg=_NAV_BG, width=160)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        ttk.Separator(win, orient='vertical').pack(side=tk.LEFT, fill=tk.Y)

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

        disp = _add_section('Display')
        tk.Label(disp, text="Display", bg=_CONT_BG, fg=_FG,
                 font=_HDR_FONT, anchor='w').pack(fill=tk.X, pady=(0, 12))
        ttk.Separator(disp).pack(fill=tk.X, pady=(0, 10))
        row = tk.Frame(disp, bg=_CONT_BG)
        row.pack(fill=tk.X, pady=6)
        tk.Label(row, text="Max pairs in 'Show Together':", bg=_CONT_BG,
                 fg=_FG, font=_LBL_FONT).pack(side=tk.LEFT)
        _max_tog_var = tk.IntVar(value=ui._settings.get('max_show_together', 5))
        ttk.Spinbox(row, from_=2, to=20, textvariable=_max_tog_var,
                    width=5).pack(side=tk.LEFT, padx=10)
        tk.Label(row, text="(2–20)", bg=_CONT_BG,
                 fg=_FG_DIM, font=('Arial', 9)).pack(side=tk.LEFT)
        row = tk.Frame(disp, bg=_CONT_BG)
        row.pack(fill=tk.X, pady=6)
        tk.Label(row, text="Minimum font size:", bg=_CONT_BG,
                 fg=_FG, font=_LBL_FONT).pack(side=tk.LEFT)
        _min_font_var = tk.IntVar(value=int(ui._settings.get('min_font_size', 9) or 9))
        ttk.Spinbox(row, from_=6, to=32, textvariable=_min_font_var,
                    width=5).pack(side=tk.LEFT, padx=10)
        tk.Label(row, text="(6–32)", bg=_CONT_BG,
                 fg=_FG_DIM, font=('Arial', 9)).pack(side=tk.LEFT)


class ExportOptionsDialog:
    """Export options dialog — returns option dict or None if cancelled."""

    @classmethod
    def show(cls, ui: "CCGReviewUI", fmt: str, preview_pair=None, selected_pairs=None):
        dlg = cls(ui, fmt, preview_pair, selected_pairs)
        dlg.win.wait_window()
        if not dlg._out:
            return None
        dlg._out['_action'] = dlg._action.get('mode', 'current')
        dlg._out['_selected_pairs'] = list(selected_pairs) if selected_pairs else []
        dlg._out['_preview_pair'] = preview_pair
        dlg._out['_fmt'] = fmt
        dlg._out['_selected_groups'] = list(dlg._selected_groups)
        return dlg._out

    def __init__(self, ui: "CCGReviewUI", fmt: str, preview_pair=None, selected_pairs=None):
        self._ui = ui
        self._fmt = fmt
        self._preview_pair = preview_pair
        self._selected_pairs = selected_pairs
        self._out: dict = {}
        self._action: dict = {'mode': 'current'}
        self._selected_groups: list = []
        self._selected_segments: list = []
        self._selected_subfolders: list = []
        self._prev_img: dict = {'obj': None}
        self._build()

    def _build(self):
        ui = self._ui
        fmt = self._fmt
        preview_pair = self._preview_pair
        selected_pairs = self._selected_pairs
        self.win = tk.Toplevel(ui.root)
        self.win.title("Export options")
        self.win.resizable(True, True)

        # ------------------------------
        # Group selection (export scope)
        # ------------------------------
        # Available groups exclude internal/special/private entries.
        all_groups = sorted(
            g for g in ui._sel_data._groups.keys()
            if g and not str(g).startswith('__')
        )
        avail_groups = [g for g in all_groups if not str(g).startswith(_SPECIAL_PREFIX)]
        self._selected_groups: list = []

        # Export defaults persisted in ui_state.json via ui._settings
        _exp_def = (ui._settings.get('export_defaults', {}) if isinstance(getattr(self, '_settings', None), dict) else {}) or {}
        def _sv(key, default=""):
            return tk.StringVar(value=str(_exp_def.get(key) or default))
        def _bv(key, default=True):
            return tk.BooleanVar(value=bool(_exp_def.get(key, default)))

        ccg_var              = _sv('ccg_color')
        base_var             = _sv('baseline_color')
        cs_shade_var         = _sv('cs_shade_color')
        tw_color_var         = _sv('test_window_color')
        tw_alpha_var         = _sv('test_window_alpha',      "0.12")
        pval_line_color_var  = _sv('pval_line_color')
        alpha_line_color_var = _sv('alpha_line_color')
        minfs_var            = _sv('min_text_size',          "8")
        ccg_a_var            = _sv('ccg_alpha',              "0.5")
        base_a_var           = _sv('baseline_alpha',         "0.3")
        xticks_var           = _sv('xticks_raw')
        show_prev_var        = tk.BooleanVar(value=False)
        show_legend_var      = _bv('show_legend',            True)
        mirror_ticks_var     = _bv('mirror_xticks',          True)
        adaptive_tw_var      = _bv('adaptive_tw_export',     False)
        title_shanks_var     = _bv('title_show_shanks',      True)
        title_inds_var       = _bv('title_show_inds',        True)
        title_type_var       = _bv('title_show_type',        True)
        title_seg_var        = _bv('title_show_seg',         True)
        title_norm_var       = _bv('title_show_norm_details', True)
        title_sess_var       = _bv('title_show_session',     False)
        # Segment export selection (multi): union of segment labels across loaded types for this session.
        # Stored as a list of strings in export_defaults['export_segments'].
        seg_export_default = list(_exp_def.get('export_segments') or []) if isinstance(_exp_def, dict) else []
        if not seg_export_default:
            seg_export_default = ["Current"]
        try:
            nd_key = ui.key.nd() if ui.key is not None else None
            type_keys = ui._sess_mgr._available_type_keys(nd_key) if nd_key is not None else []
            seg_union: set[str] = set()
            for tk_ in type_keys:
                ptr = ui.cd.ptr.get(tk_)
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
            # Unique custom segment names across all sessions
            custom_seg_names: set[str] = set()
            for _csl in (getattr(ui, '_custom_segments_by_session', None) or {}).values():
                for _cs in (_csl or []):
                    if isinstance(_cs, dict) and _cs.get('name'):
                        custom_seg_names.add(str(_cs['name']))
            seg_export_choices = (["Current", "All"] + sorted(seg_union)
                                  + sorted(custom_seg_names - seg_union))
        except Exception:
            seg_export_choices = ["Current", "All"]

        # Subfolder hierarchy selection (multi + order)
        subfolder_default = list(_exp_def.get('subfolder_by') or []) if isinstance(_exp_def, dict) else []
        _subfolder_choices = ["conn type", "excitatory/inhibitory", "session", "animal"]
        # sanitize defaults
        subfolder_default = [x for x in subfolder_default if x in _subfolder_choices]

        # Scrollable body (export options can grow tall)
        outer = ttk.Frame(self.win)
        outer.grid(row=0, column=0, sticky="nsew")
        self.win.columnconfigure(0, weight=1)
        self.win.rowconfigure(0, weight=1)

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
            try:
                canvas.itemconfigure(win_id, width=event.width)
            except Exception:
                pass

        frm.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        frm.columnconfigure(1, weight=1)

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
            if self._selected_groups:
                selected_tags_var.set("Selected group tags: " + ", ".join(self._selected_groups))
            else:
                selected_tags_var.set("")

        def _add_groups():
            sel = list(avail_lb.curselection())
            if not sel:
                return
            names = [avail_lb.get(i) for i in sel]
            # Remove from bottom-up to keep indices valid
            for i in sorted(sel, reverse=True):
                avail_lb.delete(i)
            for g in names:
                if g not in self._selected_groups:
                    self._selected_groups.append(g)
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
                    self._selected_groups.remove(g)
                except ValueError:
                    pass
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

        row0 = 1
        _color_fields = [
            ("CCG color (name or #hex):",          ccg_var,             22, "ew"),
            ("Baseline color (name or #hex):",      base_var,            22, "ew"),
            ("CS shade color (name or #hex):",      cs_shade_var,        22, "ew"),
            ("Test window color (name or #hex):",   tw_color_var,        22, "ew"),
            ("Test window alpha (0–1):",            tw_alpha_var,        10, "w"),
            ("P-value line color (name or #hex):",  pval_line_color_var, 22, "ew"),
            ("Alpha threshold color (name or #hex):", alpha_line_color_var, 22, "ew"),
            ("Min text size (pt):",                 minfs_var,           10, "w"),
            ("CCG alpha (0–1):",                    ccg_a_var,           10, "w"),
            ("Baseline alpha (0–1):",               base_a_var,          10, "w"),
        ]
        for i, (label, var, width, e_sticky) in enumerate(_color_fields):
            ttk.Label(frm, text=label).grid(row=row0 + i, column=0, sticky="w", pady=4)
            ttk.Entry(frm, textvariable=var, width=width).grid(row=row0 + i, column=1, sticky=e_sticky, padx=(8, 0), pady=4)

        ttk.Checkbutton(frm, text="Show legend", variable=show_legend_var).grid(
            row=row0 + 10, column=0, sticky="w", pady=(6, 0))

        ttk.Label(frm, text="X ticks (ms, comma-separated):").grid(row=row0 + 11, column=0, sticky="w", pady=4)
        ttk.Entry(frm, textvariable=xticks_var, width=28).grid(row=row0 + 11, column=1, sticky="ew", padx=(8, 0), pady=4)
        ttk.Checkbutton(frm, text="Mirror to negative ticks", variable=mirror_ticks_var).grid(
            row=row0 + 12, column=0, sticky="w", pady=(0, 4))
        title_frame = ttk.LabelFrame(frm, text="Title")
        title_frame.grid(row=row0 + 13, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        for _text, _var, _px in [("Shanks",       title_shanks_var, (4, 8)),
                                  ("Inds",         title_inds_var,   (0, 8)),
                                  ("Type",         title_type_var,   (0, 8)),
                                  ("Segment name", title_seg_var,    (0, 8)),
                                  ("Norm details", title_norm_var,   (0, 8)),
                                  ("Session",      title_sess_var,   (0, 4))]:
            ttk.Checkbutton(title_frame, text=_text, variable=_var).pack(side=tk.LEFT, padx=_px)
        print_stg_var  = tk.BooleanVar(value=bool(_exp_def.get('print_cs_stg',  False)))
        print_jbsi_var = tk.BooleanVar(value=bool(_exp_def.get('print_cs_jbsi', False)))
        tw_cs_frame = ttk.Frame(frm)
        tw_cs_frame.grid(row=row0 + 14, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Checkbutton(tw_cs_frame, text="Adaptive test window",
                        variable=adaptive_tw_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(tw_cs_frame, text="Print CS:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Checkbutton(tw_cs_frame, text="STG",  variable=print_stg_var).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Checkbutton(tw_cs_frame, text="JBSI", variable=print_jbsi_var).pack(side=tk.LEFT)
        lores_var  = tk.BooleanVar(value=bool(_exp_def.get('export_lores', True)))
        hires_var  = tk.BooleanVar(value=bool(_exp_def.get('export_hires', False)))
        res_frame  = ttk.Frame(frm)
        res_frame.grid(row=row0 + 15, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Label(res_frame, text="Resolution:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Checkbutton(res_frame, text="Lo-res", variable=lores_var).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Checkbutton(res_frame, text="Hi-res", variable=hires_var).pack(side=tk.LEFT)

        # Segments selection UI (same pattern as Groups)
        seg_frame = ttk.LabelFrame(frm, text="Segments to export")
        seg_frame.grid(row=row0 + 16, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        seg_frame.columnconfigure(0, weight=1)
        seg_frame.columnconfigure(2, weight=1)
        ttk.Label(seg_frame, text="Available:").grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
        ttk.Label(seg_frame, text="Selected:").grid(row=0, column=2, sticky="w", padx=6, pady=(6, 2))

        seg_avail_lb = tk.Listbox(seg_frame, height=6, exportselection=False)
        seg_sel_lb = tk.Listbox(seg_frame, height=6, exportselection=False)
        for s in seg_export_choices:
            seg_avail_lb.insert(tk.END, s)
        # preload selected segments
        self._selected_segments: list = []
        for s in seg_export_default:
            if s in seg_export_choices and s not in self._selected_segments:
                self._selected_segments.append(s)
                seg_sel_lb.insert(tk.END, s)
        # remove preselected from available
        if self._selected_segments:
            cur = [seg_avail_lb.get(i) for i in range(seg_avail_lb.size())]
            seg_avail_lb.delete(0, tk.END)
            for s in cur:
                if s not in self._selected_segments:
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
                if s not in self._selected_segments:
                    self._selected_segments.append(s)
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
                    self._selected_segments.remove(s)
                except ValueError:
                    pass
            cur = list(seg_avail_lb.get(0, tk.END))
            cur.extend(names)
            cur = sorted(set(cur), key=lambda x: (0 if x == "Current" else 1 if x == "All" else 2, x))
            seg_avail_lb.delete(0, tk.END)
            for s in cur:
                if s not in self._selected_segments:
                    seg_avail_lb.insert(tk.END, s)

        ttk.Button(seg_mid, text="Add →", command=_add_segments).pack(pady=(10, 6))
        ttk.Button(seg_mid, text="← Remove", command=_remove_segments).pack()
        seg_avail_lb.bind("<Double-Button-1>", lambda e: _add_segments())
        seg_sel_lb.bind("<Double-Button-1>", lambda e: _remove_segments())

        # Subfolder hierarchy UI (dual list; selected list is reorderable by drag)
        sf_frame = ttk.LabelFrame(frm, text="Subfolder by (optional)")
        sf_frame.grid(row=row0 + 17, column=0, columnspan=2, sticky="ew", pady=(6, 6))
        sf_frame.columnconfigure(0, weight=1)
        sf_frame.columnconfigure(2, weight=1)
        ttk.Label(sf_frame, text="Available:").grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
        ttk.Label(sf_frame, text="Selected (drag to reorder):").grid(row=0, column=2, sticky="w", padx=6, pady=(6, 2))

        sf_avail_lb = tk.Listbox(sf_frame, height=4, exportselection=False)
        sf_sel_lb = tk.Listbox(sf_frame, height=4, exportselection=False)
        self._selected_subfolders: list[str] = list(subfolder_default)
        for s in _subfolder_choices:
            if s not in self._selected_subfolders:
                sf_avail_lb.insert(tk.END, s)
        for s in self._selected_subfolders:
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
                if s not in self._selected_subfolders:
                    self._selected_subfolders.append(s)
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
                    self._selected_subfolders.remove(s)
                except ValueError:
                    pass
            cur = list(sf_avail_lb.get(0, tk.END))
            cur.extend(names)
            cur = sorted(set(cur), key=lambda x: _subfolder_choices.index(x) if x in _subfolder_choices else 999)
            sf_avail_lb.delete(0, tk.END)
            for s in cur:
                if s not in self._selected_subfolders:
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
                    self._selected_subfolders.pop(i0)
                    self._selected_subfolders.insert(i1, item)
                except Exception:
                    pass
                _drag_state['i'] = i1
                sf_sel_lb.selection_clear(0, tk.END)
                sf_sel_lb.selection_set(i1)
            except Exception:
                pass
        sf_sel_lb.bind("<Button-1>", _sf_on_press)
        sf_sel_lb.bind("<B1-Motion>", _sf_on_drag)

        self._out = {}
        self._action = {'mode': 'current'}  # 'current' | 'all'

        def _collect_opts() -> dict:
            o: dict = {}
            o['ccg_color']         = (ccg_var.get() or '').strip() or None
            o['baseline_color']    = (base_var.get() or '').strip() or None
            o['cs_shade_color']    = (cs_shade_var.get() or '').strip() or None
            o['test_window_color'] = (tw_color_var.get() or '').strip() or None
            o['pval_line_color']   = (pval_line_color_var.get() or '').strip() or None
            o['alpha_line_color']  = (alpha_line_color_var.get() or '').strip() or None
            try:
                v = float(tw_alpha_var.get())
                o['test_window_alpha'] = v if np.isfinite(v) else None
            except Exception:
                o['test_window_alpha'] = None
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
            o['export_segments']   = list(self._selected_segments) if self._selected_segments else ["Current"]
            o['subfolder_by']      = list(self._selected_subfolders)
            o['adaptive_tw_export'] = bool(adaptive_tw_var.get())
            o['print_cs_stg']       = bool(print_stg_var.get())
            o['print_cs_jbsi']      = bool(print_jbsi_var.get())
            o['export_lores']       = bool(lores_var.get())
            o['export_hires']       = bool(hires_var.get())
            o['title_show_shanks']       = bool(title_shanks_var.get())
            o['title_show_inds']         = bool(title_inds_var.get())
            o['title_show_type']         = bool(title_type_var.get())
            o['title_show_seg']          = bool(title_seg_var.get())
            o['title_show_norm_details'] = bool(title_norm_var.get())
            o['title_show_session']      = bool(title_sess_var.get())
            return o

        def _ok():
            self._out.update(_collect_opts())
            self.win.destroy()

        def _cancel():
            self._out.clear()
            self.win.destroy()

        def _save_export_defaults():
            """Persist current export settings as defaults for next time."""
            opts = _collect_opts()
            try:
                ui._settings.setdefault('export_defaults', {})
                ui._settings['export_defaults'] = {
                    'ccg_color':          opts.get('ccg_color') or "",
                    'baseline_color':     opts.get('baseline_color') or "",
                    'cs_shade_color':     opts.get('cs_shade_color') or "",
                    'test_window_color':  opts.get('test_window_color') or "",
                    'test_window_alpha':  opts.get('test_window_alpha') if opts.get('test_window_alpha') is not None else "0.12",
                    'pval_line_color':    opts.get('pval_line_color') or "",
                    'alpha_line_color':   opts.get('alpha_line_color') or "",
                    'min_text_size': opts.get('min_text_size') if opts.get('min_text_size') is not None else "8",
                    'ccg_alpha': opts.get('ccg_alpha') if opts.get('ccg_alpha') is not None else "0.5",
                    'baseline_alpha': opts.get('baseline_alpha') if opts.get('baseline_alpha') is not None else "0.3",
                    'show_legend': bool(opts.get('show_legend', True)),
                    'mirror_xticks': bool(opts.get('mirror_xticks', True)),
                    'xticks_raw': opts.get('_xticks_raw', ''),
                    'export_segments':    opts.get('export_segments', ['Current']),
                    'subfolder_by':       opts.get('subfolder_by', []),
                    'adaptive_tw_export': bool(opts.get('adaptive_tw_export', False)),
                    'print_cs_stg':       bool(opts.get('print_cs_stg',  False)),
                    'print_cs_jbsi':      bool(opts.get('print_cs_jbsi', False)),
                    'export_lores':       bool(opts.get('export_lores', True)),
                    'export_hires':       bool(opts.get('export_hires', False)),
                    'title_show_shanks':       bool(opts.get('title_show_shanks',       True)),
                    'title_show_inds':         bool(opts.get('title_show_inds',         True)),
                    'title_show_type':         bool(opts.get('title_show_type',         True)),
                    'title_show_seg':          bool(opts.get('title_show_seg',          True)),
                    'title_show_norm_details': bool(opts.get('title_show_norm_details', True)),
                    'title_show_session':      bool(opts.get('title_show_session',      False)),
                }
                # Save UI state without touching selections
                ui._group_mgr._save_all_state(selection_name=None, silent=True)
            except Exception:
                pass

        # Preview area (optional)
        prev_holder = ttk.Frame(frm)
        prev_holder.grid(row=row0 + 19, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        prev_holder.columnconfigure(0, weight=1)
        prev_holder.rowconfigure(0, weight=1)
        prev_label = ttk.Label(prev_holder, text="", anchor="center")
        prev_label.grid(row=0, column=0, sticky="nsew")
        self._prev_img = {'obj': None}

        def _render_preview():
            if not show_prev_var.get():
                prev_label.configure(text="")
                prev_label.configure(image="")
                self._prev_img['obj'] = None
                return
            if preview_pair is None:
                prev_label.configure(text="(no pair selected)")
                return
            try:
                # Build a PNG using the same rendering path (but without mutating UI state).
                # Preview uses the current segment/resolution mode and first selected pair.
                ref, tgt = int(preview_pair[0]), int(preview_pair[1])
                seg = ui._seg_idx(ui.current_segment)
                highres = bool(getattr(ui, '_highres_mode', False))
                # Render preview directly (bypass viewer PNG cache) with export overrides.
                tmp_over = {
                    'ccg_color': (ccg_var.get() or '').strip() or None,
                    'baseline_color': (base_var.get() or '').strip() or None,
                    'cs_shade_color': (cs_shade_var.get() or '').strip() or None,
                    'test_window_color': (tw_color_var.get() or '').strip() or None,
                    'test_window_alpha': (float(tw_alpha_var.get()) if str(tw_alpha_var.get()).strip() else None),
                    'pval_line_color':  (pval_line_color_var.get() or '').strip() or None,
                    'alpha_line_color': (alpha_line_color_var.get() or '').strip() or None,
                    'min_text_size': float(minfs_var.get()) if str(minfs_var.get()).strip() else None,
                    'ccg_alpha': float(ccg_a_var.get()) if str(ccg_a_var.get()).strip() else None,
                    'baseline_alpha': float(base_a_var.get()) if str(base_a_var.get()).strip() else None,
                    'show_legend': bool(show_legend_var.get()),
                    'mirror_xticks': bool(mirror_ticks_var.get()),
                    'xticks_ms': ([(float(x.strip())) for x in (xticks_var.get() or '').split(',')
                                   if x.strip()] if (xticks_var.get() or '').strip() else None),
                    'print_cs_stg':  False,
                    'print_cs_jbsi': False,
                    'title_show_shanks':       bool(title_shanks_var.get()),
                    'title_show_inds':         bool(title_inds_var.get()),
                    'title_show_type':         bool(title_type_var.get()),
                    'title_show_seg':          bool(title_seg_var.get()),
                    'title_show_norm_details': bool(title_norm_var.get()),
                    'title_show_session':      bool(title_sess_var.get()),
                }
                old = getattr(ui, '_export_overrides', None)
                ui._export_overrides = tmp_over
                try:
                    import tempfile, os as _os
                    ctx = ui._render_engine.build_context(
                        (ref, tgt), seg, highres, None, None)
                    tmp_png = tempfile.mktemp(suffix='.png')
                    ui._render_engine.write_png(ctx, tmp_png)
                    png_path = tmp_png
                finally:
                    ui._export_overrides = old

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
                    self._prev_img['obj'] = tk_im  # keep reference
                    try:
                        os.remove(png_path)
                    except Exception:
                        pass
                except Exception:
                    prev_label.configure(text=f"Preview ready: {os.path.basename(png_path)}")
            except Exception as ex:
                prev_label.configure(text=f"Preview failed: {ex}")

        ttk.Checkbutton(frm, text="Show preview", variable=show_prev_var,
                        command=_render_preview).grid(row=row0 + 18, column=0, sticky="w", pady=(8, 0))
        # Re-render preview when override fields change (only if enabled)
        for _v in (ccg_var, base_var, cs_shade_var, tw_color_var, tw_alpha_var,
                   pval_line_color_var, alpha_line_color_var,
                   minfs_var, ccg_a_var, base_a_var, xticks_var):
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
        for _bv in (title_shanks_var, title_inds_var, title_type_var,
                    title_seg_var, title_norm_var, title_sess_var):
            try:
                _bv.trace_add('write', lambda *a: _render_preview())
            except Exception:
                pass

        btns = ttk.Frame(frm)
        btns.grid(row=row0 + 20, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="Cancel", command=_cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btns, text="Save export settings", command=_save_export_defaults).pack(
            side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btns, text=f"Export current as {fmt.upper()}", command=_ok).pack(side=tk.RIGHT)

        # If multiple pairs explicitly selected in current list, also offer that
        if selected_pairs and len(selected_pairs) > 1:
            def _export_all():
                self._action['mode'] = 'all'
                _ok()
            ttk.Button(btns, text=f"Export all selected ({len(selected_pairs)})…",
                       command=_export_all).pack(side=tk.RIGHT, padx=(6, 6))

        # Export bookmarked pairs
        n_bm = len(getattr(self, '_bookmarked_pairs', set()) or set())
        if n_bm > 0:
            def _export_bookmarked():
                self._action['mode'] = 'bookmarked'
                _ok()
            ttk.Button(btns, text=f"Export bookmarked ({n_bm})…",
                       command=_export_bookmarked).pack(side=tk.RIGHT, padx=(6, 6))

        # Export pairs from groups — show buttons whenever any groups exist
        any_groups = bool(all_groups)  # all_groups includes special groups (filtered only __-prefix)
        if any_groups:
            def _export_groups():
                self._action['mode'] = 'groups'
                _ok()
            ttk.Button(btns, text="Export selected group(s)…",
                       command=_export_groups).pack(side=tk.RIGHT, padx=(6, 6))

        self.win.protocol("WM_DELETE_WINDOW", _cancel)
        self.win.bind("<Escape>", lambda e: _cancel())
        self.win.update_idletasks()
        self.win.lift()
        self.win.focus_force()


class PairTagsDialog:
    @classmethod
    def show(cls, ui: "CCGReviewUI") -> None:
        if ui.current_pair_idx >= len(ui.all_inds):
            messagebox.showinfo("Pair tags", "No pair selected.")
            return
        cls(ui).win.wait_window()

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui
        inds = tuple(ui.all_inds[ui.current_pair_idx])
        self._ref, self._tgt = int(inds[0]), int(inds[1])
        tag_data = ui._sel_data._pair_tags.get((self._ref, self._tgt), {})
        self.win = tk.Toplevel(ui.root)
        self.win.title(f"Pair Tags — [{self._ref}, {self._tgt}]")
        self.win.geometry("400x350")
        self.win.grab_set()
        self._build(tag_data)

    def _build(self, tag_data):
        win = self.win
        ttk.Label(win, text="Tags (comma-separated):").pack(anchor='w', padx=8, pady=(8, 0))
        self._tags_var = tk.StringVar(value=', '.join(tag_data.get('tags', [])))
        ttk.Entry(win, textvariable=self._tags_var, width=50).pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(win, text="Notes:").pack(anchor='w', padx=8, pady=(8, 0))
        self._notes_text = tk.Text(win, height=12, width=50, font=('Arial', 9), wrap=tk.WORD)
        self._notes_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)
        self._notes_text.insert('1.0', tag_data.get('notes', ''))
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(btn_frame, text="Save", command=self._save).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(side=tk.RIGHT)

    def _save(self):
        ui = self._ui
        ref, tgt = self._ref, self._tgt
        tags = [t.strip() for t in self._tags_var.get().split(',') if t.strip()]
        notes = self._notes_text.get('1.0', 'end-1c')
        if tags or notes:
            existing_groups = ui._sel_data._pair_tags.get((ref, tgt), {}).get('groups', [])
            entry = {'tags': tags, 'notes': notes}
            if existing_groups:
                entry['groups'] = existing_groups
            ui._sel_data._pair_tags[(ref, tgt)] = entry
        elif (ref, tgt) in ui._sel_data._pair_tags:
            del ui._sel_data._pair_tags[(ref, tgt)]
        ui.refresh_lists()
        ui._left_panel._select_pair_in_list((ref, tgt))
        self.win.destroy()


class SuggestedCCGDialog:
    @classmethod
    def show(cls, ui: "CCGReviewUI", specs: list, on_run) -> None:
        if not specs:
            messagebox.showinfo(
                "Suggested custom CCGs",
                "No suggested custom CCG entries found. Use 'Refresh suggested custom CCGs' first.")
            return
        win = tk.Toplevel(ui.root)
        win.title("Suggested custom CCGs")
        win.geometry("620x360")
        win.grab_set()
        ttk.Label(win, text="Generate custom CCGs from availability list:").pack(
            anchor='w', padx=8, pady=(8, 4))
        lb = tk.Listbox(win, selectmode=tk.MULTIPLE, height=12)
        lb.pack(fill=tk.BOTH, expand=True, padx=8)

        def _fmt_t(v):
            if isinstance(v, str) and v.lower() in ('start', 'end'):
                return v
            try:
                return ui.time_slider._sec_to_hms(float(v))
            except Exception:
                return str(v)

        for i, spec in enumerate(specs):
            name = str(spec.get('name', 'custom'))
            t0 = _fmt_t(spec.get('t0', 0.0))
            t1 = _fmt_t(spec.get('t1', 0.0))
            scope = str(spec.get('scope', 'By session'))
            n_have = len(spec.get('sessions', []) or [])
            n_total = max(1, len(ui._sess_mgr._real_nd_keys_ordered()))
            label = f"[{name} | {t0}-{t1}] for {'ALL' if scope == 'All' else scope} ({n_have}/{n_total})"
            lb.insert(tk.END, label)
            lb.select_set(i)

        def _run(selected_idxs):
            on_run([specs[int(i)] for i in selected_idxs])
            win.destroy()

        btns = ttk.Frame(win)
        btns.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(btns, text="Generate selected",
                   command=lambda: _run(lb.curselection())).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Generate all",
                   command=lambda: _run(range(len(specs)))).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.RIGHT)



# ---------------------------------------------------------------------------
# Selection dialogs (moved from SelectionPersistenceManager)
# ---------------------------------------------------------------------------

import os as _os
import shutil as _shutil
import datetime as _datetime
import json as _json
from tkinter import filedialog as _filedialog


class QuickSaveDialog:
    """Ctrl+S save dialog: name entry + Save / Save as Latest buttons."""

    @classmethod
    def show(cls, ui: 'CCGReviewUI') -> None:
        default_name = _datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        win = tk.Toplevel(ui.root)
        win.title('Save selection')
        win.geometry('360x130')
        win.transient(ui.root)
        win.grab_set()

        ttk.Label(win, text='Version name:').pack(pady=(10, 2))
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
            ui._group_mgr._do_save(name)

        def _save_latest():
            win.destroy()
            ui._group_mgr._do_save('latest')

        ttk.Button(btn_frame, text='Save',          command=_save_named).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text='Save as Latest',command=_save_latest).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text='Cancel',        command=win.destroy).pack(side=tk.LEFT, padx=6)
        entry.bind('<Return>', lambda e: _save_named())


class LoadSelectionDialog:
    """List saved selection versions; user picks one to load."""

    @classmethod
    def show(cls, ui: 'CCGReviewUI') -> None:
        versions = ui._group_mgr._list_selection_versions()
        if not versions:
            messagebox.showinfo('Load selection',
                                'No saved selections found for this key.')
            return
        win = tk.Toplevel(ui.root)
        win.title('Load Selection')
        win.geometry('620x340')
        win.grab_set()

        ttk.Label(win, text='Select a version to load:',
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
            lb.insert(tk.END, f'{pfx}{name:30s}  {saved_at[:19]}')
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
                if not messagebox.askyesno('Corrupted file',
                                           f"'{name}' appears to be corrupted.\n"
                                           'Delete it and continue?'):
                    return
                try:
                    _os.remove(path)
                except OSError as ex:
                    messagebox.showerror('Delete failed', str(ex))
                win.destroy()
                cls.show(ui)
                return
            try:
                ui._group_mgr._load_selection_from_file(path)
                win.destroy()
            except Exception as ex:
                messagebox.showerror('Load selection', f'Failed to load:\n{ex}')

        def do_delete(event=None):
            sel = lb.curselection()
            if not sel:
                idx = lb.nearest(event.y) if event else None
                if idx is not None:
                    lb.selection_clear(0, tk.END)
                    lb.selection_set(idx)
                    sel = (idx,)
                else:
                    return
            name, path, saved_at, is_valid, is_history = versions[sel[0]]
            if not messagebox.askyesno('Delete selection',
                                       f"Move '{name}' to deleted folder?",
                                       parent=win):
                return
            deleted_dir = _os.path.join(ui._sel_save_dir, 'deleted')
            _os.makedirs(deleted_dir, exist_ok=True)
            try:
                _shutil.move(path, _os.path.join(deleted_dir, _os.path.basename(path)))
            except OSError as ex:
                messagebox.showerror('Delete failed', str(ex), parent=win)
                return
            win.destroy()
            cls.show(ui)

        def _ctx_menu(event):
            menu = tk.Menu(win, tearoff=0)
            menu.add_command(label='Delete', command=lambda: do_delete(event))
            menu.tk_popup(event.x_root, event.y_root)

        lb.bind('<Button-2>', _ctx_menu)
        lb.bind('<Button-3>', _ctx_menu)
        lb.bind('<Double-Button-1>', lambda e: do_load())
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=6)
        ttk.Button(btn_frame, text='Load',   command=do_load).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text='Cancel', command=win.destroy).pack(side=tk.LEFT, padx=6)
        ttk.Label(btn_frame, text='gray = backup/autosave  ⚠ = corrupted',
                  font=('Arial', 8), foreground='#888888').pack(side=tk.RIGHT, padx=6)


class MissingPairsDialog:
    """Dialog when loaded selection contains pairs absent from current available set.

    Returns 'partial', 'admit_all', or 'cancel'.
    """

    @classmethod
    def show(cls, ui: 'CCGReviewUI', missing: set) -> str:
        win = tk.Toplevel(ui.root)
        win.title('Missing Pairs')
        win.geometry('450x320')
        win.grab_set()
        result = {'action': 'cancel'}
        n = len(missing)
        ttk.Label(win,
                  text=f'{n} selected pair(s) are no longer in available pairs:',
                  font=('Arial', 10, 'bold')).pack(pady=(8, 4))
        ttk.Label(win,
                  text='These pairs may have lost significance after CCG/epoch changes.',
                  font=('Arial', 9), foreground='#666').pack(pady=(0, 4))
        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        lb = tk.Listbox(frame, font=('Courier', 9))
        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=lb.yview)
        lb.config(yscrollcommand=sb.set)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        for ref, tgt in sorted(missing):
            lb.insert(tk.END, f'  ({ref:3d}, {tgt:3d})')
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=8)

        def _set(action):
            result['action'] = action
            win.destroy()

        ttk.Button(btn_frame, text='Keep only available', command=lambda: _set('partial')).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text='Admit all missing',   command=lambda: _set('admit_all')).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text='Cancel',              command=lambda: _set('cancel')).pack(side=tk.RIGHT, padx=4)
        win.protocol('WM_DELETE_WINDOW', lambda: _set('cancel'))
        win.update_idletasks()
        win.lift()
        win.focus_force()
        win.wait_window()
        return result['action']


# ---------------------------------------------------------------------------
# Group dialogs (moved from GroupManager)
# ---------------------------------------------------------------------------

class CreateGroupDialog:
    @classmethod
    def show(cls, ui: 'CCGReviewUI') -> None:
        win = tk.Toplevel(ui.root)
        win.title('Create group')
        win.resizable(False, False)
        win.grab_set()
        ttk.Label(win, text='Group name:').grid(row=0, column=0, padx=8, pady=(10, 4), sticky='w')
        name_var = tk.StringVar()
        entry = ttk.Entry(win, textvariable=name_var, width=26)
        entry.grid(row=0, column=1, padx=(0, 8), pady=(10, 4))
        entry.focus_set()
        special_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(win, text='Create as special group',
                        variable=special_var).grid(row=1, column=0, columnspan=2, padx=8, pady=(0, 8), sticky='w')
        btn_frame = ttk.Frame(win)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(0, 8))

        def _ok():
            name = name_var.get().strip()
            if not name:
                return
            full = (_SPECIAL_PREFIX + name) if special_var.get() else name
            if full in ui._sel_data.groups:
                kind = 'special group' if special_var.get() else 'group'
                messagebox.showinfo('Create group',
                                    f"{kind.capitalize()} '{name}' already exists.",
                                    parent=win)
                return
            ui._sel_data.get_group_metadata(full)
            ui._group_mgr._rebuild_groups_menu()
            ui.refresh_lists()
            win.destroy()

        ttk.Button(btn_frame, text='OK',     command=_ok,          width=8).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text='Cancel', command=win.destroy,  width=8).pack(side=tk.LEFT, padx=4)
        entry.bind('<Return>', lambda e: _ok())
        win.bind('<Escape>', lambda e: win.destroy())


class ExportGroupsDialog:
    @classmethod
    def show(cls, ui: 'CCGReviewUI') -> None:
        if not ui._sel_data.groups:
            messagebox.showinfo('Export groups', 'No groups to export.')
            return
        path = _filedialog.asksaveasfilename(
            title='Export groups',
            defaultextension='.json',
            filetypes=[('JSON files', '*.json')],
            initialfile='groups_export.json',
            initialdir=ui._sel_save_dir,
        )
        if not path:
            return
        ui._sel_data.save(path.removesuffix('.json'))
        print(f'[groups] exported → {path}')


class ImportGroupsDialog:
    @classmethod
    def show(cls, ui: 'CCGReviewUI') -> None:
        path = _filedialog.askopenfilename(
            title='Import groups',
            filetypes=[('JSON files', '*.json')],
            initialdir=ui._sel_save_dir,
        )
        if not path:
            return
        try:
            ui._sel_data.load(path.removesuffix('.json'))
        except (OSError, Exception) as exc:
            messagebox.showerror('Import groups', f'Failed to read file:\n{exc}')
            return
        ui._group_mgr._rebuild_groups_menu()
        ui.refresh_lists()
        print(f'[groups] imported from {path}')
