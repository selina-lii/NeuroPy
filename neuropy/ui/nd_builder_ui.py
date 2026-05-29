"""
NeuronsDataset Builder GUI

Interactive Tkinter interface for constructing NeuronsDataset objects and
generating CCGs without writing code.

Usage::

    from neuropy.plotting.nd_builder_ui import NDBuildUI
    NDBuildUI(sessions).run()
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading


class NDBuildUI:
    """GUI for configuring and building NeuronsDataset + CCGDataset.

    Parameters
    ----------
    sessions : list
        List of session objects (e.g. from subjects.ProcessData).
        Each must have .filePrefix, .neurons, .paradigm, etc.
    """

    def __init__(self, sessions):
        self._sessions = sessions if isinstance(sessions, list) else [sessions]

        self._owns_mainloop = False
        try:
            existing = tk._default_root
        except AttributeError:
            existing = None
        if existing is not None and existing.winfo_exists():
            self.root = tk.Toplevel(existing)
        else:
            self.root = tk.Tk()
            self._owns_mainloop = True
        self.root.title("NeuronsDataset Builder")
        self.root.geometry("720x620")

        # Result objects
        self.nd = None
        self.cd = None

        # Build state
        self._build_thread = None
        self._build_error = None

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    # Field specs — used by _setup_ui loops to avoid repetition.
    #
    # _ND_ENTRY_FIELDS: (label, attr, default, width, col_span, hint)
    _ND_ENTRY_FIELDS = [
        ("Name:",         "_nd_name_var",   "default",                    20, 1, None),
        ("Neuron types:", "_nd_types_var",  "pyr, inter",                 30, 2, None),
        ("Epochs:",       "_nd_epochs_var", "pre, maze, post, re-maze",   40, 2, None),
        ("N segments:",   "_nd_nseg_var",   "",                           20, 1,
         "(comma-sep, one per epoch, or blank)"),
    ]

    # _CCG_PAIR_FIELDS: list of rows; each row = two (label, attr, default, kind, width, opts)
    # kind: 'entry' | 'combo'
    _CCG_PAIR_FIELDS = [
        [("Resolution:",       "_ccg_res_var",   "lowres",      "combo", 10, ["lowres", "highres"]),
         ("Duration (ms):",    "_ccg_dur_var",   "20",          "entry",  8, None)],
        [("Alpha:",            "_ccg_alpha_var", "0.05",        "entry", 10, None),
         ("Alpha2:",           "_ccg_alpha2_var","0.1",         "entry", 10, None)],
        [("Min lag (ms):",     "_ccg_minlag_var","1",           "entry",  8, None),
         ("Max lag (ms):",     "_ccg_maxlag_var","3",           "entry",  8, None)],
        [("Conv window (ms):","_ccg_conv_var",   "5",           "entry",  8, None),
         ("Correction:",       "_ccg_mc_var",    "bonferroni",  "combo", 12, ["bonferroni", "fdr_bh"])],
    ]

    # _CCG_CHECK_FIELDS: (label, attr, default, col, col_span)
    _CCG_CHECK_FIELDS = [
        ("Symmetrize CCG",         "_ccg_symmetrize_var", True,  0, 2),
        ("GPU acceleration (CuPy)","_ccg_accel_var",      False, 2, 2),
    ]

    # _CCG_CONN_FIELDS: (label, attr, default)
    _CCG_CONN_FIELDS = [
        ("E types:", "_ccg_etypes_var", "pyr-pyr, pyr-inter"),
        ("I types:", "_ccg_itypes_var", "inter-inter, inter-pyr"),
    ]

    # _BUTTONS: (text, attr, command_name, initial_state)
    _BUTTONS = [
        ("1. Build NeuronsDataset", "_build_nd_btn",  "_on_build_nd",    "normal"),
        ("2. Generate CCGs",        "_build_ccg_btn", "_on_build_ccg",   "disabled"),
        ("3. Open Review UI",       "_review_btn",    "_on_open_review", "disabled"),
    ]

    def _setup_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # ── Session selection ──
        sess_frame = ttk.LabelFrame(main, text="Sessions", padding=6)
        sess_frame.pack(fill=tk.X, pady=(0, 6))

        self._sess_vars = []
        cols = 3
        for i, sess in enumerate(self._sessions):
            var = tk.BooleanVar(value=True)
            self._sess_vars.append((var, sess))
            ttk.Checkbutton(sess_frame, text=self._session_name(sess), variable=var).grid(
                row=i // cols, column=i % cols, sticky='w', padx=4)

        # ── NeuronsDataset Config ──
        nd_frame = ttk.LabelFrame(main, text="NeuronsDataset Config", padding=6)
        nd_frame.pack(fill=tk.X, pady=(0, 6))

        for r, (label, attr, default, width, span, hint) in enumerate(self._ND_ENTRY_FIELDS):
            ttk.Label(nd_frame, text=label).grid(row=r, column=0, sticky='w')
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(nd_frame, textvariable=var, width=width).grid(
                row=r, column=1, columnspan=span, sticky='w', padx=4)
            if hint:
                ttk.Label(nd_frame, text=hint).grid(row=r, column=1 + span, sticky='w')

        r = len(self._ND_ENTRY_FIELDS)
        self._nd_zerotimes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(nd_frame, text="Zero spike times",
                        variable=self._nd_zerotimes_var).grid(
            row=r, column=0, columnspan=2, sticky='w')

        # Sleep slicing
        r += 1
        self._nd_sleep_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(nd_frame, text="Sleep filter:",
                        variable=self._nd_sleep_var).grid(row=r, column=0, sticky='w')
        self._nd_sleep_labels_var = tk.StringVar(value="NREM, REM")
        ttk.Entry(nd_frame, textvariable=self._nd_sleep_labels_var, width=20).grid(
            row=r, column=1, sticky='w', padx=4)
        self._nd_sleep_mindur_var = tk.StringVar(value="120")
        ttk.Label(nd_frame, text="min_dur:").grid(row=r, column=2, sticky='e')
        ttk.Entry(nd_frame, textvariable=self._nd_sleep_mindur_var, width=8).grid(
            row=r, column=3, sticky='w', padx=4)

        # Ripple slicing
        r += 1
        self._nd_ripple_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(nd_frame, text="Ripple discard:",
                        variable=self._nd_ripple_var).grid(row=r, column=0, sticky='w')
        self._nd_ripple_mindur_var = tk.StringVar(value="0")
        ttk.Label(nd_frame, text="min_dur:").grid(row=r, column=1, sticky='e')
        ttk.Entry(nd_frame, textvariable=self._nd_ripple_mindur_var, width=8).grid(
            row=r, column=2, sticky='w', padx=4)

        # ── CCG Config ──
        ccg_frame = ttk.LabelFrame(main, text="CCG Config", padding=6)
        ccg_frame.pack(fill=tk.X, pady=(0, 6))

        for cr, pair in enumerate(self._CCG_PAIR_FIELDS):
            for col_offset, (label, attr, default, kind, width, opts) in enumerate(pair):
                ttk.Label(ccg_frame, text=label).grid(
                    row=cr, column=col_offset * 2, sticky='w')
                var = tk.StringVar(value=default)
                setattr(self, attr, var)
                if kind == 'combo':
                    ttk.Combobox(ccg_frame, textvariable=var, values=opts,
                                 width=width, state='readonly').grid(
                        row=cr, column=col_offset * 2 + 1, sticky='w', padx=4)
                else:
                    ttk.Entry(ccg_frame, textvariable=var, width=width).grid(
                        row=cr, column=col_offset * 2 + 1, sticky='w', padx=4)

        cr = len(self._CCG_PAIR_FIELDS)
        for label, attr, default, col, span in self._CCG_CHECK_FIELDS:
            var = tk.BooleanVar(value=default)
            setattr(self, attr, var)
            ttk.Checkbutton(ccg_frame, text=label, variable=var).grid(
                row=cr, column=col, columnspan=span, sticky='w')

        for i, (label, attr, default) in enumerate(self._CCG_CONN_FIELDS):
            row = cr + 1 + i
            ttk.Label(ccg_frame, text=label).grid(row=row, column=0, sticky='w')
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(ccg_frame, textvariable=var, width=30).grid(
                row=row, column=1, columnspan=2, sticky='w', padx=4)

        # ── Action buttons ──
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(6, 0))

        for text, attr, cmd_name, state in self._BUTTONS:
            btn = ttk.Button(btn_frame, text=text,
                             command=getattr(self, cmd_name), state=state)
            btn.pack(side=tk.LEFT, padx=4)
            setattr(self, attr, btn)

        # ── Status bar ──
        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(main, textvariable=self._status_var,
                  relief='sunken', anchor='w').pack(
            fill=tk.X, side=tk.BOTTOM, pady=(6, 0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _session_name(session):
        parts = session.filePrefix.parts[-1].split('_')[:2]
        return '_'.join(parts)

    def _parse_csv(self, text):
        """Parse comma-separated string into list of stripped strings."""
        items = [s.strip() for s in text.split(',') if s.strip()]
        return items if items else None

    def _parse_conn_types(self, text):
        """Parse 'pyr-pyr, pyr-inter' into [('pyr','pyr'), ('pyr','inter')]."""
        pairs = []
        for item in text.split(','):
            item = item.strip()
            if '-' in item:
                a, b = item.split('-', 1)
                pairs.append((a.strip(), b.strip()))
        return pairs if pairs else []

    def _selected_sessions(self):
        return [s for var, s in self._sess_vars if var.get()]

    # ------------------------------------------------------------------
    # Build NeuronsDataset
    # ------------------------------------------------------------------

    def _on_build_nd(self):
        sessions = self._selected_sessions()
        if not sessions:
            messagebox.showwarning("Build", "No sessions selected.")
            return

        from neuropy.analyses.neurons_dataset import (
            NeuronsDatasetConfig, EpochSlicingConfig, NeuronsDataset)

        epochs = self._parse_csv(self._nd_epochs_var.get())

        # N segments
        nseg_text = self._nd_nseg_var.get().strip()
        n_segments = None
        if nseg_text:
            try:
                n_segments = [int(x.strip()) for x in nseg_text.split(',')]
            except ValueError:
                messagebox.showerror("Build", "N segments must be comma-separated integers.")
                return

        sleep = None
        if self._nd_sleep_var.get():
            sleep = EpochSlicingConfig(
                labels=self._parse_csv(self._nd_sleep_labels_var.get()),
                min_dur=float(self._nd_sleep_mindur_var.get()),
                discard=False,
            )

        ripple = None
        if self._nd_ripple_var.get():
            ripple = EpochSlicingConfig(
                min_dur=float(self._nd_ripple_mindur_var.get()),
                discard=True,
            )

        ndconf = NeuronsDatasetConfig(
            name=self._nd_name_var.get(),
            neuron_types=self._parse_csv(self._nd_types_var.get()),
            epochs=epochs,
            n_segments=n_segments,
            sleep=sleep,
            ripple=ripple,
            zero_spike_times=self._nd_zerotimes_var.get(),
        )

        self._status_var.set("Building NeuronsDataset...")
        self._build_nd_btn.config(state='disabled')
        self._build_error = None

        def _worker():
            try:
                self.nd = NeuronsDataset(sessions, ndconf)
            except Exception as ex:
                self._build_error = str(ex)

        self._build_thread = threading.Thread(target=_worker, daemon=True)
        self._build_thread.start()
        self.root.after(200, self._poll_nd_build)

    def _poll_nd_build(self):
        if self._build_thread is not None and self._build_thread.is_alive():
            self.root.after(200, self._poll_nd_build)
            return
        self._build_thread = None
        self._build_nd_btn.config(state='normal')
        if self._build_error:
            self._status_var.set("Build failed")
            messagebox.showerror("Build", f"NeuronsDataset failed:\n{self._build_error}")
            return
        n_sess = len(self.nd.data)
        total_neurons = sum(v.n_neurons for v in self.nd.data.values())
        self._status_var.set(
            f"NeuronsDataset ready: {n_sess} session(s), {total_neurons} neurons")
        self._build_ccg_btn.config(state='normal')
        self.root.bell()

    # ------------------------------------------------------------------
    # Generate CCGs
    # ------------------------------------------------------------------

    def _on_build_ccg(self):
        if self.nd is None:
            messagebox.showwarning("CCG", "Build NeuronsDataset first.")
            return

        from neuropy.analyses.ms_connectivity import CCGConfig, CCGDataset

        try:
            cconf = CCGConfig(
                name=self._nd_name_var.get(),
                resolution=self._ccg_res_var.get(),
                duration=float(self._ccg_dur_var.get()) * 1e-3,
                alpha=float(self._ccg_alpha_var.get()),
                alpha2=float(self._ccg_alpha2_var.get()),
                min_lag=float(self._ccg_minlag_var.get()) * 1e-3,
                max_lag=float(self._ccg_maxlag_var.get()) * 1e-3,
                conv_window=float(self._ccg_conv_var.get()) * 1e-3,
                multiple_correction=self._ccg_mc_var.get(),
                symmetrize_ccg=self._ccg_symmetrize_var.get(),
                use_acceleration=self._ccg_accel_var.get(),
                conn_types_E=self._parse_conn_types(self._ccg_etypes_var.get()),
                conn_types_I=self._parse_conn_types(self._ccg_itypes_var.get()),
            )
        except Exception as ex:
            messagebox.showerror("CCG Config", str(ex))
            return

        self._status_var.set("Computing CCGs (this may take a while)...")
        self._build_ccg_btn.config(state='disabled')
        self._build_error = None

        def _worker():
            try:
                self.cd = CCGDataset(conf=cconf, nd=self.nd)
                self.cd.get_ccg()
            except Exception as ex:
                self._build_error = str(ex)

        self._build_thread = threading.Thread(target=_worker, daemon=True)
        self._build_thread.start()
        self.root.after(500, self._poll_ccg_build)

    def _poll_ccg_build(self):
        if self._build_thread is not None and self._build_thread.is_alive():
            self.root.after(500, self._poll_ccg_build)
            return
        self._build_thread = None
        self._build_ccg_btn.config(state='normal')
        if self._build_error:
            self._status_var.set("CCG generation failed")
            messagebox.showerror("CCG", f"CCG generation failed:\n{self._build_error}")
            return
        n_pairs = sum(
            p.n_pairs for p in self.cd.data.values() if p is not None)
        self._status_var.set(f"CCGs ready: {n_pairs} significant pairs found")
        self._review_btn.config(state='normal')
        self.root.bell()

    # ------------------------------------------------------------------
    # Open Review UI
    # ------------------------------------------------------------------

    def _on_open_review(self):
        if self.cd is None:
            return
        from neuropy.plotting.ccg_ui import CCGReviewUI
        # Open review for the first available key
        keys = list(self.cd.data.keys())
        if not keys:
            messagebox.showinfo("Review", "No data keys available.")
            return
        key = keys[0]
        CCGReviewUI(self.cd, key)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        if self._owns_mainloop:
            self.root.mainloop()
