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

    def _setup_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # ── Session selection ──
        sess_frame = ttk.LabelFrame(main, text="Sessions", padding=6)
        sess_frame.pack(fill=tk.X, pady=(0, 6))

        self._sess_vars = []
        cols = 3
        for i, sess in enumerate(self._sessions):
            name = self._session_name(sess)
            var = tk.BooleanVar(value=True)
            self._sess_vars.append((var, sess))
            ttk.Checkbutton(sess_frame, text=name, variable=var).grid(
                row=i // cols, column=i % cols, sticky='w', padx=4)

        # ── NeuronsDatasetConfig ──
        nd_frame = ttk.LabelFrame(main, text="NeuronsDataset Config", padding=6)
        nd_frame.pack(fill=tk.X, pady=(0, 6))

        r = 0
        ttk.Label(nd_frame, text="Name:").grid(row=r, column=0, sticky='w')
        self._nd_name_var = tk.StringVar(value="default")
        ttk.Entry(nd_frame, textvariable=self._nd_name_var, width=20).grid(
            row=r, column=1, sticky='w', padx=4)

        r += 1
        ttk.Label(nd_frame, text="Neuron types:").grid(row=r, column=0, sticky='w')
        self._nd_types_var = tk.StringVar(value="pyr, inter")
        ttk.Entry(nd_frame, textvariable=self._nd_types_var, width=30).grid(
            row=r, column=1, columnspan=2, sticky='w', padx=4)

        r += 1
        ttk.Label(nd_frame, text="Epochs:").grid(row=r, column=0, sticky='w')
        self._nd_epochs_var = tk.StringVar(value="pre, maze, post, re-maze")
        ttk.Entry(nd_frame, textvariable=self._nd_epochs_var, width=40).grid(
            row=r, column=1, columnspan=2, sticky='w', padx=4)

        r += 1
        ttk.Label(nd_frame, text="N segments:").grid(row=r, column=0, sticky='w')
        self._nd_nseg_var = tk.StringVar(value="")
        ttk.Entry(nd_frame, textvariable=self._nd_nseg_var, width=20).grid(
            row=r, column=1, sticky='w', padx=4)
        ttk.Label(nd_frame, text="(comma-sep, one per epoch, or blank)").grid(
            row=r, column=2, sticky='w')

        r += 1
        self._nd_zerotimes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(nd_frame, text="Zero spike times",
                        variable=self._nd_zerotimes_var).grid(
            row=r, column=0, columnspan=2, sticky='w')

        # Sleep slicing
        r += 1
        self._nd_sleep_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(nd_frame, text="Sleep filter:",
                        variable=self._nd_sleep_var).grid(
            row=r, column=0, sticky='w')
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
                        variable=self._nd_ripple_var).grid(
            row=r, column=0, sticky='w')
        self._nd_ripple_mindur_var = tk.StringVar(value="0")
        ttk.Label(nd_frame, text="min_dur:").grid(row=r, column=1, sticky='e')
        ttk.Entry(nd_frame, textvariable=self._nd_ripple_mindur_var, width=8).grid(
            row=r, column=2, sticky='w', padx=4)

        # ── CCGConfig ──
        ccg_frame = ttk.LabelFrame(main, text="CCG Config", padding=6)
        ccg_frame.pack(fill=tk.X, pady=(0, 6))

        cr = 0
        ttk.Label(ccg_frame, text="Resolution:").grid(row=cr, column=0, sticky='w')
        self._ccg_res_var = tk.StringVar(value="lowres")
        ttk.Combobox(ccg_frame, textvariable=self._ccg_res_var,
                     values=["lowres", "highres"], width=10,
                     state='readonly').grid(row=cr, column=1, sticky='w', padx=4)

        ttk.Label(ccg_frame, text="Duration (ms):").grid(row=cr, column=2, sticky='w')
        self._ccg_dur_var = tk.StringVar(value="20")
        ttk.Entry(ccg_frame, textvariable=self._ccg_dur_var, width=8).grid(
            row=cr, column=3, sticky='w', padx=4)

        cr += 1
        ttk.Label(ccg_frame, text="Alpha:").grid(row=cr, column=0, sticky='w')
        self._ccg_alpha_var = tk.StringVar(value="0.05")
        ttk.Entry(ccg_frame, textvariable=self._ccg_alpha_var, width=10).grid(
            row=cr, column=1, sticky='w', padx=4)

        ttk.Label(ccg_frame, text="Alpha2:").grid(row=cr, column=2, sticky='w')
        self._ccg_alpha2_var = tk.StringVar(value="0.1")
        ttk.Entry(ccg_frame, textvariable=self._ccg_alpha2_var, width=10).grid(
            row=cr, column=3, sticky='w', padx=4)

        cr += 1
        ttk.Label(ccg_frame, text="Min lag (ms):").grid(row=cr, column=0, sticky='w')
        self._ccg_minlag_var = tk.StringVar(value="1")
        ttk.Entry(ccg_frame, textvariable=self._ccg_minlag_var, width=8).grid(
            row=cr, column=1, sticky='w', padx=4)

        ttk.Label(ccg_frame, text="Max lag (ms):").grid(row=cr, column=2, sticky='w')
        self._ccg_maxlag_var = tk.StringVar(value="3")
        ttk.Entry(ccg_frame, textvariable=self._ccg_maxlag_var, width=8).grid(
            row=cr, column=3, sticky='w', padx=4)

        cr += 1
        ttk.Label(ccg_frame, text="Conv window (ms):").grid(row=cr, column=0, sticky='w')
        self._ccg_conv_var = tk.StringVar(value="5")
        ttk.Entry(ccg_frame, textvariable=self._ccg_conv_var, width=8).grid(
            row=cr, column=1, sticky='w', padx=4)

        ttk.Label(ccg_frame, text="Correction:").grid(row=cr, column=2, sticky='w')
        self._ccg_mc_var = tk.StringVar(value="bonferroni")
        ttk.Combobox(ccg_frame, textvariable=self._ccg_mc_var,
                     values=["bonferroni", "fdr_bh"], width=12,
                     state='readonly').grid(row=cr, column=3, sticky='w', padx=4)

        cr += 1
        self._ccg_symmetrize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ccg_frame, text="Symmetrize CCG",
                        variable=self._ccg_symmetrize_var).grid(
            row=cr, column=0, columnspan=2, sticky='w')

        self._ccg_accel_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ccg_frame, text="GPU acceleration (CuPy)",
                        variable=self._ccg_accel_var).grid(
            row=cr, column=2, columnspan=2, sticky='w')

        # Connection types
        cr += 1
        ttk.Label(ccg_frame, text="E types:").grid(row=cr, column=0, sticky='w')
        self._ccg_etypes_var = tk.StringVar(value="pyr-pyr, pyr-inter")
        ttk.Entry(ccg_frame, textvariable=self._ccg_etypes_var, width=30).grid(
            row=cr, column=1, columnspan=2, sticky='w', padx=4)

        cr += 1
        ttk.Label(ccg_frame, text="I types:").grid(row=cr, column=0, sticky='w')
        self._ccg_itypes_var = tk.StringVar(value="inter-inter, inter-pyr")
        ttk.Entry(ccg_frame, textvariable=self._ccg_itypes_var, width=30).grid(
            row=cr, column=1, columnspan=2, sticky='w', padx=4)

        # ── Action buttons ──
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(6, 0))

        self._build_nd_btn = ttk.Button(
            btn_frame, text="1. Build NeuronsDataset",
            command=self._on_build_nd)
        self._build_nd_btn.pack(side=tk.LEFT, padx=4)

        self._build_ccg_btn = ttk.Button(
            btn_frame, text="2. Generate CCGs",
            command=self._on_build_ccg, state='disabled')
        self._build_ccg_btn.pack(side=tk.LEFT, padx=4)

        self._review_btn = ttk.Button(
            btn_frame, text="3. Open Review UI",
            command=self._on_open_review, state='disabled')
        self._review_btn.pack(side=tk.LEFT, padx=4)

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
