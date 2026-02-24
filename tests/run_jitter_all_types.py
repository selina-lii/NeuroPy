"""
run_jitter_all_types.py
=======================
End-to-end script:
  1. Load CCGDataset from cache (RatU Day2NSD)
  2. Alpha-scan pval_corrected to find the threshold that gives ~100 pyr-pyr
     pairs in post0 (for EranConv first-pass selection)
  3. Re-select pairs at that alpha (fast — uses cached CCG, skips
     spike_correlations)
  4. Run jitter (njitter=200) for ALL four connection types
  5. Save intermediates to data/jitter/
  6. Plot verification figures to images/jitter_verification/

Run from repo root:
    python tests/run_jitter_all_types.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── repo path ──────────────────────────────────────────────────────────────
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "notebooks"))

# ── imports ────────────────────────────────────────────────────────────────
import neuropy.analyses.ms_connectivity as msconn
from neuropy.analyses.ms_connectivity import CCGDataset, CCGConfig, NeuronsDataset, NeuronsDatasetConfig
from neuropy.analyses.jitter import JitterConfig, JitterType
import subjects as subj

# ──────────────────────────────────────────────────────────────────────────
# 1.  Build session / NeuronsDataset
# ──────────────────────────────────────────────────────────────────────────
sess = subj.nsd.allsess[5]         # RatU Day2 NSD
print(f"Session: {sess.basepath}")

ndconf = NeuronsDatasetConfig(
    recinfo          = sess.recinfo,
    zero_spike_times = False,
    epochs           = ["post", "pre", "maze", "re-maze"],
)
nd = NeuronsDataset(sess, ndconf)

# ──────────────────────────────────────────────────────────────────────────
# 2.  Load CCGDataset from cache (alpha=1e-10 was used originally)
# ──────────────────────────────────────────────────────────────────────────
cconf = CCGConfig(
    name             = "default",
    resolution       = "lowres",
    alpha            = 1e-10,
    alpha2           = 0.1,
    min_lag          = 1e-3,
    max_lag          = 3e-3,
    duration         = 20e-3,
    mc_method        = "fdr_bh",   # matches the original cache
    use_acceleration = False,      # CuPy not available on this machine
)
cd = CCGDataset(conf=cconf, nd=nd)
print("\nLoaded CCGDataset from cache.")

# ──────────────────────────────────────────────────────────────────────────
# 3.  Quick alpha scan using cached pval_corrected
# ──────────────────────────────────────────────────────────────────────────
print("\n=== Alpha scan for pyr-pyr E pairs in post0 ===")

# CCGDataset with all epochs combined has ONE nd_key
nd_key   = list(cd._ccg.keys())[0]
ccg_data = cd._ccg[nd_key]
neurons  = nd.data[nd_key]
edge_times = nd.edge_times[nd_key]

print(f"  nd_key = {nd_key}")
print(f"  edge_times labels: {list(edge_times['label'])}")

# pval shape: [n_seg, n_ref, n_tgt, n_bins]
# Use raw pval with Bonferroni over bins (cached pval_corrected may be fdr_bh)
lb, ub = cconf.min_lag_bin, cconf.max_lag_bin
p_raw  = ccg_data.pval
n_bins = cconf.nbins
if p_raw is not None:
    p_corr = np.minimum(p_raw * n_bins, 1.0)
else:
    print("WARNING: pval not in cache — using pval_corrected as fallback")
    p_corr = ccg_data.pval_corrected
if p_corr is None:
    raise RuntimeError("Neither pval nor pval_corrected found in cache.")

# Identify pyr neuron indices
pyr_mask  = np.array([t == 'pyr' for t in neurons.neuron_type])
pyr_inds  = np.where(pyr_mask)[0]
print(f"  n_pyr = {len(pyr_inds)}, n_inter = {(~pyr_mask).sum()}")

# Find the post0 segment index in edge_times
labels = list(edge_times['label'])
post_segs = [i for i, lbl in enumerate(labels) if 'post' in str(lbl).lower()]
if post_segs:
    SEG = post_segs[0]   # first post segment
    print(f"  Post0 segment index = {SEG} (label: {labels[SEG]})")
else:
    SEG = 0
    print(f"  WARNING: No 'post' label found; using SEG=0 (label: {labels[0]})")

# For each (ref, tgt), min corrected p-value over test-window bins in post0
p_win = p_corr[SEG, :, :, lb:ub].min(axis=-1)   # [n_ref, n_tgt]
# Restrict to pyr-pyr
p_pp  = p_win[np.ix_(pyr_inds, pyr_inds)]
np.fill_diagonal(p_pp, 1.0)                       # exclude self-pairs

alpha_candidates = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 5e-4,
                    1e-3, 5e-3, 0.01, 0.02, 0.05]
counts = {}
print("  alpha        → pyr-pyr pairs in post0 (estimate, excl. other filters)")
for a in alpha_candidates:
    n = int((p_pp <= a).sum())
    counts[a] = n
    print(f"  {a:<12.2e}    {n}")

# Find the first alpha giving ≥ 80 pairs
target_alpha = None
for a in alpha_candidates:
    if counts[a] >= 80:
        target_alpha = a
        break
if target_alpha is None:
    print("\nNo alpha gives ≥ 80 pyr-pyr pairs — using 0.05")
    target_alpha = 0.05

print(f"\n→ Selected alpha = {target_alpha} "
      f"(estimated {counts[target_alpha]} pyr-pyr pairs in post0)")

# ──────────────────────────────────────────────────────────────────────────
# 4.  Re-select pairs at new alpha
# ──────────────────────────────────────────────────────────────────────────
print(f"\n=== Reselecting pairs at alpha={target_alpha} ===")
cd.reselect_pairs(target_alpha, method='bonferroni')
print("\n" + cd._session_summary(nd_key))

# ──────────────────────────────────────────────────────────────────────────
# 5.  Run jitter for ALL connection types
# ──────────────────────────────────────────────────────────────────────────
jconf = JitterConfig(
    ccg      = cconf,
    njitter  = 200,
    jscale   = 5e-3,
    alpha    = 0.05,
    jitter_type = JitterType.INTERVAL,
)

print(f"\n=== Running jitter (njitter={jconf.njitter}) for all connection types ===")
cd.refine_with_jitter(jconf)

# ──────────────────────────────────────────────────────────────────────────
# 6.  Save intermediates
# ──────────────────────────────────────────────────────────────────────────
JITTER_SAVE_ROOT = os.path.join(REPO, "data", "jitter")
saved_paths = {}
for key, j in cd._jitter.items():
    key_str  = str(key).replace(" ", "_").replace("/", "-")
    save_dir = os.path.join(JITTER_SAVE_ROOT, key_str)
    saved_paths[key] = j.save(save_dir)

print("\n=== Saved jitter intermediates ===")
for key, path in saved_paths.items():
    j = cd._jitter[key]
    n_sig = int(j.j_sig.sum()) if j.j_sig is not None else 0
    print(f"  {key}  → {path}  [{n_sig}/{j.n_pairs} sig]")

# ──────────────────────────────────────────────────────────────────────────
# 7.  Verification plots
# ──────────────────────────────────────────────────────────────────────────
PLOT_DIR = os.path.join(REPO, "images", "jitter_verification")
os.makedirs(PLOT_DIR, exist_ok=True)

for key, j in cd._jitter.items():
    if j.n_pairs == 0:
        continue
    key_str  = str(key).replace(" ", "_").replace("/", "-")
    save_path = os.path.join(PLOT_DIR, f"{key_str}.png")
    fig = j.plot_verification(
        max_pairs        = 24,
        sort_by_pval     = True,
        figsize_per_panel= (3.5, 2.8),
        ncols            = 4,
        save_path        = save_path,
    )
    plt.close(fig)
    print(f"  Saved verification plot → {save_path}")

print("\n=== DONE ===")
print(f"Jitter data:   {JITTER_SAVE_ROOT}/")
print(f"Verify plots:  {PLOT_DIR}/")
