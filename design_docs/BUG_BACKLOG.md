# Bug backlog (recovered 2026-08-02, none started)

1. Stack segments not working
2. Extend-window gappy render
3. Same-scale pair / same-scale session normalization toggles nonresponsive
4. Autoscale of ACG nonresponsive
5. Ref/tgt scale input should allow values beyond slider limits (render at far
   left/right of slider); Settings > Display needs config for max slider distance
6. JBSI silent NaN — all resolutions, all pairs
7. Tailed and global baseline nonresponsive
8. Jitter launches another GUI process (unintended?)
9. Jitter doesn't show even though spike-correlation run succeeded
10. Jitter button should stay disabled unless jitter data exists
11. Extend compute rejects legal min bin size (3.33e-05s = 1/30000) — "Bin size
    too small" error incorrectly fires at exactly 1/sampling_rate; should use
    same boundary check (`<=` not `<`, or equivalent) as regular CCG compute

Plus: segment-unify plan (see `design_docs/` or `~/.claude/plans/` for the full
custom-CCG-unification design doc) — untouched.
