# Archived — instruction-phase group-statistics alternatives

Superseded by `scripts/per_TR_loso.py` (statistics in `mc/analyse/loso.py`),
which produces the reported instruction-phase result. That test previously
lived in `scripts/svc_loso_test.py`, now archived under
`scripts/old/per_TR_loso_pre_refactor/`.

## `group_stat_bootstrap_per_TR.py`

Whole-brain **4-D cluster-mass** test with sign-flip permutations over the
same per-TR beta maps (`cropped_masked_smooth_fwhm5_{model}_beta_std.nii`),
plus an SVC cluster-mass branch and a ROI-mean 1-D timecourse test.

Not used for the reported result, for the reason stated in
`svc_loso_test.py`: cluster mass integrates height against extent, so a
strong but spatially focal effect — which is what the a-priori mPFC
hypothesis predicts — scores worse under it than a weak diffuse one. The
reported test corrects by max-t over voxels x TRs inside the a-priori mask
and reads the peak directly.

Kept because it still contains useful machinery that has no equivalent in
the reported pipeline: peak-of-cluster atlas labelling against
Harvard-Oxford / Jülich / Brainnetome with the canonical CLAUDE.md ROI
palette, the reward-cue schedule strip used on the publication timecourse
panels, and the cluster-trace / footprint-beta plotting helpers.

It reads the same inputs as `svc_loso_test.py` and imports nothing from it,
so it can be moved back up one level and run unchanged.
