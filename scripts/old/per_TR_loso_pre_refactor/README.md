# Archived — superseded by `scripts/per_TR_loso.py` + `mc/analyse/loso.py`

These four scripts were folded into one runner and one library on 2026-08-28.
Nothing here is needed to reproduce anything; they are kept for provenance.

| archived script | replaced by |
|---|---|
| `svc_loso_test.py` | `mc.analyse.loso` (statistics) + `per_TR_loso.py --mode run` |
| `svc_loso_batch.py` | `per_TR_loso.py --mode run` (it *was* the batch runner) |
| `plot_per_TR_timecourses.py` | `mc.analyse.loso.plot_per_TR_timecourses` + `--mode plot` |
| `hemisphere_contrast.py` | nothing — retired at the user's request |

## `svc_loso_test.py` — provenance of the reported number

This is the script that produced the reported instruction-phase result,
`BA32-9-10_RewDSR_svc_summary.json`: peak t = 5.079 at TR4, MNI -6/32/18,
p_FWE = .0407, n_vox = 4181, 10 000 permutations, seed 0.

Its statistics moved into `mc/analyse/loso.py` unchanged. Verified by running
both implementations on the same input:

    tstat       bit-identical: True
    null_max_t  bit-identical (pblock=250 and 1000): True
    run_loso    bit-identical to the inline LOSO of svc_loso_test.main(): True

**One deliberate behavioural difference.** `svc_loso_test.load_ref` took the
brain mask from the TR0 folder alone; `mc.analyse.loso.load_ref` intersects
`mask_all_32_subjects` across every included TR, because a voxel entering a
max-t search over voxels x TRs has to be valid at every TR in that search. The
per-TR group masks differ by ~21 voxels, so the mask voxel count shifts
slightly (4181 -> 4182 for the mPFC mask) and the reported p will not
reproduce bit-for-bit. Everything else is the same code.

## `hemisphere_contrast.py`

Tested L - R on the per-subject LOSO held-out arrays with the same sign-flip
max-t null. Its one result is in `CHANGELOG.md` (2026-08-27 evening, later):
left HC/EC > right for `next_rew` at TR7, t = +2.83, p_FWE = .024.
