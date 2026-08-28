# Re-run comparison: old vs new ROI table (2026-08-28)

Runs compared:

| analysis | old | new |
|---|---|---|
| cell RSA | `DSR_RSA_simple_ROI/2026-07-30_15-58-51-fixed_cells-fixed_perms` | `.../2026-08-27_19-18-20` |
| per-lag encoding | `per_lag_encoding/2026-08-04_07-25-15_..._relabelled-final` | `.../2026-08-28_10-18-21_..._relabelled` |
| spatial peaks | `spatial_peaks_simple/2026-08-18_08-41-43_...` | **did not complete** |

## ⚠ 1. spatial_peaks_simple did not finish

The new run directory contains only `cells_used.csv` (924 rows) and
`settings.json` — 2 files, against 31 in a completed run. It wrote the cell list
and stopped. **Re-run it.**

## 2. Cell RSA — core result survives, one number to correct

The two runs do not use the same combo *names*, so compare like for like. The
model lists confirm the match:

    OLD fmri_ctrl_dsrFULL      = [bttn_curr, bttn_next, dsr_fmri, l2_norm, location, state]
    NEW ctrl_fMRI-state_dsrFULL = [bttn_curr, bttn_next, dsr_fmri, l2_norm, location, state]   IDENTICAL

That is the combo the paper reports (β = 0.0445 matches).

| ROI | n old→new | β old→new | p_perm old→new | **q_FDR old→new** |
|---|---|---|---|---|
| HC_mid | 143→145 | 0.0830→0.0828 | 0.001→0.001 | **0.005→0.005** |
| mPFC | 65→65 | 0.0445→**0.0445** | 0.011→0.011 | **0.027→0.037** |
| HC_anterior | 162→171 | 0.0283→0.0271 | 0.047→0.053 | 0.078→0.106 |
| mOFC | 85→74 | −0.0104→−0.0126 | 0.70→0.76 | 0.70→0.84 |
| PCC | 51→51 | −0.0024→−0.0024 | 0.56→0.56 | 0.70→0.80 |

**Both significant effects hold.** mPFC's β is bit-identical (its 65 cells did not
change); only its FDR-corrected q moved, because HC_anterior's p slipped from
0.047 to 0.053 and re-ordered the BH ranking.

→ **Correct in the paper:** "mPFC: β = 0.0445, q_FDR = .027" becomes
**q_FDR = .037**. HC_mid ("β = 0.083, q_FDR = .005") is unchanged.

→ The claim that the effect is "absent from medial OFC, posterior cingulate and
anterior hippocampus (all q_FDR > 0.05)" still holds, and is now more comfortable
for HC_anterior (0.078 → 0.106).

### ⚠ The new run also changed the model set

`fdr_combos` went from 2 to 3, and a new combo `ctrl_fMRI_dsrFULL` was added that
**drops the `state` regressor**. In that combo mPFC gives q = 0.055 (n.s.). This
is a *model* change, not a consequence of the ROI rebuild — but the output now
contains two mPFC numbers, so make sure the paper quotes the state-controlled one.

## 3. Per-lag encoding — stable, slightly stronger

Cell-level tests (`results.md`), no-control Fisher-z:

| | old | new |
|---|---|---|
| mPFC n | 155 | **158** |
| mPFC 30° | r 0.051, p 0.006, FDR 0.036 | r 0.055, p 0.003, **FDR 0.011** |
| mPFC 60° | r 0.027, p 0.171 | r 0.032, p 0.099 |
| mPFC predicted-lag vs 0 | t 2.56, p 0.011 | t 2.87, **p 0.005** |
| mPFC specificity | t 2.84, p 0.005 | t 2.87, p 0.005 |
| HC_mid n | 232 | **233** |
| HC_mid 0° | r 0.041, p 0.009 | r 0.041, p 0.010 |
| HC_mid 330° | r 0.029, p 0.060 | r 0.028, p 0.065 |
| HC_anterior n | 275 | **295** |

Per-ROI n's match the rebuilt ROI table, confirming the relabel applied.
Nothing here weakens; mPFC strengthens modestly.

## ⚠ 4. The paper's Figure 3c numbers cannot be compared yet

The paper quotes **session-level** statistics — "peaked ... at 60° (mean r = 0.067,
t(31) = 2.245, p = 0.016)", "mid HC at 0°: r = 0.055, t(34) = 2.27". Those come
from `overlay_double_dissociation.py`, not from `per_lag_encoding.py`: the old
values live in
`per_lag_encoding/<old_run>/overlay_double_dissociation_noctrl/overlay_per_lag_table.csv`
(`weighting = session`).

**`overlay_double_dissociation.py` has not been re-run**, so every session-level
number in Fig 3c is still the old one. Run it after spatial_peaks completes,
pointing `DEFAULT_PER_LAG_CSV` and `DEFAULT_PER_CELL_CSV` at the new runs
(`grep -rn "RERUN-CHECK" scripts/`).

Note the cell-level peak sits at 30° in both runs while the paper reports a
session-level peak at 60°; the supplement already documents that the ordering
reverses between units of analysis, so this is expected, not new.

## 5. Cell counts to update throughout the paper

| ROI | paper | new |
|---|---:|---:|
| EC | 38 | **35** |
| HC_anterior | 275 | **295** |
| HC_mid | 232 | **233** |
| mOFC | 163 | **142** |
| mPFC | 155 | **158** |
| PCC | 61 | 61 |

RSA subset (Fig 2d caption): mPFC 65 (unchanged), ant HC 162→**171**,
mid HC 143→**145**, PCC 51 (unchanged), mOFC 85→**74**. Per-ROI *session* counts
need re-deriving from the new table as well.

## Still to run

1. `spatial_peaks_simple.py` — did not complete
2. `encoding_state_sustained_cv.py` — the EC sustained-fraction result (`12/38`)
   depends on EC dropping to 35
3. `harmonic_maps_brain_overlay.py` → `cell_gradient_master_table.py` →
   `cell_fMRI_angle_match.py` — the mPFC gradient uses coordinates directly
4. `overlay_double_dissociation.py` — the paper's session-level Fig 3c numbers

---

# Update — spatial_peaks and overlay_double_dissociation now complete

`spatial_peaks_simple/2026-08-28_10-23-56_phase_resid_paired_fixedlag` (31 files)
and `.../2026-08-28_10-18-21_.../overlay_double_dissociation_noctrl`. The overlay
picked up the new per-lag run, so this is a clean comparison — the OLD values
reproduce the paper's numbers exactly, which confirms the mapping.

## Main text, Figure 3c (session-level, overlay on per-lag)

| statistic | paper / old | new | verdict |
|---|---|---|---|
| mPFC 60° | r = 0.067, t(31) = 2.245, **p = 0.016** | r = 0.067, t(32) = 2.156, **p = 0.019** | holds |
| mid HC 0° | r = 0.055, t(34) = 2.27, **p = 0.0147** | r = 0.060, t(35) = 2.320, **p = 0.0131** | holds, stronger |
| mid HC 330° | r = 0.038, t(34) = 2.072, **p = 0.0229** | r = 0.033, t(35) = 1.745, **p = 0.0449** | **holds, but only just** |

Session counts rise (mPFC 32→33, mid HC 35→36) because ROI membership changed.

**The just-past hippocampal effect is the fragile one:** p moves 0.023 → 0.045.
Still one-sided significant, but it would not survive any further correction.

## Overlay window tests (per-lag based)

| ROI / test | old q | new q | |
|---|---|---|---|
| mPFC, window vs zero | 0.010 | 0.016 | holds |
| mPFC, specificity | 0.005 | 0.005 | holds |
| mid HC, window vs zero | 0.008 | 0.007 | holds |
| mid HC, specificity | 0.006 | 0.005 | holds |
| ant HC, window vs zero | 0.050 | **0.028** | strengthens |
| ant HC, specificity | 0.071 (n.s.) | **0.038** | **now significant** |

Anterior hippocampus gaining a significant specificity test is a *new* result, not
a correction. It sharpens the double dissociation rather than weakening it, but the
manuscript currently describes anterior HC as pointing "the same way but more
weakly", which now understates it.

## Supplement, paired-grid-group estimator (spatial_peaks)

| statistic | supplement | old run (08-18) | new | verdict |
|---|---|---|---|---|
| mPFC window vs zero | t(31) = 2.65, q = 0.019 | t(31) = 2.65, q = 0.019 | t(32) = 2.68, **q = 0.017** | holds |
| **mPFC specificity** | t(31) = 2.32, **q = 0.040** | t(31) = 2.32, q = 0.040 | t(32) = 2.19, **q = 0.054** | **no longer significant** |
| mid HC window | t(34) = 1.76, q = 0.044 | t(34) = 2.04, q = 0.037 | t(35) = 2.04, q = 0.036 | holds |
| mid HC specificity | t(34) = 1.51, q = 0.105 | t(34) = 1.77, q = 0.064 | t(35) = 1.83, q = 0.057 | n.s. both |
| ant HC window | t(47) = 1.85, q = 0.044 | t(48) = 1.76, q = 0.043 | t(51) = 1.80, q = 0.039 | holds |
| ant HC specificity | t(47) = 0.94, n.s. | t(48) = 0.90, n.s. | t(51) = 1.06, n.s. | n.s. both |

**Two separate problems here.**

1. **The supplement's mid-HC and ant-HC numbers were already stale** before the ROI
   rebuild. The supplement reports mid HC t(34) = 1.76, but the August-18 run —
   which predates any of this work — gives t(34) = 2.04. Those values came from an
   older run and need updating regardless of the ROI change.

2. **mPFC specificity crosses 0.05 under this estimator** (q 0.040 → 0.054). The
   supplement currently states the window "exceeded each session's own average over
   the remaining ten lags (t(31) = 2.32, q = 0.040)". That sentence no longer holds.

   Context that matters: the *same* test in the main per-lag estimator is
   comfortably significant (q = 0.005), and the supplement's stated purpose is to
   show the effect is robust "to the cross-validation scheme". One of the two
   robustness tests now sits just the wrong side of 0.05 while the primary one is
   an order of magnitude clear. The honest framing is that the window effect
   replicates under both estimators, and the specificity contrast replicates under
   the primary estimator but falls marginally short under the alternative — not
   that robustness fails.

## What still needs running

- `encoding_state_sustained_cv.py` — EC is now 35 cells, so "12/38, 31.6 %", the
  EC-versus-rest Fisher test and the binomial tests will all move.
- `harmonic_maps_brain_overlay.py` → `cell_gradient_master_table.py` →
  `cell_fMRI_angle_match.py` — the gradient analysis uses coordinates directly, and
  mPFC coordinates moved for two Utah subjects.
