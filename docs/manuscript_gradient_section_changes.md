# Gradient section — every value that needs changing

Section: *"During execution, the planned future is laid out along a
ventral-to-dorsal medial prefrontal gradient"* (manuscript pp. 8–10), its Figure 3
legend, and the Methods *"Gradient analysis"* subsection.

Sources: `cell_gradient_master/2026-08-28_15-19-35/final_splits/` —
`gradient_results_summary.csv`, `final_splits_summary.csv`, `fmri_angle_by_group.csv`;
and `per_lag_encoding/2026-08-28_10-18-21_.../overlay_double_dissociation_noctrl/`.

## Results text

| # | manuscript | change to | note |
|---|---|---|---|
| 1 | "A linear trend in the subject-wise centre of cluster mass along the MNI z-axis confirmed this gradient (**t(32) = 2.78, p < 0.01**)" | **unverified** | pure fMRI, no cells. `harmonic_angle_maps.py` has not been re-run, so this is *unchanged but unchecked*, not *confirmed*. |
| 2 | "progressing from theta **~=30°** most ventrally to theta **~=360°/0°** most dorsally" | **re-check** | my z-profile of the gradient mask reads ~59–87° at the ventral end, rising to 333° at z ≈ +38 and wrapping to 22–46° above z = +40. Direction and wrap are right; the ventral figure looks too low. This is a different computation from mine (surface projection, per-sub-model t ≥ 1.5), so re-derive rather than take my number. |
| 3 | "in the range theta = **30° to theta = 75°** (**74/155** mPFC units overlap the gradient...)" | **keep ~30–75°** (measured 36–84°); **87/158** | only the count changes. See the correction note at the end — my earlier "70–97°" was wrong. |
| 4 | "peaked for spatial maps aligned to 60° in the future (mean r at 60° = **0.067**, **t(31) = 2.245**, **p = 0.016**)" | mean r = **0.067**, **t(32) = 2.156**, **p = 0.019** | mean r unchanged to 2 dp (0.0667); df rises because mPFC now spans 33 sessions |
| 5 | "(n = **16** recording sites), this resulted in **42** cells at the ventral-most tip... and **32** cells slightly dorsal" | **19** sites; **48** and **39** cells | |
| 6 | "the more ventral cluster peaked at **30°**, and the more dorsal cluster at **60°**, consistent with the fMRI progression from **30-75°** within this anatomical region" | peaks **unchanged**; add a statistic; replace the fMRI clause | see below |
| 7 | "mid HC at 0°: mean r = **0.055**, **t(34) = 2.27**, **p = 0.0147**" | r = **0.060**, **t(35) = 2.320**, **p = 0.0131** | slightly stronger |
| 8 | "mid HC at 330°: mean r = **0.038**, **t(34) = 2.072**, **p = 0.0229**" | r = **0.033**, **t(35) = 1.745**, **p = 0.0449** | **weakens** — still < 0.05 but marginal |

### On #6, the sentence that needs the most work

The 30°/60° peaks are unchanged, but the sentence makes two claims that the data
no longer support as written.

**The ventral-vs-dorsal difference is not significant.** Site-level permutation
(19 sites): pooled argmax difference p = **0.37**; pooled circular-mean difference
p = **0.47**. The continuous alternatives are also null (circular-linear r = 0.069,
p = 0.87; circular-circular vs fMRI r = −0.079, p = 0.74). The null argmax shift
has median 0°, IQR 0–30°, so a one-bin shift is ordinary chance.

**What each cluster *does* support** (one-sided vs zero, at its own peak):

| cluster | cell | site | subject |
|---|---|---|---|
| ventral 30° | r 0.079, t = 2.44, **q = 0.019** | r 0.031, t = 0.81, q = 0.438 | r 0.051, t = 1.20, q = 0.260 |
| dorsal 60° | r 0.081, t = 2.21, **q = 0.033** | r 0.196, t = 2.24, q = 0.067 | r 0.139, t = 2.26, **q = 0.045** |

Neither survives FDR across all 12 lags (ventral q = 0.111, dorsal q = 0.200), so
report the **pre-specified 30+60° window** rather than the peak if a single number
is wanted.

**The fMRI clause is fine as written.** The fMRI angle at the cells' own voxels
is ventral **median 36°** (range 12–120°) and dorsal **median 84°** (range
6–100°); with the `eighths` map, 37° and 75°. That is the "30–75°" the manuscript
already states, and the ventral→dorsal ordering is in the predicted direction.
It remains descriptive — overlapping distributions, no test attached.

Suggested replacement:

> When we performed the spatial-tuning analysis on these two groups of cells, the
> more ventral cluster peaked at 30° and the more dorsal cluster at 60° (Fig 3d).
> The fMRI-derived preferred angle sampled at these cells' own coordinates was
> 36° (ventral) and 84° (dorsal), consistent with the ventral-to-dorsal
> progression of the imaging gradient. With only 19 independent recording sites this
> ventral-to-dorsal difference is not statistically reliable (site-level
> permutation, p = 0.37), and we report it as consistent with, rather than
> independent evidence for, the fMRI gradient.

## Figure 3 legend

| manuscript | change to |
|---|---|
| "mPFC: dark green, n = **155** cells, n = **32** session" | **158** cells, **33** sessions |
| "HC mid: gold, n = **232** cells, n = **35** sessions" | **233** cells, **36** sessions |
| "single-units sampled in mPFC, overlapping the gradient (n = **74** cells), and outside of the fMRI effect (n = **81**)" | **87** and **71** |

## Methods — "Gradient analysis"

| manuscript | change to |
|---|---|
| "PC1 ≈ [−0.04, −0.41, 0.91]; r = 0.98 with MNI z" | **unchanged** (derived from the fMRI mask, not the cells) |
| "Of **155** mPFC neurons, **74** fell inside the gradient mask" | **158**, **87** |
| "**42** neurons at the ventral end (axis ≤ median, **−13.81 mm**) and **32** slightly dorsal" | **48**, **−13.51 mm**, **39** |

## Why the counts went up

The gradient analysis had been reading coordinates from the per-lag per-cell CSV,
which is frozen at that pipeline's base run — `relabel_per_cell` refreshes the ROI
column and nothing else. 140 of 158 mPFC cells carried stale coordinates (median
3.3 mm, max 43.1 mm; one session still held the retired placeholder). Fixed in
`cell_gradient_master_table.py` via `refresh_coordinates()`. The stale coordinates
were *suppressing* the overlap: 74 → 87 cells inside the mask.

## Still outstanding for this section

`harmonic_angle_maps.py` and `harmonic_maps_brain_overlay.py` have not been
re-run, so items 1 and 2 are unchecked. Item 2 in particular is worth re-deriving,
since the ventral end of the 3b range does not match what I measure.


---

## Correction (2026-08-28, later)

I twice reported the wrong fMRI angle for the cell groups and, on that basis,
wrongly advised that the "30–75°" clause was unsupported. **It is supported.**

| read-out | ventral | dorsal | |
|---|---:|---:|---|
| **circular median at the cells' voxels** | **36°** | **84°** | correct; quote this |
| same, `eighths` map | 37° | 75° | |
| vector mean at the cells' voxels | 58° | 61° | collapses a broad circular sample — misleading |
| whole-mask z-profile at the group's mean z | 70° | 82° | averages y 16–70, far outside the electrodes |

Two mistakes compounded. First I summarised a broadly distributed circular
variable (0–120°) with its **vector mean**, which pulls towards the middle and
hid a 36°→84° separation behind "58 vs 61". Then, correcting that, I switched to
a **whole-mask z-profile**, which averages over the entire y and x extent of the
gradient mask rather than the neighbourhood the electrodes actually occupy.

Both read-outs are now stored — `fmri_at_cells_*` (quote these) and
`fmri_zprofile_*` (context only) — in `fmri_z_readout.csv`, with the docstring
stating which is which.
