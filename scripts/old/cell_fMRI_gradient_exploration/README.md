# Archived — cell ↔ fMRI gradient exploration

Exploratory analyses asking whether human mPFC single-unit preferred
future lag tracks the 7T fMRI dorsoventral gradient. Superseded by
`scripts/cell_fMRI_angle_match.py`, which produces the reported result.

Kept for provenance: the reported split (`pc1_ventral_dorsal`) was chosen
out of the sweep these scripts generated, so they document the search
space behind that choice.

## What is here

| script | what it explored |
|---|---|
| `cell_gradient_full_factorial.py` | 320-row factorial: control × cell set × axis × gating × split × weighting → `full_factorial.csv` |
| `cell_averaging_best_split.py` | ranked 33 axis × n_groups × weighting combinations by gradient-direction match → `best_split_ranking.csv` |
| `cell_averaging_schemes_table.py` | per-cell table of every averaging scheme |
| `cell_gradient_averaging_sweep.py` | sweep over averaging/weighting choices |
| `cell_gradient_split_table.py` | split summary tables |
| `cell_gradient_split_permgated.py` | splits restricted to permutation-significant cells |
| `cell_gradient_principal_curve.py` | curved (non-linear) gradient axis |
| `gradient_bending_axis.py` | bending/curvature of the gradient axis |
| `cell_bins_along_curve.py` | binning cells along the principal curve |
| `tier_split_geometry.py` | geometry of the 3-tier split |
| `antpost_split_figures.py` | anterior/posterior split figures |
| `cell_consistency_weighted_brain.py` | consistency-weighted brain overlay |
| `cell_240_diagnostic.py` | diagnostic for the 240° "backward" bump in the middle z-tercile |
| `gradient_brain_cells_by_lag.py` | earlier brain-overlay script; also the shared-helper module the others imported. Its colour wheel, backdrop projection and mask path are now inlined in `cell_fMRI_angle_match.py`, which no longer imports anything from this repo. |

## Caveat on the selection

`cell_averaging_best_split.py` ranks schemes **by how cleanly they match
the expected ventral→dorsal direction**, i.e. it selects on the outcome.
Note that `pc1 / 2 groups / subject-weighted` yields 90°→60° — the
opposite direction — while `pc1 / 2 / unweighted` yields the reported
30°→60°. Any report of the split should say that several
axis/weighting choices were examined, or move to a continuous
circular–linear test of cell lag against gradient-axis position, which
has no forking path.

## Running these again

They assume the old flat layout and do `import gradient_brain_cells_by_lag`
(and, for a few, `from gradient_bending_axis import ...`) as siblings.
`gradient_brain_cells_by_lag.py` and `cell_gradient_master_table.py`
stayed in `scripts/`, so to run anything here either copy it back up one
level or prepend `scripts/` to `sys.path`.
