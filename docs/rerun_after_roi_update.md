# Re-run checklist after the ROI table rebuild (2026-08-27)

## What actually changed

`neurons_with_ROI_labels.csv` was rebuilt so every coordinate comes from the
recording site's own electrode file. **The analysed cell set did not change:**

| | |
|---|---|
| cells | 984 (unchanged) |
| excluded by the `>= 3 subjects` rule | 60 — **the same 60 cells** |
| entered analyses (NaN → ROI) | **0** |
| left analyses (ROI → NaN) | **0** |
| moved between ROIs | **33** |

`EC 38→35 · HC_anterior 275→295 · HC_mid 232→233 · mPFC 155→158 · mOFC 163→142 · PCC 61→61`

Because no cell entered or left, **per-cell statistics do not need recomputing** —
only the ROI grouping does. That is what the `RELABEL_FROM` hooks are for.

## Which analyses are affected

| # | script | affected? | why |
|---|---|---|---|
| 1 | `run_cell_qc.m` | **no** | defines the cell set; unchanged |
| 2 | `cell_to_roi_july26.py` | **done** | this is the rebuild |
| 3 | `behaviour_summary.py` | **no** | behaviour only |
| 4 | `create_fMRI_model_RDMs_on_clean_beh.py` | **no** | behaviour-driven models |
| 5 | `analysis_rodents_complete_clean.py` | **no** | rodent data |
| 6 | `fMRI_run_RSA_without_rsatoolbox_clean.py` | **no** | fMRI voxels |
| 7 | `fMRI_mask_vs_cluster_extract.py` | **no** | fMRI only |
| 8 | `harmonic_angle_maps.py` | **no** | fMRI only |
| 16 | `fMRI_run_RSA_instruction.py` | **no** | fMRI only |
| 17 | `per_TR_loso.py` | **no** | fMRI only |
| 18 | `plot_cell_qc_figure.py` | **no** | QC metrics only |
| 11 | `RSA_DSR_ROIs_simple.py` | **YES — full recompute** | ROI membership defines each pseudo-population, so the data RDMs change |
| 12 | `per_lag_encoding.py` | **YES — reload + relabel** | per-cell CV is ROI-independent |
| 13 | `spatial_peaks_simple.py` | **YES — full run** | `RELOAD_FROM = None` |
| 15 | `encoding_state_sustained_cv.py` | **YES — full run** | `RELOAD_OLD_RESULTS = None` |
| 9 | `harmonic_maps_brain_overlay.py` | **YES** | reads cell coordinates, which moved |
| 10 | `cell_gradient_master_table.py` → `cell_fMRI_angle_match.py` | **YES** | downstream of 9 and 12 |
| 14 | `overlay_double_dissociation.py` | **YES** | downstream of 12/13 |

## Flags: already set correctly

| script | flag | value | meaning |
|---|---|---|---|
| `per_lag_encoding.py` | `RELOAD_FROM` | `2026-06-30_18-21-57` | reuse per-cell CV |
| | `RELABEL_FROM` | canonical ROI table | apply the new ROIs |
| `spatial_peaks_simple.py` | `RELOAD_FROM` | `None` | full run |
| | `RELABEL_FROM` | canonical ROI table | apply the new ROIs |
| `encoding_state_sustained_cv.py` | `RELOAD_OLD_RESULTS` | `None` | full run |
| | `RELABEL_FROM` | `ROI_TABLE_PATH` | apply the new ROIs |
| `RSA_DSR_ROIs_simple.py` | `REUSE_PERMS_FROM_PREVIOUS_RUNS` | `True` | **safe** — the fingerprint includes `cell_ids`, so an ROI whose membership changed rebuilds its null, and PCC (unchanged) legitimately reuses |

## Flags: changed today

`RSA_DSR_ROIs_simple.py`

```python
RELOAD_RUN = None    # was '2026-07-30_15-58-51-fixed_cells-fixed_perms'
```

This was the one genuine trap: with it set, the script skips the RSA and
permutation loop entirely and re-renders plots from the **old** saved CSVs. A
re-run would have silently reproduced the previous result. The previous run tag is
kept in a comment for comparison.

## Order to run, and what to update between steps

Four scripts hardcode an upstream **dated run directory**. They are marked in the
source — `grep -rn "RERUN-CHECK" scripts/` — and each must be pointed at the new
run before the next step, or old and new results get mixed.

```
1.  per_lag_encoding.py                 (reload + relabel; fast)
        └── writes group/per_lag_encoding/<new_run>/per_cell_ALL_ROIs.csv

2.  spatial_peaks_simple.py             (full CV + permutations; slow)
        └── writes group/spatial_peaks_simple/<new_run>/per_cell.csv

3.  encoding_state_sustained_cv.py      (full run)

4.  RSA_DSR_ROIs_simple.py              (full RSA + permutations; slowest)

5.  harmonic_maps_brain_overlay.py      (reads the rebuilt cell coordinates)

6.  cell_gradient_master_table.py
        ⚠ UPDATE `CELL_TABLE` -> step 1's new run dir
        └── writes group/cell_gradient_master/<new_run>/per_cell_master.csv

7.  cell_fMRI_angle_match.py
        ⚠ UPDATE `MASTER_DIR` -> step 6's new run dir
           (its output `final_splits/` sits inside MASTER_DIR, so this also
            stops it overwriting the old result)

8.  overlay_double_dissociation.py
        ⚠ UPDATE `DEFAULT_PER_CELL_CSV` -> step 2's new run dir
        ⚠ UPDATE `DEFAULT_PER_LAG_CSV`  -> step 1's new run dir
```

Steps 1–5 each write a fresh timestamped directory, so the old results stay intact
and old-vs-new can be compared directly. Steps 6–8 inherit their output location
from the input path, so updating the input is what protects the old output.

## Before you start

```bash
conda activate env_multiple_clocks
grep -rn "RERUN-CHECK" scripts/          # the four paths to update
```

Confirm the ROI table in place is the rebuilt one:

```bash
python -c "
import pandas as pd
n=pd.read_csv('$DATA/ephys_humans/derivatives/neurons_with_ROI_labels.csv')
print(n.coord_source.value_counts())
print(n.alt_final_roi.value_counts(dropna=False))"
```

Expect `utah_file_micro 175`, `ucla_file_micro 140`,
`baylor_v2026_bundle_micro 608`, and `mOFC 142 / EC 35 / HC_anterior 295`.

## What to compare tomorrow

The 33 moved cells are concentrated in 9 subjects, so the effects most likely to
shift are the ones where those subjects carry weight:

- **mOFC** loses 21 cells (163→142) — the state-RSA `β = 0.094` and the Fig 2c/5a
  mOFC rows.
- **EC** loses 3 (38→35) — the sustained fraction `12/38 = 31.6 %`, the
  EC-versus-rest Fisher test, and the binomial tests.
- **HC_anterior** gains 20 — the current-location RSA.
- **mPFC** gains 3 (155→158) — the gradient analysis, which also uses the
  coordinates directly (`74/155` inside the mask, the 42/32 median split).
