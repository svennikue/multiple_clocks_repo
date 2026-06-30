#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replot spatial_peaks_simple results from a previous run, with cell exclusion.

Reloads `per_cell.csv` from a previous spatial_peaks_simple run, filters
cells based on exclusion / include-only criteria, re-runs per-ROI stats
and the ROI × lag table, and re-renders all main figures into a
subdirectory `replot_<EXCL_NAME>/` within the same run directory. No heavy
re-computation — the per-cell rate-map fits and permutation draws are
NOT re-run; we just re-aggregate and re-plot from the saved per-cell JSON
columns.

EXCLUSION SPECIFICATION — identical to per_lag_encoding_replot.py:
    EXCL_NEURONS_LIST  : explicit list of neuron IDs (overrides EXCL_CSV)
    EXCL_CSV           : path to a CSV with neuron IDs + criterion column
                          (e.g. encoding_state_sustained_cv/<run>/results.csv
                           with column 'sig_r_state' to drop state-tuned cells)
    EXCL_CRITERION_COL : column in EXCL_CSV
    EXCL_CRITERION_VAL : drop cells where EXCL_CSV[EXCL_CRITERION_COL] == VAL
    EXCL_NEURON_COL    : column in EXCL_CSV holding the neuron ID
                          (defaults to 'neuron' — spatial_peaks per_cell.csv
                          uses 'neuron_id', the join is auto-mapped)
    EXCL_MODE          : 'exclude' (drop flagged) or 'include_only' (keep
                          only flagged) — the same modes as the per_lag
                          replot, for the complementary-cell analysis.

USE CASES — parallel to per_lag_encoding_replot.py.

OUTPUTS under <reload_run_dir>/replot_<EXCL_NAME>/:
    per_cell_filtered.csv
    per_roi_stats_filtered.csv
    per_roi_lag_table_filtered.csv
    replot_manifest.json
    figures: roi_x_lag_tstat_overview, roi_x_lag_meanR_heatmap,
             test1_meanR_histograms, test2_targetVsOther_lines,
             test3_permSig_bars, fixed_vs_free_comparison, zMNI_gradient_ACC

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

REPO = '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo'
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

# Importing the analysis module is safe (heavy work is behind __main__).
from spatial_peaks_simple import (
    _add_perm_p_columns,
    _per_roi_stats,
    _per_roi_lag_table,
    plot_roi_x_lag_overview,
    plot_test1_mean_r_histograms,
    plot_test2_target_vs_others_lines,
    plot_test3_perm_sig_fraction_bars,
    plot_roi_x_lag_table_heatmap,
    plot_fixed_vs_free_comparison,
    plot_zmni_gradient,
    OUT_BASE,
)


# ── Settings ──────────────────────────────────────────────────────────
RELOAD_RUN = '2026-06-26_18-47-11_phase_resid_paired_fixedlag'

# EXCL_NAME also dispatches to a named-mode auto-builder when it matches
# one of these reserved names; otherwise we fall back to the manual
# EXCL_NEURONS_LIST / EXCL_CSV settings below.
#   'no_rsa_cells'      → drop every cell listed in
#                          `RSA_CELLS_SOURCE_CSV` (roi_electrode_coords.csv
#                          from a DSR RSA run). Match by (subject, cell_idx).
#   'no_state_tuned'    → drop every cell flagged by the per-cell perm
#                          test in `STATE_TUNED_SOURCE_CSV`. Match by
#                          neuron-ID. Criterion column / value editable
#                          below.
EXCL_NAME = 'no_state_tuned'   # 'no_state_tuned' / 'no_rsa_cells' / any custom label
EXCL_MODE = 'exclude'        # 'exclude' or 'include_only'

# ── Named-mode sources (used when EXCL_NAME is a reserved name) ──────
_DATA = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'

STATE_TUNED_SOURCE_CSV    = (f'{_DATA}/group/encoding_state_sustained_cv/'
                              '2026-06-25_14-38-13/state_sustained_cv_results.csv')
STATE_TUNED_CRITERION_COL = 'sig_sustained'   # perm-sig sustained-state cell
STATE_TUNED_CRITERION_VAL = True               # use 'sig_sustained_fdr' for FDR-strict

RSA_CELLS_SOURCE_CSV      = (f'{_DATA}/group/DSR_RSA_simple_ROI/'
                              '2026-06-22_16-17-15-final-DSR/roi_electrode_coords.csv')

# ── Manual fallback (when EXCL_NAME is NOT a reserved name) ──────────
# Option A: explicit list of neuron IDs to drop (highest precedence).
EXCL_NEURONS_LIST: list[str] = []

# Option B: load flagged neuron IDs from a CSV.
EXCL_CSV           = ''                # set to a path to use this mode
EXCL_CRITERION_COL = 'sig_r_state'
EXCL_CRITERION_VAL = True
EXCL_NEURON_COL    = 'neuron'


# ── Helpers ───────────────────────────────────────────────────────────
def _build_flagged_set(per_cell):
    """Return the set of neuron IDs to be flagged for exclusion.

    Dispatches on EXCL_NAME:
      * 'no_rsa_cells'   - flags every cell whose (subject_int, cell_idx)
                            appears in RSA_CELLS_SOURCE_CSV. Resolves the
                            ID-set against `per_cell` so the returned set
                            matches the `neuron_id` column directly.
      * 'no_state_tuned' - flags every neuron in STATE_TUNED_SOURCE_CSV
                            whose STATE_TUNED_CRITERION_COL equals
                            STATE_TUNED_CRITERION_VAL.
      * any other label  - falls back to EXCL_NEURONS_LIST → EXCL_CSV.
    """
    # Named-mode dispatchers --------------------------------------------
    if EXCL_NAME == 'no_rsa_cells':
        if not os.path.exists(RSA_CELLS_SOURCE_CSV):
            raise FileNotFoundError(
                f'RSA cells source CSV not found: {RSA_CELLS_SOURCE_CSV}')
        rsa = pd.read_csv(RSA_CELLS_SOURCE_CSV)
        for c in ('subject', 'cell_idx'):
            if c not in rsa.columns:
                raise ValueError(
                    f'{RSA_CELLS_SOURCE_CSV} missing column {c!r}')
        rsa_pairs = set(zip(rsa['subject'].astype(int),
                             rsa['cell_idx'].astype(int)))
        # spatial_peaks per_cell.csv has subject_int + cell_idx → match.
        sub = per_cell[['neuron_id', 'subject_int', 'cell_idx']].copy()
        in_rsa = sub.apply(
            lambda r: (int(r['subject_int']), int(r['cell_idx'])) in rsa_pairs,
            axis=1,
        )
        flagged = set(sub.loc[in_rsa, 'neuron_id'].astype(str).tolist())
        print(f"[no_rsa_cells] {len(rsa_pairs)} RSA pairs → "
              f"{len(flagged)} matching neuron_ids in this run")
        return flagged

    if EXCL_NAME == 'no_state_tuned':
        if not os.path.exists(STATE_TUNED_SOURCE_CSV):
            raise FileNotFoundError(
                f'State CSV not found: {STATE_TUNED_SOURCE_CSV}')
        ext = pd.read_csv(STATE_TUNED_SOURCE_CSV)
        if STATE_TUNED_CRITERION_COL not in ext.columns:
            raise ValueError(
                f'{STATE_TUNED_SOURCE_CSV} missing column '
                f'{STATE_TUNED_CRITERION_COL!r}; has {list(ext.columns)}')
        if 'neuron' not in ext.columns:
            raise ValueError(
                f'{STATE_TUNED_SOURCE_CSV} missing column "neuron"')
        flagged = set(ext.loc[
            ext[STATE_TUNED_CRITERION_COL] == STATE_TUNED_CRITERION_VAL,
            'neuron',
        ].astype(str).unique().tolist())
        print(f"[no_state_tuned] {len(flagged)} neurons flagged where "
              f"{STATE_TUNED_CRITERION_COL}={STATE_TUNED_CRITERION_VAL}")
        return flagged

    # Manual fallback ---------------------------------------------------
    if EXCL_NEURONS_LIST:
        return set(map(str, EXCL_NEURONS_LIST))
    if not EXCL_CSV or not os.path.exists(EXCL_CSV):
        print(f"[warn] EXCL_NAME={EXCL_NAME!r} is not a reserved mode and "
              f"EXCL_CSV is unset/missing ({EXCL_CSV!r}); no cells "
              f"will be filtered.")
        return set()
    ext = pd.read_csv(EXCL_CSV)
    if EXCL_NEURON_COL not in ext.columns:
        raise ValueError(f"EXCL_CSV missing column {EXCL_NEURON_COL!r}; "
                         f"has {list(ext.columns)}")
    if EXCL_CRITERION_COL not in ext.columns:
        raise ValueError(f"EXCL_CSV missing column {EXCL_CRITERION_COL!r}; "
                         f"has {list(ext.columns)}")
    flagged = ext.loc[ext[EXCL_CRITERION_COL] == EXCL_CRITERION_VAL,
                       EXCL_NEURON_COL].astype(str).unique().tolist()
    return set(flagged)


def _apply_mask(per_cell_df, flagged):
    """spatial_peaks per_cell.csv uses 'neuron_id'."""
    id_col = 'neuron_id' if 'neuron_id' in per_cell_df.columns else 'neuron'
    ids = per_cell_df[id_col].astype(str)
    in_set = ids.isin(flagged).to_numpy()
    if EXCL_MODE == 'exclude':
        keep = ~in_set
    elif EXCL_MODE == 'include_only':
        keep = in_set
    else:
        raise ValueError(f"EXCL_MODE must be 'exclude' or 'include_only', "
                         f"got {EXCL_MODE!r}")
    return per_cell_df[keep].reset_index(drop=True), keep


# ── Main ──────────────────────────────────────────────────────────────
def main():
    run_dir = os.path.join(OUT_BASE, RELOAD_RUN)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    csv_path = os.path.join(run_dir, 'per_cell.csv')
    print(f"Loading {csv_path}")
    per_cell = pd.read_csv(csv_path)
    print(f"  {len(per_cell)} cells across {per_cell.roi.nunique()} ROIs")

    # Recompute perm-p columns (they're in the saved CSV but safe to refresh).
    per_cell = _add_perm_p_columns(per_cell)

    flagged = _build_flagged_set(per_cell)
    print(f"  {len(flagged)} flagged neuron IDs from exclusion source")
    per_cell_f, keep_mask = _apply_mask(per_cell, flagged)

    n_in, n_kept = int(len(per_cell)), int(len(per_cell_f))
    n_drop = n_in - n_kept
    print(f"After filter (mode={EXCL_MODE!r}): "
          f"{n_kept} kept, {n_drop} dropped (of {n_in})")
    print("  per-ROI cell counts:")
    for roi, n in per_cell_f.groupby('roi').size().items():
        in_n = int((per_cell['roi'] == roi).sum())
        print(f"    {roi:<18s} {n:>4d}  (was {in_n})")

    if n_kept < 5:
        print("\nFewer than 5 cells kept overall — aborting replot.")
        return

    out_dir = os.path.join(run_dir, f'replot_{EXCL_NAME}')
    os.makedirs(out_dir, exist_ok=True)

    per_cell_f.to_csv(os.path.join(out_dir, 'per_cell_filtered.csv'),
                       index=False)

    # Re-run stats + ROI×lag table on filtered cells.
    roi_stats   = _per_roi_stats(per_cell_f)
    roi_x_lag   = _per_roi_lag_table(per_cell_f)
    roi_stats.to_csv(os.path.join(out_dir, 'per_roi_stats_filtered.csv'),
                      index=False)
    roi_x_lag.to_csv(os.path.join(out_dir, 'per_roi_lag_table_filtered.csv'),
                      index=False)

    manifest = {
        'reload_run':       RELOAD_RUN,
        'replot_timestamp': datetime.now().isoformat(timespec='seconds'),
        'excl_name':        EXCL_NAME,
        'excl_csv':         None if EXCL_NEURONS_LIST else EXCL_CSV,
        'excl_criterion_col': None if EXCL_NEURONS_LIST else EXCL_CRITERION_COL,
        'excl_criterion_val': None if EXCL_NEURONS_LIST else EXCL_CRITERION_VAL,
        'excl_neuron_col':  EXCL_NEURON_COL,
        'excl_list_size':   len(EXCL_NEURONS_LIST),
        'excl_mode':        EXCL_MODE,
        'n_flagged':        len(flagged),
        'n_in':             n_in,
        'n_kept':           n_kept,
        'n_dropped':        n_drop,
        'kept_per_roi':     {k: int(v) for k, v in
                              per_cell_f.groupby('roi').size().items()},
    }
    with open(os.path.join(out_dir, 'replot_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    # Re-render figures — same set as the main spatial_peaks run.
    plot_roi_x_lag_overview(per_cell_f, roi_stats,
        os.path.join(out_dir, 'roi_x_lag_tstat_overview'))
    plot_roi_x_lag_table_heatmap(roi_x_lag,
        os.path.join(out_dir, 'roi_x_lag_meanR_heatmap'))
    plot_test1_mean_r_histograms(per_cell_f, roi_stats,
        os.path.join(out_dir, 'test1_meanR_histograms'))
    plot_test2_target_vs_others_lines(per_cell_f, roi_stats,
        os.path.join(out_dir, 'test2_targetVsOther_lines'))
    plot_test3_perm_sig_fraction_bars(per_cell_f, roi_stats,
        os.path.join(out_dir, 'test3_permSig_bars'))
    plot_fixed_vs_free_comparison(per_cell_f,
        os.path.join(out_dir, 'fixed_vs_free_comparison'))
    try:
        plot_zmni_gradient(per_cell_f,
            os.path.join(out_dir, 'zMNI_gradient_ACC'))
    except Exception as exc:
        print(f"[warn] zMNI gradient plot skipped: {exc}")

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == '__main__':
    main()
