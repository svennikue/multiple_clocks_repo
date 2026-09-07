#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-lag encoding restricted to cells that could NOT enter the
population-RSA pseudopopulation.

WHY
    The RSA pseudopopulation can only be built from sessions in which the
    subject solved the SAME 8 reward layouts (the sessions listed in
    `all_sessions_dsrRSA_grouping_summary.json`; see
    `RSA_DSR_ROIs_simple.py`). Every other session experienced different
    layouts and is therefore absent from that analysis. Re-running the
    per-lag spatial-consistency test on exactly those excluded sessions
    gives a cell cohort with ZERO overlap with the RSA sample, i.e. an
    independent replication of the future-tuning result.

WHAT IT DOES / DOES NOT DO
    It does NOT recompute cross-validated correlations or permutations.
    It reads the per-cell CV r (and per-cell permutation p) from an
    existing `per_lag_encoding` run, drops the RSA-eligible sessions, and
    re-runs the SAME statistics functions the main analysis uses:
        per_lag_encoding._stats_and_plots        (T1/T1a/T2/T3, both units)
        overlay_double_dissociation.make_overlay (window + peak-lag perm)

    Cohort definition is at SESSION level. A cell is "non-RSA" iff its
    session is not a key of the RSA grouping summary. Cells dropped
    inside the RSA for other reasons (e.g. NaN in a run) are NOT counted
    as non-RSA here, which keeps the criterion a single, pre-existing
    inclusion rule rather than a post-hoc per-cell one.

@author: Svenja Kuchenhoff
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import per_lag_encoding as ple
from overlay_double_dissociation import make_overlay

# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
SOURCE_RUN = os.path.join(
    DATA_DIR, 'group', 'per_lag_encoding',
    '2026-08-28_10-18-21_reload_from_2026-06-30_18-21-57_relabelled')
RSA_GROUPING_JSON = os.path.join(DATA_DIR,
                                 'all_sessions_dsrRSA_grouping_summary.json')
OUT_BASE = os.path.join(DATA_DIR, 'group', 'per_lag_encoding_nonRSA_cohort')
CTRL_MODE = 'noctrl'


def _session_key(subject_id):
    """per_cell `subject_id` (int or 'sub-03') -> zero-padded session str."""
    s = str(subject_id).replace('sub-', '')
    if s.endswith('.0'):
        s = s[:-2]
    return s.zfill(2)


def main():
    run_tag = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_nonRSA_cohort'
    out_dir = os.path.join(OUT_BASE, run_tag)
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    print(f'Output dir: {out_dir}')

    per_cell = pd.read_csv(os.path.join(SOURCE_RUN, 'per_cell_ALL_ROIs.csv'))
    per_cell = ple._canonicalize_roi_names(per_cell)

    rsa_sessions = {s.zfill(2) for s in
                    json.load(open(RSA_GROUPING_JSON)).keys()}
    session = per_cell['subject_id'].map(_session_key)
    in_rsa = session.isin(rsa_sessions)

    kept = per_cell[~in_rsa].copy()
    print(f'  {len(per_cell)} cells total -> {len(kept)} non-RSA cells '
          f'from {kept["subject_id"].nunique()} sessions '
          f'({in_rsa.sum()} cells from {int(in_rsa.groupby(session).any().sum())} '
          f'RSA sessions dropped)')
    print(kept.groupby('roi').agg(n_cells=('neuron', 'size'),
                                  n_sessions=('subject_id', 'nunique')))

    csv_path = os.path.join(out_dir, 'per_cell_ALL_ROIs.csv')
    kept.to_csv(csv_path, index=False)

    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({
            'analysis': 'per_lag_encoding restricted to non-RSA sessions',
            'source_run': SOURCE_RUN,
            'rsa_grouping_json': RSA_GROUPING_JSON,
            'cohort_rule': ('cell kept iff its session is NOT a key of the '
                            'RSA grouping summary (session-level, no '
                            'per-cell post-hoc criterion)'),
            'n_rsa_sessions': len(rsa_sessions),
            'n_cells_source': int(len(per_cell)),
            'n_cells_kept': int(len(kept)),
            'n_sessions_kept': int(kept['subject_id'].nunique()),
            'cells_per_roi': kept.groupby('roi').size().to_dict(),
            'recomputed': 'statistics only — CV r and permutation p reused',
            'ctrl_mode': CTRL_MODE,
            'roi_predicted_lags': {k: list(v) for k, v in
                                   ple.ROI_PREDICTED_LAGS_DEG.items()},
            'lags_deg': ple.LAGS_DEG,
        }, f, indent=2)

    ple._stats_and_plots(kept, out_dir, fig_dir)

    overlay_dir = os.path.join(out_dir, f'overlay_double_dissociation_{CTRL_MODE}')
    make_overlay(csv_path, overlay_dir, source='per_lag',
                 ctrl_mode=CTRL_MODE, weighting='both')

    print(f'\nDone. Outputs in {out_dir}')


if __name__ == '__main__':
    main()
