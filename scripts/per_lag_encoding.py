#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-lag single-feature elastic-net encoding.

For each ROI, for each cell, for each of 12 lag windows (0–330° in 30° steps):
fit elastic-net with just 9 location features encoding "location at lag k",
leave-one-config-out cross-validation, save per-cell CV r at each lag.

Output:
    per_lag_encoding_{ROI}.csv   one row per cell, 12 lag columns + neuron_id

Phase residualisation applied per cell before any fit.

Run:
    python scripts/per_lag_encoding.py

Customise the ROIS_TO_RUN list at the top of the script.

@author: Svenja Küchenhoff
"""

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import pearsonr, ttest_1samp
from sklearn.linear_model import ElasticNet

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import mc.analyse.cell_selection as cs
import mc.analyse.helpers_human_cells as hh
from mc.analyse.future_spatial_peaks import _residualise_phase


# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_DIR  = os.path.join(DATA_DIR, 'group', 'per_lag_encoding')

# Which ROIs to run. Set to None for all in the ROI table.
ROIS_TO_RUN = ['ACC', 'medialOFC', 'PCC', 'Parahippocampal',
               'HC_anterior', 'HC_mid', 'EC']

# Phase residualisation basis (None to skip; 'cosine' is canonical)
PHASE_RESIDUALISE = 'cosine'

# Trials filter, passed to mc.analyse.helpers_human_cells.filter_data
TRIALS = 'all_minus_explore'   # 'residualised' to also remove rep_correct slope

# Elastic-net
ALPHA = 0.001
L1_RATIO = 0.5
MAX_ITER = 2000

# Lag definition
N_BINS = 360
N_LAGS = 12
N_LOC  = 9
STEP   = N_BINS // N_LAGS    # 30 bins per lag window
LAG_BINS = [k * STEP for k in range(N_LAGS)]   # 0, 30, 60, ..., 330


# ── Per-cell fit ──────────────────────────────────────────────────────

def build_loc_at_lag(mode_loc_int, lag, n_step=STEP):
    """Return (9, 360) one-hot of the location at time (t + lag*n_step) mod 360."""
    n = len(mode_loc_int)
    out = np.zeros((N_LOC, n))
    for t in range(n):
        l = mode_loc_int[(t + lag * n_step) % n]
        if 0 <= l < N_LOC:
            out[l, t] = 1.0
    return out


def fit_per_lag_one_cell(arr_clean, idx_cfg, locs, n_cfg):
    """Returns list of CV r values, one per lag."""
    # build per-config Y and mode location series
    Y_cfg = np.zeros((n_cfg, N_BINS))
    mode_loc_cfg = np.zeros((n_cfg, N_BINS), dtype=int)
    for c in range(n_cfg):
        mask = idx_cfg == c
        Y_cfg[c] = np.nanmean(arr_clean[mask], axis=0)
        for t in range(N_BINS):
            vals = locs[mask, t]
            vals = vals[np.isfinite(vals)].astype(int)
            mode_loc_cfg[c, t] = Counter(vals).most_common(1)[0][0] - 1 if len(vals) else 0
    out = []
    for lag in range(N_LAGS):
        # build (n_cfg, 9, 360), flatten to (n_cfg * 360, 9)
        X_cfg = np.array([build_loc_at_lag(mode_loc_cfg[c], lag) for c in range(n_cfg)])
        X_flat = X_cfg.transpose(0, 2, 1).reshape(-1, N_LOC)
        Y_flat = Y_cfg.flatten()
        m = np.isfinite(Y_flat)
        X_flat = X_flat[m]; Y_flat = Y_flat[m]
        # LOO over configs
        rs = []
        split_size = X_flat.shape[0] // n_cfg
        for held_i in range(n_cfg):
            test_idx = np.arange(held_i * split_size, (held_i + 1) * split_size)
            train_idx = np.array([i for i in range(X_flat.shape[0]) if i not in test_idx])
            model = ElasticNet(alpha=ALPHA, l1_ratio=L1_RATIO, max_iter=MAX_ITER)
            model.fit(X_flat[train_idx], Y_flat[train_idx])
            yhat = model.predict(X_flat[test_idx])
            if np.std(yhat) > 0 and np.std(Y_flat[test_idx]) > 0:
                rs.append(pearsonr(yhat, Y_flat[test_idx])[0])
        out.append(np.mean(rs) if rs else np.nan)
    return out


# ── Main loop ─────────────────────────────────────────────────────────

def run_roi(roi):
    print(f'\n══ ROI: {roi} ══')
    cells = cs.load_cells(cell_set='all_in_roi_table', rois_keep=[roi])
    print(f'  {len(cells)} cells across {cells["subject_id"].nunique()} subjects')

    results = []
    neuron_labels = []
    subjects_done = 0
    for sub_str in sorted(cells['subject_id'].unique()):
        try:
            data_raw = hh.load_norm_data(DATA_DIR, [sub_str], res_data=False)
        except Exception as exc:
            print(f'  sub-{sub_str} load failed: {exc}'); continue
        if not data_raw: continue
        data = hh.filter_data(data_raw, int(sub_str), TRIALS)
        sub_dict = data[f"sub-{sub_str}"]
        beh  = sub_dict['beh'].copy().reset_index(drop=True)
        locs = sub_dict['locations'].to_numpy(dtype=float)
        uniq, _, idx_cfg, _ = np.unique(beh[['loc_A','loc_B','loc_C','loc_D']].to_numpy(),
                                        axis=0, return_index=True, return_inverse=True, return_counts=True)
        n_cfg = len(np.unique(idx_cfg))
        if n_cfg < 5: print(f'  sub-{sub_str} skipped ({n_cfg} configs)'); continue
        sub_keep = set(cells.loc[cells['subject_id']==sub_str, 'cell_idx'].tolist())
        for nid, n_df in sub_dict['normalised_neurons'].items():
            _, ci = cs.parse_neuron_label(nid)
            if ci not in sub_keep: continue
            arr = n_df.to_numpy(dtype=float)
            if PHASE_RESIDUALISE:
                arr = _residualise_phase(arr, basis=PHASE_RESIDUALISE)
            try:
                cv_rs = fit_per_lag_one_cell(arr, idx_cfg, locs, n_cfg)
            except Exception as exc:
                print(f'    {nid} failed: {exc}'); continue
            results.append(cv_rs); neuron_labels.append(nid)
        subjects_done += 1
    print(f'  Done: {len(results)} cells from {subjects_done} subjects')

    if not results:
        return None
    arr_r = np.array(results)
    df_out = pd.DataFrame(arr_r, columns=[f'lag_{l}' for l in LAG_BINS])
    df_out.insert(0, 'neuron', neuron_labels)
    df_out['roi'] = roi
    # Print population stats
    print(f'  Per-lag t-test vs 0:')
    for i, l in enumerate(LAG_BINS):
        rs = arr_r[:, i]; rs = rs[np.isfinite(rs)]
        t, p = ttest_1samp(rs, 0)
        flag = ' ★' if p < 0.05 else (' ·' if p < 0.10 else '')
        print(f'    +{l:>3d}°  mean={rs.mean():+.4f}  t={t:+.2f}  p={p:.3f}{flag}')
    return df_out


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    run_tag = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_run_dir = os.path.join(OUT_DIR, run_tag)
    os.makedirs(out_run_dir, exist_ok=True)
    print(f'Output dir: {out_run_dir}')

    # Config snapshot
    config = {
        'phase_residualise': PHASE_RESIDUALISE,
        'trials':           TRIALS,
        'alpha':            ALPHA,
        'l1_ratio':         L1_RATIO,
        'n_lags':           N_LAGS,
        'rois_to_run':      ROIS_TO_RUN,
        'run_tag':          run_tag,
    }
    import json
    with open(os.path.join(out_run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    rois = ROIS_TO_RUN or sorted(cs.load_cells(cell_set='all_in_roi_table')['roi'].unique())
    all_dfs = []
    for roi in rois:
        df = run_roi(roi)
        if df is not None:
            csv_path = os.path.join(out_run_dir, f'per_lag_{roi}.csv')
            df.to_csv(csv_path, index=False)
            print(f'  saved → {csv_path}')
            all_dfs.append(df)
    if all_dfs:
        pd.concat(all_dfs).to_csv(
            os.path.join(out_run_dir, 'per_lag_ALL_ROIs.csv'), index=False)
        print(f'\nCombined output: {out_run_dir}/per_lag_ALL_ROIs.csv')
