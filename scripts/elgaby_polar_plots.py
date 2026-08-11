#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polar plots of single neurons in el-gaby style.

For each ROI, picks a small number of cells under several criteria and
plots, per cell, an 8-config grid of polar traces (one per config) plus
a mean polar. Useful for visually checking whether phase / state tuning
in human MTL really has the polar structure that the regression
analysis assumes.

Inputs:
  - tuning CSV from Script 1 (`elgaby_tuning_characterization.py`)
  - optional encoding-results CSV from Script 2 (used to rank cells
    by held-out r); falls back to tuning-only ranking if absent

Outputs go to <encoding_run>/polar_plots/<criterion>/<roi>/<cell>.png
or (if no encoding run is found) under
  derivatives/group/elgaby_polar_plots/<run_tag>/

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import mc
from mc.plotting import elgaby_polar


# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
TUNING_BASE = os.path.join(DATA_DIR, 'group', 'elgaby_tuning')
ENCODING_BASE = os.path.join(DATA_DIR, 'group', 'encoding_analysis_elgaby')
OUT_BASE = os.path.join(DATA_DIR, 'group', 'elgaby_polar_plots')

# Which runs to pull from. None -> most recent in each base folder.
TUNING_RUN_TAG = None
ENCODING_RUN_TAG = None   # set to '' to skip the encoding-based ranking

# Which ROIs to plot. None -> all ROIs that appear in the tuning CSV.
TARGET_ROIS = None
# ROIs with fewer cells than this are skipped (no point picking 'best
# 5' from 3 cells).
MIN_CELLS_PER_ROI = 3

# How many cells to plot per (ROI, criterion).
N_CELLS_PER_CRITERION = 5

# Gaussian-smoothing sigma (in bins) applied to the polar traces before
# plotting. Use 0 / None to disable. El-Gaby's paper uses sigma=10.
SMOOTH_SIGMA = 10

# Selection criteria.
# Each entry is (key, label).  See `select_cells` for the available keys.
CRITERIA = [
    ('most_state_tuned',     'Most state-tuned (across configs)'),
    ('most_phase_tuned',     'Most phase-tuned (across configs)'),
    ('best_r',               'Highest mean held-out r'),
    ('worst_r',              'Lowest mean held-out r'),
]

# ROI label table (matches the rest of the pipeline).
ROI_TABLE_PATH = os.path.join(DATA_DIR, 'neurons_with_ROI_labels.csv')
ROI_LABEL_COLUMN = 'alt_final_roi'

N_JOBS = -1


# ── Resolve runs ─────────────────────────────────────────────────────
def find_latest_run(base):
    if not os.path.isdir(base):
        return None
    candidates = [d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d))
                  and not d.endswith('-null')]
    if not candidates:
        return None
    candidates.sort(key=lambda d: os.path.getmtime(os.path.join(base, d)))
    return candidates[-1]


if TUNING_RUN_TAG is None:
    TUNING_RUN_TAG = find_latest_run(TUNING_BASE)
if TUNING_RUN_TAG is None:
    raise FileNotFoundError(f"No tuning run found under {TUNING_BASE}")
TUNING_DIR = os.path.join(TUNING_BASE, TUNING_RUN_TAG)
TUNING_CSV = os.path.join(TUNING_DIR, 'tuning_per_neuron_config.csv')
print(f"Tuning run: {TUNING_DIR}")

if ENCODING_RUN_TAG is None:
    ENCODING_RUN_TAG = find_latest_run(ENCODING_BASE)
if ENCODING_RUN_TAG:
    ENCODING_DIR = os.path.join(ENCODING_BASE, ENCODING_RUN_TAG)
    ENCODING_CSV = os.path.join(ENCODING_DIR, 'encoding_results.csv')
    if not os.path.isfile(ENCODING_CSV):
        print(f"  encoding_results.csv not in {ENCODING_DIR}; skipping r-based picks")
        ENCODING_CSV = None
    else:
        print(f"Encoding run: {ENCODING_DIR}")
else:
    ENCODING_DIR = None
    ENCODING_CSV = None
    print("No encoding run found; selection by held-out r will be skipped.")

# Output folder: inside the encoding run if we have one, otherwise its own
# timestamped folder.
if ENCODING_DIR is not None:
    OUT_DIR = os.path.join(ENCODING_DIR, 'polar_plots')
else:
    RUN_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    OUT_DIR = os.path.join(OUT_BASE, RUN_TAG)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Writing plots into {OUT_DIR}")

with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump({
        'tuning_run':     TUNING_RUN_TAG,
        'encoding_run':   ENCODING_RUN_TAG,
        'target_rois':    TARGET_ROIS,
        'n_cells_per_criterion': N_CELLS_PER_CRITERION,
        'criteria':       [c[0] for c in CRITERIA],
        'timestamp':      datetime.now().isoformat(timespec='seconds'),
    }, f, indent=2)


# ── Load tuning + encoding tables ────────────────────────────────────
tuning_df = pd.read_csv(TUNING_CSV)
tuning_df['subject'] = tuning_df['subject'].map(lambda s: f'{int(s):02d}')
print(f"Loaded tuning CSV: {len(tuning_df)} (neuron, config) rows.")

if ENCODING_CSV is not None:
    enc_df = pd.read_csv(ENCODING_CSV)
    enc_df['subject'] = enc_df['subject'].map(lambda s: f'{int(s):02d}')
    print(f"Loaded encoding CSV: {len(enc_df)} (neuron, config) rows.")
else:
    enc_df = None


# ── Per-neuron summary used for ranking ──────────────────────────────
def per_neuron_summary(tuning_df, enc_df):
    """One row per (subject, neuron) with ranking-relevant stats."""
    rows = []
    grouper = tuning_df.groupby(['subject', 'neuron'], sort=False)
    for (sub, neuron), g in grouper:
        roi = g['roi'].iloc[0] if not g['roi'].isna().all() else None
        if roi is None:
            continue
        rows.append({
            'subject':              sub,
            'neuron':               neuron,
            'roi':                  roi,
            'n_configs':            int(len(g)),
            'n_state_tuned':        int(g['elgaby_state_tuned'].sum()),
            'n_phase_tuned':        int(g['elgaby_phase_tuned'].sum()),
            'state_tuning_p_min':   float(g['elgaby_state_tuning_p'].min()),
            'phase_tuning_p_min':   float(g['elgaby_phase_tuning_p'].min()),
        })
    df = pd.DataFrame(rows)
    if enc_df is not None and not enc_df.empty:
        agg = (enc_df.groupby(['subject', 'neuron'])
                     .agg(mean_r=('r_used', 'mean'),
                          n_r=('r_used', 'count'))
                     .reset_index())
        df = df.merge(agg, on=['subject', 'neuron'], how='left')
    else:
        df['mean_r'] = np.nan
        df['n_r'] = 0
    return df


neuron_summary = per_neuron_summary(tuning_df, enc_df)
print(f"Built per-neuron summary: {len(neuron_summary)} neurons.")


# ── Cell selection per ROI x criterion ───────────────────────────────
def select_cells(roi_df, criterion, n):
    """Return up to `n` cells from `roi_df` (already restricted to one ROI)."""
    if criterion == 'most_state_tuned':
        return roi_df.sort_values('n_state_tuned', ascending=False).head(n)
    if criterion == 'most_phase_tuned':
        return roi_df.sort_values('n_phase_tuned', ascending=False).head(n)
    if criterion == 'best_r':
        g = roi_df.dropna(subset=['mean_r'])
        return g.sort_values('mean_r', ascending=False).head(n)
    if criterion == 'worst_r':
        g = roi_df.dropna(subset=['mean_r'])
        return g.sort_values('mean_r', ascending=True).head(n)
    raise ValueError(f"Unknown criterion: {criterion!r}")


# ── Per-cell trace builder ───────────────────────────────────────────
def build_traces(neuron_label, subject, neuron_df, beh, configs):
    """Trial-averaged 360-bin trace per config, plus tuning + r per config."""
    tuning_slice = tuning_df[
        (tuning_df['subject'] == subject) &
        (tuning_df['neuron'] == neuron_label)
    ].set_index('config')

    if enc_df is not None:
        enc_slice = enc_df[
            (enc_df['subject'] == subject) &
            (enc_df['neuron'] == neuron_label)
        ].set_index('test_config')
    else:
        enc_slice = None

    traces, pref_phase, pref_state, state_t, phase_t = [], [], [], [], []
    n_trials, r_per_cfg = [], []
    for cfg in configs:
        mask = (beh['config_str'] == cfg) & (beh['correct'] == 1)
        idx = beh.index[mask].to_numpy()
        n_trials.append(int(len(idx)))
        if len(idx) == 0:
            traces.append(np.full(360, np.nan))
        else:
            traces.append(
                np.nanmean(neuron_df.iloc[idx].to_numpy(), axis=0)
            )
        if cfg in tuning_slice.index:
            row = tuning_slice.loc[cfg]
            pref_phase.append(int(row['elgaby_pref_phase']))
            pref_state.append(int(row['elgaby_pref_state']))
            state_t.append(bool(row['elgaby_state_tuned']))
            phase_t.append(bool(row['elgaby_phase_tuned']))
        else:
            pref_phase.append(-1); pref_state.append(-1)
            state_t.append(False); phase_t.append(False)
        if enc_slice is not None and cfg in enc_slice.index:
            r_per_cfg.append(float(enc_slice.loc[cfg, 'r_used']))
        else:
            r_per_cfg.append(np.nan)

    return {
        'traces': traces,
        'pref_phase': pref_phase,
        'pref_state': pref_state,
        'state_tuned': state_t,
        'phase_tuned': phase_t,
        'n_trials': n_trials,
        'r_per_cfg': r_per_cfg,
    }


# ── Plotting routine ─────────────────────────────────────────────────
def plot_one_cell(neuron_label, subject, summary_row,
                  sub_dict, criterion_label, out_path):
    """Build polar plot for one neuron and save to disk."""
    beh = sub_dict['beh'].copy().reset_index(drop=True)
    beh['config_str'] = (
        beh['loc_A'].astype(int).astype(str) + '-' +
        beh['loc_B'].astype(int).astype(str) + '-' +
        beh['loc_C'].astype(int).astype(str) + '-' +
        beh['loc_D'].astype(int).astype(str)
    )
    configs = sorted(beh['config_str'].dropna().unique().tolist())
    neuron_df = sub_dict['normalised_neurons'][neuron_label].reset_index(drop=True)

    info = build_traces(neuron_label, subject, neuron_df, beh, configs)

    title_bits = [
        f"{neuron_label}  [{summary_row['roi']}]",
        f"{criterion_label}",
        f"n_state_tuned={summary_row['n_state_tuned']}/{summary_row['n_configs']}",
        f"n_phase_tuned={summary_row['n_phase_tuned']}/{summary_row['n_configs']}",
    ]
    if np.isfinite(summary_row.get('mean_r', np.nan)):
        title_bits.append(f"mean r = {summary_row['mean_r']:+.3f}")
    title = ' | '.join(title_bits)

    elgaby_polar.plot_cell_polar(
        traces_per_config=info['traces'],
        configs=configs,
        pref_phase_per_config=info['pref_phase'],
        pref_state_per_config=info['pref_state'],
        state_tuned_per_config=info['state_tuned'],
        phase_tuned_per_config=info['phase_tuned'],
        r_per_config=info['r_per_cfg'],
        n_trials_per_config=info['n_trials'],
        smooth_sigma=SMOOTH_SIGMA,
        title=title,
        out_path=out_path,
    )


# ── Main: iterate ROIs × criteria, dispatch per-subject ──────────────
def main():
    target_set = set(TARGET_ROIS) if TARGET_ROIS else None

    # Roll up: which (subject, neuron) we need to plot for each (criterion, roi).
    work = []  # (criterion_key, criterion_label, roi, subject, neuron, summary_row)
    for roi, roi_df in neuron_summary.groupby('roi', sort=False):
        if target_set is not None and roi not in target_set:
            continue
        if len(roi_df) < MIN_CELLS_PER_ROI:
            print(f"  skipping ROI {roi}: only {len(roi_df)} cells")
            continue
        for crit_key, crit_label in CRITERIA:
            picks = select_cells(roi_df, crit_key, N_CELLS_PER_CRITERION)
            for _, row in picks.iterrows():
                work.append((crit_key, crit_label, roi,
                             row['subject'], row['neuron'], row))

    if not work:
        print("Nothing to plot.")
        return

    # Group by subject so we only load each subject's data once.
    by_subject = {}
    for item in work:
        _, _, _, sub, _, _ = item
        by_subject.setdefault(sub, []).append(item)

    n_total = 0
    for sub in sorted(by_subject):
        print(f"\nLoading sub-{sub} ({len(by_subject[sub])} cell-plots to render)...")
        sub_data = mc.analyse.helpers_human_cells.load_norm_data(
            DATA_DIR, [sub],
        )
        if f"sub-{sub}" not in sub_data:
            print(f"  no data for sub-{sub}, skipping.")
            continue
        sub_dict = sub_data[f"sub-{sub}"]

        for crit_key, crit_label, roi, _, neuron, summary_row in by_subject[sub]:
            if neuron not in sub_dict['normalised_neurons']:
                print(f"  WARN: {neuron} not in subject's neurons; skipping.")
                continue
            out_dir = os.path.join(OUT_DIR, crit_key, roi)
            os.makedirs(out_dir, exist_ok=True)
            safe_neuron = neuron.replace('/', '_').replace(' ', '_')
            out_path = os.path.join(out_dir, f"sub-{sub}_{safe_neuron}.png")
            plot_one_cell(neuron, sub, summary_row, sub_dict,
                          crit_label, out_path)
            n_total += 1

    print(f"\nSaved {n_total} polar plots under {OUT_DIR}")


if __name__ == '__main__':
    main()
