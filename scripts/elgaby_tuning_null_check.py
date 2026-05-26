#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Null calibration for el-gaby state / phase tuning.

The state-tuning test in `mc/analyse/elgaby_tuning.py` has built-in
selection bias (preferred state is picked by argmax-of-mean-z, then we
t-test the z-scores at that state). The empirical false-positive rate
under the null is therefore higher than the nominal p<0.05.

This script estimates that rate. For each correct trial we independently
circularly-shift the 360-bin trace by a uniformly-random offset; this
preserves within-trial autocorrelation but destroys state/phase
alignment. We then re-run the same tuning tests and compare the empirical
"tuned fractions" to the real-data fractions.

If real-data fractions are clearly above the null fractions, the tuning
flag is informative. If they sit close to the null, the test is
dominated by bias.

Outputs a small CSV + bar plot per ROI under
  derivatives/group/elgaby_tuning/<run_tag>-null/

The reference (real-data) run is identified by RELOAD_TUNING_RUN (or
auto-detected as the most recent run).

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import mc
from mc.analyse import elgaby_tuning


# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
TUNING_BASE = os.path.join(DATA_DIR, 'group', 'elgaby_tuning')

# Which previous tuning run to compare against. None = use the most
# recently modified subfolder of TUNING_BASE.
RELOAD_TUNING_RUN = None

N_BINS_PER_TRIAL = 360
N_STATES = 4
N_PHASES = 3

# Number of independent null draws to average over.  Each draw circular-
# shifts every correct trial by a random offset and re-computes tuning.
N_NULL_DRAWS = 5

# Joblib parallelism over neurons.
N_JOBS = -1

# Match the reference run's subject selection if available, otherwise:
SUBJECTS_TO_RUN = 'match_reference'   # 'match_reference' | 'all' | list

QUICK_TEST = False
if QUICK_TEST:
    MAX_SUBJECTS = 2
    MAX_NEURONS_PER_SUBJECT = 5
    N_NULL_DRAWS = 2
else:
    MAX_SUBJECTS = None
    MAX_NEURONS_PER_SUBJECT = None


# ── Resolve reference run ────────────────────────────────────────────
def find_latest_tuning_run(base):
    candidates = [d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d))
                  and not d.endswith('-null')]
    if not candidates:
        raise FileNotFoundError(f"No tuning runs found under {base}")
    candidates.sort(key=lambda d: os.path.getmtime(os.path.join(base, d)))
    return candidates[-1]


if RELOAD_TUNING_RUN is None:
    RELOAD_TUNING_RUN = find_latest_tuning_run(TUNING_BASE)
REF_DIR = os.path.join(TUNING_BASE, RELOAD_TUNING_RUN)
print(f"Reference tuning run: {REF_DIR}")

ref_config_path = os.path.join(REF_DIR, 'config.json')
with open(ref_config_path) as f:
    ref_config = json.load(f)
ref_roi_path = os.path.join(REF_DIR, 'roi_tuning_summary.csv')
ref_roi_df = pd.read_csv(ref_roi_path)

# Subject set.
if SUBJECTS_TO_RUN == 'match_reference':
    SUBJECTS = list(ref_config['subjects'])
elif SUBJECTS_TO_RUN == 'all':
    SUBJECTS = [f'{i:02}' for i in range(1, 64)]
elif isinstance(SUBJECTS_TO_RUN, list):
    SUBJECTS = list(SUBJECTS_TO_RUN)
else:
    raise ValueError(f"Unknown SUBJECTS_TO_RUN: {SUBJECTS_TO_RUN!r}")

# ROI table machinery (same as Script 1).
ROI_TABLE_PATH = ref_config['roi_table_path']
ROI_LABEL_COLUMN = ref_config['roi_label_column']
TARGET_ROIS = ref_config.get('target_rois', None)


# Output folder: alongside the reference, with a `-null` suffix.
OUT_DIR = os.path.join(TUNING_BASE, f'{RELOAD_TUNING_RUN}-null')
os.makedirs(OUT_DIR, exist_ok=True)
run_config = {
    'reference_run':     RELOAD_TUNING_RUN,
    'reference_dir':     REF_DIR,
    'out_dir':           OUT_DIR,
    'timestamp':         datetime.now().isoformat(timespec='seconds'),
    'n_null_draws':      N_NULL_DRAWS,
    'subjects':          SUBJECTS,
    'quick_test':        QUICK_TEST,
}
with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump(run_config, f, indent=2)


# ── ROI lookup ───────────────────────────────────────────────────────
def parse_neuron_label(label):
    try:
        sub_str, rest = label.split('_', 1)
        cell_idx_str = rest.split('-', 1)[0]
        return int(sub_str), int(cell_idx_str)
    except (ValueError, IndexError):
        return None, None


def load_roi_table(path, roi_col):
    df = pd.read_csv(path)
    df['subject']  = df['subject'].astype(int)
    df['cell idx'] = df['cell idx'].astype(int)
    return df.set_index(['subject', 'cell idx'])


ROI_TABLE = load_roi_table(ROI_TABLE_PATH, ROI_LABEL_COLUMN)


def get_neuron_roi(label):
    sub, cell_idx = parse_neuron_label(label)
    if sub is None:
        return None
    try:
        roi = ROI_TABLE.loc[(sub, cell_idx), ROI_LABEL_COLUMN]
    except KeyError:
        return None
    if isinstance(roi, pd.Series):
        roi = roi.dropna().iloc[0] if roi.notna().any() else None
    return None if (roi is None or pd.isna(roi)) else str(roi)


# ── Null tuning per neuron ───────────────────────────────────────────
def null_characterise_one_neuron(neuron_label, roi_name, neuron_df,
                                  beh, configs, n_draws, seed):
    """Run state + phase tuning per config under a circular-shift null.

    For each draw, every correct trial is independently rolled by a
    uniform random offset before the tuning tests run. We return the
    *fraction of draws* in which each (neuron, config) was tuned, so a
    well-calibrated test would give ~0.05 per (neuron, config).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cfg in configs:
        mask = (beh['config_str'] == cfg) & (beh['correct'] == 1)
        idx = beh.index[mask].to_numpy()
        n_trials = int(len(idx))
        if n_trials < 2:
            rows.append({
                'neuron':                  neuron_label,
                'roi':                     roi_name,
                'config':                  cfg,
                'n_trials':                n_trials,
                'null_state_tuned_rate':   np.nan,
                'null_phase_tuned_rate':   np.nan,
                'null_state_p_median':     np.nan,
                'null_phase_p_median':     np.nan,
            })
            continue

        trial_arr = neuron_df.iloc[idx].to_numpy()  # (n_trials, 360)
        state_hits, phase_hits = 0, 0
        state_ps, phase_ps = [], []
        for _ in range(n_draws):
            shifts = rng.integers(0, N_BINS_PER_TRIAL, size=n_trials)
            shifted = np.empty_like(trial_arr)
            for i, s in enumerate(shifts):
                shifted[i] = np.roll(trial_arr[i], s)

            st = elgaby_tuning.state_tuning(shifted)
            ph = elgaby_tuning.phase_tuning(shifted)
            if st['elgaby_state_tuned']:
                state_hits += 1
            if ph['elgaby_phase_tuned']:
                phase_hits += 1
            state_ps.append(st['elgaby_state_tuning_p'])
            phase_ps.append(ph['elgaby_phase_tuning_p'])

        rows.append({
            'neuron':                  neuron_label,
            'roi':                     roi_name,
            'config':                  cfg,
            'n_trials':                n_trials,
            'null_state_tuned_rate':   state_hits / n_draws,
            'null_phase_tuned_rate':   phase_hits / n_draws,
            'null_state_p_median':     float(np.nanmedian(state_ps)),
            'null_phase_p_median':     float(np.nanmedian(phase_ps)),
        })
    return rows


# ── Main loop ────────────────────────────────────────────────────────
def main():
    target_set = set(TARGET_ROIS) if TARGET_ROIS else None
    all_rows = []
    subjects_processed = 0
    t_overall = time.time()

    for sub_str in SUBJECTS:
        if MAX_SUBJECTS is not None and subjects_processed >= MAX_SUBJECTS:
            break

        print(f"\n========== Subject sub-{sub_str} ==========")
        t_sub = time.time()
        try:
            sub_data = mc.analyse.helpers_human_cells.load_norm_data(
                DATA_DIR, [sub_str],
            )
        except Exception as e:
            print(f"  failed to load sub-{sub_str}: {e}")
            continue
        if f"sub-{sub_str}" not in sub_data:
            continue

        sub_dict = sub_data[f"sub-{sub_str}"]
        beh = sub_dict['beh'].copy().reset_index(drop=True)
        beh['config_str'] = (
            beh['loc_A'].astype(int).astype(str) + '-' +
            beh['loc_B'].astype(int).astype(str) + '-' +
            beh['loc_C'].astype(int).astype(str) + '-' +
            beh['loc_D'].astype(int).astype(str)
        )
        configs = sorted(beh['config_str'].dropna().unique().tolist())
        neurons = sub_dict['normalised_neurons']

        neuron_labels = []
        for n_lab in neurons:
            roi = get_neuron_roi(n_lab)
            if roi is None:
                continue
            if target_set is not None and roi not in target_set:
                continue
            neuron_labels.append(n_lab)
        if MAX_NEURONS_PER_SUBJECT is not None:
            neuron_labels = neuron_labels[:MAX_NEURONS_PER_SUBJECT]
        if not neuron_labels:
            continue

        print(f"  {len(neuron_labels)} neurons, {len(configs)} configs, "
              f"{N_NULL_DRAWS} null draws each.")

        neuron_args = [
            (n_lab, get_neuron_roi(n_lab),
             neurons[n_lab].reset_index(drop=True))
            for n_lab in neuron_labels
        ]
        n_jobs_eff = 1 if len(neuron_args) == 1 else N_JOBS
        results = Parallel(n_jobs=n_jobs_eff, verbose=0)(
            delayed(null_characterise_one_neuron)(
                n_lab, roi, neuron_df, beh, configs,
                n_draws=N_NULL_DRAWS,
                seed=abs(hash((sub_str, n_lab, 'null'))) & 0xFFFFFFFF,
            )
            for n_lab, roi, neuron_df in neuron_args
        )

        for rows in results:
            for row in rows:
                row['subject'] = sub_str
            all_rows.extend(rows)

        subjects_processed += 1
        print(f"  done in {time.time() - t_sub:.1f}s "
              f"(total {time.time() - t_overall:.1f}s).")

    if not all_rows:
        print("Nothing to save.")
        return

    null_df = pd.DataFrame(all_rows)
    null_path = os.path.join(OUT_DIR, 'null_tuning_per_neuron_config.csv')
    null_df.to_csv(null_path, index=False)
    print(f"\nSaved {null_path}")

    # Per-ROI summary of null rates.
    roi_rows = []
    for roi, g in null_df.groupby('roi', sort=False):
        roi_rows.append({
            'roi':                          roi,
            'n_neuron_config_rows':         int(len(g)),
            'null_frac_state_tuned_per_cfg': float(g['null_state_tuned_rate'].mean()),
            'null_frac_phase_tuned_per_cfg': float(g['null_phase_tuned_rate'].mean()),
        })
    null_roi_df = pd.DataFrame(roi_rows).sort_values('roi').reset_index(drop=True)
    null_roi_path = os.path.join(OUT_DIR, 'null_roi_summary.csv')
    null_roi_df.to_csv(null_roi_path, index=False)
    print(f"Saved {null_roi_path}")

    # Comparison plot: per-ROI bars of real vs null tuned fractions.
    merged = ref_roi_df.merge(null_roi_df, on='roi', how='inner')
    merged = merged.sort_values('frac_state_tuned_per_config', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.4
    x = np.arange(len(merged))

    ax = axes[0]
    ax.bar(x - width/2, merged['frac_state_tuned_per_config'], width,
           label='real data')
    ax.bar(x + width/2, merged['null_frac_state_tuned_per_cfg'], width,
           label=f'null ({N_NULL_DRAWS} shifts)', color='gray')
    ax.axhline(0.05, ls='--', color='k', lw=0.8, label='nominal 0.05')
    ax.set_xticks(x)
    ax.set_xticklabels(merged['roi'], rotation=45, ha='right')
    ax.set_ylabel('fraction (neuron, config) state-tuned (p<0.05)')
    ax.set_title('State tuning vs circular-shift null')
    ax.legend()

    ax = axes[1]
    ax.bar(x - width/2, merged['frac_phase_tuned_per_config'], width,
           label='real data')
    ax.bar(x + width/2, merged['null_frac_phase_tuned_per_cfg'], width,
           label=f'null ({N_NULL_DRAWS} shifts)', color='gray')
    ax.axhline(0.05, ls='--', color='k', lw=0.8, label='nominal 0.05')
    ax.set_xticks(x)
    ax.set_xticklabels(merged['roi'], rotation=45, ha='right')
    ax.set_ylabel('fraction (neuron, config) phase-tuned (p<0.05)')
    ax.set_title('Phase tuning vs circular-shift null')
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'real_vs_null_fractions.png'), dpi=150)
    plt.close(fig)
    print(f"Saved plot to {OUT_DIR}/real_vs_null_fractions.png")

    print("\nReal vs null:")
    print(merged[['roi',
                  'frac_state_tuned_per_config', 'null_frac_state_tuned_per_cfg',
                  'frac_phase_tuned_per_config', 'null_frac_phase_tuned_per_cfg']]
          .to_string(index=False))


if __name__ == '__main__':
    main()
