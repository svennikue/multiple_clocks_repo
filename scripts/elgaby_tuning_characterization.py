#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
El-Gaby tuning characterization for human single-cell ephys (DSR paradigm).

For each (subject, neuron, config):
  - el-gaby state tuning  (paper-faithful: peak FR per state per trial,
                           row-wise z-score, t-test of preferred-state
                           z-scores against 0)
  - el-gaby phase tuning  (same logic but on the 3 within-state phase bins)
  - preferred state, preferred phase (argmax of mean per-phase activity)
  - n correct trials

Per-neuron aggregates across configs and per-ROI summaries are saved so
that the follow-up encoding script can read these tables and skip the
re-fitting on every run.

This is Script 1 of a 3-script pipeline:
  1. elgaby_tuning_characterization.py   <-- you are here
  2. encoding_analysis_elgaby.py         (reads this script's CSV)
  3. elgaby_lag_analysis.py              (reads script 2's coefs)

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
OUT_BASE = os.path.join(DATA_DIR, 'group', 'elgaby_tuning')

N_BINS_PER_TRIAL = 360
N_STATES = 4
N_PHASES = 3

# Tuning significance threshold for the boolean flags written to the CSV.
# The raw p-values are always saved so any other threshold can be applied
# downstream without re-running this script.
P_THRESHOLD = 0.05

# Joblib parallelism over neurons.
N_JOBS = -1

# ROI labelling (same convention as encoding_analysis_simple.py).
ROI_TABLE_PATH = os.path.join(DATA_DIR, 'neurons_with_final_roi_labels.csv')
ROI_LABEL_COLUMN = 'alt_final_roi'
TARGET_ROIS = None   # e.g. ['EC', 'HC_anterior', 'ACC']; None = all ROIs

# Subject selection:
#   'all'      -> sub-01 .. sub-63
#   'dsr_subs' -> subjects flagged in the DSR JSON summary
#   list       -> explicit subject ids, e.g. ['02', '31']
SUBJECTS_TO_RUN = 'all'

# Test / diagnostic mode.
QUICK_TEST = False
if QUICK_TEST:
    MAX_SUBJECTS            = 2
    MAX_NEURONS_PER_SUBJECT = 5
else:
    MAX_SUBJECTS            = None
    MAX_NEURONS_PER_SUBJECT = None


# ── Resolve subjects + run folder ────────────────────────────────────
if SUBJECTS_TO_RUN == 'dsr_subs':
    with open(os.path.join(DATA_DIR,
                           'all_sessions_dsrRSA_grouping_summary.json')) as f:
        SUBJECTS = list(json.load(f).keys())
elif SUBJECTS_TO_RUN == 'all':
    SUBJECTS = [f'{i:02}' for i in range(1, 64)]
elif isinstance(SUBJECTS_TO_RUN, list):
    SUBJECTS = list(SUBJECTS_TO_RUN)
else:
    raise ValueError(f"Unknown SUBJECTS_TO_RUN: {SUBJECTS_TO_RUN!r}")

RUN_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUT_DIR = os.path.join(OUT_BASE, RUN_TAG)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Run output: {OUT_DIR}")

run_config = {
    'run_tag':           RUN_TAG,
    'timestamp':         datetime.now().isoformat(timespec='seconds'),
    'data_dir':          DATA_DIR,
    'out_dir':           OUT_DIR,
    'roi_table_path':    ROI_TABLE_PATH,
    'roi_label_column':  ROI_LABEL_COLUMN,
    'target_rois':       TARGET_ROIS,
    'subjects':          SUBJECTS,
    'p_threshold':       P_THRESHOLD,
    'n_bins_per_trial':  N_BINS_PER_TRIAL,
    'n_states':          N_STATES,
    'n_phases':          N_PHASES,
    'max_subjects':      MAX_SUBJECTS,
    'max_neurons':       MAX_NEURONS_PER_SUBJECT,
    'quick_test':        QUICK_TEST,
}
with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump(run_config, f, indent=2)


# ── ROI lookup (same machinery as encoding_analysis_simple.py) ───────
def parse_neuron_label(label):
    """'01_07-07-chan120-EC' -> (1, 7); returns (None, None) on bad labels."""
    try:
        sub_str, rest = label.split('_', 1)
        cell_idx_str = rest.split('-', 1)[0]
        return int(sub_str), int(cell_idx_str)
    except (ValueError, IndexError):
        return None, None


def load_roi_table(path, roi_col):
    df = pd.read_csv(path)
    needed = ['subject', 'cell idx', roi_col,
              'MNI_x', 'MNI_y', 'MNI_z', 'electrode label']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"ROI table {path} is missing columns: {missing}")
    df = df.copy()
    df['subject']  = df['subject'].astype(int)
    df['cell idx'] = df['cell idx'].astype(int)
    return df.set_index(['subject', 'cell idx'])


ROI_TABLE = load_roi_table(ROI_TABLE_PATH, ROI_LABEL_COLUMN)
print(f"Loaded ROI table with {len(ROI_TABLE)} cells "
      f"({ROI_TABLE[ROI_LABEL_COLUMN].nunique()} distinct ROIs).")


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


def get_neuron_mni(label):
    sub, cell_idx = parse_neuron_label(label)
    if sub is None:
        return (np.nan, np.nan, np.nan)
    try:
        row = ROI_TABLE.loc[(sub, cell_idx)]
    except KeyError:
        return (np.nan, np.nan, np.nan)
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return (float(row['MNI_x']), float(row['MNI_y']), float(row['MNI_z']))


# ── Per-neuron computation ───────────────────────────────────────────
def characterise_one_neuron(neuron_label, roi_name, neuron_df,
                            beh, configs):
    """Run state + phase tuning per config for one neuron.

    Returns a list of dicts (one row per (neuron, config)).
    """
    rows = []
    for cfg in configs:
        mask = (beh['config_str'] == cfg) & (beh['correct'] == 1)
        idx = beh.index[mask].to_numpy()
        n_trials = int(len(idx))

        if n_trials < 2:
            rows.append(_blank_row(neuron_label, roi_name, cfg, n_trials))
            continue

        trial_arr = neuron_df.iloc[idx].to_numpy()  # (n_trials, 360)

        st = elgaby_tuning.state_tuning(
            trial_arr,
            n_bins_per_trial=N_BINS_PER_TRIAL, n_states=N_STATES,
        )
        ph = elgaby_tuning.phase_tuning(
            trial_arr,
            n_bins_per_trial=N_BINS_PER_TRIAL,
            n_states=N_STATES, n_phases=N_PHASES,
        )

        mni_x, mni_y, mni_z = get_neuron_mni(neuron_label)
        rows.append({
            'neuron':                  neuron_label,
            'roi':                     roi_name,
            'config':                  cfg,
            'n_trials':                n_trials,
            'elgaby_pref_state':       st['elgaby_pref_state'],
            'elgaby_state_tuning_t':   st['elgaby_state_tuning_t'],
            'elgaby_state_tuning_p':   st['elgaby_state_tuning_p'],
            'elgaby_state_tuned':      st['elgaby_state_tuned'],
            'elgaby_pref_phase':       ph['elgaby_pref_phase'],
            'elgaby_phase_tuning_t':   ph['elgaby_phase_tuning_t'],
            'elgaby_phase_tuning_p':   ph['elgaby_phase_tuning_p'],
            'elgaby_phase_tuned':      ph['elgaby_phase_tuned'],
            'mean_act_phase_0':        ph['mean_act_per_phase'][0],
            'mean_act_phase_1':        ph['mean_act_per_phase'][1],
            'mean_act_phase_2':        ph['mean_act_per_phase'][2],
            'MNI_x':                   mni_x,
            'MNI_y':                   mni_y,
            'MNI_z':                   mni_z,
        })
    return rows


def _blank_row(neuron_label, roi_name, cfg, n_trials):
    mni_x, mni_y, mni_z = get_neuron_mni(neuron_label)
    return {
        'neuron':                  neuron_label,
        'roi':                     roi_name,
        'config':                  cfg,
        'n_trials':                int(n_trials),
        'elgaby_pref_state':       -1,
        'elgaby_state_tuning_t':   np.nan,
        'elgaby_state_tuning_p':   np.nan,
        'elgaby_state_tuned':      False,
        'elgaby_pref_phase':       -1,
        'elgaby_phase_tuning_t':   np.nan,
        'elgaby_phase_tuning_p':   np.nan,
        'elgaby_phase_tuned':      False,
        'mean_act_phase_0':        np.nan,
        'mean_act_phase_1':        np.nan,
        'mean_act_phase_2':        np.nan,
        'MNI_x':                   mni_x,
        'MNI_y':                   mni_y,
        'MNI_z':                   mni_z,
    }


# ── Per-neuron aggregates across configs ─────────────────────────────
def per_neuron_summary(per_neuron_config_df):
    """Roll up the long table to one row per neuron."""
    rows = []
    for neuron, g in per_neuron_config_df.groupby('neuron', sort=False):
        n_configs = int(len(g))
        n_configs_state_tuned = int(g['elgaby_state_tuned'].sum())
        n_configs_phase_tuned = int(g['elgaby_phase_tuned'].sum())

        # Consistency = fraction of configs sharing the modal pref_state/phase.
        pref_states = g['elgaby_pref_state'].to_numpy()
        valid_state = pref_states[pref_states >= 0]
        state_consistency = (
            (valid_state == np.bincount(valid_state).argmax()).mean()
            if valid_state.size else np.nan
        )
        pref_phases = g['elgaby_pref_phase'].to_numpy()
        valid_phase = pref_phases[pref_phases >= 0]
        phase_consistency = (
            (valid_phase == np.bincount(valid_phase).argmax()).mean()
            if valid_phase.size else np.nan
        )

        rows.append({
            'neuron':                 neuron,
            'roi':                    g['roi'].iloc[0],
            'MNI_x':                  g['MNI_x'].iloc[0],
            'MNI_y':                  g['MNI_y'].iloc[0],
            'MNI_z':                  g['MNI_z'].iloc[0],
            'n_configs':              n_configs,
            'n_configs_state_tuned':  n_configs_state_tuned,
            'frac_configs_state_tuned': (n_configs_state_tuned / n_configs
                                         if n_configs else np.nan),
            'n_configs_phase_tuned':  n_configs_phase_tuned,
            'frac_configs_phase_tuned': (n_configs_phase_tuned / n_configs
                                         if n_configs else np.nan),
            'state_consistency':      state_consistency,
            'phase_consistency':      phase_consistency,
            'any_config_state_tuned': bool(n_configs_state_tuned > 0),
            'all_configs_state_tuned': bool(n_configs_state_tuned == n_configs),
            'any_config_phase_tuned': bool(n_configs_phase_tuned > 0),
            'all_configs_phase_tuned': bool(n_configs_phase_tuned == n_configs),
        })
    return pd.DataFrame(rows)


# ── Per-ROI summary ──────────────────────────────────────────────────
def per_roi_summary(per_neuron_df, per_neuron_config_df):
    """One row per ROI with counts and fractions of tuned neurons."""
    rows = []
    for roi, g in per_neuron_df.groupby('roi', sort=False):
        cfg_g = per_neuron_config_df[per_neuron_config_df['roi'] == roi]
        rows.append({
            'roi':                          roi,
            'n_neurons':                    int(len(g)),
            'n_neuron_config_rows':         int(len(cfg_g)),
            'n_any_state_tuned':            int(g['any_config_state_tuned'].sum()),
            'frac_any_state_tuned':         float(g['any_config_state_tuned'].mean()),
            'n_all_state_tuned':            int(g['all_configs_state_tuned'].sum()),
            'frac_all_state_tuned':         float(g['all_configs_state_tuned'].mean()),
            'n_any_phase_tuned':            int(g['any_config_phase_tuned'].sum()),
            'frac_any_phase_tuned':         float(g['any_config_phase_tuned'].mean()),
            'frac_state_tuned_per_config':  float(cfg_g['elgaby_state_tuned'].mean()),
            'frac_phase_tuned_per_config':  float(cfg_g['elgaby_phase_tuned'].mean()),
            'mean_state_consistency':       float(g['state_consistency'].mean()),
            'mean_phase_consistency':       float(g['phase_consistency'].mean()),
        })
    return pd.DataFrame(rows).sort_values('roi').reset_index(drop=True)


# ── Plots ────────────────────────────────────────────────────────────
def plot_roi_fractions(roi_summary, out_dir):
    """Bar plot: fraction of neurons state-tuned (any config) per ROI,
    side-by-side with phase-tuned fraction."""
    df = roi_summary.sort_values('frac_any_state_tuned', ascending=False)
    x = np.arange(len(df))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(df)), 5))
    ax.bar(x - width/2, df['frac_any_state_tuned'], width,
           label='state-tuned in ≥1 config')
    ax.bar(x + width/2, df['frac_any_phase_tuned'], width,
           label='phase-tuned in ≥1 config', color='tab:orange')
    ax.axhline(P_THRESHOLD, ls='--', color='k', lw=0.8,
               label=f'chance ({P_THRESHOLD:g})')
    ax.set_xticks(x)
    ax.set_xticklabels(df['roi'], rotation=45, ha='right')
    ax.set_ylabel('fraction of neurons')
    ax.set_title('El-Gaby tuning: fraction of neurons tuned per ROI')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'roi_fraction_tuned.png'), dpi=150)
    plt.close(fig)


def plot_n_configs_state_tuned(per_neuron_df, out_dir):
    """Per ROI: histogram of how many configs each neuron is state-tuned in."""
    rois = sorted(per_neuron_df['roi'].dropna().unique().tolist())
    n_cols = 3
    n_rows = int(np.ceil(len(rois) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows),
                             squeeze=False)
    max_cfg = int(per_neuron_df['n_configs'].max() or 1)
    bins = np.arange(max_cfg + 2) - 0.5
    for ax, roi in zip(axes.ravel(), rois):
        g = per_neuron_df[per_neuron_df['roi'] == roi]
        ax.hist(g['n_configs_state_tuned'], bins=bins,
                edgecolor='k', alpha=0.85)
        ax.set_title(f'{roi}  (n={len(g)})')
        ax.set_xlabel('# configs state-tuned')
        ax.set_ylabel('# neurons')
    # Hide unused axes.
    for ax in axes.ravel()[len(rois):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'n_configs_state_tuned_per_roi.png'),
                dpi=150)
    plt.close(fig)


def plot_pref_distributions(per_neuron_config_df, out_dir):
    """Per ROI: distribution of pref_state and pref_phase across tuned (n,cfg)."""
    df = per_neuron_config_df
    rois = sorted(df['roi'].dropna().unique().tolist())

    fig, axes = plt.subplots(2, 1, figsize=(max(6, 0.5 * len(rois)), 8))
    width = 0.18

    # pref state distribution
    ax = axes[0]
    tuned = df[df['elgaby_state_tuned']]
    for s in range(N_STATES):
        fracs = []
        for roi in rois:
            g = tuned[tuned['roi'] == roi]
            fracs.append((g['elgaby_pref_state'] == s).mean()
                         if len(g) else np.nan)
        x = np.arange(len(rois)) + (s - (N_STATES - 1) / 2) * width
        ax.bar(x, fracs, width, label=f'state {s}')
    ax.set_xticks(np.arange(len(rois)))
    ax.set_xticklabels(rois, rotation=45, ha='right')
    ax.set_ylabel('frac of state-tuned (neuron, config)')
    ax.set_title('Preferred state distribution per ROI '
                 '(state-tuned (neuron, config) only)')
    ax.legend(ncol=N_STATES)

    # pref phase distribution
    ax = axes[1]
    tuned = df[df['elgaby_phase_tuned']]
    for p in range(N_PHASES):
        fracs = []
        for roi in rois:
            g = tuned[tuned['roi'] == roi]
            fracs.append((g['elgaby_pref_phase'] == p).mean()
                         if len(g) else np.nan)
        x = np.arange(len(rois)) + (p - (N_PHASES - 1) / 2) * width * 1.5
        ax.bar(x, fracs, width * 1.5, label=f'phase {p}')
    ax.set_xticks(np.arange(len(rois)))
    ax.set_xticklabels(rois, rotation=45, ha='right')
    ax.set_ylabel('frac of phase-tuned (neuron, config)')
    ax.set_title('Preferred phase distribution per ROI '
                 '(phase-tuned (neuron, config) only)')
    ax.legend(ncol=N_PHASES)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pref_state_phase_distributions.png'),
                dpi=150)
    plt.close(fig)


# ── Main loop ────────────────────────────────────────────────────────
def main():
    target_set = set(TARGET_ROIS) if TARGET_ROIS is not None else None
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
            print(f"  no data for sub-{sub_str}; skipping.")
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

        # Filter by ROI.
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
            print(f"  no neurons in target ROIs — skipping.")
            continue

        print(f"  {len(neuron_labels)} neurons across "
              f"{len(configs)} configs.")

        neuron_args = [
            (n_lab, get_neuron_roi(n_lab),
             neurons[n_lab].reset_index(drop=True))
            for n_lab in neuron_labels
        ]

        n_jobs_eff = 1 if len(neuron_args) == 1 else N_JOBS
        # results = Parallel(n_jobs=n_jobs_eff, verbose=0)(
        #     delayed(characterise_one_neuron)(
        #         n_lab, roi, neuron_df, beh, configs,
        #     )
        #     for n_lab, roi, neuron_df in neuron_args
        # )
        for n_lab, roi, neuron_df in neuron_args:
            results = characterise_one_neuron(n_lab, roi, neuron_df, beh, configs)

        for rows in results:
            for row in rows:
                row['subject'] = sub_str
            all_rows.extend(rows)

        subjects_processed += 1
        print(f"  done in {time.time() - t_sub:.1f}s "
              f"(total {time.time() - t_overall:.1f}s).")

    if not all_rows:
        print("No rows generated; nothing to save.")
        return

    per_neuron_config_df = pd.DataFrame(all_rows)
    per_neuron_df = per_neuron_summary(per_neuron_config_df)
    roi_df = per_roi_summary(per_neuron_df, per_neuron_config_df)

    per_neuron_config_path = os.path.join(OUT_DIR, 'tuning_per_neuron_config.csv')
    per_neuron_path = os.path.join(OUT_DIR, 'tuning_per_neuron.csv')
    roi_path = os.path.join(OUT_DIR, 'roi_tuning_summary.csv')
    per_neuron_config_df.to_csv(per_neuron_config_path, index=False)
    per_neuron_df.to_csv(per_neuron_path, index=False)
    roi_df.to_csv(roi_path, index=False)
    print(f"\nSaved:\n  {per_neuron_config_path}\n  {per_neuron_path}\n  {roi_path}")

    plot_roi_fractions(roi_df, OUT_DIR)
    plot_n_configs_state_tuned(per_neuron_df, OUT_DIR)
    plot_pref_distributions(per_neuron_config_df, OUT_DIR)
    print(f"Saved plots to {OUT_DIR}")

    # Quick console summary so we can spot-check without opening files.
    print("\nROI summary:")
    cols_show = ['roi', 'n_neurons',
                 'frac_any_state_tuned', 'frac_any_phase_tuned',
                 'frac_state_tuned_per_config',
                 'frac_phase_tuned_per_config']
    print(roi_df[cols_show].to_string(index=False))


if __name__ == '__main__':
    main()
