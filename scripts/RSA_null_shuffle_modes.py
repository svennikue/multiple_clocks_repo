#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare permutation-null options for the DSR RSA — mPFC only, split_halves_z.

We reproduce the ``per_cell_trials`` structure that the main RSA pipeline
hands to ``mc.analyse.rsa_perm_rdms.build_perm_data_rdms`` for the mPFC ROI,
then run three permutation-null schemes locally.  All three preserve
within-trial autocorrelation (they only permute what's assigned where);
they differ in *what structure they preserve vs. break* about the
cell↔config relationship:

    * ``shift``           — CURRENT.  Random circular shift per trial per
                            cell; no reassignment.  Preserves per-cell
                            per-config firing preferences.
    * ``config_swap``     — permute each cell's trials across its configs
                            (no time shift).  Breaks per-cell per-config
                            preferences; keeps within-trial signal exactly.
    * ``shift_and_swap``  — both.  Strongest null.

For each mode we fit OLS with the same combo used in the main pipeline
(state + location + bttn_curr + dsr_fmri) and report the null distribution
of β_dsr_fmri.  The DSR-alone combo is also reported.

The heavy per-cell trial dictionary is cached at
``<RUN_DIR>/null_shuffle_diagnostic/per_cell_trials_mPFC.pkl`` so re-runs
are fast.

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mc
from mc.analyse.my_RSA import (
    compute_crosscorr,
    evaluate_model_vec,
)

# ── Settings ─────────────────────────────────────────────────────────
RUN_TAG            = '2026-07-30_13-32-23'
TARGET_ROI         = 'mPFC'
N_CONFIGS          = 8
N_CONDS_PER_CONFIG = 12
N_BINS_PER_TRIAL   = 360
N_PERMS            = 500
SEED               = 42

DATA_DIR = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                '/ephys_humans/derivatives')
RUN_DIR  = DATA_DIR / 'group/DSR_RSA_simple_ROI' / RUN_TAG
OUT_DIR  = RUN_DIR / 'null_shuffle_diagnostic'
OUT_DIR.mkdir(exist_ok=True)

CONFIGS = ['3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3',
           '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6']

COMBOS = {
    'dsr_only':     ['dsr_fmri'],
    'ctrl_dsrFULL': ['state', 'location', 'bttn_curr', 'dsr_fmri'],
}


# ── Load ROI table and pick mPFC cells ───────────────────────────────
roi_tbl = pd.read_csv(DATA_DIR / 'neurons_with_ROI_labels.csv')
for ax in ('x', 'y', 'z'):
    if f'MNI_{ax}_final' in roi_tbl.columns:
        roi_tbl[f'MNI_{ax}'] = roi_tbl[f'MNI_{ax}_final']
roi_tbl = roi_tbl.set_index(['subject', 'cell idx'])

with open(DATA_DIR / 'all_sessions_dsrRSA_grouping_summary.json') as f:
    config_summary = json.load(f)
SUBJECTS = list(config_summary.keys())


def parse_neuron_label(label):
    try:
        sub_str, rest = label.split('_', 1)
        cell_idx_str = rest.split('-', 1)[0]
        return int(sub_str), int(cell_idx_str)
    except (ValueError, IndexError):
        return None, None


def in_target_roi(label):
    sub, cell = parse_neuron_label(label)
    if sub is None:
        return False
    try:
        roi = roi_tbl.loc[(sub, cell), 'alt_final_roi']
    except KeyError:
        return False
    if isinstance(roi, pd.Series):
        roi = roi.dropna().iloc[0] if roi.notna().any() else None
    return roi == TARGET_ROI


# ── Reproduce per_cell_trial_chunks for mPFC (matches main script) ───
cache_path = OUT_DIR / f'per_cell_trials_{TARGET_ROI}.pkl'
if cache_path.exists():
    print(f"Loading cached per_cell_trials from {cache_path}")
    with open(cache_path, 'rb') as f:
        per_cell_trials_list = pickle.load(f)
    print(f"  {len(per_cell_trials_list)} cells cached")
else:
    print("Building per_cell_trials from raw subject data (one-time)...")
    per_cell_trial_chunks = {}

    grouping_logs = {}
    for sub_str in SUBJECTS:
        gpath = DATA_DIR / f's{sub_str}' / 'dsr_avg' / f's{sub_str}_dsr_grouping_log_two_runs.json'
        if not gpath.exists():
            grouping_logs[sub_str] = {}
            continue
        with open(gpath) as f:
            _g = json.load(f)
        grouping_logs[sub_str] = {c['config']: c for c in _g.get('configs', [])}

    for sub_str in SUBJECTS:
        print(f"  loading sub-{sub_str}...", flush=True)
        data_dict = mc.analyse.helpers_human_cells.load_norm_data(
            str(DATA_DIR), [sub_str], res_data=False,
        )
        beh = data_dict[f"sub-{sub_str}"]['beh'].copy().reset_index(drop=True)
        beh['config'] = list(zip(
            beh['loc_A'].astype(int), beh['loc_B'].astype(int),
            beh['loc_C'].astype(int), beh['loc_D'].astype(int),
        ))
        beh['grid_no']    = beh['grid_no'].astype(int)
        beh['config_str'] = beh['config'].apply(
            lambda t: f'{t[0]}-{t[1]}-{t[2]}-{t[3]}')
        curr_neurons = data_dict[f"sub-{sub_str}"]['normalised_neurons']

        sub_glog = grouping_logs.get(sub_str, {})
        for conf in CONFIGS:
            cfg_entry = sub_glog.get(conf)
            if cfg_entry is None:
                continue
            run1_blocks = cfg_entry['run1_blocks']
            run2_blocks = cfg_entry['run2_blocks']

            idx_all  = (beh['config_str'] == conf) & (beh['correct'] == 1)
            idx_run1 = idx_all & beh['grid_no'].isin(run1_blocks)
            idx_run2 = idx_all & beh['grid_no'].isin(run2_blocks)

            _all_orig_idx = beh.index[idx_all].to_numpy()
            _r1_orig_idx  = set(beh.index[idx_run1].to_numpy().tolist())
            _r2_orig_idx  = set(beh.index[idx_run2].to_numpy().tolist())
            row_run1_mask = np.array(
                [i in _r1_orig_idx for i in _all_orig_idx], dtype=bool)
            row_run2_mask = np.array(
                [i in _r2_orig_idx for i in _all_orig_idx], dtype=bool)

            for n_lab in curr_neurons:
                if not in_target_roi(n_lab):
                    continue
                conf_neurons_all = curr_neurons[n_lab][idx_all].to_numpy()
                if conf_neurons_all.shape[0] == 0:
                    continue
                if n_lab not in per_cell_trial_chunks:
                    per_cell_trial_chunks[n_lab] = {
                        'cell_id': n_lab, 'per_config': {}}
                per_cell_trial_chunks[n_lab]['per_config'][conf] = {
                    'trials_all': conf_neurons_all.astype(np.float64),
                    'run1_mask':  row_run1_mask.copy(),
                    'run2_mask':  row_run2_mask.copy(),
                }
        del data_dict

    per_cell_trials_list = list(per_cell_trial_chunks.values())
    with open(cache_path, 'wb') as f:
        pickle.dump(per_cell_trials_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached {len(per_cell_trials_list)} cells → {cache_path}")


# ── Pre-index cell data for fast per-perm assembly ───────────────────
# For each cell, keep a flat trial array + per-config slice offsets and
# per-config run masks. Slots not present for a cell become NaN in the
# population matrix.
CELLS = []
for cell in per_cell_trials_list:
    per_cfg = cell['per_config']
    slices = []
    per_cfg_run1 = []
    per_cfg_run2 = []
    flat_trials = []
    offset = 0
    for cfg in CONFIGS:
        entry = per_cfg.get(cfg)
        if entry is None or entry['trials_all'].shape[0] == 0:
            slices.append(None)
            per_cfg_run1.append(None)
            per_cfg_run2.append(None)
        else:
            n = entry['trials_all'].shape[0]
            slices.append((offset, offset + n))
            per_cfg_run1.append(entry['run1_mask'].copy())
            per_cfg_run2.append(entry['run2_mask'].copy())
            flat_trials.append(entry['trials_all'])
            offset += n
    if flat_trials:
        flat = np.vstack(flat_trials)
    else:
        flat = np.zeros((0, N_BINS_PER_TRIAL))
    CELLS.append({
        'flat':    flat,              # (n_trials_total, 360)
        'slices':  slices,            # list of (lo, hi) or None per config
        'r1':      per_cfg_run1,      # list of bool arrays or None per config
        'r2':      per_cfg_run2,
    })

n_cells = len(CELLS)
bin_per_cond = N_BINS_PER_TRIAL // N_CONDS_PER_CONFIG
n_half = N_CONFIGS * N_CONDS_PER_CONFIG
print(f"\nIndexed {n_cells} mPFC cells "
      f"(total trials across cells+configs: "
      f"{sum(c['flat'].shape[0] for c in CELLS)})")


def _avg_then_downsample(trials):
    if trials.shape[0] == 0:
        return np.full(N_CONDS_PER_CONFIG, np.nan)
    avg = np.nanmean(trials, axis=0)
    return avg.reshape(N_CONDS_PER_CONFIG, bin_per_cond).mean(axis=1)


def build_pop_split(mode, rng):
    """Return pop_split (n_cells, 2*n_half). Slots without data → NaN."""
    pop = np.full((n_cells, 2 * n_half), np.nan, dtype=np.float64)
    for ci, cell in enumerate(CELLS):
        flat = cell['flat']
        n_total = flat.shape[0]
        if n_total == 0:
            continue

        # Permute the mapping from configs → trial rows in `flat`.
        if mode in ('config_swap', 'shift_and_swap'):
            perm = rng.permutation(n_total)
        else:
            perm = np.arange(n_total)

        for cfg_i, sl in enumerate(cell['slices']):
            if sl is None:
                continue
            lo, hi = sl
            trial_rows_in_flat = perm[lo:hi]
            trials = flat[trial_rows_in_flat]

            # Circular shift per trial (independent).
            if mode in ('shift', 'shift_and_swap'):
                n_tr = trials.shape[0]
                shifts = rng.integers(0, N_BINS_PER_TRIAL, size=n_tr)
                idx = (np.arange(N_BINS_PER_TRIAL) - shifts[:, None]) % N_BINS_PER_TRIAL
                trials = np.take_along_axis(trials, idx, axis=1)

            r1_mask = cell['r1'][cfg_i]
            r2_mask = cell['r2'][cfg_i]
            avg_r1 = _avg_then_downsample(trials[r1_mask])
            avg_r2 = _avg_then_downsample(trials[r2_mask])

            slot_lo = cfg_i * N_CONDS_PER_CONFIG
            slot_hi = slot_lo + N_CONDS_PER_CONFIG
            pop[ci, slot_lo:slot_hi]                 = avg_r1
            pop[ci, n_half + slot_lo:n_half + slot_hi] = avg_r2
    return pop


def z_per_neuron(pop):
    mu = np.nanmean(pop, axis=1, keepdims=True)
    sd = np.nanstd(pop, axis=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return (pop - mu) / sd


def rdm_split_from_pop(pop_z):
    return compute_crosscorr(
        pop_z.T, plotting=False, include_diagonal=False,
        no_tasks=N_CONFIGS,
    )[0]


# ── Load the empirical + model RDMs ──────────────────────────────────
npz = np.load(RUN_DIR / f'rdms/rdms_{TARGET_ROI}.npz', allow_pickle=True)
Y_emp = npz['data__split_halves_z'].astype(float)

def _model(name):
    return npz[f'model__split_halves__{name}'].astype(float)

MODELS = {}
for feat_names in COMBOS.values():
    for f in feat_names:
        if f not in MODELS:
            MODELS[f] = _model(f)


# ── Run all three shuffle modes ──────────────────────────────────────
MODES = ['shift', 'config_swap', 'shift_and_swap']
mode_results = {m: {} for m in MODES}

for mode in MODES:
    print(f"\n=== mode = {mode} ({N_PERMS} perms) ===", flush=True)
    rng = np.random.default_rng(SEED)
    Y_perms = np.zeros((N_PERMS, Y_emp.shape[0]), dtype=np.float64)
    log_every = max(1, N_PERMS // 10)
    for p in range(N_PERMS):
        pop = build_pop_split(mode, rng)
        pop_z = z_per_neuron(pop)
        Y_perms[p] = rdm_split_from_pop(pop_z)
        if (p + 1) % log_every == 0:
            print(f"  {p + 1}/{N_PERMS}", flush=True)

    # Z-score each perm's RDM (mirror the 'split_halves_z' variant used in main).
    Y_perms_z = (Y_perms - np.nanmean(Y_perms, axis=1, keepdims=True))
    Y_perms_z /= np.where(np.nanstd(Y_perms, axis=1, keepdims=True) > 0,
                          np.nanstd(Y_perms, axis=1, keepdims=True), 1.0)

    for combo_name, feat_names in COMBOS.items():
        X = np.column_stack([MODELS[f] for f in feat_names])
        _, BETA_PERMS, _ = evaluate_model_vec(X, Y_perms_z)
        _, beta_emp, _   = evaluate_model_vec(X, Y_emp[None, :])
        beta_emp = beta_emp[0]

        combo_row = {}
        for k, f in enumerate(feat_names):
            null = BETA_PERMS[:, k]
            combo_row[f] = {
                'null_mean':   float(np.nanmean(null)),
                'null_std':    float(np.nanstd(null)),
                'null_median': float(np.nanmedian(null)),
                'empirical':   float(beta_emp[k]),
                'p_right':     float(np.nanmean(null >= beta_emp[k])),
                'p_centered':  float(
                    np.nanmean(null - np.nanmean(null)
                               >= beta_emp[k] - np.nanmean(null))),
            }
        mode_results[mode][combo_name] = combo_row


# ── Summary table ─────────────────────────────────────────────────────
rows = []
for mode in MODES:
    for combo_name, combo_row in mode_results[mode].items():
        for feat, stats in combo_row.items():
            rows.append({'mode': mode, 'combo': combo_name, 'feature': feat,
                         **stats})
summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / f'null_shuffle_summary_{TARGET_ROI}.csv', index=False)
print(f"\nSaved: {OUT_DIR / f'null_shuffle_summary_{TARGET_ROI}.csv'}")

print("\n" + "="*90)
print(f"NULL DISTRIBUTIONS PER MODE — {TARGET_ROI}, split_halves_z, "
      f"{N_PERMS} perms")
print("="*90)
print(summary.round(4).to_string(index=False))

print("\n── DSR summary across modes ──")
dsr = summary[summary['feature'] == 'dsr_fmri'].copy()
print(dsr[['mode', 'combo', 'null_mean', 'null_std',
           'empirical', 'p_right', 'p_centered']]
      .round(4).to_string(index=False))
