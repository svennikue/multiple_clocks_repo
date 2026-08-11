#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic: what do ACC neurons truly encode?

Loads ACC neurons EXACTLY the same way as ``scripts/RSA_DSR_ROIs_simple.py``
(same SUBJECTS, same ROI table, same per-config / per-half pooling), then
builds two data RDMs (z-scored neurons) and compares them against a battery of
model RDMs that decompose the "DSR" construct into its named ingredients.

Two data RDMs (both z-scored across neurons):
    - across_runs       : split-halves variant (concat half-1 + half-2 across
                          configs, then take the cross-half block — exactly
                          the ``data_RDM_z`` of the existing script)
    - between_tasks     : run-averaged per config, then take only between-
                          task-config cells (the ``data_RDM_across_z`` of the
                          existing script)

Model RDMs (each built in BOTH variants):
    - dsr_fmri                : mode-path → integer location IDs → rolled per
                                cond → Hamming RDM. The "fmri-style" DSR.
    - dsr_old                 : ``predictions.model_DSR`` clo_model with
                                no_phase_neurons=3 → cosine RDM. The existing
                                von-Mises clock-ring used in the real pipeline.
    - dsr_old_categorical     : from-scratch clock-ring with categorical phase
                                + every-active-bin-counted-as-peak.
    - dsr_ideal               : 9 loc × 12 lag one-hot grid (no phase) → cosine
                                RDM. The "literal location-at-lag" interpretation.
    - dsr_ideal_phases        : 9 loc × 3 phase × 12 lag graded grid → cosine.
    - phase_vM                : ``predictions.model_DSR`` phase_og_matrix
                                (von Mises tuning) → cosine RDM.
    - phase_categorical       : sharp 3-bin phase blocks per state → cosine.
    - midnight_vM             : ``predictions.model_DSR`` midn_m (location ×
                                von Mises phase) → cosine RDM.
    - midnight_categorical    : location × categorical phase → cosine RDM.

For each pair the script reports r in two masking modes:
    - FULL        : every between-config cell of the RDM (or all cross-half
                    cells for the across_runs variant).
    - WITHIN_PHASE: only same-phase pairs (cond_i % N_PHASES == cond_j % N_PHASES).

No permutations. Single ROI = ACC. Output: per-pair correlations as CSV
plus a couple of heatmap visualisations (model-model RDM correlations,
model-data RDM correlations).

@author: Svenja Kuechenhoff
"""

import os
import json
from datetime import date
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import mc
import mc.simulation.predictions as predictions
import mc.analyse.my_RSA as my_RSA


# ── Settings (mirror RSA_DSR_ROIs_simple.py for the ACC ROI) ────────────
DATA_DIR     = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/'
                'ephys_humans/derivatives')
ROI_TABLE_PATH = os.path.join(DATA_DIR, 'neurons_with_ROI_labels.csv')
ROI_LABEL_COLUMN = 'alt_final_roi'
TARGET_ROI = 'mPFC'

configs = [
    '3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3',
    '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6',
]

N_CONFIGS             = len(configs)
N_CONDS_PER_CONFIG    = 20
LEN_STANDARDISED_PATH = 12
N_PHASES              = 2
N_STATES              = 4
N_LOCATIONS           = 9
N_RAW_BINS            = 360
BINS_PER_STATE        = N_RAW_BINS // N_STATES               # 90
BINLEN_PER_COND       = N_RAW_BINS // N_CONDS_PER_CONFIG     # 30

OUT_DIR = os.path.join(
    DATA_DIR, 'group',
    f'diagnostic_ACC_dsr_encoding_{date.today().isoformat()}')
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Saving diagnostic outputs to: {OUT_DIR}")


# ── ROI table & SUBJECTS list ────────────────────────────────────────────
def parse_neuron_label(label):
    try:
        sub_str, rest = label.split('_', 1)
        cell_idx_str = rest.split('-', 1)[0]
        return int(sub_str), int(cell_idx_str)
    except (ValueError, IndexError):
        return None, None


def _load_roi_table(path, roi_col):
    df = pd.read_csv(path)
    needed = ['subject', 'cell idx', roi_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"ROI table missing columns: {missing}")
    df = df.copy()
    df['subject']  = df['subject'].astype(int)
    df['cell idx'] = df['cell idx'].astype(int)
    return df.set_index(['subject', 'cell idx'])


ROI_TABLE = _load_roi_table(ROI_TABLE_PATH, ROI_LABEL_COLUMN)


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


def in_target_roi(label):
    return get_neuron_roi(label) == TARGET_ROI


with open(os.path.join(DATA_DIR, 'all_sessions_dsrRSA_grouping_summary.json'),
          'r') as f:
    config_summary = json.load(f)
SUBJECTS = list(config_summary.keys())


# ── Helpers — fixed downsample_mode and the model builders ──────────────
def downsample_mode(x, target_len):
    """Mode-downsample x to target_len, distributing input bins evenly so
    NO bins are dropped. ``slot i = x[(i*n)//target_len : ((i+1)*n)//target_len]``."""
    x = np.asarray(x, dtype=object)
    n = len(x)
    return np.array([
        Counter(x[(i * n) // target_len:((i + 1) * n) // target_len])
            .most_common(1)[0][0]
        for i in range(target_len)
    ], dtype=object)


def build_mode_path_dsr(mode_vec, n_conds_per_config, len_per_bin):
    """Rolled integer-ID trajectory: (12, n_conds*len_per_bin)."""
    base = downsample_mode(mode_vec, target_len=n_conds_per_config * len_per_bin)
    return np.stack([np.roll(base, -pos * len_per_bin)
                     for pos in range(n_conds_per_config)], axis=0)


def build_ideal_dsr(walked, n_conds=N_CONDS_PER_CONFIG,
                    len_per_lag=LEN_STANDARDISED_PATH, n_loc=N_LOCATIONS):
    """Per-cond (n_loc × n_conds) one-hot grid: row L, col k = 1 iff mode
    location at lag k from cond start is L. Flattens to (n_conds, n_loc*n_conds).
    """
    base = downsample_mode(walked, target_len=n_conds * len_per_lag).astype(int)
    out = np.zeros((n_conds, n_loc, n_conds), dtype=float)
    for cond in range(n_conds):
        rolled = np.roll(base, -cond * len_per_lag)
        for lag in range(n_conds):
            sub = rolled[lag * len_per_lag:(lag + 1) * len_per_lag]
            mode_loc = Counter(sub.tolist()).most_common(1)[0][0]
            out[cond, mode_loc - 1, lag] = 1
    return out.reshape(n_conds, n_loc * n_conds)


def build_ideal_dsr_with_phases(walked, n_conds=N_CONDS_PER_CONFIG,
                                len_per_cond_raw=BINLEN_PER_COND,
                                n_phases=N_PHASES, n_loc=N_LOCATIONS,
                                bins_per_state=BINS_PER_STATE):
    """Per-cond (n_loc × n_phases × n_conds) graded activation.
    Cell (L, P, lag): fraction of raw bins at cond (cond+lag)%n_conds spent at
    location L while in phase P."""
    walked = np.asarray(walked, dtype=int)
    out = np.zeros((n_conds, n_loc, n_phases, n_conds), dtype=float)
    bpp = bins_per_state // n_phases    # 30
    for cond in range(n_conds):
        for lag in range(n_conds):
            target = (cond + lag) % n_conds
            t_start = target * len_per_cond_raw
            t_end = t_start + len_per_cond_raw
            for t in range(t_start, t_end):
                loc = walked[t] - 1
                phase = (t % bins_per_state) // bpp
                out[cond, loc, phase, lag] += 1
        out[cond] /= len_per_cond_raw
    return out.reshape(n_conds, n_loc * n_phases * n_conds)


def make_categorical_phase(n_phases=N_PHASES,
                           step_per_state=BINS_PER_STATE,
                           n_states=N_STATES):
    bpp = step_per_state // n_phases
    mat = np.zeros((n_phases, step_per_state * n_states), dtype=float)
    for s in range(n_states):
        for p in range(n_phases):
            mat[p,
                s * step_per_state + p * bpp:
                s * step_per_state + (p + 1) * bpp] = 1
    return mat


def make_categorical_midnight(walked, phase_cat,
                              n_phases=N_PHASES, n_loc=N_LOCATIONS):
    n_t = len(walked)
    mat = np.zeros((n_loc * n_phases, n_t), dtype=float)
    for t in range(n_t):
        loc = int(walked[t]) - 1
        for p in range(n_phases):
            if phase_cat[p, t]:
                mat[loc * n_phases + p, t] = 1
    return mat


def make_categorical_state_phase(phase_cat, n_phases=N_PHASES,
                                 n_states=N_STATES,
                                 step_per_state=BINS_PER_STATE):
    n_t = phase_cat.shape[1]
    mat = np.zeros((n_states * n_phases, n_t), dtype=float)
    for s in range(n_states):
        for p in range(n_phases):
            t_start = s * step_per_state
            t_end = (s + 1) * step_per_state
            mat[s * n_phases + p, t_start:t_end] = phase_cat[p, t_start:t_end]
    return mat


def make_categorical_clo_model(walked, phase_cat, midn_cat,
                               n_phases=N_PHASES, n_loc=N_LOCATIONS,
                               n_states=N_STATES,
                               step_per_state=BINS_PER_STATE):
    n_t = len(walked)
    state_phase_ring = make_categorical_state_phase(phase_cat)
    n_midn = n_loc * n_phases
    n_sp = n_states * n_phases
    clo = np.zeros((n_midn * n_sp, n_t), dtype=float)
    for L_p in range(n_midn):
        peak_times = np.where(midn_cat[L_p] > 0)[0]
        for t0 in peak_times:
            shifted = np.roll(state_phase_ring, t0, axis=1)
            clo[L_p * n_sp:(L_p + 1) * n_sp] += shifted
    return clo


def _ds_360(M, N=N_CONDS_PER_CONFIG, binlen=BINLEN_PER_COND):
    """(n_features, 360) -> (n_features, N) by mean over equal bins."""
    return M.reshape(M.shape[0], N, binlen).mean(axis=2)


# ── Load subject data, aggregate ACC neurons per (conf, half) ───────────
print(f"\nLoading subject data for {len(SUBJECTS)} subjects...")
SUBJECT_DATA = {}
for sub_str in SUBJECTS:
    SUBJECT_DATA[sub_str] = mc.analyse.helpers_human_cells.load_norm_data(
        DATA_DIR, [sub_str])
print(f"  cached.")


acc_neurons = {c: {1: [], 2: []} for c in configs}        # split halves
acc_neurons_all = {c: [] for c in configs}                # run-averaged
locs = {c: {1: [], 2: []} for c in configs}
locs_all = {c: [] for c in configs}

for sub_str in SUBJECTS:
    data_dict = SUBJECT_DATA[sub_str]
    beh = data_dict[f"sub-{sub_str}"]['beh'].copy().reset_index(drop=True)
    beh['config'] = list(zip(
        beh['loc_A'].astype(int), beh['loc_B'].astype(int),
        beh['loc_C'].astype(int), beh['loc_D'].astype(int),
    ))
    beh['grid_no']    = beh['grid_no'].astype(int)
    beh['config_str'] = beh['config'].apply(
        lambda t: f'{t[0]}-{t[1]}-{t[2]}-{t[3]}')
    curr_neurons = data_dict[f"sub-{sub_str}"]['normalised_neurons']

    for conf in configs:
        idx = (beh['config_str'] == conf) & (beh['correct'] == 1)
        all_locs_conf = data_dict[f"sub-{sub_str}"]['locations'][idx].to_numpy()
        locs[conf][1].append(all_locs_conf[0:10, :])
        locs[conf][2].append(all_locs_conf[10:20, :])
        locs_all[conf].append(all_locs_conf)

        for n_lab in curr_neurons:
            if not in_target_roi(n_lab):
                continue
            conf_acc = curr_neurons[n_lab][idx].to_numpy()
            avg_all = np.nanmean(conf_acc, axis=0)
            ds_all = avg_all.reshape(N_CONDS_PER_CONFIG, BINLEN_PER_COND
                                     ).mean(axis=1)
            acc_neurons_all[conf].append(ds_all)
            for th, start in [(1, 0), (2, 10)]:
                avg = np.nanmean(conf_acc[start:start + 10, :], axis=0)
                ds = avg.reshape(N_CONDS_PER_CONFIG, BINLEN_PER_COND
                                 ).mean(axis=1)
                acc_neurons[conf][th].append(ds)

n_neurons = len(acc_neurons[configs[0]][1])
print(f"\nACC neurons collected: {n_neurons}")


# ── Mode-path per (conf, half) and (conf, all) ─────────────────────────
mode_locs = {c: {} for c in configs}
mode_locs_all = {}
for c in configs:
    locs_all_per_conf = locs_all[c]
    stacked_all = np.vstack(locs_all_per_conf)
    m_all = stats.mode(stacked_all, axis=0, keepdims=False, nan_policy='omit')
    mode_locs_all[c] = m_all.mode.astype(float)
    for th in (1, 2):
        stacked = np.vstack(locs[c][th])
        m = stats.mode(stacked, axis=0, keepdims=False, nan_policy='omit')
        mode_locs[c][th] = m.mode.astype(float)
print("Mode-path arrays built.")


# ── Build the data RDMs (z-scored), two variants ───────────────────────
# 1) across_runs / split-halves variant
rows = []
for th in (1, 2):
    for c in configs:
        rows.append(np.asarray(acc_neurons[c][th]))
mat = np.hstack(rows)                          # (n_neurons, 2*8*12)
mu = np.nanmean(mat, axis=1); sd = np.nanstd(mat, axis=1); sd[sd == 0] = 1
mat_z = (mat.T - mu) / sd
data_across_runs_z = my_RSA.compute_crosscorr(
    mat_z, plotting=False, include_diagonal=False,
    no_tasks=N_CONFIGS, model='data across runs z')[0]

# 2) between_tasks / run-averaged variant
rows_all = [np.asarray(acc_neurons_all[c]) for c in configs]
mat_all = np.hstack(rows_all)                   # (n_neurons, 8*12)
mu_all = np.nanmean(mat_all, axis=1); sd_all = np.nanstd(mat_all, axis=1)
sd_all[sd_all == 0] = 1
mat_all_z = (mat_all.T - mu_all) / sd_all
data_within_z, data_between_z, data_full_sq = my_RSA.compute_crosscorr_within(
    mat_all_z, plotting=False, include_diagonal=False,
    no_tasks=N_CONFIGS, model='data between tasks z',
    block_size=N_CONDS_PER_CONFIG)
data_between_tasks_z = np.asarray(data_between_z[0], dtype=float)

print(f"\nData RDM sizes:")
print(f"  across_runs (split halves) vector length:    {len(data_across_runs_z)}")
print(f"  between_tasks (run-averaged) vector length:  {len(data_between_tasks_z)}")


# ── Build per-config model activations, then concat / between-tasks ─────
# Hamming-route uses integer IDs; cosine-route uses one-hot / continuous.
def build_per_config_models(mode_locs_dict, mode_locs_all_dict):
    """Returns a nested dict per model:
        out[model][conf]['split_h1'] / ['split_h2'] / ['all']  -> (12, F) matrices
    Plus also pre-cached walked (0-indexed) trajectories.
    """
    out = {m: {c: {} for c in configs} for m in [
        'dsr_fmri', 'dsr_old', 'dsr_old_categorical',
        'dsr_ideal', 'dsr_ideal_phases',
        'phase_vM', 'phase_categorical',
        'midnight_vM', 'midnight_categorical',
    ]}

    phase_cat = make_categorical_phase()  # (3, 360), config-independent

    for c in configs:
        # ----- per task-half models -----
        for th, walked_vec in [(1, mode_locs_dict[c][1]),
                               (2, mode_locs_dict[c][2]),
                               ('all', mode_locs_all_dict[c])]:
            walked_int = np.asarray(walked_vec, dtype=int)
            walked_0idx = (walked_int - 1).tolist()

            # Hamming-route DSR (mode-path → rolled trajectory → Hamming)
            out['dsr_fmri'][c][th] = build_mode_path_dsr(
                walked_int, N_CONDS_PER_CONFIG, LEN_STANDARDISED_PATH
            ).astype(float)

            # Clean lag-indexed location-at-lag
            out['dsr_ideal'][c][th] = build_ideal_dsr(walked_int)

            # Clean lag-indexed (loc, phase, lag), graded
            out['dsr_ideal_phases'][c][th] = build_ideal_dsr_with_phases(walked_int)

            # Cosine-route via model_DSR
            loc_og, phase_og, _state_og, midn_m, dsr_m, _, _ = (
                predictions.model_DSR(locations=walked_0idx,
                                      no_phase_neurons=N_PHASES))
            out['dsr_old'][c][th]      = _ds_360(dsr_m).T
            out['phase_vM'][c][th]     = _ds_360(phase_og).T
            out['midnight_vM'][c][th]  = _ds_360(midn_m).T

            # Categorical clock-ring (built from scratch)
            midn_cat = make_categorical_midnight(walked_int, phase_cat)
            clo_cat  = make_categorical_clo_model(walked_int, phase_cat,
                                                  midn_cat)
            out['dsr_old_categorical'][c][th] = _ds_360(clo_cat).T

            # Categorical phase (config-independent, but per-half for symmetry)
            out['phase_categorical'][c][th] = _ds_360(phase_cat).T

            # Categorical midnight (loc × categorical phase)
            out['midnight_categorical'][c][th] = _ds_360(midn_cat).T

    return out


print("\nBuilding per-config model activations (this can take a moment)...")
PCM = build_per_config_models(mode_locs, mode_locs_all)
print("  done.")


# ── Stack per model, compute RDMs in both variants ─────────────────────
hamming_route = {'dsr_fmri'}

model_RDMs_across_runs = {}     # split halves cross-block, ~4560 elements
model_RDMs_between_tasks = {}   # between-task cells of run-averaged, ~4032

for m in PCM:
    # split-halves: concat half-1 of all configs, then half-2 of all configs
    stack = []
    for th in (1, 2):
        for c in configs:
            stack.append(PCM[m][c][th])
    M_split = np.vstack(stack)
    # between-tasks: just the run-averaged version (one half worth = 8*12 rows)
    M_all = np.vstack([PCM[m][c]['all'] for c in configs])

    if m in hamming_route:
        rdm_split = my_RSA.compute_hamming_distance(
            M_split, plotting=False, include_diagonal=False,
            model_name=m, no_tasks=N_CONFIGS)
        rdm_within, rdm_across, _full = my_RSA.compute_hamming_distance_within(
            M_all, plotting=False, include_diagonal=False,
            model_name=m, no_tasks=N_CONFIGS,
            block_size=N_CONDS_PER_CONFIG)
    else:
        rdm_split = my_RSA.compute_crosscorr(
            M_split, plotting=False, include_diagonal=False,
            no_tasks=N_CONFIGS, model=m)
        rdm_within, rdm_across, _full = my_RSA.compute_crosscorr_within(
            M_all, plotting=False, include_diagonal=False,
            no_tasks=N_CONFIGS, model=m, block_size=N_CONDS_PER_CONFIG)

    model_RDMs_across_runs[m] = np.asarray(rdm_split[0], dtype=float)
    model_RDMs_between_tasks[m] = np.asarray(rdm_across[0], dtype=float)

print("\nModel RDM sizes (across_runs / between_tasks):")
for m in PCM:
    print(f"  {m:<26s}  {len(model_RDMs_across_runs[m]):>6d}  "
          f"{len(model_RDMs_between_tasks[m]):>6d}")


# ── Phase masks for both variants ───────────────────────────────────────
def make_phase_masks_for(n_rows, n_configs, n_conds_per_config, n_phases,
                         which='across_runs'):
    """Return boolean mask of within-phase pairs for the given vector layout.

    which='across_runs'    -> mask aligned with compute_crosscorr's output
                              (upper-tri off-diag of cross-half N×N block).
    which='between_tasks'  -> mask aligned with compute_*_within's between-
                              block output (off-block of N×N).
    Phase per condition = (cond_pos_within_config) % n_phases.
    """
    n = n_configs * n_conds_per_config
    phase = np.tile(np.arange(n_conds_per_config) % n_phases, n_configs)
    ii, jj = np.triu_indices(n, k=1)
    same_phase = phase[ii] == phase[jj]
    if which == 'across_runs':
        return same_phase
    if which == 'between_tasks':
        between = (ii // n_conds_per_config) != (jj // n_conds_per_config)
        return same_phase[between]
    raise ValueError(which)


mask_across_runs_wp   = make_phase_masks_for(
    None, N_CONFIGS, N_CONDS_PER_CONFIG, N_PHASES, which='across_runs')
mask_between_tasks_wp = make_phase_masks_for(
    None, N_CONFIGS, N_CONDS_PER_CONFIG, N_PHASES, which='between_tasks')


# ── Correlations ───────────────────────────────────────────────────────
def _corr(a, b, mask=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if mask is not None:
        a, b = a[mask], b[mask]
    finite = np.isfinite(a) & np.isfinite(b)
    if (finite.sum() < 3 or np.nanstd(a[finite]) == 0
            or np.nanstd(b[finite]) == 0):
        return float('nan')
    return float(np.corrcoef(a[finite], b[finite])[0, 1])


model_order = ['dsr_fmri', 'dsr_old', 'dsr_old_categorical',
               'dsr_ideal', 'dsr_ideal_phases',
               'phase_vM', 'phase_categorical',
               'midnight_vM', 'midnight_categorical']

# Model × Model correlations (per variant, per masking mode)
mm_rows = []
for variant_name, mrdms, wp_mask in [
    ('across_runs',   model_RDMs_across_runs,   mask_across_runs_wp),
    ('between_tasks', model_RDMs_between_tasks, mask_between_tasks_wp),
]:
    for i, a in enumerate(model_order):
        for b in model_order[i:]:
            r_full = _corr(mrdms[a], mrdms[b])
            r_wp   = _corr(mrdms[a], mrdms[b], wp_mask)
            mm_rows.append({
                'variant':       variant_name,
                'model_a':       a,
                'model_b':       b,
                'r_full':        r_full,
                'r_within_phase': r_wp,
            })
mm_df = pd.DataFrame(mm_rows)
mm_df.to_csv(os.path.join(OUT_DIR, 'model_model_correlations.csv'), index=False)

# Model × Data correlations
md_rows = []
for variant_name, mrdms, data_vec, wp_mask in [
    ('across_runs',   model_RDMs_across_runs,   data_across_runs_z,
     mask_across_runs_wp),
    ('between_tasks', model_RDMs_between_tasks, data_between_tasks_z,
     mask_between_tasks_wp),
]:
    for m in model_order:
        md_rows.append({
            'variant':         variant_name,
            'model':           m,
            'r_full':          _corr(mrdms[m], data_vec),
            'r_within_phase':  _corr(mrdms[m], data_vec, wp_mask),
        })
md_df = pd.DataFrame(md_rows)
md_df.to_csv(os.path.join(OUT_DIR, 'model_data_correlations.csv'), index=False)


print("\n=== Model ↔ Data RDM correlations (ACC neurons) ===")
print(md_df.to_string(index=False, float_format=lambda x: f'{x:+.3f}'))


# ── Visualisations ──────────────────────────────────────────────────────
# 1) Model-model correlation heatmap, one panel per variant × masking mode.
def _mm_matrix(df_sub, models):
    M = np.full((len(models), len(models)), np.nan)
    idx = {m: i for i, m in enumerate(models)}
    for _, row in df_sub.iterrows():
        i, j = idx[row['model_a']], idx[row['model_b']]
        M[i, j] = row['r']
        M[j, i] = row['r']
    return M

fig, axes = plt.subplots(2, 2, figsize=(12, 11),
                         constrained_layout=True)
panel_specs = [
    ('across_runs',   'r_full',          'across_runs (split halves) — FULL'),
    ('across_runs',   'r_within_phase',  'across_runs — WITHIN_PHASE'),
    ('between_tasks', 'r_full',          'between_tasks (run-averaged) — FULL'),
    ('between_tasks', 'r_within_phase',  'between_tasks — WITHIN_PHASE'),
]
for ax, (variant, col, title) in zip(axes.flat, panel_specs):
    sub = mm_df[mm_df['variant'] == variant].copy()
    sub['r'] = sub[col]
    M = _mm_matrix(sub, model_order)
    im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(len(model_order)):
        for j in range(len(model_order)):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                        fontsize=6,
                        color='white' if abs(v) > 0.55 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
fig.suptitle(f'Model-model RDM correlations (ACC, n={n_neurons})',
             fontsize=11)
fig.savefig(os.path.join(OUT_DIR, '1_model_model_correlations.png'),
            dpi=150, bbox_inches='tight')
plt.show()


# 2) Model-data correlations as a horizontal bar grid:
#    rows = model, cols = (variant × mask).
fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(model_order) + 1.6))
cols = [
    ('across_runs',   'r_full',         'across_runs FULL'),
    ('across_runs',   'r_within_phase', 'across_runs WP'),
    ('between_tasks', 'r_full',         'between_tasks FULL'),
    ('between_tasks', 'r_within_phase', 'between_tasks WP'),
]
grid = np.full((len(model_order), len(cols)), np.nan)
for j, (v, c, _label) in enumerate(cols):
    sub = md_df[md_df['variant'] == v].set_index('model')
    for i, m in enumerate(model_order):
        grid[i, j] = sub.loc[m, c] if m in sub.index else np.nan
lim = float(np.nanmax(np.abs(grid)))
lim = max(lim, 0.05)
im = ax.imshow(grid, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
ax.set_xticks(range(len(cols)))
ax.set_xticklabels([c[2] for c in cols], rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(len(model_order)))
ax.set_yticklabels(model_order, fontsize=9)
ax.set_title(f'Model ↔ ACC-data RDM correlations  (n={n_neurons} neurons)',
             fontsize=11)
for i in range(len(model_order)):
    for j in range(len(cols)):
        v = grid[i, j]
        if np.isfinite(v):
            ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                    fontsize=8,
                    color='white' if abs(v) > 0.55 * lim else 'black')
plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, '2_model_data_correlations.png'),
            dpi=150, bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────────────────────────────────
# COMBINATION GLMs: is the DSR signal in ACC just phase in disguise?
#
# For each DSR variant, fit four GLMs to each data RDM and report the unique
# DSR beta (the partial coefficient after the other regressors absorb their
# share of variance):
#
#   1. dsr_*                    (single regressor — marginal r as t,beta,p)
#   2. dsr_* + phase_vM         (control for von-Mises phase)
#   3. dsr_* + state            (control for state)
#   4. dsr_* + state + phase_vM (control for both)
#
# A real "DSR-content" signal survives (2) and (4): the DSR beta stays
# positive and significant even after phase explains its share. A signal
# that was actually phase-in-disguise loses its beta when (2) or (4) is run.
# ─────────────────────────────────────────────────────────────────────────

# Need a 'state' model RDM as well (from compute_*_within so it's aligned).
state_per_config = {}
for c in configs:
    walked = mode_locs_all[c]
    walked_0idx = (np.asarray(walked, dtype=int) - 1).tolist()
    _loc_og, _phase_og, state_og, _midn, _dsr, _, _ = (
        predictions.model_DSR(locations=walked_0idx, no_phase_neurons=N_PHASES))
    state_per_config[c] = {'all': _ds_360(state_og).T,
                           1: _ds_360(state_og).T,
                           2: _ds_360(state_og).T}
state_split = np.vstack([state_per_config[c][th]
                         for th in (1, 2) for c in configs])
state_all = np.vstack([state_per_config[c]['all'] for c in configs])
state_rdm_split = my_RSA.compute_crosscorr(
    state_split, plotting=False, include_diagonal=False,
    no_tasks=N_CONFIGS, model='state')[0]
_w, state_rdm_between, _f = my_RSA.compute_crosscorr_within(
    state_all, plotting=False, include_diagonal=False,
    no_tasks=N_CONFIGS, model='state', block_size=N_CONDS_PER_CONFIG)
model_RDMs_across_runs['state']   = np.asarray(state_rdm_split, dtype=float)
model_RDMs_between_tasks['state'] = np.asarray(state_rdm_between[0], dtype=float)


def _safe_drop_nonfinite(stacked, y):
    """Replace any non-finite cell in stacked or y with 0, but only column-
    drop a regressor if it becomes constant after that. Returns (X, y, good_idx)."""
    X = np.asarray(stacked, dtype=float).copy()
    yy = np.asarray(y, dtype=float).copy()
    X[~np.isfinite(X)] = 0
    yy[~np.isfinite(yy)] = 0
    var_ok = np.nanvar(X, axis=0) > 1e-12
    return X[:, var_ok], yy, var_ok


def _evaluate_glm(stacked, data_vec, names):
    """Returns (t_arr, beta_arr, p_arr) with NaN slots for dropped regressors."""
    n = len(names)
    nan_arr = lambda: np.full(n, np.nan, dtype=float)
    X_sub, y_sub, good = _safe_drop_nonfinite(stacked, data_vec)
    if not good.any() or np.nanvar(y_sub) == 0:
        return nan_arr(), nan_arr(), nan_arr()
    try:
        t_s, b_s, p_s = my_RSA.evaluate_model(X_sub, y_sub)
    except Exception as exc:
        print(f"    GLM failed: {exc}")
        return nan_arr(), nan_arr(), nan_arr()
    t_full, b_full, p_full = nan_arr(), nan_arr(), nan_arr()
    t_full[good] = np.asarray(t_s, dtype=float).ravel()
    b_full[good] = np.asarray(b_s, dtype=float).ravel()
    p_full[good] = np.asarray(p_s, dtype=float).ravel()
    return t_full, b_full, p_full


combo_definitions = {
    'dsr_only':          lambda d: [d],
    '+phase_vM':         lambda d: [d, 'phase_vM'],
    '+state':            lambda d: [d, 'state'],
    '+state+phase_vM':   lambda d: [d, 'state', 'phase_vM'],
}
dsr_variants = ['dsr_fmri', 'dsr_old', 'dsr_old_categorical',
                'dsr_ideal', 'dsr_ideal_phases']

combo_rows = []
for variant_name, mrdms, data_vec, wp_mask in [
    ('across_runs',   model_RDMs_across_runs,   data_across_runs_z,
     mask_across_runs_wp),
    ('between_tasks', model_RDMs_between_tasks, data_between_tasks_z,
     mask_between_tasks_wp),
]:
    for dsr_name in dsr_variants:
        for combo_label, combo_builder in combo_definitions.items():
            regs = combo_builder(dsr_name)
            stacked = np.stack([mrdms[r] for r in regs], axis=1)
            for mask_name, mask in [('full', None),
                                    ('within_phase', wp_mask)]:
                if mask is None:
                    Xm, ym = stacked, data_vec
                else:
                    Xm, ym = stacked[mask], data_vec[mask]
                t, b, p = _evaluate_glm(Xm, ym, regs)
                for idx, reg in enumerate(regs):
                    combo_rows.append({
                        'variant':       variant_name,
                        'dsr_variant':   dsr_name,
                        'combo':         combo_label,
                        'mask':          mask_name,
                        'regressor':     reg,
                        't':             float(t[idx]),
                        'beta':          float(b[idx]),
                        'p':             float(p[idx]),
                    })

combo_df = pd.DataFrame(combo_rows)
combo_df.to_csv(os.path.join(OUT_DIR, 'combination_GLM_results.csv'),
                index=False)

# Print only the DSR-regressor rows for quick scanning.
print("\n=== Combination-GLM betas (DSR row only) ===")
print("Reports how the DSR beta changes as we partial out phase / state.")
print(f"  {'variant':<14s} {'dsr':<22s} {'combo':<18s} {'mask':<14s} "
      f"{'t':>8s} {'beta':>8s} {'p':>10s}")
for _, row in combo_df.iterrows():
    if row['regressor'].startswith('dsr_'):
        print(f"  {row['variant']:<14s} {row['dsr_variant']:<22s} "
              f"{row['combo']:<18s} {row['mask']:<14s} "
              f"{row['t']:>+8.2f} {row['beta']:>+8.3f} {row['p']:>+10.4f}")


# ─────────────────────────────────────────────────────────────────────────
# Phase-residualised DSR (alternative to within-phase masking).
#
# Within-phase masking removes phase variance from BOTH the model and the
# data — but it also reduces n_pairs and (as you noticed) collapses some
# regressors. A cleaner control is to keep ALL pairs but residualise both
# sides against phase first, then correlate the residuals.
# ─────────────────────────────────────────────────────────────────────────

def _residualise(y, X):
    """Return y - X · OLS_fit(y on X). Both numeric vectors."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y_clean = y[finite]
    X_clean = X[finite]
    # add intercept
    X_int = np.column_stack([np.ones(X_clean.shape[0]), X_clean])
    coef, _, _, _ = np.linalg.lstsq(X_int, y_clean, rcond=None)
    fit = X_int @ coef
    res = y.copy()
    res[finite] = y_clean - fit
    res[~finite] = np.nan
    return res


resid_rows = []
for variant_name, mrdms, data_vec in [
    ('across_runs',   model_RDMs_across_runs,   data_across_runs_z),
    ('between_tasks', model_RDMs_between_tasks, data_between_tasks_z),
]:
    phase_rdm = mrdms['phase_vM']
    data_resid = _residualise(data_vec, phase_rdm)
    for dsr_name in dsr_variants:
        dsr_marg  = _corr(mrdms[dsr_name], data_vec)
        dsr_resid = _residualise(mrdms[dsr_name], phase_rdm)
        r_after = _corr(dsr_resid, data_resid)
        resid_rows.append({
            'variant':       variant_name,
            'dsr_variant':   dsr_name,
            'r_marginal':    dsr_marg,
            'r_phase_partialled': r_after,
            'drop':          dsr_marg - r_after,
        })
resid_df = pd.DataFrame(resid_rows)
resid_df.to_csv(os.path.join(OUT_DIR, 'phase_partialled_correlations.csv'),
                index=False)

print("\n=== DSR ↔ ACC correlation, before vs after partialling out phase_vM ===")
print("If r_phase_partialled ≈ r_marginal, the DSR signal is NOT phase-driven.")
print("If r_phase_partialled << r_marginal, the DSR effect WAS phase in disguise.")
print(resid_df.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))


# ─────────────────────────────────────────────────────────────────────────
# Across-phase-only correlations
#
# Within-phase masking can kill the variance of any phase-organised model
# (phase becomes constant). The complementary check is to look ONLY at
# across-phase pairs and ask: does the DSR signal survive there?
# If a DSR model's correlation with ACC is the SAME within- and across-
# phase, it isn't just inheriting phase structure.
# ─────────────────────────────────────────────────────────────────────────

def _across_phase_mask(within_phase_mask):
    return ~np.asarray(within_phase_mask, dtype=bool)


across_rows = []
for variant_name, mrdms, data_vec, wp_mask in [
    ('across_runs',   model_RDMs_across_runs,   data_across_runs_z,
     mask_across_runs_wp),
    ('between_tasks', model_RDMs_between_tasks, data_between_tasks_z,
     mask_between_tasks_wp),
]:
    ap_mask = _across_phase_mask(wp_mask)
    for m in model_order:
        across_rows.append({
            'variant':       variant_name,
            'model':         m,
            'r_full':        _corr(mrdms[m], data_vec),
            'r_within_p':    _corr(mrdms[m], data_vec, wp_mask),
            'r_across_p':    _corr(mrdms[m], data_vec, ap_mask),
        })
across_df = pd.DataFrame(across_rows)
across_df.to_csv(os.path.join(OUT_DIR, 'model_data_across_phase.csv'),
                 index=False)

print("\n=== Model ↔ ACC correlations by phase relation ===")
print("Full / within-phase only / across-phase only.")
print(across_df.to_string(index=False, float_format=lambda x: f'{x:+.4f}'))


print(f"\nDone. All outputs saved to:\n  {OUT_DIR}")
