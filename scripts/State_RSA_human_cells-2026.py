#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:45:17 2026

@author: Svenja Kuchenhoff

using averages created in prep_human_cells_State_RSA-2026.py and computing a
ROI-wise state RSA.

the cells are in the following format:
    - Group configs into N_GROUPS=6 pseudo-configs (merging if >6 exist)
    - Assign grid_no blocks INTACT to run1 / run2 (never split a block)
    - Average correct-trial firing rates within each (group × run)


1. per ROI (enth, hipp, amyg, OFC, PCC, ACC, mixed)
2. collapsed across the brain
3. for temporal lobe (ent+hipp) vs. frontal lobe (ACC+OFC)

"""

import os
import sys
import json
import numpy as np
import mc
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')

# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR  = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_DIR   = os.path.join(DATA_DIR, 'group', 'State_RSA')
N_GROUPS  = 6
N_BINS    = 360
INCLUDE_DIAG = False
SUBJECTS  = [f'{i:02}' for i in range(1, 64)]
EXCLUDE   = []   # add session numbers to skip, e.g. [19, 23]
SUBJECTS  = [s for s in SUBJECTS if int(s) not in EXCLUDE]
states = ['A', 'B', 'C', 'D']
rois_of_interest = ['whole_brain', 'OFC', 'EC', 'ACC', 'HC', 'PCC', 'AMY', 'R-WHITE-MATTER', 'OCCIP']
models = ['state', 'feedback']

PLOT_FIGS = True

# ── Downsampling ───────────────────────────────────────────────────────
# Set LEN_STANDARDISED_PATH to control how many bins represent a single state.
# Full resolution: 90 bins per state × 4 states = 360 bins per config half.
# e.g. LEN_STANDARDISED_PATH=3  → 3 bins per state → 12 bins per half-config.
LEN_STANDARDISED_PATH = 3    # bins per state after downsampling (must divide 90)

# ── Permutation settings ───────────────────────────────────────────────
RUN_PERMUTATIONS = True    # set False to skip permutation testing entirely
N_PERMS          = 500     # number of permutations

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers (same as DSR script) ──────────────────────────────────────

def downsample_mean(x, target_len):
    """Average bins of x down to target_len bins."""
    x = np.asarray(x)
    n = len(x)
    edges = np.linspace(0, n, target_len + 1, dtype=int)
    out = []
    for i in range(target_len):
        chunk = x[edges[i]:edges[i+1]]
        out.append(np.nan if len(chunk) == 0 else np.nanmean(chunk))
    return np.array(out)


def _scalar(arr):
    """Safely extract a Python float from a size-1 array or scalar."""
    return float(np.asarray(arr, dtype=float).ravel()[0])


# ── Load all sessions ──────────────────────────────────────────────────
rows = []
for sub in SUBJECTS:
    SUB_DIR = f"{DATA_DIR}/s{sub}/state_avg"
    npy_path = os.path.join(SUB_DIR, f's{sub}_neural_avg.npy')
    meta_path = os.path.join(SUB_DIR, f's{sub}_neuron_meta.json')
    if not os.path.exists(npy_path):
        print(f"  Skipping s{sub}: file not found ({npy_path})")
        continue
    sesh_neurons = np.load(npy_path)
    with open(meta_path, 'r') as file:
        neuron_details = json.load(file)

    for n_idx, n in enumerate(sesh_neurons):
        rows.append({
            "session":        sub,
            "neuron_idx":     n_idx,
            "neuron_label":   neuron_details["neuron_names"][n_idx],
            "roi":            neuron_details["cell_labels"][n_idx],
            "electrode_label":neuron_details["electrode_labels"][n_idx],
            "neuron_data":    n,   # shape (N_GROUPS, 2, N_BINS)
        })

neurons = pd.DataFrame(rows)
print(f"Loaded {len(neurons)} neuron-rows from {neurons['session'].nunique()} sessions.")
print(f"  Unique neurons: {neurons['neuron_label'].nunique()}")


# ── Build model matrices ───────────────────────────────────────────────
# After downsampling: LEN_STANDARDISED_PATH bins per state,
# so N_STATES * LEN_STANDARDISED_PATH bins per config half.
N_STATES       = len(states)
BINS_PER_HALF  = N_STATES * LEN_STANDARDISED_PATH   # bins per config half after DS
TOTAL_BINS     = BINS_PER_HALF * N_GROUPS            # rows in the half-matrix

print(f"\nBuilding model matrices...")
print(f"  LEN_STANDARDISED_PATH={LEN_STANDARDISED_PATH} bins/state  "
      f"→ {BINS_PER_HALF} bins/half-config × {N_GROUPS} groups = {TOTAL_BINS} rows")

# One half (N_GROUPS × BINS_PER_HALF rows, N_STATES cols)
state_one_half    = np.zeros((TOTAL_BINS, N_STATES))
feedback_one_half = np.zeros((TOTAL_BINS, N_STATES))

for g in range(N_GROUPS):
    offset = g * BINS_PER_HALF
    for s_i, s in enumerate(states):
        s_start = offset + s_i * LEN_STANDARDISED_PATH
        s_end   = offset + (s_i + 1) * LEN_STANDARDISED_PATH
        state_one_half[s_start:s_end, s_i] = 1
        if s == 'A':
            # feedback: first bin of state A only
            feedback_one_half[s_start, s_i] = 1

# Stack both halves for compute_crosscorr (which splits them internally)
state_full    = np.vstack((state_one_half,    state_one_half))
feedback_full = np.vstack((feedback_one_half, feedback_one_half))

print(f"  state_full shape: {state_full.shape}")
print(f"  feedback_full shape: {feedback_full.shape}")

state_RDM    = mc.analyse.my_RSA.compute_crosscorr(
    state_full,    plotting=PLOT_FIGS, include_diagonal=INCLUDE_DIAG, no_tasks=N_GROUPS)
feedback_RDM = mc.analyse.my_RSA.compute_crosscorr(
    feedback_full, plotting=PLOT_FIGS, include_diagonal=INCLUDE_DIAG, no_tasks=N_GROUPS)

print(f"  state_RDM vector length:    {len(state_RDM[0])}")
print(f"  feedback_RDM vector length: {len(feedback_RDM[0])}")

# Patch feedback NaNs (bins where only state A has signal → undefined elsewhere → set to 1)
nan_mask_feedback = np.isnan(feedback_RDM[0])
feedback_RDM[0][nan_mask_feedback] = 1
print(f"  Patched {nan_mask_feedback.sum()} feedback NaN entries to 1.")

model_RDMs = {
    'state':    state_RDM,
    'feedback': feedback_RDM,
}


# ── Build neural matrix for a neuron subset ────────────────────────────

def build_neuron_matrix(neuron_subset: pd.DataFrame):
    """
    Returns (matrix, n_excluded).
    matrix: (2 * N_GROUPS * BINS_PER_HALF, n_valid_neurons)
    Downsamples from N_BINS → BINS_PER_HALF per half-config.
    """
    uniq = neuron_subset['neuron_label'].unique()
    n_neurons = len(uniq)
    print(f"    build_neuron_matrix: {n_neurons} unique neurons, "
          f"downsampling {N_BINS}→{BINS_PER_HALF} bins/half-config")

    th1 = np.zeros((TOTAL_BINS, n_neurons))
    th2 = np.zeros((TOTAL_BINS, n_neurons))

    for n_idx, n_label in enumerate(uniq):
        rows_n = neuron_subset[neuron_subset['neuron_label'] == n_label]
        n_data = rows_n['neuron_data'].to_numpy()[0]   # (N_GROUPS, 2, N_BINS)
        for g in range(N_GROUPS):
            g_offset = g * BINS_PER_HALF
            raw_r1 = n_data[g, 0, :]   # (N_BINS,)
            raw_r2 = n_data[g, 1, :]
            th1[g_offset:g_offset + BINS_PER_HALF, n_idx] = downsample_mean(raw_r1, BINS_PER_HALF)
            th2[g_offset:g_offset + BINS_PER_HALF, n_idx] = downsample_mean(raw_r2, BINS_PER_HALF)

    valid_cols = ~(np.isnan(th1).any(axis=0) | np.isnan(th2).any(axis=0))
    n_excluded = int(np.sum(~valid_cols))
    if n_excluded:
        print(f"    Excluding {n_excluded} neurons with NaN.")
    th1 = th1[:, valid_cols]
    th2 = th2[:, valid_cols]
    return np.concatenate([th1, th2], axis=0), n_excluded


# ── RSA ────────────────────────────────────────────────────────────────

def run_rsa(data_mat: np.ndarray, plot: bool = False) -> dict:
    """
    Compute crosscorr data RDM, then regress model RDMs against it.
    Returns dict with 'data_rdm', 'unique' (per model), 'combined'.
    unique[m] = (t, beta, p)  — plain Python floats
    combined  = {'t': array, 'beta': array, 'p': array}
    """
    data_rdm = mc.analyse.my_RSA.compute_crosscorr(
        data_mat, plotting=plot, include_diagonal=INCLUDE_DIAG, no_tasks=N_GROUPS)

    unique = {}
    for m in models:
        res = mc.analyse.my_RSA.evaluate_model(model_RDMs[m][0], data_rdm[0])
        unique[m] = (_scalar(res[0]), _scalar(res[1]), _scalar(res[2]))

    stacked = np.stack([model_RDMs[m][0] for m in models], axis=1)
    res_c   = mc.analyse.my_RSA.evaluate_model(stacked, data_rdm[0])
    combined = {
        't':    np.asarray(res_c[0], dtype=float).ravel(),
        'beta': np.asarray(res_c[1], dtype=float).ravel(),
        'p':    np.asarray(res_c[2], dtype=float).ravel(),
    }
    return {'data_rdm': data_rdm, 'unique': unique, 'combined': combined}


print("\nRunning RSA...")

# whole brain
wb_mat, wb_excl = build_neuron_matrix(neurons)
print(f"  whole_brain matrix shape: {wb_mat.shape}, excluded: {wb_excl}")
wb_res = run_rsa(wb_mat, plot=PLOT_FIGS)
wb_res['n_neurons'] = len(neurons['neuron_label'].unique()) - wb_excl
roi_results = {'whole_brain': wb_res}
print(f"  whole_brain (n={wb_res['n_neurons']}) — "
      + ', '.join(f"{m}: t={wb_res['unique'][m][0]:.2f} b={wb_res['unique'][m][1]:.3f} p={wb_res['unique'][m][2]:.3f}"
                  for m in models))

for roi in rois_of_interest:
    if roi == 'whole_brain':
        continue
    subset = neurons[neurons['roi'] == roi]
    n_raw = len(subset['neuron_label'].unique())
    if n_raw == 0:
        print(f"  {roi}: no neurons, skipping.")
        continue
    print(f"  {roi}: {n_raw} neurons found.")
    roi_mat, roi_excl = build_neuron_matrix(subset)
    n_valid = n_raw - roi_excl
    if n_valid == 0:
        print(f"  {roi}: no valid neurons after NaN exclusion, skipping.")
        continue
    res = run_rsa(roi_mat, plot=PLOT_FIGS)
    res['n_neurons'] = n_valid
    roi_results[roi] = res
    print(f"  {roi} (n={n_valid}) — "
          + ', '.join(f"{m}: t={res['unique'][m][0]:.2f} b={res['unique'][m][1]:.3f} p={res['unique'][m][2]:.3f}"
                      for m in models))

present_rois = [r for r in rois_of_interest if r in roi_results]
print(f"\nPresent ROIs: {present_rois}")


# ── Permutation testing ────────────────────────────────────────────────
#
# Strategy: permute MODEL condition labels within each config block,
# covering all ROIs in one pass (same as DSR script).
# n_conditions = N_GROUPS * BINS_PER_HALF  (half-matrix size)

if RUN_PERMUTATIONS:
    print(f"\nRunning {N_PERMS} permutations...")
    print(f"  Strategy: permute condition labels in MODEL RDMs (covers all ROIs).")

    n_conditions = N_GROUPS * BINS_PER_HALF
    print(f"  n_conditions (half-matrix rows): {n_conditions}")

    rng = np.random.default_rng(seed=42)
    perm_full_indices = np.zeros((N_PERMS, n_conditions), dtype=int)
    for perm_i in range(N_PERMS):
        idx = np.arange(n_conditions)
        for g in range(N_GROUPS):
            blk = slice(g * BINS_PER_HALF, (g + 1) * BINS_PER_HALF)
            idx[blk] = idx[blk][rng.permutation(BINS_PER_HALF)]
        perm_full_indices[perm_i] = idx
    print(f"  Pre-computed permutation indices shape: {perm_full_indices.shape}")

    # Rebuild square model RDMs for cheap row/col reindexing
    k_flag = 0 if INCLUDE_DIAG else 1

    def _vec_to_square(vec, n):
        mat = np.full((n, n), np.nan)
        k = 0 if INCLUDE_DIAG else 1
        ii, jj = np.triu_indices(n, k=k)
        mat[ii, jj] = vec
        mat[jj, ii] = vec
        return mat

    print("  Rebuilding square model RDMs...")
    model_RDMs_2d = {}
    for m in models:
        model_RDMs_2d[m] = _vec_to_square(model_RDMs[m][0], n_conditions)
        print(f"    {m}: shape {model_RDMs_2d[m].shape}, "
              f"NaNs: {int(np.sum(np.isnan(model_RDMs_2d[m])))}")

    def _permute_model_rdm_vec(model_2d, perm_idx):
        perm_2d = model_2d[np.ix_(perm_idx, perm_idx)]
        ii, jj  = np.triu_indices(n_conditions, k=k_flag)
        return perm_2d[ii, jj].copy()

    def _run_one_model_perm(perm_i, perm_idx):
        perm_vecs = {m: _permute_model_rdm_vec(model_RDMs_2d[m], perm_idx)
                     for m in models}

        bu = {}   # roi -> {model -> beta float}
        bc = {}   # roi -> array(n_models,)
        for roi in present_rois:
            if roi not in _roi_data_rdms:
                continue
            data_vec = _roi_data_rdms[roi]

            bu[roi] = {}
            for m in models:
                res = mc.analyse.my_RSA.evaluate_model(perm_vecs[m], data_vec)
                bu[roi][m] = float(np.asarray(res[1], dtype=float).ravel()[0])

            stk   = np.stack([perm_vecs[m] for m in models], axis=1)
            res_c = mc.analyse.my_RSA.evaluate_model(stk, data_vec)
            bc[roi] = np.asarray(res_c[1], dtype=float).ravel()

        return perm_i, bu, bc

    # Store observed data RDM vectors
    _roi_data_rdms = {}
    for roi in present_rois:
        vec = roi_results[roi]['data_rdm'][0].copy().astype(float)
        _roi_data_rdms[roi] = vec
        print(f"  Data RDM stored for {roi}: length {len(vec)}, "
              f"NaNs: {int(np.sum(np.isnan(vec)))}")

    null_unique   = {roi: {m: [] for m in models} for roi in present_rois}
    null_combined = {roi: {m: [] for m in models} for roi in present_rois}

    print(f"\n  Starting permutation loop...")
    for perm_i in range(N_PERMS):
        _, bu, bc = _run_one_model_perm(perm_i, perm_full_indices[perm_i])
        for roi in present_rois:
            if roi not in bu:
                continue
            for m in models:
                null_unique[roi][m].append(bu[roi][m])
            for m_i, m in enumerate(models):
                null_combined[roi][m].append(float(bc[roi][m_i]))
        if (perm_i + 1) % 50 == 0:
            print(f"    {perm_i + 1}/{N_PERMS} done")

    print(f"  All {N_PERMS} permutations done.")

    # ── Permutation p-values ───────────────────────────────────────────
    perm_p_unique   = {roi: {} for roi in present_rois}
    perm_p_combined = {roi: {} for roi in present_rois}

    print("\n  Permutation p-values (one-tailed, beta, obs >= null):")
    for roi in present_rois:
        if roi not in _roi_data_rdms:
            continue
        for m in models:
            null_u = np.array(null_unique[roi][m])
            obs_u  = roi_results[roi]['unique'][m][1]
            p_u    = float(np.mean(null_u >= obs_u))
            perm_p_unique[roi][m] = p_u

            null_c = np.array(null_combined[roi][m])
            m_i    = models.index(m)
            obs_c  = float(roi_results[roi]['combined']['beta'][m_i])
            p_c    = float(np.mean(null_c >= obs_c))
            perm_p_combined[roi][m] = p_c

            print(f"    {roi:20s}  {m:10s}  "
                  f"unique  p={p_u:.4f} (obs_b={obs_u:.4f}, null_mean={null_u.mean():.4f})  |  "
                  f"combined p={p_c:.4f} (obs_b={obs_c:.4f})")

    # Save null distributions
    np.save(os.path.join(OUT_DIR, 'perm_null_unique.npy'),
            {roi: {m: np.array(null_unique[roi][m]) for m in models}
             for roi in present_rois})
    np.save(os.path.join(OUT_DIR, 'perm_null_combined.npy'),
            {roi: {m: np.array(null_combined[roi][m]) for m in models}
             for roi in present_rois})
    print(f"  Null distributions saved to {OUT_DIR}")


# ── Significance helper ────────────────────────────────────────────────

def _perm_star(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


# ── Plot 1: summary heatmap ────────────────────────────────────────────

if RUN_PERMUTATIONS:
    fig_hm = mc.plotting.cell_results.plot_rsa_heatmap(
        results         = roi_results,
        models          = models,
        rois            = present_rois,
        title           = (f'State RSA — t-values per model × ROI  '
                           f'(DS={LEN_STANDARDISED_PATH} bins/state)'),
        save_path       = os.path.join(OUT_DIR, 'State_RSA_heatmap.png'),
        perm_p_unique   = perm_p_unique,
        perm_p_combined = perm_p_combined,
    )
else:
    fig_hm = mc.plotting.cell_results.plot_rsa_heatmap(
        results   = roi_results,
        models    = models,
        rois      = present_rois,
        title     = (f'State RSA — t-values per model × ROI  '
                     f'(DS={LEN_STANDARDISED_PATH} bins/state)'),
        save_path = os.path.join(OUT_DIR, 'State_RSA_heatmap.png'),
    )
plt.show()


# ── Plot 2: permutation null distributions ─────────────────────────────

if RUN_PERMUTATIONS:
    for reg_label, null_dict, perm_p_dict, obs_key in [
        ('unique',   null_unique,   perm_p_unique,   'unique'),
        ('combined', null_combined, perm_p_combined, 'combined'),
    ]:
        valid_rois    = [r for r in present_rois if r in _roi_data_rdms]
        n_rois_plot   = len(valid_rois)
        n_models_plot = len(models)
        print(f"\n  Plotting permutation distributions ({reg_label}): "
              f"{n_models_plot} models × {n_rois_plot} ROIs")

        fig_p, axes_p = plt.subplots(
            n_models_plot, n_rois_plot,
            figsize=(max(4, n_rois_plot * 2.5), max(3, n_models_plot * 2.5)),
            sharex=False, sharey=False,
            constrained_layout=True,
        )
        fig_p.suptitle(
            f'Permutation distributions — beta ({reg_label})\n'
            f'DS={LEN_STANDARDISED_PATH} bins/state',
            fontsize=12, weight='bold',
        )

        if n_models_plot == 1:
            axes_p = axes_p[np.newaxis, :]
        if n_rois_plot == 1:
            axes_p = axes_p[:, np.newaxis]

        for r_idx, roi in enumerate(valid_rois):
            for m_idx, m in enumerate(models):
                ax = axes_p[m_idx, r_idx]

                null_vals = np.array(null_dict[roi][m])
                if obs_key == 'unique':
                    obs_val = roi_results[roi]['unique'][m][1]
                else:
                    obs_val = float(roi_results[roi]['combined']['beta'][models.index(m)])
                p_val = perm_p_dict[roi][m]

                ax.hist(null_vals, bins=40, color='steelblue', alpha=0.7, density=True)
                ax.axvline(obs_val, color='crimson', lw=2,
                           label=f'obs={obs_val:.3f}')
                ax.set_title(
                    f'{roi} (n={roi_results[roi]["n_neurons"]})\n{m}  {_perm_star(p_val)} (p={p_val:.3f})',
                    fontsize=7, pad=3,
                )
                ax.tick_params(labelsize=6)
                if r_idx == 0:
                    ax.set_ylabel('density', fontsize=6)
                if m_idx == n_models_plot - 1:
                    ax.set_xlabel('beta', fontsize=6)
                ax.legend(fontsize=5, loc='upper left')

        save_perm = os.path.join(OUT_DIR, f'State_RSA_perm_distributions_{reg_label}.png')
        fig_p.savefig(save_perm, dpi=150, bbox_inches='tight')
        print(f'  Saved → {save_perm}')
        plt.show()

    print("Permutation testing complete.")

print('\nAll done.')
