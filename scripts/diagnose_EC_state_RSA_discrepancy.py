#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_EC_state_RSA_discrepancy.py

Investigates why the state RSA for EC gives puzzlingly different results
between RSA_human_cells_DSR.py and State_RSA_human_cells-2026.py.

Suspected sources of discrepancy (tested in sequence):
  A. Different sessions  — DSR uses only the DSR-subset; State uses all s01-63
  B. Different data files — DSR reads dsr_avg/, State reads state_avg/
  C. Different neural matrix structure
       DSR:   8 configs × N_PHASES subpaths  (config-ordered)
       State: 6 pseudo-groups × flat states  (group-ordered)
  D. Different state/feedback model construction
       DSR:   N_PHASES=3 subpath rows per state, blocks within config
       State: 1 flat block per state, tiled across groups
  E. Phase mask in DSR (MASK_CROSS_PHASE) — masks cross-phase RDM entries
  F. N_PERMS permutation null: check whether null is degenerate in either

Each step is run independently and results printed + saved.
The script ends with a direct apples-to-apples comparison:
  DSR sessions only + state_avg data + State-style model → compare to State script.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import mc

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_DIR  = os.path.join(DATA_DIR, 'group', 'EC_discrepancy_diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

ROI_FOCUS    = 'EC'
INCLUDE_DIAG = False
N_PERMS      = 500
RNG_SEED     = 42

# Shared downsampling target: match State script resolution
# State: LEN_STANDARDISED_PATH=3 bins/state → 12 bins per half-config
LEN_STATE_BINS = 3   # bins per state in flattened representation
N_STATES       = 4
BINS_PER_HALF  = N_STATES * LEN_STATE_BINS   # 12

# DSR parameters (from RSA_human_cells_DSR.py)
N_PHASES      = 3
LEN_DSR_PATH  = 5   # LEN_STANDARDISED_PATH in DSR script
DSR_CONFIGS   = ['3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3',
                 '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6']
N_DSR_CONFIGS = len(DSR_CONFIGS)

# State script parameters
N_GROUPS_STATE = 6
TOTAL_BINS_STATE = BINS_PER_HALF * N_GROUPS_STATE   # 72

states = ['A', 'B', 'C', 'D']

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def downsample_mean(x, target_len):
    x = np.asarray(x)
    n = len(x)
    edges = np.linspace(0, n, target_len + 1, dtype=int)
    out = []
    for i in range(target_len):
        chunk = x[edges[i]:edges[i+1]]
        out.append(np.nan if len(chunk) == 0 else np.nanmean(chunk))
    return np.array(out)


def _scalar(arr):
    return float(np.asarray(arr, dtype=float).ravel()[0])


def run_rsa_simple(data_mat, model_rdms_dict, model_list, include_diag, n_tasks):
    """Compute data RDM and regress model RDMs. Returns results dict."""
    data_rdm = mc.analyse.my_RSA.compute_crosscorr(
        data_mat, plotting=False, include_diagonal=include_diag, no_tasks=n_tasks)
    unique = {}
    for m in model_list:
        res = mc.analyse.my_RSA.evaluate_model(model_rdms_dict[m][0], data_rdm[0])
        unique[m] = (_scalar(res[0]), _scalar(res[1]), _scalar(res[2]))
    stacked = np.stack([model_rdms_dict[m][0] for m in model_list], axis=1)
    res_c   = mc.analyse.my_RSA.evaluate_model(stacked, data_rdm[0])
    return {
        'data_rdm': data_rdm,
        'unique':   unique,
        'combined_beta': np.asarray(res_c[1], dtype=float).ravel(),
        'combined_p':    np.asarray(res_c[2], dtype=float).ravel(),
    }


def run_permutation_test(data_rdm_vec, model_rdms_dict, model_list,
                         n_conditions, include_diag, n_perms, rng):
    """
    Permute model condition labels within a single block (no config splitting —
    appropriate when the model has a single flat block of n_conditions rows).
    Returns perm_p_unique dict and null beta arrays.
    """
    k = 0 if include_diag else 1

    def vec_to_sq(vec):
        mat = np.full((n_conditions, n_conditions), np.nan)
        ii, jj = np.triu_indices(n_conditions, k=k)
        mat[ii, jj] = vec
        mat[jj, ii] = vec
        return mat

    model_2d = {m: vec_to_sq(model_rdms_dict[m][0]) for m in model_list}
    ii, jj   = np.triu_indices(n_conditions, k=k)

    null_unique   = {m: [] for m in model_list}
    null_combined = {m: [] for m in model_list}

    for p_i in range(n_perms):
        perm_idx = rng.permutation(n_conditions)
        perm_vecs = {}
        for m in model_list:
            perm_2d       = model_2d[m][np.ix_(perm_idx, perm_idx)]
            perm_vecs[m]  = perm_2d[ii, jj].copy()

        for m in model_list:
            res = mc.analyse.my_RSA.evaluate_model(perm_vecs[m], data_rdm_vec)
            null_unique[m].append(float(np.asarray(res[1], dtype=float).ravel()[0]))
        stk   = np.stack([perm_vecs[m] for m in model_list], axis=1)
        res_c = mc.analyse.my_RSA.evaluate_model(stk, data_rdm_vec)
        bc    = np.asarray(res_c[1], dtype=float).ravel()
        for m_i, m in enumerate(model_list):
            null_combined[m].append(float(bc[m_i]))

        if (p_i + 1) % 100 == 0:
            print(f"    perm {p_i+1}/{n_perms}")

    perm_p_unique   = {}
    perm_p_combined = {}
    for m in model_list:
        obs_u = null_unique[m]    # will be filled after we call outside
        obs_c = null_combined[m]
        # placeholder — caller replaces with observed value
        perm_p_unique[m]   = np.array(null_unique[m])
        perm_p_combined[m] = np.array(null_combined[m])

    return perm_p_unique, perm_p_combined


def compute_perm_p(obs_beta, null_arr):
    return float(np.mean(null_arr >= obs_beta))


def print_result(label, result, model_list):
    print(f"\n  [{label}]")
    for m in model_list:
        t, b, p = result['unique'][m]
        print(f"    {m:10s}  t={t:6.3f}  beta={b:7.4f}  OLS_p={p:.4f}")


def plot_null_vs_obs(null_dict, obs_dict, model_list, title, save_path):
    fig, axes = plt.subplots(1, len(model_list),
                             figsize=(4 * len(model_list), 3.5),
                             constrained_layout=True)
    if len(model_list) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=10, weight='bold')
    for ax, m in zip(axes, model_list):
        null = null_dict[m]
        obs  = obs_dict[m]
        p    = compute_perm_p(obs, null)
        ax.hist(null, bins=40, color='steelblue', alpha=0.7, density=True)
        ax.axvline(obs, color='crimson', lw=2, label=f'obs={obs:.3f}\np={p:.4f}')
        ax.set_title(f'{m}', fontsize=9)
        ax.set_xlabel('beta'); ax.set_ylabel('density')
        ax.legend(fontsize=7)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"  Saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0: Identify sessions and neuron counts in each script
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 0: Session inventory")
print("=" * 70)

# DSR sessions
with open(os.path.join(DATA_DIR, 'all_sessions_dsrRSA_grouping_summary.json')) as f:
    config_summary = json.load(f)
dsr_sessions = set(config_summary.keys())

# State sessions: all with state_avg files
all_subs = [f'{i:02}' for i in range(1, 64)]
state_sessions = set()
for sub in all_subs:
    p = os.path.join(DATA_DIR, f's{sub}', 'state_avg', f's{sub}_neural_avg.npy')
    if os.path.exists(p):
        state_sessions.add(sub)

overlap_sessions    = dsr_sessions & state_sessions
only_state_sessions = state_sessions - dsr_sessions
only_dsr_sessions   = dsr_sessions - state_sessions

print(f"  DSR sessions:                {len(dsr_sessions)}")
print(f"  State sessions:              {len(state_sessions)}")
print(f"  Overlap (in both):           {len(overlap_sessions)}")
print(f"  Only in State (not DSR):     {len(only_state_sessions)}  → {sorted(only_state_sessions)}")
print(f"  Only in DSR  (not State):    {len(only_dsr_sessions)}  → {sorted(only_dsr_sessions)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load STATE_AVG data — all sessions vs DSR-subset sessions
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: Load state_avg data; compare EC neurons across session subsets")
print("=" * 70)

def load_state_avg_neurons(session_list):
    rows = []
    for sub in sorted(session_list):
        npy  = os.path.join(DATA_DIR, f's{sub}', 'state_avg', f's{sub}_neural_avg.npy')
        meta = os.path.join(DATA_DIR, f's{sub}', 'state_avg', f's{sub}_neuron_meta.json')
        if not os.path.exists(npy):
            continue
        data = np.load(npy)
        with open(meta) as f:
            nmeta = json.load(f)
        for n_idx, n in enumerate(data):
            rows.append({
                'session':      sub,
                'neuron_label': nmeta['neuron_names'][n_idx],
                'roi':          nmeta['cell_labels'][n_idx],
                'neuron_data':  n,   # (N_GROUPS, 2, 360)
            })
    return pd.DataFrame(rows)

neurons_all = load_state_avg_neurons(state_sessions)
neurons_dsr = load_state_avg_neurons(overlap_sessions)

def count_ec(df):
    ec = df[df['roi'] == ROI_FOCUS]
    return len(ec['neuron_label'].unique())

print(f"  EC neurons — all sessions:   {count_ec(neurons_all)}")
print(f"  EC neurons — DSR-subset:     {count_ec(neurons_dsr)}")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD STATE-STYLE MODEL RDMs  (as in State script)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Building State-style model RDMs  (N_GROUPS=6, flat per-state blocks)")
print("=" * 70)

state_one_half    = np.zeros((TOTAL_BINS_STATE, N_STATES))
feedback_one_half = np.zeros((TOTAL_BINS_STATE, N_STATES))
for g in range(N_GROUPS_STATE):
    offset = g * BINS_PER_HALF
    for s_i, s in enumerate(states):
        s_start = offset + s_i * LEN_STATE_BINS
        s_end   = offset + (s_i + 1) * LEN_STATE_BINS
        state_one_half[s_start:s_end, s_i] = 1
        if s == 'A':
            feedback_one_half[s_start, s_i] = 1

state_full    = np.vstack((state_one_half, state_one_half))
feedback_full = np.vstack((feedback_one_half, feedback_one_half))

state_RDM    = mc.analyse.my_RSA.compute_crosscorr(
    state_full,    plotting=False, include_diagonal=INCLUDE_DIAG, no_tasks=N_GROUPS_STATE)
feedback_RDM = mc.analyse.my_RSA.compute_crosscorr(
    feedback_full, plotting=False, include_diagonal=INCLUDE_DIAG, no_tasks=N_GROUPS_STATE)
feedback_RDM[0][np.isnan(feedback_RDM[0])] = 1

model_rdms_state = {'state': state_RDM, 'feedback': feedback_RDM}
model_list_state = ['state', 'feedback']

n_cond_state = N_GROUPS_STATE * BINS_PER_HALF
print(f"  State model n_conditions (half-RDM rows): {n_cond_state}")
print(f"  state_RDM vector length:    {len(state_RDM[0])}")
print(f"  feedback_RDM vector length: {len(feedback_RDM[0])}")
print(f"  feedback NaNs patched:      {int(np.isnan(feedback_RDM[0]).sum())} (should be 0 now)")


def build_state_matrix(neuron_subset, n_groups=N_GROUPS_STATE, n_bins=360):
    """Build neural matrix in State-script style: group-ordered, downsampled."""
    uniq = neuron_subset['neuron_label'].unique()
    n_neurons = len(uniq)
    total = n_groups * BINS_PER_HALF
    th1 = np.zeros((total, n_neurons))
    th2 = np.zeros((total, n_neurons))
    for n_idx, nl in enumerate(uniq):
        row = neuron_subset[neuron_subset['neuron_label'] == nl].iloc[0]
        nd  = row['neuron_data']   # (N_GROUPS, 2, 360)
        for g in range(n_groups):
            g_off = g * BINS_PER_HALF
            th1[g_off:g_off + BINS_PER_HALF, n_idx] = downsample_mean(nd[g, 0, :], BINS_PER_HALF)
            th2[g_off:g_off + BINS_PER_HALF, n_idx] = downsample_mean(nd[g, 1, :], BINS_PER_HALF)
    valid = ~(np.isnan(th1).any(axis=0) | np.isnan(th2).any(axis=0))
    print(f"    build_state_matrix: {n_neurons} neurons, "
          f"{int((~valid).sum())} excluded (NaN), {int(valid.sum())} valid")
    return np.concatenate([th1[:, valid], th2[:, valid]], axis=0), int((~valid).sum())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: RSA on EC using State-style pipeline
#   2a. All sessions (= what State script runs)
#   2b. DSR-subset sessions only
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: EC RSA with STATE-style pipeline — all vs DSR-subset sessions")
print("=" * 70)

rng = np.random.default_rng(RNG_SEED)

for label, df in [('State-pipeline / all sessions', neurons_all),
                   ('State-pipeline / DSR-subset sessions', neurons_dsr)]:
    ec = df[df['roi'] == ROI_FOCUS]
    n_ec = len(ec['neuron_label'].unique())
    if n_ec == 0:
        print(f"  {label}: no EC neurons, skipping.")
        continue
    print(f"\n  {label}: {n_ec} EC neurons")
    mat, excl = build_state_matrix(ec)
    res = run_rsa_simple(mat, model_rdms_state, model_list_state,
                         INCLUDE_DIAG, N_GROUPS_STATE)
    print_result(label, res, model_list_state)

    # Check data RDM: is it near-constant (degenerate)?
    dv = res['data_rdm'][0].astype(float)
    print(f"    data RDM: min={np.nanmin(dv):.4f}  max={np.nanmax(dv):.4f}  "
          f"std={np.nanstd(dv):.6f}  NaNs={np.isnan(dv).sum()}")

    # Permutation
    print(f"    Running {N_PERMS} permutations...")
    null_u, null_c = run_permutation_test(
        res['data_rdm'][0].astype(float), model_rdms_state, model_list_state,
        n_cond_state, INCLUDE_DIAG, N_PERMS, rng)

    obs_betas = {m: res['unique'][m][1] for m in model_list_state}
    for m in model_list_state:
        p = compute_perm_p(obs_betas[m], null_u[m])
        print(f"    {m:10s}  obs_beta={obs_betas[m]:.4f}  "
              f"null_mean={null_u[m].mean():.4f}  null_std={null_u[m].std():.4f}  "
              f"perm_p={p:.4f}")

    slug = label.replace('/', '_').replace(' ', '_')
    plot_null_vs_obs(
        null_u, obs_betas, model_list_state,
        title=f'EC null distributions\n{label}',
        save_path=os.path.join(OUT_DIR, f'null_{slug}.png'))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Load DSR_AVG data for the overlap sessions and run State-style RSA
#   This tests whether the data FILE (dsr_avg vs state_avg) matters for EC
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: DSR_AVG data (overlap sessions) → State-style RSA on EC")
print("  (Is the discrepancy in the data file or the model?)")
print("=" * 70)

def load_dsr_avg_as_state_style(session_list):
    """
    Load dsr_avg data and reshape to mimic state_avg structure
    (N_GROUPS=N_CONFIGS=8, 2 runs, 360 bins).
    Each config block becomes one 'group'.
    """
    rows = []
    for sub in sorted(session_list):
        npy  = os.path.join(DATA_DIR, f's{sub}', 'dsr_avg', f's{sub}_dsr_neural_avg.npy')
        meta = os.path.join(DATA_DIR, f's{sub}', 'dsr_avg', f's{sub}_dsr_neuron_meta.json')
        if not os.path.exists(npy):
            print(f"    dsr_avg missing for s{sub}")
            continue
        data = np.load(npy)   # (n_neurons, N_CONFIGS, 2, 360)
        with open(meta) as f:
            nmeta = json.load(f)
        for n_idx in range(data.shape[0]):
            rows.append({
                'session':      sub,
                'neuron_label': nmeta['neuron_names'][n_idx],
                'roi':          nmeta['cell_labels'][n_idx],
                'neuron_data':  data[n_idx],   # (N_CONFIGS, 2, 360)
            })
    return pd.DataFrame(rows)

neurons_dsr_avg = load_dsr_avg_as_state_style(overlap_sessions)

# Build State-style model for N_CONFIGS=8 groups (same structure, more groups)
N_GROUPS_DSR    = N_DSR_CONFIGS
TOTAL_BINS_DSR8 = BINS_PER_HALF * N_GROUPS_DSR

state_one_half_8    = np.zeros((TOTAL_BINS_DSR8, N_STATES))
feedback_one_half_8 = np.zeros((TOTAL_BINS_DSR8, N_STATES))
for g in range(N_GROUPS_DSR):
    offset = g * BINS_PER_HALF
    for s_i, s in enumerate(states):
        s_start = offset + s_i * LEN_STATE_BINS
        s_end   = offset + (s_i + 1) * LEN_STATE_BINS
        state_one_half_8[s_start:s_end, s_i] = 1
        if s == 'A':
            feedback_one_half_8[s_start, s_i] = 1

state_full_8    = np.vstack((state_one_half_8, state_one_half_8))
feedback_full_8 = np.vstack((feedback_one_half_8, feedback_one_half_8))
state_RDM_8    = mc.analyse.my_RSA.compute_crosscorr(
    state_full_8,    plotting=False, include_diagonal=INCLUDE_DIAG, no_tasks=N_GROUPS_DSR)
feedback_RDM_8 = mc.analyse.my_RSA.compute_crosscorr(
    feedback_full_8, plotting=False, include_diagonal=INCLUDE_DIAG, no_tasks=N_GROUPS_DSR)
feedback_RDM_8[0][np.isnan(feedback_RDM_8[0])] = 1
model_rdms_8 = {'state': state_RDM_8, 'feedback': feedback_RDM_8}
n_cond_8     = N_GROUPS_DSR * BINS_PER_HALF
print(f"  State-style model for 8 groups: n_conditions={n_cond_8}, "
      f"RDM vec len={len(state_RDM_8[0])}")

if len(neurons_dsr_avg) > 0:
    ec_dsr_avg = neurons_dsr_avg[neurons_dsr_avg['roi'] == ROI_FOCUS]
    n_ec_dsr = len(ec_dsr_avg['neuron_label'].unique())
    print(f"  EC neurons in dsr_avg data: {n_ec_dsr}")

    if n_ec_dsr > 0:
        mat_dsr8, excl = build_state_matrix(ec_dsr_avg, n_groups=N_GROUPS_DSR)
        res_dsr8 = run_rsa_simple(mat_dsr8, model_rdms_8, model_list_state,
                                  INCLUDE_DIAG, N_GROUPS_DSR)
        print_result('DSR-avg data / State-style model / 8 groups', res_dsr8, model_list_state)

        dv = res_dsr8['data_rdm'][0].astype(float)
        print(f"    data RDM: min={np.nanmin(dv):.4f}  max={np.nanmax(dv):.4f}  "
              f"std={np.nanstd(dv):.6f}  NaNs={np.isnan(dv).sum()}")

        print(f"    Running {N_PERMS} permutations...")
        null_u_dsr8, _ = run_permutation_test(
            res_dsr8['data_rdm'][0].astype(float), model_rdms_8, model_list_state,
            n_cond_8, INCLUDE_DIAG, N_PERMS, rng)

        obs_betas_dsr8 = {m: res_dsr8['unique'][m][1] for m in model_list_state}
        for m in model_list_state:
            p = compute_perm_p(obs_betas_dsr8[m], null_u_dsr8[m])
            print(f"    {m:10s}  obs_beta={obs_betas_dsr8[m]:.4f}  "
                  f"null_mean={null_u_dsr8[m].mean():.4f}  "
                  f"null_std={null_u_dsr8[m].std():.4f}  perm_p={p:.4f}")

        plot_null_vs_obs(
            null_u_dsr8, obs_betas_dsr8, model_list_state,
            title='EC null — dsr_avg data, State-style model, 8 groups\n(overlap sessions)',
            save_path=os.path.join(OUT_DIR, 'null_dsr_avg_state_model_8groups.png'))
else:
    print("  No dsr_avg neurons loaded — check paths.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Inspect data RDM structure — is it degenerate?
#   Plot the full data RDM for EC in both pipelines side by side.
#   A degenerate RDM (all same value) will give a trivial null with p=0.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Data RDM structure inspection for EC")
print("=" * 70)

def rdm_vec_to_square(vec, n, k=1):
    mat = np.full((n, n), np.nan)
    ii, jj = np.triu_indices(n, k=k)
    mat[ii, jj] = vec
    mat[jj, ii] = vec
    return mat

fig_rdm, axes_rdm = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
fig_rdm.suptitle(f'EC data RDM structure comparison\n'
                 f'(left: State-pipeline all sessions; '
                 f'right: State-pipeline DSR-subset sessions)', fontsize=10)

for ax, (label, df) in zip(axes_rdm,
                            [('All sessions', neurons_all),
                             ('DSR-subset', neurons_dsr)]):
    ec = df[df['roi'] == ROI_FOCUS]
    if len(ec) == 0:
        ax.set_title(f'{label}: no EC neurons')
        continue
    mat, _ = build_state_matrix(ec)
    rdm_obj = mc.analyse.my_RSA.compute_crosscorr(
        mat, plotting=False, include_diagonal=INCLUDE_DIAG, no_tasks=N_GROUPS_STATE)
    rdm_vec = rdm_obj[0].astype(float)
    n_sq = N_GROUPS_STATE * BINS_PER_HALF
    rdm_sq  = rdm_vec_to_square(rdm_vec, n_sq, k=0 if INCLUDE_DIAG else 1)
    n_ec = len(ec['neuron_label'].unique())
    im = ax.imshow(rdm_sq, aspect='auto', cmap='RdBu_r', vmin=0, vmax=2)
    for g in range(1, N_GROUPS_STATE):
        ax.axvline(g * BINS_PER_HALF - 0.5, color='white', lw=0.8, ls='--')
        ax.axhline(g * BINS_PER_HALF - 0.5, color='white', lw=0.8, ls='--')
    ax.set_title(f'{label}\n(n={n_ec}, std={np.nanstd(rdm_vec):.4f})', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.6)

fig_rdm.savefig(os.path.join(OUT_DIR, 'EC_data_RDM_comparison.png'), dpi=120, bbox_inches='tight')
print(f"  Saved → {os.path.join(OUT_DIR, 'EC_data_RDM_comparison.png')}")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Check whether permuting states vs permuting configs matters
#   In the State script the permutation shuffles within each group block.
#   But the state model is perfectly periodic (same pattern in every group).
#   Shuffling within a single group may leave cross-group structure intact.
#   Alternative: permute across all conditions globally.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Permutation strategy — within-group vs global shuffle")
print("  Checks whether the null is degenerate due to block-repetition in model")
print("=" * 70)

ec_all = neurons_all[neurons_all['roi'] == ROI_FOCUS]
if len(ec_all) > 0:
    mat_all, _ = build_state_matrix(ec_all)
    res_all = run_rsa_simple(mat_all, model_rdms_state, model_list_state,
                             INCLUDE_DIAG, N_GROUPS_STATE)
    data_vec = res_all['data_rdm'][0].astype(float)
    n_cond   = N_GROUPS_STATE * BINS_PER_HALF
    k_flag   = 0 if INCLUDE_DIAG else 1

    def vec_to_sq(vec):
        mat = np.full((n_cond, n_cond), np.nan)
        ii, jj = np.triu_indices(n_cond, k=k_flag)
        mat[ii, jj] = vec
        mat[jj, ii] = vec
        return mat

    model_2d = {m: vec_to_sq(model_rdms_state[m][0]) for m in model_list_state}
    ii_t, jj_t = np.triu_indices(n_cond, k=k_flag)

    rng2 = np.random.default_rng(RNG_SEED + 1)

    for strategy, desc in [('within_group', 'Within-group block shuffle (current)'),
                             ('global',       'Global shuffle across all conditions')]:
        print(f"\n  Strategy: {desc}")
        null_state = []
        for p_i in range(N_PERMS):
            if strategy == 'within_group':
                perm_idx = np.arange(n_cond)
                for g in range(N_GROUPS_STATE):
                    blk = slice(g * BINS_PER_HALF, (g + 1) * BINS_PER_HALF)
                    perm_idx[blk] = perm_idx[blk][rng2.permutation(BINS_PER_HALF)]
            else:
                perm_idx = rng2.permutation(n_cond)

            perm_2d  = model_2d['state'][np.ix_(perm_idx, perm_idx)]
            perm_vec = perm_2d[ii_t, jj_t].copy()
            res_p    = mc.analyse.my_RSA.evaluate_model(perm_vec, data_vec)
            null_state.append(float(np.asarray(res_p[1], dtype=float).ravel()[0]))

            if (p_i + 1) % 100 == 0:
                print(f"    perm {p_i+1}/{N_PERMS}")

        null_arr = np.array(null_state)
        obs_beta = res_all['unique']['state'][1]
        p_val    = compute_perm_p(obs_beta, null_arr)
        print(f"  state obs_beta={obs_beta:.4f}  "
              f"null_mean={null_arr.mean():.4f}  null_std={null_arr.std():.6f}  "
              f"perm_p={p_val:.4f}")
        print(f"  Null unique values: {len(np.unique(np.round(null_arr, 6)))} "
              f"(if 1 → degenerate null!)")

        fig_s, ax_s = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
        ax_s.hist(null_arr, bins=40, color='steelblue', alpha=0.7, density=True)
        ax_s.axvline(obs_beta, color='crimson', lw=2,
                     label=f'obs={obs_beta:.3f}\np={p_val:.4f}')
        ax_s.set_title(f'EC state null — {desc}', fontsize=9)
        ax_s.set_xlabel('beta'); ax_s.legend(fontsize=7)
        sp = os.path.join(OUT_DIR, f'null_strategy_{strategy}.png')
        fig_s.savefig(sp, dpi=120, bbox_inches='tight')
        print(f"  Saved → {sp}")
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Summary table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY — what differs between the two scripts for EC / state")
print("=" * 70)
rows_summary = [
    ("Sessions",         f"State: {len(state_sessions)}", f"DSR: {len(dsr_sessions)} (overlap: {len(overlap_sessions)})"),
    ("Data source",      "state_avg/ (N_GROUPS groups)", "dsr_avg/ (per-config)"),
    ("Neural structure", f"6 groups × {BINS_PER_HALF} DS bins", f"8 configs × {N_PHASES*N_STATES} DS bins"),
    ("State model",      "flat per-state block × groups", "N_PHASES subpaths per state × configs"),
    ("Phase mask",       "None", "MASK_CROSS_PHASE=True"),
    ("Permutation",      "within-group block shuffle", "within-config block shuffle"),
]
print(f"  {'Dimension':<22}  {'State script':<40}  {'DSR script'}")
print("  " + "-" * 85)
for dim, sv, dv in rows_summary:
    print(f"  {dim:<22}  {sv:<40}  {dv}")

print("\nAll done. Figures saved to:", OUT_DIR)
