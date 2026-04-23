#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSR RSA — Crossnobis data RDM variant.

Mirrors RSA_human_cells_DSR.py in:
  - model-RDM construction (location, DSR, state, feedback, clocks, phase,
    midnight, loc_og),
  - unique / combined regression evaluation,
  - heatmap plotting.

Differs in how the DATA RDM is computed.  Instead of averaging split-half
patterns, we rebuild patterns from raw correct trials and compute a
cross-validated Mahalanobis (crossnobis) RDM per session, averaged across
sessions:

  1. Load raw neurons + beh for the session.
  2. For each config, find its grid_no blocks — these are its 'runs'
     (usually 3, sometimes 2 or 4).
  3. Smooth each correct trial's 360-bin timecourse with the same Gaussian
     kernel used in prep_human_cells_RSA-2026.py, then downsample to
     N_CONDS_PER_CONF timebins.
  4. Align to K = min(n_runs) across configs: per config, use the first K
     runs.  Compute X[cfg, run, tb, neuron] = mean over correct repeats.
  5. Estimate noise covariance Σ from per-run residuals pooled across
     (cfg, run, repeat, timebin), shrink by alpha·I, take Σ⁻¹.
  6. Leave-one-run-out crossnobis: for each held-out run k, let A = run k,
     B = mean of the other K-1 runs, then
         RDM_k[i,j] = (A[i]-A[j]) Σ⁻¹ (B[i]-B[j])
     Average across K folds → per-session RDM.  Vectorize the upper
     triangle, average across sessions → group data RDM.
  7. ACC stability: per (task, timebin) cross-validated pattern reliability
         r(c,tb) = (1/P) Σ_pairs  X_A[c,tb] · Σ⁻¹ · X_B[c,tb]
     averaged across sessions.  Near-zero → no stable representation.
"""
import os
import sys
import io
import json
import contextlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
import mc

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')


# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR     = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_DIR      = os.path.join(DATA_DIR, 'group', 'DSR_RSA_crossnobis')
INCLUDE_DIAG = False
PLOT_FIGS    = True

# Gaussian smoothing — must match prep_human_cells_RSA-2026.py
_GAUSS_SIGMA    = 2
_GAUSS_TRUNCATE = 2.0

# Crossnobis shrinkage
SHRINKAGE_ALPHA = 0.1

# If True, concatenate every session's neurons into a single pseudo-population
# (block-diagonal Σ⁻¹ since sessions share no trials) and compute ONE RDM,
# treating all neurons as if recorded from the same subject.  K = min n_runs
# across configs AND sessions.
# If False, compute one RDM per session and average across sessions.
POOL_PSEUDO_POP = True

# Only include sessions where every config has at least this many runs.
# Sessions below the threshold are dropped entirely (so the surviving
# neurons all have ≥MIN_RUNS_PER_NEURON runs for every config).
MIN_RUNS_PER_NEURON = 3

# Must match the model-RDM resolution
N_PHASES               = 3
states                 = ['A', 'B', 'C', 'D']
LEN_STANDARDISED_PATH  = 10
RESOLUTIONx            = 2
N_CONDS_PER_CONF       = N_PHASES * len(states) * RESOLUTIONx

# Within-run style for model RDMs (crossnobis already supplies its own
# cross-validation structure, so we pair it with averaged-half model RDMs).
WITHIN_RUN = True

MASK_CROSS_PHASE = False

rois_of_interest = ['whole_brain', 'OFC', 'EC', 'ACC', 'HC', 'PCC', 'AMY',
                    'R-WHITE-MATTER', 'OCCIP']
models           = ['location', 'dsr', 'state', 'feedback', 'clocks',
                    'phase', 'midnight', 'loc_og']
combo_models     = ['state', 'feedback', 'clocks', 'phase', 'midnight', 'loc_og']

configs = [
    '3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3',
    '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6',
]
DSR_CONFIGS_TUP = [tuple(int(x) for x in c.split('-')) for c in configs]
N_CONFIGS       = len(configs)

with open(os.path.join(DATA_DIR, 'all_sessions_dsrRSA_grouping_summary.json'), 'r') as f:
    config_summary = json.load(f)
SUBJECTS = list(config_summary.keys())

os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# ── Mode-location arrays (identical to RSA_human_cells_DSR.py) ────────
# ══════════════════════════════════════════════════════════════════════

print("Building mode-location arrays...")
accum = {c: {1: [], 2: []} for c in configs}

for sub in SUBJECTS:
    # grouping log tells us which grid_no blocks went to run1 vs run2
    glog_path_sep = os.path.join(DATA_DIR, f's{sub}', 'dsr_avg_runs_separate',
                                 f's{sub}_dsr_grouping_log.json')
    glog_path_two = os.path.join(DATA_DIR, f's{sub}', 'dsr_avg',
                                 f's{sub}_dsr_grouping_log_two_runs.json')
    glog_path = glog_path_two if os.path.exists(glog_path_two) else glog_path_sep
    if not os.path.exists(glog_path):
        continue
    with open(glog_path) as f:
        glog = json.load(f)

    data_dict, _ = mc.analyse.helpers_human_cells.get_data(int(sub))
    key = f'sub-{int(sub):02}'
    if key not in data_dict:
        continue
    beh  = data_dict[key]['beh'].copy()
    locs = data_dict[key]['locations']
    N_BINS = locs.shape[1]

    def _cfg_str(row):
        return (f"{int(row['loc_A'])}-{int(row['loc_B'])}-"
                f"{int(row['loc_C'])}-{int(row['loc_D'])}")
    beh['config_str'] = beh.apply(_cfg_str, axis=1)
    beh['grid_no']    = beh['grid_no'].astype(int)

    for cfg_entry in glog['configs']:
        c = cfg_entry['config']
        # support both "two runs" and "runs separate" log shapes
        if 'run1_blocks' in cfg_entry:
            pairs = [(1, cfg_entry['run1_blocks']), (2, cfg_entry['run2_blocks'])]
        else:
            # split the blocks list in half for the mode-location template
            blks = cfg_entry.get('blocks', [])
            mid  = len(blks) // 2 or 1
            pairs = [(1, blks[:mid]), (2, blks[mid:] or blks[:mid])]
        for run_id, blocks in pairs:
            mask = (
                (beh['config_str'] == c) &
                (beh['correct'] == 1) &
                (beh['grid_no'].isin(blocks))
            )
            idx = beh.index[mask]
            if len(idx) == 0:
                continue
            accum[c][run_id].append(locs.loc[idx].values.astype(float))

mode_locs = {}
for c in configs:
    mode_locs[c] = {}
    for run_id in [1, 2]:
        parts = accum[c][run_id]
        if not parts:
            mode_locs[c][run_id] = np.full(N_BINS, np.nan)
        else:
            stacked = np.vstack(parts)
            m = stats.mode(stacked, axis=0, keepdims=False, nan_policy='omit')
            mode_locs[c][run_id] = m.mode.astype(float)

print("Mode-location arrays built.")


# ══════════════════════════════════════════════════════════════════════
# ── Model RDMs (identical machinery to RSA_human_cells_DSR.py) ────────
# ══════════════════════════════════════════════════════════════════════

def downsample_mode(x, target_len=10):
    x = np.asarray(x)
    block = len(x) // target_len
    return np.array([
        stats.mode(x[i*block:(i+1)*block], keepdims=False).mode
        for i in range(target_len)
    ])


print("Building model matrices...")

loc_th1 = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, LEN_STANDARDISED_PATH), dtype=float)
loc_th2 = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, LEN_STANDARDISED_PATH), dtype=float)
dsr_th1 = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, LEN_STANDARDISED_PATH*N_PHASES*len(states)), dtype=float)
dsr_th2 = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, LEN_STANDARDISED_PATH*N_PHASES*len(states)), dtype=float)

for c_idx, c in enumerate(configs):
    row_curr = c_idx * N_CONDS_PER_CONF
    for run_id, loc_mat, dsr_mat in [(1, loc_th1, dsr_th1),
                                     (2, loc_th2, dsr_th2)]:
        mode_vec       = mode_locs[c][run_id]
        LEN_OG_SUBPATH = int(len(mode_vec) / N_CONDS_PER_CONF)
        dsr_first_step = downsample_mode(
            mode_vec, target_len=LEN_STANDARDISED_PATH*N_PHASES*len(states))
        for n_sp in range(N_CONDS_PER_CONF):
            sp = mode_vec[n_sp*LEN_OG_SUBPATH:(n_sp+1)*LEN_OG_SUBPATH]
            loc_mat[row_curr + n_sp, :] = downsample_mode(
                sp, target_len=LEN_STANDARDISED_PATH)
            dsr_mat[row_curr + n_sp, :] = np.roll(
                dsr_first_step, -n_sp * LEN_STANDARDISED_PATH)

state_config    = np.zeros((N_CONDS_PER_CONF, len(states)))
feedback_config = np.zeros((N_CONDS_PER_CONF, len(states)))
phase_config    = np.zeros((N_CONDS_PER_CONF, N_PHASES))
for s_i, s in enumerate(states):
    start_phase = RESOLUTIONx * s_i * N_PHASES
    state_config[start_phase: RESOLUTIONx * (s_i + 1) * N_PHASES, s_i] = 1
    if s == 'A':
        feedback_config[0:RESOLUTIONx, s_i] = 1
    for p_i in range(N_PHASES):
        phase_config[start_phase + p_i*RESOLUTIONx:
                     start_phase + (p_i+1)*RESOLUTIONx, p_i] = 1

feedback_half = np.tile(feedback_config, (len(configs), 1))
state_half    = np.tile(state_config,    (len(configs), 1))
phase_half    = np.tile(phase_config,    (len(configs), 1))

_n_clock_neurons = 9 * N_PHASES * (N_PHASES * len(states))
prep_models = {}
for m in ['clocks', 'midnight', 'loc_og']:
    prep_models[m] = {}
    for th in (1, 2):
        if m == 'clocks':
            prep_models[m][th] = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, _n_clock_neurons), dtype=float)
        elif m == 'midnight':
            prep_models[m][th] = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, 9 * N_PHASES), dtype=float)
        else:
            prep_models[m][th] = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, 9), dtype=float)

for c_idx, c in enumerate(configs):
    row_start   = c_idx * N_CONDS_PER_CONF
    task_config = [int(int(x) - 1) for x in c.split('-')]
    for run_id in (1, 2):
        curr_task = mode_locs[c][run_id].copy()
        for i in range(len(curr_task)):
            if np.isnan(curr_task[i]):
                curr_task[i] = curr_task[i - 1] if i > 0 else float(task_config[0] + 1)
        curr_task = [int(field_no - 1) for field_no in curr_task]
        LEN_OG_SUBPATH = int(len(curr_task) / N_CONDS_PER_CONF)

        with contextlib.redirect_stdout(io.StringIO()):
            loc_og_matrix = mc.simulation.predictions.set_location_ephys(
                curr_task, task_config, grid_size=3, plotting=False)
            clocks_matrix, midnight_matrix = mc.simulation.predictions.set_clocks_ephys(
                curr_task, task_config, grid_size=3, phases=N_PHASES, plotting=False)

        for n_sp in range(N_CONDS_PER_CONF):
            t0 = n_sp * LEN_OG_SUBPATH
            t1 = (n_sp + 1) * LEN_OG_SUBPATH
            for mat_name, src in [('clocks', clocks_matrix),
                                  ('midnight', midnight_matrix),
                                  ('loc_og', loc_og_matrix)]:
                sub_sp = np.nanmean(src[:, t0:t1], axis=1)
                prep_models[mat_name][run_id][row_start + n_sp, :] = np.where(
                    np.isnan(sub_sp), 0.0, sub_sp)

if WITHIN_RUN:
    model_concat = {
        'location': (loc_th1 + loc_th2) / 2,
        'dsr':      (dsr_th1 + dsr_th2) / 2,
        'state':    state_half,
        'feedback': feedback_half,
        'clocks':   (prep_models['clocks'][1]   + prep_models['clocks'][2])   / 2,
        'loc_og':   (prep_models['loc_og'][1]   + prep_models['loc_og'][2])   / 2,
        'midnight': (prep_models['midnight'][1] + prep_models['midnight'][2]) / 2,
        'phase':    phase_half,
    }
else:
    model_concat = {
        'location': np.concatenate([loc_th1,    loc_th2],    axis=0),
        'dsr':      np.concatenate([dsr_th1,    dsr_th2],    axis=0),
        'state':    np.tile(state_half,    (2, 1)),
        'feedback': np.tile(feedback_half, (2, 1)),
        'clocks':   np.concatenate([prep_models['clocks'][1],   prep_models['clocks'][2]],   axis=0),
        'loc_og':   np.concatenate([prep_models['loc_og'][1],   prep_models['loc_og'][2]],   axis=0),
        'midnight': np.concatenate([prep_models['midnight'][1], prep_models['midnight'][2]], axis=0),
        'phase':    np.tile(phase_half,    (2, 1)),
    }

print("Computing model RDMs...")
model_RDMs = {}
for m in models:
    if m in ('location', 'dsr'):
        fn = (mc.analyse.my_RSA.compute_hamming_distance_within if WITHIN_RUN
              else mc.analyse.my_RSA.compute_hamming_distance)
        model_RDMs[m] = fn(model_concat[m], plotting=PLOT_FIGS,
                           include_diagonal=INCLUDE_DIAG,
                           model_name=m, no_tasks=len(configs))
    else:
        fn = (mc.analyse.my_RSA.compute_crosscorr_within if WITHIN_RUN
              else mc.analyse.my_RSA.compute_crosscorr)
        model_RDMs[m] = fn(model_concat[m], plotting=PLOT_FIGS,
                           include_diagonal=INCLUDE_DIAG,
                           no_tasks=len(configs), model=m)

nan_mask_feedback = np.isnan(model_RDMs['feedback'][0])
model_RDMs['feedback'][0][nan_mask_feedback] = 1
for m in models:
    print(f"  {m:12s} RDM vec length: {len(model_RDMs[m][0])}")


# ══════════════════════════════════════════════════════════════════════
# ── Phase mask ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def make_phase_mask(n_phases, n_configs, n_states=4, include_diagonal=True):
    len_one_config = n_phases * n_states
    idx      = np.arange(len_one_config)
    phase_id = idx % n_phases
    sc_mask  = phase_id[:, None] == phase_id[None, :]
    all_mask = np.tile(sc_mask, (n_configs, n_configs))
    n = all_mask.shape[1]
    k = 0 if include_diagonal else 1
    return all_mask[np.triu_indices(n, k=k)]

if MASK_CROSS_PHASE:
    phase_mask = make_phase_mask(N_PHASES, N_CONFIGS, len(states),
                                 include_diagonal=INCLUDE_DIAG)
else:
    phase_mask = None


# ══════════════════════════════════════════════════════════════════════
# ── Crossnobis data RDM (per session) ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _downsample_mean_1d(x, target_len):
    n = len(x)
    edges = np.linspace(0, n, target_len + 1, dtype=int)
    out = np.empty(target_len, dtype=float)
    for i in range(target_len):
        chunk = x[edges[i]:edges[i+1]]
        out[i] = np.nan if len(chunk) == 0 else np.nanmean(chunk)
    return out


def _smooth_and_bin(row):
    sm = gaussian_filter1d(row, sigma=_GAUSS_SIGMA, truncate=_GAUSS_TRUNCATE)
    return _downsample_mean_1d(sm, N_CONDS_PER_CONF)


def load_session_trials(sub_str):
    """
    Build a per-session smoothed trial tensor + ROI labels.

    Returns:
        smooth_tensor : (n_neurons, n_trials, N_CONDS_PER_CONF)
        cell_labels   : list[str]  (length n_neurons, aligned with smooth_tensor[:,...])
        neuron_names  : list[str]
        beh           : DataFrame with 'config' tuple + int 'grid_no'
        ok            : bool
    """
    data_dict = mc.analyse.helpers_human_cells.load_norm_data(DATA_DIR, [sub_str])
    key = f'sub-{sub_str}'
    if key not in data_dict:
        return None, None, None, None, False

    beh         = data_dict[key]['beh'].copy()
    neurons_raw = data_dict[key]['normalised_neurons']
    cell_labels = data_dict[key]['cell_labels']
    if not neurons_raw:
        return None, None, None, None, False

    neuron_names = sorted(neurons_raw.keys())
    n_neurons    = len(neuron_names)
    n_trials     = len(beh)

    smooth_tensor = np.full((n_neurons, n_trials, N_CONDS_PER_CONF), np.nan)
    for n_idx, name in enumerate(neuron_names):
        arr = neurons_raw[name].to_numpy(dtype=float)
        nt  = min(arr.shape[0], n_trials)
        for t in range(nt):
            smooth_tensor[n_idx, t, :] = _smooth_and_bin(arr[t])

    beh['config'] = list(zip(
        beh['loc_A'].astype(int), beh['loc_B'].astype(int),
        beh['loc_C'].astype(int), beh['loc_D'].astype(int),
    ))
    beh['grid_no'] = beh['grid_no'].astype(int)

    # cell_labels may be longer/shorter than neuron_names (if cells were dropped
    # before saving). Truncate / pad defensively.
    if len(cell_labels) < n_neurons:
        cell_labels = list(cell_labels) + ['UNKNOWN'] * (n_neurons - len(cell_labels))
    else:
        cell_labels = list(cell_labels[:n_neurons])

    # z-score neurons
    smooth_tensor = _zscore_session_neurons(smooth_tensor)
    
    return smooth_tensor, cell_labels, neuron_names, beh, True


def _build_run_groups(beh, smooth_tensor, keep_idx):
    """
    For one session and a subset of neurons (keep_idx into axis 0), collect
    trials grouped by (c_idx, run_idx) where run_idx enumerates the config's
    blocks in ascending grid_no order.

    Returns dict {c_idx: {r_idx: ndarray (n_repeats, n_kept_neurons, N_CONDS_PER_CONF)}}
    """
    groups = {}
    for c_idx, cfg_tup in enumerate(DSR_CONFIGS_TUP):
        mask = (beh['config'] == cfg_tup) & (beh['correct'] == 1)
        sub_beh = beh[mask]
        blocks  = sorted(sub_beh['grid_no'].unique().tolist())
        runs    = {}
        for r_idx, blk in enumerate(blocks):
            tr_idx = sub_beh[sub_beh['grid_no'] == blk].index.tolist()
            tr_idx = [t for t in tr_idx if t < smooth_tensor.shape[1]]
            if not tr_idx:
                continue
            reps = smooth_tensor[np.ix_(keep_idx, tr_idx, np.arange(N_CONDS_PER_CONF))]
            # reps shape (n_neurons_kept, n_reps, n_tb) → (n_reps, n_neurons, n_tb)
            reps = np.transpose(reps, (1, 0, 2))
            # drop trials with any NaN
            valid = ~np.isnan(reps).any(axis=(1, 2))
            if not valid.any():
                continue
            runs[r_idx] = reps[valid]
        groups[c_idx] = runs
        # import pdb; pdb.set_trace()
    return groups

def _zscore_session_neurons(smooth_tensor):
    """
    Z-score each neuron within a session across all trials and timebins.
    This keeps condition differences, but puts neurons on a more comparable scale
    such that one neuron doesn't dominate the entire condition.
    """
    x = smooth_tensor.astype(float).copy()
    n_neurons = x.shape[0]
    for n in range(n_neurons):
        vals = x[n]
        mu = np.nanmean(vals)
        sd = np.nanstd(vals)
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        x[n] = (vals - mu) / sd
    return x

def _session_X_and_sigma(groups, K):
    """
    Build per-session X[cfg, run, tb, neuron] (mean over repeats) and the
    regularised noise covariance Σ estimated from per-run residuals.

    Returns (X, Sigma_inv, n_neurons) or (None, None, 0) if infeasible.
    """
    run_counts = [len(groups.get(c, {})) for c in range(N_CONFIGS)]
    if min(run_counts) < K:
        return None, None, 0

    first_reps = next(iter(next(iter(groups.values())).values()))
    n_neurons  = first_reps.shape[1]

    X = np.full((N_CONFIGS, K, N_CONDS_PER_CONF, n_neurons), np.nan)
    resid_list = []
    for c in range(N_CONFIGS):
        for k in range(K):
            reps     = groups[c][k]                         # (n_reps, n_neurons, n_tb)
            mean_pat = reps.mean(axis=0)                    # (n_neurons, n_tb)
            X[c, k]  = mean_pat.T                           # (n_tb, n_neurons)
            centered = reps - mean_pat[np.newaxis, :, :]
            flat     = np.transpose(centered, (0, 2, 1)).reshape(-1, n_neurons)
            resid_list.append(flat)

    residuals = np.vstack(resid_list)
    if residuals.shape[0] < 2:
        return None, None, n_neurons

    Sigma = np.cov(residuals, rowvar=False)
    if np.ndim(Sigma) == 0:
        Sigma = np.array([[float(Sigma)]])
    Sigma = (1 - SHRINKAGE_ALPHA) * Sigma + SHRINKAGE_ALPHA * np.eye(n_neurons)
    Sigma_inv = np.linalg.pinv(Sigma)
    return X, Sigma_inv, n_neurons


def _crossnobis_rdm_from_X(X, Sigma_inv, K):
    """Leave-one-run-out crossnobis RDM using provided Σ⁻¹."""
    n_cond = N_CONFIGS * N_CONDS_PER_CONF
    n_neurons = X.shape[-1]
    X_run = X.transpose(1, 0, 2, 3).reshape(K, n_cond, n_neurons)

    rdm_sum = np.zeros((n_cond, n_cond), dtype=float)
    for k in range(K):
        A    = X_run[k]
        rest = np.delete(np.arange(K), k)
        B    = X_run[rest].mean(axis=0)
        A_S  = A @ Sigma_inv
        AB   = A_S @ B.T
        diag = np.einsum('ij,ij->i', A_S, B)
        rdm_sum += diag[:, None] + diag[None, :] - AB - AB.T
    return rdm_sum / K


def _stability_from_X(X, Sigma_inv, K):
    """Per (cfg, tb) cross-validated pattern reliability, averaged across ordered run pairs."""
    stability_mat = np.zeros((N_CONFIGS, N_CONDS_PER_CONF), dtype=float)
    n_pairs = K * (K - 1)
    if n_pairs == 0:
        return stability_mat
    for c in range(N_CONFIGS):
        Xc   = X[c]                                         # (K, n_tb, n_neurons)
        Xc_S = Xc @ Sigma_inv
        acc  = np.zeros(N_CONDS_PER_CONF, dtype=float)
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                acc += np.einsum('tn,tn->t', Xc_S[i], Xc[j])
        stability_mat[c, :] = acc / n_pairs
    return stability_mat


def compute_crossnobis_session(groups, compute_stability=False):
    """
    One session → one RDM (+ optional stability matrix).
    Used only when POOL_PSEUDO_POP is False.
    """
    run_counts = [len(groups.get(c, {})) for c in range(N_CONFIGS)]
    if min(run_counts) < 2:
        return None, None, 0, 0
    K = min(run_counts)
    X, Sigma_inv, n_neurons = _session_X_and_sigma(groups, K)
    if X is None:
        return None, None, n_neurons, 0
    rdm = _crossnobis_rdm_from_X(X, Sigma_inv, K)
    stability_mat = _stability_from_X(X, Sigma_inv, K) if compute_stability else None
    return rdm, stability_mat, n_neurons, K


def _triu_vec(mat, include_diag):
    n = mat.shape[0]
    k = 0 if include_diag else 1
    return mat[np.triu_indices(n, k=k)]


def _collect_session_groups(roi_filter, sessions_data):
    """
    For each session, build the per-(config, run) repeat tensor limited to
    neurons in `roi_filter`.  Also report each session's min-runs-across-configs
    so the caller can pick a global K.
    """
    out = []
    for sd in sessions_data:
        cell_labels = sd['cell_labels']
        if roi_filter == 'whole_brain':
            keep = np.arange(len(cell_labels))
        else:
            keep = np.array([i for i, lb in enumerate(cell_labels)
                             if lb == roi_filter])
        if len(keep) == 0:
            continue
        groups = _build_run_groups(sd['beh'], sd['smooth_tensor'], keep)
        run_counts = [len(groups.get(c, {})) for c in range(N_CONFIGS)]
        if not run_counts or min(run_counts) < max(2, MIN_RUNS_PER_NEURON):
            continue
        out.append({'sub': sd['sub'], 'groups': groups,
                    'K_session': min(run_counts)})
    return out


def _plot_data_rdm(rdm_group, roi_filter):
    plt.figure()
    plt.imshow(rdm_group, aspect='auto', cmap='coolwarm')
    plt.title(f'Crossnobis data RDM — {roi_filter}')
    plt.colorbar()
    n = rdm_group.shape[0]
    for i in range(0, n, N_CONDS_PER_CONF):
        plt.axvline(i - 0.5, color='white', ls='dashed')
        plt.axhline(i - 0.5, color='white', ls='dashed')
    plt.yticks(np.arange(N_CONDS_PER_CONF // 2, n, N_CONDS_PER_CONF),
               configs, fontsize=7)
    save = os.path.join(OUT_DIR, f'data_rdm_{roi_filter}.png')
    plt.savefig(save, dpi=150, bbox_inches='tight')


def group_data_rdm_for_roi(roi_filter, sessions_data, compute_stability=False,
                           plot_first=False):
    """
    Build the data RDM for one ROI.  Two modes:

    - POOL_PSEUDO_POP=True (default): concatenate every session's neurons
      along the neuron axis to form one pseudo-population.  Σ is
      block-diagonal across sessions (no shared trials), so the pooled
      crossnobis RDM is the sum of per-session crossnobis RDMs computed
      with a shared global K = min_runs across sessions × configs.  One
      RDM, all neurons treated as if from the same subject.
    - POOL_PSEUDO_POP=False: compute one RDM per session (each with its
      own K) and average across sessions.

    Returns (data_rdm_vec, n_sessions_used, n_neurons_total, stability_avg).
    """
    sess_groups = _collect_session_groups(roi_filter, sessions_data)
    if not sess_groups:
        return None, 0, 0, None

    if POOL_PSEUDO_POP:
        K = min(sg['K_session'] for sg in sess_groups)
        if K < 2:
            return None, 0, 0, None

        n_cond = N_CONFIGS * N_CONDS_PER_CONF
        rdm_pool = np.zeros((n_cond, n_cond), dtype=float)
        stab_pool = (np.zeros((N_CONFIGS, N_CONDS_PER_CONF), dtype=float)
                     if compute_stability else None)
        # import pdb; pdb.set_trace()
        n_neurons_total = 0
        n_sessions_used = 0
        for sg in sess_groups:
            X, Sigma_inv, n_neurons = _session_X_and_sigma(sg['groups'], K)
            if X is None:
                continue
            # block-diagonal Σ⁻¹ ⇒ pooled RDM equals the sum of per-session
            # contributions; likewise for the bilinear stability metric.
            rdm_pool += _crossnobis_rdm_from_X(X, Sigma_inv, K)
            if compute_stability:
                stab_pool += _stability_from_X(X, Sigma_inv, K)
            n_neurons_total += n_neurons
            n_sessions_used += 1

        if n_sessions_used == 0:
            return None, 0, 0, None

        rdm_group = rdm_pool
        if plot_first and PLOT_FIGS:
            _plot_data_rdm(rdm_group, roi_filter)
        data_vec = _triu_vec(rdm_group, INCLUDE_DIAG)
        return data_vec, n_sessions_used, n_neurons_total, stab_pool

    # per-session averaging path
    rdm_mats, n_neurons_total, stabilities = [], 0, []
    for sg in sess_groups:
        K_s = sg['K_session']
        X, Sigma_inv, n_neurons = _session_X_and_sigma(sg['groups'], K_s)
        if X is None:
            continue
        rdm_mats.append(_crossnobis_rdm_from_X(X, Sigma_inv, K_s))
        n_neurons_total += n_neurons
        if compute_stability:
            stabilities.append(_stability_from_X(X, Sigma_inv, K_s))

    if not rdm_mats:
        return None, 0, 0, None

    rdm_group = np.mean(rdm_mats, axis=0)
    if plot_first and PLOT_FIGS:
        _plot_data_rdm(rdm_group, roi_filter)
    data_vec = _triu_vec(rdm_group, INCLUDE_DIAG)
    stab_avg = np.mean(stabilities, axis=0) if stabilities else None
    return data_vec, len(rdm_mats), n_neurons_total, stab_avg


# ══════════════════════════════════════════════════════════════════════
# ── RSA runner ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _scalar(a):
    return float(np.asarray(a, dtype=float).ravel()[0])


def run_rsa(data_vec):
    """Unique regression per model + combined regression — mirrors DSR script."""
    unique = {}
    for m in models:
        if m == 'dsr' and MASK_CROSS_PHASE:
            dvec = data_vec.copy().astype(float);   dvec[~phase_mask] = np.nan
            mvec = model_RDMs[m][0].copy().astype(float); mvec[~phase_mask] = np.nan
            res = mc.analyse.my_RSA.evaluate_model(mvec, dvec)
        else:
            res = mc.analyse.my_RSA.evaluate_model(model_RDMs[m][0], data_vec)
        unique[m] = (_scalar(res[0]), _scalar(res[1]), _scalar(res[2]))

    if MASK_CROSS_PHASE:
        mdict = {}
        for m in model_RDMs:
            tmp = model_RDMs[m][0].copy().astype(float)
            tmp[~phase_mask] = np.nan
            mdict[m] = tmp
        dvec = data_vec.copy().astype(float); dvec[~phase_mask] = np.nan
        stacked = np.stack([mdict[m] for m in combo_models], axis=1)
        res_c = mc.analyse.my_RSA.evaluate_model(stacked, dvec)
    else:
        stacked = np.stack([model_RDMs[m][0] for m in combo_models], axis=1)
        res_c = mc.analyse.my_RSA.evaluate_model(stacked, data_vec)

    combined = {
        't':    np.asarray(res_c[0], dtype=float).ravel(),
        'beta': np.asarray(res_c[1], dtype=float).ravel(),
        'p':    np.asarray(res_c[2], dtype=float).ravel(),
    }
    return {'unique': unique, 'combined': combined}


# ══════════════════════════════════════════════════════════════════════
# ── Load raw trials for every session once ────────────────────────────
# ══════════════════════════════════════════════════════════════════════

print("\nLoading raw trials per session (smoothing + binning)...")
sessions_data = []
for sub in SUBJECTS:
    smooth_tensor, cell_labels, neuron_names, beh, ok = load_session_trials(sub)
    if not ok:
        print(f"  s{sub}: no data, skipping.")
        continue
    sessions_data.append({
        'sub': sub,
        'smooth_tensor': smooth_tensor,
        'cell_labels':   cell_labels,
        'neuron_names':  neuron_names,
        'beh':           beh,
    })
    print(f"  s{sub}: {smooth_tensor.shape[0]} neurons, "
          f"{smooth_tensor.shape[1]} trials.")

print(f"Loaded {len(sessions_data)} sessions.")


# ══════════════════════════════════════════════════════════════════════
# ── Run RSA per ROI ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

print("\nRunning crossnobis RSA per ROI...")
roi_results  = {}
present_rois = []

for roi in rois_of_interest:
    compute_stab = (roi == 'ACC')
    data_vec, n_sessions, n_neurons, stab = group_data_rdm_for_roi(
        roi, sessions_data, compute_stability=compute_stab,
        plot_first=PLOT_FIGS)
    if data_vec is None:
        print(f"  {roi}: no valid sessions, skipping.")
        continue

    res = run_rsa(data_vec)
    res['n_neurons'] = n_neurons
    res['n_sessions'] = n_sessions
    res['stability']  = stab
    roi_results[roi]  = res
    present_rois.append(roi)
    print(f"  {roi:18s} (n_sessions={n_sessions}, n_neurons={n_neurons}) — "
          + ', '.join(f"{m}={res['unique'][m][0]:.2f}" for m in models))


# ══════════════════════════════════════════════════════════════════════
# ── ACC stability plot ────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

if 'ACC' in roi_results and roi_results['ACC']['stability'] is not None:
    stab = roi_results['ACC']['stability']    # (N_CONFIGS, N_CONDS_PER_CONF)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)

    ax = axes[0]
    vmax = np.nanmax(np.abs(stab)) or 1.0
    im = ax.imshow(stab, aspect='auto', cmap='coolwarm',
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(N_CONFIGS))
    ax.set_yticklabels(configs, fontsize=8)
    ax.set_xlabel('Timebin (within config)')
    ax.set_title('ACC — cross-validated pattern reliability\n(near zero → not stable)')
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    for c_idx in range(N_CONFIGS):
        ax.plot(stab[c_idx], label=configs[c_idx], lw=1.0)
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Timebin'); ax.set_ylabel('x-validated A·Σ⁻¹·B')
    ax.set_title('ACC — per-task stability across runs')
    ax.legend(fontsize=7, ncol=2)

    save = os.path.join(OUT_DIR, 'ACC_stability.png')
    fig.savefig(save, dpi=150, bbox_inches='tight')
    print(f"  ACC stability plot → {save}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════
# ── Summary heatmap ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

fig_unique, fig_combined = mc.plotting.cell_results.plot_rsa_heatmap(
    results      = roi_results,
    models       = models,
    combo_models = combo_models,
    rois         = present_rois,
    title        = 'DSR RSA (crossnobis) — beta per model × ROI',
    save_path    = os.path.join(OUT_DIR, 'DSR_RSA_crossnobis_heatmap.png'),
)
plt.show()

print('\nAll done.')
