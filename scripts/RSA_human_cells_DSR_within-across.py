#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSR RSA — 3-run-averaged variant; splits similarity into 'within config
block' vs 'between config blocks'.

Pipeline
--------
1. Load raw neurons + beh per session.
2. Smooth + bin trials, group trials by (config, run) via grid_no blocks.
3. Require K_RUNS runs for every config (session-level); build tensor
     X_sess[cfg, run, tb, neuron] (mean across correct repeats).
4. Per-neuron QC on X_sess:
     (a) cross-run reliability  r_rel  = corr(pattern_run0, pattern_mean(runs>=1))
     (b) drift across runs      r_drift = corr(pattern_run0, pattern_last_run)
     Neurons failing either threshold are dropped.  Diagnostics are printed
     and saved as CSV.
5. Average across runs → X_sess_avg[cfg, tb, neuron].
6. Per ROI (and whole brain):
     - pool surviving neurons across contributing sessions,
     - build mode-location from those contributing sessions only (so route
         differences between sessions don't contaminate model RDMs),
     - build 'within' / 'between' model RDMs (block_size = N_CONDS_PER_CONF),
     - build 1-Pearson data RDM,
     - split upper triangle into within-block vs between-block vectors,
     - run RSA (unique + combined) on each vector,
     - report Pearson r between within-block and between-block data vectors.
7. Save per-ROI bar plots + two summary heatmaps (within / between).
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
import math

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
# import pdb; pdb.set_trace()

# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR     = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_DIR      = os.path.join(DATA_DIR, 'group', 'DSR_RSA_within-across')
INCLUDE_DIAG = False
PLOT_FIGS    = True

_GAUSS_SIGMA    = 2
_GAUSS_TRUNCATE = 2.0

# Only sessions with ≥ K_RUNS per config contribute.  All their neurons
# are used — no per-neuron quality filtering.
K_RUNS = 'take_all' # 3 # 'take_all' # value between 1 and 4, or 'take_all'

# Model-RDM resolution (must match RSA_human_cells_DSR.py)
N_PHASES              = 3
states                = ['A', 'B', 'C', 'D']
LEN_STANDARDISED_PATH = 10
RESOLUTIONx           = 2
N_CONDS_PER_CONF      = N_PHASES * len(states) * RESOLUTIONx

N_PERMUTATIONS = 3 # None or n e.g. 300 permutations

rois_of_interest = ['whole_brain', 'OFC', 'EC', 'ACC', 'HC', 'PCC', 'AMY',
                    'OCCIP']

models           = ['dsr', 'now_and_next','state', 'feedback',
                     'midnight', 'location', 'phase',  'phase_state']

combo_models     = ['feedback', 'dsr', 'midnight', 'location']
#combo_models     = ['state', 'feedback', 'dsr', 'phase', 'midnight', 'location']



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
# ── Trial-level helpers ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _smooth_bin_axis(x, axis=-1):
    """Vectorised gaussian smooth + bin to N_CONDS_PER_CONF along `axis`."""
    sm = gaussian_filter1d(x, sigma=_GAUSS_SIGMA,
                           truncate=_GAUSS_TRUNCATE, axis=axis)
    sm = np.moveaxis(sm, axis, -1)
    n_in  = sm.shape[-1]
    edges = np.linspace(0, n_in, N_CONDS_PER_CONF + 1, dtype=int)
    out   = np.empty(sm.shape[:-1] + (N_CONDS_PER_CONF,), dtype=float)
    for i in range(N_CONDS_PER_CONF):
        chunk = sm[..., edges[i]:edges[i+1]]
        if chunk.shape[-1] == 0:
            out[..., i] = np.nan
        else:
            out[..., i] = np.nanmean(chunk, axis=-1)
    return np.moveaxis(out, -1, axis)


def _zscore_per_neuron(x, neuron_axis=0):
    """Z-score each neuron across all other axes (vectorised)."""
    x  = x.astype(float, copy=True)
    other = tuple(i for i in range(x.ndim) if i != neuron_axis)
    mu = np.nanmean(x, axis=other, keepdims=True)
    sd = np.nanstd(x,  axis=other, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return (x - mu) / sd


def _stack_raw_neurons(neurons_raw, neuron_names, n_trials):
    """Build raw[n_neurons, n_trials, n_bins_raw] (NaN-padded across trials)."""
    arrs   = []
    n_bins = None
    for name in neuron_names:
        a = neurons_raw[name].to_numpy(dtype=float)
        if n_bins is None:
            n_bins = a.shape[1]
        if a.shape[0] < n_trials:
            pad = np.full((n_trials - a.shape[0], a.shape[1]), np.nan)
            a   = np.vstack([a, pad])
        elif a.shape[0] > n_trials:
            a = a[:n_trials]
        arrs.append(a)
    return np.stack(arrs, axis=0), n_bins


def load_session_trials(sub_str, n_perms=None):
    """
    Returns smooth (n_neurons, n_trials, N_CONDS_PER_CONF) [z-scored],
            cell_labels, neuron_names, beh, locs, ok flag,
            smooth_perm (n_neurons, n_trials, N_CONDS_PER_CONF, n_perms)
            or None.  Each perm circularly shifts each (neuron, trial) raw
            row by an independent random offset, then runs the same
            vectorised smooth+bin+z-score pipeline.
    """
    data_dict = mc.analyse.helpers_human_cells.load_norm_data(DATA_DIR, [sub_str])
    key = f'sub-{sub_str}'
    if key not in data_dict:
        return None, None, None, None, None, False, None

    beh = data_dict[key]['beh'].copy().reset_index(drop=True)
    neurons_raw = data_dict[key]['normalised_neurons']
    # cell_labels = data_dict[key]['cell_labels']
    locs        = data_dict[key]['locations']
    locs = data_dict[key]['locations'].copy().reset_index(drop=True) 

    if not neurons_raw:
        return None, None, None, None, None, False, None

    neuron_names = sorted(neurons_raw.keys())
    n_neurons    = len(neuron_names)
    n_trials     = len(beh)

    # import pdb; pdb.set_trace() 
    raw, n_bins_raw = _stack_raw_neurons(neurons_raw, neuron_names, n_trials)

    smooth = _zscore_per_neuron(_smooth_bin_axis(raw, axis=-1), neuron_axis=0)

    smooth_perm = None
    
    # # this is randomly rotating each neuron, across configs.
    # if n_perms:
    #     rng = np.random.default_rng()
    #     n_flat = n_trials * n_bins_raw
    #     flat_axis = np.arange(n_flat)
    #     n_axis = np.arange(n_neurons)[:, None]
        
    #     smooth_perm = np.full((n_neurons, n_trials, N_CONDS_PER_CONF, n_perms), np.nan)
    #     # flatten trials and bins: (8, 337, 24) -> (8, 337*24)
    #     raw_flat = raw.reshape(n_neurons, n_flat)
    #     for p in range(n_perms):
    #         # one circular shift per neuron
    #         ks = rng.integers(0, n_flat, size=(n_neurons, 1))
    #         # shifted flattened indices
    #         shift_flat = (flat_axis[None, :] + ks) % n_flat
    #         # apply circular shift: shape (n_neurons, n_flat)
    #         raw_p_flat = raw_flat[n_axis, shift_flat]
    #         # reshape back: (8, 337*24) -> (8, 337, 24)
    #         raw_p = raw_p_flat.reshape(n_neurons, n_trials, n_bins_raw)
    #         smooth_perm[..., p] = _zscore_per_neuron(_smooth_bin_axis(raw_p, axis=-1),neuron_axis=0)
    
    # permuted data stays in the same config/run/trial bucket, but the within-trial temporal structure is scrambled
    if n_perms:
        # import pdb; pdb.set_trace() 
        rng     = np.random.default_rng()
        b_axis  = np.arange(n_bins_raw)
        n_axis  = np.arange(n_neurons)[:, None, None]
        t_axis  = np.arange(n_trials)[None, :, None]
        smooth_perm = np.full(
            (n_neurons, n_trials, N_CONDS_PER_CONF, n_perms), np.nan)
        for p in range(n_perms):
            # generates the shifts for all neurons and trials at the same time
            ks       = rng.integers(0, n_bins_raw, size=(n_neurons, n_trials))
            # circular shift of bin indices for all neurons and trials
            shift_b  = (b_axis[None, None, :] + ks[:, :, None]) % n_bins_raw
            # create the shift
            raw_p    = raw[n_axis, t_axis, shift_b]
            smooth_perm[..., p] = _zscore_per_neuron(
                _smooth_bin_axis(raw_p, axis=-1), neuron_axis=0)

    beh['config'] = list(zip(
        beh['loc_A'].astype(int), beh['loc_B'].astype(int),
        beh['loc_C'].astype(int), beh['loc_D'].astype(int),
    ))
    beh['grid_no']    = beh['grid_no'].astype(int)
    beh['config_str'] = beh['config'].apply(
        lambda t: f'{t[0]}-{t[1]}-{t[2]}-{t[3]}')

    # if len(cell_labels) < n_neurons:
    #     cell_labels = list(cell_labels) + ['UNKNOWN'] * (n_neurons - len(cell_labels))
    # else:
    #     cell_labels = list(cell_labels[:n_neurons])

    return smooth, neuron_names, beh, locs, True, smooth_perm
    # return smooth, cell_labels, neuron_names, beh, locs, True, smooth_perm


def _build_run_indices(beh, smooth, keep_idx):
    """groups[c][r] = ndarray of valid trial indices into smooth's trial axis.
    A trial is valid if no NaN appears across (kept neurons × time bins)."""
    groups = {}
    for c_idx, cfg_tup in enumerate(DSR_CONFIGS_TUP):
        mask    = (beh['config'] == cfg_tup) & (beh['correct'] == 1)
        sub_beh = beh[mask]
        blocks  = sorted(sub_beh['grid_no'].unique().tolist())
        runs    = {}
        for r_idx, blk in enumerate(blocks):
            tr = sub_beh[sub_beh['grid_no'] == blk].index.tolist()
            tr = np.asarray(
                [t for t in tr if t < smooth.shape[1]], dtype=int)
            if tr.size == 0:
                continue
            sub   = smooth[np.ix_(keep_idx, tr)]   # (n_kept, n_reps, n_tb)
            valid = ~np.isnan(sub).any(axis=(0, 2))
            if not valid.any():
                continue
            runs[r_idx] = tr[valid]
        groups[c_idx] = runs
    return groups


def _session_tensor(smooth, groups, keep_idx, K):
    """smooth : (n_neurons, n_trials, n_tb[, n_perms]).
    Returns X (N_CONFIGS, K, n_tb, n_kept[, n_perms]) — mean across reps —
    or None if any cfg has fewer than K runs."""
    run_counts = [len(groups.get(c, {})) for c in range(N_CONFIGS)]

    extra = smooth.shape[3:]
    n_tb  = smooth.shape[2]
    sub_smooth = smooth[keep_idx]
    
    # import pdb; pdb.set_trace() 
    if K == 'take_all':
        X_avg = np.full((N_CONFIGS, n_tb, len(keep_idx)) + extra, np.nan)
        for c in range(N_CONFIGS):
            all_reps = np.concatenate(list(groups[c].values()))
            reps = sub_smooth[:, all_reps]
            X_avg[c] = np.moveaxis(reps.mean(axis=1), 0,1)     

    else:
        if not run_counts or min(run_counts) < K:
            return None
        X = np.full((N_CONFIGS, K, n_tb, len(keep_idx)) + extra, np.nan)
        for c in range(N_CONFIGS):
            for k in range(K):
                tr = groups[c][k] # indices for one config run
                reps = sub_smooth[:, tr]  # (neurons, n_reps, n_taskbins)
                X[c, k] = np.moveaxis(reps.mean(axis=1), 0, 1) # config, runs, bins, neurons
        X_avg = X.mean(axis=1) # configs, bins, neurons
    
    # return an average across sessions
    return X_avg


# ══════════════════════════════════════════════════════════════════════
# ── Mode-locations per subject subset ────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_mode_locs(sub_locs_per_config):
    """
    sub_locs_per_config[c] is a list of (n_trials, n_bins) arrays to stack.
    Returns dict c -> (n_bins,) mode array (NaN if empty).
    """
    mode_locs = {}
    for c, parts in sub_locs_per_config.items():
        if not parts:
            mode_locs[c] = None
            continue
        stacked = np.vstack(parts)
        m = stats.mode(stacked, axis=0, keepdims=False, nan_policy='omit')
        mode_locs[c] = m.mode.astype(float)
    return mode_locs


def _accum_sub_locs(sub, beh_sub, locs_sub):
    """For one subject, return {cfg_str: [locs_array for that config, correct only]}."""
    out = {c: [] for c in configs}
    if beh_sub is None or locs_sub is None:
        return out
    for c in configs:
        mask = (beh_sub['config_str'] == c) & (beh_sub['correct'] == 1)
        idx  = beh_sub.index[mask]
        if len(idx) == 0:
            continue
        arr = locs_sub.loc[idx].values.astype(float) if hasattr(locs_sub, 'loc') \
              else locs_sub[idx]
        out[c].append(arr)
    return out


# ══════════════════════════════════════════════════════════════════════
# ── Model RDMs from a mode-locations dict ────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def downsample_mode(x, target_len=10):
    x = np.asarray(x)
    block = len(x) // target_len
    return np.array([
        stats.mode(x[i*block:(i+1)*block], keepdims=False).mode
        for i in range(target_len)
    ])


def build_model_rdms(mode_locs_single):
    """
    mode_locs_single[c] = (n_bins,) single averaged mode vector per config
    (NaN-filled fallback if no contributing sessions).
    Returns (model_RDMs_within, model_RDMs_across) each dict model->np.array.
    """
    # fall back to a trivial ramp if any config is missing (unlikely)
    N_BINS = None
    for v in mode_locs_single.values():
        if v is not None:
            N_BINS = len(v)
            break
    if N_BINS is None:
        raise RuntimeError('No mode-locations available for this subset.')

    loc_th = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, LEN_STANDARDISED_PATH), dtype=float)
    dsr_th = np.zeros((N_CONFIGS * N_CONDS_PER_CONF,
                       LEN_STANDARDISED_PATH * N_PHASES * len(states)), dtype=float)

    for c_idx, c in enumerate(configs):
        mode_vec = mode_locs_single.get(c)
        task_config = [int(int(x) - 1) for x in c.split('-')]
        if mode_vec is None or np.all(np.isnan(mode_vec)):
            # cheap fallback: repeat first location across bins
            mode_vec = np.full(N_BINS, float(task_config[0] + 1))

        LEN_OG_SUBPATH = int(len(mode_vec) / N_CONDS_PER_CONF)
        dsr_first_step = downsample_mode(
            mode_vec, target_len=LEN_STANDARDISED_PATH * N_PHASES * len(states))
        row_curr = c_idx * N_CONDS_PER_CONF
        for n_sp in range(N_CONDS_PER_CONF):
            sp = mode_vec[n_sp*LEN_OG_SUBPATH:(n_sp+1)*LEN_OG_SUBPATH]
            loc_th[row_curr + n_sp, :] = downsample_mode(
                sp, target_len=LEN_STANDARDISED_PATH)
            dsr_th[row_curr + n_sp, :] = np.roll(
                dsr_first_step, -n_sp * LEN_STANDARDISED_PATH)

    state_config    = np.zeros((N_CONDS_PER_CONF, len(states)))
    feedback_config = np.zeros((N_CONDS_PER_CONF, len(states)))
    phase_config    = np.zeros((N_CONDS_PER_CONF, N_PHASES))
    for s_i, s in enumerate(states):
        start = RESOLUTIONx * s_i * N_PHASES
        state_config[start: RESOLUTIONx * (s_i + 1) * N_PHASES, s_i] = 1
        if s == 'A':
            feedback_config[0:RESOLUTIONx, s_i] = 1
        if s == 'D':
            feedback_config[-RESOLUTIONx:, s_i] = 1
        for p_i in range(N_PHASES):
            phase_config[start + p_i*RESOLUTIONx:
                         start + (p_i+1)*RESOLUTIONx, p_i] = 1
    # import pdb; pdb.set_trace()
    feedback_half = np.tile(feedback_config, (len(configs), 1))
    state_half    = np.tile(state_config,    (len(configs), 1))
    phase_half    = np.tile(phase_config,    (len(configs), 1))

    _n_clock_neurons = 9 * N_PHASES * (N_PHASES * len(states))
    dsr_mat   = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, _n_clock_neurons),    dtype=float)
    midnight_mat = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, 9 * N_PHASES),        dtype=float)
    loc_og_mat   = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, 9),                   dtype=float)
    phase_og_mat = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, N_PHASES),            dtype=float)
    state_og_mat = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, len(states)),                   dtype=float)
    now_and_next_mat = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, _n_clock_neurons), dtype=float)
    phas_stat_mat = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, N_PHASES * len(states)), dtype=float)
    

    for c_idx, c in enumerate(configs):
        row_start   = c_idx * N_CONDS_PER_CONF
        task_config = [int(int(x) - 1) for x in c.split('-')]
        walked   = mode_locs_single.get(c)
        walked = [int(w-1) for w in walked]

        loc_og_matrix, phase_og_matrix, stat_matrix, midnight_matrix, dsr_matrix, phas_stat_matrix, clo_model_subpath = mc.simulation.predictions.model_DSR(locations = walked, no_phase_neurons=N_PHASES)
        
        
        for n_sp in range(N_CONDS_PER_CONF):
            t0 = n_sp * LEN_OG_SUBPATH
            t1 = (n_sp + 1) * LEN_OG_SUBPATH
            for dst, src in [(dsr_mat,   dsr_matrix),
                             (midnight_mat, midnight_matrix),
                             (loc_og_mat,   loc_og_matrix),
                             (phase_og_mat, phase_og_matrix),
                             (state_og_mat, stat_matrix),
                             (now_and_next_mat, clo_model_subpath),
                             (phas_stat_mat, phas_stat_matrix)
                             ]:
                sub_sp = np.nanmean(src[:, t0:t1], axis=1)
                dst[row_start + n_sp, :] = np.where(np.isnan(sub_sp), 0.0, sub_sp)

    model_concat = {
        'loc_hamming': loc_th,
        'dsr_hamming': dsr_th,
        'state':    state_half,
        'feedback': feedback_half,
        'dsr':   dsr_mat,
        'location':   loc_og_mat,
        'midnight': midnight_mat,
        'phase':    phase_half,
        'phase_og': phase_og_mat,
        'state_og': state_og_mat,
        'now_and_next': now_and_next_mat,
        'phase_state': phas_stat_mat
    }

    within, across, full = {}, {}, {}
    for m in models:
        if m in ('loc_hamming', 'dsr_hamming'):
            w, a, full[m] = mc.analyse.my_RSA.compute_hamming_distance_within(
                model_concat[m], plotting=False,
                include_diagonal=INCLUDE_DIAG,
                model_name=m, no_tasks=len(configs),
                block_size=N_CONDS_PER_CONF)
        else:
            w, a, full[m] = mc.analyse.my_RSA.compute_crosscorr_within(
                model_concat[m], plotting=False,
                include_diagonal=INCLUDE_DIAG,
                no_tasks=len(configs), model=m,
                block_size=N_CONDS_PER_CONF)
        within[m] = np.asarray(w[0], dtype=float)
        across[m] = np.asarray(a[0], dtype=float)

    # feedback: NaN → 1 (same convention as original script)
    within['feedback'] = np.where(np.isnan(within['feedback']), 1.0, within['feedback'])
    across['feedback'] = np.where(np.isnan(across['feedback']), 1.0, across['feedback'])
    full['feedback'] = np.where(np.isnan(full['feedback']), 1.0, full['feedback'])
    
    return within, across, full


# ══════════════════════════════════════════════════════════════════════
# ── RDM plotting helpers ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _apply_block_mask(mat, mode, block_size):
    """Return float copy with cells outside the requested region set to NaN.
    mode='within': keep on-block-diagonal cells (excluding main diagonal).
    mode='between': keep off-block-diagonal cells (between-config pairs)."""
    n = mat.shape[0]
    ii, jj = np.indices((n, n))
    same_block = (ii // block_size) == (jj // block_size)
    out = mat.astype(float).copy()
    if mode == 'within':
        out[~same_block] = np.nan
        np.fill_diagonal(out, np.nan)
    else:
        out[same_block] = np.nan
    return out


def _add_block_lines(ax, n):
    for p in range(0, n, N_CONDS_PER_CONF):
        ax.axvline(p - 0.5, color='white', ls='dashed', lw=0.4)
        ax.axhline(p - 0.5, color='white', ls='dashed', lw=0.4)


def _roi_keep_idx(cell_labels, roi_filter):
    # import pdb; pdb.set_trace() 

    if roi_filter == 'whole_brain':
        return np.arange(len(cell_labels))
    roi_list = []
    for cl in cell_labels:
        roi = mc.analyse.helpers_human_cells.convert_cell_label_in_roi(cl)
        roi_list.append(roi)
    # import pdb; pdb.set_trace() 
    return np.array([i for i, lb in enumerate(roi_list) if lb == roi_filter])


# ══════════════════════════════════════════════════════════════════════
# ── RSA runner ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _scalar(a):
    return float(np.asarray(a, dtype=float).ravel()[0])


def run_rsa(data_vec, model_RDMs_vec):
    """Unique regression per model + combined regression (mirrors DSR script)."""
    unique = {}
    for m in models:
        print(m)
        res = mc.analyse.my_RSA.evaluate_model(model_RDMs_vec[m], data_vec)
        unique[m] = (_scalar(res[0]), _scalar(res[1]), _scalar(res[2]))

    stacked = np.stack([model_RDMs_vec[m] for m in combo_models], axis=1)
    res_c = mc.analyse.my_RSA.evaluate_model(stacked, data_vec)
    combined = {
        't':    np.asarray(res_c[0], dtype=float).ravel(),
        'beta': np.asarray(res_c[1], dtype=float).ravel(),
        'p':    np.asarray(res_c[2], dtype=float).ravel(),
    }
    return {'unique': unique, 'combined': combined}


# ══════════════════════════════════════════════════════════════════════
# ── Load sessions (trials + locations) ───────────────────────────────
# ══════════════════════════════════════════════════════════════════════

print('\nLoading sessions...')
sessions_data = []
for sub in SUBJECTS:
    smooth, names, beh, locs, ok, smooth_perm = (
        load_session_trials(sub, n_perms=N_PERMUTATIONS))
    if not ok:
        print(f'  s{sub}: no data.')
        continue

    keep_all = np.arange(smooth.shape[0])
    groups   = _build_run_indices(beh, smooth, keep_all)
    X_avg   = _session_tensor(smooth, groups, keep_all, K_RUNS)
    if X_avg is None:
        print(f'  s{sub}: <{K_RUNS} runs per config, skipping session.')
        continue

    # average across runs → (N_CONFIGS, N_CONDS_PER_CONF, n_neurons)
    # X_avg      = X_sess.mean(axis=1)
    X_avg_perm = None
    if smooth_perm is not None:
        X_avg_perm = _session_tensor(smooth_perm, groups, keep_all, K_RUNS)

    n_neurons = int(smooth.shape[0])
    sessions_data.append({
        'sub':          sub,
        'X_avg':        X_avg,
        'X_avg_perm':   X_avg_perm,
        'neuron_names': list(names),
        'beh':          beh,
        'locs':         locs,
        'n_neurons':    n_neurons,
    })
    msg_perm = (f', perms={X_avg_perm.shape[-1]}'
                if X_avg_perm is not None else '')
    print(f'  s{sub}: {n_neurons} neurons, {K_RUNS}+ runs/config{msg_perm}.')

print(f'Loaded {len(sessions_data)} sessions.')


# ══════════════════════════════════════════════════════════════════════
# ── Per-subject location accumulators (for per-ROI mode-locs) ────────
# ══════════════════════════════════════════════════════════════════════

print('\nBuilding per-subject location accumulators...')
sub_loc_accum = {}
for sd in sessions_data:
    sub_loc_accum[sd['sub']] = _accum_sub_locs(sd['sub'], sd['beh'], sd['locs'])


# ══════════════════════════════════════════════════════════════════════
# ── Per ROI: pool neurons, build model + data RDMs, run RSA ──────────
# ══════════════════════════════════════════════════════════════════════

def _pool_roi(roi):
    """
    Pool averaged-across-runs patterns for neurons with label == roi.
    Returns
        X_pool       (n_cond, n_neurons_total),
        X_pool_perm  (n_cond, n_neurons_total, n_perms) or None,
        contributing_subs (list[str]),
        n_neurons_per_sub (dict).
    """
    parts_emp, parts_perm = [], []
    contrib_subs = []
    per_sub_count = {}
    for sd in sessions_data:
        if 'X_avg' not in sd:
            continue
        keep = _roi_keep_idx(sd['neuron_names'], roi)
        if len(keep) == 0:
            continue
        X    = sd['X_avg'][..., keep]                     # (cfg, tb, n_kept)
        patt = X.reshape(N_CONFIGS * N_CONDS_PER_CONF, -1)
        parts_emp.append(patt)

        if sd.get('X_avg_perm') is not None:
            Xp = sd['X_avg_perm'][..., keep, :]            # (cfg, tb, n_kept, n_perms)
            parts_perm.append(
                Xp.reshape(N_CONFIGS * N_CONDS_PER_CONF, -1, Xp.shape[-1]))

        contrib_subs.append(sd['sub'])
        per_sub_count[sd['sub']] = int(X.shape[-1])

    if not parts_emp:
        return None, None, [], {}
    X_pool      = np.concatenate(parts_emp, axis=1)
    X_pool_perm = (np.concatenate(parts_perm, axis=1)
                   if parts_perm else None)
    return X_pool, X_pool_perm, contrib_subs, per_sub_count


def _mode_locs_for_subs(sub_list):
    collected = {c: [] for c in configs}
    for sub in sub_list:
        sub_acc = sub_loc_accum.get(sub, {})
        for c in configs:
            collected[c].extend(sub_acc.get(c, []))
    return build_mode_locs(collected)


def _data_rdm_split(X):
    """Run 1-Pearson RDM + within/between split for a single (n_cond, n_neurons)."""
    d_w, d_a, full = mc.analyse.my_RSA.compute_crosscorr_within(
        X, plotting=False,
        include_diagonal=INCLUDE_DIAG,
        no_tasks=len(configs), model='data',
        block_size=N_CONDS_PER_CONF)
    return (np.asarray(d_w[0], dtype=float),
            np.asarray(d_a[0], dtype=float),
            full)


def _perm_betas(X_pool_perm, model_within, model_across):
    """Run RSA on each perm.  Returns dicts of arrays keyed by mode/model
    plus combined-regression beta arrays."""
    n_perms = X_pool_perm.shape[-1]
    betas_w = {m: np.full(n_perms, np.nan) for m in models}
    betas_a = {m: np.full(n_perms, np.nan) for m in models}
    combo_w = np.full((n_perms, len(combo_models)), np.nan)
    combo_a = np.full((n_perms, len(combo_models)), np.nan)

    stacked_w = np.stack([model_within[m] for m in combo_models], axis=1)
    stacked_a = np.stack([model_across[m] for m in combo_models], axis=1)

    for p in range(n_perms):
        dwv, dav, _ = _data_rdm_split(X_pool_perm[..., p])
        for m in models:
            # import pdb; pdb.set_trace() 
            _, bw, _ = mc.analyse.my_RSA.evaluate_model(model_within[m], dwv)
            _, ba, _ = mc.analyse.my_RSA.evaluate_model(model_across[m], dav)
            betas_w[m][p] = _scalar(bw)
            betas_a[m][p] = _scalar(ba)
        _, bw_c, _ = mc.analyse.my_RSA.evaluate_model(stacked_w, dwv)
        _, ba_c, _ = mc.analyse.my_RSA.evaluate_model(stacked_a, dav)
        combo_w[p] = np.asarray(bw_c, dtype=float).ravel()
        combo_a[p] = np.asarray(ba_c, dtype=float).ravel()
    return betas_w, betas_a, combo_w, combo_a


def _two_sided_perm_p(empirical, perm_dist):
    perm_dist = np.asarray(perm_dist, dtype=float)
    perm_dist = perm_dist[np.isfinite(perm_dist)]
    if perm_dist.size == 0 or not np.isfinite(empirical):
        return np.nan
    return float((np.abs(perm_dist) >= np.abs(empirical)).mean())


print('\nRunning RSA per ROI...')
roi_results = {}

for roi in rois_of_interest:
    X_pool, X_pool_perm, contrib_subs, per_sub_count = _pool_roi(roi)
    if X_pool is None:
        print(f'  {roi}: no neurons, skipping.')
        continue

    # --- per-ROI model RDMs (from contributing sessions only) ------------
    mode_locs_roi = _mode_locs_for_subs(contrib_subs)
    model_within, model_across, model_full = build_model_rdms(mode_locs_roi)

    # --- empirical data RDM ---------------------------------------------
    data_within, data_across, data_rdm = _data_rdm_split(X_pool)

    # diagnostic correlation (subsample data_across to match len)
    corr_wb = float(np.corrcoef(
        data_within, data_across[:len(data_within)])[0, 1])

    res_within = run_rsa(data_within, model_within)
    res_across = run_rsa(data_across, model_across)

    n_neurons_total = int(X_pool.shape[1])
    res_within['n_neurons']  = n_neurons_total
    res_across['n_neurons']  = n_neurons_total
    res_within['n_sessions'] = len(contrib_subs)
    res_across['n_sessions'] = len(contrib_subs)

    # --- permutation RSA -------------------------------------------------
    perm_w_betas = perm_a_betas = None
    perm_w_combo = perm_a_combo = None
    perm_p_w_unique = perm_p_a_unique = None
    perm_p_w_combo  = perm_p_a_combo  = None
    if X_pool_perm is not None:
        perm_w_betas, perm_a_betas, perm_w_combo, perm_a_combo = _perm_betas(
            X_pool_perm, model_within, model_across)
        perm_p_w_unique = {
            m: _two_sided_perm_p(res_within['unique'][m][1], perm_w_betas[m])
            for m in models}
        perm_p_a_unique = {
            m: _two_sided_perm_p(res_across['unique'][m][1], perm_a_betas[m])
            for m in models}
        perm_p_w_combo = {
            cm: _two_sided_perm_p(
                res_within['combined']['beta'][i], perm_w_combo[:, i])
            for i, cm in enumerate(combo_models)}
        perm_p_a_combo = {
            cm: _two_sided_perm_p(
                res_across['combined']['beta'][i], perm_a_combo[:, i])
            for i, cm in enumerate(combo_models)}

    roi_results[roi] = {
        'within':              res_within,
        'across':              res_across,
        'n_neurons':           n_neurons_total,
        'n_sessions':          len(contrib_subs),
        'contrib_subs':        contrib_subs,
        'per_sub_count':       per_sub_count,
        'data_rdm':            data_rdm,
        'model_full':          model_full,
        'corr_within_across':  corr_wb,
        'perm_within_betas':   perm_w_betas,
        'perm_across_betas':   perm_a_betas,
        'perm_within_combo':   perm_w_combo,
        'perm_across_combo':   perm_a_combo,
        'perm_p_within':       perm_p_w_unique,
        'perm_p_across':       perm_p_a_unique,
        'perm_p_within_combo': perm_p_w_combo,
        'perm_p_across_combo': perm_p_a_combo,
    }

    print(f'  {roi:18s} n_neurons={n_neurons_total:4d} '
          f'n_sess={len(contrib_subs):2d}  '
          f'(within: ' + ', '.join(
              f"{m}={res_within['unique'][m][1]:.2f}" for m in models) + ')')
    print(f'  {roi:18s} '
          f'(across: ' + ', '.join(
              f"{m}={res_across['unique'][m][1]:.2f}" for m in models) + ')')




# ══════════════════════════════════════════════════════════════════════
# ── Block-diagonal vs between-block RDM overview figures ─────────────
# ══════════════════════════════════════════════════════════════════════



def _rdm_grid_figure(mode, save_path_prefix):
    """
    mode ∈ {'within','between'}

    Creates TWO figures:
    1. Data RDMs per ROI
    2. Model RDMs

    Max 4 matrices per row.
    """
    present = list(roi_results.keys())
    if not present:
        return

    ref_roi = 'whole_brain' if 'whole_brain' in roi_results else present[0]
    model_full = roi_results[ref_roi]['model_full']

    max_cols = 4

    # =========================
    # ---- DATA RDM FIGURE ----
    # =========================
    n_data = len(present)
    n_cols = min(max_cols, n_data)
    n_rows = math.ceil(n_data / max_cols)

    fig_data, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3 * n_cols, 3 * n_rows),
                                 constrained_layout=True)

    axes = np.atleast_2d(axes)

    for i, roi in enumerate(present):
        r = i // max_cols
        c = i % max_cols
        ax = axes[r, c]

        mat = _apply_block_mask(roi_results[roi]['data_rdm'],
                               mode, N_CONDS_PER_CONF)

        im = ax.imshow(mat, aspect='auto', cmap='coolwarm',
                       vmin=0.7, vmax=1.3)

        ax.set_title(f'{roi} (n={roi_results[roi]["n_neurons"]})',
                     fontsize=8)
        _add_block_lines(ax, mat.shape[0])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.tick_params(labelsize=6)

    # hide unused axes
    for j in range(n_data, n_rows * n_cols):
        r = j // max_cols
        c = j % max_cols
        axes[r, c].set_visible(False)

    title = ('Within-config block RDMs'
             if mode == 'within'
             else 'Between-config block RDMs')

    fig_data.suptitle(title + ' — DATA', fontsize=12, weight='bold')
    fig_data.savefig(f'{save_path_prefix}_data.png',
                     dpi=150, bbox_inches='tight')


    # ==========================
    # ---- MODEL RDM FIGURE ----
    # ==========================
    n_mdl = len(models)
    n_cols = min(max_cols, n_mdl)
    n_rows = math.ceil(n_mdl / max_cols)

    fig_model, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(3 * n_cols, 3 * n_rows),
                                  constrained_layout=True)

    axes = np.atleast_2d(axes)

    for i, m in enumerate(models):
        r = i // max_cols
        c = i % max_cols
        ax = axes[r, c]

        mat = _apply_block_mask(model_full[m], mode, N_CONDS_PER_CONF)

        im = ax.imshow(mat, aspect='auto', cmap='coolwarm')

        ax.set_title(f'model: {m}', fontsize=8)
        _add_block_lines(ax, mat.shape[0])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.tick_params(labelsize=6)

    # hide unused axes
    for j in range(n_mdl, n_rows * n_cols):
        r = j // max_cols
        c = j % max_cols
        axes[r, c].set_visible(False)

    fig_model.suptitle(title + f' — MODELS (ROI={ref_roi})',
                       fontsize=12, weight='bold')
    fig_model.savefig(f'{save_path_prefix}_models.png',
                      dpi=150, bbox_inches='tight')


if roi_results:
    _rdm_grid_figure('within',
                     os.path.join(OUT_DIR, 'rdms_within_block_diagonal.png'))
    _rdm_grid_figure('between',
                     os.path.join(OUT_DIR, 'rdms_between_blocks.png'))
    print('Saved within-block + between-block RDM overview figures.')


# ══════════════════════════════════════════════════════════════════════
# ── Model × model correlation heatmap (within + between) ─────────────
# ══════════════════════════════════════════════════════════════════════

def _plot_model_corr_heatmap(save_path):
    present = list(roi_results.keys())
    if not present:
        return
    ref_roi = 'whole_brain' if 'whole_brain' in roi_results else present[0]
    model_full = roi_results[ref_roi]['model_full']

    n_m = len(models)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for ax, mode, title in zip(
        axes, ['within', 'across'],
        ['Within-config block', 'Between-config blocks'],
    ):
        vecs = {}
        for m in models:
            if mode == 'within':
                vecs[m] = model_within[m]
            else:
                vecs[m] = model_across[m]
        corr = np.zeros((n_m, n_m))
        for i, a in enumerate(models):
            for j, b in enumerate(models):
                va, vb = vecs[a], vecs[b]
                if np.nanstd(va) == 0 or np.nanstd(vb) == 0:
                    corr[i, j] = np.nan
                else:
                    corr[i, j] = np.corrcoef(va, vb)[0, 1]
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(n_m)); ax.set_yticks(range(n_m))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(models, fontsize=8)
        for i in range(n_m):
            for j in range(n_m):
                if np.isfinite(corr[i, j]):
                    ax.text(j, i, f'{corr[i,j]:.2f}',
                            ha='center', va='center', fontsize=7,
                            color='white' if abs(corr[i, j]) > 0.5 else 'black')
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f'Model RDM × Model RDM Pearson r   [from ROI={ref_roi}]',
                 fontsize=12, weight='bold')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')


if roi_results:
    _plot_model_corr_heatmap(
        os.path.join(OUT_DIR, 'model_rdm_correlations.png'))
    print('Saved model×model correlation heatmap.')


# ══════════════════════════════════════════════════════════════════════
# ── Permutation distribution figure ──────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _plot_perm_distribution(mode, save_path):
    """mode ∈ {'within','across'}.  Rows = ROIs, cols = models.  Each cell
    shows the histogram of permutation betas with the empirical beta as a
    vertical line and the two-sided perm p-value annotated."""
    rois_with_perm = [r for r, res in roi_results.items()
                      if res.get(f'perm_{mode}_betas') is not None]
    if not rois_with_perm:
        return

    n_rows = len(rois_with_perm)
    n_cols = len(models)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols + 1.0, 1.6 * n_rows + 1.0),
        constrained_layout=True, squeeze=False)

    beta_key = 'within' if mode == 'within' else 'across'
    p_key    = f'perm_p_{mode}'
    b_key    = f'perm_{mode}_betas'

    for i, roi in enumerate(rois_with_perm):
        res        = roi_results[roi]
        perm_betas = res[b_key]
        perm_p     = res[p_key] or {}

        for j, m in enumerate(models):
            ax  = axes[i, j]
            arr = np.asarray(perm_betas[m], dtype=float)
            arr = arr[np.isfinite(arr)]
            emp = float(res[beta_key]['unique'][m][1])

            if arr.size:
                ax.hist(arr, bins=min(20, max(3, arr.size // 2 + 1)),
                        color='#bdbdbd', edgecolor='none')
            ax.axvline(emp, color='#d62728', lw=1.4)
            ax.axvline(0, color='grey', linestyle = '-')

            pv = perm_p.get(m, np.nan)
            ax.set_title(f'{m}  p={pv:.2g}' if np.isfinite(pv) else m,
                         fontsize=7)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(
                    f'{roi}\n(n={res["n_neurons"]})', fontsize=7)
            if i < n_rows - 1:
                ax.set_xticklabels([])

    title = ('Within-config block — perm β distributions vs empirical'
             if mode == 'within'
             else 'Between-config blocks — perm β distributions vs empirical')
    fig.suptitle(title, fontsize=12, weight='bold')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')


if roi_results:
    _plot_perm_distribution(
        'within',
        os.path.join(OUT_DIR, 'perm_distribution_within.png'))
    _plot_perm_distribution(
        'across',
        os.path.join(OUT_DIR, 'perm_distribution_across.png'))
    print('Saved permutation distribution figures.')


# ══════════════════════════════════════════════════════════════════════
# ── Summary heatmaps (within + between) ──────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _flatten(metric_key):
    flat = {}
    for roi, res in roi_results.items():
        entry = dict(res[metric_key])
        entry['n_neurons']  = res['n_neurons']
        entry['n_sessions'] = res['n_sessions']
        flat[roi] = entry
    return flat


def _collect_perm_p(unique_key, combo_key):
    """Returns (perm_p_unique, perm_p_combined) dicts in the format expected
    by plot_rsa_heatmap, or (None, None) if no ROI has perm results."""
    pu = {r: roi_results[r][unique_key]
          for r in roi_results
          if roi_results[r].get(unique_key) is not None}
    pc = {r: roi_results[r][combo_key]
          for r in roi_results
          if roi_results[r].get(combo_key) is not None}
    return (pu or None, pc or None)


present_rois = list(roi_results.keys())
if present_rois:
    pu_w, pc_w = _collect_perm_p('perm_p_within', 'perm_p_within_combo')
    pu_a, pc_a = _collect_perm_p('perm_p_across', 'perm_p_across_combo')

    mc.plotting.cell_results.plot_rsa_heatmap(
        results        = _flatten('within'),
        models         = models,
        combo_models   = combo_models,
        rois           = present_rois,
        title          = 'DSR RSA — within-config block — beta per model × ROI',
        save_path      = os.path.join(OUT_DIR, 'DSR_RSA_within_heatmap.png'),
        perm_p_unique   = pu_w,
        perm_p_combined = pc_w,
    )
    mc.plotting.cell_results.plot_rsa_heatmap(
        results        = _flatten('across'),
        models         = models,
        combo_models   = combo_models,
        rois           = present_rois,
        title          = 'DSR RSA — between-config blocks — beta per model × ROI',
        save_path      = os.path.join(OUT_DIR, 'DSR_RSA_across_heatmap.png'),
        perm_p_unique   = pu_a,
        perm_p_combined = pc_a,
    )
    if PLOT_FIGS:
        plt.show()

print('\nAll done.')
