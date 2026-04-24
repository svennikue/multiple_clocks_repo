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

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')


# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR     = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_DIR      = os.path.join(DATA_DIR, 'group', 'DSR_RSA_within-across')
INCLUDE_DIAG = False
PLOT_FIGS    = True

_GAUSS_SIGMA    = 2
_GAUSS_TRUNCATE = 2.0

# Only sessions with ≥ K_RUNS per config contribute.  All their neurons
# are used — no per-neuron quality filtering.
K_RUNS = 3

# Model-RDM resolution (must match RSA_human_cells_DSR.py)
N_PHASES              = 3
states                = ['A', 'B', 'C', 'D']
LEN_STANDARDISED_PATH = 10
RESOLUTIONx           = 2
N_CONDS_PER_CONF      = N_PHASES * len(states) * RESOLUTIONx

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
# ── Trial-level helpers ───────────────────────────────────────────────
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


def _zscore_session_neurons(smooth_tensor):
    """Z-score each neuron across all trials × timebins in that session."""
    x = smooth_tensor.astype(float).copy()
    for n in range(x.shape[0]):
        mu = np.nanmean(x[n])
        sd = np.nanstd(x[n])
        if not np.isfinite(sd) or sd == 0:
            sd = 1.0
        x[n] = (x[n] - mu) / sd
    return x


def load_session_trials(sub_str):
    """
    Returns smooth_tensor (n_neurons, n_trials, N_CONDS_PER_CONF),
    cell_labels, neuron_names, beh (with 'config' tuple + int 'grid_no'),
    locations (n_trials, n_bins), ok flag.
    """
    data_dict = mc.analyse.helpers_human_cells.load_norm_data(DATA_DIR, [sub_str])
    key = f'sub-{sub_str}'
    if key not in data_dict:
        return None, None, None, None, None, False

    beh         = data_dict[key]['beh'].copy()
    neurons_raw = data_dict[key]['normalised_neurons']
    cell_labels = data_dict[key]['cell_labels']
    locs        = data_dict[key]['locations']
    if not neurons_raw:
        return None, None, None, None, None, False

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
    beh['config_str'] = beh['config'].apply(
        lambda t: f'{t[0]}-{t[1]}-{t[2]}-{t[3]}')

    if len(cell_labels) < n_neurons:
        cell_labels = list(cell_labels) + ['UNKNOWN'] * (n_neurons - len(cell_labels))
    else:
        cell_labels = list(cell_labels[:n_neurons])

    smooth_tensor = _zscore_session_neurons(smooth_tensor)
    return smooth_tensor, cell_labels, neuron_names, beh, locs, True


def _build_run_groups(beh, smooth_tensor, keep_idx):
    """
    Group trials by (cfg_idx, run_idx) where run_idx enumerates blocks of a
    config in ascending grid_no.  Returns
        groups[c_idx][r_idx] = ndarray (n_reps, n_kept_neurons, N_CONDS_PER_CONF)
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
            reps = np.transpose(reps, (1, 0, 2))   # (n_reps, n_neurons, n_tb)
            valid = ~np.isnan(reps).any(axis=(1, 2))
            if not valid.any():
                continue
            runs[r_idx] = reps[valid]
        groups[c_idx] = runs
    return groups


def _session_tensor(groups, K):
    """
    X[cfg, run, tb, neuron] = mean across correct repeats.
    Requires ≥K runs for every config.  Returns (X, n_neurons) or (None, 0).
    """
    run_counts = [len(groups.get(c, {})) for c in range(N_CONFIGS)]
    if not run_counts or min(run_counts) < K:
        return None, 0
    first = next(iter(next(iter(groups.values())).values()))
    n_neurons = first.shape[1]
    X = np.full((N_CONFIGS, K, N_CONDS_PER_CONF, n_neurons), np.nan)
    for c in range(N_CONFIGS):
        for k in range(K):
            X[c, k] = groups[c][k].mean(axis=0).T
    return X, n_neurons


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
        for p_i in range(N_PHASES):
            phase_config[start + p_i*RESOLUTIONx:
                         start + (p_i+1)*RESOLUTIONx, p_i] = 1

    feedback_half = np.tile(feedback_config, (len(configs), 1))
    state_half    = np.tile(state_config,    (len(configs), 1))
    phase_half    = np.tile(phase_config,    (len(configs), 1))

    _n_clock_neurons = 9 * N_PHASES * (N_PHASES * len(states))
    clocks_mat   = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, _n_clock_neurons), dtype=float)
    midnight_mat = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, 9 * N_PHASES),     dtype=float)
    loc_og_mat   = np.zeros((N_CONFIGS * N_CONDS_PER_CONF, 9),                dtype=float)

    for c_idx, c in enumerate(configs):
        row_start   = c_idx * N_CONDS_PER_CONF
        task_config = [int(int(x) - 1) for x in c.split('-')]
        curr_task   = mode_locs_single.get(c)
        if curr_task is None or np.all(np.isnan(curr_task)):
            curr_task = np.full(N_BINS, float(task_config[0] + 1))
        curr_task = curr_task.copy()
        for i in range(len(curr_task)):
            if np.isnan(curr_task[i]):
                curr_task[i] = curr_task[i-1] if i > 0 else float(task_config[0] + 1)
        curr_task = [int(field_no - 1) for field_no in curr_task]
        LEN_OG_SUBPATH = int(len(curr_task) / N_CONDS_PER_CONF)

        with contextlib.redirect_stdout(io.StringIO()):
            loc_og_matrix = mc.simulation.predictions.set_location_ephys(
                curr_task, task_config, grid_size=3, plotting=False)
            
            loc_model, phas_model, stat_model, midn_model, clo_model, phas_stat = mc.simulation.predictions.model_DSR(locations = curr_task, no_phase_neurons=N_PHASES)
            
            # clocks_matrix, midnight_matrix = mc.simulation.predictions.set_clocks_ephys(
            #     curr_task, task_config, grid_size=3, phases=N_PHASES, plotting=False)
        import pdb; pdb.set_trace()
        for n_sp in range(N_CONDS_PER_CONF):
            t0 = n_sp * LEN_OG_SUBPATH
            t1 = (n_sp + 1) * LEN_OG_SUBPATH
            for dst, src in [(clocks_mat,   clocks_matrix),
                             (midnight_mat, midnight_matrix),
                             (loc_og_mat,   loc_og_matrix)]:
                sub_sp = np.nanmean(src[:, t0:t1], axis=1)
                dst[row_start + n_sp, :] = np.where(np.isnan(sub_sp), 0.0, sub_sp)

    model_concat = {
        'location': loc_th,
        'dsr':      dsr_th,
        'state':    state_half,
        'feedback': feedback_half,
        'clocks':   clocks_mat,
        'loc_og':   loc_og_mat,
        'midnight': midnight_mat,
        'phase':    phase_half,
    }

    within, across = {}, {}
    for m in models:
        if m in ('location', 'dsr'):
            w, a = mc.analyse.my_RSA.compute_hamming_distance_within(
                model_concat[m], plotting=False,
                include_diagonal=INCLUDE_DIAG,
                model_name=m, no_tasks=len(configs),
                block_size=N_CONDS_PER_CONF)
        else:
            w, a = mc.analyse.my_RSA.compute_crosscorr_within(
                model_concat[m], plotting=False,
                include_diagonal=INCLUDE_DIAG,
                no_tasks=len(configs), model=m,
                block_size=N_CONDS_PER_CONF)
        within[m] = np.asarray(w[0], dtype=float)
        across[m] = np.asarray(a[0], dtype=float)

    # feedback: NaN → 1 (same convention as original script)
    within['feedback'] = np.where(np.isnan(within['feedback']), 1.0, within['feedback'])
    across['feedback'] = np.where(np.isnan(across['feedback']), 1.0, across['feedback'])

    # also return full (N×N) model RDM matrices for plotting / model-corr
    full = {}
    for m in models:
        data = model_concat[m]
        if m in ('location', 'dsr'):
            d = np.asarray(data, dtype=object)
            overlap = np.equal(d[:, None, :], d[None, :, :])
            full[m] = 1.0 - overlap.mean(axis=2)
        else:
            d = np.asarray(data, dtype=float)
            dd = d - d.mean(axis=1, keepdims=True)
            norms = np.sqrt(np.einsum('ij,ij->i', dd, dd))
            norms[norms == 0] = 1
            dd = dd / norms[:, None]
            full[m] = 1.0 - dd @ dd.T
    full['feedback'] = np.where(np.isnan(full['feedback']), 1.0, full['feedback'])
    return within, across, full


# ══════════════════════════════════════════════════════════════════════
# ── Data RDM helpers ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _one_minus_pearson_rdm(patterns):
    """patterns: (n_cond, n_features)."""
    X = patterns - patterns.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.einsum('ij,ij->i', X, X))
    norms[norms == 0] = 1
    X = X / norms[:, None]
    return 1.0 - X @ X.T


def _split_triu_blocks(mat, include_diag, block_size):
    n = mat.shape[0]
    k = 0 if include_diag else 1
    i, j = np.triu_indices(n, k=k)
    vec = mat[i, j]
    same = (i // block_size) == (j // block_size)
    return vec[same], vec[~same]


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
    if roi_filter == 'whole_brain':
        return np.arange(len(cell_labels))
    return np.array([i for i, lb in enumerate(cell_labels) if lb == roi_filter])


# ══════════════════════════════════════════════════════════════════════
# ── RSA runner ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _scalar(a):
    return float(np.asarray(a, dtype=float).ravel()[0])


def run_rsa(data_vec, model_RDMs_vec):
    """Unique regression per model + combined regression (mirrors DSR script)."""
    unique = {}
    for m in models:
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
    smooth, cell_labels, names, beh, locs, ok = load_session_trials(sub)
    if not ok:
        print(f'  s{sub}: no data.')
        continue

    keep_all = np.arange(smooth.shape[0])
    groups   = _build_run_groups(beh, smooth, keep_all)
    X_sess, n_neurons = _session_tensor(groups, K_RUNS)
    if X_sess is None:
        print(f'  s{sub}: <{K_RUNS} runs per config, skipping session.')
        continue

    # average across runs → (N_CONFIGS, N_CONDS_PER_CONF, n_neurons)
    X_avg = X_sess.mean(axis=1)

    sessions_data.append({
        'sub':         sub,
        'X_avg':       X_avg,
        'cell_labels': list(cell_labels),
        'neuron_names': list(names),
        'beh':         beh,
        'locs':        locs,
        'n_neurons':   int(n_neurons),
    })
    print(f'  s{sub}: {n_neurons} neurons, {K_RUNS}+ runs/config.')

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
    Returns X_pool (n_cond, n_neurons_total), contributing_subs (list[str]),
    n_neurons_per_sub dict.
    """
    parts = []
    contrib_subs = []
    per_sub_count = {}
    for sd in sessions_data:
        if 'X_avg' not in sd:
            continue
        keep = _roi_keep_idx(sd['cell_labels'], roi)
        if len(keep) == 0:
            continue
        X = sd['X_avg'][..., keep]                     # (cfg, tb, n_kept)
        patt = X.reshape(N_CONFIGS * N_CONDS_PER_CONF, -1)
        parts.append(patt)
        contrib_subs.append(sd['sub'])
        per_sub_count[sd['sub']] = int(X.shape[-1])
    if not parts:
        return None, [], {}
    X_pool = np.concatenate(parts, axis=1)
    return X_pool, contrib_subs, per_sub_count


def _mode_locs_for_subs(sub_list):
    collected = {c: [] for c in configs}
    for sub in sub_list:
        sub_acc = sub_loc_accum.get(sub, {})
        for c in configs:
            collected[c].extend(sub_acc.get(c, []))
    return build_mode_locs(collected)


print('\nRunning RSA per ROI...')
roi_results = {}

for roi in rois_of_interest:
    X_pool, contrib_subs, per_sub_count = _pool_roi(roi)
    if X_pool is None:
        print(f'  {roi}: no neurons, skipping.')
        continue

    # --- per-ROI model RDMs (from contributing sessions only) ------------
    mode_locs_roi = _mode_locs_for_subs(contrib_subs)
    model_within, model_across, model_full = build_model_rdms(mode_locs_roi)

    # --- data RDM (1 - Pearson), split into within/between vectors -------
    data_rdm = _one_minus_pearson_rdm(X_pool)
    data_within, data_across = _split_triu_blocks(
        data_rdm, INCLUDE_DIAG, block_size=N_CONDS_PER_CONF)

    # diagnostic: correlation between within and across vectors (only
    # meaningful if both have the same length → they don't.  Instead, we
    # report self-correlation by subsampling across to length of within.)
    corr_wb = float(np.corrcoef(
        data_within, np.random.default_rng(0).choice(
            data_across, size=len(data_within), replace=False))[0, 1]) \
        if len(data_within) > 2 and len(data_across) > 2 else float('nan')

    # --- RSA separately on within + between ------------------------------
    res_within  = run_rsa(data_within, model_within)
    res_across  = run_rsa(data_across, model_across)

    n_neurons_total = int(X_pool.shape[1])
    res_within['n_neurons']  = n_neurons_total
    res_across['n_neurons']  = n_neurons_total
    res_within['n_sessions'] = len(contrib_subs)
    res_across['n_sessions'] = len(contrib_subs)

    roi_results[roi] = {
        'within':            res_within,
        'across':            res_across,
        'n_neurons':         n_neurons_total,
        'n_sessions':        len(contrib_subs),
        'contrib_subs':      contrib_subs,
        'per_sub_count':     per_sub_count,
        'data_rdm':          data_rdm,
        'model_full':        model_full,
        'corr_within_across': corr_wb,
    }

    print(f'  {roi:18s} n_neurons={n_neurons_total:4d} '
          f'n_sess={len(contrib_subs):2d}  '
          f'(within: ' + ', '.join(
              f"{m}={res_within['unique'][m][1]:.2f}" for m in models) + ')')
    print(f'  {roi:18s} '
          f'(across: ' + ', '.join(
              f"{m}={res_across['unique'][m][1]:.2f}" for m in models) + ')')


# ══════════════════════════════════════════════════════════════════════
# ── Per-ROI bar plots (within vs between) ────────────────────────────
# ══════════════════════════════════════════════════════════════════════

if PLOT_FIGS:
    print('\nSaving per-ROI bar plots...')
    for roi, res in roi_results.items():
        w_betas = [res['within']['unique'][m][1]  for m in models]
        a_betas = [res['across']['unique'][m][1]  for m in models]
        w_p     = [res['within']['unique'][m][2]  for m in models]
        a_p     = [res['across']['unique'][m][2]  for m in models]

        x = np.arange(len(models)); width = 0.38
        fig, ax = plt.subplots(figsize=(1.05 * len(models) + 2.5, 4.0),
                               constrained_layout=True)
        bars_w = ax.bar(x - width/2, w_betas, width,
                        label='within config', color='#3b5bdb')
        bars_a = ax.bar(x + width/2, a_betas, width,
                        label='between configs', color='#fa5252')

        for bar, p in zip(bars_w, w_p):
            if p < 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        '*', ha='center', va='bottom', fontsize=11)
        for bar, p in zip(bars_a, a_p):
            if p < 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        '*', ha='center', va='bottom', fontsize=11)

        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha='right')
        ax.set_ylabel('beta (unique regression)')
        ax.set_title(
            f'{roi} — within vs between-config RSA  '
            f'(n_neurons={res["n_neurons"]}, n_sess={res["n_sessions"]})',
            fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        fig.savefig(os.path.join(OUT_DIR, f'rsa_within_across_{roi}.png'),
                    dpi=150, bbox_inches='tight')
        # plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# ── Block-diagonal vs between-block RDM overview figures ─────────────
# ══════════════════════════════════════════════════════════════════════

def _rdm_grid_figure(mode, save_path):
    """mode ∈ {'within','between'}.  Top row: data RDMs per ROI (masked).
    Bottom row: model RDMs per model (masked).  Model RDMs are taken from
    the whole_brain ROI (falls back to the first ROI if missing)."""
    present = list(roi_results.keys())
    if not present:
        return
    ref_roi = 'whole_brain' if 'whole_brain' in roi_results else present[0]
    model_full = roi_results[ref_roi]['model_full']

    n_data  = len(present)
    n_mdl   = len(models)
    n_cols  = max(n_data, n_mdl)

    fig, axes = plt.subplots(2, n_cols,
                             figsize=(2.6 * n_cols, 5.6),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    # top: data RDMs per ROI
    for j in range(n_cols):
        ax = axes[0, j]
        if j < n_data:
            roi = present[j]
            mat = _apply_block_mask(roi_results[roi]['data_rdm'],
                                    mode, N_CONDS_PER_CONF)
            # vmax = np.nanpercentile(np.abs(mat), 99) if np.isfinite(
            #     np.nanmean(mat)) else 1.0
            # im = ax.imshow(mat, aspect='auto', cmap='coolwarm',
            #                vmin=-vmax, vmax=vmax)
            im = ax.imshow(mat, aspect='auto', cmap='coolwarm',
                            vmin=0.9, vmax=1.1)
            ax.set_title(f'{roi}  (n={roi_results[roi]["n_neurons"]})',
                         fontsize=8)
            _add_block_lines(ax, mat.shape[0])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        else:
            ax.set_visible(False)
        ax.tick_params(labelsize=6)

    # bottom: model RDMs
    for j in range(n_cols):
        ax = axes[1, j]
        if j < n_mdl:
            m   = models[j]
            mat = _apply_block_mask(model_full[m], mode, N_CONDS_PER_CONF)
            im  = ax.imshow(mat, aspect='auto', cmap='coolwarm')
            ax.set_title(f'model: {m}', fontsize=8)
            _add_block_lines(ax, mat.shape[0])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        else:
            ax.set_visible(False)
        ax.tick_params(labelsize=6)

    title = ('Within-config block RDMs (block-diagonal, excl. main diag.)'
             if mode == 'within'
             else 'Between-config blocks RDMs (off-block-diagonal)')
    fig.suptitle(title + f'  [models from ROI={ref_roi}]',
                 fontsize=12, weight='bold')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')


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
                w_vec, _ = _split_triu_blocks(
                    model_full[m], INCLUDE_DIAG, N_CONDS_PER_CONF)
                vecs[m] = w_vec
            else:
                _, a_vec = _split_triu_blocks(
                    model_full[m], INCLUDE_DIAG, N_CONDS_PER_CONF)
                vecs[m] = a_vec
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


present_rois = list(roi_results.keys())
if present_rois:
    mc.plotting.cell_results.plot_rsa_heatmap(
        results      = _flatten('within'),
        models       = models,
        combo_models = combo_models,
        rois         = present_rois,
        title        = 'DSR RSA — within-config block — beta per model × ROI',
        save_path    = os.path.join(OUT_DIR, 'DSR_RSA_within_heatmap.png'),
    )
    mc.plotting.cell_results.plot_rsa_heatmap(
        results      = _flatten('across'),
        models       = models,
        combo_models = combo_models,
        rois         = present_rois,
        title        = 'DSR RSA — between-config blocks — beta per model × ROI',
        save_path    = os.path.join(OUT_DIR, 'DSR_RSA_across_heatmap.png'),
    )
    if PLOT_FIGS:
        plt.show()

print('\nAll done.')
