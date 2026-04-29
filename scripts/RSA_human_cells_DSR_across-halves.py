#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DSR RSA — cross-validated across-task-halves variant.

Loading: pre-saved per-(config × run) neural averages produced by
prep_human_cells_RSA-2026.py — same loading pattern as
RSA_human_cells_DSR.py (KEEP_RUNS_SEPRATE=False).

Pipeline mirrors RSA_human_cells_DSR_within-across.py (per-ROI
mode-locations, vectorised permutations, perm-distribution figure,
heatmap with perm-p, model×model correlation heatmap, data RDM grid),
but the data RDM is computed by crossing run-1 vs run-2 halves via
mc.analyse.my_RSA.compute_crosscorr / compute_hamming_distance instead
of the within/between-block split.
"""
import os
import sys
import json
import math
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import mc

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')


# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_DIR  = os.path.join(DATA_DIR, 'group', 'DSR_RSA_across-halves')

INCLUDE_DIAG = False
PLOT_FIGS    = True

N_PHASES              = 3
states                = ['A', 'B', 'C', 'D']
LEN_STANDARDISED_PATH = 10
RESOLUTIONx           = 2
N_CONDS_PER_CONF      = N_PHASES * len(states) * RESOLUTIONx
N_PER_HALF            = None  # filled in below once N_CONFIGS is known

N_PERMUTATIONS = 300   # set to None to skip permutations

rois_of_interest = ['whole_brain', 'OFC', 'EC', 'ACC', 'HC', 'PCC', 'AMY',
                    'OCCIP']
models       = ['dsr', 'now_and_next', 'state', 'feedback',
                'midnight', 'location', 'phase', 'phase_state']
combo_models = ['feedback', 'dsr', 'midnight', 'location']

configs = [
    '3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3',
    '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6',
]
N_CONFIGS  = len(configs)
N_PER_HALF = N_CONFIGS * N_CONDS_PER_CONF

with open(os.path.join(DATA_DIR, 'all_sessions_dsrRSA_grouping_summary.json')) as f:
    config_summary = json.load(f)
SUBJECTS = list(config_summary.keys())

os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# ── Generic helpers ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _downsample_mean_axis(x, target_len, axis=-1):
    """Vectorised downsample by averaging consecutive bins along `axis`."""
    x = np.moveaxis(x, axis, -1)
    n_in  = x.shape[-1]
    edges = np.linspace(0, n_in, target_len + 1, dtype=int)
    out   = np.empty(x.shape[:-1] + (target_len,), dtype=float)
    for i in range(target_len):
        chunk = x[..., edges[i]:edges[i+1]]
        if chunk.shape[-1] == 0:
            out[..., i] = np.nan
        else:
            out[..., i] = np.nanmean(chunk, axis=-1)
    return np.moveaxis(out, -1, axis)


def downsample_mode(x, target_len=10):
    x = np.asarray(x)
    block = len(x) // target_len
    return np.array([
        stats.mode(x[i*block:(i+1)*block], keepdims=False).mode
        for i in range(target_len)
    ])


def _zscore_columns(mat):
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (mat - mu) / sd


def _scalar(a):
    return float(np.asarray(a, dtype=float).ravel()[0])


def _two_sided_perm_p(empirical, perm_dist):
    perm_dist = np.asarray(perm_dist, dtype=float)
    perm_dist = perm_dist[np.isfinite(perm_dist)]
    if perm_dist.size == 0 or not np.isfinite(empirical):
        return np.nan
    return float((np.abs(perm_dist) >= np.abs(empirical)).mean())


def _vec_to_full(vec, n_size, include_diag=False):
    """Symmetric N×N matrix from upper-triangle vector."""
    full = np.full((n_size, n_size), np.nan)
    iu = np.triu_indices(n_size, k=0 if include_diag else 1)
    full[iu] = vec
    full[iu[1], iu[0]] = vec
    if include_diag:
        np.fill_diagonal(full, full.diagonal())
    else:
        np.fill_diagonal(full, 0.0)
    return full


# ══════════════════════════════════════════════════════════════════════
# ── Session loading (DSR.py KEEP_RUNS_SEPRATE=False style) ───────────
# ══════════════════════════════════════════════════════════════════════

def load_session(sub):
    """
    Returns dict per session:
        raw            (n_neurons, N_CONFIGS, 2_halves, n_bins_raw)
        cell_labels    list[str]
        neuron_names   list[str]
        sub_loc_accum  {run_id: {cfg_str: list of (n_trials, n_bins) arrays}}
    or None if any file is missing.
    """
    sub_dir   = os.path.join(DATA_DIR, f's{sub}', 'dsr_avg')
    npy_path  = os.path.join(sub_dir, f's{sub}_dsr_neural_avg_two_runs.npy')
    meta_path = os.path.join(sub_dir, f's{sub}_dsr_neuron_meta_two_runs.json')
    glog_path = os.path.join(sub_dir, f's{sub}_dsr_grouping_log_two_runs.json')
    if not all(os.path.exists(p) for p in (npy_path, meta_path, glog_path)):
        return None

    sesh_neurons = np.load(npy_path)        # (n_neurons, n_cfg_in_session, 2, n_bins)
    with open(meta_path) as f:
        nmeta = json.load(f)
    with open(glog_path) as f:
        glog = json.load(f)


    n_neurons   = sesh_neurons.shape[0]
    n_bins_raw  = sesh_neurons.shape[-1]

    cfg_lookup = {entry['config']: entry['config_idx']
                  for entry in config_summary[sub]}

    raw = np.full((n_neurons, N_CONFIGS, 2, n_bins_raw), np.nan)
    for c_idx, c in enumerate(configs):
        if c in cfg_lookup:
            raw[:, c_idx, :, :] = sesh_neurons[:, cfg_lookup[c], :, :]

    sub_loc_accum = {1: {c: [] for c in configs},
                     2: {c: [] for c in configs}}

    data_dict, _ = mc.analyse.helpers_human_cells.get_data(int(sub))
    key = f'sub-{int(sub):02}'
    if key in data_dict:
        beh  = data_dict[key]['beh'].copy()
        locs = data_dict[key]['locations']
        beh['config_str'] = beh.apply(
            lambda r: f"{int(r['loc_A'])}-{int(r['loc_B'])}-"
                      f"{int(r['loc_C'])}-{int(r['loc_D'])}", axis=1)
        beh['grid_no'] = beh['grid_no'].astype(int)
        for cfg_entry in glog['configs']:
            c = cfg_entry['config']
            for run_id, blocks in [(1, cfg_entry['run1_blocks']),
                                   (2, cfg_entry['run2_blocks'])]:
                mask = ((beh['config_str'] == c)
                        & (beh['correct'] == 1)
                        & (beh['grid_no'].isin(blocks)))
                idx = beh.index[mask]
                if len(idx) > 0:
                    sub_loc_accum[run_id][c].append(
                        locs.loc[idx].values.astype(float))

    cell_labels  = list(nmeta['cell_labels'][:n_neurons])
    neuron_names = list(nmeta['neuron_names'][:n_neurons])
    import pdb; pdb.set_trace()
    
    # to make sure, ignore cell_labels and rather rely on the dictionary keys.
    # roi = mc.analyse.helpers_human_cells.convert_cell_label_in_roi(cell_label)
    
    
    return {
        'sub':          sub,
        'raw':          raw,
        'cell_labels':  cell_labels,
        'neuron_names': neuron_names,
        'sub_loc_accum': sub_loc_accum,
    }


print('\nLoading sessions...')
sessions_data = []
for sub in SUBJECTS:
    sd = load_session(sub)
    if sd is None:
        print(f'  s{sub}: missing files, skipping.')
        continue
    sessions_data.append(sd)
    print(f"  s{sub}: {sd['raw'].shape[0]} neurons.")
print(f'Loaded {len(sessions_data)} sessions.')


# ══════════════════════════════════════════════════════════════════════
# ── Per-ROI mode-locations (per run) ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _mode_locs_for_subs(sub_set):
    """mode_locs[c][run_id] = (n_bins,) using only sessions in `sub_set`."""
    collected = {1: {c: [] for c in configs},
                 2: {c: [] for c in configs}}
    for sd in sessions_data:
        if sd['sub'] not in sub_set:
            continue
        for run_id in (1, 2):
            for c in configs:
                collected[run_id][c].extend(sd['sub_loc_accum'][run_id][c])

    mode_locs = {c: {1: None, 2: None} for c in configs}
    for run_id in (1, 2):
        for c in configs:
            parts = collected[run_id][c]
            if not parts:
                continue
            stacked = np.vstack(parts)
            m = stats.mode(stacked, axis=0, keepdims=False, nan_policy='omit')
            mode_locs[c][run_id] = m.mode.astype(float)
    return mode_locs


# ══════════════════════════════════════════════════════════════════════
# ── Build model RDMs (across-halves) ─────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _resolve_mode_vec(mode_vec, fallback_first_loc, n_bins):
    """Forward-fill NaNs and fall back to a constant ramp if entirely empty."""
    if mode_vec is None or np.all(np.isnan(mode_vec)):
        return np.full(n_bins, float(fallback_first_loc + 1))
    out = mode_vec.astype(float).copy()
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = out[i-1] if i > 0 else float(fallback_first_loc + 1)
    return out


def build_model_rdms(mode_locs):
    """
    Build per-model concatenated (2N, features) matrices and call
    compute_crosscorr / compute_hamming_distance.  Returns
        rdms[m]  : 1-D upper-triangle vector
        full[m]  : symmetric N×N reconstruction (for plotting)
        n_size   : N_PER_HALF
    """
    N_BINS = None
    for c in configs:
        for r in (1, 2):
            if mode_locs[c][r] is not None:
                N_BINS = len(mode_locs[c][r])
                break
        if N_BINS:
            break
    if N_BINS is None:
        raise RuntimeError('No mode-locations available for this subset.')
    LEN_OG_SUBPATH = N_BINS // N_CONDS_PER_CONF

    loc_th = {r: np.zeros((N_PER_HALF, LEN_STANDARDISED_PATH)) for r in (1, 2)}
    dsr_th = {r: np.zeros((N_PER_HALF, LEN_STANDARDISED_PATH * N_PHASES * len(states)))
              for r in (1, 2)}
    _n_clock_neurons = 9 * N_PHASES * (N_PHASES * len(states))
    feat_dim = {
        'dsr':          _n_clock_neurons,
        'midnight':     9 * N_PHASES,
        'location':     9,
        'phase_og':     N_PHASES,
        'state_og':     len(states),
        'now_and_next': _n_clock_neurons,
        'phase_state':  N_PHASES * len(states),
    }
    prep = {m: {r: np.zeros((N_PER_HALF, fdim)) for r in (1, 2)}
            for m, fdim in feat_dim.items()}

    for run_id in (1, 2):
        for c_idx, c in enumerate(configs):
            task_config = [int(int(x) - 1) for x in c.split('-')]
            mode_vec    = _resolve_mode_vec(
                mode_locs[c][run_id], task_config[0], N_BINS)

            dsr_first = downsample_mode(
                mode_vec,
                target_len=LEN_STANDARDISED_PATH * N_PHASES * len(states))
            row0 = c_idx * N_CONDS_PER_CONF
            for n_sp in range(N_CONDS_PER_CONF):
                sp = mode_vec[n_sp*LEN_OG_SUBPATH:(n_sp+1)*LEN_OG_SUBPATH]
                loc_th[run_id][row0 + n_sp, :] = downsample_mode(
                    sp, target_len=LEN_STANDARDISED_PATH)
                dsr_th[run_id][row0 + n_sp, :] = np.roll(
                    dsr_first, -n_sp * LEN_STANDARDISED_PATH)

            walked = [int(w - 1) for w in mode_vec]
            (loc_og_matrix, phase_og_matrix, stat_matrix, midnight_matrix,
             dsr_matrix, phas_stat_matrix, clo_model_subpath) = (
                mc.simulation.predictions.model_DSR(
                    locations=walked, no_phase_neurons=N_PHASES))

            for n_sp in range(N_CONDS_PER_CONF):
                t0 = n_sp * LEN_OG_SUBPATH
                t1 = (n_sp + 1) * LEN_OG_SUBPATH
                for mn, src in [
                    ('dsr',          dsr_matrix),
                    ('midnight',     midnight_matrix),
                    ('location',     loc_og_matrix),
                    ('phase_og',     phase_og_matrix),
                    ('state_og',     stat_matrix),
                    ('now_and_next', clo_model_subpath),
                    ('phase_state',  phas_stat_matrix),
                ]:
                    sub_sp = np.nanmean(src[:, t0:t1], axis=1)
                    prep[mn][run_id][row0 + n_sp, :] = np.where(
                        np.isnan(sub_sp), 0.0, sub_sp)

    # state / feedback / phase: identical in both halves
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

    feedback_half = np.tile(feedback_config, (N_CONFIGS, 1))
    state_half    = np.tile(state_config,    (N_CONFIGS, 1))
    phase_half    = np.tile(phase_config,    (N_CONFIGS, 1))

    model_concat = {
        'loc_hamming':  np.concatenate([loc_th[1], loc_th[2]], axis=0),
        'dsr_hamming':  np.concatenate([dsr_th[1], dsr_th[2]], axis=0),
        'state':        np.tile(state_half,    (2, 1)),
        'feedback':     np.tile(feedback_half, (2, 1)),
        'phase':        np.tile(phase_half,    (2, 1)),
        'dsr':          np.concatenate([prep['dsr'][1],
                                        prep['dsr'][2]],          axis=0),
        'midnight':     np.concatenate([prep['midnight'][1],
                                        prep['midnight'][2]],     axis=0),
        'location':     np.concatenate([prep['location'][1],
                                        prep['location'][2]],     axis=0),
        'phase_og':     np.concatenate([prep['phase_og'][1],
                                        prep['phase_og'][2]],     axis=0),
        'state_og':     np.concatenate([prep['state_og'][1],
                                        prep['state_og'][2]],     axis=0),
        'now_and_next': np.concatenate([prep['now_and_next'][1],
                                        prep['now_and_next'][2]], axis=0),
        'phase_state':  np.concatenate([prep['phase_state'][1],
                                        prep['phase_state'][2]],  axis=0),
    }

    rdms, full = {}, {}
    for m in models:
        if m in ('loc_hamming', 'dsr_hamming'):
            res = mc.analyse.my_RSA.compute_hamming_distance(
                model_concat[m], plotting=False,
                include_diagonal=INCLUDE_DIAG,
                model_name=m, no_tasks=N_CONFIGS)
        else:
            res = mc.analyse.my_RSA.compute_crosscorr(
                model_concat[m], plotting=False,
                include_diagonal=INCLUDE_DIAG,
                no_tasks=N_CONFIGS, model=m)
        rdms[m] = np.asarray(res[0], dtype=float)
        full[m] = _vec_to_full(rdms[m], N_PER_HALF, INCLUDE_DIAG)

    rdms['feedback'] = np.where(np.isnan(rdms['feedback']), 1.0, rdms['feedback'])
    full['feedback'] = np.where(np.isnan(full['feedback']), 1.0, full['feedback'])
    return rdms, full


# ══════════════════════════════════════════════════════════════════════
# ── Pool ROI + build data matrix ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _roi_keep(cell_labels, roi):
    if roi == 'whole_brain':
        return np.arange(len(cell_labels))
    return np.array([i for i, lb in enumerate(cell_labels) if lb == roi])


def _pool_roi(roi):
    """Returns (raw_pool, contrib_subs, per_sub_count) or (None, [], {})."""
    parts, contrib = [], []
    per_sub_count = {}
    for sd in sessions_data:
        keep = _roi_keep(sd['cell_labels'], roi)
        if len(keep) == 0:
            continue
        parts.append(sd['raw'][keep])
        contrib.append(sd['sub'])
        per_sub_count[sd['sub']] = int(len(keep))
    if not parts:
        return None, [], {}
    raw_pool = np.concatenate(parts, axis=0)   # (n_neurons_total, N_CONFIGS, 2, n_bins)
    return raw_pool, contrib, per_sub_count


def _data_matrix(raw_pool, valid_mask=None):
    """
    raw_pool : (n_neurons, N_CONFIGS, 2, n_bins_raw)
    Returns
        mat        (2 * N_PER_HALF, n_valid_neurons), per-neuron z-scored
        valid_mask boolean (n_neurons,) — neurons surviving NaN filter
    Downsample is applied to the last axis; halves are concatenated vertically.
    """
    th = _downsample_mean_axis(raw_pool, target_len=N_CONDS_PER_CONF, axis=-1)
    # th: (n_neurons, N_CONFIGS, 2, N_CONDS_PER_CONF)
    th = np.transpose(th, (2, 1, 3, 0))   # (2, N_CONFIGS, N_CONDS_PER_CONF, n_neurons)
    th1 = th[0].reshape(N_PER_HALF, -1)
    th2 = th[1].reshape(N_PER_HALF, -1)
    if valid_mask is None:
        valid_mask = ~(np.isnan(th1).any(axis=0) | np.isnan(th2).any(axis=0))
    th1 = th1[:, valid_mask]
    th2 = th2[:, valid_mask]
    mat = np.concatenate([th1, th2], axis=0)
    mat = _zscore_columns(mat)
    return mat, valid_mask


def _data_rdm(mat):
    """1-D upper-triangle vector + symmetric N×N reconstruction."""
    res = mc.analyse.my_RSA.compute_crosscorr(
        mat, plotting=False, include_diagonal=INCLUDE_DIAG,
        no_tasks=N_CONFIGS, model='data')
    vec  = np.asarray(res[0], dtype=float)
    full = _vec_to_full(vec, N_PER_HALF, INCLUDE_DIAG)
    return vec, full


# ══════════════════════════════════════════════════════════════════════
# ── RSA + permutation runner ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def run_rsa(data_vec, model_RDMs):
    unique = {}
    for m in models:
        res = mc.analyse.my_RSA.evaluate_model(model_RDMs[m], data_vec)
        unique[m] = (_scalar(res[0]), _scalar(res[1]), _scalar(res[2]))
    stacked = np.stack([model_RDMs[m] for m in combo_models], axis=1)
    res_c = mc.analyse.my_RSA.evaluate_model(stacked, data_vec)
    combined = {
        't':    np.asarray(res_c[0], dtype=float).ravel(),
        'beta': np.asarray(res_c[1], dtype=float).ravel(),
        'p':    np.asarray(res_c[2], dtype=float).ravel(),
    }
    return {'unique': unique, 'combined': combined}


def _perm_pass(raw_pool, valid_mask, model_RDMs):
    """Vectorised per-perm circular shifts (one shift per neuron, broadcast
    across configs and halves), recompute data RDM, run RSA.  Returns
        unique_betas[m] : (n_perms,)
        combo_betas     : (n_perms, len(combo_models))
    """
    n_neurons = raw_pool.shape[0]
    n_bins    = raw_pool.shape[-1]
    rng       = np.random.default_rng()

    unique_betas = {m: np.full(N_PERMUTATIONS, np.nan) for m in models}
    combo_betas  = np.full((N_PERMUTATIONS, len(combo_models)), np.nan)
    stacked      = np.stack([model_RDMs[m] for m in combo_models], axis=1)

    b_idx = np.arange(n_bins)
    for p in range(N_PERMUTATIONS):
        ks       = rng.integers(0, n_bins, size=n_neurons)
        shift_b  = (b_idx[None, :] + ks[:, None]) % n_bins   # (n_neurons, n_bins)
        raw_p    = np.take_along_axis(
            raw_pool, shift_b[:, None, None, :], axis=-1)
        mat_p, _ = _data_matrix(raw_p, valid_mask=valid_mask)
        vec_p, _ = _data_rdm(mat_p)
        for m in models:
            _, b, _ = mc.analyse.my_RSA.evaluate_model(model_RDMs[m], vec_p)
            unique_betas[m][p] = _scalar(b)
        _, bc, _ = mc.analyse.my_RSA.evaluate_model(stacked, vec_p)
        combo_betas[p] = np.asarray(bc, dtype=float).ravel()
    return unique_betas, combo_betas


# ══════════════════════════════════════════════════════════════════════
# ── Main loop ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

print('\nRunning RSA per ROI...')
roi_results = {}

for roi in rois_of_interest:
    raw_pool, contrib, per_sub_count = _pool_roi(roi)
    if raw_pool is None:
        print(f'  {roi}: no neurons, skipping.')
        continue

    mat, valid_mask = _data_matrix(raw_pool)
    if mat.shape[1] == 0:
        print(f'  {roi}: 0 valid neurons, skipping.')
        continue

    mode_locs = _mode_locs_for_subs(set(contrib))
    model_RDMs, model_full = build_model_rdms(mode_locs)

    data_vec, data_full = _data_rdm(mat)
    res = run_rsa(data_vec, model_RDMs)
    n_neurons_total = int(mat.shape[1])
    res['n_neurons']  = n_neurons_total
    res['n_sessions'] = len(contrib)

    perm_betas = perm_combo = None
    perm_p_unique = perm_p_combined = None
    if N_PERMUTATIONS:
        # restrict the perm pool to the valid neurons (matches empirical mat)
        raw_valid = raw_pool[valid_mask]
        perm_betas, perm_combo = _perm_pass(
            raw_valid, np.ones(raw_valid.shape[0], dtype=bool), model_RDMs)
        perm_p_unique = {
            m: _two_sided_perm_p(res['unique'][m][1], perm_betas[m])
            for m in models}
        perm_p_combined = {
            cm: _two_sided_perm_p(
                res['combined']['beta'][i], perm_combo[:, i])
            for i, cm in enumerate(combo_models)}

    roi_results[roi] = {
        **res,
        'data_rdm':         data_full,
        'model_full':       model_full,
        'model_rdms':       model_RDMs,
        'contrib':          contrib,
        'per_sub_count':    per_sub_count,
        'perm_betas':       perm_betas,
        'perm_combo':       perm_combo,
        'perm_p_unique':    perm_p_unique,
        'perm_p_combined':  perm_p_combined,
    }

    print(f"  {roi:15s} n={n_neurons_total:4d} n_sess={len(contrib):2d}  "
          + ', '.join(f"{m}={res['unique'][m][1]:.2f}" for m in models))


# ══════════════════════════════════════════════════════════════════════
# ── Heatmap with perm-p ──────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _flatten():
    flat = {}
    for roi, res in roi_results.items():
        flat[roi] = {
            'unique':     res['unique'],
            'combined':   res['combined'],
            'n_neurons':  res['n_neurons'],
            'n_sessions': res['n_sessions'],
        }
    return flat


def _collect_perm_p():
    pu = {r: roi_results[r]['perm_p_unique']
          for r in roi_results if roi_results[r].get('perm_p_unique') is not None}
    pc = {r: roi_results[r]['perm_p_combined']
          for r in roi_results if roi_results[r].get('perm_p_combined') is not None}
    return (pu or None, pc or None)


present_rois = list(roi_results.keys())
if present_rois:
    pu, pc = _collect_perm_p()
    mc.plotting.cell_results.plot_rsa_heatmap(
        results        = _flatten(),
        models         = models,
        combo_models   = combo_models,
        rois           = present_rois,
        title          = 'DSR RSA — across-task-halves — beta per model × ROI',
        save_path      = os.path.join(OUT_DIR, 'DSR_RSA_across_halves_heatmap.png'),
        perm_p_unique   = pu,
        perm_p_combined = pc,
    )
    print('Saved RSA heatmap.')


# ══════════════════════════════════════════════════════════════════════
# ── Permutation distribution figure ──────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _plot_perm_distribution(save_path):
    rois_with = [r for r, res in roi_results.items()
                 if res.get('perm_betas') is not None]
    if not rois_with:
        return
    n_rows = len(rois_with)
    n_cols = len(models)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.0 * n_cols + 1.0, 1.6 * n_rows + 1.0),
        constrained_layout=True, squeeze=False)

    for i, roi in enumerate(rois_with):
        res        = roi_results[roi]
        perm_betas = res['perm_betas']
        perm_p     = res['perm_p_unique'] or {}
        for j, m in enumerate(models):
            ax  = axes[i, j]
            arr = np.asarray(perm_betas[m], dtype=float)
            arr = arr[np.isfinite(arr)]
            emp = float(res['unique'][m][1])
            if arr.size:
                ax.hist(arr, bins=min(30, max(3, arr.size // 5 + 1)),
                        color='#bdbdbd', edgecolor='none')
            ax.axvline(emp, color='#d62728', lw=1.4)
            ax.axvline(0,   color='grey',    lw=0.6)
            pv = perm_p.get(m, np.nan)
            ax.set_title(f'{m}  p={pv:.2g}' if np.isfinite(pv) else m,
                         fontsize=7)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(f'{roi}\n(n={res["n_neurons"]})', fontsize=7)
            if i < n_rows - 1:
                ax.set_xticklabels([])

    fig.suptitle('Across-task-halves — perm β distributions vs empirical',
                 fontsize=12, weight='bold')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')


if roi_results:
    _plot_perm_distribution(os.path.join(OUT_DIR, 'perm_distribution.png'))
    print('Saved permutation distribution figure.')


# ══════════════════════════════════════════════════════════════════════
# ── Model × Model correlation heatmap (whole-brain ref) ──────────────
# ══════════════════════════════════════════════════════════════════════

def _plot_model_corr_heatmap(save_path):
    if not roi_results:
        return
    ref_roi = ('whole_brain' if 'whole_brain' in roi_results
               else next(iter(roi_results)))
    rdms = roi_results[ref_roi]['model_rdms']
    n_m = len(models)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    corr = np.zeros((n_m, n_m))
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            va, vb = rdms[a], rdms[b]
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
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f'Model RDM × Model RDM Pearson r   [from ROI={ref_roi}]',
                 fontsize=12, weight='bold')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')


if roi_results:
    _plot_model_corr_heatmap(
        os.path.join(OUT_DIR, 'model_rdm_correlations.png'))
    print('Saved model×model correlation heatmap.')


# ══════════════════════════════════════════════════════════════════════
# ── Data RDM grid + Model RDM grid ───────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _plot_rdm_grid(mats, titles, save_path, suptitle, vmin=None, vmax=None):
    n = len(mats)
    if n == 0:
        return
    n_cols = min(4, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 3 * n_rows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, (mat, t) in enumerate(zip(mats, titles)):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]
        im = ax.imshow(mat, aspect='auto', cmap='coolwarm',
                       vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=8)
        for p in range(0, mat.shape[0], N_CONDS_PER_CONF):
            ax.axvline(p - 0.5, color='white', ls='dashed', lw=0.4)
            ax.axhline(p - 0.5, color='white', ls='dashed', lw=0.4)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.tick_params(labelsize=6)
    for j in range(n, n_rows * n_cols):
        r, c = j // n_cols, j % n_cols
        axes[r, c].set_visible(False)
    fig.suptitle(suptitle, fontsize=12, weight='bold')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')


if roi_results:
    data_mats   = [roi_results[r]['data_rdm']
                   for r in roi_results if roi_results[r].get('data_rdm') is not None]
    data_titles = [f'{r} (n={roi_results[r]["n_neurons"]})'
                   for r in roi_results if roi_results[r].get('data_rdm') is not None]
    _plot_rdm_grid(data_mats, data_titles,
                   os.path.join(OUT_DIR, 'data_rdms.png'),
                   'Across-task-halves — data RDMs',
                   vmin=0.7, vmax=1.3)

    ref_roi = ('whole_brain' if 'whole_brain' in roi_results
               else next(iter(roi_results)))
    model_full = roi_results[ref_roi]['model_full']
    _plot_rdm_grid([model_full[m] for m in models],
                   [f'model: {m}' for m in models],
                   os.path.join(OUT_DIR, 'model_rdms.png'),
                   f'Across-task-halves — model RDMs (ROI={ref_roi})')
    print('Saved data + model RDM grids.')


if PLOT_FIGS:
    plt.show()

print('\nAll done.')
