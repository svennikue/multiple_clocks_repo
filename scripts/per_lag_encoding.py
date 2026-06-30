#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-lag location encoding for human single units — rate-map version.

For each cell, at each lag k in {0,30,...,330}°, we build a 9-d
"lag-shifted rate map" per configuration: mean firing rate at bins
where the location AT LAG k is L, for L in 1..9. Leave-one-config-out
CV gives a weighted Pearson r between the dwell-weighted mean of the
training rate maps and the held-out rate map. Permutation null is a
circular shift of the HELD-OUT LOCATION SERIES (not of y), which
preserves dwell structure under the null.

WHY THE RATE-MAP REPLACED THE BIN-LEVEL GLM
    Pearson r between a step-function predictor (9 distinct values
    painted on 360 bins) and a noisy continuous firing trace is bounded
    above by how much of bin-by-bin variance the location code captures
    — and most of that variance is bin-level noise / residual phase
    jitter / trial-timing jitter / state. Pre-averaging y by location
    BEFORE inference removes that noise floor. This matches the design
    of `spatial_peaks_simple` (which is FDR-significant for ACC at
    30°/60°) and lets the same statistic be computed at every lag and
    with optional controls.

CONTROL MODES
    no_ctrl    Rate maps computed from raw (phase-residualised) y.
    with_ctrl  Rate maps computed from y RESIDUALISED against
               state (4) + bttn_curr (5) + bttn_next (5) + per-config
               intercepts, with location-at-lag-0° added when k != 0°.
               Controls are fit ON TRAINING CONFIGS only; the training
               betas are applied to all configs, so the held-out
               residual y is leakage-free.
               L2-norm is intentionally NOT a control — fully collinear
               with the categorical location dummies in encoding space.

AGGREGATE MODELS
    dsrfull   Stacked rate-maps across all "beyond-now" lags
              (30°…330°, 11 × 9 = 99-d vector per config).
    dsrinf    Stacked rate-maps across lags 30°/60°/90° (3 × 9 = 27-d).
    Both report CV r against zero — no winner's-curse comparison
    against best-single-lag (the previous T6 was misleading).

QUALITY CONTROLS (borrowed from spatial_peaks)
    MIN_DWELL_BINS         per-location dwell threshold (default 25 bins)
    WEIGHTED_CORRELATION   weight Pearson r by min(train_dwell, held_dwell)
                           per location, so under-visited locations are
                           downweighted automatically (replaces explicit
                           grid-group pairing).

PHASE RESIDUALISATION applied per cell BEFORE any design construction
(`mc.analyse.future_spatial_peaks._residualise_phase`, cosine basis).

ROI-LEVEL STATS (per_roi_stats.csv):
    T1   one-sample t-test of mean CV r > 0          (per lag, per variant)
    T2   within-cell paired: r at predicted lag(s) vs mean r at other lags
    T3   binomial: fraction of cells with perm-p < alpha exceeds alpha
    T4   Wilcoxon signed-rank of per-cell perm-p vs 0.5 (population shift)
    T5   dsrfull and dsrinf r > 0
    T6   paired: dsrinf r vs r at first predicted lag (replaces winner-curse)

Predicted lags per ROI (used by T2 and Fig 01 outlines):
    ACC          (30°, 60°)
    HC_anterior  (0°,)
    HC_mid       (0°, 330°)

OUTPUT under DATA_DIR/group/per_lag_encoding/<run_tag>/:
    per_cell_<ROI>.csv
    per_cell_ALL_ROIs.csv
    per_roi_stats.csv
    roi_lag_table.csv
    config.json
    figures/*.pdf + .png   (01 t-stat heatmap, 02 curves, 03 sig-fraction bar,
                            03b sig-fraction heatmap, 04 dsrfull vs dsrinf,
                            05 ctrl vs noctrl, 06 r distribution at predicted lag)

Reusable plotting helpers live in `mc.plotting.cell_results`.

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import era_brewer
import mc.analyse.cell_selection as cs
import mc.analyse.helpers_human_cells as hh
from mc.analyse.future_spatial_peaks import _residualise_phase
from mc.plotting.cell_results import (
    plot_roi_lag_tstat_heatmap,
    plot_roi_lag_curves,
    p_to_stars,
)


# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_BASE = os.path.join(DATA_DIR, 'group', 'per_lag_encoding')

ROIS_TO_RUN = ['ACC', 'medialOFC', 'PCC', 'Parahippocampal',
               'HC_anterior', 'HC_mid', 'EC']

PHASE_RESIDUALISE      = 'cosine'
TRIALS                 = 'all_minus_explore'
N_PERMUTATIONS         = 1000
N_JOBS                 = -1
RANDOM_SEED            = 42
ALPHA                  = 0.05

# Rate-map QC (mirrors spatial_peaks_simple settings)
MIN_DWELL_BINS         = 25
WEIGHTED_CORRELATION   = True
MIN_SHARED_LOCS        = 3        # rate-map r needs >= this many shared loc bins

# Lag grid (degrees); 1 deg = 1 bin in the standardised 360-bin trial.
N_BINS  = 360
N_LOC   = 9
N_STATE = 4
LAGS_DEG = list(range(0, 360, 30))     # 0,30,...,330

DSR_FULL_LAGS_DEG = [l for l in LAGS_DEG if l != 0]    # 11 "beyond-now" lags
DSR_INF_LAGS_DEG  = [30, 60, 90]                        # ACC-informed subset

ROI_PREDICTED_LAGS_DEG = {
    'ACC':         (30, 60),
    'HC_anterior': (0,),
    'HC_mid':      (0, 330),
}

# bttn_next = button this many bins ahead. User-tuned (was 90 = one full
# state, now 30 = one position bin).
LAG_BINS_BTTN_NEXT = 30

BTTN_CODES = [0, 1, 2, 3, 99]
N_BTTN     = len(BTTN_CODES)
BTTN_LABEL_TO_CODE = {'LeftArrow': 0, 'UpArrow': 1, 'RightArrow': 2,
                       'DownArrow': 3, 'Return': 99}

# Plotting --------------------------------------------------------------
DPI = 300
CM = 1.0 / 2.54
FONT_BIG, FONT_AXIS, FONT_TICK = 11, 10, 9
_ROI_PAL = era_brewer.era_brew('Showgirl2', n=7)
ROI_COLOURS = {                 # CLAUDE.md project-wide convention
    'EC':              _ROI_PAL[0],
    'medialOFC':       _ROI_PAL[4],
    'ACC':             _ROI_PAL[1],
    'HC_anterior':     _ROI_PAL[2],
    'HC_mid':          _ROI_PAL[6],
    'Parahippocampal': _ROI_PAL[5],
    'PCC':             _ROI_PAL[3],
}
OBSERVED_GREEN = '#0e3d3a'


# ── Generic helpers ───────────────────────────────────────────────────
def _set_rc():
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':       FONT_TICK,
        'pdf.fonttype':    42,
        'ps.fonttype':     42,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _save(fig, save_stem):
    fig.savefig(save_stem + '.pdf', dpi=DPI, bbox_inches='tight')
    fig.savefig(save_stem + '.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def _bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    good = np.isfinite(p)
    if not good.any():
        return out
    pg = p[good]
    order = np.argsort(pg)
    ranked = pg[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out_good = np.empty_like(q); out_good[order] = q
    out[good] = out_good
    return out


def _weighted_pearson_r(x, y, w):
    """Weighted Pearson correlation. NaN-safe; returns NaN if degenerate."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if m.sum() < MIN_SHARED_LOCS:
        return np.nan
    x, y, w = x[m], y[m], w[m]
    wsum = w.sum()
    xm = (x * w).sum() / wsum
    ym = (y * w).sum() / wsum
    cov = (w * (x - xm) * (y - ym)).sum()
    vx  = (w * (x - xm) ** 2).sum()
    vy  = (w * (y - ym) ** 2).sum()
    if vx < 1e-12 or vy < 1e-12:
        return np.nan
    return float(cov / np.sqrt(vx * vy))


def _pearson_r(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < MIN_SHARED_LOCS:
        return np.nan
    if np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


# ── Per-config mode series ────────────────────────────────────────────
def _mode_per_bin_int(arr2d):
    out = np.zeros(arr2d.shape[1], dtype=int)
    for t in range(arr2d.shape[1]):
        col = arr2d[:, t]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        col = col.astype(int)
        vals, cnts = np.unique(col, return_counts=True)
        out[t] = int(vals[np.argmax(cnts)])
    return out


def _build_per_cfg_sequences(arr_clean, idx_cfg, locs, btns_int, n_cfg):
    Y_cfg   = np.zeros((n_cfg, N_BINS))
    loc_cfg = np.zeros((n_cfg, N_BINS), dtype=int)
    btn_cfg = np.zeros((n_cfg, N_BINS), dtype=int)
    for c in range(n_cfg):
        mask = idx_cfg == c
        if not mask.any():
            continue
        Y_cfg[c]   = np.nanmean(arr_clean[mask], axis=0)
        loc_cfg[c] = _mode_per_bin_int(locs[mask])
        btn_cfg[c] = _mode_per_bin_int(btns_int[mask])
    return Y_cfg, loc_cfg, btn_cfg


# ── Rate-map helpers ──────────────────────────────────────────────────
def _lag_shifted_rate_map(y_360, loc_360, lag_bins):
    """Returns (9-d rate, 9-d dwell). rate[L-1] = mean firing at bins
    where the location AT LAG +lag_bins is L."""
    loc_at_lag = np.roll(loc_360, -lag_bins)
    rate = np.full(N_LOC, np.nan)
    dwell = np.zeros(N_LOC, dtype=int)
    for L in range(1, N_LOC + 1):
        m = (loc_at_lag == L) & np.isfinite(y_360)
        d = int(m.sum())
        dwell[L - 1] = d
        if d > 0:
            rate[L - 1] = float(np.nanmean(y_360[m]))
    return rate, dwell


def _stacked_rate_map(y_360, loc_360, lags_deg):
    """Concatenate lag-shifted rate maps across `lags_deg`.
    Returns (9*len(lags),) rate and dwell."""
    rates, dwells = [], []
    for lag in lags_deg:
        r, d = _lag_shifted_rate_map(y_360, loc_360, lag)
        rates.append(r); dwells.append(d)
    return np.concatenate(rates), np.concatenate(dwells)


def _ratemap_cv_one_setup(rate_per_cfg, dwell_per_cfg,
                          loc_cfg, y_cfg_used, lags_deg,
                          n_perms, rng):
    """LOO CV over configs given pre-built per-config rate maps + dwell.
    `loc_cfg` and `y_cfg_used` are passed so we can recompute the held-out
    rate map after a circular shift of the held-out location series for
    the permutation null. Returns (mean_r, perm_p)."""
    n_cfg = rate_per_cfg.shape[0]
    if n_cfg < 2:
        return np.nan, np.nan
    fold_rs = []
    perm_r_per_fold = (np.full((n_perms, n_cfg), np.nan)
                       if n_perms > 0 else None)
    for held in range(n_cfg):
        idx_tr = [c for c in range(n_cfg) if c != held]
        train_r = rate_per_cfg[idx_tr]            # (n_tr, D)
        train_d = dwell_per_cfg[idx_tr]
        total_d = train_d.sum(axis=0)             # (D,)
        with np.errstate(invalid='ignore'):
            predicted = np.nansum(train_r * train_d, axis=0) / np.where(
                total_d > 0, total_d, np.nan)
        observed   = rate_per_cfg[held]
        held_d     = dwell_per_cfg[held]
        weights = (np.minimum(total_d, held_d)
                    if WEIGHTED_CORRELATION
                    else (np.minimum(total_d, held_d) >= 1).astype(float))
        keep = ((np.minimum(total_d, held_d) >= MIN_DWELL_BINS)
                & np.isfinite(predicted) & np.isfinite(observed))
        if keep.sum() < MIN_SHARED_LOCS:
            continue
        if WEIGHTED_CORRELATION:
            r = _weighted_pearson_r(predicted[keep], observed[keep], weights[keep])
        else:
            r = _pearson_r(predicted[keep], observed[keep])
        if np.isfinite(r):
            fold_rs.append(r)
        # Permutation: shift the held-out location series, rebuild its
        # rate map, weight by new dwell × training-total dwell, re-correlate.
        if n_perms > 0:
            for p in range(n_perms):
                # Avoid trivial near-zero shifts (would re-create the obs)
                shift = int(rng.integers(N_BINS // 12, N_BINS - N_BINS // 12))
                loc_sh = np.roll(loc_cfg[held], shift)
                obs_p, dwell_p = _stacked_rate_map(y_cfg_used[held], loc_sh,
                                                    lags_deg)
                w_p = (np.minimum(total_d, dwell_p)
                        if WEIGHTED_CORRELATION
                        else (np.minimum(total_d, dwell_p) >= 1).astype(float))
                kp = ((np.minimum(total_d, dwell_p) >= MIN_DWELL_BINS)
                       & np.isfinite(predicted) & np.isfinite(obs_p))
                if kp.sum() < MIN_SHARED_LOCS:
                    continue
                if WEIGHTED_CORRELATION:
                    rp = _weighted_pearson_r(predicted[kp], obs_p[kp], w_p[kp])
                else:
                    rp = _pearson_r(predicted[kp], obs_p[kp])
                perm_r_per_fold[p, held] = rp

    if not fold_rs:
        return np.nan, np.nan
    mean_r = float(np.mean(fold_rs))
    if n_perms == 0:
        return mean_r, np.nan
    null_means = np.nanmean(perm_r_per_fold, axis=1)
    null_means = null_means[np.isfinite(null_means)]
    if null_means.size == 0:
        return mean_r, np.nan
    pval = (np.sum(null_means >= mean_r) + 1) / (null_means.size + 1)
    return mean_r, float(pval)


# ── Control partial-out ───────────────────────────────────────────────
def _loc_onehot_at_lag_bin(loc_cfg, lag_bins):
    """For partial-out only: bin-level (n_cfg*360, 9) one-hot at lag."""
    n_cfg = loc_cfg.shape[0]
    out = np.zeros((n_cfg * N_BINS, N_LOC), dtype=float)
    for c in range(n_cfg):
        rolled = np.roll(loc_cfg[c], -lag_bins)
        for t in range(N_BINS):
            l = rolled[t]
            if 1 <= l <= N_LOC:
                out[c * N_BINS + t, l - 1] = 1.0
    return out


def _btn_onehot_at_lag_bin(btn_cfg, lag_bins):
    n_cfg = btn_cfg.shape[0]
    code_to_col = {c: i for i, c in enumerate(BTTN_CODES)}
    out = np.zeros((n_cfg * N_BINS, N_BTTN), dtype=float)
    for c in range(n_cfg):
        rolled = np.roll(btn_cfg[c], -lag_bins)
        for t in range(N_BINS):
            code = int(rolled[t])
            j = code_to_col.get(code, None)
            if j is not None:
                out[c * N_BINS + t, j] = 1.0
    return out


def _state_onehot_bin(n_cfg):
    state_per_bin = (np.arange(N_BINS) // (N_BINS // N_STATE)).astype(int)
    one_per_cfg = np.zeros((N_BINS, N_STATE))
    for s in range(N_STATE):
        one_per_cfg[state_per_bin == s, s] = 1.0
    return np.tile(one_per_cfg, (n_cfg, 1))


def _cfg_intercepts_bin(n_cfg, train_cfg_ids):
    cols = sorted(train_cfg_ids)[1:]
    X = np.zeros((n_cfg * N_BINS, len(cols)))
    for j, c in enumerate(cols):
        X[c * N_BINS:(c + 1) * N_BINS, j] = 1.0
    return X


def _partial_out_controls(Y_cfg, loc_cfg, btn_cfg, train_cfg_ids,
                            include_loc_now=True):
    """Fit y ~ state + bttn_curr + bttn_next [+ location-now] on training
    configs, return residual Y_cfg (residuals for ALL configs). Held-out
    is leakage-free because betas are trained without it."""
    n_cfg = Y_cfg.shape[0]
    blocks = [_state_onehot_bin(n_cfg)]
    if include_loc_now:
        blocks.append(_loc_onehot_at_lag_bin(loc_cfg, 0))
    blocks.append(_btn_onehot_at_lag_bin(btn_cfg, 0))
    blocks.append(_btn_onehot_at_lag_bin(btn_cfg, LAG_BINS_BTTN_NEXT))
    blocks.append(_cfg_intercepts_bin(n_cfg, train_cfg_ids))
    X = np.hstack(blocks)
    y_full = Y_cfg.reshape(-1)
    keep = np.isfinite(y_full)
    train_mask = np.ones(X.shape[0], dtype=bool)
    held_out = set(range(n_cfg)) - set(train_cfg_ids)
    for c in held_out:
        train_mask[c * N_BINS:(c + 1) * N_BINS] = False
    train_keep = train_mask & keep
    if train_keep.sum() < X.shape[1] + 1:
        return None
    try:
        beta, *_ = np.linalg.lstsq(X[train_keep], y_full[train_keep], rcond=None)
    except np.linalg.LinAlgError:
        return None
    y_pred = X @ beta
    y_resid = y_full - y_pred
    return y_resid.reshape(n_cfg, N_BINS)


# ── Fit functions per (variant × ctrl-mode) ───────────────────────────
def _fit_variant(Y_cfg, loc_cfg, btn_cfg, lags_deg,
                  with_controls, include_loc_now_in_ctrl,
                  n_perms, rng):
    """Generic rate-map LOO CV. `lags_deg` may be a single-lag list
    (per-lag fit, 9-d) or multi-lag (stacked, 9*len(lags)-d).

    With controls: partial out state + buttons (+ location-now when
    include_loc_now_in_ctrl) PER FOLD — the residualisation is refit on
    each training set so the held-out config's residual is leakage-free.
    """
    n_cfg = Y_cfg.shape[0]
    if n_cfg < 2:
        return np.nan, np.nan
    # We need to refit controls per fold for leakage-free residuals.
    # For efficiency, when with_controls=False the rate maps don't change
    # across folds, so build them ONCE and call the CV inline.
    if not with_controls:
        rate_per_cfg  = np.stack([
            _stacked_rate_map(Y_cfg[c], loc_cfg[c], lags_deg)[0]
            for c in range(n_cfg)
        ])
        dwell_per_cfg = np.stack([
            _stacked_rate_map(Y_cfg[c], loc_cfg[c], lags_deg)[1]
            for c in range(n_cfg)
        ])
        return _ratemap_cv_one_setup(
            rate_per_cfg, dwell_per_cfg, loc_cfg, Y_cfg, lags_deg,
            n_perms, rng,
        )

    # Controls mode: rate maps depend on training set → refit per fold.
    fold_rs = []
    perm_r_per_fold = (np.full((n_perms, n_cfg), np.nan)
                       if n_perms > 0 else None)
    for held in range(n_cfg):
        train_cfg = [c for c in range(n_cfg) if c != held]
        Y_resid = _partial_out_controls(
            Y_cfg, loc_cfg, btn_cfg, train_cfg,
            include_loc_now=include_loc_now_in_ctrl,
        )
        if Y_resid is None:
            continue
        # Build per-config rate maps on residual y
        rate_per_cfg = np.stack([
            _stacked_rate_map(Y_resid[c], loc_cfg[c], lags_deg)[0]
            for c in range(n_cfg)
        ])
        dwell_per_cfg = np.stack([
            _stacked_rate_map(Y_resid[c], loc_cfg[c], lags_deg)[1]
            for c in range(n_cfg)
        ])
        train_r = rate_per_cfg[train_cfg]
        train_d = dwell_per_cfg[train_cfg]
        total_d = train_d.sum(axis=0)
        with np.errstate(invalid='ignore'):
            predicted = np.nansum(train_r * train_d, axis=0) / np.where(
                total_d > 0, total_d, np.nan)
        observed = rate_per_cfg[held]
        held_d   = dwell_per_cfg[held]
        weights  = (np.minimum(total_d, held_d) if WEIGHTED_CORRELATION
                     else (np.minimum(total_d, held_d) >= 1).astype(float))
        keep = ((np.minimum(total_d, held_d) >= MIN_DWELL_BINS)
                & np.isfinite(predicted) & np.isfinite(observed))
        if keep.sum() < MIN_SHARED_LOCS:
            continue
        if WEIGHTED_CORRELATION:
            r = _weighted_pearson_r(predicted[keep], observed[keep], weights[keep])
        else:
            r = _pearson_r(predicted[keep], observed[keep])
        if np.isfinite(r):
            fold_rs.append(r)
        if n_perms > 0:
            for p in range(n_perms):
                shift = int(rng.integers(N_BINS // 12, N_BINS - N_BINS // 12))
                loc_sh = np.roll(loc_cfg[held], shift)
                obs_p, dwell_p = _stacked_rate_map(Y_resid[held], loc_sh,
                                                    lags_deg)
                w_p = (np.minimum(total_d, dwell_p) if WEIGHTED_CORRELATION
                        else (np.minimum(total_d, dwell_p) >= 1).astype(float))
                kp = ((np.minimum(total_d, dwell_p) >= MIN_DWELL_BINS)
                       & np.isfinite(predicted) & np.isfinite(obs_p))
                if kp.sum() < MIN_SHARED_LOCS:
                    continue
                if WEIGHTED_CORRELATION:
                    rp = _weighted_pearson_r(predicted[kp], obs_p[kp], w_p[kp])
                else:
                    rp = _pearson_r(predicted[kp], obs_p[kp])
                perm_r_per_fold[p, held] = rp

    if not fold_rs:
        return np.nan, np.nan
    mean_r = float(np.mean(fold_rs))
    if n_perms == 0:
        return mean_r, np.nan
    null_means = np.nanmean(perm_r_per_fold, axis=1)
    null_means = null_means[np.isfinite(null_means)]
    if null_means.size == 0:
        return mean_r, np.nan
    pval = (np.sum(null_means >= mean_r) + 1) / (null_means.size + 1)
    return mean_r, float(pval)


# ── Per-cell wrapper ──────────────────────────────────────────────────
def analyse_cell(neuron_id, arr_clean, idx_cfg, locs, btns_int, n_cfg,
                  roi, sub_str, mni, seed):
    rng = np.random.default_rng(seed)
    Y_cfg, loc_cfg, btn_cfg = _build_per_cfg_sequences(
        arr_clean, idx_cfg, locs, btns_int, n_cfg,
    )
    row = {
        'neuron': neuron_id, 'roi': roi, 'subject_id': sub_str,
        'MNI_x': float(mni[0]) if mni and np.isfinite(mni[0]) else np.nan,
        'MNI_y': float(mni[1]) if mni and np.isfinite(mni[1]) else np.nan,
        'MNI_z': float(mni[2]) if mni and np.isfinite(mni[2]) else np.nan,
        'n_cfg': int(n_cfg),
    }
    # PER-LAG sweep ---------------------------------------------------
    for lag in LAGS_DEG:
        for with_ctrl in (False, True):
            tag = f'lag{lag:03d}_{"ctrl" if with_ctrl else "noctrl"}'
            r, p = _fit_variant(
                Y_cfg, loc_cfg, btn_cfg, [lag],
                with_controls=with_ctrl,
                include_loc_now_in_ctrl=(with_ctrl and lag != 0),
                n_perms=N_PERMUTATIONS, rng=rng,
            )
            row[f'r_{tag}'] = r
            row[f'p_{tag}'] = p
    # AGGREGATE: dsrfull and dsrinf, both ctrl modes ----------------
    for variant, lags in (('dsrfull', DSR_FULL_LAGS_DEG),
                          ('dsrinf',  DSR_INF_LAGS_DEG)):
        for with_ctrl in (False, True):
            tag = f'{variant}_{"ctrl" if with_ctrl else "noctrl"}'
            r, p = _fit_variant(
                Y_cfg, loc_cfg, btn_cfg, lags,
                with_controls=with_ctrl,
                include_loc_now_in_ctrl=with_ctrl,   # 0 not in lags → safe
                n_perms=N_PERMUTATIONS, rng=rng,
            )
            row[f'r_{tag}'] = r
            row[f'p_{tag}'] = p
    return row


# ── ROI loop ──────────────────────────────────────────────────────────
def _filter_buttons(buttons_raw, keep_mask):
    btn_df = buttons_raw.loc[keep_mask.to_numpy()].reset_index(drop=True)
    vals = btn_df.values
    out = np.full(vals.shape, 99, dtype=int)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if isinstance(v, str):
                code = BTTN_LABEL_TO_CODE.get(v, None)
                if code is not None:
                    out[i, j] = code
            elif isinstance(v, (int, np.integer)):
                out[i, j] = int(v)
    return out


def run_roi(roi, cells_df, n_perms=N_PERMUTATIONS,
             n_jobs=N_JOBS, seed=RANDOM_SEED):
    print(f'\n══ ROI: {roi} ══')
    sub_list = sorted(cells_df.loc[cells_df['roi'] == roi, 'subject_id'].unique())
    print(f'  {(cells_df["roi"] == roi).sum()} cells across {len(sub_list)} subjects')

    tasks = []
    for sub_str in sub_list:
        try:
            data_raw = hh.load_norm_data(DATA_DIR, [sub_str], res_data=False)
        except Exception as exc:
            print(f'  sub-{sub_str} load failed: {exc}'); continue
        if not data_raw:
            continue
        beh_raw = data_raw[f'sub-{sub_str}']['beh']
        buttons_raw = data_raw[f'sub-{sub_str}']['buttons']
        keep_mask = beh_raw[['correct', 'rep_correct']].ne(0).any(axis=1)
        btns_int_all = _filter_buttons(buttons_raw, keep_mask)
        data = hh.filter_data(data_raw, int(sub_str), TRIALS)
        sub_dict = data[f'sub-{sub_str}']
        beh = sub_dict['beh'].copy().reset_index(drop=True)
        locs = sub_dict['locations'].to_numpy(dtype=float)
        if btns_int_all.shape[0] != locs.shape[0]:
            print(f'  sub-{sub_str} button/loc row mismatch '
                  f'({btns_int_all.shape[0]} vs {locs.shape[0]}); skipping')
            continue
        uniq, _, idx_cfg, _ = np.unique(
            beh[['loc_A', 'loc_B', 'loc_C', 'loc_D']].to_numpy(),
            axis=0, return_index=True, return_inverse=True, return_counts=True,
        )
        n_cfg = len(np.unique(idx_cfg))
        if n_cfg < 5:
            print(f'  sub-{sub_str} skipped ({n_cfg} configs)'); continue
        sub_keep = set(cells_df.loc[
            (cells_df['subject_id'] == sub_str) & (cells_df['roi'] == roi),
            'cell_idx'].tolist())
        for nid, n_df in sub_dict['normalised_neurons'].items():
            _, ci = cs.parse_neuron_label(nid)
            if ci not in sub_keep:
                continue
            arr = n_df.to_numpy(dtype=float)
            if PHASE_RESIDUALISE:
                arr = _residualise_phase(arr, basis=PHASE_RESIDUALISE)
            roi_row = cells_df[
                (cells_df['subject_id'] == sub_str)
                & (cells_df['cell_idx'] == ci)
            ].iloc[0]
            mni = (
                float(roi_row.get('MNI_x', np.nan)),
                float(roi_row.get('MNI_y', np.nan)),
                float(roi_row.get('MNI_z', np.nan)),
            )
            tasks.append(dict(
                neuron_id=nid, arr_clean=arr.copy(),
                idx_cfg=idx_cfg.copy(), locs=locs.copy(),
                btns_int=btns_int_all.copy(), n_cfg=int(n_cfg),
                roi=roi, sub_str=sub_str, mni=mni,
                seed=seed + (hash(nid) & 0x7FFFFFFF),
            ))

    if not tasks:
        return None
    print(f'  Fitting {len(tasks)} cells ({n_perms} perms each)…')
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(analyse_cell)(**t) for t in tasks
    )
    return pd.DataFrame(rows)


# ── Stats ─────────────────────────────────────────────────────────────
def _ttest_gt0(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, np.nan
    try:
        res = stats.ttest_1samp(x, 0.0, alternative='greater')
        return float(res.statistic), float(res.pvalue)
    except TypeError:
        t, p2 = stats.ttest_1samp(x, 0.0)
        return float(t), float(p2 / 2 if t > 0 else 1 - p2 / 2)


def _binom_gt_alpha(k, n, alpha=ALPHA):
    if n == 0:
        return np.nan
    try:
        return float(stats.binomtest(k, n, p=alpha, alternative='greater').pvalue)
    except AttributeError:
        return float(stats.binom_test(k, n, p=alpha, alternative='greater'))


def _wilcoxon_perm_p(p_vals):
    p = np.asarray(p_vals, dtype=float); p = p[np.isfinite(p)]
    if p.size < 5:
        return np.nan
    try:
        return float(stats.wilcoxon(p - 0.5, alternative='less').pvalue)
    except ValueError:
        return np.nan


def per_roi_stats(df, ctrl_mode):
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    rows = []
    for roi, g in df.groupby('roi'):
        n = len(g)
        rec = {'roi': roi, 'ctrl_mode': tag, 'n_cells': n}
        # T1 + T3 + T4 per lag ----------------------------------------
        for lag in LAGS_DEG:
            r = g[f'r_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
            p = g[f'p_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
            t1_t, t1_p = _ttest_gt0(r)
            k_sig = int(np.sum(np.isfinite(p) & (p < ALPHA)))
            rec[f'T1_t_lag{lag:03d}']     = t1_t
            rec[f'T1_p_lag{lag:03d}']     = t1_p
            rec[f'T1_meanR_lag{lag:03d}'] = float(np.nanmean(r))
            rec[f'T3_k_lag{lag:03d}']     = k_sig
            rec[f'T3_p_lag{lag:03d}']     = _binom_gt_alpha(k_sig, n)
            rec[f'T4_p_lag{lag:03d}']     = _wilcoxon_perm_p(p)
        # T2 within-cell paired predicted vs other --------------------
        pred_lags = ROI_PREDICTED_LAGS_DEG.get(roi, None)
        if pred_lags:
            r_mat = np.stack([
                g[f'r_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
                for lag in LAGS_DEG
            ], axis=1)
            idx_pred  = [LAGS_DEG.index(l) for l in pred_lags]
            idx_other = [i for i in range(len(LAGS_DEG)) if i not in idx_pred]
            tgt   = np.nanmean(r_mat[:, idx_pred],  axis=1)
            other = np.nanmean(r_mat[:, idx_other], axis=1)
            diff = tgt - other
            m = np.isfinite(diff)
            if m.sum() >= 2:
                res = stats.ttest_1samp(diff[m], 0.0, alternative='greater')
                rec.update({'T2_t': float(res.statistic),
                            'T2_p': float(res.pvalue),
                            'T2_meanDiff': float(diff[m].mean()),
                            'T2_n': int(m.sum())})
            else:
                rec.update({'T2_t': np.nan, 'T2_p': np.nan,
                            'T2_meanDiff': np.nan, 'T2_n': 0})
        else:
            rec.update({'T2_t': np.nan, 'T2_p': np.nan,
                        'T2_meanDiff': np.nan, 'T2_n': 0})
        # T5: dsrfull and dsrinf primary tests ------------------------
        for variant in ('dsrfull', 'dsrinf'):
            r = g[f'r_{variant}_{tag}'].to_numpy(dtype=float)
            p = g[f'p_{variant}_{tag}'].to_numpy(dtype=float)
            t, p_val = _ttest_gt0(r)
            k_sig = int(np.sum(np.isfinite(p) & (p < ALPHA)))
            rec[f'T5_{variant}_t']     = t
            rec[f'T5_{variant}_p']     = p_val
            rec[f'T5_{variant}_meanR'] = float(np.nanmean(r))
            rec[f'T3_{variant}_k']     = k_sig
            rec[f'T3_{variant}_p']     = _binom_gt_alpha(k_sig, n)
            rec[f'T4_{variant}_p']     = _wilcoxon_perm_p(p)
        # T6: dsrinf r vs r at first predicted lag (paired) ----------
        if pred_lags:
            r_inf  = g[f'r_dsrinf_{tag}'].to_numpy(dtype=float)
            r_pred = g[f'r_lag{pred_lags[0]:03d}_{tag}'].to_numpy(dtype=float)
            diff = r_inf - r_pred
            m = np.isfinite(diff)
            if m.sum() >= 2:
                res = stats.ttest_1samp(diff[m], 0.0, alternative='greater')
                rec.update({'T6_t': float(res.statistic),
                            'T6_p': float(res.pvalue),
                            'T6_meanDiff': float(diff[m].mean())})
            else:
                rec.update({'T6_t': np.nan, 'T6_p': np.nan,
                            'T6_meanDiff': np.nan})
        else:
            rec.update({'T6_t': np.nan, 'T6_p': np.nan, 'T6_meanDiff': np.nan})
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # BH-FDR within each family ---------------------------------------
    fam_t1 = [c for c in out.columns if c.startswith('T1_p_lag')]
    fam_t3 = [c for c in out.columns if c.startswith('T3_p_lag')]
    fam_t4 = [c for c in out.columns if c.startswith('T4_p_lag')]
    fam_t3_agg = [c for c in out.columns
                  if c.startswith('T3_') and c.endswith('_p') and 'lag' not in c]
    fam_t4_agg = [c for c in out.columns
                  if c.startswith('T4_') and c.endswith('_p') and 'lag' not in c]
    fam_t5 = [c for c in out.columns if c.startswith('T5_') and c.endswith('_p')]
    for fam in (fam_t1, fam_t3, fam_t4, fam_t3_agg, fam_t4_agg, fam_t5):
        for col in fam:
            out[col + '_fdr'] = _bh_fdr(out[col].to_numpy(dtype=float))
    for col in ('T2_p', 'T6_p'):
        if col in out.columns:
            out[col + '_fdr'] = _bh_fdr(out[col].to_numpy(dtype=float))
    return out


# ── Figures ───────────────────────────────────────────────────────────
def fig_roi_lag_heatmap(roi_stats, ctrl_mode, save_stem):
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    df = roi_stats[roi_stats['ctrl_mode'] == tag]
    rois = [r for r in ROIS_TO_RUN if r in df['roi'].values]
    T = np.full((len(rois), len(LAGS_DEG)), np.nan)
    Q = np.full((len(rois), len(LAGS_DEG)), np.nan)
    for ri, roi in enumerate(rois):
        rs = df[df['roi'] == roi].iloc[0]
        for li, lag in enumerate(LAGS_DEG):
            T[ri, li] = rs[f'T1_t_lag{lag:03d}']
            Q[ri, li] = rs.get(f'T1_p_lag{lag:03d}_fdr', np.nan)
    pred_str = ', '.join(f'{r}={list(ROI_PREDICTED_LAGS_DEG[r])}°'
                          for r in ROI_PREDICTED_LAGS_DEG)
    plot_roi_lag_tstat_heatmap(
        T, LAGS_DEG, rois, q_matrix=Q,
        predicted_lags_per_roi=ROI_PREDICTED_LAGS_DEG,
        save_stem=save_stem,
        title=(f'Fig 3a — ROI × lag t-stat vs 0  [{tag}]\n'
                f'Predicted lags (black outline): {pred_str}'),
    )


def fig_roi_lag_curves(per_cell, ctrl_mode, save_stem):
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    rois = [r for r in ROIS_TO_RUN if r in per_cell['roi'].unique()]
    curves = {}
    for roi in rois:
        g = per_cell[per_cell['roi'] == roi]
        curves[roi] = np.stack([
            g[f'r_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
            for lag in LAGS_DEG
        ], axis=1)
    plot_roi_lag_curves(
        curves, LAGS_DEG,
        predicted_lags_per_roi=ROI_PREDICTED_LAGS_DEG,
        roi_colours=ROI_COLOURS,
        save_stem=save_stem,
        title=f'Per-ROI mean CV r across lags  [{tag}]',
    )


def fig_perm_sig_fraction_bar(roi_stats, ctrl_mode, save_stem):
    """Best-of-predicted-lags perm-sig fraction per ROI (single number per ROI)."""
    _set_rc()
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    df = roi_stats[roi_stats['ctrl_mode'] == tag]
    rois = [r for r in ROIS_TO_RUN if r in df['roi'].values]
    fig, ax = plt.subplots(figsize=(12 * CM, 5 * CM), constrained_layout=True)
    x = np.arange(len(rois))
    fracs, q_best = [], []
    for roi in rois:
        rs = df[df['roi'] == roi].iloc[0]
        pred = ROI_PREDICTED_LAGS_DEG.get(roi, LAGS_DEG)
        ks = [rs[f'T3_k_lag{l:03d}'] for l in pred]
        qs = [rs.get(f'T3_p_lag{l:03d}_fdr', np.nan) for l in pred]
        fracs.append(max(ks) / rs['n_cells'] if rs['n_cells'] > 0 else np.nan)
        q_best.append(min([q for q in qs if np.isfinite(q)], default=np.nan))
    ax.bar(x, fracs, color=[ROI_COLOURS.get(r, '#888') for r in rois],
           edgecolor='black', linewidth=0.4, width=0.7)
    ax.axhline(ALPHA, color='black', lw=0.6, ls='--',
               label=f'chance = α = {ALPHA}')
    for xi, (f_, q_) in enumerate(zip(fracs, q_best)):
        s = p_to_stars(q_) if np.isfinite(q_) else ''
        if s and np.isfinite(f_):
            ax.text(xi, f_ + 0.005, s, ha='center', va='bottom',
                    fontsize=FONT_BIG, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=35, ha='right', fontsize=FONT_TICK)
    ax.set_ylabel(f'frac cells with p_perm < {ALPHA}', fontsize=FONT_AXIS)
    ax.set_title(f'Perm-sig fraction at predicted lag(s)  [{tag}]',
                 fontsize=FONT_TICK)
    ax.legend(fontsize=FONT_TICK - 1, frameon=False)
    _save(fig, save_stem)


def fig_perm_sig_fraction_heatmap(roi_stats, ctrl_mode, save_stem):
    """Fig 3b — ROI × lag fraction-of-perm-sig-cells heatmap with FDR stars
    and predicted-lag outlines.
    """
    _set_rc()
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    df = roi_stats[roi_stats['ctrl_mode'] == tag]
    rois = [r for r in ROIS_TO_RUN if r in df['roi'].values]
    M = np.full((len(rois), len(LAGS_DEG)), np.nan)
    Q = np.full((len(rois), len(LAGS_DEG)), np.nan)
    for ri, roi in enumerate(rois):
        rs = df[df['roi'] == roi].iloc[0]
        n = rs['n_cells']
        for li, lag in enumerate(LAGS_DEG):
            M[ri, li] = rs[f'T3_k_lag{lag:03d}'] / n if n > 0 else np.nan
            Q[ri, li] = rs.get(f'T3_p_lag{lag:03d}_fdr', np.nan)
    fig, ax = plt.subplots(figsize=(14 * CM, max(3.5, 0.55 * len(rois)) * CM),
                            constrained_layout=True)
    vmax = max(0.30, float(np.nanmax(M)) if np.isfinite(M).any() else 0.30)
    im = ax.imshow(M, cmap='Reds', vmin=0, vmax=vmax, aspect='auto')
    # Predicted-lag outlines + chance dashed level via text annotation
    for ri, roi in enumerate(rois):
        for tl in ROI_PREDICTED_LAGS_DEG.get(roi, ()):
            if tl in LAGS_DEG:
                ci = LAGS_DEG.index(tl)
                ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                            fill=False, edgecolor='black',
                                            lw=1.2))
    for ri in range(len(rois)):
        for ci in range(len(LAGS_DEG)):
            q = Q[ri, ci]
            if np.isfinite(q):
                s = p_to_stars(q)
                if s:
                    col = 'white' if M[ri, ci] > vmax * 0.55 else 'black'
                    ax.text(ci, ri, s, ha='center', va='center',
                            fontsize=FONT_TICK, fontweight='bold', color=col)
    ax.set_xticks(range(len(LAGS_DEG)))
    ax.set_xticklabels([str(l) for l in LAGS_DEG], fontsize=FONT_TICK)
    ax.set_yticks(range(len(rois)))
    ax.set_yticklabels(rois, fontsize=FONT_TICK)
    ax.set_xlabel('lag (°)', fontsize=FONT_AXIS)
    ax.set_title(f'Fig 3b — ROI × lag fraction of cells with p_perm < {ALPHA}\n'
                  f'[{tag}]  (chance = α = {ALPHA}; stars = BH-FDR binomial)',
                  fontsize=FONT_TICK)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cb.set_label('frac perm-sig', fontsize=FONT_TICK)
    cb.ax.tick_params(labelsize=FONT_TICK)
    _save(fig, save_stem)


def fig_dsrfull_vs_dsrinf_scatter(per_cell, ctrl_mode, save_stem):
    """Per-ROI scatter: dsrfull r vs dsrinf r. Replaces winner-curse Fig 4."""
    _set_rc()
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    rois = [r for r in ROIS_TO_RUN if r in per_cell['roi'].unique()]
    n_cols = min(len(rois), 4)
    n_rows = int(np.ceil(len(rois) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * CM * n_cols, 3.5 * CM * n_rows),
                              constrained_layout=True, squeeze=False)
    for ax in axes.ravel()[len(rois):]:
        ax.axis('off')
    for ax, roi in zip(axes.ravel(), rois):
        g = per_cell[per_cell['roi'] == roi]
        full = g[f'r_dsrfull_{tag}'].to_numpy(dtype=float)
        inf_ = g[f'r_dsrinf_{tag}'].to_numpy(dtype=float)
        m = np.isfinite(full) & np.isfinite(inf_)
        col = ROI_COLOURS.get(roi, '#888')
        ax.scatter(full[m], inf_[m], s=6, color=col, alpha=0.7, edgecolor='none')
        if m.any():
            lim_lo = float(np.nanmin([full[m].min(), inf_[m].min()]))
            lim_hi = float(np.nanmax([full[m].max(), inf_[m].max()]))
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                    color='gray', lw=0.5, ls='--')
        ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
        ax.set_title(roi, fontsize=FONT_TICK)
        ax.tick_params(labelsize=FONT_TICK - 1, length=1.5, pad=1)
        ax.set_xlabel('dsr_full r', fontsize=FONT_TICK)
        ax.set_ylabel('dsr_inf r', fontsize=FONT_TICK)
    fig.suptitle(f'dsr_full (11 lags) vs dsr_inf (30/60/90) CV r  [{tag}]',
                 fontsize=FONT_AXIS)
    _save(fig, save_stem)


def fig_ctrl_vs_noctrl_scatter(per_cell, save_stem):
    _set_rc()
    rois = [r for r in ROIS_TO_RUN if r in per_cell['roi'].unique()]
    n_cols = min(len(rois), 4)
    n_rows = int(np.ceil(len(rois) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * CM * n_cols, 3.5 * CM * n_rows),
                              constrained_layout=True, squeeze=False)
    for ax in axes.ravel()[len(rois):]:
        ax.axis('off')
    for ax, roi in zip(axes.ravel(), rois):
        g = per_cell[per_cell['roi'] == roi]
        pred = ROI_PREDICTED_LAGS_DEG.get(roi, [0])
        lag = pred[0]
        noc = g[f'r_lag{lag:03d}_noctrl'].to_numpy(dtype=float)
        ctl = g[f'r_lag{lag:03d}_ctrl'].to_numpy(dtype=float)
        m = np.isfinite(noc) & np.isfinite(ctl)
        col = ROI_COLOURS.get(roi, '#888')
        ax.scatter(noc[m], ctl[m], s=6, color=col, alpha=0.7, edgecolor='none')
        if m.any():
            lim_lo = float(np.nanmin([noc[m].min(), ctl[m].min()]))
            lim_hi = float(np.nanmax([noc[m].max(), ctl[m].max()]))
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                    color='gray', lw=0.5, ls='--')
        ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
        ax.set_title(f'{roi}  lag={lag}°', fontsize=FONT_TICK)
        ax.tick_params(labelsize=FONT_TICK - 1, length=1.5, pad=1)
        ax.set_xlabel('r without controls', fontsize=FONT_TICK)
        ax.set_ylabel('r with controls', fontsize=FONT_TICK)
    fig.suptitle('Per-cell CV r at first predicted lag — controls vs no-controls',
                 fontsize=FONT_AXIS)
    _save(fig, save_stem)


def fig_per_roi_r_hist(per_cell, ctrl_mode, save_stem):
    _set_rc()
    tag = 'ctrl' if ctrl_mode else 'noctrl'
    rois = [r for r in ROIS_TO_RUN if r in per_cell['roi'].unique()]
    n_cols = min(len(rois), 4)
    n_rows = int(np.ceil(len(rois) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * CM * n_cols, 2.5 * CM * n_rows),
                              constrained_layout=True, squeeze=False)
    for ax in axes.ravel()[len(rois):]:
        ax.axis('off')
    for ax, roi in zip(axes.ravel(), rois):
        g = per_cell[per_cell['roi'] == roi]
        pred = ROI_PREDICTED_LAGS_DEG.get(roi, [0])
        lag = pred[0]
        r = g[f'r_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
        p = g[f'p_lag{lag:03d}_{tag}'].to_numpy(dtype=float)
        m = np.isfinite(r); r = r[m]; p = p[m]
        col = ROI_COLOURS.get(roi, '#888')
        if r.size:
            lo, hi = np.nanpercentile(r, [1, 99])
            bins = np.linspace(lo, hi, 20)
            ax.hist(r, bins=bins, color='lightgray',
                    edgecolor='black', linewidth=0.2, alpha=0.7)
            sig = p < ALPHA
            if sig.any():
                ax.hist(r[sig], bins=bins, color=col,
                        edgecolor='black', linewidth=0.2, alpha=0.95)
            ax.axvline(r.mean(), color='black', lw=0.8)
        ax.axvline(0, color='gray', ls='--', lw=0.4)
        ax.set_title(f'{roi}  lag={lag}°', fontsize=FONT_TICK)
        ax.tick_params(labelsize=FONT_TICK - 1, length=1.5, pad=1)
        ax.set_xlabel('CV r', fontsize=FONT_TICK)
        ax.set_ylabel('# cells', fontsize=FONT_TICK)
    fig.suptitle(f'Per-ROI CV r distribution at predicted lag  [{tag}]',
                 fontsize=FONT_AXIS)
    _save(fig, save_stem)


# ── Main ──────────────────────────────────────────────────────────────
def _load_cells():
    cells = cs.load_cells(cell_set='all_in_roi_table', rois_keep=ROIS_TO_RUN)
    cells = cells.copy()
    cells['subject_id'] = cells['subject_id'].astype(str).str.zfill(2)
    if 'cell_idx' not in cells.columns:
        raise ValueError('cells table missing cell_idx column')
    for c in ('MNI_x', 'MNI_y', 'MNI_z'):
        if c not in cells.columns:
            cells[c] = np.nan
    return cells


def main():
    np.random.seed(RANDOM_SEED)
    run_tag = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = os.path.join(OUT_BASE, run_tag)
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    print(f'Output dir: {out_dir}')

    config = {
        'run_tag':              run_tag,
        'method':               'lag_shifted_rate_map_weighted_pearson',
        'rois_to_run':          ROIS_TO_RUN,
        'phase_residualise':    PHASE_RESIDUALISE,
        'trials':               TRIALS,
        'lags_deg':             LAGS_DEG,
        'dsr_full_lags_deg':    DSR_FULL_LAGS_DEG,
        'dsr_inf_lags_deg':     DSR_INF_LAGS_DEG,
        'lag_bins_bttn_next':   LAG_BINS_BTTN_NEXT,
        'min_dwell_bins':       MIN_DWELL_BINS,
        'min_shared_locs':      MIN_SHARED_LOCS,
        'weighted_correlation': WEIGHTED_CORRELATION,
        'n_permutations':       N_PERMUTATIONS,
        'random_seed':          RANDOM_SEED,
        'alpha':                ALPHA,
        'roi_predicted_lags':   {k: list(v) for k, v in
                                   ROI_PREDICTED_LAGS_DEG.items()},
        'controls':             ['state', 'bttn_curr', 'bttn_next',
                                  'location_at_lag_0_when_lag!=0'],
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    cells = _load_cells()
    all_dfs = []
    for roi in ROIS_TO_RUN:
        df = run_roi(roi, cells)
        if df is None:
            continue
        df.to_csv(os.path.join(out_dir, f'per_cell_{roi}.csv'), index=False)
        all_dfs.append(df)
    if not all_dfs:
        print('No results.'); return
    per_cell = pd.concat(all_dfs, ignore_index=True)
    per_cell.to_csv(os.path.join(out_dir, 'per_cell_ALL_ROIs.csv'), index=False)

    stats_dfs = [per_roi_stats(per_cell, ctrl_mode=False),
                 per_roi_stats(per_cell, ctrl_mode=True)]
    roi_stats = pd.concat(stats_dfs, ignore_index=True)
    roi_stats.to_csv(os.path.join(out_dir, 'per_roi_stats.csv'), index=False)
    print(f'\nSaved per_roi_stats.csv ({roi_stats.shape[0]} rows × '
          f'{roi_stats.shape[1]} cols)')

    long_rows = []
    for ctrl in (False, True):
        tag = 'ctrl' if ctrl else 'noctrl'
        rs = roi_stats[roi_stats['ctrl_mode'] == tag]
        for _, row in rs.iterrows():
            for lag in LAGS_DEG:
                long_rows.append({
                    'roi': row['roi'], 'ctrl_mode': tag, 'lag_deg': lag,
                    'mean_r':     row[f'T1_meanR_lag{lag:03d}'],
                    't_vs_0':     row[f'T1_t_lag{lag:03d}'],
                    'p_unc':      row[f'T1_p_lag{lag:03d}'],
                    'p_fdr':      row.get(f'T1_p_lag{lag:03d}_fdr', np.nan),
                    'k_perm_sig': int(row[f'T3_k_lag{lag:03d}']),
                    'p_binom':    row[f'T3_p_lag{lag:03d}'],
                    'p_binom_fdr':row.get(f'T3_p_lag{lag:03d}_fdr', np.nan),
                    'n_cells':    int(row['n_cells']),
                })
    pd.DataFrame(long_rows).to_csv(
        os.path.join(out_dir, 'roi_lag_table.csv'), index=False)

    for ctrl in (False, True):
        tag = 'ctrl' if ctrl else 'noctrl'
        fig_roi_lag_heatmap(roi_stats, ctrl,
            os.path.join(fig_dir, f'01_roi_lag_heatmap_{tag}'))
        fig_roi_lag_curves(per_cell, ctrl,
            os.path.join(fig_dir, f'02_roi_lag_curves_{tag}'))
        fig_perm_sig_fraction_bar(roi_stats, ctrl,
            os.path.join(fig_dir, f'03_perm_sig_fraction_bar_{tag}'))
        fig_perm_sig_fraction_heatmap(roi_stats, ctrl,
            os.path.join(fig_dir, f'03b_perm_sig_fraction_heatmap_{tag}'))
        fig_dsrfull_vs_dsrinf_scatter(per_cell, ctrl,
            os.path.join(fig_dir, f'04_dsrfull_vs_dsrinf_scatter_{tag}'))
        fig_per_roi_r_hist(per_cell, ctrl,
            os.path.join(fig_dir, f'06_per_roi_r_hist_{tag}'))
    fig_ctrl_vs_noctrl_scatter(per_cell,
        os.path.join(fig_dir, '05_ctrl_vs_noctrl_scatter'))

    print(f'\nDone. Outputs in {out_dir}')


if __name__ == '__main__':
    main()
