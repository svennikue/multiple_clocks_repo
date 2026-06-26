"""Future-spatial-peaks compute.

For each cell:

  1. (optional) residualise firing rate against within-state PHASE using a
     cosine basis — done on the RAW per-bin firing rate BEFORE any rate
     maps are computed.

  2. (optional) PAIR grids to maximise spatial coverage. Each pair becomes
     a single "grid group" that the rate map is computed on.

  3. For every LAG in `LAGS_DEG`, build per-grid-group rate maps by rolling
     the location series by that lag and averaging firing rate per
     location (1..9). Compute pairwise consistency across grid groups
     (mean Pearson r over all pairs).

  4. Leave-one-grid-group-out cross-validation:

      * `cv_fixed_lag`        — evaluate at PRE-SPECIFIED lag(s) (per-ROI
                                 a-priori hypothesis). No training-data
                                 peak selection.
      * `cv_free_peak`        — pick the best lag on TRAINING grids, then
                                 validate at that lag on the held-out
                                 grid. The original "cell chooses own
                                 peak" control.

  5. Permutation null: per-rep circular shift of the location series.

Significance toolkit lives at the bottom (`ttest_mean_r_vs_zero`,
`within_cell_lag_vs_others`, `binomial_sig_fraction`, `bh_fdr`).

This module is pure compute. No IO, no plotting.

A-priori per-ROI predicted lags:
  * ACC          → 30° or 60° (action / planning lookahead)
  * HC_anterior  → 0°         (current location / place)
  * HC_mid       → 0° or 330° (current ± one bin)
  * any other    → no a-priori; report free-peak control

@author: Svenja Küchenhoff (refactor: Claude)
"""
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import mc.analyse.helpers_human_cells as hh

# import pdb; pdb.set_trace()

# ====================================================================
# SETTINGS — defaults for this analysis. Override via the `cfg` dict
# passed to `analyse_one_cell` if you want a non-default run.
# ====================================================================
N_BINS_PER_TRIAL = 360
STATE_LEN        = 90                       # bins per state (4 states × 90)
N_LOC            = 9                        # 3×3 grid
LAG_STEP_DEG     = 30                       # 30° → 12 lags per loop
LAGS_DEG         = list(range(0, 360, LAG_STEP_DEG))   # [0, 30, …, 330]
MIN_DWELL_BINS   = 25                       # min samples per location to
                                            # accept the rate estimate
MIN_SHARED_LOCS  = 5                        # min co-finite locations
                                            # required to accept a
                                            # grid-pair correlation
WEIGHTED_CORRELATION = True                 # weight by dwell sum
COVERAGE_MODE        = 'paired'             # pair grids to increase
                                            # spatial coverage
SPARSITY_FILTER      = 'gridwise_qc'        # mark low-quality grid blocks
                                            # as excluded (-1)
PHASE_BASIS          = 'cosine'             # the only supported option
N_PERMUTATIONS       = 200
RANDOM_SEED          = 42

# A-priori per-ROI predicted lags (degrees). Used by the FDR-corrected
# fixed-lag analysis. Anything else falls back to free-peak control.
ROI_PREDICTED_LAGS_DEG = {
    'ACC':         (30, 60),
    'HC_anterior': (0,),
    'HC_mid':      (0, 330),
}

ALPHA = 0.05


# ====================================================================
# Core rate-map helpers (vectorised)
# ====================================================================
def rate_map(firing_rate_flat, location_flat, n_loc=N_LOC,
             min_dwell=MIN_DWELL_BINS):
    """Mean firing rate per location, vectorised.

    Parameters
    ----------
    firing_rate_flat, location_flat : 1-D arrays. Same length.
        `location_flat` is integer 1..n_loc; values ≤ 0 or non-finite are
        ignored.
    n_loc : int, number of grid locations (9 for a 3×3 grid).
    min_dwell : int, min sample count per location for a valid estimate
        (locations with fewer samples become NaN).

    Returns
    -------
    mean_firing_rate_per_loc : (n_loc,) array (NaN where dwell < min_dwell)
    dwell_time_per_loc       : (n_loc,) bin count per location
    """
    fr = np.asarray(firing_rate_flat, dtype=float)
    lo = np.asarray(location_flat, dtype=float)
    mask = np.isfinite(fr) & np.isfinite(lo) & (lo > 0)
    fr = fr[mask]
    lo = lo[mask].astype(int) - 1
    keep = (lo >= 0) & (lo < n_loc)
    fr = fr[keep]; lo = lo[keep]
    sums = np.bincount(lo, weights=fr, minlength=n_loc)
    cnts = np.bincount(lo, minlength=n_loc).astype(float)
    safe = np.maximum(cnts, 1.0)
    rates = np.where(cnts >= min_dwell, sums / safe, np.nan)
    return rates, cnts


def _weighted_pearson(x, y, w):
    x = np.asarray(x, float); y = np.asarray(y, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(y) & (w > 0)
    if m.sum() < 2:
        return np.nan
    x = x[m]; y = y[m]; w = w[m]
    mx = np.average(x, weights=w); my = np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx  = np.average((x - mx) ** 2, weights=w)
    vy  = np.average((y - my) ** 2, weights=w)
    if vx <= 0 or vy <= 0:
        return np.nan
    return cov / np.sqrt(vx * vy)


def pairwise_consistency(maps, dwell_per_loc,
                         min_shared_locs=MIN_SHARED_LOCS,
                         weighted=WEIGHTED_CORRELATION,
                         return_n_shared=False):
    """Mean (weighted) Pearson r between rate maps across grid pairs.

    Parameters
    ----------
    maps         : (n_loc, n_grid_groups) rate maps (NaN allowed)
    dwell_per_loc: (n_loc, n_grid_groups) dwell counts (≥ 0)
    """
    n_loc, n_grids = maps.shape
    rs, n_shared = [], []
    for i, j in combinations(range(n_grids), 2):
        m = np.isfinite(maps[:, i]) & np.isfinite(maps[:, j])
        if m.sum() < min_shared_locs:
            continue
        if weighted:
            w_ij = dwell_per_loc[:, i] + dwell_per_loc[:, j]
            r = _weighted_pearson(maps[m, i], maps[m, j], w_ij[m])
        else:
            v1 = maps[m, i]; v2 = maps[m, j]
            if np.std(v1) <= 0 or np.std(v2) <= 0:
                r = np.nan
            else:
                r = float(np.corrcoef(v1, v2)[0, 1])
        if np.isfinite(r):
            rs.append(r); n_shared.append(int(m.sum()))
    mean_r = float(np.nanmean(rs)) if rs else np.nan
    if return_n_shared:
        return mean_r, (float(np.mean(n_shared)) if n_shared else np.nan)
    return mean_r


# ====================================================================
# Per-cell shift curve (all lags × all grid groups)
# ====================================================================
def consistency_per_lag(neurons, locations, grid_group_idx,
                        lags_deg=LAGS_DEG,
                        min_dwell=MIN_DWELL_BINS,
                        min_shared_locs=MIN_SHARED_LOCS,
                        weighted=WEIGHTED_CORRELATION,
                        n_loc=N_LOC):
    """For one cell, return the rate maps + cross-grid consistency at every
    lag.

    Parameters
    ----------
    neurons         : (n_reps, n_bins) firing rates
    locations       : (n_reps, n_bins) integer locations (1..n_loc, NaN ok)
    grid_group_idx  : (n_reps,) grid group id per repeat. Reps with id < 0
                      are dropped.
    lags_deg        : iterable of integer bin offsets

    Returns
    -------
    fr_maps           : (n_lags, n_loc, n_grid_groups) rate maps
    dwell_per_loc     : (n_lags, n_loc, n_grid_groups) dwell counts
    consistency_per_lag : (n_lags,) mean cross-grid consistency per lag
    unique_grid_groups : (n_grid_groups,) sorted group ids used
    """
    neurons   = np.asarray(neurons, dtype=float)
    locations = np.asarray(locations, dtype=float)
    grp = np.asarray(grid_group_idx)
    valid_reps = grp >= 0
    unique_groups = np.unique(grp[valid_reps])
    n_groups = len(unique_groups)
    n_lags = len(lags_deg)

    fr_maps    = np.full((n_lags, n_loc, n_groups), np.nan)
    dwell_maps = np.zeros((n_lags, n_loc, n_groups))
    consistency = np.full(n_lags, np.nan)

    for li, lag in enumerate(lags_deg):
        locs_shifted = np.roll(locations, shift=-int(lag), axis=1)
        for gi, g in enumerate(unique_groups):
            mask = (grp == g)
            fr_flat = neurons[mask].ravel()
            lo_flat = locs_shifted[mask].ravel()
            rm, dw = rate_map(fr_flat, lo_flat, n_loc=n_loc,
                              min_dwell=min_dwell)
            fr_maps[li, :, gi] = rm
            dwell_maps[li, :, gi] = dw
        if n_groups >= 2:
            consistency[li] = pairwise_consistency(
                fr_maps[li], dwell_maps[li],
                min_shared_locs=min_shared_locs, weighted=weighted,
            )
    return fr_maps, dwell_maps, consistency, unique_groups


# ====================================================================
# Cross-validation
#   * cv_free_peak  — pick best lag on train grids (the original control).
#   * cv_fixed_lag  — at a-priori predicted lag(s); no peak selection.
# ====================================================================
def _topk_lags(curve, lags_deg, k):
    """Top-k lags by `curve`. NaNs sort last; ties broken by lag order."""
    curve = np.asarray(curve, float)
    lags = np.asarray(lags_deg)
    order = np.argsort(np.where(np.isnan(curve), -np.inf, curve))[::-1]
    sel = [int(lags[i]) for i in order if np.isfinite(curve[i])][:k]
    while len(sel) < k:
        sel.append(int(lags[order[0]]))
    return sel


def _validate_at_lag(fr_lag_maps, dwell_lag_maps, held_idx, train_mask,
                     min_shared_locs, weighted):
    """Test-vs-train consistency at a single lag for a single fold.
    Returns mean r over train grids and the per-pair shared-locs counts."""
    test_rm = fr_lag_maps[:, held_idx]
    test_dw = dwell_lag_maps[:, held_idx]
    train_rms = fr_lag_maps[:, train_mask]
    train_dws = dwell_lag_maps[:, train_mask]
    rs, shared = [], []
    for k in range(train_rms.shape[1]):
        m = np.isfinite(test_rm) & np.isfinite(train_rms[:, k])
        if m.sum() < min_shared_locs:
            continue
        if weighted:
            w = test_dw + train_dws[:, k]
            rk = _weighted_pearson(test_rm[m], train_rms[m, k], w[m])
        else:
            v1 = test_rm[m]; v2 = train_rms[m, k]
            if np.std(v1) <= 0 or np.std(v2) <= 0:
                rk = np.nan
            else:
                rk = float(np.corrcoef(v1, v2)[0, 1])
        if np.isfinite(rk):
            rs.append(rk); shared.append(int(m.sum()))
    return (float(np.mean(rs)) if rs else np.nan,
            shared)


def cv_free_peak(neurons, locations, grid_group_idx,
                 lags_deg=LAGS_DEG, n_peaks=1,
                 min_dwell=MIN_DWELL_BINS,
                 min_shared_locs=MIN_SHARED_LOCS,
                 weighted=WEIGHTED_CORRELATION,
                 n_loc=N_LOC):
    """Leave-one-grid-group-out CV with PER-FOLD training-data peak selection.

    Per fold: train grids → pick top-`n_peaks` lags by training consistency,
    validate held-out grid at exactly those lags. The original "cell picks
    its own peak" analysis — kept here as a control.

    Returns
    -------
    dict with `peak_r`, `peak_lag_plurality`, `fold_lags`, `fold_rs`,
    `n_groups_used`, `mean_shared_locs_at_peak`, `consistency_curve_full`.
    """
    fr_all, dwell_all, curve_all, groups = consistency_per_lag(
        neurons, locations, grid_group_idx, lags_deg,
        min_dwell=min_dwell, min_shared_locs=min_shared_locs,
        weighted=weighted, n_loc=n_loc,
    )
    n_groups = len(groups)
    if n_groups < 2:
        return {
            "peak_r": np.nan,
            "peak_lag_plurality": np.nan,
            "fold_lags": [],
            "fold_rs": [],
            "n_groups_used": int(n_groups),
            "mean_shared_locs_at_peak": np.nan,
            "consistency_curve_full": curve_all.tolist(),
        }

    lags_arr = np.asarray(list(lags_deg), dtype=int)
    fold_lags, fold_rs, shared_collect = [], [], []

    for held_i in range(n_groups):
        train_mask = np.ones(n_groups, dtype=bool); train_mask[held_i] = False
        # Per-lag training consistency (excluding the held-out grid).
        train_curve = np.full(len(lags_arr), np.nan)
        for li in range(len(lags_arr)):
            train_curve[li] = pairwise_consistency(
                fr_all[li][:, train_mask], dwell_all[li][:, train_mask],
                min_shared_locs=min_shared_locs, weighted=weighted,
            )
        peaks = _topk_lags(train_curve, lags_arr, n_peaks)
        peak_idxs = [int(np.where(lags_arr == p)[0][0]) for p in peaks]

        rs_for_fold = []
        for li in peak_idxs:
            mean_r, shared = _validate_at_lag(
                fr_all[li], dwell_all[li], held_i, train_mask,
                min_shared_locs, weighted,
            )
            rs_for_fold.append(mean_r)
            shared_collect.extend(shared)

        fold_lags.append(peaks)
        fold_rs.append(rs_for_fold)

    peak_r = float(np.nanmean([np.nanmean(r) for r in fold_rs])) \
        if fold_rs else np.nan
    flat = [s for fs in fold_lags for s in fs]
    peak_lag_plurality = Counter(flat).most_common(1)[0][0] if flat else np.nan
    mean_shared = (float(np.mean(shared_collect))
                   if shared_collect else np.nan)
    return {
        "peak_r": peak_r,
        "peak_lag_plurality": int(peak_lag_plurality)
            if peak_lag_plurality is not np.nan else np.nan,
        "fold_lags": fold_lags,
        "fold_rs": fold_rs,
        "n_groups_used": int(n_groups),
        "mean_shared_locs_at_peak": mean_shared,
        "consistency_curve_full": curve_all.tolist(),
    }


def cv_fixed_lag(neurons, locations, grid_group_idx,
                 fixed_lags_deg,
                 lags_deg=LAGS_DEG,
                 min_dwell=MIN_DWELL_BINS,
                 min_shared_locs=MIN_SHARED_LOCS,
                 weighted=WEIGHTED_CORRELATION,
                 n_loc=N_LOC):
    """Leave-one-grid-group-out CV at PRE-SPECIFIED lag(s). No peak
    selection on the training data — the lag is your a-priori prediction
    (e.g. 30/60° for ACC, 0° for HC).

    Also returns the cross-validated r at every lag in `lags_deg` so you
    can show per-cell tuning curves alongside.

    Returns
    -------
    dict with
        fixed_lags_deg          : list[int] (the a-priori lag(s))
        fixed_lag_per_lag_r     : list[float] — mean across folds at each
                                   FIXED lag (length len(fixed_lags_deg))
        fixed_lag_r_mean        : float — mean over (fixed lag × fold)
        per_lag_r_all_lags      : list[float] (length len(lags_deg)) —
                                   per-fold mean for EVERY analysis lag
        n_groups_used           : int
    """
    fixed_lags_deg = [int(l) for l in fixed_lags_deg]
    fr_all, dwell_all, _, groups = consistency_per_lag(
        neurons, locations, grid_group_idx, lags_deg,
        min_dwell=min_dwell, min_shared_locs=min_shared_locs,
        weighted=weighted, n_loc=n_loc,
    )
    n_groups = len(groups)
    if n_groups < 2:
        return {
            "fixed_lags_deg":       fixed_lags_deg,
            "fixed_lag_per_lag_r":  [np.nan] * len(fixed_lags_deg),
            "fixed_lag_r_mean":     np.nan,
            "per_lag_r_all_lags":   [np.nan] * len(lags_deg),
            "n_groups_used":        int(n_groups),
        }

    lags_arr = np.asarray(list(lags_deg), dtype=int)

    # CV at every lag in the analysis grid.
    per_lag_fold_rs = [[] for _ in lags_arr]
    for held_i in range(n_groups):
        train_mask = np.ones(n_groups, dtype=bool); train_mask[held_i] = False
        for li in range(len(lags_arr)):
            mean_r, _ = _validate_at_lag(
                fr_all[li], dwell_all[li], held_i, train_mask,
                min_shared_locs, weighted,
            )
            per_lag_fold_rs[li].append(mean_r)
    per_lag_r_all_lags = [float(np.nanmean(rs)) if rs else np.nan
                          for rs in per_lag_fold_rs]

    # Pull out the fixed-lag entries.
    fixed_lag_per_lag_r = []
    for fl in fixed_lags_deg:
        idx = int(np.where(lags_arr == fl)[0][0]) if fl in lags_arr else None
        fixed_lag_per_lag_r.append(per_lag_r_all_lags[idx]
                                   if idx is not None else np.nan)
    fixed_lag_r_mean = (float(np.nanmean(fixed_lag_per_lag_r))
                        if any(np.isfinite(v) for v in fixed_lag_per_lag_r)
                        else np.nan)
    return {
        "fixed_lags_deg":      fixed_lags_deg,
        "fixed_lag_per_lag_r": fixed_lag_per_lag_r,
        "fixed_lag_r_mean":    fixed_lag_r_mean,
        "per_lag_r_all_lags":  per_lag_r_all_lags,
        "n_groups_used":       int(n_groups),
    }


# ====================================================================
# Permutation null — circular shifts of the location series
# ====================================================================
def _circular_shift_locations(locations, rng):
    """Independent per-rep circular shift of the location series.
    Preserves dwell distribution + grid assignment; only breaks the
    location-time alignment to the firing series."""
    n_reps, n_bins = locations.shape
    ks = rng.integers(1, n_bins, size=n_reps)
    out = np.empty_like(locations)
    for r in range(n_reps):
        out[r] = np.roll(locations[r], -int(ks[r]))
    return out


# ====================================================================
# Vectorised perm core
#
# The old `cv_with_perms` recomputed `consistency_per_lag` twice per
# permutation (once via `cv_free_peak`, once via `cv_fixed_lag`) and
# iterated permutations in a Python loop. This was the dominant cost in
# the runner — on a typical cell we'd do ~1000 perms × 2 × ~50 ms ≈ 100 s.
#
# The vectorised path below treats the perm index as a leading axis on a
# single `_build_maps_for_shifts` call, then re-uses the same maps for the
# free-peak (cell-picked) and fixed-lag (a-priori) tests. The observed
# pass is just a shift of all-zeros and lives at index 0.
#
# Numerical equivalence vs the old loop is asserted in
# `_assert_perm_core_matches_legacy`.
# ====================================================================
def _build_maps_for_shifts(neurons, locations, grid_group_idx,
                            lags_deg, shifts, n_loc, min_dwell,
                            batch_size=400):
    """Vectorised rate-map + dwell tensor across (perm × lag × loc × group).

    Parameters
    ----------
    neurons        : (n_reps, n_bins) float — phase-residualised firing rate.
    locations      : (n_reps, n_bins) float — grid location 1..n_loc, NaN ok.
    grid_group_idx : (n_reps,) int — reps with id < 0 are dropped.
    lags_deg       : iterable of int bin offsets (e.g. [0, 30, …, 330]).
    shifts         : (n_total, n_reps) int — per-rep circular shift offsets
                     for each entry along the leading "perm" axis. Index 0
                     is conventionally the observed pass (shifts[0]=0).
    """
    n_reps, n_bins = locations.shape
    n_total = shifts.shape[0]
    n_lags = len(lags_deg)
    grp = np.asarray(grid_group_idx)
    valid_reps = grp >= 0
    groups = np.unique(grp[valid_reps])
    n_groups = len(groups)
    group_to_gi = {int(g): gi for gi, g in enumerate(groups)}
    lags_arr = np.asarray(list(lags_deg), dtype=np.int64)
    t_idx = np.arange(n_bins, dtype=np.int64)

    sums = np.zeros((n_total, n_lags, n_loc, n_groups), dtype=np.float64)
    cnts = np.zeros((n_total, n_lags, n_loc, n_groups), dtype=np.float64)

    for r in np.where(valid_reps)[0]:
        gi = group_to_gi[int(grp[r])]
        fr_r = neurons[r]
        loc_r = locations[r]
        fr_finite = np.isfinite(fr_r)
        for s_lo in range(0, n_total, batch_size):
            s_hi = min(s_lo + batch_size, n_total)
            nb = s_hi - s_lo
            shifts_r = shifts[s_lo:s_hi, r:r+1].astype(np.int64)
            # idx[p, l, t] = (t + lag[l] + shift_r[p]) mod n_bins
            # so that locs_pl[t] == np.roll(loc_r, -(lag+shift))[t]
            # — matches legacy `np.roll(loc, -lag)` + per-rep shift sign.
            idx = (t_idx[None, None, :]
                   + shifts_r[:, :, None]
                   + lags_arr[None, :, None]) % n_bins
            locs_pl = loc_r[idx]                          # (nb, n_lags, n_bins)
            valid = (fr_finite[None, None, :]
                     & np.isfinite(locs_pl)
                     & (locs_pl > 0))
            loc_int = np.where(valid, (locs_pl - 1).astype(np.int64), 0)
            np.clip(loc_int, 0, n_loc - 1, out=loc_int)
            p_idx = np.arange(nb,     dtype=np.int64)[:, None, None]
            l_idx = np.arange(n_lags, dtype=np.int64)[None, :, None]
            flat = p_idx * (n_lags * n_loc) + l_idx * n_loc + loc_int
            flat_v = flat[valid]
            fr_b = np.broadcast_to(fr_r[None, None, :], flat.shape)[valid]
            buf_sum = np.bincount(flat_v, weights=fr_b,
                                   minlength=nb * n_lags * n_loc)
            buf_cnt = np.bincount(flat_v,
                                   minlength=nb * n_lags * n_loc).astype(np.float64)
            sums[s_lo:s_hi, :, :, gi] += buf_sum.reshape(nb, n_lags, n_loc)
            cnts[s_lo:s_hi, :, :, gi] += buf_cnt.reshape(nb, n_lags, n_loc)

    safe = np.maximum(cnts, 1.0)
    maps = np.where(cnts >= min_dwell, sums / safe, np.nan)
    return maps, cnts, groups


def _vec_weighted_pearson(x, y, w, weighted=True, min_shared_locs=MIN_SHARED_LOCS):
    """Weighted (or unweighted) Pearson r along axis -1 with NaN safety.

    All inputs broadcast to the same shape ending in (n_loc,). Returns
    `r` of the leading shape and `n_shared` of the same leading shape.
    Where n_shared < `min_shared_locs` or variance is zero, r is NaN.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if weighted:
        mask &= (w > 0)
        w_eff = np.where(mask, w, 0.0)
    else:
        w_eff = mask.astype(np.float64)
    x_eff = np.where(mask, x, 0.0)
    y_eff = np.where(mask, y, 0.0)
    sw = w_eff.sum(axis=-1)
    safe_sw = np.where(sw > 0, sw, 1.0)
    mu_x = (w_eff * x_eff).sum(axis=-1) / safe_sw
    mu_y = (w_eff * y_eff).sum(axis=-1) / safe_sw
    dx = x_eff - mu_x[..., None]
    dy = y_eff - mu_y[..., None]
    cov   = (w_eff * dx * dy).sum(axis=-1) / safe_sw
    var_x = (w_eff * dx * dx).sum(axis=-1) / safe_sw
    var_y = (w_eff * dy * dy).sum(axis=-1) / safe_sw
    denom = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0))
    r = np.where(denom > 0, cov / np.where(denom > 0, denom, 1.0), np.nan)
    n_shared = mask.sum(axis=-1)
    r = np.where((sw > 0)
                 & (var_x > 0) & (var_y > 0)
                 & (n_shared >= int(min_shared_locs)),
                 r, np.nan)
    return r, n_shared


def _shifts_for_perms(rng, n_reps, n_bins, n_perms):
    """Build a (1 + n_perms, n_reps) int shift table: row 0 = observed
    (zeros), rows 1..n_perms = independent per-rep shifts."""
    shifts = np.zeros((1 + n_perms, n_reps), dtype=np.int64)
    if n_perms > 0:
        shifts[1:] = rng.integers(1, n_bins, size=(n_perms, n_reps))
    return shifts


def cv_with_perms(neurons, locations, grid_group_idx,
                  fixed_lags_deg=None,
                  lags_deg=LAGS_DEG, n_peaks=1,
                  min_dwell=MIN_DWELL_BINS,
                  min_shared_locs=MIN_SHARED_LOCS,
                  weighted=WEIGHTED_CORRELATION,
                  n_loc=N_LOC,
                  n_perms=N_PERMUTATIONS,
                  seed=RANDOM_SEED):
    """Vectorised observed + perm pass for free-peak and fixed-lag CV.

    Returns the same dict shape as the legacy loop (`free_*`, `fixed_*`,
    plus `*_perm_*` lists) — see `_assert_perm_core_matches_legacy` for
    numerical equivalence.
    """
    neurons   = np.asarray(neurons, dtype=float)
    locations = np.asarray(locations, dtype=float)
    n_reps, n_bins = locations.shape
    lags_arr = np.asarray(list(lags_deg), dtype=int)
    n_lags = len(lags_arr)
    fixed_lags_deg = [int(l) for l in fixed_lags_deg] if fixed_lags_deg else []
    rng = np.random.default_rng(seed)
    shifts = _shifts_for_perms(rng, n_reps, n_bins, int(n_perms))

    maps, dwell, groups = _build_maps_for_shifts(
        neurons, locations, grid_group_idx, lags_arr, shifts, n_loc, min_dwell,
    )
    n_total  = maps.shape[0]
    n_groups = maps.shape[-1]

    # ---- Degenerate-cell early exit (matches the legacy NaN return) ----
    if n_groups < 2:
        free_zero = {
            "peak_r": np.nan, "peak_lag_plurality": np.nan,
            "fold_lags": [], "fold_rs": [],
            "n_groups_used": int(n_groups),
            "mean_shared_locs_at_peak": np.nan,
            "consistency_curve_full": [np.nan] * n_lags,
        }
        out = {f"free_{k}": v for k, v in free_zero.items()}
        out["free_perm_peak_rs"] = []
        if fixed_lags_deg:
            fixed_zero = {
                "fixed_lags_deg":       fixed_lags_deg,
                "fixed_lag_per_lag_r":  [np.nan] * len(fixed_lags_deg),
                "fixed_lag_r_mean":     np.nan,
                "per_lag_r_all_lags":   [np.nan] * n_lags,
                "n_groups_used":        int(n_groups),
            }
            for k, v in fixed_zero.items():
                out[f"fixed_{k}"] = v
            out["fixed_perm_r_means"]    = []
            out["fixed_perm_per_lag_r"]  = []
        return out

    # ---- Per-pair weighted Pearson r across all (perm, lag) -----------
    pairs = list(combinations(range(n_groups), 2))
    pair_rs = np.full((len(pairs), n_total, n_lags), np.nan)
    for ki, (i, j) in enumerate(pairs):
        w_ij = dwell[..., i] + dwell[..., j]     # (n_total, n_lags, n_loc)
        r_ij, _ = _vec_weighted_pearson(
            maps[..., i], maps[..., j], w_ij,
            weighted=weighted, min_shared_locs=min_shared_locs,
        )
        pair_rs[ki] = r_ij

    # FULL consistency curve = nanmean across all pairs --------------
    curve_full = np.nanmean(pair_rs, axis=0)        # (n_total, n_lags)

    # ---- Per-fold train / validate consistency curves -----------------
    # For fold h: train_pairs = pairs without h. validate_pairs = pairs
    # that include h (the other endpoint is then a "train group").
    train_curves    = np.full((n_total, n_groups, n_lags), np.nan)
    validate_curves = np.full((n_total, n_groups, n_lags), np.nan)
    pair_arr = np.asarray(pairs)  # (n_pairs, 2)
    for h in range(n_groups):
        in_pair = (pair_arr[:, 0] == h) | (pair_arr[:, 1] == h)
        train_curves[:, h, :]    = np.nanmean(pair_rs[~in_pair], axis=0)
        validate_curves[:, h, :] = np.nanmean(pair_rs[ in_pair], axis=0)

    # ---- Free-peak per fold: pick best train lag → validate r ---------
    # NaN-safe argmax (NaN treated as -inf).
    train_masked = np.where(np.isnan(train_curves), -np.inf, train_curves)
    chosen_lag_idx = train_masked.argmax(axis=-1)    # (n_total, n_groups)

    p_grid = np.arange(n_total)[:, None]
    f_grid = np.arange(n_groups)[None, :]
    val_at_chosen = validate_curves[p_grid, f_grid, chosen_lag_idx]  # (n_total, n_groups)
    peak_r = np.nanmean(val_at_chosen, axis=1)                       # (n_total,)

    # ---- Fixed-lag per-perm: mean of validate_curves over folds ------
    per_lag_r_all_lags = np.nanmean(validate_curves, axis=1)         # (n_total, n_lags)
    fixed_lag_idx = [int(np.where(lags_arr == fl)[0][0])
                     for fl in fixed_lags_deg
                     if fl in lags_arr]
    if fixed_lag_idx:
        fixed_per_lag = per_lag_r_all_lags[:, fixed_lag_idx]         # (n_total, n_fixed)
        fixed_lag_r_mean = np.nanmean(fixed_per_lag, axis=1)         # (n_total,)
    else:
        fixed_per_lag = np.empty((n_total, 0))
        fixed_lag_r_mean = np.full(n_total, np.nan)

    # ---- Build the observed-pass extras the legacy API returned -------
    # `fold_lags` (per fold the chosen lag in deg) and `fold_rs` (per fold
    # the validate r at the chosen lag) come from index 0 (= observed).
    obs_lag_idx = chosen_lag_idx[0]            # (n_groups,)
    obs_fold_lags = [[int(lags_arr[obs_lag_idx[h]])] for h in range(n_groups)]
    obs_fold_rs   = [[float(val_at_chosen[0, h])]   for h in range(n_groups)]
    flat = [s for fs in obs_fold_lags for s in fs]
    obs_peak_lag_plurality = Counter(flat).most_common(1)[0][0] if flat else np.nan

    out = {
        "free_peak_r":                   float(peak_r[0]),
        "free_peak_lag_plurality":       int(obs_peak_lag_plurality)
                                            if obs_peak_lag_plurality is not np.nan else np.nan,
        "free_fold_lags":                obs_fold_lags,
        "free_fold_rs":                  obs_fold_rs,
        "free_n_groups_used":            int(n_groups),
        "free_mean_shared_locs_at_peak": np.nan,    # legacy carry-over; not used downstream
        "free_consistency_curve_full":   curve_full[0].tolist(),
        "free_perm_peak_rs":             [float(v) for v in peak_r[1:]],
    }
    if fixed_lags_deg:
        out.update({
            "fixed_fixed_lags_deg":       fixed_lags_deg,
            "fixed_fixed_lag_per_lag_r":  fixed_per_lag[0].tolist()
                                              if fixed_per_lag.size else [],
            "fixed_fixed_lag_r_mean":     float(fixed_lag_r_mean[0]),
            "fixed_per_lag_r_all_lags":   per_lag_r_all_lags[0].tolist(),
            "fixed_n_groups_used":        int(n_groups),
            "fixed_perm_r_means":         [float(v) for v in fixed_lag_r_mean[1:]],
            "fixed_perm_per_lag_r":       [row.tolist() for row in per_lag_r_all_lags[1:]],
        })
    return out


# ====================================================================
# Optional numerical-equivalence check (called only from tests / scripts)
# ====================================================================
def _assert_perm_core_matches_legacy(neurons, locations, grid_group_idx,
                                       fixed_lags_deg=(0, 30),
                                       lags_deg=LAGS_DEG,
                                       n_perms=10, seed=123, rtol=1e-9, atol=1e-9):
    """Run BOTH the new vectorised `cv_with_perms` and a slow per-perm
    loop using the legacy `cv_free_peak` + `cv_fixed_lag`, assert near-
    equality of observed + perm outputs."""
    new = cv_with_perms(
        neurons, locations, grid_group_idx,
        fixed_lags_deg=list(fixed_lags_deg), lags_deg=lags_deg,
        n_perms=n_perms, seed=seed,
    )
    rng = np.random.default_rng(seed)
    shifts = _shifts_for_perms(rng, locations.shape[0], locations.shape[1], n_perms)
    # observed via legacy
    fo = cv_free_peak(neurons, locations, grid_group_idx, lags_deg=lags_deg)
    fx = cv_fixed_lag(neurons, locations, grid_group_idx,
                       fixed_lags_deg=list(fixed_lags_deg), lags_deg=lags_deg)
    assert np.allclose(fo["peak_r"],            new["free_peak_r"],            rtol=rtol, atol=atol, equal_nan=True)
    assert np.allclose(fx["fixed_lag_r_mean"],  new["fixed_fixed_lag_r_mean"], rtol=rtol, atol=atol, equal_nan=True)
    assert np.allclose(fx["per_lag_r_all_lags"], new["fixed_per_lag_r_all_lags"],
                       rtol=rtol, atol=atol, equal_nan=True)
    # perms
    for k in range(1, n_perms + 1):
        locs_k = np.empty_like(locations)
        for r in range(locations.shape[0]):
            locs_k[r] = np.roll(locations[r], -int(shifts[k, r]))
        fo_k = cv_free_peak(neurons, locs_k, grid_group_idx, lags_deg=lags_deg)
        fx_k = cv_fixed_lag(neurons, locs_k, grid_group_idx,
                             fixed_lags_deg=list(fixed_lags_deg), lags_deg=lags_deg)
        assert np.allclose(fo_k["peak_r"],            new["free_perm_peak_rs"][k - 1],
                           rtol=rtol, atol=atol, equal_nan=True)
        assert np.allclose(fx_k["fixed_lag_r_mean"],  new["fixed_perm_r_means"][k - 1],
                           rtol=rtol, atol=atol, equal_nan=True)
    return True


# ====================================================================
# Phase residualisation — applied to the RAW per-bin firing rate BEFORE
# any rate map is computed.
#
# The spatial-peaks pipeline only uses 'cosine'. The other basis options
# (`cosine_2h`, `categorical`) are kept so the RSA + encoding scripts
# that import `_residualise_phase` with their own `basis=` choice keep
# working unchanged.
# ====================================================================
def _residualise_phase(neurons_arr, basis='cosine', state_len=STATE_LEN):
    """Per-cell linear regression of firing rate against a within-state
    phase basis; subtract only the phase component (β₀ kept) so the cell's
    mean firing rate is preserved.

    Parameters
    ----------
    neurons_arr : (n_reps, n_bins) raw firing rates.
    basis : {'cosine', 'cosine_2h', 'categorical'}
        'cosine'      — single harmonic [sin(2πφ), cos(2πφ)]   (φ = bin/state_len)
        'cosine_2h'   — adds the 2nd harmonic [sin(4πφ), cos(4πφ)]
        'categorical' — [I(early), I(middle)] with late as baseline
    """
    n_reps, n_bins = neurons_arr.shape
    phase_idx = (np.arange(n_bins) % state_len).astype(float)
    phi = phase_idx / state_len
    if basis == 'cosine':
        X_phase = np.column_stack([
            np.sin(2 * np.pi * phi),
            np.cos(2 * np.pi * phi),
        ])
    elif basis == 'cosine_2h':
        X_phase = np.column_stack([
            np.sin(2 * np.pi * phi),
            np.cos(2 * np.pi * phi),
            np.sin(4 * np.pi * phi),
            np.cos(4 * np.pi * phi),
        ])
    elif basis == 'categorical':
        early = (phase_idx <  state_len / 3).astype(float)
        mid   = ((phase_idx >= state_len / 3)
                  & (phase_idx < 2 * state_len / 3)).astype(float)
        X_phase = np.column_stack([early, mid])
    else:
        raise ValueError(f"unknown phase basis {basis!r}")

    X_phase_tiled = np.tile(X_phase, (n_reps, 1))
    y_flat = neurons_arr.reshape(-1).astype(float)
    mask = np.isfinite(y_flat)
    if mask.sum() < X_phase_tiled.shape[1] + 5:
        return neurons_arr.copy()
    X_full = np.column_stack([np.ones(X_phase_tiled.shape[0]), X_phase_tiled])
    beta, *_ = np.linalg.lstsq(X_full[mask], y_flat[mask], rcond=None)
    phase_component = X_phase_tiled @ beta[1:]
    y_clean = y_flat - phase_component
    return y_clean.reshape(n_reps, n_bins)


# Public alias used by the spatial-peaks runner. New callers should prefer
# this name; legacy callers (`_residualise_phase`) keep working unchanged.
def phase_residualise(neurons_arr, basis='cosine', state_len=STATE_LEN):
    """Thin wrapper around `_residualise_phase`. The spatial-peaks pipeline
    only uses `basis='cosine'`; other bases exist for backward compat."""
    return _residualise_phase(neurons_arr, basis=basis, state_len=state_len)


# ====================================================================
# Cell-data assembly
# ====================================================================
def prepare_cell_data(subject_data, neuron_id, neurons_df, locations_df,
                      beh, *,
                      grid_cols=("loc_A", "loc_B", "loc_C", "loc_D"),
                      coverage_mode=COVERAGE_MODE,
                      sparsity_filter=SPARSITY_FILTER,
                      phase_residualise_basis=PHASE_BASIS):
    """Assemble per-cell ndarrays for the spatial-peak compute.

    Steps (in order):
      1. Build `grid_group_idx` per repeat (which configurations belong
         to the same group).
      2. PHASE-RESIDUALISE the raw firing rate BEFORE any rate map is
         computed (cosine basis — the only option).
      3. Optional gridwise QC: bad blocks → group id −1 (ignored).
      4. Optional `paired` coverage mode: pair grids to maximise spatial
         coverage. Each pair is treated as one grid group downstream.

    Parameters
    ----------
    subject_data : per-subject dict from `load_norm_data` + `filter_data`
    neuron_id    : key into `subject_data['normalised_neurons']`

    Returns
    -------
    dict (picklable) or None if the cell can't be analysed.
    Keys:
        neuron_id, neurons, locations, grid_group_idx,
        paired_config_groups   — list[list[int]]: which configs were
                                  paired in each "grid group" (so the
                                  plotting can show the paired configs).
    """
    grid_cols = list(grid_cols)
    beh = beh.copy().reset_index(drop=True)

    # ---- 1) original per-repeat grid id (one per unique 4-tuple config)
    uniq, _, grid_group_idx, _ = np.unique(
        beh[grid_cols].to_numpy(), axis=0,
        return_index=True, return_inverse=True, return_counts=True,
    )
    if len(uniq) < 2:
        return None
    beh["grid_group_idx"] = grid_group_idx
    # `extract_consistent_grids` and `pair_grids_to_increase_spatial_coverage`
    # in `helpers_human_cells` expect the column under the older name.
    beh["idx_same_grids"] = grid_group_idx

    neurons_arr   = neurons_df.to_numpy(dtype=float)
    locations_arr = locations_df.to_numpy(dtype=float)

    # ---- 2) phase residualisation on the RAW firing rate (cosine basis)
    if phase_residualise_basis is not None:
        if phase_residualise_basis != 'cosine':
            raise ValueError(
                f"phase_residualise_basis must be 'cosine' or None; got "
                f"{phase_residualise_basis!r}")
        neurons_arr = phase_residualise(neurons_arr)

    # ---- 3) gridwise-QC sparsity filter (mark bad blocks as -1)
    if sparsity_filter == "gridwise_qc":
        beh = hh.extract_consistent_grids(neurons_arr, neuron_id, beh)
        consistent = beh[f"consistent_FR_{neuron_id}"].to_numpy()
        grid_group_idx = grid_group_idx.copy()
        grid_group_idx[~consistent] = -1

    # ---- 4) paired-grid coverage mode
    paired_config_groups = []
    if coverage_mode == "paired":
        beh = hh.pair_grids_to_increase_spatial_coverage(
            locations_arr, beh, neuron_id,
        )
        paired_col = beh[f"paired_grid_idx_{neuron_id}"].to_numpy()
        grid_group_idx = np.array(
            [int(v) if (v is not False and pd.notna(v)) else -1
             for v in paired_col],
            dtype=int,
        )
        # For the plot: which original config-tuples got paired together
        # into each group. `paired_grid_idx_*` is the same int across all
        # repeats in that pair; pull the unique original-grid ids per
        # pair.
        valid = grid_group_idx >= 0
        for g in np.unique(grid_group_idx[valid]):
            in_pair = (grid_group_idx == g) & valid
            cfgs = sorted(set(int(c) for c in beh.loc[in_pair, "grid_group_idx"]))
            paired_config_groups.append(cfgs)
    elif coverage_mode == "per_grid":
        grid_group_idx = grid_group_idx.astype(int)
        for g in np.unique(grid_group_idx[grid_group_idx >= 0]):
            paired_config_groups.append([int(g)])
    else:
        raise ValueError(f"Unknown coverage_mode {coverage_mode!r}")

    if np.unique(grid_group_idx[grid_group_idx >= 0]).size < 2:
        return None

    return {
        "neuron_id":            neuron_id,
        "neurons":              neurons_arr,
        "locations":            locations_arr,
        "grid_group_idx":       grid_group_idx,
        "paired_config_groups": paired_config_groups,
    }


# ====================================================================
# Per-cell driver — picklable for joblib
# ====================================================================
def analyse_one_cell(payload, cfg):
    """One cell's compute. Picklable for joblib.

    `cfg` keys (all optional; module defaults at the top of this file):
        roi                       — used to choose the a-priori fixed lag
        roi_predicted_lags_deg    — dict {roi: tuple[int]}; if missing or
                                    roi absent, fixed-lag analysis is
                                    skipped.
        lags_deg, n_peaks, min_dwell_bins, min_shared_locs,
        weighted_correlation, n_loc, n_permutations, random_seed
    """
    if payload is None:
        return None
    roi               = cfg.get("roi")
    predicted_lags    = (cfg.get("roi_predicted_lags_deg",
                                  ROI_PREDICTED_LAGS_DEG) or {})
    fixed_lags_deg    = list(predicted_lags.get(roi, ())) if roi else []
    n_perms           = (int(cfg.get("n_permutations", N_PERMUTATIONS))
                          if cfg.get("run_permutations", True) else 0)
    base_seed         = int(cfg.get("random_seed", RANDOM_SEED))
    cell_seed         = base_seed + (abs(hash(payload["neuron_id"])) % (2**31))
    out = cv_with_perms(
        neurons=payload["neurons"],
        locations=payload["locations"],
        grid_group_idx=payload["grid_group_idx"],
        fixed_lags_deg=fixed_lags_deg,
        lags_deg=cfg.get("lags_deg", LAGS_DEG),
        n_peaks=int(cfg.get("n_peaks", 1)),
        min_dwell=int(cfg.get("min_dwell_bins", MIN_DWELL_BINS)),
        min_shared_locs=int(cfg.get("min_shared_locs", MIN_SHARED_LOCS)),
        weighted=bool(cfg.get("weighted_correlation", WEIGHTED_CORRELATION)),
        n_loc=int(cfg.get("n_loc", N_LOC)),
        n_perms=n_perms,
        seed=cell_seed,
    )
    out["neuron_id"]            = payload["neuron_id"]
    out["roi"]                  = roi
    out["fixed_lags_deg_used"]  = fixed_lags_deg
    out["paired_config_groups"] = payload.get("paired_config_groups", [])
    return out


# ====================================================================
# Significance-testing toolkit
# ====================================================================
def bh_fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values. NaNs stay NaN."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any(): return out
    p_good = p[ok]; n = p_good.size
    order = np.argsort(p_good)
    ranked = p_good[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qok = np.empty_like(ranked); qok[order] = np.clip(ranked, 0, 1)
    out[ok] = qok
    return out


def ttest_mean_r_vs_zero(rs, alternative='greater'):
    """One-sample t-test: H_1 is mean(r) > 0.

    Returns dict(t, p, mean, n) with NaN-safe handling.
    """
    rs = np.asarray(rs, dtype=float)
    rs = rs[np.isfinite(rs)]
    if rs.size < 2:
        return {"t": np.nan, "p": np.nan,
                "mean": float(rs.mean()) if rs.size else np.nan,
                "n": int(rs.size)}
    try:
        res = stats.ttest_1samp(rs, 0.0, alternative=alternative)
        t, p = float(res.statistic), float(res.pvalue)
    except TypeError:  # older scipy fallback
        t, p_two = stats.ttest_1samp(rs, 0.0)
        if alternative == 'greater':
            p = float(p_two / 2 if t > 0 else 1.0 - p_two / 2)
        else:
            p = float(p_two)
        t = float(t)
    return {"t": t, "p": p, "mean": float(np.mean(rs)), "n": int(rs.size)}


def within_cell_lag_vs_others(per_cell_lag_curves, lags_deg, target_lags_deg,
                              alternative='greater'):
    """Within each cell, compare r at TARGET lag(s) to the mean r at the
    OTHER lags. Returns (t, p) of a paired-sample one-sided t-test of
    (r_target − r_others) across cells.

    Parameters
    ----------
    per_cell_lag_curves : (n_cells, n_lags) — per-cell consistency curves
        (the `per_lag_r_all_lags` from `cv_fixed_lag`).
    target_lags_deg : iterable of int — lag(s) treated as the predicted set.
    """
    curves = np.asarray(per_cell_lag_curves, dtype=float)
    lags = np.asarray(lags_deg, dtype=int)
    target_mask = np.isin(lags, list(target_lags_deg))
    other_mask  = ~target_mask
    if target_mask.sum() == 0 or other_mask.sum() == 0:
        return {"t": np.nan, "p": np.nan, "n": 0,
                "mean_target": np.nan, "mean_others": np.nan}
    r_target  = np.nanmean(curves[:, target_mask], axis=1)
    r_others  = np.nanmean(curves[:, other_mask],  axis=1)
    diff = r_target - r_others
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return {"t": np.nan, "p": np.nan, "n": int(diff.size),
                "mean_target": float(np.nanmean(r_target)),
                "mean_others": float(np.nanmean(r_others))}
    try:
        res = stats.ttest_1samp(diff, 0.0, alternative=alternative)
        t, p = float(res.statistic), float(res.pvalue)
    except TypeError:
        t, p_two = stats.ttest_1samp(diff, 0.0)
        p = float(p_two / 2 if t > 0 else 1.0 - p_two / 2)
        t = float(t)
    return {"t": t, "p": p, "n": int(diff.size),
            "mean_target":   float(np.nanmean(r_target)),
            "mean_others":   float(np.nanmean(r_others)),
            "mean_diff":     float(np.mean(diff))}


def binomial_sig_fraction(perm_p_values, alpha=ALPHA, alternative='greater'):
    """One-sided binomial test: of n cells, is the count with perm-p < alpha
    higher than the alpha chance rate?

    Parameters
    ----------
    perm_p_values : iterable of float — one perm-p per cell.
    alpha : float — significance threshold.
    """
    p = np.asarray(perm_p_values, dtype=float)
    p = p[np.isfinite(p)]
    n = int(p.size); k = int((p < alpha).sum())
    if n == 0:
        return {"n": 0, "k": 0, "frac": np.nan, "p_binom": np.nan}
    try:
        res = stats.binomtest(k, n, p=alpha, alternative=alternative)
        p_binom = float(res.pvalue)
    except AttributeError:  # scipy < 1.7
        p_binom = float(stats.binom_test(k, n, p=alpha,
                                          alternative=alternative))
    return {"n": n, "k": k, "frac": k / n, "p_binom": p_binom}
