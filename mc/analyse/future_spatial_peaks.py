"""Vectorised future-spatial-peaks compute.

For each cell:
  * compute the mean firing rate per location (1..9) on the location series
    rolled by each shift in `shifts_deg`,
  * measure pairwise consistency of these rate maps across grids,
  * cross-validate over grids (LOO): pick the top-`n_peaks` shifts on the
    training grids, validate at exactly those shifts on the held-out grid,
  * return the mean held-out r as `peak_r`.

Phase-agnostic by construction (rates averaged across all bins where the
shifted location equals a given grid square).

This module is pure compute. No IO, no plotting. `run_one_cell` is
picklable for joblib.
"""

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

import mc.analyse.helpers_human_cells as hh


# ── core rate-map ─────────────────────────────────────────────────────

def rate_map(fr_flat, loc_flat, n_loc=9, min_dwell=25):
    """Vectorised per-location mean firing rate.

    Parameters
    ----------
    fr_flat : 1D array, firing rates (same length as loc_flat).
    loc_flat : 1D int array, location labels in 1..n_loc. Values <= 0 or
        non-finite are ignored.
    n_loc : int, number of locations (9 on a 3x3 grid).
    min_dwell : int, minimum sample count per location to accept the
        estimate. Locations with fewer samples become NaN.

    Returns
    -------
    rm    : (n_loc,) mean rate per location (NaN where dwell < min_dwell).
    dwell : (n_loc,) bin count per location.
    """
    fr_flat = np.asarray(fr_flat, dtype=float)
    loc_flat = np.asarray(loc_flat, dtype=float)
    mask = np.isfinite(fr_flat) & np.isfinite(loc_flat) & (loc_flat > 0)
    fr = fr_flat[mask]
    lo = loc_flat[mask].astype(int) - 1
    keep = (lo >= 0) & (lo < n_loc)
    fr = fr[keep]; lo = lo[keep]
    sums = np.bincount(lo, weights=fr, minlength=n_loc)
    cnts = np.bincount(lo, minlength=n_loc).astype(float)
    safe = np.maximum(cnts, 1.0)
    rm = np.where(cnts >= min_dwell, sums / safe, np.nan)
    return rm, cnts


# ── shift curve (all shifts × all grids for one cell) ─────────────────

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


def pairwise_consistency(maps, dwell, min_shared_locs=5, weighted=False,
                         return_n_shared=False):
    """Mean over grid-pairs of (weighted) Pearson r between rate maps.

    Parameters
    ----------
    maps   : (n_loc, n_grids) rate maps (NaN-allowed).
    dwell  : (n_loc, n_grids) dwell counts (>=0).
    min_shared_locs : pairs with fewer co-finite locations are skipped.
    weighted : if True, weight by dwell[:, i] + dwell[:, j].
    return_n_shared : also return mean n shared locs across kept pairs.

    Returns
    -------
    mean_r           : float (NaN if no valid pair).
    mean_n_shared    : float (only if return_n_shared).
    """
    n_loc, n_grids = maps.shape
    rs, n_shared = [], []
    for i, j in combinations(range(n_grids), 2):
        m = np.isfinite(maps[:, i]) & np.isfinite(maps[:, j])
        if m.sum() < min_shared_locs:
            continue
        if weighted:
            w_ij = dwell[:, i] + dwell[:, j]
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


def shift_curve(neurons, locs, idx_same_grids, shifts,
                min_dwell=25, min_shared_locs=5, weighted=False,
                n_loc=9):
    """Per-shift cross-grid consistency curve for one cell.

    Parameters
    ----------
    neurons : (n_reps, n_bins) firing rates.
    locs    : (n_reps, n_bins) integer locations (1..n_loc, NaN allowed).
    idx_same_grids : (n_reps,) grid id per repeat. Repeats with id < 0
                     are dropped.
    shifts : iterable of integer bin offsets.

    Returns
    -------
    fr_maps    : (n_shifts, n_loc, n_grids) rate maps.
    dwell_maps : (n_shifts, n_loc, n_grids) dwell counts.
    mean_corr  : (n_shifts,) pairwise-consistency value per shift.
    unique_grids : (n_grids,) sorted grid ids used.
    """
    neurons = np.asarray(neurons, dtype=float)
    locs = np.asarray(locs, dtype=float)
    idx = np.asarray(idx_same_grids)
    valid_reps = idx >= 0
    unique_grids = np.unique(idx[valid_reps])
    n_grids = len(unique_grids)
    n_shifts = len(shifts)

    fr_maps    = np.full((n_shifts, n_loc, n_grids), np.nan)
    dwell_maps = np.zeros((n_shifts, n_loc, n_grids))
    mean_corr  = np.full(n_shifts, np.nan)

    # pre-roll once per shift (cheap; n_reps × n_bins, copied n_shifts times)
    for s_i, sh in enumerate(shifts):
        locs_sh = np.roll(locs, shift=-int(sh), axis=1)
        for g_i, g in enumerate(unique_grids):
            mask = (idx == g)
            fr = neurons[mask].ravel()
            lo = locs_sh[mask].ravel()
            rm, dw = rate_map(fr, lo, n_loc=n_loc, min_dwell=min_dwell)
            fr_maps[s_i, :, g_i] = rm
            dwell_maps[s_i, :, g_i] = dw
        if n_grids >= 2:
            mean_corr[s_i] = pairwise_consistency(
                fr_maps[s_i], dwell_maps[s_i],
                min_shared_locs=min_shared_locs, weighted=weighted,
            )
    return fr_maps, dwell_maps, mean_corr, unique_grids


# ── cross-validated peak ──────────────────────────────────────────────

def _topk_shifts(curve, shifts, k):
    """Return the top-k shift values from `curve`, NaNs sorted last."""
    curve = np.asarray(curve, dtype=float)
    shifts = np.asarray(shifts)
    order = np.argsort(np.where(np.isnan(curve), -np.inf, curve))[::-1]
    sel = [int(shifts[i]) for i in order
           if np.isfinite(curve[i])][:k]
    while len(sel) < k:
        sel.append(int(shifts[order[0]]))   # fallback duplicate
    return sel


def cv_peak_consistency(neurons, locs, idx_same_grids, shifts,
                        n_peaks=1,
                        min_dwell=25, min_shared_locs=5,
                        weighted=False, n_loc=9):
    """Leave-one-grid-out cross-validated peak consistency.

    For each fold:
        * train grids = all but the held-out one,
        * compute shift_curve on train,
        * pick top-`n_peaks` shifts by train mean correlation,
        * compute the test grid rate map at each of those shifts,
        * for each such shift, evaluate the mean correlation between the
          test rate map and each train grid's rate map (skipping pairs
          with fewer than `min_shared_locs` shared locations),
        * fold score = mean over the n_peaks shifts.

    The cell-level metric `peak_r` is the mean of fold scores.

    Returns
    -------
    dict with keys:
        peak_r, peak_shift_plurality, fold_shifts (list of lists),
        fold_rs (list of lists), n_grids_used, mean_shared_locs_at_peak.
    """
    fr_all, dwell_all, curve_all, grids = shift_curve(
        neurons, locs, idx_same_grids, shifts,
        min_dwell=min_dwell, min_shared_locs=min_shared_locs,
        weighted=weighted, n_loc=n_loc,
    )
    n_grids = len(grids)
    if n_grids < 2:
        return {
            "peak_r": np.nan,
            "peak_shift_plurality": np.nan,
            "fold_shifts": [],
            "fold_rs": [],
            "n_grids_used": int(n_grids),
            "mean_shared_locs_at_peak": np.nan,
            "shift_curve_full": curve_all.tolist(),
        }

    shifts_arr = np.asarray(list(shifts), dtype=int)
    fold_shifts, fold_rs, shared_collect = [], [], []

    for held_i in range(n_grids):
        train_mask = np.ones(n_grids, dtype=bool); train_mask[held_i] = False
        train_curve = np.full(len(shifts_arr), np.nan)
        for s_i in range(len(shifts_arr)):
            train_curve[s_i] = pairwise_consistency(
                fr_all[s_i][:, train_mask], dwell_all[s_i][:, train_mask],
                min_shared_locs=min_shared_locs, weighted=weighted,
            )

        peaks = _topk_shifts(train_curve, shifts_arr, n_peaks)
        peak_idxs = [int(np.where(shifts_arr == p)[0][0]) for p in peaks]

        # validate: per selected shift, mean over train grids of
        # corr(train_grid_rm, test_grid_rm) at that shift.
        rs_for_fold = []
        for s_i in peak_idxs:
            test_rm = fr_all[s_i][:, held_i]
            test_dw = dwell_all[s_i][:, held_i]
            train_rms = fr_all[s_i][:, train_mask]
            train_dws = dwell_all[s_i][:, train_mask]
            # build a (n_loc, n_train+1) block to reuse pairwise_consistency
            # but we only want test-vs-train r's, not train-vs-train.
            test_rs, test_shared = [], []
            for k in range(train_rms.shape[1]):
                m = np.isfinite(test_rm) & np.isfinite(train_rms[:, k])
                if m.sum() < min_shared_locs:
                    continue
                if weighted:
                    w_v = test_dw + train_dws[:, k]
                    rk = _weighted_pearson(test_rm[m], train_rms[m, k], w_v[m])
                else:
                    v1 = test_rm[m]; v2 = train_rms[m, k]
                    if np.std(v1) <= 0 or np.std(v2) <= 0:
                        rk = np.nan
                    else:
                        rk = float(np.corrcoef(v1, v2)[0, 1])
                if np.isfinite(rk):
                    test_rs.append(rk); test_shared.append(int(m.sum()))
            if test_rs:
                rs_for_fold.append(float(np.mean(test_rs)))
                shared_collect.extend(test_shared)
            else:
                rs_for_fold.append(np.nan)

        fold_shifts.append(peaks)
        fold_rs.append(rs_for_fold)

    peak_r = float(np.nanmean([np.nanmean(r) for r in fold_rs])) \
        if fold_rs else np.nan
    flat = [s for fs in fold_shifts for s in fs]
    peak_shift_plurality = Counter(flat).most_common(1)[0][0] if flat else np.nan
    mean_shared = (float(np.mean(shared_collect))
                   if shared_collect else np.nan)
    return {
        "peak_r": peak_r,
        "peak_shift_plurality": int(peak_shift_plurality)
            if peak_shift_plurality is not np.nan else np.nan,
        "fold_shifts": fold_shifts,
        "fold_rs": fold_rs,
        "n_grids_used": int(n_grids),
        "mean_shared_locs_at_peak": mean_shared,
        "shift_curve_full": curve_all.tolist(),
    }


# ── per-cell payload + driver ─────────────────────────────────────────

def _residualise_run(neurons_arr, grid_no):
    """Per-cell categorical subtraction of per-run (grid_no) mean firing rate.

    Each consecutive visit of the same 4-tuple reward config typically gets a
    DIFFERENT grid_no (one per block). This removes "the cell fired more on
    run-1 of config X than on run-2 of config X" while preserving the
    cell's overall mean firing rate AND any within-run spatial structure.

    Unlike _residualise_time (linear drift across rep_overall) this is
    categorical per visit and catches non-monotonic drift between visits.
    """
    out = neurons_arr.copy()
    g = np.asarray(grid_no)
    valid = np.isfinite(g) if np.issubdtype(g.dtype, np.floating) else np.ones_like(g, dtype=bool)
    grand = np.nanmean(out)
    for u in np.unique(g[valid]):
        mask = (g == u)
        mu = np.nanmean(out[mask])
        if np.isfinite(mu):
            out[mask] = out[mask] - mu + grand
    return out


def _residualise_time(neurons_arr, time_index):
    """Per-cell linear regression against a within-session time index
    (e.g. rep_overall). Removes drift / boredom / satiation across the
    task while preserving the cell's mean firing rate.

    time_index : (n_reps,) numeric. Each repeat's position in the session.
    """
    n_reps, n_bins = neurons_arr.shape
    t = np.asarray(time_index, dtype=float)
    if t.size != n_reps:
        return neurons_arr.copy()
    # tile time index across bins
    t_tiled = np.repeat(t, n_bins)
    y_flat = neurons_arr.reshape(-1).astype(float)
    mask = np.isfinite(y_flat) & np.isfinite(t_tiled)
    if mask.sum() < 10:
        return neurons_arr.copy()
    X = np.column_stack([np.ones(t_tiled.size), t_tiled])
    beta, *_ = np.linalg.lstsq(X[mask], y_flat[mask], rcond=None)
    # subtract only the time-slope component (keep intercept)
    y_clean = y_flat - beta[1] * t_tiled
    return y_clean.reshape(n_reps, n_bins)


def _residualise_config(neurons_arr, idx_same_grids):
    """Per-cell linear regression against config-dummy codes; subtract the
    per-config mean firing rate, preserving overall mean.

    For each unique config (>= 0 in idx_same_grids), subtract its own mean
    firing rate, then add back the grand mean so the cell's overall rate
    is unchanged.

    Repeats with idx_same_grids < 0 (excluded by paired/sparsity) are left
    untouched.
    """
    out = neurons_arr.copy()
    valid = idx_same_grids >= 0
    grand = np.nanmean(out[valid])
    for g in np.unique(idx_same_grids[valid]):
        mask = idx_same_grids == g
        cm = np.nanmean(out[mask])
        if np.isfinite(cm):
            out[mask] = out[mask] - cm + grand
    return out


def _physical_to_goal_relative(locs_arr, beh, mode="toroidal_mod3",
                                state_len=90, n_states=4, grid_side=3):
    """Re-map physical 1..9 locations to goal-relative coordinates.

    At each bin t, the *current goal* depends on which state is active:
        state_idx = (t // state_len) % n_states
        current_goal = beh row's loc_A / loc_B / loc_C / loc_D
        for state_idx == 0/1/2/3 respectively.

    mode : 'toroidal_mod3'
            Re-centre the 3x3 grid so current goal sits at coord (0,0)
            and wrap with modular arithmetic. Output is 1..9 (preserves
            n_loc=9 for downstream code). Tests "fires at fixed relative
            position to goal".
        : 'manhattan_distance'
            Output is Manhattan distance to current goal, 0..4.
            Tests "fires at fixed step-distance to goal".

    Parameters
    ----------
    locs_arr : (n_reps, n_bins) physical locations 1..9 (NaN allowed).
    beh      : DataFrame indexed by rep, must contain loc_A,B,C,D ints.

    Returns
    -------
    (n_reps, n_bins) int array of goal-relative locations.
    NaN inputs map to -1 (will be ignored downstream).
    """
    n_reps, n_bins = locs_arr.shape
    goal_cols = ["loc_A", "loc_B", "loc_C", "loc_D"]
    goals = beh[goal_cols].to_numpy(dtype=int)        # (n_reps, n_states)

    # state assignment per bin
    state_per_bin = (np.arange(n_bins) // state_len) % n_states

    # current goal at each (rep, bin)
    goals_per_bin = goals[:, state_per_bin]            # (n_reps, n_bins)

    # subject location coords
    loc_idx = locs_arr.astype(float)
    # mark NaN/invalid
    invalid = ~np.isfinite(loc_idx) | (loc_idx < 1) | (loc_idx > grid_side * grid_side)
    loc_int = np.where(invalid, 1, loc_idx).astype(int)
    sub_col = (loc_int - 1) % grid_side
    sub_row = (loc_int - 1) // grid_side

    goal_col = (goals_per_bin - 1) % grid_side
    goal_row = (goals_per_bin - 1) // grid_side

    if mode == "toroidal_mod3":
        rel_col = (sub_col - goal_col) % grid_side
        rel_row = (sub_row - goal_row) % grid_side
        rel = rel_row * grid_side + rel_col + 1          # 1..9, "1" = at goal
    elif mode == "manhattan_distance":
        rel = np.abs(sub_col - goal_col) + np.abs(sub_row - goal_row)
        rel = rel + 1                                    # 1..(2*(grid_side-1)+1)
    else:
        raise ValueError(f"unknown goal-relative mode {mode!r}")
    rel = rel.astype(float)
    rel[invalid] = -1.0
    return rel


def _residualise_phase(neurons_arr, basis="cosine", state_len=90):
    """Per-cell linear regression of firing rate against a within-state phase
    basis; return residuals + intercept (so mean firing rate is unchanged).

    basis options:
        'cosine'      : single harmonic [sin(2πφ), cos(2πφ)] where φ = bin/state_len
        'cosine_2h'   : adds 2nd harmonic [sin(4πφ), cos(4πφ)]
        'categorical' : [I(early=phase∈[0,30)), I(mid=phase∈[30,60))] (late = baseline)
    The first column is always a constant; β₀ is kept (not subtracted).
    """
    n_reps, n_bins = neurons_arr.shape
    phase_idx = (np.arange(n_bins) % state_len).astype(float)
    phi = phase_idx / state_len
    if basis == "cosine":
        X_phase = np.column_stack([
            np.sin(2 * np.pi * phi),
            np.cos(2 * np.pi * phi),
        ])
    elif basis == "cosine_2h":
        X_phase = np.column_stack([
            np.sin(2 * np.pi * phi),
            np.cos(2 * np.pi * phi),
            np.sin(4 * np.pi * phi),
            np.cos(4 * np.pi * phi),
        ])
    elif basis == "categorical":
        early = (phase_idx <  state_len / 3).astype(float)
        mid   = ((phase_idx >= state_len / 3) & (phase_idx < 2 * state_len / 3)).astype(float)
        X_phase = np.column_stack([early, mid])
    else:
        raise ValueError(f"unknown phase basis {basis!r}")

    # tile phase regressor across reps and fit globally
    X_phase_tiled = np.tile(X_phase, (n_reps, 1))
    y_flat = neurons_arr.reshape(-1).astype(float)
    mask = np.isfinite(y_flat)
    if mask.sum() < X_phase_tiled.shape[1] + 5:
        return neurons_arr.copy()
    # solve y = X_phase β + const   (handle const separately to keep mean unchanged)
    X_full = np.column_stack([np.ones(X_phase_tiled.shape[0]), X_phase_tiled])
    beta, *_ = np.linalg.lstsq(X_full[mask], y_flat[mask], rcond=None)
    # subtract only the phase component (skip the constant β₀)
    phase_component = X_phase_tiled @ beta[1:]
    y_clean = y_flat - phase_component
    return y_clean.reshape(n_reps, n_bins)


def build_cell_payload(sub_data, neuron_id, neurons_df, locs_df, beh,
                       trials_label, coverage_mode, sparsity_filter,
                       grid_cols=("loc_A", "loc_B", "loc_C", "loc_D"),
                       phase_residualise=None,
                       goal_relative_locs=None,
                       config_residualise=False,
                       time_residualise=False,
                       time_col="rep_overall",
                       run_residualise=False,
                       run_col="grid_no"):
    """Assemble the per-cell ndarrays (neurons, locs, idx_same_grids).

    Applies the coverage/sparsity options exactly the way the old
    wrapper did (extract_consistent_grids -> pair_grids_to_increase_…).

    phase_residualise : None or one of 'cosine' / 'cosine_2h' / 'categorical'.
        If set, each cell's firing rate is residualised against a within-state
        phase basis before downstream analysis. Mean firing rate is preserved.

    sub_data : the per-subject dict returned by load_norm_data + filter_data
    neuron_id : key into sub_data['normalised_neurons']
    Returns dict (picklable) or None if the cell can't be analysed.
    """
    grid_cols = list(grid_cols)
    beh = beh.copy().reset_index(drop=True)
    uniq, _, idx_same_grids, _ = np.unique(
        beh[grid_cols].to_numpy(), axis=0,
        return_index=True, return_inverse=True, return_counts=True,
    )
    if len(uniq) < 2:
        return None

    beh["idx_same_grids"] = idx_same_grids
    neurons_arr = neurons_df.to_numpy(dtype=float)
    locs_arr    = locs_df.to_numpy(dtype=float)

    # optional phase residualisation (per cell, before any spatial analysis)
    if phase_residualise:
        neurons_arr = _residualise_phase(neurons_arr, basis=phase_residualise)

    # optional time/drift residualisation (per cell, against rep_overall etc.)
    if time_residualise:
        if time_col in beh.columns:
            neurons_arr = _residualise_time(
                neurons_arr, beh[time_col].to_numpy(dtype=float),
            )
        else:
            print(f"  [warn] time_residualise=True but column "
                  f"{time_col!r} missing for {neuron_id}; skipping")

    # optional per-run-of-config (grid_no) categorical residualisation
    if run_residualise:
        if run_col in beh.columns:
            neurons_arr = _residualise_run(
                neurons_arr, beh[run_col].to_numpy(),
            )
        else:
            print(f"  [warn] run_residualise=True but column "
                  f"{run_col!r} missing for {neuron_id}; skipping")

    # optional goal-relative remapping of the location series
    #   IMPORTANT: applied AFTER pairing/sparsity decisions are made on the
    #   physical locs (pairing uses physical layout for coverage), and we keep
    #   coverage_mode='per_grid' as the recommended setting because pairing
    #   in goal-relative space would have different semantics.
    goal_relative_active = bool(goal_relative_locs)
    goal_rel_mode = goal_relative_locs if isinstance(goal_relative_locs, str) else "toroidal_mod3"

    # optional sparsity filter (mark bad grids as -1)
    if sparsity_filter == "gridwise_qc":
        beh = hh.extract_consistent_grids(neurons_arr, neuron_id, beh)
        consistent = beh[f"consistent_FR_{neuron_id}"].to_numpy()
        idx_same_grids = idx_same_grids.copy()
        idx_same_grids[~consistent] = -1

    # optional grid pairing for spatial coverage
    if coverage_mode == "paired":
        beh = hh.pair_grids_to_increase_spatial_coverage(
            locs_arr, beh, neuron_id,
        )
        paired = beh[f"paired_grid_idx_{neuron_id}"].to_numpy()
        # paired can contain False where a row is excluded; convert to -1
        idx_for_compute = np.array(
            [int(v) if (v is not False and pd.notna(v)) else -1 for v in paired],
            dtype=int,
        )
    elif coverage_mode == "per_grid":
        idx_for_compute = idx_same_grids.astype(int)
    else:
        raise ValueError(f"Unknown coverage_mode {coverage_mode!r}")

    if np.unique(idx_for_compute[idx_for_compute >= 0]).size < 2:
        return None

    # optional config-mean residualisation (after grid pairing decisions)
    if config_residualise:
        neurons_arr = _residualise_config(neurons_arr, idx_for_compute)

    if goal_relative_active:
        locs_out = _physical_to_goal_relative(locs_arr, beh, mode=goal_rel_mode)
    else:
        locs_out = locs_arr

    return {
        "neuron_id": neuron_id,
        "neurons":   neurons_arr,
        "locs":      locs_out,
        "idx_same_grids": idx_for_compute,
        "goal_relative_active": goal_relative_active,
        "goal_relative_mode": goal_rel_mode if goal_relative_active else None,
    }


def cv_at_fixed_lags(neurons, locs, idx_same_grids, fixed_lags,
                     min_dwell=25, min_shared_locs=5, weighted=False,
                     n_loc=9):
    """Leave-one-grid-out cross-validated consistency at PRE-SPECIFIED lags.

    No training-data peak selection. For each fold, compute the test grid's
    rate map at each fixed lag, correlate with each train grid's rate map
    at the same lag (skipping pairs with fewer than min_shared_locs shared
    locations), and average. Returns a per-(lag, fold) matrix and per-lag
    summary stats.

    Parameters
    ----------
    fixed_lags : iterable of int. Lags (in bins) to evaluate.

    Returns
    -------
    dict with keys:
        fixed_lags          : list[int]
        per_lag_r           : list[float] (n_lags), mean across folds at each lag
        per_lag_fold_rs     : list[list[float]] (n_lags × n_folds), raw fold r's
        fixed_lag_r_mean    : float, mean over (lag, fold) for the requested set
        n_grids_used        : int
    """
    fixed_lags = list(fixed_lags)
    fr_all, dwell_all, _curve_all, grids = shift_curve(
        neurons, locs, idx_same_grids, fixed_lags,
        min_dwell=min_dwell, min_shared_locs=min_shared_locs,
        weighted=weighted, n_loc=n_loc,
    )
    n_grids = len(grids)
    if n_grids < 2:
        return {
            "fixed_lags":       fixed_lags,
            "per_lag_r":        [np.nan] * len(fixed_lags),
            "per_lag_fold_rs":  [[] for _ in fixed_lags],
            "fixed_lag_r_mean": np.nan,
            "n_grids_used":     int(n_grids),
        }

    per_lag_fold_rs = [[] for _ in fixed_lags]
    for held_i in range(n_grids):
        train_mask = np.ones(n_grids, dtype=bool); train_mask[held_i] = False
        for li in range(len(fixed_lags)):
            test_rm    = fr_all[li][:, held_i]
            test_dw    = dwell_all[li][:, held_i]
            train_rms  = fr_all[li][:, train_mask]
            train_dws  = dwell_all[li][:, train_mask]
            test_rs = []
            for k in range(train_rms.shape[1]):
                m = np.isfinite(test_rm) & np.isfinite(train_rms[:, k])
                if m.sum() < min_shared_locs:
                    continue
                if weighted:
                    w_v = test_dw + train_dws[:, k]
                    rk = _weighted_pearson(test_rm[m], train_rms[m, k], w_v[m])
                else:
                    v1 = test_rm[m]; v2 = train_rms[m, k]
                    rk = float(np.corrcoef(v1, v2)[0, 1]) \
                        if (np.std(v1) > 0 and np.std(v2) > 0) else np.nan
                if np.isfinite(rk):
                    test_rs.append(rk)
            per_lag_fold_rs[li].append(float(np.mean(test_rs)) if test_rs else np.nan)

    per_lag_r = [float(np.nanmean(rs)) if rs else np.nan
                 for rs in per_lag_fold_rs]
    fixed_lag_r_mean = float(np.nanmean(per_lag_r)) if per_lag_r else np.nan
    return {
        "fixed_lags":       fixed_lags,
        "per_lag_r":        per_lag_r,
        "per_lag_fold_rs":  per_lag_fold_rs,
        "fixed_lag_r_mean": fixed_lag_r_mean,
        "n_grids_used":     int(n_grids),
    }


def cv_peak_consistency_with_perms(neurons, locs, idx_same_grids, shifts,
                                   n_peaks=1, min_dwell=25, min_shared_locs=5,
                                   weighted=False, n_loc=9,
                                   n_perms=0, seed=42):
    """As cv_peak_consistency, but also compute a null distribution of
    peak_r by circularly shifting the location series within each repeat.

    The null hypothesis: firing is independent of location given the task
    structure (grids, paired-groups, repeats, dwell distribution). We
    preserve the firing series and the grid/paired-group assignment, and
    only break the location-time alignment. Per-rep circular shift gives
    the most rigorous null: each rep gets an independent random offset
    drawn from [1, n_bins).

    Parameters
    ----------
    n_perms : int. If 0, only the observed value is computed (matches
        cv_peak_consistency behavior).
    seed    : int, RNG seed for reproducibility.

    Returns
    -------
    dict (same keys as cv_peak_consistency) plus:
        perm_peak_rs : list of float (length n_perms), null peak_r values.
                       Empty list if n_perms == 0.
    """
    obs = cv_peak_consistency(
        neurons=neurons, locs=locs, idx_same_grids=idx_same_grids,
        shifts=shifts, n_peaks=n_peaks,
        min_dwell=min_dwell, min_shared_locs=min_shared_locs,
        weighted=weighted, n_loc=n_loc,
    )
    perm_peak_rs = []
    if n_perms > 0:
        rng = np.random.default_rng(seed)
        n_reps, n_bins = np.asarray(locs).shape
        for _ in range(n_perms):
            ks = rng.integers(1, n_bins, size=n_reps)
            locs_perm = np.empty_like(locs)
            for r in range(n_reps):
                locs_perm[r] = np.roll(locs[r], -int(ks[r]))
            o = cv_peak_consistency(
                neurons=neurons, locs=locs_perm,
                idx_same_grids=idx_same_grids,
                shifts=shifts, n_peaks=n_peaks,
                min_dwell=min_dwell, min_shared_locs=min_shared_locs,
                weighted=weighted, n_loc=n_loc,
            )
            perm_peak_rs.append(o["peak_r"])
    obs["perm_peak_rs"] = perm_peak_rs
    return obs


def run_one_cell(payload, cfg):
    """Compute one cell's row. Picklable for joblib.

    If cfg['fixed_lags'] is a non-empty list of lags, ALSO compute the
    cross-validated test r at those fixed lags (no top-K selection),
    and (when permutations are enabled) the matching null distribution.

    If cfg['goal_relative_locs'] is set, the location series in the payload
    has already been remapped to goal-relative coordinates by
    build_cell_payload. n_loc is adjusted accordingly.
    """
    if payload is None:
        return None
    n_perms = cfg.get("n_permutations", 0) if cfg.get("run_permutations", False) else 0
    base_seed = int(cfg.get("random_seed", 42))
    cell_seed = base_seed + (abs(hash(payload["neuron_id"])) % (2**31))
    # adjust n_loc for goal-relative manhattan distance mode (5 categories)
    if payload.get("goal_relative_mode") == "manhattan_distance":
        n_loc = 5
    else:
        n_loc = 9
    out = cv_peak_consistency_with_perms(
        neurons=payload["neurons"],
        locs=payload["locs"],
        idx_same_grids=payload["idx_same_grids"],
        shifts=cfg["shifts_deg"],
        n_peaks=cfg["n_peaks"],
        min_dwell=cfg["min_dwell_bins"],
        min_shared_locs=cfg["min_shared_locs"],
        weighted=cfg["weighted_correlation"],
        n_loc=n_loc,
        n_perms=n_perms,
        seed=cell_seed,
    )
    out["neuron_id"] = payload["neuron_id"]

    fixed_lags = cfg.get("fixed_lags") or []
    if fixed_lags:
        fl = cv_at_fixed_lags(
            neurons=payload["neurons"],
            locs=payload["locs"],
            idx_same_grids=payload["idx_same_grids"],
            fixed_lags=fixed_lags,
            min_dwell=cfg["min_dwell_bins"],
            min_shared_locs=cfg["min_shared_locs"],
            weighted=cfg["weighted_correlation"],
            n_loc=n_loc,
        )
        out["fixed_lag_r_mean"] = fl["fixed_lag_r_mean"]
        out["fixed_lag_per_lag_r"] = fl["per_lag_r"]
        out["fixed_lag_lags"] = fl["fixed_lags"]

        # matching permutation null for the fixed-lag metric
        if n_perms > 0:
            rng = np.random.default_rng(cell_seed + 7919)
            n_reps, n_bins = np.asarray(payload["locs"]).shape
            perm_means = []
            for _ in range(n_perms):
                ks = rng.integers(1, n_bins, size=n_reps)
                locs_perm = np.empty_like(payload["locs"])
                for r in range(n_reps):
                    locs_perm[r] = np.roll(payload["locs"][r], -int(ks[r]))
                fp = cv_at_fixed_lags(
                    neurons=payload["neurons"],
                    locs=locs_perm,
                    idx_same_grids=payload["idx_same_grids"],
                    fixed_lags=fixed_lags,
                    min_dwell=cfg["min_dwell_bins"],
                    min_shared_locs=cfg["min_shared_locs"],
                    weighted=cfg["weighted_correlation"],
                    n_loc=n_loc,
                )
                perm_means.append(fp["fixed_lag_r_mean"])
            out["fixed_lag_perm_means"] = perm_means
    return out
