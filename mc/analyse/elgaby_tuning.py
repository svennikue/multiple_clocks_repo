#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State / phase tuning following El-Gaby et al. (Nature 2024).

A neuron is "el-gaby state tuned" in a given config if, after taking the
peak firing rate per state per correct trial, z-scoring within trial, and
extracting the z-scores at the preferred state, a one-sample t-test against
0 rejects (two-sided). Phase tuning is computed with the same logic on
the 3 within-state phase bins.

The names use the `elgaby_` prefix throughout so they do NOT clash with
our existing `state` model (which means something different).
"""

import numpy as np
from scipy import stats


# Default trial layout for the human DSR paradigm.
N_BINS_PER_TRIAL = 360
N_STATES = 4
N_PHASES = 3
N_BINS_PER_STATE = N_BINS_PER_TRIAL // N_STATES                # 90
N_BINS_PER_PHASE = N_BINS_PER_STATE // N_PHASES                # 30


def state_bin_index(n_bins_per_trial=N_BINS_PER_TRIAL, n_states=N_STATES):
    """Return a (n_bins_per_trial,) array of state indices 0..n_states-1."""
    bins_per_state = n_bins_per_trial // n_states
    return np.repeat(np.arange(n_states), bins_per_state)


def phase_bin_index(n_bins_per_trial=N_BINS_PER_TRIAL,
                    n_states=N_STATES, n_phases=N_PHASES):
    """Return a (n_bins_per_trial,) array of phase indices 0..n_phases-1.

    Phases tile within each state (e.g. 30 bins each for 90-bin states).
    """
    bins_per_state = n_bins_per_trial // n_states
    bins_per_phase = bins_per_state // n_phases
    one_state = np.repeat(np.arange(n_phases), bins_per_phase)
    return np.tile(one_state, n_states)


def _peak_per_state(trial_arr, state_idx, n_states):
    """(n_trials, n_bins) -> (n_trials, n_states) of within-state peak FR.

    NaNs (e.g. neuron not recorded for that bin) are ignored. Any state
    with no finite value returns NaN.
    """
    n_trials = trial_arr.shape[0]
    out = np.full((n_trials, n_states), np.nan, dtype=float)
    for s in range(n_states):
        cols = (state_idx == s)
        block = trial_arr[:, cols]
        # nanmax over an all-NaN row raises a RuntimeWarning, so guard it.
        with np.errstate(all='ignore'):
            row_has_any = np.isfinite(block).any(axis=1)
            if row_has_any.any():
                vals = np.where(row_has_any,
                                np.nanmax(np.where(np.isfinite(block),
                                                   block, -np.inf), axis=1),
                                np.nan)
                out[:, s] = vals
    return out


def _peak_per_phase(trial_arr, phase_idx, n_phases):
    """(n_trials, n_bins) -> (n_trials, n_phases) of peak FR per phase,
    pooling across the whole trial (i.e. across all states).

    This mirrors the state-tuning logic: one peak value per phase per
    trial, gathered from every bin that belongs to that phase regardless
    of which state it sits in.
    """
    n_trials = trial_arr.shape[0]
    out = np.full((n_trials, n_phases), np.nan, dtype=float)
    for p in range(n_phases):
        cols = (phase_idx == p)
        block = trial_arr[:, cols]
        with np.errstate(all='ignore'):
            row_has_any = np.isfinite(block).any(axis=1)
            if row_has_any.any():
                vals = np.where(row_has_any,
                                np.nanmax(np.where(np.isfinite(block),
                                                   block, -np.inf), axis=1),
                                np.nan)
                out[:, p] = vals
    return out


# def _row_zscore(arr):
#     """Row-wise z-score. Rows with zero std return NaNs."""
#     m = np.nanmean(arr, axis=1, keepdims=True)
#     s = np.nanstd(arr, axis=1, keepdims=True, ddof=0)
#     z = np.where(s > 1e-12, (arr - m) / s, np.nan)
#     return z

def _row_zscore(arr, eps=1e-12, fill_constant=0.0):
    """
    Row-wise z-score.
    Rows with near-zero std are filled with `fill_constant`, usually 0.
    """
    m = np.nanmean(arr, axis=1, keepdims=True)
    s = np.nanstd(arr, axis=1, keepdims=True, ddof=0)
    z = np.full_like(arr, fill_constant, dtype=float)
    np.divide(arr - m,s,out=z,where=s > eps)

    return z



def _ttest_against_zero(values):
    """Two-sided one-sample t-test against 0. Returns (t, p) or (nan, nan)."""
    vals = values[np.isfinite(values)]
    if vals.size < 2 or np.std(vals) < 1e-12:
        return np.nan, np.nan
    t = stats.ttest_1samp(vals, popmean=0.0)
    return float(t.statistic), float(t.pvalue)


def state_tuning(trial_arr,
                 n_bins_per_trial=N_BINS_PER_TRIAL,
                 n_states=N_STATES):
    """Test whether a neuron is state-tuned in this config.

    Parameters
    ----------
    trial_arr : (n_trials, n_bins_per_trial) ndarray
        Firing rate per bin for each correct trial of a single config.

    Returns
    -------
    dict with:
        n_trials              : int
        elgaby_pref_state     : int in [0, n_states) or -1 if undefined
        elgaby_state_tuning_t : float
        elgaby_state_tuning_p : float
        elgaby_state_tuned    : bool  (p < 0.05)
        z_per_trial           : (n_trials,) z-score at preferred state
                                across trials (for diagnostics)
    """
    trial_arr = np.asarray(trial_arr, dtype=float)
    if trial_arr.ndim != 2 or trial_arr.shape[0] < 2:
        return _empty_state_result(trial_arr.shape[0] if trial_arr.ndim == 2 else 0)

    state_idx = state_bin_index(n_bins_per_trial, n_states)
    peaks = _peak_per_state(trial_arr, state_idx, n_states)   # (n_trials, n_states)
    z = _row_zscore(peaks)                                    # (n_trials, n_states)

    # Preferred state = state with highest mean z across trials.
    mean_z = np.nanmean(z, axis=0)
    if not np.isfinite(mean_z).any():
        return _empty_state_result(trial_arr.shape[0])
    pref_state = int(np.nanargmax(mean_z))

    z_pref = z[:, pref_state]
    # import pdb; pdb.set_trace()
    t, p = _ttest_against_zero(z_pref)

    return {
        'n_trials':              int(trial_arr.shape[0]),
        'elgaby_pref_state':     pref_state,
        'elgaby_state_tuning_t': t,
        'elgaby_state_tuning_p': p,
        'elgaby_state_tuned':    bool(np.isfinite(p) and p < 0.05),
        'z_per_trial':           z_pref,
    }


def _empty_state_result(n_trials):
    return {
        'n_trials':              int(n_trials),
        'elgaby_pref_state':     -1,
        'elgaby_state_tuning_t': np.nan,
        'elgaby_state_tuning_p': np.nan,
        'elgaby_state_tuned':    False,
        'z_per_trial':           np.array([]),
    }


def phase_tuning(trial_arr,
                 n_bins_per_trial=N_BINS_PER_TRIAL,
                 n_states=N_STATES,
                 n_phases=N_PHASES):
    """Test whether a neuron is phase-tuned in this config.

    Mirrors `state_tuning`, but pools across states: for each state we
    compute the per-trial peak in each phase, z-score across phases within
    trial-and-state, then concatenate the z-scores at the preferred phase
    across all (trial, state) and t-test against 0.

    Preferred phase = argmax of mean activity per phase (paper-faithful,
    used by Script 2 as the regression's `pref_phase`). The significance
    test asks whether that same phase has a significantly positive z-score
    in the (n_trials, n_phases) within-trial-z-scored peak matrix.

    Returns
    -------
    dict with:
        elgaby_pref_phase     : int in [0, n_phases) or -1
        elgaby_phase_tuning_t : float
        elgaby_phase_tuning_p : float
        elgaby_phase_tuned    : bool  (p < 0.05)
        mean_act_per_phase    : (n_phases,) mean activity per phase
    """
    trial_arr = np.asarray(trial_arr, dtype=float)
    if trial_arr.ndim != 2 or trial_arr.shape[0] < 2:
        return _empty_phase_result(n_phases)

    phase_idx = phase_bin_index(n_bins_per_trial, n_states, n_phases)

    # Mean activity per phase across the whole config. argmax of this is
    # used as the regression's "preferred phase" (paper's tuning_phase_max).
    mean_act_per_phase = np.full(n_phases, np.nan, dtype=float)
    for p in range(n_phases):
        cols = (phase_idx == p)
        with np.errstate(all='ignore'):
            mean_act_per_phase[p] = np.nanmean(trial_arr[:, cols])

    if not np.isfinite(mean_act_per_phase).any():
        return _empty_phase_result(n_phases)
    pref_phase_argmax = int(np.nanargmax(mean_act_per_phase))

    # Significance test, mirroring the state-tuning logic but on phases.
    # We test the z-scores AT pref_phase_argmax (the same phase used as the
    # regression's preferred phase) so the boolean directly answers
    # "is the regression-relevant pref_phase significantly preferred?".
    peaks = _peak_per_phase(trial_arr, phase_idx, n_phases)   # (n_trials, n_phases)
    z = _row_zscore(peaks)                                    # (n_trials, n_phases)
    z_pref = z[:, pref_phase_argmax]
    # import pdb; pdb.set_trace()
    t, p = _ttest_against_zero(z_pref)

    return {
        'elgaby_pref_phase':     pref_phase_argmax,
        'elgaby_phase_tuning_t': t,
        'elgaby_phase_tuning_p': p,
        'elgaby_phase_tuned':    bool(np.isfinite(p) and p < 0.05),
        'mean_act_per_phase':    mean_act_per_phase,
    }


def _empty_phase_result(n_phases, mean_act=None, pref_argmax=-1):
    return {
        'elgaby_pref_phase':     -1 if pref_argmax < 0 else pref_argmax,
        'elgaby_phase_tuning_t': np.nan,
        'elgaby_phase_tuning_p': np.nan,
        'elgaby_phase_tuned':    False,
        'mean_act_per_phase':    (mean_act if mean_act is not None
                                  else np.full(n_phases, np.nan)),
    }
