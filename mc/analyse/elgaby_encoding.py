#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
El-Gaby-style encoding analysis helpers.

For one neuron we:
  1. build the per-config 324-feature DSR design matrix
     (mc.simulation.predictions.model_DSR with no_phase_neurons=3)
  2. for each held-out test config, restrict both training and test bins
     to the bins where the within-trial phase index equals that test
     config's preferred phase
  3. fit ElasticNet on the training bins, predict the test bins
  4. score with the paper's 4-state-means Pearson r
  5. null = circular shift of the test neuron trace before downsampling

This module does NOT do any IO. The caller (Script 2) loads the per-
subject data, builds per-trial regressor + neuron arrays, looks up
`elgaby_pref_phase` from Script 1's CSV, then calls into here.
"""

import numpy as np
from sklearn.linear_model import ElasticNet

from mc.analyse import elgaby_tuning


# Default trial layout (matches scripts/encoding_analysis_simple.py).
N_BINS_PER_TRIAL = 360
N_STATES = 4
N_PHASES = 3
N_BINS_PER_STATE = N_BINS_PER_TRIAL // N_STATES         # 90
N_BINS_PER_PHASE = N_BINS_PER_STATE // N_PHASES         # 30


def state_means_at_pref_phase(trace, pref_phase,
                              n_bins_per_trial=N_BINS_PER_TRIAL,
                              n_states=N_STATES,
                              n_phases=N_PHASES):
    """Average a (n_bins_per_trial,) trace within each state, restricted
    to bins of the preferred phase.

    Returns a (n_states,) ndarray of per-state means.
    """
    state_idx = elgaby_tuning.state_bin_index(n_bins_per_trial, n_states)
    phase_idx = elgaby_tuning.phase_bin_index(n_bins_per_trial,
                                              n_states, n_phases)
    out = np.full(n_states, np.nan, dtype=float)
    for s in range(n_states):
        mask = (state_idx == s) & (phase_idx == pref_phase)
        block = trace[mask]
        if np.isfinite(block).any():
            out[s] = float(np.nanmean(block))
    return out


def pearsonr_4(a, b):
    """Pearson r for two short vectors (typically length 4)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    a, b = a[mask], b[mask]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def fit_elgaby_one_neuron(X_per_config, y_per_config, pref_phase_per_config,
                          state_tuned_per_config,
                          n_bins_per_trial=N_BINS_PER_TRIAL,
                          n_states=N_STATES, n_phases=N_PHASES,
                          alpha=0.01, l1_ratio=0.5, positive=True,
                          max_iter=2000,
                          scoring_mode='4state',
                          use_phase_mask=True,
                          n_permutations=500, rng=None):
    """Leave-one-config-out el-gaby encoding for ONE neuron.

    Parameters
    ----------
    X_per_config : list of (P, n_bins_per_trial) ndarrays
        Trial-averaged DSR design matrix for each config (in fixed config
        order).
    y_per_config : list of (n_bins_per_trial,) ndarrays
        Trial-averaged neuron trace for each config (same order).
    pref_phase_per_config : list of int
        Preferred phase to use when the corresponding config is the test
        config.  -1 means "no pref_phase available — skip this fold".
    state_tuned_per_config : list of bool
        Whether the neuron is el-gaby state-tuned in each config; the fit
        ignores this (no gating at fit time), but it is returned per
        fold so the caller can gate at aggregation time.
    scoring_mode : '4state' | 'all_bins'
        '4state' is the paper-faithful metric: average held-out actual and
        predicted activity within each state's preferred-phase bins, then
        correlate the resulting 4-vectors. 'all_bins' instead correlates
        the bin-level held-out actual vs predicted across all 360 bins
        (matches scripts/encoding_analysis_simple.py).  The permutation
        null uses the same metric.
    use_phase_mask : bool
        When True (paper), both training and test bins are restricted to
        the bins where the within-trial phase index equals the test
        config's preferred phase. When False, all 360 bins are used for
        training and test — drops the dependence on pref_phase entirely.
        With use_phase_mask=False, '4state' scoring still averages at
        pref_phase (so you'd usually want scoring_mode='all_bins' too).
    n_permutations : int
        Number of circular-shift draws for the per-fold null. 0 disables.
    rng : np.random.Generator or None

    Returns
    -------
    list of dicts (one per fold). Each dict has:
        test_cfg_idx, pref_phase, n_train_bins, n_test_bins,
        actual_means, predicted_means, r_4state, r_bins,
        p_perm, all_coefs_zero, coefs (P,)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_configs = len(X_per_config)
    P = X_per_config[0].shape[0]
    phase_idx = elgaby_tuning.phase_bin_index(n_bins_per_trial,
                                              n_states, n_phases)

    rows = []
    for test_i in range(n_configs):
        pref_phase = pref_phase_per_config[test_i]
        # When the phase mask is on, we need a valid pref_phase to know
        # which bins to subset to. Without it, pref_phase is only used by
        # the (optional) 4-state scoring metric, so we still tolerate -1
        # but the 4-state numbers will be NaN — that's fine for
        # scoring_mode='all_bins'.
        if use_phase_mask and pref_phase < 0:
            rows.append(_blank_fold(test_i, pref_phase, P))
            continue

        # With the phase mask: True for ~120 of 360 bins (the pref-phase
        # bins of all 4 states). Without it: all bins.
        if use_phase_mask:
            bin_mask = (phase_idx == pref_phase)
        else:
            bin_mask = np.ones(n_bins_per_trial, dtype=bool)

        # Training: concat the 7 OTHER configs' bins (masked or not).
        X_train_parts, y_train_parts = [], []
        for j in range(n_configs):
            if j == test_i:
                continue
            X_train_parts.append(X_per_config[j][:, bin_mask])
            y_train_parts.append(y_per_config[j][bin_mask])
        X_train = np.concatenate(X_train_parts, axis=1).T   # (n_train, P)
        y_train = np.concatenate(y_train_parts)             # (n_train,)

        if X_train.shape[0] < 5 or np.nanstd(y_train) < 1e-12:
            rows.append(_blank_fold(test_i, pref_phase, P))
            continue

        # Drop NaN targets if any.
        if np.isnan(y_train).any():
            keep = ~np.isnan(y_train)
            X_train, y_train = X_train[keep], y_train[keep]

        reg = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio,
            positive=positive, max_iter=max_iter,
            tol=1e-3, precompute=True,
        )
        reg.fit(X_train, y_train)
        coefs = reg.coef_.copy()

        # Test: predict the held-out config's pref-phase bins.
        X_test = X_per_config[test_i]                       # (P, n_bins)
        y_test = y_per_config[test_i]                       # (n_bins,)
        y_test_pred_all = (X_test.T @ coefs)                # (n_bins,)

        # Per-state mean at pref_phase (paper-style 4-vs-4 r).
        actual_means    = state_means_at_pref_phase(y_test,         pref_phase,
                                                    n_bins_per_trial,
                                                    n_states, n_phases)
        predicted_means = state_means_at_pref_phase(y_test_pred_all, pref_phase,
                                                    n_bins_per_trial,
                                                    n_states, n_phases)
        r_4state = pearsonr_4(actual_means, predicted_means)

        # Bin-level Pearson on the *all-bin* held-out trace (matches
        # scripts/encoding_analysis_simple.py).
        r_bins_all = pearsonr_4(y_test, y_test_pred_all)

        # The metric used for the permutation null + the primary score
        # column returned to the caller.
        emp_score = r_4state if scoring_mode == '4state' else r_bins_all

        # Circular-shift null on the actual test trace.
        p_perm = np.nan
        if n_permutations > 0 and np.isfinite(emp_score):
            shifts = rng.integers(0, n_bins_per_trial, size=n_permutations)
            n_ge = 0
            n_valid = 0
            for s in shifts:
                shifted_actual = np.roll(y_test, s)
                if scoring_mode == '4state':
                    shifted_means = state_means_at_pref_phase(
                        shifted_actual, pref_phase,
                        n_bins_per_trial, n_states, n_phases,
                    )
                    r_null = pearsonr_4(shifted_means, predicted_means)
                else:
                    r_null = pearsonr_4(shifted_actual, y_test_pred_all)
                if np.isfinite(r_null):
                    n_valid += 1
                    if r_null >= emp_score:
                        n_ge += 1
            if n_valid > 0:
                p_perm = (n_ge + 1) / (n_valid + 1)

        rows.append({
            'test_cfg_idx':     test_i,
            'pref_phase':       int(pref_phase),
            'n_train_bins':     int(X_train.shape[0]),
            'n_test_bins':      int(bin_mask.sum()),
            'actual_means':     actual_means,
            'predicted_means':  predicted_means,
            'r_4state':         r_4state,
            'r_bins_all':       r_bins_all,
            'scoring_mode':     scoring_mode,
            'r_used':           emp_score,
            'p_perm':           p_perm,
            'all_coefs_zero':   bool(np.all(coefs == 0)),
            'coefs':            coefs,
            'state_tuned':      bool(state_tuned_per_config[test_i]),
        })
    return rows


def _blank_fold(test_i, pref_phase, P):
    return {
        'test_cfg_idx':     int(test_i),
        'pref_phase':       int(pref_phase),
        'n_train_bins':     0,
        'n_test_bins':      0,
        'actual_means':     np.full(N_STATES, np.nan),
        'predicted_means':  np.full(N_STATES, np.nan),
        'r_4state':         np.nan,
        'r_bins_all':       np.nan,
        'scoring_mode':     '',
        'r_used':           np.nan,
        'p_perm':           np.nan,
        'all_coefs_zero':   True,
        'coefs':            np.zeros(P),
        'state_tuned':      False,
    }
