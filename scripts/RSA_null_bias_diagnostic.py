#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic — why is the dsr_fmri permutation null positively biased?

Loads the saved model + null-data RDMs from a completed RSA_DSR_ROIs_simple.py
run, then:

  1. Reports pairwise correlations among key model RDMs, INCLUDING a
     synthetic "within-config" indicator (1 = same-config pair,
     0 = cross-config).  If DSR is highly correlated with within-config
     and the other regressors aren't, that's the mechanism behind the bias.
  2. Runs OLS of each perm's null data RDM against several combo specs
     (dsr alone, ctrl_dsrFULL, ctrl_dsrFULL + within_config, etc.) using
     the same `evaluate_model_vec` the main pipeline uses.  Reports
     mean / std of the β_dsr null distribution under each combo — the
     bias should shrink when a nuisance regressor absorbs the offending
     structure.

No permutations are recomputed; this reads from the saved cache.

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mc.analyse.my_RSA import evaluate_model_vec

# ── Settings ─────────────────────────────────────────────────────────
RUN_TAG            = '2026-07-30_13-32-23'
TARGET_ROI         = 'mPFC'
N_CONFIGS          = 8
N_CONDS_PER_CONFIG = 12

DATA_DIR = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                '/ephys_humans/derivatives')
RUN_DIR  = DATA_DIR / 'group/DSR_RSA_simple_ROI' / RUN_TAG
OUT_DIR  = RUN_DIR / 'null_bias_diagnostic'
OUT_DIR.mkdir(exist_ok=True)

MODEL_NAMES = [
    'dsr_fmri', 'dsr_fmri_fut', 'dsr_fmri_informed',
    'state', 'location', 'bttn_curr', 'bttn_next',
    'l2_norm', 'reward_path', 'phase', 'uncover',
]

COMBOS = {
    'dsr_only':                 ['dsr_fmri'],
    'ctrl_dsrFULL':             ['state', 'location', 'bttn_curr', 'dsr_fmri'],
    'ctrl_dsrFULL_plus_wcfg':   ['state', 'location', 'bttn_curr',
                                 'within_config', 'dsr_fmri'],
    'dsr_and_within_config':    ['dsr_fmri', 'within_config'],
    'dsr_and_phase':            ['dsr_fmri', 'phase'],
    'covariates_only_no_dsr':   ['state', 'location', 'bttn_curr'],
}


# ── Load saved model RDMs ────────────────────────────────────────────
npz = np.load(RUN_DIR / f'rdms/rdms_{TARGET_ROI}.npz', allow_pickle=True)

def _model(name):
    return npz[f'model__split_halves__{name}'].astype(float)

# Build the within-config indicator using the same pair layout as
# compute_crosscorr → np.triu_indices(96, k=1) on the symmetrized 96×96 RDM.
n_conds = N_CONFIGS * N_CONDS_PER_CONFIG
i_idx, j_idx = np.triu_indices(n_conds, k=1)
within_config = (i_idx // N_CONDS_PER_CONFIG
                 == j_idx // N_CONDS_PER_CONFIG).astype(float)

MODELS = {name: _model(name) for name in MODEL_NAMES}
MODELS['within_config'] = within_config

n_pairs = len(within_config)
for name, r in MODELS.items():
    assert r.shape[0] == n_pairs, f"{name}: {r.shape} vs {n_pairs}"
print(f"Loaded {len(MODELS)} model RDMs ({n_pairs} pairs; "
      f"{int(within_config.sum())} within-config pairs, "
      f"{int((1 - within_config).sum())} cross-config)")


# ── Pairwise model correlations ──────────────────────────────────────
def _corr(a, b):
    a = a.ravel(); b = b.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])

corr_names = ['dsr_fmri', 'dsr_fmri_fut', 'dsr_fmri_informed',
              'state', 'location', 'bttn_curr', 'phase', 'within_config']
CORR = pd.DataFrame(index=corr_names, columns=corr_names, dtype=float)
for a in corr_names:
    for b in corr_names:
        CORR.loc[a, b] = _corr(MODELS[a], MODELS[b])
CORR.to_csv(OUT_DIR / 'model_rdm_correlations.csv')

print("\nPairwise model RDM correlations (Pearson r):")
print(CORR.round(3).to_string())

print("\nModel ↔ within_config correlations (sorted by |r|):")
wc = CORR['within_config'].drop('within_config').abs().sort_values(ascending=False)
for m, r in wc.items():
    print(f"  {m:22s}  r = {CORR.loc[m, 'within_config']:+.3f}")


# ── Load null data RDMs and empirical ────────────────────────────────
pkl_path = RUN_DIR / f'perm_data_rdms/perm_data_rdms_{TARGET_ROI}.pkl'
with open(pkl_path, 'rb') as f:
    cache = pickle.load(f)

Y_perm = cache['perms']['split_halves_z']       # (n_perms, n_pairs)
Y_emp  = cache['empirical']['split_halves_z']   # (n_pairs,)
n_perms = Y_perm.shape[0]
print(f"\nLoaded null data RDMs: {n_perms} perms × {Y_perm.shape[1]} pairs "
      f"({TARGET_ROI}, split_halves_z)")


# ── Run OLS for each combo ──────────────────────────────────────────
rows = []
for combo_name, feat_names in COMBOS.items():
    X = np.column_stack([MODELS[f] for f in feat_names])
    _, BETA_PERMS, _ = evaluate_model_vec(X, Y_perm)      # (n_perms, n_feat)
    _, beta_emp, _   = evaluate_model_vec(X, Y_emp[None, :])
    beta_emp = beta_emp[0]

    for k, f in enumerate(feat_names):
        null = BETA_PERMS[:, k]
        rows.append({
            'combo':       combo_name,
            'feature':     f,
            'null_mean':   float(np.nanmean(null)),
            'null_std':    float(np.nanstd(null)),
            'null_median': float(np.nanmedian(null)),
            'empirical':   float(beta_emp[k]),
            'p_right':     float(np.nanmean(null >= beta_emp[k])),
            'p_right_centered': float(
                np.nanmean(null - np.nanmean(null)
                           >= beta_emp[k] - np.nanmean(null))),
        })

results = pd.DataFrame(rows)
results.to_csv(OUT_DIR / 'null_bias_by_combo.csv', index=False)

print("\nOLS β under permutation null vs. empirical (split_halves_z):")
print(results.round(4).to_string(index=False))

# Focus print: the DSR row in every combo
print("\n── DSR summary across combos ──")
dsr = results[results['feature'] == 'dsr_fmri'].copy()
dsr['abs_bias'] = dsr['null_mean'].abs()
print(dsr[['combo', 'null_mean', 'null_std',
           'empirical', 'p_right', 'p_right_centered']]
      .round(4).to_string(index=False))

print(f"\nDone.  Outputs in: {OUT_DIR}")
