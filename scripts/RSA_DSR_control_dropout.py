
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control-regressor dropout for DSR RSA.

Loads the pre-computed per-pair RDMs from a saved run (rdms/rdms_<ROI>.npz),
then re-fits OLS for every subset of CONTROLS × DSR_VARIANTS.  No permutations,
no subprocess — just fast re-fitting of the regression on the already-saved data.

Outputs go into <RUN_DIR>/dropout_analysis/:
  dropout_results.csv          — beta + t-stat per (control-subset × DSR variant)
  dropout_heatmap_beta.pdf/png — sorted heatmap of DSR betas
  dropout_line_plot.pdf/png    — mean beta vs. n_controls added

Edit RELOAD_RUN to point at a different run folder.
"""
from __future__ import annotations

import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mc

# ── Settings ──────────────────────────────────────────────────────────
RELOAD_RUN   = '2026-07-30_11-11-36'
TARGET_ROI   = 'mPFC'
TEST         = 'split_halves_z'          # data key suffix in the .npz
MODEL_TEST   = 'split_halves'            # model key suffix (no _z)
DSR_VARIANTS = ['dsr_fmri', 'dsr_fmri_fut', 'dsr_fmri_informed']
CONTROLS     = ['state', 'location', 'l2_norm', 'bttn_curr', 'bttn_next', 'reward_path']

DATA_DIR = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                '/ephys_humans/derivatives')
RUN_DIR  = DATA_DIR / 'group/DSR_RSA_simple_ROI' / RELOAD_RUN
OUT_DIR  = RUN_DIR / 'dropout_analysis'
OUT_DIR.mkdir(exist_ok=True)

# ── Colours ────────────────────────────────────────────────────────────
_sg2 = mc.plotting.cell_results.SHOWGIRL2_DISCRETE
DSR_COLORS = {
    'dsr_fmri':          _sg2[1],   # mPFC colour
    'dsr_fmri_fut':      _sg2[4],   # mOFC colour
    'dsr_fmri_informed': _sg2[2],   # HC_mid colour
}
DSR_LABELS = {
    'dsr_fmri':          'DSR (fMRI)',
    'dsr_fmri_fut':      'DSR future',
    'dsr_fmri_informed': 'DSR informed',
}

# ── Load saved per-pair RDMs ──────────────────────────────────────────
npz_path = RUN_DIR / f'rdms/rdms_{TARGET_ROI}.npz'
if not npz_path.exists():
    raise FileNotFoundError(f"No saved RDMs at {npz_path}. "
                            "Run RSA_DSR_ROIs_simple.py first.")
npz = np.load(npz_path, allow_pickle=True)

y = npz[f'data__{TEST}']   # neural RDM pairs (N_pairs,)

def _rdm(name):
    return npz[f'model__{MODEL_TEST}__{name}'].astype(float)


# ── OLS helper ────────────────────────────────────────────────────────
def ols_betas_tstats(y, regressors):
    """OLS of y ~ intercept + regressors.  Returns (betas, t_stats) sans intercept."""
    X = np.column_stack([np.ones(len(y))] + [r.astype(float) for r in regressors])
    betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat    = X @ betas
    resid    = y - y_hat
    mse      = np.dot(resid, resid) / max(len(y) - X.shape[1], 1)
    XtXinv   = np.linalg.pinv(X.T @ X)
    se       = np.sqrt(mse * np.diag(XtXinv))
    t_stats  = betas / np.where(se > 0, se, np.nan)
    return betas[1:], t_stats[1:]   # drop intercept


# ── Sanity-check against known saved result ───────────────────────────
# ctrl_dsrFULL = state + location + bttn_curr + dsr_fmri → expected beta ≈ 0.044
_check_regs  = [_rdm('state'), _rdm('location'), _rdm('bttn_curr'), _rdm('dsr_fmri')]
_check_order = ['state', 'location', 'bttn_curr', 'dsr_fmri']
_betas, _ = ols_betas_tstats(y, _check_regs)
_dsr_beta  = _betas[_check_order.index('dsr_fmri')]
print(f"[sanity] dsr_fmri beta in ctrl_dsrFULL = {_dsr_beta:.4f}  "
      f"(expected ~0.044 from saved results)")


# ── Enumerate all control subsets × DSR variants ──────────────────────
rows = []
for dsr in DSR_VARIANTS:
    dsr_rdm = _rdm(dsr)
    for n_ctrl in range(len(CONTROLS) + 1):
        for subset in combinations(CONTROLS, n_ctrl):
            ctrl_rdms = [_rdm(c) for c in subset]
            all_names = [dsr] + list(subset)
            all_rdms  = [dsr_rdm] + ctrl_rdms
            betas, tstats = ols_betas_tstats(y, all_rdms)
            rows.append({
                'dsr_variant': dsr,
                'controls':    '+'.join(subset) if subset else 'none',
                'n_controls':  n_ctrl,
                'dsr_beta':    float(betas[0]),
                'dsr_t':       float(tstats[0]),
            })

results = pd.DataFrame(rows)
results.to_csv(OUT_DIR / 'dropout_results.csv', index=False)
print(f"Wrote dropout_results.csv  ({len(results)} rows)")


# ── Heatmap: DSR beta per (control subset × DSR variant) ─────────────
pivot = results.pivot_table(
    index=['n_controls', 'controls'],
    columns='dsr_variant',
    values='dsr_beta',
    aggfunc='first',
)[DSR_VARIANTS]

# sort rows by dsr_fmri beta descending
pivot = pivot.sort_values('dsr_fmri', ascending=False)

H = pivot.to_numpy()
vmax = np.nanpercentile(np.abs(H), 98)

row_labels = []
for (n_ct, ctrl), _ in pivot.iterrows():
    tag = 'no controls' if ctrl == 'none' else ctrl.replace('+', ' + ')
    row_labels.append(f'[{n_ct}]  {tag}')

col_labels = [DSR_LABELS[d] for d in DSR_VARIANTS]

fig_h = max(6.0, 0.22 * H.shape[0])
fig, ax = plt.subplots(figsize=(5.5, fig_h), constrained_layout=True)
im = ax.imshow(H, aspect='auto', cmap='RdBu_r',
               vmin=-vmax, vmax=vmax, interpolation='nearest')
ax.set_xticks(range(len(DSR_VARIANTS)))
ax.set_xticklabels(col_labels, rotation=20, ha='right', fontsize=9)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=6)

# annotate cells with beta value
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        val = H[i, j]
        if np.isfinite(val):
            txt_col = 'white' if abs(val) > 0.6 * vmax else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=5, color=txt_col)

ax.set_xlabel('DSR variant', fontsize=10)
ax.set_title(
    f'{TARGET_ROI} — DSR β per control subset\n'
    f'(sorted by dsr_fmri β descending; OLS on {len(y)} pairs)',
    fontsize=9,
)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label('DSR β (OLS)', fontsize=8)
cb.ax.tick_params(labelsize=7)

for ext in ('pdf', 'png'):
    fig.savefig(OUT_DIR / f'dropout_heatmap_beta.{ext}',
                dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Wrote dropout_heatmap_beta.{{pdf,png}}")


# ── Line plot: mean beta per n_controls ───────────────────────────────
summary = (results.groupby(['dsr_variant', 'n_controls'])['dsr_beta']
           .agg(['mean', 'sem']).reset_index())

fig, ax = plt.subplots(figsize=(4.5, 3.0), constrained_layout=True)
for dsr in DSR_VARIANTS:
    sub = summary[summary['dsr_variant'] == dsr]
    ax.plot(sub['n_controls'], sub['mean'],
            color=DSR_COLORS[dsr], label=DSR_LABELS[dsr], lw=2, marker='o', ms=5)
    ax.fill_between(sub['n_controls'],
                    sub['mean'] - sub['sem'],
                    sub['mean'] + sub['sem'],
                    color=DSR_COLORS[dsr], alpha=0.2)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel('Number of control regressors in GLM', fontsize=10)
ax.set_ylabel('DSR β (mean ± SEM across subsets)', fontsize=10)
ax.set_title(f'{TARGET_ROI} — DSR effect by control count', fontsize=11)
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(fontsize=8, frameon=False)

for ext in ('pdf', 'png'):
    fig.savefig(OUT_DIR / f'dropout_line_plot.{ext}',
                dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Wrote dropout_line_plot.{{pdf,png}}")

# ── Print top-10 control subsets by dsr_fmri beta ────────────────────
top = (results[results['dsr_variant'] == 'dsr_fmri']
       .sort_values('dsr_beta', ascending=False)
       .head(10)[['controls', 'n_controls', 'dsr_beta', 'dsr_t']])
print(f"\nTop-10 control subsets for {TARGET_ROI} dsr_fmri:")
print(top.round(4).to_string(index=False))
print(f"\nDone. Outputs in: {OUT_DIR}")
