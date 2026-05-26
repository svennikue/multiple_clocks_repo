#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous-moderator analysis of phase tuning on encoding r.

Instead of binary gating on `elgaby_phase_tuned`, this script treats the
strength of phase tuning (`-log10(elgaby_phase_tuning_p)`) as a
continuous moderator and asks, per ROI:

    does r get larger as phase-tuning gets stronger?

Two tests per ROI:
  - **Linear regression** of r_used on the phase-tuning strength,
    one-sided H1: slope > 0. This is the cleanest answer to the
    moderator question.
  - **Weighted population r-vs-0 test**: per-row r_used weighted by
    phase-tuning strength. Same population test as the existing ROI
    summaries, just with weights instead of a binary gate.

The point: by using a continuous predictor we avoid the multiple-
comparisons problem of "which gate did we pick".  Phase tuning enters
the analysis once, monotonically, with a directional hypothesis.

No re-fitting needed — pure post-processing on the two CSVs.

Inputs:
  - latest encoding_analysis_elgaby run's encoding_results.csv
  - matching elgaby_tuning run's tuning_per_neuron_config.csv

Output: <encoding_run>/phase_moderation/

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')


# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
ENCODING_BASE = os.path.join(DATA_DIR, 'group', 'encoding_analysis_elgaby')
TUNING_BASE = os.path.join(DATA_DIR, 'group', 'elgaby_tuning')

# None -> latest run.
ENCODING_RUN_TAG = None
TUNING_RUN_TAG = None    # None -> read from encoding run's config.json

# Min p-value clip when log-transforming (paper-style: p=0 floored to
# 1e-300 -> moderator ~ 300, which we cap below).
MIN_P_FOR_LOG = 1e-50
# Hard cap on -log10(p) so a handful of outliers don't dominate the fit.
LOG10P_CAP = 50.0

# ROIs with fewer cells than this are still plotted but flagged.
SMALL_ROI_FLAG = 30


# ── Resolve runs ─────────────────────────────────────────────────────
def find_latest_run(base):
    cands = [d for d in os.listdir(base)
             if os.path.isdir(os.path.join(base, d))
             and not d.endswith('-null')]
    if not cands:
        raise FileNotFoundError(f"No runs under {base}")
    cands.sort(key=lambda d: os.path.getmtime(os.path.join(base, d)))
    return cands[-1]


if ENCODING_RUN_TAG is None:
    ENCODING_RUN_TAG = find_latest_run(ENCODING_BASE)
ENCODING_DIR = os.path.join(ENCODING_BASE, ENCODING_RUN_TAG)
ENCODING_CSV = os.path.join(ENCODING_DIR, 'encoding_results.csv')
print(f"Encoding run: {ENCODING_DIR}")

# Prefer the tuning run that this encoding run was wired to.
if TUNING_RUN_TAG is None:
    enc_cfg_path = os.path.join(ENCODING_DIR, 'config.json')
    if os.path.isfile(enc_cfg_path):
        with open(enc_cfg_path) as f:
            enc_cfg = json.load(f)
        TUNING_RUN_TAG = enc_cfg.get('tuning_run_tag', None)
    if TUNING_RUN_TAG is None:
        TUNING_RUN_TAG = find_latest_run(TUNING_BASE)
TUNING_CSV = os.path.join(TUNING_BASE, TUNING_RUN_TAG,
                          'tuning_per_neuron_config.csv')
print(f"Tuning run: {os.path.dirname(TUNING_CSV)}")

OUT_DIR = os.path.join(ENCODING_DIR, 'phase_moderation')
os.makedirs(OUT_DIR, exist_ok=True)


# ── Load + merge ─────────────────────────────────────────────────────
enc_df = pd.read_csv(ENCODING_CSV)
enc_df['subject'] = enc_df['subject'].map(lambda s: f'{int(s):02d}')
print(f"Encoding rows: {len(enc_df)}")

tun_df = pd.read_csv(TUNING_CSV)
tun_df['subject'] = tun_df['subject'].map(lambda s: f'{int(s):02d}')
keep_tun = ['subject', 'neuron', 'config',
            'elgaby_phase_tuning_p', 'elgaby_state_tuning_p',
            'elgaby_pref_phase', 'elgaby_pref_state',
            'elgaby_phase_tuned', 'elgaby_state_tuned']
tun_df = tun_df[keep_tun].rename(columns={'config': 'test_config'})
print(f"Tuning rows: {len(tun_df)}")

merged = enc_df.merge(tun_df, on=['subject', 'neuron', 'test_config'],
                      how='left')
print(f"Merged rows: {len(merged)}")

# Build the continuous moderator. NaN p's become NaN moderator → dropped.
p = merged['elgaby_phase_tuning_p'].to_numpy(dtype=float)
p_clipped = np.where(np.isfinite(p), np.clip(p, MIN_P_FOR_LOG, 1.0), np.nan)
merged['phase_strength'] = -np.log10(p_clipped)
merged['phase_strength'] = merged['phase_strength'].clip(upper=LOG10P_CAP)

# Filter to rows with finite r_used and finite moderator.
plot_df = merged[np.isfinite(merged['r_used']) &
                 np.isfinite(merged['phase_strength'])].copy()
print(f"Usable rows after filtering: {len(plot_df)}")


# ── Per-ROI tests ────────────────────────────────────────────────────
def regress_one_sided_greater(x, y):
    """OLS slope of y on x with one-sided p (H1: slope > 0)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) < 1e-12:
        return dict(slope=np.nan, intercept=np.nan,
                    pearson_r=np.nan, p_two=np.nan, p_one_greater=np.nan,
                    n=int(mask.sum()))
    res = stats.linregress(x[mask], y[mask])
    p_one = res.pvalue / 2 if res.slope > 0 else 1 - res.pvalue / 2
    return dict(slope=float(res.slope),
                intercept=float(res.intercept),
                pearson_r=float(res.rvalue),
                p_two=float(res.pvalue),
                p_one_greater=float(p_one),
                n=int(mask.sum()))


def spearman_one_sided_greater(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return dict(rho=np.nan, p_one_greater=np.nan)
    rho, p_two = stats.spearmanr(x[mask], y[mask])
    if not np.isfinite(rho):
        return dict(rho=np.nan, p_one_greater=np.nan)
    p_one = p_two / 2 if rho > 0 else 1 - p_two / 2
    return dict(rho=float(rho), p_one_greater=float(p_one))


def weighted_mean_test(values, weights):
    """Weighted mean of `values`, with a one-sided test of weighted mean > 0.

    Uses effective sample size n_eff = (sum w)^2 / sum(w^2) for the SEM.
    Equivalent to an unweighted t-test when all weights are equal.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if mask.sum() < 2:
        return dict(weighted_mean=np.nan, weighted_sem=np.nan,
                    n_eff=np.nan, t=np.nan, p_one_greater=np.nan)
    v, w = v[mask], w[mask]
    W = w.sum()
    m = (w * v).sum() / W
    # Reliability-weighted variance.
    var = (w * (v - m) ** 2).sum() / W
    n_eff = (W ** 2) / (w ** 2).sum()
    if n_eff < 2 or var < 1e-30:
        return dict(weighted_mean=float(m), weighted_sem=np.nan,
                    n_eff=float(n_eff), t=np.nan, p_one_greater=np.nan)
    sem = np.sqrt(var / n_eff)
    t = m / sem
    # one-sided greater
    df = n_eff - 1
    p_one = float(1 - stats.t.cdf(t, df=df))
    return dict(weighted_mean=float(m), weighted_sem=float(sem),
                n_eff=float(n_eff), t=float(t),
                p_one_greater=p_one)


# ── Plots + summary ──────────────────────────────────────────────────
rois = sorted(plot_df['roi'].dropna().unique().tolist())
print(f"\nROIs: {rois}")

summary_rows = []
n_cols = 3
n_rows = int(np.ceil(len(rois) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(4.5 * n_cols, 3.5 * n_rows),
                         squeeze=False)

for ax, roi in zip(axes.ravel(), rois):
    g = plot_df[plot_df['roi'] == roi]
    n_cells = g['neuron'].nunique()
    x = g['phase_strength'].to_numpy()
    y = g['r_used'].to_numpy()

    reg = regress_one_sided_greater(x, y)
    spr = spearman_one_sided_greater(x, y)
    wmt = weighted_mean_test(y, x)

    small_flag = n_cells < SMALL_ROI_FLAG
    pt_color = (0.5, 0.5, 0.5, 0.4) if small_flag else (0.25, 0.45, 0.75, 0.5)
    ax.scatter(x, y, s=12, color=pt_color, edgecolors='none')
    ax.axhline(0, color='k', lw=0.7, ls='--')

    # Regression line.
    if np.isfinite(reg['slope']):
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        ax.plot(xs, reg['intercept'] + reg['slope'] * xs,
                color='red', lw=1.5,
                label=f"slope={reg['slope']:+.3f}")
    ax.set_xlabel(r'phase-tuning strength: $-\log_{10}(p)$')
    ax.set_ylabel('r_used')
    title = f"{roi}  (n_cells={n_cells}, rows={len(g)}"
    if small_flag:
        title += ' SMALL'
    title += ')'
    ax.set_title(title, fontsize=10)

    annot = (
        f"OLS slope: {reg['slope']:+.3f}  "
        f"r={reg['pearson_r']:+.2f}  p_>={reg['p_one_greater']:.3g}\n"
        f"Spearman: rho={spr['rho']:+.2f}  "
        f"p_>={spr['p_one_greater']:.3g}\n"
        f"weighted r vs 0: m={wmt['weighted_mean']:+.3f}  "
        f"p_>={wmt['p_one_greater']:.3g}"
    )
    ax.text(0.02, 0.98, annot, transform=ax.transAxes,
            fontsize=8, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.8, edgecolor='lightgray'))
    if np.isfinite(reg['slope']):
        ax.legend(loc='lower right', fontsize=8)

    summary_rows.append({
        'roi':                roi,
        'n_cells':            int(n_cells),
        'n_rows':             int(len(g)),
        'slope':              reg['slope'],
        'slope_p_greater':    reg['p_one_greater'],
        'pearson_r_xy':       reg['pearson_r'],
        'spearman_rho':       spr['rho'],
        'spearman_p_greater': spr['p_one_greater'],
        'weighted_mean_r':    wmt['weighted_mean'],
        'weighted_sem':       wmt['weighted_sem'],
        'weighted_p_greater': wmt['p_one_greater'],
        'small_n_flag':       bool(small_flag),
    })

for ax in axes.ravel()[len(rois):]:
    ax.axis('off')

fig.suptitle(
    'Phase tuning as a continuous moderator of held-out r\n'
    '(one-sided H1: r increases with phase-tuning strength)',
    fontsize=12,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'phase_moderation_scatter.png'), dpi=140)
plt.close(fig)

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, 'phase_moderation_summary.csv')
summary_df.to_csv(summary_path, index=False)
merged_path = os.path.join(OUT_DIR, 'merged_rows.csv')
merged.to_csv(merged_path, index=False)

print(f"\nSaved scatter plot to {OUT_DIR}/phase_moderation_scatter.png")
print(f"Saved summary to {summary_path}")
print(f"Saved merged rows to {merged_path}")

print("\nPer-ROI summary:")
cols = ['roi', 'n_cells', 'slope', 'slope_p_greater',
        'spearman_rho', 'spearman_p_greater',
        'weighted_mean_r', 'weighted_p_greater', 'small_n_flag']
print(summary_df[cols].to_string(index=False))
