#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-ROI histograms of held-out encoding r, with a one-sided shift test
against 0.

For each gate variant (no gating, state-tuned, phase-tuned, both),
produces one figure with per-ROI subplots: histogram of mean-per-neuron
r values, plus annotations for n_cells, mean, sem, and one-sided
t_p_greater. The companion (neuron, config)-row test is also printed
inside the panel so we can see whether the two scales of analysis agree.

Reads the latest (or specified) `encoding_analysis_elgaby` run's
`encoding_results.csv`. No re-fitting.

Output: <run>/r_histograms/r_hist_<gate>.png + r_hist_summary.csv

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

# None = use most recent encoding run.
ENCODING_RUN_TAG = None

# Histogram bin edges (per-neuron mean r is bounded in [-1, 1]).
HIST_BINS = np.linspace(-1.0, 1.0, 41)

# ROIs with fewer cells than this are still plotted but flagged in the title.
SMALL_ROI_FLAG = 30

# The four gate variants we ran in Script 2.
GATE_VARIANTS = [
    ('all',                 False, False, 'all rows'),
    ('state_tuned',         True,  False, 'state-tuned only'),
    ('phase_tuned',         False, True,  'phase-tuned only'),
    ('state_and_phase',     True,  True,  'state AND phase tuned'),
]


# ── Resolve run + load data ──────────────────────────────────────────
def find_latest_run(base):
    cands = [d for d in os.listdir(base)
             if os.path.isdir(os.path.join(base, d))]
    if not cands:
        raise FileNotFoundError(f"No runs under {base}")
    cands.sort(key=lambda d: os.path.getmtime(os.path.join(base, d)))
    return cands[-1]


if ENCODING_RUN_TAG is None:
    ENCODING_RUN_TAG = find_latest_run(ENCODING_BASE)
RUN_DIR = os.path.join(ENCODING_BASE, ENCODING_RUN_TAG)
RESULTS_CSV = os.path.join(RUN_DIR, 'encoding_results.csv')
if not os.path.isfile(RESULTS_CSV):
    raise FileNotFoundError(f"Missing {RESULTS_CSV}")
print(f"Reading encoding run: {RUN_DIR}")

OUT_DIR = os.path.join(RUN_DIR, 'r_histograms')
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(RESULTS_CSV)
df['subject'] = df['subject'].map(lambda s: f'{int(s):02d}')
print(f"Loaded {len(df)} (neuron, test_config) rows.")


# ── Helpers ──────────────────────────────────────────────────────────
def filter_by_gate(df, gate_state, gate_phase):
    g = df[np.isfinite(df['r_used'])].copy()
    if gate_state:
        g = g[g['state_tuned']]
    if gate_phase and 'phase_tuned' in g.columns:
        g = g[g['phase_tuned']]
    return g


def per_neuron_mean_r(df):
    """One r per (subject, neuron), averaging the surviving folds."""
    return (df.groupby(['subject', 'neuron', 'roi'], as_index=False)
              .agg(mean_r_neuron=('r_used', 'mean'),
                   n_folds=('r_used', 'count')))


def one_sided_greater(values):
    vals = values[np.isfinite(values)]
    n = vals.size
    if n < 2 or np.std(vals) < 1e-12:
        return dict(n=int(n), mean=float(np.nan), sem=float(np.nan),
                    t=float(np.nan), t_p=float(np.nan),
                    wil_p=float(np.nan))
    try:
        t_res = stats.ttest_1samp(vals, 0.0, alternative='greater')
        t, t_p = float(t_res.statistic), float(t_res.pvalue)
    except Exception:
        t, t_p = float('nan'), float('nan')
    try:
        wil = stats.wilcoxon(vals, alternative='greater',
                             zero_method='wilcox')
        wil_p = float(wil.pvalue)
    except (ValueError, TypeError):
        wil_p = float('nan')
    return dict(n=int(n),
                mean=float(np.mean(vals)),
                sem=float(stats.sem(vals)),
                t=t, t_p=t_p,
                wil_p=wil_p)


# ── Build per-gate plots ─────────────────────────────────────────────
summary_rows = []
for gate_tag, gate_s, gate_p, title_suffix in GATE_VARIANTS:
    gated = filter_by_gate(df, gate_s, gate_p)
    if gated.empty:
        print(f"Gate {gate_tag}: no rows; skipping.")
        continue

    per_neuron = per_neuron_mean_r(gated)
    rois = sorted(per_neuron['roi'].dropna().unique().tolist())

    n_cols = 3
    n_rows = int(np.ceil(len(rois) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    for ax, roi in zip(axes.ravel(), rois):
        g_neuron = per_neuron[per_neuron['roi'] == roi]
        g_rows   = gated[gated['roi'] == roi]

        # Per-neuron mean r statistics (primary).
        stat_n = one_sided_greater(g_neuron['mean_r_neuron'].to_numpy())
        # Per-(neuron, config) statistics (matches Script 2's ROI summary).
        stat_r = one_sided_greater(g_rows['r_used'].to_numpy(dtype=float))

        small_flag = stat_n['n'] < SMALL_ROI_FLAG
        face = (0.75, 0.75, 0.75, 0.6) if small_flag else (0.35, 0.55, 0.8, 0.7)
        ax.hist(g_neuron['mean_r_neuron'], bins=HIST_BINS,
                edgecolor='k', color=face, linewidth=0.6)
        ax.axvline(0.0, color='k', lw=0.8, ls='--')
        if np.isfinite(stat_n['mean']):
            ax.axvline(stat_n['mean'], color='red', lw=1.5,
                       label=f"mean = {stat_n['mean']:+.3f}")
        ax.set_xlim(-1.0, 1.0)

        title = (f"{roi}  (n_cells={stat_n['n']}"
                 f"{'  SMALL' if small_flag else ''})")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('per-neuron mean r')
        ax.set_ylabel('# neurons')

        # Annotate panel with the two p-values.
        annot = (f"per-neuron:    mean={stat_n['mean']:+.3f}  "
                 f"SEM={stat_n['sem']:.3f}\n"
                 f"  t_p_greater = {stat_n['t_p']:.3g}   "
                 f"wilcoxon = {stat_n['wil_p']:.3g}\n"
                 f"per-(n,cfg):  n={stat_r['n']}  mean={stat_r['mean']:+.3f}  "
                 f"t_p={stat_r['t_p']:.3g}")
        ax.text(0.02, 0.98, annot, transform=ax.transAxes,
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white',
                          alpha=0.75, edgecolor='lightgray'))
        if np.isfinite(stat_n['mean']):
            ax.legend(loc='upper right', fontsize=8)

        summary_rows.append({
            'gate':                  gate_tag,
            'roi':                   roi,
            'n_cells':               stat_n['n'],
            'n_rows':                stat_r['n'],
            'per_neuron_mean_r':     stat_n['mean'],
            'per_neuron_sem':        stat_n['sem'],
            'per_neuron_t_p':        stat_n['t_p'],
            'per_neuron_wil_p':      stat_n['wil_p'],
            'per_row_mean_r':        stat_r['mean'],
            'per_row_t_p':           stat_r['t_p'],
            'small_n_flag':          small_flag,
        })

    for ax in axes.ravel()[len(rois):]:
        ax.axis('off')

    fig.suptitle(
        f'Per-neuron held-out r distributions — gate: {title_suffix}\n'
        f'(red line = mean, dashed = 0; one-sided H1: mean > 0; '
        f'grey panels: n_cells < {SMALL_ROI_FLAG})',
        fontsize=12,
    )
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f'r_hist_{gate_tag}.png')
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")

# Save the combined summary as CSV.
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, 'r_hist_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved combined summary -> {summary_path}")

# Print a compact preview ordered by gate then roi.
print("\nPreview (per-neuron statistics):")
print(summary_df[['gate', 'roi', 'n_cells',
                  'per_neuron_mean_r', 'per_neuron_t_p',
                  'per_neuron_wil_p', 'small_n_flag']].to_string(index=False))
