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

# Gate level:
#   'neuron_frac' : el-gaby-like. A neuron passes the gate if
#                   `frac_configs_*_tuned` >= NEURON_FRAC_THRESHOLD; we
#                   then keep ALL of that neuron's folds for the histogram.
#   'per_row'     : original Script 2 behavior. A single (neuron, config)
#                   fold is kept iff `state_tuned`/`phase_tuned` is True for
#                   that fold. No per-neuron threshold applied.
GATE_LEVEL = 'neuron_frac'

# Neuron-level fraction-of-configs thresholds to sweep (only used when
# GATE_LEVEL == 'neuron_frac'). Each threshold gets its own subfolder under
# r_histograms/by_threshold/. 0.0 means "tuned in at least one config"
# (loosest, equivalent to the original per-row gate aggregated per neuron).
# 0.74 is added because el-gaby reports 74% of mFC neurons consistently
# tuned to goal-progress; check `neurons_passing_thresholds.png` to see
# which threshold actually reproduces a ~74% pass rate in your ACC sample.
NEURON_FRAC_THRESHOLDS = [0.0, 0.25, 1.0 / 3.0, 0.5, 0.74]


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
def per_neuron_frac_tuning(df):
    """Per (subject, neuron, roi): fraction of test configs where the
    neuron is state-tuned / phase-tuned (p<0.05 single-config test from
    Script 1).  This is what we threshold for neuron-level gating.
    """
    g = (df.groupby(['subject', 'neuron', 'roi'], as_index=False)
           .agg(n_configs=('test_config', 'count'),
                n_state_tuned=('state_tuned', 'sum'),
                n_phase_tuned=('phase_tuned', 'sum')))
    g['frac_state_tuned'] = g['n_state_tuned'] / g['n_configs']
    g['frac_phase_tuned'] = g['n_phase_tuned'] / g['n_configs']
    return g


def filter_by_gate_per_row(df, gate_state, gate_phase):
    """Original Script 2 gate: keep individual (n, cfg) folds where the
    test config itself was state/phase tuned at p<0.05."""
    g = df[np.isfinite(df['r_used'])].copy()
    if gate_state:
        g = g[g['state_tuned']]
    if gate_phase and 'phase_tuned' in g.columns:
        g = g[g['phase_tuned']]
    return g


def filter_by_gate_neuron(df, gate_state, gate_phase, frac_thr):
    """El-Gaby-like neuron-level gate: keep ALL folds from neurons whose
    `frac_configs_*_tuned` is >= frac_thr.  If neither gate flag is set,
    returns all finite-r rows (the 'all' variant).
    """
    finite = df[np.isfinite(df['r_used'])].copy()
    if not (gate_state or gate_phase):
        return finite
    pn = per_neuron_frac_tuning(finite)
    keep = pd.Series(True, index=pn.index)
    if gate_state:
        keep &= pn['frac_state_tuned'] >= frac_thr
    if gate_phase:
        keep &= pn['frac_phase_tuned'] >= frac_thr
    keep_keys = pn.loc[keep, ['subject', 'neuron']]
    if keep_keys.empty:
        return finite.iloc[0:0]
    return finite.merge(keep_keys, on=['subject', 'neuron'], how='inner')


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


# ── Overview: how many neurons pass each threshold per ROI ───────────
def plot_threshold_pass_rates(df_finite, thresholds, out_path):
    """For each ROI, show fraction of neurons passing
    `frac_*_tuned >= threshold` across all thresholds, separately for
    state and phase tuning. This is the calibration plot for picking a
    threshold that matches an external reference (e.g. el-gaby's ~74%)."""
    pn = per_neuron_frac_tuning(df_finite)
    rois = sorted(pn['roi'].dropna().unique().tolist())
    if not rois or not thresholds:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.7 * len(rois)), 5),
                             sharey=True)
    for ax, frac_col, label, color_base in [
        (axes[0], 'frac_state_tuned', 'state', plt.cm.Blues),
        (axes[1], 'frac_phase_tuned', 'phase', plt.cm.Oranges),
    ]:
        for k, thr in enumerate(thresholds):
            color = color_base(0.3 + 0.6 * (k + 1) / len(thresholds))
            ys, labels = [], []
            for roi in rois:
                g = pn[pn['roi'] == roi]
                pass_frac = ((g[frac_col] >= thr).mean()
                             if len(g) else np.nan)
                ys.append(pass_frac)
                labels.append(f"{roi}\n(N={len(g)})")
            ax.plot(np.arange(len(rois)), ys, marker='o',
                    color=color, label=f'thr ≥ {thr:.2f}')
        ax.set_title(f'Fraction of neurons passing {label}-tuning gate')
        ax.set_ylabel('fraction of neurons')
        ax.set_xticks(np.arange(len(rois)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.74, ls=':', color='k', lw=0.8, alpha=0.6,
                   label='el-gaby mFC ≈ 0.74')
        ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('Neuron-level pass rate per ROI vs. fraction-of-configs '
                 f'threshold (Script 1 single-config p<0.05)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


df_finite = df[np.isfinite(df['r_used'])].copy()
plot_threshold_pass_rates(
    df_finite, NEURON_FRAC_THRESHOLDS,
    out_path=os.path.join(OUT_DIR, 'neurons_passing_thresholds.png'),
)


# ── Per-(gate, threshold) histograms ─────────────────────────────────
def plot_one_gate(gated, gate_tag, title_suffix, panel_subtitle, out_path,
                  gate_s, gate_p, summary_rows):
    """Render the per-ROI histogram grid for one (gate, threshold) combo
    and append a row per ROI to summary_rows."""
    if gated.empty:
        print(f"  {gate_tag}: 0 rows; skipping.")
        return
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
        stat_n = one_sided_greater(g_neuron['mean_r_neuron'].to_numpy())
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
        ax.set_title(f"{roi}  (n_cells={stat_n['n']}"
                     f"{'  SMALL' if small_flag else ''})",
                     fontsize=10)
        ax.set_xlabel('per-neuron mean r')
        ax.set_ylabel('# neurons')

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
            'gate_level':            GATE_LEVEL,
            'gate':                  gate_tag,
            'neuron_frac_threshold': panel_subtitle.get('threshold'),
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

    sup = (f"Per-neuron held-out r distributions — gate: {title_suffix}"
           f"  ·  {panel_subtitle['text']}\n"
           f"(red = mean, dashed = 0; one-sided H1: mean > 0; "
           f"grey panels: n_cells < {SMALL_ROI_FLAG})")
    fig.suptitle(sup, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  Saved {out_path}")


summary_rows = []
if GATE_LEVEL == 'neuron_frac':
    base_dir = os.path.join(OUT_DIR, 'by_threshold')
    os.makedirs(base_dir, exist_ok=True)
    for thr in NEURON_FRAC_THRESHOLDS:
        thr_tag = f"thr_{thr:.2f}".replace('.', 'p')
        thr_dir = os.path.join(base_dir, thr_tag)
        os.makedirs(thr_dir, exist_ok=True)
        print(f"\nThreshold = {thr:.3f}  ->  {thr_dir}")
        for gate_tag, gate_s, gate_p, title_suffix in GATE_VARIANTS:
            gated = filter_by_gate_neuron(df_finite, gate_s, gate_p, thr)
            plot_one_gate(
                gated, gate_tag, title_suffix,
                panel_subtitle={'text': f'neuron-frac ≥ {thr:.2f}',
                                'threshold': float(thr)},
                out_path=os.path.join(thr_dir, f'r_hist_{gate_tag}.png'),
                gate_s=gate_s, gate_p=gate_p,
                summary_rows=summary_rows,
            )
elif GATE_LEVEL == 'per_row':
    by_row_dir = os.path.join(OUT_DIR, 'by_row_gate')
    os.makedirs(by_row_dir, exist_ok=True)
    print(f"\nPer-row gating -> {by_row_dir}")
    for gate_tag, gate_s, gate_p, title_suffix in GATE_VARIANTS:
        gated = filter_by_gate_per_row(df_finite, gate_s, gate_p)
        plot_one_gate(
            gated, gate_tag, title_suffix,
            panel_subtitle={'text': 'per-row gate (Script 2 default)',
                            'threshold': None},
            out_path=os.path.join(by_row_dir, f'r_hist_{gate_tag}.png'),
            gate_s=gate_s, gate_p=gate_p,
            summary_rows=summary_rows,
        )
else:
    raise ValueError(f"Unknown GATE_LEVEL: {GATE_LEVEL!r}")


# ── Save combined summary CSV ────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUT_DIR, 'r_hist_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved combined summary -> {summary_path}")

# Print a compact preview ordered by threshold then gate then roi.
print("\nPreview (per-neuron statistics):")
preview_cols = ['neuron_frac_threshold', 'gate', 'roi', 'n_cells',
                'per_neuron_mean_r', 'per_neuron_t_p',
                'per_neuron_wil_p', 'small_n_flag']
print(summary_df[preview_cols].to_string(index=False))
