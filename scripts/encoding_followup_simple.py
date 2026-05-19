#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Follow-up analyses on the encoding_analysis_simple result table.

For entorhinal and ACC neurons:
  Fig 1  - distribution of held-out r values across models, per ROI:
           heatmap of mean_r (neuron x model) + violin of pooled fold r.
  Fig 2  - top vs runner-up scatter, coloured by category. Plotted twice:
           once for all neurons, once for the perm-significant subset.
  Fig 3  - per ROI, neuron split: unique-winner counts per model and
           multi-winner memberships per model.
  CSV    - per-neuron model assignment (winner / co-winners / no_fit).

Stats used per neuron:
  - paired Wilcoxon across folds, one-sided H1: r_top > r_other
    => co-winners are models the top does NOT significantly beat.
  - permutation p_perm (already in the table) => is the top model > chance?

@author: Svenja Kuchenhoff
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# ── Settings ──────────────────────────────────────────────────────────

RESULT_DIR = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/encoding_analysis_simple/2026-05-18_16-33-05/')

# RESULT_DIR = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/encoding_analysis_simple/2026-05-12_17-48-06-all-500perms-l20-001-fixedbuttons/')

RESULT_CSV = os.path.join(RESULT_DIR, 'encoding_results.csv')

TARGET_ROIS  = ['ACC', 'HC_anterior', 'HC_posterior']
ALPHA_PERM   = 0.05   # p_perm threshold: model significantly > chance
ALPHA_PAIRED = 0.05   # paired test threshold: top model > other model

# DSR variants we want to compare against each other and against state /
# state_phase.  Only the ones actually present in the result table will be
# used downstream.
DSR_VARIANTS    = ['dsr', 'dsr_now_next', 'dsr_only_fut']
STATE_VARIANTS  = ['state', 'state_phase']

RUN_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUT_DIR = os.path.join(RESULT_DIR, 'followup', RUN_TAG)
os.makedirs(OUT_DIR, exist_ok=True)


# ── Load ──────────────────────────────────────────────────────────────
def parse_folds(x):
    """Parse '[v, v, nan, ...]' string from CSV to a float ndarray."""
    if isinstance(x, (list, np.ndarray)):
        return np.asarray(x, dtype=float)
    s = str(x).strip().lstrip('[').rstrip(']')
    if not s:
        return np.array([], dtype=float)
    return np.fromstring(s, sep=',', dtype=float)


print(f"Loading {RESULT_CSV}")
df_all = pd.read_csv(RESULT_CSV)
n_before = len(df_all)
df_all = df_all.dropna(subset=['mean_r']).copy()
df_all['r_per_fold'] = df_all['r_per_fold'].map(parse_folds)
print(f"  dropped {n_before - len(df_all)} rows with NaN mean_r "
      "(all-zero-coef / unfittable neuron x model).")

# `df`     -> the two target ROIs (used by figs 1, 2, 4, 5)
# `df_all` -> every ROI in the CSV (used by the cross-ROI dsr grid)
df = df_all[df_all['roi'].isin(TARGET_ROIS)].copy()

MODELS   = sorted(df_all['model'].unique().tolist())
ALL_ROIS = sorted(df_all['roi'].unique().tolist())
print(f"  {len(df)} rows in target ROIs, {df['neuron'].nunique()} neurons; "
      f"{len(df_all)} rows total across {ALL_ROIS}; "
      f"{len(MODELS)} models.")


# ── Per-neuron classification ─────────────────────────────────────────
def paired_p_greater(r_top, r_other):
    """One-sided Wilcoxon across folds, H1: r_top > r_other."""
    d = np.asarray(r_top) - np.asarray(r_other)
    if len(d) < 2 or np.all(d == 0):
        return np.nan
    try:
        return float(stats.wilcoxon(d, alternative='greater',
                                    zero_method='wilcox').pvalue)
    except ValueError:
        return np.nan


def classify_neuron(g):
    g = g.set_index('model')
    here   = [m for m in MODELS if m in g.index]
    mean_r = {m: float(g.loc[m, 'mean_r']) for m in here}
    p_perm = {m: float(g.loc[m, 'p_perm']) for m in here}
    folds  = {m: g.loc[m, 'r_per_fold']    for m in here}

    ranked = sorted(here, key=lambda m: mean_r[m], reverse=True)
    top    = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None

    paired = {m: (np.nan if m == top
                  else paired_p_greater(folds[top], folds[m]))
              for m in here}

    # Co-winners: top + every model the top does NOT significantly beat.
    co_winners = [top] + [m for m in here
                          if m != top
                          and (np.isnan(paired[m])
                               or paired[m] >= ALPHA_PAIRED)]

    top_sig = p_perm[top] < ALPHA_PERM
    if not top_sig:
        category = 'no_fit'
    elif runner is not None and paired[runner] < ALPHA_PAIRED:
        category = 'unique_winner'
    else:
        category = 'multi_winner'

    return {
        'category':                  category,
        'top_model':                 top,
        'runner_up_model':           runner,
        'co_winner_models':          ';'.join(co_winners),
        'n_co_winners':              len(co_winners),
        'best_mean_r':               mean_r[top],
        'runner_up_mean_r':          mean_r[runner] if runner else np.nan,
        'best_p_perm':               p_perm[top],
        'paired_p_top_vs_runner_up': (paired[runner]
                                      if runner is not None else np.nan),
        'n_models_perm_sig':         sum(1 for m in here
                                         if p_perm[m] < ALPHA_PERM),
    }


def build_summary(df_roi):
    rows = []
    for neuron, g in df_roi.groupby('neuron', sort=False):
        row = {'neuron': neuron,
               'subject': g['subject'].iloc[0],
               'roi':     g['roi'].iloc[0]}
        row.update(classify_neuron(g))
        rows.append(row)
    return pd.DataFrame(rows)


summary_per_roi = {roi: build_summary(df[df['roi'] == roi])
                   for roi in TARGET_ROIS}
summary_all = pd.concat(summary_per_roi.values(), ignore_index=True)

csv_path = os.path.join(OUT_DIR, 'neuron_model_assignment.csv')
summary_all.to_csv(csv_path, index=False)
print(f"Wrote {csv_path}")
print(summary_all.groupby(['roi', 'category']).size()
      .unstack(fill_value=0).to_string())


# ── Fig 1: model-fit distributions per ROI ────────────────────────────
def plot_model_fit_distributions(df_roi, summary_roi, roi, save_path):
    # left panel: per-neuron mean_r heatmap, sorted by best model
    mat = (df_roi.pivot_table(index='neuron', columns='model',
                              values='mean_r', aggfunc='first')
                 .reindex(columns=MODELS))
    order = summary_roi.sort_values(
        ['top_model', 'best_mean_r'], ascending=[True, False])['neuron'].tolist()
    mat = mat.loc[[n for n in order if n in mat.index]]

    # right panel: pooled r_per_fold per model (NaN folds dropped)
    fold_data = []
    for m in MODELS:
        sub = df_roi.loc[df_roi['model'] == m, 'r_per_fold'].tolist()
        vals = np.concatenate(sub) if sub else np.array([])
        vals = vals[~np.isnan(vals)]
        fold_data.append(vals if vals.size else np.array([0.0]))

    fig, axes = plt.subplots(
        1, 2, figsize=(12, max(4.5, 0.035 * len(mat) + 2.5)),
        gridspec_kw={'width_ratios': [3, 2]})

    vmax = float(np.nanmax(np.abs(mat.values))) if mat.size else 0.1
    im = axes[0].imshow(mat.values, aspect='auto', cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax, interpolation='nearest')
    axes[0].set_xticks(range(len(MODELS)))
    axes[0].set_xticklabels(MODELS, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel(f'neurons (n={len(mat)})', fontsize=9)
    axes[0].set_title(f'{roi}: mean_r per (neuron x model)', fontsize=10)
    fig.colorbar(im, ax=axes[0], label='mean_r')

    parts = axes[1].violinplot(fold_data, showmedians=True, showextrema=False)
    for body in parts['bodies']:
        body.set_facecolor('tab:blue')
        body.set_alpha(0.45)
        body.set_edgecolor('none')
    if 'cmedians' in parts:
        parts['cmedians'].set_color('0.15')
    axes[1].axhline(0, color='0.5', lw=0.7)
    axes[1].set_xticks(range(1, len(MODELS) + 1))
    axes[1].set_xticklabels(MODELS, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('held-out r (per fold, pooled across neurons)',
                       fontsize=9)
    axes[1].set_title(f'{roi}: r_per_fold distribution per model',
                      fontsize=10)

    fig.suptitle(f'{roi}: how similarly do the models fit each neuron?',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {save_path}")


for roi in TARGET_ROIS:
    plot_model_fit_distributions(
        df[df['roi'] == roi], summary_per_roi[roi], roi,
        os.path.join(OUT_DIR, f'fig1_{roi}_model_fits.png'))


# ── Fig 2: top vs runner-up, paired test x perm-sig ───────────────────
CAT_STYLE = {
    'no_fit':        ('0.7',         'no perm-sig model'),
    'multi_winner':  ('tab:orange',  'perm-sig + tied tops'),
    'unique_winner': ('tab:green',   'perm-sig + unique winner'),
}


def plot_top_vs_runner(summary, save_path, suptitle):
    fig, axes = plt.subplots(1, len(TARGET_ROIS),
                             figsize=(5.5 * len(TARGET_ROIS), 5),
                             squeeze=False)
    for ax, roi in zip(axes[0], TARGET_ROIS):
        s = summary[summary['roi'] == roi]
        if len(s) == 0:
            ax.set_title(f'{roi}: no neurons')
            continue

        for cat, (color, label) in CAT_STYLE.items():
            sub = s[s['category'] == cat]
            if len(sub) == 0:
                continue
            ax.scatter(sub['runner_up_mean_r'], sub['best_mean_r'],
                       c=color, s=24, alpha=0.75, edgecolor='none',
                       label=f'{label} (n={len(sub)})')

        lo = float(min(s['runner_up_mean_r'].min(),
                       s['best_mean_r'].min(), 0))
        hi = float(max(s['runner_up_mean_r'].max(),
                       s['best_mean_r'].max()))
        ax.plot([lo, hi], [lo, hi], color='0.4', lw=0.7, ls='--')
        ax.axhline(0, color='0.7', lw=0.5)
        ax.axvline(0, color='0.7', lw=0.5)
        ax.set_xlabel('runner-up model mean_r', fontsize=9)
        ax.set_ylabel('top model mean_r', fontsize=9)
        ax.set_title(f'{roi} (n={len(s)})', fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc='lower right')

    fig.suptitle(f'{suptitle}  |  perm alpha={ALPHA_PERM}  |  '
                 f'paired alpha={ALPHA_PAIRED} (Wilcoxon)', fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {save_path}")


plot_top_vs_runner(
    summary_all,
    os.path.join(OUT_DIR, 'fig2_paired_vs_perm_all.png'),
    'Top vs runner-up model (all neurons)')

plot_top_vs_runner(
    summary_all[summary_all['category'] != 'no_fit'],
    os.path.join(OUT_DIR, 'fig2_paired_vs_perm_sig.png'),
    'Top vs runner-up model (neurons with >=1 perm-sig model)')


# ── Focal-model targeted analysis (state, dsr) ────────────────────────
# For each focal model F, drop neurons where motor / rival focal models
# paired-beat F. Then re-look at the population r distribution + the
# overlap with other models among neurons that DO perm-encode F.
FOCAL_MODELS = ['dsr', 'state', 'state_phase']
MOTOR_MODELS = ['bttn_prev', 'bttn_curr', 'bttn_next', 'uncover']
# For the focal-filter analysis, the "rival" is the model that — if it
# paired-beats the focal — should disqualify the neuron. For dsr we keep
# state/state_phase as rivals; for state/state_phase the rival is dsr.
RIVAL_FOCAL  = {'state': 'dsr', 'state_phase': 'dsr', 'dsr': 'state'}
RELATED = {
    'state':       ['state_phase'],
    'state_phase': ['state', 'phase'],
    'dsr':         ['state_phase', 'dsr_now_next', 'dsr_only_fut',
                    'phase', 'midnight', 'location'],
}


def neurons_passing_focal_filter(df_roi, focal):
    """Set of neurons where neither motor nor rival paired-beats focal."""
    drops = MOTOR_MODELS + [RIVAL_FOCAL[focal]]
    kept = set()
    for n, g in df_roi.groupby('neuron', sort=False):
        gi = g.set_index('model')
        if focal not in gi.index:
            continue
        r_focal = gi.loc[focal, 'r_per_fold']
        beaten = False
        for m in drops:
            if m not in gi.index:
                continue
            p = paired_p_greater(gi.loc[m, 'r_per_fold'], r_focal)
            if not np.isnan(p) and p < ALPHA_PAIRED:
                beaten = True
                break
        if not beaten:
            kept.add(n)
    return kept


focal_kept = {(roi, focal): neurons_passing_focal_filter(
                  df[df['roi'] == roi], focal)
              for roi in TARGET_ROIS for focal in FOCAL_MODELS}

print('\nFocal-model filter (kept / total neurons; perm-sig before filter):')
for roi in TARGET_ROIS:
    df_roi = df[df['roi'] == roi]
    for focal in FOCAL_MODELS:
        full   = df_roi.loc[df_roi['model'] == focal, 'neuron'].unique()
        kept   = focal_kept[(roi, focal)]
        f_sig  = df_roi.loc[(df_roi['model'] == focal)
                            & (df_roi['p_perm'] < ALPHA_PERM),
                            'neuron'].nunique()
        print(f'  {roi:>10} {focal:>5}: {len(kept)} / {len(full)}  '
              f'(perm-sig: {f_sig})')


# ── Fig 4: r distribution before vs after focal filter ────────────────
def plot_focal_r_distributions(save_path):
    fig, axes = plt.subplots(
        len(TARGET_ROIS), len(FOCAL_MODELS),
        figsize=(5.5 * len(FOCAL_MODELS), 3.8 * len(TARGET_ROIS)),
        squeeze=False)

    for i, roi in enumerate(TARGET_ROIS):
        df_roi = df[df['roi'] == roi]
        for j, focal in enumerate(FOCAL_MODELS):
            ax    = axes[i, j]
            full  = df_roi[df_roi['model'] == focal]
            kept  = focal_kept[(roi, focal)]
            sub   = full[full['neuron'].isin(kept)]

            r_full = full['mean_r'].dropna().values
            r_sub  = sub['mean_r'].dropna().values
            if r_full.size < 2:
                ax.set_title(f'{roi} – {focal}: no data')
                continue

            lim  = max(0.05, 1.05 * float(np.max(np.abs(r_full))))
            bins = np.linspace(-lim, lim, 21)

            _, p_full = stats.ttest_1samp(r_full, 0.0)
            p_sub = (stats.ttest_1samp(r_sub, 0.0).pvalue
                     if r_sub.size > 1 else np.nan)

            ax.hist(r_full, bins=bins, color='0.55', alpha=0.35,
                    edgecolor='none',
                    label=f'all (n={len(r_full)})  p={p_full:.1e}')
            ax.hist(r_sub, bins=bins, color='tab:blue', alpha=0.75,
                    edgecolor='none',
                    label=f'filtered (n={len(r_sub)})  p={p_sub:.1e}')
            ax.axvline(0, color='0.4', lw=0.6)
            ax.axvline(float(np.mean(r_full)), color='0.4',
                       lw=0.8, ls='--')
            if r_sub.size:
                ax.axvline(float(np.mean(r_sub)), color='tab:blue', lw=1.2)

            ax.set_title(f'{roi} – {focal}', fontsize=10)
            ax.set_xlabel('mean_r', fontsize=8)
            ax.set_ylabel('# neurons', fontsize=8)
            ax.legend(fontsize=7, frameon=False, loc='upper left')

    fig.suptitle(
        f'mean_r distribution for focal models, before vs after filter | '
        f'paired alpha={ALPHA_PAIRED}\n'
        f'exclude when motor ({", ".join(MOTOR_MODELS)}) or rival '
        f'(state↔dsr) paired-beats focal',
        fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {save_path}")


plot_focal_r_distributions(
    os.path.join(OUT_DIR, 'fig4_focal_r_distribution.png'))


# ── Fig 5: which / how-many other models co-encode (perm-sig) ─────────
def plot_focal_overlap(save_path):
    rows = len(TARGET_ROIS) * len(FOCAL_MODELS)
    fig, axes = plt.subplots(rows, 2, figsize=(13, 3.4 * rows),
                             squeeze=False)
    row = 0
    for roi in TARGET_ROIS:
        df_roi = df[df['roi'] == roi]
        for focal in FOCAL_MODELS:
            ax_n, ax_m = axes[row, 0], axes[row, 1]

            f_sig = list(df_roi.loc[
                (df_roi['model'] == focal)
                & (df_roi['p_perm'] < ALPHA_PERM), 'neuron'].unique())
            if not f_sig:
                ax_n.set_title(f'{roi} – {focal}: 0 perm-sig neurons')
                ax_m.set_title(f'{roi} – {focal}: 0 perm-sig neurons')
                row += 1
                continue

            other_models = [m for m in MODELS if m != focal]
            co_counts    = pd.Series(0, index=other_models, dtype=int)
            n_other      = []
            for n in f_sig:
                g = df_roi[df_roi['neuron'] == n]
                sig = g[(g['model'] != focal)
                        & (g['p_perm'] < ALPHA_PERM)]['model'].tolist()
                n_other.append(len(sig))
                for m in sig:
                    if m in co_counts.index:
                        co_counts[m] += 1

            n_other = np.array(n_other)
            bins = np.arange(-0.5, n_other.max() + 1.5)
            ax_n.hist(n_other, bins=bins, color='tab:blue',
                      edgecolor='white')
            ax_n.set_xlabel('# other models also perm-sig', fontsize=9)
            ax_n.set_ylabel('# neurons', fontsize=9)
            ax_n.set_xticks(np.arange(n_other.max() + 1))
            ax_n.set_title(f'{roi} – {focal}: n={len(f_sig)} perm-sig '
                           f'neurons', fontsize=10)

            related = set(RELATED.get(focal, []))
            colors  = ['tab:green' if m in related else '0.5'
                       for m in co_counts.index]
            x = np.arange(len(co_counts))
            ax_m.bar(x, co_counts.values, color=colors, edgecolor='none')
            ax_m.set_xticks(x)
            ax_m.set_xticklabels(co_counts.index, rotation=45,
                                 ha='right', fontsize=8)
            ax_m.set_ylabel('# co-encoding neurons', fontsize=9)
            ax_m.set_title(f'{roi} – {focal}: which other models '
                           f'(green = related to {focal})', fontsize=10)
            for xi, v in zip(x, co_counts.values):
                if v > 0:
                    ax_m.text(xi, v, str(int(v)), ha='center',
                              va='bottom', fontsize=7)
            row += 1

    fig.suptitle(f'Co-encoding for perm-sig state / dsr neurons '
                 f'(perm alpha={ALPHA_PERM})', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {save_path}")


plot_focal_overlap(os.path.join(OUT_DIR, 'fig5_focal_overlap.png'))


# ── Fig 6: dsr distribution across all ROIs under different filters ──
DSR_FILTER_VARIANTS = {
    'motor':              list(MOTOR_MODELS),
    'location':           ['location'],
    'task_rels':          ['dsr_now_next', 'midnight', 'location'],
    'motor + task_rels':  list(MOTOR_MODELS) + ['dsr_now_next', 'midnight',
                                                'location'],
    'motor + state':      list(MOTOR_MODELS) + ['state'],
    'phase':              ['phase'],
}


def neurons_passing_dsr_filter(df_roi, drops):
    """Set of neurons where none of `drops` paired-beats dsr."""
    kept = set()
    for n, g in df_roi.groupby('neuron', sort=False):
        gi = g.set_index('model')
        if 'dsr' not in gi.index:
            continue
        r_focal = gi.loc['dsr', 'r_per_fold']
        beaten = False
        for m in drops:
            if m not in gi.index:
                continue
            p = paired_p_greater(gi.loc[m, 'r_per_fold'], r_focal)
            if not np.isnan(p) and p < ALPHA_PAIRED:
                beaten = True
                break
        if not beaten:
            kept.add(n)
    return kept


def _ttest_one_sided_greater(vals):
    """One-sided one-sample t-test, H1: mean > 0. NaN if too few samples."""
    vals = np.asarray(vals, dtype=float)
    if vals.size < 2 or np.all(vals == vals[0]):
        return np.nan
    try:
        return float(stats.ttest_1samp(
            vals, 0.0, alternative='greater').pvalue)
    except TypeError:                          # scipy < 1.6
        t, p2 = stats.ttest_1samp(vals, 0.0)
        return float(p2 / 2 if t > 0 else 1 - p2 / 2)


def plot_dsr_filter_grid(save_path):
    variants = list(DSR_FILTER_VARIANTS.items())
    n_var    = len(variants)
    n_roi    = len(ALL_ROIS)

    # Two columns per ROI: histogram (wide), proportion-excluded bar (narrow)
    fig, axes = plt.subplots(
        n_var, n_roi * 2,
        figsize=(2.8 * n_roi, 2.1 * n_var),
        squeeze=False,
        gridspec_kw={'width_ratios': [4, 0.5] * n_roi})

    # share y across the histogram columns within each row
    for i in range(n_var):
        first_hist = axes[i, 0]
        for j in range(1, n_roi):
            axes[i, 2 * j].sharey(first_hist)

    for i, (label, drops) in enumerate(variants):
        for j, roi in enumerate(ALL_ROIS):
            ax_hist = axes[i, 2 * j]
            ax_bar  = axes[i, 2 * j + 1]

            df_roi = df_all[df_all['roi'] == roi]
            full   = df_roi[df_roi['model'] == 'dsr']
            r_full = full['mean_r'].dropna().values

            if r_full.size < 2:
                ax_hist.text(0.5, 0.5, 'n<2', transform=ax_hist.transAxes,
                             ha='center', va='center',
                             fontsize=8, color='0.5')
                ax_hist.set_xticks([]); ax_hist.set_yticks([])
                ax_bar.axis('off')
                if i == 0:
                    ax_hist.annotate(roi, xy=(0.5, 1.45),
                                     xycoords='axes fraction',
                                     ha='center', va='bottom',
                                     fontsize=10, fontweight='bold')
                if j == 0:
                    ax_hist.set_ylabel(label, fontsize=8)
                continue

            kept_ids = neurons_passing_dsr_filter(df_roi, drops)
            sub      = full[full['neuron'].isin(kept_ids)]
            excl     = full[~full['neuron'].isin(kept_ids)]
            r_sub    = sub['mean_r'].dropna().values
            r_excl   = excl['mean_r'].dropna().values

            mean_full = float(np.mean(r_full))
            mean_sub  = float(np.mean(r_sub)) if r_sub.size else np.nan

            p_sub_vs0 = _ttest_one_sided_greater(r_sub)
            if r_sub.size >= 2 and r_excl.size >= 2:
                p_shift = float(stats.mannwhitneyu(
                    r_sub, r_excl, alternative='two-sided').pvalue)
            else:
                p_shift = np.nan

            lim  = max(0.05, 1.05 * float(np.max(np.abs(r_full))))
            bins = np.linspace(-lim, lim, 21)

            ax_hist.hist(r_full, bins=bins, color='0.55', alpha=0.35,
                         edgecolor='none')
            ax_hist.hist(r_sub, bins=bins, color='tab:blue', alpha=0.75,
                         edgecolor='none')
            ax_hist.axvline(0,         color='0.4',      lw=0.7, ls=':')
            ax_hist.axvline(mean_full, color='0.35',     lw=1.0)
            if r_sub.size:
                ax_hist.axvline(mean_sub, color='tab:blue', lw=1.4)

            ax_hist.set_title(
                f"n={len(r_sub)}/{len(r_full)}   "
                f"p(>0)={p_sub_vs0:.1e}\n"
                f"Δmean p={p_shift:.1e}",
                fontsize=7)
            ax_hist.tick_params(labelsize=6)

            if i == 0:
                ax_hist.annotate(roi, xy=(0.5, 1.45),
                                 xycoords='axes fraction',
                                 ha='center', va='bottom',
                                 fontsize=10, fontweight='bold')
            if j == 0:
                ax_hist.set_ylabel(label, fontsize=8)

            # proportion-excluded bar (blue = kept, grey = excluded)
            n_total = len(full)
            p_kept  = len(sub) / n_total if n_total else 0.0
            ax_bar.bar(0, p_kept, color='tab:blue', width=0.8)
            ax_bar.bar(0, 1 - p_kept, bottom=p_kept, color='0.65', width=0.8)
            ax_bar.set_ylim(0, 1)
            ax_bar.set_xlim(-0.6, 0.6)
            ax_bar.set_xticks([])
            ax_bar.tick_params(labelsize=6)
            ax_bar.text(0, 1.02, f"{int(round((1 - p_kept) * 100))}% excl",
                        ha='center', va='bottom', fontsize=6, color='0.35')
            if j == 0:
                ax_bar.set_ylabel('frac', fontsize=6)

    fig.suptitle(
        'dsr mean_r distribution per ROI under different exclusion filters '
        f'(paired alpha={ALPHA_PAIRED})\n'
        'grey hist = all neurons (grey line = mean); '
        'blue hist = kept (blue line = mean); dotted grey = 0.   '
        'p(>0): one-sided t-test of kept-mean > 0.   '
        'Δmean p: Mann-Whitney kept vs excluded.   '
        'right bar = fraction kept (blue) / excluded (grey).',
        fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {save_path}")


plot_dsr_filter_grid(os.path.join(OUT_DIR, 'fig6_dsr_filter_grid.png'))


# ── Fig 7: dsr co-encoding stratified by # of other perm-sig models ──
def plot_dsr_co_encoding_by_count(save_path, rois):
    other_models = [m for m in MODELS if m != 'dsr']
    bin_labels   = ['0', '1', '2', '3', '4', '5', '6+']
    n_bins       = len(bin_labels)

    fig, axes = plt.subplots(
        len(rois), 1,
        figsize=(1.0 * len(other_models) + 3.5,
                 0.5 * n_bins * len(rois) + 1.5),
        squeeze=False)

    for i, roi in enumerate(rois):
        ax     = axes[i, 0]
        df_roi = df_all[df_all['roi'] == roi]

        f_sig = list(df_roi.loc[
            (df_roi['model'] == 'dsr')
            & (df_roi['p_perm'] < ALPHA_PERM), 'neuron'].unique())

        count_mat  = np.zeros((n_bins, len(other_models)), dtype=int)
        bin_totals = np.zeros(n_bins, dtype=int)

        for n in f_sig:
            g   = df_roi[df_roi['neuron'] == n]
            sig = g[(g['model'] != 'dsr')
                    & (g['p_perm'] < ALPHA_PERM)]['model'].tolist()
            k       = len(sig)
            bin_idx = min(k, n_bins - 1)            # 6+ catches >= 6
            bin_totals[bin_idx] += 1
            for m in sig:
                if m in other_models:
                    count_mat[bin_idx, other_models.index(m)] += 1

        prop_mat = count_mat.astype(float)
        for r in range(n_bins):
            if bin_totals[r] > 0:
                prop_mat[r] /= bin_totals[r]

        im = ax.imshow(prop_mat, aspect='auto', cmap='Blues',
                       vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks(range(len(other_models)))
        ax.set_xticklabels(other_models, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_bins))
        ax.set_yticklabels(
            [f'{lbl}   (n={t})' for lbl, t in zip(bin_labels, bin_totals)],
            fontsize=8)
        ax.set_ylabel('# other models perm-sig', fontsize=9)
        ax.set_title(f'{roi}: dsr perm-sig neurons (total n={len(f_sig)})  |  '
                     'colour = fraction within bin, label = raw count',
                     fontsize=10)

        for r in range(n_bins):
            for c in range(len(other_models)):
                v = count_mat[r, c]
                if v > 0:
                    color = 'white' if prop_mat[r, c] > 0.55 else 'black'
                    ax.text(c, r, str(v), ha='center', va='center',
                            fontsize=7, color=color)

        fig.colorbar(im, ax=ax, label='fraction within bin')

    fig.suptitle('dsr perm-sig neurons: which other models are co-encoded, '
                 'stratified by # co-encoded models', fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {save_path}")


plot_dsr_co_encoding_by_count(
    os.path.join(OUT_DIR, 'fig7_dsr_co_encoding_by_count.png'),
    rois=TARGET_ROIS)


# ── Fig 8: DSR encoding after removing state-encoding neurons ────────
# For ACC / HC_anterior / HC_posterior: take every DSR-variant fit and
# show how its mean_r distribution and significant-neuron fraction change
# when neurons that perm-encode state / state_phase are dropped.
def _state_encoders(df_roi, state_model):
    return set(df_roi.loc[
        (df_roi['model'] == state_model)
        & (df_roi['p_perm'] < ALPHA_PERM), 'neuron'].unique())


def plot_dsr_after_state_removal(save_path):
    dsr_present   = [m for m in DSR_VARIANTS if m in MODELS]
    state_present = [m for m in STATE_VARIANTS if m in MODELS]
    if not dsr_present:
        print("  [fig8] no DSR variants present — skipping.")
        return

    filters_spec = [('all', [])]
    for sm in state_present:
        filters_spec.append((f'minus {sm}-enc', [sm]))
    if len(state_present) > 1:
        filters_spec.append(('minus state+state_phase-enc',
                             list(state_present)))
    filter_colors = {
        'all':                         '0.55',
        'minus state-enc':             'tab:blue',
        'minus state_phase-enc':       'tab:green',
        'minus state+state_phase-enc': 'tab:red',
    }

    n_rows = len(TARGET_ROIS)
    n_cols = len(dsr_present)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for i, roi in enumerate(TARGET_ROIS):
        df_roi = df_all[df_all['roi'] == roi]
        if df_roi.empty:
            for j in range(n_cols):
                axes[i, j].set_title(f'{roi}: no data')
                axes[i, j].axis('off')
            continue

        # State-encoder sets in this ROI.
        enc_sets = {sm: _state_encoders(df_roi, sm) for sm in state_present}

        for j, dsr_m in enumerate(dsr_present):
            ax = axes[i, j]
            dsr_rows = df_roi[df_roi['model'] == dsr_m]
            if dsr_rows.empty:
                ax.set_title(f'{roi} – {dsr_m}: no data', fontsize=9)
                ax.axis('off')
                continue
            r_full = dsr_rows['mean_r'].dropna().values
            lim    = max(0.05, 1.05 * float(np.max(np.abs(r_full))))
            bins   = np.linspace(-lim, lim, 25)

            legend_lines = []
            for fname, excl_models in filters_spec:
                excl = set()
                for sm in excl_models:
                    excl |= enc_sets.get(sm, set())
                kept = dsr_rows[~dsr_rows['neuron'].isin(excl)]
                r_k  = kept['mean_r'].dropna().values
                n_k  = len(r_k)
                n_sig_k = int((kept['p_perm'] < ALPHA_PERM).sum())
                frac_sig = (n_sig_k / n_k) if n_k else 0.0
                if n_k >= 2:
                    p_gt0 = float(stats.ttest_1samp(
                        r_k, 0.0, alternative='greater').pvalue)
                else:
                    p_gt0 = np.nan

                color = filter_colors.get(fname, '0.4')
                ax.hist(
                    r_k, bins=bins,
                    color=color,
                    alpha=0.30 if fname == 'all' else 0.55,
                    edgecolor='none',
                    label=(f'{fname} (n={n_k}, '
                           f'%sig={100 * frac_sig:.0f}, p>0={p_gt0:.1e})'),
                )
                if n_k:
                    ax.axvline(float(np.mean(r_k)),
                               color=color, lw=1.2,
                               ls='-' if fname == 'all' else '--')
                legend_lines.append(fname)

            ax.axvline(0, color='0.4', lw=0.5, ls=':')
            ax.set_xlim(-lim, lim)
            ax.set_title(f'{roi} – {dsr_m}', fontsize=10)
            ax.set_xlabel('mean_r', fontsize=8)
            if j == 0:
                ax.set_ylabel('# neurons', fontsize=8)
            ax.legend(fontsize=6, frameon=False, loc='upper left')

    fig.suptitle(
        'DSR encoding before and after removing state-encoding neurons '
        f'(perm alpha={ALPHA_PERM}, one-sided t-test of mean>0)',
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {save_path}")


plot_dsr_after_state_removal(
    os.path.join(OUT_DIR, 'fig8_dsr_after_state_removal.png'))


# ── Fig 9: significant-fraction bar chart (ACC + HC), each DSR variant ─
# Companion to fig 8: a more direct read-out of "how much DSR is still
# encoded?" — fraction of neurons with p_perm(DSR variant) < ALPHA_PERM
# under each filter.
def plot_dsr_sig_fraction_bars(save_path):
    dsr_present   = [m for m in DSR_VARIANTS if m in MODELS]
    state_present = [m for m in STATE_VARIANTS if m in MODELS]
    if not dsr_present:
        print("  [fig9] no DSR variants present — skipping.")
        return

    filters_spec = [('all', [])]
    for sm in state_present:
        filters_spec.append((f'minus {sm}-enc', [sm]))
    if len(state_present) > 1:
        filters_spec.append(('minus state+state_phase-enc',
                             list(state_present)))
    f_names  = [name for name, _ in filters_spec]
    f_colors = {'all': '0.55',
                'minus state-enc': 'tab:blue',
                'minus state_phase-enc': 'tab:green',
                'minus state+state_phase-enc': 'tab:red'}

    n_rows = len(TARGET_ROIS)
    n_cols = len(dsr_present)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.6 * n_cols + 1, 2.4 * n_rows + 1),
        squeeze=False,
    )

    for i, roi in enumerate(TARGET_ROIS):
        df_roi = df_all[df_all['roi'] == roi]
        if df_roi.empty:
            for j in range(n_cols):
                axes[i, j].axis('off')
            continue
        enc_sets = {sm: _state_encoders(df_roi, sm) for sm in state_present}

        for j, dsr_m in enumerate(dsr_present):
            ax = axes[i, j]
            dsr_rows = df_roi[df_roi['model'] == dsr_m]
            if dsr_rows.empty:
                ax.set_title(f'{roi} – {dsr_m}: no data', fontsize=8)
                ax.axis('off')
                continue

            fracs  = []
            counts = []
            for fname, excl_models in filters_spec:
                excl = set()
                for sm in excl_models:
                    excl |= enc_sets.get(sm, set())
                kept = dsr_rows[~dsr_rows['neuron'].isin(excl)]
                n_k  = len(kept)
                n_sig_k = int((kept['p_perm'] < ALPHA_PERM).sum())
                fracs.append((n_sig_k / n_k) if n_k else 0.0)
                counts.append((n_sig_k, n_k))

            x_pos = np.arange(len(f_names))
            bars = ax.bar(x_pos, fracs,
                          color=[f_colors.get(n, '0.4') for n in f_names],
                          edgecolor='none')
            for b, (ns, nt) in zip(bars, counts):
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.01,
                        f'{ns}/{nt}',
                        ha='center', va='bottom', fontsize=7)
            ax.axhline(ALPHA_PERM, color='0.4', lw=0.6, ls=':',
                       label=f'chance ({ALPHA_PERM:.0%})')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(f_names, rotation=30, ha='right', fontsize=7)
            ax.set_ylabel(f'frac p_perm<{ALPHA_PERM}', fontsize=8)
            ax.set_title(f'{roi} – {dsr_m}', fontsize=9)
            ax.set_ylim(0, max(0.25, 1.15 * max(fracs) if fracs else 0.25))
            ax.tick_params(labelsize=7)

    fig.suptitle(
        'Fraction of perm-significant DSR neurons before/after removing '
        f'state-encoders (perm alpha={ALPHA_PERM})',
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {save_path}")


plot_dsr_sig_fraction_bars(
    os.path.join(OUT_DIR, 'fig9_dsr_sig_fraction_after_state_removal.png'))


print(f"\nAll outputs in: {OUT_DIR}")
