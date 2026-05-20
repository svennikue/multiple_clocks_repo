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
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# ── Settings ──────────────────────────────────────────────────────────

RESULT_DIR = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/encoding_analysis_simple/2026-05-19_17-26-12/')

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


# Only include DSR-encoding neurons (across-fold permutation p_perm < this).
DSR_COEF_P_PERM_THRESHOLD = 0.05

# ── Which analysis sections to run ────────────────────────────────────
# Pick any subset; sections are independent except that all CSV-based
# sections share the encoding_results.csv load.
#   'classification'  -> per-neuron model assignment CSV + figs 1-2
#   'focal'           -> focal-model filter + figs 4-5
#   'dsr_filter'      -> fig 6  (dsr distribution under exclusion filters)
#   'dsr_co_encoding' -> fig 7  (dsr co-encoding stratified by count)
#   'state_removal'   -> figs 8-9 (dsr after removing state encoders)
#   'coefficients'    -> figs 10-11 (DSR coefficient analysis from
#                        diagnostics.pkl — independent of the CSV)
SECTIONS_TO_RUN = ['coefficients']

_run_classification  = 'classification'  in SECTIONS_TO_RUN
_run_focal           = 'focal'           in SECTIONS_TO_RUN
_run_dsr_filter      = 'dsr_filter'      in SECTIONS_TO_RUN
_run_dsr_co_encoding = 'dsr_co_encoding' in SECTIONS_TO_RUN
_run_state_removal   = 'state_removal'   in SECTIONS_TO_RUN
_run_coefficients    = 'coefficients'    in SECTIONS_TO_RUN
_need_csv = any([_run_classification, _run_focal, _run_dsr_filter,
                 _run_dsr_co_encoding, _run_state_removal])
print(f"Sections to run: {SECTIONS_TO_RUN}")

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


if _need_csv:
    print(f"Loading {RESULT_CSV}")
    df_all = pd.read_csv(RESULT_CSV)
    n_before = len(df_all)
    df_all = df_all.dropna(subset=['mean_r']).copy()
    df_all['r_per_fold'] = df_all['r_per_fold'].map(parse_folds)
    print(f"  dropped {n_before - len(df_all)} rows with NaN mean_r "
          "(all-zero-coef / unfittable neuron x model).")

    # `df`     -> the target ROIs (used by figs 1, 2, 4, 5)
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


if _run_classification:
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


if _run_classification:
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


if _run_classification:
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


if _run_focal:
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


if _run_focal:
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


if _run_focal:
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


if _run_dsr_filter:
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


if _run_dsr_co_encoding:
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


if _run_state_removal:
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


if _run_state_removal:
    plot_dsr_sig_fraction_bars(
        os.path.join(OUT_DIR, 'fig9_dsr_sig_fraction_after_state_removal.png'))


# ── Fig 10: distribution of DSR coefficients per ROI ─────────────────
# Each DSR neuron has 324 coefficients arranged as
#   (location, phase, future-lag) = (9, 3, 12).
# Coefs are saved per held-out fold in diagnostics.pkl; for this plot we
# take the across-fold mean so each neuron contributes one (324,) vector.
# Then per ROI we look at the distribution of those values, collapsed over
# the two factors NOT on the x-axis.
DIAGNOSTICS_PKL = os.path.join(RESULT_DIR, 'diagnostics.pkl')
DSR_MODEL_NAME  = 'dsr'
DSR_N_LOCATIONS = 9
DSR_N_PHASES    = 3
DSR_N_LAGS      = 12          # 'future lag' index (== n_clocks_per_phase)
DSR_N_COEFS     = DSR_N_LOCATIONS * DSR_N_PHASES * DSR_N_LAGS  # 324
DSR_PHASE_NAMES = ('early', 'middle', 'late')



# Per-neuron coefficient normalisation, so neurons with different firing-rate
# scales contribute equally to the population summary.  Normalisation is
# applied PER FOLD, before averaging across folds, because different parts
# of a neuron can fire at different strengths from fold to fold.  Options:
#   'mean' — coef / mean(coef)   -> values are multiples of the neuron's own
#            average coefficient; a bin value of 1.5 means "1.5x the typical
#            coefficient for this neuron".  This is the interpretable scale
#            for "is some lag / phase / location over-represented".
#   'sum'  — coef / sum(coef)    (probability-distribution interpretation)
#   'l2'   — coef / ||coef||_2   (unit-norm)
#   'max'  — coef / max(coef)    (peak-relative)
#   None   — raw coefficients
DSR_COEF_NORMALIZE = 'mean'

# Reference level for the plots: the per-neuron grand mean across bins after
# normalisation.  For 'mean'-normalised coefficients this is exactly 1.0
# (above = over-represented, below = under-represented); otherwise 0.
DSR_COEF_REFERENCE = 1.0 if DSR_COEF_NORMALIZE == 'mean' else 0.0

# ROIs used as columns in the detailed histogram-grid plots.
DSR_COEF_HISTOGRAM_ROIS = ['ACC', 'HC_anterior', 'HC_posterior']


def _normalize_coefs(coefs, how):
    """Per-neuron (per-fold) normalisation; returns (coefs_norm, ok).

    `ok` is False when the normaliser cannot be computed (e.g. an
    all-zero coefficient vector), so the caller can drop that fold/neuron.
    """
    if how is None:
        return coefs, True
    coefs = np.asarray(coefs, dtype=float)
    if how == 'mean':
        s = float(coefs.mean())
        if abs(s) <= 1e-12:
            return coefs, False
        return coefs / s, True
    if how == 'sum':
        s = float(coefs.sum())
        if s <= 1e-12:
            return coefs, False
        return coefs / s, True
    if how == 'l2':
        s = float(np.sqrt((coefs ** 2).sum()))
        if s <= 1e-12:
            return coefs, False
        return coefs / s, True
    if how == 'max':
        s = float(np.max(np.abs(coefs)))
        if s <= 1e-12:
            return coefs, False
        return coefs / s, True
    raise ValueError(f"Unknown normalisation: {how!r}")


def load_dsr_coefs(diagnostics,
                   p_perm_threshold=DSR_COEF_P_PERM_THRESHOLD,
                   normalize=DSR_COEF_NORMALIZE,
                   model_name=DSR_MODEL_NAME,
                   expected_size=DSR_N_COEFS):
    """Per-neuron DSR coefficients (per-fold-normalised, then averaged).

    Each held-out fold's 324-coefficient vector is normalised on its own
    (see `_normalize_coefs`) BEFORE averaging across folds, so a fold in
    which the neuron fired more strongly does not dominate the mean.

    Parameters
    ----------
    p_perm_threshold : float or None
        Only keep neurons whose `p_perm` for this model is < threshold.
        ``None`` keeps every neuron.
    normalize : 'mean' | 'sum' | 'l2' | 'max' | None
        Per-fold normalisation so the resulting coefficients are
        comparable across neurons with very different firing-rate scales.

    Returns DataFrame with columns:
        subject, neuron, roi, p_perm, mean_r, coefs (size 324, normalised).
    """
    rows, skipped = [], {
        'missing': 0, 'wrong_shape': 0,
        'p_perm': 0, 'zero_coefs': 0,
    }
    for sub, per_neuron in diagnostics.items():
        for n_lab, per_model in per_neuron.items():
            if model_name not in per_model:
                skipped['missing'] += 1
                continue
            d = per_model[model_name]

            # p_perm filter (only DSR-encoding neurons).
            if p_perm_threshold is not None:
                p = d.get('p_perm', np.nan)
                if not np.isfinite(p) or p >= p_perm_threshold:
                    skipped['p_perm'] += 1
                    continue

            coefs_list = d.get('coefs', [])
            if not coefs_list:
                skipped['missing'] += 1
                continue
            coefs_arr = np.array(
                [np.asarray(c, dtype=float) for c in coefs_list]
            )
            if coefs_arr.ndim != 2 or coefs_arr.shape[1] != expected_size:
                skipped['wrong_shape'] += 1
                continue

            # Normalise EACH fold first, then average the normalised folds.
            # Degenerate folds (all-zero coefficients) are dropped from the
            # average rather than pulling it toward zero.
            normed_folds = []
            for fold_coefs in coefs_arr:
                nc, ok = _normalize_coefs(fold_coefs, normalize)
                if ok:
                    normed_folds.append(nc)
            if not normed_folds:
                skipped['zero_coefs'] += 1
                continue
            mean_coefs = np.mean(normed_folds, axis=0)            # (324,)

            rows.append({
                'subject': sub,
                'neuron':  n_lab,
                'roi':     d.get('roi'),
                'p_perm':  d.get('p_perm'),
                'mean_r':  d.get('mean_r'),
                'coefs':   mean_coefs,
            })

    print(f"  load_dsr_coefs: kept {len(rows)}; "
          f"filtered: p_perm={skipped['p_perm']}, "
          f"zero-coef={skipped['zero_coefs']}, "
          f"missing={skipped['missing']}, "
          f"wrong-shape={skipped['wrong_shape']}")
    return pd.DataFrame(rows)


def reshape_dsr(coefs):
    """(324,) → (9, 3, 12) = (location, phase, lag)."""
    return np.asarray(coefs, dtype=float).reshape(
        DSR_N_LOCATIONS, DSR_N_PHASES, DSR_N_LAGS)


def per_neuron_by_factor(coefs_df, factor):
    """(n_neurons, n_bins) — mean coefficient strength per neuron per bin,
    collapsed over the two factors not equal to `factor`."""
    arrs = np.stack([reshape_dsr(c) for c in coefs_df['coefs']])  # (N,9,3,12)
    if factor == 'lag':       return arrs.mean(axis=(1, 2))       # (N, 12)
    if factor == 'location':  return arrs.mean(axis=(2, 3))       # (N, 9)
    if factor == 'phase':     return arrs.mean(axis=(1, 3))       # (N, 3)
    raise ValueError(f"Unknown factor: {factor!r}")


def _friedman_test_across_bins(vals):
    """Friedman omnibus: do bins differ within-neuron?  Returns p-value."""
    n_neu, n_bins = vals.shape
    if n_neu < 2 or n_bins < 2:
        return np.nan
    cols = [vals[:, b] for b in range(n_bins)]
    try:
        return float(stats.friedmanchisquare(*cols).pvalue)
    except (ValueError, TypeError):
        return np.nan


def _per_bin_vs_grand_mean(vals):
    """Paired Wilcoxon per bin: val[bin] vs the neuron's mean across bins.

    Returns (p_corr, signs) — Bonferroni-corrected two-sided p per bin and
    sign of the median deviation from the grand mean (+1 above, -1 below).
    """
    n_neu, n_bins = vals.shape
    p_vals = np.full(n_bins, np.nan, dtype=float)
    signs  = np.zeros(n_bins, dtype=int)
    if n_neu < 2:
        return p_vals, signs
    grand_mean = vals.mean(axis=1, keepdims=True)
    centred = vals - grand_mean
    for b in range(n_bins):
        d = centred[:, b]
        if np.all(d == 0):
            continue
        try:
            res = stats.wilcoxon(d, alternative='two-sided',
                                 zero_method='wilcox')
            p_vals[b] = float(res.pvalue)
            signs[b]  = 1 if np.nanmedian(d) > 0 else -1
        except (ValueError, TypeError):
            pass
    return np.minimum(1.0, p_vals * n_bins), signs


def _print_bin_statistics(coefs_df, factor, rois, label, bin_labels):
    """Console summary of per-bin statistics, per ROI."""
    print(f"\n  --- DSR coef stats by {label} "
          f"(per-neuron norm={DSR_COEF_NORMALIZE!r}, "
          f"Bonferroni-corrected vs grand mean) ---")
    for roi in rois:
        sub = coefs_df[coefs_df['roi'] == roi]
        if sub.empty:
            continue
        vals = per_neuron_by_factor(sub, factor=factor)
        n = len(vals)
        omni_p = _friedman_test_across_bins(vals)
        p_corr, signs = _per_bin_vs_grand_mean(vals)
        means = vals.mean(axis=0)
        sems = vals.std(axis=0) / max(1, np.sqrt(n))
        omni_str = (f"{omni_p:.3e}" if np.isfinite(omni_p) else 'nan')
        print(f"    [{roi:<14s}] n={n:3d}  Friedman p={omni_str}")
        for b, lbl in enumerate(bin_labels):
            sig = ('***' if p_corr[b] < 0.001
                   else '**' if p_corr[b] < 0.01
                   else '*' if p_corr[b] < 0.05
                   else 'n.s.')
            direction = '↑' if signs[b] > 0 else '↓' if signs[b] < 0 else '–'
            p_str = (f"{p_corr[b]:.3e}" if np.isfinite(p_corr[b]) else 'nan')
            print(f"      {lbl:>10s}: mean={means[b]:+.4f} ± {sems[b]:.4f}  "
                  f"{direction}  p_corr={p_str}  {sig}")


# ── Non-parametric 3-way analysis of the DSR coefficient structure ─────
# Treats each neuron's (9 location × 3 phase × 12 lag) normalised
# coefficient array as a repeated-measures design (neuron = subject).
#   main effect of factor F  -> Friedman across F's marginal means
#   interaction A×B          -> Friedman across the flattened interaction
#                               residuals (cell minus the two main effects)
# Friedman is non-parametric (robust to the sphericity violations a
# parametric RM-ANOVA would suffer with 12 lag levels).  For interactions
# the residuals carry row/column sum-to-zero constraints, so using df=k-1
# makes the test mildly conservative — acceptable for an omnibus screen.

INTERACTION_COLLAPSE_AXIS = {
    'location x phase': 3,   # collapse lag    -> (N, 9, 3)
    'location x lag':   2,   # collapse phase  -> (N, 9, 12)
    'phase x lag':      1,   # collapse loc    -> (N, 3, 12)
}
ANOVA_EFFECTS = ['location', 'phase', 'lag',
                 'location x phase', 'location x lag', 'phase x lag']


def _friedman_full(vals):
    """Friedman test on (n_subjects, n_conditions); returns a dict with
    chi2, dof, p, kendall_w (= chi2 / (n*(k-1)), an effect size in [0, 1])."""
    n, k = vals.shape
    out = dict(chi2=np.nan, dof=k - 1, p=np.nan, kendall_w=np.nan,
               n=int(n), k=int(k))
    if n < 2 or k < 2:
        return out
    try:
        res = stats.friedmanchisquare(*[vals[:, j] for j in range(k)])
    except (ValueError, TypeError):
        return out
    out['chi2'] = float(res.statistic)
    out['p'] = float(res.pvalue)
    denom = n * (k - 1)
    out['kendall_w'] = float(res.statistic / denom) if denom > 0 else np.nan
    return out


def _interaction_residuals(table):
    """Two-way interaction residuals per subject.

    `table` : (n_subjects, a, b).  Returns the same shape, with each cell
    replaced by  cell - row_mean - col_mean + grand_mean  (per subject).
    """
    row   = table.mean(axis=2, keepdims=True)
    col   = table.mean(axis=1, keepdims=True)
    grand = table.mean(axis=(1, 2), keepdims=True)
    return table - row - col + grand


def _benjamini_hochberg(pvals):
    """BH-FDR adjusted p-values; NaN entries pass through as NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    mask = np.isfinite(p)
    pv = p[mask]
    m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv)
    adj = pv[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    res = np.empty(m)
    res[order] = adj
    out[mask] = res
    return out


def dsr_coef_anova(coefs_df, rois=None):
    """Per-ROI non-parametric 3-way analysis of the DSR coefficients.

    Returns a long-format DataFrame, one row per (roi, effect), with:
        n_neurons, friedman_chi2, dof, p_raw, kendall_w, p_fdr
    p_fdr is Benjamini-Hochberg corrected across all (roi × effect) tests.
    """
    if rois is None:
        rois = sorted(coefs_df['roi'].dropna().unique().tolist())
    rows = []
    for roi in rois:
        sub = coefs_df[coefs_df['roi'] == roi]
        if sub.empty:
            continue
        A = np.stack([reshape_dsr(c) for c in sub['coefs']])  # (N,9,3,12)
        n = A.shape[0]

        eff_vals = {
            'location': A.mean(axis=(2, 3)),   # (N, 9)
            'phase':    A.mean(axis=(1, 3)),   # (N, 3)
            'lag':      A.mean(axis=(1, 2)),   # (N, 12)
        }
        for name, collapse_axis in INTERACTION_COLLAPSE_AXIS.items():
            table = A.mean(axis=collapse_axis)               # (N, d1, d2)
            eff_vals[name] = _interaction_residuals(table).reshape(n, -1)

        for eff in ANOVA_EFFECTS:
            st = _friedman_full(eff_vals[eff])
            rows.append({
                'roi':            roi,
                'effect':         eff,
                'n_neurons':      st['n'],
                'friedman_chi2':  st['chi2'],
                'dof':            st['dof'],
                'p_raw':          st['p'],
                'kendall_w':      st['kendall_w'],
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df['p_fdr'] = _benjamini_hochberg(df['p_raw'].to_numpy())
    return df


def plot_dsr_coef_distribution_by_factor(coefs_df, factor, save_path,
                                         rois=None, n_cols=4):
    """Per-ROI distribution of DSR coefficient strength binned by `factor`.

    Shows boxplots overlaid with per-neuron jittered points so the full
    distribution is visible.  Each panel title reports the Friedman
    omnibus p-value; per-bin stars are Bonferroni-corrected paired
    Wilcoxon tests of bin vs the neuron's own grand mean (↑/↓ shows
    direction).  The per-neuron coefficients are normalised with
    DSR_COEF_NORMALIZE = {DSR_COEF_NORMALIZE!r} so neurons with different
    firing-rate scales contribute equally.
    """
    factor_meta = {
        'lag':      ('future lag', [str(i) for i in range(DSR_N_LAGS)]),
        'location': ('location',
                     [str(i + 1) for i in range(DSR_N_LOCATIONS)]),
        'phase':    ('phase', list(DSR_PHASE_NAMES)),
    }
    label, bin_labels = factor_meta[factor]

    if rois is None:
        rois = sorted(coefs_df['roi'].dropna().unique().tolist())
    if not rois:
        print(f"  no ROIs to plot for {factor}; skipping.")
        return

    n = len(rois)
    ncols = min(n_cols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.6, nrows * 2.9),
        squeeze=False,
    )

    # Shared y-limits across panels for fair comparison.
    all_vals = []
    for roi in rois:
        sub = coefs_df[coefs_df['roi'] == roi]
        if not sub.empty:
            all_vals.append(per_neuron_by_factor(sub, factor=factor).ravel())
    if all_vals:
        pooled = np.concatenate(all_vals)
        y_max = 1.10 * float(np.nanpercentile(pooled, 99))
        y_min = min(0.0, float(np.nanmin(pooled)))
    else:
        y_min, y_max = 0.0, 1.0

    rng = np.random.default_rng(0)
    for ax_idx, roi in enumerate(rois):
        ax = axes[ax_idx // ncols, ax_idx % ncols]
        sub_df = coefs_df[coefs_df['roi'] == roi]
        n_neu = len(sub_df)
        if n_neu == 0:
            ax.set_title(f'{roi}: no neurons', fontsize=9)
            ax.axis('off')
            continue

        vals = per_neuron_by_factor(sub_df, factor=factor)  # (n_neu, n_bins)
        positions = np.arange(len(bin_labels))

        # Boxplot (distribution per bin across neurons).
        ax.boxplot(vals, positions=positions, showfliers=False, widths=0.55,
                   medianprops={'color': 'tab:blue'},
                   whiskerprops={'color': '0.4'},
                   capprops={'color': '0.4'},
                   boxprops={'color': '0.4'})

        # Jittered raw points so the full distribution is visible.
        for b in range(len(bin_labels)):
            xs = positions[b] + (rng.uniform(-0.18, 0.18, size=n_neu))
            ax.scatter(xs, vals[:, b], s=6, color='tab:blue',
                       alpha=0.4, edgecolor='none')

        # Mean line on top.
        ax.plot(positions, np.nanmean(vals, axis=0),
                color='tab:red', lw=1.4, marker='o', ms=4, zorder=5)

        # Statistics.
        omni_p = _friedman_test_across_bins(vals)
        p_corr, signs = _per_bin_vs_grand_mean(vals)
        omni_str = (f"{omni_p:.2e}" if np.isfinite(omni_p) else 'nan')
        ax.set_title(f'{roi} (n={n_neu})  Friedman p={omni_str}',
                     fontsize=9)

        # Per-bin stars + direction arrow above the box.
        for b in range(len(bin_labels)):
            if not np.isfinite(p_corr[b]):
                continue
            if   p_corr[b] < 0.001: stars = '***'
            elif p_corr[b] < 0.01:  stars = '**'
            elif p_corr[b] < 0.05:  stars = '*'
            else:                   continue
            arrow = '↑' if signs[b] > 0 else '↓'
            ax.text(positions[b], y_max * 0.97, f'{arrow}{stars}',
                    ha='center', va='top', fontsize=7,
                    color='tab:red' if signs[b] > 0 else 'tab:purple',
                    fontweight='bold')

        # Reference line = the per-neuron average coefficient level
        # (1.0 for 'mean'-normalised coefs); bins above it are
        # over-represented, below it under-represented.
        ax.axhline(DSR_COEF_REFERENCE, color='0.5', lw=0.8, ls='--')
        ax.set_xticks(positions)
        ax.set_xticklabels(bin_labels, fontsize=7)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(label, fontsize=8)
        if ax_idx % ncols == 0:
            ax.set_ylabel(f'coef ({DSR_COEF_NORMALIZE}-normalised)',
                          fontsize=8)
        ax.tick_params(axis='y', labelsize=7)

    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')

    fig.suptitle(
        f'DSR coefficient strength by {label} '
        f'(per-neuron {DSR_COEF_NORMALIZE}-normalised, '
        f'p_perm<{DSR_COEF_P_PERM_THRESHOLD})  '
        f'— ↑/↓ Bonferroni-corrected vs grand mean: *p<.05 **p<.01 ***p<.001',
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {save_path}")

    _print_bin_statistics(coefs_df, factor, rois, label, bin_labels)


def plot_dsr_coef_overlay_across_rois(coefs_df, save_path, rois=None):
    """Mean ± SEM coefficient strength for every ROI overlaid; three
    panels (lag, location, phase) side by side."""
    if rois is None:
        rois = sorted(coefs_df['roi'].dropna().unique().tolist())
    if not rois:
        print("  no ROIs to plot for overlay; skipping.")
        return

    factors = [
        ('lag',      'future lag',
         [str(i) for i in range(DSR_N_LAGS)]),
        ('location', 'location',
         [str(i + 1) for i in range(DSR_N_LOCATIONS)]),
        ('phase',    'phase', list(DSR_PHASE_NAMES)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), squeeze=False)
    cmap = plt.get_cmap('tab10')

    for ax, (factor, xlabel, labels) in zip(axes[0], factors):
        for r_idx, roi in enumerate(rois):
            sub = coefs_df[coefs_df['roi'] == roi]
            if sub.empty:
                continue
            vals = per_neuron_by_factor(sub, factor=factor)   # (n, k)
            n = len(vals)
            m = np.nanmean(vals, axis=0)
            sem = np.nanstd(vals, axis=0) / max(1, np.sqrt(n))
            x = np.arange(len(labels))
            color = cmap(r_idx % 10)
            ax.plot(x, m, color=color, lw=1.4, marker='o', ms=4,
                    label=f'{roi} (n={n})')
            ax.fill_between(x, m - sem, m + sem, color=color, alpha=0.15)
        ax.axhline(DSR_COEF_REFERENCE, color='0.5', lw=0.8, ls='--')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(f'mean coef ({DSR_COEF_NORMALIZE}-normalised)',
                      fontsize=9)
        ax.set_title(f'DSR coef vs {xlabel}', fontsize=10)

    axes[0, -1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                       fontsize=7, frameon=False)
    fig.suptitle('Per-ROI DSR coefficient mean ± SEM', fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.88, 0.93])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {save_path}")


def plot_dsr_coef_histograms_by_factor(coefs_df, factor, save_path,
                                       rois=DSR_COEF_HISTOGRAM_ROIS,
                                       n_bins_hist=30):
    """Grid of per-bin coefficient histograms — rows = bins of `factor`,
    columns = ROIs.

    Layout follows the user's request: left col = ACC, middle = HC_anterior,
    right = HC_posterior (whatever `rois` lists in that order).  Within each
    cell we show the across-neuron distribution of the per-neuron coefficient
    averaged over the two non-`factor` dimensions; the red line marks the
    cell's mean.  X-range and bin edges are shared so panels are directly
    comparable.
    """
    factor_meta = {
        'lag':      ('future lag', [str(i) for i in range(DSR_N_LAGS)]),
        'location': ('location',
                     [str(i + 1) for i in range(DSR_N_LOCATIONS)]),
        'phase':    ('phase', list(DSR_PHASE_NAMES)),
    }
    label, bin_labels = factor_meta[factor]
    rois_present = [r for r in rois if r in coefs_df['roi'].unique()]
    if not rois_present:
        print(f"  no requested ROIs found for {factor}; skipping histogram.")
        return
    n_bins = len(bin_labels)
    n_rois = len(rois_present)

    # Pre-compute per-ROI values once.
    vals_per_roi = {
        roi: per_neuron_by_factor(
            coefs_df[coefs_df['roi'] == roi], factor=factor)
        for roi in rois_present
    }
    pooled = np.concatenate(
        [v.ravel() for v in vals_per_roi.values() if v.size > 0]
    ) if vals_per_roi else np.array([])
    if pooled.size == 0:
        print(f"  no values to histogram for {factor}; skipping.")
        return
    x_max = float(np.nanpercentile(pooled, 99))
    x_min = min(0.0, float(np.nanmin(pooled)))
    hist_edges = np.linspace(x_min, x_max, n_bins_hist + 1)

    fig, axes = plt.subplots(
        n_bins, n_rois,
        figsize=(n_rois * 3.0, n_bins * 1.4),
        squeeze=False, sharex=True, sharey=False,
    )

    for r_idx, roi in enumerate(rois_present):
        vals = vals_per_roi[roi]
        n_neu = len(vals)
        omni_p = _friedman_test_across_bins(vals) if n_neu > 1 else np.nan
        p_corr, signs = (
            _per_bin_vs_grand_mean(vals) if n_neu > 1
            else (np.full(n_bins, np.nan), np.zeros(n_bins, dtype=int))
        )

        for b_idx in range(n_bins):
            ax = axes[b_idx, r_idx]
            if n_neu == 0:
                ax.text(0.5, 0.5, 'no neurons',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            x = vals[:, b_idx]
            ax.hist(x, bins=hist_edges, color='steelblue',
                    edgecolor='white', alpha=0.85)
            # Reference = per-neuron average coefficient level.
            ax.axvline(DSR_COEF_REFERENCE, color='0.5', lw=0.8, ls='--')
            m = float(np.nanmean(x))
            ax.axvline(m, color='tab:red', lw=1.2)

            # Per-bin annotation: corrected p vs grand mean.
            if np.isfinite(p_corr[b_idx]) and p_corr[b_idx] < 0.05:
                stars = ('***' if p_corr[b_idx] < 0.001
                         else '**' if p_corr[b_idx] < 0.01 else '*')
                arrow = '↑' if signs[b_idx] > 0 else '↓'
                ax.text(0.97, 0.92, f'{arrow}{stars}',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=8, fontweight='bold',
                        color='tab:red' if signs[b_idx] > 0 else 'tab:purple')

            ax.tick_params(labelsize=6)
            ax.text(0.97, 0.05, f'µ={m:.3f}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=6, color='tab:red')
            if r_idx == 0:
                ax.set_ylabel(f'{label} {bin_labels[b_idx]}', fontsize=7)
            if b_idx == 0:
                ax.set_title(
                    f'{roi} (n={n_neu})\nFriedman p='
                    + (f'{omni_p:.2e}' if np.isfinite(omni_p) else 'nan'),
                    fontsize=8,
                )
            if b_idx == n_bins - 1:
                ax.set_xlabel(
                    f'coef ({DSR_COEF_NORMALIZE}-normalised)', fontsize=7)

    fig.suptitle(
        f'DSR coefficient distributions by {label}  '
        f'(per-neuron {DSR_COEF_NORMALIZE}-normalised, '
        f'p_perm<{DSR_COEF_P_PERM_THRESHOLD};  '
        f'red line = mean;  arrow+star = Bonferroni p vs grand mean)',
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {save_path}")


def plot_dsr_anova_summary(anova_df, save_path):
    """Heatmap of the non-parametric 3-way analysis: ROI × effect.

    Cell colour = Kendall's W (effect size, 0-1); annotation = BH-FDR
    p-value with significance stars; significant cells (p_fdr < 0.05) get
    a red outline.
    """
    if anova_df is None or anova_df.empty:
        print("  no ANOVA results to plot; skipping summary heatmap.")
        return
    rois = list(dict.fromkeys(anova_df['roi'].tolist()))
    n_roi, n_eff = len(rois), len(ANOVA_EFFECTS)

    W = np.full((n_roi, n_eff), np.nan)
    P = np.full((n_roi, n_eff), np.nan)
    for i, roi in enumerate(rois):
        for j, eff in enumerate(ANOVA_EFFECTS):
            r = anova_df[(anova_df['roi'] == roi)
                         & (anova_df['effect'] == eff)]
            if not r.empty:
                W[i, j] = float(r['kendall_w'].iloc[0])
                P[i, j] = float(r['p_fdr'].iloc[0])

    roi_n = {roi: int(anova_df[anova_df['roi'] == roi]['n_neurons'].iloc[0])
             for roi in rois}
    roi_labels = [f'{r} (n={roi_n[r]})' for r in rois]

    fig, ax = plt.subplots(
        figsize=(1.6 * n_eff + 2.5, 0.6 * n_roi + 2.5),
        constrained_layout=True,
    )
    vmax = float(np.nanmax(W)) if np.isfinite(W).any() else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(W, cmap='viridis', vmin=0, vmax=vmax, aspect='auto')

    ax.set_xticks(range(n_eff))
    ax.set_xticklabels(ANOVA_EFFECTS, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(n_roi))
    ax.set_yticklabels(roi_labels, fontsize=9)

    for i in range(n_roi):
        for j in range(n_eff):
            p = P[i, j]
            if not np.isfinite(p):
                continue
            stars = ('***' if p < 0.001 else '**' if p < 0.01
                     else '*' if p < 0.05 else '')
            txt = f'p={p:.3f}\n{stars}' if stars else f'p={p:.3f}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                    color='white' if W[i, j] < 0.6 * vmax else 'black')
            if p < 0.05:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor='red', lw=2.2))

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Kendall's W (effect size)", fontsize=9)
    ax.set_title(
        "DSR coefficient structure — non-parametric 3-way analysis\n"
        "Friedman per effect (main effects = factor marginals; "
        "interactions = interaction residuals);\n"
        "annotation = BH-FDR p,  red outline = p_fdr < 0.05",
        fontsize=10,
    )
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {save_path}")


def plot_dsr_interaction_patterns(coefs_df, save_path, rois=None):
    """Mean two-way interaction-residual heatmaps, ROI × interaction.

    Rows = ROIs, columns = the three two-way interactions.  Each panel is
    the across-neuron mean interaction residual (cell minus the two main
    effects); a diverging colour map centred on 0 shows which factor
    combinations are jointly over- (red) or under- (blue) represented.
    """
    if rois is None:
        rois = sorted(coefs_df['roi'].dropna().unique().tolist())
    if not rois:
        print("  no ROIs for interaction patterns; skipping.")
        return

    # (name, collapse_axis, (row_label, row_ticks), (col_label, col_ticks))
    specs = [
        ('location x phase', 3,
         ('location', [str(i + 1) for i in range(DSR_N_LOCATIONS)]),
         ('phase', list(DSR_PHASE_NAMES))),
        ('location x lag', 2,
         ('location', [str(i + 1) for i in range(DSR_N_LOCATIONS)]),
         ('lag', [str(i) for i in range(DSR_N_LAGS)])),
        ('phase x lag', 1,
         ('phase', list(DSR_PHASE_NAMES)),
         ('lag', [str(i) for i in range(DSR_N_LAGS)])),
    ]

    n_roi = len(rois)
    fig, axes = plt.subplots(
        n_roi, 3, figsize=(13, 2.7 * n_roi), squeeze=False,
    )

    for i, roi in enumerate(rois):
        sub = coefs_df[coefs_df['roi'] == roi]
        A = (np.stack([reshape_dsr(c) for c in sub['coefs']])
             if not sub.empty else None)
        for j, (name, axis, (ylab, yticks), (xlab, xticks)) in enumerate(specs):
            ax = axes[i, j]
            if A is None or A.shape[0] == 0:
                ax.set_title(f'{roi}: no neurons', fontsize=8)
                ax.axis('off')
                continue
            table = A.mean(axis=axis)                      # (N, d1, d2)
            mean_resid = _interaction_residuals(table).mean(axis=0)
            vlim = float(np.nanmax(np.abs(mean_resid))) or 1e-6
            im = ax.imshow(mean_resid, cmap='RdBu_r',
                           vmin=-vlim, vmax=vlim, aspect='auto')
            ax.set_xticks(range(len(xticks)))
            ax.set_xticklabels(xticks, fontsize=6)
            ax.set_yticks(range(len(yticks)))
            ax.set_yticklabels(yticks, fontsize=6)
            ax.set_xlabel(xlab, fontsize=7)
            ax.set_ylabel(f'{roi}\n{ylab}' if j == 0 else ylab, fontsize=7)
            if i == 0:
                ax.set_title(name, fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        'DSR two-way interaction residuals (across-neuron mean; '
        'red = over-, blue = under-represented combination)',
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {save_path}")


if _run_coefficients:
    print(f"\nLoading DSR coefficients from {DIAGNOSTICS_PKL}")
    with open(DIAGNOSTICS_PKL, 'rb') as f:
        _diagnostics = pickle.load(f)
    print(f"  loaded diagnostics for {len(_diagnostics)} subjects.")

    print("  DSR coefficient indexing reminder:")
    print(f"    324 coefficients per neuron, arranged as "
          f"(location 1..{DSR_N_LOCATIONS}) × (phase {DSR_PHASE_NAMES}) "
          f"× (lag 0..{DSR_N_LAGS - 1})")
    print("    coefs[loc * 36 + phase * 12 + lag]")
    print(f"    per-neuron normalisation = {DSR_COEF_NORMALIZE!r}, "
          f"p_perm threshold = {DSR_COEF_P_PERM_THRESHOLD}")

    dsr_coefs_df = load_dsr_coefs(
        _diagnostics,
        p_perm_threshold=DSR_COEF_P_PERM_THRESHOLD,
        normalize=DSR_COEF_NORMALIZE,
    )
    print(f"  {len(dsr_coefs_df)} DSR-encoding neurons kept "
          f"(p_perm < {DSR_COEF_P_PERM_THRESHOLD}).")
    print("  per-ROI counts:")
    print(dsr_coefs_df['roi'].value_counts().to_string())

    for factor in ['lag', 'location', 'phase']:
        plot_dsr_coef_distribution_by_factor(
            dsr_coefs_df, factor=factor,
            save_path=os.path.join(OUT_DIR, f'fig10_dsr_coef_by_{factor}.png'),
        )

    plot_dsr_coef_overlay_across_rois(
        dsr_coefs_df,
        save_path=os.path.join(OUT_DIR, 'fig10_dsr_coef_overlay.png'),
    )

    # Fig 11 — detailed per-bin histogram grid (rows=bins, cols=ROIs).
    for factor in ['lag', 'location', 'phase']:
        plot_dsr_coef_histograms_by_factor(
            dsr_coefs_df, factor=factor,
            save_path=os.path.join(
                OUT_DIR, f'fig11_dsr_coef_hist_by_{factor}.png'),
            rois=DSR_COEF_HISTOGRAM_ROIS,
        )

    # ── Non-parametric 3-way analysis: main effects + interactions ───
    print("\nDSR coefficient 3-way analysis "
          "(Friedman per effect, BH-FDR across all roi × effect tests):")
    anova_df = dsr_coef_anova(dsr_coefs_df)
    if not anova_df.empty:
        anova_csv = os.path.join(OUT_DIR, 'dsr_coef_anova.csv')
        anova_df.to_csv(anova_csv, index=False)
        print(f"Wrote {anova_csv}")
        with pd.option_context('display.max_rows', None,
                               'display.width', 160):
            print(anova_df[['roi', 'effect', 'n_neurons', 'friedman_chi2',
                            'dof', 'p_raw', 'p_fdr', 'kendall_w']]
                  .to_string(index=False))

        plot_dsr_anova_summary(
            anova_df,
            save_path=os.path.join(OUT_DIR, 'fig12_dsr_anova_summary.png'),
        )
        plot_dsr_interaction_patterns(
            dsr_coefs_df,
            save_path=os.path.join(
                OUT_DIR, 'fig13_dsr_interaction_patterns.png'),
        )
    else:
        print("  no neurons available for the 3-way analysis.")


print(f"\nAll outputs in: {OUT_DIR}")
