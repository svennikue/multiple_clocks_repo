#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Follow-up analyses on the encoding_analysis_simple result table.

# ── Which analysis sections to run ────────────────────────────────────
# Pick any subset; sections are independent except that all CSV-based
# sections share the encoding_results.csv load.
#   'classification'  -> per-neuron model assignment CSV + figs 1-2
#   'focal'           -> focal-model filter + figs 4-5
#   'dsr_filter'      -> fig 6  (dsr distribution under exclusion filters)
#   'dsr_co_encoding' -> fig 7  (dsr co-encoding stratified by count)
#   'state_removal'   -> figs 8-9 (dsr after removing state encoders)
#   'coefficients'    -> figs 10-13 (DSR coefficient analysis from
#                        diagnostics.pkl — independent of the CSV)
#   'brain_lag'       -> fig 14, 16 (DSR peak-lag glass-brain + 3D maps;
#                        diagnostics.pkl + MNI table — independent of CSV)
#   'r_dist_no_state' -> fig 17 (r-distribution grid for every non-state
#                        model, after dropping state-significant neurons)
#   'overlap_diagrams'    -> Venn + UpSet of perm-sig DSR/location/state/phase
#                            neurons, per ROI
#   'exclusion_variants'  -> generalisation of fig 17. For each exclusion set
#                            (state, location, phase, all of them, none) drop
#                            the perm-sig neurons and re-render: r-distribution
#                            grid, ROI × model t-stat heatmap, and significant
#                            proportion plot. Excluded counts saved per ROI.
#   'gradient'            -> figs 20-22: lag-coefficient gradient analyses.
#                            Per ROI, distribution of how many lags are near
#                            the peak (fig20), distance between argmax and
#                            second peak (fig21), discrete-colour peak-lag
#                            glass-brain (whole-brain + mask-only, fig22),
#                            and mask-only fig16 variants (peak-lag vs MNI z,
#                            p_perm threshold + transparency-by-r).
#   'sparse_examples'     -> fig 24: among perm-sig DSR cells, render
#                            publication examples of the sparsest encoders
#                            (few large coefficients on best fold). Layout:
#                            perm null + actual-vs-predicted trace + per-bin
#                            location, phase, and strongest-coef-lag bars.
#                            Top-N per ROI; full top-coef table saved.

@author: Svenja Kuchenhoff
"""

import os
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
# import pdb; pdb.set_trace()
# ── Settings ──────────────────────────────────────────────────────────


RESULT_DIR = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/encoding_analysis_simple/2026-06-05_17-58-57/')
# RESULT_DIR = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/encoding_analysis_simple/2026-05-22_15-57-38/')
# RESULT_DIR = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/encoding_analysis_simple/2026-05-21_10-37-45-nopositiveReg/')
#RESULT_DIR = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/encoding_analysis_simple/2026-05-20_22-57-03/')
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


SECTIONS_TO_RUN = ['gradient', 'brain_lag']

_run_classification  = 'classification'    in SECTIONS_TO_RUN
_run_focal           = 'focal'             in SECTIONS_TO_RUN
_run_dsr_filter      = 'dsr_filter'        in SECTIONS_TO_RUN
_run_dsr_co_encoding = 'dsr_co_encoding'   in SECTIONS_TO_RUN
_run_state_removal   = 'state_removal'     in SECTIONS_TO_RUN
_run_coefficients    = 'coefficients'      in SECTIONS_TO_RUN
_run_brain_lag       = 'brain_lag'         in SECTIONS_TO_RUN
_run_r_dist_no_state = 'r_dist_no_state'   in SECTIONS_TO_RUN
_run_overlap         = 'overlap_diagrams'  in SECTIONS_TO_RUN
_run_exclusion       = 'exclusion_variants' in SECTIONS_TO_RUN
_run_gradient        = 'gradient'          in SECTIONS_TO_RUN
_run_sparse          = 'sparse_examples'   in SECTIONS_TO_RUN
_need_csv = any([_run_classification, _run_focal, _run_dsr_filter,
                 _run_dsr_co_encoding, _run_state_removal,
                 _run_r_dist_no_state, _run_overlap, _run_exclusion])
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


# ── Fig 17: r-distribution grid excluding state-encoding neurons ──────
# Same figure as encoding_analysis_simple.py's plot_r_distribution_grid,
# but for every model EXCEPT 'state', after dropping neurons that are
# permutation-significant for 'state' (p_perm < ALPHA_PERM).
if _run_r_dist_no_state:
    from mc.plotting.cell_results import plot_r_distribution_grid

    state_sig = set(df_all.loc[(df_all['model'] == 'state')
                               & (df_all['p_perm'] < ALPHA_PERM), 'neuron'])
    keep = df_all[(~df_all['neuron'].isin(state_sig))
                  & (df_all['model'] != 'state')].copy()
    models_no_state = [m for m in MODELS if m != 'state']
    print(f"\nFig 17: dropped {len(state_sig)} state-significant neurons "
          f"(p_perm<{ALPHA_PERM}); plotting {len(models_no_state)} models "
          f"for {keep['neuron'].nunique()} remaining neurons.")

    # Regularisation alpha for the title (from the encoding run's config).
    _reg_alpha = None
    try:
        with open(os.path.join(RESULT_DIR, 'config.json')) as _cf:
            _reg_alpha = json.load(_cf).get('ALPHA')
    except Exception:
        pass

    _fig17_path = os.path.join(OUT_DIR, 'fig17_r_distribution_no_state.png')
    plot_r_distribution_grid(keep, models_no_state, save_path=_fig17_path,
                             alpha=ALPHA_PERM, reg_alpha=_reg_alpha)
    print(f"Wrote {_fig17_path}")


# ── Fig 18: Overlap diagrams for perm-sig DSR / location / state / phase ──
# Two complementary views per ROI:
#   (a) 3-set Venn over DSR, state, phase  (the three theoretical rivals)
#   (b) 4-set UpSet-style plot incl. location
# The Venn falls back to a hand-drawn version when matplotlib_venn is missing.
OVERLAP_ROIS = TARGET_ROIS
OVERLAP_MODELS = ['dsr', 'location', 'state', 'phase']


def _sig_neurons(df_roi, model, alpha=ALPHA_PERM):
    """Set of neurons in `df_roi` with `p_perm < alpha` for `model`."""
    return set(df_roi.loc[(df_roi['model'] == model)
                          & (df_roi['p_perm'] < alpha), 'neuron'])


def plot_3set_venn(set_dict, save_path, title=''):
    """3-set Venn diagram (matplotlib_venn if available, else fallback).

    `set_dict` is an ordered dict mapping label -> set of items.
    """
    if len(set_dict) != 3:
        raise ValueError("plot_3set_venn expects exactly 3 sets")
    labels = list(set_dict.keys())
    sets = list(set_dict.values())
    try:
        from matplotlib_venn import venn3
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        v = venn3(sets, set_labels=labels, ax=ax)
        # Use bigger fonts on the set labels + counts.
        for text in (v.set_labels or []):
            if text is not None:
                text.set_fontsize(12)
                text.set_fontweight('bold')
        for text in (v.subset_labels or []):
            if text is not None:
                text.set_fontsize(11)
        ax.set_title(title, fontsize=13, fontweight='bold')
    except ImportError:
        # Hand-drawn fallback: three overlapping circles with counts in
        # each of the 7 non-empty regions.
        A, B, C = sets
        only_A = len(A - B - C)
        only_B = len(B - A - C)
        only_C = len(C - A - B)
        AB = len((A & B) - C)
        AC = len((A & C) - B)
        BC = len((B & C) - A)
        ABC = len(A & B & C)
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        circ_kwargs = dict(alpha=0.35, fill=True, edgecolor='black', lw=1.5)
        from matplotlib.patches import Circle
        ax.add_patch(Circle((-0.6, 0.4), 1.2, color='tab:blue',
                            **circ_kwargs))
        ax.add_patch(Circle((0.6, 0.4), 1.2, color='tab:orange',
                            **circ_kwargs))
        ax.add_patch(Circle((0.0, -0.6), 1.2, color='tab:green',
                            **circ_kwargs))
        coords = {
            'only_A': (-1.4, 0.6, only_A),
            'only_B': (1.4, 0.6, only_B),
            'only_C': (0.0, -1.5, only_C),
            'AB':     (0.0, 1.0, AB),
            'AC':     (-0.9, -0.5, AC),
            'BC':     (0.9, -0.5, BC),
            'ABC':    (0.0, 0.0, ABC),
        }
        for x, y, n in coords.values():
            ax.text(x, y, str(n), ha='center', va='center',
                    fontsize=11, fontweight='bold')
        ax.text(-1.6, 1.7, labels[0], ha='center', va='center',
                fontsize=12, fontweight='bold', color='tab:blue')
        ax.text(1.6, 1.7, labels[1], ha='center', va='center',
                fontsize=12, fontweight='bold', color='tab:orange')
        ax.text(0.0, -2.1, labels[2], ha='center', va='center',
                fontsize=12, fontweight='bold', color='tab:green')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=13, fontweight='bold')
        print("  [venn] matplotlib_venn not installed — using fallback.")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {save_path} (+ .svg)")


def plot_upset(set_dict, save_path, title='', base_fontsize=12):
    """Minimal UpSet-style plot using matplotlib only.

    Top panel: bar chart of intersection sizes (sorted descending).
    Bottom panel: dot matrix indicating which sets are in each intersection.
    Empty intersections are dropped.
    """
    keys = list(set_dict.keys())
    n_sets = len(keys)
    universe = set().union(*set_dict.values())
    if not universe:
        print(f"  [upset] empty universe — skipping {save_path}")
        return

    counts = {}
    for item in universe:
        pattern = tuple(item in set_dict[k] for k in keys)
        counts[pattern] = counts.get(pattern, 0) + 1
    rows = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    n_inter = len(rows)

    fig, (ax_bar, ax_mat) = plt.subplots(
        2, 1,
        figsize=(max(7.5, 0.55 * n_inter + 3.5), 5.5),
        gridspec_kw={'height_ratios': [3, 1.4]},
        constrained_layout=True, sharex=True,
    )

    x = np.arange(n_inter)
    counts_arr = np.array([c for _, c in rows], dtype=int)
    ax_bar.bar(x, counts_arr, color='tab:blue', edgecolor='none')
    for xi, c in zip(x, counts_arr):
        ax_bar.text(xi, c + max(counts_arr.max() * 0.02, 0.5),
                    str(int(c)),
                    ha='center', va='bottom',
                    fontsize=base_fontsize - 1)
    ax_bar.set_ylabel('# neurons', fontsize=base_fontsize)
    ax_bar.set_title(title, fontsize=base_fontsize + 2, fontweight='bold')
    ax_bar.set_xticks([])
    ax_bar.spines[['top', 'right']].set_visible(False)
    ax_bar.tick_params(axis='y', labelsize=base_fontsize - 1)
    ax_bar.set_ylim(0, counts_arr.max() * 1.15)

    for j, (pattern, _) in enumerate(rows):
        rows_on = [i for i, on in enumerate(pattern) if on]
        for i in range(n_sets):
            color = 'black' if pattern[i] else '0.85'
            ax_mat.scatter(j, i, s=95, color=color, zorder=2)
        if len(rows_on) > 1:
            ax_mat.plot([j, j], [rows_on[0], rows_on[-1]],
                        color='black', lw=1.6, zorder=1)
    ax_mat.set_xticks([])
    ax_mat.set_yticks(range(n_sets))
    ax_mat.set_yticklabels(keys, fontsize=base_fontsize)
    ax_mat.set_xlim(-0.5, n_inter - 0.5)
    ax_mat.set_ylim(n_sets - 0.5, -0.5)
    ax_mat.spines[['top', 'right', 'bottom']].set_visible(False)
    ax_mat.tick_params(axis='x', length=0)
    ax_mat.tick_params(axis='y', length=0)

    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    fig.savefig(save_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {save_path} (+ .svg)")


if _run_overlap:
    print('\n=== Overlap diagrams (perm-sig neurons per model) ===')
    overlap_dir = os.path.join(OUT_DIR, 'fig18_overlap_diagrams')
    os.makedirs(overlap_dir, exist_ok=True)
    overlap_rows = []
    for roi in OVERLAP_ROIS:
        df_roi = df_all[df_all['roi'] == roi]
        sets = {m: _sig_neurons(df_roi, m, alpha=ALPHA_PERM)
                for m in OVERLAP_MODELS}
        sizes = {m: len(s) for m, s in sets.items()}
        print(f"  {roi}: " + ', '.join(f"{m}={n}" for m, n in sizes.items()))
        overlap_rows.append({'roi': roi, **sizes})

        # Three-set Venn focused on DSR / state / phase.
        venn_sets = {k: sets[k] for k in ('dsr', 'state', 'phase')
                     if k in sets}
        if len(venn_sets) == 3 and any(venn_sets.values()):
            plot_3set_venn(
                venn_sets,
                save_path=os.path.join(
                    overlap_dir, f'fig18a_venn_dsr_state_phase_{roi}.png'),
                title=(f'{roi}: perm-sig overlap '
                       f'(DSR, state, phase; alpha={ALPHA_PERM})'),
            )

        # UpSet-style 4-set view incl. location.
        if any(sets.values()):
            plot_upset(
                sets,
                save_path=os.path.join(
                    overlap_dir, f'fig18b_upset_4sets_{roi}.png'),
                title=(f'{roi}: perm-sig overlap, 4 models '
                       f'(alpha={ALPHA_PERM})'),
            )

    if overlap_rows:
        overlap_csv = os.path.join(overlap_dir, 'sig_set_sizes.csv')
        pd.DataFrame(overlap_rows).to_csv(overlap_csv, index=False)
        print(f"Wrote {overlap_csv}")


# ── Fig 19: exclusion-variant grid (generalised fig 17) ───────────────
# For each exclusion variant — drop perm-sig {state}, {location}, {phase},
# {all of those}, or nothing — re-render the publication-style figures from
# encoding_analysis_simple on the surviving neurons and note how many cells
# were excluded per ROI.  The dropped model is also removed from the plot
# axes when we exclude its perm-sig neurons (otherwise that column is
# trivially biased).
EXCLUSION_VARIANTS = {
    'no_exclusion':   [],
    'minus_state':    ['state'],
    'minus_location': ['location'],
    'minus_phase':    ['phase'],
    'minus_all':      ['state', 'location', 'phase'],
}


def apply_exclusion(df, exclude_models, alpha=ALPHA_PERM):
    """Drop every row of any neuron that is perm-sig in any `exclude_models`.

    Returns (filtered_df, excluded_per_roi: dict, total_excluded_neurons: set).
    """
    excluded = set()
    for m in exclude_models:
        excluded |= set(df.loc[(df['model'] == m)
                               & (df['p_perm'] < alpha), 'neuron'])
    filtered = df[~df['neuron'].isin(excluded)].copy()
    excl_per_roi = (df[df['neuron'].isin(excluded)]
                    .groupby('roi')['neuron'].nunique().to_dict())
    return filtered, excl_per_roi, excluded


if _run_exclusion:
    from mc.plotting.cell_results import (
        plot_r_distribution_grid,
        plot_roi_model_heatmap,
        plot_significance_proportion,
        compute_roi_model_tstats,
        bh_fdr,
        CANONICAL_ROI_ORDER,
        CANONICAL_ENC_MODEL_ORDER,
    )

    _reg_alpha = None
    try:
        with open(os.path.join(RESULT_DIR, 'config.json')) as _cf:
            _reg_alpha = json.load(_cf).get('ALPHA')
    except Exception:
        pass

    exclusion_root = os.path.join(OUT_DIR, 'fig19_exclusion_variants')
    os.makedirs(exclusion_root, exist_ok=True)
    rois_in_data = sorted(df_all['roi'].dropna().unique().tolist())

    exclusion_summary = []
    for variant_name, exclude_models in EXCLUSION_VARIANTS.items():
        filtered, excl_per_roi, excluded_neurons = apply_exclusion(
            df_all, exclude_models, alpha=ALPHA_PERM)

        # Drop the excluded model(s) from the plotted axes — that column
        # would be trivially zeroed-out by the exclusion.
        models_remain = [m for m in MODELS if m not in exclude_models]
        filtered_plot = filtered[filtered['model'].isin(models_remain)].copy()

        variant_dir = os.path.join(exclusion_root, variant_name)
        os.makedirs(variant_dir, exist_ok=True)

        print(f"\n--- Exclusion variant: {variant_name} "
              f"(drop perm-sig: {exclude_models or 'none'}) ---")
        print(f"  excluded {len(excluded_neurons)} unique neurons; "
              f"{filtered_plot['neuron'].nunique()} remain.")
        rows_for_csv = []
        for roi in rois_in_data:
            n_excl = int(excl_per_roi.get(roi, 0))
            n_keep = int(filtered_plot
                         .loc[filtered_plot['roi'] == roi, 'neuron']
                         .nunique())
            n_orig = int(df_all
                         .loc[df_all['roi'] == roi, 'neuron']
                         .nunique())
            print(f"    {roi}: -{n_excl} excluded, {n_keep}/{n_orig} kept")
            rows_for_csv.append({
                'variant':           variant_name,
                'exclusion_models':  ';'.join(exclude_models)
                                     if exclude_models else 'none',
                'roi':               roi,
                'n_original':        n_orig,
                'n_excluded':        n_excl,
                'n_remaining':       n_keep,
                'frac_excluded':     (n_excl / n_orig) if n_orig else np.nan,
            })
        pd.DataFrame(rows_for_csv).to_csv(
            os.path.join(variant_dir, 'exclusion_counts.csv'), index=False)
        exclusion_summary.extend(rows_for_csv)

        # 1. r-distribution grid.
        plot_r_distribution_grid(
            filtered_plot, models_remain,
            save_path=os.path.join(variant_dir, 'r_distribution_grid.png'),
            alpha=ALPHA_PERM, reg_alpha=_reg_alpha,
        )

        # 2. ROI × model t-stat heatmap (same style as the publication run).
        stats_df = compute_roi_model_tstats(
            filtered_plot, models=models_remain)
        if not stats_df.empty:
            stats_df.to_csv(
                os.path.join(variant_dir, 'roi_model_tstats.csv'),
                index=False)
            plot_roi_model_heatmap(
                stats_df,
                models=CANONICAL_ENC_MODEL_ORDER,
                rois=CANONICAL_ROI_ORDER,
                value_col='t', annot_col='p_t', sig_col='p_t',
                n_col='n_cells', alpha=0.05,
                value_label='t-statistic (mean_r > 0)',
                save_path=os.path.join(
                    variant_dir, 'roi_model_tstat_heatmap.png'),
                title=(f'ROI × model — t-test mean_r > 0  '
                       f'({variant_name})'),
                base_fontsize=15,
            )

        # 3. Significant-proportion plot (FDR stars, legend outside).
        plot_significance_proportion(
            filtered_plot, models_remain,
            save_path=os.path.join(
                variant_dir, 'significant_proportion.png'),
            alpha=ALPHA_PERM, reg_alpha=_reg_alpha,
            use_fdr=True, legend_outside=True,
            model_order=CANONICAL_ENC_MODEL_ORDER,
        )

    pd.DataFrame(exclusion_summary).to_csv(
        os.path.join(exclusion_root, 'exclusion_counts_all_variants.csv'),
        index=False)
    print(f"\nWrote per-variant outputs + summary -> {exclusion_root}")


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
    'state': ['state'],
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
                   expected_size=DSR_N_COEFS,
                   abs_coefs=False):
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
    abs_coefs : bool
        If True, take |coef| of each fold BEFORE normalising — sign-neutral
        lag comparison for runs whose elastic net allowed negative
        coefficients. Also keeps 'mean' normalisation safe (a signed mean
        near zero would blow up / flip signs).

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
                if abs_coefs:
                    fold_coefs = np.abs(fold_coefs)
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


def dsr_coefs_can_be_negative(diagnostics, model_name=DSR_MODEL_NAME,
                              tol=1e-6):
    """True if any saved DSR coefficient is negative — i.e. the encoding
    run did not constrain the elastic net to positive=True."""
    return any(
        float(np.min(np.asarray(c, dtype=float))) < -tol
        for per_neuron in diagnostics.values()
        for per_model in per_neuron.values()
        if model_name in per_model
        for c in per_model[model_name].get('coefs', []))


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

        # Statistics on the genuine bins only (a wrap duplicate, if any, is
        # excluded so no bin is counted twice).
        omni_p = _friedman_test_across_bins(vals)
        p_corr, signs = _per_bin_vs_grand_mean(vals)

        # `lag` wraps around: duplicate the first lag at the end so the
        # last<->first transition is visible on one continuous axis.
        if factor == 'lag':
            vals_plot   = np.hstack([vals, vals[:, [0]]])
            labels_plot = bin_labels + [bin_labels[0]]
            p_corr_plot = np.append(p_corr, p_corr[0])
            signs_plot  = np.append(signs, signs[0])
        else:
            vals_plot, labels_plot = vals, bin_labels
            p_corr_plot, signs_plot = p_corr, signs
        positions = np.arange(len(labels_plot))

        # Boxplot (distribution per bin across neurons).
        ax.boxplot(vals_plot, positions=positions, showfliers=False,
                   widths=0.55,
                   medianprops={'color': 'tab:blue'},
                   whiskerprops={'color': '0.4'},
                   capprops={'color': '0.4'},
                   boxprops={'color': '0.4'})

        # Jittered raw points so the full distribution is visible.
        for b in range(len(labels_plot)):
            xs = positions[b] + (rng.uniform(-0.18, 0.18, size=n_neu))
            ax.scatter(xs, vals_plot[:, b], s=6, color='tab:blue',
                       alpha=0.4, edgecolor='none')

        # Mean line on top.
        ax.plot(positions, np.nanmean(vals_plot, axis=0),
                color='tab:red', lw=1.4, marker='o', ms=4, zorder=5)

        omni_str = (f"{omni_p:.2e}" if np.isfinite(omni_p) else 'nan')
        ax.set_title(f'{roi} (n={n_neu})  Friedman p={omni_str}',
                     fontsize=9)

        # Dotted line marking the wrap point (last bin = first lag again).
        if factor == 'lag':
            ax.axvline(len(bin_labels) - 0.5, color='0.7', lw=0.8, ls=':')

        # Per-bin stars + direction arrow above the box.
        for b in range(len(labels_plot)):
            if not np.isfinite(p_corr_plot[b]):
                continue
            if   p_corr_plot[b] < 0.001: stars = '***'
            elif p_corr_plot[b] < 0.01:  stars = '**'
            elif p_corr_plot[b] < 0.05:  stars = '*'
            else:                        continue
            arrow = '↑' if signs_plot[b] > 0 else '↓'
            ax.text(positions[b], y_max * 0.97, f'{arrow}{stars}',
                    ha='center', va='top', fontsize=7,
                    color='tab:red' if signs_plot[b] > 0 else 'tab:purple',
                    fontweight='bold')

        # Reference line = the per-neuron average coefficient level
        # (1.0 for 'mean'-normalised coefs); bins above it are
        # over-represented, below it under-represented.
        ax.axhline(DSR_COEF_REFERENCE, color='0.5', lw=0.8, ls='--')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_plot, fontsize=7)
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

    # If the encoding run allowed negative coefficients, compare lags
    # sign-neutrally: |coef| is taken per fold before normalising, so every
    # per-bin strength (figs 10-13) is magnitude-based.
    _coefs_negative = dsr_coefs_can_be_negative(_diagnostics)

    print("  DSR coefficient indexing reminder:")
    print(f"    324 coefficients per neuron, arranged as "
          f"(location 1..{DSR_N_LOCATIONS}) × (phase {DSR_PHASE_NAMES}) "
          f"× (lag 0..{DSR_N_LAGS - 1})")
    print("    coefs[loc * 36 + phase * 12 + lag]")
    print(f"    per-neuron normalisation = {DSR_COEF_NORMALIZE!r}"
          f"{' on |coef|' if _coefs_negative else ''}, "
          f"p_perm threshold = {DSR_COEF_P_PERM_THRESHOLD}")
    if _coefs_negative:
        print("    coefficients can be negative -> sign-neutral (|coef|) "
              "lag comparison.")

    dsr_coefs_df = load_dsr_coefs(
        _diagnostics,
        p_perm_threshold=DSR_COEF_P_PERM_THRESHOLD,
        normalize=DSR_COEF_NORMALIZE,
        abs_coefs=_coefs_negative,
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


# ── Fig 14-16: DSR peak-lag brain & anatomy maps ─────────────────────
# Per-lag strength of a DSR neuron is read from its clock coefficients
# (324 = 9 locations × 3 phases × 12 lags):
#   * positive-constrained run  -> strength = mean coef over the 27 anchors
#   * unconstrained run         -> strength = mean |coef| (option A); a
#     strongly negative coefficient still means the lag is strongly
#     encoded. The sign of the signed anchor-sum at the peak is kept as an
#     excitatory / suppressive label.
# peak lag   = argmax of the 12-value strength profile.
# peakedness = profile.max() / profile.mean()  (1 = flat, up to 12 = one
#              lag dominates).
#
# Two neuron-selection criteria, each plotted separately:
#   'psig'    -> permutation-significant (p_perm < BRAIN_LAG_P_THRESHOLD)
#   'poscorr' -> non-negative cross-validated correlation (mean_r >= 0)
#
# fig14 — peak-lag glass brains (whole + ACC) and 3D mesh, mean & best fold.
# fig15 — lag peakedness per ROI (box plots), both selections.
# fig16 — ACC only: peak lag vs MNI z (dorsal-ventral axis), both selections.
if _run_brain_lag or _run_gradient:
    from mc.plotting.cell_results import (plot_peaklag_glassbrain,
                                          plot_peaklag_3d_mesh,
                                          make_brain_anatomy_figure)

    ROI_TABLE_PATH = ('/Users/xpsy1114/Documents/projects/multiple_clocks/'
                      'data/ephys_humans/derivatives/'
                      'neurons_with_final_roi_labels.csv')
    ACC_ROI_LABELS        = ('ACC',)               # glass-brain ACC zoom
    ACC_DV_ROI_LABELS     = ('ACC', 'ventral_ACC')  # dorsal-ventral analysis
    BRAIN_LAG_P_THRESHOLD = 0.05      # p_perm cutoff for the 'psig' selection
    BRAIN_JITTER_MM       = 10       # spread cells sharing an MNI coordinate
    GLASS_MARKER_SIZE     = 3         # small, so individual cells are visible

    # Coefficient-sign mode. The lag post-hoc analysis must know whether the
    # encoding run's elastic net could produce negative coefficients
    # (positive=False). This is auto-detected from the saved coefficients
    # after loading (config.json's POSITIVE field is not always in sync with
    # the run). Set this to True or False to force the mode instead.
    FORCE_COEFS_CAN_BE_NEGATIVE = None

    def dsr_lag_profile(coefs_324, use_abs):
        """One (324,) coef vector -> (peak_lag, peakedness, sign_label).

        Per-lag strength = mean over the 27 location×phase anchors of
        |coef| (use_abs) or coef. peak_lag = argmax; peakedness =
        max/mean of the 12-value profile; sign from the signed anchor-sum
        at the peak ('excitatory' / 'suppressive').
        """
        arr = reshape_dsr(coefs_324)                     # (9, 3, 12)
        signed   = arr.sum(axis=(0, 1))                  # (12,)
        strength = (np.abs(arr).mean(axis=(0, 1)) if use_abs
                    else arr.mean(axis=(0, 1)))          # (12,)
        if strength.size == 0 or float(strength.mean()) <= 1e-12:
            return np.nan, np.nan, ''
        peak = int(np.argmax(strength))
        peakedness = float(strength.max() / strength.mean())
        sign = 'excitatory' if signed[peak] >= 0 else 'suppressive'
        return peak, peakedness, sign

    def circular_linear_corr(lags, x, period=DSR_N_LAGS):
        """Mardia circular-linear correlation between circular `lags`
        (0..period-1) and linear `x`. Returns (r in [0,1], p)."""
        lags = np.asarray(lags, dtype=float)
        x    = np.asarray(x, dtype=float)
        ok = np.isfinite(lags) & np.isfinite(x)
        lags, x = lags[ok], x[ok]
        n = len(x)
        if n < 4 or np.std(x) < 1e-12:
            return np.nan, np.nan
        ang = 2 * np.pi * lags / period
        c, s = np.cos(ang), np.sin(ang)
        if np.std(c) < 1e-12 or np.std(s) < 1e-12:
            return np.nan, np.nan
        rxc = np.corrcoef(x, c)[0, 1]
        rxs = np.corrcoef(x, s)[0, 1]
        rcs = np.corrcoef(c, s)[0, 1]
        denom = 1.0 - rcs ** 2
        if abs(denom) < 1e-12:
            return np.nan, np.nan
        r2 = (rxc**2 + rxs**2 - 2*rxc*rxs*rcs) / denom
        r2 = float(np.clip(r2, 0.0, 1.0))
        return np.sqrt(r2), float(stats.chi2.sf(n * r2, df=2))

    def parse_subject_cell(neuron_label):
        """'01_07-07-chan120-EC' -> (subject=1, cell_idx=7)."""
        subj, rest = neuron_label.split('_', 1)
        return int(subj), int(rest.split('-')[0])

    def jitter_duplicate_coords(df, jitter_mm=BRAIN_JITTER_MM):
        """Deterministic angular offset for cells sharing MNI coordinates."""
        out = df.copy()
        out[['px', 'py', 'pz']] = out[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy()
        for _, idxs in out.groupby(['MNI_x', 'MNI_y', 'MNI_z']).groups.items():
            idxs = list(idxs)
            if len(idxs) <= 1:
                continue
            ang = np.linspace(0, 2 * np.pi, len(idxs), endpoint=False)
            out.loc[idxs, 'px'] += jitter_mm * np.cos(ang)
            out.loc[idxs, 'py'] += jitter_mm * np.sin(ang)
        return out

    def plot_acc_lag_vs_z(tbl, save_path, title):
        """ACC neurons: peak lag (x) vs MNI z (y, dorsal-ventral axis).

        Marker hue = peak lag; marker opacity = effect strength (mean_r).
        """
        rng = np.random.default_rng(1)
        cmap = plt.get_cmap('YlOrRd')
        norm = plt.Normalize(vmin=0, vmax=DSR_N_LAGS - 1)
        lag = tbl['peak_lag_mean'].to_numpy(dtype=float)
        z   = tbl['MNI_z'].to_numpy(dtype=float)
        jx  = lag + rng.uniform(-0.22, 0.22, len(lag))

        # Per-point opacity from effect strength (mean_r): strong = opaque.
        strength = np.clip(np.nan_to_num(
            tbl['mean_r'].to_numpy(dtype=float)), 0.0, None)
        smax = float(np.nanmax(strength)) if strength.size else 0.0
        alphas = (0.15 + 0.85 * np.clip(strength / smax, 0, 1)
                  if smax > 0 else np.full(len(strength), 1.0))
        rgba = np.array([(*cmap(norm(L))[:3], a)
                         for L, a in zip(lag, alphas)])

        fig, ax = plt.subplots(figsize=(6.4, 6))
        for sign, marker in [('excitatory', 'o'), ('suppressive', 'X')]:
            m = (tbl['lag_sign_mean'] == sign).to_numpy()
            if m.any():
                ax.scatter(jx[m], z[m], color=rgba[m],
                           marker=marker, s=46, edgecolor='0.3',
                           linewidth=0.4,
                           label=f'{sign} (n={int(m.sum())})')
        for L in range(DSR_N_LAGS):                       # mean z per lag
            zl = z[lag == L]
            if len(zl):
                ax.plot([L - 0.35, L + 0.35], [float(np.mean(zl))] * 2,
                        color='black', lw=2)
        r, p = circular_linear_corr(lag, z)
        ax.set_xlabel('DSR peak lag', fontsize=9)
        ax.set_ylabel('MNI z   (ventral → dorsal)', fontsize=9)
        ax.set_xticks(range(DSR_N_LAGS))
        ax.set_title(f'{title}\ncircular-linear r={r:.2f}, p={p:.3f}, '
                     f'n={len(tbl)}   (black bars = mean z per lag)',
                     fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc='best')
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {save_path}")

    print(f"\nLoading diagnostics for peak-lag brain maps: {DIAGNOSTICS_PKL}")
    with open(DIAGNOSTICS_PKL, 'rb') as f:
        _diag_brain = pickle.load(f)

    # Detect whether the elastic net produced negative coefficients directly
    # from the saved DSR coefficients — config.json's POSITIVE flag is not
    # always in sync with the run.
    if FORCE_COEFS_CAN_BE_NEGATIVE is not None:
        COEFS_CAN_BE_NEGATIVE = bool(FORCE_COEFS_CAN_BE_NEGATIVE)
    else:
        COEFS_CAN_BE_NEGATIVE = any(
            float(np.min(np.asarray(c, dtype=float))) < -1e-6
            for per_neuron in _diag_brain.values()
            for per_model in per_neuron.values()
            if DSR_MODEL_NAME in per_model
            for c in per_model[DSR_MODEL_NAME].get('coefs', []))
    USE_ABS = COEFS_CAN_BE_NEGATIVE
    print(f"  coefficients can be negative = {COEFS_CAN_BE_NEGATIVE}  "
          f"-> per-lag strength = "
          f"{'mean |coef|' if USE_ABS else 'mean coef'}")

    # All DSR neurons (no p filter): mean-across-fold coefs + p_perm + mean_r.
    # With negative coefficients the per-fold normaliser must be sign-safe
    # ('mean' divides by ~0 and can flip signs); 'l2' is used instead.
    _norm = DSR_COEF_NORMALIZE if not COEFS_CAN_BE_NEGATIVE else 'l2'
    lag_tbl = load_dsr_coefs(_diag_brain, p_perm_threshold=None,
                             normalize=_norm)

    _prof = lag_tbl['coefs'].apply(lambda c: dsr_lag_profile(c, USE_ABS))
    lag_tbl['peak_lag_mean']   = [t[0] for t in _prof]
    lag_tbl['peakedness_mean'] = [t[1] for t in _prof]
    lag_tbl['lag_sign_mean']   = [t[2] for t in _prof]

    # Best held-out fold (raw coefs; argmax / peakedness are scale-invariant).
    best_lag = {}
    for sub, per_neuron in _diag_brain.items():
        for n_lab, per_model in per_neuron.items():
            d = per_model.get(DSR_MODEL_NAME)
            if d is None:
                continue
            coefs_list = d.get('coefs', [])
            if not coefs_list:
                continue
            coefs_arr = np.array([np.asarray(c, dtype=float)
                                  for c in coefs_list])
            if coefs_arr.ndim != 2 or coefs_arr.shape[1] != DSR_N_COEFS:
                continue
            r = np.asarray(d.get('r_per_fold', []), dtype=float)
            if r.size == 0 or not np.isfinite(r).any():
                continue
            best_lag[(sub, n_lab)] = dsr_lag_profile(
                coefs_arr[int(np.nanargmax(r))], USE_ABS)
    _empty = (np.nan, np.nan, '')
    _subs, _neus = lag_tbl['subject'], lag_tbl['neuron']
    lag_tbl['peak_lag_best']   = [best_lag.get((s, n), _empty)[0]
                                  for s, n in zip(_subs, _neus)]
    lag_tbl['peakedness_best'] = [best_lag.get((s, n), _empty)[1]
                                  for s, n in zip(_subs, _neus)]
    lag_tbl['lag_sign_best']   = [best_lag.get((s, n), _empty)[2]
                                  for s, n in zip(_subs, _neus)]

    # Join MNI coordinates on (subject, cell idx).
    mni_tbl = pd.read_csv(ROI_TABLE_PATH)[
        ['subject', 'cell idx', 'MNI_x', 'MNI_y', 'MNI_z']].rename(
        columns={'subject': 'mni_subject', 'cell idx': 'cell_idx'})
    mni_tbl = mni_tbl.drop_duplicates(['mni_subject', 'cell_idx'])
    sc = lag_tbl['neuron'].apply(
        lambda s: pd.Series(parse_subject_cell(s),
                            index=['mni_subject', 'cell_idx']))
    lag_tbl = pd.concat([lag_tbl, sc], axis=1).merge(
        mni_tbl, on=['mni_subject', 'cell_idx'], how='left')
    n_miss = int(lag_tbl['MNI_x'].isna().sum())
    if n_miss:
        print(f"  WARNING: {n_miss} neurons without MNI match — dropped.")
    lag_tbl = lag_tbl.dropna(
        subset=['MNI_x', 'MNI_y', 'MNI_z']).reset_index(drop=True)
    print(f"  {len(lag_tbl)} DSR neurons with MNI coordinates  "
          f"(p_perm<{BRAIN_LAG_P_THRESHOLD}: "
          f"{int((lag_tbl['p_perm'] < BRAIN_LAG_P_THRESHOLD).sum())}; "
          f"mean_r>=0: {int((lag_tbl['mean_r'] >= 0).sum())})")

    lag_tbl.drop(columns=['coefs']).to_csv(
        os.path.join(OUT_DIR, 'fig14_dsr_peaklag_table.csv'), index=False)

    SELECTIONS = {
        'psig':    (lag_tbl['p_perm'] < BRAIN_LAG_P_THRESHOLD,
                    f'p_perm<{BRAIN_LAG_P_THRESHOLD}'),
        'poscorr': (lag_tbl['mean_r'] >= 0, 'mean_r>=0'),
    }

    if _run_brain_lag:
        # Standard anatomical background (transparent brain + grey hippocampus +
        # dark-grey EC + grey ACC), built once and reused for every 3D mesh.
        print("  building anatomical brain background ...")
        BRAIN_BG_FIG = make_brain_anatomy_figure()

        # ── fig14: peak-lag glass brains + 3D mesh ───────────────────────
        # Marker hue = peak lag; marker opacity = effect strength (mean_r).
        for coef_key, lag_col in [('mean', 'peak_lag_mean'),
                                  ('bestfold', 'peak_lag_best')]:
            for sel_key, (sel_mask, sel_desc) in SELECTIONS.items():
                sub_tbl = lag_tbl[sel_mask].dropna(subset=[lag_col])
                tag  = f'{sel_key}_{coef_key}'
                desc = f'{sel_desc}, {coef_key} coefs'
                if sub_tbl.empty:
                    print(f"  [{tag}] no neurons — skipped.")
                    continue

                plotted  = jitter_duplicate_coords(sub_tbl)
                coords   = plotted[['px', 'py', 'pz']].to_numpy()
                lags     = plotted[lag_col].to_numpy()
                strength = plotted['mean_r'].to_numpy()

                # Whole-brain glass brain (3 orthogonal views).
                plot_peaklag_glassbrain(
                    coords, lags,
                    os.path.join(OUT_DIR, f'fig14_glass_whole_{tag}.png'),
                    title=f'DSR peak lag — whole brain ({desc}, '
                          f'n={len(plotted)})',
                    strengths=strength,
                    n_lags=DSR_N_LAGS, marker_size=GLASS_MARKER_SIZE)

                # ACC-only glass brain.
                acc = plotted[plotted['roi'].isin(ACC_ROI_LABELS)]
                plot_peaklag_glassbrain(
                    acc[['px', 'py', 'pz']].to_numpy(),
                    acc[lag_col].to_numpy(),
                    os.path.join(OUT_DIR, f'fig14_glass_acc_{tag}.png'),
                    title=f'DSR peak lag — ACC ({desc}, n={len(acc)})',
                    strengths=acc['mean_r'].to_numpy(),
                    n_lags=DSR_N_LAGS, marker_size=GLASS_MARKER_SIZE)

                # Interactive 3D mesh on the anatomical background.
                plot_peaklag_3d_mesh(
                    coords, lags,
                    os.path.join(OUT_DIR, f'fig14_3d_whole_{tag}.html'),
                    title=f'DSR peak lag — 3D ({desc}, n={len(plotted)})',
                    strengths=strength,
                    n_lags=DSR_N_LAGS, bg_fig=BRAIN_BG_FIG)

        # ── fig16: ACC peak lag vs dorsal-ventral (z) axis ───────────────
        for sel_key, (sel_mask, sel_desc) in SELECTIONS.items():
            acc_dv = lag_tbl[sel_mask & lag_tbl['roi'].isin(ACC_DV_ROI_LABELS)]
            acc_dv = acc_dv.dropna(subset=['peak_lag_mean', 'MNI_z'])
            if len(acc_dv) < 4:
                print(f"  fig16 [{sel_key}]: <4 ACC neurons — skipped.")
                continue
            plot_acc_lag_vs_z(
                acc_dv,
                os.path.join(OUT_DIR, f'fig16_acc_lag_vs_z_{sel_key}.png'),
                title=f'ACC — DSR peak lag vs dorsal-ventral position '
                      f'({sel_desc}, mean coefs)')

    # ── figs 20-22: gradient (per-cell lag distribution) analyses ────
    if _run_gradient:
        # Settings local to this section.
        GRADIENT_MASK_PATH = (
            '/Users/xpsy1114/Documents/projects/multiple_clocks/data/'
            'masks/gradient_mask_bin.ni.gz'
        )
        # "Near-peak" lags are those within `peak_threshold * max_strength`
        # of the cell's peak (default 0.9 -> within 10% of the peak).
        GRADIENT_PEAK_THRESHOLD = 0.9
        # Discrete colour bins, requested explicitly:
        #   lags 0-2 = bright yellow, 3-5 = dark yellow,
        #   lags 6-8 = orange,        9-11 = red.
        GRADIENT_LAG_COLORS = [
            ('#fff200', range(0, 3)),    # bright yellow
            ('#e6c200', range(3, 6)),    # dark yellow
            ('#ff8c1a', range(6, 9)),    # orange
            ('#e53935', range(9, 12)),   # red
        ]
        GRADIENT_OUT_DIR = os.path.join(OUT_DIR, 'fig20_22_gradient')
        os.makedirs(GRADIENT_OUT_DIR, exist_ok=True)

        # Helper: full 12-lag strength profile per cell.
        def _lag_profile(coefs_324, use_abs):
            arr = reshape_dsr(coefs_324)                 # (9, 3, 12)
            return (np.abs(arr).mean(axis=(0, 1)) if use_abs
                    else arr.mean(axis=(0, 1)))           # (12,)

        def _lag_color(lag, fallback='0.6'):
            for color, lags_in_bin in GRADIENT_LAG_COLORS:
                if int(lag) in lags_in_bin:
                    return color
            return fallback

        def _circular_distance(a, b, period=DSR_N_LAGS):
            d = abs(int(a) - int(b)) % period
            return min(d, period - d)

        def _voxel_inside_mask(coords_mm, mask_img):
            """coords: (N, 3) MNI mm. Returns (N,) bool."""
            data = mask_img.get_fdata()
            inv = np.linalg.inv(mask_img.affine)
            out = np.zeros(len(coords_mm), dtype=bool)
            shape = np.array(data.shape)
            for i, c in enumerate(coords_mm):
                v = nib_affines_apply_affine(inv, c)
                v = np.round(v).astype(int)
                if (v >= 0).all() and (v < shape).all():
                    out[i] = bool(data[tuple(v)] > 0)
            return out

        # Import nib lazily so the script still imports cleanly if nibabel
        # isn't on the path when other sections run.
        import nibabel as nib
        from nibabel.affines import apply_affine as nib_affines_apply_affine

        # Build a per-cell 12-lag profile + summary fields. Same coefficient-
        # sign mode as fig14/16 (USE_ABS auto-detected above).
        print('\n=== Gradient analysis (lag-coefficient distribution) ===')
        profiles = np.stack([
            _lag_profile(c, USE_ABS) for c in lag_tbl['coefs']
        ])                                                # (N_cells, 12)
        # Drop degenerate (all-zero) profiles so the divisions are safe.
        profile_max = profiles.max(axis=1)
        ok = profile_max > 1e-12
        grad_tbl = lag_tbl.loc[ok].copy().reset_index(drop=True)
        profiles = profiles[ok]
        profile_max = profile_max[ok]

        # Plot A inputs: number of near-peak lags per cell.
        near_peak_mask = (profiles
                          >= GRADIENT_PEAK_THRESHOLD * profile_max[:, None])
        n_near_peak = near_peak_mask.sum(axis=1).astype(int)

        # Plot B inputs: circular distance argmax -> 2nd-strongest lag.
        first_peak = profiles.argmax(axis=1)
        masked = profiles.copy()
        masked[np.arange(len(masked)), first_peak] = -np.inf
        second_peak = masked.argmax(axis=1)
        dist_first_to_second = np.array([
            _circular_distance(a, b)
            for a, b in zip(first_peak, second_peak)
        ], dtype=int)

        grad_tbl['n_near_peak'] = n_near_peak
        grad_tbl['first_peak'] = first_peak
        grad_tbl['second_peak'] = second_peak
        grad_tbl['dist_first_to_second'] = dist_first_to_second
        grad_tbl_csv_cols = [c for c in grad_tbl.columns if c != 'coefs']
        grad_tbl[grad_tbl_csv_cols].to_csv(
            os.path.join(GRADIENT_OUT_DIR, 'gradient_per_cell.csv'),
            index=False)

        # ── fig 20: distribution of n_near_peak per ROI ──────────────
        rois_present = sorted(grad_tbl['roi'].dropna().unique().tolist())
        n_cols = 3
        n_rows = int(np.ceil(len(rois_present) / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.6 * n_cols + 1.0, 2.6 * n_rows + 1.0),
            squeeze=False, constrained_layout=True,
        )
        bin_edges = np.arange(0.5, DSR_N_LAGS + 1.5)
        for ax_idx, roi in enumerate(rois_present):
            ax = axes[ax_idx // n_cols, ax_idx % n_cols]
            sub = grad_tbl[grad_tbl['roi'] == roi]
            if sub.empty:
                ax.axis('off')
                continue
            vals = sub['n_near_peak'].to_numpy()
            ax.hist(vals, bins=bin_edges,
                    color='steelblue', edgecolor='white', linewidth=0.6)
            med = float(np.median(vals))
            ax.axvline(med, color='tab:red', lw=1.4,
                       label=f'median = {med:.1f}')
            ax.set_xticks(np.arange(1, DSR_N_LAGS + 1))
            ax.set_xlabel(f'# lags within '
                          f'{int(GRADIENT_PEAK_THRESHOLD * 100)}% of peak',
                          fontsize=10)
            ax.set_ylabel('# cells', fontsize=10)
            ax.set_title(f'{roi}  (N={len(vals)})',
                         fontsize=11, fontweight='bold')
            ax.tick_params(labelsize=9)
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(fontsize=9, frameon=False)
        for k in range(len(rois_present), n_rows * n_cols):
            axes[k // n_cols, k % n_cols].axis('off')
        fig.suptitle(
            f'Number of near-peak lags per cell  '
            f'(threshold = {GRADIENT_PEAK_THRESHOLD:g} × peak)',
            fontsize=13, fontweight='bold',
        )
        fig20_path = os.path.join(GRADIENT_OUT_DIR,
                                  'fig20_near_peak_count_per_roi.png')
        fig.savefig(fig20_path, dpi=200, bbox_inches='tight')
        fig.savefig(fig20_path.replace('.png', '.svg'), bbox_inches='tight')
        plt.close(fig)
        print(f"Wrote {fig20_path} (+ .svg)")

        # ── fig 21: circular distance argmax -> 2nd-strongest lag ─────
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(rois_present) + 2),
                                         4.5),
                                constrained_layout=True)
        data_by_roi = [
            grad_tbl.loc[grad_tbl['roi'] == r, 'dist_first_to_second']
            .to_numpy()
            for r in rois_present
        ]
        positions = np.arange(len(rois_present))
        bp = ax.boxplot(data_by_roi, positions=positions,
                        widths=0.6, showfliers=False, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor((0.35, 0.55, 0.8, 0.45))
            patch.set_edgecolor('0.3')
        for median in bp['medians']:
            median.set_color('tab:red')
            median.set_linewidth(1.6)
        # Jittered raw points on top.
        rng = np.random.default_rng(0)
        for i, vals in enumerate(data_by_roi):
            if vals.size == 0:
                continue
            xs = positions[i] + rng.uniform(-0.18, 0.18, size=vals.size)
            ax.scatter(xs, vals, s=10, color='tab:blue',
                       alpha=0.45, edgecolor='none')
        # Annotate mean per ROI just above each box.
        max_y = max([(float(v.max()) if len(v) else 0)
                     for v in data_by_roi] + [DSR_N_LAGS // 2])
        for i, vals in enumerate(data_by_roi):
            if vals.size == 0:
                continue
            m = float(np.mean(vals))
            ax.text(positions[i], max_y + 0.4,
                    f'µ={m:.2f}\nn={len(vals)}',
                    ha='center', va='bottom', fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels(rois_present, rotation=30, ha='right', fontsize=10)
        ax.set_ylabel('circular |argmax − 2nd-strongest lag|',
                      fontsize=11)
        ax.set_title('Distance from peak lag to second-strongest lag, '
                     'per ROI',
                     fontsize=12, fontweight='bold')
        ax.set_ylim(-0.5, max(max_y + 1.5, DSR_N_LAGS // 2 + 1))
        ax.tick_params(labelsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        fig21_path = os.path.join(
            GRADIENT_OUT_DIR, 'fig21_peak_to_second_distance_per_roi.png')
        fig.savefig(fig21_path, dpi=200, bbox_inches='tight')
        fig.savefig(fig21_path.replace('.png', '.svg'), bbox_inches='tight')
        plt.close(fig)
        print(f"Wrote {fig21_path} (+ .svg)")

        # ── fig 22: discrete-colour peak-lag glass-brain ─────────────
        # All cells with mean_r > 0, coloured by peak-lag bin.
        positive = grad_tbl[grad_tbl['mean_r'] > 0].copy()
        if not positive.empty:
            jittered = jitter_duplicate_coords(positive)
            coords_all = jittered[['px', 'py', 'pz']].to_numpy()
            lags_all   = jittered['first_peak'].to_numpy()
            color_all  = [_lag_color(int(l)) for l in lags_all]

            def _plot_discrete_lag_glassbrain(coords, colors_list,
                                              lags_arr, save_path, title):
                from nilearn import plotting
                if len(coords) == 0:
                    print(f"  skip {save_path}: no cells.")
                    return
                display = plotting.plot_glass_brain(
                    None, display_mode='ortho', title=title,
                    black_bg=False, plot_abs=False)
                display.add_markers(
                    coords, marker_color=colors_list,
                    marker_size=GLASS_MARKER_SIZE,
                )
                fig = plt.gcf()
                # Discrete-bin legend (4 bins).
                handles = [
                    plt.Line2D(
                        [0], [0], marker='o', linestyle='',
                        markersize=8,
                        markerfacecolor=color,
                        markeredgecolor='none',
                        label=f'lag {min(lags_in_bin)}-{max(lags_in_bin)}',
                    )
                    for color, lags_in_bin in GRADIENT_LAG_COLORS
                ]
                fig.legend(handles=handles, loc='lower center',
                           ncol=len(handles), frameon=False, fontsize=10,
                           title='peak lag', handletextpad=0.4,
                           columnspacing=1.4)
                fig.savefig(save_path, dpi=200, bbox_inches='tight')
                fig.savefig(save_path.replace('.png', '.svg'),
                            bbox_inches='tight')
                plt.close(fig)
                print(f"Wrote {save_path} (+ .svg)")

            _plot_discrete_lag_glassbrain(
                coords_all, color_all, lags_all,
                save_path=os.path.join(
                    GRADIENT_OUT_DIR,
                    'fig22a_peaklag_discrete_glassbrain_all.png'),
                title=(f'DSR peak lag (discrete bins) — '
                       f'all mean_r>0 cells (n={len(coords_all)})'),
            )

            # Mask filter — try the exact path the user provided, falling
            # back to `.nii.gz` if `.ni.gz` is unusual on their filesystem.
            mask_candidates = [GRADIENT_MASK_PATH]
            if GRADIENT_MASK_PATH.endswith('.ni.gz'):
                mask_candidates.append(
                    GRADIENT_MASK_PATH.replace('.ni.gz', '.nii.gz'))
            resolved_mask = next(
                (p for p in mask_candidates if os.path.isfile(p)), None)
            if resolved_mask is not None:
                print(f"  loading gradient mask: {resolved_mask}")
                mask_img = nib.load(resolved_mask)
                coords_mni = positive[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy()
                in_mask = _voxel_inside_mask(coords_mni, mask_img)
                n_in = int(in_mask.sum())
                print(f"  {n_in}/{len(in_mask)} cells inside mask.")
                positive_masked = positive[in_mask].copy()

                if not positive_masked.empty:
                    jittered_m = jitter_duplicate_coords(positive_masked)
                    coords_m  = jittered_m[['px', 'py', 'pz']].to_numpy()
                    lags_m    = jittered_m['first_peak'].to_numpy()
                    colors_m  = [_lag_color(int(l)) for l in lags_m]
                    _plot_discrete_lag_glassbrain(
                        coords_m, colors_m, lags_m,
                        save_path=os.path.join(
                            GRADIENT_OUT_DIR,
                            'fig22b_peaklag_discrete_glassbrain_mask.png'),
                        title=(f'DSR peak lag (discrete bins) — '
                               f'cells in gradient mask (n={n_in})'),
                    )

                    # ── fig 23: mask-only fig16 variants ─────────────
                    # (a) p_perm threshold; (b) all mean_r>0 with alpha=r.
                    mask_dv = positive_masked.dropna(
                        subset=['peak_lag_mean', 'MNI_z'])
                    mask_dv_psig = mask_dv[
                        mask_dv['p_perm'] < BRAIN_LAG_P_THRESHOLD]
                    if len(mask_dv_psig) >= 4:
                        plot_acc_lag_vs_z(
                            mask_dv_psig,
                            os.path.join(
                                GRADIENT_OUT_DIR,
                                'fig23a_mask_lag_vs_z_psig.png'),
                            title=(f'Gradient mask — peak lag vs MNI z  '
                                   f'(p_perm<{BRAIN_LAG_P_THRESHOLD}, '
                                   f'n={len(mask_dv_psig)})'),
                        )
                    else:
                        print(f"  fig23a: <4 cells passing p_perm filter "
                              f"in mask — skipped.")

                    if len(mask_dv) >= 4:
                        plot_acc_lag_vs_z(
                            mask_dv,
                            os.path.join(
                                GRADIENT_OUT_DIR,
                                'fig23b_mask_lag_vs_z_poscorr.png'),
                            title=(f'Gradient mask — peak lag vs MNI z  '
                                   f'(mean_r>0, opacity = mean_r, '
                                   f'n={len(mask_dv)})'),
                        )
                    else:
                        print(f"  fig23b: <4 cells with mean_r>0 in mask "
                              f"— skipped.")
                else:
                    print("  fig22b/23: no cells inside mask — skipped.")
            else:
                print(f"  gradient mask not found "
                      f"(tried {', '.join(mask_candidates)}) — "
                      f"skipped figs 22b, 23.")
        else:
            print("  no cells with mean_r>0 — gradient glass-brains skipped.")


# ── Fig 24: sparse-encoding DSR examples with trajectory overlay ──────
# For each perm-sig DSR neuron, score sparseness on the best-fold
# coefficient vector (few coefficients dominate the L1 mass). Render the
# sparsest N per ROI as a publication-style figure showing:
#   * perm-null histogram
#   * actual vs predicted firing trace (held-out fold)
#   * locations bar (subject's actual trajectory for that config)
#   * phases bar (e/m/l repeating 4 times)
#   * Time-lag bar (the strongest-coefficient regressor's activation)
# The top-5 coefficients (location, phase, lag, value) per chosen cell are
# written to a CSV so the figure annotations can be cross-checked.
if _run_sparse:
    print('\n=== Sparse DSR example cells ===')

    # Settings — edit to change which ROIs and how many cells per ROI.
    # Publication-figure scope: ACC only. Add other ROIs back if you also
    # want followup-style renders of those.
    SPARSE_DSR_TARGET_ROIS = ['ACC']
    SPARSE_DSR_N_PER_ROI = 5
    SPARSE_DSR_N_TOP_COEFS_TO_REPORT = 5
    # |coef| >= ACTIVE_THRESHOLD * max(|coef|) is counted as "active".
    SPARSE_DSR_ACTIVE_THRESHOLD = 0.10
    SPARSE_DSR_P_PERM_ALPHA = 0.05
    SPARSE_DSR_LAG_HIGHLIGHT_THRESHOLD = 0.50   # fraction of regressor peak
    PHASE_NAMES = ['early', 'middle', 'late']

    # Location palette (matches the 9-tile blue / teal / green grid the user
    # has been using elsewhere): rows 1-3 = blues, 4-6 = teals, 7-9 = greens.
    LOCATION_COLORS = {
        1: '#1f6ea8', 2: '#a4c9e4', 3: '#d8e5ec',   # blues  (dark -> light)
        4: '#1d4a52', 5: '#728e91', 6: '#bdd4c4',   # teals
        7: '#0f3a30', 8: '#447a62', 9: '#92c197',   # greens
    }
    # Locations whose tile is dark enough that the digit should be white.
    LOCATION_TEXT_LIGHT = {1, 4, 5, 7, 8}

    # Phases drawn in shades of burgundy (early = pale, late = darkest).
    PHASE_COLORS = {0: '#e6b3b3', 1: '#a83838', 2: '#5e1313'}
    PHASE_TEXT_LIGHT = {1, 2}

    # Lag-bar greys (publication style). Peak-firing box: light-grey fill +
    # thick black outline. Target box & arrow: black.
    PEAK_FIRING_COLOR = '0.72'
    PEAK_FIRING_EDGE  = 'black'
    PEAK_FIRING_EDGE_LW = 1.8
    TARGET_OUTLINE_COLOR = 'black'
    ARROW_COLOR = 'black'

    # Publication-figure colour overrides.
    REWARD_COLORS = {
        'A': '#C73734',   # red    (top of task wheel)
        'B': '#F39C5A',   # orange (right)
        'C': '#C1B3D6',   # light lavender (bottom)
        'D': '#6F5FA8',   # purple (left)
    }
    PUB_HIST_BG_COLOR  = '#bdd4c4'   # location-6 brightest green
    PUB_HIST_SIG_COLOR = '#0f3a30'   # location-7 darkest green
    PUB_PRED_COLOR     = '#9A383C'   # dark red from encoding-pub histogram

    SPARSE_OUT_DIR = os.path.join(OUT_DIR, 'fig24_sparse_dsr_examples')
    os.makedirs(SPARSE_OUT_DIR, exist_ok=True)

    # Data directory for behavioural data (mode locations per config).
    SPARSE_DATA_DIR = (
        '/Users/xpsy1114/Documents/projects/multiple_clocks/data/'
        'ephys_humans/derivatives'
    )

    # ``mc`` is not always imported by this script. Import lazily here.
    import sys
    sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/'
                       'multiple_clocks_repo')
    import mc
    from mc.plotting.cell_results import _draw_perm_hist

    # Reuse a previously-loaded diagnostics dict if available.
    try:
        sparse_diag = _diag_brain
    except NameError:
        try:
            sparse_diag = _diagnostics
        except NameError:
            diag_pkl = os.path.join(RESULT_DIR, 'diagnostics.pkl')
            print(f"  loading diagnostics: {diag_pkl}")
            with open(diag_pkl, 'rb') as f:
                sparse_diag = pickle.load(f)

    SPARSE_DSR_MODEL = 'dsr'

    def _decode_coef_idx(idx, n_phases=DSR_N_PHASES, n_lags=DSR_N_LAGS):
        loc = idx // (n_phases * n_lags)
        rem = idx % (n_phases * n_lags)
        phase = rem // n_lags
        lag = rem % n_lags
        return int(loc), int(phase), int(lag)

    def _sparseness(coefs, threshold=SPARSE_DSR_ACTIVE_THRESHOLD):
        a = np.abs(np.asarray(coefs, dtype=float))
        if a.max() <= 1e-12:
            return None
        sum_a = float(a.sum())
        sorted_a = np.sort(a)[::-1]
        n_active = int((a >= threshold * a.max()).sum())
        top1_share = float(sorted_a[0] / sum_a) if sum_a > 0 else np.nan
        top3_share = (float(sorted_a[:3].sum() / sum_a)
                      if sum_a > 0 and sorted_a.size >= 3 else np.nan)
        return dict(n_active=n_active, top1_share=top1_share,
                    top3_share=top3_share, max_abs=float(a.max()))

    # ── Step 1: rank candidates (perm-sig DSR cells, sorted by sparseness)
    print("  scoring perm-sig DSR cells by sparseness ...")
    cand_rows = []
    for sub_str, per_neuron in sparse_diag.items():
        for n_lab, per_model in per_neuron.items():
            d = per_model.get(SPARSE_DSR_MODEL)
            if d is None:
                continue
            roi = d.get('roi')
            p_perm = d.get('p_perm', np.nan)
            mean_r = d.get('mean_r', np.nan)
            if not (np.isfinite(p_perm)
                    and p_perm < SPARSE_DSR_P_PERM_ALPHA):
                continue
            r_per_fold = np.asarray(d.get('r_per_fold', []), dtype=float)
            if r_per_fold.size == 0 or not np.isfinite(r_per_fold).any():
                continue
            best_fold = int(np.nanargmax(r_per_fold))
            coefs_list = d.get('coefs', [])
            if best_fold >= len(coefs_list):
                continue
            best_coefs = np.asarray(coefs_list[best_fold], dtype=float)
            if best_coefs.size != DSR_N_COEFS:
                continue
            sp = _sparseness(best_coefs)
            if sp is None:
                continue
            top_idx = int(np.argmax(np.abs(best_coefs)))
            top_loc, top_phase, top_lag = _decode_coef_idx(top_idx)
            
            configs = d.get('configs', [])
            cand_rows.append({
                'subject': sub_str, 'neuron': n_lab, 'roi': roi,
                'mean_r': float(mean_r), 'p_perm': float(p_perm),
                'best_fold': best_fold,
                'best_fold_r': float(r_per_fold[best_fold]),
                'best_fold_test_config': (configs[best_fold]
                                          if best_fold < len(configs)
                                          else None),
                'n_active':          sp['n_active'],
                'top1_share':        sp['top1_share'],
                'top3_share':        sp['top3_share'],
                'max_abs_coef':      sp['max_abs'],
                'top_coef_loc_1idx': top_loc + 1,
                'top_coef_phase':    PHASE_NAMES[top_phase],
                'top_coef_lag':      top_lag,
                'top_coef_value':    float(best_coefs[top_idx]),
            })

    if not cand_rows:
        print("  no perm-sig DSR cells found — skipping fig 24.")
    else:
        cand_df = pd.DataFrame(cand_rows)
        # Per-ROI ranking: sparsest first (small n_active), then largest top1.
        ranked = (cand_df.sort_values(
                    ['roi', 'n_active', 'top1_share'],
                    ascending=[True, True, False])
                  .reset_index(drop=True))
        ranked_csv = os.path.join(SPARSE_OUT_DIR,
                                  'sparse_candidates_ranked.csv')
        ranked.to_csv(ranked_csv, index=False)
        print(f"  ranked {len(ranked)} candidates -> {ranked_csv}")

        # ── Step 2: prepare helpers + per-figure renderer ───────────
        SUBJECT_DATA_CACHE = {}

        def _get_sub_data(sub_str):
            if sub_str not in SUBJECT_DATA_CACHE:
                SUBJECT_DATA_CACHE[sub_str] = (
                    mc.analyse.helpers_human_cells.load_norm_data(
                        SPARSE_DATA_DIR, [sub_str]))
            return SUBJECT_DATA_CACHE[sub_str]

        def _config_locations(sub_str, config_str):
            """Per-bin (360,) mode location 1..9 for `config_str`."""
            sd = _get_sub_data(sub_str)
            key = f'sub-{sub_str}'
            if key not in sd:
                return None
            beh = sd[key]['beh'].copy().reset_index(drop=True)
            beh['config_str'] = (
                beh['loc_A'].astype(int).astype(str) + '-' +
                beh['loc_B'].astype(int).astype(str) + '-' +
                beh['loc_C'].astype(int).astype(str) + '-' +
                beh['loc_D'].astype(int).astype(str))
            idx = beh.index[(beh['config_str'] == config_str)
                            & (beh['correct'] == 1)].to_numpy()
            if len(idx) == 0:
                return None
            arr = sd[key]['locations'].iloc[idx].to_numpy()
            mode_vec = stats.mode(arr, axis=0, keepdims=False,
                                  nan_policy='omit').mode
            return np.asarray(mode_vec, dtype=int)

        # All in-bar text is drawn at this size (matches axis-label fonts).
        BAR_FONT_PT = 9

        def _draw_location_bar(ax, locations, font_pt=BAR_FONT_PT):
            """Solid colour blocks per consecutive-same-location run."""
            locs = np.asarray(locations, dtype=int)
            n = len(locs)
            i = 0
            while i < n:
                j = i
                while j < n and locs[j] == locs[i]:
                    j += 1
                loc = int(locs[i])
                color = LOCATION_COLORS.get(loc, '#888888')
                ax.axvspan(i, j, color=color, alpha=1.0, lw=0)
                if j - i > 8:
                    text_color = ('white' if loc in LOCATION_TEXT_LIGHT
                                  else 'black')
                    ax.text((i + j) / 2, 0.5, str(loc),
                            ha='center', va='center',
                            fontsize=font_pt, fontweight='bold',
                            color=text_color)
                i = j
            ax.set_xlim(0, n)
            ax.set_ylim(0, 1)

        def _draw_phase_bar(ax, n_states=4, n_phases=DSR_N_PHASES,
                            n_bins=360, font_pt=BAR_FONT_PT):
            """e/m/l × n_states burgundy bar."""
            bins_per_state = n_bins // n_states
            bins_per_phase = bins_per_state // n_phases
            letters = ['e', 'm', 'l']
            pos = 0
            for _ in range(n_states):
                for p in range(n_phases):
                    end = pos + bins_per_phase
                    ax.axvspan(pos, end, color=PHASE_COLORS[p],
                               alpha=1.0, lw=0)
                    text_color = ('white' if p in PHASE_TEXT_LIGHT
                                  else 'black')
                    ax.text((pos + end) / 2, 0.5, letters[p],
                            ha='center', va='center', fontsize=font_pt,
                            fontweight='bold', color=text_color)
                    pos = end
            ax.set_xlim(0, n_bins)
            ax.set_ylim(0, 1)

        def _find_target_phase_idx(locations, top_loc_0idx, top_phase_0idx,
                                   n_total_phases=DSR_N_LAGS,
                                   n_phases=DSR_N_PHASES, n_bins=360):
            """Phase position (0..11) where the preferred location is visited
            during the preferred sub-phase, or None.

            Uses *presence* (preferred loc appears in the 30-bin window) rather
            than mode-equals-preferred, because the mode criterion misses
            brief pass-throughs and produced spurious 'not visited' diagrams
            for cells whose preferred coefficient is well-supported.

            If multiple phase positions match, return the latest one (matches
            the publication example, where the target is on the right side of
            the trial).
            """
            bins_per_phase = n_bins // n_total_phases
            candidates = []
            locs = np.asarray(locations, dtype=int)
            for i in range(n_total_phases):
                if (i % n_phases) != top_phase_0idx:
                    continue
                start = i * bins_per_phase
                end = start + bins_per_phase
                window = locs[start:end] - 1
                if window.size == 0:
                    continue
                if (window == top_loc_0idx).any():
                    candidates.append(i)
            return candidates[-1] if candidates else None

        def _find_preferred_location_run(locations, target_phase_idx,
                                         top_loc_0idx,
                                         n_total_phases=DSR_N_LAGS,
                                         n_bins=360):
            """Return (start, end) of the contiguous run of the preferred
            location that overlaps with the target phase window. None if
            absent."""
            bins_per_phase = n_bins // n_total_phases
            target_start = target_phase_idx * bins_per_phase
            target_end = target_start + bins_per_phase
            locs = np.asarray(locations, dtype=int)
            is_pref = locs == (top_loc_0idx + 1)
            if not is_pref.any():
                return None
            diffs = np.diff(is_pref.astype(int))
            starts = np.where(diffs == 1)[0] + 1
            ends = np.where(diffs == -1)[0] + 1
            if is_pref[0]:
                starts = np.r_[0, starts]
            if is_pref[-1]:
                ends = np.r_[ends, len(is_pref)]
            for s, e in zip(starts, ends):
                if s < target_end and e > target_start:
                    return int(s), int(e)
            return None

        def _add_highlight_box(ax, x0, x1, y0=0.0, y1=1.0,
                               color=TARGET_OUTLINE_COLOR, lw=2.5,
                               z=10):
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, edgecolor=color, linewidth=lw, zorder=z,
            ))

        def _draw_lag_bar_explicit(ax, locations, top_loc_0idx,
                                   top_phase_0idx, top_lag,
                                   n_total_phases=DSR_N_LAGS,
                                   n_phases=DSR_N_PHASES, n_bins=360):
            """12-box lag grid showing peak firing -> target offset.

            Each box represents one of the 12 phases in the trial. The
            "Peak firing" box is solid red, the target box (preferred
            location × phase) is outlined red, and the boxes between are
            numbered 1..offset where offset = (12 - top_lag).

            Returns (peak_idx, target_idx) so the caller can also outline
            the target phase on the location and phase bars.
            """
            from matplotlib.patches import Rectangle
            bins_per_phase = n_bins // n_total_phases
            target_idx = _find_target_phase_idx(
                locations, top_loc_0idx, top_phase_0idx,
                n_total_phases=n_total_phases, n_phases=n_phases,
                n_bins=n_bins)

            # Empty 12-box grid first.
            for i in range(n_total_phases):
                start = i * bins_per_phase
                ax.add_patch(Rectangle(
                    (start, 0), bins_per_phase, 1,
                    fill=False, edgecolor='black', lw=0.8))

            if target_idx is None:
                # With the presence-based criterion this should not happen
                # for cells with a strong preferred coefficient; if it does,
                # we silently fall through to an empty lag grid rather than
                # printing the (now misleading) "not visited" message.
                ax.set_xlim(0, n_bins)
                ax.set_ylim(-0.9, 1.9)
                return None, None

            offset = (DSR_N_LAGS - top_lag) % DSR_N_LAGS
            peak_idx = (target_idx - offset) % n_total_phases
            peak_start = peak_idx * bins_per_phase
            target_start = target_idx * bins_per_phase

            # Peak firing: grey fill with thick black outline.
            ax.add_patch(Rectangle(
                (peak_start, 0), bins_per_phase, 1,
                facecolor=PEAK_FIRING_COLOR, edgecolor=PEAK_FIRING_EDGE,
                linewidth=PEAK_FIRING_EDGE_LW, zorder=8))
            # Target: thick black outline.
            ax.add_patch(Rectangle(
                (target_start, 0), bins_per_phase, 1,
                fill=False, edgecolor=TARGET_OUTLINE_COLOR,
                linewidth=PEAK_FIRING_EDGE_LW, zorder=10))

            # Number the boxes between peak and target (1, 2, ..., offset).
            for k in range(1, offset + 1):
                idx_ = (peak_idx + k) % n_total_phases
                cx = idx_ * bins_per_phase + bins_per_phase / 2
                ax.text(cx, 0.5, str(k),
                        ha='center', va='center',
                        fontsize=BAR_FONT_PT, fontweight='bold',
                        color='black')

            # "Peak firing" label below the peak box.
            ax.text(peak_start + bins_per_phase / 2, -0.5,
                    'Peak\nfiring',
                    ha='center', va='top',
                    fontsize=9, fontweight='bold', color='black')

            # Arrow from peak to target (linear if no wrap, else two segments
            # so it stays inside the panel).
            arrow_y = 1.5
            peak_cx = peak_start + bins_per_phase / 2
            target_cx = target_start + bins_per_phase / 2
            if peak_idx + offset <= n_total_phases - 1:
                ax.annotate(
                    '', xy=(target_cx, arrow_y),
                    xytext=(peak_cx, arrow_y),
                    arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                    lw=1.8),
                    annotation_clip=False)
            else:
                # Wrap: arrow goes from peak to right edge, then from left
                # edge to target. Draw two arrows.
                ax.annotate(
                    '', xy=(n_bins, arrow_y),
                    xytext=(peak_cx, arrow_y),
                    arrowprops=dict(arrowstyle='-', color=ARROW_COLOR,
                                    lw=1.8),
                    annotation_clip=False)
                ax.annotate(
                    '', xy=(target_cx, arrow_y),
                    xytext=(0, arrow_y),
                    arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                    lw=1.8),
                    annotation_clip=False)

            ax.set_xlim(0, n_bins)
            ax.set_ylim(-0.9, 1.9)
            return int(peak_idx), int(target_idx)

        def _draw_task_grid_legend(ax, config_str,
                                   reward_colors=REWARD_COLORS):
            """3x3 spatial grid with coloured A/B/C/D circles at the four
            reward positions of ``config_str`` (e.g. '7-3-1-9' → A at tile 7,
            B at tile 3, C at tile 1, D at tile 9). Tile-number convention:
            row 0 (top) = 1,2,3; row 1 = 4,5,6; row 2 (bottom) = 7,8,9."""
            try:
                rewards = list(zip('ABCD',
                                   (int(x) for x in str(config_str).split('-'))))
            except Exception:
                ax.text(0.5, 0.5, '?', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                return
            # Each cell coloured by its location-bar colour (blue/teal/green),
            # so the legend matches the locations strip in the trace area.
            for row in range(3):
                for col in range(3):
                    loc_n = row * 3 + col + 1
                    ax.add_patch(plt.Rectangle(
                        (col, 2 - row), 1, 1,
                        facecolor=LOCATION_COLORS.get(loc_n, '#cccccc'),
                        edgecolor='white', lw=0.8))
            for letter, loc in rewards:
                row = (loc - 1) // 3
                col = (loc - 1) % 3
                cx = col + 0.5
                cy = (2 - row) + 0.5
                ax.add_patch(plt.Circle(
                    (cx, cy), 0.32,
                    facecolor=reward_colors[letter],
                    edgecolor='white', lw=0.8, zorder=3))
                ax.text(cx, cy, letter,
                        ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='white', zorder=4)
            ax.set_xlim(-0.05, 3.05)
            ax.set_ylim(-0.05, 3.05)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ('top', 'right', 'left', 'bottom'):
                ax.spines[sp].set_visible(False)


        def _draw_phase_wheel_legend(ax, phase_colors=PHASE_COLORS,
                                     reward_colors=REWARD_COLORS,
                                     font_pt=BAR_FONT_PT):
            """Donut-style legend: 4 quadrants A/B/C/D, each split into
            three e/m/l burgundy wedges in clockwise order.

            Quadrants in clock terms (12 o'clock = top):
              A: 12 → 3 o'clock   (A.e at 0–1, A.m at 1–2, A.l at 2–3)
              B:  3 → 6 o'clock   (B.e at 3–4, …)
              C:  6 → 9 o'clock
              D:  9 → 12 o'clock
            Reward letter sits at the centre of each quadrant (= 1:30,
            4:30, 7:30, 10:30 o'clock)."""
            from matplotlib.patches import Wedge
            center = (0.5, 0.5)
            r_outer = 0.42
            # (reward, phase, theta1, theta2) — matplotlib CCW; 0° = 3 o'clock.
            wedges = [
                ('A', 'e',  60,  90),   # 12–1 o'clock
                ('A', 'm',  30,  60),   #  1–2
                ('A', 'l',   0,  30),   #  2–3 (reward A reached at 3)
                ('B', 'e', -30,   0),   #  3–4
                ('B', 'm', -60, -30),   #  4–5
                ('B', 'l', -90, -60),   #  5–6 (reward B at 6)
                ('C', 'e',-120, -90),   #  6–7
                ('C', 'm',-150,-120),   #  7–8
                ('C', 'l',-180,-150),   #  8–9 (reward C at 9)
                ('D', 'e', 150, 180),   #  9–10
                ('D', 'm', 120, 150),   # 10–11
                ('D', 'l',  90, 120),   # 11–12 (reward D at 12)
            ]
            phase_idx = {'e': 0, 'm': 1, 'l': 2}
            for _rew, ph, t1, t2 in wedges:
                ax.add_patch(Wedge(
                    center, r_outer, t1, t2,
                    facecolor=phase_colors[phase_idx[ph]],
                    edgecolor='white', linewidth=0.4))
            # Labels at quadrant centres (1:30, 4:30, 7:30, 10:30).
            for letter, angle in [('A',  45), ('B', -45),
                                  ('C',-135), ('D', 135)]:
                rad = np.radians(angle)
                lx = center[0] + 0.52 * np.cos(rad)
                ly = center[1] + 0.52 * np.sin(rad)
                ax.text(lx, ly, letter,
                        ha='center', va='center',
                        fontsize=font_pt, fontweight='bold',
                        color=reward_colors[letter])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ('top', 'right', 'left', 'bottom'):
                ax.spines[sp].set_visible(False)


        def _draw_reward_markers(ax, n_bins, n_states=4,
                                 reward_colors=REWARD_COLORS,
                                 draw_labels=True, lw=1.8, alpha=0.95,
                                 font_pt=BAR_FONT_PT):
            """Vertical lines at the END of each state (= reward position),
            coloured A/B/C/D. With labels=True, the letter sits just to the
            left of each line, indicating which reward is reached there.

            Convention: the trial is split into 4 states of equal length;
            reward A is reached at the end of state 1 (x = bins_per_state),
            reward B at the end of state 2, etc.
            """
            letters = ('A', 'B', 'C', 'D')
            bins_per_state = n_bins // n_states
            for k, letter in enumerate(letters):
                x_end = (k + 1) * bins_per_state
                ax.axvline(x_end, color=reward_colors[letter],
                           lw=lw, alpha=alpha, zorder=2)
                if draw_labels:
                    # text just to the LEFT of the line so it sits inside
                    # the segment whose reward it marks
                    ax.text(x_end - 0.02 * n_bins, 1.04, letter,
                            transform=ax.get_xaxis_transform(),
                            ha='right', va='bottom',
                            fontsize=font_pt, fontweight='bold',
                            color=reward_colors[letter])


        def _draw_perm_hist_pub(ax, diag, bins=12, font_pt=BAR_FONT_PT):
            """Publication-styled perm-null histogram: fewer/thicker green
            bars + a heavy dark-green empirical-r line. No legend; the
            empirical-r value goes in the main title."""
            perm_rs = np.asarray(diag.get('perm_rs', []), dtype=float)
            perm_rs = perm_rs[np.isfinite(perm_rs)]
            emp_r = float(diag.get('mean_r', np.nan))
            finite_vals = np.concatenate([
                perm_rs,
                np.array([emp_r] if np.isfinite(emp_r) else [], dtype=float),
            ])
            lim = (max(0.05, 1.05 * float(np.max(np.abs(finite_vals))))
                   if finite_vals.size else 1.0)
            edges = np.linspace(-lim, lim, bins + 1)
            if perm_rs.size:
                ax.hist(perm_rs, bins=edges, color=PUB_HIST_BG_COLOR,
                        edgecolor='white', linewidth=0.6)
            ax.axvline(0, color='0.35', lw=0.8)
            if np.isfinite(emp_r):
                ax.axvline(emp_r, color=PUB_HIST_SIG_COLOR, lw=2.6)
            ax.set_xlim(-lim, lim)
            ax.set_xlabel('mean Pearson r', fontsize=font_pt)
            ax.set_ylabel('# perm',         fontsize=font_pt)
            ax.tick_params(labelsize=font_pt)
            ax.set_title('permutation null', fontsize=font_pt)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)


        def _hide_spines(ax, sides=('top', 'left', 'right', 'bottom')):
            for s in sides:
                ax.spines[s].set_visible(False)

        def _render_one(diag, sub_str, info, save_path, rank):
            best_fold = info['best_fold']
            r_per_fold = np.asarray(diag['r_per_fold'])
            test_config = diag['configs'][best_fold]
            coefs = np.asarray(diag['coefs'][best_fold], dtype=float)
            y_test = np.asarray(diag['y_test_per_fold'][best_fold])
            y_pred = np.asarray(diag['y_pred_per_fold'][best_fold])

            locations = _config_locations(sub_str, test_config)
            if locations is None or len(locations) != len(y_test):
                print(f"    skip {info['neuron']}: cannot resolve locations "
                      f"for config {test_config!r}.")
                return False, None

            # Build the DSR design matrix for this config.
            walked = (locations - 1).astype(int).tolist()
            try:
                _, _, _, _, dsr_matrix, _, _ = (
                    mc.simulation.predictions.model_DSR(
                        locations=walked,
                        no_phase_neurons=DSR_N_PHASES))
            except Exception as e:
                print(f"    model_DSR failed for {info['neuron']}: {e}")
                return False, None

            top_idx = int(np.argmax(np.abs(coefs)))
            top_loc, top_phase, top_lag = _decode_coef_idx(top_idx)

            # Prediction if only the strongest coefficient were used. Same
            # scale as y_pred up to the elastic-net intercept / scaling.
            coefs_top = np.zeros_like(coefs)
            coefs_top[top_idx] = coefs[top_idx]
            y_pred_top = dsr_matrix.T @ coefs_top                # (n_bins,)

            # Top-N coefficients for the per-cell CSV row.
            order = (np.argsort(np.abs(coefs))[::-1]
                     [:SPARSE_DSR_N_TOP_COEFS_TO_REPORT])
            top_coefs_rows = []
            for k, idx_ in enumerate(order, start=1):
                L, P, K = _decode_coef_idx(int(idx_))
                top_coefs_rows.append({
                    'rank_in_cell':  k,
                    'coef_index':    int(idx_),
                    'location_1idx': L + 1,
                    'phase':         PHASE_NAMES[P],
                    'phase_0idx':    P,
                    'lag':           K,
                    'coef':          float(coefs[idx_]),
                    'abs_coef':      float(abs(coefs[idx_])),
                })

            # ── Publication layout (~14 cm × 10.5 cm, fits the subpanel-e
            # slot of the multi-panel figure at 2/3 A4 width). ───────────
            n_bins_per_trial = len(y_test)
            FIG_W_CM, FIG_H_CM = 14.0, 11.5
            FONT_PT = 9
            with plt.rc_context({'font.family': 'sans-serif',
                                 'font.sans-serif':
                                     ['Arial', 'DejaVu Sans'],
                                 'font.size': FONT_PT}):
                fig = plt.figure(figsize=(FIG_W_CM / 2.54,
                                          FIG_H_CM / 2.54))
                # 11 rows × 16 cols. Histogram top-left; trace + 3 bars on
                # the right. Below the histogram: task-config grid and
                # phase-wheel legends side by side. The lag bar gets only
                # 2 rows (was 4) — visibly thinner per user request.
                # top=0.86 leaves a comfortable gap below the figure title.
                gs = fig.add_gridspec(11, 16,
                                      hspace=0.50, wspace=0.55,
                                      left=0.07, right=0.97,
                                      top=0.82, bottom=0.07)
                ax_hist  = fig.add_subplot(gs[0:5, 0:4])
                ax_ts    = fig.add_subplot(gs[0:5, 4:16])
                ax_loc   = fig.add_subplot(gs[5, 4:16], sharex=ax_ts)
                ax_phase = fig.add_subplot(gs[6, 4:16], sharex=ax_ts)
                ax_lag   = fig.add_subplot(gs[7:9, 4:16], sharex=ax_ts)
                ax_grid  = fig.add_subplot(gs[7:11, 0:2])
                ax_wheel = fig.add_subplot(gs[7:11, 2:4])

                # --- Histogram (greens, no legend) ---
                _draw_perm_hist_pub(ax_hist, diag, bins=22)

                # --- Trace ---
                x = np.arange(n_bins_per_trial)
                ax_ts.plot(x, y_test, color='black', lw=0.7, zorder=3)
                ax_ts2 = ax_ts.twinx()
                ax_ts2.plot(x, y_pred,
                            color=PUB_PRED_COLOR, lw=1.2, alpha=0.95,
                            zorder=4)
                ax_ts2.plot(x, y_pred_top,
                            color=PUB_PRED_COLOR, lw=1.2, alpha=0.95,
                            linestyle='--', zorder=4)
                ax_ts.set_xlim(0, n_bins_per_trial)
                ax_ts.set_ylabel('neuron (a.u.)', fontsize=FONT_PT)
                ax_ts2.set_ylabel('predicted',
                                  color=PUB_PRED_COLOR,
                                  fontsize=FONT_PT)
                ax_ts2.tick_params(labelcolor=PUB_PRED_COLOR,
                                   labelsize=FONT_PT)
                ax_ts.tick_params(labelsize=FONT_PT)
                ax_ts.spines['top'].set_visible(False)
                ax_ts2.spines['top'].set_visible(False)
                # No per-axis subtitle — the held-out config and r are
                # already in the main figure title, and putting them here
                # collided with the A/B/C/D reward labels above the trace.

                # --- Reward markers (A/B/C/D) across trace + bars ---
                _draw_reward_markers(ax_ts, n_bins_per_trial,
                                     draw_labels=True,
                                     lw=0.9, alpha=0.85)
                for axx in (ax_loc, ax_phase, ax_lag):
                    _draw_reward_markers(axx, n_bins_per_trial,
                                         draw_labels=False,
                                         lw=0.9, alpha=0.85)

                # --- Location bar ---
                _draw_location_bar(ax_loc, locations)
                ax_loc.set_yticks([]); ax_loc.set_xticks([])
                ax_loc.set_ylabel('locations', rotation=0,
                                  ha='right', va='center',
                                  fontsize=FONT_PT)
                _hide_spines(ax_loc)

                # --- Phase bar ---
                _draw_phase_bar(ax_phase, n_bins=n_bins_per_trial)
                ax_phase.set_yticks([]); ax_phase.set_xticks([])
                ax_phase.set_ylabel('phases', rotation=0,
                                    ha='right', va='center',
                                    fontsize=FONT_PT)
                _hide_spines(ax_phase)

                # --- Time-lag bar ---
                peak_idx, target_idx = _draw_lag_bar_explicit(
                    ax_lag, locations, top_loc, top_phase, top_lag,
                    n_bins=n_bins_per_trial,
                )
                ax_lag.set_yticks([])
                ax_lag.set_ylabel('Time-lag', rotation=0,
                                  ha='right', va='center',
                                  fontsize=FONT_PT)
                ax_lag.set_xlabel('time bin', fontsize=FONT_PT)
                ax_lag.tick_params(labelsize=FONT_PT)
                _hide_spines(ax_lag, sides=('top', 'left', 'right'))

                # --- Outline target on locations & phases ---
                if target_idx is not None:
                    bins_per_phase = (n_bins_per_trial // DSR_N_LAGS)
                    target_start = target_idx * bins_per_phase
                    target_end   = target_start + bins_per_phase
                    _add_highlight_box(ax_phase,
                                       target_start, target_end, lw=1.4)
                    pref_run = _find_preferred_location_run(
                        locations, target_idx, top_loc,
                        n_bins=n_bins_per_trial,
                    )
                    if pref_run is not None:
                        _add_highlight_box(ax_loc,
                                           pref_run[0], pref_run[1],
                                           lw=1.4)

                # --- Mini legends ---
                _draw_task_grid_legend(ax_grid, test_config)
                _draw_phase_wheel_legend(ax_wheel)

                # --- One-line title with cell metadata ---
                fig.suptitle(
                    f"#{rank} sparse DSR cell in {diag.get('roi')}  |  "
                    f"sub-{sub_str} {info['neuron']}  |  "
                    f"mean r = {diag['mean_r']:.3f}  |  "
                    f"p_perm = {diag['p_perm']:.3f}  |  "
                    f"top coef: loc={top_loc + 1}, "
                    f"phase={PHASE_NAMES[top_phase]}, lag={top_lag}  |  "
                    f"n_active={info['n_active']}",
                    fontsize=FONT_PT, y=0.96)

                save_path_pdf = save_path.replace('.png', '.pdf')
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                fig.savefig(save_path_pdf, bbox_inches='tight')
                plt.close(fig)
            return True, top_coefs_rows

        # ── Step 3: render per-ROI top-N sparsest cells ─────────────
        all_top_coefs = []
        rois_to_render = ([r for r in SPARSE_DSR_TARGET_ROIS
                           if r in ranked['roi'].unique()]
                          if SPARSE_DSR_TARGET_ROIS
                          else sorted(ranked['roi'].dropna().unique()))
        for roi in rois_to_render:
            roi_df = ranked[ranked['roi'] == roi].head(SPARSE_DSR_N_PER_ROI)
            if roi_df.empty:
                print(f"  {roi}: no candidates.")
                continue
            roi_dir = os.path.join(SPARSE_OUT_DIR, roi)
            os.makedirs(roi_dir, exist_ok=True)
            print(f"  rendering top {len(roi_df)} sparse cells in {roi} ...")
            for rank, (_, row) in enumerate(roi_df.iterrows(), start=1):
                sub_str = row['subject']
                n_lab = row['neuron']
                diag = (sparse_diag.get(sub_str, {})
                        .get(n_lab, {})
                        .get(SPARSE_DSR_MODEL))
                if diag is None:
                    continue
                info = {
                    'neuron':    n_lab,
                    'best_fold': int(row['best_fold']),
                    'n_active':  int(row['n_active']),
                }
                fname = (f'fig24_top{rank}_dsr_{roi}_sub-{sub_str}_'
                         f'{n_lab}.png').replace('/', '_')
                save_path = os.path.join(roi_dir, fname)
                ok, tc_rows = _render_one(diag, sub_str, info,
                                          save_path, rank)
                if ok:
                    for tc in tc_rows:
                        tc.update({
                            'subject':              sub_str,
                            'neuron':               n_lab,
                            'roi':                  roi,
                            'rank_in_roi':          rank,
                            'best_fold_r':          float(row['best_fold_r']),
                            'best_fold_test_config':
                                row['best_fold_test_config'],
                            'n_active':             int(row['n_active']),
                            'top1_share':           float(row['top1_share']),
                            'mean_r':               float(row['mean_r']),
                            'p_perm':               float(row['p_perm']),
                        })
                        all_top_coefs.append(tc)
                    print(f"    rendered #{rank} {n_lab} "
                          f"(n_active={info['n_active']}) -> {fname}")

        if all_top_coefs:
            top_coefs_csv = os.path.join(SPARSE_OUT_DIR,
                                         'sparse_top_coefs.csv')
            pd.DataFrame(all_top_coefs).to_csv(top_coefs_csv, index=False)
            print(f"Wrote per-cell top coefficients -> {top_coefs_csv}")


print(f"\nAll outputs in: {OUT_DIR}")
