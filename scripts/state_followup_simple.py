#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-hoc state-tuning analysis from `encoding_analysis_simple.py` output.

For each cell with a significant fit of the **state-only** encoding model
(per-cell p_perm < 0.05), pulls the 4-state coefficient vector
(A, B, C, D) from the saved `diagnostics.pkl`, averages across CV folds,
and produces per-ROI overview figures.

Three figures (mirroring `scripts/plotting_human_cells.py` for the bar
and polar conventions; matches `encoding_publication_panels.py` for the
ROI list):

  F1 — preferred-state bar charts per ROI: raw counts + within-ROI
       proportions (matches `plotting_human_cells.py:pref_counts_per_roi`).
  F2 — cells × 4-state coefficient heatmap, sorted by peak state, with
       a sub-panel per ROI (matches the goal-progress-tuning style).
  F3 — clover-style polar plot of mean state tuning per ROI (matches
       `plot_state_polar_clock`).

Outputs land in
  <ENC_DIR>/followup/state_followup_<timestamp>/

@author: Svenja Kuchenhoff
"""

import os
import json
import pickle
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats


# ════════════════════════════════════════════════════════════════════════
# Settings
# ════════════════════════════════════════════════════════════════════════

ENC_DIR = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/'
           'ephys_humans/derivatives/group/encoding_analysis_simple/'
           '2026-06-05_17-58-57')
DATA_DIR = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/'
            'ephys_humans/derivatives')

# Per-cell significance threshold (matches plotting_human_cells.py).
ALPHA = 0.05
STATE_MODEL = 'state'
STATES = ['A', 'B', 'C', 'D']

# Same 9-ROI list as the DSR + encoding-panel pipelines. Visual is kept
# as a sanity-check row at the bottom (no FDR family role here).
ROI_LABEL_COLUMN = 'alt_final_roi'
ROI_ORDER = [
    'ACC', 'medial_CC',
    'HC_anterior', 'HC_mid',
    'EC', 'Parahippocampal',
    'PCC',
    'medialOFC',
    'Visual',
]

# Colours match the A/B/C/D palette used in plotting_human_cells.py
# (`plot_state_polar_clock`).
COLOURS = ['#F15A29', '#F7931E', '#C7C6E2', '#6B60AA']

RUN_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUT_DIR = os.path.join(ENC_DIR, 'followup', f'state_followup_{RUN_TAG}')
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
print(f"Output: {OUT_DIR}")


# ════════════════════════════════════════════════════════════════════════
# Load encoding results + ROI table
# ════════════════════════════════════════════════════════════════════════

print("Loading encoding_results.csv …")
df = pd.read_csv(os.path.join(ENC_DIR, 'encoding_results.csv'))
df_state = df[df['model'] == STATE_MODEL].copy()
print(f"  state-model rows total: {len(df_state)}")

# Per-cell p_perm < ALPHA filter, restricted to the ROIs of interest.
sig = df_state[(df_state['p_perm'] < ALPHA)
               & (df_state['roi'].isin(ROI_ORDER))].copy()
print(f"  significant cells (p_perm<{ALPHA}) in target ROIs: {len(sig)}")


def parse_neuron_label(label):
    """'01_07-07-chan120-EC' → ('01', 7)."""
    try:
        sub_str, rest = label.split('_', 1)
        cell_idx_str = rest.split('-', 1)[0]
        return sub_str.zfill(2), int(cell_idx_str)
    except Exception:
        return None, None


sub_list, cell_idx_list = zip(*sig['neuron'].apply(parse_neuron_label))
sig['sub'] = list(sub_list)
sig['cell_idx'] = list(cell_idx_list)


# ════════════════════════════════════════════════════════════════════════
# Load diagnostics.pkl and pull per-cell state coefficients
# ════════════════════════════════════════════════════════════════════════

print("Loading diagnostics.pkl …")
diag_path = os.path.join(ENC_DIR, 'diagnostics.pkl')
with open(diag_path, 'rb') as f:
    diag = pickle.load(f)
print(f"  loaded {sum(len(v) for v in diag.values())} cells across "
      f"{len(diag)} subjects.")


def get_state_coefs(sub, neuron_label):
    """Returns (4,) array of CV-averaged state coefs, or None if missing."""
    sub_key = sub.zfill(2) if sub is not None else None
    if sub_key is None or sub_key not in diag:
        return None
    nm = diag[sub_key]
    if neuron_label not in nm or STATE_MODEL not in nm[neuron_label]:
        return None
    coefs = np.asarray(nm[neuron_label][STATE_MODEL]['coefs'],
                        dtype=float)
    if coefs.ndim != 2 or coefs.shape[1] != 4:
        return None
    return np.nanmean(coefs, axis=0)


print("Extracting state coefficients …")
coef_mat = np.full((len(sig), 4), np.nan)
for i, (_, row) in enumerate(sig.iterrows()):
    c = get_state_coefs(row['sub'], row['neuron'])
    if c is not None:
        coef_mat[i] = c
ok = np.isfinite(coef_mat).all(axis=1)
print(f"  cells with valid coefs: {int(ok.sum())} / {len(sig)}")

sig = sig.loc[ok].reset_index(drop=True)
coef_mat = coef_mat[ok]
for i, s in enumerate(STATES):
    sig[f'coef_{s}'] = coef_mat[:, i]

# Preferred state = argmax of the CV-averaged coefficients (max positive,
# per the chosen definition — matches plotting_human_cells.py / the
# user's convention).
sig['pref_state'] = np.array(STATES)[np.argmax(coef_mat, axis=1)]

# Across-fold consistency (used to pick exemplar cells later).
def _consistency(per_fold_str):
    try:
        arr = np.asarray(json.loads(per_fold_str), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return np.nan
        return float(arr.mean()) - float(arr.std(ddof=1))
    except Exception:
        return np.nan


sig['consistency'] = sig['r_per_fold'].apply(_consistency)

# Per-cell summary table.
sig_out = sig[['neuron', 'roi', 'sub', 'cell_idx',
                'mean_r', 'p_perm', 'consistency',
                'coef_A', 'coef_B', 'coef_C', 'coef_D',
                'pref_state']].copy()
csv_path = os.path.join(OUT_DIR, 'state_followup_per_cell.csv')
sig_out.to_csv(csv_path, index=False)
print(f"  wrote {csv_path}")


# ════════════════════════════════════════════════════════════════════════
# All-cells preferred-state table (background of the bar plot) + FDR
# ════════════════════════════════════════════════════════════════════════

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok_mask = np.isfinite(p)
    if not ok_mask.any():
        return q
    pv = p[ok_mask]; n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qok = np.empty(n); qok[order] = np.clip(ranked, 0.0, 1.0)
    q[ok_mask] = qok
    return q


# All state-model cells in the target ROIs (numerator + denominator).
all_state_in_target = df_state[df_state['roi'].isin(ROI_ORDER)].copy()
sub_all, idx_all = zip(*all_state_in_target['neuron'].apply(parse_neuron_label))
all_state_in_target['sub'] = list(sub_all)
all_state_in_target['cell_idx'] = list(idx_all)

# Pull state coefs for ALL cells (for preferred-state classification of
# the denominator bar).
print("Extracting state coefs for ALL cells (for background bars) …")
coef_all = np.full((len(all_state_in_target), 4), np.nan)
for i, (_, row) in enumerate(all_state_in_target.iterrows()):
    c = get_state_coefs(row['sub'], row['neuron'])
    if c is not None:
        coef_all[i] = c
ok_all = np.isfinite(coef_all).all(axis=1)
all_state_in_target = all_state_in_target.loc[ok_all].reset_index(drop=True)
coef_all = coef_all[ok_all]
all_state_in_target['pref_state'] = np.array(STATES)[np.argmax(coef_all, axis=1)]

# BH-FDR over per-cell p_perm across the 7 ROIs.
all_state_in_target['p_fdr_cell'] = bh_fdr(
    all_state_in_target['p_perm'].to_numpy())
n_sig_fdr = int((all_state_in_target['p_fdr_cell'] < ALPHA).sum())
print(f"  cells significant at q < {ALPHA} (BH-FDR over {len(all_state_in_target)} cells): "
      f"{n_sig_fdr}")
all_state_in_target.to_csv(
    os.path.join(OUT_DIR, 'state_followup_all_cells.csv'), index=False)


# ════════════════════════════════════════════════════════════════════════
# Per-ROI counts + chi-square test of state-preference bias
# ════════════════════════════════════════════════════════════════════════

def counts_table(df_in, roi_col='roi', state_col='pref_state',
                  states=STATES, roi_order=ROI_ORDER):
    rois = [r for r in roi_order if r in set(df_in[roi_col])]
    counts = (df_in.groupby([roi_col, state_col])
                    .size().unstack(fill_value=0)
                    .reindex(index=rois, columns=states, fill_value=0))
    return counts


counts_sig = counts_table(sig)
counts_all = counts_table(all_state_in_target, roi_col='roi',
                            state_col='pref_state')
counts_sig.to_csv(os.path.join(OUT_DIR, 'counts_sig_per_roi.csv'))
counts_all.to_csv(os.path.join(OUT_DIR, 'counts_all_per_roi.csv'))

# Chi-square test per ROI: is the distribution of preferred states across
# {A, B, C, D} different from uniform within the SIGNIFICANT cells?
stats_rows = []
for roi, row in counts_sig.iterrows():
    obs = row.to_numpy()
    if obs.sum() < 4:
        stats_rows.append({'roi': roi, 'n_cells': int(obs.sum()),
                            'chi2': None, 'p_chi2': None})
        continue
    exp = np.full(4, obs.sum() / 4.0)
    chi2, p = stats.chisquare(obs, exp)
    stats_rows.append({
        'roi':     roi, 'n_cells': int(obs.sum()),
        'A': int(obs[0]), 'B': int(obs[1]),
        'C': int(obs[2]), 'D': int(obs[3]),
        'chi2':    float(chi2), 'p_chi2': float(p),
    })
stats_df = pd.DataFrame(stats_rows)
# BH-FDR over the 7 ROI χ² tests.
stats_df['p_chi2_fdr'] = bh_fdr(stats_df['p_chi2'].astype(float).to_numpy())
stats_df.to_csv(os.path.join(OUT_DIR, 'state_bias_chi2_per_roi.csv'),
                 index=False)
print("\nχ² uniformity per ROI (BH-FDR across 7 ROIs):")
print(stats_df.to_string(index=False))


# Binomial test per (ROI × state): is the proportion of sig cells in this
# (ROI, state) cell above the chance level (= ALPHA × n_total_in_state_in_roi)?
binom_rows = []
for roi in counts_sig.index:
    for s in STATES:
        n_sig   = int(counts_sig.loc[roi, s])
        n_total = int(counts_all.loc[roi, s]) if roi in counts_all.index else 0
        if n_total == 0:
            binom_rows.append({'roi': roi, 'state': s, 'n_sig': n_sig,
                                'n_total': 0, 'p_binom': None})
            continue
        res = stats.binomtest(n_sig, n_total, p=ALPHA, alternative='greater')
        binom_rows.append({'roi': roi, 'state': s, 'n_sig': n_sig,
                            'n_total': n_total,
                            'p_binom': float(res.pvalue)})
binom_df = pd.DataFrame(binom_rows)
# Two FDR alternatives: global (28 tests) and per-ROI (4 tests).
binom_df['p_binom_fdr_global']  = bh_fdr(binom_df['p_binom']
                                          .astype(float).to_numpy())
binom_df['p_binom_fdr_per_roi'] = np.nan
for roi in binom_df['roi'].unique():
    mask = binom_df['roi'] == roi
    binom_df.loc[mask, 'p_binom_fdr_per_roi'] = bh_fdr(
        binom_df.loc[mask, 'p_binom'].astype(float).to_numpy())
binom_df.to_csv(os.path.join(OUT_DIR,
                  'state_proportion_binom_per_roi_state.csv'),
                 index=False)
print("\nBinomial test per (ROI × state) — proportion of sig cells > chance "
      f"({ALPHA}); BH-FDR across 28 cells (global) and 4 per ROI:")
print(binom_df.to_string(index=False))


# ════════════════════════════════════════════════════════════════════════
# F1 — Preferred-state bar charts per ROI
# ════════════════════════════════════════════════════════════════════════

def render_F1():
    rois = counts_sig.index.tolist()
    if not rois:
        print("  F1: no ROIs — skipped.")
        return
    n_rois = len(rois)
    # Grouped bars: 4 sub-bars per ROI, one per state. Each sub-bar shows
    # the all-cells total in light grey behind the sig-cell coloured bar.
    width = 0.20
    pad   = width * 4 + 0.3   # x-spacing per ROI cluster
    x_centres = np.arange(n_rois) * pad
    fig, ax = plt.subplots(figsize=(max(13, 2.1 * n_rois), 6.5),
                            constrained_layout=True)

    bar_max = 0
    for s_i, s in enumerate(STATES):
        offsets = (s_i - 1.5) * width
        bar_x = x_centres + offsets
        n_all = counts_all.reindex(rois)[s].to_numpy()
        n_sig = counts_sig[s].to_numpy()
        # Background: total cells preferring this state (light grey).
        ax.bar(bar_x, n_all, width=width * 0.95,
                color='#dadada', edgecolor='black', linewidth=0.4,
                zorder=2)
        # Foreground: significant cells (coloured).
        ax.bar(bar_x, n_sig, width=width * 0.95,
                color=COLOURS[s_i], edgecolor='black', linewidth=0.4,
                zorder=3)
        bar_max = max(bar_max, n_all.max() if n_all.size else 0)
        # White state letter inside each background bar.
        for i, (xi, n_tot) in enumerate(zip(bar_x, n_all)):
            if n_tot <= 0:
                continue
            ax.text(xi, max(n_tot * 0.5, 1.2), s,
                    ha='center', va='center', fontsize=16,
                    fontweight='bold', color='white', zorder=5)
        # Binomial significance star above sig bar.
        for i, roi in enumerate(rois):
            row = binom_df[(binom_df['roi'] == roi)
                            & (binom_df['state'] == s)]
            if row.empty:
                continue
            p_g = float(row['p_binom_fdr_global'].iloc[0]) \
                if pd.notna(row['p_binom_fdr_global'].iloc[0]) else np.nan
            sig_str = ('***' if p_g < 0.001 else '**' if p_g < 0.01
                       else '*' if p_g < 0.05 else '')
            if sig_str:
                ax.text(bar_x[i],
                          max(n_sig[i], n_all[i]) * 1.04 + 0.4,
                          sig_str, ha='center', va='bottom',
                          fontsize=18, color='black', zorder=6)

    # Chance line: 5% of each ROI's total cells per state.
    for i, roi in enumerate(rois):
        for s_i, s in enumerate(STATES):
            n_tot = int(counts_all.loc[roi, s]) if s in counts_all.columns else 0
            chance = ALPHA * n_tot
            offset = (s_i - 1.5) * width
            xlo = x_centres[i] + offset - width / 2 * 0.95
            xhi = x_centres[i] + offset + width / 2 * 0.95
            ax.hlines(chance, xlo, xhi, colors='0.25',
                       linestyles=(0, (2, 2)), linewidth=1.0, zorder=4)

    ax.set_xticks(x_centres)
    ax.set_xticklabels(rois, rotation=20, ha='right', fontsize=14)
    ax.set_ylabel('# cells', fontsize=15)
    ax.set_title('Preferred-state cell counts per ROI\n'
                  f'(grey = all cells, coloured = state-encoding sig cells '
                  f'(p_perm < {ALPHA}); dotted = {int(ALPHA*100)}% chance line; '
                  f'stars = binomial > chance, BH-FDR over {n_rois}×4 cells)',
                  fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, bar_max * 1.18 + 1)

    stem = os.path.join(FIG_DIR, 'F1_pref_state_per_roi')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


def render_F1b_within_roi():
    """Within-ROI composition — 100% stacked bars of the SIGNIFICANT cells.
    Kept as a separate figure for the previous role of the F1 right panel."""
    rois = counts_sig.index.tolist()
    if not rois:
        return
    n_rois = len(rois)
    x = np.arange(n_rois)
    fig, ax_prop = plt.subplots(figsize=(max(9, 1.4 * n_rois), 5.5),
                                  constrained_layout=True)
    props = counts_sig.div(counts_sig.sum(axis=1).replace(0, np.nan),
                              axis=0).fillna(0.0)
    bottom = np.zeros(len(rois))
    for s_i, s in enumerate(STATES):
        ax_prop.bar(x, props[s].to_numpy(), bottom=bottom,
                     color=COLOURS[s_i], edgecolor='black',
                     label=f'state {s}')
        bottom = bottom + props[s].to_numpy()
    ax_prop.set_xticks(x)
    ax_prop.set_xticklabels(rois, rotation=20, ha='right', fontsize=13)
    ax_prop.set_ylabel('Proportion of significant cells', fontsize=13)
    ax_prop.tick_params(axis='y', labelsize=11)
    ax_prop.set_ylim(0, 1)
    # White state letter centered in each stacked segment.
    bottom = np.zeros(len(rois))
    for s_i, s in enumerate(STATES):
        for i in range(len(rois)):
            h = props.iloc[i, s_i]
            if h >= 0.06:
                ax_prop.text(x[i], bottom[i] + h / 2.0, s,
                                ha='center', va='center', fontsize=14,
                                fontweight='bold', color='white', zorder=5)
        bottom = bottom + props[s].to_numpy()
    ax_prop.set_title('Within-ROI composition of significant cells '
                        '(100% stacked)', fontsize=13)
    ax_prop.spines[['top', 'right']].set_visible(False)
    stem = os.path.join(FIG_DIR, 'F1b_within_roi_composition')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


# ════════════════════════════════════════════════════════════════════════
# F2 — cells × 4-state coefficient heatmap, sorted by argmax, per ROI
# ════════════════════════════════════════════════════════════════════════

def render_F2():
    rois = counts_sig.index.tolist()
    rois = [r for r in rois if counts_sig.loc[r].sum() > 0]
    if not rois:
        print("  F2: no ROIs — skipped.")
        return
    n = len(rois)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.6, nrows * 3.2),
                              squeeze=False, constrained_layout=True)
    # Symmetric scale across ALL ROIs so colours are comparable.
    z_mat_all = []
    roi_to_mat = {}
    for roi in rois:
        sub = sig[sig['roi'] == roi]
        c = sub[[f'coef_{s}' for s in STATES]].to_numpy()
        # z-score per cell so plots highlight tuning shape rather than scale.
        mu = c.mean(axis=1, keepdims=True)
        sd = c.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        cz = (c - mu) / sd
        order = np.argsort(np.argmax(cz, axis=1))
        cz_sorted = cz[order]
        roi_to_mat[roi] = cz_sorted
        z_mat_all.append(cz_sorted)
    z_all = np.vstack(z_mat_all)
    vmax = float(np.nanpercentile(np.abs(z_all), 99)) if z_all.size else 1.0

    for ax_idx, roi in enumerate(rois):
        ax = axes[ax_idx // ncols, ax_idx % ncols]
        m = roi_to_mat[roi]
        im = ax.imshow(m, aspect='auto', cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax,
                        interpolation='nearest')
        ax.set_xticks(range(4))
        ax.set_xticklabels(STATES, fontsize=9)
        ax.set_yticks([])
        ax.set_title(f'{roi}  (n = {m.shape[0]})', fontsize=9)
        if ax_idx == 0:
            ax.set_ylabel('cells (sorted by peak state)', fontsize=8)
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis('off')
    cb = fig.colorbar(im, ax=axes.ravel().tolist(),
                       fraction=0.02, pad=0.02, shrink=0.8)
    cb.set_label('coefficient (z-scored per cell)', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle('F2 — state coefficient heatmap per ROI '
                  '(rows = significant cells, sorted by argmax)',
                  fontsize=10)
    stem = os.path.join(FIG_DIR, 'F2_state_coef_heatmap_per_roi')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


# ════════════════════════════════════════════════════════════════════════
# F3 — clover / polar plots per ROI
# ════════════════════════════════════════════════════════════════════════

def plot_state_polar_4(ax, coefs4, title, rlim=None):
    """Polar plot of a 4-state tuning vector. Coloured wedges + outer
    step-function curve matching plot_state_polar_clock from
    plotting_human_cells.py. A→3 o'clock, B→6, C→9, D→12.
    """
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    n_segs = 360
    theta = np.linspace(0, 2 * np.pi, n_segs, endpoint=False)
    # Expand 4 coefficients to 360 bins (constant within each quadrant).
    expanded = np.repeat(coefs4, n_segs // 4)
    if rlim is None:
        rmin = float(np.nanmin(expanded))
        rmax = float(np.nanmax(expanded))
    else:
        rmin, rmax = rlim
    if rmax == rmin:
        rmax = rmin + 1e-6
    ax.set_ylim(rmin, rmax)
    # Step curve per quadrant.
    edges = np.linspace(0, n_segs, 5, dtype=int)
    for i in range(4):
        s, e = edges[i], edges[i + 1]
        ax.plot(theta[s:e], expanded[s:e],
                color=COLOURS[i], lw=3)
    # Shaded wedges up to the quadrant value.
    for i in range(4):
        s, e = edges[i], edges[i + 1]
        ang = ((s + e) / 2.0 / n_segs) * 2 * np.pi
        width = ((e - s) / n_segs) * 2 * np.pi
        m = float(coefs4[i])
        if np.isfinite(m):
            ax.bar(ang, max(0, m - rmin), width=width, bottom=rmin,
                    color=COLOURS[i], alpha=0.25, edgecolor='none',
                    zorder=0, align='center')
    label_angles = np.deg2rad([0, 90, 180, 270])  # A,B,C,D = 12,3,6,9
    # Match plotting_human_cells.py letter positions.
    label_pos = ['A', 'B', 'C', 'D']
    pad = 0.18 * (rmax - rmin)
    label_r = rmax + pad
    for lab, ang, col in zip(label_pos, label_angles, COLOURS):
        if np.isclose(ang, 0):
            ha, va = 'center', 'bottom'
        elif np.isclose(ang, np.pi / 2):
            ha, va = 'left', 'center'
        elif np.isclose(ang, np.pi):
            ha, va = 'center', 'top'
        else:
            ha, va = 'right', 'center'
        ax.text(ang, label_r, lab, ha=ha, va=va,
                fontsize=16, fontweight='bold', color=col, clip_on=False)
    ax.set_xticks([])
    ax.set_title(title, fontsize=10, pad=24, y=1.12)
    ax.grid(True, alpha=0.3)


from scipy.ndimage import gaussian_filter1d
import mc


def smooth_circular(arr, sigma=4):
    extended = np.concatenate([arr, arr, arr])
    return gaussian_filter1d(extended, sigma=sigma)[len(arr):2 * len(arr)]


def plot_smoothed_360_polar(ax, trace360, title, rlim=None, label_fs=18):
    """Polar plot of a 360-bin smoothed firing trace coloured by state quadrant.
    Faithful port of plot_state_polar_clock from plotting_human_cells.py.
    """
    n_bins = len(trace360)
    quarter = n_bins // 4
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    theta = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    # Coloured curve per state quadrant.
    for i in range(4):
        s, e = i * quarter, (i + 1) * quarter
        ax.plot(theta[s:e], trace360[s:e],
                color=COLOURS[i], lw=3)
    if rlim is None:
        rmin = float(np.nanmin(trace360))
        rmax = float(np.nanmax(trace360))
    else:
        rmin, rmax = rlim
    if rmax == rmin:
        rmax = rmin + 1e-6
    ax.set_ylim(rmin, rmax)
    # Shaded wedges up to per-state mean.
    quad_means = trace360.reshape(4, quarter).mean(axis=1)
    centres = np.linspace(0, 2 * np.pi, 4, endpoint=False) + (np.pi / 4)
    for i, m in enumerate(quad_means):
        ax.bar(centres[i], max(0, m - rmin),
                width=np.pi / 2, bottom=rmin,
                color=COLOURS[i], alpha=0.20, edgecolor='none',
                zorder=0, align='center')
    # State letters A/B/C/D at the four quadrant outside positions.
    label_angles = np.deg2rad([45, 135, 225, 315])  # centres of A,B,C,D
    pad = 0.22 * (rmax - rmin)
    label_r = rmax + pad
    for lab, ang, col in zip(['A', 'B', 'C', 'D'], label_angles, COLOURS):
        ax.text(ang, label_r, lab, ha='center', va='center',
                fontsize=label_fs, fontweight='bold',
                color=col, clip_on=False)
    ax.set_xticks([])
    ax.set_title(title, fontsize=11, pad=20, y=1.10)
    ax.grid(True, alpha=0.25)


def _pick_exemplar(roi, k=1):
    """Pick the top-k significant state cells in `roi` by a 'consistency'
    proxy: mean(r_per_fold) − std(r_per_fold). High mean + low variance
    means the state pattern reliably generalises to every held-out config.
    """
    sub = sig[(sig['roi'] == roi)
                & np.isfinite(sig['consistency'])].copy()
    if sub.empty:
        return []
    sub = sub.sort_values('consistency', ascending=False)
    return sub.head(k).to_dict(orient='records')


def render_F3_exemplars():
    """One exemplar cell per ROI. Three polar panels per cell: (i) overall
    mean across all correct trials, (ii) mean within configuration #1,
    (iii) mean within configuration #2. Together they demonstrate that
    the state pattern generalises across two different reward-location
    configurations.
    """
    rois = [r for r in ROI_ORDER if r in set(sig['roi'])]
    if not rois:
        print("  F3: no ROIs — skipped.")
        return
    # Cache load_norm_data per subject to avoid repeats.
    subj_cache = {}
    print("  loading raw neuron data for exemplar cells …")
    rows = []
    for roi in rois:
        ex = _pick_exemplar(roi, k=1)
        if not ex:
            continue
        ex = ex[0]
        sub = ex['sub']
        if sub not in subj_cache:
            subj_cache[sub] = mc.analyse.helpers_human_cells.load_norm_data(
                DATA_DIR, [sub])
        data = subj_cache[sub]
        key = f'sub-{sub}'
        if key not in data:
            continue
        # Find the neuron name in the loaded dict.
        neurons = data[key]['normalised_neurons']
        match = None
        for n_name in neurons:
            if n_name in ex['neuron'] or ex['neuron'] in n_name:
                match = n_name; break
        if match is None:
            continue
        beh = data[key]['beh']
        # Two configs with the most correct trials.
        grids = beh[['loc_A', 'loc_B', 'loc_C', 'loc_D']].to_numpy()
        uniq, inv, cnts = np.unique(grids, axis=0,
                                      return_inverse=True,
                                      return_counts=True)
        order = np.argsort(cnts)[::-1]
        top2 = order[:2]
        config_traces = []
        config_labels = []
        for c_i in top2:
            mask = (inv == c_i)
            arr = neurons[match].to_numpy()[mask]
            mean_trace = np.nanmean(arr, axis=0)
            config_traces.append(smooth_circular(mean_trace, sigma=4))
            config_labels.append('-'.join(str(int(v)) for v in uniq[c_i]))
        mean_all = smooth_circular(
            np.nanmean(neurons[match].to_numpy(), axis=0), sigma=4)

        rows.append({
            'roi':            roi,
            'neuron':         ex['neuron'],
            'mean_r':         float(ex['mean_r']),
            'consistency':    float(ex['consistency']),
            'pref_state':     ex['pref_state'],
            'mean_all':       mean_all,
            'cfg_traces':     config_traces,
            'cfg_labels':     config_labels,
        })
        print(f"    {roi}: {ex['neuron']}  (mean_r={ex['mean_r']:.3f}, "
              f"pref={ex['pref_state']}, cfgs={config_labels})")

    if not rows:
        print("  F3: no exemplar cells could be loaded — skipped.")
        return
    n_rois = len(rows)
    fig, axes = plt.subplots(n_rois, 3,
                              figsize=(3 * 3.4, n_rois * 3.4),
                              squeeze=False,
                              subplot_kw=dict(projection='polar'),
                              constrained_layout=True)
    for i, ex in enumerate(rows):
        # Shared rlim across the 3 panels of a single cell so they're
        # directly comparable.
        all_traces = np.concatenate([ex['mean_all']] + ex['cfg_traces'])
        rlim = (float(np.nanmin(all_traces)) * 1.05
                if np.nanmin(all_traces) < 0 else
                float(np.nanmin(all_traces)) * 0.95,
                float(np.nanmax(all_traces)) * 1.08)
        plot_smoothed_360_polar(
            axes[i, 0], ex['mean_all'],
            f"{ex['roi']}  —  {ex['neuron'].split('-')[-1]}\n"
            f"all configs  (mean_r = {ex['mean_r']:.3f}, "
            f"pref = {ex['pref_state']})",
            rlim=rlim)
        for j, (trace, label) in enumerate(zip(ex['cfg_traces'],
                                                ex['cfg_labels'])):
            plot_smoothed_360_polar(
                axes[i, j + 1], trace,
                f"config {label}",
                rlim=rlim)
    fig.suptitle('F3 — exemplar state-encoding cells per ROI\n'
                  '(smoothed 360-bin firing; same r-limit within each row '
                  'across the three configs)', fontsize=12)
    stem = os.path.join(FIG_DIR, 'F3_exemplar_cells_per_roi')
    fig.savefig(stem + '.pdf', bbox_inches='tight')
    fig.savefig(stem + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


# ════════════════════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════════════════════

print("\n========== Figures ==========")
render_F1()
render_F1b_within_roi()
render_F2()
render_F3_exemplars()


# ════════════════════════════════════════════════════════════════════════
# JSON archive
# ════════════════════════════════════════════════════════════════════════

archive = {
    'meta': {
        'timestamp':         datetime.now().isoformat(timespec='seconds'),
        'enc_run':           ENC_DIR,
        'alpha':             ALPHA,
        'state_model':       STATE_MODEL,
        'roi_label_column':  ROI_LABEL_COLUMN,
        'roi_order':         ROI_ORDER,
        'states':            STATES,
        'colours':           COLOURS,
        'n_total_sig_cells': int(len(sig)),
    },
    'counts_per_roi':           counts_sig.reset_index().to_dict(
        orient='records'),
    'state_bias_chi2_per_roi':  stats_df.to_dict(orient='records'),
}
out_json = os.path.join(OUT_DIR, 'state_followup_results.json')
with open(out_json, 'w') as f:
    json.dump(archive, f, indent=2,
              default=lambda o: float(o) if hasattr(o, 'item') else str(o))
print(f"\nWrote {out_json}")
print(f"All outputs under: {OUT_DIR}")
