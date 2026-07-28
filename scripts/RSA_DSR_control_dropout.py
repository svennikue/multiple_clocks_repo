#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick diagnostic — leave-N-out control-regressor dropout for the DSR RSA.

Enumerates every subset of `CONTROLS` (all sizes from 0 to len(CONTROLS)),
crosses each subset with each DSR variant in `DSR_VARIANTS`, then runs
`RSA_DSR_ROIs_simple.py` as a subprocess with these custom combos at
`N_PERMS` permutations. Heavy diagnostic outputs (glassbrains, per-cell
RDM diagnostics, pub figures) are disabled so the run finishes fast.

After the subprocess completes, this script:
  * loads the combo results,
  * applies BH-FDR within each (control-subset × DSR-variant) family
    across the 7 ROIs (mirroring `confirmatory_fdr_per_combo.csv`),
  * extracts an ACC-focused pivot,
  * draws a sorted heatmap of ACC q-vals per control subset × DSR variant.

EDIT and re-run:
  * `CONTROLS`     — base set of control regressor names
  * `DSR_VARIANTS` — DSR sub-models to test
  * `N_PERMS`      — perm count for the subprocess
  * `TARGET_ROI`   — which ROI's pivot to render

Outputs land in:
  DATA_DIR/group/DSR_RSA_simple_ROI/<YYYY-MM-DD_HH-MM-SS>_dropout/

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── User-configurable settings ────────────────────────────────────────
REPO = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
DATA = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives')
RSA_SCRIPT = REPO / 'scripts/RSA_DSR_ROIs_simple.py'
OUT_BASE = DATA / 'group/DSR_RSA_simple_ROI'

CONTROLS = ['state', 'location', 'l2_norm',
             'bttn_curr', 'bttn_next', 'reward_path']
DSR_VARIANTS = ['dsr_fmri', 'dsr_fmri_fut', 'dsr_fmri_informed']

N_PERMS = 50
TARGET_ROI = 'ACC'
PRIMARY_TEST = 'split_halves_z'
ALPHA = 0.05

# 7 ROIs over which BH-FDR is applied per (combo × DSR variant) family.
# Must match what RSA_DSR_ROIs_simple.py actually runs.
ALL_ROIS = ['ACC', 'EC', 'Parahippocampal',
             'HC_anterior', 'HC_mid', 'medialOFC', 'PCC', 'Precuneus']


# ── Build the combo dict ───────────────────────────────────────────────
def build_combos(controls, dsr_variants):
    out = {}
    for size in range(len(controls) + 1):
        for subset in combinations(controls, size):
            sub = list(subset)
            ctrl_tag = '_'.join(sub) if sub else 'NOCTRL'
            for dsr in dsr_variants:
                out[f'{ctrl_tag}__{dsr}'] = sub + [dsr]
    return out


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]; n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qok = np.empty(n); qok[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = qok
    return q


def stars(q):
    if not np.isfinite(q): return ''
    if q < 0.001: return '***'
    if q < 0.01:  return '**'
    if q < 0.05:  return '*'
    if q < 0.10:  return '·'
    return ''


def main():
    combos = build_combos(CONTROLS, DSR_VARIANTS)
    print(f"Built {len(combos)} (control-subset × DSR-variant) combos "
          f"from {len(CONTROLS)} controls × {len(DSR_VARIANTS)} DSR variants")
    n_subsets = len(combos) // len(DSR_VARIANTS)
    print(f"  = {n_subsets} control subsets × {len(DSR_VARIANTS)} DSR variants")

    run_tag = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_dropout'
    out_dir = OUT_BASE / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    combos_json = out_dir / 'combos.json'
    with open(combos_json, 'w') as f:
        json.dump(combos, f, indent=2)

    # Snapshot the diagnostic settings for reproducibility.
    with open(out_dir / 'diagnostic_config.json', 'w') as f:
        json.dump({
            'controls':       CONTROLS,
            'dsr_variants':   DSR_VARIANTS,
            'n_perms':        N_PERMS,
            'target_roi':     TARGET_ROI,
            'primary_test':   PRIMARY_TEST,
            'alpha':          ALPHA,
            'all_rois':       ALL_ROIS,
            'n_combos':       len(combos),
            'n_subsets':      n_subsets,
            'run_tag':        run_tag,
            'rsa_script':     str(RSA_SCRIPT),
        }, f, indent=2)

    env = os.environ.copy()
    env['RSA_DROPOUT_COMBOS_JSON'] = str(combos_json)
    env['RSA_DROPOUT_N_PERMS']     = str(N_PERMS)
    env['RSA_DROPOUT_OUTDIR']      = str(out_dir)
    env['RSA_DROPOUT_DISABLE_HEAVY'] = '1'

    print(f"\nLaunching RSA subprocess (this may take a while — "
          f"{len(combos)} combos × {len(ALL_ROIS)} ROIs × {N_PERMS} perms)…")
    print(f"  python {RSA_SCRIPT}")
    print(f"  output dir = {out_dir}\n")
    result = subprocess.run([sys.executable, str(RSA_SCRIPT)], env=env)
    if result.returncode != 0:
        print(f"\nERROR: subprocess returned code {result.returncode}.")
        return 1

    # Load + post-process ----------------------------------------------
    combo_csv = out_dir / 'results_summary_combos.csv'
    if not combo_csv.exists():
        print(f"\nERROR: expected {combo_csv} but it doesn't exist.")
        return 1
    df = pd.read_csv(combo_csv)
    df = df[df.test.eq(PRIMARY_TEST)
             & df.sub_model.isin(DSR_VARIANTS)].copy()
    print(f"\nLoaded {len(df)} rows from {combo_csv.name}")

    # Per-(combo, DSR-variant) BH-FDR across 7 ROIs --------------------
    df['q_fdr_dropout'] = np.nan
    for (combo, sm), g in df.groupby(['combo', 'sub_model'], sort=False):
        df.loc[g.index, 'q_fdr_dropout'] = bh_fdr(
            g['p_perm'].to_numpy(dtype=float))
    df.to_csv(out_dir / 'all_combos_results.csv', index=False)
    print(f"Wrote all_combos_results.csv")

    # ACC pivot --------------------------------------------------------
    acc = df[df.roi == TARGET_ROI].copy()
    # Re-derive control subset label (everything up to the '__' separator).
    acc['ctrl_subset'] = acc['combo'].str.rsplit('__', n=1).str[0]
    acc['n_ctrls'] = acc['ctrl_subset'].apply(
        lambda s: 0 if s == 'NOCTRL' else len(s.split('_')))
    pivot = acc.pivot_table(
        index=['n_ctrls', 'ctrl_subset'],
        columns='sub_model',
        values=['beta', 'p_perm', 'q_fdr_dropout'],
        aggfunc='first',
    )
    pivot.columns = [f'{m}_{sm}' for m, sm in pivot.columns]
    q_cols = [c for c in pivot.columns if c.startswith('q_fdr_dropout_')]
    pivot['min_q_across_dsr'] = pivot[q_cols].min(axis=1, skipna=True)
    pivot = pivot.sort_values(['min_q_across_dsr', 'n_ctrls'])
    pivot.to_csv(out_dir / f'{TARGET_ROI}_pivot.csv')
    print(f"Wrote {TARGET_ROI}_pivot.csv  ({len(pivot)} rows)")

    # Heatmap ---------------------------------------------------------
    H = pivot[[f'q_fdr_dropout_{sm}' for sm in DSR_VARIANTS]].to_numpy()
    fig_h = max(6.0, 0.18 * H.shape[0])
    fig, ax = plt.subplots(figsize=(5, fig_h), constrained_layout=True)
    im = ax.imshow(H, aspect='auto', cmap='Reds_r', vmin=0, vmax=0.3,
                   interpolation='nearest')
    ax.set_xticks(range(len(DSR_VARIANTS)))
    ax.set_xticklabels(DSR_VARIANTS, rotation=20, ha='right', fontsize=8)
    row_labels = []
    for (n_ct, ctrl), _ in pivot.iterrows():
        if ctrl == 'NOCTRL':
            row_labels.append(f'[0] (no ctrls)')
        else:
            row_labels.append(f'[{n_ct}] ' + ctrl.replace('_', '+'))
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=5)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            s = stars(H[i, j])
            if s:
                col = 'white' if H[i, j] < 0.05 else 'black'
                ax.text(j, i, s, ha='center', va='center',
                        fontsize=6, color=col, fontweight='bold')
    ax.set_xlabel('DSR sub-model')
    ax.set_title(
        f'{TARGET_ROI} — BH-FDR q across 7 ROIs per (control-subset × DSR)\n'
        f'{N_PERMS} perms, sorted by best q;  ·=q<.10  *<.05  **<.01  ***<.001',
        fontsize=9,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('q_FDR (within-combo, 7 ROIs)', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'{TARGET_ROI}_dropout_heatmap.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Wrote {TARGET_ROI}_dropout_heatmap.{{pdf,png}}")

    # Print the top-N best ACC results for the user ------------------
    print(f"\nTop-15 control subsets for {TARGET_ROI} (sorted by best q across DSR variants):")
    top = pivot.head(15)
    for sm in DSR_VARIANTS:
        b_col = f'beta_{sm}'; q_col = f'q_fdr_dropout_{sm}'
        if b_col not in top.columns: continue
    show_cols = [f'{m}_{sm}' for sm in DSR_VARIANTS for m in ('beta', 'q_fdr_dropout')]
    print(top[show_cols].round(4).to_string())
    print(f"\nDone. All outputs under: {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
