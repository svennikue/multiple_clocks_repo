#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reload a DSR-RSA run and produce two summary figures using the shared
helpers in `mc.plotting.results`. Pure plotting — no recomputation.

Reads:
  - results_summary_combos.csv   (t, beta, q per ROI × combo × sub_model)
  - perm_null_draws/perm_<ROI>.pkl
       perm_results_combo[test][combo]['beta'] : list of len n_perms,
                                                  each entry shape (n_subs,)
       combo_models[combo]                     : ordered sub-model names
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
sys.path.insert(0, str(REPO))
from mc.plotting.results import (
    plot_roi_tstat_heatmap,
    plot_per_roi_stat_histograms,
)
from mc.plotting.cell_results import SHOWGIRL2_DISCRETE


# ── User settings ─────────────────────────────────────────────────────
DATA = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives')
RSA_RUN_DIR = DATA / 'group/DSR_RSA_simple_ROI/2026-06-29_20-09-54-DSR-corrected'
TEST_VARIANT = 'split_halves_z'

# Heatmap panels: (display label, combo, sub_model)
PANELS = [
    ('DSR\n(full)',     'ctrl_dsrFULL',     'dsr_fmri'),
    ('state',           'ctrl_dsrFULL',     'state'),
    ('location',        'ctrl_dsrFULL',     'location'),
]
PANELS_all = [
    ('DSR\n(full)',     'ctrl_dsrFULL',     'dsr_fmri'),
    ('DSR\n(future)',   'ctrl_dsrFUT',      'dsr_fmri_fut'),
    ('DSR\n(informed)', 'ctrl_dsrInformed', 'dsr_fmri_informed'),
    ('state',           'ctrl_dsrFULL',     'state'),
    ('location',        'ctrl_dsrFULL',     'location'),
]
ROI_ORDER = ['ACC', 'medialOFC', 'PCC', 'Parahippocampal',
             'HC_anterior', 'HC_mid', 'EC']
ROI_COLOURS = {
    'EC':              SHOWGIRL2_DISCRETE[0],
    'ACC':             SHOWGIRL2_DISCRETE[1],
    'HC_anterior':     SHOWGIRL2_DISCRETE[2],
    'PCC':             SHOWGIRL2_DISCRETE[3],
    'medialOFC':       SHOWGIRL2_DISCRETE[4],
    'Parahippocampal': SHOWGIRL2_DISCRETE[5],
    'HC_mid':          SHOWGIRL2_DISCRETE[6],
}


def main():
    df = pd.read_csv(RSA_RUN_DIR / 'results_summary_combos.csv')
    df = df[df.test == TEST_VARIANT]
    rois = [r for r in ROI_ORDER if r in df.roi.unique()]
    out_dir = RSA_RUN_DIR / 'replot_summary'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Heatmap ───────────────────────────────────────────────────────
    T = np.full((len(rois), len(PANELS_all)), np.nan)
    Q = np.full((len(rois), len(PANELS_all)), np.nan)
    P = np.full((len(rois), len(PANELS_all)), np.nan)
    for j, (_, combo, sub) in enumerate(PANELS_all):
        for i, roi in enumerate(rois):
            row = df[(df.combo == combo) & (df.sub_model == sub) & (df.roi == roi)]
            if not row.empty:
                T[i, j] = row['t'].iloc[0]
                P[i,j] = row['p_perm'].iloc[0]
                if 'q_fdr_per_combo' in row.columns:
                    Q[i, j] = row['q_fdr_per_combo'].iloc[0]

    plot_roi_tstat_heatmap(
        T, rois, [p[0] for p in PANELS_all],
        q_matrix=Q,
        panel_groups=[(tuple(range(3)), 'PiYG_r', 'DSR variants'),
                      ((3,),            'RdBu_r', 'State'),
                      ((4,),            'RdBu_r', 'Location')],
        title=f'ROI × regressor — {TEST_VARIANT}  (stars = BH-FDR q per combo)',
        cbar_label='t vs 0', fig_size_cm=(9.5, 12.0),
        save_path=str(out_dir / f'heatmap_roi_x_regressor_FDR_corr{TEST_VARIANT}_DSR_vars'),
    )

    plot_roi_tstat_heatmap(
        T, rois, [p[0] for p in PANELS_all],
        q_matrix=P,
        panel_groups=[(tuple(range(3)), 'PiYG_r', 'DSR variants'),
                      ((3,),            'RdBu_r', 'State'),
                      ((4,),            'RdBu_r', 'Location')],
        title=f'ROI × regressor — {TEST_VARIANT}  (stars = perm p per combo)',
        cbar_label='t vs 0', fig_size_cm=(9.5, 12.0),
        save_path=str(out_dir / f'heatmap_roi_x_regressor_uncorr_{TEST_VARIANT}_DSR_vars'),
    )
    
    
    
    #import pdb; pdb.set_trace()
    # drop the 2 DSR ones that i'm not interested in
    T = np.full((len(rois), len(PANELS)), np.nan)
    Q = np.full((len(rois), len(PANELS)), np.nan)
    P = np.full((len(rois), len(PANELS)), np.nan)
    for j, (_, combo, sub) in enumerate(PANELS):
        for i, roi in enumerate(rois):
            row = df[(df.combo == combo) & (df.sub_model == sub) & (df.roi == roi)]
            if not row.empty:
                T[i, j] = row['t'].iloc[0]
                P[i,j] = row['p_perm'].iloc[0]
                if 'q_fdr_per_combo' in row.columns:
                    Q[i, j] = row['q_fdr_per_combo'].iloc[0]


    plot_roi_tstat_heatmap(
        T, rois, [p[0] for p in PANELS],
        q_matrix=Q,
        panel_groups=[((0,), 'RdBu_r', 'DSR'),
                      ((1,),            'RdBu_r', 'State'),
                      ((2,),            'RdBu_r', 'Location')],
        title=f'ROI × regressor — {TEST_VARIANT}  (stars = BH-FDR q per combo)',
        cbar_label='t vs 0', fig_size_cm=(7.5, 12.0),
        save_path=str(out_dir / f'heatmap_roi_x_regressor_FDR_corr{TEST_VARIANT}'),
    )
    # import pdb; pdb.set_trace()
    plot_roi_tstat_heatmap(
        T, rois, [p[0] for p in PANELS],
        q_matrix=P,
        panel_groups=[((0,), 'RdBu_r', 'DSR'),
                      ((1,),            'RdBu_r', 'State'),
                      ((2,),            'RdBu_r', 'Location')],
        title=f'ROI × regressor — {TEST_VARIANT}  (stars = perm p per combo)',
        cbar_label='t vs 0', fig_size_cm=(7.5, 12.0),
        save_path=str(out_dir / f'heatmap_roi_x_regressor_uncorr_{TEST_VARIANT}'),
    )
    print(f"Wrote heatmap_roi_x_regressor_{TEST_VARIANT}")

    # ── Per-ROI histograms ───────────────────────────────────────────
    # For each panel of interest: read perm-β and empirical-β straight
    # from perm_<ROI>.pkl, no OLS.
    perm_dir = RSA_RUN_DIR / 'perm_null_draws'

    for label, combo, sub in [('DSR (informed)', 'ctrl_dsrInformed', 'dsr_fmri_informed'),
                              ('state',          'ctrl_dsrFULL',     'state'),
                              ('location',       'ctrl_dsrFULL',     'location')]:
        rows = []
        emp_per_roi = {}
        for roi in rois:
            pkl = perm_dir / f'perm_{roi}.pkl'
            if not pkl.exists():
                continue
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            sub_list = d['combo_models'][combo]
            k = sub_list.index(sub)
            # perm dict is keyed by full test (e.g. split_halves_z); the
            # 'empirical_combo_results_z' dict is itself the z-scored
            # variant and is keyed by the base test (split_halves).
            base_test = TEST_VARIANT[:-2] if TEST_VARIANT.endswith('_z') else TEST_VARIANT
            perm_betas = np.asarray(d['perm_results_combo'][TEST_VARIANT][combo]['beta'])
            emp_per_roi[roi] = float(d['empirical_combo_results_z'][base_test][combo][1][k])
            for b in perm_betas[:, k]:
                rows.append({'roi': roi, 'beta': float(b)})
        hist_df = pd.DataFrame(rows)
        if hist_df.empty:
            print(f"  no perm data for {label}")
            continue
        
        q_per_roi = {}
        for roi in rois:
            row = df[(df.combo == combo) & (df.sub_model == sub) & (df.roi == roi)]
            if not row.empty and 'q_fdr_per_combo' in row.columns:
                q_per_roi[roi] = float(row['q_fdr_per_combo'].iloc[0])

        plot_per_roi_stat_histograms(
            hist_df,
            stat_col='beta',
            stat_label=f'perm β ({label})',
            roi_order=rois,
            roi_colours=ROI_COLOURS,
            q_per_roi=q_per_roi,
            empirical_per_roi=emp_per_roi,
            mark_sig_neurons=None,  # group-level RSA: no per-cell sig
            n_cols=4, panel_w_cm=3.0, panel_h_cm=1.5,
            save_path=str(out_dir / f'hist_perROI_{sub}_{TEST_VARIANT}'),
        )
        print(f"Wrote hist_perROI_{sub}_{TEST_VARIANT}")

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == '__main__':
    main()
