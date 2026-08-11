#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-figure builder for a DSR-RSA run.

Exposes ``make_publication_figures(run_dir, ...)`` which reads:
  - ``results_summary_combos.csv``   (t, p_perm, q per ROI × combo × sub_model)
  - ``perm_null_draws/perm_<ROI>.pkl`` (per-perm β + empirical β)

and writes ROI × regressor t-heatmaps and per-ROI perm-β histograms into
``run_dir/<out_subdir>/`` (default ``pub_figures_v2``).

Pure plotting — no recomputation. Called automatically at the end of
``scripts/RSA_DSR_ROIs_simple.py`` for every run; also runs standalone
against a specified older run.
"""
from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from mc.plotting.results import (
    plot_roi_tstat_heatmap,
    plot_per_roi_stat_histograms,
)
from mc.plotting.cell_results import SHOWGIRL2_DISCRETE


# ── Canonical ROI order + colours (match CLAUDE.md conventions) ──────
DEFAULT_ROI_ORDER = ['mPFC', 'mOFC', 'PCC', 'PHC',
                     'HC_anterior', 'HC_mid', 'EC']

DEFAULT_ROI_COLOURS = {
    'EC':          SHOWGIRL2_DISCRETE[0],
    'mPFC':        SHOWGIRL2_DISCRETE[1],
    'HC_mid':      SHOWGIRL2_DISCRETE[2],
    'PCC':         SHOWGIRL2_DISCRETE[3],
    'mOFC':        SHOWGIRL2_DISCRETE[4],
    'HC_anterior': '#a30d6c',
    'PHC':         '#23677E',
}

# Panels for the ROI × regressor heatmaps.
DEFAULT_PANELS_ALL = [
    ('DSR\n(full)',     'ctrl_dsrFULL',     'dsr_fmri'),
    ('DSR\n(future)',   'ctrl_dsrFUT',      'dsr_fmri_fut'),
    ('DSR\n(informed)', 'ctrl_dsrInformed', 'dsr_fmri_informed'),
    ('state',           'ctrl_dsrFULL',     'state'),
    ('location',        'ctrl_dsrFULL',     'location'),
]
DEFAULT_PANELS_CORE = [
    ('DSR\n(full)', 'ctrl_dsrFULL', 'dsr_fmri'),
    ('state',       'ctrl_dsrFULL', 'state'),
    ('location',    'ctrl_dsrFULL', 'location'),
]

# Histogram panels: (label, combo, sub_model).
DEFAULT_HIST_PANELS = [
    ('DSR (informed)', 'ctrl_dsrInformed', 'dsr_fmri_informed'),
    ('DSR (full)',     'ctrl_dsrFULL',     'dsr_fmri'),
    ('state',          'ctrl_dsrFULL',     'state'),
    ('location',       'ctrl_dsrFULL',     'location'),
]

ROI_NAME_MAP = {
    'ACC': 'mPFC',
    'medialOFC': 'mOFC',
    'medial OFC': 'mOFC',
    'Parahippocampal': 'PHC',
    'Parahippocampus': 'PHC',
}


def _pick_q_column(df):
    """Prefer the per-combo BH-FDR column when present; fall back to the
    single-family q column, then to raw p_perm."""
    for c in ('q_fdr_per_combo', 'p_fdr', 'p_perm'):
        if c in df.columns:
            return c
    return None


def _heatmap_matrices(df, rois, panels, q_col):
    T = np.full((len(rois), len(panels)), np.nan)
    P = np.full((len(rois), len(panels)), np.nan)
    Q = np.full((len(rois), len(panels)), np.nan)
    for j, (_, combo, sub) in enumerate(panels):
        for i, roi in enumerate(rois):
            row = df[(df.combo == combo) & (df.sub_model == sub) & (df.roi == roi)]
            if row.empty:
                continue
            T[i, j] = row['t'].iloc[0]
            if 'p_perm' in row.columns:
                P[i, j] = row['p_perm'].iloc[0]
            if q_col is not None and q_col in row.columns:
                Q[i, j] = row[q_col].iloc[0]
    return T, P, Q


def _label_submodel(sub_model):
    """Short, readable column label for a combo sub-model."""
    labels = {
        'dsr_fmri': 'DSR\n(full)',
        'dsr_fmri_fut': 'DSR\n(future)',
        'dsr_fmri_informed': 'DSR\n(informed)',
        'state': 'state',
        'location': 'location',
    }
    return labels.get(sub_model, str(sub_model).replace('_', '\n'))


def _combo_panels(sub_models):
    """Make heatmap panels for one combo model.

    Every DSR, state, and location regressor gets its own panel. All other
    regressors are placed together in one panel, preserving their individual
    columns and the order in which they occur in the results table.
    """
    dsr = [s for s in sub_models if 'dsr' in str(s).lower()]
    state = [s for s in sub_models if s == 'state']
    location = [s for s in sub_models if s == 'location']
    special = set(dsr + state + location)
    other = [s for s in sub_models if s not in special]

    groups = []
    col_labels = []
    used = []
    for sub in dsr:
        used.append(len(col_labels))
        col_labels.append(_label_submodel(sub))
        groups.append(((used[-1],), 'RdBu_r', 'DSR'))
    if state:
        used.append(len(col_labels))
        col_labels.append(_label_submodel(state[0]))
        groups.append(((used[-1],), 'RdBu_r', 'State'))
    if location:
        used.append(len(col_labels))
        col_labels.append(_label_submodel(location[0]))
        groups.append(((used[-1],), 'RdBu_r', 'Location'))
    if other:
        start = len(col_labels)
        col_labels.extend(_label_submodel(s) for s in other)
        groups.append((tuple(range(start, len(col_labels))), 'RdBu_r',
                       'Other covariates'))
    return [(label, sub) for label, sub in zip(col_labels,
                                                dsr + state + location + other)], groups


def _heatmap_width_cm(n_columns):
    """Keep the existing column proportions and grow for extra regressors."""
    # Existing figures use 7.5 cm for three columns and 9.5 cm for five;
    # each additional column therefore adds the established ~1.9 cm.
    if n_columns <= 3:
        return 7.5
    return 9.5 + 1.9 * (n_columns - 5)


def _dsr_colour_limit(t_matrix, panels):
    """Return one symmetric colour limit anchored to the DSR columns."""
    dsr_cols = [i for i, (_, sub) in enumerate(panels)
                if 'dsr' in str(sub).lower()]
    values = t_matrix[:, dsr_cols] if dsr_cols else t_matrix
    finite = np.abs(values[np.isfinite(values)])
    observed = float(np.max(finite)) if finite.size else 0.0
    # No arbitrary lower bound: all covariates use the scale supported by
    # the key DSR model for this combo. A tiny fallback only prevents an
    # invalid zero-width colour normalization when all values are missing.
    vmax = observed if observed > 0 else 1e-12
    return -vmax, vmax


def make_publication_figures(
    run_dir,
    test_variant: str = 'split_halves_z',
    out_subdir: str = 'pub_figures_v2',
    roi_order: Sequence[str] = DEFAULT_ROI_ORDER,
    roi_colours: dict = None,
    panels_all: list = None,
    panels_core: list = None,
    hist_panels: list = None,
    verbose: bool = True,
):
    """Build ROI × regressor heatmaps + per-ROI perm-β histograms.

    Parameters
    ----------
    run_dir : str or Path
        A completed ``RSA_DSR_ROIs_simple.py`` run directory. Must contain
        ``results_summary_combos.csv`` and (for the histograms)
        ``perm_null_draws/perm_<ROI>.pkl``.
    test_variant : str
        Which test slice of the summary CSV to plot (e.g. ``'split_halves_z'``).
    out_subdir : str
        Sub-folder of ``run_dir`` for the outputs.
    """
    run_dir = Path(run_dir)
    if roi_colours is None:
        roi_colours = DEFAULT_ROI_COLOURS
    if panels_all is None:
        panels_all = DEFAULT_PANELS_ALL
    if panels_core is None:
        panels_core = DEFAULT_PANELS_CORE
    if hist_panels is None:
        hist_panels = DEFAULT_HIST_PANELS

    summary_csv = run_dir / 'results_summary_combos.csv'
    if not summary_csv.exists():
        if verbose:
            print(f"[pub-figs] SKIP: no results_summary_combos.csv in {run_dir}")
        return
    df = pd.read_csv(summary_csv)
    if 'roi' in df.columns:
        df['roi'] = df['roi'].map(lambda r: ROI_NAME_MAP.get(r, r))
    df = df[df.test == test_variant]
    if df.empty:
        if verbose:
            print(f"[pub-figs] SKIP: no rows for test={test_variant!r}")
        return
    rois = [r for r in roi_order if r in df.roi.unique()]
    if not rois:
        if verbose:
            print(f"[pub-figs] SKIP: none of {roi_order} present in summary")
        return

    out_dir = run_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    q_col = _pick_q_column(df)

    # ── One heatmap per combo model ───────────────────────────────────
    # Use the order in which sub-models first occur in the CSV. This keeps
    # the output aligned with the model specification used for that run.
    combos = list(dict.fromkeys(df['combo'].dropna().tolist()))
    for combo in combos:
        combo_rows = df[df.combo == combo]
        sub_models = list(dict.fromkeys(combo_rows['sub_model'].dropna().tolist()))
        panels, groups = _combo_panels(sub_models)
        panels = [p for p in panels if not combo_rows[
            combo_rows.sub_model == p[1]].empty]
        if not panels:
            continue

        T, _P, Q = _heatmap_matrices(
            combo_rows, rois, [(label, combo, sub) for label, sub in panels],
            q_col)
        col_labels = [label for label, _ in panels]
        # All panels use the same diverging red-blue scale family. The
        # plotting helper still scales each panel symmetrically to its data.
        n_columns = len(col_labels)
        fig_size = (_heatmap_width_cm(n_columns), 12.0)
        vmin, vmax = _dsr_colour_limit(T, panels)
        safe_combo = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(combo))

        plot_roi_tstat_heatmap(
            T, rois, col_labels,
            q_matrix=Q,
            panel_groups=groups,
            title=f'{combo} — {test_variant}  '
                  f'(stars = BH-FDR q per combo)',
            cbar_label='t vs 0',
            fig_size_cm=fig_size,
            vmin=vmin, vmax=vmax,
            save_path=str(out_dir /
                          f'heatmap_roi_x_regressor_FDR_{safe_combo}_{test_variant}'),
        )
        if verbose:
            print(f"[pub-figs] wrote combo={combo} "
                  f"({len(rois)} ROIs × {n_columns} regressors; "
                  f"colour scale {vmin:g}…{vmax:g})")

    # ── Per-ROI perm-β histograms ─────────────────────────────────────
    perm_dir = run_dir / 'perm_null_draws'
    if not perm_dir.exists():
        if verbose:
            print(f"[pub-figs] no perm_null_draws/ — skipping histograms")
        return

    base_test = test_variant[:-2] if test_variant.endswith('_z') else test_variant

    for label, combo, sub in hist_panels:
        rows = []
        emp_per_roi = {}
        for roi in rois:
            pkl = perm_dir / f'perm_{roi}.pkl'
            if not pkl.exists():
                continue
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            combo_models = d.get('combo_models', {})
            if combo not in combo_models:
                continue
            sub_list = combo_models[combo]
            if sub not in sub_list:
                continue
            k = sub_list.index(sub)
            perm_betas = np.asarray(
                d['perm_results_combo'][test_variant][combo]['beta'])
            emp_per_roi[roi] = float(
                d['empirical_combo_results_z'][base_test][combo][1][k])
            for b in perm_betas[:, k]:
                rows.append({'roi': roi, 'beta': float(b)})
        hist_df = pd.DataFrame(rows)
        if hist_df.empty:
            if verbose:
                print(f"[pub-figs] no perm data for {label}")
            continue

        q_per_roi = {}
        for roi in rois:
            row = df[(df.combo == combo) & (df.sub_model == sub) & (df.roi == roi)]
            if not row.empty and q_col is not None and q_col in row.columns:
                q_per_roi[roi] = float(row[q_col].iloc[0])

        plot_per_roi_stat_histograms(
            hist_df, stat_col='beta',
            stat_label=f'perm β ({label})',
            roi_order=rois, roi_colours=roi_colours,
            q_per_roi=q_per_roi,
            empirical_per_roi=emp_per_roi,
            mark_sig_neurons=None,
            n_cols=4, panel_w_cm=3.0, panel_h_cm=1.5,
            save_path=str(out_dir / f'hist_perROI_{sub}_{test_variant}'),
        )
        if verbose:
            print(f"[pub-figs] wrote hist_perROI_{sub}_{test_variant}")

    if verbose:
        print(f"[pub-figs] all outputs in {out_dir}")


# ── Standalone entry — regenerate figures for one past run ────────────
if __name__ == '__main__':
    DATA = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                '/ephys_humans/derivatives')
    RUN_DIR = DATA / 'group/DSR_RSA_simple_ROI/2026-07-30_13-32-23'
    make_publication_figures(RUN_DIR, test_variant='split_halves_z')
