#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlay ROI lag-curves onto one axis to show the mPFC-vs-HC double
dissociation: mPFC peaks at future lags (30-60°), HC_anterior/HC_mid at
0° (now). Two panels: (a) mean CV r ± SEM per lag, (b) one-sample t of
CV r > 0 per lag. Both wrap lag 0 back at the end (13 x-positions:
0, 30, ..., 330, 0) so the circular structure is obvious.

For Spyder, select the result family at the top of the file with
`CALL_RESULTS_FROM = 'per_lag'` or `'spatial_peaks'`. The per-lag encoding
results use `per_cell_ALL_ROIs.csv` and its `r_lagXXX_{noctrl,ctrl}` columns;
the spatial-peaks results use `per_cell.csv` and its
`per_lag_r_all_lags_json` column.

Never displays "ACC" — remapped to "mPFC" via ROI_DISPLAY_NAMES.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")
from mc.plotting.cell_results import (
    SHOWGIRL2_DISCRETE,
)


# ---- Spyder call settings --------------------------------------------
# Change this value when running the file with Spyder's "Run file" button.
CALL_RESULTS_FROM = 'per_lag'  # 'spatial_peaks'
CALL_PER_LAG_CTRL_MODE = 'noctrl'  # 'ctrl' is also available

ROIS_TO_OVERLAY = ['mPFC', 'HC_anterior']

ROI_COLOURS = {
    'EC':              SHOWGIRL2_DISCRETE[0],
    'mPFC':            SHOWGIRL2_DISCRETE[1],
    'HC_anterior':     '#a30d6c',
    'PCC':             SHOWGIRL2_DISCRETE[3],
    'mOFC':             SHOWGIRL2_DISCRETE[4],
    'PHC':             '#23677E',
    'HC_mid':          SHOWGIRL2_DISCRETE[2],
}

# Normalize labels written by older analyses to the names used in the plot.
ROI_NAME_MAP = {
    'ACC': 'mPFC',
    'mPFC': 'mPFC',
    'medialOFC': 'mOFC',
    'mOFC': 'mOFC',
    'Parahippocampal': 'PHC',
    'Parahippocampus': 'PHC',
    'PHC': 'PHC',
}


#2026-07-01_10-35-40_reload_from_2026-06-26_18-47-11_phase_resid_paired_fixedlag-final
# ---- I/O defaults ----------------------------------------------------
# DEFAULT_PER_CELL_CSV = ("/Users/xpsy1114/Documents/projects/multiple_clocks/"
#                         "data/ephys_humans/derivatives/group/"
#                         "spatial_peaks_simple/"
#                         "2026-07-23_12-28-10_reload_from_2026-07-01_07-29-33_"
#                         "reload_from_replot_no_rsa_cells_relabelled/"
#                         "per_cell.csv")
DEFAULT_PER_CELL_CSV = ("/Users/xpsy1114/Documents/projects/multiple_clocks/"
                        "data/ephys_humans/derivatives/group/"
                        "spatial_peaks_simple/"
                        "2026-07-01_10-35-40_reload_from_2026-06-26_18-47-11_"
                        "phase_resid_paired_fixedlag-final/"
                        "per_cell.csv")
DEFAULT_PER_LAG_CSV = ("/Users/xpsy1114/Documents/projects/multiple_clocks/"
                       "data/ephys_humans/derivatives/group/"
                       "per_lag_encoding/"
                       "2026-07-29_15-52-36_reload_from_2026-06-30_18-21-57_"
                       "relabelled/per_cell_ALL_ROIs.csv")




LAGS_DEG_BASE = list(range(0, 360, 30))    # 12 lags used in spatial_peaks



# Publication sizing (see CLAUDE.md — subpanel ~7 cm at Arial 9-11 pt)
CM = 1 / 2.54
FONT_TICK, FONT_AXIS, FONT_BIG = 9, 10, 11
DPI = 300


def _display(roi):
    return roi


def _roi_colour(roi):
    """Return the requested colour for a canonical ROI name."""
    return ROI_COLOURS.get(roi, "#666")

def _load_curves(per_cell_csv, source="spatial_peaks", ctrl_mode="noctrl",
                 rois=ROIS_TO_OVERLAY):
    """Return {roi: array (n_cells, n_lags)} of per-cell CV r.

    The two analysis pipelines write the same conceptual result in different
    table formats, so the source-specific parsing is kept here rather than
    making callers know about either format.
    """
    df = pd.read_csv(per_cell_csv)
    if "roi" not in df:
        raise ValueError(f"{per_cell_csv} has no 'roi' column")
    df["roi"] = df["roi"].map(lambda roi: ROI_NAME_MAP.get(roi, roi))

    if source == "spatial_peaks":
        def curve_from_row(row):
            try:
                c = json.loads(row.get("per_lag_r_all_lags_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                c = []
            if len(c) != len(LAGS_DEG_BASE):
                return None
            return [np.nan if v is None else float(v) for v in c]
    elif source == "per_lag":
        columns = [f"r_lag{lag:03d}_{ctrl_mode}" for lag in LAGS_DEG_BASE]
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"{per_cell_csv} is missing per-lag columns: {', '.join(missing)}"
            )

        def curve_from_row(row):
            values = pd.to_numeric(row[columns], errors="coerce")
            return values.to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown source {source!r}")

    out = {}
    for roi in rois:
        g = df[df["roi"] == roi]
        curves = []
        for _, row in g.iterrows():
            c = curve_from_row(row)
            if c is not None:
                curves.append(c)
        out[roi] = np.asarray(curves) if curves else np.empty((0, len(LAGS_DEG_BASE)))
    return out


def _wrap_lag(a):
    """Append the first column at the end to close the circle."""
    if a.ndim == 1:
        return np.concatenate([a, a[:1]])
    return np.concatenate([a, a[:, :1]], axis=1)


def _tstat_gt0(curves):
    """One-sample one-sided t (r > 0) at each lag column, and its p."""
    from scipy.stats import ttest_1samp
    n_lags = curves.shape[1]
    t = np.full(n_lags, np.nan)
    p = np.full(n_lags, np.nan)
    for j in range(n_lags):
        x = curves[:, j]; x = x[np.isfinite(x)]
        if x.size >= 2:
            res = ttest_1samp(x, 0.0, alternative="greater")
            t[j] = float(res.statistic)
            p[j] = float(res.pvalue)
    return t, p


def make_overlay(per_cell_csv, out_dir, rois=ROIS_TO_OVERLAY,
                 source="spatial_peaks", ctrl_mode="noctrl"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    curves = _load_curves(per_cell_csv, source=source, ctrl_mode=ctrl_mode,
                          rois=rois)

    x_wrap = np.asarray(LAGS_DEG_BASE + [360])   # 13 positions, last shown as 360°
    tick_pos = LAGS_DEG_BASE + [360]
    tick_lab = [f"{L}°" for L in LAGS_DEG_BASE] + ["0°"]

    # ---------- Panel A: mean CV r ± SEM -------------------------------
    fig, ax = plt.subplots(figsize=(6 * CM, 4.5 * CM), constrained_layout=True)
    for roi in rois:
        C = curves[roi]
        if C.size == 0:
            continue
        m = np.nanmean(C, axis=0)
        s = (np.nanstd(C, axis=0, ddof=1) /
             np.sqrt(np.maximum(np.isfinite(C).sum(axis=0), 1)))
        m_w = _wrap_lag(m)
        s_w = _wrap_lag(s)
        col = _roi_colour(roi)
        ax.fill_between(x_wrap, m_w - s_w, m_w + s_w, color=col,
                        alpha=0.18, linewidth=0)
        ax.plot(x_wrap, m_w, "-o", color=col, lw=1.5, ms=3,
                label=f"{_display(roi)} (n = {C.shape[0]})")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, fontsize=FONT_TICK)
    ax.set_xlabel("Lag (looking into the future)", fontsize=FONT_AXIS)
    ax.set_ylabel("Mean CV r (rate map at lag L vs 0)", fontsize=FONT_AXIS)
    ax.tick_params(axis="both", labelsize=FONT_TICK, length=2, pad=1)
    ax.legend(fontsize=FONT_TICK, frameon=False, loc="upper right")
    ax.set_title("Rate-map consistency across future lags\n"
                 "mPFC peaks at 30-60°, HC at 0° (double dissociation)",
                 fontsize=FONT_BIG)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"overlay_meanR_wrapped.{ext}",
                    dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ---------- Panel B: one-sample t of r > 0 --------------------------
    fig, ax = plt.subplots(figsize=(9 * CM, 6 * CM), constrained_layout=True)
    for roi in rois:
        C = curves[roi]
        if C.size == 0:
            continue
        t, p = _tstat_gt0(C)
        t_w = _wrap_lag(t)
        col = _roi_colour(roi)
        ax.plot(x_wrap, t_w, "-o", color=col, lw=1.5, ms=3,
                label=f"{_display(roi)} (n = {C.shape[0]})")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    # 1-sided uncorrected p<0.05 threshold: t at df=large ≈ 1.65; annotate
    # a light dotted line for context (no FDR — this is a descriptive
    # overlay, not a hypothesis test).
    ax.axhline(1.65, color="grey", lw=0.5, ls=":",
                label="uncorrected p = 0.05 (1-sided)")
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, fontsize=FONT_TICK)
    ax.set_xlabel("Lag (looking into the future)", fontsize=FONT_AXIS)
    ax.set_ylabel("t-stat (CV r > 0)", fontsize=FONT_AXIS)
    ax.tick_params(axis="both", labelsize=FONT_TICK, length=2, pad=1)
    ax.legend(fontsize=FONT_TICK, frameon=False, loc="upper right")
    ax.set_title("Per-lag t-stat overlay\n"
                 "mPFC vs HC double dissociation",
                 fontsize=FONT_BIG)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"overlay_tstat_wrapped.{ext}",
                    dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ---------- Print + save table --------------------------------------
    rows = []
    for roi in rois:
        C = curves[roi]
        if C.size == 0:
            continue
        m = np.nanmean(C, axis=0)
        t, p = _tstat_gt0(C)
        for j, L in enumerate(LAGS_DEG_BASE):
            rows.append({"roi": _display(roi), "n_cells": C.shape[0],
                         "lag_deg": L, "mean_r": m[j],
                         "t_vs_0": t[j], "p_1sided_unc": p[j]})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(out_dir / "overlay_per_lag_table.csv", index=False)
    print("Wrote figures + table into", out_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("spatial_peaks", "per_lag"),
                   default=None,
                   help="Optional override; otherwise CALL_RESULTS_FROM is used.")
    p.add_argument("--per-cell-csv", default=None,
                   help="Explicit per-cell CSV; overrides the source default.")
    p.add_argument("--ctrl-mode", choices=("noctrl", "ctrl"), default=None,
                   help="Optional override; otherwise CALL_PER_LAG_CTRL_MODE is used.")
    p.add_argument("--out-dir", default=None,
                    help="Default: <per-cell dir>/overlay_double_dissociation/")
    args = p.parse_args()
    source = args.source or CALL_RESULTS_FROM
    ctrl_mode = args.ctrl_mode or CALL_PER_LAG_CTRL_MODE
    if source not in ("spatial_peaks", "per_lag"):
        p.error("CALL_RESULTS_FROM must be 'per_lag' or 'spatial_peaks'")
    if args.per_cell_csv:
        csv = Path(args.per_cell_csv)
    elif source == "spatial_peaks":
        csv = Path(DEFAULT_PER_CELL_CSV)
    else:
        csv = Path(DEFAULT_PER_LAG_CSV)
    if not csv.is_file():
        p.error(f"Per-cell CSV does not exist: {csv}")
    default_out_name = "overlay_double_dissociation"
    if source == "per_lag":
        default_out_name += f"_{ctrl_mode}"
    out_dir = (Path(args.out_dir) if args.out_dir
                else csv.parent / default_out_name)
    make_overlay(csv, out_dir, source=source, ctrl_mode=ctrl_mode)


if __name__ == "__main__":
    main()
