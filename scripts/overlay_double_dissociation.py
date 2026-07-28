#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlay ROI lag-curves onto one axis to show the mPFC-vs-HC double
dissociation: mPFC peaks at future lags (30-60°), HC_anterior/HC_mid at
0° (now). Two panels: (a) mean CV r ± SEM per lag, (b) one-sample t of
CV r > 0 per lag. Both wrap lag 0 back at the end (13 x-positions:
0, 30, ..., 330, 0) so the circular structure is obvious.

Data source: `per_cell.csv` in a spatial_peaks reload dir (default = the
2026-07-23 RSA-excluded relabelled cohort). Uses the per-cell
`per_lag_r_all_lags_json` column, which stores one r per lag per cell.

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
    SHOWGIRL2_DISCRETE, ROI_COLORS_SHOWGIRL2, ROI_DISPLAY_NAMES,
)


ROI_COLOURS = {
    'EC':              SHOWGIRL2_DISCRETE[0],
    'ACC':             SHOWGIRL2_DISCRETE[1],
    'HC_anterior':     SHOWGIRL2_DISCRETE[2],
    'PCC':             SHOWGIRL2_DISCRETE[3],
    'medialOFC':       SHOWGIRL2_DISCRETE[4],
    'Parahippocampal': '#7FB0CC',
    'HC_mid':          '#a30d6c',
    'Precuneus':       '#23677E',
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
LAGS_DEG_BASE = list(range(0, 360, 30))    # 12 lags used in spatial_peaks
# ROIS_TO_OVERLAY = ["ACC", "HC_anterior", "HC_mid"]  # the three with predicted lags
ROIS_TO_OVERLAY = ["ACC", "HC_mid"]  # the three with predicted lags


# Publication sizing (see CLAUDE.md — subpanel ~7 cm at Arial 9-11 pt)
CM = 1 / 2.54
FONT_TICK, FONT_AXIS, FONT_BIG = 9, 10, 11
DPI = 300


def _display(roi):
    return ROI_DISPLAY_NAMES.get(roi, roi)


def _roi_colour(roi):
    """Return the canonical Showgirl2 hex for `roi`, falling back to grey."""
    idx = ROI_COLORS_SHOWGIRL2.get(roi)
    if idx is None:
        return "#666"
    return SHOWGIRL2_DISCRETE[idx]

def _load_curves(per_cell_csv, rois=ROIS_TO_OVERLAY):
    """Return {roi: array (n_cells, n_lags)} of per-cell CV r at each
    of the 12 lags. Cells with missing curves are dropped."""
    df = pd.read_csv(per_cell_csv)
    out = {}
    for roi in rois:
        g = df[df["roi"] == roi]
        curves = []
        for _, row in g.iterrows():
            try:
                c = json.loads(row.get("per_lag_r_all_lags_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                c = []
            if len(c) == len(LAGS_DEG_BASE):
                curves.append([np.nan if v is None else float(v) for v in c])
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


def make_overlay(per_cell_csv, out_dir, rois=ROIS_TO_OVERLAY):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    curves = _load_curves(per_cell_csv, rois)

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
        col = ROI_COLOURS[roi]
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
        col = ROI_COLOURS[roi]
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
    p.add_argument("--per-cell-csv", default=DEFAULT_PER_CELL_CSV)
    p.add_argument("--out-dir", default=None,
                    help="Default: <per_cell dir>/overlay_double_dissociation/")
    args = p.parse_args()
    csv = Path(args.per_cell_csv)
    out_dir = (Path(args.out_dir) if args.out_dir
                else csv.parent / "overlay_double_dissociation")
    make_overlay(csv, out_dir)


if __name__ == "__main__":
    main()
