#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlay ROI lag-curves onto one axis to show the mPFC-vs-HC double
dissociation: mPFC peaks at future lags (30-60°), HC_anterior/HC_mid at
0° (now). It writes both cell-weighted and subject-balanced versions:
(a) mean CV r ± SEM per lag, (b) one-sample t of Fisher-z CV r > 0 per
lag, and (c) a side-by-side weighting comparison. Both wrap lag 0 back at
the end (13 x-positions:
0, 30, ..., 330, 0) so the circular structure is obvious.

For Spyder, select the result family at the top of the file with
`CALL_RESULTS_FROM = 'per_lag'` or `'spatial_peaks'`. The per-lag encoding
results use `per_cell_ALL_ROIs.csv` and its `r_lagXXX_{noctrl,ctrl}` columns;
the spatial-peaks results use `per_cell.csv` and its
`per_lag_r_all_lags_json` column.

Subject-balanced curves first average cells within subject and then give every
subject equal weight. Never displays "ACC" — remapped to "mPFC".
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
CALL_WEIGHTING = 'both'  # 'both', 'cell', or 'subject'

ROIS_TO_OVERLAY = ['mPFC', 'HC_mid']
STATS_ROIS = ['mPFC', 'HC_anterior', 'HC_mid']

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
                       "2026-08-04_07-25-15_reload_from_2026-06-30_18-21-57_"
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

def _load_curve_data(per_cell_csv, source="spatial_peaks", ctrl_mode="noctrl",
                     rois=ROIS_TO_OVERLAY):
    """Return per-cell curves and subject IDs for each ROI.

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

    subject_col = "subject_id"
    if subject_col not in df:
        raise ValueError(f"{per_cell_csv} has no {subject_col!r} column")
    out = {}
    for roi in rois:
        g = df[df["roi"] == roi]
        curves, subjects = [], []
        for _, row in g.iterrows():
            c = curve_from_row(row)
            if c is not None:
                curves.append(c)
                subject = str(row[subject_col]).replace("sub-", "")
                if subject.endswith(".0"):
                    subject = subject[:-2]
                subjects.append(subject.zfill(2))
        out[roi] = {
            "curves": (np.asarray(curves, float) if curves else
                       np.empty((0, len(LAGS_DEG_BASE)))),
            "subjects": np.asarray(subjects, str),
        }
    return out


def _analysis_units(record, weighting="cell", fisher=False):
    """Return cell curves or equal-weight subject-mean curves."""
    curves = record["curves"]
    if fisher:
        curves = np.arctanh(np.clip(curves, -0.9999999, 0.9999999))
    if weighting == "cell":
        return curves
    if weighting != "subject":
        raise ValueError(f"Unknown weighting {weighting!r}")
    subjects = record["subjects"]
    return np.stack([
        np.nanmean(curves[subjects == subject], axis=0)
        for subject in np.unique(subjects)
    ])


def _bh_fdr(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    good = np.isfinite(p)
    if not good.any():
        return out
    pg = p[good]
    order = np.argsort(pg)
    ranked = pg[order]
    q = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    restored = np.empty_like(q)
    restored[order] = np.minimum(q, 1)
    out[good] = restored
    return out


def _wrap_lag(a):
    """Append the first column at the end to close the circle."""
    if a.ndim == 1:
        return np.concatenate([a, a[:1]])
    return np.concatenate([a, a[:, :1]], axis=1)


def _tstat_gt0(values):
    """One-sample one-sided t (> 0) at each lag column, and its p."""
    from scipy.stats import ttest_1samp
    n_lags = values.shape[1]
    t = np.full(n_lags, np.nan)
    p_one = np.full(n_lags, np.nan)
    p_two = np.full(n_lags, np.nan)
    for j in range(n_lags):
        x = values[:, j]; x = x[np.isfinite(x)]
        if x.size >= 2:
            res = ttest_1samp(x, 0.0, alternative="greater")
            t[j] = float(res.statistic)
            p_one[j] = float(res.pvalue)
            p_two[j] = float(ttest_1samp(x, 0.0).pvalue)
    return t, p_one, p_two


def _subject_clustered_lagwise_tests(records, rois):
    """Return one lag-wise Fisher-z t test per ROI after subject aggregation.

    Each cell's CV correlation is Fisher-transformed, cells are averaged
    within subject at every lag, and the resulting independent subject means
    are tested against zero.  This avoids treating multiple cells from the
    same subject as independent observations.
    """
    rows = []
    for roi in rois:
        record = records[roi]
        if record["curves"].size == 0:
            continue
        raw_subject_means = _analysis_units(
            record, weighting="subject", fisher=False)
        z_subject_means = _analysis_units(
            record, weighting="subject", fisher=True)
        t, p_one, p_two = _tstat_gt0(z_subject_means)
        for j, lag in enumerate(LAGS_DEG_BASE):
            valid = np.isfinite(z_subject_means[:, j])
            n_subjects_valid = int(valid.sum())
            rows.append({
                "roi": _display(roi),
                "lag_deg": lag,
                "test": "one_sample_fisher_z_gt_zero",
                "analysis_unit": "subject_mean",
                "subject_aggregation": (
                    "Fisher-z CV r averaged across cells within subject"),
                "n_cells": record["curves"].shape[0],
                "n_subjects_total": np.unique(record["subjects"]).size,
                "n_subjects_valid": n_subjects_valid,
                "mean_subject_raw_r": (
                    float(np.nanmean(raw_subject_means[:, j]))
                    if n_subjects_valid else np.nan),
                "mean_subject_fisher_z": (
                    float(np.nanmean(z_subject_means[:, j]))
                    if n_subjects_valid else np.nan),
                "t_fisher_z_vs_0": t[j],
                "df": n_subjects_valid - 1,
                "p_one_sided": p_one[j],
                "p_two_sided": p_two[j],
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        # The primary correction treats the 12 circular lags within each ROI
        # as one family; the second column makes the whole 3 ROI × 12 lag
        # family available for readers who want that more conservative scope.
        out["p_one_sided_fdr_12_lags"] = (
            out.groupby("roi")["p_one_sided"].transform(_bh_fdr))
        out["p_two_sided_fdr_12_lags"] = (
            out.groupby("roi")["p_two_sided"].transform(_bh_fdr))
        out["p_one_sided_fdr_all_stats_rois_lags"] = _bh_fdr(
            out["p_one_sided"].to_numpy())
        out["p_two_sided_fdr_all_stats_rois_lags"] = _bh_fdr(
            out["p_two_sided"].to_numpy())
    return out


def _unit_label(record, roi, weighting):
    n_cells = record["curves"].shape[0]
    n_subjects = np.unique(record["subjects"]).size
    if weighting == "subject":
        return f"{_display(roi)} ({n_subjects} subjects; {n_cells} cells)"
    return f"{_display(roi)} ({n_cells} cells)"


def _draw_mean_overlay(ax, records, rois, weighting):
    x_wrap = np.asarray(LAGS_DEG_BASE + [360])
    for roi in rois:
        record = records[roi]
        C = _analysis_units(record, weighting=weighting, fisher=False)
        if C.size == 0:
            continue
        mean = np.nanmean(C, axis=0)
        sem = (np.nanstd(C, axis=0, ddof=1) /
               np.sqrt(np.maximum(np.isfinite(C).sum(axis=0), 1)))
        mean_w, sem_w = _wrap_lag(mean), _wrap_lag(sem)
        colour = _roi_colour(roi)
        ax.fill_between(x_wrap, mean_w - sem_w, mean_w + sem_w,
                        color=colour, alpha=0.18, linewidth=0)
        ax.plot(x_wrap, mean_w, "-o", color=colour, lw=1.5, ms=3,
                label=_unit_label(record, roi, weighting))
    ax.axhline(0, color="black", lw=0.5, ls="--")


def _draw_t_overlay(ax, records, rois, weighting):
    x_wrap = np.asarray(LAGS_DEG_BASE + [360])
    for roi in rois:
        record = records[roi]
        Z = _analysis_units(record, weighting=weighting, fisher=True)
        if Z.size == 0:
            continue
        t, _, _ = _tstat_gt0(Z)
        ax.plot(x_wrap, _wrap_lag(t), "-o", color=_roi_colour(roi),
                lw=1.5, ms=3, label=_unit_label(record, roi, weighting))
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axhline(1.65, color="grey", lw=0.5, ls=":",
               label="approx. one-sided p = .05")


def _target_tests(records, rois, weightings):
    """Tests of each ROI's pre-defined lag window on Fisher-z units."""
    from scipy.stats import ttest_1samp
    targets = {"mPFC": (30, 60), "HC_mid": (0, 330),
               "HC_anterior": (0, 330)}
    rows = []
    for weighting in weightings:
        for roi in rois:
            if roi not in targets or records[roi]["curves"].size == 0:
                continue
            Z = _analysis_units(records[roi], weighting=weighting, fisher=True)
            target_idx = [LAGS_DEG_BASE.index(lag) for lag in targets[roi]]
            other_idx = [i for i in range(len(LAGS_DEG_BASE))
                         if i not in target_idx]
            target = np.nanmean(Z[:, target_idx], axis=1)
            contrast = target - np.nanmean(Z[:, other_idx], axis=1)
            for test, values in (("target_mean_vs_zero", target),
                                 ("target_vs_other_lags", contrast)):
                values = values[np.isfinite(values)]
                one = ttest_1samp(values, 0, alternative="greater")
                two = ttest_1samp(values, 0)
                rows.append({
                    "weighting": weighting, "roi": roi,
                    "target_lags_deg": "+".join(map(str, targets[roi])),
                    "test": test, "n_units": len(values),
                    "n_cells": records[roi]["curves"].shape[0],
                    "n_subjects": np.unique(records[roi]["subjects"]).size,
                    "mean_fisher_z_or_difference": float(np.mean(values)),
                    "t": float(one.statistic),
                    "df": len(values) - 1,
                    "p_one_sided": float(one.pvalue),
                    "p_two_sided": float(two.pvalue),
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_one_sided_fdr_across_rois"] = (
            out.groupby(["weighting", "test"])["p_one_sided"]
            .transform(_bh_fdr))
        out["p_two_sided_fdr_across_rois"] = (
            out.groupby(["weighting", "test"])["p_two_sided"]
            .transform(_bh_fdr))
    return out


def make_overlay(per_cell_csv, out_dir, rois=ROIS_TO_OVERLAY,
                 source="spatial_peaks", ctrl_mode="noctrl",
                 weighting="both"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    loaded_rois = list(dict.fromkeys(list(rois) + STATS_ROIS))
    records = _load_curve_data(per_cell_csv, source=source, ctrl_mode=ctrl_mode,
                               rois=loaded_rois)
    weightings = ["cell", "subject"] if weighting == "both" else [weighting]

    x_wrap = np.asarray(LAGS_DEG_BASE + [360])   # 13 positions, last shown as 360°
    tick_pos = [0, 30, 60, 120, 180, 240, 300, 360]
    tick_lab = ["0°", "30°", "60°", "120°", "180°", "240°", "300°", "0°"]

    # ---------- Separate mean and t-stat figures ------------------------
    for current_weighting in weightings:
        suffix = "" if current_weighting == "cell" else "_subject_weighted"
        unit_title = ("cell-weighted" if current_weighting == "cell"
                      else "subject-balanced")
        fig, ax = plt.subplots(figsize=(6 * CM, 4 * CM), constrained_layout=True)
        _draw_mean_overlay(ax, records, rois, current_weighting)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, fontsize=FONT_TICK, rotation=45)
        ax.set_xlabel("Lag (looking into the future)", fontsize=FONT_AXIS)
        ax.set_ylabel("Mean CV r", fontsize=FONT_AXIS)
        ax.tick_params(axis="both", labelsize=FONT_TICK, length=2, pad=1)
        ax.legend(fontsize=FONT_TICK - 1, frameon=False, loc="upper right")
        ax.set_title(f"mPFC–HC double dissociation ({unit_title})",
                     fontsize=FONT_BIG)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(out_dir / f"overlay_meanR_wrapped{suffix}.{ext}",
                        dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6 * CM, 4 * CM), constrained_layout=True)
        _draw_t_overlay(ax, records, rois, current_weighting)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, fontsize=FONT_TICK, rotation=45)
        ax.set_xlabel("Lag (looking into the future)", fontsize=FONT_AXIS)
        ax.set_ylabel("t-stat (Fisher-z CV r > 0)", fontsize=FONT_AXIS)
        ax.tick_params(axis="both", labelsize=FONT_TICK, length=2, pad=1)
        ax.legend(fontsize=FONT_TICK - 1, frameon=False, loc="upper right")
        ax.set_title(f"Per-lag Fisher-z tests ({unit_title})", fontsize=FONT_BIG)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(out_dir / f"overlay_tstat_wrapped{suffix}.{ext}",
                        dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # ---------- Direct publication comparison --------------------------
    if weighting == "both":
        fig, axes = plt.subplots(1, 2, figsize=(18 * CM, 6 * CM),
                                 sharex=True, sharey=True,
                                 constrained_layout=True)
        for ax, current_weighting, title in zip(
                axes, ("cell", "subject"),
                ("Cell-weighted", "Subject-balanced")):
            _draw_mean_overlay(ax, records, rois, current_weighting)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lab, fontsize=FONT_TICK, rotation=45)
            ax.set_xlabel("Lag (looking into the future)", fontsize=FONT_AXIS)
            ax.tick_params(axis="both", labelsize=FONT_TICK, length=2, pad=1)
            ax.set_title(title, fontsize=FONT_BIG)
            ax.legend(fontsize=FONT_TICK - 2, frameon=False, loc="upper right")
        axes[0].set_ylabel("Mean CV r", fontsize=FONT_AXIS)
        fig.suptitle("mPFC–HC double dissociation: effect of population weighting",
                     fontsize=FONT_BIG)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(out_dir / f"overlay_meanR_weighting_comparison.{ext}",
                        dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    # ---------- Print + save table --------------------------------------
    rows = []
    for current_weighting in weightings:
        for roi in rois:
            record = records[roi]
            C = _analysis_units(record, weighting=current_weighting, fisher=False)
            Z = _analysis_units(record, weighting=current_weighting, fisher=True)
            if C.size == 0:
                continue
            mean_r = np.nanmean(C, axis=0)
            mean_z = np.nanmean(Z, axis=0)
            t, p_one, p_two = _tstat_gt0(Z)
            for j, lag in enumerate(LAGS_DEG_BASE):
                rows.append({
                    "weighting": current_weighting, "roi": _display(roi),
                    "n_units": C.shape[0],
                    "n_cells": record["curves"].shape[0],
                    "n_subjects": np.unique(record["subjects"]).size,
                    "lag_deg": lag, "mean_raw_r": mean_r[j],
                    "mean_fisher_z": mean_z[j], "t_fisher_vs_0": t[j],
                    "df": C.shape[0] - 1, "p_one_sided": p_one[j],
                    "p_two_sided": p_two[j],
                })
    tbl = pd.DataFrame(rows)
    tbl["p_one_sided_fdr_12_lags"] = (
        tbl.groupby(["weighting", "roi"])["p_one_sided"].transform(_bh_fdr))
    tbl["p_two_sided_fdr_12_lags"] = (
        tbl.groupby(["weighting", "roi"])["p_two_sided"].transform(_bh_fdr))
    tbl["p_one_sided_fdr_across_overlay_rois"] = (
        tbl.groupby(["weighting", "lag_deg"])["p_one_sided"].transform(_bh_fdr))
    tbl["p_two_sided_fdr_across_overlay_rois"] = (
        tbl.groupby(["weighting", "lag_deg"])["p_two_sided"].transform(_bh_fdr))
    tbl.to_csv(out_dir / "overlay_per_lag_table.csv", index=False)

    # Always write the lag-wise subject-level inference table, regardless of
    # whether the requested figures are cell-weighted, subject-balanced, or
    # both.  STATS_ROIS includes HC_anterior even though it is not in the
    # default visual overlay.
    subject_lagwise = _subject_clustered_lagwise_tests(records, STATS_ROIS)
    subject_lagwise.to_csv(
        out_dir / "overlay_subject_clustered_lagwise_ttests.csv", index=False)

    target = _target_tests(records, STATS_ROIS, weightings)
    target.to_csv(out_dir / "overlay_target_window_tests.csv", index=False)

    report = [
        "# Cell-weighted versus subject-balanced overlay", "",
        f"Input: `{per_cell_csv}`", "",
        "Subject-balanced visualization averages raw r across cells within "
        "each subject, then averages subjects. Subject-level tests Fisher-"
        "transform each cell r, average z within subject, and test those "
        "subject means. Target-window FDR is across mPFC, HC_anterior, and "
        "HC_mid, matching the three predicted-lag ROI family.", "",
        "## Key curve values", "",
        "| weighting | ROI | units | 0° | 30° | 60° | 330° |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for current_weighting in weightings:
        for roi in rois:
            d = tbl[(tbl.weighting == current_weighting) & (tbl.roi == roi)]
            by_lag = d.set_index("lag_deg")
            report.append(
                f"| {current_weighting} | {roi} | {int(d.n_units.iloc[0])} | "
                f"{by_lag.loc[0, 'mean_raw_r']:.4f} | "
                f"{by_lag.loc[30, 'mean_raw_r']:.4f} | "
                f"{by_lag.loc[60, 'mean_raw_r']:.4f} | "
                f"{by_lag.loc[330, 'mean_raw_r']:.4f} |")
    report += ["", "## Pre-defined target-window tests", "",
               "One-sided tests ask whether the predicted window is greater "
               "than the ten other lags.", "",
               "| weighting | ROI | target | n | t(df) | p | FDR p |",
               "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    focus = target[target.test == "target_vs_other_lags"]
    for _, r in focus.iterrows():
        report.append(
            f"| {r.weighting} | {r.roi} | {r.target_lags_deg}° | "
            f"{int(r.n_units)} | {r.t:.2f}({int(r.df)}) | "
            f"{r.p_one_sided:.5f} | {r.p_one_sided_fdr_across_rois:.5f} |")
    report += ["", "## Files", "",
               "- `overlay_meanR_wrapped_subject_weighted.pdf`: standalone "
               "subject-balanced publication overlay.",
               "- `overlay_meanR_weighting_comparison.pdf`: matched cell- "
               "versus subject-weighted comparison.",
               "- `overlay_per_lag_table.csv`: per-lag means and Fisher tests.",
               "- `overlay_subject_clustered_lagwise_ttests.csv`: lag-wise "
               "Fisher-z t-tests after averaging cells within subject for "
               "mPFC, HC_anterior, and HC_mid.",
               "- `overlay_target_window_tests.csv`: predicted-window tests."]
    (out_dir / "WEIGHTING_RESULTS.md").write_text("\n".join(report) + "\n")

    config = {
        "per_cell_csv": str(per_cell_csv), "source": source,
        "control_mode": ctrl_mode, "weighting": weighting,
        "overlay_rois": list(rois), "target_test_fdr_rois": STATS_ROIS,
        "lags_deg": LAGS_DEG_BASE,
        "subject_balancing": "mean Fisher z within subject, then t across subjects",
        "visual_subject_balancing": "mean raw r within subject, then mean across subjects",
    }
    (out_dir / "overlay_config.json").write_text(json.dumps(config, indent=2))
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
    p.add_argument("--weighting", choices=("both", "cell", "subject"),
                   default=None,
                   help="Write cell-weighted, subject-balanced, or both figures.")
    p.add_argument("--out-dir", default=None,
                    help="Default: <per-cell dir>/overlay_double_dissociation/")
    args = p.parse_args()
    source = args.source or CALL_RESULTS_FROM
    ctrl_mode = args.ctrl_mode or CALL_PER_LAG_CTRL_MODE
    weighting = args.weighting or CALL_WEIGHTING
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
    make_overlay(csv, out_dir, source=source, ctrl_mode=ctrl_mode,
                 weighting=weighting)


if __name__ == "__main__":
    main()
