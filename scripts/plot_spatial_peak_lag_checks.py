#!/usr/bin/env python3
"""Focused checks for spatial_peaks_simple fixed-lag result files.

This script compares each cell's self-picked peak correlation
(`fold_rs_json`, averaged across folds) with the fixed-lag correlations
(`fixed_lag_per_lag_r_json`), then makes ACC-specific lag 30/60 checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "multiple_clocks_mplconfig"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from era_brewer import ERA_PALETTES


DATA_BASE = Path(
    "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/"
    "derivatives/group/spatial_peaks_simple"
)
DEFAULT_RUNS = {
    "future_30_60": DATA_BASE
    / "2026-06-18_12-17-45_full_optimal_phaseresid_norep"
    / "per_cell.csv",
    "now_330_0": DATA_BASE
    / "2026-06-18_15-13-34_full_optimal_lags_330_0_now"
    / "per_cell.csv",
}
DEFAULT_OUT_DIR = (
    REPO_DIR / "scripts" / "figures" / "2026-06-22_spatial_peak_lag_checks"
)

ROI_ORDER = [
    "ACC",
    "medialOFC",
    "PCC",
    "Parahippocampal",
    "HC_anterior",
    "HC_mid",
    "EC",
]

SG2 = ERA_PALETTES["Showgirl2"]["colors"]
COLORS = {
    "self": SG2[2],
    "lag_a": SG2[0],
    "lag_b": SG2[5],
    "accent": SG2[6],
    "neutral": "#6f6f6f",
}


def parse_json(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return json.loads(value)
    return value


def flatten_numbers(value):
    if value is None:
        return []
    if isinstance(value, (int, float, np.integer, np.floating)):
        val = float(value)
        return [val] if np.isfinite(val) else []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(flatten_numbers(item))
        return out
    return []


def finite_values(series):
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def mean_sem(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, 0
    if vals.size == 1:
        return float(vals[0]), np.nan, 1
    return float(np.mean(vals)), float(stats.sem(vals, nan_policy="omit")), int(vals.size)


def load_spatial_peaks_csv(path):
    df = pd.read_csv(path)
    if "note" in df.columns:
        df = df[df["note"] == "ok"].copy()
    else:
        df = df.copy()

    fold_col = "fold_rs_json"
    if fold_col not in df.columns and "fold_RS_json" in df.columns:
        fold_col = "fold_RS_json"
    if fold_col not in df.columns:
        raise ValueError(f"{path} has no fold_rs_json column")

    df["self_picked_peak_r"] = df[fold_col].apply(
        lambda x: np.nanmean(flatten_numbers(parse_json(x)))
    )

    lag_order = []
    for idx, row in df.iterrows():
        lags = parse_json(row.get("fixed_lag_lags_json"))
        vals = parse_json(row.get("fixed_lag_per_lag_r_json"))
        if not lags or not vals:
            continue
        for lag, val in zip(lags, vals):
            lag = int(lag)
            if lag not in lag_order:
                lag_order.append(lag)
            df.loc[idx, f"fixed_lag_{lag}_r"] = float(val)

    return df, lag_order


def roi_order_for(df):
    present = df["roi"].dropna().astype(str).unique().tolist()
    ordered = [roi for roi in ROI_ORDER if roi in present]
    ordered.extend(sorted(roi for roi in present if roi not in ordered))
    return ordered


def build_roi_summary(run_frames):
    rows = []
    for run_label, payload in run_frames.items():
        df = payload["df"]
        lag_order = payload["lags"]
        for roi in roi_order_for(df):
            sub = df[df["roi"] == roi]
            measures = [("self_picked_peak", "self_picked_peak_r")]
            measures.extend((f"fixed_lag_{lag}", f"fixed_lag_{lag}_r") for lag in lag_order)
            for measure, col in measures:
                mean, sem, n = mean_sem(sub[col])
                rows.append(
                    {
                        "run": run_label,
                        "roi": roi,
                        "measure": measure,
                        "mean_r": mean,
                        "sem_r": sem,
                        "n_cells": n,
                    }
                )
    return pd.DataFrame(rows)


def plot_roi_summary(summary, run_frames, out_dir):
    run_order = ["future_30_60", "now_330_0"]
    run_order = [run for run in run_order if run in run_frames]
    if not run_order:
        run_order = list(run_frames)

    fig, axes = plt.subplots(
        1,
        len(run_order),
        figsize=(7.4 * len(run_order), 5.0),
        sharey=True,
        constrained_layout=True,
    )
    if len(run_order) == 1:
        axes = [axes]

    for ax, run_label in zip(axes, run_order):
        df = run_frames[run_label]["df"]
        lag_order = run_frames[run_label]["lags"]
        rois = roi_order_for(df)
        x = np.arange(len(rois))
        measures = ["self_picked_peak"] + [f"fixed_lag_{lag}" for lag in lag_order]
        labels = ["self-picked peak"] + [f"fixed lag {lag}" for lag in lag_order]
        colors = [COLORS["self"], COLORS["lag_a"], COLORS["lag_b"]]
        width = min(0.25, 0.78 / len(measures))
        offsets = (np.arange(len(measures)) - (len(measures) - 1) / 2) * width

        for i, (measure, label) in enumerate(zip(measures, labels)):
            sub = summary[(summary["run"] == run_label) & (summary["measure"] == measure)]
            sub = sub.set_index("roi").reindex(rois)
            ax.bar(
                x + offsets[i],
                sub["mean_r"],
                width=width,
                yerr=sub["sem_r"],
                capsize=2.5,
                color=colors[i % len(colors)],
                edgecolor="black",
                linewidth=0.6,
                label=label,
            )

        pretty = run_label.replace("_", " ")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(pretty, loc="left", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(rois, rotation=32, ha="right")
        ax.set_xlabel("ROI")
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel("Mean correlation r +/- SEM")
    fig.suptitle("Self-picked peak vs fixed-lag correlations per ROI", fontweight="bold")
    save_figure(fig, out_dir / "roi_mean_self_picked_vs_fixed_lags")


def get_acc_30_60(future_df):
    required = ["fixed_lag_30_r", "fixed_lag_60_r", "MNI_z"]
    missing = [col for col in required if col not in future_df.columns]
    if missing:
        raise ValueError(f"Future run is missing required ACC columns: {missing}")

    acc = future_df[future_df["roi"] == "ACC"].copy()
    acc = acc.dropna(subset=required)
    acc = acc.rename(
        columns={
            "fixed_lag_30_r": "lag30_r",
            "fixed_lag_60_r": "lag60_r",
        }
    )
    return acc


def plot_acc_paired_lags(acc, out_dir):
    fig, ax = plt.subplots(figsize=(4.6, 5.2), constrained_layout=True)
    rng = np.random.default_rng(13)
    jitter = rng.normal(0.0, 0.035, len(acc))
    x0 = np.zeros(len(acc)) + jitter
    x1 = np.ones(len(acc)) + jitter

    for _, row in acc.iterrows():
        ax.plot(
            [0, 1],
            [row["lag30_r"], row["lag60_r"]],
            color="#b7b7b7",
            alpha=0.45,
            lw=0.8,
            zorder=1,
        )

    ax.scatter(x0, acc["lag30_r"], s=14, color="#777777", alpha=0.45, zorder=2)
    ax.scatter(x1, acc["lag60_r"], s=14, color="#777777", alpha=0.45, zorder=2)

    means = [acc["lag30_r"].mean(), acc["lag60_r"].mean()]
    sems = [stats.sem(acc["lag30_r"]), stats.sem(acc["lag60_r"])]
    ax.bar(
        [0, 1],
        means,
        width=0.46,
        color=[COLORS["lag_a"], COLORS["lag_b"]],
        edgecolor="black",
        linewidth=0.7,
        alpha=0.7,
        zorder=3,
    )
    ax.errorbar(
        [0, 1],
        means,
        yerr=sems,
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        zorder=4,
    )

    corr_r, corr_p = stats.pearsonr(acc["lag30_r"], acc["lag60_r"])
    paired_t, paired_p = stats.ttest_rel(acc["lag30_r"], acc["lag60_r"])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["lag 30", "lag 60"])
    ax.set_ylabel("ACC fixed-lag correlation r")
    ax.set_title(
        f"ACC lag 30 vs lag 60\nn={len(acc)}, Pearson r={corr_r:.2f}, p={corr_p:.3g}",
        loc="left",
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.98,
        f"paired t={paired_t:.2f}, p={paired_p:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    save_figure(fig, out_dir / "acc_lag30_lag60_paired_bar")


def add_high_flags(acc, high_quantile):
    acc = acc.copy()
    q30 = float(acc["lag30_r"].quantile(high_quantile))
    q60 = float(acc["lag60_r"].quantile(high_quantile))
    acc["high_lag30"] = acc["lag30_r"] >= q30
    acc["high_lag60"] = acc["lag60_r"] >= q60
    acc["high_both"] = acc["high_lag30"] & acc["high_lag60"]
    acc["high_group"] = "neither"
    acc.loc[acc["high_lag30"] & ~acc["high_lag60"], "high_group"] = "high 30 only"
    acc.loc[~acc["high_lag30"] & acc["high_lag60"], "high_group"] = "high 60 only"
    acc.loc[acc["high_both"], "high_group"] = "high 30 and 60"
    return acc, q30, q60


def plot_acc_mni_z(acc, q30, q60, high_quantile, out_dir):
    high_label = f"top {int(round((1 - high_quantile) * 100))}%"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.8),
        gridspec_kw={"width_ratios": [1.08, 0.92]},
        constrained_layout=True,
    )
    ax = axes[0]
    sc = ax.scatter(
        acc["lag30_r"],
        acc["lag60_r"],
        c=acc["MNI_z"],
        cmap="viridis",
        s=34,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.9,
    )
    hb = acc[acc["high_both"]]
    ax.scatter(
        hb["lag30_r"],
        hb["lag60_r"],
        facecolors="none",
        edgecolors=COLORS["accent"],
        linewidths=1.8,
        s=96,
        label=f"high both (n={len(hb)})",
    )
    ax.axvline(q30, color=COLORS["lag_a"], lw=1.0, ls="--")
    ax.axhline(q60, color=COLORS["lag_b"], lw=1.0, ls="--")
    ax.axvline(0, color="black", lw=0.6, alpha=0.5)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.set_xlabel("ACC r at fixed lag 30")
    ax.set_ylabel("ACC r at fixed lag 60")
    ax.set_title(f"Lag 30/60 overlap ({high_label} thresholds)", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("MNI z")

    ax = axes[1]
    group_order = ["high 30 only", "high 60 only", "high 30 and 60"]
    group_colors = [COLORS["lag_a"], COLORS["lag_b"], COLORS["accent"]]
    rng = np.random.default_rng(31)
    for i, (group, color) in enumerate(zip(group_order, group_colors)):
        vals = acc.loc[acc["high_group"] == group, "MNI_z"].dropna().to_numpy(float)
        if vals.size == 0:
            continue
        x = rng.normal(i, 0.045, vals.size)
        ax.scatter(x, vals, color=color, edgecolor="black", linewidth=0.4, s=45, alpha=0.85)
        ax.hlines(np.mean(vals), i - 0.22, i + 0.22, color="black", lw=1.3)
    labels = [
        f"{group}\n(n={(acc['high_group'] == group).sum()})" for group in group_order
    ]
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("MNI z")
    ax.set_title("MNI z-coordinate for high-lag ACC cells", loc="left")

    save_figure(fig, out_dir / "acc_high_lag_mni_z")


def save_figure(fig, stem):
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(summary, acc, q30, q60, high_quantile, out_dir):
    acc_cols = [
        "neuron_id",
        "subject_id",
        "subject_int",
        "cell_idx",
        "roi",
        "MNI_x",
        "MNI_y",
        "MNI_z",
        "lag30_r",
        "lag60_r",
        "high_lag30",
        "high_lag60",
        "high_both",
        "high_group",
    ]
    acc_cols = [col for col in acc_cols if col in acc.columns]
    acc[acc_cols].to_csv(out_dir / "acc_lag30_lag60_cells.csv", index=False)
    acc.loc[acc["high_both"], acc_cols].to_csv(
        out_dir / "acc_high_lag30_lag60_cells.csv", index=False
    )
    summary.to_csv(out_dir / "roi_mean_self_picked_vs_fixed_lags.csv", index=False)

    corr_r, corr_p = stats.pearsonr(acc["lag30_r"], acc["lag60_r"])
    paired_t, paired_p = stats.ttest_rel(acc["lag30_r"], acc["lag60_r"])
    lines = [
        "Spatial peak lag checks",
        "",
        f"ACC cells included: {len(acc)}",
        f"High threshold quantile: {high_quantile:.2f}",
        f"Lag 30 high threshold: {q30:.6f}",
        f"Lag 60 high threshold: {q60:.6f}",
        f"High lag 30 only: {int((acc['high_group'] == 'high 30 only').sum())}",
        f"High lag 60 only: {int((acc['high_group'] == 'high 60 only').sum())}",
        f"High lag 30 and 60: {int(acc['high_both'].sum())}",
        f"Pearson lag30 vs lag60: r={corr_r:.6f}, p={corr_p:.6g}",
        f"Paired t lag30 vs lag60: t={paired_t:.6f}, p={paired_p:.6g}",
        "",
        "Files:",
        "roi_mean_self_picked_vs_fixed_lags.png/pdf",
        "acc_lag30_lag60_paired_bar.png/pdf",
        "acc_high_lag_mni_z.png/pdf",
        "roi_mean_self_picked_vs_fixed_lags.csv",
        "acc_lag30_lag60_cells.csv",
        "acc_high_lag30_lag60_cells.csv",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--future-csv",
        type=Path,
        default=DEFAULT_RUNS["future_30_60"],
        help="per_cell.csv with fixed lags [30, 60].",
    )
    parser.add_argument(
        "--now-csv",
        type=Path,
        default=DEFAULT_RUNS["now_330_0"],
        help="per_cell.csv with fixed lags [330, 0].",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for figures and tables.",
    )
    parser.add_argument(
        "--high-quantile",
        type=float,
        default=0.75,
        help="Quantile used to label high lag-30 and high lag-60 ACC cells.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.high_quantile < 1:
        raise ValueError("--high-quantile must be between 0 and 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_frames = {}
    for label, path in [
        ("future_30_60", args.future_csv),
        ("now_330_0", args.now_csv),
    ]:
        df, lags = load_spatial_peaks_csv(path)
        run_frames[label] = {"df": df, "lags": lags, "path": path}
        print(f"Loaded {label}: {len(df)} ok cells, fixed lags {lags}")

    summary = build_roi_summary(run_frames)
    plot_roi_summary(summary, run_frames, args.out_dir)

    acc = get_acc_30_60(run_frames["future_30_60"]["df"])
    acc, q30, q60 = add_high_flags(acc, args.high_quantile)
    plot_acc_paired_lags(acc, args.out_dir)
    plot_acc_mni_z(acc, q30, q60, args.high_quantile, args.out_dir)
    write_summary(summary, acc, q30, q60, args.high_quantile, args.out_dir)

    print(f"Saved outputs to {args.out_dir}")
    print(
        "ACC high-high cells: "
        f"{int(acc['high_both'].sum())} / {len(acc)} "
        f"(lag30 >= {q30:.4f}, lag60 >= {q60:.4f})"
    )


if __name__ == "__main__":
    main()
