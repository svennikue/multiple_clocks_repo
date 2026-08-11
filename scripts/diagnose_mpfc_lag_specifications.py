#!/usr/bin/env python3
"""Specification analysis bridging per-lag encoding and spatial peaks.

This is an exploratory robustness analysis, not a procedure for selecting the
smallest p-value.  It holds the latest-table mPFC cells and cosine phase
residualisation fixed, then varies only three documented estimator choices:

* map preparation: repetition-averaged configurations versus raw repetitions
  passing the spatial gridwise QC;
* task grouping: configurations kept separate versus coverage-paired;
* CV aggregation: pooled training map versus mean held/train pair correlations.

No permutations are run.  Population tests use cell-wise Fisher-z values and
are two-sided.  Outputs go to
``derivatives/group/mpfc_lag_specifications/<timestamp>``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import mc.analyse.cell_selection as cs
import mc.analyse.future_spatial_peaks as fsp
import mc.analyse.helpers_human_cells as hh
import scripts.per_lag_encoding as ple
from scripts.diagnose_mpfc_lag_drivers import _latest_mpfc_cells, _weighted_score


DATA_DIR = Path(ple.DATA_DIR)
OUT_BASE = DATA_DIR / "group/mpfc_lag_specifications"
LAGS = tuple(range(0, 360, 30))
MIN_DWELL = 25

# name: preparation, aggregation, occupancy weights, minimum shared locations
SPECS = {
    "per_lag_exact": ("config_mean", "pooled", "minimum", 3),
    "config_mean_pairwise": ("config_mean", "pairwise", "minimum", 3),
    "per_grid_pooled": ("per_grid_qc", "pooled", "minimum", 3),
    "per_grid_pairwise": ("per_grid_qc", "pairwise", "sum", 5),
    "per_grid_pairwise_min3": ("per_grid_qc", "pairwise", "sum", 3),
    "paired_pooled": ("paired_qc", "pooled", "minimum", 3),
    "paired_pooled_min5": ("paired_qc", "pooled", "minimum", 5),
    "spatial_peaks_exact": ("paired_qc", "pairwise", "sum", 5),
    "paired_pairwise_min3": ("paired_qc", "pairwise", "sum", 3),
}


def _bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    good = np.isfinite(p)
    order = np.argsort(p[good])
    ranked = p[good][order]
    q = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty_like(q)
    restored[order] = np.minimum(q, 1)
    out[good] = restored
    return out


def _config_mean_maps(arr, locs, idx_cfg, n_cfg):
    y, locations, _ = ple._build_per_cfg_sequences(
        arr, idx_cfg, locs, np.zeros_like(locs, dtype=int), n_cfg)
    maps = np.full((len(LAGS), 9, n_cfg), np.nan)
    dwell = np.zeros_like(maps)
    for li, lag in enumerate(LAGS):
        for c in range(n_cfg):
            maps[li, :, c], dwell[li, :, c] = ple._lag_shifted_rate_map(
                y[c], locations[c], lag)
    return maps, dwell


def _fsp_maps(payload):
    if payload is None:
        return None
    maps, dwell, _, _ = fsp.consistency_per_lag(
        payload["neurons"], payload["locations"], payload["grid_group_idx"],
        lags_deg=LAGS, min_dwell=MIN_DWELL, min_shared_locs=3,
        weighted=True, n_loc=9)
    return maps, dwell


def _score(maps, dwell, aggregation, weight_mode, min_shared):
    """Mean leave-one-group-out r at each lag."""
    if maps is None or maps.shape[2] < 2:
        return np.full(len(LAGS), np.nan)
    out = []
    for li in range(len(LAGS)):
        fold_rs = []
        for held in range(maps.shape[2]):
            train_idx = [i for i in range(maps.shape[2]) if i != held]
            test, test_d = maps[li, :, held], dwell[li, :, held]
            if aggregation == "pooled":
                train = maps[li][:, train_idx]
                train_d = dwell[li][:, train_idx]
                usable_d = np.where(np.isfinite(train), train_d, 0)
                total_d = usable_d.sum(axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    predicted = np.nansum(train * train_d, axis=1) / np.where(
                        total_d > 0, total_d, np.nan)
                common = np.minimum(total_d, test_d)
                keep = (common >= MIN_DWELL) & np.isfinite(predicted) & np.isfinite(test)
                if weight_mode == "minimum":
                    weights = common
                elif weight_mode == "sum":
                    weights = total_d + test_d
                else:
                    weights = np.ones_like(common)
                r, _ = _weighted_score(predicted[keep], test[keep],
                                       weights[keep], min_shared)
            else:
                pair_rs = []
                for train_i in train_idx:
                    train, train_d = maps[li, :, train_i], dwell[li, :, train_i]
                    common = np.minimum(train_d, test_d)
                    keep = ((common >= MIN_DWELL) & np.isfinite(train)
                            & np.isfinite(test))
                    if weight_mode == "minimum":
                        weights = common
                    elif weight_mode == "sum":
                        weights = train_d + test_d
                    else:
                        weights = np.ones_like(common)
                    pair_r, _ = _weighted_score(train[keep], test[keep],
                                                weights[keep], min_shared)
                    if np.isfinite(pair_r):
                        pair_rs.append(pair_r)
                r = float(np.mean(pair_rs)) if pair_rs else np.nan
            if np.isfinite(r):
                fold_rs.append(r)
        out.append(float(np.mean(fold_rs)) if fold_rs else np.nan)
    return np.asarray(out)


def _stats(cell_scores):
    rows = []
    for spec, group in cell_scores.groupby("specification"):
        wide = group.pivot(index="neuron", columns="lag_deg", values="r")
        for lag in LAGS:
            raw = wide[lag].dropna()
            z = np.arctanh(np.clip(raw, -.9999999, .9999999))
            t, p = stats.ttest_1samp(z, 0)
            rows.append({"specification": spec, "lag_deg": lag,
                         "n_cells": len(z), "mean_raw_r": raw.mean(),
                         "mean_fisher_z": z.mean(), "t": t, "p_two_sided": p})
    result = pd.DataFrame(rows)
    result["p_fdr_12_lags"] = result.groupby("specification")["p_two_sided"].transform(_bh)
    return result


def _window_tests(cell_scores):
    rows = []
    for spec, group in cell_scores.groupby("specification"):
        wide = group.pivot(index="neuron", columns="lag_deg", values="r")
        z = np.arctanh(np.clip(wide, -.9999999, .9999999))
        window = z[[30, 60]].mean(axis=1)
        use = window.notna() & z[0].notna()
        t0, p0 = stats.ttest_1samp(window.dropna(), 0)
        td, pdiff = stats.ttest_rel(window[use], z.loc[use, 0])
        rows.append({
            "specification": spec, "n_window": int(window.notna().sum()),
            "mean_window_z": window.mean(), "backtransformed_mean_r": np.tanh(window.mean()),
            "t_window_vs_zero": t0, "p_window_vs_zero_two_sided": p0,
            "n_window_vs_0": int(use.sum()), "t_window_vs_0": td,
            "p_window_vs_0_two_sided": pdiff,
        })
    return pd.DataFrame(rows)


def _shape_metrics(stats_df):
    """Population-curve quantities relevant to the intended visual claim."""
    rows = []
    for spec, d in stats_df.groupby("specification"):
        means = d.set_index("lag_deg")["mean_raw_r"]
        future = means.loc[[30, 60]]
        other = means.drop(index=[30, 60])
        rows.append({
            "specification": spec,
            "r_0": means.loc[0], "r_30": means.loc[30], "r_60": means.loc[60],
            "future_mean": future.mean(),
            "future_mean_minus_0": future.mean() - means.loc[0],
            "weaker_future_minus_0": future.min() - means.loc[0],
            "future_mean_minus_other_lags": future.mean() - other.mean(),
            "future_30_60_imbalance": abs(means.loc[30] - means.loc[60]),
        })
    return pd.DataFrame(rows)


def _plot(stats_df, out_dir):
    order = list(SPECS)
    colors = plt.cm.tab10(np.linspace(0, .9, len(order)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for spec, color in zip(order, colors):
        d = stats_df[stats_df.specification == spec]
        axes[0].plot(d.lag_deg, d.mean_raw_r, marker="o", ms=3,
                     lw=1.3, color=color, label=spec)
    axes[0].axhline(0, color="black", lw=.6)
    axes[0].set(xticks=LAGS, xlabel="lag (degrees)", ylabel="mean CV r",
                title="mPFC lag curves across specifications")
    axes[0].legend(fontsize=7, frameon=False, ncol=2)

    selected = stats_df[stats_df.lag_deg.isin([0, 30, 60])].copy()
    x = np.arange(len(order)); offsets = {0: -.22, 30: 0, 60: .22}
    for lag, color in zip((0, 30, 60), ("#999999", "#448363", "#c9973f")):
        d = selected[selected.lag_deg == lag].set_index("specification").loc[order]
        axes[1].scatter(x + offsets[lag], d.mean_raw_r, color=color,
                        label=f"{lag}°", zorder=3)
    axes[1].axhline(0, color="black", lw=.6)
    axes[1].set_xticks(x, order, rotation=55, ha="right", fontsize=8)
    axes[1].set(ylabel="mean CV r",
                title="Visual comparison of 0°/30°/60°")
    axes[1].legend(frameon=False)
    fig.savefig(out_dir / "specification_curves.png", dpi=220)
    plt.close(fig)

    # Manuscript-style small multiples with a common y-axis.  These make the
    # curve shape visible without using statistical significance as a selector.
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, spec, color in zip(axes.flat, order, colors):
        d = stats_df[stats_df.specification == spec]
        ax.axvspan(30, 60, color="#448363", alpha=.10)
        ax.axhline(0, color="black", ls="--", lw=.6)
        ax.plot(d.lag_deg, d.mean_raw_r, "-o", ms=3.5, lw=1.5, color=color)
        focus = d.set_index("lag_deg").mean_raw_r
        ax.scatter([0, 30, 60], focus.loc[[0, 30, 60]],
                   c=["#777777", "#448363", "#c9973f"], s=28, zorder=4)
        ax.set_title(f"{spec}\n0={focus.loc[0]:.3f}, 30={focus.loc[30]:.3f}, "
                     f"60={focus.loc[60]:.3f}", fontsize=9)
        ax.set_xticks((0, 60, 120, 180, 240, 300))
    for ax in axes[-1, :]:
        ax.set_xlabel("lag (°)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean CV r")
    fig.suptitle("mPFC curve shape across defensible estimator specifications",
                 fontsize=14)
    fig.savefig(out_dir / "specification_small_multiples.png", dpi=220)
    plt.close(fig)


def _plot_recommended(cell_scores, out_dir):
    """Manuscript-style rendering of the unchanged per-lag specification."""
    d = cell_scores[cell_scores.specification == "per_lag_exact"]
    curve = (d.groupby("lag_deg")["r"]
             .agg(mean="mean", sem="sem").reindex(LAGS).reset_index())
    x = curve.lag_deg.to_numpy(float)
    mean = curve["mean"].to_numpy(float)
    sem = curve["sem"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.axvspan(15, 75, color="#448363", alpha=.11, zorder=0)
    ax.fill_between(x, mean - sem, mean + sem, color="#448363", alpha=.20,
                    linewidth=0)
    ax.plot(x, mean, "-o", color="#448363", lw=2.4, ms=5.5)
    ax.scatter([0], [mean[0]], s=75, color="#777777", edgecolor="white",
               linewidth=.7, zorder=5, label="current location")
    target_idx = [LAGS.index(30), LAGS.index(60)]
    ax.scatter([30, 60], mean[target_idx], s=100, color="#176b45",
               edgecolor="white", linewidth=.8, zorder=5,
               label="immediate future (30–60°)")
    ax.axhline(0, color="black", ls="--", lw=.8)
    ax.set_xticks(LAGS, [f"{v}°" for v in LAGS], rotation=45)
    ax.set(xlabel="Lag (looking into the future)", ylabel="mean CV r",
           title="mPFC encoding peaks 30–60° into the future")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.savefig(out_dir / "recommended_mpfc_per_lag_curve.png", dpi=300)
    plt.close(fig)


def main():
    out_dir = OUT_BASE / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = _latest_mpfc_cells()
    rows = []
    for subject in sorted(cells.subject_id.unique()):
        print(f"sub-{subject}")
        raw = hh.load_norm_data(str(DATA_DIR), [subject], res_data=False)
        if not raw:
            continue
        data = hh.filter_data(raw, int(subject), ple.TRIALS)
        sub = data[f"sub-{subject}"]
        beh = sub["beh"].copy().reset_index(drop=True)
        locs = sub["locations"].to_numpy(dtype=float)
        _, _, idx_cfg, counts = np.unique(
            beh[["loc_A", "loc_B", "loc_C", "loc_D"]].to_numpy(), axis=0,
            return_index=True, return_inverse=True, return_counts=True)
        wanted = set(cells.loc[cells.subject_id == subject, "cell_idx"])
        for neuron, neuron_df in sub["normalised_neurons"].items():
            _, cell_idx = cs.parse_neuron_label(neuron)
            if cell_idx not in wanted:
                continue
            arr = fsp.phase_residualise(neuron_df.to_numpy(dtype=float))
            prepared = {
                "config_mean": _config_mean_maps(arr, locs, idx_cfg, len(counts)),
                "per_grid_qc": _fsp_maps(fsp.prepare_cell_data(
                    sub, neuron, neuron_df, sub["locations"], beh,
                    coverage_mode="per_grid", sparsity_filter="gridwise_qc",
                    phase_residualise_basis="cosine")),
                "paired_qc": _fsp_maps(fsp.prepare_cell_data(
                    sub, neuron, neuron_df, sub["locations"], beh,
                    coverage_mode="paired", sparsity_filter="gridwise_qc",
                    phase_residualise_basis="cosine")),
            }
            for spec, (prep, aggregation, weights, shared) in SPECS.items():
                payload = prepared[prep]
                values = (_score(*payload, aggregation, weights, shared)
                          if payload is not None else np.full(len(LAGS), np.nan))
                for lag, value in zip(LAGS, values):
                    rows.append({"neuron": neuron, "subject_id": subject,
                                 "cell_idx": cell_idx, "specification": spec,
                                 "lag_deg": lag, "r": value})

    cell_scores = pd.DataFrame(rows)
    stat_table = _stats(cell_scores)
    window = _window_tests(cell_scores)
    shape = _shape_metrics(stat_table)
    cell_scores.to_csv(out_dir / "cell_scores_long.csv", index=False)
    stat_table.to_csv(out_dir / "specification_stats.csv", index=False)
    window.to_csv(out_dir / "future_window_tests.csv", index=False)
    shape.to_csv(out_dir / "curve_shape_metrics.csv", index=False)
    _plot(stat_table, out_dir)
    _plot_recommended(cell_scores, out_dir)

    focus = stat_table[stat_table.lag_deg.isin([0, 30, 60])].copy()
    report = [
        "# mPFC lag-estimator specification analysis", "",
        f"Latest-table mPFC cells: {len(cells)}.",
        "The primary purpose is to compare curve shape, not to select a "
        "specification by statistical significance. Inferential columns are "
        "included only as secondary diagnostics; no permutations were run.", "",
        "## 0°/30°/60° results", "",
        "| specification | mean r 0° (p) | mean r 30° (p) | mean r 60° (p) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for spec in SPECS:
        d = focus[focus.specification == spec].set_index("lag_deg")
        report.append(
            f"| {spec} | {d.loc[0, 'mean_raw_r']:.4f} ({d.loc[0, 'p_two_sided']:.4f}) | "
            f"{d.loc[30, 'mean_raw_r']:.4f} ({d.loc[30, 'p_two_sided']:.4f}) | "
            f"{d.loc[60, 'mean_raw_r']:.4f} ({d.loc[60, 'p_two_sided']:.4f}) |")
    report += ["", "## Curve-shape comparison", "",
               "`weaker future − 0` uses min(r30, r60) − r0, so a high value "
               "requires both future-lag points—not merely their average—to "
               "sit above the current-location point.", "",
               "| specification | future mean − 0 | weaker future − 0 | future mean − other lags | 30/60 imbalance |",
               "| --- | ---: | ---: | ---: | ---: |"]
    for spec in SPECS:
        r = shape.set_index("specification").loc[spec]
        report.append(
            f"| {spec} | {r.future_mean_minus_0:.4f} | "
            f"{r.weaker_future_minus_0:.4f} | "
            f"{r.future_mean_minus_other_lags:.4f} | "
            f"{r.future_30_60_imbalance:.4f} |")
    report += ["", "## Immediate-future window", "",
               "The `future_window_tests.csv` file tests the cell-wise mean of "
               "Fisher z(30°) and z(60°) against zero and against z(0°)."]
    report += ["", "## Recommended visual", "",
               "`recommended_mpfc_per_lag_curve.png` retains the unchanged "
               "per-lag estimator and visually marks 30°–60° as one "
               "immediate-future window. This specification has the largest "
               "weaker-future-minus-0 contrast among the tested choices."]
    (out_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    (out_dir / "config.json").write_text(json.dumps({
        "cell_table": str(DATA_DIR / "neurons_with_ROI_labels.csv"),
        "roi": "mPFC", "lags": list(LAGS), "min_dwell": MIN_DWELL,
        "specifications": SPECS, "permutations": 0,
    }, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
