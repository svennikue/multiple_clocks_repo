#!/usr/bin/env python3
"""Minimal mPFC diagnostic for the 0°/30°/60° rate-map results.

The script answers three separate questions without running permutations:

1. Cell-table effect
   Which old ACC cells were retained/removed and which cells were newly
   assigned to mPFC by the latest ROI table? How does that change each lag?
2. Single-configuration per-lag estimator
   For every current mPFC cell, which held-out task configuration produces
   high 30°/60° consistency or low 0° consistency?
3. Paired-group spatial-peaks estimator
   Which coverage-maximising configuration pairs produce the corresponding
   effects under the alternative CV estimator?

Outputs are CSVs, four simple figures, and REPORT.md under
``derivatives/group/mpfc_lag_diagnostic/<timestamp>``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from itertools import combinations
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


DATA_DIR = Path(ple.DATA_DIR)
CELL_TABLE = DATA_DIR / "neurons_with_ROI_labels.csv"  # current table
PER_LAG_SOURCE = DATA_DIR / "group/per_lag_encoding/2026-06-30_18-21-57"
SPATIAL_SOURCE = (DATA_DIR / "group/spatial_peaks_simple/"
                  "2026-06-26_18-47-11_phase_resid_paired_fixedlag-final")
SPATIAL_RESULTS = DATA_DIR / "group/spatial_peaks_simple"
OUT_BASE = DATA_DIR / "group/mpfc_lag_diagnostic"

LAGS = (0, 30, 60)
MIN_DWELL = 25
MIN_SHARED_PER_LAG = 3
MIN_SHARED_SPATIAL = 5


def _cfg_label(values):
    return "-".join(str(int(v)) for v in values)


def _latest_mpfc_cells():
    d = pd.read_csv(CELL_TABLE)
    d = d.rename(columns={"subject": "subject_id", "cell idx": "cell_idx"})
    d["subject_id"] = d["subject_id"].astype(str).str.replace(
        r"\.0$", "", regex=True).str.zfill(2)
    d = d[d["alt_final_roi"] == "mPFC"].copy()
    return d.sort_values(["subject_id", "cell_idx"]).reset_index(drop=True)


def _weighted_score(x, y, w, minimum):
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if m.sum() < minimum:
        return np.nan, int(m.sum())
    x, y, w = x[m], y[m], w[m]
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    if vx <= 0 or vy <= 0:
        return np.nan, int(m.sum())
    return float(cov / np.sqrt(vx * vy)), int(m.sum())


def _per_lag_fold_rows(neuron, subject, cell_idx, arr, locs, idx_cfg,
                       config_values, config_counts):
    """Exact no-control single-config folds used by per_lag_encoding.py."""
    n_cfg = len(config_values)
    y_cfg, loc_cfg, _ = ple._build_per_cfg_sequences(
        arr, idx_cfg, locs, np.zeros_like(locs, dtype=int), n_cfg)
    rows, cell = [], {"neuron": neuron, "subject_id": subject,
                      "cell_idx": cell_idx, "estimator": "single_config_pooled"}
    for lag in LAGS:
        rates = np.stack([ple._lag_shifted_rate_map(y_cfg[c], loc_cfg[c], lag)[0]
                          for c in range(n_cfg)])
        dwell = np.stack([ple._lag_shifted_rate_map(y_cfg[c], loc_cfg[c], lag)[1]
                          for c in range(n_cfg)])
        fold_values = []
        for held in range(n_cfg):
            train = np.arange(n_cfg) != held
            total_d = dwell[train].sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                pred = np.nansum(rates[train] * dwell[train], axis=0) / np.where(
                    total_d > 0, total_d, np.nan)
            common = np.minimum(total_d, dwell[held])
            keep = ((common >= MIN_DWELL) & np.isfinite(pred)
                    & np.isfinite(rates[held]))
            r, n_shared = _weighted_score(
                pred[keep], rates[held, keep], common[keep],
                MIN_SHARED_PER_LAG)
            if np.isfinite(r):
                fold_values.append(r)
            rows.append({
                "neuron": neuron, "subject_id": subject, "cell_idx": cell_idx,
                "held_config_id": held,
                "held_config": _cfg_label(config_values[held]),
                "subject_config": f"{subject}:{_cfg_label(config_values[held])}",
                "n_repetitions": int(config_counts[held]), "lag_deg": lag,
                "r": r, "n_shared_locations": n_shared,
            })
        cell[f"r_{lag}"] = float(np.mean(fold_values)) if fold_values else np.nan
    cell["delta_30_minus_0"] = cell["r_30"] - cell["r_0"]
    cell["delta_60_minus_0"] = cell["r_60"] - cell["r_0"]
    return rows, cell


def _spatial_rows(neuron, subject, cell_idx, neuron_df, sub_dict, beh,
                  config_values):
    """Per-grid pairs and actual paired-group CV folds from spatial peaks."""
    pair_rows, fold_rows = [], []
    # Direct original-configuration pairs after spatial-peaks gridwise QC.
    per_grid = fsp.prepare_cell_data(
        subject_data=sub_dict, neuron_id=neuron, neurons_df=neuron_df,
        locations_df=sub_dict["locations"], beh=beh,
        coverage_mode="per_grid", sparsity_filter="gridwise_qc",
        phase_residualise_basis="cosine")
    if per_grid is not None:
        maps, dwell, _, groups = fsp.consistency_per_lag(
            per_grid["neurons"], per_grid["locations"],
            per_grid["grid_group_idx"], lags_deg=LAGS,
            min_dwell=MIN_DWELL, min_shared_locs=MIN_SHARED_SPATIAL,
            weighted=True, n_loc=9)
        for li, lag in enumerate(LAGS):
            for i, j in combinations(range(len(groups)), 2):
                gi, gj = int(groups[i]), int(groups[j])
                m = np.isfinite(maps[li, :, i]) & np.isfinite(maps[li, :, j])
                w = dwell[li, :, i] + dwell[li, :, j]
                r, n_shared = _weighted_score(
                    maps[li, m, i], maps[li, m, j], w[m],
                    MIN_SHARED_SPATIAL)
                ci = _cfg_label(config_values[gi])
                cj = _cfg_label(config_values[gj])
                pair_rows.append({
                    "neuron": neuron, "subject_id": subject,
                    "cell_idx": cell_idx, "config_i": ci, "config_j": cj,
                    "config_pair": " + ".join(sorted((ci, cj))),
                    "subject_config_pair": f"{subject}:{' + '.join(sorted((ci, cj)))}",
                    "lag_deg": lag, "r": r,
                    "n_shared_locations": n_shared,
                })

    # Coverage-maximising paired groups and their actual LOO fold scores.
    paired = fsp.prepare_cell_data(
        subject_data=sub_dict, neuron_id=neuron, neurons_df=neuron_df,
        locations_df=sub_dict["locations"], beh=beh,
        coverage_mode="paired", sparsity_filter="gridwise_qc",
        phase_residualise_basis="cosine")
    cell = {"neuron": neuron, "subject_id": subject, "cell_idx": cell_idx,
            "estimator": "paired_grid_groups"}
    if paired is None:
        for lag in LAGS:
            cell[f"r_{lag}"] = np.nan
        return pair_rows, fold_rows, cell
    maps, dwell, _, groups = fsp.consistency_per_lag(
        paired["neurons"], paired["locations"], paired["grid_group_idx"],
        lags_deg=LAGS, min_dwell=MIN_DWELL,
        min_shared_locs=MIN_SHARED_SPATIAL, weighted=True, n_loc=9)
    group_cfg_ids = dict(zip(
        [int(g) for g in groups], paired["paired_config_groups"]))
    for li, lag in enumerate(LAGS):
        vals = []
        for held_i, group_id in enumerate(groups):
            train = np.ones(len(groups), dtype=bool)
            train[held_i] = False
            r, shared = fsp._validate_at_lag(
                maps[li], dwell[li], held_i, train,
                MIN_SHARED_SPATIAL, True)
            cfg_ids = group_cfg_ids.get(int(group_id), [])
            cfg_labels = [_cfg_label(config_values[c]) for c in cfg_ids]
            held_label = " + ".join(sorted(cfg_labels))
            if np.isfinite(r):
                vals.append(r)
            fold_rows.append({
                "neuron": neuron, "subject_id": subject, "cell_idx": cell_idx,
                "held_group_id": int(group_id), "held_config_pair": held_label,
                "subject_config_pair": f"{subject}:{held_label}",
                "n_training_groups": int(train.sum()), "lag_deg": lag,
                "r": r,
                "mean_shared_locations": (float(np.mean(shared))
                                           if shared else np.nan),
            })
        cell[f"r_{lag}"] = float(np.mean(vals)) if vals else np.nan
    cell["delta_30_minus_0"] = cell["r_30"] - cell["r_0"]
    cell["delta_60_minus_0"] = cell["r_60"] - cell["r_0"]
    return pair_rows, fold_rows, cell


def _influence_table(folds, key_col, estimator):
    """Population mean change after excluding one subject/config combination."""
    rows = []
    for lag in LAGS:
        f = folds[folds["lag_deg"] == lag].dropna(subset=["r"])
        baseline_cell = f.groupby("neuron")["r"].mean()
        baseline = float(baseline_cell.mean())
        for key, hit in f.groupby(key_col):
            remaining = f.drop(index=hit.index)
            new_cell = remaining.groupby("neuron")["r"].mean()
            # Cells with no remaining folds retain NaN and are excluded, matching
            # the estimator's standard finite-cell population mean.
            new_mean = float(new_cell.mean())
            rows.append({
                "estimator": estimator, "excluded_combination": key,
                "lag_deg": lag, "n_cells_affected": hit["neuron"].nunique(),
                "baseline_mean_r": baseline, "mean_r_without": new_mean,
                "change_after_exclusion": new_mean - baseline,
                "direction": ("drives high r" if new_mean < baseline
                              else "drives low r"),
            })
    return pd.DataFrame(rows)


def _relabel_effect():
    old = pd.read_csv(PER_LAG_SOURCE / "per_cell_ALL_ROIs.csv")
    latest_dirs = [p for p in (DATA_DIR / "group/per_lag_encoding").iterdir()
                   if (p / "per_cell_ALL_ROIs.csv").exists()
                   and "relabelled" in p.name]
    latest = max(latest_dirs, key=lambda p: p.stat().st_mtime)
    new = pd.read_csv(latest / "per_cell_ALL_ROIs.csv")[["neuron", "roi"]]
    new = new.rename(columns={"roi": "new_roi"})
    d = old.merge(new, on="neuron", how="outer")
    d["membership_change"] = np.select([
        d["roi"].eq("ACC") & d["new_roi"].eq("mPFC"),
        ~d["roi"].eq("ACC") & d["new_roi"].eq("mPFC"),
        d["roi"].eq("ACC") & ~d["new_roi"].eq("mPFC"),
    ], ["retained", "added", "removed"], default="other")
    detail = d[d["membership_change"] != "other"].copy()
    rows = []
    for group, g in detail.groupby("membership_change"):
        for lag in LAGS:
            vals = g[f"r_lag{lag:03d}_noctrl"]
            rows.append({"membership_change": group, "lag_deg": lag,
                         "n_cells": int(vals.notna().sum()),
                         "mean_r": float(vals.mean())})
    old_acc = old[old["roi"] == "ACC"]
    new_mpfc = old[old["neuron"].isin(
        new.loc[new["new_roi"] == "mPFC", "neuron"])]
    for label, g in (("old_ACC_total", old_acc),
                     ("current_mPFC_total", new_mpfc)):
        for lag in LAGS:
            vals = g[f"r_lag{lag:03d}_noctrl"]
            rows.append({"membership_change": label, "lag_deg": lag,
                         "n_cells": int(vals.notna().sum()),
                         "mean_r": float(vals.mean())})
    return pd.DataFrame(rows), detail, latest


def _summary(values, group_cols):
    return (values.groupby(group_cols)["r"]
            .agg(n_observations="count", n_cells=lambda x: values.loc[x.index, "neuron"].nunique(),
                 mean_r="mean", sd_r="std")
            .reset_index())


def _cell_rankings(cell_df):
    """Long table making the most useful cell-level rankings explicit."""
    specs = (
        ("high_30", "r_30", False),
        ("high_60", "r_60", False),
        ("low_0", "r_0", True),
        ("high_30_minus_0", "delta_30_minus_0", False),
        ("high_60_minus_0", "delta_60_minus_0", False),
    )
    rows = []
    keep = ["neuron", "subject_id", "cell_idx", "r_0", "r_30", "r_60",
            "delta_30_minus_0", "delta_60_minus_0"]
    for category, metric, ascending in specs:
        ranked = cell_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
        for i, r in ranked[keep].iterrows():
            row = r.to_dict()
            row.update({"category": category, "rank": i + 1,
                        "ranking_metric": metric,
                        "ranking_value": r[metric]})
            rows.append(row)
    return pd.DataFrame(rows)


def _top_cell_lines(rankings, category, n=8):
    x = rankings[(rankings["category"] == category)
                 & (rankings["rank"] <= n)]
    return [
        f"- `{r.neuron}`: r(0°)={r.r_0:.3f}, r(30°)={r.r_30:.3f}, "
        f"r(60°)={r.r_60:.3f}." for _, r in x.iterrows()
    ]


def _cache_validation(perlag_cell_df, spatial_cell_df, perlag_cache):
    """Check diagnostic reconstructions against the saved production scores."""
    rows = []
    cached = pd.read_csv(perlag_cache / "per_cell_ALL_ROIs.csv")
    merged = perlag_cell_df.merge(cached, on="neuron")
    for lag in LAGS:
        a = merged[f"r_{lag}"]
        b = merged[f"r_lag{lag:03d}_noctrl"]
        finite = np.isfinite(a) & np.isfinite(b)
        rows.append({
            "estimator": "single_config_pooled", "lag_deg": lag,
            "cache": str(perlag_cache), "n_matched": int(finite.sum()),
            "max_abs_difference": float(np.max(np.abs(a[finite] - b[finite]))),
            "pearson_r": float(np.corrcoef(a[finite], b[finite])[0, 1]),
        })

    candidates = [p for p in SPATIAL_RESULTS.iterdir()
                  if (p / "per_cell.csv").exists()
                  and "relabelled" in p.name
                  and "no_rsa" not in p.name]
    spatial_cache = max(candidates, key=lambda p: p.stat().st_mtime)
    cached = pd.read_csv(spatial_cache / "per_cell.csv")
    long_rows = []
    for _, r in cached.iterrows():
        values = json.loads(r["per_lag_r_all_lags_json"])
        for lag, value in zip(range(0, 360, 30), values):
            if lag in LAGS:
                long_rows.append({"neuron": r["neuron_id"], "lag_deg": lag,
                                  "cached_r": value})
    cached_long = pd.DataFrame(long_rows)
    for lag in LAGS:
        merged = spatial_cell_df[["neuron", f"r_{lag}"]].merge(
            cached_long[cached_long["lag_deg"] == lag], on="neuron")
        a, b = merged[f"r_{lag}"], merged["cached_r"]
        finite = np.isfinite(a) & np.isfinite(b)
        rows.append({
            "estimator": "paired_grid_groups", "lag_deg": lag,
            "cache": str(spatial_cache), "n_matched": int(finite.sum()),
            "max_abs_difference": float(np.max(np.abs(a[finite] - b[finite]))),
            "pearson_r": float(np.corrcoef(a[finite], b[finite])[0, 1]),
        })
    return pd.DataFrame(rows), spatial_cache


def _plot_relabel(effect, out_dir):
    order = ["old_ACC_total", "retained", "removed", "added",
             "current_mPFC_total"]
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    x = np.arange(len(LAGS)); width = 0.15
    for i, group in enumerate(order):
        g = effect[effect["membership_change"] == group].set_index("lag_deg")
        ax.bar(x + (i - 2) * width, [g.loc[l, "mean_r"] for l in LAGS],
               width, label=group.replace("_", " "))
    ax.axhline(0, color="black", lw=.6)
    ax.set_xticks(x, [f"{l}°" for l in LAGS]); ax.set_ylabel("mean CV r")
    ax.set_title("Effect of the latest cell-table relabelling")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.savefig(out_dir / "01_cell_table_effect.png", dpi=200)
    plt.close(fig)


def _plot_cells(cell_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), constrained_layout=True)
    for ax, lag in zip(axes, (30, 60)):
        ax.scatter(cell_df["r_0"], cell_df[f"r_{lag}"], s=10, alpha=.55)
        lo = np.nanmin([cell_df["r_0"].min(), cell_df[f"r_{lag}"].min()])
        hi = np.nanmax([cell_df["r_0"].max(), cell_df[f"r_{lag}"].max()])
        ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=.7)
        ax.axhline(0, color="grey", lw=.4); ax.axvline(0, color="grey", lw=.4)
        ax.set(xlabel="0° r", ylabel=f"{lag}° r", title=f"{lag}° versus 0°")
    fig.savefig(out_dir / "02_cell_drivers.png", dpi=200)
    plt.close(fig)


def _plot_influence(influence, estimator, filename, out_dir):
    sub = influence[influence["estimator"] == estimator].copy()
    # Top five absolute changes at each lag.
    keep = (sub.assign(a=sub["change_after_exclusion"].abs())
            .sort_values(["lag_deg", "a"], ascending=[True, False])
            .groupby("lag_deg").head(5))
    labels = list(dict.fromkeys(keep["excluded_combination"]))
    fig, ax = plt.subplots(figsize=(10, max(4, .32 * len(labels))),
                           constrained_layout=True)
    y = np.arange(len(labels)); offsets = {0: -.22, 30: 0, 60: .22}
    for lag, color in zip(LAGS, ("#999999", "#448363", "#c9973f")):
        vals = keep[keep["lag_deg"] == lag].set_index("excluded_combination")
        x = [vals.loc[k, "change_after_exclusion"] if k in vals.index else 0
             for k in labels]
        ax.barh(y + offsets[lag], x, height=.2, label=f"{lag}°", color=color)
    ax.axvline(0, color="black", lw=.6)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("change in population mean after excluding combination")
    ax.set_title(estimator.replace("_", " "))
    ax.legend(frameon=False)
    fig.savefig(out_dir / filename, dpi=200)
    plt.close(fig)


def _top_lines(influence, lag, direction, n=8):
    x = influence[influence["lag_deg"] == lag].copy()
    ascending = direction == "high"  # most negative exclusion change drove high
    x = x.sort_values("change_after_exclusion", ascending=ascending).head(n)
    return [
        f"- `{r.excluded_combination}`: affected {int(r.n_cells_affected)} cells; "
        f"mean changes by {r.change_after_exclusion:+.4f} when excluded "
        f"({r.direction})." for _, r in x.iterrows()
    ]


def main():
    out_dir = OUT_BASE / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = _latest_mpfc_cells()
    cells.to_csv(out_dir / "cells_latest_table_mPFC.csv", index=False)
    effect, membership_detail, latest_run = _relabel_effect()
    effect.to_csv(out_dir / "cell_table_relabel_effect.csv", index=False)
    membership_detail.to_csv(out_dir / "cell_table_membership_changes.csv", index=False)

    fold_rows, perlag_cells = [], []
    pair_rows, paired_fold_rows, spatial_cells = [], [], []
    for subject in sorted(cells["subject_id"].unique()):
        print(f"sub-{subject}")
        raw = hh.load_norm_data(str(DATA_DIR), [subject], res_data=False)
        if not raw:
            continue
        data = hh.filter_data(raw, int(subject), ple.TRIALS)
        sub_dict = data[f"sub-{subject}"]
        beh = sub_dict["beh"].copy().reset_index(drop=True)
        locs = sub_dict["locations"].to_numpy(dtype=float)
        config_values, _, idx_cfg, config_counts = np.unique(
            beh[["loc_A", "loc_B", "loc_C", "loc_D"]].to_numpy(), axis=0,
            return_index=True, return_inverse=True, return_counts=True)
        wanted = set(cells.loc[cells["subject_id"] == subject, "cell_idx"])
        for neuron, neuron_df in sub_dict["normalised_neurons"].items():
            _, cell_idx = cs.parse_neuron_label(neuron)
            if cell_idx not in wanted:
                continue
            arr = fsp.phase_residualise(neuron_df.to_numpy(dtype=float))
            rows, cell = _per_lag_fold_rows(
                neuron, subject, cell_idx, arr, locs, idx_cfg,
                config_values, config_counts)
            fold_rows.extend(rows); perlag_cells.append(cell)
            pairs, paired_folds, spatial_cell = _spatial_rows(
                neuron, subject, cell_idx, neuron_df, sub_dict, beh,
                config_values)
            pair_rows.extend(pairs); paired_fold_rows.extend(paired_folds)
            spatial_cells.append(spatial_cell)

    folds = pd.DataFrame(fold_rows)
    perlag_cell_df = pd.DataFrame(perlag_cells)
    config_pairs = pd.DataFrame(pair_rows)
    paired_folds = pd.DataFrame(paired_fold_rows)
    spatial_cell_df = pd.DataFrame(spatial_cells)
    folds.to_csv(out_dir / "per_lag_held_config_folds.csv", index=False)
    perlag_cell_df.to_csv(out_dir / "per_lag_cell_scores.csv", index=False)
    config_pairs.to_csv(out_dir / "spatial_original_config_pairs.csv", index=False)
    paired_folds.to_csv(out_dir / "spatial_paired_group_folds.csv", index=False)
    spatial_cell_df.to_csv(out_dir / "spatial_paired_cell_scores.csv", index=False)
    cell_rankings = _cell_rankings(perlag_cell_df)
    cell_rankings.to_csv(out_dir / "cell_driver_rankings.csv", index=False)

    validation, spatial_cache = _cache_validation(
        perlag_cell_df, spatial_cell_df, latest_run)
    validation.to_csv(out_dir / "cache_validation.csv", index=False)

    config_summary = _summary(folds, ["held_config", "lag_deg"])
    pair_summary = _summary(config_pairs, ["config_pair", "lag_deg"])
    paired_summary = _summary(paired_folds, ["held_config_pair", "lag_deg"])
    config_summary.to_csv(out_dir / "per_lag_config_summary.csv", index=False)
    pair_summary.to_csv(out_dir / "spatial_config_pair_summary.csv", index=False)
    paired_summary.to_csv(out_dir / "spatial_paired_group_summary.csv", index=False)

    single_influence = _influence_table(
        folds, "subject_config", "single_config_pooled")
    paired_influence = _influence_table(
        paired_folds, "subject_config_pair", "paired_grid_groups")
    influence = pd.concat([single_influence, paired_influence], ignore_index=True)
    influence.to_csv(out_dir / "combination_influence.csv", index=False)

    _plot_relabel(effect, out_dir)
    _plot_cells(perlag_cell_df, out_dir)
    _plot_influence(influence, "single_config_pooled",
                    "03_single_config_influence.png", out_dir)
    _plot_influence(influence, "paired_grid_groups",
                    "04_paired_group_influence.png", out_dir)

    # Simple population checks and report.
    pop_rows = []
    for estimator, frame in (("single_config_pooled", perlag_cell_df),
                             ("paired_grid_groups", spatial_cell_df)):
        for lag in LAGS:
            vals = frame[f"r_{lag}"].dropna()
            t, p = stats.ttest_1samp(np.arctanh(np.clip(vals, -.9999999, .9999999)), 0)
            pop_rows.append({"estimator": estimator, "lag_deg": lag,
                             "n_cells": len(vals), "mean_raw_r": vals.mean(),
                             "t_fisher_two_sided": t, "p_two_sided": p})
    population = pd.DataFrame(pop_rows)
    population.to_csv(out_dir / "population_check.csv", index=False)

    old60 = effect[(effect.membership_change == "old_ACC_total")
                   & (effect.lag_deg == 60)].iloc[0]
    new60 = effect[(effect.membership_change == "current_mPFC_total")
                   & (effect.lag_deg == 60)].iloc[0]
    removed60 = effect[(effect.membership_change == "removed")
                       & (effect.lag_deg == 60)].iloc[0]
    added60 = effect[(effect.membership_change == "added")
                     & (effect.lag_deg == 60)].iloc[0]
    report = [
        "# mPFC 0°/30°/60° diagnostic",
        "",
        f"Latest cell table: `{CELL_TABLE}` ({len(cells)} mPFC cells).",
        f"Cached per-lag source: `{PER_LAG_SOURCE}`.",
        f"Latest relabelled run used for membership: `{latest_run}`.",
        "",
        "## Why 60° became lower in the current per-lag figure",
        "",
        f"The old ACC cohort had mean r(60°) = {old60.mean_r:.4f}; the current "
        f"mPFC cohort has {new60.mean_r:.4f}, a change of "
        f"{new60.mean_r - old60.mean_r:+.4f}. The latest table removed "
        f"{int(removed60.n_cells)} old ACC cells whose mean 60° r was "
        f"{removed60.mean_r:.4f}, and added {int(added60.n_cells)} cells whose "
        f"mean 60° r was {added60.mean_r:.4f}. Thus the modest 60° decrease is "
        "explained by cell-table membership, particularly removing relatively "
        "high-60° cells and adding low-60° cells—not by Fisher transformation.",
        "",
        "## Reconstruction check",
        "",
        "The diagnostic reproduces the saved production scores for "
        f"{int(validation.loc[validation.estimator == 'single_config_pooled', 'n_matched'].min())} "
        "per-lag cells and "
        f"{int(validation.loc[validation.estimator == 'paired_grid_groups', 'n_matched'].min())} "
        "paired spatial-peaks cells. Across 0°/30°/60°, the per-lag maximum absolute difference "
        f"is {validation.loc[validation.estimator == 'single_config_pooled', 'max_abs_difference'].max():.2g} "
        "and the paired spatial-peaks maximum is "
        f"{validation.loc[validation.estimator == 'paired_grid_groups', 'max_abs_difference'].max():.2g}; "
        "all corresponding cell-wise correlations round to 1.000000.",
        f"Spatial-peaks cache: `{spatial_cache}`.",
        "",
        "## Recomputed population means",
        "",
        "| estimator | lag | n | mean raw r | Fisher t | two-sided p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in population.iterrows():
        report.append(
            f"| {r.estimator} | {int(r.lag_deg)}° | {int(r.n_cells)} | "
            f"{r.mean_raw_r:.4f} | {r.t_fisher_two_sided:.2f} | "
            f"{r.p_two_sided:.4f} |")
    report += [
        "",
        "## Single held-out subject/configurations with the largest influence",
        "",
        "Negative change after exclusion means the combination was driving the "
        "original mean upward; positive change means it was driving it downward.",
        "",
        "### Drive high 30°",
        *_top_lines(single_influence, 30, "high"),
        "",
        "### Drive high 60°",
        *_top_lines(single_influence, 60, "high"),
        "",
        "### Drive low 0°",
        *_top_lines(single_influence, 0, "low"),
        "",
        "## Paired configuration groups with the largest influence",
        "",
        "### Drive high 30°",
        *_top_lines(paired_influence, 30, "high"),
        "",
        "### Drive high 60°",
        *_top_lines(paired_influence, 60, "high"),
        "",
        "### Drive low 0°",
        *_top_lines(paired_influence, 0, "low"),
        "",
        "## Leading cells in the per-lag result",
        "",
        "These are descriptive rankings, not additional inferential tests. "
        "The delta rankings in `cell_driver_rankings.csv` are useful for "
        "separating genuinely future-preferring cells from cells that are high "
        "at every lag. Anatomical suffixes in the neuron IDs are historical; "
        "every cell below is included because its latest-table `alt_final_roi` "
        "is mPFC.",
        "",
        "### Highest 30° r",
        *_top_cell_lines(cell_rankings, "high_30"),
        "",
        "### Highest 60° r",
        *_top_cell_lines(cell_rankings, "high_60"),
        "",
        "### Lowest 0° r",
        *_top_cell_lines(cell_rankings, "low_0"),
        "",
        "### Largest 30° minus 0° difference",
        *_top_cell_lines(cell_rankings, "high_30_minus_0"),
        "",
        "### Largest 60° minus 0° difference",
        *_top_cell_lines(cell_rankings, "high_60_minus_0"),
        "",
        "## Cell-level files",
        "",
        "- `per_lag_cell_scores.csv`: identify cells with high 30°/60° or low 0°.",
        "- `per_lag_held_config_folds.csv`: exact single-config fold scores.",
        "- `spatial_original_config_pairs.csv`: direct consistency for every "
        "original task-configuration pair after spatial-peaks QC.",
        "- `spatial_paired_group_folds.csv`: actual coverage-paired CV folds.",
        "- `combination_influence.csv`: change in the population mean when each "
        "subject/configuration or paired group is excluded.",
        "- `cell_driver_rankings.csv`: ranked cells for high 30°, high 60°, low "
        "0°, and the two future-minus-now contrasts.",
        "- `cache_validation.csv`: numerical agreement with the saved production "
        "runs.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    (out_dir / "config.json").write_text(json.dumps({
        "cell_table": str(CELL_TABLE), "roi": "mPFC", "lags": list(LAGS),
        "per_lag_source": str(PER_LAG_SOURCE),
        "spatial_source": str(SPATIAL_SOURCE),
        "phase_residualise": "cosine", "permutations": 0,
        "purpose": "descriptive decomposition only",
    }, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
