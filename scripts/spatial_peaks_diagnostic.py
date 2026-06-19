#!/usr/bin/env python3
"""Diagnostic overview for spatial_peaks runs.

Loads one or more per_cell.csv outputs and builds:
    * a per-cell diagnostic CSV (peakiness, fold-agreement, near-max count)
    * a multi-panel overview figure

Optionally joins state-significance from the encoding pipeline.

Usage (script-style, edit the paths in __main__ below):
    python spatial_peaks_diagnostic.py
"""

import json
import os
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


# ── parse JSON-encoded columns ─────────────────────────────────────────

def parse_curve(s):
    if not isinstance(s, str) or not s:
        return None
    return [v if v is not None else np.nan for v in json.loads(s)]


def parse_folds(s):
    if not isinstance(s, str) or not s:
        return None
    return json.loads(s)


def add_diagnostic_cols(df):
    df = df.copy()
    df["curve"]      = df["shift_curve_full_json"].apply(parse_curve)
    df["fold_shifts"] = df["fold_shifts_json"].apply(parse_folds)

    def curve_stat(c, fn):
        if c is None:
            return np.nan
        arr = np.array(c, dtype=float)
        finite = arr[np.isfinite(arr)]
        return fn(finite) if finite.size else np.nan

    df["curve_max"]   = df["curve"].apply(lambda c: curve_stat(c, np.max))
    df["curve_min"]   = df["curve"].apply(lambda c: curve_stat(c, np.min))
    df["curve_range"] = df["curve_max"] - df["curve_min"]
    df["curve_mean"]  = df["curve"].apply(lambda c: curve_stat(c, np.mean))
    df["curve_std"]   = df["curve"].apply(lambda c: curve_stat(c, np.std))
    df["peakiness"]   = (df["curve_max"] - df["curve_mean"]) / (df["curve_std"] + 1e-9)

    def n_near(c, frac=0.8):
        if c is None:
            return np.nan
        arr = np.array(c, dtype=float)
        finite = arr[np.isfinite(arr)]
        if not finite.size:
            return np.nan
        thr = finite.max() * frac if finite.max() > 0 else -np.inf
        return int((arr >= thr).sum())

    df["n_near_max_80pct"] = df["curve"].apply(n_near)

    def fold_agree(fs):
        if not fs:
            return np.nan, np.nan
        flat = [v for sub in fs for v in sub]
        if not flat:
            return np.nan, np.nan
        n_unique = len(set(flat))
        frac_top = Counter(flat).most_common(1)[0][1] / len(flat)
        return n_unique, frac_top

    fa = df["fold_shifts"].apply(fold_agree)
    df["n_unique_fold_shifts"] = [t[0] if isinstance(t, tuple) else np.nan for t in fa]
    df["frac_top_fold_shift"]  = [t[1] if isinstance(t, tuple) else np.nan for t in fa]
    return df


def load_state_significance(path, alpha=0.05):
    """Returns DataFrame keyed by neuron with state_sig (bool)."""
    enc = pd.read_csv(path)
    state = enc[enc["model"] == "state"][["neuron", "p_perm", "mean_r"]].copy()
    state.rename(columns={"neuron": "neuron_id",
                          "p_perm": "state_p_perm",
                          "mean_r": "state_mean_r"}, inplace=True)
    state["state_sig"] = state["state_p_perm"] < alpha
    return state


# ── plotting ───────────────────────────────────────────────────────────

ROI_ORDER = ["ACC", "medialOFC", "PCC", "Parahippocampal",
             "HC_anterior", "HC_mid", "EC"]


def _stratified_box(ax, df, group_col, value_col, group_order, title,
                    ylabel="peak_r"):
    data = []; labels = []
    for g in group_order:
        v = df.loc[df[group_col] == g, value_col].dropna().to_numpy()
        if v.size:
            data.append(v); labels.append(f"{g}\n(n={v.size})")
    if not data:
        ax.set_title(title + " (no data)"); return
    bp = ax.boxplot(data, labels=labels, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="red",
                                   markeredgecolor="red", markersize=4),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=8)


def _curve_per_roi(ax, df, title):
    rois = [r for r in ROI_ORDER if r in df["roi"].unique()]
    cmap = plt.get_cmap("tab10")
    shifts = list(range(0, 360, 30))
    for i, roi in enumerate(rois):
        sub = df[df["roi"] == roi]
        curves = np.array([c for c in sub["curve"] if c is not None and len(c) == 12])
        if not curves.size:
            continue
        mean = np.nanmean(curves, axis=0)
        sem  = np.nanstd(curves, axis=0) / np.sqrt(np.sum(np.isfinite(curves), axis=0))
        ax.plot(shifts, mean, label=f"{roi} (n={len(curves)})",
                color=cmap(i), lw=1.5)
        ax.fill_between(shifts, mean - sem, mean + sem,
                        color=cmap(i), alpha=0.15)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticks(shifts)
    ax.set_xticklabels(shifts, fontsize=7)
    ax.set_xlabel("Shift (°)")
    ax.set_ylabel("Mean train consistency")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best", ncol=2, frameon=False)


def _ttest_str(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return "(no data)"
    t, p = st.ttest_1samp(x, 0)
    sign = "+" if t > 0 else "-"
    return f"t={sign}{abs(t):.2f}, p={p:.3g}, n={x.size}"


def _split_by_state(df_state):
    rows = []
    for r in ROI_ORDER:
        for sig, label in [(True, "state-sig"), (False, "state-NS")]:
            sub = df_state[(df_state["roi"] == r) & (df_state["state_sig"] == sig)]
            if not len(sub):
                continue
            x = sub["peak_r"].dropna().to_numpy()
            if not x.size:
                continue
            rows.append({"roi": r, "group": label, "n": int(x.size),
                         "mean": float(x.mean()), "std": float(x.std()),
                         "t_p": _ttest_str(x)})
    return pd.DataFrame(rows)


# ── main diagnostic ────────────────────────────────────────────────────

def run_diagnostic(runs, encoding_results_csv, out_dir):
    """runs is dict label -> path to run dir (containing per_cell.csv)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    dfs = {}
    for label, rd in runs.items():
        df = pd.read_csv(Path(rd) / "per_cell.csv")
        df = df[df["note"] == "ok"].copy()
        df = add_diagnostic_cols(df)
        dfs[label] = df
        print(f"  loaded {label}: n={len(df)} ok cells")

    base = dfs["baseline"]

    # join state info
    state = load_state_significance(encoding_results_csv)
    base_with_state = base.merge(state, on="neuron_id", how="left")
    print(f"  state info joined: {base_with_state['state_sig'].notna().sum()} / {len(base_with_state)} matched")

    # save per-cell diagnostic table
    keep_cols = ["neuron_id", "subject_id", "roi", "peak_r",
                 "peak_shift_plurality", "n_grids_used",
                 "mean_shared_locs_at_peak",
                 "curve_max", "curve_min", "curve_range",
                 "curve_mean", "curve_std", "peakiness",
                 "n_near_max_80pct",
                 "n_unique_fold_shifts", "frac_top_fold_shift",
                 "state_p_perm", "state_mean_r", "state_sig"]
    diag = base_with_state[keep_cols].copy()
    diag.to_csv(out_dir / "per_cell_diagnostic.csv", index=False)
    print(f"  saved per_cell_diagnostic.csv")

    # ── overview figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 22), constrained_layout=False)
    gs = fig.add_gridspec(6, 3, hspace=0.55, wspace=0.3)

    # row 0: peak_r per ROI (boxplot) for K=1
    ax = fig.add_subplot(gs[0, :])
    _stratified_box(ax, base, "roi", "peak_r",
                    [r for r in ROI_ORDER if r in base["roi"].unique()],
                    "Baseline (K=1, residualised): peak_r by ROI")

    # row 1 col 0+1: K sweep
    base["K"] = "K=1"
    k2 = dfs["K2"].copy(); k2["K"] = "K=2"
    k3 = dfs["K3"].copy(); k3["K"] = "K=3"
    pooled = pd.concat([base, k2, k3])
    ax = fig.add_subplot(gs[1, 0])
    _stratified_box(ax, pooled, "K", "peak_r", ["K=1", "K=2", "K=3"],
                    "n_peaks sweep (all ROIs pooled)")

    # row 1 col 1: K sweep per ROI
    ax = fig.add_subplot(gs[1, 1:])
    rois = [r for r in ROI_ORDER if r in base["roi"].unique()]
    x = np.arange(len(rois)); width = 0.27
    for i, (label, dfi) in enumerate([("K=1", base), ("K=2", k2), ("K=3", k3)]):
        means = [dfi.loc[dfi["roi"] == r, "peak_r"].mean() for r in rois]
        sems  = [dfi.loc[dfi["roi"] == r, "peak_r"].sem()  for r in rois]
        ax.bar(x + (i - 1) * width, means, width, yerr=sems, label=label, capsize=2)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(rois, rotation=20, fontsize=8)
    ax.set_ylabel("mean peak_r (± SEM)"); ax.set_title("n_peaks sweep per ROI", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # row 2: trials sweep
    base["trials"] = "residualised"
    t_all = dfs["trials_all"].copy(); t_all["trials"] = "all_minus_explore"
    pooled_t = pd.concat([base, t_all])
    ax = fig.add_subplot(gs[2, 0])
    _stratified_box(ax, pooled_t, "trials", "peak_r",
                    ["residualised", "all_minus_explore"],
                    "trials sweep (all ROIs pooled)")
    ax = fig.add_subplot(gs[2, 1:])
    for i, (label, dfi) in enumerate([("residualised", base),
                                      ("all_minus_explore", t_all)]):
        means = [dfi.loc[dfi["roi"] == r, "peak_r"].mean() for r in rois]
        sems  = [dfi.loc[dfi["roi"] == r, "peak_r"].sem()  for r in rois]
        ax.bar(x + (i - 0.5) * width * 1.4, means, width * 1.4,
               yerr=sems, label=label, capsize=2)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(rois, rotation=20, fontsize=8)
    ax.set_ylabel("mean peak_r (± SEM)"); ax.set_title("trials sweep per ROI", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # row 3 col 0: peak_r vs fold agreement (binned)
    base["agree_bin"] = pd.cut(base["frac_top_fold_shift"],
                               bins=[0, 0.4, 0.6, 0.8, 1.01],
                               labels=["<0.4", "0.4-0.6", "0.6-0.8", ">0.8"])
    ax = fig.add_subplot(gs[3, 0])
    _stratified_box(ax, base, "agree_bin", "peak_r",
                    ["<0.4", "0.4-0.6", "0.6-0.8", ">0.8"],
                    "peak_r vs fold-agreement (top-shift vote fraction)")

    # row 3 col 1: peak_r vs peakiness
    base["peaky_bin"] = pd.cut(base["peakiness"],
                               bins=[0, 1.5, 2, 2.5, 5],
                               labels=["<1.5", "1.5-2", "2-2.5", ">2.5"])
    ax = fig.add_subplot(gs[3, 1])
    _stratified_box(ax, base, "peaky_bin", "peak_r",
                    ["<1.5", "1.5-2", "2-2.5", ">2.5"],
                    "peak_r vs peakiness ratio (sharper → real?)")

    # row 3 col 2: peak_r vs n_near_max
    ax = fig.add_subplot(gs[3, 2])
    base["nnm_bin"] = base["n_near_max_80pct"].clip(upper=4)
    _stratified_box(ax, base, "nnm_bin", "peak_r",
                    [1, 2, 3, 4],
                    "peak_r vs # near-max shifts (1=single peak)")

    # row 4: shift-curve average per ROI
    ax = fig.add_subplot(gs[4, :2])
    _curve_per_roi(ax, base, "Mean training shift-curve per ROI (baseline K=1)")
    ax = fig.add_subplot(gs[4, 2])
    # peak_shift histogram per ROI (heatmap)
    pivot = base.groupby(["roi", "peak_shift_plurality"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(index=[r for r in ROI_ORDER if r in pivot.index])
    pivot = pivot.reindex(columns=list(range(0, 360, 30)), fill_value=0)
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(12)); ax.set_xticklabels(list(range(0, 360, 30)), fontsize=7)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("peak shift (°)"); ax.set_title("Cell count by ROI × peak shift", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.04)

    # row 5: state cells split
    bws = base_with_state.dropna(subset=["state_sig"])
    bws["state_label"] = bws["state_sig"].map({True: "state-sig", False: "state-NS"})
    ax = fig.add_subplot(gs[5, 0])
    _stratified_box(ax, bws, "state_label", "peak_r",
                    ["state-sig", "state-NS"],
                    "peak_r: state-sig vs state-NS (pooled)")
    ax = fig.add_subplot(gs[5, 1:])
    rois_state = [r for r in ROI_ORDER if r in bws["roi"].unique()]
    x = np.arange(len(rois_state))
    for i, lbl in enumerate(["state-sig", "state-NS"]):
        m = [bws.loc[(bws["roi"] == r) & (bws["state_label"] == lbl),
                     "peak_r"].mean() for r in rois_state]
        s = [bws.loc[(bws["roi"] == r) & (bws["state_label"] == lbl),
                     "peak_r"].sem()  for r in rois_state]
        ax.bar(x + (i - 0.5) * 0.35, m, 0.35, yerr=s, label=lbl, capsize=2)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(rois_state, rotation=20, fontsize=8)
    ax.set_ylabel("mean peak_r (± SEM)")
    ax.set_title("peak_r per ROI split by state significance (encoding p_perm<0.05)",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("spatial_peaks diagnostic overview", fontsize=14, fontweight="bold",
                 y=0.995)
    fig_path = out_dir / "diagnostic_overview.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"  saved diagnostic_overview.png  ->  {fig_path}")

    # ── summary tables to stdout ────────────────────────────────────────
    print("\n=== n_peaks sweep (per ROI mean peak_r) ===")
    sweep = pd.concat([base.assign(K="K=1"), k2.assign(K="K=2"), k3.assign(K="K=3")])
    print(sweep.groupby(["roi", "K"])["peak_r"].mean().unstack().round(3))

    print("\n=== trials sweep (per ROI mean peak_r) ===")
    print(pooled_t.groupby(["roi", "trials"])["peak_r"].mean().unstack().round(3))

    print("\n=== state-cell split (per ROI mean peak_r) ===")
    print(_split_by_state(bws).to_string(index=False))

    print("\n=== fold-agreement split (pooled) ===")
    print(base.groupby("agree_bin")["peak_r"].agg(["count", "mean", "std"]).round(3))

    print("\n=== peakiness split (pooled) ===")
    print(base.groupby("peaky_bin")["peak_r"].agg(["count", "mean", "std"]).round(3))

    return diag


if __name__ == "__main__":
    base_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/spatial_peaks_simple"
    runs = {
        "baseline":   f"{base_dir}/2026-06-17_16-08-31_all_K1",
        "K2":         f"{base_dir}/2026-06-17_16-09-01_all_K2",
        "K3":         f"{base_dir}/2026-06-17_16-09-32_all_K3",
        "trials_all": f"{base_dir}/2026-06-17_16-10-01_all_K1_trials_all",
    }
    enc_csv = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/encoding_analysis_simple/2026-06-05_17-58-57/encoding_results.csv"
    out = f"{base_dir}/_diagnostic_2026-06-17"
    run_diagnostic(runs, enc_csv, out)
