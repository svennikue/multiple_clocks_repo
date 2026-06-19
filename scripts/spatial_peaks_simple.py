#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2026-06-17

@author: Svenja Kuchenhoff

Future spatial peaks (clean re-implementation).

Per cell, identify cross-validated future-location encoding via shift-tuned
rate-map consistency across grids. Phase-agnostic by construction (rates
averaged over all bins where the shifted location equals a grid square).

Designed to be directly comparable with:
  * scripts/RSA_DSR_ROIs_simple.py
  * scripts/encoding_analysis_simple.py

Driven by a JSON config (see scripts/configs/spatial_peaks_default.json).
First-pass mode (no permutations).
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")

import mc
import mc.analyse.cell_selection as cs
import mc.analyse.future_spatial_peaks as fsp
import mc.plotting.spatial_peaks as sp


DATA_DIR = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
OUT_BASE = os.path.join(
    DATA_DIR, "group", "spatial_peaks_simple",
)
DEFAULT_CONFIG = (
    "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/"
    "scripts/configs/spatial_peaks_default.json"
)


def load_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    # validation
    assert cfg["cell_set"] in ("rsa", "all_in_roi_table")
    assert cfg["coverage_mode"] in ("per_grid", "paired")
    assert cfg["sparsity_filter"] in (None, "gridwise_qc")
    assert cfg["n_peaks"] >= 1
    assert cfg["min_dwell_bins"] >= 1
    assert cfg["min_shared_locs"] >= 2
    pr = cfg.get("phase_residualise")
    assert pr in (None, "cosine", "cosine_2h", "categorical"), \
        f"phase_residualise must be one of None / cosine / cosine_2h / categorical, got {pr!r}"
    fl = cfg.get("fixed_lags")
    if fl is not None:
        assert isinstance(fl, list) and all(isinstance(x, int) for x in fl), \
            "fixed_lags must be a list of int bin offsets"
    gr = cfg.get("goal_relative_locs")
    assert gr in (None, False, "toroidal_mod3", "manhattan_distance"), \
        f"goal_relative_locs must be None / 'toroidal_mod3' / 'manhattan_distance', got {gr!r}"
    assert cfg.get("config_residualise") in (None, False, True), \
        "config_residualise must be bool"
    assert cfg.get("time_residualise") in (None, False, True), \
        "time_residualise must be bool"
    assert cfg.get("run_residualise") in (None, False, True), \
        "run_residualise must be bool"
    return cfg


def build_payloads_for_subject(sub_str, cfg, neuron_ids_keep=None):
    """Load one subject and build per-cell payloads.
    Returns list of (payload, neuron_id) and the cell-registry rows.
    """
    data_raw = mc.analyse.helpers_human_cells.load_norm_data(
        DATA_DIR, [sub_str],
        res_data=(cfg["trials"] == "residualised"),
    )
    if not data_raw:
        return []
    data = mc.analyse.helpers_human_cells.filter_data(
        data_raw, int(sub_str), cfg["trials"],
    )
    sub_dict = data[f"sub-{sub_str}"]
    norm_neurons = sub_dict["normalised_neurons"]
    payloads = []
    for nid in norm_neurons:
        if neuron_ids_keep is not None and nid not in neuron_ids_keep:
            continue
        try:
            p = fsp.build_cell_payload(
                sub_data=sub_dict,
                neuron_id=nid,
                neurons_df=norm_neurons[nid],
                locs_df=sub_dict["locations"],
                beh=sub_dict["beh"],
                trials_label=cfg["trials"],
                coverage_mode=cfg["coverage_mode"],
                sparsity_filter=cfg["sparsity_filter"],
                phase_residualise=cfg.get("phase_residualise"),
                goal_relative_locs=cfg.get("goal_relative_locs"),
                config_residualise=bool(cfg.get("config_residualise")),
                time_residualise=bool(cfg.get("time_residualise")),
                run_residualise=bool(cfg.get("run_residualise")),
            )
        except Exception as exc:
            print(f"  [{nid}] payload build failed: {exc!r}")
            p = None
        payloads.append((nid, p))
    return payloads


def compute_subject(sub_str, cfg, cells_df):
    """Process all kept cells for one subject, returning a list of result rows."""
    sub_int = int(sub_str)
    # restrict to neuron_ids that map to RSA-kept cells (by cell_idx)
    kept_cell_idx = set(cells_df.loc[cells_df["subject_int"] == sub_int, "cell_idx"].tolist())
    # we only know neuron_ids after loading the subject; pass cell-idx filter as None
    # and prune by mapping label -> cell_idx after construction.

    payloads = build_payloads_for_subject(sub_str, cfg, neuron_ids_keep=None)

    rows = []
    for nid, payload in payloads:
        _, cell_idx = cs.parse_neuron_label(nid)
        if cell_idx is None or cell_idx not in kept_cell_idx:
            continue
        roi_row = cells_df[(cells_df["subject_int"] == sub_int) &
                           (cells_df["cell_idx"] == cell_idx)].iloc[0]
        if payload is None:
            rows.append({
                "neuron_id":    nid,
                "subject_id":   sub_str,
                "subject_int":  sub_int,
                "cell_idx":     cell_idx,
                "roi":          roi_row["roi"],
                "MNI_x":        roi_row["MNI_x"],
                "MNI_y":        roi_row["MNI_y"],
                "MNI_z":        roi_row["MNI_z"],
                "peak_r":               np.nan,
                "peak_shift_plurality": np.nan,
                "n_grids_used":         0,
                "mean_shared_locs_at_peak": np.nan,
                "fold_shifts_json":     "",
                "fold_rs_json":         "",
                "shift_curve_full_json": "",
                "note":                 "payload_none",
            })
            continue
        try:
            out = fsp.run_one_cell(payload, cfg)
        except Exception as exc:
            print(f"  [{nid}] compute failed: {exc!r}")
            out = None
        if out is None:
            note = "compute_failed"
            row_out = {"peak_r": np.nan, "peak_shift_plurality": np.nan,
                       "n_grids_used": 0, "mean_shared_locs_at_peak": np.nan,
                       "fold_shifts": [], "fold_rs": [], "shift_curve_full": []}
        else:
            note = "ok"
            row_out = out
        perm_rs = row_out.get("perm_peak_rs", []) or []
        fl_mean = row_out.get("fixed_lag_r_mean", np.nan)
        fl_per  = row_out.get("fixed_lag_per_lag_r", [])
        fl_lags = row_out.get("fixed_lag_lags", [])
        fl_perm = row_out.get("fixed_lag_perm_means", []) or []
        rows.append({
            "neuron_id":    nid,
            "subject_id":   sub_str,
            "subject_int":  sub_int,
            "cell_idx":     cell_idx,
            "roi":          roi_row["roi"],
            "MNI_x":        roi_row["MNI_x"],
            "MNI_y":        roi_row["MNI_y"],
            "MNI_z":        roi_row["MNI_z"],
            "peak_r":                  row_out["peak_r"],
            "peak_shift_plurality":    row_out["peak_shift_plurality"],
            "n_grids_used":            row_out["n_grids_used"],
            "mean_shared_locs_at_peak": row_out["mean_shared_locs_at_peak"],
            "fold_shifts_json":        json.dumps(row_out.get("fold_shifts", [])),
            "fold_rs_json":            json.dumps(_finite_nested(row_out.get("fold_rs", []))),
            "shift_curve_full_json":   json.dumps(_finite_list(row_out.get("shift_curve_full", []))),
            "perm_peak_rs_json":       json.dumps(_finite_list(perm_rs)),
            "n_permutations_done":     int(len(perm_rs)),
            "fixed_lag_r_mean":        float(fl_mean) if (fl_mean is not None and np.isfinite(fl_mean)) else np.nan,
            "fixed_lag_lags_json":     json.dumps(list(fl_lags)) if fl_lags else "",
            "fixed_lag_per_lag_r_json": json.dumps(_finite_list(fl_per)) if fl_per else "",
            "fixed_lag_perm_means_json": json.dumps(_finite_list(fl_perm)) if fl_perm else "",
            "note":                    note,
        })
    return rows


def _finite_list(xs):
    return [float(x) if (x is not None and np.isfinite(x)) else None for x in xs]


def _finite_nested(xss):
    return [_finite_list(xs) for xs in xss]


def run(cfg_path):
    cfg = load_config(cfg_path)
    np.random.seed(cfg["random_seed"])

    run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + cfg["analysis_name"]
    out_dir = Path(OUT_BASE) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # snapshot the config
    shutil.copy(cfg_path, out_dir / "config.json")

    # cell registry
    cells_df = cs.load_cells(
        cell_set=cfg["cell_set"],
        subjects=cfg["subjects"],
        rois_keep=cfg["rois_keep"],
    )
    cells_df.to_csv(out_dir / "cells_used.csv", index=False)
    subjects = sorted(cells_df["subject_id"].unique())
    print(f"[{run_tag}]")
    print(f"  cell_set={cfg['cell_set']}  → {len(cells_df)} cells across {len(subjects)} subjects")
    print(f"  out_dir = {out_dir}")

    # dispatch
    all_rows = []
    t_total = time.time()
    if cfg["n_jobs"] in (None, 0, 1):
        for sub_str in subjects:
            t0 = time.time()
            rows = compute_subject(sub_str, cfg, cells_df)
            all_rows.extend(rows)
            print(f"  sub-{sub_str}: {len(rows)} cells in {time.time() - t0:.1f}s")
    else:
        from joblib import Parallel, delayed
        print(f"  joblib n_jobs={cfg['n_jobs']}")
        results = Parallel(n_jobs=cfg["n_jobs"], backend="loky", verbose=5)(
            delayed(compute_subject)(sub_str, cfg, cells_df)
            for sub_str in subjects
        )
        for rows in results:
            all_rows.extend(rows)
    print(f"  total compute time: {time.time() - t_total:.1f}s")

    per_cell = pd.DataFrame(all_rows)

    # ── per-cell p_perm + per-ROI BH-FDR ─────────────────────────────────
    if cfg.get("run_permutations", False) and cfg.get("n_permutations", 0) > 0:
        per_cell = _add_p_perm(per_cell)
        per_cell = _add_q_bh_per_roi(per_cell)
        print(f"  added p_perm + q_BH_within_roi")
        if cfg.get("fixed_lags"):
            per_cell = _add_fixed_lag_p_perm(per_cell)
            print(f"  added fixed_lag_p_perm + fixed_lag_q_BH_within_roi")

    per_cell.to_csv(out_dir / "per_cell.csv", index=False)
    print(f"  saved per_cell.csv ({len(per_cell)} rows)")

    # ── subset binomial-per-ROI analysis ────────────────────────────────
    if cfg.get("run_permutations", False):
        try:
            subset_df = _run_subset_binomials(per_cell, cfg, out_dir)
            print(f"  ran subset binomials → {len(subset_df)} ROI×subset rows")
        except Exception as exc:
            print(f"  subset binomial failed: {exc!r}")

    # plots
    try:
        sp.plot_all(per_cell, out_dir, cfg)
    except Exception as exc:
        print(f"  plotting failed: {exc!r}")

    # changelog
    _append_changelog(out_dir, cfg, per_cell, run_tag)
    return per_cell, out_dir


def _add_p_perm(df):
    """Per-cell one-sided p = (k+1)/(M+1) with k = #perms >= observed."""
    p_perm = np.full(len(df), np.nan, dtype=float)
    for i, row in df.iterrows():
        try:
            perms = json.loads(row.get("perm_peak_rs_json", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            perms = []
        perms = [p for p in perms if p is not None and np.isfinite(p)]
        obs = row["peak_r"]
        if not perms or not np.isfinite(obs):
            continue
        k = sum(1 for p in perms if p >= obs)
        p_perm[i] = (k + 1) / (len(perms) + 1)
    df = df.copy()
    df["p_perm"] = p_perm
    return df


def _bh_fdr(pvals):
    p = np.asarray(pvals, float)
    m = int(np.sum(~np.isnan(p)))
    adj = np.full_like(p, np.nan, dtype=float)
    if m == 0:
        return adj
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    ranks = np.arange(1, len(p) + 1, dtype=float)
    p_ord = p[order]
    with np.errstate(invalid="ignore"):
        running = np.minimum.accumulate((m / ranks[order]) * p_ord[::-1])[::-1]
    adj[order] = np.clip(running, 0, 1)
    return adj


def _add_q_bh_per_roi(df, alpha=0.05):
    """BH-FDR within each ROI's cell pool."""
    df = df.copy()
    df["q_BH_within_roi"] = np.nan
    for roi, grp in df.groupby("roi"):
        q = _bh_fdr(grp["p_perm"].to_numpy())
        df.loc[grp.index, "q_BH_within_roi"] = q
    df["sig_FDR_within_roi"] = df["q_BH_within_roi"] < alpha
    return df


def _add_fixed_lag_p_perm(df, alpha=0.05):
    """Per-cell p_perm + BH-FDR for the fixed-lag metric."""
    df = df.copy()
    p_perm = np.full(len(df), np.nan, dtype=float)
    for i, row in df.iterrows():
        try:
            perms = json.loads(row.get("fixed_lag_perm_means_json", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            perms = []
        perms = [p for p in perms if p is not None and np.isfinite(p)]
        obs = row.get("fixed_lag_r_mean", np.nan)
        if not perms or not np.isfinite(obs):
            continue
        k = sum(1 for p in perms if p >= obs)
        p_perm[i] = (k + 1) / (len(perms) + 1)
    df["fixed_lag_p_perm"] = p_perm
    df["fixed_lag_q_BH_within_roi"] = np.nan
    for roi, grp in df.groupby("roi"):
        q = _bh_fdr(grp["fixed_lag_p_perm"].to_numpy())
        df.loc[grp.index, "fixed_lag_q_BH_within_roi"] = q
    df["fixed_lag_sig_FDR_within_roi"] = df["fixed_lag_q_BH_within_roi"] < alpha
    return df


def _load_encoding_significance(enc_csv, model_name, alpha=0.05):
    """Return DataFrame [neuron_id, <model>_p, <model>_sig]."""
    enc = pd.read_csv(enc_csv)
    sub = enc[enc["model"] == model_name][["neuron", "p_perm"]].copy()
    sub.columns = ["neuron_id", f"{model_name}_p_perm"]
    sub[f"{model_name}_sig"] = sub[f"{model_name}_p_perm"] < alpha
    return sub


def _run_subset_binomials(per_cell, cfg, out_dir, alpha=0.05):
    """Per-ROI binomial test of fraction(p_perm<alpha) on multiple
    pre-specified subsets. BH-FDR across the (subset×ROI) family.
    """
    from scipy.stats import binomtest
    enc_csv = cfg.get("encoding_results_csv",
        "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/"
        "derivatives/group/encoding_analysis_simple/2026-06-05_17-58-57/"
        "encoding_results.csv")
    df = per_cell.copy()
    if Path(enc_csv).is_file():
        for m in ("state", "dsr"):
            try:
                aux = _load_encoding_significance(enc_csv, m, alpha=alpha)
                df = df.merge(aux, on="neuron_id", how="left")
            except Exception as exc:
                print(f"  [subset] couldn't load encoding '{m}': {exc!r}")
    subsets = {
        "all":       np.ones(len(df), dtype=bool),
        "state_NS":  ~df.get("state_sig", pd.Series([False] * len(df))).fillna(False).to_numpy(),
        "DSR_sig":    df.get("dsr_sig",   pd.Series([False] * len(df))).fillna(False).to_numpy(),
    }
    rows = []
    for label, mask in subsets.items():
        sub = df[mask & df["p_perm"].notna()]
        for roi, g in sub.groupby("roi"):
            n = len(g); k = int((g["p_perm"] < alpha).sum())
            if n == 0:
                continue
            p_binom = binomtest(k, n, p=alpha, alternative="greater").pvalue
            rows.append({"subset": label, "roi": roi, "n": n, "k": k,
                         "frac": k / n, "p_binom": p_binom})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_BH_family"] = _bh_fdr(out["p_binom"].to_numpy())
        out["sig_FDR"] = out["q_BH_family"] < alpha
    out.to_csv(Path(out_dir) / "subset_binomial.csv", index=False)
    return out


def _append_changelog(out_dir, cfg, df, run_tag):
    changelog_path = Path(DATA_DIR) / "group" / "spatial_peaks_simple" / "CHANGELOG.md"
    changelog_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = int((df["note"] == "ok").sum())
    n_total = len(df)
    by_roi = df[df["note"] == "ok"].groupby("roi")["peak_r"].agg(["count", "mean"])
    summary_lines = [f"  - {r}: n={int(c)}, mean peak_r={m:+.3f}"
                     for r, (c, m) in by_roi.iterrows()]
    entry = "\n".join([
        f"## {run_tag}",
        f"- analysis_name: {cfg['analysis_name']}",
        f"- cell_set: {cfg['cell_set']}, trials: {cfg['trials']}, "
        f"coverage_mode: {cfg['coverage_mode']}, n_peaks: {cfg['n_peaks']}",
        f"- n_cells: {n_ok} ok / {n_total} attempted",
        "- per-ROI peak_r:",
        *summary_lines,
        "",
    ])
    if changelog_path.exists():
        prev = changelog_path.read_text()
    else:
        prev = "# spatial_peaks_simple changelog\n\n"
    changelog_path.write_text(prev + entry + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default=DEFAULT_CONFIG,
                    help="Path to JSON config (default: scripts/configs/spatial_peaks_default.json)")
    args = ap.parse_args()
    run(args.config)
