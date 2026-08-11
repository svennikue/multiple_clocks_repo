#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Future spatial peaks — runner.

This is the SURFACE script: all settings live here, the heavy compute
lives in `mc.analyse.future_spatial_peaks`. Edit the SETTINGS block and
re-run.

Pipeline per cell:
  1. Phase-residualise the raw per-bin firing rate (cosine basis, BEFORE
     any rate maps are computed).
  2. Pair grids to maximise spatial coverage (each pair = one "grid
     group").
  3. Compute cross-validated spatial consistency at every lag in
     `LAGS_DEG`. Two complementary CV variants per cell:
       * fixed_lag_r — at PRE-SPECIFIED lag(s) per ROI (a-priori).
                       mPFC → 30°/60°. HC_anterior/mid → 0°/330°.
                       For all other ROIs we don't fix a lag (NaN).
       * free_peak_r — cell picks its own peak on training grids,
                       validated on held-out grid. ORIGINAL control.
  4. Permutation null via per-rep circular shifts of the location series.

RELATION TO `per_lag_encoding.py`
  Shared: phase residualisation, twelve lagged 9-location rate maps,
  leave-one-group-out validation, dwell-weighted Pearson correlations, and
  predicted-lag vs-zero / predicted-vs-other-lag population tests.

  Unique here (robustness analysis): configurations are paired into
  coverage-maximising grid groups; the held-out group is correlated with
  each training group separately and those r values are averaged; at least
  five shared locations are required; and each repetition receives an
  independent circular shift in the permutation null. This analysis also
  includes a training-selected free-peak control and rate-map examples.

  By contrast, the manuscript-primary no-control per-lag analysis leaves out
  single configurations, pools all training configurations into one
  dwell-weighted predicted map, requires three shared locations, and shifts
  the held-out configuration series once per fold/permutation. Thus paired
  configurations are an important difference, but not the only difference.

Per-ROI statistical tests:
  (a) one-sample t-test of fixed-lag r > 0.
  (b) within-cell paired t-test: r at predicted lag vs r at other lags.
  (c) binomial test: fraction of cells with perm-p < α exceeds α chance.
  All three p-values are BH-FDR-corrected across the reported ROIs.

Plots written under <OUT_DIR>:
  * per_roi_stats.csv          — (a)/(b)/(c) results per ROI
  * roi_x_lag_overview.{pdf,png} — mean r-vs-0 t-stat per (ROI, lag),
                                    with stars for FDR sig.
  * fixed_vs_free_comparison.{pdf,png} — for mPFC, HC_anterior, HC_mid:
                                          fixed-lag vs free-peak r.
  * rate_map_examples/<ROI>/   — one PDF per "good" cell (max spatial
                                  coverage), showing the per-CV-split
                                  rate maps (one panel per grid group).
  * paired_grid_configs/<ROI>/ — small visualisations of which configs
                                  the paired-coverage mode collapsed
                                  together, for the example cells.

@author: Svenja Küchenhoff (refactor: Claude)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import stats

sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")

import mc
import mc.analyse.cell_selection as cs
import mc.analyse.future_spatial_peaks as fsp


# ====================================================================
# SETTINGS — change here to re-run
# ====================================================================
# Run identity --------------------------------------------------------
ANALYSIS_NAME   = "phase_resid_paired_fixedlag"
DATA_DIR        = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
OUT_BASE        = os.path.join(DATA_DIR, "group", "spatial_peaks_simple")

# Reload mode ---------------------------------------------------------
# Set to a previous run directory (or its per_cell.csv) to skip the
# heavy CV + permutation compute and just re-run stats + plots from the
# cached per-cell results. Outputs are written to a fresh `<run>_reload`
# directory so the original is preserved.
RELOAD_FROM     = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/spatial_peaks_simple/2026-06-26_18-47-11_phase_resid_paired_fixedlag-final/"
# RELOAD_FROM   = None   # set to None to run the full CV + perm pipeline

# Optional. Path to a fresh neurons_with_final_roi_labels.csv. Only used when
# RELOAD_FROM is set; on reload the per-cell CSV's `roi` column is overwritten
# by the fresh table's `alt_final_roi` (joined on subject + cell_idx), so we
# can rebuild ROI stats + figures without re-running CV + perms. See
# mc.analyse.roi_relabel.relabel_per_cell. Set to None to disable.
RELABEL_FROM    = ("/Users/xpsy1114/Documents/projects/multiple_clocks/"
                    "data/ephys_humans/derivatives/"
                    "neurons_with_ROI_labels.csv")
# RELABEL_FROM  = None

# Cells / data --------------------------------------------------------
CELL_SET        = "all_in_roi_table"     # 'rsa', 'not_in_rsa', or 'all_in_roi_table'
SUBJECTS        = "all"                  # 'all' or list[int]
ROIS_KEEP       = None                   # None = all in registry
TRIALS          = "all_minus_explore"    # passed to filter_data

# Compute settings (override module defaults if you want) -------------
LAGS_DEG               = list(range(0, 360, 30))   # [0, 30, …, 330]
COVERAGE_MODE          = "paired"                  # pair grids → max coverage
SPARSITY_FILTER        = "gridwise_qc"
PHASE_RESIDUALISE      = "cosine"                  # only option supported
WEIGHTED_CORRELATION   = True
MIN_DWELL_BINS         = 25
MIN_SHARED_LOCS        = 5
N_PEAKS_FREE           = 1                         # for free-peak control
N_PERMUTATIONS         = 1000
RUN_PERMUTATIONS       = True
RANDOM_SEED            = 42
N_JOBS                 = -1
# A-priori per-ROI predicted lags (FIXED-lag analysis) ---------------
# mPFC: action / planning lookahead. HC: current location / place.
ROI_PREDICTED_LAGS_DEG = {
    "mPFC":        (30, 60),
    "ACC":         (30, 60),  # backwards compatibility for cached runs
    "HC_anterior": (0, 330),
    "HC_mid":      (0, 330),
}

# Single lags at which we run a lag-agnostic per-ROI t-test on the
# per-cell CV r. One test per (ROI × lag) — no lag sets. Complements the
# `predicted-lag` tests above by showing exactly where the signal sits.
SINGLE_LAGS_FOR_TESTS = [0, 30, 60, 330]

# Display-name override retained for cached runs that still use `ACC`.
ROI_DISPLAY_NAMES = {"ACC": "mPFC"}
def _disp(roi):
    return ROI_DISPLAY_NAMES.get(roi, roi)

# Statistical thresholds ---------------------------------------------
ALPHA = 0.05

# Plotting -----------------------------------------------------------
N_EXAMPLE_CELLS_PER_ROI         = 5         # top-N rate-map example PDFs
                                            # per (ROI, predicted-lag)
EXAMPLE_MIN_VALID_LOCATIONS     = 4         # accept if mean rate map has
                                            # ≥ this many finite locations
EXAMPLE_FOCUS_ROIS              = ["mPFC", "HC_anterior", "HC_mid"]
DPI                              = 300
CM                               = 1 / 2.54
FONT_TICK, FONT_AXIS, FONT_BIG   = 9, 10, 11

# z-MNI gradient analysis (mPFC) -------------------------------------
ZMNI_GRADIENT_ROI               = "mPFC"
ZMNI_GRADIENT_SUBSET_LAGS_DEG   = (0, 30, 60)   # "early-future" lags
ZMNI_GRADIENT_TOP_K_PER_SUBJECT = 1               # per subject per lag

# Colour scheme — canonical CLAUDE.md mapping. Source of truth is
# `mc.plotting.cell_results.SHOWGIRL2_DISCRETE` + `ROI_COLORS_SHOWGIRL2`.
# mPFC = idx 1 = dark teal-green ("#448363").
from mc.plotting.cell_results import (
    SHOWGIRL2_DISCRETE as _SHOWGIRL2,
    ROI_COLORS_SHOWGIRL2 as _ROI_IDX,
)
ROI_COLOURS = {r: _SHOWGIRL2[i] for r, i in _ROI_IDX.items()}
LOCATION_COLOURS = {
    1: "#0a607a", 2: "#7eb1c4", 3: "#b6d4e0",
    4: "#175e62", 5: "#5b9b8d", 6: "#c8e0d0",
    7: "#0e3d3a", 8: "#3d8b7d", 9: "#a7d9b2",
}


# ====================================================================
# Helpers — IO / aggregation
# ====================================================================
def _finite_list(xs):
    return [float(x) if (x is not None and np.isfinite(x)) else None
            for x in xs]


def _finite_nested(xss):
    return [_finite_list(xs) for xs in xss]


def _settings_dict():
    """Snapshot of every SETTINGS constant — saved alongside outputs."""
    return {
        "analysis_name": ANALYSIS_NAME,
        "cell_set": CELL_SET, "subjects": SUBJECTS, "rois_keep": ROIS_KEEP,
        "trials": TRIALS,
        "lags_deg": LAGS_DEG,
        "coverage_mode": COVERAGE_MODE, "sparsity_filter": SPARSITY_FILTER,
        "phase_residualise": PHASE_RESIDUALISE,
        "weighted_correlation": WEIGHTED_CORRELATION,
        "min_dwell_bins": MIN_DWELL_BINS, "min_shared_locs": MIN_SHARED_LOCS,
        "n_peaks_free": N_PEAKS_FREE,
        "n_permutations": N_PERMUTATIONS, "run_permutations": RUN_PERMUTATIONS,
        "random_seed": RANDOM_SEED, "n_jobs": N_JOBS,
        "roi_predicted_lags_deg": {k: list(v) for k, v in
                                    ROI_PREDICTED_LAGS_DEG.items()},
        "alpha": ALPHA,
    }


def _cfg_for_module(roi):
    """Build the per-cell `cfg` dict passed to `fsp.analyse_one_cell`."""
    return {
        "roi": roi,
        "roi_predicted_lags_deg": ROI_PREDICTED_LAGS_DEG,
        "lags_deg":               LAGS_DEG,
        "n_peaks":                N_PEAKS_FREE,
        "min_dwell_bins":         MIN_DWELL_BINS,
        "min_shared_locs":        MIN_SHARED_LOCS,
        "weighted_correlation":   WEIGHTED_CORRELATION,
        "n_loc":                  9,
        "n_permutations":         N_PERMUTATIONS,
        "run_permutations":       RUN_PERMUTATIONS,
        "random_seed":            RANDOM_SEED,
    }


def _row_from_result(out, neuron_id, sub_str, sub_int, cell_idx, roi_row):
    """Flatten one cell's compute output into a CSV row."""
    if out is None:
        return {
            "neuron_id":   neuron_id, "subject_id": sub_str,
            "subject_int": sub_int, "cell_idx": cell_idx,
            "roi": roi_row["roi"],
            "MNI_x": roi_row["MNI_x"], "MNI_y": roi_row["MNI_y"],
            "MNI_z": roi_row["MNI_z"],
            "note": "compute_failed",
        }
    # FIXED-lag block (may be absent if ROI has no a-priori lag).
    fixed_lags = out.get("fixed_fixed_lags_deg") or []
    fixed_r_mean      = out.get("fixed_fixed_lag_r_mean", np.nan)
    fixed_per_lag     = out.get("fixed_fixed_lag_per_lag_r", [])
    per_lag_all_lags  = out.get("fixed_per_lag_r_all_lags",
                                  out.get("free_consistency_curve_full", []))
    fixed_perm_means  = out.get("fixed_perm_r_means", [])
    fixed_perm_per_lag = out.get("fixed_perm_per_lag_r", [])

    # FREE-peak block.
    free_peak_r           = out.get("free_peak_r", np.nan)
    free_peak_lag         = out.get("free_peak_lag_plurality", np.nan)
    free_fold_lags        = out.get("free_fold_lags", [])
    free_fold_rs          = out.get("free_fold_rs", [])
    free_perm             = out.get("free_perm_peak_rs", [])
    free_n_groups         = out.get("free_n_groups_used", 0)
    free_shared           = out.get("free_mean_shared_locs_at_peak", np.nan)
    free_curve            = out.get("free_consistency_curve_full", [])

    return {
        "neuron_id":   neuron_id, "subject_id": sub_str,
        "subject_int": sub_int, "cell_idx": cell_idx,
        "roi": roi_row["roi"],
        "MNI_x": roi_row["MNI_x"], "MNI_y": roi_row["MNI_y"],
        "MNI_z": roi_row["MNI_z"],
        # free-peak control
        "free_peak_r":           float(free_peak_r) if np.isfinite(free_peak_r) else np.nan,
        "free_peak_lag":         int(free_peak_lag) if (isinstance(free_peak_lag, (int, np.integer))
                                                          or (np.isfinite(free_peak_lag) if isinstance(free_peak_lag, float) else False)) else np.nan,
        "free_n_groups_used":    int(free_n_groups),
        "free_mean_shared_locs": float(free_shared) if np.isfinite(free_shared) else np.nan,
        "free_fold_lags_json":   json.dumps(free_fold_lags),
        "free_fold_rs_json":     json.dumps(_finite_nested(free_fold_rs)),
        "free_curve_json":       json.dumps(_finite_list(free_curve)),
        "free_perm_peak_rs_json":json.dumps(_finite_list(free_perm)),
        # fixed-lag (a-priori per ROI)
        "fixed_lags_deg_used":   json.dumps(list(fixed_lags)),
        "fixed_lag_r_mean":      float(fixed_r_mean) if np.isfinite(fixed_r_mean) else np.nan,
        "fixed_per_lag_r_json":  json.dumps(_finite_list(fixed_per_lag)),
        "fixed_lag_perm_means_json": json.dumps(_finite_list(fixed_perm_means)),
        "fixed_lag_perm_per_lag_json": json.dumps(_finite_nested(fixed_perm_per_lag)),
        # per-lag full sweep (every cell, every lag) — needed for ROI×lag overview
        "per_lag_r_all_lags_json": json.dumps(_finite_list(per_lag_all_lags)),
        # paired-grid-config groupings (for the "paired_grid_configs" plot)
        "paired_config_groups_json": json.dumps(out.get("paired_config_groups", [])),
        "note": "ok",
    }


def _perm_p(observed, perms, alternative="greater"):
    """One-sided perm p with the standard (k+1)/(M+1) adjustment."""
    perms = [p for p in (perms or []) if p is not None and np.isfinite(p)]
    if not perms or not np.isfinite(observed):
        return np.nan
    if alternative == "greater":
        k = sum(1 for p in perms if p >= observed)
    else:
        k = sum(1 for p in perms if p <= observed)
    return (k + 1) / (len(perms) + 1)


def _add_perm_p_columns(df):
    """Add per-cell perm p-values for both fixed-lag and free-peak metrics."""
    df = df.copy()
    p_fixed, p_free = [], []
    for _, row in df.iterrows():
        try:
            free_perm = json.loads(row.get("free_perm_peak_rs_json", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            free_perm = []
        p_free.append(_perm_p(row.get("free_peak_r"), free_perm))
        try:
            fx_perm = json.loads(row.get("fixed_lag_perm_means_json", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            fx_perm = []
        p_fixed.append(_perm_p(row.get("fixed_lag_r_mean"), fx_perm))
    df["p_perm_free"]  = p_free
    df["p_perm_fixed"] = p_fixed
    return df


# ====================================================================
# Per-subject compute
# ====================================================================
def _build_payloads_for_subject(sub_str):
    data_raw = mc.analyse.helpers_human_cells.load_norm_data(
        DATA_DIR, [sub_str], res_data=(TRIALS == "residualised"),
    )
    if not data_raw:
        return None, None
    data = mc.analyse.helpers_human_cells.filter_data(
        data_raw, int(sub_str), TRIALS,
    )
    sub_dict = data[f"sub-{sub_str}"]
    return sub_dict, sub_dict["normalised_neurons"]


def _build_all_payloads(subjects, cells_df):
    """Load each subject's data ONCE, build payloads for every cell.

    Returns a flat list of dicts:
        { neuron_id, sub_str, sub_int, cell_idx, roi_row, payload, cfg }
    Subject data is discarded after its payloads are built, so peak
    memory is one subject at a time.
    """
    tasks = []
    for sub_str in subjects:
        sub_int = int(sub_str)
        kept_cell_idx = set(cells_df.loc[cells_df["subject_int"] == sub_int,
                                          "cell_idx"].tolist())
        if not kept_cell_idx:
            continue
        t_sub = time.time()
        sub_dict, norm_neurons = _build_payloads_for_subject(sub_str)
        if sub_dict is None:
            print(f"  sub-{sub_str}: SKIPPED (load failed)")
            continue
        sub_tasks = []
        for nid in norm_neurons:
            _, cell_idx = cs.parse_neuron_label(nid)
            if cell_idx is None or cell_idx not in kept_cell_idx:
                continue
            roi_row = cells_df[
                (cells_df["subject_int"] == sub_int)
                & (cells_df["cell_idx"]   == cell_idx)
            ].iloc[0].to_dict()
            roi = roi_row["roi"]
            try:
                payload = fsp.prepare_cell_data(
                    subject_data=sub_dict, neuron_id=nid,
                    neurons_df=norm_neurons[nid],
                    locations_df=sub_dict["locations"],
                    beh=sub_dict["beh"],
                    coverage_mode=COVERAGE_MODE,
                    sparsity_filter=SPARSITY_FILTER,
                    phase_residualise_basis=PHASE_RESIDUALISE,
                )
            except Exception as exc:
                print(f"  [{nid}] prepare_cell_data failed: {exc!r}")
                payload = None
            sub_tasks.append({
                "neuron_id": nid,
                "sub_str":   sub_str,
                "sub_int":   sub_int,
                "cell_idx":  cell_idx,
                "roi_row":   roi_row,
                "payload":   payload,
                "cfg":       _cfg_for_module(roi),
            })
        # free subject dict before moving on
        del sub_dict, norm_neurons
        print(f"  sub-{sub_str}: {len(sub_tasks)} payloads built "
              f"in {time.time() - t_sub:.1f}s")
        tasks.extend(sub_tasks)
    return tasks


def _run_one_task(task):
    """Compute one cell. Picklable for joblib (loky backend)."""
    nid     = task["neuron_id"]
    payload = task["payload"]
    cfg     = task["cfg"]
    if payload is None:
        out = None
    else:
        try:
            out = fsp.analyse_one_cell(payload, cfg)
        except Exception as exc:
            print(f"  [{nid}] analyse_one_cell failed: {exc!r}")
            out = None
    return _row_from_result(
        out, nid, task["sub_str"], task["sub_int"],
        task["cell_idx"], task["roi_row"],
    )


# ====================================================================
# Per-ROI statistical tests
# ====================================================================
#
# THREE TESTS, RUN PER ROI ON THE SET OF CELLS IN THAT ROI:
#   TEST 1 — MEAN_R_VS_0:
#       One-sample t-test: across cells in the ROI, is the *cross-
#       validated consistency r at the predicted lag* greater than 0?
#       (For ROIs without a predicted lag the test falls back to the
#       free-peak r, which is itself an unbiased control.)
#   TEST 2 — TARGET_VS_OTHER_LAGS:
#       Within each cell, compare r at the predicted lag(s) against the
#       mean r at the OTHER lags. One-sided paired t-test of that
#       per-cell difference across cells.
#   TEST 3 — PERM_SIG_FRACTION:
#       Per cell we computed a permutation p-value (observed r vs null
#       from circular shifts of the location series). Test if the
#       *fraction of cells with p_perm < α* exceeds chance (binomial,
#       one-sided greater, baseline = α).
#
# FDR scope:
#   Each test family is BH-FDR corrected ACROSS THE ROIs REPORTED IN
#   THIS TABLE — i.e. one FDR family per test, k = n_rois_reported. The
#   FDR-adjusted p is in the column ending in `_p_fdr`. Cross-test
#   correction is NOT applied (the three tests are reported jointly as
#   converging evidence, not as a single significance claim).
# ====================================================================
def _per_roi_stats(df):
    rows = []
    for roi, g in df.groupby("roi"):
        rec = {"roi": roi, "n_cells": int(len(g))}
        target_lags = ROI_PREDICTED_LAGS_DEG.get(roi, None)
        # TEST 1 — mean r vs 0 ----------------------------------------
        if target_lags is not None:
            r_vals = g["fixed_lag_r_mean"].to_numpy(dtype=float)
            rec["analysis_kind"]   = "fixed_lag"
            rec["target_lags_deg"] = list(target_lags)
        else:
            r_vals = g["free_peak_r"].to_numpy(dtype=float)
            rec["analysis_kind"]   = "free_peak"
            rec["target_lags_deg"] = None
        a = fsp.ttest_mean_r_vs_zero(r_vals)
        rec.update({
            "test1_meanR_n":      a["n"],
            "test1_meanR_mean":   a["mean"],
            "test1_meanR_t":      a["t"],
            "test1_meanR_p_unc":  a["p"],
        })
        # TEST 2 — predicted lag vs other lags within cell ------------
        if target_lags is not None:
            curves = []
            for _, row in g.iterrows():
                try:
                    c = json.loads(row.get("per_lag_r_all_lags_json", "[]") or "[]")
                except (json.JSONDecodeError, TypeError):
                    c = []
                if len(c) == len(LAGS_DEG):
                    curves.append([np.nan if v is None else float(v) for v in c])
            if curves:
                b = fsp.within_cell_lag_vs_others(
                    np.asarray(curves), LAGS_DEG, target_lags,
                )
            else:
                b = {"t": np.nan, "p": np.nan, "n": 0,
                     "mean_target": np.nan, "mean_others": np.nan,
                     "mean_diff": np.nan}
        else:
            b = {"t": np.nan, "p": np.nan, "n": 0,
                 "mean_target": np.nan, "mean_others": np.nan,
                 "mean_diff": np.nan}
        rec.update({
            "test2_targetVsOther_n":              b["n"],
            "test2_targetVsOther_mean_target":    b.get("mean_target", np.nan),
            "test2_targetVsOther_mean_others":    b.get("mean_others", np.nan),
            "test2_targetVsOther_mean_diff":      b.get("mean_diff",   np.nan),
            "test2_targetVsOther_t":              b["t"],
            "test2_targetVsOther_p_unc":          b["p"],
        })
        # TEST 3 — binomial on per-cell perm-p < ALPHA ---------------
        p_col = "p_perm_fixed" if target_lags is not None else "p_perm_free"
        c = fsp.binomial_sig_fraction(g[p_col].to_numpy(dtype=float),
                                        alpha=ALPHA)
        rec.update({
            "test3_permSig_k":         c["k"],
            "test3_permSig_n":         c["n"],
            "test3_permSig_frac":      c["frac"],
            "test3_permSig_p_unc":     c["p_binom"],
        })
        rows.append(rec)
    out = pd.DataFrame(rows)
    # BH-FDR within each test family, scope = all ROIs in this table
    for p_col, q_col in [
        ("test1_meanR_p_unc",         "test1_meanR_p_fdr"),
        ("test2_targetVsOther_p_unc", "test2_targetVsOther_p_fdr"),
        ("test3_permSig_p_unc",       "test3_permSig_p_fdr"),
    ]:
        out[q_col] = fsp.bh_fdr(out[p_col].to_numpy(dtype=float))
    return out


def _per_roi_lag_table(df):
    """Per-(ROI, lag) summary of the cross-validated consistency r.

    For every ROI × every lag we report:
        n_cells (with finite r at that lag),
        mean(r), SEM(r),
        one-sample t-stat & p (H1: mean > 0, uncorrected),
        BH-FDR-adjusted p, SCOPE = all (ROI × lag) cells in this table.
    """
    rois = sorted(df["roi"].dropna().unique().tolist())
    n_lags = len(LAGS_DEG)
    rows = []
    for roi in rois:
        g = df[df["roi"] == roi]
        curves = []
        for _, row in g.iterrows():
            try:
                c = json.loads(row.get("per_lag_r_all_lags_json", "[]") or "[]")
            except (json.JSONDecodeError, TypeError):
                c = []
            if len(c) == n_lags:
                curves.append([np.nan if v is None else float(v) for v in c])
        curves = np.asarray(curves) if curves else np.empty((0, n_lags))
        is_target = {l: (l in ROI_PREDICTED_LAGS_DEG.get(roi, ()))
                     for l in LAGS_DEG}
        for li, lag in enumerate(LAGS_DEG):
            r = curves[:, li] if curves.size else np.array([])
            r = r[np.isfinite(r)]
            if r.size >= 2:
                tt = stats.ttest_1samp(r, 0.0, alternative="greater")
                t_, p_ = float(tt.statistic), float(tt.pvalue)
                m_, s_ = float(r.mean()), float(r.std(ddof=1) / np.sqrt(r.size))
            else:
                t_, p_, m_, s_ = (np.nan, np.nan,
                                    float(r.mean()) if r.size else np.nan, np.nan)
            rows.append({
                "roi": roi, "lag_deg": lag,
                "is_predicted_lag": bool(is_target[lag]),
                "n_cells": int(r.size),
                "mean_r": m_, "sem_r": s_,
                "t_vs_0": t_, "p_unc": p_,
            })
    out = pd.DataFrame(rows)
    out["p_fdr"] = fsp.bh_fdr(out["p_unc"].to_numpy(dtype=float))
    return out


# ====================================================================
# Plots — all surface, intentionally inline + factored for clarity
# ====================================================================
def _stars_from_q(q):
    if not np.isfinite(q): return ""
    return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else ""


def _set_rc():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": FONT_TICK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _save(fig, save_stem):
    fig.savefig(save_stem + ".pdf", dpi=DPI, bbox_inches="tight")
    fig.savefig(save_stem + ".png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_roi_x_lag_overview(per_cell, roi_stats, save_stem):
    """Heatmap: rows = ROIs, cols = lags; colour = per-(ROI, lag) one-sample
    t-stat of CV r vs 0. Stars marked for the a-priori predicted lag(s) of
    each ROI (BH-FDR across the predicted-lag tests in `roi_stats`)."""
    _set_rc()
    rois = sorted(per_cell["roi"].dropna().unique().tolist())
    n_lags = len(LAGS_DEG)
    T = np.full((len(rois), n_lags), np.nan)
    for ri, roi in enumerate(rois):
        g = per_cell[per_cell["roi"] == roi]
        curves = []
        for _, row in g.iterrows():
            try:
                c = json.loads(row.get("per_lag_r_all_lags_json", "[]") or "[]")
            except (json.JSONDecodeError, TypeError):
                c = []
            if len(c) == n_lags:
                curves.append([np.nan if v is None else float(v) for v in c])
        if not curves: continue
        curves = np.asarray(curves)
        for li in range(n_lags):
            r = curves[:, li]
            r = r[np.isfinite(r)]
            if r.size >= 2:
                T[ri, li] = float(stats.ttest_1samp(r, 0.0,
                                                      alternative="greater").statistic)

    vmax = float(np.nanmax(np.abs(T))) if np.isfinite(T).any() else 1.0
    fig, ax = plt.subplots(figsize=(14 * CM, 6 * CM), constrained_layout=True)
    im = ax.imshow(T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    # FDR stars per ROI (TEST 1 q-val) — no predicted-lag outline
    q_lookup = dict(zip(roi_stats["roi"], roi_stats["test1_meanR_p_fdr"]))
    for ri, roi in enumerate(rois):
        star = _stars_from_q(q_lookup.get(roi, np.nan))
        if not star:
            continue
        target_lags = ROI_PREDICTED_LAGS_DEG.get(roi, ())
        if target_lags and target_lags[0] in LAGS_DEG:
            ci = LAGS_DEG.index(target_lags[0])
            col = "white" if abs(T[ri, ci]) > vmax * 0.55 else "black"
            ax.text(ci, ri, star, ha="center", va="center",
                    fontsize=FONT_BIG + 1, fontweight="bold",
                    color=col, zorder=5)
    ax.set_xticks(range(n_lags))
    ax.set_xticklabels([str(l) for l in LAGS_DEG], fontsize=FONT_TICK)
    ax.set_yticks(range(len(rois)))
    ax.set_yticklabels(rois, fontsize=FONT_TICK)
    ax.set_xlabel("lag (°)", fontsize=FONT_AXIS)
    ax.set_title("ROI × lag — per-cell CV r t-stat vs 0", fontsize=FONT_BIG)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cb.set_label("t (one-sided > 0)", fontsize=FONT_TICK)
    cb.ax.tick_params(labelsize=FONT_TICK)
    _save(fig, save_stem)


def plot_fixed_vs_free_comparison(per_cell, save_stem):
    """For each ROI in EXAMPLE_FOCUS_ROIS:
        box+jitter of (fixed-lag r) vs (free-peak r) across cells.
        Paired t-test annotation.
    """
    _set_rc()
    rois = [r for r in EXAMPLE_FOCUS_ROIS if r in per_cell["roi"].unique()]
    if not rois: return
    fig, axes = plt.subplots(1, len(rois), figsize=(4.5 * CM * len(rois), 5 * CM),
                              constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, roi in zip(axes, rois):
        g = per_cell[per_cell["roi"] == roi].copy()
        fx = g["fixed_lag_r_mean"].to_numpy(dtype=float)
        fr = g["free_peak_r"].to_numpy(dtype=float)
        m  = np.isfinite(fx) & np.isfinite(fr)
        fx, fr = fx[m], fr[m]
        # paired t-test
        if fx.size >= 2:
            try:
                res = stats.ttest_rel(fx, fr)
                t, p = float(res.statistic), float(res.pvalue)
            except Exception:
                t, p = np.nan, np.nan
        else:
            t, p = np.nan, np.nan
        c = ROI_COLOURS.get(roi, "#888")
        bp = ax.boxplot([fx, fr], positions=[1, 2], widths=0.55,
                         showfliers=False, patch_artist=True,
                         medianprops=dict(color="black", lw=0.8))
        for patch in bp["boxes"]:
            patch.set_facecolor(c); patch.set_alpha(0.55)
            patch.set_edgecolor("black")
        rng = np.random.default_rng(42)
        for i, vals in enumerate([fx, fr]):
            ax.scatter(np.full(vals.size, i + 1)
                        + 0.08 * rng.standard_normal(vals.size),
                       vals, s=6, color="black", alpha=0.4, zorder=3)
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["fixed lag", "free peak"], fontsize=FONT_TICK)
        ax.set_ylabel("CV consistency r", fontsize=FONT_AXIS)
        ax.set_title(f"{roi}\n n = {fx.size}\n"
                     f"paired t = {t:+.2f}, p = {p:.3g}",
                     fontsize=FONT_TICK)
        ax.tick_params(axis="both", labelsize=FONT_TICK, length=2, pad=1)
    fig.suptitle("Fixed-lag (a-priori) vs free-peak (cell-picked) — "
                 "predicted ROIs", fontsize=FONT_BIG)
    _save(fig, save_stem)


def _select_example_cells(per_cell, roi, n_want=N_EXAMPLE_CELLS_PER_ROI,
                          coverage_min=EXAMPLE_MIN_VALID_LOCATIONS):
    """Pick high-r, high-spatial-coverage cells from one ROI for the rate-
    map example PDFs. Requires the cell to have n ≥ `coverage_min` valid
    locations on its mean rate map at the predicted lag (or at the peak
    lag if no predicted lag)."""
    g = per_cell[per_cell["roi"] == roi].copy()
    if g.empty: return []
    if roi in ROI_PREDICTED_LAGS_DEG:
        g = g.dropna(subset=["fixed_lag_r_mean"])
        g = g.sort_values("fixed_lag_r_mean", ascending=False)
    else:
        g = g.dropna(subset=["free_peak_r"])
        g = g.sort_values("free_peak_r", ascending=False)
    return g.head(n_want * 3).to_dict("records")   # head extra; we drop on coverage


def _rate_maps_for_cell(neuron_id, sub_str, lag_deg):
    """Re-compute the per-grid-group rate maps for one cell at one lag,
    just for plotting. Returns (rate_maps (n_loc, n_groups),
    dwell_maps, paired_config_groups, group_n_reps)."""
    sub_dict, norm_neurons = _build_payloads_for_subject(sub_str)
    if sub_dict is None or neuron_id not in norm_neurons:
        return None
    payload = fsp.prepare_cell_data(
        subject_data=sub_dict, neuron_id=neuron_id,
        neurons_df=norm_neurons[neuron_id],
        locations_df=sub_dict["locations"], beh=sub_dict["beh"],
        coverage_mode=COVERAGE_MODE, sparsity_filter=SPARSITY_FILTER,
        phase_residualise_basis=PHASE_RESIDUALISE,
    )
    if payload is None: return None
    fr_maps, dwell_maps, _, groups = fsp.consistency_per_lag(
        payload["neurons"], payload["locations"], payload["grid_group_idx"],
        lags_deg=[int(lag_deg)],
        min_dwell=MIN_DWELL_BINS, min_shared_locs=MIN_SHARED_LOCS,
        weighted=WEIGHTED_CORRELATION, n_loc=9,
    )
    # group_n_reps: how many reps were in each grid group
    grp_arr = payload["grid_group_idx"]
    n_reps_per_group = [int((grp_arr == g).sum()) for g in groups]
    return {
        "fr_maps":   fr_maps[0],
        "dwell":     dwell_maps[0],
        "groups":    list(groups),
        "n_reps":    n_reps_per_group,
        "paired_config_groups": payload.get("paired_config_groups", []),
    }


# def _plot_one_cell_rate_maps(cell_row, lag_deg, save_stem, label_extra=""):
#     """One PDF for one cell: a row of rate-map heatmaps, one per grid
#     group (= per CV split if coverage_mode='paired'). Also annotates which
#     original configs are paired in each group (top of each panel)."""
#     _set_rc()
#     info = _rate_maps_for_cell(cell_row["neuron_id"],
#                                 str(cell_row["subject_id"]),
#                                 lag_deg=lag_deg)
#     if info is None or info["fr_maps"].shape[1] < 1:
#         return
#     n_panels = info["fr_maps"].shape[1] + 1   # +1 for mean
#     fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * CM * n_panels,
#                                                       3.2 * CM),
#                               constrained_layout=True, squeeze=False)
#     axes = axes[0]
#     # shared vmin/vmax for visual comparison across panels
#     all_vals = info["fr_maps"][np.isfinite(info["fr_maps"])]
#     vmin = float(np.nanpercentile(all_vals, 5))  if all_vals.size else 0.0
#     vmax = float(np.nanpercentile(all_vals, 95)) if all_vals.size else 1.0
#     cmap = "coolwarm"
#     paired = info["paired_config_groups"] or [[g] for g in info["groups"]]
#     for gi, ax in enumerate(axes[:-1]):
#         rm = info["fr_maps"][:, gi].reshape(3, 3)
#         ax.imshow(rm, cmap=cmap, vmin=vmin, vmax=vmax)
#         ax.set_xticks([]); ax.set_yticks([])
#         cfgs = paired[gi] if gi < len(paired) else [info["groups"][gi]]
#         ax.set_title(f"grp {gi+1}\ncfgs {cfgs}\nn reps {info['n_reps'][gi]}",
#                      fontsize=FONT_TICK)
#     # mean panel
#     ax = axes[-1]
#     mean_rm = np.nanmean(info["fr_maps"], axis=1).reshape(3, 3)
#     im = ax.imshow(mean_rm, cmap=cmap, vmin=vmin, vmax=vmax)
#     ax.set_xticks([]); ax.set_yticks([])
#     ax.set_title("mean", fontsize=FONT_TICK)
#     cb = fig.colorbar(im, ax=axes[-1], fraction=0.07, pad=0.04)
#     cb.ax.tick_params(labelsize=FONT_TICK)
#     fig.suptitle(
#         f"{cell_row['neuron_id']}  [{cell_row['roi']}]  lag = {lag_deg}°"
#         + (f"  {label_extra}" if label_extra else ""),
#         fontsize=FONT_TICK,
#     )
#     _save(fig, save_stem)
def _plot_one_cell_rate_maps(cell_row, lag_deg, save_stem, label_extra=""):
    """One PDF for one cell: a row of rate-map heatmaps, one per grid
    group (= per CV split if coverage_mode='paired'). Also annotates which
    original configs are paired in each group (top of each panel)."""
    _set_rc()
    info = _rate_maps_for_cell(cell_row["neuron_id"],
                                str(cell_row["subject_id"]),
                                lag_deg=lag_deg)

    # double_lag_degs = np.tile(range(0,360, 30),2)
    # ctrl_lag = np.where(double_lag_degs == lag_deg+180)[0][0]
    if lag_deg >= 180:
        ctrl_lag = lag_deg-180
    else:
        ctrl_lag = lag_deg+180
        
    info_ctrl = _rate_maps_for_cell(cell_row["neuron_id"],
                                str(cell_row["subject_id"]),
                                lag_deg=ctrl_lag)
    
    if info is None or info["fr_maps"].shape[1] < 1 or info_ctrl is None:
        return

    n_panels = info["fr_maps"].shape[1] + 1   # +1 for mean
    # add a second control lag.
    fig, axes_all = plt.subplots(2, n_panels, figsize=(2.6 * CM * n_panels,
                                                      2*3.2 * CM),
                              constrained_layout=True, squeeze=False)
    
    
    axes = axes_all[0,:]
    # shared vmin/vmax for visual comparison across panels
    all_vals = info["fr_maps"][np.isfinite(info["fr_maps"])]
    vmin = float(np.nanpercentile(all_vals, 5))  if all_vals.size else 0.0
    vmax = float(np.nanpercentile(all_vals, 95)) if all_vals.size else 1.0
    cmap = "coolwarm"
    paired = info["paired_config_groups"] or [[g] for g in info["groups"]]
    for gi, ax in enumerate(axes[:-1]):
        rm = info["fr_maps"][:, gi].reshape(3, 3)
        ax.imshow(rm, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        cfgs = paired[gi] if gi < len(paired) else [info["groups"][gi]]
        ax.set_title(f"grp {gi+1}\ncfgs {cfgs}\nn reps {info['n_reps'][gi]}",
                     fontsize=FONT_TICK)
    # mean panel
    ax = axes_all[0,-1]
    mean_rm = np.nanmean(info["fr_maps"], axis=1).reshape(3, 3)
    im = ax.imshow(mean_rm, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"mean lag {lag_deg}", fontsize=FONT_TICK)
    cb = fig.colorbar(im, ax=axes[-1], fraction=0.07, pad=0.04)
    cb.ax.tick_params(labelsize=FONT_TICK)
    
    #import pdb; pdb.set_trace()
    # second row.
    axes_ctrl = axes_all[1,:]
    # shared vmin/vmax for visual comparison across panels
    all_ctrl_vals = info_ctrl["fr_maps"][np.isfinite(info_ctrl["fr_maps"])]
    vmin = float(np.nanpercentile(all_ctrl_vals, 5))  if all_ctrl_vals.size else 0.0
    vmax = float(np.nanpercentile(all_ctrl_vals, 95)) if all_ctrl_vals.size else 1.0
    cmap = "coolwarm"
    paired = info["paired_config_groups"] or [[g] for g in info["groups"]]
    for gi, ax in enumerate(axes_ctrl[:-1]):
        rm_ctrl = info_ctrl["fr_maps"][:, gi].reshape(3, 3)
        ax.imshow(rm_ctrl, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        cfgs = paired[gi] if gi < len(paired) else [info["groups"][gi]]
        ax.set_title(f"grp {gi+1}\ncfgs {cfgs}\nn reps {info['n_reps'][gi]}",
                      fontsize=FONT_TICK)
    # mean panel
    ax = axes_all[1, -1]
    mean_rm_ctrl = np.nanmean(info_ctrl["fr_maps"], axis=1).reshape(3, 3)
    im = ax.imshow(mean_rm_ctrl, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"mean lag {ctrl_lag}", fontsize=FONT_TICK)
    cb = fig.colorbar(im, ax=axes_ctrl[-1], fraction=0.07, pad=0.04)
    cb.ax.tick_params(labelsize=FONT_TICK)
    
    
    fig.suptitle(
        f"{cell_row['neuron_id']}  [{cell_row['roi']}]  lag = {lag_deg} \n control lag = {ctrl_lag}°"
        + (f"  {label_extra}" if label_extra else ""),
        fontsize=FONT_TICK,
    )
    _save(fig, save_stem)
    

# ====================================================================
# Per-test result figures
# ====================================================================
def _rois_for_stats(roi_stats):
    """Return ROIs in the order present in roi_stats (sorted)."""
    return sorted(roi_stats["roi"].dropna().unique().tolist())


def plot_test1_mean_r_histograms(per_cell, roi_stats, save_stem):
    """TEST 1 — population shift of CV r.

    One subplot per ROI: histogram of the cell-level r (fixed-lag r where
    a predicted lag exists, free-peak r otherwise). Perm-significant cells
    (p_perm < α) overlaid as a darker bin colour. Annotates t / p / p_fdr.
    """
    _set_rc()
    rois = _rois_for_stats(roi_stats)
    n = len(rois)
    n_cols = min(n, 4)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.0 * CM * n_cols, 3.6 * CM * n_rows),
                              constrained_layout=True, squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[n:]:
        ax.axis("off")
    for ax, roi in zip(axes_flat, rois):
        g = per_cell[per_cell["roi"] == roi]
        kind = roi_stats.loc[roi_stats["roi"] == roi, "analysis_kind"].iloc[0]
        if kind == "fixed_lag":
            r_col, p_col = "fixed_lag_r_mean", "p_perm_fixed"
            xlab = "CV r at predicted lag"
        else:
            r_col, p_col = "free_peak_r", "p_perm_free"
            xlab = "CV r at free peak"
        r = g[r_col].to_numpy(dtype=float)
        p = g[p_col].to_numpy(dtype=float)
        m = np.isfinite(r)
        r, p = r[m], p[m]
        sig = np.isfinite(p) & (p < ALPHA)

        col = ROI_COLOURS.get(roi, "#888")
        # bin range covering both populations
        if r.size:
            lo, hi = float(np.nanmin(r)), float(np.nanmax(r))
            pad = max(0.02, 0.05 * (hi - lo))
            bins = np.linspace(lo - pad, hi + pad, 21)
            ax.hist(r, bins=bins, color=col, alpha=0.45, edgecolor="white",
                    linewidth=0.4, label=f"all (n={r.size})")
            if sig.any():
                ax.hist(r[sig], bins=bins, color=col, alpha=1.0,
                        edgecolor="black", linewidth=0.4,
                        label=f"perm-sig (n={int(sig.sum())})")
        ax.axvline(0.0, color="black", lw=0.6, ls="--")
        rs = roi_stats[roi_stats["roi"] == roi].iloc[0]
        t_ = rs["test1_meanR_t"]
        p_ = rs["test1_meanR_p_unc"]
        q_ = rs["test1_meanR_p_fdr"]
        ax.set_title(
            f"{roi}\n t = {t_:+.2f}   p = {p_:.3g}\n p_FDR = {q_:.3g}",
            fontsize=FONT_TICK,
        )
        ax.set_xlabel(xlab, fontsize=FONT_TICK)
        ax.set_ylabel("# cells",       fontsize=FONT_TICK)
        ax.tick_params(axis="both", labelsize=FONT_TICK, length=2, pad=1)
        ax.legend(fontsize=FONT_TICK - 1, frameon=False, loc="upper left")
    fig.suptitle(
        "TEST 1 — population shift of CV r away from 0\n"
        "(one-sample t-test, one-sided > 0; FDR across "
        f"{len(rois)} ROIs)",
        fontsize=FONT_BIG,
    )
    _save(fig, save_stem)


def plot_test2_target_vs_others_lines(per_cell, roi_stats, save_stem):
    """TEST 2 — per-(ROI) line plot of mean r ± SEM across all lags,
    predicted lag(s) highlighted in dark green. Each cell contributes one
    full curve to the average. Annotates within-cell paired t (target vs
    others) and its FDR-corrected p.
    """
    _set_rc()
    rois = _rois_for_stats(roi_stats)
    rois_with_target = [r for r in rois if ROI_PREDICTED_LAGS_DEG.get(r)]
    rois_show = rois_with_target or rois
    n = len(rois_show)
    n_cols = min(n, 4)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * CM * n_cols, 6 * CM * n_rows),
                              constrained_layout=True, squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[n:]:
        ax.axis("off")
    n_lags = len(LAGS_DEG)
    x = np.asarray(LAGS_DEG)
    for ax, roi in zip(axes_flat, rois_show):
        g = per_cell[per_cell["roi"] == roi]
        curves = []
        for _, row in g.iterrows():
            try:
                c = json.loads(row.get("per_lag_r_all_lags_json", "[]") or "[]")
            except (json.JSONDecodeError, TypeError):
                c = []
            if len(c) == n_lags:
                curves.append([np.nan if v is None else float(v) for v in c])
        col = ROI_COLOURS.get(roi, "#888")
        if curves:
            curves = np.asarray(curves)
            m = np.nanmean(curves, axis=0)
            s = np.nanstd(curves, axis=0, ddof=1) / np.sqrt(
                np.maximum(np.isfinite(curves).sum(axis=0), 1)
            )
            ax.fill_between(x, m - s, m + s, color=col, alpha=0.25,
                            linewidth=0)
            ax.plot(x, m, color=col, lw=1.6, marker="o", ms=2.5)
        ax.axhline(0.0, color="black", lw=0.5, ls="--")
        for tl in ROI_PREDICTED_LAGS_DEG.get(roi, ()):
            ax.axvline(tl, color="#0e3d3a", lw=0.9, ls=":", alpha=0.7)
        rs = roi_stats[roi_stats["roi"] == roi].iloc[0]
        t_ = rs["test2_targetVsOther_t"]
        p_ = rs["test2_targetVsOther_p_unc"]
        q_ = rs["test2_targetVsOther_p_fdr"]
        ax.set_title(
            f"{roi}   target = {list(ROI_PREDICTED_LAGS_DEG.get(roi, ()))}°\n"
            f"paired t = {t_:+.2f}   p = {p_:.3g}\n p_FDR = {q_:.3g}",
            fontsize=FONT_TICK,
        )
        ax.set_xlabel("lag (°)", fontsize=FONT_TICK)
        ax.set_ylabel("mean CV r", fontsize=FONT_TICK)
        ax.set_xticks(LAGS_DEG[::2])
        ax.tick_params(axis="both", labelsize=FONT_TICK, length=2, pad=1)
    fig.suptitle(
        "TEST 2 — within-cell predicted-lag r > other-lags r\n"
        "(paired t-test, one-sided greater; "
        f"FDR across {sum(1 for r in rois if ROI_PREDICTED_LAGS_DEG.get(r))} "
        "predicted-lag ROIs)",
        fontsize=FONT_BIG,
    )
    # import pdb; pdb.set_trace()
    _save(fig, save_stem)


def plot_test3_perm_sig_fraction_bars(per_cell, roi_stats, save_stem):
    """TEST 3 — per-ROI fraction of cells with p_perm < α.

    Bars: fraction. Horizontal line: chance = α. Stars: BH-FDR-significant.
    """
    _set_rc()
    rois = _rois_for_stats(roi_stats)
    fracs, ks, ns, qs = [], [], [], []
    for roi in rois:
        rs = roi_stats[roi_stats["roi"] == roi].iloc[0]
        fracs.append(float(rs["test3_permSig_frac"]) if np.isfinite(rs["test3_permSig_frac"]) else np.nan)
        ks.append(int(rs["test3_permSig_k"]) if np.isfinite(rs["test3_permSig_k"]) else 0)
        ns.append(int(rs["test3_permSig_n"]) if np.isfinite(rs["test3_permSig_n"]) else 0)
        qs.append(float(rs["test3_permSig_p_fdr"]) if np.isfinite(rs["test3_permSig_p_fdr"]) else np.nan)

    fig, ax = plt.subplots(figsize=(9 * CM, 5.5 * CM), constrained_layout=True)
    xs = np.arange(len(rois))
    bar_cols = [ROI_COLOURS.get(r, "#888") for r in rois]
    ax.bar(xs, fracs, color=bar_cols, edgecolor="black", linewidth=0.5, width=0.7)
    ax.axhline(ALPHA, color="black", lw=0.7, ls="--",
               label=f"chance = α = {ALPHA}")
    for xi, (frac, k, n, q) in enumerate(zip(fracs, ks, ns, qs)):
        if not np.isfinite(frac):
            continue
        ax.text(xi, frac + 0.005, f"{k}/{n}",
                ha="center", va="bottom", fontsize=FONT_TICK - 1)
        s = _stars_from_q(q)
        if s:
            ax.text(xi, frac + 0.03, s, ha="center", va="bottom",
                    fontsize=FONT_BIG + 1, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(rois, rotation=35, ha="right", fontsize=FONT_TICK)
    ax.set_ylabel(f"frac. cells with p_perm < {ALPHA}", fontsize=FONT_AXIS)
    ax.set_title(
        "TEST 3 — fraction of cells exceeding the permutation null\n"
        f"(binomial vs chance = α; FDR across {len(rois)} ROIs)",
        fontsize=FONT_TICK,
    )
    ax.legend(fontsize=FONT_TICK - 1, frameon=False, loc="upper right")
    ax.tick_params(axis="y", labelsize=FONT_TICK, length=2, pad=1)
    _save(fig, save_stem)


def plot_roi_x_lag_table_heatmap(roi_x_lag, save_stem):
    """Companion to TEST 2 / overview: heatmap of mean r per (ROI, lag),
    with BH-FDR-significant cells starred (FDR across the FULL table)."""
    _set_rc()
    rois = sorted(roi_x_lag["roi"].unique().tolist())
    n_lags = len(LAGS_DEG)
    M = np.full((len(rois), n_lags), np.nan)
    Q = np.full((len(rois), n_lags), np.nan)
    for ri, roi in enumerate(rois):
        sub = roi_x_lag[roi_x_lag["roi"] == roi]
        for _, row in sub.iterrows():
            li = LAGS_DEG.index(int(row["lag_deg"]))
            M[ri, li] = row["mean_r"]
            Q[ri, li] = row["p_fdr"]
    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 0.05
    fig, ax = plt.subplots(figsize=(14 * CM, 6 * CM), constrained_layout=True)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    # FDR stars per (ROI, lag)
    for ri in range(len(rois)):
        for li in range(n_lags):
            q = Q[ri, li]
            if not np.isfinite(q): continue
            s = _stars_from_q(q)
            if not s: continue
            col = "white" if abs(M[ri, li]) > vmax * 0.55 else "black"
            ax.text(li, ri, s, ha="center", va="center",
                    fontsize=FONT_TICK, fontweight="bold", color=col)
    ax.set_xticks(range(n_lags))
    ax.set_xticklabels([str(l) for l in LAGS_DEG], fontsize=FONT_TICK)
    ax.set_yticks(range(len(rois)))
    ax.set_yticklabels(rois, fontsize=FONT_TICK)
    ax.set_xlabel("lag (°)", fontsize=FONT_AXIS)
    ax.set_title(
        "ROI × lag overview — mean CV r per (ROI, lag)\n"
        "(stars: BH-FDR across the full ROI × lag table)",
        fontsize=FONT_TICK,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cb.set_label("mean r", fontsize=FONT_TICK)
    cb.ax.tick_params(labelsize=FONT_TICK)
    _save(fig, save_stem)


# ====================================================================
# mPFC z-MNI gradient across lags
# ====================================================================
def _per_subject_lag_z_matrix(per_cell, roi, lags, top_k=1):
    """For each subject in `roi`, pick the top-k cells by per-lag r at
    each lag; return mean MNI_z per (subject, lag).

    Returns matrix of shape (n_subjects, n_lags) and the subject ids.
    Subjects with NO cells in this ROI are dropped.
    """
    g = per_cell[per_cell["roi"] == roi].copy()
    if g.empty:
        return np.empty((0, len(lags))), []
    g["_per_lag"] = g["per_lag_r_all_lags_json"].apply(
        lambda s: json.loads(s) if isinstance(s, str) and s else []
    )
    lag_to_idx = {l: LAGS_DEG.index(l) for l in lags}
    subjects = sorted(g["subject_id"].unique().tolist())
    Z = np.full((len(subjects), len(lags)), np.nan)
    for si, sub in enumerate(subjects):
        gs = g[g["subject_id"] == sub]
        for li, lag in enumerate(lags):
            ix = lag_to_idx[lag]
            rs, zs = [], []
            for _, row in gs.iterrows():
                vec = row["_per_lag"]
                if len(vec) != len(LAGS_DEG):
                    continue
                v = vec[ix]
                if v is None or not np.isfinite(v):
                    continue
                if not np.isfinite(row.get("MNI_z", np.nan)):
                    continue
                rs.append(float(v))
                zs.append(float(row["MNI_z"]))
            if not rs:
                continue
            order = np.argsort(rs)[::-1]
            picks = order[:top_k]
            Z[si, li] = float(np.mean([zs[k] for k in picks]))
    return Z, subjects


def _linear_trend_test(Z, lag_positions):
    """Per-subject OLS slope of z against lag position. One-sample t-test
    that mean slope ≠ 0. Returns dict(n, slope_mean, t, p_two)."""
    x = np.asarray(lag_positions, dtype=float)
    slopes = []
    for row in Z:
        m = np.isfinite(row)
        if m.sum() < 2:
            continue
        b, _ = np.polyfit(x[m], row[m], 1), None
        slopes.append(float(b[0]))
    slopes = np.asarray(slopes, dtype=float)
    if slopes.size < 2:
        return {"n": int(slopes.size), "slope_mean": np.nan,
                "t": np.nan, "p_two": np.nan}
    res = stats.ttest_1samp(slopes, 0.0)
    return {"n": int(slopes.size), "slope_mean": float(slopes.mean()),
            "t": float(res.statistic), "p_two": float(res.pvalue)}


def plot_zmni_gradient(per_cell, save_stem):
    """Two-panel z-MNI gradient for mPFC neurons across lags.
    LEFT  — predicted-lag subset ZMNI_GRADIENT_SUBSET_LAGS_DEG (e.g. 0/30/60)
    RIGHT — all lags in LAGS_DEG
    Each panel: per-subject grey lines + across-subject mean ± SEM + the
    fitted linear trend; annotates t / p of the per-subject slope test.
    """
    _set_rc()
    roi = ZMNI_GRADIENT_ROI
    K   = ZMNI_GRADIENT_TOP_K_PER_SUBJECT
    fig, axes = plt.subplots(1, 2, figsize=(15 * CM, 6 * CM),
                              constrained_layout=True)
    panels = [
        ("subset",  list(ZMNI_GRADIENT_SUBSET_LAGS_DEG)),
        ("all",     list(LAGS_DEG)),
    ]
    for ax, (kind, lags) in zip(axes, panels):
        Z, subjects = _per_subject_lag_z_matrix(per_cell, roi, lags, top_k=K)
        if Z.size == 0 or Z.shape[0] == 0:
            ax.text(0.5, 0.5, f"no {roi} cells", ha="center", va="center")
            ax.axis("off"); continue
        x = np.asarray(lags, dtype=float)
        # per-subject lines (grey)
        for row in Z:
            m = np.isfinite(row)
            if m.sum() < 2: continue
            ax.plot(x[m], row[m], color="lightgray", lw=0.8, alpha=0.6)
        # group mean ± SEM
        mean = np.nanmean(Z, axis=0)
        sem  = np.nanstd(Z, axis=0, ddof=1) / np.sqrt(
            np.maximum(np.isfinite(Z).sum(axis=0), 1)
        )
        col = ROI_COLOURS.get(roi, "darkred")
        ax.fill_between(x, mean - sem, mean + sem, color=col, alpha=0.25,
                        linewidth=0)
        ax.errorbar(x, mean, yerr=sem, marker="o", lw=1.8, capsize=3,
                    color=col, label=f"{roi}: n = {Z.shape[0]} subj")
        # linear trend fit + test
        finite_mean = np.isfinite(mean)
        if finite_mean.sum() >= 2:
            b1, b0 = np.polyfit(x[finite_mean], mean[finite_mean], 1)
            xs_dense = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs_dense, b0 + b1 * xs_dense, color="#0e3d3a", lw=1.2,
                    ls="--", label=f"trend slope = {b1:+.3f} mm/°")
        trend = _linear_trend_test(Z, x)
        ax.axhline(np.nanmean(Z), color="gray", lw=0.4, ls=":")
        ax.set_xlabel("lag (°)", fontsize=FONT_AXIS)
        ax.set_ylabel("MNI z (mm)", fontsize=FONT_AXIS)
        title_kind = ("predicted-lag subset" if kind == "subset"
                       else "all lags")
        ax.set_title(
            f"{roi} z-MNI gradient — {title_kind}\n"
            f"per-subj slope t = {trend['t']:+.2f}  "
            f"p = {trend['p_two']:.3g}  (n = {trend['n']} subj)",
            fontsize=FONT_TICK,
        )
        ax.set_xticks(lags)
        ax.tick_params(axis="both", labelsize=FONT_TICK, length=2, pad=1)
        ax.legend(fontsize=FONT_TICK - 1, frameon=False, loc="best")
    fig.suptitle(
        f"{roi} — z-MNI projection of top-{K} cells per subject per lag",
        fontsize=FONT_BIG,
    )
    _save(fig, save_stem)


def _rank_cells_by_lag(per_cell, roi, lag_deg):
    """Sort cells in `roi` by per-cell consistency r at `lag_deg` (descending)."""
    g = per_cell[per_cell["roi"] == roi].copy()
    if g.empty:
        return g
    ix = LAGS_DEG.index(int(lag_deg))
    rs = []
    for _, row in g.iterrows():
        try:
            c = json.loads(row.get("per_lag_r_all_lags_json", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            c = []
        rs.append(float(c[ix]) if (len(c) == len(LAGS_DEG) and c[ix] is not None
                                     and np.isfinite(c[ix])) else np.nan)
    g["_r_at_lag"] = rs
    g = g.dropna(subset=["_r_at_lag"])
    return g.sort_values("_r_at_lag", ascending=False)


def plot_rate_map_examples(per_cell, out_root):
    """Top-N rate-map PDFs per (ROI, predicted-lag) for the focus ROIs.

    Writes to:  <out_root>/rate_map_examples/<ROI>/lag<deg>/<rank>_<id>.pdf
    Ranking: per-cell observed consistency r at THAT lag (not averaged across
    predicted lags), with a coverage filter of `EXAMPLE_MIN_VALID_LOCATIONS`
    finite locations on the cell's mean rate map at that lag.
    """
    base = Path(out_root) / "rate_map_examples"
    base.mkdir(parents=True, exist_ok=True)
    for roi in EXAMPLE_FOCUS_ROIS:
        target_lags = ROI_PREDICTED_LAGS_DEG.get(roi, ())
        if not target_lags:
            continue
        for lag in target_lags:
            lag_dir = base / roi / f"lag{int(lag):03d}"
            lag_dir.mkdir(parents=True, exist_ok=True)
            ranked = _rank_cells_by_lag(per_cell, roi, lag)
            saved, tried = 0, 0
            for _, cell_row in ranked.iterrows():
                if saved >= N_EXAMPLE_CELLS_PER_ROI:
                    break
                tried += 1
                # cap how many we try (saves I/O)
                if tried > N_EXAMPLE_CELLS_PER_ROI * 6:
                    break
                try:
                    info = _rate_maps_for_cell(cell_row["neuron_id"],
                                                str(cell_row["subject_id"]),
                                                lag_deg=int(lag))
                except Exception as exc:
                    print(f"    [{cell_row['neuron_id']}] rate-map failed: {exc!r}")
                    continue
                if info is None or info["fr_maps"].shape[1] < 1:
                    continue
                # coverage check on the MEAN rate map
                mean_rm = np.nanmean(info["fr_maps"], axis=1)
                if np.isfinite(mean_rm).sum() < EXAMPLE_MIN_VALID_LOCATIONS:
                    continue
                stem = lag_dir / (
                    f"{saved+1:02d}_sub-{cell_row['subject_id']}"
                    f"_{cell_row['neuron_id'][:30]}"
                )
                r_at_lag = cell_row.get("_r_at_lag", float("nan"))
                try:
                    _plot_one_cell_rate_maps(
                        cell_row, int(lag), str(stem),
                        label_extra=f"r@{int(lag)}° = {r_at_lag:+.3f}",
                    )
                except Exception as exc:
                    print(f"    [{cell_row['neuron_id']}] plot failed: {exc!r}")
                    continue
                saved += 1
            print(f"  rate-map examples — {roi} @ lag {int(lag)}°: "
                  f"saved {saved} PDFs → {lag_dir}")


# ====================================================================
# Runner
# ====================================================================
def _reload_per_cell(reload_from):
    """Load per-cell results from a previous run.

    `reload_from` may be either a direct CSV path, or a directory in
    which case we look (in order) for:
        per_cell.csv           — full run output
        per_cell_filtered.csv  — subset replot output (e.g. replot_no_rsa_cells)
    """
    p = Path(reload_from)
    if p.suffix == ".csv":
        csv = p
    else:
        candidates = [p / "per_cell.csv", p / "per_cell_filtered.csv"]
        csv = next((c for c in candidates if c.exists()), candidates[0])
    if not csv.exists():
        raise FileNotFoundError(
            f"No per_cell.csv or per_cell_filtered.csv found under {p}"
        )
    df = pd.read_csv(csv)
    if "p_perm_fixed" not in df.columns or "p_perm_free" not in df.columns:
        df = _add_perm_p_columns(df)
    return df, csv.parent


def run():
    np.random.seed(RANDOM_SEED)

    # ---- Reload-only branch -------------------------------------------
    if RELOAD_FROM is not None:
        per_cell, src_dir = _reload_per_cell(RELOAD_FROM)

        relabel_tag = ""
        if RELABEL_FROM is not None:
            from mc.analyse.roi_relabel import relabel_per_cell
            per_cell, _audit = relabel_per_cell(
                per_cell, roi_table_csv=RELABEL_FROM,
                roi_col_in_table="alt_final_roi",
                subject_key_per_cell="subject_int",
                cell_key_per_cell="cell_idx")
            relabel_tag = "_relabelled"

        run_tag = (datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                   + f"_reload_from_{src_dir.name}{relabel_tag}")
        out_dir = Path(OUT_BASE) / run_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        settings_snap = {**_settings_dict(), "reload_from": str(src_dir)}
        if RELABEL_FROM is not None:
            settings_snap["relabel_from"] = str(RELABEL_FROM)
        with open(out_dir / "settings.json", "w") as f:
            json.dump(settings_snap, f, indent=2)
        print(f"[{run_tag}]")
        print(f"  RELOAD mode — using per_cell from {src_dir}")
        if RELABEL_FROM is not None:
            print(f"  RELABEL mode — ROI col overwritten from {RELABEL_FROM}")
        print(f"  {len(per_cell)} cells loaded; skipping CV + perms.")
        print(f"  out_dir = {out_dir}")
        per_cell.to_csv(out_dir / "per_cell.csv", index=False)
        return _stats_and_plots(per_cell, out_dir, skip_raw_plots=True)

    run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + ANALYSIS_NAME
    out_dir = Path(OUT_BASE) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # snapshot settings
    with open(out_dir / "settings.json", "w") as f:
        json.dump(_settings_dict(), f, indent=2)

    # cell registry
    cells_df = cs.load_cells(cell_set=CELL_SET, subjects=SUBJECTS,
                              rois_keep=ROIS_KEEP)
    cells_df.to_csv(out_dir / "cells_used.csv", index=False)
    subjects = sorted(cells_df["subject_id"].unique())
    print(f"[{run_tag}]")
    print(f"  cell_set={CELL_SET}  → {len(cells_df)} cells across "
          f"{len(subjects)} subjects")
    print(f"  out_dir = {out_dir}")

    # ---- Stage 1: load + build payloads serially per subject -----------
    print(f"\n[1/2] Building cell payloads (sequential per subject)")
    t_build = time.time()
    tasks = _build_all_payloads(subjects, cells_df)
    print(f"  built {len(tasks)} cell payloads in {time.time() - t_build:.1f}s\n")

    # ---- Stage 2: dispatch compute per cell with progress reporting ---
    print(f"[2/2] Computing CV + {N_PERMUTATIONS} perms per cell  "
          f"(n_jobs={N_JOBS})")
    t0 = time.time()
    all_rows = []
    n_done = 0
    n_total = len(tasks)
    log_every = max(1, n_total // 25)   # ~25 progress prints over the run

    if N_JOBS in (None, 0, 1):
        for task in tasks:
            t_cell = time.time()
            all_rows.append(_run_one_task(task))
            n_done += 1
            if n_done % log_every == 0 or n_done == n_total:
                elapsed = time.time() - t0
                avg = elapsed / n_done
                eta = avg * (n_total - n_done)
                print(f"  [{n_done:4d}/{n_total}]  "
                      f"sub-{task['sub_str']} {task['neuron_id'][:30]:<30}  "
                      f"+{time.time() - t_cell:5.1f}s   "
                      f"elapsed {elapsed:6.1f}s   ETA {eta:6.1f}s")
    else:
        from joblib import Parallel, delayed
        # `return_as='generator_unordered'` streams results as workers
        # complete them, so we can print live progress. Falls back to
        # plain list collection on older joblib.
        # batch_size=1 streams results as workers complete each cell
        # (rather than waiting for joblib's auto-batched chunks).
        try:
            results_iter = Parallel(
                n_jobs=N_JOBS, backend="loky", verbose=0,
                batch_size=1, return_as="generator_unordered",
            )(delayed(_run_one_task)(t) for t in tasks)
        except TypeError:
            results_iter = Parallel(
                n_jobs=N_JOBS, backend="loky", verbose=5,
                batch_size=1,
            )(delayed(_run_one_task)(t) for t in tasks)
        for row in results_iter:
            all_rows.append(row)
            n_done += 1
            if n_done % log_every == 0 or n_done == n_total:
                elapsed = time.time() - t0
                avg = elapsed / n_done
                eta = avg * (n_total - n_done)
                print(f"  [{n_done:4d}/{n_total}]  "
                      f"elapsed {elapsed:6.1f}s  "
                      f"avg/cell {avg:5.2f}s  "
                      f"ETA {eta:6.1f}s")
    print(f"  total compute time: {time.time() - t0:.1f}s")

    per_cell = pd.DataFrame(all_rows)
    per_cell = _add_perm_p_columns(per_cell)
    per_cell.to_csv(out_dir / "per_cell.csv", index=False)
    print(f"  saved per_cell.csv ({len(per_cell)} rows)")

    return _stats_and_plots(per_cell, out_dir)


def _extract_per_cell_lag_curves(per_cell):
    """Parse `per_lag_r_all_lags_json` into a (n_cells, n_lags) matrix
    aligned with the current LAGS_DEG."""
    n_lags = len(LAGS_DEG)
    curves = np.full((len(per_cell), n_lags), np.nan)
    for i, row in enumerate(per_cell.itertuples(index=False)):
        try:
            v = json.loads(getattr(row, "per_lag_r_all_lags_json", "[]") or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        if len(v) == n_lags:
            curves[i] = [np.nan if x is None else float(x) for x in v]
    return curves


def per_roi_single_lag_stats(per_cell,
                              single_lags=SINGLE_LAGS_FOR_TESTS):
    """One-sample t-test of per-cell CV r > 0 at each individual lag,
    per ROI. BH-FDR within each lag family (across ROIs). Complements
    the predicted-lag `_per_roi_stats` — no lag sets, no averaging."""
    curves = _extract_per_cell_lag_curves(per_cell)
    rois_col = per_cell["roi"].to_numpy()
    rows = []
    valid_rois = sorted({r for r in rois_col if isinstance(r, str)})
    for roi in valid_rois:
        idx = np.where(rois_col == roi)[0]
        C = curves[idx]
        n_cells = len(idx)
        for lag in single_lags:
            li = LAGS_DEG.index(lag) if lag in LAGS_DEG else None
            if li is None:
                continue
            r = C[:, li]; r = r[np.isfinite(r)]
            if r.size >= 2:
                res = stats.ttest_1samp(r, 0.0, alternative="greater")
                t_, p_ = float(res.statistic), float(res.pvalue)
                m_ = float(r.mean())
            else:
                t_, p_, m_ = np.nan, np.nan, (float(r.mean()) if r.size else np.nan)
            rows.append({"roi": roi, "lag_deg": lag,
                         "n_cells_finite": int(r.size),
                         "n_cells_total": int(n_cells),
                         "mean_r": m_, "t_vs_0": t_, "p_unc": p_})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_fdr"] = np.nan
    for lag, idx in out.groupby("lag_deg").indices.items():
        idx = list(idx)
        out.loc[idx, "p_fdr"] = fsp.bh_fdr(
            out.loc[idx, "p_unc"].to_numpy(dtype=float))
    return out


def plot_per_lag_r_hist_all_rois(per_cell, single_lag_df,
                                   fixed_lag_deg, save_stem):
    """One figure per fixed lag: per-ROI histogram of CV r at that lag,
    annotated with the single-lag t vs 0 and its BH-FDR p."""
    _set_rc()
    curves = _extract_per_cell_lag_curves(per_cell)
    li = LAGS_DEG.index(fixed_lag_deg)
    r_lag = curves[:, li]
    rois = sorted(per_cell["roi"].dropna().unique().tolist())
    n = len(rois)
    n_cols = min(n, 4)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4.0 * CM * n_cols, 3.4 * CM * n_rows),
                              constrained_layout=True, squeeze=False)
    axes_flat = axes.ravel()
    for ax in axes_flat[n:]:
        ax.axis("off")
    all_r = r_lag[np.isfinite(r_lag)]
    if all_r.size:
        lo, hi = np.nanpercentile(all_r, [1, 99])
        bins = np.linspace(lo, hi, 22)
    else:
        bins = 20
    rois_col = per_cell["roi"].to_numpy()
    for ax, roi in zip(axes_flat, rois):
        m = rois_col == roi
        r = r_lag[m]; r = r[np.isfinite(r)]
        col = ROI_COLOURS.get(roi, "#888")
        if r.size:
            ax.hist(r, bins=bins, color=col, alpha=0.55,
                    edgecolor="black", linewidth=0.4)
            ax.axvline(r.mean(), color="black", lw=0.8)
        ax.axvline(0, color="gray", ls="--", lw=0.4)
        row = single_lag_df[
            (single_lag_df["roi"] == roi)
            & (single_lag_df["lag_deg"] == fixed_lag_deg)
        ]
        if not row.empty:
            rr = row.iloc[0]
            title = (f'{_disp(roi)}\n'
                     f't = {rr["t_vs_0"]:+.2f}  p = {rr["p_unc"]:.3g}\n'
                     f'p_FDR = {rr["p_fdr"]:.3g}   n = {rr["n_cells_finite"]}')
        else:
            title = _disp(roi)
        ax.set_title(title, fontsize=FONT_TICK)
        ax.tick_params(axis="both", labelsize=FONT_TICK, length=2, pad=1)
        ax.set_xlabel("CV r", fontsize=FONT_TICK)
        ax.set_ylabel("# cells", fontsize=FONT_TICK)
    fig.suptitle(
        f'Per-ROI CV r distribution at lag {fixed_lag_deg}°\n'
        f'(one-sample t vs 0, one-sided; FDR across {n} ROIs)',
        fontsize=FONT_BIG,
    )
    _save(fig, save_stem)


def _stats_and_plots(per_cell, out_dir, skip_raw_plots=False):
    """Recompute per-ROI stats, ROI×lag table and all figures from a
    pre-computed per_cell DataFrame. Shared by the full and reload paths.

    `skip_raw_plots=True` (default in reload mode) skips figures that need
    to re-load subject raw data (`plot_rate_map_examples`)."""
    # per-ROI stats
    roi_stats = _per_roi_stats(per_cell)
    roi_stats.to_csv(out_dir / "per_roi_stats.csv", index=False)
    print("  saved per_roi_stats.csv")
    print()
    print("  TEST 1  mean_r_vs_0       : one-sample t (1-sided > 0) on")
    print("                              per-cell CV r at predicted lag.")
    print("  TEST 2  target_vs_other   : paired t (1-sided > 0) on per-cell")
    print("                              (r_predicted_lag − mean r_other_lags).")
    print("  TEST 3  perm_sig_fraction : binomial test that fraction(p_perm < α)")
    print(f"                              exceeds α={ALPHA}.")
    print("  FDR scope = ROIs in this table (n_rois) for each test family.")
    print()
    print(roi_stats[[
        "roi", "n_cells", "analysis_kind",
        "test1_meanR_t",      "test1_meanR_p_unc",         "test1_meanR_p_fdr",
        "test2_targetVsOther_t",
        "test2_targetVsOther_p_unc", "test2_targetVsOther_p_fdr",
        "test3_permSig_k", "test3_permSig_n",
        "test3_permSig_p_unc",       "test3_permSig_p_fdr",
    ]].round(4).to_string(index=False))

    # per-(ROI, lag) table — written + FDR-corrected across the full table
    roi_x_lag = _per_roi_lag_table(per_cell)
    roi_x_lag.to_csv(out_dir / "per_roi_lag_table.csv", index=False)
    print("\n  saved per_roi_lag_table.csv (ROI × lag, FDR across full table)")

    # Lag-agnostic single-lag summary (one t-test per (ROI × lag))
    single_lag_df = per_roi_single_lag_stats(per_cell, SINGLE_LAGS_FOR_TESTS)
    single_lag_df.to_csv(out_dir / "per_roi_single_lag_stats.csv", index=False)
    print(f"  saved per_roi_single_lag_stats.csv "
          f"({single_lag_df.shape[0]} rows) for lags {SINGLE_LAGS_FOR_TESTS}")

    # plots
    plot_test1_mean_r_histograms(per_cell, roi_stats,
                                   str(out_dir / "test1_meanR_histograms"))
    plot_test2_target_vs_others_lines(per_cell, roi_stats,
                                        str(out_dir / "test2_targetVsOther_lines"))
    plot_test3_perm_sig_fraction_bars(per_cell, roi_stats,
                                        str(out_dir / "test3_permSig_bars"))
    plot_roi_x_lag_table_heatmap(roi_x_lag,
                                   str(out_dir / "roi_x_lag_meanR_heatmap"))
    plot_roi_x_lag_overview(per_cell, roi_stats,
                              str(out_dir / "roi_x_lag_tstat_overview"))
    plot_fixed_vs_free_comparison(per_cell,
                                    str(out_dir / "fixed_vs_free_comparison"))
    plot_zmni_gradient(per_cell, str(out_dir / "zMNI_gradient_ACC"))
    # Per-fixed-lag histograms across all ROIs.
    for fl in SINGLE_LAGS_FOR_TESTS:
        plot_per_lag_r_hist_all_rois(per_cell, single_lag_df, fl,
            str(out_dir / f"per_lag_r_hist_all_rois_lag{fl:03d}"))
    if skip_raw_plots:
        print("  skipped plot_rate_map_examples (raw subject data not "
              "re-loaded in reload mode).")
    else:
        plot_rate_map_examples(per_cell, out_dir)

    return per_cell, roi_stats, out_dir


if __name__ == "__main__":
    run()
