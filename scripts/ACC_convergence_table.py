#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACC DSR convergence table.

Pulls per-cell numbers from the three single-cell analyses (per_lag_encoding,
spatial_peaks_simple, encoding_analysis_simple) and the RSA run, restricted
to ACC, and reports the joint statistic (one-sided t-test as primary,
binomial fraction-of-significant-cells as supportive) for each.

Future window definitions:
  - W1 = {30}        ('next')
  - W2 = {30, 60}    ('next, next+1')
  - W3 = {30, 60, 90}('next, next+1, next+2', user hypothesis)

Inputs are hard-coded to the latest runs (edit the constants below).

@author: Svenja Kuchenhoff
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, binomtest

# ── Run pointers ──────────────────────────────────────────────────────
DATA = Path("/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives")
PER_LAG_CSV = DATA / "group/per_lag_encoding/2026-06-18_22-37-01/per_lag_ACC.csv"
SP_PEAKS_CSV = DATA / "group/spatial_peaks_simple/2026-06-18_15-13-34_full_optimal_lags_330_0_now/per_cell.csv"
ENC_CSV = DATA / "group/encoding_analysis_simple/2026-06-18_22-07-42/encoding_results.csv"
RSA_FDR_CSV = DATA / "group/DSR_RSA_simple_ROI/2026-06-20_08-58-59/confirmatory_fdr.csv"
RSA_COMBO_CSV = DATA / "group/DSR_RSA_simple_ROI/2026-06-20_08-58-59/results_summary_combos.csv"

OUT_DIR = DATA / "group/ACC_convergence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROI = "ACC"
WINDOWS = {
    "W1_next":          [30],
    "W2_next_plus1":    [30, 60],
    "W3_next_plus2":    [30, 60, 90],
}
ALPHA = 0.05


def fmt_p(p):
    if not np.isfinite(p): return "  n/a"
    if p < 0.001: return f"{p:.3g}***"
    if p < 0.01:  return f"{p:.3f}**"
    if p < 0.05:  return f"{p:.3f}*"
    if p < 0.10:  return f"{p:.3f}·"
    return f"{p:.3f}"


def binom_one_sided(k, n, p_chance=ALPHA):
    if n == 0:
        return np.nan
    try:
        return binomtest(int(k), int(n), p=p_chance, alternative="greater").pvalue
    except Exception:
        return np.nan


# ── 1) per_lag_encoding ────────────────────────────────────────────────
pl = pd.read_csv(PER_LAG_CSV)
print(f"\n[1/4] per_lag_encoding  →  {len(pl)} ACC cells")
lag_cols = [c for c in pl.columns if c.startswith("lag_")]
lag_of = {int(c.split("_")[1]): c for c in lag_cols}

rows = []
for wname, lags in WINDOWS.items():
    cols = [lag_of[l] for l in lags if l in lag_of]
    # per-cell mean r in the window
    score = pl[cols].mean(axis=1).dropna().to_numpy()
    # one-sided t against 0
    t, p_two = ttest_1samp(score, 0)
    p_one = (p_two / 2) if t > 0 else (1 - p_two / 2)
    # supportive: per-cell ≈ no cell-level p_perm available in this CSV
    # use a leave-out fallback: a cell counts as 'window-positive' if its
    # mean r in the window > 0 AND its max-window-lag r > max-out-of-window r.
    out_cols = [c for c in lag_cols if c not in cols]
    in_max  = pl[cols].max(axis=1).to_numpy()
    out_max = pl[out_cols].max(axis=1).to_numpy()
    pos = (score > 0)
    won = (in_max > out_max)
    k = int(((pos) & (won)).sum())
    n = int(np.isfinite(score).sum())
    p_binom = binom_one_sided(k, n, p_chance=len(cols)/len(lag_cols)/2)  # chance ~ window_frac/2
    rows.append({
        "script": "per_lag_encoding",
        "window": wname,
        "n_cells": n,
        "mean_score": float(np.mean(score)),
        "t_stat": float(t),
        "t_p_one_sided": float(p_one),
        "k_window_winners": k,
        "frac_winners": k / n if n else np.nan,
        "binom_p": float(p_binom) if np.isfinite(p_binom) else np.nan,
        "binom_chance": len(cols)/len(lag_cols)/2,
        "metric_note": "per-cell mean r across window lags; binom: # cells where max-r in window > max-r out of window AND mean>0",
    })

# ── 2) spatial_peaks_simple ────────────────────────────────────────────
sp = pd.read_csv(SP_PEAKS_CSV)
sp_acc = sp[sp.roi == ROI].copy()
print(f"[2/4] spatial_peaks    →  {len(sp_acc)} ACC cells")
shifts = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

def per_shift_arr(curve_json):
    try:
        v = json.loads(curve_json) if isinstance(curve_json, str) else None
        if v is None or len(v) != len(shifts): return None
        return np.array([np.nan if x is None else float(x) for x in v])
    except Exception:
        return None

curves = sp_acc.shift_curve_full_json.apply(per_shift_arr)
M = np.vstack([c if c is not None else np.full(len(shifts), np.nan) for c in curves])  # (n_cells, 12)

for wname, lags in WINDOWS.items():
    idx_in  = [shifts.index(l) for l in lags]
    idx_out = [i for i, s in enumerate(shifts) if s not in lags]
    score = np.nanmean(M[:, idx_in], axis=1)
    score = score[np.isfinite(score)]
    t, p_two = ttest_1samp(score, 0)
    p_one = (p_two / 2) if t > 0 else (1 - p_two / 2)
    # supportive: peak_shift_plurality in window
    psp = sp_acc["peak_shift_plurality"].dropna().astype(int)
    k = int(psp.isin(lags).sum())
    n = int(len(psp))
    p_binom = binom_one_sided(k, n, p_chance=len(lags)/len(shifts))
    rows.append({
        "script": "spatial_peaks",
        "window": wname,
        "n_cells": len(score),
        "mean_score": float(np.mean(score)),
        "t_stat": float(t),
        "t_p_one_sided": float(p_one),
        "k_window_winners": k,
        "frac_winners": k / n if n else np.nan,
        "binom_p": float(p_binom) if np.isfinite(p_binom) else np.nan,
        "binom_chance": len(lags)/len(shifts),
        "metric_note": "per-cell mean r across window shifts; binom: cells whose training-peak shift fell in window vs chance n_win/12",
    })

# ── 3) encoding_analysis_simple ─────────────────────────────────────────
enc = pd.read_csv(ENC_CSV)
enc_acc = enc[enc.roi == ROI].copy()
print(f"[3/4] encoding_analysis → {enc_acc.neuron.nunique()} ACC cells")
# This run did NOT include dsr_only_fut / dsr_now_next as singles, so we can
# only report the legacy single-DSR test as a proxy. Mark explicitly.
for m in ["dsr", "state", "location", "midnight"]:
    sub = enc_acc[enc_acc.model == m]
    rs = sub.mean_r.dropna().to_numpy()
    ps = sub.p_perm.dropna().to_numpy()
    t, p_two = ttest_1samp(rs, 0)
    p_one = (p_two / 2) if t > 0 else (1 - p_two / 2)
    k = int((ps < ALPHA).sum())
    n = int(len(ps))
    rows.append({
        "script": "encoding_analysis (legacy single)",
        "window": f"model={m}",
        "n_cells": len(rs),
        "mean_score": float(np.mean(rs)),
        "t_stat": float(t),
        "t_p_one_sided": float(p_one),
        "k_window_winners": k,
        "frac_winners": k / n if n else np.nan,
        "binom_p": float(binom_one_sided(k, n, p_chance=ALPHA)) if np.isfinite(binom_one_sided(k, n)) else np.nan,
        "binom_chance": ALPHA,
        "metric_note": f"per-cell CV r for {m} (full design, no future-only restriction; deferred task #45)",
    })

# ── 4) RSA  ────────────────────────────────────────────────────────────
rsa_fdr = pd.read_csv(RSA_FDR_CSV)
rsa_combo = pd.read_csv(RSA_COMBO_CSV)
# pull ACC dsr_fmri inside the FDR-combo
acc_combo = rsa_combo[(rsa_combo.roi == ROI) &
                      (rsa_combo.model == "dsr_fmri") &
                      (rsa_combo.combo == "bttn_loc_l2_state_midn") &
                      (rsa_combo.test == "split_halves_z")]
acc_fdr = rsa_fdr[rsa_fdr.roi == ROI]
beta = float(acc_combo["beta"].iloc[0]) if len(acc_combo) else np.nan
pperm = float(acc_combo["p_perm"].iloc[0]) if len(acc_combo) else np.nan
q_bh = float(acc_fdr["p_fdr"].iloc[0]) if "p_fdr" in acc_fdr.columns and len(acc_fdr) else np.nan
n_subs = int(acc_combo["n"].iloc[0]) if "n" in acc_combo.columns and len(acc_combo) else np.nan
print(f"[4/4] RSA              →  {n_subs} ACC subjects (group-level)")
rows.append({
    "script": "RSA_DSR_ROIs_simple",
    "window": "dsr_fmri | bttn_loc_l2_state_midn",
    "n_cells": n_subs,                  # subjects, not cells
    "mean_score": beta,
    "t_stat": np.nan,
    "t_p_one_sided": pperm,
    "k_window_winners": np.nan,
    "frac_winners": np.nan,
    "binom_p": np.nan,
    "binom_chance": np.nan,
    "metric_note": f"group β (split_halves_z, residualised), p_perm from circ-shift null. BH-FDR q across 7 ROIs = {q_bh:.3f}",
})

# ── Write table ─────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
out_csv = OUT_DIR / "ACC_convergence_table.csv"
df.to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}")

# Pretty printout
print("\n" + "="*100)
print(f"ACC DSR CONVERGENCE  (ROI={ROI}, α={ALPHA})")
print("="*100)
for script, grp in df.groupby("script", sort=False):
    print(f"\n── {script}")
    for _, r in grp.iterrows():
        line = (f"  {r['window']:<32s}  n={int(r['n_cells']) if np.isfinite(r['n_cells']) else '?':>4}"
                f"  mean={r['mean_score']:+.4f}")
        if np.isfinite(r["t_stat"]):
            line += f"  t={r['t_stat']:+.2f}  p₁={fmt_p(r['t_p_one_sided'])}"
        else:
            line += f"  p_perm={fmt_p(r['t_p_one_sided'])}      "
        if np.isfinite(r["binom_p"]):
            line += f"  k/n={int(r['k_window_winners'])}/{int(r['n_cells'])} ({100*r['frac_winners']:.1f}%, chance {100*r['binom_chance']:.1f}%)  binom p={fmt_p(r['binom_p'])}"
        print(line)
print("\n  primary  = one-sided t-test of per-cell score (or group β for RSA)")
print("  support  = binomial test of per-cell winners vs chance")
print("  W1 = {30°}      'next'")
print("  W2 = {30°,60°}  'next, next+1'")
print("  W3 = {30°,60°,90°}  'next, next+1, next+2'  (user hypothesis)\n")
