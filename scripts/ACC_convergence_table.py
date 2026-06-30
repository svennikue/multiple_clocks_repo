#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACC DSR convergence table — phase-residualised runs (2026-06-28 snapshot).

Pulls per-cell numbers from the three single-cell analyses (per_lag_encoding
rate-map version, spatial_peaks_simple fixed-lag, encoding_analysis_simple
legacy) and the RSA run, restricted to ACC, and reports the joint statistic
(one-sided t-test as primary, binomial fraction-of-significant-cells as
supportive) for each.

Future window definitions:
  - W1 = {30}         ('next')
  - W2 = {30, 60}     ('next, next+1')
  - W3 = {30, 60, 90} ('next, next+1, next+2', user hypothesis)

═══════════════════════════════════════════════════════════════════════════
CONVERGENCE OBSERVATIONS — as of 2026-06-28
═══════════════════════════════════════════════════════════════════════════
All analyses below use phase-residualised data (cosine basis) at the per-cell
level and the dsr_fmri family of models (no phase regressor). The earlier
"phase-included" DSR family (dsr_old) survived FDR in ACC; the present
phase-stripped runs are weaker and motivate the convergence-table approach.

ACC findings (synthesis across analyses):
  ✓ FDR-significant in:
      - spatial_peaks_simple fixed-lag (30°/60°):  t=2.67, q_FDR=0.026
      - spatial_peaks_simple T2 within-cell paired: t=2.82, q_FDR=0.008
      - spatial_peaks_simple T3 perm-sig frac:      10.7%, q_FDR=0.019
      - per_lag_encoding rate-map noctrl lag 30°:   t=2.41, q_FDR=0.030
  ~ Trend (uncorrected significant, FDR not reached):
      - per_lag_encoding T2 within-cell paired noctrl: t=3.04, p=0.001
      - per_lag_encoding dsr_inf noctrl:                t=1.65, p=0.051
      - RSA combo ctrl_dsrFULL  dsr_fmri:               β=0.040, q_per_combo=0.072
      - RSA combo ctrl_dsrFUT   dsr_fmri_fut:           β=0.038, q_per_combo=0.072
      - RSA combo fdr_dsrInformed dsr_fmri_informed:    β=0.030, q_per_combo=0.093
  ✗ Collapses under strict (bin-level) controls:
      - per_lag_encoding rate-map WITH-ctrl lag 30°/60°: t=0.39 / 0.10
        — interpreted as ACC future-action variance being SHARED with
        current location + current button + next button, not as
        spurious (multiple weaker tests still positive).

HC_anterior:
  ✓ Strongest cleanly-replicating effect of any ROI.
    - spatial_peaks fixed-lag 0°:        t=2.30, q_FDR=0.026
    - per_lag_encoding noctrl lag 90°:   t=3.41, q_FDR=0.003
    - per_lag_encoding WITH-ctrl lag 0°: t=2.96, q_FDR=0.012 (STRENGTHENS
                                          with controls — clean place code)
    - RSA fdr_dsrInformed:               β=0.060, q_per_combo=0.014

HC_mid:
  ✓ Current-location code (lag 0°) FDR-sig in noctrl: t=2.62, q=0.033
  ~ With ctrl drops to trend: t=1.98, q=0.086
  ? lag 30° flips strongly negative under controls (t=-3.60) — likely a
    suppression / over-fitting artefact; flag in supplement.
  - RSA ctrl_dsrFULL: β=0.072, q_per_combo=0.014 ✓

Parahippocampal:
  ? Surprise positive at lag 30° (noctrl t=3.47, q=0.003; WITH-ctrl t=1.49, n.s.)
  - Not significant in spatial_peaks (uses free-peak, no a-priori lag).
  - Not significant in RSA (q_per_combo=0.32).
  → Treat as exploratory; suggests PHPC may carry early-future spatial
    structure but doesn't replicate across analyses.

medialOFC, PCC, EC: no consistent signal in any analysis at this stage.
EC is underpowered (n=28).

KEY METHODOLOGICAL CAVEAT (PHASE):
  The original DSR family (dsr_old) included a phase-cosine regressor and
  survived FDR in ACC for both RSA and encoding. Once phase is removed at
  the data level (cosine basis residualisation) AND the DSR model drops the
  phase column, the ACC effect drops to trend. The fMRI RSA does NOT
  partial out phase between phase-bins (compares reward-with-reward and
  path-with-path within phase), which is mechanistically equivalent to
  phase-masking — i.e. the fMRI test was already phase-independent by
  construction. The cell-level RSA chose data-level residualisation
  instead; doing BOTH (residualisation + phase-masking) would double-remove
  phase variance and is not recommended.

═══════════════════════════════════════════════════════════════════════════
@author: Svenja Kuchenhoff
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, binomtest

# ── Run pointers (latest, phase-residualised) ─────────────────────────
DATA = Path("/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives")

# Per-lag encoding — rate-map version, phase-residualised.
PER_LAG_RUN = "2026-06-28_15-30-46"
PER_LAG_CSV = DATA / f"group/per_lag_encoding/{PER_LAG_RUN}/per_cell_ACC.csv"

# Spatial peaks — phase-resid, paired grid groups, fixed-lag for ACC.
SP_PEAKS_RUN = "2026-06-26_18-47-11_phase_resid_paired_fixedlag"
SP_PEAKS_CSV = DATA / f"group/spatial_peaks_simple/{SP_PEAKS_RUN}/per_cell.csv"

# Legacy elasticnet encoding (NOT re-run with phase residualisation).
ENC_CSV = DATA / "group/encoding_analysis_simple/2026-06-18_22-07-42/encoding_results.csv"

# RSA — uses dsr_fmri family, phase residualised, with per-combo FDR.
RSA_RUN = "2026-06-22_16-17-15-final-DSR"
RSA_COMBO_CSV     = DATA / f"group/DSR_RSA_simple_ROI/{RSA_RUN}/results_summary_combos.csv"
RSA_PERCOMBO_CSV  = DATA / f"group/DSR_RSA_simple_ROI/{RSA_RUN}/confirmatory_fdr_per_combo.csv"

OUT_DIR = DATA / "group/ACC_convergence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROI = "ACC"
WINDOWS = {
    "W1_next":          [30],
    "W2_next_plus1":    [30, 60],
    "W3_next_plus2":    [30, 60, 90],
}
ALPHA = 0.05
LAGS_DEG = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]


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


rows = []


# ── 1) per_lag_encoding — rate-map (noctrl AND ctrl) ──────────────────
pl = pd.read_csv(PER_LAG_CSV)
print(f"\n[1/4] per_lag_encoding rate-map  →  {len(pl)} ACC cells  (run {PER_LAG_RUN})")

def _per_lag_score(pl_df, lags, ctrl_tag):
    """Per-cell mean CV r in `lags`, plus per-cell perm-p in `lags`."""
    r_cols = [f'r_lag{l:03d}_{ctrl_tag}' for l in lags]
    p_cols = [f'p_lag{l:03d}_{ctrl_tag}' for l in lags]
    r_in   = pl_df[r_cols].mean(axis=1).to_numpy(dtype=float)
    p_in   = pl_df[p_cols].min(axis=1).to_numpy(dtype=float)  # window-min p
    return r_in, p_in


for ctrl_tag in ('noctrl', 'ctrl'):
    for wname, lags in WINDOWS.items():
        r_in, p_in = _per_lag_score(pl, lags, ctrl_tag)
        score = r_in[np.isfinite(r_in)]
        t, p_two = ttest_1samp(score, 0)
        p_one = (p_two / 2) if t > 0 else (1 - p_two / 2)
        # Supportive: how many cells have at least one window-lag at p_perm < α
        n = int(np.isfinite(r_in).sum())
        k = int(np.sum(np.isfinite(p_in) & (p_in < ALPHA)))
        # Chance for "any-of-len(lags) perm-p < α" is bounded by len(lags)*α
        chance = min(1.0, len(lags) * ALPHA)
        p_binom = binom_one_sided(k, n, p_chance=chance)
        rows.append({
            "script":       f"per_lag_encoding (rate-map, {ctrl_tag})",
            "window":       wname,
            "n_cells":      n,
            "mean_score":   float(np.mean(score)),
            "t_stat":       float(t),
            "t_p_one_sided": float(p_one),
            "k_window_winners": k,
            "frac_winners": k / n if n else np.nan,
            "binom_p":      float(p_binom) if np.isfinite(p_binom) else np.nan,
            "binom_chance": chance,
            "metric_note":  (f"per-cell mean rate-map r across {lags}°; "
                              f"binom = cells with ≥1 window-lag at p_perm<α "
                              f"vs chance {chance:.2f}"),
        })


# ── 2) spatial_peaks_simple — phase-resid, fixed-lag for ACC ──────────
sp = pd.read_csv(SP_PEAKS_CSV)
sp_acc = sp[sp.roi == ROI].copy()
print(f"[2/4] spatial_peaks (phase-resid)  →  {len(sp_acc)} ACC cells  (run {SP_PEAKS_RUN})")


def _sp_per_lag_curve(curve_json):
    try:
        v = json.loads(curve_json) if isinstance(curve_json, str) else None
        if v is None or len(v) != len(LAGS_DEG): return None
        return np.array([np.nan if x is None else float(x) for x in v])
    except Exception:
        return None


curves = sp_acc['per_lag_r_all_lags_json'].apply(_sp_per_lag_curve)
M = np.vstack([c if c is not None else np.full(len(LAGS_DEG), np.nan)
               for c in curves])  # (n_cells, 12)

for wname, lags in WINDOWS.items():
    idx_in = [LAGS_DEG.index(l) for l in lags]
    score = np.nanmean(M[:, idx_in], axis=1)
    score = score[np.isfinite(score)]
    t, p_two = ttest_1samp(score, 0)
    p_one = (p_two / 2) if t > 0 else (1 - p_two / 2)
    # Supportive: how many cells have fixed-lag perm-p < α (only available
    # for ACC where fixed-lag analysis ran with target_lags=[30, 60])
    p_fixed = sp_acc['p_perm_fixed'].dropna().to_numpy(dtype=float)
    k = int(np.sum(p_fixed < ALPHA))
    n = int(len(p_fixed))
    p_binom = binom_one_sided(k, n, p_chance=ALPHA)
    rows.append({
        "script":       f"spatial_peaks (phase-resid, paired groups)",
        "window":       wname,
        "n_cells":      len(score),
        "mean_score":   float(np.mean(score)),
        "t_stat":       float(t),
        "t_p_one_sided": float(p_one),
        "k_window_winners": k,
        "frac_winners": k / n if n else np.nan,
        "binom_p":      float(p_binom) if np.isfinite(p_binom) else np.nan,
        "binom_chance": ALPHA,
        "metric_note":  (f"per-cell mean rate-map r across {lags}°; binom "
                          f"on fixed-lag perm-p (target=[30,60]) vs α"),
    })


# ── 3) encoding_analysis_simple — LEGACY (not re-run, phase-INcluded) ──
try:
    enc = pd.read_csv(ENC_CSV)
    enc_acc = enc[enc.roi == ROI].copy()
    print(f"[3/4] encoding_analysis (legacy)  →  {enc_acc.neuron.nunique()} "
          f"ACC cells  (NOT phase-residualised — for comparison only)")
    for m in ["dsr", "state", "location", "midnight"]:
        sub = enc_acc[enc_acc.model == m]
        rs = sub.mean_r.dropna().to_numpy()
        ps = sub.p_perm.dropna().to_numpy()
        if rs.size < 2:
            continue
        t, p_two = ttest_1samp(rs, 0)
        p_one = (p_two / 2) if t > 0 else (1 - p_two / 2)
        k = int((ps < ALPHA).sum())
        n = int(len(ps))
        rows.append({
            "script":       "encoding_analysis (legacy, no phase-resid)",
            "window":       f"model={m}",
            "n_cells":      len(rs),
            "mean_score":   float(np.mean(rs)),
            "t_stat":       float(t),
            "t_p_one_sided": float(p_one),
            "k_window_winners": k,
            "frac_winners": k / n if n else np.nan,
            "binom_p":      float(binom_one_sided(k, n, p_chance=ALPHA)),
            "binom_chance": ALPHA,
            "metric_note":  (f"per-cell CV r for {m} (phase-INCLUDED, "
                              f"original ElasticNet pipeline)"),
        })
except FileNotFoundError:
    print(f"[3/4] encoding_analysis: file missing, skipping")


# ── 4) RSA — per-combo FDR across 7 ROIs ──────────────────────────────
rsa_combo = pd.read_csv(RSA_COMBO_CSV)
try:
    rsa_perc = pd.read_csv(RSA_PERCOMBO_CSV)
except FileNotFoundError:
    rsa_perc = pd.DataFrame()

print(f"[4/4] RSA  (run {RSA_RUN})  →  per-combo BH-FDR across 7 ROIs")

for combo, submodel in [
    ('ctrl_dsrFULL',    'dsr_fmri'),
    ('ctrl_dsrFUT',     'dsr_fmri_fut'),
    ('fdr_dsrInformed', 'dsr_fmri_informed'),
]:
    sub = rsa_combo[(rsa_combo.roi == ROI)
                     & (rsa_combo.combo == combo)
                     & (rsa_combo.sub_model == submodel)
                     & (rsa_combo.test == 'split_halves_z')]
    if sub.empty:
        rows.append({
            "script":       "RSA_DSR_ROIs_simple",
            "window":       f"{combo} | {submodel}",
            "n_cells":      np.nan,
            "mean_score":   np.nan,
            "t_stat":       np.nan,
            "t_p_one_sided": np.nan,
            "k_window_winners": np.nan,
            "frac_winners": np.nan,
            "binom_p":      np.nan,
            "binom_chance": np.nan,
            "metric_note":  f"no rows in summary_combo for combo={combo} sub={submodel}",
        })
        continue
    row = sub.iloc[0]
    q_per_combo = np.nan
    if not rsa_perc.empty:
        pc = rsa_perc[(rsa_perc.roi == ROI) & (rsa_perc.combo == combo)
                       & (rsa_perc.sub_model == submodel)]
        if not pc.empty:
            q_per_combo = float(pc.iloc[0]['q_fdr'])
    rows.append({
        "script":       "RSA_DSR_ROIs_simple",
        "window":       f"{combo} | {submodel}",
        "n_cells":      int(row['n_neurons']),
        "mean_score":   float(row['beta']),
        "t_stat":       float(row['t']),
        "t_p_one_sided": float(row['p_perm']),
        "k_window_winners": np.nan,
        "frac_winners": np.nan,
        "binom_p":      np.nan,
        "binom_chance": np.nan,
        "metric_note":  (f"group β (split_halves_z), p_perm from circ-shift "
                          f"null. Per-combo BH-FDR across 7 ROIs: q={q_per_combo:.3f}"),
    })


# ── Write table ─────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
out_csv = OUT_DIR / "ACC_convergence_table.csv"
df.to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}")

# Pretty printout
print("\n" + "=" * 110)
print(f"ACC DSR CONVERGENCE  (ROI={ROI}, α={ALPHA})  —  phase-residualised runs")
print("=" * 110)
for script, grp in df.groupby("script", sort=False):
    print(f"\n── {script}")
    for _, r in grp.iterrows():
        n = int(r['n_cells']) if np.isfinite(r['n_cells']) else None
        n_str = f"{n:>4d}" if n is not None else "  ? "
        line = f"  {r['window']:<36s}  n={n_str}  mean={r['mean_score']:+.4f}"
        if np.isfinite(r["t_stat"]):
            line += f"  t={r['t_stat']:+.2f}  p₁={fmt_p(r['t_p_one_sided'])}"
        else:
            line += f"  p_perm={fmt_p(r['t_p_one_sided'])}      "
        if np.isfinite(r["binom_p"]):
            kw = int(r['k_window_winners'])
            line += (f"  k/n={kw}/{n_str.strip()} "
                     f"({100*r['frac_winners']:.1f}%, chance "
                     f"{100*r['binom_chance']:.1f}%)  "
                     f"binom p={fmt_p(r['binom_p'])}")
        print(line)
print("\n  primary  = one-sided t-test of per-cell score (or group β for RSA)")
print("  support  = binomial test of per-cell winners vs chance")
print("  W1 = {30°}              'next'")
print("  W2 = {30°, 60°}         'next, next+1'")
print("  W3 = {30°, 60°, 90°}    'next, next+1, next+2'  (user hypothesis)\n")
