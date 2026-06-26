#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSA-vs-non-RSA cell subset diagnostic for per_lag_encoding and
spatial_peaks_simple results.

Motivation
----------
`per_lag_encoding.py` runs on EVERY ACC cell in the ROI table (n≈159),
while `RSA_DSR_ROIs_simple.py` is restricted to cells from the 28
subjects with usable DSR-RSA grouping logs (n≈68). The 68 RSA-cells
are a strict subset of the 159 per_lag cells.

This means picking a "winning lag" from per_lag and testing it in RSA
is partially circular if the same cells drive both. To check this, we
split the per_lag and spatial_peaks results into:

  * 'in RSA'     — the cells that *also* enter the RSA analysis
  * 'NOT in RSA' — cells from subjects without RSA-eligible task data

If the per_lag / spatial_peaks signals survive in the 'NOT in RSA'
subset on their own, then using their lag finding to motivate the RSA
`dsr_fmri_informed` model (lags 1, 2) is statistically honest:
the prior comes from a disjoint cell pool, and the RSA test on the 68
RSA cells is independent confirmation.

What this script does
---------------------
* Loads the RSA grouping JSON to identify the 28 RSA-eligible subjects.
* Loads existing per_lag_encoding and spatial_peaks_simple per-cell
  output CSVs (no re-running of any heavy analysis).
* For each ROI of interest, runs one-sample t-tests against 0 on:
    – per_lag: every lag column individually, plus three windowed averages
      (lag_30 only, lag_30+60 'informed', lag_30+60+90 '123')
    – spatial_peaks: every shift column from the 12-shift consistency curve
  for three subsets: ALL, in RSA, NOT in RSA.
* Writes one CSV per analysis × ROI and an aggregate Markdown report.

Run
---
    python scripts/per_lag_peaks_RSA_subset_check.py

Outputs land in
    {DATA_DIR}/group/RSA_subset_diagnostic/{run_tag}/
        per_lag_subset_{ROI}.csv
        spatial_peaks_subset_{ROI}.csv
        report.md

Edit the constants at the top to point at different source runs.

@author: Svenja Kuchenhoff
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

# ── Input / output ────────────────────────────────────────────────────
DATA_DIR = Path("/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives")

PER_LAG_RUN_DIR = DATA_DIR / "group/per_lag_encoding/2026-06-18_22-37-01"
SP_PEAKS_CSV    = DATA_DIR / "group/spatial_peaks_simple/2026-06-18_15-13-34_full_optimal_lags_330_0_now/per_cell.csv"
RSA_SUBJECTS_JSON = DATA_DIR / "all_sessions_dsrRSA_grouping_summary.json"

OUT_BASE = DATA_DIR / "group/RSA_subset_diagnostic"

# Which ROIs to inspect. None = every ROI present in both source CSVs.
ROIS = ['ACC', 'medialOFC', 'PCC', 'Parahippocampal',
        'HC_anterior', 'HC_mid', 'EC']

# Per_lag windowed aggregates: name -> list of lag-step degrees to average.
# Each lag step is 30° (= one of the 12 phase×state subpaths in the 360-bin
# trajectory). Window 'informed' = lags 1+2 = the dsr_fmri_informed model.
PER_LAG_WINDOWS = {
    'lag_30_only':        [30],
    'lag_30+60_informed': [30, 60],
    'lag_30+60+90_123':   [30, 60, 90],
}

# Spatial-peaks shifts to score in the same 12-bin coordinate system.
SP_SHIFTS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]


# ── Helpers ───────────────────────────────────────────────────────────
def parse_sub_from_neuron(neuron_id):
    """neuron labels look like '02_15-15-chan120-ACC' -> '02'."""
    m = re.match(r'^(\d+)_', str(neuron_id))
    return m.group(1).zfill(2) if m else None


def load_rsa_subjects():
    with open(RSA_SUBJECTS_JSON) as f:
        return set(json.load(f).keys())


def one_sided_t(rs):
    """One-sided t (greater than 0) against H0: mean=0. Returns t, p, n, mean."""
    rs = np.asarray(rs, dtype=float)
    rs = rs[np.isfinite(rs)]
    n = len(rs)
    if n < 2:
        return np.nan, np.nan, n, np.nan
    t_stat, p_two = ttest_1samp(rs, 0)
    p_one = (p_two / 2) if t_stat > 0 else (1 - p_two / 2)
    return float(t_stat), float(p_one), int(n), float(rs.mean())


def stars(p):
    if not np.isfinite(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '·'
    return ''


# ── Per-lag analysis ──────────────────────────────────────────────────
def run_per_lag(rsa_subjects, out_dir):
    """For each ROI, t-test each lag column (and windowed aggregates) on
    ALL / in-RSA / NOT-in-RSA subsets. Saves one CSV per ROI plus a
    combined long-form CSV."""
    rows_all = []
    for roi in ROIS:
        csv_path = PER_LAG_RUN_DIR / f"per_lag_{roi}.csv"
        if not csv_path.exists():
            print(f"  [per_lag/{roi}] missing CSV, skipping: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df['subject'] = df['neuron'].apply(parse_sub_from_neuron)
        df['in_rsa'] = df['subject'].isin(rsa_subjects)

        lag_cols = [c for c in df.columns if c.startswith('lag_')]
        n_total = len(df); n_in = int(df.in_rsa.sum()); n_out = int((~df.in_rsa).sum())
        print(f"\n  per_lag/{roi}: total={n_total}, in_RSA={n_in}, not_in_RSA={n_out}")

        for metric_name, cols in (
            [(c, [c]) for c in lag_cols]
            + [(w, [f'lag_{l}' for l in lags]) for w, lags in PER_LAG_WINDOWS.items()]
        ):
            for label, mask in [('ALL', np.ones(len(df), dtype=bool)),
                                ('in_RSA', df.in_rsa.values),
                                ('NOT_in_RSA', (~df.in_rsa).values)]:
                vals = df.loc[mask, cols].mean(axis=1).to_numpy()
                t, p, n, m = one_sided_t(vals)
                rows_all.append({
                    'roi': roi, 'analysis': 'per_lag', 'metric': metric_name,
                    'subset': label, 'n_cells': n, 'mean': m,
                    't_stat': t, 'p_one_sided': p, 'stars': stars(p),
                })

    pl_df = pd.DataFrame(rows_all)
    pl_df.to_csv(out_dir / 'per_lag_subset_long.csv', index=False)
    print(f"\n  → {out_dir/'per_lag_subset_long.csv'}")
    return pl_df


# ── Spatial-peaks analysis ────────────────────────────────────────────
def parse_shift_curve(json_blob):
    """Parse a 'shift_curve_full_json' entry to a length-12 array (NaN
    where the entry was null)."""
    try:
        v = json.loads(json_blob) if isinstance(json_blob, str) else None
        if v is None or len(v) != len(SP_SHIFTS):
            return None
        return np.array([np.nan if x is None else float(x) for x in v])
    except Exception:
        return None


def run_spatial_peaks(rsa_subjects, out_dir):
    """Per-shift t-test of the spatial-peaks consistency curve in each
    ROI, split by RSA membership. Saves one long-form CSV."""
    sp = pd.read_csv(SP_PEAKS_CSV)
    sp['subject'] = sp['subject_id'].astype(str).str.zfill(2)
    sp['in_rsa'] = sp['subject'].isin(rsa_subjects)

    rows = []
    for roi in ROIS:
        df = sp[sp.roi == roi].copy()
        if df.empty:
            print(f"  [spatial_peaks/{roi}] no rows in source CSV, skipping")
            continue
        curves = df.shift_curve_full_json.apply(parse_shift_curve)
        M = np.vstack([
            c if c is not None else np.full(len(SP_SHIFTS), np.nan)
            for c in curves
        ])
        n_total = len(df); n_in = int(df.in_rsa.sum()); n_out = int((~df.in_rsa).sum())
        print(f"\n  spatial_peaks/{roi}: total={n_total}, in_RSA={n_in}, not_in_RSA={n_out}")

        masks = [
            ('ALL', np.ones(len(df), dtype=bool)),
            ('in_RSA', df.in_rsa.values),
            ('NOT_in_RSA', (~df.in_rsa).values),
        ]
        for i, s in enumerate(SP_SHIFTS):
            for label, mask in masks:
                vals = M[mask, i]
                t, p, n, m = one_sided_t(vals)
                rows.append({
                    'roi': roi, 'analysis': 'spatial_peaks',
                    'metric': f'shift_{s}',
                    'subset': label, 'n_cells': n, 'mean': m,
                    't_stat': t, 'p_one_sided': p, 'stars': stars(p),
                })

    sp_df = pd.DataFrame(rows)
    sp_df.to_csv(out_dir / 'spatial_peaks_subset_long.csv', index=False)
    print(f"\n  → {out_dir/'spatial_peaks_subset_long.csv'}")
    return sp_df


# ── Combined report ───────────────────────────────────────────────────
def write_report(pl_df, sp_df, rsa_subjects, out_dir):
    """Markdown summary highlighting the in-RSA vs NOT-in-RSA contrast
    for each ROI's headline lags / shifts."""
    lines = []
    lines.append(f"# RSA-vs-non-RSA cell subset diagnostic")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"RSA-eligible subjects (n={len(rsa_subjects)}): "
                 f"{', '.join(sorted(rsa_subjects))}")
    lines.append("")
    lines.append("Source runs:")
    lines.append(f"- per_lag_encoding: `{PER_LAG_RUN_DIR}`")
    lines.append(f"- spatial_peaks_simple: `{SP_PEAKS_CSV}`")
    lines.append("")

    for roi in ROIS:
        lines.append(f"## {roi}")
        # per_lag windows
        pl_sub = pl_df[(pl_df.roi == roi) & (pl_df.metric.isin(PER_LAG_WINDOWS))]
        if not pl_sub.empty:
            lines.append("")
            lines.append("### per_lag (windowed)")
            lines.append("| window | subset | n | mean | t | p (one-sided) | sig |")
            lines.append("|---|---|---|---|---|---|---|")
            for w in PER_LAG_WINDOWS:
                for label in ('ALL', 'in_RSA', 'NOT_in_RSA'):
                    r = pl_sub[(pl_sub.metric == w) & (pl_sub.subset == label)]
                    if r.empty: continue
                    r = r.iloc[0]
                    lines.append(
                        f"| {w} | {label} | {int(r.n_cells)} | "
                        f"{r['mean']:+.4f} | {r.t_stat:+.2f} | "
                        f"{r.p_one_sided:.3f} | {r.stars} |"
                    )
            lines.append("")

        # spatial peaks — show all 12 shifts
        sp_sub = sp_df[sp_df.roi == roi]
        if not sp_sub.empty:
            lines.append("### spatial_peaks (per shift)")
            lines.append("| shift | subset | n | mean | t | p (one-sided) | sig |")
            lines.append("|---|---|---|---|---|---|---|")
            for s in SP_SHIFTS:
                metric = f'shift_{s}'
                for label in ('ALL', 'in_RSA', 'NOT_in_RSA'):
                    r = sp_sub[(sp_sub.metric == metric) & (sp_sub.subset == label)]
                    if r.empty: continue
                    r = r.iloc[0]
                    lines.append(
                        f"| {s}° | {label} | {int(r.n_cells)} | "
                        f"{r['mean']:+.4f} | {r.t_stat:+.2f} | "
                        f"{r.p_one_sided:.3f} | {r.stars} |"
                    )
            lines.append("")

    out_dir.joinpath("report.md").write_text("\n".join(lines))
    print(f"\n  → {out_dir / 'report.md'}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    run_tag = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = OUT_BASE / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    rsa_subjects = load_rsa_subjects()
    print(f"RSA-eligible subjects: {len(rsa_subjects)}")

    pl_df = run_per_lag(rsa_subjects, out_dir)
    sp_df = run_spatial_peaks(rsa_subjects, out_dir)

    write_report(pl_df, sp_df, rsa_subjects, out_dir)

    # also save a config snapshot for reproducibility
    cfg = {
        'run_tag': run_tag,
        'rsa_subjects_json': str(RSA_SUBJECTS_JSON),
        'per_lag_run_dir':   str(PER_LAG_RUN_DIR),
        'sp_peaks_csv':      str(SP_PEAKS_CSV),
        'rois':              ROIS,
        'per_lag_windows':   PER_LAG_WINDOWS,
        'sp_shifts':         SP_SHIFTS,
        'n_rsa_subjects':    len(rsa_subjects),
        'rsa_subjects':      sorted(rsa_subjects),
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"  → {out_dir / 'config.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
