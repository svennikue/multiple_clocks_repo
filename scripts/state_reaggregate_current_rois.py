#!/usr/bin/env python
"""
Re-aggregate the cell-wise sustained-state results under the CURRENT ROI
labelling convention, without refitting any GLM.

Why this is valid without refitting
-----------------------------------
In `encoding_state_sustained_cv.py` the per-cell design matrix contains
state, phase, state x phase and per-configuration intercepts. ROI never
enters the design. Every per-cell statistic
(`min_phase_contrast`, `r_state`, `r_interaction`, and their permutation
p-values) is therefore invariant to the ROI label.

ROI enters only at the group level, in three places, all of which this
script recomputes from scratch:
  1. within-ROI BH-FDR on the per-cell permutation p-values
     (`add_fdr_columns`),
  2. the per-ROI summary rows and their binomial / t tests
     (`make_roi_summary`), which also runs the per-ROI Wilcoxon
     population-shift tests + BH-FDR across ROIs
     (`add_population_shift_tests`),
  3. the chi-squared omnibus and the planned EC-vs-pooled-rest Fisher
     test (`roi_omnibus_and_ec_tests`).

All three are imported from the original script so the numbers are
produced by exactly the same code that produced the published ones.

Reads   (read-only)
  <derivatives>/neurons_with_ROI_labels.csv        <- authoritative labels
  <derivatives>/group/encoding_state_sustained_cv/<run>/state_sustained_cv_results.csv

Writes  (only inside --out-dir, default: a new directory; never a
         pre-existing run directory)
  state_sustained_cv_results.csv        per-cell, with new roi + new q
  state_sustained_cv_roi_summary.csv    per-ROI summary
  roi_relabel_crosstab.csv              old ROI -> new ROI cell counts
  group_tests.json                      chi2 omnibus + EC-vs-rest Fisher
  reaggregate_config.json               provenance

Usage
  python state_reaggregate_current_rois.py \
      --source-run 2026-06-25_14-38-13_relabelled_2026-07-29_15-05-10
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC_SCRIPT = HERE / "encoding_state_sustained_cv.py"
DERIV = Path("/Users/xpsy1114/Documents/projects/multiple_clocks/data/"
             "ephys_humans/derivatives")
GROUP = DERIV / "group" / "encoding_state_sustained_cv"
LABEL_FILE = DERIV / "neurons_with_ROI_labels.csv"


def load_original_module():
    """Import encoding_state_sustained_cv.py without running its main()."""
    spec = importlib.util.spec_from_file_location("_ess_cv", SRC_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ess_cv"] = mod
    spec.loader.exec_module(mod)          # module is __main__-guarded
    return mod


def build_key(df, subj_col, cell_col):
    return (df[subj_col].astype(float).astype(int).astype(str) + "_"
            + df[cell_col].astype(float).astype(int).astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", required=True,
                    help="run directory under group/encoding_state_sustained_cv "
                         "supplying the per-cell fits")
    ap.add_argument("--roi-column", default="alt_final_roi")
    ap.add_argument("--label-file", default=str(LABEL_FILE))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--drop-unassigned", action="store_true", default=True,
                    help="drop cells with no ROI label (default: True)")
    ap.add_argument("--keep-unassigned", dest="drop_unassigned",
                    action="store_false")
    ap.add_argument("--min-cells", type=int, default=20,
                    help="ROIs with fewer cells than this are reported but "
                         "flagged as underpowered")
    args = ap.parse_args()

    src_dir = GROUP / args.source_run
    src_csv = src_dir / "state_sustained_cv_results.csv"
    if not src_csv.exists():
        sys.exit(f"no per-cell results at {src_csv}")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        GROUP / f"{args.source_run}_reaggregated_{stamp}")
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"refusing to write into non-empty existing dir {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = load_original_module()

    res = pd.read_csv(src_csv)
    lab = pd.read_csv(args.label_file)
    roi_col = args.roi_column
    if roi_col not in lab.columns:
        sys.exit(f"{roi_col} not in {args.label_file}")

    res["__key"] = build_key(res, "subject_int", "cell_idx")
    lab["__key"] = build_key(lab, "subject", "cell idx")
    dup = lab["__key"].duplicated().sum()
    if dup:
        print(f"[warn] {dup} duplicate subject/cell keys in label file; "
              f"keeping first")
        lab = lab.drop_duplicates("__key", keep="first")

    n_before = len(res)
    res["roi_source"] = res["roi"]
    res = res.drop(columns=["roi"]).merge(
        lab[["__key", roi_col]].rename(columns={roi_col: "roi"}),
        on="__key", how="left")
    matched = res["roi"].notna().sum()
    print(f"[info] {matched}/{n_before} cells matched a label; "
          f"{n_before - matched} unmatched/unassigned")

    crosstab = pd.crosstab(res["roi_source"],
                           res["roi"].fillna("UNASSIGNED"))
    crosstab.to_csv(out_dir / "roi_relabel_crosstab.csv")

    if args.drop_unassigned:
        res = res[res["roi"].notna()].copy()
    else:
        res["roi"] = res["roi"].fillna("UNASSIGNED")
    res["roi_changed"] = res["roi"] != res["roi_source"]
    print(f"[info] {int(res.roi_changed.sum())}/{len(res)} retained cells "
          f"changed ROI")

    # --- the three ROI-dependent group steps, from the original script ---
    res = mod.add_fdr_columns(res)
    summary = mod.make_roi_summary(res)

    group_tests = {}
    for sig_col, name in [("sig_sustained", "sustained"),
                          ("sig_r_state", "r_state"),
                          ("sig_r_interaction", "r_interaction")]:
        if sig_col in res.columns:
            group_tests[name] = mod.roi_omnibus_and_ec_tests(res, sig_col)

    small = summary.loc[summary.n_cells < args.min_cells, "roi"].tolist()
    if small:
        print(f"[warn] underpowered ROIs (n < {args.min_cells}): {small}")

    res.drop(columns="__key").to_csv(
        out_dir / "state_sustained_cv_results.csv", index=False)
    summary.to_csv(out_dir / "state_sustained_cv_roi_summary.csv", index=False)
    with open(out_dir / "group_tests.json", "w") as f:
        json.dump(group_tests, f, indent=2)
    with open(out_dir / "reaggregate_config.json", "w") as f:
        json.dump({
            "script": Path(__file__).name,
            "timestamp": stamp,
            "source_run": str(src_dir),
            "label_file": args.label_file,
            "roi_column_used": roi_col,
            "drop_unassigned": bool(args.drop_unassigned),
            "n_cells_source": int(n_before),
            "n_cells_retained": int(len(res)),
            "n_roi_changed": int(res.roi_changed.sum()),
            "rois": {r: int(n) for r, n in
                     zip(summary.roi, summary.n_cells)},
            "underpowered_rois": small,
            "note": "per-cell GLM statistics copied unchanged from source "
                    "run; ROI enters only via within-ROI FDR, the per-ROI "
                    "summary/Wilcoxon tests, and the chi2/Fisher group tests, "
                    "all recomputed here with the original functions",
        }, f, indent=2)

    cols = ["roi", "n_cells", "n_sustained", "frac_sustained",
            "mean_r_state", "mean_r_interaction",
            "wilcoxon_p_r_interaction", "q_wilcoxon_r_interaction",
            "fwe_wilcoxon_r_interaction",
            "wilcoxon_p_min_phase", "q_wilcoxon_min_phase",
            "binom_p_sustained"]
    have = [c for c in cols if c in summary.columns]
    print("\n" + summary[have].to_string(index=False))
    print("\ngroup tests:", json.dumps(group_tests, indent=2))
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
