#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-fix ROI-labelling overview: one glassbrain per ROI + one summary
glassbrain with all included cells colour-coded. Also emits a companion
CSV listing every excluded ('leftover' or subject-filter NaN) cell so the
methods paragraph can cite exact numbers.

Data source: `neurons_with_final_roi_labels.csv` — produced by
`scripts/cell_to_roi_MNI.py`. Uses MNI152 coordinates.

Never labels a cell as "ACC" in figure text — remapped to "mPFC".
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting

import sys
sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")
from mc.plotting.cell_results import (
    SHOWGIRL2_DISCRETE, ROI_COLORS_SHOWGIRL2, ROI_DISPLAY_NAMES,
    _EXTRA_ROI_COLORS,
)

DEFAULT_ROI_TABLE = ("/Users/xpsy1114/Documents/projects/multiple_clocks/"
                     "data/ephys_humans/derivatives/"
                     "neurons_with_ROI_labels.csv")

# Colours for included ROIs (canonical CLAUDE.md mapping + extras)
def _col(roi):
    idx = ROI_COLORS_SHOWGIRL2.get(roi)
    if idx is not None:
        return SHOWGIRL2_DISCRETE[idx]
    if roi in _EXTRA_ROI_COLORS:
        return _EXTRA_ROI_COLORS[roi]
    return "#666"


def _disp(roi):
    return ROI_DISPLAY_NAMES.get(roi, roi)


ROIS_ORDER = ["EC", "mOFC", "mPFC", "HC_anterior", "HC_mid",
                "PHC", "PCC"]


def draw(roi_table: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(roi_table)
    coords = ["MNI_x", "MNI_y", "MNI_z"]

    # ---- Panel A: single glassbrain, all INCLUDED cells coloured by ROI ---
    display = plotting.plot_glass_brain(
        None, display_mode="ortho", plot_abs=False,
        title=("Cell coverage — all cells with a valid alt_final_roi "
                "(single-marker per unique MNI, dot colour = ROI)"),
        black_bg=False, alpha=0.4,
    )
    for roi in ROIS_ORDER:
        sub = t[t["alt_final_roi"] == roi]
        if sub.empty:
            continue
        xyz = sub[coords].to_numpy(float)
        display.add_markers(xyz, marker_color=_col(roi), marker_size=15,
                             alpha=0.85, edgecolors="black")
    out_a = out_dir / "included_cells_by_roi.pdf"
    display.savefig(str(out_a), dpi=300)
    display.savefig(str(out_a.with_suffix(".png")), dpi=300)
    plotting.show()

    # ---- Panel B: excluded cells (leftover + subject-filter NaN) --------
    excl = t[t["alt_final_roi"].isna()].copy()
    display = plotting.plot_glass_brain(
        None, display_mode="ortho", plot_abs=False,
        title=(f"Excluded cells (n = {len(excl)}): 'leftover' + "
                "subject-filter (Visual)"),
        black_bg=False, alpha=0.4,
    )
    display.add_markers(excl[coords].to_numpy(float),
                         marker_color="#c8302b", marker_size=15,
                         alpha=0.85, edgecolors="black")
    out_b = out_dir / "excluded_cells.pdf"
    display.savefig(str(out_b), dpi=300)
    display.savefig(str(out_b.with_suffix(".png")), dpi=300)
    plotting.show()

    # ---- Panel C: one glassbrain per ROI --------------------------------
    per_roi_dir = out_dir / "per_roi"
    per_roi_dir.mkdir(exist_ok=True)
    for roi in ROIS_ORDER:
        sub = t[t["alt_final_roi"] == roi]
        if sub.empty:
            continue
        display = plotting.plot_glass_brain(
            None, display_mode="ortho", plot_abs=False,
            title=(f"{_disp(roi)}  —  n = {len(sub)} cells, "
                    f"{sub['subject'].nunique()} subjects"),
            black_bg=False, alpha=0.4,
        )
        display.add_markers(sub[coords].to_numpy(float),
                             marker_color=_col(roi), marker_size=18,
                             alpha=0.9, edgecolors="black")
        stem = per_roi_dir / f"{roi}_cells.pdf"
        display.savefig(str(stem), dpi=300)
        display.savefig(str(stem.with_suffix(".png")), dpi=300)
        plotting.show()

    # ---- CSV: excluded cells for methods paragraph ----------------------
    keep = ["subject", "cell idx", "electrode label", "region label",
            "Subject Label", "Recording Site",
            "MNI_x", "MNI_y", "MNI_z",
            "HO_cortical", "HO_subcortical", "Juelich", "Brainnetome",
            "final_roi", "alt_final_roi",
            "proximity_assigned", "proximity_distance_mm"]
    keep = [c for c in keep if c in excl.columns]
    excl[keep].to_csv(out_dir / "excluded_cells.csv", index=False)

    # ---- Summary table --------------------------------------------------
    inc = (t.groupby("alt_final_roi", dropna=False)
            .agg(n_cells=("subject", "size"),
                 n_subjects=("subject", "nunique")))
    inc.to_csv(out_dir / "included_summary.csv")

    site_break = (t.groupby(["alt_final_roi", "Recording Site"], dropna=False)
                    .size().unstack(fill_value=0))
    site_break.to_csv(out_dir / "roi_by_site.csv")

    print(f"Wrote {out_a}, {out_b}, per-ROI panels, and CSVs to {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roi-table", default=DEFAULT_ROI_TABLE)
    p.add_argument("--out-dir", default=(
        "/Users/xpsy1114/Documents/projects/multiple_clocks/data/"
        "ephys_humans/derivatives/roi_labelling_overview_v2"))
    args = p.parse_args()
    draw(Path(args.roi_table), Path(args.out_dir))


if __name__ == "__main__":
    main()
