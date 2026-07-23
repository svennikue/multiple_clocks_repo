#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mPFC (ACC) coord-shift glassbrain — explains why the DSR RSA effect
changed after the v2026 microwire coord update.

Three cell classes are overlaid on a nilearn glass brain:
  * STAYED   — cell was in mPFC before AND after (both old + new RSA runs)
                Pre-microwire coord: open marker. Post: filled marker.
                Line connects the two.
  * JOINED   — cell entered mPFC (was in a neighbouring ROI, e.g.
                ventral_ACC, before the microwire update).
  * LEFT     — cell exited mPFC after the update.

The y = acc_y_cutoff plane (from cell_to_roi_MNI.py; default y = 10) is
shown as a dashed vertical line on the sagittal projection — this is
the ACC vs ventral_ACC boundary in `alt_final_roi`.

Never displays "ACC" in figure text — remapped to "mPFC".
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting

sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")
from mc.plotting.cell_results import (
    SHOWGIRL2_DISCRETE, ROI_COLORS_SHOWGIRL2, ROI_DISPLAY_NAMES,
)


OLD_RSA_CSV = ("/Users/xpsy1114/Documents/projects/multiple_clocks/data/"
                "ephys_humans/derivatives/group/DSR_RSA_simple_ROI/"
                "2026-06-29_20-09-54-DSR-corrected/roi_electrode_coords.csv")
NEW_RSA_CSV = ("/Users/xpsy1114/Documents/projects/multiple_clocks/data/"
                "ephys_humans/derivatives/group/DSR_RSA_simple_ROI/"
                "2026-07-23_12-07-25/roi_electrode_coords.csv")
ROI_TABLE   = ("/Users/xpsy1114/Documents/projects/multiple_clocks/data/"
                "ephys_humans/derivatives/neurons_with_final_roi_labels.csv")

ROI_KEY = "ACC"                 # internal roi key; display as "mPFC"
DISPLAY = ROI_DISPLAY_NAMES.get(ROI_KEY, ROI_KEY)   # -> "mPFC"

# Colour classes ------------------------------------------------------
COL_STAYED_OLD = "#cccccc"       # pre-shift (grey / open marker)
COL_STAYED_NEW = SHOWGIRL2_DISCRETE[ROI_COLORS_SHOWGIRL2[ROI_KEY]]  # mPFC teal
COL_JOINED     = "#F15A29"       # orange — 'gained'
COL_LEFT       = "#6B60AA"       # purple — 'lost'

ACC_Y_CUTOFF = 10                 # matches cell_to_roi_MNI.py:76


def _load(old_csv=OLD_RSA_CSV, new_csv=NEW_RSA_CSV, roi_tbl=ROI_TABLE):
    o = pd.read_csv(old_csv)
    n = pd.read_csv(new_csv)
    tbl = pd.read_csv(roi_tbl).rename(columns={"cell idx": "cell_idx"})
    o_acc = o[o["roi"] == ROI_KEY][["subject", "cell_idx",
                                      "MNI_x", "MNI_y", "MNI_z"]]
    n_acc = n[n["roi"] == ROI_KEY][["subject", "cell_idx",
                                      "MNI_x", "MNI_y", "MNI_z"]]
    stayed_ids = set(map(tuple, o_acc[["subject", "cell_idx"]].values)) \
                    & set(map(tuple, n_acc[["subject", "cell_idx"]].values))
    joined_ids = set(map(tuple, n_acc[["subject", "cell_idx"]].values)) \
                    - set(map(tuple, o_acc[["subject", "cell_idx"]].values))
    left_ids   = set(map(tuple, o_acc[["subject", "cell_idx"]].values)) \
                    - set(map(tuple, n_acc[["subject", "cell_idx"]].values))
    stayed = pd.DataFrame(sorted(stayed_ids), columns=["subject", "cell_idx"])
    joined = pd.DataFrame(sorted(joined_ids), columns=["subject", "cell_idx"])
    left   = pd.DataFrame(sorted(left_ids),   columns=["subject", "cell_idx"])
    stayed = stayed.merge(o_acc.rename(columns={c: c + "_old"
                                                 for c in ("MNI_x","MNI_y","MNI_z")}),
                          on=["subject","cell_idx"])
    stayed = stayed.merge(n_acc.rename(columns={c: c + "_new"
                                                 for c in ("MNI_x","MNI_y","MNI_z")}),
                          on=["subject","cell_idx"])
    joined = joined.merge(n_acc, on=["subject","cell_idx"])
    joined = joined.merge(tbl[["subject","cell_idx",
                                "MNI_x_pre_microwire",
                                "MNI_y_pre_microwire",
                                "MNI_z_pre_microwire"]],
                          on=["subject","cell_idx"], how="left")
    left = left.merge(o_acc, on=["subject","cell_idx"])
    left = left.merge(tbl[["subject","cell_idx",
                            "MNI_x_pre_microwire",
                            "MNI_y_pre_microwire",
                            "MNI_z_pre_microwire"]],
                      on=["subject","cell_idx"], how="left")
    return stayed, joined, left


def _to_arr(df, cols):
    return df[list(cols)].to_numpy(float)


def draw(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stayed, joined, left = _load()
    print(f"stayed = {len(stayed)}   joined = {len(joined)}   left = {len(left)}")

    # ---- Figure A: glass brain, sagittal + coronal + axial ------------
    display = plotting.plot_glass_brain(
        None, display_mode="ortho", plot_abs=False,
        title=(f"{DISPLAY} DSR cell set — coord shift (old vs new).\n"
                f"Stayed n={len(stayed)}, joined n={len(joined)}, "
                f"left n={len(left)}; dashed line = y={ACC_Y_CUTOFF} "
                f"(ACC vs ventral_ACC cutoff)"),
        black_bg=False, alpha=0.35,
    )

    # STAYED cells — old (grey) → new (mPFC teal) with connecting lines
    if len(stayed):
        old_c = _to_arr(stayed, ("MNI_x_old","MNI_y_old","MNI_z_old"))
        new_c = _to_arr(stayed, ("MNI_x_new","MNI_y_new","MNI_z_new"))
        display.add_markers(old_c, marker_color=COL_STAYED_OLD,
                             marker_size=20, alpha=0.75,
                             edgecolors="black")
        display.add_markers(new_c, marker_color=COL_STAYED_NEW,
                             marker_size=25, alpha=0.85,
                             edgecolors="black")
    # JOINED cells — pre coord (dashed cross via open orange), post filled
    if len(joined):
        pre = _to_arr(joined, ("MNI_x_pre_microwire",
                                "MNI_y_pre_microwire",
                                "MNI_z_pre_microwire"))
        post = _to_arr(joined, ("MNI_x","MNI_y","MNI_z"))
        display.add_markers(pre, marker_color="white",
                             marker_size=45, alpha=0.9,
                             edgecolors=COL_JOINED)
        display.add_markers(post, marker_color=COL_JOINED,
                             marker_size=55, alpha=0.95,
                             edgecolors="black")
    # LEFT cells — pre (mPFC teal) → post (purple)
    if len(left):
        pre = _to_arr(left, ("MNI_x","MNI_y","MNI_z"))
        post = _to_arr(left, ("MNI_x_pre_microwire",
                               "MNI_y_pre_microwire",
                               "MNI_z_pre_microwire"))
        # note: pre for 'left' is OLD RSA coord (was ACC), post is NEW MNI
        display.add_markers(pre, marker_color=COL_STAYED_NEW,
                             marker_size=55, alpha=0.6,
                             edgecolors="black")
        display.add_markers(post, marker_color=COL_LEFT,
                             marker_size=55, alpha=0.95,
                             edgecolors="black")

    out_path = out_dir / f"mpfc_coord_shift_glassbrain.pdf"
    display.savefig(str(out_path), dpi=300)
    display.savefig(str(out_path.with_suffix(".png")), dpi=300)
    plotting.show()

    # ---- Figure B: sagittal detail with y=ACC_Y_CUTOFF plane ----------
    # Show a zoomed 2D sagittal-like view of (y, z) with the y=10 cutoff
    # so it is crystal-clear who crossed the boundary.
    fig, ax = plt.subplots(figsize=(9 / 2.54, 8 / 2.54),
                            constrained_layout=True)
    if len(stayed):
        old_c = _to_arr(stayed, ("MNI_x_old","MNI_y_old","MNI_z_old"))
        new_c = _to_arr(stayed, ("MNI_x_new","MNI_y_new","MNI_z_new"))
        for o_i, n_i in zip(old_c, new_c):
            ax.plot([o_i[1], n_i[1]], [o_i[2], n_i[2]],
                    color="#bbbbbb", lw=0.4, alpha=0.6)
        ax.scatter(old_c[:,1], old_c[:,2], s=8, facecolor="white",
                    edgecolor=COL_STAYED_OLD, linewidth=0.5,
                    label=f"stayed {DISPLAY} (pre)")
        ax.scatter(new_c[:,1], new_c[:,2], s=10, facecolor=COL_STAYED_NEW,
                    edgecolor="black", linewidth=0.4,
                    label=f"stayed {DISPLAY} (post)")
    if len(joined):
        pre = _to_arr(joined, ("MNI_x_pre_microwire",
                                "MNI_y_pre_microwire",
                                "MNI_z_pre_microwire"))
        post = _to_arr(joined, ("MNI_x","MNI_y","MNI_z"))
        for a, b in zip(pre, post):
            ax.annotate("", xy=(b[1], b[2]), xytext=(a[1], a[2]),
                        arrowprops=dict(arrowstyle="->", lw=0.7,
                                         color=COL_JOINED, alpha=0.9))
        ax.scatter(post[:,1], post[:,2], s=32, facecolor=COL_JOINED,
                    edgecolor="black", linewidth=0.4,
                    label=f"joined {DISPLAY} (post)")
    if len(left):
        pre = _to_arr(left, ("MNI_x","MNI_y","MNI_z"))
        post = _to_arr(left, ("MNI_x_pre_microwire",
                               "MNI_y_pre_microwire",
                               "MNI_z_pre_microwire"))
        for a, b in zip(pre, post):
            ax.annotate("", xy=(b[1], b[2]), xytext=(a[1], a[2]),
                        arrowprops=dict(arrowstyle="->", lw=0.7,
                                         color=COL_LEFT, alpha=0.9))
        ax.scatter(post[:,1], post[:,2], s=32, facecolor=COL_LEFT,
                    edgecolor="black", linewidth=0.4,
                    label=f"left {DISPLAY} (post)")
    ax.axvline(ACC_Y_CUTOFF, color="black", ls="--", lw=0.7,
                label=f"{DISPLAY} y-cutoff (y = {ACC_Y_CUTOFF})")
    ax.set_xlabel("MNI y (mm, +anterior)", fontsize=9)
    ax.set_ylabel("MNI z (mm, +superior)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8, length=2, pad=1)
    ax.set_title(f"{DISPLAY} DSR cells — (y, z) view of coord shift",
                 fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"mpfc_coord_shift_yz.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved to", out_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=(
        "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/"
        "derivatives/group/DSR_RSA_simple_ROI/"
        "2026-07-23_12-07-25/mpfc_coord_shift_diagnostic"))
    args = p.parse_args()
    draw(Path(args.out_dir))


if __name__ == "__main__":
    main()
