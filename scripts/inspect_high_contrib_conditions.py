#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overview of high-contributing datapoints: where (locations) subjects are when
conditions are active, and which condition-pairs are compared.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect conditions for high-contributing datapoints."
    )
    parser.add_argument(
        "--summary-csv",
        default="data/derivatives/group/RDM_plots/summary_group_contrib_DSR_ortho_location.csv",
        help="Group summary CSV from inspect_group_contrib_ortho_DSR.py",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="Top-N datapoints by mean_contrib (positive only).",
    )
    parser.add_argument(
        "--subjects",
        default="",
        help="Comma-separated subject IDs (e.g., sub-01,sub-02).",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory (defaults next to summary csv).",
    )
    return parser.parse_args()


def extract_evs_from_label(label):
    """Return list of EV strings from a label like 'A1... | A2... vs B1... | B2...'"""
    if not label:
        return []
    parts = label.split(" vs ")
    evs = []
    for part in parts:
        for ev in part.split(" | "):
            evs.append(ev.strip())
    return [e for e in evs if e]


def main():
    args = parse_args()

    source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
    if not os.path.isdir(source_dir):
        source_dir = "/home/fs0/xpsy1114/scratch"

    summary_path = args.summary_csv
    if not os.path.isabs(summary_path):
        summary_path = os.path.join(source_dir, summary_path)

    out_dir = args.out_dir
    if not out_dir:
        out_dir = os.path.dirname(summary_path)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(summary_path)
    df = df[df["mean_contrib"] > 0].sort_values("mean_contrib", ascending=False)
    top_df = df.head(args.top_n)

    labels = top_df["label"].fillna("").tolist()
    top_evs = sorted(set(sum((extract_evs_from_label(l) for l in labels), [])))

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        # default: infer from data directory
        data_dir = os.path.join(source_dir, "data/derivatives")
        subjects = sorted([d for d in os.listdir(data_dir) if d.startswith("sub-")])
    exclude_subs = {"sub-10", "sub-21", "sub-39"}
    subjects = [s for s in subjects if s not in exclude_subs]

    locs = list(range(1, 10))
    loc_counts = {loc: 0 for loc in locs}
    combo_counts = {}

    for sub in subjects:
        beh_dir = os.path.join(source_dir, "data/derivatives", sub, "beh")
        beh_path = os.path.join(beh_dir, f"{sub}_beh_fmri_clean.csv")
        if not os.path.exists(beh_path):
            continue
        beh_df = pd.read_csv(beh_path)
        for ev in top_evs:
            ev_rows = beh_df[beh_df["unique_time_bin_type"] == ev]
            if ev_rows.empty:
                continue
            # unique locations visited during this EV (count each EV once)
            curr_locs = sorted(set(ev_rows["curr_loc"].dropna().astype(int).tolist()))
            for loc in curr_locs:
                if loc in loc_counts:
                    loc_counts[loc] += 1
            # also count reward locations once per EV, if available
            if "curr_rew" in ev_rows.columns:
                rew_locs = sorted(set(ev_rows["curr_rew"].dropna().astype(int).tolist()))
                for loc in rew_locs:
                    if loc in loc_counts:
                        loc_counts[loc] += 1
            # count location-combinations as the set of unique locations (per EV)
            if len(curr_locs) >= 2:
                combo = tuple(curr_locs)
                combo_counts[combo] = combo_counts.get(combo, 0) + 1

    # Bar plot for locations and location-combinations (top 20 labels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(list(loc_counts.keys()), list(loc_counts.values()), color="#4c78a8")
    axes[0].set_title("Location counts (unique per EV)", fontsize=12)
    axes[0].set_xlabel("Location", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)

    if combo_counts:
        combos_sorted = sorted(combo_counts.items(), key=lambda kv: kv[1], reverse=True)
        combo_labels = ["-".join(map(str, combo)) for combo, _ in combos_sorted]
        combo_vals = [v for _, v in combos_sorted]
        axes[1].barh(combo_labels[::-1], combo_vals[::-1], color="#f58518")
    axes[1].set_title("Location-combination counts (unique per EV)", fontsize=12)
    axes[1].set_xlabel("Count", fontsize=10)
    axes[1].set_ylabel("Location pair", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bar_locations_and_combos_top20.png"), dpi=200)
    #plt.close(fig)

    # Plot condition-pair labels (top 30 by mean_contrib)
    top_labels = (
        top_df[["label", "mean_contrib"]]
        .dropna()
        .head(30)
        .iloc[::-1]
    )
    if len(top_labels) > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        ax2.barh(top_labels["label"], top_labels["mean_contrib"])
        ax2.set_title("Top condition-pair labels (by mean_contrib)", fontsize=12)
        ax2.set_xlabel("Mean contribution", fontsize=10)
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "bar_top_condition_pairs.png"), dpi=200)
        # plt.close(fig2)
    # Save EV list
    with open(os.path.join(out_dir, "top_contrib_EVs.txt"), "w") as f:
        for ev in top_evs:
            f.write(ev + "\n")


if __name__ == "__main__":
    main()
