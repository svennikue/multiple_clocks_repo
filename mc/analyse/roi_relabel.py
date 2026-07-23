#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROI relabelling utility for cell-level per-cell result tables.

Cell CV / permutation results are computed at the cell level and stored in
``per_cell*.csv`` files. When we re-run `cell_to_roi_MNI.py` with new MNI
coordinates and get a fresh `neurons_with_final_roi_labels.csv`, the neural
compute doesn't change — only the ROI membership does. So instead of
recomputing 1000 permutations × N cells, we can just join the new ROI
column onto the existing per-cell CSV and re-run the aggregation +
plotting layer.

This module is shared by the three cell-level analyses that support a
reload path (sustained state, spatial peaks, per-lag encoding).
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd


DEFAULT_ROI_TABLE = ("/Users/xpsy1114/Documents/projects/multiple_clocks/"
                     "data/ephys_humans/derivatives/"
                     "neurons_with_final_roi_labels.csv")


def relabel_per_cell(
    per_cell: pd.DataFrame,
    roi_table_csv: str = DEFAULT_ROI_TABLE,
    roi_col_in_table: str = "alt_final_roi",
    roi_col_in_per_cell: str = "roi",
    subject_key_per_cell: str = "subject",
    cell_key_per_cell: str = "cell_idx",
    audit_stream=None,
):
    """Overwrite the ROI column of `per_cell` from the fresh ROI table.

    Join keys:
      per_cell[(`subject_key_per_cell`, `cell_key_per_cell`)]
      == roi_table[('subject', 'cell idx')]

    Returns
    -------
    (relabelled_df, audit_dict)
        `relabelled_df`: copy of `per_cell` with:
          - `roi` overwritten from `roi_col_in_table`
          - `roi_previous` column preserving the old label
          - `roi_changed` bool column
        `audit_dict`: {n_total, n_matched, n_changed, roi_previous,
                        n_dropped_new_nan, transitions_df}.
    """
    roi = pd.read_csv(roi_table_csv)[["subject", "cell idx", roi_col_in_table]]
    roi = roi.rename(columns={"cell idx": cell_key_per_cell,
                              "subject":  subject_key_per_cell,
                              roi_col_in_table: "_roi_new"})
    out = per_cell.merge(roi, on=[subject_key_per_cell, cell_key_per_cell],
                        how="left")
    n_total = len(out)
    n_matched = int(out["_roi_new"].notna().sum() + out["_roi_new"].isna().sum())

    prev_col = f"{roi_col_in_per_cell}_previous"
    out[prev_col] = out[roi_col_in_per_cell]
    out[roi_col_in_per_cell] = out["_roi_new"]
    out["roi_changed"] = out[roi_col_in_per_cell] != out[prev_col]
    out = out.drop(columns=["_roi_new"])

    n_matched_lookup = int(out[roi_col_in_per_cell].notna().sum())
    n_dropped_new_nan = int(out[roi_col_in_per_cell].isna().sum())
    n_changed = int(out["roi_changed"].sum())

    transitions = (out.groupby([prev_col, roi_col_in_per_cell], dropna=False)
                     .size()
                     .reset_index(name="n_cells")
                     .sort_values("n_cells", ascending=False))
    transitions_changed = transitions[transitions[prev_col]
                                       != transitions[roi_col_in_per_cell]]

    audit = {
        "n_total": n_total,
        "n_matched_new_roi_not_na": n_matched_lookup,
        "n_now_nan": n_dropped_new_nan,
        "n_changed": n_changed,
        "transitions_changed": transitions_changed,
        "roi_previous_counts": out[prev_col].value_counts(dropna=False),
        "roi_new_counts": out[roi_col_in_per_cell].value_counts(dropna=False),
    }

    if audit_stream is None:
        _print_audit(audit)
    elif audit_stream is False:
        pass
    else:
        _print_audit(audit, stream=audit_stream)

    return out, audit


def _print_audit(audit, stream=None):
    def emit(s=""):
        if stream is None:
            print(s)
        else:
            stream.write(s + "\n")

    emit("=== ROI relabel audit ===")
    emit(f"n cells in per-cell table              : {audit['n_total']}")
    emit(f"n with a non-NaN new ROI               : "
         f"{audit['n_matched_new_roi_not_na']}")
    emit(f"n whose new ROI is NaN (dropped/small) : {audit['n_now_nan']}")
    emit(f"n that changed ROI label               : {audit['n_changed']}")
    emit("")
    emit("Previous ROI counts:")
    emit(audit["roi_previous_counts"].to_string())
    emit("")
    emit("New ROI counts:")
    emit(audit["roi_new_counts"].to_string())
    if not audit["transitions_changed"].empty:
        emit("")
        emit("Transitions (previous -> new), only rows where the label changed:")
        emit(audit["transitions_changed"].to_string(index=False))


def drop_nan_roi(df: pd.DataFrame, roi_col: str = "roi") -> pd.DataFrame:
    """Convenience: drop rows whose (post-relabel) ROI is NaN. Downstream
    aggregations already skip NaN ROIs, but doing this explicitly gives a
    cleaner audit trail in the reload output."""
    return df.loc[df[roi_col].notna()].copy()
