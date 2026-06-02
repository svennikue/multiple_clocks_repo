#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assign each recorded human cell to an anatomical ROI based on its MNI
coordinates and save the labelled table + per-ROI count CSVs.

This script is **CSV-only** — all brain-coverage plotting has been moved
to scripts/roi_brain_visualization.py so the labelling pipeline stays
fast and easy to re-run.

Outputs (all into the same folder as `path_to_cell_table`):
  - neurons_with_final_roi_labels.csv  (per-cell, both `final_roi`
                                        and `alt_final_roi`)
  - proximity_assigned_cells.csv        (cells whose ROI was assigned
                                        via centroid proximity)
  - roi_counts.csv                      (cell + subject counts per
                                        `final_roi`)
  - alt_roi_counts.csv                  (cell + subject counts per
                                        `alt_final_roi`)
  - roi_centroids_used_for_proximity.csv
  - roi_count_overview_final.png        (bar plot of counts, kept here
                                        as a data summary; it isn't
                                        brain-coverage plotting)
  - roi_count_overview_alt.png
"""

import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn import datasets


# =============================================================================
# USER SETTINGS
# =============================================================================

path_to_cell_table = (
    "/Users/xpsy1114/Documents/projects/multiple_clocks/"
    "data/ephys_humans/derivatives/neurons_MNI_latest.csv"
)

path_to_brainnetome = "/Users/xpsy1114/Documents/toolboxes/Brainnatome"

brainnetome_nii = os.path.join(path_to_brainnetome, "BN_Atlas_246_1mm.nii.gz")
brainnetome_lut = os.path.join(path_to_brainnetome, "BN_Atlas_246_LUT.txt")

output_dir = os.path.dirname(path_to_cell_table)

site_col = "Recording Site"     # Baylor rows are treated as MNI305
coord_cols = ["MNI_x", "MNI_y", "MNI_z"]

hc_ap_cutoff = -21              # y >= -21 anterior HC, y < -21 posterior HC
acc_y_cutoff = 10               # only fairly anterior cingulate counts as ACC

# `alt_final_roi` ROIs with fewer than this many contributing subjects are
# dropped (set to NaN) so downstream analyses don't try to model them.
MIN_SUBJECTS_PER_ALT_ROI = 3

roi_order = [
    "EC",
    "Parahippocampal",
    "HC_anterior",
    "HC_mid",
    "ventral_ACC",
    "ACC",
    "medial_CC",
    "PCC",
    "OFC11",
    "OFC13",
    "Visual",
]

# Alternative labelling (see assign_alt_roi): ACC is split by `acc_y_cutoff`
# into ACC / medial_CC, and OFC11 + OFC13 + ventral_ACC collapse into
# medialOFC.  Any of these whose cells come from < MIN_SUBJECTS_PER_ALT_ROI
# distinct subjects is dropped from the column after assignment.
alt_roi_order = [
    "EC",
    "Parahippocampal",
    "HC_anterior",
    "HC_mid",
    "medialOFC",
    "ACC",
    "medial_CC",
    "PCC",
    "Visual",
]


# =============================================================================
# SMALL HELPERS
# =============================================================================

def contains(text, pattern):
    return pattern.lower() in str(text).lower()


def contains_any(text, patterns):
    text = str(text).lower()
    return any(p.lower() in text for p in patterns)


def hc_label_from_y(y):
    return "HC_anterior" if float(y) >= hc_ap_cutoff else "HC_mid"


def get_img(atlas_or_img):
    if hasattr(atlas_or_img, "maps"):
        img = atlas_or_img.maps
    else:
        img = atlas_or_img
    return nib.load(img) if isinstance(img, str) else img


# =============================================================================
# MNI305 -> MNI152 TRANSFORM
# =============================================================================

MNI305_TO_MNI152 = np.array([
    [ 0.9975, -0.0073,  0.0176, -0.0429],
    [ 0.0146,  1.0009, -0.0024,  1.5496],
    [-0.0130, -0.0093,  0.9971,  1.1840],
    [ 0.0000,  0.0000,  0.0000,  1.0000],
])


def mni305_to_mni152(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords[None, :]
    coords_h = np.c_[coords, np.ones(len(coords))]
    return (coords_h @ MNI305_TO_MNI152.T)[:, :3]


# =============================================================================
# ATLAS LOOKUP CLASSES
# =============================================================================

class AtlasLookup:
    def __init__(self, atlas):
        self.img = get_img(atlas)
        self.data = self.img.get_fdata()
        self.inv_affine = np.linalg.inv(self.img.affine)
        self.labels = list(atlas.labels)

    def label_at(self, x, y, z):
        voxel = nib.affines.apply_affine(self.inv_affine, [x, y, z])
        voxel = np.round(voxel).astype(int)

        if np.any(voxel < 0) or np.any(voxel >= self.data.shape):
            return "outside atlas"

        idx = int(self.data[tuple(voxel)])

        if idx == 0:
            return "background"

        if idx < 0 or idx >= len(self.labels):
            return f"unknown index {idx}"

        return self.labels[idx]


def load_brainnetome_lut(lut_path):
    labels = {0: "background"}

    with open(lut_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            idx = None
            for p in parts:
                try:
                    idx = int(p)
                    break
                except ValueError:
                    pass

            if idx is None:
                continue

            text_parts = [
                p for p in parts
                if not re.match(r"^-?\d+(\.\d+)?$", p)
            ]

            label = " ".join(text_parts).strip()
            labels[idx] = label if label else f"Brainnetome_{idx}"

    return labels


class BrainnetomeLookup:
    def __init__(self, nii_path, lut_path):
        self.img = nib.load(nii_path)
        self.data = self.img.get_fdata()
        self.inv_affine = np.linalg.inv(self.img.affine)
        self.labels = load_brainnetome_lut(lut_path)

    def label_at(self, x, y, z):
        voxel = nib.affines.apply_affine(self.inv_affine, [x, y, z])
        voxel = np.round(voxel).astype(int)

        if np.any(voxel < 0) or np.any(voxel >= self.data.shape):
            return "outside atlas"

        idx = int(self.data[tuple(voxel)])

        if idx == 0:
            return "background"

        return self.labels.get(idx, f"Brainnetome index {idx}")


# =============================================================================
# INITIAL ROI ASSIGNMENT FROM ATLASES
# =============================================================================
manual_roi_col = "correct ROI"   # change to the exact column name in your CSV


def assign_initial_roi(row):
    # if MNI coordinates are missing, trust only manual PCC
    if row[coord_cols].isna().all():
        manual_roi = str(row.get(manual_roi_col, "")).strip().upper()
        if manual_roi == "PCC":
            return "PCC"
        return np.nan

    y = float(row["MNI_y"])

    ho_cort = str(row["HO_cortical"])
    ho_sub = str(row["HO_subcortical"])
    juelich = str(row["Juelich"])
    brainnetome = str(row["Brainnetome"])

    if contains_any(juelich, ["gm hippocampus entorhinal cortex", "entorhinal"]):
        return "EC"

    if contains_any(brainnetome, ["tl_r", "tl_l"]):
        return "Parahippocampal"

    if contains_any(ho_cort, ["parahippocampal gyrus", "parahippocampal"]):
        return "Parahippocampal"

    if contains(ho_sub, "hippocampus"):
        return hc_label_from_y(y)

    if contains_any(juelich, ["hippocampus subiculum", "subiculum"]):
        return hc_label_from_y(y)

    if contains(brainnetome, "a14m"):
        return "ventral_ACC"

    if contains_any(brainnetome, ["a32sg", "a32p", "a24rv"]):
        return "ACC"

    if contains(ho_cort, "cingulate gyrus, anterior division"):
        return "ACC"

    if contains(brainnetome, "a23"):
        return "PCC"

    if contains(ho_cort, "cingulate gyrus, posterior division"):
        return "PCC"

    if contains(brainnetome, "a11m"):
        return "OFC11"

    if contains_any(brainnetome, ["a13_r", "a13_l", "a13"]):
        return "OFC13"

    if contains_any(ho_cort, [
        "occipital",
        "cuneal",
        "lingual",
        "intracalcarine",
        "supracalcarine",
        "occipital pole",
    ]):
        return "Visual"

    if contains_any(juelich, ["v1", "v2", "v3", "visual", "calcarine"]):
        return "Visual"

    return "leftover"


def assign_alt_roi(row):
    """Alternative ROI label derived from the finalised `final_roi`.

    `final_roi` itself is left untouched.  Differences:
      1. ACC is split on the anterior-posterior axis at `acc_y_cutoff`:
         MNI_y >= cutoff stays 'ACC'; more posterior cingulate -> 'medial_CC'.
      2. OFC11, OFC13 and ventral_ACC collapse into a single 'medialOFC'.
    """
    roi = row["final_roi"]
    if roi == "ACC":
        y = row["MNI_y"]
        if pd.isna(y):
            return "ACC"
        return "ACC" if float(y) >= acc_y_cutoff else "medial_CC"
    if roi in ("OFC11", "OFC13", "ventral_ACC"):
        return "medialOFC"
    return roi


# =============================================================================
# PROXIMITY-BASED LEFTOVER ASSIGNMENT
# =============================================================================

def assign_leftovers_by_roi_centroid(df_in):
    """
    Assign every leftover to the nearest existing ROI centroid.

    Adds:
      final_roi_initial
      final_roi
      proximity_assigned
      proximity_distance_mm
      proximity_second_distance_mm
      proximity_margin_mm
      proximity_ratio

    proximity_ratio = nearest distance / second nearest distance.
    Smaller ratio means more confident.
    """
    df = df_in.copy()
    df["final_roi_initial"] = df["final_roi"]

    anchor = df[df["final_roi"].isin(roi_order)].copy()

    if anchor.empty:
        raise ValueError("No non-leftover ROI cells available for proximity assignment.")

    centroids = (
        anchor
        .groupby("final_roi")[coord_cols]
        .mean()
        .reindex(roi_order)
        .dropna()
    )

    df["proximity_assigned"] = False
    df["proximity_distance_mm"] = np.nan
    df["proximity_second_distance_mm"] = np.nan
    df["proximity_margin_mm"] = np.nan
    df["proximity_ratio"] = np.nan

    leftover_mask = df["final_roi"].eq("leftover")

    for idx, row in df[leftover_mask].iterrows():
        point = row[coord_cols].to_numpy(dtype=float)
        centroid_coords = centroids.to_numpy(dtype=float)

        dists = np.linalg.norm(centroid_coords - point, axis=1)
        order = np.argsort(dists)

        nearest_i = order[0]
        second_i = order[1] if len(order) > 1 else order[0]

        nearest_roi = centroids.index[nearest_i]
        nearest_dist = float(dists[nearest_i])
        second_dist = float(dists[second_i])

        df.loc[idx, "final_roi"] = nearest_roi
        df.loc[idx, "proximity_assigned"] = True
        df.loc[idx, "proximity_distance_mm"] = nearest_dist
        df.loc[idx, "proximity_second_distance_mm"] = second_dist
        df.loc[idx, "proximity_margin_mm"] = second_dist - nearest_dist
        df.loc[idx, "proximity_ratio"] = nearest_dist / second_dist if second_dist > 0 else np.nan

    return df, centroids


# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(path_to_cell_table)

required_cols = ["subject", "cell idx", "electrode label",
                 "MNI_x", "MNI_y", "MNI_z"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df[coord_cols] = df[coord_cols].astype(float)

df["MNI_x_original"] = df["MNI_x"]
df["MNI_y_original"] = df["MNI_y"]
df["MNI_z_original"] = df["MNI_z"]


# =============================================================================
# BAYLOR MNI305 -> MNI152
# =============================================================================

if site_col in df.columns:
    baylor_mask = df[site_col].astype(str).str.lower().eq("baylor")
else:
    print(f"Warning: column '{site_col}' not found. No Baylor transform applied.")
    baylor_mask = pd.Series(False, index=df.index)

df["coordinate_space_original"] = np.where(baylor_mask, "MNI305", "assumed_MNI152")
df["coordinate_space_plot"] = "MNI152"

if baylor_mask.sum() > 0:
    df.loc[baylor_mask, coord_cols] = mni305_to_mni152(
        df.loc[baylor_mask, coord_cols].to_numpy()
    )

print(f"Transformed {baylor_mask.sum()} Baylor rows from MNI305 to MNI152.")


# =============================================================================
# LOAD ATLASES
# =============================================================================

print("Loading atlases...")

ho_cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
ho_sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
juelich = datasets.fetch_atlas_juelich("maxprob-thr25-2mm")

ho_cort_lookup = AtlasLookup(ho_cort)
ho_sub_lookup = AtlasLookup(ho_sub)
juelich_lookup = AtlasLookup(juelich)
brainnetome_lookup = BrainnetomeLookup(brainnetome_nii, brainnetome_lut)

print("Atlases loaded.")


# =============================================================================
# QUERY ATLASES
# =============================================================================

def query_atlases(row):
    x, y, z = row["MNI_x"], row["MNI_y"], row["MNI_z"]
    return pd.Series({
        "HO_cortical": ho_cort_lookup.label_at(x, y, z),
        "HO_subcortical": ho_sub_lookup.label_at(x, y, z),
        "Juelich": juelich_lookup.label_at(x, y, z),
        "Brainnetome": brainnetome_lookup.label_at(x, y, z),
    })


print("Querying atlas labels...")

atlas_df = df.apply(query_atlases, axis=1)
df_labeled = pd.concat([df, atlas_df], axis=1)

df_labeled["unique_unit_id"] = (
    "sub-" + df_labeled["subject"].astype(str)
    + "_cell-" + df_labeled["cell idx"].astype(str)
    + "_elec-" + df_labeled["electrode label"].astype(str)
)

df_labeled["final_roi"] = df_labeled.apply(assign_initial_roi, axis=1)

print("Initial ROI counts:")
print(df_labeled["final_roi"].value_counts(dropna=False))


# =============================================================================
# ASSIGN LEFTOVERS BY COORDINATE PROXIMITY
# =============================================================================

df_labeled, roi_centroids = assign_leftovers_by_roi_centroid(df_labeled)

print("\nROI centroids used for proximity assignment:")
print(roi_centroids)

print("\nFinal ROI counts after proximity assignment:")
print(df_labeled["final_roi"].value_counts(dropna=False))


# =============================================================================
# ALTERNATIVE ROI LABELLING (alt_final_roi)
# =============================================================================

df_labeled["alt_final_roi"] = df_labeled.apply(assign_alt_roi, axis=1)

# Exclude `alt_final_roi` ROIs that contribute fewer than
# MIN_SUBJECTS_PER_ALT_ROI distinct subjects: their cells get NaN'd so the
# downstream analyses just skip them.  `final_roi` is left untouched.
sub_counts_alt = df_labeled.groupby("alt_final_roi")["subject"].nunique()
sparse_alt_rois = sub_counts_alt[
    sub_counts_alt < MIN_SUBJECTS_PER_ALT_ROI].index.tolist()
if sparse_alt_rois:
    n_dropped = int(
        df_labeled["alt_final_roi"].isin(sparse_alt_rois).sum())
    print(f"\nDropping alt_final_roi ROIs with < "
          f"{MIN_SUBJECTS_PER_ALT_ROI} subjects "
          f"({n_dropped} cells affected): {sparse_alt_rois}")
    df_labeled.loc[
        df_labeled["alt_final_roi"].isin(sparse_alt_rois),
        "alt_final_roi"] = np.nan

# Restrict the plotting/order list to ROIs that survived the filter.
alt_roi_order = [r for r in alt_roi_order if r not in sparse_alt_rois]

print("\nAlternative ROI counts (alt_final_roi, post-filter):")
print(df_labeled["alt_final_roi"].value_counts(dropna=False))


# =============================================================================
# COUNTS AND DATA-SUMMARY BAR PLOT
# =============================================================================

def make_roi_count_table(df, roi_field, order, subject_field="subject"):
    """Per-ROI table with `n_cells` and `n_subjects`, ordered by `order`.

    `n_subjects` is the number of unique values in `subject_field` that
    contribute at least one cell to that ROI.
    """
    grouped = (df.groupby(roi_field)
                 .agg(n_cells=(subject_field, "size"),
                      n_subjects=(subject_field, "nunique")))
    return (grouped.reindex(order, fill_value=0)
                   .rename_axis(roi_field)
                   .reset_index())


def plot_roi_count_overview(counts_df, roi_field, title, save_path):
    """Bar plot of cell counts per ROI; each x-label carries both the
    cell count and the number of contributing subjects.
    Kept here as a data summary (no spatial brain plotting)."""
    labels = [f"{r}\n(n = {n} cells; {s} subjects)"
              for r, n, s in zip(counts_df[roi_field],
                                 counts_df["n_cells"],
                                 counts_df["n_subjects"])]
    plt.figure(figsize=(11, 5))
    plt.bar(labels, counts_df["n_cells"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of cells")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved {save_path}")


roi_counts = make_roi_count_table(
    df_labeled, "final_roi", roi_order)
alt_roi_counts = make_roi_count_table(
    df_labeled, "alt_final_roi", alt_roi_order)

print("\nROI counts (final_roi):")
print(roi_counts)
print("\nROI counts (alt_final_roi):")
print(alt_roi_counts)

plot_roi_count_overview(
    roi_counts, "final_roi", "Cell counts by final_roi",
    os.path.join(output_dir, "roi_count_overview_final.png"))
plot_roi_count_overview(
    alt_roi_counts, "alt_final_roi", "Cell counts by alt_final_roi",
    os.path.join(output_dir, "roi_count_overview_alt.png"))


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

labeled_out = os.path.join(output_dir, "neurons_with_final_roi_labels.csv")
proximity_out = os.path.join(output_dir, "proximity_assigned_cells.csv")
counts_out = os.path.join(output_dir, "roi_counts.csv")
alt_counts_out = os.path.join(output_dir, "alt_roi_counts.csv")
centroids_out = os.path.join(output_dir, "roi_centroids_used_for_proximity.csv")

proximity_df = df_labeled[df_labeled["proximity_assigned"]].copy()

df_labeled.to_csv(labeled_out, index=False)
proximity_df.to_csv(proximity_out, index=False)
roi_counts.to_csv(counts_out, index=False)
alt_roi_counts.to_csv(alt_counts_out, index=False)
roi_centroids.to_csv(centroids_out)

print("\nSaved files:")
print(labeled_out)
print(proximity_out)
print(counts_out)
print(alt_counts_out)
print(centroids_out)
print("\nFor brain-coverage figures (glassbrain / mosaic / mne.viz.Brain),"
      " run scripts/roi_brain_visualization.py")
