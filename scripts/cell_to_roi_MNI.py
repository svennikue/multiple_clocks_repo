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
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn import datasets

sys.path.insert(
    0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo"
)
# Project-wide ROI palette — hardcoded source-of-truth lives in
# mc.plotting.cell_results.SHOWGIRL2_DISCRETE (era_brewer.era_brew at n=7
# silently interpolates / duplicates colours, so we don't use it here).
from mc.plotting.cell_results import SHOWGIRL2_DISCRETE, roi_display


# =============================================================================
# USER SETTINGS
# =============================================================================

path_to_cell_table = (
    "/Users/xpsy1114/Documents/projects/multiple_clocks/"
    "data/ephys_humans/derivatives/neurons_MNI_latest.csv"
)

# Baylor microwire coordinate updates (v2026). One CSV per subject; filenames
# look like `YEJ-electrodes_v2026.csv`, so the 3-letter prefix matches the tail
# of the main table's `Subject Label` (e.g. 'BY2-YEJ'). Rows with
# Type == 'microwires' carry bundle-level MNI152 coords; the main table stores
# per-channel labels (`mRT2bHaEa02`), so we match by stripping the trailing 2
# digits (case-insensitively) and re-use one bundle coord for all its channels.
path_to_microwire_updates_dir = (
    "/Users/xpsy1114/Documents/projects/multiple_clocks/"
    "data/ephys_humans/ABCD_pts_elecFilesForSvenja_v2026"
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

# Proximity fallback: cells that no atlas rule matched are only assigned to
# the nearest ROI centroid if the distance is <= this. Beyond this, the cell
# stays "leftover" (and hence NaN in `alt_final_roi`). Prior versions used no
# distance cap (pulling thalamus / insula white matter / etc. into
# ACC/HC/PCC at 15-30 mm) or a 5 mm cap (which excluded three site-labelled
# microwire bundles at 7-9 mm — BY2-YEX mLF2aCa 'LACC', BY2-YFM mLF3aOFC
# 'LOFC', UT202409 chan115 'RHC'). 10 mm keeps out the 21+ mm garbage while
# recovering the site-labelled bundles that sit just outside atlas rules.
PROXIMITY_MAX_DIST_MM = 10.0

roi_order = [
    "EC",
    "Parahippocampal",
    "HC_anterior",
    "HC_mid",
    "ventral_ACC",
    "ACC",
    "medial_CC",
    "PCC",
    "Precuneus",       # Brainnetome A31 / dmPOS (dorsomedial parietal)
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
    "Precuneus",
    "Visual",
]


# Project-wide ROI colours (CLAUDE.md mapping on the canonical Showgirl2
# discrete palette pulled from SHOWGIRL2_DISCRETE).
ROI_COLOURS = {
    "EC":              SHOWGIRL2_DISCRETE[0],   # dark red
    "ACC":             SHOWGIRL2_DISCRETE[1],   # orange
    "HC_anterior":     SHOWGIRL2_DISCRETE[2],   # tan
    "PCC":             SHOWGIRL2_DISCRETE[3],   # pale yellow
    "medialOFC":       SHOWGIRL2_DISCRETE[4],   # pale green
    "Parahippocampal": SHOWGIRL2_DISCRETE[5],   # sage
    "HC_mid":          SHOWGIRL2_DISCRETE[6],   # dark teal-green
    # Final-roi only (collapse into alt-roi parents — share parent hue).
    "OFC11":           SHOWGIRL2_DISCRETE[4],
    "OFC13":           SHOWGIRL2_DISCRETE[4],
    "ventral_ACC":     SHOWGIRL2_DISCRETE[1],
    "medial_CC":       "#888888",
    "Precuneus":       "#23677E",   # CLAUDE.md-defined blue for Precuneus
    "Visual":          "#bdbdbd",
}


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
# MICROWIRE COORDINATE UPDATES (Baylor v2026)
# =============================================================================

def _bundle_key(label):
    """Strip trailing 2 digits, lowercase — matches per-channel labels
    (e.g. `mRT2bHaEa04`) to bundle-level labels (`mRT2bHaEa01`)."""
    if pd.isna(label):
        return None
    return re.sub(r"\d{2}$", "", str(label)).lower()


# Reliability filter: for a subject to be "reliable", the file's MNI152 column
# must be self-consistent with (file's MNI305 column) transformed by the standard
# Fischl MNI305 -> MNI152 matrix. Clean subjects agree within a few mm; a couple
# of files (e.g. YEN, YFT) have a corrupted MNI152 column (wrong hemisphere,
# tens of mm off). Cutoff chosen from the observed distribution — the clean set
# tops out around 6 mm mean discrepancy; the broken ones sit at 14+ mm.
MICROWIRE_MNI152_TOL_MM = 8.0


def load_microwire_updates(folder):
    """Scan v2026 electrode CSVs and return
    ``(updates, reliability_df)``.

    `updates`: ``{(subject_code, bundle_key): (MNI152_x, MNI152_y, MNI152_z)}``,
    only populated for subjects whose file MNI152 passes the reliability check.
    `subject_code` is the 3-letter filename prefix
    (`YEJ-electrodes_v2026.csv` -> `YEJ`), which matches the tail of
    `Subject Label` in the main table (`'BY2-YEJ'`).

    `reliability_df`: per-subject DataFrame with `n_microwires`,
    `mni152_vs_305transform_mean_mm`, `max_mm` and `reliable` (bool).
    Subjects flagged unreliable are NOT written into `updates`; their cells
    keep the main table's original MNI (later flagged in
    `microwire_mni_source`).
    """
    updates = {}
    reliability_rows = []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith("-electrodes_v2026.csv"):
            continue
        subj_code = fn.split("-")[0]
        fdf = pd.read_csv(os.path.join(folder, fn))
        micro = fdf[fdf["Type"] == "microwires"].copy()
        if micro.empty:
            continue

        # self-consistency check on the file's own MNI152 vs its MNI305
        chk = micro[["MNI305_x", "MNI305_y", "MNI305_z",
                     "MNI152_x", "MNI152_y", "MNI152_z"]].dropna()
        if chk.empty:
            reliable = False
            mean_d = np.nan
            max_d = np.nan
        else:
            a305 = chk[["MNI305_x", "MNI305_y", "MNI305_z"]].to_numpy(float)
            a152_file = chk[["MNI152_x", "MNI152_y", "MNI152_z"]].to_numpy(float)
            a152_recomp = mni305_to_mni152(a305)
            d = np.linalg.norm(a152_file - a152_recomp, axis=1)
            mean_d = float(np.mean(d))
            max_d = float(np.max(d))
            reliable = mean_d <= MICROWIRE_MNI152_TOL_MM

        reliability_rows.append({
            "subject_code": subj_code,
            "n_microwires": len(micro),
            "mni152_vs_305transform_mean_mm": mean_d,
            "mni152_vs_305transform_max_mm": max_d,
            "reliable": reliable,
        })

        if not reliable:
            continue

        for _, r in micro.iterrows():
            key = (subj_code, _bundle_key(r["Label"]))
            try:
                x = float(r["MNI152_x"])
                y = float(r["MNI152_y"])
                z = float(r["MNI152_z"])
            except (TypeError, ValueError):
                continue
            if any(np.isnan(v) for v in (x, y, z)):
                continue
            updates[key] = (x, y, z)

    return updates, pd.DataFrame(reliability_rows)


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

    # Brainnetome (fine-grained) checks come BEFORE Harvard-Oxford coarse
    # cingulate masks: HO's "Cingulate Gyrus, anterior division" bleeds into
    # middle-cingulate cells that Brainnetome correctly labels as area 23
    # (posterior cingulate). Fine-grained wins.
    if contains(brainnetome, "a14m"):
        return "ventral_ACC"

    if contains_any(brainnetome, ["a32sg", "a32p", "a24rv"]):
        return "ACC"

    if contains(brainnetome, "a23"):
        return "PCC"

    if contains(ho_cort, "cingulate gyrus, anterior division"):
        return "ACC"

    if contains(ho_cort, "cingulate gyrus, posterior division"):
        return "PCC"

    # Precuneus (dorsomedial parietal) — a distinct medial-parietal
    # region separate from PCC (BA23). Brainnetome A31 = medial precuneus,
    # dmPOS = dorsomedial parieto-occipital sulcus. HO "Precuneous Cortex"
    # is a coarser mask retained as a fallback.
    if contains_any(brainnetome, ["a31_l", "a31_r", "dmpos_l", "dmpos_r"]):
        return "Precuneus"

    if contains(ho_cort, "precuneous cortex"):
        return "Precuneus"

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
      3. Cells still labelled 'leftover' after the proximity cap become NaN.
    """
    roi = row["final_roi"]
    if roi == "leftover" or pd.isna(roi):
        return np.nan
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

        # Distance cap: leave "leftover" if beyond PROXIMITY_MAX_DIST_MM.
        if nearest_dist > PROXIMITY_MAX_DIST_MM:
            df.loc[idx, "proximity_distance_mm"] = nearest_dist
            df.loc[idx, "proximity_second_distance_mm"] = second_dist
            df.loc[idx, "proximity_margin_mm"] = second_dist - nearest_dist
            df.loc[idx, "proximity_ratio"] = (nearest_dist / second_dist
                                                if second_dist > 0 else np.nan)
            continue

        df.loc[idx, "final_roi"] = nearest_roi
        df.loc[idx, "proximity_assigned"] = True
        df.loc[idx, "proximity_distance_mm"] = nearest_dist
        df.loc[idx, "proximity_second_distance_mm"] = second_dist
        df.loc[idx, "proximity_margin_mm"] = second_dist - nearest_dist
        df.loc[idx, "proximity_ratio"] = nearest_dist / second_dist if second_dist > 0 else np.nan

    n_leftover_final = int(df["final_roi"].eq("leftover").sum())
    n_prox_assigned  = int(df["proximity_assigned"].sum())
    n_leftover_start = int(leftover_mask.sum())
    print(f"Proximity assignment (cap = {PROXIMITY_MAX_DIST_MM} mm): "
          f"{n_prox_assigned}/{n_leftover_start} leftovers assigned; "
          f"{n_leftover_final} stay leftover (will be NaN in alt_final_roi).")

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
# OVERWRITE BAYLOR MICROWIRE COORDS FROM v2026 ELECTRODE FILES
# =============================================================================
# Baylor microwire bundles have updated MNI152 coords in the v2026 electrode
# CSVs; overwrite here so the ROI atlas lookups below see the corrected
# positions. Non-microwire (sEEG etc.) and non-Baylor rows are untouched.

mw_updates, mw_reliability = load_microwire_updates(
    path_to_microwire_updates_dir)
print(f"\nMicrowire electrode files scanned: {len(mw_reliability)} subjects.")
print(f"Reliability check (file MNI152 vs Fischl-transformed file MNI305; "
      f"cutoff = {MICROWIRE_MNI152_TOL_MM} mm mean):")
print(mw_reliability.round(2).to_string(index=False))
print(f"-> {len(mw_updates)} bundle coords accepted from "
      f"{int(mw_reliability['reliable'].sum())} reliable subjects.")

df["subject_code"] = (
    df["Subject Label"].astype(str).str.strip("'\" ").str.split("-").str[-1])
df["bundle_key"] = df["electrode label"].apply(_bundle_key)

df["MNI_x_pre_microwire"] = df["MNI_x"]
df["MNI_y_pre_microwire"] = df["MNI_y"]
df["MNI_z_pre_microwire"] = df["MNI_z"]
df["microwire_updated"] = False

# Coord provenance for every row (audit column, per user request):
#   'main_table_original'         -> untouched main-table value (post 305->152
#                                    for Baylor rows).
#   'microwire_file_MNI152'       -> overwritten with reliable microwire file.
#   'microwire_file_unreliable'   -> Baylor row whose subject file was flagged
#                                    unreliable; not overwritten.
#   'no_matching_microwire_bundle'-> Baylor row whose bundle wasn't found in
#                                    a reliable file.
unreliable_codes = set(
    mw_reliability.loc[~mw_reliability["reliable"], "subject_code"])

df["microwire_mni_source"] = "main_table_original"

for idx in df.index[baylor_mask]:
    code = df.at[idx, "subject_code"]
    key = (code, df.at[idx, "bundle_key"])
    if key in mw_updates:
        x, y, z = mw_updates[key]
        df.at[idx, "MNI_x"] = x
        df.at[idx, "MNI_y"] = y
        df.at[idx, "MNI_z"] = z
        df.at[idx, "microwire_updated"] = True
        df.at[idx, "microwire_mni_source"] = "microwire_file_MNI152"
    elif code in unreliable_codes:
        df.at[idx, "microwire_mni_source"] = "microwire_file_unreliable"
    else:
        df.at[idx, "microwire_mni_source"] = "no_matching_microwire_bundle"

n_upd = int(df["microwire_updated"].sum())
n_upd_subj = int(df.loc[df["microwire_updated"], "subject_code"].nunique())
print(f"\nOverwrote MNI152 coords for {n_upd} rows across {n_upd_subj} Baylor "
      f"subjects using reliable microwire files.")

src_counts = (df.loc[baylor_mask, "microwire_mni_source"]
              .value_counts(dropna=False))
print(f"Baylor coord-source breakdown:\n{src_counts.to_string()}")


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
# MICROWIRE-UPDATE DIFF REPORT
# =============================================================================
# For every cell whose coords were overwritten from the v2026 microwire files,
# re-run the atlas lookup + initial ROI assignment on the *pre-update* coords
# and print a summary of which cells changed label. This is the first-run
# sanity check requested: expected to move some hippocampal / entorhinal cells.

mw_mask = df_labeled["microwire_updated"]
if mw_mask.any():
    print("\n=== Microwire-update diff (initial ROI, pre vs post overwrite) ===")

    pre_view = df_labeled.loc[mw_mask].copy()
    pre_view["MNI_x"] = pre_view["MNI_x_pre_microwire"]
    pre_view["MNI_y"] = pre_view["MNI_y_pre_microwire"]
    pre_view["MNI_z"] = pre_view["MNI_z_pre_microwire"]

    pre_atlas = pre_view.apply(query_atlases, axis=1)
    for c in pre_atlas.columns:
        pre_view[c] = pre_atlas[c]
    pre_view["initial_roi_pre_microwire"] = pre_view.apply(
        assign_initial_roi, axis=1)

    df_labeled["initial_roi_pre_microwire"] = np.nan
    df_labeled.loc[mw_mask, "initial_roi_pre_microwire"] = (
        pre_view["initial_roi_pre_microwire"].values)

    post = df_labeled.loc[mw_mask, coord_cols].to_numpy(dtype=float)
    pre = df_labeled.loc[mw_mask, ["MNI_x_pre_microwire",
                                   "MNI_y_pre_microwire",
                                   "MNI_z_pre_microwire"]].to_numpy(dtype=float)
    delta = post - pre
    shifts_mm = np.linalg.norm(delta, axis=1)
    df_labeled["microwire_shift_mm"] = np.nan
    df_labeled.loc[mw_mask, "microwire_shift_mm"] = shifts_mm

    n_valid = int(np.sum(~np.isnan(shifts_mm)))
    print(f"Coordinate shift (mm) for {n_valid}/{int(mw_mask.sum())} updated "
          f"cells with valid pre coords: "
          f"mean={np.nanmean(shifts_mm):.2f}, "
          f"median={np.nanmedian(shifts_mm):.2f}, "
          f"max={np.nanmax(shifts_mm):.2f}")
    print(f"Signed component means (post - pre, mm): "
          f"dx={np.nanmean(delta[:,0]):+.2f}, "
          f"dy={np.nanmean(delta[:,1]):+.2f}, "
          f"dz={np.nanmean(delta[:,2]):+.2f}   "
          f"(non-zero -> systematic offset)")

    subj_shift = (df_labeled.loc[mw_mask]
                  .assign(dx=delta[:,0], dy=delta[:,1], dz=delta[:,2],
                          shift_mm=shifts_mm)
                  .groupby("subject_code")
                  .agg(n=("shift_mm", "size"),
                       shift_mean=("shift_mm", "mean"),
                       shift_median=("shift_mm", "median"),
                       shift_max=("shift_mm", "max"),
                       dx_mean=("dx", "mean"),
                       dy_mean=("dy", "mean"),
                       dz_mean=("dz", "mean"))
                  .round(2)
                  .sort_values("shift_mean", ascending=False))
    print("\nPer-subject microwire-vs-old shift "
          "(large mean / large signed dy,dz -> the old coords for that "
          "subject were likely in the wrong space, not a fine refinement):")
    print(subj_shift.to_string())

    changed_mask = (df_labeled.loc[mw_mask, "initial_roi_pre_microwire"]
                    != df_labeled.loc[mw_mask, "final_roi"])
    n_changed = int(changed_mask.sum())
    print(f"{n_changed} of {int(mw_mask.sum())} updated cells changed initial "
          f"ROI label.")

    transitions = (df_labeled.loc[mw_mask]
                   .assign(_changed=changed_mask.values)
                   .query("_changed")
                   .groupby(["initial_roi_pre_microwire", "final_roi"],
                            dropna=False)
                   .size().reset_index(name="n_cells")
                   .sort_values("n_cells", ascending=False))
    if not transitions.empty:
        print("\nTransitions pooled across subjects (pre -> post):")
        print(transitions.to_string(index=False))

        # Per-subject label-change count with mean shift (to distinguish
        # small-shift atlas-boundary flips from large-shift corrections).
        per_subj = (df_labeled.loc[mw_mask]
                    .assign(_changed=changed_mask.values,
                            shift_mm=shifts_mm)
                    .groupby("subject_code")
                    .agg(n_updated=("_changed", "size"),
                         n_changed_label=("_changed", "sum"),
                         mean_shift_mm=("shift_mm", "mean"))
                    .round(2)
                    .sort_values("n_changed_label", ascending=False))
        print("\nPer-subject label-change count & mean shift:")
        print(per_subj.to_string())

        # Transitions per subject for the top 3 movers (helps see e.g. YEN's
        # broken hemisphere jump vs. YFP's normal HC/EC boundary refinements).
        top_movers = per_subj.head(3).index.tolist()
        for code in top_movers:
            sub = (df_labeled.loc[mw_mask]
                   .assign(_changed=changed_mask.values,
                           shift_mm=shifts_mm)
                   .query("subject_code == @code and _changed"))
            if sub.empty:
                continue
            print(f"\nTop mover — {code} (n changed = {len(sub)}, "
                  f"mean shift = {sub['shift_mm'].mean():.1f} mm):")
            t = (sub.groupby(["initial_roi_pre_microwire", "final_roi"],
                              dropna=False)
                    .size().reset_index(name="n_cells")
                    .sort_values("n_cells", ascending=False))
            print(t.to_string(index=False))


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


def plot_roi_count_overview(counts_df, roi_field, title, save_path,
                            colour_map=ROI_COLOURS):
    """Bar plot of cell counts per ROI; each x-label carries both the
    cell count and the number of contributing subjects. Bar colours come
    from the project-wide ROI palette (`colour_map`, default ROI_COLOURS).
    Kept here as a data summary (no spatial brain plotting)."""
    labels = [f"{roi_display(r)}\n(n = {n} cells; {s} subjects)"
              for r, n, s in zip(counts_df[roi_field],
                                 counts_df["n_cells"],
                                 counts_df["n_subjects"])]
    bar_colours = [colour_map.get(r, "#888888") for r in counts_df[roi_field]]
    plt.figure(figsize=(11, 5))
    plt.bar(labels, counts_df["n_cells"], color=bar_colours,
            edgecolor="black", linewidth=0.4)
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
