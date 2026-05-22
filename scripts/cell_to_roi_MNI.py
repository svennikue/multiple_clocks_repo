#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn import datasets, plotting
from nilearn.image import new_img_like

import matplotlib.cm as cm
import matplotlib.colors as mcolors

import plotly.graph_objects as go
from skimage.measure import marching_cubes


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
acc_y_cutoff = 10   # only fairly anterior cingulate counts as ACC
marker_size_glassbrain = 18     # smaller dots
marker_size_3d = 4              # smaller dots in 3D
jitter_duplicate_points = True  # helps show multiple cells at identical coordinates
jitter_mm = 0.7                 # visual-only jitter, in mm

roi_order = [
    "EC",
    "Parahippocampal",
    "HC_anterior",
    "HC_posterior",
    "ventral_ACC",
    "ACC",
    "medial_CC",
    "posterior_PCC",
    "OFC11",
    "OFC13",
    "Visual"
]

# Alternative labelling (see assign_alt_roi): ACC is split by `acc_y_cutoff`
# into ACC / mACC, and OFC11 + OFC13 + ventral_ACC collapse into medialOFC.
alt_roi_order = [
    "EC",
    "Parahippocampal",
    "HC_anterior",
    "HC_posterior",
    "medialOFC",
    "ACC",
    "mACC",
    "medial_CC",
    "posterior_PCC",
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
    return "HC_anterior" if float(y) >= hc_ap_cutoff else "HC_posterior"


def get_img(atlas_or_img):
    if hasattr(atlas_or_img, "maps"):
        img = atlas_or_img.maps
    else:
        img = atlas_or_img
    return nib.load(img) if isinstance(img, str) else img


def make_subject_colors(subjects):
    subjects = sorted(pd.Series(subjects).astype(str).unique())
    cmap = cm.get_cmap("tab20", max(len(subjects), 1))
    return {sub: mcolors.to_hex(cmap(i)) for i, sub in enumerate(subjects)}


def add_visual_jitter(df_in):
    """
    Adds tiny deterministic offsets to duplicate coordinates so every cell is visible.
    This is only for plotting; original MNI coordinates remain unchanged.
    """
    df = df_in.copy()
    df[["plot_x", "plot_y", "plot_z"]] = df[["MNI_x", "MNI_y", "MNI_z"]]

    if not jitter_duplicate_points:
        return df

    grouped = df.groupby(["MNI_x", "MNI_y", "MNI_z"], sort=False)

    for _, idxs in grouped.groups.items():
        idxs = list(idxs)
        n = len(idxs)

        if n <= 1:
            continue

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

        df.loc[idxs, "plot_x"] += jitter_mm * np.cos(angles)
        df.loc[idxs, "plot_y"] += jitter_mm * np.sin(angles)

    return df


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
            return "posterior_PCC"
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
    
    # if contains_any(brainnetome, ["a32sg", "a32p", "a24rv"]):
    #     return "ACC" if y >= acc_y_cutoff else "medial_CC"

    # if contains(ho_cort, "cingulate gyrus, anterior division"):
    #     return "ACC" if y >= acc_y_cutoff else "medial_CC"
    if contains_any(brainnetome, ["a32sg", "a32p", "a24rv"]):
        return "ACC"

    if contains(ho_cort, "cingulate gyrus, anterior division"):
        return "ACC"

    if contains(brainnetome, "a23"):
        return "posterior_PCC"

    if contains(ho_cort, "cingulate gyrus, posterior division"):
        return "posterior_PCC"

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
         MNI_y >= cutoff stays 'ACC'; more posterior cingulate -> 'mACC'.
      2. OFC11, OFC13 and ventral_ACC collapse into a single 'medialOFC'.
    """
    roi = row["final_roi"]
    if roi == "ACC":
        y = row["MNI_y"]
        if pd.isna(y):
            return "ACC"
        return "ACC" if float(y) >= acc_y_cutoff else "mACC"
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

required_cols = ["subject", "cell idx", "electrode label", "MNI_x", "MNI_y", "MNI_z"]
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

print("\nAlternative ROI counts (alt_final_roi):")
print(df_labeled["alt_final_roi"].value_counts(dropna=False))


# =============================================================================
# COUNTS AND BAR PLOTS
# =============================================================================

def make_roi_count_table(series, order, roi_field):
    """Cell-count table for one ROI column, ordered like `order`."""
    return (series.value_counts()
            .reindex(order, fill_value=0)
            .rename_axis(roi_field)
            .reset_index(name="n_cells"))


def plot_roi_count_overview(counts_df, roi_field, title, save_path):
    """Bar plot of cell counts per ROI; the count is written into each
    x-label, e.g. 'ACC (n = 68 cells)'."""
    labels = [f"{r} (n = {n} cells)"
              for r, n in zip(counts_df[roi_field], counts_df["n_cells"])]
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
    df_labeled["final_roi"], roi_order, "final_roi")
alt_roi_counts = make_roi_count_table(
    df_labeled["alt_final_roi"], alt_roi_order, "alt_final_roi")

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
# ROI MASKS FOR PLOTTING
# =============================================================================

def label_indices_containing(labels, patterns):
    idxs = []
    for i, lab in enumerate(labels):
        if any(p.lower() in str(lab).lower() for p in patterns):
            idxs.append(i)
    return idxs


def make_mask_from_atlas(atlas, patterns):
    img = get_img(atlas)
    data = img.get_fdata()
    idxs = label_indices_containing(atlas.labels, patterns)

    mask = np.isin(data, idxs).astype(float) if idxs else np.zeros(data.shape)
    return new_img_like(img, mask)


def make_brainnetome_mask(patterns):
    img = brainnetome_lookup.img
    data = brainnetome_lookup.data

    idxs = []
    for idx, lab in brainnetome_lookup.labels.items():
        if any(p.lower() in str(lab).lower() for p in patterns):
            idxs.append(idx)

    mask = np.isin(data, idxs).astype(float) if idxs else np.zeros(data.shape)
    return new_img_like(img, mask)


def make_hc_mask(ap="both"):
    img = get_img(ho_sub)
    data = img.get_fdata()

    hc_idxs = label_indices_containing(ho_sub.labels, ["hippocampus"])
    mask = np.isin(data, hc_idxs)

    coords = np.indices(data.shape).reshape(3, -1).T
    mni_coords = nib.affines.apply_affine(img.affine, coords)
    y_coords = mni_coords[:, 1].reshape(data.shape)

    if ap == "anterior":
        mask = mask & (y_coords >= hc_ap_cutoff)
    elif ap == "posterior":
        mask = mask & (y_coords < hc_ap_cutoff)

    return new_img_like(img, mask.astype(float))


def make_roi_mask(final_roi):
    if final_roi == "EC":
        return make_mask_from_atlas(juelich, ["entorhinal"])

    if final_roi == "Parahippocampal":
        return make_mask_from_atlas(ho_cort, ["parahippocampal"])

    if final_roi == "HC_anterior":
        return make_hc_mask(ap="anterior")

    if final_roi == "HC_posterior":
        return make_hc_mask(ap="posterior")

    if final_roi == "ventral_ACC":
        return make_brainnetome_mask(["a14m"])

    if final_roi == "ACC":
        return make_mask_from_atlas(ho_cort, ["cingulate gyrus, anterior division"])

    if final_roi == "medial_PCC":
        return make_mask_from_atlas(ho_cort, ["cingulate gyrus, posterior division"])

    if final_roi == "posterior_PCC":
        return make_mask_from_atlas(
            ho_cort,
            ["cingulate gyrus, posterior division", "precuneous", "precuneus"]
        )

    if final_roi == "OFC11":
        return make_brainnetome_mask(["a11m"])

    if final_roi == "OFC13":
        return make_brainnetome_mask(["a13"])
    
    if final_roi == "Visual":
        return make_mask_from_atlas(
            ho_cort,
            [
                "occipital",
                "cuneal",
                "lingual",
                "intracalcarine",
                "supracalcarine",
                "occipital pole",
            ],
        )
    return None


# =============================================================================
# GLASS-BRAIN PLOTS
# =============================================================================

def plot_roi_glassbrain(df_in, roi, out_dir, roi_col="final_roi"):
    roi_df = df_in[df_in[roi_col] == roi].copy()

    if roi_df.empty:
        print(f"Skipping glass brain {roi}: no cells.")
        return

    roi_df = add_visual_jitter(roi_df)

    coords = roi_df[["plot_x", "plot_y", "plot_z"]].to_numpy()
    colors = make_subject_colors(roi_df["subject"])
    marker_colors = [colors[str(s)] for s in roi_df["subject"]]

    display = plotting.plot_glass_brain(
        None,
        display_mode="lyrz",
        title=f"{roi} | n = {len(roi_df)}",
        black_bg=False,
        plot_abs=False,
    )

    roi_mask = make_roi_mask(roi)
    if roi_mask is not None:
        display.add_overlay(roi_mask, threshold=0.5, alpha=0.35, cmap="autumn")

    display.add_markers(
        coords,
        marker_color=marker_colors,
        marker_size=marker_size_glassbrain,
    )

    fig = plt.gcf()

    handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=col,
            markeredgecolor=col,
            label=f"sub-{sub}",
        )
        for sub, col in colors.items()
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 6),
        frameon=False,
    )

    path = os.path.join(out_dir, f"glassbrain_{roi}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved {path}")


# =============================================================================
# 3D ROI MESH PLOTS
# =============================================================================

def mask_to_mesh(mask_img, step_size=1):
    data = mask_img.get_fdata() > 0

    if data.sum() == 0:
        return None

    data = np.pad(data.astype(float), pad_width=1, mode="constant")

    affine = mask_img.affine.copy()
    affine[:3, 3] -= mask_img.affine[:3, :3].sum(axis=1)

    verts, faces, _, _ = marching_cubes(
        data,
        level=0.5,
        step_size=step_size,
    )

    verts_mni = nib.affines.apply_affine(affine, verts)
    x, y, z = verts_mni.T
    i, j, k = faces.T

    return x, y, z, i, j, k


def make_brain_background_traces(brain_mesh):
    """Standard background traces shared by every mesh3d plot: a transparent
    whole-brain shell, with the hippocampus and ACC shaded grey and the
    entorhinal cortex shaded dark grey inside it."""
    traces = []

    if brain_mesh is not None:
        bx, by, bz, bi, bj, bk = brain_mesh
        traces.append(go.Mesh3d(
            x=bx, y=by, z=bz, i=bi, j=bj, k=bk,
            color="lightgray", opacity=0.06, name="brain",
            hoverinfo="skip", showlegend=True,
        ))

    for mask_img, color, opacity, name in [
        (make_hc_mask(ap="both"),                       "gray",    0.30, "hippocampus"),
        (make_mask_from_atlas(juelich, ["entorhinal"]), "dimgray", 0.50, "EC"),
        (make_mask_from_atlas(
            ho_cort, ["cingulate gyrus, anterior division"]),
                                                        "gray",    0.30, "ACC"),
    ]:
        mesh = mask_to_mesh(mask_img, step_size=1)
        if mesh is None:
            continue
        mx, my, mz, mi, mj, mk = mesh
        traces.append(go.Mesh3d(
            x=mx, y=my, z=mz, i=mi, j=mj, k=mk,
            color=color, opacity=opacity, name=name,
            hoverinfo="skip", showlegend=True,
        ))
    return traces


def plot_roi_3d_mesh(df_in, roi, out_dir, roi_col="final_roi", bg_fig=None):
    """Per-ROI 3-D plot: the standard brain background (transparent shell +
    grey hippocampus / ACC + dark-grey EC) with this ROI's electrodes."""
    roi_df = df_in[df_in[roi_col] == roi].copy()

    if roi_df.empty:
        print(f"Skipping 3D mesh {roi}: no cells.")
        return

    roi_df = add_visual_jitter(roi_df)

    colors = make_subject_colors(roi_df["subject"])
    marker_colors = [colors[str(s)] for s in roi_df["subject"]]

    hover = (
        "subject: " + roi_df["subject"].astype(str)
        + "<br>cell: " + roi_df["cell idx"].astype(str)
        + "<br>electrode: " + roi_df["electrode label"].astype(str)
        + "<br>MNI: "
        + roi_df["MNI_x"].round(2).astype(str) + ", "
        + roi_df["MNI_y"].round(2).astype(str) + ", "
        + roi_df["MNI_z"].round(2).astype(str)
    )

    # Start from a copy of the shared brain background.
    fig = go.Figure(bg_fig) if bg_fig is not None else go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=roi_df["plot_x"],
            y=roi_df["plot_y"],
            z=roi_df["plot_z"],
            mode="markers",
            marker=dict(
                size=marker_size_3d,
                color=marker_colors,
                opacity=0.95,
            ),
            hovertext=hover,
            hoverinfo="text",
            name="electrodes",
        )
    )

    fig.update_layout(
        title=f"{roi} | n = {len(roi_df)}",
        width=900,
        height=750,
        scene=dict(
            xaxis_title="MNI x",
            yaxis_title="MNI y",
            zaxis_title="MNI z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    path = os.path.join(out_dir, f"mesh3d_{roi}.html")
    fig.write_html(path)
    print(f"Saved {path}")

    fig.show()


# =============================================================================
# COMBINED / WHOLE-BRAIN PLOTS
# =============================================================================

def make_roi_color_map(rois):
    rois = list(rois)
    cmap = cm.get_cmap("tab20", max(len(rois), 1))
    return {roi: mcolors.to_hex(cmap(i)) for i, roi in enumerate(rois)}


def plot_all_rois_glassbrain_mosaic(df_in, roi_col, order, save_path):
    """All per-ROI glass-brains drawn as panels on a single figure.

    Mirrors the per-ROI standalone style (colour by subject, ROI region
    as an overlay) but uses a compact 'ortho' display per panel so the
    whole set fits in one figure.
    """
    rois_with_data = [r for r in order
                      if not df_in[df_in[roi_col] == r].empty]
    if not rois_with_data:
        print("Skipping all-ROI mosaic: no ROIs have data.")
        return

    n = len(rois_with_data)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.5, nrows * 2.6),
        squeeze=False,
    )

    for ax_idx, roi in enumerate(rois_with_data):
        ax = axes[ax_idx // ncols, ax_idx % ncols]
        roi_df = df_in[df_in[roi_col] == roi].copy()
        roi_df = add_visual_jitter(roi_df)
        coords = roi_df[["plot_x", "plot_y", "plot_z"]].to_numpy()
        colors = make_subject_colors(roi_df["subject"])
        marker_colors = [colors[str(s)] for s in roi_df["subject"]]

        display = plotting.plot_glass_brain(
            None, display_mode="ortho", axes=ax,
            black_bg=False, plot_abs=False,
        )
        roi_mask = make_roi_mask(roi)
        if roi_mask is not None:
            display.add_overlay(roi_mask, threshold=0.5, alpha=0.35,
                                cmap="autumn")
        display.add_markers(
            coords, marker_color=marker_colors,
            marker_size=marker_size_glassbrain * 0.45,
        )
        ax.set_title(f"{roi}  |  n={len(roi_df)}", fontsize=10)

    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")

    fig.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved {save_path}")


def plot_all_cells_single_glassbrain(df_in, roi_col, order, roi_colors,
                                     save_path):
    """Standard 4-view glass-brain showing every cell, colour-coded by ROI."""
    df = df_in[df_in[roi_col].isin(order)].copy()
    if df.empty:
        print("Skipping single all-cells glass-brain: no data.")
        return
    df = add_visual_jitter(df)
    marker_colors = [roi_colors[r] for r in df[roi_col]]
    coords = df[["plot_x", "plot_y", "plot_z"]].to_numpy()

    display = plotting.plot_glass_brain(
        None, display_mode="lyrz",
        title=f"All cells coloured by {roi_col}  |  n = {len(df)}",
        black_bg=False, plot_abs=False,
    )
    display.add_markers(
        coords, marker_color=marker_colors,
        marker_size=max(6, int(marker_size_glassbrain * 0.55)),
    )

    fig = plt.gcf()
    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="",
            markersize=6,
            markerfacecolor=roi_colors[roi],
            markeredgecolor=roi_colors[roi],
            label=f"{roi} (n={int((df[roi_col] == roi).sum())})",
        )
        for roi in order
        if (df[roi_col] == roi).any()
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=min(len(handles), 6), frameon=False, fontsize=8,
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved {save_path}")


def _load_mni152_brain_mesh(step_size=2):
    """Whole-brain MNI152 mask, converted to a marching-cubes mesh.

    Returns (x, y, z, i, j, k) or None.
    """
    try:
        brain_mask_img = datasets.load_mni152_brain_mask()
    except Exception as e:
        print(f"Could not load MNI152 brain mask: {e}")
        return None
    return mask_to_mesh(brain_mask_img, step_size=step_size)


def plot_all_cells_3d_full_brain(df_in, roi_col, order, roi_colors,
                                 bg_fig, save_path):
    """Standard brain background + every cell, coloured by ROI.

    One Scatter3d trace per ROI (shade from `roi_colors`), so different
    recording sites stay visually separable inside the brain.
    """
    df = df_in[df_in[roi_col].isin(order)].copy()
    if df.empty:
        print("Skipping all-cells 3-D plot: no data.")
        return
    df = add_visual_jitter(df)

    fig = go.Figure(bg_fig) if bg_fig is not None else go.Figure()

    for roi in order:
        sub = df[df[roi_col] == roi]
        if sub.empty:
            continue
        hover = (
            "subject: " + sub["subject"].astype(str)
            + "<br>cell: " + sub["cell idx"].astype(str)
            + "<br>electrode: " + sub["electrode label"].astype(str)
            + "<br>roi: " + roi
            + "<br>MNI: "
            + sub["MNI_x"].round(2).astype(str) + ", "
            + sub["MNI_y"].round(2).astype(str) + ", "
            + sub["MNI_z"].round(2).astype(str)
        )
        fig.add_trace(go.Scatter3d(
            x=sub["plot_x"], y=sub["plot_y"], z=sub["plot_z"],
            mode="markers",
            marker=dict(size=marker_size_3d,
                        color=roi_colors[roi],
                        opacity=0.95),
            hovertext=hover, hoverinfo="text",
            name=f"{roi} (n={len(sub)})",
        ))

    fig.update_layout(
        title=f"All cells in MNI brain ({roi_col})  |  n = {len(df)}",
        width=950, height=800,
        scene=dict(
            xaxis_title="MNI x", yaxis_title="MNI y", zaxis_title="MNI z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(save_path)
    print(f"Saved {save_path}")
    fig.show()


# =============================================================================
# MAKE PLOTS
# =============================================================================

glassbrain_out_dir = os.path.join(output_dir, "glassbrain_roi_plots")
mesh3d_out_dir = os.path.join(output_dir, "mesh3d_roi_plots")

os.makedirs(glassbrain_out_dir, exist_ok=True)
os.makedirs(mesh3d_out_dir, exist_ok=True)

# Standard mesh3d background, built once: transparent whole-brain shell +
# grey hippocampus + dark-grey EC + grey ACC.
print("Loading MNI152 brain mesh ...")
brain_mesh = _load_mni152_brain_mesh(step_size=2)
print("Building standard brain background ...")
BRAIN_BG_FIG = go.Figure(data=make_brain_background_traces(brain_mesh))

# Every figure is produced twice — once per ROI labelling — into its own
# sub-folder so the 'final_roi' and 'alt_final_roi' versions never collide.
for roi_col, order in [("final_roi", roi_order),
                       ("alt_final_roi", alt_roi_order)]:
    print(f"\n=== Plots for {roi_col} ===")
    gb_dir = os.path.join(glassbrain_out_dir, roi_col)
    m3_dir = os.path.join(mesh3d_out_dir, roi_col)
    os.makedirs(gb_dir, exist_ok=True)
    os.makedirs(m3_dir, exist_ok=True)

    df_for_plots = df_labeled[df_labeled[roi_col].isin(order)].copy()
    roi_colors = make_roi_color_map(order)

    plot_all_rois_glassbrain_mosaic(
        df_for_plots, roi_col, order,
        save_path=os.path.join(gb_dir, "glassbrain_all_rois_mosaic.png"),
    )
    plot_all_cells_single_glassbrain(
        df_for_plots, roi_col, order, roi_colors,
        save_path=os.path.join(gb_dir, "glassbrain_all_cells_by_roi.png"),
    )
    plot_all_cells_3d_full_brain(
        df_for_plots, roi_col, order, roi_colors, BRAIN_BG_FIG,
        save_path=os.path.join(m3_dir, "mesh3d_all_cells_in_brain.html"),
    )

    for roi in order:
        plot_roi_glassbrain(df_for_plots, roi, gb_dir, roi_col=roi_col)
        plot_roi_3d_mesh(df_for_plots, roi, m3_dir, roi_col=roi_col,
                         bg_fig=BRAIN_BG_FIG)


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
print(glassbrain_out_dir)
print(mesh3d_out_dir)