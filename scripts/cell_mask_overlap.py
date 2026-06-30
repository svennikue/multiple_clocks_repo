#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell / fMRI cluster anatomical overlap.

Builds a binary "DSR main effect" mask by thresholding the cluster-mass
PALM FWE map at p < 0.05 (the file stores 1-p values, so the cut is
> 0.95), then checks how many recorded cells from
`neurons_with_final_roi_labels.csv` fall inside that mask.

For both the dorsomedial-PFC (ACC) cluster and the lateral-OFC cluster
that emerge from connected-component labelling of the mask, we report:
    * cells inside / outside the cluster, split by `alt_final_roi`.
    * overlap of the matching ROI label with that cluster
      (ACC × ACC cluster; medialOFC × lOFC cluster).
    * total cells inside / outside the full thresholded mask.

Per cell subset (all / dsr_rsa / not_dsr_rsa) we render an
`mne.viz.Brain` figure (Showgirl2 palette, same style as
scripts/roi_brain_visualization.py) with cells inside the mask coloured
by ROI and cells outside drawn in grey.

Outputs land under
    <DATA_DIR>/cell_mask_overlap/<subset>/...
along with a `counts.csv` summarising the overlap per subset.

@author: Svenja Kuchenhoff
"""

import json
import os
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.affines import apply_affine
from scipy.ndimage import label as nd_label

try:
    import era_brewer
    HAS_ERA_BREWER = True
except Exception:
    HAS_ERA_BREWER = False

try:
    from mne.datasets import fetch_fsaverage
    from mne.viz import Brain
    HAS_MNE_BRAIN = True
except Exception:
    HAS_MNE_BRAIN = False


# =============================================================================
# SETTINGS
# =============================================================================

DATA_DIR_EPHYS = (
    "/Users/xpsy1114/Documents/projects/multiple_clocks/"
    "data/ephys_humans/derivatives"
)
MASK_DIR = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/masks"
TSTAT_FWEP_PATH = (
    "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/"
    "group/Main_Results_fMRI/"
    "RSA_quarters_DSR_controls_glmbase_all-paths-fixed_stickrews_split-"
    "buttons_smooth5_palm_p0_01/"
    "cropped_masked_smooth_fwhm5_DSR-DSR-contr_except_prev_but-mask_"
    "reward-path_beta_std_clusterm_tstat_fwep_c1.nii"
)
DSR_MAIN_EFFECT_MASK_PATH = os.path.join(
    MASK_DIR, "DSR_main_effect_mask.nii.gz")
GRADIENT_MASK_PATH = os.path.join(MASK_DIR, "gradient_mask_bin.nii.gz")

# Each entry: short name → path to the binary mask (built on the fly for
# the DSR main effect if missing).
INPUT_MASKS = {
    "DSR_main_effect": DSR_MAIN_EFFECT_MASK_PATH,
    # (the pre-built `gradient` mask is no longer plotted here — the
    #  new yellow→dark-red rebuild lives in the GRADIENT BLOB section.)
}

CELL_TABLE_PATH = os.path.join(
    DATA_DIR_EPHYS, "neurons_with_final_roi_labels.csv")
DSR_JSON_PATH = os.path.join(
    DATA_DIR_EPHYS, "all_sessions_dsrRSA_grouping_summary.json")

# PALM cluster-mass FWE map stores 1-p, so p < 0.05  ⇔  value > 0.95.
P_THRESHOLD = 0.05
VALUE_THRESHOLD = 1.0 - P_THRESHOLD

# Sub-clusters of interest (selected by anatomical centre-of-mass after
# connected-component labelling).  Anything smaller than this is treated
# as a stray voxel and ignored when identifying the two main clusters.
MIN_CLUSTER_VOXELS = 50

ROI_COL = "alt_final_roi"
ROI_ORDER = [
    "ACC", "medial_CC",
    "HC_anterior", "HC_mid",
    "EC", "Parahippocampal",
    "PCC",
    "medialOFC",
    "Visual",
]

PALETTE_NAME = "Showgirl2"
GREY_HEX = "#bdbdbd"
MASK_OVERLAY_HEX = "#ff8c1a"   # orange used for the surface cluster shading.

MNE_FOCI_SCALE_IN = 0.45       # cells inside the mask
MNE_FOCI_SCALE_OUT = 0.45      # SAME size as `_IN`, per user request
MNE_FOCI_JITTER_MM = 2.5       # 3-D jitter so co-located cells separate
MNE_SURF_ALPHA = 0.30
MNE_VIEWS_BOTH_HEMI = ("lateral", "dorsal")
MNE_VIEWS_SOLO_HEMI = ("medial", "lateral")
MNE_HEMIS_SOLO = ("lh", "rh")
MNE_FILTER_FOCI_BY_HEMI = True

CELL_SUBSETS = ["all", "dsr_rsa", "not_dsr_rsa"]

OUTPUT_ROOT = os.path.join(DATA_DIR_EPHYS, "cell_mask_overlap")


# =============================================================================
# GRADIENT OVERLAP (separate figure)
#
# Mirrors figure 1c of the manuscript draft: per-condition mPFC blobs in
# yellow→dark-red along the future-lag gradient, with overlapping cells
# drawn in the ACC era_brewer colour.
# =============================================================================
GRADIENT_TSTAT_DIR = (
    "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/"
    "group/Main_Results_fMRI/"
    "RSA_which-fut-isin-DSR_glmbase_all-paths-fixed_stickrews_split-buttons_"
    "smooth5_palm_masked_p0_01"
)
GRADIENT_TSTAT_MAPS = {  # display name → filename inside GRADIENT_TSTAT_DIR
    "current": "CURR_QUARTER-split_quarters_DSR_masked_vox_tstat_c1.nii.gz",
    "next":    "NEXT_QUARTER-split_quarters_DSR_masked_vox_tstat_c1.nii.gz",
    "next+2":  "NEXT2_QUARTER-split_quarters_DSR_masked_vox_tstat_c1.nii.gz",
    "next+3":  "NEXT3_QUARTER-split_quarters_DSR_masked_vox_tstat_c1.nii.gz",
}
# Per-condition voxel-wise t threshold for the per-map blob. The four
# blobs are then UNIONed into a single "gradient" mask which is rendered
# with a smooth yellow→dark-red ramp along anterior–posterior (see
# `GRADIENT_RAMP_HEX` below). Looser thresholds = bigger union = more
# cells overlapping.
GRADIENT_TSTAT_THRESHOLDS = {
    "current": 2.00,
    "next":    2.00,
    "next+2":  1.70,
    "next+3":  1.50,
}
# Alternative: if you already have a pre-built binary mask of the
# gradient, point this at it and we'll skip rebuilding from the 4 maps.
# Set to None to always rebuild from `GRADIENT_TSTAT_MAPS`.
GRADIENT_PREBUILT_MASK     = None      # e.g. GRADIENT_MASK_PATH
# Smooth ramp ALONG DORSAL–VENTRAL (MNI z):
#   dorsal tip  (high z) = bright yellow  — "distant future"
#   ventral end (low z)  = very dark red  — "next action"
# Order is low z → high z, so first entry maps to fmin, last to fmax.
# GRADIENT_RAMP_HEX          = ["#3D0000", "#8B0000", "#D2691E", "#FF8C00", "#FCE300"]
GRADIENT_RAMP_HEX = ["#FCE300", "#FF8C00", "#D2691E", "#8B0000", "#3D0000"]
# Restrict the rendering to medial-view L+R.
GRADIENT_VIEW              = "medial"
GRADIENT_HEMIS             = ("lh", "rh")
GRADIENT_OVERLAY_ALPHA     = 0.70
# Cell dot scales — SAME size for grey and ACC dots, with a small 3D
# jitter so co-located cells visually separate.
GRADIENT_CELL_SCALE        = 0.35
GRADIENT_CELL_JITTER_MM    = 2.5
GRADIENT_BRAIN_SIZE        = (1400, 1200)
GRADIENT_OUT_SUBDIR        = "gradient_blobs"

# Coronal section for the DSR main effect / lOFC overlap.
DSR_CORONAL_ENABLED        = True
DSR_CORONAL_PINK_HEX       = "#BF567F"     # era_brewer Lover2 index 0 (n=5)
DSR_CORONAL_GREY_HEX       = "#bdbdbd"
DSR_CORONAL_MARKER_SIZE    = 90            # nilearn add_markers scale
DSR_CORONAL_JITTER_MM      = 2.5
DSR_CORONAL_Y_TOL_MM       = 8.0           # cells within ±this of the cut


# =============================================================================
# MASK CONSTRUCTION
# =============================================================================

def build_dsr_main_effect_mask(tstat_path, out_path, value_threshold):
    """Threshold the cluster-mass FWE map and save a binary mask.

    The PALM map stores 1-p_FWE, so a p<0.05 cut is `> 0.95`.
    """
    print(f"Loading FWE-p map: {tstat_path}")
    img = nib.load(tstat_path)
    data = img.get_fdata()
    mask = (data > value_threshold).astype(np.uint8)
    print(f"  threshold > {value_threshold:.3f}  ⇒  "
          f"{int(mask.sum())} voxels above threshold "
          f"(p < {1 - value_threshold:.3f}).")
    out_img = nib.Nifti1Image(mask, img.affine, img.header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, out_path)
    print(f"  wrote {out_path}")
    return out_img


def identify_subclusters(mask_img, min_voxels=MIN_CLUSTER_VOXELS):
    """Connected-component label the mask, return ACC & lOFC sub-masks.

    The two clusters are picked from components with > `min_voxels`
    voxels by their centre-of-mass location:
        * ACC cluster: dorsomedial — |x| < 20, z > 5
        * lOFC cluster: ventrolateral — z < 0

    Returns a dict ``{"ACC": Nifti1Image, "lOFC": Nifti1Image,
                       "full": Nifti1Image}``.
    """
    data = mask_img.get_fdata().astype(np.uint8)
    aff = mask_img.affine
    labels, n = nd_label(data)
    print(f"  connected-component label → {n} components")

    cluster_info = []
    for k in range(1, n + 1):
        size = int((labels == k).sum())
        if size < min_voxels:
            continue
        idx = np.argwhere(labels == k)
        com_vox = idx.mean(axis=0)
        com_mni = apply_affine(aff, com_vox)
        cluster_info.append((k, size, com_mni))
        print(f"    cluster {k}: {size} voxels, "
              f"COM MNI = ({com_mni[0]:.1f}, {com_mni[1]:.1f}, "
              f"{com_mni[2]:.1f})")

    # Pick the largest cluster matching each anatomical signature.
    acc_cands = [(k, s) for k, s, com in cluster_info
                 if com[2] > 5 and abs(com[0]) < 20]
    ofc_cands = [(k, s) for k, s, com in cluster_info if com[2] < 0]
    acc_label = max(acc_cands, key=lambda t: t[1])[0] if acc_cands else None
    ofc_label = max(ofc_cands, key=lambda t: t[1])[0] if ofc_cands else None

    print(f"  selected ACC cluster   = label {acc_label}")
    print(f"  selected lOFC cluster  = label {ofc_label}")

    out = {"full": mask_img}
    for name, k in (("ACC", acc_label), ("lOFC", ofc_label)):
        if k is None:
            print(f"  no sub-cluster identified for {name} "
                  f"(skipping — full mask is the only one for this input).")
            continue
        sub = (labels == k).astype(np.uint8)
        out[name] = nib.Nifti1Image(sub, aff, mask_img.header)
    return out


def load_input_mask(name, path):
    """Load a binary mask. Builds the DSR main-effect mask on demand from
    the cluster-mass FWE map; everything else is loaded as-is."""
    if name == "DSR_main_effect" and not os.path.isfile(path):
        build_dsr_main_effect_mask(
            TSTAT_FWEP_PATH, path, VALUE_THRESHOLD)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Mask {name!r} missing: {path}")
    print(f"Loading {name} mask: {path}")
    return nib.load(path)


# =============================================================================
# CELL HANDLING
# =============================================================================

def load_dsr_subjects():
    if not os.path.isfile(DSR_JSON_PATH):
        return None
    with open(DSR_JSON_PATH) as f:
        d = json.load(f)
    return {str(k).zfill(2) for k in d.keys()}


def get_cell_subset(df, subset, dsr_subjects):
    """Return (subset_df, descriptive_label)."""
    if subset == "all":
        return df.copy(), f"all cells (n={len(df)})"
    is_rsa = df["subject"].astype(str).str.zfill(2).isin(
        dsr_subjects or set())
    if subset == "dsr_rsa":
        sub = df[is_rsa].copy()
        return sub, (f"DSR RSA cells "
                     f"(n={len(sub)}, "
                     f"{sub['subject'].nunique()} subjects)")
    if subset == "not_dsr_rsa":
        sub = df[~is_rsa].copy()
        return sub, (f"non-RSA cells "
                     f"(n={len(sub)}, "
                     f"{sub['subject'].nunique()} subjects)")
    raise ValueError(f"unknown subset: {subset!r}")


def voxel_inside_mask(coords_mm, mask_img):
    """coords_mm: (N, 3) array in MNI mm. Returns (N,) bool."""
    data = mask_img.get_fdata()
    inv = np.linalg.inv(mask_img.affine)
    shape = np.array(data.shape)
    out = np.zeros(len(coords_mm), dtype=bool)
    for i, c in enumerate(coords_mm):
        v = np.round(apply_affine(inv, c)).astype(int)
        if (v >= 0).all() and (v < shape).all():
            out[i] = bool(data[tuple(v)] > 0)
    return out


# =============================================================================
# COUNT TABLE
# =============================================================================

def overlap_counts(df, masks, roi_col):
    """Long-format DataFrame of overlap counts per (mask, ROI)."""
    rows = []
    for mask_name, mask_img in masks.items():
        coords = df[["MNI_x", "MNI_y", "MNI_z"]].to_numpy(dtype=float)
        inside = voxel_inside_mask(coords, mask_img)
        rows.append({
            "mask": mask_name, "roi": "<all>",
            "n_inside": int(inside.sum()),
            "n_outside": int((~inside).sum()),
            "n_total": int(len(inside)),
        })
        for roi in df[roi_col].dropna().unique():
            sel = df[roi_col].to_numpy() == roi
            inside_roi = inside & sel
            rows.append({
                "mask": mask_name, "roi": roi,
                "n_inside": int(inside_roi.sum()),
                "n_outside": int((sel & ~inside).sum()),
                "n_total": int(sel.sum()),
            })
    return pd.DataFrame(rows)


# =============================================================================
# PALETTE
# =============================================================================

def make_roi_palette(rois, palette_name=PALETTE_NAME):
    if HAS_ERA_BREWER:
        try:
            colors = era_brewer.era_brew(
                palette_name, n=max(len(rois), 2),
                brew_type="discrete")
            colors = [mcolors.to_hex(c) for c in colors]
        except Exception as e:
            print(f"  [era_brewer] {palette_name!r} failed: {e}")
            colors = None
    else:
        colors = None
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [mcolors.to_hex(cmap(i % 10)) for i in range(len(rois))]
    return {r: colors[i % len(colors)] for i, r in enumerate(rois)}


# =============================================================================
# MNE BRAIN PLOTTING (in/out of mask)
# =============================================================================

_FSAVERAGE_PATH = None


def _ensure_fsaverage():
    global _FSAVERAGE_PATH
    if _FSAVERAGE_PATH is not None:
        return _FSAVERAGE_PATH
    try:
        fs_dir = fetch_fsaverage()
    except Exception as e:
        print(f"  [mne] fetch_fsaverage failed: {e}")
        return None
    _FSAVERAGE_PATH = os.path.dirname(fs_dir)
    return _FSAVERAGE_PATH


def _set_brain_surface_alpha(brain, alpha):
    try:
        renderer = brain._renderer
    except AttributeError:
        return
    fig = (getattr(renderer, "plotter", None)
           or getattr(renderer, "figure", None))
    if fig is None:
        return
    actors = getattr(fig, "actors", None)
    if actors is None and hasattr(fig, "renderer"):
        actors = getattr(fig.renderer, "actors", None)
    if not actors:
        return
    iterable = actors.values() if hasattr(actors, "values") else actors
    for actor in iterable:
        try:
            actor.GetProperty().SetOpacity(float(alpha))
        except Exception:
            continue


def plot_mne_brain_inout(df_in, df_out, roi_col, roi_colors,
                         save_path, title,
                         hemi="both", views=("lateral", "medial"),
                         scale_in=MNE_FOCI_SCALE_IN,
                         scale_out=MNE_FOCI_SCALE_OUT,
                         jitter_mm=MNE_FOCI_JITTER_MM,
                         filter_foci_by_hemi=MNE_FILTER_FOCI_BY_HEMI,
                         mask_overlay_img=None,
                         overlay_color=MASK_OVERLAY_HEX,
                         overlay_alpha=0.6):
    """Render mne.viz.Brain with `df_in` coloured by ROI and `df_out`
    drawn in grey, optionally shading the mask cluster on the surface."""
    if not HAS_MNE_BRAIN:
        print("  [mne] not installed — skipping plot.")
        return
    subjects_dir = _ensure_fsaverage()
    if subjects_dir is None:
        print("  [mne] fsaverage unavailable — skipping.")
        return
    if df_in.empty and df_out.empty:
        print(f"  skip {save_path}: no cells.")
        return

    try:
        brain = Brain(
            subject="fsaverage", hemi=hemi, surf="pial",
            background="white", size=(900, 700),
            subjects_dir=subjects_dir, alpha=MNE_SURF_ALPHA,
            title=title,
        )
    except TypeError:
        try:
            brain = Brain(
                subject="fsaverage", hemi=hemi, surf="pial",
                background="white", size=(900, 700),
                subjects_dir=subjects_dir, title=title,
            )
            _set_brain_surface_alpha(brain, MNE_SURF_ALPHA)
        except Exception as e:
            print(f"  [mne] Brain(hemi={hemi!r}) failed: {e}")
            return
    except Exception as e:
        print(f"  [mne] Brain(hemi={hemi!r}) failed: {e}")
        return

    foci_hemis = ("lh", "rh") if hemi == "both" else (hemi,)

    # Orange shading of the cluster on the cortical surface.
    if mask_overlay_img is not None:
        try:
            from nilearn import surface as _surface
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list(
                "mask_overlay", [overlay_color, overlay_color])
            for foci_h in foci_hemis:
                surf_path = os.path.join(
                    subjects_dir, "fsaverage", "surf", f"{foci_h}.pial")
                texture = _surface.vol_to_surf(
                    mask_overlay_img, surf_path,
                    interpolation="nearest").astype(float)
                # Mask vertices below 0.5 to NaN so MNE renders them
                # transparent without needing the `thresh` kwarg
                # (older MNE versions only accept thresh=None).
                texture[texture < 0.5] = np.nan
                if not np.any(np.isfinite(texture)):
                    continue
                kwargs = dict(
                    hemi=foci_h, fmin=0.5, fmid=0.75, fmax=1.0,
                    colormap=cmap, alpha=overlay_alpha, colorbar=False,
                )
                try:
                    brain.add_data(texture, smoothing_steps=0, **kwargs)
                except TypeError:
                    brain.add_data(texture, **kwargs)
        except Exception as e:
            print(f"  [mne] surface overlay failed: {e}")

    def _plant(sub_df, color, name, scale):
        if sub_df.empty:
            return
        coords = sub_df[["MNI_x", "MNI_y", "MNI_z"]].to_numpy(dtype=float)
        if hemi in ("lh", "rh") and filter_foci_by_hemi:
            keep = coords[:, 0] <= 0 if hemi == "lh" else coords[:, 0] >= 0
            coords = coords[keep]
            if coords.shape[0] == 0:
                return
        coords = _jitter_coords(coords, jitter_mm,
                                 seed=hash(name + str(hemi)) & 0xFFFF)
        for foci_h in foci_hemis:
            try:
                brain.add_foci(
                    coords, coords_as_verts=False, hemi=foci_h,
                    color=color, scale_factor=scale, name=name,
                )
            except Exception as e:
                print(f"  [mne] add_foci ({name}, {foci_h}) failed: {e}")

    # First the grey background, then the ROI-coloured highlights on top.
    _plant(df_out, GREY_HEX, "outside_mask", scale_out)
    for roi, color in roi_colors.items():
        _plant(df_in[df_in[roi_col] == roi], color, roi, scale_in)

    base, ext = os.path.splitext(save_path)
    saved_any = False
    for view in views:
        try:
            brain.show_view(view)
            out = f"{base}_{view}{ext or '.png'}"
            brain.save_image(out)
            print(f"  wrote {out}")
            saved_any = True
        except Exception as e:
            print(f"  [mne] view {view!r} failed: {e}")
    if not saved_any:
        try:
            brain.save_image(save_path)
            print(f"  wrote {save_path}")
        except Exception as e:
            print(f"  [mne] save_image failed: {e}")
    try:
        brain.close()
    except Exception:
        pass


# =============================================================================
# COUNT FIGURE
# =============================================================================

def plot_counts_summary(counts_df, save_path, title):
    """Bar plot summarising inside / outside cell counts per mask."""
    masks_seen = counts_df["mask"].unique().tolist()
    rois_seen = [r for r in counts_df["roi"].unique() if r != "<all>"]
    rows = ["<all>"] + sorted(rois_seen)
    fig, axes = plt.subplots(
        1, len(masks_seen),
        figsize=(3.2 * len(masks_seen) + 1, 0.45 * len(rows) + 2),
        squeeze=False, constrained_layout=True,
    )
    for j, mname in enumerate(masks_seen):
        ax = axes[0, j]
        sub = counts_df[counts_df["mask"] == mname].set_index("roi")
        ins = [int(sub.loc[r, "n_inside"]) if r in sub.index else 0
               for r in rows]
        outs = [int(sub.loc[r, "n_outside"]) if r in sub.index else 0
                for r in rows]
        y = np.arange(len(rows))
        ax.barh(y, ins, color="#448363", label="inside mask")
        ax.barh(y, outs, left=ins, color="#cccccc", label="outside mask")
        for i, (a, b) in enumerate(zip(ins, outs)):
            ax.text(a + b + 0.5, i, f"{a}/{a + b}",
                    va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(rows, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("# cells", fontsize=10)
        ax.set_title(f"{mname} cluster", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        if j == len(masks_seen) - 1:
            ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.savefig(save_path.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {save_path} (+ .svg)")


# =============================================================================
# GRADIENT-OVERLAP RENDERING
# =============================================================================

def build_gradient_union_mask(tstat_dir, file_dict, threshold_dict,
                                prebuilt_path=None):
    """Build a single binary union mask of the four thresholded quarter
    blobs. If `prebuilt_path` is set and exists, that file is loaded
    instead and the per-map step is skipped.
    """
    if prebuilt_path and os.path.isfile(prebuilt_path):
        print(f"  [gradient] using pre-built mask: {prebuilt_path}")
        return nib.load(prebuilt_path)

    union_data = None
    union_aff = None
    union_hdr = None
    for name, fname in file_dict.items():
        path = os.path.join(tstat_dir, fname)
        if not os.path.isfile(path):
            print(f"  [gradient] missing {name!r}: {path} — skipped.")
            continue
        img = nib.load(path)
        data = img.get_fdata()
        thr = float(threshold_dict.get(name, 2.5))
        bin_arr = (data > thr).astype(np.uint8)
        n_vox = int(bin_arr.sum())
        print(f"  [gradient] {name:7s} t > {thr:>4.2f}  →  {n_vox:>5d} voxels")
        if union_data is None:
            union_data = bin_arr.copy()
            union_aff, union_hdr = img.affine, img.header
        else:
            union_data |= bin_arr
        del data, bin_arr, img
    if union_data is None:
        return None
    n_union = int(union_data.sum())
    print(f"  [gradient] union mask = {n_union} voxels")
    return nib.Nifti1Image(union_data.astype(np.uint8),
                            union_aff, union_hdr)


def acc_era_brewer_colour():
    """ACC hue from era_brewer Showgirl2 (index 1, per CLAUDE.md)."""
    if HAS_ERA_BREWER:
        try:
            colors = era_brewer.era_brew(PALETTE_NAME, n=7,
                                           brew_type="discrete")
            return mcolors.to_hex(colors[1])
        except Exception as e:
            print(f"  [era_brewer] ACC colour fallback: {e}")
    return "#C0492C"   # fallback close to the Showgirl2 ACC orange-red


def lofc_pink_colour():
    """lOFC hue: era_brewer Lover2 index 0 (n=5) — the deep pink."""
    if HAS_ERA_BREWER:
        try:
            colors = era_brewer.era_brew("Lover2", n=5, brew_type="discrete")
            return mcolors.to_hex(colors[0])
        except Exception as e:
            print(f"  [era_brewer] lOFC pink fallback: {e}")
    return DSR_CORONAL_PINK_HEX


def _save_brain_image_as_pdf(png_path, pdf_path, title=None,
                              footer=None, dpi=300):
    """Embed an MNE-rendered PNG into a PDF page with optional title /
    footer. The brain itself remains a raster screenshot; metadata text is
    vector."""
    img = plt.imread(png_path)
    fig = plt.figure(figsize=(img.shape[1] / dpi, img.shape[0] / dpi + 0.7),
                     dpi=dpi)
    ax = fig.add_axes([0.0, 0.07, 1.0, 0.86])
    ax.imshow(img)
    ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11, y=0.985)
    if footer:
        fig.text(0.5, 0.02, footer, ha="center", va="bottom",
                 fontsize=9, color="#333")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _jitter_coords(coords, mm, seed=0):
    """Add a tiny isotropic 3-D jitter so co-located cells visually separate."""
    if coords.size == 0 or mm <= 0:
        return coords
    rng = np.random.default_rng(seed)
    return coords + rng.uniform(-mm, mm, size=coords.shape)


def plot_gradient_overlap_brain(df, union_img, hemi, view, save_stem,
                                 acc_color, title, footer,
                                 ramp_hex=GRADIENT_RAMP_HEX,
                                 cell_scale=GRADIENT_CELL_SCALE,
                                 jitter_mm=GRADIENT_CELL_JITTER_MM,
                                 overlay_alpha=GRADIENT_OVERLAY_ALPHA,
                                 brain_size=GRADIENT_BRAIN_SIZE):
    """Render one medial hemisphere with the gradient blob coloured by
    anterior–posterior MNI-y (low y → dark red, high y → bright yellow)
    and overlay cells: grey small dots outside, ACC-coloured small dots
    inside the union mask. Saves PNG + PDF.
    """
    if not HAS_MNE_BRAIN:
        print("  [mne] not installed — skipping gradient plot.")
        return
    subjects_dir = _ensure_fsaverage()
    if subjects_dir is None:
        print("  [mne] fsaverage unavailable — skipping.")
        return
    if union_img is None:
        print(f"  [mne] no union mask for {hemi}/{view} — skipping.")
        return

    try:
        brain = Brain(
            subject="fsaverage", hemi=hemi, surf="pial",
            background="white", size=brain_size,
            subjects_dir=subjects_dir, alpha=MNE_SURF_ALPHA,
            title=None,
        )
    except TypeError:
        brain = Brain(
            subject="fsaverage", hemi=hemi, surf="pial",
            background="white", size=brain_size,
            subjects_dir=subjects_dir, title=None,
        )
        _set_brain_surface_alpha(brain, MNE_SURF_ALPHA)

    # --- single overlay coloured by per-vertex DORSAL–VENTRAL (MNI z) ---
    try:
        from nilearn import surface as _surface
        from matplotlib.colors import LinearSegmentedColormap
        from nibabel.freesurfer import read_geometry

        surf_path = os.path.join(
            subjects_dir, "fsaverage", "surf", f"{hemi}.pial")
        verts, _ = read_geometry(surf_path)        # (n_vert, 3), MNI mm
        texture_bin = _surface.vol_to_surf(
            union_img, surf_path, interpolation="nearest").astype(float)
        in_mask = texture_bin >= 0.5
        data = np.full_like(texture_bin, np.nan, dtype=float)
        if in_mask.any():
            data[in_mask] = verts[in_mask, 2]      # MNI z of each vertex
            z_min = float(np.nanmin(data))
            z_max = float(np.nanmax(data))
            if z_max - z_min < 1e-3:
                z_max = z_min + 1.0
            cmap = LinearSegmentedColormap.from_list(
                "gradient_DV", ramp_hex)            # low z → very dark red, high z → yellow
            kwargs = dict(
                hemi=hemi, fmin=z_min, fmid=(z_min + z_max) / 2,
                fmax=z_max, colormap=cmap, alpha=overlay_alpha,
                colorbar=False,
            )
            try:
                brain.add_data(data, smoothing_steps=0, **kwargs)
            except TypeError:
                brain.add_data(data, **kwargs)
        else:
            print(f"  [mne] union mask has no surface vertices on {hemi}")
    except Exception as e:
        print(f"  [mne] surface-overlay setup failed: {e}")

    # --- planters: grey for non-overlap, ACC for overlap (equal size) ---
    coords_all = df[["MNI_x", "MNI_y", "MNI_z"]].to_numpy(dtype=float)
    inside = voxel_inside_mask(coords_all, union_img)
    if hemi == "lh":
        hemi_keep = coords_all[:, 0] <= 0
    else:
        hemi_keep = coords_all[:, 0] >= 0
    coords_out = coords_all[(~inside) & hemi_keep]
    coords_in  = coords_all[(inside)  & hemi_keep]
    print(f"  [{hemi}] cells outside={coords_out.shape[0]}  "
          f"inside={coords_in.shape[0]}  "
          f"(union total inside, both hemis = {int(inside.sum())})")
    coords_out_j = _jitter_coords(coords_out, jitter_mm, seed=hash(hemi+"out") & 0xFFFF)
    coords_in_j  = _jitter_coords(coords_in,  jitter_mm, seed=hash(hemi+"in")  & 0xFFFF)

    if coords_out_j.shape[0]:
        try:
            brain.add_foci(coords_out_j, coords_as_verts=False, hemi=hemi,
                            color=GREY_HEX, scale_factor=cell_scale,
                            name="outside_gradient")
        except Exception as e:
            print(f"  [mne] add_foci(grey) failed: {e}")
    if coords_in_j.shape[0]:
        try:
            brain.add_foci(coords_in_j, coords_as_verts=False, hemi=hemi,
                            color=acc_color, scale_factor=cell_scale,
                            name="inside_gradient")
        except Exception as e:
            print(f"  [mne] add_foci(ACC) failed: {e}")

    # ---- render → PNG → embed in PDF ----------------------------------
    try:
        brain.show_view(view)
        png_path = f"{save_stem}.png"
        brain.save_image(png_path)
        pdf_path = f"{save_stem}.pdf"
        _save_brain_image_as_pdf(png_path, pdf_path,
                                  title=title, footer=footer)
        print(f"  wrote {pdf_path}")
    except Exception as e:
        print(f"  [mne] save {save_stem!r} failed: {e}")
    finally:
        try:
            brain.close()
        except Exception:
            pass


def plot_dsr_coronal_section(df, full_mask_img, lofc_mask_img, roi_col,
                              save_stem, title,
                              pink_hex=None,
                              grey_hex=DSR_CORONAL_GREY_HEX,
                              marker_size=DSR_CORONAL_MARKER_SIZE,
                              jitter_mm=DSR_CORONAL_JITTER_MM,
                              y_tol_mm=DSR_CORONAL_Y_TOL_MM):
    """Coronal slice through the lOFC sub-cluster of the DSR main effect.

    * Background: MNI152 anatomical (nilearn default), with the DSR main
      effect mask shaded orange.
    * Markers: cells within ±`y_tol_mm` mm of the lOFC centre-of-mass y.
      lOFC cells coloured Lover2 pink; cells of other ROIs in grey.
      Coords are jittered isotropically so co-located cells separate.
    """
    if lofc_mask_img is None or full_mask_img is None:
        print("  [coronal] no lOFC sub-cluster — skipping.")
        return
    try:
        from nilearn import plotting as _plotting
    except Exception as e:
        print(f"  [coronal] nilearn unavailable: {e}")
        return
    pink_hex = pink_hex or lofc_pink_colour()

    # Centre-of-mass y of the lOFC sub-cluster.
    lofc_data = lofc_mask_img.get_fdata()
    idx = np.argwhere(lofc_data > 0)
    if idx.size == 0:
        print("  [coronal] empty lOFC mask — skipping.")
        return
    com_vox = idx.mean(axis=0)
    com_mni = apply_affine(lofc_mask_img.affine, com_vox)
    y_cut = float(com_mni[1])
    print(f"  [coronal] lOFC COM (MNI) = "
          f"({com_mni[0]:+.1f}, {com_mni[1]:+.1f}, {com_mni[2]:+.1f})  "
          f"→ coronal slice at y = {y_cut:+.1f}")

    # Cells: keep those within the slice window.
    coords_all = df[["MNI_x", "MNI_y", "MNI_z"]].to_numpy(dtype=float)
    near_slice = np.abs(coords_all[:, 1] - y_cut) <= float(y_tol_mm)
    roi = df[roi_col].to_numpy()
    in_lofc_roi = (roi == "medialOFC")
    inside_full = voxel_inside_mask(coords_all, full_mask_img)

    coords_pink = coords_all[near_slice & in_lofc_roi]
    coords_grey = coords_all[near_slice & (~in_lofc_roi)]
    coords_pink_j = _jitter_coords(coords_pink, jitter_mm, seed=2026)
    coords_grey_j = _jitter_coords(coords_grey, jitter_mm, seed=2027)
    n_pink_overlap = int((near_slice & in_lofc_roi & inside_full).sum())

    fig = plt.figure(figsize=(7.5, 5.5), dpi=300)
    display = _plotting.plot_roi(
        full_mask_img, bg_img=None,
        display_mode="y", cut_coords=[y_cut],
        cmap="autumn", alpha=0.55, colorbar=False,
        title=title, figure=fig,
    )
    if coords_grey_j.shape[0]:
        display.add_markers(
            coords_grey_j, marker_color=grey_hex,
            marker_size=marker_size * 0.7,
        )
    if coords_pink_j.shape[0]:
        display.add_markers(
            coords_pink_j, marker_color=pink_hex,
            marker_size=marker_size,
        )
    footer = (
        f"y = {y_cut:+.1f} mm  ±{y_tol_mm:.0f} mm   |   "
        f"medialOFC cells in slice = {coords_pink.shape[0]}, "
        f"of which {n_pink_overlap} overlap the DSR mask"
    )
    fig.text(0.5, 0.02, footer, ha="center", va="bottom",
             fontsize=9, color="#333")

    pdf_path = f"{save_stem}.pdf"
    png_path = f"{save_stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf_path}")


def gradient_overlap_counts(df, union_img, roi_col):
    """Long-form table of cells × ROI inside the gradient union mask."""
    if union_img is None:
        return pd.DataFrame()
    coords = df[["MNI_x", "MNI_y", "MNI_z"]].to_numpy(dtype=float)
    inside = voxel_inside_mask(coords, union_img)
    rows = [{
        "mask": "gradient_union", "roi": "<all>",
        "n_inside": int(inside.sum()),
        "n_outside": int((~inside).sum()),
        "n_total": int(len(inside)),
    }]
    for roi in df[roi_col].dropna().unique():
        sel = (df[roi_col].to_numpy() == roi)
        rows.append({
            "mask": "gradient_union", "roi": roi,
            "n_inside": int((inside & sel).sum()),
            "n_outside": int((~inside & sel).sum()),
            "n_total": int(sel.sum()),
        })
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def _subset_session_counts(df_all, dsr_subjects):
    subs = df_all["subject"].astype(str).str.zfill(2)
    in_rsa = subs.isin(dsr_subjects or set())
    return {
        "all": (int(df_all["subject"].nunique()), len(df_all)),
        "dsr_rsa": (int(subs[in_rsa].nunique()), int(in_rsa.sum())),
        "not_dsr_rsa": (int(subs[~in_rsa].nunique()), int((~in_rsa).sum())),
    }


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(OUTPUT_ROOT, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Load cells + DSR subject list (shared across all input masks).
    print(f"\nLoading cell table: {CELL_TABLE_PATH}")
    df_all = pd.read_csv(CELL_TABLE_PATH)
    df_all = df_all.dropna(subset=[ROI_COL, "MNI_x", "MNI_y", "MNI_z"])
    df_all = df_all[df_all[ROI_COL].isin(ROI_ORDER)].copy()
    print(f"  {len(df_all)} cells with usable ROI + MNI coords.")
    dsr_subjects = load_dsr_subjects()
    if dsr_subjects is None:
        print("  WARNING: DSR JSON missing — 'dsr_rsa' subsets will be empty.")
        dsr_subjects = set()

    sess_counts = _subset_session_counts(df_all, dsr_subjects)
    print("\nSubset session / cell counts:")
    for s, (n_sess, n_cell) in sess_counts.items():
        print(f"  {s:>12}: {n_sess:>3} sessions, {n_cell:>4} cells")

    rois_seen = [r for r in ROI_ORDER if (df_all[ROI_COL] == r).any()]
    roi_palette = make_roi_palette(rois_seen, PALETTE_NAME)

    all_counts_rows = []
    for mask_name, mask_path in INPUT_MASKS.items():
        print(f"\n========== Input mask: {mask_name} ==========")
        try:
            full_mask_img = load_input_mask(mask_name, mask_path)
        except FileNotFoundError as e:
            print(f"  {e} — skipping.")
            continue
        masks = identify_subclusters(full_mask_img)

        mask_dir = os.path.join(run_dir, mask_name)
        os.makedirs(mask_dir, exist_ok=True)

        for subset in CELL_SUBSETS:
            df_sub, subset_label = get_cell_subset(
                df_all, subset, dsr_subjects)
            n_sess, _ = sess_counts[subset]
            subset_label = f"{subset_label}, {n_sess} sessions"
            print(f"\n--- {mask_name} | subset {subset} "
                  f"— {subset_label} ---")
            subset_dir = os.path.join(mask_dir, subset)
            os.makedirs(subset_dir, exist_ok=True)
            if df_sub.empty:
                print("  no cells — skipping.")
                continue

            counts = overlap_counts(df_sub, masks, ROI_COL)
            counts["subset"] = subset
            counts["input_mask"] = mask_name
            counts_path = os.path.join(subset_dir, "counts.csv")
            counts.to_csv(counts_path, index=False)
            print(f"  wrote {counts_path}")
            print(counts.to_string(index=False))
            all_counts_rows.append(counts)

            def _ix(mname, roi):
                row = counts[(counts["mask"] == mname)
                             & (counts["roi"] == roi)]
                return None if row.empty else int(row.iloc[0]["n_inside"])
            print(f"\n  Headline counts (mask={mask_name}, subset={subset}):")
            print(f"    ACC cells in ACC sub-cluster   : "
                  f"{_ix('ACC', 'ACC')}")
            print(f"    medialOFC cells in lOFC sub-cl.: "
                  f"{_ix('lOFC', 'medialOFC')}")
            print(f"    all cells in full mask         : "
                  f"{_ix('full', '<all>')}")

            plot_counts_summary(
                counts,
                save_path=os.path.join(subset_dir, "counts_summary.png"),
                title=(f"{mask_name}  |  {subset_label}"),
            )

            coords_mm = df_sub[["MNI_x", "MNI_y", "MNI_z"]].to_numpy(
                dtype=float)
            inside_full = voxel_inside_mask(coords_mm, masks["full"])
            df_in = df_sub.iloc[inside_full].copy()
            df_out = df_sub.iloc[~inside_full].copy()
            n_in, n_out = len(df_in), len(df_out)
            title = (f"{mask_name} | {subset_label}  |  "
                     f"inside mask: {n_in}/{n_in + n_out}")
            plot_mne_brain_inout(
                df_in, df_out, ROI_COL, roi_palette,
                save_path=os.path.join(
                    subset_dir, f"mne_brain_both_{subset}.png"),
                title=title,
                hemi="both", views=MNE_VIEWS_BOTH_HEMI,
                mask_overlay_img=masks["full"],
            )
            for h in MNE_HEMIS_SOLO:
                plot_mne_brain_inout(
                    df_in, df_out, ROI_COL, roi_palette,
                    save_path=os.path.join(
                        subset_dir, f"mne_brain_{h}_{subset}.png"),
                    title=f"{title} — {h}",
                    hemi=h, views=MNE_VIEWS_SOLO_HEMI,
                    mask_overlay_img=masks["full"],
                )

            # ---- Coronal section through the lOFC overlap -----------
            if DSR_CORONAL_ENABLED and mask_name == "DSR_main_effect":
                plot_dsr_coronal_section(
                    df_sub, masks.get("full"), masks.get("lOFC"),
                    ROI_COL,
                    save_stem=os.path.join(
                        subset_dir, f"coronal_lOFC_{subset}"),
                    title=(f"DSR main effect — lOFC coronal slice  "
                           f"|  {subset_label}"),
                )

    if all_counts_rows:
        merged = pd.concat(all_counts_rows, ignore_index=True)
        merged_path = os.path.join(run_dir, "all_counts.csv")
        merged.to_csv(merged_path, index=False)
        print(f"\nMerged counts table: {merged_path}")

    # =====================================================================
    # GRADIENT BLOB  (figure 1c style — single union mask coloured yellow
    # → dark red along anterior–posterior, medial lh+rh views)
    # =====================================================================
    print("\n========== Gradient blob overlap ==========")
    union_img = build_gradient_union_mask(
        GRADIENT_TSTAT_DIR, GRADIENT_TSTAT_MAPS, GRADIENT_TSTAT_THRESHOLDS,
        prebuilt_path=GRADIENT_PREBUILT_MASK,
    )
    if union_img is None:
        print("  no gradient union mask built — skipping.")
    else:
        acc_color = acc_era_brewer_colour()
        grad_root = os.path.join(run_dir, GRADIENT_OUT_SUBDIR)
        os.makedirs(grad_root, exist_ok=True)

        with open(os.path.join(grad_root, "gradient_settings.json"), "w") as f:
            json.dump({
                "tstat_thresholds":    GRADIENT_TSTAT_THRESHOLDS,
                "ramp_hex":            GRADIENT_RAMP_HEX,
                "acc_cell_colour":     acc_color,
                "view":                GRADIENT_VIEW,
                "hemis":               list(GRADIENT_HEMIS),
                "files":               GRADIENT_TSTAT_MAPS,
                "prebuilt_mask":       GRADIENT_PREBUILT_MASK,
                "cell_scale":          GRADIENT_CELL_SCALE,
                "jitter_mm":           GRADIENT_CELL_JITTER_MM,
                "overlay_alpha":       GRADIENT_OVERLAY_ALPHA,
            }, f, indent=2)

        for subset in CELL_SUBSETS:
            df_sub, subset_label = get_cell_subset(
                df_all, subset, dsr_subjects)
            n_sess, _ = sess_counts[subset]
            subset_label = f"{subset_label}, {n_sess} sessions"
            print(f"\n--- gradient | subset {subset} — {subset_label} ---")
            subset_dir = os.path.join(grad_root, subset)
            os.makedirs(subset_dir, exist_ok=True)
            if df_sub.empty:
                print("  no cells — skipping.")
                continue

            counts = gradient_overlap_counts(df_sub, union_img, ROI_COL)
            counts["subset"] = subset
            counts.to_csv(os.path.join(subset_dir, "counts.csv"),
                          index=False)
            print(counts.to_string(index=False))

            n_overlap = int(counts.loc[counts["roi"] == "<all>", "n_inside"].iloc[0])
            n_total   = int(counts.loc[counts["roi"] == "<all>", "n_total"].iloc[0])
            footer = (f"cells overlapping gradient: {n_overlap}/{n_total}   "
                       f"(t-thresholds: " +
                       ", ".join(f"{k}>{v:.2f}"
                                  for k, v in GRADIENT_TSTAT_THRESHOLDS.items())
                       + ")")
            print(f"  overlap with gradient union: {n_overlap}/{n_total}")

            for h in GRADIENT_HEMIS:
                stem = os.path.join(subset_dir,
                                     f"gradient_overlap_{h}_{GRADIENT_VIEW}")
                title = (f"Gradient overlap — {subset_label}   "
                          f"({h} {GRADIENT_VIEW})")
                plot_gradient_overlap_brain(
                    df_sub, union_img,
                    hemi=h, view=GRADIENT_VIEW,
                    save_stem=stem, acc_color=acc_color,
                    title=title, footer=footer,
                )

    print(f"\nAll outputs under: {run_dir}")


if __name__ == "__main__":
    main()
