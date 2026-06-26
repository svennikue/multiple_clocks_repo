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
    "gradient":        GRADIENT_MASK_PATH,
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

MNE_FOCI_SCALE_IN = 0.9        # cells inside the mask — bumped up so the
MNE_FOCI_SCALE_OUT = 0.35      # ROI colour isn't lost among the grey backdrop.
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
# Per-condition voxel-wise t threshold. The figure-1c look comes from
# slightly different cuts per map (the high-t bright-yellow blob is
# sharper, the dark-red blob a bit looser). Tune these to taste.
GRADIENT_TSTAT_THRESHOLDS = {
    "current": 3.10,   # roughly p99
    "next":    3.50,   # roughly p99
    "next+2":  2.70,   # roughly p99
    "next+3":  2.40,   # roughly p99
}
# Yellow → dark-red ramp matching figure 1c (curr = bright yellow at the
# anterior tip, next+3 = dark red posterior).
GRADIENT_COLOURS = {
    "current": "#FCE300",   # bright yellow
    "next":    "#FF8C00",   # orange
    "next+2":  "#D2691E",   # dark orange
    "next+3":  "#8B0000",   # dark red
}
# Restrict the rendering to medial-view L+R, which is the panel the user
# asked for.
GRADIENT_VIEW              = "medial"
GRADIENT_HEMIS             = ("lh", "rh")
GRADIENT_OVERLAY_ALPHA     = 0.70
# Cell dot scales — smaller than the main-effect plot so individual cells
# show up among the ~50 that overlap.
GRADIENT_CELL_SCALE_IN     = 0.45
GRADIENT_CELL_SCALE_OUT    = 0.20
GRADIENT_BRAIN_SIZE        = (1200, 1000)
GRADIENT_OUT_SUBDIR        = "gradient_blobs"


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

    if all_counts_rows:
        merged = pd.concat(all_counts_rows, ignore_index=True)
        merged_path = os.path.join(run_dir, "all_counts.csv")
        merged.to_csv(merged_path, index=False)
        print(f"\nMerged counts table: {merged_path}")

    print(f"\nAll outputs under: {run_dir}")


if __name__ == "__main__":
    main()
