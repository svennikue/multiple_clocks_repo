#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain-coverage visualisation sandbox.

Reads `neurons_with_final_roi_labels.csv` produced by
scripts/cell_to_roi_MNI.py and produces brain plots in a few different
styles + colour palettes so we can pick the best look for figures.

Two cell subsets are rendered:
  * 'all'      : every cell in the ROI table.
  * 'dsr_rsa'  : only cells whose `subject` appears in
                 `all_sessions_dsrRSA_grouping_summary.json` (the subset
                 actually entering the DSR RSA / encoding pipeline).

Visualisations cycled per subset:
  * `glassbrain_lyrz`   — nilearn 4-view glass brain, markers by ROI.
  * `glassbrain_mosaic` — one mini ortho glass brain per ROI.
  * `mne_brain`         — `mne.viz.Brain` on fsaverage pial surface,
                          foci added per ROI. Skipped automatically if
                          MNE / pyvista / fsaverage aren't available.

Colour palettes cycled per visualisation:
  * `era_brewer` palettes (`midnight2`, `showgirl2`, `nineteeneightynine`,
    `Lover2`) discrete-with-n=7 if `era_brewer` is installed.
  * If `era_brewer` is missing we fall back to hardcoded hex strings for
    the first three palettes (Lover2 hex not provided -> tab10 fallback).

All outputs land under
  <DATA_DIR>/roi_visualization_sandbox/<subset>/<palette>/...

@author: Svenja Kuchenhoff
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from nilearn import plotting

sys.path.insert(
    0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo"
)
from mc.plotting.cell_results import (
    plot_roi_electrodes_glassbrain,
    plot_roi_beta_glassbrain,
    roi_display,
)


# ── era_brewer (palette source) ──────────────────────────────────────
try:
    import era_brewer
    HAS_ERA_BREWER = True
except Exception:
    HAS_ERA_BREWER = False


# ── mne.viz.Brain (surface visualisation) ────────────────────────────
try:
    from mne.viz import Brain
    from mne.datasets import fetch_fsaverage
    HAS_MNE_BRAIN = True
except Exception:
    HAS_MNE_BRAIN = False


# =============================================================================
# SETTINGS
# =============================================================================

DATA_DIR = (
    "/Users/xpsy1114/Documents/projects/multiple_clocks/"
    "data/ephys_humans/derivatives"
)
CELL_TABLE_PATH = os.path.join(DATA_DIR, "neurons_with_ROI_labels.csv")
DSR_JSON_PATH = os.path.join(
    DATA_DIR, "all_sessions_dsrRSA_grouping_summary.json")

# Which ROI column to plot ('alt_final_roi' is the post-filter labelling).
ROI_COL = "alt_final_roi"
# Canonical display order. ROIs absent from the data (e.g. dropped by the
# <3-subject filter in cell_to_roi_MNI.py) just get skipped.
ROI_ORDER = [
    "mPFC", "medial_CC",
    "HC_anterior", "HC_mid",
    "EC", "PHC",
    "PCC",
    "mOFC",
    "Visual",
]


# Project-wide ROI colours (CLAUDE.md mapping on the native Showgirl2
# discrete palette). NOTE: ``era_brewer.era_brew('Showgirl2', n=7)``
# interpolates and reorders the colours, so the canonical 7-hue palette
# is hardcoded in ``mc.plotting.cell_results.SHOWGIRL2_DISCRETE`` and
# pulled from there to keep all figures consistent.
from mc.plotting.cell_results import SHOWGIRL2_DISCRETE as _SHOWGIRL2
ROI_COLOURS = {
    "EC":          _SHOWGIRL2[0],   # dark red
    "mPFC":        _SHOWGIRL2[1],   # teal-green
    "HC_mid":      _SHOWGIRL2[2],   # tan
    "PCC":         _SHOWGIRL2[3],   # pale green
    "mOFC":        _SHOWGIRL2[4],   # orange-red
    "HC_anterior": "#a30d6c",       # magenta (CLAUDE.md override)
    "PHC":         "#23677E",       # teal    (CLAUDE.md override)
    "medial_CC":   "#888888",
    "Visual":      "#bdbdbd",
}

# Which cell subsets to render.
CELL_SUBSETS = ["all", "dsr_rsa"]

# Which palettes to cycle (era_brewer names).  The pseudo-palette
# ``"showgirl2_adjusted"`` is treated specially: it uses the corrected
# Showgirl2 hex codes (from ``mc.plotting.cell_results.SHOWGIRL2_DISCRETE``)
# and assigns each ROI its CLAUDE.md-canonical index, rather than colouring
# ROIs in the order they happen to appear in the data. This is the
# CORRECT mapping for publication panels — the raw "showgirl2" cycle
# above is kept for visual comparison only.
ERA_BREWER_PALETTES = [
    "midnight2",
    "showgirl2",
    "showgirl2_adjusted",
    "nineteeneightynine",
    "Lover2",
]
ERA_BREWER_N = 7    # 7 distinct colours per palette where possible.

# Which visualisation styles to render.
PLOT_GLASSBRAIN_LYRZ = True
PLOT_GLASSBRAIN_MOSAIC = True
PLOT_MNE_BRAIN = True   # auto-skipped if MNE / pyvista aren't installed.
# Publication-grade glass-brain using mc.plotting.cell_results helpers,
# saved as PDF + PNG with the CLAUDE.md canonical ROI palette.
PLOT_PUBLICATION_GLASSBRAIN = True

# Marker sizing.
MARKER_SIZE_GLASSBRAIN = 18
MNE_FOCI_SCALE = 0.5
# Transparency of the cortical surface in the mne.viz.Brain plots so
# foci sitting deep inside the brain stay visible.  0 = invisible,
# 1 = opaque.  Around 0.25–0.40 usually shows the surface outline while
# letting the markers come through.
MNE_SURF_ALPHA = 0.30

# mne.viz.Brain render passes per (subset, palette):
# * BOTH: hemi='both', external surface views; medial wall is hidden by
#   the opposite hemisphere here, so add the per-hemi pass below to see it.
# * SOLO: render hemi='lh' alone and hemi='rh' alone — combined with the
#   'medial' view this gives a clean shot of the inter-hemispheric wall
#   (where most of the cingulate / medial-OFC electrodes sit).
MNE_VIEWS_BOTH_HEMI = ("lateral", "dorsal")
MNE_VIEWS_SOLO_HEMI = ("medial", "lateral")
MNE_HEMIS_SOLO = ("lh", "rh")
# When rendering a single hemisphere, restrict the foci to that hemisphere
# (based on MNI_x sign).  Set False to project every electrode onto the
# rendered surface regardless of side.
MNE_FILTER_FOCI_BY_HEMI = True

# Jitter visually-identical coordinates so individual cells are visible.
JITTER_MM = 0.7

OUTPUT_ROOT = os.path.join(DATA_DIR, "roi_visualization_sandbox")


# Fallback hex strings — used when `era_brewer` isn't importable.
FALLBACK_PALETTES = {
    "midnight2":          ["#411D30", "#9A383C", "#BC7E6A", "#CBCBCD",
                           "#3D5F82", "#2B4159", "#202D3C"],
    "showgirl2":          ["#B74C2D", "#DC673E", "#CCB178", "#EAEDC4",
                           "#C1DCBF", "#7BB594", "#448363"],
    "nineteeneightynine": ["#C34533", "#A55D38", "#CFA176", "#CACCC7",
                           "#8CACC4", "#7AA1BD", "#5A88AE"],
    # Lover2 hex strings not provided — falls through to tab10.
}


# =============================================================================
# DATA LOADING + SUBSET FILTERING
# =============================================================================

print(f"Loading cell table: {CELL_TABLE_PATH}")
DF_ALL = pd.read_csv(CELL_TABLE_PATH)
for _ax in ('x', 'y', 'z'):
    if f'MNI_{_ax}_final' in DF_ALL.columns:
        DF_ALL[f'MNI_{_ax}'] = DF_ALL[f'MNI_{_ax}_final']
print(f"  {len(DF_ALL)} cells; "
      f"{DF_ALL['subject'].nunique()} subjects; "
      f"ROI column = {ROI_COL!r}")
DF_ALL = DF_ALL.dropna(subset=[ROI_COL] + ["MNI_x", "MNI_y", "MNI_z"])
print(f"  {len(DF_ALL)} after dropping NaN ROI / MNI rows.")


def load_dsr_subjects():
    """Return list of subject ids (zero-padded strings) that contributed
    to the DSR RSA pipeline. None if the JSON is unavailable."""
    if not os.path.isfile(DSR_JSON_PATH):
        return None
    with open(DSR_JSON_PATH) as f:
        d = json.load(f)
    return [str(k).zfill(2) for k in d.keys()]


def filter_subset(df, subset):
    """Return (filtered_df, descriptive_label) for 'all' or 'dsr_rsa'."""
    if subset == "all":
        return df.copy(), f"all electrodes (n={len(df)})"
    if subset == "dsr_rsa":
        subs = load_dsr_subjects()
        if subs is None:
            print("  DSR JSON not found — using all cells for 'dsr_rsa'.")
            return df.copy(), f"all electrodes (DSR JSON missing, n={len(df)})"
        keep = df["subject"].astype(str).str.zfill(2).isin(subs)
        sub_df = df[keep].copy()
        return sub_df, (f"DSR RSA subset "
                        f"(n={len(sub_df)} cells from "
                        f"{sub_df['subject'].nunique()} subjects)")
    raise ValueError(f"Unknown subset: {subset!r}")


# =============================================================================
# PALETTE / COLOUR HELPERS
# =============================================================================

def _palette_colors(palette_name, n):
    """Discrete list of `n` hex colours from `era_brewer`, with hex fallback.

    Order of preference:
      1. era_brewer.era_brew(palette_name, n=n, brew_type='discrete')
      2. FALLBACK_PALETTES[palette_name] (cycled to length n)
      3. matplotlib 'tab10'
    """
    if HAS_ERA_BREWER:
        try:
            colors = era_brewer.era_brew(
                palette_name, n=max(n, 2), brew_type="discrete")
            return [mcolors.to_hex(c) for c in colors]
        except Exception as e:
            print(f"    [era_brewer] palette {palette_name!r} failed: {e}")
    if palette_name in FALLBACK_PALETTES:
        base = FALLBACK_PALETTES[palette_name]
        return [base[i % len(base)] for i in range(n)]
    cmap = plt.get_cmap("tab10")
    return [mcolors.to_hex(cmap(i % 10)) for i in range(n)]


def make_roi_color_map(rois, palette_name, n=ERA_BREWER_N):
    """ROI -> hex colour mapping for the given palette.

    Special case: ``palette_name == 'showgirl2_adjusted'`` returns the
    project's canonical CLAUDE.md mapping (``ROI_COLOURS`` defined at the
    top of this script). Any ROI not in ``ROI_COLOURS`` falls back to a
    neutral grey so it's obvious it's an off-canvas ROI.

    All other palette names sample ``_palette_colors`` and assign
    sequentially in the order ``rois`` appears.
    """
    if palette_name == "showgirl2_adjusted":
        return {r: ROI_COLOURS.get(r, "#888888") for r in rois}
    colors = _palette_colors(palette_name, max(n, len(rois)))
    return {roi: colors[i % len(colors)] for i, roi in enumerate(rois)}


# =============================================================================
# JITTER (visual only)
# =============================================================================

def add_visual_jitter(df, jitter_mm=JITTER_MM):
    out = df.copy()
    out[["plot_x", "plot_y", "plot_z"]] = out[["MNI_x", "MNI_y", "MNI_z"]]
    grouped = out.groupby(["MNI_x", "MNI_y", "MNI_z"], sort=False)
    for _, idxs in grouped.groups.items():
        idxs = list(idxs)
        if len(idxs) <= 1:
            continue
        angles = np.linspace(0, 2 * np.pi, len(idxs), endpoint=False)
        out.loc[idxs, "plot_x"] += jitter_mm * np.cos(angles)
        out.loc[idxs, "plot_y"] += jitter_mm * np.sin(angles)
    return out


def rois_present(df, roi_col, order):
    return [r for r in order if (df[roi_col] == r).any()]


# =============================================================================
# VISUALISATION 1: nilearn 4-view glass brain coloured by ROI
# =============================================================================

def plot_glassbrain_lyrz(df, roi_col, order, roi_colors,
                          save_path, title):
    df = df[df[roi_col].isin(order)]
    if df.empty:
        print(f"  skip {save_path}: no cells.")
        return
    df = add_visual_jitter(df)
    marker_colors = [roi_colors[r] for r in df[roi_col]]
    coords = df[["plot_x", "plot_y", "plot_z"]].to_numpy()
    display = plotting.plot_glass_brain(
        None, display_mode="lyrz", title=title,
        black_bg=False, plot_abs=False,
    )
    display.add_markers(
        coords, marker_color=marker_colors,
        marker_size=MARKER_SIZE_GLASSBRAIN,
    )
    fig = plt.gcf()
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                   markerfacecolor=roi_colors[r],
                   markeredgecolor=roi_colors[r],
                   label=f"{roi_display(r)} (n={int((df[roi_col] == r).sum())})")
        for r in order if (df[roi_col] == r).any()
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 5),
               frameon=False, fontsize=8)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {save_path}")


# =============================================================================
# VISUALISATION 2: per-ROI ortho mosaic
# =============================================================================

def plot_glassbrain_mosaic(df, roi_col, order, roi_colors,
                            save_path, title):
    rois = rois_present(df, roi_col, order)
    if not rois:
        print(f"  skip {save_path}: no cells.")
        return
    df = add_visual_jitter(df)
    n = len(rois)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 5.0, nrows * 2.4),
        squeeze=False,
    )
    for k, roi in enumerate(rois):
        ax = axes[k // ncols, k % ncols]
        sub = df[df[roi_col] == roi]
        display = plotting.plot_glass_brain(
            None, display_mode="ortho", axes=ax,
            black_bg=False, plot_abs=False,
        )
        display.add_markers(
            sub[["plot_x", "plot_y", "plot_z"]].to_numpy(),
            marker_color=roi_colors[roi],
            marker_size=MARKER_SIZE_GLASSBRAIN * 0.5,
        )
        ax.set_title(f"{roi_display(roi)}  (n={len(sub)})", fontsize=10,
                     color=roi_colors[roi])
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {save_path}")


# =============================================================================
# VISUALISATION 3: mne.viz.Brain on fsaverage pial surface
# =============================================================================

_FSAVERAGE_PATH = None  # cached after first fetch.


def _ensure_fsaverage():
    """Download / locate fsaverage. Returns subjects_dir or None on failure."""
    global _FSAVERAGE_PATH
    if _FSAVERAGE_PATH is not None:
        return _FSAVERAGE_PATH
    try:
        fs_dir = fetch_fsaverage()              # returns path to fsaverage/
    except Exception as e:
        print(f"  [mne] fetch_fsaverage failed: {e}")
        return None
    _FSAVERAGE_PATH = os.path.dirname(fs_dir)   # subjects_dir = parent
    return _FSAVERAGE_PATH


def _set_brain_surface_alpha(brain, alpha):
    """Make the cortical surface translucent post-hoc.

    Used as a fallback when ``Brain(..., alpha=...)`` isn't supported by
    the installed MNE version. Best-effort: silently skips any actor it
    can't reach so we don't crash the figure if MNE's internals change.
    """
    try:
        renderer = brain._renderer
    except AttributeError:
        return
    fig = getattr(renderer, "plotter", None) or getattr(renderer, "figure", None)
    if fig is None:
        return
    # `fig` is either a pyvista.Plotter or an mne CustomFigure wrapping one.
    actors = getattr(fig, "actors", None)
    if actors is None and hasattr(fig, "renderer"):
        actors = getattr(fig.renderer, "actors", None)
    if not actors:
        return
    for actor in (actors.values() if hasattr(actors, "values") else actors):
        try:
            prop = actor.GetProperty()
            prop.SetOpacity(float(alpha))
        except Exception:
            continue


def plot_mne_brain(df, roi_col, order, roi_colors,
                    save_path, title,
                    hemi="both",
                    views=("lateral", "medial"),
                    surf="pial", background="white",
                    foci_scale=MNE_FOCI_SCALE,
                    filter_foci_by_hemi=MNE_FILTER_FOCI_BY_HEMI):
    """Render `mne.viz.Brain` with one set of foci per ROI.

    Parameters
    ----------
    hemi : {'lh', 'rh', 'both'}
        Hemisphere(s) to render. Pass 'lh' or 'rh' to expose that
        hemisphere's medial wall — the 'both' rendering hides each
        hemisphere's medial side behind the other.
    views : iterable of str
        Camera views to save (saved one file per view).
    filter_foci_by_hemi : bool
        When `hemi` is 'lh' or 'rh', skip foci whose MNI_x is on the
        opposite side so we don't project right-hemisphere electrodes
        onto the left surface (or vice-versa).
    """
    if not HAS_MNE_BRAIN:
        print("  [mne] not installed — skipping mne.viz.Brain.")
        return
    subjects_dir = _ensure_fsaverage()
    if subjects_dir is None:
        print("  [mne] fsaverage unavailable — skipping.")
        return
    rois = rois_present(df, roi_col, order)
    if not rois:
        print(f"  skip {save_path}: no cells.")
        return

    # Try Brain(..., alpha=...) first (mne >= 0.24 supports it). Fall back
    # to constructing the surface opaque and adjusting the actor opacity
    # after the fact so older mne versions still get a transparent brain.
    try:
        brain = Brain(
            subject="fsaverage", hemi=hemi, surf=surf,
            background=background, size=(900, 700),
            subjects_dir=subjects_dir, alpha=MNE_SURF_ALPHA,
            title=title,
        )
    except TypeError:
        try:
            brain = Brain(
                subject="fsaverage", hemi=hemi, surf=surf,
                background=background, size=(900, 700),
                subjects_dir=subjects_dir,
                title=title,
            )
            _set_brain_surface_alpha(brain, MNE_SURF_ALPHA)
        except Exception as e:
            print(f"  [mne] Brain(hemi={hemi!r}) failed: {e}")
            return
    except Exception as e:
        print(f"  [mne] Brain(hemi={hemi!r}) failed: {e}")
        return

    # Which hemispheres receive the foci. When hemi='both' we plant on
    # both surfaces so every electrode shows up regardless of side.
    if hemi == "both":
        foci_hemis = ("lh", "rh")
    else:
        foci_hemis = (hemi,)

    for roi in rois:
        sub = df[df[roi_col] == roi]
        coords = sub[["MNI_x", "MNI_y", "MNI_z"]].to_numpy(dtype=float)
        if coords.shape[0] == 0:
            continue
        # Per-hemisphere render: drop coords on the opposite side.
        if hemi in ("lh", "rh") and filter_foci_by_hemi:
            if hemi == "lh":
                keep = coords[:, 0] <= 0
            else:
                keep = coords[:, 0] >= 0
            coords = coords[keep]
            if coords.shape[0] == 0:
                continue
        for foci_h in foci_hemis:
            try:
                brain.add_foci(
                    coords, coords_as_verts=False, hemi=foci_h,
                    color=roi_colors[roi], scale_factor=foci_scale,
                    name=roi,
                )
            except Exception as e:
                print(f"  [mne] add_foci ({roi}, hemi={foci_h}) failed: {e}")

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
# VISUALISATION 4: publication glass-brain (mc.plotting.cell_results)
# =============================================================================

def plot_publication_glassbrain(df, roi_col, order, save_stem,
                                 title_combined, title_panels,
                                 roi_colors=None):
    """Wraps ``mc.plotting.cell_results.plot_roi_electrodes_glassbrain``
    with the CLAUDE.md ROI palette and saves BOTH a combined-view PDF and
    a per-ROI-mosaic PDF (plus PNG copies for previews).

    Two outputs per call:
      <save_stem>_combined.{pdf,png}  — all electrodes on one glass-brain
      <save_stem>_perroi.{pdf,png}    — one mini glass-brain per ROI

    A small wrapper around the canonical helper so this is the single
    place to tweak fonts / DPI for the headline brain-coverage figure.
    """
    df = df[df[roi_col].isin(order)]
    if df.empty:
        print(f"  skip publication glass-brain: no cells in any ROI.")
        return
    # Build ROI -> (n, 3) MNI coords. Order follows `order` for legend
    # stability across runs.
    electrodes = {}
    for roi in order:
        sub = df[df[roi_col] == roi]
        if sub.empty:
            continue
        electrodes[roi] = sub[["MNI_x", "MNI_y", "MNI_z"]].to_numpy(dtype=float)
    if not electrodes:
        print("  skip publication glass-brain: no ROI matches order.")
        return

    pdf_combined = save_stem + "_combined.pdf"
    png_combined = save_stem + "_combined.png"
    pdf_panels   = save_stem + "_perroi.pdf"
    png_panels   = save_stem + "_perroi.png"

    # Combined view — one glass-brain, all ROIs colour-coded ----------
    fig = plot_roi_electrodes_glassbrain(
        electrodes, save_path=pdf_combined,
        title=title_combined, marker_size=20,
        per_roi_panels=False, roi_colors=roi_colors,
    )
    if fig is not None:
        fig.savefig(png_combined, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  wrote {pdf_combined}  (+ PNG preview)")

    # Per-ROI mosaic ---------------------------------------------------
    fig = plot_roi_electrodes_glassbrain(
        electrodes, save_path=pdf_panels,
        title=title_panels, marker_size=20,
        per_roi_panels=True, roi_colors=roi_colors,
    )
    if fig is not None:
        fig.savefig(png_panels, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  wrote {pdf_panels}  (+ PNG preview)")


# =============================================================================
# MAIN LOOP
# =============================================================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)
if not HAS_ERA_BREWER:
    print("Note: era_brewer not importable; using hardcoded fallbacks "
          "for the palettes it is missing.")
if PLOT_MNE_BRAIN and not HAS_MNE_BRAIN:
    print("Note: mne.viz.Brain not importable; that visualisation is "
          "skipped.")

for subset in CELL_SUBSETS:
    df_sub, subset_label = filter_subset(DF_ALL, subset)
    print(f"\n=== Subset: {subset} — {subset_label} ===")
    if df_sub.empty:
        print("  no cells in this subset; skipping.")
        continue
    subset_dir = os.path.join(OUTPUT_ROOT, subset)
    os.makedirs(subset_dir, exist_ok=True)

    rois = rois_present(df_sub, ROI_COL, ROI_ORDER)
    if not rois:
        print(f"  no ROIs from {ROI_ORDER} present in this subset.")
        continue

    # Publication glass-brain — single render per subset using CLAUDE.md
    # canonical ROI colours (PDF for figure inclusion, PNG for preview).
    if PLOT_PUBLICATION_GLASSBRAIN:
        pub_stem = os.path.join(
            subset_dir, f"publication_glassbrain_{subset}")
        plot_publication_glassbrain(
            df_sub, ROI_COL, rois,
            save_stem=pub_stem,
            title_combined=f"Electrode coverage — {subset_label}",
            title_panels=f"Per-ROI coverage — {subset_label}",
            roi_colors=ROI_COLOURS,
        )

    for palette in ERA_BREWER_PALETTES:
        roi_colors = make_roi_color_map(rois, palette, n=ERA_BREWER_N)
        pal_dir = os.path.join(subset_dir, palette)
        os.makedirs(pal_dir, exist_ok=True)
        print(f"\n  -- Palette {palette!r} -> {pal_dir}")

        if PLOT_GLASSBRAIN_LYRZ:
            plot_glassbrain_lyrz(
                df_sub, ROI_COL, rois, roi_colors,
                save_path=os.path.join(
                    pal_dir, f"glassbrain_lyrz_{subset}_{palette}.svg"),
                title=f"All electrodes — {subset_label}  |  {palette}",
            )

        if PLOT_GLASSBRAIN_MOSAIC:
            plot_glassbrain_mosaic(
                df_sub, ROI_COL, rois, roi_colors,
                save_path=os.path.join(
                    pal_dir, f"glassbrain_mosaic_{subset}_{palette}.png"),
                title=f"Per-ROI mosaic — {subset_label}  |  {palette}",
            )

        if PLOT_MNE_BRAIN:
            # Pass 1: both hemispheres in one figure (best for whole-brain
            # external surface views).
            plot_mne_brain(
                df_sub, ROI_COL, rois, roi_colors,
                save_path=os.path.join(
                    pal_dir, f"mne_brain_both_{subset}_{palette}.png"),
                title=f"MNE brain — {subset_label}  |  {palette}",
                hemi="both", views=MNE_VIEWS_BOTH_HEMI,
            )
            # Pass 2: each hemisphere alone so its medial wall is visible.
            for h in MNE_HEMIS_SOLO:
                plot_mne_brain(
                    df_sub, ROI_COL, rois, roi_colors,
                    save_path=os.path.join(
                        pal_dir,
                        f"mne_brain_{h}_{subset}_{palette}.png"),
                    title=(f"MNE brain ({h}) — {subset_label}  "
                           f"|  {palette}"),
                    hemi=h, views=MNE_VIEWS_SOLO_HEMI,
                )

print(f"\nAll outputs under: {OUTPUT_ROOT}")
