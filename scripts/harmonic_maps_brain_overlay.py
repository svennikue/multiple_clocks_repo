#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Surface overlays of the group-level harmonic maps with mPFC single-unit
cell coordinates plotted on top — mirrors the fsaverage / mne.viz.Brain
rendering used in ``scripts/cell_mask_overlap.py``.

For each dataset in DATASETS × each map in MAPS:
  * angle_deg / angle_deg_mPFC    → transformed to |angle| ∈ [0°, 180°]
    and rendered with a bright-yellow → dark-red ramp (0° = yellow;
    ±90° = orange; ±180° = dark red).  Sign is dropped by design so the
    colour is symmetric around 0°.
  * cos_group / sin_group         → RdBu_r diverging colormap centred
    on 0 (blue = negative, white = 0, red = positive).

Every map is gated by the dataset's ``amplitude.nii.gz`` so voxels with
no harmonic signal don't fake a "0° = current" reading in yellow.

Cells: filtered to ``alt_final_roi == 'mPFC'`` in
``neurons_with_ROI_labels.csv`` (canonical MNI coords from
``MNI_{x,y,z}_final``), rendered as dark-green foci on the lh + rh
medial views of the fsaverage pial surface, hemisphere-split by
sign of ``MNI_x`` and jittered by a few mm so co-located cells separate.

Outputs (PDF + PNG per (dataset × map × hemi)):
  <harmonic_angle_maps>/brain_overlays_with_mPFC_cells/<dataset>/…

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

try:
    from mne.datasets import fetch_fsaverage
    from mne.viz import Brain
    HAS_MNE_BRAIN = True
except Exception as _exc:
    print(f"[warn] mne not available: {_exc}")
    HAS_MNE_BRAIN = False

try:
    from nilearn import surface as _surface
    from nilearn.image import resample_to_img
except Exception as _exc:
    print(f"[error] nilearn required: {_exc}")
    sys.exit(1)

# Import mask helpers from cell_mask_overlap so we don't duplicate the
# DSR-main-effect / gradient-union construction logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import cell_mask_overlap as cmo
    HAS_CMO = True
except Exception as _exc:
    print(f"[warn] cell_mask_overlap unavailable: {_exc}")
    HAS_CMO = False


# ── Settings ─────────────────────────────────────────────────────────
BASE_HARMONIC = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                     '/derivatives/group/Main_Results_fMRI/harmonic_angle_maps')
CELL_TABLE = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                  '/ephys_humans/derivatives/neurons_with_ROI_labels.csv')
ROI_LABEL_COLUMN = 'alt_final_roi'
CELL_ROI_TO_PLOT = 'mPFC'

MPFC_MASK_PATH = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                      '/masks/mask_PFC_LR_smoothed_resampled.nii.gz')

OUT_ROOT = BASE_HARMONIC / 'brain_overlays_with_mPFC_cells'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = ['quarters', 'eighths']

# Each map file: short_name → filename
MAP_FILES = {
    'angle_deg': 'angle_deg.nii.gz',
    'cos_group': 'cos_group.nii.gz',
    'sin_group': 'sin_group.nii.gz',
}

# Amplitude threshold — voxels below this are treated as background.
# amplitude = √(cos_G² + sin_G²), so any voxel with real (cos, sin)
# activity crosses this trivially; only true-background voxels stay 0.
AMP_EPS = 1e-6

# For the amp-gated circular mode: percentile of the in-signal +
# in-mask amplitude distribution below which voxels are hidden.
# Milder than the Hotelling / Rayleigh p<0.05 test but strong enough
# to suppress the small-magnitude arctan2 noise (bottom ~15% of voxels
# are where the "angle jumps" between neighbours come from).
AMP_GATE_PERCENTILE = 35.0

# What to render.  (map, mask_key, mode)
#   mask_key = 'whole' | 'mPFC' | 'DSR_main' | 'gradient'  (looked up in MASKS)
#   mode     = 'abs_yellow_red' (yellow=0°, red=±180°) | 'diverging' (RdBu_r)
COMBINATIONS = [
    # Symmetric |angle| (yellow=0°, red=±180°, sign lost)
    ('angle_deg', 'whole',    'abs_yellow_red'),
    ('angle_deg', 'mPFC',     'abs_yellow_red'),
    ('angle_deg', 'DSR_main', 'abs_yellow_red'),
    ('angle_deg', 'gradient', 'abs_yellow_red'),
    # Cyclic wheel (yellow=0°, red=+90°, blue=±180°, green=-90°)
    ('angle_deg', 'whole',    'circular'),
    ('angle_deg', 'mPFC',     'circular'),
    ('angle_deg', 'DSR_main', 'circular'),
    ('angle_deg', 'gradient', 'circular'),
    # Cyclic wheel + amp gate (hides bottom AMP_GATE_PERCENTILE% of amp
    # within the mask to suppress arctan2 jitter at low-signal voxels)
    ('angle_deg', 'whole',    'circular_gated'),
    ('angle_deg', 'mPFC',     'circular_gated'),
    ('angle_deg', 'DSR_main', 'circular_gated'),
    ('angle_deg', 'gradient', 'circular_gated'),
    # Cos / sin (diverging RdBu_r, unchanged)
    ('cos_group', 'whole',    'diverging'),
    ('sin_group', 'whole',    'diverging'),
]

# Short tag per mode for output filenames so all modes coexist.
MODE_TAG = {
    'abs_yellow_red': 'abs',
    'circular':       'circ',
    'circular_gated': f'circg{int(AMP_GATE_PERCENTILE)}',
    'diverging':      'div',
}

# Custom yellow → dark-red ramp (same palette used for the gradient
# figure in cell_mask_overlap.py). Bright yellow at low end = 0°;
# dark red at top end = ±180°.  Sign of the angle is discarded.
YELLOW_RED_HEX = ['#FCE300', '#FF8C00', '#D2691E', '#8B0000', '#3D0000']

# Cyclic colour wheel: matches the reference wheel — anchor colours at
# angles -180°, -90°, 0°, +90°, +180° so opposite angles are opposite
# colours and neighbouring angles blend smoothly.  Linear interpolation
# yields teal between blue/green and orange between yellow/red.
CIRCULAR_ANCHORS_HEX = [
    '#1E88E5',   # -180°  → blue  (wraps to +180°)
    '#43A047',   #  -90°  → green
    '#FCE300',   #    0°  → yellow
    '#E53935',   #  +90°  → red
    '#1E88E5',   # +180°  → blue  (same as -180°)
]

# (AMP_EPS and AMP_GATE_PERCENTILE are defined above MODE_TAG, near
#  the head of the settings block, because MODE_TAG references
#  AMP_GATE_PERCENTILE in its f-string.)

CELL_COLOR = '#0e3d3a'   # CLAUDE.md "observed value" dark green
CELL_SCALE = 0.35
JITTER_MM  = 2.5

HEMIS = ('lh', 'rh')
VIEWS = ('medial', 'lateral')

# Fast-render filter — set to None to render every combination × hemi × view
# in the lists above.  Set to a dict to restrict to a subset (useful when
# iterating on a single plot; a full run is ~50 min).  The keys are optional
# — omit any of them to keep the full list for that axis.
RENDER_FILTER = {
    'combinations': [('angle_deg', 'gradient', 'circular_gated')],
    'views':        ('medial',),
    # 'hemis':      ('lh',),          # example: uncomment to restrict hemis
    # 'datasets':   ('quarters',),    # example: uncomment to restrict datasets
}
# RENDER_FILTER = None
BRAIN_SIZE    = (1000, 900)
SURF_ALPHA    = 0.30
OVERLAY_ALPHA = 0.75


# ── Cached fsaverage path ────────────────────────────────────────────
_FSAVERAGE_PATH = None
def _ensure_fsaverage():
    global _FSAVERAGE_PATH
    if _FSAVERAGE_PATH is not None:
        return _FSAVERAGE_PATH
    fs_dir = fetch_fsaverage()
    _FSAVERAGE_PATH = os.path.dirname(fs_dir)
    return _FSAVERAGE_PATH


def _set_brain_surface_alpha(brain, alpha):
    """Older MNE Brain versions don't accept alpha=... in the constructor.
    Walk the underlying VTK actors and set opacity directly."""
    try:
        renderer = brain._renderer
    except AttributeError:
        return
    fig = (getattr(renderer, 'plotter', None)
           or getattr(renderer, 'figure', None))
    if fig is None:
        return
    actors = getattr(fig, 'actors', None)
    if actors is None and hasattr(fig, 'renderer'):
        actors = getattr(fig.renderer, 'actors', None)
    if not actors:
        return
    iterable = actors.values() if hasattr(actors, 'values') else actors
    for actor in iterable:
        try:
            actor.GetProperty().SetOpacity(float(alpha))
        except Exception:
            continue


def _jitter(coords, mm, seed=0):
    if coords.size == 0 or mm <= 0:
        return coords
    rng = np.random.default_rng(seed)
    return coords + rng.uniform(-mm, mm, size=coords.shape)


def _resample_mask(mask_img, ref_img):
    """Nearest-neighbour resample `mask_img` onto `ref_img`'s grid if needed."""
    if (mask_img.shape[:3] == ref_img.shape[:3]
            and np.allclose(mask_img.affine, ref_img.affine, atol=1e-3)):
        return mask_img
    print(f"  [resample] {mask_img.shape[:3]} → {ref_img.shape[:3]}")
    return resample_to_img(mask_img, ref_img, interpolation='nearest')


def load_masks(ref_img):
    """Return dict {mask_key: nifti_img or None} resampled to `ref_img`.
    Silently drops any mask that fails to build/load."""
    masks = {'whole': None}
    # mPFC
    if MPFC_MASK_PATH.exists():
        masks['mPFC'] = _resample_mask(nib.load(str(MPFC_MASK_PATH)), ref_img)
        print(f"  mPFC mask: {int((masks['mPFC'].get_fdata() > 0.5).sum())} vox")
    else:
        print(f"  [warn] mPFC mask missing: {MPFC_MASK_PATH}")

    if not HAS_CMO:
        print("  [warn] cell_mask_overlap unavailable — no DSR/gradient masks.")
        return masks

    # DSR main effect (built on demand from cluster-mass PALM output).
    try:
        dsr = cmo.load_input_mask('DSR_main_effect',
                                   cmo.DSR_MAIN_EFFECT_MASK_PATH)
        masks['DSR_main'] = _resample_mask(dsr, ref_img)
        print(f"  DSR_main mask: "
              f"{int((masks['DSR_main'].get_fdata() > 0.5).sum())} vox")
    except Exception as exc:
        print(f"  [warn] DSR_main mask unavailable: {exc}")

    # Gradient (rebuild union from per-lag tstat maps).
    try:
        grad = cmo.build_gradient_union_mask(
            cmo.GRADIENT_TSTAT_DIR, cmo.GRADIENT_TSTAT_MAPS,
            cmo.GRADIENT_TSTAT_THRESHOLDS,
            prebuilt_path=cmo.GRADIENT_PREBUILT_MASK,
        )
        if grad is not None:
            masks['gradient'] = _resample_mask(grad, ref_img)
            print(f"  gradient mask: "
                  f"{int((masks['gradient'].get_fdata() > 0.5).sum())} vox")
    except Exception as exc:
        print(f"  [warn] gradient mask unavailable: {exc}")

    return masks


def load_cells():
    df = pd.read_csv(CELL_TABLE)
    for ax in ('x', 'y', 'z'):
        if f'MNI_{ax}_final' in df.columns:
            df[f'MNI_{ax}'] = df[f'MNI_{ax}_final']
    df = df.dropna(subset=[ROI_LABEL_COLUMN, 'MNI_x', 'MNI_y', 'MNI_z'])
    df = df[df[ROI_LABEL_COLUMN] == CELL_ROI_TO_PLOT].copy()
    print(f"Loaded {len(df)} '{CELL_ROI_TO_PLOT}' cells "
          f"across {df['subject'].nunique()} subjects")
    return df


def _make_brain(hemi, subjects_dir, title=None):
    try:
        brain = Brain(subject='fsaverage', hemi=hemi, surf='pial',
                      background='white', size=BRAIN_SIZE,
                      subjects_dir=subjects_dir, alpha=SURF_ALPHA,
                      title=title)
    except TypeError:
        brain = Brain(subject='fsaverage', hemi=hemi, surf='pial',
                      background='white', size=BRAIN_SIZE,
                      subjects_dir=subjects_dir, title=title)
        _set_brain_surface_alpha(brain, SURF_ALPHA)
    return brain


def _project_and_transform(nii_path, amp_path, mode, hemi, subjects_dir,
                            roi_mask_img=None):
    """Load the map + amplitude (+ optional ROI mask), project each to
    `hemi` pial surface, apply the mode-specific transform, and return
    (data, cmap, fmin, fmid, fmax) ready to hand to brain.add_data.

    When `roi_mask_img` is given, vertices outside the mask become NaN
    so MNE renders them transparent (the underlying pial surface shows
    through as light grey)."""
    img = nib.load(str(nii_path))
    amp_img = nib.load(str(amp_path))
    surf_path = os.path.join(subjects_dir, 'fsaverage', 'surf', f'{hemi}.pial')

    texture = _surface.vol_to_surf(img,     surf_path,
                                    interpolation='nearest').astype(float)
    amp_txt = _surface.vol_to_surf(amp_img, surf_path,
                                    interpolation='nearest').astype(float)
    has_signal = amp_txt > AMP_EPS
    if roi_mask_img is not None:
        roi_txt = _surface.vol_to_surf(roi_mask_img, surf_path,
                                        interpolation='nearest').astype(float)
        has_signal = has_signal & (roi_txt >= 0.5)

    if mode == 'abs_yellow_red':
        # Symmetric |angle| in degrees, 0 at yellow, 180 at dark red.
        data = np.where(has_signal, np.abs(texture), np.nan)
        cmap = LinearSegmentedColormap.from_list('yellow_to_darkred',
                                                  YELLOW_RED_HEX)
        fmin, fmid, fmax = 0.0, 90.0, 180.0
        cbar_info = {'vmin': 0.0, 'vmax': 180.0,
                     'ticks': [0, 45, 90, 135, 180],
                     'label': '|preferred angle| (°)'}
    elif mode in ('circular', 'circular_gated'):
        # Cyclic wheel: keep the sign of the angle. Shift into [1, 361]
        # so MNE's fmin can stay strictly positive (avoids the same
        # `calculate_lut` crash that hits negative fmin for diverging).
        # The colourbar we add externally uses raw −180..+180° so the
        # user never sees the shifted numbers.
        if mode == 'circular_gated' and np.any(has_signal):
            amp_thr = float(np.percentile(amp_txt[has_signal],
                                           AMP_GATE_PERCENTILE))
            has_signal = has_signal & (amp_txt > amp_thr)
        shifted = np.where(has_signal, texture + 181.0, np.nan)
        data = shifted
        cmap = LinearSegmentedColormap.from_list('circular_wheel',
                                                  CIRCULAR_ANCHORS_HEX)
        fmin, fmid, fmax = 1.0, 181.0, 361.0
        cbar_info = {'vmin': -180.0, 'vmax': 180.0,
                     'ticks': [-180, -90, 0, 90, 180],
                     'label': 'preferred angle (°)'}
    elif mode == 'diverging':
        raw = np.where(has_signal, texture, np.nan)
        vmax = float(np.nanmax(np.abs(raw))) if np.any(has_signal) else 1.0
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        # MNE's add_data can crash on negative fmin; shift the data
        # into [0, 2·vmax] so raw −vmax → 0, raw 0 → vmax (colormap
        # centre = white), raw +vmax → 2·vmax. RdBu_r goes blue→white
        # →red across [0, 1], so this preserves the diverging look.
        data = raw + vmax
        cmap = plt.get_cmap('RdBu_r')
        fmin, fmid, fmax = 0.0, vmax, 2.0 * vmax
        cbar_info = {'vmin': -vmax, 'vmax': vmax,
                     'ticks': [-vmax, 0, vmax],
                     'label': 'group mean'}
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return data, cmap, fmin, fmid, fmax, cbar_info


def render_one(ds, map_name, mask_name, nii_path, amp_path, mode,
                cells_df, subjects_dir, hemi, view,
                roi_mask_img=None):
    """Build one Brain, overlay data + cells, save PDF+PNG.

    `mask_name` is used only for output filenames / titles; the actual
    gating is via `roi_mask_img` (pass None to render whole brain)."""
    if not HAS_MNE_BRAIN:
        print("  [mne] not installed — skipping.")
        return
    if not nii_path.exists():
        print(f"  [skip] map missing: {nii_path}")
        return
    if not amp_path.exists():
        print(f"  [skip] amplitude missing: {amp_path}")
        return

    data, cmap, fmin, fmid, fmax, cbar_info = _project_and_transform(
        nii_path, amp_path, mode, hemi, subjects_dir,
        roi_mask_img=roi_mask_img)
    if not np.any(np.isfinite(data)):
        print(f"  [skip] no supra-threshold voxels on {hemi} for "
              f"{map_name}×{mask_name}")
        return

    brain = _make_brain(hemi, subjects_dir, title=None)

    # Disable MNE's built-in colourbar — we add a matplotlib one during
    # PDF embedding so the labels can show raw (unshifted) values.
    add_kwargs = dict(hemi=hemi, fmin=fmin, fmid=fmid, fmax=fmax,
                      colormap=cmap, alpha=OVERLAY_ALPHA, colorbar=False)
    try:
        brain.add_data(data, smoothing_steps=0, **add_kwargs)
    except TypeError:
        brain.add_data(data, **add_kwargs)

    # Cells (hemisphere-split)
    coords = cells_df[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(dtype=float)
    keep = coords[:, 0] <= 0 if hemi == 'lh' else coords[:, 0] >= 0
    coords_h = _jitter(coords[keep], JITTER_MM,
                        seed=hash((hemi, map_name)) & 0xFFFF)
    if coords_h.shape[0]:
        try:
            brain.add_foci(coords_h, coords_as_verts=False, hemi=hemi,
                            color=CELL_COLOR, scale_factor=CELL_SCALE,
                            name=f'{CELL_ROI_TO_PLOT}_cells')
        except Exception as exc:
            print(f"  [mne] add_foci failed: {exc}")

    try:
        brain.show_view(view)
    except Exception as exc:
        print(f"  [mne] show_view({view!r}) failed: {exc}")

    out_dir = OUT_ROOT / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = MODE_TAG.get(mode, mode)
    stem = out_dir / f'{map_name}__{mask_name}__{mode_tag}__{hemi}_{view}'
    png_path = str(stem) + '.png'
    pdf_path = str(stem) + '.pdf'
    try:
        brain.save_image(png_path)
        # Embed PNG in a PDF page with a matplotlib colourbar underneath
        # so the labels show raw values (not the shifted numbers MNE
        # would otherwise print).
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        img = plt.imread(png_path)
        h_in = img.shape[0] / 300; w_in = img.shape[1] / 300
        fig = plt.figure(figsize=(w_in, h_in + 0.55), dpi=300)
        gs = fig.add_gridspec(
            2, 1, height_ratios=[img.shape[0], 55],
            hspace=0.02, left=0.0, right=1.0, bottom=0.0, top=1.0,
        )
        ax_img = fig.add_subplot(gs[0])
        ax_img.imshow(img); ax_img.axis('off')
        ax_cb = fig.add_subplot(gs[1])
        # Shrink the colourbar horizontally so it doesn't look absurd.
        cb_pos = ax_cb.get_position()
        cb_width_frac = 0.55
        ax_cb.set_position([(1 - cb_width_frac) / 2, cb_pos.y0,
                             cb_width_frac, cb_pos.height * 0.4])
        norm = Normalize(vmin=cbar_info['vmin'], vmax=cbar_info['vmax'])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
        cbar.set_ticks(cbar_info['ticks'])
        cbar.set_ticklabels([f'{t:g}' for t in cbar_info['ticks']])
        cbar.set_label(cbar_info['label'], fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        # Also overwrite the PNG so both formats have the fixed colourbar.
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  wrote {pdf_path}")
    except Exception as exc:
        print(f"  [mne] save failed: {exc}")
    finally:
        try:
            brain.close()
        except Exception:
            pass


def main():
    cells_df = load_cells()
    subjects_dir = _ensure_fsaverage()
    if subjects_dir is None:
        print("[error] fsaverage unavailable — aborting.")
        return

    # Reference image: any harmonic output (they share the fMRI grid).
    ref_img = None
    for ds in DATASETS:
        cand = BASE_HARMONIC / ds / 'amplitude.nii.gz'
        if cand.exists():
            ref_img = nib.load(str(cand))
            break
    if ref_img is None:
        print("[error] no harmonic outputs found — aborting.")
        return

    print("\n=== Loading masks ===")
    masks = load_masks(ref_img)
    print(f"Masks available: {sorted(masks.keys())}")

    # Apply optional render filter (see RENDER_FILTER at the top).
    rf = RENDER_FILTER or {}
    combos_to_run = rf.get('combinations', COMBINATIONS)
    views_to_run  = rf.get('views',        VIEWS)
    hemis_to_run  = rf.get('hemis',        HEMIS)
    ds_to_run     = rf.get('datasets',     DATASETS)
    if RENDER_FILTER:
        print(f"\n[RENDER_FILTER] active — restricting to:")
        print(f"  combinations: {combos_to_run}")
        print(f"  views:        {views_to_run}")
        print(f"  hemis:        {hemis_to_run}")
        print(f"  datasets:     {ds_to_run}")

    for ds in ds_to_run:
        ds_dir = BASE_HARMONIC / ds
        if not ds_dir.exists():
            print(f"[skip] {ds}: {ds_dir} not found")
            continue
        amp_path = ds_dir / 'amplitude.nii.gz'
        print(f"\n{'='*70}\n=== {ds} ===\n{'='*70}")
        for map_name, mask_name, mode in combos_to_run:
            if mask_name not in masks:
                print(f"\n[{ds}]  {map_name} × {mask_name}  — mask unavailable, skipping")
                continue
            fname = MAP_FILES[map_name]
            nii_path = ds_dir / fname
            roi_mask_img = masks[mask_name]
            print(f"\n[{ds}]  {map_name} × {mask_name}  ({mode})  ← {fname}")
            for hemi in hemis_to_run:
                for view in views_to_run:
                    render_one(ds, map_name, mask_name, nii_path, amp_path,
                                mode, cells_df, subjects_dir, hemi, view,
                                roi_mask_img=roi_mask_img)

    print(f"\nAll overlays under: {OUT_ROOT}")


if __name__ == '__main__':
    main()
