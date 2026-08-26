#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Medial fsaverage surfaces: quarters fMRI preferred-angle on the
gradient_thr_1.5 mask, with mPFC cells plotted on top and coloured by
their OWN preferred lag on the same cyclic colour wheel.

Backdrop processing matches harmonic_maps_brain_overlay.py's circular mode:
  * project cos/sin group volumes (quarters), arctan2 per vertex,
  * BILATERAL_SYMMETRISE  (x-flip cos/sin and average),
  * PRE_PROJ_SMOOTH_FWHM_MM = 3 mm Gaussian on cos/sin,
  * SURFACE_SMOOTHING_STEPS = 5 MNE mesh-neighbour iterations,
  * gated to the gradient_thr_1.5 mask.

Two cell-colourings are produced (each lh + rh medial → 2 PNGs):
  * ``argmax``   — cell's discrete argmax lag,
  * ``harmonic`` — cell's continuous first-harmonic angle.

Reads the latest per_cell_master.csv (built by cell_gradient_master_table.py)
so cell coordinates and preferences stay consistent. PNG only.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

from mne.datasets import fetch_fsaverage
from mne.viz import Brain
from nilearn import surface as _surface
from nilearn.image import new_img_like, smooth_img


# ── Settings ─────────────────────────────────────────────────────────
MASTER_DIR = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
    '/derivatives/group/cell_gradient_master/2026-08-13_09-45-28'
)
HARMONIC_ROOT = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
    '/derivatives/group/Main_Results_fMRI/harmonic_angle_maps'
    '/unit_vector_derived'
)
DATASET = 'quarters'
GRAD15_MASK_PATH = Path('/Users/xpsy1114/Documents/projects/multiple_clocks'
                        '/data/masks/gradient_thr_1.5.nii.gz')
OUT_DIR = MASTER_DIR / 'gradient_brain_cells_by_lag'

PRE_PROJ_SMOOTH_FWHM_MM = 3.0
SURFACE_SMOOTHING_STEPS = 5
BILATERAL_SYMMETRISE = True
AMP_EPS = 1e-6
OVERLAY_ALPHA = 0.75
CELL_SCALE = 0.4
JITTER_MM = 2.5
BRAIN_SIZE = (1000, 900)
SURF_ALPHA = 0.30

CELL_COLOURINGS = {
    'argmax': 'argmax_lag_deg',
    'harmonic': 'harmonic_angle_deg',
    'tier': None,          # out=grey; in-mask low-z=ventral / high-z=dorsal
    'ztercile': None,      # all cells, equal-n z-terciles, coloured by pooled lag
}
# Only render these this run (set to None to render all).
RENDER_ONLY = ['tier']
LAGS_DEG = np.arange(0, 360, 30)
OUTSIDE_GREY = (0.6, 0.6, 0.6, 1.0)

# Cyclic wheel: -180 blue, -90 green, 0 yellow, +90 red, +180 blue.
CIRCULAR_ANCHORS_HEX = ['#1E88E5', '#43A047', '#FCE300', '#E53935', '#1E88E5']
CIRC_CMAP = LinearSegmentedColormap.from_list('circular_wheel',
                                              CIRCULAR_ANCHORS_HEX)


def _symmetrise(img):
    d = img.get_fdata()
    return new_img_like(img, (d + d[::-1, ...]) / 2.0)


def angles_to_colours(angles_360):
    """[0,360) angle -> signed (-180,180] -> cyclic wheel RGBA."""
    a = np.asarray(angles_360, float)
    signed = ((a + 180.0) % 360.0) - 180.0
    return CIRC_CMAP(Normalize(vmin=-180, vmax=180)(signed))


def backdrop_texture(hemi, subjects_dir):
    """Return the +181-shifted angle texture for the gradient-masked
    backdrop on this hemisphere (NaN outside mask / no signal)."""
    surf = os.path.join(subjects_dir, 'fsaverage', 'surf', f'{hemi}.pial')
    cos_img = nib.load(str(HARMONIC_ROOT / DATASET / 'cos_group.nii.gz'))
    sin_img = nib.load(str(HARMONIC_ROOT / DATASET / 'sin_group.nii.gz'))
    if BILATERAL_SYMMETRISE:
        cos_img, sin_img = _symmetrise(cos_img), _symmetrise(sin_img)
    if PRE_PROJ_SMOOTH_FWHM_MM:
        cos_img = smooth_img(cos_img, PRE_PROJ_SMOOTH_FWHM_MM)
        sin_img = smooth_img(sin_img, PRE_PROJ_SMOOTH_FWHM_MM)
    cos_txt = _surface.vol_to_surf(cos_img, surf, interpolation='linear')
    sin_txt = _surface.vol_to_surf(sin_img, surf, interpolation='linear')
    angle = np.degrees(np.arctan2(sin_txt, cos_txt))
    amp = np.hypot(cos_txt, sin_txt)
    mask_txt = _surface.vol_to_surf(
        nib.load(str(GRAD15_MASK_PATH)), surf, interpolation='nearest')
    keep = (amp > AMP_EPS) & (mask_txt >= 0.5) & np.isfinite(angle)
    return np.where(keep, angle + 181.0, np.nan)


def _pooled_argmax(R, sel):
    return int(LAGS_DEG[np.nanargmax(np.nanmean(R[sel], 0))])


def tier_colours(cells):
    """out-of-mask = grey; in-mask ventral/dorsal split along the PC1 gradient
    axis, each coloured by that group's pooled-argmax lag on the cyclic wheel.
    (PC1 median split reproduces the ventral=30 / dorsal=60 figure.)"""
    inm = cells['in_gradient_mask'].to_numpy(bool)
    proj = cells['grad_axis_coord'].to_numpy(float)
    R = cells[[f'r_lag{a:03d}_noctrl' for a in LAGS_DEG]].to_numpy(float)
    med = np.median(proj[inm])
    ventral, dorsal = inm & (proj <= med), inm & (proj > med)
    ang_v, ang_d = _pooled_argmax(R, ventral), _pooled_argmax(R, dorsal)
    colours = np.tile(OUTSIDE_GREY, (len(cells), 1))
    colours[ventral] = angles_to_colours([ang_v])[0]
    colours[dorsal] = angles_to_colours([ang_d])[0]
    print(f'  tier pooled argmax (PC1): ventral={ang_v}, dorsal={ang_d}')
    return colours, (f'ventral-in-mask={ang_v}deg, '
                     f'dorsal-in-mask={ang_d}deg, outside=grey')


def ztercile_colours(cells):
    """All cells split into equal-n z-terciles, each coloured by its
    pooled-argmax lag (ignores mask membership)."""
    z = cells['MNI_z'].to_numpy(float)
    R = cells[[f'r_lag{a:03d}_noctrl' for a in LAGS_DEG]].to_numpy(float)
    q = np.quantile(z, [1/3, 2/3])
    groups = {'low': z <= q[0], 'mid': (z > q[0]) & (z <= q[1]), 'high': z > q[1]}
    colours = np.tile(OUTSIDE_GREY, (len(cells), 1))
    angs = {}
    for name, sel in groups.items():
        angs[name] = _pooled_argmax(R, sel)
        colours[sel] = angles_to_colours([angs[name]])[0]
    print(f'  z-tercile pooled argmax: {angs}  edges={np.round(q,1)}')
    lo, mi, hi = angs['low'], angs['mid'], angs['high']
    return colours, (f'low-z={lo}deg, mid-z={mi}deg, '
                     f'high-z={hi}deg (equal-n z-terciles)')


def render(cells, colours, tag, subjects_dir, label):
    coords = cells[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)
    for hemi in ('lh', 'rh'):
        try:
            brain = Brain(subject='fsaverage', hemi=hemi, surf='pial',
                          background='white', size=BRAIN_SIZE,
                          subjects_dir=subjects_dir, alpha=SURF_ALPHA)
        except TypeError:
            brain = Brain(subject='fsaverage', hemi=hemi, surf='pial',
                          background='white', size=BRAIN_SIZE,
                          subjects_dir=subjects_dir)
        data = backdrop_texture(hemi, subjects_dir)
        if np.any(np.isfinite(data)):
            try:
                brain.add_data(data, hemi=hemi, fmin=1.0, fmid=181.0,
                               fmax=361.0, colormap=CIRC_CMAP,
                               alpha=OVERLAY_ALPHA, colorbar=False,
                               smoothing_steps=SURFACE_SMOOTHING_STEPS)
            except TypeError:
                brain.add_data(data, hemi=hemi, fmin=1.0, fmid=181.0,
                               fmax=361.0, colormap=CIRC_CMAP,
                               alpha=OVERLAY_ALPHA, colorbar=False)

        keep = coords[:, 0] <= 0 if hemi == 'lh' else coords[:, 0] >= 0
        rng = np.random.default_rng(hash((hemi, tag)) & 0xFFFF)
        ch = coords[keep] + rng.uniform(-JITTER_MM, JITTER_MM,
                                        size=coords[keep].shape)
        for c, col in zip(ch, colours[keep]):
            try:
                brain.add_foci(c[np.newaxis, :], coords_as_verts=False,
                               hemi=hemi, color=tuple(col[:3]),
                               scale_factor=CELL_SCALE)
            except Exception as exc:
                print(f'  add_foci failed: {exc}')
        try:
            brain.show_view('medial')
        except Exception:
            pass

        png = str(OUT_DIR / f'gradient_backdrop_cells_{tag}_{hemi}_medial.png')
        brain.save_image(png)
        _add_colourbar(png, tag, hemi, int(keep.sum()), label)
        try:
            brain.close()
        except Exception:
            pass
        print(f'  wrote {png}')


def _add_colourbar(png, tag, hemi, n_cells, label):
    img = plt.imread(png)
    fig = plt.figure(figsize=(img.shape[1] / 300, img.shape[0] / 300 + 0.6),
                     dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[img.shape[0], 60],
                          hspace=0.02, left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(gs[0]); ax.imshow(img); ax.axis('off')
    cax = fig.add_subplot(gs[1])
    pos = cax.get_position()
    cax.set_position([0.25, pos.y0, 0.5, pos.height * 0.4])
    sm = ScalarMappable(cmap=CIRC_CMAP, norm=Normalize(vmin=-180, vmax=180))
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_ticks([-180, -90, 0, 90, 180])
    cbar.set_label(f'preferred angle (deg) — backdrop: fMRI quarters '
                   f'(gradient mask);  cells: {tag}  '
                   f'[{hemi}, n={n_cells}]\n{label}', fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(MASTER_DIR / 'per_cell_master.csv')
    subjects_dir = os.path.dirname(fetch_fsaverage())
    print(f'{len(cells)} mPFC cells; output -> {OUT_DIR}')
    for tag, col in CELL_COLOURINGS.items():
        if RENDER_ONLY and tag not in RENDER_ONLY:
            continue
        print(f'\n=== cell colouring: {tag} ===')
        if tag == 'tier':
            sub = cells.dropna(subset=['MNI_z'])
            colours, label = tier_colours(sub)
        elif tag == 'ztercile':
            sub = cells.dropna(subset=['MNI_z'])
            colours, label = ztercile_colours(sub)
        else:
            sub = cells.dropna(subset=[col])
            colours = angles_to_colours(sub[col].to_numpy(float))
            label = f'cells coloured by continuous {tag} lag on the wheel'
        render(sub, colours, tag, subjects_dir, label)
    print(f'\nDone. PNGs in {OUT_DIR}')


if __name__ == '__main__':
    main()
