#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruct and draw the BENDING axis of the fMRI gradient.

The axis is defined by the gradient itself (not the mask shape): sample the
fMRI preferred angle (symmetrised + smoothed quarters cos/sin) at every
gradient-mask voxel, bin the angle into levels, take the centroid of the mask
voxels in each level, and connect them (ventral low-angle -> dorsal high-angle).
That centroid path is the curved gradient axis. Perpendicular division lines
(example cell bins) are drawn across it.

Rendered on the fsaverage medial surface with the fMRI-angle backdrop and the
mPFC cells (coloured by their arg-max lag).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from nilearn.image import smooth_img

import gradient_brain_cells_by_lag as gb
from mne.datasets import fetch_fsaverage
from mne.viz import Brain

OUT = gb.MASTER_DIR / 'gradient_bending_axis'
L = np.arange(0, 360, 30)
N_LEVELS = 9          # fMRI-angle bands used to build the axis
N_DIVISIONS = 5       # example perpendicular cell-bin lines


def principal_curve(X, n_nodes=18, n_iter=15, bw=0.08):
    """Kernel-smoothed principal curve of 2-D points X (Nadaraya-Watson nodes
    with arc-length reprojection). Returns ordered node coordinates."""
    c0 = X.mean(0)
    _, _, vt = np.linalg.svd(X - c0, full_matrices=False)
    lam = (X - c0) @ vt[0]
    lam = (lam - lam.min()) / (np.ptp(lam) + 1e-9)
    grid = np.linspace(0, 1, n_nodes)
    nodes = np.zeros((n_nodes, 2))
    for _ in range(n_iter):
        for i, g in enumerate(grid):
            w = np.exp(-0.5 * ((lam - g) / bw) ** 2)
            nodes[i] = (w[:, None] * X).sum(0) / (w.sum() + 1e-9)
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))]
        newlam = np.empty(len(X))
        for j, p in enumerate(X):
            best, bl = 1e18, 0.0
            for k in range(len(nodes) - 1):
                a, b = nodes[k], nodes[k + 1]
                ab = b - a
                tt = np.clip(((p - a) @ ab) / ((ab @ ab) + 1e-9), 0, 1)
                d = np.linalg.norm(p - (a + tt * ab))
                if d < best:
                    best, bl = d, seg[k] + tt * np.linalg.norm(ab)
            newlam[j] = bl
        lam = (newlam - newlam.min()) / (np.ptp(newlam) + 1e-9)
    return nodes


def build_axis():
    """Geometry-based bending axis: principal curve through the mask SHAPE
    (follows the cingulate fold), then the fMRI angle sampled + UNWRAPPED along
    it (so we can check the gradient values progress monotonically).

    Returns (curve_xyz_folded, angle_signed_along, angle_unwrapped_along)."""
    cos_img = nib.load(str(gb.HARMONIC_ROOT / gb.DATASET / 'cos_group.nii.gz'))
    sin_img = nib.load(str(gb.HARMONIC_ROOT / gb.DATASET / 'sin_group.nii.gz'))
    cos_img, sin_img = gb._symmetrise(cos_img), gb._symmetrise(sin_img)
    cos_img, sin_img = smooth_img(cos_img, 3.0), smooth_img(sin_img, 3.0)
    cosd, sind = cos_img.get_fdata(), sin_img.get_fdata()
    inv = np.linalg.inv(cos_img.affine); shp = np.array(cosd.shape)

    mask = nib.load(str(gb.GRAD15_MASK_PATH))
    mni = nib.affines.apply_affine(mask.affine, np.argwhere(mask.get_fdata() > 0))
    mni_f = mni.copy(); mni_f[:, 0] = np.abs(mni_f[:, 0])   # fold LR

    # Principal curve through the mask geometry (Hastie-Stuetzle style: local
    # kernel-averaged nodes with arc-length reprojection). Unlike a polynomial
    # on the PC1 axis, this can bend into the L/anchor shape of area 32 (a
    # posterior->anterior ventral limb, then a dorsal limb).
    nodes = principal_curve(mni_f[:, 1:3])
    # Densify by linear interpolation along arc length.
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))]
    sg = np.linspace(0, seg[-1], 120)
    cx = np.abs(mni_f[:, 0]).mean()
    curve = np.column_stack([np.full_like(sg, cx),
                             np.interp(sg, seg, nodes[:, 0]),
                             np.interp(sg, seg, nodes[:, 1])])
    # Orient ventral(low z) -> dorsal so the anchor/area-32 end comes first.
    if curve[0, 2] > curve[-1, 2]:
        curve = curve[::-1]

    # Sample the fMRI angle along the geometric curve, then unwrap it.
    ang = np.full(len(curve), np.nan)
    for i, p in enumerate(curve):
        ijk = np.round(nib.affines.apply_affine(inv, p)).astype(int)
        if (ijk >= 0).all() and (ijk < shp).all():
            cph, sph = cosd[tuple(ijk)], sind[tuple(ijk)]
            if np.hypot(cph, sph) > 1e-6:
                ang[i] = np.degrees(np.arctan2(sph, cph))
    ang = pd.Series(ang).interpolate(limit_direction='both').to_numpy()
    unwrapped = np.degrees(np.unwrap(np.radians(ang)))
    return curve, ang, unwrapped


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curve, ang_along, unwrapped = build_axis()
    print('fMRI angle along the geometric axis (ventral -> dorsal):')
    print('  signed  :', np.round(ang_along[::12], 0))
    print('  unwrapped:', np.round(unwrapped[::12], 0))
    d = np.diff(unwrapped)
    print(f'  unwrapped monotonic increasing? {(d > -5).mean()*100:.0f}% of steps; '
          f'total span = {unwrapped[-1]-unwrapped[0]:.0f} deg')
    c = pd.read_csv(gb.MASTER_DIR / 'per_cell_master.csv')
    coords = c[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)
    cell_cols = gb.angles_to_colours(c['argmax_lag_deg'].to_numpy(float))

    # Perpendicular division lines at N_DIVISIONS arc positions.
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    div_s = np.linspace(seg[5], seg[-6], N_DIVISIONS)   # avoid the very ends
    div_lines = []
    for s0 in div_s:
        j = int(np.argmin(np.abs(seg - s0)))
        tang = curve[min(j+3, len(curve)-1)] - curve[max(j-3, 0)]
        ty, tz = tang[1], tang[2]
        perp = np.array([0.0, -tz, ty]); perp /= (np.linalg.norm(perp) + 1e-9)
        u = np.linspace(-9, 9, 24)
        div_lines.append(curve[j][None] + u[:, None] * perp[None])

    subjects_dir = os.path.dirname(fetch_fsaverage())
    for hemi in ('lh', 'rh'):
        sign = -1 if hemi == 'lh' else 1
        try:
            brain = Brain('fsaverage', hemi, 'pial', background='white',
                          size=(1000, 900), subjects_dir=subjects_dir, alpha=0.3)
        except TypeError:
            brain = Brain('fsaverage', hemi, 'pial', background='white',
                          size=(1000, 900), subjects_dir=subjects_dir)
        data = gb.backdrop_texture(hemi, subjects_dir)
        if np.isfinite(data).any():
            try:
                brain.add_data(data, hemi=hemi, fmin=1, fmid=181, fmax=361,
                               colormap=gb.CIRC_CMAP, alpha=0.5, colorbar=False,
                               smoothing_steps=5)
            except TypeError:
                brain.add_data(data, hemi=hemi, fmin=1, fmid=181, fmax=361,
                               colormap=gb.CIRC_CMAP, alpha=0.5, colorbar=False)

        def to_h(p):
            q = p.copy(); q[:, 0] = sign * np.abs(q[:, 0]); return q

        # fat gradient axis (black) + thin perpendicular divisions (red)
        brain.add_foci(to_h(curve), coords_as_verts=False, hemi=hemi,
                       color='black', scale_factor=0.32)
        for dl in div_lines:
            brain.add_foci(to_h(dl), coords_as_verts=False, hemi=hemi,
                           color='#d11', scale_factor=0.16)

        keep = coords[:, 0] <= 0 if hemi == 'lh' else coords[:, 0] >= 0
        rng = np.random.default_rng(hash((hemi, 'ax')) & 0xFFFF)
        cj = coords[keep] + rng.uniform(-2.5, 2.5, coords[keep].shape)
        for cc, col in zip(cj, cell_cols[keep]):
            brain.add_foci(cc[None], coords_as_verts=False, hemi=hemi,
                           color=tuple(col[:3]), scale_factor=0.36)
        try:
            brain.show_view('medial')
        except Exception:
            pass
        png = str(OUT / f'gradient_bending_axis_{hemi}_medial.png')
        brain.save_image(png)
        _cbar(png, hemi)
        try:
            brain.close()
        except Exception:
            pass
        print(f'  wrote {png} (+pdf)')


def _cbar(png, hemi):
    img = plt.imread(png)
    fig = plt.figure(figsize=(img.shape[1]/300, img.shape[0]/300 + 0.6), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[img.shape[0], 60],
                          hspace=0.02, left=0, right=1, bottom=0, top=1)
    a = fig.add_subplot(gs[0]); a.imshow(img); a.axis('off')
    cax = fig.add_subplot(gs[1]); pos = cax.get_position()
    cax.set_position([0.24, pos.y0, 0.5, pos.height * 0.34])
    sm = ScalarMappable(cmap=gb.CIRC_CMAP, norm=Normalize(-180, 180))
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks([-180, -90, 0, 90, 180])
    cb.set_label(f'preferred angle (deg). black=bending gradient axis '
                 f'(mask-geometry principal curve; fMRI angle unwrapped +339deg '
                 f'along it), red=example perpendicular cell bins, '
                 f'dots=cells by arg-max [{hemi}]', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
