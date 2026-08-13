#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the PC1 gradient axis and the ventral/dorsal split boundary on the
tier MNE surface, so the split geometry is visible.

  * cells coloured by PC1 tier (ventral 30 / dorsal 60 / outside grey),
  * PC1 axis = black line of foci through the gradient-mask centroid,
  * split boundary (median plane) = red line of foci (perpendicular to PC1),
  * the right-hemisphere misfit site (MNI ~7.9, 41.9, 2.9; classified dorsal
    but sits ventral) marked in magenta.

Backdrop = quarters fMRI angle on the gradient mask (as in the tier figure).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

import gradient_brain_cells_by_lag as gb
from mne.datasets import fetch_fsaverage
from mne.viz import Brain

OUT = gb.MASTER_DIR / 'gradient_brain_cells_by_lag'
L = np.arange(0, 360, 30)
MISFIT_MNI = np.array([7.896969, 41.945395, 2.855552])   # RH dorsal-but-ventral


def gradient_axis():
    m = nib.load(str(gb.GRAD15_MASK_PATH))
    mni = nib.affines.apply_affine(m.affine, np.argwhere(m.get_fdata() > 0))
    mni[:, 0] = np.abs(mni[:, 0])
    centroid = mni.mean(0)
    _, _, vt = np.linalg.svd(mni - centroid, full_matrices=False)
    pc1 = vt[0]
    if pc1[2] < 0:
        pc1 = -pc1
    return centroid, pc1


def main():
    cells = pd.read_csv(gb.MASTER_DIR / 'per_cell_master.csv')
    colours, label = gb.tier_colours(cells)
    coords = cells[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)
    proj = cells['grad_axis_coord'].to_numpy(float)
    inm = cells['in_gradient_mask'].to_numpy(bool)
    med = np.median(proj[inm])
    centroid, pc1 = gradient_axis()

    # PC1 axis line: points centroid + s*pc1 over the in-mask projection range.
    s = np.linspace(proj[inm].min() - 1, proj[inm].max() + 1, 40)
    line_fold = centroid[None] + s[:, None] * pc1[None]      # folded x
    # Boundary line: through the median point, perpendicular to PC1 in the
    # sagittal (y,z) plane.
    pmed = centroid + med * pc1
    perp = np.array([0.0, -pc1[2], pc1[1]])
    perp /= np.linalg.norm(perp)
    u = np.linspace(-12, 12, 30)
    bnd_fold = pmed[None] + u[:, None] * perp[None]

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
                               colormap=gb.CIRC_CMAP, alpha=0.55,
                               colorbar=False, smoothing_steps=5)
            except TypeError:
                brain.add_data(data, hemi=hemi, fmin=1, fmid=181, fmax=361,
                               colormap=gb.CIRC_CMAP, alpha=0.55, colorbar=False)

        def to_hemi(pts_fold):
            p = pts_fold.copy()
            p[:, 0] = sign * np.abs(p[:, 0])
            return p

        # PC1 axis (black) and boundary (red)
        brain.add_foci(to_hemi(line_fold), coords_as_verts=False, hemi=hemi,
                       color='black', scale_factor=0.22)
        brain.add_foci(to_hemi(bnd_fold), coords_as_verts=False, hemi=hemi,
                       color='#d11', scale_factor=0.22)

        # cells (this hemisphere)
        keep = coords[:, 0] <= 0 if hemi == 'lh' else coords[:, 0] >= 0
        rng = np.random.default_rng(hash((hemi, 'tier')) & 0xFFFF)
        cj = coords[keep] + rng.uniform(-2.5, 2.5, coords[keep].shape)
        for cc, col in zip(cj, colours[keep]):
            brain.add_foci(cc[None], coords_as_verts=False, hemi=hemi,
                           color=tuple(col[:3]), scale_factor=0.42)
        # misfit site (rh only), magenta ring-ish big marker
        if hemi == 'rh':
            brain.add_foci(MISFIT_MNI[None], coords_as_verts=False, hemi=hemi,
                           color='#ff00ff', scale_factor=0.8)
        try:
            brain.show_view('medial')
        except Exception:
            pass
        png = str(OUT / f'tier_split_geometry_{hemi}_medial.png')
        brain.save_image(png)
        _cbar(png, hemi, label)
        try:
            brain.close()
        except Exception:
            pass
        print(f'  wrote {png} (+pdf)')


def _cbar(png, hemi, label):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    img = plt.imread(png)
    fig = plt.figure(figsize=(img.shape[1]/300, img.shape[0]/300 + 0.7), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[img.shape[0], 70],
                          hspace=0.02, left=0, right=1, bottom=0, top=1)
    a = fig.add_subplot(gs[0]); a.imshow(img); a.axis('off')
    cax = fig.add_subplot(gs[1]); pos = cax.get_position()
    cax.set_position([0.22, pos.y0, 0.5, pos.height * 0.32])
    sm = ScalarMappable(cmap=gb.CIRC_CMAP, norm=Normalize(-180, 180))
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks([-180, -90, 0, 90, 180])
    cb.set_label(f'preferred angle (deg).  black=PC1 axis, red=ventral/dorsal '
                 f'boundary, magenta=RH misfit site.  {label} [{hemi}]',
                 fontsize=7)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
