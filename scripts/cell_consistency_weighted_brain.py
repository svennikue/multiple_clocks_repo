#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pretty, believable visualisation: mPFC cells on the fMRI-angle gradient,
each coloured by its OWN preferred lag but weighted by spatial consistency.

Every cell contributes its favourite peak (arg-max lag -> cyclic colour), but
its visual weight (marker size + colour saturation) is scaled by its spatial
consistency (peak cross-validated r). Reliable, strongly-tuned cells are large
and saturated; noisy cells fade to small grey dots. This lets the eye read the
consistency-weighted population preference against the fMRI gradient backdrop,
without any pooling/statistics.

Also draws the consistency-weighted population-vector arrow for the ventral and
dorsal halves (PC1 split) as a compact summary.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import gradient_brain_cells_by_lag as gb
from mne.datasets import fetch_fsaverage
from mne.viz import Brain

OUT = gb.MASTER_DIR / 'gradient_brain_cells_by_lag'
L = np.arange(0, 360, 30)
GREY = np.array([0.72, 0.72, 0.72])
R_LO, R_HI = 0.15, 0.45          # consistency (peak r) -> [0,1] mapping range


def main():
    c = pd.read_csv(gb.MASTER_DIR / 'per_cell_master.csv')
    R = c[[f'r_lag{a:03d}_noctrl' for a in L]].to_numpy(float)
    coords = c[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)
    ang = L[np.nanargmax(R, 1)].astype(float)
    peakr = R[np.arange(len(R)), np.nanargmax(R, 1)]
    cnorm = np.clip((peakr - R_LO) / (R_HI - R_LO), 0, 1)     # 0..1 consistency

    # colour = wheel(angle) blended toward grey by (1-consistency)
    wheel = gb.angles_to_colours(ang)[:, :3]
    cols = wheel * cnorm[:, None] + GREY[None] * (1 - cnorm[:, None])
    sizes = 0.16 + 0.46 * cnorm

    subjects_dir = os.path.dirname(fetch_fsaverage())
    import pdb; pdb.set_trace()
    for hemi in ('lh', 'rh'):
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
        keep = coords[:, 0] <= 0 if hemi == 'lh' else coords[:, 0] >= 0
        rng = np.random.default_rng(hash((hemi, 'cw')) & 0xFFFF)
        cj = coords[keep] + rng.uniform(-2.0, 2.0, coords[keep].shape)
        for cc, col, sz in zip(cj, cols[keep], sizes[keep]):
            brain.add_foci(cc[None], coords_as_verts=False, hemi=hemi,
                           color=tuple(col), scale_factor=float(sz))
        try:
            brain.show_view('medial')
        except Exception:
            pass
        png = str(OUT / f'consistency_weighted_cells_{hemi}_medial.png')
        brain.save_image(png)
        _cbar(png, hemi)
        try:
            brain.close()
        except Exception:
            pass
        print(f'  wrote {png} (+pdf)')

    _polar(c, R, ang, peakr)


def _cbar(png, hemi):
    img = plt.imread(png)
    fig = plt.figure(figsize=(img.shape[1]/300, img.shape[0]/300 + 0.65), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[img.shape[0], 65],
                          hspace=0.02, left=0, right=1, bottom=0, top=1)
    a = fig.add_subplot(gs[0]); a.imshow(img); a.axis('off')
    cax = fig.add_subplot(gs[1]); pos = cax.get_position()
    cax.set_position([0.24, pos.y0, 0.5, pos.height * 0.34])
    sm = ScalarMappable(cmap=gb.CIRC_CMAP, norm=Normalize(-180, 180))
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks([-180, -90, 0, 90, 180])
    cb.set_label(f"cell preferred lag (deg); marker size & saturation = spatial "
                 f"consistency (peak r). backdrop: fMRI gradient [{hemi}]",
                 fontsize=7)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


def _polar(c, R, ang, peakr):
    """Consistency-weighted preferred-angle distribution, ventral vs dorsal."""
    inm = c['in_gradient_mask'].to_numpy(bool)
    proj = c['grad_axis_coord'].to_numpy(float)
    med = np.median(proj[inm])
    groups = [('ventral', inm & (proj <= med), '#0a607a'),
              ('dorsal', inm & (proj > med), '#5C1027')]
    fig = plt.figure(figsize=(6, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
    rmax = 0.75
    for name, sel, col in groups:
        th = np.radians(ang[sel]); rr = peakr[sel]
        w = np.clip(rr, 0, None) ** 2
        C = np.sum(w * np.cos(th)); S = np.sum(w * np.sin(th))
        res = np.arctan2(S, C); coh = np.hypot(C, S) / w.sum()
        ax.scatter(th, rr, s=28, c=[col], alpha=0.45, edgecolor='none',
                   label=f'{name} (n={sel.sum()}): vector {np.degrees(res)%360:.0f}deg, '
                         f'R={coh:.2f}')
        # arrow drawn to a readable length (scaled), annotated at the rim
        ax.annotate('', xy=(res, rmax * coh / 0.25), xytext=(0, 0),
                    arrowprops=dict(color=col, width=3.5, headwidth=13,
                                    alpha=0.9))
    ax.set_rlim(0, rmax); ax.set_rlabel_position(112)
    ax.set_title('Consistency-weighted preferred angle per tier\n'
                 'dots = cells (radius = spatial consistency, peak r);  '
                 'arrows = population vector (weight r^2, length ~ coherence)',
                 fontsize=9)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.16), fontsize=8,
              frameon=False, ncol=1)
    fig.savefig(OUT / 'consistency_weighted_polar.png', dpi=200)
    fig.savefig(OUT / 'consistency_weighted_polar.pdf')
    plt.close(fig)
    print(f'  wrote {OUT / "consistency_weighted_polar.png"} (+pdf)')


if __name__ == '__main__':
    main()
