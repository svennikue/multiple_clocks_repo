#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anterior/posterior (y-axis) split figures, subject-weighted.

In the ventral part of the gradient the fMRI angle runs anterior->posterior
(Spearman(fMRI, y) = +0.73 > +0.50 for z). Splitting the in-mask cells at the
median y and pooling SUBJECT-weighted gives posterior ~60deg (fMRI ~66deg) and
anterior ~90deg (fMRI ~87deg).

Boundary convention: posterior = y <= median, anterior = y > median (the
tie-cluster at y=median goes to posterior; the < convention would flip it and
send the anterior group to 240deg - see the 'tie side' caveat).

Produces three figures (mirroring the PC1 tier set):
  1. lag-profile figure (subject-weighted pooled profiles),
  2. MNE surface with the anterior-posterior axis + coronal split line drawn,
  3. MNE surface with cells coloured by their group's subject-weighted angle.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import gradient_brain_cells_by_lag as gb
from mne.datasets import fetch_fsaverage
from mne.viz import Brain

OUT = gb.MASTER_DIR / 'antpost_split'
L = np.arange(0, 360, 30)


def subj_profile(R, sel, subj):
    return np.nanmean(np.vstack(
        [np.nanmean(R[sel & (subj == s)], 0) for s in np.unique(subj[sel])]), 0)


def circ_mean(a):
    a = np.radians(np.asarray(a, float)); a = a[np.isfinite(a)]
    return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(gb.MASTER_DIR / 'per_cell_master.csv')
    R = c[[f'r_lag{a:03d}_noctrl' for a in L]].to_numpy(float)
    subj = c['subject_id'].to_numpy()
    y = c['MNI_y'].to_numpy(float); z = c['MNI_z'].to_numpy(float)
    coords = c[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)
    fmri = c['fmri_angle_quarters_deg'].to_numpy(float)
    inm = c['in_gradient_mask'].to_numpy(bool)
    med = np.median(y[inm])
    post = inm & (y <= med); ant = inm & (y > med); out = ~inm

    groups = {'posterior (y<=med)': post, 'anterior (y>med)': ant,
              'outside mask': out}
    angs = {}
    for nm, sel in groups.items():
        p = subj_profile(R, sel, subj)
        angs[nm] = int(L[np.nanargmax(p)])
        print(f'{nm:20s} n={sel.sum():3d} subj-w argmax={angs[nm]:3d} '
              f'peak_r={np.nanmax(p):.3f}  fMRI={circ_mean(fmri[sel]):.0f}')

    # ---- Figure 1: subject-weighted lag profiles ----
    fig, axp = plt.subplots(figsize=(5.6, 4), constrained_layout=True)
    line_cols = {'posterior (y<=med)': gb.angles_to_colours([angs['posterior (y<=med)']])[0],
                 'anterior (y>med)': gb.angles_to_colours([angs['anterior (y>med)']])[0],
                 'outside mask': (0.6, 0.6, 0.6, 1)}
    for nm, sel in groups.items():
        p = subj_profile(R, sel, subj)
        axp.plot(L, p, '-o', color=line_cols[nm], ms=4,
                 label=f'{nm} (n={sel.sum()}, peak@{angs[nm]}deg)')
        axp.axvline(angs[nm], color=line_cols[nm], ls=':', lw=1)
    axp.axhline(0, color='k', lw=0.5, ls='--')
    axp.set_xlabel('lag / preferred angle (deg)')
    axp.set_ylabel('subject-weighted pooled CV r')
    axp.set_title('Anterior/posterior split (subject-weighted)\n'
                  'posterior 60deg (fMRI 66), anterior 90deg (fMRI 87)')
    axp.set_xticks(L); axp.tick_params(axis='x', labelsize=7)
    axp.legend(fontsize=8, frameon=False)
    fig.savefig(OUT / 'antpost_lag_profiles.png', dpi=200)
    fig.savefig(OUT / 'antpost_lag_profiles.pdf')
    plt.close(fig)

    # Cell colours by group subject-weighted angle (outside grey).
    colours = np.tile((0.6, 0.6, 0.6, 1.0), (len(c), 1))
    colours[post] = gb.angles_to_colours([angs['posterior (y<=med)']])[0]
    colours[ant] = gb.angles_to_colours([angs['anterior (y>med)']])[0]
    label = (f"posterior(y<=med)={angs['posterior (y<=med)']}deg, "
             f"anterior(y>med)={angs['anterior (y>med)']}deg, outside=grey")

    # Split-line geometry: anterior-posterior axis (black) through the cluster,
    # coronal boundary at y=median (red).
    cx, cz = np.abs(coords[inm, 0]).mean(), z[inm].mean()
    yline = np.column_stack([np.full(40, cx),
                             np.linspace(y[inm].min()-2, y[inm].max()+2, 40),
                             np.full(40, cz)])
    bline = np.column_stack([np.full(30, cx), np.full(30, med),
                             np.linspace(z[inm].min()-3, z[inm].max()+3, 30)])

    subjects_dir = os.path.dirname(fetch_fsaverage())
    for draw_lines in (False, True):
        tag = 'cells_with_splitlines' if draw_lines else 'cells_only'
        for hemi in ('lh', 'rh'):
            sign = -1 if hemi == 'lh' else 1
            try:
                brain = Brain('fsaverage', hemi, 'pial', background='white',
                              size=(1000, 900), subjects_dir=subjects_dir,
                              alpha=0.3)
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
                                   colormap=gb.CIRC_CMAP, alpha=0.55,
                                   colorbar=False)
            if draw_lines:
                def to_h(p):
                    q = p.copy(); q[:, 0] = sign * np.abs(q[:, 0]); return q
                brain.add_foci(to_h(yline), coords_as_verts=False, hemi=hemi,
                               color='black', scale_factor=0.22)
                brain.add_foci(to_h(bline), coords_as_verts=False, hemi=hemi,
                               color='#d11', scale_factor=0.22)
            keep = coords[:, 0] <= 0 if hemi == 'lh' else coords[:, 0] >= 0
            rng = np.random.default_rng(hash((hemi, 'ap')) & 0xFFFF)
            cj = coords[keep] + rng.uniform(-2.5, 2.5, coords[keep].shape)
            for cc, col in zip(cj, colours[keep]):
                brain.add_foci(cc[None], coords_as_verts=False, hemi=hemi,
                               color=tuple(col[:3]), scale_factor=0.42)
            try:
                brain.show_view('medial')
            except Exception:
                pass
            png = str(OUT / f'antpost_{tag}_{hemi}_medial.png')
            brain.save_image(png)
            _cbar(png, hemi, label, draw_lines)
            try:
                brain.close()
            except Exception:
                pass
            print(f'  wrote {png} (+pdf)')


def _cbar(png, hemi, label, lines):
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
    extra = ('  black=ant-post axis, red=y-median boundary' if lines else '')
    cb.set_label(f'preferred angle (deg) — cells: subject-weighted ant/post '
                 f'group angle.{extra}  {label} [{hemi}]', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
