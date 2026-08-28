#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project the fMRI future-angle map onto MNI z and read it off at the cell groups.

Sampling the angle map at each cell's own coordinate (what
`cell_gradient_master_table.py` does, with `SPHERE_RADIUS_MM = 0`) is a
single-voxel lookup: it inherits x/y variation and voxel noise, and it does not
measure the quantity the gradient claim is about. The gradient is a progression
along the dorsoventral axis, so the honest read-out is the map's own z-profile —
the vector-mean angle across all gradient-mask voxels in a z-slab — evaluated at
where each cell group sits.

Angles are combined as unit vectors (circular mean), never arithmetically.

Outputs, next to the gradient run:
    final_splits/fmri_angle_z_profile.csv    angle per 1 mm z step
    final_splits/fmri_angle_by_group.csv     angle at each group's z position

Usage:
    conda activate env_multiple_clocks
    python scripts/gradient_fmri_z_projection.py

@author: Svenja Kuchenhoff
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HARMONIC_ROOT = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                     '/derivatives/group/Main_Results_fMRI/harmonic_angle_maps'
                     '/unit_vector_derived')
GRADIENT_MASK = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                     '/masks/gradient_thr_1.5.nii.gz')
# >>> RERUN-CHECK: hardcoded upstream run dir -- update after re-running
# cell_gradient_master_table.py
RUN = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
           '/derivatives/group/cell_gradient_master/2026-08-28_15-19-35')

DATASET = 'quarters'
SLAB_MM = 3.0          # half-width of the z window, so each step averages +-3 mm
MIN_VOX = 20
Z_RANGE = (-6.0, 12.0)


def circular_mean_deg(cos_vals, sin_vals):
    return np.rad2deg(np.arctan2(np.mean(sin_vals), np.mean(cos_vals))) % 360


def main():
    cos_i = nib.load(str(HARMONIC_ROOT / DATASET / 'cos_group.nii.gz'))
    sin_i = nib.load(str(HARMONIC_ROOT / DATASET / 'sin_group.nii.gz'))
    C, S = cos_i.get_fdata(), sin_i.get_fdata()
    msk = nib.load(str(GRADIENT_MASK)).get_fdata() > 0

    idx = np.argwhere(msk)
    mm = nib.affines.apply_affine(cos_i.affine, idx)
    z = mm[:, 2]
    c, s = C[tuple(idx.T)], S[tuple(idx.T)]

    rows = []
    for zz in np.arange(Z_RANGE[0], Z_RANGE[1] + 1e-9, 1.0):
        m = (z >= zz - SLAB_MM) & (z < zz + SLAB_MM)
        if m.sum() < MIN_VOX:
            continue
        rows.append(dict(z_mm=zz, angle_deg=circular_mean_deg(c[m], s[m]),
                         n_voxels=int(m.sum())))
    prof = pd.DataFrame(rows)

    # interpolate on the unwrapped profile so the read-out does not jump at 0/360
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(prof.angle_deg.to_numpy())))
    def angle_at(zz):
        return float(np.interp(zz, prof.z_mm, unwrapped))

    f = pd.read_csv(RUN / 'final_splits' / 'final_splits_per_cell.csv')
    im = f[f.in_gradient_mask == True]

    out = []
    for grp in ('ventral', 'dorsal'):
        g = im[im.pc1_ventral_dorsal_group == grp]
        zs = g.MNI_z.to_numpy(float)
        a_lo, a_hi = angle_at(zs.min()), angle_at(zs.max())
        out.append(dict(
            group=grp, n_cells=len(g),
            n_sites=len(g[['MNI_x', 'MNI_y', 'MNI_z']].round(2).drop_duplicates()),
            z_mean=zs.mean(), z_min=zs.min(), z_max=zs.max(),
            fmri_angle_at_z_mean_deg=angle_at(zs.mean()),
            fmri_angle_lo_deg=min(a_lo, a_hi),
            fmri_angle_hi_deg=max(a_lo, a_hi)))
    grp_df = pd.DataFrame(out)

    d = RUN / 'final_splits'
    prof.to_csv(d / 'fmri_angle_z_profile.csv', index=False)
    grp_df.to_csv(d / 'fmri_angle_by_group.csv', index=False)

    print("fMRI preferred angle projected onto MNI z (gradient mask):")
    print("  " + "  ".join(f"z{r.z_mm:+.0f}:{r.angle_deg:3.0f}"
                           for _, r in prof.iterrows()))
    print()
    print(grp_df.round(1).to_string(index=False))
    print(f"\nSaved -> {d}/fmri_angle_z_profile.csv")
    print(f"Saved -> {d}/fmri_angle_by_group.csv")


if __name__ == '__main__':
    main()
