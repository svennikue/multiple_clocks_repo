#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bin cells along the curved (bending) gradient axis and compare their pooled
arg-max lag to the fMRI angle along the curve.

Uses the principal-curve axis from gradient_bending_axis.py. Each cell is
projected to its nearest point on the curve (arc-length position + perpendicular
distance). Cells are binned by arc length; per bin we read the pooled cell
arg-max lag (cell- and subject-weighted) and compare to the fMRI angle sampled
along the curve (unwrapped) and at the cells.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gradient_brain_cells_by_lag as gb
from gradient_bending_axis import build_axis

L = np.arange(0, 360, 30)
OUT = gb.MASTER_DIR / 'gradient_bending_axis'


def cmean(a):
    a = np.radians(np.asarray(a, float)); a = a[np.isfinite(a)]
    return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360 \
        if a.size else np.nan


def pooled(R, sel, subj, w):
    if w == 'cell':
        p = np.nanmean(R[sel], 0)
    else:
        p = np.nanmean(np.vstack([np.nanmean(R[sel & (subj == s)], 0)
                                  for s in np.unique(subj[sel])]), 0)
    return int(L[np.nanargmax(p)]), float(np.nanmax(p))


def main():
    curve, ang_along, unwrapped = build_axis()
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    cyz = curve[:, 1:3]

    c = pd.read_csv(gb.MASTER_DIR / 'per_cell_master.csv')
    R = c[[f'r_lag{a:03d}_noctrl' for a in L]].to_numpy(float)
    subj = c['subject_id'].to_numpy()
    fmri = c['fmri_angle_quarters_deg'].to_numpy(float)
    inm = c['in_gradient_mask'].to_numpy(bool)
    yz = c[['MNI_y', 'MNI_z']].to_numpy(float)

    # Project each cell to nearest curve point: arc-length s, perp dist, and
    # the unwrapped fMRI angle of the curve there.
    s = np.empty(len(c)); perp = np.empty(len(c)); uang = np.empty(len(c))
    for i, p in enumerate(yz):
        d = np.linalg.norm(cyz - p, axis=1); j = int(np.argmin(d))
        s[i], perp[i], uang[i] = seg[j], d[j], unwrapped[j]

    # Use in-mask cells (they lie on the gradient); report along arc length.
    print('fMRI angle along curve (unwrapped) spans '
          f'{unwrapped.min():.0f}..{unwrapped.max():.0f} deg over '
          f'{seg[-1]:.0f} mm')
    print(f'in-mask cells: arc-length {s[inm].min():.0f}..{s[inm].max():.0f} mm, '
          f'perp dist median {np.median(perp[inm]):.1f} mm')

    for nb in (3, 4, 5):
        q = np.quantile(s[inm], np.linspace(0, 1, nb + 1)[1:-1])
        lab = np.digitize(s, q)
        print(f'\n=== {nb} arc-length bins (in-mask), ventral->dorsal ===')
        rows = []
        for b in range(nb):
            sel = inm & (lab == b)
            if sel.sum() < 3:
                continue
            ac, rc = pooled(R, sel, subj, 'cell')
            asu, _ = pooled(R, sel, subj, 'subject')
            fm = cmean(fmri[sel]); fu = np.nanmean(uang[sel])
            rows.append((np.nanmean(s[sel]), ac, asu, rc, fm, fu, sel.sum()))
            print(f'  bin{b} n={sel.sum():2d} arc={np.nanmean(s[sel]):4.0f}mm  '
                  f'cell(cell-w)={ac:3d}(r={rc:.3f}) cell(subj-w)={asu:3d}  '
                  f'fMRI(cells)={fm:3.0f} fMRI(curve,unwrap)={fu:3.0f}')
        if nb == 4:
            _plot(rows)


def _plot(rows):
    arc = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    ax.plot(arc, [r[5] for r in rows], '--s', color='#23677E', lw=2,
            label='fMRI angle along curve (unwrapped)')
    ax.plot(arc, [r[4] for r in rows], ':D', color='#7BAFD4', lw=1.5,
            label='fMRI angle at cells (wrapped)')
    ax.plot(arc, [r[1] for r in rows], '-o', color='#0e3d3a', lw=2, ms=7,
            label='cell arg-max (cell-weighted)')
    ax.plot(arc, [r[2] for r in rows], '-^', color='#5C1027', lw=2, ms=7,
            label='cell arg-max (subject-weighted)')
    ax.set_xlabel('arc length along bending gradient axis (mm; ventral -> dorsal)')
    ax.set_ylabel('preferred angle / lag (deg)')
    ax.set_title('Cells binned along the curved gradient axis vs fMRI')
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(OUT / 'cells_along_curve_match.png', dpi=200)
    fig.savefig(OUT / 'cells_along_curve_match.pdf')
    plt.close(fig)
    print(f'\nWrote {OUT / "cells_along_curve_match.png"} (+pdf)')


if __name__ == '__main__':
    main()
