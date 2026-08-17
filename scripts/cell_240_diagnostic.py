#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose the mid-axis 240-degree-preferring cells.

Identify the arc-length bin along the bending gradient axis whose pooled profile
peaks at 240deg, and inspect those cells: full 12-lag profiles, whether 240 is a
single clear peak, permutation reliability (p at 240), and spatial-tuning
strength (peak r, relative strength) vs the ventral (60deg) reference bin and
the whole mPFC population.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gradient_brain_cells_by_lag as gb
from gradient_bending_axis import build_axis

L = np.arange(0, 360, 30)
OUT = gb.MASTER_DIR / 'gradient_bending_axis'
SRC = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
       '/derivatives/group/per_lag_encoding'
       '/2026-08-04_07-25-15_reload_from_2026-06-30_18-21-57_relabelled'
       '/per_cell_ALL_ROIs.csv')


def main():
    curve, _, _ = build_axis()
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    cyz = curve[:, 1:3]

    c = pd.read_csv(gb.MASTER_DIR / 'per_cell_master.csv')
    s = pd.read_csv(SRC); s = s[s.roi == 'mPFC']
    pcols = [f'p_lag{a:03d}_noctrl' for a in L]
    c = c.merge(s[['neuron'] + pcols], on='neuron')
    R = c[[f'r_lag{a:03d}_noctrl' for a in L]].to_numpy(float)
    P = c[pcols].to_numpy(float)
    inm = c['in_gradient_mask'].to_numpy(bool)
    yz = c[['MNI_y', 'MNI_z']].to_numpy(float)

    arc = np.array([seg[np.argmin(np.linalg.norm(cyz - p, axis=1))] for p in yz])
    q = np.quantile(arc[inm], [.25, .5, .75])
    lab = np.digitize(arc, q)
    # find the in-mask bin whose pooled profile peaks at 240
    tgt = None
    for b in range(4):
        sel = inm & (lab == b)
        if sel.sum() >= 3 and L[np.nanargmax(np.nanmean(R[sel], 0))] == 240:
            tgt = b
    if tgt is None:
        print('no 240-peaking arc bin found; using argmax==240 in-mask cells')
        grp = inm & (np.nanargmax(R, 1) == np.where(L == 240)[0][0])
    else:
        grp = inm & (lab == tgt)
    ref = inm & (arc <= np.median(arc[inm])) & ~grp   # ventral reference cells

    idx240 = np.where(L == 240)[0][0]
    prof = np.nanmean(R[grp], 0)
    print(f'=== 240-group: {grp.sum()} cells (arc bin {tgt}) ===')
    print('pooled profile:', ' '.join(f'{a}:{v:+.3f}' for a, v in zip(L, prof)))
    order = np.argsort(-prof)
    print(f'  pooled peak@{L[order[0]]}(r={prof[order[0]]:.3f}), '
          f'2nd@{L[order[1]]}(r={prof[order[1]]:.3f}), '
          f'3rd@{L[order[2]]}(r={prof[order[2]]:.3f})')
    # per-cell
    cell_arg = L[np.nanargmax(R, 1)]
    print('\n  per-cell argmax distribution in this group:',
          {int(a): int((cell_arg[grp] == a).sum()) for a in L
           if (cell_arg[grp] == a).sum()})
    r_at_240 = R[grp, idx240]; p_at_240 = P[grp, idx240]
    peak_r = np.nanmax(R[grp], 1)
    print(f'  r at 240deg: mean={np.nanmean(r_at_240):.3f} '
          f'range {np.nanmin(r_at_240):.3f}..{np.nanmax(r_at_240):.3f}')
    print(f'  cells sig at 240 (p<.05): {(p_at_240 < .05).sum()}/{grp.sum()}')
    print(f'  peak r (each cell best lag): mean={peak_r.mean():.3f} '
          f'median={np.median(peak_r):.3f}')
    # significant-at-any-lag & subject/site clustering
    print(f'  cells sig at >=1 lag: {(P[grp] < .05).any(1).sum()}/{grp.sum()}')
    print(f'  subjects: {np.unique(c["subject_id"].to_numpy()[grp]).size}, '
          f'unique coords: {len(np.unique(yz[grp], axis=0))}')

    print(f'\n=== reference ventral bin (pooled 60deg): {ref.sum()} cells ===')
    pr = np.nanmean(R[ref], 0)
    print(f'  peak@{L[np.nanargmax(pr)]}(r={np.nanmax(pr):.3f}), '
          f'peak r/cell mean={np.nanmax(R[ref],1).mean():.3f}')
    print(f'\n=== whole mPFC (n={len(c)}): peak r/cell mean='
          f'{np.nanmax(R,1).mean():.3f} median={np.median(np.nanmax(R,1)):.3f}')

    # figure: spaghetti of individual profiles + pooled
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for sel, title, col in [(grp, f'240-group (n={grp.sum()})', '#5C1027'),
                            (ref, f'ventral 60 ref (n={ref.sum()})', '#0e3d3a')]:
        a = ax[0] if sel is grp else ax[1]
        for row in R[sel]:
            a.plot(L, row, color=col, alpha=0.18, lw=0.8)
        a.plot(L, np.nanmean(R[sel], 0), color=col, lw=2.6, marker='o',
               label='pooled mean')
        a.axhline(0, color='k', lw=0.5, ls='--')
        a.axvline(240 if sel is grp else 60, color='grey', ls=':', lw=1)
        a.set_title(title); a.set_xlabel('lag (deg)'); a.set_xticks(L)
        a.tick_params(axis='x', labelsize=7); a.set_ylabel('CV r')
        a.legend(fontsize=8, frameon=False)
    fig.suptitle('Individual cell profiles (thin) + pooled (thick)')
    fig.savefig(OUT / 'cells_240_diagnostic.png', dpi=200)
    fig.savefig(OUT / 'cells_240_diagnostic.pdf')
    plt.close(fig)
    print(f'\nWrote {OUT / "cells_240_diagnostic.png"} (+pdf)')


if __name__ == '__main__':
    main()
