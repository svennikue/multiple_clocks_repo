#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does permutation-gating sharpen the pooled preferred lag? Check all splits.

For every candidate split (same as cell_gradient_split_table.py) and group,
compute the pooled preferred lag under four pooling rules:

  * ``none``       : mean 12-lag profile over all cells in the group.
  * ``perlag@a``   : at each lag L, average ONLY over cells whose permutation
                     p at lag L is < a (>= MIN_CONTRIB contributing cells,
                     else that lag is dropped). This is the "cells survive the
                     perm at lag L to contribute to tuning at lag L" rule.
  * ``cellfilt@a`` : keep only cells with >= 1 significant lag (p < a), then
                     pool their full profiles.

Reports the argmax lag under each rule, cell- and subject-weighted where noted,
plus the local fMRI angle, so we can see whether gating helps or just thins the
data. No plotting. Writes split_permgated_comparison.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MASTER_DIR = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
    '/derivatives/group/cell_gradient_master/2026-08-13_09-45-28'
)
SOURCE_CSV = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
    '/derivatives/group/per_lag_encoding'
    '/2026-08-04_07-25-15_reload_from_2026-06-30_18-21-57_relabelled'
    '/per_cell_ALL_ROIs.csv'
)
LAGS_DEG = np.arange(0, 360, 30)
ALPHAS = [0.05, 0.01]
MIN_CONTRIB = 3            # min sig cells required for a per-lag pooled value


def circ_mean(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    th = np.radians(a)
    return np.degrees(np.arctan2(np.sin(th).mean(), np.cos(th).mean())) % 360


def argmax_none(R, sel):
    p = np.nanmean(R[sel], 0)
    if not np.isfinite(p).any():
        return np.nan, np.nan
    return int(LAGS_DEG[np.nanargmax(p)]), float(np.nanmax(p))


def argmax_perlag(R, P, sel, alpha):
    prof = np.full(12, np.nan)
    contrib = np.zeros(12, int)
    for j in range(12):
        c = sel & (P[:, j] < alpha)
        contrib[j] = c.sum()
        if c.sum() >= MIN_CONTRIB:
            prof[j] = np.nanmean(R[c, j])
    if not np.isfinite(prof).any():
        return np.nan, int(contrib.max())
    return int(LAGS_DEG[np.nanargmax(prof)]), int(np.median(contrib))


def argmax_cellfilt(R, P, sel, alpha):
    keep = sel & (P < alpha).any(1)
    if keep.sum() < MIN_CONTRIB:
        return np.nan, int(keep.sum())
    p = np.nanmean(R[keep], 0)
    return int(LAGS_DEG[np.nanargmax(p)]), int(keep.sum())


def qcut_labels(x, edges_q, names, mask=None):
    x = np.asarray(x, float)
    base = x if mask is None else x[mask]
    e = np.quantile(base, edges_q)
    lab = np.array([names[i] for i in np.digitize(x, e)], dtype=object)
    if mask is not None:
        lab[~mask] = None
    return lab


def main():
    cells = pd.read_csv(MASTER_DIR / 'per_cell_master.csv')
    src = pd.read_csv(SOURCE_CSV)
    src = src[src.roi == 'mPFC'][
        ['neuron'] + [f'p_lag{a:03d}_noctrl' for a in LAGS_DEG]]
    cells = cells.merge(src, on='neuron', how='left', validate='one_to_one')

    R = cells[[f'r_lag{a:03d}_noctrl' for a in LAGS_DEG]].to_numpy(float)
    P = cells[[f'p_lag{a:03d}_noctrl' for a in LAGS_DEG]].to_numpy(float)
    z = cells['MNI_z'].to_numpy(float)
    pc1 = cells['grad_axis_coord'].to_numpy(float)
    inm = cells['in_gradient_mask'].to_numpy(bool)
    fmri = cells['fmri_angle_quarters_deg'].to_numpy(float)
    n = len(cells)

    schemes = {
        'mask_in_out': np.where(inm, 'in_mask', 'out_mask').astype(object),
        'z_half': qcut_labels(z, [0.5], ['ventral', 'dorsal']),
        'z_tercile': qcut_labels(z, [1/3, 2/3], ['ventral', 'mid', 'dorsal']),
        'pc1_half': qcut_labels(pc1, [0.5], ['begin', 'end']),
        'pc1_tercile': qcut_labels(pc1, [1/3, 2/3], ['begin', 'middle', 'end']),
        'pc1_quartile': qcut_labels(pc1, [.25, .5, .75],
                                    ['Q1', 'Q2', 'Q3', 'Q4']),
        'inmask_pc1_half': qcut_labels(pc1, [0.5], ['begin', 'end'], mask=inm),
        'inmask_pc1_tercile': qcut_labels(
            pc1, [1/3, 2/3], ['begin', 'middle', 'end'], mask=inm),
    }

    rows = []
    for name, labels in schemes.items():
        for g in pd.unique(labels[labels != None]):   # noqa: E711
            sel = labels == g
            if sel.sum() < MIN_CONTRIB:
                continue
            lag_none, peak = argmax_none(R, sel)
            row = dict(split=name, group=str(g), n=int(sel.sum()),
                       lag_none=lag_none, peak_r=round(peak, 3),
                       fmri=round(circ_mean(fmri[sel]), 0))
            for a in ALPHAS:
                lag_pl, med_c = argmax_perlag(R, P, sel, a)
                lag_cf, n_cf = argmax_cellfilt(R, P, sel, a)
                row[f'perlag@{a}'] = lag_pl
                row[f'plContrib@{a}'] = med_c
                row[f'cellfilt@{a}'] = lag_cf
                row[f'cfN@{a}'] = n_cf
            rows.append(row)

    tbl = pd.DataFrame(rows)
    tbl.to_csv(MASTER_DIR / 'split_permgated_comparison.csv', index=False)
    pd.set_option('display.width', 240); pd.set_option('display.max_rows', 100)
    show = ['split', 'group', 'n', 'lag_none', 'perlag@0.05', 'plContrib@0.05',
            'perlag@0.01', 'cellfilt@0.05', 'cfN@0.05', 'peak_r', 'fmri']
    print(tbl[show].to_string(index=False))
    print(f'\nWrote {MASTER_DIR / "split_permgated_comparison.csv"}')
    print('\nplContrib = median #cells contributing per lag under the gate; '
          'if it is ~3-5 the gated lag is itself fragile.')


if __name__ == '__main__':
    main()
