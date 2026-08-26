#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare many candidate cell splits for the gradient <-> cell question.

For each split scheme and each group it reports, side by side:
  * n_cells, n_subjects
  * pooled preferred lag (argmax of the mean 12-lag profile), CELL-weighted
  * the same argmax but SUBJECT-weighted (mean within subject, then across)
  * pooled peak r (reliability of that pooled profile) -> tiny = noise
  * mean fMRI-predicted angle at those cells (circular mean, quarters)
  * rough colour name for the cell lag and for the fMRI angle
  * |cell argmax - fMRI angle| circular error (deg)

No plotting. Reads per_cell_master.csv, prints a table, writes split_comparison.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MASTER_DIR = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
    '/derivatives/group/cell_gradient_master/2026-08-13_09-45-28'
)
LAGS_DEG = np.arange(0, 360, 30)

# Wheel anchors every 45 deg (signed), matching the display colour wheel.
_COLOUR_ANCHORS = {0: 'yellow', 45: 'orange', 90: 'red', 135: 'magenta',
                   180: 'blue', -135: 'purple', -90: 'green', -45: 'yel-green'}


def colour_name(a):
    if not np.isfinite(a):
        return '-'
    s = ((a + 180) % 360) - 180
    centres = np.array(list(_COLOUR_ANCHORS))
    diff = np.abs((centres - s + 180) % 360 - 180)
    return _COLOUR_ANCHORS[int(centres[np.argmin(diff)])]


def circ_mean(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    th = np.radians(a)
    return np.degrees(np.arctan2(np.sin(th).mean(), np.cos(th).mean())) % 360


def circ_err(a, b):
    return abs((a - b + 180) % 360 - 180)


def pooled_argmax_r(R, sel):
    p = np.nanmean(R[sel], 0)
    return int(LAGS_DEG[np.nanargmax(p)]), float(np.nanmax(p))


def subj_argmax(R, sel, subj):
    su = np.unique(subj[sel])
    per = [np.nanmean(R[sel & (subj == s)], 0) for s in su]
    p = np.nanmean(np.vstack(per), 0)
    return int(LAGS_DEG[np.nanargmax(p)])


def rows_for_split(name, labels, R, subj, fmri):
    out = []
    for g in pd.unique(labels[labels != None]):  # noqa: E711
        sel = labels == g
        if sel.sum() < 3:
            continue
        c_arg, peak = pooled_argmax_r(R, sel)
        s_arg = subj_argmax(R, sel, subj)
        f_ang = circ_mean(fmri[sel])
        out.append(dict(
            split=name, group=str(g), n_cells=int(sel.sum()),
            n_subj=int(np.unique(subj[sel]).size),
            cell_lag=c_arg, subj_lag=s_arg, peak_r=round(peak, 3),
            fmri_angle=round(f_ang, 0) if np.isfinite(f_ang) else np.nan,
            cell_colour=colour_name(c_arg), fmri_colour=colour_name(f_ang),
            err_cell_vs_fmri=round(circ_err(c_arg, f_ang), 0)
            if np.isfinite(f_ang) else np.nan))
    return out


def qcut_labels(x, edges_q, names, mask=None):
    """Assign names by quantile edges of x (over mask if given). None outside."""
    x = np.asarray(x, float)
    base = x if mask is None else x[mask]
    e = np.quantile(base, edges_q)
    idx = np.digitize(x, e)
    lab = np.array([names[i] for i in idx], dtype=object)
    if mask is not None:
        lab[~mask] = None
    return lab


def main():
    cells = pd.read_csv(MASTER_DIR / 'per_cell_master.csv')
    R = cells[[f'r_lag{a:03d}_noctrl' for a in LAGS_DEG]].to_numpy(float)
    subj = cells['subject_id'].to_numpy()
    z = cells['MNI_z'].to_numpy(float)
    pc1 = cells['grad_axis_coord'].to_numpy(float)     # PC1 gradient axis
    inm = cells['in_gradient_mask'].to_numpy(bool)
    fmri = cells['fmri_angle_quarters_deg'].to_numpy(float)
    n = len(cells)

    splits = []
    # 1. mask membership
    splits += rows_for_split('mask_in_out',
        np.where(inm, 'in_mask', 'out_mask').astype(object), R, subj, fmri)
    # 2-3. z halves / terciles (all cells)
    splits += rows_for_split('z_half',
        qcut_labels(z, [0.5], ['ventral', 'dorsal']), R, subj, fmri)
    splits += rows_for_split('z_tercile',
        qcut_labels(z, [1/3, 2/3], ['ventral', 'mid', 'dorsal']), R, subj, fmri)
    # 4-6. PC1 gradient-axis halves / terciles / quartiles (all cells)
    splits += rows_for_split('pc1_half',
        qcut_labels(pc1, [0.5], ['begin', 'end']), R, subj, fmri)
    splits += rows_for_split('pc1_tercile',
        qcut_labels(pc1, [1/3, 2/3], ['begin', 'middle', 'end']),
        R, subj, fmri)
    splits += rows_for_split('pc1_quartile',
        qcut_labels(pc1, [.25, .5, .75], ['Q1', 'Q2', 'Q3', 'Q4']),
        R, subj, fmri)
    # 7-8. WITHIN mask, along PC1 (your "beginning vs middle of gradient")
    splits += rows_for_split('inmask_pc1_half',
        qcut_labels(pc1, [0.5], ['begin', 'end'], mask=inm), R, subj, fmri)
    splits += rows_for_split('inmask_pc1_tercile',
        qcut_labels(pc1, [1/3, 2/3], ['begin', 'middle', 'end'], mask=inm),
        R, subj, fmri)
    # 9. fMRI-angle terciles (direct cross-modal binning)
    okf = np.isfinite(fmri)
    lab = np.full(n, None, dtype=object)
    lab[okf] = qcut_labels(fmri[okf], [1/3, 2/3], ['loA', 'midA', 'hiA'])
    splits += rows_for_split('fmri_angle_tercile', lab, R, subj, fmri)

    tbl = pd.DataFrame(splits)
    tbl.to_csv(MASTER_DIR / 'split_comparison.csv', index=False)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 100)
    print(tbl.to_string(index=False))
    print(f'\nWrote {MASTER_DIR / "split_comparison.csv"}')
    print('\nReading guide: peak_r < ~0.06 = pooled profile is basically flat '
          '(argmax is noise). A "good" split = groups with peak_r high AND '
          'cell_lag == subj_lag AND small err_cell_vs_fmri.')


if __name__ == '__main__':
    main()
