#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rank every averaging scheme by how well the pooled cell lag tracks the fMRI
gradient DIRECTION.

Axes: pc1 (diagonal), y (ant-post), z (dorsoventral). Group counts: 2,3,4,5.
Weightings: unw / consist (spatial-consistency weighted) / subj.

For each scheme, groups are ordered ventral->dorsal (increasing fMRI). We score:
  * dir_score  : fraction of consecutive steps where the pooled cell lag moves
                 in the SAME circular direction as the fMRI angle (1 = perfectly
                 tracks the gradient direction; the 240-deg backward jumps count
                 against it).
  * mean_err   : mean |circular(cell_lag - fMRI)| across groups (absolute match).
Best = high dir_score, then low mean_err. Prints the ranked table + sequences.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MD = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
          '/derivatives/group/cell_gradient_master/2026-08-13_09-45-28')
L = np.arange(0, 360, 30)
MIN_PER_GROUP = 5


def cmean(a):
    a = np.radians(np.asarray(a, float)); a = a[np.isfinite(a)]
    return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360 \
        if a.size else np.nan


def sdiff(a, b):
    return (a - b + 180) % 360 - 180


def pooled_lag(R, sel, subj, consist, weighting):
    if sel.sum() < MIN_PER_GROUP:
        return np.nan
    if weighting == 'unw':
        prof = np.nanmean(R[sel], 0)
    elif weighting == 'consist':
        w = np.clip(consist[sel], 0, None)
        prof = np.nansum(w[:, None] * R[sel], 0) / (w.sum() + 1e-12)
    else:
        prof = np.nanmean(np.vstack([np.nanmean(R[sel & (subj == s)], 0)
                                     for s in np.unique(subj[sel])]), 0)
    return float(L[np.nanargmax(prof)]) if np.isfinite(prof).any() else np.nan


def main():
    c = pd.read_csv(MD / 'per_cell_master.csv')
    R = c[[f'r_lag{a:03d}_noctrl' for a in L]].to_numpy(float)
    subj = c['subject_id'].to_numpy()
    inm = c['in_gradient_mask'].to_numpy(bool)
    fmri = c['fmri_angle_quarters_deg'].to_numpy(float)
    consist = c['argmax_r'].to_numpy(float)
    axes = {'pc1': c['grad_axis_coord'].to_numpy(float),
            'y': c['MNI_y'].to_numpy(float),
            'z': c['MNI_z'].to_numpy(float)}

    rows = []
    for ax_name, x in axes.items():
        for n in (2, 3, 4, 5):
            edges = np.quantile(x[inm], np.linspace(0, 1, n + 1)[1:-1])
            lab = np.full(len(x), -1)
            lab[inm] = np.digitize(x[inm], edges)
            groups = sorted(set(lab[inm]))
            # order groups ventral->dorsal by mean axis
            order = sorted(groups, key=lambda g: np.nanmean(x[inm & (lab == g)]))
            for w in ('unw', 'consist', 'subj'):
                cl, fm, ns = [], [], []
                for g in order:
                    sel = inm & (lab == g)
                    cl.append(pooled_lag(R, sel, subj, consist, w))
                    fm.append(cmean(fmri[sel]))
                    ns.append(int(sel.sum()))
                cl, fm = np.array(cl), np.array(fm)
                if np.isnan(cl).any():
                    continue
                cs = sdiff(cl[1:], cl[:-1])          # cell step directions
                fs = sdiff(fm[1:], fm[:-1])          # fMRI step directions
                dir_score = float(np.mean(np.sign(cs) == np.sign(fs)))
                mean_err = float(np.mean(np.abs(sdiff(cl, fm))))
                rows.append(dict(
                    axis=ax_name, n_groups=n, weighting=w,
                    dir_score=round(dir_score, 2), mean_err=round(mean_err, 0),
                    cell_lags='->'.join(f'{v:.0f}' for v in cl),
                    fmri_angles='->'.join(f'{v:.0f}' for v in fm),
                    min_n=min(ns)))

    t = pd.DataFrame(rows).sort_values(
        ['dir_score', 'mean_err'], ascending=[False, True]).reset_index(drop=True)
    t.to_csv(MD / 'best_split_ranking.csv', index=False)
    pd.set_option('display.width', 200); pd.set_option('display.max_rows', 120)
    print('=== ALL SCHEMES RANKED (best direction-match first) ===')
    print(t.to_string(index=False))
    print('\nBEST (dir_score=1 means pooled cell lag increases ventral->dorsal '
          'exactly like the fMRI):')
    best = t[t.dir_score == t.dir_score.max()]
    print(best.to_string(index=False))
    print(f'\nWrote {MD/"best_split_ranking.csv"}')


if __name__ == '__main__':
    main()
