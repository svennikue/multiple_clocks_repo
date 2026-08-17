#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consolidated averaging-scheme table.

For every candidate way of averaging the mPFC cells, report the group each cell
falls in, the group's pooled preferred lag under three weightings, and the
group's mean fMRI angle -- so the cell lag and the fMRI can be compared per
scheme in one place.

Axes (all split WITHIN the gradient mask; outside-mask cells labelled 'outside'):
  * pc1  : gradient-mask PC1 (the diagonal: dorsoventral with an ant-post tilt)
  * y    : MNI_y  (posterior -> anterior)
  * z    : MNI_z  (ventral -> dorsal, pure)
Splits: 2 groups (median) and 3 groups (terciles).
Weightings for the pooled lag:
  * unw     : plain mean of the 12-lag profiles -> argmax
  * consist : mean of profiles weighted by each cell's spatial consistency
              (argmax r) -> argmax
  * subj    : mean within subject, then across subjects -> argmax

Writes:
  scheme_match_overview.csv   -- one row per (axis, split, group): pooled lag
                                 (3 weightings), group fMRI, |lag-fMRI| errors, n
  per_cell_averaging_schemes.csv -- one row per cell: its group + the group's
                                 pooled lag (3 weightings) + group fMRI, per scheme
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MD = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
          '/derivatives/group/cell_gradient_master/2026-08-13_09-45-28')
L = np.arange(0, 360, 30)


def cmean(a):
    a = np.radians(np.asarray(a, float)); a = a[np.isfinite(a)]
    return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360 \
        if a.size else np.nan


def circ_err(a, b):
    return abs((a - b + 180) % 360 - 180) if np.isfinite(a) and np.isfinite(b) \
        else np.nan


def pooled_lag(R, sel, subj, w_consist, weighting):
    if sel.sum() == 0:
        return np.nan
    if weighting == 'unw':
        prof = np.nanmean(R[sel], 0)
    elif weighting == 'consist':
        w = np.clip(w_consist[sel], 0, None)
        prof = np.nansum(w[:, None] * R[sel], 0) / (w.sum() + 1e-12)
    else:  # subject
        prof = np.nanmean(np.vstack([np.nanmean(R[sel & (subj == s)], 0)
                                     for s in np.unique(subj[sel])]), 0)
    return int(L[np.nanargmax(prof)]) if np.isfinite(prof).any() else np.nan


def group_labels(x, inm, n):
    """Median (n=2) or tercile (n=3) split within the mask; 'outside' else."""
    edges = np.quantile(x[inm], np.linspace(0, 1, n + 1)[1:-1])
    names = (['low', 'high'] if n == 2 else ['low', 'mid', 'high'])
    lab = np.array(['outside'] * len(x), dtype=object)
    lab[inm] = [names[i] for i in np.digitize(x[inm], edges)]
    return lab


def main():
    c = pd.read_csv(MD / 'per_cell_master.csv')
    R = c[[f'r_lag{a:03d}_noctrl' for a in L]].to_numpy(float)
    subj = c['subject_id'].to_numpy()
    inm = c['in_gradient_mask'].to_numpy(bool)
    fmri = c['fmri_angle_quarters_deg'].to_numpy(float)
    consist = c['argmax_r'].to_numpy(float)          # spatial consistency
    axes = {'pc1': c['grad_axis_coord'].to_numpy(float),
            'y': c['MNI_y'].to_numpy(float),
            'z': c['MNI_z'].to_numpy(float)}

    per_cell = c[['neuron', 'subject_id', 'MNI_x', 'MNI_y', 'MNI_z',
                  'grad_axis_coord', 'in_gradient_mask']].copy()
    per_cell['cell_argmax_lag'] = c['argmax_lag_deg']
    per_cell['cell_consistency_r'] = consist
    per_cell['cell_fmri_angle'] = fmri

    overview = []
    for ax_name, x in axes.items():
        for n in (2, 3):
            lab = group_labels(x, inm, n)
            tag = f'{ax_name}_{n}g'
            per_cell[f'{tag}_group'] = lab
            # per-group pooled values
            glag = {w: {} for w in ('unw', 'consist', 'subj')}
            gfmri, gaxis, gn = {}, {}, {}
            for grp in ['low', 'mid', 'high', 'outside']:
                sel = lab == grp
                if sel.sum() == 0:
                    continue
                for w in ('unw', 'consist', 'subj'):
                    glag[w][grp] = pooled_lag(R, sel, subj, consist, w)
                gfmri[grp] = round(cmean(fmri[sel]), 0)
                gaxis[grp] = round(np.nanmean(x[sel]), 1)
                gn[grp] = int(sel.sum())
                overview.append(dict(
                    axis=ax_name, split=f'{n}g', group=grp, n=gn[grp],
                    mean_axis=gaxis[grp],
                    lag_unw=glag['unw'][grp], lag_consist=glag['consist'][grp],
                    lag_subj=glag['subj'][grp], fmri_angle=gfmri[grp],
                    err_unw=round(circ_err(glag['unw'][grp], gfmri[grp]), 0),
                    err_subj=round(circ_err(glag['subj'][grp], gfmri[grp]), 0)))
            # broadcast group pooled values onto each cell
            for w in ('unw', 'consist', 'subj'):
                per_cell[f'{tag}_lag_{w}'] = [glag[w].get(g, np.nan) for g in lab]
            per_cell[f'{tag}_fmri'] = [gfmri.get(g, np.nan) for g in lab]

    ov = pd.DataFrame(overview)
    ov.to_csv(MD / 'scheme_match_overview.csv', index=False)
    per_cell.to_csv(MD / 'per_cell_averaging_schemes.csv', index=False)

    pd.set_option('display.width', 220); pd.set_option('display.max_rows', 120)
    print('=== SCHEME MATCH OVERVIEW (pooled cell lag vs group fMRI) ===')
    print(ov.to_string(index=False))
    print(f'\nWrote {MD/"scheme_match_overview.csv"}')
    print(f'Wrote {MD/"per_cell_averaging_schemes.csv"}  ({len(per_cell)} cells, '
          f'{per_cell.shape[1]} cols)')


if __name__ == '__main__':
    main()
