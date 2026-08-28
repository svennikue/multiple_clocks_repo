#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-sided tests against zero for the ventral/dorsal mPFC gradient clusters.

Reports the same contrast at three units of analysis, because cells on one
microwire bundle share a coordinate exactly and are not independent:

    cell     every unit (matches the shading in Fig 3d)
    site     cells averaged within a recording site first
    subject  cells averaged within a subject first

Two lag families:
    peak       each group's own argmax lag (ventral 30 deg, dorsal 60 deg).
               Selecting the lag from the same data is circular, so the
               FDR across all 12 lags is reported alongside.
    window     the pre-specified immediate-future window (30+60 deg) the
               main text already commits to for mPFC.

FDR is Benjamini-Hochberg, applied within each (unit, family) across the two
groups, and — for the peak family — additionally across the 12 lags, which is
what makes the peak test non-circular.

Output: <run>/final_splits/gradient_split_vs_zero.csv

Usage:
    conda activate env_multiple_clocks
    python scripts/gradient_split_vs_zero.py

@author: Svenja Kuchenhoff
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# >>> RERUN-CHECK: hardcoded upstream run dir -- update after re-running
# cell_gradient_master_table.py
RUN = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
           '/derivatives/group/cell_gradient_master/2026-08-28_15-19-35')

LAGS_DEG = list(range(0, 360, 30))
PEAK_LAG = {'ventral': 30, 'dorsal': 60}       # each group's argmax
WINDOW_LAGS = [30, 60]                         # pre-specified immediate future


def bh_fdr(p):
    """Benjamini-Hochberg, returns q in the input order."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def fisher_z(r):
    return np.arctanh(np.clip(np.asarray(r, float), -0.999, 0.999))


def one_sided_t(values):
    """One-sided (greater-than-zero) t-test on Fisher-z values."""
    v = fisher_z(pd.Series(values).dropna())
    if len(v) < 2:
        return np.nan, np.nan, len(v)
    t, p_two = stats.ttest_1samp(v, 0.0)
    p = p_two / 2 if t > 0 else 1.0 - p_two / 2
    return float(t), float(p), int(len(v))


def load():
    m = pd.read_csv(RUN / 'per_cell_master.csv')
    f = pd.read_csv(RUN / 'final_splits' / 'final_splits_per_cell.csv')
    cols = [f'r_lag{a:03d}_noctrl' for a in LAGS_DEG]
    j = (f[f.in_gradient_mask == True]
         [['neuron', 'subject_id', 'MNI_x', 'MNI_y', 'MNI_z',
           'pc1_ventral_dorsal_group']]
         .merge(m[['neuron'] + cols], on='neuron', how='left'))
    j['site'] = (j.MNI_x.round(2).astype(str) + '_' +
                 j.MNI_y.round(2).astype(str) + '_' +
                 j.MNI_z.round(2).astype(str))
    return j


def aggregate(g, col, unit):
    if unit == 'cell':
        return g[col]
    return g.groupby(unit)[col].mean()


def main():
    j = load()
    rows = []

    # ---- family 1: each group's own peak lag, plus the full 12-lag sweep ----
    for unit in ('cell', 'site', 'subject_id'):
        for grp in ('ventral', 'dorsal'):
            g = j[j.pc1_ventral_dorsal_group == grp]
            for lag in LAGS_DEG:
                col = f'r_lag{lag:03d}_noctrl'
                vals = aggregate(g, col, unit)
                t, p, n = one_sided_t(vals)
                rows.append(dict(
                    family='all_lags', unit=unit.replace('_id', ''), group=grp,
                    lag_deg=lag, is_group_peak=(lag == PEAK_LAG[grp]),
                    n=n, mean_r=float(pd.Series(vals).dropna().mean()),
                    t=t, p_one_sided=p))

    # ---- family 2: the pre-specified 30+60 window ----
    wcols = [f'r_lag{a:03d}_noctrl' for a in WINDOW_LAGS]
    for unit in ('cell', 'site', 'subject_id'):
        for grp in ('ventral', 'dorsal'):
            g = j[j.pc1_ventral_dorsal_group == grp].copy()
            g['w'] = g[wcols].mean(axis=1)
            vals = aggregate(g, 'w', unit)
            t, p, n = one_sided_t(vals)
            rows.append(dict(
                family='window_30_60', unit=unit.replace('_id', ''), group=grp,
                lag_deg=np.nan, is_group_peak=True, n=n,
                mean_r=float(pd.Series(vals).dropna().mean()),
                t=t, p_one_sided=p))

    out = pd.DataFrame(rows)

    # FDR across the two groups, within (family, unit, lag)
    out['p_fdr_across_groups'] = np.nan
    for (fam, unit, lag), idx in out.groupby(
            ['family', 'unit', out.lag_deg.fillna(-1)]).groups.items():
        out.loc[idx, 'p_fdr_across_groups'] = bh_fdr(out.loc[idx, 'p_one_sided'])

    # FDR across the 12 lags, within (unit, group) -- removes the circularity
    # of testing at an argmax chosen from the same data
    out['p_fdr_across_12_lags'] = np.nan
    sweep = out.family == 'all_lags'
    for (unit, grp), idx in out[sweep].groupby(['unit', 'group']).groups.items():
        out.loc[idx, 'p_fdr_across_12_lags'] = bh_fdr(out.loc[idx, 'p_one_sided'])

    dest = RUN / 'final_splits' / 'gradient_split_vs_zero.csv'
    out.to_csv(dest, index=False)

    print(f"n cells {len(j)} | sites {j.site.nunique()} | "
          f"subjects {j.subject_id.nunique()}\n")
    show = out[(out.is_group_peak) &
               ((out.family == 'window_30_60') | sweep)].copy()
    print(show[['family', 'unit', 'group', 'lag_deg', 'n', 'mean_r', 't',
                'p_one_sided', 'p_fdr_across_groups',
                'p_fdr_across_12_lags']].round(4).to_string(index=False))
    print(f"\nSaved -> {dest}")


if __name__ == '__main__':
    main()
