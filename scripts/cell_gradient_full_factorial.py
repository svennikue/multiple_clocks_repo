#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full factorial robustness table for the cell<->gradient question.

Crosses every knob we have discussed:
  * ctrl_mode : noctrl | ctrl               (which r columns)
  * weighting : cell   | subject            (average all cells vs subject-first)
  * gating    : none   | permgated(.05)     (per-lag perm-p gate on contribution)
  * cell_set  : all    | in_mask            (all mPFC vs only gradient-mask cells)
  * axis      : pc1    | z                   (PC1 of gradient mask vs MNI z)
  * split     : half(2) | tercile(3) | quartile(4) | correlation

For binned splits each group reports the pooled preferred lag (argmax of the
pooled 12-lag profile), pooled peak r, and the local fMRI angle. For
``correlation`` it reports Spearman(future_score, axis) with a circular-shift
p (same estimator drives empirical and null). Writes full_factorial.csv and
prints the correlation block + a pooled-lag pivot.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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
LAGS = np.arange(0, 360, 30)
FUTURE = np.isin(LAGS, [30, 60, 90])
GATE_ALPHA = 0.05
MIN_CONTRIB = 3
N_PERM = 2000
SEED = 42


def circ_mean(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    th = np.radians(a)
    return np.degrees(np.arctan2(np.sin(th).mean(), np.cos(th).mean())) % 360


def circ_err(a, b):
    return abs((a - b + 180) % 360 - 180) if np.isfinite(a) and np.isfinite(b) \
        else np.nan


def pooled_lag(R, M, sel, weighting, subj):
    """Pooled argmax lag + peak r. M = per-cell-per-lag contribution mask."""
    if weighting == 'cell':
        prof = np.full(12, np.nan)
        for j in range(12):
            c = sel & M[:, j]
            if c.sum() >= MIN_CONTRIB:
                prof[j] = np.nanmean(R[c, j])
    else:
        mat = []
        for s in np.unique(subj[sel]):
            sp = np.full(12, np.nan)
            for j in range(12):
                c = sel & (subj == s) & M[:, j]
                if c.any():
                    sp[j] = np.nanmean(R[c, j])
            mat.append(sp)
        prof = np.nanmean(np.vstack(mat), 0)
    if not np.isfinite(prof).any():
        return np.nan, np.nan
    return int(LAGS[np.nanargmax(prof)]), float(np.nanmax(prof))


def future_score(R):
    return np.nanmean(R[:, FUTURE], 1) - np.nanmean(R[:, ~FUTURE], 1)


def corr_test(R, sel, coord, weighting, subj, rng):
    idx = np.where(sel)[0]
    x, subj_i = coord[idx], subj[idx]

    def spear(Rmat):
        fs = future_score(Rmat)[idx]
        if weighting == 'subject':
            su = np.unique(subj_i)
            xs = [np.nanmean(x[subj_i == s]) for s in su]
            ys = [np.nanmean(fs[subj_i == s]) for s in su]
            return spearmanr(xs, ys)[0]
        return spearmanr(x, fs)[0]

    obs = spear(R)
    null = np.empty(N_PERM)
    for p in range(N_PERM):
        sh = rng.integers(0, 12, len(R))
        Rs = np.take_along_axis(
            R, (np.arange(12)[None, :] - sh[:, None]) % 12, axis=1)
        null[p] = spear(Rs)
    pval = (1 + np.sum(null >= obs)) / (N_PERM + 1)
    return obs, pval, len(idx)


def qcut(x, edges, names):
    e = np.quantile(x, edges)
    return np.array([names[i] for i in np.digitize(x, e)], dtype=object)


def main():
    cells = pd.read_csv(MASTER_DIR / 'per_cell_master.csv')
    src = pd.read_csv(SOURCE_CSV)
    src = src[src.roi == 'mPFC']
    keep_cols = ['neuron']
    for a in LAGS:
        keep_cols += [f'r_lag{a:03d}_ctrl', f'p_lag{a:03d}_noctrl',
                      f'p_lag{a:03d}_ctrl']
    cells = cells.merge(src[keep_cols], on='neuron', validate='one_to_one')

    subj = cells['subject_id'].to_numpy()
    inm = cells['in_gradient_mask'].to_numpy(bool)
    fmri = cells['fmri_angle_quarters_deg'].to_numpy(float)
    axes = {'pc1': cells['grad_axis_coord'].to_numpy(float),
            'z': cells['MNI_z'].to_numpy(float)}
    R_by = {c: cells[[f'r_lag{a:03d}_{c}' for a in LAGS]].to_numpy(float)
            for c in ('noctrl', 'ctrl')}
    P_by = {c: cells[[f'p_lag{a:03d}_{c}' for a in LAGS]].to_numpy(float)
            for c in ('noctrl', 'ctrl')}

    binned_splits = {'half': ([0.5], ['begin', 'end']),
                     'tercile': ([1/3, 2/3], ['begin', 'middle', 'end']),
                     'quartile': ([.25, .5, .75], ['Q1', 'Q2', 'Q3', 'Q4'])}

    rng = np.random.default_rng(SEED)
    rows = []
    for ctrl in ('noctrl', 'ctrl'):
        R, P = R_by[ctrl], P_by[ctrl]
        for cs_name, cs in (('all', np.ones(len(cells), bool)),
                            ('in_mask', inm)):
            for ax_name, ax in axes.items():
                coord_sub = ax[cs]
                for gate in ('none', 'permgated'):
                    M = np.ones_like(R, bool) if gate == 'none' else (P < GATE_ALPHA)
                    # binned
                    for sp_name, (edges, names) in binned_splits.items():
                        lab = np.full(len(cells), None, dtype=object)
                        lab[cs] = qcut(coord_sub, edges, names)
                        for g in names:
                            sel = lab == g
                            if sel.sum() < MIN_CONTRIB:
                                continue
                            for w in ('cell', 'subject'):
                                lag, peak = pooled_lag(R, M, sel, w, subj)
                                fa = circ_mean(fmri[sel])
                                rows.append(dict(
                                    ctrl=ctrl, cell_set=cs_name, axis=ax_name,
                                    gating=gate, split=sp_name, group=g,
                                    weighting=w, n=int(sel.sum()),
                                    lag=lag, peak_r=round(peak, 3) if
                                    np.isfinite(peak) else np.nan,
                                    fmri=round(fa, 0) if np.isfinite(fa)
                                    else np.nan,
                                    err=round(circ_err(lag, fa), 0)))
                    # correlation (gate = cell-level: keep cells with >=1 sig lag)
                    for w in ('cell', 'subject'):
                        sel = cs.copy()
                        if gate == 'permgated':
                            sel = sel & (P < GATE_ALPHA).any(1)
                        if sel.sum() < 5:
                            continue
                        r, pv, nn = corr_test(R, sel, ax, w, subj, rng)
                        rows.append(dict(
                            ctrl=ctrl, cell_set=cs_name, axis=ax_name,
                            gating=gate, split='correlation', group='-',
                            weighting=w, n=nn, lag=np.nan, peak_r=round(r, 3),
                            fmri=np.nan, err=round(pv, 3)))

    tbl = pd.DataFrame(rows)
    tbl.to_csv(MASTER_DIR / 'full_factorial.csv', index=False)
    pd.set_option('display.width', 260); pd.set_option('display.max_rows', 200)

    print('=' * 70)
    print('CORRELATION: Spearman(future_score, axis)  [peak_r col = r, '
          'err col = shift-p]')
    print('=' * 70)
    cor = tbl[tbl.split == 'correlation'][
        ['ctrl', 'cell_set', 'axis', 'gating', 'weighting', 'n',
         'peak_r', 'err']].rename(columns={'peak_r': 'spearman_r', 'err': 'p'})
    print(cor.to_string(index=False))

    print('\n' + '=' * 70)
    print('BINNED pooled preferred lag  (rows=split/group, cols=knobs)')
    print('=' * 70)
    piv = tbl[tbl.split != 'correlation'].pivot_table(
        index=['cell_set', 'axis', 'split', 'group'],
        columns=['ctrl', 'weighting', 'gating'], values='lag',
        aggfunc='first')
    print(piv.to_string())

    print(f'\nWrote {MASTER_DIR / "full_factorial.csv"}  ({len(tbl)} rows)')


if __name__ == '__main__':
    main()
