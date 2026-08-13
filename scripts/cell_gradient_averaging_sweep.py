#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep cell-averaging strategies to find the cleanest match to the fMRI
angle gradient.

The fMRI side is settled: within the gradient mask the fMRI preferred angle
rises with z (Spearman +0.50). Here we search the CELL-averaging space: bin
cells along an anatomical axis, pool their 12-lag profiles, read the bin's
arg-max preferred lag, and ask whether those bin means track the fMRI angle
(and the axis) the way the fMRI does.

Factors swept:
  * axis      : z | pc1 (PC1 of the gradient mask)
  * cell_set  : all | in_mask
  * gating    : none | perm (only cells significant at >=1 lag, p_noctrl<.05)
  * binning   : fixed count (10, 15 cells/bin) | quantile (2, 3, 4, 5 bins)
  * weighting : cell | subject (subject-mean profile first, then across)

For each scheme it reports, across bins:
  * circular corr(cell bin arg-max angle, fMRI bin mean angle)  -> cross-modal
  * circular-linear corr(cell bin arg-max angle, axis)          -> does cell
                                                                   angle climb
  * the ordered cell angles and fMRI angles (so you can eyeball).
Top schemes get a circular-shift p (roll each cell's profile, re-pool, re-match).
Writes averaging_sweep.csv and plots the best schemes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

MASTER_DIR = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
    '/derivatives/group/cell_gradient_master/2026-08-13_09-45-28'
)
SRC = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
    '/derivatives/group/per_lag_encoding'
    '/2026-08-04_07-25-15_reload_from_2026-06-30_18-21-57_relabelled'
    '/per_cell_ALL_ROIs.csv'
)
OUT = MASTER_DIR / 'averaging_sweep'
L = np.arange(0, 360, 30)
N_PERM = 2000
CIRC = LinearSegmentedColormap.from_list(
    'wheel', ['#1E88E5', '#43A047', '#FCE300', '#E53935', '#1E88E5'])


def wheel(a):
    return CIRC(Normalize(-180, 180)(((np.asarray(a, float) + 180) % 360) - 180))


def circ_corr(a, b):
    """Jammalamadaka-SenGupta circular-circular correlation."""
    a, b = np.radians(a), np.radians(b)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3:
        return np.nan
    abar = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    bbar = np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    num = np.sum(np.sin(a - abar) * np.sin(b - bbar))
    den = np.sqrt(np.sum(np.sin(a - abar) ** 2) * np.sum(np.sin(b - bbar) ** 2))
    return num / den if den > 0 else np.nan


def circ_lin_corr(theta_deg, x):
    """Circular(theta)-linear(x) correlation coefficient."""
    t = np.radians(theta_deg)
    m = np.isfinite(t) & np.isfinite(x)
    t, x = t[m], x[m]
    if t.size < 3:
        return np.nan
    rxc = np.corrcoef(x, np.cos(t))[0, 1]
    rxs = np.corrcoef(x, np.sin(t))[0, 1]
    rcs = np.corrcoef(np.cos(t), np.sin(t))[0, 1]
    return np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs) / (1 - rcs**2))


def circ_mean(a):
    a = np.radians(np.asarray(a, float))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360


def bin_edges(coord, binning):
    """Return integer bin labels for cells given a binning spec."""
    order = np.argsort(coord)
    lab = np.full(len(coord), -1)
    if binning[0] == 'fixed':
        k = binning[1]
        nb = max(1, len(coord) // k)
        chunks = np.array_split(order, nb)
        for i, ch in enumerate(chunks):
            lab[ch] = i
    else:  # quantile
        q = np.quantile(coord, np.linspace(0, 1, binning[1] + 1)[1:-1])
        lab = np.digitize(coord, q)
    return lab


def pooled_angle(R, sel, weighting, subj):
    if weighting == 'cell':
        p = np.nanmean(R[sel], 0)
    else:
        mat = [np.nanmean(R[sel & (subj == s)], 0) for s in np.unique(subj[sel])]
        p = np.nanmean(np.vstack(mat), 0)
    return int(L[np.nanargmax(p)]) if np.isfinite(p).any() else np.nan


def scheme_angles(R, labels, weighting, subj, fmri, coord):
    """Per-bin cell arg-max angle, fMRI mean angle, mean axis coord."""
    ca, fa, xa = [], [], []
    for b in range(labels.max() + 1):
        sel = labels == b
        if sel.sum() < 3:
            continue
        ca.append(pooled_angle(R, sel, weighting, subj))
        fa.append(circ_mean(fmri[sel]))
        xa.append(np.nanmean(coord[sel]))
    return np.array(ca, float), np.array(fa, float), np.array(xa, float)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(MASTER_DIR / 'per_cell_master.csv')
    s = pd.read_csv(SRC)
    s = s[s.roi == 'mPFC'][['neuron'] + [f'p_lag{a:03d}_noctrl' for a in L]]
    c = c.merge(s, on='neuron')
    R = c[[f'r_lag{a:03d}_noctrl' for a in L]].to_numpy(float)
    P = c[[f'p_lag{a:03d}_noctrl' for a in L]].to_numpy(float)
    subj = c['subject_id'].to_numpy()
    fmri = c['fmri_angle_quarters_deg'].to_numpy(float)
    inm = c['in_gradient_mask'].to_numpy(bool)
    axes = {'z': c['MNI_z'].to_numpy(float),
            'pc1': c['grad_axis_coord'].to_numpy(float)}

    binnings = [('fixed', 10), ('fixed', 15),
                ('quantile', 2), ('quantile', 3),
                ('quantile', 4), ('quantile', 5)]
    rng = np.random.default_rng(42)

    rows = []
    for ax_name, ax in axes.items():
        for cs_name, cs in (('all', np.ones(len(c), bool)), ('in_mask', inm)):
            for gate in ('none', 'perm'):
                sub = cs & ((P < 0.05).any(1) if gate == 'perm' else True)
                idx = np.where(sub)[0]
                if idx.size < 6:
                    continue
                Rs, subjs, fm, cor = R[idx], subj[idx], fmri[idx], ax[idx]
                for binning in binnings:
                    labels = bin_edges(cor, binning)
                    for w in ('cell', 'subject'):
                        ca, fa, xa = scheme_angles(Rs, labels, w, subjs, fm, cor)
                        if len(ca) < 3:
                            continue
                        m_cf = circ_corr(ca, fa)
                        m_cx = circ_lin_corr(ca, xa)
                        rows.append(dict(
                            axis=ax_name, cell_set=cs_name, gating=gate,
                            binning=f'{binning[0]}{binning[1]}', weighting=w,
                            n_cells=idx.size, n_bins=len(ca),
                            match_cell_fmri=round(m_cf, 3),
                            match_cell_axis=round(m_cx, 3),
                            cell_angles='|'.join(f'{a:.0f}' for a in ca),
                            fmri_angles='|'.join(f'{a:.0f}' for a in fa),
                            _R=Rs, _lab=labels, _w=w, _subj=subjs, _fm=fm,
                            _cor=cor))
    tbl = pd.DataFrame(rows)

    # Circular-shift p for the cross-modal match of each scheme.
    def shift_p(row):
        obs = row['match_cell_fmri']
        if not np.isfinite(obs):
            return np.nan
        R_, lab, w, sj, fm = row['_R'], row['_lab'], row['_w'], row['_subj'], row['_fm']
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            sh = rng.integers(0, 12, len(R_))
            Rr = np.take_along_axis(
                R_, (np.arange(12)[None] - sh[:, None]) % 12, 1)
            ca, fa, _ = scheme_angles(Rr, lab, w, sj, fm, fm)
            null[i] = circ_corr(ca, fa)
        return (1 + np.sum(null >= obs)) / (N_PERM + 1)

    tbl['shift_p'] = tbl.apply(shift_p, axis=1)
    out_cols = ['axis', 'cell_set', 'gating', 'binning', 'weighting',
                'n_cells', 'n_bins', 'match_cell_fmri', 'shift_p',
                'match_cell_axis', 'cell_angles', 'fmri_angles']
    tbl[out_cols].to_csv(OUT / 'averaging_sweep.csv', index=False)

    pd.set_option('display.width', 260); pd.set_option('display.max_rows', 40)
    top = tbl.sort_values('match_cell_fmri', ascending=False).head(15)
    print('=== TOP 15 schemes by cross-modal match (cell arg-max vs fMRI) ===')
    print(top[['axis', 'cell_set', 'gating', 'binning', 'weighting', 'n_bins',
               'match_cell_fmri', 'shift_p', 'match_cell_axis',
               'cell_angles', 'fmri_angles']].to_string(index=False))

    # Plot the best few schemes: cell angle + fMRI angle along the axis.
    best = top.head(4).reset_index(drop=True)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax_, (_, row) in zip(axs.ravel(), best.iterrows()):
        ca, fa, xa = scheme_angles(row['_R'], row['_lab'], row['_w'],
                                   row['_subj'], row['_fm'], row['_cor'])
        o = np.argsort(xa)
        ax_.plot(xa[o], fa[o], '--s', color='#23677E', lw=2, ms=6,
                 label='fMRI angle')
        ax_.scatter(xa[o], ca[o], c=wheel(ca[o]), s=90, edgecolor='k',
                    lw=0.5, zorder=3)
        ax_.plot(xa[o], ca[o], '-', color='#555', lw=1, zorder=2,
                 label='cell arg-max')
        ax_.set_title(f"{row['axis']}/{row['cell_set']}/{row['gating']}/"
                      f"{row['binning']}/{row['weighting']}\n"
                      f"match r={row['match_cell_fmri']:+.2f}, "
                      f"p={row['shift_p']:.3f}", fontsize=9)
        ax_.set_xlabel(f"{row['axis']} (ventral -> dorsal)")
        ax_.set_ylabel('preferred angle (deg)')
        ax_.set_yticks(L); ax_.tick_params(labelsize=7)
        ax_.legend(fontsize=7, frameon=False)
    fig.suptitle('Best cell-averaging schemes vs fMRI gradient', fontsize=11)
    fig.savefig(OUT / 'best_schemes_match.png', dpi=200)
    plt.close(fig)
    print(f'\nWrote {OUT / "averaging_sweep.csv"} and best_schemes_match.png')


if __name__ == '__main__':
    main()
