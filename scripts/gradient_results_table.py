#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated results table for the mPFC future-gradient ventral/dorsal analysis.

Every statistic quoted for Figure 3d in one tidy CSV, so the manuscript numbers
have a single source. Three blocks:

  contrast    Does the dorsal cluster peak LATER than the ventral one? This is
              what the sentence "the more ventral cluster peaked at 30 deg, and
              the more dorsal cluster at 60 deg" actually asserts. Tested by
              permuting the ventral/dorsal label across RECORDING SITES and
              rebuilding the pooled profiles through the same code path, because
              cells on one microwire bundle share a coordinate exactly (87
              in-mask cells sit at 19 sites).

  vs_zero     Is each cluster above zero at its peak / in the pre-specified
              30+60 deg window? Read from gradient_split_vs_zero.csv
              (scripts/gradient_split_vs_zero.py).

  fmri_z      Where does each cluster sit on the fMRI gradient, read from the
              map's z-profile rather than from single-voxel lookups at cell
              coordinates? Read from fmri_angle_by_group.csv
              (scripts/gradient_fmri_z_projection.py).

Run those two scripts first; this one merges their output and adds the contrast
tests, which exist nowhere else.

Output: <run>/final_splits/gradient_results_summary.csv

Usage:
    conda activate env_multiple_clocks
    python scripts/gradient_results_table.py

@author: Svenja Kuchenhoff
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# >>> RERUN-CHECK: hardcoded upstream run dir -- update after re-running
# cell_gradient_master_table.py
RUN = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
           '/derivatives/group/cell_gradient_master/2026-08-28_15-19-35')

LAGS_DEG = list(range(0, 360, 30))
N_PERM = 10000
SEED = 42


def load_cells():
    m = pd.read_csv(RUN / 'per_cell_master.csv')
    f = pd.read_csv(RUN / 'final_splits' / 'final_splits_per_cell.csv')
    cols = [f'r_lag{a:03d}_noctrl' for a in LAGS_DEG]
    j = (f[f.in_gradient_mask == True]
         [['neuron', 'subject_id', 'MNI_x', 'MNI_y', 'MNI_z', 'fmri_angle',
           'pc1_ventral_dorsal_group']]
         .merge(m[['neuron', 'harmonic_angle_deg', 'grad_axis_coord'] + cols],
                on='neuron', how='left'))
    j['site'] = (j.MNI_x.round(2).astype(str) + '_' +
                 j.MNI_y.round(2).astype(str) + '_' +
                 j.MNI_z.round(2).astype(str))
    return j, cols


def pooled_argmax(R, mask):
    return LAGS_DEG[int(np.nanargmax(np.nanmean(R[mask], axis=0)))]


def pooled_circmean(R, mask):
    prof = np.nanmean(R[mask], axis=0)
    w = np.clip(prof - np.nanmin(prof), 0, None)
    return np.rad2deg(np.angle(np.sum(w * np.exp(1j * np.deg2rad(LAGS_DEG))))) % 360


def circ_lin_corr(theta, x):
    s, c = np.sin(theta), np.cos(theta)
    rxs = np.corrcoef(x, s)[0, 1]
    rxc = np.corrcoef(x, c)[0, 1]
    rcs = np.corrcoef(c, s)[0, 1]
    return np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs) / (1 - rcs**2))


def circ_circ_corr(a, b):
    a1 = a - np.angle(np.mean(np.exp(1j*a)))
    b1 = b - np.angle(np.mean(np.exp(1j*b)))
    return (np.sum(np.sin(a1)*np.sin(b1)) /
            np.sqrt(np.sum(np.sin(a1)**2) * np.sum(np.sin(b1)**2)))


def signed(x):
    """Wrap a circular difference to (-180, 180] so 'later' is positive."""
    return np.where(np.asarray(x) > 180, np.asarray(x) - 360, np.asarray(x))


def contrast_tests(j, cols):
    rng = np.random.default_rng(SEED)
    R = j[cols].to_numpy(float)
    grp = j.pc1_ventral_dorsal_group.to_numpy()
    site = j.site.to_numpy()
    usite = np.unique(site)
    site_grp = {s: grp[site == s][0] for s in usite}
    labels = np.array([site_grp[s] for s in usite])

    v, d = grp == 'ventral', grp == 'dorsal'
    obs_arg = (pooled_argmax(R, d) - pooled_argmax(R, v)) % 360
    obs_circ = (pooled_circmean(R, d) - pooled_circmean(R, v)) % 360

    n_arg, n_circ = [], []
    for _ in range(N_PERM):
        perm = rng.permutation(labels)
        mp = dict(zip(usite, perm))
        g = np.array([mp[s] for s in site])
        if (g == 'ventral').sum() < 3 or (g == 'dorsal').sum() < 3:
            continue
        n_arg.append((pooled_argmax(R, g == 'dorsal')
                      - pooled_argmax(R, g == 'ventral')) % 360)
        n_circ.append((pooled_circmean(R, g == 'dorsal')
                       - pooled_circmean(R, g == 'ventral')) % 360)
    p_arg = (np.sum(signed(n_arg) >= signed(obs_arg)) + 1) / (len(n_arg) + 1)
    p_circ = (np.sum(signed(n_circ) >= signed(obs_circ)) + 1) / (len(n_circ) + 1)

    # continuous alternatives to the median split
    th = np.deg2rad(j.harmonic_angle_deg.to_numpy(float))
    ax = j.grad_axis_coord.to_numpy(float)
    ok = np.isfinite(th) & np.isfinite(ax)
    site_ax = {s: ax[site == s][0] for s in usite}
    r_cl = circ_lin_corr(th[ok], ax[ok])
    null_cl = []
    for _ in range(N_PERM):
        mp = dict(zip(usite, rng.permutation(usite)))
        null_cl.append(circ_lin_corr(th[ok],
                                     np.array([site_ax[mp[s]] for s in site[ok]])))
    p_cl = (np.sum(np.array(null_cl) >= r_cl) + 1) / (N_PERM + 1)

    fa = np.deg2rad(j.fmri_angle.to_numpy(float))
    ok2 = ok & np.isfinite(fa)
    site_fa = {s: fa[site == s][0] for s in usite}
    r_cc = circ_circ_corr(th[ok2], fa[ok2])
    null_cc = []
    for _ in range(N_PERM):
        mp = dict(zip(usite, rng.permutation(usite)))
        null_cc.append(circ_circ_corr(
            th[ok2], np.array([site_fa[mp[s]] for s in site[ok2]])))
    p_cc = (np.sum(np.array(null_cc) >= r_cc) + 1) / (N_PERM + 1)

    n_sites, n_cells = len(usite), len(j)
    return [
        dict(block='contrast', test='pooled_argmax_difference', unit='site',
             group='dorsal_minus_ventral', n=n_sites, n_cells=n_cells,
             statistic='degrees', value=float(obs_arg), p_one_sided=float(p_arg),
             note=(f'ventral {pooled_argmax(R, v)} deg -> dorsal '
                   f'{pooled_argmax(R, d)} deg; null median '
                   f'{np.median(signed(n_arg)):.0f} deg, IQR '
                   f'{np.percentile(signed(n_arg),25):.0f} to '
                   f'{np.percentile(signed(n_arg),75):.0f}')),
        dict(block='contrast', test='pooled_circular_mean_difference', unit='site',
             group='dorsal_minus_ventral', n=n_sites, n_cells=n_cells,
             statistic='degrees', value=float(obs_circ), p_one_sided=float(p_circ),
             note=(f'ventral {pooled_circmean(R, v):.0f} deg -> dorsal '
                   f'{pooled_circmean(R, d):.0f} deg')),
        dict(block='contrast', test='circular_linear_corr_angle_vs_axis',
             unit='site', group='all_in_mask', n=n_sites, n_cells=int(ok.sum()),
             statistic='r', value=float(r_cl), p_one_sided=float(p_cl),
             note='continuous alternative to the median split'),
        dict(block='contrast', test='circular_circular_corr_cell_vs_fmri',
             unit='site', group='all_in_mask', n=n_sites, n_cells=int(ok2.sum()),
             statistic='r', value=float(r_cc), p_one_sided=float(p_cc),
             note='cell preferred angle vs fMRI angle sampled at its own voxel'),
    ]


def main():
    j, cols = load_cells()
    rows = contrast_tests(j, cols)

    d = RUN / 'final_splits'
    vz = pd.read_csv(d / 'gradient_split_vs_zero.csv')
    keep = vz.is_group_peak == True
    for _, r in vz[keep].iterrows():
        rows.append(dict(
            block='vs_zero', test=f'{r.family}_vs_zero_one_sided', unit=r.unit,
            group=r.group, n=int(r.n), n_cells=np.nan, statistic='mean_r',
            value=float(r.mean_r), p_one_sided=float(r.p_one_sided),
            p_fdr_across_groups=float(r.p_fdr_across_groups),
            p_fdr_across_12_lags=(float(r.p_fdr_across_12_lags)
                                  if pd.notna(r.p_fdr_across_12_lags) else np.nan),
            note=(f't = {r.t:.2f}'
                  + (f', lag {r.lag_deg:.0f} deg' if pd.notna(r.lag_deg) else ''))))

    fz = pd.read_csv(d / 'fmri_angle_by_group.csv')
    for _, r in fz.iterrows():
        rows.append(dict(
            block='fmri_z', test='fmri_angle_at_group_mean_z', unit='voxel',
            group=r.group, n=int(r.n_sites), n_cells=int(r.n_cells),
            statistic='degrees', value=float(r.fmri_angle_at_z_mean_deg),
            p_one_sided=np.nan,
            note=(f'z mean {r.z_mean:.1f} (range {r.z_min:.1f} to {r.z_max:.1f}); '
                  f'fMRI angle range {r.fmri_angle_lo_deg:.0f}-'
                  f'{r.fmri_angle_hi_deg:.0f} deg; descriptive, no test')))

    out = pd.DataFrame(rows)
    order = ['block', 'test', 'unit', 'group', 'n', 'n_cells', 'statistic',
             'value', 'p_one_sided', 'p_fdr_across_groups',
             'p_fdr_across_12_lags', 'note']
    out = out.reindex(columns=order)
    dest = d / 'gradient_results_summary.csv'
    out.to_csv(dest, index=False)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 46)
    print(out.drop(columns=['n_cells']).round(4).to_string(index=False))
    print(f"\nSaved -> {dest}")


if __name__ == '__main__':
    main()
