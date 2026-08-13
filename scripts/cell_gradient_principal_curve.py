#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Principal-curve (bent-axis) version of the cell<->gradient analysis.

Instead of a straight PC1 axis, fit a 1-D curve that follows the bending
gradient (ventral-rostral -> dorsal), give every cell an arc-length position
`s` along it plus a perpendicular distance `d`, then slide a window along `s`
and read the pooled preferred lag. This is the continuous, curvature-aware
version of the terciles.

Deliverables
------------
(a) Figure: pooled cell lag (sliding-window argmax of the mean 12-lag profile)
    vs arc-length, overlaid with the fMRI-predicted angle sampled at the cells,
    plus a per-cell future-score scatter. Shows "as you move along the
    gradient, pooled cell tuning tracks the fMRI angle".
(b) Test: Spearman correlation between each cell's future-score and its
    arc-length position, with a circular-shift null (roll each cell's 12-lag
    profile) -> the SAME estimator computes empirical and null, per the
    permutation rule. Reported cell-weighted and with a subject bootstrap CI.

The curve is fit to the gradient-mask geometry (fMRI-independent), so the
anatomical ordering stays the headline. Reads per_cell_master.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import spearmanr


MASTER_DIR = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans'
    '/derivatives/group/cell_gradient_master/2026-08-13_09-45-28'
)
GRAD15_MASK_PATH = Path('/Users/xpsy1114/Documents/projects/multiple_clocks'
                        '/data/masks/gradient_thr_1.5.nii.gz')
OUT_DIR = MASTER_DIR / 'principal_curve'

LAGS_DEG = np.arange(0, 360, 30)
FUTURE_LAGS = [30, 60, 90]          # near-future window (matches fMRI dorsal)
WINDOW_MM = 8.0                     # sliding-window half-width along arc length
N_PERM = 5000
N_BOOT = 2000
SEED = 42

CIRC = LinearSegmentedColormap.from_list(
    'wheel', ['#1E88E5', '#43A047', '#FCE300', '#E53935', '#1E88E5'])


def wheel(a):
    signed = ((np.asarray(a, float) + 180.0) % 360.0) - 180.0
    return CIRC(Normalize(-180, 180)(signed))


# ── Principal curve ──────────────────────────────────────────────────
def fit_curve(mask_yz, n_samples=400, degree=3):
    """Cubic-polynomial principal curve through 2-D (y,z) mask points.

    Parametrise points by their first-PC projection, fit y(t) and z(t),
    resample densely, and return (curve_pts, arc_length_grid)."""
    c = mask_yz - mask_yz.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    t = c @ vt[0]
    tg = np.linspace(t.min(), t.max(), n_samples)
    py = np.polyfit(t, mask_yz[:, 0], degree)
    pz = np.polyfit(t, mask_yz[:, 1], degree)
    curve = np.column_stack([np.polyval(py, tg), np.polyval(pz, tg)])
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    return curve, seg


def project_to_curve(pts_yz, curve, arc):
    """Nearest-point arc-length `s` and perpendicular distance `d` per point."""
    s = np.empty(len(pts_yz))
    d = np.empty(len(pts_yz))
    for i, p in enumerate(pts_yz):
        dist = np.linalg.norm(curve - p, axis=1)
        j = int(np.argmin(dist))
        s[i], d[i] = arc[j], dist[j]
    return s, d


# ── Scores ───────────────────────────────────────────────────────────
def future_score(R):
    """mean r over the near-future window minus mean r over the other lags."""
    fut = np.isin(LAGS_DEG, FUTURE_LAGS)
    return np.nanmean(R[:, fut], 1) - np.nanmean(R[:, ~fut], 1)


def sliding_pooled_argmax(R, s, s_grid, half):
    """Pooled mean-profile argmax (deg) and cell count in each s-window."""
    lag = np.full(len(s_grid), np.nan)
    n = np.zeros(len(s_grid), int)
    for i, s0 in enumerate(s_grid):
        sel = np.abs(s - s0) <= half
        n[i] = sel.sum()
        if sel.sum() >= 5:
            prof = np.nanmean(R[sel], 0)
            lag[i] = LAGS_DEG[np.nanargmax(prof)]
    return lag, n


def sliding_mean(x, s, s_grid, half):
    out = np.full(len(s_grid), np.nan)
    for i, s0 in enumerate(s_grid):
        sel = np.abs(s - s0) <= half
        if sel.sum() >= 5:
            out[i] = np.nanmean(x[sel])
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(MASTER_DIR / 'per_cell_master.csv')
    R = cells[[f'r_lag{a:03d}_noctrl' for a in LAGS_DEG]].to_numpy(float)
    coords = cells[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(float)
    subjects = cells['subject_id'].to_numpy()

    # Fit curve to gradient-mask voxels in the folded-x (y,z) plane.
    mask = nib.load(str(GRAD15_MASK_PATH))
    mni = nib.affines.apply_affine(mask.affine, np.argwhere(mask.get_fdata() > 0))
    curve, arc = fit_curve(mni[:, 1:3])         # (y, z)
    s, d = project_to_curve(coords[:, 1:3], curve, arc)
    # Orient arc length so that increasing s = more dorsal (higher z).
    if np.corrcoef(s, coords[:, 2])[0, 1] < 0:
        s = arc.max() - s
        curve = curve[::-1]
        arc = arc.max() - arc[::-1]

    fs = future_score(R)
    cells['arc_length_s'] = s
    cells['perp_dist_d'] = d
    cells['future_score'] = fs
    cells.to_csv(OUT_DIR / 'per_cell_curve.csv', index=False)

    # ── Test: Spearman(future_score, s) + circular-shift null ──────────
    obs_r, _ = spearmanr(fs, s)
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    for p in range(N_PERM):
        shifts = rng.integers(0, 12, len(R))
        Rs = np.take_along_axis(
            R, (np.arange(12)[None, :] - shifts[:, None]) % 12, axis=1)
        null[p], _ = spearmanr(future_score(Rs), s)
    p_perm = (1 + np.sum(null >= obs_r)) / (N_PERM + 1)

    # Subject bootstrap CI on the Spearman r.
    subj_u = np.unique(subjects)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        take = rng.choice(subj_u, len(subj_u), replace=True)
        idx = np.concatenate([np.where(subjects == u)[0] for u in take])
        boot[b], _ = spearmanr(fs[idx], s[idx])
    ci = np.percentile(boot, [2.5, 97.5])

    print(f'Spearman(future_score, arc_length) = {obs_r:+.3f}')
    print(f'  circular-shift p = {p_perm:.4f}  (null mean {null.mean():+.3f})')
    print(f'  subject-bootstrap 95% CI = [{ci[0]:+.3f}, {ci[1]:+.3f}]')

    # ── Sliding-window pooled readouts along the curve ────────────────
    s_grid = np.linspace(s.min(), s.max(), 60)
    pooled_lag, n_win = sliding_pooled_argmax(R, s, s_grid, WINDOW_MM)
    fmri_along = sliding_mean(cells['fmri_angle_quarters_deg'].to_numpy(float),
                              s, s_grid, WINDOW_MM)
    fs_along = sliding_mean(fs, s, s_grid, WINDOW_MM)

    # ── Figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 1, figsize=(6.5, 7), constrained_layout=True,
                           sharex=True)
    # Panel 1: pooled cell lag + fMRI angle along the curve
    ax[0].scatter(s, cells['argmax_lag_deg'], c=wheel(cells['argmax_lag_deg']),
                  s=18, alpha=0.5, edgecolor='none', label='cells (argmax)')
    ax[0].plot(s_grid, pooled_lag, '-o', color='#0e3d3a', ms=4, lw=2,
               label=f'pooled cell lag (±{WINDOW_MM:.0f} mm window)')
    ax[0].plot(s_grid, fmri_along, '--', color='#23677E', lw=2,
               label='fMRI angle (quarters) at cells')
    ax[0].set_ylabel('preferred angle / lag (deg)')
    ax[0].set_yticks(LAGS_DEG)
    ax[0].tick_params(labelsize=7)
    ax[0].legend(fontsize=7, frameon=False, loc='upper left')
    ax[0].set_title('Pooled cell lag along the bent gradient axis')
    # Panel 2: future-score vs arc length (the tested quantity)
    ax[1].scatter(s, fs, s=16, c='#888', alpha=0.5, edgecolor='none')
    ax[1].plot(s_grid, fs_along, '-', color='#5C1027', lw=2,
               label='sliding-window mean')
    ax[1].axhline(0, color='k', lw=0.5, ls='--')
    ax[1].set_xlabel('arc length along gradient curve (mm; ventral -> dorsal)')
    ax[1].set_ylabel('future-score\n(r[30/60/90] - r[rest])')
    ax[1].legend(fontsize=7, frameon=False, loc='upper left')
    ax[1].set_title(f'Future-score vs position   '
                    f'Spearman r={obs_r:+.3f}, shift-p={p_perm:.3g}, '
                    f'boot95%=[{ci[0]:+.2f},{ci[1]:+.2f}]', fontsize=9)
    fig.savefig(OUT_DIR / 'principal_curve_pooled_lag.png', dpi=200)
    plt.close(fig)

    config = dict(master_dir=str(MASTER_DIR), future_lags=FUTURE_LAGS,
                  window_mm=WINDOW_MM, n_perm=N_PERM, n_boot=N_BOOT, seed=SEED,
                  curve='cubic principal curve on gradient-mask (y,z) voxels',
                  spearman_r=float(obs_r), shift_p=float(p_perm),
                  boot_ci=[float(ci[0]), float(ci[1])])
    (OUT_DIR / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'\nOutputs in {OUT_DIR}')


if __name__ == '__main__':
    main()
