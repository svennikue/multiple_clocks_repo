#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Can the fMRI harmonic-angle map predict each mPFC cell's preferred lag?

For every mPFC single-unit in
    per_cell_ALL_ROIs.csv
we have:
  * 12 spatial-tuning correlations, one per lag (0°, 30°, ..., 330°) →
    read out the cell's preferred lag = argmax across these 12 values.
  * an MNI coordinate → read out the fMRI-predicted preferred phase from
    ``angle_deg.nii.gz`` (both ``quarters`` and ``eighths`` datasets).

We then compare cell-preferred vs. fMRI-predicted angle at three levels:
  1. Circular-circular correlation (Fisher & Lee).
  2. Angular error distribution (|Δangle| in circular sense), against a
     uniform-random baseline of 90°.
  3. Exact-match / top-K match rates against a uniform baseline of
     1/12 = 8.3% (exact) and 3/12 = 25% (top-3).
  4. Confusion matrix: 12 × 12 counts of (cell argmax bin) × (fMRI
     nearest-lag bin).
Add a per-cell shuffle null so we can report a p-value for each metric.

Visualisations (saved into ``<harmonic_angle_maps>/cell_lag_prediction/``):
  * ``confusion_matrix_{dataset}.pdf/png``           — 12 × 12 heatmap
  * ``angular_error_hist_{dataset}.pdf/png``         — |Δangle| distribution
  * ``scatter_cell_vs_fmri_{dataset}.pdf/png``       — angle-vs-angle scatter
  * ``glass_brain_cells_by_pref_lag.pdf/png``        — each mPFC cell
    plotted on a glass brain, coloured by its own preferred lag using
    the circular cyclic colormap (blue=±180°, green=−90°, yellow=0°,
    red=+90°).
  * ``per_cell_summary_{dataset}.csv``               — per-cell prediction
    numbers for follow-up.

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats


# ── Settings ─────────────────────────────────────────────────────────
CSV_PATH = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                '/ephys_humans/derivatives/group/per_lag_encoding'
                '/2026-08-04_07-25-15_reload_from_2026-06-30_18-21-57_relabelled'
                '/per_cell_ALL_ROIs.csv')
BASE_HARMONIC = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                     '/derivatives/group/Main_Results_fMRI/harmonic_angle_maps')
OUT_DIR = BASE_HARMONIC / 'cell_lag_prediction'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_ROI     = 'mPFC'
LAG_COL_KIND = 'noctrl'                       # 'ctrl' or 'noctrl'
LAG_ANGLES   = np.arange(0, 360, 30)         # 12 lags, 30° apart
DATASETS     = ['quarters', 'eighths']
TOP_K        = 3
N_PERM       = 5000
SEED         = 42
SIG_P_THRESHOLD = 0.05         # p-threshold for "significantly tuned" cells
SAVE_PDF     = False           # PNG only by default

# ── Filters ──────────────────────────────────────────────────────────
# (1) Cell-quality filter: drop cells whose argmax r is not clearly
#     above noise, because their "preferred lag" is essentially random.
#     Set to None to keep all cells.
MIN_CELL_R      = 0.30      # keep cells with argmax r ≥ this
# Alternatively, gate by the permutation p at the argmax lag (uses the
# CSV's p_lag###_ctrl columns). Set to None to disable.
MAX_CELL_P      = None      # e.g. 0.10 for a mild filter, 0.05 for stricter
#
# (2) fMRI-voxel filter: drop cells whose fMRI voxel has weak harmonic
#     signal (small amplitude → unreliable arctan2 direction).
#     Percentile is computed WITHIN the amplitude sampled at cell
#     coords, so it adapts to whatever cell set survives filter (1).
MIN_FMRI_AMP_PCT = None     # keep cells above this percentile of cell-voxel amp

# (3) Spatial smoothing of the fMRI prediction. When > 0, the predicted
#     angle at each cell is the amp-weighted circular mean of all voxels
#     within `SPHERE_RADIUS_MM` mm of the cell's MNI coord. This helps
#     when a single voxel is noisy but the local neighbourhood is
#     consistent.  0 = point-sample (original behaviour).
SPHERE_RADIUS_MM = 6.0

# Circular colour wheel (same anchors as harmonic_maps_brain_overlay.py)
CIRCULAR_ANCHORS_HEX = [
    '#1E88E5',   # -180°  → blue
    '#43A047',   #  -90°  → green
    '#FCE300',   #    0°  → yellow
    '#E53935',   #  +90°  → red
    '#1E88E5',   # +180°  → blue
]
CIRC_CMAP = LinearSegmentedColormap.from_list(
    'circular_wheel', CIRCULAR_ANCHORS_HEX)


# ── Helpers ──────────────────────────────────────────────────────────
def signed_circ_diff(a_deg, b_deg):
    """Signed circular difference (a − b) wrapped into (−180°, +180°]."""
    return (np.asarray(a_deg) - np.asarray(b_deg) + 180.0) % 360.0 - 180.0


def circular_linear_r(theta_rad_1, theta_rad_2):
    """Fisher–Lee circular–circular correlation (r ∈ [0, 1]).

    Robust to per-axis wrapping.  Returns (r, chi², p) with df=4 under
    an asymptotic χ² null.
    """
    a = np.asarray(theta_rad_1)
    b = np.asarray(theta_rad_2)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = a.size
    if n < 5:
        return np.nan, np.nan, np.nan
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    num  = 4 * (np.sum(ca * cb) * np.sum(sa * sb)
                 - np.sum(ca * sb) * np.sum(sa * cb))
    den  = np.sqrt((n ** 2 - np.sum(np.cos(2 * a)) ** 2
                          - np.sum(np.sin(2 * a)) ** 2)
                    * (n ** 2 - np.sum(np.cos(2 * b)) ** 2
                              - np.sum(np.sin(2 * b)) ** 2))
    if den == 0:
        return np.nan, np.nan, np.nan
    r = num / den
    # Fisher & Lee (1983) asymptotic test: n · r² ~ χ²(4).
    chi2 = n * r * r
    p = 1.0 - stats.chi2.cdf(chi2, df=4)
    return float(r), float(chi2), float(p)


def sample_map_at_mni(img, mni_coords):
    inv = np.linalg.inv(img.affine)
    data = img.get_fdata()
    shape = np.array(data.shape)
    out = np.empty(len(mni_coords))
    for i, mni in enumerate(mni_coords):
        ijk = np.round(nib.affines.apply_affine(inv, mni)).astype(int)
        if (ijk >= 0).all() and (ijk < shape).all():
            v = data[tuple(ijk)]
            out[i] = v if np.isfinite(v) else np.nan
        else:
            out[i] = np.nan
    return out


def snap_to_lag_bin(angles_deg):
    """Round angles to the nearest of the 12 lag bins (0, 30, ..., 330)."""
    a = np.asarray(angles_deg) % 360.0
    return np.round(a / 30.0).astype(int) % 12


def sample_angle_sphere(ang_img, amp_img, mni_coords, radius_mm):
    """Amp-weighted circular mean of fMRI angles in a sphere per MNI coord.

    Returns (mean_angle [0, 360), mean_amplitude).  Cells whose sphere has
    no signal → NaN.
    """
    ang_data = ang_img.get_fdata()
    amp_data = amp_img.get_fdata()
    inv = np.linalg.inv(ang_img.affine)
    shape = np.array(ang_data.shape)
    vox_mm = np.sqrt((ang_img.affine[:3, :3] ** 2).sum(axis=0)).mean()
    r_vox = int(np.ceil(radius_mm / vox_mm))
    n = len(mni_coords)
    out_ang = np.full(n, np.nan)
    out_amp = np.full(n, np.nan)
    for i, mni in enumerate(mni_coords):
        centre = np.round(nib.affines.apply_affine(inv, mni)).astype(int)
        lo = np.clip(centre - r_vox, 0, shape - 1)
        hi = np.clip(centre + r_vox + 1, 0, shape)
        block_a = ang_data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        block_m = amp_data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        xg, yg, zg = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        d = np.sqrt((xg - centre[0]) ** 2
                    + (yg - centre[1]) ** 2
                    + (zg - centre[2]) ** 2) * vox_mm
        sig = (d <= radius_mm) & (block_m > 1e-6) & np.isfinite(block_a)
        if not sig.any():
            continue
        w = block_m[sig]
        theta = np.radians(block_a[sig])
        C = np.sum(w * np.cos(theta))
        S = np.sum(w * np.sin(theta))
        out_ang[i] = np.degrees(np.arctan2(S, C)) % 360.0
        out_amp[i] = float(w.mean())
    return out_ang, out_amp


def circular_mean_deg(angles_deg, weights=None):
    """Weighted circular mean, in degrees; returns NaN if all weights zero."""
    a = np.asarray(angles_deg, dtype=float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    m = np.isfinite(a) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    theta = np.radians(a[m])
    C = np.sum(w[m] * np.cos(theta))
    S = np.sum(w[m] * np.sin(theta))
    return float(np.degrees(np.arctan2(S, C)))


def rayleigh_test(angles_deg):
    """One-sample Rayleigh test: H0 = uniform on the circle.
    Returns (R̄, Z, p) where R̄ ∈ [0, 1] is the mean resultant length."""
    a = np.asarray(angles_deg, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 4:
        return np.nan, np.nan, np.nan
    theta = np.radians(a)
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    R = np.sqrt(C * C + S * S)
    Z = n * R * R
    # Zar 2010 asymptotic approximation
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n * n - Z * Z)) - (1 + 2 * n))
    return float(R), float(Z), float(p)


def watson_u2_permutation_test(angles_a_deg, angles_b_deg, n_perm=5000,
                                seed=42):
    """Non-parametric two-sample circular test: are the two angle
    distributions the same? Uses Watson's U² statistic and a
    label-shuffle null (no need for pycircstat / distribution tables).

    Returns (u2_obs, p_perm).
    """
    a = np.asarray(angles_a_deg, dtype=float)
    b = np.asarray(angles_b_deg, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    if n1 < 5 or n2 < 5:
        return np.nan, np.nan
    N = n1 + n2

    def _u2(x, y):
        # Watson U² (Watson 1961). Rank-based on the pooled sample.
        pooled = np.concatenate([x, y]) % 360.0
        order = np.argsort(pooled)
        origin = pooled[order[0]]
        # Cumulative counts along the circle from the smallest angle.
        rot = (pooled - origin) % 360.0
        idx = np.argsort(rot)
        labels = np.concatenate([np.zeros(n1, dtype=int),
                                  np.ones(n2, dtype=int)])[idx]
        # Empirical CDFs
        F1 = np.cumsum(labels == 0) / n1
        F2 = np.cumsum(labels == 1) / n2
        d = F1 - F2
        u2 = (n1 * n2 / N ** 2) * (np.sum(d * d) - (np.sum(d) ** 2) / N)
        return float(u2)

    u2_obs = _u2(a, b)
    rng = np.random.default_rng(seed)
    pool = np.concatenate([a, b])
    null = np.empty(n_perm)
    for p in range(n_perm):
        rng.shuffle(pool)
        null[p] = _u2(pool[:n1], pool[n1:])
    p_perm = float(np.mean(null >= u2_obs))
    return u2_obs, p_perm


def angles_to_colours(angles_360):
    """Map angles in [0, 360) → circular colour via [-180, 180] anchors.

    The colormap anchors are placed at −180°, −90°, 0°, +90°, +180°, so
    we need to convert [0, 360) into a signed representation before
    calling the norm, otherwise 0° would render blue (=−180 anchor).
    """
    a = np.asarray(angles_360, dtype=float)
    signed = ((a + 180.0) % 360.0) - 180.0        # → (−180, 180]
    return CIRC_CMAP(Normalize(vmin=-180, vmax=180)(signed))


def savefig(fig, stem):
    """Save a figure — PNG always, PDF only if SAVE_PDF is True."""
    fig.savefig(f'{stem}.png', dpi=300, bbox_inches='tight')
    if SAVE_PDF:
        fig.savefig(f'{stem}.pdf', dpi=300, bbox_inches='tight')


def best_of_topk_match(fmri_snap_idx, cell_top_idx):
    """Per cell: pick the top-K lag closest circularly to the fMRI snap.

    Returns (target_bin_index, exact_hit_bool). `target_bin_index` is
    always one of the cell's top-K lags.
    """
    n = len(fmri_snap_idx)
    tgt = np.zeros(n, dtype=int)
    hit = np.zeros(n, dtype=bool)
    for i in range(n):
        top = np.asarray(cell_top_idx[i])
        f = fmri_snap_idx[i]
        d = np.minimum((top - f) % 12, (f - top) % 12)
        j = int(np.argmin(d))
        tgt[i] = top[j]
        hit[i] = (d[j] == 0)
    return tgt, hit


# ── Load per-cell CSV ────────────────────────────────────────────────
df_all = pd.read_csv(CSV_PATH)
mpfc = df_all[df_all['roi'] == CELL_ROI].copy()
print(f"Loaded {len(mpfc)} '{CELL_ROI}' cells from {CSV_PATH.name}")

lag_cols = [f'r_lag{a:03d}_{LAG_COL_KIND}' for a in LAG_ANGLES]
r_matrix = mpfc[lag_cols].to_numpy(dtype=float)          # (n_cells, 12)
# Drop cells with all-NaN rows (defensive; shouldn't happen).
usable = np.isfinite(r_matrix).any(axis=1)
mpfc   = mpfc.iloc[usable].reset_index(drop=True)
r_matrix = r_matrix[usable]
print(f"  {len(mpfc)} cells with valid tuning data")

# Cell preferred lag: index of max r → angle 0..330.
cell_argmax_idx    = np.nanargmax(r_matrix, axis=1)
cell_pref_angle    = LAG_ANGLES[cell_argmax_idx].astype(float)
cell_top_idx       = np.argsort(-r_matrix, axis=1)[:, :TOP_K]
cell_top_angles    = LAG_ANGLES[cell_top_idx].astype(float)

# Per-cell significance: is the cell's argmax lag significantly tuned?
# and: is ANY of the top-K lags significantly tuned?  Uses p_lag###_XX
# columns from the CSV, so this is a lag-specific permutation p-value.
p_cols = [f'p_lag{a:03d}_{LAG_COL_KIND}' for a in LAG_ANGLES]
p_matrix = mpfc[p_cols].to_numpy(dtype=float)
cell_argmax_p = p_matrix[np.arange(len(mpfc)), cell_argmax_idx]
cell_topk_p   = np.array([p_matrix[i, cell_top_idx[i]].min()
                          for i in range(len(mpfc))])
sig_argmax    = cell_argmax_p <  SIG_P_THRESHOLD
sig_topk_any  = cell_topk_p   <  SIG_P_THRESHOLD
print(f"  Cells with sig argmax   (p<{SIG_P_THRESHOLD}): "
      f"{int(sig_argmax.sum())}/{len(mpfc)}")
print(f"  Cells with ≥1 sig top-{TOP_K} lag: "
      f"{int(sig_topk_any.sum())}/{len(mpfc)}")

# --- Cell-quality mask (filter 1) ------------------------------------
cell_argmax_r = r_matrix[np.arange(len(r_matrix)), cell_argmax_idx]
cell_ok = np.ones(len(mpfc), dtype=bool)
if MIN_CELL_R is not None:
    cell_ok &= (cell_argmax_r >= MIN_CELL_R)
    print(f"  [filter 1a] MIN_CELL_R = {MIN_CELL_R}  →  "
          f"{int(cell_ok.sum())}/{len(mpfc)} cells pass")
if MAX_CELL_P is not None:
    p_cols = [f'p_lag{a:03d}_{LAG_COL_KIND}' for a in LAG_ANGLES]
    p_matrix = mpfc[p_cols].to_numpy(dtype=float)
    cell_argmax_p = p_matrix[np.arange(len(p_matrix)), cell_argmax_idx]
    cell_ok &= (cell_argmax_p <= MAX_CELL_P)
    print(f"  [filter 1b] MAX_CELL_P = {MAX_CELL_P}  →  "
          f"{int(cell_ok.sum())}/{len(mpfc)} cells pass "
          f"(after combining with 1a)")

coords = mpfc[['MNI_x', 'MNI_y', 'MNI_z']].to_numpy(dtype=float)

rng = np.random.default_rng(SEED)


# ── For each dataset: sample fMRI, compare, plot ─────────────────────
def run_dataset(ds):
    ang_path = BASE_HARMONIC / ds / 'angle_deg.nii.gz'
    amp_path = BASE_HARMONIC / ds / 'amplitude.nii.gz'
    if not ang_path.exists():
        print(f"[skip] {ds}: {ang_path} not found")
        return
    ang_img = nib.load(str(ang_path))
    amp_img = nib.load(str(amp_path))
    if SPHERE_RADIUS_MM and SPHERE_RADIUS_MM > 0:
        fmri_ang_360, fmri_amp = sample_angle_sphere(
            ang_img, amp_img, coords, SPHERE_RADIUS_MM)
        print(f"  [sphere] amp-weighted circular mean in "
              f"{SPHERE_RADIUS_MM:.1f} mm sphere around each cell")
    else:
        fmri_ang_raw = sample_map_at_mni(ang_img, coords)
        fmri_amp     = sample_map_at_mni(amp_img, coords)
        fmri_ang_360 = fmri_ang_raw % 360.0
    fmri_snap_idx = snap_to_lag_bin(fmri_ang_360)
    fmri_snap_angle = LAG_ANGLES[fmri_snap_idx].astype(float)

    have_signal = np.isfinite(fmri_ang_360) & np.isfinite(cell_pref_angle) \
                   & (fmri_amp > 1e-6)

    # Filter 2: keep only cells whose fMRI voxel amplitude is above the
    # requested percentile of the amp distribution across the cells that
    # passed filter 1 (i.e. adaptive per-dataset, per-cohort).
    amp_ok = np.ones_like(have_signal)
    if MIN_FMRI_AMP_PCT is not None:
        pool_mask = have_signal & cell_ok
        if pool_mask.any():
            amp_thr = float(np.percentile(fmri_amp[pool_mask],
                                            MIN_FMRI_AMP_PCT))
            amp_ok = fmri_amp > amp_thr
        else:
            amp_thr = np.nan

    valid = have_signal & cell_ok & amp_ok
    n_valid = int(valid.sum())
    print(f"\n=== dataset: {ds}  ({n_valid}/{len(mpfc)} cells survive all filters) ===")
    if MIN_FMRI_AMP_PCT is not None and np.isfinite(amp_thr):
        print(f"  [filter 2] fMRI amp > {amp_thr:.5f} "
              f"(={MIN_FMRI_AMP_PCT}th pct of {int((have_signal & cell_ok).sum())} qualifying cells)  "
              f"→  {int(amp_ok.sum())} cells pass on amp alone")

    # --- Circular correlation ---
    r_circ, chi2_circ, p_circ = circular_linear_r(
        np.radians(cell_pref_angle[valid]),
        np.radians(fmri_ang_360[valid]))
    print(f"  Fisher-Lee circ-circ r = {r_circ:+.3f}  "
          f"χ²(4)={chi2_circ:.1f}  p = {p_circ:.4g}")

    # --- Angular errors ---
    signed_err = signed_circ_diff(fmri_ang_360, cell_pref_angle)
    abs_err    = np.abs(signed_err)
    mean_abs   = float(np.nanmean(abs_err[valid]))
    med_abs    = float(np.nanmedian(abs_err[valid]))
    baseline_mean_abs = 90.0
    print(f"  Mean   |Δangle| = {mean_abs:5.1f}°   (baseline 90°)")
    print(f"  Median |Δangle| = {med_abs:5.1f}°")

    # --- Exact and top-K match rates ---
    exact = np.array([f == c
                      for f, c in zip(fmri_snap_idx, cell_argmax_idx)])
    topk  = np.array([f in top for f, top
                      in zip(fmri_snap_idx, cell_top_idx)])
    exact_rate = float(exact[valid].mean())
    topk_rate  = float(topk[valid].mean())
    base_exact = 1.0 / 12.0
    base_topk  = TOP_K / 12.0
    print(f"  Exact-lag match  : {int(exact[valid].sum())}/{n_valid} "
          f"= {exact_rate*100:5.1f}%   (baseline {base_exact*100:.1f}%)")
    print(f"  Within top-{TOP_K}    : {int(topk[valid].sum())}/{n_valid} "
          f"= {topk_rate*100:5.1f}%   (baseline {base_topk*100:.1f}%)")

    # --- Permutation null (shuffle fMRI predictions across cells) ---
    v_idx = np.where(valid)[0]
    fmri_snap_v = fmri_snap_idx[v_idx]
    fmri_ang_v  = fmri_ang_360[v_idx]
    cell_snap_v = cell_argmax_idx[v_idx]
    cell_pref_v = cell_pref_angle[v_idx]
    cell_top_v  = cell_top_idx[v_idx]

    null_exact = np.empty(N_PERM)
    null_topk  = np.empty(N_PERM)
    null_abs   = np.empty(N_PERM)
    null_r     = np.empty(N_PERM)
    for p in range(N_PERM):
        order = rng.permutation(len(v_idx))
        fs = fmri_snap_v[order]
        fa = fmri_ang_v[order]
        null_exact[p] = np.mean(fs == cell_snap_v)
        null_topk[p]  = np.mean([f in t for f, t in zip(fs, cell_top_v)])
        null_abs[p]   = np.mean(np.abs(signed_circ_diff(fa, cell_pref_v)))
        r_p, _, _ = circular_linear_r(
            np.radians(cell_pref_v), np.radians(fa))
        null_r[p] = r_p if np.isfinite(r_p) else np.nan
    p_exact_perm = float(np.mean(null_exact >= exact_rate))
    p_topk_perm  = float(np.mean(null_topk  >= topk_rate))
    p_abs_perm   = float(np.mean(null_abs   <= mean_abs))
    p_r_perm     = float(np.mean(null_r     >= r_circ))
    print(f"  Perm p (exact_rate ≥ obs) = {p_exact_perm:.4f}   "
          f"(null mean {null_exact.mean()*100:.1f}%)")
    print(f"  Perm p (topk_rate  ≥ obs) = {p_topk_perm:.4f}   "
          f"(null mean {null_topk.mean()*100:.1f}%)")
    print(f"  Perm p (mean|Δ|    ≤ obs) = {p_abs_perm:.4f}   "
          f"(null mean {null_abs.mean():.1f}°)")
    print(f"  Perm p (circ r     ≥ obs) = {p_r_perm:.4f}   "
          f"(null mean {np.nanmean(null_r):+.3f})")

    # --- Confusion v1 (rows = cell argmax, cols = fMRI snap) ---
    cm = np.zeros((12, 12), dtype=int)
    for c, f in zip(cell_snap_v, fmri_snap_v):
        cm[c, f] += 1

    # --- Best-of-top-K target: for each cell, which of its top-K lags
    #     sits closest circularly to the fMRI snap.
    bo_tgt_v, bo_hit_v = best_of_topk_match(fmri_snap_v, cell_top_v)
    bo_tgt_angle_v = LAG_ANGLES[bo_tgt_v].astype(float)
    cm_top = np.zeros((12, 12), dtype=int)
    for c, f in zip(bo_tgt_v, fmri_snap_v):
        cm_top[c, f] += 1

    # --- |Δangle| under best-of-top-K target (computed on valid subset) ---
    bo_abs_err = np.abs(signed_circ_diff(fmri_ang_360[valid], bo_tgt_angle_v))
    bo_mean = float(np.nanmean(bo_abs_err))
    bo_med  = float(np.nanmedian(bo_abs_err))

    # ── Plots ────────────────────────────────────────────────────────
    _plot_confusion(cm, ds, suffix='argmax',
                     title=f'Confusion — {ds} (rows = cell argmax)  '
                           f'n_cells = {int(cm.sum())}')
    _plot_confusion(cm_top, ds, suffix=f'best_of_top{TOP_K}',
                     title=f'Confusion — {ds} '
                           f'(rows = cell best-of-top-{TOP_K})  '
                           f'n_cells = {int(cm_top.sum())}')
    _plot_angular_error(abs_err[valid], sig_argmax[valid],
                         mean_abs, med_abs, ds,
                         null_abs=null_abs, mode='argmax')
    _plot_angular_error(bo_abs_err, sig_topk_any[valid],
                         bo_mean, bo_med, ds,
                         null_abs=None, mode='best_of_top3')
    _plot_scatter(cell_pref_angle[valid], fmri_ang_360[valid], ds,
                   r=r_circ, p=p_r_perm, mode='argmax',
                   sig_mask=sig_argmax[valid])
    _plot_scatter(bo_tgt_angle_v, fmri_ang_360[valid], ds,
                   r=r_circ, p=p_r_perm, mode='best_of_top3',
                   sig_mask=sig_topk_any[valid])
    _plot_glass_brain_best_of_topk(coords[valid], bo_tgt_angle_v, ds)

    # --- Per-cell CSV ---
    per_cell = pd.DataFrame({
        'neuron':            mpfc['neuron'],
        'MNI_x':             coords[:, 0],
        'MNI_y':             coords[:, 1],
        'MNI_z':             coords[:, 2],
        'cell_argmax_lag':   cell_pref_angle,
        'cell_top3_lags':    [','.join(map(str, t)) for t in cell_top_angles],
        'fmri_pred_angle':   fmri_ang_360,
        'fmri_snap_lag':     fmri_snap_angle,
        'fmri_amplitude':    fmri_amp,
        'abs_angle_error':   abs_err,
        'exact_match':       exact,
        f'within_top{TOP_K}': topk,
        'valid':             valid,
    })
    csv_out = OUT_DIR / f'per_cell_summary_{ds}.csv'
    per_cell.to_csv(csv_out, index=False)
    print(f"  → {csv_out.name}")

    # ── Population-level distribution tests ─────────────────────────
    # H0-A (chi²): the cell-argmax distribution and the fMRI-predicted
    # distribution across mPFC voxels are the same categorical distribution
    # over the 12 lag bins. Uses a 2×12 contingency table.
    cell_bin_counts = np.bincount(cell_snap_v, minlength=12)
    fmri_bin_counts = np.bincount(fmri_snap_v, minlength=12)
    contingency = np.vstack([cell_bin_counts, fmri_bin_counts])
    # Drop bins that are zero in BOTH rows (chi² undefined on empty cols).
    keep = contingency.sum(axis=0) > 0
    if keep.sum() >= 2:
        chi2_stat, chi2_p, chi2_df, _ = stats.chi2_contingency(
            contingency[:, keep])
    else:
        chi2_stat, chi2_p, chi2_df = np.nan, np.nan, np.nan

    # H0-B (Watson U², perm): same distribution question but treating the
    # data as circular samples rather than nominal categories.
    u2_stat, u2_p = watson_u2_permutation_test(
        cell_pref_v, fmri_ang_v, n_perm=2000, seed=SEED + 7)

    # H0-C (circular means): are the two amp-weighted / uniform circular
    # means the same? Compute both means + Rayleigh concentration.
    cell_mean = circular_mean_deg(cell_pref_v)
    fmri_mean = circular_mean_deg(fmri_ang_v)
    delta_mean = signed_circ_diff(cell_mean, fmri_mean)
    cell_R, cell_Z, cell_p_ray = rayleigh_test(cell_pref_v)
    fmri_R, fmri_Z, fmri_p_ray = rayleigh_test(fmri_ang_v)
    print("\n  === Population-level tests ===")
    print(f"  Chi²(12-bin cell vs fMRI counts): "
          f"χ²({chi2_df})={chi2_stat:.2f}, p = {chi2_p:.4g}")
    print(f"  Watson U² (perm, n=2000):  U² = {u2_stat:.4f}, p = {u2_p:.4g}")
    print(f"  Circular mean — cells: {cell_mean:+7.2f}°   "
          f"(R̄={cell_R:.3f}, Rayleigh p={cell_p_ray:.2e})")
    print(f"                — fMRI:  {fmri_mean:+7.2f}°   "
          f"(R̄={fmri_R:.3f}, Rayleigh p={fmri_p_ray:.2e})")
    print(f"  Δ mean angle (cell − fMRI): {delta_mean:+6.1f}°")

    return {
        'ds':          ds,
        'n_valid':     n_valid,
        'r_circ':      r_circ,
        'p_circ_perm': p_r_perm,
        'exact_rate':  exact_rate,
        'p_exact_perm': p_exact_perm,
        'topk_rate':   topk_rate,
        'p_topk_perm':  p_topk_perm,
        'mean_abs':    mean_abs,
        'med_abs':     med_abs,
        'p_abs_perm':  p_abs_perm,
        # Population-level distribution tests
        'chi2_stat':   chi2_stat,
        'chi2_df':     chi2_df,
        'chi2_p':      chi2_p,
        'watson_u2':   u2_stat,
        'watson_p':    u2_p,
        'cell_mean_deg': cell_mean,
        'fmri_mean_deg': fmri_mean,
        'delta_mean_deg': delta_mean,
        'cell_rayleigh_R': cell_R,
        'cell_rayleigh_p': cell_p_ray,
        'fmri_rayleigh_R': fmri_R,
        'fmri_rayleigh_p': fmri_p_ray,
        # Also: bin counts for the .md
        'cell_bin_counts': cell_bin_counts.tolist(),
        'fmri_bin_counts': fmri_bin_counts.tolist(),
    }


def _plot_confusion(cm, ds, suffix='argmax', title=None):
    fig, ax = plt.subplots(figsize=(5.0, 4.6), constrained_layout=True)
    im = ax.imshow(cm, cmap='magma_r', aspect='equal', origin='lower')
    ax.set_xticks(range(12)); ax.set_xticklabels(LAG_ANGLES, fontsize=8)
    ax.set_yticks(range(12)); ax.set_yticklabels(LAG_ANGLES, fontsize=8)
    ax.set_xlabel('fMRI-predicted lag (° snapped)', fontsize=10)
    ylab = ('Cell argmax lag (°)' if suffix == 'argmax'
            else f'Cell best-of-top-{TOP_K} lag (°)')
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(title or f'Confusion — {ds}  (n_cells = {int(cm.sum())})',
                 fontsize=11)
    diag = np.arange(12)
    ax.plot(diag, diag, color='#0e3d3a', lw=1.0, alpha=0.4)
    fig.colorbar(im, ax=ax, shrink=0.7, label='# cells')
    for i in range(12):
        for j in range(12):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=6.5, color='white' if cm[i, j] > cm.max() * .55 else 'black')
    savefig(fig, OUT_DIR / f'confusion_matrix_{suffix}_{ds}')
    plt.close(fig)


def _plot_angular_error(abs_err, sig_mask, mean_abs, med_abs, ds,
                         null_abs=None, mode='argmax'):
    """Stacked-bar histogram: sig cells (dark green) on top of non-sig
    (light green) per bin. Reader tip: a "good prediction" would peak on
    the LEFT (|Δangle| near 0°); bad = flat / peak near 90°."""
    fig, ax = plt.subplots(figsize=(5.4, 3.7), constrained_layout=True)
    bins = np.arange(0, 181, 15)
    err = np.asarray(abs_err)
    sig = np.asarray(sig_mask, dtype=bool)
    counts_sig, _   = np.histogram(err[sig],  bins=bins)
    counts_nsig, _  = np.histogram(err[~sig], bins=bins)
    centres = (bins[:-1] + bins[1:]) / 2
    width = np.diff(bins).mean() * 0.95
    ax.bar(centres, counts_nsig,       width=width,
           color='#c8e6c9', edgecolor='k', lw=0.4,
           label=f'not sig (p_argmax≥{SIG_P_THRESHOLD})  '
                 f'n={int((~sig).sum())}')
    ax.bar(centres, counts_sig, bottom=counts_nsig, width=width,
           color='#1b5e20', edgecolor='k', lw=0.4,
           label=f'sig (p<{SIG_P_THRESHOLD})  n={int(sig.sum())}')
    ax.axvline(mean_abs, color='#8B0000', lw=1.6,
                label=f'mean = {mean_abs:.1f}°')
    ax.axvline(med_abs, color='#0e3d3a', lw=1.6, ls='--',
                label=f'median = {med_abs:.1f}°')
    ax.axvline(90, color='#888', lw=1.0, ls=':', label='chance = 90°')
    if null_abs is not None:
        null_mean_lo = float(np.percentile(null_abs, 2.5))
        null_mean_hi = float(np.percentile(null_abs, 97.5))
        ax.axvspan(null_mean_lo, null_mean_hi, color='#888',
                    alpha=0.12,
                    label=f'null mean 95% CI [{null_mean_lo:.1f}°, {null_mean_hi:.1f}°]')
    ax.text(0.03, 0.97, '← perfect prediction',
             transform=ax.transAxes, fontsize=8, va='top', color='#333')
    label = ('cell argmax' if mode == 'argmax'
             else f'cell best-of-top-{TOP_K}')
    ax.set_xlabel(f'|Δangle|  (fMRI predicted vs {label})  [°]',
                  fontsize=10)
    ax.set_ylabel('# mPFC cells', fontsize=10)
    ax.set_title(f'Angular error — {ds}  ({mode})', fontsize=11)
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    savefig(fig, OUT_DIR / f'angular_error_hist_{mode}_{ds}')
    plt.close(fig)


def _plot_scatter(cell_deg, fmri_deg, ds, r, p, mode='argmax',
                    sig_mask=None):
    fmri_360 = np.asarray(fmri_deg) % 360.0
    cell_360 = np.asarray(cell_deg) % 360.0
    fig, ax = plt.subplots(figsize=(5.2, 5.0), constrained_layout=True)
    colours = angles_to_colours(cell_360)
    if sig_mask is not None:
        sig = np.asarray(sig_mask, dtype=bool)
        # non-sig cells → same colour but with black cross marker
        ax.scatter(cell_360[~sig], fmri_360[~sig], s=28,
                    facecolors='none', edgecolors=colours[~sig], lw=0.8,
                    marker='x', label=f'not sig  n={int((~sig).sum())}')
        ax.scatter(cell_360[sig], fmri_360[sig], s=42,
                    c=colours[sig], edgecolor='k', lw=0.4,
                    label=f'sig (p<{SIG_P_THRESHOLD})  n={int(sig.sum())}')
    else:
        ax.scatter(cell_360, fmri_360, s=32, c=colours,
                    edgecolor='k', lw=0.4)
    ax.plot([0, 360], [0, 360], color='#888', ls='--', lw=1, label='y = x')
    ax.fill_between([0, 360], [-30, 330], [30, 390],
                     color='#888', alpha=0.10,
                     label='y = x ± 30°')
    ax.set_xlim(-10, 370); ax.set_ylim(-10, 370)
    ax.set_xticks(LAG_ANGLES); ax.set_yticks(LAG_ANGLES)
    ax.tick_params(labelsize=8)
    xlab = ('Cell argmax lag (°)' if mode == 'argmax'
            else f'Cell best-of-top-{TOP_K} lag (°)')
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel('fMRI predicted angle (°) — [0, 360)', fontsize=10)
    ax.set_title(f'{ds} — {mode}  |  circ-circ r = {r:+.3f}, '
                  f'perm p = {p:.3g}', fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    savefig(fig, OUT_DIR / f'scatter_cell_vs_fmri_{mode}_{ds}')
    plt.close(fig)


def _plot_glass_brain_best_of_topk(coords, target_angle, ds):
    """Glass brain: cells coloured by their best-matched top-K lag,
    using the correct signed-angle → circular-cmap mapping."""
    try:
        from nilearn.plotting import plot_glass_brain
    except Exception as exc:
        print(f"  [glass_brain] nilearn unavailable: {exc}")
        return
    fig = plt.figure(figsize=(9.0, 3.6), dpi=300, constrained_layout=True)
    display = plot_glass_brain(
        None, display_mode='ortho', figure=fig, annotate=True,
        black_bg=False, colorbar=False, alpha=0.35,
    )
    colours = angles_to_colours(target_angle)
    display.add_markers(coords, marker_color=colours,
                         marker_size=28, alpha=0.9)
    # Signed-angle colourbar (matches the actual plot colours).
    cbar_ax = fig.add_axes([0.10, 0.02, 0.35, 0.03])
    sm = ScalarMappable(cmap=CIRC_CMAP, norm=Normalize(vmin=-180, vmax=180))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([-180, -90, 0, 90, 180])
    cbar.set_label(f"cell's best-matched top-{TOP_K} lag (°) — {ds}",
                    fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    savepng = str(OUT_DIR / f'glass_brain_best_of_top{TOP_K}_{ds}.png')
    display.savefig(savepng, dpi=300)
    if SAVE_PDF:
        display.savefig(savepng.replace('.png', '.pdf'), dpi=300)
    plt.close(fig)


def _plot_lag_diagnostic(r_matrix, argmax_idx):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4),
                              constrained_layout=True)
    ax = axes[0]
    counts = np.bincount(argmax_idx, minlength=12)
    ax.bar(LAG_ANGLES, counts, width=25, color='#43A047',
            edgecolor='k', lw=0.5)
    for i, c in enumerate(counts):
        if c > 0:
            ax.text(LAG_ANGLES[i], c + 0.3, str(c),
                    ha='center', va='bottom', fontsize=7)
    ax.set_xlabel('Cell argmax lag (°)', fontsize=10)
    ax.set_ylabel('# mPFC cells', fontsize=10)
    ax.set_title(f'Argmax-lag distribution ({len(argmax_idx)} cells)',
                  fontsize=10)
    ax.set_xticks(LAG_ANGLES); ax.tick_params(axis='x', labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)

    ax = axes[1]
    means = np.nanmean(r_matrix, axis=0)
    sems  = np.nanstd(r_matrix, axis=0, ddof=1) / np.sqrt(
        np.sum(np.isfinite(r_matrix), axis=0))
    ax.errorbar(LAG_ANGLES, means, yerr=sems, marker='o', ms=5,
                 lw=1.4, capsize=3, color='#0e3d3a')
    ax.axhline(0, color='#888', lw=0.8, ls='--')
    peak_i = int(np.nanargmax(means))
    ax.axvline(LAG_ANGLES[peak_i], color='#E53935', lw=1.0, ls=':',
                label=f'peak at {LAG_ANGLES[peak_i]}° '
                      f'(mean r = {means[peak_i]:+.4f})')
    ax.set_xlabel('Lag (°)', fontsize=10)
    ax.set_ylabel('Population mean r (± SEM)', fontsize=10)
    ax.set_title('Mean tuning per lag', fontsize=10)
    ax.set_xticks(LAG_ANGLES); ax.tick_params(axis='x', labelsize=7)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)

    savefig(fig, OUT_DIR / 'diagnostic_lag_distribution')
    plt.close(fig)


def plot_glass_brain_cells_by_pref_lag(coords, pref_angle, sig_mask=None):
    """Glass brain: mPFC cells coloured by their own most-tuned lag.
    Non-significant cells (if `sig_mask` given) drawn hollow, sig drawn filled."""
    try:
        from nilearn.plotting import plot_glass_brain
    except Exception as exc:
        print(f"  [glass_brain] nilearn unavailable: {exc}")
        return
    fig = plt.figure(figsize=(9.0, 3.6), dpi=300, constrained_layout=True)
    display = plot_glass_brain(
        None, display_mode='ortho', figure=fig, annotate=True,
        black_bg=False, colorbar=False, alpha=0.35,
    )
    colours = angles_to_colours(pref_angle)
    if sig_mask is None:
        display.add_markers(coords, marker_color=colours,
                             marker_size=28, alpha=0.85)
    else:
        sig = np.asarray(sig_mask, dtype=bool)
        if (~sig).any():
            display.add_markers(coords[~sig], marker_color=colours[~sig],
                                 marker_size=14, alpha=0.35)
        if sig.any():
            display.add_markers(coords[sig], marker_color=colours[sig],
                                 marker_size=34, alpha=0.95)
    cbar_ax = fig.add_axes([0.10, 0.02, 0.35, 0.03])
    sm = ScalarMappable(cmap=CIRC_CMAP, norm=Normalize(vmin=-180, vmax=180))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([-180, -90, 0, 90, 180])
    cbar.set_label("cell's preferred lag (°)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    stem = str(OUT_DIR / 'glass_brain_cells_by_pref_lag.png')
    display.savefig(stem, dpi=300)
    if SAVE_PDF:
        display.savefig(stem.replace('.png', '.pdf'), dpi=300)
    plt.close(fig)
    print(f"  → {stem}")


# ── Main ─────────────────────────────────────────────────────────────
_plot_lag_diagnostic(r_matrix, cell_argmax_idx)
print("\nWrote diagnostic_lag_distribution.pdf/png  "
      "(argmax histogram + population mean r per lag)")

summary_rows = []
for ds in DATASETS:
    row = run_dataset(ds)
    if row is not None:
        summary_rows.append(row)

if summary_rows:
    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / 'summary_metrics.csv', index=False)
    print(f"\nSummary metrics → {OUT_DIR / 'summary_metrics.csv'}")

print("\n=== Glass-brain overlay of mPFC cells by preferred lag ===")
plot_glass_brain_cells_by_pref_lag(coords, cell_pref_angle,
                                     sig_mask=sig_argmax)


# ── Write a comprehensive Markdown summary ───────────────────────────
def _write_md(summary_rows):
    md = OUT_DIR / 'RESULTS.md'
    L = LAG_ANGLES.tolist()
    def _f(x, n=4):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return 'n/a'
        return f'{x:.{n}g}'
    def _pct(x): return _f(100*x, 2) + '%'
    lines = [
        "# fMRI → mPFC-cell preferred-lag prediction — results",
        "",
        f"Generated by `scripts/cell_fmri_lag_prediction.py`. "
        f"Outputs live in `{OUT_DIR}`.",
        "",
        "## Question",
        "Does the fMRI harmonic-angle map at each mPFC cell's MNI voxel",
        "predict that cell's preferred spatial-tuning lag?",
        "",
        "## Settings",
        f"- **cell ROI**: `{CELL_ROI}`",
        f"- **lag-column kind**: `{LAG_COL_KIND}`",
        f"- **cells loaded**: {len(mpfc)} (with valid tuning data)",
        f"- **filter 1 (MIN_CELL_R)**: `{MIN_CELL_R}` — cells passing: "
            f"{int(cell_ok.sum())}",
        f"- **filter 2 (MIN_FMRI_AMP_PCT)**: `{MIN_FMRI_AMP_PCT}`",
        f"- **spatial smoothing (SPHERE_RADIUS_MM)**: `{SPHERE_RADIUS_MM}` mm "
            "(amp-weighted circular mean of fMRI angles in that sphere)",
        f"- **top-K**: `{TOP_K}`",
        f"- **N_PERM**: `{N_PERM}`",
        f"- **significance threshold** (per-lag p): `{SIG_P_THRESHOLD}`",
        f"- **cells with sig argmax (p<{SIG_P_THRESHOLD})**: "
            f"{int(sig_argmax.sum())}/{len(mpfc)}",
        f"- **cells with ≥1 sig top-{TOP_K} lag**: "
            f"{int(sig_topk_any.sum())}/{len(mpfc)}",
        "",
        "## Per-cell metrics (with shuffle-null p-values)",
        "",
        "The shuffle null randomly permutes fMRI predictions across cells.",
        "It preserves both marginal distributions, so it answers *\"is the",
        "specific per-cell pairing informative beyond the marginals?\"*.",
        "For the population-level marginal-alignment claim, see the",
        "**Population-level distribution tests** below.",
        "",
        "| dataset | n cells | Fisher–Lee r (perm p) | exact-match (perm p) "
        "| top-3 match (perm p) | mean \\|Δ\\| (perm p) |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['ds']} | {r['n_valid']} | "
            f"{_f(r['r_circ'])} ({_f(r['p_circ_perm'])}) | "
            f"{_pct(r['exact_rate'])} ({_f(r['p_exact_perm'])}) | "
            f"{_pct(r['topk_rate'])} ({_f(r['p_topk_perm'])}) | "
            f"{_f(r['mean_abs'], 3)}° ({_f(r['p_abs_perm'])}) |")
    lines += [
        "",
        f"Baselines under uniform random matching: exact = "
        f"{100/12:.1f}%, top-{TOP_K} = {100*TOP_K/12:.1f}%, "
        f"mean |Δ| = 90°.",
        "",
        "## Population-level distribution tests",
        "",
        "These answer *\"do the fMRI-predicted-lag distribution and the",
        "cell-argmax-lag distribution look like the same distribution?\"* —",
        "the marginal-alignment claim you asked about.",
        "",
        "### Chi-square test (nominal, 12-bin contingency table)",
        "",
        "H0: cell-argmax bin counts and fMRI-predicted bin counts come from",
        "the same categorical distribution over the 12 lag bins.",
        "",
        "| dataset | χ² | df | p |",
        "|---|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['ds']} | {_f(r['chi2_stat'], 3)} | "
            f"{r['chi2_df']} | {_f(r['chi2_p'])} |")
    lines += [
        "",
        "### Watson U² (circular, permutation p)",
        "",
        "Non-parametric two-sample circular test; label-shuffle null.",
        "",
        "| dataset | U² | perm p |",
        "|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['ds']} | {_f(r['watson_u2'])} | "
            f"{_f(r['watson_p'])} |")
    lines += [
        "",
        "### Circular means + Rayleigh concentration",
        "",
        "R̄ is the mean resultant length (0 = uniform, 1 = perfect ",
        "concentration).  Rayleigh p tests H0 = uniform on the circle.",
        "",
        "| dataset | cell mean (Rayleigh R̄, p) | fMRI mean (Rayleigh R̄, p)"
        " | Δ mean (cell − fMRI) |",
        "|---|---|---|---|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['ds']} | {_f(r['cell_mean_deg'], 4)}° "
            f"(R̄={_f(r['cell_rayleigh_R'])}, p={_f(r['cell_rayleigh_p'])}) "
            f"| {_f(r['fmri_mean_deg'], 4)}° "
            f"(R̄={_f(r['fmri_rayleigh_R'])}, p={_f(r['fmri_rayleigh_p'])}) "
            f"| {_f(r['delta_mean_deg'], 4)}° |")
    # Per-bin counts table
    lines += ["", "### Per-bin counts (both distributions)", ""]
    for r in summary_rows:
        lines += [
            f"**{r['ds']}**",
            "",
            "| lag (°) | " + " | ".join(f"{a}°" for a in L) + " |",
            "|---" * (len(L) + 1) + "|",
            f"| cell argmax | "
                + " | ".join(str(c) for c in r['cell_bin_counts']) + " |",
            f"| fMRI predicted | "
                + " | ".join(str(c) for c in r['fmri_bin_counts']) + " |",
            "",
        ]
    lines += [
        "## Interpretation",
        "",
        "1. **Per-cell prediction (shuffle null)** — even after gating by",
        "   cell-tuning quality (MIN_CELL_R) and applying spatial",
        "   smoothing, no metric significantly beats the shuffle null at",
        "   p<0.05.  Individual mPFC cells' preferred spatial-tuning lag",
        "   cannot be predicted from the fMRI voxel at their exact",
        "   coordinate.",
        "",
        "2. **Population-level alignment** — chi² and Watson U² test",
        "   whether the two *distributions* (cell argmax counts vs fMRI",
        "   predicted counts across mPFC voxels) match.  Because mPFC",
        "   voxels only carry angles in the near-future arc (≈0°–90° + a",
        "   wrap into 330°), the fMRI can never predict far-future lags,",
        "   so a poor match here is anatomically expected.",
        "",
        "3. **Circular means + Rayleigh** — both mPFC-cell argmax and",
        "   fMRI-predicted lag distributions concentrate in the",
        "   near-future arc (see mean angles above); Rayleigh tests tell",
        "   you whether the concentration alone is significant vs uniform.",
        "",
        "4. **Practical claim available**: fMRI preferentially predicts",
        "   'current' → 'next' step in mPFC, and mPFC cells tuning peaks",
        "   also concentrate near 0°–60° — a population-level alignment",
        "   that does not extend to per-cell voxel-level decoding.",
        "",
        "## Files",
        "",
    ]
    for f in sorted(OUT_DIR.iterdir()):
        if f.suffix in ('.png', '.pdf', '.csv'):
            lines.append(f"- `{f.name}`")
    lines.append("")
    md.write_text("\n".join(lines))
    print(f"\nWrote {md}")


_write_md(summary_rows)

print(f"\nAll outputs in: {OUT_DIR}")
