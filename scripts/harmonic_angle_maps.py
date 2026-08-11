#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group-level "preferred future-step" angle map from per-subject β_std maps.

Follows the same DATASETS logic as
``future_step_dominance_mPFC_lOFC.py``.  For each dataset:

  1. Load the per-step 4-D per-subject β maps (one file per step).
  2. Per subject, per voxel, project onto the first Fourier harmonic
     across steps:
         cos_s(v) = Σ_k cos(θ_k) · β_{s,k}(v)
         sin_s(v) = Σ_k sin(θ_k) · β_{s,k}(v)
     where θ_k = 2π·k / n_steps.
     For 4 steps: cos weights [+1, 0, −1,  0]
                  sin weights [ 0,+1,  0, −1]
     For 8 steps: cos(2πk/8) / sin(2πk/8), k = 0..7.
  3. Group means across subjects give cos_G(v), sin_G(v).
  4. Amplitude(v)   = √(cos_G² + sin_G²)
     Angle(v) [rad] = arctan2(sin_G, cos_G)
     Angle in steps = (angle / 2π · n_steps) mod n_steps
  5. Hotelling T² one-sample test per voxel (H0: (cos, sin) = (0, 0))
     → F, p, −log10(p), plus a p<0.05 binary mask, and an angle map
     restricted to significant voxels for cleaner fsleyes display.

Outputs (into ``OUT_DIR/<dataset_label>/``):
  cos_persubj.nii.gz          — per-subject 4-D cos projection
  sin_persubj.nii.gz          — per-subject 4-D sin projection
  cos_group.nii.gz            — group-mean cos projection
  sin_group.nii.gz            — group-mean sin projection
  amplitude.nii.gz            — √(cos_G² + sin_G²)
  angle_rad.nii.gz            — preferred angle in radians, full map
  angle_steps.nii.gz          — preferred angle mapped to [0, n_steps)
  angle_rad_masked_p05.nii.gz — angle only where Hotelling p<0.05
  hotelling_F.nii.gz          — Hotelling F statistic
  hotelling_p.nii.gz          — Hotelling p value
  hotelling_neglog10p.nii.gz  — −log10(p) for fsleyes overlay
  hotelling_sig_p05.nii.gz    — binary Hotelling p<0.05 mask
  config.json                 — run settings
  README.md                   — quick fsleyes viewing notes

Quick fsleyes usage (per dataset folder):
    fsleyes $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz \\
        angle_rad.nii.gz          -cm hsv -dr -3.1416 3.1416 \\
        amplitude.nii.gz          -cm hot -dr 0 <cap> \\
        hotelling_neglog10p.nii.gz -cm red-yellow -dr 1.3 6 -a 60

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import f as f_dist, chi2 as chi2_dist

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ── Settings ─────────────────────────────────────────────────────────
# All per-subject 4-D β_std files live here, with the
# ``cropped_masked_smooth_fwhm5_`` prefix.
BASE_DIR = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group'
    '/group_RSA_DSR_quarters_except_prev_button_state'
    '_glmbase_all-paths-fixed_stickrews_split-buttons')

# Optional secondary ROI mask (mPFC). If the file is present, every dataset
# also writes ROI-restricted outputs and a BH-FDR-corrected Hotelling map
# within these voxels.
MPFC_MASK_PATH = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/masks'
    '/mask_PFC_LR_smoothed_resampled.nii.gz')

OUT_DIR = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
               '/derivatives/group/Main_Results_fMRI/harmonic_angle_maps')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── DATASETS ─────────────────────────────────────────────────────────
# All datasets share ``BASE_DIR`` and the ``cropped_masked_smooth_fwhm5_``
# filename prefix. Two 4-step variants of the quarters split (with vs.
# without ``_state`` in the GLM) are analysed separately.
DATASETS = [
    {
        'label':    'quarters',
        'base_dir': BASE_DIR,
        'files': {
            'current':  'cropped_masked_smooth_fwhm5_CURR_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            'next':     'cropped_masked_smooth_fwhm5_NEXT_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            'next +2':  'cropped_masked_smooth_fwhm5_NEXT2_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            'next +3':  'cropped_masked_smooth_fwhm5_NEXT3_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
        },
    },
    {
        # Same 4 steps, but the underlying GLM also regressed out state.
        'label':    'quarters_state',
        'base_dir': BASE_DIR,
        'files': {
            'current':  'cropped_masked_smooth_fwhm5_CURR_QUARTER-split_quarters_DSR_except_prev_button_state-mask_reward-path_beta_std.nii',
            'next':     'cropped_masked_smooth_fwhm5_NEXT_QUARTER-split_quarters_DSR_except_prev_button_state-mask_reward-path_beta_std.nii',
            'next +2':  'cropped_masked_smooth_fwhm5_NEXT2_QUARTER-split_quarters_DSR_except_prev_button_state-mask_reward-path_beta_std.nii',
            'next +3':  'cropped_masked_smooth_fwhm5_NEXT3_QUARTER-split_quarters_DSR_except_prev_button_state-mask_reward-path_beta_std.nii',
        },
    },
    {
        'label':    'rot_quarters',
        'base_dir': BASE_DIR,
        'files': {
            'current':  'cropped_masked_smooth_fwhm5_ROT_CURR_QUARTER-split_rot_quarters_DSR_except_prev_but-mask_reward-path_beta_std.nii',
            'next':     'cropped_masked_smooth_fwhm5_ROT_NEXT_QUARTER-split_rot_quarters_DSR_except_prev_but-mask_reward-path_beta_std.nii',
            'next +2':  'cropped_masked_smooth_fwhm5_ROT_NEXT2_QUARTER-split_rot_quarters_DSR_except_prev_but-mask_reward-path_beta_std.nii',
            'next +3':  'cropped_masked_smooth_fwhm5_ROT_NEXT3_QUARTER-split_rot_quarters_DSR_except_prev_but-mask_reward-path_beta_std.nii',
        },
    },
    {
        # ``LOCATION-split_eighths_*`` file is the zero-lag (now) map for
        # the 8-way split — see the eighths dataset in
        # future_step_dominance_mPFC_lOFC.py.
        'label':    'eighths',
        'base_dir': BASE_DIR,
        'files': {
            'now':    'cropped_masked_smooth_fwhm5_LOCATION-split_eighths_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            '+1 fut': 'cropped_masked_smooth_fwhm5_DSR_ONEFUT-split_eighths_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            '+2 fut': 'cropped_masked_smooth_fwhm5_DSR_TWOFUT-split_eighths_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            '+3 fut': 'cropped_masked_smooth_fwhm5_DSR_THREEFUT-split_eighths_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            '+4 fut': 'cropped_masked_smooth_fwhm5_DSR_FOURFUT-split_eighths_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            '+5 fut': 'cropped_masked_smooth_fwhm5_DSR_FIVEFUT-split_eighths_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            '+6 fut': 'cropped_masked_smooth_fwhm5_DSR_SIXFUT-split_eighths_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            '+7 fut': 'cropped_masked_smooth_fwhm5_DSR_SEVENFUT-split_eighths_DSR_except_prev_button-mask_reward-path_beta_std.nii',
        },
    },
]

DATASETS_TO_RUN = ['quarters', 'quarters_state', 'rot_quarters', 'eighths']


# ── Helpers ──────────────────────────────────────────────────────────
def _load_4d(fname, base_dir):
    """Load one per-step nifti; promote 3-D → 4-D so downstream code is uniform."""
    path = Path(base_dir) / fname
    if not path.is_file():
        raise FileNotFoundError(f"missing: {path}")
    img = nib.load(str(path))
    data = img.get_fdata()
    if data.ndim == 3:
        data = data[..., None]
    elif data.ndim != 4:
        raise ValueError(f"{fname}: expected 3-D or 4-D, got {data.shape}")
    return img, data


def _save_like(ref_img, arr, out_path):
    """Save `arr` as a nifti in the reference image's grid. All NaNs are
    replaced by 0 so fsleyes displays cleanly without holes."""
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0,
                        posinf=0.0, neginf=0.0)
    nib.save(nib.Nifti1Image(arr, ref_img.affine, ref_img.header),
             str(out_path))


def _load_roi_mask(path, ref_img):
    """Load an ROI mask and, if needed, resample nearest-neighbour onto
    the reference image's grid. Returns a bool array or None if the file
    doesn't exist."""
    if not Path(path).is_file():
        return None
    img = nib.load(str(path))
    if img.shape[:3] == ref_img.shape[:3] and np.allclose(
            img.affine, ref_img.affine, atol=1e-3):
        return img.get_fdata() > 0.5
    from nilearn.image import resample_img
    resampled = resample_img(img,
                             target_affine=ref_img.affine,
                             target_shape=ref_img.shape[:3],
                             interpolation='nearest')
    return resampled.get_fdata() > 0.5


def _rayleigh_voxelwise(cos_stack, sin_stack):
    """Per-voxel Rayleigh test of angle uniformity across subjects.

    For each voxel, computes each subject's angle
        θ_s = arctan2(sin_s, cos_s)
    then the mean resultant length
        R̄ = |mean_s(e^{iθ_s})|  ∈ [0, 1]
    and its Rayleigh p-value (Zar 2010, eq. 27.4):
        Z    = n · R̄²
        p    ≈ exp(√(1 + 4n + 4(n² − Z²)) − (1 + 2n))     (asymptotic)
    Complementary to Hotelling T²: Hotelling is sensitive to the group
    vector's MAGNITUDE (would flag a voxel where everyone has a big
    (cos, sin) but pointing in slightly different directions if the
    average still has decent length); Rayleigh is sensitive purely to
    ANGLE consistency (would flag a voxel where every subject has
    small (cos, sin) but they all point in the same direction).

    Returns
    -------
    R_mean, Z, p, neglog10p, valid : ndarrays of shape (X, Y, Z)
    """
    n_subj = cos_stack.shape[-1]
    theta = np.arctan2(sin_stack, cos_stack)          # (X,Y,Z,n_subj)
    finite = np.isfinite(theta)
    ok_all = finite.all(axis=-1)
    # Background gate: if EVERY subject has (cos, sin) ≈ (0, 0) then all
    # arctan2 values collapse to 0 and R̄ = 1 spuriously.  Require at
    # least one subject to carry non-zero signal at this voxel.
    mag_any = np.any(np.abs(cos_stack) + np.abs(sin_stack) > 0, axis=-1)
    with np.errstate(invalid='ignore'):
        C = np.nansum(np.cos(theta), axis=-1) / n_subj
        S = np.nansum(np.sin(theta), axis=-1) / n_subj
    R_mean = np.sqrt(C * C + S * S)
    Z = n_subj * R_mean * R_mean
    with np.errstate(invalid='ignore'):
        # Zar 2010 asymptotic approximation, accurate for n ≥ 8.
        pval = np.exp(np.sqrt(1 + 4 * n_subj + 4 * (n_subj * n_subj - Z * Z))
                      - (1 + 2 * n_subj))
    valid = ok_all & mag_any & np.isfinite(pval)
    pval = np.where(valid, pval, np.nan)
    R_mean = np.where(valid, R_mean, np.nan)
    Z = np.where(valid, Z, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        neglog10 = -np.log10(pval)
    neglog10 = np.where(np.isfinite(neglog10), neglog10, 0.0)
    return R_mean, Z, pval, neglog10, valid


def _fisher_combine(p_a, p_b):
    """Fisher's method: combine two per-voxel p-values into one χ²(4) test.

    X²(v) = −2 · [ln p_a(v) + ln p_b(v)],   p_comb(v) = 1 − F_{χ²(4)}(X²)

    Returns (chi2, pval, neglog10p) arrays of the same spatial shape.
    Voxels where either input is NaN or ≤0 come out as NaN.
    Fisher assumes independent p-values; here p_H and p_R come from the
    same voxel's (cos, sin) data, so they're not strictly independent —
    treat the combined p as an approximate summary rather than an exact
    joint test.
    """
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    valid = np.isfinite(p_a) & np.isfinite(p_b) & (p_a > 0) & (p_b > 0)
    p_a_c = np.clip(p_a, 1e-300, 1.0)
    p_b_c = np.clip(p_b, 1e-300, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = -2.0 * (np.log(p_a_c) + np.log(p_b_c))
    chi2 = np.where(valid, chi2, np.nan)
    with np.errstate(invalid='ignore'):
        pval = np.where(valid, 1.0 - chi2_dist.cdf(chi2, df=4), np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        neglog10 = -np.log10(pval)
    neglog10 = np.where(np.isfinite(neglog10), neglog10, 0.0)
    return chi2, pval, neglog10


def _bh_fdr(pvals_1d):
    """Benjamini–Hochberg q-values on a 1-D array. NaN input → NaN q."""
    p = np.asarray(pvals_1d, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pk = p[ok]
    order = np.argsort(pk)
    n = pk.size
    ranked = pk[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q_ok = np.empty(n)
    q_ok[order] = np.clip(ranked, 0, 1)
    q[ok] = q_ok
    return q


def _hotelling_t2_voxelwise(cos_stack, sin_stack):
    """Per-voxel one-sample Hotelling T² against (0, 0).

    Parameters
    ----------
    cos_stack, sin_stack : ndarray of shape (X, Y, Z, n_subj)

    Returns
    -------
    F, p, neglog10p : ndarray (X, Y, Z)
    valid           : bool ndarray (X, Y, Z) — voxels where the test ran
                      (finite in all subjects and non-degenerate cov).
    """
    n_subj = cos_stack.shape[-1]
    if n_subj < 4:
        raise ValueError(f"Hotelling T² needs n_subj > 2; got {n_subj}")

    # per-voxel means
    mx = np.nanmean(cos_stack, axis=-1)
    my = np.nanmean(sin_stack, axis=-1)

    # per-voxel covariance components (unbiased, ddof=1)
    dx = cos_stack - mx[..., None]
    dy = sin_stack - my[..., None]
    a = np.nanmean(dx * dx, axis=-1) * n_subj / (n_subj - 1)   # var(cos)
    c = np.nanmean(dy * dy, axis=-1) * n_subj / (n_subj - 1)   # var(sin)
    b = np.nanmean(dx * dy, axis=-1) * n_subj / (n_subj - 1)   # cov(cos, sin)

    det = a * c - b * b
    # analytic 2x2 inverse: inv = (1/det) * [[c, -b], [-b, a]]
    # T² = n * m' * inv * m = n / det * (c mx² - 2 b mx my + a my²)
    with np.errstate(divide='ignore', invalid='ignore'):
        T2 = n_subj * (c * mx * mx - 2 * b * mx * my + a * my * my) / det
    valid = np.isfinite(T2) & (det > 0)
    p = 2
    F = ((n_subj - p) / (p * (n_subj - 1))) * T2
    F = np.where(valid, F, np.nan)
    with np.errstate(invalid='ignore'):
        pval = np.where(valid, 1.0 - f_dist.cdf(F, p, n_subj - p), np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        neglog10 = -np.log10(pval)
    neglog10 = np.where(np.isfinite(neglog10), neglog10, 0.0)
    return F, pval, neglog10, valid


def run_dataset(cfg):
    label    = cfg['label']
    base_dir = Path(cfg['base_dir'])
    file_map = cfg['files']
    steps    = list(file_map.keys())
    n_steps  = len(steps)
    out_dir  = OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}\n### DATASET: {label}  "
          f"({n_steps} steps)  base={base_dir.name}\n{'#'*70}")

    # ── Load all step files, sanity-check shapes ────────────────────
    ref_img = None
    step_data = {}
    for step, fname in file_map.items():
        try:
            img, data = _load_4d(fname, base_dir)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}  — skipping this step")
            return None
        if ref_img is None:
            ref_img = img
            spatial = data.shape[:3]
            n_subj = data.shape[3]
        else:
            if data.shape[:3] != spatial:
                raise ValueError(
                    f"{fname}: spatial shape {data.shape[:3]} vs {spatial}")
            if data.shape[3] != n_subj:
                raise ValueError(
                    f"{fname}: n_subj {data.shape[3]} vs {n_subj}")
        step_data[step] = data
        print(f"  loaded  {step:8s}  {fname}  shape={data.shape}")

    # ── Build Fourier weights ───────────────────────────────────────
    theta = 2 * np.pi * np.arange(n_steps) / n_steps
    cos_w = np.cos(theta)   # e.g. 4-step:  [+1, 0, −1,  0]
    sin_w = np.sin(theta)   # e.g. 4-step:  [ 0,+1,  0, −1]
    print(f"\n  θ_k       = {np.round(theta, 3)}")
    print(f"  cos_w[k]  = {np.round(cos_w, 3)}")
    print(f"  sin_w[k]  = {np.round(sin_w, 3)}")

    # ── Per-subject cos / sin projections (vectorised) ──────────────
    # Stack step maps into a single (X, Y, Z, n_subj, n_steps) array
    # then contract over the last axis with the weight vector.
    stack = np.stack([step_data[s] for s in steps], axis=-1)   # (X,Y,Z,n_subj,n_steps)
    cos_persubj = np.tensordot(stack, cos_w, axes=([-1], [0]))  # (X,Y,Z,n_subj)
    sin_persubj = np.tensordot(stack, sin_w, axes=([-1], [0]))
    print(f"\n  cos/sin per-subject stacks built  shape={cos_persubj.shape}")

    _save_like(ref_img, cos_persubj, out_dir / 'cos_persubj.nii.gz')
    _save_like(ref_img, sin_persubj, out_dir / 'sin_persubj.nii.gz')

    # ── Group means, amplitude, angle ───────────────────────────────
    cos_G = np.nanmean(cos_persubj, axis=-1)
    sin_G = np.nanmean(sin_persubj, axis=-1)
    amplitude = np.sqrt(cos_G * cos_G + sin_G * sin_G)
    angle_rad = np.arctan2(sin_G, cos_G)                       # (−π, π]
    # map to [0, n_steps): first shift wrap into [0, 2π) then scale
    angle_steps = ((angle_rad + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi) * n_steps

    # Degrees are the human-friendly version: multiply by 180/π. Both
    # signed (−180, 180] and 0..360 wraps are saved so you can pick
    # whichever is easier to reason about in fsleyes.
    angle_deg      = np.degrees(angle_rad)
    angle_deg_0360 = np.mod(angle_deg, 360.0)
    # "Yellow-at-zero" version: with a red-yellow colormap in fsleyes
    # (dark red = low, bright yellow = high), this makes 0° render yellow
    # and ±180° render red, symmetric across the wrap. Sign is lost:
    # +90° and −90° both map to 90.
    angle_deg_y0   = 180.0 - np.abs(angle_deg)

    _save_like(ref_img, cos_G,          out_dir / 'cos_group.nii.gz')
    _save_like(ref_img, sin_G,          out_dir / 'sin_group.nii.gz')
    _save_like(ref_img, amplitude,      out_dir / 'amplitude.nii.gz')
    _save_like(ref_img, angle_rad,      out_dir / 'angle_rad.nii.gz')
    _save_like(ref_img, angle_deg,      out_dir / 'angle_deg.nii.gz')
    _save_like(ref_img, angle_deg_0360, out_dir / 'angle_deg_0to360.nii.gz')
    _save_like(ref_img, angle_deg_y0,   out_dir / 'angle_deg_yellow0.nii.gz')
    _save_like(ref_img, angle_steps,    out_dir / 'angle_steps.nii.gz')

    # ── Voxel-wise Hotelling T² ─────────────────────────────────────
    F, pval, neglog10p, valid = _hotelling_t2_voxelwise(
        cos_persubj, sin_persubj)
    sig_mask = (pval < 0.05).astype(np.uint8)
    angle_rad_masked = np.where(sig_mask.astype(bool), angle_rad, np.nan)

    angle_deg_masked    = np.where(sig_mask.astype(bool), angle_deg,    np.nan)
    angle_deg_y0_masked = np.where(sig_mask.astype(bool), angle_deg_y0, np.nan)

    _save_like(ref_img, F,                   out_dir / 'hotelling_F.nii.gz')
    _save_like(ref_img, pval,                out_dir / 'hotelling_p.nii.gz')
    _save_like(ref_img, neglog10p,           out_dir / 'hotelling_neglog10p.nii.gz')
    _save_like(ref_img, sig_mask,            out_dir / 'hotelling_sig_p05.nii.gz')
    _save_like(ref_img, angle_rad_masked,    out_dir / 'angle_rad_masked_p05.nii.gz')
    _save_like(ref_img, angle_deg_masked,    out_dir / 'angle_deg_masked_p05.nii.gz')
    _save_like(ref_img, angle_deg_y0_masked, out_dir / 'angle_deg_yellow0_masked_p05.nii.gz')

    n_vox_valid = int(np.count_nonzero(valid))
    n_vox_sig   = int(np.count_nonzero(sig_mask))
    print(f"\n  Hotelling T² (2 dof):  {n_vox_valid} voxels tested, "
          f"{n_vox_sig} sig at p<0.05 (uncorrected)")

    # ── Rayleigh test on per-subject angles ─────────────────────────
    R_mean, Z_ray, ray_p, ray_neglog10p, ray_valid = _rayleigh_voxelwise(
        cos_persubj, sin_persubj)
    ray_sig_mask = (ray_p < 0.05).astype(np.uint8)
    _save_like(ref_img, R_mean,          out_dir / 'rayleigh_R.nii.gz')
    _save_like(ref_img, Z_ray,           out_dir / 'rayleigh_Z.nii.gz')
    _save_like(ref_img, ray_p,           out_dir / 'rayleigh_p.nii.gz')
    _save_like(ref_img, ray_neglog10p,   out_dir / 'rayleigh_neglog10p.nii.gz')
    _save_like(ref_img, ray_sig_mask,    out_dir / 'rayleigh_sig_p05.nii.gz')

    n_ray_sig = int(np.count_nonzero(ray_sig_mask))
    print(f"  Rayleigh (angle consistency): "
          f"{n_ray_sig} voxels sig at p<0.05 (uncorrected)")

    # ── Combined tests: intersect / union / Fisher ──────────────────
    # Intersection = "both Hotelling AND Rayleigh flagged this voxel"
    # (high-confidence, strict). Union = "either test flagged it"
    # (lenient, high sensitivity). Fisher = one combined χ²(4) test
    # statistic from the two per-voxel p-values.
    combined_intersect = ((pval < 0.05) & (ray_p < 0.05)).astype(np.uint8)
    combined_union     = ((pval < 0.05) | (ray_p < 0.05)).astype(np.uint8)
    fisher_chi2, fisher_p, fisher_neglog10p = _fisher_combine(pval, ray_p)
    fisher_sig = (fisher_p < 0.05).astype(np.uint8)

    _save_like(ref_img, combined_intersect,   out_dir / 'combined_intersect_p05.nii.gz')
    _save_like(ref_img, combined_union,       out_dir / 'combined_union_p05.nii.gz')
    _save_like(ref_img, fisher_chi2,          out_dir / 'combined_fisher_chi2.nii.gz')
    _save_like(ref_img, fisher_p,             out_dir / 'combined_fisher_p.nii.gz')
    _save_like(ref_img, fisher_neglog10p,     out_dir / 'combined_fisher_neglog10p.nii.gz')
    _save_like(ref_img, fisher_sig,           out_dir / 'combined_fisher_sig_p05.nii.gz')

    # Angle maps gated by the intersection and by Fisher.
    angle_deg_intersect_masked    = np.where(combined_intersect.astype(bool),
                                              angle_deg, np.nan)
    angle_deg_y0_intersect_masked = np.where(combined_intersect.astype(bool),
                                              angle_deg_y0, np.nan)
    angle_deg_fisher_masked       = np.where(fisher_sig.astype(bool),
                                              angle_deg, np.nan)
    angle_deg_y0_fisher_masked    = np.where(fisher_sig.astype(bool),
                                              angle_deg_y0, np.nan)
    _save_like(ref_img, angle_deg_intersect_masked,
               out_dir / 'angle_deg_masked_intersect_p05.nii.gz')
    _save_like(ref_img, angle_deg_y0_intersect_masked,
               out_dir / 'angle_deg_yellow0_masked_intersect_p05.nii.gz')
    _save_like(ref_img, angle_deg_fisher_masked,
               out_dir / 'angle_deg_masked_fisher_p05.nii.gz')
    _save_like(ref_img, angle_deg_y0_fisher_masked,
               out_dir / 'angle_deg_yellow0_masked_fisher_p05.nii.gz')

    n_intersect = int(np.count_nonzero(combined_intersect))
    n_union     = int(np.count_nonzero(combined_union))
    n_fisher    = int(np.count_nonzero(fisher_sig))
    print(f"  Combined (Hotelling AND Rayleigh, both p<.05):  {n_intersect} vox")
    print(f"  Combined (Hotelling OR Rayleigh, either p<.05): {n_union} vox")
    print(f"  Fisher-combined p<.05 (uncorrected):            {n_fisher} vox")

    # ── mPFC-restricted outputs + within-mask BH-FDR ────────────────
    # We recompute nothing — just zero-out (or NaN-out for the angle map)
    # everything outside the mPFC mask, and BH-correct the Hotelling
    # p-values within the mask so the "significant voxels in mPFC" story
    # gets a properly small multiple-comparisons family.
    mpfc_bool = _load_roi_mask(MPFC_MASK_PATH, ref_img)
    mpfc_stats = None
    if mpfc_bool is None:
        print(f"\n  [mPFC] mask not found at {MPFC_MASK_PATH} — "
              "skipping ROI-restricted outputs")
    else:
        n_vox_mask = int(mpfc_bool.sum())
        # BH-FDR within the mask, on Hotelling voxels that were valid.
        pv_flat = pval[mpfc_bool]
        q_flat  = _bh_fdr(pv_flat)
        q_map   = np.full(pval.shape, np.nan, dtype=np.float32)
        q_map[mpfc_bool] = q_flat
        sig_q05        = np.where(mpfc_bool, (q_map < 0.05), False).astype(np.uint8)
        sig_p05_mpfc   = np.where(mpfc_bool, (pval < 0.05), False).astype(np.uint8)

        # Spatially restricted display maps: zero outside mask for numeric
        # maps, NaN for angle maps so fsleyes' HSV colormap ignores them.
        amp_mpfc         = np.where(mpfc_bool, amplitude, 0.0)
        F_mpfc           = np.where(mpfc_bool, F,          0.0)
        pval_mpfc        = np.where(mpfc_bool, pval,       np.nan)
        neglog10p_mpfc   = np.where(mpfc_bool, neglog10p,  0.0)
        angle_rad_mpfc          = np.where(mpfc_bool, angle_rad,    np.nan)
        angle_deg_mpfc          = np.where(mpfc_bool, angle_deg,    np.nan)
        angle_deg_y0_mpfc       = np.where(mpfc_bool, angle_deg_y0, np.nan)
        angle_p05_mpfc          = np.where(sig_p05_mpfc.astype(bool),
                                            angle_rad,    np.nan)
        angle_deg_p05_mpfc      = np.where(sig_p05_mpfc.astype(bool),
                                            angle_deg,    np.nan)
        angle_deg_y0_p05_mpfc   = np.where(sig_p05_mpfc.astype(bool),
                                            angle_deg_y0, np.nan)
        angle_qfdr_mpfc         = np.where(sig_q05.astype(bool),
                                            angle_rad,    np.nan)
        angle_deg_qfdr_mpfc     = np.where(sig_q05.astype(bool),
                                            angle_deg,    np.nan)
        angle_deg_y0_qfdr_mpfc  = np.where(sig_q05.astype(bool),
                                            angle_deg_y0, np.nan)

        _save_like(ref_img, amp_mpfc,               out_dir / 'amplitude_mPFC.nii.gz')
        _save_like(ref_img, angle_rad_mpfc,         out_dir / 'angle_rad_mPFC.nii.gz')
        _save_like(ref_img, angle_deg_mpfc,         out_dir / 'angle_deg_mPFC.nii.gz')
        _save_like(ref_img, angle_deg_y0_mpfc,      out_dir / 'angle_deg_yellow0_mPFC.nii.gz')
        _save_like(ref_img, angle_p05_mpfc,         out_dir / 'angle_rad_mPFC_masked_p05.nii.gz')
        _save_like(ref_img, angle_deg_p05_mpfc,     out_dir / 'angle_deg_mPFC_masked_p05.nii.gz')
        _save_like(ref_img, angle_deg_y0_p05_mpfc,  out_dir / 'angle_deg_yellow0_mPFC_masked_p05.nii.gz')
        _save_like(ref_img, angle_qfdr_mpfc,        out_dir / 'angle_rad_mPFC_masked_qFDR05.nii.gz')
        _save_like(ref_img, angle_deg_qfdr_mpfc,    out_dir / 'angle_deg_mPFC_masked_qFDR05.nii.gz')
        _save_like(ref_img, angle_deg_y0_qfdr_mpfc, out_dir / 'angle_deg_yellow0_mPFC_masked_qFDR05.nii.gz')
        _save_like(ref_img, F_mpfc,          out_dir / 'hotelling_F_mPFC.nii.gz')
        _save_like(ref_img, pval_mpfc,       out_dir / 'hotelling_p_mPFC.nii.gz')
        _save_like(ref_img, neglog10p_mpfc,  out_dir / 'hotelling_neglog10p_mPFC.nii.gz')
        _save_like(ref_img, sig_p05_mpfc,    out_dir / 'hotelling_sig_p05_mPFC.nii.gz')
        _save_like(ref_img, q_map,           out_dir / 'hotelling_qFDR_mPFC.nii.gz')
        _save_like(ref_img, sig_q05,         out_dir / 'hotelling_sig_qFDR05_mPFC.nii.gz')

        n_sig_p05_mpfc = int(sig_p05_mpfc.sum())
        n_sig_q05_mpfc = int(sig_q05.sum())

        # Rayleigh mPFC-restricted + within-mask BH-FDR
        ray_pv_flat = ray_p[mpfc_bool]
        ray_q_flat  = _bh_fdr(ray_pv_flat)
        ray_q_map   = np.full(ray_p.shape, np.nan, dtype=np.float32)
        ray_q_map[mpfc_bool] = ray_q_flat
        ray_sig_p05_mpfc = np.where(mpfc_bool, (ray_p < 0.05), False).astype(np.uint8)
        ray_sig_q05      = np.where(mpfc_bool, (ray_q_map < 0.05),
                                    False).astype(np.uint8)
        R_mpfc         = np.where(mpfc_bool, R_mean,        0.0)
        ray_neglog_mpfc = np.where(mpfc_bool, ray_neglog10p, 0.0)
        _save_like(ref_img, R_mpfc,           out_dir / 'rayleigh_R_mPFC.nii.gz')
        _save_like(ref_img, ray_neglog_mpfc,  out_dir / 'rayleigh_neglog10p_mPFC.nii.gz')
        _save_like(ref_img, ray_sig_p05_mpfc, out_dir / 'rayleigh_sig_p05_mPFC.nii.gz')
        _save_like(ref_img, ray_q_map,        out_dir / 'rayleigh_qFDR_mPFC.nii.gz')
        _save_like(ref_img, ray_sig_q05,      out_dir / 'rayleigh_sig_qFDR05_mPFC.nii.gz')
        # Rayleigh-gated angle maps in the mPFC
        angle_deg_ray_p05_mpfc  = np.where(ray_sig_p05_mpfc.astype(bool),
                                             angle_deg, np.nan)
        angle_deg_ray_qfdr_mpfc = np.where(ray_sig_q05.astype(bool),
                                             angle_deg, np.nan)
        _save_like(ref_img, angle_deg_ray_p05_mpfc,
                   out_dir / 'angle_deg_mPFC_masked_rayleigh_p05.nii.gz')
        _save_like(ref_img, angle_deg_ray_qfdr_mpfc,
                   out_dir / 'angle_deg_mPFC_masked_rayleigh_qFDR05.nii.gz')

        n_ray_sig_p05_mpfc = int(ray_sig_p05_mpfc.sum())
        n_ray_sig_q05_mpfc = int(ray_sig_q05.sum())

        # Combined tests within the mPFC mask + within-mask BH-FDR on
        # the Fisher-combined χ²(4) p-values.
        combined_intersect_mpfc = np.where(
            mpfc_bool, (pval < 0.05) & (ray_p < 0.05), False).astype(np.uint8)
        combined_union_mpfc     = np.where(
            mpfc_bool, (pval < 0.05) | (ray_p < 0.05), False).astype(np.uint8)
        fisher_pv_flat  = fisher_p[mpfc_bool]
        fisher_q_flat   = _bh_fdr(fisher_pv_flat)
        fisher_q_map    = np.full(fisher_p.shape, np.nan, dtype=np.float32)
        fisher_q_map[mpfc_bool] = fisher_q_flat
        fisher_sig_p05_mpfc = np.where(mpfc_bool,
                                        (fisher_p < 0.05), False).astype(np.uint8)
        fisher_sig_q05_mpfc = np.where(mpfc_bool,
                                        (fisher_q_map < 0.05), False).astype(np.uint8)
        fisher_neglog_mpfc  = np.where(mpfc_bool, fisher_neglog10p, 0.0)

        _save_like(ref_img, combined_intersect_mpfc,
                   out_dir / 'combined_intersect_p05_mPFC.nii.gz')
        _save_like(ref_img, combined_union_mpfc,
                   out_dir / 'combined_union_p05_mPFC.nii.gz')
        _save_like(ref_img, fisher_neglog_mpfc,
                   out_dir / 'combined_fisher_neglog10p_mPFC.nii.gz')
        _save_like(ref_img, fisher_sig_p05_mpfc,
                   out_dir / 'combined_fisher_sig_p05_mPFC.nii.gz')
        _save_like(ref_img, fisher_q_map,
                   out_dir / 'combined_fisher_qFDR_mPFC.nii.gz')
        _save_like(ref_img, fisher_sig_q05_mpfc,
                   out_dir / 'combined_fisher_sig_qFDR05_mPFC.nii.gz')

        # Angle maps gated by intersection / Fisher within mPFC
        ang_int_mpfc        = np.where(combined_intersect_mpfc.astype(bool),
                                        angle_deg,    np.nan)
        ang_int_y0_mpfc     = np.where(combined_intersect_mpfc.astype(bool),
                                        angle_deg_y0, np.nan)
        ang_fisher_p05_mpfc = np.where(fisher_sig_p05_mpfc.astype(bool),
                                        angle_deg,    np.nan)
        ang_fisher_p05_y0_mpfc = np.where(fisher_sig_p05_mpfc.astype(bool),
                                           angle_deg_y0, np.nan)
        ang_fisher_qfdr_mpfc = np.where(fisher_sig_q05_mpfc.astype(bool),
                                         angle_deg,    np.nan)
        ang_fisher_qfdr_y0_mpfc = np.where(fisher_sig_q05_mpfc.astype(bool),
                                            angle_deg_y0, np.nan)
        _save_like(ref_img, ang_int_mpfc,
                   out_dir / 'angle_deg_mPFC_masked_intersect_p05.nii.gz')
        _save_like(ref_img, ang_int_y0_mpfc,
                   out_dir / 'angle_deg_yellow0_mPFC_masked_intersect_p05.nii.gz')
        _save_like(ref_img, ang_fisher_p05_mpfc,
                   out_dir / 'angle_deg_mPFC_masked_fisher_p05.nii.gz')
        _save_like(ref_img, ang_fisher_p05_y0_mpfc,
                   out_dir / 'angle_deg_yellow0_mPFC_masked_fisher_p05.nii.gz')
        _save_like(ref_img, ang_fisher_qfdr_mpfc,
                   out_dir / 'angle_deg_mPFC_masked_fisher_qFDR05.nii.gz')
        _save_like(ref_img, ang_fisher_qfdr_y0_mpfc,
                   out_dir / 'angle_deg_yellow0_mPFC_masked_fisher_qFDR05.nii.gz')

        n_intersect_mpfc = int(combined_intersect_mpfc.sum())
        n_union_mpfc     = int(combined_union_mpfc.sum())
        n_fisher_p05_mpfc = int(fisher_sig_p05_mpfc.sum())
        n_fisher_q05_mpfc = int(fisher_sig_q05_mpfc.sum())

        mpfc_stats = {
            'mask_path':                    str(MPFC_MASK_PATH),
            'n_vox_in_mask':                n_vox_mask,
            'hotelling_n_sig_p05_uncorr':   n_sig_p05_mpfc,
            'hotelling_n_sig_qFDR05':       n_sig_q05_mpfc,
            'rayleigh_n_sig_p05_uncorr':    n_ray_sig_p05_mpfc,
            'rayleigh_n_sig_qFDR05':        n_ray_sig_q05_mpfc,
            'combined_intersect_p05':       n_intersect_mpfc,
            'combined_union_p05':           n_union_mpfc,
            'fisher_n_sig_p05_uncorr':      n_fisher_p05_mpfc,
            'fisher_n_sig_qFDR05':          n_fisher_q05_mpfc,
        }
        print(f"\n  [mPFC] {n_vox_mask} voxels in mask;  "
              f"Hotelling {n_sig_p05_mpfc} sig at p<0.05 uncorr, "
              f"{n_sig_q05_mpfc} sig at q<0.05 BH-FDR")
        print(f"           Rayleigh   {n_ray_sig_p05_mpfc} sig at p<0.05 uncorr, "
              f"{n_ray_sig_q05_mpfc} sig at q<0.05 BH-FDR")
        print(f"           Intersect  {n_intersect_mpfc} (both p<.05),  "
              f"Union {n_union_mpfc} (either p<.05)")
        print(f"           Fisher     {n_fisher_p05_mpfc} sig at p<0.05 uncorr, "
              f"{n_fisher_q05_mpfc} sig at q<0.05 BH-FDR")

    # ── Config + README ─────────────────────────────────────────────
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({
            'label':     label,
            'base_dir':  str(base_dir),
            'files':     file_map,
            'steps':     steps,
            'n_steps':   n_steps,
            'n_subj':    n_subj,
            'theta':     [float(x) for x in theta],
            'cos_w':     [float(x) for x in cos_w],
            'sin_w':     [float(x) for x in sin_w],
            'n_vox_valid':                    n_vox_valid,
            'hotelling_n_sig_p05_uncorr':     n_vox_sig,
            'rayleigh_n_sig_p05_uncorr':      n_ray_sig,
            'combined_intersect_p05':         n_intersect,
            'combined_union_p05':             n_union,
            'fisher_n_sig_p05_uncorr':        n_fisher,
            'mpfc':                            mpfc_stats,
        }, f, indent=2)

    _write_readme(out_dir, label, n_steps, n_subj)
    print(f"\n  → outputs in {out_dir}")
    return out_dir


def _write_readme(out_dir, label, n_steps, n_subj):
    p = out_dir / 'README.md'
    step_examples = "\n".join(
        f"- `angle_steps ≈ {k}` → preferred step = index {k}  "
        f"(≈ {int(round(360.0 * k / n_steps))}° in `angle_deg_0to360`)"
        for k in range(n_steps)
    )
    lines = [
        f"# harmonic angle maps — {label}",
        "",
        f"- n_steps = {n_steps}",
        f"- n_subj  = {n_subj}",
        "",
        "## Whole-brain files",
        "- `cos_persubj.nii.gz`, `sin_persubj.nii.gz` — per-subject 4-D",
        "- `cos_group.nii.gz`, `sin_group.nii.gz` — group-mean projections",
        "- `amplitude.nii.gz` — √(cos_G² + sin_G²), the length of the group mean vector",
        "- `angle_rad.nii.gz` — arctan2(sin_G, cos_G), range (−π, π]",
        "- `angle_deg.nii.gz` — same angle in degrees, range (−180, 180]",
        "- `angle_deg_0to360.nii.gz` — degrees wrapped into [0, 360)",
        "- `angle_deg_yellow0.nii.gz` — 180 − |angle_deg|, range [0, 180].",
        "  Display with `-cm red-yellow -dr 0 180`: 0° renders yellow,",
        "  ±180° renders red. Sign lost (+90° and −90° both map to 90).",
        f"- `angle_steps.nii.gz` — angle mapped to [0, {n_steps})",
        "- `angle_{rad,deg}_masked_p05.nii.gz` — angle only where Hotelling p<0.05",
        "- `hotelling_{F,p,neglog10p,sig_p05}.nii.gz`",
        "- `rayleigh_{R,Z,p,neglog10p,sig_p05}.nii.gz` — per-voxel angle-consistency test",
        "- `combined_intersect_p05.nii.gz` — voxels where BOTH Hotelling AND Rayleigh p<.05",
        "- `combined_union_p05.nii.gz` — voxels where EITHER Hotelling OR Rayleigh p<.05",
        "- `combined_fisher_{chi2,p,neglog10p,sig_p05}.nii.gz` — Fisher's method",
        "  combining Hotelling p and Rayleigh p into one χ²(4) test",
        "- `angle_deg_masked_{intersect,fisher}_p05.nii.gz` — angle gated by",
        "  the intersection or Fisher-combined p<.05",
        "",
        "## mPFC-restricted files (only voxels inside the PFC mask)",
        "- `amplitude_mPFC.nii.gz`, `angle_{rad,deg}_mPFC.nii.gz`, `hotelling_{F,p,neglog10p,sig_p05}_mPFC.nii.gz`",
        "- `hotelling_qFDR_mPFC.nii.gz`, `hotelling_sig_qFDR05_mPFC.nii.gz` — BH-FDR within the mask",
        "- `angle_{rad,deg}_mPFC_masked_p05.nii.gz` — angle where Hotelling p<0.05 within mask",
        "- `angle_{rad,deg}_mPFC_masked_qFDR05.nii.gz` — angle where Hotelling q<0.05 within mask",
        "- `rayleigh_{R,neglog10p,sig_p05,qFDR,sig_qFDR05}_mPFC.nii.gz` — Rayleigh test within mask",
        "- `angle_deg_mPFC_masked_rayleigh_p05.nii.gz` — angle where Rayleigh p<0.05 within mask",
        "- `angle_deg_mPFC_masked_rayleigh_qFDR05.nii.gz` — angle where Rayleigh q<0.05 within mask",
        "- `combined_intersect_p05_mPFC.nii.gz` — both Hotelling AND Rayleigh p<.05 within mask",
        "- `combined_union_p05_mPFC.nii.gz` — either p<.05 within mask",
        "- `combined_fisher_{neglog10p,sig_p05,qFDR,sig_qFDR05}_mPFC.nii.gz` —",
        "  Fisher χ²(4) combined test + within-mask BH-FDR",
        "- `angle_deg_mPFC_masked_{intersect,fisher}_{p05,qFDR05}.nii.gz` —",
        "  angle gated by the intersection or Fisher-combined thresholds within mask",
        "",
        "## fsleyes recipe (whole brain, degrees)",
        "```bash",
        "fsleyes $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz \\",
        "    angle_deg.nii.gz            -cm hsv -dr -180 180 \\",
        "    amplitude.nii.gz            -cm hot -dr 0 <cap> \\",
        "    hotelling_neglog10p.nii.gz  -cm red-yellow -dr 1.3 6 -a 60",
        "```",
        "",
        "## fsleyes recipe (yellow-at-zero, red-yellow colormap)",
        "```bash",
        "fsleyes $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz \\",
        "    angle_deg_yellow0_masked_p05.nii.gz  -cm red-yellow -dr 0 180",
        "```",
        "Yellow = voxel prefers `current` (0°); red = voxel prefers the",
        "opposite step (±180°); intermediate = mixed. Sign of the angle",
        "is discarded (so `next` and `+3` look the same).",
        "Use `angle_deg_masked_p05.nii.gz` in place of `angle_deg.nii.gz`",
        "to hide sub-threshold voxels.",
        "",
        "## fsleyes recipe (mPFC only, degrees)",
        "```bash",
        "fsleyes $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz \\",
        "    angle_deg_mPFC_masked_qFDR05.nii.gz  -cm hsv -dr -180 180 \\",
        "    hotelling_neglog10p_mPFC.nii.gz      -cm red-yellow -dr 1.3 6 -a 60",
        "```",
        "",
        "## Interpretation",
        f"For {n_steps} steps at angles θ_k = 2πk/{n_steps}:",
        "",
        step_examples,
        "",
        "- **Amplitude** — length of the group-mean (cos, sin) vector; big",
        "  → the voxel's β-across-steps has a clean sinusoidal shape.",
        "  Small → either flat or noisy across steps.",
        "- **Hotelling T² p / q** — is the group-mean 2-D vector significantly",
        "  different from (0, 0)?  Sensitive to vector MAGNITUDE.  Use for",
        "  \"does this voxel have any harmonic signal at all\".",
        "- **Rayleigh p / q** — do subjects agree on the ANGLE at this voxel,",
        "  regardless of magnitude?  Sensitive purely to angle consistency.",
        "  A voxel where every subject has a small but same-direction (cos, sin)",
        "  will fail Hotelling but pass Rayleigh.  Complementary — inspect both.",
        "- **Intersection / Union** — logical AND / OR of Hotelling & Rayleigh",
        "  binary masks at p<.05.  Intersection = strict high-confidence set,",
        "  Union = lenient exploratory set.",
        "- **Fisher χ²(4) combined p** — one summary p-value per voxel from",
        "  Fisher's method on the two per-voxel p's.  Use `combined_fisher_qFDR_mPFC`",
        "  for a single BH-FDR-corrected number within the mPFC family.",
        "- **Angle** — *which* step the voxel prefers.  Radians in",
        "  `angle_rad*.nii.gz`, degrees in `angle_deg*.nii.gz`.  Only",
        "  meaningful at voxels that pass one of the tests above.",
        "",
        "Two voxels with the same angle but different amplitudes both",
        "prefer the same step; the higher-amplitude one just shows it",
        "more cleanly.",
    ]
    p.write_text('\n'.join(lines))


# ── Main entry ───────────────────────────────────────────────────────
if __name__ == '__main__':
    for cfg in DATASETS:
        if cfg['label'] not in DATASETS_TO_RUN:
            print(f"[skip] dataset '{cfg['label']}' not in DATASETS_TO_RUN")
            continue
        run_dataset(cfg)
    print(f"\nAll dataset outputs under: {OUT_DIR}")
