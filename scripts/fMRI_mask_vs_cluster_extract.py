#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI-mean extraction (and mask-vs-cluster validation) for the DSR and
STATE group-level fMRI maps.

This produces the ANATOMICAL-ROI half of the two-step fMRI strategy: a
directional, hypothesis-driven test inside a predefined mask, reported
alongside the whole-brain / small-volume PALM searchlight result. It
answers two questions per (effect, mask):

  (a) What is the group effect averaged over the whole mask?
      Every voxel in the mask contributes equally, so the estimate does
      not depend on where the searchlight peak happened to fall. This is
      what the paper reports as "we averaged each participant's beta in
      an anatomically predefined mask" — NOT a peak-voxel readout. (The
      earlier get_subj_effects_fMRI.py did extract single peak voxels,
      which is circular when the peak is chosen from the same data.)

  (b) Does the mask actually cover what PALM found significant?
      We locate the peak inside the mask and the peak of the best-
      overlapping significant cluster, and report the distance between
      them. A small distance means the anatomical ROI and the data-driven
      cluster are pointing at the same thing.

EFFECTS AND MASKS
-----------------
  DSR   (concurrent action plan)  -> lOFC, mPFC
  STATE (position in sequence)    -> vmPFC, EC

Note the naming: the mask file used for the STATE `vmPFC` region is the
one the manuscript refers to as the mOFC mask.

WHAT IT DOES, per (effect, mask)
--------------------------------
1. Resample the mask onto the subject 4D grid (nearest-neighbour,
   binarised) if the grids differ.

2. Split the mask into bilateral / left / right by MNI x (x = 0 midline
   excluded from both unilateral masks), and pick a `matched`
   hemisphere automatically:
     - if any significant PALM cluster overlaps either hemisphere, take
       the hemisphere with the larger overlap (`matched_basis =
       cluster_overlap`);
     - otherwise take the hemisphere with the higher within-mask peak t
       (`matched_basis = peak_t`).
   NB `matched` is chosen using the data, so it is a descriptive
   convenience for plotting — quote `bilateral` (or a hemisphere fixed a
   priori) when reporting an inferential test.

3. Peak within the mask, from the PALM voxelwise t map where one is
   supplied, and from our own voxelwise one-sample t recomputed from the
   subject 4D as a cross-check (`*_own_t` columns).

4. Peak of each significant cluster and its overlap with the mask.
   Clusters come from the PALM cluster-mass FWE map thresholded at
   1 − p > 0.95; for STATE, which has no such map at whole-brain level,
   any nonzero voxel of the cluster-mass t map marks cluster membership.
   Euclidean MNI distance between mask-peak and cluster-peak is reported.

5. Small-volume correction report (STATE only): how many voxels inside
   each anatomical mask survive the PFC+MTL voxel-FWE PALM correction,
   and the MNI + p_FWE of the best one. This is where the left-EC
   headline number comes from.

6. Per-subject mean within the mask (and within each hemisphere
   variant), one-sample t-test against 0. NOTE: the printed and stored
   p-values are TWO-SIDED; the manuscript reports one-sided p for these
   directional hypotheses, i.e. p_two / 2.

OUTPUTS
-------
  data/derivatives/group/Main_Results_fMRI/mask_vs_cluster_extract/
      mask_vs_cluster_summary.csv   one row per (effect, region), with
                                    bilateral / left / right / matched
                                    columns and the SVC report
      per_subject_mean_in_mask.csv  long form, one row per
                                    (effect, region, variant, subject) —
                                    this is what the figures plot
      details.json                  full provenance: input maps, mask
                                    paths, cluster overlaps, per-subject
                                    vectors

Run:
    python scripts/fMRI_mask_vs_cluster_extract.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img, load_img
from scipy.stats import ttest_1samp
from scipy.ndimage import label


# ── Paths (sync these with get_subj_effects_fMRI.py if anything moves) ──
DATA_ROOT = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives')
FMRI_BASE = DATA_ROOT / 'group/Main_Results_fMRI'
MASKS = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/masks')

# DSR effect
DSR_SUBJ_4D = (
    FMRI_BASE
    / 'group_RSA_quarters_DSR_controls_glmbase_all-paths-fixed_stickrews_split-buttons_cropped'
    / 'cropped_masked_smooth_fwhm5_DSR-DSR-contr_except_prev_but-mask_reward-path_beta_std.nii'
)
DSR_PALM_DIR = (
    FMRI_BASE
    / 'RSA_quarters_DSR_controls_glmbase_all-paths-fixed_stickrews_split-buttons_smooth5_palm_p0_01'
)
DSR_CLUSTER_FWEP_MAP = (
    DSR_PALM_DIR
    / 'cropped_masked_smooth_fwhm5_DSR-DSR-contr_except_prev_but-mask_reward-path_beta_std_clusterm_tstat_fwep_c1.nii'
)
DSR_VOX_TSTAT_MAP = (
    DSR_PALM_DIR
    / 'cropped_masked_smooth_fwhm5_DSR-DSR-contr_except_prev_but-mask_reward-path_beta_std_vox_tstat_c1.nii'
)

# STATE effect
STATE_DIR = FMRI_BASE / 'Final_state'
STATE_SUBJ_4D = STATE_DIR / 'STATE-DSR_all_controls_noint.nii'
STATE_CLUSTER_FWEP_MAP = STATE_DIR / 'STATE-DSR_all_controls_noint_clusterm_tstat_fwep_c1.nii'
STATE_CLUSTERM_TSTAT_MAP = STATE_DIR / 'STATE-DSR_all_controls_noint_clusterm_tstat_c1.nii'
# Primary inference is a PFC + MTL small-volume voxel-FWE PALM result.
# Use the SVC vox-tstat map as the within-mask peak statistic so the
# reported peak numbers line up with the PALM headline figure
# (left EC peak at MNI (-20, 2, -36), p_FWE_SVC = 0.039).
STATE_SVC_DIR = STATE_DIR / 'PFC-and-MonaMTL'
STATE_SVC_VOX_TSTAT_MAP = STATE_SVC_DIR / 'STATE-DSR_all_controls_noint_vox_tstat_c1.nii'
STATE_SVC_VOX_FWEP_MAP  = STATE_SVC_DIR / 'STATE-DSR_all_controls_noint_vox_tstat_fwep_c1.nii'

# Masks
MASK_PATHS = {
    'lOFC':  MASKS / 'lOFC.nii.gz',
    'mPFC':  MASKS / 'mask_PFC_LR_smoothed_resampled.nii.gz',
    'vmPFC': MASKS / 'vMPFC.nii.gz',
    'EC':    MASKS / 'mask_entorhinal_all_33_subs.nii',
}

OUT_DIR = FMRI_BASE / 'mask_vs_cluster_extract'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# PALM's cluster_*_fwep map stores ``1 - p`` so significant voxels are
# CLOSE TO 1; threshold at 0.95 ⇔ p_FWE < 0.05.
FWE_THRESH = 0.95


# ── Helpers ─────────────────────────────────────────────────────────────
def resample_to_subj_grid(mask_path, subj_img):
    """Load mask and resample onto subj_img's grid as a binary mask."""
    m = load_img(str(mask_path))
    if m.shape[:3] == subj_img.shape[:3] and np.allclose(m.affine, subj_img.affine):
        bool_data = np.asarray(m.get_fdata(), dtype=float) > 0
    else:
        m_rs = resample_to_img(m, subj_img, interpolation='nearest')
        bool_data = np.asarray(m_rs.get_fdata(), dtype=float) > 0
    return bool_data


def split_hemispheres(mask_bool, affine):
    """Return three boolean masks: bilateral / left / right.
    Splitting is by MNI x: x < 0 = left, x > 0 = right, x = 0 midline excluded
    from both unilateral masks.
    """
    n = mask_bool.size
    out_bil = mask_bool.copy()
    out_L = np.zeros_like(mask_bool)
    out_R = np.zeros_like(mask_bool)
    ii, jj, kk = np.where(mask_bool)
    if ii.size == 0:
        return {'bilateral': out_bil, 'left': out_L, 'right': out_R}
    coords = nib.affines.apply_affine(
        affine, np.stack([ii, jj, kk], axis=1).astype(float))
    xs = coords[:, 0]
    is_left  = xs < 0
    is_right = xs > 0
    out_L[ii[is_left],  jj[is_left],  kk[is_left]]  = True
    out_R[ii[is_right], jj[is_right], kk[is_right]] = True
    return {'bilateral': out_bil, 'left': out_L, 'right': out_R}


def peak_in_mask(stat_3d, mask_bool, affine):
    """Return MNI + voxel index + value of the max-stat voxel inside mask.
    Returns None if mask empty OR every in-mask voxel is non-finite."""
    if not mask_bool.any():
        return None
    s = np.asarray(stat_3d, dtype=float)
    masked = np.where(mask_bool & np.isfinite(s), s, -np.inf)
    if not np.isfinite(masked).any():
        return None
    idx = np.unravel_index(np.argmax(masked), masked.shape)
    if not np.isfinite(masked[idx]):
        return None
    mni = nib.affines.apply_affine(affine, np.asarray(idx, dtype=float))
    return {
        'voxel_idx': tuple(int(x) for x in idx),
        'mni':       tuple(float(x) for x in mni),
        'value':     float(s[idx]),
        'n_mask_voxels': int(mask_bool.sum()),
    }


def largest_clusters(cluster_map_3d, threshold, n_top=2):
    """Threshold + label + return masks of the top-N clusters by size."""
    sig = cluster_map_3d > threshold
    if not sig.any():
        return []
    labeled, n_found = label(sig)
    sizes = np.array([(labeled == cid).sum() for cid in range(1, n_found + 1)])
    top = (np.argsort(sizes)[::-1] + 1)[:n_top]
    return [
        {'cluster_id': int(cid),
         'mask': (labeled == cid),
         'n_voxels': int((labeled == cid).sum())}
        for cid in top
    ]


def voxelwise_t_from_4d(subj_4d, axis=-1):
    """One-sample t-stat at every voxel from a 4D (X,Y,Z,N) subject array.
    NaN-safe; voxels with <2 finite subjects return NaN."""
    arr = np.asarray(subj_4d, dtype=float)
    mean = np.nanmean(arr, axis=axis)
    std = np.nanstd(arr, axis=axis, ddof=1)
    n = np.sum(np.isfinite(arr), axis=axis)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where((n >= 2) & (std > 0), mean / (std / np.sqrt(n)), np.nan)
    return t, mean, std, n


def mni_distance(a, b):
    if a is None or b is None:
        return float('nan')
    return float(np.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b))))


def process_effect(label_, subj_4d_path, mask_specs,
                   cluster_fwep_map=None, vox_tstat_map=None,
                   cluster_localizer_map=None,
                   svc_vox_fwep_map=None,
                   subj_data_cache=None):
    """Run the mask + cluster extraction for one effect (DSR or STATE).

    mask_specs: list of (region_name, mask_path).
    cluster_fwep_map: PALM 1−p cluster map for thresholding (FWE 0.95).
    vox_tstat_map: optional pre-computed vox t-stat map; if None, compute
                   from the subject 4D file (one-sample t per voxel).
    cluster_localizer_map: alternative to cluster_fwep_map for finding
                   significant clusters (e.g. cluster-mass t map for state,
                   where any nonzero voxel = part of a sig cluster).
    svc_vox_fwep_map: optional PALM voxel-FWE 1−p map from a small-volume
                   correction (e.g. PFC+MTL for state). If given, we report
                   how many SVC-FWE-significant voxels fall inside each
                   anatomical mask + the peak SVC voxel's MNI / 1-p.
    """
    subj_img = nib.load(str(subj_4d_path))
    if subj_data_cache is None:
        subj_data_cache = {}
    if subj_4d_path not in subj_data_cache:
        subj_data_cache[subj_4d_path] = subj_img.get_fdata()
    subj_4d = subj_data_cache[subj_4d_path]
    affine = subj_img.affine
    print(f"\n========================================================================")
    print(f"  EFFECT = {label_}  (subj 4D: {Path(subj_4d_path).name}, shape={subj_4d.shape})")
    print(f"========================================================================")

    # Vox-wise one-sample t from the subject 4D (always our 'main_tstat'
    # for the within-mask peak search — independent of PALM's exact map).
    t_4d, mean_3d, sd_3d, n_3d = voxelwise_t_from_4d(subj_4d)
    print(f"  computed voxelwise one-sample t from 4D: shape {t_4d.shape}, "
          f"finite voxels = {int(np.isfinite(t_4d).sum())}")

    # If the user provided a pre-computed PALM vox tstat, use that for
    # the cluster-peak (apples-to-apples with previous publication).
    palm_t = (load_img(str(vox_tstat_map)).get_fdata()
              if vox_tstat_map is not None else t_4d)

    # Find significant clusters (only meaningful if a thresholdable map
    # was provided).
    if cluster_fwep_map is not None:
        cmap = load_img(str(cluster_fwep_map)).get_fdata()
        clusters = largest_clusters(cmap, threshold=FWE_THRESH, n_top=4)
        print(f"  PALM cluster-FWEP map @ 1−p>{FWE_THRESH}: "
              f"{len(clusters)} significant cluster(s) found "
              f"(top sizes: {[c['n_voxels'] for c in clusters]})")
    elif cluster_localizer_map is not None:
        cmap = load_img(str(cluster_localizer_map)).get_fdata()
        # Any nonzero voxel = inside a significant cluster
        clusters = largest_clusters(cmap, threshold=1e-9, n_top=4)
        print(f"  cluster-localiser map (cluster-mass t, nonzero): "
              f"{len(clusters)} cluster(s) "
              f"(top sizes: {[c['n_voxels'] for c in clusters]})")
    else:
        clusters = []

    # SVC voxel-FWE map (e.g. state PFC+MTL) — for the per-mask SVC report.
    svc_fwep = (load_img(str(svc_vox_fwep_map)).get_fdata()
                if svc_vox_fwep_map is not None else None)
    if svc_fwep is not None:
        svc_sig = svc_fwep >= FWE_THRESH
        print(f"  SVC voxel-FWEP map: {int(svc_sig.sum())} voxel(s) "
              f"survive (1-p ≥ {FWE_THRESH})")
    else:
        svc_sig = None

    rows = []
    persub_rows = []
    details_per_mask = {}

    for region_name, mask_path in mask_specs:
        print(f"\n  --- region: {region_name} (mask: {Path(mask_path).name}) ---")
        mask_bool_full = resample_to_subj_grid(mask_path, subj_img)
        if not mask_bool_full.any():
            print(f"    mask empty after resample, skipping.")
            continue
        hemi_masks = split_hemispheres(mask_bool_full, affine)
        # Pick the "matched" hemisphere automatically: whichever hemisphere
        # has the largest overlap with ANY significant PALM cluster. If no
        # cluster overlaps either hemisphere (state/vmPFC case), pick the
        # hemisphere with the higher within-mask peak t.
        hemi_choice_basis = {}
        for hemi_lbl in ('left', 'right'):
            m = hemi_masks[hemi_lbl]
            overlap_total = sum(int((c['mask'] & m).sum()) for c in clusters)
            pk = peak_in_mask(palm_t, m, affine)
            hemi_choice_basis[hemi_lbl] = {
                'cluster_overlap_total': overlap_total,
                'peak_t':                pk['value'] if pk else -np.inf,
            }
        # Decision rule:
        if (hemi_choice_basis['left']['cluster_overlap_total'] > 0
                or hemi_choice_basis['right']['cluster_overlap_total'] > 0):
            matched_hemi = max(
                ('left', 'right'),
                key=lambda h: hemi_choice_basis[h]['cluster_overlap_total'])
            matched_basis = 'cluster_overlap'
        else:
            matched_hemi = max(
                ('left', 'right'),
                key=lambda h: hemi_choice_basis[h]['peak_t'])
            matched_basis = 'peak_t'
        print(f"    hemisphere selection: matched = {matched_hemi}  "
              f"(basis = {matched_basis})")
        print(f"      L: cluster_overlap={hemi_choice_basis['left']['cluster_overlap_total']:>5d}  "
              f"peak_t={hemi_choice_basis['left']['peak_t']:+.3f}")
        print(f"      R: cluster_overlap={hemi_choice_basis['right']['cluster_overlap_total']:>5d}  "
              f"peak_t={hemi_choice_basis['right']['peak_t']:+.3f}")

        # Iterate over all three mask variants (bilateral / left / right /
        # plus the auto-matched copy so the figure code can just read
        # variant == 'matched'). Stats and extraction logic is the same.
        variants = {
            'bilateral': mask_bool_full,
            'left':      hemi_masks['left'],
            'right':     hemi_masks['right'],
            'matched':   hemi_masks[matched_hemi],
        }
        cluster_in_mask_overlap = []
        nearest_cluster_peak = None
        nearest_cluster_overlap_vox = 0
        # Use the BILATERAL mask for the cluster-overlap loop (so the
        # cluster-peak comparison reflects the published anatomical region
        # as a whole, not just one hemisphere).
        for c in clusters:
            overlap = int((c['mask'] & mask_bool_full).sum())
            peak_c = peak_in_mask(palm_t, c['mask'], affine)
            cluster_in_mask_overlap.append({
                'cluster_id':       c['cluster_id'],
                'cluster_n_voxels': c['n_voxels'],
                'overlap_voxels':   overlap,
                'overlap_frac_of_mask':    overlap / int(mask_bool_full.sum()),
                'overlap_frac_of_cluster': overlap / c['n_voxels'],
                'cluster_peak_mni':   peak_c['mni'] if peak_c else None,
                'cluster_peak_t':     peak_c['value'] if peak_c else None,
            })
            if overlap > nearest_cluster_overlap_vox:
                nearest_cluster_overlap_vox = overlap
                nearest_cluster_peak = peak_c

        # Bilateral peaks (existing behaviour, used as the headline numbers)
        mask_bool = mask_bool_full
        n_mask = int(mask_bool.sum())
        peak_mask = peak_in_mask(palm_t, mask_bool, affine)
        peak_mask_own = peak_in_mask(t_4d, mask_bool, affine)

        dist_mask_to_cluster_peak = mni_distance(
            peak_mask['mni'] if peak_mask else None,
            nearest_cluster_peak['mni'] if nearest_cluster_peak else None,
        )

        # SVC voxel-FWE survivors inside this mask (peak voxel + count).
        svc_in_mask = None
        if svc_fwep is not None:
            mask_and_sig = mask_bool_full & svc_sig
            n_svc = int(mask_and_sig.sum())
            peak_svc = peak_in_mask(svc_fwep, mask_and_sig, affine) if n_svc else None
            svc_in_mask = {
                'n_svc_sig_voxels_in_mask': n_svc,
                'peak_svc_mni':             peak_svc['mni'] if peak_svc else None,
                'peak_svc_oneminus_p':      peak_svc['value'] if peak_svc else None,
                'peak_svc_p_fwe':           (1.0 - peak_svc['value']) if peak_svc else None,
            }
            if peak_svc:
                print(f"    SVC voxel-FWE in mask: {n_svc} sig voxel(s); peak "
                      f"MNI=({peak_svc['mni'][0]:+5.1f}, "
                      f"{peak_svc['mni'][1]:+5.1f}, "
                      f"{peak_svc['mni'][2]:+5.1f})  "
                      f"p_FWE_SVC={1.0 - peak_svc['value']:.4g}")
            else:
                print(f"    SVC voxel-FWE in mask: 0 sig voxels")

        # Per-subject mean within the mask
        ii, jj, kk = np.where(mask_bool)
        per_subj = np.nanmean(subj_4d[ii, jj, kk, :], axis=0)
        valid = per_subj[np.isfinite(per_subj)]
        if valid.size >= 2:
            t_stat, p_two = ttest_1samp(valid, 0.0)
        else:
            t_stat, p_two = float('nan'), float('nan')

        # Print
        if peak_mask:
            print(f"    peak within mask  (PALM t):  "
                  f"MNI=({peak_mask['mni'][0]:+5.1f}, "
                  f"{peak_mask['mni'][1]:+5.1f}, "
                  f"{peak_mask['mni'][2]:+5.1f})  t={peak_mask['value']:+.3f}")
        else:
            print(f"    peak within mask  (PALM t):  no finite voxel in mask")
        if peak_mask_own:
            print(f"    peak within mask  (own  t):  "
                  f"MNI=({peak_mask_own['mni'][0]:+5.1f}, "
                  f"{peak_mask_own['mni'][1]:+5.1f}, "
                  f"{peak_mask_own['mni'][2]:+5.1f})  t={peak_mask_own['value']:+.3f}")
        else:
            print(f"    peak within mask  (own  t):  no finite voxel in mask")
        if nearest_cluster_peak:
            print(f"    peak of best-overlap cluster: "
                  f"MNI=({nearest_cluster_peak['mni'][0]:+5.1f}, "
                  f"{nearest_cluster_peak['mni'][1]:+5.1f}, "
                  f"{nearest_cluster_peak['mni'][2]:+5.1f})  "
                  f"t={nearest_cluster_peak['value']:+.3f}  "
                  f"(overlap = {nearest_cluster_overlap_vox} vox; "
                  f"= {100 * nearest_cluster_overlap_vox / n_mask:.1f}% of mask)")
            print(f"    distance(mask-peak, cluster-peak) = "
                  f"{dist_mask_to_cluster_peak:.1f} mm")
        else:
            print(f"    no significant cluster overlapped this mask.")
        print(f"    mean-within-mask per-subject t-test:  "
              f"n={valid.size}  mean={float(np.nanmean(valid)):+.4f}  "
              f"t={t_stat:+.2f}  p={p_two:.4f}")

        # Per-hemisphere extractions (peak + per-subject mean)
        hemi_results = {}
        for variant_name, vmask in variants.items():
            n_vox = int(vmask.sum())
            if n_vox == 0:
                hemi_results[variant_name] = None
                continue
            pk = peak_in_mask(palm_t, vmask, affine)
            ii, jj, kk = np.where(vmask)
            per_subj_v = np.nanmean(subj_4d[ii, jj, kk, :], axis=0)
            valid_v = per_subj_v[np.isfinite(per_subj_v)]
            if valid_v.size >= 2:
                t_v, p_v = ttest_1samp(valid_v, 0.0)
            else:
                t_v, p_v = float('nan'), float('nan')
            hemi_results[variant_name] = {
                'n_voxels':    n_vox,
                'peak_mni':    pk['mni'] if pk else None,
                'peak_t':      pk['value'] if pk else None,
                'mean':        float(np.nanmean(valid_v)) if valid_v.size else float('nan'),
                'sd':          float(np.nanstd(valid_v, ddof=1)) if valid_v.size > 1 else float('nan'),
                't':           float(t_v) if np.isfinite(t_v) else float('nan'),
                'p_two':       float(p_v) if np.isfinite(p_v) else float('nan'),
                'per_subject': [float(x) if np.isfinite(x) else None for x in per_subj_v],
            }
            tag = ('★' if np.isfinite(p_v) and p_v < 0.05 else '·' if np.isfinite(p_v) and p_v < 0.10 else '')
            print(f"    [{variant_name:>9s}] n_vox={n_vox:>5d}  peak_t={(pk['value'] if pk else float('nan')):>+5.2f}  "
                  f"mean={hemi_results[variant_name]['mean']:>+7.4f}  "
                  f"t={hemi_results[variant_name]['t']:>+5.2f}  p={hemi_results[variant_name]['p_two']:.4f}{tag}")

        row = {
            'effect':                  label_,
            'region':                  region_name,
            'mask_file':               str(mask_path),
            'matched_hemisphere':      matched_hemi,
            'matched_basis':           matched_basis,
            'n_mask_voxels':           n_mask,
            'mask_peak_palm_mni_x':    peak_mask['mni'][0] if peak_mask else np.nan,
            'mask_peak_palm_mni_y':    peak_mask['mni'][1] if peak_mask else np.nan,
            'mask_peak_palm_mni_z':    peak_mask['mni'][2] if peak_mask else np.nan,
            'mask_peak_palm_t':        peak_mask['value'] if peak_mask else np.nan,
            'mask_peak_own_mni_x':     peak_mask_own['mni'][0] if peak_mask_own else np.nan,
            'mask_peak_own_mni_y':     peak_mask_own['mni'][1] if peak_mask_own else np.nan,
            'mask_peak_own_mni_z':     peak_mask_own['mni'][2] if peak_mask_own else np.nan,
            'mask_peak_own_t':         peak_mask_own['value'] if peak_mask_own else np.nan,
            'nearest_cluster_overlap_voxels': nearest_cluster_overlap_vox,
            'nearest_cluster_peak_mni_x':     nearest_cluster_peak['mni'][0] if nearest_cluster_peak else np.nan,
            'nearest_cluster_peak_mni_y':     nearest_cluster_peak['mni'][1] if nearest_cluster_peak else np.nan,
            'nearest_cluster_peak_mni_z':     nearest_cluster_peak['mni'][2] if nearest_cluster_peak else np.nan,
            'nearest_cluster_peak_t':         nearest_cluster_peak['value'] if nearest_cluster_peak else np.nan,
            'dist_mask_to_cluster_peak_mm':   dist_mask_to_cluster_peak,
            'n_subjects':              int(valid.size),
            'mean_within_mask':        float(np.nanmean(valid)) if valid.size else np.nan,
            'sd_within_mask':          float(np.nanstd(valid, ddof=1)) if valid.size > 1 else np.nan,
            't_within_mask':           float(t_stat) if np.isfinite(t_stat) else np.nan,
            'p_within_mask_two_sided': float(p_two) if np.isfinite(p_two) else np.nan,
            'svc_sig_voxels_in_mask':  (svc_in_mask['n_svc_sig_voxels_in_mask']
                                        if svc_in_mask else None),
            'svc_peak_mni_x':          (svc_in_mask['peak_svc_mni'][0]
                                        if (svc_in_mask and svc_in_mask['peak_svc_mni']) else None),
            'svc_peak_mni_y':          (svc_in_mask['peak_svc_mni'][1]
                                        if (svc_in_mask and svc_in_mask['peak_svc_mni']) else None),
            'svc_peak_mni_z':          (svc_in_mask['peak_svc_mni'][2]
                                        if (svc_in_mask and svc_in_mask['peak_svc_mni']) else None),
            'svc_peak_p_fwe':          (svc_in_mask['peak_svc_p_fwe']
                                        if svc_in_mask else None),
        }
        # Add per-hemisphere variants to the row for the summary CSV.
        for variant_name in ('bilateral', 'left', 'right', 'matched'):
            hr = hemi_results.get(variant_name)
            if hr is None:
                continue
            for k in ('n_voxels', 'peak_t', 'mean', 'sd', 't', 'p_two'):
                row[f'{variant_name}_{k}'] = hr[k]
            if hr['peak_mni']:
                row[f'{variant_name}_peak_mni_x'] = hr['peak_mni'][0]
                row[f'{variant_name}_peak_mni_y'] = hr['peak_mni'][1]
                row[f'{variant_name}_peak_mni_z'] = hr['peak_mni'][2]
        rows.append(row)

        # Per-subject vectors for figure plotting later (one row per
        # (effect, region, variant, subject) — long-form, easy to filter).
        for variant_name in ('bilateral', 'left', 'right', 'matched'):
            hr = hemi_results.get(variant_name)
            if hr is None:
                continue
            for s_idx, v in enumerate(hr['per_subject']):
                persub_rows.append({
                    'effect':    label_,
                    'region':    region_name,
                    'variant':   variant_name,
                    'subject':   s_idx,
                    'mean_in_mask': float(v) if (v is not None and np.isfinite(v)) else np.nan,
                })

        details_per_mask[region_name] = {
            'mask_path':                str(mask_path),
            'n_mask_voxels':            n_mask,
            'matched_hemisphere':       matched_hemi,
            'matched_basis':            matched_basis,
            'peak_within_mask_palm_t':  peak_mask,
            'peak_within_mask_own_t':   peak_mask_own,
            'clusters_overlap':         cluster_in_mask_overlap,
            'nearest_cluster_peak':     nearest_cluster_peak,
            'dist_mask_to_cluster_peak_mm': dist_mask_to_cluster_peak,
            'per_subject_mean_in_mask': [
                float(x) if np.isfinite(x) else None for x in per_subj
            ],
            'mean_within_mask_stats':   {
                'n':       int(valid.size),
                'mean':    float(np.nanmean(valid)) if valid.size else None,
                'sd':      float(np.nanstd(valid, ddof=1)) if valid.size > 1 else None,
                't':       float(t_stat) if np.isfinite(t_stat) else None,
                'p_two':   float(p_two) if np.isfinite(p_two) else None,
            },
            'hemisphere_variants':      hemi_results,
            'svc_in_mask':              svc_in_mask,
        }
    return rows, persub_rows, details_per_mask


# ── Main ────────────────────────────────────────────────────────────────
def main():
    all_summary = []
    all_persub = []
    all_details = {}

    # --- DSR effect ---
    dsr_rows, dsr_persub, dsr_details = process_effect(
        label_='DSR',
        subj_4d_path=DSR_SUBJ_4D,
        mask_specs=[('lOFC', MASK_PATHS['lOFC']),
                    ('mPFC', MASK_PATHS['mPFC'])],
        cluster_fwep_map=DSR_CLUSTER_FWEP_MAP,
        vox_tstat_map=DSR_VOX_TSTAT_MAP,
    )
    all_summary += dsr_rows
    all_persub  += dsr_persub
    all_details['DSR'] = dsr_details

    # --- STATE effect ---
    # State has no _vox_tstat_c1.nii (only FWE / FDR / unc p maps), so we
    # use the cluster-mass t map as the localiser AND fall back to our
    # own voxelwise t for the within-mask peak search.
    state_rows, state_persub, state_details = process_effect(
        label_='STATE',
        subj_4d_path=STATE_SUBJ_4D,
        mask_specs=[('vmPFC', MASK_PATHS['vmPFC']),
                    ('EC',    MASK_PATHS['EC'])],
        cluster_fwep_map=STATE_CLUSTER_FWEP_MAP,
        vox_tstat_map=STATE_SVC_VOX_TSTAT_MAP,  # PALM-published t-stat (SVC mask)
        cluster_localizer_map=STATE_CLUSTERM_TSTAT_MAP,
        svc_vox_fwep_map=STATE_SVC_VOX_FWEP_MAP,
    )
    all_summary += state_rows
    all_persub  += state_persub
    all_details['STATE'] = state_details

    # ── Save ─────────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_summary)
    persub_df  = pd.DataFrame(all_persub)
    summary_df.to_csv(OUT_DIR / 'mask_vs_cluster_summary.csv', index=False)
    persub_df.to_csv(OUT_DIR / 'per_subject_mean_in_mask.csv', index=False)

    details_payload = {
        'timestamp':        datetime.now().isoformat(timespec='seconds'),
        'inputs': {
            'DSR_subj_4d':         str(DSR_SUBJ_4D),
            'DSR_cluster_fwep':    str(DSR_CLUSTER_FWEP_MAP),
            'DSR_vox_tstat':       str(DSR_VOX_TSTAT_MAP),
            'STATE_subj_4d':       str(STATE_SUBJ_4D),
            'STATE_cluster_fwep':  str(STATE_CLUSTER_FWEP_MAP),
            'STATE_clusterm_tstat':str(STATE_CLUSTERM_TSTAT_MAP),
            'STATE_svc_vox_tstat': str(STATE_SVC_VOX_TSTAT_MAP),
            'STATE_svc_vox_fwep':  str(STATE_SVC_VOX_FWEP_MAP),
            'masks':               {k: str(v) for k, v in MASK_PATHS.items()},
            'fwe_threshold_oneminus_p': FWE_THRESH,
        },
        'effects': all_details,
    }
    with open(OUT_DIR / 'details.json', 'w') as f:
        json.dump(details_payload, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("Saved:")
    print(f"  {OUT_DIR / 'mask_vs_cluster_summary.csv'}")
    print(f"  {OUT_DIR / 'per_subject_mean_in_mask.csv'}")
    print(f"  {OUT_DIR / 'details.json'}")

    # ── Final summary table ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY (mean-within-mask vs cluster-peak per region)")
    print("=" * 70)
    show_cols = [
        'effect', 'region', 'n_mask_voxels',
        'mask_peak_palm_mni_x', 'mask_peak_palm_mni_y', 'mask_peak_palm_mni_z',
        'nearest_cluster_peak_mni_x', 'nearest_cluster_peak_mni_y', 'nearest_cluster_peak_mni_z',
        'dist_mask_to_cluster_peak_mm',
        'n_subjects', 'mean_within_mask', 't_within_mask', 'p_within_mask_two_sided',
    ]
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(summary_df[show_cols].to_string(index=False))


if __name__ == '__main__':
    main()
