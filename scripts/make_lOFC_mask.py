#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a bilateral lateral-OFC mask from the Brainnetome probabilistic atlas.

Steps
-----
1. Load the six Brainnetome probability maps that cover lateral OFC:
     A11l L/R  (orbital gyrus, lateral part)
     A12/47l L/R  (lateral orbital area, lateral part)
     A12/47o L/R  (lateral orbital area, orbital part)
2. Threshold each map at PROB_THRESHOLD (default 25% probability).
3. Union across the six binary maps → a single lateral-OFC mask in
   Brainnetome native space (1 mm).
4. FLIRT-resample onto the fMRI-result grid (2 mm, MNI152) using
   nearest-neighbour so the output stays binary.
5. Save at:
     data/masks/mask_lateral_OFC_LR_bin_1mm.nii.gz  (native)
     data/masks/mask_lateral_OFC_LR_resampled.nii.gz (fMRI grid)

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


# ── Settings ─────────────────────────────────────────────────────────
MASK_DIR = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/masks')

PROB_THRESHOLD = 25   # keep voxels with prob ≥ 25%

ATLAS_FILES = [
    'brainnetome_prob_A11l_R.nii.gz',
    'brainnetome_prob_A11l_L.nii.gz',
    'brainnetome_prob_A12_47o_R.nii.gz',
    'brainnetome_prob_A12_47o_L.nii.gz',
    'brainnetome_prob_A12_47l_R.nii.gz',
    'brainnetome_prob_A12_47l_L.nii.gz',
]

# Reference grid for FLIRT: any fMRI result file at the target 2 mm resolution.
# The existing PFC mask is already on that grid, so we use it directly.
FMRI_REFERENCE = MASK_DIR / 'mask_PFC_LR_smoothed_resampled.nii.gz'

OUT_NATIVE = MASK_DIR / 'mask_lateral_OFC_LR_bin_1mm.nii.gz'
OUT_FMRI   = MASK_DIR / 'mask_lateral_OFC_LR_resampled.nii.gz'


# ── Build 1 mm binary union ──────────────────────────────────────────
print(f"Thresholding six probabilistic Brainnetome maps at "
      f"≥ {PROB_THRESHOLD}% and taking the union...")

union = None
ref_img = None
for fname in ATLAS_FILES:
    path = MASK_DIR / fname
    img = nib.load(str(path))
    data = img.get_fdata()
    bin_map = (data >= PROB_THRESHOLD).astype(np.uint8)
    n_vox = int(bin_map.sum())
    print(f"  {fname:40s}  {n_vox:>7d} vox above threshold")
    if union is None:
        union = bin_map
        ref_img = img
    else:
        if bin_map.shape != union.shape:
            raise ValueError(
                f"Shape mismatch: {fname} has {bin_map.shape}, "
                f"expected {union.shape}")
        union = np.maximum(union, bin_map)

n_union = int(union.sum())
print(f"\nUnion mask (1 mm native):  {n_union} voxels")

nib.save(nib.Nifti1Image(union, ref_img.affine, ref_img.header), str(OUT_NATIVE))
print(f"Saved: {OUT_NATIVE}")


# ── FLIRT resample to fMRI-result grid ───────────────────────────────
print(f"\nFLIRT-resampling to fMRI grid ({FMRI_REFERENCE.name})...")
cmd = [
    'flirt',
    '-in',       str(OUT_NATIVE),
    '-ref',      str(FMRI_REFERENCE),
    '-applyxfm', '-usesqform',
    '-out',      str(OUT_FMRI),
    '-interp',   'nearestneighbour',
]
print(' '.join(cmd))
subprocess.run(cmd, check=True)

# Re-binarise as a safety net (nearest-neighbour should keep it binary already).
resampled = nib.load(str(OUT_FMRI))
rd = resampled.get_fdata()
rd_bin = (rd > 0.5).astype(np.uint8)
nib.save(nib.Nifti1Image(rd_bin, resampled.affine, resampled.header),
         str(OUT_FMRI))
n_resampled = int(rd_bin.sum())
print(f"\nSaved: {OUT_FMRI}")
print(f"Resampled voxel count (2 mm): {n_resampled}")
