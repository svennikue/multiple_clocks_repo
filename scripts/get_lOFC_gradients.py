#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient analysis across subjects — LATERAL OFC edition.

Same pipeline as ``get_subj_gradients.py`` (peak / cluster-COM extraction,
linear-trend + directional tests, orderness, plots), but applied to the
lateral-OFC mask built by ``make_lOFC_mask.py``.

Reads the "cropped grey-matter, not mPFC-masked" 4-D per-subject β maps
in ``complete_quarters_subj_maps/`` and applies the lOFC mask in memory
so voxels outside lOFC don't influence the percentile / cluster steps.

Only quarters files are present in that dir (no ``rot_`` variants, no
``eighths``); the reusable ``DATASETS`` from ``get_subj_gradients.py`` is
overridden with a local one that matches the actual filenames.

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import nibabel as nib
import nilearn.image

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Reuse the low-level helpers from the mPFC gradient script — its main
# loop is guarded by ``if __name__ == '__main__'`` so this import is safe.
from get_subj_gradients import (
    load_niftis,
    extract_projection,
    run_stats_and_plots,
    run_pipeline,
    PEAK_MODES,
    N_CLUSTERS,
    axis_index,
)


# ── Settings ─────────────────────────────────────────────────────────
MASK_PATH = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
                 '/masks/mask_lateral_OFC_LR_resampled.nii.gz')

# Per-subject 4-D β_std files, cropped grey-matter only (not mPFC-masked).
BASE_DIR = ('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
            '/derivatives/group/Main_Results_fMRI/complete_quarters_subj_maps')

# Local DATASETS override — only quarters variants are on disk here.
DATASETS = [
    {
        'label': 'splits (4) curr and next button',
        'n_conditions': 4,
        'files': {
            'current quarter':
                'masked_smooth_fwhm5_CURR_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            'next quarter':
                'masked_smooth_fwhm5_NEXT_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            'next +2 quarter':
                'masked_smooth_fwhm5_NEXT2_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            'next +3 quarter':
                'masked_smooth_fwhm5_NEXT3_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
        },
    },
    {
        'label': 'slits (4) state curr and next button',
        'n_conditions': 4,
        'files': {
            'current quarter':
                'masked_smooth_fwhm5_CURR_QUARTER-split_quarters_DSR_except_prev_button_state-mask_reward-path_beta_std.nii',
            'next quarter':
                'masked_smooth_fwhm5_NEXT_QUARTER-split_quarters_DSR_except_prev_button_state-mask_reward-path_beta_std.nii',
            'next +2 quarter':
                'masked_smooth_fwhm5_NEXT2_QUARTER-split_quarters_DSR_except_prev_button_state-mask_reward-path_beta_std.nii',
            'next +3 quarter':
                'masked_smooth_fwhm5_NEXT3_QUARTER-split_quarters_DSR_except_prev_button_state-mask_reward-path_beta_std.nii',
        },
    },
]

LABEL_PREFIX = 'lOFC'


# ── Load lOFC mask (2 mm fMRI grid) ──────────────────────────────────
lOFC_img = nib.load(str(MASK_PATH))
lOFC_data = lOFC_img.get_fdata().astype(bool)
print(f"lOFC mask: {int(lOFC_data.sum())} voxels, "
      f"shape={lOFC_img.shape}")


def _apply_mask_to_nifti(img, mask_bool):
    """Zero-out voxels outside `mask_bool` in a 3- or 4-D nifti.

    Voxels outside the mask are set to 0 so downstream percentile and
    cluster steps behave the same way they do on the pre-masked mPFC
    files (which also carry zeros outside the mPFC mask).
    """
    data = img.get_fdata().copy()
    if data.ndim == 3:
        if data.shape != mask_bool.shape:
            raise ValueError(
                f"Nifti shape {data.shape} vs mask shape {mask_bool.shape}")
        data[~mask_bool] = 0
    elif data.ndim == 4:
        if data.shape[:3] != mask_bool.shape:
            raise ValueError(
                f"Nifti spatial shape {data.shape[:3]} vs mask "
                f"shape {mask_bool.shape}")
        data[~mask_bool, :] = 0
    else:
        raise ValueError(f"Unsupported nifti ndim: {data.ndim}")
    return nilearn.image.new_img_like(img, data)


# Prefix dataset labels so mPFC and lOFC outputs don't clash if run
# side-by-side, and set up an ROI-specific output dir.
DATASETS_LOFC = [{**d, 'label': f"{LABEL_PREFIX} · {d['label']}"} for d in DATASETS]

_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = os.path.join(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group'
    '/Main_Results_fMRI/lOFC_gradient_results',
    _ts,
)

# ── Run the shared pipeline with an in-memory lOFC-masking hook ──────
def _mask_niftis(niftis):
    return {cond: _apply_mask_to_nifti(img, lOFC_data)
            for cond, img in niftis.items()}

run_pipeline(
    datasets=DATASETS_LOFC,
    base_dir=BASE_DIR,
    out_dir=OUT_DIR,
    roi_label=LABEL_PREFIX,
    postprocess=_mask_niftis,
)
