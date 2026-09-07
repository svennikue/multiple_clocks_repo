#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the methods figure for the anatomical-gradient (harmonic angle) analysis
implemented in ``scripts/harmonic_angle_maps.py``.

Panels
------
  a  the concurrent code cut into four quarters (current, next, +2, +3),
     which enter one searchlight GLM as four competing regressors
  b  the four β's that every voxel therefore has, for two example voxels
  c  the angle assigned to each quarter — the CENTRE of its angular bin
     (45°, 135°, 225°, 315°) — and the cos/sin projection formula
  d  cos and sin as the two components of one vector per voxel:
     its angle is the preferred future step, its length the effect size
  e  every subject contributes a vector; the group mean is tested against
     (0, 0) with Hotelling T²
  f  the same subjects projected onto the unit circle: mean resultant length
     R̄ and the Rayleigh test of angle agreement (``USE_UNIT_VECTOR_MAPS``)
  g  doing that at every voxel → the preferred-angle map

Everything plotted is real. The β profiles, per-subject (cos, sin) vectors,
Hotelling/Rayleigh statistics and the angle map are read from the analysis
outputs. The two example voxels are picked by a fixed rule, never by hand:
among mPFC voxels with Hotelling p < 0.05, the ventral and dorsal quintiles
along MNI z, and within each the voxel with the largest group amplitude.

Output goes to ``data/derivatives/group/gradient_schematic_<date>/`` — one
assembled overview plus every panel as its own PDF/PNG, and a settings JSON.

@author: Svenja Kuechenhoff
"""

import json
import os
from datetime import date
from pathlib import Path

import numpy as np

import mc
from mc.plotting import method_schematic as msch
from mc.plotting import gradient_schematic as grad

SOURCE_DIR = Path('/Users/xpsy1114/Documents/projects/multiple_clocks')
GROUP = SOURCE_DIR / 'data/derivatives/group'

BETA_DIR = (GROUP / 'group_RSA_DSR_quarters_except_prev_button_state'
                    '_glmbase_all-paths-fixed_stickrews_split-buttons')
BETA_FILES = [
    'cropped_masked_smooth_fwhm5_CURR_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
    'cropped_masked_smooth_fwhm5_NEXT_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
    'cropped_masked_smooth_fwhm5_NEXT2_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
    'cropped_masked_smooth_fwhm5_NEXT3_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
]
HARMONIC_DIR = GROUP / 'Main_Results_fMRI/harmonic_angle_maps/quarters'
MPFC_MASK = SOURCE_DIR / 'data/masks/mask_PFC_LR_smoothed_resampled.nii.gz'
TEMPLATE = Path(os.environ.get('FSLDIR', '/Users/xpsy1114/fsl')) / \
    'data/standard/MNI152_T1_2mm_brain.nii.gz'

# The example task whose concurrent code is cut up in panel a — the same
# example used in the RSA methods figure, so the two figures line up.
FMRI_SUB = 'sub-02'
FMRI_TASK = '5-9-4-3'
EXAMPLE_BIN = 1                 # A_reward: the code read out at that bin

SAGITTAL_X_MNI = None      # None = the sagittal slice with the most
                           # suprathreshold mPFC voxels (reported below)
OUT_DIR = GROUP / f"gradient_schematic_{date.today().strftime('%d-%m-%Y')}"
SHOW = False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"loading β maps from {BETA_DIR.name}")
    D = grad.load_gradient_data(BETA_DIR, HARMONIC_DIR, MPFC_MASK, BETA_FILES)
    print(f"  {D['n_subj']} subjects, {D['n_steps']} quarters, "
          f"bin centres {np.round(D['theta_deg'], 1)}°")
    print(f"  cos weights {np.round(np.cos(D['theta']), 3)}")
    print(f"  sin weights {np.round(np.sin(D['theta']), 3)}")

    picks = grad.pick_example_voxels(D)
    for tag, pk in picks.items():
        print(f"  [{tag:>7}] MNI {pk['mni']}  β = "
              f"{np.round(pk['beta'].mean(0), 4).tolist()}  "
              f"angle {pk['angle']:.1f}°  amplitude {pk['amp']:.4f}  "
              f"Hotelling F {pk['F']:.2f} p {pk['p']:.4f}  "
              f"Rayleigh R {pk['R']:.3f} p {pk['p_rayleigh']:.3f}")
    print(f"  (chosen from {picks['ventral']['n_sig_pool']} Hotelling-significant "
          f"mPFC voxels; z cuts {picks['ventral']['z_cut']})")

    xi, x_best, n_at_x = grad.best_sagittal_slice(D)
    print(f"  densest sagittal slice: x = {x_best:+.0f} mm ({n_at_x} sig voxels)")

    examples, _ = msch.build_fmri_examples(FMRI_SUB, str(SOURCE_DIR))
    ex = examples[FMRI_TASK]

    stem = str(OUT_DIR / 'gradient_schematic')
    page = grad.make_gradient_figure(D, picks, ex, TEMPLATE, stem,
                                     x_mni=SAGITTAL_X_MNI, show=SHOW)
    panel_dir = OUT_DIR / 'panels'
    made = grad.save_panels(D, picks, ex, TEMPLATE, str(panel_dir),
                            x_mni=SAGITTAL_X_MNI, show=SHOW)
    print(f"\n  overview page: {page[0]:.1f} x {page[1]:.1f} cm")
    for k, v in made.items():
        print(f"    {k:<16} {v}")

    settings = dict(
        beta_dir=str(BETA_DIR), beta_files=BETA_FILES,
        harmonic_dir=str(HARMONIC_DIR), mpfc_mask=str(MPFC_MASK),
        template=str(TEMPLATE), n_subj=int(D['n_subj']),
        n_quarters=int(D['n_steps']),
        bin_centres_deg=[float(x) for x in D['theta_deg']],
        cos_weights=[float(x) for x in np.cos(D['theta'])],
        sin_weights=[float(x) for x in np.sin(D['theta'])],
        example_task=FMRI_TASK, example_subject=FMRI_SUB,
        example_bin=ex['bin_labels'][EXAMPLE_BIN],
        sagittal_x_mni=(SAGITTAL_X_MNI if SAGITTAL_X_MNI is not None
                        else float(x_best)),
        sagittal_n_sig_voxels_in_slice=int(n_at_x),
        voxel_selection=('mPFC voxels with Hotelling p<0.05; ventral/dorsal '
                         'quintile along MNI z; largest amplitude within each'),
        example_voxels={
            tag: dict(mni=pk['mni'],
                      beta_mean=[float(x) for x in pk['beta'].mean(0)],
                      beta_sem=[float(x) for x in
                                pk['beta'].std(0, ddof=1) / np.sqrt(D['n_subj'])],
                      cos_group=float(pk['cos'].mean()),
                      sin_group=float(pk['sin'].mean()),
                      angle_deg=pk['angle'], amplitude=pk['amp'],
                      hotelling_F=pk['F'], hotelling_p=pk['p'],
                      rayleigh_R=pk['R'], rayleigh_p=pk['p_rayleigh'])
            for tag, pk in picks.items()},
        n_hotelling_sig_mpfc=int(picks['ventral']['n_sig_pool']),
        overview_page_cm=[float(page[0]), float(page[1])],
        panel_sizes_cm=made,
        panel_dir=str(panel_dir))
    with open(OUT_DIR / 'gradient_schematic_settings.json', 'w') as f:
        json.dump(settings, f, indent=2)
    print(f"\nfigures written to {OUT_DIR}")


if __name__ == '__main__':
    main()
