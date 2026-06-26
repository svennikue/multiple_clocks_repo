#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-ROI summary of subject-level effects for the two main fMRI results
(DSR action-plan effect and STATE effect). For each effect we:
  * extract per-subject mean beta within anatomical ROI masks
      - pass 1: full bilateral mask
      - pass 2: hemisphere-restricted mask, where the hemisphere is chosen by
        the FWE-significant voxel count within the ROI
  * find peak voxel(s) within FWE-significant clusters
  * report the FWE-corrected p (cluster-level whole-brain for DSR; voxel-wise
    within PFC+MTL for STATE)
  * save a publication-style box+scatter figure per pass and a stats JSON.

@author: xpsy1114
"""

import os
import json
from datetime import datetime
import numpy as np
import nibabel as nib
import nilearn.image
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.stats import ttest_1samp

import era_brewer
import mc.plotting.figure_layout as figure_layout


# =====================================================
# ======================= PATHS =======================
# =====================================================
source_dir      = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group/Main_Results_fMRI"
masks_dir       = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/masks"

# --- DSR ---
dsr_main_dir         = f"{source_dir}/RSA_quarters_DSR_controls_glmbase_all-paths-fixed_stickrews_split-buttons_smooth5_palm_p0_01"
dsr_subj_dir         = f"{source_dir}/group_RSA_quarters_DSR_controls_glmbase_all-paths-fixed_stickrews_split-buttons_cropped"
subject4d_file_dsr   = os.path.join(dsr_subj_dir, "cropped_masked_smooth_fwhm5_DSR-DSR-contr_except_prev_but-mask_reward-path_beta_std.nii")
group_t_file_dsr     = os.path.join(dsr_main_dir, "cropped_masked_smooth_fwhm5_DSR-DSR-contr_except_prev_but-mask_reward-path_beta_std_vox_tstat_c1.nii")
cluster_FWE_dsr      = os.path.join(dsr_main_dir, "cropped_masked_smooth_fwhm5_DSR-DSR-contr_except_prev_but-mask_reward-path_beta_std_clusterm_tstat_fwep_c1.nii")
lOFC_mask            = os.path.join(masks_dir, "lOFC.nii.gz")
mPFC_mask            = os.path.join(masks_dir, "mask_PFC_LR_smoothed_resampled.nii.gz")

# --- STATE ---
subject4d_file_state = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group/Main_Results_fMRI/Final_state/STATE-DSR_all_controls_noint.nii"
group_t_file_state   = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group/Main_Results_fMRI/Final_state/PFC-and-MonaMTL/STATE-DSR_all_controls_noint_vox_tstat_c1.nii"
cluster_FWE_state    = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group/Main_Results_fMRI/Final_state/PFC-and-MonaMTL/STATE-DSR_all_controls_noint_clusterm_tstat_fwep_c1.nii"
voxel_FWE_file_state = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group/Main_Results_fMRI/Final_state/PFC-and-MonaMTL/STATE-DSR_all_controls_noint_vox_tstat_fwep_c1.nii"
EC_mask              = os.path.join(masks_dir, "mask_entorhinal_all_33_subs.nii")
vmPFC_mask           = os.path.join(masks_dir, "vMPFC.nii.gz")


# =====================================================
# ====================== SETTINGS =====================
# =====================================================
FWE_THRESH     = 0.95            # 1 - p > 0.95  <=>  p_FWE < 0.05
N_TOP_CLUSTERS = 2
FFA_MNI        = np.array([42.0, -55.0, -15.0])
FFA_RADIUS_MM  = 6.0
PUB_FIG_DIR    = os.path.join(source_dir, 'pub_figures')
os.makedirs(PUB_FIG_DIR, exist_ok=True)

np.random.seed(42)


# =====================================================
# ======================= HELPERS =====================
# =====================================================
def load_mask_on_grid(mask_path, ref_img):
    """Load a binary mask and (if needed) resample to the reference image's grid."""
    mask_img = nilearn.image.load_img(mask_path)
    same_grid = (mask_img.shape == ref_img.shape[:3]
                 and np.allclose(mask_img.affine, ref_img.affine))
    if not same_grid:
        mask_img = nilearn.image.resample_to_img(mask_img, ref_img, interpolation='nearest')
    return mask_img.get_fdata() > 0


def mni_x_grid(affine, shape):
    """Return the MNI-x coordinate of every voxel centre in the grid."""
    i, j, k = np.indices(shape)
    coords = np.stack([i, j, k, np.ones_like(i)], axis=-1).reshape(-1, 4)
    return (affine @ coords.T).T[:, 0].reshape(shape)


def sphere_mask_on_grid(center_mni, radius_mm, affine, shape):
    """Spherical ROI of radius_mm around center_mni, sampled on the voxel grid."""
    i, j, k = np.indices(shape)
    coords = np.stack([i, j, k, np.ones_like(i)], axis=-1).reshape(-1, 4)
    mni_xyz = (affine @ coords.T).T[:, :3]
    d = np.linalg.norm(mni_xyz - np.asarray(center_mni, dtype=float), axis=-1)
    return (d <= radius_mm).reshape(shape)


def mask_mean_per_subject(subj_4d_data, mask_bool):
    """Per subject: mean of finite voxels inside mask. Returns 1D array (n_subj,)."""
    if not mask_bool.any():
        return np.full(subj_4d_data.shape[3], np.nan)
    voxels = subj_4d_data[mask_bool, :].astype(float)
    return np.nanmean(voxels, axis=0)


def dominant_hemisphere_in_roi(mask_bool, fwep_data, affine,
                               threshold=FWE_THRESH):
    """Inside `mask_bool`, where do the FWE-significant voxels concentrate?
    Returns 'L' (x<0), 'R' (x>=0), or None if no significant voxels."""
    sig = (fwep_data > threshold) & mask_bool
    if not sig.any():
        return None
    mni_x = mni_x_grid(affine, mask_bool.shape)
    n_left  = int((sig & (mni_x <  0)).sum())
    n_right = int((sig & (mni_x >= 0)).sum())
    if n_left == n_right == 0:
        return None
    return 'L' if n_left > n_right else 'R'


def restrict_to_hemisphere(mask_bool, hemi, affine):
    """hemi in {'L','R'} → restrict mask to that side; None → return mask unchanged."""
    if hemi is None:
        return mask_bool
    mni_x = mni_x_grid(affine, mask_bool.shape)
    return mask_bool & ((mni_x < 0) if hemi == 'L' else (mni_x >= 0))


def peaks_in_significant_clusters(fwep_data, intensity_data, affine,
                                  threshold=FWE_THRESH, n_top=N_TOP_CLUSTERS):
    """Threshold 1-p map, label connected components, sort by size descending,
    keep top n_top. For each cluster, return peak voxel (argmax of intensity_data
    inside the cluster), cluster size, and min FWE p in the cluster."""
    sig_mask = fwep_data > threshold
    if not sig_mask.any():
        return []
    labeled, n_found = label(sig_mask)
    sizes = np.array([(labeled == cid).sum() for cid in range(1, n_found + 1)])
    top_ids = (np.argsort(sizes)[::-1] + 1)[:n_top]
    peaks = []
    for cid in top_ids:
        mask = labeled == cid
        tmp = intensity_data.copy()
        tmp[~mask] = -np.inf
        vox_idx = np.unravel_index(np.argmax(tmp), tmp.shape)
        mni = nib.affines.apply_affine(affine, vox_idx)
        peaks.append({
            'voxel_idx':       tuple(int(x) for x in vox_idx),
            'mni':             np.asarray(mni, dtype=float),
            'peak_intensity':  float(intensity_data[vox_idx]),
            'n_voxels':        int(mask.sum()),
            'fwe_p_min':       float(1.0 - fwep_data[mask].max()),
        })
    return peaks


def label_peak_by_roi(peak_idx, roi_masks):
    for name, mb in roi_masks.items():
        if mb[peak_idx]:
            return name
    return ''


def one_sample_stats(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return {'n': int(v.size), 'mean': float('nan'), 'sd': float('nan'),
                't': float('nan'), 'p': float('nan'), 'sig': 'n.s.'}
    t, p = ttest_1samp(v, 0, alternative = 'greater')
    sig = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.')
    return {'n': int(v.size), 'mean': float(np.nanmean(v)),
            'sd':   float(np.nanstd(v, ddof=1)),
            't':    float(t), 'p': float(p), 'sig': sig}


# =====================================================
# ===================== PLOTTING ======================
# =====================================================
FIG_WIDTH_CM, FIG_HEIGHT_CM, FIG_FONT_PT = 5.0, 4.0, 11


def plot_roi_effects(values_by_region, region_order, region_stats, save_stem,
                     ylabel='ROI-mean β (z-scored)', title=None,
                     palette_name='Showgirl2'):
    cm_per_in = figure_layout.CM_PER_IN
    figsize = (FIG_WIDTH_CM / cm_per_in, FIG_HEIGHT_CM / cm_per_in)

    roi_colour_dict = { 'EC': 0,
                       'medialOFC': 4,
                       'ACC': 1,
                       'HC_anterior': 2,
                       'HC_mid': 6,
                       'Parahippocampal ': 5,
                       'PCC': 3}
    
    palette_all = era_brewer.era_brew(palette_name)
    palette_others = era_brewer.era_brew('Lover2')
    palette = []
    test_order = []
    for r in region_order:
        if r in roi_colour_dict:
            palette.append(palette_all[roi_colour_dict[r]])
            test_order.append(r)
        elif 'control' in r:
            palette.append('#3D3D3D')
            test_order.append(r)
        elif r == 'lateral OFC':
            palette.append(palette_others[0])
            test_order.append(r)
        elif r == 'vmPFC':
            palette.append(palette_all[roi_colour_dict['medialOFC']])
        elif r == 'mPFC':
            palette.append(palette_all[roi_colour_dict['ACC']])
            test_order.append(r)
            
    
    # palette = (list(palette_all) + list(palette_all))[:len(region_order)]




    rc = {
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':       FIG_FONT_PT,
        'axes.labelsize':  FIG_FONT_PT,
        'axes.titlesize':  FIG_FONT_PT,
        'xtick.labelsize': FIG_FONT_PT,
        'ytick.labelsize': FIG_FONT_PT - 1,
        'legend.fontsize': FIG_FONT_PT - 1,
    }
    
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        positions = np.arange(len(region_order)) + 1
        box_data = [np.asarray(values_by_region[r], dtype=float)
                    [np.isfinite(values_by_region[r])] for r in region_order]

        bp = ax.boxplot(box_data, positions=positions, widths=0.55,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.2))
        
        for patch, c in zip(bp['boxes'], palette):
            patch.set_facecolor(c); patch.set_alpha(0.45); patch.set_edgecolor('black')

        rng = np.random.default_rng(42)
        for i, (d, c) in enumerate(zip(box_data, palette)):
            jitter = 0.07 * rng.standard_normal(d.size)
            ax.scatter(positions[i] + jitter, d, s=14, color=c,
                       edgecolor='black', linewidth=0.5, zorder=3)

        ax.axhline(0, color='gray', ls='--', lw=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(region_order, rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

        # Mask-mean t-test stars (descriptive only; FWE stats reported separately).
        ymin, ymax = ax.get_ylim()
        scale = ymax - ymin
        star_y = ymax + scale * 0.04
        for i, r in enumerate(region_order):
            s = region_stats[r]['sig']
            if s == 'n.s.':
                continue
            ax.text(positions[i], star_y, s, ha='center', va='bottom',
                    fontsize=FIG_FONT_PT + 2)
        ax.set_ylim(ymin, star_y + scale * 0.20)

        fig.savefig(save_stem + '.pdf', bbox_inches='tight')
        fig.savefig(save_stem + '.png', dpi=300, bbox_inches='tight')
        plt.show()
    return fig


# =====================================================
# ============== RUN ONE EFFECT (DSR or STATE) ========
# =====================================================
def run_effect(name, subj_4d_path, fwep_path, tstat_path,
               roi_specs, fwe_mode_label, save_dir):
    """
    Parameters
    ----------
    name : str
        Effect label ('DSR' or 'STATE').
    subj_4d_path : str
        Subject-level betas concatenated on the 4th axis.
    fwep_path : str
        FWE-corrected map storing (1 - p), used to (a) decide significance,
        (b) define clusters, (c) pick the dominant hemisphere per ROI.
    tstat_path : str
        Group t-stat map used to find the peak voxel inside each cluster.
    roi_specs : dict
        Ordered dict of ROI specs. Each value is either
        {'kind': 'mask',   'path': <nifti>} or
        {'kind': 'sphere', 'center': <MNI>, 'radius_mm': <float>}.
    fwe_mode_label : str
        Human-readable description of the FWE correction (printed and logged).
    save_dir : str
        Where to write the figures and JSON.
    """
    print(f"\n================ {name} ================")
    subj_img  = nilearn.image.load_img(subj_4d_path)
    fwep_img  = nilearn.image.load_img(fwep_path)
    tstat_img = nilearn.image.load_img(tstat_path)

    # Make sure FWE / t-stat maps live on the subject 4D grid.
    def _match(img, ref):
        same = (img.shape == ref.shape[:3] and np.allclose(img.affine, ref.affine))
        return img if same else nilearn.image.resample_to_img(img, ref, interpolation='linear')
    fwep_img  = _match(fwep_img,  subj_img)
    tstat_img = _match(tstat_img, subj_img)

    subj_data  = subj_img.get_fdata()
    fwep_data  = fwep_img.get_fdata()
    tstat_data = tstat_img.get_fdata()
    affine     = subj_img.affine
    shape      = subj_img.shape[:3]
    n_subj     = subj_data.shape[3]

    # ---- Build ROI masks (full / bilateral) on the subject grid ----
    full_masks = {}
    for roi, spec in roi_specs.items():
        if spec['kind'] == 'mask':
            full_masks[roi] = load_mask_on_grid(spec['path'], subj_img)
        elif spec['kind'] == 'sphere':
            full_masks[roi] = sphere_mask_on_grid(spec['center'], spec['radius_mm'],
                                                  affine, shape)
        else:
            raise ValueError(f"Unknown ROI kind: {spec['kind']}")

    # ---- Pass 1: full mask average ----
    full_vals  = {roi: mask_mean_per_subject(subj_data, mb)
                  for roi, mb in full_masks.items()}
    full_stats = {roi: one_sample_stats(full_vals[roi]) for roi in full_masks}

    # ---- Pass 2: hemisphere-restricted (dominant hemi inside the ROI) ----
    unilat_masks, unilat_hemi = {}, {}
    for roi, mb in full_masks.items():
        hemi = dominant_hemisphere_in_roi(mb, fwep_data, affine)
        unilat_hemi[roi]  = hemi
        unilat_masks[roi] = restrict_to_hemisphere(mb, hemi, affine)
    unilat_vals  = {roi: mask_mean_per_subject(subj_data, mb)
                    for roi, mb in unilat_masks.items()}
    unilat_stats = {roi: one_sample_stats(unilat_vals[roi]) for roi in unilat_masks}

    # ---- Peaks in FWE-significant clusters ----
    peaks = peaks_in_significant_clusters(fwep_data, tstat_data, affine)

    # ---- Print summary ----
    print(f"  n_subjects = {n_subj}")
    print(f"  FWE correction: {fwe_mode_label}")
    print(f"  FWE threshold (1-p): {FWE_THRESH}  (≙ p_FWE < {1 - FWE_THRESH:.3f})")

    def _print_block(label_str, vals_d, stats_d, masks_d, hemi_d=None):
        print(f"\n  -- {label_str} --")
        hdr = f"  {'region':18s} {'hemi':>4s} {'nvox':>5s} {'n':>3s} {'mean':>8s} {'sd':>8s} {'t':>6s} {'p':>8s}  {'sig':>4s}"
        print(hdr)
        for roi, rs in stats_d.items():
            h = hemi_d[roi] if hemi_d else 'L+R'
            h = h if h is not None else '—'
            print(f"  {roi:18s} {h:>4s} {int(masks_d[roi].sum()):>5d} "
                  f"{rs['n']:>3d} {rs['mean']:>+8.4f} {rs['sd']:>8.4f} "
                  f"{rs['t']:>+6.2f} {rs['p']:>8.4f}  {rs['sig']:>4s}")

    _print_block("Pass 1: full bilateral mask (per-subject ROI mean)",
                 full_vals, full_stats, full_masks)
    _print_block("Pass 2: hemisphere-restricted mask (driven by FWE peak side)",
                 unilat_vals, unilat_stats, unilat_masks, unilat_hemi)

    print("\n  -- FWE-significant peaks --")
    if not peaks:
        print("  (no FWE-significant voxels at threshold)")
    for i, pk in enumerate(peaks, start=1):
        roi_in = label_peak_by_roi(pk['voxel_idx'], full_masks)
        roi_tag = f"  [inside {roi_in}]" if roi_in else ''
        print(f"  peak {i}: MNI = ({pk['mni'][0]:+5.1f}, {pk['mni'][1]:+5.1f}, "
              f"{pk['mni'][2]:+5.1f})   peak t = {pk['peak_intensity']:+.3f}   "
              f"cluster = {pk['n_voxels']} vox   FWE p ≤ {pk['fwe_p_min']:.4f}"
              f"{roi_tag}")

    # ---- Plot both passes ----
    stem_full   = os.path.join(save_dir, f'fig_{name.lower()}_roi_betas_full')
    stem_unilat = os.path.join(save_dir, f'fig_{name.lower()}_roi_betas_unilateral')
    plot_roi_effects(full_vals,   list(full_masks.keys()),   full_stats,   stem_full,
                     title=f'{name}: full bilateral mask')
    plot_roi_effects(unilat_vals, list(unilat_masks.keys()), unilat_stats, stem_unilat,
                     title=f'{name}: hemisphere-restricted (FWE-driven)')

    # ---- Stats JSON ----
    def _spec_summary(roi, spec):
        if spec['kind'] == 'mask':
            return {'kind': 'mask', 'path': spec['path']}
        return {'kind': 'sphere',
                'center_mni': [float(x) for x in spec['center']],
                'radius_mm':  float(spec['radius_mm'])}

    def _pass_summary(vals_d, stats_d, masks_d, hemi_d=None):
        return {
            roi: {
                **stats_d[roi],
                'hemisphere':       (hemi_d[roi] if hemi_d else 'L+R'),
                'n_voxels_in_mask': int(masks_d[roi].sum()),
                'values':           [float(v) for v in vals_d[roi].tolist()],
            }
            for roi in masks_d
        }

    stats_summary = {
        'timestamp':   datetime.now().isoformat(timespec='seconds'),
        'effect':      name,
        'script':      os.path.basename(__file__),
        'inputs': {
            'subject_betas_4d': subj_4d_path,
            'fwe_map':          fwep_path,
            'tstat_map':        tstat_path,
            'n_subjects':       int(n_subj),
            'grid_shape':       list(shape),
        },
        'settings': {
            'fwe_threshold':       FWE_THRESH,
            'fwe_p_equivalent':    float(round(1.0 - FWE_THRESH, 4)),
            'n_top_clusters_kept': N_TOP_CLUSTERS,
            'fwe_correction_mode': fwe_mode_label,
            'rois': {roi: _spec_summary(roi, spec) for roi, spec in roi_specs.items()},
        },
        'pass1_full_mask':              _pass_summary(full_vals,   full_stats,   full_masks),
        'pass2_unilateral_mask':        _pass_summary(unilat_vals, unilat_stats, unilat_masks, unilat_hemi),
        'fwe_peaks': [
            {'rank':                 i + 1,
             'mni':                  [float(x) for x in pk['mni']],
             'voxel_index':          list(pk['voxel_idx']),
             'peak_tstat':           pk['peak_intensity'],
             'cluster_size_voxels':  pk['n_voxels'],
             'fwe_p_min':            pk['fwe_p_min'],
             'inside_roi':           label_peak_by_roi(pk['voxel_idx'], full_masks)}
            for i, pk in enumerate(peaks)
        ],
    }
    json_path = os.path.join(save_dir, f'fig_{name.lower()}_roi_betas_stats.json')
    with open(json_path, 'w') as f:
        json.dump(stats_summary, f, indent=2)

    print(f"\n  saved figures → {stem_full}.{{pdf,png}}")
    print(f"                 {stem_unilat}.{{pdf,png}}")
    print(f"  saved stats   → {json_path}")
    return stats_summary


# =====================================================
# ===================== DSR ===========================
# =====================================================
roi_specs_dsr = {
    'mPFC':          {'kind': 'mask',   'path': mPFC_mask},
    'lateral OFC':   {'kind': 'mask',   'path': lOFC_mask},
    'FFA (control)': {'kind': 'sphere', 'center': FFA_MNI, 'radius_mm': FFA_RADIUS_MM},
}
run_effect(
    name           = 'DSR',
    subj_4d_path   = subject4d_file_dsr,
    fwep_path      = cluster_FWE_dsr,
    tstat_path     = group_t_file_dsr,
    roi_specs      = roi_specs_dsr,
    fwe_mode_label = 'whole-brain cluster-level FWE',
    save_dir       = PUB_FIG_DIR,
)


# =====================================================
# ===================== STATE =========================
# =====================================================
roi_specs_state = {
    'EC':            {'kind': 'mask',   'path': EC_mask},
    'vmPFC':         {'kind': 'mask',   'path': vmPFC_mask},
    'FFA (control)': {'kind': 'sphere', 'center': FFA_MNI, 'radius_mm': FFA_RADIUS_MM},
}
run_effect(
    name           = 'STATE',
    subj_4d_path   = subject4d_file_state,
    fwep_path      = voxel_FWE_file_state,
    tstat_path     = group_t_file_state,
    roi_specs      = roi_specs_state,
    fwe_mode_label = 'voxel-wise FWE within PFC+MTL mask',
    save_dir       = PUB_FIG_DIR,
)
