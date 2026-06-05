#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:23:54 2026

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
# import pdb; pdb.set_trace()



# =====================================================
# ======================= PATHS =======================
# =====================================================

source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group/Main_Results_fMRI"

main_result = f"{source_dir}/RSA_quarters_DSR_controls_glmbase_all-paths-fixed_stickrews_split-buttons_smooth5_palm_p0_01"
subject_results = f"{source_dir}/group_RSA_quarters_DSR_controls_glmbase_all-paths-fixed_stickrews_split-buttons_cropped"

file_name = "cropped_masked_smooth_fwhm5_DSR-DSR-contr_except_prev_but-mask_reward-path_beta_std.nii"
cluster_name = "cropped_masked_smooth_fwhm5_DSR-DSR-contr_except_prev_but-mask_reward-path_beta_std_clusterm_tstat_fwep_c1.nii"
main_name = "cropped_masked_smooth_fwhm5_DSR-DSR-contr_except_prev_but-mask_reward-path_beta_std_vox_tstat_c1.nii"
fpath = os.path.join(subject_results, file_name)
DSR_nifti = nilearn.image.load_img(fpath)

cluster_path = os.path.join(main_result, cluster_name)
cluster_nifit = nilearn.image.load_img(cluster_path)

main_path = os.path.join(main_result, main_name)
main_nifit = nilearn.image.load_img(main_path)


# =====================================================
# ====== SETTINGS for peak-voxel ROI analysis =========
# =====================================================
# The cluster map from PALM stores ``1 − p`` (so significant voxels are CLOSE
# TO 1, not close to 0). Threshold + label connected components, then take
# the top-N clusters by size as our ROIs of interest.
FWE_THRESH       = 0.95         # 1 − p > 0.95  ⇔  p_FWE < 0.05
N_TOP_CLUSTERS   = 2            # how many clusters to retain
FFA_MNI          = np.array([42.0, -55.0, -15.0])  # standard right FFA control
PUB_FIG_DIR      = os.path.join(source_dir, 'pub_figures')
os.makedirs(PUB_FIG_DIR, exist_ok=True)


# =====================================================
# ============== helpers ==============================
# =====================================================
def peak_voxels_in_top_clusters(cluster_map, main_map, affine,
                                threshold=FWE_THRESH, n_top=N_TOP_CLUSTERS):
    """Threshold the cluster map (PALM stores ``1 − p``), label connected
    components, keep the ``n_top`` largest, and for each return the
    peak-tstat voxel from ``main_map`` inside that cluster.

    Returns a list of dicts ``{'voxel_idx', 'mni', 'peak_tstat', 'n_voxels'}``,
    sorted by cluster size (largest first).
    """
    sig_mask = cluster_map > threshold
    if not sig_mask.any():
        return []
    labeled, n_found = label(sig_mask)
    sizes = np.array([(labeled == cid).sum() for cid in range(1, n_found + 1)])
    top_ids = (np.argsort(sizes)[::-1] + 1)[:n_top]

    peaks = []
    for cid in top_ids:
        mask = labeled == cid
        tmp = main_map.copy()
        tmp[~mask] = -np.inf
        vox_idx = np.unravel_index(np.argmax(tmp), tmp.shape)
        mni = nib.affines.apply_affine(affine, vox_idx)
        peaks.append({
            'voxel_idx':  tuple(int(x) for x in vox_idx),
            'mni':        np.asarray(mni, dtype=float),
            'peak_tstat': float(main_map[vox_idx]),
            'n_voxels':   int(mask.sum()),
        })
    return peaks


def voxel_index_from_mni(mni_xyz, affine):
    """MNI → voxel index, rounded to nearest integer."""
    inv = np.linalg.inv(affine)
    vox = nib.affines.apply_affine(inv, np.asarray(mni_xyz, dtype=float))
    return tuple(int(round(v)) for v in vox)


def per_subject_betas(subj_4d_data, vox_idx):
    """Pull the per-subject vector at a single voxel from a 4D ``(x,y,z,subj)``
    array. NaNs are returned as-is so the caller can filter."""
    i, j, k = vox_idx
    return subj_4d_data[i, j, k, :].astype(float)


# =====================================================
# ====== Identify peak voxels + extract per-subject ===
# =====================================================
cluster_data = cluster_nifit.get_fdata()
main_data    = main_nifit.get_fdata()
subj_data    = DSR_nifti.get_fdata()           # shape (X, Y, Z, n_subjects)
affine       = cluster_nifit.affine

# Affines must match for direct voxel-index extraction; check + fall back to
# MNI-based lookup if not (here they do match — same crop / mask grid).
assert np.allclose(DSR_nifti.affine, affine), (
    "subject 4D file is not on the same grid as the cluster map; would need "
    "resampling before voxel-wise extraction.")

peaks = peak_voxels_in_top_clusters(cluster_data, main_data, affine)
if len(peaks) < 2:
    raise RuntimeError(f"Expected ≥2 significant clusters, got {len(peaks)}")

# Assign the medial peak (smallest |x|) → mPFC, the other → lateral OFC.
peaks_sorted_by_lateral = sorted(peaks[:2], key=lambda p: abs(p['mni'][0]))
mPFC_peak = peaks_sorted_by_lateral[0]
lOFC_peak = peaks_sorted_by_lateral[1]
FFA_vox   = voxel_index_from_mni(FFA_MNI, affine)

print("\n================ PEAK VOXELS ================")
print(f"  mPFC          peak MNI = ({mPFC_peak['mni'][0]:+5.1f}, "
      f"{mPFC_peak['mni'][1]:+5.1f}, {mPFC_peak['mni'][2]:+5.1f})"
      f"   peak t = {mPFC_peak['peak_tstat']:.3f}"
      f"   cluster = {mPFC_peak['n_voxels']} vox")
print(f"  lateral OFC   peak MNI = ({lOFC_peak['mni'][0]:+5.1f}, "
      f"{lOFC_peak['mni'][1]:+5.1f}, {lOFC_peak['mni'][2]:+5.1f})"
      f"   peak t = {lOFC_peak['peak_tstat']:.3f}"
      f"   cluster = {lOFC_peak['n_voxels']} vox")
print(f"  FFA (control) MNI       = ({FFA_MNI[0]:+5.1f}, "
      f"{FFA_MNI[1]:+5.1f}, {FFA_MNI[2]:+5.1f})  (a-priori coord)")

regions = {
    'mPFC':        per_subject_betas(subj_data, mPFC_peak['voxel_idx']),
    'lateral OFC': per_subject_betas(subj_data, lOFC_peak['voxel_idx']),
    'FFA':         per_subject_betas(subj_data, FFA_vox),
}

print("\n================ PER-SUBJECT EFFECTS ================")
print(f"  {'region':14s} {'n':>3s}  {'mean':>8s}  {'sd':>8s}  "
      f"{'t':>6s}  {'p':>8s}  {'sig':>4s}")
region_stats = {}
for name, vals in regions.items():
    v = vals[np.isfinite(vals)]
    if v.size < 2:
        t_val, p_val = np.nan, np.nan
    else:
        t_val, p_val = ttest_1samp(v, 0)
    sig = ('***' if p_val < 0.001 else '**' if p_val < 0.01
           else '*' if p_val < 0.05 else 'n.s.')
    region_stats[name] = {'n': int(v.size), 'mean': float(np.nanmean(v)),
                          'sd': float(np.nanstd(v, ddof=1)),
                          't': float(t_val), 'p': float(p_val), 'sig': sig}
    print(f"  {name:14s} {v.size:>3d}  {float(np.nanmean(v)):>+8.4f}  "
          f"{float(np.nanstd(v, ddof=1)):>8.4f}  {float(t_val):>+6.2f}  "
          f"{float(p_val):>8.4f}  {sig:>4s}")


# =====================================================
# ============== PUBLICATION FIGURE ====================
# =====================================================
# Targets ~6 × 4 cm on an A4 page (i.e. width_frac ≈ 6 / 7.09 ≈ 0.85 of the
# A4 usable column). Save at the actual print size so font size 11 pt renders
# at 11 pt with no scaling.
FIG_WIDTH_CM   = 5.0
FIG_HEIGHT_CM  = 4.0
FIG_FONT_PT    = 11

def plot_peak_voxel_effects(values_by_region, region_order, region_stats,
                            save_stem,
                            fig_width_cm=FIG_WIDTH_CM,
                            fig_height_cm=FIG_HEIGHT_CM,
                            font_pt=FIG_FONT_PT,
                            palette_name='Lover2',
                            ylabel='β (z-scored)'):
    """Box + scatter of per-subject betas at each region's peak voxel, with
    one-sample t-test stars above each box. Sized to ``fig_width_cm`` ×
    ``fig_height_cm`` (default 6 × 4 cm) so the saved file can be dropped
    into an A4 page at 100 % and the fonts render at ``font_pt`` exactly.

    ``palette_name`` is an era_brewer key (default 'Lover2').
    """
    cm_per_in = figure_layout.CM_PER_IN
    figsize = (fig_width_cm / cm_per_in, fig_height_cm / cm_per_in)

    palette_all = era_brewer.era_brew(palette_name)
    # After era_brew reorders: [0]=#BF567F deep pink, [1]=#4478ac deep blue,
    # [2]=#D9A8E8 lavender, [3]=#6CB4DC light blue, [4]=#FAAFD3 light pink.
    # Pick pink/blue for the two effect regions, lavender for the control.
    palette = [palette_all[0], palette_all[1], palette_all[2]]
    palette = (palette + palette_all)[:len(region_order)]

    rc = {
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':       font_pt,
        'axes.labelsize':  font_pt,
        'axes.titlesize':  font_pt,
        'xtick.labelsize': font_pt,
        'ytick.labelsize': font_pt - 1,
        'legend.fontsize': font_pt - 1,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        positions = np.arange(len(region_order)) + 1
        box_data = []
        for r in region_order:
            v = np.asarray(values_by_region[r], dtype=float)
            box_data.append(v[np.isfinite(v)])

        bp = ax.boxplot(box_data, positions=positions, widths=0.55,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.2))
        for patch, c in zip(bp['boxes'], palette):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
            patch.set_edgecolor('black')

        rng = np.random.default_rng(42)
        for i, (d, c) in enumerate(zip(box_data, palette)):
            jitter = 0.07 * rng.standard_normal(d.size)
            ax.scatter(positions[i] + jitter, d, s=14, color=c,
                       edgecolor='black', linewidth=0.5, zorder=3)

        ax.axhline(0, color='gray', ls='--', lw=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(region_order, rotation=20, ha='right')
        ax.set_ylabel(ylabel)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

        # Stars above each box. ``n.s.`` left empty (cleaner figure).
        ymin, ymax = ax.get_ylim()
        scale = ymax - ymin
        star_y = ymax + scale * 0.04
        for i, r in enumerate(region_order):
            sig = region_stats[r]['sig']
            if sig == 'n.s.':
                continue
            ax.text(positions[i], star_y, sig, ha='center', va='bottom',
                    fontsize=font_pt + 2)
        ax.set_ylim(ymin, star_y + scale * 0.20)

        # Save both vector and high-res preview.
        fig.savefig(save_stem + '.pdf', bbox_inches='tight')
        fig.savefig(save_stem + '.png', dpi=300, bbox_inches='tight')
        plt.show()
    return fig


region_order = ['mPFC', 'lateral OFC', 'FFA']
plot_peak_voxel_effects(
    values_by_region=regions, region_order=region_order,
    region_stats=region_stats,
    save_stem=os.path.join(PUB_FIG_DIR, 'fig_peak_voxel_DSR_betas'))

print(f"\nSaved publication figure to: {PUB_FIG_DIR}/fig_peak_voxel_DSR_betas.{{pdf,png}}")


# =====================================================
# ============== STATS JSON ============================
# =====================================================
# Persist everything that went into the figure so it can be reconstructed /
# cited later without re-running the analysis.
stats_summary = {
    'timestamp':    datetime.now().isoformat(timespec='seconds'),
    'script':       os.path.basename(__file__),
    'inputs': {
        'subject_betas_4d': os.path.join(subject_results, file_name),
        'cluster_map':      os.path.join(main_result, cluster_name),
        'main_tstat':       os.path.join(main_result, main_name),
        'n_subjects':       int(subj_data.shape[3]),
        'grid_shape':       list(subj_data.shape[:3]),
    },
    'settings': {
        'fwe_threshold':       FWE_THRESH,   # 1 - p > threshold ⇔ p_FWE < (1-threshold)
        'fwe_p_equivalent':    float(round(1.0 - FWE_THRESH, 4)),
        'n_top_clusters_kept': N_TOP_CLUSTERS,
        'ffa_mni_apriori':     [float(x) for x in FFA_MNI],
    },
    'peaks': {
        'mPFC': {
            'mni':         [float(x) for x in mPFC_peak['mni']],
            'voxel_index': list(mPFC_peak['voxel_idx']),
            'peak_tstat':  float(mPFC_peak['peak_tstat']),
            'cluster_size_voxels': int(mPFC_peak['n_voxels']),
        },
        'lateral OFC': {
            'mni':         [float(x) for x in lOFC_peak['mni']],
            'voxel_index': list(lOFC_peak['voxel_idx']),
            'peak_tstat':  float(lOFC_peak['peak_tstat']),
            'cluster_size_voxels': int(lOFC_peak['n_voxels']),
        },
        'FFA': {
            'mni':         [float(x) for x in FFA_MNI],
            'voxel_index': list(FFA_vox),
            'peak_tstat':  float(main_data[FFA_vox]),
            'cluster_size_voxels': None,   # a-priori coord, no cluster
        },
    },
    'per_subject_effects': {
        r: {
            'n':           region_stats[r]['n'],
            'mean':        region_stats[r]['mean'],
            'sd':          region_stats[r]['sd'],
            'sem':         (region_stats[r]['sd']
                            / max(region_stats[r]['n'] ** 0.5, 1)),
            't':           region_stats[r]['t'],
            'p_two_sided': region_stats[r]['p'],
            'sig':         region_stats[r]['sig'],
            'values':      [float(v) for v in regions[r].tolist()],
        }
        for r in region_order
    },
}
stats_path = os.path.join(PUB_FIG_DIR, 'fig_peak_voxel_DSR_betas_stats.json')
with open(stats_path, 'w') as f:
    json.dump(stats_summary, f, indent=2)
print(f"Saved statistical summary to: {stats_path}")