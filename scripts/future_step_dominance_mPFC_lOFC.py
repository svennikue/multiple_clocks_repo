#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Which future step dominates mPFC vs. lateral OFC?

For every (ROI × future-step) cell we compute BOTH an effect-size summary
and a proper mask-corrected significance test:

  1. Mean β per subject inside the ROI mask → group mean ± SEM. Display
     value; SVC / cluster / LOSO carry the significance.
  2. Small-volume-corrected max-t sign-flip permutation, restricted to the
     ROI voxels (Nichols & Holmes 2002). "Is there ANY voxel in this mask
     whose one-sample t against 0 exceeds the null?". Focal-effect friendly.
  3. Cluster-mass sign-flip within the ROI (cluster-forming t = CLUSTER_CFT).
  4. Leave-one-subject-out cross-validated mean β on the training-set
     top-k voxels — unbiased effect size for the mask's active subset.
  5. Pairwise contrasts between steps, within each ROI, via sign-flip on
     the paired-difference means (Bonferroni across the step-pairs).
  6. Shape contrasts (linear vs. peak) per ROI and, when both ROIs are
     included, their ROI × contrast interaction.

The script runs over one or more DATASETS. Each dataset has its own base
directory, file map (step → nifti), and list of ROIs to analyse — so you
can compare unmasked quarters (mPFC + lOFC) alongside the mPFC-masked
`rot_quarters` and `eighths` sets in a single run. Which datasets fire
is controlled by the ``DATASETS_TO_RUN`` list at the top of the script.

Outputs land in ``OUT_DIR/<dataset_label>/`` — mean β CSV, per-(ROI×step)
LOSO held-out arrays, a JSON with all stats, and three figures
(mean β line, significance heatmap across the three tests, LOSO effect
size line).

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats as sstats

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from scipy.ndimage import label as _ndlabel

from mc.plotting.cell_results import SHOWGIRL2_DISCRETE
from svc_loso_test import tstat, null_max_t


# ── Settings ─────────────────────────────────────────────────────────
MASK_DIR = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/masks')
MPFC_MASK_PATH = MASK_DIR / 'mask_PFC_LR_smoothed_resampled.nii.gz'
LOFC_MASK_PATH = MASK_DIR / 'mask_lateral_OFC_LR_resampled.nii.gz'

# Base dirs used by the different datasets below.
#   UNMASKED    — cropped grey-matter, NOT mPFC-masked (per-subject 4-D β_std).
#                 The only place lOFC can pick up real signal.
#   MPFC_MASKED — the classic get_subj_gradients.py source dir, mPFC-restricted.
#                 Fine for mPFC-only follow-ups (rot_quarters, eighths).
BASE_DIR_UNMASKED = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
    '/derivatives/group/Main_Results_fMRI/complete_quarters_subj_maps')
BASE_DIR_MPFC_MASKED = Path(
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
    '/derivatives/group'
    '/group_RSA_DSR_quarters_except_prev_button_state'
    '_glmbase_all-paths-fixed_stickrews_split-buttons_cropped_masked')

OUT_DIR = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data'
               '/derivatives/group/Main_Results_fMRI/mPFC_vs_lOFC_by_step')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERM = 500
SEED   = 0
LOSO_K = [25, 50, 100]         # top-k voxels for LOSO CV

# Cluster-mass sign-flip null: cluster-forming t-threshold. 2.6 ≈ one-tailed
# p ≈ 0.007 for n=33 df=32, standard-ish choice for a 3-D ROI cluster test.
CLUSTER_CFT       = 2.6
CLUSTER_N_PERM    = 500   # cheaper than SVC N_PERM; still tight FWE

# ROI colours per CLAUDE.md conventions.
ROI_COLOURS = {
    'mPFC': SHOWGIRL2_DISCRETE[1],
    'lOFC': SHOWGIRL2_DISCRETE[4],
}


# ── DATASETS ─────────────────────────────────────────────────────────
# Each entry defines one full analysis: which files feed which steps,
# which base dir holds them, and which ROIs to test. Add more entries or
# trim DATASETS_TO_RUN to control what runs on any given execution.
DATASETS = [
    {
        'label':    'quarters',
        'base_dir': BASE_DIR_UNMASKED,
        'rois':     ['mPFC', 'lOFC'],
        'files': {
            'current':  'masked_smooth_fwhm5_CURR_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            'next':     'masked_smooth_fwhm5_NEXT_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            'next +2':  'masked_smooth_fwhm5_NEXT2_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
            'next +3':  'masked_smooth_fwhm5_NEXT3_QUARTER-split_quarters_DSR_except_prev_button-mask_reward-path_beta_std.nii',
        },
    },
    {
        # 4-way rotated quarters — masked to mPFC only, so lOFC is skipped.
        'label':    'rot_quarters',
        'base_dir': BASE_DIR_MPFC_MASKED,
        'rois':     ['mPFC'],
        'files': {
            'current':  'ROT_CURR_QUARTER-split_rot_quarters_DSR_except_prev_but_masked.nii.gz',
            'next':     'ROT_NEXT_QUARTER-split_rot_quarters_DSR_except_prev_but_masked.nii.gz',
            'next +2':  'ROT_NEXT2_QUARTER-split_rot_quarters_DSR_except_prev_but_masked.nii.gz',
            'next +3':  'ROT_NEXT3_QUARTER-split_rot_quarters_DSR_except_prev_but_masked.nii.gz',
        },
    },
    {
        # 8-way eighths — mPFC-masked in the source dir, so mPFC only.
        'label':    'eighths',
        'base_dir': BASE_DIR_MPFC_MASKED,
        'rois':     ['mPFC'],
        'files': {
            'now':    'LOCATION-split_eighths_DSR_except_prev_button-mask_reward-path_beta_std.nii.gz',
            '+1 fut': 'DSR_ONEFUT-split_eighths_DSR_except_prev_button_masked.nii.gz',
            '+2 fut': 'DSR_TWOFUT-split_eighths_DSR_except_prev_button_masked.nii.gz',
            '+3 fut': 'DSR_THREEFUT-split_eighths_DSR_except_prev_button_masked.nii.gz',
            '+4 fut': 'DSR_FOURFUT-split_eighths_DSR_except_prev_button_masked.nii.gz',
            '+5 fut': 'DSR_FIVEFUT-split_eighths_DSR_except_prev_button_masked.nii.gz',
            '+6 fut': 'DSR_SIXFUT-split_eighths_DSR_except_prev_button_masked.nii.gz',
            '+7 fut': 'DSR_SEVENFUT-split_eighths_DSR_except_prev_button_masked.nii.gz',
        },
    },
]

# Which dataset labels to run (subset of DATASETS above).
DATASETS_TO_RUN = ['quarters', 'rot_quarters', 'eighths']


# ── Mask loader ──────────────────────────────────────────────────────
def _load_mask(path):
    img = nib.load(str(path))
    d = img.get_fdata()
    return img, (d > 0.5)

mpfc_img, mpfc_bool = _load_mask(MPFC_MASK_PATH)
lofc_img, lofc_bool = _load_mask(LOFC_MASK_PATH)
print(f"mPFC mask: {int(mpfc_bool.sum())} voxels, shape={mpfc_img.shape}")
print(f"lOFC mask: {int(lofc_bool.sum())} voxels, shape={lofc_img.shape}")
overlap = int((mpfc_bool & lofc_bool).sum())
print(f"mPFC ∩ lOFC overlap: {overlap} voxels "
      f"({100 * overlap / max(1, int(lofc_bool.sum())):.1f}% of lOFC)")

ALL_MASKS = {'mPFC': mpfc_bool, 'lOFC': lofc_bool}


# ── Extract per-(subj, vox) β per (ROI, step) ────────────────────────
def _load_step_beta(fname, base_dir):
    path = Path(base_dir) / fname
    if not path.is_file():
        raise FileNotFoundError(f"missing: {path}")
    img = nib.load(str(path))
    data = img.get_fdata()
    if data.ndim not in (3, 4):
        raise ValueError(f"{fname}: expected 3-D or 4-D nifti, got {data.shape}")
    if data.ndim == 3:
        data = data[..., None]   # (X,Y,Z,1)
    return img, data


def _mask_extract(data, mask_bool):
    """Return (vals, keep_cols) where vals is (n_subj, n_kept_vox) β and
    keep_cols is a bool selector into the flat-mask index order (needed
    to map columns back to 3-D for cluster analysis). Drops columns that
    are all-NaN or all-zero across subjects — background voxels the file
    happens to include."""
    vals = data[mask_bool, :].T                        # (n_subj, n_mask_vox)
    if vals.size == 0:
        return vals, np.zeros(0, dtype=bool)
    finite_any = np.any(np.isfinite(vals) & (vals != 0), axis=0)
    return vals[:, finite_any], finite_any


# ── Per-(ROI, step) stats ────────────────────────────────────────────
def _svc_max_t(vals, n_perm, seed):
    """SVC peak-t sign-flip permutation on (n_subj, n_vox)."""
    if vals.shape[1] == 0:
        return None
    t_obs = tstat(vals)
    max_t_obs = float(np.nanmax(t_obs))
    null = null_max_t(np.nan_to_num(vals), n_perm=n_perm, seed=seed)
    p_fwe = float((null >= max_t_obs).mean())
    return {
        'n_vox':          int(vals.shape[1]),
        'peak_t':         max_t_obs,
        'peak_vox_idx':   int(np.nanargmax(t_obs)),
        'peak_p_FWE':     p_fwe,
        't_crit_FWE05':   float(np.percentile(null, 95)),
        'n_supra_FWE05':  int((t_obs >= np.percentile(null, 95)).sum()),
    }


def _loso_topk(vals, k_values, n_perm, seed_base):
    """LOSO CV: pick top-k voxels per fold using n-1 subjects (peak-t),
    read held-out subject's mean β on that subset. One-sample t on
    held-out means, plus sign-flip p on those means."""
    out = {}
    held_by_k = {}
    n_subj, n_vox = vals.shape
    if n_vox == 0:
        return out, held_by_k
    for k in k_values:
        kk = int(min(k, n_vox))
        held = np.zeros(n_subj)
        for s in range(n_subj):
            train = np.delete(np.arange(n_subj), s)
            t_tr  = tstat(np.nan_to_num(vals[train]))
            top   = np.argsort(-t_tr)[:kk]
            held[s] = np.nanmean(vals[s, top])
        t_stat, p_two = sstats.ttest_1samp(held, 0.0, nan_policy='omit')
        held_col = held.reshape(-1, 1)
        null     = null_max_t(np.nan_to_num(held_col),
                              n_perm=n_perm, seed=seed_base + kk)
        p_perm   = float((null >= float(tstat(held_col))).mean())
        out[str(kk)] = {
            'k':          kk,
            'held_mean':  float(np.nanmean(held)),
            'held_sem':   float(np.nanstd(held, ddof=1) / np.sqrt(n_subj)),
            't':          float(t_stat),
            'p_ttest_2s': float(p_two),
            'p_perm_1s':  p_perm,
        }
        held_by_k[str(kk)] = held.copy()
    return out, held_by_k


def _stars(p):
    if not np.isfinite(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '·'
    return 'n.s.'


def _cluster_mass_signflip(vals, mask_bool, keep_cols, cft, n_perm, seed):
    """Cluster-mass sign-flip null within a 3-D ROI (26-connectivity)."""
    if vals.shape[1] == 0:
        return None

    ijk = np.where(mask_bool)
    ijk = tuple(ax[keep_cols] for ax in ijk)

    def _max_mass(t_vec):
        vol = np.zeros(mask_bool.shape, dtype=np.float32)
        vol[ijk] = t_vec
        supra = vol >= cft
        lbl, n_c = _ndlabel(supra)
        if n_c == 0:
            return 0.0
        return float(max(vol[lbl == c].sum() for c in range(1, n_c + 1)))

    t_obs = tstat(vals)
    obs_mass = _max_mass(t_obs)

    rng = np.random.RandomState(seed)
    n_sub = vals.shape[0]
    null = np.empty(n_perm)
    vals_nz = np.nan_to_num(vals)
    for p in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_sub)
        null[p] = _max_mass(tstat(vals_nz * signs[:, None]))
    p_fwe = float((null >= obs_mass).mean()) if obs_mass > 0 else 1.0
    return {
        'cft':           cft,
        'obs_max_mass':  obs_mass,
        'p_FWE_cluster': p_fwe,
        'null_p95':      float(np.percentile(null, 95)),
        'n_perm':        n_perm,
    }


# ── Sign-flip helpers for step contrasts ─────────────────────────────
def _paired_signflip(x, y, n_perm, seed):
    """Sign-flip 2-sided p on paired difference. Returns (t_obs, p)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return np.nan, np.nan
    t_obs = float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(d.size)))
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    D = signs * d[None, :]
    m = D.mean(1); s = D.std(1, ddof=1)
    T = np.where(s > 0, m / (s / np.sqrt(d.size)), 0.0)
    return t_obs, float(np.mean(np.abs(T) >= abs(t_obs)))


def _signflip_1s(x, n_perm, seed, direction='greater'):
    """One-sample sign-flip. direction='greater' → P(T_null ≥ T_obs)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan, np.nan
    denom = np.std(x, ddof=1) / np.sqrt(x.size)
    if denom == 0:
        return np.nan, np.nan
    t_obs = float(np.mean(x) / denom)
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, x.size))
    D = signs * x[None, :]
    m = D.mean(1); s = D.std(1, ddof=1)
    T = np.where(s > 0, m / (s / np.sqrt(x.size)), 0.0)
    if direction == 'greater':
        return t_obs, float(np.mean(T >= t_obs))
    return t_obs, float(np.mean(T <= t_obs))


def _shape_contrasts_for_nsteps(n_steps):
    """Return dict of unit-normalised shape contrasts appropriate for the
    given step count. Empty dict → shape contrasts are skipped."""
    if n_steps == 4:
        lin  = np.array([+1.5, +0.5, -0.5, -1.5])
        step = np.array([+1.0, +1.0, -1.0, -1.0])
        out = {
            'linear': lin  / np.linalg.norm(lin),
            'step':   step / np.linalg.norm(step),
        }
    elif n_steps == 8:
        # linear: monotone decline from now → +7 fut
        lin = np.arange(n_steps - 1, -n_steps, -2, dtype=float)
        # peak_at_+4: mirrors the get_subj_gradients "hand-crafted" weight
        # for eighths, [-3,-2,-1,0,+3,0,-1,-2], mean-centred
        peak = np.array([-3, -2, -1, 0, +3, 0, -1, -2], dtype=float)
        peak = peak - peak.mean()
        out = {
            'linear':      lin  / np.linalg.norm(lin),
            'peak_at_+4':  peak / np.linalg.norm(peak),
        }
    else:
        out = {}
    return out


def _subject_contrast(df, roi, steps, contrast):
    """Per-subject contrast values (subjects sorted). NaN if any β missing."""
    subs = sorted(df[df['roi'] == roi]['subject'].unique())
    vals = []
    for s in subs:
        beta = np.array([
            df[(df['roi']==roi) & (df['step']==step) & (df['subject']==s)]
              ['mean_beta'].mean() for step in steps
        ], dtype=float)
        vals.append(float(contrast @ beta) if np.all(np.isfinite(beta)) else np.nan)
    return np.array(vals)


# ── Main per-dataset runner ──────────────────────────────────────────
def run_dataset(cfg, n_perm=N_PERM, cluster_n_perm=CLUSTER_N_PERM, seed=SEED):
    """Full compute + plots for one dataset entry. Writes into
    ``OUT_DIR / cfg['label'] /``.  Returns the results dict."""
    label     = cfg['label']
    base_dir  = Path(cfg['base_dir'])
    file_map  = cfg['files']
    rois      = [r for r in cfg['rois'] if r in ALL_MASKS]
    masks     = {r: ALL_MASKS[r] for r in rois}
    out_dir   = OUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    steps     = list(file_map.keys())
    n_steps   = len(steps)

    print(f"\n{'#'*70}\n### DATASET: {label}   "
          f"({n_steps} steps × {len(rois)} ROI{'s' if len(rois)>1 else ''})   "
          f"base={base_dir.name}\n{'#'*70}")

    subject_records = []
    results = {roi: {} for roi in rois}

    for step, fname in file_map.items():
        try:
            _, data = _load_step_beta(fname, base_dir)
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}")
            continue
        n_subj = data.shape[3]
        print(f"\n=== [{label}] step={step}  ({n_subj} subjects) ===")

        for roi_name, mask_bool in masks.items():
            if data.shape[:3] != mask_bool.shape:
                raise ValueError(
                    f"{fname}: spatial shape {data.shape[:3]} does not match "
                    f"{roi_name} mask shape {mask_bool.shape}")
            vals, keep_cols = _mask_extract(data, mask_bool)
            n_vox = vals.shape[1]

            with np.errstate(invalid='ignore'):
                mean_per_sub = (np.nanmean(vals, axis=1)
                                if n_vox > 0 else np.full(n_subj, np.nan))
            for s in range(n_subj):
                subject_records.append({
                    'subject': s, 'step': step, 'roi': roi_name,
                    'mean_beta': float(mean_per_sub[s]),
                })
            if n_vox > 0:
                t_mean, p_mean_2s = sstats.ttest_1samp(
                    mean_per_sub, 0.0, nan_policy='omit')
            else:
                t_mean, p_mean_2s = np.nan, np.nan

            svc = _svc_max_t(vals, n_perm=n_perm, seed=seed)
            loso, held_by_k = _loso_topk(
                vals, LOSO_K, n_perm=n_perm, seed_base=seed + 1000)
            for kk, arr in held_by_k.items():
                np.save(out_dir /
                        f'loso_held_out_{roi_name}_{step.replace(" ","")}_k{kk}.npy',
                        arr)
            cluster = _cluster_mass_signflip(
                vals, mask_bool, keep_cols,
                cft=CLUSTER_CFT, n_perm=cluster_n_perm, seed=seed + 5000)

            results[roi_name][step] = {
                'n_subj':            int(n_subj),
                'n_vox_in_mask':     int(mask_bool.sum()),
                'n_vox_with_signal': int(n_vox),
                'mean_beta':         float(np.nanmean(mean_per_sub)) if n_vox else np.nan,
                'sem_beta':          float(np.nanstd(mean_per_sub, ddof=1) /
                                           np.sqrt(n_subj)) if n_vox else np.nan,
                't_mean_beta':       float(t_mean) if np.isfinite(t_mean) else np.nan,
                'p_mean_beta_2s':    float(p_mean_2s) if np.isfinite(p_mean_2s) else np.nan,
                'svc':               svc,
                'loso':              loso,
                'cluster':           cluster,
            }

            if svc is not None:
                print(f"  {roi_name:5s}  mean β = "
                      f"{results[roi_name][step]['mean_beta']:+.4f}"
                      f" ± {results[roi_name][step]['sem_beta']:.4f}")
                print(f"         SVC peak    : t={svc['peak_t']:.2f}, "
                      f"p_FWE={svc['peak_p_FWE']:.4f}  {_stars(svc['peak_p_FWE'])}")
                if cluster is not None:
                    print(f"         Cluster mass: obs={cluster['obs_max_mass']:.1f} "
                          f"(cft={cluster['cft']}), "
                          f"p_FWE={cluster['p_FWE_cluster']:.4f}  "
                          f"{_stars(cluster['p_FWE_cluster'])}")
                for kk, v in loso.items():
                    print(f"         LOSO k={v['k']:>3d}  : held β={v['held_mean']:+.4f}"
                          f" ± {v['held_sem']:.4f}  t={v['t']:+.2f}  "
                          f"p_perm={v['p_perm_1s']:.4f}  {_stars(v['p_perm_1s'])}")
            else:
                print(f"  {roi_name:5s}  n_vox=0 (no signal in mask)")

    # Per-subject long-format CSV
    df_sub = pd.DataFrame(subject_records,
                          columns=['subject', 'step', 'roi', 'mean_beta'])
    df_sub.to_csv(out_dir / 'mean_beta_per_subject_step_roi.csv', index=False)

    # Pairwise step contrasts within each ROI
    n_pairs = n_steps * (n_steps - 1) // 2
    pairwise = {roi: [] for roi in rois}
    if df_sub.empty:
        print(f"\n[{label}] no subject data loaded — skipping pairwise contrasts.")
    else:
        for roi in rois:
            print(f"\n--- [{label}] pairwise (sign-flip, Bonferroni×{n_pairs}) — {roi} ---")
            for i, si in enumerate(steps):
                for j, sj in enumerate(steps):
                    if i >= j: continue
                    xi = (df_sub[(df_sub['roi']==roi) & (df_sub['step']==si)]
                          .sort_values('subject')['mean_beta'].to_numpy())
                    xj = (df_sub[(df_sub['roi']==roi) & (df_sub['step']==sj)]
                          .sort_values('subject')['mean_beta'].to_numpy())
                    if xi.size == 0 or xj.size == 0:
                        continue
                    t, p = _paired_signflip(xi, xj,
                                             n_perm=n_perm,
                                             seed=seed + i*10 + j)
                    p_bonf = min(p * n_pairs, 1.0) if np.isfinite(p) else np.nan
                    pairwise[roi].append({'a': si, 'b': sj, 't': t,
                                           'p_perm_2s': p, 'p_bonf': p_bonf})
                    print(f"  {si:8s} vs {sj:8s}  t={t:+.2f}  "
                          f"p_perm={p:.4f}  p_bonf={p_bonf:.4f}  {_stars(p_bonf)}")

    # Shape contrasts + ROI × contrast interaction
    shape_stats = {}
    contrasts = _shape_contrasts_for_nsteps(n_steps)
    if contrasts and not df_sub.empty:
        print(f"\n{'='*70}\n[{label}] SHAPE CONTRASTS  (unit-normalised)\n{'='*70}")
        per_roi = {}
        for roi in rois:
            per_roi[roi] = {}
            print(f"\n--- {roi} ---")
            for cname, cvec in contrasts.items():
                proj = _subject_contrast(df_sub, roi, steps, cvec)
                per_roi[roi][cname] = proj
                t, p = _signflip_1s(proj, n_perm, seed + 900 + hash(cname) % 100,
                                     direction='greater')
                shape_stats.setdefault(roi, {})[f'{cname}_gt0'] = {
                    't': t, 'p_perm_1s': p}
                print(f"  {cname:12s} > 0 :  t={t:+.2f}  "
                      f"p_perm(1s)={p:.4f}  {_stars(p)}")
            # within-ROI: which shape wins?
            names = list(contrasts.keys())
            if len(names) == 2:
                a, b = names
                t_d, p_d = _paired_signflip(per_roi[roi][a], per_roi[roi][b],
                                             n_perm=n_perm, seed=seed + 902)
                shape_stats[roi][f'{a}_minus_{b}'] = {'t': t_d, 'p_perm_2s': p_d}
                print(f"  {a} − {b}:  t={t_d:+.2f}  p_perm(2s)={p_d:.4f}  "
                      f"{_stars(p_d)}   (t>0 → '{a}' fits better)")

        # ROI × contrast interaction (only when both ROIs present)
        if len(rois) == 2 and set(rois) == {'mPFC', 'lOFC'}:
            print(f"\n--- [{label}] ROI × contrast interaction ---")
            for cname in contrasts:
                a = per_roi['mPFC'][cname]
                b = per_roi['lOFC'][cname]
                t_i, p_i = _paired_signflip(a, b, n_perm=n_perm,
                                             seed=seed + 950 + hash(cname) % 100)
                shape_stats.setdefault('interaction', {})[
                    f'{cname}_mPFC_vs_lOFC'] = {'t': t_i, 'p_perm_2s': p_i}
                print(f"  {cname:12s} (mPFC − lOFC):  t={t_i:+.2f}  "
                      f"p_perm(2s)={p_i:.4f}  {_stars(p_i)}")

    # Persist full stats
    full_out = {
        'per_step_per_roi': results,
        'pairwise':         pairwise,
        'shape_contrasts':  shape_stats,
        'settings': {
            'n_perm':         n_perm,
            'cluster_n_perm': cluster_n_perm,
            'cluster_cft':    CLUSTER_CFT,
            'seed':           seed,
            'loso_k':         LOSO_K,
            'base_dir':       str(base_dir),
            'rois':           rois,
            'steps':          steps,
            'files':          file_map,
        },
    }
    with open(out_dir / 'results_by_step_roi.json', 'w') as f:
        json.dump(full_out, f, indent=2, default=lambda o:
                  None if isinstance(o, float) and not np.isfinite(o) else o)
    print(f"\n[{label}] saved: {out_dir / 'results_by_step_roi.json'}")

    # ── Plots ────────────────────────────────────────────────────────
    if df_sub.empty:
        return full_out
    x_positions = np.arange(n_steps)
    offset = 0.07 if len(rois) == 2 else 0.0
    fig_w = max(6.0, 0.9 * n_steps + 2.0)

    # Mean β per (step × ROI) — SVC stars
    fig, ax = plt.subplots(figsize=(fig_w, 4.6), constrained_layout=True)
    for roi_i, roi in enumerate(rois):
        dx = (offset if roi_i else -offset) if len(rois) == 2 else 0.0
        sub_ids = sorted(df_sub[df_sub.roi == roi]['subject'].unique())
        for s in sub_ids:
            y = [df_sub[(df_sub.roi==roi) & (df_sub.step==step) & (df_sub.subject==s)]
                 ['mean_beta'].mean() for step in steps]
            ax.plot(x_positions + dx, y, color=ROI_COLOURS[roi],
                    alpha=0.15, lw=0.7)
        means = [results[roi][step]['mean_beta']
                 if step in results[roi] else np.nan for step in steps]
        sems  = [results[roi][step]['sem_beta']
                 if step in results[roi] else np.nan for step in steps]
        ax.errorbar(x_positions + dx, means, yerr=sems,
                    marker='o', ms=7, lw=2.2, capsize=4,
                    color=ROI_COLOURS[roi], label=roi, zorder=5)
        for i, step in enumerate(steps):
            svc = results[roi].get(step, {}).get('svc')
            if svc is not None:
                s = _stars(svc['peak_p_FWE'])
                m = means[i] if np.isfinite(means[i]) else 0
                sem = sems[i] if np.isfinite(sems[i]) else 0
                ax.text(x_positions[i] + dx, m + sem + 0.001, s,
                        ha='center', va='bottom', fontsize=9,
                        color=ROI_COLOURS[roi], fontweight='bold')
    ax.axhline(0, color='k', lw=0.8, ls='--', zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(steps, fontsize=10, rotation=25, ha='right')
    ax.set_xlabel('Future step', fontsize=11)
    ax.set_ylabel('mean β within ROI  (per subject)', fontsize=11)
    ax.set_title(f'[{label}]  Future-step dominance\n'
                 '(stars = SVC max-t sign-flip p_FWE within ROI)',
                 fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'future_step_dominance.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Significance heatmap (SVC / cluster / LOSO@k_min)
    LOSO_K_SHOW = LOSO_K[0]
    tests = [
        ('SVC max-t\np_FWE',           'svc',     'peak_p_FWE'),
        (f'Cluster mass\np_FWE (cft={CLUSTER_CFT})',
                                       'cluster', 'p_FWE_cluster'),
        (f'LOSO k={LOSO_K_SHOW}\np_perm',
                                       'loso',    (str(LOSO_K_SHOW), 'p_perm_1s')),
    ]
    fig, axes = plt.subplots(
        len(tests), 1, figsize=(fig_w, 1.6 * len(tests)),
        constrained_layout=True, sharex=True,
    )
    for ax, (ttl, key, path) in zip(axes, tests):
        H = np.full((len(rois), n_steps), np.nan)
        for i, roi in enumerate(rois):
            for j, step in enumerate(steps):
                node = results[roi].get(step, {}).get(key)
                if node is None: continue
                if isinstance(path, tuple):
                    sub_k, field = path
                    node = node.get(sub_k)
                    if node is None: continue
                    H[i, j] = node[field]
                else:
                    H[i, j] = node[path]
        im = ax.imshow(H, cmap='Reds_r', vmin=0, vmax=0.10, aspect='auto')
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                val = H[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f'{val:.3f}\n{_stars(val)}',
                            ha='center', va='center', fontsize=8,
                            color='white' if val < 0.03 else 'black')
        ax.set_yticks(range(len(rois))); ax.set_yticklabels(rois, fontsize=10)
        ax.set_title(ttl, fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    axes[-1].set_xticks(range(n_steps))
    axes[-1].set_xticklabels(steps, fontsize=9, rotation=25, ha='right')
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'significance_heatmap.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)

    # LOSO held-out β line (unbiased effect size)
    fig, ax = plt.subplots(figsize=(fig_w, 4.0), constrained_layout=True)
    for roi_i, roi in enumerate(rois):
        dx = (offset if roi_i else -offset) if len(rois) == 2 else 0.0
        means = []; sems = []; ps = []
        for step in steps:
            node = results[roi].get(step, {}).get('loso', {}).get(str(LOSO_K_SHOW))
            if node is None:
                means.append(np.nan); sems.append(np.nan); ps.append(np.nan)
            else:
                means.append(node['held_mean'])
                sems.append(node['held_sem'])
                ps.append(node['p_perm_1s'])
        ax.errorbar(x_positions + dx, means, yerr=sems,
                    marker='o', ms=7, lw=2.2, capsize=4,
                    color=ROI_COLOURS[roi], label=roi, zorder=5)
        for i, (m, sem, p) in enumerate(zip(means, sems, ps)):
            if np.isfinite(p):
                ax.text(x_positions[i] + dx, (m or 0) + (sem or 0) + 0.001,
                        _stars(p), ha='center', va='bottom', fontsize=9,
                        color=ROI_COLOURS[roi], fontweight='bold')
    ax.axhline(0, color='k', lw=0.8, ls='--', zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(steps, fontsize=10, rotation=25, ha='right')
    ax.set_xlabel('Future step', fontsize=11)
    ax.set_ylabel(f'LOSO held-out β  (top-{LOSO_K_SHOW} vox picked on n-1 subj)',
                  fontsize=10)
    ax.set_title(f'[{label}]  Unbiased LOSO cross-validated effect size\n'
                 '(stars = sign-flip p on held-out means)', fontsize=10)
    ax.legend(frameon=False, fontsize=10, loc='best')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'loso_held_out_k{LOSO_K_SHOW}.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)

    return full_out


# ── Main entry ───────────────────────────────────────────────────────
if __name__ == '__main__':
    for cfg in DATASETS:
        if cfg['label'] not in DATASETS_TO_RUN:
            print(f"[skip] dataset '{cfg['label']}' not in DATASETS_TO_RUN")
            continue
        run_dataset(cfg)
    print(f"\nAll dataset outputs under: {OUT_DIR}")
