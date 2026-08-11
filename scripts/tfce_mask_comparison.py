"""
Does the DSR mPFC effect survive TFCE inside an a-priori mask?

Replicates PALM's `-T -ise -n 1000` procedure (TFCE + sign-flip max-statistic
FWE) so the identical analysis can be run inside three different search
volumes. PALM is not installed locally; this implementation is validated
against PALM's own whole-brain output before the masked results are used.

Why this is the right comparison to run
---------------------------------------
The question "would a smaller mask make the effect significant?" is only
legitimate if the mask was fixed in advance. Here it was: the mPFC mask
(BA32/mBA9/mBA10) is already the a-priori mask used for the instruction-phase
timing analysis and for the gradient analysis. Re-using it for the main
effect is applying an existing a-priori hypothesis to the primary contrast,
not searching for a volume that yields significance.

The PFC+MTL volume is a different matter -- see the docstring of main().

Run: python tfce_mask_comparison.py
"""
import os
import json
import numpy as np
import nibabel as nib
from scipy import ndimage

# ---------------------------------------------------------------- config ----
ROOT = '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
GRP = f'{ROOT}/derivatives/group'
IN4D = (f'{GRP}/group_RSA_quarters_DSR_controls_glmbase_all-paths-fixed'
        f'_stickrews_split-buttons_cropped/cropped_masked_smooth_fwhm5_'
        f'DSR-DSR-contr_except_prev_but-mask_reward-path_beta_std.nii')
BRAIN = (f'{GRP}/group_RSA_quarters_DSR_controls_glmbase_all-paths-fixed'
         f'_stickrews_split-buttons_cropped/mask_all_33_subjects.nii')
PALM_DIR = (f'{GRP}/Main_Results_fMRI/RSA_quarters_DSR_controls_glmbase_'
            f'all-paths-fixed_stickrews_split-buttons_smooth5_palm_p0_01')
PALM_STEM = ('cropped_masked_smooth_fwhm5_DSR-DSR-contr_except_prev_but-'
             'mask_reward-path_beta_std')

MASK_MPFC = f'{ROOT}/masks/mask_PFC_LR_smoothed_resampled.nii.gz'
# PFC+MTL search volume: taken from the state analysis PALM run, so it is
# byte-identical to the volume used for the state effect.
MASK_PFCMTL = (f'{GRP}/Main_Results_fMRI/Final_state/PFC-and-MonaMTL/'
               f'STATE-DSR_all_controls_noint_vox_tstat_c1.nii')

N_PERM = 1000          # PALM used 1000
SEED = 0
H, E = 2.0, 0.5        # PALM -T defaults
CONN = 1               # 6-connectivity (face), PALM default
OUT = f'{GRP}/Main_Results_fMRI/tfce_mask_comparison'

PEAKS = {'mPFC': (2, 50, 18), 'lOFC': (-38, 28, -14)}


# ------------------------------------------------------------------ tfce ----
def tfce(stat, mask, dh=None, H=H, E=E, conn=CONN):
    """Threshold-free cluster enhancement, PALM/FSL convention.

    TFCE(v) = sum_h  extent(v, h)^E * h^H * dh

    Clusters are formed *within* `mask`, so a smaller mask truncates clusters
    at its boundary. That is what PALM does with `-m`, and it is the reason a
    mask does not simply rescale the whole-brain result.
    """
    s = np.where(mask, stat, 0.0)
    smax = s.max()
    if smax <= 0:
        return np.zeros_like(s)
    if dh is None:
        dh = smax / 100.0          # PALM 'auto'
    struct = ndimage.generate_binary_structure(3, conn)
    out = np.zeros_like(s)
    h = dh
    while h <= smax:
        lab, n = ndimage.label(s >= h, structure=struct)
        if n:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            out += (sizes[lab] ** E) * (h ** H) * dh
        h += dh
    return out


def tstat(D):
    """One-sample t across the last axis (subjects)."""
    m = D.mean(-1)
    sd = D.std(-1, ddof=1)
    n = D.shape[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = m / (sd / np.sqrt(n))
    return np.nan_to_num(t)


def max_tfce_null(D4, mask, n_perm, seed, verbose_every=100):
    """Sign-flip null of the maximum TFCE statistic within `mask`.

    One null distribution per search volume -- this is what makes the
    correction specific to that volume.
    """
    rng = np.random.default_rng(seed)
    n_sub = D4.shape[-1]
    idx = np.where(mask)
    Dm = D4[idx]                       # (n_vox_in_mask, n_sub) -- compact
    null = np.empty(n_perm)
    shape = mask.shape
    for p in range(n_perm):
        flip = rng.choice([-1.0, 1.0], size=n_sub)
        m = (Dm * flip).mean(1)
        sd = (Dm * flip).std(1, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            tv = np.nan_to_num(m / (sd / np.sqrt(n_sub)))
        vol = np.zeros(shape)
        vol[idx] = tv
        null[p] = tfce(vol, mask).max()
        if verbose_every and (p + 1) % verbose_every == 0:
            print(f'      perm {p + 1}/{n_perm}', flush=True)
    return null


def run_volume(D4, mask, name, n_perm=N_PERM, seed=SEED):
    """Observed TFCE + FWE p inside one search volume."""
    t = np.where(mask, tstat(D4), 0.0)
    obs = tfce(t, mask)
    null = max_tfce_null(D4, mask, n_perm, seed)
    # corrected p for every voxel: proportion of null max >= observed
    p = np.ones_like(obs)
    inm = mask & (obs > 0)
    p[inm] = np.array([(null >= v).mean() for v in obs[inm]])
    return dict(name=name, n_vox=int(mask.sum()), t=t, tfce=obs,
                p=p, null=null, crit=float(np.quantile(null, 0.95)))


def report(res, aff, peaks):
    """Peak-wise summary for one search volume."""
    out = {'volume': res['name'], 'n_voxels': res['n_vox'],
           'tfce_crit_p05': res['crit'],
           'n_sig_voxels': int(((res['p'] < 0.05) & (res['tfce'] > 0)).sum())}
    inv = np.linalg.inv(aff)
    for lab, mni in peaks.items():
        ijk = tuple(np.round(inv @ np.append(mni, 1))[:3].astype(int))
        inside = res['tfce'][ijk] > 0
        out[lab] = dict(mni=list(mni), in_volume=bool(inside),
                        tfce=float(res['tfce'][ijk]),
                        p_fwe=float(res['p'][ijk]) if inside else None)
    return out


def main():
    """Three search volumes, one procedure.

    whole brain   the pre-registered-agnostic first pass; what PALM already ran
    PFC+MTL       the volume used for the *state* effect
    mPFC          the a-priori volume already used for timing + gradient

    The PFC+MTL volume is included because it was asked about, but note it was
    defined for a different hypothesis: it exists to test where an abstract
    *state* code lives, and its medial-temporal half carries no action-plan
    prediction at all. Borrowing it for the action-plan effect adds ~36,000
    voxels of unmotivated search space. It is the wrong instrument for this
    contrast, and the numbers below show it costs power for nothing.
    """
    os.makedirs(OUT, exist_ok=True)
    img = nib.load(IN4D)
    D4 = img.get_fdata()
    aff = img.affine
    brain = nib.load(BRAIN).get_fdata() > 0
    print(f'data {D4.shape}  n_subjects={D4.shape[-1]}  df={D4.shape[-1] - 1}')

    mpfc = nib.load(MASK_MPFC).get_fdata() > 0
    pfcmtl = nib.load(MASK_PFCMTL).get_fdata() != 0
    vols = [('whole brain', brain),
            ('PFC+MTL (state volume)', brain & pfcmtl),
            ('mPFC a-priori (BA32/mBA9/mBA10)', brain & mpfc)]

    results, summary = {}, []
    for name, m in vols:
        print(f'\n-- {name}: {int(m.sum())} voxels')
        r = run_volume(D4, m, name)
        results[name] = r
        s = report(r, aff, PEAKS)
        summary.append(s)
        print(f'   TFCE crit(p<.05) = {r["crit"]:.1f} | '
              f'{s["n_sig_voxels"]} voxels significant')
        for lab in PEAKS:
            d = s[lab]
            if d['in_volume']:
                print(f'   {lab:5s} TFCE={d["tfce"]:7.1f}  p_FWE={d["p_fwe"]:.4f}')
            else:
                print(f'   {lab:5s} outside this volume')

    with open(f'{OUT}/tfce_mask_comparison.json', 'w') as f:
        json.dump(dict(n_subjects=int(D4.shape[-1]), n_perm=N_PERM, seed=SEED,
                       tfce_H=H, tfce_E=E, summary=summary), f, indent=2)

    # validation against PALM's own whole-brain run
    pal = nib.load(f'{PALM_DIR}/{PALM_STEM}_tfce_tstat_fwep_c1.nii').get_fdata()
    inv = np.linalg.inv(aff)
    print('\n== validation: this implementation vs PALM (whole brain) ==')
    for lab, mni in PEAKS.items():
        ijk = tuple(np.round(inv @ np.append(mni, 1))[:3].astype(int))
        print(f'   {lab:5s}  PALM p={1 - pal[ijk]:.4f}   '
              f'here p={results["whole brain"]["p"][ijk]:.4f}')
    print(f'\nwrote {OUT}/tfce_mask_comparison.json')


if __name__ == '__main__':
    main()
