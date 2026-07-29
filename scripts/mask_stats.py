#!/usr/bin/env python
"""
Compare two correction methods on a-priori mask:
  1. Voxel-level FDR (Benjamini-Hochberg, q < 0.05)
  2. Permutation-based peak voxel FWE (sign-flip max-t, as in svc_loso_test.py)

Both applied to the joint voxel x TR search space.

Usage:
    python mask_stats.py --mask mask.nii.gz --model rewDSR --out out_tag
"""
import argparse, json, os, sys
import numpy as np
import nibabel as nib
from scipy import stats


def load_ref(root, tr0_dirname_pattern):
    tr0_dir = tr0_dirname_pattern.format(tr=0)
    ref_path = os.path.join(root, tr0_dir, "mask_all_32_subjects.nii")
    ref = nib.load(ref_path)
    return ref, ref.get_fdata() > 0


def load_mask(mask_path, ref):
    img = nib.load(mask_path)
    same_grid = (img.shape[:3] == ref.shape[:3] and
                 np.allclose(img.affine, ref.affine, atol=1e-3))
    if same_grid:
        m = img.get_fdata() > 0
    else:
        from nilearn.image import resample_img
        print(f"[info] resampling mask {img.shape[:3]} -> {ref.shape[:3]}",
              file=sys.stderr)
        m = resample_img(img, target_affine=ref.affine, target_shape=ref.shape,
                         interpolation="nearest").get_fdata() > 0
    return m


def extract_betas(root, tr_dirname_pattern, model, trs, mask, brain):
    """Return (n_subj, n_mask_vox, n_tr) array + ijk indices."""
    sel = mask & brain
    ix, iy, iz = np.where(sel)
    out = None
    for j, tr in enumerate(trs):
        d = tr_dirname_pattern.format(tr=tr)
        f = os.path.join(root, d, f"cropped_masked_smooth_fwhm5_{model}_beta_std.nii")
        img = nib.load(f)
        data = img.get_fdata()
        if data.ndim == 3:
            data = data[..., None]
        n_subj = data.shape[-1]
        if out is None:
            out = np.empty((n_subj, len(ix), len(trs)), dtype=np.float32)
        out[:, :, j] = data[ix, iy, iz, :].T
        print(f"[read] TR{tr}: {n_subj} subjects", file=sys.stderr)
    return out, (ix, iy, iz)


def tstat(A):
    """One-sample t-test: A shape (n_subj, ...) -> t per column."""
    n = A.shape[0]
    m = A.mean(0)
    s = A.std(0, ddof=1)
    return np.where(s > 0, m / np.where(s > 0, s, 1) * np.sqrt(n), 0.0)


def null_max_t(A, n_perm, seed, pblock=1000):
    """Sign-flip null: return max t observed per permutation."""
    rng = np.random.RandomState(seed)
    n = A.shape[0]
    S2 = (A ** 2).sum(0)
    out = np.empty(n_perm)
    for s0 in range(0, n_perm, pblock):
        b = min(pblock, n_perm - s0)
        F = rng.choice([-1.0, 1.0], size=(b, n))
        M = (F @ A) / n
        var = (S2[None, :] - n * M ** 2) / (n - 1)
        T = np.where(var > 0, M * np.sqrt(n) / np.sqrt(np.where(var > 0, var, 1)), 0.0)
        out[s0:s0 + b] = T.max(1)
    return out


def benjamini_hochberg(p_values, alpha=0.05):
    """Return boolean array of FDR-significant tests (and q-values)."""
    p = np.asarray(p_values, dtype=np.float64)
    N = len(p)
    idx = np.argsort(p)
    threshold = (np.arange(1, N + 1) / N) * alpha
    mask = p[idx] <= threshold
    if mask.any():
        k = np.where(mask)[0][-1]
        cutoff = p[idx[k]]
    else:
        cutoff = -1
    q = np.empty_like(p)
    for i, pi in enumerate(p[idx]):
        rank = np.sum(p <= pi)
        q[idx[i]] = pi * N / rank
    q = np.minimum.accumulate(q[::-1])[::-1]  # monotonize
    return q <= alpha, q


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/Users/xpsy1114/Documents/projects/multiple_clocks/"
                                       "data/derivatives/group/per_TR")
    ap.add_argument("--dir-pattern", default="group_RSA_instruction_per_TR_glmbase_01-TR{tr}_cropped")
    ap.add_argument("--mask", required=True)
    ap.add_argument("--model", default="rewDSR")
    ap.add_argument("--trs", default="0,1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--tag", default="mask", help="output file prefix")
    args = ap.parse_args()

    trs = [int(x) for x in args.trs.split(",")]
    os.makedirs(args.out, exist_ok=True)

    ref, brain = load_ref(args.root, args.dir_pattern)
    mask = load_mask(args.mask, ref)
    print(f"[info] mask voxels: {(mask & brain).sum()}", file=sys.stderr)

    D, ijk = extract_betas(args.root, args.dir_pattern, args.model, trs, mask, brain)
    n_subj, n_vox, n_tr = D.shape
    print(f"[info] betas shape: {D.shape}", file=sys.stderr)

    # Flatten to (n_subj, n_voxel*n_tr) for the permutation, but keep the 2-D
    # (n_vox, n_tr) t-map so the peak can be located by unravel_index.
    D_flat = D.reshape(n_subj, -1)
    Tobs = tstat(D)                       # (n_vox, n_tr)
    Tobs_flat = Tobs.reshape(-1)          # C-order: index = vox * n_tr + tr

    # ---- Method 1: Voxel-level FDR ----
    p_uncorr = 2 * (1 - stats.t.cdf(np.abs(Tobs_flat), n_subj - 1))
    sig_fdr, q_vals = benjamini_hochberg(p_uncorr, alpha=0.05)
    n_sig_fdr = sig_fdr.sum()

    # Peak location. Unravel on the 2-D t-map -- D is (subj, vox, TR), so a
    # C-order flat index divides by n_tr, NOT n_vox. Getting that backwards
    # returns the right peak VALUE with a wrong voxel and TR attached to it.
    vox_idx, tr_idx = np.unravel_index(np.argmax(Tobs), Tobs.shape)
    peak_idx = int(vox_idx) * n_tr + int(tr_idx)
    peak_t_fdr = float(Tobs[vox_idx, tr_idx])
    peak_p_fdr = float(p_uncorr[peak_idx])
    peak_q_fdr = float(q_vals[peak_idx])

    ix, iy, iz = ijk
    mni = nib.affines.apply_affine(ref.affine, [ix[vox_idx], iy[vox_idx], iz[vox_idx]])

    # ---- Method 2: Permutation-based peak voxel FWE ----
    nmt = null_max_t(D_flat, n_perm=args.n_perm, seed=args.seed)
    peak_p_perm = float((nmt >= np.abs(peak_t_fdr)).mean())
    tcrit_perm = float(np.percentile(nmt, 95))

    results = dict(
        mask_nvox=n_vox,
        n_tr=n_tr,
        peak_t=peak_t_fdr,
        peak_mni=[round(float(v)) for v in mni],
        peak_TR=int(tr_idx),
        fdr_method=dict(
            n_sig_voxTR=int(n_sig_fdr),
            peak_p_uncorr=float(peak_p_fdr),
            peak_q_FDR05=float(peak_q_fdr),
            sig_threshold_q=0.05
        ),
        perm_method=dict(
            peak_p_FWE=peak_p_perm,
            t_crit_FWE05=tcrit_perm,
            n_perm=args.n_perm
        )
    )

    # Timecourse inside FDR-significant voxels
    if n_sig_fdr > 0:
        sig_ix = np.where(sig_fdr)[0]
        tc = D_flat[:, sig_ix].reshape(n_subj, -1, n_tr).mean((0, 1))
        tc_sem = D_flat[:, sig_ix].reshape(n_subj, -1, n_tr).std(0, ddof=1).mean(0) / np.sqrt(n_subj)
        results["fdr_timecourse"] = dict(
            mean=[float(v) for v in tc],
            sem=[float(v) for v in tc_sem]
        )

    json.dump(results, open(os.path.join(args.out, f"{args.tag}_stats.json"), "w"), indent=2)

    # Print summary
    print(f"\n=== {args.tag} ===")
    print(f"Mask: {n_vox} voxels × {n_tr} TRs = {n_vox * n_tr} tests")
    print(f"Peak: t={peak_t_fdr:.2f} at TR{tr_idx} MNI={results['peak_mni']}")
    print(f"\nFDR voxel-level correction (α=0.05):")
    print(f"  n_significant: {n_sig_fdr} out of {n_vox*n_tr}")
    print(f"  peak p (uncorr): {peak_p_fdr:.4f}")
    print(f"  peak q (FDR):    {peak_q_fdr:.4f}  {'✓ PASS' if peak_q_fdr < 0.05 else '✗ FAIL'}")
    print(f"\nPermutation-based peak voxel (sign-flip max-t, {args.n_perm} perms):")
    print(f"  peak p_FWE:  {peak_p_perm:.4f}  {'✓ PASS' if peak_p_perm < 0.05 else '✗ FAIL'}")
    print(f"  t_crit .05:  {tcrit_perm:.2f}")

    print(f"\nResults saved to: {os.path.join(args.out, f'{args.tag}_stats.json')}")

if __name__ == "__main__":
    main()
