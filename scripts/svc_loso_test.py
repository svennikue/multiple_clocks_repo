#!/usr/bin/env python
"""
*** THIS IS THE GROUP TEST REPORTED FOR THE INSTRUCTION-PHASE RESULT. ***

Small-volume-corrected max-t permutation test + leave-one-subject-out (LOSO)
cross-validated timecourse, for an a-priori mask, over the per-TR
instruction-phase RSA beta maps.

The manuscript's instruction-phase numbers come from this script's
`{tag}_svc_summary.json` (peak t, peak TR, peak MNI, p_FWE, n_vox) and its
`{tag}_loso_results.json` (the cross-validated timecourse). The whole-brain
cluster-mass script that reads the same inputs is a different test and is
NOT what is reported — it has been archived to
`scripts/old/instruction_phase_alternatives/`.

Pipeline position
-----------------
  fMRI_run_RSA_instruction.py   per subject, per TR: searchlight RSA of the
                                instruction phase -> one beta map per model
                                per second per subject
  (group assembly)              per-subject maps smoothed (FWHM 5 mm),
                                masked to voxels present in all subjects,
                                stacked into
                                <root>/group_RSA_instruction_per_TR_glmbase_01-TR{tr}_cropped/
                                    cropped_masked_smooth_fwhm5_{model}_beta_std.nii  (X,Y,Z,n_subj)
                                    mask_all_32_subjects.nii
  THIS SCRIPT                   group inference inside an a-priori mask

Why max-t and not whole-brain cluster mass
------------------------------------------
Cluster-mass inference trades peak height against spatial extent: a strong
but FOCAL effect (a handful of voxels) scores worse than a weak diffuse one,
because cluster mass integrates over both. The a-priori hypothesis here is a
compact region (BA32 / medial BA9 / medial BA10) and the effect is expected
to be small in extent, so we correct only over that region's voxels x TRs
and read the peak directly.

Two questions, both by max-statistic sign-flip permutation (Nichols &
Holmes 2002). Under H0 the sign of a subject's whole beta map is arbitrary,
so each permutation multiplies every subject's map by +-1, recomputes the
group t-map, and keeps ONLY the largest t over all voxels AND all TRs. A
single null therefore corrects simultaneously for the spatial and the
temporal search — there is no separate correction over time.

  1. SVC peak test: is there a voxel x TR combination in this mask exceeding
     what sign-flipping predicts? The corrected p for any voxel x TR is the
     proportion of permutations whose max-t is at least its observed t.
     `t_crit_FWE05` is the 95th percentile of that null; the significant
     temporal window at the peak voxel is the set of TRs whose t exceeds it.
  2. LOSO cross-validated timecourse: choose the top-k voxels by their
     by-TR t-statistic using only n-1 subjects, then read out the HELD-OUT
     subject's mean beta in those voxels, repeating for every subject. This
     removes the circularity of selecting voxels on the same data one then
     averages, so the timecourse is an unbiased estimate of effect size
     rather than an inflated one, and its own permutation test stays valid.
     Run for several k so the shape can be checked for robustness to the
     selection size.

NOTE ON df: the group has 32 subjects, so the one-sample t has df = 31.
`tstat()` divides by the sample SD with ddof=1 and multiplies by sqrt(n).

Usage
-----
    conda activate env_multiple_clocks
    python svc_loso_test.py \
        --mask /path/to/candidate_mask.nii.gz \
        --model rewDSR \
        --n-perm 10000 \
        --cv --k 50,100,200 \
        --out-dir /path/to/output_folder \
        --tag my_new_mask

Outputs (into --out-dir, prefixed by --tag):
    {tag}_svc_summary.json    -- peak stats for this one mask
    {tag}_loso_k{K}.npy       -- per-subject held-out beta x TR, one per K
    {tag}_loso_results.json   -- per-K timecourse mean/sem/t/p_FWE

Runtime: ~2-4 min per mask on a laptop (10000 perms, dominated by reading
the 12 whole-brain beta niftis once).
"""
import argparse, json, os, sys
import numpy as np
import nibabel as nib


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
        print(f"[info] mask grid differs from data grid -- resampling "
              f"(nearest-neighbour) {img.shape[:3]} -> {ref.shape[:3]}",
              file=sys.stderr)
        m = resample_img(img, target_affine=ref.affine, target_shape=ref.shape,
                          interpolation="nearest").get_fdata() > 0
    return m


def extract_betas(root, tr_dirname_pattern, model, trs, mask, brain):
    """Return (n_subj, n_mask_vox, n_tr) array + the mask voxel ijk indices."""
    sel = mask & brain
    ix, iy, iz = np.where(sel)
    n_vox = len(ix)
    out = None
    for j, tr in enumerate(trs):
        d = tr_dirname_pattern.format(tr=tr)
        f = os.path.join(root, d, f"cropped_masked_smooth_fwhm5_{model}_beta_std.nii")
        img = nib.load(f)
        data = img.get_fdata()          # (X, Y, Z, n_subj) or (X, Y, Z) per-subj set
        if data.ndim == 3:
            data = data[..., None]
        n_subj = data.shape[-1]
        if out is None:
            out = np.empty((n_subj, n_vox, len(trs)), dtype=np.float32)
        out[:, :, j] = data[ix, iy, iz, :].T
        print(f"[read] TR{tr}: {f} -> {n_subj} subjects", file=sys.stderr)
    return out, (ix, iy, iz)


def tstat(A):
    """A: (n_subj, ...) -> per-column one-sample t (paired against 0)."""
    n = A.shape[0]
    m = A.mean(0)
    s = A.std(0, ddof=1)
    return np.where(s > 0, m / np.where(s > 0, s, 1) * np.sqrt(n), 0.0)


def null_max_t(A, n_perm, seed, pblock=1000):
    """Sign-flip max-t null over all columns of A (n_subj, n_cols).
    Chunked over permutations to bound memory; exploits that sign-flipping
    preserves sum(x^2), so no per-permutation full recompute is needed."""
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


def run_svc(D, ref, ijk, n_perm, seed):
    n, n_vox, n_tr = D.shape
    Tobs = tstat(D)
    nmt = null_max_t(D.reshape(n, -1), n_perm=n_perm, seed=seed)
    pk = np.unravel_index(np.argmax(Tobs), Tobs.shape)
    peak_t = float(Tobs[pk])
    peak_p = float((nmt >= peak_t).mean())
    ix, iy, iz = ijk
    vi = pk[0]
    mni = nib.affines.apply_affine(ref.affine, [ix[vi], iy[vi], iz[vi]])
    tcrit = float(np.percentile(nmt, 95))
    return dict(n_vox=n_vox, n_tr=n_tr, peak_t=peak_t, peak_p_FWE=peak_p,
                peak_TR=int(pk[1]), peak_mni=[round(float(v)) for v in mni],
                t_crit_FWE05=tcrit, n_supra_FWE05=int((Tobs >= tcrit).sum()),
                n_perm=n_perm, seed=seed)


def run_loso(D, k_values, n_perm, seed0):
    n, n_vox, n_tr = D.shape
    out = {}
    for k in k_values:
        k = min(k, n_vox)
        held = np.zeros((n, n_tr))
        for s in range(n):
            train = np.delete(np.arange(n), s)
            Ttr = tstat(D[train])                    # (vox, TR) from n-1 subjects
            top = np.argsort(-Ttr.max(1))[:k]        # spatial selection, TRAIN ONLY
            held[s] = D[s, top, :].mean(0)            # unbiased held-out time course
        t = tstat(held)
        nm = null_max_t(held, n_perm=n_perm, seed=seed0 + k)
        p = [float((nm >= v).mean()) for v in t]
        out[str(k)] = dict(
            k=k, mean=[float(v) for v in held.mean(0)],
            sem=[float(v) for v in held.std(0, ddof=1) / np.sqrt(n)],
            t=[float(v) for v in t], p_FWE=p, t_crit_FWE05=float(np.percentile(nm, 95)))
    return out, held if False else None  # held per-k saved by caller


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/Users/xpsy1114/Documents/projects/multiple_clocks/"
                                       "data/derivatives/group/per_TR",
                    help="dir holding group_RSA_instruction_per_TR_glmbase_01-TR{tr}_cropped/")
    ap.add_argument("--dir-pattern", default="group_RSA_instruction_per_TR_glmbase_01-TR{tr}_cropped")
    ap.add_argument("--mask", required=True, help="path to a-priori mask NIfTI")
    ap.add_argument("--model", default="rewDSR")
    ap.add_argument("--trs", default="0,1,2,3,4,5,6,7,8,9,10,11",
                    help="comma-separated TR indices to include")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv", action="store_true", help="also run LOSO cross-validation")
    ap.add_argument("--k", default="50,100,200", help="voxel-selection sizes for --cv")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--tag", default="mask")
    args = ap.parse_args()

    trs = [int(x) for x in args.trs.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    ref, brain = load_ref(args.root, args.dir_pattern)
    mask = load_mask(args.mask, ref)
    print(f"[info] mask voxels (in-brain): {(mask & brain).sum()}", file=sys.stderr)

    D, ijk = extract_betas(args.root, args.dir_pattern, args.model, trs, mask, brain)
    print(f"[info] betas shape: {D.shape} (n_subj, n_vox, n_tr)", file=sys.stderr)

    svc = run_svc(D, ref, ijk, n_perm=args.n_perm, seed=args.seed)
    svc_path = os.path.join(args.out_dir, f"{args.tag}_svc_summary.json")
    json.dump(svc, open(svc_path, "w"), indent=2)
    print(f"\n=== SVC peak test: {args.tag} ({svc['n_vox']} vox) ===")
    print(f"  peak t={svc['peak_t']:.2f} at TR{svc['peak_TR']} MNI={svc['peak_mni']}  "
          f"p_FWE={svc['peak_p_FWE']:.4f}  (t_crit .05 = {svc['t_crit_FWE05']:.2f})")
    print(f"  -> {svc_path}")

    if args.cv:
        k_values = [int(x) for x in args.k.split(",")]
        n = D.shape[0]
        loso_out = {}
        for k in k_values:
            kk = min(k, D.shape[1])
            held = np.zeros((n, D.shape[2]))
            for s in range(n):
                train = np.delete(np.arange(n), s)
                Ttr = tstat(D[train])
                top = np.argsort(-Ttr.max(1))[:kk]
                held[s] = D[s, top, :].mean(0)
            t = tstat(held)
            nm = null_max_t(held, n_perm=args.n_perm, seed=args.seed + 100 + kk)
            p = [float((nm >= v).mean()) for v in t]
            loso_out[str(kk)] = dict(
                k=kk, mean=[float(v) for v in held.mean(0)],
                sem=[float(v) for v in held.std(0, ddof=1) / np.sqrt(n)],
                t=[float(v) for v in t], p_FWE=p,
                t_crit_FWE05=float(np.percentile(nm, 95)))
            np.save(os.path.join(args.out_dir, f"{args.tag}_loso_k{kk}.npy"), held)
            print(f"\n=== LOSO cross-validated, k={kk} voxels ===")
            print(f"  mean beta : {np.round(loso_out[str(kk)]['mean'], 4)}")
            print(f"  t         : {np.round(loso_out[str(kk)]['t'], 2)}")
            print(f"  p_FWE     : {np.round(p, 4)}  (t_crit={loso_out[str(kk)]['t_crit_FWE05']:.2f})")
        loso_path = os.path.join(args.out_dir, f"{args.tag}_loso_results.json")
        json.dump(loso_out, open(loso_path, "w"), indent=2)
        print(f"\n  -> {loso_path}")


if __name__ == "__main__":
    main()
