#!/usr/bin/env python
"""
Batch version of `scripts/svc_loso_test.py`: the SAME group test (small-volume
corrected max-t sign-flip permutation over voxels x TRs, plus the LOSO
cross-validated timecourse), run over EVERY model/contrast map of one per-TR
RSA output folder, in several a-priori masks.

Nothing about the statistics changes -- `tstat`, `null_max_t` and the LOSO
selection are imported from `svc_loso_test.py`, so empirical and permutation
values come from the identical code path as the reported instruction-phase
result.

What this adds over the single-map script
-----------------------------------------
1. It discovers all `cropped_masked_smooth_fwhm5_{model}_beta_std.nii[.gz]`
   maps in the TR0 folder and loops over them.
2. Each model's 12 TR niftis are read ONCE and the union of all requested
   mask voxels is extracted; every mask is then a column subset of that
   single read (the read is the expensive part).
3. Both signs are reported. The reported instruction test is one-sided
   positive (max over t). Several of the maps here are regressors inside a
   multiple-regression combo model, where a negative effect is interpretable,
   so the negative peak is reported alongside. Sign-flipping makes the null of
   max(-t) identical to the null of max(t), so the SAME null is reused --
   `peak_p_FWE_neg` is a second one-sided test, NOT corrected for having
   looked in both directions. Treat the positive column as the direct
   replication of the reported test and the negative column as descriptive.

Multiple comparisons across the 27 maps are NOT corrected here (by request):
each map's `peak_p_FWE` is corrected over its own voxels x TRs only. The
family size is written into the settings json and printed in the summary.

Usage
-----
    conda activate env_multiple_clocks
    python svc_loso_batch.py \
        --root .../derivatives/group/per_TR \
        --dir-pattern group_RSA_instr_test_full_glmbase_01-TR{tr}_cropped \
        --mask mPFC=.../mask_PFC_LR_smoothed_resampled.nii.gz \
        --mask MTL=.../Garvert_MTL_2mm.nii.gz \
        --mask visual=.../visual_occipital_HO25_2mm.nii.gz \
        --n-perm 10000 --k 50,100,200 \
        --out-dir .../per_TR_svc_instr_test_full_<date>

Outputs (into --out-dir):
    settings.json                      every parameter + the resolved inputs
    {mask}/{model}_svc_summary.json    peak stats, both signs, per-TR traces
    {mask}/{model}_loso_results.json   LOSO timecourse per k
    {mask}/{model}_loso_k{K}.npy       per-subject held-out beta x TR
    summary_table.csv                  one row per (mask, model)
"""
import argparse, csv, glob, json, os, re, sys, time
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svc_loso_test import load_mask, resolve_nii, tstat, null_max_t

FILE_RE = re.compile(r"^cropped_masked_smooth_fwhm5_(.+)_beta_std\.nii(\.gz)?$")


def load_ref(root, dir_pattern, trs):
    """Reference image + brain mask = voxels present in all subjects at EVERY
    included TR. `svc_loso_test.load_ref` reads TR0's mask only; the group
    masks differ by a handful of voxels between TRs, and a voxel entering the
    max-t search must be valid at every TR that search runs over."""
    ref, brain = None, None
    for tr in trs:
        p = resolve_nii(os.path.join(root, dir_pattern.format(tr=tr),
                                     "mask_all_32_subjects.nii"))
        img = nib.load(p)
        m = img.get_fdata() > 0
        ref, brain = (img, m) if ref is None else (ref, brain & m)
    return ref, brain


def discover_models(root, dir_pattern, tr0):
    d = os.path.join(root, dir_pattern.format(tr=tr0))
    models = []
    for f in sorted(os.listdir(d)):
        m = FILE_RE.match(f)
        if m:
            models.append(m.group(1))
    return models


def read_model_union(root, dir_pattern, model, trs, union_ijk):
    """(n_subj, n_union_vox, n_tr) -- one pass over the 12 TR niftis."""
    ix, iy, iz = union_ijk
    out = None
    for j, tr in enumerate(trs):
        f = resolve_nii(os.path.join(root, dir_pattern.format(tr=tr),
                                     f"cropped_masked_smooth_fwhm5_{model}_beta_std.nii"))
        data = nib.load(f).get_fdata()
        if data.ndim == 3:
            data = data[..., None]
        if out is None:
            out = np.empty((data.shape[-1], len(ix), len(trs)), dtype=np.float32)
        out[:, :, j] = data[ix, iy, iz, :].T
        del data
    return out


def adaptive_pblock(n_cols, max_elements=2e7):
    return int(max(1, min(1000, max_elements // max(n_cols, 1))))


def svc_both_signs(D, ref, ijk, n_perm, seed):
    """Identical to svc_loso_test.run_svc, plus the negative-direction peak
    (same sign-flip null, which is symmetric) and the per-TR traces."""
    n, n_vox, n_tr = D.shape
    Tobs = tstat(D)                                    # (n_vox, n_tr)
    flat = D.reshape(n, -1)
    nmt = null_max_t(flat, n_perm=n_perm, seed=seed,
                     pblock=adaptive_pblock(flat.shape[1]))
    tcrit = float(np.percentile(nmt, 95))
    ix, iy, iz = ijk

    def _peak(T):
        pk = np.unravel_index(np.argmax(T), T.shape)
        mni = nib.affines.apply_affine(ref.affine, [ix[pk[0]], iy[pk[0]], iz[pk[0]]])
        return pk, float(T[pk]), [int(round(float(v))) for v in mni]

    pk_p, t_p, mni_p = _peak(Tobs)
    pk_n, t_n, mni_n = _peak(-Tobs)
    return dict(
        n_vox=int(n_vox), n_tr=int(n_tr), n_subj=int(n),
        peak_t=t_p, peak_p_FWE=float((nmt >= t_p).mean()),
        peak_TR=int(pk_p[1]), peak_mni=mni_p,
        peak_t_neg=float(-t_n), peak_p_FWE_neg=float((nmt >= t_n).mean()),
        peak_TR_neg=int(pk_n[1]), peak_mni_neg=mni_n,
        t_crit_FWE05=tcrit,
        n_supra_FWE05=int((Tobs >= tcrit).sum()),
        n_supra_FWE05_neg=int((-Tobs >= tcrit).sum()),
        peak_voxel_t_by_TR=[float(v) for v in Tobs[pk_p[0], :]],
        max_t_by_TR=[float(v) for v in Tobs.max(0)],
        n_perm=int(n_perm), seed=int(seed))


def loso(D, k_values, n_perm, seed):
    """Identical selection/readout to svc_loso_test.main()'s --cv branch."""
    n, n_vox, n_tr = D.shape
    out, held_by_k = {}, {}
    for k in k_values:
        kk = min(k, n_vox)
        held = np.zeros((n, n_tr))
        for s in range(n):
            train = np.delete(np.arange(n), s)
            Ttr = tstat(D[train])
            top = np.argsort(-Ttr.max(1))[:kk]
            held[s] = D[s, top, :].mean(0)
        t = tstat(held)
        nm = null_max_t(held, n_perm=n_perm, seed=seed + 100 + kk)
        out[str(kk)] = dict(
            k=int(kk), mean=[float(v) for v in held.mean(0)],
            sem=[float(v) for v in held.std(0, ddof=1) / np.sqrt(n)],
            t=[float(v) for v in t],
            p_FWE=[float((nm >= v).mean()) for v in t],
            t_crit_FWE05=float(np.percentile(nm, 95)))
        held_by_k[kk] = held
    return out, held_by_k


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/Users/xpsy1114/Documents/projects/multiple_clocks/"
                                      "data/derivatives/group/per_TR")
    ap.add_argument("--dir-pattern",
                    default="group_RSA_instr_test_full_glmbase_01-TR{tr}_cropped")
    ap.add_argument("--mask", action="append", required=True,
                    help="name=path, repeatable")
    ap.add_argument("--models", default="", help="comma-separated; default = all found")
    ap.add_argument("--trs", default="0,1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", default="50,100,200")
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    np.random.seed(42)
    trs = [int(x) for x in args.trs.split(",")]
    k_values = [int(x) for x in args.k.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    ref, brain = load_ref(args.root, args.dir_pattern, trs)
    masks = {}
    for spec in args.mask:
        name, path = spec.split("=", 1)
        m = load_mask(path, ref) & brain
        masks[name] = dict(path=path, bool=m, n_vox=int(m.sum()))
        print(f"[mask] {name:8s} {m.sum():6d} in-brain voxels  ({path})", file=sys.stderr)

    union = np.zeros(brain.shape, bool)
    for v in masks.values():
        union |= v["bool"]
    union_ijk = np.where(union)
    lin = {}
    for name, v in masks.items():
        # column index of this mask's voxels within the union extraction
        lin[name] = np.flatnonzero(v["bool"][union_ijk])
    print(f"[info] union of masks: {union.sum()} voxels", file=sys.stderr)

    models = ([m for m in args.models.split(",") if m] if args.models
              else discover_models(args.root, args.dir_pattern, trs[0]))
    print(f"[info] {len(models)} maps x {len(masks)} masks x {len(trs)} TRs", file=sys.stderr)

    json.dump(dict(root=args.root, dir_pattern=args.dir_pattern, trs=trs,
                   masks={k: dict(path=v["path"], n_vox_in_brain=v["n_vox"])
                          for k, v in masks.items()},
                   models=models, n_models=len(models), n_perm=args.n_perm,
                   brain_mask_note=("group brain mask = intersection of "
                                    "mask_all_32_subjects over all included TRs "
                                    f"({int(brain.sum())} voxels)"),
                   seed=args.seed, k_values=k_values, cv=not args.no_cv,
                   numpy_seed=42,
                   multiple_comparison_note=(
                       "peak_p_FWE is corrected over voxels x TRs WITHIN each mask "
                       f"only. {len(models)} maps x {len(masks)} masks were tested; "
                       "no correction across that family was applied (by request)."),
                   negative_direction_note=(
                       "peak_p_FWE_neg reuses the same sign-flip null (symmetric) as a "
                       "second one-sided test; it is not corrected for testing both signs."),
                   ), open(os.path.join(args.out_dir, "settings.json"), "w"), indent=2)

    for name in masks:
        os.makedirs(os.path.join(args.out_dir, name), exist_ok=True)

    rows = []
    for mi, model in enumerate(models, 1):
        t0 = time.time()
        Du = read_model_union(args.root, args.dir_pattern, model, trs, union_ijk)
        for name, v in masks.items():
            cols = lin[name]
            D = np.ascontiguousarray(Du[:, cols, :])
            ijk = tuple(a[cols] for a in union_ijk)
            svc = svc_both_signs(D, ref, ijk, n_perm=args.n_perm, seed=args.seed)
            svc["model"], svc["mask"] = model, name
            json.dump(svc, open(os.path.join(args.out_dir, name,
                                             f"{model}_svc_summary.json"), "w"), indent=2)
            row = dict(mask=name, model=model, n_vox=svc["n_vox"],
                       peak_t=round(svc["peak_t"], 3), peak_TR=svc["peak_TR"],
                       peak_mni="/".join(str(c) for c in svc["peak_mni"]),
                       p_FWE=svc["peak_p_FWE"], t_crit_FWE05=round(svc["t_crit_FWE05"], 3),
                       n_supra=svc["n_supra_FWE05"],
                       peak_t_neg=round(svc["peak_t_neg"], 3), peak_TR_neg=svc["peak_TR_neg"],
                       peak_mni_neg="/".join(str(c) for c in svc["peak_mni_neg"]),
                       p_FWE_neg=svc["peak_p_FWE_neg"], n_supra_neg=svc["n_supra_FWE05_neg"])
            if not args.no_cv:
                lo, held_by_k = loso(D, k_values, n_perm=args.n_perm, seed=args.seed)
                for kk, held in held_by_k.items():
                    np.save(os.path.join(args.out_dir, name, f"{model}_loso_k{kk}.npy"), held)
                json.dump(lo, open(os.path.join(args.out_dir, name,
                                                f"{model}_loso_results.json"), "w"), indent=2)
                kref = str(min(k_values, key=lambda x: abs(x - 100)))
                kref = kref if kref in lo else list(lo)[0]
                tvec = np.array(lo[kref]["t"])
                row.update(loso_k=int(lo[kref]["k"]),
                           loso_peak_t=round(float(tvec.max()), 3),
                           loso_peak_TR=int(tvec.argmax()),
                           loso_p_FWE=lo[kref]["p_FWE"][int(tvec.argmax())])
            rows.append(row)
            flag = "*" if svc["peak_p_FWE"] < 0.05 else " "
            print(f"{flag} [{mi:2d}/{len(models)}] {name:7s} {model:42s} "
                  f"t={svc['peak_t']:5.2f} TR{svc['peak_TR']:<2d} "
                  f"p={svc['peak_p_FWE']:.4f}   (neg t={svc['peak_t_neg']:6.2f} "
                  f"p={svc['peak_p_FWE_neg']:.4f})", flush=True)
        del Du
        print(f"    -- {model} done in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)

    csv_path = os.path.join(args.out_dir, "summary_table.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {csv_path}")


if __name__ == "__main__":
    main()
