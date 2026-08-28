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
    {mask}/{model}_t.nii.gz            observed t-map (3-D, or 4-D over TRs)
    {mask}/{model}_voxelFWEp.nii.gz    voxel-wise FWE p against the max-t null
    {mask}/{model}_voxel1minusFWEp.nii.gz      1-p; threshold 0.95 for p<.05
    {mask}/{model}_voxel1minusFWEp_neg.nii.gz  same, negative direction
    {mask}/{model}_null_max_t.npy      the permutation max-t null itself
    {mask}/{model}_svc_summary.json    peak stats, both signs, per-TR traces
    {mask}/{model}_loso_results.json   LOSO timecourse per k
    {mask}/{model}_loso_k{K}.npy       per-subject held-out beta x TR
    summary_table.csv                  one row per (mask, model)

With --wholebrain, additionally (per model, for visualisation):
    wholebrain/{model}_t.nii.gz                observed t, 4-D over TRs
    wholebrain/{model}_1minusFWEp.nii.gz       1-p, FWE over whole brain x TRs
    wholebrain/{model}_1minusp_uncorr.nii.gz   1-p, uncorrected
    wholebrain/{model}_summary.json            peak stats
    wholebrain_summary_table.csv               one row per model
The whole-brain FWE null corrects over the entire brain mask and every
included TR at once, so it is much stricter than the small-volume p above and
the two must not be compared. The uncorrected map is for scrolling around the
brain, not for claims.
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
    """Permutations per block, so one block stays near `max_elements` floats."""
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
    return Tobs, nmt, dict(
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


def voxel_fwe_p(T, nmt):
    """Voxel-wise FWE p from the max-t null: p(v) = fraction of permutations
    whose max t is >= this voxel's observed t. Same null, same convention as
    the archived cluster script's `_voxelFWEp` maps."""
    srt = np.sort(nmt)
    return ((len(srt) - np.searchsorted(srt, T, side="left")) / len(srt)).astype(np.float32)


def vol_from_cols(vals, fill, ref, ijk, n_tr):
    """Scatter a (n_vox, n_tr) column array back into image space.

    3-D when a single TR was analysed, 4-D (X, Y, Z, TR) when several were --
    so the TR slider in fsleyes scrubs through the instruction period."""
    ix, iy, iz = ijk
    v = np.full(ref.shape[:3] + (n_tr,), fill, dtype=np.float32)
    v[ix, iy, iz, :] = vals
    return v[..., 0] if n_tr == 1 else v


def write_wholebrain_maps(out_dir, model, Tobs, p_fwe, p_unc, p_unc_neg, nmt,
                          ref, ijk, n_tr):
    """Whole-brain volumes for visualisation: t, and both p images in the FSL
    1-p convention (threshold at 0.95 for p < .05, at 0.999 for p < .001).

    `_1minusFWEp` is corrected over the whole brain x all TRs -- far stricter
    than the small-volume p in the mask folders. `_1minusp_uncorr` is NOT
    corrected for anything; it is there to zoom around the brain, not to make
    claims. With 1000 permutations its floor is p = 0.001.

    A voxel no permutation ever beat gets p = 0 (so 1-p = 1); read that as
    "p < 1/n_perm", the resolution limit, not as a real zero."""
    hdr = ref.header.copy()
    vols = [("t", Tobs, 0.0),
            ("1minusFWEp", 1.0 - p_fwe, 0.0),
            ("1minusp_uncorr", 1.0 - p_unc, 0.0)]
    if p_unc_neg is not None:
        vols += [("1minusFWEp_neg", 1.0 - voxel_fwe_p(-Tobs, nmt), 0.0),
                 ("1minusp_uncorr_neg", 1.0 - p_unc_neg, 0.0)]
    for suffix, vals, fill in vols:
        nib.save(nib.Nifti1Image(vol_from_cols(vals, fill, ref, ijk, n_tr),
                                 ref.affine, hdr),
                 os.path.join(out_dir, f"{model}_{suffix}.nii.gz"))
    np.save(os.path.join(out_dir, f"{model}_null_max_t.npy"), nmt.astype(np.float32))


def write_maps(out_dir, model, Tobs, nmt, ref, ijk, n_tr):
    """Write the observed t-map and its voxel-wise FWE maps, both signs.

    3-D when one TR is analysed, 4-D (X, Y, Z, TR) when several are -- open in
    fsleyes and scrub the TR slider. Everything outside the mask is 0 in the
    t-map and p = 1 (so 1-p = 0) in the p-maps. Threshold the
    `_voxel1minusFWEp` maps at 0.95 for p_FWE < .05."""
    def _vol(vals, fill):
        return vol_from_cols(vals, fill, ref, ijk, n_tr)

    p_pos = voxel_fwe_p(Tobs, nmt)
    p_neg = voxel_fwe_p(-Tobs, nmt)
    hdr = ref.header.copy()
    for suffix, vals, fill in (
            ("t",                   Tobs,        0.0),
            ("voxelFWEp",           p_pos,       1.0),
            ("voxel1minusFWEp",     1.0 - p_pos, 0.0),
            ("voxel1minusFWEp_neg", 1.0 - p_neg, 0.0)):
        nib.save(nib.Nifti1Image(_vol(vals, fill), ref.affine, hdr),
                 os.path.join(out_dir, f"{model}_{suffix}.nii.gz"))
    np.save(os.path.join(out_dir, f"{model}_null_max_t.npy"), nmt.astype(np.float32))


def perm_wholebrain(A, T_obs, n_perm, seed, pblock, want_neg=False):
    """Sign-flip permutation over every column of A, keeping BOTH the max-t
    null (for FWE) and a per-column tally of exceedances (for uncorrected p).

    The per-permutation t is computed with exactly the arithmetic of
    `svc_loso_test.null_max_t` -- sign-flipping preserves sum(x^2), so
    var = (S2 - n*M^2)/(n-1) -- which is algebraically the same one-sample t
    as `tstat` computes for the observed map. That identity is asserted below
    on the all-plus-one flip, so the permutation statistic and the empirical
    statistic are verifiably the same statistic (CLAUDE.md rule 4).

    Only the tallies are kept, never the (n_perm, n_vox) null itself, so
    memory stays at `pblock` permutations regardless of n_perm.

    Returns (null_max_t, count_ge, count_le) -- counts of permutations whose t
    at that column was >= (resp. <=) the observed t. count_le is None unless
    `want_neg`."""
    n, n_cols = A.shape
    S2 = (A ** 2).sum(0)

    def _t_from_flips(F):
        M = (F @ A) / n
        var = (S2[None, :] - n * M ** 2) / (n - 1)
        return np.where(var > 0, M * np.sqrt(n) / np.sqrt(np.where(var > 0, var, 1)), 0.0)

    # identity flip must reproduce the observed t-map exactly
    assert np.allclose(_t_from_flips(np.ones((1, n)))[0], T_obs, atol=1e-4), \
        "permutation t and empirical tstat disagree on the identity sign-flip"

    rng = np.random.RandomState(seed)
    null_max = np.empty(n_perm)
    cnt_ge = np.zeros(n_cols, dtype=np.int32)
    cnt_le = np.zeros(n_cols, dtype=np.int32) if want_neg else None
    for s0 in range(0, n_perm, pblock):
        b = min(pblock, n_perm - s0)
        T = _t_from_flips(rng.choice([-1.0, 1.0], size=(b, n)))
        null_max[s0:s0 + b] = T.max(1)
        cnt_ge += (T >= T_obs[None, :]).sum(0, dtype=np.int32)
        if want_neg:
            cnt_le += (T <= T_obs[None, :]).sum(0, dtype=np.int32)
    return null_max, cnt_ge, cnt_le


def run_wholebrain(D, ref, ijk, n_perm, seed, want_neg=False):
    """Whole-brain t + FWE + uncorrected p, over voxels x TRs jointly.

    FWE corrects over the WHOLE brain mask and all included TRs at once (max-t
    null, Nichols & Holmes 2002) -- so it is much stricter than the
    small-volume p in the mask folders and the two are not comparable.
    Uncorrected p is that voxel-and-TR's own permutation p, for visualisation
    only."""
    n, n_vox, n_tr = D.shape
    Tobs = tstat(D)
    flat_obs = Tobs.reshape(-1)
    A = D.reshape(n, -1)
    nmt, cnt_ge, cnt_le = perm_wholebrain(
        A, flat_obs, n_perm=n_perm, seed=seed,
        pblock=adaptive_pblock(A.shape[1], max_elements=1e7), want_neg=want_neg)

    p_unc = (cnt_ge / n_perm).reshape(n_vox, n_tr).astype(np.float32)
    p_fwe = voxel_fwe_p(Tobs, nmt)
    ix, iy, iz = ijk
    pk = np.unravel_index(np.argmax(Tobs), Tobs.shape)
    mni = nib.affines.apply_affine(ref.affine, [ix[pk[0]], iy[pk[0]], iz[pk[0]]])
    tcrit = float(np.percentile(nmt, 95))
    summary = dict(
        n_vox=int(n_vox), n_tr=int(n_tr), n_subj=int(n), n_perm=int(n_perm),
        seed=int(seed), peak_t=float(Tobs[pk]), peak_TR=int(pk[1]),
        peak_mni=[int(round(float(v))) for v in mni],
        peak_p_FWE=float((nmt >= Tobs[pk]).mean()), t_crit_FWE05=tcrit,
        # counted straight off p_fwe so it always equals what you get by
        # thresholding the _1minusFWEp map at 0.95 (the t_crit percentile and
        # the exceedance proportion drift apart at low n_perm)
        n_p_FWE_lt_05=int((p_fwe < 0.05).sum()),
        n_p_unc_lt_001=int((p_unc <= 0.001).sum()),
        max_t_by_TR=[float(v) for v in Tobs.max(0)])
    p_unc_neg = (cnt_le / n_perm).reshape(n_vox, n_tr).astype(np.float32) if want_neg else None
    return Tobs, nmt, p_fwe, p_unc, p_unc_neg, summary


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
    ap.add_argument("--wholebrain", action="store_true",
                    help="also write whole-brain t / FWE-p / uncorrected-p volumes")
    ap.add_argument("--n-perm-wholebrain", type=int, default=1000,
                    help="permutations for the whole-brain maps (default 1000)")
    ap.add_argument("--wholebrain-neg", action="store_true",
                    help="also write the negative-direction whole-brain p maps")
    ap.add_argument("--wholebrain-models", default="",
                    help="comma-separated subset to write whole-brain maps for "
                         "(default: all). ~25 MB x 3 volumes per model.")
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

    # One read per model serves everything: whole brain when asked for, else
    # just the union of the masks. Mask columns are a subset of whichever.
    if args.wholebrain:
        union = brain.copy()
    else:
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
                   wholebrain=bool(args.wholebrain),
                   n_perm_wholebrain=args.n_perm_wholebrain,
                   wholebrain_neg=bool(args.wholebrain_neg),
                   wholebrain_note=(
                       "whole-brain maps are corrected over the entire brain mask "
                       "x all included TRs (max-t null); that FWE p is much "
                       "stricter than, and not comparable to, the small-volume p "
                       "in the mask folders. The uncorrected p maps are for "
                       "visualisation only."),
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

    wb_models = ([m for m in args.wholebrain_models.split(",") if m]
                 if args.wholebrain_models else models)
    if args.wholebrain:
        os.makedirs(os.path.join(args.out_dir, "wholebrain"), exist_ok=True)
        print(f"[info] whole-brain maps for {len(wb_models)} models, "
              f"{brain.sum()} voxels x {len(trs)} TRs, "
              f"{args.n_perm_wholebrain} perms", file=sys.stderr)

    rows, wb_rows = [], []
    for mi, model in enumerate(models, 1):
        t0 = time.time()
        Du = read_model_union(args.root, args.dir_pattern, model, trs, union_ijk)

        if args.wholebrain and model in wb_models:
            wb_dir = os.path.join(args.out_dir, "wholebrain")
            Tw, nmw, p_fwe, p_unc, p_unc_neg, wsum = run_wholebrain(
                Du, ref, union_ijk, n_perm=args.n_perm_wholebrain,
                seed=args.seed, want_neg=args.wholebrain_neg)
            wsum["model"] = model
            write_wholebrain_maps(wb_dir, model, Tw, p_fwe, p_unc, p_unc_neg,
                                  nmw, ref, union_ijk, len(trs))
            json.dump(wsum, open(os.path.join(wb_dir, f"{model}_summary.json"), "w"),
                      indent=2)
            wb_rows.append(dict(model=model, n_vox=wsum["n_vox"],
                                peak_t=round(wsum["peak_t"], 3),
                                peak_TR=wsum["peak_TR"],
                                peak_mni="/".join(str(c) for c in wsum["peak_mni"]),
                                p_FWE=wsum["peak_p_FWE"],
                                t_crit_FWE05=round(wsum["t_crit_FWE05"], 3),
                                n_p_FWE_lt_05=wsum["n_p_FWE_lt_05"],
                                n_p_unc_lt_001=wsum["n_p_unc_lt_001"]))
            print(f"  {'*' if wsum['peak_p_FWE'] < 0.05 else ' '} [{mi:2d}/{len(models)}] "
                  f"WHOLEBRAIN {model:42s} t={wsum['peak_t']:5.2f} "
                  f"TR{wsum['peak_TR']:<2d} p={wsum['peak_p_FWE']:.4f} "
                  f"MNI={wsum['peak_mni']}", flush=True)
            del Tw, nmw, p_fwe, p_unc, p_unc_neg

        for name, v in masks.items():
            cols = lin[name]
            D = np.ascontiguousarray(Du[:, cols, :])
            ijk = tuple(a[cols] for a in union_ijk)
            Tobs, nmt, svc = svc_both_signs(D, ref, ijk, n_perm=args.n_perm,
                                            seed=args.seed)
            svc["model"], svc["mask"] = model, name
            write_maps(os.path.join(args.out_dir, name), model, Tobs, nmt,
                       ref, ijk, len(trs))
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

    if wb_rows:
        wb_csv = os.path.join(args.out_dir, "wholebrain_summary_table.csv")
        with open(wb_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(wb_rows[0].keys()))
            w.writeheader(); w.writerows(wb_rows)
        print(f"-> {wb_csv}")

    csv_path = os.path.join(args.out_dir, "summary_table.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {csv_path}")


if __name__ == "__main__":
    main()
