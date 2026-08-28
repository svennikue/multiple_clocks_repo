#!/usr/bin/env python
"""
Direct left-vs-right test on the LOSO timecourses written by `svc_loso_batch.py`.

Why this exists: "significant in the left mask, not significant in the right"
is not evidence of lateralisation -- a difference in significance is not a
significant difference. This tests L - R itself.

It reuses the per-subject held-out arrays the LOSO already wrote
(`{mask}/{model}_loso_k{K}.npy`, shape (n_subj, n_TR)), so nothing is refitted.
Each hemisphere selected its own top-k voxels on n-1 subjects and read out the
held-out subject, so the per-subject difference is an unbiased paired
contrast. It is tested with the SAME sign-flip max-t null (`null_max_t`) used
for every other statistic in this pipeline, corrected over the TR axis
(CLAUDE.md rule 4).

Usage
    python hemisphere_contrast.py --in-dir <svc_loso_batch out-dir> \
        --left MTL_L --right MTL_R \
        --models curr_rew,next_rew,two_next_rew,three_next_rew --k 100
"""
import argparse, csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svc_loso_test import tstat, null_max_t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--k", default="100")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-name", default="hemisphere_contrast")
    args = ap.parse_args()

    rows = []
    for model in [m for m in args.models.split(",") if m]:
        L = np.load(os.path.join(args.in_dir, args.left,  f"{model}_loso_k{args.k}.npy"))
        R = np.load(os.path.join(args.in_dir, args.right, f"{model}_loso_k{args.k}.npy"))
        assert L.shape == R.shape, f"{model}: {L.shape} vs {R.shape}"
        n_subj, n_tr = L.shape

        diff = L - R
        t_d = tstat(diff)
        # two one-sided nulls off the same symmetric sign-flip distribution
        nm = null_max_t(diff, n_perm=args.n_perm, seed=args.seed)
        p_L_gt_R = np.array([(nm >= v).mean() for v in t_d])
        p_R_gt_L = np.array([(nm >= -v).mean() for v in t_d])

        t_L, t_R = tstat(L), tstat(R)
        pk = int(np.argmax(np.abs(t_d)))
        rows.append(dict(
            model=model, k=int(args.k), n_subj=n_subj,
            peak_TR=pk, t_LminusR=round(float(t_d[pk]), 3),
            p_FWE_L_gt_R=float(p_L_gt_R[pk]), p_FWE_R_gt_L=float(p_R_gt_L[pk]),
            t_L_at_peak=round(float(t_L[pk]), 3), t_R_at_peak=round(float(t_R[pk]), 3),
            mean_L=round(float(L[:, pk].mean()), 5),
            mean_R=round(float(R[:, pk].mean()), 5),
            n_TR_lateralised_p05=int(((p_L_gt_R < .05) | (p_R_gt_L < .05)).sum())))
        print(f"\n=== {model} (k={args.k}) ===")
        print("  TR      " + "".join(f"{i:>7d}" for i in range(n_tr)))
        for lbl, v in (("t left ", t_L), ("t right", t_R), ("t L-R  ", t_d)):
            print(f"  {lbl} " + "".join(f"{x:>7.2f}" for x in v))
        print("  p(L>R)  " + "".join(f"{x:>7.3f}" for x in p_L_gt_R))
        print(f"  -> largest |L-R| at TR{pk}: t = {t_d[pk]:+.2f}, "
              f"p_FWE(L>R) = {p_L_gt_R[pk]:.4f}, p_FWE(R>L) = {p_R_gt_L[pk]:.4f}")

    out = os.path.join(args.in_dir, f"{args.out_name}_{args.left}_vs_{args.right}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
