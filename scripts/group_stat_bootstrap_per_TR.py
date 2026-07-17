"""
Group-level 4-D one-sample cluster-mass test with sign-flip permutations,
built for the instruction-phase per-TR RSA outputs.

Inputs (per model):
    {root}/group_RSA_instruction_per_TR_glmbase_01-TR{TR}_cropped/
        cropped_masked_smooth_fwhm5_{model}_beta_std.nii     # 4-D: (X, Y, Z, n_subj)
        mask_all_32_subjects.nii                             # 3-D: (X, Y, Z)  (same for all TRs)

Data are ALREADY smoothed and mask-restricted at the subject level — this
script does no further smoothing/masking of the input beta maps.

Per model it produces:
    - group_rdm-{model}_t.nii.gz             4-D observed t-map (open in fsleyes,
                                             scrub the TR slider to see anatomy)
    - group_rdm-{model}_clust1minusFWEp.nii.gz  4-D 1-p (threshold 0.95 = p<0.05 FWE)
    - group_rdm-{model}_clusterlabels.nii.gz 4-D integer cluster labels
    - group_rdm-{model}_clusters.json        per-cluster mass, p, peak MNI, peak TR, trace
    - cluster_traces_{model}.pdf             mean-t-over-TRs per top-N cluster
    - null_vs_observed.pdf                   summary (all models on one page)

Smoke run:
    python group_stat_bootstrap_per_TR.py --n_perm 10

Real run (cluster):
    python group_stat_bootstrap_per_TR.py \
        --root /home/fs0/xpsy1114/scratch/data/derivatives/group/per_TR \
        --output_dir /home/fs0/xpsy1114/scratch/data/derivatives/group/per_TR_cluster \
        --n_perm 5000
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import nibabel as nib
from scipy import ndimage, stats
from joblib import Parallel, delayed
from matplotlib import pyplot as plt


# ---- constants ----
MODELS = ("DSR", "rewDSR", "simple")
FILE_PATTERN = "cropped_masked_smooth_fwhm5_{model}_beta_std.nii"
DIR_GLOB = "group_RSA_instruction_per_TR_glmbase_01-TR*_cropped"
MASK_NAME = "mask_all_32_subjects.nii"


# ---- data ----
def find_tr_dirs(root, dir_glob=DIR_GLOB):
    hits = sorted(glob.glob(os.path.join(root, dir_glob)))
    if not hits:
        raise FileNotFoundError(f"no per-TR dirs under {root} matching {dir_glob}")
    out = {}
    for p in hits:
        m = re.search(r"01-TR(\d+)_cropped$", p)
        if m is None:
            continue
        out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def load_mask(tr_dirs):
    first = next(iter(tr_dirs.values()))
    mask_path = os.path.join(first, MASK_NAME)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"mask not found at {mask_path}")
    img = nib.load(mask_path)
    return img.get_fdata() > 0, img


def load_model_data(model, tr_dirs):
    per_tr = []
    tr_order = []
    for tr, d in tr_dirs.items():
        f = os.path.join(d, FILE_PATTERN.format(model=model))
        if not os.path.exists(f):
            print(f"  [{model}] WARN: TR{tr} missing ({f}); skipping this TR")
            continue
        arr = nib.load(f).get_fdata()          # (X, Y, Z, n_subj)
        per_tr.append(np.moveaxis(arr, -1, 0)) # (n_subj, X, Y, Z)
        tr_order.append(tr)
    if not per_tr:
        raise RuntimeError(f"no TR files found for model {model}")
    return np.stack(per_tr, axis=-1), tr_order


# ---- stats ----
def get_stat(data):
    non_zero = np.any(data, axis=0) & ~np.any(np.isnan(data), axis=0)
    stat = np.zeros_like(data[0])
    if non_zero.any():
        m = np.mean(data[:, non_zero], axis=0)
        s = np.std(data[:, non_zero], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            stat[non_zero] = np.where(s > 0, m / s * np.sqrt(len(data)), 0.0)
    return stat


def get_perm_stat(data, seed, mask4d):
    rng = np.random.RandomState(seed)
    flips = 1 - 2 * rng.randint(0, 2, size=len(data))
    return get_stat(data * flips[:, None, None, None, None]) * mask4d


def get_clusters(stat, ref_t, structure):
    return ndimage.label(stat > ref_t, structure=structure)


def get_cluster_mass(stat, cluster_map, n_clusters):
    if n_clusters == 0:
        return np.array([0.0])
    return np.array([stat[cluster_map == i + 1].sum() for i in range(n_clusters)])


def get_perm_max_mass(data, seed, ref_t, structure, mask4d):
    stat = get_perm_stat(data, seed, mask4d)
    cmap, ncl = get_clusters(stat, ref_t, structure)
    return float(np.max(get_cluster_mass(stat, cmap, ncl)))


# ---- per-cluster geometry ----
def cluster_peak(cluster_map, stat_map, affine, cid):
    """Peak voxel + MNI + peak TR for one cluster."""
    mask = cluster_map == cid
    vals = np.where(mask, stat_map, -np.inf)
    idx = np.unravel_index(int(np.argmax(vals)), vals.shape)  # (x, y, z, t)
    peak_mni = (affine @ np.array([idx[0], idx[1], idx[2], 1.0]))[:3]
    return {
        "peak_ijk": [int(idx[0]), int(idx[1]), int(idx[2])],
        "peak_tr_idx": int(idx[3]),
        "peak_t": float(stat_map[idx]),
        "peak_mni": [float(x) for x in peak_mni],
        "n_voxels_4d": int(mask.sum()),
        "n_voxels_spatial": int(mask.any(axis=-1).sum()),
        "tr_span": [int(i) for i in np.where(mask.any(axis=(0, 1, 2)))[0]],
    }


def cluster_trace(cluster_map, stat_map, cid):
    """Mean t across the cluster's spatial footprint, per TR."""
    mask = cluster_map == cid
    footprint = mask.any(axis=-1)              # (X, Y, Z)
    xs, ys, zs = np.where(footprint)
    vals = stat_map[xs, ys, zs, :]             # (n_footprint_vox, n_TR)
    return vals.mean(axis=0)


# ---- plots ----
def plot_cluster_traces(records, tr_order, ref_t, model_name, save_path,
                        sig_thresh_1mp=0.95):
    """One panel: mean-t-across-cluster-footprint per TR, one line per cluster.
       Star at each cluster's peak TR; shaded band over its TR span.
       Significant clusters (1-p >= 0.95) drawn solid; others dashed."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = list(tr_order)
    if not records:
        ax.text(0.5, 0.5, "no clusters", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(f"{model_name}: no supra-threshold clusters")
        fig.tight_layout(); fig.savefig(save_path); plt.close(fig); return

    for r in records:
        trace = np.asarray(r["trace"])
        mni = r["peak_mni"]
        cid = r["cluster_id"]
        sig = r["one_minus_p_FWE"] >= sig_thresh_1mp
        style = "-" if sig else "--"
        alpha = 1.0 if sig else 0.55
        label = (f"C{cid}  MNI ({mni[0]:+.0f},{mni[1]:+.0f},{mni[2]:+.0f})  "
                 f"1-p={r['one_minus_p_FWE']:.2f}")
        line, = ax.plot(xs, trace, marker="o", ls=style, alpha=alpha, label=label)
        # shade the TRs this cluster occupies
        for tr_i in r["tr_span"]:
            ax.axvspan(xs[tr_i] - 0.25, xs[tr_i] + 0.25,
                       color=line.get_color(), alpha=0.06)
        # star at peak TR
        peak_i = r["peak_tr_idx"]
        ax.plot(xs[peak_i], trace[peak_i], marker="*", ms=18,
                color=line.get_color(), mec="k", mew=1.0, zorder=5)

    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(ref_t, color="grey", lw=0.5, ls=":",
               label=f"cluster-forming t (df) = {ref_t:.2f}")
    ax.set_xlabel("TR")
    ax.set_ylabel("mean t inside cluster spatial footprint")
    ax.set_title(f"{model_name}: cluster time-courses "
                 f"(* = peak TR; solid = FWE-significant)")
    ax.set_xticks(xs)
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _fmt_mni(mni):
    return f"({mni[0]:+.0f}, {mni[1]:+.0f}, {mni[2]:+.0f})"


def write_paper_summary(output_dir, root, dir_glob, tr_order, n_subj, n_perm,
                        p_thres, ref_t, connectivity, results_by_model,
                        all_1mp_sig=0.95):
    """Write analysis_summary.md — settings + per-model per-cluster tables +
    prose that can be paraphrased into a methods/results section."""
    df = n_subj - 1
    p_fwe_thresh = 1.0 - all_1mp_sig                # e.g. 0.05
    lines = []
    L = lines.append

    L("# Group-level 4-D cluster-mass test — analysis summary\n")
    L(f"_Generated by `group_stat_bootstrap_per_TR.py`._\n")

    # ---- methods paragraph (paraphrase for paper) ----
    L("## Method (paraphrase for paper)\n")
    L(f"For each of the {len(results_by_model)} model RSA maps "
      f"({', '.join(results_by_model.keys())}), subject-level regression "
      f"coefficients were tested at the group level using a one-sample "
      f"sign-flip cluster-mass permutation test (Nichols & Holmes, 2002; "
      f"Winkler et al., 2014) over the joint space × time (TR) domain.\n")
    L(f"- **Test statistic.** At every (voxel, TR) point, the group-level "
      f"one-sample t-statistic against zero was computed as "
      f"`mean(β) / SD(β) × √N` across N = {n_subj} subjects (df = {df}).")
    L(f"- **Cluster-forming threshold.** Voxels/TRs with t > {ref_t:.3f} "
      f"(one-tailed p < {p_thres}) were binarised. Supra-threshold points "
      f"were grouped into clusters using 4-D face-neighbour connectivity "
      f"(`scipy.ndimage.generate_binary_structure(rank=4, "
      f"connectivity={connectivity})`) — connectivity across the three "
      f"spatial axes **and** across TR.")
    L(f"- **Cluster statistic.** Summed t within each cluster (cluster mass).")
    L(f"- **Null distribution.** {n_perm} sign-flip permutations. In each "
      f"permutation, every subject's whole (X, Y, Z, TR) volume was "
      f"multiplied by an independent ±1 (subject-level flip), the t-map "
      f"was recomputed, clusters extracted, and the **max cluster mass** "
      f"across clusters recorded.")
    L(f"- **Family-wise error correction.** Each observed cluster's mass was "
      f"compared to the max-mass permutation null. The reported "
      f"`p_FWE` is the fraction of permutations whose max cluster mass "
      f"equalled or exceeded the observed cluster's mass. Clusters with "
      f"p_FWE < {p_fwe_thresh:.2f} survive family-wise correction **within "
      f"a model**.")
    L(f"- **Multiple models.** The {len(results_by_model)} models were "
      f"tested independently. Cluster-level p_FWE is corrected within each "
      f"model; no additional correction across models is applied, as the "
      f"models are treated as separate a-priori hypotheses.")
    L(f"- **Directionality.** The test is one-tailed positive — only "
      f"clusters where the group mean β is greater than zero are flagged.")
    L(f"- **Preprocessing.** Subject-level maps were already spatially "
      f"smoothed (FWHM = 5 mm) and mask-restricted to voxels present in "
      f"all {n_subj} subjects before entering this test. No temporal "
      f"smoothing was applied.")
    L("")

    # ---- settings table ----
    L("## Settings\n")
    L("| parameter | value |")
    L("|---|---|")
    L(f"| n_subjects | {n_subj} |")
    L(f"| n_TRs used | {len(tr_order)} |")
    L(f"| TR indices | {tr_order} |")
    L(f"| n permutations | {n_perm} |")
    L(f"| cluster-forming p_thres (one-tailed) | {p_thres} |")
    L(f"| cluster-forming t-threshold | {ref_t:.4f} (df = {df}) |")
    L(f"| FWE significance threshold | p_FWE < {p_fwe_thresh:.2f} |")
    L(f"| ndimage connectivity | rank=4, connectivity={connectivity} "
      f"(1 = face nbrs) |")
    L(f"| input root | `{root}` |")
    L(f"| dir_glob | `{dir_glob}` |")
    L("")

    # ---- per-model results tables ----
    L("## Results per model\n")
    for model, recs in results_by_model.items():
        n_all = len(recs["all"])
        n_sig = sum(1 for r in recs["all"] if r["one_minus_p_FWE"] >= all_1mp_sig)
        L(f"### {model}\n")
        L(f"{n_all} supra-threshold cluster(s); "
          f"{n_sig} survive p_FWE < {p_fwe_thresh:.2f}.\n")
        if n_all == 0:
            L("_No clusters._\n")
            continue
        L("| # | cluster_id | mass | p_FWE | peak MNI (x, y, z) | peak TR | "
          "peak t | n_vox 4-D | n_vox spatial | TRs spanned | sig |")
        L("|---|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(recs["all"], start=1):
            p_fwe = 1.0 - r["one_minus_p_FWE"]
            sig_mark = "**\\***" if r["one_minus_p_FWE"] >= all_1mp_sig else ""
            trs_spanned = [tr_order[j] for j in r["tr_span"]]
            L(f"| {i} | {r['cluster_id']} | {r['cluster_mass']:.2f} | "
              f"{p_fwe:.3f} | {_fmt_mni(r['peak_mni'])} | "
              f"{tr_order[r['peak_tr_idx']]} | {r['peak_t']:.2f} | "
              f"{r['n_voxels_4d']} | {r['n_voxels_spatial']} | "
              f"{trs_spanned} | {sig_mark} |")
        L("")

    # ---- outputs ----
    L("## Files produced\n")
    L("Per model:")
    L("- `group_rdm-{model}_t.nii.gz` — 4-D observed t-map "
      "(open in fsleyes, scrub TR slider).")
    L("- `group_rdm-{model}_clust1minusFWEp.nii.gz` — 4-D `1 − p_FWE` "
      f"(threshold at {all_1mp_sig:.2f} to see p_FWE < {p_fwe_thresh:.2f} "
      "clusters).")
    L("- `group_rdm-{model}_clusterlabels.nii.gz` — 4-D integer cluster labels.")
    L("- `group_rdm-{model}_clusters.json` — machine-readable per-cluster stats.")
    L("- `cluster_traces_{model}.pdf` — mean t across each top-N cluster's "
      "spatial footprint per TR (star = peak TR; band = TRs cluster spans).")
    L("")
    L("Shared:")
    L("- `args.json` — full argument list this run was launched with.")
    L("- `null_vs_observed.pdf` — max-mass null + observed cluster masses "
      "per model.")
    L("- `analysis_summary.md` — this file.")
    L("")

    # ---- interpretation notes ----
    L("## Interpretation notes\n")
    L("- **One-tailed positive.** Only clusters where the group mean is > 0 "
      "are tested. Negative effects (evidence against a model) are ignored.")
    L("- **FWE within model, not FDR.** Correction is family-wise error via "
      "the max cluster-mass null distribution over the whole 4-D "
      "(space × TR) search volume. No FDR was applied. No correction "
      "across models is applied (models are separate a-priori hypotheses).")
    L("- **Cluster interpretation.** A cluster spans a contiguous chunk of "
      "space × time. Its \"peak TR\" is the TR containing the cluster's "
      "peak-t voxel; the shaded band in `cluster_traces_{model}.pdf` "
      "covers every TR at which the cluster has any supra-threshold voxel.")
    L("- **Test power.** With only 4-D face-neighbour connectivity, "
      "connections across TR require literal voxel-wise overlap at "
      "adjacent TRs. If your effects drift spatially over TRs, consider "
      "`--connectivity 2` (which also connects edge neighbours across TR).")
    L("")

    with open(os.path.join(output_dir, "analysis_summary.md"), "w") as f:
        f.write("\n".join(lines))


def plot_null_vs_observed(all_stats, n_perm, save_path):
    """Compact summary: histogram of null max cluster masses + observed masses."""
    n = len(all_stats)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.0), squeeze=False)
    axes = axes[0]
    for ax, (model, (null_dist, records)) in zip(axes, all_stats.items()):
        bins = max(10, int(n_perm / 10))
        counts, edges = np.histogram(null_dist, bins=bins)
        ax.stairs(counts, edges, fill=True, color=(0.82, 0.82, 0.82),
                  label="null max mass")
        if len(null_dist):
            crit = np.percentile(null_dist, 95)
            ax.axvline(crit, color="red", lw=1,
                       label=f"null 95% = {crit:.1f}")
        # observed cluster masses
        for r in records:
            m = r["cluster_mass"]
            col = "black" if r["one_minus_p_FWE"] < 0.95 else "#2C7A2C"
            ax.axvline(m, color=col, lw=0.5, alpha=0.6)
            ax.plot([m], [max(counts) * 0.05 if len(counts) else 0.05],
                    "*", ms=13, color=col, mec="k")
            ax.annotate(f"C{r['cluster_id']} 1-p={r['one_minus_p_FWE']:.2f}",
                        xy=(m, max(counts) * 0.55 if len(counts) else 0.5),
                        rotation=90, fontsize=7, ha="right", va="top",
                        color=col)
        ax.set_title(f"{model}   (green * = FWE sig)")
        ax.set_xlabel("cluster mass")
        ax.set_ylabel("count")
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ---- driver ----
def run(root, output_dir, n_perm=10, p_thres=0.001, n_jobs=-1, models=MODELS,
        connectivity=1, top_n=5, tr_include=None, dir_glob=DIR_GLOB):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump({
            "root": root, "output_dir": output_dir, "n_perm": n_perm,
            "p_thres": p_thres, "models": list(models),
            "connectivity": connectivity, "top_n": top_n,
            "tr_include": tr_include, "dir_glob": dir_glob,
        }, f, indent=2)

    tr_dirs = find_tr_dirs(root, dir_glob=dir_glob)
    if tr_include is not None:
        wanted = set(int(x) for x in tr_include)
        tr_dirs = {tr: d for tr, d in tr_dirs.items() if tr in wanted}
        if not tr_dirs:
            raise RuntimeError(
                f"--tr_include {sorted(wanted)} matched none of the "
                f"discovered TRs under {root}"
            )
    print(f"found {len(tr_dirs)} TR dirs: {list(tr_dirs.keys())}")

    mask, mask_img = load_mask(tr_dirs)
    print(f"mask shape={mask.shape}, in-mask voxels={int(mask.sum())}")
    affine = mask_img.affine

    structure = ndimage.generate_binary_structure(rank=4, connectivity=connectivity)

    all_stats = {}
    results_by_model = {}          # for analysis_summary.md
    ref_t_shared = None
    n_subj_shared = None
    tr_order_shared = None

    for model in models:
        print(f"\n=== model: {model} ===")
        data, tr_order = load_model_data(model, tr_dirs)
        n_subj = data.shape[0]
        n_tr = data.shape[-1]
        print(f"  data shape = {data.shape}  (n_subj={n_subj}, n_TR={n_tr})")

        mask4d = mask[..., None].astype(float)
        ref_t = float(stats.t.ppf(1 - p_thres, n_subj - 1))
        print(f"  ref_t (p<{p_thres}, df={n_subj-1}) = {ref_t:.3f}")

        # ---- null ----
        print(f"  running {n_perm} sign-flip perms (n_jobs={n_jobs}) ...")
        null_max_mass = np.asarray(
            Parallel(n_jobs=n_jobs)(
                delayed(get_perm_max_mass)(data, seed=i, ref_t=ref_t,
                                           structure=structure, mask4d=mask4d)
                for i in range(n_perm)
            )
        )

        # ---- observed ----
        stat_map = get_stat(data) * mask4d
        cluster_map, n_clusters = get_clusters(stat_map, ref_t, structure)
        cluster_mass = get_cluster_mass(stat_map, cluster_map, n_clusters)

        # p per observed cluster + geometry + trace
        p_map = np.zeros_like(stat_map)
        cluster_records = []
        for i in range(n_clusters):
            m = cluster_mass[i]
            if m == 0:
                continue
            cid = i + 1
            one_minus_p = float(np.mean(null_max_mass < m))
            p_map[cluster_map == cid] = one_minus_p
            peak = cluster_peak(cluster_map, stat_map, affine, cid)
            trace = cluster_trace(cluster_map, stat_map, cid)
            cluster_records.append({
                "cluster_id": int(cid),
                "cluster_mass": float(m),
                "one_minus_p_FWE": one_minus_p,
                "trace": [float(x) for x in trace],
                **peak,
            })

        # sort by cluster mass, big first
        cluster_records.sort(key=lambda r: r["cluster_mass"], reverse=True)

        if n_clusters == 0:
            print("  no supra-threshold clusters in the observed map.")
        else:
            print(f"  {n_clusters} observed cluster(s); top-5 by mass:")
            for r in cluster_records[:5]:
                mni = r["peak_mni"]
                print(f"    C{r['cluster_id']:>2}  mass={r['cluster_mass']:8.1f}  "
                      f"1-p={r['one_minus_p_FWE']:.2f}  "
                      f"peak MNI=({mni[0]:+.0f},{mni[1]:+.0f},{mni[2]:+.0f})  "
                      f"peak TR={tr_order[r['peak_tr_idx']]}  "
                      f"peak t={r['peak_t']:.2f}")

        # ---- save NIfTIs + JSON ----
        header = mask_img.header
        nib.save(nib.Nifti1Image(stat_map.astype(np.float32), affine, header),
                 os.path.join(output_dir, f"group_rdm-{model}_t.nii.gz"))
        nib.save(nib.Nifti1Image(p_map.astype(np.float32), affine, header),
                 os.path.join(output_dir, f"group_rdm-{model}_clust1minusFWEp.nii.gz"))
        nib.save(nib.Nifti1Image(cluster_map.astype(np.int32), affine, header),
                 os.path.join(output_dir, f"group_rdm-{model}_clusterlabels.nii.gz"))
        with open(os.path.join(output_dir, f"group_rdm-{model}_clusters.json"), "w") as f:
            json.dump({
                "TR_order": tr_order,
                "n_subj": n_subj,
                "ref_t": ref_t,
                "p_thres_cluster_forming": p_thres,
                "n_perm": n_perm,
                "clusters": cluster_records,
            }, f, indent=2)

        # ---- per-model plot ----
        top_records = cluster_records[:top_n]
        plot_cluster_traces(
            top_records, tr_order, ref_t, model,
            os.path.join(output_dir, f"cluster_traces_{model}.pdf"),
        )

        all_stats[model] = (null_max_mass, cluster_records[:top_n])
        results_by_model[model] = {"all": cluster_records}
        # these are identical across models (same n_subj, TR list), but capture once
        ref_t_shared = ref_t
        n_subj_shared = n_subj
        tr_order_shared = tr_order

    # ---- cross-model summary ----
    plot_null_vs_observed(
        all_stats, n_perm, os.path.join(output_dir, "null_vs_observed.pdf"),
    )

    # ---- paper-ready markdown summary ----
    if results_by_model:
        write_paper_summary(
            output_dir=output_dir, root=root, dir_glob=dir_glob,
            tr_order=tr_order_shared, n_subj=n_subj_shared, n_perm=n_perm,
            p_thres=p_thres, ref_t=ref_t_shared, connectivity=connectivity,
            results_by_model=results_by_model,
        )
    print(f"\nDone. Outputs in: {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/Users/xpsy1114/Documents/projects/multiple_clocks/data/"
                            "derivatives/group/per_TR")
    ap.add_argument("--output_dir",
                    default="/Users/xpsy1114/Documents/projects/multiple_clocks/data/"
                            "derivatives/group/per_TR_cluster_smoke")
    ap.add_argument("--n_perm", type=int, default=10)
    ap.add_argument("--p_thres", type=float, default=0.001)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--connectivity", type=int, default=1,
                    help="4-D binary_structure connectivity (1=face nbrs)")
    ap.add_argument("--top_n", type=int, default=5,
                    help="how many top-mass clusters to visualise per model")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--tr_include", nargs="+", type=int, default=None,
                    help="if given, only use these TR indices "
                         "(e.g. --tr_include 0 1 2 3 4 5 6 7 8)")
    ap.add_argument("--dir_glob", default=DIR_GLOB,
                    help="glob pattern (relative to --root) that selects the "
                         "per-TR folders for this analysis. Must contain "
                         "'01-TR*_cropped' so TR indices can be parsed. "
                         "Examples: "
                         "'group_RSA_instruction_per_TR_glmbase_01-TR*_cropped', "
                         "'group_RSA_split_DSR_per_TR_glmbase_01-TR*_cropped'")
    args = ap.parse_args()

    run(root=args.root, output_dir=args.output_dir, n_perm=args.n_perm,
        p_thres=args.p_thres, n_jobs=args.n_jobs, connectivity=args.connectivity,
        models=args.models, top_n=args.top_n, tr_include=args.tr_include,
        dir_glob=args.dir_glob)
