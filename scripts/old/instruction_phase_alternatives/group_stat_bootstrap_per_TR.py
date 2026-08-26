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
#MODELS = ("DSR", "rewDSR", "simple")
MODELS = (["rewDSR"])
FILE_PATTERN = "cropped_masked_smooth_fwhm5_{model}_beta_std.nii"
DIR_GLOB = "group_RSA_instruction_per_TR_glmbase_01-TR*_cropped"
MASK_NAME = "mask_all_32_subjects.nii"

# Reward-cue schedule for the fMRI instruction phase (times in seconds
# from the start of the reward-cue period). Two passes: a first
# 1.5-s/reward reveal (A, B, C, D, 0–6 s), then a faster 1-s/reward
# refresh (A, B, C, D, 6–10 s). Mirrors the timing defined in
# scripts/mc/latest_experiment/3x3_fMRI_part1.py. Used by the publication
# footprint plot to draw a reward-schedule strip along the panel top.
# Colours follow the CLAUDE.md state palette (A orange … D dark purple).
REWARD_SCHEDULE = [
    (0.0, 1.5, "A", "#F15A29"),
    (1.5, 3.0, "B", "#F7931E"),
    (3.0, 4.5, "C", "#C7C6E2"),
    (4.5, 6.0, "D", "#6B60AA"),
    (6.0, 7.0, "A", "#F15A29"),
    (7.0, 8.0, "B", "#F7931E"),
    (8.0, 9.0, "C", "#C7C6E2"),
    (9.0, 10.0, "D", "#6B60AA"),
]


# ---- canonical ROI palette (CLAUDE.md Showgirl2 mapping, hardcoded so we
#      don't depend on era_brewer on the cluster). Peak-of-cluster MNIs get
#      looked up in Harvard-Oxford / Juelich (+ Brainnetome if available)
#      and coloured by the ROI their peak falls in. Any peak that lands
#      outside these 7 canonical ROIs is given a rotating fallback colour.
CANONICAL_ROI_COLOURS = {
    "EC":              "#B74C2D",   # dark red
    "ACC":             "#448363",   # mPFC / dark teal-green
    "HC_anterior":     "#CCB178",   # tan
    "PCC":             "#C1DCBF",   # pale green
    "medialOFC":       "#DC673E",   # red / orange-red
    "Parahippocampal": "#7BB594",   # sage
    "HC_mid":          "#629E7E",   # mid-dark green
}
_FALLBACK_ROI_COLOURS = [
    "#23677E",   # CLAUDE.md blue — Precuneus (or first extra)
    "#6B60AA",   # dark purple
    "#F15A29",   # orange
    "#0e3d3a",   # dark teal
    "#7eb1c4",   # blue
    "#3d8b7d",   # mid teal
    "#a7d9b2",   # pale green
    "#888888",   # grey
]

# Deterministic fallback colour per ROI name (rotates through _FALLBACK_ROI_COLOURS
# in first-seen order — reset per run so ordering is stable within one output).
_fallback_colour_assignments = {}


def _colour_for_roi(roi):
    if roi is None:
        return "#333333"
    if roi in CANONICAL_ROI_COLOURS:
        return CANONICAL_ROI_COLOURS[roi]
    if roi not in _fallback_colour_assignments:
        idx = len(_fallback_colour_assignments) % len(_FALLBACK_ROI_COLOURS)
        _fallback_colour_assignments[roi] = _FALLBACK_ROI_COLOURS[idx]
    return _fallback_colour_assignments[roi]


# ---- atlas lookup for peak-of-cluster ROI labelling ------------------
# Small versions of the AtlasLookup / BrainnetomeLookup classes from
# scripts/cell_to_roi_MNI.py, inlined here so this script runs stand-alone
# on the cluster (importing cell_to_roi_MNI would fire its module-level
# CSV loads). Atlases are fetched via nilearn (Harvard-Oxford + Juelich)
# and are network-free after the first call.
class _AtlasLookup:
    def __init__(self, atlas):
        img = atlas.maps if hasattr(atlas, "maps") else atlas
        self.img = nib.load(img) if isinstance(img, str) else img
        self.data = self.img.get_fdata()
        self.inv_affine = np.linalg.inv(self.img.affine)
        self.labels = list(atlas.labels)

    def label_at(self, x, y, z):
        v = nib.affines.apply_affine(self.inv_affine, [x, y, z])
        v = np.round(v).astype(int)
        if np.any(v < 0) or np.any(v >= self.data.shape):
            return "outside atlas"
        idx = int(self.data[tuple(v)])
        if idx == 0:
            return "background"
        if idx < 0 or idx >= len(self.labels):
            return f"unknown index {idx}"
        return self.labels[idx]


class _BrainnetomeLookup:
    def __init__(self, nii_path, lut_path):
        self.img = nib.load(nii_path)
        self.data = self.img.get_fdata()
        self.inv_affine = np.linalg.inv(self.img.affine)
        labels = {0: "background"}
        with open(lut_path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                idx = None
                for p in parts:
                    try:
                        idx = int(p); break
                    except ValueError:
                        pass
                if idx is None:
                    continue
                text = " ".join(p for p in parts if not re.match(r"^-?\d+(\.\d+)?$", p))
                labels[idx] = text if text else f"Brainnetome_{idx}"
        self.labels = labels

    def label_at(self, x, y, z):
        v = nib.affines.apply_affine(self.inv_affine, [x, y, z])
        v = np.round(v).astype(int)
        if np.any(v < 0) or np.any(v >= self.data.shape):
            return "outside atlas"
        idx = int(self.data[tuple(v)])
        return self.labels.get(idx, f"Brainnetome index {idx}")


_atlas_cache = {"ho_cort": None, "ho_sub": None, "juelich": None, "brainnetome": None}


def _try_load_atlases():
    """Lazy-load atlases. Uses nilearn's cached HO + Juelich; tries to find
    a local Brainnetome via ``BRAINNETOME_DIR`` env var. Missing atlases stay
    None — labelling falls through to whatever atlases succeeded."""
    if _atlas_cache["ho_cort"] is not None:
        return _atlas_cache
    try:
        from nilearn import datasets
        _atlas_cache["ho_cort"] = _AtlasLookup(
            datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm"))
        _atlas_cache["ho_sub"] = _AtlasLookup(
            datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm"))
        _atlas_cache["juelich"] = _AtlasLookup(
            datasets.fetch_atlas_juelich("maxprob-thr25-2mm"))
    except Exception as exc:
        print(f"  [atlas] nilearn atlases unavailable ({exc}); "
              f"cluster ROI labels will be limited to Brainnetome or 'Other'.")
    bn_dir = os.environ.get("BRAINNETOME_DIR",
                             "/Users/xpsy1114/Documents/toolboxes/Brainnatome")
    nii = os.path.join(bn_dir, "BN_Atlas_246_1mm.nii.gz")
    lut = os.path.join(bn_dir, "BN_Atlas_246_LUT.txt")
    if os.path.isfile(nii) and os.path.isfile(lut):
        try:
            _atlas_cache["brainnetome"] = _BrainnetomeLookup(nii, lut)
        except Exception as exc:
            print(f"  [atlas] Brainnetome load failed ({exc}); skipping.")
    return _atlas_cache


def _contains_any(text, patterns):
    s = str(text).lower()
    return any(p.lower() in s for p in patterns)


def _hc_label_from_y(y):
    return "HC_anterior" if float(y) >= -21 else "HC_mid"


def _roi_from_mni(x, y, z):
    """Return the canonical CLAUDE.md ROI name for a peak at (x, y, z), or
    a shorter atlas-based label when the peak falls outside the 7 canonical
    ROIs. Mirrors the fine-grained-first rule cascade in
    ``scripts/cell_to_roi_MNI.py::assign_initial_roi``."""
    a = _try_load_atlases()
    ho_c = a["ho_cort"].label_at(x, y, z) if a["ho_cort"] else ""
    ho_s = a["ho_sub"].label_at(x, y, z)  if a["ho_sub"]  else ""
    ju   = a["juelich"].label_at(x, y, z) if a["juelich"] else ""
    bn   = a["brainnetome"].label_at(x, y, z) if a["brainnetome"] else ""

    # Same rule order as assign_initial_roi:
    if _contains_any(ju, ["gm hippocampus entorhinal cortex", "entorhinal"]):
        return "EC"
    if _contains_any(bn, ["tl_r", "tl_l"]):
        return "Parahippocampal"
    if _contains_any(ho_c, ["parahippocampal gyrus", "parahippocampal"]):
        return "Parahippocampal"
    if "hippocampus" in str(ho_s).lower():
        return _hc_label_from_y(y)
    if _contains_any(ju, ["hippocampus subiculum", "subiculum"]):
        return _hc_label_from_y(y)
    if "a14m" in str(bn).lower():
        return "medialOFC"      # ventral_ACC collapses into medialOFC in alt-ROI
    if _contains_any(bn, ["a32sg", "a32p", "a24rv"]):
        return "ACC"
    if "a23" in str(bn).lower():
        return "PCC"
    if "cingulate gyrus, anterior division" in str(ho_c).lower():
        return "ACC"
    if "cingulate gyrus, posterior division" in str(ho_c).lower():
        return "PCC"
    if _contains_any(bn, ["a31_l", "a31_r", "dmpos_l", "dmpos_r"]):
        return "Precuneus"
    if "precuneous cortex" in str(ho_c).lower():
        return "Precuneus"
    if _contains_any(bn, ["a11m", "a13_r", "a13_l", "a13"]):
        return "medialOFC"
    if _contains_any(ho_c, ["occipital", "cuneal", "lingual",
                             "intracalcarine", "supracalcarine",
                             "occipital pole"]):
        return "Visual"
    if _contains_any(ju, ["v1", "v2", "v3", "visual", "calcarine"]):
        return "Visual"
    # Fall back to whichever atlas returned something specific.
    for name in (bn, ho_c, ho_s, ju):
        s = str(name).strip()
        if s and s.lower() not in ("background", "outside atlas", "nan"):
            return s.split(",")[0]   # short label
    return "Other"


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
    """Sign-flip permutation: returns (max cluster mass, max voxel t)
    both taken inside `mask4d`. One perm t-map, two null statistics."""
    stat = get_perm_stat(data, seed, mask4d)
    cmap, ncl = get_clusters(stat, ref_t, structure)
    max_mass = float(np.max(get_cluster_mass(stat, cmap, ncl)))
    max_t = float(np.max(stat)) if stat.size else 0.0
    return max_mass, max_t


# ---- small-volume (mask) helpers -------------------------------------
def load_mask_nifti(mask_path, expected_shape, expected_affine):
    """Load a mask NIfTI onto the data grid. If shape+affine already match,
    use as-is; otherwise nearest-neighbour resample."""
    img = nib.load(mask_path)
    if (img.shape[:3] == expected_shape[:3]
            and np.allclose(img.affine, expected_affine, atol=1e-3)):
        return img.get_fdata() > 0
    from nilearn.image import resample_img
    ref = nib.Nifti1Image(np.zeros(expected_shape[:3], dtype=np.uint8),
                           expected_affine)
    return resample_img(img, target_affine=ref.affine,
                         target_shape=ref.shape,
                         interpolation="nearest").get_fdata() > 0


def roi_mean_per_subject_per_tr(data, roi_mask):
    """Return (n_subj, n_TR) matrix: mean β inside `roi_mask` per subject per TR.
    `data` has shape (n_subj, X, Y, Z, n_TR)."""
    x, y, z = np.where(roi_mask)
    if len(x) == 0:
        raise ValueError("roi_mask contains no voxels")
    return data[:, x, y, z, :].mean(axis=1)  # (n_subj, n_TR)


def group_t_per_tr(roi_ts):
    """(n_TR,) group t against zero from a (n_subj, n_TR) ROI-mean matrix."""
    n = roi_ts.shape[0]
    m = roi_ts.mean(axis=0)
    s = roi_ts.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, m / s * np.sqrt(n), 0.0)


def clusters_1d(t_vec, thresh):
    """Return (list of (start, end_exclusive) tuples, mass per cluster) for
    each contiguous run of `t_vec > thresh`. 1-D over TR."""
    above = t_vec > thresh
    clusters = []
    masses = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            clusters.append((i, j))
            masses.append(float(t_vec[i:j].sum()))
            i = j
        else:
            i += 1
    return clusters, np.asarray(masses if masses else [0.0])


def perm_max_mass_1d(roi_ts, seed, ref_t):
    """Sign-flip permutation for 1-D cluster mass over TR."""
    rng = np.random.RandomState(seed)
    flips = 1 - 2 * rng.randint(0, 2, size=len(roi_ts))
    t_vec = group_t_per_tr(roi_ts * flips[:, None])
    _, masses = clusters_1d(t_vec, ref_t)
    return float(masses.max())


def _mask_stem(path):
    b = os.path.basename(path)
    for ext in (".nii.gz", ".nii", ".mnc"):
        if b.endswith(ext):
            return b[: -len(ext)]
    return os.path.splitext(b)[0]


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
def _save_all_formats(fig, save_path):
    """Save the figure as .pdf, .svg AND .jpg (in that order) alongside
    whatever extension the caller passed. ``save_path`` may end in any
    of these — we strip it and rewrite all three."""
    base, _ = os.path.splitext(save_path)
    for ext in ("pdf", "svg", "jpg"):
        fig.savefig(f"{base}.{ext}", dpi=600, bbox_inches="tight")


def plot_cluster_traces(records, tr_order, ref_t, model_name, save_path,
                        sig_thresh_1mp=0.95, x_label="TR"):
    """SVC cluster-traces panel at publication size (4.5 × 3 cm, Arial).

    One line per top-N cluster, coloured by the ROI its peak falls in
    (canonical CLAUDE.md palette + rotating fallback). Solid = FWE-sig,
    dashed = n.s. Cluster's TR span shaded at low alpha in the cluster
    colour; peak TR marked with a small star. Sig clusters get a tiny
    'ROI · p_FWE<.05' label near their peak. Matches the ROI-mean
    time-course preview style so both panels can sit side by side."""
    CM = 1.0 / 2.54
    FIG_W, FIG_H = 5.5 * CM, 3.6 * CM   # slightly wider than the 4.5×3
                                          # timecourse to fit the legend
    FONT_TICK = 6
    FONT_AXIS = 7
    FONT_LABEL = 6

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": FONT_TICK,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 1.6,
        "ytick.major.size": 1.6,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    xs = list(tr_order)
    if not records:
        ax.text(0.5, 0.5, "no clusters", ha="center", va="center",
                fontsize=FONT_LABEL, transform=ax.transAxes)
        ax.set_axis_off()
        _save_all_formats(fig, save_path)
        plt.close(fig); return

    y_top = -np.inf
    for r in records:
        trace = np.asarray(r["trace"])
        mni = r["peak_mni"]
        cid = r["cluster_id"]
        sig = r["one_minus_p_FWE"] >= sig_thresh_1mp
        style = "-" if sig else "--"
        alpha_line = 1.0 if sig else 0.55
        roi = r.get("peak_roi") or _roi_from_mni(*mni)
        r["peak_roi"] = roi
        col = _colour_for_roi(roi)
        ax.plot(xs, trace, ls=style, color=col, alpha=alpha_line,
                lw=0.9, marker="o", ms=1.5, mec="none",
                label=f"C{cid} · {roi}")
        # subtle shading over the cluster's TR span
        if sig and r["tr_span"]:
            span_lo = xs[min(r["tr_span"])] - 0.5
            span_hi = xs[max(r["tr_span"])] + 0.5
            ax.axvspan(span_lo, span_hi, color=col, alpha=0.14, linewidth=0)
        # star at peak TR
        peak_i = r["peak_tr_idx"]
        ax.plot(xs[peak_i], trace[peak_i], marker="*", ms=4.5,
                color=col, mec="k", mew=0.35, zorder=5)
        y_top = max(y_top, trace.max())

    ax.axhline(0, color="k", lw=0.35, zorder=1)
    ax.axhline(ref_t, color="grey", lw=0.35, ls=":",
               label=f"t$_{{crit}}$ = {ref_t:.2f}")
    # single "FWE<.05" mark once, in the legend
    ax.set_xlabel(x_label, fontsize=FONT_AXIS, labelpad=1)
    ax.set_ylabel("mean t (cluster footprint)", fontsize=FONT_AXIS, labelpad=1)
    ax.set_xticks(xs[::2])
    ax.tick_params(labelsize=FONT_TICK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # tight compact legend inside the panel (top-right); one entry per line
    leg = ax.legend(fontsize=FONT_LABEL, frameon=False, loc="upper right",
                     handlelength=1.4, handletextpad=0.4, borderpad=0.2,
                     labelspacing=0.25)
    # Compat: `legend_handles` (mpl >=3.7) vs `legendHandles` (older).
    handles = (getattr(leg, "legend_handles", None)
                or getattr(leg, "legendHandles", []))
    for lh in handles:
        try:
            lh.set_linewidth(1.0)
        except AttributeError:
            pass

    _save_all_formats(fig, save_path)
    plt.close(fig)


def cluster_footprint_beta(data, cluster_map, cluster_id):
    """Return (n_subj, n_TR) mean β inside a cluster's *spatial footprint*
    (union of voxels the cluster occupies across any TR). Used to visualise
    the temporal profile of the effect at the voxels that actually drive
    the cluster, without diluting by the whole mask.

    `data` shape:        (n_subj, X, Y, Z, n_TR)
    `cluster_map` shape: (X, Y, Z, n_TR) with integer cluster labels
    """
    footprint = (cluster_map == cluster_id).any(axis=-1)   # (X, Y, Z)
    xs, ys, zs = np.where(footprint)
    if xs.size == 0:
        raise ValueError(f"cluster {cluster_id} has empty footprint")
    return data[:, xs, ys, zs, :].mean(axis=1)             # (n_subj, n_TR)


def _readable_text_colour(hexc):
    """Black or white, whichever contrasts more against this patch colour."""
    r, g, b = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in (r, g, b)]
    Y = .2126 * lin[0] + .7152 * lin[1] + .0722 * lin[2]
    return "white" if (1.05 / (Y + .05)) > ((Y + .05) / .05) else "black"


def plot_cluster_footprint_timecourse(subj_beta, tr_order, cluster_rec,
                                       model_name, save_path,
                                       x_label="time from cue onset (s)",
                                       reward_schedule=None,
                                       peak_marker_x="none"):
    """Publication panel: mean β inside a single cluster's spatial footprint,
    per subject × TR, with SEM ribbon. Matches the roi-mean sage-green
    style. Adds a reward-cue schedule strip along the top of the panel
    (coloured segments per reward).

    Parameters
    ----------
    subj_beta : (n_subj, n_TR) mean β inside cluster footprint.
    tr_order  : list of EV indices; each EV i is read as the interval
        [i, i+1) s, so it is plotted at its bin CENTRE (i + 0.5). This is
        what keeps the reward-schedule strip (drawn in data seconds) aligned
        with the trace -- plotting at the bin's left edge instead shifts
        every point 0.5 s early relative to the labelled reward segments.
    cluster_rec : dict from cluster records (has peak_tr_idx, peak_t, etc.).
    reward_schedule : list of (start_s, end_s, label, colour); default
        REWARD_SCHEDULE. Set to [] to disable the strip.
    peak_marker_x : x-position (seconds) for an optional dotted reference
        line. Default "none" draws nothing -- there is no principled default
        peak location to mark (the previous default, right-edge-of-peak-bin,
        was just re-stating a point already on the curve). Pass an explicit
        float if you want a specific timepoint flagged.
    """
    if reward_schedule is None:
        reward_schedule = REWARD_SCHEDULE

    CM = 1.0 / 2.54
    FIG_W, FIG_H = 4.5 * CM, 3.0 * CM
    DARK_SAGE = "#3E6B4D"
    FONT_TICK, FONT_AXIS, FONT_LABEL = 6, 7, 6

    plt.rcParams.update({
        "font.family": "Arial", "font.size": FONT_TICK,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.4, "ytick.major.width": 0.4,
        "xtick.major.size": 1.6, "ytick.major.size": 1.6,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    n_subj, n_tr = subj_beta.shape
    tr_order = list(tr_order)
    xs = [t + 0.5 for t in tr_order]           # bin centres, in seconds
    mean = subj_beta.mean(axis=0)
    sem  = subj_beta.std(axis=0, ddof=1) / np.sqrt(n_subj)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)
    ax.fill_between(xs, mean - sem, mean + sem, color=DARK_SAGE, alpha=0.32,
                    linewidth=0)
    ax.plot(xs, mean, "-o", color=DARK_SAGE, lw=0.9, ms=1.8, mec="none", zorder=3)
    ax.axhline(0, color="k", lw=0.35, zorder=1)

    # ---- optional reference line (off by default -- see docstring) ----
    if peak_marker_x not in (None, "none"):
        ax.axvline(peak_marker_x, color="0.35", lw=0.6, ls=":", zorder=1)

    ax.set_xticks(tr_order[::2])
    ax.set_xlim(tr_order[0], tr_order[-1] + 1)

    # ---- headroom for the schedule strip, so it never sits on the data ----
    if reward_schedule:
        dmax, dmin = float((mean + sem).max()), float((mean - sem).min())
        pad = 0.04 * (dmax - dmin) if dmax > dmin else 0.001
        bottom = dmin - pad
        ax.set_ylim(bottom, bottom + (dmax - bottom) / 0.80)

    # ---- reward-cue schedule strip along the panel top ----
    # x in DATA SECONDS (matches the bin-centre trace above), y in axes
    # fraction -- a reward boundary falling mid-bin (e.g. 1.5 s, 4.5 s)
    # correctly renders as splitting that bin rather than being rounded away.
    if reward_schedule:
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        strip_lo, strip_hi = 0.905, 0.985
        x0, x1 = ax.get_xlim()
        for (a, b, label, col) in reward_schedule:
            if b <= x0 or a >= x1:
                continue
            a, b = max(a, x0), min(b, x1)
            ax.add_patch(plt.Rectangle((a, strip_lo), b - a, strip_hi - strip_lo,
                                         transform=trans, facecolor=col,
                                         edgecolor="white", lw=0.3,
                                         zorder=4, clip_on=False))
            ax.text((a + b) / 2, (strip_lo + strip_hi) / 2, label,
                    transform=trans, ha="center", va="center",
                    fontsize=FONT_LABEL - 1, color=_readable_text_colour(col),
                    zorder=5, clip_on=False)

    ax.set_xlabel(x_label, fontsize=FONT_AXIS, labelpad=1)
    ax.set_ylabel("β (cluster footprint)", fontsize=FONT_AXIS, labelpad=1)
    ax.tick_params(labelsize=FONT_TICK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    _save_all_formats(fig, save_path)
    plt.close(fig)


def plot_roi_mean_timecourse(roi_ts, tr_order, ref_t, clusters, mass_per_cl,
                              p_fwe_per_cl, model_name, mask_name, save_path,
                              x_label="TR"):
    """ROI-mean β per TR at publication size (4.5 × 3 cm, Arial).
    Dark sage line + SEM ribbon, FWE-significant TR runs shaded, tiny
    'p_FWE<.05' annotation above each sig run. Matches the SVC cluster-
    traces panel so they can appear side by side."""
    CM = 1.0 / 2.54
    FIG_W, FIG_H = 4.5 * CM, 3.0 * CM
    DARK_SAGE = "#3E6B4D"
    FONT_TICK = 6
    FONT_AXIS = 7
    FONT_LABEL = 6

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": FONT_TICK,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 1.6,
        "ytick.major.size": 1.6,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    n_subj, n_tr = roi_ts.shape
    xs = list(tr_order)
    mean = roi_ts.mean(axis=0)
    sem = roi_ts.std(axis=0, ddof=1) / np.sqrt(n_subj)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=True)

    # FWE-significant cluster shading (drawn first, low alpha, ROI colour
    # of the mask if known — else dark sage)
    ymax_est = float((mean + sem).max())
    for (a, b), m, p in zip(clusters, mass_per_cl, p_fwe_per_cl):
        if p < 0.05:
            ax.axvspan(xs[a] - 0.5, xs[b - 1] + 0.5,
                       color=DARK_SAGE, alpha=0.16, linewidth=0)
            mid = (xs[a] + xs[b - 1]) / 2
            ax.annotate("p$_{FWE}$<.05",
                        xy=(mid, ymax_est), xytext=(mid, ymax_est * 1.08),
                        ha="center", va="bottom",
                        fontsize=FONT_LABEL, color=DARK_SAGE)

    ax.fill_between(xs, mean - sem, mean + sem,
                    color=DARK_SAGE, alpha=0.32, linewidth=0)
    ax.plot(xs, mean, "-o", color=DARK_SAGE, lw=0.9, ms=1.8,
            mec="none", zorder=3)
    ax.axhline(0, color="k", lw=0.35, zorder=1)

    ax.set_xticks(xs[::2])
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    ax.set_xlabel(x_label, fontsize=FONT_AXIS, labelpad=1)
    ax.set_ylabel("β (mask mean)", fontsize=FONT_AXIS, labelpad=1)
    ax.tick_params(labelsize=FONT_TICK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    _save_all_formats(fig, save_path)
    plt.close(fig)


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
def _run_svc_and_roimean(model, data, tr_order, roi_mask, mask_name,
                          out_dir, affine, header, ref_t, n_perm, n_jobs,
                          structure, x_label="TR"):
    """Per-mask branch: (a) 4-D SVC cluster-mass test restricted to mask,
    (b) ROI-mean 1-D time-course cluster-mass test."""
    n_subj = data.shape[0]
    mask4d = roi_mask[..., None].astype(float)

    # ---- (a) SVC 4-D cluster-mass + voxel-FWE ----
    print(f"  [SVC 4D · {mask_name}] running {n_perm} sign-flip perms ...")
    perm_out = Parallel(n_jobs=n_jobs)(
        delayed(get_perm_max_mass)(data, seed=i, ref_t=ref_t,
                                    structure=structure, mask4d=mask4d)
        for i in range(n_perm)
    )
    null_max_mass = np.asarray([m for m, _ in perm_out])
    null_max_t    = np.asarray([t for _, t in perm_out])
    stat_map = get_stat(data) * mask4d
    # Voxel-wise FWE: p at each voxel = fraction of perms whose max t >= observed.
    voxel_p_fwe = np.ones_like(stat_map, dtype=np.float32)
    in_mask = mask4d[..., 0].astype(bool)      # 3-D mask
    for t_idx in range(stat_map.shape[-1]):
        obs = stat_map[..., t_idx][in_mask]
        p = np.mean(null_max_t[:, None] >= obs[None, :], axis=0).astype(np.float32)
        voxel_p_fwe[..., t_idx][in_mask] = p
    cluster_map, n_clusters = get_clusters(stat_map, ref_t, structure)
    cluster_mass = get_cluster_mass(stat_map, cluster_map, n_clusters)
    p_map = np.zeros_like(stat_map)
    recs = []
    for i in range(n_clusters):
        m = cluster_mass[i]
        if m == 0:
            continue
        cid = i + 1
        one_minus_p = float(np.mean(null_max_mass < m))
        p_map[cluster_map == cid] = one_minus_p
        peak = cluster_peak(cluster_map, stat_map, affine, cid)
        trace = cluster_trace(cluster_map, stat_map, cid)
        rec = {"cluster_id": int(cid), "cluster_mass": float(m),
                "one_minus_p_FWE": one_minus_p,
                "trace": [float(x) for x in trace], **peak}
        rec["peak_roi"] = _roi_from_mni(*rec["peak_mni"])
        # Voxel-wise FWE at the cluster peak (a priori peak-voxel test).
        pi, pj, pk = peak["peak_ijk"]; pt = peak["peak_tr_idx"]
        rec["peak_voxel_p_FWE"] = float(voxel_p_fwe[pi, pj, pk, pt])
        recs.append(rec)
    recs.sort(key=lambda r: r["cluster_mass"], reverse=True)

    stem = f"{mask_name}_svc4d_group_rdm-{model}"
    nib.save(nib.Nifti1Image(stat_map.astype(np.float32), affine, header),
              os.path.join(out_dir, f"{stem}_t.nii.gz"))
    nib.save(nib.Nifti1Image(p_map.astype(np.float32), affine, header),
              os.path.join(out_dir, f"{stem}_clust1minusFWEp.nii.gz"))
    nib.save(nib.Nifti1Image(cluster_map.astype(np.int32), affine, header),
              os.path.join(out_dir, f"{stem}_clusterlabels.nii.gz"))
    # Voxel-wise FWE: raw p at every in-mask voxel.
    nib.save(nib.Nifti1Image(voxel_p_fwe, affine, header),
              os.path.join(out_dir, f"{stem}_voxelFWEp.nii.gz"))
    # And the 1 - p_FWE convention to match the cluster map (threshold at 0.95).
    nib.save(nib.Nifti1Image((1.0 - voxel_p_fwe).astype(np.float32), affine, header),
              os.path.join(out_dir, f"{stem}_voxel1minusFWEp.nii.gz"))
    with open(os.path.join(out_dir, f"{stem}_clusters.json"), "w") as f:
        json.dump({"mask": mask_name, "TR_order": tr_order,
                    "n_subj": n_subj, "ref_t": ref_t, "n_perm": n_perm,
                    "clusters": recs}, f, indent=2)
    plot_cluster_traces(recs[:5], tr_order, ref_t,
                        f"{mask_name} · {model}",
                        os.path.join(out_dir, f"{stem}_cluster_traces.pdf"),
                        x_label=x_label)
    # Publication-style panel: mean β inside the top cluster's spatial
    # footprint per TR. Only the largest cluster (by mass).
    if recs:
        top = recs[0]
        beta_ts = cluster_footprint_beta(data, cluster_map, top["cluster_id"])
        plot_cluster_footprint_timecourse(
            beta_ts, tr_order, top, f"{mask_name} · {model}",
            os.path.join(out_dir, f"{stem}_footprint_beta.pdf"),
            x_label=x_label)
        np.save(os.path.join(out_dir, f"{stem}_footprint_persubj_beta.npy"),
                 beta_ts)

    # ---- (b) ROI-mean 1-D over TR ----
    roi_ts = roi_mean_per_subject_per_tr(data, roi_mask)   # (n_subj, n_TR)
    t_vec = group_t_per_tr(roi_ts)
    obs_clusters, obs_masses = clusters_1d(t_vec, ref_t)
    print(f"  [ROI-mean · {mask_name}] running {n_perm} sign-flip perms ...")
    null_max_1d = np.asarray(
        Parallel(n_jobs=n_jobs)(
            delayed(perm_max_mass_1d)(roi_ts, seed=100_000 + i, ref_t=ref_t)
            for i in range(n_perm)
        )
    )
    p_fwe_per_cl = [float(np.mean(null_max_1d >= m)) if m > 0 else 1.0
                     for m in obs_masses]

    # save per-subject β and group t per TR — the "effect map across time"
    pd_stem = f"{mask_name}_roimean_group_rdm-{model}"
    np.save(os.path.join(out_dir, f"{pd_stem}_persubj_beta.npy"), roi_ts)
    np.save(os.path.join(out_dir, f"{pd_stem}_group_t.npy"), t_vec)
    np.save(os.path.join(out_dir, f"{pd_stem}_null_max_mass.npy"), null_max_1d)
    with open(os.path.join(out_dir, f"{pd_stem}_clusters.json"), "w") as f:
        json.dump({"mask": mask_name, "TR_order": tr_order, "n_subj": n_subj,
                    "ref_t": ref_t, "n_perm": n_perm,
                    "clusters": [{"tr_start": tr_order[a], "tr_end": tr_order[b-1],
                                   "cluster_mass": float(m),
                                   "p_FWE": p} for (a, b), m, p in
                                  zip(obs_clusters, obs_masses, p_fwe_per_cl)]},
                   f, indent=2)
    plot_roi_mean_timecourse(roi_ts, tr_order, ref_t,
                             obs_clusters, obs_masses, p_fwe_per_cl,
                             model, mask_name,
                             os.path.join(out_dir, f"{pd_stem}_timecourse.pdf"),
                             x_label=x_label)


def run(root, output_dir, n_perm=10, p_thres=0.001, n_jobs=-1, models=MODELS,
        connectivity=1, top_n=5, tr_include=None, dir_glob=DIR_GLOB,
        masks=None, x_label="TR"):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump({
            "root": root, "output_dir": output_dir, "n_perm": n_perm,
            "p_thres": p_thres, "models": list(models),
            "connectivity": connectivity, "top_n": top_n,
            "tr_include": tr_include, "dir_glob": dir_glob,
            "masks": list(masks or []), "x_label": x_label,
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

    # Pre-load SVC masks (validate against the whole-brain data grid).
    svc_masks = {}   # name -> boolean mask array
    if masks:
        # Any data volume gives us the reference shape/affine; use the whole-
        # brain mask image loaded above (matches every model's spatial grid).
        for mp in masks:
            name = _mask_stem(mp)
            m = load_mask_nifti(mp, expected_shape=mask.shape,
                                 expected_affine=affine)
            svc_masks[name] = m
            print(f"  loaded SVC mask '{name}' — {int(m.sum())} voxels")

    for model in models:
        print(f"\n=== model: {model} ===")
        data, tr_order = load_model_data(model, tr_dirs)
        n_subj = data.shape[0]
        print(f"  data shape = {data.shape}")

        mask4d = mask[..., None].astype(float)
        ref_t = float(stats.t.ppf(1 - p_thres, n_subj - 1))
        print(f"  ref_t (p<{p_thres}, df={n_subj-1}) = {ref_t:.3f}")

        # --- per-mask SVC + ROI-mean branches ---
        for mask_name, roi_mask in svc_masks.items():
            svc_dir = os.path.join(output_dir, f"svc_{mask_name}")
            os.makedirs(svc_dir, exist_ok=True)
            _run_svc_and_roimean(
                model=model, data=data, tr_order=tr_order,
                roi_mask=roi_mask, mask_name=mask_name,
                out_dir=svc_dir, affine=affine, header=mask_img.header,
                ref_t=ref_t, n_perm=n_perm, n_jobs=n_jobs,
                structure=structure, x_label=x_label,
            )

        # ---- null (cluster mass + voxel-max t) ----
        print(f"  running {n_perm} sign-flip perms (n_jobs={n_jobs}) ...")
        
        # # just for debugging
        # for i in range(n_perm):
        #     perm_out = get_perm_max_mass(data, seed=i, ref_t=ref_t,
        #                            structure=structure, mask4d=mask4d)
        
        perm_out = Parallel(n_jobs=n_jobs)(
            delayed(get_perm_max_mass)(data, seed=i, ref_t=ref_t,
                                        structure=structure, mask4d=mask4d)
            for i in range(n_perm)
        )
        null_max_mass = np.asarray([m for m, _ in perm_out])
        null_max_t    = np.asarray([t for _, t in perm_out])

        # ---- observed ----
        stat_map = get_stat(data) * mask4d
        # voxel-wise FWE map (whole-brain family)
        voxel_p_fwe = np.ones_like(stat_map, dtype=np.float32)
        in_mask = mask4d[..., 0].astype(bool)
        for t_idx in range(stat_map.shape[-1]):
            obs = stat_map[..., t_idx][in_mask]
            p = np.mean(null_max_t[:, None] >= obs[None, :], axis=0).astype(np.float32)
            voxel_p_fwe[..., t_idx][in_mask] = p
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
            rec = {
                "cluster_id": int(cid),
                "cluster_mass": float(m),
                "one_minus_p_FWE": one_minus_p,
                "trace": [float(x) for x in trace],
                **peak,
            }
            rec["peak_roi"] = _roi_from_mni(*rec["peak_mni"])
            pi, pj, pk = peak["peak_ijk"]; pt = peak["peak_tr_idx"]
            rec["peak_voxel_p_FWE"] = float(voxel_p_fwe[pi, pj, pk, pt])
            cluster_records.append(rec)

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
        nib.save(nib.Nifti1Image(voxel_p_fwe, affine, header),
                 os.path.join(output_dir, f"group_rdm-{model}_voxelFWEp.nii.gz"))
        nib.save(nib.Nifti1Image((1.0 - voxel_p_fwe).astype(np.float32), affine, header),
                 os.path.join(output_dir, f"group_rdm-{model}_voxel1minusFWEp.nii.gz"))
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
            x_label=x_label,
        )
        # Publication-style panel for the top cluster only.
        if top_records:
            top = top_records[0]
            beta_ts = cluster_footprint_beta(data, cluster_map, top["cluster_id"])
            plot_cluster_footprint_timecourse(
                beta_ts, tr_order, top, model,
                os.path.join(output_dir, f"footprint_beta_{model}.pdf"),
                x_label=x_label)
            np.save(os.path.join(output_dir,
                                    f"footprint_persubj_beta_{model}.npy"),
                     beta_ts)

        all_stats[model] = (null_max_mass, cluster_records[:top_n])

    # ---- cross-model summary ----
    plot_null_vs_observed(
        all_stats, n_perm, os.path.join(output_dir, "null_vs_observed.pdf"),
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
    ap.add_argument("--n_perm", type=int, default=100)
    ap.add_argument("--p_thres", type=float, default=0.01)
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
    ap.add_argument("--masks", nargs="+", default=None,
                    help="Optional list of small-volume (SVC) mask NIfTIs. "
                         "For each mask, runs (1) a 4-D cluster-mass test with "
                         "the mask applied before cluster formation and before "
                         "the sign-flip null, and (2) a ROI-mean 1-D "
                         "time-course test (per-subject β averaged inside the "
                         "mask at each TR, sign-flip cluster-mass over TR). "
                         "Masks must match the data affine + spatial shape. "
                         "Outputs go to `<output_dir>/svc_<mask_stem>/`. "
                         "Example: --masks /path/mPFC.nii.gz /path/PFC_MTL.nii.gz")
    ap.add_argument("--x_label", default="TR",
                    help="Label used on the x-axis of the per-model plots. "
                         "For an EV-based instruction analysis with 1 EV/s "
                         "and no additional HRF delay to add, use "
                         "'EV (s from cue onset)'.")
    args = ap.parse_args()

    run(root=args.root, output_dir=args.output_dir, n_perm=args.n_perm,
        p_thres=args.p_thres, n_jobs=args.n_jobs, connectivity=args.connectivity,
        models=args.models, top_n=args.top_n, tr_include=args.tr_include,
        dir_glob=args.dir_glob, masks=args.masks, x_label=args.x_label)
