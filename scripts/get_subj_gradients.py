#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gradient analysis across subjects for linear trend.

Loops over multiple file-sets (4 or 8 ordered conditions) and over all
extraction modes (voxel / cluster_peak / cluster_com).

4-condition case:
- directed -cosine test
- hand-crafted trend: [-1, 0, 1, 0]

8-condition case:
- directed -cosine test
- hand-crafted trend: [-3, -2, -1, 0, 3, 0, -1, -2]
- circular regression / phase / Hotelling test
"""

import json
import os
import re
from datetime import datetime

import numpy as np
import nibabel as nib
import nilearn.image
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import ttest_rel, ttest_1samp, f as f_dist


# =====================================================
# ===================== SETTINGS ======================
# =====================================================

# Peak modes to iterate over
PEAK_MODES = ["voxel", "cluster_peak", "cluster_com"]

N_CLUSTERS = 3
CLUSTER_THRESHOLD = 90  # 0, "z", a value between 20 and 99 for percentile
AXIS = "z"              # "x", "y", "z"
PLOT_SUBJECT_LINES = True

# Figure colors
COSINE_COLOR = "darkred"
TREND_COLOR = "darkgreen"
OBSERVED_COLOR = "black"

axis_index = {"x": 0, "y": 1, "z": 2}[AXIS]

# Orderness measure
USE_ORDERNESS = True
ORDERNESS_METHOD = "monotonic"
# options:
# "monotonic"  -> strict increasing/decreasing consistency
# "correlation" -> correlation with expected order index


# =====================================================
# ======================= PATHS =======================
# =====================================================

source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group"
if not os.path.isdir(source_dir):
    source_dir = "/home/fs0/xpsy1114/scratch/data/derivatives/group"

result_dir = (
    f"{source_dir}/group_RSA_DSR_quarters_except_prev_button_state_glmbase_"
    f"all-paths-fixed_stickrews_split-buttons_cropped_masked"
)


# =====================================================
# ==================== DATASETS =======================
# =====================================================
# Each entry describes one file-set (4 or 8 conditions).
# "files" maps condition label -> filename inside result_dir.
# "label" is used for plot titles / on-disk figure folders.

DATASETS = [
    {
        "label": "quarters_button (4)",
        "n_conditions": 4,
        "files": {
            "current quarter":
                "CURR_QUARTER-split_quarters_DSR_except_prev_button_masked.nii.gz",
            "next quarter":
                "NEXT_QUARTER-split_quarters_DSR_except_prev_button_masked.nii.gz",
            "next +2 quarter":
                "NEXT2_QUARTER-split_quarters_DSR_except_prev_button_masked.nii.gz",
            "next +3 quarter":
                "NEXT3_QUARTER-split_quarters_DSR_except_prev_button_masked.nii.gz",
        },
    },
    {
        "label": "quarters_button_state (4)",
        "n_conditions": 4,
        "files": {
            "current quarter":
                "CURR_QUARTER-split_quarters_DSR_except_prev_button_state_masked.nii.gz",
            "next quarter":
                "NEXT_QUARTER-split_quarters_DSR_except_prev_button_state_masked.nii.gz",
            "next +2 quarter":
                "NEXT2_QUARTER-split_quarters_DSR_except_prev_button_state_masked.nii.gz",
            "next +3 quarter":
                "NEXT3_QUARTER-split_quarters_DSR_except_prev_button_state_masked.nii.gz",
        },
    },
    {
        "label": "rot_quarters (4)",
        "n_conditions": 4,
        "files": {
            "current quarter":
                "ROT_CURR_QUARTER-split_rot_quarters_DSR_except_prev_but_masked.nii.gz",
            "next quarter":
                "ROT_NEXT_QUARTER-split_rot_quarters_DSR_except_prev_but_masked.nii.gz",
            "next +2 quarter":
                "ROT_NEXT2_QUARTER-split_rot_quarters_DSR_except_prev_but_masked.nii.gz",
            "next +3 quarter":
                "ROT_NEXT3_QUARTER-split_rot_quarters_DSR_except_prev_but_masked.nii.gz",
        },
    },
    {
        "label": "eighths (8)",
        "n_conditions": 8,
        "files": {
            "now":
                "LOCATION-split_eighths_DSR_except_prev_button-mask_reward-path_beta_std.nii.gz",
            "+1 fut":
                "DSR_ONEFUT-split_eighths_DSR_except_prev_button_masked.nii.gz",
            "+2 fut":
                "DSR_TWOFUT-split_eighths_DSR_except_prev_button_masked.nii.gz",
            "+3 fut":
                "DSR_THREEFUT-split_eighths_DSR_except_prev_button_masked.nii.gz",
            "+4 fut":
                "DSR_FOURFUT-split_eighths_DSR_except_prev_button_masked.nii.gz",
            "+5 fut":
                "DSR_FIVEFUT-split_eighths_DSR_except_prev_button_masked.nii.gz",
            "+6 fut":
                "DSR_SIXFUT-split_eighths_DSR_except_prev_button_masked.nii.gz",
            "+7 fut":
                "DSR_SEVENFUT-split_eighths_DSR_except_prev_button_masked.nii.gz",
        },
    },
]


# =====================================================
# ============== PEAK / CLUSTER EXTRACTION ============
# =====================================================

def extract_clusters(subj_data, affine, mode="cluster_peak",
                     n_clusters=1, threshold=CLUSTER_THRESHOLD):
    if mode == "voxel":
        peak_index = np.unravel_index(np.argmax(subj_data), subj_data.shape)
        return [nib.affines.apply_affine(affine, peak_index)]

    if threshold == 0:
        binary = subj_data > 0
    elif threshold == "z":
        z = (subj_data - np.mean(subj_data)) / np.std(subj_data)
        binary = z > 1.0
    elif threshold > 20:
        threshold = np.percentile(subj_data, threshold)
        binary = subj_data > threshold
    labeled_array, n_found = label(binary)

    if n_found == 0:
        return []

    masses = []
    for cid in range(1, n_found + 1):
        mask = labeled_array == cid
        masses.append(subj_data[mask].sum())

    masses = np.array(masses)
    sorted_ids = np.argsort(masses)[::-1] + 1

    coords = []
    for cid in sorted_ids[:n_clusters]:
        mask = labeled_array == cid

        if mode == "cluster_peak":
            tmp = subj_data.copy()
            tmp[~mask] = -np.inf
            idx = np.unravel_index(np.argmax(tmp), subj_data.shape)

        elif mode == "cluster_com":
            idx = center_of_mass(subj_data, labeled_array, cid)

        else:
            raise ValueError("mode must be one of: voxel, cluster_peak, cluster_com")

        mni = nib.affines.apply_affine(affine, idx)
        coords.append(mni)

    return coords


def hotelling_t2_test(X):
    """
    One-sample Hotelling's T^2 test against zero vector.
    X: array of shape (n_subjects, 2)
    Returns: T2, F, p
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    if n <= p:
        raise ValueError("Hotelling's T^2 requires n_subjects > number of dimensions.")

    mean_vec = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.pinv(cov)

    t2 = n * float(mean_vec.T @ inv_cov @ mean_vec)
    f_stat = ((n - p) / (p * (n - 1))) * t2
    p_val = 1.0 - f_dist.cdf(f_stat, p, n - p)

    return t2, f_stat, p_val


def one_sided_p_greater(t_stat, p_two_sided):
    """One-sided p-value for a directional hypothesis expecting t > 0."""
    if t_stat > 0:
        return p_two_sided / 2
    return 1 - (p_two_sided / 2)


def scale_model_to_data(model, mean_vals):
    """Scale a theoretical model to the mean/SD of observed data for plotting only."""
    model = np.asarray(model, dtype=float)
    model = model - model.mean()
    maxabs = np.max(np.abs(model))
    if maxabs > 0:
        model = model / maxabs
    return model * np.std(mean_vals, ddof=1) + np.mean(mean_vals)


def plot_summary(matrix, conditions, axis_label, title,
                 main_color="black", plot_subject_lines=True, save_path=None):
    x_positions = np.arange(len(conditions))
    mean_vals = np.mean(matrix, axis=0)
    sem_vals = np.std(matrix, axis=0, ddof=1) / np.sqrt(matrix.shape[0])

    low = np.percentile(matrix, 5)
    high = np.percentile(matrix, 95)

    fig = plt.figure(figsize=(5.8, 5.2))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])

    if plot_subject_lines:
        for subj in range(matrix.shape[0]):
            ax.plot(x_positions, matrix[subj, :],
                    color="lightgray", alpha=0.45, linewidth=1)

    ax.fill_between(x_positions, mean_vals - sem_vals, mean_vals + sem_vals,
                    color=main_color, alpha=0.2)
    ax.errorbar(x_positions, mean_vals, yerr=sem_vals,
                marker="o", linewidth=3, capsize=5, color=main_color)

    ax.set_ylim(low, high)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, fontsize=14, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel(axis_label, fontsize=15)
    ax.set_title(title, fontsize=13, pad=15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        for ext in ("pdf", "png"):
            fig.savefig(f"{save_path}.{ext}", dpi=200, bbox_inches="tight")
    plt.show()


def plot_hypothesis_comparison(matrix, conditions, axis_label, plot_title,
                               weights_a, label_a, p_a,
                               weights_b, label_b, p_b,
                               color_a=COSINE_COLOR, color_b=TREND_COLOR,
                               plot_subject_lines=True, save_path=None):
    x_positions = np.arange(len(conditions))
    mean_vals = matrix.mean(axis=0)
    sem_vals = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])

    model_a = scale_model_to_data(weights_a, mean_vals)
    model_b = scale_model_to_data(weights_b, mean_vals)

    fig, ax = plt.subplots(figsize=(7, 5))

    if plot_subject_lines:
        for subj in range(matrix.shape[0]):
            ax.plot(x_positions, matrix[subj, :],
                    color="lightgray", alpha=0.35, linewidth=1)

    ax.fill_between(x_positions, mean_vals - sem_vals, mean_vals + sem_vals,
                    color="gray", alpha=0.2, zorder=1)
    ax.errorbar(x_positions, mean_vals, yerr=sem_vals,
                marker="o", linewidth=3, capsize=5,
                color=OBSERVED_COLOR, label="Observed mean", zorder=3)

    ax.plot(x_positions, model_a, linewidth=3, color=color_a,
            label=f"{label_a} (one-sided p={p_a:.3f})", zorder=4)
    ax.plot(x_positions, model_b, linewidth=3, linestyle="--", color=color_b,
            label=f"{label_b} (one-sided p={p_b:.3f})", zorder=4)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_xlabel("Condition")
    ax.set_ylabel(axis_label)
    ax.set_title(plot_title, fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        for ext in ("pdf", "png"):
            fig.savefig(f"{save_path}.{ext}", dpi=200, bbox_inches="tight")
    plt.show()


def monotonic_orderness(seq):
    seq = np.asarray(seq)
    diffs = np.diff(seq)
    return np.mean(diffs > 0)


def correlation_orderness(seq):
    seq = np.asarray(seq)
    ideal = np.arange(len(seq))
    return np.corrcoef(seq, ideal)[0, 1]


def build_orderness_sequences(projection, conditions, n_clusters):
    """
    Build per-subject flattened sequences:
    [cond1_peak1, cond1_peak2, ..., cond2_peak1, ...]
    """
    n_subjects = len(projection[0][conditions[0]])
    sequences = []

    for subj in range(n_subjects):
        seq = []
        valid = True
        for cond in conditions:
            for c in range(n_clusters):
                val = projection[c][cond][subj]
                if np.isnan(val):
                    valid = False
                seq.append(val)
        if valid:
            sequences.append(seq)

    return np.array(sequences)


# =====================================================
# ============== ANALYSIS PER DATASET / MODE ==========
# =====================================================

def _slug(text):
    """Filesystem-safe short label used in filenames."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def _to_serializable(obj):
    """Recursively convert numpy scalars / arrays to plain Python types
    so json.dump can handle them. NaN / inf are turned into None."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_serializable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _stats_to_markdown(stats):
    """Render one dataset×mode stats dict as a markdown block."""
    lines = []
    lines.append(f"### {stats['dataset_label']} — mode: `{stats['peak_mode']}`")
    lines.append("")
    lines.append(f"- **n subjects used:** {stats['n_subjects_used']}")
    lines.append(f"- **conditions ({stats['n_conditions']}):** "
                 + ", ".join(f"`{c}`" for c in stats['conditions']))
    lines.append(f"- **axis:** `{stats['axis']}`")
    lines.append("")
    lines.append("**Pairwise paired t (Bonferroni corrected):**")
    lines.append("")
    lines.append("| a | b | t | p_unc | p_corr |")
    lines.append("|---|---|---|---|---|")
    for r in stats['pairwise']:
        def _f(x): return "n/a" if x is None else f"{x:.4f}"
        lines.append(f"| {r['a']} | {r['b']} | {_f(r['t'])} "
                     f"| {_f(r['p_unc'])} | {_f(r['p_corr'])} |")
    lines.append("")
    lt = stats['linear_trend']
    lines.append(f"**Linear trend:** t={lt['t']}, p={lt['p']}")
    cs = stats['cosine_directional']
    lines.append(f"**−cosine directional:** t={cs['t']}, "
                 f"p_one_sided={cs['p_one_sided']}")
    hc = stats['handcrafted_trend']
    lines.append(f"**Hand-crafted trend ({hc['label']}):** "
                 f"t={hc['t']}, p_one_sided={hc['p_one_sided']}")
    if stats.get('circular'):
        c = stats['circular']
        lines.append(f"**Circular fit (Hotelling T²):** T²={c['T2']}, "
                     f"F={c['F']}, p={c['p']}, "
                     f"preferred_phase={c['preferred_phase_steps']}"
                     f"/{stats['n_conditions']}, amplitude={c['amplitude']}")
    if stats.get('orderness'):
        o = stats['orderness']
        lines.append(f"**Orderness ({o['method']}):** mean={o['mean']}, "
                     f"t={o['t']}, p={o['p']}")
    lines.append("")
    return "\n".join(lines)


def load_niftis(files_temp_order, base_dir):
    niftis = {}
    for cond, fname in files_temp_order.items():
        fpath = os.path.join(base_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Missing file for condition '{cond}': {fpath}")
        niftis[cond] = nilearn.image.load_img(fpath)
    return niftis


def extract_projection(niftis, conditions, peak_mode, n_clusters, axis_idx):
    results = {cond: [] for cond in conditions}
    for cond, img in niftis.items():
        data = img.get_fdata()
        n_subjects = data.shape[3]
        for subj in range(n_subjects):
            subj_data = data[:, :, :, subj]
            coords = extract_clusters(
                subj_data, img.affine,
                mode=peak_mode, n_clusters=n_clusters,
            )
            results[cond].append(coords)

    projection = {i: {} for i in range(n_clusters)}
    for cluster_idx in range(n_clusters):
        for cond in conditions:
            values = []
            for subj_coords in results[cond]:
                if cluster_idx < len(subj_coords):
                    values.append(subj_coords[cluster_idx][axis_idx])
                else:
                    values.append(np.nan)
            projection[cluster_idx][cond] = np.array(values)
    return projection


def run_stats_and_plots(projection, conditions, dataset_label, peak_mode,
                        save_dir=None):
    """Compute all gradient stats + render figures.

    If ``save_dir`` is provided, saves per-(dataset × mode) figures as
    PDF+PNG and a JSON of the computed statistics into that directory.
    Always returns the stats dict (or None if the matrix ends up empty).
    """
    n_conditions = len(conditions)
    if n_conditions not in (4, 7, 8):
        raise ValueError("Stats configured for 4, 7 or 8 conditions only.")

    print(f"\n================ {dataset_label} | mode={peak_mode} ================\n")

    cluster_idx = 0  # strongest cluster
    matrix = np.vstack([projection[cluster_idx][c] for c in conditions]).T
    valid_rows = ~np.isnan(matrix).any(axis=1)
    matrix = matrix[valid_rows]

    print(f"Using {matrix.shape[0]} subjects with complete data.\n")

    if matrix.shape[0] < 2:
        print("[WARN] fewer than 2 subjects with complete data — "
              "skipping stats and plots.")
        return None

    tag = f"{_slug(dataset_label)}__{_slug(peak_mode)}"

    # 1) Pairwise comparisons
    pairs = list(combinations(range(len(conditions)), 2))
    n_tests = len(pairs)
    print("Pairwise paired t-tests (Bonferroni corrected):")
    pairwise_records = []
    for i, j in pairs:
        t_stat, p_val = ttest_rel(matrix[:, i], matrix[:, j])
        p_corr = min(p_val * n_tests, 1.0)
        print(f"  {conditions[i]} vs {conditions[j]}: "
              f"t={t_stat:.3f}, p_unc={p_val:.4f}, p_corr={p_corr:.4f}")
        pairwise_records.append({
            'a': conditions[i], 'b': conditions[j],
            't': t_stat, 'p_unc': p_val, 'p_corr': p_corr,
        })

    # 2) Linear trend
    x_positions = np.arange(len(conditions))
    slopes = [np.polyfit(x_positions, matrix[s, :], 1)[0]
              for s in range(matrix.shape[0])]
    t_stat_linear, p_linear = ttest_1samp(slopes, 0)
    print(f"\nLinear trend test: t={t_stat_linear:.3f}, p={p_linear:.4f}")

    # 3) Directional tests
    angles = np.linspace(0, 2 * np.pi, n_conditions, endpoint=False)
    weights_cos = -np.cos(angles)
    weights_cos = weights_cos - weights_cos.mean()
    contrast_vals_cos = matrix @ weights_cos
    t_cos, p_cos_two = ttest_1samp(contrast_vals_cos, 0)
    p_cos_one = one_sided_p_greater(t_cos, p_cos_two)

    if n_conditions == 4:
        weights_peak_raw = np.array([-1, 0, 1, 0], dtype=float)
        trend_label = "Next +2 quarter"
    elif n_conditions == 8:
        weights_peak_raw = np.array([-3, -2, -1, 0, 3, 0, -1, -2], dtype=float)
        trend_label = "Next +4 fut"
    else:  # 7: same shape as 8 but with "now" dropped
        weights_peak_raw = np.array([-2, -1, 0, 3, 0, -1, -2], dtype=float)
        trend_label = "Next +4 fut"

    weights_peak = weights_peak_raw - weights_peak_raw.mean()
    contrast_vals_peak = matrix @ weights_peak
    t_peak, p_peak_two = ttest_1samp(contrast_vals_peak, 0)
    p_peak_one = one_sided_p_greater(t_peak, p_peak_two)

    print(f"\n-cosine test: t={t_cos:.3f}, p_one-sided={p_cos_one:.4f}")
    print(f"Hand-crafted trend ({trend_label}): "
          f"t={t_peak:.3f}, p_one-sided={p_peak_one:.4f}")

    means = matrix.mean(axis=0)
    if n_conditions == 8:
        print("Increasing 0..4:",
              means[0] < means[1] < means[2] < means[3] < means[4])
        print("Decreasing 4..7:",
              means[4] > means[5] > means[6] > means[7])
    elif n_conditions == 7:
        # "now" missing -> peak shifts from idx 4 to idx 3 ("+4 fut")
        print("Increasing 0..3:",
              means[0] < means[1] < means[2] < means[3])
        print("Decreasing 3..6:",
              means[3] > means[4] > means[5] > means[6])
    else:
        print("current < next < next+2:",
              means[0] < means[1] < means[2])
        print("next+2 > next+3:", means[2] > means[3])

    # 4) Optional circular test (8 or 7 conditions)
    circular_stats = None
    if n_conditions in (7, 8):
        circ_design = np.column_stack([
            np.ones(len(angles)), np.cos(angles), np.sin(angles)
        ])
        betas = []
        for subj in range(matrix.shape[0]):
            beta, *_ = np.linalg.lstsq(circ_design, matrix[subj, :], rcond=None)
            betas.append(beta)
        betas = np.asarray(betas)
        intercepts = betas[:, 0]
        cos_betas = betas[:, 1]
        sin_betas = betas[:, 2]

        circ_vec = np.column_stack([cos_betas, sin_betas])
        t2, f_stat, p_circ = hotelling_t2_test(circ_vec)
        mean_cos = np.mean(cos_betas)
        mean_sin = np.mean(sin_betas)
        preferred_phase = np.arctan2(mean_sin, mean_cos)
        preferred_phase_steps = (preferred_phase / (2 * np.pi)) * n_conditions
        preferred_phase_steps = preferred_phase_steps % n_conditions
        amplitude = np.sqrt(mean_cos ** 2 + mean_sin ** 2)

        print("\nCircular structure test:")
        print(f"  T^2={t2:.3f}, F={f_stat:.3f}, p={p_circ:.6f}")
        print(f"  Mean cosine beta={mean_cos:.4f}, "
              f"sine beta={mean_sin:.4f}, amplitude={amplitude:.4f}")
        print(f"  Preferred phase={preferred_phase_steps:.3f}/{n_conditions}")

        circular_stats = {
            "angles": angles,
            "mean_intercept": np.mean(intercepts),
            "mean_cos": mean_cos,
            "mean_sin": mean_sin,
            "preferred_phase_steps": preferred_phase_steps,
            "amplitude": amplitude,
            "T2": t2,
            "F":  f_stat,
            "p":  p_circ,
        }

    # 5) Orderness
    orderness_stats = None
    if USE_ORDERNESS:
        sequences = build_orderness_sequences(projection, conditions, N_CLUSTERS)
        print(f"\nOrderness analysis on {sequences.shape[0]} subjects")
        orderness_vals = []
        for s in range(sequences.shape[0]):
            seq = sequences[s]
            if ORDERNESS_METHOD == "monotonic":
                orderness_vals.append(monotonic_orderness(seq))
            elif ORDERNESS_METHOD == "correlation":
                orderness_vals.append(correlation_orderness(seq))
            else:
                raise ValueError("Unknown ORDERNESS_METHOD")
        orderness_vals = np.array(orderness_vals)
        chance = 0.5 if ORDERNESS_METHOD == "monotonic" else 0.0
        if orderness_vals.size >= 2:
            t_ord, p_ord = ttest_1samp(orderness_vals, chance)
            mean_ord = float(np.nanmean(orderness_vals))
        else:
            t_ord, p_ord, mean_ord = np.nan, np.nan, np.nan
        print(f"  method={ORDERNESS_METHOD}, "
              f"mean={mean_ord if np.isfinite(mean_ord) else 'nan'}, "
              f"t={t_ord}, p={p_ord}")
        orderness_stats = {
            'method':      ORDERNESS_METHOD,
            'chance':      chance,
            'n_subjects':  int(orderness_vals.size),
            'mean':        mean_ord,
            't':           float(t_ord) if np.isfinite(t_ord) else np.nan,
            'p':           float(p_ord) if np.isfinite(p_ord) else np.nan,
        }

    # 6) Figures
    main_color = "darkgreen" if (np.isfinite(p_linear) and p_linear < 0.05) else "black"
    axis_label = f"{AXIS}-coordinate (MNI mm)"
    title_prefix = f"{dataset_label} | {peak_mode}"

    fig1_path = os.path.join(save_dir, f"summary_{tag}") if save_dir else None
    plot_summary(
        matrix=matrix, conditions=conditions, axis_label=axis_label,
        title=f"{title_prefix}\nLinear trend t={t_stat_linear:.2f}, p={p_linear:.3f}",
        main_color=main_color, plot_subject_lines=PLOT_SUBJECT_LINES,
        save_path=fig1_path,
    )

    fig2_path = os.path.join(save_dir, f"hypothesis_{tag}") if save_dir else None
    plot_hypothesis_comparison(
        matrix=matrix, conditions=conditions, axis_label=axis_label,
        plot_title=f"{title_prefix} | {n_conditions}-cond hypothesis comparison",
        weights_a=weights_cos, label_a="-cosine", p_a=p_cos_one,
        weights_b=weights_peak, label_b="Hand-crafted trend", p_b=p_peak_one,
        color_a=COSINE_COLOR, color_b=TREND_COLOR,
        plot_subject_lines=PLOT_SUBJECT_LINES,
        save_path=fig2_path,
    )

    if circular_stats is not None:
        dense_angles = np.linspace(0, 2 * np.pi, 400)
        dense_x = dense_angles / (2 * np.pi) * n_conditions
        fitted_curve = (
            circular_stats["mean_intercept"]
            + circular_stats["mean_cos"] * np.cos(dense_angles)
            + circular_stats["mean_sin"] * np.sin(dense_angles)
        )
        mean_vals = matrix.mean(axis=0)
        sem_vals = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])

        fig, ax = plt.subplots(figsize=(7, 5))
        if PLOT_SUBJECT_LINES:
            for subj in range(matrix.shape[0]):
                ax.plot(x_positions, matrix[subj, :],
                        color="lightgray", alpha=0.4, linewidth=1)
        ax.fill_between(x_positions, mean_vals - sem_vals, mean_vals + sem_vals,
                        color="gray", alpha=0.2, zorder=1)
        ax.errorbar(x_positions, mean_vals, yerr=sem_vals,
                    marker="o", linewidth=2.5, capsize=5,
                    color=OBSERVED_COLOR, label="Observed mean", zorder=3)
        ax.plot(dense_x, fitted_curve, linewidth=3,
                color=COSINE_COLOR, label="Circular fit", zorder=4)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions, rotation=45, ha="right")
        ax.set_xlabel("Condition")
        ax.set_ylabel(axis_label)
        ax.set_title(
            f"{title_prefix}\nT² p={circular_stats['p']:.4g}, "
            f"phase={circular_stats['preferred_phase_steps']:.2f}/{n_conditions}",
            fontsize=13,
        )
        ax.legend()
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        if save_dir:
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(save_dir, f"circular_{tag}.{ext}"),
                            dpi=200, bbox_inches="tight")
        plt.show()

    # ── Assemble stats dict + persist per-(dataset × mode) JSON ──────
    stats = {
        'dataset_label':     dataset_label,
        'peak_mode':         peak_mode,
        'axis':              AXIS,
        'n_conditions':      int(n_conditions),
        'n_subjects_used':   int(matrix.shape[0]),
        'conditions':        list(conditions),
        'means_per_condition': means,
        'sem_per_condition':   (matrix.std(axis=0, ddof=1)
                                 / np.sqrt(matrix.shape[0])),
        'per_subject_matrix':  matrix,   # (n_subj, n_cond)
        'pairwise':            pairwise_records,
        'linear_trend': {
            't': t_stat_linear, 'p': p_linear,
            'per_subject_slopes': slopes,
        },
        'cosine_directional': {
            'weights': weights_cos,
            't': t_cos, 'p_two_sided': p_cos_two, 'p_one_sided': p_cos_one,
        },
        'handcrafted_trend': {
            'label':   trend_label,
            'weights': weights_peak_raw,
            't': t_peak, 'p_two_sided': p_peak_two, 'p_one_sided': p_peak_one,
        },
        'circular':  circular_stats,
        'orderness': orderness_stats,
    }
    stats_ser = _to_serializable(stats)

    if save_dir:
        with open(os.path.join(save_dir, f"stats_{tag}.json"), "w") as f:
            json.dump(stats_ser, f, indent=2)
        print(f"  ↳ saved figures + stats to {save_dir}  (tag: {tag})")

    return stats_ser


# =====================================================
# ===================== MAIN LOOP =====================
# =====================================================

def run_pipeline(datasets, base_dir, out_dir, roi_label="", postprocess=None):
    """Loop over datasets × peak modes, run stats, and persist everything.

    ``postprocess`` (optional) is a callable ``dict[str, Nifti] -> dict[str, Nifti]``
    invoked on the loaded niftis for each dataset, before extraction. Used
    e.g. by the lOFC gradient wrapper to apply an ROI mask in memory.

    Writes into ``out_dir``:
      summary_<slug>.{pdf,png}         — per (dataset × mode) linear-trend plot
      hypothesis_<slug>.{pdf,png}      — per (dataset × mode) contrast plot
      circular_<slug>.{pdf,png}        — per (dataset × mode) circular fit (7/8 cond)
      stats_<slug>.json                — per (dataset × mode) full stats
      all_gradient_results.json        — aggregate list of all stats dicts
      gradient_results.md              — human-readable markdown summary
    """
    os.makedirs(out_dir, exist_ok=True)
    all_stats = []
    run_meta = {
        "roi_label":            roi_label,
        "base_dir":             base_dir,
        "out_dir":              out_dir,
        "timestamp":            datetime.now().isoformat(timespec="seconds"),
        "axis":                 AXIS,
        "peak_modes":           list(PEAK_MODES),
        "n_clusters":           N_CLUSTERS,
        "cluster_threshold":    CLUSTER_THRESHOLD,
        "orderness_method":     ORDERNESS_METHOD if USE_ORDERNESS else None,
    }

    for dataset in datasets:
        label_str = dataset["label"]
        files_temp_order = dict(dataset["files"])  # copy so we can prune

        # Drop missing files instead of aborting the whole dataset.
        missing = [
            (cond, fn) for cond, fn in files_temp_order.items()
            if not os.path.isfile(os.path.join(base_dir, fn))
        ]
        for cond, fn in missing:
            print(f"[WARN] '{label_str}': missing file for '{cond}' "
                  f"({fn}) — dropping this condition.")
            del files_temp_order[cond]

        conditions = list(files_temp_order.keys())
        n_conditions = len(conditions)
        if n_conditions not in (4, 7, 8):
            print(f"[SKIP] Dataset '{label_str}': "
                  f"{n_conditions} conditions left after pruning "
                  f"(need 4, 7, or 8).")
            continue

        niftis = load_niftis(files_temp_order, base_dir)
        if postprocess is not None:
            niftis = postprocess(niftis)

        for peak_mode in PEAK_MODES:
            projection = extract_projection(
                niftis, conditions,
                peak_mode=peak_mode,
                n_clusters=N_CLUSTERS,
                axis_idx=axis_index,
            )
            stats = run_stats_and_plots(
                projection=projection,
                conditions=conditions,
                dataset_label=label_str,
                peak_mode=peak_mode,
                save_dir=out_dir,
            )
            if stats is not None:
                all_stats.append(stats)

    # Aggregate outputs
    aggregate_path = os.path.join(out_dir, "all_gradient_results.json")
    with open(aggregate_path, "w") as f:
        json.dump({"run": run_meta, "results": all_stats}, f, indent=2)

    md_path = os.path.join(out_dir, "gradient_results.md")
    with open(md_path, "w") as f:
        f.write(f"# Gradient results — {roi_label or 'run'}\n\n")
        f.write(f"- **timestamp:** {run_meta['timestamp']}\n")
        f.write(f"- **base_dir:** `{base_dir}`\n")
        f.write(f"- **axis:** `{AXIS}`\n")
        f.write(f"- **peak modes:** {', '.join(PEAK_MODES)}\n")
        f.write(f"- **cluster threshold:** {CLUSTER_THRESHOLD}\n")
        if USE_ORDERNESS:
            f.write(f"- **orderness method:** {ORDERNESS_METHOD}\n")
        f.write("\n")
        for s in all_stats:
            f.write(_stats_to_markdown(s))
            f.write("\n---\n\n")

    print(f"\n{'='*60}\n"
          f"Saved {len(all_stats)} (dataset × mode) result blocks to:\n"
          f"  {out_dir}\n"
          f"  ↳ all_gradient_results.json\n"
          f"  ↳ gradient_results.md\n"
          f"{'='*60}")
    return all_stats


if __name__ == "__main__":
    _ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    OUT_DIR = os.path.join(result_dir, f"gradient_results_{_ts}")
    run_pipeline(DATASETS, base_dir=result_dir, out_dir=OUT_DIR,
                 roi_label="mPFC")
