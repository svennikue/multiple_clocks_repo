#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gradient analysis across subjects
Supports 4 or 8 ordered conditions.

4-condition case:
- quarter-based files
- directed -cosine test
- hand-crafted trend: [-1, 0, 1, 0]
- publishable figures

8-condition case:
- eighth-based files
- directed -cosine test
- hand-crafted trend: [-3, -2, -1, 0, 3, 0, -1, -2]
- circular regression / phase / Hotelling test
- publishable figures
"""

import os
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

# Choose how many inputs you want to use:
#   4 -> current quarter, next quarter, next +2 quarter, next +3 quarter
#   8 -> now, +1 fut, +2 fut, +3 fut, +4 fut, +5 fut, +6 fut, +7 fut
INPUT_MODE = 4 

# Choose extraction mode:
# "voxel"        -> strongest voxel
# "cluster_peak" -> peak within largest clusters
# "cluster_com"  -> center-of-mass of largest clusters
# PEAK_MODE = "cluster_peak"
# PEAK_MODE = "cluster_com"
PEAK_MODE = "voxel"

N_CLUSTERS = 3
CLUSTER_THRESHOLD = 90 # 0, "z", a value between 20 and 99 for percentile
AXIS = "z"           # "x", "y", "z"
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
    f"{source_dir}/group_RSA_which-fut-isn-DSR_glmbase_all-paths-fixed_"
    f"stickrews_split-buttons_cropped"
)


# =====================================================
# ==================== FILE SETS ======================
# =====================================================

FILESETS = {
    4: {
        "current quarter": "CURRENT_QUARTER-split_quarters_DSR_beta.std.nii.gz",
        "next quarter": "NEXT_QUARTER-split_quarters_DSR_beta.std.nii.gz",
        "next +2 quarter": "NEXT2_QUARTER-split_quarters_DSR_beta.std.nii.gz",
        "next +3 quarter": "NEXT3_QUARTER-split_quarters_DSR_beta.std.nii.gz",
    },
    8: {
        "now": "LOCATION-split_eighths_masked_beta.nii.gz",
        "+1 fut": "ONEFUT-split_eighths_masked_beta.nii.gz",
        "+2 fut": "TWOFUT-split_eighths_masked_beta.nii.gz",
        "+3 fut": "THREEFUT-split_eighths_masked_beta.nii.gz",
        "+4 fut": "FOURFUT-split_eighths_masked_beta.nii.gz",
        "+5 fut": "FIVEFUT-split_eighths_masked_beta.nii.gz",
        "+6 fut": "SIXFUT-split_eighths_masked_beta.nii.gz",
        "+7 fut": "SEVENFUT-split_eighths_masked_beta.nii.gz",
    }
}

if INPUT_MODE not in FILESETS:
    raise ValueError(f"INPUT_MODE must be one of {list(FILESETS.keys())}")

files_temp_order = FILESETS[INPUT_MODE]
conditions = list(files_temp_order.keys())
n_conditions = len(conditions)


# =====================================================
# ================= LOAD IMAGES =======================
# =====================================================

niftis = {}
for cond, fname in files_temp_order.items():
    fpath = os.path.join(result_dir, fname)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Missing file for condition '{cond}': {fpath}")
    niftis[cond] = nilearn.image.load_img(fpath)


# =====================================================
# ============== PEAK / CLUSTER EXTRACTION ============
# =====================================================

def extract_clusters(subj_data, affine, mode="cluster_peak", n_clusters=1, threshold = CLUSTER_THRESHOLD):
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
    """
    One-sided p-value for a directional hypothesis expecting t > 0.
    """
    if t_stat > 0:
        return p_two_sided / 2
    return 1 - (p_two_sided / 2)


def scale_model_to_data(model, mean_vals):
    """
    Scale a theoretical model to the mean/SD of observed data for plotting only.
    """
    model = np.asarray(model, dtype=float)
    model = model - model.mean()
    maxabs = np.max(np.abs(model))
    if maxabs > 0:
        model = model / maxabs
    return model * np.std(mean_vals, ddof=1) + np.mean(mean_vals)


def plot_summary(matrix, conditions, axis_label, title, main_color="black", plot_subject_lines=True):
    x_positions = np.arange(len(conditions))
    mean_vals = np.mean(matrix, axis=0)
    sem_vals = np.std(matrix, axis=0, ddof=1) / np.sqrt(matrix.shape[0])

    low = np.percentile(matrix, 5)
    high = np.percentile(matrix, 95)

    fig = plt.figure(figsize=(5.8, 5.2))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])

    if plot_subject_lines:
        for subj in range(matrix.shape[0]):
            ax.plot(
                x_positions,
                matrix[subj, :],
                color="lightgray",
                alpha=0.45,
                linewidth=1
            )

    ax.fill_between(
        x_positions,
        mean_vals - sem_vals,
        mean_vals + sem_vals,
        color=main_color,
        alpha=0.2
    )

    ax.errorbar(
        x_positions,
        mean_vals,
        yerr=sem_vals,
        marker="o",
        linewidth=3,
        capsize=5,
        color=main_color
    )

    ax.set_ylim(low, high)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, fontsize=14, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel(axis_label, fontsize=15)
    ax.set_title(title, fontsize=13, pad=15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_hypothesis_comparison(
    matrix,
    conditions,
    axis_label,
    plot_title,
    weights_a,
    label_a,
    p_a,
    weights_b,
    label_b,
    p_b,
    color_a=COSINE_COLOR,
    color_b=TREND_COLOR,
    plot_subject_lines=True,
):
    x_positions = np.arange(len(conditions))
    mean_vals = matrix.mean(axis=0)
    sem_vals = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])

    model_a = scale_model_to_data(weights_a, mean_vals)
    model_b = scale_model_to_data(weights_b, mean_vals)

    fig, ax = plt.subplots(figsize=(7, 5))

    if plot_subject_lines:
        for subj in range(matrix.shape[0]):
            ax.plot(
                x_positions,
                matrix[subj, :],
                color="lightgray",
                alpha=0.35,
                linewidth=1
            )

    ax.fill_between(
        x_positions,
        mean_vals - sem_vals,
        mean_vals + sem_vals,
        color="gray",
        alpha=0.2,
        zorder=1
    )

    ax.errorbar(
        x_positions,
        mean_vals,
        yerr=sem_vals,
        marker="o",
        linewidth=3,
        capsize=5,
        color=OBSERVED_COLOR,
        label="Observed mean",
        zorder=3
    )

    ax.plot(
        x_positions,
        model_a,
        linewidth=3,
        color=color_a,
        label=f"{label_a} (one-sided p={p_a:.3f})",
        zorder=4
    )

    ax.plot(
        x_positions,
        model_b,
        linewidth=3,
        linestyle="--",
        color=color_b,
        label=f"{label_b} (one-sided p={p_b:.3f})",
        zorder=4
    )

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
    plt.show()

def monotonic_orderness(seq):
    """
    Fraction of correctly ordered steps.
    """
    seq = np.asarray(seq)
    diffs = np.diff(seq)
    return np.mean(diffs > 0)  # increasing assumption


def correlation_orderness(seq):
    """
    Correlation with ideal linear ordering.
    """
    seq = np.asarray(seq)
    ideal = np.arange(len(seq))
    return np.corrcoef(seq, ideal)[0, 1]


# =====================================================
# ================= RUN EXTRACTION ====================
# =====================================================

results = {cond: [] for cond in conditions}

for cond, img in niftis.items():
    data = img.get_fdata()
    n_subjects = data.shape[3]

    for subj in range(n_subjects):
        subj_data = data[:, :, :, subj]
        coords = extract_clusters(
            subj_data,
            img.affine,
            mode=PEAK_MODE,
            n_clusters=N_CLUSTERS
        )
        results[cond].append(coords)


# =====================================================
# ================= AXIS PROJECTION ===================
# =====================================================

projection = {i: {} for i in range(N_CLUSTERS)}

for cluster_idx in range(N_CLUSTERS):
    for cond in conditions:
        values = []
        for subj_coords in results[cond]:
            if cluster_idx < len(subj_coords):
                values.append(subj_coords[cluster_idx][axis_index])
            else:
                values.append(np.nan)
        projection[cluster_idx][cond] = np.array(values)

import pdb; pdb.set_trace()

# =====================================================
# ============ BUILD ORDERED SEQUENCES =================
# =====================================================

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
# ====================== STATS ========================
# =====================================================

print("\n================ STATISTICS ================\n")

cluster_idx = 0  # strongest cluster

# import pdb; pdb.set_trace()
# here I could play around with the different clusters.

matrix = np.vstack([projection[cluster_idx][c] for c in conditions]).T

valid_rows = ~np.isnan(matrix).any(axis=1)
matrix = matrix[valid_rows]

print(f"Using {matrix.shape[0]} subjects with complete data for stats.\n")

# ------------------------------------------
# 1) Pairwise comparisons
# ------------------------------------------

pairs = list(combinations(range(len(conditions)), 2))
n_tests = len(pairs)

print("Pairwise paired t-tests (Bonferroni corrected):\n")

for i, j in pairs:
    t_stat, p_val = ttest_rel(matrix[:, i], matrix[:, j])
    p_corr = min(p_val * n_tests, 1.0)
    print(
        f"{conditions[i]} vs {conditions[j]}: "
        f"t={t_stat:.3f}, p_unc={p_val:.4f}, p_corr={p_corr:.4f}"
    )

# ------------------------------------------
# 2) Linear trend test
# ------------------------------------------

x_positions = np.arange(len(conditions))
slopes = []

for subj in range(matrix.shape[0]):
    slope = np.polyfit(x_positions, matrix[subj, :], 1)[0]
    slopes.append(slope)

t_stat_linear, p_linear = ttest_1samp(slopes, 0)

print("\nLinear trend test:")
print(f"t={t_stat_linear:.3f}, p={p_linear:.4f}")


# =====================================================
# ========= MODE-SPECIFIC DIRECTIONAL TESTS ===========
# =====================================================

if n_conditions not in (4, 8):
    raise ValueError("This script is set up for 4 or 8 conditions only.")

angles = np.linspace(0, 2 * np.pi, n_conditions, endpoint=False)

# ------------------------------------------
# Directed circular test: -cosine
# ------------------------------------------

weights_cos = -np.cos(angles)
weights_cos = weights_cos - weights_cos.mean()

contrast_vals_cos = matrix @ weights_cos
t_cos, p_cos_two = ttest_1samp(contrast_vals_cos, 0)
p_cos_one = one_sided_p_greater(t_cos, p_cos_two)

# ------------------------------------------
# Hand-crafted trend (user-specified)
# 4-condition: [-1, 0, 1, 0]
# 8-condition: [-3, -2, -1, 0, 3, 0, -1, -2]
# ------------------------------------------

if n_conditions == 4:
    weights_peak_raw = np.array([-1, 0, 1, 0], dtype=float)
    trend_label = "Next +2 quarter"
else:
    weights_peak_raw = np.array([-3, -2, -1, 0, 3, 0, -1, -2], dtype=float)
    trend_label = "Next +4 fut"

weights_peak = weights_peak_raw - weights_peak_raw.mean()

contrast_vals_peak = matrix @ weights_peak
t_peak, p_peak_two = ttest_1samp(contrast_vals_peak, 0)
p_peak_one = one_sided_p_greater(t_peak, p_peak_two)

print("\nDirected circular test (-cosine):")
print(f"Weights: {weights_cos}")
print(f"t={t_cos:.3f}")
print(f"p (two-sided)={p_cos_two:.4f}")
print(f"p (one-sided)={p_cos_one:.4f}")

print(f"\nHand-crafted trend ({trend_label} is furthest away):")
print(f"Weights: {weights_peak}")
print(f"t={t_peak:.3f}")
print(f"p (two-sided)={p_peak_two:.4f}")
print(f"p (one-sided)={p_peak_one:.4f}")

print("\nStrict ordering checks:")
means = matrix.mean(axis=0)

if n_conditions == 8:
    print("Increasing part:", means[0] < means[1] < means[2] < means[3] < means[4])
    print("Decreasing part:", means[4] > means[5] > means[6] > means[7])
else:
    print("current < next < next+2:", means[0] < means[1] < means[2])
    print("next+2 > next+3:", means[2] > means[3])


# =====================================================
# ========== OPTIONAL 8-CONDITION CIRCULAR TEST =======
# =====================================================

circular_stats = None

if n_conditions == 8:
    circ_design = np.column_stack([
        np.ones(len(angles)),
        np.cos(angles),
        np.sin(angles)
    ])

    betas = []
    for subj in range(matrix.shape[0]):
        y = matrix[subj, :]
        beta, *_ = np.linalg.lstsq(circ_design, y, rcond=None)
        betas.append(beta)

    betas = np.asarray(betas)
    intercepts = betas[:, 0]
    cos_betas = betas[:, 1]
    sin_betas = betas[:, 2]

    circ_vec = np.column_stack([cos_betas, sin_betas])
    t2, f_stat, p_circ = hotelling_t2_test(circ_vec)

    mean_intercept = np.mean(intercepts)
    mean_cos = np.mean(cos_betas)
    mean_sin = np.mean(sin_betas)

    preferred_phase = np.arctan2(mean_sin, mean_cos)
    preferred_phase_steps = (preferred_phase / (2 * np.pi)) * n_conditions
    preferred_phase_steps = preferred_phase_steps % n_conditions

    amplitude = np.sqrt(mean_cos ** 2 + mean_sin ** 2)

    print("\nCircular structure test (sine/cosine regression):")
    print(f"Hotelling T^2={t2:.3f}, F={f_stat:.3f}, p={p_circ:.6f}")
    print(f"Mean cosine beta={mean_cos:.4f}")
    print(f"Mean sine beta={mean_sin:.4f}")
    print(f"Circular amplitude={amplitude:.4f}")
    print(f"Preferred phase (condition units)={preferred_phase_steps:.3f} / {n_conditions}")

    circular_stats = {
        "angles": angles,
        "mean_intercept": mean_intercept,
        "mean_cos": mean_cos,
        "mean_sin": mean_sin,
        "preferred_phase_steps": preferred_phase_steps,
        "amplitude": amplitude,
        "p_circ": p_circ,
    }


# =====================================================
# ================= ORDERNESS ANALYSIS =================
# =====================================================

if USE_ORDERNESS:
    sequences = build_orderness_sequences(projection, conditions, N_CLUSTERS)

    print(f"\nUsing {sequences.shape[0]} subjects for orderness analysis.\n")

    orderness_vals = []

    for subj in range(sequences.shape[0]):
        seq = sequences[subj]

        if ORDERNESS_METHOD == "monotonic":
            score = monotonic_orderness(seq)

        elif ORDERNESS_METHOD == "correlation":
            score = correlation_orderness(seq)

        else:
            raise ValueError("Unknown ORDERNESS_METHOD")

        orderness_vals.append(score)

    orderness_vals = np.array(orderness_vals)

    # Test against chance
    if ORDERNESS_METHOD == "monotonic":
        chance = 0.5
    else:
        chance = 0.0

    t_ord, p_ord = ttest_1samp(orderness_vals, chance)

    print("Orderness test:")
    print(f"Method: {ORDERNESS_METHOD}")
    print(f"Mean orderness = {orderness_vals.mean():.3f}")
    print(f"t = {t_ord:.3f}, p = {p_ord:.4f}")
    
# =====================================================
# =========== PUBLISH-READY FIGURES ===================
# =====================================================

main_color = "darkgreen" if p_linear < 0.05 else "black"
axis_label = f"{AXIS}-coordinate (MNI mm)"

# Summary plot
plot_summary(
    matrix=matrix,
    conditions=conditions,
    axis_label=axis_label,
    title=f"Linear trend: t={t_stat_linear:.2f}, p={p_linear:.3f}",
    main_color=main_color,
    plot_subject_lines=PLOT_SUBJECT_LINES,
)

# Hypothesis comparison plot
plot_hypothesis_comparison(
    matrix=matrix,
    conditions=conditions,
    axis_label=axis_label,
    plot_title=f"{n_conditions}-condition hypothesis comparison",
    weights_a=weights_cos,
    label_a="-cosine",
    p_a=p_cos_one,
    weights_b=weights_peak,
    label_b="Hand-crafted trend",
    p_b=p_peak_one,
    color_a=COSINE_COLOR,
    color_b=TREND_COLOR,
    plot_subject_lines=PLOT_SUBJECT_LINES,
)

# 8-condition circular fit plot
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
            ax.plot(
                x_positions,
                matrix[subj, :],
                color="lightgray",
                alpha=0.4,
                linewidth=1
            )

    ax.fill_between(
        x_positions,
        mean_vals - sem_vals,
        mean_vals + sem_vals,
        color="gray",
        alpha=0.2,
        zorder=1
    )

    ax.errorbar(
        x_positions,
        mean_vals,
        yerr=sem_vals,
        marker="o",
        linewidth=2.5,
        capsize=5,
        color=OBSERVED_COLOR,
        label="Observed mean",
        zorder=3
    )

    ax.plot(
        dense_x,
        fitted_curve,
        linewidth=3,
        color=COSINE_COLOR,
        label="Circular fit",
        zorder=4
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.set_xlabel("Condition")
    ax.set_ylabel(axis_label)
    ax.set_title(
        f"Circular structure: Hotelling T² p={circular_stats['p_circ']:.4g}, "
        f"preferred phase={circular_stats['preferred_phase_steps']:.2f}/{n_conditions}",
        fontsize=13
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()