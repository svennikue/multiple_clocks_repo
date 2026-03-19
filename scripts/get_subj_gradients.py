#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gradient analysis across subjects
Clean modular version
"""

import os
import numpy as np
import nibabel as nib
import nilearn.image
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import ttest_rel, ttest_1samp


# =====================================================
# ===================== SETTINGS ======================
# =====================================================

# PEAK_MODE = "cluster_peak"
PEAK_MODE = "cluster_com"
# options:
# "voxel"         -> strongest voxel
# "cluster_peak"  -> peak within largest clusters
# "cluster_com"   -> center-of-mass of largest clusters

N_CLUSTERS = 1       # 1, 2, or 3 clusters
AXIS = "z"           # "x", "y", "z"
PLOT_SUBJECT_LINES = True

# =====================================================


# ---------- PATHS ----------
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group"
if not os.path.isdir(source_dir):
    source_dir = "/home/fs0/xpsy1114/scratch/data/derivatives/group"

result_dir = f"{source_dir}/group_RSA_which-fut-isn-DSR_glmbase_all-paths-fixed_stickrews_split-buttons_cropped"

files_temp_order = {
    "curr":  "CURRENT_QUARTER-split_quarters_DSR_beta.std.nii.gz",
    "next":  "NEXT_QUARTER-split_quarters_DSR_beta.std.nii.gz",
    "next2": "NEXT2_QUARTER-split_quarters_DSR_beta.std.nii.gz",
    "next3": "NEXT3_QUARTER-split_quarters_DSR_beta.std.nii.gz"
}

conditions = ["curr", "next", "next2", "next3"]
axis_index = {"x": 0, "y": 1, "z": 2}[AXIS]


# ---------- LOAD IMAGES ----------
niftis = {
    cond: nilearn.image.load_img(f"{result_dir}/{fname}")
    for cond, fname in files_temp_order.items()
}


# =====================================================
# ============== PEAK / CLUSTER EXTRACTION ============
# =====================================================

def extract_clusters(subj_data, affine, mode="cluster_peak", n_clusters=1):

    if mode == "voxel":
        peak_index = np.unravel_index(np.argmax(subj_data), subj_data.shape)
        return [nib.affines.apply_affine(affine, peak_index)]

    # ----- CLUSTER BASED -----
    binary = subj_data > 0
    labeled_array, n_found = label(binary)

    if n_found == 0:
        return []

    # compute cluster mass
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

        mni = nib.affines.apply_affine(affine, idx)
        coords.append(mni)

    return coords


# =====================================================
# ================== RUN EXTRACTION ===================
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
# =================== AXIS PROJECTION =================
# =====================================================

# results structure:
# results[condition][subject][cluster_index][axis]

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


# =====================================================
# ====================== PLOTTING =====================
# =====================================================

x_positions = np.arange(len(conditions))
n_subjects = len(results["curr"])

# line styles for up to 3 clusters
linestyles = ["-", "--", ":"]
colors = ["tab:blue", "tab:orange", "tab:green"]

plt.figure(figsize=(9, 6))

for cluster_idx in range(N_CLUSTERS):

    matrix = np.vstack(
        [projection[cluster_idx][c] for c in conditions]
    ).T

    mean_vals = np.nanmean(matrix, axis=0)
    sem_vals = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(n_subjects)

    style = linestyles[cluster_idx]
    color = colors[cluster_idx]

    # -------- SUBJECT LINES --------
    if PLOT_SUBJECT_LINES:
        for subj in range(n_subjects):
            plt.plot(
                x_positions,
                matrix[subj, :],
                color="lightgray",
                alpha=0.5,
                linewidth=1,
                linestyle=style
            )

    # -------- MEAN LINE --------
    plt.errorbar(
        x_positions,
        mean_vals,
        yerr=sem_vals,
        marker="o",
        linewidth=3,
        capsize=5,
        linestyle=style,
        color=color,
        label=f"Cluster {cluster_idx+1}"
    )

plt.xticks(x_positions, conditions)
plt.ylabel(f"{AXIS}-coordinate (MNI mm)")
plt.xlabel("Condition")
plt.title(f"{PEAK_MODE} | Top {N_CLUSTERS} cluster(s)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =====================================================
# ====================== STATS ========================
# =====================================================

print("\n================ STATISTICS ================\n")

cluster_idx = 0  # test strongest cluster

matrix = np.vstack(
    [projection[cluster_idx][c] for c in conditions]
).T

# ------------------------------------------
# 1️⃣ Pairwise comparisons
# ------------------------------------------

pair_results = []
pairs = list(combinations(range(len(conditions)), 2))
n_tests = len(pairs)

print("Pairwise paired t-tests (Bonferroni corrected):\n")

for i, j in pairs:
    
    t_stat, p_val = ttest_rel(matrix[:, i], matrix[:, j])
    p_corr = min(p_val * n_tests, 1.0)
    
    pair_results.append((i, j, p_corr))
    
    print(
        f"{conditions[i]} vs {conditions[j]}: "
        f"t={t_stat:.3f}, p_unc={p_val:.4f}, p_corr={p_corr:.4f}"
    )

# ------------------------------------------
# 2️⃣ Linear trend test
# ------------------------------------------

x_positions = np.arange(len(conditions))
slopes = []

for subj in range(matrix.shape[0]):
    slope = np.polyfit(x_positions, matrix[subj, :], 1)[0]
    slopes.append(slope)

t_stat, p_linear = ttest_1samp(slopes, 0)

print("\nLinear trend test:")
print(f"t={t_stat:.3f}, p={p_linear:.4f}")

# =====================================================
# =========== SIGNIFICANCE PLOT =======================
# =====================================================

low = np.percentile(matrix, 5)
high = np.percentile(matrix, 95)

# =====================================================
# ===== INFORMATIVE COMPACT LINEAR TREND PANEL =======
# =====================================================

mean_vals = matrix.mean(axis=0)
sem_vals = matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])

# choose color based on significance
if p_linear < 0.05:
    main_color = "darkred"
else:
    main_color = "black"

fig = plt.figure(figsize=(5, 5))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])

# ---------- SUBJECT LINES ----------
for subj in range(matrix.shape[0]):
    ax.plot(
        x_positions,
        matrix[subj, :],
        color="lightgray",
        alpha=0.5,
        linewidth=1
    )

# ---------- SEM SHADED BAND ----------
ax.fill_between(
    x_positions,
    mean_vals - sem_vals,
    mean_vals + sem_vals,
    color=main_color,
    alpha=0.2
)

# ---------- MEAN LINE ----------
ax.plot(
    x_positions,
    mean_vals,
    linewidth=5,
    color=main_color
)

# ---------- Y-LIMITS in the 95% interval of data ----------
data_min = matrix.min()
data_max = matrix.max()
data_range = data_max - data_min
ax.set_ylim(low, high)

# ---------- AXIS FORMAT ----------
ax.set_xticks(x_positions)
ax.set_xticklabels(conditions, fontsize=16)
ax.tick_params(axis='y', labelsize=14)

ax.set_ylabel(f"{AXIS}-coordinate (MNI mm)", fontsize=16)

# ---------- SIGNIFICANCE STARS ----------
if p_linear < 0.001:
    stars = "***"
elif p_linear < 0.01:
    stars = "**"
elif p_linear < 0.05:
    stars = "*"
else:
    stars = "n.s."

ax.text(
    0.5,
    0.92,
    stars,
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=28,
    fontweight="bold",
    color=main_color
)

# ---------- TITLE WITH STATISTICS ----------
ax.set_title(
    f"Linear trend: t={t_stat:.2f}, p={p_linear:.3f}",
    fontsize=14,
    pad=15
)

# remove unnecessary spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.show()
plt.tight_layout()


