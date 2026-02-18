# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Feb 17 15:34:08 2026

# @author: xpsy1114
# """

# import numpy as np

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import mc
import pickle
import sys
from datetime import date
import json
from fnmatch import fnmatch


# --- Load configuration ---
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
    print("Running on laptop.")
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
    print(f"Running on Cluster, setting {source_dir} as data directory")

config_file = sys.argv[2] if len(sys.argv) > 2 else "rsa_config_DSR_bias-path-rew-splitfuts_combos.json"
with open(f"{config_path}/{config_file}", "r") as f:
    config = json.load(f)

# SETTINGS
EV_string = config.get("load_EVs_from")
regression_version = config.get("regression_version")
split_rew_from_path = config.get("split_rew_from_path", False)

today_str = date.today().strftime("%d-%m-%Y")
name_RSA = config.get("name_of_RSA")
RDM_version = f"{name_RSA}"


# Subjects
if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '05'  
subjects = [f"sub-{subj_no}"]

# Flags
smoothing = config.get("smoothing", True)
fwhm = config.get("fwhm", 5)
load_searchlights = config.get("load_searchlights", False)
masked_conditions = config.get("masked_conds", None)
conditions_masking = None
include_diagonal = config.get("diagonal_included", True)

# conditions selection
conditions = config.get("EV_condition_selection")
parts_to_use = conditions["parts"]

folder = f'{source_dir}/data/derivatives/group/RDM_plots'
file = 'vox_45_88_39_data_RDM_DSR_rew-vs-path_stepwise_combos_glmbase_all-paths-fixed_stickrews_split-buttons_allsubs.npy'
subj_RDMs = np.load(f"{folder}/{file}")


def pair_correct_tasks(data_dict, keys_list):
    """
    data_dict: dict with keys like 'A1_forw_A_reward'
    keys_list: ordered list of keys you want to include and in what order
    Returns two matrices: one for the first element of each pair, one for its match.
    """
    # Define task pairing relationships
    task_pairs = {'1_forw': '2_backw', '1_backw': '2_forw'}
    th_1, th_2, paired_list_control  = [], [], []
    # Loop through keys in the *specified order*
    for key in keys_list:
        assert key in data_dict, "Missmatch between model rdm keys and data RDM keys"
        task, direction, state, phase = key.split('_')  # e.g. ['A1', 'forw', 'A', 'reward']
        # Create the pairing suffix (e.g. from '1_forw' → '2_backw')
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        # Build the paired key (e.g. 'A2_backw_A_reward')
        pair_key = f"{task[0]}{pair_suffix}_{state}_{phase}"
        # Only add if both keys exist
        if pair_key in data_dict:
            th_1.append(np.asarray(data_dict[key]))
            th_2.append(np.asarray(data_dict[pair_key]))
            paired_list_control.append(f"{key} with {pair_key}")

    # import pdb; pdb.set_trace()       
    th_1 = np.vstack(th_1)
    th_2 = np.vstack(th_2)
    # print(paired_list_control)
    return th_1, th_2, paired_list_control

for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
        # DONT FORGET TO COMMENT THIS OUT!!!!
        # regression_version = '03-4'
        only_load_labels = True
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        only_load_labels = False
        print(f"Running on Cluster, setting {data_dir} as data directory")
      
    modelled_conditions_dir = f"{data_dir}/beh/modelled_EVs"
    data_rdm_dir = f"{data_dir}/func/data_RDMs_glmbase_{regression_version}"

    results_dir = f"{data_dir}/func/RSA_{RDM_version}_{today_str}_glmbase_{regression_version}/results" 
    if smoothing == True:
       results_dir = f"{data_dir}/func/RSA_{RDM_version}_{today_str}_glmbase_{regression_version}_smooth{fwhm}/results" 
    os.makedirs(results_dir, exist_ok=True)

    # loading conditions for model and data RDMs
    #
    # loading the model EVs into dict
    with open(f"{modelled_conditions_dir}/{sub}_modelled_EVs_{EV_string}.pkl", 'rb') as file:
        model_EVs = pickle.load(file)
    selected_models = config.get("models", list(model_EVs.keys()))
    # loading the labels
    _, all_EV_keys = mc.analyse.my_RSA.load_data_EVs(data_dir, regression_version=regression_version, only_load_labels = True)
    
    # if you don't want all conditions created through FSL, exclude some here!
    # based on config file:
    # Ensure all four parts exist in config
    for _p in ("task", "direction", "state", "phase"):
        if _p not in parts_to_use:
            raise ValueError(f"Missing selection.parts['{_p}'] in config.")
            
    EV_keys = []        
    for ev in sorted(all_EV_keys):
        task, direction, state, phase = ev.split('_')
        # simple include/exclude logic
        for name, value in zip(["task", "direction", "state", "phase"], [task, direction, state, phase]):
            part = parts_to_use[name]
            includes = part.get("include", [])
            excludes = part.get("exclude", [])
            # Exclude first
            if any(fnmatch(value, pat) for pat in excludes):
                break  
            # If include list non-empty → must match at least one
            if includes and not any(fnmatch(value, pat) for pat in includes):
                break
        else:
            # only append if none of the 4 parts triggered 'break'
            EV_keys.append(ev)
    
    # 
    # Step 3: compute the model and data RDMs.
    models_concat = {}
    model_RDM_dir = {}
    
    for model in model_EVs:
        model_th1, model_th2, model_paired_labels = pair_correct_tasks(model_EVs[model], EV_keys)
        # finally, concatenate th1 and th2 to do the cross-correlation after
        models_concat[model] = np.concatenate((model_th1, model_th2), axis = 0)
        
    if masked_conditions:
        conditions_masking = mc.analyse.my_RSA.make_category_masks(models_concat['path_rew'], plotting = False, include_diagonal=include_diagonal, mask_only_path_rew_combos=True)
        pair_mask = conditions_masking['mask_reward-path']
            
   
# -------------------------------------------------------
# Pearson correlation (no scipy needed)
# -------------------------------------------------------
def pearsonr_np(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return np.nan if denom == 0 else float((x * y).sum() / denom)


def infer_n_from_vec_length(vec_len, include_diagonal):
    if include_diagonal:
        # n(n+1)/2 = vec_len
        n = int((np.sqrt(1 + 8 * vec_len) - 1) / 2)
        if n * (n + 1) // 2 != vec_len:
            raise ValueError("Vector length is incompatible with include_diagonal=True.")
    else:
        # n(n-1)/2 = vec_len
        n = int((1 + np.sqrt(1 + 8 * vec_len)) / 2)
        if n * (n - 1) // 2 != vec_len:
            raise ValueError("Vector length is incompatible with include_diagonal=False.")
    return n


def vec_to_symmetric_matrix(vec, include_diagonal):
    vec = np.asarray(vec, dtype=float)
    n = infer_n_from_vec_length(len(vec), include_diagonal)
    mat = np.full((n, n), np.nan, dtype=float)
    k = 0 if include_diagonal else 1
    iu = np.triu_indices(n, k=k)
    mat[iu] = vec
    mat = np.where(np.isnan(mat), mat.T, mat)
    lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    return np.ma.array(mat, mask=lower_mask), iu


def apply_block_lines(ax, n_cond, block_size=8):
    for b in range(block_size, n_cond, block_size):
        ax.axhline(b - 0.5, color="white", lw=1.0)
        ax.axvline(b - 0.5, color="white", lw=1.0)


def plot_masked_model_rdms(
    D_state,
    D_sa,
    masks,
    include_diagonal,
    out_path,
    state_label="DSR",
    sa_label="state-action-glob",
):
    state_upper, iu = vec_to_symmetric_matrix(D_state, include_diagonal=include_diagonal)
    sa_upper, _ = vec_to_symmetric_matrix(D_sa, include_diagonal=include_diagonal)
    n_cond = state_upper.shape[0]

    finite_vals = np.concatenate(
        [
            np.asarray(D_state)[np.isfinite(D_state)],
            np.asarray(D_sa)[np.isfinite(D_sa)],
        ]
    )
    vmin = float(np.min(finite_vals)) if finite_vals.size else 0.0
    vmax = float(np.max(finite_vals)) if finite_vals.size else 2.0
    norm_rdm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_rdm = plt.get_cmap("RdBu").copy()
    cmap_rdm.set_bad("white")

    mask_order = ["highS_highA", "highS_lowA", "lowS_highA", "lowS_lowA"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))

    for col, mask_name in enumerate(mask_order):
        mask_vec = np.asarray(masks[mask_name], dtype=bool)
        idxs = np.where(mask_vec)[0]

        for row, (rdm_upper, model_name) in enumerate(
            [(state_upper, state_label), (sa_upper, sa_label)]
        ):
            ax = axes[row, col]
            ax.imshow(
                rdm_upper,
                cmap=cmap_rdm,
                norm=norm_rdm,
                interpolation="None",
                aspect="equal",
            )
            ax.set_title(f"{model_name} | {mask_name} (n={len(idxs)})", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            apply_block_lines(ax, n_cond=n_cond, block_size=8)

            for idx in idxs:
                i, j = iu[0][idx], iu[1][idx]
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#d7301f",
                        linewidth=0.8,
                    )
                )

    fig.suptitle(
        f"Model RDMs with 4 masks highlighted in red ({state_label} vs {sa_label})",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # fig.savefig(out_path, dpi=200)
    # plt.close(fig)


# -------------------------------------------------------
# Build RDM vector using your exact pipeline
# -------------------------------------------------------
def build_rdm_vector(data, k=1):
    data = np.asarray(data, dtype=object)
    N = data.shape[0]
    half = N // 2

    overlap = np.equal(data[:, None, :], data[None, :, :])
    sim = overlap.mean(axis=2)
    rdm_full = 1.0 - sim

    rdm_small = rdm_full[half:, :half]
    rdm = (rdm_small + rdm_small.T) / 2.0

    iu = np.triu_indices(half, k=k)
    return rdm[iu]


# -------------------------------------------------------
# Extract action-only RDM (ignore state prefix)
# -------------------------------------------------------
def build_action_only_rdm_vector(sa_data, k=1):
    sa_data = np.asarray(sa_data, dtype=object)

    # extract action part (after "-")
    actions_only = np.vectorize(lambda x: x.split('-')[1])(sa_data)

    return build_rdm_vector(actions_only, k=k)


# -------------------------------------------------------
# Main function: percentile-based 4-way split
# -------------------------------------------------------
def split_corr_4way_percentile(models_concat,
                               pair_mask,
                               key_state='DSR',
                               key_sa='state_action_glob',
                               percentile=75,
                               k=1):

    # --- Build RDM vectors ---
    D_state = build_rdm_vector(models_concat[key_state], k=k)
    D_sa    = build_rdm_vector(models_concat[key_sa],    k=k)

    # --- Build action-only similarity ---
    D_action_only = build_action_only_rdm_vector(models_concat[key_sa], k=k)

    # --- Convert distances to similarity ---
    S_state  = 1.0 - D_state
    S_action = 1.0 - D_action_only

    pair_mask = np.asarray(pair_mask, dtype=bool)

    if len(pair_mask) != len(S_state):
        raise ValueError("Mask length mismatch.")

    # Apply mask before thresholding
    S_state_masked  = S_state[pair_mask]
    S_action_masked = S_action[pair_mask]

    # --- Determine thresholds ---
    state_thresh  = np.percentile(S_state_masked, percentile)
    action_thresh = np.percentile(S_action_masked, percentile)

    # --- Define high/low similarity bins ---
    highS = S_state >= state_thresh
    lowS  = S_state <  state_thresh

    highA = S_action >= action_thresh
    lowA  = S_action <  action_thresh

    # Combine with mask
    masks = {
        "highS_highA": pair_mask & highS & highA,
        "highS_lowA":  pair_mask & highS & lowA,
        "lowS_highA":  pair_mask & lowS  & highA,
        "lowS_lowA":   pair_mask & lowS  & lowA,
    }

    results = {}

    for key, m in masks.items():
        if m.sum() > 5:
            r = pearsonr_np(D_state[m], D_sa[m])
        else:
            r = np.nan
        results[key] = {
            "r": r,
            "n_pairs": int(m.sum())
        }

    results["overall_masked"] = {
        "r": pearsonr_np(D_state[pair_mask], D_sa[pair_mask]),
        "n_pairs": int(pair_mask.sum())
    }

    results["thresholds"] = {
        "state_similarity_threshold": state_thresh,
        "action_similarity_threshold": action_thresh
    }
    return {
        "results": results,
        "masks": masks,
        "D_state": D_state,
        "D_sa": D_sa,
        "D_action_only": D_action_only,
    }


split_out = split_corr_4way_percentile(
    models_concat,
    pair_mask,
    key_state='DSR',
    key_sa='state_action_glob',
    percentile=90,
    k=1
)
print(split_out['results'])

plot_file = os.path.join(results_dir, f"{subjects[-1]}_DSR_vs_state-action-glob_4masks.png")
plot_masked_model_rdms(
    split_out["D_state"],
    split_out["D_sa"],
    split_out["masks"],
    include_diagonal=include_diagonal,
    out_path=plot_file,
    state_label="DSR",
    sa_label="state-action-glob",
)
print(f"Saved mask-overlaid RDM panel to: {plot_file}")



