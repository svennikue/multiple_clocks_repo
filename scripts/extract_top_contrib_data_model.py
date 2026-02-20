#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import pickle
from fnmatch import fnmatch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def resolve_source_dirs():
    source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
    if os.path.isdir(source_dir):
        config_dir = f"{source_dir}/multiple_clocks_repo/condition_files"
        data_root = f"{source_dir}/data/derivatives"
    else:
        source_dir = "/home/fs0/xpsy1114/scratch"
        config_dir = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
        data_root = f"{source_dir}/data/derivatives"
    return config_dir, data_root


def pair_correct_tasks(data_dict, keys_list):
    task_pairs = {"1_forw": "2_backw", "1_backw": "2_forw"}
    th_1, th_2, paired = [], [], []
    for key in keys_list:
        if key not in data_dict:
            raise ValueError(f"Key not in model EVs: {key}")
        task, direction, state, phase = key.split("_")
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        pair_key = f"{task[0]}{pair_suffix}_{state}_{phase}"
        if pair_key in data_dict:
            th_1.append(np.asarray(data_dict[key]))
            th_2.append(np.asarray(data_dict[pair_key]))
            paired.append(f"{key} with {pair_key}")
    if not th_1:
        raise ValueError("No paired EVs found.")
    return np.vstack(th_1), np.vstack(th_2), paired


def build_paired_labels(keys_list, available_keys):
    task_pairs = {"1_forw": "2_backw", "1_backw": "2_forw"}
    paired = []
    available = set(available_keys)
    for key in keys_list:
        task, direction, state, phase = key.split("_")
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        pair_key = f"{task[0]}{pair_suffix}_{state}_{phase}"
        if pair_key in available:
            paired.append(f"{key} with {pair_key}")
    if not paired:
        raise ValueError("No paired labels found.")
    return paired


def load_data_ev_labels(data_dir, regression_version):
    labels = []
    for th in (1, 2):
        ev_txt = f"{data_dir}/func/EVs_{regression_version}_pt0{th}/task-to-EV.txt"
        with open(ev_txt, "r") as f:
            for line in f:
                _, name_ev = line.strip().split(" ", 1)
                name = name_ev.replace("ev_", "")
                if name not in ["press_EV", "up", "down", "left", "right"]:
                    labels.append(name)
    return labels


def filter_ev_keys(all_ev_keys, parts_to_use):
    ev_keys = []
    for ev in sorted(all_ev_keys):
        task, direction, state, phase = ev.split("_")
        keep = True
        for name, value in zip(["task", "direction", "state", "phase"], [task, direction, state, phase]):
            part = parts_to_use[name]
            includes = part.get("include", [])
            excludes = part.get("exclude", [])
            if any(fnmatch(value, pat) for pat in excludes):
                keep = False
                break
            if includes and not any(fnmatch(value, pat) for pat in includes):
                keep = False
                break
        if keep:
            ev_keys.append(ev)
    return ev_keys


def compute_hamming_difference(data_chunk, combination, include_diagonal=False):
    data = np.asarray(data_chunk, dtype=object)

    states = np.char.partition(data.astype(str), "-")[..., 0]
    actions = np.char.partition(data.astype(str), "-")[..., 2]

    state_sim = states[:, None, :] == states[None, :, :]
    state_dissim = states[:, None, :] != states[None, :, :]
    action_sim = actions[:, None, :] == actions[None, :, :]
    action_dissim = actions[:, None, :] != actions[None, :, :]

    if combination.startswith("sa_ss"):
        overlap = action_sim & state_sim
    elif combination.startswith("sa_ds"):
        overlap = action_sim & state_dissim
    elif combination.startswith("da_ss"):
        overlap = action_dissim & state_sim
    elif combination.startswith("da_ds"):
        overlap = action_dissim & state_dissim
    else:
        raise ValueError(f"Unknown hamming-difference combination: {combination}")

    hamming_sim = overlap.mean(axis=2)
    rdm_both_halves = 1 - hamming_sim
    rdm_small = rdm_both_halves[int(len(rdm_both_halves) / 2):, 0:int(len(rdm_both_halves) / 2)]
    rdm = (rdm_small + rdm_small.T) / 2
    k = 0 if include_diagonal else 1
    return rdm[np.triu_indices(rdm.shape[0], k=k)]


def zscore(v):
    mu = np.nanmean(v)
    sd = np.nanstd(v)
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(v)
    return (v - mu) / sd


def vec_to_rdm(vec, n, include_diagonal=False):
    mat = np.full((n, n), np.nan, dtype=float)
    k = 0 if include_diagonal else 1
    tri = np.triu_indices(n, k=k)
    mat[tri] = vec
    mat[(tri[1], tri[0])] = mat[tri]
    if not include_diagonal:
        np.fill_diagonal(mat, 0.0)
    return mat


def main():
    parser = argparse.ArgumentParser(description="Top correlation-contributing datapoints between data and model RDMs.")
    parser.add_argument(
        "--data-rdm-path",
        default="/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/group/RDM_plots/vox_45_88_39_data_RDM_DSR_rew-vs-path_stepwise_combos_glmbase_all-paths-fixed_stickrews_split-buttons_allsubs.npy",
    )
    parser.add_argument("--config", default="rsa_config_state-action-playaround.json")
    parser.add_argument("--subject", default="02")
    parser.add_argument("--model", default="state_action_glob-da_ss_diff")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--save-dir", default="scripts/figures/2026-02-20")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    config_dir, data_root = resolve_source_dirs()
    config_path = args.config if os.path.isabs(args.config) else os.path.join(config_dir, args.config)
    with open(config_path, "r") as f:
        config = json.load(f)

    regression_version = config["regression_version"]
    ev_string = config["load_EVs_from"]
    include_diagonal = bool(config.get("diagonal_included", False))
    if include_diagonal:
        raise ValueError("This script expects diagonal_included=False for vector length matching.")

    sub = f"sub-{args.subject}"
    data_dir = os.path.join(data_root, sub)
    model_pkl = os.path.join(data_dir, "beh", "modelled_EVs", f"{sub}_modelled_EVs_{ev_string}.pkl")
    with open(model_pkl, "rb") as f:
        model_evs = pickle.load(f)

    all_ev_keys = load_data_ev_labels(data_dir, regression_version)
    parts_to_use = config["EV_condition_selection"]["parts"]
    ev_keys = filter_ev_keys(all_ev_keys, parts_to_use)
    paired_labels = build_paired_labels(ev_keys, all_ev_keys)

    if args.model.endswith("diff"):
        base_model = args.model.split("-")[0]
        diff_combo = args.model.split("-")[1]
    else:
        raise ValueError("This script currently supports '*-<combo>_diff' models.")

    th1, th2, _ = pair_correct_tasks(model_evs[base_model], ev_keys)
    model_concat = np.concatenate((th1, th2), axis=0)
    model_vec = compute_hamming_difference(model_concat, combination=diff_combo, include_diagonal=False)

    data_arr = np.load(args.data_rdm_path)
    if data_arr.ndim == 2:
        data_vec = np.nanmean(data_arr.astype(float), axis=0)
    elif data_arr.ndim == 1:
        data_vec = data_arr.astype(float)
    else:
        raise ValueError(f"Unsupported data RDM shape: {data_arr.shape}")

    if data_vec.shape[0] != model_vec.shape[0]:
        raise ValueError(f"Length mismatch: data={data_vec.shape[0]} model={model_vec.shape[0]}")

    valid = np.isfinite(data_vec) & np.isfinite(model_vec)
    z_data = np.full_like(data_vec, np.nan, dtype=float)
    z_model = np.full_like(model_vec, np.nan, dtype=float)
    z_data[valid] = zscore(data_vec[valid])
    z_model[valid] = zscore(model_vec[valid])
    contrib = z_data * z_model

    valid_idx = np.where(valid)[0]
    ord_local = np.argsort(contrib[valid])[::-1]
    topk = min(args.topk, ord_local.size)
    top_idx = valid_idx[ord_local[:topk]]

    corr = np.corrcoef(data_vec[valid], model_vec[valid])[0, 1]
    print(f"Pearson r(data, {args.model}) = {corr:.4f}")
    print(f"Using top {len(top_idx)} datapoints by contribution z(data)*z(model).")

    n = len(paired_labels)
    tri = np.triu_indices(n, k=1)
    rows = []
    for rank, vec_idx in enumerate(top_idx, start=1):
        i = int(tri[0][vec_idx])
        j = int(tri[1][vec_idx])
        rows.append((rank, int(vec_idx), i, j, paired_labels[i], paired_labels[j], float(contrib[vec_idx])))
    print("Top 10:")
    for r in rows[:10]:
        print(f"{r[0]:>3}. ({r[2]:>2},{r[3]:>2}) {r[4]} <-> {r[5]} contrib={r[6]:.4f}")

    data_mat = vec_to_rdm(data_vec, n, include_diagonal=False)
    model_mat = vec_to_rdm(model_vec, n, include_diagonal=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    im0 = axes[0].imshow(data_mat, aspect="auto", cmap="coolwarm")
    axes[0].set_title("Average Data RDM (top contributions in red)")
    axes[1].imshow(model_mat, aspect="auto", cmap="coolwarm")
    axes[1].set_title(f"Model RDM: {args.model} (top contributions in red)")

    for ax in axes:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(paired_labels, rotation=90, fontsize=5)
        ax.set_yticklabels(paired_labels, fontsize=5)
        for vec_idx in top_idx:
            i = int(tri[0][vec_idx])
            j = int(tri[1][vec_idx])
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.0))
            ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.0))
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    os.makedirs(args.save_dir, exist_ok=True)
    fig_path = os.path.join(args.save_dir, f"data_vs_{args.model}_top{len(top_idx)}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")

    csv_path = os.path.join(args.save_dir, f"data_vs_{args.model}_top{len(top_idx)}.csv")
    with open(csv_path, "w") as f:
        f.write("rank,vec_index,row,col,label_row,label_col,contribution\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},\"{r[4]}\",\"{r[5]}\",{r[6]}\n")
    print(f"Saved table: {csv_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()

