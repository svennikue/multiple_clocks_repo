#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore RSA model EVs and model RDMs without running searchlights.

- Loads model EVs like fMRI_run_RSA_without_rsatoolbox_clean.py
- Lets you select EVs and models
- Computes model similarity matrices using mc.analyse.my_RSA helpers
- Plots model EVs for a single task and model RDMs for a single task
- Shows where two model RDMs covary strongest
"""

import argparse
import json
import os
import pickle
from fnmatch import fnmatch
from datetime import date

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import mc

# Optional: set this to True when running directly from Spyder and you want
# explicit scripted args instead of CLI args.
SPYDER_RUN_SETTINGS = {
    "enabled": False,
    "args": [
        # Example:
        # "--config", "condition_files/rsa_config_state-action-playaround.json",
        # "--subject", "02",
        # "--only-reward-evs",
    ],
}


def pair_correct_tasks(data_dict, keys_list):
    """
    data_dict: dict with keys like 'A1_forw_A_reward'
    keys_list: ordered list of keys you want to include and in what order
    Returns two matrices: one for the first element of each pair, one for its match.
    """
    task_pairs = {'1_forw': '2_backw', '1_backw': '2_forw'}
    th_1, th_2, paired_list_control = [], [], []
    for key in keys_list:
        if key not in data_dict:
            raise ValueError(f"Mismatch between model/data EV keys and requested EV: {key}")
        task, direction, state, phase = key.split('_')
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        pair_key = f"{task[0]}{pair_suffix}_{state}_{phase}"
        if pair_key in data_dict:
            th_1.append(np.asarray(data_dict[key]))
            th_2.append(np.asarray(data_dict[pair_key]))
            paired_list_control.append(f"{key} with {pair_key}")
    if not th_1:
        raise ValueError("No paired EVs found. Check EV selection and pairing rules.")
    return np.vstack(th_1), np.vstack(th_2), paired_list_control


def build_paired_labels(keys_list, available_keys):
    task_pairs = {'1_forw': '2_backw', '1_backw': '2_forw'}
    paired_labels = []
    available_set = set(available_keys)
    for key in keys_list:
        task, direction, state, phase = key.split('_')
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        pair_key = f"{task[0]}{pair_suffix}_{state}_{phase}"
        if pair_key in available_set:
            paired_labels.append(f"{key} with {pair_key}")
    if not paired_labels:
        raise ValueError("No paired labels found. Check EV selection and pairing rules.")
    return paired_labels


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def resolve_source_dirs():
    source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
    if os.path.isdir(source_dir):
        config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
        data_root = f"{source_dir}/data/derivatives"
        print("Running on laptop.")
    else:
        source_dir = "/home/fs0/xpsy1114/scratch"
        config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"
        data_root = f"{source_dir}/data/derivatives"
        print("Running on cluster.")
    return config_path, data_root


def filter_ev_keys(all_ev_keys, parts_to_use, include_patterns=None, exclude_patterns=None, explicit_list=None):
    if explicit_list:
        missing = [k for k in explicit_list if k not in all_ev_keys]
        if missing:
            raise ValueError(f"Explicit EVs not found in loaded EVs: {missing}")
        return explicit_list

    ev_keys = []
    for ev in sorted(all_ev_keys):
        task, direction, state, phase = ev.split('_')
        for name, value in zip(["task", "direction", "state", "phase"], [task, direction, state, phase]):
            part = parts_to_use[name]
            includes = part.get("include", [])
            excludes = part.get("exclude", [])
            if any(fnmatch(value, pat) for pat in excludes):
                break
            if includes and not any(fnmatch(value, pat) for pat in includes):
                break
        else:
            ev_keys.append(ev)

    if include_patterns:
        ev_keys = [k for k in ev_keys if any(fnmatch(k, pat) for pat in include_patterns)]
    if exclude_patterns:
        ev_keys = [k for k in ev_keys if not any(fnmatch(k, pat) for pat in exclude_patterns)]

    return ev_keys


def filter_reward_evs(ev_keys):
    return [k for k in ev_keys if k.endswith("reward")]


def infer_n_from_labels(labels):
    return len(labels)


def vec_to_rdm(vec, n, include_diagonal=True):
    rdm = np.full((n, n), np.nan, dtype=float)
    k = 0 if include_diagonal else 1
    tri = np.triu_indices(n, k=k)
    rdm[tri] = vec
    rdm[(tri[1], tri[0])] = rdm[tri]
    if not include_diagonal:
        np.fill_diagonal(rdm, 0.0)
    return rdm


def zscore_vec(vec):
    mean = np.nanmean(vec)
    std = np.nanstd(vec)
    if std == 0 or np.isnan(std):
        return vec * 0.0
    return (vec - mean) / std


def build_model_rdms(model_evs, ev_keys, include_diagonal=True):
    models_concat = {}
    model_rdms = {}
    for model in model_evs:
        th1, th2, _ = pair_correct_tasks(model_evs[model], ev_keys)
        models_concat[model] = np.concatenate((th1, th2), axis=0)

    for model in model_evs:
        if model == 'path_rew':
            model_rdms[model] = mc.analyse.my_RSA.make_categorical_RDM(models_concat[model], plotting=False, include_diagonal=include_diagonal)
        elif model == 'duration':
            model_rdms[model] = mc.analyse.my_RSA.make_distance_RDM(models_concat[model], plotting=False, include_diagonal=include_diagonal)
        elif model in ['location', 'DSR', 'prev_buttons', 'buttons_out', 'next_buttons', 'state_action_glob', 'state_action_loc']:
            model_rdms[model] = mc.analyse.my_RSA.compute_hamming_distance(models_concat[model], plotting=False, include_diagonal=include_diagonal)
        elif model.endswith('diff'):
            cond = model.split('-')[1]
            model_rdms[model] = mc.analyse.my_RSA.compute_hamming_difference(models_concat[model], combination=cond, plotting=False, include_diagonal=include_diagonal)
        elif model.startswith('button'):
            model_rdms[model] = mc.analyse.my_RSA.make_distance_RDM_cosine_normratio(models_concat[model], plotting=False, include_diagonal=include_diagonal)
        else:
            model_rdms[model] = mc.analyse.my_RSA.compute_crosscorr(models_concat[model], plotting=False, include_diagonal=include_diagonal)
    return model_rdms


def prepare_selected_model_evs(selected_models, model_evs):
    """
    Build a model dict that supports diff-model aliases even when only the base
    EV key exists in model_evs (e.g. state_action_glob-da_ss_diff -> state_action_glob).
    """
    prepared = {}
    missing = []
    for model in selected_models:
        if model in model_evs:
            prepared[model] = model_evs[model]
            continue
        if model.endswith('diff'):
            base_model = model.split('-')[0]
            if base_model in model_evs:
                prepared[model] = model_evs[base_model]
                continue
        missing.append(model)
    return prepared, missing


def select_ev_by_task(ev_keys, task_pattern):
    return [k for k in ev_keys if fnmatch(k, task_pattern)]


def plot_ev_matrix(ev_dict, ev_keys, title, save_path=None, show=True):
    mat = np.vstack([np.asarray(ev_dict[k]).squeeze() for k in ev_keys])
    if mat.ndim == 1:
        mat = mat[:, None]
    is_numeric = np.issubdtype(mat.dtype, np.number)

    plt.figure(figsize=(max(6, mat.shape[1] / 6), max(4, mat.shape[0] / 3)))
    if is_numeric and mat.shape[1] == 1:
        plt.bar(np.arange(mat.shape[0]), mat[:, 0])
        plt.xticks(np.arange(mat.shape[0]), ev_keys, rotation=90, fontsize=6)
        plt.ylabel('EV value')
    elif not is_numeric:
        # Categorical EVs (e.g. state-action labels): encode to integer IDs for plotting.
        flat = mat.astype(str).ravel()
        uniq = sorted(np.unique(flat).tolist())
        enc_map = {u: i for i, u in enumerate(uniq)}
        mat_enc = np.vectorize(enc_map.get)(mat.astype(str))
        plt.imshow(mat_enc, aspect='auto', cmap='tab20')
        cbar = plt.colorbar(label='Category')
        cbar.set_ticks(np.arange(len(uniq)))
        cbar.set_ticklabels(uniq)
        plt.yticks(np.arange(mat.shape[0]), ev_keys, fontsize=6)
        plt.xlabel('Feature index')
    else:
        plt.imshow(mat, aspect='auto', cmap='viridis')
        plt.colorbar(label='EV value')
        plt.yticks(np.arange(mat.shape[0]), ev_keys, fontsize=6)
        plt.xlabel('Feature index')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()


def plot_rdm(rdm_vec, labels, include_diagonal=True, title=None, save_path=None, show=True):
    n = infer_n_from_labels(labels)
    rdm = vec_to_rdm(rdm_vec, n, include_diagonal=include_diagonal)
    plt.figure(figsize=(6, 6))
    plt.imshow(rdm, aspect='auto', cmap='coolwarm', vmin=0, vmax=2)
    plt.xticks(np.arange(n), labels, rotation=90, fontsize=6)
    plt.yticks(np.arange(n), labels, fontsize=6)
    plt.title(title or 'RDM')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()


def top_covarying_pairs(model_vec_a, model_vec_b, labels, include_diagonal=True, topk=10):
    n = infer_n_from_labels(labels)
    k = 0 if include_diagonal else 1
    tri = np.triu_indices(n, k=k)

    z_a = zscore_vec(model_vec_a)
    z_b = zscore_vec(model_vec_b)
    covary = z_a * z_b
    mask = np.isfinite(covary)

    covary_valid = covary[mask]
    tri_i = tri[0][mask]
    tri_j = tri[1][mask]

    if covary_valid.size == 0:
        return []

    idx = np.argsort(covary_valid)[::-1]
    idx = idx[:topk]

    results = []
    for i in idx:
        ii = tri_i[i]
        jj = tri_j[i]
        results.append({
            'pair': (labels[ii], labels[jj]),
            'covary': float(covary_valid[i])
        })
    return results


def plot_covary_matrix(model_vec_a, model_vec_b, labels, include_diagonal=True, title=None, save_path=None, show=True):
    n = infer_n_from_labels(labels)
    z_a = zscore_vec(model_vec_a)
    z_b = zscore_vec(model_vec_b)
    covary = z_a * z_b
    covary_mat = vec_to_rdm(covary, n, include_diagonal=include_diagonal)

    plt.figure(figsize=(6, 6))
    plt.imshow(covary_mat, aspect='auto', cmap='coolwarm')
    plt.xticks(np.arange(n), labels, rotation=90, fontsize=6)
    plt.yticks(np.arange(n), labels, fontsize=6)
    plt.title(title or 'Model covariation (zA * zB)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()


def model_rdm_correlation(model_vec_a, model_vec_b):
    mask = np.isfinite(model_vec_a) & np.isfinite(model_vec_b)
    if mask.sum() == 0:
        return np.nan
    return float(np.corrcoef(model_vec_a[mask], model_vec_b[mask])[0, 1])


def load_and_average_data_rdm(path):
    arr = np.load(path)
    if arr.ndim == 1:
        return arr.astype(float), arr.shape
    if arr.ndim == 2:
        # Typical group format: (n_subs, n_pairs)
        return np.nanmean(arr.astype(float), axis=0), arr.shape
    raise ValueError(f"Unsupported data RDM array shape: {arr.shape}")


def top_contributing_points(data_vec, model_vec, topk=100):
    valid = np.isfinite(data_vec) & np.isfinite(model_vec)
    if valid.sum() == 0:
        raise ValueError("No finite overlapping entries between data and model RDM vectors.")

    z_data = np.full_like(data_vec, np.nan, dtype=float)
    z_model = np.full_like(model_vec, np.nan, dtype=float)
    z_data[valid] = zscore_vec(data_vec[valid])
    z_model[valid] = zscore_vec(model_vec[valid])

    contributions = z_data * z_model
    valid_idx = np.where(valid)[0]
    order_local = np.argsort(contributions[valid])[::-1]
    topk = min(topk, order_local.size)
    top_idx = valid_idx[order_local[:topk]]
    return contributions, top_idx, valid


def contribution_summary_table(top_idx, contributions, labels, include_diagonal=True):
    n = infer_n_from_labels(labels)
    tri = np.triu_indices(n, k=0 if include_diagonal else 1)
    rows = []
    for rank, vec_idx in enumerate(top_idx, start=1):
        i = int(tri[0][vec_idx])
        j = int(tri[1][vec_idx])
        rows.append({
            "rank": rank,
            "vec_index": int(vec_idx),
            "row": i,
            "col": j,
            "label_row": labels[i],
            "label_col": labels[j],
            "contribution": float(contributions[vec_idx]),
        })
    return rows


def plot_rdm_with_highlights(rdm_vec, labels, highlight_idx, include_diagonal, title, ax):
    n = infer_n_from_labels(labels)
    mat = vec_to_rdm(rdm_vec, n, include_diagonal=include_diagonal)
    img = ax.imshow(mat, aspect='auto', cmap='coolwarm')
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_title(title)

    tri = np.triu_indices(n, k=0 if include_diagonal else 1)
    for vec_idx in highlight_idx:
        i = int(tri[0][vec_idx])
        j = int(tri[1][vec_idx])
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=1.0))
        if i != j:
            ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, edgecolor='red', linewidth=1.0))
    return img


def extract_tasks_from_label(label):
    left, right = label.split(" with ")
    def parse_side(side):
        side = side.replace("_reward", "")
        task_with_dir, _state = side.rsplit("_", 1)
        task = task_with_dir.split("_", 1)[0]
        return task
    return parse_side(left), parse_side(right)


def task_contribution_analysis(model_vec_a, model_vec_b, labels, include_diagonal=True):
    n = infer_n_from_labels(labels)
    k = 0 if include_diagonal else 1
    tri = np.triu_indices(n, k=k)
    z_a = zscore_vec(model_vec_a)
    z_b = zscore_vec(model_vec_b)
    covary = z_a * z_b
    mask = np.isfinite(covary)

    covary_valid = covary[mask]
    tri_i = tri[0][mask]
    tri_j = tri[1][mask]

    task_scores = {}
    for idx, val in enumerate(covary_valid):
        i = tri_i[idx]
        j = tri_j[idx]
        task_i, task_j = extract_tasks_from_label(labels[i])
        # each pair contributes to both tasks
        task_scores.setdefault(task_i, []).append(val)
        task_scores.setdefault(task_j, []).append(val)

    task_mean = {t: float(np.nanmean(v)) for t, v in task_scores.items()}
    ranked = sorted(task_mean.items(), key=lambda x: x[1], reverse=True)
    return ranked


def main(cli_args=None):
    parser = argparse.ArgumentParser(description="Explore RSA EVs and similarity matrices.")
    parser.add_argument("--config", default=None, help="Path to RSA config JSON.")
    parser.add_argument("--subject", default="02", help="Subject number (e.g., 02).")
    parser.add_argument("--models", default=None, help="Comma-separated model names to include.")
    parser.add_argument("--ev-include", action="append", default=[], help="Glob pattern(s) to include EVs.")
    parser.add_argument("--ev-exclude", action="append", default=[], help="Glob pattern(s) to exclude EVs.")
    parser.add_argument("--ev-list", default=None, help="Comma-separated explicit EV keys to use.")
    parser.add_argument("--only-reward-evs", action="store_true", help="Only include EVs whose name ends with 'reward'.")
    parser.add_argument("--task-pattern", default=None, help="Glob for plotting EVs (e.g., 'A1_forw_*').")
    parser.add_argument("--plot-evs-model", default=None, help="Model name whose EVs to plot.")
    parser.add_argument("--plot-task-rdm", default=None, help="Glob for plotting RDM for a single task.")
    parser.add_argument("--plot-rdm-model", default=None, help="Model name to plot RDM for.")
    parser.add_argument("--plot-full-rdm", action="append", default=[], help="Model name(s) to plot full RDM for.")
    parser.add_argument("--compare-models", nargs=2, default=None, help="Two model names to compare covariation.")
    parser.add_argument("--topk", type=int, default=10, help="Top-k covarying pairs to list.")
    parser.add_argument("--data-rdm-path", default=None, help="Path to data RDM .npy (1D vector or 2D subject x vector).")
    parser.add_argument("--diagnostic-model", default="state_action_glob-da_ss_diff", help="Model name for data-vs-model diagnostic.")
    parser.add_argument("--diagnostic-topk", type=int, default=100, help="Top-k datapoints to highlight in data-vs-model diagnostic.")
    parser.add_argument("--save-dir", default=None, help="Directory to save plots.")
    parser.add_argument("--no-show", action="store_true", help="Do not call plt.show().")

    # parse_known_args keeps this script runnable in Spyder/IPython where extra
    # kernel args can be injected automatically.
    args, _ = parser.parse_known_args(cli_args)

    config_path_base, data_root = resolve_source_dirs()
    config_path = args.config or os.path.join(config_path_base, "rsa_config_DSR_bias-path-rew-splitfuts_combos.json")
    config = load_config(config_path)

    ev_string = config.get("load_EVs_from")
    regression_version = config.get("regression_version")
    include_diagonal = config.get("diagonal_included", True)

    sub = f"sub-{args.subject}"
    data_dir = os.path.join(data_root, sub)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    modelled_conditions_dir = os.path.join(data_dir, "beh", "modelled_EVs")

    with open(os.path.join(modelled_conditions_dir, f"{sub}_modelled_EVs_{ev_string}.pkl"), "rb") as f:
        model_evs = pickle.load(f)

    selected_models = config.get("models", list(model_evs.keys()))
    if args.models:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.data_rdm_path and args.diagnostic_model not in selected_models:
        selected_models.append(args.diagnostic_model)

    _, all_ev_keys = mc.analyse.my_RSA.load_data_EVs(
        data_dir,
        regression_version=regression_version,
        only_load_labels=True,
    )

    parts_to_use = config.get("EV_condition_selection", {}).get("parts", {})
    for _p in ("task", "direction", "state", "phase"):
        if _p not in parts_to_use:
            raise ValueError(f"Missing selection.parts['{_p}'] in config.")

    explicit_list = None
    if args.ev_list:
        explicit_list = [k.strip() for k in args.ev_list.split(",") if k.strip()]

    ev_keys = filter_ev_keys(
        all_ev_keys,
        parts_to_use,
        include_patterns=args.ev_include or None,
        exclude_patterns=args.ev_exclude or None,
        explicit_list=explicit_list,
    )
    if args.only_reward_evs:
        ev_keys = filter_reward_evs(ev_keys)
        if not ev_keys:
            raise ValueError("No EVs left after applying reward-only filter.")
        print("Applied reward-only EV filter (suffix: 'reward').")
    print(f"Including {len(ev_keys)} EVs.")

    selected_model_evs, missing_models = prepare_selected_model_evs(selected_models, model_evs)
    if missing_models:
        print(f"Skipping unavailable model keys: {missing_models}")
    if not selected_model_evs:
        raise ValueError("No valid models available after resolving selected models.")

    # Build model RDMs (full selection)
    model_rdms = build_model_rdms(selected_model_evs, ev_keys, include_diagonal=include_diagonal)

    # Labels for RDM axes
    paired_labels = build_paired_labels(ev_keys, all_ev_keys)

    show_plots = not args.no_show
    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Plot EVs for a single task
    if args.task_pattern and args.plot_evs_model:
        task_keys = select_ev_by_task(ev_keys, args.task_pattern)
        if not task_keys:
            raise ValueError(f"No EVs match task pattern: {args.task_pattern}")
        model_name = args.plot_evs_model
        if model_name not in model_evs:
            raise ValueError(f"Model not found: {model_name}")
        ev_title = f"EVs for {model_name} ({args.task_pattern})"
        save_path = os.path.join(save_dir, f"EVs_{model_name}_{args.task_pattern}.png") if save_dir else None
        plot_ev_matrix(model_evs[model_name], task_keys, ev_title, save_path=save_path, show=show_plots)

    # If no actions specified, default to comparing l2_norm vs prev_buttons
    if not any([args.task_pattern, args.plot_evs_model, args.plot_task_rdm, args.plot_rdm_model, args.plot_full_rdm, args.compare_models, args.data_rdm_path]):
        # args.compare_models = ["l2_norm", "prev_buttons"]
        # args.plot_full_rdm = ["l2_norm", "prev_buttons"]
        args.compare_models = ["l2_norm", "buttons_out"]
        args.plot_full_rdm = ["l2_norm", "buttons_out"]

    # Plot RDM for a single task
    if args.plot_task_rdm and args.plot_rdm_model:
        task_keys = select_ev_by_task(ev_keys, args.plot_task_rdm)
        if not task_keys:
            raise ValueError(f"No EVs match task pattern: {args.plot_task_rdm}")
        task_labels = build_paired_labels(task_keys, all_ev_keys)
        model_name = args.plot_rdm_model
        if model_name not in model_evs:
            raise ValueError(f"Model not found: {model_name}")
        model_rdms_task = build_model_rdms({model_name: model_evs[model_name]}, task_keys, include_diagonal=include_diagonal)
        rdm_vec = model_rdms_task[model_name][0]
        title = f"Model RDM {model_name} ({args.plot_task_rdm})"
        save_path = os.path.join(save_dir, f"RDM_{model_name}_{args.plot_task_rdm}.png") if save_dir else None
        plot_rdm(rdm_vec, task_labels, include_diagonal=include_diagonal, title=title, save_path=save_path, show=show_plots)

    # Plot full RDMs
    if args.plot_full_rdm:
        for model_name in args.plot_full_rdm:
            if model_name not in model_rdms:
                raise ValueError(f"Model not found: {model_name}")
            rdm_vec = model_rdms[model_name][0]
            title = f"Model RDM {model_name} (full)"
            save_path = os.path.join(save_dir, f"RDM_{model_name}_full.png") if save_dir else None
            plot_rdm(rdm_vec, paired_labels, include_diagonal=include_diagonal, title=title, save_path=save_path, show=show_plots)

    # Compare models: covariation
    
    if args.compare_models:
        m1, m2 = args.compare_models
        if m1 not in model_rdms or m2 not in model_rdms:
            raise ValueError(f"Model(s) not found in computed RDMs: {m1}, {m2}")

        vec_a = model_rdms[m1][0]
        vec_b = model_rdms[m2][0]

        r = model_rdm_correlation(vec_a, vec_b)
        print(f"Correlation between full model RDMs ({m1} vs {m2}): r={r:.3f}")

        cov_title = f"Covariation: {m1} vs {m2}"
        save_path = os.path.join(save_dir, f"Covary_{m1}_vs_{m2}.png") if save_dir else None
        plot_covary_matrix(vec_a, vec_b, paired_labels, include_diagonal=include_diagonal, title=cov_title, save_path=save_path, show=show_plots)

        top_pairs = top_covarying_pairs(vec_a, vec_b, paired_labels, include_diagonal=include_diagonal, topk=args.topk)
        print(f"Top {len(top_pairs)} covarying pairs (zA * zB):")
        for rank, item in enumerate(top_pairs, start=1):
            a, b = item['pair']
            print(f"{rank:>2}. {a}  <->  {b}   covary={item['covary']:.3f}")

        # Task contribution analysis
        ranked_tasks = task_contribution_analysis(vec_a, vec_b, paired_labels, include_diagonal=include_diagonal)
        print("\n=== TASK CONTRIBUTION (mean covariation) ===")
        for task, mean_val in ranked_tasks:
            print(f"{task:>10}: mean covary={mean_val:.4f}")

        if ranked_tasks:
            top_task = ranked_tasks[0][0]
            print(f"\nTop task by mean covariation: {top_task}")
            task_pattern = f"{top_task}_*"
            task_keys = select_ev_by_task(ev_keys, task_pattern)
            if not task_keys:
                print(f"No EVs matched task pattern for top task: {task_pattern}")
            else:
                task_labels = build_paired_labels(task_keys, all_ev_keys)
                m1_task = build_model_rdms({m1: model_evs[m1]}, task_keys, include_diagonal=include_diagonal)[m1][0]
                m2_task = build_model_rdms({m2: model_evs[m2]}, task_keys, include_diagonal=include_diagonal)[m2][0]
                plot_rdm(m1_task, task_labels, include_diagonal=include_diagonal, title=f"RDM {m1} ({top_task})", show=show_plots)
                plot_rdm(m2_task, task_labels, include_diagonal=include_diagonal, title=f"RDM {m2} ({top_task})", show=show_plots)

                # Plot pure conditions (inputs) for this task
                print(f"\n=== INPUT EVs for {top_task} ===")
                for model_name in (m1, m2):
                    if model_name in model_evs:
                        ev_title = f"EVs {model_name} ({top_task})"
                        plot_ev_matrix(model_evs[model_name], task_keys, ev_title, show=show_plots)
                            
    if args.data_rdm_path:
        model_name = args.diagnostic_model
        if model_name not in model_rdms:
            if model_name not in model_evs:
                raise ValueError(f"Diagnostic model '{model_name}' not found in loaded model EVs.")
            extra_rdms = build_model_rdms({model_name: model_evs[model_name]}, ev_keys, include_diagonal=include_diagonal)
            model_rdms[model_name] = extra_rdms[model_name]

        data_vec, data_shape = load_and_average_data_rdm(args.data_rdm_path)
        model_vec = np.asarray(model_rdms[model_name][0], dtype=float)
        if data_vec.shape[0] != model_vec.shape[0]:
            raise ValueError(
                f"Length mismatch between averaged data RDM ({data_vec.shape[0]}) "
                f"and model RDM ({model_vec.shape[0]})."
            )

        contributions, top_idx, valid = top_contributing_points(data_vec, model_vec, topk=args.diagnostic_topk)
        corr = model_rdm_correlation(data_vec, model_vec)
        print(f"\nData-vs-model diagnostic for '{model_name}'")
        print(f"Input data RDM shape: {data_shape} -> averaged vector length {data_vec.shape[0]}")
        print(f"Finite overlapping pairs: {int(valid.sum())}")
        print(f"Pearson r(data, model): {corr:.4f}")
        print(f"Top highlighted datapoints: {len(top_idx)}")

        rows = contribution_summary_table(top_idx, contributions, paired_labels, include_diagonal=include_diagonal)
        print("\nTop contributing datapoints:")
        for row in rows[:10]:
            print(
                f"{row['rank']:>3}. ({row['row']:>2}, {row['col']:>2}) "
                f"{row['label_row']} <-> {row['label_col']} "
                f"contrib={row['contribution']:.4f}"
            )

        fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
        img0 = plot_rdm_with_highlights(
            data_vec,
            paired_labels,
            top_idx,
            include_diagonal=include_diagonal,
            title="Average Data RDM (top contributions in red)",
            ax=axes[0],
        )
        img1 = plot_rdm_with_highlights(
            model_vec,
            paired_labels,
            top_idx,
            include_diagonal=include_diagonal,
            title=f"Model RDM: {model_name} (top contributions in red)",
            ax=axes[1],
        )
        fig.colorbar(img0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(img1, ax=axes[1], fraction=0.046, pad=0.04)

        if save_dir:
            diagnostic_png = os.path.join(save_dir, f"diagnostic_data_vs_{model_name}_top{len(top_idx)}.png")
            fig.savefig(diagnostic_png, dpi=300, bbox_inches="tight")
            print(f"Saved diagnostic figure: {diagnostic_png}")

            out_tsv = os.path.join(save_dir, f"diagnostic_data_vs_{model_name}_top{len(top_idx)}.tsv")
            with open(out_tsv, "w") as f:
                f.write("rank\tvec_index\trow\tcol\tlabel_row\tlabel_col\tcontribution\n")
                for row in rows:
                    f.write(
                        f"{row['rank']}\t{row['vec_index']}\t{row['row']}\t{row['col']}\t"
                        f"{row['label_row']}\t{row['label_col']}\t{row['contribution']}\n"
                    )
            print(f"Saved top datapoints table: {out_tsv}")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    # Quick summary
    print("\n=== SETTINGS SUMMARY ===")
    print(f"subject: {sub}")
    print(f"config: {config_path}")
    print(f"EV string: {ev_string}")
    print(f"regression version: {regression_version}")
    print(f"n EVs selected: {len(ev_keys)}")
    print(f"models evaluated: {selected_models}")
    print(f"diagonal included: {include_diagonal}")
    print(f"today: {date.today().strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    if SPYDER_RUN_SETTINGS["enabled"]:
        main(cli_args=SPYDER_RUN_SETTINGS["args"])
    else:
        main()
