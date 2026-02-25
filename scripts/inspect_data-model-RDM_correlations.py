#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect which parts of the RDM drive correlations by comparing subject data RDMs
against model RDMs (e.g., DSR) and plotting the RDM vectors as lines.
"""

import argparse
import csv
import collections
import textwrap
import json
import os
import pickle
from fnmatch import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

import mc


# =========================
# User-Adjustable Settings
# =========================
# Edit values here for Spyder-style runs. CLI flags can still override these.
SETTINGS = {
    # Data/config/model defaults
    "data_npy": (
        "data/derivatives/group/RDM_plots/"
        "vox_45_88_39_data_RDM_DSR_rew-vs-path_stepwise_combos_glmbase_"
        "all-paths-fixed_stickrews_split-buttons_allsubs.npy"
    ),
    "config": "rsa_config_state-action-playaround.json",
    "model": "state_action_glob-da_ss_diff",
    # "model": "DSR",
    "models": "",  # comma-separated model names, "" -> use config['models']
    "subjects": "",  # comma-separated subject IDs, "" -> subject range below
    "plot_all_models": False,
    "out_dir": "",
    # Subject range used when subjects == ""
    "default_subject_start": 2,
    "default_subject_stop": 3,  # inclusive
    "exclude_subjects_if_full_sample": ["sub-21", "sub-29"],
    # Subject order in the group data RDM file (row mapping)
    "data_rdm_subject_start": 1,
    "data_rdm_subject_stop": 35,
    "data_rdm_subject_exclude": [21, 29],
    # Default analysis mode: run two masked views per subject.
    # 1) Exclude reward-path pairs
    # 2) Reward-only (path conditions fully excluded)
    "run_default_dual_mask_views": True,
    # Optional single-mask fallback (used only when run_default_dual_mask_views=False)
    "use_conditions_masking": False,
    "conditions_mask_name": "mask_reward-path", # "path-path",  # e.g. path-path, reward-reward, reward-path, mask_reward-path
    "mask_only_path_rew_combos": False,
    # Paths
    "local_source_dir": "/Users/xpsy1114/Documents/projects/multiple_clocks",
    "cluster_source_dir": "/home/fs0/xpsy1114/scratch",
    "cluster_config_subdir": "analysis/multiple_clocks_repo/condition_files",
    "local_config_subdir": "multiple_clocks_repo/condition_files",
    "behaviour_file_name": "{sub}_beh_fmri_clean.csv",
    "top_n_diagnostic": 200,
    "top_n_print": 50,
}

HAMMING_DISTANCE_MODELS = {
    "location",
    "DSR",
    "prev_buttons",
    "buttons_out",
    "next_buttons",
    "state_action_glob",
    "state_action_loc",
    "rewDSR",
    "pathDSR",
    "rew_stateactionDSR",
    "path_stateactionDSR",
}
HAMMING_DIFF_COMBINATIONS = (
    "sa_ss_diff",
    "sa_ds_diff",
    "da_ss_diff",
    "da_ds_diff",
)
REVERSE_MODEL_BASES = (
    "state_action_glob",
    "rew_stateactionDSR",
    "path_stateactionDSR",
)

# Set to True when running from Spyder/Jupyter and use SETTINGS values above.
USE_SPYDER_DEFAULTS = False
SPYDER_DEFAULTS = {
    "data_npy": SETTINGS["data_npy"],
    "config": SETTINGS["config"],
    "model": SETTINGS["model"],
    "models": SETTINGS["models"],
    "subjects": SETTINGS["subjects"],
    "plot_all_models": SETTINGS["plot_all_models"],
    "out_dir": SETTINGS["out_dir"],
}


def pair_correct_tasks(data_dict, keys_list):
    """
    data_dict: dict with keys like 'A1_forw_A_reward'
    keys_list: ordered list of keys you want to include and in what order
    Returns two matrices: one for the first element of each pair, one for its match.
    """
    task_pairs = {"1_forw": "2_backw", "1_backw": "2_forw"}
    th_1, th_2, paired_list_control = [], [], []
    for key in keys_list:
        assert key in data_dict, "Missmatch between model rdm keys and data RDM keys"
        task, direction, state, phase = key.split("_")
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        pair_key = f"{task[0]}{pair_suffix}_{state}_{phase}"
        if pair_key in data_dict:
            th_1.append(np.asarray(data_dict[key]))
            th_2.append(np.asarray(data_dict[pair_key]))
            paired_list_control.append(f"{key} | {pair_key}")

    th_1 = np.vstack(th_1)
    th_2 = np.vstack(th_2)
    return th_1, th_2, paired_list_control


def build_ev_keys(all_EV_keys, parts_to_use):
    for _p in ("task", "direction", "state", "phase"):
        if _p not in parts_to_use:
            raise ValueError(f"Missing selection.parts['{_p}'] in config.")

    EV_keys = []
    for ev in sorted(all_EV_keys):
        task, direction, state, phase = ev.split("_")
        for name, value in zip(
            ["task", "direction", "state", "phase"],
            [task, direction, state, phase],
        ):
            part = parts_to_use[name]
            includes = part.get("include", [])
            excludes = part.get("exclude", [])
            if any(fnmatch(value, pat) for pat in excludes):
                break
            if includes and not any(fnmatch(value, pat) for pat in includes):
                break
        else:
            EV_keys.append(ev)

    return EV_keys


def build_point_labels(paired_labels, include_diagonal):
    n = len(paired_labels)
    k = 0 if include_diagonal else 1
    iu = np.triu_indices(n, k=k)
    point_labels = [
        f"{paired_labels[i]} vs {paired_labels[j]}" for i, j in zip(iu[0], iu[1])
    ]
    return point_labels


def build_paired_labels_from_keys(keys_list):
    task_pairs = {"1_forw": "2_backw", "1_backw": "2_forw"}
    keys_set = set(keys_list)
    paired_labels = []
    for key in keys_list:
        task, direction, state, phase = key.split("_")
        pair_suffix = task_pairs.get(f"{task[-1]}_{direction}")
        pair_key = f"{task[0]}{pair_suffix}_{state}_{phase}"
        if pair_key in keys_set:
            paired_labels.append(f"{key} | {pair_key}")
    return paired_labels


def load_unique_time_bin_types_from_behaviour(data_dir, sub):
    beh_file = SETTINGS["behaviour_file_name"].format(sub=sub)
    beh_path = os.path.join(data_dir, "beh", beh_file)
    if not os.path.exists(beh_path):
        raise FileNotFoundError(f"Behaviour file not found: {beh_path}")

    unique_bins = set()
    with open(beh_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "unique_time_bin_type" not in (reader.fieldnames or []):
            raise ValueError(
                f"'unique_time_bin_type' column missing in behaviour file: {beh_path}"
            )
        for row in reader:
            value = row.get("unique_time_bin_type", "")
            if value:
                unique_bins.add(value)
    return sorted(unique_bins)


def build_pair_index_map(paired_labels, include_diagonal):
    n = len(paired_labels)
    k = 0 if include_diagonal else 1
    iu = np.triu_indices(n, k=k)
    pair_to_idx = {}
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        pair_to_idx[(paired_labels[i], paired_labels[j])] = idx
    return pair_to_idx


def subset_vector_indices_from_paired_labels(
    full_paired_labels, subset_paired_labels, include_diagonal
):
    full_map = build_pair_index_map(full_paired_labels, include_diagonal)
    n_sub = len(subset_paired_labels)
    k = 0 if include_diagonal else 1
    iu_sub = np.triu_indices(n_sub, k=k)
    indices = []
    for i, j in zip(iu_sub[0], iu_sub[1]):
        pair_key = (subset_paired_labels[i], subset_paired_labels[j])
        if pair_key not in full_map:
            raise ValueError(
                f"Could not map subset pair to full vector: {pair_key}"
            )
        indices.append(full_map[pair_key])
    return np.asarray(indices, dtype=int)


def parse_paired_label(lbl):
    if " | " in lbl:
        left, right = lbl.split(" | ")
    else:
        left, right = lbl.split(" with ")
    l = left.split("_")
    r = right.split("_")
    arrow = {"backw": "<-", "forw": "->"}
    block = f"{l[0]}{arrow.get(l[1], '')}|{r[0]}{arrow.get(r[1], '')}"
    within = f"{l[2]}-{l[3]}".replace("reward", "rew")
    return block, within


def parse_label_events(label):
    parts = label.split(" vs ")
    events = []
    for side in parts:
        if " | " in side:
            halves = side.split(" | ")
        else:
            halves = side.split(" with ")
        for ev in halves:
            tokens = ev.split("_")
            if len(tokens) == 4:
                events.append(
                    {
                        "task": tokens[0],
                        "direction": tokens[1],
                        "state": tokens[2],
                        "phase": tokens[3],
                    }
                )
    return events


def summarize_top_labels(top_labels):
    phase_counts = collections.Counter()
    task_counts = collections.Counter()
    task_combo_counts = collections.Counter()
    direction_counts = collections.Counter()
    combo_phase_counts = collections.Counter()
    label_direction_presence = collections.Counter()

    for label in top_labels:
        if not label:
            continue
        norm_label = normalize_label_for_analysis(label)
        events = parse_label_events(norm_label)
        for ev in events:
            phase_counts[ev["phase"]] += 1
            task_counts[ev["task"][0]] += 1
            direction_counts[ev["direction"]] += 1

        # phase combo across the two sides (left vs right)
        sides = norm_label.split(" vs ")
        if len(sides) == 2:
            left_events = parse_label_events(sides[0])
            right_events = parse_label_events(sides[1])
            if left_events and right_events:
                left_phase = left_events[0]["phase"]
                right_phase = right_events[0]["phase"]
                combo = "-".join(sorted([left_phase, right_phase]))
                combo_phase_counts[combo] += 1
                left_task = left_events[0]["task"][0]
                right_task = right_events[0]["task"][0]
                combo_task = "-".join(sorted([left_task, right_task]))
                task_combo_counts[combo_task] += 1

        # whether any backw/forw appears in label
        if any(ev["direction"] == "backw" for ev in events):
            label_direction_presence["labels_with_backw"] += 1
        if any(ev["direction"] == "forw" for ev in events):
            label_direction_presence["labels_with_forw"] += 1

    return {
        "phase_counts": phase_counts,
        "task_counts": task_counts,
        "task_combo_counts": task_combo_counts,
        "direction_counts": direction_counts,
        "phase_combo_counts": combo_phase_counts,
        "label_direction_presence": label_direction_presence,
    }


def sort_counter(counter_obj):
    return collections.OrderedDict(
        sorted(counter_obj.items(), key=lambda kv: kv[1], reverse=True)
    )


def normalize_label_for_analysis(label):
    return label


def select_top_positive_contrib(
    contrib,
    min_abs=1e-6,
    min_cum=0.30,
    fallback_cum=0.80,
    min_top_n=10,
    smooth_window=25,
):
    pos_idx = np.where(np.isfinite(contrib) & (contrib > 0))[0]
    if pos_idx.size == 0:
        return np.array([], dtype=int)

    pos_vals = contrib[pos_idx]
    order = np.argsort(pos_vals)[::-1]
    vals = pos_vals[order]

    if smooth_window > 1 and vals.size >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        vals_s = np.convolve(vals, kernel, mode="same")
    else:
        vals_s = vals

    x = np.arange(vals_s.size)
    y = vals_s
    if y.size > 1 and y.max() != y.min():
        x0, y0 = x[0], y[0]
        x1, y1 = x[-1], y[-1]
        denom = np.hypot(x1 - x0, y1 - y0)
        if denom == 0:
            d = np.zeros_like(y)
        else:
            d = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
        knee_idx = int(np.argmax(d))
    else:
        knee_idx = min(vals.size - 1, min_top_n - 1)

    if vals[knee_idx] < min_abs:
        knee_idx = min_top_n - 1

    cum = np.cumsum(vals)
    total = cum[-1]
    knee_cum = cum[knee_idx] / total if total > 0 else 0

    if knee_cum < min_cum:
        cutoff = fallback_cum * total
        knee_idx = int(np.searchsorted(cum, cutoff, side="left"))

    knee_idx = max(knee_idx, min_top_n - 1)
    knee_idx = min(knee_idx, vals.size - 1, 200 - 1)

    return pos_idx[order[: knee_idx + 1]]


def compute_contributions(model_vec, data_vec):
    mask = np.isfinite(model_vec) & np.isfinite(data_vec)
    if mask.sum() == 0:
        return None, mask
    m = model_vec[mask]
    d = data_vec[mask]
    m_std = np.std(m)
    d_std = np.std(d)
    if m_std == 0 or d_std == 0:
        return None, mask
    m_z = (m - np.mean(m)) / m_std
    d_z = (d - np.mean(d)) / d_std
    contrib = np.full_like(model_vec, np.nan, dtype=float)
    contrib[mask] = m_z * d_z
    return contrib, mask


def split_diff_model_name(model_name):
    for combo in HAMMING_DIFF_COMBINATIONS:
        dash_suffix = f"-{combo}"
        under_suffix = f"_{combo}"
        if model_name.endswith(dash_suffix):
            return model_name[: -len(dash_suffix)], combo
        if model_name.endswith(under_suffix):
            return model_name[: -len(under_suffix)], combo
    return None, None


def expand_reverse_models(model_evs, models_reverse):
    if not models_reverse:
        return
    for base_model in REVERSE_MODEL_BASES:
        if base_model not in model_evs:
            continue
        for rev in models_reverse:
            model_evs[f"{base_model}-{rev}"] = model_evs[base_model].copy()


def prepare_selected_model_evs(selected_models, model_evs):
    prepared = {}
    missing = []
    for model in selected_models:
        if model in model_evs:
            prepared[model] = model_evs[model]
            continue
        base_model, diff_combo = split_diff_model_name(model)
        if diff_combo and base_model in model_evs:
            prepared[model] = model_evs[base_model]
            continue
        missing.append(model)
    return prepared, missing


def compute_model_rdm(model_name, model_concat, include_diagonal):
    if model_name == "path_rew":
        return mc.analyse.my_RSA.make_categorical_RDM(
            model_concat, plotting=False, include_diagonal=include_diagonal
        )
    if model_name == "duration":
        return mc.analyse.my_RSA.make_distance_RDM(
            model_concat, plotting=False, include_diagonal=include_diagonal
        )
    if model_name in HAMMING_DISTANCE_MODELS:
        return mc.analyse.my_RSA.compute_hamming_distance(
            model_concat,
            plotting=False,
            include_diagonal=include_diagonal,
            model_name=model_name,
        )

    _, diff_combo = split_diff_model_name(model_name)
    if diff_combo:
        return mc.analyse.my_RSA.compute_hamming_difference(
            model_concat,
            combination=diff_combo,
            plotting=False,
            include_diagonal=include_diagonal,
            model_name=model_name,
        )

    return mc.analyse.my_RSA.compute_crosscorr(
        model_concat, plotting=False, include_diagonal=include_diagonal
    )


def build_data_rdm_subject_order():
    excluded = {int(s) for s in SETTINGS["data_rdm_subject_exclude"]}
    subjects = []
    for i in range(
        SETTINGS["data_rdm_subject_start"], SETTINGS["data_rdm_subject_stop"] + 1
    ):
        if i in excluded:
            continue
        subjects.append(f"sub-{i:02}")
    return subjects


def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Correlate subject data RDMs with model RDMs and plot vectors."
    )
    parser.add_argument(
        "--data-npy",
        # default=(
        #     "data/derivatives/group/RDM_plots/"
        #     "vox_33_19_48_data_RDM_DSR_rew-vs-path_stepwise_combos_glmbase_"
        #     "all-paths-fixed_stickrews_split-buttons.npy"
        # ),
        # /avg_RDM_vox_65_50_64_data_RDM_DSR_rew_stepwise_combos_23-01-2026_glmbase_all-paths-fixed_stickrews_split-buttons.npy
        # default=(
        #     "data/derivatives/group/RDM_plots/"
        #     "vox_65_50_64_data_RDM_DSR_rew_stepwise_combos_23-01-2026_glmbase_"
        #     "all-paths-fixed_stickrews_split-buttons.npy"
        # ),
        default=SETTINGS["data_npy"],
    )
    # "vox_33_81_25_data_RDM_DSR_rew-vs-path_stepwise_combos_glmbase_"
        
    parser.add_argument(
        "--config",
        #default="rsa_config_DSR_rew_vs_path_stepwise_combos.json",
        default=SETTINGS["config"],
        help="RSA config file (in condition_files).",
    )
    parser.add_argument(
        "--model",
        default=SETTINGS["model"],
        help="Model RDM of interest for correlation.",
    )
    parser.add_argument(
        "--models",
        default=SETTINGS["models"],
        help=(
            "Optional comma-separated models to build. "
            "If empty, uses config['models']; --model is always added."
        ),
    )
    parser.add_argument(
        "--subjects",
        default=SETTINGS["subjects"],
        help="Comma-separated subject IDs (e.g., sub-01,sub-02).",
    )
    parser.add_argument(
        "--plot-all-models",
        action="store_true",
        help="Plot all model RDMs (not just the model of interest).",
    )
    parser.add_argument(
        "--out-dir",
        default=SETTINGS["out_dir"],
        help="Output directory for plots (defaults next to data-npy).",
    )
    return parser.parse_args(arg_list)


def get_runtime_args():
    if USE_SPYDER_DEFAULTS:
        return argparse.Namespace(**SPYDER_DEFAULTS)
    return parse_args()


def main():
    args = get_runtime_args()

    source_dir = SETTINGS["local_source_dir"]
    if os.path.isdir(source_dir):
        config_path = os.path.join(source_dir, SETTINGS["local_config_subdir"])
    else:
        source_dir = SETTINGS["cluster_source_dir"]
        config_path = os.path.join(source_dir, SETTINGS["cluster_config_subdir"])

    data_npy_path = args.data_npy
    if not os.path.isabs(data_npy_path):
        data_npy_path = os.path.join(source_dir, data_npy_path)

    with open(os.path.join(config_path, args.config), "r") as f:
        config = json.load(f)

    EV_string = config.get("load_EVs_from")
    regression_version = config.get("regression_version")
    include_diagonal = config.get("diagonal_included", True)
    conditions = config.get("EV_condition_selection")
    parts_to_use = conditions["parts"]

    data_rdms = np.load(data_npy_path, allow_pickle=True)
    if data_rdms.ndim == 1:
        data_rdms = data_rdms[None, :]

    base_name = os.path.basename(data_npy_path)
    voxel_tag = base_name.split("_data_RDM_")[0]

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = [
            f"sub-{i:02}"
            for i in range(
                SETTINGS["default_subject_start"], SETTINGS["default_subject_stop"] + 1
            )
        ]
        if len(subjects) > 30:
            for excluded_sub in SETTINGS["exclude_subjects_if_full_sample"]:
                if excluded_sub in subjects:
                    subjects.remove(excluded_sub)
        # subjects = ["sub-01", "sub-02", "sub-03"]

    if data_rdms.shape[0] == 0:
        raise ValueError("No data rows found in the data RDM file.")

    data_rdm_subject_order = build_data_rdm_subject_order()
    if data_rdms.shape[0] != len(data_rdm_subject_order):
        raise ValueError(
            "Data RDM row count does not match configured subject order: "
            f"rows={data_rdms.shape[0]}, expected={len(data_rdm_subject_order)} "
            "(from SETTINGS data_rdm_subject_start/stop/exclude)."
        )
    subject_to_data_row = {s: i for i, s in enumerate(data_rdm_subject_order)}
    missing_subjects = [s for s in subjects if s not in subject_to_data_row]
    if missing_subjects:
        raise ValueError(
            "Requested subjects are not present in configured data-RDM subject order: "
            + ", ".join(missing_subjects)
        )
    subj_indices = [subject_to_data_row[s] for s in subjects]
    #import pdb; pdb.set_trace()
    data_rdms = np.stack([data_rdms[i] for i in subj_indices], axis=0)

    out_dir = args.out_dir
    if not out_dir:
        out_dir = os.path.dirname(data_npy_path)
    os.makedirs(out_dir, exist_ok=True)

    # Build labels once from behavioural file (no data-EV folder access)
    first_data_dir = os.path.join(source_dir, "data/derivatives", subjects[0])
    all_EV_keys = load_unique_time_bin_types_from_behaviour(first_data_dir, subjects[0])
    EV_keys = build_ev_keys(all_EV_keys, parts_to_use)
    paired_labels = build_paired_labels_from_keys(EV_keys)
    if not paired_labels:
        raise ValueError(
            "No paired labels found from behavioural unique_time_bin_type keys."
        )
    point_labels = build_point_labels(paired_labels, include_diagonal)

    summary_rows = []
    vector_rows = []

    for sub, data_vec in zip(subjects, data_rdms):
        data_dir = os.path.join(source_dir, "data/derivatives", sub)
        modelled_conditions_dir = os.path.join(data_dir, "beh/modelled_EVs")

        with open(
            os.path.join(
                modelled_conditions_dir, f"{sub}_modelled_EVs_{EV_string}.pkl"
            ),
            "rb",
        ) as file:
            model_EVs = pickle.load(file)

        expand_reverse_models(model_EVs, config.get("models_reverse", None))

        if args.models:
            selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
        else:
            selected_models = config.get("models", list(model_EVs.keys()))
        if args.model not in selected_models:
            selected_models.append(args.model)

        selected_model_evs, missing_models = prepare_selected_model_evs(
            selected_models, model_EVs
        )
        if missing_models:
            print(
                f"{sub}: skipping unavailable models: {', '.join(missing_models)}"
            )

        models_concat = {}
        model_RDM_dir = {}
        for model, model_ev in selected_model_evs.items():
            model_th1, model_th2, _ = pair_correct_tasks(model_ev, EV_keys)
            models_concat[model] = np.concatenate((model_th1, model_th2), axis=0)
            model_RDM_dir[model] = compute_model_rdm(
                model, models_concat[model], include_diagonal
            )

        if args.model not in model_RDM_dir:
            print(f"{sub}: model '{args.model}' not found; skipping correlation.")
            continue

        model_th1_interest, model_th2_interest, model_paired_labels_interest = (
            pair_correct_tasks(selected_model_evs[args.model], EV_keys)
        )
        model_input_vec_by_label = {}
        for lbl, v1, v2 in zip(
            model_paired_labels_interest, model_th1_interest, model_th2_interest
        ):
            model_input_vec_by_label[lbl] = np.concatenate(
                (np.ravel(np.asarray(v1)), np.ravel(np.asarray(v2)))
            )

        model_vec_full = np.asarray(model_RDM_dir[args.model][0])
        data_vec_full = np.asarray(data_vec)
        if model_vec_full.shape[0] != data_vec_full.shape[0]:
            print(
                f"{sub}: length mismatch model={model_vec_full.shape[0]} "
                f"data={data_vec_full.shape[0]}"
            )
            continue

        analysis_cases = []
        if SETTINGS["run_default_dual_mask_views"]:
            if "path_rew" in models_concat:
                path_rew_concat = models_concat["path_rew"]
            elif "path_rew" in model_EVs:
                path_th1, path_th2, _ = pair_correct_tasks(model_EVs["path_rew"], EV_keys)
                path_rew_concat = np.concatenate((path_th1, path_th2), axis=0)
            else:
                raise ValueError(
                    f"{sub}: run_default_dual_mask_views=True requires model EV 'path_rew'."
                )
            conditions_masking = mc.analyse.my_RSA.make_category_masks(
                path_rew_concat,
                plotting=False,
                include_diagonal=include_diagonal,
                mask_only_path_rew_combos=True,
            )
            if "mask_reward-path" not in conditions_masking:
                raise ValueError(
                    f"{sub}: expected 'mask_reward-path' in masks. "
                    f"Available: {list(conditions_masking.keys())}"
                )
            analysis_cases.append(
                {
                    "mask_tag": "masked-no_reward-path",
                    "vector_mask": np.asarray(
                        conditions_masking["mask_reward-path"], dtype=bool
                    ),
                    "compact_plot": False,
                }
            )
            reward_ev_keys = [k for k in EV_keys if k.split("_")[-1] == "reward"]
            reward_paired_labels = build_paired_labels_from_keys(reward_ev_keys)
            reward_indices = subset_vector_indices_from_paired_labels(
                paired_labels, reward_paired_labels, include_diagonal
            )
            analysis_cases.append(
                {
                    "mask_tag": "reward-only",
                    "selected_indices": reward_indices,
                    "compact_plot": True,
                    "paired_labels_plot": reward_paired_labels,
                }
            )
        elif SETTINGS["use_conditions_masking"]:
            if "path_rew" in models_concat:
                path_rew_concat = models_concat["path_rew"]
            elif "path_rew" in model_EVs:
                path_th1, path_th2, _ = pair_correct_tasks(model_EVs["path_rew"], EV_keys)
                path_rew_concat = np.concatenate((path_th1, path_th2), axis=0)
            else:
                raise ValueError(
                    f"{sub}: settings.use_conditions_masking=True requires model EV 'path_rew'."
                )
            conditions_masking = mc.analyse.my_RSA.make_category_masks(
                path_rew_concat,
                plotting=False,
                include_diagonal=include_diagonal,
                mask_only_path_rew_combos=SETTINGS["mask_only_path_rew_combos"],
            )
            cond_name = SETTINGS["conditions_mask_name"]
            if cond_name not in conditions_masking:
                raise ValueError(
                    f"{sub}: unknown conditions mask '{cond_name}'. "
                    f"Available: {list(conditions_masking.keys())}"
                )
            analysis_cases.append(
                {
                    "mask_tag": f"masked-{cond_name}",
                    "vector_mask": np.asarray(conditions_masking[cond_name], dtype=bool),
                    "compact_plot": False,
                }
            )
        else:
            analysis_cases.append(
                {
                    "mask_tag": "unmasked",
                    "vector_mask": np.ones(model_vec_full.shape[0], dtype=bool),
                    "compact_plot": False,
                }
            )

        for case in analysis_cases:
            mask_tag = case["mask_tag"]
            compact_plot = bool(case.get("compact_plot", False))
            if "selected_indices" in case:
                selected_indices = np.asarray(case["selected_indices"], dtype=int)
                vector_mask = np.zeros(model_vec_full.shape[0], dtype=bool)
                vector_mask[selected_indices] = True
            else:
                vector_mask = np.asarray(case["vector_mask"], dtype=bool)
                if vector_mask.shape[0] != model_vec_full.shape[0]:
                    raise ValueError(
                        f"{sub}: mask length mismatch mask={vector_mask.shape[0]} "
                        f"model={model_vec_full.shape[0]} ({mask_tag})"
                    )
                selected_indices = np.where(vector_mask)[0]
            model_vec = model_vec_full[selected_indices]
            data_vec = data_vec_full[selected_indices]
            if compact_plot:
                paired_labels_case = case["paired_labels_plot"]
                point_labels_selected = build_point_labels(
                    paired_labels_case, include_diagonal
                )
            else:
                paired_labels_case = paired_labels
                point_labels_selected = [
                    point_labels[i] if i < len(point_labels) else "" for i in selected_indices
                ]

            mask = np.isfinite(model_vec) & np.isfinite(data_vec)
            if mask.sum() == 0:
                print(f"{sub}: no finite values to correlate ({mask_tag}).")
                continue
            corr = np.corrcoef(model_vec[mask], data_vec[mask])[0, 1]
            print(
                f"{sub}: {args.model} vs data r = {corr:.4f} "
                f"(n={mask.sum()}, {mask_tag}, kept={vector_mask.sum()}/{len(vector_mask)})"
            )
            # import pdb; pdb.set_trace()
            contrib, _ = compute_contributions(model_vec, data_vec)
            if contrib is None:
                print(f"{sub}: cannot compute contributions (zero variance, {mask_tag}).")
                continue

            top_print_n = int(SETTINGS["top_n_print"])
            top_store_n = int(SETTINGS["top_n_diagnostic"])
            top_pos_idx = select_top_positive_contrib(
                contrib,
                min_abs=1e-6,
                min_cum=0.30,
                fallback_cum=0.80,
                min_top_n=10,
                smooth_window=25,
            )
            if top_pos_idx.size == 0:
                print(f"{sub}: no positive contributions to select ({mask_tag}).")
                continue

            top_store_idx = top_pos_idx[: min(top_store_n, len(top_pos_idx))]

            for rank, idx in enumerate(top_store_idx, start=1):
                label = point_labels_selected[idx] if idx < len(point_labels_selected) else ""
                summary_rows.append(
                    [
                        sub,
                        mask_tag,
                        rank,
                        idx,
                        int(selected_indices[idx]),
                        label,
                        float(model_vec[idx]),
                        float(data_vec[idx]),
                        float(contrib[idx]),
                    ]
                )

            top_category_counter = collections.Counter()
            for rank, idx in enumerate(top_store_idx, start=1):
                label = point_labels_selected[idx] if idx < len(point_labels_selected) else ""
                if " vs " not in label:
                    continue
                left_cond, right_cond = label.split(" vs ")
                for side, cond_label in (("left", left_cond), ("right", right_cond)):
                    input_vec = model_input_vec_by_label.get(cond_label, None)
                    if input_vec is None:
                        continue
                    input_vec = np.asarray(input_vec)
                    for entry in np.ravel(input_vec):
                        top_category_counter[str(entry)] += 1
                    vector_rows.append(
                        [
                            sub,
                            mask_tag,
                            rank,
                            int(idx),
                            int(selected_indices[idx]),
                            cond_label,
                            side,
                            json.dumps(input_vec.tolist()),
                        ]
                    )

            overall_category_counter = collections.Counter()
            for idx in range(len(point_labels_selected)):
                label = point_labels_selected[idx]
                if " vs " not in label:
                    continue
                left_cond, right_cond = label.split(" vs ")
                for cond_label in (left_cond, right_cond):
                    input_vec = model_input_vec_by_label.get(cond_label, None)
                    if input_vec is None:
                        continue
                    input_vec = np.asarray(input_vec)
                    for entry in np.ravel(input_vec):
                        overall_category_counter[str(entry)] += 1

            top_labels = [
                point_labels_selected[i] if i < len(point_labels_selected) else ""
                for i in top_pos_idx
            ]
            summary = summarize_top_labels(top_labels)
            analysis_str = (
                f"phase={dict(sort_counter(summary['phase_counts']))} | "
                f"task={dict(sort_counter(summary['task_counts']))} | "
                f"task_combo={dict(sort_counter(summary['task_combo_counts']))} | "
                f"direction={dict(sort_counter(summary['direction_counts']))} | "
                f"phase_combo={dict(sort_counter(summary['phase_combo_counts']))} | "
                f"label_dir={dict(sort_counter(summary['label_direction_presence']))}"
            )

            n = len(paired_labels_case)
            k = 0 if include_diagonal else 1
            iu = np.triu_indices(n, k=k)

            if compact_plot:
                full_top_idx = top_pos_idx
                contrib_mat = np.full((n, n), np.nan, dtype=float)
                contrib_vec = np.zeros_like(model_vec, dtype=float)
                contrib_vec[top_pos_idx] = 1.0
                contrib_mat[iu] = contrib_vec
                contrib_mat = np.where(np.isnan(contrib_mat), contrib_mat.T, contrib_mat)

                model_mat = np.full((n, n), np.nan, dtype=float)
                model_mat[iu] = model_vec
                model_mat = np.where(np.isnan(model_mat), model_mat.T, model_mat)

                data_mat = np.full((n, n), np.nan, dtype=float)
                data_mat[iu] = data_vec
                data_mat = np.where(np.isnan(data_mat), data_mat.T, data_mat)
            else:
                full_top_idx = selected_indices[top_pos_idx]
                contrib_vec_full = np.full(model_vec_full.shape[0], np.nan, dtype=float)
                contrib_vec_full[vector_mask] = 0.0
                contrib_vec_full[full_top_idx] = 1.0

                model_vec_plot = np.full(model_vec_full.shape[0], np.nan, dtype=float)
                model_vec_plot[vector_mask] = model_vec_full[vector_mask]
                data_vec_plot = np.full(data_vec_full.shape[0], np.nan, dtype=float)
                data_vec_plot[vector_mask] = data_vec_full[vector_mask]

                n_full = len(paired_labels)
                iu_full = np.triu_indices(n_full, k=k)

                contrib_mat = np.full((n_full, n_full), np.nan, dtype=float)
                contrib_mat[iu_full] = contrib_vec_full
                contrib_mat = np.where(np.isnan(contrib_mat), contrib_mat.T, contrib_mat)

                model_mat = np.full((n_full, n_full), np.nan, dtype=float)
                model_mat[iu_full] = model_vec_plot
                model_mat = np.where(np.isnan(model_mat), model_mat.T, model_mat)

                data_mat = np.full((n_full, n_full), np.nan, dtype=float)
                data_mat[iu_full] = data_vec_plot
                data_mat = np.where(np.isnan(data_mat), data_mat.T, data_mat)
                n = n_full
                iu = iu_full
                paired_labels_case = paired_labels

            first_block_str = (
                paired_labels_case[0].split("_")[0]
                + "_"
                + paired_labels_case[0].split("_")[1]
            )
            block_size = None
            for i, l in enumerate(paired_labels_case):
                if first_block_str in l:
                    block_size = i + 1
            if block_size is None:
                block_size = n

            parsed = [parse_paired_label(l) for l in paired_labels_case]
            block_labels = [parsed[i][0] for i in range(0, n, block_size)]
            centers = np.arange(block_size / 2 - 0.5, n, block_size)
            within = [parsed[i][1] for i in range(0, block_size)]

            cmap_contrib = mpl.colors.ListedColormap(["#4575b4", "white", "#d7301f"])
            cmap_contrib.set_bad("white")
            norm_contrib = mpl.colors.BoundaryNorm(
                [-1.5, -0.5, 0.5, 1.5], cmap_contrib.N
            )

            def apply_block_labels(ax):
                for b in range(block_size, n, block_size):
                    ax.axhline(b - 0.5, color="white", lw=1.0)
                    ax.axvline(b - 0.5, color="white", lw=1.0)
                ax.set_xticks([])
                ax.set_yticks(centers)
                ax.set_yticklabels(block_labels, fontsize=8)
                ax.yaxis.tick_right()
                ax.tick_params(length=0, pad=1, labelsize=8)
                ax.set_xlabel(
                    "Within-block: " + " | ".join(within), fontsize=8, labelpad=8
                )

            cmap_rdm = plt.get_cmap("RdBu").copy()
            cmap_rdm.set_bad("white")
            model_vals = model_vec[np.isfinite(model_vec)]
            data_vals = data_vec[np.isfinite(data_vec)]
            if model_vals.size == 0 or data_vals.size == 0:
                rdm_vmin, rdm_vmax = 0.0, 2.0
            else:
                model_vmin, model_vmax = float(np.min(model_vals)), float(np.max(model_vals))
                data_vmin, data_vmax = float(np.min(data_vals)), float(np.max(data_vals))
                rdm_vmin, rdm_vmax = min(model_vmin, data_vmin), max(model_vmax, data_vmax)
            norm_rdm = mpl.colors.TwoSlopeNorm(vmin=rdm_vmin, vcenter=1.0, vmax=rdm_vmax)

            lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
            contrib_upper = np.ma.array(contrib_mat, mask=lower_mask)
            model_upper = np.ma.array(model_mat, mask=lower_mask)
            data_upper = np.ma.array(data_mat, mask=lower_mask)

            def draw_contrib_boxes(ax):
                for idx in full_top_idx:
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

            fig_c, axes_c = plt.subplots(1, 3, figsize=(18, 6))
            axes_c[0].imshow(
                contrib_upper,
                cmap=cmap_contrib,
                norm=norm_contrib,
                interpolation="None",
                aspect="equal",
            )
            axes_c[0].set_title(
                f"{sub} {voxel_tag}: \n{mask_tag} top {len(top_pos_idx)} pos contrib (r={corr:.3f})",
                fontsize=14,
            )
            apply_block_labels(axes_c[0])
            draw_contrib_boxes(axes_c[0])

            axes_c[1].imshow(
                model_upper,
                # cmap=cmap_rdm,
                #norm=norm_rdm,
                cmap = 'coolwarm',
                vmax=1, vmin=0,
                interpolation="None",
                aspect="equal",
            )
            axes_c[1].set_title(
                f"{sub} {voxel_tag}: \n {args.model} \n model RDM ({mask_tag}, r={corr:.3f})",
                fontsize=14,
            )
            apply_block_labels(axes_c[1])
            draw_contrib_boxes(axes_c[1])

            axes_c[2].imshow(
                data_upper,
                cmap=cmap_rdm,
                norm=norm_rdm,
                interpolation="None",
                aspect="equal", 
                #vmin=0, vmax=1
            )
            axes_c[2].set_title(
                f"{sub} {voxel_tag}: \n data RDM ({mask_tag}, r={corr:.3f})",
                fontsize=14,
            )
            apply_block_labels(axes_c[2])
            draw_contrib_boxes(axes_c[2])

            analysis_wrapped = textwrap.fill(analysis_str, width=160)
            fig_c.text(
                0.5,
                0.01,
                analysis_wrapped,
                ha="center",
                va="bottom",
                fontsize=10,
            )
            map_path = os.path.join(
                out_dir,
                f"{sub}_toppos{len(top_pos_idx)}_contrib_panels_{args.model}_{mask_tag}.png",
            )
            fig_c.tight_layout()
            fig_c.savefig(map_path, dpi=200)

            if overall_category_counter:
                categories_sorted = sorted(overall_category_counter.keys())
                cat_names = categories_sorted
                pct_values = []
                top_counts = []
                overall_counts = []
                for c in cat_names:
                    top_c = int(top_category_counter.get(c, 0))
                    all_c = int(overall_category_counter.get(c, 0))
                    pct = (100.0 * top_c / all_c) if all_c > 0 else 0.0
                    top_counts.append(top_c)
                    overall_counts.append(all_c)
                    pct_values.append(pct)
                fig_b, ax_b = plt.subplots(
                    1, 1, figsize=(max(10, len(cat_names) * 0.45), 6)
                )
                bars = ax_b.bar(np.arange(len(cat_names)), pct_values, color="#4575b4")
                ax_b.set_xticks(np.arange(len(cat_names)))
                ax_b.set_xticklabels(cat_names, rotation=90, fontsize=8)
                ax_b.set_ylabel("% of overall category count")
                ax_b.set_title(
                    f"{sub} {args.model} {mask_tag}: top-category occurrence as % of full included RDM"
                )
                for b, pct, top_c, all_c in zip(bars, pct_values, top_counts, overall_counts):
                    ax_b.text(
                        b.get_x() + b.get_width() / 2.0,
                        b.get_height(),
                        f"{pct:.1f}%\n({top_c}/{all_c})",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )
                fig_b.tight_layout()
                bar_path = os.path.join(
                    out_dir,
                    f"{sub}_top{len(top_store_idx)}_input_vector_category_pct_{args.model}_{mask_tag}.png",
                )
                fig_b.savefig(bar_path, dpi=200)

    if SETTINGS["run_default_dual_mask_views"]:
        summary_suffix = "dual-mask-default"
    elif SETTINGS["use_conditions_masking"]:
        summary_suffix = f"masked-{SETTINGS['conditions_mask_name']}"
    else:
        summary_suffix = "unmasked"
    summary_path = os.path.join(
        out_dir, f"summary_toppos_contrib_{args.model}_{summary_suffix}.csv"
    )
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject",
                "mask",
                "rank",
                "index_in_selected_vector",
                "index_in_full_vector",
                "label",
                "model_value",
                "data_value",
                "contribution",
            ]
        )
        writer.writerows(summary_rows)

    vectors_path = os.path.join(
        out_dir, f"top{SETTINGS['top_n_diagnostic']}_input_vectors_{args.model}_{summary_suffix}.csv"
    )
    with open(vectors_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "subject",
                "mask",
                "rank",
                "index_in_selected_vector",
                "index_in_full_vector",
                "condition_label",
                "side_in_pair",
                "input_vector_json",
            ]
        )
        writer.writerows(vector_rows)

    print("done with analysis.")


if __name__ == "__main__":
    main()
