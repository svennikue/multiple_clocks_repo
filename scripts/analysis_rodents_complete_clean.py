#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean rodent ephys analysis - reproduces poster figure (a):
"Validation of representational similarity analysis approach and modelling rodent data".

For every mouse it:
    1. loads + cleans the raw ephys recordings,
    2. averages the binned DSR / progress / location / state models across runs,
    3. builds RDMs and regresses the model RDMs onto the neural RDM (across tasks),
    4. collects one beta per model per mouse and plots them (right panel),
       optionally with the activation- and cov-by-time RSA panels (left).

This replaces the relevant bits of the old, chunky ``Analysis_compl_Mousedata.py``.

@author: Svenja Kuechenhoff (cleaned)
"""

import os
import pickle
import json
import numpy as np
from scipy.stats import ttest_1samp

import mc.analyse.analyse_ephys_clean as ae
import mc.simulation.predictions_clean as predictions_clean


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
DATA_FOLDER = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/'
SAVE = False
WRITE_METHODS_REPORT = True
PLOT_RSA_PANELS = True       # figure (a) left panels (activation- / cov-by-time)
NO_BINS_PER_STATE = 10
NUMBER_PHASE_NEURONS = 3
MASK_WITHIN = True
SEGMENTATION = 'reward_dwell'

date = np.datetime64('today')
OUT_PATH = f"/Users/xpsy1114/Documents/projects/multiple_clocks/output/{date}/"
METHODS_REPORT_NAME = "rodent_clean_methods_config.json"

MICE = ['mouse_a', 'mouse_b', 'mouse_c', 'mouse_d', 'mouse_e', 'mouse_f', 'mouse_g', 'mouse_h']

N_CONDS_PER_CONFIG = 12   # for the model_DSR/my_RSA analysis (matches the human pipeline)
CONTINUOUS_VARIANTS = ['normal', 'standardized']
DSR_VARIANTS = ['across_z', 'within_z', 'full_z']

SENSITIVITY_EXCLUDE_SESSIONS = {
    # Original file-session indices, not cleaned-list indices.
    'ah04_09122021_10122021': [3],
    'me10_09122021_10122021': [3, 5],
}

SUSPICIOUS_SESSION_NOTES = {
    'ah04_09122021_10122021': {
        'sessions': [3],
        'reason': 'raw file session 3 has unusually low invalid/bridge-location fraction and no normalised file',
    },
    'me10_09122021_10122021': {
        'sessions': [3, 5],
        'reason': 'session 3 has unusually low normalised adjacent-trial neural correlation; session 5 matches the flagged task-5 concern and is retained in normalised DSR unless manually excluded',
    },
}


def _collect_and_plot(results, title, save):
    """Collect per-mouse betas in label order and draw the boxplot+scatter."""
    if not results:
        return
    label_regs = list(next(iter(results.values()))['label_regs'])
    labels = [ae.MODEL_LABELS[m] for m in label_regs]
    data_to_plot = [[results[mouse]['coefs'][i] for mouse in results]
                    for i in range(len(label_regs))]
    tick_list = [i + 0.5 for i in range(len(data_to_plot))]
    ae.plotting_hist_scat(data_to_plot, labels, tick_list, title, save_fig=save)


def _json_float(value):
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(len(values))
    if n == 0:
        return {'n': 0, 'mean': None, 'std': None, 'sem': None, 'median': None,
                'min': None, 'max': None}
    std = float(np.std(values, ddof=1)) if n > 1 else None
    return {
        'n': n,
        'mean': _json_float(np.mean(values)),
        'std': None if std is None else _json_float(std),
        'sem': None if std is None else _json_float(std / np.sqrt(n)),
        'median': _json_float(np.median(values)),
        'min': _json_float(np.min(values)),
        'max': _json_float(np.max(values)),
    }


def _summarise_results(results):
    if not results:
        return {'n_mice': 0, 'mouse_ids': [], 'models': {}}

    label_regs = list(next(iter(results.values()))['label_regs'])
    mouse_ids = list(results.keys())
    summary = {'n_mice': len(mouse_ids), 'mouse_ids': mouse_ids, 'models': {}}

    for idx, model in enumerate(label_regs):
        coefs_by_mouse = {mouse: _json_float(results[mouse]['coefs'][idx]) for mouse in mouse_ids}
        t_by_mouse = {mouse: _json_float(results[mouse]['t_vals'][idx]) for mouse in mouse_ids}
        p_by_mouse = {mouse: _json_float(results[mouse]['p_vals'][idx]) for mouse in mouse_ids}
        coefs = np.asarray([value for value in coefs_by_mouse.values()
                            if value is not None], dtype=float)
        if len(coefs) > 1:
            group_t, group_p = ttest_1samp(coefs, 0, alternative='greater')
            group_test = {'t_one_sample_greater_than_zero': _json_float(group_t),
                          'p_one_sample_greater_than_zero': _json_float(group_p)}
        else:
            group_test = {'t_one_sample_greater_than_zero': None,
                          'p_one_sample_greater_than_zero': None}

        summary['models'][model] = {
            'label': ae.MODEL_LABELS.get(model, model),
            'coef_summary': _stats(coefs),
            'mouse_coefs': coefs_by_mouse,
            'mouse_t_vals': t_by_mouse,
            'mouse_p_vals': p_by_mouse,
            'group_test': group_test,
        }
    return summary


def _clean_loaded(mouse_entry, manual_exclusions=None):
    return ae.clean_ephys_data(
        mouse_entry["rewards_configs"], mouse_entry["locations"],
        mouse_entry["neurons"], mouse_entry["timings"], mouse_entry["recday"],
        session_ids=mouse_entry.get("session_ids"),
        manual_exclusions=manual_exclusions,
        return_metadata=True)


def _reward_dwell_timing_diagnostics(task_configs, locations_all, neurons, timings_all):
    """Summarise how reward-dwell boundaries differ from recorded timings."""
    min_trialno = min(len(t) for t in timings_all)
    shifts = []
    mismatches = 0
    n_runs = 0

    for repeat_no in range(min_trialno):
        run_no = -1 * (repeat_no + 1)
        for task_no, task_config in enumerate(task_configs):
            beh_dict, _ = ae.prep_ephys_per_trial(
                timings_all, locations_all, run_no, task_no, task_config, neurons)
            recorded = predictions_clean._subpath_boundaries(
                beh_dict, segmentation='recorded_timings')
            reward_dwell = predictions_clean._subpath_boundaries(
                beh_dict, segmentation='reward_dwell')
            n_runs += 1
            if reward_dwell[-1] != len(beh_dict['trajectory']):
                mismatches += 1
            shifts.extend([reward_dwell[i] - recorded[i]
                           for i in range(1, len(reward_dwell) - 1)])

    return {
        'n_runs_evaluated': int(n_runs),
        'bin_safe_length_mismatch_count': int(mismatches),
        'intermediate_boundary_shift_bins': _stats(shifts),
    }


def _run_analysis(mouse_data, mouse_data_norm, exclusions_by_recday=None,
                  label='main', plot_rsa_panels=False):
    exclusions_by_recday = {} if exclusions_by_recday is None else exclusions_by_recday
    continuous_results = {variant: {} for variant in CONTINUOUS_VARIANTS}
    dsr_results = {variant: {} for variant in DSR_VARIANTS}
    cleaning_metadata = {'raw': {}, 'normalised': {}}
    analysis_metadata = {'continuous': {}, 'dsr': {}, 'reward_dwell_timing': {}}

    for mouse in sorted(mouse_data):
        recday = mouse_data[mouse]['recday']
        manual_exclusions = exclusions_by_recday.get(recday, [])
        exclusion_msg = f" excluding sessions {manual_exclusions}" if manual_exclusions else ""
        print(f"\n=== {label}: {mouse} ({recday}){exclusion_msg} ===")

        # --- analysis 1: continuous models, across tasks (figure a) ---------
        cfg, loc, neu, tim, clean_meta = _clean_loaded(
            mouse_data[mouse], manual_exclusions=manual_exclusions)
        cleaning_metadata['raw'][mouse] = clean_meta
        analysis_metadata['reward_dwell_timing'][mouse] = _reward_dwell_timing_diagnostics(
            cfg, loc, neu, tim)

        continuous = ae.reg_across_tasks(
            cfg, loc, neu, tim, recday,
            plotting=plot_rsa_panels, no_bins_per_state=NO_BINS_PER_STATE,
            number_phase_neurons=NUMBER_PHASE_NEURONS, mask_within=MASK_WITHIN,
            split_by_phase=False, save_path=OUT_PATH if SAVE else None,
            segmentation=SEGMENTATION)
        for variant in CONTINUOUS_VARIANTS:
            continuous_results[variant][mouse] = continuous[variant]
        analysis_metadata['continuous'][mouse] = continuous['metadata']

        # --- analysis 2: model_DSR / my_RSA, z-scored ----------------------
        cfg_n, loc_n, neu_n, tim_n, clean_meta_n = _clean_loaded(
            mouse_data_norm[mouse], manual_exclusions=manual_exclusions)
        cleaning_metadata['normalised'][mouse] = clean_meta_n
        dsr = ae.reg_across_tasks_DSR(
            cfg_n, loc_n, neu_n, tim_n, recday,
            n_conds_per_config=N_CONDS_PER_CONFIG,
            no_phase_neurons=NUMBER_PHASE_NEURONS)
        for variant in DSR_VARIANTS:
            dsr_results[variant][mouse] = dsr[variant]
        analysis_metadata['dsr'][mouse] = dsr['metadata']

    return continuous_results, dsr_results, cleaning_metadata, analysis_metadata


def _build_report(main_continuous, main_dsr, sensitivity_continuous, sensitivity_dsr,
                  main_cleaning, sensitivity_cleaning, main_metadata,
                  sensitivity_metadata):
    return {
        'analysis_date': str(date),
        'data_folder': DATA_FOLDER,
        'mice': MICE,
        'settings': {
            'continuous_segmentation': SEGMENTATION,
            'no_bins_per_state': int(NO_BINS_PER_STATE),
            'number_phase_neurons': int(NUMBER_PHASE_NEURONS),
            'mask_within_for_continuous': bool(MASK_WITHIN),
            'n_conds_per_config_for_dsr': int(N_CONDS_PER_CONFIG),
            'shortest_path_filtering': 'not applied',
            'continuous_variants': {
                'normal': 'unstandardized GLM on RDMs',
                'standardized': 'z-scored data and regressors before GLM',
            },
            'dsr_variants': {
                'across_z': 'across-task/off-block pairs only',
                'within_z': 'within-task off-diagonal pairs only',
                'full_z': 'all off-diagonal pairs; diagonal removed only',
            },
        },
        'suspicious_sessions': SUSPICIOUS_SESSION_NOTES,
        'sensitivity_exclusions_by_recday': SENSITIVITY_EXCLUDE_SESSIONS,
        'main_cleaning': main_cleaning,
        'sensitivity_cleaning': sensitivity_cleaning,
        'main_analysis_metadata': main_metadata,
        'sensitivity_analysis_metadata': sensitivity_metadata,
        'results': {
            'continuous': {
                variant: _summarise_results(main_continuous[variant])
                for variant in CONTINUOUS_VARIANTS
            },
            'dsr': {
                variant: _summarise_results(main_dsr[variant])
                for variant in DSR_VARIANTS
            },
            'sensitivity_without_suspicious_sessions': {
                'continuous': {
                    variant: _summarise_results(sensitivity_continuous[variant])
                    for variant in CONTINUOUS_VARIANTS
                },
                'dsr': {
                    variant: _summarise_results(sensitivity_dsr[variant])
                    for variant in DSR_VARIANTS
                },
            },
        },
    }


def main():
    if (SAVE or WRITE_METHODS_REPORT) and not os.path.isdir(OUT_PATH):
        os.makedirs(OUT_PATH)

    # two views of the same recordings: raw (variable length) for the continuous
    # set_continous_models_ephys pipeline, and normalised (360 bins/trial) for
    # the model_DSR / my_RSA pipeline.
    print("Loading mouse data (raw + normalised) ...")
    mouse_data = ae.load_ephys_data(MICE, Data_folder=DATA_FOLDER, raw=True)
    mouse_data_norm = ae.load_ephys_data(MICE, Data_folder=DATA_FOLDER, raw=False)

    all_results, dsr_results, main_cleaning, main_metadata = _run_analysis(
        mouse_data, mouse_data_norm, label='main',
        plot_rsa_panels=PLOT_RSA_PANELS)
    sensitivity_results, sensitivity_dsr_results, sensitivity_cleaning, sensitivity_metadata = _run_analysis(
        mouse_data, mouse_data_norm, exclusions_by_recday=SENSITIVITY_EXCLUDE_SESSIONS,
        label='sensitivity', plot_rsa_panels=False)

    if SAVE:
        with open(os.path.join(OUT_PATH, "mouse_clean_res_dic"), 'wb') as f:
            pickle.dump({
                'continuous': all_results,
                'dsr': dsr_results,
                'sensitivity_continuous': sensitivity_results,
                'sensitivity_dsr': sensitivity_dsr_results,
            }, f)

    report = _build_report(
        all_results, dsr_results, sensitivity_results, sensitivity_dsr_results,
        main_cleaning, sensitivity_cleaning, main_metadata, sensitivity_metadata)
    if WRITE_METHODS_REPORT:
        report_path = os.path.join(OUT_PATH, METHODS_REPORT_NAME)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote methods/config report: {report_path}")

    # ----- figure (a) right: continuous-model betas across mice -------------
    for variant in CONTINUOUS_VARIANTS:
        _collect_and_plot(
            all_results[variant],
            f'betas per mouse - continuous models {variant}, across tasks, averaged over runs',
            save=OUT_PATH if SAVE else False)

    # ----- second version: model_DSR / my_RSA betas across mice -------------
    for variant in DSR_VARIANTS:
        _collect_and_plot(
            dsr_results[variant],
            f'betas per mouse - model_DSR RSA {variant}, z-scored neurons',
            save=OUT_PATH if SAVE else False)
    return all_results, dsr_results


if __name__ == '__main__':
    all_results, dsr_results = main()
