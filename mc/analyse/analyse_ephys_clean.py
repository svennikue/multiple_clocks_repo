#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleaned-up rodent ephys analysis used for the RSA-validation poster figure (a).

Keeps only the "across tasks, averaged over runs" pipeline that produces:
    - the per-mouse betas for the SMB/DSR, location, subgoal-progress and state models
      (right panel of figure a), and
    - the activation-by-time and cov-by-time RSA panels (left of figure a).

Companion to the bug-fixed, vectorised ``mc.simulation.predictions_clean``.
The heavy RDM/GLM helpers (``within_task_RDM``, ``GLM_RDMs``) and the matrix
plotter (``plot_without_legends``) are reused from the original modules - they
are already vectorised and carry no bug.

@author: Svenja Kuechenhoff (cleaned)
"""

import os
import numpy as np
import scipy.stats
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp

import mc.simulation.predictions_clean as predictions_clean
import mc.simulation.RDMs as RDMs
import mc.simulation.predictions as predictions  # model_DSR + plot_without_legends
import mc.analyse.my_RSA as my_RSA               # shared RSA helpers (human pipeline)


# the four models entering the GLM (midnight deliberately excluded), in the
# order GLM_RDMs returns them (alphabetical).
REGRESSORS_TO_INCLUDE = ['clo_model', 'loc_model', 'phas_model', 'stat_model']

# human-readable labels for the four regressors (used by both analyses)
MODEL_LABELS = {
    'clo_model': 'DSR', 'dsr': 'DSR',
    'stat_model': 'Location in Task', 'stat': 'Location in Task',
    'loc_model': 'Physical Location', 'loc': 'Physical Location',
    'phas_model': 'Subgoal Progress', 'phas': 'Subgoal Progress',
}


# ---------------------------------------------------------------------------
# data loading / cleaning
# ---------------------------------------------------------------------------
def load_ephys_data(dict_labels, Data_folder, raw=True):
    """Load the ephys recordings for the requested mice.

    ``raw=True``  -> the variable-length raw recordings (``*_raw_*`` files),
                     used by ``reg_across_tasks``.
    ``raw=False`` -> the already-normalised recordings binned to 360 bins/trial
                     (90 per state): ``Neuron_*`` (n_neurons, n_trials, 360) and
                     ``Location_*`` (n_trials, 360). Used by ``reg_across_tasks_DSR``.
    """
    rec_days = ['me11_05122021_06122021', 'me11_01122021_02122021', 'me10_09122021_10122021',
                'me08_10092021_11092021', 'ah04_09122021_10122021', 'ah04_05122021_06122021',
                'ah04_01122021_02122021', 'ah04_01122021_02122021', 'ah03_18082021_19082021']

    loc_prefix = 'Location_raw_' if raw else 'Location_'
    neu_prefix = 'Neuron_raw_' if raw else 'Neuron_'

    data = {}
    for i, mouse in enumerate(dict_labels):
        mouse_recday = rec_days[i]
        data[mouse] = {}
        rewards_configs = np.load(Data_folder + 'Task_data_' + mouse_recday + '.npy')
        if mouse == 'mouse_d':
            # the ephys file for the last task on that day was lost -> drop it
            rewards_configs = rewards_configs[0:-1, :].copy()
        data[mouse]["anchor_lag"] = np.load(Data_folder + 'Anchor_lag_' + mouse_recday + '.npy')
        data[mouse]["anchor_lag_threshold"] = np.load(Data_folder + 'Anchor_lag_threshold_' + mouse_recday + '.npy')
        data[mouse]["cells"] = np.load(Data_folder + 'Phase_state_place_anchored_' + mouse_recday + '.npy')

        no_task_configs = len(rewards_configs)
        # some sessions have no normalised recording (e.g. session 3 is missing
        # for a couple of recdays). Session indices are preserved in the file
        # names, so only load the sessions that exist and keep the matching
        # task-config rows aligned with the loaded neurons/locations.
        locations, neurons, timings, kept_sessions = [], [], [], []
        for session in range(no_task_configs):
            neu_file = Data_folder + neu_prefix + mouse_recday + '_' + str(session) + '.npy'
            if not os.path.exists(neu_file):
                continue
            locations.append(np.load(Data_folder + loc_prefix + mouse_recday + '_' + str(session) + '.npy'))
            neurons.append(np.load(neu_file))
            timings.append(np.load(Data_folder + 'trialtimes_' + mouse_recday + '_' + str(session) + '.npy'))
            kept_sessions.append(session)
        # drop task-config rows whose session was not recorded
        data[mouse]["rewards_configs"] = rewards_configs[kept_sessions]
        data[mouse]["locations"] = locations
        data[mouse]["neurons"] = neurons
        data[mouse]["timings"] = timings
        data[mouse]["session_ids"] = kept_sessions
        data[mouse]["recday"] = mouse_recday

        # one-hot of each neuron's preferred anchor lag
        anchor_lag = data[mouse]["anchor_lag"]
        neuron_type = np.zeros((len(anchor_lag), anchor_lag.shape[1]))
        neuron_type[np.arange(len(anchor_lag)), np.argmax(anchor_lag, axis=1)] = 1
        data[mouse]["neuron_type"] = neuron_type

    return data


def clean_ephys_data(task_configs, locations_all, neurons, timings_all, mouse_recday,
                     ignore_double_tasks=True, session_ids=None, manual_exclusions=None,
                     return_metadata=False):
    """Drop task configs that are far too short, duplicated, or manually excluded.

    ``session_ids`` should contain the original file-session index for each loaded
    task. If omitted, list indices are used for backwards compatibility.
    """
    if session_ids is None:
        session_ids = list(range(len(task_configs)))
    session_ids = [int(s) for s in session_ids]
    manual_exclusions = set() if manual_exclusions is None else {int(s) for s in manual_exclusions}

    max_length = max(len(run) for run in locations_all)
    too_short = [i for i, run in enumerate(locations_all) if len(run) < max_length / 3]

    ignore = set(too_short)
    reasons = {i: ['too_short'] for i in too_short}

    for idx, session_id in enumerate(session_ids):
        if session_id in manual_exclusions:
            ignore.add(idx)
            reasons.setdefault(idx, []).append('manual_exclusion')

    duplicate_groups = []
    if ignore_double_tasks:
        configs = [tuple(t) for t in task_configs]
        seen = {}
        for idx, cfg in enumerate(configs):
            seen.setdefault(cfg, []).append(idx)
        # for each set of duplicates keep the run with the most trials
        for cfg, idxs in seen.items():
            if len(idxs) > 1:
                best = max(idxs, key=lambda i: len(timings_all[i]))
                duplicate_groups.append({
                    'task_config': [int(x) for x in cfg],
                    'list_indices': [int(i) for i in idxs],
                    'session_ids': [int(session_ids[i]) for i in idxs],
                    'kept_list_index': int(best),
                    'kept_session_id': int(session_ids[best]),
                })
                for i in idxs:
                    if i != best:
                        ignore.add(i)
                        reasons.setdefault(i, []).append('duplicate_task_config')

    keep = [i for i in range(len(task_configs)) if i not in ignore]
    task_configs_clean = [task_configs[i] for i in keep]
    locations_clean = [locations_all[i] for i in keep]
    neurons_clean = [neurons[i] for i in keep]
    timings_clean = [timings_all[i] for i in keep]

    if not return_metadata:
        return task_configs_clean, locations_clean, neurons_clean, timings_clean

    metadata = {
        'mouse_recday': mouse_recday,
        'n_loaded_tasks': int(len(task_configs)),
        'loaded_session_ids': session_ids,
        'n_kept_tasks': int(len(keep)),
        'kept_list_indices': [int(i) for i in keep],
        'kept_session_ids': [int(session_ids[i]) for i in keep],
        'too_short_threshold_bins': float(max_length / 3),
        'too_short_session_ids': [int(session_ids[i]) for i in too_short],
        'manual_exclusion_session_ids': sorted(manual_exclusions),
        'duplicate_groups': duplicate_groups,
        'excluded': [
            {
                'list_index': int(i),
                'session_id': int(session_ids[i]),
                'reasons': reasons.get(i, []),
                'location_length_bins': int(len(locations_all[i])),
                'n_trials': int(len(timings_all[i])),
            }
            for i in sorted(ignore)
        ],
    }
    return task_configs_clean, locations_clean, neurons_clean, timings_clean, metadata


# ---------------------------------------------------------------------------
# per-trial preprocessing (vectorised)
# ---------------------------------------------------------------------------
def prep_ephys_per_trial(timings_all, locations_all, no_trial_in_each_task, task_no, task_config, neurons):
    """Slice out one run, clean the trajectory and derive the behaviour summary."""
    # ms -> bin number (1 bin = 25 ms); integer-truncated, as in the original
    timings_task = (np.asarray(timings_all[task_no]) // 25).astype(int)

    # clean locations: bridges (>9) and NaNs are forward-filled with the last valid node
    loc = np.asarray(locations_all[task_no], dtype=float)
    bad = np.isnan(loc) | (loc > 9)
    valid_idx = np.where(~bad, np.arange(len(loc)), 0)
    np.maximum.accumulate(valid_idx, out=valid_idx)
    loc = loc[valid_idx]
    # fields are 1-9 -> make them 0-8 integers
    locations_task = (loc - 1).astype(int)
    task_config = [int(field - 1) for field in task_config]

    row = timings_task[no_trial_in_each_task]
    trajectory = locations_task[row[0]:row[-1]].copy()

    # z-score the neurons of this run
    curr_neurons = neurons[task_no][:, row[0]:row[-1]].copy()
    curr_neurons = scipy.stats.zscore(curr_neurons, axis=1)

    # subpath boundaries relative to the run start
    timings_curr_run = [int(elem - row[0]) for elem in row]

    # steps per subpath = number of location changes within each subpath
    step_number = []
    for s in range(4):
        seg = locations_task[row[s]:row[s + 1]]
        step_number.append(int(np.sum(seg[1:] != seg[:-1])) if len(seg) > 1 else 0)

    # indices where a step is made (location changes), starting with 0
    changes = np.where(trajectory[1:] != trajectory[:-1])[0] + 1
    index_make_step = [0] + changes.tolist()

    prep_behaviour_dict = {
        'trajectory': trajectory,
        'timings_repeat': timings_curr_run,
        'index_make_step': index_make_step,
        'step_number': step_number,
    }
    return prep_behaviour_dict, curr_neurons


# ---------------------------------------------------------------------------
# the across-tasks RSA + GLM
# ---------------------------------------------------------------------------
def glm_rdms_standardized(data_matrix, regressor_dict, mask_within=True, no_tasks=None):
    """Standardized-beta GLM on RDMs, on the same footing as ``my_RSA.evaluate_model``.

    Same masking / upper-triangle extraction as ``mc.simulation.RDMs.GLM_RDMs``,
    but each regressor and the data RDM are z-scored (within this mouse) before
    the OLS, so the betas are partial standardized coefficients - directly
    comparable across mice and with the model_DSR pipeline.
    """
    data_matrix = data_matrix.copy()
    regressor_dict = {m: regressor_dict[m].copy() for m in regressor_dict}

    if mask_within:
        block = int(np.round(len(data_matrix) / no_tasks))
        within_task_mask = np.kron(np.eye(no_tasks), np.ones((block, block)))
        # nudge the mask to the data-matrix size (early/mid/late binning can be ±1)
        while len(within_task_mask) < len(data_matrix):
            within_task_mask = np.concatenate((within_task_mask, within_task_mask[:, -2:-1]), axis=1)
            within_task_mask = np.concatenate((within_task_mask, within_task_mask[-2:-1, :]), axis=0)
        while len(within_task_mask) > len(data_matrix):
            within_task_mask = np.delete(within_task_mask, -1, 0)
            within_task_mask = np.delete(within_task_mask, -1, 1)
        data_matrix[within_task_mask == 1] = np.nan
        for m in regressor_dict:
            regressor_dict[m][within_task_mask == 1] = np.nan

    dimension = len(data_matrix)
    triu = np.triu_indices(dimension, -1)

    reg_labels = list(regressor_dict.keys())
    X = np.vstack([regressor_dict[m][triu] for m in reg_labels])  # (n_regs, n_entries)
    y = data_matrix[triu]

    # drop entries that are NaN in the data or any regressor
    nan_mask = np.isnan(y) | np.isnan(X).any(axis=0)
    X = X[:, ~nan_mask]
    y = y[~nan_mask]

    # z-score each regressor and the data vector (parity with evaluate_model)
    Xz = (X - np.nanmean(X, axis=1, keepdims=True)) / np.nanstd(X, axis=1, keepdims=True)
    yz = (y - np.nanmean(y)) / np.nanstd(y)

    est = sm.OLS(yz, sm.add_constant(np.transpose(Xz))).fit()
    return {
        'coefs': np.asarray(est.params[1:]),      # standardized betas (no intercept)
        'label_regs': reg_labels,
        't_vals': np.asarray(est.tvalues[1:]),
        'p_vals': np.asarray(est.pvalues[1:]),
    }


def glm_rdms_unstandardized(data_matrix, regressor_dict, mask_within=True, no_tasks=None):
    """Unstandardized-beta GLM using the original RDM helper."""
    data_matrix = data_matrix.copy()
    regressor_dict = {m: regressor_dict[m].copy() for m in regressor_dict}
    return RDMs.GLM_RDMs(data_matrix, regressor_dict, mask_within=mask_within,
                         no_tasks=no_tasks, plotting=False)


def reg_across_tasks(task_configs, locations_all, neurons, timings_all, mouse_recday,
                     plotting=False, no_bins_per_state=10, number_phase_neurons=3,
                     mask_within=True, split_by_phase=False, save_path=None,
                     segmentation='reward_dwell'):
    """Average the binned models across runs, build RDMs and run the GLM.

    Returns ``normal`` (unstandardized betas) and ``standardized`` (z-scored
    data/regressor betas), plus per-phase results when ``split_by_phase`` is set.
    """
    # only use as many repeats as the task with the fewest runs has
    min_trialno = min(len(t) for t in timings_all)

    sum_models = None
    for repeat_no in range(min_trialno):
        per_repeat = {}  # model -> betas concatenated across task configs
        for task_no, task_config in enumerate(task_configs):
            # take the nth-from-last run of each task
            run_no = -1 * (repeat_no + 1)
            beh_dict, curr_neurons = prep_ephys_per_trial(
                timings_all, locations_all, run_no, task_no, task_config, neurons)

            model_dict = predictions_clean.set_continous_models_ephys(
                beh_dict, no_phase_neurons=number_phase_neurons,
                segmentation=segmentation)
            model_dict['curr_neurons'] = curr_neurons.copy()

            regs = predictions_clean.create_x_regressors_per_state(
                beh_dict, no_regs_per_state=no_bins_per_state,
                segmentation=segmentation)

            for model in sorted(model_dict):
                betas = predictions_clean.transform_data_to_betas(model_dict[model], regs)
                if task_no == 0:
                    per_repeat[model] = betas
                else:
                    per_repeat[model] = np.concatenate((per_repeat[model], betas), axis=1)

        if sum_models is None:
            sum_models = {m: per_repeat[m].copy() for m in per_repeat}
        else:
            for m in per_repeat:
                sum_models[m] += per_repeat[m]

    ave_models = {m: sum_models[m] / min_trialno for m in sum_models}
    print(f"averaged {min_trialno} repeats for {mouse_recday}")

    # RDMs for every model and the data
    RDM_dict = {m: RDMs.within_task_RDM(ave_models[m], plotting=False) for m in ave_models}

    # standardized-beta GLM (comparable across mice and with the model_DSR pipeline)
    regressors = {m: RDM_dict[m] for m in REGRESSORS_TO_INCLUDE}
    results_normal = glm_rdms_unstandardized(RDM_dict['curr_neurons'], regressors,
                                             mask_within, no_tasks=len(task_configs))
    results_standardized = glm_rdms_standardized(RDM_dict['curr_neurons'], regressors,
                                                 mask_within, no_tasks=len(task_configs))
    result_dict = {
        'normal': results_normal,
        'standardized': results_standardized,
        'metadata': {
            'mouse_recday': mouse_recday,
            'segmentation': segmentation,
            'n_task_configs': int(len(task_configs)),
            'min_trialno_repeats_used': int(min_trialno),
            'no_bins_per_state': int(no_bins_per_state),
            'number_phase_neurons': int(number_phase_neurons),
            'mask_within': bool(mask_within),
            'n_neurons': int(ave_models['curr_neurons'].shape[0]),
            'n_model_columns': int(ave_models['curr_neurons'].shape[1]),
            'regressors_to_include': REGRESSORS_TO_INCLUDE.copy(),
        },
    }

    if plotting:
        plot_rsa_panels(ave_models['curr_neurons'], no_bins_per_state, len(task_configs),
                        title=f"{mouse_recday}", save_path=save_path)

    if split_by_phase:
        result_dict['phases'] = _reg_split_by_phase(
            ave_models, no_bins_per_state, len(task_configs), mask_within)

    return result_dict


def _reg_split_by_phase(ave_models, no_bins_per_state, no_tasks, mask_within):
    """Run the same GLM separately for the early / mid / late thirds of each state.

    Phase membership is read straight off the regressor structure: each state's
    ``no_bins_per_state`` binned columns are split into three near-equal groups.
    """
    phase_string = ['early', 'mid', 'late']
    n_states = 4
    total_cols = ave_models['curr_neurons'].shape[1]
    cols_per_task = n_states * no_bins_per_state

    # within one state's bins, assign each bin to a phase third
    edges = np.linspace(0, no_bins_per_state, 4).round().astype(int)
    bin_phase = np.empty(no_bins_per_state, dtype=int)
    for p in range(3):
        bin_phase[edges[p]:edges[p + 1]] = p
    # tile across states and tasks
    col_phase = np.tile(np.repeat(bin_phase[None, :], n_states, axis=0).ravel(),
                        total_cols // cols_per_task)

    results_phase = {}
    for p, phase in enumerate(phase_string):
        mask = np.where(col_phase == p)[0]
        RDM_dict = {m: RDMs.within_task_RDM(ave_models[m][:, mask], plotting=False) for m in ave_models}
        regressors = {m: RDM_dict[m].copy() for m in REGRESSORS_TO_INCLUDE}
        results_phase[phase] = RDMs.GLM_RDMs(RDM_dict['curr_neurons'].copy(), regressors,
                                             mask_within, no_tasks=no_tasks)
    return results_phase


# ---------------------------------------------------------------------------
# second analysis: across-task RSA built exactly like the human-cell pipeline
# (scripts/RSA_DSR_ROIs_simple.py) - z-scored neurons, model_DSR, my_RSA
# ---------------------------------------------------------------------------
def _mode_path_360(location_trials):
    """Mode location across trials -> a clean 360-bin integer node path (0-8).

    ``location_trials`` is the normalised ``Location_*`` array (n_trials, 360),
    with nodes 1-9, bridges 10-21 and NaNs. Bridges/NaNs are forward-filled
    with the last visited node, then shifted to 0-based indices for ``model_DSR``.
    """
    mode = scipy.stats.mode(location_trials, axis=0, keepdims=False, nan_policy='omit').mode
    mode = np.asarray(mode, dtype=float)
    bad = np.isnan(mode) | (mode > 9)
    valid_idx = np.where(~bad, np.arange(len(mode)), 0)
    np.maximum.accumulate(valid_idx, out=valid_idx)
    mode = mode[valid_idx]
    return (mode - 1).astype(int)


def _upper_no_diag(matrix):
    """Flatten the upper triangle with the diagonal removed."""
    matrix = np.asarray(matrix)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _evaluate_dsr_variant(model_vectors, data_vector, order):
    """Run the shared z-scored RSA GLM for one DSR vector variant."""
    stacked = np.stack([model_vectors[k] for k in order], axis=1)
    t_vals, betas, p_vals = my_RSA.evaluate_model(stacked, np.asarray(data_vector))
    return {
        'coefs': np.asarray(betas, dtype=float).ravel(),
        't_vals': np.asarray(t_vals, dtype=float).ravel(),
        'p_vals': np.asarray(p_vals, dtype=float).ravel(),
        'label_regs': order,
    }


def reg_across_tasks_DSR(task_configs, locations_all, neurons, timings_all, mouse_recday,
                         n_conds_per_config=12, no_phase_neurons=3, plotting=False):
    """Across-task RSA in parallel with ``scripts/RSA_DSR_ROIs_simple.py``.

    Uses the *normalised* recordings (360 bins/trial). For each task config it
    averages over repeats, downsamples to ``n_conds_per_config`` conditions and
    z-scores per neuron; the across-task neural RDM is regressed on the four
    ``model_DSR`` model RDMs (DSR, state, location, phase) using ``my_RSA``.

    Returns three z-scored RSA variants:

        - ``across_z``: across-task/off-block pairs only (the previous DSR result).
        - ``within_z``: within-task off-diagonal pairs only.
        - ``full_z``: all off-diagonal pairs, with only the diagonal removed.
    """
    N = n_conds_per_config
    if 360 % N != 0:
        raise ValueError(f"n_conds_per_config={N} must divide 360")
    binlen = 360 // N
    no_tasks = len(task_configs)

    # --- neural data: (n_neurons, n_configs * N), then z-score per neuron ----
    data_cols = []
    for task_no in range(no_tasks):
        neu = neurons[task_no]                      # (n_neurons, n_trials, 360)
        avg = np.nanmean(neu, axis=1)               # (n_neurons, 360)
        ds = avg.reshape(avg.shape[0], N, binlen).mean(axis=2)   # (n_neurons, N)
        data_cols.append(ds)
    mat_all = np.hstack(data_cols)                  # (n_neurons, n_configs*N)
    mu = np.nanmean(mat_all, axis=1)
    sd = np.nanstd(mat_all, axis=1)
    sd[sd == 0] = 1
    mat_all_z = (mat_all.T - mu) / sd               # (n_configs*N, n_neurons)

    data_RDM_within, data_RDM_across, data_RDM_full = my_RSA.compute_crosscorr_within(
        mat_all_z, plotting=plotting, include_diagonal=False,
        no_tasks=no_tasks, model=f'data z-scored {mouse_recday}', block_size=N)
    data_vectors = {
        'across_z': data_RDM_across[0],
        'within_z': data_RDM_within[0],
        'full_z': _upper_no_diag(data_RDM_full),
    }

    # --- model RDMs from model_DSR, same across-task split -------------------
    # model_DSR returns: loc, phas, stat, midn, clo(=dsr), phas_stat, clo_subpath
    model_cols = {'dsr': [], 'stat': [], 'loc': [], 'phas': []}
    for task_no in range(no_tasks):
        walked = _mode_path_360(locations_all[task_no])
        loc_m, phas_m, stat_m, _midn, dsr_m, _phas_stat, _dsr_nn = predictions.model_DSR(
            locations=walked, no_phase_neurons=no_phase_neurons)
        for key, M in [('dsr', dsr_m), ('stat', stat_m), ('loc', loc_m), ('phas', phas_m)]:
            ds = M.reshape(M.shape[0], N, binlen).mean(axis=2)   # (features, N)
            model_cols[key].append(ds)

    model_RDM = {'across_z': {}, 'within_z': {}, 'full_z': {}}
    for key, cols in model_cols.items():
        concat = np.concatenate(cols, axis=1).T     # (n_configs*N, features)
        within, across, full = my_RSA.compute_crosscorr_within(
            concat, plotting=False, include_diagonal=False,
            no_tasks=no_tasks, model=key, block_size=N)
        model_RDM['across_z'][key] = across[0]
        model_RDM['within_z'][key] = within[0]
        model_RDM['full_z'][key] = _upper_no_diag(full)

    # --- one GLM with all four model RDMs (parallel to a my_RSA combo) -------
    order = ['dsr', 'stat', 'loc', 'phas']
    result = {
        variant: _evaluate_dsr_variant(model_RDM[variant], data_vectors[variant], order)
        for variant in ['across_z', 'within_z', 'full_z']
    }
    result['metadata'] = {
        'mouse_recday': mouse_recday,
        'n_task_configs': int(no_tasks),
        'n_conds_per_config': int(N),
        'binlen': int(binlen),
        'number_phase_neurons': int(no_phase_neurons),
        'neural_input': 'normalised recordings averaged over trials and z-scored per neuron',
        'variants': {
            'across_z': 'across-task/off-block pairs only',
            'within_z': 'within-task off-diagonal pairs only',
            'full_z': 'all off-diagonal pairs; diagonal removed only',
        },
        'regressors_to_include': order,
        'n_neurons': int(mat_all.shape[0]),
        'n_model_columns': int(mat_all.shape[1]),
    }
    return result


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def plot_rsa_panels(data_matrix, no_bins_per_state, no_tasks, title='', save_path=None):
    """Figure (a) left panels: activation-by-time and the cov/RDM-by-time matrix."""
    intervalline = 4 * no_bins_per_state
    predictions.plot_without_legends(
        data_matrix, titlestring=f"activation by time - {title}",
        intervalline=intervalline, saving_file=save_path)
    RDMs.within_task_RDM(
        data_matrix, plotting=True,
        titlestring=f"cov by time - {title}", intervalline=intervalline)


def plotting_hist_scat(data_list, label_string_list, label_tick_list, title_string, save_fig=False):
    """Boxplot + scatter of the per-mouse betas with one-sample t-test stars."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data_list, medianprops=dict(color='black'))

    sigma, mu = 0.08, 0.01
    # first model highlighted (SMB/DSR), rest in the darker colour
    colors = ['#96C5D8'] + ['#882048'] * (len(data_list) - 1)

    global_min, global_max = 0, 0
    for index, contrast in enumerate(data_list):
        noise = sigma * np.random.randn(len(contrast)) + mu
        data_to_plot = np.array(contrast)
        x_positions = index + 1 + noise
        global_min = min(global_min, np.min(data_to_plot))
        global_max = max(global_max, np.max(data_to_plot))
        ax.scatter(x_positions, data_to_plot, color=colors[index], marker='o',
                   s=100, edgecolors='black', linewidth=1)
    
    ax.set_xticks(label_tick_list)
    plt.xticks(rotation=45)
    ax.set_xticklabels(label_string_list)
    ax.set_ylabel('Betas')

    padding = global_max / 10
    ax.set_ylim([global_min - padding * 4, global_max + padding])
    plt.axhline(0, color='grey', ls='dashed', linewidth=1)

    # one-sample t-test against 0 (greater), drawn as significance stars
    for i, model in enumerate(data_list):
        _, p_value = ttest_1samp(model, 0, alternative='greater')
        significance = '***' if p_value < 0.001 else '**' if p_value < 0.005 else '*' if p_value < 0.05 else ''
        if significance:
            ax.text(i + 1, global_min - padding * 4, significance,
                    ha='center', va='bottom', fontsize=30, color='black')
    # import pdb; pdb.set_trace()
    plt.title(title_string, pad=30)
    plt.rcParams.update({'font.size': 30})
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    if save_fig:
        fig.savefig(f"{save_fig}{title_string}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{save_fig}{title_string}.tiff", dpi=300, bbox_inches='tight')
