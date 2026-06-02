#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleaned-up model generation for the rodent ephys validation analysis (figure a).

This is a slimmed, vectorised companion to ``mc.simulation.predictions`` that only
keeps what is needed to reproduce the "validation of RSA" poster figure:

    - create_x_regressors_per_state : bin the data into n regressors per state
    - set_continous_models_ephys    : the continuous DSR / progress / location / state models
    - transform_data_to_betas       : collapse a matrix into one beta per bin (vectorised)

Bug fix vs. the original ``mc.simulation.predictions.set_continous_models_ephys``
--------------------------------------------------------------------------------
The original re-defined state boundaries with ``find_start_end_indices`` and sliced
the last subpath as ``walked_path[prev:-1]``, dropping the final time-bin. The
resulting models were always one column shorter than both the regressors and the
neural data (the "1027 vs 1028" crash in ``transform_data_to_betas``).

This clean version supports both segmentation choices:

    - ``recorded_timings``: use the directly recorded reward-arrival timings.
    - ``reward_dwell``: reproduce the old reward-dwell boundary convention for
      intermediate rewards, but force the final boundary to ``len(walked_path)``
      so models, regressors and neural data all share the same number of columns.

@author: Svenja Kuechenhoff (cleaned)
"""

import numpy as np
import scipy.stats
from scipy.stats import multivariate_normal
from itertools import product
from scipy.signal import argrelextrema


VALID_SEGMENTATIONS = ('recorded_timings', 'reward_dwell')


def find_start_end_indices(locations, index):
    """Return the start and inclusive end index of the dwell at ``index``."""
    locations = np.asarray(locations)
    if len(locations) == 0:
        raise ValueError("cannot find a dwell in an empty trajectory")

    index = int(index)
    if index < 0:
        index = len(locations) + index
    if index < 0 or index >= len(locations):
        raise IndexError(f"index {index} is out of bounds for trajectory length {len(locations)}")

    target_value = locations[index]
    start_idx = index
    end_idx = index

    while start_idx > 0 and locations[start_idx - 1] == target_value:
        start_idx -= 1
    while end_idx < len(locations) - 1 and locations[end_idx + 1] == target_value:
        end_idx += 1

    return start_idx, end_idx


def _subpath_boundaries(beh_data_curr_rep_dict, segmentation='recorded_timings'):
    """Return length-safe subpath boundaries for the selected segmentation."""
    if segmentation not in VALID_SEGMENTATIONS:
        raise ValueError(f"segmentation must be one of {VALID_SEGMENTATIONS}, got {segmentation!r}")

    step_no = beh_data_curr_rep_dict['step_number']
    subpath_timings = [int(x) for x in beh_data_curr_rep_dict['timings_repeat']]
    walked_path = np.asarray(beh_data_curr_rep_dict['trajectory'])
    n_states = len(step_no)
    path_len = len(walked_path)

    if len(subpath_timings) < n_states + 1:
        raise ValueError("timings_repeat must contain start plus one boundary per state")

    if segmentation == 'recorded_timings':
        boundaries = [max(0, min(path_len, int(subpath_timings[i]))) for i in range(n_states + 1)]
        boundaries[0] = 0
        boundaries[-1] = path_len
    else:
        boundaries = [0]
        for state in range(n_states - 1):
            reward_found_at = int(subpath_timings[state + 1])
            if reward_found_at >= path_len:
                candidate = path_len
            else:
                _, end_at_curr_rew = find_start_end_indices(walked_path, reward_found_at)
                # Match the original convention: this inclusive reward-dwell
                # end index becomes the next slice boundary.
                candidate = int(end_at_curr_rew)

            previous = boundaries[-1]
            recorded = max(previous, min(path_len, int(subpath_timings[state + 1])))
            if candidate < previous or candidate > path_len:
                candidate = recorded
            boundaries.append(candidate)
        boundaries.append(path_len)

    for idx in range(1, len(boundaries)):
        if boundaries[idx] < boundaries[idx - 1]:
            raise ValueError(f"non-monotonic {segmentation} boundaries: {boundaries}")

    return boundaries


def create_x_regressors_per_state(beh_data_curr_rep_dict, no_regs_per_state=10,
                                  segmentation='recorded_timings'):
    """Build a [n_states * no_regs_per_state, n_timebins] binning matrix.

    Each subpath (state) is split into ``no_regs_per_state`` near-equal,
    consecutive chunks; every chunk is a 1-hot row that averages the bins it
    covers when used in ``transform_data_to_betas``.

    Vectorised equivalent of ``mc.simulation.predictions.create_x_regressors_per_state``.
    ``segmentation`` must match the model segmentation used for the same run.
    """
    step_no = beh_data_curr_rep_dict['step_number']
    n_states = len(step_no)
    boundaries = _subpath_boundaries(beh_data_curr_rep_dict, segmentation=segmentation)
    seg_lengths = np.diff(boundaries).astype(int)
    total_len = int(np.sum(seg_lengths))

    regressors = np.zeros([n_states * no_regs_per_state, total_len])

    col_start = 0
    for state, seg_len in enumerate(seg_lengths):
        # near-equal split: the first (seg_len % no_regs) chunks get one extra bin
        chunk_sizes = [seg_len // no_regs_per_state + (1 if x < seg_len % no_regs_per_state else 0)
                       for x in range(no_regs_per_state)]
        chunk_edges = np.cumsum([0] + chunk_sizes)
        for nth in range(no_regs_per_state):
            row = nth + state * no_regs_per_state
            regressors[row, col_start + chunk_edges[nth]: col_start + chunk_edges[nth + 1]] = 1
        col_start += seg_len

    return regressors


def transform_data_to_betas(data_matrix, regressors, intercept=False):
    """One beta per regressor (bin) per row of ``data_matrix``.

    Vectorised replacement for the per-row ``LinearRegression`` loop in
    ``mc.simulation.predictions.transform_data_to_betas``: a single least-squares
    solve over all rows at once. With ``fit_intercept=False`` this is identical
    (up to floating point) to fitting each row separately.

    data_matrix : [n_rows, n_timebins]
    regressors  : [n_regressors, n_timebins]
    returns     : [n_rows, n_regressors]
    """
    data_matrix = np.nan_to_num(data_matrix)
    X = np.transpose(regressors)  # [n_timebins, n_regressors]
    if intercept:
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    # solve X @ B = data_matrix.T  ->  B = [n_regs(+1), n_rows]
    betas, *_ = np.linalg.lstsq(X, data_matrix.T, rcond=None)
    betas = betas.T  # [n_rows, n_regs(+1)]
    if intercept:
        betas = betas[:, 1:]
    return betas


def set_continous_models_ephys(beh_data_curr_rep_dict, grid_size=3, no_phase_neurons=3,
                               fire_radius=0.25, wrap_around=1, plot=False,
                               segmentation='recorded_timings'):
    """Continuous location / phase / state / progress(midnight) / clock(DSR) models.

    Cleaned, length-safe version of
    ``mc.simulation.predictions.set_continous_models_ephys``. Every segmentation
    option covers the full trajectory, so every model has exactly
    ``len(trajectory)`` columns.

    Returns a dict with keys:
        loc_model, phas_model, stat_model, midn_model, phas_stat_model, clo_model
    """
    step_number = beh_data_curr_rep_dict['step_number']
    subpath_timings = beh_data_curr_rep_dict['timings_repeat']
    walked_path = beh_data_curr_rep_dict['trajectory']
    n_states = len(step_number)
    subpath_boundaries = _subpath_boundaries(beh_data_curr_rep_dict, segmentation=segmentation)

    # --- location tuning: a 2D gaussian centred on every grid node ----------
    all_coords = [list(p) for p in product(range(grid_size), range(grid_size))]
    n_fields = grid_size * grid_size
    # precompute a [n_fields(centre) x n_fields(location)] lookup so the per-bin
    # location matrix is just a column-index into this table.
    loc_lookup = np.empty([n_fields, n_fields])
    for centre in range(n_fields):
        f = multivariate_normal(all_coords[centre], cov=fire_radius)
        for loc in range(n_fields):
            loc_lookup[centre, loc] = f.pdf(all_coords[loc])

    # --- phase tuning: gaussians / von Mises tiling one subpath -------------
    neuron_phase_functions = []
    if wrap_around == 1:
        means_at_phase = np.linspace(-np.pi, np.pi, (no_phase_neurons * 2) + 1)[1::2]
        for div in means_at_phase:
            neuron_phase_functions.append(scipy.stats.vonmises(1 / (no_phase_neurons / 5), loc=div))
    else:
        means_at_phase = np.linspace(0, 1, (no_phase_neurons * 2) + 1)[1::2]
        for div in means_at_phase:
            neuron_phase_functions.append(scipy.stats.norm(loc=div, scale=1 / (no_phase_neurons / 2)))

    # --- build the models subpath by subpath --------------------------------
    loc_parts, phase_parts, state_parts, midn_parts, phas_stat_parts = [], [], [], [], []
    for count_paths in range(n_states):
        curr_path = walked_path[subpath_boundaries[count_paths]:subpath_boundaries[count_paths + 1]]
        curr_path = np.asarray(curr_path, dtype=int)
        seg_len = len(curr_path)

        # location model: column per bin = gaussian readout for the visited field
        loc_matrix = loc_lookup[:, curr_path]

        # phase model: evaluate every phase function across the subpath at once
        samplepoints = (np.linspace(-np.pi, np.pi, seg_len) if wrap_around == 1
                        else np.linspace(0, 1, seg_len))
        phase_matrix_subpath = np.array([f.pdf(samplepoints) for f in neuron_phase_functions])

        # state model: "lagged" state encoding, as in the original (subpath 0 empty)
        state_matrix = np.zeros([n_states, seg_len])
        if count_paths > 0:
            state_matrix[count_paths - 1] = 1
        # phase x (current) state, used to fill the clock
        helper_state_matrix = np.zeros([n_states, seg_len])
        helper_state_matrix[count_paths] = 1
        phase_state_subpath = np.repeat(helper_state_matrix, repeats=len(phase_matrix_subpath), axis=0)
        for phase in range(0, len(phase_state_subpath), len(phase_matrix_subpath)):
            phase_state_subpath[phase: phase + len(phase_matrix_subpath)] = (
                phase_matrix_subpath * phase_state_subpath[phase: phase + len(phase_matrix_subpath)])

        # midnight model: make the location neurons phase sensitive
        midnight_model_subpath = np.repeat(loc_matrix, repeats=no_phase_neurons, axis=0)
        for location in range(0, len(midnight_model_subpath), no_phase_neurons):
            midnight_model_subpath[location:location + no_phase_neurons] = (
                midnight_model_subpath[location:location + no_phase_neurons] * phase_matrix_subpath)

        loc_parts.append(loc_matrix)
        phase_parts.append(phase_matrix_subpath)
        state_parts.append(state_matrix)
        midn_parts.append(midnight_model_subpath)
        phas_stat_parts.append(phase_state_subpath)

    result_model_dict = {
        'loc_model': np.concatenate(loc_parts, axis=1),
        'phas_model': np.concatenate(phase_parts, axis=1),
        'stat_model': np.concatenate(state_parts, axis=1),
        'midn_model': np.concatenate(midn_parts, axis=1),
        'phas_stat_model': np.concatenate(phas_stat_parts, axis=1),
    }

    # sanity check that the bug-fix held: model length == recorded run length
    if result_model_dict['midn_model'].shape[1] != len(walked_path):
        raise ValueError(f"model length {result_model_dict['midn_model'].shape[1]} "
                         f"!= run length {len(walked_path)} (segmentation mismatch)")

    # --- clock (DSR) model: fill the midnight rings with progress neurons ----
    midn = result_model_dict['midn_model']
    phas_stat = result_model_dict['phas_stat_model']
    norm_midn = (midn - np.min(midn)) / (np.max(midn) - np.min(midn))
    norm_phas_stat = (phas_stat - np.min(phas_stat)) / (np.max(phas_stat) - np.min(phas_stat))

    n_rings = len(norm_phas_stat)
    clo_model = np.zeros([len(norm_midn) * n_rings, len(norm_midn[0])])
    # every ring's first (midnight) neuron starts from the midnight activation
    for row in range(len(norm_midn)):
        clo_model[row * n_rings, :] = norm_midn[row, :].copy()

    for row in range(len(norm_midn)):
        # peaks of activation of this midnight neuron
        local_maxima = argrelextrema(norm_midn[row, :], np.greater_equal, order=5, mode='wrap')[0].copy()
        # ignore neighbouring maxima
        for index, maxima in enumerate(local_maxima):
            if maxima == local_maxima[index - 1] + 1:
                local_maxima = np.delete(local_maxima, index)
        for activation_neuron in local_maxima:
            horizontal_shift_by = np.argmax(norm_phas_stat[:, activation_neuron])
            shifted_clock = np.roll(norm_phas_stat, horizontal_shift_by * -1, axis=0)
            firing_factor = norm_midn[row, activation_neuron].copy()
            shifted_adjusted_clock = shifted_clock * firing_factor
            # the first row is already seeded from the midnight model
            shifted_adjusted_clock[0] = 0
            clo_model[row * n_rings: row * n_rings + n_rings, :] = (
                clo_model[row * n_rings: row * n_rings + n_rings, :] + shifted_adjusted_clock)

    result_model_dict['clo_model'] = clo_model

    if plot:
        import mc
        for model in result_model_dict:
            mc.simulation.predictions.plot_without_legends(
                result_model_dict[model], titlestring=model, timings_curr_run=subpath_boundaries)

    return result_model_dict
