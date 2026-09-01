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
import re
import numpy as np
import scipy.stats
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp

import mc.simulation.predictions_clean as predictions_clean
import mc.simulation.RDMs as RDMs
import mc.simulation.predictions as predictions  # model_DSR + plot_without_legends
import mc.analyse.my_RSA as my_RSA               # shared RSA helpers (human pipeline)
import mc.plotting.figure_layout as figure_layout  # A4-aware figsize helper
# import pdb; pdb.set_trace()

# the four models entering the GLM (midnight deliberately excluded), in the
# order GLM_RDMs returns them (alphabetical).
REGRESSORS_TO_INCLUDE = ['clo_model', 'loc_model', 'phas_model', 'stat_model', 'midnight']

# human-readable labels for each model. Used by both the rodent pipeline and
# the human-cells pipeline (scripts/RSA_DSR_ROIs_simple.py); multiple aliases
# may map to the same human-readable label so each script can use its own keys.
MODEL_LABELS = {
    # rodent pipeline keys
    'clo_model': 'Action Plan', 'dsr': 'Action Plan',
    'dsr_fmri': 'Action Plan',      # human-pipeline DSR variant, added 2026-07-26
    'stat_model': 'Position in Seq.', 'stat': 'Position in Seq.',
    'loc_model': 'Physical Location', 'loc': 'Physical Location',
    'phas_model': 'Subgoal Progress', 'phas': 'Subgoal Progress',
    # human-cells pipeline aliases
    'dsr_old':      'Action Plan',
    'midnight':     'Location x Phase', 'midn': 'Location x Phase',
    'state':        'Position in Seq.',
    'location_old': 'Physical Location',
    'phase':        'Subgoal Progress',
}

# Which model-row-structure each key represents. Drives the nested y-axis
# tick labelling inside ``_imshow_with_task_grid`` (full DSR vs midnight vs
# loc/stat/phas).
MODEL_KINDS = {
    'clo_model':    'dsr',  'dsr':          'dsr',
    'dsr_fmri':     'dsr',
    'dsr_old':      'dsr',
    'midnight':     'midn',
    'stat_model':   'stat', 'stat':         'stat',  'state':        'stat',
    'loc_model':    'loc',  'loc':          'loc',   'location_old': 'loc',
    'phas_model':   'phas', 'phas':         'phas',  'phase':        'phas',
}


# ---------------------------------------------------------------------------
# data loading / cleaning
# ---------------------------------------------------------------------------
def discover_recdays(Data_folder):
    """Return the authoritative recday list shipped with the dataset.

    Reads ``Recording_days_combined.npy`` (the manifest written by the original
    authors). Falls back to scanning ``Task_data_*.npy`` if that file is missing.
    """
    manifest = os.path.join(Data_folder, 'Recording_days_combined.npy')
    if os.path.exists(manifest):
        return [str(x) for x in np.load(manifest)]
    pat = re.compile(r'^Task_data_(.+)\.npy$')
    return sorted(m.group(1) for m in (pat.match(f) for f in os.listdir(Data_folder)) if m)


def _discover_sessions(Data_folder, recday, kind):
    """Find every session index for which BOTH the location and neuron files
    exist on disk for the requested ``kind``.

    ``kind`` is ``'raw'`` (variable-length recordings) or ``'norm'`` (360 bins/trial).
    """
    loc_prefix = 'Location_raw_' if kind == 'raw' else 'Location_'
    neu_prefix = 'Neuron_raw_'  if kind == 'raw' else 'Neuron_'
    loc_pat = re.compile(rf'^{loc_prefix}{re.escape(recday)}_(\d+)\.npy$')
    neu_pat = re.compile(rf'^{neu_prefix}{re.escape(recday)}_(\d+)\.npy$')
    loc_sessions, neu_sessions = set(), set()
    for f in os.listdir(Data_folder):
        m = loc_pat.match(f)
        if m: loc_sessions.add(int(m.group(1)))
        m = neu_pat.match(f)
        if m: neu_sessions.add(int(m.group(1)))
    return sorted(loc_sessions & neu_sessions)


# ---------------------------------------------------------------------------
# normalisation: raw (25 ms bins) -> normalised (90 bins per state)
# ---------------------------------------------------------------------------
# Transcribed from the authors' own code — `partition` / `normalise` /
# `raw_to_norm` in Basic_analysis.ipynb cell 21 of
# github.com/mohamadyelgaby/mFC_schema — so the rodent recordings can be
# normalised ourselves. Needed because the OSF release
# (https://osf.io/3d9r2/) ships raw recordings ONLY: there are no normalised
# `Neuron_*` / `Location_*` arrays there, and the ones in the private Drive
# share cover 8 of the 25 recdays.
#
# CAVEAT, stated by the authors: the normalisation they settled on in the end
# is NOT the one published here, and they have not shared it. Running this
# code reproduces their shipped normalised arrays only to r ~ 0.88 (neurons)
# and ~0.80 (locations), never exactly. So this is the closest available
# approximation, not a reproduction — which is precisely why it must be
# applied to ALL recdays uniformly rather than mixed with their files.
RAW_BIN_MS         = 25     # raw recordings are binned at 25 ms
BINS_PER_STATE     = 90     # -> 360 bins per ABCD loop
STATES_PER_TRIAL   = 4


def _partition(alist, indices):
    """Split ``alist`` at ``indices`` — the authors' `partition`."""
    return [np.asarray(alist[i:j]) for i, j in zip(indices[:-1], indices[1:])]


def normalise_segment(xx, num_bins=BINS_PER_STATE, statistic='mean',
                      rate_scaled=True):
    """Resample one state's samples onto ``num_bins`` — the authors' `normalise`.

    The short-segment rule is theirs and is load-bearing: a segment with fewer
    samples than bins is stretched tenfold before binning, so it can fill the
    90 bins at all.

    ``rate_scaled`` controls the accompanying ``/ 10``. The authors' code always
    divides, which is right for a FIRING RATE (ten copies each carrying a tenth
    of the rate leaves the rate unchanged) and wrong for a categorical LOCATION
    (it turns node 7 into 0.7). Their released Location_* arrays hold clean
    integers, so they evidently do not divide there. Pass rate_scaled=False for
    locations. Affects 1.12% of state segments (109/9720), across 38 sessions.
    """
    xx = np.asarray(xx, dtype=float)
    lenxx = len(xx)
    if lenxx == 0:
        return np.full(num_bins, np.nan)
    if lenxx < num_bins:
        xx = np.repeat(xx, 10)
        if rate_scaled:
            xx = xx / 10
        lenxx = lenxx * 10
    return scipy.stats.binned_statistic(
        np.arange(lenxx), xx, statistic, bins=num_bins)[0]


def state_boundaries(trialtimes, raw_bin_ms=RAW_BIN_MS):
    """Raw-bin indices of every state transition, flat across trials.

    Exactly the authors' ``Trial_times_conc``: columns 0-3 of every trial
    concatenated, then the final trial's column 4, converted from ms to raw
    bins. Consecutive entries bracket one state, so state D of trial t runs to
    the START of trial t+1 (in this dataset those two timestamps are equal).
    """
    tt = np.asarray(trialtimes)
    return np.hstack((np.concatenate(tt[:, :-1]), tt[-1, -1])) // raw_bin_ms


def raw_to_norm(raw, trialtimes, statistic='mean', rate_scaled=None,
                num_bins=BINS_PER_STATE, num_states=STATES_PER_TRIAL):
    """Normalise one session's raw recording to (…, n_trials, 360).

    ``raw`` is ``(n_neurons, n_raw_bins)`` (a Neuron_raw array) or
    ``(n_raw_bins,)`` (a Location_raw array); the returned array keeps that
    leading structure, i.e. ``(n_neurons, n_trials, 360)`` or
    ``(n_trials, 360)``.

    ``statistic`` is passed to ``binned_statistic``: 'mean' for firing rates,
    'max' for locations (the authors' ``take_max``) — a mean of node IDs would
    be meaningless. NOTE this choice is not documented by the authors and is
    the one place this function guesses; see `--location-statistic`.

    ``rate_scaled`` defaults to True for 'mean' and False otherwise, so
    locations are not divided by 10 by the short-segment rule (see
    ``normalise_segment``). Pass it explicitly to override.

    No smoothing: the authors smooth (sigma=10) only in the
    ``raw_to_norm(return_mean=True)`` branch that averages over trials, not in
    the per-trial arrays the RSA uses.
    """
    if rate_scaled is None:
        rate_scaled = (statistic == 'mean')
    edges = state_boundaries(trialtimes)
    raw = np.asarray(raw)
    flat = raw.reshape(1, -1) if raw.ndim == 1 else raw

    out = []
    for row in flat:
        segs = _partition(list(row), list(edges))
        binned = np.asarray([normalise_segment(s, num_bins, statistic, rate_scaled)
                             for s in segs])
        # Drop a trailing partial trial so the reshape is exact (the authors do
        # the same); with well-formed trialtimes there is never one to drop.
        binned = binned[:len(binned) - len(binned) % num_states]
        out.append(binned.reshape(-1, num_bins * num_states))

    out = np.asarray(out)
    return out[0] if raw.ndim == 1 else out


def cross_view_session_ids(raw_data, norm_data):
    """For each recday present in both views, return the usable session ids.

    A session is dropped unless it is present AND non-empty in both views.
    The original authors flag a bad session in two different ways, and both
    have to be caught:
      - the normalised file is simply absent from the release, or
      - the file exists but holds an empty array (shape ``(0,)``).
    ah04_05122021_06122021_3, ah04_09122021_10122021_3 and
    me10_09122021_10122021_8 are the second kind — checking only for a missing
    file lets them through as zero-length sessions and blows up pooling.
    """
    out = {}
    for recday in raw_data:
        if recday not in norm_data:
            continue
        usable = {}
        for view in (raw_data, norm_data):
            entry = view[recday]
            usable[id(view)] = {
                sid for sid, loc, neu in zip(entry['session_ids'],
                                             entry['locations'],
                                             entry['neurons'])
                if np.size(loc) > 0 and np.size(neu) > 0}
        out[recday] = sorted(set.intersection(*usable.values()))
    return out


def load_ephys_data(Data_folder, recdays=None, raw=True, norm_folder=None):
    """Load the ephys recordings for the requested recdays.

    ``recdays`` defaults to ``discover_recdays(Data_folder)`` (the manifest).
    ``raw=True``  -> the variable-length raw recordings (``*_raw_*`` files),
                     used by ``reg_across_tasks``.
    ``raw=False`` -> the already-normalised recordings binned to 360 bins/trial
                     (90 per state). Used by ``reg_across_tasks_DSR``.

    ``norm_folder`` (raw=False only) reads the normalised arrays from a folder
    other than ``Data_folder`` — used to load a self-normalised set out of
    derivatives/ without touching the authors' released files.

    Sessions are discovered from the filesystem: only sessions that have BOTH
    a matching ``Location*`` and ``Neuron*`` file (and a ``trialtimes_*`` file)
    of the requested kind are loaded. ``rewards_configs`` is sliced to match
    so the i-th loaded session always aligns with the i-th task config.

    Returns a dict keyed by recday.
    """
    if recdays is None:
        recdays = discover_recdays(Data_folder)

    kind = 'raw' if raw else 'norm'
    loc_prefix = 'Location_raw_' if raw else 'Location_'
    neu_prefix = 'Neuron_raw_'  if raw else 'Neuron_'
    # The normalised view can live somewhere other than the raw release — see
    # scripts/normalise_rodent_ephys.py, which writes a self-normalised set into
    # derivatives/ rather than overwriting the authors' files. Task_data,
    # trialtimes and the optional extras always come from ``Data_folder``.
    view_folder = Data_folder if (raw or norm_folder is None) else norm_folder

    data = {}
    for recday in recdays:
        rewards_configs = np.load(os.path.join(Data_folder, f'Task_data_{recday}.npy'))
        sessions = _discover_sessions(view_folder, recday, kind)
        # only keep sessions that also have a trialtimes file (always present for raw,
        # but check explicitly so a missing trialtimes file fails loudly here, not later)
        sessions = [s for s in sessions
                    if os.path.exists(os.path.join(Data_folder, f'trialtimes_{recday}_{s}.npy'))]
        # Task_data has one row per intended task; drop any row whose session
        # didn't make it onto disk (handles e.g. me08 where the last ephys
        # file was lost, or recdays where the normalised view skips a session).
        if sessions and max(sessions) >= len(rewards_configs):
            raise ValueError(
                f"{recday}: session id {max(sessions)} exceeds Task_data rows "
                f"({len(rewards_configs)}) — manifest/file mismatch.")
        kept_rewards = rewards_configs[sessions]

        locations, neurons, timings = [], [], []
        for s in sessions:
            locations.append(np.load(os.path.join(view_folder, f'{loc_prefix}{recday}_{s}.npy')))
            neurons.append(np.load(os.path.join(view_folder, f'{neu_prefix}{recday}_{s}.npy')))
            timings.append(np.load(os.path.join(Data_folder, f'trialtimes_{recday}_{s}.npy')))

        entry = {
            'recday':          recday,
            'rewards_configs': kept_rewards,
            'locations':       locations,
            'neurons':         neurons,
            'timings':         timings,
            'session_ids':     sessions,
        }

        # extras (anchor lag etc.) — optional, only loaded if present
        for key, fname in [
            ('anchor_lag',           f'Anchor_lag_{recday}.npy'),
            ('anchor_lag_threshold', f'Anchor_lag_threshold_{recday}.npy'),
            ('cells',                f'Phase_state_place_anchored_{recday}.npy'),
        ]:
            path = os.path.join(Data_folder, fname)
            if os.path.exists(path):
                entry[key] = np.load(path)

        if 'anchor_lag' in entry:
            anchor_lag = entry['anchor_lag']
            neuron_type = np.zeros((len(anchor_lag), anchor_lag.shape[1]))
            neuron_type[np.arange(len(anchor_lag)), np.argmax(anchor_lag, axis=1)] = 1
            entry['neuron_type'] = neuron_type

        data[recday] = entry

    return data


def clean_ephys_data(task_configs, locations_all, neurons, timings_all, mouse_recday,
                     session_ids=None, manual_exclusions=None, keep_session_ids=None,
                     return_metadata=False):
    """Minimal cleaning. Keeps every session unless it is manually excluded or
    explicitly absent from ``keep_session_ids`` (typically the intersection of
    available raw + normalised sessions, i.e. the original authors' implicit
    exclusion list).

    Duplicate task configurations (deliberate repeats by the experimenters) are
    kept and tagged via ``duplicate_groups`` in the metadata, so downstream
    pooling can treat them as extra repeats of the same task.

    ``session_ids`` should contain the original file-session index for each
    loaded task; if omitted, list indices are used.
    """
    if session_ids is None:
        session_ids = list(range(len(task_configs)))
    session_ids = [int(s) for s in session_ids]
    manual_exclusions = set() if manual_exclusions is None else {int(s) for s in manual_exclusions}
    keep_session_ids = (None if keep_session_ids is None
                        else {int(s) for s in keep_session_ids})

    reasons = {}
    ignore = set()

    for idx, session_id in enumerate(session_ids):
        if session_id in manual_exclusions:
            ignore.add(idx)
            reasons.setdefault(idx, []).append('manual_exclusion')
        if keep_session_ids is not None and session_id not in keep_session_ids:
            ignore.add(idx)
            reasons.setdefault(idx, []).append('missing_in_other_view')

    # tag duplicates (do NOT drop — they are extra repeats of the same task config)
    configs = [tuple(int(x) for x in t) for t in task_configs]
    seen = {}
    for idx, cfg in enumerate(configs):
        seen.setdefault(cfg, []).append(idx)
    duplicate_groups = [
        {
            'task_config':  list(cfg),
            'list_indices': [int(i) for i in idxs],
            'session_ids':  [int(session_ids[i]) for i in idxs],
        }
        for cfg, idxs in seen.items() if len(idxs) > 1
    ]

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
        'manual_exclusion_session_ids': sorted(manual_exclusions),
        'keep_session_ids_filter': (None if keep_session_ids is None
                                    else sorted(keep_session_ids)),
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
# trial selection (apply BEFORE pool_by_task_config)
# ---------------------------------------------------------------------------
def keep_last_n_trials(task_configs, locations_all, neurons, timings_all,
                       n_required, kind='raw', session_ids=None,
                       return_metadata=False):
    """For every session, drop it if it has fewer than ``n_required`` trials,
    otherwise keep only its last ``n_required`` trials (most learned / stable).

    Apply BEFORE ``pool_by_task_config`` so the per-session minimum is
    enforced before duplicate-config sessions are merged.

    ``kind='raw'``  -> trims only ``timings_all`` (bin-major locations/neurons
                       stay intact; the analysis picks trials by timestamp).
    ``kind='norm'`` -> trims ``locations_all``, ``neurons`` and ``timings_all``
                       along the trial axis (axis 0 / axis 1 / axis 0).
    """
    if session_ids is None:
        session_ids = list(range(len(task_configs)))

    keep, dropped = [], []
    for i in range(len(task_configs)):
        n_trials = np.asarray(timings_all[i]).shape[0]
        if n_trials >= n_required:
            keep.append(i)
        else:
            dropped.append({'session_id': int(session_ids[i]), 'n_trials': int(n_trials)})

    out_cfg = np.asarray([task_configs[i] for i in keep])
    out_loc, out_neu, out_tim = [], [], []
    for i in keep:
        if kind == 'raw':
            out_loc.append(locations_all[i])
            out_neu.append(neurons[i])
            out_tim.append(np.asarray(timings_all[i])[-n_required:])
        elif kind == 'norm':
            out_loc.append(np.asarray(locations_all[i])[-n_required:])
            out_neu.append(np.asarray(neurons[i])[:, -n_required:, :])
            out_tim.append(np.asarray(timings_all[i])[-n_required:])
        else:
            raise ValueError(f"kind must be 'raw' or 'norm', got {kind!r}")

    if not return_metadata:
        return out_cfg, out_loc, out_neu, out_tim

    metadata = {
        'kind':             kind,
        'n_required':       int(n_required),
        'n_sessions_in':    int(len(task_configs)),
        'n_sessions_out':   int(len(keep)),
        'kept_session_ids': [int(session_ids[i]) for i in keep],
        'dropped_sessions': dropped,
    }
    return out_cfg, out_loc, out_neu, out_tim, metadata


# ---------------------------------------------------------------------------
# pool sessions that share the same reward configuration
# ---------------------------------------------------------------------------
MS_PER_BIN = 25   # one location/neuron bin = 25 ms (raw view)


def _forward_fill_session_locations(loc):
    """Forward-fill bridges/NaNs within a single session's 1-D location array.
    Leading bads are back-filled with the first valid node. Keeps the original
    1-9 node encoding so downstream code is unchanged. This is the per-session
    pre-fill applied before pooling so the global fill in
    ``prep_ephys_per_trial`` cannot propagate the last node of session A into
    leading bridges of session B (which would be a spurious teleport).
    """
    loc = np.asarray(loc, dtype=float)
    bad = np.isnan(loc) | (loc > 9)
    if bad.all():
        return loc
    good = np.where(~bad)[0]
    first_good = int(good[0])
    valid_idx = np.where(~bad, np.arange(len(loc)), -1)
    valid_idx[:first_good] = first_good
    np.maximum.accumulate(valid_idx, out=valid_idx)
    return loc[valid_idx]


def pool_by_task_config(task_configs, locations_all, neurons, timings_all,
                        kind='raw', session_ids=None, return_metadata=False):
    """Concatenate sessions sharing the same reward configuration into one
    "long run" each. Trials from the second, third, ... session of a group
    pick up after the trials of the first.

    ``kind='raw'``  -> bin-major arrays. Locations: 1-D (n_bins,). Neurons:
                       (n_neurons, n_bins). Timings: (n_trials, 5) in ms. The
                       second session's timings are shifted by
                       ``len(locations_first) * MS_PER_BIN`` so they keep
                       indexing the concatenated bin axis.
    ``kind='norm'`` -> fixed 360 bins/trial. Locations: (n_trials, 360).
                       Neurons: (n_neurons, n_trials, 360). Timings:
                       (n_trials, 5). Only the trial axis is concatenated.

    Returns the same 4-tuple but with one entry per unique task config.
    """
    if session_ids is None:
        session_ids = list(range(len(task_configs)))

    # group session indices by their task config (preserve first-occurrence order)
    groups, order = {}, []
    for idx, cfg in enumerate(task_configs):
        key = tuple(int(x) for x in cfg)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(idx)

    pooled_cfgs, pooled_locs, pooled_neus, pooled_tims, pooled_session_ids = [], [], [], [], []
    pooled_groups = []

    for cfg_key in order:
        idxs = groups[cfg_key]

        if kind == 'raw':
            # Forward-fill each session's locations BEFORE concatenating so
            # the global forward-fill in prep_ephys_per_trial cannot smear
            # session A's last node into session B's leading bridges/NaNs.
            session_locs = [_forward_fill_session_locations(locations_all[i]) for i in idxs]
            loc_concat = np.concatenate(session_locs, axis=0)
            neu_concat = np.concatenate([neurons[i] for i in idxs], axis=1)
            tim_pieces = [np.asarray(timings_all[idxs[0]])]
            offset_bins = len(session_locs[0])
            for k, i in enumerate(idxs[1:], start=1):
                tim_pieces.append(np.asarray(timings_all[i]) + offset_bins * MS_PER_BIN)
                offset_bins += len(session_locs[k])
            tim_concat = np.concatenate(tim_pieces, axis=0)
        elif kind == 'norm':
            loc_concat = np.concatenate([locations_all[i] for i in idxs], axis=0)
            neu_concat = np.concatenate([neurons[i] for i in idxs], axis=1)
            tim_concat = np.concatenate([np.asarray(timings_all[i]) for i in idxs], axis=0)
        else:
            raise ValueError(f"kind must be 'raw' or 'norm', got {kind!r}")

        pooled_cfgs.append(np.asarray(cfg_key))
        pooled_locs.append(loc_concat)
        pooled_neus.append(neu_concat)
        pooled_tims.append(tim_concat)
        pooled_session_ids.append([int(session_ids[i]) for i in idxs])
        pooled_groups.append({
            'task_config':       list(cfg_key),
            'source_session_ids':[int(session_ids[i]) for i in idxs],
            'n_trials_combined': int(tim_concat.shape[0]),
        })

    pooled_cfgs = np.asarray(pooled_cfgs)

    if not return_metadata:
        return pooled_cfgs, pooled_locs, pooled_neus, pooled_tims

    metadata = {
        'kind': kind,
        'n_sessions_in':  int(len(task_configs)),
        'n_configs_out':  int(len(pooled_cfgs)),
        'pooled_groups':  pooled_groups,
        'pooled_session_ids_per_config': pooled_session_ids,
    }
    return pooled_cfgs, pooled_locs, pooled_neus, pooled_tims, metadata


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
def _clean_node_path(traj):
    """Forward-fill bridges/NaNs in a 1-D trajectory. Leading bads get the
    first valid value (back-fill). Shifts node labels from 1-9 to 0-8.
    """
    traj = np.asarray(traj, dtype=float)
    bad = np.isnan(traj) | (traj > 9)
    if bad.all():
        return np.zeros(len(traj), dtype=int)
    good = np.where(~bad)[0]
    first_good = int(good[0])
    valid_idx = np.where(~bad, np.arange(len(traj)), -1)
    valid_idx[:first_good] = first_good           # back-fill any leading bads
    np.maximum.accumulate(valid_idx, out=valid_idx)
    return (traj[valid_idx] - 1).astype(int)


def _mode_path_360(location_trials):
    """Mode across trials -> a clean 360-bin integer node path (0-8)."""
    mode = scipy.stats.mode(location_trials, axis=0,
                            keepdims=False, nan_policy='omit').mode
    return _clean_node_path(np.asarray(mode, dtype=float))


def _upper_no_diag(matrix):
    """Flatten the upper triangle with the diagonal removed."""
    matrix = np.asarray(matrix)
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def degenerate_data_vector(data_vector):
    """True when an RDM vector carries no information the GLM could fit.

    Two cases, both of which make every regression coefficient meaningless:
      - every entry is NaN, or
      - the finite entries are all identical (zero variance).

    Both arise for a recday with a SINGLE neuron: the population RDM is a
    correlation distance across neurons, and with one feature per column every
    pair collapses to the same value. me10_20122021_21122021 (1 neuron) hits
    this — its RDM is uniformly 1.0, so OLS returns exactly 0.0 for every
    regressor and that 0.0 was entering the group t-test as if it were a
    measurement. It is not a small effect; it is no effect at all.

    This is a validity condition, not a data-dependent cut-off: it asks whether
    the quantity is defined, never how big it is, so it cannot select on the
    result.
    """
    v = np.asarray(data_vector, dtype=float).ravel()
    finite = v[np.isfinite(v)]
    return finite.size == 0 or np.ptp(finite) == 0


def _evaluate_dsr_variant(model_vectors, data_vector, order):
    """Run the shared z-scored RSA GLM for one DSR vector variant.

    Returns NaN coefficients for a degenerate data vector (see
    ``degenerate_data_vector``) so the recday drops out of the group test —
    ``methods_results_stats`` already filters non-finite coefficients.
    """
    if degenerate_data_vector(data_vector):
        nan = np.full(len(order), np.nan)
        return {'coefs': nan, 't_vals': nan.copy(), 'p_vals': nan.copy(),
                'label_regs': order, 'degenerate': True}
    stacked = np.stack([model_vectors[k] for k in order], axis=1)
    t_vals, betas, p_vals = my_RSA.evaluate_model(stacked, np.asarray(data_vector))
    return {
        'coefs': np.asarray(betas, dtype=float).ravel(),
        't_vals': np.asarray(t_vals, dtype=float).ravel(),
        'p_vals': np.asarray(p_vals, dtype=float).ravel(),
        'label_regs': order,
    }


def _phase_residualise_task(neu_task, basis):
    """Per-cell phase residualisation for one task's (n_neurons, n_trials, 360)
    firing-rate array. Shares the residualiser used by the human single-unit
    RSA (`mc.analyse.future_spatial_peaks.phase_residualise`, cosine basis by
    default). Cells are residualised across their (n_trials × 360) matrix so
    the phase basis sees every trial. Returns a new array of the same shape;
    input is not mutated. `basis=None` returns the input unchanged.
    """
    if basis is None:
        return neu_task
    from mc.analyse.future_spatial_peaks import phase_residualise as _resid
    out = np.empty_like(neu_task, dtype=float)
    for c in range(neu_task.shape[0]):
        out[c] = _resid(neu_task[c].astype(float), basis=basis)
    return out


def _downsample_mode_1d(x, target_len):
    """Downsample a 1-D integer node path to `target_len` samples via mode
    over evenly-distributed slots (no bin discarding). Mirrors
    ``downsample_mode`` in RSA_DSR_ROIs_simple.py — slot i uses input bins
    ``[(i*n)//target_len : ((i+1)*n)//target_len]``. Slot sizes differ by
    at most 1 when target_len does not divide n.
    """
    from collections import Counter
    x = np.asarray(x)
    n = len(x)
    if n == target_len:
        return x.astype(int)
    return np.array([
        Counter(x[(i * n) // target_len:((i + 1) * n) // target_len])
            .most_common(1)[0][0]
        for i in range(target_len)
    ], dtype=int)


#: Downsampled bins per condition used by the human-pipeline `dsr_fmri`
# model. Matches LEN_STANDARDISED_PATH in scripts/RSA_DSR_ROIs_simple.py.
#: With N=12 conds this makes a (12 × 144) integer roll matrix per task.
LEN_STANDARDISED_PATH_DSR_FMRI = 12


def _build_dsr_fmri_task(walked_360, n_conds_per_config,
                          len_per_bin=LEN_STANDARDISED_PATH_DSR_FMRI):
    """Build a (n_conds_per_config × n_conds_per_config*len_per_bin) matrix
    matching ``build_mode_path_dsr`` in RSA_DSR_ROIs_simple.py: take the
    mode trajectory, downsample to ``n_conds × len_per_bin`` integer node
    IDs, then for each of the ``n_conds`` rows roll the flattened vector
    left by ``pos * len_per_bin`` so 'current' sits at the front. This
    matrix feeds ``my_RSA.compute_hamming_distance_within``.

    ``len_per_bin`` defaults to LEN_STANDARDISED_PATH_DSR_FMRI (=12), the
    downsampled path length per condition used in the human fMRI pipeline
    — NOT the raw-binlen (360 / N = 30) used elsewhere in this module."""
    base = _downsample_mode_1d(walked_360,
                                target_len=n_conds_per_config * len_per_bin)
    return np.stack([np.roll(base, -pos * len_per_bin)
                     for pos in range(n_conds_per_config)], axis=0)


def _build_dsr_model_cols(locations_all, no_phase_neurons, pool_method, N, binlen):
    """Build per-task binned model matrices. ``pool_method`` picks how trials are pooled:

        - ``'mode_path'``: take the mode trajectory across trials, build one
          model per task from it.
        - ``'per_run_avg'``: build one model per trial, average across trials.
    """
    out = {'dsr': [], 'stat': [], 'loc': [], 'phas': [], 'midn': [],
            'dsr_fmri': []}
    no_tasks = len(locations_all)

    for task_no in range(no_tasks):
        trial_locs = np.asarray(locations_all[task_no])  # (n_trials, 360)

        if pool_method == 'mode_path':
            walked = _mode_path_360(trial_locs)
            loc_m, phas_m, stat_m, midn_m, dsr_m, _, _ = predictions.model_DSR(
                locations=walked, no_phase_neurons=no_phase_neurons)
            per_task = {'dsr': dsr_m, 'stat': stat_m, 'loc': loc_m,
                        'phas': phas_m, 'midn': midn_m}

        elif pool_method == 'per_run_avg':
            sums = None
            for t in range(trial_locs.shape[0]):
                walked = _clean_node_path(trial_locs[t])
                # loc_model, phas_model, stat_model, midn_model, clo_model, phas_stat, clo_model_subpath
                loc_m, phas_m, stat_m, midn_m, dsr_m, _, _ = predictions.model_DSR(
                    locations=walked, no_phase_neurons=no_phase_neurons)
                if sums is None:
                    sums = {'dsr': dsr_m.copy(), 'stat': stat_m.copy(),
                            'loc': loc_m.copy(), 'phas': phas_m.copy(),
                            'midn': midn_m.copy()}
                else:
                    sums['dsr']  += dsr_m
                    sums['stat'] += stat_m
                    sums['loc']  += loc_m
                    sums['phas'] += phas_m
                    sums['midn'] += midn_m
            # For dsr_fmri the "average per-run" pooling would smear the integer
            # location IDs into non-integers; use the mode across runs' walks
            # instead, matching what mode_path does.
            walked_pool = _mode_path_360(trial_locs)
            n_trials = trial_locs.shape[0]
            per_task = {k: v / n_trials for k, v in sums.items()}

        else:
            raise ValueError(f"pool_method must be 'mode_path' or 'per_run_avg', got {pool_method!r}")

        # dsr_fmri (Hamming-space): 12×144 integer roll matrix per task from
        # the mode-path. Same construction as RSA_DSR_ROIs_simple.build_mode_path_dsr.
        walked_for_fmri = walked if pool_method == 'mode_path' else walked_pool
        dsr_fmri_task = _build_dsr_fmri_task(walked_for_fmri, N)   # len_per_bin defaults to 12
        out['dsr_fmri'].append(dsr_fmri_task)

        for key, M in per_task.items():
            out[key].append(M.reshape(M.shape[0], N, binlen).mean(axis=2))

    return out


def process_one_recday(recday,
                       raw_cleaned, raw_kept_session_ids,
                       norm_cleaned, norm_kept_session_ids,
                       n_required, config):
    """Run all RSAs for one recday and one trial-filter setting.

    This is a top-level pickleable function so it can be dispatched by
    ``joblib.Parallel`` from a script running at module scope.

    ``raw_cleaned``  and ``norm_cleaned`` are post-cleaning ``(cfg, loc, neu, tim)``
    tuples (one entry per session); ``*_kept_session_ids`` are their
    corresponding session-id lists. ``n_required`` is the trial-count cut-off
    (or ``None``). ``config`` is a dict carrying the analysis hyperparameters.

    Returns a dict with continuous, DSR (per pool method), across-halves results
    and trim/pool metadata for this (recday, filter) combination.
    """
    # ----- Raw view: trim, pool, continuous RSA -----
    # Skipped when config['run_continuous'] is False — saves the slow per-trial
    # set_continous_models_ephys loop. ``raw_cleaned`` can then be ``None``.
    cont, trim_raw, pool_raw = None, None, None
    if config.get('run_continuous', True) and raw_cleaned is not None:
        cfg, loc, neu, tim = raw_cleaned
        sid_raw = raw_kept_session_ids
        if n_required is not None:
            cfg, loc, neu, tim, trim_raw = keep_last_n_trials(
                cfg, loc, neu, tim, n_required=n_required, kind='raw',
                session_ids=sid_raw, return_metadata=True)
            sid_raw = trim_raw['kept_session_ids']
        cfg, loc, neu, tim, pool_raw = pool_by_task_config(
            cfg, loc, neu, tim, kind='raw',
            session_ids=sid_raw, return_metadata=True)
        cont = reg_across_tasks(
            cfg, loc, neu, tim, recday,
            plotting=config['plot_rsa_panels'],
            no_bins_per_state=config['no_bins_per_state'],
            number_phase_neurons=config['number_phase_neurons'],
            mask_within=config['mask_within'],
            split_by_phase=False, save_path=None,
            segmentation=config['segmentation'])

    # ----- Normalised view: trim, pool, DSR RSA per pool method -----
    cfg_n, loc_n, neu_n, tim_n = norm_cleaned
    trim_norm = None
    sid_norm = norm_kept_session_ids
    if n_required is not None:
        cfg_n, loc_n, neu_n, tim_n, trim_norm = keep_last_n_trials(
            cfg_n, loc_n, neu_n, tim_n, n_required=n_required, kind='norm',
            session_ids=sid_norm, return_metadata=True)
        sid_norm = trim_norm['kept_session_ids']
    cfg_np, loc_np, neu_np, tim_np, pool_norm = pool_by_task_config(
        cfg_n, loc_n, neu_n, tim_n, kind='norm',
        session_ids=sid_norm, return_metadata=True)

    phase_res = config.get('phase_residualise', None)
    combo_order = config.get('combo_order', None)   # None = full 6-model default
    dsr_by_pool = {}
    for pool_method in config['dsr_pool_methods']:
        dsr_by_pool[pool_method] = reg_across_tasks_DSR(
            cfg_np, loc_np, neu_np, tim_np, recday,
            n_conds_per_config=config['n_conds_per_config'],
            no_phase_neurons=config['number_phase_neurons'],
            pool_method=pool_method,
            phase_residualise=phase_res,
            combo_order=combo_order)

    # ----- Across-task-halves: uses POST-trim / PRE-pool normalised data -----
    halves = reg_across_task_halves_DSR(
        cfg_n, loc_n, neu_n, tim_n, recday, session_ids=sid_norm,
        n_conds_per_config=config['n_conds_per_config'],
        no_phase_neurons=config['number_phase_neurons'],
        phase_residualise=phase_res,
        combo_order=combo_order)

    return {
        'recday':       recday,
        'continuous':   cont,
        'dsr_by_pool':  dsr_by_pool,
        'halves':       halves,
        'trim_pool_raw':  {'trim': trim_raw, 'pooling': pool_raw},
        'trim_pool_norm': {'trim': trim_norm, 'pooling': pool_norm},
    }


def split_sessions_into_halves(task_configs, session_ids, timings_all):
    """Split the duplicate-config sessions of a recday into two balanced halves.

    For every task config that appears in >=2 sessions, half-1 starts as the
    first source session and half-2 as the second; any further source sessions
    (e.g. me10 had a triple) are added to whichever half currently has fewer
    total trials. Configs with only one source session are skipped.

    Returns a list of dicts with keys ``config``, ``half1_indices``,
    ``half2_indices``, ``half{1,2}_session_ids``, ``half{1,2}_n_trials``.
    """
    groups, order = {}, []
    for i, cfg in enumerate(task_configs):
        key = tuple(int(x) for x in cfg)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)

    out = []
    for cfg in order:
        idxs = groups[cfg]
        if len(idxs) < 2:
            continue
        h1, h2 = [idxs[0]], [idxs[1]]
        for extra in idxs[2:]:
            n1 = sum(int(np.asarray(timings_all[i]).shape[0]) for i in h1)
            n2 = sum(int(np.asarray(timings_all[i]).shape[0]) for i in h2)
            (h2 if n2 < n1 else h1).append(extra)
        out.append({
            'config':            list(cfg),
            'half1_indices':     h1,
            'half2_indices':     h2,
            'half1_session_ids': [int(session_ids[i]) for i in h1],
            'half2_session_ids': [int(session_ids[i]) for i in h2],
            'half1_n_trials':    sum(int(np.asarray(timings_all[i]).shape[0]) for i in h1),
            'half2_n_trials':    sum(int(np.asarray(timings_all[i]).shape[0]) for i in h2),
        })
    return out


def reg_across_task_halves_DSR(task_configs, locations_all, neurons, timings_all,
                               mouse_recday, session_ids=None,
                               n_conds_per_config=12, no_phase_neurons=3,
                               phase_residualise=None,
                               combo_order=None):
    """Across-task-halves split-half RSA on the normalised view.

    Uses the natural duplicate-config sessions as the two halves (no within-
    session trial splitting, so the halves share no trials and no
    autocorrelation between halves is possible). Configs with only one source
    session are excluded. Pool method is always ``mode_path``.

    Stack convention (matches ``RSA_DSR_ROIs_simple.py``):
        [half1(task1), half1(task2), ..., half1(taskK), half2(task1), ..., half2(taskK)]
    is passed to ``my_RSA.compute_crosscorr``, which returns the half-2-vs-
    half-1 lower-left block symmetrised — i.e. an unbiased across-half RDM.

    Inputs are POST-cleaning, PRE-pooling: one entry per session.
    """
    N = n_conds_per_config
    if 360 % N != 0:
        raise ValueError(f"n_conds_per_config={N} must divide 360")
    binlen = 360 // N
    if session_ids is None:
        session_ids = list(range(len(task_configs)))

    qualifying = split_sessions_into_halves(task_configs, session_ids, timings_all)

    if not qualifying:
        return {'metadata': {
            'mouse_recday':         mouse_recday,
            'n_qualifying_configs': 0,
            'qualifying_groups':    [],
            'n_conds_per_config':   int(N),
            'pool_method':          'mode_path',
        }}

    K = len(qualifying)

    def _half_neural(half):
        """Stack per-task averaged + downsampled neural columns -> (n_neurons, K*N)."""
        cols = []
        for cfg_info in qualifying:
            idxs = cfg_info[f'half{half}_indices']
            neu_concat = np.concatenate([neurons[i] for i in idxs], axis=1)   # (n_neurons, n_trials, 360)
            neu_concat = _phase_residualise_task(neu_concat, phase_residualise)
            avg = np.nanmean(neu_concat, axis=1)                              # (n_neurons, 360)
            ds = avg.reshape(avg.shape[0], N, binlen).mean(axis=2)            # (n_neurons, N)
            cols.append(ds)
        return np.hstack(cols)

    half1_neu = _half_neural(1)
    half2_neu = _half_neural(2)
    mat = np.vstack([half1_neu.T, half2_neu.T])    # (2*K*N, n_neurons)
    mu = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    sd[sd == 0] = 1
    mat_z = (mat - mu) / sd
    data_vec = my_RSA.compute_crosscorr(
        mat_z, plotting=False, include_diagonal=False,
        no_tasks=K, model=f'half-split {mouse_recday}')[0]

    def _half_models(half):
        """Build the six model RDM-input matrices for one half: (K*N, features).
        dsr_fmri is stored as a (K*N, len_per_bin*N) integer matrix (Hamming);
        the other five are cosine-space (K*N, features) rate maps."""
        per_model = {'dsr': [], 'stat': [], 'loc': [], 'phas': [], 'midn': [],
                      'dsr_fmri': []}
        for cfg_info in qualifying:
            idxs = cfg_info[f'half{half}_indices']
            loc_concat = np.concatenate([locations_all[i] for i in idxs], axis=0)
            walked = _mode_path_360(loc_concat)
            loc_m, phas_m, stat_m, midn_m, dsr_m, _, _ = predictions.model_DSR(
                locations=walked, no_phase_neurons=no_phase_neurons)
            for key, M in [('dsr', dsr_m), ('stat', stat_m), ('loc', loc_m),
                           ('phas', phas_m), ('midn', midn_m)]:
                per_model[key].append(M.reshape(M.shape[0], N, binlen).mean(axis=2))
            per_model['dsr_fmri'].append(_build_dsr_fmri_task(walked, N))
        out = {k: np.hstack(v).T for k, v in per_model.items()
                if k != 'dsr_fmri'}
        # dsr_fmri: stack row-wise (K*N, 144 integer node IDs)
        out['dsr_fmri'] = np.concatenate(per_model['dsr_fmri'], axis=0)
        return out

    h1_models = _half_models(1)
    h2_models = _half_models(2)

    model_RDMs = {}
    for k in ('dsr', 'stat', 'loc', 'phas', 'midn'):
        m_combined = np.vstack([h1_models[k], h2_models[k]])
        model_RDMs[k] = my_RSA.compute_crosscorr(
            m_combined, plotting=False, include_diagonal=False,
            no_tasks=K, model=k)[0]
    # dsr_fmri uses Hamming (integer node IDs).
    m_combined = np.vstack([h1_models['dsr_fmri'], h2_models['dsr_fmri']])
    model_RDMs['dsr_fmri'] = my_RSA.compute_hamming_distance(
        m_combined, plotting=False, include_diagonal=False,
        no_tasks=K, model_name='dsr_fmri')[0]

    order = combo_order or ['dsr', 'dsr_fmri', 'stat', 'loc', 'phas', 'midn']
    stacked = np.stack([model_RDMs[k] for k in order], axis=1)
    if degenerate_data_vector(data_vec):
        # Same guard as the full_z path — see degenerate_data_vector.
        nan = np.full(len(order), np.nan)
        t_vals, betas, p_vals = nan, nan.copy(), nan.copy()
    else:
        t_vals, betas, p_vals = my_RSA.evaluate_model(stacked, np.asarray(data_vec))

    return {
        'across_halves': {
            'coefs':       np.asarray(betas, dtype=float).ravel(),
            't_vals':      np.asarray(t_vals, dtype=float).ravel(),
            'p_vals':      np.asarray(p_vals, dtype=float).ravel(),
            'label_regs':  order,
        },
        'metadata': {
            'mouse_recday':         mouse_recday,
            'n_qualifying_configs': K,
            'qualifying_groups':    qualifying,
            'n_conds_per_config':   int(N),
            'pool_method':          'mode_path',
        },
    }


def dsr_example_recday_matrices(task_configs, locations_all, neurons, timings_all,
                                n_conds_per_config=12, no_phase_neurons=3):
    """For one (cleaned + pooled) recday, build:
        - data activation (n_neurons, K*N), z-scored per neuron
        - data RDM (K*N, K*N), full square (1 − cosine)
        - per-model activation matrices, model -> (n_features, K*N)
        - per-model RDMs,                 model -> (K*N, K*N)
    Uses the ``mode_path`` strategy for the models (matches the chosen pipeline).
    """
    N = n_conds_per_config
    binlen = 360 // N
    no_tasks = len(task_configs)

    # data
    data_z, data_rdm = dsr_activation_and_rdm(
        task_configs, locations_all, neurons, timings_all,
        n_conds_per_config=N)

    # models (mode_path pooling). Cosine-space models (dsr, stat, loc, phas,
    # midn) use the standard (n_features, n_tasks*N) activation + cosine RDM.
    # dsr_fmri has a different feature space (12 × 144 integer roll matrix
    # per task) and uses Hamming, so we build its display activation + RDM
    # separately below.
    model_cols = _build_dsr_model_cols(locations_all, no_phase_neurons,
                                       'mode_path', N, binlen)
    fmri_cols = model_cols.pop('dsr_fmri')     # list of (12, 144) integer matrices
    model_activations = {k: np.hstack(v) for k, v in model_cols.items()}

    # dsr_fmri activation (144, n_tasks*N): stack per-task (12,144) transposes
    # horizontally so the "conditions" axis (12 per task) sits horizontally
    # and the "144 sequence positions" run vertically, matching the layout
    # convention used for the other models.
    model_activations['dsr_fmri'] = np.hstack([m.T for m in fmri_cols])

    # model RDMs (full square)
    def _full_rdm(matrix):
        X = matrix.T.astype(float)
        X = X - X.mean(axis=1, keepdims=True)
        denom = np.sqrt(np.einsum('ij,ij->i', X, X))
        denom[denom == 0] = 1
        X = X / denom[:, None]
        return 1 - X @ X.T

    model_rdms = {k: _full_rdm(model_activations[k])
                   for k in model_activations if k != 'dsr_fmri'}
    # dsr_fmri RDM: full-square (n_tasks*N, n_tasks*N) Hamming.
    fmri_stack = np.concatenate(fmri_cols, axis=0)   # (n_tasks*N, 144) int
    M = fmri_stack.shape[0]
    ham_full = np.zeros((M, M), dtype=float)
    for i in range(M):
        ham_full[i, i+1:] = np.mean(fmri_stack[i][None, :] != fmri_stack[i+1:], axis=1)
    ham_full = ham_full + ham_full.T
    model_rdms['dsr_fmri'] = ham_full

    return data_z, data_rdm, model_activations, model_rdms


def dsr_across_halves_matrices(task_configs, locations_all, neurons, timings_all,
                               session_ids=None, n_conds_per_config=12,
                               no_phase_neurons=3):
    """For one recday's POST-CLEAN PRE-POOL normalised data, return the matrices
    used by the across-task-halves figures.

    Returns ``(data_activation, data_rdm, model_activations, model_rdms, K)``
    where the activations are 2*K*N columns wide (half-1 of every qualifying
    task, then half-2) and the RDMs are reconstructed K*N × K*N symmetric
    cross-half matrices. Returns ``None`` if no qualifying configs.
    """
    N = n_conds_per_config
    binlen = 360 // N
    if session_ids is None:
        session_ids = list(range(len(task_configs)))

    qualifying = split_sessions_into_halves(task_configs, session_ids, timings_all)
    if not qualifying:
        return None
    K = len(qualifying)
    M = K * N
    triu = np.triu_indices(M, k=1)

    def _half_neural(half):
        cols = []
        for q in qualifying:
            idxs = q[f'half{half}_indices']
            neu_concat = np.concatenate([neurons[i] for i in idxs], axis=1)
            avg = np.nanmean(neu_concat, axis=1)
            cols.append(avg.reshape(avg.shape[0], N, binlen).mean(axis=2))
        return np.hstack(cols)

    h1_neu, h2_neu = _half_neural(1), _half_neural(2)
    data_activation = np.hstack([h1_neu, h2_neu])
    mat = data_activation.T
    mu = mat.mean(axis=0); sd = mat.std(axis=0); sd[sd == 0] = 1
    mat_z = (mat - mu) / sd
    data_rdm_vec = my_RSA.compute_crosscorr(
        mat_z, plotting=False, include_diagonal=False, no_tasks=K)[0]
    data_rdm = np.zeros((M, M)); data_rdm[triu] = data_rdm_vec
    data_rdm = data_rdm + data_rdm.T

    def _half_models(half):
        out = {'dsr': [], 'stat': [], 'loc': [], 'phas': [], 'midn': [],
                'dsr_fmri': []}
        for q in qualifying:
            idxs = q[f'half{half}_indices']
            loc_concat = np.concatenate([locations_all[i] for i in idxs], axis=0)
            walked = _mode_path_360(loc_concat)
            loc_m, phas_m, stat_m, midn_m, dsr_m, _, _ = predictions.model_DSR(
                locations=walked, no_phase_neurons=no_phase_neurons)
            for key, MM in [('dsr', dsr_m), ('stat', stat_m), ('loc', loc_m),
                            ('phas', phas_m), ('midn', midn_m)]:
                out[key].append(MM.reshape(MM.shape[0], N, binlen).mean(axis=2))
            out['dsr_fmri'].append(_build_dsr_fmri_task(walked, N))
        # Cosine-space keys collapse via hstack (n_features, K*N).
        collapsed = {k: np.hstack(v) for k, v in out.items() if k != 'dsr_fmri'}
        # dsr_fmri collapses via row-stack (K*N, 144 integer node IDs).
        collapsed['dsr_fmri'] = np.concatenate(out['dsr_fmri'], axis=0)
        return collapsed

    h1_models, h2_models = _half_models(1), _half_models(2)
    model_activations = {}
    for k in h1_models:
        if k == 'dsr_fmri':
            # Display activation: transpose per-half to (144, K*N) then hstack
            model_activations[k] = np.hstack([h1_models[k].T, h2_models[k].T])
        else:
            model_activations[k] = np.hstack([h1_models[k], h2_models[k]])

    model_rdms = {}
    for k in h1_models:
        if k == 'dsr_fmri':
            # dsr_fmri: pairwise Hamming between half-1 rows and half-2 rows,
            # then symmetrise. Returns a (K*N × K*N) across-halves cross-block
            # matching the cosine-space models' shape convention. Within-half
            # blocks are NOT included — same "REMOVE half of the matrix"
            # invariant as compute_crosscorr for the other models.
            h1 = h1_models[k]; h2 = h2_models[k]     # (K*N, 144) each
            # Pairwise Hamming: mean fraction of positions where h1[i] != h2[j].
            H12 = np.mean(h1[:, None, :] != h2[None, :, :], axis=-1)  # (K*N, K*N)
            model_rdms[k] = 0.5 * (H12 + H12.T)
            continue
        m_combined = np.vstack([h1_models[k].T, h2_models[k].T])
        rdm_vec = my_RSA.compute_crosscorr(
            m_combined, plotting=False, include_diagonal=False, no_tasks=K)[0]
        rdm_mat = np.zeros((M, M)); rdm_mat[triu] = rdm_vec
        model_rdms[k] = rdm_mat + rdm_mat.T

    return data_activation, data_rdm, model_activations, model_rdms, K


def dsr_activation_and_rdm(task_configs, locations_all, neurons, timings_all,
                           n_conds_per_config=12):
    """For an example recday: build the z-scored activation matrix and the
    full (square) data RDM that the DSR pipeline operates on. Returned matrices
    are used by ``pub_figure_validation``.
    """
    N = n_conds_per_config
    binlen = 360 // N
    no_tasks = len(task_configs)
    data_cols = []
    for task_no in range(no_tasks):
        avg = np.nanmean(neurons[task_no], axis=1)
        data_cols.append(avg.reshape(avg.shape[0], N, binlen).mean(axis=2))
    mat_all = np.hstack(data_cols)
    mu = np.nanmean(mat_all, axis=1)
    sd = np.nanstd(mat_all, axis=1)
    sd[sd == 0] = 1
    mat_all_z = (mat_all - mu[:, None]) / sd[:, None]    # (n_neurons, K*N)

    # full square RDM
    X = (mat_all_z.T - mat_all_z.T.mean(axis=1, keepdims=True))
    X /= np.sqrt(np.einsum('ij,ij->i', X, X))[:, None]
    rdm_full = 1 - X @ X.T                                # (K*N, K*N)
    return mat_all_z, rdm_full


def reg_across_tasks_DSR(task_configs, locations_all, neurons, timings_all, mouse_recday,
                         n_conds_per_config=12, no_phase_neurons=3,
                         pool_method='mode_path', plotting=False,
                         phase_residualise=None,
                         combo_order=None):
    """Across-task RSA in parallel with ``scripts/RSA_DSR_ROIs_simple.py``.

    Uses the *normalised* recordings (360 bins/trial). Neural data is averaged
    across trials per task, downsampled to ``n_conds_per_config`` conditions
    and z-scored per neuron. Model RDMs are built from ``model_DSR`` using the
    pooling strategy chosen by ``pool_method`` (see ``_build_dsr_model_cols``).

    Returns three z-scored RSA variants:

        - ``across_z``: across-task/off-block pairs only.
        - ``within_z``: within-task off-diagonal pairs only.
        - ``full_z``:   all off-diagonal pairs, diagonal removed only.
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
        neu = _phase_residualise_task(neu, phase_residualise)
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

    # --- model RDMs from model_DSR, with chosen pool method ------------------
    model_cols = _build_dsr_model_cols(locations_all, no_phase_neurons,
                                       pool_method, N, binlen)

    model_RDM = {'across_z': {}, 'within_z': {}, 'full_z': {}}
    HAMMING_KEYS = {'dsr_fmri'}    # Hamming instead of cosine RDM
    for key, cols in model_cols.items():
        if key in HAMMING_KEYS:
            # dsr_fmri columns are (12 × 144) integer roll matrices per task;
            # stack row-wise to (n_configs*12, 144) and use Hamming.
            concat = np.concatenate(cols, axis=0)   # (n_configs*N, 144)
            within, across, full = my_RSA.compute_hamming_distance_within(
                concat, plotting=False, include_diagonal=False,
                no_tasks=no_tasks, model_name=key, block_size=N)
        else:
            concat = np.concatenate(cols, axis=1).T     # (n_configs*N, features)
            within, across, full = my_RSA.compute_crosscorr_within(
                concat, plotting=False, include_diagonal=False,
                no_tasks=no_tasks, model=key, block_size=N)
        model_RDM['across_z'][key] = across[0]
        model_RDM['within_z'][key] = within[0]
        model_RDM['full_z'][key] = _upper_no_diag(full)

    # --- one GLM with the caller-specified combo -----------------------
    # Default = the full 6-model joint (both DSR variants + 4 controls). Pass
    # combo_order to swap in a different model set — e.g.
    # ['dsr_fmri', 'stat', 'loc', 'phas'] to fit only dsr_fmri against 3
    # controls (Pipeline B — no dsr, no midn).
    order = combo_order or ['dsr', 'dsr_fmri', 'stat', 'loc', 'phas', 'midn']
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
        'pool_method': pool_method,
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


# ---------------------------------------------------------------------------
# publication figures for the chosen key analysis (DSR mode_path full_z)
# ---------------------------------------------------------------------------
def _benjamini_hochberg(pvals):
    """BH-FDR adjusted p-values (q-values). NaN inputs stay NaN."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q_ok = np.empty(n)
    q_ok[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = q_ok
    return q


# Categorical colours: DSR is highlighted in light blue, others in dark red
# (matches the original ``plotting_hist_scat`` scheme).
DSR_COLOUR     = '#96C5D8'
OTHER_COLOUR   = '#882048'


def _midnight_cmap():
    """Sequential cmap from white to dark midnight blue (era_brewer Midnight2)."""
    try:
        import era_brewer
        from matplotlib.colors import LinearSegmentedColormap
        cols = era_brewer.era_brew('Midnight2')
        return LinearSegmentedColormap.from_list(
            'Midnight2_seq',
            ['#FFFFFF', cols[7], cols[1], cols[5]], N=256)   # white -> mid -> dark navy
    except Exception:
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list(
            'Midnight_fallback', ['#FFFFFF', '#7191A9', '#2B4159', '#202D3C'], N=256)


def _model_palette(model_order):
    """Per-model categorical colours; DSR highlighted, the rest in dark red."""
    return [DSR_COLOUR if m in ('dsr', 'clo_model', 'dsr_fmri') else OTHER_COLOUR
            for m in model_order]


def _save_fig(fig, save_stem):
    """Save fig as both ``<stem>.pdf`` (vector) and ``<stem>.jpg`` (preview)."""
    if not save_stem:
        return
    fig.savefig(save_stem + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(save_stem + '.jpg', dpi=200, bbox_inches='tight')


def methods_results_stats(per_recday_results, n_neurons_per_recday,
                          n_tasks_per_recday, model_label_order=None,
                          alpha_fdr=0.05):
    """Build a methods/results-section-ready stats dict for one analysis.

    ``per_recday_results``: ``{recday: {coefs, t_vals, p_vals, label_regs}}``
    ``n_neurons_per_recday`` / ``n_tasks_per_recday``: dicts keyed by recday.
    ``model_label_order``: optional list of model keys controlling output order.
    """
    recdays = list(per_recday_results)
    if not recdays:
        return {'n_recdays': 0}
    label_regs = list(next(iter(per_recday_results.values()))['label_regs'])
    if model_label_order is None:
        model_label_order = label_regs
    #  pdb; pdb.set_trace()
    out = {
        'pipeline':           'DSR mode_path / all_trials / full_z',
        'n_recdays':          len(recdays),
        'recdays':            recdays,
        'n_neurons':          {rd: int(n_neurons_per_recday[rd]) for rd in recdays},
        'n_tasks':            {rd: int(n_tasks_per_recday[rd]) for rd in recdays},
        'n_neurons_summary':  {
            'mean':   float(np.mean(list(n_neurons_per_recday.values()))),
            'median': float(np.median(list(n_neurons_per_recday.values()))),
            'min':    int(min(n_neurons_per_recday.values())),
            'max':    int(max(n_neurons_per_recday.values())),
        },
        'n_tasks_summary': {
            'mean':   float(np.mean(list(n_tasks_per_recday.values()))),
            'median': float(np.median(list(n_tasks_per_recday.values()))),
            'min':    int(min(n_tasks_per_recday.values())),
            'max':    int(max(n_tasks_per_recday.values())),
        },
        'models': {},
        'fdr_alpha': float(alpha_fdr),
    }

    # collect group p-values per model first so we can BH-correct across models
    group_pvals = {}
    for m in model_label_order:
        mi = label_regs.index(m)
        coefs = np.asarray([per_recday_results[rd]['coefs'][mi] for rd in recdays],
                           dtype=float)
        coefs = coefs[np.isfinite(coefs)]
        if coefs.size > 1:
            t_grp, p_grp = ttest_1samp(coefs, 0, alternative='greater')
        else:
            t_grp, p_grp = np.nan, np.nan
        out['models'][m] = {
            'label': MODEL_LABELS.get(m, m),
            'coefs_by_recday':  {rd: float(per_recday_results[rd]['coefs'][mi]) for rd in recdays},
            'mean':   float(np.nanmean(coefs)),
            'sd':     float(np.nanstd(coefs, ddof=1)) if coefs.size > 1 else None,
            'sem':    float(np.nanstd(coefs, ddof=1) / np.sqrt(coefs.size)) if coefs.size > 1 else None,
            'median': float(np.nanmedian(coefs)),
            'min':    float(np.nanmin(coefs)),
            'max':    float(np.nanmax(coefs)),
            'n':      int(coefs.size),
            't_group':            float(t_grp) if np.isfinite(t_grp) else None,
            'p_group_uncorrected':float(p_grp) if np.isfinite(p_grp) else None,
        }
        group_pvals[m] = p_grp

    # BH-FDR across the model_label_order family
    pvec = np.asarray([group_pvals[m] for m in model_label_order])
    qvec = _benjamini_hochberg(pvec)
    for m, q in zip(model_label_order, qvec):
        out['models'][m]['p_group_fdr'] = float(q) if np.isfinite(q) else None
        out['models'][m]['sig_fdr']     = bool(q < alpha_fdr) if np.isfinite(q) else False

    return out


def _draw_betas_box_panel(ax, coefs_by_model, model_order,
                          fdr_pvals=None, font_size=11):
    """Box + scatter of per-recday betas with FDR-corrected stars placed
    on the OPPOSITE side of zero from the data (so they never overlap the
    box/scatter).

    ``star_y_below`` is used when the data are above zero (stars go below);
    ``star_y_above`` is used when the data are below zero (stars go above).
    """
    palette = _model_palette(model_order)
    box_data = [np.asarray(coefs_by_model[m], dtype=float) for m in model_order]
    box_data = [d[np.isfinite(d)] for d in box_data]
    n_models = len(model_order)
    positions = np.arange(n_models) + 1

    bp = ax.boxplot(box_data, positions=positions, widths=0.5,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.2))
    for patch, c in zip(bp['boxes'], palette):
        patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor('black')
    for i, (d, c) in enumerate(zip(box_data, palette)):
        jitter = 0.07 * np.random.randn(d.size)
        ax.scatter(positions[i] + jitter, d, s=40, color=c,
                   edgecolor='black', linewidth=0.6, zorder=3)

    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in model_order],
                       rotation=20, ha='right')
    ax.set_ylabel('β (z-scored)', fontsize=font_size)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


    ymin, ymax = ax.get_ylim()
    scale = ymax -ymin 
    star_y_below = - (scale/10)  
    if fdr_pvals is not None:
        for i, (m, d) in enumerate(zip(model_order, box_data)):
            p = fdr_pvals.get(m)
            if p is None or not np.isfinite(p):
                continue
            sig = ('***' if p < 0.001 else '**' if p < 0.01
                   else '*' if p < 0.05 else '')
            if not sig:
                continue
            min_val = float(np.nanmin(d)) if d.size else 0.0
            if min_val >= 0:
                y_star, va = star_y_below, 'top'
                # y_star, va = star_y_below, 'top'
            if min_val <= 0:
                y_star, va = min_val - (scale/10), 'top'
                star_y_below = y_star

            ax.text(positions[i], y_star, sig, ha='center', va=va,
                    fontsize=font_size + 4)

    ax.set_ylim(min(ymin, star_y_below - (scale/10)  ))


_PUB_RC = {
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size':       11,
    'axes.labelsize':  11,
    'axes.titlesize':  12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
}


def _ytick_labels_for_model(name, n_rows, no_phase_neurons=3):
    """Per-model y-tick labels + optional row-group separators (semantic, not
    numeric). Returns ``(positions, labels, separator_rows)`` or
    ``(None, None, None)`` if no semantic labelling is appropriate.

    ``name`` is one of ``'loc'``, ``'stat'``, ``'phas'``, ``'phas_stat'``,
    ``'midn'``, ``'dsr'`` (full clo_model), or ``'data'``.
    """
    phase_short  = (['early', 'mid', 'late'] if no_phase_neurons == 3
                    else [f'p{i+1}' for i in range(no_phase_neurons)])
    state_letter = ['A', 'B', 'C', 'D']

    if name == 'loc':
        return list(range(n_rows)), [f'Loc {i+1}' for i in range(n_rows)], None

    if name == 'stat':
        return (list(range(n_rows)),
                [f'State {state_letter[i]}' for i in range(n_rows)], None)

    if name == 'phas':
        return list(range(n_rows)), phase_short[:n_rows], None

    if name == 'phas_stat':
        # one DSR module — 4 states × no_phase_neurons phase = lags 0..(n-1)
        labels = [f'lag {i}' for i in range(n_rows)]
        # separators between state blocks
        seps = [i * no_phase_neurons for i in range(1, 4)]
        return list(range(n_rows)), labels, seps

    if name == 'midn':
        # 9 locations × no_phase_neurons phase neurons.
        positions, labels = [], []
        n_loc = n_rows // no_phase_neurons
        for loc in range(n_loc):
            for pi in range(no_phase_neurons):
                positions.append(loc * no_phase_neurons + pi)
                labels.append(f'L{loc+1}-{phase_short[pi]}')
        seps = [loc * no_phase_neurons for loc in range(1, n_loc)]
        return positions, labels, seps

    if name == 'dsr':
        # Full clo_model: 9 locations × no_phase_neurons phase × (4 × no_phase_neurons) lag
        n_lags = 4 * no_phase_neurons
        n_groups = n_rows // n_lags                # = n_loc * no_phase_neurons
        n_loc = n_groups // no_phase_neurons
        positions, labels = [], []
        # tick at the middle of each (location, phase) group of n_lags rows
        for loc in range(n_loc):
            for pi in range(no_phase_neurons):
                g = loc * no_phase_neurons + pi
                positions.append(g * n_lags + n_lags / 2)
                labels.append(f'L{loc+1}-{phase_short[pi]}')
        # separators between locations (thicker visually applied later)
        seps = [loc * no_phase_neurons * n_lags for loc in range(1, n_loc)]
        return positions, labels, seps

    return None, None, None


def _set_nested_task_x_axis(ax, K, N, x_axis_groups=None, *,
                            rotation=45, ha='right', tick_fs=None,
                            outer_fs=None, outer_y_bar=-0.30,
                            outer_y_text=-0.42, draw_outer=True):
    """Inner ``task N`` x-tick labels (restarting per group) + optional outer
    group brackets/labels below.

    ``x_axis_groups``: ``[(label, n_in_group), ...]`` summing to ``K``.
    Outer labels are drawn with a small horizontal bracket below the inner ticks.
    """
    inner_pos = np.arange(K) * N + N / 2
    if x_axis_groups:
        inner_labels = []
        for _, n_in_grp in x_axis_groups:
            inner_labels += [f"task {i+1}" for i in range(n_in_grp)]
    else:
        inner_labels = [f"task {k+1}" for k in range(K)]
    ax.set_xticks(inner_pos)
    if tick_fs is not None:
        ax.set_xticklabels(inner_labels, rotation=rotation, ha=ha, fontsize=tick_fs)
    else:
        ax.set_xticklabels(inner_labels, rotation=rotation, ha=ha)
    if not x_axis_groups or not draw_outer:
        return
    xtrans = ax.get_xaxis_transform()
    offsets = np.cumsum([0] + [n for _, n in x_axis_groups[:-1]])
    for (label, n_in_grp), off in zip(x_axis_groups, offsets):
        x0 = off * N - 0.5
        x1 = (off + n_in_grp) * N - 0.5
        # bracket
        ax.plot([x0, x1], [outer_y_bar, outer_y_bar], color='black', lw=0.8,
                transform=xtrans, clip_on=False)
        for xe in (x0, x1):
            ax.plot([xe, xe], [outer_y_bar, outer_y_bar - 0.04], color='black',
                    lw=0.8, transform=xtrans, clip_on=False)
        ax.text((x0 + x1) / 2, outer_y_text, label, ha='center', va='top',
                transform=xtrans, clip_on=False,
                fontsize=(outer_fs or plt.rcParams['axes.labelsize']))


def _set_nested_task_y_axis(ax, K, N, x_axis_groups=None, *,
                            tick_fs=None, outer_fs=None,
                            outer_x_bar=-0.14, outer_x_text=-0.22,
                            draw_outer=True):
    """Same as ``_set_nested_task_x_axis`` but applied to the (square-RDM)
    y-axis. Inner labels are vertical-friendly; outer label is rotated 90°.
    """
    inner_pos = np.arange(K) * N + N / 2
    if x_axis_groups:
        inner_labels = []
        for _, n_in_grp in x_axis_groups:
            inner_labels += [f"task {i+1}" for i in range(n_in_grp)]
    else:
        inner_labels = [f"task {k+1}" for k in range(K)]
    ax.set_yticks(inner_pos)
    if tick_fs is not None:
        ax.set_yticklabels(inner_labels, fontsize=tick_fs)
    else:
        ax.set_yticklabels(inner_labels)
    if not x_axis_groups or not draw_outer:
        return
    ytrans = ax.get_yaxis_transform()
    offsets = np.cumsum([0] + [n for _, n in x_axis_groups[:-1]])
    for (label, n_in_grp), off in zip(x_axis_groups, offsets):
        y0 = off * N - 0.5
        y1 = (off + n_in_grp) * N - 0.5
        ax.plot([outer_x_bar, outer_x_bar], [y0, y1], color='black', lw=0.8,
                transform=ytrans, clip_on=False)
        for ye in (y0, y1):
            ax.plot([outer_x_bar, outer_x_bar + 0.04], [ye, ye], color='black',
                    lw=0.8, transform=ytrans, clip_on=False)
        ax.text(outer_x_text, (y0 + y1) / 2, label, ha='right', va='center',
                rotation=90, transform=ytrans, clip_on=False,
                fontsize=(outer_fs or plt.rcParams['axes.labelsize']))


def _apply_nested_y_dsr_midn(ax, kind, n_rows, no_phase_neurons=3, *,
                             inner_fs=None, outer_fs=None,
                             outer_x_text=-0.18, sep_color='white'):
    """Nested y-axis: phase letter (inner: E/M/L) + location number (outer).
    For ``'midn'`` (27 rows) and ``'dsr'`` (324 rows, the full clo_model).

    Also draws thin row separators between locations.
    """
    phase_short = (['E', 'M', 'L'] if no_phase_neurons == 3
                   else [f'p{i+1}' for i in range(no_phase_neurons)])

    if kind == 'midn':
        n_loc = n_rows // no_phase_neurons
        inner_pos = list(range(n_rows))
        inner_labels = [phase_short[i % no_phase_neurons] for i in range(n_rows)]
        outer_pos = [loc * no_phase_neurons + (no_phase_neurons - 1) / 2
                     for loc in range(n_loc)]
        loc_seps  = [loc * no_phase_neurons - 0.5 for loc in range(1, n_loc)]

    elif kind == 'dsr':
        n_lags = 4 * no_phase_neurons
        n_loc = (n_rows // n_lags) // no_phase_neurons
        inner_pos, inner_labels = [], []
        for loc in range(n_loc):
            for pi in range(no_phase_neurons):
                g = loc * no_phase_neurons + pi
                inner_pos.append(g * n_lags + n_lags / 2)
                inner_labels.append(phase_short[pi])
        outer_pos = [loc * no_phase_neurons * n_lags
                     + (no_phase_neurons * n_lags) / 2
                     for loc in range(n_loc)]
        loc_seps  = [loc * no_phase_neurons * n_lags - 0.5
                     for loc in range(1, n_loc)]
    else:
        return

    ax.set_yticks(inner_pos)
    if inner_fs is not None:
        ax.set_yticklabels(inner_labels, fontsize=inner_fs)
    else:
        ax.set_yticklabels(inner_labels)

    ytrans = ax.get_yaxis_transform()
    for pos, loc_idx in zip(outer_pos, range(1, len(outer_pos) + 1)):
        ax.text(outer_x_text, pos, str(loc_idx), ha='right', va='center',
                transform=ytrans, clip_on=False,
                fontsize=(outer_fs or plt.rcParams['axes.labelsize']))
    for row in loc_seps:
        ax.axhline(row, color='grey', lw=0.5, alpha=0.7)


def _imshow_with_task_grid(ax, M, K, N, *, cmap, vmin=None, vmax=None,
                           aspect='auto', square=False, gridcolor='white',
                           model_kind=None, no_phase_neurons=3,
                           ytick_fontsize=None, x_axis_groups=None,
                           draw_outer_x=True, draw_outer_y=True):
    """imshow with task-boundary lines + (nested) tick labels.

    ``x_axis_groups``: optional ``[(label, n_in_group), ...]`` to draw outer
    brackets below the inner ``task N`` labels (and to the left of the y-axis
    on square RDMs). Vertical/horizontal dividers at group boundaries are
    drawn thicker than inner task dividers.
    """
    im = ax.imshow(M, aspect=('equal' if square else aspect), cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    group_bounds = set()
    if x_axis_groups:
        cum = 0
        for _, n_in_grp in x_axis_groups[:-1]:
            cum += n_in_grp
            group_bounds.add(cum)

    for k in range(1, K):
        lw = 1.4 if k in group_bounds else 0.8
        ax.axvline(k * N - 0.5, color=gridcolor, lw=lw)
        if square:
            ax.axhline(k * N - 0.5, color=gridcolor, lw=lw)

    _set_nested_task_x_axis(ax, K, N, x_axis_groups=x_axis_groups,
                            tick_fs=ytick_fontsize, draw_outer=draw_outer_x)

    if square:
        _set_nested_task_y_axis(ax, K, N, x_axis_groups=x_axis_groups,
                                tick_fs=ytick_fontsize, draw_outer=draw_outer_y)
        return im

    if model_kind in ('dsr', 'midn'):
        _apply_nested_y_dsr_midn(ax, model_kind, M.shape[0],
                                 no_phase_neurons=no_phase_neurons,
                                 inner_fs=ytick_fontsize)
    elif model_kind is not None:
        positions, labels, seps = _ytick_labels_for_model(
            model_kind, M.shape[0], no_phase_neurons=no_phase_neurons)
        if positions is not None:
            ax.set_yticks(positions)
            if ytick_fontsize is not None:
                ax.set_yticklabels(labels, fontsize=ytick_fontsize)
            else:
                ax.set_yticklabels(labels)
        if seps:
            for row in seps:
                ax.axhline(row - 0.5, color='black', lw=0.5, alpha=0.7)
    return im


# def pub_figure_dsr_overview(dsr_model_activation, dsr_model_rdm,
#                             coefs_by_model, model_order, fdr_pvals,
#                             n_tasks, n_conds_per_task=12, recday_label='',
#                             save_stem=None, font_size=11,
#                             width_fracs=(0.4, 0.3, 0.3), height_in=3.6,
#                             x_axis_groups=None):
def pub_figure_dsr_overview(dsr_model_activation, dsr_model_rdm,
                            coefs_by_model, model_order, fdr_pvals,
                            n_tasks, n_conds_per_task=12, recday_label='',
                            save_stem=None, font_size=11,
                            width_fracs=(0.3, 0.3, 0.4), height_in=1.9,
                            x_axis_groups=None):
    
    """Main publication figure (3 panels):
        a) DSR modelled neurons — activation across all tasks (single subject's
           binning grid; n_features × K*N).
        b) DSR model RDM — full square, lower triangle drawn.
        c) Betas across recdays for the four models, FDR-corrected stars.

    ``width_fracs`` is the fraction of usable A4 page width each subpanel will
    occupy in the final printed document — saved figsize is set so the file
    can be dropped in at 100% and the ``font_size`` renders correctly.
    """
    K, N = n_tasks, n_conds_per_task
    seq_cmap = _midnight_cmap()
    layout = figure_layout.subpanel_figure(
        width_fracs=width_fracs, height_in=height_in,
        target_font_pt=font_size)
    with plt.rc_context(layout['rc']):
        fig, axes = plt.subplots(
            1, 3, figsize=layout['figsize'],
            gridspec_kw={'width_ratios': layout['width_ratios']},
            constrained_layout=True)

        # (a) DSR modelled neurons (full clo_model: location × phase × lag)
        ax = axes[0]
        _imshow_with_task_grid(ax, dsr_model_activation, K, N, cmap=seq_cmap,
                               model_kind='dsr',
                               ytick_fontsize=max(font_size - 6, 4),
                               x_axis_groups=x_axis_groups)
        # Push the "simulated neurons" label past the nested loc/phase labels
        ax.set_ylabel('simulated neurons', labelpad=20)
        #ax.set_title(f'a) DSR modelled neurons — {recday_label}', loc='left')

        # (b) DSR model RDM (lower triangle, RdBu_r). For the across-halves
        # variant the activation matrix has 2*K_h*N columns but the RDM is
        # already collapsed by ``compute_crosscorr`` to a K_h*N square — so
        # infer the RDM's task count from its shape rather than reusing K.
        ax = axes[1]
        rdm_disp = dsr_model_rdm.copy()
        rdm_disp[np.triu_indices_from(rdm_disp, k=1)] = np.nan
        vmax = np.nanmax(np.abs(rdm_disp))
        K_rdm = rdm_disp.shape[0] // N
        im = _imshow_with_task_grid(ax, rdm_disp, K_rdm, N, cmap='RdBu_r',
                                    vmin=-vmax, vmax=vmax, square=True,
                                    gridcolor='white')
        #ax.set_title('b) DSR model RDM', loc='left')
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label='1 − r')

        # (c) betas with FDR stars
        ax = axes[2]
        _draw_betas_box_panel(ax, coefs_by_model, model_order,
                              fdr_pvals=fdr_pvals, font_size=font_size)
        #ax.set_title('c) betas across recdays (FDR *)', loc='left')

        _save_fig(fig, save_stem)
        plt.show()
        # import pdb; pdb.set_trace()
    return fig


def pub_figure_example_subject(data_activation, data_rdm,
                               model_activations, model_rdms, model_order,
                               n_tasks, n_conds_per_task=12, recday_label='',
                               save_stem=None, font_size=10,
                               total_width_frac=1.0, height_in=6.5,
                               page_width_in=figure_layout.A4_HEIGHT_IN,
                               x_axis_groups=None):
    """Single-subject supplementary figure: per-column layout.
    col 0 = recorded data; cols 1..n = each model. Top row = activation matrix
    (sequential cmap, no individual colorbars), bottom row = RDM (RdBu_r,
    shared symmetric scale, single colorbar at the right of the row).

    ``x_axis_groups``: optional outer grouping for nested x-tick labels (e.g.
    ``[('run 1', K_h), ('run 2', K_h)]`` for the across-halves variant).
    """
    K, N = n_tasks, n_conds_per_task
    seq_cmap = _midnight_cmap()
    columns = [('data', data_activation, data_rdm, None)] + [
        (MODEL_LABELS.get(m, m), model_activations[m], model_rdms[m],
         MODEL_KINDS.get(m))
        for m in model_order
    ]
    n_cols = len(columns)

    def _lower(rdm):
        r = rdm.copy()
        r[np.triu_indices_from(r, k=1)] = np.nan
        return r
    vmax = max(np.nanmax(np.abs(_lower(c[2]))) for c in columns if c[2] is not None)

    # Uniform activation / RDM row heights so RDMs align across columns.
    h_top = 1.0

    layout = figure_layout.subpanel_figure(
        width_fracs=[total_width_frac / n_cols] * n_cols,
        height_in=height_in, target_font_pt=font_size,
        page_width_in=page_width_in)
    with plt.rc_context(layout['rc']):
        fig = plt.figure(figsize=layout['figsize'])
        # 2 rows × (n_cols + 1) cells. Last column is the shared colorbar slot;
        # the activation (top) row leaves it empty.
        outer = fig.add_gridspec(
            2, n_cols + 1,
            height_ratios=[h_top, 1.0],
            width_ratios=[1.0] * n_cols + [0.04],
            wspace=0.55, hspace=0.45)

        rdm_axes = []
        for col_idx, (name, act, rdm, mkind) in enumerate(columns):
            # activation (top) — no colorbar, no y-label
            ax_act = fig.add_subplot(outer[0, col_idx])
            _imshow_with_task_grid(
                ax_act, act, K, N, cmap=seq_cmap, model_kind=mkind,
                ytick_fontsize=(max(font_size - 4, 4)
                                if mkind in ('dsr', 'midn') else None),
                x_axis_groups=x_axis_groups)
            ax_act.set_title(name, loc='left')

            # RDM (bottom) — square, no individual colorbar. For across-
            # halves the activation has 2*K_h*N columns but ``compute_crosscorr``
            # collapses the RDM to a K_h*N square — infer the RDM's task
            # count from its shape so we draw the right number of dividers.
            ax_rdm = fig.add_subplot(outer[1, col_idx])
            K_rdm = rdm.shape[0] // N
            _imshow_with_task_grid(ax_rdm, _lower(rdm), K_rdm, N,
                                   cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                                   square=True, gridcolor='black')
            ax_rdm.set_title('RDM', loc='left')
            rdm_axes.append(ax_rdm)

        # Single shared colorbar for the bottom RDM row only.
        cax = fig.add_subplot(outer[1, n_cols])
        fig.colorbar(rdm_axes[-1].images[0], cax=cax, label='1 − r')

        fig.suptitle(f'Example subject — {recday_label}',
                     fontsize=font_size + 2, y=0.995)

        _save_fig(fig, save_stem)
        plt.show()
    return fig


def pub_figure_model_schematics(walked_path, task_config=None,
                                no_phase_neurons=3, recday_label='',
                                save_stem=None, font_size=11,
                                total_width_frac=1.0, height_in=3.2):
    """One panel per model building block, for a single example task
    configuration. Panels (left → right):

        Physical Location, Location in Task, Subgoal Progress,
        Midnight model ("Now" DSR), DSR (full).

    The Midnight model encodes only currently-visited locations (loc × phase,
    no future-lag content). The full DSR adds the 4 × no_phase_neurons lag
    rows per (location, phase) — i.e. its row count is the Midnight model
    repeated once per lag.
    """
    import mc.simulation.predictions as predictions
    loc_m, phas_m, stat_m, midn_m, dsr_m, _phas_stat_m, _ = predictions.model_DSR(
        locations=walked_path, no_phase_neurons=no_phase_neurons)

    seq_cmap = _midnight_cmap()
    n_bins = loc_m.shape[1]
    bins_per_state = n_bins // 4
    n_lags = 4 * no_phase_neurons

    panels = [
        ('Physical\nLocation', loc_m, 'loc',
         '9 grid nodes\n(at-node binary)'),
        ('Location in\nTask',  stat_m, 'stat',
         '4 subgoals\n(active binary)'),
        ('Subgoal\nProgress',  phas_m, 'phas',
         f'{no_phase_neurons} phase neurons\ntiled / subgoal'),
        ('DSR\n(only current)', midn_m, 'midn',
         f'9 loc × {no_phase_neurons} phase\n= {9*no_phase_neurons} ("now"-only)'),
        ('DSR\n(1 module)',     _phas_stat_m, 'phas_stat',
         f'{n_lags} lags\n= {n_lags} neurons'),
        ('DSR (full)',          dsr_m, 'dsr',
         f'9 × {no_phase_neurons} × {n_lags} lags\n= {9*no_phase_neurons*n_lags} neurons'),
    ]

    cfg_str = (f'task {tuple(int(x) for x in task_config)}'
               if task_config is not None else '')

    layout = figure_layout.subpanel_figure(
        width_fracs=[total_width_frac / len(panels)] * len(panels),
        height_in=height_in, target_font_pt=font_size)
    with plt.rc_context(layout['rc']):
        fig, axes = plt.subplots(1, len(panels), figsize=layout['figsize'],
                                 constrained_layout=True)
        for ax, (name, M, kind, descr) in zip(axes, panels):
            ax.imshow(M, aspect='auto', cmap=seq_cmap, interpolation='nearest')
            ax.set_title(name, loc='left')
            # vertical subgoal dividers (4 states A..D)
            for k in range(1, 4):
                ax.axvline(k * bins_per_state - 0.5, color='black', lw=0.8)
            ax.set_xticks([(k + 0.5) * bins_per_state for k in range(4)])
            ax.set_xticklabels(['A', 'B', 'C', 'D'])
            ax.set_xlabel('subgoal\n\n' + descr,
                          fontsize=font_size - 1, linespacing=1.3)

            # semantic y-tick labels per model. For DSR/midnight, use nested
            # labels (inner phase letter, outer location number).
            if kind in ('dsr', 'midn'):
                _apply_nested_y_dsr_midn(
                    ax, kind, M.shape[0],
                    no_phase_neurons=no_phase_neurons,
                    inner_fs=max(font_size - 4, 5),
                    outer_fs=max(font_size - 2, 6),
                    outer_x_text=-0.25)
            else:
                positions, labels, seps = _ytick_labels_for_model(
                    kind, M.shape[0], no_phase_neurons=no_phase_neurons)
                if positions is not None:
                    ax.set_yticks(positions)
                    ax.set_yticklabels(labels, fontsize=max(font_size - 2, 6))
                if seps:
                    for row in seps:
                        ax.axhline(row - 0.5, color='white', lw=0.4, alpha=0.7)

        suptitle = ' — '.join(s for s in (cfg_str, recday_label) if s)
        if suptitle:
            fig.suptitle(suptitle, fontsize=font_size + 1)
        _save_fig(fig, save_stem)
        plt.show()
    return fig


def plot_betas_grid(panel_data, row_labels, col_labels, suptitle,
                    save_path=None, figsize_per_panel=(1.3, 1.3), font_size=8,
                    point_colour='#882048'):
    """Compact grid of box+scatter plots — one panel per (row_label, col_label).

    ``panel_data[row_label][col_label]`` must be a 1-D list/array of one
    beta-per-recday. Rows share a y-axis. One-sample t-tests vs zero (greater)
    are annotated as stars at the bottom of each panel.

    Font changes are wrapped in ``plt.rc_context`` so they don't leak globally.
    ``save_path`` is a full filename (e.g. ``overview.jpeg``); the extension
    determines the format. ``plt.show()`` is called; figures are NOT closed.
    """
    n_rows, n_cols = len(row_labels), len(col_labels)
    figsize = (figsize_per_panel[0] * n_cols + 1.4,
               figsize_per_panel[1] * n_rows + 0.8)

    with plt.rc_context({
        'font.size':       font_size,
        'axes.labelsize':  font_size,
        'axes.titlesize':  font_size,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'legend.fontsize': font_size - 1,
    }):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                                 sharey='row', constrained_layout=True)
        axes = np.atleast_2d(axes)
        if axes.shape[1] != n_cols:   # single-row corner case
            axes = axes.reshape(n_rows, n_cols)

        for r, row_label in enumerate(row_labels):
            # per-row symmetric y-limits
            row_vals = np.concatenate([
                np.asarray(panel_data[row_label].get(c, []), dtype=float).ravel()
                for c in col_labels])
            row_vals = row_vals[np.isfinite(row_vals)]
            ylim = 1.2 * np.max(np.abs(row_vals)) if row_vals.size else 1.0

            for c, col_label in enumerate(col_labels):
                ax = axes[r, c]
                vals = np.asarray(panel_data[row_label].get(col_label, []), dtype=float)
                vals = vals[np.isfinite(vals)]

                if vals.size:
                    ax.boxplot([vals], widths=0.55, showfliers=False,
                               medianprops=dict(color='black'))
                    jitter = 0.07 * np.random.randn(vals.size)
                    ax.scatter(1 + jitter, vals, s=10, color=point_colour,
                               edgecolor='black', linewidth=0.4, zorder=3)
                    if vals.size > 1:
                        _, p = ttest_1samp(vals, 0, alternative='greater')
                        sig = ('***' if p < 0.001
                               else '**' if p < 0.01
                               else '*' if p < 0.05 else '')
                        if sig:
                            ax.text(0.5, 0.02, sig, transform=ax.transAxes,
                                    ha='center', va='bottom',
                                    fontsize=font_size + 2)

                ax.axhline(0, color='gray', ls='--', lw=0.5)
                ax.set_xticks([])
                ax.set_ylim(-ylim, ylim)
                for spine in ('top', 'right'):
                    ax.spines[spine].set_visible(False)

                if r == 0:
                    ax.set_title(col_label, pad=2)
                if c == 0:
                    ax.set_ylabel(row_label, rotation=0, ha='right', va='center',
                                  labelpad=22)

        fig.suptitle(suptitle, fontsize=font_size + 2)

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.show()

    return fig


def plotting_hist_scat(data_list, label_string_list, label_tick_list, title_string, save_fig=False):
    """Boxplot + scatter of the per-mouse betas with one-sample t-test stars."""
    fig, ax = plt.subplots(figsize=(6, 8))
    # fig, ax = plt.subplots(figsize=(8, 6))
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
