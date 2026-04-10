#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_location_snippets.py

For each subject/neuron/grid_no/phase-episode visited:
  1. Average location vectors across correct repeats of the same grid_no run.
  2. Build a circular (wrapping) consensus: treat the 360-bin trial as a ring.
     Find stable location visits: >=CONSEC_MIN consecutive bins (wrapping)
     where the same location appears in majority of the repeats.
  3. Assign each visit as 'reward' or 'path':
       reward: the reward bin (89/179/269/359) falls within the visit duration,
               OR (location matches D reward location AND bin 0 is in visit = wrap-around D).
       path: everything else.
  4. Fuse the last and first visit if both are 'reward' and same slot (circular wrap).
  5. Group consecutive path visits into path-episodes; each reward visit is its own episode.
  6. For each episode, chop the neuron timecourse into a snippet:
       PRE_BINS before onset + POST_BINS from onset (onset at index PRE_BINS). Wraps.
       Neuron is averaged across the same correct repeats.
  7. Annotate with 7 lookahead phase-episodes (wrapping):
       next+k_loc  : list of locations (path) or single location (reward)
       next+k_slot : slot (A/B/C/D)
       next+k_phase: 'reward' or 'path'

Output CSV columns:
  session, neuron_label, roi, config, grid_no, n_repeats,
  current_location (list for path, int for reward), onset_bin, run_length, slot, phase,
  next+1_loc, next+1_slot, next+1_phase, ..., next+7_loc, next+7_slot, next+7_phase,
  bin_00 ... bin_29  (snippet; onset at bin_10)
"""

import os
import numpy as np
import pandas as pd
import mc
from collections import defaultdict

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
DATA_FOLDER  = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"

PRE_BINS     = 10    # bins before onset in snippet
POST_BINS    = 20    # bins from onset in snippet
SNIPPET_LEN  = PRE_BINS + POST_BINS   # 30 total, onset at index PRE_BINS

CONSEC_MIN   = 5    # min consecutive bins to call a stable location visit

N_LOOKAHEAD  = 7    # number of future phase-episodes to annotate

REWARD_BINS  = {'A': 89, 'B': 179, 'C': 269, 'D': 359}
SLOTS        = ['A', 'B', 'C', 'D']

SNIPPET_COLS = [f'bin_{b:02d}' for b in range(SNIPPET_LEN)]
FIXED_COLS   = (
    ['session', 'neuron_label', 'roi', 'config', 'grid_no', 'n_repeats',
     'current_location', 'onset_bin', 'run_length', 'slot', 'phase']
    + [c for k in range(1, N_LOOKAHEAD + 1)
       for c in (f'next+{k}_loc', f'next+{k}_slot', f'next+{k}_phase')]
    + SNIPPET_COLS
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def assign_visit_phase(onset, run_length, cfg_locs, current_loc):
    """
    Returns (slot, phase) where phase is 'reward' or 'path'.

    Reward: any reward bin (89/179/269/359) falls within the visit duration,
            OR the visit wraps around bin 0 at the D reward location
            (carry-over from the end of the circular trial).
    Path: everything else.

    Slot: for reward visits, the slot whose reward bin is in the visit.
          For path visits, the next upcoming reward bin from onset.
    """
    N = 360
    visit_bins = set((onset + d) % N for d in range(run_length))

    # Check if any reward bin falls inside the visit
    for slot, rb in REWARD_BINS.items():
        if rb in visit_bins:
            return slot, 'reward'

    # Wrap-around carry-over for D: visit contains bin 0 and subject is at D location
    if 0 in visit_bins and current_loc == cfg_locs[3]:   # cfg_locs[3] = D location
        return 'D', 'reward'

    # Path: find next upcoming reward bin from onset
    min_dist = N + 1
    next_slot = None
    for slot, rb in REWARD_BINS.items():
        dist = (rb - onset) % N
        if dist < min_dist:
            min_dist = dist
            next_slot = slot
    return next_slot, 'path'


def consensus_location(loc_matrix):
    """
    loc_matrix: (n_reps, 360) int array
    Returns: (360,) float — majority location at each bin, NaN if no majority.
    """
    n_reps, n_bins = loc_matrix.shape
    consensus = np.full(n_bins, np.nan)
    for b in range(n_bins):
        col = loc_matrix[:, b]
        vals, counts = np.unique(col, return_counts=True)
        best_idx = np.argmax(counts)
        consensus[b] = vals[best_idx]
    return consensus


def find_stable_onsets_circular(consensus_locs):
    """
    Find stable location visits in the circular (wrapping) consensus vector.
    Returns list of (onset_bin, location, run_length) sorted by onset_bin.
    Only visits with run_length >= CONSEC_MIN are included.
    """
    n = len(consensus_locs)
    circ = np.concatenate([consensus_locs, consensus_locs])

    runs = []
    b = 0
    while b < 2 * n:
        loc = circ[b]
        if np.isnan(loc):
            b += 1
            continue
        end = b + 1
        while end < 2 * n and circ[end] == loc:
            end += 1
        runs.append((b, end - b, int(loc)))
        b = end

    onsets = []
    seen_onsets = set()
    for start, length, loc in runs:
        if length < CONSEC_MIN:
            continue
        onset = start % n
        if onset in seen_onsets:
            continue
        seen_onsets.add(onset)
        onsets.append((onset, loc, min(length, n)))   # cap at 360

    onsets.sort(key=lambda x: x[0])
    return onsets


def fuse_circular_reward(visit_annotations):
    """
    If the first and last visit are both 'reward' with the same slot,
    they are the same circular reward episode split by the 360-bin boundary.
    Fuse them: the last visit becomes the canonical one with combined run_length,
    onset = last onset, run_length = last_run + first_run (wrapping).
    The first entry is removed.

    visit_annotations: list of (onset_bin, loc, run_length, slot, phase)
    Returns modified list.
    """
    if len(visit_annotations) < 2:
        return visit_annotations

    first = visit_annotations[0]
    last  = visit_annotations[-1]

    if first[4] == 'reward' and last[4] == 'reward' and first[3] == last[3]:
        # fuse: last onset stays, run_length = last_run + first_run
        fused_run = last[2] + first[2]
        fused = (last[0], last[1], fused_run, last[3], last[4])
        return [fused] + visit_annotations[1:-1]

    return visit_annotations


def build_phase_episodes(visit_annotations):
    """
    Group consecutive 'path' visits into path-episodes.
    Each 'reward' visit is its own episode.

    Returns list of episodes, each:
      (onset_bin, run_length, slot, phase, locs)
    where locs is a list of location(s):
      - reward: [loc]
      - path: [loc1, loc2, ...] (all locations in the consecutive path visits)

    Episodes are in temporal order.
    """
    episodes = []
    i = 0
    while i < len(visit_annotations):
        onset_bin, loc, run_length, slot, phase = visit_annotations[i]
        if phase == 'reward':
            episodes.append((onset_bin, run_length, slot, phase, [loc]))
            i += 1
        else:
            # collect consecutive path visits
            path_locs = [loc]
            path_onset = onset_bin
            path_run = run_length
            j = i + 1
            while j < len(visit_annotations) and visit_annotations[j][4] == 'path':
                path_locs.append(visit_annotations[j][1])
                path_run += visit_annotations[j][2]   # total span
                j += 1
            episodes.append((path_onset, path_run, slot, 'path', path_locs))
            i = j
    return episodes


def extract_snippet_circular(mean_tc, onset_bin):
    """
    Extract PRE_BINS + POST_BINS bins centred on onset_bin, wrapping around
    the 360-bin boundary. Returns array of length SNIPPET_LEN.
    """
    n = len(mean_tc)
    indices = [(onset_bin - PRE_BINS + i) % n for i in range(SNIPPET_LEN)]
    return mean_tc[indices]


# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------
all_sessions_rows = []

for sub in range(1, 64):
    sub_str = f"{sub:02d}"
    sub_key = f"sub-{sub_str}"

    configs_path = os.path.join(DATA_FOLDER, f"s{sub_str}", "cells_and_beh",
                                f"all_configs_sub{sub_str}.csv")
    if not os.path.exists(configs_path):
        print(f"s{sub_str}: configs file missing, skipping.")
        continue

    configs_arr = np.genfromtxt(configs_path, delimiter=',').astype(int)
    if configs_arr.ndim == 1:
        configs_arr = configs_arr.reshape(1, -1)
    grid_to_config = {i + 1: tuple(configs_arr[i]) for i in range(len(configs_arr))}

    data_dict, source_path = mc.analyse.helpers_human_cells.get_data(sub)
    if sub_key not in data_dict:
        print(f"s{sub_str}: data not found, skipping.")
        continue

    beh         = data_dict[sub_key]['beh']
    cell_labels = data_dict[sub_key]['cell_labels']
    norm_neurons = data_dict[sub_key]['normalised_neurons']   # {key: DataFrame (n_trials, 360)}
    locs         = data_dict[sub_key]['locations']            # DataFrame (n_trials, 360)

    neuron_keys = list(norm_neurons.keys())

    # group correct trial indices by grid_no
    gridno_to_trials = defaultdict(list)
    for trial_idx, row in beh.iterrows():
        gno = int(row['grid_no'])
        if gno not in grid_to_config:
            continue
        if row['correct'] == 1:
            gridno_to_trials[gno].append(trial_idx)

    if not gridno_to_trials:
        print(f"s{sub_str}: no correct trials found, skipping.")
        continue

    session_rows = []

    for gno, trial_indices in gridno_to_trials.items():
        cfg        = grid_to_config[gno]
        config_str = f"{cfg[0]}-{cfg[1]}-{cfg[2]}-{cfg[3]}"
        n_reps     = len(trial_indices)

        # (n_reps, 360) location matrix for this grid run
        loc_matrix = locs.iloc[trial_indices, :].values.astype(float)

        # consensus location vector (circular)
        consensus = consensus_location(loc_matrix)

        # find stable onsets (circular)
        onsets = find_stable_onsets_circular(consensus)
        if not onsets:
            continue

        cfg_locs = list(cfg)   # [loc_A, loc_B, loc_C, loc_D]

        # per-neuron mean timecourse averaged over correct repeats
        neuron_means = {}
        for n_idx, nkey in enumerate(neuron_keys):
            arr = norm_neurons[nkey].values   # (n_trials, 360)
            neuron_means[nkey] = arr[trial_indices, :].mean(axis=0)  # (360,)

        # assign phase to every visit
        visit_annotations = []
        for onset_bin, loc, run_length in onsets:
            slot, phase = assign_visit_phase(onset_bin, run_length, cfg_locs, loc)
            visit_annotations.append((onset_bin, loc, run_length, slot, phase))

        # fuse circular reward (first + last if same slot)
        visit_annotations = fuse_circular_reward(visit_annotations)

        # build phase-episodes
        episodes = build_phase_episodes(visit_annotations)
        n_ep = len(episodes)

        if n_ep == 0:
            continue

        # Pre-compute episode-level lookahead (7 future phase-episodes)
        episode_lookaheads = []
        for ep_idx in range(n_ep):
            lookahead = []
            for k in range(1, N_LOOKAHEAD + 1):
                _, _, fut_slot, fut_phase, fut_locs = episodes[(ep_idx + k) % n_ep]
                # reward → single loc; path → list of locs
                fut_loc_val = fut_locs[0] if fut_phase == 'reward' else fut_locs
                lookahead.append((fut_loc_val, fut_slot, fut_phase))
            episode_lookaheads.append(lookahead)

        # Map each individual visit to its parent episode index
        visit_ep_map = []
        for ep_idx_inner, (_, _, _, ep_phase, ep_locs) in enumerate(episodes):
            n_visits_in_ep = 1 if ep_phase == 'reward' else len(ep_locs)
            visit_ep_map.extend([ep_idx_inner] * n_visits_in_ep)

        # One row per individual visit (own onset, run_length, snippet, loc)
        # but lookahead is shared within an episode
        for vi, (onset_bin, loc, run_length, slot, phase) in enumerate(visit_annotations):
            ep_idx_for_visit = visit_ep_map[vi]
            lookahead = episode_lookaheads[ep_idx_for_visit]

            for n_idx, nkey in enumerate(neuron_keys):
                roi     = cell_labels[n_idx] if n_idx < len(cell_labels) else 'unknown'
                snippet = extract_snippet_circular(neuron_means[nkey], onset_bin)

                row = {
                    'session':          f"s{sub_str}",
                    'neuron_label':     nkey,
                    'roi':              roi,
                    'config':           config_str,
                    'grid_no':          gno,
                    'n_repeats':        n_reps,
                    'current_location': loc,
                    'onset_bin':        onset_bin,
                    'run_length':       run_length,
                    'slot':             slot,
                    'phase':            phase,
                }
                for k in range(1, N_LOOKAHEAD + 1):
                    fut_loc_val, fut_slot, fut_phase = lookahead[k - 1]
                    row[f'next+{k}_loc']   = str(fut_loc_val) if isinstance(fut_loc_val, list) else fut_loc_val
                    row[f'next+{k}_slot']  = fut_slot
                    row[f'next+{k}_phase'] = fut_phase
                for b, v in enumerate(snippet):
                    row[SNIPPET_COLS[b]] = v

                session_rows.append(row)

    if session_rows:
        df_sess   = pd.DataFrame(session_rows, columns=FIXED_COLS)
        out_path  = os.path.join(DATA_FOLDER, f"s{sub_str}", "cells_and_beh",
                                 f"location_snippets_s{sub_str}.csv")
        df_sess.to_csv(out_path, index=False)
        n_visits_total = len(session_rows) // len(neuron_keys) if neuron_keys else 0
        phase_counts   = pd.Series([r['phase'] for r in session_rows[:n_visits_total]]).value_counts().to_dict()
        print(f"s{sub_str}: {len(df_sess)} rows "
              f"({len(neuron_keys)} neurons × ~{n_visits_total} episodes across {len(gridno_to_trials)} grids) "
              f"phases: {phase_counts} → {os.path.basename(out_path)}")
        all_sessions_rows.append(df_sess)
    else:
        print(f"s{sub_str}: no snippets extracted.")

# -----------------------------------------------------------------------
# Save combined file
# -----------------------------------------------------------------------
if all_sessions_rows:
    combined      = pd.concat(all_sessions_rows, ignore_index=True)
    combined_path = os.path.join(DATA_FOLDER, "all_location_snippets.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined: {combined_path}  shape={combined.shape}")
    print(combined[['session', 'neuron_label', 'config', 'grid_no',
                    'current_location', 'onset_bin', 'run_length', 'slot', 'phase',
                    'next+1_loc', 'next+1_slot', 'next+1_phase']].head(8).to_string(index=False))
