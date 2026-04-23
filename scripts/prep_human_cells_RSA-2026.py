#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preparation script for human-cell RSA  (state RSA  OR  DSR RSA).

Set MODE = 'state' or MODE = 'dsr' below.

──────────────────────────────────────────────────────────────────────
MODE = 'state'
──────────────────────────────────────────────────────────────────────
All 63 sessions.  Unique ABCD configs are grouped into N_GROUPS=6
pseudo-configs (merging when n_unique > 6) and split into 2 runs.

Balancing:
  - Extra configs roll across group slots across sessions.
  - Odd-block groups alternate which run gets the extra block.

Output per session  →  DATA_DIR/s{sub}/state_avg/
  s{sub}_neural_avg.npy          (n_neurons, 6, 2, 360)
  s{sub}_neuron_meta.json
  s{sub}_grouping_log.json

──────────────────────────────────────────────────────────────────────
MODE = 'dsr'
──────────────────────────────────────────────────────────────────────
Only the 28 sessions that share the same 8 ABCD configurations.
Each of the 8 configs is kept separate (no merging) and split into
2 runs.  Odd-block balancing uses the same rolling counter.

Config axis order (fixed across all sessions):
  0: 3-7-9-5   1: 8-2-6-7   2: 1-9-5-8   3: 4-8-1-3
  4: 6-4-2-9   5: 9-1-3-4   6: 7-3-4-2   7: 2-5-7-6

Output per session  →  DATA_DIR/s{sub}/dsr_avg/
  s{sub}_dsr_neural_avg.npy      (n_neurons, 8, 2, 360)
  s{sub}_dsr_neuron_meta.json
  s{sub}_dsr_grouping_log.json

@author: Svenja Küchenhoff, 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
from mc.analyse.helpers_human_cells import load_norm_data
# import pdb; pdb.set_trace()


# ══════════════════════════════════════════════════════════════════════
# ── Settings  ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

MODE = 'dsr'     # 'state'  or  'dsr'

# If True, skip run-balancing entirely: for each config, average correct
# trials within each grid_no block separately and stack the per-block
# averages. Output per config is (n_neurons, n_runs_for_this_config, N_BINS),
# with n_runs varying across configs — saved as an .npz with one array per
# config label. Currently supported for MODE='dsr' only.
KEEP_RUNS_SEPRATE = True

# If True, apply Gaussian smoothing to each trial before averaging.
SMOOTH_DATA = True

DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
N_BINS   = 360

# ── State-RSA settings ────────────────────────────────────────────────
N_GROUPS       = 6
STATE_SUBJECTS = [f'{i:02}' for i in range(1, 64)]
STATE_EXCLUDE  = []
STATE_SUBJECTS = [s for s in STATE_SUBJECTS if int(s) not in STATE_EXCLUDE]

# ── DSR-RSA settings ──────────────────────────────────────────────────
DSR_SUBJECTS = [
    '27','28','31','32','33','34','35','36','37','38',
    '40','43','44','45','46','49','50','51','53','55',
    '56','57','58','59','60','61','62','63',
]
# Fixed config order (matches pivot table rows 0-7)
DSR_CONFIGS = [
    (3,7,9,5), (8,2,6,7), (1,9,5,8), (4,8,1,3),
    (6,4,2,9), (9,1,3,4), (7,3,4,2), (2,5,7,6),
]
DSR_CONFIG_LABELS = [f'{c[0]}-{c[1]}-{c[2]}-{c[3]}' for c in DSR_CONFIGS]
N_DSR_CONFIGS = len(DSR_CONFIGS)

BEH_COLUMNS = [
    'rep_correct', 't_A', 't_B', 't_C', 't_D',
    'loc_A', 'loc_B', 'loc_C', 'loc_D',
    'rep_overall', 'new_grid_onset', 'session_no', 'grid_no', 'correct',
]


# ══════════════════════════════════════════════════════════════════════
# ── Shared helpers ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def load_beh(sub_str):
    """Lightweight beh-only loader (used in planning phase)."""
    path = os.path.join(DATA_DIR, f's{sub_str}', 'cells_and_beh',
                        f'all_trial_times_{sub_str}.csv')
    if not os.path.exists(path):
        return None
    beh = pd.read_csv(path, header=None)
    beh.columns = BEH_COLUMNS
    return beh


def assign_blocks_to_runs_balanced(blocks, beh_correct, counters):
    """
    Assign each grid_no block *in full* to run1 or run2.

    For even block counts: greedy by trial count.
    For odd block counts:  greedy, then swap run1/run2 on every other
                           encounter (rolling counters['odd_block']).

    Returns: run1_blocks, run2_blocks, n1, n2
    """
    block_sizes = {b: int((beh_correct['grid_no'] == b).sum()) for b in blocks}

    run1, run2 = [], []
    n1, n2 = 0, 0
    for b in blocks:
        if n1 <= n2:
            run1.append(b); n1 += block_sizes[b]
        else:
            run2.append(b); n2 += block_sizes[b]

    if len(blocks) % 2 == 1:
        if counters['odd_block'] % 2 == 1:
            run1, run2 = run2, run1
            n1,   n2   = n2,   n1
        counters['odd_block'] += 1

    return run1, run2, n1, n2


# Gaussian smoothing applied to each trial before averaging
# width = 4 bins (truncate at 2σ from centre), σ = 2 bins
_GAUSS_SIGMA    = 2    # bins
_GAUSS_TRUNCATE = 2.0  # kernel extends 2σ = 4 bins from centre

def avg_neuron(neuron_df, trial_indices):
    valid = [i for i in trial_indices if i < len(neuron_df)]
    if not valid:
        return np.full(N_BINS, np.nan)
    trials = neuron_df.iloc[valid].to_numpy().astype(float)
    if SMOOTH_DATA:
        trials = np.array([
            gaussian_filter1d(row, sigma=_GAUSS_SIGMA, truncate=_GAUSS_TRUNCATE)
            for row in trials
        ])
    return np.nanmean(trials, axis=0)


# ══════════════════════════════════════════════════════════════════════
# ── State-RSA helpers ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def build_groups_balanced(beh_correct, n_groups, counters):
    """
    Group unique ABCD configs into n_groups pseudo-configs.
    Extra configs (n_unique % n_groups remainder) are assigned via a rolling
    counter that rotates across group slots across sessions.

    Returns: groups (list of dicts), config_to_blocks (dict)
    """
    beh = beh_correct.copy()
    beh['config'] = list(zip(
        beh['loc_A'].astype(int), beh['loc_B'].astype(int),
        beh['loc_C'].astype(int), beh['loc_D'].astype(int),
    ))

    config_to_blocks = {}
    for cfg, rows in beh.groupby('config', sort=False):
        config_to_blocks[cfg] = sorted(rows['grid_no'].unique().tolist())

    unique_configs = list(config_to_blocks.keys())
    n_unique = len(unique_configs)
    base     = n_unique // n_groups
    n_extra  = n_unique % n_groups

    groups  = [{'configs': [], 'blocks': []} for _ in range(n_groups)]
    cfg_idx = 0

    for g in range(n_groups):
        for _ in range(base):
            groups[g]['configs'].append(unique_configs[cfg_idx])
            cfg_idx += 1

    for i in range(n_extra):
        g_slot = (counters['extra_config'] + i) % n_groups
        groups[g_slot]['configs'].append(unique_configs[cfg_idx])
        cfg_idx += 1
    counters['extra_config'] = (counters['extra_config'] + n_extra) % n_groups

    for g in groups:
        g['blocks'] = sorted(
            b for cfg in g['configs'] for b in config_to_blocks[cfg]
        )

    while len(groups) < n_groups:
        groups.append({'configs': [], 'blocks': []})

    return groups, config_to_blocks


def plan_state_session(beh, counters):
    beh_correct = beh[beh['correct'] == 1].copy()
    if beh_correct.empty:
        return None
    groups, _ = build_groups_balanced(beh_correct, N_GROUPS, counters)
    plan = []
    for group in groups:
        blocks = group['blocks']
        if not blocks:
            plan.append({'n1': 0, 'n2': 0}); continue
        _, _, n1, n2 = assign_blocks_to_runs_balanced(blocks, beh_correct, counters)
        plan.append({'n1': n1, 'n2': n2})
    return plan


def prepare_state_session(sub_str, counters):
    print(f"\n--- s{sub_str} ---")
    sub_data = load_norm_data(DATA_DIR, [sub_str])
    key = f'sub-{sub_str}'
    if key not in sub_data:
        print("  data not found, skipping"); return None

    beh          = sub_data[key]['beh']
    neurons      = sub_data[key]['normalised_neurons']
    cell_labels  = sub_data[key]['cell_labels']
    elec_labels  = sub_data[key]['electrode_labels']
    neuron_names = sorted(neurons.keys())
    n_neurons    = len(neuron_names)

    beh_correct = beh[beh['correct'] == 1].copy()
    if beh_correct.empty:
        print("  no correct trials, skipping"); return None

    groups, _ = build_groups_balanced(beh_correct, N_GROUPS, counters)
    avg_array  = np.full((n_neurons, N_GROUPS, 2, N_BINS), np.nan)
    log_groups = []

    for g_idx, group in enumerate(groups):
        blocks = group['blocks']
        if not blocks:
            log_groups.append({'group_idx': g_idx, 'configs': [], 'blocks': [],
                               'run1_blocks': [], 'run2_blocks': [],
                               'n_trials_run1': 0, 'n_trials_run2': 0})
            continue

        run1_blocks, run2_blocks, n1, n2 = assign_blocks_to_runs_balanced(
            blocks, beh_correct, counters)
        run1_idx = beh_correct[beh_correct['grid_no'].isin(run1_blocks)].index.tolist()
        run2_idx = beh_correct[beh_correct['grid_no'].isin(run2_blocks)].index.tolist()

        for n_idx, name in enumerate(neuron_names):
            ndf = neurons[name]
            avg_array[n_idx, g_idx, 0, :] = avg_neuron(ndf, run1_idx)
            avg_array[n_idx, g_idx, 1, :] = avg_neuron(ndf, run2_idx)

        log_groups.append({'group_idx': g_idx,
                           'configs':       [list(c) for c in group['configs']],
                           'blocks':        blocks,
                           'run1_blocks':   run1_blocks,
                           'run2_blocks':   run2_blocks,
                           'n_trials_run1': n1,
                           'n_trials_run2': n2})
        print(f"  group {g_idx}: {blocks} → run1={run1_blocks}({n1}), run2={run2_blocks}({n2})")

    out_dir = os.path.join(DATA_DIR, f's{sub_str}', 'state_avg')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f's{sub_str}_neural_avg.npy'), avg_array)

    with open(os.path.join(out_dir, f's{sub_str}_grouping_log.json'), 'w') as f:
        json.dump({'session': sub_str, 'n_neurons': n_neurons,
                   'n_correct_trials': int(len(beh_correct)),
                   'groups': log_groups}, f, indent=2)
    with open(os.path.join(out_dir, f's{sub_str}_neuron_meta.json'), 'w') as f:
        json.dump({'neuron_names': neuron_names,
                   'cell_labels': cell_labels,
                   'electrode_labels': elec_labels}, f, indent=2)

    n_nan = int(np.isnan(avg_array).sum())
    print(f"  saved → shape {avg_array.shape},  NaN={n_nan}")
    return log_groups


# ══════════════════════════════════════════════════════════════════════
# ── DSR-RSA helpers ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def _tag_config(beh_correct):
    """Add a 'config' tuple column to a beh_correct DataFrame."""
    bc = beh_correct.copy()
    bc['config'] = list(zip(
        bc['loc_A'].astype(int), bc['loc_B'].astype(int),
        bc['loc_C'].astype(int), bc['loc_D'].astype(int),
    ))
    return bc


def plan_dsr_session(beh, counters):
    """
    Compute the full block assignment plan for one DSR session.
    Stores run1_blocks and run2_blocks (not just trial counts) so that
    Phase 2 can execute directly from the plan without re-deriving anything.
    """
    beh_correct = _tag_config(beh[beh['correct'] == 1].copy())
    if beh_correct.empty:
        return None
    plan = []
    for cfg in DSR_CONFIGS:
        blocks = sorted(beh_correct[beh_correct['config'] == cfg]['grid_no'].unique().tolist())
        if not blocks:
            plan.append({'blocks': [], 'run1_blocks': [], 'run2_blocks': [],
                         'n1': 0, 'n2': 0})
            continue
        run1_blocks, run2_blocks, n1, n2 = assign_blocks_to_runs_balanced(
            blocks, beh_correct, counters)
        plan.append({'blocks': blocks, 'run1_blocks': run1_blocks,
                     'run2_blocks': run2_blocks, 'n1': n1, 'n2': n2})
    return plan


def prepare_dsr_session(sub_str, plan):
    """
    Execute the pre-computed plan for one DSR session.
    Uses the run1_blocks / run2_blocks already decided in Phase 1 —
    no counters needed here.
    """
    print(f"\n--- s{sub_str} ---")
    sub_data = load_norm_data(DATA_DIR, [sub_str])
    key = f'sub-{sub_str}'
    if key not in sub_data:
        print("  data not found, skipping"); return None

    beh          = sub_data[key]['beh']
    neurons      = sub_data[key]['normalised_neurons']
    cell_labels  = sub_data[key]['cell_labels']
    elec_labels  = sub_data[key]['electrode_labels']
    neuron_names = sorted(neurons.keys())
    n_neurons    = len(neuron_names)

    beh_correct = _tag_config(beh[beh['correct'] == 1].copy())
    if beh_correct.empty:
        print("  no correct trials, skipping"); return None

    avg_array   = np.full((n_neurons, N_DSR_CONFIGS, 2, N_BINS), np.nan)
    log_configs = []
    
    for c_idx, cfg in enumerate(DSR_CONFIGS):
        label      = DSR_CONFIG_LABELS[c_idx]
        cfg_plan   = plan[c_idx]
        run1_blocks = cfg_plan['run1_blocks']
        run2_blocks = cfg_plan['run2_blocks']
        n1          = cfg_plan['n1']
        n2          = cfg_plan['n2']

        if not cfg_plan['blocks']:
            print(f"  config {label}: no blocks in plan — leaving NaN")
            log_configs.append({'config_idx': c_idx, 'config': label,
                                'blocks': [], 'run1_blocks': [], 'run2_blocks': [],
                                'n_trials_run1': 0, 'n_trials_run2': 0})
            continue

        run1_idx = beh_correct[beh_correct['grid_no'].isin(run1_blocks)].index.tolist()
        run2_idx = beh_correct[beh_correct['grid_no'].isin(run2_blocks)].index.tolist()

        for n_idx, name in enumerate(neuron_names):
            ndf = neurons[name]
            # neuron, config, run, data
            # import pdb; pdb.set_trace()
            avg_array[n_idx, c_idx, 0, :] = avg_neuron(ndf, run1_idx)
            avg_array[n_idx, c_idx, 1, :] = avg_neuron(ndf, run2_idx)

        log_configs.append({'config_idx': c_idx, 'config': label,
                            'blocks':        cfg_plan['blocks'],
                            'run1_blocks':   run1_blocks,
                            'run2_blocks':   run2_blocks,
                            'n_trials_run1': n1,
                            'n_trials_run2': n2})
        print(f"  config {label}: {cfg_plan['blocks']} → "
              f"run1={run1_blocks}({n1}), run2={run2_blocks}({n2})")

    out_dir = os.path.join(DATA_DIR, f's{sub_str}', 'dsr_avg')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f's{sub_str}_dsr_neural_avg_two_runs.npy'), avg_array)
    # axis legend saved alongside so downstream code is self-documenting
    with open(os.path.join(out_dir, f's{sub_str}_dsr_grouping_log_two_runs.json'), 'w') as f:
        json.dump({'session': sub_str, 'n_neurons': n_neurons,
                   'n_correct_trials': int(len(beh_correct)),
                   'config_axis_order': DSR_CONFIG_LABELS,
                   'configs': log_configs}, f, indent=2)
    with open(os.path.join(out_dir, f's{sub_str}_dsr_neuron_meta_two_runs.json'), 'w') as f:
        json.dump({'neuron_names': neuron_names,
                   'cell_labels': cell_labels,
                   'electrode_labels': elec_labels}, f, indent=2)

    n_nan = int(np.isnan(avg_array).sum())
    print(f"  saved → shape {avg_array.shape},  NaN={n_nan}")
    return log_configs


def prepare_dsr_session_separate_runs(sub_str):
    """
    KEEP_RUNS_SEPRATE path: one 'run' == one grid_no block.
    For each config, average correct trials within each block and stack the
    per-block averages. n_runs varies by config, so we save one (n_neurons,
    n_runs, N_BINS) array per config in an .npz.
    """
    print(f"\n--- s{sub_str} ---")
    sub_data = load_norm_data(DATA_DIR, [sub_str])
    key = f'sub-{sub_str}'
    if key not in sub_data:
        print("  data not found, skipping"); return None

    beh          = sub_data[key]['beh']
    neurons      = sub_data[key]['normalised_neurons']
    cell_labels  = sub_data[key]['cell_labels']
    elec_labels  = sub_data[key]['electrode_labels']
    neuron_names = sorted(neurons.keys())
    n_neurons    = len(neuron_names)

    beh_correct = _tag_config(beh[beh['correct'] == 1].copy())
    if beh_correct.empty:
        print("  no correct trials, skipping"); return None

    per_config_arrays = {}
    log_configs = []

    for c_idx, cfg in enumerate(DSR_CONFIGS):
        label  = DSR_CONFIG_LABELS[c_idx]
        blocks = sorted(beh_correct[beh_correct['config'] == cfg]['grid_no'].unique().tolist())
        if not blocks:
            print(f"  config {label}: no blocks — skipping")
            log_configs.append({'config_idx': c_idx, 'config': label,
                                'blocks': [], 'n_runs': 0,
                                'n_trials_per_run': []})
            continue

        n_runs   = len(blocks)
        cfg_arr  = np.full((n_neurons, n_runs, N_BINS), np.nan)
        n_trials = []

        for r_idx, block in enumerate(blocks):
            block_idx = beh_correct[beh_correct['grid_no'] == block].index.tolist()
            n_trials.append(len(block_idx))
            for n_idx, name in enumerate(neuron_names):
                cfg_arr[n_idx, r_idx, :] = avg_neuron(neurons[name], block_idx)

        per_config_arrays[label] = cfg_arr
        log_configs.append({'config_idx': c_idx, 'config': label,
                            'blocks': blocks, 'n_runs': n_runs,
                            'n_trials_per_run': n_trials})
        print(f"  config {label}: {blocks} → {n_runs} runs, trials={n_trials}")

    out_dir = os.path.join(DATA_DIR, f's{sub_str}', 'dsr_avg_runs_separate')
    os.makedirs(out_dir, exist_ok=True)
    
    np.savez(os.path.join(out_dir, f's{sub_str}_dsr_neural_avg'),
             **per_config_arrays)
    with open(os.path.join(out_dir, f's{sub_str}_dsr_grouping_log.json'), 'w') as f:
        json.dump({'session': sub_str, 'n_neurons': n_neurons,
                   'n_correct_trials': int(len(beh_correct)),
                   'config_axis_order': DSR_CONFIG_LABELS,
                   'smooth_data': SMOOTH_DATA,
                   'configs': log_configs}, f, indent=2)
    with open(os.path.join(out_dir, f's{sub_str}_dsr_neuron_meta.json'), 'w') as f:
        json.dump({'neuron_names': neuron_names,
                   'cell_labels': cell_labels,
                   'electrode_labels': elec_labels}, f, indent=2)

    total_runs = sum(lg['n_runs'] for lg in log_configs)
    print(f"  saved → {len(per_config_arrays)} configs, {total_runs} total runs")
    return log_configs


# ══════════════════════════════════════════════════════════════════════
# ── Main: choose pipeline based on MODE ───────────────────────────────
# ══════════════════════════════════════════════════════════════════════

if KEEP_RUNS_SEPRATE and MODE != 'dsr':
    raise NotImplementedError(
        "KEEP_RUNS_SEPRATE is currently implemented only for MODE='dsr'."
    )

if KEEP_RUNS_SEPRATE:
    # Skip planning entirely — every grid_no block is its own run.
    SUBJECTS = DSR_SUBJECTS
    print("=" * 100)
    print("EXECUTION  [DSR RSA, KEEP_RUNS_SEPRATE=True] — no session planning")
    print(f"SMOOTH_DATA = {SMOOTH_DATA}")
    print("=" * 100)

    all_logs = {}
    for sub in SUBJECTS:
        result = prepare_dsr_session_separate_runs(sub)
        if result is not None:
            all_logs[sub] = result

    summary_path = os.path.join(
        DATA_DIR, 'all_sessions_dsrRSA_separateRuns_grouping_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_logs, f, indent=2)

    print('\nAll done.')
    sys.exit(0)

if MODE == 'state':
    SUBJECTS      = STATE_SUBJECTS
    plan_fn       = plan_state_session
    summary_key   = 'all_sessions_stateRSA_grouping_summary.json'
    counters_init = lambda: {'extra_config': 0, 'odd_block': 0}
elif MODE == 'dsr':
    SUBJECTS      = DSR_SUBJECTS
    plan_fn       = plan_dsr_session
    summary_key   = 'all_sessions_dsrRSA_grouping_summary.json'
    counters_init = lambda: {'odd_block': 0}   # no extra_config needed for DSR
else:
    raise ValueError(f"MODE must be 'state' or 'dsr', got {MODE!r}")

# ── PHASE 1 — PLANNING ────────────────────────────────────────────────

n_cols = N_GROUPS if MODE == 'state' else N_DSR_CONFIGS
col_labels = [f'G{g}' for g in range(n_cols)] if MODE == 'state' \
             else DSR_CONFIG_LABELS

print("=" * 100)
print(f"PHASE 1 — ALLOCATION PLAN  [{MODE.upper()} RSA]  (trials per config/group × run)")
print("=" * 100)
hdr = "  ".join(f"{lbl[:7]:>7}r1  {lbl[:7]:>7}r2" for lbl in col_labels)
print(f"{'session':>9} | {hdr}")
print("-" * 100)

counters_plan = counters_init()
all_plans     = {}
skipped_plan  = []



for sub in SUBJECTS:
    beh = load_beh(sub)
    if beh is None:
        skipped_plan.append(sub); continue
    plan = plan_fn(beh, counters_plan)
    if plan is None:
        skipped_plan.append(sub); continue
    all_plans[sub] = plan
    cells = "  ".join(f"{p['n1']:9d}  {p['n2']:9d}" for p in plan)
    print(f"  s{sub}   | {cells}")

print("-" * 100)
print("MEAN per column/run across sessions:")


for c in range(n_cols):
    r1 = [all_plans[s][c]['n1'] for s in all_plans]
    r2 = [all_plans[s][c]['n2'] for s in all_plans]
    print(f"  {col_labels[c]:>10}: run1 mean={np.mean(r1):.1f}  run2 mean={np.mean(r2):.1f}")

print()
all_r1 = [all_plans[s][c]['n1'] for s in all_plans for c in range(n_cols)]
all_r2 = [all_plans[s][c]['n2'] for s in all_plans for c in range(n_cols)]
print(f"OVERALL — run1: mean={np.mean(all_r1):.1f}  std={np.std(all_r1):.1f}  min={np.min(all_r1)}")
print(f"OVERALL — run2: mean={np.mean(all_r2):.1f}  std={np.std(all_r2):.1f}  min={np.min(all_r2)}")
if skipped_plan:
    print(f"Not found (skipped from plan): {skipped_plan}")

# ── PHASE 2 — EXECUTION ───────────────────────────────────────────────


print()
print("=" * 100)
print(f"PHASE 2 — EXECUTION  [{MODE.upper()} RSA]")
print("=" * 100)

all_logs          = {}
counters_exec     = counters_init()   # shared across sessions for state mode

for sub in SUBJECTS:
    if MODE == 'dsr':
        if sub not in all_plans:
            print(f"\n--- s{sub}: no plan, skipping ---")
            continue
        result = prepare_dsr_session(sub, all_plans[sub])
    else:
        result = prepare_state_session(sub, counters_exec)
    if result is not None:
        all_logs[sub] = result

with open(os.path.join(DATA_DIR, summary_key), 'w') as f:
    json.dump(all_logs, f, indent=2)

print('\nAll done.')
