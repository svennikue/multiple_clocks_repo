#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session overview for human ephys dataset.
Collects per-session metadata:
  - ABCD reward configurations solved
  - Runs and correct repeats per configuration
  - Neuron counts per ROI
  - Cross-session config overlap

Reads only lightweight metadata files (no neuron firing data).
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

DATA_FOLDER  = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
OUTPUT_CSV   = "/Users/xpsy1114/Documents/projects/multiple_clocks/results/session_overview.csv"
CONFIG_CSV   = "/Users/xpsy1114/Documents/projects/multiple_clocks/results/config_cross_session_overview.csv"
PIVOT_CSV    = "/Users/xpsy1114/Documents/projects/multiple_clocks/results/config_pivot_table.csv"

ROIS = ('entorhinal', 'hippocampus', 'ACC', 'amygdala', 'PCC', 'OFC', 'other')

BEH_COLUMNS = [
    'rep_correct', 't_A', 't_B', 't_C', 't_D',
    'loc_A', 'loc_B', 'loc_C', 'loc_D',
    'rep_overall', 'new_grid_onset', 'session_no', 'grid_no', 'correct'
]

# ROI mapping following rename_rois() logic in helpers_human_cells.py
def get_roi(label):
    if any(k in label for k in ('ACC', 'vCC', 'AMC', 'vmPFC')):
        return 'ACC'
    elif 'PCC' in label:
        return 'PCC'
    elif 'OFC' in label:
        return 'OFC'
    elif 'MCC' in label or 'HC' in label:
        return 'hippocampus'
    elif 'EC' in label:
        return 'entorhinal'
    elif 'AMYG' in label:
        return 'amygdala'
    else:
        return 'other'


def load_session(sub_str):
    """Load lightweight metadata for one session. Returns dict or None."""
    sub_folder = os.path.join(DATA_FOLDER, f"s{sub_str}", "cells_and_beh")
    if not os.path.exists(sub_folder):
        return None

    result = {}

    # --- ABCD configurations (one row per grid block) ---
    configs_path = os.path.join(sub_folder, f"all_configs_sub{sub_str}.csv")
    if not os.path.exists(configs_path):
        print(f"  Warning: configs file missing for s{sub_str}")
        return None
    configs = np.genfromtxt(configs_path, delimiter=',')
    if configs.ndim == 1:
        configs = configs.reshape(1, -1)
    result['configs'] = configs.astype(int)   # shape: (n_grid_blocks, 4)

    # --- Behavioural data ---
    beh_path = os.path.join(sub_folder, f"all_trial_times_{sub_str}.csv")
    if not os.path.exists(beh_path):
        print(f"  Warning: beh file missing for s{sub_str}")
        return None
    beh = pd.read_csv(beh_path, header=None)
    beh.columns = BEH_COLUMNS
    result['beh'] = beh

    # --- Cell region labels ---
    labels_path = os.path.join(sub_folder, f"all_cells_region_labels_sub{sub_str}.txt")
    if not os.path.exists(labels_path):
        print(f"  Warning: region labels missing for s{sub_str}")
        return None
    with open(labels_path) as f:
        result['cell_labels'] = [line.strip() for line in f if line.strip()]

    return result


def config_label(cfg_tuple):
    """Format a config tuple as '6-4-5-9' (A-B-C-D positions)."""
    return f"{cfg_tuple[0]}-{cfg_tuple[1]}-{cfg_tuple[2]}-{cfg_tuple[3]}"


def parse_session(sub_str, data):
    """Return (row_dict, config_to_grids) for one session."""
    beh     = data['beh']
    configs = data['configs']   # (n_blocks, 4)
    labels  = data['cell_labels']

    # Map grid_no (1-indexed) → config tuple
    grid_to_config = {i + 1: tuple(configs[i]) for i in range(len(configs))}

    # Group grid_nos by unique config
    config_to_grids = {}
    for gno, cfg in grid_to_config.items():
        config_to_grids.setdefault(cfg, []).append(gno)

    n_unique_configs = len(config_to_grids)
    configs_str = ' | '.join(config_label(c) for c in config_to_grids)

    # Runs and correct repeats
    runs_per_config     = [len(grids) for grids in config_to_grids.values()]
    repeats_per_run_all = []
    for grids in config_to_grids.values():
        for gno in grids:
            mask = beh['grid_no'] == gno
            if mask.any():
                repeats_per_run_all.append(int(beh.loc[mask, 'rep_correct'].max()))

    n_runs_total         = sum(runs_per_config)
    mean_runs_per_config = round(np.mean(runs_per_config), 1)
    mean_correct_repeats = round(np.mean(repeats_per_run_all), 1) if repeats_per_run_all else np.nan

    # ROI counts
    roi_counts = {r: 0 for r in ROIS}
    for lbl in labels:
        roi_counts[get_roi(lbl)] += 1

    row = {
        'session':                      f"s{sub_str}",
        'n_unique_configs':             n_unique_configs,
        'configs_ABCD':                 configs_str,
        'n_runs_total':                 n_runs_total,
        'mean_runs_per_config':         mean_runs_per_config,
        'mean_correct_repeats_per_run': mean_correct_repeats,
        'n_neurons_total':              len(labels),
        'n_entorhinal':                 roi_counts['entorhinal'],
        'n_hippocampus':                roi_counts['hippocampus'],
        'n_ACC':                        roi_counts['ACC'],
        'n_amygdala':                   roi_counts['amygdala'],
        'n_PCC':                        roi_counts['PCC'],
        'n_OFC':                        roi_counts['OFC'],
        'n_other':                      roi_counts['other'],
    }
    return row, config_to_grids, roi_counts


# --- Main ---
session_dirs = sorted(
    d[1:] for d in os.listdir(DATA_FOLDER)
    if d.startswith('s') and os.path.isdir(os.path.join(DATA_FOLDER, d))
)

# Collect per-session data; build global registries
rows              = []
session_roi       = {}    # sub_str → roi_counts dict
global_registry   = defaultdict(list)   # cfg_tuple → [(session_label, n_runs), ...]

for sub_str in session_dirs:
    print(f"Processing s{sub_str} ...")
    data = load_session(sub_str)
    if data is None:
        continue
    row, config_to_grids, roi_counts = parse_session(sub_str, data)
    rows.append(row)
    session_roi[sub_str] = roi_counts
    for cfg, grids in config_to_grids.items():
        global_registry[cfg].append((f"s{sub_str}", len(grids)))

# Add cross-session overlap column to session overview
for row in rows:
    sub_str   = row['session'][1:]
    parts = []
    for cfg_str in row['configs_ABCD'].split(' | '):
        # Recover cfg_tuple from label string
        vals = tuple(int(x) for x in cfg_str.split('-'))
        entries = global_registry[vals]
        total   = sum(n for _, n in entries)
        details = ', '.join(f"{s}({n})" for s, n in entries)
        parts.append(f"{cfg_str}: {total} runs [{details}]")
    row['config_cross_session'] = ' | '.join(parts)

df = pd.DataFrame(rows)

# -----------------------------------------------------------------------
# Table 1 – config_cross_session_overview.csv
#   One row per unique config. Columns: sequence, total_runs, sessions,
#   session_run_details (the per-session breakdown string).
# -----------------------------------------------------------------------
config_summary_rows = []
for cfg, entries in sorted(global_registry.items(), key=lambda x: -sum(n for _, n in x[1])):
    total_runs     = sum(n for _, n in entries)
    sessions_list  = str([s for s, _ in entries])           # e.g. "['s02', 's04', ...]"
    run_details    = ', '.join(f"{s}({n})" for s, n in entries)
    config_summary_rows.append({
        'sequence':           config_label(cfg),
        'total_runs':         total_runs,
        'n_sessions':         len(entries),
        'sessions':           sessions_list,
        'session_run_details': run_details,
    })
df_config_summary = pd.DataFrame(config_summary_rows)

# -----------------------------------------------------------------------
# Table 2 – config_pivot_table.csv
#   Rows = unique configs, columns = each session + summary + neuron ROIs.
#   Cell value = n_runs for that config in that session (0 if absent).
#   Neuron columns = summed across all sessions that ran the config.
# -----------------------------------------------------------------------
all_sessions = [f"s{s}" for s in session_dirs]
pivot_rows = []
for cfg, entries in sorted(global_registry.items(), key=lambda x: -sum(n for _, n in x[1])):
    entry_dict = dict(entries)    # session_label → n_runs
    total_runs = sum(entry_dict.values())

    # Neurons: sum ROI counts across all sessions that ran this config
    roi_totals = {r: 0 for r in ROIS}
    for sess_label, _ in entries:
        sub_str = sess_label[1:]   # strip 's'
        if sub_str in session_roi:
            for r in ROIS:
                roi_totals[r] += session_roi[sub_str][r]

    pivot_row = {'sequence': config_label(cfg)}
    for sess in all_sessions:
        pivot_row[sess] = entry_dict.get(sess, 0)
    pivot_row['total_runs'] = total_runs
    for r in ROIS:
        pivot_row[f'n_{r}'] = roi_totals[r]
    pivot_rows.append(pivot_row)

df_pivot = pd.DataFrame(pivot_rows)

# -----------------------------------------------------------------------
# Save everything
# -----------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
df_config_summary.to_csv(CONFIG_CSV, index=False)
df_pivot.to_csv(PIVOT_CSV, index=False)

print(f"\n=== Config Cross-Session Summary (top 10 by total runs) ===")
print(df_config_summary.head(10).to_string(index=False))
print(f"\n=== Pivot Table (first 5 rows, summary columns only) ===")
summary_cols = ['sequence', 'total_runs'] + [f'n_{r}' for r in ROIS]
print(df_pivot[summary_cols].head(10).to_string(index=False))
print(f"\nSession overview saved to:   {OUTPUT_CSV}")
print(f"Config summary saved to:     {CONFIG_CSV}")
print(f"Config pivot table saved to: {PIVOT_CSV}")
