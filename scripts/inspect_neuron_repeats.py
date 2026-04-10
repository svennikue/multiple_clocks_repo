#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper: report how many repeats of each location each neuron has.

Input:  all_neurons_avg_per_gridrun.csv  (one row per neuron x grid_no run)
        config 'A-B-C-D' means loc_A=A, loc_B=B, loc_C=C, loc_D=D
        train set = slots A, B, C  |  test set = slot D

Output:
  - per-neuron repeat counts for train and test
  - threshold scan: for a given (min_train_reps, min_train_locs), how many
    neurons survive and does the population still cover all 9 test locations?
  - heatmap figures saved to RESULTS_DIR
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR    = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
RESULTS_DIR = "/Users/xpsy1114/Documents/projects/multiple_clocks/results"
NEURON_CSV  = os.path.join(DATA_DIR, "all_neurons_avg_per_gridrun.csv")

# -----------------------------------------------------------------------
# Load and build long format
# -----------------------------------------------------------------------
df = pd.read_csv(NEURON_CSV)
df[['loc_A', 'loc_B', 'loc_C', 'loc_D']] = df['config'].str.split('-', expand=True).astype(int)

long = []
for slot, loc_col in [('A','loc_A'), ('B','loc_B'), ('C','loc_C'), ('D','loc_D')]:
    tmp = df[['session', 'neuron_label', 'grid_no', 'config', loc_col]].copy()
    tmp.columns = ['session', 'neuron_label', 'grid_no', 'config', 'location_label']
    tmp['slot'] = slot
    tmp['set']  = 'test' if slot == 'D' else 'train'
    long.append(tmp)
long_df = pd.concat(long, ignore_index=True)

train = long_df[long_df['set'] == 'train']
test  = long_df[long_df['set'] == 'test']

# repeat count tables: rows = neuron, cols = location (1..9)
train_rep = train.groupby(['neuron_label', 'location_label']).size().unstack(fill_value=0)
test_rep  = test.groupby(['neuron_label',  'location_label']).size().unstack(fill_value=0)

# ensure all 9 location columns present
for loc in range(1, 10):
    if loc not in train_rep.columns: train_rep[loc] = 0
    if loc not in test_rep.columns:  test_rep[loc]  = 0
train_rep = train_rep[sorted(train_rep.columns)]
test_rep  = test_rep[sorted(test_rep.columns)]

n_neurons = len(train_rep)
print(f"Total neurons: {n_neurons}")

# -----------------------------------------------------------------------
# Summary statistics
# -----------------------------------------------------------------------
print("\n--- Train repeats per (neuron, location) ---")
flat_train = train_rep.values[train_rep.values > 0]  # ignore zeros (location not seen)
print(pd.Series(flat_train).describe().to_string())

print("\n--- Test repeats per (neuron, location) ---")
flat_test = test_rep.values[test_rep.values > 0]
print(pd.Series(flat_test).describe().to_string())

print("\nPer-neuron: how many locations covered in train (any repeats)?")
print((train_rep > 0).sum(axis=1).value_counts().sort_index().to_string())

print("\nPer-neuron: min train repeats across covered locations:")
min_train = train_rep[train_rep > 0].min(axis=1)  # NaN where neuron has 0 for that loc
# actually compute min only over covered locations
min_train_reps = train_rep.apply(lambda row: row[row > 0].min() if (row > 0).any() else 0, axis=1)
print(min_train_reps.value_counts().sort_index().to_string())

# -----------------------------------------------------------------------
# Threshold scan
# -----------------------------------------------------------------------
print("\n--- Threshold scan ---")
print(f"{'min_train_reps':>15} {'min_train_locs':>15} {'n_neurons':>10} {'test_locs_covered':>18}")

scan_results = []
for min_reps in [2, 3, 4, 6]:
    for min_locs in [7, 8, 9]:
        # a neuron survives if it has >= min_reps for >= min_locs locations (in train)
        n_locs_above = (train_rep >= min_reps).sum(axis=1)
        kept = n_locs_above[n_locs_above >= min_locs].index

        # test population coverage
        te_pooled_locs = (test_rep.loc[kept] > 0).any(axis=0)
        n_test_locs = te_pooled_locs.sum()

        print(f"{min_reps:>15} {min_locs:>15} {len(kept):>10} {n_test_locs:>18}/9")
        scan_results.append({
            'min_train_reps': min_reps,
            'min_train_locs': min_locs,
            'n_neurons':      len(kept),
            'test_locs':      n_test_locs,
        })

# -----------------------------------------------------------------------
# Heatmap: mean train repeats per neuron x location  (sorted by n_locs covered)
# -----------------------------------------------------------------------
order = (train_rep > 0).sum(axis=1).sort_values(ascending=False).index
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, rep_df, title in zip(axes,
                              [train_rep.loc[order], test_rep.loc[order]],
                              ['Train repeats (slots A+B+C)', 'Test repeats (slot D)']):
    im = ax.imshow(rep_df.values, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=rep_df.values.max())
    ax.set_xlabel('Location (1–9)')
    ax.set_ylabel('Neuron (sorted by train coverage)')
    ax.set_xticks(range(9))
    ax.set_xticklabels(range(1, 10))
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='n repeats')

plt.suptitle('Repeats per neuron × location', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'neuron_repeat_counts.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nHeatmap saved.")

# -----------------------------------------------------------------------
# Distribution plot: histogram of min-repeats per neuron (train)
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(min_train_reps.values, bins=range(0, int(min_train_reps.max()) + 2),
             color='steelblue', edgecolor='white')
axes[0].set_xlabel('Min train repeats across covered locations')
axes[0].set_ylabel('Number of neurons')
axes[0].set_title('Per-neuron minimum train repeats')

n_locs_covered = (train_rep > 0).sum(axis=1)
axes[1].hist(n_locs_covered.values, bins=range(0, 12), color='steelblue', edgecolor='white')
axes[1].set_xlabel('Number of train locations covered')
axes[1].set_ylabel('Number of neurons')
axes[1].set_title('Per-neuron train location coverage')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'neuron_repeat_distribution.png'), dpi=150)
plt.close()
print("Distribution plot saved.")
