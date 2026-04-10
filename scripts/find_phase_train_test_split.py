#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_phase_train_test_split.py

Extends find_best_train_test_split.py to phase-resolved decoding.

Instead of splitting by slot (A/B/C/D), we split by:
  (neuron, location, phase)  — where phase ∈ {reward, early, late}

Source: all_location_snippets.csv produced by create_location_snippets.py.
Each row is one (neuron × grid_no × visit), so a given neuron visiting
location 3 in the 'early' phase on grid run 7 contributes one observation.

For each phase separately:
  1. Count how many observations each neuron has per location.
  2. Find the threshold (MIN_OBS) that maximises observations per neuron/location
     while retaining >=70% of neurons that have all 9 locations covered.
  3. Split each (neuron, location) pool into train/test using the same
     ceil-half rule as find_best_train_test_split.py.
  4. Save assignment and summary CSVs, one set per phase.

Output files (in RESULTS_DIR):
  phase_snippets_{phase}_assignment.csv   — which row_ids go to train/test
  phase_snippets_{phase}_summary.csv      — n_train, n_test per neuron×location
  phase_snippets_{phase}_threshold.txt    — chosen MIN_OBS and neuron counts
  
  
  OLD
  
  IGNORE
  
  
  
  
  
"""

import os
import numpy as np
import pandas as pd

DATA_DIR    = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
RESULTS_DIR = "/Users/xpsy1114/Documents/projects/multiple_clocks/results"
SNIPPETS_CSV = os.path.join(DATA_DIR, "all_location_snippets.csv")

ALL_LOCS   = list(range(1, 10))
ALL_PHASES = ['reward', 'early', 'late']
RANDOM_SEED = 42
RETAIN_FRAC = 0.70   # target: keep at least this fraction of neurons
# Accept up to 30% loss; prefer higher obs per neuron if pct > 70%.
# If 70% can never be reached, fall back to the threshold that retains the most neurons.


# -----------------------------------------------------------------------
# Load snippets
# -----------------------------------------------------------------------
print(f"Loading {SNIPPETS_CSV} ...")
snip = pd.read_csv(SNIPPETS_CSV)
print(f"  {len(snip)} rows | {snip['neuron_label'].nunique()} neurons | "
      f"{snip['current_location'].nunique()} locations | "
      f"phases: {sorted(snip['phase'].unique())}")

# Add a unique row id for the assignment table
snip = snip.reset_index(drop=True)
snip.index.name = 'row_id'
snip = snip.reset_index()   # row_id is now a column


# -----------------------------------------------------------------------
# Per phase: count, threshold scan, split, save
# -----------------------------------------------------------------------
for phase in ALL_PHASES:
    print(f"\n{'='*60}")
    print(f"Phase: {phase.upper()}")
    print(f"{'='*60}")

    ph = snip[snip['phase'] == phase].copy()
    print(f"  Rows in phase: {len(ph)}")

    # ---- count observations per (neuron, location) ----
    counts = (ph.groupby(['neuron_label', 'current_location'])
                .size()
                .unstack(fill_value=0))
    # ensure all 9 locations are columns
    for loc in ALL_LOCS:
        if loc not in counts.columns:
            counts[loc] = 0
    counts = counts[ALL_LOCS]

    total_neurons = len(counts)
    print(f"  Neurons with >=1 observation in this phase: {total_neurons}")

    # ---- threshold scan ----
    # a neuron is "qualifying" if it has >=min_obs for ALL 9 locations
    print(f"\n  Threshold scan (target: retain >={RETAIN_FRAC:.0%} of neurons):")
    print(f"  {'min_obs':>7}  {'n_retained':>10}  {'pct':>6}  {'min_train_possible':>18}")
    best_min_obs  = 1
    best_retained = (counts >= 1).all(axis=1).sum()   # fallback: best we can do
    for min_obs in range(1, int(counts.min(axis=1).max()) + 1):
        n_retained = (counts >= min_obs).all(axis=1).sum()
        pct = n_retained / total_neurons if total_neurons else 0
        min_train = (min_obs + 1) // 2
        print(f"  {min_obs:>7}  {n_retained:>10}  {pct:>6.1%}  {min_train:>18}")
        # choose highest min_obs that still meets retention target
        if pct >= RETAIN_FRAC:
            best_min_obs  = min_obs
            best_retained = n_retained

    print(f"\n  → Chosen MIN_OBS={best_min_obs}  (retains {best_retained}/{total_neurons} "
          f"= {best_retained/total_neurons:.1%} neurons)")
    if best_retained / total_neurons < RETAIN_FRAC:
        print(f"  NOTE: {RETAIN_FRAC:.0%} retention impossible for this phase. "
              f"Using best achievable ({best_retained/total_neurons:.1%}).")

    # ---- apply split ----
    rng = np.random.default_rng(RANDOM_SEED)

    # restrict to qualifying neurons
    qualifying = list(counts.index[(counts >= best_min_obs).all(axis=1)])
    ph_q = ph[ph['neuron_label'].isin(qualifying)].copy()

    split_rows      = []
    assignment_rows = []

    for (neu, loc), grp in ph_q.groupby(['neuron_label', 'current_location']):
        row_ids = grp['row_id'].values
        n_total = len(row_ids)

        # ceil-half to test, at least 1 to train
        n_test  = max(0, min(n_total - 1, (n_total + 1) // 2))
        n_train = n_total - n_test

        perm      = rng.permutation(n_total)
        test_ids  = row_ids[perm[:n_test]]
        train_ids = row_ids[perm[n_test:]]

        split_rows.append({
            'neuron_label':     neu,
            'location_label':   loc,
            'phase':            phase,
            'n_total':          n_total,
            'n_train':          n_train,
            'n_test':           n_test,
        })

        for rid in train_ids:
            assignment_rows.append({'row_id': rid, 'neuron_label': neu,
                                    'location_label': loc, 'phase': phase,
                                    'set': 'train'})
        for rid in test_ids:
            assignment_rows.append({'row_id': rid, 'neuron_label': neu,
                                    'location_label': loc, 'phase': phase,
                                    'set': 'test'})

    split_df  = pd.DataFrame(split_rows)
    assign_df = pd.DataFrame(assignment_rows)

    # ---- summary stats ----
    print(f"\n  Split summary ({len(qualifying)} qualifying neurons):")
    print(f"  n_train per (neuron,loc):  "
          f"min={split_df['n_train'].min()}  "
          f"median={split_df['n_train'].median():.0f}  "
          f"max={split_df['n_train'].max()}")
    print(f"  n_test per (neuron,loc):   "
          f"min={split_df['n_test'].min()}  "
          f"median={split_df['n_test'].median():.0f}  "
          f"max={split_df['n_test'].max()}")

    # pooled per location
    tr_pool = assign_df[assign_df['set']=='train'].groupby('location_label').size()
    te_pool = assign_df[assign_df['set']=='test'].groupby('location_label').size()
    print(f"\n  Pooled train obs per location:\n{tr_pool.to_string()}")
    print(f"\n  Pooled test obs per location:\n{te_pool.to_string()}")

    # ---- save ----
    assign_path = os.path.join(RESULTS_DIR, f"phase_snippets_{phase}_assignment.csv")
    split_path  = os.path.join(RESULTS_DIR, f"phase_snippets_{phase}_summary.csv")
    thresh_path = os.path.join(RESULTS_DIR, f"phase_snippets_{phase}_threshold.txt")

    assign_df.to_csv(assign_path, index=False)
    split_df.to_csv(split_path,   index=False)

    with open(thresh_path, 'w') as f:
        f.write(f"Phase: {phase}\n")
        f.write(f"MIN_OBS: {best_min_obs}\n")
        f.write(f"Neurons retained: {best_retained} / {total_neurons} "
                f"({best_retained/total_neurons:.1%})\n")
        f.write(f"Min train per (neuron,loc): {split_df['n_train'].min()}\n")
        f.write(f"Min test per (neuron,loc):  {split_df['n_test'].min()}\n")

    print(f"\n  Saved:\n    {assign_path}\n    {split_path}\n    {thresh_path}")

print("\nDone.")
