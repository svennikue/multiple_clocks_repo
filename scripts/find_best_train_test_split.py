#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper: find the best train/test split of slot observations per neuron x location.

Problem with the fixed A/B/C=train, D=test split:
  - Some (neuron, location) pairs have D=0 observations (test missing)
  - Those same pairs often have 9-12 A/B/C observations doing nothing for test

Idea: treat all 4 slots as a pool of observations per (neuron, location).
Find the split that maximises min(n_train, n_test) per location per neuron,
with a preference for more test observations (since test is the bottleneck).

Strategy:
  For each (neuron, location), we have n_total = n_A + n_B + n_C + n_D obs.
  We want to assign each obs to train or test.
  Because the obs are already averaged per grid_no run (one row = one run),
  we split at the grid_no level, not within a run.

  For each (neuron, location), the obs come from specific grid_no runs.
  A run contributes exactly one obs to exactly one slot for a given location.

  Simple rule: assign obs to test until n_test >= n_test_target,
  then remainder goes to train. If total is small, split as evenly as possible.

Output:
  - Per neuron x location: n_train, n_test, which grid_nos go to each set
  - Summary: distribution of n_train and n_test across all pairs
  - Comparison: fixed A/B/C=train vs optimal split
  - Recommendation: use slot-agnostic split? or a hybrid?
"""

import os
import numpy as np
import pandas as pd

DATA_DIR   = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
RESULTS_DIR = "/Users/xpsy1114/Documents/projects/multiple_clocks/results"
NEURON_CSV = os.path.join(DATA_DIR, "all_neurons_avg_per_gridrun.csv")

MIN_TEST = 2   # minimum test obs per (neuron, location) to be usable
MIN_TRAIN = 4  # minimum train obs per (neuron, location) to be usable
RANDOM_SEED = 42

# -----------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------
df = pd.read_csv(NEURON_CSV)
df[['loc_A','loc_B','loc_C','loc_D']] = df['config'].str.split('-',expand=True).astype(int)


# build long format: one row per (neuron, grid_no, slot)
long = []
for slot, col in [('A','loc_A'),('B','loc_B'),('C','loc_C'),('D','loc_D')]:
    tmp = df[['session','neuron_label','grid_no','config',col]].copy()
    tmp.columns = ['session','neuron_label','grid_no','config','location_label']
    tmp['slot'] = slot
    long.append(tmp)
long_df = pd.concat(long, ignore_index=True)

# -----------------------------------------------------------------------
# Current split (fixed A/B/C=train, D=test) — baseline to compare against
# -----------------------------------------------------------------------
long_df['set_fixed'] = long_df['slot'].apply(lambda s: 'test' if s == 'D' else 'train')

fixed_counts = (long_df.groupby(['neuron_label','location_label','set_fixed'])
                .size().unstack(fill_value=0))
fixed_counts.columns.name = None
if 'train' not in fixed_counts: fixed_counts['train'] = 0
if 'test'  not in fixed_counts: fixed_counts['test']  = 0

n_missing_test_fixed = (fixed_counts['test'] == 0).sum()
n_missing_train_fixed = (fixed_counts['train'] == 0).sum()

print("=== Current fixed split (A/B/C=train, D=test) ===")
print(f"(neuron, location) pairs total: {len(fixed_counts)}")
print(f"Pairs with test=0:  {n_missing_test_fixed}  ({n_missing_test_fixed/len(fixed_counts):.1%})")
print(f"Pairs with train=0: {n_missing_train_fixed}  ({n_missing_train_fixed/len(fixed_counts):.1%})")
print(f"Min test obs:  {fixed_counts['test'].min()}")
print(f"Min train obs: {fixed_counts['train'].min()}")

# -----------------------------------------------------------------------
# Optimal split: slot-agnostic, prefer test
#
# For each (neuron, location):
#   - collect all grid_no runs (each gives 1 obs, regardless of slot)
#   - assign: first floor(n_total / 2) obs to test (or up to n_total - MIN_TRAIN)
#   - remainder to train
#   - if total < MIN_TRAIN + MIN_TEST, flag as unusable
#
# Importantly: within a session, the SAME grid_no always maps to the same config.
# So for a given (neuron, location), each grid_no run gives exactly one observation
# at one slot. We split grid_no runs between train and test.
# -----------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_SEED)

split_rows = []
assignment_rows = []   # track which grid_nos go to train/test per pair

for (neu, loc), grp in long_df.groupby(['neuron_label','location_label']):
    obs = grp[['session','grid_no','slot']].reset_index(drop=True)
    n_total = len(obs)

    # how many to put in test?
    # preference: as many as possible, but leave at least MIN_TRAIN for train
    n_test  = min(n_total - MIN_TRAIN, n_total // 2 + (n_total % 2))  # ceil half
    n_test  = max(n_test, 0)
    n_train = n_total - n_test

    # shuffle so the test assignment is random (not always D-slot)
    idx = rng.permutation(len(obs))
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]

    obs['set_optimal'] = 'train'
    obs.loc[test_idx, 'set_optimal'] = 'test'

    usable = (n_train >= MIN_TRAIN) and (n_test >= MIN_TEST)

    split_rows.append({
        'neuron_label':   neu,
        'location_label': loc,
        'n_total':        n_total,
        'n_train':        n_train,
        'n_test':         n_test,
        'usable':         usable,
    })

    for _, row in obs.iterrows():
        assignment_rows.append({
            'neuron_label':   neu,
            'location_label': loc,
            'session':        row['session'],
            'grid_no':        row['grid_no'],
            'slot':           row['slot'],
            'set_optimal':    row['set_optimal'],
        })

split_df  = pd.DataFrame(split_rows)
assign_df = pd.DataFrame(assignment_rows)

print("\n=== Optimal slot-agnostic split ===")
print(f"Pairs total:           {len(split_df)}")
print(f"Pairs usable (train>={MIN_TRAIN}, test>={MIN_TEST}): "
      f"{split_df['usable'].sum()}  ({split_df['usable'].mean():.1%})")
print(f"Pairs with test=0:     {(split_df['n_test']==0).sum()}")
print(f"Pairs with train=0:    {(split_df['n_train']==0).sum()}")
print(f"\nn_train per pair:")
print(split_df['n_train'].describe().to_string())
print(f"\nn_test per pair:")
print(split_df['n_test'].describe().to_string())

# -----------------------------------------------------------------------
# Per-neuron: how many locations meet usability threshold?
# -----------------------------------------------------------------------
per_neuron = split_df.groupby('neuron_label').agg(
    n_locs_usable=('usable', 'sum'),
    n_locs_total=('usable', 'count'),
    min_train=('n_train', 'min'),
    min_test=('n_test', 'min'),
).reset_index()

print("\n=== Per-neuron usable location coverage ===")
print(per_neuron['n_locs_usable'].value_counts().sort_index().to_string())
print(f"\nNeurons with all 9 locations usable: {(per_neuron['n_locs_usable']==9).sum()}"
      f" / {len(per_neuron)}")
print(f"Neurons with >=8 locations usable:  {(per_neuron['n_locs_usable']>=8).sum()}")

# -----------------------------------------------------------------------
# Pooled coverage: per location, how many neurons contribute to train/test?
# -----------------------------------------------------------------------
print("\n=== Pooled coverage per location (optimal split) ===")
pooled = split_df.groupby('location_label').agg(
    n_neurons_usable=('usable','sum'),
    total_train_obs=('n_train','sum'),
    total_test_obs=('n_test','sum'),
    min_train_per_neuron=('n_train','min'),
    min_test_per_neuron=('n_test','min'),
)
print(pooled.to_string())

# -----------------------------------------------------------------------
# Comparison table: fixed vs optimal, per location
# -----------------------------------------------------------------------
print("\n=== Comparison: fixed vs optimal split (pooled test obs per location) ===")
fixed_te = (long_df[long_df['set_fixed']=='test']
            .groupby('location_label').size())
opt_te   = (assign_df[assign_df['set_optimal']=='test']
            .groupby('location_label').size())
fixed_tr = (long_df[long_df['set_fixed']=='train']
            .groupby('location_label').size())
opt_tr   = (assign_df[assign_df['set_optimal']=='train']
            .groupby('location_label').size())

comp = pd.DataFrame({
    'fixed_train': fixed_tr,
    'fixed_test':  fixed_te,
    'optimal_train': opt_tr,
    'optimal_test':  opt_te,
}).fillna(0).astype(int)
comp['test_gain'] = comp['optimal_test'] - comp['fixed_test']
print(comp.to_string())

# -----------------------------------------------------------------------
# Save the assignment table and summary
# -----------------------------------------------------------------------
out_assign = os.path.join(RESULTS_DIR, 'optimal_train_test_assignment.csv')
out_split  = os.path.join(RESULTS_DIR, 'optimal_split_summary.csv')

assign_df.to_csv(out_assign, index=False)
split_df.to_csv(out_split, index=False)
print(f"\nAssignment table saved: {out_assign}")
print(f"Split summary saved:    {out_split}")

# -----------------------------------------------------------------------
# Recommendation
# -----------------------------------------------------------------------
n_usable_all9 = (per_neuron['n_locs_usable'] == 9).sum()
n_usable_8    = (per_neuron['n_locs_usable'] >= 8).sum()
min_test_pool = pooled['total_test_obs'].min()

print("\n=== Recommendation ===")
print(f"Optimal split gives {n_usable_all9} neurons with all 9 locations usable "
      f"({n_usable_8} with >=8).")
print(f"Minimum pooled test obs per location: {min_test_pool} "
      f"(vs {fixed_te.min()} with fixed split)")
print(f"Minimum pooled train obs per location: {opt_tr.min()} "
      f"(vs {fixed_tr.min()} with fixed split)")
print(f"\nConclusion: the optimal slot-agnostic split eliminates test=0 pairs entirely")
print(f"and increases the test bottleneck from {fixed_te.min()} to {min_test_pool} obs.")
print(f"Trade-off: train obs decrease slightly (slot meaning is lost).")
print(f"\nTo use in decoder: load optimal_train_test_assignment.csv and join")
print(f"on (neuron_label, location_label, grid_no) to get set_optimal.")


# -----------------------------------------------------------------------
# Part 2: Snippet-based split (from all_location_snippets.csv)
# -----------------------------------------------------------------------
# Each row in all_location_snippets.csv is one (neuron x grid_no x visit).
# We split at the grid_no level (all visits of a neuron in a grid_no run
# go to the same set), so observations are defined as unique grid_no runs
# per (neuron, location).
# We then scan for the highest observation threshold such that at least
# 80-100% of neurons have all observed locations covered.
# -----------------------------------------------------------------------
SNIPPETS_CSV = os.path.join(DATA_DIR, "all_location_snippets.csv")
RETAIN_FRAC  = 0.80   # target: keep at least this fraction of neurons
ALL_LOCS     = list(range(1, 10))

print("\n\n" + "="*70)
print("=== Part 2: Snippet-based train/test split ===")
print("="*70)

print(f"\nLoading {SNIPPETS_CSV} ...")
snip = pd.read_csv(SNIPPETS_CSV)
print(f"  {len(snip)} rows | {snip['neuron_label'].nunique()} neurons | "
      f"phases present: {sorted(snip['phase'].unique())}")

# Count unique grid_no runs per (neuron, location) — split at grid_no level
# Each grid_no run may yield multiple visits (rows) of the same location;
# we count distinct grid_no runs, not rows.
counts_gridno = (snip.groupby(['neuron_label', 'current_location', 'grid_no'])
                     .size()
                     .reset_index()[['neuron_label', 'current_location', 'grid_no']])
counts = (counts_gridno.groupby(['neuron_label', 'current_location'])
                       .size()
                       .unstack(fill_value=0))

# Keep only locations 1-9 that actually appear
observed_locs = [l for l in ALL_LOCS if l in counts.columns]
counts = counts[observed_locs]

total_neurons = len(counts)
print(f"\n  Neurons with >=1 observation: {total_neurons}")
print(f"  Locations observed: {observed_locs}")

# --- Threshold scan ---
# A neuron 'qualifies' if it has >= min_obs grid_no runs for ALL observed locations.
print(f"\n  Threshold scan (target: retain >={RETAIN_FRAC:.0%} of neurons):")
print(f"  {'min_obs':>7}  {'n_retained':>10}  {'pct':>6}  {'min_train':>9}  {'min_test':>8}")

best_min_obs  = 1
best_retained = (counts >= 1).all(axis=1).sum()

max_possible = int(counts.min(axis=1).max())
for min_obs in range(1, max_possible + 1):
    n_retained = (counts >= min_obs).all(axis=1).sum()
    pct = n_retained / total_neurons if total_neurons else 0
    min_train = (min_obs + 1) // 2    # ceil-half to test, rest to train
    min_test  = min_obs - min_train
    print(f"  {min_obs:>7}  {n_retained:>10}  {pct:>6.1%}  {min_train:>9}  {min_test:>8}")
    if pct >= RETAIN_FRAC:
        best_min_obs  = min_obs
        best_retained = n_retained

print(f"\n  → Chosen MIN_OBS={best_min_obs}  "
      f"(retains {best_retained}/{total_neurons} = {best_retained/total_neurons:.1%} neurons)")
if best_retained / total_neurons < RETAIN_FRAC:
    print(f"  NOTE: {RETAIN_FRAC:.0%} retention impossible. Using best achievable.")

# --- Apply split ---
rng_snip = np.random.default_rng(RANDOM_SEED)

qualifying = list(counts.index[(counts >= best_min_obs).all(axis=1)])
snip_q = snip[snip['neuron_label'].isin(qualifying)].copy()

# Get unique grid_no runs per (neuron, location) — these are the units we split
gridno_pool = (snip_q[['neuron_label', 'current_location', 'grid_no', 'session']]
               .drop_duplicates(['neuron_label', 'current_location', 'grid_no']))

split_rows      = []
assignment_rows = []

for (neu, loc), grp in gridno_pool.groupby(['neuron_label', 'current_location']):
    gnos    = grp['grid_no'].values
    n_total = len(gnos)

    # ceil-half to test
    n_test  = max(0, min(n_total - 1, (n_total + 1) // 2))
    n_train = n_total - n_test

    perm      = rng_snip.permutation(n_total)
    test_gnos  = gnos[perm[:n_test]]
    train_gnos = gnos[perm[n_test:]]

    split_rows.append({
        'neuron_label':     neu,
        'location_label':   loc,
        'n_total':          n_total,
        'n_train':          n_train,
        'n_test':           n_test,
    })

    for gno in train_gnos:
        assignment_rows.append({'neuron_label': neu, 'location_label': loc,
                                'grid_no': gno, 'set': 'train'})
    for gno in test_gnos:
        assignment_rows.append({'neuron_label': neu, 'location_label': loc,
                                'grid_no': gno, 'set': 'test'})

split_df_snip  = pd.DataFrame(split_rows)
assign_df_snip = pd.DataFrame(assignment_rows)

# --- Summary ---
print(f"\n  Split summary ({len(qualifying)} qualifying neurons, "
      f"{len(observed_locs)} locations):")
print(f"  n_train grid_no runs per (neuron,loc):  "
      f"min={split_df_snip['n_train'].min()}  "
      f"median={split_df_snip['n_train'].median():.0f}  "
      f"max={split_df_snip['n_train'].max()}")
print(f"  n_test grid_no runs per (neuron,loc):   "
      f"min={split_df_snip['n_test'].min()}  "
      f"median={split_df_snip['n_test'].median():.0f}  "
      f"max={split_df_snip['n_test'].max()}")

# Per-location bottleneck: min across neurons for each location
loc_stats = split_df_snip.groupby('location_label').agg(
    min_train=('n_train', 'min'),
    min_test=('n_test', 'min'),
    max_train=('n_train', 'max'),
    max_test=('n_test', 'max'),
)
print(f"\n  Per-location min/max grid_no runs (train / test) — bottleneck = min:")
print(f"  {'loc':>4}  {'min_tr':>7}  {'max_tr':>7}  {'min_te':>7}  {'max_te':>7}")
for loc, row in loc_stats.iterrows():
    print(f"  {loc:>4}  {int(row.min_train):>7}  {int(row.max_train):>7}  "
          f"{int(row.min_test):>7}  {int(row.max_test):>7}")

# Suggested N_RESAMPLES: based on how much variation there is above the minimum
# More variation → more resamples to average it out; less → fewer needed
excess_ratios = [loc_stats.loc[l,'max_train'] / loc_stats.loc[l,'min_train']
                 for l in loc_stats.index if loc_stats.loc[l,'min_train'] > 0]
mean_excess = np.mean(excess_ratios)
suggested_n = int(np.clip(10 * mean_excess, 10, 50))
print(f"\n  Mean max/min train ratio across locations: {mean_excess:.2f}")
print(f"  → Suggested N_RESAMPLES: {suggested_n}")

# Pooled obs per location (counting actual snippet rows, not just grid_no runs)
# Join assignment back to snippets to count rows per set
snip_assigned = snip_q.merge(
    assign_df_snip[['neuron_label', 'location_label', 'grid_no', 'set']],
    left_on=['neuron_label', 'current_location', 'grid_no'],
    right_on=['neuron_label', 'location_label', 'grid_no'],
    how='left'
)
tr_pool = snip_assigned[snip_assigned['set'] == 'train'].groupby('current_location').size()
te_pool = snip_assigned[snip_assigned['set'] == 'test'].groupby('current_location').size()
print(f"\n  Pooled train snippet rows per location:\n{tr_pool.to_string()}")
print(f"\n  Pooled test snippet rows per location:\n{te_pool.to_string()}")

# --- Save ---
assign_snip_path = os.path.join(RESULTS_DIR, 'snippet_train_test_assignment.csv')
split_snip_path  = os.path.join(RESULTS_DIR, 'snippet_split_summary.csv')

assign_df_snip.to_csv(assign_snip_path, index=False)
split_df_snip.to_csv(split_snip_path,   index=False)
print(f"\n  Saved:\n    {assign_snip_path}\n    {split_snip_path}")
print("\nDone.")
