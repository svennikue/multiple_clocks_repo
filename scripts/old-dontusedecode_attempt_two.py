#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
DATA_DIR    = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
RESULTS_DIR = "/Users/xpsy1114/Documents/projects/multiple_clocks/results"
NEURON_CSV  = os.path.join(DATA_DIR, "all_neurons_avg_per_gridrun.csv")

MIN_REPS_TRAIN = 4
N_BINS      = 360
REW_BINS    = {'A': 89, 'B': 179, 'C': 269, 'D': 359}
DECODE_LOCS = list(range(2, 10))   # 2..9 only
RANDOM_SEED = 42

TEST_MODE = False
if TEST_MODE:
    HALF_WIN            = 10
    N_RESAMPLES         = 3
    N_SHUFFLE_RESAMPLES = 2
    print("*** TEST MODE: reduced offsets and resamples ***")
else:
    HALF_WIN            = 20
    N_RESAMPLES         = 50
    N_SHUFFLE_RESAMPLES = 20

# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------
df = pd.read_csv(NEURON_CSV)
df[['loc_A', 'loc_B', 'loc_C', 'loc_D']] = (
    df['config'].str.split('-', expand=True).astype(int)
)

print(f"Loaded {len(df)} rows | {df['neuron_label'].nunique()} neurons | "
      f"{df['config'].nunique()} configs | {df['grid_no'].nunique()} grid runs")

# -----------------------------------------------------------------------
# Build long-format table for one offset
# -----------------------------------------------------------------------
def build_long_for_offset(offset):
    rows = []
    for slot, loc_col in [('A','loc_A'), ('B','loc_B'), ('C','loc_C'), ('D','loc_D')]:
        abs_bin = (REW_BINS[slot] + offset) % N_BINS
        bin_col = f'bin_{abs_bin:03d}'
        tmp = df[['session', 'neuron_label', 'grid_no', 'config', loc_col]].copy()
        tmp.columns = ['session', 'neuron_label', 'grid_no', 'config', 'location_label']
        tmp['slot']     = slot
        tmp['set']      = 'test' if slot == 'D' else 'train'
        tmp['activity'] = df[bin_col].values
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    return out[out['location_label'].isin(DECODE_LOCS)].copy()

# -----------------------------------------------------------------------
# Select neurons
# Strong requirement on train, weak requirement on test
# -----------------------------------------------------------------------
def select_retained_neurons(min_reps_train=MIN_REPS_TRAIN):
    long0 = build_long_for_offset(0)
    train0 = long0[long0['set'] == 'train']
    test0  = long0[long0['set'] == 'test']

    train_rep = (train0.groupby(['neuron_label', 'location_label'])
                 .size().unstack(fill_value=0))
    test_rep = (test0.groupby(['neuron_label', 'location_label'])
                .size().unstack(fill_value=0))

    for loc in DECODE_LOCS:
        if loc not in train_rep.columns:
            train_rep[loc] = 0
        if loc not in test_rep.columns:
            test_rep[loc] = 0

    train_rep = train_rep[DECODE_LOCS]
    test_rep  = test_rep[DECODE_LOCS]

    # require good train support for all decode locations
    keep_train = (train_rep >= min_reps_train).all(axis=1)

    # require at least one test observation somewhere
    keep_test_any = (test_rep > 0).any(axis=1)

    kept = sorted(train_rep.index[keep_train & keep_test_any])

    print(f"\n--- Neuron inclusion for 8-way decode (locs 2..9) ---")
    print(f"MIN_REPS_TRAIN={min_reps_train}")
    print(f"Retained neurons: {len(kept)} / {df['neuron_label'].nunique()}")

    if len(kept) == 0:
        raise RuntimeError("No neurons satisfy the relaxed inclusion rule.")

    print("\nTrain obs per decoded location (pooled across retained neurons):")
    print(train_rep.loc[kept].sum(axis=0).to_string())

    print("\nTest obs per decoded location (pooled across retained neurons):")
    print(test_rep.loc[kept].sum(axis=0).to_string())

    print("\nFraction of retained neurons with ZERO test obs per location:")
    print(((test_rep.loc[kept] == 0).mean(axis=0)).round(3).to_string())

    print("\nFraction of retained neurons with >=1 test obs per location:")
    print(((test_rep.loc[kept] > 0).mean(axis=0)).round(3).to_string())

    return kept, train_rep.loc[kept], test_rep.loc[kept]

# -----------------------------------------------------------------------
# Build observation arrays
# train can be dense by design; test can be sparse
# -----------------------------------------------------------------------
def build_obs_arrays(long_df, retained_neurons):
    n_neurons = len(retained_neurons)
    neu_to_idx = {n: i for i, n in enumerate(retained_neurons)}
    loc_to_idx = {loc: i for i, loc in enumerate(DECODE_LOCS)}

    train_df = long_df[(long_df['set'] == 'train') & (long_df['neuron_label'].isin(retained_neurons))]
    test_df  = long_df[(long_df['set'] == 'test')  & (long_df['neuron_label'].isin(retained_neurons))]

    train_arr = [[np.array([], dtype=float) for _ in range(n_neurons)] for _ in DECODE_LOCS]
    test_arr  = [[np.array([], dtype=float) for _ in range(n_neurons)] for _ in DECODE_LOCS]

    for (neu, loc), grp in train_df.groupby(['neuron_label', 'location_label']):
        vals = grp['activity'].dropna().values
        if len(vals):
            train_arr[loc_to_idx[loc]][neu_to_idx[neu]] = vals

    for (neu, loc), grp in test_df.groupby(['neuron_label', 'location_label']):
        vals = grp['activity'].dropna().values
        if len(vals):
            test_arr[loc_to_idx[loc]][neu_to_idx[neu]] = vals

    # per-neuron mean over all train observations (used only for test fallback)
    neu_train_mean = np.zeros(n_neurons, dtype=float)
    for n_idx in range(n_neurons):
        vals = []
        for loc_idx in range(len(DECODE_LOCS)):
            arr = train_arr[loc_idx][n_idx]
            if len(arr):
                vals.append(arr)
        if len(vals):
            neu_train_mean[n_idx] = np.concatenate(vals).mean()
        else:
            neu_train_mean[n_idx] = 0.0

    return train_arr, test_arr, neu_train_mean

# -----------------------------------------------------------------------
# Choose pseudo-trials per class
# Based on pooled class counts, not per-neuron minima
# -----------------------------------------------------------------------
def compute_n_per_class(long_df, retained_neurons, default_train=4):
    train_df = long_df[(long_df['set'] == 'train') & (long_df['neuron_label'].isin(retained_neurons))]
    test_df  = long_df[(long_df['set'] == 'test')  & (long_df['neuron_label'].isin(retained_neurons))]

    tr_counts = train_df.groupby('location_label').size().reindex(DECODE_LOCS, fill_value=0)
    te_counts = test_df.groupby('location_label').size().reindex(DECODE_LOCS, fill_value=0)

    # training pseudo-trials can be fixed conservatively
    n_train_per_class = min(default_train, int(tr_counts.min()))

    # test pseudo-trials can be larger or smaller; balanced accuracy does not require equality
    # but for simplicity keep them matched to the train count by default
    n_test_per_class = min(n_train_per_class, int(te_counts.min()))

    print("\nPooled observations per decoded location:")
    print("Train:")
    print(tr_counts.to_string())
    print("Test:")
    print(te_counts.to_string())

    print(f"\nUsing n_train_per_class={n_train_per_class}")
    print(f"Using n_test_per_class={n_test_per_class}")

    if n_train_per_class < 1 or n_test_per_class < 1:
        raise RuntimeError("Not enough pooled observations to form pseudo-trials.")

    return n_train_per_class, n_test_per_class

# -----------------------------------------------------------------------
# Build pseudo-trials
# Train must be observed.
# Test may be sparse; missing test entries are filled with train mean.
# -----------------------------------------------------------------------
def build_pseudotrials(train_arr, test_arr, neu_train_mean,
                       n_train_per_class, n_test_per_class, rng):
    n_neurons = len(train_arr[0])
    n_locs = len(DECODE_LOCS)

    X_train = np.empty((n_locs * n_train_per_class, n_neurons))
    X_test  = np.empty((n_locs * n_test_per_class, n_neurons))
    y_train = np.repeat(DECODE_LOCS, n_train_per_class)
    y_test  = np.repeat(DECODE_LOCS, n_test_per_class)

    missing_test_mask = np.zeros_like(X_test, dtype=bool)

    for loc_idx, loc in enumerate(DECODE_LOCS):
        tr0 = loc_idx * n_train_per_class
        tr1 = tr0 + n_train_per_class
        te0 = loc_idx * n_test_per_class
        te1 = te0 + n_test_per_class

        for n_idx in range(n_neurons):
            tr_obs = train_arr[loc_idx][n_idx]
            te_obs = test_arr[loc_idx][n_idx]

            # train side: should be available by train inclusion rule
            if len(tr_obs) < 1:
                raise RuntimeError(f"No train obs for loc={loc}, neuron_idx={n_idx}")

            X_train[tr0:tr1, n_idx] = rng.choice(
                tr_obs, size=n_train_per_class, replace=(len(tr_obs) < n_train_per_class)
            )

            # test side: allow sparse coverage, fallback to train mean
            if len(te_obs) >= 1:
                X_test[te0:te1, n_idx] = rng.choice(
                    te_obs, size=n_test_per_class, replace=(len(te_obs) < n_test_per_class)
                )
            else:
                X_test[te0:te1, n_idx] = neu_train_mean[n_idx]
                missing_test_mask[te0:te1, n_idx] = True

    return X_train, y_train, X_test, y_test, missing_test_mask

# -----------------------------------------------------------------------
# Fit decoder
# -----------------------------------------------------------------------
def fit_decoder(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    Xtr_z  = scaler.fit_transform(X_train)
    Xte_z  = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        multi_class='multinomial',
        random_state=RANDOM_SEED
    )
    clf.fit(Xtr_z, y_train)
    y_pred = clf.predict(Xte_z)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    raw_acc = (y_pred == y_test).mean()
    return bal_acc, raw_acc, y_pred

# -----------------------------------------------------------------------
# Missingness-only control
# Features are just whether a neuron had any test observation for that class
# -----------------------------------------------------------------------
def run_missingness_control(retained_neurons):
    long0 = build_long_for_offset(0)
    train0 = long0[(long0['set'] == 'train') & (long0['neuron_label'].isin(retained_neurons))]
    test0  = long0[(long0['set'] == 'test')  & (long0['neuron_label'].isin(retained_neurons))]

    train_rep = (train0.groupby(['neuron_label', 'location_label']).size()
                 .unstack(fill_value=0))
    test_rep = (test0.groupby(['neuron_label', 'location_label']).size()
                .unstack(fill_value=0))

    for loc in DECODE_LOCS:
        if loc not in train_rep.columns:
            train_rep[loc] = 0
        if loc not in test_rep.columns:
            test_rep[loc] = 0

    train_rep = train_rep.reindex(index=retained_neurons, columns=DECODE_LOCS, fill_value=0)
    test_rep  = test_rep.reindex(index=retained_neurons, columns=DECODE_LOCS, fill_value=0)

    Xtr = (train_rep > 0).astype(float).T.values
    Xte = (test_rep > 0).astype(float).T.values
    y = np.array(DECODE_LOCS)

    scaler = StandardScaler()
    Xtr_z = scaler.fit_transform(Xtr)
    Xte_z = scaler.transform(Xte)

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        multi_class='multinomial',
        random_state=RANDOM_SEED
    )
    clf.fit(Xtr_z, y)
    y_pred = clf.predict(Xte_z)
    ba = balanced_accuracy_score(y, y_pred)

    print("\n--- Missingness-only control ---")
    print(f"Balanced accuracy from coverage structure alone: {ba:.3f}")
    print(f"Chance for 8-way = {1/8:.3f}")
    return ba

# -----------------------------------------------------------------------
# Transfer decoding
# -----------------------------------------------------------------------
def run_transfer_decoding(retained_neurons):
    offsets = np.arange(-HALF_WIN, HALF_WIN)
    n_off = len(offsets)

    bal_acc = np.zeros((N_RESAMPLES, n_off))
    raw_acc = np.zeros((N_RESAMPLES, n_off))
    miss_frac = np.zeros((N_RESAMPLES, n_off))

    print(f"\n--- Transfer decoding: A/B/C -> D, locations 2..9 only ---")

    for t_idx, offset in enumerate(offsets):
        long_df = build_long_for_offset(offset)
        train_arr, test_arr, neu_train_mean = build_obs_arrays(long_df, retained_neurons)
        n_train_per_class, n_test_per_class = compute_n_per_class(long_df, retained_neurons)

        for r in range(N_RESAMPLES):
            rng = np.random.default_rng(RANDOM_SEED + 1000 * t_idx + r)
            Xtr, ytr, Xte, yte, miss_mask = build_pseudotrials(
                train_arr, test_arr, neu_train_mean,
                n_train_per_class, n_test_per_class, rng
            )
            ba, ra, _ = fit_decoder(Xtr, ytr, Xte, yte)
            bal_acc[r, t_idx] = ba
            raw_acc[r, t_idx] = ra
            miss_frac[r, t_idx] = miss_mask.mean()

        if (t_idx + 1) % 5 == 0:
            print(f"  {t_idx+1}/{n_off} offsets done")

    return offsets, bal_acc, raw_acc, miss_frac

# -----------------------------------------------------------------------
# Shuffle control
# -----------------------------------------------------------------------
def run_shuffle_control(retained_neurons):
    offsets = np.arange(-HALF_WIN, HALF_WIN)
    n_off = len(offsets)
    shuf_acc = np.zeros((N_SHUFFLE_RESAMPLES, n_off))

    for t_idx, offset in enumerate(offsets):
        long_df = build_long_for_offset(offset)
        train_arr, test_arr, neu_train_mean = build_obs_arrays(long_df, retained_neurons)
        n_train_per_class, n_test_per_class = compute_n_per_class(long_df, retained_neurons)

        for r in range(N_SHUFFLE_RESAMPLES):
            rng = np.random.default_rng(RANDOM_SEED + 1000 * t_idx + r + 99999)
            Xtr, ytr, Xte, yte, _ = build_pseudotrials(
                train_arr, test_arr, neu_train_mean,
                n_train_per_class, n_test_per_class, rng
            )

            scaler = StandardScaler()
            Xtr_z = scaler.fit_transform(Xtr)
            Xte_z = scaler.transform(Xte)

            clf = LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver='lbfgs',
                multi_class='multinomial',
                random_state=RANDOM_SEED
            )
            clf.fit(Xtr_z, rng.permutation(ytr))
            shuf_acc[r, t_idx] = balanced_accuracy_score(yte, clf.predict(Xte_z))

    return offsets, shuf_acc

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
print(f"\nSettings: MIN_REPS_TRAIN={MIN_REPS_TRAIN}, DECODE_LOCS={DECODE_LOCS}, "
      f"N_RESAMPLES={N_RESAMPLES}, HALF_WIN={HALF_WIN}")

retained_neurons, train_rep_kept, test_rep_kept = select_retained_neurons(MIN_REPS_TRAIN)
print(f"Retained neuron fraction: {len(retained_neurons) / df['neuron_label'].nunique():.3f}")

run_missingness_control(retained_neurons)

print("\n--- Sanity check at offset 0 ---")
long0 = build_long_for_offset(0)
train_arr0, test_arr0, neu_train_mean0 = build_obs_arrays(long0, retained_neurons)
n_train_per_class0, n_test_per_class0 = compute_n_per_class(long0, retained_neurons)

rng0 = np.random.default_rng(RANDOM_SEED)
Xtr0, ytr0, Xte0, yte0, miss0 = build_pseudotrials(
    train_arr0, test_arr0, neu_train_mean0,
    n_train_per_class0, n_test_per_class0, rng0
)

print(f"X_train shape: {Xtr0.shape}")
print(f"X_test shape:  {Xte0.shape}")
print(f"Any NaN train? {np.isnan(Xtr0).any()}")
print(f"Any NaN test?  {np.isnan(Xte0).any()}")
print(f"Test imputation fraction at offset 0: {miss0.mean():.3f}")

ba0, ra0, ypred0 = fit_decoder(Xtr0, ytr0, Xte0, yte0)
print(f"offset=0: balanced_acc={ba0:.3f}, raw_acc={ra0:.3f}, chance={1/8:.3f}")

print("\nPer-class recall at offset 0:")
for loc in DECODE_LOCS:
    mask = (yte0 == loc)
    recall = (ypred0[mask] == loc).mean()
    print(f"  loc {loc}: {recall:.3f} (n={mask.sum()})")

offsets, bal_acc_transfer, raw_acc_transfer, miss_frac = run_transfer_decoding(retained_neurons)
offsets, shuf_acc = run_shuffle_control(retained_neurons)

tr_mean = bal_acc_transfer.mean(axis=0)
tr_sd   = bal_acc_transfer.std(axis=0)
sh_mean = shuf_acc.mean(axis=0)
miss_mean = miss_frac.mean(axis=0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(offsets, tr_mean, 'k-o', ms=3, label='transfer 8-way')
ax.fill_between(offsets, tr_mean - tr_sd, tr_mean + tr_sd, alpha=0.2, color='k')
ax.plot(offsets, sh_mean, color='gray', label='shuffle')
ax.axhline(1/8, color='lightgray', linestyle=':', label='chance')
ax.axvline(0, color='blue', linestyle=':', alpha=0.5)
ax.set_xlabel('Offset from reward bin')
ax.set_ylabel('Balanced accuracy')
ax.set_title(f'Pseudo-population decoder, locs 2..9 only\n'
             f'({len(retained_neurons)} neurons, relaxed test coverage)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'decode_curr_rew_loc_8way_relaxed_test.png'), dpi=150)

pd.DataFrame({
    'offset': offsets,
    'transfer_bal_acc_mean': tr_mean,
    'transfer_bal_acc_sd': tr_sd,
    'shuffle_mean': sh_mean,
    'mean_test_imputation_fraction': miss_mean
}).to_csv(os.path.join(RESULTS_DIR, 'decode_curr_rew_loc_8way_relaxed_test.csv'), index=False)

print("\nDone.")