#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pseudo-population decoder: current reward location.

Train on A/B/C reward timepoints, test on D reward timepoint.
Input: all_neurons_avg_per_gridrun.csv  (one row per neuron x grid_no run)

Key design choices:
  - Fixed neuron set across all offsets (determined once from train pool)
  - Multiple pseudo-trials per class, NOT single mean per location
  - Pseudo-trials built by sampling one observation per neuron per trial
  - Balanced classes (same n pseudo-trials per location)
  - Repeated resampling (N_RESAMPLES) to average over sampling variability
  - Train: A+B+C slots  |  Test: D slot
  - Metric: balanced accuracy (primary), raw accuracy, per-class recall
  - Shuffle control + missingness-only control
  - Within-train decoding control vs transfer (A/B/C -> D)

Missing data in test:
  - By design, neurons only contribute to locations they actually observed.
  - Test (D) has high missingness for some locations (up to ~75% for loc 1).
  - Missing test entries are imputed with the neuron's training mean.
  - This is reported explicitly in diagnostics.
  - Train has zero missing (enforced by inclusion criterion).
"""

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

MIN_REPS  = 4    # min train observations per location per neuron
MIN_LOCS  = 9    # min locations a neuron must cover in train
N_BINS    = 360
REW_BINS  = {'A': 89, 'B': 179, 'C': 269, 'D': 359}
ALL_LOCS  = list(range(1, 10))
RANDOM_SEED = 42

# -----------------------------------------------------------------------
# TEST_MODE: fast settings for checking the pipeline end-to-end.
# Set TEST_MODE = False for the real analysis.
# -----------------------------------------------------------------------
TEST_MODE = True

if TEST_MODE:
    HALF_WIN            = 10    # only 10 offsets (-5..+4)
    N_RESAMPLES         = 3
    N_SHUFFLE_RESAMPLES = 2
    print("*** TEST MODE: reduced offsets and resamples ***")
else:
    HALF_WIN            = 20   # full 40 offsets
    N_RESAMPLES         = 50
    N_SHUFFLE_RESAMPLES = 20

# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------
df = pd.read_csv(NEURON_CSV)
df[['loc_A', 'loc_B', 'loc_C', 'loc_D']] = (
    df['config'].str.split('-', expand=True).astype(int)
)
BIN_COLS = [f'bin_{b:03d}' for b in range(N_BINS)]
print(f"Loaded {len(df)} rows | {df['neuron_label'].nunique()} neurons | "
      f"{df['config'].nunique()} configs | {df['grid_no'].nunique()} grid runs")


# -----------------------------------------------------------------------
# Build long-format table for a given offset
# -----------------------------------------------------------------------
def build_long_for_offset(offset):
    """
    Returns long-format DataFrame with columns:
      session, neuron_label, grid_no, config, location_label, slot, set, activity
    activity = value at (reward_bin[slot] + offset) % N_BINS for each slot.
    """
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
    return pd.concat(rows, ignore_index=True)


# -----------------------------------------------------------------------
# Select retained neurons (fixed across all offsets, based on train pool)
# -----------------------------------------------------------------------
def select_retained_neurons(min_reps=MIN_REPS, min_locs=MIN_LOCS):
    """
    Uses offset=0 to determine train repeat counts (activity values don't matter here,
    only structure). Returns sorted list of retained neuron labels.
    """
    long0 = build_long_for_offset(0)
    train0 = long0[long0['set'] == 'train']
    test0  = long0[long0['set'] == 'test']

    train_rep = (train0.groupby(['neuron_label', 'location_label'])
                 .size().unstack(fill_value=0))
    test_rep  = (test0.groupby(['neuron_label', 'location_label'])
                 .size().unstack(fill_value=0))
    for loc in ALL_LOCS:
        if loc not in train_rep.columns: train_rep[loc] = 0
        if loc not in test_rep.columns:  test_rep[loc]  = 0
    train_rep = train_rep[ALL_LOCS]
    test_rep  = test_rep[ALL_LOCS]

    # neuron must have >= min_reps train obs for >= min_locs locations
    n_locs_ok = (train_rep >= min_reps).sum(axis=1)
    kept = sorted(n_locs_ok[n_locs_ok >= min_locs].index)

    # population-level test coverage check
    te_covered = (test_rep.loc[kept] > 0).any(axis=0)
    missing_test = [l for l in ALL_LOCS if not te_covered.get(l, False)]

    print(f"\n--- Neuron inclusion (min_reps={min_reps}, min_locs={min_locs}) ---")
    print(f"Retained: {len(kept)} / {df['neuron_label'].nunique()} neurons")
    if missing_test:
        print(f"WARNING: test locations not covered at population level: {missing_test}")
        print("9-way decoding not feasible.")
        raise RuntimeError("Test coverage insufficient.")
    else:
        print("9-way pooled coverage: OK in both train and test.")

    # report missingness in test for retained neurons
    te = test_rep.loc[kept]
    tr = train_rep.loc[kept]
    
    print("\nPooled train obs per location:")
    print(tr.sum(axis=0).to_string())
    print("\nPooled test obs per location:")
    print(te.sum(axis=0).to_string())
    print("\nFraction of retained neurons with ZERO test obs per location:")
    print(((te == 0).sum(axis=0) / len(kept)).round(3).to_string())

    
    # per-neuron coverage distributions
    print("\nPer-neuron train location coverage:")
    print((tr > 0).sum(axis=1).value_counts().sort_index().to_string())
    print("\nPer-neuron test location coverage:")
    print((te > 0).sum(axis=1).value_counts().sort_index().to_string())
    # import pdb; pdb.set_trace() 
    return kept


# -----------------------------------------------------------------------
# Pre-build observation arrays (called once per offset, outside resample loop)
# -----------------------------------------------------------------------
def build_obs_arrays(long_df, retained_neurons):
    """
    Pre-indexes all observations into numpy arrays for fast resampling.

    Returns:
      train_arr[loc_idx, neu_idx] -> 1-D array of non-NaN activity values
      test_arr [loc_idx, neu_idx] -> 1-D array of non-NaN activity values  (may be empty)
      neu_train_mean              -> 1-D array shape (n_neurons,) for imputation
    """
    n_neurons  = len(retained_neurons)
    neu_to_idx = {n: i for i, n in enumerate(retained_neurons)}

    train_df = long_df[long_df['set'] == 'train']
    test_df  = long_df[long_df['set'] == 'test']

    # restrict to retained neurons once
    tr = train_df[train_df['neuron_label'].isin(retained_neurons)]
    te = test_df[ test_df['neuron_label'].isin(retained_neurons)]

    # index per (location, neuron) -> values array
    train_arr = [[np.array([], dtype=float)] * n_neurons for _ in ALL_LOCS]
    test_arr  = [[np.array([], dtype=float)] * n_neurons for _ in ALL_LOCS]

    for (neu, loc), grp in tr.groupby(['neuron_label', 'location_label']):
        if neu in neu_to_idx:
            vals = grp['activity'].dropna().values
            if len(vals): train_arr[loc - 1][neu_to_idx[neu]] = vals

    for (neu, loc), grp in te.groupby(['neuron_label', 'location_label']):
        if neu in neu_to_idx:
            vals = grp['activity'].dropna().values
            if len(vals): test_arr[loc - 1][neu_to_idx[neu]] = vals

    # per-neuron mean over all train obs (for imputation)
    neu_train_mean = np.zeros(n_neurons)
    for n_idx in range(n_neurons):
        all_vals = np.concatenate([train_arr[l][n_idx] for l in range(9)
                                   if len(train_arr[l][n_idx])])
        if len(all_vals):
            neu_train_mean[n_idx] = all_vals.mean()

    return train_arr, test_arr, neu_train_mean


def build_pseudotrials(train_arr, test_arr, neu_train_mean, n_per_class, rng):
    """
    Fast pseudo-trial construction using pre-built observation arrays.

    Sampling: without replacement if pool >= n_per_class, else with replacement.
    Missing test entries imputed with per-neuron train mean.

    Returns: X_train (9*n_per_class, n_neurons), y_train,
             X_test  (9*n_per_class, n_neurons), y_test
    """
    n_neurons = len(neu_train_mean)
    n_locs    = len(ALL_LOCS)
    n_total   = n_locs * n_per_class

    X_train = np.empty((n_total, n_neurons))
    X_test  = np.empty((n_total, n_neurons))
    y       = np.repeat(ALL_LOCS, n_per_class)

    for loc_idx in range(n_locs):
        r0 = loc_idx * n_per_class
        r1 = r0 + n_per_class
        for n_idx in range(n_neurons):
            # --- train ---
            obs = train_arr[loc_idx][n_idx]
            if len(obs):
                X_train[r0:r1, n_idx] = rng.choice(
                    obs, size=n_per_class, replace=(len(obs) < n_per_class))
            else:
                X_train[r0:r1, n_idx] = neu_train_mean[n_idx]

            # --- test ---
            obs = test_arr[loc_idx][n_idx]
            if len(obs):
                X_test[r0:r1, n_idx] = rng.choice(
                    obs, size=n_per_class, replace=(len(obs) < n_per_class))
            else:
                X_test[r0:r1, n_idx] = neu_train_mean[n_idx]

    return X_train, y, X_test, y


# -----------------------------------------------------------------------
# Determine n_per_class: limited by min pooled test obs across locations
# -----------------------------------------------------------------------
def compute_n_per_class(retained_neurons):
    """
    Conservative choice: n_per_class = min pooled test obs across locations.
    Also report train bottleneck.
    """
    long0 = build_long_for_offset(0)
    train0 = long0[long0['set'] == 'train']
    test0  = long0[long0['set'] == 'test']

    tr = (train0[train0['neuron_label'].isin(retained_neurons)]
          .groupby('location_label').size())
    te = (test0[test0['neuron_label'].isin(retained_neurons)]
          .groupby('location_label').size())

    n_train_lim = int(tr.min())
    n_test_lim  = int(te.min())
    n_per_class = n_test_lim   # limited by test bottleneck
    
    # import pdb; pdb.set_trace() 
    print(f"\nPooled obs per location — train min: {n_train_lim} "
          f"(loc {tr.idxmin()}), test min: {n_test_lim} (loc {te.idxmin()})")
    print(f"n_per_class (pseudo-trials per location per set): {n_per_class}")
    print(f"  → {n_per_class * 9} total train pseudo-trials, "
          f"{n_per_class * 9} total test pseudo-trials per resample")
    return n_per_class


# -----------------------------------------------------------------------
# Missingness fraction diagnostics
# -----------------------------------------------------------------------

# nice to have but not required for the analysis.


# def report_missingness(retained_neurons):
#     import pdb; pdb.set_trace() 
#     long0 = build_long_for_offset(0)
#     test0  = long0[long0['set'] == 'test']
#     test_rep = (test0.groupby(['neuron_label', 'location_label']).size()
#                 .unstack(fill_value=0))
#     for loc in ALL_LOCS:
#         if loc not in test_rep.columns: test_rep[loc] = 0
#     test_rep = test_rep.reindex(retained_neurons, fill_value=0)[ALL_LOCS]
#     total = test_rep.size
#     missing = (test_rep == 0).sum().sum()
#     print(f"\nTest missingness in retained neuron set: "
#           f"{missing}/{total} entries = {missing/total:.3f}")
#     print("Per-location fraction missing in test:")
#     print(((test_rep == 0).sum() / len(retained_neurons)).round(3).to_string())
#     print("Imputation method: per-neuron mean over all train observations")


# -----------------------------------------------------------------------
# Fit and evaluate one decoder
# -----------------------------------------------------------------------
def fit_decoder(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    Xtr_z  = scaler.fit_transform(X_train)
    Xte_z  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                             multi_class='multinomial', random_state=RANDOM_SEED)
    clf.fit(Xtr_z, y_train)
    y_pred = clf.predict(Xte_z)

    bal_acc  = balanced_accuracy_score(y_test, y_pred)
    raw_acc  = (y_pred == y_test).mean()
    return bal_acc, raw_acc, y_pred, scaler, clf


# -----------------------------------------------------------------------
# Run transfer decoding (train A/B/C -> test D) across offsets
# -----------------------------------------------------------------------
def run_transfer_decoding(retained_neurons, n_per_class):
    offsets = np.arange(-HALF_WIN, HALF_WIN)
    n_off   = len(offsets)
    bal_acc = np.zeros((N_RESAMPLES, n_off))
    raw_acc = np.zeros((N_RESAMPLES, n_off))

    print(f"\n--- Transfer decoding (A/B/C -> D) | {N_RESAMPLES} resamples ---")
    for t_idx, offset in enumerate(offsets):
        long = build_long_for_offset(offset)
        tr_arr, te_arr, neu_mean = build_obs_arrays(long, retained_neurons)
        for r in range(N_RESAMPLES):
            rng_r = np.random.default_rng(RANDOM_SEED + t_idx * 1000 + r)
            Xtr, ytr, Xte, yte = build_pseudotrials(tr_arr, te_arr, neu_mean,
                                                     n_per_class, rng_r)
            ba, ra, _, _, _ = fit_decoder(Xtr, ytr, Xte, yte)
            bal_acc[r, t_idx] = ba
            raw_acc[r, t_idx] = ra

        if (t_idx + 1) % 5 == 0:
            print(f"  {t_idx+1}/{n_off} offsets done")

    return offsets, bal_acc, raw_acc


# -----------------------------------------------------------------------
# Run shuffle control
# -----------------------------------------------------------------------
def run_shuffle_control(retained_neurons, n_per_class):
    offsets  = np.arange(-HALF_WIN, HALF_WIN)
    n_off    = len(offsets)
    shuf_acc = np.zeros((N_SHUFFLE_RESAMPLES, n_off))

    print(f"\n--- Shuffle control | {N_SHUFFLE_RESAMPLES} resamples ---")
    for t_idx, offset in enumerate(offsets):
        long = build_long_for_offset(offset)
        tr_arr, te_arr, neu_mean = build_obs_arrays(long, retained_neurons)
        for r in range(N_SHUFFLE_RESAMPLES):
            rng_r = np.random.default_rng(RANDOM_SEED + t_idx * 1000 + r + 99999)
            Xtr, ytr, Xte, yte = build_pseudotrials(tr_arr, te_arr, neu_mean,
                                                     n_per_class, rng_r)
            scaler = StandardScaler()
            Xtr_z  = scaler.fit_transform(Xtr)
            Xte_z  = scaler.transform(Xte)
            clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                     multi_class='multinomial',
                                     random_state=RANDOM_SEED)
            clf.fit(Xtr_z, rng_r.permutation(ytr))
            shuf_acc[r, t_idx] = balanced_accuracy_score(yte, clf.predict(Xte_z))

        if (t_idx + 1) % 5 == 0:
            print(f"  {t_idx+1}/{n_off} offsets done")

    return offsets, shuf_acc


# -----------------------------------------------------------------------
# Run missingness-only control
# Features = 1 if neuron has ANY test obs for this location, else 0.
# Checks whether neuron availability pattern alone drives decoding.
# -----------------------------------------------------------------------

# i dont think i need this either. keep because these might be useful controls later on.

# def run_missingness_control(retained_neurons, n_per_class):
#     long0 = build_long_for_offset(0)
#     test0 = long0[long0['set'] == 'test']
#     test_rep = (test0.groupby(['neuron_label','location_label']).size()
#                 .unstack(fill_value=0))
#     for loc in ALL_LOCS:
#         if loc not in test_rep.columns: test_rep[loc] = 0
#     test_rep = test_rep.reindex(retained_neurons, fill_value=0)[ALL_LOCS]

#     # coverage matrix: rows = locations (9), cols = neurons (n_neurons)
#     # test_rep is (n_neurons x 9), so transpose to get (9 x n_neurons)
#     X_cov = (test_rep[ALL_LOCS] > 0).astype(float).T.values   # (9, n_neurons)

#     # train coverage: same shape (9, n_neurons)
#     train0 = long0[long0['set'] == 'train']
#     train_rep = (train0.groupby(['neuron_label', 'location_label']).size()
#                  .unstack(fill_value=0))
#     for loc in ALL_LOCS:
#         if loc not in train_rep.columns: train_rep[loc] = 0
#     train_rep = train_rep.reindex(retained_neurons, fill_value=0)
#     X_cov_tr = (train_rep[ALL_LOCS] > 0).astype(float).T.values   # (9, n_neurons)
#     # import pdb; pdb.set_trace() 
    
#     y = np.array(ALL_LOCS)
#     scaler = StandardScaler()
#     Xtr_z  = scaler.fit_transform(X_cov_tr)
#     Xte_z  = scaler.transform(X_cov)
#     clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
#                              multi_class='multinomial', random_state=RANDOM_SEED)
#     clf.fit(Xtr_z, y)
#     ypred = clf.predict(Xte_z)
#     ba = balanced_accuracy_score(y, ypred)
#     print(f"\n--- Missingness-only control ---")
#     print(f"Balanced accuracy from coverage structure alone: {ba:.3f}  "
#           f"(chance = {1/9:.3f})")
#     print(f"  If this is above chance, neuron availability covaries with location.")
#     return ba


# -----------------------------------------------------------------------
# Within-train decoding control (A/B/C internal split)
# -----------------------------------------------------------------------
def run_within_train_decoding(retained_neurons, n_per_class):
    """
    Split train observations (A/B/C) into internal train / val halves.
    Decodes location within the train pool only, to check whether
    location signal exists at all vs whether it transfers to D.
    """
    offsets  = np.arange(-HALF_WIN, HALF_WIN)
    n_off    = len(offsets)
    bal_acc  = np.zeros((N_RESAMPLES, n_off))

    print(f"\n--- Within-train decoding (A/B/C internal split) | {N_RESAMPLES} resamples ---")
    for t_idx, offset in enumerate(offsets):
        long      = build_long_for_offset(offset)
        # build obs arrays from train-only: treat first half as "train", second as "test"
        tr_arr, _, neu_mean_full = build_obs_arrays(long, retained_neurons)
        n_neurons = len(retained_neurons)

        for r in range(N_RESAMPLES):
            rng_r = np.random.default_rng(RANDOM_SEED + t_idx * 1000 + r + 55555)

            # split each (loc, neuron) pool 50/50
            itr_arr  = [[np.array([], dtype=float)] * n_neurons for _ in ALL_LOCS]
            ival_arr = [[np.array([], dtype=float)] * n_neurons for _ in ALL_LOCS]
            for loc_idx in range(9):
                for n_idx in range(n_neurons):
                    obs = tr_arr[loc_idx][n_idx].copy()
                    if len(obs):
                        rng_r.shuffle(obs)
                        half = max(1, len(obs) // 2)
                        itr_arr[loc_idx][n_idx]  = obs[:half]
                        ival_arr[loc_idx][n_idx] = obs[half:] if len(obs) > 1 else obs

            # internal train mean from itr half
            neu_itr_mean = np.array([
                np.concatenate([itr_arr[l][n] for l in range(9)
                                if len(itr_arr[l][n])]).mean()
                if any(len(itr_arr[l][n]) for l in range(9)) else 0.0
                for n in range(n_neurons)
            ])

            Xtr, ytr, Xval, yval = build_pseudotrials(
                itr_arr, ival_arr, neu_itr_mean, n_per_class, rng_r)
            ba, _, _, _, _ = fit_decoder(Xtr, ytr, Xval, yval)
            bal_acc[r, t_idx] = ba

        if (t_idx + 1) % 5 == 0:
            print(f"  {t_idx+1}/{n_off} offsets done")

    return offsets, bal_acc


# -----------------------------------------------------------------------
# Confusion matrix at a specific offset
# -----------------------------------------------------------------------
def confusion_at_offset(offset, retained_neurons, n_per_class, label):
    long = build_long_for_offset(offset)
    tr_arr, te_arr, neu_mean = build_obs_arrays(long, retained_neurons)
    rng  = np.random.default_rng(RANDOM_SEED)
    Xtr, ytr, Xte, yte = build_pseudotrials(tr_arr, te_arr, neu_mean, n_per_class, rng)
    _, _, ypred, _, _ = fit_decoder(Xtr, ytr, Xte, yte)
    cm = confusion_matrix(yte, ypred, labels=ALL_LOCS)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(9)); ax.set_xticklabels(ALL_LOCS, fontsize=8)
    ax.set_yticks(range(9)); ax.set_yticklabels(ALL_LOCS, fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion matrix — {label} (offset={offset})')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f'decode_confusion_{label}.png')
    plt.savefig(out, dpi=150)
    #plt.close()
    print(f"Confusion matrix saved: {out}")
    return cm


# -----------------------------------------------------------------------
# Plot timecourse
# -----------------------------------------------------------------------
def plot_timecourse(offsets, bal_acc_transfer, bal_acc_within,
                    shuf_acc, filename='decode_curr_rew_loc.png'):
    tr_mean = bal_acc_transfer.mean(axis=0)
    tr_sem  = bal_acc_transfer.std(axis=0) / np.sqrt(N_RESAMPLES)
    wi_mean = bal_acc_within.mean(axis=0)
    wi_sem  = bal_acc_within.std(axis=0) / np.sqrt(N_RESAMPLES)
    sh_mean = shuf_acc.mean(axis=0)
    sh_sd   = shuf_acc.std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(offsets, sh_mean - sh_sd, sh_mean + sh_sd,
                    alpha=0.2, color='gray')
    ax.plot(offsets, sh_mean,  color='gray',   lw=1,   label='shuffle')
    ax.fill_between(offsets, wi_mean - wi_sem, wi_mean + wi_sem,
                    alpha=0.2, color='steelblue')
    ax.plot(offsets, wi_mean,  color='steelblue', lw=1.5, label='within-train A/B/C')
    ax.fill_between(offsets, tr_mean - tr_sem, tr_mean + tr_sem,
                    alpha=0.2, color='black')
    ax.plot(offsets, tr_mean,  'k-o', ms=3,   lw=1.5, label='transfer A/B/C→D')
    ax.axhline(1/9,  color='lightgray', ls=':', lw=1)
    ax.axvline(0,    color='blue',      ls=':', alpha=0.4, lw=1)
    ax.set_xlabel('Timebin offset from reward bin')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title(f'Pseudo-population decoder: current reward location\n'
                 f'({len(retained_neurons)} neurons, {n_per_class}×9 pseudo-trials, '
                 f'{N_RESAMPLES} resamples)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, filename)
    plt.savefig(out, dpi=150)
    #plt.close()
    print(f"Timecourse plot saved: {out}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
print(f"\nSettings: MIN_REPS={MIN_REPS}, MIN_LOCS={MIN_LOCS}, "
      f"N_RESAMPLES={N_RESAMPLES}, HALF_WIN={HALF_WIN}")

retained_neurons = select_retained_neurons(MIN_REPS, MIN_LOCS)
# report_missingness(retained_neurons)

# import pdb; pdb.set_trace() 

# -----------------------------------------------------------------------
# STEP 1: neuron selection and coverage diagnostics
#   Run this first. Saves retained_neurons.csv to RESULTS_DIR.
#   Fast — no decoding.
# -----------------------------------------------------------------------
n_per_class = compute_n_per_class(retained_neurons)
# run_missingness_control(retained_neurons, n_per_class)

pd.Series(retained_neurons, name='neuron_label').to_csv(
    os.path.join(RESULTS_DIR, 'decode_retained_neurons.csv'), index=False)
print(f"\nSTEP 1 done. Retained neuron list saved.")
print(f"n_per_class={n_per_class}  n_neurons={len(retained_neurons)}")

# -----------------------------------------------------------------------
# STEP 2: sanity-check one offset (offset=0)
#   Builds pseudo-trials for offset=0 once, fits decoder once, prints result.
#   Saves X_train shape and sample y values.
# -----------------------------------------------------------------------
print("\n--- STEP 2: single-offset sanity check (offset=0) ---")
long0 = build_long_for_offset(0)
tr_arr0, te_arr0, neu_mean0 = build_obs_arrays(long0, retained_neurons)
rng0 = np.random.default_rng(RANDOM_SEED)
Xtr0, ytr0, Xte0, yte0 = build_pseudotrials(tr_arr0, te_arr0, neu_mean0, n_per_class, rng0)
print(f"X_train shape: {Xtr0.shape}  X_test shape: {Xte0.shape}")
print(f"y_train unique: {np.unique(ytr0)}  y_test unique: {np.unique(yte0)}")
print(f"X_train NaN: {np.isnan(Xtr0).sum()}  X_test NaN: {np.isnan(Xte0).sum()}")
ba0, ra0, ypred0, _, _ = fit_decoder(Xtr0, ytr0, Xte0, yte0)
print(f"offset=0  balanced_acc={ba0:.3f}  raw_acc={ra0:.3f}  chance={1/9:.3f}")
print("Per-class predictions vs truth:")
for loc in ALL_LOCS:
    mask = yte0 == loc
    correct = (ypred0[mask] == loc).sum()
    print(f"  loc {loc}: {correct}/{mask.sum()} correct")
print("STEP 2 done.")


import pdb; pdb.set_trace()

# -----------------------------------------------------------------------
# STEP 3: transfer decoding (A/B/C -> D) across all offsets
#   Saves decode_transfer_acc.csv and a quick plot.
#   This is the main slow step.
# -----------------------------------------------------------------------
print("\n--- STEP 3: transfer decoding ---")
offsets, bal_acc_transfer, raw_acc_transfer = run_transfer_decoding(
    retained_neurons, n_per_class)

tr_df = pd.DataFrame(bal_acc_transfer,
                     columns=[f'off_{o}' for o in offsets])
tr_df.to_csv(os.path.join(RESULTS_DIR, 'decode_transfer_acc.csv'), index=False)

tr_mean = bal_acc_transfer.mean(axis=0)
best_offset = offsets[np.argmax(tr_mean)]
print(f"Transfer max balanced_acc: {tr_mean.max():.3f} at offset {best_offset}")
print(f"Transfer mean over offsets: {tr_mean.mean():.3f}")
print("STEP 3 done. Results saved to decode_transfer_acc.csv")

# quick interim plot
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(offsets, tr_mean, 'k-o', ms=3)
ax.fill_between(offsets,
                tr_mean - bal_acc_transfer.std(axis=0),
                tr_mean + bal_acc_transfer.std(axis=0),
                alpha=0.2, color='k')
ax.axhline(1/9, color='gray', ls=':')
ax.axvline(0,   color='blue', ls=':', alpha=0.5)
ax.set_xlabel('Offset'); ax.set_ylabel('Balanced acc'); ax.set_title('Transfer decoding')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'decode_transfer_interim.png'), dpi=120)
# plt.close()

# -----------------------------------------------------------------------
# STEP 4: within-train decoding control
#   Saves decode_within_acc.csv
# -----------------------------------------------------------------------
print("\n--- STEP 4: within-train decoding ---")
offsets, bal_acc_within = run_within_train_decoding(retained_neurons, n_per_class)

pd.DataFrame(bal_acc_within,
             columns=[f'off_{o}' for o in offsets]).to_csv(
    os.path.join(RESULTS_DIR, 'decode_within_acc.csv'), index=False)
wi_mean = bal_acc_within.mean(axis=0)
print(f"Within-train max balanced_acc: {wi_mean.max():.3f} at offset "
      f"{offsets[np.argmax(wi_mean)]}")
print("STEP 4 done. Results saved to decode_within_acc.csv")

# -----------------------------------------------------------------------
# STEP 5: shuffle control
#   Saves decode_shuffle_acc.csv
# -----------------------------------------------------------------------
print("\n--- STEP 5: shuffle control ---")
offsets, shuf_acc = run_shuffle_control(retained_neurons, n_per_class)

pd.DataFrame(shuf_acc,
             columns=[f'off_{o}' for o in offsets]).to_csv(
    os.path.join(RESULTS_DIR, 'decode_shuffle_acc.csv'), index=False)
sh_mean = shuf_acc.mean(axis=0)
print(f"Shuffle mean balanced_acc: {sh_mean.mean():.3f} (chance={1/9:.3f})")
print("STEP 5 done. Results saved to decode_shuffle_acc.csv")

# -----------------------------------------------------------------------
# STEP 6: confusion matrices and final plot
# -----------------------------------------------------------------------
print("\n--- STEP 6: confusion matrices and final plot ---")
early_offset = max(offsets[0], -10)
confusion_at_offset(early_offset, retained_neurons, n_per_class,
                    f'pre_reward_off{early_offset}')
confusion_at_offset(0,            retained_neurons, n_per_class, 'reward_bin_off0')
confusion_at_offset(best_offset,  retained_neurons, n_per_class,
                    f'best_off{best_offset}')

plot_timecourse(offsets, bal_acc_transfer, bal_acc_within, shuf_acc)

# combined results CSV
offsets_df = pd.DataFrame({
    'offset':                offsets,
    'transfer_bal_acc_mean': tr_mean,
    'transfer_bal_acc_sem':  bal_acc_transfer.std(axis=0) / np.sqrt(N_RESAMPLES),
    'within_bal_acc_mean':   wi_mean,
    'within_bal_acc_sem':    bal_acc_within.std(axis=0) / np.sqrt(N_RESAMPLES),
    'shuffle_mean':          sh_mean,
    'shuffle_sd':            shuf_acc.std(axis=0),
})
offsets_df.to_csv(os.path.join(RESULTS_DIR, 'decode_curr_rew_loc.csv'), index=False)
print("STEP 6 done. Final results saved to decode_curr_rew_loc.csv")
