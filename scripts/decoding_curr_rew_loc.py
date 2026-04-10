#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New attempt in only keeping what makes sense.

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

# Optimal split files (produced by find_best_train_test_split.py)
ASSIGN_CSV = os.path.join(RESULTS_DIR, "optimal_train_test_assignment.csv")
SPLIT_CSV  = os.path.join(RESULTS_DIR, "optimal_split_summary.csv")

# Thresholds derived from optimal split scan:
#   min_train=4 always (split guarantees it), min_test=3 retains 97.6% of neurons
MIN_REPS      = 4    # min train observations per location per neuron
MIN_REPS_TEST = 3    # min test observations per location per neuron
MIN_LOCS      = 9    # min locations a neuron must cover in train
N_BINS    = 360
REW_BINS  = {'A': 89, 'B': 179, 'C': 269, 'D': 359}
ALL_LOCS  = list(range(1, 10))
RANDOM_SEED = 42

# -----------------------------------------------------------------------
# TEST_MODE: fast settings for checking the pipeline end-to-end.
# Set TEST_MODE = False for the real analysis.
# -----------------------------------------------------------------------
TEST_MODE = False

if TEST_MODE:
    HALF_WIN            = 10    # only 10 offsets (-5..+4)
    N_RESAMPLES         = 3
    N_SHUFFLE_RESAMPLES = 2
    print("*** TEST MODE: reduced offsets and resamples ***")
else:
    HALF_WIN            = 45   # full 90 offsets 
    N_RESAMPLES         = 5 # i don't think this makes a lot of sense for me bc I don't have that much data.
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

# Load the slot-agnostic train/test assignment
assign_df = pd.read_csv(ASSIGN_CSV)
# Build a plain dict for fast lookup: (neuron_label, grid_no) -> set_optimal ('train'/'test')
# deduplicate first: each (neuron, grid_no) gets the same set regardless of slot
assign_dict = (assign_df.drop_duplicates(['neuron_label', 'grid_no'])
               .set_index(['neuron_label', 'grid_no'])['set_optimal']
               .to_dict())

split_df = pd.read_csv(SPLIT_CSV)
print(f"Loaded optimal split: {len(assign_df)} assignment rows, "
      f"{len(assign_dict)} unique (neuron, grid_no) entries")


# -----------------------------------------------------------------------
# Build long-format table for a given offset
# -----------------------------------------------------------------------
def build_long_for_offset(offset):
    """
    Returns long-format DataFrame with columns:
      session, neuron_label, grid_no, config, location_label, slot, set, activity

    'set' is determined by the slot-agnostic optimal split (not fixed A/B/C=train, D=test).
    activity = value at (reward_bin[slot] + offset) % N_BINS for each slot.
    """
    rows = []
    for slot, loc_col in [('A','loc_A'), ('B','loc_B'), ('C','loc_C'), ('D','loc_D')]:
        abs_bin = (REW_BINS[slot] + offset) % N_BINS
        bin_col = f'bin_{abs_bin:03d}'
        tmp = df[['session', 'neuron_label', 'grid_no', 'config', loc_col]].copy()
        tmp.columns = ['session', 'neuron_label', 'grid_no', 'config', 'location_label']
        tmp['slot']     = slot
        tmp['activity'] = df[bin_col].values
        # join optimal set assignment on (neuron_label, grid_no)
        idx_key = list(zip(tmp['neuron_label'], tmp['grid_no']))
        tmp['set'] = [assign_dict.get(k, 'train') for k in idx_key]
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


# -----------------------------------------------------------------------
# Select retained neurons (fixed across all offsets, based on train pool)
# -----------------------------------------------------------------------
def select_retained_neurons(min_reps=MIN_REPS, min_reps_test=MIN_REPS_TEST, min_locs=MIN_LOCS):
    """
    Uses the pre-computed optimal split summary to select neurons.
    A neuron is retained if for all 9 locations it has:
      - >= min_reps train observations
      - >= min_reps_test test observations
    (both must hold for ALL 9 locations, not just min_locs of them)

    Returns sorted list of retained neuron labels.
    """
    # pivot split_df to (neuron_label x location_label) for train and test counts
    train_rep = split_df.pivot(index='neuron_label', columns='location_label',
                               values='n_train').fillna(0)
    test_rep  = split_df.pivot(index='neuron_label', columns='location_label',
                               values='n_test').fillna(0)

    for loc in ALL_LOCS:
        if loc not in train_rep.columns: train_rep[loc] = 0
        if loc not in test_rep.columns:  test_rep[loc]  = 0
    train_rep = train_rep[ALL_LOCS]
    test_rep  = test_rep[ALL_LOCS]

    # neuron must meet thresholds across all 9 locations
    train_ok = (train_rep >= min_reps).all(axis=1)
    test_ok  = (test_rep  >= min_reps_test).all(axis=1)
    kept = sorted(train_rep.index[train_ok & test_ok])

    # population-level test coverage check
    te_covered = (test_rep.loc[kept] > 0).any(axis=0)
    missing_test = [l for l in ALL_LOCS if not te_covered.get(l, False)]

    print(f"\n--- Neuron inclusion (min_reps={min_reps}, min_reps_test={min_reps_test}, "
          f"min_locs={min_locs}) ---")
    print(f"Retained: {len(kept)} / {df['neuron_label'].nunique()} neurons")
    if missing_test:
        print(f"WARNING: test locations not covered at population level: {missing_test}")
        print("9-way decoding not feasible.")
        raise RuntimeError("Test coverage insufficient.")
    else:
        print("9-way pooled coverage: OK in both train and test.")

    te = test_rep.loc[kept]
    tr = train_rep.loc[kept]

    print("\nPooled train obs per location:")
    print(tr.sum(axis=0).to_string())
    print("\nPooled test obs per location:")
    print(te.sum(axis=0).to_string())
    print("\nMin train obs per location (worst neuron):")
    print(tr.min(axis=0).to_string())
    print("\nMin test obs per location (worst neuron):")
    print(te.min(axis=0).to_string())

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
    # DONT USE THIS BECUASE ITS SUPER IMPRECISE!
    neu_train_mean = np.zeros(n_neurons)
    for n_idx in range(n_neurons):
        all_vals = np.concatenate([train_arr[l][n_idx] for l in range(9)
                                   if len(train_arr[l][n_idx])])
        if len(all_vals):
            neu_train_mean[n_idx] = all_vals.mean()

    return train_arr, test_arr, neu_train_mean




def build_pseudotrials(train_arr, test_arr, neu_train_mean, rng):
    """
    Fast pseudo-trial construction using pre-built observation arrays.

    Draws exactly MIN_REPS train samples and MIN_REPS_TEST test samples
    per (location, neuron). Missing test entries imputed with per-neuron train mean.

    Returns: X_train (9*MIN_REPS, n_neurons), y_train,
             X_test  (9*MIN_REPS_TEST, n_neurons), y_test
    """
    n_neurons = len(neu_train_mean)
    n_locs    = len(ALL_LOCS)

    # because the train neurons have been subselected based on MIN_LOCS and MIN_REPS
    # I should be able to randomly draw the amount of samples I need.
    # in the end, X_train should be 
    X_train = np.empty((MIN_LOCS*MIN_REPS, n_neurons))
    y_train = np.repeat(ALL_LOCS, MIN_REPS)
    
    X_test  = np.empty((MIN_LOCS*MIN_REPS_TEST, n_neurons))
    y_test = np.repeat(ALL_LOCS, MIN_REPS_TEST)
    
    for loc_idx in range(n_locs):
        for n_idx in range(n_neurons):
            # --- train ---
            obs = train_arr[loc_idx][n_idx]
            if len(obs):
                X_train[loc_idx*MIN_REPS:loc_idx*MIN_REPS+MIN_REPS, n_idx] = rng.choice(
                    obs, size=MIN_REPS, replace=len(obs) < MIN_REPS)
            else:
                # fallback: impute with per-neuron train mean (rare, only if all values NaN)
                X_train[loc_idx*MIN_REPS:loc_idx*MIN_REPS+MIN_REPS, n_idx] = neu_train_mean[n_idx]

            # --- test ---
            obs = test_arr[loc_idx][n_idx]
            if len(obs):
                X_test[loc_idx*MIN_REPS_TEST:loc_idx*MIN_REPS_TEST+MIN_REPS_TEST, n_idx] = rng.choice(
                    obs, size=MIN_REPS_TEST, replace=len(obs) < MIN_REPS_TEST)
            else:
                X_test[loc_idx*MIN_REPS_TEST:loc_idx*MIN_REPS_TEST+MIN_REPS_TEST, n_idx] = neu_train_mean[n_idx]
            
    return X_train, y_train, X_test, y_test



# -----------------------------------------------------------------------
# Fit and evaluate one decoder
# -----------------------------------------------------------------------
def fit_decoder(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    Xtr_z  = scaler.fit_transform(X_train)
    Xte_z  = scaler.transform(X_test)
    #  import pdb; pdb.set_trace()
    
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
def run_transfer_decoding(retained_neurons):
    offsets = np.arange(-HALF_WIN, HALF_WIN)
    n_off   = len(offsets)
    bal_acc = np.zeros((N_RESAMPLES, n_off))
    raw_acc = np.zeros((N_RESAMPLES, n_off))

    print(f"\n--- Transfer decoding (A/B/C -> D) | {N_RESAMPLES} resamples ---")
    
    # import pdb; pdb.set_trace()
    
    for t_idx, offset in enumerate(offsets):
        long = build_long_for_offset(offset)
        tr_arr, te_arr, neu_mean = build_obs_arrays(long, retained_neurons)
        for r in range(N_RESAMPLES):
            rng_r = np.random.default_rng(RANDOM_SEED + t_idx * 1000 + r)
            Xtr, ytr, Xte, yte = build_pseudotrials(tr_arr, te_arr, neu_mean, rng_r)
            ba, ra, _, _, _ = fit_decoder(Xtr, ytr, Xte, yte)
            bal_acc[r, t_idx] = ba
            raw_acc[r, t_idx] = ra

        if (t_idx + 1) % 5 == 0:
            print(f"  {t_idx+1}/{n_off} offsets done")
    
    return offsets, bal_acc, raw_acc


# -----------------------------------------------------------------------
# Run shuffle control
# -----------------------------------------------------------------------
def run_shuffle_control(retained_neurons):
    offsets  = np.arange(-HALF_WIN, HALF_WIN)
    n_off    = len(offsets)
    shuf_acc = np.zeros((N_SHUFFLE_RESAMPLES, n_off))

    print(f"\n--- Shuffle control | {N_SHUFFLE_RESAMPLES} resamples ---")
    for t_idx, offset in enumerate(offsets):
        long = build_long_for_offset(offset)
        tr_arr, te_arr, neu_mean = build_obs_arrays(long, retained_neurons)
        for r in range(N_SHUFFLE_RESAMPLES):
            rng_r = np.random.default_rng(RANDOM_SEED + t_idx * 1000 + r + 99999)
            Xtr, ytr, Xte, yte = build_pseudotrials(tr_arr, te_arr, neu_mean, rng_r)
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
# Within-train decoding control (A/B/C internal split)
# -----------------------------------------------------------------------
def run_within_train_decoding(retained_neurons):
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
                itr_arr, ival_arr, neu_itr_mean, rng_r)
            ba, _, _, _, _ = fit_decoder(Xtr, ytr, Xval, yval)
            bal_acc[r, t_idx] = ba

        if (t_idx + 1) % 5 == 0:
            print(f"  {t_idx+1}/{n_off} offsets done")

    return offsets, bal_acc


# -----------------------------------------------------------------------
# Confusion heatmap: best vs worst transfer decoding offset
# -----------------------------------------------------------------------
def plot_confusion_best_worst(offsets, bal_acc_transfer, retained_neurons):
    """
    Finds the best and worst offset from the transfer decoding timecourse,
    fits one decoder at each, and plots row-normalised confusion heatmaps
    side by side (so each cell shows P(predicted | true)).
    """
    tr_mean     = bal_acc_transfer.mean(axis=0)
    best_idx    = np.argmax(tr_mean)
    worst_idx   = np.argmin(tr_mean)
    best_offset = offsets[best_idx]
    worst_offset = offsets[worst_idx]

    def get_confusion(offset):
        long = build_long_for_offset(offset)
        tr_arr, te_arr, neu_mean = build_obs_arrays(long, retained_neurons)
        rng = np.random.default_rng(RANDOM_SEED)
        Xtr, ytr, Xte, yte = build_pseudotrials(tr_arr, te_arr, neu_mean, rng)
        _, _, ypred, _, _ = fit_decoder(Xtr, ytr, Xte, yte)
        cm = confusion_matrix(yte, ypred, labels=ALL_LOCS).astype(float)
        # row-normalise: each row sums to 1 → P(predicted | true location)
        cm = cm / cm.sum(axis=1, keepdims=True)
        return cm

    cm_best  = get_confusion(best_offset)
    cm_worst = get_confusion(worst_offset)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    locs = [str(l) for l in ALL_LOCS]

    for ax, cm, title, acc in [
        (axes[0], cm_best,  f'Best offset={best_offset}',  tr_mean[best_idx]),
        (axes[1], cm_worst, f'Worst offset={worst_offset}', tr_mean[worst_idx]),
    ]:
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(9)); ax.set_xticklabels(locs, fontsize=8)
        ax.set_yticks(range(9)); ax.set_yticklabels(locs, fontsize=8)
        ax.set_xlabel('Predicted location')
        ax.set_ylabel('True location')
        ax.set_title(f'{title}\nbal. acc = {acc:.3f}', fontsize=9)
        # annotate cells with value
        for i in range(9):
            for j in range(9):
                ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center',
                        fontsize=5, color='white' if cm[i,j] > 0.5 else 'black')
        plt.colorbar(im, ax=ax, label='P(pred | true)')

    plt.suptitle('Confusion: which locations are confused?\n'
                 '(row-normalised, train A+B+C → test D)', fontsize=10)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'decode_confusion_best_worst.png')
    plt.savefig(out, dpi=150)
    print(f"Confusion heatmap saved: {out}")
    print(f"  Best offset: {best_offset}  (bal acc {tr_mean[best_idx]:.3f})")
    print(f"  Worst offset: {worst_offset}  (bal acc {tr_mean[worst_idx]:.3f})")


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
                 f'({len(retained_neurons)} neurons, train={MIN_REPS}×9 / test={MIN_REPS_TEST}×9 pseudo-trials, '
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
print(f"\nSettings: MIN_REPS={MIN_REPS}, MIN_REPS_TEST={MIN_REPS_TEST}, MIN_LOCS={MIN_LOCS}, "
      f"N_RESAMPLES={N_RESAMPLES}, HALF_WIN={HALF_WIN}")

retained_neurons = select_retained_neurons(MIN_REPS, MIN_REPS_TEST, MIN_LOCS)

pd.Series(retained_neurons, name='neuron_label').to_csv(
    os.path.join(RESULTS_DIR, 'decode_retained_neurons.csv'), index=False)
print(f"\nSTEP 1 done. Retained neuron list saved.")
print(f"n_neurons={len(retained_neurons)}")

# -----------------------------------------------------------------------
# STEP 2: sanity-check one offset (offset=0)
#   Builds pseudo-trials for offset=0 once, fits decoder once, prints result.
#   Saves X_train shape and sample y values.
# -----------------------------------------------------------------------
print("\n--- STEP 2: single-offset sanity check (offset=0) ---")
long0 = build_long_for_offset(0)

# first, vectorise everything.
tr_arr0, te_arr0, neu_mean0 = build_obs_arrays(long0, retained_neurons)
rng0 = np.random.default_rng(RANDOM_SEED)

Xtr0, ytr0, Xte0, yte0 = build_pseudotrials(tr_arr0, te_arr0, neu_mean0, rng0)


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

# import pdb; pdb.set_trace()


# -----------------------------------------------------------------------
# STEP 3: transfer decoding (A/B/C -> D) across all offsets
#   Saves decode_transfer_acc.csv and a quick plot.
#   This is the main slow step.
# -----------------------------------------------------------------------
print("\n--- STEP 3: transfer decoding ---")
offsets, bal_acc_transfer, raw_acc_transfer = run_transfer_decoding(retained_neurons)

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
offsets, bal_acc_within = run_within_train_decoding(retained_neurons)

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
offsets, shuf_acc = run_shuffle_control(retained_neurons)

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
plot_confusion_best_worst(offsets, bal_acc_transfer, retained_neurons)
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
