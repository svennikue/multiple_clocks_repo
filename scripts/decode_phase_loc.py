#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decode_phase_loc.py

Pseudo-population decoder for current location, run separately per phase
(reward / early / late).

Data source: all_location_snippets.csv  (produced by create_location_snippets.py)
  Each row = one (neuron × grid_no × location visit), averaged across ~10 correct
  repeats of that grid_no block.
  Snippet bins: bin_00..bin_29, onset of current location at bin_10 (index 10).

Train/test split: phase_snippets_{phase}_assignment.csv
  (produced by find_phase_train_test_split.py)

Decoder:
  - Pseudo-population: one feature per retained neuron, pooled across sessions.
  - Time-resolved: slide a single-bin window across the 30-bin snippet.
    Offsets are relative to location onset (bin_10 = offset 0):
      offset -10 → bin_00  (10 bins before onset)
      offset   0 → bin_10  (onset)
      offset +19 → bin_29  (last bin)
  - Per offset: sample MIN_REPS_TRAIN / MIN_REPS_TEST obs per (neuron, location),
    fit LogisticRegression, score balanced_accuracy.
  - N_RESAMPLES repetitions per offset; offsets parallelised with joblib.
  - Shuffle control: same pipeline, permuted train labels.

Future extension (labels stored in snippets, not yet decoded):
  - next+1_loc ... next+10_loc: future location labels
  - next+k_phase: phase of each future location
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
DATA_DIR     = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
RESULTS_DIR  = "/Users/xpsy1114/Documents/projects/multiple_clocks/results"
SNIPPETS_CSV = os.path.join(DATA_DIR, "all_location_snippets.csv")

SNIPPET_LEN = 30
ONSET_IDX   = 10   # bin_10 = location onset
OFFSETS     = np.arange(-ONSET_IDX, SNIPPET_LEN - ONSET_IDX)   # -10 .. +19, 30 total

ALL_LOCS    = list(range(1, 10))
RANDOM_SEED = 42

# Per-phase min observations to draw per (neuron, location) for train and test.
# Based on find_phase_train_test_split.py output:
#   late:   819 neurons retained at min_obs=2 → min_train=1, min_test=1
#   early:  507 neurons retained at min_obs=1 → min_train=1, min_test=1 (with replacement)
#   reward: 113 neurons at min_obs=1 → very sparse, exploratory
PHASE_CONFIG = {
    'late':   {'min_train': 1, 'min_test': 1},
    'early':  {'min_train': 1, 'min_test': 1},
    'reward': {'min_train': 1, 'min_test': 1},
}

TEST_MODE = True

if TEST_MODE:
    N_RESAMPLES         = 3
    N_SHUFFLE_RESAMPLES = 5
    N_JOBS              = 2
    print("*** TEST MODE ***")
else:
    N_RESAMPLES         = 20
    N_SHUFFLE_RESAMPLES = 50
    N_JOBS              = -1   # use all cores


# -----------------------------------------------------------------------
# Load snippets once
# -----------------------------------------------------------------------
print(f"Loading {SNIPPETS_CSV} ...")
snip_all = pd.read_csv(SNIPPETS_CSV)
snip_all = snip_all.reset_index(drop=True)
snip_all.index.name = 'row_id'
snip_all = snip_all.reset_index()   # row_id is now a column

SNIPPET_COLS = [f'bin_{b:02d}' for b in range(SNIPPET_LEN)]
print(f"  {len(snip_all)} rows | {snip_all['neuron_label'].nunique()} neurons | "
      f"phases: {sorted(snip_all['phase'].unique())}")


# -----------------------------------------------------------------------
# Helper: select retained neurons and attach set labels to phase_df
# Returns (retained_neurons list, phase_df_with_set column)
# -----------------------------------------------------------------------
def select_retained_neurons(phase_df, assign_csv, min_train, min_test):
    assign_df   = pd.read_csv(assign_csv)
    assign_dict = assign_df.set_index('row_id')['set'].to_dict()

    phase_df = phase_df.copy()
    phase_df['set'] = phase_df['row_id'].map(assign_dict).fillna('train')

    tr_counts = (phase_df[phase_df['set'] == 'train']
                 .groupby(['neuron_label', 'current_location']).size()
                 .unstack(fill_value=0))
    te_counts = (phase_df[phase_df['set'] == 'test']
                 .groupby(['neuron_label', 'current_location']).size()
                 .unstack(fill_value=0))

    for loc in ALL_LOCS:
        if loc not in tr_counts.columns: tr_counts[loc] = 0
        if loc not in te_counts.columns: te_counts[loc] = 0
    tr_counts = tr_counts[ALL_LOCS]
    te_counts = te_counts[ALL_LOCS]

    all_neurons = tr_counts.index.union(te_counts.index)
    tr_counts   = tr_counts.reindex(all_neurons, fill_value=0)
    te_counts   = te_counts.reindex(all_neurons, fill_value=0)

    train_ok = (tr_counts >= min_train).all(axis=1)
    test_ok  = (te_counts >= min_test).all(axis=1)
    kept     = sorted(all_neurons[train_ok & test_ok])

    if not kept:
        raise RuntimeError(f"No neurons retained with min_train={min_train}, min_test={min_test}")

    missing_test = [l for l in ALL_LOCS if (te_counts.loc[kept, l] == 0).all()]
    if missing_test:
        raise RuntimeError(f"Test locations not covered at population level: {missing_test}")

    total = len(all_neurons)
    print(f"  Retained: {len(kept)} / {total} neurons")
    print(f"  Min train per location (worst neuron): "
          f"{tr_counts.loc[kept].min(axis=0).values}")
    print(f"  Min test  per location (worst neuron): "
          f"{te_counts.loc[kept].min(axis=0).values}")

    # restrict to retained neurons; set column already attached
    phase_df = phase_df[phase_df['neuron_label'].isin(kept)].copy()
    return kept, phase_df


# -----------------------------------------------------------------------
# Helper: build observation arrays for one offset
#   Input: phase_df already has 'set' column and is restricted to retained neurons
#   Returns: train_arr[loc_idx][neu_idx] -> 1-D float array
#            test_arr [loc_idx][neu_idx] -> 1-D float array
#            neu_train_mean              -> (n_neurons,) float array
# -----------------------------------------------------------------------
def build_obs_arrays(phase_df, neu_to_idx, n_neurons, bin_col):
    train_arr = [[np.array([], dtype=float)] * n_neurons for _ in ALL_LOCS]
    test_arr  = [[np.array([], dtype=float)] * n_neurons for _ in ALL_LOCS]

    tr = phase_df[phase_df['set'] == 'train']
    te = phase_df[phase_df['set'] == 'test']

    for (neu, loc), grp in tr.groupby(['neuron_label', 'current_location']):
        if neu not in neu_to_idx: continue
        vals = grp[bin_col].dropna().values
        if len(vals):
            train_arr[loc - 1][neu_to_idx[neu]] = vals

    for (neu, loc), grp in te.groupby(['neuron_label', 'current_location']):
        if neu not in neu_to_idx: continue
        vals = grp[bin_col].dropna().values
        if len(vals):
            test_arr[loc - 1][neu_to_idx[neu]] = vals

    neu_train_mean = np.zeros(n_neurons)
    for n_idx in range(n_neurons):
        all_vals = np.concatenate([train_arr[l][n_idx] for l in range(9)
                                   if len(train_arr[l][n_idx])])
        if len(all_vals):
            neu_train_mean[n_idx] = all_vals.mean()

    return train_arr, test_arr, neu_train_mean


# -----------------------------------------------------------------------
# Helper: build one set of pseudo-trials from pre-built obs arrays
# -----------------------------------------------------------------------
def build_pseudotrials(train_arr, test_arr, neu_train_mean, min_train, min_test, rng):
    n_neurons = len(neu_train_mean)
    n_locs    = len(ALL_LOCS)

    X_train = np.empty((n_locs * min_train, n_neurons))
    y_train = np.repeat(ALL_LOCS, min_train)
    X_test  = np.empty((n_locs * min_test,  n_neurons))
    y_test  = np.repeat(ALL_LOCS, min_test)

    for loc_idx in range(n_locs):
        r_tr0, r_tr1 = loc_idx * min_train, (loc_idx + 1) * min_train
        r_te0, r_te1 = loc_idx * min_test,  (loc_idx + 1) * min_test
        for n_idx in range(n_neurons):
            obs = train_arr[loc_idx][n_idx]
            X_train[r_tr0:r_tr1, n_idx] = (
                rng.choice(obs, size=min_train, replace=len(obs) < min_train)
                if len(obs) else neu_train_mean[n_idx])
            obs = test_arr[loc_idx][n_idx]
            X_test[r_te0:r_te1, n_idx] = (
                rng.choice(obs, size=min_test, replace=len(obs) < min_test)
                if len(obs) else neu_train_mean[n_idx])

    return X_train, y_train, X_test, y_test


# -----------------------------------------------------------------------
# Helper: fit one decoder
# -----------------------------------------------------------------------
def fit_decoder(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    Xtr_z  = scaler.fit_transform(X_train)
    Xte_z  = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                             multi_class='multinomial', random_state=RANDOM_SEED)
    clf.fit(Xtr_z, y_train)
    y_pred = clf.predict(Xte_z)
    return (balanced_accuracy_score(y_test, y_pred),
            (y_pred == y_test).mean(),
            y_pred)


# -----------------------------------------------------------------------
# Worker: decode one offset — called in parallel across offsets
# phase_df already has 'set' column, restricted to retained neurons
# -----------------------------------------------------------------------
def decode_one_offset(offset, phase_df, neu_to_idx, n_neurons,
                      min_train, min_test, n_resamples, seed_base, shuffle):
    bin_col = f'bin_{(ONSET_IDX + offset):02d}'
    tr_arr, te_arr, neu_mean = build_obs_arrays(
        phase_df, neu_to_idx, n_neurons, bin_col)

    ba_out = np.zeros(n_resamples)
    for r in range(n_resamples):
        rng = np.random.default_rng(seed_base + abs(offset) * 1000 + r)
        Xtr, ytr, Xte, yte = build_pseudotrials(
            tr_arr, te_arr, neu_mean, min_train, min_test, rng)
        if shuffle:
            ytr = rng.permutation(ytr)
        ba, _, _ = fit_decoder(Xtr, ytr, Xte, yte)
        ba_out[r] = ba

    return ba_out   # shape (n_resamples,)


# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------
def plot_timecourse(offsets, ba_transfer, ba_shuffle, phase, n_neurons,
                    min_train, min_test):
    tr_mean = ba_transfer.mean(axis=0)
    tr_sem  = ba_transfer.std(axis=0) / np.sqrt(ba_transfer.shape[0])
    sh_mean = ba_shuffle.mean(axis=0)
    sh_sd   = ba_shuffle.std(axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(offsets, sh_mean - sh_sd, sh_mean + sh_sd,
                    alpha=0.2, color='gray')
    ax.plot(offsets, sh_mean, color='gray', lw=1, label='shuffle')
    ax.fill_between(offsets, tr_mean - tr_sem, tr_mean + tr_sem,
                    alpha=0.2, color='steelblue')
    ax.plot(offsets, tr_mean, 'o-', color='steelblue', ms=3, lw=1.5,
            label='decoder (train→test)')
    ax.axhline(1/9, color='lightgray', ls=':', lw=1)
    ax.axvline(0, color='blue', ls=':', alpha=0.4, lw=1, label='location onset')
    ax.set_xlabel('Timebin offset from location onset')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title(f'Location decoder — {phase} phase\n'
                 f'({n_neurons} neurons, '
                 f'train={min_train}×9 / test={min_test}×9 pseudo-trials, '
                 f'{ba_transfer.shape[0]} resamples)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f'decode_phase_loc_{phase}.png')
    plt.savefig(out, dpi=150)
    # plt.close()
    print(f"  Timecourse plot: {out}")


def plot_confusion(offsets, ba_transfer, phase, phase_df, neu_to_idx, n_neurons,
                   min_train, min_test):
    tr_mean   = ba_transfer.mean(axis=0)
    best_off  = offsets[np.argmax(tr_mean)]
    worst_off = offsets[np.argmin(tr_mean)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, off, tag in [(axes[0], best_off, 'Best'), (axes[1], worst_off, 'Worst')]:
        bin_col = f'bin_{(ONSET_IDX + off):02d}'
        tr_arr, te_arr, neu_mean = build_obs_arrays(
            phase_df, neu_to_idx, n_neurons, bin_col)
        rng = np.random.default_rng(RANDOM_SEED)
        Xtr, ytr, Xte, yte = build_pseudotrials(
            tr_arr, te_arr, neu_mean, min_train, min_test, rng)
        _, _, ypred = fit_decoder(Xtr, ytr, Xte, yte)
        cm = confusion_matrix(yte, ypred, labels=ALL_LOCS).astype(float)
        cm /= cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        locs_s = [str(l) for l in ALL_LOCS]
        ax.set_xticks(range(9)); ax.set_xticklabels(locs_s, fontsize=8)
        ax.set_yticks(range(9)); ax.set_yticklabels(locs_s, fontsize=8)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        off_idx = np.where(offsets == off)[0][0]
        ax.set_title(f'{tag} offset={off:+d}\nbal.acc={tr_mean[off_idx]:.3f}', fontsize=9)
        for i in range(9):
            for j in range(9):
                ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center',
                        fontsize=5, color='white' if cm[i, j] > 0.5 else 'black')
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'Confusion — {phase} phase (row-normalised)', fontsize=10)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f'decode_phase_loc_{phase}_confusion.png')
    plt.savefig(out, dpi=150)
    # plt.close()
    print(f"  Confusion plot:  {out}")


# -----------------------------------------------------------------------
# Main loop — one phase at a time
# -----------------------------------------------------------------------
all_results = {}

for phase in ['late', 'early', 'reward']:
    print(f"\n{'='*60}")
    print(f"Phase: {phase.upper()}")
    print(f"{'='*60}")

    cfg        = PHASE_CONFIG[phase]
    min_train  = cfg['min_train']
    min_test   = cfg['min_test']
    assign_csv = os.path.join(RESULTS_DIR, f"phase_snippets_{phase}_assignment.csv")

    if not os.path.exists(assign_csv):
        print(f"  Assignment file missing: {assign_csv} — skipping.")
        continue

    phase_df = snip_all[snip_all['phase'] == phase].copy()
    print(f"  {len(phase_df)} rows in phase")

    # Select retained neurons; phase_df gets 'set' column, restricted to retained
    try:
        retained_neurons, phase_df = select_retained_neurons(
            phase_df, assign_csv, min_train, min_test)
    except RuntimeError as e:
        print(f"  Skipping: {e}")
        continue

    n_neurons  = len(retained_neurons)
    neu_to_idx = {n: i for i, n in enumerate(retained_neurons)}

    # Save retained neuron list
    pd.Series(retained_neurons, name='neuron_label').to_csv(
        os.path.join(RESULTS_DIR, f'decode_phase_{phase}_neurons.csv'), index=False)

    # --- parallel transfer decoding ---
    print(f"\n  Transfer decoding | {N_RESAMPLES} resamples | {len(OFFSETS)} offsets ...")
    ba_list = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(decode_one_offset)(
            off, phase_df, neu_to_idx, n_neurons,
            min_train, min_test, N_RESAMPLES, RANDOM_SEED, shuffle=False)
        for off in OFFSETS
    )
    # ba_list[i] = array(N_RESAMPLES,) for offset i
    ba_transfer = np.stack(ba_list, axis=1)   # (N_RESAMPLES, n_offsets)

    tr_mean  = ba_transfer.mean(axis=0)
    best_off = OFFSETS[np.argmax(tr_mean)]
    print(f"  Peak balanced_acc: {tr_mean.max():.3f} at offset {best_off:+d}")
    print(f"  Mean over offsets: {tr_mean.mean():.3f}  (chance={1/9:.3f})")

    # --- parallel shuffle control ---
    print(f"\n  Shuffle control | {N_SHUFFLE_RESAMPLES} resamples ...")
    sh_list = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(decode_one_offset)(
            off, phase_df, neu_to_idx, n_neurons,
            min_train, min_test, N_SHUFFLE_RESAMPLES, RANDOM_SEED + 99999, shuffle=True)
        for off in OFFSETS
    )
    ba_shuffle = np.stack(sh_list, axis=1)   # (N_SHUFFLE_RESAMPLES, n_offsets)
    sh_mean = ba_shuffle.mean(axis=0)
    print(f"  Shuffle mean: {sh_mean.mean():.3f} ± {ba_shuffle.std(axis=0).mean():.3f}")

    # --- save CSVs ---
    pd.DataFrame(ba_transfer,
                 columns=[f'off_{o:+d}' for o in OFFSETS]).to_csv(
        os.path.join(RESULTS_DIR, f'decode_phase_{phase}_transfer.csv'), index=False)

    pd.DataFrame(ba_shuffle,
                 columns=[f'off_{o:+d}' for o in OFFSETS]).to_csv(
        os.path.join(RESULTS_DIR, f'decode_phase_{phase}_shuffle.csv'), index=False)

    pd.DataFrame({
        'offset':        OFFSETS,
        'transfer_mean': tr_mean,
        'transfer_sem':  ba_transfer.std(axis=0) / np.sqrt(N_RESAMPLES),
        'shuffle_mean':  sh_mean,
        'shuffle_sd':    ba_shuffle.std(axis=0),
        'n_neurons':     n_neurons,
        'phase':         phase,
    }).to_csv(os.path.join(RESULTS_DIR, f'decode_phase_{phase}_summary.csv'), index=False)

    # --- plots ---
    plot_timecourse(OFFSETS, ba_transfer, ba_shuffle, phase,
                    n_neurons, min_train, min_test)
    plot_confusion(OFFSETS, ba_transfer, phase, phase_df, neu_to_idx, n_neurons,
                   min_train, min_test)

    all_results[phase] = {
        'ba_transfer': ba_transfer,
        'ba_shuffle':  ba_shuffle,
        'n_neurons':   n_neurons,
    }
    print(f"\n  Phase {phase} done.")


# -----------------------------------------------------------------------
# Combined timecourse: all phases on one axis
# -----------------------------------------------------------------------
if len(all_results) > 1:
    colors = {'late': 'steelblue', 'early': 'darkorange', 'reward': 'crimson'}
    fig, ax = plt.subplots(figsize=(10, 4))
    for phase, res in all_results.items():
        tr_mean = res['ba_transfer'].mean(axis=0)
        tr_sem  = res['ba_transfer'].std(axis=0) / np.sqrt(res['ba_transfer'].shape[0])
        c = colors.get(phase, 'black')
        ax.fill_between(OFFSETS, tr_mean - tr_sem, tr_mean + tr_sem,
                        alpha=0.15, color=c)
        ax.plot(OFFSETS, tr_mean, '-o', color=c, ms=2, lw=1.5,
                label=f'{phase} (n={res["n_neurons"]})')
    ax.axhline(1/9, color='lightgray', ls=':', lw=1)
    ax.axvline(0,   color='blue',      ls=':', alpha=0.4, lw=1)
    ax.set_xlabel('Timebin offset from location onset')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title('Location decoder — all phases')
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'decode_phase_loc_combined.png')
    plt.savefig(out, dpi=150)
    # plt.close()
    print(f"\nCombined plot: {out}")

print("\nAll phases done.")
