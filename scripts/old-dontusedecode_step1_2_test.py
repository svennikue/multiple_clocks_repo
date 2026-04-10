"""
Quick test of steps 1 and 2 only (no decoding loop).
Run this first to verify the pipeline before the full run.
"""
import sys
sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

DATA_DIR    = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
RESULTS_DIR = "/Users/xpsy1114/Documents/projects/multiple_clocks/results"
NEURON_CSV  = os.path.join(DATA_DIR, "all_neurons_avg_per_gridrun.csv")

MIN_REPS = 4
MIN_LOCS = 9
N_BINS   = 360
REW_BINS = {'A': 89, 'B': 179, 'C': 269, 'D': 359}
ALL_LOCS = list(range(1, 10))
RANDOM_SEED = 42

# load
df = pd.read_csv(NEURON_CSV)
df[['loc_A','loc_B','loc_C','loc_D']] = df['config'].str.split('-',expand=True).astype(int)
print(f"Loaded {len(df)} rows | {df['neuron_label'].nunique()} neurons")

# build long for offset 0
def build_long_for_offset(offset):
    rows = []
    for slot, loc_col in [('A','loc_A'),('B','loc_B'),('C','loc_C'),('D','loc_D')]:
        abs_bin = (REW_BINS[slot] + offset) % N_BINS
        bin_col = f'bin_{abs_bin:03d}'
        tmp = df[['session','neuron_label','grid_no','config',loc_col]].copy()
        tmp.columns = ['session','neuron_label','grid_no','config','location_label']
        tmp['slot'] = slot
        tmp['set']  = 'test' if slot == 'D' else 'train'
        tmp['activity'] = df[bin_col].values
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)

# select retained neurons
long0   = build_long_for_offset(0)
train0  = long0[long0['set']=='train']
test0   = long0[long0['set']=='test']
tr_rep  = train0.groupby(['neuron_label','location_label']).size().unstack(fill_value=0)
te_rep  = test0.groupby(['neuron_label','location_label']).size().unstack(fill_value=0)
for loc in ALL_LOCS:
    if loc not in tr_rep.columns: tr_rep[loc]=0
    if loc not in te_rep.columns: te_rep[loc]=0
tr_rep = tr_rep[ALL_LOCS]; te_rep = te_rep[ALL_LOCS]
kept = sorted(tr_rep.index[(tr_rep>=MIN_REPS).sum(axis=1)>=MIN_LOCS])
print(f"\nRetained neurons: {len(kept)}")

# build obs arrays
def build_obs_arrays(long_df, retained_neurons):
    n_neurons  = len(retained_neurons)
    neu_to_idx = {n: i for i, n in enumerate(retained_neurons)}
    tr = long_df[long_df['set']=='train']
    te = long_df[long_df['set']=='test']
    tr = tr[tr['neuron_label'].isin(retained_neurons)]
    te = te[te['neuron_label'].isin(retained_neurons)]
    train_arr = [[np.array([],dtype=float)]*n_neurons for _ in ALL_LOCS]
    test_arr  = [[np.array([],dtype=float)]*n_neurons for _ in ALL_LOCS]
    for (neu,loc),grp in tr.groupby(['neuron_label','location_label']):
        if neu in neu_to_idx:
            v = grp['activity'].dropna().values
            if len(v): train_arr[loc-1][neu_to_idx[neu]] = v
    for (neu,loc),grp in te.groupby(['neuron_label','location_label']):
        if neu in neu_to_idx:
            v = grp['activity'].dropna().values
            if len(v): test_arr[loc-1][neu_to_idx[neu]] = v
    neu_train_mean = np.array([
        np.concatenate([train_arr[l][n] for l in range(9) if len(train_arr[l][n])]).mean()
        if any(len(train_arr[l][n]) for l in range(9)) else 0.0
        for n in range(n_neurons)
    ])
    return train_arr, test_arr, neu_train_mean

print("\nBuilding obs arrays...")
tr_arr, te_arr, neu_mean = build_obs_arrays(long0, kept)
print("Done.")

# check pool sizes for a few neurons
n_neurons = len(kept)
print(f"\nSample pool sizes (neuron 0, all locations):")
for loc_idx in range(9):
    print(f"  loc {loc_idx+1}: train={len(tr_arr[loc_idx][0])}, test={len(te_arr[loc_idx][0])}")

print(f"\nneu_train_mean range: {neu_mean.min():.3f} .. {neu_mean.max():.3f}")
print(f"Any NaN in neu_mean: {np.isnan(neu_mean).any()}")

# compute n_per_class
tr_pool = train0[train0['neuron_label'].isin(kept)].groupby('location_label').size()
te_pool = test0[test0['neuron_label'].isin(kept)].groupby('location_label').size()
n_per_class = int(te_pool.min())
print(f"\nPooled train obs min: {tr_pool.min()} (loc {tr_pool.idxmin()})")
print(f"Pooled test  obs min: {te_pool.min()} (loc {te_pool.idxmin()})")
print(f"n_per_class: {n_per_class}")

# build one set of pseudo-trials
def build_pseudotrials(train_arr, test_arr, neu_train_mean, n_per_class, rng):
    n_neurons = len(neu_train_mean)
    n_total   = 9 * n_per_class
    X_train = np.empty((n_total, n_neurons))
    X_test  = np.empty((n_total, n_neurons))
    y       = np.repeat(ALL_LOCS, n_per_class)
    for loc_idx in range(9):
        r0, r1 = loc_idx*n_per_class, (loc_idx+1)*n_per_class
        for n_idx in range(n_neurons):
            obs = train_arr[loc_idx][n_idx]
            X_train[r0:r1, n_idx] = (rng.choice(obs, n_per_class, replace=len(obs)<n_per_class)
                                     if len(obs) else neu_train_mean[n_idx])
            obs = test_arr[loc_idx][n_idx]
            X_test[r0:r1, n_idx]  = (rng.choice(obs, n_per_class, replace=len(obs)<n_per_class)
                                     if len(obs) else neu_train_mean[n_idx])
    return X_train, y, X_test, y

print("\nBuilding pseudo-trials...")
rng = np.random.default_rng(RANDOM_SEED)
Xtr, ytr, Xte, yte = build_pseudotrials(tr_arr, te_arr, neu_mean, n_per_class, rng)
print(f"X_train: {Xtr.shape}  NaN: {np.isnan(Xtr).sum()}")
print(f"X_test:  {Xte.shape}  NaN: {np.isnan(Xte).sum()}")

# fit decoder
print("\nFitting decoder...")
scaler = StandardScaler()
Xtr_z  = scaler.fit_transform(Xtr)
Xte_z  = scaler.transform(Xte)
clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                         multi_class='multinomial', random_state=RANDOM_SEED)
clf.fit(Xtr_z, ytr)
ypred = clf.predict(Xte_z)
ba = balanced_accuracy_score(yte, ypred)
ra = (ypred == yte).mean()
print(f"\noffset=0: balanced_acc={ba:.3f}  raw_acc={ra:.3f}  chance={1/9:.3f}")
print("\nPer-class recall:")
for loc in ALL_LOCS:
    mask = yte == loc
    recall = (ypred[mask] == loc).mean()
    print(f"  loc {loc}: {recall:.3f}  (n={mask.sum()})")

print("\nSTEP 1+2 test PASSED. Ready to run full decode_curr_rew_loc.py")
