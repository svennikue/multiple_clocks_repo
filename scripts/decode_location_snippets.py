#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decode_location_snippets.py

Pseudo-population location decoder using snippet timecourses, with full bias
controls: behavioral baseline, current-location conditioning, within-group
shuffle null, prediction distribution check, and location tuning curves.

Source data: all_location_snippets.csv
  - Each row: one (neuron x grid_no x visit), 30-bin snippet (onset at bin 10)
Split:       snippet_train_test_assignment.csv
Decode targets: current_location, next+1_loc, next+2_loc, next+3_loc

Outputs (in RESULTS_DIR):
  snippet_decode_{target}_accuracy.csv
  snippet_decode_{target}_summary.csv
  snippet_decode_{target}_confusion.png
  snippet_decode_{target}_conditioned_summary.csv   (next+k only)
  snippet_decode_{target}_shuffle_summary.csv       (next+k only)
  snippet_decode_{target}_prediction_dist.csv
  snippet_decode_all_targets.png
  snippet_tuning_curves.png
"""

import os
import ast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR    = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
RESULTS_DIR = "/Users/xpsy1114/Documents/projects/multiple_clocks/results"

SNIPPETS_CSV = os.path.join(DATA_DIR, "all_location_snippets.csv")
ASSIGN_CSV   = os.path.join(RESULTS_DIR, "snippet_train_test_assignment.csv")
SPLIT_CSV    = os.path.join(RESULTS_DIR, "snippet_split_summary.csv")

SNIPPET_LEN   = 30
WIN_SIZE      = 5
N_WINDOWS     = SNIPPET_LEN // WIN_SIZE   # 6
WINDOW_LABELS = [f'W{i} (bins {i*WIN_SIZE}-{i*WIN_SIZE+WIN_SIZE-1})' for i in range(N_WINDOWS)]
ONSET_BIN     = 10

ALL_LOCS    = list(range(1, 10))
RANDOM_SEED = 42

DECODE_TARGETS = [
    ('current', 'current_location'),
    ('next+1',  'next+1_loc'),
    ('next+2',  'next+2_loc'),
    ('next+3',  'next+3_loc'),
]

LR_KWARGS = dict(max_iter=2000, solver='lbfgs', multi_class='multinomial',
                 C=1.0, random_state=RANDOM_SEED)

SNIPPET_COLS = [f'bin_{b:02d}' for b in range(SNIPPET_LEN)]


# ---------------------------------------------------------------------------
# Helper: extract integer location from a column that may be int, float,
# or a string like "[4]" or "[4, 5]"
# ---------------------------------------------------------------------------
def extract_loc(val):
    """Return first integer location from val, or NaN."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip()
    if s.startswith('['):
        try:
            lst = ast.literal_eval(s)
            return int(lst[0]) if lst else np.nan
        except Exception:
            return np.nan
    try:
        return int(float(s))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
print("Loading snippets ...")
snip = pd.read_csv(SNIPPETS_CSV)
snip = snip[snip['current_location'].isin(ALL_LOCS)].copy()
snip = snip.reset_index(drop=True)
print(f"  {len(snip)} rows | {snip['neuron_label'].nunique()} neurons")

print("Loading split assignment ...")
assign   = pd.read_csv(ASSIGN_CSV)
split_df = pd.read_csv(SPLIT_CSV)

qualifying = sorted(assign['neuron_label'].unique())
print(f"  Qualifying neurons: {len(qualifying)}")
n_neurons  = len(qualifying)
neu_to_idx = {n: i for i, n in enumerate(qualifying)}

snip = snip[snip['neuron_label'].isin(qualifying)].copy()
snip = snip.merge(
    assign.rename(columns={'location_label': 'current_location'}),
    on=['neuron_label', 'current_location', 'grid_no'],
    how='left'
)
snip = snip.dropna(subset=['set'])
print(f"  Snippets after assignment join: {len(snip)}")

for k in range(1, 4):
    col = f'next+{k}_loc'
    if col in snip.columns:
        snip[f'{col}_int'] = snip[col].apply(extract_loc)

for w in range(N_WINDOWS):
    cols = SNIPPET_COLS[w * WIN_SIZE : (w + 1) * WIN_SIZE]
    snip[f'win_{w}'] = snip[cols].mean(axis=1)

WIN_COLS = [f'win_{w}' for w in range(N_WINDOWS)]

# ---------------------------------------------------------------------------
# Suggested N_RESAMPLES from split_df
# ---------------------------------------------------------------------------
loc_stats = split_df.groupby('location_label').agg(
    min_train=('n_train', 'min'),
    max_train=('n_train', 'max'),
).reindex(ALL_LOCS)

excess_ratios = [
    loc_stats.loc[l, 'max_train'] / loc_stats.loc[l, 'min_train']
    for l in ALL_LOCS if loc_stats.loc[l, 'min_train'] > 0
]
mean_excess = np.mean(excess_ratios)
N_RESAMPLES = int(np.clip(10 * mean_excess, 10, 50))
print(f"\n  Mean max/min train ratio: {mean_excess:.2f}  -> N_RESAMPLES = {N_RESAMPLES}")


# ---------------------------------------------------------------------------
# Pool builder
# ---------------------------------------------------------------------------
def build_pools_for_label(df, label_col, win_col, target_locs=None):
    """
    Build pools[loc_idx][neu_idx] = np.array of window values,
    grouped by label_col. target_locs sets the class list; defaults to ALL_LOCS.
    Returns (pools, locs_used).
    """
    locs = target_locs if target_locs is not None else ALL_LOCS
    pools = [[None] * n_neurons for _ in range(len(locs))]
    valid = df[df[label_col].isin(locs)].copy()
    valid['_neu_idx'] = valid['neuron_label'].map(neu_to_idx)
    grp = valid.groupby([label_col, '_neu_idx'])[win_col].apply(np.array)
    for (loc, ni), arr in grp.items():
        if int(loc) not in locs:
            continue
        li = locs.index(int(loc))
        pools[li][int(ni)] = arr[~np.isnan(arr)]
    for li in range(len(locs)):
        for ni in range(n_neurons):
            if pools[li][ni] is None:
                pools[li][ni] = np.array([])
    return pools, locs


# ---------------------------------------------------------------------------
# Decoder core
# ---------------------------------------------------------------------------
def decode_one_window(train_pools, test_pools,
                      n_train_per_loc, n_test_per_loc,
                      n_resamples, seed,
                      locs=None, return_predictions=False):
    """
    Run pseudo-population decoder for one time window.

    Parameters
    ----------
    return_predictions : bool
        If True, also return concatenated predicted labels across all resamples.
    """
    if locs is None:
        locs = ALL_LOCS
    rng    = np.random.default_rng(seed)
    n_neu  = len(train_pools[0])
    n_locs = len(locs)
    scores = np.full(n_resamples, np.nan)
    all_preds = [] if return_predictions else None

    total_tr = sum(n_train_per_loc)
    total_te = sum(n_test_per_loc)

    for r in range(n_resamples):
        X_train = np.zeros((total_tr, n_neu))
        X_test  = np.zeros((total_te, n_neu))
        y_train = np.concatenate([np.full(n_train_per_loc[li], locs[li])
                                  for li in range(n_locs)])
        y_test  = np.concatenate([np.full(n_test_per_loc[li], locs[li])
                                  for li in range(n_locs)])
        ok = True
        tr_off = te_off = 0
        for li in range(n_locs):
            ntr, nte = n_train_per_loc[li], n_test_per_loc[li]
            for ni in range(n_neu):
                tr_obs = train_pools[li][ni]
                te_obs = test_pools[li][ni]
                if len(tr_obs) == 0 or len(te_obs) == 0:
                    ok = False
                    break
                X_train[tr_off:tr_off+ntr, ni] = rng.choice(
                    tr_obs, size=ntr, replace=len(tr_obs) < ntr)
                X_test[te_off:te_off+nte, ni] = rng.choice(
                    te_obs, size=nte, replace=len(te_obs) < nte)
            if not ok:
                break
            tr_off += ntr
            te_off += nte

        if not ok:
            continue

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_train)
        X_te_sc  = scaler.transform(X_test)
        np.nan_to_num(X_tr_sc, copy=False)
        np.nan_to_num(X_te_sc, copy=False)
        clf = LogisticRegression(**LR_KWARGS)
        clf.fit(X_tr_sc, y_train)
        preds = clf.predict(X_te_sc)
        scores[r] = balanced_accuracy_score(y_test, preds)
        if return_predictions:
            all_preds.extend(preds.tolist())

    return scores, all_preds


# ---------------------------------------------------------------------------
# Wrapper: decode across all windows (returns list of score arrays + pred lists)
# ---------------------------------------------------------------------------
def decode_all_windows(tr_pools_all, te_pools_all,
                       n_train_per_loc, n_test_per_loc,
                       base_seed, locs=None, return_predictions=False):
    seeds = np.random.SeedSequence(base_seed).spawn(N_WINDOWS)
    results = Parallel(n_jobs=N_WINDOWS)(
        delayed(decode_one_window)(
            tr_pools_all[w], te_pools_all[w],
            n_train_per_loc, n_test_per_loc,
            N_RESAMPLES,
            int(seeds[w].generate_state(1)[0]),
            locs=locs,
            return_predictions=return_predictions,
        )
        for w in range(N_WINDOWS)
    )
    scores_list = [r[0] for r in results]
    preds_list  = [r[1] for r in results]
    return scores_list, preds_list


# ---------------------------------------------------------------------------
# Behavioral baseline (transition-matrix predictor)
# ---------------------------------------------------------------------------
def compute_behavioral_baseline(df, label_col):
    """
    Build P(next_loc | current_loc) from df, predict argmax, return balanced_accuracy.
    Also return the transition matrix.
    """
    valid = df[df['current_location'].isin(ALL_LOCS) & df[label_col].isin(ALL_LOCS)].copy()
    trans = pd.crosstab(valid['current_location'], valid[label_col])
    trans = trans.reindex(index=ALL_LOCS, columns=ALL_LOCS, fill_value=0)
    trans_prob = trans.div(trans.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    argmax_next = {c: trans_prob.loc[c].idxmax() for c in ALL_LOCS}
    valid['predicted'] = valid['current_location'].map(argmax_next)
    ba = balanced_accuracy_score(valid[label_col], valid['predicted'])
    return ba, trans_prob


# ---------------------------------------------------------------------------
# Conditioned-on-current-location decode (next+k only)
# ---------------------------------------------------------------------------
def decode_conditioned_on_current_loc(train_df, test_df, label_col, win_col,
                                      base_seed, current_locs=None):
    """
    For each current_location c, decode label_col using only rows where
    current_location == c. Returns per-current-loc results dict.
    """
    if current_locs is None:
        current_locs = ALL_LOCS

    results_by_cloc = {}

    for c in current_locs:
        tr_sub = train_df[train_df['current_location'] == c].copy()
        te_sub = test_df[test_df['current_location'] == c].copy()

        present_locs = sorted(set(
            tr_sub[label_col].dropna().astype(int).unique()
        ).intersection(
            te_sub[label_col].dropna().astype(int).unique()
        ))
        present_locs = [l for l in present_locs if l in ALL_LOCS]

        if len(present_locs) < 2:
            continue

        tr_pools_all = [build_pools_for_label(tr_sub, label_col, f'win_{w}',
                                               target_locs=present_locs)[0]
                        for w in range(N_WINDOWS)]
        te_pools_all = [build_pools_for_label(te_sub, label_col, f'win_{w}',
                                               target_locs=present_locs)[0]
                        for w in range(N_WINDOWS)]

        # Balance: subsample to min count across next_loc classes
        def min_counts_for_sub(pools):
            counts = []
            for li in range(len(present_locs)):
                loc_min = min(
                    len(pools[li][ni]) for ni in range(n_neurons)
                    if len(pools[li][ni]) > 0
                )
                counts.append(loc_min)
            return counts

        n_train_sub = min_counts_for_sub(tr_pools_all[0])
        n_test_sub  = min_counts_for_sub(te_pools_all[0])

        if min(n_train_sub) == 0 or min(n_test_sub) == 0:
            continue

        seed_c = base_seed + c * 1000
        scores_list, _ = decode_all_windows(
            tr_pools_all, te_pools_all,
            n_train_sub, n_test_sub,
            seed_c, locs=present_locs
        )

        n_obs = len(tr_sub[tr_sub[label_col].isin(present_locs)]) + \
                len(te_sub[te_sub[label_col].isin(present_locs)])

        results_by_cloc[c] = {
            'scores_per_window': scores_list,
            'present_locs': present_locs,
            'n_obs': n_obs,
            'n_train_sub': n_train_sub,
            'n_test_sub': n_test_sub,
        }

    return results_by_cloc


# ---------------------------------------------------------------------------
# Within-current-loc shuffle control
# ---------------------------------------------------------------------------
def decode_with_shuffle(train_df, test_df, label_col,
                        tr_pools_fn_args, te_pools_fn_args,
                        n_train_per_loc, n_test_per_loc,
                        base_seed, n_shuffle=20):
    """
    Permute label_col within each current_location group (train set only),
    then build pools and decode. Repeat n_shuffle times.
    Returns scores array of shape (n_shuffle, N_WINDOWS).
    """
    shuffle_scores = np.full((n_shuffle, N_WINDOWS), np.nan)
    rng_shuf = np.random.default_rng(base_seed + 77777)

    for s in range(n_shuffle):
        tr_shuf = train_df.copy()
        for c in ALL_LOCS:
            mask = tr_shuf['current_location'] == c
            idx  = tr_shuf.index[mask]
            vals = tr_shuf.loc[idx, label_col].values.copy()
            rng_shuf.shuffle(vals)
            tr_shuf.loc[idx, label_col] = vals

        tr_pools_all = [build_pools_for_label(tr_shuf, label_col, f'win_{w}')[0]
                        for w in range(N_WINDOWS)]
        te_pools_all = [build_pools_for_label(test_df, label_col, f'win_{w}')[0]
                        for w in range(N_WINDOWS)]

        seed_s = int(rng_shuf.integers(0, 2**31))
        scores_list, _ = decode_all_windows(
            tr_pools_all, te_pools_all,
            n_train_per_loc, n_test_per_loc,
            seed_s
        )
        for w in range(N_WINDOWS):
            valid = scores_list[w][~np.isnan(scores_list[w])]
            shuffle_scores[s, w] = valid.mean() if len(valid) > 0 else np.nan

    return shuffle_scores


# ---------------------------------------------------------------------------
# Prediction distribution check
# ---------------------------------------------------------------------------
def check_prediction_distribution(preds_list, target_name):
    """
    preds_list: list of N_WINDOWS lists of predicted labels.
    Returns a DataFrame with freq per class per window, plus entropy flags.
    """
    rows = []
    for w, preds in enumerate(preds_list):
        if preds is None or len(preds) == 0:
            continue
        preds_arr = np.array(preds)
        total = len(preds_arr)
        for loc in ALL_LOCS:
            freq = np.sum(preds_arr == loc) / total
            rows.append({'window': w, 'predicted_loc': loc, 'frequency': freq})

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        return pred_df

    entropy_rows = []
    for w in pred_df['window'].unique():
        freqs = pred_df[pred_df['window'] == w]['frequency'].values
        freqs_safe = freqs[freqs > 0]
        ent = -np.sum(freqs_safe * np.log(freqs_safe))
        max_ent = np.log(len(ALL_LOCS))
        flag = any(freqs > 0.30)
        entropy_rows.append({'window': w, 'entropy': ent,
                              'max_entropy': max_ent,
                              'dominant_class_flag': flag})
        if flag:
            dominant = pred_df[(pred_df['window'] == w) &
                               (pred_df['frequency'] > 0.30)]['predicted_loc'].values
            print(f"  [WARN] {target_name} W{w}: dominant class {dominant} "
                  f"captures >30% predictions (entropy={ent:.3f}/{max_ent:.3f})")

    return pred_df


# ---------------------------------------------------------------------------
# Per-class recall helper
# ---------------------------------------------------------------------------
def compute_per_class_recall(train_pools, test_pools,
                              n_train_per_loc, n_test_per_loc,
                              seed, locs=None):
    if locs is None:
        locs = ALL_LOCS
    rng    = np.random.default_rng(seed)
    n_neu  = len(train_pools[0])
    n_locs = len(locs)
    conf_mats = []
    total_tr = sum(n_train_per_loc)
    total_te = sum(n_test_per_loc)

    for _ in range(N_RESAMPLES):
        X_train = np.zeros((total_tr, n_neu))
        X_test  = np.zeros((total_te, n_neu))
        y_train = np.concatenate([np.full(n_train_per_loc[li], locs[li])
                                  for li in range(n_locs)])
        y_test  = np.concatenate([np.full(n_test_per_loc[li], locs[li])
                                  for li in range(n_locs)])
        ok = True
        tr_off = te_off = 0
        for li in range(n_locs):
            ntr, nte = n_train_per_loc[li], n_test_per_loc[li]
            for ni in range(n_neu):
                tr_obs = train_pools[li][ni]
                te_obs = test_pools[li][ni]
                if len(tr_obs) == 0 or len(te_obs) == 0:
                    ok = False
                    break
                X_train[tr_off:tr_off+ntr, ni] = rng.choice(
                    tr_obs, size=ntr, replace=len(tr_obs) < ntr)
                X_test[te_off:te_off+nte, ni] = rng.choice(
                    te_obs, size=nte, replace=len(te_obs) < nte)
            if not ok:
                break
            tr_off += ntr
            te_off += nte
        if not ok:
            continue
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)
        np.nan_to_num(X_tr_sc, copy=False)
        np.nan_to_num(X_te_sc, copy=False)
        clf = LogisticRegression(**LR_KWARGS)
        clf.fit(X_tr_sc, y_train)
        conf_mats.append(confusion_matrix(y_test, clf.predict(X_te_sc), labels=locs))

    return conf_mats


# ---------------------------------------------------------------------------
# Tuning curves
# ---------------------------------------------------------------------------
def plot_tuning_curves(train_df):
    """
    For each location L in ALL_LOCS, z-score each neuron's snippet, then
    average across neurons. Plot mean ± sem timecourse (30 bins, onset at bin 10).
    Save as snippet_tuning_curves.png.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey=True)
    axes_flat = axes.flatten()
    x = np.arange(SNIPPET_LEN)

    for idx, loc in enumerate(ALL_LOCS):
        ax = axes_flat[idx]
        sub = train_df[train_df['current_location'] == loc][SNIPPET_COLS].values.astype(float)

        if sub.shape[0] == 0:
            ax.set_title(f'Loc {loc} (no data)')
            continue

        # z-score each neuron (row) across its 30 bins
        row_mean = sub.mean(axis=1, keepdims=True)
        row_std  = sub.std(axis=1, keepdims=True)
        row_std[row_std == 0] = 1.0
        sub_z = (sub - row_mean) / row_std

        mean_tc = sub_z.mean(axis=0)
        sem_tc  = sub_z.std(axis=0) / np.sqrt(sub_z.shape[0])

        ax.fill_between(x, mean_tc - sem_tc, mean_tc + sem_tc, alpha=0.3)
        ax.plot(x, mean_tc, linewidth=1.5)
        ax.axvline(ONSET_BIN, color='red', linestyle='--', linewidth=1, label='Onset')
        ax.set_title(f'Location {loc}  (n={sub.shape[0]})', fontsize=9)
        ax.set_xlabel('Bin', fontsize=8)
        ax.set_ylabel('z-score', fontsize=8)
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle('Location tuning curves (train set, z-scored snippets)', fontsize=12)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, 'snippet_tuning_curves.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved tuning curves: {out}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
train_df = snip[snip['set'] == 'train'].copy()
test_df  = snip[snip['set'] == 'test'].copy()

all_summaries         = {}   # target_name -> summary_df
all_cond_summaries    = {}   # target_name -> per-current-loc mean BA per window
all_shuffle_summaries = {}   # target_name -> (mean, sem) per window
all_behav_baselines   = {}   # target_name -> scalar BA

for target_name, raw_col in DECODE_TARGETS:
    print(f"\n{'='*60}")
    print(f"Decode target: {target_name}  (column: {raw_col})")
    print(f"{'='*60}")

    is_future = raw_col != 'current_location'
    label_col = raw_col if not is_future else f'{raw_col}_int'

    # ---- count runs per (neuron, location) --------------------------------
    def count_runs(df, lcol):
        valid = df[df[lcol].isin(ALL_LOCS)].copy()
        return (valid.groupby(['neuron_label', lcol, 'grid_no'])
                     .size().reset_index()
                     .groupby(['neuron_label', lcol])
                     .size()
                     .unstack(fill_value=0)
                     .reindex(columns=ALL_LOCS, fill_value=0))

    tr_counts = count_runs(train_df, label_col)
    te_counts = count_runs(test_df,  label_col)
    both      = tr_counts.index.intersection(te_counts.index)
    tr_counts = tr_counts.loc[both]
    te_counts = te_counts.loc[both]

    n_train_per_loc = [int(tr_counts[loc].min()) for loc in ALL_LOCS]
    n_test_per_loc  = [int(te_counts[loc].min())  for loc in ALL_LOCS]

    print(f"  Min train per loc: {dict(zip(ALL_LOCS, n_train_per_loc))}")
    print(f"  Min test  per loc: {dict(zip(ALL_LOCS, n_test_per_loc))}")

    if min(n_train_per_loc) == 0 or min(n_test_per_loc) == 0:
        print(f"  WARNING: some locations have 0 obs for {target_name}, skipping.")
        continue

    # ---- build pools for all windows --------------------------------------
    print("  Building pools ...")
    tr_pools_all = [build_pools_for_label(train_df, label_col, f'win_{w}')[0]
                    for w in range(N_WINDOWS)]
    te_pools_all = [build_pools_for_label(test_df,  label_col, f'win_{w}')[0]
                    for w in range(N_WINDOWS)]

    target_idx = [t[0] for t in DECODE_TARGETS].index(target_name)
    base_seed  = RANDOM_SEED + target_idx * 10000

    # ---- main decode across all windows -----------------------------------
    print("  Decoding main ...")
    scores_list, preds_list = decode_all_windows(
        tr_pools_all, te_pools_all,
        n_train_per_loc, n_test_per_loc,
        base_seed, return_predictions=True
    )

    # Save accuracy rows
    rows = []
    for w, scores in enumerate(scores_list):
        for r, sc in enumerate(scores):
            rows.append({'window': w, 'window_label': WINDOW_LABELS[w],
                         'resample': r, 'balanced_accuracy': sc,
                         'target': target_name})
    acc_df   = pd.DataFrame(rows)
    acc_path = os.path.join(RESULTS_DIR, f'snippet_decode_{target_name}_accuracy.csv')
    acc_df.to_csv(acc_path, index=False)

    summary_rows = []
    for w in range(N_WINDOWS):
        sc = acc_df[acc_df['window'] == w]['balanced_accuracy'].dropna()
        summary_rows.append({'window': w, 'window_label': WINDOW_LABELS[w],
                             'mean_ba': sc.mean(), 'sem_ba': sc.sem(),
                             'n_resamples': len(sc)})
    summary_df = pd.DataFrame(summary_rows)
    sum_path   = os.path.join(RESULTS_DIR, f'snippet_decode_{target_name}_summary.csv')
    summary_df.to_csv(sum_path, index=False)
    all_summaries[target_name] = summary_df

    print(f"  Chance = {1/9:.3f}")
    print(summary_df[['window_label', 'mean_ba', 'sem_ba']].to_string(index=False))

    # ---- prediction distribution ------------------------------------------
    print("  Checking prediction distribution ...")
    pred_dist_df = check_prediction_distribution(preds_list, target_name)
    if not pred_dist_df.empty:
        pred_dist_df.to_csv(
            os.path.join(RESULTS_DIR, f'snippet_decode_{target_name}_prediction_dist.csv'),
            index=False
        )

    # ---- per-class recall + confusion at best window ----------------------
    best_w   = int(summary_df['mean_ba'].idxmax())
    conf_mats = compute_per_class_recall(
        tr_pools_all[best_w], te_pools_all[best_w],
        n_train_per_loc, n_test_per_loc,
        seed=base_seed + 999
    )

    if conf_mats:
        mean_conf      = np.mean(conf_mats, axis=0)
        mean_conf_norm = mean_conf / mean_conf.sum(axis=1, keepdims=True)

        # Save per-class recall table for best window
        recall_rows = []
        for w in range(N_WINDOWS):
            conf_all = compute_per_class_recall(
                tr_pools_all[w], te_pools_all[w],
                n_train_per_loc, n_test_per_loc,
                seed=base_seed + 1000 + w
            )
            if not conf_all:
                continue
            mc   = np.mean(conf_all, axis=0)
            mcn  = mc / mc.sum(axis=1, keepdims=True)
            for li, loc in enumerate(ALL_LOCS):
                recall_rows.append({'window': w, 'location': loc,
                                    'recall': mcn[li, li]})
        recall_df = pd.DataFrame(recall_rows)
        recall_df.to_csv(
            os.path.join(RESULTS_DIR, f'snippet_decode_{target_name}_per_class_recall.csv'),
            index=False
        )

        # Confusion matrix plot
        n_locs = len(ALL_LOCS)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mean_conf_norm, vmin=0, cmap='Blues')
        ax.set_xticks(range(n_locs)); ax.set_xticklabels(ALL_LOCS)
        ax.set_yticks(range(n_locs)); ax.set_yticklabels(ALL_LOCS)
        ax.set_xlabel('Predicted location'); ax.set_ylabel('True location')
        ax.set_title(f'Confusion -- {target_name}, W{best_w} ({WINDOW_LABELS[best_w]})\n'
                     f'mean BA = {summary_df.loc[best_w,"mean_ba"]:.3f}')
        plt.colorbar(im, ax=ax, label='Recall')
        for i in range(n_locs):
            for j in range(n_locs):
                ax.text(j, i, f'{mean_conf_norm[i,j]:.2f}', ha='center', va='center',
                        fontsize=7, color='white' if mean_conf_norm[i,j] > 0.5 else 'black')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'snippet_decode_{target_name}_confusion.png'), dpi=150)
        plt.close()

    # ---- future-location extras -------------------------------------------
    if is_future:

        # 1. Behavioral baseline
        print("  Computing behavioral baseline ...")
        behav_ba, trans_prob = compute_behavioral_baseline(snip, label_col)
        all_behav_baselines[target_name] = behav_ba
        print(f"  Behavioral baseline BA = {behav_ba:.3f}")

        # 2. Conditioned-on-current-location decode
        print("  Conditioned decode (per current_location) ...")
        cond_results = decode_conditioned_on_current_loc(
            train_df, test_df, label_col, 'win_0',
            base_seed=base_seed + 5000
        )

        cond_rows = []
        if cond_results:
            total_weight   = sum(v['n_obs'] for v in cond_results.values())
            for c, res in cond_results.items():
                for w in range(N_WINDOWS):
                    sc = res['scores_per_window'][w]
                    valid_sc = sc[~np.isnan(sc)]
                    mean_ba  = valid_sc.mean() if len(valid_sc) > 0 else np.nan
                    cond_rows.append({
                        'current_loc': c,
                        'window': w,
                        'window_label': WINDOW_LABELS[w],
                        'mean_ba': mean_ba,
                        'n_obs': res['n_obs'],
                        'present_locs': str(res['present_locs']),
                    })

            cond_df = pd.DataFrame(cond_rows)
            cond_df.to_csv(
                os.path.join(RESULTS_DIR,
                             f'snippet_decode_{target_name}_conditioned_summary.csv'),
                index=False
            )

            # Weighted mean BA per window
            cond_window_mean = {}
            for w in range(N_WINDOWS):
                wdf = cond_df[cond_df['window'] == w].dropna(subset=['mean_ba'])
                if len(wdf) == 0:
                    cond_window_mean[w] = np.nan
                    continue
                weights   = wdf['n_obs'].values.astype(float)
                weights  /= weights.sum()
                cond_window_mean[w] = float(np.average(wdf['mean_ba'].values, weights=weights))

            all_cond_summaries[target_name] = cond_window_mean
            print(f"  Conditioned BA (weighted mean, best window): "
                  f"{max(cond_window_mean.values(), default=np.nan):.3f}")
        else:
            print("  Conditioned decode: insufficient data for any current_loc.")

        # 3. Shuffle control (within-current-loc)
        print("  Running shuffle control ...")
        N_SHUFFLE = min(N_RESAMPLES, 20)
        shuf_scores = decode_with_shuffle(
            train_df, test_df, label_col,
            None, None,
            n_train_per_loc, n_test_per_loc,
            base_seed=base_seed + 9999,
            n_shuffle=N_SHUFFLE
        )

        shuf_rows = []
        shuf_window_stats = {}
        for w in range(N_WINDOWS):
            col_sc = shuf_scores[:, w]
            valid  = col_sc[~np.isnan(col_sc)]
            m      = valid.mean() if len(valid) > 0 else np.nan
            s      = (valid.std() / np.sqrt(len(valid))) if len(valid) > 1 else np.nan
            shuf_window_stats[w] = (m, s)
            shuf_rows.append({'window': w, 'window_label': WINDOW_LABELS[w],
                              'shuffle_mean_ba': m, 'shuffle_sem_ba': s,
                              'n_shuffle': N_SHUFFLE})
        shuf_df = pd.DataFrame(shuf_rows)
        shuf_df.to_csv(
            os.path.join(RESULTS_DIR,
                         f'snippet_decode_{target_name}_shuffle_summary.csv'),
            index=False
        )
        all_shuffle_summaries[target_name] = shuf_window_stats
        print(f"  Shuffle BA (mean over windows): "
              f"{np.nanmean([v[0] for v in shuf_window_stats.values()]):.3f}")

    print(f"  Saved: {acc_path}, {sum_path}")


# ---------------------------------------------------------------------------
# Combined timecourse plot
# ---------------------------------------------------------------------------
if all_summaries:
    fig, axes_list = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes_grid = axes_list.flatten()
    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
    x = np.arange(N_WINDOWS)

    for ax_idx, ((target_name, raw_col), color) in enumerate(zip(DECODE_TARGETS, colors)):
        if target_name not in all_summaries:
            continue
        ax = axes_grid[ax_idx]
        sd = all_summaries[target_name]
        means = sd['mean_ba'].values
        sems  = sd['sem_ba'].values

        ax.axhline(1/9, color='gray', linestyle='--', linewidth=1, label='Chance (1/9)')
        ax.axvline(ONSET_BIN / WIN_SIZE - 0.5, color='black', linestyle=':',
                   linewidth=1, label='Onset')

        ax.fill_between(x, means - sems, means + sems, alpha=0.25, color=color)
        ax.plot(x, means, marker='o', linewidth=2, color=color, label='Neural decoder')

        is_future = raw_col != 'current_location'

        if is_future and target_name in all_behav_baselines:
            ba_beh = all_behav_baselines[target_name]
            ax.axhline(ba_beh, color='purple', linestyle='-.', linewidth=1.5,
                       label=f'Behavioral baseline ({ba_beh:.3f})')

        if is_future and target_name in all_cond_summaries:
            cond_means = np.array([all_cond_summaries[target_name].get(w, np.nan)
                                   for w in range(N_WINDOWS)])
            ax.plot(x, cond_means, marker='s', linewidth=1.5, linestyle='--',
                    color='teal', label='Conditioned decode')

        if is_future and target_name in all_shuffle_summaries:
            shuf_m = np.array([all_shuffle_summaries[target_name][w][0]
                                for w in range(N_WINDOWS)])
            shuf_s = np.array([all_shuffle_summaries[target_name][w][1]
                                for w in range(N_WINDOWS)])
            ax.fill_between(x, shuf_m - shuf_s, shuf_m + shuf_s,
                            alpha=0.15, color='dimgray')
            ax.plot(x, shuf_m, marker='x', linewidth=1.5, linestyle=':',
                    color='dimgray', label='Shuffle null')

        ax.set_xticks(x)
        ax.set_xticklabels([f'W{i}\n{i*WIN_SIZE}-{i*WIN_SIZE+WIN_SIZE-1}'
                            for i in range(N_WINDOWS)], fontsize=8)
        ax.set_xlabel('Time window (snippet bin)')
        ax.set_ylabel('Balanced accuracy')
        ax.set_title(f'Decode: {target_name}')
        ax.legend(fontsize=7)

    fig.suptitle(f'Location decoding\n{n_neurons} neurons, {N_RESAMPLES} resamples',
                 fontsize=12)
    plt.tight_layout()
    combined_path = os.path.join(RESULTS_DIR, 'snippet_decode_all_targets.png')
    plt.savefig(combined_path, dpi=150)
    plt.close()
    print(f"\nSaved combined plot: {combined_path}")


# ---------------------------------------------------------------------------
# Tuning curves
# ---------------------------------------------------------------------------
plot_tuning_curves(train_df)

print("\nDone.")
