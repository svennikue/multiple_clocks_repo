#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose the negative correlation between `state` and `dsr_old` model RDMs in
the ACROSS-config part. Identifies which condition pairs / state-phase
combinations / config pairs drive it most.

Each condition is labelled (config, state, phase) where:
    state ∈ {A, B, C, D}, phase ∈ {early, middle, late}.

Borrows the model-construction logic from RSA_DSR_ROIs_simple.py.
"""

import os
import sys
import json
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

import mc

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')


# ── Settings ─────────────────────────────────────────────────────────
DATA_DIR = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
OUT_BASE = os.path.join(DATA_DIR, 'group', 'state_dsr_anticorr_diagnosis')

configs = [
    '3-7-9-5', '8-2-6-7', '1-9-5-8', '4-8-1-3',
    '6-4-2-9', '9-1-3-4', '7-3-4-2', '2-5-7-6',
]
N_CONFIGS = len(configs)
N_CONDS_PER_CONF = 12
LEN_STANDARDISED_PATH = 10
N_PHASES = 3
states = ['A', 'B', 'C', 'D']
phases = ['early', 'middle', 'late']
RESOLUTIONx = 1
TOP_N = 25


RUN_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUT_DIR = os.path.join(OUT_BASE, RUN_TAG)
os.makedirs(OUT_BASE, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ── helpers (borrowed from main script) ──────────────────────────────
def downsample_mode(x, target_len=10):
    x = np.asarray(x, dtype=object)
    block = len(x) // target_len
    return np.array([
        Counter(x[i*block:(i+1)*block]).most_common(1)[0][0]
        for i in range(target_len)
    ], dtype=object)


def crosscorr_dist(M):
    """1 - Pearson(rows) -> (n_rows, n_rows) distance matrix."""
    M = np.asarray(M, dtype=float)
    Mc = M - M.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Mc, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Mn = Mc / norms
    return 1.0 - Mn @ Mn.T


# ── Load subject data, derive mode_locs (pooled over both halves) ────
with open(os.path.join(DATA_DIR, 'all_sessions_dsrRSA_grouping_summary.json')) as f:
    config_summary = json.load(f)
SUBJECTS = list(config_summary.keys())

print("Loading subject data...")
locs_pool = {c: [] for c in configs}
for sub_str in SUBJECTS:
    data_dict = mc.analyse.helpers_human_cells.load_norm_data(DATA_DIR, [sub_str])
    beh = data_dict[f"sub-{sub_str}"]['beh'].copy().reset_index(drop=True)
    beh['config'] = list(zip(
        beh['loc_A'].astype(int), beh['loc_B'].astype(int),
        beh['loc_C'].astype(int), beh['loc_D'].astype(int),
    ))
    beh['config_str'] = beh['config'].apply(
        lambda t: f'{t[0]}-{t[1]}-{t[2]}-{t[3]}')

    for conf in configs:
        idx = (beh['config_str'] == conf) & (beh['correct'] == 1)
        locs_pool[conf].append(
            data_dict[f"sub-{sub_str}"]['locations'][idx].to_numpy()
        )

mode_locs = {}
for c in configs:
    stacked = np.vstack(locs_pool[c])
    m = stats.mode(stacked, axis=0, keepdims=False, nan_policy='omit')
    mode_locs[c] = m.mode.astype(float)


# ── Build state design matrix ────────────────────────────────────────
state_config = np.zeros((N_CONDS_PER_CONF, len(states)))
for s_i in range(len(states)):
    start = RESOLUTIONx * s_i * N_PHASES
    state_config[start: RESOLUTIONx * (s_i + 1) * N_PHASES, s_i] = 1
state_concat = np.tile(state_config, (N_CONFIGS, 1))


# ── Build dsr_old design matrix ──────────────────────────────────────
dsr_old_list = []
for c in configs:
    walked = [int(w - 1) for w in mode_locs[c]]
    out = mc.simulation.predictions.model_DSR(
        locations=walked, no_phase_neurons=N_PHASES,
    )
    dsr_matrix = out[4]  # 5th return: dsr_matrix
    dsr_downsampled = dsr_matrix.reshape(
        dsr_matrix.shape[0],
        N_CONDS_PER_CONF,
        int(360 / N_CONDS_PER_CONF),
    ).mean(axis=2)
    dsr_old_list.append(dsr_downsampled)
dsr_concat = np.transpose(np.concatenate(dsr_old_list, axis=1))


# ── Compute RDMs ─────────────────────────────────────────────────────
state_rdm = crosscorr_dist(state_concat)
dsr_rdm = crosscorr_dist(dsr_concat)
n = state_rdm.shape[0]


# ── Per-condition labels ─────────────────────────────────────────────
labels = []
for c in configs:
    for sp in range(N_CONDS_PER_CONF):
        s_idx = sp // N_PHASES
        p_idx = sp % N_PHASES
        labels.append({
            'config':       c,
            'subpath':      sp,
            'state':        states[s_idx],
            'phase':        phases[p_idx],
            'state_phase':  f'{states[s_idx]}-{phases[p_idx]}',
        })


# ── Across-config upper-triangle pairs only ──────────────────────────
iu_i, iu_j = np.triu_indices(n, k=1)
config_i_all = np.array([labels[i]['config'] for i in iu_i])
config_j_all = np.array([labels[j]['config'] for j in iu_j])
across_mask = config_i_all != config_j_all

i_arr = iu_i[across_mask]
j_arr = iu_j[across_mask]
state_vec = state_rdm[i_arr, j_arr]
dsr_vec = dsr_rdm[i_arr, j_arr]

r_all, p_all = stats.pearsonr(state_vec, dsr_vec)
print(f"\nAcross-config Pearson r between state and dsr_old: "
      f"r = {r_all:.4f}, p = {p_all:.3e}, n_pairs = {len(state_vec)}")


# ── Per-pair contribution to the correlation ────────────────────────
mu_s, mu_d = state_vec.mean(), dsr_vec.mean()
sd_s, sd_d = state_vec.std(ddof=0), dsr_vec.std(ddof=0)
s_dev = state_vec - mu_s
d_dev = dsr_vec - mu_d
contrib = s_dev * d_dev          # numerator term per pair
norm = sd_s * sd_d * len(state_vec)   # divide by this to recover r

print(f"  sanity check: sum(contrib) / (sd_s*sd_d*n) = "
      f"{contrib.sum()/norm:.4f}  (should equal r)")


# ── Build pair-level dataframe ───────────────────────────────────────
pair_rows = []
for i, j, sd_v, dr_v, sd_dv, dr_dv, ctr in zip(
    i_arr, j_arr, state_vec, dsr_vec, s_dev, d_dev, contrib
):
    li, lj = labels[i], labels[j]
    pair_rows.append({
        'i': int(i),
        'j': int(j),
        'config_i':       li['config'],
        'config_j':       lj['config'],
        'state_i':        li['state'],
        'state_j':        lj['state'],
        'phase_i':        li['phase'],
        'phase_j':        lj['phase'],
        'state_phase_i':  li['state_phase'],
        'state_phase_j':  lj['state_phase'],
        'state_dist':     float(sd_v),
        'dsr_dist':       float(dr_v),
        's_dev':          float(sd_dv),
        'd_dev':          float(dr_dv),
        'contrib':        float(ctr),
    })
pair_df = pd.DataFrame(pair_rows)
pair_df['contrib_normed'] = pair_df['contrib'] / norm     # fraction of r per pair

# unordered keys for symmetric pair categories
pair_df['state_pair']       = pair_df.apply(
    lambda r: '-'.join(sorted([r['state_i'], r['state_j']])), axis=1)
pair_df['phase_pair']       = pair_df.apply(
    lambda r: '-'.join(sorted([r['phase_i'], r['phase_j']])), axis=1)
pair_df['state_phase_pair'] = pair_df.apply(
    lambda r: ' | '.join(sorted([r['state_phase_i'], r['state_phase_j']])), axis=1)
pair_df['config_pair']      = pair_df.apply(
    lambda r: '|'.join(sorted([r['config_i'], r['config_j']])), axis=1)
pair_df['same_state']       = pair_df['state_i'] == pair_df['state_j']
pair_df['same_phase']       = pair_df['phase_i'] == pair_df['phase_j']

# Sort by contribution (lowest = most anticorrelating)
pair_df = pair_df.sort_values('contrib').reset_index(drop=True)
pair_df.to_csv(os.path.join(OUT_DIR, 'across_pair_contributions.csv'), index=False)


# ── Aggregations ─────────────────────────────────────────────────────
def agg_by(df, group_cols):
    out = (df
           .groupby(group_cols)
           .agg(n_pairs=('contrib', 'size'),
                contrib_sum=('contrib', 'sum'),
                contrib_mean=('contrib', 'mean'),
                state_dist_mean=('state_dist', 'mean'),
                dsr_dist_mean=('dsr_dist', 'mean'))
           .reset_index())
    out['fraction_of_r'] = out['contrib_sum'] / norm
    return out.sort_values('contrib_sum').reset_index(drop=True)


agg_state_pair       = agg_by(pair_df, ['state_pair'])
agg_phase_pair       = agg_by(pair_df, ['phase_pair'])
agg_state_phase_pair = agg_by(pair_df, ['state_phase_pair'])
agg_same_flags       = agg_by(pair_df, ['same_state', 'same_phase'])
agg_config_pair      = agg_by(pair_df, ['config_pair'])

agg_state_pair.to_csv(os.path.join(OUT_DIR, 'agg_state_pair.csv'), index=False)
agg_phase_pair.to_csv(os.path.join(OUT_DIR, 'agg_phase_pair.csv'), index=False)
agg_state_phase_pair.to_csv(os.path.join(OUT_DIR, 'agg_state_phase_pair.csv'), index=False)
agg_same_flags.to_csv(os.path.join(OUT_DIR, 'agg_same_state_phase_flags.csv'), index=False)
agg_config_pair.to_csv(os.path.join(OUT_DIR, 'agg_config_pair.csv'), index=False)


# Per-condition involvement: total contribution over all partners
def condition_involvement(df, label_col):
    a = df.groupby(f'{label_col}_i')['contrib'].sum()
    b = df.groupby(f'{label_col}_j')['contrib'].sum()
    return a.add(b, fill_value=0).sort_values()


inv_state_phase = condition_involvement(pair_df, 'state_phase')
inv_state_phase.to_frame('contrib_sum').to_csv(
    os.path.join(OUT_DIR, 'involvement_state_phase.csv')
)


# ── Console summary ──────────────────────────────────────────────────
print(f"\n=== TOP {TOP_N} most negatively-contributing pairs ===")
print(pair_df.head(TOP_N)[
    ['config_i', 'state_phase_i', 'config_j', 'state_phase_j',
     'state_dist', 'dsr_dist', 'contrib', 'contrib_normed']
].to_string())

print("\n=== Most negative state-pair groupings (sum of contrib) ===")
print(agg_state_pair.to_string())

print("\n=== Most negative state_phase-pair groupings (top 15) ===")
print(agg_state_phase_pair.head(15).to_string())

print("\n=== Most negative phase-pair groupings ===")
print(agg_phase_pair.to_string())

print("\n=== Same-state vs same-phase flags ===")
print(agg_same_flags.to_string())

print("\n=== State_phase involvement (lowest = most anticorrelating) ===")
print(inv_state_phase.head(15).to_string())

print("\n=== Most-negative config-pair groupings (top 10) ===")
print(agg_config_pair.head(10).to_string())


# ── Plots ────────────────────────────────────────────────────────────
def colored_scatter(df, color_col, title, fname, cmap='tab20', s=4, alpha=0.4):
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    cats = sorted(df[color_col].unique())
    cmap_obj = plt.get_cmap(cmap, max(len(cats), 1))
    for i, cat in enumerate(cats):
        sub = df[df[color_col] == cat]
        ax.scatter(sub['state_dist'], sub['dsr_dist'], s=s, alpha=alpha,
                   color=cmap_obj(i), label=str(cat))
    ax.set_xlabel('state distance')
    ax.set_ylabel('dsr_old distance')
    ax.set_title(title)
    ax.legend(fontsize=7, markerscale=2, loc='best', ncol=2)
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
    plt.show()


# (1) RDM heatmaps with across mask shown
fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
im0 = axes[0].imshow(state_rdm, cmap='viridis')
axes[0].set_title('state RDM (full)')
plt.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(dsr_rdm, cmap='viridis')
axes[1].set_title('dsr_old RDM (full)')
plt.colorbar(im1, ax=axes[1])
fig.suptitle(f'across-config r = {r_all:.3f}, n_across_pairs = {len(state_vec)}')
fig.savefig(os.path.join(OUT_DIR, 'rdms.png'), dpi=150)
plt.show()


# (2) scatter colored by state_pair
colored_scatter(pair_df, 'state_pair',
                f'across-config pairs (r={r_all:.3f}) — colored by state pair',
                'scatter_state_pair.png')

# (3) scatter colored by state_phase_pair
colored_scatter(pair_df, 'state_phase_pair',
                f'across-config pairs (r={r_all:.3f}) — colored by state-phase pair',
                'scatter_state_phase_pair.png')


# (4) bar: contribution per state-pair
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
sp_sorted = agg_state_pair.sort_values('contrib_sum')
colors = ['tab:red' if v < 0 else 'tab:blue' for v in sp_sorted['contrib_sum']]
ax.bar(sp_sorted['state_pair'], sp_sorted['contrib_sum'], color=colors)
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('Σ (state-dev)(dsr-dev)')
ax.set_title('Across-config: drivers by state-pair (red = anticorrelating)')
plt.xticks(rotation=40, ha='right')
fig.savefig(os.path.join(OUT_DIR, 'bar_state_pair_contrib.png'), dpi=150)
plt.show()


# (5) bar: contribution per state_phase pair
fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
spp_sorted = agg_state_phase_pair.sort_values('contrib_sum')
colors = ['tab:red' if v < 0 else 'tab:blue' for v in spp_sorted['contrib_sum']]
ax.bar(range(len(spp_sorted)), spp_sorted['contrib_sum'], color=colors)
ax.set_xticks(range(len(spp_sorted)))
ax.set_xticklabels(spp_sorted['state_phase_pair'],
                   rotation=70, ha='right', fontsize=7)
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('Σ (state-dev)(dsr-dev)')
ax.set_title('Across-config: drivers by state-phase pair')
fig.savefig(os.path.join(OUT_DIR, 'bar_state_phase_pair_contrib.png'), dpi=150)
plt.show()


# (6) per-condition (state-phase) involvement bar (12 entries, summed over partners)
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
inv_sorted = inv_state_phase.sort_values()
colors = ['tab:red' if v < 0 else 'tab:blue' for v in inv_sorted.values]
ax.bar(inv_sorted.index, inv_sorted.values, color=colors)
ax.axhline(0, color='black', lw=0.8)
ax.set_ylabel('Σ contrib over all partners')
ax.set_title('Per-condition (state-phase) involvement in across-config r')
plt.xticks(rotation=45, ha='right', fontsize=8)
fig.savefig(os.path.join(OUT_DIR, 'bar_condition_involvement.png'), dpi=150)
plt.show()


# (7) heatmap: contribution per config-pair (8x8)
config_idx = {c: i for i, c in enumerate(configs)}
H = np.zeros((N_CONFIGS, N_CONFIGS))
for _, row in agg_config_pair.iterrows():
    a, b = row['config_pair'].split('|')
    ia, ib = config_idx[a], config_idx[b]
    H[ia, ib] = row['contrib_sum']
    H[ib, ia] = row['contrib_sum']

fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
vmax = np.nanmax(np.abs(H))
vmax = vmax if vmax > 0 else 1.0
im = ax.imshow(H, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax.set_xticks(range(N_CONFIGS))
ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(N_CONFIGS))
ax.set_yticklabels(configs, fontsize=8)
for i in range(N_CONFIGS):
    for j in range(N_CONFIGS):
        if i != j:
            ax.text(j, i, f'{H[i, j]:.2f}', ha='center', va='center', fontsize=6)
plt.colorbar(im, ax=ax, label='Σ contrib')
ax.set_title('Across-config: contribution per config-pair')
fig.savefig(os.path.join(OUT_DIR, 'heatmap_config_pair_contrib.png'), dpi=150)
plt.show()


# ── Save run config ──────────────────────────────────────────────────
with open(os.path.join(OUT_DIR, 'config.json'), 'w') as f:
    json.dump({
        'run_tag':         RUN_TAG,
        'data_dir':        DATA_DIR,
        'out_dir':         OUT_DIR,
        'configs':         configs,
        'N_CONDS_PER_CONF': N_CONDS_PER_CONF,
        'N_PHASES':        N_PHASES,
        'states':          states,
        'phases':          phases,
        'r_all':           float(r_all),
        'p_all':           float(p_all),
        'n_across_pairs':  int(len(state_vec)),
    }, f, indent=2)

print(f"\nResults written to: {OUT_DIR}")
