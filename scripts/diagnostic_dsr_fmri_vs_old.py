#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic: dsr_fmri vs dsr_old on two hand-crafted configs.

Goal: understand why the two location-DSR formulations only correlate ~0.5
in the real ROI pipeline. Builds each model with the EXACT same code path as
``scripts/RSA_DSR_ROIs_simple.py``:

    dsr_fmri / loc_fmri  : mode-path -> integer location IDs (downsample_mode) ->
                           rolled trajectory per bin (build_mode_path_dsr) ->
                           Hamming dissimilarity RDM
                           (compute_hamming_distance_within).

    dsr_old / midnight / loc_old / state / phase  :
                           mode-path -> mc.simulation.predictions.model_DSR
                           (one-hot state x phase x location features) ->
                           cosine dissimilarity RDM
                           (compute_crosscorr_within).

Two configs (chosen by the user):
    config 1: rewards at 1,7,5,3 ; path 3-2-1-4-7-8-5-6-3
    config 2: rewards at 5,9,4,3 ; path 3-2-5-8-9-6-5-4-1-2-3

Outputs to OUT_DIR:
    1_paths.png                — paths on 3x3 grid
    2_walked_trajectories.png  — 360-bin walked vectors
    3_model_activations.png    — per-model 12 x feature activation per config
    4_rdms_<variant>.png       — between-config RDMs for full / within-phase /
                                 across-phase variants
    5_scatter_agreement.png    — per-cell agreement between Hamming and cosine
                                 model RDMs, coloured by phase relation
    correlations.txt           — printed correlation table

@author: Svenja Kuechenhoff
"""

import os
from collections import Counter
from datetime import date

import numpy as np
import matplotlib.pyplot as plt

import mc.simulation.predictions as predictions
import mc.analyse.my_RSA as my_RSA


# ── Settings (mirror RSA_DSR_ROIs_simple.py) ────────────────────────────
N_CONDS_PER_CONFIG    = 12
LEN_STANDARDISED_PATH = 12
N_PHASES              = 3
N_STATES              = 4
N_LOCATIONS           = 9
N_RAW_BINS            = 360
BINS_PER_STATE        = N_RAW_BINS // N_STATES            # 90
BINLEN_PER_COND       = N_RAW_BINS // N_CONDS_PER_CONFIG  # 30
LEN_OG_SUBPATH        = BINLEN_PER_COND                   # 30, matches script

OUT_DIR = (
    '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/'
    f'derivatives/group/diagnostic_dsr_fmri_vs_old_{date.today().isoformat()}'
)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Saving diagnostic outputs to: {OUT_DIR}")


# ── Hand-crafted configs ────────────────────────────────────────────────
# Each config: a list of 4 sub-paths (state A,B,C,D), where each sub-path
# begins at the previous reward and ends at the current state's reward.
configs = {
    'rewards_1-7-5-3': {
        'rewards':  [1, 7, 5, 3],
        'subpaths': [[3, 2, 1],
                     [1, 4, 7],
                     [7, 8, 5],
                     [5, 6, 3]],
    },
    'rewards_5-9-4-3': {
        'rewards':  [5, 9, 4, 3],
        'subpaths': [[3, 2, 5],
                     [5, 8, 9],
                     [9, 6, 5, 4],
                     [4, 1, 2, 3]],
    },
}


# ── Helpers copied verbatim from RSA_DSR_ROIs_simple.py ─────────────────
def downsample_mode(x, target_len):
    """Mode-downsample x to ``target_len`` slots, distributing input bins
    EVENLY across slots so that NO bins are silently discarded.

    The earlier ``block = len(x) // target_len`` version truncated: with
    ``len(x)=360`` and ``target_len=144`` it used only 288 of 360 bins,
    and every cond's window was misaligned by 6 bins from the conceptual
    30-bin layout. This version uses
    ``slot i = x[(i*n)//target : ((i+1)*n)//target]`` so all bins are
    used; slot sizes differ by at most 1.
    """
    x = np.asarray(x, dtype=object)
    n = len(x)
    return np.array([
        Counter(x[(i * n) // target_len:((i + 1) * n) // target_len])
            .most_common(1)[0][0]
        for i in range(target_len)
    ], dtype=object)


def build_mode_path_dsr(mode_vec, n_conds_per_config, len_per_bin):
    base = downsample_mode(mode_vec, target_len=n_conds_per_config * len_per_bin)
    return np.stack([np.roll(base, -pos * len_per_bin)
                     for pos in range(n_conds_per_config)], axis=0)


def build_walked_360(subpaths):
    """Stretch each sub-path into BINS_PER_STATE (=90) raw bins by even
    repetition, so the full walked vector has length 4*90 = 360. This matches
    the segmentation that model_DSR expects (4 equal-length state chunks)."""
    walked = []
    for sub in subpaths:
        n = len(sub)
        reps = [BINS_PER_STATE // n] * n
        for i in range(BINS_PER_STATE % n):
            reps[i] += 1
        for loc, r in zip(sub, reps):
            walked.extend([loc] * r)
    return np.asarray(walked, dtype=int)


# ── Build the walked vectors ────────────────────────────────────────────
walked = {name: build_walked_360(cfg['subpaths']) for name, cfg in configs.items()}
for name, w in walked.items():
    print(f"  {name}: walked length = {len(w)}, "
          f"distinct locations = {sorted(set(w.tolist()))}")


# ── Build per-config feature matrices for each model ────────────────────
# Each entry: (N_CONDS_PER_CONFIG, n_features) - one row per condition.

per_model_per_config = {m: {} for m in (
    'loc_fmri', 'dsr_fmri', 'loc_old', 'midnight', 'dsr_old', 'phase', 'state')}

for name, w in walked.items():
    N = N_CONDS_PER_CONFIG

    # ---- Hamming-route models (mirror the script's mats['loc'] and mats['dsr_fmri']) ----
    loc_mat = np.zeros((N, LEN_STANDARDISED_PATH), dtype=float)
    for n_subpath in range(N):
        subpath = w[n_subpath * LEN_OG_SUBPATH:(n_subpath + 1) * LEN_OG_SUBPATH]
        loc_mat[n_subpath] = np.asarray(
            downsample_mode(subpath, target_len=LEN_STANDARDISED_PATH),
            dtype=float)
    per_model_per_config['loc_fmri'][name] = loc_mat

    dsr_fmri_mat = build_mode_path_dsr(w, N, LEN_STANDARDISED_PATH).astype(float)
    per_model_per_config['dsr_fmri'][name] = dsr_fmri_mat

    # ---- Cosine-route models (via mc.simulation.predictions.model_DSR) ----
    walked_0idx = (w - 1).tolist()
    loc_og, phase_og, state_og, midn_m, dsr_m, _, _ = (
        predictions.model_DSR(locations=walked_0idx,
                              no_phase_neurons=N_PHASES))

    def _ds(M):
        return M.reshape(M.shape[0], N, BINLEN_PER_COND).mean(axis=2).T  # (N, n_neur)

    per_model_per_config['loc_old'][name]  = _ds(loc_og)
    per_model_per_config['phase'][name]    = _ds(phase_og)
    per_model_per_config['state'][name]    = _ds(state_og)
    per_model_per_config['midnight'][name] = _ds(midn_m)
    per_model_per_config['dsr_old'][name]  = _ds(dsr_m)


# ── Stack across configs (24 conditions = 2 configs x 12) ───────────────
model_stack = {
    m: np.vstack([per_model_per_config[m][name] for name in configs])
    for m in per_model_per_config
}


# ── Compute RDMs (within-config / between-config / full square) ─────────
hamming_models = ('loc_fmri', 'dsr_fmri')
cosine_models  = ('loc_old', 'midnight', 'dsr_old', 'phase', 'state')

rdm_across_vec = {}   # 1-D vector of between-config pairs
for m in hamming_models:
    _w, _across, _full = my_RSA.compute_hamming_distance_within(
        model_stack[m], plotting=False, include_diagonal=False,
        model_name=m, no_tasks=2, block_size=N_CONDS_PER_CONFIG)
    rdm_across_vec[m] = np.asarray(_across[0], dtype=float)

for m in cosine_models:
    _w, _across, _full = my_RSA.compute_crosscorr_within(
        model_stack[m], plotting=False, include_diagonal=False,
        model=m, no_tasks=2, block_size=N_CONDS_PER_CONFIG)
    rdm_across_vec[m] = np.asarray(_across[0], dtype=float)


# ── Phase masks aligned with the between-config vector ─────────────────
n_total = 2 * N_CONDS_PER_CONFIG     # 24
phase_vec = np.tile(np.arange(N_CONDS_PER_CONFIG) % N_PHASES, 2)
ii, jj = np.triu_indices(n_total, k=1)
between_cfg = (ii // N_CONDS_PER_CONFIG) != (jj // N_CONDS_PER_CONFIG)
same_phase = phase_vec[ii] == phase_vec[jj]
mask_within_phase = same_phase[between_cfg]
mask_across_phase = ~mask_within_phase


def _corr(a, b, mask=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if mask is not None:
        a, b = a[mask], b[mask]
    finite = np.isfinite(a) & np.isfinite(b)
    if (finite.sum() < 3 or np.nanstd(a[finite]) == 0
            or np.nanstd(b[finite]) == 0):
        return float('nan')
    return float(np.corrcoef(a[finite], b[finite])[0, 1])


# ── Correlation table ──────────────────────────────────────────────────
print("\n=== RDM-vector correlations (between-config pairs) ===")
header = f"{'pair':<35s} {'full':>8s} {'within_p':>9s} {'across_p':>9s}"
print(header)
lines = [header]
pairs_to_compare = [
    ('dsr_fmri', 'dsr_old'),
    ('loc_fmri', 'loc_old'),
    ('dsr_fmri', 'midnight'),
    ('dsr_old',  'midnight'),
    ('loc_fmri', 'dsr_fmri'),
    ('loc_old',  'dsr_old'),
    ('dsr_fmri', 'phase'),
    ('dsr_old',  'phase'),
    ('dsr_old',  'state'),
    ('dsr_fmri', 'state'),
]
for a, b in pairs_to_compare:
    line = (f"{a:>15s} vs {b:<15s} "
            f"{_corr(rdm_across_vec[a], rdm_across_vec[b]):>8.3f} "
            f"{_corr(rdm_across_vec[a], rdm_across_vec[b], mask_within_phase):>9.3f} "
            f"{_corr(rdm_across_vec[a], rdm_across_vec[b], mask_across_phase):>9.3f}")
    print(line)
    lines.append(line)

with open(os.path.join(OUT_DIR, 'correlations.txt'), 'w') as f:
    f.write('\n'.join(lines))


# ── Figure 1: paths on 3x3 grid ────────────────────────────────────────
COORDS = {i + 1: (i % 3, 2 - (i // 3)) for i in range(N_LOCATIONS)}
STATE_COLORS = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple']

fig1, axes1 = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, (name, cfg) in zip(axes1, configs.items()):
    for loc, (x, y) in COORDS.items():
        ax.add_patch(plt.Circle((x, y), 0.28, facecolor='whitesmoke',
                                edgecolor='black', lw=1))
        ax.text(x, y, str(loc), ha='center', va='center', fontsize=10)
    for state_i, sub in enumerate(cfg['subpaths']):
        xs = [COORDS[l][0] for l in sub]
        ys = [COORDS[l][1] for l in sub]
        ax.plot(xs, ys, '-o', color=STATE_COLORS[state_i], lw=2, alpha=0.7,
                markersize=4,
                label=f'state {chr(65 + state_i)}, rew={cfg["rewards"][state_i]}')
        rx, ry = COORDS[cfg['rewards'][state_i]]
        ax.scatter(rx, ry, color=STATE_COLORS[state_i], s=160, zorder=5,
                   edgecolor='black')
    ax.set_xlim(-0.7, 2.7)
    ax.set_ylim(-0.7, 2.7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(name, fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
fig1.suptitle('Paths on the 3×3 grid (one panel per config)', fontsize=11)
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, '1_paths.png'),
             dpi=150, bbox_inches='tight')
plt.show()


# ── Figure 2: walked-360 trajectories ──────────────────────────────────
fig2, axes2 = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
for ax, (name, w) in zip(axes2, walked.items()):
    ax.plot(np.arange(N_RAW_BINS), w, color='black', lw=1)
    for s in range(1, N_STATES):
        ax.axvline(s * BINS_PER_STATE, color='red', lw=0.7, ls='--',
                   alpha=0.7)
    for c in range(1, N_CONDS_PER_CONFIG):
        ax.axvline(c * BINLEN_PER_COND, color='gray', lw=0.4, alpha=0.4)
    ax.set_ylabel('location')
    ax.set_yticks(range(1, N_LOCATIONS + 1))
    ax.set_title(name, fontsize=10)
axes2[-1].set_xlabel('raw bin (0–360); red = state boundary, '
                     'grey = condition boundary')
fig2.suptitle('Walked trajectories (raw 360 bins)', fontsize=11)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, '2_walked_trajectories.png'),
             dpi=150, bbox_inches='tight')
plt.show()


# ── Figure 3: per-model activation matrices, per config ─────────────────
# Shows what the GLM design rows actually look like for each model.
models_for_act = ['loc_fmri', 'dsr_fmri', 'loc_old', 'midnight', 'dsr_old']
fig3, axes3 = plt.subplots(
    len(configs), len(models_for_act),
    figsize=(2.7 * len(models_for_act), 3.0 * len(configs)))
for r, (name, _) in enumerate(configs.items()):
    for c, m in enumerate(models_for_act):
        ax = axes3[r, c]
        M = per_model_per_config[m][name]
        im = ax.imshow(M, aspect='auto',
                       cmap='Greys' if m in hamming_models else 'viridis')
        ax.set_title(f'{m}\nshape={M.shape}', fontsize=8)
        ax.set_xlabel('feature')
        if c == 0:
            ax.set_ylabel(f'{name}\ncondition (0..11)', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
fig3.suptitle('Per-condition model activations '
              '(Hamming-route in greys, cosine-route in viridis)', fontsize=11)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, '3_model_activations.png'),
             dpi=150, bbox_inches='tight')
plt.show()


# ── Figure 4: between-config RDMs at full / within-phase / across-phase ─
def _square_from_across(vec, mask=None):
    """Inflate a between-config RDM vector to a 24x24 square (NaN on the
    diagonal blocks and on masked-out cells)."""
    n = n_total
    M = np.full((n, n), np.nan)
    triu_between_idx = np.where(between_cfg)[0]
    vec = np.asarray(vec, dtype=float).copy()
    if mask is not None:
        vec[~mask] = np.nan
    for k, idx in enumerate(triu_between_idx):
        i, j = ii[idx], jj[idx]
        M[i, j] = vec[k]
        M[j, i] = vec[k]
    return M


CMAP_SHARED = 'RdBu_r'   # blue = similar (low), red = dissimilar (high)
VMIN_SHARED, VMAX_SHARED = 0.0, 1.0
# All RDMs are now plotted on the SAME colour scale [0, 1] with the same
# diverging colormap. Convention: 0 = identical, 1 = "maximally dissimilar"
# in the natural semantics of each metric (Hamming: no shared positions;
# cosine: orthogonal). Cosine values that drift above 1 (genuinely anti-
# correlated after demeaning) saturate at darkest red — that's *more*
# dissimilar than orthogonal and visually flags itself.

for variant, mask, label in [
    ('full',         None,              'full (all between-config pairs)'),
    ('within_phase', mask_within_phase, 'within-phase pairs only'),
    ('across_phase', mask_across_phase, 'across-phase pairs only'),
]:
    fig4, axes4 = plt.subplots(
        1, len(models_for_act), figsize=(3.4 * len(models_for_act), 3.7))
    for ax, m in zip(axes4, models_for_act):
        Msq = _square_from_across(rdm_across_vec[m], mask=mask)
        im = ax.imshow(Msq, cmap=CMAP_SHARED, aspect='equal',
                       vmin=VMIN_SHARED, vmax=VMAX_SHARED)
        ax.axvline(N_CONDS_PER_CONFIG - 0.5, color='black', lw=0.8)
        ax.axhline(N_CONDS_PER_CONFIG - 0.5, color='black', lw=0.8)
        for c in range(N_PHASES, n_total, N_PHASES):
            ax.axvline(c - 0.5, color='gray', lw=0.3, alpha=0.5)
            ax.axhline(c - 0.5, color='gray', lw=0.3, alpha=0.5)
        metric = 'Hamming' if m in hamming_models else 'cosine'
        ax.set_title(f'{m}\n({metric})', fontsize=10)
        ax.set_xlabel('condition')
        ax.set_ylabel('condition')
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig4.suptitle(f'Between-config RDMs — {label}  '
                  f'(shared scale: blue=similar, red=dissimilar, '
                  f'vmin=0, vmax=1)', fontsize=11)
    fig4.tight_layout()
    fig4.savefig(os.path.join(OUT_DIR, f'4_rdms_{variant}.png'),
                 dpi=150, bbox_inches='tight')
    plt.show()


# ── Figure 5: scatter — per-cell agreement between Hamming and cosine ──
scatter_pairs = [
    ('dsr_fmri', 'dsr_old'),
    ('loc_fmri', 'loc_old'),
    ('dsr_fmri', 'midnight'),
    ('dsr_fmri', 'phase'),
]
fig5, axes5 = plt.subplots(1, len(scatter_pairs),
                           figsize=(4.0 * len(scatter_pairs), 4))
if len(scatter_pairs) == 1:
    axes5 = [axes5]
for ax, (a, b) in zip(axes5, scatter_pairs):
    xs = rdm_across_vec[a]
    ys = rdm_across_vec[b]
    # Shared axes [0, max(1.1, max_observed)] — same scale on x and y so the
    # 1:1 line is meaningful.
    lim = float(max(1.05, np.nanmax(np.concatenate([xs, ys]))))
    ax.plot([0, lim], [0, lim], color='gray', lw=0.8, ls='--',
            label='y = x (perfect agreement)')
    ax.scatter(xs[mask_within_phase], ys[mask_within_phase],
               c='tab:blue', alpha=0.7, s=36, edgecolor='black',
               linewidth=0.4, label='within-phase')
    ax.scatter(xs[mask_across_phase], ys[mask_across_phase],
               c='tab:orange', alpha=0.7, s=36, edgecolor='black',
               linewidth=0.4, label='across-phase')
    r_all = _corr(xs, ys)
    r_wp  = _corr(xs, ys, mask_within_phase)
    r_ap  = _corr(xs, ys, mask_across_phase)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.set_xlabel(f'{a}  (RDM cell value)')
    ax.set_ylabel(f'{b}  (RDM cell value)')
    ax.set_title(f'{a} vs {b}\n'
                 f'r_full={r_all:.2f}  r_wp={r_wp:.2f}  r_ap={r_ap:.2f}',
                 fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)
fig5.suptitle('Per-cell agreement between Hamming-route and cosine-route '
              'model RDMs (shared axes, dashed line = perfect agreement)',
              fontsize=11)
fig5.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, '5_scatter_agreement.png'),
             dpi=150, bbox_inches='tight')
plt.show()


# ── Figure 6: divergence heatmap (dsr_fmri − dsr_old) ──────────────────
# Where is dsr_fmri MORE dissimilar than dsr_old? Where LESS? Diverging
# scale centred at zero; both inputs are on a comparable [0, ~1.1] scale.
diff_vec = rdm_across_vec['dsr_fmri'] - rdm_across_vec['dsr_old']
diff_sq  = _square_from_across(diff_vec)
abs_max  = float(np.nanmax(np.abs(diff_vec)))

fig6, axes6 = plt.subplots(1, 3, figsize=(13, 4))
for ax, (Vec_or_Sq, title, vmin, vmax, cmap) in zip(axes6, [
    (_square_from_across(rdm_across_vec['dsr_fmri']),
     'dsr_fmri (Hamming)', 0, 1.0, 'RdBu_r'),
    (_square_from_across(rdm_across_vec['dsr_old']),
     'dsr_old (cosine)',   0, 1.0, 'RdBu_r'),
    (diff_sq, 'dsr_fmri − dsr_old\n(red: fmri MORE dissimilar)',
     -abs_max, abs_max, 'PuOr_r'),
]):
    im = ax.imshow(Vec_or_Sq, cmap=cmap, aspect='equal',
                   vmin=vmin, vmax=vmax)
    ax.axvline(N_CONDS_PER_CONFIG - 0.5, color='black', lw=0.8)
    ax.axhline(N_CONDS_PER_CONFIG - 0.5, color='black', lw=0.8)
    for c in range(N_PHASES, n_total, N_PHASES):
        ax.axvline(c - 0.5, color='gray', lw=0.3, alpha=0.5)
        ax.axhline(c - 0.5, color='gray', lw=0.3, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('condition'); ax.set_ylabel('condition')
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
fig6.suptitle('Side-by-side: dsr_fmri | dsr_old | their difference',
              fontsize=11)
fig6.tight_layout()
fig6.savefig(os.path.join(OUT_DIR, '6_divergence_heatmap.png'),
             dpi=150, bbox_inches='tight')
plt.show()


# ── Figure 7: per-cell decomposition for the most-divergent within-phase pairs ─
# Pick the cell pairs where |dsr_fmri - dsr_old| is largest among the
# within-phase between-config cells, and show:
#   (a) the two rolled 144-element trajectories side-by-side with matching
#       timesteps marked → shows what Hamming counted.
#   (b) the two dsr_old activation vectors side-by-side with shared "on"
#       features marked → shows what cosine counted.
#
# Most-divergent within-phase pairs are the cells where the two metrics
# disagree the most. Looking at the raw features behind those cells is the
# clearest way to see WHY they disagree.

def _cond_label(c):
    cfg_i = c // N_CONDS_PER_CONFIG
    local = c % N_CONDS_PER_CONFIG
    state_i = local // N_PHASES
    phase_i = local % N_PHASES
    cfg_name = list(configs)[cfg_i]
    return (f'cfg{cfg_i+1} ({cfg_name}) cond{local} '
            f'state {chr(65 + state_i)} phase {phase_i}')


def _cosine_for_pair(ci, cj):
    """Look up the dsr_old cosine cell value for a between-config pair."""
    target = (min(ci, cj), max(ci, cj))
    triu_lookup = next(
        (k for k, idx in enumerate(np.where(between_cfg)[0])
         if (ii[idx], jj[idx]) == target),
        None)
    return (float(rdm_across_vec['dsr_old'][triu_lookup])
            if triu_lookup is not None else float('nan'))


def _hamming_for_pair(ci, cj):
    target = (min(ci, cj), max(ci, cj))
    triu_lookup = next(
        (k for k, idx in enumerate(np.where(between_cfg)[0])
         if (ii[idx], jj[idx]) == target),
        None)
    return (float(rdm_across_vec['dsr_fmri'][triu_lookup])
            if triu_lookup is not None else float('nan'))


# ── Build the table of within-phase pairs ranked by divergence ─────────
# Two rankings:
#   A) most divergent overall (|dsr_fmri − dsr_old| largest)  -- fig 7
#   B) Hamming-similar but cosine-dissimilar (dsr_old − dsr_fmri largest) -- fig 8
# Save both to a single text file so the pair identities are easy to read off.

within_phase_triu_idx = np.where(between_cfg)[0][mask_within_phase]
diff_within = diff_vec[mask_within_phase]
n_pairs = len(diff_within)

# Long table of every within-phase pair, sorted by signed difference, so the
# user can read off which exact (cfg, state, phase, location) pairs sit where
# in the scatter cloud.
all_within_pairs = []
for k in range(n_pairs):
    ci = ii[within_phase_triu_idx[k]]
    cj = jj[within_phase_triu_idx[k]]
    hd = _hamming_for_pair(ci, cj)
    cs = _cosine_for_pair(ci, cj)
    all_within_pairs.append({
        'ci': int(ci), 'cj': int(cj),
        'label_i': _cond_label(ci),
        'label_j': _cond_label(cj),
        'dsr_fmri': hd,
        'dsr_old':  cs,
        'fmri_minus_old':  hd - cs,
        'old_minus_fmri':  cs - hd,
        'abs_diff':        abs(hd - cs),
    })

# Sort + write a single text file.
table_lines = []
def _format_pair_block(title, pairs, sort_key, top=5):
    pairs_sorted = sorted(pairs, key=lambda r: r[sort_key], reverse=True)[:top]
    lines = [title, '=' * len(title)]
    header = (f"{'rank':>4s}  {'dsr_fmri':>8s}  {'dsr_old':>8s}  "
              f"{'diff':>8s}  {'pair (cond_i  vs  cond_j)'}")
    lines.append(header)
    for rank, r in enumerate(pairs_sorted, 1):
        lines.append(
            f"{rank:>4d}  {r['dsr_fmri']:>8.3f}  {r['dsr_old']:>8.3f}  "
            f"{(r['dsr_fmri']-r['dsr_old']):>+8.3f}  "
            f"{r['label_i']}  vs  {r['label_j']}")
    lines.append('')
    return lines

table_lines += _format_pair_block(
    "TOP-5 |dsr_fmri − dsr_old|  (most divergent within-phase pairs, either direction)",
    all_within_pairs, sort_key='abs_diff', top=5)
table_lines += _format_pair_block(
    "TOP-5 dsr_fmri − dsr_old   (Hamming says DISSIMILAR, cosine says SIMILAR  — the bottom-right outliers)",
    all_within_pairs, sort_key='fmri_minus_old', top=5)
table_lines += _format_pair_block(
    "TOP-5 dsr_old − dsr_fmri   (Hamming says SIMILAR, cosine says DISSIMILAR  — the opposite direction)",
    all_within_pairs, sort_key='old_minus_fmri', top=5)

# Also append a full sorted listing of every within-phase pair, low-to-high
# in fmri_minus_old, so you can read off the entire cloud.
table_lines += [
    "FULL within-phase pair table  (sorted by dsr_fmri − dsr_old, low → high)",
    "=" * 78,
    f"{'dsr_fmri':>8s}  {'dsr_old':>8s}  {'diff':>8s}  pair",
]
for r in sorted(all_within_pairs, key=lambda r: r['fmri_minus_old']):
    table_lines.append(
        f"{r['dsr_fmri']:>8.3f}  {r['dsr_old']:>8.3f}  "
        f"{(r['dsr_fmri']-r['dsr_old']):>+8.3f}  "
        f"{r['label_i']}  vs  {r['label_j']}")

table_text = '\n'.join(table_lines)
print('\n' + table_text)
with open(os.path.join(OUT_DIR, 'within_phase_outlier_table.txt'), 'w') as f:
    f.write(table_text)


# ── Per-pair decomposition figure (reused for fig 7 and fig 8) ─────────
def _draw_decomposition(pairs, suptitle, save_path):
    fig, axes = plt.subplots(len(pairs), 2,
                             figsize=(13, 3.5 * len(pairs)), squeeze=False)
    for r, (ci, cj, label_extra) in enumerate(pairs):
        # (a) trajectories side-by-side
        ax = axes[r, 0]
        traj_i = model_stack['dsr_fmri'][ci].astype(int)
        traj_j = model_stack['dsr_fmri'][cj].astype(int)
        match = traj_i == traj_j
        ax.plot(traj_i, '-', color='tab:blue',
                label=f'{_cond_label(ci)}', lw=1.4)
        ax.plot(traj_j, '-', color='tab:orange',
                label=f'{_cond_label(cj)}', lw=1.4)
        for t in np.where(match)[0]:
            ax.axvspan(t - 0.4, t + 0.4, alpha=0.18, color='tab:green', lw=0)
        hd = float((traj_i != traj_j).mean())
        ax.set_title(
            f'(a) Rolled trajectories: Hamming={hd:.3f}, '
            f'matches/144={int(match.sum())}     {label_extra}\n'
            f'green bands = positions Hamming counts as a match',
            fontsize=9)
        ax.set_xlabel('rolled timestep (0..143)')
        ax.set_ylabel('location ID')
        ax.set_yticks(range(1, N_LOCATIONS + 1))
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)

        # (b) dsr_old activations side-by-side
        ax = axes[r, 1]
        act_i = model_stack['dsr_old'][ci]
        act_j = model_stack['dsr_old'][cj]
        n_feat = act_i.size
        ax.plot(act_i, color='tab:blue', lw=1.0,
                label=f'{_cond_label(ci)}')
        ax.plot(act_j + 0.02, color='tab:orange', lw=1.0,
                label=f'{_cond_label(cj)}')
        on_both = (act_i > 1e-6) & (act_j > 1e-6)
        if on_both.any():
            ax.scatter(np.where(on_both)[0], act_i[on_both],
                       marker='o', s=30, color='tab:green', zorder=4,
                       label=f'both active (n={int(on_both.sum())})')
        cs = _cosine_for_pair(ci, cj)
        ax.set_title(
            f'(b) dsr_old "clock-ring" features ({n_feat} dims): '
            f'cosine dist={cs:.3f}\n'
            f'green dots = both active → contribute to cosine similarity',
            fontsize=9)
        ax.set_xlabel('feature index (clock-ring neuron)')
        ax.set_ylabel('activation')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ── Figure 7: bottom-right outliers (Hamming high, cosine low) ─────────
top_fmri_high = sorted(all_within_pairs,
                       key=lambda r: r['fmri_minus_old'],
                       reverse=True)[:3]
_draw_decomposition(
    [(p['ci'], p['cj'],
      f"(fmri={p['dsr_fmri']:.2f}, old={p['dsr_old']:.2f})")
     for p in top_fmri_high],
    suptitle=('Top-3 within-phase pairs where Hamming says DISSIMILAR '
              'but cosine says SIMILAR\n'
              '(= the bottom-right outliers in fig 5)'),
    save_path=os.path.join(OUT_DIR, '7_outliers_fmri-high_cosine-low.png'))


# ── Figure 8: opposite-direction outliers (Hamming low, cosine high) ───
top_old_high = sorted(all_within_pairs,
                      key=lambda r: r['old_minus_fmri'],
                      reverse=True)[:3]
_draw_decomposition(
    [(p['ci'], p['cj'],
      f"(fmri={p['dsr_fmri']:.2f}, old={p['dsr_old']:.2f})")
     for p in top_old_high],
    suptitle=('Top-3 within-phase pairs where Hamming says SIMILAR '
              'but cosine says DISSIMILAR\n'
              '(= the opposite direction — top-left of fig 5)'),
    save_path=os.path.join(OUT_DIR, '8_outliers_fmri-low_cosine-high.png'))


# ─────────────────────────────────────────────────────────────────────────
# Phase encoding comparison: von Mises vs categorical
#
# Concept check: phase should not depend on which encoding we use. We build
# a categorical phase matrix (3 rows × 360 timesteps, sharp blocks of 30
# bins each = "early/middle/late" of each state's 90-bin window) and
# compare its RDM against the von Mises `phase` model. If they match, the
# phase encoding isn't the source of any divergence further downstream.
# ─────────────────────────────────────────────────────────────────────────

def make_categorical_phase(n_phases, step_per_state=BINS_PER_STATE,
                           n_states=N_STATES):
    """Sharp 3-row × 360 categorical phase: row p is 1 within the p-th third
    of every state's 90-bin window, 0 elsewhere."""
    bpp = step_per_state // n_phases
    mat = np.zeros((n_phases, step_per_state * n_states), dtype=float)
    for s in range(n_states):
        for p in range(n_phases):
            mat[p,
                s * step_per_state + p * bpp:
                s * step_per_state + (p + 1) * bpp] = 1
    return mat


def make_categorical_midnight(walked, phase_cat,
                              n_phases=N_PHASES, n_loc=N_LOCATIONS):
    """midnight[L * n_phases + p, t] = 1 when location[t] == L AND phase_cat[p, t] == 1."""
    n_t = len(walked)
    mat = np.zeros((n_loc * n_phases, n_t), dtype=float)
    for t in range(n_t):
        loc = int(walked[t]) - 1
        for p in range(n_phases):
            if phase_cat[p, t]:
                mat[loc * n_phases + p, t] = 1
    return mat


def make_categorical_state_phase(phase_cat, n_phases=N_PHASES,
                                 n_states=N_STATES,
                                 step_per_state=BINS_PER_STATE):
    """4 states × 3 phases × 360 timesteps. Row (s*n_phases + p) is 1 when
    in state s AND phase_cat says phase p."""
    n_t = phase_cat.shape[1]
    mat = np.zeros((n_states * n_phases, n_t), dtype=float)
    for s in range(n_states):
        for p in range(n_phases):
            t_start = s * step_per_state
            t_end = (s + 1) * step_per_state
            mat[s * n_phases + p, t_start:t_end] = phase_cat[p, t_start:t_end]
    return mat


def make_categorical_clo_model(walked, phase_cat, midn_cat,
                               n_phases=N_PHASES, n_loc=N_LOCATIONS,
                               n_states=N_STATES,
                               step_per_state=BINS_PER_STATE):
    """A from-scratch clock-ring with categorical phase. Mirrors the
    construction in mc.simulation.predictions.model_DSR (`clo_model`):

      for each midnight neuron M = (L, p):
          for each TIMESTEP t where M is active:
              roll the full state-phase ring (12 rows × 360) by t
              add to clo_model rows (M*12 : M*12+12)

    The only difference from the original is that phase is categorical
    instead of von Mises. If this matches the original clo_model's RDM,
    the von Mises encoding is not the source of divergence. If it
    matches dsr_fmri's RDM, the clock-ring construction faithfully
    represents the rolled trajectory and the divergence is purely the
    von Mises tuning."""
    n_t = len(walked)
    state_phase_ring = make_categorical_state_phase(phase_cat)  # (12, 360)
    n_midn = n_loc * n_phases
    n_sp = n_states * n_phases
    clo = np.zeros((n_midn * n_sp, n_t), dtype=float)
    for L_p in range(n_midn):
        peak_times = np.where(midn_cat[L_p] > 0)[0]
        for t0 in peak_times:
            shifted = np.roll(state_phase_ring, t0, axis=1)
            clo[L_p * n_sp:(L_p + 1) * n_sp] += shifted
    return clo


phase_cat_matrix = make_categorical_phase(N_PHASES)  # (3, 360); same for both configs

# Per-config: midnight_cat, dsr_clock_cat, plus "flat-phase midnight" = location.
flat_phase_matrix = np.ones((1, N_RAW_BINS), dtype=float)
for name, w in walked.items():
    midn_cat = make_categorical_midnight(w, phase_cat_matrix)
    clo_cat  = make_categorical_clo_model(w, phase_cat_matrix, midn_cat)
    # "midnight with flat phase" reduces to plain location
    midn_flat = make_categorical_midnight(w, flat_phase_matrix,
                                          n_phases=1)
    def _ds(M):
        return M.reshape(M.shape[0], N_CONDS_PER_CONFIG,
                         BINLEN_PER_COND).mean(axis=2).T

    per_model_per_config.setdefault('phase_categorical', {})[name] = \
        _ds(phase_cat_matrix)
    per_model_per_config.setdefault('midnight_categorical', {})[name] = \
        _ds(midn_cat)
    per_model_per_config.setdefault('dsr_old_categorical', {})[name] = \
        _ds(clo_cat)
    per_model_per_config.setdefault('midnight_flat_phase', {})[name] = \
        _ds(midn_flat)

for m in ('phase_categorical', 'midnight_categorical',
          'dsr_old_categorical', 'midnight_flat_phase'):
    model_stack[m] = np.vstack(
        [per_model_per_config[m][name] for name in configs])
    _w, _across, _full = my_RSA.compute_crosscorr_within(
        model_stack[m], plotting=False, include_diagonal=False,
        model=m, no_tasks=2, block_size=N_CONDS_PER_CONFIG)
    rdm_across_vec[m] = np.asarray(_across[0], dtype=float)


print("\n=== Phase-encoding comparison (von Mises vs categorical) ===")
print(f"  phase (von Mises) vs phase_categorical:           "
      f"r = {_corr(rdm_across_vec['phase'], rdm_across_vec['phase_categorical']):.3f}")
print(f"  midnight (vM, N=3) vs midnight_categorical:       "
      f"r = {_corr(rdm_across_vec['midnight'], rdm_across_vec['midnight_categorical']):.3f}")
print(f"  dsr_old (vM, N=3) vs dsr_old_categorical:         "
      f"r = {_corr(rdm_across_vec['dsr_old'], rdm_across_vec['dsr_old_categorical']):.3f}")
print(f"  midnight_flat_phase vs loc_old (should be ~1.00): "
      f"r = {_corr(rdm_across_vec['midnight_flat_phase'], rdm_across_vec['loc_old']):.3f}")


# ─────────────────────────────────────────────────────────────────────────
# Targeted test: where does the divergence come from?
#
# We construct a small family of variants that isolate one source at a time:
#
#   dsr_fmri          Hamming on rolled location IDs (reference; what we want
#                     to match)
#   dsr_fmri_cosine   Same data, just one-hot encoded and compared with
#                     cosine. By construction this equals (1 − Hamming
#                     similarity) up to normalisation — sanity check that
#                     the metric change alone doesn't cause divergence.
#   dsr_old_N1        clo_model from model_DSR with no_phase_neurons=1.
#                     Strips the phase tuning, so each midnight visit
#                     produces ONE peak instead of three. Tests whether the
#                     divergence is driven by phase-tuning complexity.
#   midnight_N1       midn_model from model_DSR with no_phase_neurons=1
#                     (just location × phase). No clock-ring rolling. Tests
#                     what we lose when we strip the lag prediction.
#
# If dsr_fmri ≈ dsr_fmri_cosine ≈ midnight_N1 but dsr_old_N1 ≠ them, the
# divergence comes from the CLOCK-RING ROLLING (the smearing). If
# dsr_old_N1 ≈ them but dsr_old (N=3) ≠ them, the divergence comes from
# the PHASE TUNING.
# ─────────────────────────────────────────────────────────────────────────

print("\n=== Targeted test: where does the divergence come from? ===")

# (1) dsr_fmri_cosine: one-hot encode the rolled location IDs, then cosine.
def _one_hot_rows(loc_id_matrix, n_locations=N_LOCATIONS):
    """(N_rows, T) integer locations -> (N_rows, T * n_locations) one-hot."""
    M = np.asarray(loc_id_matrix, dtype=int)
    n_rows, T = M.shape
    out = np.zeros((n_rows, T * n_locations), dtype=float)
    for r in range(n_rows):
        for t in range(T):
            loc = M[r, t]
            # location IDs are 1..9; clip and map to 0..8
            if 1 <= loc <= n_locations:
                out[r, t * n_locations + (loc - 1)] = 1.0
    return out

dsr_fmri_onehot_per_config = {
    name: _one_hot_rows(per_model_per_config['dsr_fmri'][name].astype(int))
    for name in configs
}
model_stack['dsr_fmri_cosine'] = np.vstack(
    [dsr_fmri_onehot_per_config[name] for name in configs])


# (2 + 3) Re-run model_DSR with N_PHASES=1.
per_model_per_config_N1 = {'dsr_old_N1': {}, 'midnight_N1': {}}
for name, w in walked.items():
    walked_0idx = (w - 1).tolist()
    loc_og_1, phase_og_1, state_og_1, midn_1, dsr_1, _, _ = (
        predictions.model_DSR(locations=walked_0idx, no_phase_neurons=1))

    def _ds(M):
        return M.reshape(M.shape[0], N_CONDS_PER_CONFIG, BINLEN_PER_COND
                         ).mean(axis=2).T

    per_model_per_config_N1['dsr_old_N1'][name]  = _ds(dsr_1)
    per_model_per_config_N1['midnight_N1'][name] = _ds(midn_1)

for m in per_model_per_config_N1:
    model_stack[m] = np.vstack(
        [per_model_per_config_N1[m][name] for name in configs])


# Compute RDMs for the four variants using the cosine-route helper.
variant_models = ('dsr_fmri_cosine', 'dsr_old_N1', 'midnight_N1')
for m in variant_models:
    _w, _across, _full = my_RSA.compute_crosscorr_within(
        model_stack[m], plotting=False, include_diagonal=False,
        model=m, no_tasks=2, block_size=N_CONDS_PER_CONFIG)
    rdm_across_vec[m] = np.asarray(_across[0], dtype=float)


# Cross-correlation table among the DSR-family RDMs.
five = ['dsr_fmri', 'dsr_fmri_cosine', 'midnight_flat_phase',
        'midnight_N1', 'midnight_categorical',
        'dsr_old_N1', 'dsr_old_categorical', 'dsr_old']
print(f"\nCorrelations among the DSR-family RDMs "
      f"(between-config vector; columns/rows in same order):")
header = "          " + "  ".join(f'{m:>16s}' for m in five)
print(header)
table_lines2 = [header]
for a in five:
    row_vals = ['{:>10s}'.format(a)]
    for b in five:
        row_vals.append('{:>18.3f}'.format(
            _corr(rdm_across_vec[a], rdm_across_vec[b])))
    line = '  '.join(row_vals)
    print(line)
    table_lines2.append(line)
with open(os.path.join(OUT_DIR, 'dsr_family_correlations.txt'), 'w') as f:
    f.write('\n'.join(table_lines2))


# ── Figure 9: side-by-side RDMs of the five DSR-family variants ─────────
fig9, axes9 = plt.subplots(1, len(five), figsize=(3.4 * len(five), 3.7))
for ax, m in zip(axes9, five):
    Msq = _square_from_across(rdm_across_vec[m])
    metric = 'Hamming' if m == 'dsr_fmri' else 'cosine'
    im = ax.imshow(Msq, cmap=CMAP_SHARED, aspect='equal',
                   vmin=VMIN_SHARED, vmax=VMAX_SHARED)
    ax.axvline(N_CONDS_PER_CONFIG - 0.5, color='black', lw=0.8)
    ax.axhline(N_CONDS_PER_CONFIG - 0.5, color='black', lw=0.8)
    for c in range(N_PHASES, n_total, N_PHASES):
        ax.axvline(c - 0.5, color='gray', lw=0.3, alpha=0.5)
        ax.axhline(c - 0.5, color='gray', lw=0.3, alpha=0.5)
    ax.set_title(f'{m}\n({metric})', fontsize=10)
    ax.set_xlabel('condition')
    ax.set_ylabel('condition')
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
fig9.suptitle('DSR-family variants — reading the divergence chain: '
              'dsr_fmri → dsr_fmri_cosine → midnight_N1 → dsr_old_N1 → dsr_old',
              fontsize=11)
fig9.tight_layout()
fig9.savefig(os.path.join(OUT_DIR, '9_dsr_family_variants.png'),
             dpi=150, bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────────────────────────────────
# IDEAL clock-ring: the representation the user EXPECTS dsr_old to be.
#
# For each cond i, build a (9 locations × N_CONDS_PER_CONFIG lags) one-hot
# grid where cell (L, k) = 1 iff the mode location at lag k from cond i's
# start is L. Flatten to a 108-dim vector per cond. By construction this
# is a clean "what location at what future lag" representation: exactly
# the "30% similarity" semantics from your example.
#
# Prediction: cosine RDM of dsr_ideal should match Hamming RDM of dsr_fmri
# (after collapsing each lag's 12 sub-timesteps to a single mode).
# ─────────────────────────────────────────────────────────────────────────

def build_ideal_dsr(walked, n_conds=N_CONDS_PER_CONFIG,
                    len_per_lag=LEN_STANDARDISED_PATH, n_loc=N_LOCATIONS):
    """Per-cond (n_loc × n_conds) one-hot grid: row L, col k = 1 iff the
    mode location at lag k from cond start is L. Flattens to (n_loc * n_conds,)."""
    base = downsample_mode(walked, target_len=n_conds * len_per_lag).astype(int)
    out = np.zeros((n_conds, n_loc, n_conds), dtype=float)
    for cond in range(n_conds):
        rolled = np.roll(base, -cond * len_per_lag)
        for lag in range(n_conds):
            sub = rolled[lag * len_per_lag:(lag + 1) * len_per_lag]
            mode_loc = Counter(sub.tolist()).most_common(1)[0][0]
            out[cond, mode_loc - 1, lag] = 1
    return out.reshape(n_conds, n_loc * n_conds)  # (12, 108)


for name, w in walked.items():
    per_model_per_config.setdefault('dsr_ideal', {})[name] = build_ideal_dsr(w)
    
# import pdb; pdb.set_trace()
model_stack['dsr_ideal'] = np.vstack(
    [per_model_per_config['dsr_ideal'][name] for name in configs])
_w, _across, _full = my_RSA.compute_crosscorr_within(
    model_stack['dsr_ideal'], plotting=False, include_diagonal=False,
    model='dsr_ideal', no_tasks=2, block_size=N_CONDS_PER_CONFIG)
rdm_across_vec['dsr_ideal'] = np.asarray(_across[0], dtype=float)


# ── Figure 10: conceptual visualisation — "what dsr_old SHOULD look like" ─
# Pick one example condition (cfg 1, cond 0 = state A, phase 0) and show:
#   (A) the rolled trajectory underlying dsr_fmri (144 ints)
#   (B) the IDEAL 9 × 12 grid: "location at lag k from cond start"
#   (C) the ACTUAL clo_model activation reshaped to (27 midnight × 12 ring),
#       i.e. dsr_old's 324-dim vector for the same condition
#   (D) the three RDMs side-by-side: dsr_fmri, dsr_ideal, dsr_old.
#
# (A) and (B) encode the SAME information just in two layouts.  By
# construction the cosine RDM of (B) equals the Hamming RDM of (A) up to
# small mode-collapse rounding.  (C) is what the implementation actually
# produces — same target neuron count (~324) but a fundamentally different
# layout that smears each visit across all 360 timesteps.

example_cond = 0  # cfg 1, state A, phase 0
example_name = list(configs)[0]
ex_walked = walked[example_name]

# Panel A
ex_traj = model_stack['dsr_fmri'][example_cond].astype(int)

# Panel B (re-derive directly so the figure is self-contained)
ex_ideal_grid = build_ideal_dsr(ex_walked)[example_cond].reshape(
    N_LOCATIONS, N_CONDS_PER_CONFIG)

# Panel C
ex_old = model_stack['dsr_old'][example_cond].reshape(
    N_LOCATIONS * N_PHASES, N_STATES * N_PHASES)

fig10 = plt.figure(figsize=(15, 9))

ax_a = fig10.add_subplot(2, 3, 1)
ax_a.plot(ex_traj, color='black', lw=1)
ax_a.set_yticks(range(1, N_LOCATIONS + 1))
ax_a.set_xlabel('rolled timestep (0..143 = 12 lags × 12 sub-timesteps)')
ax_a.set_ylabel('location ID')
ax_a.set_title(
    f'(A)  dsr_fmri row for {_cond_label(example_cond)}\n'
    f'144 integer location IDs along the rolled trajectory',
    fontsize=9)
for k in range(1, N_CONDS_PER_CONFIG):
    ax_a.axvline(k * LEN_STANDARDISED_PATH - 0.5, color='gray',
                 lw=0.4, alpha=0.6)
ax_a.grid(alpha=0.3)

ax_b = fig10.add_subplot(2, 3, 2)
ax_b.imshow(ex_ideal_grid, cmap='Greys', aspect='auto')
ax_b.set_xticks(range(N_CONDS_PER_CONFIG))
ax_b.set_xticklabels([f'lag{k}' for k in range(N_CONDS_PER_CONFIG)],
                     fontsize=7, rotation=45)
ax_b.set_yticks(range(N_LOCATIONS))
ax_b.set_yticklabels([f'loc {L+1}' for L in range(N_LOCATIONS)], fontsize=7)
ax_b.set_title('(B)  IDEAL dsr_old: 9 locations × 12 lags one-hot grid\n'
               '(what you described — same info as A, different layout)',
               fontsize=9)
# annotate each lag with the location ID
for k in range(N_CONDS_PER_CONFIG):
    loc = int(np.argmax(ex_ideal_grid[:, k])) + 1
    ax_b.text(k, loc - 1, str(loc), ha='center', va='center',
              fontsize=8, color='white', fontweight='bold')

ax_c = fig10.add_subplot(2, 3, 3)
im_c = ax_c.imshow(ex_old, cmap='viridis', aspect='auto')
ax_c.set_xlabel('state-phase ring row (0..11 = A0,A1,A2,B0,...,D2)')
ax_c.set_ylabel('midnight neuron (0..26 = (loc, phase) pair)')
ax_c.set_title(
    '(C)  ACTUAL dsr_old: 27 midnight × 12 state-phase ring\n'
    'rolled-ring smear, not a (location × lag) grid',
    fontsize=9)
plt.colorbar(im_c, ax=ax_c, fraction=0.045, pad=0.02)

# Bottom row: the three RDMs
for col, m in enumerate(['dsr_fmri', 'dsr_ideal', 'dsr_old']):
    ax = fig10.add_subplot(2, 3, 4 + col)
    Msq = _square_from_across(rdm_across_vec[m])
    im = ax.imshow(Msq, cmap=CMAP_SHARED, aspect='equal',
                   vmin=VMIN_SHARED, vmax=VMAX_SHARED)
    ax.axvline(N_CONDS_PER_CONFIG - 0.5, color='black', lw=0.8)
    ax.axhline(N_CONDS_PER_CONFIG - 0.5, color='black', lw=0.8)
    for c in range(N_PHASES, n_total, N_PHASES):
        ax.axvline(c - 0.5, color='gray', lw=0.3, alpha=0.5)
        ax.axhline(c - 0.5, color='gray', lw=0.3, alpha=0.5)
    ax.set_title(f'(D{col+1})  {m}  '
                 f'r(vs dsr_fmri)={_corr(rdm_across_vec[m], rdm_across_vec["dsr_fmri"]):.3f}',
                 fontsize=9)
    ax.set_xlabel('condition'); ax.set_ylabel('condition')
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

fig10.suptitle(
    'What the implementation does vs. what the conceptual model assumes\n'
    '(A) = (B) at the row-content level; (C) is a different layout entirely; '
    '(D1) ≈ (D2), (D3) diverges',
    fontsize=11)
fig10.tight_layout()
fig10.savefig(os.path.join(OUT_DIR, '10_ideal_vs_actual_dsr.png'),
              dpi=150, bbox_inches='tight')
plt.show()

print(f"\n=== Ideal vs actual dsr_old (between-config RDM correlations) ===")
print(f"  dsr_fmri vs dsr_ideal:  r = "
      f"{_corr(rdm_across_vec['dsr_fmri'], rdm_across_vec['dsr_ideal']):.3f}    "
      f"(should be very high — same info, just re-shaped to a 9×12 grid)")
print(f"  dsr_ideal vs dsr_old:   r = "
      f"{_corr(rdm_across_vec['dsr_ideal'], rdm_across_vec['dsr_old']):.3f}    "
      f"(should be low — actual clo_model is NOT a 9×12 grid)")

# ─────────────────────────────────────────────────────────────────────────
# IDEAL clock-ring WITH 3 phases (graded activation).
#
# The simple `dsr_ideal` is "9 location × 12 lag" — assumes no phase
# information. The conceptual model the user actually intends is:
#
#   9 location × 3 phase × 12 lag  =  324 graded activations.
#
# For each cond i, at each future lag k, we look at the 30-bin window of
# raw cond (i+k) % 12 and count, per (location, phase) pair, what fraction
# of that window was spent there. Phase is the categorical position
# within the state ( (bin % 90) // 30 ). So briefly-visited (loc, phase)
# pairs get small activations, fully-occupied ones get 1.0, and unvisited
# get 0. This is the user's "even lightly activated dsr modules" idea.
#
# Hypothesis: this should be a much closer match to dsr_old than the
# plain dsr_ideal — they share the (location, phase) feature space; the
# only remaining difference is dsr_old's rolled-ring smearing vs. our
# clean lag indexing.
# ─────────────────────────────────────────────────────────────────────────

def build_ideal_dsr_with_phases(walked, n_conds=N_CONDS_PER_CONFIG,
                                len_per_cond_raw=BINLEN_PER_COND,
                                n_phases=N_PHASES, n_loc=N_LOCATIONS,
                                bins_per_state=BINS_PER_STATE):
    """Per-cond (n_loc × n_phases × n_conds) graded activation.

    For cond i, at lag k, the activation at (L, P, k) is the fraction of
    raw bins in cond (i+k) % n_conds's window where the agent was at
    location L AND in categorical phase P. Phase = (bin % 90) // 30.
    """
    walked = np.asarray(walked, dtype=int)
    out = np.zeros((n_conds, n_loc, n_phases, n_conds), dtype=float)
    bpp = bins_per_state // n_phases  # bins per phase = 30
    for cond in range(n_conds):
        for lag in range(n_conds):
            target_cond = (cond + lag) % n_conds
            t_start = target_cond * len_per_cond_raw
            t_end = t_start + len_per_cond_raw
            for t in range(t_start, t_end):
                loc = walked[t] - 1
                phase = (t % bins_per_state) // bpp
                out[cond, loc, phase, lag] += 1
        out[cond] /= len_per_cond_raw
    # Flatten to (n_conds, n_loc * n_phases * n_conds) = (12, 324)
    return out.reshape(n_conds, n_loc * n_phases * n_conds)


for name, w in walked.items():
    per_model_per_config.setdefault('dsr_ideal_phases', {})[name] = \
        build_ideal_dsr_with_phases(w)
model_stack['dsr_ideal_phases'] = np.vstack(
    [per_model_per_config['dsr_ideal_phases'][name] for name in configs])
_w, _across, _full = my_RSA.compute_crosscorr_within(
    model_stack['dsr_ideal_phases'], plotting=False, include_diagonal=False,
    model='dsr_ideal_phases', no_tasks=2, block_size=N_CONDS_PER_CONFIG)
rdm_across_vec['dsr_ideal_phases'] = np.asarray(_across[0], dtype=float)


print("\n=== dsr_ideal with 3 phases vs originals ===")
for a, b, note in [
    ('dsr_fmri',         'dsr_ideal',          'baseline: should be near 1'),
    ('dsr_fmri',         'dsr_ideal_phases',   'adding phases drops correlation slightly'),
    ('dsr_ideal',        'dsr_ideal_phases',   'how much phase changes the geometry'),
    ('dsr_ideal_phases', 'dsr_old',            'KEY: shared feature space, only diff is smearing'),
    ('dsr_ideal_phases', 'dsr_old_categorical','vs the simpler clo construction'),
]:
    r = _corr(rdm_across_vec[a], rdm_across_vec[b])
    print(f"  {a:<22s} vs {b:<22s}  r = {r:>+.3f}    ({note})")


# ── Figure 11: dsr_ideal_phases activation + RDM comparison ──────────
example_cond = 0
example_name = list(configs)[0]

# Activation grids (top row, all for cfg 1 cond 0)
grid_ideal = build_ideal_dsr(
    walked[example_name])[example_cond].reshape(
        N_LOCATIONS, N_CONDS_PER_CONFIG)
grid_phases = build_ideal_dsr_with_phases(
    walked[example_name])[example_cond].reshape(
        N_LOCATIONS * N_PHASES, N_CONDS_PER_CONFIG)
grid_old_raw = model_stack['dsr_old'][example_cond]
grid_old = grid_old_raw.reshape(N_LOCATIONS * N_PHASES, N_STATES * N_PHASES)

fig11 = plt.figure(figsize=(16, 9))

ax = fig11.add_subplot(2, 4, 1)
ax.imshow(grid_ideal, cmap='Greys', aspect='auto', vmin=0, vmax=1)
ax.set_title('(A) dsr_ideal\n9 loc × 12 lag (one-hot)', fontsize=9)
ax.set_xlabel('lag'); ax.set_ylabel('location (1..9)')
ax.set_yticks(range(N_LOCATIONS))
ax.set_yticklabels([str(L + 1) for L in range(N_LOCATIONS)], fontsize=7)
ax.set_xticks(range(N_CONDS_PER_CONFIG))
ax.set_xticklabels([str(k) for k in range(N_CONDS_PER_CONFIG)], fontsize=7)

ax = fig11.add_subplot(2, 4, 2)
im = ax.imshow(grid_phases, cmap='Greys', aspect='auto', vmin=0, vmax=1)
ax.set_title('(B) dsr_ideal_phases\n27 (loc, phase) × 12 lag (graded)',
             fontsize=9)
ax.set_xlabel('lag')
ax.set_ylabel('(loc, phase)  — 3 phases per location')
# y-tick every 3rd row = each location
ax.set_yticks(range(0, N_LOCATIONS * N_PHASES, N_PHASES))
ax.set_yticklabels([f'L{L+1} p0' for L in range(N_LOCATIONS)], fontsize=7)
ax.set_xticks(range(N_CONDS_PER_CONFIG))
ax.set_xticklabels([str(k) for k in range(N_CONDS_PER_CONFIG)], fontsize=7)
plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

ax = fig11.add_subplot(2, 4, 3)
im = ax.imshow(grid_old, cmap='viridis', aspect='auto')
ax.set_title('(C) actual dsr_old\n27 midnight × 12 state-phase ring',
             fontsize=9)
ax.set_xlabel('ring row (A0,A1,A2,B0,...,D2)')
ax.set_ylabel('midnight neuron (loc × phase)')
plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

ax = fig11.add_subplot(2, 4, 4)
grid_old_cat_raw = model_stack['dsr_old_categorical'][example_cond]
grid_old_cat = grid_old_cat_raw.reshape(N_LOCATIONS * N_PHASES,
                                        N_STATES * N_PHASES)
im = ax.imshow(grid_old_cat, cmap='viridis', aspect='auto')
ax.set_title('(D) dsr_old_categorical\n27 midnight × 12 state-phase ring',
             fontsize=9)
ax.set_xlabel('ring row')
ax.set_ylabel('midnight neuron')
plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

# Bottom row: RDMs for the four DSR variants on the same shared scale
for col, m in enumerate(['dsr_fmri', 'dsr_ideal', 'dsr_ideal_phases', 'dsr_old']):
    ax = fig11.add_subplot(2, 4, 5 + col)
    Msq = _square_from_across(rdm_across_vec[m])
    im = ax.imshow(Msq, cmap=CMAP_SHARED, aspect='equal',
                   vmin=VMIN_SHARED, vmax=VMAX_SHARED)
    ax.axvline(N_CONDS_PER_CONFIG - 0.5, color='black', lw=0.8)
    ax.axhline(N_CONDS_PER_CONFIG - 0.5, color='black', lw=0.8)
    for c in range(N_PHASES, n_total, N_PHASES):
        ax.axvline(c - 0.5, color='gray', lw=0.3, alpha=0.5)
        ax.axhline(c - 0.5, color='gray', lw=0.3, alpha=0.5)
    r_fmri = _corr(rdm_across_vec[m], rdm_across_vec['dsr_fmri'])
    r_old  = _corr(rdm_across_vec[m], rdm_across_vec['dsr_old'])
    ax.set_title(f'{m}\nr(fmri)={r_fmri:.2f}  r(old)={r_old:.2f}',
                 fontsize=9)
    ax.set_xlabel('condition'); ax.set_ylabel('condition')
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

fig11.suptitle('Adding 3-phase modulation to the ideal DSR — and vs the actual dsr_old',
               fontsize=11)
fig11.tight_layout()
fig11.savefig(os.path.join(OUT_DIR, '11_dsr_ideal_with_phases.png'),
              dpi=150, bbox_inches='tight')
plt.show()


# ─────────────────────────────────────────────────────────────────────────
# Phase-stratified correlations for the key DSR-family pairs
#
# Answer to "does the categorical clo_model match dsr_fmri better when we
# restrict to within-phase pairs / to across-phase pairs?"
# Splits each pair's between-config RDM cells by the phase relationship of
# the two endpoints, and reports r in each subset.
# ─────────────────────────────────────────────────────────────────────────

phase_strat_pairs = [
    # (a, b, comment)
    ('dsr_fmri', 'dsr_old',               'original (cosine vM clock-ring)'),
    ('dsr_fmri', 'dsr_old_categorical',   'simpler clo construction'),
    ('dsr_fmri', 'dsr_ideal',             'no-phase ideal (location × lag)'),
    ('dsr_fmri', 'dsr_ideal_phases',      'graded (loc × phase × lag)'),
    ('dsr_old',  'dsr_old_categorical',   'two clock-ring variants'),
    ('dsr_old',  'dsr_ideal_phases',      'clock-ring vs graded ideal'),
    ('dsr_old_categorical', 'dsr_ideal_phases',
                                           'categorical clock-ring vs graded ideal'),
]
print("\n=== Phase-stratified correlations between DSR-family RDMs ===")
header = f"  {'pair':<48s} {'full':>8s} {'within_p':>10s} {'across_p':>10s}    note"
print(header)
phase_strat_lines = [header]
for a, b, note in phase_strat_pairs:
    if a not in rdm_across_vec or b not in rdm_across_vec:
        continue
    r_full = _corr(rdm_across_vec[a], rdm_across_vec[b])
    r_wp   = _corr(rdm_across_vec[a], rdm_across_vec[b], mask_within_phase)
    r_ap   = _corr(rdm_across_vec[a], rdm_across_vec[b], mask_across_phase)
    line = (f"  {a + ' vs ' + b:<48s} {r_full:>+8.3f} {r_wp:>+10.3f} "
            f"{r_ap:>+10.3f}    ({note})")
    print(line)
    phase_strat_lines.append(line)

with open(os.path.join(OUT_DIR, 'phase_stratified_correlations.txt'), 'w') as f:
    f.write('\n'.join(phase_strat_lines))

print(f"\nDone. All figures + correlation + outlier tables saved to:\n  {OUT_DIR}")
