#!/usr/bin/env python3
"""
Publication-ready figure: walking-path consistency across DSR sessions.

Restricts every analysis to the first N=10 correct repeats per
(subject, config).

Outputs
-------
1. One paths figure (population, all configs in a single A4-portrait page):
     - top: large showcase grid for SHOWCASE_CFG with all 4 reward markers
       and the population paths overlaid (single grid, full A→D).
     - below: 7 remaining configs × 4 state panels (rows of 4 small grids).
2. One consistency-across-repeats figure (modal-path agreement and mean
   pairwise Jaccard per repeat).
3. CSV summaries:
     - path_consistency_summary.csv  (per (config, state) consistency stats)
     - subject_level_behaviour.csv   (per-subject behavioural counts)
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from itertools import combinations

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import mc

# ── Settings ──────────────────────────────────────────────────────────
DATA_DIR  = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
SAVE_FIG  = True
OUT_DIR   = os.path.join(DATA_DIR, 'group', 'DSR_path_plots')

DSR_SUBJECTS = [27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 40, 43, 44, 45, 46,
                49, 50, 51, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63]

#DSR_SUBJECTS = [i for i in range(1,64)]


DSR_CONFIGS = [
    (3, 7, 9, 5), (8, 2, 6, 7), (1, 9, 5, 8), (4, 8, 1, 3),
    (6, 4, 2, 9), (9, 1, 3, 4), (7, 3, 4, 2), (2, 5, 7, 6),
]
DSR_CONFIG_LABELS = {c: f'{c[0]}-{c[1]}-{c[2]}-{c[3]}' for c in DSR_CONFIGS}
SHOWCASE_CFG = (8, 2, 6, 7)

STATE_ORDER = ['A', 'B', 'C', 'D']
STATE_BINS = {'A': (0, 89), 'B': (90, 179), 'C': (180, 269), 'D': (270, 359)}

# Grid: loc 1-9 on 3×3, row-major top-left
LOC_TO_XY = {1: (0, 2), 2: (1, 2), 3: (2, 2),
             4: (0, 1), 5: (1, 1), 6: (2, 1),
             7: (0, 0), 8: (1, 0), 9: (2, 0)}

# Colours
light_lavender = (190/255, 190/255, 220/255)
dark_lavender  = (117/255, 107/255, 176/255)
dark_orange    = (204/255,  85/255,   0/255)
bright_orange  = (255/255, 140/255,   0/255)
STATE_COLORS = {'A': dark_orange, 'B': bright_orange,
                'C': light_lavender, 'D': dark_lavender}

N_REPEATS_USED = 10  # first N correct trials per (subject, config)

# Publication font sizes (≥ 9 everywhere).
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 11,
})


# ── Path extraction ───────────────────────────────────────────────────
def compress(loc_array: np.ndarray) -> tuple[int, ...]:
    """Drop NaN and collapse consecutive repeated locations."""
    clean = loc_array[~np.isnan(loc_array)].astype(int)
    if len(clean) == 0:
        return ()
    out = [clean[0]]
    for v in clean[1:]:
        if v != out[-1]:
            out.append(v)
    return tuple(out)


def state_path(loc_row: pd.Series, state: str) -> tuple[int, ...]:
    s, e = STATE_BINS[state]
    return compress(loc_row.values[s:e + 1])


def first_n_correct(beh: pd.DataFrame, locs: pd.DataFrame,
                    cfg: tuple, n: int) -> pd.DataFrame:
    mask = ((beh['loc_A'] == cfg[0]) & (beh['loc_B'] == cfg[1]) &
            (beh['loc_C'] == cfg[2]) & (beh['loc_D'] == cfg[3]) &
            (beh['correct'] == 1))
    return locs.loc[mask].head(n)


def collect_population():
    """Returns:
        counters: dict[(cfg, state)] -> Counter of paths (across subjects, ≤N each)
        repeats:  dict[(cfg, state)] -> list[list[path]] one per subject
                                        each inner list ordered by repeat index
        beh_rows: list of dicts, per-subject behavioural counts
    """
    counters = {(cfg, s): Counter() for cfg in DSR_CONFIGS for s in STATE_ORDER}
    repeats = {(cfg, s): [] for cfg in DSR_CONFIGS for s in STATE_ORDER}
    beh_rows = []

    for sub in DSR_SUBJECTS:
        print(f'  loading s{sub:02}...')
        data_dict, _ = mc.analyse.helpers_human_cells.get_data(sub)
        key = f'sub-{sub:02}'
        if key not in data_dict:
            continue
        beh = data_dict[key]['beh']
        locs = data_dict[key]['locations']

        # ── lower-level behavioural counts ─────────────────────
        n_trials_total = int(len(beh))
        n_correct_total = int((beh['correct'] == 1).sum())

        # repeats per config (correct + total)
        per_cfg_correct = []
        per_cfg_total = []
        n_configs_seen = 0
        for cfg in DSR_CONFIGS:
            cfg_mask = ((beh['loc_A'] == cfg[0]) & (beh['loc_B'] == cfg[1]) &
                        (beh['loc_C'] == cfg[2]) & (beh['loc_D'] == cfg[3]))
            n_total = int(cfg_mask.sum())
            n_corr = int((cfg_mask & (beh['correct'] == 1)).sum())
            per_cfg_total.append(n_total)
            per_cfg_correct.append(n_corr)
            if n_total > 0:
                n_configs_seen += 1

        beh_rows.append({
            'subject':                 sub,
            'n_trials_total':          n_trials_total,
            'n_correct_total':         n_correct_total,
            'accuracy':                (n_correct_total / n_trials_total
                                        if n_trials_total else float('nan')),
            'n_configs_seen':          n_configs_seen,
            'mean_correct_per_config': float(np.mean(per_cfg_correct)),
            'min_correct_per_config':  int(np.min(per_cfg_correct)),
            'max_correct_per_config':  int(np.max(per_cfg_correct)),
            'sd_correct_per_config':   float(np.std(per_cfg_correct, ddof=1)
                                              if len(per_cfg_correct) > 1 else 0.0),
            'mean_total_per_config':   float(np.mean(per_cfg_total)),
            'min_total_per_config':    int(np.min(per_cfg_total)),
            'max_total_per_config':    int(np.max(per_cfg_total)),
            'sd_total_per_config':     float(np.std(per_cfg_total, ddof=1)
                                              if len(per_cfg_total) > 1 else 0.0),
        })

        # ── path collection (first N correct only) ─────────────
        for cfg in DSR_CONFIGS:
            trials = first_n_correct(beh, locs, cfg, N_REPEATS_USED)
            if len(trials) == 0:
                for s in STATE_ORDER:
                    repeats[(cfg, s)].append([])
                continue

            sub_state_paths = {s: [] for s in STATE_ORDER}
            for _, row in trials.iterrows():
                for s in STATE_ORDER:
                    p = state_path(row, s)
                    sub_state_paths[s].append(p)
                    if p:
                        counters[(cfg, s)][p] += 1
            for s in STATE_ORDER:
                repeats[(cfg, s)].append(sub_state_paths[s])

    return counters, repeats, beh_rows


# ── Stats ─────────────────────────────────────────────────────────────
def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return float('nan')
    p = np.array([c / total for c in counter.values()], dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def jaccard(p1, p2) -> float:
    s1, s2 = set(p1), set(p2)
    if not s1 and not s2:
        return float('nan')
    return len(s1 & s2) / max(len(s1 | s2), 1)


def mean_pairwise_jaccard(paths) -> float:
    paths = [p for p in paths if p]
    if len(paths) < 2:
        return float('nan')
    vals = [jaccard(a, b) for a, b in combinations(paths, 2)]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float('nan')


def per_state_stats(counter: Counter, paths_flat) -> dict:
    n = sum(counter.values())
    if n == 0:
        return dict(n_trials=0, n_unique=0, modal_prop=float('nan'),
                    entropy=float('nan'), mean_pairwise_jaccard=float('nan'))
    modal = counter.most_common(1)[0][1]
    return dict(
        n_trials=n,
        n_unique=len(counter),
        modal_prop=modal / n,
        entropy=shannon_entropy(counter),
        mean_pairwise_jaccard=mean_pairwise_jaccard(paths_flat),
    )


def compute_summary(counters, repeats) -> pd.DataFrame:
    rows = []
    for cfg in DSR_CONFIGS:
        for s in STATE_ORDER:
            flat = [p for sub_lst in repeats[(cfg, s)] for p in sub_lst]
            st = per_state_stats(counters[(cfg, s)], flat)
            st['cfg'] = DSR_CONFIG_LABELS[cfg]
            st['state'] = s
            st['n_subjects'] = sum(1 for sl in repeats[(cfg, s)] if sl)
            rows.append(st)
    df = pd.DataFrame(rows)
    return df[['cfg', 'state', 'n_subjects', 'n_trials', 'n_unique',
               'modal_prop', 'entropy', 'mean_pairwise_jaccard']]


def repeat_curves(repeats):
    """For each repeat r=1..N, mean across-subject modal-path proportion
    and mean pairwise Jaccard across all (cfg, state) groupings."""
    n = N_REPEATS_USED
    modal_means, modal_stds = [], []
    jacc_means, jacc_stds = [], []
    for r in range(n):
        modals, jaccs = [], []
        for per_subj in repeats.values():
            paths_at_r = [lst[r] for lst in per_subj
                          if len(lst) > r and lst[r]]
            if len(paths_at_r) < 2:
                continue
            cnt = Counter(paths_at_r)
            modals.append(cnt.most_common(1)[0][1] / len(paths_at_r))
            j = mean_pairwise_jaccard(paths_at_r)
            if not math.isnan(j):
                jaccs.append(j)
        modal_means.append(float(np.mean(modals)) if modals else float('nan'))
        modal_stds.append(float(np.std(modals, ddof=1)) if len(modals) > 1 else 0.0)
        jacc_means.append(float(np.mean(jaccs)) if jaccs else float('nan'))
        jacc_stds.append(float(np.std(jaccs, ddof=1)) if len(jaccs) > 1 else 0.0)
    return (np.array(modal_means), np.array(modal_stds),
            np.array(jacc_means),  np.array(jacc_stds))


# ── Drawing primitives ────────────────────────────────────────────────
def draw_grid(ax, lw=0.7):
    for x in range(4):
        ax.plot([x - 0.5, x - 0.5], [-0.5, 2.5], color='0.82', lw=lw, zorder=0)
    for y in range(4):
        ax.plot([-0.5, 2.5], [y - 0.5, y - 0.5], color='0.82', lw=lw, zorder=0)


def draw_reward(ax, loc, state, fontsize=10, marker_size=380):
    if loc not in LOC_TO_XY:
        return
    x, y = LOC_TO_XY[loc]
    ax.scatter(x, y, s=marker_size, c=[STATE_COLORS[state]],
               marker='o', zorder=4)
    ax.text(x, y, state, ha='center', va='center',
            fontsize=fontsize, color='white', weight='bold', zorder=5)


def draw_paths(ax, path_counter: Counter, color,
               max_lw=3.0, min_alpha=0.12, max_alpha=0.85):
    if not path_counter:
        return
    mx = max(path_counter.values())
    for path, count in sorted(path_counter.items(), key=lambda kv: kv[1]):
        frac = count / mx
        alpha = min_alpha + frac * (max_alpha - min_alpha)
        lw = 1.0 + frac * (max_lw - 1.0)
        xs = [LOC_TO_XY[l][0] for l in path if l in LOC_TO_XY]
        ys = [LOC_TO_XY[l][1] for l in path if l in LOC_TO_XY]
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha,
                solid_capstyle='round', solid_joinstyle='round', zorder=2)
        ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle='-|>', color=color,
                                    lw=lw * 0.8, mutation_scale=10,
                                    shrinkA=1, shrinkB=5, alpha=alpha),
                    zorder=3)


def style_panel(ax, show_loc_numbers=False, loc_fontsize=9):
    draw_grid(ax)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.55, 2.55)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if show_loc_numbers:
        for loc, (x, y) in LOC_TO_XY.items():
            ax.text(x + 0.38, y + 0.38, str(loc),
                    ha='center', va='center',
                    fontsize=loc_fontsize, color='0.55')


def draw_full_grid(ax, cfg, state_counters, show_loc_numbers=True,
                   marker_size=520, marker_fontsize=12, loc_fontsize=9):
    """Single grid: all 4 rewards + paths for all states overlaid."""
    style_panel(ax, show_loc_numbers=show_loc_numbers, loc_fontsize=loc_fontsize)
    for s in STATE_ORDER:
        draw_paths(ax, state_counters.get(s, Counter()),
                   color=STATE_COLORS[s])
    for i, s in enumerate(STATE_ORDER):
        draw_reward(ax, cfg[i], s, fontsize=marker_fontsize,
                    marker_size=marker_size)


# ── Figure 1: paths (showcase + small grids) ──────────────────────────
def build_paths_figure(counters, summary_df) -> plt.Figure:
    other_cfgs = [c for c in DSR_CONFIGS if c != SHOWCASE_CFG]
    n_rows_small = len(other_cfgs)
    n_cols = len(STATE_ORDER)

    # A4 portrait: 8.27 x 11.69. Leave a little margin.
    fig = plt.figure(figsize=(8.0, 11.0))

    # Top showcase + bottom small-grid block
    gs = gridspec.GridSpec(
        nrows=2, ncols=1,
        height_ratios=[3.0, n_rows_small * 1.05],
        figure=fig, hspace=0.16,
    )

    # Showcase (centered, square)
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[0], width_ratios=[1, 1.6, 1])
    ax_show = fig.add_subplot(gs_top[0, 1])
    showcase_counters = {s: counters[(SHOWCASE_CFG, s)] for s in STATE_ORDER}
    draw_full_grid(ax_show, SHOWCASE_CFG, showcase_counters,
                   show_loc_numbers=True, marker_size=560,
                   marker_fontsize=12, loc_fontsize=9)
    ax_show.set_title(
        f'{DSR_CONFIG_LABELS[SHOWCASE_CFG]} — full A→B→C→D path',
        fontsize=11, fontweight='bold', pad=4,
    )

    # Bottom: rows × cols of state panels. Wider wspace leaves a vertical
    # gutter between state columns for the per-panel stats text.
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        n_rows_small, n_cols, subplot_spec=gs[1],
        wspace=0.55, hspace=0.20,
    )

    sum_lookup = summary_df.set_index(['cfg', 'state'])

    for r, cfg in enumerate(other_cfgs):
        cfg_label = DSR_CONFIG_LABELS[cfg]
        for c, state in enumerate(STATE_ORDER):
            ax = fig.add_subplot(gs_bot[r, c])
            style_panel(ax, show_loc_numbers=False)
            draw_paths(ax, counters[(cfg, state)], color=STATE_COLORS[state])
            draw_reward(ax, cfg[c], state, fontsize=9, marker_size=190)

            # Top-row column titles
            if r == 0:
                ax.set_title(f'State {state}', fontsize=9, pad=3,
                             color=STATE_COLORS[state], fontweight='bold')

            # Per-panel stats annotation, placed in the vertical gutter
            # immediately to the right of the panel so it doesn't overlap
            # the grid. Two lines: n on top, modal % below.
            row = sum_lookup.loc[(cfg_label, state)]
            ax.text(1.05, 0.5,
                    f"n={int(row['n_trials'])}\nmod={row['modal_prop']*100:.0f}%",
                    transform=ax.transAxes, fontsize=9,
                    ha='left', va='center', color='0.30',
                    linespacing=1.05)

            if c == 0:
                # Vertical config label flush left of the row
                ax.set_ylabel(cfg_label, fontsize=9, labelpad=3)

    # Legend: state colours + path-frequency opacity legend
    state_handles = [
        mpatches.Patch(facecolor=STATE_COLORS[s], label=f'State {s}')
        for s in STATE_ORDER
    ]
    freq_handles = [
        mpatches.Patch(facecolor='0.4', alpha=0.18, label='rare path'),
        mpatches.Patch(facecolor='0.4', alpha=0.85, label='common path'),
    ]
    fig.legend(
        handles=state_handles + freq_handles,
        loc='lower center', ncol=6, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        f'Walking-path consistency across {len(DSR_SUBJECTS)} subjects '
        f'(first {N_REPEATS_USED} correct repeats per task)',
        fontsize=11, fontweight='bold', y=0.995,
    )
    return fig


# ── Figure 2: consistency across repeats ──────────────────────────────
def build_consistency_figure(repeats) -> plt.Figure:
    modal_m, modal_s, jacc_m, jacc_s = repeat_curves(repeats)
    x = np.arange(1, N_REPEATS_USED + 1)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6),
                             constrained_layout=True)

    ax = axes[0]
    ax.plot(x, modal_m, 'o-', color='tab:blue', lw=1.6, label='mean')
    ax.fill_between(x, modal_m - modal_s, modal_m + modal_s,
                    color='tab:blue', alpha=0.15, label='±1 SD')
    ax.set_xlabel('Repeat number')
    ax.set_ylabel('Modal-path proportion')
    ax.set_title('Across-subject path agreement per repeat')
    ax.set_xticks(x)
    ax.set_ylim(0, 1.03)
    ax.legend(loc='lower right', fontsize=9)

    ax = axes[1]
    ax.plot(x, jacc_m, 'o-', color='tab:green', lw=1.6, label='mean')
    ax.fill_between(x, jacc_m - jacc_s, jacc_m + jacc_s,
                    color='tab:green', alpha=0.15, label='±1 SD')
    ax.set_xlabel('Repeat number')
    ax.set_ylabel('Mean pairwise Jaccard')
    ax.set_title('Path-overlap (Jaccard) per repeat')
    ax.set_xticks(x)
    ax.set_ylim(0, 1.03)
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle(
        f'Behavioural consistency across the first {N_REPEATS_USED} repeats '
        f'(averaged over configs × states)',
        fontsize=10, fontweight='bold',
    )
    return fig


# ── Print helpers ─────────────────────────────────────────────────────
def print_subject_level_stats(beh_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(beh_rows)
    print('\n=== Lower-level behavioural counts (per subject) ===')
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 160):
        print(df.to_string(index=False))

    def _summ(name, vals):
        a = np.asarray(vals, dtype=float)
        return (f"  {name:<32s} mean={np.mean(a):7.2f}  "
                f"sd={np.std(a, ddof=1):6.2f}  "
                f"min={np.min(a):6.2f}  max={np.max(a):6.2f}  "
                f"n={len(a)}")

    print('\n=== Across-subject summary ===')
    for col in ['n_trials_total', 'n_correct_total', 'accuracy',
                'n_configs_seen', 'mean_correct_per_config',
                'min_correct_per_config', 'max_correct_per_config',
                'sd_correct_per_config', 'mean_total_per_config',
                'sd_total_per_config']:
        print(_summ(col, df[col].to_numpy()))
    return df


def print_path_summary(summary_df: pd.DataFrame) -> None:
    print('\n=== Per (config, state) path-consistency stats ===')
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 140):
        print(summary_df.to_string(index=False))

    print('\n=== Aggregated path-consistency (mean ± SD over cfg × state) ===')
    for col in ['modal_prop', 'entropy', 'n_unique', 'mean_pairwise_jaccard']:
        v = summary_df[col].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        print(f"  {col:<26s} mean={v.mean():.3f}  sd={v.std(ddof=1):.3f}  "
              f"min={v.min():.3f}  max={v.max():.3f}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Collecting paths across {len(DSR_SUBJECTS)} DSR sessions '
          f'(first {N_REPEATS_USED} correct repeats per task)...')
    counters, repeats, beh_rows = collect_population()

    beh_df = print_subject_level_stats(beh_rows)
    beh_df.to_csv(os.path.join(OUT_DIR, 'subject_level_behaviour.csv'),
                  index=False)

    summary_df = compute_summary(counters, repeats)
    summary_df.to_csv(os.path.join(OUT_DIR, 'path_consistency_summary.csv'),
                      index=False)
    print_path_summary(summary_df)

    print('\nBuilding figures...')
    fig1 = build_paths_figure(counters, summary_df)
    fig2 = build_consistency_figure(repeats)

    if SAVE_FIG:
        out1 = os.path.join(OUT_DIR, 'population_DSR_paths_combined.png')
        out2 = os.path.join(OUT_DIR, 'population_DSR_repeat_consistency.png')
        fig1.savefig(out1, dpi=300, bbox_inches='tight')
        fig1.savefig(out1.replace('.png', '.pdf'), bbox_inches='tight')
        fig2.savefig(out2, dpi=300, bbox_inches='tight')
        fig2.savefig(out2.replace('.png', '.pdf'), bbox_inches='tight')
        print(f'\nSaved → {out1}')
        print(f'Saved → {out2}')

    plt.show()


if __name__ == '__main__':
    main()
