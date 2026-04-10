#!/usr/bin/env python3
"""
Plot reward layouts and most-common walking paths for DSR sessions
(human single-cell ephys dataset).

Two modes:
  PLOT_MODE = 'single'  →  one subject (set SUBJECT below)
  PLOT_MODE = 'population'  →  all 28 DSR sessions overlaid

For 'population' mode each unique path is drawn as a semi-transparent line
whose alpha scales with how frequently that path was taken across subjects
and trials (more common = more opaque).

Configs plotted (rows):  3-7-9-5 | 8-2-6-7 | 1-9-5-8 | 4-8-1-3
                          6-4-2-9 | 9-1-3-4 | 7-3-4-2 | 2-5-7-6
States (columns in each panel):  A  B  C  D
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import mc

# ── Settings ──────────────────────────────────────────────────────────
PLOT_MODE = 'population'   # 'single'  or  'population'
SUBJECT   = 27             # used only when PLOT_MODE == 'single'

DATA_DIR  = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives'
SAVE_FIG  = True
OUT_DIR   = os.path.join(DATA_DIR, 'group', 'DSR_path_plots')

DSR_SUBJECTS = [27,28,31,32,33,34,35,36,37,38,40,43,44,45,46,
                49,50,51,53,55,56,57,58,59,60,61,62,63]

DSR_CONFIGS = [
    (3,7,9,5), (8,2,6,7), (1,9,5,8), (4,8,1,3),
    (6,4,2,9), (9,1,3,4), (7,3,4,2), (2,5,7,6),
]
DSR_CONFIG_LABELS = [f'{c[0]}-{c[1]}-{c[2]}-{c[3]}' for c in DSR_CONFIGS]
STATE_ORDER  = ['A', 'B', 'C', 'D']
STATE_BINS   = {'A': (0, 89), 'B': (90, 179), 'C': (180, 269), 'D': (270, 359)}

# Grid: loc 1-9 on 3×3, row-major top-left
LOC_TO_XY = {1:(0,2), 2:(1,2), 3:(2,2),
              4:(0,1), 5:(1,1), 6:(2,1),
              7:(0,0), 8:(1,0), 9:(2,0)}

# Colours
light_lavender = (190/255, 190/255, 220/255)
dark_lavender  = (117/255, 107/255, 176/255)
dark_orange    = (204/255,  85/255,   0/255)
bright_orange  = (255/255, 140/255,   0/255)
STATE_COLORS   = {'A': dark_orange, 'B': bright_orange,
                  'C': light_lavender, 'D': dark_lavender}


# ── Path extraction ────────────────────────────────────────────────────

def compress(loc_array: np.ndarray) -> tuple[int, ...]:
    """Remove NaN and collapse consecutive repeated locations."""
    clean = loc_array[~np.isnan(loc_array)].astype(int)
    if len(clean) == 0:
        return ()
    out = [clean[0]]
    for v in clean[1:]:
        if v != out[-1]:
            out.append(v)
    return tuple(out)


def paths_for_config_state(beh: pd.DataFrame, locs: pd.DataFrame,
                            cfg: tuple, state: str) -> list[tuple[int, ...]]:
    """
    Return a list of compressed location-sequences (one per correct trial)
    for a given config × state window.
    """
    mask = ((beh['loc_A'] == cfg[0]) & (beh['loc_B'] == cfg[1]) &
            (beh['loc_C'] == cfg[2]) & (beh['loc_D'] == cfg[3]) &
            (beh['correct'] == 1))
    if mask.sum() == 0:
        return []
    s, e = STATE_BINS[state]
    paths = []
    for _, row in locs.loc[mask].iterrows():
        p = compress(row.values[s:e+1])
        if p:
            paths.append(p)
    return paths


# ── Drawing ────────────────────────────────────────────────────────────

def draw_grid(ax: plt.Axes) -> None:
    for x in range(4):
        ax.plot([x - 0.5, x - 0.5], [-0.5, 2.5], color='0.82', lw=0.8, zorder=0)
    for y in range(4):
        ax.plot([-0.5, 2.5], [y - 0.5, y - 0.5], color='0.82', lw=0.8, zorder=0)


def draw_reward_marker(ax: plt.Axes, loc: int, state: str) -> None:
    if loc not in LOC_TO_XY:
        return
    x, y = LOC_TO_XY[loc]
    ax.scatter(x, y, s=420, c=[STATE_COLORS[state]], marker='o', zorder=4)
    ax.text(x, y, state, ha='center', va='center',
            fontsize=12, color='white', weight='bold', zorder=5)


def draw_paths(ax: plt.Axes, path_counter: Counter,
               color: tuple, max_lw: float = 3.5,
               min_alpha: float = 0.12, max_alpha: float = 0.85) -> None:
    """
    Draw each unique path weighted by its count.
    Alpha and line width scale with frequency (count / max_count).
    """
    if not path_counter:
        return
    max_count = max(path_counter.values())
    # Sort ascending so most-common paths are drawn on top
    for path, count in sorted(path_counter.items(), key=lambda x: x[1]):
        frac  = count / max_count
        alpha = min_alpha + frac * (max_alpha - min_alpha)
        lw    = 1.0 + frac * (max_lw - 1.0)
        xs = [LOC_TO_XY[l][0] for l in path if l in LOC_TO_XY]
        ys = [LOC_TO_XY[l][1] for l in path if l in LOC_TO_XY]
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha,
                solid_capstyle='round', solid_joinstyle='round', zorder=2)
        # Arrow on last segment
        ax.annotate('', xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle='-|>', color=color,
                                   lw=lw * 0.8, mutation_scale=12,
                                   shrinkA=1, shrinkB=6, alpha=alpha),
                    zorder=3)
    # Dot at start of path (most common)
    most_common_path = path_counter.most_common(1)[0][0]
    if most_common_path and most_common_path[0] in LOC_TO_XY:
        sx, sy = LOC_TO_XY[most_common_path[0]]
        ax.scatter(sx, sy, s=60, color=color, zorder=4, alpha=0.7)


def style_panel(ax: plt.Axes, cfg_label: str, state: str,
                reward_loc: int, n_trials: int) -> None:
    draw_grid(ax)
    draw_reward_marker(ax, reward_loc, state)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.55, 2.55)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    # Location number labels
    for loc, (x, y) in LOC_TO_XY.items():
        ax.text(x + 0.38, y + 0.38, str(loc),
                ha='center', va='center', fontsize=7, color='0.55')
    ax.set_title(f'state {state}  (n={n_trials})',
                 fontsize=8, pad=3, color=STATE_COLORS[state])


# ── Per-session path collection ────────────────────────────────────────

def collect_paths_single(sub: int) -> dict:
    """
    Returns nested dict:
        paths[cfg_tuple][state] = Counter of compressed paths
    """
    data_dict, _ = mc.analyse.helpers_human_cells.get_data(sub)
    key = f'sub-{sub:02}'
    if key not in data_dict:
        return {}
    beh  = data_dict[key]['beh']
    locs = data_dict[key]['locations']

    paths: dict = {}
    for cfg in DSR_CONFIGS:
        paths[cfg] = {}
        for state in STATE_ORDER:
            p_list = paths_for_config_state(beh, locs, cfg, state)
            paths[cfg][state] = Counter(p_list)
    return paths


def collect_paths_population() -> dict:
    """Aggregate path counters across all 28 DSR sessions."""
    agg: dict = {cfg: {s: Counter() for s in STATE_ORDER} for cfg in DSR_CONFIGS}
    for sub in DSR_SUBJECTS:
        print(f'  loading s{sub:02}...')
        sub_paths = collect_paths_single(sub)
        if not sub_paths:
            continue
        for cfg in DSR_CONFIGS:
            for state in STATE_ORDER:
                agg[cfg][state] += sub_paths[cfg][state]
    return agg


# ── Figure builder ─────────────────────────────────────────────────────

def build_figure(paths: dict, title: str) -> plt.Figure:
    """
    Layout: 8 rows (configs) × 4 columns (states A-D).
    Each panel shows the 3×3 grid with paths.
    """
    n_rows = len(DSR_CONFIGS)
    n_cols = len(STATE_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.4, n_rows * 2.4),
                             constrained_layout=True)

    for r, cfg in enumerate(DSR_CONFIGS):
        cfg_label = DSR_CONFIG_LABELS[r]
        reward_locs = {'A': cfg[0], 'B': cfg[1], 'C': cfg[2], 'D': cfg[3]}

        for c, state in enumerate(STATE_ORDER):
            ax = axes[r, c]
            counter = paths.get(cfg, {}).get(state, Counter())
            n_trials = sum(counter.values())

            style_panel(ax, cfg_label, state, reward_locs[state], n_trials)
            draw_paths(ax, counter, color=STATE_COLORS[state])

            # Row label on leftmost column
            if c == 0:
                ax.set_ylabel(cfg_label, fontsize=8, rotation=0,
                              labelpad=40, va='center')

    # Column headers
    for c, state in enumerate(STATE_ORDER):
        axes[0, c].set_title(f'State {state}\n(n=...)',
                             fontsize=9, color=STATE_COLORS[state])
        # Overwrite with proper n for first config
        cfg0 = DSR_CONFIGS[0]
        n = sum(paths.get(cfg0, {}).get(state, Counter()).values())
        axes[0, c].set_title(f'State {state}', fontsize=10,
                             color=STATE_COLORS[state], pad=4)

    # Alpha legend
    legend_handles = [
        mpatches.Patch(facecolor='0.3', alpha=0.15, label='rare path'),
        mpatches.Patch(facecolor='0.3', alpha=0.85, label='common path'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=2, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(title, fontsize=13, weight='bold', y=1.01)
    return fig


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    if PLOT_MODE == 'single':
        print(f'Collecting paths for s{SUBJECT:02}...')
        paths = collect_paths_single(SUBJECT)
        title = f's{SUBJECT:02} — DSR walking paths per config × state (correct trials only)'
        fname = f's{SUBJECT:02}_DSR_paths.png'
    else:
        print('Collecting paths across all 28 DSR sessions...')
        paths = collect_paths_population()
        title = 'Population (n=28 sessions) — DSR walking paths per config × state\n' \
                'Opacity ∝ frequency of path taken'
        fname = 'population_DSR_paths.png'

    print('Building figure...')
    fig = build_figure(paths, title)

    if SAVE_FIG:
        out_path = os.path.join(OUT_DIR, fname)
        fig.savefig(out_path, dpi=180, bbox_inches='tight')
        print(f'Saved → {out_path}')

    plt.show()


if __name__ == '__main__':
    main()
