#!/usr/bin/env python3
"""Spyder-friendly plotting of reward layouts and most-common paths per task."""
from __future__ import annotations

import os
from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# =====================
# User settings (Spyder)
# =====================
SUBJECT = "02"  # "05" or "sub-05"
CSV_PATH: Optional[str] = None  # e.g. "/full/path/sub-05_beh_fmri_clean.csv"
TASKS_TO_PLOT: Optional[list[str]] = None  # e.g. ["A1_backw", "A1_forw"]; None = all tasks
TASK_NAME_MUST_CONTAIN = None #"1"  # default: only plot tasks like A1_backw/A1_forw

SHOW_FIGURE = True
SAVE_FIGURE = False
OUTPUT_PATH: Optional[str] = None  # e.g. "scripts/figures/sub-05_task_config_reward_paths.png"
DPI = 300
STATE_LABEL_FONTSIZE = 15
STEP_LABEL_FONTSIZE = 10
TASK_TITLE_FONTSIZE = 12
SUPTITLE_FONTSIZE = 16


LOC_TO_XY = {
    1: (0, 2),
    2: (1, 2),
    3: (2, 2),
    4: (0, 1),
    5: (1, 1),
    6: (2, 1),
    7: (0, 0),
    8: (1, 0),
    9: (2, 0),
}
STATE_ORDER = ["A", "B", "C", "D"]

light_lavender = (190/255, 190/255, 220/255)
dark_lavender  = (117/255, 107/255, 176/255)
dark_orange    = (204/255, 85/255, 0/255)
bright_orange  = (255/255, 140/255, 0/255)


STATE_COLORS = {
    "A": dark_orange,
    "B": bright_orange,
    "C": light_lavender,
    "D": dark_lavender,
}
TEMP_ORDER = [
    "A_path",
    "A_reward",
    "B_path",
    "B_reward",
    "C_path",
    "C_reward",
    "D_path",
    "D_reward",
]

# Create colormap for path progression
white = (1.0, 1.0, 1.0)
dark_red = (0.55, 0.05, 0.25)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "white_to_darkred",
    [white, dark_red],
)


def resolve_beh_csv(subject: str, csv_path: Optional[str]) -> str:
    if csv_path:
        return csv_path

    laptop_csv = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{subject}/beh/{subject}_beh_fmri_clean.csv"
    if os.path.isfile(laptop_csv):
        return laptop_csv

    cluster_csv = f"/home/fs0/xpsy1114/scratch/data/derivatives/{subject}/beh/{subject}_beh_fmri_clean.csv"
    if os.path.isfile(cluster_csv):
        return cluster_csv

    raise FileNotFoundError(
        f"Could not find cleaned behavior CSV for {subject}. "
        "Set CSV_PATH explicitly at the top of this script."
    )


def most_common_reward_per_state(task_df: pd.DataFrame) -> dict[str, int]:
    rewards = {}
    rewards_df = task_df[task_df["time_bin_type"] == "reward"]

    for state in STATE_ORDER:
        state_rews = rewards_df.loc[rewards_df["state"] == state, "curr_rew"].dropna().astype(int)
        if state_rews.empty:
            continue
        rewards[state] = Counter(state_rews).most_common(1)[0][0]

    return rewards


def most_common_path_for_reg(df_reg: pd.DataFrame) -> np.ndarray:
    raw_loc_paths = []
    if df_reg.empty:
        return np.array([], dtype=int)

    max_rep = int(df_reg["repeat"].max())
    for rep in range(max_rep + 1):
        rep_locs = df_reg.loc[df_reg["repeat"] == rep, "curr_loc"].dropna().astype(int).to_numpy()
        if rep_locs.size:
            raw_loc_paths.append(tuple(rep_locs))

    if not raw_loc_paths:
        return np.array([], dtype=int)

    # Same logic as create_fMRI_model_RDMs_on_clean_beh.py (location model)
    return np.array(Counter(raw_loc_paths).most_common(1)[0][0], dtype=int)


def representative_task_chunks(
    task_df: pd.DataFrame, task_name: str, reward_map: dict[str, int]
) -> dict[str, np.ndarray]:
    if "unique_time_bin_type" not in task_df.columns:
        return {}

    state_chunks = {}
    for state in STATE_ORDER:
        reg = f"{task_name}_{state}_path"
        df_reg = task_df[task_df["unique_time_bin_type"] == reg]
        rep_path = most_common_path_for_reg(df_reg)
        if not rep_path.size:
            continue

        # Add exactly one reward endpoint so movement-step counts stay interpretable.
        if state in reward_map:
            rep_path = np.concatenate([rep_path, np.array([reward_map[state]], dtype=int)])
        state_chunks[state] = rep_path

    return state_chunks


def offset_vectors(n: int, scale: float) -> list[tuple[float, float]]:
    if n <= 1:
        return [(0.0, 0.0)]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(float(np.cos(a) * scale), float(np.sin(a) * scale)) for a in angles]


def overlap_offsets(n: int, scale: float) -> list[tuple[float, float]]:
    """Offsets for repeated visits: first stays fixed, later ones shift to avoid overlap."""
    if n <= 1:
        return [(0.0, 0.0)]
    # Keep the first visit at the default position. Then use diagonal/cross offsets.
    base = [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (1.5, 0.0),
        (-1.5, 0.0),
        (0.0, 1.5),
        (0.0, -1.5),
    ]
    out = []
    for i in range(n):
        bx, by = base[i % len(base)]
        out.append((bx * scale, by * scale))
    return out


def build_shifted_chunk_coords(
    rep_task_chunks: dict[str, np.ndarray],
    point_scale: float = 0.14,
    text_scale: float = 0.24,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Assign non-overlapping offsets for repeated locations across the full A->D route."""
    ordered_states = [s for s in STATE_ORDER if s in rep_task_chunks and rep_task_chunks[s].size]
    if not ordered_states:
        return {}

    flat_locs: list[int] = []
    chunk_lengths: dict[str, int] = {}
    for state in ordered_states:
        locs = [int(loc) for loc in rep_task_chunks[state].tolist() if int(loc) in LOC_TO_XY]
        chunk_lengths[state] = len(locs)
        flat_locs.extend(locs)

    if not flat_locs:
        return {}

    counts_total = Counter(flat_locs)
    seen = Counter()
    flat_point_coords: list[tuple[float, float]] = []
    flat_text_coords: list[tuple[float, float]] = []

    for loc in flat_locs:
        seen[loc] += 1
        occ_idx = seen[loc] - 1
        n_occ = counts_total[loc]
        base_x, base_y = LOC_TO_XY[loc]
        p_dx, p_dy = overlap_offsets(n_occ, point_scale)[occ_idx]
        flat_point_coords.append((base_x + p_dx, base_y + p_dy))
        # Place step labels near upper-left corner of each visited cell.
        corner_x = base_x - 0.24
        corner_y = base_y + 0.24
        # First visit keeps the default good position; repeated visits get shifted.
        if occ_idx == 0:
            flat_text_coords.append((corner_x, corner_y))
        elif n_occ == 2 and occ_idx == 1:
            # For exactly two visits: keep first at upper-left, place second at upper-right.
            flat_text_coords.append((base_x + 0.24, base_y + 0.24))
        else:
            t_dx, t_dy = overlap_offsets(n_occ, text_scale)[occ_idx]
            flat_text_coords.append((corner_x + t_dx, corner_y + t_dy))

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    start = 0
    for state in ordered_states:
        n = chunk_lengths[state]
        end = start + n
        out[state] = (
            np.array(flat_point_coords[start:end], dtype=float),
            np.array(flat_text_coords[start:end], dtype=float),
        )
        start = end
    return out


def draw_grid(ax: plt.Axes, x_offset: float) -> None:
    for x in range(4):
        ax.plot([x_offset + x - 0.5, x_offset + x - 0.5], [-0.5, 2.5], color="0.8", lw=1)
    for y in range(4):
        ax.plot([x_offset - 0.5, x_offset + 2.5], [y - 0.5, y - 0.5], color="0.8", lw=1)


def plot_task_panel(
    ax: plt.Axes,
    task_name: str,
    reward_map: dict[str, int],
    rep_task_chunks: dict[str, np.ndarray],
) -> None:
    draw_grid(ax, x_offset=0.0)
    draw_grid(ax, x_offset=4.0)

    # Left grid: reward locations
    rewards_by_loc = {}
    for state, loc in sorted(reward_map.items()):
        rewards_by_loc.setdefault(loc, []).append(state)

    for loc, states in rewards_by_loc.items():
        if loc not in LOC_TO_XY:
            continue
        x0, y0 = LOC_TO_XY[loc]
        for (dx, dy), state in zip(offset_vectors(len(states), 0.12), states):
            x = x0 + dx
            y = y0 + dy
            ax.scatter(x, y, s=600, c=[STATE_COLORS[state]], marker="o", zorder=3)
            ax.text(
                x,
                y,
                state,
                ha="center",
                va="center",
                fontsize=STATE_LABEL_FONTSIZE,
                color="white",
                weight="bold",
                zorder=4,
            )

    # Right grid: 4 gradient chunks (A->D), draw every step with arrowheads
    # Start slightly darker than white so the first chunk remains visible on white background.
    gradient_colors = [cmap(v) for v in np.linspace(0.22, 1.0, 4)]
    shifted_chunks = build_shifted_chunk_coords(rep_task_chunks, point_scale=0.14, text_scale=0.24)

    # Add wrap-around segment: D-end -> first A-point, using first chunk color.
    states_present = [s for s in STATE_ORDER if s in shifted_chunks]
    if len(states_present) >= 2:
        first_state = states_present[0]
        last_state = states_present[-1]
        first_xy_path = shifted_chunks[first_state][0].copy()
        last_xy_path = shifted_chunks[last_state][0].copy()
        if first_xy_path.size and last_xy_path.size:
            first_xy_path[:, 0] += 4.0
            last_xy_path[:, 0] += 4.0
            x0, y0 = last_xy_path[-1]
            x1, y1 = first_xy_path[0]
            ax.plot([x0, x1], [y0, y1], color=gradient_colors[0], lw=3.4, alpha=0.95, zorder=2)
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=gradient_colors[0],
                    lw=2.2,
                    mutation_scale=15,
                    shrinkA=1,
                    shrinkB=1,
                    alpha=0.95,
                ),
                zorder=3,
            )

    step_counter = 1
    prev_endpoint = None
    for state_idx, state in enumerate(STATE_ORDER):
        if state not in shifted_chunks:
            continue

        xy_path, xy_text = shifted_chunks[state]
        if not xy_path.size:
            continue

        xy_path[:, 0] += 4.0
        xy_text[:, 0] += 4.0
        color = gradient_colors[state_idx]

        # Optionally connect from previous chunk endpoint so the full route is continuous.
        if prev_endpoint is not None:
            x0, y0 = prev_endpoint
            x1, y1 = xy_path[0]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=2.4,
                    mutation_scale=15,
                    shrinkA=1,
                    shrinkB=1,
                    alpha=0.95,
                ),
                zorder=3,
            )

        # Draw every movement step in this chunk.
        for i in range(len(xy_path) - 1):
            x0, y0 = xy_path[i]
            x1, y1 = xy_path[i + 1]
            ax.plot([x0, x1], [y0, y1], color=color, lw=3.4, alpha=0.95, zorder=2)
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=2.2,
                    mutation_scale=15,
                    shrinkA=1,
                    shrinkB=1,
                    alpha=0.95,
                ),
                zorder=3,
            )
        for x_t, y_t in xy_text:
            ax.text(
                x_t,
                y_t,
                str(step_counter),
                fontsize=STEP_LABEL_FONTSIZE,
                color="black",
                ha="center",
                va="center",
            )
            step_counter += 1
        prev_endpoint = (xy_path[-1, 0], xy_path[-1, 1])

    ax.set_xlim(-0.8, 6.8)
    ax.set_ylim(-0.8, 2.8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(task_name, fontsize=TASK_TITLE_FONTSIZE)


def build_figure(beh_df: pd.DataFrame, subject: str, tasks_to_plot: Optional[list[str]]) -> plt.Figure:
    task_col = "task_config_ex" if "task_config_ex" in beh_df.columns else "task_config_exe"
    tasks = sorted(beh_df[task_col].dropna().unique())
    if TASK_NAME_MUST_CONTAIN:
        tasks = [t for t in tasks if TASK_NAME_MUST_CONTAIN in t]

    if tasks_to_plot:
        selected = [t for t in tasks if t in set(tasks_to_plot)]
        missing = [t for t in tasks_to_plot if t not in set(tasks)]
        if missing:
            print(f"Warning: task(s) not found and skipped: {missing}")
        tasks = selected

    if not tasks:
        raise ValueError("No tasks available to plot.")

    payload = []
    for task in tasks:
        task_df = beh_df[beh_df[task_col] == task]
        rew_map = most_common_reward_per_state(task_df)
        rep_task_chunks = representative_task_chunks(task_df, task, rew_map)
        payload.append((task, rew_map, rep_task_chunks))

    n_tasks = len(payload)
    ncols = 4
    nrows = int(np.ceil(n_tasks / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 2.8), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (task, rew_map, rep_task_chunks) in zip(axes, payload):
        plot_task_panel(ax, task, rew_map, rep_task_chunks)

    for ax in axes[n_tasks:]:
        ax.axis("off")

    fig.suptitle(
        f"{subject}: Reward Locations and Repeat Paths (filtered: '*{TASK_NAME_MUST_CONTAIN}*')",
        fontsize=SUPTITLE_FONTSIZE,
    )
    return fig


def main() -> None:
    subject = SUBJECT if SUBJECT.startswith("sub-") else f"sub-{SUBJECT}"
    csv_path = resolve_beh_csv(subject, CSV_PATH)
    beh_df = pd.read_csv(csv_path)

    fig = build_figure(beh_df, subject, TASKS_TO_PLOT)

    if SAVE_FIGURE:
        out_path = OUTPUT_PATH or os.path.join(
            os.path.dirname(csv_path), f"{subject}_task_config_reward_paths.png"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved figure to {out_path}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
