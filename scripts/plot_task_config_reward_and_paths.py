#!/usr/bin/env python3
"""Spyder-friendly plotting of reward layouts and most-common paths per task."""
from __future__ import annotations

import os
from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =====================
# User settings (Spyder)
# =====================
SUBJECT = "33"  # "05" or "sub-05"
CSV_PATH: Optional[str] = None  # e.g. "/full/path/sub-05_beh_fmri_clean.csv"
TASKS_TO_PLOT: Optional[list[str]] = None  # e.g. ["A1_backw", "A1_forw"]; None = all tasks
TASK_NAME_MUST_CONTAIN = "1"  # default: only plot tasks like A1_backw/A1_forw

SHOW_FIGURE = True
SAVE_FIGURE = False
OUTPUT_PATH: Optional[str] = None  # e.g. "scripts/figures/sub-05_task_config_reward_paths.png"
DPI = 300


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
STATE_COLORS = {
    "A": "tab:red",
    "B": "tab:orange",
    "C": "tab:green",
    "D": "tab:blue",
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


def representative_task_path(task_df: pd.DataFrame, task_name: str) -> np.ndarray:
    if "unique_time_bin_type" not in task_df.columns:
        return np.array([], dtype=int)

    chunks = []
    for suffix in TEMP_ORDER:
        reg = f"{task_name}_{suffix}"
        df_reg = task_df[task_df["unique_time_bin_type"] == reg]
        rep_chunk = most_common_path_for_reg(df_reg)
        if rep_chunk.size:
            chunks.append(rep_chunk)

    if not chunks:
        return np.array([], dtype=int)
    return np.concatenate(chunks)


def offset_vectors(n: int, scale: float) -> list[tuple[float, float]]:
    if n <= 1:
        return [(0.0, 0.0)]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(float(np.cos(a) * scale), float(np.sin(a) * scale)) for a in angles]


def jitter_repeated_locations(path: np.ndarray, scale: float = 0.08) -> np.ndarray:
    if path.size == 0:
        return np.zeros((0, 2), dtype=float)
    counts_total = Counter(path.tolist())
    seen = Counter()
    out = []
    for loc in path.tolist():
        if loc not in LOC_TO_XY:
            continue
        seen[loc] += 1
        offsets = offset_vectors(counts_total[loc], scale)
        dx, dy = offsets[seen[loc] - 1]
        x, y = LOC_TO_XY[loc]
        out.append((x + dx, y + dy))
    return np.array(out, dtype=float)


def draw_grid(ax: plt.Axes, x_offset: float) -> None:
    for x in range(4):
        ax.plot([x_offset + x - 0.5, x_offset + x - 0.5], [-0.5, 2.5], color="0.8", lw=1)
    for y in range(4):
        ax.plot([x_offset - 0.5, x_offset + 2.5], [y - 0.5, y - 0.5], color="0.8", lw=1)


def plot_task_panel(
    ax: plt.Axes,
    task_name: str,
    reward_map: dict[str, int],
    rep_task_path: np.ndarray,
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
            ax.scatter(x, y, s=160, color=STATE_COLORS[state], zorder=3)
            ax.text(x, y, state, ha="center", va="center", fontsize=8, color="white", weight="bold", zorder=4)

    # Right grid: representative task-level sequence from location-model logic
    if rep_task_path.size:
        xy_path = jitter_repeated_locations(rep_task_path, scale=0.08)
        if xy_path.size:
            xy_path[:, 0] += 4.0
            ax.plot(xy_path[:, 0], xy_path[:, 1], color="black", lw=1.3, alpha=0.8, zorder=2)
            ax.scatter(xy_path[:, 0], xy_path[:, 1], color="black", s=20, alpha=0.85, zorder=3)
            for i_step, (x, y) in enumerate(xy_path, start=1):
                ax.text(x + 0.08, y + 0.08, str(i_step), fontsize=7, color="black")

    ax.set_xlim(-0.8, 6.8)
    ax.set_ylim(-0.8, 2.8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(task_name, fontsize=9)


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
        rep_task_path = representative_task_path(task_df, task)
        payload.append((task, rew_map, rep_task_path))

    n_tasks = len(payload)
    ncols = 4
    nrows = int(np.ceil(n_tasks / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 2.8), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (task, rew_map, rep_task_path) in zip(axes, payload):
        plot_task_panel(ax, task, rew_map, rep_task_path)

    for ax in axes[n_tasks:]:
        ax.axis("off")

    fig.suptitle(
        f"{subject}: Reward Locations and Repeat Paths (filtered: '*{TASK_NAME_MUST_CONTAIN}*')",
        fontsize=13,
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
