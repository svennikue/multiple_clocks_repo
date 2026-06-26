#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example cells from sustained state encoding analysis.

Plots:
1. A "beautiful" sustained EC cell (has sustained signature with robust state coding)
2. A counter-example with "high phase coding" but selective (all positive contrasts
   but NOT marked sustained because doesn't meet both p-value criteria)

Uses polar coordinate plots showing per-phase per-config mean traces, similar to
fig2C from publication_figures_human_cells.py.

Usage:
    python scripts/plot_sustained_state_examples.py \
      --run-dir /path/to/encoding_state_sustained_cv/run \
      --data-dir /path/to/ephys_humans/derivatives

"""

import argparse
import json
import sys
from pathlib import Path
import tempfile
import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "multiple_clocks_mplconfig"),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")
import era_brewer
import mc.analyse.cell_selection as cell_selection
import mc.analyse.helpers_human_cells as hh


# ── Settings ──────────────────────────────────────────────────────────
FIGURE_SIZE = (14, 6)
DPI_PUBLICATION = 300
FONT_SIZE = 11
COLOR_PALETTE = era_brewer.era_brew("Showgirl2", n=7)

# Thresholds
N_PHASES = 3
PHASES = ["Early", "Middle", "Late"]
PHASE_COLORS = [COLOR_PALETTE[0], COLOR_PALETTE[1], COLOR_PALETTE[2]]


def parse_phase_contrasts(row):
    """Extract phase contrasts from JSON column."""
    if pd.isna(row.get("phase_contrast_json")):
        return None
    try:
        d = json.loads(row["phase_contrast_json"])
        return np.array([d.get("early"), d.get("middle"), d.get("late")])
    except:
        return None


def load_sustained_results(run_dir):
    """Load results from encoding_state_sustained_cv run."""
    run_dir = Path(run_dir)
    results = pd.read_csv(run_dir / "state_sustained_cv_results.csv")
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    return config, results


def identify_example_cells(results):
    """
    Find:
    1. Best sustained EC cell (highest mean_r among sustained_sig=True in EC)
    2. Best phase-specific EC cell (highest mean_r among sustained_sig=False but 
       all positive contrasts in EC)
    """
    results["phase_contrasts"] = results.apply(parse_phase_contrasts, axis=1)
    
    def all_contrasts_positive(contrasts):
        if contrasts is None:
            return False
        return np.all(np.isfinite(contrasts)) and np.all(contrasts > 0)
    
    results["has_all_positive_contrasts"] = results["phase_contrasts"].apply(
        all_contrasts_positive
    )
    
    # EC cells only
    ec_data = results[results["roi"] == "EC"].copy()
    
    # Sustained EC cells
    ec_sustained = ec_data[ec_data["sustained_sig"]].sort_values(
        "mean_r", ascending=False
    )
    
    # Phase-specific (all positive but NOT sustained)
    ec_phase_specific = ec_data[
        (~ec_data["sustained_sig"]) & (ec_data["has_all_positive_contrasts"])
    ].sort_values("mean_r", ascending=False)
    
    sustained_cell = ec_sustained.iloc[0] if len(ec_sustained) > 0 else None
    phase_cell = ec_phase_specific.iloc[0] if len(ec_phase_specific) > 0 else None
    
    return sustained_cell, phase_cell


def parse_neuron_label(label):
    """'01_07-07-chan120-EC' -> (subject:int, cell_idx:int)."""
    try:
        sub_str, rest = str(label).split("_", 1)
        cell_idx_str = rest.split("-", 1)[0]
        return int(sub_str), int(cell_idx_str)
    except (ValueError, IndexError):
        return None, None


def load_cell_data(neuron_id, data_dir):
    """Load behavioral and neural data for a specific neuron."""
    sub, cell_idx = parse_neuron_label(neuron_id)
    if sub is None:
        return None
    
    sub_str = f"{sub:02d}"
    try:
        data = hh.load_norm_data(data_dir, [sub_str], res_data=False)
    except Exception as exc:
        print(f"Failed to load sub-{sub_str}: {exc}")
        return None
    
    key = f"sub-{sub_str}"
    if key not in data:
        return None
    
    sub_dict = data[key]
    neurons = sub_dict["normalised_neurons"]
    
    if neuron_id not in neurons:
        return None
    
    # Build config_str
    beh = sub_dict["beh"].copy().reset_index(drop=True)
    beh["config_str"] = (
        beh["loc_A"].astype(int).astype(str)
        + "-"
        + beh["loc_B"].astype(int).astype(str)
        + "-"
        + beh["loc_C"].astype(int).astype(str)
        + "-"
        + beh["loc_D"].astype(int).astype(str)
    )
    
    neuron_df = neurons[neuron_id].reset_index(drop=True)
    
    return {
        "neuron_id": neuron_id,
        "beh": beh,
        "neuron_trace": neuron_df,
        "subject": sub,
        "cell_idx": cell_idx,
    }


def get_per_config_traces(cell_data, n_phases=3, phase_bins=30):
    """
    Extract per-config per-phase mean traces.
    Returns dict of {config: [phase_0_mean, phase_1_mean, phase_2_mean]}
    """
    beh = cell_data["beh"]
    neuron_df = cell_data["neuron_trace"]
    
    correct_mask = (beh["correct"] == 1).to_numpy()
    configs = sorted(beh.loc[correct_mask, "config_str"].dropna().unique().tolist())
    
    per_config = {}
    for cfg in configs:
        idx = beh.index[(beh["config_str"] == cfg) & (beh["correct"] == 1)].to_numpy()
        if len(idx) == 0:
            continue
        
        cfg_traces = neuron_df.iloc[idx].to_numpy(dtype=float)
        mean_trace = np.nanmean(cfg_traces, axis=0)
        
        # Split into phases
        phase_traces = []
        for p in range(n_phases):
            start_bin = p * phase_bins
            end_bin = start_bin + phase_bins
            phase_mean = np.nanmean(mean_trace[start_bin:end_bin])
            phase_traces.append(phase_mean)
        
        per_config[cfg] = phase_traces
    
    return per_config


def plot_polar_state_cell(
    cell_row,
    cell_data,
    cell_type="sustained",
    save_path=None,
):
    """
    Create a polar plot showing state encoding across phases and configs.
    
    Parameters
    ----------
    cell_row : pd.Series
        Row from results DataFrame
    cell_data : dict
        Data loaded from load_cell_data
    cell_type : str
        "sustained" or "phase_specific"
    save_path : str
        Where to save figure
    """
    per_config = get_per_config_traces(cell_data)
    if not per_config:
        return
    
    configs = sorted(per_config.keys())
    n_configs = len(configs)
    
    # Create figure with polar + stats layout
    fig = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35, height_ratios=[3, 1])
    
    # Polar plots for each config
    for cfg_idx, cfg in enumerate(configs):
        ax = fig.add_subplot(gs[0, cfg_idx % 3], projection="polar")
        phase_vals = per_config[cfg]
        
        # Create 3 angular positions for the 3 phases
        theta = np.linspace(0, 2 * np.pi, N_PHASES + 1)[:-1]
        
        # Normalize for better visualization
        v_min = min([min(per_config[c]) for c in configs])
        v_max = max([max(per_config[c]) for c in configs])
        v_range = v_max - v_min if v_max > v_min else 1.0
        
        for p, (angle, val) in enumerate(zip(theta, phase_vals)):
            # Normalize to radial distance
            r = (val - v_min) / v_range + 0.5 if v_range > 0 else 1.0
            ax.plot([angle, angle], [0, r], color=PHASE_COLORS[p], linewidth=3, alpha=0.8)
            ax.scatter([angle], [r], color=PHASE_COLORS[p], s=150, zorder=5, edgecolor="black", linewidth=1)
        
        # Close the loop
        phase_vals_plot = np.append(phase_vals, phase_vals[0])
        theta_plot = np.append(theta, theta[0])
        r_vals = (np.array(phase_vals_plot) - v_min) / v_range + 0.5 if v_range > 0 else np.ones_like(phase_vals_plot)
        ax.plot(theta_plot, r_vals, "k--", alpha=0.2, linewidth=0.5)
        
        # Set phase labels
        ax.set_xticks(theta)
        ax.set_xticklabels(PHASES, fontsize=FONT_SIZE - 1)
        ax.set_ylim(0, max(r_vals) * 1.2)
        ax.set_yticks([])
        ax.set_title(f"Config: {cfg}", fontsize=FONT_SIZE, pad=15)
        ax.grid(True, alpha=0.3)
    
    # Statistics panel
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.axis("off")
    
    # Build stats text
    stats_text = f"""
    Cell: {cell_data['neuron_id']} (ROI: EC)
    Type: {cell_type.replace('_', ' ').title()}
    
    Mean state encoding r: {cell_row['mean_r']:+.4f}
    p(state encoding):     {cell_row['p_perm_state_r']:.4f}
    
    Min phase contrast:     {cell_row['min_phase_contrast']:+.4f}
    p(phase contrast):      {cell_row['p_perm_min_phase_contrast']:.4f}
    
    All phases positive:    {cell_row['all_phase_positive']}
    Sustained signature:    {cell_row['sustained_sig']}
    """
    
    if cell_type == "sustained":
        title = "Sustained State Cell: Consistent state coding across all phases"
        title_color = COLOR_PALETTE[0]
    else:
        title = "Phase-Specific Cell: Strong encoding in some phases, weak in others"
        title_color = COLOR_PALETTE[2]
    
    ax_stats.text(
        0.05, 0.95, stats_text, transform=ax_stats.transAxes,
        fontsize=FONT_SIZE - 1, verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3)
    )
    
    fig.suptitle(
        title,
        fontsize=FONT_SIZE + 2, fontweight="bold", color=title_color
    )
    
    if save_path:
        fig.savefig(save_path, dpi=DPI_PUBLICATION, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Path to encoding_state_sustained_cv run directory")
    parser.add_argument("--data-dir", required=True,
                        help="Path to ephys_humans/derivatives")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <run_dir>/example_cells)")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "example_cells"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {run_dir}")
    config, results = load_sustained_results(run_dir)
    
    print(f"Identifying example EC cells...")
    sustained_cell, phase_cell = identify_example_cells(results)
    
    if sustained_cell is None:
        print("ERROR: No sustained EC cells found")
        return
    
    print(f"\n✓ Sustained EC cell: {sustained_cell['neuron']}")
    print(f"  mean_r = {sustained_cell['mean_r']:+.4f}")
    print(f"  sustained_sig = {sustained_cell['sustained_sig']}")
    
    if phase_cell is not None:
        print(f"\n✓ Phase-specific EC cell: {phase_cell['neuron']}")
        print(f"  mean_r = {phase_cell['mean_r']:+.4f}")
        print(f"  sustained_sig = {phase_cell['sustained_sig']}")
        print(f"  has_all_positive_contrasts = {phase_cell['phase_contrasts']}")
    else:
        print("\n⚠ No phase-specific EC cells found (all positive cells are also sustained)")
    
    # Plot sustained cell
    print(f"\nLoading data for sustained cell...")
    sustained_data = load_cell_data(sustained_cell["neuron"], data_dir)
    if sustained_data:
        sustained_path = output_dir / "EC_sustained_cell.pdf"
        plot_polar_state_cell(
            sustained_cell, sustained_data,
            cell_type="sustained",
            save_path=str(sustained_path)
        )
        print(f"Saved: {sustained_path}")
    else:
        print(f"Could not load data for sustained cell")
    
    # Plot phase-specific cell
    if phase_cell is not None:
        print(f"\nLoading data for phase-specific cell...")
        phase_data = load_cell_data(phase_cell["neuron"], data_dir)
        if phase_data:
            phase_path = output_dir / "EC_phase_specific_cell.pdf"
            plot_polar_state_cell(
                phase_cell, phase_data,
                cell_type="phase_specific",
                save_path=str(phase_path)
            )
            print(f"Saved: {phase_path}")
        else:
            print(f"Could not load data for phase-specific cell")
    
    print(f"\nAll example cells saved to: {output_dir}")


if __name__ == "__main__":
    main()
