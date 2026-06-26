#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize sustained state signature results from encoding_state_sustained_cv.py

Loads results from a completed run and creates publication-quality plots.

Usage:
    python scripts/visualize_sustained_state.py --run-dir <path_to_run>

"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).parent.parent / ".mplconfig"),
)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")
import era_brewer

# ── Settings ──────────────────────────────────────────────────────────
FIGURE_SIZE = (12, 10)
DPI_DISPLAY = 100
DPI_PUBLICATION = 300
FONT_SIZE = 11
COLOR_PALETTE = era_brewer.era_brew("Showgirl2", n=7)


# ── Helper functions ──────────────────────────────────────────────────
def load_results(run_dir):
    """Load all results from a sustained state encoding run."""
    run_dir = Path(run_dir)
    
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    
    results = pd.read_csv(run_dir / "state_sustained_cv_results.csv")
    roi_summary = pd.read_csv(run_dir / "state_sustained_cv_roi_summary.csv")
    
    return config, results, roi_summary


def setup_figure_style():
    """Set up matplotlib for publication quality figures."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "Arial",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    })


def plot_roi_summary(roi_summary, ax):
    """Bar plot of sustained signature counts per ROI."""
    rois = roi_summary["roi"].values
    n_sust = roi_summary["n_sustained_sig"].values
    n_total = roi_summary["n_cells"].values
    binom_p = roi_summary["binom_p_sustained_sig"].values
    
    # Determine significance colors
    colors = [COLOR_PALETTE[0] if p < 0.05 else COLOR_PALETTE[2] 
              for p in binom_p]
    
    x_pos = np.arange(len(rois))
    bars = ax.bar(x_pos, n_sust, color=colors, alpha=0.8, edgecolor="black", linewidth=1.0)
    
    # Add value labels
    for i, (bar, count, total, p) in enumerate(zip(bars, n_sust, n_total, binom_p)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{int(count)}/{int(total)}\np={p:.3f}",
                ha="center", va="bottom", fontsize=FONT_SIZE-2)
    
    ax.set_xlabel("ROI", fontsize=FONT_SIZE)
    ax.set_ylabel("N cells with sustained signature", fontsize=FONT_SIZE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rois, rotation=45, ha="right")
    ax.set_title("Sustained State Signature: Cell Counts per ROI", fontsize=FONT_SIZE+1, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Add legend
    sig_patch = mpatches.Patch(color=COLOR_PALETTE[0], alpha=0.8, label="Binomial p < 0.05")
    nonsig_patch = mpatches.Patch(color=COLOR_PALETTE[2], alpha=0.8, label="Binomial p ≥ 0.05")
    ax.legend(handles=[sig_patch, nonsig_patch], loc="upper right", fontsize=FONT_SIZE-1)


def plot_fraction_sustained(roi_summary, ax):
    """Plot fraction of cells with sustained signature per ROI."""
    rois = roi_summary["roi"].values
    frac = roi_summary["frac_sustained_sig"].values
    binom_p = roi_summary["binom_p_sustained_sig"].values
    
    colors = [COLOR_PALETTE[0] if p < 0.05 else COLOR_PALETTE[2] 
              for p in binom_p]
    
    x_pos = np.arange(len(rois))
    bars = ax.bar(x_pos, frac * 100, color=colors, alpha=0.8, edgecolor="black", linewidth=1.0)
    
    # Add horizontal reference line at 5% (expected by chance at alpha=0.05)
    ax.axhline(y=5.0, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label="Expected by chance (α=0.05)")
    
    for bar, f in zip(bars, frac):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{f*100:.1f}%",
                ha="center", va="bottom", fontsize=FONT_SIZE-2)
    
    ax.set_xlabel("ROI", fontsize=FONT_SIZE)
    ax.set_ylabel("Fraction with sustained signature (%)", fontsize=FONT_SIZE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rois, rotation=45, ha="right")
    ax.set_title("Sustained State Signature: Fraction per ROI", fontsize=FONT_SIZE+1, pad=10)
    ax.set_ylim(0, max(frac * 100) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=FONT_SIZE-1)


def plot_binomial_pvalues(roi_summary, ax):
    """Plot binomial p-values per ROI (log scale)."""
    rois = roi_summary["roi"].values
    pvals = roi_summary["binom_p_sustained_sig"].values
    
    # Clip p-values for log scale
    pvals_clipped = np.clip(pvals, 1e-5, 1.0)
    colors = [COLOR_PALETTE[0] if p < 0.05 else COLOR_PALETTE[2] 
              for p in pvals]
    
    x_pos = np.arange(len(rois))
    bars = ax.bar(x_pos, -np.log10(pvals_clipped), color=colors, alpha=0.8, edgecolor="black", linewidth=1.0)
    
    # Add reference line for p=0.05
    ax.axhline(y=-np.log10(0.05), color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="p = 0.05")
    
    for bar, p in zip(bars, pvals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{p:.4f}",
                ha="center", va="bottom", fontsize=FONT_SIZE-2)
    
    ax.set_xlabel("ROI", fontsize=FONT_SIZE)
    ax.set_ylabel("-log10(p-value)", fontsize=FONT_SIZE)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rois, rotation=45, ha="right")
    ax.set_title("Binomial Test: Sustained State Signature Significance", fontsize=FONT_SIZE+1, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=FONT_SIZE-1)


def plot_state_r_distribution(results, roi_summary, ax):
    """Box plot of mean state correlation by ROI."""
    rois = roi_summary["roi"].values
    
    data_per_roi = []
    for roi in rois:
        roi_data = results[results["roi"] == roi]["mean_r"].values
        roi_data = roi_data[np.isfinite(roi_data)]
        data_per_roi.append(roi_data)
    
    bp = ax.boxplot(data_per_roi, labels=rois, patch_artist=True)
    
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_PALETTE[0])
        patch.set_alpha(0.7)
    
    ax.set_xlabel("ROI", fontsize=FONT_SIZE)
    ax.set_ylabel("Mean state correlation (r)", fontsize=FONT_SIZE)
    ax.set_title("State Encoding Strength Distribution", fontsize=FONT_SIZE+1, pad=10)
    ax.set_xticklabels(rois, rotation=45, ha="right")
    ax.axhline(y=0, color="black", linestyle=":", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle=":")


def plot_phase_contrast_distribution(results, roi_summary, ax):
    """Box plot of minimum phase contrast by ROI."""
    rois = roi_summary["roi"].values
    
    data_per_roi = []
    for roi in rois:
        roi_data = results[results["roi"] == roi]["min_phase_contrast"].values
        roi_data = roi_data[np.isfinite(roi_data)]
        data_per_roi.append(roi_data)
    
    bp = ax.boxplot(data_per_roi, labels=rois, patch_artist=True)
    
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_PALETTE[1])
        patch.set_alpha(0.7)
    
    ax.set_xlabel("ROI", fontsize=FONT_SIZE)
    ax.set_ylabel("Min phase contrast", fontsize=FONT_SIZE)
    ax.set_title("Cross-Phase Contrast Distribution", fontsize=FONT_SIZE+1, pad=10)
    ax.set_xticklabels(rois, rotation=45, ha="right")
    ax.axhline(y=0, color="black", linestyle=":", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle=":")


def plot_roi_statistics_table(roi_summary, ax):
    """Create a text-based summary table of ROI statistics."""
    ax.axis("off")
    
    # Create table data
    table_data = []
    table_data.append(["ROI", "N", "n_sust", "frac", "mean_r", "min_contrast", "binom_p"])
    
    for _, row in roi_summary.iterrows():
        table_data.append([
            row["roi"],
            f"{int(row['n_cells'])}",
            f"{int(row['n_sustained_sig'])}",
            f"{row['frac_sustained_sig']:.3f}",
            f"{row['mean_state_r']:.4f}",
            f"{row['mean_min_phase_contrast']:.4f}",
            f"{row['binom_p_sustained_sig']:.4e}",
        ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc="center", loc="center",
                    colWidths=[0.12, 0.08, 0.12, 0.12, 0.12, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE - 2)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor(COLOR_PALETTE[3])
        table[(0, i)].set_text_props(weight="bold")
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor(COLOR_PALETTE[4])
            else:
                table[(i, j)].set_facecolor("white")
    
    ax.set_title("ROI Summary Statistics", fontsize=FONT_SIZE+1, pad=10)


def create_comprehensive_figure(config, results, roi_summary, output_dir):
    """Create a comprehensive multi-panel figure."""
    setup_figure_style()
    
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Panel layout
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1:, 2])
    
    # Create plots
    plot_roi_summary(roi_summary, ax1)
    plot_fraction_sustained(roi_summary, ax2)
    plot_binomial_pvalues(roi_summary, ax3)
    plot_state_r_distribution(results, roi_summary, ax4)
    plot_phase_contrast_distribution(results, roi_summary, ax5)
    plot_roi_statistics_table(roi_summary, ax6)
    
    # Overall title
    run_tag = config.get("run_tag", "Unknown")
    fig.suptitle(f"Sustained State Signature Analysis - {run_tag}", 
                fontsize=FONT_SIZE+2, fontweight="bold", y=0.995)
    
    # Save figure
    output_path_png = output_dir / "sustained_state_comprehensive_overview.png"
    output_path_pdf = output_dir / "sustained_state_comprehensive_overview.pdf"
    
    fig.savefig(output_path_png, dpi=DPI_PUBLICATION, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path_pdf, bbox_inches="tight", facecolor="white")
    
    print(f"Saved comprehensive figure:")
    print(f"  PNG: {output_path_png}")
    print(f"  PDF: {output_path_pdf}")
    
    plt.close(fig)


def create_focused_figures(config, results, roi_summary, output_dir):
    """Create individual focused figures for each aspect."""
    setup_figure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: ROI Summary
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_roi_summary(roi_summary, ax)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_roi_sustained_counts.pdf", dpi=DPI_PUBLICATION, bbox_inches="tight")
    fig.savefig(output_dir / "fig1_roi_sustained_counts.png", dpi=DPI_PUBLICATION, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: fig1_roi_sustained_counts.*")
    
    # Figure 2: Fraction and Binomial
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_fraction_sustained(roi_summary, ax1)
    plot_binomial_pvalues(roi_summary, ax2)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_fraction_and_binomial.pdf", dpi=DPI_PUBLICATION, bbox_inches="tight")
    fig.savefig(output_dir / "fig2_fraction_and_binomial.png", dpi=DPI_PUBLICATION, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: fig2_fraction_and_binomial.*")
    
    # Figure 3: Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_state_r_distribution(results, roi_summary, ax1)
    plot_phase_contrast_distribution(results, roi_summary, ax2)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_distributions.pdf", dpi=DPI_PUBLICATION, bbox_inches="tight")
    fig.savefig(output_dir / "fig3_distributions.png", dpi=DPI_PUBLICATION, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: fig3_distributions.*")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Path to encoding_state_sustained_cv run directory")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for figures (default: <run_dir>/figures)")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {run_dir}")
    config, results, roi_summary = load_results(run_dir)
    
    print(f"\nLoaded {len(results)} cells across {len(roi_summary)} ROIs")
    print("\nROI Summary:")
    print(roi_summary[["roi", "n_cells", "n_sustained_sig", "binom_p_sustained_sig"]].to_string(index=False))
    
    print(f"\nCreating visualizations...")
    create_comprehensive_figure(config, results, roi_summary, output_dir)
    create_focused_figures(config, results, roi_summary, output_dir)
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
