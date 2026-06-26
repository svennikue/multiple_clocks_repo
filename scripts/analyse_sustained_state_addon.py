#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add-on analyses for encoding_state_sustained_cv results.

Investigates:
1. Overlap: Are the sustained cells the same ones with positive phase contrasts?
2. ROI differences: Are min_phase_contrast distributions significantly different?

Usage:
    python scripts/analyse_sustained_state_addon.py --run-dir <path>

"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")


def load_results(run_dir):
    """Load results from sustained state encoding run."""
    run_dir = Path(run_dir)
    results = pd.read_csv(run_dir / "state_sustained_cv_results.csv")
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    return config, results


def parse_phase_contrasts(row):
    """Extract phase contrasts from JSON column."""
    import json
    if pd.isna(row["phase_contrast_json"]):
        return None
    try:
        d = json.loads(row["phase_contrast_json"])
        return np.array([d.get("early"), d.get("middle"), d.get("late")])
    except:
        return None


def addon_sustained_and_positive_overlap(results):
    """
    Check 1: Are the sustained cells the ones with consistently positive contrasts?
    
    Compares sustained_sig flag with actual observed phase contrasts.
    """
    print("\n" + "="*70)
    print("ADDON 1: Sustained Cells vs. Positive Contrasts Overlap")
    print("="*70)
    
    # Parse phase contrasts for all cells
    results["phase_contrasts"] = results.apply(parse_phase_contrasts, axis=1)
    
    # Determine if cell has all positive contrasts (from data, not from flag)
    def all_contrasts_positive(contrasts):
        if contrasts is None:
            return False
        return np.all(np.isfinite(contrasts)) and np.all(contrasts > 0)
    
    results["has_all_positive_contrasts"] = results["phase_contrasts"].apply(all_contrasts_positive)
    
    # Cross-tabulation
    crosstab = pd.crosstab(
        results["sustained_sig"],
        results["has_all_positive_contrasts"],
        margins=True,
        rownames=["sustained_sig"],
        colnames=["all_positive"]
    )
    
    print("\nCrosstabulation: Sustained Sig vs All Positive Contrasts")
    print(crosstab)
    
    # Overall stats
    n_sustained = results["sustained_sig"].sum()
    n_all_positive = results["has_all_positive_contrasts"].sum()
    n_both = (results["sustained_sig"] & results["has_all_positive_contrasts"]).sum()
    
    print(f"\nSummary:")
    print(f"  Cells marked as sustained_sig:           {n_sustained}")
    print(f"  Cells with all positive contrasts:       {n_all_positive}")
    print(f"  Cells with BOTH criteria:                {n_both}")
    
    if n_sustained > 0:
        overlap_frac = n_both / n_sustained
        print(f"  Fraction of sustained with all positive: {overlap_frac:.3f}")
    
    # Per-ROI breakdown
    print("\n" + "-"*70)
    print("Per-ROI Breakdown:")
    print("-"*70)
    
    roi_overlap = []
    for roi in sorted(results["roi"].unique()):
        roi_data = results[results["roi"] == roi]
        n_sust_roi = roi_data["sustained_sig"].sum()
        n_pos_roi = roi_data["has_all_positive_contrasts"].sum()
        n_both_roi = (roi_data["sustained_sig"] & roi_data["has_all_positive_contrasts"]).sum()
        frac_roi = n_both_roi / n_sust_roi if n_sust_roi > 0 else np.nan
        
        roi_overlap.append({
            "roi": roi,
            "n_sustained": int(n_sust_roi),
            "n_all_positive": int(n_pos_roi),
            "n_both": int(n_both_roi),
            "frac_sustained_with_all_positive": float(frac_roi) if np.isfinite(frac_roi) else np.nan,
        })
        
        print(f"\n{roi}:")
        print(f"  Sustained:               {n_sust_roi}")
        print(f"  All positive contrasts:  {n_pos_roi}")
        print(f"  Both criteria:           {n_both_roi}")
        print(f"  Frac sustained w/ all +: {frac_roi:.3f}")
    
    overlap_df = pd.DataFrame(roi_overlap)
    return overlap_df


def addon_roi_contrast_differences(results):
    """
    Check 2: Are min_phase_contrast distributions significantly different between ROIs?
    
    Tests whether the "cleanness" of state encoding (proximity to 0) differs by ROI.
    """
    print("\n" + "="*70)
    print("ADDON 2: ROI Differences in Min Phase Contrast")
    print("="*70)
    
    print("\nQuestion: Do ROIs differ in how close to zero their min contrasts are?")
    print("  - Closer to 0 = more consistent state encoding (less drop in any phase)")
    print("  - More negative = some phases lose the state signal")
    
    # Gather data per ROI
    roi_data = []
    rois = sorted(results["roi"].unique())
    
    print("\n" + "-"*70)
    print("Descriptive Statistics:")
    print("-"*70)
    
    for roi in rois:
        roi_subset = results[results["roi"] == roi]["min_phase_contrast"].dropna()
        roi_subset = roi_subset[np.isfinite(roi_subset)]
        
        if len(roi_subset) > 0:
            print(f"\n{roi} (n={len(roi_subset)}):")
            print(f"  Mean:       {roi_subset.mean():+.6f}")
            print(f"  Median:     {roi_subset.median():+.6f}")
            print(f"  Std:        {roi_subset.std():.6f}")
            print(f"  Min:        {roi_subset.min():+.6f}")
            print(f"  Max:        {roi_subset.max():+.6f}")
            print(f"  % close to 0 (abs < 0.05): {(np.abs(roi_subset) < 0.05).sum() / len(roi_subset) * 100:.1f}%")
            
            roi_data.append({
                "roi": roi,
                "n": len(roi_subset),
                "mean": float(roi_subset.mean()),
                "median": float(roi_subset.median()),
                "std": float(roi_subset.std()),
                "min": float(roi_subset.min()),
                "max": float(roi_subset.max()),
            })
    
    # Statistical tests
    print("\n" + "-"*70)
    print("Statistical Tests:")
    print("-"*70)
    
    # Prepare data for ANOVA/Kruskal-Wallis
    roi_groups = [results[results["roi"] == roi]["min_phase_contrast"].dropna().values 
                  for roi in rois]
    roi_groups = [g[np.isfinite(g)] for g in roi_groups]  # Remove NaNs
    
    # ANOVA (parametric)
    if len(roi_groups) > 1:
        f_stat, p_anova = stats.f_oneway(*roi_groups)
        print(f"\nOne-way ANOVA:")
        print(f"  F = {f_stat:.4f}, p = {p_anova:.6f}")
        if p_anova < 0.05:
            print(f"  *** Significant difference between ROIs (p < 0.05)")
        else:
            print(f"  No significant difference (p >= 0.05)")
    
    # Kruskal-Wallis (non-parametric, robust to outliers)
    if len(roi_groups) > 1:
        h_stat, p_kw = stats.kruskal(*roi_groups)
        print(f"\nKruskal-Wallis test (non-parametric):")
        print(f"  H = {h_stat:.4f}, p = {p_kw:.6f}")
        if p_kw < 0.05:
            print(f"  *** Significant difference between ROIs (p < 0.05)")
        else:
            print(f"  No significant difference (p >= 0.05)")
    
    # Pairwise comparisons (post-hoc)
    if len(roi_groups) > 1 and p_kw < 0.05:
        print("\n" + "-"*70)
        print("Pairwise Comparisons (Mann-Whitney U tests with Bonferroni correction):")
        print("-"*70)
        
        n_pairs = len(rois) * (len(rois) - 1) // 2
        bonf_alpha = 0.05 / n_pairs
        
        pair_list = []
        for i, roi1 in enumerate(rois):
            for roi2 in rois[i+1:]:
                group1 = results[results["roi"] == roi1]["min_phase_contrast"].dropna().values
                group2 = results[results["roi"] == roi2]["min_phase_contrast"].dropna().values
                group1 = group1[np.isfinite(group1)]
                group2 = group2[np.isfinite(group2)]
                
                if len(group1) > 0 and len(group2) > 0:
                    u_stat, p_mw = stats.mannwhitneyu(group1, group2, alternative="two-sided")
                    sig = "**" if p_mw < bonf_alpha else ""
                    print(f"\n{roi1} vs {roi2}:")
                    print(f"  Mean diff: {group1.mean() - group2.mean():+.6f}")
                    print(f"  p = {p_mw:.6f} {sig} (Bonf α={bonf_alpha:.6f})")
                    
                    pair_list.append({
                        "roi1": roi1,
                        "roi2": roi2,
                        "mean_diff": float(group1.mean() - group2.mean()),
                        "p_value": float(p_mw),
                        "significant_bonf": p_mw < bonf_alpha,
                    })
    
    contrast_df = pd.DataFrame(roi_data)
    return contrast_df


def addon_roi_sustained_fraction_differences(results):
    """
    Bonus: Do ROIs differ in their fraction of sustained cells?
    """
    print("\n" + "="*70)
    print("BONUS: ROI Differences in Sustained Signature Fraction")
    print("="*70)
    
    roi_fracs = []
    for roi in sorted(results["roi"].unique()):
        roi_subset = results[results["roi"] == roi]
        n_cells = len(roi_subset)
        n_sustained = roi_subset["sustained_sig"].sum()
        frac = n_sustained / n_cells if n_cells > 0 else 0
        
        roi_fracs.append({
            "roi": roi,
            "n_cells": int(n_cells),
            "n_sustained": int(n_sustained),
            "frac_sustained": float(frac),
        })
    
    frac_df = pd.DataFrame(roi_fracs)
    print("\nSustained Signature Fraction by ROI:")
    print(frac_df.to_string(index=False))
    
    return frac_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Path to encoding_state_sustained_cv run directory")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for results (default: <run_dir>/addon_analysis)")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "addon_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {run_dir}")
    config, results = load_results(run_dir)
    print(f"Loaded {len(results)} cells")
    
    # Run add-ons
    overlap_df = addon_sustained_and_positive_overlap(results)
    contrast_df = addon_roi_contrast_differences(results)
    frac_df = addon_roi_sustained_fraction_differences(results)
    
    # Save results
    overlap_path = output_dir / "sustained_positive_overlap.csv"
    overlap_df.to_csv(overlap_path, index=False)
    print(f"\n\nSaved overlap analysis to: {overlap_path}")
    
    contrast_path = output_dir / "roi_contrast_statistics.csv"
    contrast_df.to_csv(contrast_path, index=False)
    print(f"Saved contrast statistics to: {contrast_path}")
    
    frac_path = output_dir / "roi_sustained_fractions.csv"
    frac_df.to_csv(frac_path, index=False)
    print(f"Saved fraction analysis to: {frac_path}")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
