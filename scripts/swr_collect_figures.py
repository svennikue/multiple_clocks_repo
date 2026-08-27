#!/usr/bin/env python3
"""Collect per-session SWR QC figures into final_results/ripple_analysis/figures
so they can be reviewed in one place alongside methods.md."""
import os, sys, shutil, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io

FINAL = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/final_results/ripple_analysis/figures"

def collect(analysis_name="swr_v1"):
    swr_io.start_log(FINAL, "swr_collect_figures")
    R = swr_io.get_data_root(); os.makedirs(FINAL, exist_ok=True)
    n = 0
    for src, tag in ((f"LFP-clean/{analysis_name}/qc_psd.png", "psd"),
                     (f"LFP-ripples/{analysis_name}/qc_ripples.png", "ripples"),
                     (f"LFP-ripples/{analysis_name}/qc_ripples.pdf", "ripples")):
        for p in sorted(glob.glob(os.path.join(swr_io.derivatives_dir(R), "s*", src))):
            sess = p.split(os.sep)[-4]
            ext = os.path.splitext(p)[1]
            shutil.copy2(p, os.path.join(FINAL, f"{sess}_{tag}{ext}"))
            n += 1
    print(f"collected {n} figures -> {FINAL}")

if __name__ == "__main__":
    collect()
