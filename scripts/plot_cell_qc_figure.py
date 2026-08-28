#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication figure for the single-unit quality control.

Reads the per-cell QC metrics recorded by the MATLAB QC run
(`derivatives/qc_all_sessions.mat`) and plots the three acceptance criteria
plus the resulting attrition. Nothing is recomputed here -- the metrics and
the pass/fail decisions are taken exactly as stored.

Usage:
    conda activate env_multiple_clocks
    python scripts/plot_cell_qc_figure.py

@author: Svenja Kuchenhoff
"""

import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io

# thresholds actually used by the run that produced abcd_passed.mat
MIN_SPIKES = 300
RPV_THRESH = 0.01
RPV_REFRAC_MS = 1.5
CORR_THRESH = 0.50
CORR_BIN_MS = 100

PASS_C = "#448363"      # era_brewer Showgirl2 green
FAIL_C = "#B74C2D"      # era_brewer Showgirl2 rust
GREY = "#9A9A9A"
OBS = "#0e3d3a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_qc(deriv):
    """Per-cell QC metrics and decisions, exactly as stored by the MATLAB run."""
    q = sio.loadmat(os.path.join(deriv, "qc_all_sessions.mat"),
                    squeeze_me=True, struct_as_record=False)["qc_all"]
    rows = []
    for si, s in enumerate(np.atleast_1d(q.sessions)):
        for c in np.atleast_1d(s.qc):
            fr = c.fail_reasons
            fr = [str(x) for x in np.atleast_1d(fr)] if fr is not None else []
            why = "pass"
            if not bool(c.is_reliable):
                why = ("few_spikes" if any("few spikes" in x for x in fr)
                       else "high_corr" if any("high-corr" in x for x in fr)
                       else "other")
            rows.append(dict(
                session=si, subject=str(s.subject_id),
                n_spikes=float(c.n_spikes), FR=float(c.overall_FR_Hz),
                RPV=float(c.RPV_frac),
                corr_max=float(c.corr_max) if np.isfinite(c.corr_max) else np.nan,
                passed=bool(c.is_reliable), why=why))
    return pd.DataFrame(rows), q.meta


def main():
    root = swr_io.get_data_root()
    deriv = swr_io.derivatives_dir(root)
    d, meta = load_qc(deriv)
    out_dir = os.path.join(deriv, "group", "cell_qc")
    os.makedirs(out_dir, exist_ok=True)

    npass = int(d.passed.sum())
    nfs = int((d.why == "few_spikes").sum())
    nhc = int((d.why == "high_corr").sum())

    # 17.4 cm wide, 4.6 cm tall panels -> a half-A4-width 4-panel strip
    fig, ax = plt.subplots(1, 4, figsize=(17.4 / 2.54, 4.8 / 2.54))

    # --- a. spike count -------------------------------------------------
    a = ax[0]
    bins = np.logspace(np.log10(max(d.n_spikes.min(), 1)),
                       np.log10(d.n_spikes.max()), 34)
    a.hist(d.loc[d.passed, "n_spikes"], bins=bins, color=PASS_C,
           alpha=.85, label=f"included ({npass})")
    a.hist(d.loc[~d.passed, "n_spikes"], bins=bins, color=FAIL_C,
           alpha=.85, label=f"excluded ({len(d)-npass})")
    a.axvline(MIN_SPIKES, color=OBS, ls="--", lw=1.2)
    a.set_xscale("log")
    a.xaxis.set_major_locator(LogLocator(numticks=4))
    a.set_xlabel("spikes in task window (n)")
    a.set_ylabel("cells (n)")
    a.set_title("a   spike count", loc="left", fontweight="bold")
    a.text(MIN_SPIKES * 1.15, a.get_ylim()[1] * .93, f"{MIN_SPIKES}",
           color=OBS, fontsize=8, va="top")

    # --- b. refractory period violations --------------------------------
    b = ax[1]
    v = np.clip(d.RPV * 100, 0, 2)
    bb = np.linspace(0, 2, 41)
    b.hist(v[d.passed], bins=bb, color=PASS_C, alpha=.85)
    b.hist(v[~d.passed], bins=bb, color=FAIL_C, alpha=.85)
    b.axvline(RPV_THRESH * 100, color=OBS, ls="--", lw=1.2)
    b.set_yscale("log")
    b.set_xlabel(f"ISI < {RPV_REFRAC_MS} ms (% of spikes)")
    b.set_ylabel("cells (n, log)")
    b.set_title("b   isolation", loc="left", fontweight="bold")
    b.set_xlim(0, 2)
    b.set_xticks([0, 1, 2])
    b.text(RPV_THRESH * 100 + .07, b.get_ylim()[1] * .30, "1%",
           color=OBS, fontsize=8)

    # --- c. within-bundle correlation -----------------------------------
    c = ax[2]
    cm = d.corr_max.dropna()
    cb = np.linspace(-.2, 1, 31)
    c.hist(d.loc[d.passed, "corr_max"].dropna(), bins=cb, color=PASS_C, alpha=.85)
    c.hist(d.loc[d.why == "high_corr", "corr_max"], bins=cb, color=FAIL_C, alpha=.95)
    c.axvline(CORR_THRESH, color=OBS, ls="--", lw=1.2)
    c.set_yscale("log")
    c.set_xlabel(f"max pairwise r ({CORR_BIN_MS} ms bins)")
    c.set_ylabel("cells (n, log)")
    c.set_title("c   duplicates", loc="left", fontweight="bold")
    c.text(CORR_THRESH + .03, c.get_ylim()[1] * .35, "0.50",
           color=OBS, fontsize=8)

    # --- d. attrition ---------------------------------------------------
    e = ax[3]
    e.bar([0, 1], [nfs, nhc], color=[FAIL_C, FAIL_C], width=.6,
          alpha=1.0, edgecolor="none")
    e.patches[1].set_alpha(.55)
    for i, (val, lab) in enumerate([(nfs, "< 300\nspikes"), (nhc, "within-bundle\nduplicate")]):
        e.text(i, val + 1.6, str(val), ha="center", fontsize=9, fontweight="bold",
               color=FAIL_C)
        e.text(i, -6.5, lab, ha="center", va="top", fontsize=7.5)
    e.set_xticks([])
    e.set_ylim(0, max(nfs, nhc) * 1.42)
    e.set_xlim(-.62, 1.62)
    e.set_ylabel("cells excluded (n)")
    e.set_title("d   why cells were dropped", loc="left", fontweight="bold")
    e.text(.5, max(nfs, nhc) * 1.30,
           f"1042 sorted \u2192 {npass} kept ({100*npass/len(d):.1f}%)",
           ha="center", fontsize=8, color=GREY)

    ax[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(-.02, -.34),
                 ncol=2, handlelength=1.1, columnspacing=1.0)
    fig.subplots_adjust(left=.055, right=.995, top=.84, bottom=.36, wspace=.50)

    stem = os.path.join(out_dir, "cell_qc_criteria")
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)

    d.to_csv(os.path.join(out_dir, "cell_qc_metrics.csv"), index=False)
    swr_io.write_settings(out_dir, {
        "analysis_name": "cell_qc_figure",
        "source": "derivatives/qc_all_sessions.mat",
        "qc_run_timestamp": str(meta.timestamp),
        "min_spikes": MIN_SPIKES, "rpv_refrac_ms": RPV_REFRAC_MS,
        "rpv_frac_thresh": RPV_THRESH, "corr_thresh": CORR_THRESH,
        "corr_bin_ms": CORR_BIN_MS,
        "n_sorted": int(len(d)), "n_passed": npass,
        "n_excluded_few_spikes": nfs, "n_excluded_duplicate": nhc,
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"cells {len(d)} -> passed {npass} "
          f"(few spikes {nfs}, duplicates {nhc})")
    print(f"saved -> {stem}.pdf/.png/.svg")


if __name__ == "__main__":
    main()
