#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication figure: ripple rate by state x discovery.

Tests SK's hypothesis against its obvious alternative on one 4x2 panel:
  hypothesis  -> ripples elevated specifically at the FIRST uncovering of D,
                 when the route first becomes knowable (an interaction)
  alternative -> D elevated whenever reached (a main effect of state)

Rates are computed per derivation and averaged, so each derivation contributes
equally; pooling events would let long or ripple-rich derivations dominate.
Error bars are SEM across derivations.

Sized for a ~6 cm journal subpanel: Arial, 9 pt labels, 10 pt title.
State colours follow the project convention (CLAUDE.md).

Usage:
    python scripts/swr_plot_discovery.py --suffix=_1s
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io

try:
    import fire
except ImportError:
    fire = None

# Project-wide state colours (CLAUDE.md): orange -> purple
STATE_COLORS = {"A": "#F15A29", "B": "#F7931E", "C": "#C7C6E2", "D": "#6B60AA"}
FINAL = ("/Users/xpsy1114/Documents/projects/multiple_clocks/data/"
         "final_results/ripple_analysis/figures")
CM = 1 / 2.54

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,     # editable text in vector output
})


def plot(suffix="_1s", lock_s=1.0, save=True):
    swr_io.start_log(FINAL, "swr_plot_discovery")
    R = swr_io.get_data_root()
    p = os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                     f"window_counts_discovery{suffix}.csv")
    d = pd.read_csv(p)

    # rate per derivation per cell, then average across derivations
    per = (d.groupby(["session", "pair_id", "state", "discovery"])
             .agg(n=("n_ripples", "sum"), exp=("exposure_s", "sum"))
             .reset_index())
    per = per[per.exp > 0]
    per["rate"] = per.n / per.exp

    states = ["A", "B", "C", "D"]
    fig, ax = plt.subplots(figsize=(6.2 * CM, 5.0 * CM))
    width = 0.36
    x = np.arange(len(states))

    for k, disc in enumerate(["first", "later"]):
        m, e = [], []
        for st in states:
            v = per[(per.state == st) & (per.discovery == disc)].rate.to_numpy()
            m.append(np.mean(v) if len(v) else np.nan)
            e.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan)
        cols = [STATE_COLORS[s] for s in states]
        ax.bar(x + (k - 0.5) * width, m, width * 0.92,
               yerr=e, capsize=1.8,
               color=cols if disc == "first" else "white",
               edgecolor=cols, linewidth=1.1,
               error_kw=dict(lw=0.8, ecolor="0.35"),
               label="first" if disc == "first" else "later")

    ax.set_xticks(x)
    ax.set_xticklabels(states)
    ax.set_xlabel("Reward uncovered", labelpad=2)
    ax.set_ylabel("Ripple rate (Hz)")
    ax.set_title(f"First vs later uncovering\n({lock_s:.0f} s window)", pad=6)

    # legend as fill/outline rather than colour, since colour encodes state
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="0.45", edgecolor="0.45", label="first"),
                       Patch(facecolor="white", edgecolor="0.45", label="later")],
              frameon=False, loc="upper left", handlelength=1.1,
              borderpad=0.2, labelspacing=0.25)

    n_der = per.pair_id.nunique()
    n_sess = per.session.nunique()
    # annotation goes BELOW the axis so it can never obscure the data
    ax.annotate(f"{n_der} derivations, {n_sess} sessions",
                xy=(0.5, -0.30), xycoords="axes fraction",
                ha="center", va="top", fontsize=6.5, color="0.45")

    fig.tight_layout(pad=0.3)
    if save:
        os.makedirs(FINAL, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FINAL, f"discovery_state_x_first{suffix}.{ext}"),
                        dpi=300, bbox_inches="tight")
        print(f"saved -> {FINAL}/discovery_state_x_first{suffix}.pdf/.png")
    plt.close(fig)

    print("\nmean rate across derivations (Hz):")
    tab = (per.pivot_table(index="state", columns="discovery",
                           values="rate", aggfunc="mean").round(3))
    tab["first - later"] = (tab["first"] - tab["later"]).round(3)
    print(tab.to_string())
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(plot)
    else:
        plot()
