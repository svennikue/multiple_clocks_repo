#!/usr/bin/env python
"""
Per-TR timecourses from an `svc_loso_batch.py` output folder.

Answers "when in the instruction period is each reward channel represented?"
by plotting, for every mask, the LOSO cross-validated held-out beta against
instruction-period second, with the reward-reveal schedule drawn along the top.

The LOSO curve is the right thing to plot here: voxels are chosen on n-1
subjects and read out in the held-out one, so the timecourse is an unbiased
effect-size estimate rather than the inflated one you get from selecting and
averaging on the same data. Values are plotted exactly as `svc_loso_batch`
computed them -- nothing is refitted, rescaled or smoothed here.

Reward schedule (from mc/latest_experiment/3x3_fMRI_part1.py, `show_rewards`):
only ONE reward is on screen at a time. A slow first pass at 1.5 s/reward
(A 0-1.5, B 1.5-3, C 3-4.5, D 4.5-6), then a 1 s/reward refresh
(A 6-7, B 7-8, C 8-9, D 9-12).

Usage
    python plot_per_TR_timecourses.py --in-dir <svc_loso_batch out-dir> \
        --models curr_rew,next_rew,two_next_rew,three_next_rew --k 100
"""
import argparse, json, os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})

# CLAUDE.md state palette — the four reward channels are the A/B/C/D rewards.
CHANNEL_COLOURS = {"curr_rew": "#F15A29", "next_rew": "#F7931E",
                   "two_next_rew": "#C7C6E2", "three_next_rew": "#6B60AA"}
CHANNEL_LABELS = {"curr_rew": "reward A (curr)", "next_rew": "reward B (next)",
                  "two_next_rew": "reward C (+2)", "three_next_rew": "reward D (+3)"}
REWARD_SCHEDULE = [(0.0, 1.5, "A", "#F15A29"), (1.5, 3.0, "B", "#F7931E"),
                   (3.0, 4.5, "C", "#C7C6E2"), (4.5, 6.0, "D", "#6B60AA"),
                   (6.0, 7.0, "A", "#F15A29"), (7.0, 8.0, "B", "#F7931E"),
                   (8.0, 9.0, "C", "#C7C6E2"), (9.0, 12.0, "D", "#6B60AA")]
ROI_DISPLAY_NAMES = {"mPFC": "mPFC", "MTL": "HC / EC", "visual": "occipital"}


def base_channel(model):
    """'CURR_REW-split_rew_DSR_combo' and 'curr_rew' both -> 'curr_rew'."""
    stem = model.split("-")[0]
    return stem.lower()


def draw_schedule(ax, x_max):
    """Reward-reveal strip just above the axes, in figure-independent units."""
    y0, y1 = ax.get_ylim()
    h = (y1 - y0) * 0.07
    for t0, t1, label, col in REWARD_SCHEDULE:
        if t0 >= x_max:
            continue
        t1 = min(t1, x_max)
        ax.add_patch(plt.Rectangle((t0 - 0.5, y1), t1 - t0, h, facecolor=col,
                                   edgecolor="white", lw=0.5, clip_on=False))
        ax.text((t0 + t1) / 2 - 0.5, y1 + h / 2, label, ha="center", va="center",
                fontsize=7, color="#333333", clip_on=False)
    ax.set_ylim(y0, y1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--models", required=True, help="comma-separated, in plot order")
    ap.add_argument("--masks", default="", help="default: every mask folder found")
    ap.add_argument("--k", default="100", help="LOSO voxel-selection size to plot")
    ap.add_argument("--out-name", default="per_TR_timecourses")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    masks = ([m for m in args.masks.split(",") if m] if args.masks else
             sorted(d for d in os.listdir(args.in_dir)
                    if os.path.isdir(os.path.join(args.in_dir, d)) and d != "wholebrain"))

    fig, axes = plt.subplots(1, len(masks), figsize=(3.0 * len(masks), 3.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    peak_rows = []
    for ax, mask in zip(axes, masks):
        for model in models:
            lo = json.load(open(os.path.join(args.in_dir, mask,
                                             f"{model}_loso_results.json")))
            key = args.k if args.k in lo else sorted(lo, key=lambda s: int(s))[0]
            rec = lo[key]
            mean = np.asarray(rec["mean"]); sem = np.asarray(rec["sem"])
            p = np.asarray(rec["p_FWE"]); x = np.arange(len(mean))
            col = CHANNEL_COLOURS.get(base_channel(model), "#666666")
            ax.plot(x, mean, "-o", color=col, ms=3, lw=1.4,
                    label=CHANNEL_LABELS.get(base_channel(model), model))
            ax.fill_between(x, mean - sem, mean + sem, color=col, alpha=0.18, lw=0)
            sig = p < 0.05
            if sig.any():
                ax.plot(x[sig], mean[sig], "o", color=col, ms=6.5,
                        mec="#0e3d3a", mew=1.0, zorder=5)
            peak_rows.append(dict(mask=mask, model=model, k=int(rec["k"]),
                                  peak_TR=int(np.argmax(np.asarray(rec["t"]))),
                                  peak_t=round(float(np.max(rec["t"])), 3),
                                  p_at_peak=float(p[int(np.argmax(rec["t"]))]),
                                  n_sig_TR=int(sig.sum())))
        ax.axhline(0, color="#999999", lw=0.8, ls="--")
        ax.set_title(ROI_DISPLAY_NAMES.get(mask, mask), pad=14)
        ax.set_xlabel("instruction period (s)")
        ax.set_xticks(range(0, 12, 2))
        draw_schedule(ax, x_max=12)
    axes[0].set_ylabel("held-out beta\n(LOSO cross-validated)")
    handles = [Line2D([], [], color=CHANNEL_COLOURS[c], marker="o", ms=3, lw=1.4,
                      label=CHANNEL_LABELS[c]) for c in CHANNEL_COLOURS]
    handles.append(Line2D([], [], color="none", marker="o", ms=6.5, mec="#0e3d3a",
                          mew=1.0, label="p$_{FWE}$ < .05"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Reward-channel representation across the instruction period",
                 y=1.13, fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "jpeg"):
        fig.savefig(os.path.join(args.in_dir, f"{args.out_name}.{ext}"),
                    dpi=300, bbox_inches="tight")

    import csv
    with open(os.path.join(args.in_dir, f"{args.out_name}_peaks.csv"), "w",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(peak_rows[0].keys()))
        w.writeheader(); w.writerows(peak_rows)
    print(f"-> {args.in_dir}/{args.out_name}.pdf  (+ .jpeg, _peaks.csv)")
    for r in peak_rows:
        print(f"  {r['mask']:7s} {r['model']:32s} peak TR{r['peak_TR']:<2d} "
              f"t={r['peak_t']:5.2f} p={r['p_at_peak']:.4f}  "
              f"{r['n_sig_TR']} TRs p<.05")
    plt.show(block=False)


if __name__ == "__main__":
    main()
