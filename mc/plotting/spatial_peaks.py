"""Plots for the future-spatial-peaks pipeline.

All functions take a per-cell DataFrame produced by
``mc.analyse.future_spatial_peaks`` and described in
``scripts/spatial_peaks_simple.py``.  Required columns:

    neuron_id, roi, peak_r, peak_shift_plurality, n_grids_used

Optional columns:

    p_perm  (only used by the binomial plot when run_permutations=True)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import scipy.stats as st
from scipy.stats import binomtest, norm


# ── small helpers ─────────────────────────────────────────────────────

def stars(p):
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "n.s."


def one_tailed_ttest_greater_than_zero(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    t_stat, p_two = st.ttest_1samp(x, 0.0, nan_policy="omit")
    p_one = p_two / 2 if t_stat > 0 else 1 - (p_two / 2)
    return float(t_stat), float(p_one), float(np.mean(x))


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    m = int(np.sum(~np.isnan(p)))
    adj = np.full_like(p, np.nan, dtype=float)
    if m == 0:
        return adj
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    ranks = np.arange(1, len(p) + 1, dtype=float)
    p_ord = p[order]
    with np.errstate(invalid="ignore"):
        running = np.minimum.accumulate((m / ranks[order]) * p_ord[::-1])[::-1]
    adj[order] = np.clip(running, 0, 1)
    return adj


def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - (1 - conf) / 2)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return center - half, center + half


# ── ROI relabelling ───────────────────────────────────────────────────

def rename_rois_from_neuron_id(df, collapse_pfc=False,
                               plot_by_cingulate_and_MTL=False):
    """Map raw electrode-label ROI strings to canonical bins. Used when the
    DataFrame has no precomputed 'roi' column (falls back to neuron_id parsing
    matching the legacy ``rename_rois`` in the wrapper). New pipelines should
    pass the canonical roi straight through.
    """
    roi_label = []
    for _, row in df.iterrows():
        cell_label = row["neuron_id"]
        if collapse_pfc:
            if any(k in cell_label for k in ("ACC", "vCC", "AMC", "vmPFC", "OFC", "PCC")):
                roi = "PFC"
            elif any(k in cell_label for k in ("MCC", "HC")):
                roi = "hippocampal"
            elif "EC" in cell_label:
                roi = "entorhinal"
            elif "AMYG" in cell_label:
                roi = "amygdala"
            else:
                roi = "mixed"
        elif plot_by_cingulate_and_MTL:
            if any(k in cell_label for k in ("ACC", "vCC", "AMC", "vmPFC", "PCC")):
                roi = "Cingulate"
            elif any(k in cell_label for k in ("MCC", "HC", "EC", "AMYG")):
                roi = "MTL"
            elif "OFC" in cell_label:
                roi = "OFC"
            else:
                roi = "mixed"
        else:
            if any(k in cell_label for k in ("ACC", "vCC", "AMC", "vmPFC")):
                roi = "ACC"
            elif "PCC" in cell_label:
                roi = "PCC"
            elif "OFC" in cell_label:
                roi = "OFC"
            elif "MCC" in cell_label or "HC" in cell_label:
                roi = "hippocampal"
            elif "EC" in cell_label:
                roi = "entorhinal"
            elif "AMYG" in cell_label:
                roi = "amygdala"
            else:
                roi = "mixed"
        roi_label.append(roi)
    return roi_label


# ── plots ─────────────────────────────────────────────────────────────

def plot_peak_r_histograms_per_roi(df, now_set=(330, 0, 30),
                                   title="Cross-validated spatial consistency",
                                   bins=20, save_path=None):
    """Three rows (now/future/all) × n_rois of peak_r histograms with one-
    sample t-test annotations."""
    rois = sorted(df["roi"].dropna().unique().tolist())
    if not rois:
        print("No ROIs to plot."); return None
    now_set = set(int(s) for s in now_set)

    rows = [
        (f"now ({sorted(now_set)})", df[df["peak_shift_plurality"].isin(now_set)]),
        ("future lags",              df[~df["peak_shift_plurality"].isin(now_set)]),
        ("all lags",                 df),
    ]
    n_rows, n_cols = len(rows), len(rois)

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(max(8.3, 3 * n_cols), 11.0),
                             sharex=True, sharey=False,
                             gridspec_kw={"hspace": 0.18, "wspace": 0.3})
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    cache = [[None] * n_cols for _ in range(n_rows)]
    row_max_top, row_max_all = 0, 0
    for r, (rname, df_r) in enumerate(rows):
        for c, roi in enumerate(rois):
            vals = df_r.loc[df_r["roi"] == roi, "peak_r"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            counts, _ = np.histogram(vals, bins=bins)
            cache[r][c] = (vals, counts)
            mx = counts.max() if counts.size else 0
            if rname.startswith("all"):
                row_max_all = max(row_max_all, mx)
            else:
                row_max_top = max(row_max_top, mx)

    for r, (rname, _df_r) in enumerate(rows):
        for c, roi in enumerate(rois):
            ax = axes[r, c]
            vals, _ = cache[r][c]
            ax.hist(vals, bins=bins, color="teal", alpha=0.5, edgecolor="teal")
            ax.axvline(0, color="black", linestyle="dashed", linewidth=1.5)
            t_stat, p_one, mval = one_tailed_ttest_greater_than_zero(vals)
            sig = stars(p_one)
            ax.text(0.98, 0.96,
                    f"n={vals.size}\nmean={mval:.2f}\n{sig} (p={p_one:.1e})",
                    transform=ax.transAxes, ha="right", va="top",
                    bbox=dict(facecolor="white", edgecolor="black",
                              boxstyle="round"),
                    fontsize=10)
            if r == 0:
                ax.set_title(roi, fontsize=11, pad=4)
            if c == 0:
                ax.text(-0.16, 0.5, rname, transform=ax.transAxes,
                        ha="right", va="center", rotation=90, fontsize=11)
            ylim = row_max_all if rname.startswith("all") else row_max_top
            ax.set_ylim(0, max(1, int(ylim * 1.04)))
            ax.tick_params(axis="both", labelsize=10, width=1.1, length=4)

    fig.supxlabel("Cross-validated peak r", fontsize=11)
    fig.supylabel("Cell count", fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.91,
                        hspace=0.18, wspace=0.3)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.965)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_peak_shift_stack_per_roi(df, now_set=(330, 0, 30),
                                  title="Peak-shift distribution per ROI",
                                  bins=20, save_path=None):
    """One subplot per ROI; stacked histogram of peak_r coloured by
    now/future lag bucket."""
    rois = sorted(df["roi"].dropna().unique().tolist())
    if not rois:
        print("No ROIs to plot."); return None
    now_set = set(int(s) for s in now_set)
    mask_now = df["peak_shift_plurality"].isin(now_set)
    mask_future = (~mask_now) & df["peak_shift_plurality"].notna()

    all_vals = df.loc[mask_now | mask_future, "peak_r"].dropna().to_numpy()
    if all_vals.size == 0:
        print("No data to plot."); return None
    edges = np.histogram_bin_edges(all_vals, bins=bins)

    n_roi = len(rois)
    fig, axes = plt.subplots(1, n_roi, figsize=(max(5, n_roi * 4.5), 5),
                             sharey=True)
    if n_roi == 1:
        axes = [axes]
    colors = {"now": "lightcoral", "future": "teal"}
    legend = [
        Patch(facecolor=colors["now"],    edgecolor="black", label="now lag"),
        Patch(facecolor=colors["future"], edgecolor="black", label="future lags"),
    ]

    for ax, roi in zip(axes, rois):
        sub = df[df["roi"] == roi]
        x_now    = sub.loc[mask_now.loc[sub.index],    "peak_r"].dropna().to_numpy()
        x_future = sub.loc[mask_future.loc[sub.index], "peak_r"].dropna().to_numpy()
        ax.hist([x_now, x_future], bins=edges, stacked=True,
                color=[colors["now"], colors["future"]],
                edgecolor="black", alpha=0.9)
        ax.axvline(0, color="black", linestyle="dashed", linewidth=2)
        ax.set_title(f"{roi}\n(n={len(x_now) + len(x_future)})", fontsize=12)
        ax.set_xlabel("peak r", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Cell count", fontsize=12)
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_binomial_per_roi(df, title, p_col="p_perm", alpha=0.05,
                          use_fdr=True, save_path=None):
    """Fraction of cells with p < alpha per ROI, with Wilson CIs and a
    binomial test against the null expectation alpha."""
    if p_col not in df.columns:
        print(f"[binomial plot] column {p_col} not in dataframe; skipping.")
        return None
    p_emp = df[p_col].to_numpy(float)
    regions = df["roi"].fillna("unassigned").to_numpy()

    rows = []
    for r in np.unique(regions):
        m = (regions == r)
        x = p_emp[m]; x = x[np.isfinite(x)]
        n = int(x.size); k = int((x <= alpha).sum())
        p_b = binomtest(k, n, p=alpha, alternative="greater").pvalue if n else np.nan
        lo, hi = wilson_ci(k, n) if n else (np.nan, np.nan)
        frac = (k / n) if n else np.nan
        rows.append((r, n, k, frac, lo, hi, p_b))
    if not rows:
        print("No regions to plot."); return None
    rows.sort(key=lambda t: (-t[1], t[0]))
    labels, ns, ks, fracs, los, his, ps = map(list, zip(*rows))
    ps = np.array(ps, float)
    p_adj = bh_fdr(ps) if use_fdr else np.full_like(ps, np.nan)

    fracs_arr = np.array(fracs, float)
    los_arr   = np.array(los,   float)
    his_arr   = np.array(his,   float)
    yerr_low  = np.nan_to_num(fracs_arr - los_arr, nan=0.0)
    yerr_high = np.nan_to_num(his_arr   - fracs_arr, nan=0.0)
    yerr = np.vstack([yerr_low, yerr_high])

    data_top = float(np.nanmax(fracs_arr + yerr_high)) \
        if np.isfinite(fracs_arr + yerr_high).any() else 0.1
    y_max = min(1.0, max(0.15, data_top + 0.07))

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    x = np.arange(len(labels))
    cmap = plt.get_cmap("tab20", max(3, len(labels)))
    colors = cmap(range(len(labels)))
    ax.bar(x, fracs_arr, width=0.65, color=colors, edgecolor="none", alpha=0.95)
    ax.errorbar(x, fracs_arr, yerr=yerr, fmt="none", capsize=4, lw=1.6,
                color="black", zorder=3)
    ax.axhline(alpha, color="gray", lw=1.2, ls="--",
               label=f"Expected under null (α={alpha:g})")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Fraction significant")
    ax.set_title(title + (" (stars = BH-FDR q)" if use_fdr else ""))
    p_for_stars = p_adj if use_fdr else ps
    for i, (k, n, p_raw, q) in enumerate(zip(ks, ns, ps, p_adj)):
        top_i = fracs_arr[i] + yerr_high[i]
        y_i = min(y_max - 0.01, top_i + 0.03)
        if use_fdr:
            txt = f"{k}/{n}\n{stars(p_for_stars[i])} (q={q:.2g}, p={p_raw:.2g})"
        else:
            txt = f"{k}/{n}\n{stars(p_for_stars[i])} (p={p_raw:.2g})"
        ax.text(x[i], y_i, txt, ha="center", va="bottom", fontsize=11, zorder=4)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return pd.DataFrame({
        "roi": labels, "n": ns, "k": ks, "frac": fracs_arr,
        "ci_lo": los_arr, "ci_hi": his_arr,
        "p_raw": ps, "p_adj": p_adj,
    })


def plot_subset_binomial_per_roi(subset_df, alpha=0.05, save_path=None):
    """Side-by-side per-ROI binomial: all / state_NS / DSR_sig.

    Expects a DataFrame with columns: subset, roi, n, k, frac, p_binom,
    q_BH_family, sig_FDR (as produced by spatial_peaks_simple._run_subset_binomials).
    """
    if subset_df.empty:
        print("[subset binomial plot] empty input"); return None
    subsets = ["all", "state_NS", "DSR_sig"]
    rois = sorted(subset_df["roi"].unique().tolist())
    rois = [r for r in ["ACC", "medialOFC", "PCC", "Parahippocampal",
                        "HC_anterior", "HC_mid", "EC"] if r in rois] + \
           [r for r in rois if r not in {"ACC", "medialOFC", "PCC",
                                          "Parahippocampal", "HC_anterior",
                                          "HC_mid", "EC"}]

    x = np.arange(len(rois)); w = 0.27
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(max(11, 1.3 * len(rois)), 6))
    colors = {"all": "#5c5c5c", "state_NS": "#1f77b4", "DSR_sig": "#d62728"}

    for i, sub in enumerate(subsets):
        d = subset_df[subset_df["subset"] == sub].set_index("roi")
        fracs = [d["frac"].get(r, np.nan) for r in rois]
        cis_lo, cis_hi = [], []
        for r in rois:
            if r in d.index:
                lo, hi = wilson_ci(int(d.loc[r, "k"]), int(d.loc[r, "n"]))
                cis_lo.append(lo); cis_hi.append(hi)
            else:
                cis_lo.append(np.nan); cis_hi.append(np.nan)
        cis_lo = np.array(cis_lo); cis_hi = np.array(cis_hi)
        fracs = np.array(fracs)
        # clip to handle Wilson-CI edge cases where rounding makes lo>frac slightly
        yerr_lo = np.clip(np.nan_to_num(fracs - cis_lo, nan=0.0), 0, None)
        yerr_hi = np.clip(np.nan_to_num(cis_hi - fracs, nan=0.0), 0, None)
        ax.bar(x + (i - 1) * w, fracs, w, yerr=np.vstack([yerr_lo, yerr_hi]),
               label=sub, capsize=2, color=colors[sub], alpha=0.9)
        # annotate sig stars from q_BH_family
        for j, r in enumerate(rois):
            if r in d.index:
                q = d.loc[r, "q_BH_family"]
                k = int(d.loc[r, "k"]); n = int(d.loc[r, "n"])
                yi = (fracs[j] if np.isfinite(fracs[j]) else 0) + yerr_hi[j] + 0.01
                txt = f"{k}/{n}\n{stars(q)}"
                ax.text(x[j] + (i - 1) * w, yi, txt, ha="center", va="bottom",
                        fontsize=8)
    ax.axhline(alpha, color="gray", lw=1.1, ls="--",
               label=f"null α={alpha:g}")
    ax.set_xticks(x); ax.set_xticklabels(rois, rotation=15)
    ax.set_ylabel("Fraction of cells with p_perm < α")
    ax.set_title("Per-ROI binomial test by cell subset (stars = BH-FDR q across family)",
                 fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_all(df, out_dir, cfg):
    """Run the standard plot set. Saves into out_dir/figs/ ."""
    out_dir = Path(out_dir)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    label_extra = f"{cfg['trials']} trials | coverage={cfg['coverage_mode']} | n_peaks={cfg['n_peaks']}"

    plot_peak_r_histograms_per_roi(
        df, now_set=cfg["now_set"],
        title=f"Cross-validated peak r per ROI\n{label_extra}",
        save_path=figs_dir / "peak_r_histograms_per_roi.png",
    )
    plt.show(block=False)

    plot_peak_shift_stack_per_roi(
        df, now_set=cfg["now_set"],
        title=f"Peak r split by now/future lag\n{label_extra}",
        save_path=figs_dir / "peak_shift_stack_per_roi.png",
    )
    plt.show(block=False)

    if "p_perm" in df.columns and df["p_perm"].notna().any():
        plot_binomial_per_roi(
            df, title=f"Binomial per ROI (permutation p)\n{label_extra}",
            save_path=figs_dir / "binomial_per_roi.png",
        )
        plt.show(block=False)

    # subset binomial (all / state_NS / DSR_sig)
    subset_csv = out_dir / "subset_binomial.csv"
    if subset_csv.is_file():
        try:
            subset_df = pd.read_csv(subset_csv)
            plot_subset_binomial_per_roi(
                subset_df,
                save_path=figs_dir / "subset_binomial_per_roi.png",
            )
            plt.show(block=False)
        except Exception as exc:
            print(f"  subset binomial plot failed: {exc!r}")
