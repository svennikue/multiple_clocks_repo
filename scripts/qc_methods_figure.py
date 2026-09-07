"""Methods figures for single-unit quality control and ROI assignment.

Two publication panels, each 18 cm wide x 4 cm high, Arial, min font 9 pt
(11 pt bold panel letters):

  Figure 1 -- Single-unit quality control
      a  spike count per unit, 300-spike inclusion threshold
      b  firing rate of included vs excluded units (how little the
         rejected units actually fire)
      c  pooled inter-spike-interval histogram with the 1.5 ms
         refractory window, plus the per-unit RPV summary
      d  maximum pairwise correlation per unit, r = 0.50 dedup threshold

  Figure 2 -- From recorded to analysed units
      a  unit yield: composition of all sorted units
      b  hippocampal MNI-y distribution, anterior/mid split at y = -21 mm
      c  units per ROI, with number of contributing sessions

Inputs (never modified):
    derivatives/qc_all_sessions_rebuild.mat   -- output of call_cell_wise_QC.m
    derivatives/neurons_with_ROI_labels.csv   -- output of cell_to_roi_july26.py
    abcd_data_08-Sep-2025.mat                 -- raw spike times (ISI panel only)

Outputs -> derivatives/QC_methods_figure_<date>/
"""

import json
import os
from datetime import date

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

from mc.plotting.cell_results import get_roi_colour, roi_display

# ----------------------------------------------------------------- config
SOURCE_DIR = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
DERIV_DIR = os.path.join(SOURCE_DIR, "derivatives")
QC_MAT = os.path.join(DERIV_DIR, "qc_all_sessions_rebuild.mat")
ROI_CSV = os.path.join(DERIV_DIR, "neurons_with_ROI_labels.csv")
RAW_MAT = os.path.join(SOURCE_DIR, "abcd_data_08-Sep-2025.mat")

OUT_DIR = os.path.join(DERIV_DIR, f"QC_methods_figure_{date.today().isoformat()}")
ISI_CACHE = os.path.join(OUT_DIR, "pooled_isi_histogram.npz")

# QC settings as used in call_cell_wise_QC.m -- kept here only for the
# threshold lines, never re-derived.
MIN_SPIKES = 300
REFRAC_MS = 1.5
RPV_THRESH = 0.01
CORR_THRESH = 0.50
ISI_XMAX_MS = 20.0           # x-limit of the pooled ISI panel
HC_ANT_MID_Y = -21.0          # Poppenk & Moscovitch 2013

# ROIs analysed in the paper (>= 3 sessions of coverage).
ANALYSED_ROIS = ["EC", "mPFC", "mOFC", "HC_anterior", "HC_mid", "PCC"]

# Colours (project convention, see CLAUDE.md).
C_KEEP = "#5b9b8d"            # included units      (grid teal-green)
C_DROP = "#D7657F"            # excluded units      (phase rose)
C_THRESH = "#5C1027"          # threshold lines     (bordeaux)
C_OTHER = "#bdbdbd"           # not-analysed ROIs

# Typography: 18 x 4 cm printed at 100 %, so set the true point sizes.
CM = 1 / 2.54
FS_SMALL = 9                  # ticks, annotations  -- never below this
FS_LABEL = 9                  # axis labels
FS_TITLE = 11                 # panel letters / headings

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS_SMALL,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_SMALL,
    "ytick.labelsize": FS_SMALL,
    "legend.fontsize": FS_SMALL,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "pdf.fonttype": 42,        # keep text editable as text in Illustrator
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ------------------------------------------------------------ data loading
def load_qc_table():
    """Flatten qc_all_sessions_rebuild.mat into one row per sorted unit."""
    qc_all = sio.loadmat(QC_MAT, squeeze_me=True, struct_as_record=False)["qc_all"]
    rows = []
    for sess_i, sess in enumerate(qc_all.sessions, start=1):
        cells = np.atleast_1d(sess.qc)
        if cells.size == 0 or cells[0] is None:
            continue
        for cell_i, cell in enumerate(cells, start=1):
            reasons = np.atleast_1d(cell.fail_reasons)
            reasons = "; ".join(str(r) for r in reasons) if reasons.size else ""
            rows.append({
                "session": sess_i,
                "subject_id": str(sess.subject_id),
                "cell_idx": cell_i,
                "region_label": str(cell.regionLabel),
                "n_spikes": float(cell.n_spikes),
                "FR_Hz": float(cell.overall_FR_Hz),
                "RPV_frac": float(cell.RPV_frac),
                "corr_max": float(cell.corr_max),
                "grid_FR_CV": float(cell.grid_FR_CV),
                "base_accept": bool(cell.base_accept),
                "is_reliable": bool(cell.is_reliable),
                "fail_reasons": reasons,
            })
    qc = pd.DataFrame(rows)
    qc["excl_spikes"] = qc.fail_reasons.str.contains("few spikes")
    qc["excl_rpv"] = qc.fail_reasons.str.contains("RPV")
    qc["excl_dup"] = qc.fail_reasons.str.contains("high-corr")
    return qc


def pooled_isi_histogram(qc, max_ms=50.0, bin_ms=0.25):
    """Pooled ISI histogram over all QC-passed units, read from raw spikes.

    Cached to OUT_DIR so re-plotting does not re-touch the 6 GB raw file.
    """
    if os.path.exists(ISI_CACHE):
        cached = np.load(ISI_CACHE)
        return cached["edges"], cached["counts"], int(cached["n_units"])

    edges = np.arange(0.0, max_ms + bin_ms, bin_ms)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    n_units = 0

    with h5py.File(RAW_MAT, "r") as f:
        neural = f["abcd_data"]["neural_data"]
        for sess_i in range(neural.shape[0]):
            sess = f[neural[sess_i, 0]]
            spike_refs = sess["spikeTimes"]
            keep = qc.loc[qc.session == sess_i + 1].set_index("cell_idx")
            for cell_i in range(spike_refs.shape[0]):
                row = keep.loc[cell_i + 1] if (cell_i + 1) in keep.index else None
                if row is None or not row.is_reliable:
                    continue
                st = np.sort(np.ravel(f[spike_refs[cell_i, 0]][:]))
                isi_ms = np.diff(st) * 1e3
                counts += np.histogram(isi_ms, bins=edges)[0]
                n_units += 1

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(ISI_CACHE, edges=edges, counts=counts, n_units=n_units)
    return edges, counts, n_units


# ------------------------------------------------------------ panel helper
def panel_letter(ax, letter, title):
    """Bold panel letter + short heading above the axes."""
    ax.set_title(f"$\\bf{{{letter}}}$   {title}", fontsize=FS_TITLE,
                 loc="left", pad=3)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ------------------------------------------------------------- figure 1
def figure_quality_control(qc, isi):
    edges, counts, n_isi_units = isi
    keep = qc[qc.is_reliable]
    drop_spikes = qc[qc.excl_spikes]
    drop_dup = qc[qc.excl_dup]

    fig, axes = plt.subplots(
        1, 4, figsize=(18 * CM, 4 * CM),
        gridspec_kw=dict(wspace=0.46, left=0.055, right=0.985,
                         bottom=0.30, top=0.80))

    # -- a: spike count per unit ------------------------------------------
    ax = axes[0]
    bins = np.logspace(np.log10(qc.n_spikes.min()), np.log10(qc.n_spikes.max()), 40)
    ax.hist([keep.n_spikes, drop_spikes.n_spikes], bins=bins, stacked=True,
            color=[C_KEEP, C_DROP], edgecolor="black", linewidth=0.25)
    ax.axvline(MIN_SPIKES, color=C_THRESH, ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Spikes per unit")
    ax.set_ylabel("Units")
    ax.set_xticks([1e2, 1e3, 1e4, 1e5])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.text(0.98, 0.99, f"cut-off\n{MIN_SPIKES} spikes", color=C_THRESH,
            transform=ax.transAxes, fontsize=FS_SMALL, ha="right", va="top",
            linespacing=1.15)
    panel_letter(ax, "a", "Spike count")
    tidy(ax)

    # -- b: firing rate, included vs excluded -----------------------------
    # Direct labels instead of a legend: at 4 cm height a legend box either
    # covers the data or the panel heading.
    ax = axes[1]
    bins = np.logspace(np.log10(qc.FR_Hz.min()), np.log10(qc.FR_Hz.max()), 40)
    ax.hist(keep.FR_Hz, bins=bins, color=C_KEEP, edgecolor="black",
            linewidth=0.25)
    ax.hist(drop_spikes.FR_Hz, bins=bins, color=C_DROP, edgecolor="black",
            linewidth=0.25)
    ax.set_xscale("log")
    ax.set_xlabel("Firing rate (Hz)")
    ax.set_ylabel("Units")
    ax.set_xticks([1e-1, 1e0, 1e1])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.40)
    ax.text(0.02, 0.99, f"excl. {len(drop_spikes)}\n"
                        f"{drop_spikes.FR_Hz.median():.2f} Hz",
            transform=ax.transAxes, ha="left", va="top", fontsize=FS_SMALL,
            color=C_DROP, linespacing=1.15)
    ax.text(0.98, 0.99, f"incl. {len(keep)}\n"
                        f"{keep.FR_Hz.median():.2f} Hz",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS_SMALL,
            color=C_KEEP, linespacing=1.15)
    panel_letter(ax, "b", "Firing rate")
    tidy(ax)

    # -- c: pooled ISI histogram ------------------------------------------
    ax = axes[2]
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centres, counts / 1e3, width=np.diff(edges), color=C_KEEP,
           edgecolor="none", align="center")
    ax.axvspan(0, REFRAC_MS, color=C_THRESH, alpha=0.25, lw=0)
    ax.axvline(REFRAC_MS, color=C_THRESH, ls="--", lw=1.0)
    ax.set_xlim(0, ISI_XMAX_MS)
    ax.set_ylim(0, (counts[centres <= ISI_XMAX_MS].max() / 1e3) * 1.40)
    ax.set_xlabel("Inter-spike interval (ms)")
    ax.set_ylabel("Spike pairs ($\\times 10^3$)")
    ax.set_xticks([0, 10, 20])
    rpv_pct = 100 * keep.RPV_frac
    ax.text(0.98, 0.99,
            f"RPV median {rpv_pct.median():.0f} %\nmax {rpv_pct.max():.2f} %",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS_SMALL,
            linespacing=1.15)
    panel_letter(ax, "c", f"Refractory ({REFRAC_MS} ms)")
    tidy(ax)

    # -- d: duplicate detection -------------------------------------------
    ax = axes[3]
    bins = np.linspace(-0.05, 1.0, 42)
    ax.hist([keep.corr_max, drop_dup.corr_max], bins=bins, stacked=True,
            color=[C_KEEP, C_DROP], edgecolor="black", linewidth=0.25)
    ax.axvline(CORR_THRESH, color=C_THRESH, ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Max pairwise $r$")
    ax.set_ylabel("Units")
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_ylim(0.7, 700)
    ax.text(0.98, 0.99, f"{len(drop_dup)} removed\nas duplicate",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS_SMALL,
            color=C_DROP, linespacing=1.15,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    panel_letter(ax, "d", "Duplicate units")
    tidy(ax)

    return fig


# ------------------------------------------------------------- figure 2
def figure_roi_assignment(qc, roi):
    n_total = len(qc)
    n_spikes_out = int(qc.excl_spikes.sum())
    n_dup_out = int(qc.excl_dup.sum())
    n_passed = int(qc.is_reliable.sum())

    roi_counts = roi.atlas_roi.value_counts()
    roi_sessions = roi.groupby("atlas_roi")["subject"].nunique()
    n_analysed = int(roi_counts.reindex(ANALYSED_ROIS).sum())
    n_lowcov = n_passed - n_analysed

    fig, axes = plt.subplots(
        1, 3, figsize=(18 * CM, 4 * CM),
        gridspec_kw=dict(width_ratios=[0.85, 1.3, 1.35], wspace=0.45,
                         left=0.055, right=0.985, bottom=0.36, top=0.80))

    # -- a: unit yield -----------------------------------------------------
    # Horizontal bars: at 4 cm height these are the only stage labels that
    # fit unrotated.
    ax = axes[0]
    stages = [("analysed", n_analysed, "#448363"),
              ("QC pass", n_passed, C_KEEP),
              ("sorted", n_total, C_OTHER)]
    y = np.arange(3)
    ax.barh(y, [v for _, v, _ in stages], color=[c for _, _, c in stages],
            edgecolor="black", linewidth=0.3, height=0.62)
    for yi, (_, value, _) in zip(y, stages):
        ax.text(value + 0.025 * n_total, yi, str(value), ha="left",
                va="center", fontsize=FS_SMALL)
    ax.set_yticks(y)
    ax.set_yticklabels([s for s, _, _ in stages])
    ax.set_xlim(0, n_total * 1.30)
    ax.set_xticks([0, 500, 1000])
    ax.set_xlabel("Units")
    ax.set_ylim(-0.65, 2.65)
    panel_letter(ax, "a", "Unit yield")
    tidy(ax)

    # -- b: hippocampal anterior/mid split --------------------------------
    ax = axes[1]
    hc = roi[roi.atlas_roi.isin(["HC_anterior", "HC_mid"])]
    y_ant = hc.loc[hc.atlas_roi == "HC_anterior", "MNI_y_final"]
    y_mid = hc.loc[hc.atlas_roi == "HC_mid", "MNI_y_final"]
    bins = np.linspace(hc.MNI_y_final.min(), hc.MNI_y_final.max(), 34)
    ax.hist([y_mid, y_ant], bins=bins, stacked=True,
            color=[get_roi_colour("HC_mid"), get_roi_colour("HC_anterior")],
            edgecolor="black", linewidth=0.25)
    ax.axvline(HC_ANT_MID_Y, color=C_THRESH, ls="--", lw=1.0)
    ax.set_xlabel("MNI $y$ (mm)  posterior $\\rightarrow$ anterior")
    ax.set_ylabel("Units")
    ax.set_xticks([-30, -20, -10])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.30)
    ax.text(0.02, 0.97, f"mid\n{len(y_mid)}", transform=ax.transAxes,
            ha="left", va="top", fontsize=FS_SMALL, linespacing=1.15,
            color=get_roi_colour("HC_mid"), fontweight="bold")
    ax.text(0.98, 0.97, f"anterior\n{len(y_ant)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=FS_SMALL, linespacing=1.15,
            color=get_roi_colour("HC_anterior"), fontweight="bold")
    ax.annotate(f"{HC_ANT_MID_Y:.0f} mm",
                xy=(HC_ANT_MID_Y, 0.99), xycoords=("data", "axes fraction"),
                xytext=(3, 0), textcoords="offset points",
                ha="left", va="top", fontsize=FS_SMALL, color=C_THRESH)
    panel_letter(ax, "b", f"Hippocampal split ($n$ = {len(hc)})")
    tidy(ax)

    # -- c: units per ROI --------------------------------------------------
    ax = axes[2]
    labels = ["EC", "mPFC", "mOFC", "HC ant", "HC mid", "PCC", "other"]
    values = [int(roi_counts.get(r, 0)) for r in ANALYSED_ROIS] + [n_lowcov]
    sessions = [int(roi_sessions.get(r, 0)) for r in ANALYSED_ROIS]
    sessions += [int(roi.loc[~roi.atlas_roi.isin(ANALYSED_ROIS),
                             "subject"].nunique())]
    colours = [get_roi_colour(r) for r in ANALYSED_ROIS] + [C_OTHER]
    x = np.arange(len(labels))
    ax.bar(x, values, color=colours, edgecolor="black", linewidth=0.3,
           width=0.72)
    for xi, value, n_sess in zip(x, values, sessions):
        ax.text(xi, value + 0.03 * max(values), str(n_sess), ha="center",
                va="bottom", fontsize=FS_SMALL, color="#4a4a4a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Units")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_yticks([0, 150, 300])
    ax.text(0.02, 0.99, "grey = sessions", transform=ax.transAxes,
            ha="left", va="top", fontsize=FS_SMALL, color="#4a4a4a")
    panel_letter(ax, "c", "Units per ROI")
    tidy(ax)

    return fig


# ----------------------------------------------------------------- runner
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    qc = load_qc_table()
    roi = pd.read_csv(ROI_CSV)
    isi = pooled_isi_histogram(qc)

    qc.to_csv(os.path.join(OUT_DIR, "qc_per_unit_flat.csv"), index=False)

    fig1 = figure_quality_control(qc, isi)
    fig2 = figure_roi_assignment(qc, roi)

    for name, fig in [("fig_QC_single_units", fig1),
                      ("fig_QC_roi_assignment", fig2)]:
        for ext in ("pdf", "png", "svg"):
            fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=300)
        print(f"saved {name}.[pdf|png|svg]")

    settings = {
        "created": date.today().isoformat(),
        "inputs": {"qc_mat": QC_MAT, "roi_csv": ROI_CSV, "raw_mat": RAW_MAT},
        "qc_thresholds": {
            "min_spikes": MIN_SPIKES,
            "refractory_ms": REFRAC_MS,
            "rpv_frac_thresh": RPV_THRESH,
            "corr_bin_s": 0.10,
            "corr_thresh": CORR_THRESH,
        },
        "hc_split_y_mm": HC_ANT_MID_Y,
        "hc_split_reference": "Poppenk & Moscovitch 2013",
        "analysed_rois": ANALYSED_ROIS,
        "counts": {
            "sorted_units": int(len(qc)),
            "excluded_few_spikes": int(qc.excl_spikes.sum()),
            "excluded_rpv": int(qc.excl_rpv.sum()),
            "excluded_duplicate": int(qc.excl_dup.sum()),
            "qc_passed": int(qc.is_reliable.sum()),
            "analysed": int(roi.atlas_roi.isin(ANALYSED_ROIS).sum()),
            "per_roi": roi.atlas_roi.value_counts().to_dict(),
            "sessions_per_roi": roi.groupby("atlas_roi")["subject"]
                                   .nunique().to_dict(),
        },
        "figure_spec": {"width_cm": 18, "height_cm": 4,
                        "font": "Arial", "min_font_pt": FS_SMALL,
                        "heading_font_pt": FS_TITLE},
    }
    with open(os.path.join(OUT_DIR, "figure_settings.json"), "w") as fh:
        json.dump(settings, fh, indent=2)

    print(json.dumps(settings["counts"], indent=2))
    print(f"\nOutput: {OUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()
