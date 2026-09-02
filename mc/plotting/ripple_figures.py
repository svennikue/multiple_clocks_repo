#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication figures for the hippocampal sharp-wave-ripple analysis.

The main entry point, `chen_figure`, reproduces the layout of Chen et al. 2025
(J Neurosci 45:e1502252025) Figure 2a-b, which is the field's standard evidence
that a ripple detector is finding ripples:

    a  hippocampal contacts on a template brain
    b  grand-average ripple-locked voltage + time-frequency decomposition,
       and the same for one example ripple, with the marginal amplitude
       spectrum drawn over the TFR

`sharp_wave_figure` is not in Chen -- it exists because "sharp-wave ripple"
names a two-part event and the sharp wave deserves to be looked at rather than
assumed. See the note on polarity in `sharp_wave_profiles`.

@author: Svenja Kuchenhoff
"""

import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.fft import next_fast_len

import era_brewer

# ---------------------------------------------------------------- style ----
# These become subpanels of a half-A4 figure, so the whole thing is drawn at
# roughly the size it will be printed: ~18 cm wide, and font sizes are then
# real point sizes rather than something to be rescaled later.
CM = 1 / 2.54
FS_TITLE, FS_LABEL, FS_TICK = 11, 10, 9

PAL = era_brewer.era_brew("Showgirl2", n=7)
RIPPLE_C = "#B74C2D"        # ripple-band trace
RAW_C = "#3d3d3d"           # broadband trace
SW_C = "#448363"            # sharp wave
ROI_C = {"HC_anterior": "#23677E", "HC_mid": PAL[2], "HC": PAL[2],
         "HC_posterior": "#0e3d3a"}

RIPPLE_BAND = (80.0, 120.0)
SW_BAND_HZ = 20.0           # low-pass for the sharp wave; see sharp_wave_profiles
TFR_FREQS = np.arange(4.0, 176.0, 2.0)
TFR_BASELINE_S = (-1.5, -0.5)


def _rc():
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    }


# ------------------------------------------------------------- signals ----
def _bp(x, fs, lo, hi, order=4):
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def _lp(x, fs, hi, order=4):
    sos = butter(order, hi / (fs / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x)


def trough_lock(events, ripple_band_signal):
    """Sample index of each event's largest negative ripple-band deflection.

    Detection marks each event at the peak of its RMS *envelope*, which is the
    right definition for counting events but the wrong one for averaging them:
    the envelope says nothing about the phase of the underlying oscillation, so
    a grand average built on it cancels almost completely. Measured on the
    development set, envelope-locking retains 7% of the single-event ripple
    amplitude in the average; locking to the trough retains 95%.

    This is a display choice for the ripple-triggered average only. Detection,
    event counts, rates and every statistic are untouched by it.
    """
    out = []
    for _, r in events.iterrows():
        a, b = int(r.start_sample), int(r.stop_sample)
        if b <= a:
            out.append(int(r.peak_sample))
        else:
            out.append(a + int(np.argmin(ripple_band_signal[a:b])))
    return np.asarray(out, int)


def _snips(x, peaks, half):
    keep = [p for p in peaks if p - half >= 0 and p + half < len(x)]
    if not keep:
        return np.zeros((0, 2 * half))
    return np.stack([x[p - half:p + half] for p in keep])


def ripple_tfr(raw, fs, peaks, freqs=TFR_FREQS, win_s=0.25,
               baseline_s=TFR_BASELINE_S, n_cycles=6.0):
    """Ripple-locked TFR as % change from a pre-event baseline.

    Chen's normalisation: the baseline is taken from -1.5 to -0.5 s around each
    event, not from the edges of the display window. That matters -- a ripple is
    a transient sitting on 1/f activity, and normalising to the window edges
    leaves the low frequencies looking suppressed simply because the window is
    short.

    Returns (tfr_pct, times_ms, freqs).
    """
    half = int(round(win_s * fs))
    b0, b1 = int(round(baseline_s[0] * fs)), int(round(baseline_s[1] * fs))
    peaks = np.asarray([p for p in peaks
                        if p + b0 >= 0 and p + half < len(raw)], int)
    tfr = np.full((len(freqs), 2 * half), np.nan)
    if not len(peaks):
        return tfr, np.linspace(-win_s, win_s, 2 * half) * 1000, freqs

    for j, f0 in enumerate(freqs):
        bw = max(2.0, f0 / n_cycles * 2.0)
        lo, hi = max(1.0, f0 - bw / 2), min(fs / 2 - 1.0, f0 + bw / 2)
        if hi <= lo:
            continue
        sos = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
        # hilbert FFTs at the signal's own length, which for a real recording is
        # an awkward number (2433051 for s02) and costs 1.08 s per frequency
        # against 0.14 s at the next fast length -- 8x, and this runs once per
        # frequency per derivation. The zero-padding perturbs only the edges:
        # measured on s02, median difference 8e-8 uV and interior max 1.1e-5 uV
        # against a 0.35 uV median envelope. Events near the edges are excluded
        # from the average anyway.
        amp = np.abs(hilbert(sosfiltfilt(sos, raw), N=next_fast_len(len(raw)))[:len(raw)])
        ev = _snips(amp, peaks, half)
        base = np.array([amp[p + b0:p + b1].mean() for p in peaks])
        ok = base > 0
        if ok.sum():
            tfr[j] = ((ev[ok] - base[ok, None]) / base[ok, None] * 100).mean(0)

    t_ms = np.linspace(-win_s, win_s, 2 * half) * 1000
    return tfr, t_ms, freqs


# --------------------------------------------------------- sharp wave ----
def sharp_wave_profiles(raw, fs, peaks, sw_hz=SW_BAND_HZ, win_s=0.25):
    """Ripple-locked low-frequency deflection, per contact.

    Two things make the sharp wave hard to see in this montage, and both are
    properties of the recording rather than of the detector:

    1. Its polarity depends on where the contact sits relative to the CA1
       pyramidal layer -- the dipole reverses across it. Averaging contacts
       without regard to sign therefore cancels a real signal.
    2. Bipolar re-referencing between neighbouring contacts is a spatial
       derivative, and the sharp wave is a spatially broad, low-frequency
       dipole. Much of it is common to both contacts and subtracts away, while
       the spatially focal ripple survives. This is why Chen's own grand
       average shows a ripple with no sharp wave under it.

    So this returns the per-contact profile and its sign, and leaves the
    interpretation to the caller. Nothing here flips anything.
    """
    half = int(round(win_s * fs))
    sw = _lp(raw, fs, sw_hz)
    sn = _snips(sw, peaks, half)
    if not len(sn):
        return None
    m = sn.mean(0)
    c, w = half, int(round(0.025 * fs))
    edge = int(round(0.05 * fs))
    flank = np.r_[m[:edge], m[-edge:]]
    return {
        "mean": m,
        "sem": sn.std(0) / np.sqrt(len(sn)),
        "n": len(sn),
        "deflection_uv": float(m[c - w:c + w].mean()),
        "flank_sd": float(flank.std()),
        "snr": float(m[c - w:c + w].mean() / flank.std()) if flank.std() else np.nan,
        "t_ms": np.linspace(-win_s, win_s, 2 * half) * 1000,
    }


# ------------------------------------------------------------- panels ----
def _panel_glassbrain(ax, coords, rois=None, title="Hippocampal contacts"):
    """Contacts on a glass brain. Falls back to a scatter if nilearn is absent."""
    coords = np.atleast_2d(np.asarray(coords, float))
    try:
        from nilearn import plotting as nlplot
        disp = nlplot.plot_glass_brain(
            None, display_mode="lyrz", axes=ax, black_bg=False,
            alpha=0.12, colorbar=False)
        if rois is None:
            disp.add_markers(coords, marker_color=ROI_C["HC_mid"],
                             marker_size=18)
        else:
            for roi in pd.unique(pd.Series(rois)):
                sel = np.asarray(rois) == roi
                disp.add_markers(coords[sel],
                                 marker_color=ROI_C.get(roi, PAL[2]),
                                 marker_size=18)
        ax.set_title(title, fontsize=FS_TITLE)
        return disp
    except Exception as e:                                # nilearn optional
        ax.scatter(coords[:, 0], coords[:, 2], s=14, c=ROI_C["HC_mid"])
        ax.set_xlabel("MNI x (mm)"); ax.set_ylabel("MNI z (mm)")
        ax.set_title(f"{title}\n(glass brain unavailable: {type(e).__name__})",
                     fontsize=FS_TICK)
        return None


def _panel_trace(ax, t_ms, y, color, ylabel, title=None, sem=None, xlim=250,
                 overlay=None, overlay_label=None):
    ax.plot(t_ms, y, color=color, lw=1.1)
    if overlay is not None:
        # A single unaveraged hippocampal trace is busy -- Chen's example panel
        # is too -- so the band-passed signal is drawn over it, scaled to the
        # axis, to show where the ripple actually is.
        sc = np.nanmax(np.abs(y)) / max(np.nanmax(np.abs(overlay)), 1e-9) * 0.55
        ax.plot(t_ms, overlay * sc, color=RIPPLE_C, lw=0.8,
                label=overlay_label or f"80–120 Hz (x{sc:.0f})")
        ax.legend(frameon=False, fontsize=FS_TICK - 2, loc="lower right")
    if sem is not None:
        ax.fill_between(t_ms, y - sem, y + sem, color=color, alpha=0.20, lw=0)
    ax.axvline(0, color="0.65", ls=":", lw=0.8)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    if title:
        ax.set_title(title, fontsize=FS_TITLE)
    ax.tick_params(labelbottom=False)


def _panel_tfr(ax, tfr, t_ms, freqs, xlim=250, vmax=None, show_marginal=True,
               cbar=False):
    if vmax is None:
        vmax = np.nanpercentile(np.abs(tfr), 99) or 1.0
    im = ax.pcolormesh(t_ms, freqs, tfr, cmap="RdYlBu_r", shading="auto",
                       vmin=-vmax, vmax=vmax, rasterized=True)
    for f in RIPPLE_BAND:
        ax.axhline(f, color="0.25", ls=":", lw=0.6)
    if show_marginal:
        # Chen draw the marginal amplitude spectrum as a curve over the TFR:
        # mean % change across time for each frequency, mapped onto the x axis.
        marg = np.nanmean(tfr, axis=1)
        span = np.nanmax(np.abs(marg))
        if span:
            x = marg / span * (xlim * 0.42)
            ax.plot(x, freqs, color="k", lw=1.0)
            ax.axvline(0, color="k", lw=0.4, alpha=0.3)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(freqs[0], freqs[-1])
    ax.set_xlabel("Time from ripple (ms)", fontsize=FS_LABEL)
    ax.set_ylabel("Frequency (Hz)", fontsize=FS_LABEL)
    if cbar:
        cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
        cb.set_label("Amplitude (% change)", fontsize=FS_LABEL)
        cb.ax.tick_params(labelsize=FS_TICK)
    return im


# ------------------------------------------------------------- figure ----
def chen_figure(raw_by_pair, fs, events, coords=None, rois=None,
                out_stem=None, title="", example=None, win_s=0.25):
    """Chen et al. 2025 Figure 2a-b, reproduced.

    raw_by_pair : {pair_id: 1-D float array}  bipolar signal per derivation
    events      : DataFrame of ACCEPTED events with pair_id, start/stop/peak_sample
    coords      : (n, 3) MNI coordinates of the contacts, for panel a
    example     : (pair_id, peak_sample) to feature, or None to pick the
                  strongest event automatically
    """
    half = int(round(win_s * fs))
    ga_raw, ga_bp, all_peaks_by_pair = [], [], {}

    for pid, raw in raw_by_pair.items():
        e = events[events.pair_id == pid]
        if not len(e):
            continue
        rb = _bp(raw, fs, *RIPPLE_BAND)
        pk = trough_lock(e, rb)
        all_peaks_by_pair[pid] = pk
        ga_raw.append(_snips(raw, pk, half))
        ga_bp.append(_snips(rb, pk, half))

    if not ga_raw:
        print("  chen_figure: no events"); return None
    ga_raw = np.concatenate(ga_raw); ga_bp = np.concatenate(ga_bp)
    t_ms = np.linspace(-win_s, win_s, 2 * half) * 1000

    # grand-average TFR: pool events across derivations
    tfr_stack = []
    for pid, pk in all_peaks_by_pair.items():
        tf, _, fr = ripple_tfr(raw_by_pair[pid], fs, pk, win_s=win_s)
        if np.isfinite(tf).any():
            tfr_stack.append(tf)
    ga_tfr = np.nanmean(np.stack(tfr_stack), axis=0)

    # the example: strongest accepted event, unless one was named
    if example is None:
        best = events.sort_values("rms_peak_z", ascending=False).iloc[0]
        ex_pid = best.pair_id
        ex_pk = trough_lock(best.to_frame().T, _bp(raw_by_pair[ex_pid], fs,
                                                   *RIPPLE_BAND))[0]
    else:
        ex_pid, ex_pk = example
    ex_raw = raw_by_pair[ex_pid][ex_pk - half:ex_pk + half]
    ex_tfr, _, _ = ripple_tfr(raw_by_pair[ex_pid], fs, [ex_pk], win_s=win_s)

    return chen_panels(ga_raw, ga_tfr, t_ms, ex_raw, ex_tfr,
                       coords=coords, rois=rois, out_stem=out_stem, title=title,
                       ga_bp=ga_bp)


def chen_panels(ga_raw, ga_tfr, t_ms, ex_raw, ex_tfr, coords=None, rois=None,
                out_stem=None, title="", ga_bp=None, n_contacts=None,
                ex_bp=None):
    """Draw Chen Fig 2a-b from already-averaged data.

    ga_raw : (n_ripples, n_times) or (n_contacts, n_times) stack to average
    """
    ga_raw = np.atleast_2d(ga_raw)
    n_ev = len(ga_raw)
    lab = (f"Grand average ({n_contacts} contacts)" if n_contacts
           else f"Grand average (n = {n_ev})")

    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(18 * CM, 10.5 * CM))
        gs = gridspec.GridSpec(
            2, 3, figure=fig, width_ratios=[1.35, 1, 1],
            height_ratios=[0.72, 1], hspace=0.12, wspace=0.42,
            left=0.06, right=0.97, top=0.90, bottom=0.11)

        # (a) contacts -----------------------------------------------------
        ax_a = fig.add_subplot(gs[:, 0])
        if coords is not None and len(coords):
            _panel_glassbrain(ax_a, coords, rois)
        else:
            ax_a.axis("off")
        ax_a.text(-0.02, 1.02, "a", transform=ax_a.transAxes,
                  fontsize=FS_TITLE + 2, fontweight="bold", va="bottom")

        # (b) grand average -------------------------------------------------
        ax_b1 = fig.add_subplot(gs[0, 1])
        _panel_trace(ax_b1, t_ms, ga_raw.mean(0), RAW_C,
                     r"Voltage ($\mu$V)", lab,
                     sem=ga_raw.std(0) / np.sqrt(len(ga_raw)))
        ax_b1.text(-0.28, 1.06, "b", transform=ax_b1.transAxes,
                   fontsize=FS_TITLE + 2, fontweight="bold", va="bottom")
        ax_b2 = fig.add_subplot(gs[1, 1])
        _panel_tfr(ax_b2, ga_tfr, t_ms, TFR_FREQS)

        # (c) example -------------------------------------------------------
        ax_c1 = fig.add_subplot(gs[0, 2])
        _panel_trace(ax_c1, t_ms, ex_raw, RAW_C, r"Voltage ($\mu$V)",
                     "Example ripple", overlay=ex_bp)
        ax_c2 = fig.add_subplot(gs[1, 2])
        _panel_tfr(ax_c2, ex_tfr, t_ms, TFR_FREQS, cbar=True)

        if title:
            fig.suptitle(title, fontsize=FS_TITLE, y=0.985)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf "
                  f"({n_ev} rows)")
    out = dict(n=n_ev, ga_raw_ptp=float(np.ptp(ga_raw.mean(0))))
    if ga_bp is not None and len(ga_bp):
        out["ga_bp_ptp"] = float(np.ptp(np.atleast_2d(ga_bp).mean(0)))
        out["single_bp_uv"] = float(np.abs(np.atleast_2d(ga_bp)).max(1).mean())
    return out


def sharp_wave_figure(raw_by_pair, fs, events, out_stem=None, title="",
                      win_s=0.25):
    """Per-contact ripple-locked low-frequency deflection.

    One line per derivation, with its own sign kept. If the sharp wave were
    being recovered, the lines would deflect together at t = 0; if the montage
    has cancelled it, they will not. Either way the figure states which.
    """
    rows, profiles = [], {}
    for pid, raw in raw_by_pair.items():
        e = events[events.pair_id == pid]
        if len(e) < 30:
            continue
        pk = trough_lock(e, _bp(raw, fs, *RIPPLE_BAND))
        pr = sharp_wave_profiles(raw, fs, pk, win_s=win_s)
        if pr is None:
            continue
        profiles[pid] = pr
        rows.append(dict(pair_id=pid, n=pr["n"],
                         deflection_uv=pr["deflection_uv"],
                         flank_sd=pr["flank_sd"], snr=pr["snr"]))
    if not profiles:
        return pd.DataFrame()
    tab = pd.DataFrame(rows)

    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(17 * CM, 6.5 * CM),
                                 gridspec_kw=dict(width_ratios=[1.6, 1],
                                                  wspace=0.45))
        fig.subplots_adjust(top=0.74, bottom=0.20, left=0.10, right=0.97)
        ax = axes[0]
        for pid, pr in profiles.items():
            ax.plot(pr["t_ms"], pr["mean"], lw=1.0, alpha=0.85, label=pid)
        ax.axvline(0, color="0.65", ls=":", lw=0.8)
        ax.axhline(0, color="0.85", lw=0.6)
        ax.set_xlim(-win_s * 1000, win_s * 1000)
        ax.set_xlabel("Time from ripple (ms)")
        ax.set_ylabel(r"Low-frequency (< %g Hz) ($\mu$V)" % SW_BAND_HZ)
        ax.set_title("Slow deflection at ripple", fontsize=FS_TITLE - 1)
        if len(profiles) <= 6:
            ax.legend(frameon=False, fontsize=FS_TICK - 2, loc="best")

        ax = axes[1]
        col = [RIPPLE_C if v < 0 else SW_C for v in tab.deflection_uv]
        ax.barh(np.arange(len(tab)), tab.deflection_uv, color=col, height=0.7)
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_yticks(np.arange(len(tab)))
        ax.set_yticklabels(tab.pair_id, fontsize=FS_TICK - 2)
        ax.set_xlabel(r"Deflection at ripple ($\mu$V)")
        n_neg = int((tab.deflection_uv < 0).sum())
        ax.set_title(f"Polarity: {n_neg} negative / {len(tab) - n_neg} positive",
                     fontsize=FS_TITLE - 1)

        if title:
            fig.suptitle(title, fontsize=FS_TITLE, y=0.99)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return tab


def monopolar_sharpwave_figure(profiles, out_stem=None, title="", ncol=3):
    """Monopolar contacts vs their bipolar difference, ripple-locked.

    `profiles` maps "<pair> | <trace>" to a sharp_wave_profiles dict. One panel
    per pair, with contact A, contact B and A-B overlaid, so the question "does
    the subtraction remove a deflection that was there" is answered by looking
    at one panel rather than by comparing figures.
    """
    pairs = sorted({k.split(" | ")[0] for k in profiles})
    if not pairs:
        return None
    ncol = min(ncol, len(pairs))
    nrow = int(np.ceil(len(pairs) / ncol))
    style = {"contact A": dict(color="#7eb1c4", lw=1.1),
             "contact B": dict(color="#175e62", lw=1.1),
             "bipolar A-B": dict(color=RIPPLE_C, lw=1.4)}

    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                                 figsize=(6.2 * ncol * CM, 5.6 * nrow * CM))
        for ax in axes.ravel():
            ax.axis("off")
        for j, pid in enumerate(pairs):
            ax = axes[j // ncol][j % ncol]; ax.axis("on")
            for tag, st in style.items():
                pr = profiles.get(f"{pid} | {tag}")
                if pr is None:
                    continue
                ax.plot(pr["t_ms"], pr["mean"], label=tag, **st)
            ax.axvline(0, color="0.65", ls=":", lw=0.8)
            ax.axhline(0, color="0.88", lw=0.6)
            ax.set_title(pid, fontsize=FS_TICK)
            if j // ncol == nrow - 1:
                ax.set_xlabel("Time from ripple (ms)")
            if j % ncol == 0:
                ax.set_ylabel(r"< %g Hz ($\mu$V)" % SW_BAND_HZ)
        axes[0][0].legend(frameon=False, fontsize=FS_TICK - 2, loc="best")
        if title:
            fig.suptitle(title, fontsize=FS_TITLE)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


CRIT_C = {"raw_amplitude": "#F15A29", "first_derivative": "#F7931E",
          "hf_rms_250": "#6B60AA", "broadband_1_60": "#23677E",
          "ied_janca": "#a30d6c"}
CRIT_SHORT = {"raw_amplitude": "amplitude", "first_derivative": "gradient",
              "hf_rms_250": ">250 Hz", "broadband_1_60": "1–60 Hz",
              "ied_janca": "IED"}


def artifact_figure(raw_by_pair, fs, masks, stats, out_stem=None, title="",
                    n_examples=4, win_s=1.0):
    """What the artifact rejection removes, and whether it should have.

    The criteria together discard 35-54% of a typical recording. That is a large
    fraction to delete on the strength of a percentage in a table, so this shows
    the removed data itself: which criterion fired, where in the recording, and
    what the excluded stretches actually look like next to kept ones at the same
    scale.

    masks : {pair_id: {"bad": bool array, "per": {criterion: bool array}}}
    stats : {pair_id: {criterion: fraction flagged}}
    """
    pairs = [p for p in raw_by_pair if p in masks]
    if not pairs:
        return None
    crits = [c for c in CRIT_C if any(c in stats[p] for p in pairs)]
    half = int(round(win_s * fs))

    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(18 * CM, 15 * CM))
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 0.7, 1.5],
                               hspace=0.95, wspace=0.28,
                               left=0.09, right=0.97, top=0.91, bottom=0.07)

        # (a) how much each criterion flags ---------------------------------
        ax = fig.add_subplot(gs[0, 0])
        w = 0.8 / max(1, len(pairs))
        for i, p in enumerate(pairs):
            v = [100 * stats[p].get(c, 0.0) for c in crits]
            ax.bar(np.arange(len(crits)) + i * w - 0.4 + w / 2, v, width=w,
                   color=[CRIT_C[c] for c in crits], edgecolor="w", linewidth=0.4)
        ax.set_xticks(np.arange(len(crits)))
        ax.set_xticklabels([CRIT_SHORT.get(c, c) for c in crits],
                           fontsize=FS_TICK - 1)
        ax.set_ylabel("% of recording flagged")
        ax.set_title("Each criterion alone", fontsize=FS_TITLE - 1)
        ax.legend(handles=[plt.Line2D([], [], marker="s", ls="", ms=5,
                                      color=CRIT_C[c], label=CRIT_SHORT.get(c, c))
                           for c in crits],
                  frameon=False, fontsize=FS_TICK - 2, ncol=2,
                  loc="upper right", handletextpad=0.3, columnspacing=0.8)

        # (b) and how much survives after union + padding ---------------------
        ax = fig.add_subplot(gs[0, 1])
        comb = [100 * stats[p].get("combined_after_pad", np.nan) for p in pairs]
        ax.bar(np.arange(len(pairs)), comb, color=RIPPLE_C, edgecolor="w")
        ax.axhline(100 * 2 / 3, color="0.3", ls="--", lw=0.9)
        ax.text(len(pairs) - 0.5, 100 * 2 / 3 + 2, "contact excluded above 2/3",
                ha="right", fontsize=FS_TICK - 1, color="0.3")
        ax.set_xticks(np.arange(len(pairs)))
        ax.set_xticklabels([p.split("-")[0] for p in pairs], rotation=20,
                           ha="right", fontsize=FS_TICK - 2)
        ax.set_ylabel("% removed"); ax.set_ylim(0, 100)
        ax.set_title("Union + 1 s padding", fontsize=FS_TITLE - 1)

        # (c) where in the recording ------------------------------------------
        ax = fig.add_subplot(gs[1, :])
        for i, p in enumerate(pairs):
            bad = masks[p]["bad"]
            t = np.arange(len(bad)) / fs / 60.0
            ax.fill_between(t, i, i + 0.8, where=bad, color=RIPPLE_C, lw=0)
            ax.text(-0.004 * t[-1], i + 0.4, p, ha="right", va="center",
                    fontsize=FS_TICK - 2)
        ax.set_ylim(-0.2, len(pairs)); ax.set_yticks([])
        ax.set_xlabel("Time in recording (min)")
        ax.set_title("When it was removed (red = excluded)", fontsize=FS_TITLE - 1)

        # (d) what the removed data looks like, vs kept ------------------------
        p0 = pairs[int(np.argmax([stats[p].get("combined_after_pad", 0)
                                  for p in pairs]))]
        x = raw_by_pair[p0]; bad = masks[p0]["bad"]; per = masks[p0]["per"]
        # rank removed stretches by how extreme they are
        lab_bad = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]) == 1)
        ends = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]) == -1)
        segs = [(a, b) for a, b in zip(lab_bad, ends)
                if a - half >= 0 and b + half < len(x)]
        segs.sort(key=lambda ab: -np.abs(x[ab[0]:ab[1]]).max() if ab[1] > ab[0] else 0)
        good = np.flatnonzero(~bad)
        rng = np.random.RandomState(42)

        inner = gridspec.GridSpecFromSubplotSpec(
            2, n_examples, subplot_spec=gs[2, :], hspace=0.55, wspace=0.25)
        ylims = []
        for j in range(n_examples):
            # removed
            ax = fig.add_subplot(inner[0, j])
            if j < len(segs):
                a, b = segs[j]
                c = (a + b) // 2
                seg = x[c - half:c + half]
                tt = (np.arange(len(seg)) - half) / fs
                ax.plot(tt, seg, color="0.35", lw=0.7)
                ax.axvspan((a - c) / fs, (b - c) / fs, color=RIPPLE_C, alpha=0.22, lw=0)
                # Which criteria fired, as colour marks: spelling them out
                # collides between panels once several fire at once, which on
                # an IED they usually do.
                fired = [k for k in CRIT_C if k in per and per[k][a:b].any()]
                for m, k in enumerate(fired):
                    ax.plot(0.06 + 0.10 * m, 1.06, "s", color=CRIT_C[k],
                            ms=4, transform=ax.transAxes, clip_on=False)
                if not fired:
                    ax.set_title("padding only", fontsize=FS_TICK - 2,
                                 color="0.5")
            ax.set_xticks([])
            ylims.append(ax.get_ylim())
            if j == 0:
                ax.set_ylabel("REMOVED\n" + r"($\mu$V)", fontsize=FS_TICK)
            # kept, matched window
            ax = fig.add_subplot(inner[1, j])
            if len(good) > 2 * half:
                c = int(good[rng.randint(half, len(good) - half)])
                if c - half >= 0 and c + half < len(x):
                    seg = x[c - half:c + half]
                    ax.plot((np.arange(len(seg)) - half) / fs, seg,
                            color=SW_C, lw=0.7)
            # same scale as the panel above: the point is the size difference
            if j < len(ylims):
                ax.set_ylim(ylims[j])
            ax.set_xlabel("s", fontsize=FS_TICK - 1)
            if j == 0:
                ax.set_ylabel("KEPT\n" + r"($\mu$V)", fontsize=FS_TICK)

        fig.suptitle(title or f"Artifact rejection — {p0}", fontsize=FS_TITLE)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig


def sharp_wave_examples(raw_by_pair, fs, events, sw_table, out_stem=None,
                        title="", n=6, win_s=0.35):
    """Single ripples on the contacts where a sharp wave actually survives.

    Picked from the derivation with the largest surviving low-frequency
    deflection (`sw_table`), then ranked within it by how large that event's own
    slow deflection is. This is deliberately a best-case display: it shows that
    the morphology exists in these data, not how typical it is. The typical case
    is the per-derivation table, where 4 of 9 show nothing.
    """
    if sw_table is None or not len(sw_table):
        return None
    best = sw_table.reindex(sw_table.deflection_uv.abs().sort_values(
        ascending=False).index)
    pid = None
    for _, r in best.iterrows():
        if r.pair_id in raw_by_pair:
            pid = r.pair_id; break
    if pid is None:
        return None

    raw = raw_by_pair[pid]
    rb = _bp(raw, fs, *RIPPLE_BAND)
    sw = _lp(raw, fs, SW_BAND_HZ)
    e = events[events.pair_id == pid]
    if not len(e):
        return None
    pk = trough_lock(e, rb)
    half = int(round(win_s * fs))
    ok = [(p, abs(sw[p])) for p in pk if p - half >= 0 and p + half < len(raw)]
    if not ok:
        return None
    ok.sort(key=lambda t: -t[1])
    ok = ok[:n]

    ncol = min(3, len(ok)); nrow = int(np.ceil(len(ok) / ncol))
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                                 figsize=(6.2 * ncol * CM, 5.2 * nrow * CM))
        for ax in axes.ravel():
            ax.axis("off")
        for j, (p, _) in enumerate(ok):
            ax = axes[j // ncol][j % ncol]; ax.axis("on")
            t = (np.arange(-half, half) / fs) * 1000
            ax.plot(t, raw[p - half:p + half], color="0.55", lw=0.6,
                    label="broadband")
            ax.plot(t, sw[p - half:p + half], color=SW_C, lw=1.6,
                    label=f"< {SW_BAND_HZ:g} Hz")
            sc = np.abs(raw[p - half:p + half]).max() / max(
                np.abs(rb[p - half:p + half]).max(), 1e-9) * 0.45
            ax.plot(t, rb[p - half:p + half] * sc, color=RIPPLE_C, lw=0.9,
                    label=f"80–120 Hz (x{sc:.0f})")
            ax.axvline(0, color="0.65", ls=":", lw=0.8)
            ax.set_title(f"t = {p / fs:.1f} s", fontsize=FS_TICK)
            if j // ncol == nrow - 1:
                ax.set_xlabel("Time from ripple (ms)")
            if j % ncol == 0:
                ax.set_ylabel(r"$\mu$V")
        axes[0][0].legend(frameon=False, fontsize=FS_TICK - 2, loc="upper left")
        fig.suptitle(title or f"Ripples with a sharp wave — {pid}",
                     fontsize=FS_TITLE)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf ({pid})")
    return pid


STATE_COLORS = {"A": "#F15A29", "B": "#F7931E", "C": "#C7C6E2", "D": "#6B60AA"}
PHASE_COLORS = {"discovery": "#D7657F", "early": "#FCDDE3",
                "middle": "#D7657F", "later": "#5C1027", "late": "#5C1027"}
OBS_LINE_C = "#0e3d3a"


def _rate_by(counts, col):
    """Pooled rate per level: summed events over summed exposure.

    Not the mean of per-window rates -- that weights a 0.5 s window as heavily
    as a 5 s one and is dominated by the short ones.
    """
    g = counts.groupby(col).agg(n=("n_ripples", "sum"), e=("exposure_s", "sum"))
    g["rate_hz"] = g["n"] / g["e"].replace(0, np.nan)
    # subject-level spread, for an error bar that reflects the unit of inference
    per_s = (counts.groupby([col, "subject_key"])
             .agg(n=("n_ripples", "sum"), e=("exposure_s", "sum")))
    per_s["r"] = per_s["n"] / per_s["e"].replace(0, np.nan)
    # `sem` shadows DataFrame.sem, so every read of it must be by key
    g["sem"] = per_s.groupby(level=0)["r"].agg(
        lambda v: v.std() / max(np.sqrt(v.notna().sum()), 1))
    g["n_subjects"] = per_s.groupby(level=0)["r"].count()
    return g


def hypothesis_figure(results, out_stem=None, title="", ncol=4):
    """One panel per hypothesis: the condition rates, plus the permutation null.

    Bars are the pooled rate (events / artifact-free seconds), error bars the
    SEM across subjects, because the subject is the unit of inference. The
    inset histogram is the circular-shift null with the observed value marked,
    since that -- not the GLM p -- is the primary inference.
    """
    keys = [k for k in ("H1", "H2", "H3", "H4") if k in results]
    if not keys:
        return None
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(2, len(keys), squeeze=False,
                                 figsize=(5.2 * len(keys) * CM, 11 * CM),
                                 gridspec_kw=dict(height_ratios=[1.5, 1],
                                                  hspace=0.75, wspace=0.5))
        for j, k in enumerate(keys):
            r = results[k]
            c = r.get("counts")
            ax = axes[0][j]
            if k == "H3" and c is not None and len(c):
                ax.scatter(c.rate_hz, c.errors_per_repeat_after, s=9,
                           color=PHASE_COLORS["discovery"], alpha=0.6,
                           edgecolor="none")
                if c.rate_hz.notna().sum() > 2:
                    b = np.polyfit(c.rate_hz, c.errors_per_repeat_after, 1)
                    xs = np.linspace(c.rate_hz.min(), c.rate_hz.max(), 20)
                    ax.plot(xs, np.polyval(b, xs), color=OBS_LINE_C, lw=1.2)
                ax.set_xlabel("Ripple rate after first-D (Hz)")
                ax.set_ylabel("Errors per later repeat")
            elif c is not None and len(c):
                # H2's split column is `phase_after` (the phase of the repeat
                # that follows the pause), not `phase`
                col = ("condition" if k == "H1" else
                       "phase_after" if k == "H2" else "feedback")
                if col not in c.columns:
                    col = "condition"
                g = _rate_by(c, col)
                cols = [PHASE_COLORS.get(str(i), PAL[j % len(PAL)]) for i in g.index]
                ax.bar(np.arange(len(g)), g["rate_hz"], yerr=g["sem"], color=cols,
                       edgecolor="w", capsize=2.5, error_kw=dict(lw=0.8))
                ax.set_xticks(np.arange(len(g)))
                ax.set_xticklabels([str(i).replace("_", "\n") for i in g.index],
                                   fontsize=FS_TICK - 1)
                ax.set_ylabel("Ripple rate (Hz)")
            ax.set_title(f"{k}", fontsize=FS_TITLE)

            # the null
            ax = axes[1][j]
            null = r.get("null")
            if null is None and "null_mean" in r:
                null = None
            if null is not None and len(np.ravel(null)):
                ax.hist(np.ravel(null), bins=30, color="0.8", edgecolor="w",
                        linewidth=0.3)
            obs = r.get("observed_log_rr", np.nan)
            if np.isfinite(obs):
                ax.axvline(obs, color=OBS_LINE_C, lw=1.6)
            ax.set_xlabel("log rate ratio" if k != "H3" else "coefficient")
            ax.set_ylabel("permutations")
            p = r.get("p_perm", np.nan)
            ax.set_title(f"p = {p:.3f}" if np.isfinite(p) else "p = n/a",
                         fontsize=FS_TICK)
        if title:
            fig.suptitle(title, fontsize=FS_TITLE, y=1.0)
        if out_stem:
            os.makedirs(os.path.dirname(out_stem), exist_ok=True)
            fig.savefig(out_stem + ".png", dpi=300)
            fig.savefig(out_stem + ".pdf")
            plt.close(fig)
            print(f"    wrote {os.path.basename(out_stem)}.png/.pdf")
    return fig
