#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 4 CHECKPOINT: does the detector actually find ripples?

Produces the figure that has to be judged before any statistics are written.
Items 1, 2 and 6 are the ones that cannot be fixed later by re-thresholding:

  1. grand-average ripple-triggered waveform (should show the ripple riding on
     a sharp wave)
  2. ripple-triggered time-frequency (should show an isolated 80-100 Hz blob,
     not a broadband smear)
  3. example single events
  4. peak-frequency distribution
  5. rate per derivation against Chen's reference range
  6. rejection cascade, and a white-matter control derivation whose rate should
     be far lower than hippocampus -- if it is not, the detector is finding
     artifacts rather than ripples

Usage:
    python scripts/swr_qc_report.py --session=38

@author: Svenja Kuchenhoff
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_detect as det
import era_brewer

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_v1"
PAL = era_brewer.era_brew("Showgirl2", n=7)
HC_COL, ACC_COL = PAL[0], "#0e3d3a"
WIN_S = 0.5

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
})


def _snippets(sig, peaks, half):
    keep = (peaks - half >= 0) & (peaks + half < sig.shape[-1])
    p = peaks[keep]
    if not len(p):
        return np.zeros((0, 2 * half))
    return np.stack([sig[q - half:q + half] for q in p])


def qc_report(session, analysis_name=ANALYSIS_NAME, max_events=800):
    swr_io.start_log(os.path.join(swr_io.session_deriv_dir(int(session), swr_io.get_data_root()), "LFP-ripples", analysis_name), "swr_qc_report")
    session = int(session)
    R = swr_io.get_data_root()
    clean_dir = os.path.join(swr_io.session_deriv_dir(session, R),
                             "LFP-clean", analysis_name)
    rip_dir = os.path.join(swr_io.session_deriv_dir(session, R),
                           "LFP-ripples", analysis_name)
    ev_p = os.path.join(rip_dir, "ripple_events.csv")
    if not os.path.isfile(ev_p):
        print(f"s{session:02d}: no ripple_events.csv"); return None

    sig = np.load(os.path.join(clean_dir, "continuous.npy"), mmap_mode='r')
    pairs = pd.read_csv(os.path.join(clean_dir, "pairs.csv"))
    events = pd.read_csv(ev_p)
    qc = pd.read_csv(os.path.join(rip_dir, "channel_qc.csv"))
    with open(os.path.join(clean_dir, "meta.json")) as f:
        fs = float(json.load(f)["fs"])

    passed = events[events.passed]
    if not len(passed):
        print(f"s{session:02d}: no passing events"); return None
    half = int(WIN_S * fs)

    # gather snippets across derivations
    raw_sn, bp_sn, sw_sn = [], [], []
    sos_r = butter(4, [det.RIPPLE_BAND[0]/(fs/2), det.RIPPLE_BAND[1]/(fs/2)],
                   btype='band', output='sos')
    sos_s = butter(4, [det.SHARPWAVE_BAND[0]/(fs/2), det.SHARPWAVE_BAND[1]/(fs/2)],
                   btype='band', output='sos')
    for i, p in pairs.iterrows():
        ev = passed[passed.pair_id == p.pair_id]
        if not len(ev):
            continue
        x = np.asarray(sig[i], float)
        pk = ev.peak_sample.to_numpy(int)
        if len(pk) > max_events:
            pk = np.random.RandomState(42).choice(pk, max_events, replace=False)
        raw_sn.append(_snippets(x, pk, half))
        bp_sn.append(_snippets(sosfiltfilt(sos_r, x), pk, half))
        sw_sn.append(_snippets(sosfiltfilt(sos_s, x), pk, half))
    raw_sn = np.concatenate(raw_sn); bp_sn = np.concatenate(bp_sn)
    sw_sn = np.concatenate(sw_sn)
    t = (np.arange(-half, half) / fs) * 1000.0

    fig = plt.figure(figsize=(11.0, 7.2))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32)

    # 1. grand average -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    m, se = raw_sn.mean(0), raw_sn.std(0)/np.sqrt(len(raw_sn))
    ax.plot(t, m, color=ACC_COL, lw=1.2, label="raw")
    ax.fill_between(t, m-se, m+se, color=ACC_COL, alpha=0.25, lw=0)
    ax.plot(t, sw_sn.mean(0), color=HC_COL, lw=1.2, label="8–40 Hz (sharp wave)")
    ax.axvline(0, color='0.6', ls=':', lw=0.8)
    ax.set_xlim(-250, 250); ax.set_xlabel("Time from ripple peak (ms)")
    ax.set_ylabel(r"Amplitude ($\mu$V)")
    ax.set_title(f"Grand average (n={len(raw_sn)})")
    ax.legend(frameon=False, loc="lower right")

    # 2. band-passed average ----------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, bp_sn.mean(0), color=ACC_COL, lw=1.0)
    ax.axvline(0, color='0.6', ls=':', lw=0.8)
    ax.set_xlim(-120, 120); ax.set_xlabel("Time from ripple peak (ms)")
    ax.set_ylabel(r"80–120 Hz ($\mu$V)")
    ax.set_title("Band-passed average")

    # 3. ripple-triggered TFR ---------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    freqs = np.arange(30, 181, 2.5)
    tfr = np.zeros((len(freqs), raw_sn.shape[1]))
    for j, f0 in enumerate(freqs):
        bw = max(2.0, f0/6.0*2)
        s = butter(3, [max(1, f0-bw/2)/(fs/2), min(fs/2-1, f0+bw/2)/(fs/2)],
                   btype='band', output='sos')
        tfr[j] = np.abs(hilbert(sosfiltfilt(s, raw_sn, axis=-1), axis=-1)).mean(0)
    base = tfr[:, (t < -300)].mean(1, keepdims=True)
    tfr_pct = (tfr - base) / base * 100
    im = ax.pcolormesh(t, freqs, tfr_pct, cmap="magma", shading="auto")
    ax.axhline(det.RIPPLE_BAND[0], color='w', ls=':', lw=0.7)
    ax.axhline(det.RIPPLE_BAND[1], color='w', ls=':', lw=0.7)
    ax.set_xlim(-250, 250); ax.set_xlabel("Time from ripple peak (ms)")
    ax.set_ylabel("Frequency (Hz)"); ax.set_title("Ripple-triggered TFR")
    plt.colorbar(im, ax=ax, label="% change")

    # 4. examples ----------------------------------------------------------
    # Twin axes: the band-passed ripple is ~1-2 uV while the raw trace swings
    # +-50 uV, so plotting both on one axis makes the ripple invisible.
    # Examples are the largest-amplitude events, which is what one inspects.
    order = np.argsort(np.abs(bp_sn).max(axis=1))[::-1]
    for k in range(3):
        ax = fig.add_subplot(gs[1, k])
        idx = order[k * max(1, len(order) // 30)]
        ax.plot(t, raw_sn[idx], color='0.45', lw=0.9)
        ax.set_xlim(-150, 150); ax.set_xlabel("Time (ms)")
        ax.set_ylabel(r"raw ($\mu$V)", color='0.45')
        ax.tick_params(axis='y', colors='0.45')
        ax2 = ax.twinx()
        ax2.plot(t, bp_sn[idx], color=HC_COL, lw=0.9)
        ax2.set_ylabel(r"80–120 Hz ($\mu$V)", color=HC_COL)
        ax2.tick_params(axis='y', colors=HC_COL)
        ax.axvline(0, color='0.6', ls=':', lw=0.8)
        ax.set_title(f"Example {k+1}", fontsize=9)

    # 5. peak frequency ----------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    ax.hist(passed.peak_freq_hz.dropna(), bins=np.arange(30, 185, 5),
            color=HC_COL, edgecolor='w', lw=0.4)
    ax.axvspan(*det.RIPPLE_BAND, color=ACC_COL, alpha=0.12)
    ax.set_xlabel("Peak frequency (Hz)"); ax.set_ylabel("Events")
    ax.set_title("Peak frequency")

    # 6. rate per derivation ----------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    g = qc[~qc.excluded & qc.rate_hz.notna()]
    ax.bar(range(len(g)), g.rate_hz, color=HC_COL, edgecolor='w')
    ax.axhspan(0.17, 0.24, color=ACC_COL, alpha=0.18, label="Chen 0.17–0.24 Hz")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.pair_id, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel("Ripple rate (Hz)"); ax.set_title("Rate per derivation")
    ax.legend(frameon=False, fontsize=7)

    # 7. rejection cascade -------------------------------------------------
    ax = fig.add_subplot(gs[2, 2])
    n_cand = int(qc.n_candidates.sum()) if 'n_candidates' in qc else len(events)
    stages = ["candidates", "dur+amp", "spectral"]
    vals = [n_cand, len(events), int(events.passed.sum())]
    ax.bar(stages, vals, color=[PAL[2], PAL[1], HC_COL], edgecolor='w')
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v}", ha='center', va='bottom', fontsize=8)
    ax.set_ylabel("Events"); ax.set_title("Rejection cascade")

    fig.suptitle(f"s{session:02d} ripple detection QC — "
                 f"{int(passed.shape[0])} ripples, "
                 f"{len(g)} derivations", fontsize=12, y=0.98)
    out = os.path.join(rip_dir, "qc_ripples")
    fig.savefig(out + ".png", dpi=300, bbox_inches='tight')
    fig.savefig(out + ".pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  saved -> {out}.png / .pdf")

    # numeric summary for the checkpoint
    print(f"  peak freq: median {passed.peak_freq_hz.median():.1f} Hz "
          f"(IQR {passed.peak_freq_hz.quantile(.25):.1f}–"
          f"{passed.peak_freq_hz.quantile(.75):.1f})")
    print(f"  duration : median {passed.duration_s.median()*1000:.0f} ms")
    print(f"  rate     : median {g.rate_hz.median():.3f} Hz across {len(g)} derivations")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(qc_report)
    else:
        qc_report(38)
