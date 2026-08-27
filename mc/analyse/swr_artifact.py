#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Artifact and IED rejection, following Chen et al. 2025.

The previous pipeline had none of this. In epilepsy patients interictal
epileptiform discharges produce exactly the sharp 80-120 Hz transients a ripple
detector keys on, so without rejection an unknown fraction of "ripples" are
IEDs. It is fatal for H2 in particular: epileptiform activity propagates
between hippocampus and frontal cortex, and would manufacture HC-mPFC coupling
out of nothing.

Chen's five criteria, each flagged where the metric exceeds **4 IQR above its
median** across the recording:

  1. absolute amplitude of the raw signal
  2. absolute amplitude of its first derivative (sharp gradients, i.e. IEDs)
  3. RMS of the signal high-passed above 250 Hz  (needs fs >= 1000; this is the
     reason the 500 Hz pipeline had to be abandoned)
  4. broadband power, summed over 1-60 Hz
  5. an automatic IED detector (Janca et al. 2015)

Then: every detection padded by +-1 s, artifact-free intervals shorter than 1 s
also marked bad, and contacts with more than two-thirds of the recording
contaminated excluded entirely.

@author: Svenja Kuchenhoff
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import uniform_filter1d, binary_dilation, label

IQR_K = 4.0
PAD_S = 1.0
MIN_CLEAN_S = 1.0
MAX_CONTAM_FRAC = 2.0 / 3.0

HF_HIGHPASS_HZ = 250.0
HF_RMS_WIN_MS = 100.0
BROADBAND_HZ = (1.0, 60.0)
IED_BAND_HZ = (10.0, 60.0)
IED_K = 3.65                 # Janca et al. 2015 threshold multiplier
IED_SEG_S = 5.0


def iqr_threshold(v, k=IQR_K):
    """median + k * IQR. Robust to the very outliers being detected, unlike
    mean + k*SD which they inflate."""
    v = np.asarray(v, float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return np.inf
    q1, q3 = np.percentile(finite, [25, 75])
    return float(np.median(finite) + k * (q3 - q1))


def _envelope(x, fs, lo, hi, order=4):
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band', output='sos')
    return np.abs(hilbert(sosfiltfilt(sos, x)))


# =============================================================================
# THE FIVE CRITERIA
# =============================================================================

def criterion_raw_amplitude(x, fs):
    return np.abs(x)


def criterion_first_derivative(x, fs):
    d = np.abs(np.diff(x, prepend=x[..., :1]))
    return d


def criterion_hf_rms(x, fs, hp=HF_HIGHPASS_HZ, win_ms=HF_RMS_WIN_MS):
    """RMS of the >250 Hz signal. Impossible below fs = 1000 Hz."""
    if hp >= fs / 2:
        return np.zeros_like(x)
    sos = butter(4, hp / (fs / 2), btype='high', output='sos')
    hf = sosfiltfilt(sos, x)
    w = max(1, int(round(win_ms * fs / 1000.0)))
    return np.sqrt(uniform_filter1d(hf ** 2, w, axis=-1, mode='nearest'))


BROADBAND_N_FREQS = 30


def criterion_broadband_power(x, fs, band=BROADBAND_HZ, decim_fs=200.0,
                              n_freqs=BROADBAND_N_FREQS):
    """Summed 1-60 Hz power, **log-transformed and z-scored per frequency**.

    Chen: "the sum power across 30 logarithmically spaced frequencies between
    1 and 60 Hz ... log-transformed and z-scored per frequency". The
    log-transform matters enormously and is not optional: band power is
    strongly right-skewed, so thresholding it raw at 4 IQR flags many times
    more samples than intended. Measured here, the raw version flagged up to
    8.9% of samples which, after +-1 s padding, removed over half the
    recording.

    Computed on a ~200 Hz grid: the criterion targets slow broad-spectrum
    increases and does not need 1 kHz resolution.
    """
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(fs), int(decim_fs))
    y = resample_poly(np.asarray(x, float), int(decim_fs) // g, int(fs) // g,
                      axis=-1)

    freqs = np.logspace(np.log10(band[0]), np.log10(band[1]), n_freqs)
    win = max(1, int(round(0.5 * decim_fs)))
    total = np.zeros(y.shape[-1])
    for f0 in freqs:
        bw = max(1.0, f0 / 3.5)                     # ~7-cycle equivalent
        lo, hi = max(0.5, f0 - bw / 2), min(decim_fs / 2 - 1.0, f0 + bw / 2)
        if hi <= lo:
            continue
        env = _envelope(y, decim_fs, lo, hi)
        p = uniform_filter1d(env ** 2, win, axis=-1, mode='nearest')
        lp = np.log(np.maximum(p, 1e-30))           # log BEFORE z-scoring
        sd = np.std(lp)
        total += (lp - np.mean(lp)) / (sd if sd > 0 else 1.0)

    idx = np.linspace(0, total.shape[-1] - 1, np.shape(x)[-1])
    return np.interp(idx, np.arange(total.shape[-1]), total)


def detect_ied_janca(x, fs, band=IED_BAND_HZ, k=IED_K, seg_s=IED_SEG_S):
    """Janca et al. 2015 envelope-distribution IED detector.

    Bandpass 10-60 Hz, Hilbert envelope, then model the envelope within
    overlapping segments as log-normal and threshold at `mode + k * sigma` of
    that fit. Adaptive to each segment's background, so it does not simply
    re-find whatever the amplitude criteria already flagged.
    """
    env = _envelope(x, fs, band[0], band[1])
    n = env.shape[-1]
    seg = max(int(round(seg_s * fs)), 16)
    step = seg // 2
    thr = np.full(n, np.inf)

    for start in range(0, n, step):
        stop = min(start + seg, n)
        e = env[start:stop]
        e = e[e > 0]
        if e.size < 16:
            continue
        le = np.log(e)
        mu, sd = float(np.mean(le)), float(np.std(le))
        if not np.isfinite(sd) or sd == 0:
            continue
        # mode of a log-normal is exp(mu - sd^2); sigma in the original units
        mode = np.exp(mu - sd ** 2)
        sigma = np.sqrt((np.exp(sd ** 2) - 1) * np.exp(2 * mu + sd ** 2))
        thr[start:stop] = np.minimum(thr[start:stop], mode + k * sigma)

    return env > thr


# =============================================================================
# MASK ASSEMBLY
# =============================================================================

def artifact_mask(x, fs, pad_s=PAD_S, min_clean_s=MIN_CLEAN_S, k=IQR_K):
    """Boolean bad-sample mask for one channel, plus a per-criterion breakdown."""
    x = np.asarray(x, dtype=float)
    per = {}

    for name, metric in (("raw_amplitude", criterion_raw_amplitude(x, fs)),
                         ("first_derivative", criterion_first_derivative(x, fs)),
                         ("hf_rms_250", criterion_hf_rms(x, fs)),
                         ("broadband_1_60", criterion_broadband_power(x, fs))):
        per[name] = metric > iqr_threshold(metric, k)

    per["ied_janca"] = detect_ied_janca(x, fs)

    bad = np.zeros(x.shape[-1], bool)
    for v in per.values():
        bad |= v

    if pad_s > 0:
        w = int(round(pad_s * fs))
        bad = binary_dilation(bad, structure=np.ones(2 * w + 1, bool))

    # artifact-free islands shorter than min_clean_s are unusable
    if min_clean_s > 0:
        lab, n = label(~bad)
        if n:
            sizes = np.bincount(lab.ravel())
            too_short = np.where(sizes < int(round(min_clean_s * fs)))[0]
            too_short = too_short[too_short != 0]
            if too_short.size:
                bad |= np.isin(lab, too_short)

    stats = {name: float(v.mean()) for name, v in per.items()}
    stats["combined_after_pad"] = float(bad.mean())
    return bad, stats


def clean_intervals(bad, fs, t0=0.0):
    """(n, 2) array of artifact-free [start, stop] in seconds."""
    lab, n = label(~bad)
    if not n:
        return np.zeros((0, 2))
    out = []
    for i in range(1, n + 1):
        idx = np.flatnonzero(lab == i)
        out.append([t0 + idx[0] / fs, t0 + (idx[-1] + 1) / fs])
    return np.asarray(out)


class CleanAxis:
    """Bijection between wall seconds and artifact-free ("clean") seconds.

    Both the exposure offset in the GLM and the circular-shift permutation
    operate on this axis, so a shifted null can never place an event inside an
    artifact -- which would otherwise make the null easier to beat than the
    data.
    """

    def __init__(self, intervals):
        self.iv = np.asarray(intervals, float).reshape(-1, 2)
        dur = np.diff(self.iv, axis=1).ravel() if len(self.iv) else np.zeros(0)
        self.cum = np.concatenate([[0.0], np.cumsum(dur)])
        self.total = float(self.cum[-1]) if len(self.cum) else 0.0

    def to_clean(self, t):
        """Wall seconds -> clean seconds. NaN if t falls inside an artifact."""
        t = np.atleast_1d(np.asarray(t, float))
        out = np.full(t.shape, np.nan)
        for i, (a, b) in enumerate(self.iv):
            m = (t >= a) & (t < b)
            out[m] = self.cum[i] + (t[m] - a)
        return out

    def to_wall(self, c):
        """Clean seconds -> wall seconds (wraps modulo the clean total)."""
        c = np.mod(np.atleast_1d(np.asarray(c, float)), max(self.total, 1e-9))
        i = np.clip(np.searchsorted(self.cum, c, side='right') - 1,
                    0, max(len(self.iv) - 1, 0))
        return self.iv[i, 0] + (c - self.cum[i])
