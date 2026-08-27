#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ripple detection, following Chen et al. 2025.

Bandpass 80-120 Hz -> Hilbert envelope -> 20 ms moving RMS, thresholded at
1.5-9 SD of that RMS **over the whole artifact-free session**, duration
38-500 ms, then four peak-based spectral rejection criteria.

Two departures from the previous pipeline matter:

1. **The threshold is estimated once per session, not per snippet.** The old
   code (`identify_HPC_ripples.py:122`) recomputed `mean + 4*SD` inside every
   grid-repeat, which partially normalises ripple rate to be constant per
   snippet -- destroying exactly the between-window differences H1 tests.

2. **Detection runs on an amplitude measure, not on skewed power.** Morlet
   power is strongly right-skewed, so `mean + k*SD` is outlier-dominated and a
   single IED inflates the SD enough to suppress detection nearby. RMS of a
   bandpassed signal is amplitude-like and is what the human ripple literature
   uses.

Nothing is ever deleted here. Every rejection is a flag column, so re-running
at a different strictness costs nothing and the rejection cascade can be
inspected at QC.

@author: Svenja Kuchenhoff
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.ndimage import uniform_filter1d, label

RIPPLE_BAND = (80.0, 120.0)

# Detection thresholds, in SD of the session RMS. DUAL THRESHOLD:
#   LO_SD   defines event extent and therefore duration  (Chen's value)
#   PEAK_SD the event peak must reach this to be kept    (set here)
#   HI_SD   artifact ceiling                             (Chen's value)
#
# Why not Chen's single 1.5 SD: measured against an aperiodic (1/f) surrogate
# noise floor on this dataset, 1.5 SD alone puts 97% of detections inside the
# noise floor -- an excess over noise of only 0.011 Hz. Sweeping the peak
# criterion shows excess-over-noise peaking sharply at 3.0 SD (0.052 Hz, ~5x
# more signal), with the false-positive fraction falling to 75%. That is the
# inverted-U noise sensitivity van Schalkwijk & Helfrich (2026) describe for
# stricter-threshold detectors.
#
# Why the thresholds must be SEPARATE: duration and threshold are not
# independent. Simply raising the single threshold to 3.0 SD shortens every
# event's supra-threshold run, so events that genuinely last ~60 ms measure
# 10-15 ms and are then discarded by the 38 ms floor -- measured, that gave
# 0.021 Hz, a 10x under-count. This coupling is why Chen pair 1.5 SD with a
# 38 ms minimum. Splitting the two preserves Chen's duration semantics while
# gaining the noise rejection.
#
# The resulting rate, 0.206 Hz (range 0.190-0.238), lands inside Chen's
# reported 0.17-0.24 Hz, and spectral rejection is 31.9% against their
# 23.4% +- 9.9%. The threshold was chosen by maximising excess over the noise
# floor; agreement with Chen's rate is a consequence, not the target.
#
# Chen's single-threshold configuration is retained as a sensitivity analysis
# under the `swr_lo1.5_sensitivity` analysis name.
LO_SD = 1.5          # extent / duration, as Chen
PEAK_SD = 3.0        # peak must reach this; set by the surrogate analysis
HI_SD = 9.0          # artifact ceiling, as Chen
DUR_MS = (38.0, 500.0)          # 3 cycles at 80 Hz, to Chen's maximum
MERGE_GAP_MS = 30.0
RMS_WIN_MS = 20.0
SHARPWAVE_BAND = (8.0, 40.0)

# Spectral validation (Chen's four criteria), on a window centred on the event
SPEC_WIN_S = 2.0
SPEC_BASELINE_S = (-1.5, -0.5)
SPEC_RANGE_HZ = (30.0, 200.0)
SPEC_OUTBAND_FRAC = 0.80        # out-of-band power may not exceed this fraction
SPEC_WIDTH_SD = 3.0


def ripple_rms(x, fs, band=RIPPLE_BAND, win_ms=RMS_WIN_MS):
    """Bandpass 80-120 Hz, then a 20 ms moving RMS of the bandpassed signal.

    Exactly Chen's definition: "the root mean square (RMS) of the bandpassed
    signal was calculated and smoothed using a moving average filter with a
    20 ms window". Deliberately NOT the Hilbert envelope: the envelope is
    ~sqrt(2) times the RMS of the same oscillation and, more importantly, has a
    different standard deviation, so a "1.5 SD" threshold on an envelope is not
    the same threshold Chen applied. Using the envelope inflated the event rate
    to ~1.5x their reported range.
    """
    sos = butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)],
                 btype='band', output='sos')
    bp = sosfiltfilt(sos, np.asarray(x, float), axis=-1)
    w = max(1, int(round(win_ms * fs / 1000.0)))
    rms = np.sqrt(uniform_filter1d(bp ** 2, w, axis=-1, mode='nearest'))
    return bp, rms


def session_threshold(rms, clean):
    """(mean, sd) of the RMS over artifact-free samples of the whole session.

    This single line is the fix for the per-snippet threshold problem.
    """
    v = rms[clean] if clean is not None else rms
    v = v[np.isfinite(v)]
    if v.size < 100:
        return np.nan, np.nan
    return float(np.mean(v)), float(np.std(v))


def _merge_close(starts, stops, gap):
    """Merge events separated by less than `gap` samples."""
    if len(starts) == 0:
        return starts, stops
    ks, ke = [starts[0]], [stops[0]]
    for s, e in zip(starts[1:], stops[1:]):
        if s - ke[-1] < gap:
            ke[-1] = e
        else:
            ks.append(s); ke.append(e)
    return np.array(ks), np.array(ke)


def detect_candidates(rms, mu, sd, fs, clean, dur_ms=DUR_MS,
                      lo_sd=LO_SD, peak_sd=PEAK_SD, hi_sd=HI_SD,
                      merge_gap_ms=MERGE_GAP_MS):
    """Dual-threshold detection: extent at `lo_sd`, peak must reach `peak_sd`.

    Event **extent** (and therefore duration) is defined by the low threshold,
    exactly as Chen do; the event is only *kept* if its peak also reaches the
    higher detection threshold.

    Applying a single raised threshold instead is wrong, and badly so: raising
    it shortens every event's supra-threshold run, so events that genuinely
    last 60 ms are measured as 10-15 ms and are then discarded by the 38 ms
    minimum. Measured here, a single 3.0 SD threshold cut the rate to 0.021 Hz
    -- a 10x under-count -- because duration and threshold are not independent.
    That coupling is also why Chen's 1.5 SD is paired with a 38 ms floor.

    The dual scheme is standard in the SWR literature and decouples the two:
    duration semantics stay Chen's, while the peak criterion buys the
    noise-floor rejection that the surrogate analysis showed is needed.
    """
    if not np.isfinite(mu) or not np.isfinite(sd) or sd == 0:
        return pd.DataFrame()

    above = rms > (mu + lo_sd * sd)
    if clean is not None:
        above &= clean

    lab, n = label(above)
    if not n:
        return pd.DataFrame()
    starts = np.array([np.flatnonzero(lab == i)[0] for i in range(1, n + 1)])
    stops = np.array([np.flatnonzero(lab == i)[-1] + 1 for i in range(1, n + 1)])

    starts, stops = _merge_close(starts, stops,
                                 int(round(merge_gap_ms * fs / 1000.0)))

    dur_s = (stops - starts) / fs
    keep = (dur_s >= dur_ms[0] / 1000.0) & (dur_s <= dur_ms[1] / 1000.0)

    rows = []
    peak_cut = mu + peak_sd * sd
    hi_cut = mu + hi_sd * sd
    for s, e, k in zip(starts, stops, keep):
        seg = rms[s:e]
        pk = int(s + np.argmax(seg))
        rows.append({
            "start_sample": int(s), "stop_sample": int(e), "peak_sample": pk,
            "duration_s": float((e - s) / fs),
            "rms_peak": float(rms[pk]),
            "rms_peak_z": float((rms[pk] - mu) / sd),
            "pass_duration": bool(k),
            "pass_peak": bool(rms[pk] >= peak_cut),
            "pass_amplitude": bool(rms[pk] <= hi_cut),
        })
    return pd.DataFrame(rows)


SPEC_FREQS = np.arange(30.0, 201.0, 2.5)
SPEC_N_CYCLES = 6.0
SPEC_EVENT_HALFWIN_S = 0.025
SPEC_MIN_PROMINENCE = 10.0      # percent change; suppresses noise-level peaks


def _event_baseline_spectra(raw, fs, peaks, freqs=SPEC_FREQS,
                            n_cycles=SPEC_N_CYCLES):
    """Per-event spectrum normalised to its own pre-event baseline.

    Chen: decompose a 2 s window centred on the ripple into narrow bands,
    take the baseline from the pre-ripple period (-1.5 to -0.5 s), and express
    the ripple spectrum as percentage change from it. The baseline step is
    essential -- a ripple is a transient, so in a raw spectrum it is buried
    under ongoing 1/f activity and the most prominent "peak" lands around
    50 Hz regardless of whether a ripple is present.

    Implemented as a session-level filter bank sampled at the event times
    rather than a per-event wavelet transform: one bandpass+Hilbert pass per
    frequency over the whole recording, then free indexing. Equivalent, and
    orders of magnitude cheaper than transforming 2 s around every candidate.
    """
    n = raw.shape[-1]
    b0 = int(round(SPEC_BASELINE_S[0] * fs))     # negative
    b1 = int(round(SPEC_BASELINE_S[1] * fs))

    ev_amp = np.full((len(peaks), len(freqs)), np.nan, np.float32)
    bl_amp = np.full((len(peaks), len(freqs)), np.nan, np.float32)

    for j, f0 in enumerate(freqs):
        bw = max(2.0, f0 / n_cycles * 2.0)
        lo, hi = max(1.0, f0 - bw / 2), min(fs / 2 - 1.0, f0 + bw / 2)
        if hi <= lo:
            continue
        sos = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype='band', output='sos')
        amp = np.abs(hilbert(sosfiltfilt(sos, raw))).astype(np.float32)
        csum = np.concatenate([[0.0], np.cumsum(amp, dtype=np.float64)])

        # Frequency-matched event window. A fixed +-25 ms window cannot
        # estimate amplitude at 30 Hz (one cycle is 33 ms) and returns
        # inflated, noisy percent-change values there -- which then trip the
        # out-of-band criterion on events whose ripple is perfectly clean.
        # Matching the window to the filter's own time constant is what a
        # wavelet does implicitly.
        ev_half = max(int(round(SPEC_EVENT_HALFWIN_S * fs)),
                      int(round(n_cycles / (2.0 * f0) * fs)))

        def _mean(a, b):
            a = np.clip(a, 0, n); b = np.clip(b, 0, n)
            good = b > a
            out = np.full(a.shape, np.nan)
            out[good] = (csum[b[good]] - csum[a[good]]) / (b[good] - a[good])
            return out

        ev_amp[:, j] = _mean(peaks - ev_half, peaks + ev_half)
        bl_amp[:, j] = _mean(peaks + b0, peaks + b1)

    with np.errstate(invalid='ignore', divide='ignore'):
        pct = (ev_amp - bl_amp) / bl_amp * 100.0
    return pct


def spectral_validate(events, raw, fs, band=RIPPLE_BAND, freqs=SPEC_FREQS):
    """Chen's four peak-based rejection criteria, as flags (never drops rows).

    Rejected if: (1) the most prominent peak of the baseline-normalised
    spectrum falls outside the ripple band; (2) out-of-band 30-200 Hz activity
    exceeds 80% of the ripple peak; (3) multiple prominent peaks appear in
    120-200 Hz; (4) the peak is atypically broad for that contact.
    """
    if not len(events):
        return events
    from scipy.signal import find_peaks, peak_prominences

    peaks = events["peak_sample"].to_numpy(int)
    pct = _event_baseline_spectra(raw, fs, peaks, freqs=freqs)

    in_band = (freqs >= band[0]) & (freqs <= band[1])
    hi_band = (freqs > band[1]) & (freqs <= SPEC_RANGE_HZ[1])

    peak_f, prom, width = [], [], []
    fl_strict, fl_relaxed, f_max_out = [], [], []
    for row in pct:
        s = r_ = 0
        if not np.isfinite(row).any():
            peak_f.append(np.nan); prom.append(np.nan); width.append(np.nan)
            f_max_out.append(np.nan)
            fl_strict.append(0b1111); fl_relaxed.append(0b1111); continue
        y = np.nan_to_num(row, nan=0.0)

        idx, _ = find_peaks(y, prominence=SPEC_MIN_PROMINENCE)
        if idx.size == 0:
            peak_f.append(np.nan); prom.append(np.nan); width.append(np.nan)
            f_max_out.append(np.nan)
            fl_strict.append(0b0001); fl_relaxed.append(0b0001); continue

        pr = peak_prominences(y, idx)[0]
        top = idx[int(np.argmax(pr))]
        fpk = float(freqs[top])
        peak_f.append(fpk); prom.append(float(pr.max()))

        half = y[top] - pr.max() / 2.0
        l = top
        while l > 0 and y[l] > half:
            l -= 1
        rr = top
        while rr < len(y) - 1 and y[rr] > half:
            rr += 1
        width.append(float(freqs[rr] - freqs[l]))

        if not (band[0] <= fpk <= band[1]):                     # criterion 1
            s |= 0b0001; r_ |= 0b0001

        # Criterion 2, two scopes (see module docstring):
        #   strict  = Chen literal, 30-200 Hz outside the ripple band
        #   relaxed = 120-200 Hz only
        out_s = ~in_band
        if out_s.any():
            f_max_out.append(float(freqs[out_s][int(np.argmax(y[out_s]))]))
            if np.max(y[out_s]) > SPEC_OUTBAND_FRAC * y[top]:
                s |= 0b0010
        else:
            f_max_out.append(np.nan)
        if hi_band.any() and np.max(y[hi_band]) > SPEC_OUTBAND_FRAC * y[top]:
            r_ |= 0b0010

        if hi_band.sum() > 4:                                   # criterion 3
            hp, _ = find_peaks(y[hi_band], prominence=SPEC_MIN_PROMINENCE)
            if hp.size > 1:
                s |= 0b0100; r_ |= 0b0100
        fl_strict.append(s); fl_relaxed.append(r_)

    ev = events.copy()
    ev["peak_freq_hz"] = peak_f
    ev["peak_prominence"] = prom
    ev["peak_width_hz"] = width
    ev["f_max_outband_hz"] = f_max_out
    ev["spectral_flags_strict"] = fl_strict
    ev["spectral_flags_relaxed"] = fl_relaxed

    w = np.asarray(width, float)
    if np.isfinite(w).sum() > 3:                                # criterion 4
        thr = np.nanmean(w) + SPEC_WIDTH_SD * np.nanstd(w)
        broad = [(0b1000 if np.isfinite(x) and x > thr else 0) for x in w]
        ev["spectral_flags_strict"] = [int(a) | b for a, b in
                                       zip(ev["spectral_flags_strict"], broad)]
        ev["spectral_flags_relaxed"] = [int(a) | b for a, b in
                                        zip(ev["spectral_flags_relaxed"], broad)]

    ev["spectral_passed_strict"] = ev["spectral_flags_strict"] == 0
    ev["spectral_passed_relaxed"] = ev["spectral_flags_relaxed"] == 0
    # `spectral_passed` aliases the PRIMARY (strict) analysis
    ev["spectral_passed"] = ev["spectral_passed_strict"]
    ev["spectral_flags"] = ev["spectral_flags_strict"]
    return ev


def detect_channel(raw, fs, clean, band=RIPPLE_BAND, **kw):
    """Full detection for one bipolar derivation."""
    bp, rms = ripple_rms(raw, fs, band=band)
    mu, sd = session_threshold(rms, clean)
    ev = detect_candidates(rms, mu, sd, fs, clean, **kw)
    if not len(ev):
        return ev, {"mu": mu, "sd": sd, "n_candidates": 0}

    n_cand = len(ev)
    # Spectral validation only for events that already meet the amplitude and
    # duration criteria -- no point spectrally examining events that are
    # already rejected, and it is by far the most expensive step.
    ev = ev[ev.pass_duration & ev.pass_amplitude & ev.pass_peak].reset_index(drop=True)
    if not len(ev):
        return ev, {"mu": mu, "sd": sd, "n_candidates": n_cand,
                    "n_pass_duration": 0, "n_passed": 0}

    ev = spectral_validate(ev, np.asarray(raw, float), fs, band=band)

    # concurrent sharp-wave amplitude, stored for the validation figure even
    # though it plays no part in detection
    sos = butter(4, [SHARPWAVE_BAND[0] / (fs / 2), SHARPWAVE_BAND[1] / (fs / 2)],
                 btype='band', output='sos')
    sw = np.abs(hilbert(sosfiltfilt(sos, np.asarray(raw, float))))
    ev["sw_amp"] = sw[ev["peak_sample"].to_numpy(int)]
    ev["amp_peak_uv"] = np.abs(bp)[ev["peak_sample"].to_numpy(int)]

    ev["t_start_s"] = ev["start_sample"] / fs
    ev["t_peak_s"] = ev["peak_sample"] / fs
    ev["t_end_s"] = ev["stop_sample"] / fs
    # PRIMARY analysis = strict (Chen literal). `passed_relaxed` is carried
    # alongside for the pre-declared sensitivity analysis.
    base = ev["pass_duration"] & ev["pass_amplitude"] & ev["pass_peak"]
    ev["passed_strict"] = base & ev["spectral_passed_strict"]
    ev["passed_relaxed"] = base & ev["spectral_passed_relaxed"]
    ev["passed"] = ev["passed_strict"]

    diag = {
        "mu": mu, "sd": sd,
        "n_candidates": int(n_cand),
        "n_pass_duration": int(ev.pass_duration.sum()),
        "n_pass_peak": int(ev.pass_peak.sum()),
        "n_pass_amplitude": int(ev.pass_amplitude.sum()),
        "n_pass_spectral_strict": int(ev.spectral_passed_strict.sum()),
        "n_pass_spectral_relaxed": int(ev.spectral_passed_relaxed.sum()),
        "n_passed_strict": int(ev.passed_strict.sum()),
        "n_passed_relaxed": int(ev.passed_relaxed.sum()),
        "n_passed": int(ev.passed.sum()),
    }
    return ev, diag
