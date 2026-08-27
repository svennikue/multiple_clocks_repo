#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aperiodic (1/f) surrogate control for ripple detection.

van Schalkwijk & Helfrich (2026, Nat Commun 17, s41467-026-68404-5) show that
**77% of awake ripples in the medial temporal lobe fall within the 1/f noise
floor** across five common detection algorithms, and -- critically for this
project -- that *task-related changes in the spectral exponent produce spurious
condition differences in ripple rate*. In their visual-search and motor data the
relationship between ripple density and condition disappeared once surrogate
density (which mirrors the 1/f change) was partialled out.

That is a direct threat to H1: if the planning and execution phases differ in
background 1/f, a ripple-rate difference between them could be entirely
aperiodic. Two things are therefore needed, and both live here:

  1. a per-derivation **noise floor** -- run the identical detector on colored
     noise matched to that derivation's own aperiodic spectrum;
  2. a per-window **spectral exponent**, to be carried as a covariate in the H1
     model rather than assumed constant.

This replaces the white-matter control originally planned. A distant
white-matter contact can pick up hippocampal ripples by volume conduction, so
its rate is not a clean false-positive floor; in the literature white matter is
used as a *reference* to subtract common noise (which the bipolar montage here
already does), not as a control channel.

Per CLAUDE.md rule 4, the surrogate is passed through the **same**
`swr_detect.detect_channel` used for the real data -- never a reimplementation.

@author: Svenja Kuchenhoff
"""

import numpy as np
from scipy.signal import welch

import mc.analyse.swr_detect as det

# Fitting range for the aperiodic component. Chosen to bracket the ripple band
# without including it: the fit must describe the background one would expect
# AT 80-120 Hz in the absence of ripples, so fitting only 1-45 Hz and
# extrapolating would be an assumption, and fitting through the ripple band
# would absorb the ripples themselves into the "background".
FIT_RANGE_HZ = (20.0, 200.0)
FIT_EXCLUDE_HZ = (75.0, 125.0)
N_SURROGATES = 5
SEED = 42

# A surrogate only has to yield a RATE, and a rate estimate converges long
# before a full session is simulated. Detection is dominated by a 69-frequency
# filter bank run over the whole signal, so simulating a 5000 s session cost
# ~17x more than necessary for the same number. At ~0.3 Hz, 300 s gives ~90
# events per surrogate, and N_SURROGATES of them, which is ample.
MAX_SURROGATE_S = 300.0


def fit_aperiodic(x, fs, fit_range=FIT_RANGE_HZ, exclude=FIT_EXCLUDE_HZ,
                  nperseg=2048):
    """Fit the aperiodic 1/f component. Returns (offset, exponent, r2).

    Uses FOOOF where available, falling back to a robust log-log linear fit
    (which is what the aperiodic term of FOOOF is, in `fixed` mode).
    """
    f, P = welch(np.asarray(x, float), fs=fs, nperseg=min(nperseg, len(x)))
    m = (f >= fit_range[0]) & (f <= fit_range[1]) & (f > 0)
    m &= ~((f >= exclude[0]) & (f <= exclude[1]))
    if m.sum() < 10:
        return np.nan, np.nan, np.nan
    ff, PP = f[m], P[m]

    try:
        from fooof import FOOOF
        fm = FOOOF(aperiodic_mode='fixed', verbose=False)
        fm.fit(ff, PP)
        off, expo = float(fm.aperiodic_params_[0]), float(fm.aperiodic_params_[1])
        return off, expo, float(fm.r_squared_)
    except Exception:
        lf, lp = np.log10(ff), np.log10(np.maximum(PP, 1e-30))
        A = np.vstack([np.ones_like(lf), -lf]).T
        coef, res, *_ = np.linalg.lstsq(A, lp, rcond=None)
        pred = A @ coef
        ss = 1 - np.sum((lp - pred) ** 2) / max(np.sum((lp - lp.mean()) ** 2), 1e-30)
        return float(coef[0]), float(coef[1]), float(ss)


def simulate_aperiodic(exponent, n_samples, fs, reference_signal=None,
                       fit_range=FIT_RANGE_HZ, exclude=FIT_EXCLUDE_HZ, rng=None):
    """Colored noise with the given 1/f exponent, scaled to the reference.

    Scaling matches the surrogate's power to the real signal **within the
    fitting range only** (ripple band excluded). The ripple-band power of the
    surrogate then follows from the fitted exponent rather than being copied
    from the data -- which is the whole point: it is the power one would expect
    there if no ripples existed.
    """
    rng = rng or np.random.default_rng(SEED)
    w = rng.standard_normal(n_samples)
    F = np.fft.rfft(w)
    f = np.fft.rfftfreq(n_samples, 1.0 / fs)
    f[0] = f[1] if len(f) > 1 else 1.0
    y = np.fft.irfft(F * f ** (-exponent / 2.0), n_samples)

    if reference_signal is not None:
        nps = min(2048, n_samples)
        fr, Pr = welch(np.asarray(reference_signal, float), fs=fs, nperseg=nps)
        fy, Py = welch(y, fs=fs, nperseg=nps)
        m = ((fr >= fit_range[0]) & (fr <= fit_range[1])
             & ~((fr >= exclude[0]) & (fr <= exclude[1])))
        if m.sum() > 3:
            y *= np.sqrt(np.median(Pr[m]) / max(np.median(Py[m]), 1e-30))
    return y.astype(np.float32)


def surrogate_noise_floor(x, fs, clean, n_surrogates=N_SURROGATES, seed=SEED,
                          verbose=False, n_observed=None, clean_s=None):
    """Expected ripple rate from aperiodic noise alone, for one derivation.

    Fits the derivation's own aperiodic spectrum on artifact-free data,
    simulates matched noise, and runs the **identical** detector on it.
    Returns observed rate, surrogate rate, and the implied false-positive
    fraction.
    """
    x = np.asarray(x, float)
    clean_x = x[clean] if clean is not None else x
    if clean_x.size < int(10 * fs):
        return {}

    off, expo, r2 = fit_aperiodic(clean_x, fs)
    if not np.isfinite(expo):
        return {}

    # Reuse the detection stage's result when the caller has it; re-detecting
    # costs a full 69-frequency filter bank over the session for no new
    # information.
    if clean_s is None:
        clean_s = float(np.sum(clean) / fs) if clean is not None else len(x) / fs
    if n_observed is None:
        ev_obs, _ = det.detect_channel(x, fs, clean)
        n_observed = int(ev_obs.passed.sum()) if len(ev_obs) else 0
    n_obs = int(n_observed)
    rate_obs = n_obs / clean_s if clean_s > 0 else np.nan

    rng = np.random.default_rng(seed)
    n_sur = int(min(len(clean_x), MAX_SURROGATE_S * fs))
    sur_rates = []
    for _ in range(n_surrogates):
        y = simulate_aperiodic(expo, n_sur, fs,
                               reference_signal=clean_x, rng=rng)
        # identical detector, identical settings; the surrogate is fully clean
        ev_s, _ = det.detect_channel(y, fs, np.ones(len(y), bool))
        n_s = int(ev_s.passed.sum()) if len(ev_s) else 0
        sur_rates.append(n_s / (len(y) / fs))

    sur = float(np.mean(sur_rates))
    out = {
        "aperiodic_offset": off, "aperiodic_exponent": expo, "aperiodic_r2": r2,
        "clean_s": clean_s, "n_observed": n_obs,
        "rate_observed_hz": rate_obs,
        "rate_surrogate_hz": sur,
        "rate_surrogate_sd": float(np.std(sur_rates)),
        "false_positive_frac": (sur / rate_obs) if rate_obs > 0 else np.nan,
        "rate_excess_hz": rate_obs - sur,
        "n_surrogates": n_surrogates,
    }
    if verbose:
        print(f"      chi={expo:.2f} (r2={r2:.2f})  observed={rate_obs:.3f} Hz  "
              f"surrogate={sur:.3f} Hz  -> FP {out['false_positive_frac']:.0%}")
    return out


def windowed_exponent(x, fs, windows, clean=None, min_s=5.0):
    """Aperiodic exponent per behavioural window, for use as an H1 covariate.

    `windows` is an (n, 2) array of [start_s, stop_s]. Returns one exponent per
    window (NaN where there is too little artifact-free data).
    """
    x = np.asarray(x, float)
    out = np.full(len(windows), np.nan)
    for i, (a, b) in enumerate(np.asarray(windows, float)):
        lo, hi = int(round(a * fs)), int(round(b * fs))
        lo, hi = max(0, lo), min(len(x), hi)
        if hi - lo < int(min_s * fs):
            continue
        seg = x[lo:hi]
        if clean is not None:
            seg = seg[clean[lo:hi]]
        if seg.size < int(min_s * fs):
            continue
        _, expo, _ = fit_aperiodic(seg, fs)
        out[i] = expo
    return out
