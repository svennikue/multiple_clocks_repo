#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event-locked ripple analysis following Sakon & Kahana (2022) and He et al. (2026).

Both papers solve the problem this project ran into -- that a rate contrast
between two conditions is contaminated by any baseline difference between them
-- and they solve it the same way: compare a window around the event with the
SAME window shifted earlier in the same trial. The state difference cancels
because it is present in both.

Parameters are theirs, not ours, so the design is not tuned on these data:

  Sakon & Kahana 2022 (PNAS 119:e2201657119)
    BIN_S        0.100     peri-event time histogram bin
    PRE_WIN      -0.6 to -0.1 s     "PRE", the pre-event window
    BASE_WIN     -1.6 to -1.1 s     the same window shifted 1 s earlier (Eq. 2)
    POST_WIN     +0.2 to +0.7 s     post-event window
    DEDUP_S      2.0       events within 2 s of another are dropped, so no
                           ripple is counted for two events
    Eq. 1  rate ~ condition + (condition|subject) + (condition|subject:session)
    Eq. 2  per subject: rate ~ window_indicator + (1|session), then a
           one-sample t-test of the per-subject t-scores against zero
    smoothing: 5-bin triangle, FOR VISUALISATION ONLY

  He et al. 2026 (Nat Neurosci 29:1711)
    PERI_WIN     -0.25 to +0.25 s
    NONPERI      -0.75 to -0.25 s and +0.25 to +0.75 s combined, which they
                 note gives an unbiased estimate because it is symmetric
    pre-peak vs post-peak split of the peri window
    LME with nested subject/contact random effects

Deviation from Sakon, deliberately: rate is events per ARTIFACT-FREE second,
not per wall-clock second. They do not have an artifact mask; we do, and a
window half-removed by artifact rejection offers half the opportunity to
observe a ripple. Windows below `MIN_CLEAN_FRAC` clean are dropped rather than
given an unstable rate.

@author: Svenja Kuchenhoff
"""

import warnings

import numpy as np
import pandas as pd

from mc.analyse.swr_windows import clean_exposure

# MixedLM warns on nearly every per-subject fit that a variance component sits
# on the boundary (i.e. that random effect is ~0). That is expected with few
# sessions per subject and does not invalidate the fixed effect, but it buries
# the results. Silenced locally at the fit, never globally.
_CONV = "ignore"

# --- Sakon & Kahana 2022
BIN_S = 0.100
PRE_WIN = (-0.6, -0.1)
BASE_WIN = (-1.6, -1.1)
POST_WIN = (0.2, 0.7)
DEDUP_S = 2.0
SMOOTH_BINS = 5

# --- He et al. 2026
PERI_WIN = (-0.25, 0.25)
NONPERI_WINS = ((-0.75, -0.25), (0.25, 0.75))

MIN_CLEAN_FRAC = 0.5      # ours: a window must be at least half artifact-free


def dedup_events(t, min_gap_s=DEDUP_S):
    """Drop events within `min_gap_s` of the previous one.

    Sakon's rule: "Recalls within 2 s of a previous recall were removed from
    consideration in order to avoid double-counting ripples." With a +-1.6 s
    analysis window, two events closer than that share ripples, and those
    ripples then appear as independent observations.
    """
    t = np.sort(np.asarray(t, float))
    t = t[np.isfinite(t)]
    if not t.size:
        return t
    keep = np.concatenate([[True], np.diff(t) >= min_gap_s])
    return t[keep]


def window_rate(event_t, ripple_t, intervals, win, min_clean_frac=MIN_CLEAN_FRAC):
    """Exposure-corrected ripple rate in `win` around each event.

    Returns (rate, n_ripples, exposure_s) with NaN where the window is more
    than `1 - min_clean_frac` artifact.
    """
    event_t = np.asarray(event_t, float)
    starts = event_t + win[0]
    stops = event_t + win[1]
    t = np.sort(np.asarray(ripple_t, float))
    n = (np.searchsorted(t, stops, side="right")
         - np.searchsorted(t, starts, side="left")).astype(float)
    expo = clean_exposure(intervals, starts, stops)
    dur = win[1] - win[0]
    bad = expo < min_clean_frac * dur
    rate = np.where(bad, np.nan, n / np.where(expo > 0, expo, np.nan))
    return rate, n, expo


def multi_window_rate(event_t, ripple_t, intervals, wins,
                      min_clean_frac=MIN_CLEAN_FRAC):
    """Rate over a UNION of windows -- He et al.'s non-peri-ripple estimate.

    Summing counts and exposure across the two flanking windows rather than
    averaging two rates, so the symmetric estimate is not dominated by whichever
    flank happens to be cleaner.
    """
    n_tot = np.zeros(len(event_t))
    e_tot = np.zeros(len(event_t))
    dur = 0.0
    for w in wins:
        _, n, e = window_rate(event_t, ripple_t, intervals, w, min_clean_frac=0.0)
        n_tot += n
        e_tot += e
        dur += w[1] - w[0]
    bad = e_tot < min_clean_frac * dur
    rate = np.where(bad, np.nan, n_tot / np.where(e_tot > 0, e_tot, np.nan))
    return rate, n_tot, e_tot


def peth(event_t, ripple_t, intervals, half_s=2.0, bin_s=BIN_S):
    """Peri-event time histogram, exposure-corrected, per event.

    Returns (centres, rate) with rate shaped (n_events, n_bins).
    """
    edges = np.arange(-half_s, half_s + bin_s / 2, bin_s)
    centres = edges[:-1] + bin_s / 2
    event_t = np.asarray(event_t, float)
    t = np.sort(np.asarray(ripple_t, float))
    out = np.full((len(event_t), len(centres)), np.nan)
    for k in range(len(centres)):
        starts = event_t + edges[k]
        stops = event_t + edges[k + 1]
        n = (np.searchsorted(t, stops, side="right")
             - np.searchsorted(t, starts, side="left")).astype(float)
        e = clean_exposure(intervals, starts, stops)
        out[:, k] = np.where(e > 0, n / e, np.nan)
    return centres, out


def triangle_smooth(y, n=SMOOTH_BINS):
    """Sakon's 5-bin triangle smoothing -- VISUALISATION ONLY, never for stats."""
    y = np.asarray(y, float)
    k = np.concatenate([np.arange(1, n // 2 + 2), np.arange(n // 2, 0, -1)])
    k = k / k.sum()
    pad = len(k) // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, k, mode="valid")


# =============================================================================
# THE TWO MODELS
# =============================================================================
# Sakon's Eq. 1 and Eq. 2 answer different questions and they say so
# explicitly: Eq. 1 asks whether the effect is STRONGER in one condition than
# another, Eq. 2 asks whether it RISES ABOVE that trial's own baseline. Eq. 2
# is the one that removes a between-condition baseline difference, because the
# comparison is within trial.

def fit_eq1(df, condition_col="condition", rate_col="rate",
            subject_col="subject_key", session_col="session"):
    """Sakon Eq. 1: rate ~ condition + (condition|subject) + (condition|subject:session).

    Random intercept AND slope for subject, plus session nested in subject.
    Returns a dict with the fixed effect of condition.
    """
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[rate_col, condition_col, subject_col]).copy()
    if d[condition_col].nunique() < 2 or len(d) < 20:
        return None
    lv = sorted(d[condition_col].unique())
    d["_g"] = (d[condition_col] == lv[1]).astype(float)   # 0 = first level
    d["_sess"] = d[session_col].astype(str)
    vc = {"sess": "0 + C(_sess)", "sess_g": "0 + C(_sess):_g"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(_CONV)
            md = smf.mixedlm(f"{rate_col} ~ _g", d, groups=d[subject_col],
                             re_formula="~_g", vc_formula=vc)
            res = md.fit(reml=True, method="nm", maxiter=2000)
    except Exception:
        try:                       # fall back to subject random intercept only
            md = smf.mixedlm(f"{rate_col} ~ _g", d, groups=d[subject_col])
            res = md.fit(reml=True, method="nm", maxiter=2000)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
    return {"levels": lv, "beta": float(res.params.get("_g", np.nan)),
            "se": float(res.bse.get("_g", np.nan)),
            "z": float(res.tvalues.get("_g", np.nan)),
            "p": float(res.pvalues.get("_g", np.nan)),
            "n_obs": int(len(d)),
            "n_subjects": int(d[subject_col].nunique()),
            "direction": f"positive = {lv[1]} > {lv[0]}"}


def fit_eq2(df, rate_event="rate_event", rate_base="rate_base",
            subject_col="subject_key", session_col="session",
            n_perm_sign=10000):
    """Sakon Eq. 2: per subject, event window vs its own baseline window.

    Per subject: rate ~ window_indicator + (1|session). Then a one-sample
    t-test of the per-subject t-scores against zero, which makes the SUBJECT
    the unit of inference rather than the trial.

    This is the test that survives a baseline difference between conditions,
    because both windows come from the same trial.
    """
    import statsmodels.formula.api as smf
    from scipy import stats as st

    long = []
    for w, col in (("base", rate_base), ("event", rate_event)):
        s = df[[subject_col, session_col, col]].rename(columns={col: "rate"})
        s["_w"] = 1.0 if w == "event" else 0.0
        long.append(s)
    d = pd.concat(long, ignore_index=True).dropna(subset=["rate"])
    d["_sess"] = d[session_col].astype(str)

    rows = []
    for subj, g in d.groupby(subject_col):
        if g["_w"].nunique() < 2 or len(g) < 10:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(_CONV)
                if g["_sess"].nunique() > 1:
                    res = smf.mixedlm("rate ~ _w", g, groups=g["_sess"]).fit(
                        reml=True, method="nm", maxiter=2000)
                else:
                    import statsmodels.api as sm
                    res = sm.OLS(g["rate"], sm.add_constant(g["_w"])).fit()
            rows.append({"subject_key": subj,
                         "t": float(res.tvalues.get("_w", np.nan)),
                         "beta": float(res.params.get("_w", np.nan)),
                         "n": int(len(g))})
        except Exception:
            continue
    per = pd.DataFrame(rows)
    if len(per) < 3:
        return {"error": f"only {len(per)} subjects fitted", "per_subject": per}
    tt = per["t"].dropna()
    t_stat, p = st.ttest_1samp(tt, 0.0)
    pm = perm_sign_flip(tt, n_perm=n_perm_sign)
    return {"per_subject": per, "n_subjects": int(len(tt)),
            "mean_t": float(tt.mean()), "t": float(t_stat), "p": float(p),
            "p_perm": pm.get("p_perm", np.nan), "perm": pm,
            "df": int(len(tt) - 1),
            "direction": "positive = event window above its own baseline"}


def perm_sign_flip(values, n_perm=10000, seed=42):
    """Non-parametric p for "is the per-subject mean different from zero".

    Sign-flipping is the exact permutation for a within-subject contrast: under
    the null, whether a subject's event window is above or below its own
    baseline is a coin flip, so flipping signs generates the null distribution
    without assuming normality. Two-sided.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return {"error": f"only {v.size} values"}
    rng = np.random.default_rng(seed)
    obs = float(v.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, v.size))
    null = (flips * v).mean(axis=1)
    p = float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + n_perm))
    return {"observed": obs, "p_perm": p, "n": int(v.size),
            "n_perm": int(n_perm), "null_sd": float(null.std()),
            "z": float((obs - null.mean()) / (null.std() + 1e-12))}


def sliding_window(rates_by_subject, bin_s=BIN_S, width_s=0.5):
    """Boxcar-average each subject's time course into a sliding window.

    Removes the window choice: instead of picking 0-0.5 s because that is where
    the peak looked biggest, every position is evaluated and the multiple
    comparisons across positions are handled by the cluster permutation. A
    `width_s` window is just a moving average of `width_s / bin_s` bins, so the
    whole family comes free from the peri-event histogram already computed.

    Returns (smoothed, valid_centre_index) -- positions where the full window
    fits are marked valid; the rest are NaN rather than edge-padded, because
    edge padding would invent data at exactly the extremes a reader inspects.
    """
    X = np.asarray(rates_by_subject, float)
    k = max(int(round(width_s / bin_s)), 1)
    out = np.full(X.shape, np.nan)
    if X.shape[1] < k:
        return out, np.zeros(X.shape[1], bool)
    ker = np.ones(k) / k
    half = k // 2
    for i in range(X.shape[0]):
        v = np.convolve(X[i], ker, mode="valid")
        out[i, half:half + len(v)] = v
    valid = np.zeros(X.shape[1], bool)
    valid[half:half + (X.shape[1] - k + 1)] = True
    return out, valid


def sliding_window_cluster(rates_by_subject, centres, bin_s=BIN_S,
                           width_s=0.5, n_perm=1000, seed=42, alpha=0.05):
    """Sliding-window test with cluster correction over window positions.

    `rates_by_subject` should already be baseline-subtracted, so each value is
    "this window minus this subject's own baseline" -- the Eq. 2 quantity,
    evaluated everywhere instead of at one chosen place.
    """
    X, valid = sliding_window(rates_by_subject, bin_s=bin_s, width_s=width_s)
    Xv = X[:, valid]
    if Xv.shape[1] < 2:
        return None
    t_obs, cl, pv, nullinfo = cluster_perm_time(Xv, n_perm=n_perm, seed=seed,
                                                alpha=alpha)
    c = np.asarray(centres, float)[valid]
    return {"centres": c, "t": t_obs, "width_s": width_s, "null": nullinfo,
            "n_subjects": int(Xv.shape[0]),
            "clusters": [{"t_start_s": float(c[a]), "t_stop_s": float(c[b - 1]),
                          "p": p, "peak_t": float(np.nanmax(np.abs(t_obs[a:b]))),
                          "peak_at_s": float(c[a + int(np.nanargmax(np.abs(t_obs[a:b])))])}
                         for (a, b), p in zip(cl, pv)]}


def cluster_perm_time(rates_by_subject, n_perm=1000, seed=42, alpha=0.05):
    """Cluster-based permutation over time bins (He et al.).

    `rates_by_subject` is (n_subjects, n_bins) of baseline-subtracted rate.
    Sign-flipping across subjects builds the null, which is the standard
    choice when the unit of inference is the subject.

    Returns (t_obs, clusters, p_values) where clusters are (start, stop) bin
    index pairs.
    """
    from scipy import stats as st
    X = np.asarray(rates_by_subject, float)
    rng = np.random.default_rng(seed)

    def _clusters(t, thr):
        sig = np.abs(t) > thr
        out, i = [], 0
        while i < len(sig):
            if sig[i]:
                j = i
                while j + 1 < len(sig) and sig[j + 1]:
                    j += 1
                out.append((i, j + 1))
                i = j + 1
            else:
                i += 1
        return out

    n = X.shape[0]
    thr = st.t.ppf(1 - alpha / 2, n - 1)
    t_obs = st.ttest_1samp(X, 0.0, nan_policy="omit").statistic
    t_obs = np.asarray(t_obs, float)
    obs = _clusters(t_obs, thr)
    obs_mass = [np.nansum(np.abs(t_obs[a:b])) for a, b in obs]

    null = np.empty(n_perm)
    for k in range(n_perm):
        sgn = rng.choice([-1.0, 1.0], size=(n, 1))
        tk = st.ttest_1samp(X * sgn, 0.0, nan_policy="omit").statistic
        tk = np.asarray(tk, float)
        cl = _clusters(tk, thr)
        null[k] = max([np.nansum(np.abs(tk[a:b])) for a, b in cl], default=0.0)

    p = [float((1 + np.sum(null >= m)) / (1 + n_perm)) for m in obs_mass]
    return t_obs, obs, p, {"null_mass": null, "obs_mass": obs_mass,
                           "threshold_t": float(thr)}
