#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploration probes: descriptive checks on what actually drives ripple rate here.

The question behind the project is whether hippocampal ripples are how the
hippocampus talks to cortex to assemble an action plan. Before any of that can
be tested, three things have to be understood about this dataset:

  * what ripple rate does around a BUTTON PRESS, since every task event here is
    a press and movement is known to suppress ripples;
  * what it does with time standing still, since "thinking" periods are the
    only rest-like state the task has;
  * whether it drifts with time in the session, which would manufacture
    differences between early and late conditions.

Each probe is cheap, returns a tidy table, and makes no inferential claim. They
exist so a later contrast can be chosen for a reason rather than by default.

@author: Svenja Kuchenhoff
"""

import numpy as np
import pandas as pd

import mc.analyse.swr_behaviour as swb
import mc.analyse.swr_sakon as sk


def press_times(session, beh, data_root=None):
    """All key-press times in a session, split into movement and uncover.

    Returns (move_t, uncover_t) in session seconds. Both come from the 25 ms
    button series, so they are on the same clock as the ripples.
    """
    mv, un = [], []
    for grid, g in beh.groupby("grid_no"):
        btn = swb._grid_series(session, grid, "buttons", data_root)
        if btn is None:
            continue
        onset = float(g.new_grid_onset.iloc[0])
        b = btn.astype(str)
        trans = np.flatnonzero(b[1:] != b[:-1]) + 1
        for t in trans:
            tt = onset + t * swb.BIN_S
            if b[t] in swb.MOVE_KEYS:
                mv.append(tt)
            elif b[t] == swb.UNCOVER_KEY:
                un.append(tt)
    return np.sort(np.asarray(mv, float)), np.sort(np.asarray(un, float))


def still_periods(move_t, uncover_t, min_s=0.5):
    """Gaps between consecutive presses of ANY kind -- the task's rest periods.

    Returns a frame with start, stop, duration. A "still" period here means no
    key was pressed; the subject is looking at the screen and, presumably,
    thinking.
    """
    t = np.sort(np.concatenate([move_t, uncover_t]))
    if t.size < 2:
        return pd.DataFrame()
    d = np.diff(t)
    keep = d >= min_s
    return pd.DataFrame({"start_s": t[:-1][keep], "end_s": t[1:][keep],
                         "duration_s": d[keep]})


def peri_event_summary(event_t, ripple_t, intervals, label=""):
    """He et al.'s peri vs non-peri comparison for one event set.

    Returns one row per event with the peri rate, the symmetric non-peri rate
    and their difference, so the caller can aggregate at whatever level the
    unit of inference demands.
    """
    t = sk.dedup_events(event_t)
    if not t.size:
        return pd.DataFrame()
    r_peri, n_peri, e_peri = sk.window_rate(t, ripple_t, intervals, sk.PERI_WIN)
    r_non, n_non, e_non = sk.multi_window_rate(t, ripple_t, intervals,
                                               sk.NONPERI_WINS)
    return pd.DataFrame({"event_t": t, "label": label,
                         "rate_peri": r_peri, "rate_nonperi": r_non,
                         "diff": r_peri - r_non,
                         "n_peri": n_peri, "exposure_peri_s": e_peri})


def rate_vs_still_duration(still, ripple_t, intervals, edges=None):
    """Ripple rate inside still periods, binned by how long the stillness lasts.

    If ripples index an offline/planning state, longer stillness should carry a
    higher rate. If they merely track "not moving", the relationship should be
    flat once the window is long enough to be reliable.
    """
    if not len(still):
        return pd.DataFrame()
    edges = edges if edges is not None else np.array([0.5, 1, 2, 4, 8, 1e9])
    from mc.analyse.swr_windows import clean_exposure
    starts = still.start_s.to_numpy(float)
    stops = still.end_s.to_numpy(float)
    t = np.sort(np.asarray(ripple_t, float))
    n = (np.searchsorted(t, stops, side="right")
         - np.searchsorted(t, starts, side="left")).astype(float)
    e = clean_exposure(intervals, starts, stops)
    d = still.duration_s.to_numpy(float)
    b = np.digitize(d, edges) - 1
    out = []
    for k in range(len(edges) - 1):
        m = b == k
        if not m.any() or e[m].sum() <= 0:
            continue
        out.append({"bin_lo_s": edges[k], "bin_hi_s": edges[k + 1],
                    "n_periods": int(m.sum()), "n_ripples": float(n[m].sum()),
                    "exposure_s": float(e[m].sum()),
                    "rate_hz": float(n[m].sum() / e[m].sum())})
    return pd.DataFrame(out)


def rate_vs_session_time(ripple_t, intervals, n_bins=10):
    """Ripple rate across the session, to expose drift.

    A condition that happens early (exploration) and one that happens late
    (execution) will differ in rate for no interesting reason if the rate
    drifts. This is the check that has to come before any such contrast.
    """
    iv = np.asarray(intervals, float).reshape(-1, 2)
    if not len(iv):
        return pd.DataFrame()
    t0, t1 = iv[:, 0].min(), iv[:, 1].max()
    edges = np.linspace(t0, t1, n_bins + 1)
    from mc.analyse.swr_windows import clean_exposure
    t = np.sort(np.asarray(ripple_t, float))
    n = (np.searchsorted(t, edges[1:], side="right")
         - np.searchsorted(t, edges[:-1], side="left")).astype(float)
    e = clean_exposure(iv, edges[:-1], edges[1:])
    return pd.DataFrame({"bin": np.arange(n_bins),
                         "frac_through": (np.arange(n_bins) + 0.5) / n_bins,
                         "n_ripples": n, "exposure_s": e,
                         "rate_hz": np.where(e > 0, n / e, np.nan)})


def long_vs_short_still(still, ripple_t, intervals, short=(0.5, 2.0),
                        long_min=8.0):
    """Rate in LONG still periods vs short ones, as one row per derivation.

    Split out from `rate_vs_still_duration` so the comparison has a unit of
    inference: the pooled table mixes derivations and subjects, and a rate
    difference there could be one subject with many long pauses.
    """
    from mc.analyse.swr_windows import clean_exposure
    if not len(still):
        return None
    t = np.sort(np.asarray(ripple_t, float))

    def _rate(sel):
        if not sel.any():
            return np.nan, 0.0, 0.0
        s = still.start_s.to_numpy(float)[sel]
        e_ = still.end_s.to_numpy(float)[sel]
        n = float((np.searchsorted(t, e_, side="right")
                   - np.searchsorted(t, s, side="left")).sum())
        e = float(clean_exposure(intervals, s, e_).sum())
        return (n / e if e > 0 else np.nan), n, e

    d = still.duration_s.to_numpy(float)
    r_s, n_s, e_s = _rate((d >= short[0]) & (d < short[1]))
    r_l, n_l, e_l = _rate(d >= long_min)
    return {"rate_short": r_s, "rate_long": r_l, "diff": r_l - r_s,
            "n_short": n_s, "n_long": n_l,
            "exposure_short_s": e_s, "exposure_long_s": e_l}


def within_grid_drift(beh, ripple_t, intervals, n_bins=5):
    """Ripple rate as a function of position WITHIN a grid.

    The relevant drift control for any first-vs-later contrast: the first
    traversal is always at the start of its grid, so a rate that falls or rises
    across a grid would produce a first-vs-later difference with no reference to
    what the subject knows.
    """
    from mc.analyse.swr_windows import clean_exposure
    t = np.sort(np.asarray(ripple_t, float))
    rows = []
    for grid, g in beh.groupby("grid_no"):
        t0 = float(g.new_grid_onset.min())
        t1 = float(g.t_D.max())
        if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
            continue
        edges = np.linspace(t0, t1, n_bins + 1)
        n = (np.searchsorted(t, edges[1:], side="right")
             - np.searchsorted(t, edges[:-1], side="left")).astype(float)
        e = clean_exposure(intervals, edges[:-1], edges[1:])
        for k in range(n_bins):
            rows.append({"grid_no": int(grid), "bin": k,
                         "frac_through_grid": (k + 0.5) / n_bins,
                         "n_ripples": n[k], "exposure_s": e[k]})
    return pd.DataFrame(rows)


def label_still_by_phase(still, beh):
    """Attach the task phase (explore / plan / execute) to each still period."""
    from mc.analyse.swr_windows import add_phase3
    b = add_phase3(beh)
    mid = 0.5 * (still.start_s.to_numpy(float) + still.end_s.to_numpy(float))
    on = b.new_grid_onset.to_numpy(float)
    end = b.t_D.to_numpy(float)
    ph = np.full(len(mid), None, object)
    for i, m in enumerate(mid):
        hit = np.flatnonzero((on <= m) & (end >= m))
        if hit.size:
            ph[i] = b.phase3.to_numpy()[hit[0]]
    out = still.copy()
    out["phase3"] = ph
    return out[out.phase3.notna()]


def fixed_window_by_pause_length(still, ripple_t, intervals,
                                 offset_s=0.5, width_s=1.0,
                                 edges=(1.5, 3.0, 6.0, 12.0, 1e9)):
    """Rate in a FIXED window after the press, binned by how long the pause lasts.

    The control F1 needs. A 0.5-2 s still period is entirely "close to a key
    press", while an 8 s one has a long stretch far from any press. If ripples
    are rarer near presses for any reason, F1 follows trivially from window
    length rather than from stillness.

    Here every window is the SAME width and the SAME distance from the
    preceding press; only the eventual duration of the pause differs. A rate
    that still rises with pause length is prospective -- the brain is in a
    different state from the outset, not merely given more quiet time.
    """
    from mc.analyse.swr_windows import clean_exposure
    if not len(still):
        return pd.DataFrame()
    d = still.duration_s.to_numpy(float)
    ok = d >= offset_s + width_s          # the fixed window must fit
    if not ok.any():
        return pd.DataFrame()
    s0 = still.start_s.to_numpy(float)[ok] + offset_s
    s1 = s0 + width_s
    d = d[ok]
    t = np.sort(np.asarray(ripple_t, float))
    n = (np.searchsorted(t, s1, side="right")
         - np.searchsorted(t, s0, side="left")).astype(float)
    e = clean_exposure(intervals, s0, s1)
    edges = np.asarray(edges, float)
    b = np.digitize(d, edges)
    out = []
    for k in range(len(edges)):
        m = b == k
        if not m.any() or e[m].sum() <= 0:
            continue
        lo = offset_s + width_s if k == 0 else edges[k - 1]
        hi = edges[k] if k < len(edges) else np.inf
        out.append({"pause_lo_s": lo, "pause_hi_s": hi,
                    "n_periods": int(m.sum()), "n_ripples": float(n[m].sum()),
                    "exposure_s": float(e[m].sum()),
                    "rate_hz": float(n[m].sum() / e[m].sum())})
    return pd.DataFrame(out)


def grid_onset_times(beh):
    """One time per grid: when a new reward configuration begins.

    The moment the task hands the subject a new problem. If ripples assemble a
    plan anywhere, this is a candidate -- it is the analogue of the instruction
    phase in the fMRI result.
    """
    return (beh.groupby("grid_no").new_grid_onset.min()
            .to_numpy(float))
