#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavioural windows for H1, and event-to-window assignment.

H1: hippocampal ripples support "loading" an action plan once the trajectory
becomes known.

The task distinction is **knowledge, not mobility** -- all three phases involve
movement (clarified by SK):

  1. exploration  -- reward locations unknown, subject is searching
  2. planning     -- all four rewards uncovered correctly, route now plannable
  3. execution    -- the known route is repeated

So the classic immobility confound (ripple rate rises at rest) largely does not
apply here; there is no immobile condition. Movement is still carried as a
covariate because its amount and speed plausibly differ between phases.

Four designs, from weakest to strongest control:

| design | contrast | controls for |
|---|---|---|
| `sections` | exploration / first_correct / later_repeats | — baseline contrast |
| `reward_locked` | after first t_D vs after first t_A/t_B/t_C | movement, visual input, duration |
| `pauses` | between-repeat transitions, across repeat number | movement, duration |
| `solve_number` | first correct solve vs later solves, same grid | movement, visual input, structure |

`reward_locked` and `solve_number` are the strongest: they hold movement and
visual input constant by construction, which `sections` cannot.

Exposure is **artifact-free seconds**, not window duration. A window half
removed by artifact rejection offers half the opportunity to observe a ripple,
and the GLM offset must reflect that.

@author: Svenja Kuchenhoff
"""

import numpy as np
import pandas as pd

STATE_COLS = ("t_A", "t_B", "t_C", "t_D")
REWARD_LOCK_S = 5.0          # window after a reward arrival
MIN_WINDOW_S = 0.5


# =============================================================================
# PHASE LABELLING
# =============================================================================

def add_phase(beh):
    """Label each repeat exploration / first_correct / later_repeats.

    `plan_known` is the cumulative max of `correct` within a grid, so a later
    error does not send the subject back to "exploration" -- once the route is
    known it stays known. `correct` alone is per-repeat accuracy and must not
    be used as a planning state.
    """
    b = beh.copy()
    cum = (b.groupby(['session_no', 'grid_no'])['correct']
             .transform(lambda s: s.astype(int).cumsum()))
    b['phase'] = np.where(~b.plan_known.to_numpy(bool), "exploration",
                          np.where(cum.to_numpy() == 1, "first_correct",
                                   "later_repeats"))
    b['solve_index'] = cum.to_numpy()
    return b


# =============================================================================
# WINDOW BUILDERS
# =============================================================================

def _finalise(w):
    w = w[(w.end_s - w.start_s) >= MIN_WINDOW_S].copy()
    w["duration_s"] = w.end_s - w.start_s
    return w.reset_index(drop=True)


def windows_sections(beh):
    """One window per repeat, labelled by learning phase."""
    b = add_phase(beh)
    return _finalise(pd.DataFrame({
        "start_s": b.new_grid_onset.to_numpy(float),
        "end_s": b.t_D.to_numpy(float),
        "condition": b.phase.to_numpy(),
        "grid_no": b.grid_no.to_numpy(int),
        "rep_overall": b.rep_overall.to_numpy(int),
        "solve_index": b.solve_index.to_numpy(int),
    }))


def windows_reward_locked(beh, lock_s=REWARD_LOCK_S):
    """`lock_s` after each reward arrival on the FIRST traversal of each grid.

    The test window follows t_D -- the moment the last reward is uncovered and
    the full route becomes known. Control windows follow t_A/t_B/t_C in the
    same traversal: same event type, same movement, same duration, differing
    only in whether the route is now complete.

    Windows are truncated at the next state change so they cannot overlap; the
    resulting duration variation is absorbed by the exposure offset.
    """
    b = add_phase(beh)
    rows = []
    for (blk, grid), g in b.groupby(['session_no', 'grid_no']):
        g = g.sort_values('rep_overall')
        first = g.iloc[0]
        times = [float(first[c]) for c in STATE_COLS]
        nxt = times[1:] + [float(first['t_D']) + lock_s]
        for k, (c, t0, t1) in enumerate(zip(STATE_COLS, times, nxt)):
            if not np.isfinite(t0):
                continue
            rows.append({
                "start_s": t0,
                "end_s": min(t0 + lock_s, t1 if np.isfinite(t1) else t0 + lock_s),
                "condition": f"after_{c[-1]}",
                "is_test": c == "t_D",
                "grid_no": int(grid), "rep_overall": int(first['rep_overall']),
                "solve_index": int(first['solve_index']),
            })
    return _finalise(pd.DataFrame(rows))


def windows_pauses(beh):
    """t_D of repeat n to t_A of repeat n+1, within a grid.

    The between-repeat transition. Repeated many times per grid, so
    `repeat_number` is a continuous predictor: if ripples support loading, the
    rate should decline as the route becomes automatic (Chen's block-number
    logic).
    """
    b = add_phase(beh)
    rows = []
    for (blk, grid), g in b.groupby(['session_no', 'grid_no']):
        g = g.sort_values('rep_overall')
        for (_, a), (_, c) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
            t0, t1 = float(a['t_D']), float(c['t_A'])
            if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
                continue
            rows.append({
                "start_s": t0, "end_s": t1, "condition": "pause",
                "repeat_number": int(c['rep_overall']),
                "solve_index": int(c['solve_index']),
                "phase_after": c['phase'],
                "grid_no": int(grid), "rep_overall": int(c['rep_overall']),
            })
    return _finalise(pd.DataFrame(rows))


def windows_solve_number(beh):
    """Full traversals, first correct solve vs later correct solves of the
    same grid. Identical window type and task; only planning demand differs."""
    b = add_phase(beh)
    g = b[b.correct.astype(int) == 1].copy()
    return _finalise(pd.DataFrame({
        "start_s": g.new_grid_onset.to_numpy(float),
        "end_s": g.t_D.to_numpy(float),
        "condition": np.where(g.solve_index.to_numpy() == 1,
                              "first_correct_solve", "later_solve"),
        "solve_index": g.solve_index.to_numpy(int),
        "grid_no": g.grid_no.to_numpy(int),
        "rep_overall": g.rep_overall.to_numpy(int),
    }))


DESIGNS = {
    "sections": windows_sections,
    "reward_locked": windows_reward_locked,
    "pauses": windows_pauses,
    "solve_number": windows_solve_number,
}


def build_windows(beh, design, **kwargs):
    if design not in DESIGNS:
        raise ValueError(f"unknown design '{design}'; have {list(DESIGNS)}")
    w = DESIGNS[design](beh, **kwargs)
    w.insert(0, "design", design)
    w.insert(1, "window_id", np.arange(len(w)))
    return w


# =============================================================================
# EXPOSURE AND ASSIGNMENT
# =============================================================================

def clean_exposure(intervals, starts, stops):
    """Artifact-free seconds inside each [start, stop], vectorised.

    Uses the cumulative clean time as a function of wall time, so the whole
    thing is two interpolations rather than a loop over windows x intervals.
    """
    iv = np.asarray(intervals, float).reshape(-1, 2)
    if not len(iv):
        return np.zeros(len(starts))
    order = np.argsort(iv[:, 0])
    iv = iv[order]
    dur = np.diff(iv, axis=1).ravel()
    # knots: cumulative clean seconds at each interval edge
    xs = np.empty(2 * len(iv)); ys = np.empty(2 * len(iv))
    xs[0::2], xs[1::2] = iv[:, 0], iv[:, 1]
    cum = np.concatenate([[0.0], np.cumsum(dur)])
    ys[0::2], ys[1::2] = cum[:-1], cum[1:]
    return np.interp(stops, xs, ys) - np.interp(starts, xs, ys)


def assign_events_to_windows(events, windows, intervals, shift_s=0.0,
                             clean_total_s=None):
    """Count events per window, with artifact-free exposure.

    `shift_s` circularly shifts the event train **on the artifact-free axis**,
    so a shifted null can never place an event inside an artifact -- which
    would otherwise make the null easier to beat than the data.

    `shift_s = 0` IS the observed case and goes through this identical code
    path, per CLAUDE.md rule 4.
    """
    w = windows.sort_values("start_s").reset_index(drop=True)
    starts = w.start_s.to_numpy(float)
    stops = w.end_s.to_numpy(float)

    t = np.sort(np.asarray(events, float))
    if shift_s and len(t):
        from mc.analyse.swr_artifact import CleanAxis
        ax = CleanAxis(intervals)
        c = ax.to_clean(t)
        c = c[np.isfinite(c)]
        t = np.sort(ax.to_wall(c + shift_s))

    # window boundaries can overlap only across designs, never within one, so
    # a pair of searchsorteds is exact
    n = (np.searchsorted(t, stops, side='right')
         - np.searchsorted(t, starts, side='left'))

    out = w.copy()
    out["n_ripples"] = n.astype(int)
    out["exposure_s"] = clean_exposure(intervals, starts, stops)
    out["clean_frac"] = out.exposure_s / out.duration_s.replace(0, np.nan)
    out["rate_hz"] = out.n_ripples / out.exposure_s.replace(0, np.nan)
    return out


def windows_discovery(beh, lock_s=REWARD_LOCK_S):
    """State (A/B/C/D) x discovery (first traversal vs later): the design that
    separates SK's hypothesis from the obvious alternative.

    The hypothesis is that ripples support *loading* an action plan at the
    moment the route first becomes knowable -- the FIRST uncovering of D, which
    is where the fMRI action-plan representation appears. It does **not**
    predict elevated ripples whenever D is reached: once the subject knows where
    to go, there is nothing left to load.

    The alternative -- that D is special for some other reason (it is last, it
    ends the traversal, it carries different reward) -- predicts elevation at
    every D.

    These make different predictions about the same 4x2 table:
        hypothesis  -> interaction: D x first elevated specifically
        alternative -> main effect of state: D elevated regardless of first/later

    Every repeat in this task reaches all four rewards, so the first traversal
    (lowest `rep_overall` in a grid) is the discovery traversal.
    """
    b = add_phase(beh)
    rows = []
    for (blk, grid), g in b.groupby(['session_no', 'grid_no']):
        g = g.sort_values('rep_overall')
        first_rep = int(g.rep_overall.iloc[0])
        for _, r in g.iterrows():
            times = [float(r[c]) for c in STATE_COLS]
            nxt = times[1:] + [times[-1] + lock_s]
            is_first = int(r['rep_overall']) == first_rep
            for c, t0, t1 in zip(STATE_COLS, times, nxt):
                if not np.isfinite(t0):
                    continue
                rows.append({
                    "start_s": t0,
                    "end_s": min(t0 + lock_s, t1 if np.isfinite(t1) else t0 + lock_s),
                    "condition": f"{c[-1]}_{'first' if is_first else 'later'}",
                    "state": c[-1],
                    "discovery": "first" if is_first else "later",
                    "is_test": bool(is_first and c == "t_D"),
                    "repeat_number": int(r['rep_overall']),
                    "correct": int(r['correct']),
                    "grid_no": int(grid), "rep_overall": int(r['rep_overall']),
                })
    return _finalise(pd.DataFrame(rows))


def windows_error_correct(beh, lock_s=REWARD_LOCK_S, skip_first=True):
    """Reward-arrival windows split by whether the repeat was executed correctly.

    CAVEAT: `correct` is **per-repeat** accuracy, not per-uncover. Per-uncover
    feedback (`feedback.csv` in the old raw session directories) is not present
    in derivatives, so an "error" window here means "a reward arrival during a
    repeat that contained an error", not "the moment an error was made". If the
    per-uncover feedback is recovered, this design should be rebuilt on it.

    The discovery traversal is excluded by default, since it is uninformative
    about execution accuracy and would confound the error contrast with the
    novelty contrast tested by `windows_discovery`.
    """
    b = add_phase(beh)
    rows = []
    for (blk, grid), g in b.groupby(['session_no', 'grid_no']):
        g = g.sort_values('rep_overall')
        first_rep = int(g.rep_overall.iloc[0])
        for _, r in g.iterrows():
            if skip_first and int(r['rep_overall']) == first_rep:
                continue
            times = [float(r[c]) for c in STATE_COLS]
            nxt = times[1:] + [times[-1] + lock_s]
            lab = "correct" if int(r['correct']) == 1 else "error"
            for c, t0, t1 in zip(STATE_COLS, times, nxt):
                if not np.isfinite(t0):
                    continue
                rows.append({
                    "start_s": t0,
                    "end_s": min(t0 + lock_s, t1 if np.isfinite(t1) else t0 + lock_s),
                    "condition": lab, "state": c[-1],
                    "repeat_number": int(r['rep_overall']),
                    "grid_no": int(grid), "rep_overall": int(r['rep_overall']),
                })
    return _finalise(pd.DataFrame(rows))


DESIGNS["discovery"] = windows_discovery
DESIGNS["error_correct"] = windows_error_correct
