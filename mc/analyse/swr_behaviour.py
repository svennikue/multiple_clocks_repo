#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-uncovering behaviour, on the same clock as the ripples.

`all_trial_times_{XX}.csv` is one row per *repeat*: when each reward was
collected and whether the repeat as a whole was correct. That is enough for the
discovery and phase designs, but not for anything about individual uncovering
attempts -- how many errors a repeat contained, or when each piece of feedback
arrived.

Those are recoverable. The derivatives carry, per grid, a 25 ms-binned record of
the last key pressed (`buttons_per_25ms_*`) and of the location occupied
(`locations_per_25ms_*`). Transitions into the uncover key are the uncovering
attempts; the four reward times per repeat in `all_trial_times` say which of
them were correct; the rest are errors.

Two things make this trustworthy rather than a guess:

1. **The clock is exact.** `session_seconds = grid_onset + bin * 0.025`,
   verified against `all_trial_times` on s02 to within one bin (max 0.02 s).
2. **The counts are validated against the raw log.** The stimulus PC's own
   `BEH_*.mat` records every press with `pressed_to_uncover` and `correct`. On
   s47 it holds 1903 uncover presses of which 1288 correct; this reconstruction
   finds 1891 (99.4%), and 1288 is exactly 4 x 322 repeats, so the correct ones
   are recovered without approximation. Only the *error* count can be slightly
   short, and only where two uncover presses fall in the same 25 ms bin or
   follow each other with no intervening movement key.

The raw `BEH_*.mat` is NOT used directly, for a reason worth recording: its
`BUTTON_PRESS_TIMES` are on the stimulus PC's clock (s54 starts at 509637 s --
machine uptime), not the photodiode clock the neural data lives on, and only 5
of those files are present locally anyway.

@author: Svenja Kuchenhoff
"""

import os
import glob

import numpy as np
import pandas as pd

import mc.analyse.swr_io as swr_io

BIN_S = 0.025                 # the derivatives' behavioural bin
UNCOVER_KEY = "Return"        # the key that uncovers a location in this task
MOVE_KEYS = ("LeftArrow", "RightArrow", "UpArrow", "DownArrow")
MATCH_TOL_S = 0.10            # a press this close to a reward time IS that reward


def _beh_dir(session, data_root=None):
    return os.path.join(swr_io.session_deriv_dir(int(session), data_root),
                        "cells_and_beh")


def _grid_series(session, grid, kind, data_root=None):
    """The 25 ms series for one grid, or None if that file is absent."""
    d = _beh_dir(session, data_root)
    hits = glob.glob(os.path.join(d, f"{kind}_per_25ms_grid{int(grid)}_sub*.csv"))
    if not hits:
        return None
    return pd.read_csv(hits[0], header=None).iloc[0].to_numpy()


def uncover_events(session, beh=None, data_root=None, tol_s=MATCH_TOL_S):
    """Every uncovering attempt in one session, in session seconds.

    Returns one row per attempt:
        t_s          when the location was uncovered
        correct      1 if it collected a reward, 0 if it was an error
        state        'A'..'D' for correct attempts, NaN for errors
        loc          the grid location uncovered (1-9)
        grid_no, rep_overall, is_discovery, n_errors_in_repeat

    Errors are attributed to the repeat whose reward window they fall in, so an
    error before the first reward of a repeat belongs to that repeat.
    """
    session = int(session)
    beh = swr_io.load_behaviour(session, data_root=data_root) if beh is None else beh
    rows = []

    for grid, g in beh.groupby("grid_no"):
        g = g.sort_values("rep_overall")
        btn = _grid_series(session, grid, "buttons", data_root)
        loc = _grid_series(session, grid, "locations", data_root)
        if btn is None:
            continue
        onset = float(g.new_grid_onset.iloc[0])

        btn = btn.astype(str)
        trans = np.flatnonzero(btn[1:] != btn[:-1]) + 1
        press_bins = np.array([t for t in trans if btn[t] == UNCOVER_KEY], int)
        if not press_bins.size:
            continue
        t_press = onset + press_bins * BIN_S

        # reward times, and which repeat/state each belongs to
        rw_t, rw_state, rw_rep = [], [], []
        for _, r in g.iterrows():
            for s in ("A", "B", "C", "D"):
                v = float(r[f"t_{s}"])
                if np.isfinite(v):
                    rw_t.append(v); rw_state.append(s)
                    rw_rep.append(int(r.rep_overall))
        rw_t = np.asarray(rw_t, float)
        if not rw_t.size:
            continue
        order = np.argsort(rw_t)
        rw_t = rw_t[order]
        rw_state = np.asarray(rw_state)[order]
        rw_rep = np.asarray(rw_rep)[order]

        first_rep = int(g.rep_overall.iloc[0])
        rep_bounds = g.sort_values("rep_overall")[["rep_overall", "t_D"]]

        for t in t_press:
            j = int(np.argmin(np.abs(rw_t - t)))
            hit = np.abs(rw_t[j] - t) <= tol_s
            if hit:
                rep, state, ok = int(rw_rep[j]), rw_state[j], 1
            else:
                # an error belongs to the repeat it happened during: the first
                # repeat whose t_D has not passed yet
                after = rep_bounds[rep_bounds.t_D >= t]
                rep = int(after.rep_overall.iloc[0]) if len(after) else int(
                    rep_bounds.rep_overall.iloc[-1])
                state, ok = np.nan, 0
            bin_i = int(round((t - onset) / BIN_S))
            rows.append({
                "session": session, "grid_no": int(grid), "rep_overall": rep,
                "t_s": float(t), "correct": ok, "state": state,
                "loc": float(loc[bin_i]) if loc is not None and
                       0 <= bin_i < len(loc) else np.nan,
                "is_discovery": int(rep == first_rep),
            })

    ev = pd.DataFrame(rows)
    if not len(ev):
        return ev
    ev = ev.sort_values("t_s").reset_index(drop=True)
    err = (ev[ev.correct == 0].groupby(["grid_no", "rep_overall"]).size()
           .rename("n_errors_in_repeat"))
    ev = ev.merge(err, on=["grid_no", "rep_overall"], how="left")
    ev["n_errors_in_repeat"] = ev["n_errors_in_repeat"].fillna(0).astype(int)
    return ev


def movement_series(session, grid, data_root=None):
    """(times_s, is_move) at 25 ms for one grid — the movement covariate.

    Every window in every design gets a movement count from this, because the
    conditions being compared differ in how much the subject was moving, and
    ripple rate is suppressed during movement.
    """
    btn = _grid_series(session, grid, "buttons", data_root)
    if btn is None:
        return None, None
    btn = btn.astype(str)
    trans = np.flatnonzero(btn[1:] != btn[:-1]) + 1
    is_move = np.zeros(len(btn), bool)
    is_move[[t for t in trans if btn[t] in MOVE_KEYS]] = True
    return np.arange(len(btn)) * BIN_S, is_move


def presses_in_windows(session, beh, windows, data_root=None):
    """Number of movement key presses inside each window. Index-aligned."""
    out = np.zeros(len(windows), float)
    onsets = (beh.groupby("grid_no").new_grid_onset.first().to_dict())
    cache = {}
    for i, w in enumerate(windows.itertuples()):
        g = int(getattr(w, "grid_no", -1))
        if g not in cache:
            t, mv = movement_series(session, g, data_root)
            cache[g] = (t, mv)
        t, mv = cache[g]
        if t is None or g not in onsets:
            out[i] = np.nan; continue
        abs_t = onsets[g] + t
        out[i] = float(mv[(abs_t >= w.start_s) & (abs_t < w.end_s)].sum())
    return out


def repeat_table(session, beh=None, data_root=None):
    """One row per repeat: errors, duration, and whether the plan was known.

    This is the table the 'ripples predict subsequent performance' test needs —
    a per-repeat outcome that a preceding ripple rate can be regressed on.
    """
    beh = swr_io.load_behaviour(session, data_root=data_root) if beh is None else beh
    ev = uncover_events(session, beh=beh, data_root=data_root)
    rows = []
    for grid, g in beh.groupby("grid_no"):
        g = g.sort_values("rep_overall")
        first_rep = int(g.rep_overall.iloc[0])
        for _, r in g.iterrows():
            rep = int(r.rep_overall)
            sel = ev[(ev.grid_no == grid) & (ev.rep_overall == rep)] if len(ev) \
                else ev
            rows.append({
                "session": int(session), "grid_no": int(grid),
                "rep_overall": rep, "is_discovery": int(rep == first_rep),
                "t_start": float(r.t_A), "t_D": float(r.t_D),
                "duration_s": float(r.t_D) - float(r.t_A),
                "correct": int(r.correct),
                "n_uncover": int(len(sel)),
                "n_errors": int((sel.correct == 0).sum()) if len(sel) else 0,
            })
    tab = pd.DataFrame(rows)
    if not len(tab):
        return tab
    # repeats remaining until the grid is first solved: the learning outcome
    tab = tab.sort_values(["grid_no", "rep_overall"])
    solved = (tab[tab.correct == 1].groupby("grid_no").rep_overall.min()
              .rename("first_solved_rep"))
    tab = tab.merge(solved, on="grid_no", how="left")
    tab["reps_to_solve"] = tab.first_solved_rep - tab.rep_overall
    return tab.reset_index(drop=True)
