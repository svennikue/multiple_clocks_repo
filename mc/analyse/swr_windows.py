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


def windows_feedback(uncov, lock_s=1.0, min_gap_s=None):
    """Windows locked to each uncovering's FEEDBACK, split correct vs error.

    Replaces the earlier `windows_error_correct`, which could only mark a whole
    repeat as containing an error because per-uncover feedback was not thought to
    be in the derivatives. It is: `mc.analyse.swr_behaviour.uncover_events`
    reconstructs every attempt and its outcome in session seconds, validated
    against the stimulus PC's own log at 99.8% of correct uncoverings.

    `lock_s` is short on purpose (1 s). Correct and incorrect feedback are
    followed by different behaviour -- after a correct uncovering the subject
    moves on, after an error they usually retry -- so a long window would compare
    two different behavioural states rather than two kinds of feedback. Windows
    are additionally truncated at the next uncovering so they never contain the
    following event.
    """
    if not len(uncov):
        return pd.DataFrame()
    u = uncov.sort_values("t_s").reset_index(drop=True)
    nxt = u.t_s.shift(-1).to_numpy()
    end = np.minimum(u.t_s.to_numpy() + lock_s,
                     np.where(np.isfinite(nxt), nxt, np.inf))
    rows = pd.DataFrame({
        "start_s": u.t_s.to_numpy(),
        "end_s": end,
        "condition": np.where(u.correct == 1, "correct", "error"),
        "feedback": np.where(u.correct == 1, "correct", "error"),
        "phase": np.where(u.is_discovery == 1, "discovery", "later"),
        "state": u.state.to_numpy(),
        "grid_no": u.grid_no.to_numpy(int),
        "rep_overall": u.rep_overall.to_numpy(int),
        "is_test": (u.correct == 0).to_numpy(),
    })
    if min_gap_s:
        rows = rows[(rows.end_s - rows.start_s) >= min_gap_s]
    return _finalise(rows)


def windows_first_D(beh, pre_s=0.0, post_s=2.0, truncate_next=False):
    """The single moment the plan first becomes knowable, per grid.

    The fMRI result this is modelled on peaks exactly when the fourth reward is
    revealed, so this isolates that one event: the FIRST arrival at D in a grid,
    against every later arrival at D in the same grid. Unlike the 4x2 discovery
    design it makes no use of A/B/C, which keeps it a single pre-declared
    contrast rather than a table to search.

    The window is `[t_D - pre_s, t_D + post_s]` -- a FIXED span from the moment
    D is uncovered, not "how long the subject stays at D".

    `post_s=None` instead uses the dwell time: t_D to the start of the next
    repeat (t_A of repeat n+1), or the grid's end for the last repeat. That is a
    different question -- occupancy rather than a time-locked response -- and it
    makes the window length differ systematically between conditions, so the
    exposure offset is doing much more work.

    `truncate_next=True` clips the fixed window at the start of the next repeat,
    so a long `post_s` cannot spill into the next traversal. Without it a 2 s
    window after a later D routinely contains the beginning of the next repeat.
    """
    b = add_phase(beh)
    rows = []
    for (blk, grid), g in b.groupby(['session_no', 'grid_no']):
        g = g.sort_values('rep_overall')
        first_rep = int(g.rep_overall.iloc[0])
        nxt = list(g.t_A.to_numpy(float)[1:]) + [np.nan]
        for (_, r), t_next in zip(g.iterrows(), nxt):
            t = float(r['t_D'])
            if not np.isfinite(t):
                continue
            if post_s is None:
                end = t_next if np.isfinite(t_next) else t + REWARD_LOCK_S
            else:
                end = t + post_s
                if truncate_next and np.isfinite(t_next):
                    end = min(end, t_next)
            is_first = int(r['rep_overall']) == first_rep
            rows.append({
                "start_s": t - pre_s, "end_s": end,
                "condition": "first_D" if is_first else "later_D",
                "discovery": "first" if is_first else "later",
                "is_test": bool(is_first),
                "grid_no": int(grid), "rep_overall": int(r['rep_overall']),
            })
    return _finalise(pd.DataFrame(rows))


DESIGNS["first_D"] = windows_first_D
# `feedback` is deliberately NOT in DESIGNS: it takes the uncover-event table
# rather than `beh`, so it cannot go through build_windows' (beh, design) call.


def add_phase3(beh):
    """Three phases, separating exploring from planning from executing.

    `add_phase` collapses everything before the first correct solve into
    "exploration", which merges two behaviourally different things: searching
    for rewards whose identity is still unknown, and knowing all four rewards
    but not yet executing them in order without error.

        explore    the first traversal of a grid -- the rewards are being found
                   for the first time, D is not yet known
        plan       after the first traversal, before the first error-free solve
                   -- all four rewards are known, the route is not yet reliable
        execute    from the first error-free solve onward
    """
    b = beh.copy()
    out = []
    for (blk, grid), g in b.groupby(['session_no', 'grid_no']):
        g = g.sort_values('rep_overall').copy()
        first_rep = int(g.rep_overall.iloc[0])
        solved = g[g.correct.astype(int) == 1]
        first_solved = int(solved.rep_overall.iloc[0]) if len(solved) else np.inf
        ph = np.where(g.rep_overall.to_numpy() == first_rep, "explore",
                      np.where(g.rep_overall.to_numpy() < first_solved,
                               "plan", "execute"))
        g['phase3'] = ph
        out.append(g)
    return pd.concat(out).sort_index()


def windows_phase3(beh, min_pause_s=0.5):
    """Pause windows labelled explore / plan / execute.

    Pauses rather than whole repeats, because whole repeats differ enormously in
    movement and duration between the phases while the gap between traversals is
    the closest thing this task has to a rest period.

    The pause is labelled by the phase of the repeat that FOLLOWS it, so a pause
    labelled "plan" is one the subject entered already knowing all four rewards.
    """
    b = add_phase3(beh)
    rows = []
    for (blk, grid), g in b.groupby(['session_no', 'grid_no']):
        g = g.sort_values('rep_overall')
        for (_, a), (_, c) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
            t0, t1 = float(a['t_D']), float(c['t_A'])
            if not (np.isfinite(t0) and np.isfinite(t1)) or t1 - t0 < min_pause_s:
                continue
            rows.append({
                "start_s": t0, "end_s": t1,
                "condition": c['phase3'], "phase3": c['phase3'],
                "repeat_number": int(c['rep_overall']),
                "grid_no": int(grid), "rep_overall": int(c['rep_overall']),
            })
    return _finalise(pd.DataFrame(rows))


def windows_informative(uncov, lock_s=1.0):
    """Feedback split by whether it could actually change behaviour.

    SK's point: feedback is not equally useful at all times. While the rewards
    are still being discovered, a *correct* uncovering is the informative one --
    it reveals where the next reward is. Once the route is known, a *correct*
    uncovering tells the subject nothing new, and it is an *error* that carries
    the information, because it says the plan just went wrong.

        informative     correct during discovery, error afterwards
        uninformative   error during discovery, correct afterwards

    This is the diagonal of the same feedback x phase table as `windows_feedback`
    and is reported separately rather than instead: the main effect and this
    interaction answer different questions.
    """
    if not len(uncov):
        return pd.DataFrame()
    u = uncov.sort_values("t_s").reset_index(drop=True)
    nxt = u.t_s.shift(-1).to_numpy()
    end = np.minimum(u.t_s.to_numpy() + lock_s,
                     np.where(np.isfinite(nxt), nxt, np.inf))
    disc = u.is_discovery.to_numpy().astype(bool)
    corr = (u.correct.to_numpy() == 1)
    informative = (disc & corr) | (~disc & ~corr)
    rows = pd.DataFrame({
        "start_s": u.t_s.to_numpy(), "end_s": end,
        "condition": np.where(informative, "informative", "uninformative"),
        "informative": np.where(informative, "informative", "uninformative"),
        "feedback": np.where(corr, "correct", "error"),
        "phase": np.where(disc, "discovery", "later"),
        "grid_no": u.grid_no.to_numpy(int),
        "rep_overall": u.rep_overall.to_numpy(int),
        "is_test": informative,
    })
    return _finalise(rows)


DESIGNS["phase3"] = windows_phase3


def peri_event_rate(event_t, align_t, intervals, half_s=5.0, bin_s=0.25):
    """Ripple rate as a function of time relative to a set of alignment events.

    A peri-event time histogram, but **exposure-corrected**: each bin is
    ripples-in-bin divided by the artifact-free seconds in that bin, summed
    over alignment events. Without that correction a dip in rate can be a dip
    in recording quality, which is exactly the artifact this pipeline removes
    everywhere else.

    Returns (centres_s, rate_hz, n_align, exposure_s_per_bin).
    """
    align_t = np.asarray(align_t, float)
    align_t = align_t[np.isfinite(align_t)]
    edges = np.arange(-half_s, half_s + bin_s / 2, bin_s)
    centres = edges[:-1] + bin_s / 2
    if not len(align_t):
        return centres, np.full(len(centres), np.nan), 0, np.zeros(len(centres))

    t = np.sort(np.asarray(event_t, float))
    counts = np.zeros(len(centres))
    expo = np.zeros(len(centres))
    for a in align_t:
        starts = a + edges[:-1]
        stops = a + edges[1:]
        counts += (np.searchsorted(t, stops, side="right")
                   - np.searchsorted(t, starts, side="left"))
        expo += clean_exposure(intervals, starts, stops)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(expo > 0, counts / expo, np.nan)
    return centres, rate, int(len(align_t)), expo


D_WINDOW_KINDS = {
    "uncover_locked": "fixed window from the moment D is uncovered (the press)",
    "arrival_locked": "fixed window from arriving at the D location",
    "deliberation": "arrival -> uncovering: standing on the reward, before pressing",
    "dwell": "uncovering -> leaving the D location",
    "pre_arrival": "fixed window BEFORE arriving at D (approach)",
    "post_leave": "fixed window after stepping off D",
}


def windows_from_d_events(dev, kind="uncover_locked", w_s=1.0,
                          truncate_next=True, min_window_s=None):
    """H1 windows from the arrival/uncover/leave table (`swb.d_events`).

    `kind` selects what "around D" means -- see D_WINDOW_KINDS. The distinction
    matters: dwell time at D is ~3x longer on the discovery traversal than
    later, so any fixed window extending past the press compares a subject who
    is still standing on the reward against one who has already moved on.
    The `deliberation` window is the only one that is naturally matched
    (~0.7 s in both conditions), because it ends at the press by construction.

    `condition` is first_D / later_D, so a positive log(first/later) rate ratio
    supports the hypothesis.
    """
    if not len(dev):
        return pd.DataFrame()
    rows = []
    for r in dev.itertuples():
        if kind == "uncover_locked":
            t0, t1 = r.t_uncover, r.t_uncover + w_s
        elif kind == "arrival_locked":
            t0, t1 = r.t_arrive, r.t_arrive + w_s
        elif kind == "deliberation":
            t0, t1 = r.t_arrive, r.t_uncover
        elif kind == "dwell":
            t0, t1 = r.t_uncover, r.t_leave
        elif kind == "pre_arrival":
            t0, t1 = r.t_arrive - w_s, r.t_arrive
        elif kind == "post_leave":
            t0, t1 = r.t_leave, r.t_leave + w_s
        else:
            raise ValueError(f"unknown kind '{kind}'; have {list(D_WINDOW_KINDS)}")
        if truncate_next and np.isfinite(r.t_next_rep):
            t1 = min(t1, r.t_next_rep)
        rows.append({
            "start_s": t0, "end_s": t1,
            "condition": "first_D" if r.is_discovery else "later_D",
            "discovery": "first" if r.is_discovery else "later",
            "is_test": bool(r.is_discovery),
            "grid_no": int(r.grid_no), "rep_overall": int(r.rep_overall),
            "dwell_s": r.dwell_s, "deliberation_s": r.deliberation_s,
        })
    w = pd.DataFrame(rows)
    if not len(w):
        return w
    # MIN_WINDOW_S (0.5 s) is far too coarse for the short variants: it drops
    # 55% of first-D and 46% of later-D deliberation windows -- differentially,
    # and specifically the fast decisions. The floor is therefore explicit here
    # rather than inherited from the module default.
    floor = MIN_WINDOW_S if min_window_s is None else float(min_window_s)
    w = w[(w.end_s - w.start_s) >= floor].copy()
    w["duration_s"] = w.end_s - w.start_s
    return w.reset_index(drop=True)


def windows_phase3_whole(beh):
    """WHOLE phases, not the pauses between repeats.

    `windows_phase3` builds pause windows labelled by the following repeat,
    which makes "explore" structurally impossible: explore is the first
    traversal, and the first traversal has no preceding pause. That is why the
    H5 run contained only {plan, execute}.

    Phases here, per SK:
        explore   grid onset -> t_D of the first traversal
                  (until all four rewards have been uncovered once)
        plan      end of explore -> start of the first error-free solve
        execute   from the first error-free solve to the end of the grid

    These differ enormously in movement and duration, which is exactly why the
    pause version existed. The exposure offset handles duration; movement stays
    a covariate and a confound to be shown, not assumed away.
    """
    b = add_phase3(beh)
    rows = []
    for (blk, grid), g in b.groupby(['session_no', 'grid_no']):
        g = g.sort_values('rep_overall')
        first = g.iloc[0]
        t_start = float(first.new_grid_onset)
        t_explore_end = float(first.t_D)
        solved = g[g.correct.astype(int) == 1]
        t_exec_start = (float(solved.iloc[0].new_grid_onset) if len(solved)
                        else np.nan)
        t_end = float(g.t_D.max())
        spans = [("explore", t_start, t_explore_end)]
        if np.isfinite(t_exec_start) and t_exec_start > t_explore_end:
            spans.append(("plan", t_explore_end, t_exec_start))
            spans.append(("execute", t_exec_start, t_end))
        elif np.isfinite(t_exec_start):
            spans.append(("execute", max(t_exec_start, t_explore_end), t_end))
        else:
            spans.append(("plan", t_explore_end, t_end))
        for name, t0, t1 in spans:
            if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
                continue
            rows.append({"start_s": t0, "end_s": t1, "condition": name,
                         "phase3": name, "grid_no": int(grid),
                         "rep_overall": int(first.rep_overall)})
    return _finalise(pd.DataFrame(rows))


def windows_immobility(beh, move_times, is_move, min_still_s=1.0,
                       max_still_s=None):
    """Periods with no movement key press, labelled by task phase.

    H2 as SK meant it: the subject is standing still and thinking. Comparing
    immobility during exploration/planning with immobility during execution
    holds the behavioural state constant and varies only what is known --
    which the inter-repeat pause version does not, because a pause also
    contains whatever movement precedes the next traversal.

    `move_times` / `is_move` come from `swr_behaviour.movement_series`, already
    on the session clock.
    """
    b = add_phase3(beh)
    t = np.asarray(move_times, float)
    mv = np.asarray(is_move, bool)
    idx = np.flatnonzero(mv)
    if idx.size < 2:
        return pd.DataFrame()
    # a still period is the gap between consecutive movement presses
    rows = []
    for a, c in zip(idx[:-1], idx[1:]):
        t0, t1 = t[a], t[c]
        if t1 - t0 < min_still_s:
            continue
        if max_still_s is not None and t1 - t0 > max_still_s:
            continue
        mid = 0.5 * (t0 + t1)
        # which repeat is this inside?
        hit = b[(b.new_grid_onset <= mid) & (b.t_D >= mid)]
        if not len(hit):
            continue
        r = hit.iloc[0]
        rows.append({"start_s": t0, "end_s": t1,
                     "condition": r.phase3, "phase3": r.phase3,
                     "grid_no": int(r.grid_no),
                     "rep_overall": int(r.rep_overall)})
    return _finalise(pd.DataFrame(rows))
