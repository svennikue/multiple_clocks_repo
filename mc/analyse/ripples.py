#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared functions for the hippocampal ripple analyses.

Everything both ripple scripts need lives here, so a change to how a condition
is defined or a test is run happens in one place:

    scripts/swr_ripple_tests.py   the conditions we settled on, tested and plotted
    scripts/swr_explore.py        scratch probes, nothing claimed

Sections:
    1) Loading            the bundle written by the cluster
    2) Task conditions    stages, valence, and which reward was being sought
    3) Rates              exposure-corrected ripple rate in windows and over time
    4) Tests              sliding window vs the trial's own baseline, with a
                          cluster permutation over window positions
    5) Plots

Design parameters are Sakon & Kahana (2022, PNAS 119:e2201657119) and He et al.
(2026, Nat Neurosci 29:1711), not tuned on this dataset.

@author: Svenja Kuchenhoff
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy import stats

import mc.analyse.swr_io as swr_io


# ── Settings ──────────────────────────────────────────────────────────
BIN_S           = 0.100      # peri-event histogram bin (Sakon)
BASELINE_WIN    = (-1.6, -1.1)   # the same window shifted 1 s earlier (Sakon Eq. 2)
DEDUP_S         = 2.0        # events closer than this share ripples; drop the later
HALF_S          = 2.0        # peri-event window half-width
SLIDE_WIDTHS_S  = (0.3, 0.5) # sliding window widths to report
MIN_CLEAN_FRAC  = 0.5        # a window must be at least half artifact-free
MIN_EVENTS      = 20         # below this a condition is not estimated
N_SIGN_FLIPS    = 1000       # permutations for the cluster test
CLUSTER_ALPHA   = 0.05       # cluster-forming threshold, two-sided

STAGES = ('first uncovers', 'while learning', 'once known')
VALENCE = ('correct', 'error')
REWARDS = ('A', 'B', 'C', 'D')

# Feedback valence sets the hue, stage sets the lightness, so a crossed label
# like "error, while learning" is never drawn the same colour as its partner.
VALENCE_COLOUR = {'correct': '#0E3D3A', 'error': '#B03A5B'}
STAGE_LIGHTEN  = {'first uncovers': 0.0, 'while learning': 0.35, 'once known': 0.65}


# ── 1) Loading ────────────────────────────────────────────────────────

def load_bundle(bundle_dir):
    """Read the bundle the cluster wrote. Everything downstream starts here."""
    out = {}
    for name in ('ripples', 'intervals', 'channel_qc', 'behaviour', 'uncover',
                 'pairs'):
        path = os.path.join(bundle_dir, f'{name}.csv')
        out[name] = pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()
    return out


def sessions_in(bundle):
    return sorted(bundle['ripples'].session.unique())


def derivations(bundle, session):
    """(pair_id, ripple times, artifact-free intervals) for each good derivation."""
    rip = bundle['ripples']
    iv = bundle['intervals']
    qc = bundle['channel_qc']
    qc_s = qc[qc.session == session].set_index('pair_id')
    out = []
    for pair_id, e in rip[rip.session == session].groupby('pair_id'):
        if pair_id in qc_s.index and bool(qc_s.loc[pair_id, 'excluded']):
            continue
        intervals = iv[(iv.session == session) & (iv.pair_id == pair_id)]
        intervals = intervals[['start_s', 'stop_s']].to_numpy()
        if not len(intervals):
            continue
        out.append((pair_id, e.t_peak_s.to_numpy(float), intervals))
    return out


def subject_of(bundle, session):
    r = bundle['ripples']
    hit = r[r.session == session]
    return str(hit.subject_key.iloc[0]) if len(hit) else f's{session:02d}'


# ── 2) Task conditions ────────────────────────────────────────────────

def stage_of_repeat(beh_one_grid):
    """Learning stage of every repeat of one grid.

        first uncovers   the first traversal
        while learning   up to AND INCLUDING the first fully correct repeat
        once known       every repeat after that

    The boundary is deliberate: the first error-free repeat is the one on which
    the route is first demonstrated, not yet merely relied on.
    """
    g = beh_one_grid.sort_values('rep_overall')
    reps = g.rep_overall.to_numpy(int)
    solved = reps[g.correct.to_numpy(int) == 1]
    first_solved = int(solved[0]) if solved.size else np.inf
    out = {}
    for rep in reps:
        if rep == reps[0]:
            out[int(rep)] = 'first uncovers'
        elif rep <= first_solved:
            out[int(rep)] = 'while learning'
        else:
            out[int(rep)] = 'once known'
    return out


def uncover_table(bundle, session):
    """Every uncovering attempt, labelled with everything a condition needs.

    Columns: t_s, valence (correct/error), stage, reward, grid_no, rep_overall.

    `reward` is the reward the subject was SEEKING. A correct uncovering names
    its own reward; an error inherits the one being searched for, counted as the
    (k+1)th after k rewards have already been collected in that repeat.
    """
    beh = bundle['behaviour']
    beh = beh[beh.session == session]
    unc = bundle['uncover']
    unc = unc[unc.session == session]
    if not len(beh) or not len(unc):
        return pd.DataFrame()

    stages, reward_times = {}, {}
    for grid, g in beh.groupby('grid_no'):
        stages.update({(int(grid), r): s for r, s in stage_of_repeat(g).items()})
        for _, row in g.iterrows():
            reward_times[(int(grid), int(row.rep_overall))] = [
                float(row[f't_{x}']) for x in REWARDS]

    rows = []
    for e in unc.itertuples():
        key = (int(e.grid_no), int(e.rep_overall))
        if key not in stages or key not in reward_times:
            continue
        collected = sum(1 for t in reward_times[key]
                        if np.isfinite(t) and t < e.t_s)
        reward = REWARDS[min(collected, 3)]
        if int(e.correct) == 1 and isinstance(e.state, str):
            reward = e.state
        rows.append({'t_s': float(e.t_s), 'grid_no': int(e.grid_no),
                     'rep_overall': int(e.rep_overall),
                     'valence': 'correct' if int(e.correct) == 1 else 'error',
                     'stage': stages[key], 'reward': reward})
    return pd.DataFrame(rows)


def dedup(times, min_gap_s=DEDUP_S):
    """Drop events within `min_gap_s` of the previous one (Sakon).

    With a +-2 s analysis window, two events closer than that share ripples,
    and those ripples would then be counted as independent observations.
    """
    t = np.sort(np.asarray(times, float))
    t = t[np.isfinite(t)]
    if not t.size:
        return t
    return t[np.concatenate([[True], np.diff(t) >= min_gap_s])]


# ── 3) Rates ──────────────────────────────────────────────────────────

def clean_seconds(intervals, starts, stops):
    """Artifact-free seconds inside each [start, stop]."""
    iv = np.asarray(intervals, float).reshape(-1, 2)
    if not len(iv):
        return np.zeros(len(starts))
    iv = iv[np.argsort(iv[:, 0])]
    dur = np.diff(iv, axis=1).ravel()
    xs = np.empty(2 * len(iv))
    ys = np.empty(2 * len(iv))
    xs[0::2], xs[1::2] = iv[:, 0], iv[:, 1]
    cum = np.concatenate([[0.0], np.cumsum(dur)])
    ys[0::2], ys[1::2] = cum[:-1], cum[1:]
    return np.interp(stops, xs, ys) - np.interp(starts, xs, ys)


def rate_in_window(event_times, ripple_times, intervals, window):
    """Exposure-corrected ripple rate in `window` around each event.

    NaN where the window is more than half artifact, rather than an unstable
    rate from a sliver of clean time.
    """
    ev = np.asarray(event_times, float)
    starts, stops = ev + window[0], ev + window[1]
    t = np.sort(np.asarray(ripple_times, float))
    n = (np.searchsorted(t, stops, side='right')
         - np.searchsorted(t, starts, side='left')).astype(float)
    exposure = clean_seconds(intervals, starts, stops)
    too_dirty = exposure < MIN_CLEAN_FRAC * (window[1] - window[0])
    rate = np.where(too_dirty, np.nan,
                    n / np.where(exposure > 0, exposure, np.nan))
    return rate, n, exposure


def peri_event_rate(event_times, ripple_times, intervals, half_s=HALF_S,
                    bin_s=BIN_S):
    """Ripple rate in bins around each event. Returns (bin centres, rate)."""
    edges = np.arange(-half_s, half_s + bin_s / 2, bin_s)
    centres = edges[:-1] + bin_s / 2
    ev = np.asarray(event_times, float)
    t = np.sort(np.asarray(ripple_times, float))
    out = np.full((len(ev), len(centres)), np.nan)
    for k in range(len(centres)):
        starts, stops = ev + edges[k], ev + edges[k + 1]
        n = (np.searchsorted(t, stops, side='right')
             - np.searchsorted(t, starts, side='left')).astype(float)
        exposure = clean_seconds(intervals, starts, stops)
        out[:, k] = np.where(exposure > 0, n / exposure, np.nan)
    return centres, out


def rate_by_subject(bundle, events_per_session, half_s=HALF_S, bin_s=BIN_S):
    """Peri-event rate per subject, averaged over that subject's derivations.

    `events_per_session` maps session -> event times. Returns
    (bin centres, {subject: profile}, counts) where counts records how much data
    went into the condition -- never left implicit.
    """
    per_subject, centres = {}, None
    counts = {'n_sessions': 0, 'n_derivations': 0,
              'n_events_raw': 0, 'n_events_used': 0}
    for session, raw in events_per_session.items():
        t = dedup(raw)
        counts['n_events_raw'] += len(raw)
        if t.size < MIN_EVENTS:
            continue
        counts['n_sessions'] += 1
        counts['n_events_used'] += int(t.size)
        subject = subject_of(bundle, session)
        for pair_id, ripples, intervals in derivations(bundle, session):
            counts['n_derivations'] += 1
            centres, profile = peri_event_rate(t, ripples, intervals,
                                               half_s=half_s, bin_s=bin_s)
            per_subject.setdefault(subject, []).append(np.nanmean(profile, axis=0))
    per_subject = {s: np.nanmean(np.vstack(v), axis=0)
                   for s, v in per_subject.items()}
    counts['n_subjects'] = len(per_subject)
    return centres, per_subject, counts


# ── 4) Tests ──────────────────────────────────────────────────────────

def baseline_subtract(profiles, centres, baseline=BASELINE_WIN):
    """Subtract each subject's OWN baseline (Sakon Eq. 2).

    This is what makes conditions with different overall rates comparable: a
    transient is measured against the same trial's floor, so a between-condition
    baseline difference cancels instead of masquerading as an effect.
    """
    subjects = sorted(profiles)
    X = np.vstack([profiles[s] for s in subjects])
    in_base = (centres >= baseline[0]) & (centres < baseline[1])
    return subjects, X - np.nanmean(X[:, in_base], axis=1, keepdims=True)


def _clusters(t_values, threshold):
    """Runs of consecutive positions exceeding the threshold, either sign."""
    over = np.abs(t_values) > threshold
    out, i = [], 0
    while i < len(over):
        if over[i]:
            j = i
            while j + 1 < len(over) and over[j + 1]:
                j += 1
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def _smooth_and_t(X, centres, width_s, bin_s):
    """Moving-average each subject's course, then a one-sample t per position."""
    k = max(int(round(width_s / bin_s)), 1)
    if X.shape[1] < k or X.shape[0] < 3:
        return None, None, None
    kernel = np.ones(k) / k
    smoothed = np.vstack([np.convolve(row, kernel, mode='valid') for row in X])
    lo = k // 2
    times = np.asarray(centres, float)[lo:lo + smoothed.shape[1]]
    t = np.asarray(stats.ttest_1samp(smoothed, 0.0, nan_policy='omit').statistic,
                   float)
    return smoothed, times, t


def sliding_window_test(profiles_by_condition, centres, width_s=0.3,
                        bin_s=BIN_S, n_perm=N_SIGN_FLIPS, seed=42,
                        alpha=CLUSTER_ALPHA, correct_over='time'):
    """Test every window position in every condition, then correct.

    No window is chosen: a window of `width_s` is a moving average of
    width_s/bin_s bins, so every position is evaluated and the multiple
    comparisons are handled by a cluster-mass permutation. Subjects' signs are
    flipped at random, which is the exact null for a within-subject contrast.

    correct_over
        'time'                 family-wise across window POSITIONS, separately
                               for each condition. A p of 0.02 means a cluster
                               this large appears anywhere in the time course 2%
                               of the time -- but running six conditions then
                               gives six such tests, uncorrected between them.
        'time_and_conditions'  family-wise across positions AND conditions. The
                               same sign-flip is applied to a subject in every
                               condition, which preserves the dependence between
                               conditions that share subjects, and the null takes
                               the maximum cluster mass over the whole family.
                               This is the honest p when several conditions are
                               inspected together, and it is stricter.

    `profiles_by_condition` is {label: {subject: time course}}, already
    baseline-subtracted. Returns {label: result dict} with the same fields as
    before, plus the correction actually applied.
    """
    prepared = {}
    for label, profiles in profiles_by_condition.items():
        subjects = sorted(profiles)
        X = np.vstack([profiles[s] for s in subjects])
        smoothed, times, t_obs = _smooth_and_t(X, centres, width_s, bin_s)
        if smoothed is None:
            continue
        prepared[label] = {'subjects': subjects, 'smoothed': smoothed,
                           'times': times, 't': t_obs,
                           'threshold': float(stats.t.ppf(1 - alpha / 2,
                                                          smoothed.shape[0] - 1))}
    if not prepared:
        return {}

    for d in prepared.values():
        d['found'] = _clusters(d['t'], d['threshold'])
        d['mass'] = [float(np.nansum(np.abs(d['t'][a:b]))) for a, b in d['found']]

    rng = np.random.default_rng(seed)
    all_subjects = sorted({s for d in prepared.values() for s in d['subjects']})
    shared = correct_over == 'time_and_conditions'
    nulls = {label: np.zeros(n_perm) for label in prepared}
    pooled = np.zeros(n_perm)

    for i in range(n_perm):
        # one flip per subject, reused across conditions when correcting over
        # the family, so conditions sharing subjects stay correlated in the null
        flip_of = {s: rng.choice([-1.0, 1.0]) for s in all_subjects}
        biggest = 0.0
        for label, d in prepared.items():
            if shared:
                signs = np.array([[flip_of[s]] for s in d['subjects']])
            else:
                signs = rng.choice([-1.0, 1.0], size=(len(d['subjects']), 1))
            t_i = np.asarray(stats.ttest_1samp(d['smoothed'] * signs, 0.0,
                                               nan_policy='omit').statistic,
                             float)
            masses = [np.nansum(np.abs(t_i[a:b]))
                      for a, b in _clusters(t_i, d['threshold'])]
            m = max(masses, default=0.0)
            nulls[label][i] = m
            biggest = max(biggest, m)
        pooled[i] = biggest

    out = {}
    for label, d in prepared.items():
        null = pooled if shared else nulls[label]
        clusters = []
        for (a, b), mass in zip(d['found'], d['mass']):
            p = float((1 + np.sum(null >= mass)) / (1 + n_perm))
            peak = a + int(np.nanargmax(np.abs(d['t'][a:b])))
            clusters.append({'start_s': float(d['times'][a]),
                             'stop_s': float(d['times'][b - 1]),
                             'peak_s': float(d['times'][peak]),
                             'peak_t': float(d['t'][peak]), 'mass': mass,
                             'p': p,
                             'direction': 'increase' if d['t'][peak] > 0
                                          else 'decrease'})
        out[label] = {'times': d['times'], 't': d['t'],
                      'threshold': d['threshold'], 'clusters': clusters,
                      'null_mass': null, 'width_s': width_s,
                      'n_subjects': int(d['smoothed'].shape[0]),
                      'corrected_over': ('window positions and conditions'
                                         if shared else 'window positions'),
                      'n_conditions_in_family': len(prepared) if shared else 1}
    return out


def window_test(profiles, centres, window, baseline=BASELINE_WIN,
                n_perm=10000, seed=42):
    """One named window against the trial's own baseline, per subject.

    Reported alongside the sliding test because a named window is easier to
    quote, not because it is the primary result -- it involves a choice the
    sliding version does not.
    """
    subjects, X = baseline_subtract(profiles, centres, baseline)
    in_win = (centres >= window[0]) & (centres < window[1])
    values = np.nanmean(X[:, in_win], axis=1)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return None
    t, p = stats.ttest_1samp(values, 0.0)
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1.0, 1.0], size=(n_perm, values.size))
            * values).mean(axis=1)
    p_perm = float((1 + np.sum(np.abs(null) >= abs(values.mean())))
                   / (1 + n_perm))
    return {'window_s': list(window), 'n_subjects': int(values.size),
            'mean_hz': float(values.mean()), 't': float(t), 'p': float(p),
            'p_perm': p_perm}


# ── 5) Plots ──────────────────────────────────────────────────────────

def condition_colour(label, index=0):
    """Valence sets the hue, stage sets the lightness."""
    low = str(label).lower()
    valence = next((v for v in VALENCE if v in low), None)
    if valence is None:
        return plt.get_cmap('tab10')(index % 10)
    lighten = next((f for s, f in STAGE_LIGHTEN.items() if s in low), 0.0)
    base = np.array(mcolors.to_rgb(VALENCE_COLOUR[valence]))
    return tuple(base + (1.0 - base) * lighten)


def plot_condition(centres, profiles_by_condition, sliding_by_condition,
                   title, out_png, baseline=BASELINE_WIN):
    """Three panels: the rate over time, the sliding test, and the null.

    Left    peri-event rate, mean +- SEM across subjects, baseline window shaded
    Middle  the t course at every window position, surviving clusters shaded
    Right   the permutation null of cluster mass, with the observed clusters
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2),
                             gridspec_kw=dict(width_ratios=[1.25, 1.25, 1]))
    fig.subplots_adjust(wspace=0.28, top=0.80)

    ax = axes[0]
    ax.axvspan(*baseline, color='0.88', lw=0, zorder=0)
    for i, (label, profiles) in enumerate(profiles_by_condition.items()):
        X = np.vstack([profiles[s] for s in sorted(profiles)])
        mean = np.nanmean(X, axis=0)
        sem = np.nanstd(X, axis=0) / max(np.sqrt(X.shape[0]), 1)
        colour = condition_colour(label, i)
        ax.plot(centres, mean, color=colour, lw=1.5,
                label=f'{label} (n={X.shape[0]})')
        ax.fill_between(centres, mean - sem, mean + sem, color=colour,
                        alpha=0.20, lw=0)
    ax.axvline(0, color='0.35', lw=1.1)
    ax.set_xlabel('Time from event (s)')
    ax.set_ylabel('Ripple rate (Hz)')
    ax.set_title('Peri-event rate\ngrey = baseline window', fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[1]
    for i, (label, sliding) in enumerate(sliding_by_condition.items()):
        if sliding is None:
            continue
        colour = condition_colour(label, i)
        ax.plot(sliding['times'], sliding['t'], color=colour, lw=1.5)
        for cluster in sliding['clusters']:
            if cluster['p'] < 0.05:
                ax.axvspan(cluster['start_s'], cluster['stop_s'], color=colour,
                           alpha=0.16, lw=0)
                ax.annotate(f"p = {cluster['p']:.3f}",
                            xy=(cluster['peak_s'], cluster['peak_t']),
                            xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=8, color=colour)
        for sign in (1, -1):
            ax.axhline(sign * sliding['threshold'], color='0.6', lw=0.8, ls=':')
    ax.axhline(0, color='0.45', lw=0.9)
    ax.axvline(0, color='0.35', lw=1.1)
    ax.set_xlabel('Centre of the sliding window (s)')
    ax.set_ylabel('t vs own baseline')
    ax.set_title('Every window position tested\ndotted = cluster threshold',
                 fontsize=10)

    ax = axes[2]
    for i, (label, sliding) in enumerate(sliding_by_condition.items()):
        if sliding is None:
            continue
        colour = condition_colour(label, i)
        mass = sliding['null_mass']
        positive = mass[mass > 0]
        if positive.size:
            ax.hist(positive, bins=40, color=colour, alpha=0.40, lw=0)
        for cluster in sliding['clusters']:
            ax.axvline(cluster['mass'], color=colour, lw=1.6)
    ax.set_yscale('log')
    ax.set_xlabel('Max cluster mass, sign-flipped')
    ax.set_ylabel('Permutations (log)')
    ax.set_title('Permutation null\nlines = observed clusters', fontsize=10)

    fig.suptitle(title, fontsize=12, y=0.98)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_png
