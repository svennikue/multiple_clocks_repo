#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ripple tests — the conditions we settled on, tested and plotted.

The question: hippocampal ripples are how the hippocampus tells mPFC what the
plan is. So ripples should appear when a piece of route information is acquired,
and not when nothing is learned.

Each test below defines a set of conditions, aligns ripples to those events, and
asks one question: does ripple rate depart from THAT SAME TRIAL'S baseline?
Comparing a condition with its own baseline rather than with another condition
is what makes conditions of different overall rate comparable — an earlier
version of this analysis compared rates between conditions and missed the effect
entirely, because the first traversal has a lower floor.

No window is chosen. Every position of a sliding window is tested and the
multiple comparisons across positions are corrected by a cluster permutation.

Tests:
    1) stage          uncovering D, by learning stage
    2) reward         each reward A-D, first traversal only
    3) feedback       correct vs error, pooled
    4) feedback_stage feedback valence crossed with learning stage
    5) reward_feedback  correct vs error for each reward
    6) full           reward x valence x stage (cells with enough events)

Outputs, per test, in <out_dir>/:
    <test>.png            rate over time, the sliding test, the null
    <test>_result.json    hypothesis, every number, the conclusion
    <test>_counts.csv     subjects, sessions, derivations, events per condition

    python scripts/swr_ripple_tests.py --bundle=<bundle dir>
    python scripts/swr_ripple_tests.py --bundle=<dir> --tests="['feedback_stage']"

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.ripples as rip

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)


# ── Settings ──────────────────────────────────────────────────────────
ALL_TESTS = ('stage', 'reward', 'feedback', 'feedback_stage',
             'reward_feedback', 'full')
NAMED_WINDOWS = {'pre  (-0.6..-0.1)': (-0.6, -0.1),
                 'post (0..0.5)': (0.0, 0.5),
                 'post (0.5..1.0)': (0.5, 1.0)}

QUESTIONS = {
    'stage':  'Do ripples rise after uncovering D, and only the first time?',
    'reward': 'Is the rise specific to D, or does any first reward do it?',
    'feedback': 'Does learning something (correct) raise ripples and learning '
                'nothing (error) lower them?',
    'feedback_stage': 'Is that valence effect specific to when the route is '
                      'still unknown?',
    'reward_feedback': 'Does it matter which reward was being sought?',
    'full': 'Reward x valence x stage, for cells with enough events.',
}


# ── Conditions ────────────────────────────────────────────────────────
# Each returns {condition label: {session: event times}}. Nothing else in this
# script knows how a condition is built, so adding one is a single function.

def conditions_stage(bundle):
    """Uncovering D, split by learning stage."""
    out = {f'D, {s}': {} for s in rip.STAGES}
    for session in rip.sessions_in(bundle):
        table = rip.uncover_table(bundle, session)
        if not len(table):
            continue
        hit = table.query("valence == 'correct' and reward == 'D'")
        for stage, g in hit.groupby('stage'):
            out[f'D, {stage}'][session] = g.t_s.to_numpy(float)
    return out


def conditions_reward(bundle):
    """Each reward A-D, on the first traversal only."""
    out = {f'first {r}': {} for r in rip.REWARDS}
    for session in rip.sessions_in(bundle):
        table = rip.uncover_table(bundle, session)
        if not len(table):
            continue
        hit = table.query("valence == 'correct' and stage == 'first uncovers'")
        for reward, g in hit.groupby('reward'):
            out[f'first {reward}'][session] = g.t_s.to_numpy(float)
    return out


def conditions_feedback(bundle):
    """Correct vs error uncoverings, pooled over stage and reward."""
    out = {v: {} for v in rip.VALENCE}
    for session in rip.sessions_in(bundle):
        table = rip.uncover_table(bundle, session)
        if not len(table):
            continue
        for valence, g in table.groupby('valence'):
            out[valence][session] = g.t_s.to_numpy(float)
    return out


def conditions_feedback_stage(bundle):
    """Valence crossed with learning stage."""
    out = {f'{v}, {s}': {} for v in rip.VALENCE for s in rip.STAGES}
    for session in rip.sessions_in(bundle):
        table = rip.uncover_table(bundle, session)
        if not len(table):
            continue
        for (valence, stage), g in table.groupby(['valence', 'stage']):
            out[f'{valence}, {stage}'][session] = g.t_s.to_numpy(float)
    return out


def conditions_reward_feedback(bundle):
    """Correct vs error for each reward being sought."""
    out = {f'{v} {r}': {} for r in rip.REWARDS for v in rip.VALENCE}
    for session in rip.sessions_in(bundle):
        table = rip.uncover_table(bundle, session)
        if not len(table):
            continue
        for (valence, reward), g in table.groupby(['valence', 'reward']):
            out[f'{valence} {reward}'][session] = g.t_s.to_numpy(float)
    return out


def conditions_full(bundle):
    """Reward x valence x stage."""
    out = {}
    for session in rip.sessions_in(bundle):
        table = rip.uncover_table(bundle, session)
        if not len(table):
            continue
        for (valence, reward, stage), g in table.groupby(
                ['valence', 'reward', 'stage']):
            out.setdefault(f'{valence} {reward}, {stage}', {})[session] = \
                g.t_s.to_numpy(float)
    return out


BUILDERS = {'stage': conditions_stage, 'reward': conditions_reward,
            'feedback': conditions_feedback,
            'feedback_stage': conditions_feedback_stage,
            'reward_feedback': conditions_reward_feedback,
            'full': conditions_full}


# ── One test ──────────────────────────────────────────────────────────

def run_one_test(bundle, name, out_dir, n_perm=rip.N_SIGN_FLIPS,
                 correct_over='time'):
    print(f"\n{'=' * 74}\n {name}: {QUESTIONS[name]}\n{'=' * 74}")
    by_condition = BUILDERS[name](bundle)

    profiles, counts, centres = {}, {}, None
    for label, per_session in by_condition.items():
        centres_i, per_subject, count = rip.rate_by_subject(bundle, per_session)
        if len(per_subject) < 3:
            print(f"  {label:34s} skipped ({count['n_events_used']} events, "
                  f"{len(per_subject)} subjects)")
            continue
        centres = centres_i
        profiles[label] = per_subject
        counts[label] = count
    if not profiles:
        print("  nothing computable")
        return None

    print(f"  {'condition':34s} {'subj':>5s} {'sess':>5s} {'deriv':>6s} "
          f"{'events':>7s}")
    for label, count in counts.items():
        print(f"  {label:34s} {count['n_subjects']:5d} {count['n_sessions']:5d} "
              f"{count['n_derivations']:6d} {count['n_events_used']:7d}")

    results = {'test': name, 'question': QUESTIONS[name],
               'conditions': {}, 'baseline_window_s': list(rip.BASELINE_WIN)}
    sliding_for_plot = {}

    what = ('window positions AND conditions'
            if correct_over == 'time_and_conditions' else 'window positions')
    print(f"\n  sliding window, cluster-corrected over {what} "
          f"({n_perm} sign-flips, no window chosen)")
    baselined = {}
    for label, per_subject in profiles.items():
        subjects, X = rip.baseline_subtract(per_subject, centres)
        baselined[label] = {s: X[i] for i, s in enumerate(subjects)}
    for width in rip.SLIDE_WIDTHS_S:
        sliding = rip.sliding_window_test(baselined, centres, width_s=width,
                                          n_perm=n_perm,
                                          correct_over=correct_over)
        for label, res in sliding.items():
            keep = [c for c in res['clusters'] if c['p'] < 0.05]
            results['conditions'].setdefault(label, {}) \
                .setdefault('sliding', {})[f'{width:g}s'] = keep
            if width == rip.SLIDE_WIDTHS_S[0]:
                sliding_for_plot[label] = res
            if keep:
                for c in keep:
                    print(f"    {width:g}s  {label:32s} {c['direction']:8s} "
                          f"{c['start_s']:+.2f}..{c['stop_s']:+.2f} s "
                          f"(peak {c['peak_s']:+.2f}, p = {c['p']:.4f})")
            else:
                print(f"    {width:g}s  {label:32s} none")

    print(f"\n  named windows vs the same trial's baseline "
          f"(reported, not primary)")
    for label, per_subject in profiles.items():
        for wname, window in NAMED_WINDOWS.items():
            got = rip.window_test(per_subject, centres, window)
            if got is None:
                continue
            results['conditions'][label].setdefault('windows', {})[wname] = got
            star = ' *' if got['p_perm'] < 0.05 else ''
            print(f"    {label:32s} {wname:20s} t = {got['t']:+6.2f}  "
                  f"p_perm = {got['p_perm']:.4g}{star}")

    survived = {lab: [c for w in d.get('sliding', {}).values() for c in w]
                for lab, d in results['conditions'].items()}
    survived = {k: v for k, v in survived.items() if v}
    conclusion = ('; '.join(
        f"{k}: {v[0]['direction']} {v[0]['start_s']:+.2f}..{v[0]['stop_s']:+.2f} s "
        f"(p = {v[0]['p']:.4f})" for k, v in survived.items())
        or 'no condition shows a cluster surviving correction')

    os.makedirs(out_dir, exist_ok=True)
    swr_io.write_result(out_dir, name, hypothesis=QUESTIONS[name],
                        tests=results['conditions'], conclusion=conclusion,
                        extra={'baseline_window_s': list(rip.BASELINE_WIN),
                               'bin_s': rip.BIN_S, 'dedup_s': rip.DEDUP_S,
                               'sliding_widths_s': list(rip.SLIDE_WIDTHS_S),
                               'n_sign_flips': n_perm,
                               'corrected_over': what,
                               'min_events_per_condition': rip.MIN_EVENTS})
    counts_df = pd.DataFrame(counts).T.reset_index().rename(
        columns={'index': 'condition'})
    counts_df.to_csv(os.path.join(out_dir, f'{name}_counts.csv'), index=False)
    rip.plot_condition(centres, profiles, sliding_for_plot,
                       f'{name}: {QUESTIONS[name]}',
                       os.path.join(out_dir, f'{name}.png'))
    print(f"\n  conclusion: {conclusion}")
    print(f"  wrote {name}.png, {name}_result.json, {name}_counts.csv")
    return results


# ── Main ──────────────────────────────────────────────────────────────

def run(bundle=None, tests=None, out_dir=None, n_perm=rip.N_SIGN_FLIPS,
        correct_over='time'):
    """correct_over: 'time' (per condition) or 'time_and_conditions' (family)."""
    if bundle is None:
        bundle = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                              'group', 'swr', 'bundle')
    if out_dir is None:
        out_dir = os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()),
                               'group', 'swr', 'ripple_tests')
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, 'swr_ripple_tests')
    wanted = [tests] if isinstance(tests, str) else (list(tests) if tests
                                                     else list(ALL_TESTS))

    data = rip.load_bundle(bundle)
    ripples = data['ripples']
    qc = data['channel_qc']
    qc = qc[~qc.excluded.fillna(False)] if 'excluded' in qc else qc
    print(f"\n  bundle: {bundle}")
    print(f"  {ripples.session.nunique()} sessions | "
          f"{ripples.subject_key.nunique()} subjects | {len(qc)} derivations | "
          f"{len(ripples)} ripples | {qc.clean_s.sum() / 3600:.1f} h clean")

    summary = {}
    for name in wanted:
        summary[name] = run_one_test(data, name, out_dir, n_perm=n_perm,
                                     correct_over=correct_over)

    with open(os.path.join(out_dir, 'all_tests.json'), 'w') as f:
        json.dump({'created': datetime.now().isoformat(timespec='seconds'),
                   'bundle': bundle, 'tests': summary}, f, indent=2,
                  default=str)
    print(f"\n saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(run)
    else:
        run()
