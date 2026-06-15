#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Behavioural summary for fMRI + ephys datasets — methods-section ready.

Per dataset we report:
    1) Per-loop time (A→D within a single attempt) per subject, per repeat
       index, and across subjects.
    2) Learning-curve slope (linear regression of loop time on repeat
       index per subject, plus group t vs. 0).
    3) fMRI-only fine-grained timing: median step time, dwell time, and
       button RT per subject.
    4) fMRI-only forw vs. backw comparison (paired t per subject).
    5) Ephys-only completion rate + count of incorrect attempts.
    6) Completeness counts: number of unique reward-location configs and
       correct repeats per config.

"Session" is defined by the rewarded-location tuple (loc_A, loc_B, loc_C,
loc_D), so two halves with the same 4 rewards count as the same session.

Outputs land in
    data/behaviour_summary/<timestamp>/
with a single combined JSON, per-subject CSVs, and overview plots.

@author: Svenja Kuchenhoff
"""

import os
import json
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ── Settings ──────────────────────────────────────────────────────────
DATA_ROOT       = '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
FMRI_BEH_GLOB   = os.path.join(
    DATA_ROOT, 'derivatives', 'sub-*', 'beh', 'sub-*_beh_fmri_clean.csv')
FMRI_EXCLUDE    = {'sub-21', 'sub-29'}    # mirrors clean_fmri_behaviour.py
EPHYS_DERIV     = os.path.join(DATA_ROOT, 'ephys_humans', 'derivatives')
EPHYS_BEH_COLS  = ['rep_correct', 't_A', 't_B', 't_C', 't_D',
                   'loc_A', 'loc_B', 'loc_C', 'loc_D', 'rep_overall',
                   'new_grid_onset', 'session_no', 'grid_no', 'correct']

OUT_BASE        = os.path.join(DATA_ROOT, 'behaviour_summary')
RUN_TAG         = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUT_DIR         = os.path.join(OUT_BASE, RUN_TAG)
PLOT_DIR        = os.path.join(OUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)
print(f"Writing to {OUT_DIR}")


# ── Helpers ───────────────────────────────────────────────────────────
def _describe(x):
    """{mean, sd, median, n} for an array; NaNs ignored."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {'mean': None, 'sd': None, 'median': None, 'n': 0}
    return {
        'mean':   float(np.mean(x)),
        'sd':     float(np.std(x, ddof=1)) if x.size > 1 else None,
        'median': float(np.median(x)),
        'n':      int(x.size),
    }


# ── fMRI side ─────────────────────────────────────────────────────────
def fmri_loop_table(df, sub):
    """One row per (task_half, instruction, loc-tuple, repeat)."""
    rows = []
    grouper = ['task_half', 'instruction', 'task_config_seq', 'repeat']
    for keys, g in df.groupby(grouper, sort=False):
        if not (g['state'] == 'D').any():
            continue
        # Loop time: arrival at D minus arrival at start (first row of this
        # (cfg, repeat)). first row's t_curr_loc is the A-onset.
        t_start = g['t_curr_loc'].dropna().iloc[0] if g['t_curr_loc'].notna().any() else np.nan
        t_d     = g.loc[g['state'] == 'D', 't_curr_rew'].dropna()
        if t_d.empty or not np.isfinite(t_start):
            continue
        loop_t = float(t_d.iloc[-1] - t_start)
        # fine-grained timing.  `button_rts` in the cleaned CSV stores
        # absolute within-run times, not reaction times, so we skip it
        # and rely on dwell + step.
        step_t   = float(g['t_move_to_next_loc'].mean(skipna=True))
        dwell_t  = float(g.loc[g['state'].notna(), 't_dwell_curr_loc']
                         .mean(skipna=True))
        # Floor / ceiling-speed reference.  `length_step` and
        # `reward_delay` are the enforced per-step ISI and reward
        # waiting time set by the experiment (3x3_fMRI_part1.py: jitter()
        # → secs_per_step → wait()).  We sum them over every row of the
        # loop EXCEPT the D-arrival row — its length_step is the forced
        # ISI into the next loop and its reward_delay is the post-D dwell,
        # both of which happen after loop end (t_curr_rew of D).
        d_arrival_idx = g.index[(g['state'] == 'D')
                                & g['t_curr_rew'].notna()]
        if len(d_arrival_idx):
            within_loop = g.drop(d_arrival_idx[-1])
        else:
            within_loop = g
        floor_t  = float(within_loop['t_move_to_next_loc'].fillna(0).sum()
                         + within_loop['reward_delay'].fillna(0).sum())
        rows.append({
            'subject':         sub,
            'task_half':       int(keys[0]),
            'instruction':     keys[1],
            'task_config_seq': keys[2],
            'repeat':          int(keys[3]),
            'loop_time':       loop_t,
            'floor_loop_time': floor_t,
            'slack':           loop_t - floor_t,
            'step_time_mean':  step_t,
            'dwell_time_mean': dwell_t,
        })
    return pd.DataFrame(rows)


def fmri_summarise():
    paths = sorted(glob.glob(FMRI_BEH_GLOB))
    per_loop_all = []
    per_subject = []
    for path in paths:
        sub = os.path.basename(path).split('_')[0]
        if sub in FMRI_EXCLUDE:
            continue
        df = pd.read_csv(path)
        loop_df = fmri_loop_table(df, sub)
        if loop_df.empty:
            print(f"  {sub}: no usable loops — skipped.")
            continue
        per_loop_all.append(loop_df)

        # Per-subject collapse.
        subj_loop = loop_df['loop_time'].to_numpy()
        # Learning slope: regress loop time on repeat index, pooled across
        # halves / configs / instructions.
        slope, intercept, r, p, se = stats.linregress(
            loop_df['repeat'].to_numpy(dtype=float),
            subj_loop)
        # Forw / backw paired contrast averaged per repeat within subject.
        per_rep_dir = (
            loop_df.groupby(['repeat', 'instruction'])['loop_time']
                   .mean().unstack('instruction'))
        if {'forw', 'backw'}.issubset(per_rep_dir.columns):
            paired = per_rep_dir.dropna()
            if len(paired) >= 2:
                t_fb, p_fb = stats.ttest_rel(paired['forw'], paired['backw'])
                forw_minus_backw_mean = float(
                    paired['forw'].mean() - paired['backw'].mean())
            else:
                t_fb = p_fb = np.nan
                forw_minus_backw_mean = np.nan
        else:
            t_fb = p_fb = np.nan
            forw_minus_backw_mean = np.nan

        per_subject.append({
            'subject':            sub,
            'n_loops':            int(len(loop_df)),
            'n_unique_configs':   int(loop_df['task_config_seq'].nunique()),
            'n_repeats_max':      int(loop_df['repeat'].max()) + 1,
            'loop_time_mean':     float(np.nanmean(subj_loop)),
            'loop_time_sd':       float(np.nanstd(subj_loop, ddof=1))
                                  if subj_loop.size > 1 else None,
            'floor_loop_time_mean': float(np.nanmean(
                loop_df['floor_loop_time'])),
            'slack_mean':         float(np.nanmean(loop_df['slack'])),
            'slack_sd':           float(np.nanstd(loop_df['slack'], ddof=1))
                                  if len(loop_df) > 1 else None,
            'loop_time_by_repeat': {
                int(r_): float(np.nanmean(
                    loop_df.loc[loop_df['repeat'] == r_, 'loop_time']))
                for r_ in sorted(loop_df['repeat'].unique())
            },
            'learning_slope':     float(slope),
            'learning_slope_p':   float(p),
            'step_time_median':   float(np.nanmedian(loop_df['step_time_mean'])),
            'dwell_time_median':  float(np.nanmedian(loop_df['dwell_time_mean'])),
            'forw_minus_backw_loop_mean': forw_minus_backw_mean,
            'forw_vs_backw_paired_t':     None if np.isnan(t_fb) else float(t_fb),
            'forw_vs_backw_paired_p':     None if np.isnan(p_fb) else float(p_fb),
        })

    if not per_subject:
        print("No fMRI subjects passed filter.")
        return None, None, None

    all_loops = pd.concat(per_loop_all, ignore_index=True)
    subj_df   = pd.DataFrame(per_subject)
    subj_df.to_csv(os.path.join(OUT_DIR, 'fmri_per_subject.csv'), index=False)
    all_loops.to_csv(os.path.join(OUT_DIR, 'fmri_loops.csv'), index=False)

    # Group-level summary.
    means = subj_df['loop_time_mean'].to_numpy()
    slopes = subj_df['learning_slope'].to_numpy()
    t_slope, p_slope = stats.ttest_1samp(slopes, 0.0)
    # paired t on forw - backw across subjects
    fb = subj_df['forw_minus_backw_loop_mean'].dropna().to_numpy()
    if fb.size >= 2:
        t_fb_grp, p_fb_grp = stats.ttest_1samp(fb, 0.0)
    else:
        t_fb_grp = p_fb_grp = np.nan

    # By repeat across subjects
    loop_by_rep = (
        all_loops.groupby(['subject', 'repeat'])['loop_time'].mean()
                 .unstack('repeat'))
    by_rep = {
        int(r_): _describe(loop_by_rep[r_].to_numpy())
        for r_ in sorted(loop_by_rep.columns)
    }

    group = {
        'n_subjects':            int(len(subj_df)),
        'excluded':              sorted(FMRI_EXCLUDE),
        'loop_time_across_subj': _describe(means),
        'floor_loop_time_across_subj': _describe(
            subj_df['floor_loop_time_mean'].to_numpy()),
        'slack_across_subj':     _describe(
            subj_df['slack_mean'].to_numpy()),
        'loop_time_by_repeat':   by_rep,
        'learning_slope_group':  {
            **_describe(slopes),
            't': float(t_slope), 'p': float(p_slope),
        },
        'step_time_median':      _describe(
            subj_df['step_time_median'].to_numpy()),
        'dwell_time_median':     _describe(
            subj_df['dwell_time_median'].to_numpy()),
        'forw_vs_backw_subj_diff': {
            **_describe(fb),
            't_vs_0': None if np.isnan(t_fb_grp) else float(t_fb_grp),
            'p_vs_0': None if np.isnan(p_fb_grp) else float(p_fb_grp),
        },
        'completeness_n_configs': _describe(
            subj_df['n_unique_configs'].to_numpy()),
    }
    return subj_df, all_loops, group


# ── Ephys side ────────────────────────────────────────────────────────
def ephys_loop_table(df, sub):
    """One row per attempt; loop time = t_D − t_A. Adds session-by-locs."""
    df = df.copy()
    df['loc_tuple'] = df[['loc_A', 'loc_B', 'loc_C', 'loc_D']].astype(int).agg(
        lambda r: f"{r[0]}-{r[1]}-{r[2]}-{r[3]}", axis=1)
    df['subject'] = sub
    df['loop_time'] = df['t_D'] - df['t_A']
    return df


def ephys_summarise():
    sub_dirs = sorted([
        d for d in os.listdir(EPHYS_DERIV)
        if d.startswith('s') and os.path.isdir(
            os.path.join(EPHYS_DERIV, d, 'cells_and_beh'))
    ])
    per_session = []
    all_attempts = []
    for d in sub_dirs:
        sub = d[1:]
        path = os.path.join(EPHYS_DERIV, d, 'cells_and_beh',
                            f'all_trial_times_{sub}.csv')
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path, header=None)
        if df.shape[1] != len(EPHYS_BEH_COLS):
            print(f"  s{sub}: unexpected n_cols={df.shape[1]} — skipped.")
            continue
        df.columns = EPHYS_BEH_COLS
        tbl = ephys_loop_table(df, sub)
        all_attempts.append(tbl)

        correct = tbl[tbl['correct'] == 1]
        if correct.empty:
            print(f"  s{sub}: no correct trials — skipped.")
            continue

        # Learning slope on rep_correct (0-9).
        slope, intercept, r, p, se = stats.linregress(
            correct['rep_correct'].to_numpy(dtype=float),
            correct['loop_time'].to_numpy(dtype=float))

        n_attempts = int(len(tbl))
        n_correct  = int(len(correct))
        n_incorr   = int((tbl['correct'] == 0).sum())
        # "configurations solved" = unique loc-tuples with ≥1 correct trial.
        n_configs_solved = int(
            correct.groupby('loc_tuple').ngroups)
        per_session.append({
            'subject':                  sub,
            'n_attempts':               n_attempts,
            'n_correct':                n_correct,
            'completion_rate':          float(n_correct / n_attempts),
            'n_incorrect_attempts':     n_incorr,
            'incorrect_proportion':     float(n_incorr / n_attempts),
            'n_unique_loc_configs':     int(tbl['loc_tuple'].nunique()),
            'n_configs_solved':         n_configs_solved,
            'incorrect_per_config':     float(
                tbl[tbl['correct'] == 0].groupby('loc_tuple').size().mean()
                if (tbl['correct'] == 0).any() else 0.0),
            'loop_time_mean':         float(correct['loop_time'].mean()),
            'loop_time_sd':           float(correct['loop_time'].std(ddof=1))
                                      if len(correct) > 1 else None,
            'loop_time_by_repeat': {
                int(r_): float(np.nanmean(
                    correct.loc[correct['rep_correct'] == r_, 'loop_time']))
                for r_ in sorted(correct['rep_correct'].unique())
            },
            'learning_slope':         float(slope),
            'learning_slope_p':       float(p),
        })

    if not per_session:
        print("No ephys sessions parsed.")
        return None, None, None

    all_df = pd.concat(all_attempts, ignore_index=True)
    sess_df = pd.DataFrame(per_session)
    sess_df.to_csv(os.path.join(OUT_DIR, 'ephys_per_session.csv'), index=False)
    all_df.to_csv(os.path.join(OUT_DIR, 'ephys_attempts.csv'), index=False)

    # Group summary.  Drop any NaN slopes (sessions with <2 correct reps).
    slopes = sess_df['learning_slope'].to_numpy(dtype=float)
    slopes_ok = slopes[np.isfinite(slopes)]
    t_slope, p_slope = stats.ttest_1samp(slopes_ok, 0.0)

    # Loop time by rep_correct across sessions; cap at 9 (per task design).
    corr_all = all_df[(all_df['correct'] == 1)
                      & (all_df['rep_correct'].between(0, 9))]
    by_rep_session_mean = (
        corr_all.groupby(['subject', 'rep_correct'])['loop_time'].mean()
                .unstack('rep_correct'))
    by_rep = {
        int(r_): _describe(by_rep_session_mean[r_].to_numpy())
        for r_ in sorted(by_rep_session_mean.columns)
    }

    group = {
        'n_sessions':           int(len(sess_df)),
        'loop_time_across_subj': _describe(
            sess_df['loop_time_mean'].to_numpy()),
        'loop_time_by_rep_correct': by_rep,
        'learning_slope_group': {
            **_describe(slopes_ok),
            't': float(t_slope) if np.isfinite(t_slope) else None,
            'p': float(p_slope) if np.isfinite(p_slope) else None,
        },
        'completion_rate':      _describe(
            sess_df['completion_rate'].to_numpy()),
        'n_incorrect_attempts': _describe(
            sess_df['n_incorrect_attempts'].to_numpy()),
        'incorrect_proportion': _describe(
            sess_df['incorrect_proportion'].to_numpy()),
        'incorrect_per_config': _describe(
            sess_df['incorrect_per_config'].to_numpy()),
        'completeness_n_configs': _describe(
            sess_df['n_unique_loc_configs'].to_numpy()),
        'n_configs_solved':     _describe(
            sess_df['n_configs_solved'].to_numpy()),
        'rep_correct_overflow_note': (
            "rep_correct nominally runs 0-9 (10 correct repeats/grid). "
            "12 trials across 9 sessions have rep_correct == 10 "
            "(s05, s07, s11, s13, s18, s20, s24, s26, s27). "
            "Capped at 9 in all summary tables."),
    }
    return sess_df, all_df, group


# ── Plots (quick overview, not publication) ──────────────────────────
_FLOOR_COLOR = '#448363'        # enforced-floor reference colour
_FLOOR_LABEL = 'enforced floor (mean ± s.d.)'


def _plot_loop_by_repeat(by_rep, title, save_path,
                          rep_label='repeat',
                          floor_mean=None, floor_sd=None):
    reps = sorted(by_rep.keys())
    means = [by_rep[r]['mean'] for r in reps]
    sds   = [by_rep[r]['sd'] if by_rep[r]['sd'] is not None else 0
             for r in reps]
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    if floor_mean is not None:
        ax.axhline(floor_mean, color=_FLOOR_COLOR, ls='--', lw=1.4,
                   label=_FLOOR_LABEL)
        if floor_sd is not None:
            ax.axhspan(floor_mean - floor_sd, floor_mean + floor_sd,
                       color=_FLOOR_COLOR, alpha=0.12)
    ax.errorbar(reps, means, yerr=sds, marker='o',
                color='#2B4159', ecolor='#7191A9', capsize=3, lw=2,
                label='observed loop time (mean ± s.d.)')
    ax.set_xlabel(rep_label)
    ax.set_ylabel('ABCD-loop time [s]')
    ax.set_title(title, fontsize=11)
    if floor_mean is not None:
        ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_per_subj_loop(loops_df, x_col, title, save_path,
                          floor_series_col=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    for sub, g in loops_df.groupby('subject'):
        per_rep = g.groupby(x_col)['loop_time'].mean().sort_index()
        ax.plot(per_rep.index, per_rep.values, color='0.7', alpha=0.5, lw=0.7)
    group_mean = loops_df.groupby(x_col)['loop_time'].mean().sort_index()
    ax.plot(group_mean.index, group_mean.values, color='#9A383C', lw=2.5,
            label='observed (group mean)')
    if floor_series_col and floor_series_col in loops_df.columns:
        floor_mean = (loops_df.groupby(x_col)[floor_series_col]
                              .mean().sort_index())
        ax.plot(floor_mean.index, floor_mean.values, color=_FLOOR_COLOR,
                ls='--', lw=2.0, label='enforced floor (group mean)')
    ax.set_xlabel(x_col)
    ax.set_ylabel('ABCD-loop time [s]')
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_hist(values, title, xlabel, save_path, floor_values=None):
    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    # If a floor distribution is supplied, share a common bin range so the
    # two histograms are directly comparable.
    if floor_values is not None:
        floor = np.asarray(floor_values, dtype=float)
        floor = floor[np.isfinite(floor)]
        lo = float(np.nanmin(np.concatenate([vals, floor]))) - 0.5
        hi = float(np.nanmax(np.concatenate([vals, floor]))) + 0.5
        bins = np.linspace(lo, hi, 14)
        ax.hist(vals, bins=bins, color='#BC7E6A', edgecolor='white',
                label='observed')
        ax.hist(floor, bins=bins, color=_FLOOR_COLOR, edgecolor='white',
                alpha=0.55, label='enforced floor')
        ax.legend(frameon=False, fontsize=9)
    else:
        ax.hist(vals, bins=12, color='#BC7E6A', edgecolor='white')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('# subjects')
    ax.set_title(title, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ── Run ───────────────────────────────────────────────────────────────
print("\n=== fMRI ===")
fmri_subj, fmri_loops, fmri_group = fmri_summarise()
print("\n=== Ephys ===")
eph_sess, eph_attempts, eph_group = ephys_summarise()

# Plots.
if fmri_group is not None:
    floor_mean = fmri_group['floor_loop_time_across_subj']['mean']
    floor_sd   = fmri_group['floor_loop_time_across_subj']['sd']
    _plot_loop_by_repeat(
        fmri_group['loop_time_by_repeat'],
        'fMRI — loop time across repeats',
        os.path.join(PLOT_DIR, 'fmri_loop_by_repeat.png'),
        floor_mean=floor_mean, floor_sd=floor_sd)
    _plot_per_subj_loop(
        fmri_loops, 'repeat',
        f'fMRI — per-subject loop time (n={fmri_group["n_subjects"]})',
        os.path.join(PLOT_DIR, 'fmri_per_subject_loop.png'),
        floor_series_col='floor_loop_time')
    _plot_hist(
        fmri_subj['loop_time_mean'].to_numpy(),
        'fMRI — subject-mean loop time',
        'loop time [s]',
        os.path.join(PLOT_DIR, 'fmri_subj_mean_hist.png'),
        floor_values=fmri_subj['floor_loop_time_mean'].to_numpy())
    _plot_hist(
        fmri_subj['slack_mean'].to_numpy(),
        'fMRI — per-subject voluntary slack (observed − floor)',
        'slack [s]',
        os.path.join(PLOT_DIR, 'fmri_slack_hist.png'))
    # Observed loop time vs. enforced floor — slack = headroom to speed up.
    fig, ax = plt.subplots(figsize=(4.6, 4.4), constrained_layout=True)
    ax.scatter(fmri_subj['floor_loop_time_mean'],
               fmri_subj['loop_time_mean'],
               color='#2B4159', s=24)
    lo = min(fmri_subj['floor_loop_time_mean'].min(),
             fmri_subj['loop_time_mean'].min()) * 0.95
    hi = max(fmri_subj['floor_loop_time_mean'].max(),
             fmri_subj['loop_time_mean'].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='y = x (no slack)')
    ax.set_xlabel('enforced floor [s] (sum length_step + reward_delay)')
    ax.set_ylabel('observed loop time [s]')
    ax.set_title('fMRI — observed vs. enforced floor per subject',
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(os.path.join(PLOT_DIR, 'fmri_floor_vs_observed.png'),
                dpi=200)
    plt.close(fig)
    # forw vs backw subj-level scatter
    fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)
    forw = (fmri_loops[fmri_loops['instruction'] == 'forw']
            .groupby('subject')['loop_time'].mean())
    backw = (fmri_loops[fmri_loops['instruction'] == 'backw']
             .groupby('subject')['loop_time'].mean())
    common = forw.index.intersection(backw.index)
    ax.scatter(forw.loc[common], backw.loc[common], color='#2B4159')
    lim_lo = min(forw.loc[common].min(), backw.loc[common].min()) * 0.95
    lim_hi = max(forw.loc[common].max(), backw.loc[common].max()) * 1.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', lw=0.8)
    ax.set_xlabel('forw mean loop time [s]')
    ax.set_ylabel('backw mean loop time [s]')
    ax.set_title('fMRI — forw vs. backw per subject', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(os.path.join(PLOT_DIR, 'fmri_forw_vs_backw.png'), dpi=200)
    plt.close(fig)

if eph_group is not None:
    _plot_loop_by_repeat(
        eph_group['loop_time_by_rep_correct'],
        'Ephys — loop time across correct repeats',
        os.path.join(PLOT_DIR, 'ephys_loop_by_rep_correct.png'),
        rep_label='rep_correct (0-9)')
    # Cap at rep_correct ≤ 9 so the rare overflow grids (12 trials across
    # 9 sessions) don't drive the y-axis.
    correct = eph_attempts[(eph_attempts['correct'] == 1)
                           & (eph_attempts['rep_correct'].between(0, 9))].copy()
    _plot_per_subj_loop(
        correct, 'rep_correct',
        f'Ephys — per-session loop time (n={eph_group["n_sessions"]})',
        os.path.join(PLOT_DIR, 'ephys_per_session_loop.png'))
    _plot_hist(
        eph_sess['completion_rate'].to_numpy(),
        'Ephys — session completion rate',
        'fraction correct',
        os.path.join(PLOT_DIR, 'ephys_completion_hist.png'))
    _plot_hist(
        eph_sess['incorrect_proportion'].to_numpy(),
        'Ephys — proportion of incorrect attempts per session',
        'fraction incorrect',
        os.path.join(PLOT_DIR, 'ephys_incorrect_proportion_hist.png'))
    _plot_hist(
        eph_sess['n_incorrect_attempts'].to_numpy(),
        'Ephys — # incorrect attempts per session',
        '# incorrect attempts',
        os.path.join(PLOT_DIR, 'ephys_incorrect_hist.png'))
    _plot_hist(
        eph_sess['n_configs_solved'].to_numpy(),
        'Ephys — # reward configurations solved per session',
        '# configurations solved (≥1 correct trial)',
        os.path.join(PLOT_DIR, 'ephys_configs_solved_hist.png'))


# ── Final combined JSON ──────────────────────────────────────────────
combined = {
    'meta': {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'out_dir':   OUT_DIR,
        'definitions': {
            'loop_time': "fMRI: t_curr_rew(D) − t_curr_loc of first row of "
                         "that (cfg, repeat). Ephys: t_D − t_A per attempt.",
            'session':   "Defined by the rewarded-location tuple "
                         "(loc_A, loc_B, loc_C, loc_D).",
            'fmri_exclude': sorted(FMRI_EXCLUDE),
            'ephys_scope':  "All sessions with cells_and_beh/all_trial_times_*.csv (n=63).",
            'ephys_loop_time_repeat_axis': "rep_correct (0-9), correct trials only.",
        },
    },
    'fmri':  fmri_group,
    'ephys': eph_group,
}
out_json = os.path.join(OUT_DIR, 'behaviour_summary.json')
with open(out_json, 'w') as f:
    json.dump(combined, f, indent=2,
              default=lambda o: float(o) if hasattr(o, 'item') else str(o))
print(f"\nWrote {out_json}")
print(f"All outputs under: {OUT_DIR}")
