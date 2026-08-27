#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is event rejection systematic with respect to the variables of interest?

This is a confound check, not a QC nicety. Two separable things can bias H1:

  (a) ARTIFACT rejection removes TIME. If planning windows are more
      contaminated than execution windows, exposure differs systematically and
      any rate comparison is biased even with a correct offset -- because the
      surviving clean time is not a random sample of the window.

  (b) SPECTRAL rejection removes EVENTS. If the fraction of candidates
      rejected varies by phase, the surviving ripple count is differentially
      filtered across exactly the conditions being compared.

Either would manufacture (or mask) an H1 effect. Both are tested here against
the task variables: learning phase, whether the repeat was correct, and
proximity to reward delivery.

Usage:
    python scripts/swr_rejection_bias.py
    python scripts/swr_rejection_bias.py --sessions="[38,26,2]"

@author: Svenja Kuchenhoff
"""

import os
import sys
import glob
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_v1"
REWARD_WIN_S = 2.0          # window after a reward-location arrival


def phase_of(beh):
    """Label each repeat by learning phase."""
    lab = np.where(~beh.plan_known.to_numpy(bool), "exploration",
                   np.where(beh.groupby(['session_no', 'grid_no'])['plan_known']
                            .transform(lambda s: s.astype(int).cumsum()).to_numpy() == 1,
                            "first_correct", "later_repeats"))
    return lab


def build_windows(beh):
    """One row per repeat: extent on the session clock plus its labels."""
    w = pd.DataFrame({
        "start_s": beh.new_grid_onset.to_numpy(float),
        "end_s": beh.t_D.to_numpy(float),
        "phase": phase_of(beh),
        "correct": beh.correct.to_numpy(int),
        "rep_correct": beh.rep_correct.to_numpy(int),
        "grid_no": beh.grid_no.to_numpy(int),
    })
    w = w[w.end_s > w.start_s].reset_index(drop=True)
    w["duration_s"] = w.end_s - w.start_s
    return w


def clean_fraction(intervals, a, b):
    """Fraction of [a,b] that survives artifact rejection."""
    if not len(intervals):
        return np.nan
    lo = np.maximum(intervals[:, 0], a)
    hi = np.minimum(intervals[:, 1], b)
    return float(np.clip(hi - lo, 0, None).sum() / max(b - a, 1e-9))


def analyse(sessions=None, analysis_name=ANALYSIS_NAME, save_all=True):
    swr_io.start_log(os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr"), "swr_rejection_bias")
    R = swr_io.get_data_root()
    dirs = sorted(glob.glob(os.path.join(swr_io.derivatives_dir(R), "s*",
                                         "LFP-ripples", analysis_name,
                                         "ripple_events.csv")))
    rows_w, rows_e = [], []
    for p in dirs:
        sess = int(p.split(os.sep)[-4][1:])
        if sessions is not None and sess not in [int(s) for s in sessions]:
            continue
        ev = pd.read_csv(p)
        iv_p = os.path.join(os.path.dirname(p), "clean_intervals.csv")
        iv_all = pd.read_csv(iv_p) if os.path.isfile(iv_p) else pd.DataFrame()
        try:
            beh = swr_io.load_behaviour(sess, data_root=R)
        except Exception:
            continue
        w = build_windows(beh)

        # reward-arrival times, for the reward-proximity test
        rew = np.concatenate([beh[c].to_numpy(float)
                              for c in ("t_A", "t_B", "t_C", "t_D")])
        rew = rew[np.isfinite(rew)]

        for pair_id, evp in ev.groupby("pair_id"):
            iv = (iv_all[iv_all.pair_id == pair_id][["start_s", "stop_s"]].to_numpy()
                  if len(iv_all) else np.zeros((0, 2)))
            t = evp.t_peak_s.to_numpy(float)
            near_rew = np.zeros(len(t), bool)
            if len(rew):
                d = np.abs(t[:, None] - rew[None, :]).min(axis=1)
                near_rew = d <= REWARD_WIN_S

            e = evp.copy()
            e["near_reward"] = near_rew
            idx = np.searchsorted(w.start_s.to_numpy(), t, side='right') - 1
            idx = np.clip(idx, 0, len(w) - 1)
            inside = (t >= w.start_s.to_numpy()[idx]) & (t <= w.end_s.to_numpy()[idx])
            e["phase"] = np.where(inside, w.phase.to_numpy()[idx], "outside")
            e["correct"] = np.where(inside, w.correct.to_numpy()[idx], -1)
            e["session"] = sess; e["pair_id"] = pair_id
            rows_e.append(e)

            for _, ww in w.iterrows():
                rows_w.append({"session": sess, "pair_id": pair_id,
                               "phase": ww.phase, "correct": int(ww.correct),
                               "duration_s": ww.duration_s,
                               "clean_frac": clean_fraction(iv, ww.start_s, ww.end_s)})

    if not rows_e:
        print("no detection output found"); return None
    E = pd.concat(rows_e, ignore_index=True)
    W = pd.DataFrame(rows_w)

    print("\n" + "=" * 76)
    print(" (a) ARTIFACT REJECTION -- does surviving clean TIME vary by condition?")
    print("=" * 76)
    g = W[W.phase != "outside"].groupby("phase")["clean_frac"]
    print(g.agg(['mean', 'std', 'count']).round(3).to_string())
    gc = W.groupby("correct")["clean_frac"].agg(['mean', 'std', 'count']).round(3)
    print("\nby repeat correctness (0 = error, 1 = correct):")
    print(gc.to_string())

    print("\n" + "=" * 76)
    print(" (b) SPECTRAL REJECTION -- does the rejected FRACTION vary by condition?")
    print("=" * 76)
    E["rejected"] = ~E.passed_strict if "passed_strict" in E else ~E.passed
    r = E[E.phase != "outside"].groupby("phase")["rejected"]
    print(r.agg(['mean', 'count']).rename(columns={'mean': 'rejected_frac'})
          .round(3).to_string())
    print("\nby repeat correctness:")
    print(E[E.correct >= 0].groupby("correct")["rejected"]
          .agg(['mean', 'count']).rename(columns={'mean': 'rejected_frac'})
          .round(3).to_string())
    print("\nnear a reward arrival (<=%.0fs) vs not:" % REWARD_WIN_S)
    print(E.groupby("near_reward")["rejected"]
          .agg(['mean', 'count']).rename(columns={'mean': 'rejected_frac'})
          .round(3).to_string())

    # chi-square: is rejection independent of phase?
    from scipy.stats import chi2_contingency
    tab = pd.crosstab(E[E.phase != "outside"].phase, E[E.phase != "outside"].rejected)
    if tab.shape[0] > 1 and tab.shape[1] > 1:
        chi2, pv, dof, _ = chi2_contingency(tab)
        print(f"\nchi2 test, rejection x phase: chi2={chi2:.2f}, dof={dof}, p={pv:.2g}")
        print("  -> " + ("REJECTION IS NOT INDEPENDENT OF PHASE (confound; must be "
                         "modelled)" if pv < 0.05 else
                         "no evidence of phase-dependent rejection"))

    if save_all:
        out = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
        os.makedirs(out, exist_ok=True)
        W.to_csv(os.path.join(out, "rejection_bias_windows.csv"), index=False)
        E.drop(columns=[c for c in E.columns if c.startswith('_')], errors='ignore')\
         .to_csv(os.path.join(out, "rejection_bias_events.csv"), index=False)
        print(f"\nsaved -> {out}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(analyse)
    else:
        analyse()
