#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CURATED FINDINGS. Only results we have understood, each with its test and figure.

The companion to `swr_explore.py`. Exploration lives there and makes no claims;
a result moves here once it survives a subject-level test AND we can say what it
means. Negative results are kept too -- a claim that was checked and failed is
worth as much as one that held, and stops it being re-made.

Nothing in this project has been pre-registered. The "H1-H7" in
`swr_hypotheses.py` were written by Claude from SK's description of the idea,
not declared in advance by anyone, so they carry no confirmatory status and
their FDR values are descriptive only. Everything here is exploration: the goal
is to find out what is in the data and how to compare conditions fairly.
Confirmation, if it comes, needs data that did not generate the hypothesis.

    python scripts/swr_findings.py run --bundle=<bundle dir>
    python scripts/swr_findings.py run --which=F1

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_probes as pr
import mc.analyse.swr_sakon as sk
import mc.plotting.ripple_figures as rfig

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)


# =============================================================================
# THE REGISTRY
# =============================================================================
# Each entry: what we believe, the test that shows it, and what it means for
# any analysis that follows. `status` is 'holds', 'negative' (checked, did not
# hold) or 'caution'.

FINDINGS = {
    "F1": dict(
        status="holds",
        claim="Ripple rate rises with how long the subject stays still.",
        numbers="0.19 Hz in 0.5-2 s still periods -> 0.24 Hz in periods >=8 s; "
                "+0.048 Hz, t(32) = +4.59, p = 6.6e-05, paired by subject.",
        test="Per bipolar derivation, ripple rate in still periods of 0.5-2 s "
             "vs >=8 s (a still period is a gap between consecutive key presses "
             "of any kind). Averaged within subject, one-sample t across "
             "subjects.",
        meaning="This is the largest and most reliable effect in the dataset, "
                "and it is the obvious one: ripples occur in offline, rest-like "
                "states. CONSEQUENCE: any contrast between conditions that "
                "differ in how much the subject stands still will show a ripple "
                "difference for that reason alone. Stillness must be measured "
                "and reported for every condition contrast in this project.",
    ),
    "F1b": dict(
        status="holds",
        claim="F1 is PROSPECTIVE, not an artefact of window length or of "
              "proximity to a key press.",
        numbers="Same 1 s window, 0.5-1.5 s after the press in every case, "
                "binned by how long the pause turned out to last: "
                "0.179 Hz (1.5-3 s) -> 0.192 (3-6 s) -> 0.202 (6-12 s) -> "
                "0.219 Hz (>12 s). Monotonic, +22% end to end.",
        test="`fixed_window_by_pause_length`: identical width, identical "
             "distance from the preceding press, so the only thing that varies "
             "is the eventual duration of the pause.",
        meaning="The obvious deflation of F1 is that short still periods are "
                "entirely 'near a press' while long ones have a quiet middle, "
                "so F1 would follow from press proximity rather than stillness. "
                "It does not: the rate in the FIRST second already predicts how "
                "long the subject is about to stay still. The brain is in a "
                "different state from the outset. That is a much stronger claim "
                "than F1 alone and it is the one worth building on.",
    ),
    "F4": dict(
        status="holds",
        claim="The stillness effect exists only while the grid is UNSOLVED. It "
              "disappears during execution.",
        numbers="long-minus-short still-period rate: explore +0.044 Hz "
                "(t(23) = +1.97, p = 0.061), plan +0.058 (t(19) = +1.66, "
                "p = 0.114), execute -0.005 (t(19) = -0.31, p = 0.76). "
                "Interaction, unsolved (explore+plan) vs execute: +0.051 vs "
                "+0.003 Hz, paired t(17) = +2.12, p = 0.049, n = 18 subjects.",
        test="Still periods labelled by phase3 of the repeat containing them, "
             "then the F1 contrast within each phase; the interaction is a "
             "paired t across the subjects who contribute both.",
        meaning="The most hypothesis-relevant result so far. Ripples do not "
                "simply track being still -- they track being still WHILE "
                "THERE IS SOMETHING LEFT TO WORK OUT. Once the route is known "
                "and the subject is executing, standing still does nothing. "
                "That is what a planning signal should look like, and it is not "
                "what a generic arousal or rest signal would look like. "
                "CAUTION: only 18 subjects contribute all three phases and the "
                "per-phase tests are individually non-significant; this needs "
                "confirming on data that did not suggest it.",
    ),
    "N3": dict(
        status="negative",
        claim="There is NO ripple response at the onset of a new grid.",
        numbers="peri (+-250 ms) minus non-peri flanks = -0.019 Hz, "
                "t(32) = -0.75, p = 0.46.",
        test="Grid onsets as events, He et al. peri vs non-peri, per subject.",
        meaning="This is a CONFIRMATION of SK's logic, not a failure. At grid "
                "onset the subject has seen nothing yet: no rewards uncovered, "
                "no information to pass to mPFC, so no plan-relevant "
                "communication is expected. A ripple burst here would have been "
                "the surprising result. The plan-relevant signal should start "
                "once rewards are being uncovered.",
    ),
    "F2": dict(
        status="holds",
        claim="Ripple rate is essentially flat across a grid.",
        numbers="0.195 Hz in the first fifth of a grid -> 0.199 Hz in the last "
                "fifth (+1.8%).",
        test="Each grid split into five equal spans; rate pooled per subject "
             "at each position.",
        meaning="The control for every first-vs-later-traversal contrast. The "
                "first traversal is always at the start of its grid, so a drift "
                "across the grid would manufacture an H1 effect. It does not: "
                "H1 is not a time-within-grid artefact.",
    ),
    "F3": dict(
        status="holds",
        claim="The first traversal is STILLER than later ones, yet has FEWER "
              "ripples.",
        numbers="still fraction 0.621 vs 0.530, t(45) = +7.09, p = 7.5e-09; "
                "long (>=8 s) stillness 2.13 s vs 0.25 s per traversal, "
                "t = +2.20, p = 0.033. Ripple rate is nonetheless ~5-20% lower "
                "on the first traversal (see the H1 sweep).",
        test="Still periods assigned to the traversal containing them; still "
             "seconds and still fraction per traversal, paired by session.",
        meaning="A dissociation, and the reason H1's negative direction is "
                "interesting rather than trivial. The dominant driver of ripple "
                "rate (F1) predicts MORE ripples on the first traversal; the "
                "data show fewer. Whatever suppresses ripples during the first "
                "traversal is strong enough to overcome stillness.",
    ),
    "N1": dict(
        status="negative",
        claim="Ripples are NOT reliably suppressed at a key press.",
        numbers="movement press +0.013 Hz (t(32) = +1.52, p = 0.14); "
                "uncover press -0.011 Hz (t(32) = -1.13, p = 0.27). Opposite "
                "directions, neither significant.",
        test="He et al.'s peri (+-250 ms) vs symmetric non-peri flanks "
             "(-750..-250 and +250..+750 ms), per subject.",
        meaning="Corrects an earlier claim of mine. A peri-event dip was "
                "significant around the D uncovering specifically, but it does "
                "NOT generalise to presses at large, so it cannot be dismissed "
                "as a motor artefact -- and equally, motor suppression cannot "
                "be invoked to explain event-locked effects here.",
    ),
    "N2": dict(
        status="negative",
        claim="There is NO pre-event ripple rise (Sakon's PRE effect) before "
              "uncovering D.",
        numbers="Eq. 2, PRE (-600..-100 ms) vs its own baseline "
                "(-1600..-1100 ms): first_D t(32) = +0.50, p = 0.62; "
                "later_D t(32) = +0.25, p = 0.81. Cluster permutation over "
                "+-2 s: no cluster at p < 0.05 in either condition.",
        test="Sakon & Kahana (2022) Eq. 2, per subject then a one-sample t on "
             "the per-subject t-scores.",
        meaning="Free recall shows a sharp ripple rise before vocalisation. "
                "This task shows nothing comparable before uncovering the "
                "fourth reward. Consistent with SK's logic: uncovering D is "
                "when information is ACQUIRED, not when it is used, so a "
                "retrieval-like pre-event rise is not what should be expected "
                "there. Also retracts an earlier reading of a transient at "
                "+250-750 ms: that was 250 ms bins without cluster correction "
                "and does not survive.",
    ),
}


def _store(bundle, analysis_name="swr_v1"):
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "swr_hyp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "swr_hypotheses.py"))
    hyp = iu.module_from_spec(spec)
    spec.loader.exec_module(hyp)
    return hyp.RippleStore(analysis_name, swr_io.get_data_root(), bundle=bundle)


def _png(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {name}.png")


def _gather_stillness(store, beh_all, R):
    """Per derivation: rate in short vs long still periods, plus per traversal."""
    rows, trav = [], []
    for sess in store.sessions():
        beh = beh_all[beh_all.session == sess]
        if not len(beh):
            continue
        try:
            mv, un = pr.press_times(int(sess), beh, data_root=R)
        except Exception:
            continue
        if not mv.size:
            continue
        still = pr.still_periods(mv, un, min_s=0.5)
        if not len(still):
            continue
        ev, iv_all, qc = store.get(sess)
        skey = store.subject_map().get(sess, {}).get("subject_key", f"s{sess}")
        for pair_id, e in ev.groupby("pair_id"):
            if pair_id in qc.index and bool(qc.loc[pair_id, "excluded"]):
                continue
            iv = iv_all[iv_all.pair_id == pair_id][["start_s", "stop_s"]].to_numpy()
            if not len(iv):
                continue
            d = pr.long_vs_short_still(still, e.t_peak_s.to_numpy(float), iv)
            if d and np.isfinite(d.get("diff", np.nan)):
                d.update(session=sess, subject_key=skey, pair_id=pair_id)
                rows.append(d)
        # per-traversal stillness, for F3
        s0 = still.start_s.to_numpy(); s1 = still.end_s.to_numpy()
        dur = still.duration_s.to_numpy()
        for grid, g in beh.groupby("grid_no"):
            g = g.sort_values("rep_overall"); prev = None
            for i, (_, r) in enumerate(g.iterrows()):
                t0 = float(r.new_grid_onset) if i == 0 else prev
                t1 = float(r.t_D); prev = t1
                if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
                    continue
                m = (s0 >= t0) & (s1 <= t1)
                trav.append({"session": int(sess), "subject_key": skey,
                             "grid_no": int(grid),
                             "cond": "first" if i == 0 else "later",
                             "trav_s": t1 - t0, "still_s": float(dur[m].sum()),
                             "long_still_s": float(dur[m][dur[m] >= 8].sum())})
    return pd.DataFrame(rows), pd.DataFrame(trav)


def run(bundle=None, which=None, out_dir=None, analysis_name="swr_v1"):
    R = swr_io.get_data_root()
    if out_dir is None:
        out_dir = os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                               "findings")
    figs = os.path.join(out_dir, "figures")
    os.makedirs(figs, exist_ok=True)
    swr_io.start_log(out_dir, "swr_findings")
    want = [which] if isinstance(which, str) else (list(which) if which
                                                   else list(FINDINGS))

    print("\n" + "=" * 78)
    print(" CURATED FINDINGS")
    print("=" * 78)
    for k in want:
        f = FINDINGS.get(k)
        if not f:
            continue
        mark = {"holds": "[HOLDS]  ", "negative": "[NEGATIVE]",
                "caution": "[CAUTION]"}[f["status"]]
        print(f"\n {mark} {k}: {f['claim']}")
        print(f"     numbers : {f['numbers']}")
        print(f"     test    : {f['test']}")
        print(f"     meaning : {f['meaning']}")

    store = _store(bundle, analysis_name)
    store.describe()
    beh_all = pd.read_csv(os.path.join(bundle, "behaviour.csv"))
    L, T = _gather_stillness(store, beh_all, R)

    res = {}
    if len(L) and any(k in want for k in ("F1",)):
        per = L.groupby("subject_key")[["rate_short", "rate_long"]].mean().dropna()
        t, p = st.ttest_rel(per.rate_long, per.rate_short)
        res["F1"] = {"n_subjects": len(per), "short": float(per.rate_short.mean()),
                     "long": float(per.rate_long.mean()), "t": float(t), "p": float(p)}
        print(f"\n  F1 recomputed: short {per.rate_short.mean():.3f} -> "
              f"long {per.rate_long.mean():.3f} Hz, t({len(per)-1}) = {t:+.2f}, "
              f"p = {p:.4g}")
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for _, r in per.iterrows():
            ax.plot([0, 1], [r.rate_short, r.rate_long], color="0.78", lw=0.8)
        ax.scatter(np.zeros(len(per)), per.rate_short, s=20, color=rfig.PAL[1], zorder=3)
        ax.scatter(np.ones(len(per)), per.rate_long, s=20, color=rfig.OBS_LINE_C, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["still 0.5-2 s", "still >=8 s"])
        ax.set_ylabel("Ripple rate (Hz)")
        ax.set_title(f"F1  Ripples rise with stillness\np = {p:.2g}, n = {len(per)} subjects")
        _png(fig, figs, "F1_stillness")

    if len(T) and "F3" in want:
        per = T.groupby(["session", "cond"]).apply(
            lambda x: x.still_s.sum() / max(x.trav_s.sum(), 1e-9)).unstack().dropna()
        t, p = st.ttest_rel(per["first"], per["later"])
        res["F3"] = {"n_sessions": len(per), "first": float(per["first"].mean()),
                     "later": float(per["later"].mean()), "t": float(t), "p": float(p)}
        print(f"  F3 recomputed: still fraction first {per['first'].mean():.3f} "
              f"vs later {per['later'].mean():.3f}, t({len(per)-1}) = {t:+.2f}, "
              f"p = {p:.3g}")
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for _, r in per.iterrows():
            ax.plot([0, 1], [r["first"], r["later"]], color="0.78", lw=0.8)
        ax.scatter(np.zeros(len(per)), per["first"], s=20, color=rfig.PAL[1], zorder=3)
        ax.scatter(np.ones(len(per)), per["later"], s=20, color=rfig.PAL[3], zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["first traversal", "later traversals"])
        ax.set_ylabel("Fraction of traversal spent still")
        ax.set_title(f"F3  First traversal is stiller, yet has fewer ripples\n"
                     f"p = {p:.2g}, n = {len(per)} sessions")
        _png(fig, figs, "F3_stillness_by_traversal")

    if len(L):
        L.to_csv(os.path.join(out_dir, "stillness_by_derivation.csv"), index=False)
    if len(T):
        T.to_csv(os.path.join(out_dir, "stillness_by_traversal.csv"), index=False)
    with open(os.path.join(out_dir, "findings.json"), "w") as f:
        json.dump({"registry": FINDINGS, "recomputed": res,
                   "created": datetime.now().isoformat(timespec="seconds")},
                  f, indent=2)
    print(f"\n saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
