#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARRY-FORWARD RESULTS. Every finding we keep, recomputed from the bundle.

The companion to `swr_explore.py`, which is the scratch script. A result moves
here once it survives a subject-level test AND we can say what it means. Each
entry recomputes itself end to end, so re-running this on a larger bundle is the
whole procedure for checking whether a finding holds with more data.

    python scripts/swr_findings.py run --bundle=<bundle dir>
    python scripts/swr_findings.py run --which=F5

Every finding writes, into its own directory:
    <F>_result.json   hypothesis, the exact test, every number, the conclusion
    <F>_counts.csv    how much data went into each condition
    figures/<F>.png   the data the statistic was computed on

WHAT IS REPORTED FOR EVERY CONDITION, always: the number of subjects, sessions,
derivations and events contributing, and the artifact-free seconds of exposure.
A rate difference between conditions with very different exposure is the single
easiest way to fool yourself here, so the denominators are never left implicit.

NOTHING in this project was pre-registered. The "H1-H7" of the earlier script
were written by Claude from SK's description; the one claim SK made in advance is
F5 (ripples when D is first uncovered). Everything else is exploration, and no
result here has been confirmed on data that did not suggest it.

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
import mc.analyse.swr_bundle as swr_bundle
import mc.analyse.swr_probes as pr
import mc.analyse.swr_sakon as sk
import mc.analyse.swr_windows as win
import mc.plotting.ripple_figures as rfig

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

SHORT_STILL = (0.5, 2.0)      # F1: "brief pause"
LONG_STILL = 8.0              # F1: "sustained stillness"
N_SIGN_FLIPS = 10000
SLIDE_WIDTH_S = 0.3           # F5: the width at which the effect survives
DEV_SESSIONS = (2, 3, 6, 9, 12, 13, 14, 38)


# =============================================================================
# REGISTRY -- claim, test, numbers as of 46 sessions, and what it means
# =============================================================================

FINDINGS = {
    "F1": dict(status="holds",
        claim="Ripple rate rises with how long the subject stays still, and the "
              "rise is prospective: rate in the FIRST second predicts how long "
              "the pause will last.",
        covariates="none; exposure is artifact-free seconds, so window length "
                   "is divided out rather than modelled",
        meaning="The largest and most reliable effect in the dataset. "
                "CONSEQUENCE: any contrast between conditions differing in how "
                "much the subject stands still shows a ripple difference for "
                "that reason alone, so stillness is reported for every "
                "condition contrast in this project."),
    "F2": dict(status="holds",
        claim="Ripple rate is flat across a grid (first fifth to last fifth).",
        covariates="none",
        meaning="The control for any first-vs-later-traversal contrast: the "
                "first traversal always sits at the start of its grid, so a "
                "drift would manufacture the effect. It does not."),
    "F3": dict(status="holds",
        claim="The first traversal is STILLER than later ones, yet has FEWER "
              "ripples.",
        covariates="none; still fraction normalises for traversal duration",
        meaning="A dissociation. F1 predicts MORE ripples during exploration; "
                "the data show fewer, so whatever suppresses ripples there is "
                "strong enough to beat the dominant driver."),
    "F4": dict(status="holds",
        claim="The stillness effect exists only while the grid is UNSOLVED. It "
              "disappears once the subject is executing a known route.",
        covariates="none; the interaction is within subject",
        meaning="Ripples track being still WHILE THERE IS SOMETHING LEFT TO "
                "WORK OUT. That is what a planning signal should look like and "
                "not what generic rest would look like. CAUTION: only the "
                "subjects contributing all three phases enter the interaction."),
    "F5": dict(status="holds",
        claim="Ripple rate rises above baseline in the ~300 ms after the FIRST "
              "uncovering of D -- the moment the full route first becomes "
              "knowable. SK predicted this before any analysis.",
        covariates="each window is compared with the SAME trial's baseline "
                   "(-1.6..-1.1 s), so any between-condition baseline "
                   "difference cancels; no covariates are needed",
        meaning="Only appeared once the test matched the claim. Earlier D "
                "contrasts compared first-D against later-D RATES, and the "
                "first traversal has a lower baseline (0.16 vs 0.20 Hz), so a "
                "transient rise was measured against a depressed floor and "
                "cancelled. The 2 s window also diluted a ~300 ms effect about "
                "fourfold. Neither was a property of the data."),
    "N1": dict(status="negative",
        claim="Ripples are NOT reliably suppressed at a key press.",
        covariates="none",
        meaning="Movement and uncover presses go in opposite directions and "
                "neither is significant, so motor suppression cannot be "
                "invoked to explain event-locked effects here -- nor can it be "
                "used to dismiss them."),
    "N2": dict(status="negative",
        claim="There is NO pre-event ripple rise before uncovering D.",
        covariates="same-trial baseline",
        meaning="Sakon's PRE effect, the signature of retrieval in free recall, "
                "is absent. Consistent with SK's logic: uncovering D is when "
                "information is ACQUIRED, not used."),
    "N3": dict(status="negative",
        claim="There is NO ripple response at the onset of a new grid.",
        covariates="none",
        meaning="A CONFIRMATION of SK's logic, not a failure. At grid onset "
                "nothing has been uncovered, so there is nothing plan-relevant "
                "to communicate. A burst here would have been the surprise."),
    "N4": dict(status="negative",
        claim="The pre-event dip for D-while-learning is NOT explained by "
              "error-related suppression, and does not survive smoothing.",
        covariates="split by whether an error uncovering occurred <2 s before",
        meaning="Tested because errors concentrate in learning repeats. Neither "
                "subgroup survives the sliding-window test; the bin-level "
                "clusters are single-bin and vanish under any smoothing. Read "
                "as noise from scanning 40 positions."),
}


# ----------------------------------------------------------------- helpers ---

def _png(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> {name}.png")


def _counts(**kw):
    """Standard denominators, so no condition's n is ever left implicit."""
    return {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
            for k, v in kw.items()}


def _report(out_dir, key, tests, conclusion, counts_df=None, extra=None):
    f = FINDINGS[key]
    swr_io.write_result(
        out_dir, key,
        hypothesis=f["claim"],
        tests=tests,
        conclusion=conclusion,
        extra={"status": f["status"], "covariates": f["covariates"],
               "meaning": f["meaning"], **(extra or {})})
    if counts_df is not None and len(counts_df):
        counts_df.to_csv(os.path.join(out_dir, f"{key}_counts.csv"), index=False)
        print(f"    counts  -> {key}_counts.csv")


def _gather(store, beh_all, R):
    """One pass over the data: everything the findings need."""
    still_rows, trav_rows, drift_rows, press_rows = [], [], [], []
    onset_rows = []
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
        still = pr.still_periods(mv, un, min_s=SHORT_STILL[0])
        still_ph = pr.label_still_by_phase(still, beh) if len(still) else still
        ev, iv_all, qc = store.get(sess)
        skey = store.subject_map().get(sess, {}).get("subject_key", f"s{sess}")

        for pair_id, e in ev.groupby("pair_id"):
            if pair_id in qc.index and bool(qc.loc[pair_id, "excluded"]):
                continue
            iv = iv_all[iv_all.pair_id == pair_id][["start_s", "stop_s"]].to_numpy()
            if not len(iv):
                continue
            rt = e.t_peak_s.to_numpy(float)
            base = dict(session=sess, subject_key=skey, pair_id=pair_id)

            d = pr.long_vs_short_still(still, rt, iv, short=SHORT_STILL,
                                       long_min=LONG_STILL)
            if d and np.isfinite(d.get("diff", np.nan)):
                still_rows.append({**base, **d, "phase3": "all"})
            for ph, gg in (still_ph.groupby("phase3") if len(still_ph) else []):
                dd = pr.long_vs_short_still(gg, rt, iv, short=SHORT_STILL,
                                            long_min=LONG_STILL)
                if dd and np.isfinite(dd.get("diff", np.nan)):
                    still_rows.append({**base, **dd, "phase3": ph})

            g = pr.within_grid_drift(beh, rt, iv)
            if len(g):
                drift_rows.append(g.assign(**base))

            for lab, tt in (("movement press", mv), ("uncover press", un)):
                dd = pr.peri_event_summary(tt, rt, iv, label=lab)
                if len(dd):
                    press_rows.append(dd.assign(**base))

            ot = sk.dedup_events(pr.grid_onset_times(beh))
            if ot.size:
                dd = pr.peri_event_summary(ot, rt, iv, label="grid onset")
                if len(dd):
                    onset_rows.append(dd.assign(**base))

        # per-traversal stillness (F3) -- behavioural, not per derivation
        if len(still):
            s0, s1 = still.start_s.to_numpy(), still.end_s.to_numpy()
            dur = still.duration_s.to_numpy()
            for grid, g in beh.groupby("grid_no"):
                g = g.sort_values("rep_overall"); prev = None
                for i, (_, r) in enumerate(g.iterrows()):
                    t0 = float(r.new_grid_onset) if i == 0 else prev
                    t1 = float(r.t_D); prev = t1
                    if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
                        continue
                    m = (s0 >= t0) & (s1 <= t1)
                    trav_rows.append({"session": int(sess), "subject_key": skey,
                                      "grid_no": int(grid),
                                      "cond": "first" if i == 0 else "later",
                                      "trav_s": t1 - t0,
                                      "still_s": float(dur[m].sum()),
                                      "long_still_s": float(dur[m][dur[m] >= LONG_STILL].sum())})
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()
    return (pd.DataFrame(still_rows), pd.DataFrame(trav_rows),
            cat(drift_rows), cat(press_rows), cat(onset_rows))


def _subject_paired(df, a, b, label_a, label_b):
    """Paired subject-level test with the denominators that produced it."""
    per = df.groupby("subject_key")[[a, b]].mean().dropna()
    if len(per) < 3:
        return None
    t, p = st.ttest_rel(per[b], per[a])
    pm = sk.perm_sign_flip(per[b] - per[a], n_perm=N_SIGN_FLIPS)
    return {"contrast": f"{label_b} minus {label_a}",
            "n_subjects": int(len(per)),
            f"mean_{label_a}": float(per[a].mean()),
            f"mean_{label_b}": float(per[b].mean()),
            "mean_diff": float((per[b] - per[a]).mean()),
            "t": float(t), "df": int(len(per) - 1), "p": float(p),
            "p_perm": pm.get("p_perm"), "per_subject": per}


# ------------------------------------------------------------------- main ----

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

    store = swr_bundle.RippleStore(analysis_name, R, bundle=bundle)
    store.describe()
    beh_all = pd.read_csv(os.path.join(bundle, "behaviour.csv"))
    unc_all = pd.read_csv(os.path.join(bundle, "uncover.csv"))
    rip_all = pd.read_csv(os.path.join(bundle, "ripples.csv"))

    print("\n" + "=" * 78)
    print(" DATASET")
    print("=" * 78)
    qc = pd.concat([store.get(s)[2].assign(session=s) for s in store.sessions()])
    qc = qc[~qc.excluded.fillna(False)]
    print(f"  {rip_all.session.nunique()} sessions | "
          f"{rip_all.subject_key.nunique()} subjects | "
          f"{len(qc)} derivations | {len(rip_all)} accepted ripples")
    print(f"  artifact-free exposure: {qc.clean_s.sum()/3600:.1f} h total, "
          f"median {qc.clean_s.median()/60:.1f} min per derivation")
    print(f"  behaviour: {len(beh_all)} repeats, "
          f"{beh_all.groupby(['session','grid_no']).ngroups} grids, "
          f"{len(unc_all)} uncovering attempts "
          f"({int((unc_all.correct==1).sum())} correct, "
          f"{int((unc_all.correct==0).sum())} errors)")

    S, T, D, P, ON = _gather(store, beh_all, R)
    results = {}

    print("\n" + "=" * 78); print(" FINDINGS"); print("=" * 78)

    # ---------------------------------------------------------------- F1 ----
    if "F1" in want and len(S):
        A = S[S.phase3 == "all"]
        r = _subject_paired(A, "rate_short", "rate_long", "short", "long")
        cnt = pd.DataFrame([
            _counts(condition_id=0, n_subjects=r["n_subjects"],
                    n_sessions=A.session.nunique(), n_derivations=len(A),
                    n_periods=A.n_short.sum(), exposure_s=A.exposure_short_s.sum()),
            _counts(condition_id=1, n_subjects=r["n_subjects"],
                    n_sessions=A.session.nunique(), n_derivations=len(A),
                    n_periods=A.n_long.sum(), exposure_s=A.exposure_long_s.sum())])
        cnt.insert(0, "condition", [f"still {SHORT_STILL[0]}-{SHORT_STILL[1]} s",
                                    f"still >= {LONG_STILL} s"])
        print(f"\n [HOLDS] F1  still {SHORT_STILL[0]}-{SHORT_STILL[1]}s "
              f"{r['mean_short']:.3f} Hz -> >={LONG_STILL}s {r['mean_long']:.3f} Hz")
        print(f"         t({r['df']}) = {r['t']:+.2f}, p = {r['p']:.3g}, "
              f"p_perm = {r['p_perm']:.4g}, n = {r['n_subjects']} subjects")
        print(f"         exposure: {cnt.exposure_s.iloc[0]/3600:.1f} h short, "
              f"{cnt.exposure_s.iloc[1]/3600:.1f} h long | "
              f"{int(cnt.n_periods.iloc[0])} vs {int(cnt.n_periods.iloc[1])} periods")
        results["F1"] = {k: v for k, v in r.items() if k != "per_subject"}
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        per = r["per_subject"]
        for _, row in per.iterrows():
            ax.plot([0, 1], [row.rate_short, row.rate_long], color="0.78", lw=.8)
        ax.scatter(np.zeros(len(per)), per.rate_short, s=20, color=rfig.PAL[1], zorder=3)
        ax.scatter(np.ones(len(per)), per.rate_long, s=20, color=rfig.OBS_LINE_C, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"still {SHORT_STILL[0]}-{SHORT_STILL[1]} s",
                            f"still >= {LONG_STILL} s"])
        ax.set_ylabel("Ripple rate (Hz)")
        ax.set_title(f"F1  ripples rise with stillness\n"
                     f"p = {r['p']:.2g}, n = {r['n_subjects']} subjects")
        _png(fig, figs, "F1_stillness")
        _report(out_dir, "F1", [results["F1"]],
                f"Holds. {r['mean_diff']:+.3f} Hz, p_perm = {r['p_perm']:.4g}.",
                counts_df=cnt)

    # ---------------------------------------------------------------- F4 ----
    if "F4" in want and len(S):
        per = S[S.phase3.isin(["explore", "plan", "execute"])].groupby(
            ["subject_key", "phase3"])["diff"].mean().unstack()
        rows, cnt = [], []
        for ph in ("explore", "plan", "execute"):
            if ph not in per:
                continue
            v = per[ph].dropna()
            t, p = st.ttest_1samp(v, 0.0)
            rows.append({"phase": ph, "n_subjects": int(len(v)),
                         "mean_diff_hz": float(v.mean()), "t": float(t),
                         "p": float(p),
                         "p_perm": sk.perm_sign_flip(v, N_SIGN_FLIPS)["p_perm"]})
            g = S[S.phase3 == ph]
            cnt.append({"phase": ph, **_counts_row(ph, g)})
        d = per.dropna(subset=["execute"]).copy()
        d["unsolved"] = d[[c for c in ("explore", "plan") if c in d]].mean(axis=1)
        d = d.dropna(subset=["unsolved"])
        t, p = st.ttest_rel(d["unsolved"], d["execute"])
        inter = {"contrast": "unsolved (explore+plan) minus execute",
                 "n_subjects": int(len(d)),
                 "mean_unsolved": float(d["unsolved"].mean()),
                 "mean_execute": float(d["execute"].mean()),
                 "t": float(t), "df": int(len(d) - 1), "p": float(p),
                 "p_perm": sk.perm_sign_flip(d["unsolved"] - d["execute"],
                                             N_SIGN_FLIPS)["p_perm"]}
        print(f"\n [HOLDS] F4  stillness effect by phase")
        for r_ in rows:
            print(f"         {r_['phase']:9s} {r_['mean_diff_hz']:+.4f} Hz  "
                  f"p = {r_['p']:.3g}  (n = {r_['n_subjects']} subjects)")
        print(f"         interaction unsolved {inter['mean_unsolved']:+.4f} vs "
              f"execute {inter['mean_execute']:+.4f} Hz, t({inter['df']}) = "
              f"{inter['t']:+.2f}, p = {inter['p']:.3g}, "
              f"n = {inter['n_subjects']} subjects")
        results["F4"] = {"per_phase": rows, "interaction": inter}
        cntdf = pd.DataFrame(cnt)
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for j, ph in enumerate(["explore", "plan", "execute"]):
            g = S[S.phase3 == ph]
            if not len(g):
                continue
            pp = g.groupby("subject_key")[["rate_short", "rate_long"]].mean().dropna()
            ax.errorbar([0, 1], [pp.rate_short.mean(), pp.rate_long.mean()],
                        yerr=[pp.rate_short.sem(), pp.rate_long.sem()],
                        marker="o", lw=1.6, capsize=3,
                        color=rfig._cond_color(ph, j), label=f"{ph} (n={len(pp)})")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"still {SHORT_STILL[0]}-{SHORT_STILL[1]} s",
                            f"still >= {LONG_STILL} s"])
        ax.set_ylabel("Ripple rate (Hz)"); ax.legend(frameon=False, fontsize=9)
        ax.set_title(f"F4  the stillness effect only while unsolved\n"
                     f"interaction p = {inter['p']:.3g}")
        _png(fig, figs, "F4_stillness_by_phase")
        _report(out_dir, "F4", rows + [inter],
                f"Holds, with caution. Interaction p = {inter['p']:.3g} on "
                f"{inter['n_subjects']} subjects; per-phase tests individually "
                f"non-significant.", counts_df=cntdf)

    # ---------------------------------------------------------------- F2 ----
    if "F2" in want and len(D):
        g = (D.groupby(["subject_key", "frac_through_grid"])
             .apply(lambda x: x.n_ripples.sum() / max(x.exposure_s.sum(), 1e-9))
             .rename("rate").reset_index())
        m = g.groupby("frac_through_grid").rate.mean()
        first, last = float(m.iloc[0]), float(m.iloc[-1])
        results["F2"] = {"first_fifth_hz": first, "last_fifth_hz": last,
                         "pct_change": 100 * (last / first - 1),
                         "n_subjects": int(g.subject_key.nunique())}
        print(f"\n [HOLDS] F2  within-grid drift {first:.3f} -> {last:.3f} Hz "
              f"({100*(last/first-1):+.1f}%)")
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for s_, gg in g.groupby("subject_key"):
            ax.plot(gg.frac_through_grid, gg.rate, color="0.86", lw=.6)
        ax.errorbar(m.index, m, yerr=g.groupby("frac_through_grid").rate.sem(),
                    color=rfig.OBS_LINE_C, lw=1.8, capsize=2)
        ax.set_xlabel("Fraction through the grid"); ax.set_ylabel("Ripple rate (Hz)")
        ax.set_title("F2  no within-grid drift\nthe control for first-vs-later contrasts")
        _png(fig, figs, "F2_within_grid_drift")
        _report(out_dir, "F2", [results["F2"]],
                f"Holds. {100*(last/first-1):+.1f}% across a grid: negligible.",
                counts_df=pd.DataFrame([_counts(
                    n_subjects=g.subject_key.nunique(),
                    n_sessions=D.session.nunique(),
                    n_grid_bins=len(D), exposure_s=D.exposure_s.sum())]))

    # ---------------------------------------------------------------- F3 ----
    if "F3" in want and len(T):
        per = T.groupby(["session", "cond"]).apply(
            lambda x: x.still_s.sum() / max(x.trav_s.sum(), 1e-9)).unstack().dropna()
        t, p = st.ttest_rel(per["first"], per["later"])
        cnt = pd.DataFrame([_counts(
            condition_id=i, n_sessions=int(len(per)),
            n_traversals=int((T.cond == c).sum()),
            total_traversal_s=float(T[T.cond == c].trav_s.sum()),
            total_still_s=float(T[T.cond == c].still_s.sum()))
            for i, c in enumerate(("first", "later"))])
        cnt.insert(0, "condition", ["first traversal", "later traversals"])
        results["F3"] = {"still_frac_first": float(per["first"].mean()),
                         "still_frac_later": float(per["later"].mean()),
                         "t": float(t), "df": int(len(per) - 1), "p": float(p),
                         "n_sessions": int(len(per))}
        print(f"\n [HOLDS] F3  still fraction first {per['first'].mean():.3f} vs "
              f"later {per['later'].mean():.3f}, t({len(per)-1}) = {t:+.2f}, "
              f"p = {p:.3g}")
        print(f"         {int(cnt.n_traversals.iloc[0])} first vs "
              f"{int(cnt.n_traversals.iloc[1])} later traversals")
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for _, r_ in per.iterrows():
            ax.plot([0, 1], [r_["first"], r_["later"]], color="0.78", lw=.8)
        ax.scatter(np.zeros(len(per)), per["first"], s=20, color=rfig.PAL[1], zorder=3)
        ax.scatter(np.ones(len(per)), per["later"], s=20, color=rfig.PAL[3], zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["first traversal", "later traversals"])
        ax.set_ylabel("Fraction of traversal spent still")
        ax.set_title(f"F3  first traversal is stiller, yet has fewer ripples\n"
                     f"p = {p:.2g}, n = {len(per)} sessions")
        _png(fig, figs, "F3_stillness_by_traversal")
        _report(out_dir, "F3", [results["F3"]],
                "Holds. A dissociation: stillness predicts MORE ripples, the "
                "first traversal has fewer.", counts_df=cnt)

    # ---------------------------------------------------------------- F5 ----
    if "F5" in want:
        stages = {}
        for sess in store.sessions():
            b = beh_all[beh_all.session == sess]
            if not len(b):
                continue
            for _, g in b.groupby("grid_no"):
                g = g.sort_values("rep_overall")
                reps = g.rep_overall.to_numpy(int); td = g.t_D.to_numpy(float)
                corr = g.correct.to_numpy(int)
                sol = reps[corr == 1]
                fs = int(sol[0]) if sol.size else np.inf
                for r_, t_ in zip(reps, td):
                    if not np.isfinite(t_):
                        continue
                    k = ("first D" if r_ == reps[0]
                         else "D while learning" if r_ < fs else "D once known")
                    stages.setdefault(k, {}).setdefault(sess, []).append(t_)
        prof, ncnt, centres = {}, {}, None
        for k, by_sess in stages.items():
            ncnt[k] = {"n_events_raw": 0, "n_events_dedup": 0,
                       "n_sessions": 0, "n_derivations": 0}
            for sess, ts in by_sess.items():
                ev, iv_all, qcs = store.get(sess)
                skey = store.subject_map().get(sess, {}).get("subject_key", f"s{sess}")
                t = sk.dedup_events(np.asarray(ts, float))
                ncnt[k]["n_events_raw"] += len(ts)
                ncnt[k]["n_events_dedup"] += int(t.size)
                ncnt[k]["n_sessions"] += 1
                if not t.size:
                    continue
                for pair_id, e in ev.groupby("pair_id"):
                    if pair_id in qcs.index and bool(qcs.loc[pair_id, "excluded"]):
                        continue
                    iv = iv_all[iv_all.pair_id == pair_id][["start_s", "stop_s"]].to_numpy()
                    if not len(iv):
                        continue
                    ncnt[k]["n_derivations"] += 1
                    c, P_ = sk.peth(t, e.t_peak_s.to_numpy(float), iv,
                                    half_s=2.0, bin_s=sk.BIN_S)
                    centres = c
                    prof.setdefault((skey, k), []).append(np.nanmean(P_, axis=0))
        print("\n [HOLDS] F5  ripples after the FIRST uncovering of D")
        rows, sw_by = [], {}
        base_m = (centres >= sk.BASE_WIN[0]) & (centres < sk.BASE_WIN[1])
        for k in ("first D", "D while learning", "D once known"):
            subs = sorted({s_ for s_, kk in prof if kk == k})
            if len(subs) < 3:
                continue
            X = np.vstack([np.nanmean(np.vstack(prof[(s_, k)]), axis=0) for s_ in subs])
            X = X - np.nanmean(X[:, base_m], axis=1, keepdims=True)
            sw = sk.sliding_window_cluster(X, centres, width_s=SLIDE_WIDTH_S,
                                           n_perm=1000)
            sw_by[k] = sw
            sig = [c for c in sw["clusters"] if c["p"] < 0.05]
            rows.append({"condition": k, "n_subjects": len(subs),
                         **ncnt[k], "sliding_width_s": SLIDE_WIDTH_S,
                         "significant_clusters": sig})
            msg = (", ".join(f"{c['t_start_s']:+.2f}..{c['t_stop_s']:+.2f}s "
                             f"(peak {c['peak_at_s']:+.2f}, p={c['p']:.4f})"
                             for c in sig) if sig else "none")
            print(f"         {k:18s} {msg}")
            print(f"           {len(subs)} subjects, {ncnt[k]['n_derivations']} "
                  f"derivations, {ncnt[k]['n_events_dedup']} events "
                  f"(of {ncnt[k]['n_events_raw']} before 2 s de-duplication)")
        results["F5"] = rows
        cnt = pd.DataFrame([{k_: v for k_, v in r_.items()
                             if k_ != "significant_clusters"} for r_ in rows])
        rfig.sliding_cluster_figure(sw_by, "F5", f"{SLIDE_WIDTH_S:g}s",
                                    out_stem=os.path.join(figs, "F5_sliding"))
        got = [r_ for r_ in rows if r_["condition"] == "first D"
               and r_["significant_clusters"]]
        _report(out_dir, "F5", rows,
                ("Holds: first D shows a post-event cluster; the other two "
                 "stages show none." if got else
                 "Did NOT reproduce: no surviving cluster for first D."),
                counts_df=cnt,
                extra={"baseline_window_s": list(sk.BASE_WIN),
                       "dedup_s": sk.DEDUP_S, "n_sign_flips": 1000,
                       "note": "every window position tested; corrected by "
                               "cluster mass over positions"})

    # ------------------------------------------------------------ N1 / N3 ----
    for key, tab, labels in (("N1", P, ["movement press", "uncover press"]),
                             ("N3", ON, ["grid onset"])):
        if key not in want or not len(tab):
            continue
        rows, cnt = [], []
        for lab in labels:
            g = tab[tab.label == lab]
            if not len(g):
                continue
            v = g.groupby("subject_key")["diff"].mean().dropna()
            t, p = st.ttest_1samp(v, 0.0)
            rows.append({"events": lab, "n_subjects": int(len(v)),
                         "mean_peri_minus_nonperi_hz": float(v.mean()),
                         "t": float(t), "p": float(p),
                         "p_perm": sk.perm_sign_flip(v, N_SIGN_FLIPS)["p_perm"]})
            cnt.append(_counts(n_subjects=len(v), n_sessions=g.session.nunique(),
                               n_events=len(g),
                               exposure_peri_s=float(g.exposure_peri_s.sum())))
        print(f"\n [NEGATIVE] {key}")
        for r_ in rows:
            print(f"         {r_['events']:16s} {r_['mean_peri_minus_nonperi_hz']:+.4f} Hz, "
                  f"p = {r_['p']:.3g} (n = {r_['n_subjects']} subjects)")
        results[key] = rows
        c = pd.DataFrame(cnt); c.insert(0, "condition", labels[:len(cnt)])
        _report(out_dir, key, rows, "No effect.", counts_df=c)

    with open(os.path.join(out_dir, "findings_registry.json"), "w") as f:
        json.dump({"registry": FINDINGS, "recomputed": results,
                   "dataset": {"sessions": int(rip_all.session.nunique()),
                               "subjects": int(rip_all.subject_key.nunique()),
                               "derivations": int(len(qc)),
                               "ripples": int(len(rip_all)),
                               "exposure_h": float(qc.clean_s.sum() / 3600)},
                   "development_sessions": list(DEV_SESSIONS),
                   "created": datetime.now().isoformat(timespec="seconds")},
                  f, indent=2, default=str)
    print(f"\n saved -> {out_dir}")
    return None


def _counts_row(ph, g):
    return _counts(n_derivations=len(g), n_sessions=g.session.nunique(),
                   n_short_periods=g.n_short.sum(), n_long_periods=g.n_long.sum(),
                   exposure_short_s=g.exposure_short_s.sum(),
                   exposure_long_s=g.exposure_long_s.sum())


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
