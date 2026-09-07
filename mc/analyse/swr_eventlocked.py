#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event-locked ripple analysis in the Sakon & Kahana (2022) / He et al. (2026)
design. Implementation only; the CLI is `scripts/swr_explore.py event_locked`.

Both papers solve the problem this project ran into -- a rate contrast between
two conditions is contaminated by any baseline difference between them -- the
same way: compare a window around the event with the SAME window shifted earlier
in the same trial, so the state difference cancels.

@author: Svenja Kuchenhoff
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

import mc.analyse.swr_io as swr_io
import mc.analyse.swr_bundle as swr_bundle
import mc.analyse.swr_sakon as sk
import mc.analyse.swr_behaviour as swb
import mc.plotting.ripple_figures as rfig

ANALYSIS_NAME = "swr_v1"
HALF_S = 2.0
DEV_SESSIONS = (2, 3, 6, 9, 12, 13, 14, 38)     # the 8 used to develop the pipeline


def _settings(which, n_perm, bundle):
    return {"bin_s": sk.BIN_S, "pre_win_s": list(sk.PRE_WIN),
            "base_win_s": list(sk.BASE_WIN), "post_win_s": list(sk.POST_WIN),
            "peri_win_s": list(sk.PERI_WIN),
            "dedup_s": sk.DEDUP_S, "min_clean_frac": sk.MIN_CLEAN_FRAC,
            "half_s": HALF_S, "n_perm": n_perm, "which": which,
            "bundle": bundle, "held_out_split": "8 development sessions vs 38 new",
            "design_source": "Sakon & Kahana 2022 PNAS; He et al. 2026 Nat Neurosci",
            "created": datetime.now().isoformat(timespec="seconds")}


# Windows tested against the SAME baseline (-1.6..-1.1 s), so "before vs during
# vs after" is three comparisons of identical width against one reference.
TEST_WINDOWS = {
    "pre  (-0.6..-0.1)": (-0.6, -0.1),
    "post (0..0.5)": (0.0, 0.5),
    "post (0.5..1.0)": (0.5, 1.0),
}


def _alignments(which, beh, unc, sess):
    """(label -> event times) for one session.

    All alignments are to the moment a reward is UNCOVERED (the successful key
    press, t_A..t_D in all_trial_times), not to arrival at the location.
    """
    b = beh[beh.session == sess]
    u = unc[unc.session == sess]
    if which == "rewards":
        # each reward, collapsed over all traversals
        return {st: b[f"t_{st}"].to_numpy(float)[
            np.isfinite(b[f"t_{st}"].to_numpy(float))] for st in "ABCD"}
    if which == "rewards_first":
        out = {}
        for st in "ABCD":
            ts = []
            for _, g in b.groupby("grid_no"):
                g = g.sort_values("rep_overall")
                v = float(g.iloc[0][f"t_{st}"])
                if np.isfinite(v):
                    ts.append(v)
            out[st] = np.asarray(ts, float)
        return out
    if which == "reward_first_vs_later":
        first, later = [], []
        for _, g in b.groupby("grid_no"):
            g = g.sort_values("rep_overall")
            for st in "ABCD":
                v = g[f"t_{st}"].to_numpy(float)
                v = v[np.isfinite(v)]
                if v.size:
                    first.append(v[0]); later += list(v[1:])
        return {"first traversal": np.asarray(first, float),
                "later traversals": np.asarray(later, float)}
    if which in ("feedback_x_stage", "reward_x_feedback",
                 "reward_x_feedback_x_stage"):
        # ONE phase definition, used everywhere (SK, 2026-09-04):
        #   first uncovers   the first traversal of the grid
        #   while learning   up to AND INCLUDING the first fully correct repeat
        #   once known       every repeat after that
        # Note the boundary: the first error-free repeat counts as learning,
        # not as execution, because that is the repeat on which the route is
        # first demonstrated rather than merely relied on.
        stage_of, seeking = {}, {}
        for _, g in b.groupby("grid_no"):
            g = g.sort_values("rep_overall")
            reps = g.rep_overall.to_numpy(int)
            corr = g.correct.to_numpy(int)
            sol = reps[corr == 1]
            fs = int(sol[0]) if sol.size else np.inf
            for r_ in reps:
                stage_of[(int(g.grid_no.iloc[0]), int(r_))] = (
                    "first uncovers" if r_ == reps[0]
                    else "while learning" if r_ <= fs else "once known")
            # which reward was being SOUGHT at each moment of each repeat:
            # after k rewards have been collected, the subject is looking for
            # the (k+1)th. Errors inherit the reward they were searching for.
            for _, r_row in g.iterrows():
                ts = [float(r_row[f"t_{x}"]) for x in "ABCD"]
                seeking[(int(g.grid_no.iloc[0]), int(r_row.rep_overall))] = ts

        rows = []
        for e in u.itertuples():
            key = (int(e.grid_no), int(e.rep_overall))
            stg = stage_of.get(key)
            ts = seeking.get(key)
            if stg is None or ts is None:
                continue
            n_before = int(np.sum([np.isfinite(t_) and t_ < e.t_s for t_ in ts]))
            tgt = "ABCD"[min(n_before, 3)]
            if int(e.correct) == 1 and isinstance(e.state, str):
                tgt = e.state          # a correct uncovering names its own reward
            rows.append({"t_s": float(e.t_s),
                         "valence": "correct" if int(e.correct) == 1 else "error",
                         "stage": stg, "reward": tgt})
        R_ = pd.DataFrame(rows)
        if not len(R_):
            return {}
        if which == "feedback_x_stage":
            return {f"{v}, {s_}": R_.query("valence == @v and stage == @s_")
                    .t_s.to_numpy(float)
                    for v in ("correct", "error")
                    for s_ in ("first uncovers", "while learning", "once known")}
        if which == "reward_x_feedback":
            return {f"{v} {rw}": R_.query("valence == @v and reward == @rw")
                    .t_s.to_numpy(float)
                    for rw in "ABCD" for v in ("correct", "error")}
        out = {}
        for rw in "ABCD":
            for v in ("correct", "error"):
                for s_ in ("first uncovers", "while learning", "once known"):
                    k = f"{v} {rw}, {s_}"
                    sel = R_.query("valence == @v and reward == @rw and stage == @s_")
                    if len(sel) >= 20:      # too few events for a stable estimate
                        out[k] = sel.t_s.to_numpy(float)
        return out
    if which == "feedback_x_phase":
        # The 2x2 behind the descriptive figure: what the subject was told
        # (correct / error) crossed with whether the route was still being
        # discovered. A correct uncovering during discovery IS the acquisition
        # of a new reward location, so this condition overlaps F5 by
        # construction -- it contains all four A-D uncoverings of the first
        # traversal, not only D.
        out = {}
        for lab, q in (("correct, discovery", "correct == 1 and is_discovery == 1"),
                       ("error, discovery", "correct == 0 and is_discovery == 1"),
                       ("correct, later", "correct == 1 and is_discovery == 0"),
                       ("error, later", "correct == 0 and is_discovery == 0")):
            out[lab] = u.query(q).t_s.to_numpy(float)
        return out
    if which == "feedback":
        return {"correct": u[u.correct == 1].t_s.to_numpy(float),
                "error": u[u.correct == 0].t_s.to_numpy(float)}
    if which == "D_learn_by_error":
        # SK's idea: the pre-event dip for "D while learning" may be the
        # error-related suppression (H4), since learning repeats are exactly the
        # ones containing errors. Split those D uncoverings by whether an ERROR
        # uncovering happened in the 2 s before.
        errs = u[u.correct == 0].t_s.to_numpy(float)
        after, clean = [], []
        for _, g in b.groupby("grid_no"):
            g = g.sort_values("rep_overall")
            reps = g.rep_overall.to_numpy(int)
            td = g.t_D.to_numpy(float)
            corr = g.correct.to_numpy(int)
            solved = reps[corr == 1]
            first_solved = int(solved[0]) if solved.size else np.inf
            for r, t in zip(reps, td):
                if not np.isfinite(t) or r == reps[0] or r >= first_solved:
                    continue
                near = bool(np.any((errs >= t - 2.0) & (errs < t)))
                (after if near else clean).append(t)
        return {"error <2 s before": np.asarray(after, float),
                "no recent error": np.asarray(clean, float)}
    if which.endswith("_stages") and which[0] in "ABCD":
        st = which[0]
        first, learn, known = [], [], []
        for _, g in b.groupby("grid_no"):
            g = g.sort_values("rep_overall")
            reps = g.rep_overall.to_numpy(int)
            td = g[f"t_{st}"].to_numpy(float)
            ok = np.isfinite(td)
            corr = g.correct.to_numpy(int)
            solved = reps[corr == 1]
            first_solved = int(solved[0]) if solved.size else np.inf
            for r, t in zip(reps[ok], td[ok]):
                if r == reps[0]:
                    first.append(t)
                elif r < first_solved:
                    learn.append(t)
                else:
                    known.append(t)
        return {f"first {st}": np.asarray(first, float),
                f"{st} while learning": np.asarray(learn, float),
                f"{st} once known": np.asarray(known, float)}
    if which == "H1":
        out = {}
        for lab, first in (("first_D", True), ("later_D", False)):
            ts = []
            for _, g in b.groupby("grid_no"):
                g = g.sort_values("rep_overall")
                sel = g.iloc[:1] if first else g.iloc[1:]
                ts += [float(v) for v in sel.t_D if np.isfinite(v)]
            out[lab] = np.asarray(ts, float)
        return out
    if which == "H4":
        return {"correct": u[u.correct == 1].t_s.to_numpy(float),
                "error": u[u.correct == 0].t_s.to_numpy(float)}
    if which == "H7":
        return {"informative": u[((u.correct == 1) & (u.is_discovery == 1)) |
                                 ((u.correct == 0) & (u.is_discovery == 0))].t_s.to_numpy(float),
                "uninformative": u[((u.correct == 0) & (u.is_discovery == 1)) |
                                   ((u.correct == 1) & (u.is_discovery == 0))].t_s.to_numpy(float)}
    raise ValueError(f"unknown which '{which}'")


def run(which="H1", bundle=None, analysis_name=ANALYSIS_NAME, n_perm=1000,
        out_dir=None, half_s=HALF_S, save_all=True):
    R = swr_io.get_data_root()
    if out_dir is None:
        out_dir = os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                               "event_locked")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, f"swr_event_locked_{which}")

    store = swr_bundle.RippleStore(analysis_name, R, bundle=bundle)
    store.describe()
    subj = store.subject_map()
    beh = pd.read_csv(os.path.join(bundle, "behaviour.csv"))
    unc = pd.read_csv(os.path.join(bundle, "uncover.csv"))

    print(f"\n  {which}: {sk.BIN_S*1000:.0f} ms bins | PRE {sk.PRE_WIN} | "
          f"BASE {sk.BASE_WIN} | dedup {sk.DEDUP_S} s")

    rows, peth_rows = [], []
    centres = None
    for sess in store.sessions():
        try:
            aligns = _alignments(which, beh, unc, sess)
        except Exception as e:
            print(f"  s{sess:02d}: {type(e).__name__}: {e}"); continue
        ev, iv_all, qc = store.get(sess)
        skey = subj.get(sess, {}).get("subject_key", f"s{sess}")
        for pair_id, e in ev.groupby("pair_id"):
            if pair_id in qc.index and bool(qc.loc[pair_id, "excluded"]):
                continue
            iv = iv_all[iv_all.pair_id == pair_id][["start_s", "stop_s"]].to_numpy()
            if not len(iv):
                continue
            rt = e.t_peak_s.to_numpy(float)
            for cond, t_raw in aligns.items():
                t = sk.dedup_events(t_raw)
                if not t.size:
                    continue
                r_base, _, _ = sk.window_rate(t, rt, iv, sk.BASE_WIN)
                r_peri, _, _ = sk.window_rate(t, rt, iv, sk.PERI_WIN)
                r_nonperi, _, _ = sk.multi_window_rate(t, rt, iv, sk.NONPERI_WINS)
                d = {"session": sess, "subject_key": skey, "pair_id": pair_id,
                     "condition": cond, "event_t": t, "rate_base": r_base,
                     "rate_peri": r_peri, "rate_nonperi": r_nonperi}
                for wname, w in TEST_WINDOWS.items():
                    d[f"rate::{wname}"], _, _ = sk.window_rate(t, rt, iv, w)
                d["rate_pre"] = d["rate::pre  (-0.6..-0.1)"]
                rows.append(pd.DataFrame(d))
                c, P = sk.peth(t, rt, iv, half_s=half_s, bin_s=sk.BIN_S)
                centres = c
                peth_rows.append((skey, cond, np.nanmean(P, axis=0), len(t)))
        print(f"  s{sess:02d}: {sum(len(v) for v in aligns.values())} events "
              f"({', '.join(f'{k} {len(sk.dedup_events(v))}' for k, v in aligns.items())} after dedup)")

    if not rows:
        print("\n  nothing computable"); return None
    tab = pd.concat(rows, ignore_index=True)
    tab["is_dev"] = tab.session.isin(DEV_SESSIONS)
    print(f"\n  {len(tab)} event-derivation rows, {tab.session.nunique()} sessions, "
          f"{tab.subject_key.nunique()} subjects")

    # ---- per-subject PETH, for the figure and the cluster test
    pk = {}
    for skey, cond, prof, n in peth_rows:
        pk.setdefault((skey, cond), []).append(prof)
    prof_tab = {k: np.nanmean(np.vstack(v), axis=0) for k, v in pk.items()}
    conds = sorted({c for _, c in prof_tab})

    print("\n" + "=" * 78)
    print(f" {which}  --  Sakon & Kahana design")
    print("=" * 78)

    results = {"which": which}

    # ---- Eq. 1: between conditions, in the PRE window
    e1 = sk.fit_eq1(tab.rename(columns={"rate_pre": "rate"}))
    if e1 and "beta" in e1:
        print(f"\n  Eq. 1  (PRE window, between conditions)")
        print(f"    beta = {e1['beta']:+.4f} +- {e1['se']:.4f}, z = {e1['z']:+.2f}, "
              f"p = {e1['p']:.4g}   [{e1['direction']}]")
        print(f"    {e1['n_obs']} observations, {e1['n_subjects']} subjects")
    results["eq1"] = e1

    # ---- Eq. 2: BEFORE vs DURING vs AFTER, each against the same baseline
    print(f"\n  Eq. 2  (each window vs the SAME baseline "
          f"{sk.BASE_WIN[0]}..{sk.BASE_WIN[1]} s, per subject then one-sample t)")
    print(f"    {'condition':22s} {'window':20s} {'mean t':>8s} {'t':>8s} "
          f"{'p':>10s} {'p_perm':>10s}   (10k sign-flips)")
    results["eq2"] = {}
    for cond in conds:
        sub = tab[tab.condition == cond]
        results["eq2"][cond] = {}
        for wname in TEST_WINDOWS:
            e2 = sk.fit_eq2(sub, rate_event=f"rate::{wname}",
                            rate_base="rate_base")
            results["eq2"][cond][wname] = e2
            if e2 and "t" in e2:
                pp = e2.get("p_perm", np.nan)
                star = " *" if (np.isfinite(pp) and pp < 0.05) else ""
                print(f"    {cond:22s} {wname:20s} {e2['mean_t']:+8.3f} "
                      f"{e2['t']:+8.2f} {e2['p']:10.4g} {pp:10.4g}{star}")
        # keep the flat PRE entry the figure expects
        results["eq2"][cond]["_pre"] = results["eq2"][cond].get(
            "pre  (-0.6..-0.1)")

    # ---- He et al.: peri vs non-peri (symmetric flanks)
    print(f"\n  He et al.  (peri {sk.PERI_WIN} vs non-peri flanks, per subject)")
    results["peri"] = {}
    for cond in conds:
        sub = tab[tab.condition == cond]
        e = sk.fit_eq2(sub, rate_event="rate_peri", rate_base="rate_nonperi")
        results["peri"][cond] = e
        if e and "t" in e:
            print(f"    {cond:14s} mean t = {e['mean_t']:+.3f}, "
                  f"t({e['df']}) = {e['t']:+.2f}, p = {e['p']:.4g}")

    # ---- sliding window: every position tested, cluster-corrected over positions
    print(f"\n  sliding window ({n_perm} sign-flips, cluster-corrected over "
          f"window positions -- no window chosen)")
    results["sliding"] = {}
    sliding_raw = {}
    base_m0 = (centres >= sk.BASE_WIN[0]) & (centres < sk.BASE_WIN[1])
    for width in (0.3, 0.5):
        for cond in conds:
            subs = sorted({s for s, c in prof_tab if c == cond})
            if len(subs) < 3:
                continue
            X = np.vstack([prof_tab[(s, cond)] for s in subs])
            X = X - np.nanmean(X[:, base_m0], axis=1, keepdims=True)
            sw = sk.sliding_window_cluster(X, centres, width_s=width,
                                           n_perm=n_perm)
            if sw is None:
                continue
            results["sliding"].setdefault(f"{width:g}s", {})[cond] = sw["clusters"]
            sliding_raw.setdefault(f"{width:g}s", {})[cond] = sw
            sig = [c for c in sw["clusters"] if c["p"] < 0.05]
            msg = (", ".join(f"{c['t_start_s']:+.2f}..{c['t_stop_s']:+.2f} s "
                             f"(peak {c['peak_at_s']:+.2f}, p={c['p']:.4f})"
                             for c in sig) if sig else "none")
            print(f"    width {width:g}s  {cond:22s} {msg}")

    # ---- cluster permutation over time, baseline-subtracted per subject
    print(f"\n  cluster permutation over time ({n_perm} sign-flips, subject-level)")
    results["clusters"] = {}
    base_m = (centres >= sk.BASE_WIN[0]) & (centres < sk.BASE_WIN[1])
    for cond in conds:
        subs = sorted({s for s, c in prof_tab if c == cond})
        if len(subs) < 3:
            continue
        X = np.vstack([prof_tab[(s, cond)] for s in subs])
        X = X - np.nanmean(X[:, base_m], axis=1, keepdims=True)
        t_obs, cl, pv, _null = sk.cluster_perm_time(X, n_perm=n_perm)
        results["clusters"][cond] = [
            {"t_start_s": float(centres[a]), "t_stop_s": float(centres[b - 1]),
             "p": p, "peak_t": float(np.nanmax(np.abs(t_obs[a:b])))}
            for (a, b), p in zip(cl, pv)]
        sig = [c for c in results["clusters"][cond] if c["p"] < 0.05]
        print(f"    {cond:14s} {len(cl)} clusters, {len(sig)} with p<0.05"
              + ("".join(f"\n        {c['t_start_s']:+.2f} to {c['t_stop_s']:+.2f} s, "
                         f"p = {c['p']:.4f}" for c in sig) if sig else ""))

    if save_all:
        tab.to_csv(os.path.join(out_dir, f"{which}_event_rates.csv"), index=False)
        prof = pd.DataFrame({f"{s}|{c}": v for (s, c), v in prof_tab.items()},
                            index=np.round(centres, 4))
        prof.to_csv(os.path.join(out_dir, f"{which}_peth_by_subject.csv"))
        with open(os.path.join(out_dir, f"{which}_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
        with open(os.path.join(out_dir, f"{which}_settings.json"), "w") as f:
            json.dump(_settings(which, n_perm, bundle), f, indent=2)
        try:
            for wkey, byc in sliding_raw.items():
                rfig.sliding_cluster_figure(
                    byc, which, wkey,
                    out_stem=os.path.join(out_dir, "figures",
                                          f"sliding_{which}_{wkey}"))
            rfig.pvth_figure(centres, prof_tab, results, which,
                             out_stem=os.path.join(out_dir, "figures",
                                                   f"pvth_{which}"),
                             windows=TEST_WINDOWS)
            sig = {c: [w for w, e in v.items()
                       if w != "_pre" and e and e.get("p", 1) < 0.05]
                   for c, v in results["eq2"].items()}
            swr_io.write_result(
                out_dir, f"event_locked_{which}",
                hypothesis=("Ripple rate around reward uncovering, tested "
                            "before / during / after the event against the same "
                            "pre-event baseline (Sakon & Kahana Eq. 2), plus a "
                            "cluster permutation over the whole time course."),
                tests=[{"eq1": results.get("eq1")},
                       {"eq2": {c: {w: {k: v for k, v in (e or {}).items()
                                        if k != "per_subject"}
                                    for w, e in d.items() if w != "_pre"}
                                for c, d in results["eq2"].items()}},
                       {"clusters": results.get("clusters")}],
                conclusion=("Windows significant vs baseline: "
                            + ("; ".join(f"{c}: {', '.join(w)}"
                                         for c, w in sig.items() if w)
                               or "none in any condition")),
                extra={"aligned_to": "reward uncovering (t_A..t_D), the "
                                     "successful key press",
                       "windows": {k: list(v) for k, v in TEST_WINDOWS.items()},
                       "baseline": list(sk.BASE_WIN)})
        except Exception as e:
            print(f"  figure failed: {type(e).__name__}: {e}")
        print(f"\n saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
