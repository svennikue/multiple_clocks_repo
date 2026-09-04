#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event-locked ripple analysis in the Sakon & Kahana / He et al. design.

Why this exists: every window-averaged contrast in `swr_hypotheses.py` compares
two conditions that differ in BASELINE ripple rate, so the contrast measures the
baseline difference and not the event. Both reference papers avoid this the same
way -- compare a window around the event with the same window shifted earlier in
the SAME trial (Sakon Eq. 2), so the state difference cancels.

    python scripts/swr_event_locked.py run --bundle=<bundle dir>
    python scripts/swr_event_locked.py run --which=H1 --n_perm=1000

Design parameters are the papers', not tuned here:
    100 ms bins; PRE -600..-100 ms; BASELINE -1600..-1100 ms; POST +200..+700 ms
    events within 2 s of another event dropped (no ripple counted twice)
    Eq. 1  between conditions, LME with subject and session random effects
    Eq. 2  within trial vs its own baseline, per subject, then a one-sample t
    cluster-based permutation over time bins, sign-flipped across subjects

The one deliberate deviation: rate is per ARTIFACT-FREE second. Sakon has no
artifact mask; we do, and a window half-removed by rejection offers half the
opportunity to see a ripple.

@author: Svenja Kuchenhoff
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_sakon as sk
import mc.analyse.swr_behaviour as swb
import mc.plotting.ripple_figures as rfig

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

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


def _alignments(which, beh, unc, sess):
    """(label -> event times) for one session."""
    b = beh[beh.session == sess]
    u = unc[unc.session == sess]
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
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "swr_hyp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "swr_hypotheses.py"))
    hyp = iu.module_from_spec(spec)
    spec.loader.exec_module(hyp)

    R = swr_io.get_data_root()
    if out_dir is None:
        out_dir = os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                               "event_locked")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, f"swr_event_locked_{which}")

    store = hyp.RippleStore(analysis_name, R, bundle=bundle)
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
                r_pre, n_pre, _ = sk.window_rate(t, rt, iv, sk.PRE_WIN)
                r_base, _, _ = sk.window_rate(t, rt, iv, sk.BASE_WIN)
                r_post, _, _ = sk.window_rate(t, rt, iv, sk.POST_WIN)
                r_peri, _, _ = sk.window_rate(t, rt, iv, sk.PERI_WIN)
                r_nonperi, _, _ = sk.multi_window_rate(t, rt, iv, sk.NONPERI_WINS)
                rows.append(pd.DataFrame({
                    "session": sess, "subject_key": skey, "pair_id": pair_id,
                    "condition": cond, "event_t": t,
                    "rate_pre": r_pre, "rate_base": r_base, "rate_post": r_post,
                    "rate_peri": r_peri, "rate_nonperi": r_nonperi,
                    "n_pre": n_pre}))
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

    # ---- Eq. 2: within trial, event window vs its own baseline, per condition
    print(f"\n  Eq. 2  (PRE vs its own baseline {sk.BASE_WIN[0]}..{sk.BASE_WIN[1]} s, "
          f"per subject then one-sample t)")
    results["eq2"] = {}
    for cond in conds:
        sub = tab[tab.condition == cond]
        e2 = sk.fit_eq2(sub, rate_event="rate_pre", rate_base="rate_base")
        results["eq2"][cond] = e2
        if e2 and "t" in e2:
            print(f"    {cond:14s} mean t = {e2['mean_t']:+.3f}, "
                  f"t({e2['df']}) = {e2['t']:+.2f}, p = {e2['p']:.4g}, "
                  f"n = {e2['n_subjects']} subjects")
        else:
            print(f"    {cond:14s} {e2.get('error') if e2 else 'failed'}")

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
        t_obs, cl, pv = sk.cluster_perm_time(X, n_perm=n_perm)
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
            rfig.pvth_figure(centres, prof_tab, results, which,
                             out_stem=os.path.join(out_dir, "figures",
                                                   f"pvth_{which}"))
        except Exception as e:
            print(f"  figure failed: {type(e).__name__}: {e}")
        print(f"\n saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
