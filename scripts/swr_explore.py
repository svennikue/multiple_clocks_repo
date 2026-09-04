#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXPLORATION. Descriptive probes on what drives ripple rate in this task.

This is the scratch script: many quick checks, PNG only, no permutations, no
multiple-comparison correction, nothing here is an inference. Its job is to
find out which contrasts are worth running properly.

Findings that survive and are understood get moved to `swr_findings.py`, which
is the curated set.

    python scripts/swr_explore.py run --bundle=<bundle dir>
    python scripts/swr_explore.py run --probes="['movement','stillness']"

@author: Svenja Kuchenhoff
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

ALL_PROBES = ("movement", "stillness", "drift", "peth_presses",
              "still_long", "grid_drift",
              "still_by_phase", "prospective", "grid_onset")


def _store(bundle, analysis_name="swr_v1"):
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "swr_hyp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "swr_hypotheses.py"))
    hyp = iu.module_from_spec(spec)
    spec.loader.exec_module(hyp)
    return hyp.RippleStore(analysis_name, swr_io.get_data_root(), bundle=bundle)


def _subject_t(df, value="diff", subject_col="subject_key"):
    """Per-subject mean, then a one-sample t. The subject is the unit."""
    from scipy import stats as st
    per = df.groupby(subject_col)[value].mean().dropna()
    if len(per) < 3:
        return None
    t, p = st.ttest_1samp(per, 0.0)
    return {"n_subjects": int(len(per)), "mean": float(per.mean()),
            "t": float(t), "p": float(p), "per_subject": per}


def _png(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{name}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {name}.png")


def run(bundle=None, probes=None, out_dir=None, analysis_name="swr_v1",
        half_s=1.5):
    R = swr_io.get_data_root()
    if out_dir is None:
        out_dir = os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                               "explore")
    figs = os.path.join(out_dir, "figures")
    os.makedirs(figs, exist_ok=True)
    swr_io.start_log(out_dir, "swr_explore")
    probes = list(probes) if probes else list(ALL_PROBES)

    store = _store(bundle, analysis_name)
    store.describe()
    subj = store.subject_map()
    beh_all = pd.read_csv(os.path.join(bundle, "behaviour.csv"))
    print(f"\n  probes: {', '.join(probes)}   (PNG only, no permutations)")

    press_rows, still_rows, drift_rows = [], [], []
    longshort_rows, griddrift_rows = [], []
    phase_rows, prosp_rows = [], []
    onset_acc = {}; onset_rows = []
    peth_acc = {}
    centres = None

    for sess in store.sessions():
        beh = beh_all[beh_all.session == sess]
        if not len(beh):
            continue
        try:
            mv_t, un_t = pr.press_times(sess, beh, data_root=R)
        except Exception as e:
            print(f"  s{sess:02d}: presses unreadable ({type(e).__name__})"); continue
        if not mv_t.size:
            continue
        still = pr.still_periods(mv_t, un_t, min_s=0.5)
        ev, iv_all, qc = store.get(sess)
        skey = subj.get(sess, {}).get("subject_key", f"s{sess}")

        for pair_id, e in ev.groupby("pair_id"):
            if pair_id in qc.index and bool(qc.loc[pair_id, "excluded"]):
                continue
            iv = iv_all[iv_all.pair_id == pair_id][["start_s", "stop_s"]].to_numpy()
            if not len(iv):
                continue
            rt = e.t_peak_s.to_numpy(float)

            if "movement" in probes:
                for lab, tt in (("movement press", mv_t), ("uncover press", un_t)):
                    d = pr.peri_event_summary(tt, rt, iv, label=lab)
                    if len(d):
                        d["session"] = sess; d["subject_key"] = skey
                        d["pair_id"] = pair_id
                        press_rows.append(d)

            if "stillness" in probes and len(still):
                d = pr.rate_vs_still_duration(still, rt, iv)
                if len(d):
                    d["session"] = sess; d["subject_key"] = skey
                    still_rows.append(d)

            if "still_long" in probes and len(still):
                d = pr.long_vs_short_still(still, rt, iv)
                if d and np.isfinite(d.get("diff", np.nan)):
                    d.update(session=sess, subject_key=skey, pair_id=pair_id)
                    longshort_rows.append(d)

            if "grid_drift" in probes:
                d = pr.within_grid_drift(beh, rt, iv)
                if len(d):
                    d["session"] = sess; d["subject_key"] = skey
                    griddrift_rows.append(d)

            if "still_by_phase" in probes and len(still):
                sp = pr.label_still_by_phase(still, beh)
                for ph, gg in sp.groupby("phase3"):
                    d = pr.long_vs_short_still(gg, rt, iv)
                    if d and np.isfinite(d.get("diff", np.nan)):
                        d.update(session=sess, subject_key=skey,
                                 pair_id=pair_id, phase3=ph)
                        phase_rows.append(d)

            if "prospective" in probes and len(still):
                d = pr.fixed_window_by_pause_length(still, rt, iv)
                if len(d):
                    d["session"] = sess; d["subject_key"] = skey
                    prosp_rows.append(d)

            if "grid_onset" in probes:
                ot = pr.grid_onset_times(beh)
                ot = sk.dedup_events(ot)
                if ot.size:
                    d = pr.peri_event_summary(ot, rt, iv, label="grid onset")
                    if len(d):
                        d["session"] = sess; d["subject_key"] = skey
                        onset_rows.append(d)
                    c, P = sk.peth(ot, rt, iv, half_s=4.0, bin_s=sk.BIN_S)
                    onset_acc.setdefault(skey, []).append(
                        (c, np.nanmean(P, axis=0)))

            if "drift" in probes:
                d = pr.rate_vs_session_time(rt, iv)
                if len(d):
                    d["session"] = sess; d["subject_key"] = skey
                    drift_rows.append(d)

            if "peth_presses" in probes:
                for lab, tt in (("movement press", mv_t), ("uncover press", un_t)):
                    t = sk.dedup_events(tt)
                    if not t.size:
                        continue
                    c, P = sk.peth(t, rt, iv, half_s=half_s, bin_s=sk.BIN_S)
                    centres = c
                    peth_acc.setdefault((skey, lab), []).append(np.nanmean(P, axis=0))
        print(f"  s{sess:02d}: {mv_t.size} movement, {un_t.size} uncover presses, "
              f"{len(still)} still periods")

    out = {}
    print("\n" + "=" * 78)
    print(" EXPLORATION  (descriptive; no permutations, no correction)")
    print("=" * 78)

    # ---------------------------------------------------------------- presses
    if press_rows:
        P = pd.concat(press_rows, ignore_index=True)
        P.to_csv(os.path.join(out_dir, "press_locked.csv"), index=False)
        print("\n  [movement] ripple rate around a key press "
              "(peri +-250 ms vs symmetric flanks)")
        for lab, g in P.groupby("label"):
            r = _subject_t(g)
            if r:
                print(f"    {lab:16s} mean diff = {r['mean']:+.4f} Hz, "
                      f"t({r['n_subjects']-1}) = {r['t']:+.2f}, p = {r['p']:.4g}"
                      f"   ({'SUPPRESSED' if r['mean'] < 0 else 'elevated'} at the press)")
                out[f"press::{lab}"] = {k: v for k, v in r.items()
                                        if k != "per_subject"}
        fig, ax = plt.subplots(figsize=(7, 4.2))
        labs = sorted(P.label.unique())
        for j, lab in enumerate(labs):
            per = P[P.label == lab].groupby("subject_key")["diff"].mean().dropna()
            x = j + (np.random.default_rng(0).random(len(per)) - 0.5) * 0.28
            ax.scatter(x, per, s=16, color=rfig.PAL[j], alpha=0.75,
                       edgecolor="none")
            ax.plot([j - 0.28, j + 0.28], [per.mean()] * 2,
                    color=rfig.PAL[j], lw=2.5)
        ax.axhline(0, color="0.5", ls=":", lw=1)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
        ax.set_ylabel("peri - non-peri ripple rate (Hz)")
        ax.set_title("Ripple rate around a key press\n"
                     "one point per subject; below 0 = suppressed at the press")
        _png(fig, figs, "press_locked")

    # -------------------------------------------------------------- stillness
    if still_rows:
        S = pd.concat(still_rows, ignore_index=True)
        S.to_csv(os.path.join(out_dir, "stillness.csv"), index=False)
        g = (S.groupby(["bin_lo_s", "bin_hi_s"])
             .agg(n_ripples=("n_ripples", "sum"), exposure_s=("exposure_s", "sum"),
                  n_periods=("n_periods", "sum")).reset_index())
        g["rate_hz"] = g.n_ripples / g.exposure_s
        print("\n  [stillness] ripple rate by how long the subject stayed still")
        print("   " + g.round(3).to_string(index=False).replace("\n", "\n   "))
        out["stillness"] = g.to_dict("records")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        lab = [f"{a:g}-{b:g}s" if b < 1e8 else f">{a:g}s"
               for a, b in zip(g.bin_lo_s, g.bin_hi_s)]
        ax.bar(np.arange(len(g)), g.rate_hz, color=rfig.PAL[2], edgecolor="w")
        ax.set_xticks(np.arange(len(g))); ax.set_xticklabels(lab)
        ax.set_xlabel("Duration of the still period")
        ax.set_ylabel("Ripple rate (Hz)")
        for i, r in g.iterrows():
            ax.text(i, r.rate_hz, f"{int(r.n_periods)}", ha="center",
                    va="bottom", fontsize=8)
        ax.set_title("Ripple rate vs how long the subject stood still\n"
                     "(n periods above each bar)")
        _png(fig, figs, "stillness")

    # ------------------------------------------------------------------ drift
    if drift_rows:
        D = pd.concat(drift_rows, ignore_index=True)
        D.to_csv(os.path.join(out_dir, "session_drift.csv"), index=False)
        g = (D.groupby("frac_through")
             .agg(n=("n_ripples", "sum"), e=("exposure_s", "sum")).reset_index())
        g["rate_hz"] = g.n / g.e
        print("\n  [drift] ripple rate across the session")
        print(f"    first decile {g.rate_hz.iloc[0]:.3f} Hz -> "
              f"last decile {g.rate_hz.iloc[-1]:.3f} Hz "
              f"({100*(g.rate_hz.iloc[-1]/g.rate_hz.iloc[0]-1):+.1f}%)")
        out["drift"] = g.to_dict("records")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        per = (D.groupby(["subject_key", "frac_through"])
               .apply(lambda x: x.n_ripples.sum() / max(x.exposure_s.sum(), 1e-9))
               .rename("rate").reset_index())
        for s, gg in per.groupby("subject_key"):
            ax.plot(gg.frac_through, gg.rate, color="0.8", lw=0.6)
        m = per.groupby("frac_through").rate.mean()
        se = per.groupby("frac_through").rate.sem()
        ax.errorbar(m.index, m, yerr=se, color=rfig.OBS_LINE_C, lw=1.8, capsize=2)
        ax.set_xlabel("Fraction through the session")
        ax.set_ylabel("Ripple rate (Hz)")
        ax.set_title("Does ripple rate drift within a session?\n"
                     "grey = subjects, green = mean +- SEM")
        _png(fig, figs, "session_drift")

    # ------------------------------------------------------------ press PETHs
    if peth_acc and centres is not None:
        panels = []
        tr = {}
        for lab in sorted({l for _, l in peth_acc}):
            subs = sorted({s for s, l in peth_acc if l == lab})
            X = np.vstack([np.nanmean(np.vstack(peth_acc[(s, lab)]), axis=0)
                           for s in subs])
            tr[lab] = (centres, np.nanmean(X, axis=0),
                       np.nanstd(X, axis=0) / max(np.sqrt(len(subs)), 1), len(subs))
        panels.append(("Aligned to key presses", tr))
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        for j, (lab, (c, m, se, n)) in enumerate(tr.items()):
            ax.plot(c, m, color=rfig.PAL[j], lw=1.5, label=f"{lab} (n={n})")
            ax.fill_between(c, m - se, m + se, color=rfig.PAL[j], alpha=0.22, lw=0)
        ax.axvline(0, color="0.4", lw=1)
        ax.set_xlabel("Time from press (s)"); ax.set_ylabel("Ripple rate (Hz)")
        ax.legend(frameon=False, fontsize=9)
        ax.set_title("Ripple rate around key presses (100 ms bins, mean +- SEM)")
        _png(fig, figs, "peth_presses")

    # ------------------------------------------------- long vs short stillness
    if longshort_rows:
        L = pd.DataFrame(longshort_rows)
        L.to_csv(os.path.join(out_dir, "still_long_vs_short.csv"), index=False)
        r = _subject_t(L)
        print("\n  [still_long] long (>=8 s) vs short (0.5-2 s) still periods")
        if r:
            print(f"    mean diff = {r['mean']:+.4f} Hz, "
                  f"t({r['n_subjects']-1}) = {r['t']:+.2f}, p = {r['p']:.4g}"
                  f"   ({'MORE' if r['mean'] > 0 else 'fewer'} ripples when still longer)")
            out["still_long"] = {k: v for k, v in r.items() if k != "per_subject"}
            fig, ax = plt.subplots(figsize=(5.4, 4.2))
            per = L.groupby("subject_key")[["rate_short", "rate_long"]].mean().dropna()
            for _, row in per.iterrows():
                ax.plot([0, 1], [row.rate_short, row.rate_long], color="0.75", lw=0.8)
            ax.scatter(np.zeros(len(per)), per.rate_short, s=18,
                       color=rfig.PAL[1], zorder=3)
            ax.scatter(np.ones(len(per)), per.rate_long, s=18,
                       color=rfig.OBS_LINE_C, zorder=3)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["still 0.5-2 s", "still >=8 s"])
            ax.set_ylabel("Ripple rate (Hz)")
            ax.set_title(f"Ripple rate vs length of stillness\n"
                         f"paired by subject, p = {r['p']:.3g}")
            _png(fig, figs, "still_long_vs_short")

    # ------------------------------------------------------ within-grid drift
    if griddrift_rows:
        G = pd.concat(griddrift_rows, ignore_index=True)
        G.to_csv(os.path.join(out_dir, "within_grid_drift.csv"), index=False)
        per = (G.groupby(["subject_key", "frac_through_grid"])
               .apply(lambda x: x.n_ripples.sum() / max(x.exposure_s.sum(), 1e-9))
               .rename("rate").reset_index())
        m = per.groupby("frac_through_grid").rate.mean()
        se = per.groupby("frac_through_grid").rate.sem()
        print("\n  [grid_drift] ripple rate across a grid (first->last fifth)")
        print(f"    {m.iloc[0]:.3f} -> {m.iloc[-1]:.3f} Hz "
              f"({100*(m.iloc[-1]/m.iloc[0]-1):+.1f}%)")
        out["grid_drift"] = {"first": float(m.iloc[0]), "last": float(m.iloc[-1])}
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for s_, gg in per.groupby("subject_key"):
            ax.plot(gg.frac_through_grid, gg.rate, color="0.85", lw=0.6)
        ax.errorbar(m.index, m, yerr=se, color=rfig.OBS_LINE_C, lw=1.8, capsize=2)
        ax.set_xlabel("Fraction through the grid")
        ax.set_ylabel("Ripple rate (Hz)")
        ax.set_title("Within-grid drift\n"
                     "the control for any first-vs-later-traversal contrast")
        _png(fig, figs, "within_grid_drift")

    # ------------------------------------------------- stillness x phase
    if phase_rows:
        Ph = pd.DataFrame(phase_rows)
        Ph.to_csv(os.path.join(out_dir, "still_by_phase.csv"), index=False)
        print("\n  [still_by_phase] does the stillness effect differ by phase?")
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        for j, ph in enumerate(["explore", "plan", "execute"]):
            g = Ph[Ph.phase3 == ph]
            if not len(g):
                continue
            r = _subject_t(g)
            per = g.groupby("subject_key")[["rate_short", "rate_long"]].mean().dropna()
            if r:
                print(f"    {ph:9s} short {per.rate_short.mean():.3f} -> "
                      f"long {per.rate_long.mean():.3f} Hz | diff "
                      f"{r['mean']:+.4f}, t({r['n_subjects']-1}) = {r['t']:+.2f}, "
                      f"p = {r['p']:.4g}")
                out[f"still_by_phase::{ph}"] = {k: v for k, v in r.items()
                                                if k != "per_subject"}
            ax.errorbar([0, 1], [per.rate_short.mean(), per.rate_long.mean()],
                        yerr=[per.rate_short.sem(), per.rate_long.sem()],
                        marker="o", lw=1.6, capsize=3,
                        color=rfig._cond_color(ph, j), label=f"{ph} (n={len(per)})")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["still 0.5-2 s", "still >=8 s"])
        ax.set_ylabel("Ripple rate (Hz)"); ax.legend(frameon=False, fontsize=9)
        ax.set_title("Stillness effect by task phase")
        _png(fig, figs, "still_by_phase")

    # ------------------------------------------------------- prospective
    if prosp_rows:
        Pr = pd.concat(prosp_rows, ignore_index=True)
        Pr.to_csv(os.path.join(out_dir, "prospective.csv"), index=False)
        g = (Pr.groupby(["pause_lo_s", "pause_hi_s"])
             .agg(n=("n_ripples", "sum"), e=("exposure_s", "sum"),
                  k=("n_periods", "sum")).reset_index())
        g["rate_hz"] = g.n / g.e
        print("\n  [prospective] SAME 1 s window (0.5-1.5 s after the press), "
              "binned by how long the pause turned out to be")
        print("   " + g.round(3).to_string(index=False).replace("\n", "\n   "))
        out["prospective"] = g.to_dict("records")
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        lab = [f"{a:g}-{b:g}s" if b < 1e8 else f">{a:g}s"
               for a, b in zip(g.pause_lo_s, g.pause_hi_s)]
        ax.bar(np.arange(len(g)), g.rate_hz, color=rfig.PAL[4], edgecolor="w")
        ax.set_xticks(np.arange(len(g))); ax.set_xticklabels(lab)
        ax.set_xlabel("Eventual length of the pause")
        ax.set_ylabel("Ripple rate in the fixed 1 s window (Hz)")
        for i, r in g.iterrows():
            ax.text(i, r.rate_hz, f"{int(r.k)}", ha="center", va="bottom", fontsize=8)
        ax.set_title("Prospective test of F1\n"
                     "identical window, identical distance from the press")
        _png(fig, figs, "prospective")

    # --------------------------------------------------------- grid onset
    if onset_rows:
        On = pd.concat(onset_rows, ignore_index=True)
        On.to_csv(os.path.join(out_dir, "grid_onset.csv"), index=False)
        r = _subject_t(On)
        print("\n  [grid_onset] ripple rate around the start of a new grid")
        if r:
            print(f"    peri - non-peri = {r['mean']:+.4f} Hz, "
                  f"t({r['n_subjects']-1}) = {r['t']:+.2f}, p = {r['p']:.4g}")
            out["grid_onset"] = {k: v for k, v in r.items() if k != "per_subject"}
        if onset_acc:
            subs = sorted(onset_acc)
            c0 = onset_acc[subs[0]][0][0]
            X = np.vstack([np.nanmean(np.vstack([p for _, p in onset_acc[s]]), axis=0)
                           for s in subs])
            m = np.nanmean(X, axis=0); se = np.nanstd(X, axis=0) / np.sqrt(len(subs))
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            ax.plot(c0, m, color=rfig.OBS_LINE_C, lw=1.6)
            ax.fill_between(c0, m - se, m + se, color=rfig.OBS_LINE_C,
                            alpha=0.22, lw=0)
            ax.axvline(0, color="0.4", lw=1)
            ax.set_xlabel("Time from grid onset (s)")
            ax.set_ylabel("Ripple rate (Hz)")
            ax.set_title(f"New grid onset: the moment a new problem is given\n"
                         f"mean +- SEM across {len(subs)} subjects")
            _png(fig, figs, "grid_onset")

    pd.DataFrame([{"probe": k, **(v if isinstance(v, dict) else {})}
                  for k, v in out.items() if isinstance(v, dict)]).to_csv(
        os.path.join(out_dir, "explore_summary.csv"), index=False)
    print(f"\n saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
