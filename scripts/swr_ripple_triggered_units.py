#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do mPFC units change their firing around hippocampal ripples -- and only while
the subject still has a plan to work out?

SK's mechanism, in her words: the subject uncovers rewards, the hippocampus
tracks where they are, tells mPFC what to do, mPFC then carries a representation
of how to execute, and execution proceeds. So hippocampus-to-mPFC communication
is expected WHILE THE ROUTE IS STILL BEING WORKED OUT and not once it is known.

Pooling every ripple in the session therefore averages two different states
together, which is the same mistake that sank every window-averaged contrast in
this project. Ripples are split by the phase of the repeat they fall in:

    unsolved   explore + plan -- the grid has not yet been solved without error
    solved     execute        -- the route is known and is being repeated

The prediction is an interaction: mPFC coupling in `unsolved`, little or none in
`solved`. Hippocampal units are the positive control in BOTH, since they should
fire with ripples regardless of what the subject knows.

The project's claim is that hippocampus informs mPFC. This is the most direct
test available in this dataset without any new preprocessing: ripple times from
the bundle, single-unit firing from the 25 ms matrices, one clock.

  python scripts/swr_ripple_triggered_units.py run --bundle=<bundle dir>
  python scripts/swr_ripple_triggered_units.py run --half_s=1.0 --n_shift=200

Read the HC panel FIRST. Hippocampal units must show a peri-ripple firing
increase; that is the most reproducible fact about ripples in the literature. If
they do not, the spike and LFP clocks are misaligned and the mPFC panel is
meaningless. It is a positive control, not a result.

Test window is pre-declared as 0 to +200 ms after the ripple peak (the standard
window for ripple-locked cortical activity), so it is not chosen after seeing
the time course. The whole course is plotted regardless.

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
import mc.analyse.swr_units as swu
import mc.plotting.ripple_figures as rfig

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

HALF_S = 1.0
N_SHIFT = 200
SEED = 42
TEST_WIN_S = (0.0, 0.200)      # PRE-DECLARED, not tuned on these data
REGIONS = {"mPFC": ("ACC",), "HC": ("HC",), "EC": ("EC",), "OFC": ("OFC",),
           "PCC": ("PCC",), "AMY": ("AMY",)}

# explore + plan = the route is not yet known; execute = it is
UNSOLVED = ("explore", "plan")
SOLVED = ("execute",)
# The circular shift must stay inside the state the ripple came from. Without
# this, unsolved ripples -- which occupy a short span at the start of each grid
# -- get shifted into execution, and their null is estimated from a regime with
# different firing. That is what made the hippocampal positive control fail in
# the unsolved state while passing in solved.
_PHASES = {"unsolved": UNSOLVED, "solved": SOLVED, "all": None}


def _ripple_phase(ripple_t, beh):
    """Phase of the repeat each ripple falls in, or None if outside any repeat."""
    import mc.analyse.swr_windows as win
    b = win.add_phase3(beh)
    on = b.new_grid_onset.to_numpy(float)
    end = b.t_D.to_numpy(float)
    ph = b.phase3.to_numpy()
    out = np.full(len(ripple_t), None, object)
    for i, t in enumerate(np.asarray(ripple_t, float)):
        hit = np.flatnonzero((on <= t) & (end >= t))
        if hit.size:
            out[i] = ph[hit[0]]
    return out


def _settings(half_s, n_shift, bundle):
    return {"half_s": half_s, "n_shift": n_shift, "bin_s": swu.BIN_S,
            "test_window_s": list(TEST_WIN_S), "dedup_s": swu.DEDUP_S,
            "seed": SEED, "bundle": bundle,
            "row_order_source": "all_cells_region_labels_sub{XX}.txt",
            "created": datetime.now().isoformat(timespec="seconds")}


def run(bundle=None, half_s=HALF_S, n_shift=N_SHIFT, out_dir=None,
        save_all=True, split_by_phase=True, n_draws=5):
    R = swr_io.get_data_root()
    if out_dir is None:
        out_dir = os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                               "ripple_triggered_units")
    os.makedirs(out_dir, exist_ok=True)
    swr_io.start_log(out_dir, "swr_ripple_triggered_units")
    np.random.seed(SEED)

    if bundle is None:
        bundle = os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                              "bundle")
    rip = pd.read_csv(os.path.join(bundle, "ripples.csv"))
    print(f"\n  {len(rip)} ripples across {rip.session.nunique()} sessions")

    rows, curves = [], []
    for sess, r in rip.groupby("session"):
        sess = int(sess)
        lab = swu.unit_labels(sess, R)
        if not lab:
            continue
        try:
            beh = swr_io.load_behaviour(sess, data_root=R)
        except Exception as e:
            print(f"  s{sess:02d}: behaviour unreadable ({e})"); continue
        t_all = swu.dedup_ripples(r.t_peak_s.to_numpy(float))
        subj = str(r.subject_key.iloc[0])
        if split_by_phase:
            ph = _ripple_phase(t_all, beh)
            subsets = {"unsolved": t_all[np.isin(ph, UNSOLVED)],
                       "solved": t_all[np.isin(ph, SOLVED)]}
            # MATCH THE EVENT COUNT. z is (obs - null_mean)/null_sd, and the
            # null sd shrinks with more events, so a state with twice as many
            # ripples gets a systematically larger |z| for the same underlying
            # coupling. Solved has ~2x the ripples of unsolved here (median 625
            # vs 247), and |z| correlates with log n, so an unmatched
            # comparison manufactures exactly the interaction we are looking
            # for. Each state is therefore subsampled to the smaller n, several
            # times, and the z averaged.
            n_match = min((len(v) for v in subsets.values()), default=0)
        else:
            subsets = {"all": t_all}
            n_match = None
        for state, t in subsets.items():
            if len(t) < 20:          # too few ripples for a stable null
                continue
            if n_match and len(t) > n_match:
                zs = []
                for d in range(n_draws):
                    rs = np.random.default_rng(SEED + d)
                    tt = np.sort(rs.choice(t, size=n_match, replace=False))
                    g2 = swu.peri_ripple_matrix(sess, tt, beh, half_s=half_s,
                                                n_shift=n_shift, data_root=R,
                                                seed=SEED + d,
                                                restrict_phases=_PHASES.get(state))
                    if g2 is None:
                        continue
                    offs, obs, nm, nsd, labels, n_used = g2
                    zs.append((obs - nm) / np.where(nsd > 0, nsd, np.nan))
                if not zs:
                    continue
                z = np.nanmean(np.stack(zs), axis=0)
                n_used = n_match
            else:
                got = swu.peri_ripple_matrix(sess, t, beh, half_s=half_s,
                                             n_shift=n_shift, data_root=R,
                                             seed=SEED,
                                             restrict_phases=_PHASES.get(state))
                if got is None:
                    continue
                offs, obs, nm, nsd, labels, n_used = got
                z = (obs - nm) / np.where(nsd > 0, nsd, np.nan)
            m = (offs >= TEST_WIN_S[0]) & (offs < TEST_WIN_S[1])
            for i, lb in enumerate(labels):
                region = next((k for k, v in REGIONS.items()
                               if lb.upper() in v), "other")
                rows.append({"session": sess, "subject_key": subj, "unit": i,
                             "native_label": lb, "region": region,
                             "state": state,
                             "z_test_window": float(np.nanmean(z[i, m])),
                             "z_peak": float(np.nanmax(z[i])),
                             "n_ripple_windows": n_used})
                curves.append(np.r_[[sess, i], z[i]])
            print(f"  s{sess:02d} {state:9s}: {len(labels):3d} units, "
                  f"{len(t):5d} ripples, {n_used:5d} windows")

    if not rows:
        print("\n  nothing computable"); return None
    tab = pd.DataFrame(rows)
    Z = pd.DataFrame(curves, columns=["session", "unit"] + [f"{o:.3f}" for o in offs])

    print("\n" + "=" * 74)
    print(f" RIPPLE-TRIGGERED FIRING  (z vs {n_shift} within-grid circular shifts)")
    print(f" pre-declared test window {TEST_WIN_S[0]*1000:.0f}-{TEST_WIN_S[1]*1000:.0f} ms")
    print("=" * 74)
    from scipy import stats as st
    summ = []
    grp = ["region", "state"] if "state" in tab.columns else ["region"]
    for key, g in tab.groupby(grp):
        region = key[0] if isinstance(key, tuple) else key
        state = key[1] if isinstance(key, tuple) else "all"
        if len(g) < 3:
            continue
        # unit-level t-test, and a subject-level one because units within a
        # subject are not independent
        t_u, p_u = st.ttest_1samp(g.z_test_window.dropna(), 0.0)
        per_s = g.groupby("subject_key").z_test_window.mean().dropna()
        t_s, p_s = (st.ttest_1samp(per_s, 0.0) if len(per_s) > 2
                    else (np.nan, np.nan))
        summ.append({"region": region, "state": state, "n_units": len(g),
                     "n_subjects": int(g.subject_key.nunique()),
                     "mean_z": float(g.z_test_window.mean()),
                     "t_units": float(t_u), "p_units": float(p_u),
                     "t_subjects": float(t_s), "p_subjects": float(p_s)})
    summ = pd.DataFrame(summ).sort_values("region")
    print(summ.round(4).to_string(index=False))
    print("\n  HC is the positive control -- if it is not clearly positive,")
    print("  the spike and LFP clocks are not aligned and nothing else counts.")

    # ---- the interaction: does mPFC coupling depend on whether the route is known?
    inter = {}
    if "state" in tab.columns and set(tab.state) >= {"unsolved", "solved"}:
        print("\n  INTERACTION  (per unit, unsolved minus solved)")
        for region in ("HC", "mPFC"):
            g = tab[tab.region == region]
            w = g.pivot_table(index=["session", "unit"], columns="state",
                              values="z_test_window")
            w = w.dropna(subset=["unsolved", "solved"])
            if len(w) < 3:
                continue
            d = w["unsolved"] - w["solved"]
            t_u, p_u = st.ttest_1samp(d, 0.0)
            key = g.drop_duplicates(["session", "unit"]).set_index(["session", "unit"])
            per_s = d.to_frame("d").join(key["subject_key"]).groupby(
                "subject_key").d.mean().dropna()
            t_s, p_s = (st.ttest_1samp(per_s, 0.0) if len(per_s) > 2
                        else (np.nan, np.nan))
            inter[region] = {"n_units": int(len(d)), "n_subjects": int(len(per_s)),
                             "mean_diff": float(d.mean()),
                             "t_units": float(t_u), "p_units": float(p_u),
                             "t_subjects": float(t_s), "p_subjects": float(p_s)}
            print(f"    {region:5s} unsolved-solved = {d.mean():+.3f}, "
                  f"units t({len(d)-1}) = {t_u:+.2f} p = {p_u:.4g} | "
                  f"subjects t({len(per_s)-1}) = {t_s:+.2f} p = {p_s:.4g}")

    if save_all:
        tab.to_csv(os.path.join(out_dir, "unit_ripple_modulation.csv"), index=False)
        Z.to_csv(os.path.join(out_dir, "peri_ripple_z_curves.csv"), index=False)
        summ.to_csv(os.path.join(out_dir, "region_summary.csv"), index=False)
        with open(os.path.join(out_dir, "settings.json"), "w") as f:
            json.dump(_settings(half_s, n_shift, bundle), f, indent=2)
        try:
            rfig.ripple_triggered_units_figure(
                offs, Z, tab, test_win=TEST_WIN_S,
                out_stem=os.path.join(out_dir, "figures",
                                      "ripple_triggered_units"),
                title="Single-unit firing around hippocampal ripples")
        except Exception as e:
            print(f"  figure failed: {type(e).__name__}: {e}")
        # the output contract: what was asked, what came back, what we take it to mean
        mpfc_u = summ[(summ.region == "mPFC") & (summ.state == "unsolved")] \
            if "state" in summ.columns else summ[summ.region == "mPFC"]
        hc_u = summ[(summ.region == "HC") & (summ.state == "unsolved")] \
            if "state" in summ.columns else summ[summ.region == "HC"]
        hc_ok = bool(len(hc_u) and hc_u.iloc[0]["p_units"] < 0.05
                     and hc_u.iloc[0]["mean_z"] > 0)
        mpfc_sig = bool(len(mpfc_u) and mpfc_u.iloc[0]["p_subjects"] < 0.05)
        inter_sig = bool(inter.get("mPFC", {}).get("p_subjects", 1) < 0.05)
        if not hc_ok:
            concl = ("POSITIVE CONTROL FAILED: hippocampal units do not show "
                     "ripple-locked firing, so the spike and LFP clocks are not "
                     "aligned. No statement about mPFC is interpretable.")
        elif mpfc_sig and inter_sig:
            concl = ("mPFC units are ripple-coupled while the route is still "
                     "being worked out, and significantly less so once it is "
                     "known -- the predicted interaction. Exploratory; needs "
                     "confirming on data that did not suggest it.")
        elif mpfc_sig:
            concl = ("mPFC units are ripple-coupled during unsolved trials, but "
                     "the unsolved-vs-solved interaction is not significant, so "
                     "state-specificity is not established.")
        else:
            concl = ("No detectable ripple-locked mPFC firing, in either state. "
                     "The positive control passes, so this is a real null for "
                     "this measure -- it does not rule out a representational "
                     "effect (reactivation of the plan) that a mean firing rate "
                     "would miss.")
        swr_io.write_result(
            out_dir, "ripple_triggered_units",
            hypothesis=("Hippocampal ripples carry plan-relevant information to "
                        "mPFC while the subject is still working out the route, "
                        "so mPFC units should be ripple-coupled during unsolved "
                        "trials (explore+plan) and less so during execution. "
                        "HC units are the positive control in both states."),
            tests=summ.to_dict("records") + [{"interaction": inter}],
            conclusion=concl,
            extra={"test_window_s": list(TEST_WIN_S), "n_shift": n_shift,
                   "positive_control_passed": hc_ok})
        print(f"\n saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire({"run": run})
    else:
        run()
