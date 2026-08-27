#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 6: the H1 statistical test.

Poisson GLM on ripple counts with a log(artifact-free exposure) offset and
cluster-robust SEs by patient, plus a circular-shift permutation test on the
artifact-free axis -- the latter being the PRIMARY inference, because
cluster-robust SEs are anticonservative with few clusters.

This runs the pre-registered confirmatory test specified in
final_results/ripple_analysis/methods.md section 12.4. Do not change the
specification when moving to held-out data.

Outputs (derivatives/group/swr/):
    h1_glm_{design}{suffix}.csv
    h1_permutation_{design}{suffix}.json
    h1_null_{design}{suffix}_{contrast}.npy

Usage:
    python scripts/swr_h1_stats.py                       # dev set, 1 s window
    python scripts/swr_h1_stats.py --n_perms=1000
    python scripts/swr_h1_stats.py --sessions="[33,34,35]"   # held-out

@author: Svenja Kuchenhoff
"""

import os
import sys
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.swr_windows as win
import mc.analyse.swr_stats as st

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_v1"
LOCK_S = 1.0                 # pre-registered
DESIGN = "discovery"
N_PERMS = 1000
SEED = 42

np.random.seed(SEED)


def _gather(analysis_name, design, lock_s, sessions=None):
    """Per-derivation event trains + per-session windows."""
    R = swr_io.get_data_root()
    per_pair, wbys = [], {}
    for evp in sorted(glob.glob(os.path.join(swr_io.derivatives_dir(R), "s*",
                                             "LFP-ripples", analysis_name,
                                             "ripple_events.csv"))):
        sess = int(evp.split(os.sep)[-4][1:])
        if sessions is not None and sess not in [int(s) for s in sessions]:
            continue
        rd = os.path.dirname(evp)
        ev = pd.read_csv(evp)
        iv_all = pd.read_csv(os.path.join(rd, "clean_intervals.csv"))
        qc = pd.read_csv(os.path.join(rd, "channel_qc.csv")).set_index("pair_id")
        try:
            beh = swr_io.load_behaviour(sess, data_root=R)
        except Exception as e:
            print(f"  s{sess:02d}: behaviour unreadable ({e})")
            continue
        kw = {"lock_s": lock_s} if design in ("discovery", "reward_locked",
                                              "error_correct") else {}
        wbys[sess] = win.build_windows(beh, design, **kw)
        for pid, e in ev[ev.passed_strict].groupby("pair_id"):
            if pid in qc.index and bool(qc.loc[pid, "excluded"]):
                continue
            per_pair.append({
                "session": sess, "pair_id": pid,
                "events": e.t_peak_s.to_numpy(float),
                "intervals": iv_all[iv_all.pair_id == pid][["start_s", "stop_s"]].to_numpy(),
                "meta": {},
            })
    return per_pair, wbys


def run(design=DESIGN, lock_s=LOCK_S, analysis_name=ANALYSIS_NAME,
        n_perms=N_PERMS, seed=SEED, sessions=None, suffix="_1s",
        save_all=True):
    swr_io.start_log(os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr"), "swr_h1_stats")
    R = swr_io.get_data_root()
    gdir = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
    counts_p = os.path.join(gdir, f"window_counts_{design}{suffix}.csv")
    if not os.path.isfile(counts_p):
        raise FileNotFoundError(
            f"{counts_p} not found -- run scripts/swr_build_windows.py "
            f"--designs=\"['{design}']\" --lock_s={lock_s} --suffix={suffix}")
    d = pd.read_csv(counts_p)
    if sessions is not None:
        d = d[d.session.isin([int(s) for s in sessions])]

    print(f"\n{design} ({lock_s} s): {len(d)} rows, {d.session.nunique()} sessions, "
          f"{d.pair_id.nunique()} derivations, {d.subject_key.nunique()} subjects")

    # ---------------- GLM (reported, but NOT the primary inference) --------
    out = {"design": design, "lock_s": lock_s, "n_perms": n_perms, "seed": seed,
           "n_sessions": int(d.session.nunique()),
           "n_derivations": int(d.pair_id.nunique()),
           "n_subjects": int(d.subject_key.nunique()),
           "created": datetime.now().isoformat(timespec="seconds")}

    glm_tab = None
    if design == "discovery":
        d["discovery"] = pd.Categorical(d.discovery, categories=["later", "first"])
        d["state"] = pd.Categorical(d.state, categories=["A", "B", "C", "D"])
        r = st.fit_count_glm(d, "n_ripples ~ C(state)*C(discovery)",
                             cluster_col="subject_key")
        if r["overdispersed"]:
            print(f"  dispersion {r['dispersion']:.2f} > 1.5 -> refitting negative binomial")
            r = st.fit_count_glm(d, "n_ripples ~ C(state)*C(discovery)",
                                 cluster_col="subject_key", family="nb")
        glm_tab = r["table"]
        out.update(dispersion=r["dispersion"], glm_family=r["family"],
                   n_clusters=r["n_clusters"])
        print(f"\n  {r['family'].upper()} GLM | dispersion={r['dispersion']:.2f} "
              f"| clusters={r['n_clusters']}")
        if r["n_clusters"] < 30:
            print(f"  !! only {r['n_clusters']} clusters -- cluster-robust SEs are "
                  f"ANTICONSERVATIVE here. Use the permutation p-value.")
        keep = [i for i in glm_tab.index if "discovery" in i]
        print(glm_tab.loc[keep, ["coef", "se", "z", "p", "rate_ratio"]].round(3).to_string())

    # ---------------- permutation (PRIMARY) --------------------------------
    per_pair, wbys = _gather(analysis_name, design, lock_s, sessions)
    print(f"\n  permutation: {len(per_pair)} derivations, {n_perms} shifts "
          f"(shift=0 is row 0 = observed)")

    contrasts = {"interaction": st.interaction_contrast,
                 "simple_D": st.simple_contrast}
    for name, fn in contrasts.items():
        res = st.circular_shift_test(per_pair, wbys, fn, n_perms=n_perms,
                                     seed=seed, scope="session")
        print(f"\n  --- {name} ---")
        print(f"    observed {res['observed']:+.4f} Hz | "
              f"null {res['null_mean']:+.4f} +- {res['null_sd']:.4f} | "
              f"z {res['z_vs_null']:+.2f}")
        print(f"    p one-tailed {res['p_one_tailed']:.4f} | "
              f"two-tailed {res['p_two_tailed']:.4f}   [PRIMARY]")
        null = res.pop("null")
        out[name] = res
        if save_all:
            np.save(os.path.join(gdir, f"h1_null_{design}{suffix}_{name}.npy"), null)

    if save_all:
        os.makedirs(gdir, exist_ok=True)
        if glm_tab is not None:
            glm_tab.to_csv(os.path.join(gdir, f"h1_glm_{design}{suffix}.csv"))
        with open(os.path.join(gdir, f"h1_permutation_{design}{suffix}.json"), "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n  saved -> {gdir}/h1_*{design}{suffix}.*")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(run)
    else:
        run()
