#!/usr/bin/env python3
"""1/f surrogate noise floor per derivation (van Schalkwijk & Helfrich 2026).

Reuses the detection stage's event counts and clean intervals, so the only new
cost is the surrogates themselves.
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import mc.analyse.swr_io as swr_io, mc.analyse.swr_artifact as art
import mc.analyse.swr_surrogate as sur
try: import fire
except ImportError: fire = None

np.random.seed(42)
ANALYSIS_NAME = "swr_v1"

def run(sessions=None, analysis_name=ANALYSIS_NAME, n_surrogates=3, save_all=True):
    swr_io.start_log(os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr"), "swr_surrogate_control")
    R = swr_io.get_data_root(); rows = []
    paths = sorted(glob.glob(os.path.join(swr_io.derivatives_dir(R), "s*",
                             "LFP-ripples", analysis_name, "channel_qc.csv")))
    for qp in paths:
        sess = int(qp.split(os.sep)[-4][1:])
        if sessions is not None and sess not in [int(s) for s in sessions]: continue
        cd = os.path.join(swr_io.session_deriv_dir(sess, R), "LFP-clean", analysis_name)
        if not os.path.isfile(os.path.join(cd, "continuous.npy")): continue
        sig = np.load(os.path.join(cd, "continuous.npy"), mmap_mode='r')
        pairs = pd.read_csv(os.path.join(cd, "pairs.csv"))
        fs = float(json.load(open(os.path.join(cd, "meta.json")))["fs"])
        qc = pd.read_csv(qp).set_index("pair_id")
        # Reuse the artifact mask the detection stage already computed and
        # saved. Recomputing it costs 30 filter+Hilbert passes over the whole
        # session per derivation, for information already on disk.
        ivp = os.path.join(os.path.dirname(qp), "clean_intervals.csv")
        iv_all = pd.read_csv(ivp) if os.path.isfile(ivp) else pd.DataFrame()
        print(f"\n=== s{sess:02d} ({len(pairs)} pairs) ===", flush=True)
        for i, p in pairs.iterrows():
            if p.pair_id not in qc.index or bool(qc.loc[p.pair_id, "excluded"]): continue
            x = np.asarray(sig[i], float)
            clean = np.zeros(len(x), bool)
            for _, iv in iv_all[iv_all.pair_id == p.pair_id].iterrows():
                clean[int(iv.start_s * fs):int(iv.stop_s * fs)] = True
            if not clean.any():
                continue
            r = sur.surrogate_noise_floor(
                x, fs, clean, n_surrogates=n_surrogates,
                n_observed=int(qc.loc[p.pair_id, "n_events"]),
                clean_s=float(qc.loc[p.pair_id, "clean_s"]))
            if not r: continue
            r.update(session=sess, pair_id=p.pair_id, roi=p.get("pair_roi_atlas"))
            rows.append(r)
            print(f"  {p.pair_id:22s} chi={r['aperiodic_exponent']:.2f} "
                  f"r2={r['aperiodic_r2']:.2f} obs={r['rate_observed_hz']:.3f} "
                  f"sur={r['rate_surrogate_hz']:.3f}+-{r['rate_surrogate_sd']:.3f} "
                  f"FP={r['false_positive_frac']:.0%}", flush=True)
    if not rows: print("nothing to do"); return
    d = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print(f"derivations: {len(d)} across {d.session.nunique()} sessions")
    print(f"median false-positive fraction : {d.false_positive_frac.median():.0%}"
          f"   [van Schalkwijk & Helfrich 2026: ~77% awake MTL]")
    print(f"median observed rate           : {d.rate_observed_hz.median():.3f} Hz")
    print(f"median surrogate (noise) rate  : {d.rate_surrogate_hz.median():.3f} Hz")
    print(f"median excess over noise       : {d.rate_excess_hz.median():.3f} Hz")
    print(f"aperiodic exponent chi         : median {d.aperiodic_exponent.median():.2f} "
          f"(range {d.aperiodic_exponent.min():.2f}-{d.aperiodic_exponent.max():.2f}), "
          f"fit r2 median {d.aperiodic_r2.median():.2f}")
    if save_all:
        out = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
        os.makedirs(out, exist_ok=True)
        d.to_csv(os.path.join(out, "surrogate_noise_floor.csv"), index=False)
        print(f"saved -> {out}/surrogate_noise_floor.csv")

if __name__ == "__main__":
    fire.Fire(run) if fire else run()
