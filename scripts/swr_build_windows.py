#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 5: build H1 window counts across sessions.

For every design, produces one row per (session, derivation, window) with the
ripple count, artifact-free exposure, and the covariates the H1 model needs.

Output: derivatives/group/swr/window_counts_{design}.csv

Usage:
    python scripts/swr_build_windows.py
    python scripts/swr_build_windows.py --designs="['sections','pauses']"

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
import mc.analyse.swr_windows as win

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_v1"


def _movement_rate(session, data_root, starts, stops):
    """Button presses per second within each window.

    Uses the ragged per-repeat `button_presses.csv`, which maps 1:1 onto
    behaviour rows. The rectangular `buttons_per_25ms_*` files have an
    unverified time origin (a grid-1 file spans 408 s against a 172 s grid), so
    they are deliberately not used until that is resolved.
    """
    p = os.path.join(swr_io.session_deriv_dir(session, data_root),
                     "cells_and_beh", "button_presses.csv")
    if not os.path.isfile(p):
        return np.full(len(starts), np.nan)
    try:
        beh = swr_io.load_behaviour(session, data_root=data_root)
    except Exception:
        return np.full(len(starts), np.nan)
    counts = []
    with open(p) as f:
        for line in f:
            toks = [t for t in line.strip().split(',') if t]
            counts.append(len(toks))
    n = min(len(counts), len(beh))
    if n == 0:
        return np.full(len(starts), np.nan)
    rep_start = beh.new_grid_onset.to_numpy(float)[:n]
    rep_end = beh.t_D.to_numpy(float)[:n]
    rep_rate = np.asarray(counts[:n], float) / np.maximum(rep_end - rep_start, 1e-6)
    # a window's movement rate = that of the repeat it falls in
    idx = np.clip(np.searchsorted(rep_start, starts, side='right') - 1, 0, n - 1)
    inside = (starts >= rep_start[idx]) & (starts <= rep_end[idx])
    out = np.where(inside, rep_rate[idx], np.nan)
    return out


def build(designs=None, analysis_name=ANALYSIS_NAME, save_all=True, verbose=True,
          lock_s=None, suffix=''):
    swr_io.start_log(os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr"), "swr_build_windows")
    R = swr_io.get_data_root()
    designs = list(designs) if designs else list(win.DESIGNS)
    manifest = pd.read_csv(os.path.join(swr_io.derivatives_dir(R), "group",
                                        "swr", "session_manifest.csv"))
    subj = manifest.set_index("session")[["subject_key", "recording_site",
                                          "subject_label"]].to_dict("index")
    # first-session flag per subject (task familiarity may itself affect rate)
    first_of = (manifest.dropna(subset=["subject_key"])
                        .sort_values("session")
                        .groupby("subject_key")["session"].min().to_dict())

    # aperiodic exponent per derivation, if the surrogate stage has run
    chi_p = os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                         "surrogate_noise_floor.csv")
    chi = (pd.read_csv(chi_p).set_index(["session", "pair_id"])
           if os.path.isfile(chi_p) else None)

    out = {d: [] for d in designs}
    for evp in sorted(glob.glob(os.path.join(swr_io.derivatives_dir(R), "s*",
                                             "LFP-ripples", analysis_name,
                                             "ripple_events.csv"))):
        sess = int(evp.split(os.sep)[-4][1:])
        rd = os.path.dirname(evp)
        ev = pd.read_csv(evp)
        iv_all = pd.read_csv(os.path.join(rd, "clean_intervals.csv"))
        qc = pd.read_csv(os.path.join(rd, "channel_qc.csv")).set_index("pair_id")
        try:
            beh = swr_io.load_behaviour(sess, data_root=R)
        except Exception as e:
            print(f"  s{sess:02d}: behaviour unreadable ({e})"); continue

        meta = subj.get(sess, {})
        for design in designs:
            try:
                kw = {'lock_s': lock_s} if (lock_s and design in
                      ('reward_locked','discovery','error_correct')) else {}
                w = win.build_windows(beh, design, **kw)
            except Exception as e:
                print(f"  s{sess:02d} {design}: {e}"); continue
            if not len(w):
                continue
            mv = _movement_rate(sess, R, w.start_s.to_numpy(), w.end_s.to_numpy())

            for pair_id, evp_ in ev[ev.passed_strict].groupby("pair_id"):
                if pair_id in qc.index and bool(qc.loc[pair_id, "excluded"]):
                    continue
                iv = iv_all[iv_all.pair_id == pair_id][["start_s", "stop_s"]].to_numpy()
                a = win.assign_events_to_windows(
                    evp_.t_peak_s.to_numpy(float), w, iv, shift_s=0.0)
                a["session"] = sess
                a["pair_id"] = pair_id
                a["pair_roi"] = qc.loc[pair_id, "pair_roi"] if pair_id in qc.index else None
                a["subject_key"] = meta.get("subject_key")
                a["recording_site"] = meta.get("recording_site")
                a["is_first_session"] = (first_of.get(meta.get("subject_key")) == sess)
                a["movement_rate"] = mv
                if chi is not None and (sess, pair_id) in chi.index:
                    a["aperiodic_chi"] = float(chi.loc[(sess, pair_id), "aperiodic_exponent"])
                    a["surrogate_rate_hz"] = float(chi.loc[(sess, pair_id), "rate_surrogate_hz"])
                out[design].append(a)
        if verbose:
            print(f"  s{sess:02d} done", flush=True)

    gdir = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
    os.makedirs(gdir, exist_ok=True)
    for design, parts in out.items():
        if not parts:
            continue
        d = pd.concat(parts, ignore_index=True)
        d = d[d.exposure_s > 0]
        print("\n" + "=" * 72)
        print(f" DESIGN: {design}   ({len(d)} rows, {d.session.nunique()} sessions, "
              f"{d.pair_id.nunique()} derivations)")
        print("=" * 72)
        s = (d.groupby("condition")
               .agg(n_windows=("n_ripples", "size"),
                    ripples=("n_ripples", "sum"),
                    exposure_s=("exposure_s", "sum"),
                    median_exposure=("exposure_s", "median")))
        s["rate_hz"] = (s.ripples / s.exposure_s).round(4)
        print(s.round(2).to_string())
        if save_all:
            d.to_csv(os.path.join(gdir, f"window_counts_{design}{suffix}.csv"), index=False)
    if save_all:
        print(f"\nsaved -> {gdir}/window_counts_*.csv")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(build)
    else:
        build()
