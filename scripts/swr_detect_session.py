#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 4 of the SWR pipeline: artifact rejection + ripple detection.

Reads `continuous.npy` from swr_extract_continuous, applies Chen's five
artifact criteria plus the Janca IED detector, estimates the ripple threshold
over the whole artifact-free session, detects events and flags them against
Chen's four spectral criteria.

Fast to re-run (~seconds to a minute per session) because it never touches the
raw files -- which is the entire reason extraction and detection are separate
stages.

Outputs, per session:
    derivatives/s{XX}/LFP-ripples/{name}/ripple_events.csv
    derivatives/s{XX}/LFP-ripples/{name}/channel_qc.csv
    derivatives/s{XX}/LFP-ripples/{name}/clean_intervals.csv
    derivatives/s{XX}/LFP-ripples/{name}/detector_diag.json
    derivatives/s{XX}/LFP-ripples/{name}/settings.json

Usage:
    python scripts/swr_detect_session.py --session=38

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
import mc.analyse.swr_preproc as pp
import mc.analyse.swr_artifact as art
import mc.analyse.swr_detect as det
import mc.analyse.swr_report as swr_report

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_v1"


def _settings_dict(session, analysis_name):
    return {
        "analysis_name": analysis_name, "session": int(session),
        "ripple_band_hz": list(det.RIPPLE_BAND),
        "threshold_extent_sd": det.LO_SD,
        "threshold_peak_sd": det.PEAK_SD,
        "threshold_ceiling_sd": det.HI_SD,
        "threshold_rationale": ("3.0 SD chosen by maximising excess over the 1/f "
                                "surrogate noise floor; Chen used 1.5 SD, which "
                                "here leaves 97% of detections inside the floor"),
        "sensitivity_analysis": "swr_lo1.5_sensitivity (Chen's 1.5 SD)",
        "threshold_scope": "whole artifact-free session (NOT per snippet)",
        "duration_ms": list(det.DUR_MS),
        "merge_gap_ms": det.MERGE_GAP_MS,
        "rms_window_ms": det.RMS_WIN_MS,
        "artifact_iqr_k": art.IQR_K,
        "artifact_pad_s": art.PAD_S,
        "min_clean_s": art.MIN_CLEAN_S,
        "max_contaminated_frac": art.MAX_CONTAM_FRAC,
        "ied_detector": "Janca et al. 2015 (log-normal envelope model)",
        "spectral_criteria": "Chen et al. 2025, 4 peak-based flags (bitmask)",
        "criterion2_scope_primary": "strict = Chen literal, 30-200 Hz outside band",
        "criterion2_scope_sensitivity": "relaxed = 120-200 Hz only",
        "primary_analysis": "strict (pre-declared)",
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def detect_session(session, analysis_name=ANALYSIS_NAME, save_all=True,
                   verbose=True):
    swr_io.start_log(os.path.join(swr_io.session_deriv_dir(int(session), swr_io.get_data_root()), "LFP-ripples", analysis_name), "swr_detect_session")
    session = int(session)
    data_root = swr_io.get_data_root()
    clean_dir = os.path.join(swr_io.session_deriv_dir(session, data_root),
                             "LFP-clean", analysis_name)
    sig_p = os.path.join(clean_dir, "continuous.npy")
    if not os.path.isfile(sig_p):
        print(f"s{session:02d}: no continuous.npy -- run swr_extract_continuous first")
        return None

    sig = np.load(sig_p, mmap_mode='r')
    pairs = pd.read_csv(os.path.join(clean_dir, "pairs.csv"))
    with open(os.path.join(clean_dir, "meta.json")) as f:
        meta = json.load(f)
    fs = float(meta["fs"])
    total_s = sig.shape[1] / fs

    print(f"\ns{session:02d}: {sig.shape[0]} pairs, {total_s:.0f}s @ {fs:.0f}Hz")

    all_ev, qc, diags, iv_rows = [], [], {}, []
    for i, (_, p) in enumerate(pairs.iterrows()):
        x = np.asarray(sig[i], float)

        bad, astats = art.artifact_mask(x, fs)
        clean = ~bad
        contam = float(bad.mean())
        excluded = contam > art.MAX_CONTAM_FRAC

        iv = art.clean_intervals(bad, fs)
        clean_s = float(np.diff(iv, axis=1).sum()) if len(iv) else 0.0

        row = {"session": session, "pair_id": p.pair_id,
               "pair_roi": p.get("pair_roi_native"),
               "hemisphere": p.get("hemisphere"),
               "contaminated_frac": round(contam, 4),
               "clean_s": round(clean_s, 1),
               "excluded": excluded,
               **{f"frac_{k}": round(v, 4) for k, v in astats.items()}}

        if excluded:
            row.update({"n_events": 0, "rate_hz": np.nan})
            qc.append(row)
            if verbose:
                print(f"  {p.pair_id:24s} EXCLUDED ({contam:.0%} contaminated)")
            continue

        ev, diag = det.detect_channel(x, fs, clean)
        diags[p.pair_id] = diag

        n_pass = int(ev.passed.sum()) if len(ev) else 0
        rate = n_pass / clean_s if clean_s > 0 else np.nan
        row.update({"n_candidates": diag.get("n_candidates", 0),
                    "n_events": n_pass,
                    "rate_hz": round(rate, 4) if np.isfinite(rate) else np.nan})
        qc.append(row)

        if len(ev):
            ev.insert(0, "session", session)
            ev.insert(1, "pair_id", p.pair_id)
            ev["pair_roi"] = p.get("pair_roi_native")
            ev["hemisphere"] = p.get("hemisphere")
            ev["subject_label"] = p.get("subject_label")
            all_ev.append(ev)
        for a, b in iv:
            iv_rows.append({"pair_id": p.pair_id, "start_s": a, "stop_s": b})

        if verbose:
            print(f"  {p.pair_id:24s} clean={clean_s:7.0f}s "
                  f"cand={diag.get('n_candidates',0):5d} "
                  f"pass={n_pass:5d}  rate={rate:.3f} Hz")

    events = pd.concat(all_ev, ignore_index=True) if all_ev else pd.DataFrame()
    qc_df = pd.DataFrame(qc)

    if len(events):
        print(f"\n  total {int(events.passed.sum())} ripples "
              f"({len(events)} candidates) across {qc_df.excluded.eq(False).sum()} pairs")
        good = qc_df[~qc_df.excluded & qc_df.rate_hz.notna()]
        if len(good):
            print(f"  rate: median {good.rate_hz.median():.3f} Hz "
                  f"(range {good.rate_hz.min():.3f}-{good.rate_hz.max():.3f}); "
                  f"Chen reference ~0.17-0.24 Hz")
            rs = 1 - events.spectral_passed_strict.mean()
            rr = 1 - events.spectral_passed_relaxed.mean()
            print(f"  spectral rejection: strict {rs:.1%} (PRIMARY) | "
                  f"relaxed {rr:.1%}   [Chen: 23.4% +- 9.9%]")
            n_s = int(events.passed_strict.sum()); n_r = int(events.passed_relaxed.sum())
            cs = float(qc_df.loc[~qc_df.excluded, "clean_s"].sum())
            if cs > 0:
                print(f"  pooled rate: strict {n_s/cs:.3f} Hz | relaxed {n_r/cs:.3f} Hz")

    rep = swr_report.InclusionReport(
        "detection", analysis_name,
        f"s{session:02d}: bipolar derivations entering ripple detection.")
    for _, r in qc_df.iterrows():
        u = f"s{session:02d}/{r.pair_id}"
        if bool(r.excluded):
            rep.exclude(u, f">2/3 of the recording contaminated "
                           f"({r.contaminated_frac:.0%})", roi=r.pair_roi)
        elif float(r.get("clean_s", 0)) <= 0:
            rep.exclude(u, "no artifact-free time", roi=r.pair_roi)
        else:
            rep.include(u, "", roi=r.pair_roi, n_events=int(r.get("n_events", 0)),
                        clean_s=r.clean_s, rate_hz=r.get("rate_hz"))

    if save_all:
        out_dir = os.path.join(swr_io.session_deriv_dir(session, data_root),
                               "LFP-ripples", analysis_name)
        os.makedirs(out_dir, exist_ok=True)
        rep.write(out_dir)
        if len(events):
            events.to_csv(os.path.join(out_dir, "ripple_events.csv"), index=False)
        qc_df.to_csv(os.path.join(out_dir, "channel_qc.csv"), index=False)
        pd.DataFrame(iv_rows).to_csv(
            os.path.join(out_dir, "clean_intervals.csv"), index=False)
        with open(os.path.join(out_dir, "detector_diag.json"), "w") as f:
            json.dump(diags, f, indent=2, default=str)
        swr_io.write_settings(out_dir, _settings_dict(session, analysis_name))
        print(f"  saved -> {out_dir}")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(detect_session)
    else:
        detect_session(38)
