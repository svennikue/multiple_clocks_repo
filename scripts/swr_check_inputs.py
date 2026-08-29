#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preflight: does this machine have everything the SWR pipeline needs?

Run this FIRST on any new machine (especially the cluster). It reports, per
stage, what is present and what is missing, and tells you what to copy.

Usage:
    python scripts/swr_check_inputs.py
    python scripts/swr_check_inputs.py --sessions="[1,2,3]"

@author: Svenja Kuchenhoff
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

OK, MISS, WARN = "  OK  ", " MISS ", " WARN "


def _p(status, label, detail=""):
    print(f"[{status}] {label}" + (f"   {detail}" if detail else ""))


def check(sessions=None):
    R = swr_io.get_data_root()
    print(f"\ndata root: {R}\n" + "=" * 74)
    problems, warnings = [], []

    # ---- python packages -------------------------------------------------
    print("\n-- packages --")
    for mod, needed_for in [("numpy", "everything"), ("scipy", "everything"),
                            ("pandas", "everything"), ("neo", "Blackrock raw"),
                            ("yaml", "config"), ("statsmodels", "stage 7 GLM"),
                            ("matplotlib", "figures"), ("fire", "CLI args")]:
        try:
            __import__(mod)
            _p(OK, mod)
        except ImportError:
            _p(MISS, mod, f"needed for {needed_for}")
            problems.append(f"pip install {mod}")
    for mod, why in [("fooof", "stage 5 surrogate (falls back to log-log fit)"),
                     ("nilearn", "REQUIRED - hippocampal contacts are chosen from MNI coords"),
                     ("h5py", "v7.3 .mat files")]:
        try:
            __import__(mod); _p(OK, mod, f"({why})")
        except ImportError:
            _p(WARN, mod, f"optional: {why}")

    # `fire` missing is dangerous, not merely inconvenient: every script falls
    # back to running its entry function with DEFAULTS, so an array job would
    # silently process the same default session in every task.
    try:
        import fire as _f
    except ImportError:
        problems.append("!! fire missing: --session=N would be IGNORED by array jobs")

    # ---- shared inputs ---------------------------------------------------
    print("\n-- shared inputs --")
    cfg_p = os.path.join(R, swr_io.CONFIG_NAME)
    _p(OK if os.path.isfile(cfg_p) else MISS, swr_io.CONFIG_NAME)
    if not os.path.isfile(cfg_p):
        problems.append(f"copy {swr_io.CONFIG_NAME} to {R}/")

    v2026 = os.path.join(R, "ABCD_pts_elecFilesForSvenja_v2026")
    n_bay = len(glob.glob(os.path.join(v2026, "*-electrodes_v2026.csv")))
    n_ucla = len(glob.glob(os.path.join(v2026, "*_localizations.xlsx")))
    if n_bay:
        _p(OK, "ABCD_pts_elecFilesForSvenja_v2026",
           f"{n_bay} baylor CSVs, {n_ucla} ucla xlsx")
    else:
        _p(MISS, "ABCD_pts_elecFilesForSvenja_v2026", "needed for stage 1 (~1.7 MB)")
        problems.append(f"rsync ABCD_pts_elecFilesForSvenja_v2026/ -> {R}/")

    cells = os.path.join(swr_io.derivatives_dir(R), "neurons_with_ROI_labels.csv")
    _p(OK if os.path.isfile(cells) else MISS, "derivatives/neurons_with_ROI_labels.csv",
       "subject_key mapping")
    if not os.path.isfile(cells):
        problems.append("copy derivatives/neurons_with_ROI_labels.csv (subject_key)")

    mni = os.path.join(swr_io.derivatives_dir(R), "neurons_MNI_latest.csv")
    if not os.path.isfile(mni):
        _p(WARN, "derivatives/neurons_MNI_latest.csv",
           "only needed to auto-discover Utah .mat files")
        warnings.append("neurons_MNI_latest.csv absent -> Utah discovery may fail")
    else:
        _p(OK, "derivatives/neurons_MNI_latest.csv")

    # ---- per-session -----------------------------------------------------
    print("\n-- per session --")
    try:
        cfg = swr_io.load_config(R)
        sess_list = sorted(int(k) for k in cfg.keys())
    except Exception as e:
        print(f"cannot read config: {e}"); return
    if sessions is not None:
        sess_list = [int(s) for s in sessions]

    rows = []
    for s in sess_list:
        cs = swr_io.session_config(s, cfg=cfg, data_root=R)
        site = cs.get("recording_site")
        files, kind, _ = swr_io.discover_raw_files(s, cs, data_root=R)
        beh_ok = os.path.isfile(os.path.join(
            swr_io.session_deriv_dir(s, R), "cells_and_beh",
            f"all_trial_times_{s:02d}.csv"))
        pairs_ok = os.path.isfile(os.path.join(
            swr_io.session_deriv_dir(s, R), "LFP", f"bipolar_pairs_{s:02d}.csv"))
        utah_mat = ""
        if site == "utah":
            for n in ("Electrodes.mat", "ChannelMap.mat"):
                if os.path.isfile(os.path.join(R, f"s{s:02d}", "electrodes", n)):
                    utah_mat = n; break
        rows.append({"session": s, "site": site, "raw": len(files),
                     "beh": beh_ok, "pairs": pairs_ok,
                     "utah_mat": utah_mat or ("-" if site != "utah" else "MISSING")})
    d = pd.DataFrame(rows)

    print(f"  sessions in config      : {len(d)}")
    print(f"  with raw LFP files      : {int((d.raw > 0).sum())}")
    print(f"  with behaviour          : {int(d.beh.sum())}")
    print(f"  with bipolar_pairs (st1): {int(d.pairs.sum())}")
    print(f"  READY for stage 2       : {int(((d.raw > 0) & d.beh & d.pairs).sum())}")
    print(f"  ready for stage 1 only  : {int(((d.raw > 0) & d.beh & ~d.pairs).sum())}"
          f"   (need swr_build_contacts)")

    utah_missing = d[(d.site == "utah") & (d.utah_mat == "MISSING") & (d.raw > 0)]
    if len(utah_missing):
        _p(WARN, f"{len(utah_missing)} utah sessions without electrodes/*.mat",
           f"{list(utah_missing.session)[:8]}")
        warnings.append("utah sessions missing electrodes/*.mat -> no contacts for them")

    no_beh = d[(d.raw > 0) & ~d.beh]
    if len(no_beh):
        _p(WARN, f"{len(no_beh)} sessions with raw but no behaviour",
           f"{list(no_beh.session)[:8]}")

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 74)
    if problems:
        print(" BLOCKERS")
        for x in problems:
            print(f"   - {x}")
    else:
        print(" No blockers. You can run the pipeline from stage 0.")
    if warnings:
        print("\n NON-BLOCKING")
        for x in warnings:
            print(f"   - {x}")

    # Contact selection is coordinate-based since 2026-08-29, so the atlases are
    # a hard requirement rather than a cross-check.
    print("\n atlas check (REQUIRED - contacts are selected from MNI coords):")
    try:
        import mc.analyse.anatomy_atlas as _aa
        import numpy as _np
        p = _aa.hippocampal_probability(_np.array([[-26.0, -22.0, -14.0]]))[0]
        if p > 50:
            print(f"   OK   HarvardOxford sub-prob-2mm reachable "
                  f"(P(hippocampus) at a canonical HC voxel = {p:.0f}%)")
        else:
            print(f"   FAIL sub-prob-2mm returned {p:.0f}% at a canonical HC voxel - "
                  "wrong volume index or wrong atlas")
    except Exception as _e:
        print(f"   FAIL {type(_e).__name__}: {_e}")
        print("        rsync ~/nilearn_data/fsl to the cluster and set NILEARN_DATA,")
        print("        or the contact stage will select zero hippocampal contacts.")

    out = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
    try:
        os.makedirs(out, exist_ok=True)
        d.to_csv(os.path.join(out, "input_check.csv"), index=False)
        print(f"\n saved -> {out}/input_check.csv")
    except Exception:
        pass
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(check)
    else:
        check()
