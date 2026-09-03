#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milestone 0 of the SWR pipeline: audit every session BEFORE any signal
processing.

Reconciles three sources that are known to disagree:
    1. config_human_ABCD_iEEG.yaml  (recording site, segment, blocks)
    2. all_trial_times_{XX}.csv     (the behavioural clock)
    3. the raw files actually on disk

and writes a session manifest that becomes the authority for everything
downstream. The YAML is demoted to a hint.

Why this runs first
-------------------
A large minority of sessions have a config that does not describe the data:
missing `blocks` keys, `segment: null`, block counts that disagree with the
behaviour. And 25 sessions are multi-block, where the behavioural clock runs
continuously across recording gaps of +7 s to +2910 s that the LFP files do
not contain. Getting the block structure wrong misassigns every ripple in
block 2+ to the wrong behavioural window -- systematically, so it looks like
a null result rather than like noise.

Nothing downstream is worth building until the sessions flagged here are
resolved by hand.

Usage
-----
    conda activate env_multiple_clocks
    python scripts/swr_audit_sessions.py
    python scripts/swr_audit_sessions.py --sessions="[5,7,18]" --verbose=True

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
import mc.analyse.swr_report as swr_report
import mc.analyse.swr_preproc as pre

try:
    import fire
except ImportError:                                   # cluster envs vary
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_audit"

# A block is only usable if the behavioural events can be placed inside the
# recording with this much margin at each end.
CLOCK_MARGIN_S = 5.0

# A behavioural clock can overrun the end of the recording for two completely
# different reasons, and only one of them is a problem:
#
#   truncation   the recording was stopped a second or two before the task
#                ended, so the last event or two has no data. Harmless: such a
#                window gets zero artifact-free exposure and is dropped by the
#                GLM automatically (verified: clean_exposure returns 0.0 past
#                the last interval, and fit_count_glm filters exposure_s > 0).
#   mis-mapping  the block structure itself is wrong, so events are assigned to
#                the wrong stretch of recording. Corrupts everything downstream.
#
# The two are told apart by magnitude: a few seconds is a stopped recording,
# minutes means the mapping is wrong.
TRUNCATION_TOL_S = 5.0


def _settings_dict(sessions):
    return {
        "analysis_name": ANALYSIS_NAME,
        "sessions": list(sessions),
        "clock_margin_s": CLOCK_MARGIN_S,
        "beh_cols": swr_io.BEH_COLS,
        "exclude_attempts": swr_io.EPHYS_EXCLUDE_ATTEMPTS,
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def audit_one_session(session, cfg, subj_map, data_root, check_clock=True):
    """Return (summary_row, block_rows, detail_dict) for one session."""
    session = int(session)
    warnings = []

    cfg_s = swr_io.session_config(session, cfg=cfg, data_root=data_root)
    warnings += [f"config: {w}" for w in cfg_s.get('config_warnings', [])]
    site = cfg_s.get('recording_site')

    label, site_from_cells = subj_map.get(session, (None, None))
    subject_key = swr_io.normalise_subject_key(label)
    if site and site_from_cells and site != site_from_cells:
        warnings.append(f"site mismatch: yaml={site} cells={site_from_cells}")

    # --- behaviour -------------------------------------------------------
    beh, blocks_df = None, pd.DataFrame()
    n_beh_blocks = 0
    try:
        beh = swr_io.load_behaviour(session, data_root=data_root)
        warnings += [f"beh: {w}" for w in swr_io.behaviour_warnings(beh)]
        blocks_df = swr_io.block_table(beh)
        n_beh_blocks = len(blocks_df)
    except FileNotFoundError:
        warnings.append("beh: all_trial_times file missing")
    except ValueError as e:
        warnings.append(f"beh: {e}")

    # --- raw files -------------------------------------------------------
    files, kind, fwarn = swr_io.discover_raw_files(session, cfg_s, data_root=data_root)
    warnings += [f"files: {w}" for w in fwarn]
    n_files = len(files)

    cfg_blocks = len(cfg_s.get('blocks', []) or [])
    if n_beh_blocks and n_files and n_beh_blocks != n_files:
        warnings.append(f"block mismatch: behaviour={n_beh_blocks} files={n_files}")
    if cfg_blocks and n_beh_blocks and cfg_blocks != n_beh_blocks:
        warnings.append(f"block mismatch: yaml={cfg_blocks} behaviour={n_beh_blocks}")

    # --- the clock gate --------------------------------------------------
    # The warnings above are mostly bookkeeping: a null `segment`, a missing
    # `blocks` key, a YAML block count that disagrees with the behaviour. None
    # of those can corrupt a result, because the YAML is only a hint (SS3.3).
    # What CAN corrupt a result is the behavioural clock not fitting inside the
    # recording, which is how a multi-block session silently misassigns every
    # event in block 2+. That is checked here by reading the file headers.
    clock = {"clock_status": "not_checked", "min_head_margin_s": np.nan,
             "min_tail_margin_s": np.nan}
    if check_clock and n_files and beh is not None:
        try:
            blocks, _, _ = pre.session_block_table(session, data_root=data_root)
            if len(blocks) and "head_margin_s" in blocks:
                h = blocks["head_margin_s"].astype(float)
                t = blocks["tail_margin_s"].astype(float)
                clock["min_head_margin_s"] = float(np.nanmin(h)) if h.notna().any() else np.nan
                clock["min_tail_margin_s"] = float(np.nanmin(t)) if t.notna().any() else np.nan
                worst = np.nanmin([clock["min_head_margin_s"],
                                   clock["min_tail_margin_s"]])
                if not np.isfinite(worst):
                    clock["clock_status"] = "unknown"
                elif worst < -TRUNCATION_TOL_S:
                    clock["clock_status"] = "FAILED"
                    warnings.append(
                        f"clock: behaviour falls OUTSIDE the recording by "
                        f"{-worst:.1f}s -- block mapping is wrong")
                elif worst < 0:
                    clock["clock_status"] = "truncated"
                    warnings.append(
                        f"clock: recording stops {-worst:.1f}s before the "
                        f"behaviour does; the trailing events have no data and "
                        f"are dropped automatically (zero exposure)")
                elif worst < CLOCK_MARGIN_S and len(blocks) > 1:
                    # The margin criterion exists to validate multi-block
                    # OFFSETS. A single-block session has no offset to get
                    # wrong -- behaviour and LFP share t=0 -- so a small
                    # positive margin there just means the recording ran on
                    # after the task ended, which is normal and not a warning.
                    clock["clock_status"] = "tight"
                    warnings.append(
                        f"clock: only {worst:.1f}s margin across "
                        f"{len(blocks)} blocks (< {CLOCK_MARGIN_S}s)")
                else:
                    clock["clock_status"] = "ok"
            if len(blocks) > n_beh_blocks and n_beh_blocks:
                clock["extra_files"] = int(len(blocks) - n_beh_blocks)
        except Exception as e:
            clock["clock_status"] = f"error: {type(e).__name__}"
            warnings.append(f"clock: could not be checked ({type(e).__name__}: {e})")

    # --- status ----------------------------------------------------------
    if beh is None:
        status = "no_behaviour"
    elif n_files == 0:
        status = "no_raw_files"
    elif clock["clock_status"] == "FAILED":
        status = "clock_failed"
    elif clock["clock_status"] == "truncated":
        status = "needs_review"
    elif warnings:
        status = "needs_review"
    else:
        status = "ok"

    max_gap = float(blocks_df['gap_to_prev_s'].max()) if n_beh_blocks > 1 else np.nan

    row = {
        "session": session,
        "subject_label": label,
        "subject_key": subject_key,
        "recording_site": site,
        "file_format": cfg_s.get('LFP_file_format'),
        "sampling_rate": cfg_s.get('sampling_rate'),
        "reader": kind,
        "n_raw_files": n_files,
        "n_yaml_blocks": cfg_blocks,
        "n_beh_blocks": n_beh_blocks,
        "n_repeats": int(len(beh)) if beh is not None else 0,
        "n_grids": int(beh['grid_no'].nunique()) if beh is not None else 0,
        "beh_duration_s": float(beh['t_D'].max()) if beh is not None else np.nan,
        "max_block_gap_s": max_gap,
        "multi_block": n_beh_blocks > 1,
        "status": status,
        "clock_status": clock["clock_status"],
        "min_head_margin_s": clock["min_head_margin_s"],
        "min_tail_margin_s": clock["min_tail_margin_s"],
        "n_warnings": len(warnings),
        "warnings": "; ".join(warnings),
    }

    block_rows = []
    for _, b in blocks_df.iterrows():
        br = {"session": session, "subject_key": subject_key}
        br.update(b.to_dict())
        br["raw_file"] = (os.path.basename(files[int(b['session_no']) - 1])
                          if kind == 'blackrock' and n_files >= int(b['session_no'])
                          else None)
        block_rows.append(br)

    detail = {
        "session": session,
        "config": {k: v for k, v in cfg_s.items() if k != 'config_warnings'},
        "raw_files": [os.path.basename(f) if isinstance(f, str) else
                      [os.path.basename(x) for x in f][:3] for f in files],
        "warnings": warnings,
        "status": status,
    }
    return row, block_rows, detail


def audit_sessions(sessions=None, save_all=True, verbose=False,
                   return_data=False, check_clock=True):
    """Audit every session and write the manifest.

    sessions    : list of ints, or None for every key in the YAML.
    return_data : return the DataFrames. Off by default so the CLI does not
                  echo two DataFrames over the report.
    """
    swr_io.start_log(os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr"), "swr_audit_sessions")
    data_root = swr_io.get_data_root()
    cfg = swr_io.load_config(data_root)
    subj_map = swr_io.load_session_subject_map(data_root)

    if sessions is None:
        sessions = sorted(int(k) for k in cfg.keys())
    sessions = [int(s) for s in sessions]

    print(f"Auditing {len(sessions)} sessions from {data_root}\n")

    summary, blocks, details = [], [], []
    for s in sessions:
        row, brows, detail = audit_one_session(s, cfg, subj_map, data_root,
                                              check_clock=check_clock)
        summary.append(row)
        blocks.extend(brows)
        details.append(detail)
        if verbose and row['warnings']:
            print(f"  s{s:02d}: {row['warnings']}")

    manifest = pd.DataFrame(summary)
    block_df = pd.DataFrame(blocks)

    # ---- report ---------------------------------------------------------
    print("\n" + "=" * 74)
    print(" SESSION AUDIT")
    print("=" * 74)
    print(manifest['status'].value_counts().to_string())

    print(f"\nSites: {manifest['recording_site'].value_counts().to_dict()}")
    print(f"Multi-block sessions: {int(manifest['multi_block'].sum())}")

    keys = manifest['subject_key'].dropna()
    print(f"Sessions: {len(manifest)} | distinct subject_key: {keys.nunique()}")

    print("\n--- subject_key map (EYEBALL THIS: it is the GLM cluster variable) ---")
    smap = (manifest.dropna(subset=['subject_key'])
                    .groupby('subject_key')['session'].apply(list))
    collapsed = {k: v for k, v in smap.items() if len(v) > 1}
    for k, v in collapsed.items():
        labs = manifest.loc[manifest.subject_key == k, 'subject_label'].unique()
        flag = "  <-- MERGED DIFFERENT LABELS" if len(labs) > 1 else ""
        print(f"  {k:12s} sessions={v}  labels={list(labs)}{flag}")

    if manifest['multi_block'].any():
        print("\n--- multi-block sessions (behavioural gap NOT present in LFP) ---")
        mb = manifest[manifest.multi_block].sort_values('max_block_gap_s', ascending=False)
        for _, r in mb.iterrows():
            print(f"  s{int(r.session):02d}  blocks={int(r.n_beh_blocks)}  "
                  f"max_gap={r.max_block_gap_s:9.1f}s  files={int(r.n_raw_files)}")

    # ---- triage: which warnings actually matter ------------------------
    if "clock_status" in manifest:
        print("\n--- CLOCK CHECK (the only warning that can corrupt a result) ---")
        print(manifest.clock_status.value_counts().to_string())
        bad = manifest[manifest.clock_status.isin(
            ["FAILED", "truncated", "tight", "unknown"])]
        for _, r in bad.iterrows():
            print(f"  s{int(r.session):02d}  {r.clock_status:8s} "
                  f"head {r.min_head_margin_s:8.1f}s  tail {r.min_tail_margin_s:8.1f}s"
                  f"  files={int(r.n_raw_files)} beh_blocks={int(r.n_beh_blocks)}")
        if not len(bad):
            print("  every session's behaviour fits inside its recording "
                  f"with >= {CLOCK_MARGIN_S}s margin at both ends.")
        else:
            print(f"  FAILED    = mapping is wrong (overrun > {TRUNCATION_TOL_S}s). "
                  "Do not analyse.")
            print(f"  truncated = recording stopped < {TRUNCATION_TOL_S}s before the "
                  "task ended. Usable:")
            print("              the trailing windows get zero exposure and are "
                  "dropped by the GLM.")

        cosmetic = manifest[(manifest.status == "needs_review")
                            & (manifest.clock_status == "ok")]
        if len(cosmetic):
            print(f"\n--- {len(cosmetic)} sessions flagged, but the clock is fine ---")
            print("  These are bookkeeping warnings -- a null `segment`, a missing")
            print("  `blocks` key, a YAML block count that disagrees with the")
            print("  behaviour. The YAML is a hint, not the authority (methods SS3.3),")
            print("  so none of these can corrupt a result. Usable as they are:")
            print("    " + ", ".join(f"s{int(x):02d}" for x in sorted(cosmetic.session)))

    flagged = manifest[manifest.status != 'ok']
    if len(flagged):
        print(f"\n--- {len(flagged)} sessions NOT ok (resolve before building) ---")
        # group by the leading part of the reason: 60 identical lines hide the
        # signal, and one systematic problem is very different from 60 separate
        # ones
        flagged = flagged.copy()
        flagged["_key"] = (flagged.status.astype(str) + " | "
                           + flagged.warnings.astype(str).str.slice(0, 70))
        for key, grp in flagged.groupby("_key"):
            sess = ", ".join(f"s{int(x):02d}" for x in sorted(grp.session)[:14])
            more = f" (+{len(grp)-14} more)" if len(grp) > 14 else ""
            print(f"  [{len(grp):2d}] {key}")
            print(f"       {sess}{more}")

    # ---- save -----------------------------------------------------------
    # inclusion/exclusion report -- every session accounted for, with a reason
    rep = swr_report.InclusionReport(
        "audit", ANALYSIS_NAME,
        "Sessions reconciled across config YAML, behaviour and raw files on disk.")
    for _, r in manifest.iterrows():
        u = f"s{int(r.session):02d}"
        if r.status == "ok":
            rep.include(u, "", site=r.recording_site, blocks=r.n_beh_blocks,
                        repeats=r.n_repeats)
        elif r.status == "needs_review":
            rep.include(u, f"usable with caveat: {str(r.warnings)[:90]}",
                        site=r.recording_site)
        else:
            reason = {"no_raw_files": "no raw LFP files on this machine",
                      "no_behaviour": "behaviour file missing or unreadable"
                      }.get(r.status, str(r.status))
            rep.exclude(u, reason, site=r.recording_site)
    rep.note("`no raw LFP files on this machine` is usually a data-location "
             "issue, not a data-quality one: re-run the audit on the cluster.")

    if save_all:
        out_dir = os.path.join(swr_io.derivatives_dir(data_root), "group", "swr")
        rep.write(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        manifest.to_csv(os.path.join(out_dir, "session_manifest.csv"), index=False)
        block_df.to_csv(os.path.join(out_dir, "session_blocks.csv"), index=False)
        with open(os.path.join(out_dir, "session_manifest.json"), 'w') as f:
            json.dump(details, f, indent=2, default=str)
        swr_io.write_settings(out_dir, _settings_dict(sessions))
        print(f"\nSaved manifest to {out_dir}")

    return (manifest, block_df) if return_data else None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(audit_sessions)
    else:
        audit_sessions()
