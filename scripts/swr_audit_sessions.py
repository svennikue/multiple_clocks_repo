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

try:
    import fire
except ImportError:                                   # cluster envs vary
    fire = None

print("ARGS:", sys.argv)

ANALYSIS_NAME = "swr_audit"

# A block is only usable if the behavioural events can be placed inside the
# recording with this much margin at each end.
CLOCK_MARGIN_S = 5.0


def _settings_dict(sessions):
    return {
        "analysis_name": ANALYSIS_NAME,
        "sessions": list(sessions),
        "clock_margin_s": CLOCK_MARGIN_S,
        "beh_cols": swr_io.BEH_COLS,
        "exclude_attempts": swr_io.EPHYS_EXCLUDE_ATTEMPTS,
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def audit_one_session(session, cfg, subj_map, data_root):
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

    # --- status ----------------------------------------------------------
    if beh is None:
        status = "no_behaviour"
    elif n_files == 0:
        status = "no_raw_files"
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


def audit_sessions(sessions=None, save_all=True, verbose=False, return_data=False):
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
        row, brows, detail = audit_one_session(s, cfg, subj_map, data_root)
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

    flagged = manifest[manifest.status != 'ok']
    if len(flagged):
        print(f"\n--- {len(flagged)} sessions NOT ok (resolve before building) ---")
        for _, r in flagged.iterrows():
            print(f"  s{int(r.session):02d}  {r.status:14s}  {r.warnings[:110]}")

    # ---- save -----------------------------------------------------------
    if save_all:
        out_dir = os.path.join(swr_io.derivatives_dir(data_root), "group", "swr")
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
