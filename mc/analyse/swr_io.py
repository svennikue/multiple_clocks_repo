#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I/O foundations for the sharp-wave-ripple (SWR) pipeline.

Path resolution, the behaviour loader, raw-file discovery and the session
manifest. Everything downstream (preprocessing, detection, statistics) reads
its ground truth from here.

Deliberately NOT registered in mc/analyse/__init__.py: that file imports
eagerly, and this module pulls in neo/yaml. Import it explicitly:

    import mc.analyse.swr_io as swr_io

Why this module exists
----------------------
Three defects in the previous pipeline are fixed at this layer, before any
signal is touched:

1. `scripts/identify_HPC_ripples.py:57` and `scripts/preprocess_LFP.py:128`
   assign 13 column names to a 14-column behaviour file, so both raise
   `ValueError: Length mismatch` on every session. The 14th column is
   `correct` and the contract already exists at
   `scripts/behaviour_summary.py:47` and `mc/analyse/helpers_human_cells.py:379`.

2. `config_human_ABCD_iEEG.yaml` disagrees with the behaviour and with the
   files on disk for a large minority of sessions (missing `blocks` keys,
   `segment: null`, block counts that do not match). The YAML is therefore
   demoted to a *hint*; the manifest built here is the authority.

3. The behavioural clock is continuous across recording blocks but the LFP
   files are not -- blocks are separated by real recording gaps of +7 s to
   +2910 s. `preprocess_LFP.py:213` maps behaviour into block 2 by
   subtracting the *file duration*, which is only valid if recording never
   stopped. See `estimate_block_offsets` in swr_preproc.

@author: Svenja Kuchenhoff
"""

import os
import glob
import json
import re

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# PATHS
# =============================================================================

DATA_ROOT_LOCAL = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
DATA_ROOT_CEPH = "/ceph/behrens/svenja/human_ABCD_ephys"

CONFIG_NAME = "config_human_ABCD_iEEG.yaml"


def get_data_root(verbose=True):
    """Resolve the data root, local first then ceph (house convention)."""
    if os.path.isdir(DATA_ROOT_LOCAL):
        return DATA_ROOT_LOCAL
    if verbose:
        print("running on ceph")
    return DATA_ROOT_CEPH


def derivatives_dir(data_root=None):
    return os.path.join(data_root or get_data_root(), "derivatives")


def session_deriv_dir(session, data_root=None):
    return os.path.join(derivatives_dir(data_root), f"s{int(session):02d}")


# =============================================================================
# BEHAVIOUR
# =============================================================================

# Mirrors scripts/behaviour_summary.py:47 (EPHYS_BEH_COLS) and
# mc/analyse/helpers_human_cells.py:379. Kept here so library code does not
# have to import from a script. If these ever diverge, behaviour_summary.py
# is the historical source -- reconcile, do not fork.
BEH_COLS = ['rep_correct', 't_A', 't_B', 't_C', 't_D',
            'loc_A', 'loc_B', 'loc_C', 'loc_D', 'rep_overall',
            'new_grid_onset', 'session_no', 'grid_no', 'correct']

STATE_TIME_COLS = ['t_A', 't_B', 't_C', 't_D']

# Mirrors scripts/behaviour_summary.py:52. A single 314.58 s repeat that
# looks like a recording interruption; the rest of s23 is retained.
EPHYS_EXCLUDE_ATTEMPTS = [
    {'session': 23, 'session_no': 1, 'grid_no': 3, 'rep_correct': 7},
]


def load_behaviour(session, data_root=None, apply_exclusions=True):
    """Load all_trial_times_{XX}.csv with the correct 14-column contract.

    Returns a DataFrame sorted by (session_no, new_grid_onset) with two
    added columns:
        `plan_known`  -- True from the first correct solve of the grid onward.
                         Derived as a cumulative max of `correct` within grid,
                         NOT read off `correct` directly: `correct` is
                         per-repeat accuracy, so a later error is an error,
                         not a forgotten plan.
        `excluded`    -- flagged by EPHYS_EXCLUDE_ATTEMPTS.

    Raises ValueError on an unexpected column count rather than silently
    mislabelling, which is what the previous loaders did.
    """
    session = int(session)
    path = os.path.join(session_deriv_dir(session, data_root),
                        "cells_and_beh", f"all_trial_times_{session:02d}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    beh = pd.read_csv(path, header=None)
    if beh.shape[1] != len(BEH_COLS):
        raise ValueError(
            f"s{session:02d}: behaviour file has {beh.shape[1]} columns, "
            f"expected {len(BEH_COLS)} ({BEH_COLS}). Path: {path}")
    beh.columns = BEH_COLS

    beh['session'] = session
    beh = beh.sort_values(['session_no', 'new_grid_onset']).reset_index(drop=True)

    # plan_known: cumulative-max of `correct` within each (block, grid).
    beh['plan_known'] = (
        beh.groupby(['session_no', 'grid_no'])['correct']
           .transform(lambda s: s.astype(int).cummax()).astype(bool)
    )

    beh['excluded'] = False
    if apply_exclusions:
        for rule in EPHYS_EXCLUDE_ATTEMPTS:
            if rule['session'] != session:
                continue
            hit = ((beh['session_no'] == rule['session_no'])
                   & (beh['grid_no'] == rule['grid_no'])
                   & (beh['rep_correct'] == rule['rep_correct']))
            beh.loc[hit, 'excluded'] = True

    return beh


def behaviour_warnings(beh):
    """Structural problems worth surfacing per session. Returns a list of str."""
    warn = []

    # State times must increase within a repeat.
    t = beh[STATE_TIME_COLS].to_numpy(dtype=float)
    bad_within = int(np.sum(np.any(np.diff(t, axis=1) <= 0, axis=1)))
    if bad_within:
        warn.append(f"{bad_within} repeats with non-increasing t_A..t_D")

    # t_D must increase across repeats within a block.
    for blk, sub in beh.groupby('session_no'):
        td = sub['t_D'].to_numpy(dtype=float)
        if np.any(np.diff(td) <= 0):
            warn.append(f"block {int(blk)}: t_D non-monotone across repeats")

    # new_grid_onset should precede t_A.
    if np.any(beh['new_grid_onset'].to_numpy(float)
              >= beh['t_A'].to_numpy(float)):
        warn.append("new_grid_onset >= t_A for at least one repeat")

    return warn


def block_table(beh):
    """One row per behavioural block: extent on the session clock and the
    gap to the previous block.

    The gap is the crux of the whole timing problem: it is real elapsed time
    during which the amplifier was stopped, so it exists in the behavioural
    clock but NOT in the concatenated LFP files.
    """
    rows = []
    prev_end = None
    for blk, sub in beh.groupby('session_no'):
        start = float(sub['new_grid_onset'].min())
        end = float(sub['t_D'].max())
        rows.append({
            'session_no': int(blk),
            'beh_start_s': start,
            'beh_end_s': end,
            'beh_duration_s': end - start,
            'gap_to_prev_s': np.nan if prev_end is None else start - prev_end,
            'n_repeats': int(len(sub)),
            'n_grids': int(sub['grid_no'].nunique()),
        })
        prev_end = end
    return pd.DataFrame(rows)


# =============================================================================
# CONFIG (a hint, not the authority)
# =============================================================================

def load_config(data_root=None):
    path = os.path.join(data_root or get_data_root(), CONFIG_NAME)
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def session_config(session, cfg=None, data_root=None):
    """Per-session config with the known defects normalised.

    Guards every defect found in the audit: missing `blocks`, `segment: null`,
    and segment/blocks length mismatches. Returns a dict that always has
    `recording_site`, `sampling_rate`, `LFP_file_format`, `segment` (list) and
    `blocks` (list), plus `config_warnings`.
    """
    cfg = cfg if cfg is not None else load_config(data_root)
    key = f"{int(session):02d}"
    raw = cfg.get(key)
    if raw is None:
        return {'recording_site': None, 'config_warnings': [f"no config entry for '{key}'"]}

    out = dict(raw)
    warn = []

    if out.get('segment') is None:
        out['segment'] = []
        warn.append("segment is null")
    elif not isinstance(out['segment'], list):
        out['segment'] = [out['segment']]

    if 'blocks' not in raw or raw.get('blocks') is None:
        out['blocks'] = []
        warn.append("no 'blocks' key")
    elif not isinstance(out['blocks'], list):
        out['blocks'] = [out['blocks']]

    if out['segment'] and out['blocks'] and len(out['segment']) != len(out['blocks']):
        warn.append(f"len(segment)={len(out['segment'])} != len(blocks)={len(out['blocks'])}")

    # NOTE: duplicate entries in `blocks` (s18 ['blk-01','blk-01','blk-02'],
    # s28 ['blk-01','blk-02','blk-02']) are NOT a defect. They are filename
    # labels, and the files behind them are distinct sequential recordings.
    # `_sort_blackrock` orders by EMU number, which is the real chronological
    # key, and block diagnosis confirms all three files of s18 and s28 match
    # their behavioural blocks by duration. Recorded as info, not a warning.
    out['duplicate_block_labels'] = bool(
        out['blocks'] and len(set(out['blocks'])) != len(out['blocks']))

    out['config_warnings'] = warn
    return out


# =============================================================================
# RAW FILE DISCOVERY
# =============================================================================

_EMU_RE = re.compile(r'EMU-(\d+)')
_BLK_RE = re.compile(r'blk-(\d+)')
_NCS_BLOCK_RE = re.compile(r'_(\d{4})\.ncs$')

# Neuralynx .ncs: 16 KB ASCII header, then 1044-byte records.
NCS_HEADER_BYTES = 16 * 1024
NCS_RECORD_BYTES = 1044


def _sort_blackrock(files):
    """Order Blackrock files by EMU number, falling back to blk- then name.

    EMU number is the acquisition counter, so it is the only reliable
    chronological key; blk- labels are duplicated in some sessions
    (s18 has ['blk-01','blk-01','blk-02']).
    """
    def key(f):
        base = os.path.basename(f)
        emu = _EMU_RE.search(base)
        blk = _BLK_RE.search(base)
        return (int(emu.group(1)) if emu else 10**9,
                int(blk.group(1)) if blk else 10**9,
                base)
    return sorted(files, key=key)


def discover_raw_files(session, cfg_sesh, data_root=None):
    """Find the raw LFP files for a session, in chronological order.

    Returns (files, kind, warnings). `kind` is 'blackrock' or 'neuralynx'.

    UCLA stores one .ncs per contact, and a second recording block appears as
    `{stem}_0001.ncs`. That block split is not represented in the YAML at all,
    so it is discovered here.
    """
    root = data_root or get_data_root()
    sdir = os.path.join(root, f"s{int(session):02d}")
    warn = []

    site = (cfg_sesh or {}).get('recording_site')
    fmt = (cfg_sesh or {}).get('LFP_file_format')

    if site == 'ucla' or fmt == 'ncs':
        pats = [os.path.join(sdir, '**', '*.ncs')]
        found = sorted({f for p in pats for f in glob.glob(p, recursive=True)})
        if not found:
            return [], 'neuralynx', ["no .ncs files found"]

        # A .ncs file is a 16 KB ASCII header followed by 1044-byte records.
        # Files of exactly NCS_HEADER_BYTES contain a header and no data --
        # aborted recordings ("TimeClosed File was not closed properly").
        # s03 has 164 such stubs alongside 144 real files; counting the stubs
        # as a recording block is what made the YAML look inconsistent.
        files = [f for f in found if os.path.getsize(f) > NCS_HEADER_BYTES]
        n_empty = len(found) - len(files)
        if n_empty:
            warn.append(f"{n_empty}/{len(found)} .ncs files are header-only "
                        f"stubs (no data) and were skipped")
        if not files:
            return [], 'neuralynx', warn + ["all .ncs files are header-only"]

        # Group by trailing _NNNN block index; bare stem is block 0.
        blocks = {}
        for f in files:
            m = _NCS_BLOCK_RE.search(os.path.basename(f))
            blocks.setdefault(int(m.group(1)) if m else 0, []).append(f)
        if len(blocks) > 1:
            warn.append(f"{len(blocks)} .ncs recording blocks with data "
                        f"(suffixes {sorted(blocks)}) not represented in YAML")
        return [sorted(blocks[k]) for k in sorted(blocks)], 'neuralynx', warn

    ext = f"ns{fmt}" if fmt in (2, 3, '2', '3') else "ns*"
    pats = [os.path.join(sdir, f"*.{ext}"),
            os.path.join(sdir, 'LFP', f"*.{ext}")]
    files = sorted({f for p in pats for f in glob.glob(p)})
    if not files:
        return [], 'blackrock', [f"no *.{ext} files found under {sdir}"]
    return _sort_blackrock(files), 'blackrock', warn


# =============================================================================
# SUBJECT KEY
# =============================================================================

_CELL_TABLE = "neurons_with_ROI_labels.csv"


def load_session_subject_map(data_root=None):
    """session (int) -> raw 'Subject Label' from the curated cell table."""
    path = os.path.join(derivatives_dir(data_root), _CELL_TABLE)
    if not os.path.isfile(path):
        return {}
    d = pd.read_csv(path, usecols=['subject', 'Subject Label', 'Recording Site'])
    d = d.drop_duplicates('subject')
    return {
        int(row['subject']): (str(row['Subject Label']).strip().strip("'"),
                              str(row['Recording Site']).strip().lower())
        for _, row in d.iterrows()
    }


def normalise_subject_key(subject_label):
    """Collapse the four Utah label formats onto one key per patient.

    s29 is 'UT1-202314' and s30 is 'UT202314' -- the same person. Clustering
    robust SEs on the raw label would give 43 clusters where the truth is 42,
    which is anticonservative. Baylor 'BY2-YEK' -> 'YEK'; UCLA 'UC3-0559' ->
    'UC0559'.
    """
    if subject_label is None or (isinstance(subject_label, float) and np.isnan(subject_label)):
        return None
    lab = str(subject_label).strip().strip("'")

    m = re.search(r'(\d{6}[a-zA-Z]?)\s*$', lab)          # Utah numeric patient id
    if m and lab.upper().startswith('UT'):
        return f"UT{m.group(1)}"

    m = re.search(r'^BY\d*[-_]?([A-Z]{3})$', lab, re.I)   # Baylor 3-letter code
    if m:
        return m.group(1).upper()

    m = re.search(r'^UC\d*[-_]?(\d+)$', lab, re.I)        # UCLA numeric
    if m:
        return f"UC{int(m.group(1)):04d}"

    return re.sub(r'[^A-Za-z0-9]', '', lab).upper()


# =============================================================================
# SETTINGS SNAPSHOT
# =============================================================================

def write_settings(out_dir, settings):
    """House convention: a settings.json beside every result
    (cf. scripts/spatial_peaks_simple.py:1374)."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=2, default=str)


# =============================================================================
# LOGGING
# =============================================================================

class _Tee:
    """Duplicate a stream to a file, so console output is preserved verbatim."""

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, s):
        self._stream.write(s)
        self._fh.write(s)
        self._fh.flush()          # cheap, and survives a killed job
        return len(s)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


def start_log(out_dir, name, argv=None):
    """Tee stdout and stderr to `out_dir/logs/{name}_{timestamp}.log`.

    Every script writes one, so a run can be reconstructed after the fact --
    which matters most for cluster array jobs, where the console is gone and
    the SLURM .out files are scattered under logs/<timestamp>/.

    Returns the log path. Safe to call when out_dir is not writable: falls
    back to no logging rather than killing the run.
    """
    import sys as _sys
    from datetime import datetime as _dt
    try:
        ldir = os.path.join(out_dir, "logs")
        os.makedirs(ldir, exist_ok=True)
        stamp = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(ldir, f"{name}_{stamp}.log")
        fh = open(path, "a")
        fh.write(f"# {name}\n# started {_dt.now().isoformat(timespec='seconds')}\n")
        fh.write(f"# argv: {' '.join(argv or _sys.argv)}\n")
        fh.write(f"# cwd: {os.getcwd()}\n\n")
        _sys.stdout = _Tee(_sys.stdout, fh)
        _sys.stderr = _Tee(_sys.stderr, fh)
        print(f"[log] {path}")
        return path
    except Exception as e:
        print(f"[log] could not open log file ({type(e).__name__}: {e}); continuing")
        return None
