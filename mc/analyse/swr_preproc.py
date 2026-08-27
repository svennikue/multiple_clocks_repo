#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous LFP preprocessing for the SWR pipeline.

Loads a whole session's hippocampal bipolar derivations onto the behavioural
clock: raw -> 1000 Hz -> bipolar -> notch, with the block structure resolved
from the recordings themselves.

Three decisions here differ from the previous pipeline, each for a measured
reason:

1. **Continuous, not per-snippet.** The detection threshold has to be estimated
   over the whole artifact-free session (see swr_detect). Cropping first makes
   the threshold snippet-dependent, which normalises away the very rate
   differences the analysis is about.

2. **`resample_poly`, not `resample`.** `scipy.signal.resample` is FFT-based and
   assumes periodicity, so it wraps the end of the recording into the
   beginning. Applied per snippet, as `preprocess_LFP.py:106` did, it rings at
   every crop boundary. Polyphase decimation uses a proper anti-alias FIR and
   does not wrap.

3. **1000 Hz, not 500.** 500 Hz makes Chen's >250 Hz artifact criterion and the
   120-200 Hz spectral rejection impossible.

Block alignment
---------------
The behavioural clock is **cumulative file duration**: block *k* starts at
`sum(durations of files 0..k-1)`. Verified on all 14 multi-block sessions with
local raw data -- every behavioural event lands inside its file.

Do NOT align by wall clock. The NSx header `TimeOrigin` is parseable, but it
records real elapsed time between recordings (for s18: 1943 s and 12459 s apart
versus file durations of 560 s and 4040 s). The behaviour was timestamped
against the concatenated recording, so wall-clock anchoring would displace
block 2+ by tens of minutes.

Because the clock is the concatenation, the resampled blocks laid end to end
ARE the session timeline: `sample = round(t_session * fs)`.

@author: Svenja Kuchenhoff
"""

import os
import struct
import datetime

import numpy as np
from scipy.signal import resample_poly, iirnotch, sosfiltfilt, tf2sos

import mc.analyse.swr_io as swr_io


TARGET_FS = 1000.0
LINE_FREQS = (60.0, 120.0, 180.0)      # US sites; 120 sits on the ripple band edge
NOTCH_Q = 30.0
CHUNK_S = 600.0                        # per-block read chunk
CHUNK_PAD_S = 5.0                      # discarded either side to avoid edge effects


# =============================================================================
# NEURALYNX .ncs  (neo cannot read these files)
# =============================================================================
#
# neo's NeuralynxRawIO.parse_header() raises
#   TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'
# at neuralynxrawio.py:505 because global_t_start/global_t_stop are None.
# The format is simple enough to read directly, and doing so also exposes the
# per-record microsecond timestamps.

NCS_HEADER_BYTES = 16 * 1024
NCS_REC = np.dtype([('ts', '<u8'), ('ch', '<u4'), ('fs', '<u4'),
                    ('nvalid', '<u4'), ('samp', '<i2', 512)])


def ncs_header(path):
    """Parse the 16 KB ASCII header into a dict."""
    with open(path, 'rb') as f:
        raw = f.read(NCS_HEADER_BYTES).decode('latin-1', errors='replace')
    out = {}
    for line in raw.split('\n'):
        line = line.strip().strip('\x00').strip()
        if line.startswith('-'):
            parts = line[1:].split(None, 1)
            out[parts[0]] = parts[1].strip() if len(parts) > 1 else ''
    return out


def read_ncs(path, t0_s=None, t1_s=None):
    """Return (signal float32, fs, t_start_s). Optionally crop by seconds
    relative to the file's own first timestamp."""
    size = os.path.getsize(path)
    n_rec = (size - NCS_HEADER_BYTES) // NCS_REC.itemsize
    if n_rec <= 0:
        return np.zeros(0, np.float32), np.nan, np.nan

    recs = np.memmap(path, dtype=NCS_REC, mode='r',
                     offset=NCS_HEADER_BYTES, shape=(int(n_rec),))
    fs = float(recs['fs'][0])
    t_start = float(recs['ts'][0]) / 1e6

    lo, hi = 0, int(n_rec)
    if t0_s is not None:
        lo = max(0, int(np.floor(t0_s * fs / 512.0)))
    if t1_s is not None:
        hi = min(int(n_rec), int(np.ceil(t1_s * fs / 512.0)) + 1)
    if hi <= lo:
        return np.zeros(0, np.float32), fs, t_start

    block = recs[lo:hi]
    sig = block['samp'].reshape(-1).astype(np.float32)

    hdr = ncs_header(path)
    try:                                    # convert AD units to microvolts
        sig *= float(hdr.get('ADBitVolts', 1.0)) * 1e6
    except (TypeError, ValueError):
        pass
    return sig, fs, t_start + lo * 512.0 / fs


def nsx_time_origin(path):
    """NSx 2.2/2.3 TimeOrigin (Windows SYSTEMTIME at byte 294). Returns None
    for the BRSMPGRP variant. Recorded for provenance only -- it must NOT be
    used for block alignment (see module docstring)."""
    with open(path, 'rb') as f:
        head = f.read(320)
    if head[:8] != b'NEURALCD':
        return None
    y, mo, _dw, d, h, mi, s, ms = struct.unpack('<8H', head[294:310])
    try:
        return datetime.datetime(y, mo, d, h, mi, s, ms * 1000)
    except ValueError:
        return None


# =============================================================================
# BLOCK STRUCTURE
# =============================================================================

def blackrock_block_info(path, nsx):
    """(duration_s, fs, n_channels, seg_index) of the real recording segment.

    Blackrock files carry a ~2 s stub segment alongside the recording; the
    longest segment is the real one. Picking it independently reproduces the
    YAML `segment` field wherever that field is populated.
    """
    import neo
    reader = neo.io.BlackrockIO(filename=path, nsx_to_load=int(nsx))
    best = (0.0, np.nan, 0, None)
    for bi in range(reader.block_count()):
        for si in range(reader.segment_count(bi)):
            try:
                seg = reader.read_segment(block_index=bi, seg_index=si, lazy=True)
            except Exception:
                continue
            for a in seg.analogsignals:
                fs = float(a.sampling_rate.magnitude)
                dur = float(a.shape[0]) / fs
                if dur > best[0]:
                    best = (dur, fs, int(a.shape[1]), si)
    return best


def session_block_table(session, data_root=None):
    """Per-block file, duration, sampling rate and session-clock offset.

    offset_k = cumulative duration of files 0..k-1 (see module docstring).
    Adds `beh_start_s` / `beh_end_s` and the head/tail margin so the caller can
    assert every behavioural event lands inside its recording.
    """
    import pandas as pd
    data_root = data_root or swr_io.get_data_root()
    cfg = swr_io.session_config(session, data_root=data_root)
    files, kind, warn = swr_io.discover_raw_files(session, cfg, data_root=data_root)
    beh = swr_io.load_behaviour(session, data_root=data_root)
    bt = swr_io.block_table(beh)

    rows = []
    for k, f in enumerate(files):
        if kind == 'blackrock':
            dur, fs, nch, seg = blackrock_block_info(f, cfg['LFP_file_format'])
            name, t_origin = os.path.basename(f), nsx_time_origin(f)
        else:
            probe = f[0]
            size = os.path.getsize(probe)
            n_rec = (size - NCS_HEADER_BYTES) // NCS_REC.itemsize
            hdr = ncs_header(probe)
            fs = float(hdr.get('SamplingFrequency', np.nan))
            dur, nch, seg = n_rec * 512.0 / fs, len(f), None
            name, t_origin = f"{len(f)} .ncs files", hdr.get('TimeCreated')
        rows.append({"block": k + 1, "file": name, "seg_index": seg,
                     "fs_raw": fs, "n_channels": nch, "duration_s": dur,
                     "time_origin": t_origin})

    df = pd.DataFrame(rows)
    if df.empty:
        return df, bt, warn
    df["offset_s"] = np.concatenate([[0.0], np.cumsum(df["duration_s"])[:-1]])

    for k in range(min(len(df), len(bt))):
        b = bt.iloc[k]
        df.loc[k, "beh_start_s"] = b.beh_start_s
        df.loc[k, "beh_end_s"] = b.beh_end_s
        df.loc[k, "head_margin_s"] = b.beh_start_s - df.loc[k, "offset_s"]
        df.loc[k, "tail_margin_s"] = (df.loc[k, "offset_s"]
                                      + df.loc[k, "duration_s"] - b.beh_end_s)
    return df, bt, warn


# =============================================================================
# FILTERING
# =============================================================================

NOTCH_RATIO_THRESHOLD = 2.0    # notch a harmonic only if it exceeds this


def measure_line_noise(x, fs, freqs=LINE_FREQS, halfwidth=1.5, side=(4.0, 12.0)):
    """Peak-to-flank power ratio at each line harmonic. 1.0 = no peak.

    Measured across sites: Baylor is essentially clean (60 Hz ratio ~1.1),
    UCLA has strong referential line noise that bipolar removes (60.9 -> 1.14),
    and Utah is heavily contaminated even after bipolar (1.1e8 -> 5.5e4).
    """
    from scipy.signal import welch
    f, P = welch(np.asarray(x, float), fs=fs,
                 nperseg=min(4096, np.shape(x)[-1]), axis=-1)
    out = {}
    for f0 in freqs:
        if f0 >= fs / 2.0:
            continue
        inb = np.abs(f - f0) <= halfwidth
        sd = (np.abs(f - f0) >= side[0]) & (np.abs(f - f0) <= side[1])
        if not inb.any() or not sd.any():
            continue
        out[float(f0)] = float(np.max(P[..., inb], axis=-1).mean()
                               / max(np.median(P[..., sd]), 1e-30))
    return out


def notch_filter(x, fs, freqs=LINE_FREQS, q=NOTCH_Q,
                 ratio_threshold=NOTCH_RATIO_THRESHOLD):
    """Zero-phase narrow notches, applied ONLY where line noise is present.

    Returns (filtered, applied) where `applied` maps frequency -> measured
    ratio for the harmonics that were actually notched.

    Adaptive rather than blanket because 120 Hz sits on the upper edge of the
    80-120 Hz ripple band: notching it where there is no line noise discards
    real ripple-band signal for nothing. Baylor needs no notch at all, Utah
    needs it badly, UCLA is cleaned by the bipolar montage alone.
    """
    ratios = measure_line_noise(x, fs, freqs)
    y = np.asarray(x, dtype=np.float64)
    applied = {}
    for f0, r in ratios.items():
        if r < ratio_threshold:
            continue
        b, a = iirnotch(f0, q, fs)
        y = sosfiltfilt(tf2sos(b, a), y, axis=-1)
        applied[f0] = r
    return y.astype(np.float32), applied, ratios


def resample_to(x, fs_in, fs_out=TARGET_FS):
    """Polyphase decimation. Never scipy.signal.resample -- see docstring."""
    if abs(fs_in - fs_out) < 1e-6:
        return np.asarray(x, dtype=np.float32)
    from math import gcd
    g = gcd(int(round(fs_in)), int(round(fs_out)))
    up, down = int(round(fs_out)) // g, int(round(fs_in)) // g
    return resample_poly(np.asarray(x, dtype=np.float64), up, down,
                         axis=-1).astype(np.float32)


# =============================================================================
# CONTINUOUS LOADING
# =============================================================================

def _load_blackrock_channels(path, nsx, seg_index, ch_positions,
                             fs_raw, duration_s, verbose=False):
    """Load selected channels of one Blackrock block, resampled to TARGET_FS.

    Read in chunks with a discarded pad: a full block of 20 channels at 2 kHz
    over 4800 s is ~1.5 GB as float64 from neo. Only the channels we actually
    need are ever materialised -- `channel_indexes` is passed to neo.
    """
    import neo
    reader = neo.io.BlackrockIO(filename=path, nsx_to_load=int(nsx))
    seg = reader.read_segment(block_index=0, seg_index=int(seg_index), lazy=True)
    asig = max(seg.analogsignals, key=lambda a: a.shape[0])
    t0_file = float(asig.t_start.magnitude)

    out = [[] for _ in ch_positions]
    t = 0.0
    while t < duration_s:
        lo = max(0.0, t - CHUNK_PAD_S)
        hi = min(duration_s, t + CHUNK_S + CHUNK_PAD_S)
        chunk = asig.load(time_slice=(t0_file + lo, t0_file + hi),
                          channel_indexes=list(ch_positions))
        arr = np.asarray(chunk.magnitude, dtype=np.float32).T   # (ch, time)
        arr = resample_to(arr, fs_raw, TARGET_FS)

        pre = int(round((t - lo) * TARGET_FS))
        want = int(round((min(t + CHUNK_S, duration_s) - t) * TARGET_FS))
        for i in range(len(ch_positions)):
            out[i].append(arr[i, pre:pre + want])
        del chunk, arr
        t += CHUNK_S
        if verbose:
            print(f"      ...{min(t, duration_s):.0f}/{duration_s:.0f}s", end="\r")

    return np.stack([np.concatenate(c) for c in out], axis=0)


def _load_ncs_channels(files, stems_wanted, duration_s, verbose=False):
    """Load selected UCLA contacts of one .ncs block, resampled to TARGET_FS."""
    by_stem = {os.path.splitext(os.path.basename(f))[0]: f for f in files}
    rows = []
    for stem in stems_wanted:
        path = by_stem.get(stem)
        if path is None:
            rows.append(np.zeros(int(duration_s * TARGET_FS), np.float32))
            continue
        sig, fs, _ = read_ncs(path)
        rows.append(resample_to(sig, fs, TARGET_FS))
    n = min(len(r) for r in rows)
    return np.stack([r[:n] for r in rows], axis=0)


def preprocess_session(session, pairs, data_root=None, verbose=True):
    """Build the continuous bipolar array for one session.

    Returns (signal, meta) where `signal` is (n_pairs, n_samples) float32 at
    TARGET_FS on the behavioural clock, so `sample = round(t_session * fs)`.

    `pairs` is the session's bipolar_pairs table; only the channels those pairs
    reference are ever read from disk.
    """
    data_root = data_root or swr_io.get_data_root()
    cfg = swr_io.session_config(session, data_root=data_root)
    files, kind, _ = swr_io.discover_raw_files(session, cfg, data_root=data_root)
    blocks, bt, _ = session_block_table(session, data_root=data_root)
    if blocks.empty or not len(pairs):
        raise RuntimeError(f"s{session:02d}: no blocks or no pairs")

    if kind == 'blackrock':
        wanted = sorted({int(v) for v in
                         list(pairs.ns_pos_a) + list(pairs.ns_pos_b)})
        index_of = {v: i for i, v in enumerate(wanted)}
    else:
        wanted = sorted({str(v) for v in
                         list(pairs.ns_label_a) + list(pairs.ns_label_b)})
        index_of = {v: i for i, v in enumerate(wanted)}

    if verbose:
        print(f"  s{session:02d}: {len(blocks)} block(s), {len(pairs)} pairs, "
              f"{len(wanted)} unique channels")

    per_block = []
    for k, b in blocks.iterrows():
        if verbose:
            print(f"    block {int(b.block)}: {b.duration_s:.0f}s @ {b.fs_raw:.0f}Hz")
        if kind == 'blackrock':
            raw = _load_blackrock_channels(
                os.path.join(os.path.dirname(files[k]), os.path.basename(files[k])),
                cfg['LFP_file_format'], b.seg_index, wanted,
                b.fs_raw, b.duration_s, verbose=verbose)
        else:
            raw = _load_ncs_channels(files[k], wanted, b.duration_s, verbose=verbose)
        per_block.append(raw)

    # Concatenating the blocks in order reproduces the behavioural clock,
    # because that clock IS the cumulative file duration.
    raw_all = np.concatenate(per_block, axis=1)
    del per_block

    # Bipolar first (removes most common-mode line noise), then notch.
    sig = np.empty((len(pairs), raw_all.shape[1]), dtype=np.float32)
    for i, (_, p) in enumerate(pairs.iterrows()):
        ka = index_of[int(p.ns_pos_a) if kind == 'blackrock' else str(p.ns_label_a)]
        kb = index_of[int(p.ns_pos_b) if kind == 'blackrock' else str(p.ns_label_b)]
        sig[i] = raw_all[ka] - raw_all[kb]
    del raw_all

    sig, notch_applied, notch_ratios = notch_filter(sig, TARGET_FS)
    if verbose:
        print("    line-noise ratio (1.0 = none): "
              + ", ".join(f"{f:.0f}Hz={r:.2f}" for f, r in notch_ratios.items()))
        print(f"    notched: {[f'{f:.0f}Hz' for f in notch_applied] or 'none needed'}")

    meta = {
        "session": int(session),
        "fs": TARGET_FS,
        "n_pairs": int(len(pairs)),
        "n_samples": int(sig.shape[1]),
        "duration_s": float(sig.shape[1] / TARGET_FS),
        "recording_site": cfg.get('recording_site'),
        "reader": kind,
        "pair_ids": list(pairs.pair_id),
        "blocks": blocks.to_dict("records"),
        "clock": "behavioural = cumulative file duration; sample = t*fs",
        "notch_candidates_hz": list(LINE_FREQS),
        "notch_applied_hz": {f"{k:.0f}": v for k, v in notch_applied.items()},
        "line_noise_ratio": {f"{k:.0f}": v for k, v in notch_ratios.items()},
        "notch_ratio_threshold": NOTCH_RATIO_THRESHOLD,
        "resample": "scipy.signal.resample_poly",
    }
    return sig, meta
