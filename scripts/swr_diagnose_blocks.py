#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose block structure by reading the RAW FILES, not the YAML.

For each raw file: open every neo segment, read its duration and channel
count, then match the resulting recording blocks against the behavioural
block durations. Resolves sessions the config describes wrongly.

The match is a duration comparison. A behavioural block spans
[new_grid_onset.min(), t_D.max()] on the session clock; the recording that
contains it must be at least that long, and for a session where recording
started shortly before the task and stopped shortly after, it should be only
a little longer.

Usage:
    python scripts/swr_diagnose_blocks.py --sessions="[18,28,32,3]"

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


def probe_blackrock(path, nsx):
    """Every segment in a Blackrock file: duration, n samples, n channels."""
    import neo
    out = []
    try:
        reader = neo.io.BlackrockIO(filename=path, nsx_to_load=int(nsx))
    except Exception as e:
        return [{'error': f"{type(e).__name__}: {e}"}]

    try:
        n_blocks = reader.block_count()
    except Exception:
        n_blocks = 1

    for bi in range(n_blocks):
        try:
            n_seg = reader.segment_count(bi)
        except Exception:
            n_seg = 1
        for si in range(n_seg):
            try:
                seg = reader.read_segment(block_index=bi, seg_index=si, lazy=True)
                asigs = seg.analogsignals
                if not asigs:
                    continue
                # pick the analogsignal with the most samples
                k = int(np.argmax([a.shape[0] for a in asigs]))
                a = asigs[k]
                out.append({
                    'block_index': bi,
                    'seg_index': si,
                    'asig_index': k,
                    'n_samples': int(a.shape[0]),
                    'n_channels': int(a.shape[1]),
                    'fs': float(a.sampling_rate.magnitude),
                    't_start_s': float(a.t_start.magnitude),
                    'duration_s': float(a.shape[0] / a.sampling_rate.magnitude),
                })
            except Exception as e:
                out.append({'block_index': bi, 'seg_index': si,
                            'error': f"{type(e).__name__}: {e}"})
    return out


def probe_neuralynx(ncs_files):
    """Duration of one UCLA .ncs recording block (probe a single contact)."""
    import neo
    out = []
    for f in ncs_files[:1]:                      # one contact defines the block
        try:
            reader = neo.io.NeuralynxIO(dirname=os.path.dirname(f),
                                        include_filenames=[os.path.basename(f)])
            seg = reader.read_segment(lazy=True)
            a = seg.analogsignals[0]
            out.append({
                'file': os.path.basename(f),
                'n_samples': int(a.shape[0]),
                'n_channels': int(a.shape[1]),
                'fs': float(a.sampling_rate.magnitude),
                't_start_s': float(a.t_start.magnitude),
                'duration_s': float(a.shape[0] / a.sampling_rate.magnitude),
            })
        except Exception as e:
            out.append({'file': os.path.basename(f),
                        'error': f"{type(e).__name__}: {e}"})
    return out


def diagnose(sessions, save_all=False):
    swr_io.start_log(os.path.join(swr_io.derivatives_dir(swr_io.get_data_root()), "group", "swr"), "swr_diagnose_blocks")
    data_root = swr_io.get_data_root()
    cfg = swr_io.load_config(data_root)
    rows = []

    for s in [int(x) for x in sessions]:
        cfg_s = swr_io.session_config(s, cfg=cfg, data_root=data_root)
        site = cfg_s.get('recording_site')
        files, kind, fwarn = swr_io.discover_raw_files(s, cfg_s, data_root=data_root)

        print("\n" + "=" * 78)
        print(f" s{s:02d}   site={site}   yaml_blocks={cfg_s.get('blocks')}   "
              f"yaml_segment={cfg_s.get('segment')}")
        print("=" * 78)

        try:
            beh = swr_io.load_behaviour(s, data_root=data_root)
            bt = swr_io.block_table(beh)
        except Exception as e:
            print(f"  behaviour unreadable: {e}")
            continue

        print("\n  behavioural blocks (session clock):")
        for _, b in bt.iterrows():
            gap = "" if np.isnan(b.gap_to_prev_s) else f"  gap_to_prev={b.gap_to_prev_s:8.1f}s"
            print(f"    block {int(b.session_no)}: "
                  f"{b.beh_start_s:9.1f} -> {b.beh_end_s:9.1f} s  "
                  f"(span {b.beh_duration_s:8.1f}s, {int(b.n_repeats)} repeats){gap}")

        print(f"\n  raw recordings found ({kind}), {len(files)} block(s):")
        probes = []
        for i, f in enumerate(files):
            if kind == 'neuralynx':
                info = probe_neuralynx(f)
                name = f"ncs block {i} ({len(f)} contacts)"
            else:
                info = probe_blackrock(f, cfg_s.get('LFP_file_format', 3))
                name = os.path.basename(f)
            print(f"    [{i}] {name}")
            for d in info:
                if 'error' in d:
                    print(f"          ERROR {d['error']}")
                    continue
                print(f"          seg{d.get('seg_index','-')} "
                      f"dur={d['duration_s']:9.1f}s  fs={d['fs']:.0f}  "
                      f"ch={d['n_channels']:4d}  t_start={d['t_start_s']:.1f}")
                probes.append({**d, 'file_index': i, 'file': name})

        # --- match longest segment per file against behavioural spans -----
        usable = [p for p in probes if p.get('duration_s', 0) > 20]
        if usable:
            per_file = {}
            for p in usable:
                k = p['file_index']
                if k not in per_file or p['duration_s'] > per_file[k]['duration_s']:
                    per_file[k] = p
            rec = [per_file[k] for k in sorted(per_file)]

            print(f"\n  MATCH: {len(bt)} behavioural block(s) vs "
                  f"{len(rec)} usable recording(s)")
            n = min(len(bt), len(rec))
            for i in range(n):
                b = bt.iloc[i]
                r = rec[i]
                fits = r['duration_s'] >= b.beh_duration_s
                slack = r['duration_s'] - b.beh_duration_s
                print(f"    beh block {int(b.session_no)} (span {b.beh_duration_s:8.1f}s)"
                      f"  <-  [{r['file_index']}] seg{r.get('seg_index','-')} "
                      f"dur={r['duration_s']:9.1f}s   "
                      f"{'FITS' if fits else 'TOO SHORT'} (slack {slack:+9.1f}s)")
                rows.append({'session': s, 'beh_block': int(b.session_no),
                             'beh_span_s': b.beh_duration_s,
                             'file_index': r['file_index'], 'file': r['file'],
                             'seg_index': r.get('seg_index'),
                             'rec_duration_s': r['duration_s'],
                             'slack_s': slack, 'fits': fits})
            if len(rec) > len(bt):
                print(f"    !! {len(rec) - len(bt)} extra recording(s) with no "
                      f"behavioural block -- likely a non-task recording")
            if len(bt) > len(rec):
                print(f"    !! {len(bt) - len(rec)} behavioural block(s) with no "
                      f"recording")

    df = pd.DataFrame(rows)
    if save_all and len(df):
        out = os.path.join(swr_io.derivatives_dir(data_root), "group", "swr")
        os.makedirs(out, exist_ok=True)
        df.to_csv(os.path.join(out, "block_diagnosis.csv"), index=False)
        print(f"\nSaved -> {out}/block_diagnosis.csv")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(diagnose)
    else:
        diagnose([18, 28, 32, 3])
