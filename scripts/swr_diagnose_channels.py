#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why did a session's channels fail to join its electrode table?

`build_macro_table` joins on a site-specific key:
    utah    channel name must be exactly `chanN`  -> amplifier channel column
    ucla    `.ncs` filename stem                  -> v2026 xlsx `electrode`
    baylor  label with the trailing `-NNN` removed -> v2026 CSV `Label`

If the raw header names channels some other way, every channel drops with
`resolved = False` and the session yields no contacts. This prints, side by
side, what the channel list actually contains and what the electrode table
expects, so the mismatch is visible rather than inferred.

Prints only; writes nothing outside the data root.

Usage:
    python scripts/swr_diagnose_channels.py --sessions="[4,41,42,47,48,50,51,53]"

@author: Svenja Kuchenhoff
"""

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io
import mc.analyse.anatomy_sources as anat_src
import mc.analyse.contact_anatomy as ca

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)


def diagnose(sessions=None, data_root=None):
    R = data_root or swr_io.get_data_root()
    mf = pd.read_csv(os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                                  "session_manifest.csv"))
    if sessions is None:
        q = pd.read_csv(os.path.join(swr_io.derivatives_dir(R), "group", "swr",
                                     "contact_qc.csv"))
        sessions = sorted(int(s) for s in q.loc[q.n_hpc_pairs == 0, "session"])
    sessions = [int(s) for s in sessions]

    v2026 = os.path.join(R, "ABCD_pts_elecFilesForSvenja_v2026")
    utah_idx = anat_src.index_utah_mats_by_id(R)
    bay = ca.load_baylor_macros(v2026)
    ucla = ca.load_ucla_macros(v2026)

    for s in sessions:
        row = mf[mf.session == s]
        site = str(row.recording_site.iloc[0]).lower() if len(row) else "?"
        lab = (str(row.subject_label.iloc[0]).strip().strip("'")
               if len(row) and pd.notna(row.subject_label.iloc[0]) else None)
        chans, src, path = ca.load_channel_list(s, R)

        print("\n" + "=" * 74)
        print(f" s{s:02d}  {site}  {lab}")
        print("=" * 74)
        print(f"  channel source : {src}")
        print(f"  read from      : {path}")
        print(f"  n channels     : {len(chans)}")
        print(f"  first 10 names : {chans[:10]}")

        if site == "utah":
            pid = anat_src.subject_numeric_id(lab) if lab else None
            if pid in utah_idx:
                loc, mat = utah_idx[pid]
                t = ca.load_utah_macros(mat)
                # the two candidate keys the join now chooses between
                amp = set(t["utah_chan"].dropna().astype(int)) if "utah_chan" in t else set()
                by_chan = sum(1 for c in chans
                              if re.match(r'^chan\d+$', c)
                              and int(c[4:]) in amp)
                tl = set(str(x).strip() for x in t["anat_label"])
                by_label = sum(1 for c in chans if c in tl)
                print(f"  join by chanN  : {by_chan}/{len(chans)}")
                print(f"  join by label  : {by_label}/{len(chans)}"
                      + ("   <-- this key wins" if by_label > by_chan else ""))
                if max(by_chan, by_label) == 0:
                    print("  ** NEITHER KEY MATCHES - the channel list and the "
                          "electrode table use different vocabularies **")
                print(f"  electrode table: {loc}  ({len(t)} rows)")
                print(f"    labels       : {[str(x) for x in t['anat_label'].head(6)]}")
                if "utah_chan" in t.columns:
                    v = t["utah_chan"].dropna()
                    print(f"    amp channels : {sorted(v.astype(int))[:10]} "
                          f"({len(v)} finite)")
            else:
                print(f"  electrode table: NONE for patient {pid}")

        elif site == "ucla":
            t = ucla.get(lab)
            print(f"  electrode table: "
                  + (f"{len(t)} rows" if t is not None else f"NONE for '{lab}'"))
            if t is not None:
                print(f"    electrode col : {[str(x) for x in t['anat_label'].head(6)]}")
            stems = [re.sub(r'\.ncs$|_\d{4}$', "", c) for c in chans[:6]]
            print("    channel stems : " + str(stems))

        elif site == "baylor":
            code = re.sub(r'^BY\d*[-_]?', '', lab or "")
            t = bay.get(code)
            print(f"  electrode table: "
                  + (f"{len(t)} rows for {code}" if t is not None else f"NONE for '{code}'"))
            if t is not None:
                print(f"    labels       : {[str(x) for x in t['anat_label'].head(6)]}")
            stems = [re.sub(r'-\d+$', "", c) for c in chans[:6]]
            print("    channel stems : " + str(stems))


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(diagnose)
    else:
        diagnose()
