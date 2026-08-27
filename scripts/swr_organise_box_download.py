#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sort a flat Box download into the layout the SWR pipeline expects.

Target layout:
    <data_root>/s{NN}/LFP/*.ns2 | *.ns3
    <data_root>/s{NN}/micros_and_macros/*.ncs        (UCLA)
    <data_root>/s{NN}/electrodes/Electrodes.mat      (Utah)

DRY RUN BY DEFAULT. Nothing moves until you pass --execute.

Session assignment is *proposed*, never guessed silently:
  * Baylor filenames carry the patient code (`EMU-058_subj-YER_...ns3`), and a
    patient can span several sessions (YER = s18 and s19), so the mapping is
    ambiguous from the filename alone. Where a patient has exactly one session,
    it is assigned; where they have several, the file is listed as AMBIGUOUS
    with the candidate sessions and left for you to resolve.
  * Utah/UCLA filenames rarely carry a code at all, so those are reported for
    manual assignment.

Resolve ambiguities by writing a small CSV (`--mapping map.csv`) with columns
`filename,session` and re-running.

Usage:
    python scripts/swr_organise_box_download.py --src=/ceph/.../all_box_data
    python scripts/swr_organise_box_download.py --src=... --mapping=map.csv --execute

@author: Svenja Kuchenhoff
"""

import os
import re
import sys
import glob
import shutil

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mc.analyse.swr_io as swr_io

try:
    import fire
except ImportError:
    fire = None

print("ARGS:", sys.argv)

# where each kind of file belongs, relative to <data_root>/s{NN}/
DEST_SUBDIR = {".ns2": "LFP", ".ns3": "LFP", ".ns5": "LFP", ".nev": "LFP",
               ".ncs": "micros_and_macros", ".mat": "electrodes"}

_SUBJ_RE = re.compile(r'subj-([A-Z]{3})', re.I)

# Box folders are named `s{NN}[suffix]_{SITE}_{CODE}_{day}`, e.g.
# `s10_BCM_YEL_day1`, `s17_UT202306`, `s47new_UT202302`. The folder name is a
# far more reliable session key than the filename, which for Utah/UCLA carries
# no code at all. Anything that does not match is reported, never guessed --
# `s6?_BCM_YFP_day2?` is a real example from the Box listing.
_BOXDIR_RE = re.compile(r'^s(\d+)([A-Za-z]*)_(.+)$')


# Folders whose name does not parse but whose identity is known. The
# recording-length vs behaviour gate in swr_extract_continuous is the real
# check: if this assignment is wrong, every behavioural event will fail to fall
# inside the recording and the session is rejected loudly rather than analysed
# incorrectly. So a reasoned guess here is safe.
FOLDER_ALIASES = {
    "s6?_BCM_YFP_day2?": None,   # YFP day 2; session number to be filled in
}


def session_from_path(path, src):
    """Session number from the Box folder a file sits in, plus its patient tag."""
    rel = os.path.relpath(path, src)
    for part in rel.split(os.sep):
        if part in FOLDER_ALIASES and FOLDER_ALIASES[part] is not None:
            return FOLDER_ALIASES[part], "alias", "YFP"
        m = _BOXDIR_RE.match(part)
        if m:
            return int(m.group(1)), m.group(2), m.group(3)
    return None, "", None


def _subject_code(name):
    m = _SUBJ_RE.search(name)
    return m.group(1).upper() if m else None


def organise(src, data_root=None, mapping=None, execute=False, move=False):
    """Propose (or perform) the reorganisation."""
    R = data_root or swr_io.get_data_root()
    swr_io.start_log(os.path.join(swr_io.derivatives_dir(R), "group", "swr"),
                     "swr_organise_box_download")
    if not os.path.isdir(src):
        raise FileNotFoundError(src)

    # patient code -> sessions, from the curated cell table
    subj_map = swr_io.load_session_subject_map(R)     # {session: (label, site)}
    code_to_sessions = {}
    for sess, (label, _site) in subj_map.items():
        key = swr_io.normalise_subject_key(label)
        if key:
            code_to_sessions.setdefault(key, []).append(int(sess))

    # session -> expected patient code, for cross-checking folder names
    expected_code = {}
    for sess, (label, _site) in subj_map.items():
        k = swr_io.normalise_subject_key(label)
        if k:
            expected_code[int(sess)] = k

    manual = {}
    if mapping and os.path.isfile(mapping):
        m = pd.read_csv(mapping)
        manual = {str(r.filename).strip(): int(r.session) for r in m.itertuples()}
        print(f"loaded {len(manual)} manual assignments from {mapping}")

    rows = []
    for p in sorted(glob.glob(os.path.join(src, "**", "*"), recursive=True)):
        if not os.path.isfile(p):
            continue
        base = os.path.basename(p)
        ext = os.path.splitext(base)[1].lower()
        if ext not in DEST_SUBDIR:
            rows.append({"file": base, "ext": ext, "session": None,
                         "status": "IGNORED (not a data file)", "dest": ""})
            continue

        sess, status = None, ""
        if base in manual:
            sess, status = manual[base], "manual"
        else:
            # 1. the Box folder name (most reliable)
            fsess, fsuf, ftag = session_from_path(p, src)
            if fsess is not None:
                sess, status = fsess, f"folder s{fsess}{fsuf} ({ftag})"
                # cross-check the folder's patient tag against the manifest;
                # s47_UT202311 vs s47new_UT202302 is a real conflict in Box
                exp = expected_code.get(fsess)
                if exp and ftag:
                    a = str(exp).upper().replace("UT", "")
                    b = str(ftag).upper().replace("UT", "").replace("SJ", "")
                    if a and a not in b:
                        status = (f"CONFLICT folder says {ftag}, manifest says "
                                  f"{exp} for s{fsess}")
                        sess = None
            else:
                # 2. fall back to the patient code in the filename
                code = _subject_code(base)
                cands = code_to_sessions.get(code, []) if code else []
                if len(cands) == 1:
                    sess, status = cands[0], f"filename ({code})"
                elif len(cands) > 1:
                    status = f"AMBIGUOUS {code} -> sessions {sorted(cands)}"
                else:
                    status = "UNRESOLVED (no session in folder or filename)"

        dest = (os.path.join(R, f"s{sess:02d}", DEST_SUBDIR[ext], base)
                if sess else "")
        rows.append({"file": base, "ext": ext, "session": sess,
                     "status": status, "dest": dest,
                     "size_mb": round(os.path.getsize(p) / 1e6, 1), "src": p})

    d = pd.DataFrame(rows)
    data = d[d.ext.isin(DEST_SUBDIR)]

    print("\n" + "=" * 74)
    print(f" {len(data)} data files found under {src}")
    print("=" * 74)
    print(data.groupby("ext").agg(n=("file", "size"),
                                  total_mb=("size_mb", "sum")).to_string())
    print()
    resolved = data[data.session.notna()]
    print(f"  resolved   : {len(resolved)} files -> "
          f"{sorted(resolved.session.dropna().astype(int).unique())}")
    conflict = data[data.status.str.startswith("CONFLICT", na=False)]
    if len(conflict):
        print("\n-- CONFLICT: folder name disagrees with the manifest --")
        for f, g in conflict.groupby("status"):
            print(f"   {f}   ({len(g)} files)")
    amb = data[data.status.str.startswith("AMBIGUOUS", na=False)]
    unres = data[data.status.str.startswith("UNRESOLVED", na=False)]
    print(f"  ambiguous  : {len(amb)}")
    print(f"  unresolved : {len(unres)}")

    if len(amb):
        print("\n-- AMBIGUOUS (one patient, several sessions) --")
        for _, r in amb.iterrows():
            print(f"   {r.file:58s} {r.status}")
    if len(unres):
        print("\n-- UNRESOLVED (assign by hand) --")
        for _, r in unres.head(20).iterrows():
            print(f"   {r.file:58s} {r.ext}")
        if len(unres) > 20:
            print(f"   ... and {len(unres) - 20} more")

    out = os.path.join(swr_io.derivatives_dir(R), "group", "swr")
    os.makedirs(out, exist_ok=True)
    plan_p = os.path.join(out, "box_organise_plan.csv")
    d.to_csv(plan_p, index=False)
    print(f"\n plan saved -> {plan_p}")

    if len(amb) or len(unres):
        print("\n To resolve: make a CSV with columns `filename,session`, then")
        print("   python scripts/swr_organise_box_download.py --src=... "
              "--mapping=map.csv [--execute]")

    if not execute:
        print("\n DRY RUN — nothing moved. Re-run with --execute to apply.")
        for _, r in resolved.head(10).iterrows():
            print(f"   {'move' if move else 'copy'}  {r.file}  ->  {r.dest}")
        if len(resolved) > 10:
            print(f"   ... and {len(resolved) - 10} more")
        return None

    n = 0
    for _, r in resolved.iterrows():
        os.makedirs(os.path.dirname(r.dest), exist_ok=True)
        if os.path.exists(r.dest):
            print(f"   exists, skipping: {r.file}")
            continue
        (shutil.move if move else shutil.copy2)(r.src, r.dest)
        n += 1
    print(f"\n {'moved' if move else 'copied'} {n} files.")
    print(" Now run:  python scripts/swr_check_inputs.py")
    return None


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(organise)
    else:
        organise(src=sys.argv[1] if len(sys.argv) > 1 else ".")
