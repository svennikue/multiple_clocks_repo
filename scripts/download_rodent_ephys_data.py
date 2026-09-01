#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selective downloader for the El-Gaby et al. (2024) rodent dataset.

Two sources exist and they are NOT the same dataset:

  osf    https://osf.io/3d9r2/  — the complete public release.
         25 combined ABCD recdays from 7 mice. Ships RAW recordings only
         (Neuron_raw / Location_raw, 25 ms bins) plus trialtimes and Task_data.
         There are NO normalised 360-bin `Neuron_*` / `Location_*` files here.

  drive  the private Google Drive share — 14 combined ABCD recdays from 5 mice,
         trialtimes for only 8 of them, but it DOES carry the authors'
         normalised 360-bin arrays for those 8.

The 8 recdays currently on disk came from the Drive. The 17 that are missing
(including all of ab03 and ah07 — the two mice absent from the Drive share
entirely) are on OSF. That is only ~2.0 GB: this script fetches per-file, so
it never has to build the multi-GB archive that makes the web download fall
over, and it skips whatever is already on disk so a crashed run just resumes.

    per mouse (combined ABCD recdays, on disk / on OSF)
        ab03  0/3      ah03  1/2      ah04  3/5      ah07  0/3
        me08  1/3      me10  1/4      me11  2/5

*** NORMALISATION CAVEAT — read before using the new recdays ***
The DSR analysis runs on the NORMALISED view (n_neurons x n_trials x 360, 90
bins per state). OSF does not ship it, so for the 17 new recdays it has to be
rebuilt from Neuron_raw + trialtimes.

The authors' recipe is public — `raw_to_norm`/`normalise`, Basic_analysis.ipynb
cell 21 of github.com/mohamadyelgaby/mFC_schema:

    Trial_times_conc = np.hstack((np.concatenate(tt[:,:-1]), tt[-1,-1])) // 25
    segments  = partition(raw_neuron, Trial_times_conc)     # one per state
    per_state = binned_statistic(arange(L), seg, 'mean', bins=90)[0]
                # if len(seg) < 90: seg = np.repeat(seg,10)/10 first
    Neuron_norm = per_state.reshape(n_states//4, 360)       # NO smoothing here
    # Location uses the same call with statistic='max'.

Running it verbatim does NOT reproduce the shipped normalised files: mean
r = 0.877 for neurons, 0.785 for locations over 18 sessions, no exact match,
trial counts off by one in 8/18. The shipped `Neuron_raw` is evidently not the
exact array that was fed to it (bin numerators agree, denominators do not).

So the authors' 8 normalised recdays are NOT reproducible, and must not be
mixed with a home-made rebuild for the new 17 — the preprocessing difference
would line up exactly with the mouse/recday split and confound the group test.
Rebuild the normalised view from raw for ALL 25 recdays with the recipe above.

USAGE
-----
    conda activate env_multiple_clocks
    python scripts/download_rodent_ephys_data.py --dry-run
    python scripts/download_rodent_ephys_data.py                 # 17 recdays, ~2.0 GB
    python scripts/download_rodent_ephys_data.py --recdays ab03_01092023_02092023
    python scripts/download_rodent_ephys_data.py --tone-only     # skip the 5 no-tone recdays
    python scripts/download_rodent_ephys_data.py --source drive  # normalised files (5 mice only)

@author: Svenja Kuechenhoff
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np


# ── Settings ──────────────────────────────────────────────────────────
DATA_FOLDER = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/'

OSF_NODE = '3d9r2'
# OSF folder ids, resolved once from the API (Data/<folder>). Keyed by the file
# stem that lives in them, because the release splits one recday across folders.
OSF_FOLDERS = {
    'Task_data':    ('Tasks',                     '66c85d7a5e4b1eb1092d360d'),
    'trialtimes':   ('Trial_times',               '66c85d9453d80de7d1c5d2cf'),
    'Neuron_raw':   ('Neuronal_activity/Awake',   '66c85c95f71062efcc2d36a4'),
    'Location_raw': ('Maze location',             '66c77a5c0f13cac694c5dba6'),
    'XY_raw':       ('XY position',               '66c7773679e8d68efadbec9d'),
    'MetaData':     ('MetaData',                  '66c863194c9e11e329c5cca6'),
}
OSF_LIST     = 'https://api.osf.io/v2/nodes/{node}/files/osfstorage/{fid}/?page%5Bsize%5D=100'
OSF_DOWNLOAD = 'https://files.osf.io/v1/resources/{node}/providers/osfstorage/{fid}'

DRIVE_FOLDER_ID = '1vJw8AVZmHQrUnvqkASUwAd4t549uKN6b'

# The 25 combined ABCD recdays, i.e. OSF MetaData/combined_ABCDonly_days.npy.
# Hard-coded so the script works before anything has been downloaded.
ALL_RECDAYS = [
    'ab03_01092023_02092023', 'ab03_05092023_06092023', 'ab03_29082023_30082023',
    'ah03_12082021_13082021', 'ah03_18082021_19082021',
    'ah04_01122021_02122021', 'ah04_05122021_06122021', 'ah04_07122021_08122021',
    'ah04_09122021_10122021', 'ah04_14122021_16122021',
    'ah07_01092023_02092023', 'ah07_27082023_28082023', 'ah07_29082023_30082023',
    'me08_06092021_09092021', 'me08_10092021_11092021', 'me08_12092021_13092021',
    'me10_09122021_10122021', 'me10_14122021_15122021', 'me10_17122021_19122021',
    'me10_20122021_21122021',
    'me11_01122021_02122021', 'me11_05122021_06122021', 'me11_07122021_08122021',
    'me11_09122021_10122021', 'me11_12122021_13122021',
]

# combined_ABCDonly_notone_days.npy — recorded without the state tones. A
# different sensory regime, so keep them separable rather than silently pooled.
NOTONE_RECDAYS = {
    'ah04_14122021_16122021', 'me10_17122021_19122021', 'me10_20122021_21122021',
    'me11_09122021_10122021', 'me11_12122021_13122021',
}

# Per-session stems load_ephys_data(raw=True) opens, plus optional XY.
OSF_CORE_STEMS  = ['Neuron_raw', 'Location_raw', 'trialtimes']
OSF_EXTRA_STEMS = ['XY_raw']
# Dataset-level files worth having (all in MetaData on OSF).
OSF_SHARED = ['combined_ABCDonly_days.npy', 'single_ABCDonly_days.npy',
              'combined_ABCDE_days.npy', 'combined_ABCDonly_notone_days.npy',
              'Edge_grid.npy']

# Drive-only: the authors' normalised 360-bin arrays.
DRIVE_STEMS = ['Location', 'Neuron', 'Location_raw', 'Neuron_raw', 'trialtimes']
DRIVE_EXTRA = ['Anchor_lag', 'Anchor_lag_threshold', 'Phase_state_place_anchored']


def _get_json(url, tries=6):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"  retry {attempt}: {e}", file=sys.stderr)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")


# ── Indexes: filename -> download url ─────────────────────────────────
def osf_index(cache, refresh=False):
    """{filename: download_url} across the OSF folders we care about.

    The release is ~3800 files spread over six folders, so this is ~30 API
    calls; cached because it only changes when the authors re-upload.
    """
    if os.path.exists(cache) and not refresh:
        with open(cache) as f:
            return json.load(f)

    print("Indexing OSF (a few dozen API calls, ~1 min) ...")
    index = {}
    for stem, (label, fid) in OSF_FOLDERS.items():
        url, n = OSF_LIST.format(node=OSF_NODE, fid=fid), 0
        while url:
            d = _get_json(url)
            for f in d['data']:
                if f['attributes']['kind'] != 'file':
                    continue
                index[f['attributes']['name']] = {
                    'url':  OSF_DOWNLOAD.format(node=OSF_NODE, fid=f['id']),
                    'size': f['attributes'].get('size') or 0}
                n += 1
            url = d['links'].get('next')
        print(f"  {label:28s} {n:5d} files")
    with open(cache, 'w') as f:
        json.dump(index, f, indent=1)
    print(f"  -> {cache} ({len(index)} files)")
    return index


def drive_index(cache, refresh=False):
    """{filename: download_url} for the Drive share.

    Scrapes `embeddedfolderview`: gdown's folder listing is capped at 50
    entries per folder by Google's folder HTML, which cannot enumerate this
    folder's 5037 files.
    """
    if os.path.exists(cache) and not refresh:
        with open(cache) as f:
            cached = json.load(f)
        # An earlier version of this script cached {name: file_id} instead of
        # {name: {'url':..,'size':..}}. Upgrade in place rather than forcing a
        # re-scrape, so an old cache on disk cannot break the run.
        if cached and isinstance(next(iter(cached.values())), str):
            cached = {n: {'url': f'gdrive:{fid}', 'size': 0}
                      for n, fid in cached.items()}
            with open(cache, 'w') as f:
                json.dump(cached, f, indent=1)
        return cached

    print("Fetching Drive listing ...")
    url = f'https://drive.google.com/embeddedfolderview?id={DRIVE_FOLDER_ID}#list'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        html = resp.read().decode('utf-8', errors='replace')
    entries = re.findall(
        r'id="entry-([A-Za-z0-9_-]+)".*?flip-entry-title">([^<]+)</div>',
        html, flags=re.S)
    if not entries:
        raise RuntimeError("Drive listing came back empty — is the folder still shared?")
    index = {name: {'url': f'gdrive:{fid}', 'size': 0} for fid, name in entries}
    with open(cache, 'w') as f:
        json.dump(index, f, indent=1)
    print(f"  indexed {len(index)} files -> {cache}")
    return index


# ── Which files a recday needs ────────────────────────────────────────
def sessions_of(recday, index, stem):
    return {int(m.group(1)): m.group(0) for m in
            (re.match(rf'^{stem}_{re.escape(recday)}_(\d+)\.npy$', n) for n in index)
            if m}


def files_for_recday(recday, index, source, extras=False):
    """Every file this recday needs, in download order.

    Only sessions with ALL required stems present are included — a session
    missing e.g. its trialtimes is unusable, and `load_ephys_data` would drop
    it anyway.
    """
    stems  = OSF_CORE_STEMS if source == 'osf' else DRIVE_STEMS
    if extras:
        stems = stems + (OSF_EXTRA_STEMS if source == 'osf' else [])
    core   = OSF_CORE_STEMS if source == 'osf' else DRIVE_STEMS
    by_stem = {st: sessions_of(recday, index, st) for st in stems}
    common  = sorted(set.intersection(*(set(by_stem[st]) for st in core)))

    wanted = [f'Task_data_{recday}.npy']
    for s in common:
        wanted += [by_stem[st][s] for st in stems if s in by_stem[st]]
    if source == 'drive':
        wanted += [f'{st}_{recday}.npy' for st in DRIVE_EXTRA]
    return [f for f in wanted if f in index], common


def is_on_disk(fname, folder):
    """Present means: exists AND np.load can open it — a half-written file from
    a crashed download must be re-fetched, not silently trusted."""
    path = os.path.join(folder, fname)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    if not fname.endswith('.npy'):
        return True
    try:
        np.load(path, allow_pickle=True)
        return True
    except Exception:
        return False


# ── Download ──────────────────────────────────────────────────────────
def download_one(fname, url, folder, retries=3):
    """Fetch one file and verify it loads. Returns True on success."""
    path = os.path.join(folder, fname)
    for attempt in range(1, retries + 1):
        try:
            if url.startswith('gdrive:'):
                import gdown
                gdown.download(id=url.split(':', 1)[1], output=path, quiet=True)
            else:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=300) as r, open(path, 'wb') as out:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
        except Exception as e:
            print(f"    attempt {attempt}/{retries} failed: {e}")
        if is_on_disk(fname, folder):
            return True
        if os.path.exists(path):
            os.remove(path)          # drop the corrupt partial before retrying
        time.sleep(2 * attempt)      # back off
    return False


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', choices=['osf', 'drive'], default='osf')
    p.add_argument('--recdays', nargs='+', default=None,
                   help='recdays to fetch (default: all 25 combined ABCD recdays)')
    p.add_argument('--tone-only', action='store_true',
                   help='skip the 5 recdays recorded without state tones')
    p.add_argument('--extras', action='store_true',
                   help='also fetch XY_raw (head position; unused by the pipeline)')
    p.add_argument('--data-folder', default=DATA_FOLDER)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--refresh-index', action='store_true')
    p.add_argument('--retries', type=int, default=3)
    args = p.parse_args()

    os.makedirs(args.data_folder, exist_ok=True)
    if args.source == 'osf':
        index = osf_index(os.path.join(args.data_folder, '_osf_index.json'),
                          args.refresh_index)
    else:
        index = drive_index(os.path.join(args.data_folder, '_drive_root_index.json'),
                            args.refresh_index)

    recdays = args.recdays or ALL_RECDAYS
    if args.tone_only:
        recdays = [r for r in recdays if r not in NOTONE_RECDAYS]
    unknown = [r for r in recdays if f'Task_data_{r}.npy' not in index]
    if unknown:
        sys.exit(f"Not in the {args.source} release: {unknown}")

    # ── plan ──
    todo, plan_gb = [], 0.0
    print(f"\n{'recday':28s} {'tone':>5s} {'sess':>5s} {'needed':>7s} "
          f"{'on disk':>8s} {'fetch':>6s} {'GB':>7s}")
    for recday in recdays:
        wanted, common = files_for_recday(recday, index, args.source, args.extras)
        missing = [f for f in wanted if not is_on_disk(f, args.data_folder)]
        gb = sum(index[f]['size'] for f in missing) / 1e9
        plan_gb += gb
        todo += [(recday, f) for f in missing]
        print(f"{recday:28s} {'no' if recday in NOTONE_RECDAYS else 'yes':>5s} "
              f"{len(common):>5d} {len(wanted):>7d} {len(wanted) - len(missing):>8d} "
              f"{len(missing):>6d} {gb:>7.2f}")

    if args.source == 'osf':
        todo += [('(shared)', f) for f in OSF_SHARED
                 if f in index and not is_on_disk(f, args.data_folder)]

    print(f"\n{len(todo)} files, {plan_gb:.2f} GB to download.")
    if args.dry_run:
        for recday, f in todo:
            print(f"  {recday:28s} {f}")
        return
    if not todo:
        print("Nothing to do — everything is already on disk.")
        return

    failed = []
    for i, (recday, fname) in enumerate(todo, 1):
        print(f"[{i:4d}/{len(todo)}] {fname}")
        if not download_one(fname, index[fname]['url'], args.data_folder, args.retries):
            print(f"    FAILED: {fname}")
            failed.append(fname)

    print(f"\nDone. {len(todo) - len(failed)}/{len(todo)} downloaded.")
    if failed:
        print("Failed (just re-run the script, it resumes):")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)

    # Session ids do not always start at 0 (me08_06092021_09092021 starts at 1),
    # so probe the ids the index actually lists rather than assuming session 0.
    def _has(stem, r):
        return any(is_on_disk(f'{stem}_{r}_{s}.npy', args.data_folder)
                   for s in sessions_of(r, index, 'Neuron_raw'))
    raw_ok = [r for r in ALL_RECDAYS
              if is_on_disk(f'Task_data_{r}.npy', args.data_folder)
              and _has('Neuron_raw', r)]
    norm_ok = [r for r in raw_ok if _has('Neuron', r)]
    print(f"\nOn disk: {len(raw_ok)} recdays with raw "
          f"({len({r.split('_')[0] for r in raw_ok})} mice), "
          f"{len(norm_ok)} of them with the authors' normalised arrays.")
    if len(norm_ok) < len(raw_ok):
        print("  -> the rest need the normalised view rebuilt from raw; see the "
              "NORMALISATION CAVEAT at the top of this file before doing that.")


if __name__ == '__main__':
    main()
