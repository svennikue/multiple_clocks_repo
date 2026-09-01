#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the normalised (90 bins/state) rodent view from the raw OSF recordings.

WHY
---
The DSR RSA runs on the normalised view — (n_neurons, n_trials, 360), 90 bins
per ABCD state. The OSF release (https://osf.io/3d9r2/) ships RAW recordings
only, and the private Drive share carries the authors' normalised arrays for
just 8 of the 25 combined ABCD recdays. To analyse all 25 (7 mice) we have to
normalise ourselves.

*** THIS IS AN APPROXIMATION, AND THAT IS THE POINT ***
The authors have confirmed that the normalisation they settled on is NOT the
one published in Basic_analysis.ipynb, and it has not been shared. Running the
published `raw_to_norm` reproduces their shipped arrays only to r ~ 0.88
(neurons) / ~0.80 (locations) — never exactly.

So the released normalised files CANNOT be reproduced, and must not be mixed
with a self-normalised set: the preprocessing difference would line up exactly
with the mouse/recday split (their 8 vs our 17) and confound the group test.
This script therefore normalises ALL recdays it can, including the 8 that
already have released versions, so the analysis runs on one uniform
preprocessing. `analysis_rodents_complete_clean.py` should point its
`NORM_FOLDER` here and ignore the released `Neuron_*` / `Location_*` files.

WHAT IT DOES  (mc.analyse.analyse_ephys_clean.raw_to_norm)
    Trial_times_conc = hstack((concatenate(tt[:,:-1]), tt[-1,-1])) // 25
    partition the raw trace at those boundaries -> one segment per state
    binned_statistic(arange(L), segment, stat, bins=90)
        with the authors' short-segment rule: len < 90 -> repeat(x,10)/10 first
    reshape -> (n_trials, 360)
    no smoothing (the authors smooth only when averaging over trials)

Neurons use statistic='mean' (firing rate). Locations use 'max' by default —
the authors' `take_max` option; a mean of node IDs would be meaningless. This
is the one undocumented choice, so it is exposed as --location-statistic and
recorded in the settings JSON.

OUTPUT
    <data>/derivatives/normalised_<statistic>_<timestamp>/
        Neuron_{recday}_{s}.npy      (n_neurons, n_trials, 360)
        Location_{recday}_{s}.npy    (n_trials, 360)
        Recording_days_combined.npy  manifest of recdays fully normalised here
        normalisation_settings.json  parameters + per-session provenance
The authors' raw files are never modified.

USAGE
    conda activate env_multiple_clocks
    python scripts/normalise_rodent_ephys.py --dry-run
    python scripts/normalise_rodent_ephys.py

@author: Svenja Kuechenhoff
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np

import mc.analyse.analyse_ephys_clean as ae


DATA_FOLDER = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/'
OUT_BASE    = os.path.join(DATA_FOLDER, 'derivatives')

# The 25 combined ABCD recdays (OSF MetaData/combined_ABCDonly_days.npy).
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
NOTONE_RECDAYS = {
    'ah04_14122021_16122021', 'me10_17122021_19122021', 'me10_20122021_21122021',
    'me11_09122021_10122021', 'me11_12122021_13122021',
}


def sessions_on_disk(recday, folder):
    """Session ids with a complete raw triplet (Neuron_raw, Location_raw, trialtimes)."""
    out, s = [], 0
    while s < 64:
        need = [f'Neuron_raw_{recday}_{s}.npy', f'Location_raw_{recday}_{s}.npy',
                f'trialtimes_{recday}_{s}.npy']
        if all(os.path.exists(os.path.join(folder, f)) for f in need):
            out.append(s)
        s += 1
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-folder', default=DATA_FOLDER)
    p.add_argument('--out-dir', default=None,
                   help='default: <data>/derivatives/normalised_<stat>_<timestamp>/')
    p.add_argument('--recdays', nargs='+', default=ALL_RECDAYS)
    p.add_argument('--location-statistic', default='max',
                   choices=['max', 'mean', 'median', 'min'],
                   help="binned_statistic for Location_raw (default: max, the "
                        "authors' take_max). Neurons always use 'mean'.")
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = args.out_dir or os.path.join(
        OUT_BASE, f'normalised_loc-{args.location_statistic}_{stamp}')

    print(f"Raw in : {args.data_folder}")
    print(f"Norm out: {out_dir}")
    print(f"Location statistic: {args.location_statistic!r}  (neurons: 'mean')\n")

    print(f"{'recday':28s} {'tone':>5s} {'sessions':>9s} {'neurons':>8s} {'trials':>7s}")
    plan, provenance = [], {}
    for recday in args.recdays:
        sess = sessions_on_disk(recday, args.data_folder)
        if not sess:
            print(f"{recday:28s} {'-':>5s} {'MISSING RAW — skipped':>30s}")
            continue
        plan.append((recday, sess))
        n_neu = np.load(os.path.join(
            args.data_folder, f'Neuron_raw_{recday}_{sess[0]}.npy'), mmap_mode='r').shape[0]
        n_tr = sum(len(np.load(os.path.join(
            args.data_folder, f'trialtimes_{recday}_{s}.npy'))) for s in sess)
        print(f"{recday:28s} {'no' if recday in NOTONE_RECDAYS else 'yes':>5s} "
              f"{len(sess):>9d} {n_neu:>8d} {n_tr:>7d}")

    mice = sorted({r.split('_')[0] for r, _ in plan})
    print(f"\n{len(plan)} recdays from {len(mice)} mice ({', '.join(mice)}), "
          f"{sum(len(s) for _, s in plan)} sessions.")
    if args.dry_run:
        return
    if not plan:
        print("Nothing to normalise — download the raw files first "
              "(scripts/download_rodent_ephys_data.py).")
        return

    os.makedirs(out_dir, exist_ok=True)
    complete = []
    for recday, sess in plan:
        wrote = 0
        for s in sess:
            raw_n = np.load(os.path.join(args.data_folder, f'Neuron_raw_{recday}_{s}.npy'))
            raw_l = np.load(os.path.join(args.data_folder, f'Location_raw_{recday}_{s}.npy'))
            tt    = np.load(os.path.join(args.data_folder, f'trialtimes_{recday}_{s}.npy'))
            # An empty raw array is the authors' bad-session flag; propagate it
            # as an empty normalised array so cross_view_session_ids drops it.
            if raw_n.size == 0 or raw_l.size == 0 or tt.size == 0:
                neu = np.asarray([]); loc = np.asarray([])
            else:
                neu = ae.raw_to_norm(raw_n, tt, statistic='mean')
                loc = ae.raw_to_norm(raw_l, tt, statistic=args.location_statistic)
            np.save(os.path.join(out_dir, f'Neuron_{recday}_{s}.npy'), neu)
            np.save(os.path.join(out_dir, f'Location_{recday}_{s}.npy'), loc)
            provenance[f'{recday}_{s}'] = {
                'raw_shape':    list(raw_n.shape),
                'n_trials':     int(len(tt)),
                'neuron_shape': list(neu.shape),
                'location_shape': list(loc.shape),
                'empty':        bool(neu.size == 0),
            }
            wrote += 1
        complete.append(recday)
        print(f"  {recday:28s} {wrote} sessions -> {out_dir}")

    np.save(os.path.join(out_dir, 'Recording_days_combined.npy'), np.array(complete))

    settings = {
        'created':             datetime.now().isoformat(timespec='seconds'),
        'source_raw_folder':   args.data_folder,
        'out_dir':             out_dir,
        'method':              'mc.analyse.analyse_ephys_clean.raw_to_norm',
        'method_source':       ('partition/normalise/raw_to_norm, Basic_analysis.ipynb '
                                'cell 21, github.com/mohamadyelgaby/mFC_schema'),
        'raw_bin_ms':          ae.RAW_BIN_MS,
        'bins_per_state':      ae.BINS_PER_STATE,
        'states_per_trial':    ae.STATES_PER_TRIAL,
        'neuron_statistic':    'mean',
        'location_statistic':  args.location_statistic,
        'smoothing':           None,
        'caveat': ('The authors confirmed their final normalisation differs from '
                   'the published one and has not been shared. This reproduces '
                   'their released arrays only to r~0.88 (neurons) / ~0.80 '
                   '(locations). Use this set for ALL recdays; never mix it with '
                   'the released Neuron_*/Location_* files.'),
        'n_recdays':           len(complete),
        'n_mice':              len({r.split('_')[0] for r in complete}),
        'recdays':             complete,
        'notone_recdays':      sorted(NOTONE_RECDAYS & set(complete)),
        'per_session':         provenance,
    }
    with open(os.path.join(out_dir, 'normalisation_settings.json'), 'w') as f:
        json.dump(settings, f, indent=2)

    print(f"\nWrote {len(complete)} recdays "
          f"({settings['n_mice']} mice, {len(provenance)} sessions).")
    print(f"Settings: {os.path.join(out_dir, 'normalisation_settings.json')}")
    print(f"\nPoint the analysis at it:\n    NORM_FOLDER = '{out_dir}'")


if __name__ == '__main__':
    main()
