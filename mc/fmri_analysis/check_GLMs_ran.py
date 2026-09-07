#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check whether first-level FEAT GLMs actually finished, before trusting them.

A FEAT job can die in ways that leave a plausible-looking .feat directory
behind: killed mid-FILM (stats/ half full), submitted against a missing EV
folder, or re-run into a '+.feat' twin while the original stale directory is the
one every downstream script keeps reading. This walks the expected
(subject x task half x GLM) grid and reports each of those separately, so the
output is a list of jobs to resubmit rather than a single pass/fail.

WHAT IS CHECKED, per GLM
    1. glm_{glm}_pt0{th}.feat exists
    2. stats/ exists
    3. design.mat is there and says how many regressors to expect (/NumWaves,
       which includes the motion/PNM confound columns FEAT appends)
    4. pe1..pe{NumWaves}.nii.gz all exist and are non-empty
    5. FILM's end-of-run outputs are there (dof, smoothness, sigmasquareds,
       threshac1) -- these are written last, so they are the completion marker
    6. every PE the RSA will actually read exists. The RSA maps EV -> PE through
       EVs_{glm}_pt0{th}/task-to-EV.txt, so a GLM can be complete in itself and
       still be unusable if that file disagrees with the design.
    7. no glm_{glm}_pt0{th}+.feat twin (FEAT does not overwrite; it appends '+'
       and leaves the old directory in place)

    --check-data additionally loads every PE the RSA reads and flags all-zero or
    non-finite maps. That catches a GLM that ran to completion on a degenerate
    design. It is slow (one nifti read per PE), so it is off by default.

USAGE
    # the 11 instruction-epoch GLMs, all subjects, both halves
    python check_GLMs_ran.py

    # any other set of GLMs
    python check_GLMs_ran.py --glms 01-TR0 01-TR1 01-TR2
    python check_GLMs_ran.py --glms all-paths-fixed_stickrews_split-buttons

    # only some subjects, and verify the PE data too
    python check_GLMs_ran.py --subjects 01 02 35 --check-data

@author: Svenja Kuechenhoff
"""

import argparse
import json
import os
import sys

# --- where things live ----------------------------------------------------
source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if os.path.isdir(source_dir):
    data_dir_deriv = f"{source_dir}/data/derivatives"
    config_path = f"{source_dir}/multiple_clocks_repo/condition_files"
else:
    source_dir = "/home/fs0/xpsy1114/scratch"
    data_dir_deriv = f"{source_dir}/data/derivatives"
    config_path = f"{source_dir}/analysis/multiple_clocks_repo/condition_files"

DEFAULT_SUBJECTS = [f"{i:02}" for i in range(1, 36) if i not in (21, 29)]
# written by FILM at the very end of a first-level run
FILM_DONE_MARKERS = ['dof', 'smoothness', 'sigmasquareds.nii.gz', 'threshac1.nii.gz']
# EV names that are nuisance regressors, skipped by the RSA loader
NUISANCE_EVS = {'press_EV', 'up', 'down', 'left', 'right'}


def glm_names_from_config(config_file):
    """The same names create_EVs_instruction_period.py builds its folders from,
    so this checker and the EV script cannot drift apart."""
    with open(f"{config_path}/{config_file}", "r") as f:
        config = json.load(f)
    version = config['name']
    return [f"{version}_{seg['name'].replace('_', '-')}" for seg in config['segments']]


def num_waves(design_mat):
    """Regressor count FEAT actually fitted, from the design it wrote."""
    with open(design_mat, 'r') as f:
        for line in f:
            if line.startswith('/NumWaves'):
                return int(line.split()[1])
    return None


def rsa_pe_indices(EV_folder):
    """(pe number, EV name) for every EV the RSA reads, mirroring
    mc.analyse.my_RSA.load_data_EVs_instr_TRwise: line i of task-to-EV.txt maps
    to pe{i+1}, nuisance regressors skipped."""
    EV_txt = os.path.join(EV_folder, 'task-to-EV.txt')
    if not os.path.exists(EV_txt):
        return None
    out = []
    with open(EV_txt, 'r') as f:
        for line in f:
            index, name_ev = line.strip().split(' ', 1)
            name = name_ev.replace('ev_', '')
            if name in NUISANCE_EVS:
                continue
            out.append((int(index) + 1, name))
    return out


def check_one(sub, th, glm, check_data=False):
    """Return (status, detail). status 'OK' means safe to use."""
    func = f"{data_dir_deriv}/sub-{sub}/func"
    feat = f"{func}/glm_{glm}_pt0{th}.feat"
    EV_folder = f"{func}/EVs_{glm}_pt0{th}"

    if not os.path.isdir(EV_folder):
        return 'NO_EV_FOLDER', EV_folder
    if not os.path.isdir(feat):
        return 'NO_FEAT_DIR', feat

    stats = f"{feat}/stats"
    if not os.path.isdir(stats):
        return 'NO_STATS', 'FEAT started but never got to FILM'

    design = f"{feat}/design.mat"
    if not os.path.exists(design):
        return 'NO_DESIGN', 'design.mat missing'
    n_waves = num_waves(design)
    if n_waves is None:
        return 'BAD_DESIGN', 'design.mat has no /NumWaves'

    missing_pe = [i for i in range(1, n_waves + 1)
                  if not os.path.exists(f"{stats}/pe{i}.nii.gz")
                  or os.path.getsize(f"{stats}/pe{i}.nii.gz") == 0]
    if missing_pe:
        return 'INCOMPLETE_PES', (f"{len(missing_pe)}/{n_waves} PEs missing or empty "
                                  f"(first: pe{missing_pe[0]})")

    # prune_feat_dirs.py strips everything but the PEs from old GLMs, which
    # removes two of the FILM markers. The PEs are still the real ones.
    pruned = os.path.exists(f"{feat}/PRUNED.json")
    missing_marker = [m for m in FILM_DONE_MARKERS if not os.path.exists(f"{stats}/{m}")]
    if missing_marker and not pruned:
        return 'FILM_UNFINISHED', f"all PEs present but missing {', '.join(missing_marker)}"

    rsa_pes = rsa_pe_indices(EV_folder)
    if rsa_pes is None:
        return 'NO_TASK_TO_EV', 'task-to-EV.txt missing, RSA cannot map EVs to PEs'
    beyond = [f"pe{i} ({n})" for i, n in rsa_pes if i > n_waves]
    if beyond:
        return 'EV_PE_MISMATCH', (f"task-to-EV.txt points past the design "
                                  f"({n_waves} waves): {', '.join(beyond[:3])}")

    if check_data:
        import numpy as np
        import nibabel as nib
        for i, name in rsa_pes:
            d = np.asarray(nib.load(f"{stats}/pe{i}.nii.gz").dataobj)
            if not np.isfinite(d).any() or not np.any(d):
                return 'EMPTY_PE', f"pe{i} ({name}) is all-zero or non-finite"

    # Completed, but a later run may have gone somewhere else: FEAT never
    # overwrites an existing output dir, it appends '+' and leaves this one.
    if os.path.isdir(f"{func}/glm_{glm}_pt0{th}+.feat"):
        return 'STALE_TWIN', ("a '+.feat' twin exists -- this directory is from "
                              "the FIRST run and is what the RSA will read")
    return 'OK', ''


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--glms', nargs='+', default=None,
                    help='GLM names to check. Default: the epochs in --ev-config.')
    ap.add_argument('--ev-config', default='EV_config_instruction.json',
                    help='EV config the default GLM names are derived from.')
    ap.add_argument('--subjects', nargs='+', default=DEFAULT_SUBJECTS)
    ap.add_argument('--task-halves', nargs='+', type=int, default=[1, 2])
    ap.add_argument('--check-data', action='store_true',
                    help='also load every PE the RSA reads and flag all-zero / non-finite maps (slow)')
    ap.add_argument('--save-report', default=None,
                    help='write the failures to this file (put it in data/derivatives/group/, not the repo)')
    args = ap.parse_args()

    glms = args.glms if args.glms else glm_names_from_config(args.ev_config)
    print(f"checking {len(glms)} GLM(s) x {len(args.subjects)} subjects x "
          f"{len(args.task_halves)} halves = "
          f"{len(glms) * len(args.subjects) * len(args.task_halves)} runs")
    print(f"under {data_dir_deriv}\n")

    problems, per_glm = [], {}
    for glm in glms:
        counts = {}
        for sub in args.subjects:
            for th in args.task_halves:
                status, detail = check_one(sub, th, glm, args.check_data)
                counts[status] = counts.get(status, 0) + 1
                if status != 'OK':
                    problems.append((glm, sub, th, status, detail))
        per_glm[glm] = counts
        n_ok = counts.get('OK', 0)
        total = sum(counts.values())
        other = ', '.join(f"{k}:{v}" for k, v in sorted(counts.items()) if k != 'OK')
        print(f"  {glm:<40} {n_ok:>3}/{total} OK" + (f"   [{other}]" if other else ""))

    print()
    if not problems:
        print("All runs complete.")
        return 0

    print(f"=== {len(problems)} run(s) need attention ===")
    by_status = {}
    for glm, sub, th, status, detail in problems:
        by_status.setdefault(status, []).append((glm, sub, th, detail))
    for status, rows in sorted(by_status.items()):
        print(f"\n{status} ({len(rows)}):")
        for glm, sub, th, detail in rows[:15]:
            print(f"    sub-{sub} pt{th} {glm}" + (f"  --  {detail}" if detail else ""))
        if len(rows) > 15:
            print(f"    ... and {len(rows) - 15} more")

    # Resubmit list: the distinct subjects per GLM that failed, so the failures
    # can be re-run without redoing everything.
    print("\n=== to resubmit (per GLM: the subjects that failed) ===")
    resubmit = {}
    for glm, sub, th, status, detail in problems:
        resubmit.setdefault(glm, set()).add(sub)
    lines = []
    for glm in glms:
        if glm in resubmit:
            lines.append(f"{glm}: {' '.join(sorted(resubmit[glm]))}")
    print('\n'.join(lines))

    if args.save_report:
        with open(args.save_report, 'w') as f:
            f.write(f"checked {len(glms)} GLMs, {len(problems)} problems\n\n")
            for glm, sub, th, status, detail in problems:
                f.write(f"sub-{sub}\tpt{th}\t{glm}\t{status}\t{detail}\n")
            f.write("\nresubmit:\n" + '\n'.join(lines) + "\n")
        print(f"\n(report written to {args.save_report})")
    return 1


if __name__ == '__main__':
    sys.exit(main())
