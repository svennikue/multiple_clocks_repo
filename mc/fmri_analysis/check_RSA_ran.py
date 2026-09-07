#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decide which (subject x instruction epoch) searchlight RSAs still need running.

`fMRI_run_RSA_instruction.py` is expensive (one searchlight pass per subject per
epoch) and it reads the first-level GLMs directly, so submitting it blindly
wastes queue time two ways: on subjects whose epoch GLM never finished (the job
dies on a missing PE, or worse, reads a half-written one), and on subjects that
are already done. This walks the (subject x epoch) grid and sorts it into:

    READY            all inputs present, no result yet          -> submit
    RERUN_CHANGED    a result exists but was made with DIFFERENT settings
                     than this config asks for                  -> submit
                     (it overwrites the old maps; --skip-changed leaves it)
    DONE             a result exists and its settings match     -> skip
    GLM_NOT_READY    the epoch GLM has no complete run in the base directory
                     the RSA reads                              -> skip
    MISSING_INPUT    modelled EVs / reference image / mask absent -> skip

GLM readiness is not re-implemented here: it calls `check_GLMs_ran.check_one`,
the same function that produces the FEAT audit, so 'complete' means exactly the
same thing in both places. Note that PROMOTE_TWIN does NOT count as ready --
that status means the finished run is sitting in a '+' twin while the base
directory the RSA reads is broken, so the cleanup has to run first.

'ALREADY DONE' MEANS DONE *WITH THESE SETTINGS*
    The completion marker is `{sub}_settings_summary.json`, which
    fMRI_run_RSA_instruction.py writes as its very last action -- so its
    presence means the whole run finished, not just some of the maps. Its
    contents are then compared field by field against what this config asks
    for (`COMPARED_KEYS`). A result built with other models, another scope,
    different smoothing or a different GLM is NOT treated as done. A summary
    written by an older version of the RSA script that lacks a compared field
    also counts as changed: the settings cannot be shown to match.

WHAT IT WRITES  (--out-dir, default derivatives/group/rsa_audit_<name>_<date>/)
    report.txt      every (subject, epoch) that is not READY, with the reason
    todo_rsa.txt    'subject epoch_config' per job to submit
    settings.json   what this audit was run with
  and one per-epoch config snapshot per epoch, next to the base config:
    <base>_<epoch>.json   = the base config with regression_version set to the
                            epoch GLM and TR set to null, which is what makes
                            fMRI_run_RSA_instruction.py read
                            glm_instr_<epoch>_pt0{1,2}.feat.

USAGE
    python3 check_RSA_ran.py                      # audit, write the todo list
    python3 check_RSA_ran.py --skip-changed       # leave existing results alone
    python3 check_RSA_ran.py --subjects 01 02     # a subset
    bash submit_RSA_instruction_epochs.sh         # runs this, then submits

@author: Svenja Kuechenhoff
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_GLMs_ran as glmcheck

# A GLM is usable by the RSA only if the BASE directory is complete: that is the
# only name my_RSA.load_data_EVs_instr_TRwise ever opens. 'PROMOTE_TWIN' means a
# twin finished but the base did not, so it is deliberately not in this set.
GLM_READY = {'OK', 'DUPLICATES'}

# Settings that must match for an existing result to count as done. These are
# the keys of fMRI_run_RSA_instruction.py's summary that come from the config
# rather than from the data (n_cells_per_searchlight, paired_labels etc. are
# outputs, not settings, so they are not compared).
COMPARED_KEYS = ['EV_string', 'regression_version', 'TR', 'regression_version_full',
                 'RDM_version', 'smoothing', 'fwhm', 'searchlight_mask',
                 'data_rdm_scope', 'models_evaluated', 'run_single_models',
                 'run_combo_models', 'combo_models']

# Mirrors SCOPE_ALIASES in fMRI_run_RSA_instruction.py: the config may use the
# short form, the summary always stores the canonical one.
SCOPE_ALIASES = {'across_only': 'across_only', 'across': 'across_only',
                 'within_only': 'within_only', 'within': 'within_only',
                 'full_no_diag': 'full_no_diag', 'full': 'full_no_diag'}


def expected_settings(config):
    """What fMRI_run_RSA_instruction.py would write into its summary, given this
    config. Defaults repeated from the script -- keep them in step with it."""
    TR = config.get("TR")
    regression_version = config.get("regression_version")
    combo_cfg = [dict(c) for c in config.get("combo_models", [])]
    if config.get("add_block_nuisance", False):
        for combo in combo_cfg:
            if 'block' not in combo["regressors"]:
                combo["regressors"] = list(combo["regressors"]) + ['block']
    return {
        'EV_string': config.get("load_EVs_from"),
        'regression_version': regression_version,
        'TR': TR,
        'regression_version_full': (regression_version if TR is None
                                    else f"{regression_version}-TR{TR}"),
        'RDM_version': config.get("name_of_RSA"),
        'smoothing': config.get("smoothing", True),
        'fwhm': config.get("fwhm", 5),
        'searchlight_mask': config.get("searchlight_mask", None),
        'data_rdm_scope': SCOPE_ALIASES[config.get("data_rdm_scope", "across_only")],
        'models_evaluated': config.get("selected_models", ['DSR', 'rewDSR', 'simple']),
        'run_single_models': config.get("run_single_models", True),
        'run_combo_models': config.get("run_combo_models", bool(combo_cfg)),
        'combo_models': combo_cfg,
    }


def results_dir_for(data_dir, config):
    """The same path fMRI_run_RSA_instruction.py builds."""
    want = expected_settings(config)
    base = f"{data_dir}/func/RSA_{want['RDM_version']}_glmbase_{want['regression_version_full']}"
    if want['smoothing']:
        base = f"{base}_smooth{want['fwhm']}"
    return f"{base}/results"


def settings_differences(summary, want):
    """Which compared settings disagree. A key missing from the summary counts
    as a difference: an older run cannot be shown to have used these settings."""
    diffs = []
    for k in COMPARED_KEYS:
        if k not in summary:
            diffs.append(f"{k}: absent from summary")
        elif summary[k] != want[k]:
            diffs.append(f"{k}: {summary[k]!r} != {want[k]!r}")
    return diffs


def missing_inputs(data_dir, sub, config):
    """Inputs the RSA opens directly, other than the GLM itself."""
    missing = []
    EV_string = config.get("load_EVs_from")
    pkl = f"{data_dir}/beh/modelled_EVs/{sub}_modelled_EVs_{EV_string}.pkl"
    if not os.path.exists(pkl):
        missing.append(f"modelled EVs: {pkl}")
    ref = f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz"
    if not os.path.exists(ref):
        missing.append(f"reference image: {ref}")
    mask_kind = config.get("searchlight_mask", None)
    if mask_kind == 'grey_matter':
        m = f"{data_dir}/anat/grey_matter_mask_func_01.nii.gz"
    elif mask_kind == 'no_CSF':
        m = f"{data_dir}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz"
    else:
        m = None
    if m and not os.path.exists(m):
        missing.append(f"searchlight mask: {m}")
    return missing


def check_one_rsa(sub_tag, glm, config):
    """(status, detail) for one (subject, epoch)."""
    sub = f"sub-{sub_tag}"
    data_dir = f"{glmcheck.data_dir_deriv}/{sub}"

    # 1. the GLM the RSA reads, both task halves, via the FEAT audit itself
    for th in (1, 2):
        status, detail, _ = glmcheck.check_one(sub_tag, th, glm)
        if status not in GLM_READY:
            if status == 'PROMOTE_TWIN':
                detail = ("the finished run is in a '+' twin, the base directory "
                          "the RSA reads is not usable -- run the FEAT cleanup first")
            return 'GLM_NOT_READY', f"pt{th} {status}: {detail}"

    # 2. everything else the script opens
    missing = missing_inputs(data_dir, sub, config)
    if missing:
        return 'MISSING_INPUT', '; '.join(missing)

    # 3. is there already a finished result, and was it made like this?
    summary_path = f"{results_dir_for(data_dir, config)}/{sub}_settings_summary.json"
    if not os.path.exists(summary_path):
        return 'READY', ''
    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except (ValueError, OSError) as e:
        return 'RERUN_CHANGED', f"summary unreadable ({e}), treating as not done"
    diffs = settings_differences(summary, expected_settings(config))
    if diffs:
        return 'RERUN_CHANGED', '; '.join(diffs[:3]) + (
            f" (+{len(diffs) - 3} more)" if len(diffs) > 3 else "")
    return 'DONE', summary_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-config', default='rsa_instruction_cumulative_rew.json',
                    help='RSA config the per-epoch snapshots are derived from')
    ap.add_argument('--ev-config', default='EV_config_instruction.json',
                    help='EV config the epoch names come from')
    ap.add_argument('--epochs', nargs='+', default=None,
                    help='full GLM names, e.g. instr_see-A-first. Default: all from --ev-config')
    ap.add_argument('--subjects', nargs='+', default=glmcheck.DEFAULT_SUBJECTS)
    ap.add_argument('--data-dir', default=None,
                    help='derivatives directory. Default: laptop path if it exists, else cluster')
    ap.add_argument('--config-dir', default=None,
                    help='where the per-epoch config snapshots are written. '
                         'Default: the repo condition_files directory')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--skip-changed', action='store_true',
                    help='do NOT resubmit results whose settings differ (default is to '
                         'resubmit them, which overwrites the old maps)')
    ap.add_argument('--no-write', action='store_true', help='print only, write nothing')
    args = ap.parse_args()

    if args.data_dir:
        glmcheck.data_dir_deriv = args.data_dir.rstrip('/')
    config_dir = args.config_dir or glmcheck.config_path
    stamp = datetime.date.today().isoformat()

    with open(f"{config_dir}/{args.base_config}") as f:
        base_config = json.load(f)
    name_RSA = base_config.get("name_of_RSA")
    out_dir = args.out_dir or f"{glmcheck.data_dir_deriv}/group/rsa_audit_{name_RSA}_{stamp}"

    epochs = args.epochs or glmcheck.glm_names_from_config(args.ev_config)
    print(f"RSA '{name_RSA}' from {args.base_config}")
    print(f"{len(epochs)} epoch(s) x {len(args.subjects)} subject(s) = "
          f"{len(epochs) * len(args.subjects)} jobs")
    print(f"under {glmcheck.data_dir_deriv}\n")

    # One config snapshot per epoch: the base config pointed at that epoch's GLM.
    # TR = None makes the RSA treat regression_version as the full GLM name.
    epoch_configs = {}
    for glm in epochs:
        epoch_tag = glm.split('_', 1)[1] if '_' in glm else glm
        cfg = dict(base_config)
        cfg['regression_version'] = glm
        cfg['TR'] = None
        epoch_configs[glm] = (f"{os.path.splitext(args.base_config)[0]}_{epoch_tag}.json", cfg)

    rows, todo, counts_all = [], [], {}
    for glm in epochs:
        cfg_name, cfg = epoch_configs[glm]
        counts = {}
        for sub in args.subjects:
            status, detail = check_one_rsa(sub, glm, cfg)
            counts[status] = counts.get(status, 0) + 1
            counts_all[status] = counts_all.get(status, 0) + 1
            if status != 'DONE':
                rows.append((glm, sub, status, detail))
            if status == 'READY' or (status == 'RERUN_CHANGED' and not args.skip_changed):
                todo.append((sub, cfg_name))
        n_ready = counts.get('READY', 0) + (0 if args.skip_changed else counts.get('RERUN_CHANGED', 0))
        other = ', '.join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"  {glm:<40} submit {n_ready:>3}/{len(args.subjects)}   [{other}]")

    print(f"\n=== {len(todo)} job(s) to submit ===")
    if counts_all.get('DONE'):
        print(f"    {counts_all['DONE']} already done with these exact settings -- skipped")
    if counts_all.get('RERUN_CHANGED'):
        what = "SKIPPED (--skip-changed)" if args.skip_changed else "WILL BE OVERWRITTEN"
        print(f"    {counts_all['RERUN_CHANGED']} existing result(s) built with different "
              f"settings -- {what}")
    for glm, sub, status, detail in rows:
        if status == 'RERUN_CHANGED':
            print(f"        sub-{sub} {glm}: {detail}")
            break
    if counts_all.get('GLM_NOT_READY'):
        print(f"    {counts_all['GLM_NOT_READY']} blocked: the epoch GLM is not complete "
              f"(run check_GLMs_ran.py)")
    if counts_all.get('MISSING_INPUT'):
        print(f"    {counts_all['MISSING_INPUT']} blocked: inputs missing")

    by_status = {}
    for glm, sub, status, detail in rows:
        by_status.setdefault(status, []).append((glm, sub, detail))
    for status in sorted(by_status):
        rws = by_status[status]
        print(f"\n{status} ({len(rws)}):")
        for glm, sub, detail in rws[:10]:
            print(f"    sub-{sub} {glm}" + (f"  --  {detail}" if detail else ""))
        if len(rws) > 10:
            print(f"    ... and {len(rws) - 10} more (see report.txt)")

    if args.no_write:
        return 0

    os.makedirs(out_dir, exist_ok=True)
    for cfg_name, cfg in epoch_configs.values():
        with open(f"{config_dir}/{cfg_name}", 'w') as f:
            json.dump(cfg, f, indent=2)

    with open(f"{out_dir}/todo_rsa.txt", 'w') as f:
        f.write("# subject epoch_config -- RSA jobs still to run.\n")
        f.write("# feed to: bash submit_RSA_instruction_epochs.sh todo_rsa.txt\n")
        for sub, cfg_name in todo:
            f.write(f"{sub} {cfg_name}\n")

    with open(f"{out_dir}/report.txt", 'w') as f:
        f.write(f"RSA audit {stamp} -- {name_RSA} ({args.base_config})\n")
        f.write(f"{len(todo)} job(s) to submit\n\nsub\tepoch\tstatus\tdetail\n")
        for glm, sub, status, detail in rows:
            f.write(f"sub-{sub}\t{glm}\t{status}\t{detail}\n")

    with open(f"{out_dir}/settings.json", 'w') as f:
        json.dump({'date': stamp, 'base_config': args.base_config,
                   'data_dir_deriv': glmcheck.data_dir_deriv,
                   'epochs': epochs, 'subjects': args.subjects,
                   'skip_changed': args.skip_changed,
                   'compared_keys': COMPARED_KEYS,
                   'expected_settings': expected_settings(base_config),
                   'counts': counts_all, 'n_todo': len(todo)}, f, indent=2)

    print(f"\nwritten to {out_dir}:")
    print(f"    todo_rsa.txt   {len(todo)} job(s)")
    print(f"    report.txt     why each of the others is not being submitted")
    print(f"    settings.json  what this audit was run with")
    print(f"    ({len(epoch_configs)} per-epoch config snapshot(s) in {config_dir})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
