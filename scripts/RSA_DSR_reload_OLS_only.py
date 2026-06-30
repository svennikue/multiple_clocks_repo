#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reload an existing RSA run and re-run ONLY the OLS step.

The slow parts of the main RSA pipeline are (a) loading every subject's
single-unit data into memory, and (b) building the 1000 permuted data
RDMs per ROI. Both are deterministically cached on disk by
`RSA_DSR_ROIs_simple.py`:

* model + empirical data RDMs   →  <run>/rdms/rdms_<ROI>.npz
* permuted data RDMs            →  <run>/perm_data_rdms/perm_data_rdms_<ROI>.pkl

This standalone script reads those caches and re-runs the empirical
and permutation OLS using the same numerical conventions as the main
script (`evaluate_model` for empirical, normalised vectorised OLS for
perms — both z-score regressors and target, matching what
`mc.analyse.my_RSA.evaluate_model` does). The output overwrites the
run's `results_summary.csv` and `results_summary_combos.csv` so any
downstream FDR / plotting code that reads those files picks up the
new values.

USAGE
-----
Edit ``RSA_RUN_DIR`` and ``OVERWRITE_CSV`` below, then:

    python scripts/RSA_DSR_reload_OLS_only.py

Set ``OVERWRITE_CSV = False`` to save to a sibling sub-folder for
side-by-side comparison rather than overwriting.

NOT covered
-----------
* Diagnostic plots (RDM heatmaps, perm-null histograms, glassbrains,
  pub figures) are NOT re-rendered — those depend on per-cell data
  this script never sees. If you need them, run the full pipeline.
* The script reads `combo_models` and `tests` from the saved config of
  the reload run, so the resulting CSVs match the original combo
  structure. Override the constants at the top to change them.

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
sys.path.insert(0, str(REPO))

# Shared standardised-OLS function — used identically for empirical and
# permutation fits (CLAUDE.md rule #4).
from mc.analyse.my_RSA import evaluate_model_vec


# ── User settings ─────────────────────────────────────────────────────
DATA_DIR = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives')
OUT_BASE = DATA_DIR / 'group/DSR_RSA_simple_ROI'

# Point at the run whose saved RDMs you want to reload + re-fit.
RSA_RUN_DIR = OUT_BASE / '2026-06-29_13-13-23'

# True → overwrite the run's results_summary*.csv with the new OLS output.
# False → write to <RSA_RUN_DIR>/reload_OLS_<timestamp>/.
OVERWRITE_CSV = True

# Override the combos / tests defined in the saved config. None = read
# from <RSA_RUN_DIR>/config.json.
COMBO_MODELS_OVERRIDE = None
TESTS_OVERRIDE = None


# All OLS goes through the SAME shared function for empirical and perm
# (CLAUDE.md rule #4 — imported above as `evaluate_model_vec`).


# ── Main ─────────────────────────────────────────────────────────────
def main():
    if not RSA_RUN_DIR.exists():
        sys.exit(f"ERROR: RSA_RUN_DIR not found: {RSA_RUN_DIR}")

    cfg_path = RSA_RUN_DIR / 'config.json'
    if not cfg_path.exists():
        sys.exit(f"ERROR: no config.json in {RSA_RUN_DIR}")
    with open(cfg_path) as f:
        run_cfg = json.load(f)
    print(f"Reloading run: {RSA_RUN_DIR.name}")
    print(f"  original run timestamp: {run_cfg.get('timestamp')}")

    combo_models = COMBO_MODELS_OVERRIDE or run_cfg['combo_models']
    models       = run_cfg['models']
    tests        = TESTS_OVERRIDE or ['split_halves', 'split_halves_z',
                                       'between_tasks', 'between_tasks_z']
    rois         = run_cfg.get('rois', [
        'ACC', 'EC', 'HC_anterior', 'HC_mid', 'PCC',
        'Parahippocampal', 'medialOFC',
    ])
    print(f"  rois: {rois}")
    print(f"  models: {models}")
    print(f"  combos: {list(combo_models)}")
    print(f"  tests:  {tests}")

    # Output location
    if OVERWRITE_CSV:
        out_dir = RSA_RUN_DIR
    else:
        out_dir = (RSA_RUN_DIR
                   / f'reload_OLS_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  writing to: {out_dir}")

    summary_rows       = []
    summary_combo_rows = []

    for roi in rois:
        # Load saved RDMs (model + data, all test variants)
        rdms_path = RSA_RUN_DIR / 'rdms' / f'rdms_{roi}.npz'
        if not rdms_path.exists():
            print(f"  [{roi}] skipping — no rdms file at {rdms_path}")
            continue
        with np.load(rdms_path) as nz:
            n_neurons = int(nz['__n_neurons__'])
            # data RDMs per test
            data_per_test = {
                'split_halves':    np.asarray(nz['data__split_halves'],     dtype=float),
                'split_halves_z':  np.asarray(nz['data__split_halves_z'],   dtype=float),
                'between_tasks':   np.asarray(nz['data__between_tasks'],    dtype=float),
                'between_tasks_z': np.asarray(nz['data__between_tasks_z'],  dtype=float),
            }
            # model RDMs per test
            model_per_test = {'split_halves': {}, 'between_tasks': {}}
            for k in nz.files:
                if k.startswith('model__split_halves__'):
                    m = k.replace('model__split_halves__', '')
                    model_per_test['split_halves'][m] = np.asarray(nz[k], dtype=float)
                elif k.startswith('model__between_tasks__'):
                    m = k.replace('model__between_tasks__', '')
                    model_per_test['between_tasks'][m] = np.asarray(nz[k], dtype=float)

        # Load perm-data RDMs (z-only)
        perm_pkl_path = (RSA_RUN_DIR / 'perm_data_rdms'
                          / f'perm_data_rdms_{roi}.pkl')
        if not perm_pkl_path.exists():
            print(f"  [{roi}] skipping — no perm pickle at {perm_pkl_path}")
            continue
        with open(perm_pkl_path, 'rb') as f:
            perm_data = pickle.load(f)
        perm_rdms_z = perm_data['perms']   # {'split_halves_z': (n_perms,n_pairs), 'between_tasks_z': ...}
        n_perms_actual = next(iter(perm_rdms_z.values())).shape[0]
        print(f"  [{roi}] n_cells={n_neurons}  n_perms={n_perms_actual}")

        for test_name in tests:
            # Pick the data RDM and model dict for this test
            base_test = test_name[:-2] if test_name.endswith('_z') else test_name
            y_emp = data_per_test[test_name]
            mdict = model_per_test[base_test]
            has_perms = test_name in perm_rdms_z
            Y_perms = perm_rdms_z[test_name] if has_perms else None

            # ── Single models ──────────────────────────────────────────────
            for m in models:
                if m not in mdict:
                    continue
                # Empirical fit — shared function
                t_emp, b_emp, p_emp = evaluate_model_vec(mdict[m], y_emp)
                t_e = float(np.asarray(t_emp).ravel()[0])
                b_e = float(np.asarray(b_emp).ravel()[0])
                p_e = float(np.asarray(p_emp).ravel()[0])
                if has_perms:
                    # SAME shared function on the perm stack
                    _, BETA_PERMS, _ = evaluate_model_vec(mdict[m], Y_perms)
                    perm_betas = np.asarray(BETA_PERMS).ravel()
                    valid = np.isfinite(perm_betas)
                    if valid.any() and np.isfinite(b_e):
                        p_perm = ((np.sum(perm_betas[valid] >= b_e) + 1)
                                   / (valid.sum() + 1))
                    else:
                        p_perm = np.nan
                else:
                    p_perm = np.nan
                summary_rows.append({
                    'roi':       roi,
                    'n_neurons': n_neurons,
                    'test':      test_name,
                    'model':     m,
                    't':         t_e,
                    'beta':      b_e,
                    'p_param':   p_e,
                    'p_perm':    p_perm,
                })

            # ── Combos ─────────────────────────────────────────────────────
            for combo_name, sub_models in combo_models.items():
                # Drop missing regressors gracefully
                sub_models = [m for m in sub_models if m in mdict]
                if len(sub_models) < 2:
                    continue
                X = np.stack([mdict[m] for m in sub_models], axis=1)
                # Empirical fit — shared function (1-D Y → 1-D outputs)
                t_emp_arr, b_emp_arr, p_emp_arr = evaluate_model_vec(X, y_emp)
                if has_perms:
                    # SAME shared function on the perm stack →
                    # (n_perms, n_features)
                    T_PERMS, BETA_PERMS, _ = evaluate_model_vec(X, Y_perms)
                else:
                    BETA_PERMS = None
                for sub_idx, sub_model in enumerate(sub_models):
                    b_e = float(b_emp_arr[sub_idx])
                    if has_perms and BETA_PERMS is not None:
                        perm_betas = BETA_PERMS[:, sub_idx]
                        valid = np.isfinite(perm_betas)
                        if valid.any() and np.isfinite(b_e):
                            p_perm = ((np.sum(perm_betas[valid] >= b_e) + 1)
                                       / (valid.sum() + 1))
                        else:
                            p_perm = np.nan
                    else:
                        p_perm = np.nan
                    summary_combo_rows.append({
                        'roi':       roi,
                        'n_neurons': n_neurons,
                        'test':      test_name,
                        'combo':     combo_name,
                        'sub_model': sub_model,
                        't':         float(t_emp_arr[sub_idx]),
                        'beta':      b_e,
                        'p_param':   float(p_emp_arr[sub_idx]),
                        'p_perm':    p_perm,
                    })

    sdf = pd.DataFrame(summary_rows)
    scdf = pd.DataFrame(summary_combo_rows)

    out_singles = out_dir / 'results_summary.csv'
    out_combos  = out_dir / 'results_summary_combos.csv'
    sdf.to_csv(out_singles, index=False)
    scdf.to_csv(out_combos, index=False)
    print(f"\nWrote {out_singles}  ({len(sdf)} rows)")
    print(f"Wrote {out_combos}  ({len(scdf)} rows)")

    # Print ACC DSR results so you can eyeball them against the OLD numbers
    print("\n=== ACC DSR results (split_halves_z) ===")
    show = scdf[(scdf.roi == 'ACC') & (scdf.test == 'split_halves_z')
                 & scdf.sub_model.str.startswith('dsr_')]
    print(show[['combo', 'sub_model', 'beta', 't', 'p_param', 'p_perm']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
