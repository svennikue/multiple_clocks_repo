#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regressor collinearity for every combo GLM, fMRI or cells.

Replaces plot_model_RDM_correlations.py. All the model-vector building and
correlation maths now lives in `mc.analyse.model_rdm_correlations`; the
heatmap is `mc.plotting.results.plot_model_correlation_matrix_pub`. This
file is only the driver.

What is correlated is the design matrix `X` that actually reaches
`mc.analyse.my_RSA.evaluate_model` -- after the condition/phase mask the
pipeline applies and after degenerate columns are dropped. Nothing is
recomputed or reweighted for the figure.

  fMRI   one matrix per (combo), Fisher-z averaged over subjects. Model
         RDMs are rebuilt per subject from that subject's EV pickle.
  cells  one matrix per (combo x ROI), from the model vectors saved in
         `rdms/rdms_<ROI>.npz`. Model geometries are simulated from the
         modal route of the sessions contributing to each region, so they
         genuinely differ between ROIs -- hence one matrix per ROI.

USAGE
-----
    python model_rdm_correlations.py --source fmri
        [--config rsa_config_quarters_DSR_controls.json] [--combo NAME]
    python model_rdm_correlations.py --source cells
        [--run 2026-08-27_19-18-20] [--combo NAME] [--roi mPFC]

@author: Svenja Kuchenhoff, 2026
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
import mc
from mc.analyse import model_rdm_correlations as mrc

SOURCE_DIR = "/Users/xpsy1114/Documents/projects/multiple_clocks"
if not os.path.isdir(SOURCE_DIR):
    SOURCE_DIR = "/home/fs0/xpsy1114/scratch"
DERIV = f"{SOURCE_DIR}/data/derivatives"
CONFIG_DIR = f"{SOURCE_DIR}/multiple_clocks_repo/condition_files"
CELL_RUNS = f"{SOURCE_DIR}/data/ephys_humans/derivatives/group/DSR_RSA_simple_ROI"

FIG_WIDTH_CM, FIG_HEIGHT_CM, FIG_FONT_PT = 4.0, 4.0, 7


def _save(out_dir, stem, corr, models, settings, sd=None, per_subject=None):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(corr, index=models, columns=models).round(4).to_csv(
        f"{out_dir}/{stem}_mean.csv")
    if sd is not None:
        pd.DataFrame(sd, index=models, columns=models).round(4).to_csv(
            f"{out_dir}/{stem}_sd.csv")
    if per_subject is not None:
        np.save(f"{out_dir}/{stem}_per_subject.npy", per_subject)
    with open(f"{out_dir}/{stem}_settings.json", "w") as f:
        json.dump(settings, f, indent=2)
    mc.plotting.results.plot_model_correlation_matrix_pub(
        corr, mrc.display_labels(models),
        save_stem=f"{out_dir}/{stem}",
        fig_width_cm=FIG_WIDTH_CM, fig_height_cm=FIG_HEIGHT_CM,
        font_pt=FIG_FONT_PT, show=False)
    print(f"    -> {out_dir}/{stem}.pdf")


def run_fmri(config_file, only_combo=None):
    with open(f"{CONFIG_DIR}/{config_file}") as f:
        config = json.load(f)
    EV_string = config.get("load_EVs_from")
    include_diagonal = config.get("diagonal_included", True)
    masked_conditions = config.get("masked_conds", None)

    combos = [c for c in config["combo_models"]
              if only_combo is None or c["name"] == only_combo]
    if not combos:
        raise SystemExit(f"combo {only_combo!r} not in {config_file}")

    today = date.today().strftime("%d-%m-%Y")
    out_dir = f"{DERIV}/group/model_RDM_correlations_fMRI_{today}"

    ev_paths = []
    for sub in sorted(os.listdir(DERIV)):
        pkl = f"{DERIV}/{sub}/beh/modelled_EVs/{sub}_modelled_EVs_{EV_string}.pkl"
        if sub.startswith('sub-') and os.path.exists(pkl):
            ev_paths.append((sub, pkl))
    print(f"fMRI: {len(ev_paths)} subjects, {len(combos)} combo(s)")

    for combo in combos:
        models = list(combo["regressors"])
        name = combo["name"]
        print(f"\n  [{name}] {models}")
        mats, subs, kept_ref, mask_name, n_total = [], [], None, None, None
        for sub, pkl in ev_paths:
            with open(pkl, 'rb') as fh:
                model_EVs = pickle.load(fh)
            X, kept, mask_name, n_total = mrc.fmri_combo_design(
                model_EVs, models, include_diagonal=include_diagonal,
                masked_conditions=masked_conditions)
            if kept_ref is None:
                kept_ref = kept
            elif kept != kept_ref:
                print(f"    [warn] {sub}: kept columns differ ({kept}); skipped")
                continue
            mats.append(mrc.correlation_matrix(X))
            subs.append(sub)
        stack = np.array(mats)
        corr = mrc.fisher_mean(stack)
        print(pd.DataFrame(corr, index=kept_ref, columns=kept_ref).round(3).to_string())
        _save(out_dir, f"corr_{name}", corr, kept_ref,
              {"source": "fmri", "config_file": config_file, "combo": name,
               "regressors_requested": models, "regressors_used": kept_ref,
               "EV_string": EV_string, "diagonal_included": include_diagonal,
               "condition_mask": mask_name, "n_RDM_cells_used": int(stack.shape[0] and X.shape[0]),
               "n_RDM_cells_total": n_total, "subjects": subs,
               "n_subjects": len(subs),
               "group_average": "Fisher-z mean of per-subject Pearson correlations"},
              sd=np.nanstd(stack, axis=0), per_subject=stack)


def run_cells(run_tag, only_combo=None, only_roi=None):
    run_dir = f"{CELL_RUNS}/{run_tag}"
    with open(f"{run_dir}/config.json") as f:
        cfg = json.load(f)
    test_name = cfg.get("fdr_test", "split_halves_z")
    combos = {k: v for k, v in cfg["combo_models"].items()
              if only_combo is None or k == only_combo}
    rois = [r for r in cfg["rois"] if only_roi is None or r == only_roi]
    print(f"cells: run {run_tag}, test {test_name}, "
          f"{len(combos)} combo(s) x {len(rois)} ROI(s)")

    out_dir = f"{run_dir}/model_RDM_correlations"
    for roi in rois:
        npz_path = f"{run_dir}/rdms/rdms_{roi}.npz"
        if not os.path.exists(npz_path):
            print(f"  [skip] no rdms for {roi}")
            continue
        npz = np.load(npz_path, allow_pickle=True)
        for name, models in combos.items():
            print(f"\n  [{roi} / {name}] {models}")
            X, kept, n_total = mrc.cells_combo_design(
                npz, list(models), test_name=test_name, phase_mask=None)
            corr = mrc.correlation_matrix(X)
            print(pd.DataFrame(corr, index=kept, columns=kept).round(3).to_string())
            _save(out_dir, f"corr_{roi}_{name}", corr, kept,
                  {"source": "cells", "run_tag": run_tag, "roi": roi,
                   "combo": name, "regressors_requested": list(models),
                   "regressors_used": kept, "test": test_name,
                   "phase_mask_mode": cfg.get("phase_mask_mode"),
                   "n_RDM_cells_used": int(X.shape[0]),
                   "n_RDM_cells_total": int(n_total),
                   "n_neurons": int(npz['__n_neurons__'])})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["fmri", "cells"], required=True)
    ap.add_argument("--config", default="rsa_config_quarters_DSR_controls.json")
    ap.add_argument("--run", default=None,
                    help="cell run tag; default = newest under DSR_RSA_simple_ROI")
    ap.add_argument("--combo", default=None, help="single combo (default: all)")
    ap.add_argument("--roi", default=None, help="cells only: single ROI")
    a = ap.parse_args()

    if a.source == "fmri":
        run_fmri(a.config, a.combo)
    else:
        tag = a.run
        if tag is None:
            tags = [d for d in os.listdir(CELL_RUNS)
                    if os.path.isdir(f"{CELL_RUNS}/{d}")
                    and os.path.exists(f"{CELL_RUNS}/{d}/config.json")]
            tag = sorted(tags)[-1]
            print(f"(no --run given, using newest: {tag})")
        run_cells(tag, a.combo, a.roi)
