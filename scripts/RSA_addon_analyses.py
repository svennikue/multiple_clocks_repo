#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flexible add-on analyses for a finished RSA_DSR_ROIs_simple.py run.

Loads only the saved artefacts of an existing run directory:
    {RUN_DIR}/config.json
    {RUN_DIR}/perm_null_draws/perm_{ROI}.pkl
    {RUN_DIR}/rdms/rdms_{ROI}.npz       (saved by RSA_DSR_ROIs_simple.py
                                          after 2026-06-22; older runs may
                                          need a single re-run with the
                                          current script to populate)
and runs add-on analyses on them WITHOUT touching the raw cell data.

Outputs are written back into {RUN_DIR}/addon/ so they stay attached to the
source run.

Currently provided add-ons:
    * phase_mask_comparison    re-evaluate every (combo, test) under the three
                               phase-mask modes (full / within_phase /
                               across_phase) and save the long-form
                               phase_mask_replay.csv

Add new add-ons by defining a function `addon_<name>(ctx)` (ctx is a dict
with everything pre-loaded) and listing it in ADDONS at the top.

Run
---
    python scripts/RSA_addon_analyses.py

Edit RUN_DIR and ADDONS at the top of the file.

@author: Svenja Kuchenhoff
"""

import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo")
import mc.analyse.my_RSA  # for evaluate_model fallback


# ── Settings ──────────────────────────────────────────────────────────
RUN_DIR = Path(
    "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/"
    "derivatives/group/DSR_RSA_simple_ROI/2026-06-22_17-49-19"
)
# Which add-ons to run. Each name must match a `addon_<name>` function below.
ADDONS = [
    "phase_mask_comparison",
]
# If you only want a subset of ROIs (else use whatever the run dir has).
ROIS = None     # e.g. ['ACC'] or None


# ── Context loading ──────────────────────────────────────────────────
def _safe_scalar(arr):
    a = np.asarray(arr)
    return a.item() if a.ndim == 0 else a.tolist()


def load_context(run_dir, rois=None):
    """Load everything an add-on needs into one dict.

    Returns
    -------
    ctx : dict
        Keys:
          run_dir, addon_dir (created), config (dict from config.json),
          rois (list of str), per_roi[roi] = {
              perm   : dict (loaded from perm_{ROI}.pkl)
              rdms   : dict[str -> np.ndarray] (from rdms_{ROI}.npz)
              empirical, empirical_z (subset of perm), combos, tests
          }
    """
    run_dir = Path(run_dir)
    assert run_dir.is_dir(), f"Run dir missing: {run_dir}"
    cfg_path = run_dir / "config.json"
    assert cfg_path.exists(), f"Missing {cfg_path}"
    config = json.loads(cfg_path.read_text())

    pkl_dir = run_dir / "perm_null_draws"
    rdm_dir = run_dir / "rdms"
    if not rdm_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {rdm_dir}. The current script saves RDMs as "
            f"rdms/rdms_<ROI>.npz on every fresh run. If this run pre-dates "
            f"that change, re-run RSA_DSR_ROIs_simple.py once "
            f"(N_PERMUTATIONS=None for speed) to populate them."
        )
    have_pickles = pkl_dir.is_dir() and bool(list(pkl_dir.glob("perm_*.pkl")))
    if not have_pickles:
        print(f"  [info] no perm pickles in {pkl_dir} (or dir missing) — "
              f"falling back to config.json for metadata. Add-ons that "
              f"need the perm null draws will be skipped automatically.")

    # Discover ROIs from npz filenames (always present) and intersect with pickles if any
    npz_rois = sorted(p.stem.replace("rdms_", "") for p in rdm_dir.glob("rdms_*.npz"))
    if not npz_rois:
        raise FileNotFoundError(f"No rdms_{{ROI}}.npz files in {rdm_dir}")
    rois = rois or npz_rois

    # Config-derived metadata fallback when perm pickle is absent
    cfg_models       = config.get("models",       [])
    cfg_combo_models = config.get("combo_models", {})
    cfg_tests        = ["split_halves", "split_halves_z",
                        "between_tasks", "between_tasks_z"]

    per_roi = {}
    for roi in rois:
        npz_p = rdm_dir / f"rdms_{roi}.npz"
        if not npz_p.exists():
            print(f"  [warn] skipping {roi}: no {npz_p}")
            continue
        npz = np.load(npz_p, allow_pickle=True)
        rdms = {k: npz[k] for k in npz.files}

        perm = None
        n_neurons   = int(rdms["__n_neurons__"].item()) if "__n_neurons__" in rdms else None
        n_perms     = None
        tests       = cfg_tests
        models      = cfg_models
        combo_models = cfg_combo_models
        if have_pickles:
            pkl_p = pkl_dir / f"perm_{roi}.pkl"
            if pkl_p.exists():
                with open(pkl_p, "rb") as f:
                    perm = pickle.load(f)
                n_neurons    = perm.get("n_neurons", n_neurons)
                n_perms      = perm.get("n_permutations")
                tests        = perm.get("tests", tests)
                models       = perm.get("models", models)
                combo_models = perm.get("combo_models", combo_models)
            else:
                print(f"  [warn] {roi}: no perm pickle, using config metadata.")

        per_roi[roi] = {
            "perm":         perm,
            "rdms":         rdms,
            "n_neurons":    n_neurons,
            "n_perms":      n_perms,
            "tests":        tests,
            "models":       models,
            "combo_models": combo_models,
        }
        print(f"  loaded {roi}: n_neurons={n_neurons}, n_perms={n_perms}, "
              f"models={len(models or [])}, combos={len(combo_models or {})}, "
              f"rdm arrays={len(rdms)}")

    addon_dir = run_dir / "addon"
    addon_dir.mkdir(exist_ok=True)
    ctx = {
        "run_dir":   run_dir,
        "addon_dir": addon_dir,
        "config":    config,
        "rois":      sorted(per_roi.keys()),
        "per_roi":   per_roi,
    }
    return ctx


# ── Phase-mask construction (mirror of main script) ──────────────────
def make_phase_masks(n_configs, n_conds_per_config, n_phases,
                     include_diagonal=False):
    """Same definition as RSA_DSR_ROIs_simple.py make_phase_masks_for_cells."""
    n = n_configs * n_conds_per_config
    phase = np.tile(np.arange(n_conds_per_config) % n_phases, n_configs)
    k = 0 if include_diagonal else 1
    ii, jj = np.triu_indices(n, k=k)
    same_phase = phase[ii] == phase[jj]
    between_block = (ii // n_conds_per_config) != (jj // n_conds_per_config)
    split_halves = {
        "full":         np.ones_like(same_phase, dtype=bool),
        "within_phase": same_phase,
        "across_phase": ~same_phase,
    }
    same_phase_bt = same_phase[between_block]
    between_tasks = {
        "full":         np.ones_like(same_phase_bt, dtype=bool),
        "within_phase": same_phase_bt,
        "across_phase": ~same_phase_bt,
    }
    return {"split_halves": split_halves, "between_tasks": between_tasks}


# ── Add-on: phase-mask comparison ────────────────────────────────────
def addon_phase_mask_comparison(ctx):
    """For every (ROI, test, combo, sub_model) re-evaluate the regression
    under each of the three phase-mask modes (full / within_phase /
    across_phase). Writes phase_mask_replay.csv into {RUN_DIR}/addon/.
    """
    print("\n[add-on] phase_mask_comparison")
    cfg = ctx["config"]
    n_configs        = int(_safe_scalar(cfg["N_CONFIGS"]))
    n_conds_per_conf = int(_safe_scalar(cfg["N_CONDS_PER_CONF"]))
    n_phases         = int(_safe_scalar(cfg["N_PHASES"]))
    masks = make_phase_masks(n_configs, n_conds_per_conf, n_phases)

    ALL_MODES = ("full", "within_phase", "across_phase")
    rows = []

    for roi, payload in ctx["per_roi"].items():
        rdms = payload["rdms"]
        combos = payload["combo_models"] or {}
        # Tests that have a phase mask defined.
        for test_name in payload["tests"]:
            # Strip trailing _z to map to the test family.
            base = test_name[:-2] if test_name.endswith("_z") else test_name
            if base not in masks:
                continue
            data_key  = f"data__{test_name}"
            model_pre = f"model__{base}__"
            if data_key not in rdms:
                print(f"  [warn] {roi}/{test_name}: no {data_key} in npz, skip")
                continue
            data_vec = rdms[data_key]
            for mode in ALL_MODES:
                mask = masks[base][mode]
                d_m = data_vec[mask]
                if d_m.size < 3 or np.nanstd(d_m) < 1e-12:
                    continue
                for combo_name, sub_models in combos.items():
                    # Stack the per-sub-model RDM columns under this mask
                    stacked_cols = []
                    miss = []
                    for sm in sub_models:
                        key = f"{model_pre}{sm}"
                        if key not in rdms:
                            miss.append(sm); continue
                        stacked_cols.append(rdms[key][mask])
                    if miss:
                        print(f"  [warn] {roi}/{test_name}/{combo_name}: missing sub-models {miss}, skip")
                        continue
                    X = np.column_stack(stacked_cols)
                    if not np.isfinite(X).all() or not np.isfinite(d_m).all():
                        # mirror main-script defensive guard
                        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                        d_m = np.nan_to_num(d_m, nan=0.0, posinf=0.0, neginf=0.0)
                    try:
                        t_arr, beta_arr, p_arr = mc.analyse.my_RSA.evaluate_model(X, d_m)
                    except Exception as exc:
                        print(f"  [skip] {roi}/{test_name}/{combo_name}/{mode}: {exc!r}")
                        continue
                    t_arr    = np.asarray(t_arr, dtype=float).ravel()
                    beta_arr = np.asarray(beta_arr, dtype=float).ravel()
                    p_arr    = np.asarray(p_arr, dtype=float).ravel()
                    n_pairs_kept = int(mask.sum())
                    for sub_idx, sm in enumerate(sub_models):
                        rows.append({
                            "roi":           roi,
                            "test":          test_name,
                            "mode":          mode,
                            "combo":         combo_name,
                            "sub_model":     sm,
                            "beta":          float(beta_arr[sub_idx]),
                            "t":             float(t_arr[sub_idx]),
                            "p_param":       float(p_arr[sub_idx]),
                            "n_pairs_kept":  n_pairs_kept,
                            "n_pairs_total": int(mask.size),
                            "n_neurons":     int(payload["n_neurons"]),
                        })

    if not rows:
        print("  [phase_mask_comparison] no rows produced (empty inputs?)")
        return None

    df = pd.DataFrame(rows)
    out_csv = ctx["addon_dir"] / "phase_mask_replay.csv"
    df.to_csv(out_csv, index=False)
    print(f"  -> {out_csv}  ({len(df)} rows)")

    # Headline print: ACC dsr-family sub-models across modes (if present)
    show = df[
        (df["sub_model"].str.startswith("dsr"))
        & (df["test"] == "split_halves_z")
    ]
    if not show.empty:
        print("\n  ACC dsr sub-models (split_halves_z) across modes:")
        piv = show[show["roi"] == "ACC"].pivot_table(
            index=["combo", "sub_model"], columns="mode",
            values="beta", aggfunc="first",
        )
        if not piv.empty:
            print(piv.to_string(float_format=lambda v: f"{v:+.4f}"))
    return df


# ── Dispatch ──────────────────────────────────────────────────────────
ADDON_REGISTRY = {
    "phase_mask_comparison": addon_phase_mask_comparison,
}


def main():
    print(f"Add-on analyses on {RUN_DIR}")
    print(f"Loading context...")
    ctx = load_context(RUN_DIR, rois=ROIS)
    print(f"\nROIs found: {ctx['rois']}")
    print(f"Running add-ons: {ADDONS}")

    # Log the run
    log_path = ctx["addon_dir"] / "addon_log.md"
    timestamp = datetime.now().isoformat(timespec="seconds")
    with log_path.open("a") as f:
        f.write(f"\n## {timestamp}\n- rois: {ctx['rois']}\n- addons: {ADDONS}\n")

    for name in ADDONS:
        if name not in ADDON_REGISTRY:
            print(f"\n[warn] unknown add-on '{name}', skipping. Known: {sorted(ADDON_REGISTRY)}")
            continue
        ADDON_REGISTRY[name](ctx)

    print("\nDone.")


if __name__ == "__main__":
    main()
