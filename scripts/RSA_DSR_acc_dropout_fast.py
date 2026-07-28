#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast ACC control-dropout diagnostic for the DSR RSA.

Reads the perm-data-RDM cache and the model RDMs that the main RSA
script (`RSA_DSR_ROIs_simple.py`) already saves, and runs every
leave-N-out subset of CONTROLS × DSR_VARIANTS through a single
vectorised OLS — no permutations are recomputed.

For every (control-subset × DSR-variant) cell of the diagnostic table
this script returns:
    β_DSR, t_DSR, p_perm_DSR

`p_perm` is computed by comparing the empirical β against the same
``(n_perms,)`` distribution of permuted-data-RDM betas that arises from
projecting the model design onto the cached perm-RDM stack.

EDIT and re-run with no constraints — runtime should be seconds.

Inputs
------
RSA_RUN_DIR : path to a main RSA run dir that contains:
    rdms_<ROI>.npz                 — model + data RDMs per test variant
    perm_data_rdms_<ROI>.pkl       — cached perm RDMs (built by the
                                      main RSA via mc.analyse.rsa_perm_rdms)

Outputs
-------
<RSA_RUN_DIR>/control_dropout_<ROI>_<run_tag>/
    all_combos_results.csv
    <ROI>_pivot.csv
    <ROI>_dropout_heatmap.{pdf, png}
    config.json

@author: Svenja Küchenhoff
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Shared standardised-OLS function — used for both empirical fits and
# permutation null fits (CLAUDE.md rule #4).
import sys as _sys
_sys.path.insert(0, '/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
from mc.analyse.my_RSA import evaluate_model_vec


# ── User-configurable settings ───────────────────────────────────────
REPO     = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
DATA_DIR = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives')
# Freshest full RSA run (2026-07-26, uses v2 ROI labels — proximity cap
# + Brainnetome-first cingulate rules). To point at a different run just
# swap the folder name here.
RSA_RUN_DIR = (DATA_DIR / 'group/DSR_RSA_simple_ROI/'
                          '2026-07-23_12-07-25')

ROI          = 'ACC'
TEST_VARIANT = 'between_tasks_z'    # one of 'split_halves_z', 'between_tasks_z'

CONTROLS     = ['state', 'location', 'l2_norm',
                'bttn_curr', 'bttn_next', 'reward_path']
DSR_VARIANTS = ['dsr_fmri', 'dsr_fmri_fut', 'dsr_fmri_informed']

ALPHA = 0.05


# ── Helpers ──────────────────────────────────────────────────────────
def _bh_fdr(pvals):
    """BH-FDR adjusted q-values (NaN-safe)."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]; n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qok = np.empty(n); qok[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = qok
    return q


# OLS helpers removed — empirical and perm fits both use the shared
# mc.analyse.my_RSA.evaluate_model_vec function (CLAUDE.md rule #4).


def _stars(q):
    if not np.isfinite(q): return ''
    if q < 0.001: return '***'
    if q < 0.01:  return '**'
    if q < 0.05:  return '*'
    if q < 0.10:  return '·'
    return ''


def _enumerate_subsets(controls):
    """All subsets of `controls` (sizes 0..len), with stable label."""
    out = []
    for size in range(len(controls) + 1):
        for subset in combinations(controls, size):
            label = '_'.join(subset) if subset else 'NOCTRL'
            out.append((size, label, list(subset)))
    return out


# ── Main ─────────────────────────────────────────────────────────────
def main():
    # Load the perm pickle ----------------------------------------------
    # The main RSA writes the cache to <run>/perm_data_rdms/perm_data_rdms_<ROI>.pkl;
    # older runs may have it at the run root. Try the canonical location
    # first, then fall back.
    perm_candidates = [
        RSA_RUN_DIR / 'perm_data_rdms' / f'perm_data_rdms_{ROI}.pkl',
        RSA_RUN_DIR / f'perm_data_rdms_{ROI}.pkl',
    ]
    perm_path = next((p for p in perm_candidates if p.exists()), None)
    if perm_path is None:
        candidates_str = '\n  '.join(str(p) for p in perm_candidates)
        sys.exit(f"\nERROR: missing perm pickle. Looked for:\n  "
                 f"{candidates_str}\n"
                 f"Run RSA_DSR_ROIs_simple.py first so it builds and "
                 f"caches the perm RDMs for {ROI!r}.")
    with open(perm_path, 'rb') as f:
        perm_data = pickle.load(f)
    fp = perm_data['fingerprint']
    y_emp_full   = perm_data['empirical'][TEST_VARIANT]  # (n_pairs,)
    Y_perms_full = perm_data['perms'][TEST_VARIANT]       # (n_perms, n_pairs)
    n_perms      = Y_perms_full.shape[0]
    print(f"Loaded perm pickle: {perm_path}")
    print(f"  ROI={fp['roi']}  n_cells={fp['n_cells']}  n_perms={n_perms}  "
          f"test={TEST_VARIANT}  n_pairs={y_emp_full.size}")

    # Load model RDMs from the per-ROI npz ----------------------------
    # Canonical location is <run>/rdms/rdms_<ROI>.npz; older runs may
    # have used <run>/rdm_diagnostics/ or the run root.
    rdms_candidates = [
        RSA_RUN_DIR / 'rdms' / f'rdms_{ROI}.npz',
        RSA_RUN_DIR / 'rdm_diagnostics' / f'rdms_{ROI}.npz',
        RSA_RUN_DIR / f'rdms_{ROI}.npz',
    ]
    rdms_npz = next((p for p in rdms_candidates if p.exists()), None)
    if rdms_npz is None:
        candidates_str = '\n  '.join(str(p) for p in rdms_candidates)
        sys.exit(f"\nERROR: missing model-RDMs npz. Looked for:\n  "
                 f"{candidates_str}\n"
                 f"The main RSA writes this for every ROI it processes.")
    with np.load(rdms_npz) as nz:
        # Strip _z off the test variant for model lookup since model RDMs
        # are the same for z and non-z evaluations.
        variant_no_z = TEST_VARIANT[:-2] if TEST_VARIANT.endswith('_z') else TEST_VARIANT
        prefix = f'model__{variant_no_z}__'
        model_RDMs = {k[len(prefix):]: nz[k] for k in nz.files
                       if k.startswith(prefix)}
    missing = [m for m in (CONTROLS + DSR_VARIANTS) if m not in model_RDMs]
    if missing:
        sys.exit(f"\nERROR: model RDMs missing from {rdms_npz}: {missing}\n"
                 f"Available: {sorted(model_RDMs)}")
    print(f"Loaded model RDMs ({len(model_RDMs)}) from {rdms_npz}")

    # Sanity: every model RDM length must match the data RDM length.
    n_pairs = y_emp_full.size
    for m in (CONTROLS + DSR_VARIANTS):
        if model_RDMs[m].shape[0] != n_pairs:
            sys.exit(f"\nERROR: model RDM {m!r} has {model_RDMs[m].shape[0]} "
                     f"pairs but data RDM has {n_pairs}. "
                     f"Test variant mismatch?")

    # Output dir --------------------------------------------------------
    run_tag = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = RSA_RUN_DIR / f'control_dropout_{ROI}_{run_tag}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the subset grid ---------------------------------------------
    subsets = _enumerate_subsets(CONTROLS)
    print(f"\nEnumerating {len(subsets)} control subsets × "
          f"{len(DSR_VARIANTS)} DSR variants = "
          f"{len(subsets) * len(DSR_VARIANTS)} combos")

    rows = []
    for size, ctrl_label, sub_ctrls in subsets:
        for dsr in DSR_VARIANTS:
            sub_models = sub_ctrls + [dsr]
            X = np.stack([model_RDMs[m] for m in sub_models], axis=1)
            dsr_col = len(sub_ctrls)   # DSR is the last regressor

            # SAME shared standardised-OLS function for both empirical and
            # permutation fits (CLAUDE.md rule #4).
            t_emp, beta_emp, p_emp = evaluate_model_vec(X, y_emp_full)
            _, BETA_PERMS, _ = evaluate_model_vec(X, Y_perms_full)
            null_dsr = BETA_PERMS[:, dsr_col]
            beta_dsr_emp = float(beta_emp[dsr_col])
            valid = np.isfinite(null_dsr)
            if valid.any() and np.isfinite(beta_dsr_emp):
                p_perm = (np.sum(null_dsr[valid] >= beta_dsr_emp) + 1) / (
                    valid.sum() + 1)
            else:
                p_perm = np.nan

            rows.append({
                'n_ctrls':       size,
                'ctrl_subset':   ctrl_label,
                'controls_in':   ', '.join(sub_ctrls),
                'dsr_variant':   dsr,
                'combo':         f'{ctrl_label}__{dsr}',
                'n_features':    len(sub_models),
                'beta_dsr':      beta_dsr_emp,
                't_dsr':         float(t_emp[dsr_col]),
                'p_perm':        float(p_perm),
            })
    df = pd.DataFrame(rows)
    # Within (ctrl_subset × DSR_variant) family there's only one ROI,
    # so q_FDR == p_perm. We still write a q column for compatibility.
    df['q_fdr_within_family'] = df['p_perm']
    df.to_csv(out_dir / 'all_combos_results.csv', index=False)
    print(f"\nWrote all_combos_results.csv  ({len(df)} rows)")

    # Pivot for visualisation -----------------------------------------
    pivot = df.pivot_table(
        index=['n_ctrls', 'ctrl_subset'],
        columns='dsr_variant',
        values=['beta_dsr', 'p_perm'],
        aggfunc='first',
    )
    pivot.columns = [f'{m}_{sm}' for m, sm in pivot.columns]
    p_cols = [f'p_perm_{sm}' for sm in DSR_VARIANTS if f'p_perm_{sm}' in pivot.columns]
    pivot['min_p_across_dsr'] = pivot[p_cols].min(axis=1)
    pivot = pivot.sort_values('min_p_across_dsr')
    pivot.to_csv(out_dir / f'{ROI}_pivot.csv')

    # Heatmap ---------------------------------------------------------
    H = pivot[p_cols].to_numpy()
    fig_h = max(6.0, 0.20 * H.shape[0])
    fig, ax = plt.subplots(figsize=(5.5, fig_h), constrained_layout=True)
    im = ax.imshow(H, aspect='auto', cmap='Reds_r', vmin=0, vmax=0.30,
                   interpolation='nearest')
    ax.set_xticks(range(len(DSR_VARIANTS)))
    ax.set_xticklabels(DSR_VARIANTS, rotation=20, ha='right', fontsize=8)
    row_labels = []
    for (n_ct, ctrl), _ in pivot.iterrows():
        if ctrl == 'NOCTRL':
            row_labels.append('[0] no ctrls')
        else:
            row_labels.append(f'[{n_ct}] ' + ctrl.replace('_', '+'))
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=5)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            s = _stars(H[i, j])
            if s:
                col = 'white' if H[i, j] < 0.05 else 'black'
                ax.text(j, i, s, ha='center', va='center',
                        fontsize=6, color=col, fontweight='bold')
    ax.set_xlabel('DSR sub-model')
    ax.set_title(
        f'{ROI} — p_perm per (control subset × DSR variant)\n'
        f'{n_perms} perms ({TEST_VARIANT});  ·<.10  *<.05  **<.01  ***<.001',
        fontsize=9,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('p_perm', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir / f'{ROI}_dropout_heatmap.{ext}', dpi=300,
                    bbox_inches='tight')
    plt.close(fig)

    # Config snapshot --------------------------------------------------
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({
            'rsa_run_dir':       str(RSA_RUN_DIR),
            'roi':               ROI,
            'test_variant':      TEST_VARIANT,
            'controls':          CONTROLS,
            'dsr_variants':      DSR_VARIANTS,
            'n_subsets':         len(subsets),
            'n_combos':          len(rows),
            'n_perms':           int(n_perms),
            'perm_pickle':       str(perm_path),
            'model_rdms_npz':    str(rdms_npz),
            'perm_fingerprint':  {k: (list(v) if isinstance(v, tuple) else v)
                                   for k, v in fp.items()},
        }, f, indent=2, default=str)

    print(f"\nTop 15 control subsets for {ROI} (sorted by best p_perm across DSR variants):")
    show = pivot.head(15)
    show_cols = [f'{m}_{sm}' for sm in DSR_VARIANTS
                 for m in ('beta_dsr', 'p_perm')]
    print(show[show_cols].round(4).to_string())

    print(f"\nDone. Outputs in {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
