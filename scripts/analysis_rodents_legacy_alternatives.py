#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rodent DSR RSA — legacy comparison-matrix sweep.

Runs the full {trial filter} × {DSR pool method} × {RDM variant} grid plus the
continuous-pipeline RSA on the raw recordings, producing:
    - overview_continuous.jpeg : 2 trial filters × 8 (variant × model) panels
    - overview_dsr.jpeg        : 4 (trial × pool) × 12 (variant × model) panels
    - overview_across_halves.jpeg : 2 trial filters × 4 model panels
    - rodent_methods_config.json : full per-recday + per-variant stats dump
    - rodent_results.pkl   (optional, when SAVE_PICKLE=True)

For the chosen key analyses (full_z and across-halves with mode_path) and the
publication figures, use ``analysis_rodents_complete_clean.py``.

@author: Svenja Kuechenhoff
"""

import os
import json
import pickle
from datetime import datetime

import numpy as np
from scipy.stats import ttest_1samp
from joblib import Parallel, delayed

import mc.analyse.analyse_ephys_clean as ae


# ── Settings ──────────────────────────────────────────────────────────
DATA_FOLDER = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/'
OUT_BASE    = f"{DATA_FOLDER}derivatives/rodent_DSR_RSA_legacy/"

PLOT_RSA_PANELS = False        # per-recday activation- and cov-by-time panels (heavy)
SAVE_PICKLE     = False        # write a pickle of all per-recday results

NO_BINS_PER_STATE    = 10
NUMBER_PHASE_NEURONS = 3
MASK_WITHIN          = True
SEGMENTATION         = 'reward_dwell'
N_CONDS_PER_CONFIG   = 12
N_TRIALS_REQUIRED    = 10      # for the 'last_n' trial selection

TRIAL_FILTERS       = [('all_trials', None), ('last_n', N_TRIALS_REQUIRED)]
DSR_POOL_METHODS    = ['mode_path', 'per_run_avg']
CONTINUOUS_VARIANTS = ['normal', 'standardized']
DSR_VARIANTS        = ['across_z', 'within_z', 'full_z']

MODEL_ORDER_CONT = ['clo_model', 'phas_model', 'loc_model', 'stat_model']
MODEL_ORDER_DSR  = ['dsr', 'phas', 'loc', 'stat']

N_JOBS = -1

ANALYSIS_CONFIG = {
    'no_bins_per_state':    NO_BINS_PER_STATE,
    'number_phase_neurons': NUMBER_PHASE_NEURONS,
    'mask_within':          MASK_WITHIN,
    'segmentation':         SEGMENTATION,
    'n_conds_per_config':   N_CONDS_PER_CONFIG,
    'dsr_pool_methods':     DSR_POOL_METHODS,
    'plot_rsa_panels':      PLOT_RSA_PANELS,
    'run_continuous':       True,
}

np.random.seed(42)


# ── Output folder ─────────────────────────────────────────────────────
RUN_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUT_DIR = os.path.join(OUT_BASE, RUN_TAG)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Run output: {OUT_DIR}")


# ── Small helpers (used only by the report-writing block below) ───────
def _json_float(x):
    x = float(x)
    return x if np.isfinite(x) else None


def _stats(values):
    """Summary stats of a 1-D array of per-recday betas."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(len(values))
    if n == 0:
        return {'n': 0, 'mean': None, 'std': None, 'sem': None,
                'median': None, 'min': None, 'max': None}
    std = float(np.std(values, ddof=1)) if n > 1 else None
    return {
        'n':      n,
        'mean':   _json_float(np.mean(values)),
        'std':    None if std is None else _json_float(std),
        'sem':    None if std is None else _json_float(std / np.sqrt(n)),
        'median': _json_float(np.median(values)),
        'min':    _json_float(np.min(values)),
        'max':    _json_float(np.max(values)),
    }


def _summarise_results(per_recday):
    if not per_recday:
        return {'n_recdays': 0, 'recdays': [], 'models': {}}
    label_regs = list(next(iter(per_recday.values()))['label_regs'])
    recdays    = list(per_recday)
    out = {'n_recdays': len(recdays), 'recdays': recdays, 'models': {}}
    for i, model in enumerate(label_regs):
        coefs  = {rd: _json_float(per_recday[rd]['coefs'][i])  for rd in recdays}
        t_vals = {rd: _json_float(per_recday[rd]['t_vals'][i]) for rd in recdays}
        p_vals = {rd: _json_float(per_recday[rd]['p_vals'][i]) for rd in recdays}
        arr = np.asarray([v for v in coefs.values() if v is not None], dtype=float)
        if len(arr) > 1:
            t, p = ttest_1samp(arr, 0, alternative='greater')
            group = {'t_one_sample_greater_than_zero': _json_float(t),
                     'p_one_sample_greater_than_zero': _json_float(p)}
        else:
            group = {'t_one_sample_greater_than_zero': None,
                     'p_one_sample_greater_than_zero': None}
        out['models'][model] = {
            'label':           ae.MODEL_LABELS.get(model, model),
            'coef_summary':    _stats(arr),
            'coefs_by_recday': coefs,
            't_by_recday':     t_vals,
            'p_by_recday':     p_vals,
            'group_test':      group,
        }
    return out


def _coefs_per_recday(per_recday, model_idx):
    return [per_recday[rd]['coefs'][model_idx] for rd in per_recday]


# ── Load + clean ──────────────────────────────────────────────────────
print("Loading mouse data (raw + normalised) ...")
mouse_data      = ae.load_ephys_data(DATA_FOLDER, raw=True)
mouse_data_norm = ae.load_ephys_data(DATA_FOLDER, raw=False)
print(f"  loaded {len(mouse_data)} recdays: {list(mouse_data)}")

keep_by_recday = ae.cross_view_session_ids(mouse_data, mouse_data_norm)

cleaned    = {'raw': {}, 'norm': {}}
clean_meta = {'raw': {}, 'norm': {}}
for recday in sorted(mouse_data):
    for view, src in (('raw', mouse_data), ('norm', mouse_data_norm)):
        cfg, loc, neu, tim, m = ae.clean_ephys_data(
            src[recday]['rewards_configs'], src[recday]['locations'],
            src[recday]['neurons'], src[recday]['timings'], recday,
            session_ids=src[recday]['session_ids'],
            keep_session_ids=keep_by_recday[recday], return_metadata=True)
        cleaned[view][recday]    = (cfg, loc, neu, tim)
        clean_meta[view][recday] = m


# ── Run the full comparison matrix ────────────────────────────────────
continuous_results = {f: {v: {} for v in CONTINUOUS_VARIANTS} for f, _ in TRIAL_FILTERS}
dsr_results        = {f: {p: {v: {} for v in DSR_VARIANTS} for p in DSR_POOL_METHODS}
                      for f, _ in TRIAL_FILTERS}
halves_results     = {f: {} for f, _ in TRIAL_FILTERS}
trim_pool_metadata = {f: {'raw': {}, 'normalised': {}} for f, _ in TRIAL_FILTERS}

for filter_name, n_required in TRIAL_FILTERS:
    print(f"\n##### Trial selection: {filter_name} (n_required={n_required}) #####")
    print(f"  dispatching {len(mouse_data)} recdays across {N_JOBS} workers ...")

    per_recday = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(ae.process_one_recday)(
            recday,
            cleaned['raw'][recday],  clean_meta['raw'][recday]['kept_session_ids'],
            cleaned['norm'][recday], clean_meta['norm'][recday]['kept_session_ids'],
            n_required, ANALYSIS_CONFIG,
        )
        for recday in sorted(mouse_data)
    )

    for r in per_recday:
        recday = r['recday']
        for variant in CONTINUOUS_VARIANTS:
            continuous_results[filter_name][variant][recday] = r['continuous'][variant]
        for pool_method in DSR_POOL_METHODS:
            for variant in DSR_VARIANTS:
                dsr_results[filter_name][pool_method][variant][recday] = (
                    r['dsr_by_pool'][pool_method][variant])
        if 'across_halves' in r['halves']:
            halves_results[filter_name][recday] = r['halves']['across_halves']
        trim_pool_metadata[filter_name]['raw'][recday]        = r['trim_pool_raw']
        trim_pool_metadata[filter_name]['normalised'][recday] = r['trim_pool_norm']
        print(f"    {recday}: raw->{r['trim_pool_raw']['pooling']['n_configs_out']} cfgs, "
              f"norm->{r['trim_pool_norm']['pooling']['n_configs_out']} cfgs, "
              f"halves->{r['halves']['metadata']['n_qualifying_configs']} qualifying")


# ── Build overview panel data ─────────────────────────────────────────
# continuous: rows = trial filter, cols = variant × model
cont_rows = [f for f, _ in TRIAL_FILTERS]
cont_cols = [f"{v}/{ae.MODEL_LABELS[m]}"
             for v in CONTINUOUS_VARIANTS for m in MODEL_ORDER_CONT]
cont_panel = {row: {} for row in cont_rows}
for row in cont_rows:
    for v in CONTINUOUS_VARIANTS:
        per_recday = continuous_results[row][v]
        label_regs = list(next(iter(per_recday.values()))['label_regs'])
        for m in MODEL_ORDER_CONT:
            mi = label_regs.index(m)
            cont_panel[row][f"{v}/{ae.MODEL_LABELS[m]}"] = _coefs_per_recday(per_recday, mi)

# DSR: rows = trial filter × pool method, cols = variant × model
dsr_rows = [f"{f} / {p}" for f, _ in TRIAL_FILTERS for p in DSR_POOL_METHODS]
dsr_cols = [f"{v}/{ae.MODEL_LABELS[m]}"
            for v in DSR_VARIANTS for m in MODEL_ORDER_DSR]
dsr_panel = {row: {} for row in dsr_rows}
for f, _ in TRIAL_FILTERS:
    for p in DSR_POOL_METHODS:
        row = f"{f} / {p}"
        for v in DSR_VARIANTS:
            per_recday = dsr_results[f][p][v]
            label_regs = list(next(iter(per_recday.values()))['label_regs'])
            for m in MODEL_ORDER_DSR:
                mi = label_regs.index(m)
                dsr_panel[row][f"{v}/{ae.MODEL_LABELS[m]}"] = _coefs_per_recday(per_recday, mi)

# across-halves: rows = trial filter, cols = model
halves_rows = [f for f, _ in TRIAL_FILTERS]
halves_cols = [ae.MODEL_LABELS[m] for m in MODEL_ORDER_DSR]
halves_panel = {row: {} for row in halves_rows}
for row in halves_rows:
    per_recday = halves_results[row]
    if not per_recday:
        for col in halves_cols:
            halves_panel[row][col] = []
        continue
    label_regs = list(next(iter(per_recday.values()))['label_regs'])
    for m in MODEL_ORDER_DSR:
        mi = label_regs.index(m)
        halves_panel[row][ae.MODEL_LABELS[m]] = _coefs_per_recday(per_recday, mi)


# ── Overview plots ────────────────────────────────────────────────────
ae.plot_betas_grid(
    cont_panel, cont_rows, cont_cols,
    suptitle=f"Continuous RSA — betas across {len(mouse_data)} recdays",
    save_path=os.path.join(OUT_DIR, 'overview_continuous.jpeg'))

ae.plot_betas_grid(
    dsr_panel, dsr_rows, dsr_cols,
    suptitle=f"DSR (my_RSA) — betas across {len(mouse_data)} recdays",
    save_path=os.path.join(OUT_DIR, 'overview_dsr.jpeg'))

ae.plot_betas_grid(
    halves_panel, halves_rows, halves_cols,
    suptitle=f"Across-task-halves — betas across qualifying recdays",
    save_path=os.path.join(OUT_DIR, 'overview_across_halves.jpeg'))


# ── Report + (optional) pickle ────────────────────────────────────────
report = {
    'run_tag':       RUN_TAG,
    'analysis_date': str(np.datetime64('today')),
    'data_folder':   DATA_FOLDER,
    'recdays':       list(mouse_data),
    'settings': {
        'continuous_segmentation':    SEGMENTATION,
        'no_bins_per_state':          int(NO_BINS_PER_STATE),
        'number_phase_neurons':       int(NUMBER_PHASE_NEURONS),
        'mask_within_for_continuous': bool(MASK_WITHIN),
        'n_conds_per_config_for_dsr': int(N_CONDS_PER_CONFIG),
        'n_trials_required':          int(N_TRIALS_REQUIRED),
        'trial_filters':              [f for f, _ in TRIAL_FILTERS],
        'dsr_pool_methods':           DSR_POOL_METHODS,
    },
    'clean_metadata':     clean_meta,
    'trim_pool_metadata': trim_pool_metadata,
    'results': {
        'continuous':    {f: {v: _summarise_results(continuous_results[f][v])
                              for v in CONTINUOUS_VARIANTS}
                          for f, _ in TRIAL_FILTERS},
        'dsr':           {f: {p: {v: _summarise_results(dsr_results[f][p][v])
                                  for v in DSR_VARIANTS}
                              for p in DSR_POOL_METHODS}
                          for f, _ in TRIAL_FILTERS},
        'across_halves': {f: _summarise_results(halves_results[f])
                          for f, _ in TRIAL_FILTERS},
    },
}

report_path = os.path.join(OUT_DIR, 'rodent_methods_config.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"\nWrote methods/config report: {report_path}")

if SAVE_PICKLE:
    with open(os.path.join(OUT_DIR, 'rodent_results.pkl'), 'wb') as f:
        pickle.dump({'continuous': continuous_results, 'dsr': dsr_results,
                     'halves': halves_results}, f)
