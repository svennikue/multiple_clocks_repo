#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rodent DSR RSA — chosen key analyses.

Two main analyses are run for every recday:
    1. z-scored full DSR RDM  (mode_path / all_trials / full_z)
    2. across-task-halves RDM  (mode_path; duplicate-config sessions as halves)

For the example recday (most neurons by default), three publication figures are
built per analysis (figs 1 & 2 are per-variant, fig 3 is variant-independent):
    - fig 1: DSR overview (modelled neurons + DSR model RDM + group betas with FDR)
    - fig 2: example-subject row (data + each model's activation and RDM)
    - fig 3: model schematics (one configuration; midnight palette)

A stats JSON containing mean / SD / SEM / group t / FDR-corrected p per model
is written for both analyses.

The full comparison-matrix sweep (trial filter × pool method × variant + the
continuous pipeline) lives in ``analysis_rodents_legacy_alternatives.py``.

@author: Svenja Kuechenhoff
"""

import os
import json
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed

import mc.analyse.analyse_ephys_clean as ae


# ── Settings ──────────────────────────────────────────────────────────
DATA_FOLDER = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/'
OUT_BASE    = f"{DATA_FOLDER}derivatives/rodent_DSR_RSA/"

NUMBER_PHASE_NEURONS = 2
N_CONDS_PER_CONFIG   = 8       # 12 conditions / task config (matches human pipeline)
# Both DSR variants are computed for every recday:
#   'dsr'      — rodent-native DSR (predictions.model_DSR, cosine RDM)
#   'dsr_fmri' — human-pipeline DSR (mode-path rolled by bin, Hamming RDM;
#                same construction as RSA_DSR_ROIs_simple.build_mode_path_dsr)
# PIPELINE B — dsr_fmri (Hamming, fMRI-style) as the primary DSR regressor.
# The joint GLM fits ONLY these 4 regressors: dsr_fmri + 3 rodent controls.
# Rationale (see diagnose_dsr_vs_dsr_fmri.py output 2026-07-27_22-28-27):
# `dsr` and `dsr_fmri` are r≈0.6 correlated, so putting them in the same
# GLM makes them cannibalise each other's β. `dsr_fmri` survives the 4
# control stack on its own (see diagnostic drop-in trace) — this combo is
# the clean "dsr_fmri + controls" report.
MODEL_ORDER_DSR      = ['dsr_fmri', 'stat', 'loc', 'phas']
# The schematic figures (fig 1 top-panel DSR display, fig 2 model row) show
# every model in MODEL_ORDER_DSR — dsr_fmri included, now that its display
# activation + Hamming RDM are built alongside the cosine-space models.
MODEL_ORDER_FIG      = list(MODEL_ORDER_DSR)
# Whichever DSR variant is present in MODEL_ORDER_DSR is used as the "DSR"
# panel at the top of fig 1. Pipeline A -> 'dsr'; Pipeline B -> 'dsr_fmri'.
DSR_DISPLAY_KEY      = 'dsr_fmri' if 'dsr_fmri' in MODEL_ORDER_DSR else 'dsr'

# Phase residualisation: per-cell OLS of firing rate against a within-state
# phase basis, subtracting the phase component before RDM computation.
#   'cosine'      — 2 basis functions (sin, cos of 2πφ). Same as the human
#                    RSA in RSA_DSR_ROIs_simple.py. Removes the smooth
#                    first-harmonic phase but leaves higher harmonics.
#   'cosine_2h'   — 4 basis functions (adds sin, cos of 4πφ). Removes
#                    first + second harmonic. This is what we use here
#                    because the rodent `phas` model (3 von Mises tuning
#                    curves per state, κ ≈ 3.33) has non-trivial 2nd-
#                    harmonic content that a bare 'cosine' basis leaves
#                    behind. In humans, the phase model is already null
#                    after 'cosine' residualisation so this refinement
#                    isn't needed there; the rodent phas structure is
#                    sharper and needs the extra basis functions.
#   'categorical' — 3 boxcar indicators (early/middle/late). Tightest
#                    match to the 3-neuron structure of `phas`, but
#                    departs further from the human convention.
#   None          — no residualisation.
PHASE_RESIDUALISE    = 'cosine_2h'

EXAMPLE_RECDAY = None           # None -> pick the recday with the most neurons
N_JOBS         = -1             # joblib: -1 = all cores, 1 = serial (for debugging)

# Per-recday worker config.
ANALYSIS_CONFIG = {
    'number_phase_neurons': NUMBER_PHASE_NEURONS,
    'n_conds_per_config':   N_CONDS_PER_CONFIG,
    'dsr_pool_methods':     ['mode_path'],
    'run_continuous':       False,   # skip slow per-trial loop; not used here
    'phase_residualise':    PHASE_RESIDUALISE,
    # Combo the joint GLM actually fits. Must match MODEL_ORDER_DSR so the
    # stats aggregation, per-recday β column, and figure box-plot all use
    # the same set of regressors. Pipeline A (rodent-native) would be
    # ['dsr', 'stat', 'loc', 'phas', 'midn']. Pipeline B (this file) uses
    # ['dsr_fmri', 'stat', 'loc', 'phas'] — same list as MODEL_ORDER_DSR.
    'combo_order':          list(MODEL_ORDER_DSR),
    # The following are only consulted when run_continuous is True:
    'no_bins_per_state':    10,
    'mask_within':          True,
    'segmentation':         'reward_dwell',
    'plot_rsa_panels':      False,
}

np.random.seed(42)


# ── Output folder (timestamped) ───────────────────────────────────────
RUN_TAG = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUT_DIR = os.path.join(OUT_BASE, RUN_TAG)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Run output: {OUT_DIR}")


# ── Load + clean ──────────────────────────────────────────────────────
print("Loading mouse data (raw + normalised) ...")
mouse_data      = ae.load_ephys_data(DATA_FOLDER, raw=True)
mouse_data_norm = ae.load_ephys_data(DATA_FOLDER, raw=False)
print(f"  loaded {len(mouse_data)} recdays: {list(mouse_data)}")

# Sessions surviving into BOTH views — the original authors' implicit "bad
# session" flag is "missing from the normalised view".
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


# ── Duplicate-config overview ─────────────────────────────────────────
# How many task configs in each recday have ≥2 source sessions (i.e. qualify
# for the across-halves analysis), and how many trials feed each half.
print("\nDuplicate-config overview (qualifying tasks for across-halves analysis):")
print(f"  {'recday':35s} {'n_qualifying':12s} {'configs (h1 trials / h2 trials)':40s}")
for recday in sorted(mouse_data):
    cfg, _, _, tim = cleaned['norm'][recday]
    sid = clean_meta['norm'][recday]['kept_session_ids']
    q   = ae.split_sessions_into_halves(cfg, sid, tim)
    summary = ', '.join(f"{tuple(g['config'])}={g['half1_n_trials']}/{g['half2_n_trials']}"
                        for g in q)
    print(f"  {recday:35s} {len(q):<12d} {summary}")


# ── Repeats overview ──────────────────────────────────────────────────
# Per-task trial counts, per-recday totals, and the average over tasks. Two
# scopes are computed so they can be archived in the stats JSON:
#   - all_data:       all task configs in the cleaned normalised view
#   - across_halves:  only qualifying configs (≥2 sessions, used by the
#                     across-task-halves analysis), split into h1 / h2.
print("\nRepeats overview (after cleaning, normalised view):")
print(f"  {'recday':35s} {'per-task trial counts':40s} "
      f"{'total':>7s} {'avg/task':>10s}")
repeats_all = {}     # per-recday: dict of summary stats for all data
total_per_recday    = {}
avg_per_task_recday = {}
for recday in sorted(mouse_data):
    _, _, _, tim = cleaned['norm'][recday]
    per_task = [int(len(tim[t])) for t in range(len(tim))]
    total = int(sum(per_task))
    avg   = total / max(len(per_task), 1)
    total_per_recday[recday]    = total
    avg_per_task_recday[recday] = avg
    repeats_all[recday] = {
        'n_tasks':            len(per_task),
        'per_task_repeats':   per_task,
        'total_repeats':      total,
        'avg_repeats_per_task': avg,
    }
    pt_str = ' '.join(str(x) for x in per_task)
    print(f"  {recday:35s} {pt_str:40s} {total:>7d} {avg:>10.1f}")

overall_total          = int(sum(total_per_recday.values()))
overall_avg_per_task   = float(np.mean(list(avg_per_task_recday.values())))
overall_avg_per_recday = overall_total / max(len(total_per_recday), 1)
print(f"  {'-' * 90}")
print(f"  {'OVERALL':35s} {'-':40s} {overall_total:>7d} "
      f"{overall_avg_per_task:>10.1f}  (avg/recday: {overall_avg_per_recday:.1f})")

# Across-halves view: only qualifying configs (≥2 sessions) and their
# half-1 / half-2 trial counts.
repeats_halves = {}
for recday in sorted(mouse_data):
    cfg, _, _, tim = cleaned['norm'][recday]
    sid = clean_meta['norm'][recday]['kept_session_ids']
    q   = ae.split_sessions_into_halves(cfg, sid, tim)
    if not q:
        repeats_halves[recday] = {'n_qualifying_configs': 0,
                                  'per_config': [], 'totals': {}}
        continue
    per_cfg = [{'config':  [int(x) for x in g['config']],
                'half1_n_trials': int(g['half1_n_trials']),
                'half2_n_trials': int(g['half2_n_trials'])}
               for g in q]
    h1 = sum(g['half1_n_trials'] for g in per_cfg)
    h2 = sum(g['half2_n_trials'] for g in per_cfg)
    repeats_halves[recday] = {
        'n_qualifying_configs': len(per_cfg),
        'per_config':           per_cfg,
        'totals': {'half1': h1, 'half2': h2,
                   'avg_per_config_half1': h1 / len(per_cfg),
                   'avg_per_config_half2': h2 / len(per_cfg)},
    }

repeats_overview = {
    'all_data': {
        'per_recday': repeats_all,
        'overall': {
            'total_repeats':         overall_total,
            'avg_repeats_per_task':  overall_avg_per_task,
            'avg_repeats_per_recday': overall_avg_per_recday,
            'n_recdays':             len(repeats_all),
        },
    },
    'across_halves': {
        'per_recday': repeats_halves,
        'overall': {
            'n_recdays_with_qualifying_configs': sum(
                1 for v in repeats_halves.values()
                if v['n_qualifying_configs'] > 0),
            'total_qualifying_configs': sum(
                v['n_qualifying_configs'] for v in repeats_halves.values()),
            'total_half1_trials': sum(
                v['totals'].get('half1', 0) for v in repeats_halves.values()),
            'total_half2_trials': sum(
                v['totals'].get('half2', 0) for v in repeats_halves.values()),
        },
    },
}


# ── Run both main analyses in parallel ────────────────────────────────
print(f"\nDispatching {len(mouse_data)} recdays across {N_JOBS} workers ...")
per_recday = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(ae.process_one_recday)(
        recday,
        cleaned['raw'][recday],  clean_meta['raw'][recday]['kept_session_ids'],
        cleaned['norm'][recday], clean_meta['norm'][recday]['kept_session_ids'],
        None, ANALYSIS_CONFIG,
    )
    for recday in sorted(mouse_data)
)

full_z_results = {r['recday']: r['dsr_by_pool']['mode_path']['full_z'] for r in per_recday}
halves_results = {r['recday']: r['halves']['across_halves'] for r in per_recday
                  if 'across_halves' in r['halves']}

n_neurons_per_recday    = {r['recday']: int(cleaned['norm'][r['recday']][2][0].shape[0])
                           for r in per_recday}
n_pooled_per_recday     = {r['recday']: int(r['trim_pool_norm']['pooling']['n_configs_out'])
                           for r in per_recday}
n_qualifying_per_recday = {r['recday']: int(r['halves']['metadata']['n_qualifying_configs'])
                           for r in per_recday}


# ── Pick example recday for figures ───────────────────────────────────
example_recday = EXAMPLE_RECDAY or max(n_neurons_per_recday, key=n_neurons_per_recday.get)
print(f"\nExample recday: {example_recday} "
      f"({n_neurons_per_recday[example_recday]} neurons)")


# ── Stats helpers (small, local) ──────────────────────────────────────
def _print_stats(stats):
    print(f"\n=== {stats['pipeline']} ===")
    print(f"  n_recdays: {stats['n_recdays']}")
    print(f"  n_neurons: {stats['n_neurons_summary']}")
    print(f"  n_tasks:   {stats['n_tasks_summary']}")
    print(f"  {'model':18s} {'mean':>8s} {'sd':>8s} {'sem':>8s} "
          f"{'t':>7s} {'p_unc':>9s} {'p_fdr':>9s} {'sig':>4s}")
    for m in MODEL_ORDER_DSR:
        s = stats['models'][m]
        print(f"  {s['label']:18s} {s['mean']:>8.3f} {(s['sd'] or 0):>8.3f} "
              f"{(s['sem'] or 0):>8.3f} {(s['t_group'] or 0):>7.2f} "
              f"{(s['p_group_uncorrected'] or 1):>9.4f} "
              f"{(s['p_group_fdr'] or 1):>9.4f} {'*' if s['sig_fdr'] else '':>4s}")


def _coefs_and_fdr(stats):
    coefs = {m: np.asarray([stats['models'][m]['coefs_by_recday'][rd]
                            for rd in stats['recdays']])
             for m in MODEL_ORDER_DSR}
    fdr   = {m: stats['models'][m]['p_group_fdr'] for m in MODEL_ORDER_DSR}
    return coefs, fdr


# ── Main analysis 1: full DSR z-scored RDM ────────────────────────────
stats_full_z = ae.methods_results_stats(
    full_z_results, n_neurons_per_recday, n_pooled_per_recday,
    model_label_order=MODEL_ORDER_DSR)
stats_full_z['pipeline'] = 'DSR mode_path / all_trials / full_z'
_print_stats(stats_full_z)

# Build example matrices from the pooled normalised view.
cfg_ex, loc_ex, neu_ex, tim_ex = cleaned['norm'][example_recday]
cfg_ex, loc_ex, neu_ex, tim_ex, _ = ae.pool_by_task_config(
    cfg_ex, loc_ex, neu_ex, tim_ex, kind='norm',
    session_ids=clean_meta['norm'][example_recday]['kept_session_ids'],
    return_metadata=True)
fz_data_act, fz_data_rdm, fz_model_acts, fz_model_rdms = (
    ae.dsr_example_recday_matrices(
        cfg_ex, loc_ex, neu_ex, tim_ex,
        n_conds_per_config=N_CONDS_PER_CONFIG,
        no_phase_neurons=NUMBER_PHASE_NEURONS))

fz_coefs, fz_fdr = _coefs_and_fdr(stats_full_z)
ae.pub_figure_dsr_overview(
    dsr_model_activation=fz_model_acts[DSR_DISPLAY_KEY],
    dsr_model_rdm=fz_model_rdms[DSR_DISPLAY_KEY],
    coefs_by_model=fz_coefs, model_order=MODEL_ORDER_FIG, fdr_pvals=fz_fdr,
    n_tasks=len(cfg_ex), n_conds_per_task=N_CONDS_PER_CONFIG,
    recday_label=example_recday,
    save_stem=os.path.join(OUT_DIR, 'fig1_full_z'))
ae.pub_figure_example_subject(
    data_activation=fz_data_act, data_rdm=fz_data_rdm,
    model_activations=fz_model_acts, model_rdms=fz_model_rdms,
    model_order=MODEL_ORDER_FIG,
    n_tasks=len(cfg_ex), n_conds_per_task=N_CONDS_PER_CONFIG,
    recday_label=example_recday,
    save_stem=os.path.join(OUT_DIR, 'fig2_full_z'))


# ── Main analysis 2: across-task-halves (mode_path) ───────────────────
stats_halves = ae.methods_results_stats(
    halves_results, n_neurons_per_recday, n_qualifying_per_recday,
    model_label_order=MODEL_ORDER_DSR)
stats_halves['pipeline'] = 'DSR mode_path / across-task-halves'
_print_stats(stats_halves)

# Example matrices: POST-clean PRE-pool (we need per-session structure for halves).
cfg_pre, loc_pre, neu_pre, tim_pre = cleaned['norm'][example_recday]
halves_mats = ae.dsr_across_halves_matrices(
    cfg_pre, loc_pre, neu_pre, tim_pre,
    session_ids=clean_meta['norm'][example_recday]['kept_session_ids'],
    n_conds_per_config=N_CONDS_PER_CONFIG,
    no_phase_neurons=NUMBER_PHASE_NEURONS)

if halves_mats is not None:
    h_data_act, h_data_rdm, h_model_acts, h_model_rdms, K_h = halves_mats
    h_coefs, h_fdr = _coefs_and_fdr(stats_halves)
    # Across-halves matrices stack [half-1 of all qualifying configs, half-2
    # of all qualifying configs], so the display axis has 2*K_h task columns.
    K_display = 2 * K_h
    ae.pub_figure_dsr_overview(
        dsr_model_activation=h_model_acts[DSR_DISPLAY_KEY],
        dsr_model_rdm=h_model_rdms[DSR_DISPLAY_KEY],
        coefs_by_model=h_coefs, model_order=MODEL_ORDER_DSR, fdr_pvals=h_fdr,
        n_tasks=K_display, n_conds_per_task=N_CONDS_PER_CONFIG,
        recday_label=f'{example_recday} (across-halves, K={K_h})',
        x_axis_groups=[('Task half 1', K_h), ('Task half 2', K_h)],
        save_stem=os.path.join(OUT_DIR, 'fig1_across_halves'))
    ae.pub_figure_example_subject(
        data_activation=h_data_act, data_rdm=h_data_rdm,
        model_activations=h_model_acts, model_rdms=h_model_rdms,
        model_order=MODEL_ORDER_FIG,
        n_tasks=K_display, n_conds_per_task=N_CONDS_PER_CONFIG,
        recday_label=f'{example_recday} (across-halves, K={K_h})',
        x_axis_groups=[('run 1', K_h), ('run 2', K_h)],
        save_stem=os.path.join(OUT_DIR, 'fig2_across_halves'))
else:
    print(f"\nExample recday {example_recday} has no qualifying configs; "
          f"skipping across-halves figures.")


# ── Figure 3: model schematics (variant-independent) ──────────────────
ae.pub_figure_model_schematics(
    walked_path=ae._mode_path_360(loc_ex[0]),
    task_config=cfg_ex[0],
    no_phase_neurons=NUMBER_PHASE_NEURONS,
    recday_label=example_recday,
    save_stem=os.path.join(OUT_DIR, 'fig3_model_schematics'))


# ── Write stats JSON ──────────────────────────────────────────────────
stats_path = os.path.join(OUT_DIR, 'key_analysis_stats.json')
with open(stats_path, 'w') as f:
    json.dump({'full_z':           stats_full_z,
               'across_halves':    stats_halves,
               'repeats_overview': repeats_overview},
              f, indent=2)
print(f"\nWrote stats: {stats_path}")

# Variables persist in Spyder's variable explorer because the script runs at
# module scope. Uncomment if you want a hard pause before exit.
# import pdb; pdb.set_trace()
