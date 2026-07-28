#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic — why is `dsr_fmri` weaker than `dsr` in the rodent RSA?

For every recday and every pipeline variant, computes:
    * Spearman ρ between the `dsr` (cosine) and `dsr_fmri` (Hamming) model
      RDMs — how similar are their second-order geometries at all?
    * β / t / p of each model fit to the neural data RDM ALONE.
    * β of each model in the JOINT GLM `data ~ dsr + dsr_fmri + phas`
      (partialled — tells you what each model uniquely explains).
    * Same numbers at three RDM slices: `across_z` (across-task off-block),
      `within_z` (within-task off-diagonal), `full_z` (all off-diagonal).

Pipeline variants:
    P1_mode_resid      — mode across all trials, phase-residualised (canonical).
    P2_mode_noresid    — mode across all trials, no phase-residualisation
                          (isolates the effect of residualisation).
    P3_pertrial_resid  — mode-of-per-trial-modes, phase-residualised
                          (isolates the effect of the pooling choice).

Plus a synthetic-halves variant that splits each task's trials into odd /
even trials and evaluates the across-halves RSA — lets you test the
"across-halves" logic without needing multi-session tasks.

Outputs:
    <OUT_BASE>/<timestamp>_diagnostic/
        diagnostic_summary.csv       — all rows, one per (recday × variant × slice)
        halves_synthetic_summary.csv — halves rows
        console — top-3 best / bottom-3 worst per (variant × slice) ranked by β_dsr

No permutations. Runtime ~1 min on 8 recdays.

@author: Svenja Kuechenhoff
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path('/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo')
sys.path.insert(0, str(REPO))
import mc.analyse.analyse_ephys_clean as ae
import mc.analyse.my_RSA as my_RSA
from mc.simulation import predictions


# ── Settings ──────────────────────────────────────────────────────────
DATA_FOLDER = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/'
OUT_BASE    = f"{DATA_FOLDER}derivatives/rodent_DSR_RSA_diagnostic/"

N_CONDS_PER_CONFIG = 12
NUMBER_PHASE_NEURONS = 3
N = N_CONDS_PER_CONFIG
BINLEN = 360 // N        # = 30 raw bins per condition

VARIANTS = [
    ('P1_mode_resid',      dict(pool_method='mode_path',       phase_residualise='cosine')),
    ('P2_mode_noresid',    dict(pool_method='mode_path',       phase_residualise=None)),
    ('P3_pertrial_resid',  dict(pool_method='per_trial_modes', phase_residualise='cosine')),
]

SLICES = ('across_z', 'within_z', 'full_z')


# ── Helpers ────────────────────────────────────────────────────────────
def _apply_residualise(neurons_task, basis):
    """(n_neurons, n_trials, 360) → same shape, per-cell phase residualised."""
    return ae._phase_residualise_task(neurons_task, basis)


def _bin_and_zscore(neu, cfg):
    """Per-task avg over trials + downsample to N bins + z-score per neuron.
    Returns (n_configs*N, n_neurons) matrix ready for compute_crosscorr_within."""
    cols = []
    for task_no in range(len(cfg)):
        avg = np.nanmean(neu[task_no], axis=1)
        cols.append(avg.reshape(avg.shape[0], N, BINLEN).mean(axis=2))
    mat_all = np.hstack(cols)
    mu = np.nanmean(mat_all, axis=1)
    sd = np.nanstd(mat_all, axis=1); sd[sd == 0] = 1
    return (mat_all.T - mu) / sd


def build_data_rdms(neu, cfg, phase_residualise=None):
    """Return dict of data-RDM triu vectors: across_z / within_z / full_z."""
    neu_r = [_apply_residualise(neu[t], phase_residualise) for t in range(len(neu))]
    mat_all_z = _bin_and_zscore(neu_r, cfg)
    within, across, full = my_RSA.compute_crosscorr_within(
        mat_all_z, plotting=False, include_diagonal=False,
        no_tasks=len(cfg), model='data', block_size=N)
    return {'across_z': across[0],
            'within_z': within[0],
            'full_z':   ae._upper_no_diag(full)}


def _build_model_cols_per_trial_modes(loc_pooled, no_phase_neurons):
    """Alternative pooling: take the mode path of each trial, then mode across
    those per-trial modes. Contrast with 'mode_path' (single mode across all
    concatenated trial bins). Returns the same dict shape as
    _build_dsr_model_cols('mode_path')."""
    out = {'dsr': [], 'stat': [], 'loc': [], 'phas': [], 'midn': [],
            'dsr_fmri': []}
    for task_no in range(len(loc_pooled)):
        trial_locs = np.asarray(loc_pooled[task_no])         # (n_trials, 360)
        per_trial_modes = np.stack(
            [ae._clean_node_path(trial_locs[t]) for t in range(trial_locs.shape[0])],
            axis=0)
        walked = ae._mode_path_360(per_trial_modes)          # mode of per-trial modes
        loc_m, phas_m, stat_m, midn_m, dsr_m, _, _ = predictions.model_DSR(
            locations=walked, no_phase_neurons=no_phase_neurons)
        for key, M in [('dsr', dsr_m), ('stat', stat_m), ('loc', loc_m),
                        ('phas', phas_m), ('midn', midn_m)]:
            out[key].append(M.reshape(M.shape[0], N, BINLEN).mean(axis=2))
        out['dsr_fmri'].append(ae._build_dsr_fmri_task(walked, N))
    return out


def build_model_rdms(loc_pooled, n_tasks, pool_method='mode_path'):
    """Return {slice: {model: rdm_vec}} for all six models."""
    if pool_method == 'per_trial_modes':
        model_cols = _build_model_cols_per_trial_modes(
            loc_pooled, NUMBER_PHASE_NEURONS)
    else:
        model_cols = ae._build_dsr_model_cols(
            loc_pooled, NUMBER_PHASE_NEURONS, pool_method, N, BINLEN)

    out = {slc: {} for slc in SLICES}
    for key, cols in model_cols.items():
        if key == 'dsr_fmri':
            concat = np.concatenate(cols, axis=0)
            within, across, full = my_RSA.compute_hamming_distance_within(
                concat, plotting=False, include_diagonal=False,
                no_tasks=n_tasks, model_name=key, block_size=N)
        else:
            concat = np.concatenate(cols, axis=1).T
            within, across, full = my_RSA.compute_crosscorr_within(
                concat, plotting=False, include_diagonal=False,
                no_tasks=n_tasks, model=key, block_size=N)
        out['across_z'][key] = across[0]
        out['within_z'][key] = within[0]
        out['full_z'][key] = ae._upper_no_diag(full)
    return out


def rsa_alone(data_vec, model_vec):
    """β / t / p of a single-model RSA."""
    if len(model_vec) < 3 or np.nanstd(model_vec) < 1e-12:
        return dict(beta=np.nan, t=np.nan, p=np.nan)
    stacked = np.asarray(model_vec).reshape(-1, 1)
    t, beta, p = my_RSA.evaluate_model(stacked, np.asarray(data_vec))
    return dict(beta=float(np.asarray(beta).ravel()[0]),
                t=float(np.asarray(t).ravel()[0]),
                p=float(np.asarray(p).ravel()[0]))


def rsa_joint(data_vec, model_dict, order):
    """Joint OLS. Returns per-model {β, t, p}."""
    stacked = np.stack([model_dict[k] for k in order], axis=1)
    t, beta, p = my_RSA.evaluate_model(stacked, np.asarray(data_vec))
    return {k: dict(beta=float(beta[i]), t=float(t[i]), p=float(p[i]))
            for i, k in enumerate(order)}


def dropin_series(data_vec, model_dict, target, control_order):
    """Incremental joint OLS: start with `target` alone, then add each
    control in `control_order` one at a time. Returns a list of dicts,
    each carrying `β / t / p` of the target model at that step.
    Lets you see which control is what makes `target` collapse."""
    trace = []
    used_ctrls = []
    for step in range(len(control_order) + 1):
        keys = [target] + used_ctrls
        stacked = np.stack([model_dict[k] for k in keys], axis=1)
        t, beta, p = my_RSA.evaluate_model(stacked, np.asarray(data_vec))
        trace.append({
            'step': step,
            'ctrls_in':  '+'.join(used_ctrls) if used_ctrls else 'NONE',
            'beta':      float(beta[0]),
            't':         float(t[0]),
            'p':         float(p[0]),
        })
        if step < len(control_order):
            used_ctrls.append(control_order[step])
    return trace


def spearman(a, b):
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else np.nan


# ── Synthetic-halves (odd / even trials) ──────────────────────────────
def synthetic_halves_matrices(cfg, loc, neu, phase_residualise=None,
                               pool_method='mode_path'):
    """Analog of reg_across_task_halves_DSR, but each config's trials are split
    into odd / even trials (rather than sessions). All configs qualify — you
    only need ≥2 trials.

    Returns dict of {slice: {model: rdm_vec}} for the data + each model,
    where 'slice' is only 'across_z' here (compute_crosscorr's cross-block).
    """
    # Use the SMALLER of (neu trial count, loc trial count) per task —
    # they should be paired but some pooled recdays have a 1-row drift
    # between the two arrays. Capping avoids an out-of-bounds index.
    qualifying = []
    idxs_by_task = []
    for t in range(len(cfg)):
        n_trials = min(neu[t].shape[1], np.asarray(loc[t]).shape[0])
        if n_trials >= 2:
            odd  = list(range(0, n_trials, 2))
            even = list(range(1, n_trials, 2))
            idxs_by_task.append((odd, even))
            qualifying.append(t)
    K = len(qualifying)
    if K == 0:
        return None

    def _half_neural(half_key):
        cols = []
        for (odd, even), task_no in zip(idxs_by_task, qualifying):
            idxs = odd if half_key == 1 else even
            sub = neu[task_no][:, idxs, :]                        # (n_neurons, n_sub, 360)
            sub = _apply_residualise(sub, phase_residualise)
            avg = np.nanmean(sub, axis=1)
            cols.append(avg.reshape(avg.shape[0], N, BINLEN).mean(axis=2))
        return np.hstack(cols)

    h1_neu = _half_neural(1)
    h2_neu = _half_neural(2)
    mat = np.vstack([h1_neu.T, h2_neu.T])
    mu = np.nanmean(mat, axis=0); sd = np.nanstd(mat, axis=0); sd[sd == 0] = 1
    mat_z = (mat - mu) / sd
    data_vec = my_RSA.compute_crosscorr(
        mat_z, plotting=False, include_diagonal=False,
        no_tasks=K, model='synth-halves')[0]

    def _half_models(half_key):
        out = {'dsr': [], 'stat': [], 'loc': [], 'phas': [], 'midn': [],
                'dsr_fmri': []}
        for (odd, even), task_no in zip(idxs_by_task, qualifying):
            idxs = odd if half_key == 1 else even
            trial_locs = np.asarray(loc[task_no])[idxs]          # (n_sub, 360)
            walked = ae._mode_path_360(trial_locs)
            loc_m, phas_m, stat_m, midn_m, dsr_m, _, _ = predictions.model_DSR(
                locations=walked, no_phase_neurons=NUMBER_PHASE_NEURONS)
            for key, M in [('dsr', dsr_m), ('stat', stat_m), ('loc', loc_m),
                            ('phas', phas_m), ('midn', midn_m)]:
                out[key].append(M.reshape(M.shape[0], N, BINLEN).mean(axis=2))
            out['dsr_fmri'].append(ae._build_dsr_fmri_task(walked, N))
        return out

    h1_models = _half_models(1)
    h2_models = _half_models(2)
    model_rdms = {}
    for k in h1_models:
        if k == 'dsr_fmri':
            h1 = np.concatenate(h1_models[k], axis=0)
            h2 = np.concatenate(h2_models[k], axis=0)
            H12 = np.mean(h1[:, None, :] != h2[None, :, :], axis=-1)
            H12 = 0.5 * (H12 + H12.T)
            n = H12.shape[0]
            model_rdms[k] = H12[np.triu_indices(n, k=1)]
        else:
            h1 = np.hstack(h1_models[k])
            h2 = np.hstack(h2_models[k])
            m_combined = np.vstack([h1.T, h2.T])
            model_rdms[k] = my_RSA.compute_crosscorr(
                m_combined, plotting=False, include_diagonal=False,
                no_tasks=K, model=k)[0]
    return {'data': data_vec, 'models': model_rdms, 'K': K}


# ── Main ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading rodent data ...")
    mouse_data      = ae.load_ephys_data(DATA_FOLDER, raw=True)
    mouse_data_norm = ae.load_ephys_data(DATA_FOLDER, raw=False)
    keep_by_recday  = ae.cross_view_session_ids(mouse_data, mouse_data_norm)
    recdays = sorted(mouse_data)
    print(f"  {len(recdays)} recdays.")

    pooled = {}
    n_neurons = {}
    for recday in recdays:
        cfg, loc, neu, tim, _ = ae.clean_ephys_data(
            mouse_data_norm[recday]['rewards_configs'],
            mouse_data_norm[recday]['locations'],
            mouse_data_norm[recday]['neurons'],
            mouse_data_norm[recday]['timings'], recday,
            session_ids=mouse_data_norm[recday]['session_ids'],
            keep_session_ids=keep_by_recday[recday],
            return_metadata=True)
        sid = keep_by_recday[recday]
        cfg_p, loc_p, neu_p, tim_p, _ = ae.pool_by_task_config(
            cfg, loc, neu, tim, kind='norm',
            session_ids=sid, return_metadata=True)
        pooled[recday] = (cfg_p, loc_p, neu_p, tim_p)
        n_neurons[recday] = int(neu_p[0].shape[0])

    # ── Main diagnostic loop ──────────────────────────────────────────
    rows = []
    for recday in recdays:
        cfg, loc, neu, tim = pooled[recday]
        n_tasks = len(cfg)
        print(f"\n[{recday}]  n_tasks={n_tasks}  n_neurons={n_neurons[recday]}")
        for variant_name, vc in VARIANTS:
            data_rdms = build_data_rdms(neu, cfg,
                                         phase_residualise=vc['phase_residualise'])
            model_rdms = build_model_rdms(loc, n_tasks,
                                           pool_method=vc['pool_method'])
            for slc in SLICES:
                dv = data_rdms[slc]
                mv = model_rdms[slc]
                # Correlations between model RDMs
                r_dsr_dsr_fmri = spearman(mv['dsr'], mv['dsr_fmri'])
                r_dsr_phas     = spearman(mv['dsr'], mv['phas'])
                r_dsr_fmri_phas = spearman(mv['dsr_fmri'], mv['phas'])
                # Alone fits
                a_dsr      = rsa_alone(dv, mv['dsr'])
                a_dsr_fmri = rsa_alone(dv, mv['dsr_fmri'])
                a_phas     = rsa_alone(dv, mv['phas'])
                # Joint OLS: data ~ dsr + dsr_fmri + phas
                j_dphas = rsa_joint(dv, mv, ['dsr', 'dsr_fmri', 'phas'])
                # Canonical rodent joint with full control stack — target = dsr
                j_dsr_ctrl = rsa_joint(
                    dv, mv, ['dsr', 'stat', 'loc', 'phas', 'midn'])
                # Same stack but swap dsr for dsr_fmri — target = dsr_fmri
                j_dsr_fmri_ctrl = rsa_joint(
                    dv, mv, ['dsr_fmri', 'stat', 'loc', 'phas', 'midn'])
                # Both targets in one 6-model joint (matches the main-pipeline
                # 'DSR (fMRI)' column: what's dsr_fmri worth when dsr is also
                # in the model?)
                j_both_ctrl = rsa_joint(
                    dv, mv, ['dsr', 'dsr_fmri', 'stat', 'loc', 'phas', 'midn'])
                # Incremental control drop-in for dsr_fmri — reveals WHICH
                # control eats its β.
                dropin = dropin_series(
                    dv, mv, target='dsr_fmri',
                    control_order=['loc', 'stat', 'phas', 'midn'])
                dropin_flat = {f'dropin_beta_dsr_fmri__{d["ctrls_in"]}': d['beta']
                                for d in dropin}
                rows.append({
                    'recday': recday,
                    'n_neurons': n_neurons[recday],
                    'variant': variant_name,
                    'slice': slc,
                    'r_dsr_vs_dsr_fmri':   r_dsr_dsr_fmri,
                    'r_dsr_vs_phas':        r_dsr_phas,
                    'r_dsr_fmri_vs_phas':   r_dsr_fmri_phas,
                    'beta_dsr_alone':      a_dsr['beta'],
                    't_dsr_alone':         a_dsr['t'],
                    'beta_dsr_fmri_alone': a_dsr_fmri['beta'],
                    't_dsr_fmri_alone':    a_dsr_fmri['t'],
                    'beta_phas_alone':     a_phas['beta'],
                    't_phas_alone':        a_phas['t'],
                    # dsr+dsr_fmri+phas (the previous 3-model joint)
                    'joint3_beta_dsr':      j_dphas['dsr']['beta'],
                    'joint3_beta_dsr_fmri': j_dphas['dsr_fmri']['beta'],
                    'joint3_beta_phas':     j_dphas['phas']['beta'],
                    # dsr + 4 controls (canonical rodent joint)
                    'joint_ctrl_beta_dsr':      j_dsr_ctrl['dsr']['beta'],
                    'joint_ctrl_t_dsr':         j_dsr_ctrl['dsr']['t'],
                    'joint_ctrl_p_dsr':         j_dsr_ctrl['dsr']['p'],
                    # dsr_fmri + 4 controls
                    'joint_ctrl_beta_dsr_fmri': j_dsr_fmri_ctrl['dsr_fmri']['beta'],
                    'joint_ctrl_t_dsr_fmri':    j_dsr_fmri_ctrl['dsr_fmri']['t'],
                    'joint_ctrl_p_dsr_fmri':    j_dsr_fmri_ctrl['dsr_fmri']['p'],
                    # both targets + controls (matches main pipeline)
                    'joint_both_beta_dsr':      j_both_ctrl['dsr']['beta'],
                    'joint_both_beta_dsr_fmri': j_both_ctrl['dsr_fmri']['beta'],
                    'joint_both_beta_loc':      j_both_ctrl['loc']['beta'],
                    'joint_both_beta_stat':     j_both_ctrl['stat']['beta'],
                    'joint_both_beta_phas':     j_both_ctrl['phas']['beta'],
                    'joint_both_beta_midn':     j_both_ctrl['midn']['beta'],
                    **dropin_flat,
                })

    df = pd.DataFrame(rows)

    # ── Write the MAIN table FIRST so a downstream error doesn't cost the whole run ──
    run_tag = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_diagnostic'
    out_dir = os.path.join(OUT_BASE, run_tag)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'diagnostic_summary.csv'), index=False)
    print(f"\nWrote diagnostic_summary.csv ({len(df)} rows) → {out_dir}")

    # ── Synthetic-halves diagnostic (per-recday try/except so one bad recday
    # doesn't block the others) ────────────────────────────────────
    halves_rows = []
    for recday in recdays:
        cfg, loc, neu, tim = pooled[recday]
        for variant_name, vc in VARIANTS:
            if vc['pool_method'] == 'per_trial_modes':
                continue    # keep synthetic-halves simple, mode_path only
            try:
                sh = synthetic_halves_matrices(
                    cfg, loc, neu,
                    phase_residualise=vc['phase_residualise'])
            except Exception as exc:
                print(f"  [synth-halves skip] {recday} / {variant_name}: {exc}")
                continue
            if sh is None:
                continue
            dv = sh['data']; mv = sh['models']
            r_dsr_dsr_fmri = spearman(mv['dsr'], mv['dsr_fmri'])
            a_dsr      = rsa_alone(dv, mv['dsr'])
            a_dsr_fmri = rsa_alone(dv, mv['dsr_fmri'])
            j = rsa_joint(dv, mv, ['dsr', 'dsr_fmri', 'phas'])
            halves_rows.append({
                'recday': recday,
                'n_neurons': n_neurons[recday],
                'variant': variant_name,
                'K_qualifying': sh['K'],
                'r_dsr_vs_dsr_fmri': r_dsr_dsr_fmri,
                'beta_dsr_alone':      a_dsr['beta'],
                't_dsr_alone':         a_dsr['t'],
                'beta_dsr_fmri_alone': a_dsr_fmri['beta'],
                't_dsr_fmri_alone':    a_dsr_fmri['t'],
                'joint_beta_dsr':      j['dsr']['beta'],
                'joint_beta_dsr_fmri': j['dsr_fmri']['beta'],
                'joint_beta_phas':     j['phas']['beta'],
            })
    df_halves = pd.DataFrame(halves_rows)
    df_halves.to_csv(os.path.join(out_dir, 'halves_synthetic_summary.csv'),
                      index=False)
    print(f"Wrote halves_synthetic_summary.csv ({len(df_halves)} rows)")

    # ── Console report: top-3 / bottom-3 per (variant × slice) ────
    pd.set_option('display.width', 240)
    pd.set_option('display.max_columns', 30)
    for variant_name, _ in VARIANTS:
        for slc in SLICES:
            sub = (df[(df['variant'] == variant_name) & (df['slice'] == slc)]
                   .sort_values('beta_dsr_alone', ascending=False))
            cols = ['recday', 'n_neurons',
                    'beta_dsr_alone', 'beta_dsr_fmri_alone',
                    'joint_ctrl_beta_dsr', 'joint_ctrl_beta_dsr_fmri',
                    'joint_both_beta_dsr', 'joint_both_beta_dsr_fmri']
            print(f"\n=== {variant_name} · {slc} — top-3 by β_dsr (alone) ===")
            print(sub.head(3)[cols].round(3).to_string(index=False))
            print(f"=== {variant_name} · {slc} — bottom-3 by β_dsr (alone) ===")
            print(sub.tail(3)[cols].round(3).to_string(index=False))

    # ── Drop-in trace summary: median across recdays per variant × slice ──
    print("\n=== dsr_fmri β as controls are dropped in (median across recdays) ===")
    dropin_cols = [c for c in df.columns if c.startswith('dropin_beta_dsr_fmri__')]
    for variant_name, _ in VARIANTS:
        for slc in SLICES:
            sub = df[(df['variant'] == variant_name) & (df['slice'] == slc)]
            m = sub[dropin_cols].median()
            print(f"  {variant_name} · {slc}:")
            for c, v in m.items():
                label = c.replace('dropin_beta_dsr_fmri__', '')
                print(f"    ctrls={label:<22s}  β_dsr_fmri = {v:+.3f}")

    print(f"\nSynthetic-halves diagnostic (odd/even trial split):")
    for variant_name, _ in VARIANTS:
        if variant_name == 'P3_pertrial_resid':
            continue
        sub = (df_halves[df_halves['variant'] == variant_name]
                .sort_values('beta_dsr_alone', ascending=False))
        cols = ['recday', 'n_neurons', 'K_qualifying',
                'beta_dsr_alone', 'beta_dsr_fmri_alone',
                'r_dsr_vs_dsr_fmri',
                'joint_beta_dsr', 'joint_beta_dsr_fmri']
        print(f"\n=== halves · {variant_name} ===")
        print(sub[cols].round(3).to_string(index=False))
    print(f"\nDone. Outputs at:\n  {out_dir}")
