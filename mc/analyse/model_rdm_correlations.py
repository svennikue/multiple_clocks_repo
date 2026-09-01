#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collinearity of the regressors in an RSA design.

Answers "how correlated are the model RDMs that actually enter my OLS?"
for both analysis families in this project, using ONE definition of
"actually enter": the columns of the design matrix `X` that
`mc.analyse.my_RSA.evaluate_model` is called with -- after every mask the
pipeline applies, and after degenerate (constant / non-finite) columns are
dropped, exactly as `evaluate_combo_safe` does.

Two sources, one output shape:

  fMRI  `fmri_combo_design`  -- loads a subject's EV pickle
        (written by create_fMRI_model_RDMs_on_clean_beh.py), pairs
        conditions across task halves with `pair_correct_tasks`, and runs
        the same metric dispatch as
        fMRI_run_RSA_without_rsatoolbox_clean.py: Hamming for the
        order-preserving codes, cosine crosscorr otherwise, `A-state`
        NaNs -> 1, upper triangle with the config's `diagonal_included`,
        then the `masked_conds` category mask (path-path and
        reward-reward pairs only).

  cells `cells_combo_design` -- reads the model vectors saved by
        RSA_DSR_ROIs_simple.py into `rdms/rdms_<ROI>.npz` and applies the
        same phase mask that run used.

Both return (X, kept_model_names). `correlation_matrix` turns that into a
Pearson matrix; `fisher_mean` averages matrices over subjects. Plotting is
`mc.plotting.results.plot_model_correlation_matrix_pub`, which takes a
precomputed matrix and changes nothing about it.

@author: Svenja Kuchenhoff, 2026
"""

from __future__ import annotations

import numpy as np

import mc


# ── model -> metric dispatch, shared with the fMRI RSA ────────────────
# Keep in sync with fMRI_run_RSA_without_rsatoolbox_clean.py.
HAMMING_MODELS = [
    'location', 'DSR', 'prev_buttons', 'buttons_out', 'next_buttons',
    'phys_abstr_space', 'action_DSR', 'state_action_DSR',
    'state_action_glob', 'state_action_loc', 'rewDSR', 'pathDSR',
    'rew_stateactionDSR', 'path_stateactionDSR',
    'DSR_onefut', 'DSR_twofut', 'DSR_threefut', 'DSR_fourfut',
    'DSR_fivefut', 'DSR_sixfut', 'DSR_sevenfut',
    'curr_quarter', 'next_quarter', 'next2_quarter', 'next3_quarter',
    'rot_curr_quarter', 'rot_next_quarter', 'rot_next2_quarter',
    'rot_next3_quarter',
]

# Short labels so a 4 x 4 cm panel stays readable. Covers both families.
DISPLAY_NAMES = {
    # fMRI EV names
    'DSR': 'DSR', 'location': 'location', 'A-state': 'A-state',
    'l2_norm': 'L2 norm', 'next_buttons': 'next buttons',
    'prev_buttons': 'prev buttons', 'buttons_out': 'buttons out',
    'rot_curr_quarter': 'DSR now', 'rot_next_quarter': 'DSR +1',
    'rot_next2_quarter': 'DSR +2', 'rot_next3_quarter': 'DSR +3',
    'state': 'state', 'path_rew': 'path/rew', 'duration': 'duration',
    # cell-RSA names
    'dsr_fmri': 'DSR', 'dsr_fmri_fut': 'DSR future', 'dsr_fmri_informed': 'DSR 30/60',
    'bttn_curr': 'curr button', 'bttn_next': 'next button',
    'bttn_prev': 'prev button', 'curr_quarter': 'DSR now',
    'next_quarter': 'DSR +1', 'next2_quarter': 'DSR +2',
    'next3_quarter': 'DSR +3', 'phase': 'phase', 'midnight': 'midnight',
    'reward_path': 'path/rew', 'uncover': 'uncover',
    'repeat_counter': 'repeat', 'state_phase': 'state x phase',
}


def display_labels(models):
    """Short display label per model name, in the given order."""
    return [DISPLAY_NAMES.get(m, m.replace('_', ' ')) for m in models]


# ── fMRI ──────────────────────────────────────────────────────────────
def fmri_model_vector(model, model_EVs, EV_keys, include_diagonal):
    """One fMRI model -> its vectorised across-halves RDM.

    Same construction as fMRI_run_RSA_without_rsatoolbox_clean.py. Returns
    (vector, concatenated_EV) -- the second is needed to build the
    condition mask from `path_rew`.
    """
    th1, th2, _ = mc.analyse.my_RSA.pair_correct_tasks(model_EVs[model], EV_keys)
    concat = np.concatenate((th1, th2), axis=0)
    if model == 'path_rew':
        vec = mc.analyse.my_RSA.make_categorical_RDM(
            concat, plotting=False, include_diagonal=include_diagonal)[0]
    elif model == 'duration':
        vec = mc.analyse.my_RSA.make_distance_RDM(
            concat, plotting=False, include_diagonal=include_diagonal)[0]
    elif model in HAMMING_MODELS:
        vec = mc.analyse.my_RSA.compute_hamming_distance(
            concat, plotting=False, include_diagonal=include_diagonal,
            model_name=model)[0]
    else:
        vec = mc.analyse.my_RSA.compute_crosscorr(
            concat, plotting=False, include_diagonal=include_diagonal)[0]
        if model == 'A-state':
            # as in the RSA script: the NaN cells of A-state become 1
            vec = np.where(np.isnan(np.asarray(vec, dtype=float)), 1.0, vec)
    return np.asarray(vec, dtype=float), concat


def fmri_combo_design(model_EVs, models, *, include_diagonal=True,
                      masked_conditions=None):
    """Design matrix for one fMRI combo, as handed to the OLS.

    Returns (X, kept_models, mask_name, n_cells_total).
    """
    EV_keys = sorted(model_EVs['location'].keys())
    vectors = {m: fmri_model_vector(m, model_EVs, EV_keys, include_diagonal)[0]
               for m in models}

    if masked_conditions:
        _, path_rew_concat = fmri_model_vector(
            'path_rew', model_EVs, EV_keys, include_diagonal)
        masks = mc.analyse.my_RSA.make_category_masks(
            path_rew_concat, plotting=False, include_diagonal=include_diagonal,
            mask_only_path_rew_combos=True)
        if len(masks) != 1:
            raise ValueError(f"expected one condition mask, got {list(masks)}")
        mask_name, cell_mask = next(iter(masks.items()))
    else:
        cell_mask = np.ones(len(vectors[models[0]]), dtype=bool)
        mask_name = 'all_cells'

    X = np.stack([vectors[m][cell_mask] for m in models], axis=1)
    X, kept = drop_degenerate(X, models)
    return X, kept, mask_name, int(len(cell_mask))


# ── cells ─────────────────────────────────────────────────────────────
def cells_combo_design(npz, models, *, test_name='split_halves_z',
                       phase_mask=None):
    """Design matrix for one cell-RSA combo, as handed to the OLS.

    `npz` is a loaded `rdms/rdms_<ROI>.npz` from RSA_DSR_ROIs_simple.py.
    Model vectors are stored under the un-suffixed test name (the `_z`
    variants differ only in how the DATA RDM is scaled), so a
    'split_halves_z' request reads 'model__split_halves__<name>'.

    Returns (X, kept_models, n_cells_total).
    """
    base = test_name[:-2] if test_name.endswith('_z') else test_name
    cols = []
    for m in models:
        key = f'model__{base}__{m}'
        if key not in npz:
            raise KeyError(f"{key} not in npz (has {len(npz.files)} arrays)")
        cols.append(np.asarray(npz[key], dtype=float).ravel())
    X = np.stack(cols, axis=1)
    n_total = X.shape[0]
    if phase_mask is not None:
        X = X[np.asarray(phase_mask, dtype=bool), :]
    X, kept = drop_degenerate(X, models)
    return X, kept, n_total


# ── shared ────────────────────────────────────────────────────────────
def drop_degenerate(X, models, atol=1e-12):
    """Drop constant / non-finite columns, exactly as evaluate_combo_safe."""
    X = np.asarray(X, dtype=float)
    good = np.array([np.isfinite(X[:, i]).all() and np.nanvar(X[:, i]) > atol
                     for i in range(X.shape[1])])
    if not good.all():
        dropped = [models[i] for i in range(len(models)) if not good[i]]
        print(f"    [drop] degenerate under mask: {dropped}")
    return X[:, good], [models[i] for i in range(len(models)) if good[i]]


def correlation_matrix(X):
    """Pearson correlation between the columns of a design matrix."""
    return np.corrcoef(np.asarray(X, dtype=float), rowvar=False)


def fisher_mean(corr_stack):
    """Fisher-z mean of a (n_subjects, n, n) stack of correlation matrices."""
    stack = np.asarray(corr_stack, dtype=float)
    z = np.arctanh(np.clip(stack, -0.999999, 0.999999))
    return np.tanh(np.nanmean(z, axis=0))
