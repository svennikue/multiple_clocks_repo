#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 16:06:31 2025

easy linear model for states.


# START AGAIN.
# get one p-value per regressor, and bonferroni correct for amount of neurons (in this area)
# also test the distribution of slopes per ROI against zero


"""

import os
import mc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices, build_design_matrices
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from scipy import stats
from matplotlib.ticker import PercentFormatter

def get_data(sub, trials):
    data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
    if not os.path.isdir(data_folder):
        print("running on ceph")
        data_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives"
    if trials == 'residualised':
        res_data = True
    else:
        res_data = False
    data_norm = mc.analyse.helpers_human_cells.load_norm_data(data_folder, [f"{sub:02}"], res_data = res_data)
    return data_norm, data_folder 


def make_long_df(
    neuron: np.ndarray,
    beh_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a long DataFrame (trials x bins) with y, state, state_phase,
    and repeated behavioral columns. Optionally add one-hot columns
    for each state and state_phase category.

    Parameters
    ----------
    neuron : np.ndarray, shape (n_trials, 360)
    beh_df : pd.DataFrame, length n_trials, with columns:
             'rep_correct', 'grid_no', 'correct'
    add_one_hot : bool, default True
        If True, add one-hot columns for each category in state and state_phase.
    drop_first : bool, default False
        If True, drop the first category in each set (useful if you'll include
        an intercept in a linear model).

    Returns
    -------
    df_long : pd.DataFrame, shape (n_trials*360, ...)
    """

    n_trials, n_bins = neuron.shape
    assert n_bins == 360, "Expected 360 bins per trial."
    assert len(beh_df) == n_trials, "beh_df length must match neuron.shape[0]."

    # --- category definitions ---
    state_categories = ["A", "B", "C", "D"]
    state_bins = {
        "A": slice(0, 90),
        "B": slice(90, 180),
        "C": slice(180, 270),
        "D": slice(270, 360),
    }
    state_phase_categories = [
        "A_early", "A_mid", "A_rew",
        "B_early", "B_mid", "B_rew",
        "C_early", "C_mid", "C_rew",
        "D_early", "D_mid", "D_rew",
    ]
    phase_bins = {
        "A_early": slice(0, 30),   "A_mid": slice(30, 60),   "A_rew": slice(60, 90),
        "B_early": slice(90, 120), "B_mid": slice(120, 150), "B_rew": slice(150, 180),
        "C_early": slice(180, 210),"C_mid": slice(210, 240), "C_rew": slice(240, 270),
        "D_early": slice(270, 300),"D_mid": slice(300, 330), "D_rew": slice(330, 360),
    }

    # --- per-trial bin labels ---
    state_per_bin = np.empty(n_bins, dtype=object)
    for lbl, sl in state_bins.items():
        state_per_bin[sl] = lbl

    state_phase_per_bin = np.empty(n_bins, dtype=object)
    for lbl, sl in phase_bins.items():
        state_phase_per_bin[sl] = lbl

    # --- expand over all trials ---
    y_long = neuron.reshape(-1)
    state_long = np.tile(state_per_bin, n_trials)
    state_phase_long = np.tile(state_phase_per_bin, n_trials)

    rep_correct_long = np.repeat(beh_df["rep_correct"].to_numpy(), n_bins)
    grid_no_long     = np.repeat(beh_df["grid_no"].to_numpy(), n_bins)
    correct_long     = np.repeat(beh_df["correct"].to_numpy(), n_bins)

    # --- assemble df ---
    df_long = pd.DataFrame({
        "y": y_long,
        "state": state_long,
        "state_phase": state_phase_long,
        "rep_correct": rep_correct_long,
        "grid_no": grid_no_long,
        "correct": correct_long,
    })

    # keep ordered categoricals
    df_long["state"] = pd.Categorical(df_long["state"],
                                      categories=state_categories, ordered=True)
    df_long["state_phase"] = pd.Categorical(df_long["state_phase"],
                                            categories=state_phase_categories, ordered=True)
    
    # which categories to encode (optionally drop first)
    state_cats_encode = state_categories
    phase_cats_encode = state_phase_categories

    # one-hot (int8 for memory efficiency)
    svals = df_long["state"].to_numpy()
    for cat in state_cats_encode:
        df_long[cat] = (svals == cat).astype("int8")

    spvals = df_long["state_phase"].to_numpy()
    for cat in phase_cats_encode:
        df_long[cat] = (spvals == cat).astype("int8")


    df_long = df_long.replace([np.inf, -np.inf], np.nan)
    df_long = df_long.dropna(subset=["y","A","B","C","D","rep_correct","correct"])
    # keep valid one-hot rows
    df_long = df_long[df_long[["A","B","C","D"]].sum(axis=1) == 1]
    # center repeat count per neuron
    mu_rep = df_long["rep_correct"].mean()
    df_long["rep_c"] = df_long["rep_correct"] - mu_rep
    # import pdb; pdb.set_trace()
    return df_long





# first step: load data of a single neuron.

# I THINK that session 50, 05 elect36 left insular, is an A state-neuron. look at that one first.
# '50_05-05-elec36-LINS'
sesh = 50
trials = 'all_minus_explore'
# load data
data_raw, source_dir = get_data(sesh, trials=trials)
group_dir_state = f"{source_dir}/group/state_tuning"

print(f"this is the folder {group_dir_state}")


# filter data for only those repeats that were 1) correct and 2) not the first one
data = mc.analyse.helpers_human_cells.filter_data(data_raw, sesh, trials)
beh_df = data[f"sub-{sesh:02}"]['beh'].copy()
neurons = data[f"sub-{sesh:02}"]['normalised_neurons'].copy()
neuron_labels = []
roi_label = []
for n in neurons:
    neuron_labels.append(n)
    if 'ACC' in n or 'vCC' in n or 'AMC' in n or 'vmPFC' in n:
                roi = 'ACC'
    elif 'PCC' in n:
        roi = 'PCC'
    elif 'OFC' in n:
        roi = 'OFC'
    elif 'MCC' in n or 'HC' in n:
        roi = 'hippocampal'
    elif 'EC' in n:
        roi = 'entorhinal'
    elif 'AMYG' in n:
        roi = 'amygdala'
    else:
        roi = 'mixed'
    roi_label.append(roi)
    
# next, transform into long format:
# one long df that is 248*360 long, 
#neuron: np.ndarray of shape (248, 360)  # trials x bins
#    beh_df: pd.DataFrame with length 248 and columns: 'rep_correct', 'grid_no', 'correct'
#    Returns a long/tidy DataFrame with 248*360 rows and columns:
#      - y (the original neuron values)
#      - state (A/B/C/D per 90 bins)
#      - state_phase (A_early, A_mid, A_rew, ..., D_rew per 30 bins)
#      - rep_correct, grid_no, correct (repeated for each trial across its 360 bins)
neuron = neurons['50_05-05-elec36-LINS'].to_numpy()
df = make_long_df(neuron, beh_df)
df.head()


# now, fit the full model with interaction terms.
# get one p-value per regressor -> for group stats
# and the beta (slope) per regressor -> for group stats

cols = ["A","B","C","D","rep_c","correct"]

X = df[cols].to_numpy(float)
y = df["y"].to_numpy(float)
res = sm.OLS(y, X).fit(cov_type="HC3")

tab = pd.DataFrame({
    "beta":   res.params,
    "t":      res.tvalues,
    "pval":   res.pvalues,
}, index=cols)

import pdb; pdb.set_trace()

print(pd.concat([betas, tstats, pvalues], axis=1))


# Block tests
state_F  = res.f_test("A = B = C = D = 0")
inter_F  = res.f_test("A_rep = B_rep = C_rep = D_rep = 0")

    
