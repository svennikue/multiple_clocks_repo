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
    
    # NEW: state × repeat interactions (keep it simple)
    df_long["A_rep"] = df_long["A"].astype("float32") * df_long["rep_c"].astype("float32")
    df_long["B_rep"] = df_long["B"].astype("float32") * df_long["rep_c"].astype("float32")
    df_long["C_rep"] = df_long["C"].astype("float32") * df_long["rep_c"].astype("float32")
    df_long["D_rep"] = df_long["D"].astype("float32") * df_long["rep_c"].astype("float32")

    # import pdb; pdb.set_trace()
    # CENTRE ALL VARIABLES!!! y and states etc.
    
    return df_long


def nested_f_test(res_full, k_full, res_reduced, k_reduced):
    """Classic nested-model F-test from two OLS fits.
    res_* are statsmodels results; k_* are number of regressors in each model."""
    sse_full = float(np.sum(res_full.resid**2))
    sse_red  = float(np.sum(res_reduced.resid**2))
    df1 = int(k_full - k_reduced)
    df2 = int(res_full.df_resid)
    if df1 <= 0 or df2 <= 0 or sse_full <= 0:
        return np.nan, np.nan
    F = ((sse_red - sse_full) / df1) / (sse_full / df2)
    p = stats.f.sf(F, df1, df2)
    return F, p


def determine_roi(n):
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
    return roi



def run_linear_models(df, session, neuron_id, roi):
    result_rows = [] 
    # now, fit the full model with interaction terms.
    # get one p-value per regressor -> for group stats
    # and the beta (slope) per regressor -> for group stats
    # import pdb; pdb.set_trace()
    y = df["y"].to_numpy(float)
    y = stats.zscore(y)

    # Model 1 (main effects): y ~ A+B+C+D + rep_c + correct (no intercept)
    # drop C because of collinearity issue and add an intercept.
    #cols_m1 = ["A","B","C","D","rep_c","correct"]
    cols_m1 = ["A","B", "C", "D","rep_c","correct"]
    # here z-score each variable, y as well as each X var
    # will make beta and t the same and the betas will be interpretable
    # 
    
    X_m1 = df[cols_m1].astype(float)
    # std = np.array(X_m1.std())
    # if (not np.isnan(std).any()) and (not (std == 0).any()):
    #     # put a condition that checks for nans
    #     # because then the zscoring doesnt worl
    #     X_m1 = stats.zscore(X_m1, axis=0)
    
    res1 = sm.OLS(y, X_m1).fit(cov_type="HC3")
    
    # #
    # #
    # #
    # y = df["y"].to_numpy(float)
    # # y = stats.zscore(y)

    # # Model 1 (main effects): y ~ A+B+C+D + rep_c + correct (no intercept)
    # # drop C because of collinearity issue and add an intercept.
    # #cols_m1 = ["A","B","C","D","rep_c","correct"]
    # cols_m1 = ["A","B","D","rep_c","correct"]
    # # here z-score each variable, y as well as each X var
    # # will make beta and t the same and the betas will be interpretable
    # # 
    
    
    # X_m1 = df[cols_m1].astype(float)
    # std = np.array(X_m1.std())
    # if (not np.isnan(std).any()) and (not (std == 0).any()):
    #     # put a condition that checks for nans
    #     # because then the zscoring doesnt worl
    #     X_m1 = stats.zscore(X_m1, axis=0)
    
    
    # # manual fit 
    # np.linalg.pinv(X_m1) @ y
    # res1 = sm.OLS(y, X_m1).fit(cov_type="HC3")
    
    
    # # do the same thing with random data.
    # y_rand = np.random.rand(45000,)
    # y_rand = stats.zscore(y_rand)

    # cols_3_states = ["A","B","D","rep_c","correct"]
    # cols_4_states = ["A","B","C","D","rep_c","correct"]
    
    # X_4 = df[cols_4_states].astype(float)
    # X_3 = df[cols_3_states].astype(float)
    
    # res3_rand = sm.OLS(y_rand, X_3).fit(cov_type="HC3")
    # res4_rand = sm.OLS(y_rand, X_4).fit(cov_type="HC3")
    
    
    # # testing random vs not random.
    # # testing 4 states vs 3 states.
    
    
    # # dont demean abcd, but demean the data. 
    # # look at feat fsl website for main effect of state (F test)
    

    #########
    ########
    
    
    # beta = np.linalg.pinv(X_m1) @ y
    
    # import pdb; pdb.set_trace()
    # per-regressor rows (M1)
    for term in cols_m1:
        result_rows.append({
            "session": session, "neuron": neuron_id, "roi": roi,
            "model": "M1_main", "term": term,
            "beta": float(res1.params.get(term, np.nan)),
            "t":    float(res1.tvalues.get(term, np.nan)),
            "p":    float(res1.pvalues.get(term, np.nan)),
            "F":    np.nan
        })
    # State main effect.
    #f_state = res1.f_test("A = 0, B = 0, C = 0, D = 0")
    cols_red_state = ["rep_c","correct"]
    X_red_state = df[cols_red_state].astype(float)
    std = np.array(X_red_state.std())
    if (not np.isnan(std).any()) and (not (std == 0).any()):
        # put a condition that checks for nans
        # because then the zscoring doesnt worl
        X_red_state = stats.zscore(X_red_state, axis = 0)
    
    
    res1_reduced = sm.OLS(y, X_red_state).fit()  # classic for SSE
    # res1_full can reuse res1, but we need classic SSE; refit without robust is simplest:
    res1_full_classic = sm.OLS(y, X_m1).fit()
    F_state, p_state = nested_f_test(res1_full_classic, k_full=len(cols_m1),
                                     res_reduced=res1_reduced, k_reduced=len(cols_red_state))
    
    result_rows.append({
        "session": session, "neuron": neuron_id, "roi": roi,
        "model": "M1_main", "term": "State_main_effect",
        "beta": np.nan, "t": np.nan,
        "p": p_state, "F": F_state
    })

    # Model 2 (with interaction): y ~ A+B+C+D + rep_c + correct + (A_rep+B_rep+C_rep+D_rep)
    #cols_m2 = ["A","B","C","D","rep_c","correct","A_rep","B_rep","C_rep","D_rep"]
    cols_m2 = ["A","B","D","rep_c","correct","A_rep","B_rep","C_rep","D_rep"]
    X_m2 = df[cols_m2].astype(float)
    std = np.array(X_m2.std())
    if (not np.isnan(std).any()) and (not (std == 0).any()):
        # put a condition that checks for nans
        # because then the zscoring doesnt worl
        X_m2 = stats.zscore(X_m2, axis = 0)
    res2 = sm.OLS(y, X_m2).fit(cov_type="HC3")

    for term in cols_m2:
        result_rows.append({
            "session": session, "neuron": neuron_id, "roi": roi,
            "model": "M2_interact", "term": term,
            "beta": float(res2.params.get(term, np.nan)),
            "t":    float(res2.tvalues.get(term, np.nan)),
            "p":    float(res2.pvalues.get(term, np.nan)),
            "F":    np.nan
        })
        
    # STATExREP via nested F: reduced = M1, full = M2
    res2_reduced = sm.OLS(y, X_m1).fit()  # classic for SSE
    res2_full_classic = sm.OLS(y, X_m2).fit()
    F_int, p_int = nested_f_test(res2_full_classic, k_full=len(cols_m2),
                                 res_reduced=res2_reduced, k_reduced=len(cols_m1))
    result_rows.append({
        "session": session, "neuron": neuron_id, "roi": roi,
        "model": "M2_interact", "term": "STATExREP_BLOCK",
        "beta": np.nan, "t": np.nan, "p": p_int, "F": F_int
    })

    return result_rows

    
def compute_state_lin_mod_all(sessions, trials, save_all=False):
    all_result_rows = []
    for sesh in sessions:
        # # first step: load data of a single neuron.
        
        # # I THINK that session 50, 05 elect36 left insular, is an A state-neuron. look at that one first.
        # # '50_05-05-elec36-LINS'
        # sesh = 50
        # trials = 'all_minus_explore'
        # load data
        data_raw, source_dir = get_data(sesh, trials=trials)
        group_dir_state = f"{source_dir}/group/state_tuning"
        # import pdb; pdb.set_trace()
        # if this session doesn't exist, skip
        if not data_raw:
            print(f"no raw data found for {sesh}, so skipping")
            continue
    
        # filter data for only those repeats that were 1) correct and 2) not the first one
        data = mc.analyse.helpers_human_cells.filter_data(data_raw, sesh, trials)
        beh_df = data[f"sub-{sesh:02}"]['beh'].copy()
        neurons = data[f"sub-{sesh:02}"]['normalised_neurons'].copy()
        
        # estimates_combined_model_rdms = Parallel(n_jobs=3)(delayed(mc.analyse.my_RSA.evaluate_model)(stacked_model_RDMs, d) for d in tqdm(data_RDMs, desc=f"running GLM for all searchlights in {combo_model_name}"))
        
        # parallelise this
        for n in neurons:
            df = make_long_df(neurons[n].to_numpy(), beh_df)
            roi = determine_roi(n)
            results = run_linear_models(df, sesh, n, roi)
            all_result_rows.extend(results)
    
    # import pdb; pdb.set_trace()
    results_df = pd.DataFrame(all_result_rows, columns=["session","neuron","roi","model","term","beta","t","p","F"])
    print(results_df.head())
    
    
    
    if save_all == True:
        group_dir_state = f"{source_dir}/group/state_lin_regs"
        if not os.path.isdir(group_dir_state):
            os.mkdir(group_dir_state)
        name_result = f"{group_dir_state}/state_rep_int_{trials}.csv"
        results_df.to_csv(name_result)

    import pdb; pdb.set_trace()
    print(f"saved cross-validated state tuning values in {name_result}")  
            
        

if __name__ == "__main__":
    # trials can be 'all', 'all_correct', 'early', 'late', 'all_minus_explore', 'residualised'
    # they can also be: 'first_correct', 'one_correct', ... 'nine_correct'
    # NOTE: if you do per-repeat estimates, use every grid! grid mean will be super unreliable anyways
    compute_state_lin_mod_all(sessions=list(range(0,64)), trials = 'all_minus_explore', save_all = True)
    