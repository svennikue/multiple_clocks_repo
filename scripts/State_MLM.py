#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:28:15 2025

@author: Svenja Küchenhoff


run a linear mixed effects model to test for the preferred states vs. the
effect of repeats per region.

1. first step: create a long table of all neurons, that contains what I want to model:  
    subject (ID)
    neuron (unique ID; e.g., "S12_N034")
    ROI (hippocampus / EC / ACC / …)
    task (label)
    repeat (1–10)
    state (A/B/C/D)
    y (the response you want to model)

2. fit an MLM 
y ~ ROI * state * repeat_c


"""

import os
import mc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices, build_design_matrices



sessions=list(range(0,64))
trials = 'all_minus_explore'
sparsity_c = 'gridwise_qc'
save_all=True
only_BCD = False

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


# first step: call long table, otherwise
sesh = sessions[0]
# load data
data_raw, source_dir = get_data(sesh, trials=trials)
group_dir_state = f"{source_dir}/group/state_tuning"
    
#if not os.path.isfile(f"{group_dir_state}/avg_state_long_table.csv"):
if os.path.isfile(f"{group_dir_state}/avg_state_long_table.csv"):
    # create a long table of all neurons, that contains what I want to model:
        
    # subject (ID)
    # neuron (unique ID; e.g., "S12_N034")
    # ROI (hippocampus / EC / ACC / …)
    # task (label)
    # repeat (1–10)
    # state (A/B/C/D)
    # y (the response you want to model)
    
    # boundaries for A,B,C,D within 360 timepoints
    
    state_bins = {"A": slice(0, 90), "B": slice(90, 180), "C": slice(180, 270), "D": slice(270, 360)}
    
    phase_bins = {"A_early": slice(0, 30), "A_mid": slice(30, 60),"A_rew": slice(60, 90), 
                  "B_early": slice(90, 120), "B_mid": slice(120, 150),"B_rew": slice(150, 180),
                  "C_early": slice(180, 210),"C_mid": slice(210, 240),"C_rew": slice(240, 270),
                  "D_early": slice(270, 300),"D_mid": slice(300, 330),"D_rew": slice(330, 360)}
    

    
    
    
    rows = []
    rows_phase = []
    for sesh in sessions:
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
        
        # clean each neuron such that you only consider the repeats that are reliable
        for neuron_idx, curr_neuron in enumerate(data[f"sub-{sesh:02}"]['normalised_neurons']):
            print(f"adding neuron {curr_neuron}")
            # resetting unique tasks for each neuron.
            # determine identical grids
            grid_cols = ['loc_A', 'loc_B', 'loc_C', 'loc_D']
            unique_grids, _, idx_same_grids, _ = np.unique(
                beh_df[grid_cols].to_numpy(),
                axis=0,
                return_index=True,
                return_inverse=True,
                return_counts=True
            )
        
            beh_df['idx_same_grids'] = idx_same_grids
    
            # clean the data such that I don't consider 'bad' blocks of repeats
            # with super low firing 
            if sparsity_c:
                beh_df = mc.analyse.helpers_human_cells.extract_consistent_grids(data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].to_numpy(), curr_neuron, beh_df)
                consistent_grids_mask = beh_df[f'consistent_FR_{curr_neuron}'].to_numpy()
                # set grid indexes I want to ignore to -1
                idx_same_grids[~consistent_grids_mask] = -1
                
                # if after excluding inconsistent grids there aren't enough grids for CV left,
                # kick this neuron.
                unique_grids = np.unique(beh_df['idx_same_grids'][beh_df[f'consistent_FR_{curr_neuron}']])
            
            if sparsity_c and len(unique_grids) < 3:
                print(f"excluding {curr_neuron} in sesh {sesh} because there were not enough grids with consistent FR!")
                continue
            
            # based on the cleaned behaviour, store the average value per state.
            for count_task, task_id in enumerate(unique_grids):
                mask_task = (idx_same_grids == task_id)
                neuron_curr_task = data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].loc[mask_task].to_numpy()
                # first z-score the neuron within task
                neuron_curr_task = (neuron_curr_task - neuron_curr_task.mean()) / neuron_curr_task.std(ddof=1)
                beh_curr_task = beh_df[mask_task]
                
                for corr_rep in range(0,10):
                    # import pdb; pdb.set_trace()  
                    mask_rep = (beh_curr_task['rep_correct'].isin([corr_rep])& beh_curr_task['correct']== 1)
                    neuron_curr_task_curr_rep = np.mean(neuron_curr_task[mask_rep], axis =0)
                    #roi
                    for s, st_bins in state_bins.items():
                        rows.append({
                            "subject": sesh,
                            "neuron_id": f"{sesh}_{curr_neuron}",
                            "task": f"task_{task_id}",
                            "repeat": corr_rep,
                            "state": s,
                            "y": float(np.mean(neuron_curr_task_curr_rep[st_bins]))
                        })
                    for ph, ph_bins in phase_bins.items():
                        rows_phase.append({
                            "subject": sesh,
                            "neuron_id": f"{sesh}_{curr_neuron}",
                            "task": f"task_{task_id}",
                            "repeat": corr_rep,
                            "phase": ph,
                            "y": float(np.mean(neuron_curr_task_curr_rep[ph_bins]))
                            })
 
    
    df = pd.DataFrame(rows)
    
    df["state"] = pd.Categorical(df["state"], ["A","B","C","D"], ordered=True)
    df["repeat_c"] = df["repeat"] - df["repeat"].mean()
    df['roi']= mc.analyse.helpers_human_cells.rename_rois(df,collapse_pfc=False,plot_by_cingulate_and_MTL=False)
    df["roi"] = df["roi"].astype("category")
                
    # store the df so you don't always need to regenerate.
    df.to_csv(f"{group_dir_state}/avg_state_long_table.csv")         
    
     
    
    
    df_ph = pd.DataFrame(rows_phase)
    
    df_ph["phase"] = pd.Categorical(df_ph["phase"], ["A_early", "A_mid", "A_rew",
                                                  "B_early", "B_mid", "B_rew",
                                                  "C_early", "C_mid", "C_rew",
                                                  "D_early", "D_mid", "D_rew"], ordered=True)
    
    df_ph["repeat_c"] = df_ph["repeat"] - df_ph["repeat"].mean()
    df_ph['roi']= mc.analyse.helpers_human_cells.rename_rois(df_ph,collapse_pfc=False,plot_by_cingulate_and_MTL=False)
    df_ph["roi"] = df_ph["roi"].astype("category")
    # store the df so you don't always need to regenerate.
    df_ph.to_csv(f"{group_dir_state}/avg_phase-state_long_table.csv")         

    import pdb; pdb.set_trace() 
else:
    df_ph = pd.read_csv(f"{group_dir_state}/avg_phase-state_long_table.csv")
    df = pd.read_csv(f"{group_dir_state}/avg_state_long_table.csv")
                


import pdb; pdb.set_trace()             
 

results = []
roi_list = list(df["roi"].cat.categories)
for roi, d in df.groupby("roi"):
    d = d.dropna(subset=["y","state","repeat","neuron_id"]).copy()
    d["repeat_c"] = d["repeat"] - d["repeat"].mean()

    # Build design explicitly so we know exactly which rows are used
    y, X = dmatrices("y ~ C(state) * repeat_c", d, return_type="dataframe")

    # Cluster ids aligned to those rows
    groups = d.loc[y.index, "neuron_id"].to_numpy()

    # Fit OLS with cluster-robust SEs (no get_robustcov_results needed)
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})

    # Tests
    f_state = res.f_test("C(state)[T.B] = 0, C(state)[T.C] = 0, C(state)[T.D] = 0")
    f_inter = res.f_test(
        "C(state)[T.B]:repeat_c = 0, C(state)[T.C]:repeat_c = 0, C(state)[T.D]:repeat_c = 0"
    )

    print(f"\nROI: {roi}")
    print(f"State main effect:      F={float(f_state.fvalue):.2f}, p={float(f_state.pvalue):.3g}")
    print(f"State × Repeat effect:  F={float(f_inter.fvalue):.2f}, p={float(f_inter.pvalue):.3g}")


def plot_roi_state_repeat_estimate(d_roi, roi_name):
    d_roi = d_roi.dropna(subset=["y","state","repeat","neuron_id"]).copy()
    d_roi["state"] = pd.Categorical(d_roi["state"], ["A","B","C","D"], ordered=True)
    d_roi["repeat_c"] = d_roi["repeat"] - d_roi["repeat"].mean()

    # Build design so rows used are explicit
    y, X = dmatrices("y ~ C(state) * repeat_c", d_roi, return_type="dataframe")
    groups = d_roi.loc[y.index, "neuron_id"].to_numpy()

    # OLS with cluster-robust SEs (by neuron)
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})

    # Prediction grid (1..10 repeats, all states), with same design columns
    reps = np.arange(1, 11)
    grid = pd.DataFrame([(s, r, r - d_roi["repeat"].mean())
                         for s in ["A","B","C","D"] for r in reps],
                        columns=["state","repeat","repeat_c"])
    Xg = build_design_matrices([X.design_info], grid)[0]
    Xg = np.asarray(Xg)
    beta = res.params.values
    cov  = res.cov_params().values

    yhat = Xg @ beta
    se   = np.sqrt(np.einsum("ij,jk,ik->i", Xg, cov, Xg))   # diag(Xg cov Xg^T)
    grid["y_hat"] = yhat
    grid["lo"] = yhat - 1.96*se
    grid["hi"] = yhat + 1.96*se

    # Plot
    plt.figure(figsize=(7,4))
    for s in ["A","B","C","D"]:
        g = grid[grid["state"] == s]
        plt.plot(g["repeat"], g["y_hat"], marker="o", label=s)
        plt.fill_between(g["repeat"], g["lo"], g["hi"], alpha=0.15)
    plt.axhline(0, ls="--", lw=1)
    plt.title(f"{roi_name}: modelled state profiles across repeats")
    plt.xlabel("Repeat"); plt.ylabel("Predicted response")
    plt.legend(title="State"); plt.tight_layout(); plt.show()

    # Strongest state per repeat (based on model predictions)
    wide = grid.pivot(index="repeat", columns="state", values="y_hat")
    strongest = wide.idxmax(axis=1)
    print(f"\n{roi_name} — strongest state per repeat (model predictions):")
    print(strongest.to_string())
    return res, grid




def plot_wta_proportions(d_roi, roi_name):
    d = d_roi.dropna(subset=["y","state","repeat","neuron_id"]).copy()
    d["state"] = pd.Categorical(d["state"], ["A","B","C","D"], ordered=True)

    # neuron-level means per state×repeat
    dn = d.groupby(["neuron_id","state","repeat"])["y"].mean().reset_index()
    # pivot to states as columns; some neurons may miss a state→dropna row-wise
    mat = dn.pivot_table(index=["neuron_id","repeat"], columns="state", values="y")
    mat = mat.dropna(how="any")  # keep rows with all 4 states present

    winner = mat.idxmax(axis=1)   # state label per (neuron, repeat)
    prop = (winner.groupby(level="repeat")
                  .value_counts(normalize=True)
                  .unstack(fill_value=0)
                  .reindex(columns=["A","B","C","D"], fill_value=0))

    plt.figure(figsize=(7,4))
    for s in ["A","B","C","D"]:
        plt.plot(prop.index, prop[s], marker="o", label=s)
    plt.ylim(0,1)
    plt.title(f"{roi_name}: proportion of neurons preferring each state")
    plt.xlabel("Repeat"); plt.ylabel("Proportion of neurons")
    plt.legend(title="State"); plt.tight_layout(); plt.show()

    print(f"\n{roi_name} — WTA proportions (head):")
    print(prop.head())



def plot_fit_vs_raw_for_roi(d_roi, roi_name):
    # tidy + center
    d = d_roi.dropna(subset=["y","state","repeat","neuron_id"]).copy()
    d["state"]  = pd.Categorical(d["state"], ["A","B","C","D"], ordered=True)
    d["repeat_c"] = d["repeat"] - d["repeat"].mean()

    # design + cluster ids
    y, X = dmatrices("y ~ C(state) * repeat_c", d, return_type="dataframe")
    groups = d.loc[y.index, "neuron_id"].to_numpy()

    # fit once (cluster-robust by neuron)
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})

    # --- MODEL PREDICTIONS + CIs ---
    reps = np.arange(0,10)
    grid = pd.DataFrame([(s, r, r - d["repeat"].mean())
                         for s in ["A","B","C","D"] for r in reps],
                        columns=["state","repeat","repeat_c"])
    Xg = np.asarray(build_design_matrices([X.design_info], grid)[0])
    beta = res.params.values
    cov  = res.cov_params().values
    yhat = Xg @ beta
    se   = np.sqrt(np.einsum("ij,jk,ik->i", Xg, cov, Xg))

    grid["y_hat"] = yhat
    grid["lo"]    = yhat - 1.96*se
    grid["hi"]    = yhat + 1.96*se

    # --- RAW: neuron-level means then mean±SEM across neurons ---
    # collapse to one value per neuron x state x repeat
    dn = (d.groupby(["neuron_id","state","repeat"])["y"]
            .mean().reset_index())
    raw_mean = dn.groupby(["state","repeat"])["y"].mean().reset_index(name="raw_mean")
    raw_sem  = dn.groupby(["state","repeat"])["y"].sem().reset_index(name="raw_sem")
    raw = pd.merge(raw_mean, raw_sem, on=["state","repeat"], how="left")

    # --- PLOT ---
    plt.figure(figsize=(7,4))
    for s in ["A","B","C","D"]:
        g = grid[grid["state"]==s]
        r = raw [raw ["state"]==s]
        # model line + 95% CI
        plt.plot(g["repeat"], g["y_hat"], marker="o", label=f"{s} (model)")
        plt.fill_between(g["repeat"], g["lo"], g["hi"], alpha=0.15)
        # raw mean±SEM as ribbon only (or change to errorbars if you prefer)
        plt.fill_between(r["repeat"], r["raw_mean"]-r["raw_sem"], r["raw_mean"]+r["raw_sem"], alpha=0.30)
    plt.axhline(0, ls="--", lw=1)
    plt.title(f"{roi_name}: model vs raw")
    plt.xlabel("Repeat"); plt.ylabel("Response")
    plt.legend(title="State"); plt.tight_layout(); plt.show()   

    return res, grid, raw




# use per ROI
for roi, d_roi in df.groupby("roi"):
    plot_wta_proportions(d_roi, roi)
    res, grid, raw = plot_fit_vs_raw_for_roi(d_roi, roi)
    res_e, grid_e = plot_roi_state_repeat_estimate(d_roi, roi)
    

import pdb; pdb.set_trace()  
for roi, dn in df_ph.groupby("roi"):
    raw_mean = dn.groupby(["phase","repeat"])["y"].mean().reset_index(name="raw_mean")
    raw_sem  = dn.groupby(["phase","repeat"])["y"].sem().reset_index(name="raw_sem")
    raw = pd.merge(raw_mean, raw_sem, on=["phase","repeat"], how="left")
    for state in ['A', 'B', 'C', 'D']:
        plt.figure()
        for p in [f"{state}_early", f"{state}_mid", f"{state}_rew"]:
            r = raw [raw ["phase"]==p]
            plt.fill_between(r["repeat"], r["raw_mean"]-r["raw_sem"], r["raw_mean"]+r["raw_sem"], alpha=0.30)
            plt.plot(r["repeat"], r["raw_mean"], marker="o", label=p);
        plt.legend(title="Phase")
        plt.axhline(0, ls="--", lw=1)
        plt.title(f"Mean firing rate, z-scored per task, for phases of state {state} + repeats \n for {roi}")
        plt.xlabel("Repeat"); plt.ylabel("Response");
        plt.legend(title="Phase"); plt.tight_layout(); plt.show()    


    raw_mean_rep = dn.groupby(["repeat"])["y"].mean().reset_index(name="raw_mean")
    raw_sem_rep  = dn.groupby(["repeat"])["y"].sem().reset_index(name="raw_sem")
    raw_rep = pd.merge(raw_mean_rep, raw_sem_rep, on=["repeat"], how="left")
    plt.figure(); 
    plt.plot(raw_rep["repeat"], raw_rep["raw_mean"], marker="o"); 
    plt.fill_between(raw_rep["repeat"], raw_rep["raw_mean"]-raw_rep["raw_sem"], raw_rep["raw_mean"]+raw_rep["raw_sem"], alpha=0.30)
    plt.axhline(0, ls="--", lw=1)
    plt.title(f"Mean firing, z-scored per task, collapsed across repeats \n for {roi}")
    
    
    

