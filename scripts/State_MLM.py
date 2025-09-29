#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:28:15 2025

@author: Svenja Küchenhoff


run a linear mixed effects model to test for the preferred states vs. the
effect of repeats per region.

"""

import os
import mc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as st
from joblib import Parallel, delayed



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


#
#
# chat is proposing this:

# fr: numpy array with shape (N_neurons, N_repeats, 360)
# neuron_ids, repeat_ids are optional labels (otherwise 0..N-1 etc.)

def make_long_state_means(fr, neuron_ids=None, repeat_ids=None):
    N, R, B = fr.shape
    assert B == 360, "Expecting 360 bins."
    if neuron_ids is None: neuron_ids = np.arange(N)
    if repeat_ids is None: repeat_ids = np.arange(R)

    # Masks for states
    A_mask = np.arange(360) < 90
    B_mask = (np.arange(360) >= 90) & (np.arange(360) < 180)
    C_mask = (np.arange(360) >= 180) & (np.arange(360) < 270)
    D_mask = np.arange(360) >= 270

    # Mean over bins within each state (axis=2)
    mA = fr[:, :, A_mask].mean(axis=2)
    mB = fr[:, :, B_mask].mean(axis=2)
    mC = fr[:, :, C_mask].mean(axis=2)
    mD = fr[:, :, D_mask].mean(axis=2)

    # Stack into long format: one row per neuron × repeat × state
    rows = []
    for i, nid in enumerate(neuron_ids):
        for r, rid in enumerate(repeat_ids):
            rows.append((nid, rid, 'A', mA[i, r]))
            rows.append((nid, rid, 'B', mB[i, r]))
            rows.append((nid, rid, 'C', mC[i, r]))
            rows.append((nid, rid, 'D', mD[i, r]))

    df = pd.DataFrame(rows, columns=['neuron', 'repeat', 'state', 'fr'])
    # Early vs late: split repeats in half (customize if you need a different rule)
    half = int(np.floor(fr.shape[1] / 2))
    df['epoch'] = np.where(df['repeat'] < half, 'early', 'late')

    # Categorical ordering
    df['state'] = pd.Categorical(df['state'], categories=['A','B','C','D'], ordered=True)
    df['epoch'] = pd.Categorical(df['epoch'], categories=['early','late'], ordered=True)
    return df


import statsmodels.formula.api as smf
import numpy as np



#
#
#

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


    perms = 1
 
    # for each cell, cross-validate the peak task-lag shift for spatial consistency.
    for neuron_idx, curr_neuron in enumerate(data[f"sub-{sesh:02}"]['normalised_neurons']):
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
        
        
        # collect one big df with all neurons.
    
    
# make the big dataset
# df_long = make_long_state_means(fr, neuron_ids, repeat_ids)


# then test for interaction and main effect
# do this per ROI!!

# Mixed model: random intercept per neuron (you can add random slopes if needed)
m = smf.mixedlm("fr ~ C(state) * C(epoch)", data=df_long,
                groups=df_long["neuron"])  # re_formula="~C(state)" is possible but optional
res = m.fit(method="lbfgs")
print(res.summary())
# Collect all parameter names that are interaction terms
pnames = res.params.index
ix = [i for i, name in enumerate(pnames) if "C(state)" in name and "C(epoch)" in name]
L = np.zeros((len(ix), len(pnames)))
for r, i in enumerate(ix):
    L[r, i] = 1.0
print(res.f_test(L))  # F-stat (Wald) for the interaction




 
    