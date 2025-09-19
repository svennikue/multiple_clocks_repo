#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 16:18:47 2025

@author: Svenja Küchenhoff
Regressing out the effect of no of repeats
Then storing the data as a cleaned file
"""

import os
import mc
import numpy as np
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as st
import statsmodels.api as sm


def get_data(sub):
    data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
    if not os.path.isdir(data_folder):
        print("running on ceph")
        data_folder = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives"
    data_norm = mc.analyse.helpers_human_cells.load_norm_data(data_folder, [f"{sub:02}"])
    return data_norm, data_folder

def regress_out_repeat(neuron_data, repeats):
    n_trials, n_time = neuron_data.shape
    
    # Flatten data: (n_trials*n_time,)
    y = neuron_data.reshape(-1)

    # Repeat labels: expand so each timepoint has the trial's repeat index
    X_repeat = np.repeat(repeats, n_time)
    X = sm.add_constant(X_repeat)

    model = sm.OLS(y, X).fit()
    resid = model.resid

    # Reshape back
    cleaned = resid.reshape(n_trials, n_time)

    effect_strength = model.rsquared
    beta = model.params[1]

    return cleaned, effect_strength, beta

    

def residualise_by_repeat_no(sessions, save_all = False):
    for sesh in sessions:
        # load data
        data, source_dir = get_data(sesh)
        out_dir = f"source_dir/{s{sesh:02}}/cells_and_beh/cleaned_from_reps"
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
            
        if not data:
            print(f"no raw data found for {sesh}, so skipping")
            continue
        beh_df = data[f"sub-{sesh:02}"]['beh'].copy()
        
        for neuron_idx, curr_neuron in enumerate(data[f"sub-{sesh:02}"]['normalised_neurons']):
            raw_neural_data = data[f"sub-{sesh:02}"]['normalised_neurons'][curr_neuron].to_numpy()
            repeats = beh_df['rep_correct'].to_numpy()
            cleaned, effect_strength, betas = regress_out_repeat(raw_neural_data, repeats)
            print(f"effect strengt for {curr_neuron} is {effect_strength}")
            if save_all == True:
                # CONTINUE HERE!
                # this needs to eb in the same format as the one I loaded in the beginniong
                # also collect a results table as for state such that I can plot the effect by cell.
                import pdb; pdb.set_trace()

    

    
if __name__ == "__main__":
    # trials can be 'all', 'all_correct', 'early', 'late', 'all_minus_explore'
    residualise_by_repeat_no(sessions=list(range(0,64)), save_all=True)
    residualise_by_repeat_no(sessions=[4], save_all=True)
        