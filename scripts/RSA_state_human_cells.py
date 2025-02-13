#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:34:42 2025


This is just a state RSA.


@author: Svenja Küchenhoff

##Location npy files:
These contain the location of the animal in each bin (should correspond exactly to the neuron bins)
locations1-9 are the 9 nodes

##Neuron_raw arrays are matrices of shape neurons X bins
each bin is the firing rate in a 50 ms timewindow
Location_raw arrays are arrays of length equal to the number of bins for the Neuron_raw matrix (may be 1 off)

"""

import numpy as np
import mc
import matplotlib.pyplot as plt
import os
import pickle

# first, load all csv files as numpys
# exclude 27 and 44 for now

### SETTINGS
data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
group_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group"

subjects = [f"{i:02}" for i in range(1, 55) if i not in [6, 9, 27, 44]]


save_results = True
repeats = [1,10] # define a range between [0,10]

only_reward_times = False
no_bins_per_state = 5

if only_reward_times == True:
    file_name_all_subj_prep = f"state_and_neuron_data_firstrep{repeats[0]}_lastrep{repeats[1]}_{no_bins_per_state}bins_only_rew"
if only_reward_times == False:
    file_name_all_subj_prep = f"state_and_neuron_data_firstrep{repeats[0]}_lastrep{repeats[1]}_{no_bins_per_state}bins"

if not os.path.isdir(group_folder):
    os.mkdir(group_folder)

# Steps: 
    # 1. Simulate neural timecourses based on behaviour
    # 2. Run regression on simulated and real neurons to put into RDMs
    # 3. Predict data RDMs with model RDM

if os.path.isfile(os.path.join(group_folder,file_name_all_subj_prep)):
    with open(os.path.join(group_folder,file_name_all_subj_prep), 'rb') as f:
        prep_data = pickle.load(f)
        print(f"opened stored dataset")
else:  
    print(f"loading and running preprocessing of human cells")
    data = mc.analyse.helpers_human_cells.load_cell_data(data_folder, subjects)
    prep_data = mc.analyse.helpers_human_cells.prep_neurons_and_state(data, repeats = repeats, only_reward_times =only_reward_times, no_bins_per_state =no_bins_per_state, sim_fake_data = True)   
    if save_results == True:
        # save the all_modelled_data dict such that I don't need to always run it again.
        with open(os.path.join(group_folder,file_name_all_subj_prep), 'wb') as f:
            pickle.dump(prep_data, f)
        print(f"saved the modelled data as {group_folder}/{file_name_all_subj_prep}")
     
    
results_RSA = mc.analyse.helpers_human_cells.run_state_RSA(prep_data, per_ROI = True, plotting = True, only_reward_times =only_reward_times, no_bins_per_state =no_bins_per_state, sim_fake_data = True)


# some plotting functions.
# collapse all neurons across locations: per neuron, plot firing rate for locations [now] 1-9;
# location [next] 1-9, location [next reward], location [current reward]




