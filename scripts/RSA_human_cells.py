#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 14:25:03 2024

this script runs the musicbox RSA on human cells, treating all subjects as the one.

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

subjects = [28,31,32,33,34,35,36,37,38,40,43,45,46,49,50,51,53,55,56,57,58]

# subjects = [f"{i:02}" for i in range(1, 51) if i not in [6, 9, 27, 44]]

certain_model = ['stat_model']
certain_model = False
save_results = True
models = ['simple_models'] # ['all_models']
###

regression_version = '03' #for every tasks, only the rewards are modelled [using a stick function]
RDM_version = '03-1' # modelling only reward rings + split ‘clocks model’ = just rotating the reward location around
models_I_want = mc.analyse.analyse_MRI_behav.select_models_I_want(RDM_version)  
repeats = [1, 10] # first and last repeat
file_name_all_modelled = f"all_modelled_data_{models}_dict_firstrep{repeats[0]}_lastrep{repeats[1]}"

    
if not os.path.isdir(group_folder):
    os.mkdir(group_folder)

# Steps: 
    # 1. Simulate neural timecourses based on behaviour
    # 2. Run regression on simulated and real neurons to put into RDMs
    # 3. Predict data RDMs with model RDM

if os.path.isfile(os.path.join(group_folder,file_name_all_modelled)):
    with open(os.path.join(group_folder,file_name_all_modelled), 'rb') as f:
        all_modelled_data = pickle.load(f)
        print("opened stored dataset")
else:  
    print("loading and running preprocessing of human cells")
    data = mc.analyse.helpers_human_cells.load_cell_data(data_folder, subjects)
    if 'simple_models' in models:
        all_modelled_data = mc.analyse.helpers_human_cells.prep_and_model_human_cells(data, repeats = repeats, model_simple = True)   
    else:
        all_modelled_data = mc.analyse.helpers_human_cells.prep_and_model_human_cells(data, repeats = repeats, model_simple = False)   
    if save_results == True:
        # save the all_modelled_data dict such that I don't need to always run it again.
        with open(os.path.join(group_folder,file_name_all_modelled), 'wb') as f:
            pickle.dump(all_modelled_data, f)
        print(f"saved the modelled data as {group_folder}/{file_name_all_modelled}")
     
    
# all_modelled_data = mc.simulation.predictions.set_continous_models_ephys(data_prep, no_phase_neurons= number_phase_neurons, plot = False)

# choose some settings
# e.g. which models in RSA
# and which cells to clump together
if 'simple_models' in models:
    results_RSA = mc.analyse.helpers_human_cells.run_RSA(all_modelled_data, only_specific_model = certain_model, 
                                                         per_ROI = True, plotting = True, simple_models = True, 
                                                         dont_avg_rep_tasks= True)

    results_RSA = mc.analyse.helpers_human_cells.run_RSA(all_modelled_data, only_specific_model = certain_model, 
                                                         per_ROI = True, plotting = True, simple_models = True)
else:
    results_RSA = mc.analyse.helpers_human_cells.run_RSA(all_modelled_data, only_specific_model = certain_model, per_ROI = True, plotting = True, simple_models = False)

# results_RSA = mc.analyse.helpers_human_cells.run_RSA(all_modelled_data, only_specific_model = False, per_ROI = True, plotting = True)

import pdb; pdb.set_trace()
