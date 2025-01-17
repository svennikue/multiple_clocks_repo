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

# first, load all csv files as numpys
# exclude 27 and 44 for now

regression_version = '03' #for every tasks, only the rewards are modelled [using a stick function]
RDM_version = '03-1' # modelling only reward rings + split ‘clocks model’ = just rotating the reward location around. 
 
models_I_want = mc.analyse.analyse_MRI_behav.select_models_I_want(RDM_version)


subjects = [28,31,32,33,34,35,36,37,38,40,43,45,46,49,50]
data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
data = mc.analyse.helpers_human_cells.load_cell_data(data_folder, subjects)

# Steps: 
    # 1. Simulate neural timecourses based on behaviour
    # 2. Run regression on simulated and real neurons to put into RDMs
    # 3. Predict data RDMs with model RDM
    
# Questions:
    # - How can I make sure to combine all cells of a certain region into 
    #   "ROI RDMs" across subjects?
    #   for this, I need to somehow normalise the behaviour; i.e. how long
    #   subjects spent at each location, where they go etc. Does this work??
    #   might work for the reward locations, but not for the paths. Start with that
    #   This would mean:
    #       1. Per subject, create neural timecourses
    #       2. Run regression such that you have e.g. 4 reward-bins and equal no of repeats
    #       3. then concatenate all cells in the same ROI across all subjects 
    #       4. then compute the RSA for the ROI data RDMs
    


#
#
# CONTINUE HERE 

# THIS IS ALL COPIED FROM ANALSYIS MOSUE DATA
    
# # all_results[mouse] = mc.analyse.analyse_ephys.reg_across_tasks(mouse_data_clean[mouse]["rewards_configs"], mouse_data_clean[mouse]["locations"], mouse_data_clean[mouse]["neurons"], mouse_data_clean[mouse]["timings"], mouse_data[mouse]["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within = True, split_by_phase = True)
# trajectory, timings_curr_run, index_make_step, step_number, curr_neurons = mc.analyse.analyse_ephys.prep_ephys_per_trial(timings_all, locations_all, run_no, task_no, task_config, neurons)
            
data_prep = mc.analyse.helpers_human_cells.prep_and_model_human_cells(data)            
# all_modelled_data = mc.simulation.predictions.set_continous_models_ephys(data_prep, no_phase_neurons= number_phase_neurons, plot = False)

all_modelled_data = mc.simulation.predictions.prep_and_model_human_cells(data)

import pdb; pdb.set_trace()




# # THIS IS ALL COPIED FROM RSA FMRI
# for subject in data:
#     # Step 1: create model RDMs.
#     # focus for now: 'split clock model' 
#     # note: save them and load old ones if existing.
    

#     # Step 2: loading and computing the data RDMs
#     data_RDM_file_2d= mc.analyse.analyse_MRI_behav.read_in_RDM_conds(regression_version, RDM_version, data_dir, RDM_dir, no_RDM_conditions, sort_as = 'dict-two-halves')
#     condition_names = mc.analyse.analyse_MRI_behav.get_conditions_list(RDM_dir)
    
#     # Step 3: load and compute the model RDMs.
#     # 3-1 load the data files I created.
#     data_dirs = {}
#     for model in models_I_want:
#         data_dirs[model]= np.load(os.path.join(RDM_dir, f"data{model}_{sub}_fmri_both_halves.npy")) 
    
#     # step 3-2: create model RDMs
#     # first, each model gets its own, separate estimation.
#     model_RDM_dir = {}
#     RDM_my_model_dir = {}
#     for model in data_dirs:
#         print(model)
#         model_data = mc.analyse.analyse_MRI_behav.prepare_model_data(data_dirs[model], no_RDM_conditions, RDM_version)
#         model_RDM_dir[model] = rsr.calc_rdm(model_data, method='crosscorr', descriptor='conds', cv_descriptor='sessions')
#             import pdb; pdb.set_trace()
#             # I changed it now such that instead of only a quarter of the matrix, the entire thing is saved.
#             # now in this step I need to cut it correctly! BUT i do want to include the diagonal.

#         fig, ax, ret_vla = rsatoolbox.vis.show_rdm(model_RDM_dir[model])
#         # set up the model object
#         model_model = rsatoolbox.model.ModelFixed(f"{model}_only", model_RDM_dir[model])


#     # ACTUAL RSA - single models
#     # STEP 4: evaluate the model fit between model and data RDMs.
#     # for d in data_RDM:
#     #     RDM_my_model_dir[model] = mc.analyse.analyse_MRI_behav.evaluate_model(model_model, d)
    
    
#     for d in data_RDM:
#         RDM_my_model_dir[model] = mc.analyse.analyse_MRI_behav.evaluate_model(model_model, d)
#     RDM_my_model_dir[model] = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(model_model, d) for d in tqdm(data_RDM, desc=f"running GLM for all searchlights in {model}"))
#     mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=RDM_my_model_dir[model], data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model}", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)

         
#     # and the multiple regressors one     
#     multiple_regressors = ['action-box','buttonsXphase', 'buttons', 'location', 'phase', 'state']
#     results_combo_model= mc.analyse.analyse_MRI_behav.multiple_RDMs_RSA(multiple_regressors, model_RDM_dir, data_RDM)
#     model_name = 'combo-act-buph-bu-loc-ph-st'
#     for i, model in enumerate(multiple_regressors):
#         mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_combo_model, data_RDM_file=data_RDM, file_path = results_dir, file_name= f"{model.upper()}-{model_name}", mask=mask, number_regr = i, ref_image_for_affine_path=ref_img)
