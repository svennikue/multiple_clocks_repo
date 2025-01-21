#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:55:36 2023

@author: Svenja Küchenhoff
this script runs the entire analysis of Mohamadys data
"""

import numpy as np
import mc
import matplotlib.pyplot as plt
import sys
#import joypy
#from matplotlib import cm
#import plotly.figure_factory as ff
#import plotly.io as pio
#pio.renderers.default = "browser"
import scipy.stats 
import os
import pickle
import colormaps as cmaps

#
save = False
load_old = False
do_per_run = False
do_neuron_subset = False
date = np.datetime64('today')
out_path = f"/Users/xpsy1114/Documents/projects/multiple_clocks/output/{date}"

    
# Part 1: load data
# mouse_a, mouse_b, mouse_c, mouse_d, mouse_e, mouse_f, mouse_g, mouse_h = mc.analyse.analyse_ephys.load_ephys_data(Data_folder = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/')
mice_recdays = ['mouse_a', 'mouse_b', 'mouse_c', 'mouse_d', 'mouse_e', 'mouse_f', 'mouse_g', 'mouse_h']
mouse_data = mc.analyse.analyse_ephys.load_ephys_data(mice_recdays, Data_folder = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/')

print("all mouse data is loaded")

# defining contrasts.
# if this is supposed to be with state, then it has to be one longer:
contrast_matrix = ((1,0,0,0,0), (0,1,0,0,0), (0,0,1,0,0), (0,0,0,1,0), (0,0,0,0,1), (1,-1,0,0,0), (1, 0,-1,0,0), (1,0,0,-1,0), (1,0,0,0,-1), (0,1,-1,0,0), (0,1,0,-1,0), (0,0,1,-1,0), (0,0,1,0,-1))


#contrast_matrix = ((1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1), (1,-1,0,0), (1, 0,-1,0), (1,0,0,-1), (0,1,-1,0), (0,1,0,-1), (0,0,1,-1))

# temporarily exclude locaiton
# contrast_split_by_phase = ((1,0,0), (0,1,0), (0,0,1), (1,-1,0), (1, 0,-1), (0,1,-1))
contrast_split_by_phase = ((1,0),(0,1), (1,-1))

if save: 
    # to save the data later
    res_path = "/Users/xpsy1114/Documents/projects/multiple_clocks/output/2023-06-26"
    if os.path.isdir(out_path) == False:
        os.mkdir(out_path)



if load_old: 
    # if I want to load old stuff, use this:
        # USE THIS TOMORROW
    # eval(f"{mice_recdays[0]}['recday']")
    clock_results = {}
    midnight_results = {}
    all_results = {}
    perrun_results = {}
    res_path = "/Users/xpsy1114/Documents/projects/multiple_clocks/output/2023-06-28_with_loc"
    for i, mouse_index in enumerate(mice_recdays): 
        mouse_recday = eval(f"{mouse_index}['recday']")
        # clock_file_name = f"{mouse_recday}_clock_dic"
        midn_file_name = f"{mouse_recday}_midn_dic"
        perrun_file_name = f"{mouse_recday}_res_perrun"
        results_file_name = f"{mouse_recday}_res_dic"
        #with open(os.path.join(res_path,clock_file_name), 'rb') as f:
        #    clock_results[clock_file_name] = pickle.load(f)
        with open(os.path.join(res_path,midn_file_name), 'rb') as f:
            midnight_results[midn_file_name] = pickle.load(f)
        with open(os.path.join(res_path,perrun_file_name), 'rb') as f:
            perrun_results[midn_file_name] = pickle.load(f)
        with open(os.path.join(res_path,results_file_name), 'rb') as f:
            all_results[results_file_name] = pickle.load(f)
    sys.exit(0)
                


# my goal is to show that my model can predict Mohamadys data.
# 13.06.2023
# first, figure out which model is good. 
# compare continous vs normal
# compare different amount of cells
# compare different binnings.


# 17.01.2025 I deleted quite a big chunk here. look at git if I want to recover.
mouse_data_clean, regs_per_run, all_results, midn_result_dict = {}, {}, {}, {}

for mouse in mice_recdays:
    mouse_data_clean[mouse],  regs_per_run[mouse], all_results[mouse], midn_result_dict[mouse] = {}, {}, {}, {}
        
        
for mouse in sorted(mouse_data):
    print("Now starting to run simulations and RSAs.")
    mouse_data_clean[mouse]["rewards_configs"], mouse_data_clean[mouse]["locations"], mouse_data_clean[mouse]["neurons"], mouse_data_clean[mouse]["timings"]  = mc.analyse.analyse_ephys.clean_ephys_data(mouse_data[mouse]["rewards_configs"], mouse_data[mouse]["locations"], mouse_data[mouse]["neurons"], mouse_data[mouse]["timings"], mouse_data[mouse]["recday"])
    if do_per_run == True:
        # cleaned datasat
        regs_per_run[mouse] = mc.analyse.analyse_ephys.reg_between_tasks_singleruns(mouse_data_clean[mouse]["rewards_configs"], mouse_data_clean[mouse]["locations"], mouse_data_clean[mouse]["neurons"], mouse_data_clean[mouse]["timings"], contrast_matrix, mouse_data[mouse]["recday"], contrast_split= contrast_split_by_phase ,continuous = True, no_bins_per_state = 10, split_by_phase = 1, number_phase_neurons = 3, mask_within = True,plotting=False)                                         
    else:    
        # cleaned datasat
        all_results[mouse] = mc.analyse.analyse_ephys.reg_across_tasks(mouse_data_clean[mouse]["rewards_configs"], mouse_data_clean[mouse]["locations"], mouse_data_clean[mouse]["neurons"], mouse_data_clean[mouse]["timings"], mouse_data[mouse]["recday"], plotting = False, continuous = True, no_bins_per_state = 10, number_phase_neurons = 3, mask_within = True, split_by_phase = False)

    if do_neuron_subset == True:
        # define a mask for clock and midnight neurons, respectively
        midnight_neurons = []
        midnight_mask = np.where(mouse_data[mouse]["neuron_type"][:,0] == 1)[0]
        for neurons in mouse_data_clean[mouse]["neurons"]:
            midnight_neurons.append(neurons[midnight_mask, :])
        midn_result_dict[mouse] = mc.analyse.analyse_ephys.reg_between_tasks_singleruns(mouse_data_clean[mouse]["rewards_configs"], mouse_data_clean[mouse]["locations"], midnight_neurons, mouse_data_clean[mouse]["timings"], contrast_matrix, mouse_data[mouse]["recday"], contrast_split= contrast_split_by_phase ,continuous = True, no_bins_per_state = 10, split_by_phase = 1, number_phase_neurons = 3, mask_within = True)
               
if save: 
    if do_per_run == True:
        with open(os.path.join(out_path,f"mouse_data_clean_regs_perrun"), 'wb') as f:
            pickle.dump(regs_per_run, f)
    else:     
        with open(os.path.join(out_path,f"mouse_clean_res_dic"), 'wb') as f:
            pickle.dump(all_results, f)
    if do_neuron_subset == True:
        with open(os.path.join(out_path,f"mouse_clean_midn_dic"), 'wb') as f:
            pickle.dump(midn_result_dict, f)
               
# PLOTTING what I have on my posters in january 2025
#  4 regressors
data_normal_clocks_clocks, data_normal_clocks_midn, data_normal_clocks_phas, data_normal_clocks_loc = [], [], [], []
data_normal_midn_clocks, data_normal_midn_midn, data_normal_midn_phas, data_normal_midn_loc = [], [], [], []
data_normal_all_clocks_t, data_normal_all_midn_t, data_normal_all_phas_t, data_normal_all_loc_t, data_normal_all_state_t= [], [], [], [], []
# for the t-test
data_normal_all_clocks_b, data_normal_all_midn_b, data_normal_all_phas_b, data_normal_all_loc_b, data_normal_all_state_b= [], [], [], [], []

if load_old == True:   
    for mouse_res in midnight_results:
        data_normal_midn_clocks.append(midnight_results[mouse_res]['normal']['t_vals'][1])
        data_normal_midn_midn.append(midnight_results[mouse_res]['normal']['t_vals'][2])
        data_normal_midn_phas.append(midnight_results[mouse_res]['normal']['t_vals'][3])
        data_normal_midn_loc.append(midnight_results[mouse_res]['normal']['t_vals'][4])

label_string_list_check = []
data_t = {}
data_b = {}

# to make sure to not mess up the order, rather loop through the label lists.
for i, model in enumerate(all_results['mouse_a']['normal']['label_regs']):
    label_string_list_check.append(model)
    data_b[model] = []
    data_t[model] = []
    for mouse_res in all_results:
        data_b[model].append(all_results[mouse_res]['normal']['coefs'][i])
        data_t[model].append(all_results[mouse_res]['normal']['t_vals'][i+1])
        
    # # depending if I want to include more regressors, you have to add more! 
    # data_normal_all_clocks_t.append(all_results[mouse_res]['normal']['t_vals'][1])
    # data_normal_all_phas_t.append(all_results[mouse_res]['normal']['t_vals'][2])
    # data_normal_all_loc_t.append(all_results[mouse_res]['normal']['t_vals'][3])
    # data_normal_all_state_t.append(all_results[mouse_res]['normal']['t_vals'][4])
    
    # data_normal_all_clocks_b.append(all_results[mouse_res]['normal']['coefs'][0])
    # data_normal_all_phas_b.append(all_results[mouse_res]['normal']['coefs'][1])
    # data_normal_all_loc_b.append(all_results[mouse_res]['normal']['coefs'][2])
    # data_normal_all_state_b.append(all_results[mouse_res]['normal']['coefs'][3])

# betas
# betas_across_subjects = [data_normal_all_clocks_b, data_normal_all_phas_b, data_normal_all_loc_b, data_normal_all_state_b]

data_to_plot = []
for i, model in enumerate(data_b):
    data_to_plot.append(data_b[model])
    print(f"{i} next will be {model}")

# t-vals
#data_to_plot = [data_normal_all_clocks_t, data_normal_all_phas_t, data_normal_all_loc_t, data_normal_all_state_t]

label_string_list_plot = ['SMB-Model', 'Location', "Subgoal-\n Progress", 'State']
print(f"make sure that these two are in same order: <{label_string_list_plot}> and <{label_string_list_check}>")

label_tick_list_plot = [1,2,3,4]
label_tick_list_plot = [0.5, 1.5, 2.5, 3.5]

title_string_plot = 'betas per complete mouse dataset, regression with 5 models, across tasks, averaged over runs'
mc.analyse.analyse_ephys.plotting_hist_scat(data_to_plot, label_string_list_plot, label_tick_list_plot, title_string_plot, save_fig=out_path) 





# 17.01.2025 THIS IS WHERE I STOPPED!
# I restructred the data into dicitonaries instead of singular mice.

# then also just plot the betas per mouse across repeats.
# so one dot is one beta where all different tasks are concatinated, but every run is treated separately.
data_only_clocks_totest, data_only_clocks_toplot = [], []
for mouse in regs_per_run:
    data_only_clocks_totest.append(regs_per_run[mouse]['coeffs_only_clock'])
    data_only_clocks_toplot.append(regs_per_run[mouse]['t-vals_only_clock'][:,1])

for i, mouse in enumerate(data_only_clocks_totest):
    t_statistic, p_value = ttest_1samp(mouse, 0, alternative='greater')
    # Output the results
    print(f'T-statistic: {t_statistic} for mouse no {i}')
    print(f'P-value: {p_value} for mouse no {i}')

label_string_mice = ['mouse a', 'mouse b', 'mouse c', 'mouse d', 'mouse e', 'mouse f', 'mouse g', 'mouse h']
label_tick_mice = [1,2,3,4,5,6,7,8]
title_string_plot='tvals for the clocks model for single runs across tasks separately for each mouse dataset'
mc.analyse.analyse_ephys.plotting_hist_scat(data_only_clocks_toplot, label_string_mice, label_tick_mice, title_string_plot, save_fig=out_path) 


################

###############

#############

############# 22.11.2023 this is where I stopped!






# data_comp = [data_normal_all_clocks, data_normal_all_midn, data_normal_all_phas, data_normal_all_loc, data_normal_clocks_clocks, data_normal_clocks_midn, data_normal_clocks_phas, data_normal_clocks_loc, data_normal_midn_clocks, data_normal_midn_midn, data_normal_midn_phas, data_normal_midn_loc]
# label_string_list_comp = ['all_clocks', 'all_midn', 'all_phas', 'all_loc', 'cl_clocks', 'cl_midn', 'cl_phas', 'cl_loc', 'mid_clocks', 'mid_midn', 'mid_phas', 'mid_loc']
#label_tick_list_comp = [1,2,3,4,5,6,7,8]
#title_string_comp = 'tvals. all neurons vs midnight. cont models, 10 bins, 3 phase neurons, mask within tasks'

#mc.analyse.analyse_ephys.plotting_hist_scat(data_comp, label_string_list_comp, label_tick_list_comp, title_string_comp) 





# plot contrast clocks - midnight
data_all_clphas = []
data_early_all_clphas = []
data_mid_all_clphas = []
data_late_all_clphas = []

data_clocks_clphas = []
data_early_clocks_clphas = []
data_mid_clocks_clphas = []
data_late_clocks_clphas = []

for mouse_res in clock_results:
    data_clocks_clphas.append(clock_results[mouse_res]['normal']['coefs'][0] - clock_results[mouse_res]['normal']['coefs'][1])
    data_early_clocks_clphas.append(clock_results[mouse_res]['early']['coefs'][0] - clock_results[mouse_res]['early']['coefs'][1])
    data_mid_clocks_clphas.append(clock_results[mouse_res]['mid']['coefs'][0] - clock_results[mouse_res]['mid']['coefs'][1])
    data_late_clocks_clphas.append(clock_results[mouse_res]['late']['coefs'][0] - clock_results[mouse_res]['late']['coefs'][1])

for mouse_res in all_results:
    data_all_clphas.append(all_results[mouse_res]['normal']['coefs'][0] - all_results[mouse_res]['normal']['coefs'][1])
    data_early_all_clphas.append(all_results[mouse_res]['early']['coefs'][0] - all_results[mouse_res]['early']['coefs'][1])
    data_mid_all_clphas.append(all_results[mouse_res]['mid']['coefs'][0] - all_results[mouse_res]['mid']['coefs'][1])
    data_late_all_clphas.append(all_results[mouse_res]['late']['coefs'][0] - all_results[mouse_res]['late']['coefs'][1])


data_contr = [data_all_clphas, data_early_all_clphas, data_mid_all_clphas, data_late_all_clphas, data_clocks_clphas, data_early_clocks_clphas, data_mid_clocks_clphas, data_late_clocks_clphas]
label_string_contr = ['all cl-mid', 'early all cl-mid', 'mid_all cl-mid', 'late_all cl-mid', 'clocks cl-mid', 'early_clocks cl-mid', 'mid_clocks cl-mid', 'late_clockscl-mid']
label_tick_list_contr = [1,2,3,4,5,6,7,8]
title_string_contr = 'beta contrasts. Continuous models, 10 bins, 3 phase neurons, mask within tasks'

mc.analyse.analyse_ephys.plotting_hist_scat(data_contr, label_string_contr, label_tick_list_contr, title_string_contr) 




# split by phase
data_early_clocks_clocks = []
data_early_clocks_midn = []
data_mid_clocks_clocks = []
data_mid_clocks_midn = []
data_late_clocks_clocks = []
data_late_clocks_midn = []

data_early_midn_clocks = []
data_early_midn_midn = []
data_mid_midn_clocks = []
data_mid_midn_midn = []
data_late_midn_clocks = []
data_late_midn_midn = []

data_early_all_clocks = []
data_early_all_midn = []
data_mid_all_clocks = []
data_mid_all_midn = []
data_late_all_clocks = []
data_late_all_midn = []

for mouse_res in clock_results:
    data_early_clocks_clocks.append(clock_results[mouse_res]['early']['t_vals'][1])
    data_early_clocks_midn.append(clock_results[mouse_res]['early']['t_vals'][2])
    data_mid_clocks_clocks.append(clock_results[mouse_res]['mid']['t_vals'][1])
    data_mid_clocks_midn.append(clock_results[mouse_res]['mid']['t_vals'][2])
    data_late_clocks_clocks.append(clock_results[mouse_res]['late']['t_vals'][1])
    data_late_clocks_midn.append(clock_results[mouse_res]['late']['t_vals'][2])
    
for mouse_res in midnight_results:
    data_early_midn_clocks.append(midnight_results[mouse_res]['early']['t_vals'][1])
    data_early_midn_midn.append(midnight_results[mouse_res]['early']['t_vals'][2])
    data_mid_midn_clocks.append(midnight_results[mouse_res]['mid']['t_vals'][1])
    data_mid_midn_midn.append(midnight_results[mouse_res]['mid']['t_vals'][2])
    data_late_midn_clocks.append(midnight_results[mouse_res]['late']['t_vals'][1])
    data_late_midn_midn.append(midnight_results[mouse_res]['late']['t_vals'][2])

for mouse_res in all_results:    
    data_early_all_clocks.append(all_results[mouse_res]['early']['t_vals'][1])
    data_early_all_midn.append(all_results[mouse_res]['early']['t_vals'][2])
    data_mid_all_clocks.append(all_results[mouse_res]['mid']['t_vals'][1])
    data_mid_all_midn.append(all_results[mouse_res]['mid']['t_vals'][2])
    data_late_all_clocks.append(all_results[mouse_res]['late']['t_vals'][1])
    data_late_all_midn.append(all_results[mouse_res]['late']['t_vals'][2])

data_split = [data_early_clocks_clocks, data_early_clocks_midn, data_mid_clocks_clocks, data_mid_clocks_midn, data_late_clocks_clocks , data_late_clocks_midn, data_early_midn_clocks, data_early_midn_midn, data_mid_midn_clocks, data_mid_midn_midn, data_late_midn_clocks, data_late_midn_midn, data_early_all_clocks, data_early_all_midn, data_mid_all_clocks, data_mid_all_midn, data_late_all_clocks, data_late_all_midn]
label_string_list_split = ['early_clocks-all', 'early_midn-all', 'mid_clocks-all', 'mid_midn-all', 'late_clocks-all', 'late_midn-all', 'early_clocks-cl', 'early_midn-cl', 'mid_clocks-cl', 'mid_midn-cl', 'late_clocks-cl', 'late_midn-cl','early_clocks-mid', 'early_midn-mid', 'mid_clocks-mid', 'mid_midn-mid', 'late_clocks-mid', 'late_midn-mid']
label_tick_list_split = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
title_string_split = 'tvals. Continuous models, 10 bins, 3 phase neurons, mask within tasks, split by phase'


mc.analyse.analyse_ephys.plotting_hist_scat(data_split, label_string_list_split, label_tick_list_split, title_string_split) 




# now plot the results if I do everything separetly per run.
if do_per_run == True:
    # firstly, plot the t-values.
    # "The clock’s model can significantly and reliably across tasks and runs predict the neural data, even if phase, location and midnight is included as regressors
    # "This is still true if all auto-correlations (within-task) are removed
    # "I can do this for a single run across-tasks, or for all runs averaged and all tasks"
    phase_split = ['early', 'mid', 'late']
    letters = ['a','b','c','d','e','f', 'g', 'h']
    # do one where I plot the split contrast clocks vs. everything and just the clock t-vals
    if load_old == True:
        for mouse_result in enumerate(perrun_results):
            if i == 0:
                t_clocks = perrun_results[mouse_result]["t-values"][:,1]
                t_midn = perrun_results[mouse_result]["t-values"][:,2]
                t_phas = perrun_results[mouse_result]["t-values"][:,3]
                t_loc = perrun_results[mouse_result]["t-values"][:,4]
            elif i > 0:
                t_clocks = np.concatenate((t_clocks, perrun_results[mouse_result]["t-values"][:,1]))
                t_midn = np.concatenate((t_midn,perrun_results[mouse_result]["t-values"][:,2]))
                t_phas = np.concatenate((t_phas, perrun_results[mouse_result]["t-values"][:,3]))
                t_loc = np.concatenate((t_loc, perrun_results[mouse_result]["t-values"][:,4]))
        
    if load_old == False:
        for i, mouse in enumerate(letters):
            dataset = eval(f"{mouse}_reg_per_run")
            if i == 0:
                t_clocks = dataset["t-values"][:,1]
                t_midn = dataset["t-values"][:,2]
                t_phas = dataset["t-values"][:,3]
                t_loc = dataset["t-values"][:,4]
            elif i > 0:
                t_clocks = np.concatenate((t_clocks, dataset["t-values"][:,1]))
                t_midn = np.concatenate((t_midn,dataset["t-values"][:,2]))
                t_phas = np.concatenate((t_phas, dataset["t-values"][:,3]))
                t_loc = np.concatenate((t_loc, dataset["t-values"][:,4]))
            

    data_tval_perrun = [t_clocks, t_midn, t_phas, t_loc]
    label_tval_perrun = ["t_clocks", "t_midn", "t_phas", "t_loc"]
    label_tick_list_tperrun = [1,2,3,4]
    title_string_tperrun = 'tvals. Reg for LAST 6 RUNS of each task. Contin, 10 bins, 3 phase neurons, mask within tasks'

    mc.analyse.analyse_ephys.plotting_hist_scat(data_tval_perrun, label_tval_perrun, label_tick_list_tperrun, title_string_tperrun) 

   
    # REORDERED PER RUN
    # temporarily leave out b because its crap
    letters = ['a','c','d','e','f', 'g', 'h']
    #letters = ['a','b','c','d','e','f', 'g', 'h']
    for i, mouse in enumerate(letters):
        dataset = eval(f"{mouse}_reg_per_run")
        if i == 0:
            t_clocks = dataset["reord_t-vals"][:,1]
            t_midn = dataset["reord_t-vals"][:,2]
            t_loc = dataset["reord_t-vals"][:,3]

        elif i > 0:
            t_clocks = np.concatenate((t_clocks, dataset["reord_t-vals"][:,1]))
            t_midn = np.concatenate((t_midn,dataset["reord_t-vals"][:,2]))
            t_loc = np.concatenate((t_loc, dataset["reord_t-vals"][:,3]))
        

    data_reord_perrun = [t_clocks, t_midn, t_loc]
    label_reord_perrun = ["Musicbox", "Midnight", "Location"]
    label_tick_reord_tperrun = [1,2,3]
    title_string_tperrun_reord = 'tvals per run, reordered'

    mc.analyse.analyse_ephys.plotting_hist_scat(data_reord_perrun, label_reord_perrun, label_tick_reord_tperrun, title_string_tperrun_reord) 
    
    
    # MIDNIGTH, REORDRED PER RUN
    for i, mouse in enumerate(letters):
        dataset = eval(f"{mouse}_midn_result_dict")
        if i == 0:
            t_clocks_midn = dataset["reord_t-vals"][:,1]
            t_midn_midn = dataset["reord_t-vals"][:,2]
            t_loc_midn = dataset["reord_t-vals"][:,3]

        elif i > 0:
            t_clocks_midn = np.concatenate((t_clocks_midn, dataset["reord_t-vals"][:,1]))
            t_midn_midn = np.concatenate((t_midn_midn,dataset["reord_t-vals"][:,2]))
            t_loc_midn = np.concatenate((t_loc_midn, dataset["reord_t-vals"][:,3]))
    
    data_comp_mid = [t_clocks, t_clocks_midn, t_midn, t_midn_midn, t_loc, t_loc_midn]
    label_comp_mid = ["Musicbox, all","Musicbox, MidnN", "Midnight, all", "Midnight, MidnN", "Loc, all", "Loc, MidnN"]
    label_tick_comp_mid = [1,2,3,4,5,6]
    title_string_comp_mid = 'tvals per run, reordered'

    mc.analyse.analyse_ephys.plotting_hist_scat(data_comp_mid, label_comp_mid, label_tick_comp_mid, title_string_comp_mid) 
    
    
    
    # WITHOUT LOCATION
    letters = ['a','b','c','d','e','f', 'g', 'h']
    if load_old == False:
        for i, mouse in enumerate(letters):
            dataset = eval(f"{mouse}_reg_per_run")
            if i == 0:
                t_clocks = dataset["reord_t-vals"][:,1]
                t_midn = dataset["reord_t-vals"][:,2]
                #t_loc = dataset["reord_t-vals"][:,3]
    
            elif i > 0:
                t_clocks = np.concatenate((t_clocks, dataset["reord_t-vals"][:,1]))
                t_midn = np.concatenate((t_midn,dataset["reord_t-vals"][:,2]))
                #t_loc = np.concatenate((t_loc, dataset["reord_t-vals"][:,3]))
    if load_old == True:
        for i, mouse in enumerate(perrun_results):
            if mouse == 'me11_01122021_02122021_midn_dic':
                continue
            for i, mouse in enumerate(perrun_results):
                if i == 0:
                    t_clocks = perrun_results[mouse]["reord_t-vals"][:,1]
                    t_midn = perrun_results[mouse]["reord_t-vals"][:,2]
                elif i > 0:
                     t_clocks = np.concatenate((t_clocks, perrun_results[mouse]["reord_t-vals"][:,1]))
                     t_midn = np.concatenate((t_midn, perrun_results[mouse]["reord_t-vals"][:,2]))
                     

    # MIDNIGTH, REORDRED PER RUN
    if load_old == False:
        for i, mouse in enumerate(letters):
            dataset = eval(f"{mouse}_midn_result_dict")
            if i == 0:
                t_clocks_midn = dataset["reord_t-vals"][:,1]
                t_midn_midn = dataset["reord_t-vals"][:,2]
                #t_loc_midn = dataset["reord_t-vals"][:,3]
    
            elif i > 0:
                t_clocks_midn = np.concatenate((t_clocks_midn, dataset["reord_t-vals"][:,1]))
                t_midn_midn = np.concatenate((t_midn_midn,dataset["reord_t-vals"][:,2]))
                #t_loc_midn = np.concatenate((t_loc_midn, dataset["reord_t-vals"][:,3]))
    if load_old == True:
        for i, mouse in enumerate(midnight_results):
            if mouse == 'me11_01122021_02122021_midn_dic':
                continue
            if i == 0:
                t_clocks_midn = midnight_results[mouse]["reord_t-vals"][:,1]
                t_midn_midn = midnight_results[mouse]["reord_t-vals"][:,2]
            elif i > 0:
                 t_clocks_midn = np.concatenate((t_clocks_midn, midnight_results[mouse]["reord_t-vals"][:,1]))
                 t_midn_midn = np.concatenate((t_midn_midn, midnight_results[mouse]["reord_t-vals"][:,2]))
                 
    data_comp_mid = [t_clocks, t_clocks_midn, t_midn, t_midn_midn]
    label_comp_mid = ["Musicbox, all","Musicbox, MidnN", "Midnight, all", "Midnight, MidnN"]
    label_tick_comp_mid = [1,2,3,4]
    title_string_comp_mid = 'tvals per run, reordered'

    mc.analyse.analyse_ephys.plotting_hist_scat(data_comp_mid, label_comp_mid, label_tick_comp_mid, title_string_comp_mid) 
    
    
    data_comp_mid = [t_midn, t_midn_midn]
    label_comp_mid = ["Midnight, all", "Midnight, MidnN"]
    label_tick_comp_mid = [1,2]
    title_string_comp_mid = 'tvals per run, reordered'

    mc.analyse.analyse_ephys.plotting_hist_scat(data_comp_mid, label_comp_mid, label_tick_comp_mid, title_string_comp_mid) 
    
    # find out where these outliers come from
    if load_old == True:
        for i, mouse in enumerate(midnight_results):
            print(np.mean(midnight_results[mouse]["reord_t-vals"][:,1]))
    
    
    
    
    
    # secondly, plot the contrast clocks-midnight.
    # "The clocks model is a better predictor than the midnight model (contrast)
    
    
    # # then, plot 6 early vs 6 late runs for early/mid/late.
    # for i, mouse in enumerate(letters):
    #     dataset = eval(f"{mouse}_reg_per_run")
    #     if i == 0:
    #         t_cl_early = [runs[2] for runs in dataset["tval_early_without_phase"]]
    #         t_cl_mid  = [runs[2] for runs in dataset["tval_mid_without_phase"]]
    #         t_cl_late = [runs[2] for runs in dataset["tval_late_without_phase"]]
    #         t_mid_early = [runs[1] for runs in dataset["tval_early_without_phase"]]
    #         t_mid_mid = [runs[1] for runs in dataset["tval_mid_without_phase"]]
    #         t_mid_late = [runs[1] for runs in dataset["tval_late_without_phase"]]
    #     if i > 0:
    #         t_cl_early = np.concatenate((t_cl_early, [runs[2] for runs in dataset["tval_early_without_phase"]]))
    #         t_cl_mid  = np.concatenate((t_cl_mid, [runs[2] for runs in dataset["tval_mid_without_phase"]]))
    #         t_cl_late = np.concatenate((t_cl_late,[runs[2] for runs in dataset["tval_late_without_phase"]]))
    #         t_mid_early = np.concatenate((t_mid_early, [runs[1] for runs in dataset["tval_early_without_phase"]]))
    #         t_mid_mid = np.concatenate((t_mid_mid, [runs[1] for runs in dataset["tval_mid_without_phase"]]))
    #         t_mid_late = np.concatenate((t_mid_late, [runs[1] for runs in dataset["tval_late_without_phase"]]))
    # data_split_tval_perrun = [t_cl_early, t_cl_mid, t_cl_late, t_mid_early, t_mid_mid, t_mid_late]
    # label_split_tval_perrun = ['t_cl_early', 't_cl_mid', 't_cl_late', 't_mid_early', 't_mid_mid', 't_mid_late']
    # label_split_tick_list_tperrun = [1,2,3,4,5,6]
    # title_split_string_tperrun = 'tvals. SPLIT. Reg for FIRST 6 RUNS of each task. Contin, 10 bins, 3 phase neurons, mask within tasks'
    # mc.analyse.analyse_ephys.plotting_hist_scat(data_split_tval_perrun, label_split_tval_perrun, label_split_tick_list_tperrun, title_split_string_tperrun) 
    
    # for i, mouse in enumerate(letters):
    #     dataset = eval(f"{mouse}_reg_per_run")
    #     if i == 0:
    #         t_cl_early = [runs[2] for runs in dataset["tval_early_without_phase"]]
    #         t_cl_mid  = [runs[2] for runs in dataset["tval_mid_without_phase"]]
    #         t_cl_late = [runs[2] for runs in dataset["tval_late_without_phase"]]
    #         t_mid_early = [runs[1] for runs in dataset["tval_early_without_phase"]]
    #         t_mid_mid = [runs[1] for runs in dataset["tval_mid_without_phase"]]
    #         t_mid_late = [runs[1] for runs in dataset["tval_late_without_phase"]]
    #     if i > 0:
    #         t_cl_early = np.concatenate((t_cl_early, [runs[2] for runs in dataset["tval_early_without_phase"]]))
    #         t_cl_mid  = np.concatenate((t_cl_mid, [runs[2] for runs in dataset["tval_mid_without_phase"]]))
    #         t_cl_late = np.concatenate((t_cl_late,[runs[2] for runs in dataset["tval_late_without_phase"]]))
    #         t_mid_early = np.concatenate((t_mid_early, [runs[1] for runs in dataset["tval_early_without_phase"]]))
    #         t_mid_mid = np.concatenate((t_mid_mid, [runs[1] for runs in dataset["tval_mid_without_phase"]]))
    #         t_mid_late = np.concatenate((t_mid_late, [runs[1] for runs in dataset["tval_late_without_phase"]]))
    # data_split_tval_perrun = [t_cl_early, t_cl_mid, t_cl_late, t_mid_early, t_mid_mid, t_mid_late]
    # label_split_tval_perrun = ['t_cl_early', 't_cl_mid', 't_cl_late', 't_mid_early', 't_mid_mid', 't_mid_late']
    # label_split_tick_list_tperrun = [1,2,3,4,5,6]
    # title_split_string_tperrun = 'tvals. SPLIT. Reg for FIRST 6 RUNS of each task. Contin, 10 bins, 3 phase neurons, mask within tasks'
    # mc.analyse.analyse_ephys.plotting_hist_scat(data_split_tval_perrun, label_split_tval_perrun, label_split_tick_list_tperrun, title_split_string_tperrun) 
    
    
    
    
    # 26.06.: OK SO IF I DID EVERYTHING CORRECTLY, THEN ACTUALLY THE PREDICTION FOR MIDNIGHT BECOMES WORSE IF I ONLY
    # INCLUDE MOHAMADYS MIDNIGHT NEURONS.
    # DOUBLE CHECK WHAT WENT WRONG HERE!!
    # COULD IT BE THAT THE FILES WERE WRONGLY USED?
    # HOW MUCH OVERLAP IS THERE WITH THE OLD MIDNIGHT DEFINITION?
    # AND WHAT WOULD HAPPEN IF I'D TAKE OUT LOCATION FROM THE SPLIT REGRESSION?
    # now, mainly do comparisons between midnight neurons and complete neurons.
    # I want: 
    # contrast_result[1], t-values[2], reord_contrast[1], reord_t-vals[2], split_contrasts[012,2,:]
    datasets = []
    phase_split = ['early', 'mid', 'late']
    letters = ['a','b','c','d','e','f', 'g', 'h']
    key_list = []
    for key in perrun_results:
        key_list.append(key)
        
        
        
    # for i, mouse in enumerate(letters):
    #     if load_old == True: 
    #         dataset = perrun_results.copy()
    #     if load_old == False:
    #         datasets.append(f"{mouse}_reg_per_run")
    #         datasets.append(f"{mouse}_midn_result_dict")
    
    # if load_old == False:
    #     for i, mouse in enumerate(letters):
    #         datasets.append(f"{mouse}_reg_per_run")
    #         datasets.append(f"{mouse}_midn_result_dict")
    
    # elif load_old == True: 
    #         dataset = perrun_results.copy()
    #         dataset.append(midnight_results.copy())))
    #     if load_old == False:
    #         datasets.append(f"{mouse}_reg_per_run")
    #         datasets.append(f"{mouse}_midn_result_dict")
    datasets = {} 
    for i, mouse in enumerate(letters):
        if load_old == False:
            datasets["all"] = eval(f"{mouse}_reg_per_run")
            datasets["mid"] = eval(f"{mouse}_midn_result_dict")
        if load_old == True: 
            datasets["all"] = perrun_results[key_list[i]].copy()
            datasets["mid"] = (midnight_results[key_list[i]].copy())
    
        for one, dataset in enumerate(datasets):
            if one == 0:
                if i == 0: 
                        contr = [runs[1] for runs in datasets[dataset]["contrast_results"]]
                        t_val = [runs[2] for runs in datasets[dataset]["t-values"]]
                        reord_contr = [runs[1] for runs in datasets[dataset]["reord_contrasts"]]
                        reord_tval = [runs[2] for runs in datasets[dataset]["reord_t-vals"]]
                        earl_contr = [runs[1] for runs in datasets[dataset]["split_contrasts"][0]]
                        mid_contr = [runs[1] for runs in datasets[dataset]["split_contrasts"][1]]
                        late_contr = [runs[1] for runs in datasets[dataset]["split_contrasts"][2]]
                        earl_tval = datasets[dataset]["split_t-vals"][0,:,2]
                        mid_tval = datasets[dataset]["split_t-vals"][1,:,2]
                        late_tval = datasets[dataset]["split_t-vals"][2,:,2]
                elif i > 0:
                    contr = np.concatenate((contr, [runs[1] for runs in datasets[dataset]["contrast_results"]]))
                    t_val = np.concatenate((t_val, [runs[2] for runs in datasets[dataset]["t-values"]]))
                    reord_contr = np.concatenate((reord_contr,[runs[1] for runs in datasets[dataset]["reord_contrasts"]]))
                    reord_tval = np.concatenate((reord_tval,[runs[2] for runs in datasets[dataset]["reord_t-vals"]]))
                    earl_contr = np.concatenate((earl_contr,[runs[1] for runs in datasets[dataset]["split_contrasts"][0]]))
                    mid_contr = np.concatenate((mid_contr,[runs[1] for runs in datasets[dataset]["split_contrasts"][1]]))
                    late_contr = np.concatenate((late_contr, [runs[1] for runs in datasets[dataset]["split_contrasts"][2]]))
                    earl_tval = np.concatenate((earl_tval, datasets[dataset]["split_t-vals"][0,:,2]))
                    mid_tval = np.concatenate((mid_tval, datasets[dataset]["split_t-vals"][1,:,2]))
                    late_tval = np.concatenate((late_tval, datasets[dataset]["split_t-vals"][2,:,2]))
            
            elif one == 1: 
                if i == 0:
                        contr_mid = [runs[1] for runs in datasets[dataset]["contrast_results"]]
                        t_val_mid = [runs[2] for runs in datasets[dataset]["t-values"]]
                        reord_contr_mid = [runs[1] for runs in datasets[dataset]["reord_contrasts"]]
                        reord_tval_mid = [runs[2] for runs in datasets[dataset]["reord_t-vals"]]
                        earl_contr_mid = [runs[1] for runs in datasets[dataset]["split_contrasts"][0]]
                        mid_contr_mid = [runs[1] for runs in datasets[dataset]["split_contrasts"][1]]
                        late_contr_mid = [runs[1] for runs in datasets[dataset]["split_contrasts"][2]]
                        earl_tval_mid = datasets[dataset]["split_t-vals"][0,:,2]
                        mid_tval_mid = datasets[dataset]["split_t-vals"][1,:,2]
                        late_tval_mid = datasets[dataset]["split_t-vals"][2,:,2]
                elif i > 0: 
                        contr_mid = np.concatenate((contr_mid, [runs[1] for runs in datasets[dataset]["contrast_results"]]))
                        t_val_mid = np.concatenate((t_val_mid, [runs[2] for runs in datasets[dataset]["t-values"]]))
                        reord_contr_mid = np.concatenate((reord_contr_mid,[runs[1] for runs in datasets[dataset]["reord_contrasts"]]))
                        reord_tval_mid = np.concatenate((reord_tval_mid, [runs[2] for runs in datasets[dataset]["reord_t-vals"]]))
                        earl_contr_mid = np.concatenate((earl_contr_mid, [runs[1] for runs in datasets[dataset]["split_contrasts"][0]]))
                        mid_contr_mid = np.concatenate((mid_contr_mid, [runs[1] for runs in datasets[dataset]["split_contrasts"][1]]))
                        late_contr_mid = np.concatenate((late_contr_mid, [runs[1] for runs in datasets[dataset]["split_contrasts"][2]]))
                        earl_tval_mid = np.concatenate((earl_tval_mid, datasets[dataset]["split_t-vals"][0,:,2]))
                        mid_tval_mid = np.concatenate((mid_tval_mid, datasets[dataset]["split_t-vals"][1,:,2]))
                        late_tval_mid = np.concatenate((late_tval_mid, datasets[dataset]["split_t-vals"][2,:,2]))
                        
        data_mid_comp_contr = [contr, contr_mid, reord_contr, reord_contr_mid, earl_contr, earl_contr_mid, mid_contr, mid_contr_mid, late_contr, late_contr_mid]
        label_mid_comp_contr = ["contr", "contr_mid", "reord_contr", "reord_contr_mid", "earl_contr", "earl_contr_mid", "mid_contr", "mid_contr_mid", "late_contr", "late_contr_mid"]
        tick_mid_comp_contr = [1,2,3,4,5,6,7,8,9,10]
        title_mid_comp_contr = "comparing the midnight contrasts: all neurons vs. midnight neurons, per run."
        mc.analyse.analyse_ephys.plotting_hist_scat(data_mid_comp_contr, label_mid_comp_contr, tick_mid_comp_contr, title_mid_comp_contr) 
        
        
        data_mid_comp_t = [t_val, t_val_mid, reord_tval, reord_tval_mid]
        label_mid_comp_t = ["t_val", "tval_mid", "reord_tval", "reord_tval_mid"]
        tick_mid_comp_t = [1,2,3,4]
        title_mid_comp_t = "comparing the midnight contrasts: all neurons vs. midnight neurons, per run."
        mc.analyse.analyse_ephys.plotting_hist_scat(data_mid_comp_t, label_mid_comp_t, tick_mid_comp_t, title_mid_comp_t) 
        
        
    
# to answer my first question lab meeting.

letters = ['a','b','c','d','e','f', 'g', 'h']
tval_clocks=[]
tval_midn=[]
tval_phase=[]
tval_loc=[]
tval_reord_clocks=[]
tval_reord_midn=[]
tval_reord_locs=[]

for i, mouse in enumerate(letters):   
    if load_old == False:
        dataset = eval(f"{mouse}_reg_result_dict")
        tval_clocks.append(dataset["normal"]["t_vals"][1])
        tval_midn.append(dataset["normal"]["t_vals"][2])
        tval_phase.append(dataset["normal"]["t_vals"][3])
        tval_loc.append(dataset["normal"]["t_vals"][4])
        tval_reord_clocks.append(dataset["reord_t-vals"][1])
        tval_reord_midn.append(dataset["reord_t-vals"][2])
        tval_reord_locs.append(dataset["reord_t-vals"][3])

data_compl = [tval_clocks, tval_phase, tval_loc, tval_midn]  
string_compl = ["tval musicbox", "tval phase", "tval location", "tval midnight"]          
tick_compl = [1,2,3,4]
title_compl = "t values for the complete model, averaged across tasks"
mc.analyse.analyse_ephys.plotting_hist_scat(data_compl, string_compl, tick_compl, title_compl) 

data_compl_reord = [tval_reord_clocks, tval_reord_locs, tval_reord_midn]  
string_compl_reord = ["musicbox","location", "midnight"]          
tick_compl_reord = [1,2,3]
title_compl_reord = "t values for the reordered model, averaged across tasks"
mc.analyse.analyse_ephys.plotting_hist_scat(data_compl_reord, string_compl_reord, tick_compl_reord, title_compl_reord) 
  

# # PLOT THE RESULTS OF THIS!!
# # one plot: all neurons
# data_all = [[a_reg_result_dict["reg_early_phase_midnight-clocks"][0],b_reg_result_dict["reg_early_phase_midnight-clocks"][0], c_reg_result_dict["reg_early_phase_midnight-clocks"][0], d_reg_result_dict["reg_early_phase_midnight-clocks"][0], e_reg_result_dict["reg_early_phase_midnight-clocks"][0]],
#               [a_reg_result_dict["reg_early_phase_midnight-clocks"][1],b_reg_result_dict["reg_early_phase_midnight-clocks"][1], c_reg_result_dict["reg_early_phase_midnight-clocks"][1], d_reg_result_dict["reg_early_phase_midnight-clocks"][1], e_reg_result_dict["reg_early_phase_midnight-clocks"][1]],
#               [a_reg_result_dict["reg_mid_phase_midnight-clocks"][0],b_reg_result_dict["reg_mid_phase_midnight-clocks"][0], c_reg_result_dict["reg_mid_phase_midnight-clocks"][0], d_reg_result_dict["reg_mid_phase_midnight-clocks"][0], e_reg_result_dict["reg_mid_phase_midnight-clocks"][0]],
#               [a_reg_result_dict["reg_mid_phase_midnight-clocks"][1],b_reg_result_dict["reg_mid_phase_midnight-clocks"][1], c_reg_result_dict["reg_mid_phase_midnight-clocks"][1], d_reg_result_dict["reg_mid_phase_midnight-clocks"][1], e_reg_result_dict["reg_mid_phase_midnight-clocks"][1]],
#               [a_reg_result_dict["reg_late_phase_midnight-clocks"][0],b_reg_result_dict["reg_late_phase_midnight-clocks"][0], c_reg_result_dict["reg_late_phase_midnight-clocks"][0], d_reg_result_dict["reg_late_phase_midnight-clocks"][0], e_reg_result_dict["reg_late_phase_midnight-clocks"][0]],
#               [a_reg_result_dict["reg_late_phase_midnight-clocks"][1],b_reg_result_dict["reg_late_phase_midnight-clocks"][1], c_reg_result_dict["reg_late_phase_midnight-clocks"][1], d_reg_result_dict["reg_late_phase_midnight-clocks"][1], e_reg_result_dict["reg_late_phase_midnight-clocks"][1]],
#               [a_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0],b_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0], c_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0], d_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0], e_reg_result_dict["reg_all_reversedphase_midnight-clocks"][0]],
#               [a_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1],b_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1], c_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1], d_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1], e_reg_result_dict["reg_all_reversedphase_midnight-clocks"][1]]]


# label_string_list_all = ['early midight', 'early clocks', 'mid midnight', 'mid clocks', 'late midnight', 'late clocks', 'put together midnight', 'put together clocks']
# label_tick_list_all = [0,1,2,3,4,5,6,7]
# title_string_all = 'All neurons included, averaged over trials, between tasks, binned in 10 bins per state, continuous model'

# mc.analyse.analyse_ephys.plotting_hist_scat(data_all, label_string_list_all, label_tick_list_all, title_string_all)


# # second plot: only midnight
# data_midn = [[a_midn_result_dict["reg_early_phase_midnight-clocks"][0],b_midn_result_dict["reg_early_phase_midnight-clocks"][0], c_midn_result_dict["reg_early_phase_midnight-clocks"][0], d_midn_result_dict["reg_early_phase_midnight-clocks"][0], e_midn_result_dict["reg_early_phase_midnight-clocks"][0]],
#               [a_midn_result_dict["reg_early_phase_midnight-clocks"][1],b_midn_result_dict["reg_early_phase_midnight-clocks"][1], c_midn_result_dict["reg_early_phase_midnight-clocks"][1], d_midn_result_dict["reg_early_phase_midnight-clocks"][1], e_midn_result_dict["reg_early_phase_midnight-clocks"][1]],
#               [a_midn_result_dict["reg_mid_phase_midnight-clocks"][0],b_midn_result_dict["reg_mid_phase_midnight-clocks"][0], c_midn_result_dict["reg_mid_phase_midnight-clocks"][0], d_midn_result_dict["reg_mid_phase_midnight-clocks"][0], e_midn_result_dict["reg_mid_phase_midnight-clocks"][0]],
#               [a_midn_result_dict["reg_mid_phase_midnight-clocks"][1],b_midn_result_dict["reg_mid_phase_midnight-clocks"][1], c_midn_result_dict["reg_mid_phase_midnight-clocks"][1], d_midn_result_dict["reg_mid_phase_midnight-clocks"][1], e_midn_result_dict["reg_mid_phase_midnight-clocks"][1]],
#               [a_midn_result_dict["reg_late_phase_midnight-clocks"][0],b_midn_result_dict["reg_late_phase_midnight-clocks"][0], c_midn_result_dict["reg_late_phase_midnight-clocks"][0], d_midn_result_dict["reg_late_phase_midnight-clocks"][0], e_midn_result_dict["reg_late_phase_midnight-clocks"][0]],
#               [a_midn_result_dict["reg_late_phase_midnight-clocks"][1],b_midn_result_dict["reg_late_phase_midnight-clocks"][1], c_midn_result_dict["reg_late_phase_midnight-clocks"][1], d_midn_result_dict["reg_late_phase_midnight-clocks"][1], e_midn_result_dict["reg_late_phase_midnight-clocks"][1]],
#               [a_midn_result_dict["reg_all_reversedphase_midnight-clocks"][0],b_midn_result_dict["reg_all_reversedphase_midnight-clocks"][0], c_midn_result_dict["reg_all_reversedphase_midnight-clocks"][0], d_midn_result_dict["reg_all_reversedphase_midnight-clocks"][0], e_midn_result_dict["reg_all_reversedphase_midnight-clocks"][0]],
#               [a_midn_result_dict["reg_all_reversedphase_midnight-clocks"][1],b_midn_result_dict["reg_all_reversedphase_midnight-clocks"][1], c_midn_result_dict["reg_all_reversedphase_midnight-clocks"][1], d_midn_result_dict["reg_all_reversedphase_midnight-clocks"][1], e_midn_result_dict["reg_all_reversedphase_midnight-clocks"][1]]]


# label_string_list_all = ['early midight', 'early clocks', 'mid midnight', 'mid clocks', 'late midnight', 'late clocks', 'put together midnight', 'put together clocks']
# label_tick_list_all = [0,1,2,3,4,5,6,7]
# title_string_all = 'Only MIDNIGHT neurons included, averaged over trials, between tasks, binned in 10 bins per state, continuous model'

# mc.analyse.analyse_ephys.plotting_hist_scat(data_midn, label_string_list_all, label_tick_list_all, title_string_all)


# # third plot: the whole clocks
# data_clocks = [[a_clocks_result_dict["reg_early_phase_midnight-clocks"][0],b_clocks_result_dict["reg_early_phase_midnight-clocks"][0], c_clocks_result_dict["reg_early_phase_midnight-clocks"][0], d_clocks_result_dict["reg_early_phase_midnight-clocks"][0], e_clocks_result_dict["reg_early_phase_midnight-clocks"][0]],
#               [a_clocks_result_dict["reg_early_phase_midnight-clocks"][1],b_clocks_result_dict["reg_early_phase_midnight-clocks"][1], c_clocks_result_dict["reg_early_phase_midnight-clocks"][1], d_clocks_result_dict["reg_early_phase_midnight-clocks"][1], e_clocks_result_dict["reg_early_phase_midnight-clocks"][1]],
#               [a_clocks_result_dict["reg_mid_phase_midnight-clocks"][0],b_clocks_result_dict["reg_mid_phase_midnight-clocks"][0], c_clocks_result_dict["reg_mid_phase_midnight-clocks"][0], d_clocks_result_dict["reg_mid_phase_midnight-clocks"][0], e_clocks_result_dict["reg_mid_phase_midnight-clocks"][0]],
#               [a_clocks_result_dict["reg_mid_phase_midnight-clocks"][1],b_clocks_result_dict["reg_mid_phase_midnight-clocks"][1], c_clocks_result_dict["reg_mid_phase_midnight-clocks"][1], d_clocks_result_dict["reg_mid_phase_midnight-clocks"][1], e_clocks_result_dict["reg_mid_phase_midnight-clocks"][1]],
#               [a_clocks_result_dict["reg_late_phase_midnight-clocks"][0],b_clocks_result_dict["reg_late_phase_midnight-clocks"][0], c_clocks_result_dict["reg_late_phase_midnight-clocks"][0], d_clocks_result_dict["reg_late_phase_midnight-clocks"][0], e_clocks_result_dict["reg_late_phase_midnight-clocks"][0]],
#               [a_clocks_result_dict["reg_late_phase_midnight-clocks"][1],b_clocks_result_dict["reg_late_phase_midnight-clocks"][1], c_clocks_result_dict["reg_late_phase_midnight-clocks"][1], d_clocks_result_dict["reg_late_phase_midnight-clocks"][1], e_clocks_result_dict["reg_late_phase_midnight-clocks"][1]],
#               [a_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][0],b_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][0], c_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][0], d_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][0], e_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][0]],
#               [a_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][1],b_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][1], c_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][1], d_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][1], e_clocks_result_dict["reg_all_reversedphase_midnight-clocks"][1]]]


# label_string_list_all = ['early midight', 'early clocks', 'mid midnight', 'mid clocks', 'late midnight', 'late clocks', 'put together midnight', 'put together clocks']
# label_tick_list_all = [0,1,2,3,4,5,6,7]
# title_string_all = 'Only CLOCK neurons included, averaged over trials, between tasks, binned in 10 bins per state, continuous model'

# mc.analyse.analyse_ephys.plotting_hist_scat(data_clocks, label_string_list_all, label_tick_list_all, title_string_all)


# # build another figure.
# # this time, build it with the normal regression
# letters = ['a','b','c','d','e','f', 'g', 'h']
# list_allneurons =[]
# list_midn = []
# list_clockn = []
# for i, mouse in enumerate(letters):
#     dataset_all = eval(f"{mouse}_reg_result_dict")
#     dataset_midn = eval(f"{mouse}_midn_result_dict")
#     dataset_clock = eval(f"{mouse}_clocks_result_dict")
#     list_allneurons.append(dataset_all['reg_all_midnight-clocks-loc-phase'])
#     list_midn.append(dataset_midn['reg_all_midnight-clocks-loc-phase'])
#     list_clockn.append(dataset_clock['reg_all_midnight-clocks-loc-phase'])

# plotting_clockn = np.empty((len(list_clockn[0]), len(list_clockn)))
# for col, regressor in enumerate(range(0, len(list_clockn[0]))):
#     for row, dataset in enumerate(range(0, len(list_clockn))):
#         plotting_clockn[col, row]= list_clockn[dataset][regressor]

# list_plot_clock = []
# for row in plotting_clockn:
#     list_plot_clock.append(row)

# plotting_midn = np.empty((len(list_midn[0]), len(list_midn)))
# for col, regressor in enumerate(range(0, len(list_midn[0]))):
#     for row, dataset in enumerate(range(0, len(list_midn))):
#         plotting_midn[col, row]= list_midn[dataset][regressor]

# list_plot_midn = []
# for row in plotting_midn:
#     list_plot_midn.append(row)

# plotting_all = np.empty((len(list_allneurons[0]), len(list_allneurons)))
# for col, regressor in enumerate(range(0, len(list_allneurons[0]))):
#     for row, dataset in enumerate(range(0, len(list_allneurons))):
#         plotting_all[col, row]= list_allneurons[dataset][regressor]

# list_plot_all = []
# for row in plotting_all:
#     list_plot_all.append(row)
     
# for elem in list_plot_midn:
#     list_plot_all.append(elem)

# for elem in list_plot_midn:
#     list_plot_all.append(elem)

# labels_compl_reg = ['all_midn', 'all_clocks', 'all_loc', 'all_phase', 'clock_midn', 'clock_clocks', 'clock_loc', 'clock_phase','midn_midn', 'midn_clocks', 'midn_loc', 'midn_phase']
# label_ticks_compl_reg = [1,2,3,4,5,6,7,8,9,10,11,12]
# title_compl_reg = ['Complete regression, leaving neurons out'] 
# mc.analyse.analyse_ephys.plotting_hist_scat(list_plot_all, labels_compl_reg, label_ticks_compl_reg, title_compl_reg)   
 
 

# # build another figure.
# # use the early-mid-late weights
# letters = ['a','b','c','d','e','f', 'g', 'h']
# list_allneurons =[]
# list_midn = []
# list_clockn = []
# for part in ['reg_early_phase_midnight-clocks', 'reg_mid_phase_midnight-clocks', 'reg_late_phase_midnight-clocks']:
#     for i, mouse in enumerate(letters):
#         dataset_all = eval(f"{mouse}_reg_result_dict")
#         dataset_midn = eval(f"{mouse}_midn_result_dict")
#         dataset_clock = eval(f"{mouse}_clocks_result_dict")    
#         list_allneurons.append(dataset_all[part])
#         list_midn.append(dataset_midn[part])
#         list_clockn.append(dataset_clock[part])

# early_midnight_all = []
# early_clock_all = []
# early_midnight_midn = []
# early_clock_midn = []
# early_midnight_clo = []
# early_clock_clo = []
# for dataset,x in enumerate(letters):
#     early_midnight_all.append(list_allneurons[dataset][0])
#     early_clock_all.append(list_allneurons[dataset][1])
#     early_midnight_midn.append(list_midn[dataset][0])
#     early_clock_midn.append(list_midn[dataset][1])
#     early_midnight_clo.append(list_clockn[dataset][0])
#     early_clock_clo.append(list_clockn[dataset][1])

# mid_midnight_all = []
# mid_clocks_all = []
# mid_midnight_midn = []
# mid_clock_midn = []
# mid_midnight_clo = []
# mid_clock_clo = []
# for dataset,x in enumerate(letters):
#     mid_midnight_all.append(list_allneurons[dataset+len(letters)][0])
#     mid_clocks_all.append(list_allneurons[dataset+len(letters)][1])
#     mid_midnight_midn.append(list_midn[dataset+len(letters)][0])
#     mid_clock_midn.append(list_midn[dataset+len(letters)][1])
#     mid_midnight_clo.append(list_clockn[dataset+len(letters)][0])
#     mid_clock_clo.append(list_clockn[dataset+len(letters)][1])

# late_midnight_all = []
# late_clocks_all = []
# late_midnight_midn = []
# late_clock_midn = []
# late_midnight_clo = []
# late_clock_clo = []
# for dataset,x in enumerate(letters):
#     late_midnight_all.append(list_allneurons[dataset+len(letters)*2][0])
#     late_clocks_all.append(list_allneurons[dataset+len(letters)*2][1])
#     late_midnight_midn.append(list_midn[dataset+len(letters)*2][0])
#     late_clock_midn.append(list_midn[dataset+len(letters)*2][1])
#     late_midnight_clo.append(list_clockn[dataset+len(letters)*2][0])
#     late_clock_clo.append(list_clockn[dataset+len(letters)*2][1])


# plotting_early = [early_midnight_all, early_midnight_clo, early_midnight_midn, early_clock_all, early_clock_clo, early_clock_midn]    

# labels_early_reg = ['midnight_all', 'midnight_clo', 'midnight_midn', 'clock_all', 'clock_clo', 'clock_midn']
# label_ticks_early_reg = [1,2,3,4,5,6]
# title_early_reg = ['Early comparing all, only clock and only midn neurons'] 
# mc.analyse.analyse_ephys.plotting_hist_scat(plotting_early, labels_early_reg, label_ticks_early_reg, title_early_reg)   
 
# plotting_mid = [mid_midnight_all, mid_midnight_clo, mid_midnight_midn, mid_clocks_all, mid_clock_clo, mid_clock_midn]    

# labels_mid_reg = ['midnight_all', 'midnight_clo', 'midnight_midn', 'clock_all', 'clock_clo', 'clock_midn']
# label_ticks_mid_reg = [1,2,3,4,5,6]
# title_mid_reg = ['Mid comparing all, only clock and only midn neurons'] 
# mc.analyse.analyse_ephys.plotting_hist_scat(plotting_mid, labels_mid_reg, label_ticks_mid_reg, title_mid_reg)   

# plotting_late = [late_midnight_all, late_midnight_clo, late_midnight_midn, late_clocks_all, late_clock_clo, late_clock_midn]    

# labels_late_reg = ['midnight_all', 'midnight_clo', 'midnight_midn', 'clock_all', 'clock_clo', 'clock_midn']
# label_ticks_late_reg = [1,2,3,4,5,6]
# title_late_reg = ['Late comparing all, only clock and only midn neurons'] 
# mc.analyse.analyse_ephys.plotting_hist_scat(plotting_late, labels_late_reg, label_ticks_late_reg, title_late_reg)   


################################################################################

 
    
# # PART 2: 

# # # THIS IS THE INTERESTIGN THING RN

# # # # SOMETHING GOES WRONG HERE, SUPER LARGE betas

# # # # SOMETHING GOES WRONG HERE, SUPER SMALLbetas    


# # # plot one violin plot per subject (i.e. 8 violins) where I visualize the variability across tasks per subject
# # # per contrast 



# # Here I want to compare what the binning does to a within task regression.
# # to this means, plot the respective contrasts next to each other.


# # COMPARE BINNING BETWEEN TASKS
# #  between-task regression based on single runs rather than averaging and then doing the regression.
# # possibility to try different binnings


# # look at what happens with more hase neurons: 11 PHASE NEURONS
# # COMPARE BETWEEN BINS IF I SPLIT BY PHASE

# # so for some reason the 3 binning doesn't work at all here. check this result!!
# # I just get crazy high values for phase.


# # DO THE BETWEEN COMPARISON FOR 11 NEURONS and DIFFERENT BINS IF I DO  NOT SPLIT PHASE.
# results_between_mouse_a_3bin_11_one = mc.analyse.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 3, ignore_double_tasks=1, split_by_phase= 0, number_phase_neurons= 11)
# results_between_mouse_a_10bin_11_one = mc.analyse.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 10, ignore_double_tasks=1, split_by_phase= 0, number_phase_neurons=11)
# results_between_mouse_a_30bin_11_one = mc.analyse.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, mouse_recday = 'me11_05122021_06122021', continuous = True, no_bins_per_state = 30, ignore_double_tasks=1, split_by_phase= 0, number_phase_neurons=11)


# # I want to know:
#     # 1. if there is a difference between the bins
#     # 2. how the betas change between early mid late
#     # 3. how it changes if I include phase or not
#     # 4. how it changes depending on the neuron-number in phase


# # CHECK WHAT CHANGES IF I ONLY TAKE CERTAIN NEURONS.
# # here I onlye select the anchored neurons. the midnight and clocks model should get a lot better
# # in predicting the data.


# # Since the phase-regressor is always way stronger for the between-data analysis, 
# # next step is to make phase irrelavant.
# # it seemed as if the 30-binning was quite a good choice. This will also allow me 
# # to more or less get rid of phase.
# results_mouse_a_6bins = mc.analyse.analyse_ephys.reg_between_tasks_singleruns(a_rewards_configs, a_locations, a_neurons, a_timings, contrast_matrix, continuous = True, no_bins_per_state = 6, mouse_recday = 'me11_05122021_06122021', split_by_phase = 1)


# # GROUP LEVEL ANALYSIS next.
# # problem: there might be outliers!!
# # coorperate a breakpoint if beta > 100 or so to check what is going wrong with the GLM in this case

    
#     # 1. subject level.
#     # for every mouse and every run, compute a GLM with my 3 regressors.
#     # 2. compute contrasts:
#     # I want to know: [0 0 1], [0 1 0], [1 0 0] and [-1 1 0], [0 -1 1], ....
#     # (every regressor at its own, and the cotnrast between 2 betas (MRI: PEs) which is 
#     # 1 minus the other)
#     # take all of these values and average 
#     #        1. across runs within one task config
#     #       2. across task configs
#     # Finally, you end up with 9 betas (MRI: COPEs) for every mouse (contrasts)
#     # 3. Group level:
#         # compute a random effects GLM for every of the contrasts, using
#         # each mouse-beta as an input
    
    
    
#     # next step: Stats! > group statistics?
#     # > multiple runs? use the regressor model since there are different fields the mouse runs on? 
#     # across tasks?? 
#     # also check the correlation values, independent from the other regressors
    
#     # before: within a mouse > fixed effects (+averaging betas) to compare runs 
    
    
#     # last step: check ou FSL FEAT -> compare betas across mice with random effects if 
#     # thats possible with only 8 mice, otherwise fixed effects 
    
    
    
#     # potentially, also have a look at a second thing:
#         # concatenate all trials across task cofngis
#         # then reduce the size of the data to steps instead of ms
#         # by using the step-regressors from the fMRI model (or the other way around)
#     # afterwards, follow the same group-stats and contrasts
#     # these contrasts are probably even more significant and will be more like my fmRI analyssi
    



'''
####################
###Data structure###
####################


##Neuron npy files:
These contain the normalized firing rates of each neuron - spikes per frame (frame rate is 60 Hz)

npy files for each session
matrix with dimensions [neuron,trial,bin]

360 bins per trial, every state is 90 bins

##Location npy files:
These contain the location of the animal in each bin (should correspond exactly to the neuron bins)
locations1-9 are the 9 nodes

Then the remaining locations 10-21 are the bridges, coded in the "Edge_grid" array. just subtract 10 and that
gives you the index of the Edge_grid array which tells you which bridges are being referenced
e.g. an entry of 10 means index 0 which is array([1, 2]) (i.e. animal is at the the bridge between nodes 1 and 2)

##task_data
this is the task sequence used for that day (which nodes are rewarded in what order)


#####Raw data:

Neuron_raw arrays are matrices of shape neurons X bins
each bin is the firing rate in a 25 ms timewindow

Location_raw arrays are arrays of length equal to the number of bins for the Neuron_raw matrix (may be 1 off)

trialtimes arrays are times (in ms) of each state: the first four columns are the start of each state
the fifth column is the end of the last state (D)

Pokes and Tone times in ms aligned to first trial - some will have negative timestamps if anumal poked before activating
first A

Note that you'll have to convert the trial times, pokes and tone tomes from ms to bin number to subset the neuron
and location arrays (i.e. divide by 25)

###Scores

A boolean matrix of N trials X N states (4) showing whether animal took shortest route between two
reward locations or not (1 if shortest route taken; 0 if not)


####Analysis

Phase_state_place_anchoring a Nx4 boolean matrix (N= number of neurons) outlining results of tuning analysis: columns
aranged as follows: Phase, State, Place and Anchoring
e.g. a cell thats True, True, True, False is phase, state and place tuned but not spatially anchored
Phase and Place determined by beta values obtained from a GLM looking at tuning consistency across tasks
State boolean is determined by comparing peaks in mean firing in state space to those from shuffled data - a state cell
has significant peaks in atleast half of the tasks
Anchoring via relative rotation to possible anchors - an anchored cell has consiustent anchoring in atleast half of
the tasks


##Anchor lags:

Anchor_lag_
An array of 12 bins per neurons, the value at index 0 corresponds to the mean correlation of the neuron's
spatial maps averaged across all tasks - index 1 corresponds to spatial correlations of activity lagged by one phase
(one third of a state) in the past (activity of neuron relative to where the animal was one phase in the past)...and
the lags continue on by one phase at a time for each of the remaining 10 indices


Anchor_lag_threshold_
A single value per neuron for the threshold to determine spatial correlation to be significant,
calculated using a shuffling approach.


########
All thresholds currently being refined


'''
