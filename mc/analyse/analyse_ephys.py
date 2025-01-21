#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:10:30 2023

this module can be called to prepare raw ephys data as well as to do any analysis with the ephzs data.
Currently, the first function prepares the data and runs a regression of my models onto the data, 
separetly for one task [reg_per_task_config]
The second function [reg_across_tasks_playground] does a similar thing, but for all task configs of 
one mouse recording. It also includes a bit of playing around with stuff.


@author: Svenja Küchenhoff
"""

import math
import numpy as np
import mc
from matplotlib import pyplot as plt
import scipy
import colormaps as cmaps 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import textwrap
from scipy.stats import ttest_1samp


def reg_per_task_config(task_configs, locations_all, neurons, timings_all, contrast_m, mouse_recday, continuous = True, no_bins_per_state = 0):
    # import pdb; pdb.set_trace()
    # mouse a
    contrast_m = np.array(contrast_m)
    coefficient = list()
    contrast_results_all = list()
    for task_no, task_config in enumerate(task_configs):
        # first convert trial times from ms to bin number to match neuron and location arrays 
        # (1 bin = 25ms)
        timings_task = timings_all[task_no].copy()
        for r, row in enumerate(timings_task):
            for c, element in enumerate(row):
                timings_task[r,c] = element/25
    
        # second, change locations and rewards to 0 and ignoring bridges
        locations_task = locations_all[task_no]
        for i, field in enumerate(locations_task):
            if field > 9: 
                locations_task[i] = locations_task[i-1].copy()
            if math.isnan(field):
                # keep the location bc of timebins
                locations_task[i] = locations_task[i-1].copy()
                    
    
        # important: fields need to be between 0 and 8, and keep them as integers!
        locations_task = [int((field_no-1)) for field_no in locations_task]
        task_config = [int((field_no-1)) for field_no in task_config]

        # start of the second loop: loop through every single trial for this specific task config.
        coefficients_per_trial = np.zeros((len(timings_task),len(contrast_m[0])))
        for trial_no, row in enumerate(timings_task):
            
            # current data of this specific run
            trajectory = locations_task[row[0]:row[-1]].copy()
        

            curr_neurons = neurons[task_no][:,row[0]:row[-1]].copy()
            # z-score neurons
            curr_neurons = scipy.stats.zscore(curr_neurons, axis=1)
            
            # ISSUE 21.04.23:
            # if there are ONLY 0 for one timestep, the np.corrcoef will output nan for that instance. Maybe better:
            # replace by super super low value
            # Update 9th of May: DONT DO THAT!
            # better: ignore those rows/ values in the correlation
            # for col_no, column in enumerate(curr_neurons.T):
            #     if np.all(column == 0):
            #         curr_neurons[:,col_no] = 0.00001
        
            # some pre-processing to create my models.
            # to count subpaths
            subpath_file = [locations_task[row[0]:row[1]+1], locations_task[row[1]+1:row[2]+1], locations_task[row[2]+1:row[3]+1], locations_task[row[3]+1:row[4]+1]]
            timings_curr_run = [(elem - row[0]) for elem in row]
        
            # to find out the step number per subpath
            step_number = [0,0,0,0] 
            for path_no, subpath in enumerate(subpath_file):
                for i, field in enumerate(subpath):
                    if i == 0:
                        count = 0
                    elif field != subpath[i-1]:
                        count+=1
                step_number[path_no] = count
               
            # mark where steps are made
            for field_no, field in enumerate(trajectory):
                if field_no == 0:
                    index_make_step = [0]
                elif field != trajectory[field_no-1]:
                    index_make_step.append(field_no)
                    
            
            if continuous == True:
                location_model, phase_model, state_model, midnight_model, clocks_model, phase_state_model = mc.simulation.predictions.set_continous_models_ephys(trajectory, timings_curr_run, index_make_step, step_number)
            
                # can delete but nice to have in case I'm interested in plotting
                # plt.figure(); 
                # plt.imshow(location_model, aspect='auto')
                # plt.figure(); 
                # plt.imshow(phase_model, aspect='auto')
                # plt.figure();  
                # plt.imshow(state_model, aspect='auto')
                # plt.figure()
                # plt.imshow(midnight_model, aspect='auto')
                # plt.figure(); 
                # plt.imshow(phase_state_model, aspect='auto')
                # plt.figure(); 
                # plt.imshow(clocks_model, aspect='auto')
                # for row in range(0, len(clocks_model), len(phase_state_model)):
                #     plt.axhline(row, color='white', ls='dashed')
            
            else:
                location_model = mc.simulation.predictions.set_location_raw_ephys(trajectory, step_time = 1, grid_size=3, plotting = False, field_no_given= 1)
                midnight_model, clocks_model, midnight_two = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, step_number, field_no_given= 1, plotting=False)
                phase_model = mc.simulation.predictions.set_phase_model_ephys(trajectory, timings_curr_run, index_make_step, step_number)

            # then bin the data if wanted
            if no_bins_per_state > 0:
                # first generate regressors per phase
                regs_phase_state_run = mc.simulation.predictions.create_x_regressors_per_state(trajectory, timings_curr_run, step_number, no_regs_per_state = no_bins_per_state)
                # then use these regressors to timebin
                curr_neurons = mc.simulation.predictions.transform_data_to_betas(curr_neurons, regs_phase_state_run)
                clocks_model = mc.simulation.predictions.transform_data_to_betas(clocks_model, regs_phase_state_run)
                midnight_model= mc.simulation.predictions.transform_data_to_betas(midnight_model, regs_phase_state_run)
                location_model = mc.simulation.predictions.transform_data_to_betas(location_model, regs_phase_state_run)
                phase_model = mc.simulation.predictions.transform_data_to_betas(phase_model, regs_phase_state_run)
                
            
            # now create the model RDMs
            RSM_location = mc.simulation.RDMs.within_task_RDM(location_model, plotting = False, titlestring = 'Location RDM')
            RSM_clock = mc.simulation.RDMs.within_task_RDM(clocks_model, plotting = False, titlestring = 'Clock RDM')
            RSM_midnight = mc.simulation.RDMs.within_task_RDM(midnight_model, plotting = False, titlestring = 'Midnight RDM')
            RSM_phase = mc.simulation.RDMs.within_task_RDM(phase_model, plotting = False, titlestring = 'Phase RDM')
        
            # now create the data RDM
            # I am wondering if this is correct, though - maybe should I select those neurons where I know fit my predictions?
            RSM_neurons = mc.simulation.RDMs.within_task_RDM(curr_neurons, plotting = False, titlestring = 'Data RDM')
        
            # Lastly, create a linear regression with RSM_loc,clock and midnight as regressors and data to be predicted
            results_reg, scipy_regression_results = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons, regressor_one_matrix=RSM_clock, regressor_two_matrix= RSM_midnight, regressor_three_matrix= RSM_location, regressor_four_matrix= RSM_phase)
            # print(f" The beta for the clocks model is {reg_res.coef_[0]}, for the midnight model is {reg_res.coef_[1]}, and for the location model is {reg_res.coef_[2]}")
            coefficients_per_trial[trial_no] = results_reg.coef_
            # print(f" Computed betas for run {trial_no} of task {task_config}")
            if sum(results_reg.coef_) > 100:
                import pdb; pdb.set_trace()
                
                # break because this can't be true!
                
            # then compute contrasts
            # I want to know: [0 0 1], [0 1 0], [1 0 0] and [-1 1 0], [0 -1 1], ....
        
        # # in case I want to have an overview of all betas for this trial config
        # x = np.linspace(0,len(coefficients_per_trial)-1,len(coefficients_per_trial))
        # plt.figure(); plt.plot(x, coefficients_per_trial[:,0], label = 'clocks'); plt.plot(x, coefficients_per_trial[:,1], label = 'midnight'); plt.plot(x, coefficients_per_trial[:,2], label = 'location'); plt.plot(x, coefficients_per_trial[:,3], label = 'phase'); plt.legend(loc="upper left"); plt.ylabel('beta'); plt.xlabel('run number'); plt.axhline(0, color='grey', ls='dashed'); plt.title(f"Recording day {mouse_recday} task {task_no}")
        
        
        contrast_results = np.zeros((len(contrast_m),len(coefficients_per_trial)))
        for contrast_no, contrast in enumerate(contrast_m):
            contrast_results[contrast_no,:] = np.matmul(contrast,coefficients_per_trial.transpose())
              
        # at the end of the trial, store the whole matrix in coefficient:
        coefficient.append(coefficients_per_trial)
        contrast_results_all.append(contrast_results)
        print(f"done with task_config {task_no}")
      
    return coefficient, contrast_results_all




def reg_between_tasks_singleruns(task_configs, locations_all, neurons, timings_all, contrast_m,  mouse_recday, contrast_split = None, continuous = True, no_bins_per_state = 0, split_by_phase = 1, number_phase_neurons = 3, mask_within = True, plotting = False):
    #import pdb; pdb.set_trace()
    
    # first, find out which is the largest shared trial number between all task configs
    # based on the biggest shared run number,
    # always concatenate the first,...nth run of one task with all other tasks
    labels_regs = []
    min_trialno = 60
    for task_number in timings_all:
        curr_trialno = len(task_number)
        if curr_trialno < min_trialno:
            min_trialno = curr_trialno
    
    # always take the 6 first ones.
    # min_trialno = 6
    
    contrast_m = np.array(contrast_m)
    coefficients_per_trial = np.zeros((min_trialno,len(contrast_m[0])))
    tvals_per_trial = np.zeros((min_trialno,1+len(contrast_m[0])))
    contrast_results = np.zeros((min_trialno, len(contrast_m)))
    
    coefficients_per_trial_only_clo = np.zeros((min_trialno,1))
    tvals_per_trial_only_cl = np.zeros((min_trialno,2))
    
    if split_by_phase == 1:
        phase_split = ['early', 'mid', 'late']
        coefficients_per_trial_split = np.zeros((len(phase_split), min_trialno,len(contrast_split[0])))
        tvals_per_trial_split = np.zeros((len(phase_split),min_trialno,1+len(contrast_split[0])))
        contrast_results_split = np.zeros((len(phase_split),min_trialno,len(contrast_split)))
        tval_reor = np.zeros((min_trialno,1+len(contrast_split[0])))
        coef_reor = np.zeros((min_trialno,len(contrast_split[0])))
        contr_reor = np.zeros((min_trialno, len(contrast_split)))
        

    for no_trial_in_each_task in range(0, min_trialno):
        for task_no, task_config in enumerate(task_configs):
            run_no = -1*(no_trial_in_each_task + 1)
            # run_no = no_trial_in_each_task
            dict_per_repeat = {}
            dict_per_repeat['trajectory'], dict_per_repeat['timings_repeat'], dict_per_repeat['index_make_step'], dict_per_repeat['step_number'], curr_neurons = mc.analyse.analyse_ephys.prep_ephys_per_trial(timings_all, locations_all, run_no, task_no, task_config, neurons)
                    
            if continuous == True:
                location_model, phase_model, state_model, midnight_model, clocks_model, phase_state_model = mc.simulation.predictions.set_continous_models_ephys(dict_per_repeat, no_phase_neurons= number_phase_neurons)
            
                # can delete but nice to have in case I'm interested in plotting
                # plt.figure(); 
                # plt.imshow(location_model, aspect='auto')
                # plt.figure(); 
                # plt.imshow(phase_model, aspect='auto')
                # plt.figure();  
                # plt.imshow(state_model, aspect='auto')
                # plt.figure()
                # plt.imshow(midnight_model, aspect='auto')
                # plt.figure(); 
                # plt.imshow(phase_state_model, aspect='auto')
                # plt.figure(); 
                # plt.imshow(clocks_model, aspect='auto')
                # for row in range(0, len(clocks_model), len(phase_state_model)):
                #     plt.axhline(row, color='white', ls='dashed')
            
            else:
                location_model = mc.simulation.predictions.set_location_raw_ephys(dict_per_repeat['trajectory'], step_time = 1, grid_size=3, plotting = False, field_no_given= 1)
                midnight_model, clocks_model, midnight_two = mc.simulation.predictions.set_clocks_raw_ephys(dict_per_repeat['trajectory'], dict_per_repeat['timings_repeat'], dict_per_repeat['step_number'], field_no_given= 1, plotting=False)
                phase_model = mc.simulation.predictions.set_phase_model_ephys(dict_per_repeat['trajectory'], dict_per_repeat['timings_repeat'], dict_per_repeat['index_make_step'], dict_per_repeat['step_number'])

            # then bin the data if wanted
            if no_bins_per_state > 0:
                # first generate regressors per phase
                regs_phase_state_run = mc.simulation.predictions.create_x_regressors_per_state(dict_per_repeat['trajectory'], dict_per_repeat['timings_repeat'], dict_per_repeat['step_number'], no_regs_per_state = no_bins_per_state)
                # then use these regressors to timebin
                curr_neurons = mc.simulation.predictions.transform_data_to_betas(curr_neurons, regs_phase_state_run)
                clocks_model = mc.simulation.predictions.transform_data_to_betas(clocks_model, regs_phase_state_run)
                midnight_model= mc.simulation.predictions.transform_data_to_betas(midnight_model, regs_phase_state_run)
                location_model = mc.simulation.predictions.transform_data_to_betas(location_model, regs_phase_state_run)
                phase_model = mc.simulation.predictions.transform_data_to_betas(phase_model, regs_phase_state_run)
                state_model = mc.simulation.predictions.transform_data_to_betas(state_model, regs_phase_state_run)
     
                
            # these need to be concatenated for each run and task
            if task_no == 0:
            #if task_no == 0 and trial_no == 0:
                neurons_between = curr_neurons.copy()
                clocks_between = clocks_model.copy()
                midnight_between = midnight_model.copy()
                location_between = location_model.copy()
                phase_between = phase_model.copy()
                state_between = state_model.copy()
                # # check if I want to split by phase.
                if split_by_phase == 1:
                    phase_separation = mc.simulation.predictions.set_phase_model_ephys(dict_per_repeat['trajectory'], dict_per_repeat['timings_repeat'], dict_per_repeat['index_make_step'], dict_per_repeat['step_number'])
                    phase_separation = np.round(mc.simulation.predictions.transform_data_to_betas( phase_separation, regs_phase_state_run))
                    
            else:
                neurons_between = np.concatenate((neurons_between, curr_neurons), axis = 1)
                clocks_between = np.concatenate((clocks_between, clocks_model), axis = 1)
                midnight_between = np.concatenate((midnight_between, midnight_model), axis = 1)
                location_between = np.concatenate((location_between, location_model), axis = 1)
                phase_between = np.concatenate((phase_between,phase_model), axis = 1)
                state_between = np.concatenate((state_between, state_model), axis = 1)
                # # check if I want to split by phase.
                if split_by_phase == 1:
                    phase_separation_temp = mc.simulation.predictions.set_phase_model_ephys(dict_per_repeat['trajectory'], dict_per_repeat['timings_repeat'], dict_per_repeat['index_make_step'], dict_per_repeat['step_number'])
                    phase_separation_temp = np.round(mc.simulation.predictions.transform_data_to_betas(phase_separation_temp, regs_phase_state_run))
                    phase_separation = np.concatenate((phase_separation, phase_separation_temp), axis = 1)

        
        
        import pdb; pdb.set_trace()
        # then, for the RDMs which concatenate th nth run of each task config, create between-task RDMs
        # now create the model RDMs
        RSM_location = mc.simulation.RDMs.within_task_RDM(location_between, plotting = False, titlestring = 'Location RDM')
        RSM_clock = mc.simulation.RDMs.within_task_RDM(clocks_between, plotting = False, titlestring = 'Clock RDM')
        RSM_midnight = mc.simulation.RDMs.within_task_RDM(midnight_between, plotting = False, titlestring = 'Midnight RDM')
        RSM_phase = mc.simulation.RDMs.within_task_RDM(phase_between, plotting = False, titlestring = 'Phase RDM')
        RSM_state = mc.simulation.RDMs.within_task_RDM(state_between, plotting = False, titlestring= 'State RDM')
        # now create the data RDM
        RSM_neurons = mc.simulation.RDMs.within_task_RDM(neurons_between, plotting = False, titlestring = 'Data RDM')
    
        # Lastly, create a linear regression with RSM_loc,clock and midnight as regressors and data to be predicted
        regressors = {}
        regressors['clocks']=RSM_clock
        regressors['midnight']=RSM_midnight
        regressors['phase']=RSM_phase
        regressors['location']=RSM_location
        regressors['state'] = RSM_state\
            
        
        results_reg = mc.simulation.RDMs.GLM_RDMs(RSM_neurons, regressors, mask_within, no_tasks = len(task_configs), plotting= False)
        
        only_clock_dict = {'clocks': regressors['clocks']}
        only_clocks_reg = mc.simulation.RDMs.GLM_RDMs(RSM_neurons, only_clock_dict, mask_within, no_tasks = len(task_configs), plotting= False)
        
        
        # similarities_kendall = {}
        # for i, curr_RSM_one in enumerate(regressors):
        #     for j, curr_RSM_two in enumerate(regressors):
        #         curr_corr = f"{curr_RSM_one}_with_{curr_RSM_two}"
        #         temp_corr = mc.simulation.RDMs.corr_matrices_kendall(regressors[curr_RSM_one], regressors[curr_RSM_two])
        #         similarities_kendall[curr_corr] = temp_corr.correlation
        
        # similarities_exclude_autocorr = {}
        # for i, curr_RSM_one in enumerate(regressors):
        #     for j, curr_RSM_two in enumerate(regressors):
        #         curr_corr = f"{curr_RSM_one}_with_{curr_RSM_two}"
        #         temp_corr = mc.simulation.RDMs.corr_matrices_pearson(regressors[curr_RSM_one], regressors[curr_RSM_two], no_tasks = task_no, mask_within= True, exclude_diag=True)
        #         similarities_exclude_autocorr[curr_corr] = temp_corr[0,1]
        # import pdb; pdb.set_trace()
        if plotting:
          
            regressors['data'] = RSM_neurons
            for RDM in regressors:
                fig, ax = plt.subplots(figsize=(5,4))
                cmaps.BlueYellowRed
                cmap = plt.get_cmap('BlueYellowRed')
                # Set the upper triangle to be empty
                corr_mat = regressors[RDM]
                corr_mat[np.triu_indices(280, k=1)] = np.nan
                im = ax.imshow(corr_mat, cmap=cmap, interpolation = 'none', aspect = 'equal', vmin=-1, vmax=1); 
                for i in range(39,280,40):
                    ax.axhline(i, color='white', linewidth=1)
                    ax.axvline(i, color='white', linewidth=1)
                    
                # #Add a colorbar to the right of the plot with a colormap toolbox
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes("right", size="5%", pad=0.1)
                # cbar = fig.colorbar(im, cax=cax)
                # cbar.set_label('Correlation', rotation=270, labelpad=15)
                
                # Set x-axis and y-axis ticks and labels
                ticks = np.arange(20, 281, 40)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_xticklabels(['Task {}'.format(i // 40 + 1) for i in ticks], rotation=45, ha = 'right', fontsize=16)
                ax.set_yticklabels(['Task {}'.format(i // 40 + 1) for i in ticks], fontsize=16)
                
                # Set axis labels and title
                ax.set_title(f"Model RDM for {RDM} model", fontsize=18)
                
                # Adjust the appearance of ticks and grid lines
                ax.grid(False)
                cbar = ax.figure.colorbar(im, ax=ax)
                cbar.ax.set_ylabel("Pearson's r", rotation=-90, va="bottom")
                
                # Adjust the layout to prevent cutoff of labels and colorbar
                plt.tight_layout()
                fig.savefig(f"/Users/xpsy1114/Documents/projects/multiple_clocks/output/Model_RDM_{RDM}_between_tasks_1mouse.png", dpi=300, bbox_inches='tight')
                fig.savefig(f"/Users/xpsy1114/Documents/projects/multiple_clocks/output/Model_RDM_{RDM}_between_tasks_1mouse.tiff", dpi=300, bbox_inches='tight')
                
            # in the end, remove data from the dict again!
            del regressors['data']
                
               
                
        # import pdb; pdb.set_trace()
        
        sim_exclude_autocorr_ephys = {}
        for i, curr_RSM_one in enumerate(regressors):
            for j, curr_RSM_two in enumerate(regressors):
                curr_corr = f"{curr_RSM_one}_with_{curr_RSM_two}"
                temp_corr = mc.simulation.RDMs.corr_matrices_pearson(regressors[curr_RSM_one], regressors[curr_RSM_two], no_tasks = task_no, mask_within= True, exclude_diag=True)
                sim_exclude_autocorr_ephys[curr_corr] = temp_corr[0,1]
                
                
        
        #results_reg, tvals = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons, regressor_one_matrix=RSM_clock, regressor_two_matrix= RSM_midnight, regressor_three_matrix= RSM_location, regressor_four_matrix= RSM_phase, t_val= 1)
        # print(f" The beta for the clocks model is {reg_res.coef_[0]}, for the midnight model is {reg_res.coef_[1]}, and for the location model is {reg_res.coef_[2]}")
        tvals_per_trial[no_trial_in_each_task]= results_reg['t_vals']
        coefficients_per_trial[no_trial_in_each_task] = results_reg['coefs']
        labels_regs.append(results_reg['label_regs'])
        

        tvals_per_trial_only_cl[no_trial_in_each_task]= only_clocks_reg['t_vals']
        coefficients_per_trial_only_clo[no_trial_in_each_task] = only_clocks_reg['coefs']

        
        # print(f" Computed betas for run {trial_no} of task {task_config}")
        
        # then compute contrasts
        # I want to know: [0 0 1], [0 1 0], [1 0 0] and [-1 1 0], [0 -1 1], ....
    
        # # in case I want to have an overview of all betas for this trial config
        # x = np.linspace(0,len(coefficients_per_trial)-1,len(coefficients_per_trial))
        # plt.figure(); plt.plot(x, coefficients_per_trial[:,0], label = 'clocks'); plt.plot(x, coefficients_per_trial[:,1], label = 'midnight'); plt.plot(x, coefficients_per_trial[:,2], label = 'location'); plt.plot(x, coefficients_per_trial[:,3], label = 'phase'); plt.legend(loc="upper left"); plt.ylabel('beta'); plt.xlabel('run number'); plt.axhline(0, color='grey', ls='dashed'); plt.title(f"Recording day {mouse_recday} task {task_no}")
         
        for contrast_no, contrast in enumerate(contrast_m):
            contrast_results[no_trial_in_each_task, contrast_no] = np.matmul(contrast,coefficients_per_trial[no_trial_in_each_task].transpose())
              
            
        
        if split_by_phase == 1:
            # do the same thing but per phase by splitting everything into the phase-mask-defined bins first
            label_regs_split = []
            for no_phase, phase in enumerate(phase_split):
                currphase_mask = np.where(phase_separation[no_phase,:] == 1)[0]
                # prepare the re-ordered matrices.
                if no_phase == 0:
                    reordered_neurons = neurons_between[:,currphase_mask].copy()
                    # take out location temporarily
                    # reordered_location = location_between[:, currphase_mask].copy()
                    reordered_clocks = clocks_between[:,currphase_mask].copy()
                    reordered_midnight = midnight_between[:,currphase_mask].copy()
                elif no_phase > 0:
                    reordered_neurons = np.concatenate((reordered_neurons, neurons_between[:,currphase_mask]), axis = 1)
                    # take out location temporarily
                    # reordered_location = np.concatenate((reordered_location, location_between[:, currphase_mask]), axis = 1)
                    reordered_clocks = np.concatenate((reordered_clocks, clocks_between[:,currphase_mask]), axis =1 )
                    reordered_midnight = np.concatenate((reordered_midnight, midnight_between[:,currphase_mask]), axis = 1)
                # take out location temporarily
                # RSM_location_currphase = mc.simulation.RDMs.within_task_RDM(location_between[:, currphase_mask], plotting = False, titlestring = f"{phase} Location RDM")
                RSM_clocks_currphase = mc.simulation.RDMs.within_task_RDM(clocks_between[:,currphase_mask], plotting = False, titlestring = f"{phase} Clock RDM")
                RSM_midnight_currphase = mc.simulation.RDMs.within_task_RDM(midnight_between[:,currphase_mask], plotting = False, titlestring = f"{phase} Midnight RDM")
                # I cannot include phase here bc it's nearly 1. It would mess up my regression weights.
                # RSM_phase_currphase = mc.simulation.RDMs.within_task_RDM(phase_between[:,currphase_mask], plotting = False, titlestring = f"{phase} Phase RDM")
                # now create the data RDM
                RSM_neurons_currphase = mc.simulation.RDMs.within_task_RDM(neurons_between[:,currphase_mask], plotting = False, titlestring = f"{phase} Data RDM") 
                
                regs_currphase = {}
                regs_currphase['clocks']= RSM_clocks_currphase
                regs_currphase['midnight']= RSM_midnight_currphase
                # I cannot include phase here bc it's nearly 1. It would mess up my regression weights.
                # regs_currphase['phase']= RSM_phase_currphase
                # take out location temporarily
                # regs_currphase['location']= RSM_location_currphase
                results_reg_currphase = mc.simulation.RDMs.GLM_RDMs(RSM_neurons_currphase, regs_currphase, mask_within, no_tasks = len(task_configs), plotting= False)
                # import pdb; pdb.set_trace()
                tvals_per_trial_split[no_phase, no_trial_in_each_task, :]= results_reg_currphase['t_vals']
                coefficients_per_trial_split[no_phase,no_trial_in_each_task,:] = results_reg_currphase['coefs']
                label_regs_split.append(results_reg_currphase["label_regs"])
                
                for contrast_no, contrast in enumerate(contrast_split):
                    contrast_results_split[no_phase, no_trial_in_each_task, contrast_no] = np.matmul(contrast,coefficients_per_trial_split[no_phase, no_trial_in_each_task,:].transpose())
            
            # reordered regression
            # take out location temporarily
            # RSM_reordered_loc = mc.simulation.RDMs.within_task_RDM(reordered_location, plotting= False, titlestring= "reordered Location")
            RSM_reordered_neurons = mc.simulation.RDMs.within_task_RDM(reordered_neurons, plotting= False, titlestring= "reordered neurons")
            RSM_reordered_clocks = mc.simulation.RDMs.within_task_RDM(reordered_clocks, plotting= False, titlestring= "reordered clocks")
            RSM_reordered_midnight = mc.simulation.RDMs.within_task_RDM(reordered_midnight, plotting= False, titlestring= "reordered midnight")
            
            regs_reordered = {}
            regs_reordered['clocks'] = RSM_reordered_clocks
            regs_reordered['midnight'] = RSM_reordered_midnight
            # take out location temporarily
            # regs_reordered['location'] = RSM_reordered_loc
            
            # mask within false because the reordering screws with the within-task order
            results_reg_reordered = mc.simulation.RDMs.GLM_RDMs(RSM_reordered_neurons, regs_reordered, mask_within = False, no_tasks = len(task_configs), plotting= False)
            tval_reor[no_trial_in_each_task] = results_reg_reordered['t_vals']
            coef_reor[no_trial_in_each_task] = results_reg_reordered['coefs']
            
            
            for contrast_no, contrast in enumerate(contrast_split):
                contr_reor[no_trial_in_each_task, contrast_no] = np.matmul(contrast,coef_reor[no_trial_in_each_task,:].transpose())
                
        print(f"done with trial_no {no_trial_in_each_task}")           
    # import pdb; pdb.set_trace()        
    result = {}
    result["coefficients_per_trial"] = coefficients_per_trial
    result["contrast_results"] = contrast_results
    result["t-values"] = tvals_per_trial
    result["labels"] = labels_regs
    result["t-vals_only_clock"] = tvals_per_trial_only_cl
    result["coeffs_only_clock"] = coefficients_per_trial_only_clo


    if split_by_phase == True:
        result["split_coef_per_trial"] = coefficients_per_trial_split
        result["split_contrasts"] = contrast_results_split
        result["split_t-vals"] = tvals_per_trial_split
        result["split_labels"] = label_regs_split
        result["reord_coefs"] = coef_reor
        result["reord_t-vals"] = tval_reor
        result["reord_contrasts"] = contr_reor
 
    return result



# with this one, I want to compare different models and find the optimal output for my analysis.
def reg_across_tasks(task_configs, locations_all, neurons, timings_all, mouse_recday, plotting = False, continuous = True, no_bins_per_state = 3, number_phase_neurons = 3, mask_within = True, split_by_phase = True):
    #import pdb; pdb.set_trace()
    # this is now all  based on creating an average across runs first.   
    # find out which is the largest shared trial number between all task configs
    min_trialno = 60
    for task_number in timings_all:
        curr_trialno = len(task_number)
        if curr_trialno < min_trialno:
            min_trialno = curr_trialno
    
    # goal is to compare between grids, so set this up before to not overwrite.
    sim_models_between_grids_dict = {}
    sum_all_repeats_between_grids = {}
    for repeat_no in range(0, min_trialno):
        # stop when the max trial number is reached, such that we only consider the same amount of repeat per task
        # take the first, second, ... nth repeat
        for task_no, task_config in enumerate(task_configs):
            # do this for each task config and 
            # concatenate the first, second, ... nth repeat of each task config.
            # (but start from the end to take the final runs not the first ones)
            run_no = -1*(repeat_no + 1)
            dict_per_repeat, curr_neurons = mc.analyse.analyse_ephys.prep_ephys_per_trial(timings_all, locations_all, run_no, task_no, task_config, neurons)

            if continuous == False: 
                location_model = mc.simulation.predictions.set_location_raw_ephys(dict_per_repeat['trajectory'], step_time = 1, grid_size=3, plotting = False, field_no_given= 1)
                # i can choose which non-continuous model I want.
                # possible outputs are: midnight_matrix, full_clock_matrix, alternative_midnight, alternative_clock, compromise_midnight, compromise_clock
                # compromise models: consecutive activation, but on for the entire phase.
                midnight_model_part, clocks_model_fullphase, midnight_full, clocks_part, midnight_model, clocks_model = mc.simulation.predictions.set_clocks_raw_ephys(dict_per_repeat['trajectory'], dict_per_repeat['timings_repeat'], dict_per_repeat['index_make_step'], dict_per_repeat['step_number'], field_no_given= 1, plotting=False)
                # full phase models 
                # midnight_model_part, midnight_model, clocks_model, clocks_part, compromise_midnight, compromise_clock = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=False)
                # both with gaps
                # midnight_model, clocks_model_fullphase, midnight_full, clocks_model, compromise_midnight, compromise_clock = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=False)
                # what I originally looked at:
                # midnight_model, clocks_model, alternative_midnight, alternative_clock, compromise_midnight, compromise_clock = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=False)
                #midnight_model, clocks_model = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=False)
                phase_model = mc.simulation.predictions.set_phase_model_ephys(dict_per_repeat['trajectory'], dict_per_repeat['timings_repeat'], dict_per_repeat['index_make_step'], dict_per_repeat['step_number'])
            
            if continuous == True:
                model_dict = mc.simulation.predictions.set_continous_models_ephys(dict_per_repeat, no_phase_neurons= number_phase_neurons, plot = False)
                #location_model, phase_model, state_model, midnight_model, clocks_model, phase_state_model = mc.simulation.predictions.set_continous_models_ephys(trajectory, timings_curr_run, index_make_step, step_number, no_phase_neurons= number_phase_neurons, plot = False)
                     
            model_dict['curr_neurons'] = curr_neurons.copy()
            # now create the regressors per run
            regs_phase_state_run = mc.simulation.predictions.create_x_regressors_per_state(walked_path = dict_per_repeat['trajectory'], subpath_timings = dict_per_repeat['timings_repeat'], step_no = dict_per_repeat['step_number'], no_regs_per_state = no_bins_per_state)
            # then use these regressors to generate a beta per neuron per run
            
            beta_sim_models_dict = {}
            for model in sorted(model_dict):
                beta_sim_models_dict[model] = mc.simulation.predictions.transform_data_to_betas(model_dict[model], regs_phase_state_run)
                # these need to be concatenated for each run and task
                if task_no == 0:
                    sim_models_between_grids_dict[model] = beta_sim_models_dict[model].copy()
                else:
                    sim_models_between_grids_dict[model] = np.concatenate((sim_models_between_grids_dict[model], beta_sim_models_dict[model]), axis = 1)
            if split_by_phase:       
                if task_no == 0:
                    phase_string = ['early', 'mid', 'late']
                    phase_separation = mc.simulation.predictions.set_phase_model_ephys(dict_per_repeat['trajectory'], dict_per_repeat['timings_repeat'], dict_per_repeat['index_make_step'], dict_per_repeat['step_number'])
                    phase_separation = np.round(mc.simulation.predictions.transform_data_to_betas(phase_separation, regs_phase_state_run))
                else:
                    phase_separation_temp = mc.simulation.predictions.set_phase_model_ephys(dict_per_repeat['trajectory'], dict_per_repeat['timings_repeat'], dict_per_repeat['index_make_step'], dict_per_repeat['step_number'])
                    phase_separation_temp = np.round(mc.simulation.predictions.transform_data_to_betas(phase_separation_temp, regs_phase_state_run))
                    phase_separation = np.concatenate((phase_separation, phase_separation_temp), axis = 1)
    
        # once I concatenated all nth repeat across each grid, create an average.
        
        if repeat_no == 0:
            for model in sorted(sim_models_between_grids_dict):
                sum_all_repeats_between_grids[model] = sim_models_between_grids_dict[model].copy()
            if split_by_phase:
                sum_phase_separation = phase_separation.copy()
                
        if repeat_no > 0:
            for model in sorted(sim_models_between_grids_dict):
                sum_all_repeats_between_grids[model] = sum_all_repeats_between_grids[model].copy() + sim_models_between_grids_dict[model].copy()
            if split_by_phase:
                sum_phase_separation = sum_phase_separation.copy() + phase_separation.copy()
        
    # once all repeats are added, divide by no of repeats
    ave_models_between = {}
    for model in sorted(sum_all_repeats_between_grids):
        ave_models_between[model] = sum_all_repeats_between_grids[model]/min_trialno
    
    print(f"last repeat was {repeat_no} and I divided by {min_trialno}")
    
    if split_by_phase:
        ave_phase_separation = sum_phase_separation/min_trialno
        max_val = np.max(ave_phase_separation)
        # import pdb; pdb.set_trace()
        phase_masks = {}
        for i, phase in enumerate(phase_string):
            # early = 0; mid = 1; late = 2
            phase_masks[phase] = np.where(ave_phase_separation[i,:] == max_val)[0]


    # import pdb; pdb.set_trace()
    RDM_dict = {}
    if plotting == True:
        #import pdb; pdb.set_trace()
        # plot the averaged simulated and cleaned data
        for model in ave_models_between:
            mc.simulation.predictions.plot_without_legends(ave_models_between[model], titlestring= f"{model} model, averaged across runs for single mouse", intervalline= 4*no_bins_per_state, saving_file='/Users/xpsy1114/Documents/projects/multiple_clocks/output/')
            RDM_dict[model] = mc.simulation.RDMs.within_task_RDM(ave_models_between[model], plotting = True, titlestring = f"Between tasks {model} RSM, 12*12, averaged over runs", intervalline= 4*no_bins_per_state)
        mc.simulation.RDMs.plot_RDMs(RDM_dict, len(task_configs))       
        # separately per phase
        if split_by_phase:
            RDM_dict_phases = {}
            for model in ave_models_between:
                RDM_dict_phases[model] = {}
                # ceither all models or fine-tune this to only those I want by defining a string of models I loop through
                for phase in phase_string:
                    mc.simulation.predictions.plot_without_legends(ave_models_between[model][:, phase_masks[phase]], titlestring=f"{phase} {model} across tasks", intervalline= 4*no_bins_per_state/3)
                    RDM_dict_phases[model][phase] = mc.simulation.RDMs.within_task_RDM(ave_models_between[model][:, phase_masks[phase]], plotting=True, titlestring=f"RSM {phase} {model}, averaged over runs", intervalline= 4*no_bins_per_state/3)
    
    elif plotting == False: 
        # for all phases
        for model in ave_models_between:
            RDM_dict[model] = mc.simulation.RDMs.within_task_RDM(ave_models_between[model], plotting = False)
        if split_by_phase:
            RDM_dict_phases = {}
            for model in ave_models_between:
                # for each phase separately
                RDM_dict_phases[model] = {}
                for phase in phase_string:
                    RDM_dict_phases[model][phase] = mc.simulation.RDMs.within_task_RDM(ave_models_between[model][:, phase_masks[phase]], plotting = False)
                
    # run regressions separetly for each phase
    # import pdb; pdb.set_trace()
    # try the new regression.
    
    
    # midn_model, phas_model, loc_model, stat_model, phas_stat, clo_model, curr_neurons
    regressors_to_include = ['clo_model', 'phas_model', 'loc_model', 'stat_model'] #INCLUDE MIDNIGHT AGAIN AT SOME POINT!!
    regressors = {}
    for i, model in enumerate(sorted(regressors_to_include)):
        print(f"the order of the regressors here is at {i} comes {model}")
        regressors[model] = RDM_dict[model].copy()

    results_normal = mc.simulation.RDMs.GLM_RDMs(RDM_dict['curr_neurons'], regressors, mask_within, no_tasks = len(task_configs), plotting= False)
    # all_results[mouse_res]['normal']['t_vals'][1]
    
    # collect all values in a results table which I then output in the end.
    result_dict = {}
    result_dict['normal']= results_normal.copy()
    if split_by_phase:
        regressors_phases = {}
        results_phase = {}
        for phase in phase_string:
            regressors_phases[phase] = {} 
            for model in regressors_to_include:
                regressors_phases[phase][model] = RDM_dict_phases[model][phase].copy()
        for phase in phase_string:
            results_phase[phase] = mc.simulation.RDMs.GLM_RDMs(RDM_dict_phases['curr_neurons'][phase], regressors_phases[phase], mask_within, no_tasks = len(task_configs))
        # this isn't the most elegant solution, also doesn't currently fit with all
        # phase split-stuff that comes after.
        # keep in mind!
        # adjust if needed - 20th of jan 2025
        result_dict['phases'] = results_phase.copy()
    return result_dict



def load_ephys_data(dict_labels, Data_folder):
    # import pdb; pdb.set_trace()
    data = {}
    rec_days = ['me11_05122021_06122021', 'me11_01122021_02122021', 'me10_09122021_10122021', 'me08_10092021_11092021', 
                'ah04_09122021_10122021', 'ah04_05122021_06122021', 'ah04_01122021_02122021', 'ah04_01122021_02122021' ,
                'ah03_18082021_19082021']
    for mouse in dict_labels:
        data[mouse] = {}
        
    for i, mouse in enumerate(data):
        mouse_recday=rec_days[i]
        data[mouse]["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
        if mouse == 'mouse_d':
            data[mouse]["rewards_configs"] = data[mouse]["rewards_configs"][0:-1, :].copy()
            # apparently there is one run less for this day..., so exclude that one
            # mohammady says: The ephys file for the last task on that day was lost
        data[mouse]["anchor_lag"]= np.load(Data_folder+'Anchor_lag_'+mouse_recday+'.npy')
        data[mouse]["anchor_lag_threshold"] = np.load(Data_folder+'Anchor_lag_threshold_'+mouse_recday+'.npy')
        no_task_configs = len(data[mouse]["rewards_configs"])
        data[mouse]["cells"] = np.load(Data_folder+'Phase_state_place_anchored_' + mouse_recday + '.npy')
        locations, neurons, timings = [], [], []
        for session in range(0, no_task_configs):
            locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
            neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
            timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))
        
        #import pdb; pdb.set_trace()
        data[mouse]["locations"] = locations
        data[mouse]["neurons"] = neurons
        data[mouse]["timings"] = timings
        data[mouse]["recday"] = mouse_recday
        
        data[mouse]["neuron_type"] = np.zeros((len(data[mouse]["anchor_lag"]), len(data[mouse]["anchor_lag"][1])))
        for i, neuron in enumerate(data[mouse]["anchor_lag"]):
            max_neuron = np.argmax(neuron)
            data[mouse]["neuron_type"][i, max_neuron] = 1
    return(data)



def clean_ephys_data(task_configs, locations_all, neurons, timings_all, mouse_recday, ignore_double_tasks = 1):
    # first clean data.
    # import pdb; pdb.set_trace()
    # load dataset
    # first thing: check for missing data. mark those and potentially ignore.

    # throw out those with significantly lower run numbers  
    too_short = []
    max_length = len(neurons[0][0])
    for task_config_no, run in enumerate(locations_all):
        # first find max
        if max_length < len(run):
            max_length = len(run)
    for task_config_no, run in enumerate(locations_all):
        # then compare the others against max
        if len(run)< max_length/3:
            too_short.append(task_config_no)
    
    ignore = too_short
    
    if ignore_double_tasks == 1:
        # mark repeated tasks
        repeated_tasks = []
        for count, curr_task in enumerate(task_configs):
            comp_task_configs = np.concatenate((task_configs[:count], task_configs[count+1:]))
            for task_no, task in enumerate(comp_task_configs):
                if all(x == y for x, y in zip(curr_task, task)):
                    repeated_tasks.append(count)
                    # print(curr_task, task, count)
        
        # check which of the repeated tasks is the worse one
        drop_repeats = []
        for i, task in enumerate(repeated_tasks):
            comp_rep_tasks = repeated_tasks[:i]+repeated_tasks[i+1:]
            for comp_task in comp_rep_tasks:
                if all(x == y for x, y in zip(task_configs[task], task_configs[comp_task])): 
                    if len(timings_all[task])<len(timings_all[comp_task]):
                        drop_repeats.append(task)
                    elif len(timings_all[task])>len(timings_all[comp_task]):
                        if comp_task not in drop_repeats:
                            drop_repeats.append(comp_task)
    
        # finally, compare values in drop_repeats and repeated_tasks
        ignore = drop_repeats + too_short
        ignore_list = []
        for i in set(ignore):
            ignore_list.append(i)
        ignore_list.sort(reverse= True)

    task_configs_clean = [elem for elem in task_configs]
    locations_all_clean = locations_all.copy()
    neurons_clean = neurons.copy()
    timings_all_clean = timings_all.copy()
    
    for ignore_task in ignore_list:
        task_configs_clean.pop(ignore_task)
        locations_all_clean.pop(ignore_task)
        neurons_clean.pop(ignore_task)
        timings_all_clean.pop(ignore_task)
        
    return(task_configs_clean,locations_all_clean,neurons_clean,timings_all_clean)




def prep_ephys_per_trial(timings_all, locations_all, no_trial_in_each_task, task_no, task_config, neurons):
    # first convert trial times from ms to bin number to match neuron and location arrays 
    # (1 bin = 25ms)
    # import pdb; pdb.set_trace()
    timings_task = timings_all[task_no].copy()
    for r, row in enumerate(timings_task):
        for c, element in enumerate(row):
            timings_task[r,c] = element/25

    # second, change locations and rewards to 0 and ignoring bridges
    locations_task = locations_all[task_no].copy()
    for i, field in enumerate(locations_task):
        if field > 9: 
            locations_task[i] = locations_task[i-1].copy()
        if math.isnan(field):
            # keep the location bc of timebins
            locations_task[i] = locations_task[i-1].copy()
                

    # important: fields need to be between 0 and 8, and keep them as integers!
    locations_task = [int((field_no-1)) for field_no in locations_task]
    task_config = [int((field_no-1)) for field_no in task_config]

    # now select the run you are currently focussing on.
    row = timings_task[no_trial_in_each_task]
    # current data of this specific run
    trajectory = locations_task[row[0]:row[-1]].copy()

    curr_neurons = neurons[task_no][:,row[0]:row[-1]].copy()
    # z-score neurons
    curr_neurons = scipy.stats.zscore(curr_neurons, axis=1)
    
    # ISSUE 21.04.23:
    # if there are ONLY 0 for one timestep, the np.corrcoef will output nan for that instance. Maybe better:
    # replace by super super low value
    # Update 9th of May: DONT DO THAT!
    # better: ignore those rows/ values in the correlation
    # for col_no, column in enumerate(curr_neurons.T):
    #     if np.all(column == 0):
    #         curr_neurons[:,col_no] = 0.00001

    # some pre-processing to create my models.
    # to count subpaths
    subpath_file = [locations_task[row[0]:row[1]], locations_task[row[1]:row[2]], locations_task[row[2]:row[3]], locations_task[row[3]:row[4]]]
    timings_curr_run = [(elem - row[0]) for elem in row]

    # to find out the step number per subpath
    step_number = [0,0,0,0] 
    for path_no, subpath in enumerate(subpath_file):
        for i, field in enumerate(subpath):
            if i == 0:
                count = 0
            elif field != subpath[i-1]:
                count+=1
        step_number[path_no] = count
       
    # mark where steps are made
    for field_no, field in enumerate(trajectory):
        if field_no == 0:
            index_make_step = [0]
        elif field != trajectory[field_no-1]:
            index_make_step.append(field_no)
    prep_behaviour_dict = {}
    prep_behaviour_dict['trajectory'] = trajectory
    prep_behaviour_dict['timings_repeat'] = timings_curr_run
    prep_behaviour_dict['index_make_step']=  index_make_step
    prep_behaviour_dict['step_number'] = step_number
    
    return(prep_behaviour_dict, curr_neurons)



############################
### PLOTTING FUNCTIONS #####
############################
import colormaps as cmaps


def plotting_hist_scat(data_list, label_string_list, label_tick_list, title_string, save_fig = False):
    # import pdb; pdb.set_trace()

    fig, ax = plt.subplots(figsize=(8,6))
    ax.boxplot(data_list, medianprops=dict(color='black'))
    
    sigma = 0.08
    mu = 0.01
    # cmap = plt.get_cmap('Pastel1')
    # colors = cmap(np.linspace(0, 1, len(data_list)))
    # Custom colors
    colors = ['#96C5D8'] + ['#882048'] * (len(data_list) - 1)  # First color for the first group, the rest in another color

    global_min = 0
    global_max = 0
    
    for index, contrast in enumerate(data_list):
        noise = sigma * np.random.randn(len(contrast)) + mu
        data_to_plot = np.array(contrast)  # Ensure data_to_plot is an array for direct operations
        x_positions = index + 1 + noise  # Adjust index for boxplot's 1-based index
        if np.min(data_to_plot) < global_min:
            global_min = np.min(data_to_plot)
        if np.max(data_to_plot) > global_max:
            global_max = np.max(data_to_plot)
        ax.scatter(x_positions, data_to_plot, color=colors[index], marker='o', s=100, edgecolors='black', linewidth=1)
    
    
    ax.set_xticks(label_tick_list)
    #ax.set_xticks(range(1, len(data_list) + 1))
    plt.xticks(rotation=45)
    #ax.set_xticklabels(label_string_list, fontsize=26)
    ax.set_xticklabels(label_string_list)
    ax.set_ylabel('Betas')
    
    padding = global_max/10
    
    ax.set_ylim([global_min - padding*4, global_max+ padding])
    
    plt.axhline(0, color='grey', ls='dashed', linewidth=1)
    
    # Add statistical significance stars
    for i, model in enumerate(data_list):
        t_statistic, p_value = ttest_1samp(model, 0, alternative='greater')
        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.005:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = ''
        if significance:
            ax.text(i + 1, global_min - padding*4, significance, ha='center', va='bottom', fontsize=30, color='black')
    
    # # check if these are better than 0.
    # from scipy.stats import ttest_1samp
    # for i, model in enumerate(data_list):
    #     t_statistic, p_value = ttest_1samp(model, 0, alternative='greater')
    #     # Output the results
    #     print(f'T-statistic: {t_statistic} for {label_string_list[i]}')
    #     print(f'P-value: {p_value} for {label_string_list[i]}')

    
    plt.title(title_string, pad = 30)
    plt.rcParams.update({'font.size': 30})
    # Customize grid lines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust the layout
    plt.tight_layout()
    plt.show()


    if save_fig:
        fig.savefig(f"{save_fig}{title_string}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{save_fig}{title_string}.tiff", dpi=300, bbox_inches='tight')
    
