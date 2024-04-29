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
            
            trajectory, timings_curr_run, index_make_step, step_number, curr_neurons = mc.analyse.analyse_ephys.prep_ephys_per_trial(timings_all, locations_all, run_no, task_no, task_config, neurons)
                    
            if continuous == True:
                location_model, phase_model, state_model, midnight_model, clocks_model, phase_state_model = mc.simulation.predictions.set_continous_models_ephys(trajectory, timings_curr_run, index_make_step, step_number, no_phase_neurons= number_phase_neurons)
            
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
                    phase_separation = mc.simulation.predictions.set_phase_model_ephys(trajectory, timings_curr_run, index_make_step, step_number)
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
                    phase_separation_temp = mc.simulation.predictions.set_phase_model_ephys(trajectory, timings_curr_run, index_make_step, step_number)
                    phase_separation_temp = np.round(mc.simulation.predictions.transform_data_to_betas(phase_separation_temp, regs_phase_state_run))
                    phase_separation = np.concatenate((phase_separation, phase_separation_temp), axis = 1)

        
        
        # import pdb; pdb.set_trace()
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



# # This is to play around with the data.
# def reg_across_tasks_playground(task_configs, locations_all, neurons, timings_all, mouse_recday, continuous = True):
#     # import pdb; pdb.set_trace()
#     coefficient = list()
    
#     # find out which is the largest shared trial number between all task configs
#     min_trialno = 60
#     for task_number in timings_all:
#         curr_trialno = len(task_number)
#         if curr_trialno < min_trialno:
#             min_trialno = curr_trialno
     
#     # bit ugly that 4 is hard-coded, this is just the number of regressors, i.e. number of RDMs/models
#     # I am checking all trials across all tasks. because they have different amounts of trials, take the smalles shared minimum
#     coefficients_per_trial = np.zeros((min_trialno,4))
    
#     for no_trial_in_each_task in range(0, min_trialno):
#         for task_no, task_config in enumerate(task_configs):
#             # Some_configs are the same!! e.g. mouse b: 0 and 3 are the same
#             # first convert trial times from ms to bin number to match neuron and location arrays 
#             # (1 bin = 25ms)
#             timings_task = timings_all[task_no].copy()
#             for r, row in enumerate(timings_task):
#                 for c, element in enumerate(row):
#                     timings_task[r,c] = element/25
        
#             # second, change locations and rewards to 0 and ignoring bridges
#             locations_task = locations_all[task_no].copy()
#             for i, field in enumerate(locations_task):
#                 if field > 9: 
#                     locations_task[i] = locations_task[i-1]
#                 if math.isnan(field):
#                     # keep the location bc of timebins
#                     locations_task[i] = locations_task[i-1]
                        
        
#             # important: fields need to be between 0 and 8, and keep them as integers!
#             locations_task = [int((field_no-1)) for field_no in locations_task]
#             task_config = [int((field_no-1)) for field_no in task_config]
    
#             # start of the second loop: loop through every single trial for this specific task config.
#             # FOR NOW, ONLY DO THIS ON THE LAST 3 TRIALS PER TASK!!
#             # timings_trials_I_take = timings_task[-3:]
#             # new try: only do this for the last trial per task
#             # timings_trials_I_take = timings_task[-1:]
#             # new try: take all, but always only one at a time >
#             # find out the max shared trial number across tasks!
            
#             timings_trials_I_take = timings_task[-(no_trial_in_each_task+1),:].copy()
            
#             # THIS IS PROBLEMATIC NOW THAT I TAKE ONLY ONE ROW
#             # change temporarily
#             #for trial_no, row in enumerate(timings_trials_I_take):
#                 # CHANGE BACK IF I TAKE SEVERAL TRIALS PER REGRESSION!!   
#             row = timings_trials_I_take.copy()
#             # current data of this specific run
#             trajectory = locations_task[row[0]:row[-1]].copy()
#             curr_neurons = neurons[task_no][:,row[0]:row[-1]].copy()
#             # ISSUE 21.04.23:
#             # if there are ONLY 0 for one timestep, the np.corrcoef will output nan for that instance. Maybe better:
#             # replace by super super low value
#             # Update 09.05.23: DON'T Do that! Rather ignore those values in correlation
#             # for col_no, column in enumerate(curr_neurons.T):
#             #     if np.all(column == 0):
#             #         curr_neurons[:,col_no] = 0.00001

        
#             # some pre-processing to create my models.
#             # to count subpaths
#             subpath_file = [locations_task[row[0]:row[1]+1], locations_task[row[1]+1:row[2]+1], locations_task[row[2]+1:row[3]+1], locations_task[row[3]+1:row[4]+1]].copy()
#             timings_curr_run = [(elem - row[0]) for elem in row]
        
#             # to find out the step number per subpath
#             step_number = [0,0,0,0] 
#             for path_no, subpath in enumerate(subpath_file):
#                 for i, field in enumerate(subpath):
#                     if i == 0:
#                         count = 0
#                     elif field != subpath[i-1]:
#                         count+=1
#                 step_number[path_no] = count
               
#             # mark where steps are made
#             for field_no, field in enumerate(trajectory):
#                 if field_no == 0:
#                     index_make_step = [0]
#                 elif field != trajectory[field_no-1]:
#                     index_make_step.append(field_no)
            
                    
#             if continuous == False:
#                 location_model = mc.simulation.predictions.set_location_raw_ephys(trajectory, step_time = 1, grid_size=3, plotting = False, field_no_given= 1)
                
#                 # I now output a lot of different clock models.
#                 # CHANGE LATER!!! 
#                 # Here I now try around with which model describes the data best.
#                 # midnight_matrix, full_clock_matrix, alternative_midnight, alternative_clock, compromise_midnight, compromise_clock
#                 midnight_model_part, clocks_model_fullphase, midnight_full, clocks_part, midnight_model, clocks_model = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=False)
                
#                 #midnight_model, clocks_model = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=False)
#                 phase_model = mc.simulation.predictions.set_phase_model_ephys(trajectory, timings_curr_run, index_make_step, step_number)
            
#             if continuous == True:
#                 location_model, phase_model, state_model, midnight_model, clocks_model, phase_state_model = mc.simulation.predictions.set_continous_models_ephys(trajectory, timings_curr_run, index_make_step, step_number)
            
                

#             # now create the regressors per run
#             regs_phase_state_run = mc.simulation.predictions.create_regressors_per_state_phase_ephys(walked_path=trajectory, subpath_timings=timings_curr_run, step_no=step_number)
#             # then use these regressors to generate a beta per neuron per run
#             neurons_phase_state = mc.simulation.predictions.transform_data_to_betas(curr_neurons, regs_phase_state_run)
#             clock_phase_state = mc.simulation.predictions.transform_data_to_betas(clocks_model, regs_phase_state_run)
#             midnight_phase_state= mc.simulation.predictions.transform_data_to_betas(midnight_model, regs_phase_state_run)
#             location_phase_state = mc.simulation.predictions.transform_data_to_betas(location_model, regs_phase_state_run)
#             phase_phase_state = mc.simulation.predictions.transform_data_to_betas(phase_model, regs_phase_state_run)
            
#             # these need to be concatenated for each run and task
#             if task_no == 0:
#             #if task_no == 0 and trial_no == 0:
#                 neurons_between = neurons_phase_state.copy()
#                 clocks_between = clock_phase_state.copy()
#                 midnight_between = midnight_phase_state.copy()
#                 location_between = location_phase_state.copy()
#                 phase_between = phase_phase_state.copy()
#             else:
#                 neurons_between = np.concatenate((neurons_between, neurons_phase_state), axis = 1)
#                 clocks_between = np.concatenate((clocks_between, clock_phase_state), axis = 1)
#                 midnight_between = np.concatenate((midnight_between, midnight_phase_state), axis = 1)
#                 location_between = np.concatenate((location_between, location_phase_state), axis = 1)
#                 phase_between = np.concatenate((phase_between,phase_phase_state), axis = 1)
      
           
#         # try how the stat_phase RDMS look like
#         RSM_location_betas = mc.simulation.RDMs.within_task_RDM(location_between, plotting = False, titlestring = 'Location phase*state dim RSM')
#         RSM_clock_betas = mc.simulation.RDMs.within_task_RDM(clocks_between, plotting = False, titlestring = 'Clock phase*state dim RSM')
#         RSM_midnight_betas = mc.simulation.RDMs.within_task_RDM(midnight_between, plotting = False, titlestring = 'Midnight phase*state dim RSM')
#         RSM_phase_betas = mc.simulation.RDMs.within_task_RDM(phase_between, plotting = False, titlestring = 'Phase phase*state dim RSM')
#         RSM_neurons_betas = mc.simulation.RDMs.within_task_RDM(neurons_between, plotting = False, titlestring = 'Data phase*state dim RDM')
        
#         # import pdb; pdb.set_trace()
        
#         # CLEAN UP FROM HERE!!
#         # WHAT IS MY ACTUAL OUTPUT???
        
#         # UNCOMMENT if you are interested in individual coefficient values per one trial.
#         # results_reg, scipy_regression_results = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_betas, regressor_one_matrix=RSM_clock_betas, regressor_two_matrix= RSM_midnight_betas, regressor_three_matrix= RSM_location_betas, regressor_four_matrix= RSM_phase_betas)
#         # coefficients_per_trial[no_trial_in_each_task] = results_reg.coef_
#         # at the end of the trial, store the whole matrix in coefficient:
#         # coefficient.append(coefficients_per_trial) 
        
#         # additionally, create an averaged RDM. 
#         # it's possible that the mice take different routes every time. By averaging, 
#         # I should be able to account for these variations.
#         if no_trial_in_each_task == 0:
#             sum_location_between = location_between.copy()
#             sum_clocks_between = clocks_between.copy()
#             sum_midnight_between = midnight_between.copy()
#             sum_phase_between = phase_between.copy()
#             sum_neurons_between = neurons_between.copy()
#         if no_trial_in_each_task > 0:
#             sum_location_between = sum_location_between.copy() + location_between.copy()
#             sum_clocks_between = sum_clocks_between.copy() + clocks_between.copy()
#             sum_midnight_between = sum_midnight_between.copy() + midnight_between.copy()
#             sum_phase_between = sum_phase_between.copy() + phase_between.copy()
#             sum_neurons_between = sum_neurons_between.copy() + neurons_between.copy()
        
#     ave_location_between = sum_location_between/no_trial_in_each_task
#     ave_clocks_between = sum_clocks_between/no_trial_in_each_task
#     ave_midnight_between = sum_midnight_between/no_trial_in_each_task
#     ave_phase_between = sum_phase_between/no_trial_in_each_task
#     ave_neurons_between = sum_neurons_between/no_trial_in_each_task

    

#     # clean according to recording day.
    
#     # if mouse_recday == 'me11_05122021_06122021': #mouse a
#     # ALL FINE WITH a!
#         # task 5 and 9 are the same, as well as 6 and 7
#         # data of the first 4 tasks look similar, and tasks 5,6,7,8,9 look more similar
    
#     if mouse_recday == 'me11_01122021_02122021':#mouse b
#         # get rid of the last task because it looks somewhat whacky
#         ave_clocks_between = ave_clocks_between[:,0:-12].copy()
#         ave_phase_between = ave_phase_between[:,0:-12].copy()
#         ave_midnight_between = ave_midnight_between[:,0:-12].copy()
#         ave_location_between = ave_location_between[:,0:-12].copy()
#         ave_neurons_between = ave_neurons_between[:, 0:-12].copy()
        
#     if mouse_recday == 'me10_09122021_10122021': #mouse c 
#         # same tasks are: 1,4; and  5,6,9
#         # 4 and 9 look whacky, so remove those
#         # so then after removal 4 and 5 are the same 
#         ave_clocks_between = np.concatenate((ave_clocks_between[:, 0:36], ave_clocks_between[:, 48:96]), axis = 1)
#         ave_neurons_between = np.concatenate((ave_neurons_between[:, 0:36], ave_neurons_between[:,48:96]), axis = 1)
#         ave_midnight_between = np.concatenate((ave_midnight_between[:, 0:36], ave_midnight_between[:,48:96]), axis = 1)
#         ave_location_between = np.concatenate((ave_location_between[:, 0:36], ave_location_between[:,48:96]), axis = 1)
#         ave_phase_between = np.concatenate((ave_phase_between[:, 0:36], ave_phase_between[:,48:96]), axis = 1)
#         # consider also removing the penultimum one... this was before task 7, now it is 6
#         # so far this is still inside
        
        
#     # if mouse_recday == 'me08_10092021_11092021': #mouse d 
#     # same tasks: 1, 4
#     # ALL FINE WITH d ONCE THE LAST BUT THE LAST EPHYS FILE WAS LOST 
    
#     if mouse_recday == 'ah04_09122021_10122021': #mouse e range 0,8
#     # throw out the 4th 
#     # same tasks: (all tasks are unique, before 1 and 4 were the same but 4 is gone)
#         ave_clocks_between = np.concatenate((ave_clocks_between[:, 0:36], ave_clocks_between[:, 48::]), axis = 1)
#         ave_neurons_between = np.concatenate((ave_neurons_between[:, 0:36], ave_neurons_between[:,48::]), axis = 1)
#         ave_midnight_between = np.concatenate((ave_midnight_between[:, 0:36], ave_midnight_between[:,48::]), axis = 1)
#         ave_location_between = np.concatenate((ave_location_between[:, 0:36], ave_location_between[:,48::]), axis = 1)
#         ave_phase_between = np.concatenate((ave_phase_between[:, 0:36], ave_phase_between[:,48::]), axis = 1)

#     if mouse_recday == 'ah04_05122021_06122021': #mouse f range 0,8
#     # throw out number 4
#     # new 4 (previous 5) and last one - 7 (previous 8) are the same
#         ave_clocks_between = np.concatenate((ave_clocks_between[:, 0:36], ave_clocks_between[:, 48::]), axis = 1)
#         ave_neurons_between = np.concatenate((ave_neurons_between[:, 0:36], ave_neurons_between[:,48::]), axis = 1)
#         ave_midnight_between = np.concatenate((ave_midnight_between[:, 0:36], ave_midnight_between[:,48::]), axis = 1)
#         ave_location_between = np.concatenate((ave_location_between[:, 0:36], ave_location_between[:,48::]), axis = 1)
#         ave_phase_between = np.concatenate((ave_phase_between[:, 0:36], ave_phase_between[:,48::]), axis = 1)
        
        
#     #if mouse_recday == 'ah04_01122021_02122021': #mouse g range 0,8
#     # same tasks: 1,4 and 5,8
#     # ALL FINE WITH g 
           
#     if mouse_recday == 'ah03_18082021_19082021': #mouse h range 0,8
#         # hmmmm here I am not sure... maybe it is alright??
#         # the fourth task looks a bit off, but I am leaving it in for now
#         # same tasks: 1,4 and 5,8
#         print('yey')
    
    
#     # plot those that I am interested in
#     mc.simulation.predictions.plot_without_legends(ave_location_between, titlestring= 'location average', intervalline= 12)
#     mc.simulation.predictions.plot_without_legends(ave_clocks_between, titlestring= 'clock average', intervalline= 12)
#     mc.simulation.predictions.plot_without_legends(ave_midnight_between, titlestring= 'midnight average', intervalline= 12)
#     mc.simulation.predictions.plot_without_legends(ave_phase_between, titlestring= 'phase average', intervalline= 12)
#     mc.simulation.predictions.plot_without_legends(ave_neurons_between, titlestring= 'neuron average', intervalline= 12)
    
#     # task_one = ave_neurons_between[:, 0:12].copy()
#     # task_four = ave_neurons_between[:,36:48].copy()
#     # task_two = ave_neurons_between[:, 12:24].copy()
#     # corr_one_two = mc.simulation.RDMs.corr_matrices_pearson(task_one, task_two)
    
    
#     RSM_location_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_location_between, plotting = True, titlestring = 'Between tasks Location RSM, 12*12, averaged over runs')
#     RSM_clock_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_clocks_between, plotting = True, titlestring = 'Between tasks Clocks RSM, 12*12, averaged over runs', intervalline= 12)
#     RSM_midnight_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_midnight_between, plotting = True, titlestring = 'Between tasks Midnight RSM, 12*12, averaged over runs')
#     RSM_phase_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_phase_between, plotting = True, titlestring = 'Between tasks Pgase RSM, 12*12, averaged over runs')
#     RSM_neurons_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_neurons_between, plotting = True, titlestring = 'Between tasks Data RSM, 12*12, averaged over runs')
    
    
#     # DICKING AROUND WITH THE DATA
    
#     # first step: z-scoring!
#     # z-score the neuron matrices
#     import scipy
#     ave_neurons_between_z = scipy.stats.zscore(ave_neurons_between, axis=1)
#     mc.simulation.predictions.plot_without_legends(ave_neurons_between_z, titlestring='z-scored neuron average', intervalline= 12)
#     RSM_neurons_betas_ave_z = mc.simulation.RDMs.within_task_RDM(ave_neurons_between_z, plotting = True, titlestring = 'Between tasks Data RSM, 12*12, averaged over runs', intervalline= 12)
    
    
#     # this was based on something for task b. probably remove??
#     # RSM_clock_betas_ave_minusonetask = mc.simulation.RDMs.within_task_RDM(ave_clocks_between_minusonetask, plotting = True, titlestring = 'Between tasks Clocks RSM, 12*12, averaged over runs, withoutlasttask', intervalline= 12)
#     # RSM_phase_betas_ave_minusonetask = mc.simulation.RDMs.within_task_RDM(ave_phase_between_minusonetask, plotting = True, titlestring = 'Between tasks Pgase RSM, 12*12, averaged over runs withoutlasttask', intervalline= 12)
#     # RSM_midnight_betas_ave_minusonetask = mc.simulation.RDMs.within_task_RDM(ave_midnight_between_minusonetask, plotting = True, titlestring = 'Between tasks Midnight RSM, 12*12, averaged over runs withoutlasttask', intervalline= 12)
#     # RSM_location_betas_ave_minusonetask = mc.simulation.RDMs.within_task_RDM(ave_location_between_minusonetask, plotting = True, titlestring = 'Between tasks Location RSM, 12*12, averaged over runs withoutlasttask', intervalline= 12)
    
#     # # and also for the data 
#     # RSM_neurons_betas_ave_z_minusonetask = mc.simulation.RDMs.within_task_RDM(ave_neuron_data_minusonetask, plotting = True, titlestring = 'Between tasks Data RSM, 12*12, averaged over runs', intervalline= 12)
    
    
    
#     # # NEXT: create matrices where I plot each phase separetly: take every 3rd row
#     # mc.simulation.predictions.plot_without_legends(ave_clocks_between_minusonetask[:, 0::3], titlestring='early clocks across tasks', intervalline= 4)
#     # mc.simulation.predictions.plot_without_legends(ave_neuron_data_minusonetask[:, 0::3], titlestring='early neurons across tasks', intervalline= 4)
#     # mc.simulation.predictions.plot_without_legends(ave_midnight_between_minusonetask[:, 0::3], titlestring='early midnight across tasks', intervalline= 4)
    
#     # mc.simulation.predictions.plot_without_legends(ave_clocks_between_minusonetask[:, 1::3], titlestring='mid clocks across tasks', intervalline= 4)
#     # mc.simulation.predictions.plot_without_legends(ave_neuron_data_minusonetask[:, 1::3], titlestring='mid neurons across tasks', intervalline= 4)
#     # mc.simulation.predictions.plot_without_legends(ave_midnight_between_minusonetask[:, 1::3], titlestring='mid midnight across tasks', intervalline= 4)
    
#     # mc.simulation.predictions.plot_without_legends(ave_clocks_between_minusonetask[:, 2::3], titlestring='late clocks across tasks', intervalline= 4)
#     # mc.simulation.predictions.plot_without_legends(ave_neuron_data_minusonetask[:, 2::3], titlestring='late neurons across tasks', intervalline= 4)
#     # mc.simulation.predictions.plot_without_legends(ave_midnight_between_minusonetask[:, 2::3], titlestring='late midnight across tasks', intervalline= 4)
    
    
#     # RSM_early_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between_minusonetask[:, 0::3], plotting=True, titlestring='RSM early clocks, averaged over runs', intervalline= 4)
#     # RSM_early_neuron = mc.simulation.RDMs.within_task_RDM(ave_neuron_data_minusonetask[:, 0::3], plotting=True, titlestring='RSM early neurons, averaged over runs', intervalline= 4)
#     # RSM_early_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between_minusonetask[:, 0::3], plotting=True, titlestring='RSM early midnight, averaged over runs', intervalline= 4)
    
#     # RSM_mid_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between_minusonetask[:, 1::3], plotting=True, titlestring='RSM mid clocks, averaged over runs', intervalline= 4)
#     # RSM_mid_neuron = mc.simulation.RDMs.within_task_RDM(ave_neuron_data_minusonetask[:, 1::3], plotting=True, titlestring='RSM mid neurons, averaged over runs', intervalline= 4)
#     # RSM_mid_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between_minusonetask[:, 1::3], plotting=True, titlestring='RSM mid midnight, averaged over runs', intervalline= 4)
    
#     # RSM_late_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between_minusonetask[:, 2::3], plotting=True, titlestring='RSM late clocks, averaged over runs', intervalline= 4)
#     # RSM_late_neuron = mc.simulation.RDMs.within_task_RDM(ave_neuron_data_minusonetask[:, 2::3], plotting=True, titlestring='RSM late neurons, averaged over runs', intervalline= 4)
#     # RSM_late_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between_minusonetask[:, 2::3], plotting=True, titlestring='RSM late midnight, averaged over runs', intervalline= 4)
#     # NEXT: create matrices where I plot each phase separetly: take every 3rd row
#     mc.simulation.predictions.plot_without_legends(ave_clocks_between[:, 0::3], titlestring='early clocks across tasks', intervalline= 4)
#     mc.simulation.predictions.plot_without_legends(ave_neurons_between_z[:, 0::3], titlestring='early neurons across tasks', intervalline= 4)
#     mc.simulation.predictions.plot_without_legends(ave_midnight_between[:, 0::3], titlestring='early midnight across tasks', intervalline= 4)
    
#     mc.simulation.predictions.plot_without_legends(ave_clocks_between[:, 1::3], titlestring='mid clocks across tasks', intervalline= 4)
#     mc.simulation.predictions.plot_without_legends(ave_neurons_between_z[:, 1::3], titlestring='mid neurons across tasks', intervalline= 4)
#     mc.simulation.predictions.plot_without_legends(ave_midnight_between[:, 1::3], titlestring='mid midnight across tasks', intervalline= 4)
    
#     mc.simulation.predictions.plot_without_legends(ave_clocks_between[:, 2::3], titlestring='late clocks across tasks', intervalline= 4)
#     mc.simulation.predictions.plot_without_legends(ave_neurons_between_z[:, 2::3], titlestring='late neurons across tasks', intervalline= 4)
#     mc.simulation.predictions.plot_without_legends(ave_midnight_between[:, 2::3], titlestring='late midnight across tasks', intervalline= 4)
    
    
#     RSM_early_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, 0::3], plotting=True, titlestring='RSM early clocks, averaged over runs', intervalline= 4)
#     RSM_early_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between_z[:, 0::3], plotting=True, titlestring='RSM early neurons, averaged over runs', intervalline= 4)
#     RSM_early_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, 0::3], plotting=True, titlestring='RSM early midnight, averaged over runs', intervalline= 4)
    
#     RSM_mid_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, 1::3], plotting=True, titlestring='RSM mid clocks, averaged over runs', intervalline= 4)
#     RSM_mid_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between_z[:, 1::3], plotting=True, titlestring='RSM mid neurons, averaged over runs', intervalline= 4)
#     RSM_mid_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, 1::3], plotting=True, titlestring='RSM mid midnight, averaged over runs', intervalline= 4)
    
#     RSM_late_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, 2::3], plotting=True, titlestring='RSM late clocks, averaged over runs', intervalline= 4)
#     RSM_late_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between_z[:, 2::3], plotting=True, titlestring='RSM late neurons, averaged over runs', intervalline= 4)
#     RSM_late_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, 2::3], plotting=True, titlestring='RSM late midnight, averaged over runs', intervalline= 4)
    
#     # run regressions separetly for each phase
#     reg_early, scipy_early = mc.simulation.RDMs.lin_reg_RDMs(RSM_early_neuron, regressor_one_matrix = RSM_early_midnight, regressor_two_matrix= RSM_early_clocks)
#     print('results for early are [midnight, clocks]', reg_early.coef_)
#     reg_mid, scipy_mid = mc.simulation.RDMs.lin_reg_RDMs(RSM_mid_neuron, regressor_one_matrix = RSM_mid_midnight, regressor_two_matrix= RSM_mid_clocks)
#     print('results for mid are [midnight, clocks]', reg_mid.coef_)
#     reg_late, scipy_late = mc.simulation.RDMs.lin_reg_RDMs(RSM_late_neuron, regressor_one_matrix = RSM_late_midnight, regressor_two_matrix= RSM_late_clocks)
#     print('results for late are [midnight, clocks]', reg_late.coef_)

#     # LITTLE REGRESSION PLAYAROUND DELeTE LATER
#     from sklearn.linear_model import LinearRegression
    
#     dimension = len(RSM_early_clocks)
#     Yearly = list(RSM_early_neuron[np.tril_indices(dimension , -1)])
#     Ymid = list(RSM_mid_neuron[np.tril_indices(dimension , -1)])
#     Ylate = list(RSM_late_neuron[np.tril_indices(dimension , -1)])
#     Yall = np.hstack((Yearly, Ymid, Ylate))
    
#     Xearly = list(RSM_early_midnight[np.tril_indices(dimension, -1)])
#     Xclock_early = list(RSM_early_clocks[np.tril_indices(dimension, -1)])
#     Xearly = np.vstack((Xearly, Xclock_early))
    
#     Xmid = list(RSM_mid_midnight[np.tril_indices(dimension, -1)])
#     Xclock_mid = list(RSM_mid_clocks[np.tril_indices(dimension, -1)])
#     Xmid = np.vstack((Xmid, Xclock_mid))
    
#     Xlate = list(RSM_late_midnight[np.tril_indices(dimension, -1)])
#     Xclock_late = list(RSM_late_clocks[np.tril_indices(dimension, -1)])
#     Xlate = np.vstack((Xlate, Xclock_late))
    
#     Xall = np.hstack((Xearly, Xmid, Xlate))
#     x_all_reshaped = np.transpose(Xall)
#     regression_results = LinearRegression().fit(x_all_reshaped, Yall)
#     print('results for late are [midnight, clocks]', regression_results.coef_)

#     # do whole regression with only midnight and clock
#     RSM_neurons = mc.simulation.RDMs.within_task_RDM(ave_neurons_between_z)
#     RSM_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between)
#     RSM_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between)
    
#     reg_mid_clock, scipyblah = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons, regressor_one_matrix = RSM_midnight, regressor_two_matrix= RSM_clocks)
#     print('results for late are [midnight, clocks]', reg_mid_clock.coef_)
    
    
#     # then test how jus the correlation with the data RDMs looks like
#     # correlation
    
#     corr_clocks_data = mc.simulation.RDMs.corr_matrices_pearson(RSM_neurons, RSM_clocks)
#     print('correlation between clocks and data is', corr_clocks_data)
#     # .6453
#     corr_phase_data = mc.simulation.RDMs.corr_matrices_pearson(RSM_neurons, RSM_phase_betas_ave)
#     # .6644
#     print('correlation between phase and data is', corr_phase_data)
    
#     # then do a regression:
#     # regression with just the two most important ones - phase and clocks
#     #results_phase_clocks_minusonetask, scipy_regression_results = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_betas_ave_z_minusonetask, regressor_one_matrix=RSM_clock_betas_ave_minusonetask, regressor_two_matrix= RSM_phase_betas_ave_minusonetask)
#     # print('regression results are: [clocks, phase]:', results_phase_clocks_minusonetask.coef_)

#     # clocks beta = 0.16255517, phase beta = 0.25516436
    
#     # regression with all models
#     results_average, scipy_regression_results = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_betas_ave_z, regressor_one_matrix=RSM_clock_betas_ave, regressor_two_matrix= RSM_midnight_betas_ave, regressor_three_matrix= RSM_location_betas_ave, regressor_four_matrix= RSM_phase_betas_ave)
#     print('regression results are: [clocks, phase, location]:', results_average.coef_)
    
#     return results_average, scipy_regression_results, coefficient




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

    for no_trial_in_each_task in range(0, min_trialno):
        for task_no, task_config in enumerate(task_configs):
            # to take the final runs not the first ones.
            run_no = -1*(no_trial_in_each_task + 1)
            trajectory, timings_curr_run, index_make_step, step_number, curr_neurons = mc.analyse.analyse_ephys.prep_ephys_per_trial(timings_all, locations_all, run_no, task_no, task_config, neurons)
            
            if continuous == False:
                location_model = mc.simulation.predictions.set_location_raw_ephys(trajectory, step_time = 1, grid_size=3, plotting = False, field_no_given= 1)
                
                # i can choose which non-continuous model I want.
                # possible outputs are: midnight_matrix, full_clock_matrix, alternative_midnight, alternative_clock, compromise_midnight, compromise_clock
                
                # compromise models: consecutive activation, but on for the entire phase.
                midnight_model_part, clocks_model_fullphase, midnight_full, clocks_part, midnight_model, clocks_model = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=False)
                
                # full phase models 
                # midnight_model_part, midnight_model, clocks_model, clocks_part, compromise_midnight, compromise_clock = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=False)
                
                # both with gaps
                # midnight_model, clocks_model_fullphase, midnight_full, clocks_model, compromise_midnight, compromise_clock = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=False)
                
                # what I originally looked at:
                # midnight_model, clocks_model, alternative_midnight, alternative_clock, compromise_midnight, compromise_clock = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=False)
                
                #midnight_model, clocks_model = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=False)
                phase_model = mc.simulation.predictions.set_phase_model_ephys(trajectory, timings_curr_run, index_make_step, step_number)
            
            if continuous == True:
                location_model, phase_model, state_model, midnight_model, clocks_model, phase_state_model = mc.simulation.predictions.set_continous_models_ephys(trajectory, timings_curr_run, index_make_step, step_number, no_phase_neurons= number_phase_neurons, plot = False)
                   
            
            # now create the regressors per run
            regs_phase_state_run = mc.simulation.predictions.create_x_regressors_per_state(walked_path = trajectory, subpath_timings = timings_curr_run, step_no = step_number, no_regs_per_state = no_bins_per_state)
            # then use these regressors to generate a beta per neuron per run
            neurons_phase_state = mc.simulation.predictions.transform_data_to_betas(curr_neurons, regs_phase_state_run)
            clock_phase_state = mc.simulation.predictions.transform_data_to_betas(clocks_model, regs_phase_state_run)
            midnight_phase_state= mc.simulation.predictions.transform_data_to_betas(midnight_model, regs_phase_state_run)
            location_phase_state = mc.simulation.predictions.transform_data_to_betas(location_model, regs_phase_state_run)
            phase_phase_state = mc.simulation.predictions.transform_data_to_betas(phase_model, regs_phase_state_run)
            state_phase_state = mc.simulation.predictions.transform_data_to_betas(state_model, regs_phase_state_run)
            
 
            
            # these need to be concatenated for each run and task
            if task_no == 0:
            #if task_no == 0 and trial_no == 0:
                neurons_between = neurons_phase_state.copy()
                clocks_between = clock_phase_state.copy()
                midnight_between = midnight_phase_state.copy()
                location_between = location_phase_state.copy()
                phase_between = phase_phase_state.copy()
                state_between = state_phase_state.copy()
                if split_by_phase:
                    phase_separation = mc.simulation.predictions.set_phase_model_ephys(trajectory, timings_curr_run, index_make_step, step_number)
                    phase_separation = np.round(mc.simulation.predictions.transform_data_to_betas(phase_separation, regs_phase_state_run))
            else:
                neurons_between = np.concatenate((neurons_between, neurons_phase_state), axis = 1)
                clocks_between = np.concatenate((clocks_between, clock_phase_state), axis = 1)
                midnight_between = np.concatenate((midnight_between, midnight_phase_state), axis = 1)
                location_between = np.concatenate((location_between, location_phase_state), axis = 1)
                phase_between = np.concatenate((phase_between,phase_phase_state), axis = 1)
                state_between = np.concatenate((state_between, state_phase_state), axis = 1)
                if split_by_phase == 1:
                    phase_separation_temp = mc.simulation.predictions.set_phase_model_ephys(trajectory, timings_curr_run, index_make_step, step_number)
                    phase_separation_temp = np.round(mc.simulation.predictions.transform_data_to_betas(phase_separation_temp, regs_phase_state_run))
                    phase_separation = np.concatenate((phase_separation, phase_separation_temp), axis = 1)
                
        
        
        # create an averaged neuron file and RDM. 
        if no_trial_in_each_task == 0:
            sum_location_between = location_between.copy()
            sum_clocks_between = clocks_between.copy()
            sum_midnight_between = midnight_between.copy()
            sum_phase_between = phase_between.copy()
            sum_neurons_between = neurons_between.copy()
            sum_state_between = state_between.copy()
            if split_by_phase:
                sum_phase_separation = phase_separation.copy()
        if no_trial_in_each_task > 0:
            sum_location_between = sum_location_between.copy() + location_between.copy()
            sum_clocks_between = sum_clocks_between.copy() + clocks_between.copy()
            sum_midnight_between = sum_midnight_between.copy() + midnight_between.copy()
            sum_phase_between = sum_phase_between.copy() + phase_between.copy()
            sum_neurons_between = sum_neurons_between.copy() + neurons_between.copy()
            sum_state_between = sum_state_between.copy() + state_between.copy()
            if split_by_phase:
                sum_phase_separation = sum_phase_separation.copy() + phase_separation.copy()
        
    
    # import pdb; pdb.set_trace()
    
    ave_location_between = sum_location_between/no_trial_in_each_task
    ave_clocks_between = sum_clocks_between/no_trial_in_each_task
    ave_midnight_between = sum_midnight_between/no_trial_in_each_task
    ave_phase_between = sum_phase_between/no_trial_in_each_task
    ave_neurons_between = sum_neurons_between/no_trial_in_each_task
    ave_state_between = sum_state_between/no_trial_in_each_task
    
    if split_by_phase:
        ave_phase_separation = sum_phase_separation/no_trial_in_each_task
        max_val = np.max(ave_phase_separation)
        # import pdb; pdb.set_trace()
        early_mask = np.where(ave_phase_separation[0,:] == max_val)[0]
        mid_mask = np.where(ave_phase_separation[1,:] == max_val)[0]
        late_mask = np.where(ave_phase_separation[2,:] == max_val)[0]
    
    import pdb; pdb.set_trace()
    if plotting == True:
        #import pdb; pdb.set_trace()
        # plot the averaged simulated and cleaned data
        mc.simulation.predictions.plot_without_legends(ave_location_between, titlestring= 'Location model, averaged across runs in mouse a', intervalline= 4*no_bins_per_state, saving_file='/Users/xpsy1114/Documents/projects/multiple_clocks/output/')
        mc.simulation.predictions.plot_without_legends(ave_clocks_between, titlestring= 'Schema model, averaged across runs in mouse a', intervalline= 4*no_bins_per_state, saving_file='/Users/xpsy1114/Documents/projects/multiple_clocks/output/')
        mc.simulation.predictions.plot_without_legends(ave_midnight_between, titlestring= 'Partial schema model, averaged across runs in mouse a', intervalline= 4*no_bins_per_state, saving_file='/Users/xpsy1114/Documents/projects/multiple_clocks/output/')
        mc.simulation.predictions.plot_without_legends(ave_phase_between, titlestring= 'Task progress model, averaged across runs in mouse a', intervalline= 4*no_bins_per_state, saving_file='/Users/xpsy1114/Documents/projects/multiple_clocks/output/')
        mc.simulation.predictions.plot_without_legends(ave_neurons_between, titlestring= 'Recorded neurons, averaged across runs in mouse a', intervalline= 4*no_bins_per_state, saving_file='/Users/xpsy1114/Documents/projects/multiple_clocks/output/')
        mc.simulation.predictions.plot_without_legends(ave_state_between, titlestring= 'State model, averaged across runs in mouse a', intervalline= 4*no_bins_per_state, saving_file='/Users/xpsy1114/Documents/projects/multiple_clocks/output/')
        
        
        RDM_dict = {}
        
        RDM_dict['RSM_location_betas_ave'] = mc.simulation.RDMs.within_task_RDM(ave_location_between, plotting = True, titlestring = 'Between tasks Location RSM, 12*12, averaged over runs', intervalline= 4*no_bins_per_state)
        RDM_dict['RSM_clock_betas_ave'] = mc.simulation.RDMs.within_task_RDM(ave_clocks_between, plotting = True, titlestring = 'Between tasks Musicbox RSM, 12*12, averaged over runs', intervalline= 4*no_bins_per_state)
        RDM_dict['RSM_midnight_betas_ave'] = mc.simulation.RDMs.within_task_RDM(ave_midnight_between, plotting = True, titlestring = 'Between tasks Midnight RSM, 12*12, averaged over runs', intervalline= 4*no_bins_per_state)
        RDM_dict['RSM_phase_betas_ave'] = mc.simulation.RDMs.within_task_RDM(ave_phase_between, plotting = True, titlestring = 'Between tasks Phase RSM, 12*12, averaged over runs', intervalline= 4*no_bins_per_state)
        RDM_dict['RSM_neurons_betas_ave'] = mc.simulation.RDMs.within_task_RDM(ave_neurons_between, plotting = True, titlestring = 'Between tasks Data RSM, 12*12, averaged over runs', intervalline= 4*no_bins_per_state)
        
        
        mc.simulation.RDMs.plot_RDMs(RDM_dict, len(task_configs))       

        
        
        # separately per phase
        if split_by_phase:
            mc.simulation.predictions.plot_without_legends(ave_clocks_between[:, early_mask], titlestring='early clocks across tasks', intervalline= 4*no_bins_per_state/3)
            mc.simulation.predictions.plot_without_legends(ave_neurons_between[:, early_mask], titlestring='early neurons across tasks', intervalline= 4*no_bins_per_state/3)
            mc.simulation.predictions.plot_without_legends(ave_midnight_between[:, early_mask], titlestring='early midnight across tasks', intervalline= 4*no_bins_per_state/3)
            
            mc.simulation.predictions.plot_without_legends(ave_clocks_between[:, mid_mask], titlestring='mid clocks across tasks', intervalline= 4*no_bins_per_state/3)
            mc.simulation.predictions.plot_without_legends(ave_neurons_between[:, mid_mask], titlestring='mid neurons across tasks', intervalline= 4*no_bins_per_state/3)
            mc.simulation.predictions.plot_without_legends(ave_midnight_between[:, mid_mask], titlestring='mid midnight across tasks', intervalline= 4*no_bins_per_state/3)
            
            mc.simulation.predictions.plot_without_legends(ave_clocks_between[:, late_mask], titlestring='late clocks across tasks', intervalline= 4*no_bins_per_state/3)
            mc.simulation.predictions.plot_without_legends(ave_neurons_between[:, late_mask], titlestring='late neurons across tasks', intervalline= 4*no_bins_per_state/3)
            mc.simulation.predictions.plot_without_legends(ave_midnight_between[:, late_mask], titlestring='late midnight across tasks', intervalline= 4*no_bins_per_state/3)
            
            RSM_early_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, early_mask], plotting=True, titlestring='RSM early clocks, averaged over runs', intervalline= 4*no_bins_per_state/3)
            RSM_early_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between[:, early_mask], plotting=True, titlestring='RSM early neurons, averaged over runs', intervalline= 4*no_bins_per_state/3)
            RSM_early_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, early_mask], plotting=True, titlestring='RSM early midnight, averaged over runs', intervalline= 4*no_bins_per_state/3)
            
            RSM_mid_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, mid_mask], plotting=True, titlestring='RSM mid clocks, averaged over runs', intervalline= 4*no_bins_per_state/3)
            RSM_mid_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between[:, mid_mask], plotting=True, titlestring='RSM mid neurons, averaged over runs', intervalline= 4*no_bins_per_state/3)
            RSM_mid_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, mid_mask], plotting=True, titlestring='RSM mid midnight, averaged over runs', intervalline= 4*no_bins_per_state/3)
            
            RSM_late_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, late_mask], plotting=True, titlestring='RSM late clocks, averaged over runs', intervalline= 4*no_bins_per_state/3)
            RSM_late_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between[:, late_mask], plotting=True, titlestring='RSM late neurons, averaged over runs', intervalline= 4*no_bins_per_state/3)
            RSM_late_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, late_mask], plotting=True, titlestring='RSM late midnight, averaged over runs', intervalline= 4*no_bins_per_state/3)
            
    
    elif plotting == False: 
        # for all phases
        RSM_location_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_location_between, plotting = False)
        RSM_clock_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_clocks_between, plotting = False)
        RSM_midnight_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_midnight_between, plotting = False)
        RSM_phase_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_phase_between, plotting = False)
        RSM_neurons_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_neurons_between, plotting = False)
        RSM_state_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_state_between, plotting = False)
        
        
        
        if split_by_phase:
            # for each phase separately
            RSM_early_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, early_mask], plotting = False)
            RSM_early_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between[:, early_mask], plotting = False)
            RSM_early_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, early_mask], plotting = False)
            
            RSM_mid_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, mid_mask], plotting = False)
            RSM_mid_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between[:, mid_mask], plotting = False)
            RSM_mid_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, mid_mask], plotting = False)
            
            RSM_late_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, late_mask], plotting = False)
            RSM_late_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between[:, late_mask], plotting = False)
            RSM_late_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, late_mask], plotting = False)
            

    
    # run regressions separetly for each phase
    # import pdb; pdb.set_trace()
    # try the new regression.
    # import pdb; pdb.set_trace()
    regressors = {}
    regressors['clocks']=RSM_clock_betas_ave
    regressors['midnight']=RSM_midnight_betas_ave
    regressors['phase']=RSM_phase_betas_ave
    regressors['location']=RSM_location_betas_ave
    regressors['stat']=RSM_state_betas_ave
    results_normal = mc.simulation.RDMs.GLM_RDMs(RSM_neurons_betas_ave, regressors, mask_within, no_tasks = len(task_configs), plotting= True)
    
    # collect all values in a results table which I then output in the end.
    result_dict = {}
    result_dict['normal']= results_normal
    
    if split_by_phase:
        regressors_early = {}
        regressors_early['clocks']=RSM_early_clocks
        regressors_early['midnight']=RSM_early_midnight
        regressors_mid = {}
        regressors_mid['clocks']=RSM_mid_clocks
        regressors_mid['midnight']=RSM_mid_midnight
        regressors_late = {}
        regressors_late['clocks']=RSM_late_clocks
        regressors_late['midnight']=RSM_late_midnight
        
        results_early = mc.simulation.RDMs.GLM_RDMs(RSM_early_neuron, regressors_early, mask_within=False, no_tasks = len(task_configs))
        results_mid = mc.simulation.RDMs.GLM_RDMs(RSM_mid_neuron, regressors_mid, mask_within=False, no_tasks = len(task_configs))
        results_late = mc.simulation.RDMs.GLM_RDMs(RSM_late_neuron, regressors_late, mask_within=False, no_tasks = len(task_configs))
        
        result_dict['early']=results_early
        result_dict['mid']= results_mid
        result_dict['late']= results_late
        
        # and for the phases put back together.
        reord_clocks = np.concatenate((ave_clocks_between[:, early_mask], ave_clocks_between[:, mid_mask]), axis = 1)
        reord_clocks = np.concatenate((reord_clocks, ave_clocks_between[:, late_mask]), axis = 1)
        RSM_reord_clocks = mc.simulation.RDMs.within_task_RDM(reord_clocks, plotting = False)
        
        reord_locs = np.concatenate((ave_location_between[:, early_mask], ave_location_between[:, mid_mask]), axis = 1)
        reord_locs = np.concatenate((reord_locs, ave_location_between[:, late_mask]), axis = 1)
        RSM_reord_locs = mc.simulation.RDMs.within_task_RDM(reord_locs, plotting = False)
        
        reord_midn = np.concatenate((ave_midnight_between[:, early_mask], ave_midnight_between[:, mid_mask]), axis = 1)
        reord_midn = np.concatenate((reord_midn, ave_midnight_between[:, late_mask]), axis = 1)
        RSM_reord_midn = mc.simulation.RDMs.within_task_RDM(reord_midn, plotting = False)
        
        reord_neurons = np.concatenate((ave_neurons_between[:, early_mask], ave_neurons_between[:, mid_mask]), axis = 1)
        reord_neurons = np.concatenate((reord_neurons, ave_neurons_between[:, late_mask]), axis = 1)
        RSM_reord_neurons = mc.simulation.RDMs.within_task_RDM(reord_neurons, plotting = False)

        regs_reordered = {}
        regs_reordered['clocks'] = RSM_reord_clocks
        regs_reordered['midnight'] = RSM_reord_midn
        regs_reordered['location'] = RSM_reord_locs
        results_reord = mc.simulation.RDMs.GLM_RDMs(RSM_reord_neurons, regs_reordered, mask_within=False, no_tasks = len(task_configs))
        result_dict["reord_coefs"] = results_reord['coefs']
        result_dict["reord_t-vals"] = results_reord['t_vals']
        
    #import pdb; pdb.set_trace()
    # # LITTLE REGRESSION PLAYAROUND
    # # this is a regression where I put all early phases of all tasks behind each other
    # # then all mid phases of each tasks, etc; and then do the regression.
    # from sklearn.linear_model import LinearRegression
    
    # Yearly = list(RSM_early_neuron[np.tril_indices(len(RSM_early_neuron) , -1)])
    # Ymid = list(RSM_mid_neuron[np.tril_indices(len(RSM_mid_neuron) , -1)])
    # Ylate = list(RSM_late_neuron[np.tril_indices(len(RSM_late_neuron) , -1)])
    # Yall = np.hstack((Yearly, Ymid, Ylate))
    
    # Xearly = list(RSM_early_midnight[np.tril_indices(len(RSM_early_neuron), -1)])
    # Xclock_early = list(RSM_early_clocks[np.tril_indices(len(RSM_early_neuron), -1)])
    # Xearly = np.vstack((Xearly, Xclock_early))
    
    # Xmid = list(RSM_mid_midnight[np.tril_indices(len(RSM_mid_neuron), -1)])
    # Xclock_mid = list(RSM_mid_clocks[np.tril_indices(len(RSM_mid_neuron), -1)])
    # Xmid = np.vstack((Xmid, Xclock_mid))
    
    # Xlate = list(RSM_late_midnight[np.tril_indices(len(RSM_late_neuron), -1)])
    # Xclock_late = list(RSM_late_clocks[np.tril_indices(len(RSM_late_neuron), -1)])
    # Xlate = np.vstack((Xlate, Xclock_late))
    
    # Xall = np.hstack((Xearly, Xmid, Xlate))
    # x_all_reshaped = np.transpose(Xall)
    # reversed_phases_reg_results = LinearRegression().fit(x_all_reshaped, Yall)
    # print('results for putting all neurons together are [midnight, clocks]', reversed_phases_reg_results.coef_)

    # # do whole regression with only midnight and clock
    # reg_mid_clock, scipyblah = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_betas_ave, regressor_one_matrix = RSM_midnight_betas_ave, regressor_two_matrix= RSM_clock_betas_ave)
    # print('results all normal RSMs are [midnight, clocks]', reg_mid_clock.coef_)
    
    # # regression with all models
    # results_average, scipy_regression_results = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_betas_ave, regressor_one_matrix=RSM_midnight_betas_ave, regressor_two_matrix= RSM_clock_betas_ave, regressor_three_matrix= RSM_location_betas_ave, regressor_four_matrix= RSM_phase_betas_ave)
    # print('regression results are: [midnight, clocks,location, phase]:', results_average.coef_)

    
    # result_dict['reg_early_phase_midnight-clocks'] = reg_early.coef_
    # result_dict['reg_mid_phase_midnight-clocks'] = reg_mid.coef_
    # result_dict['reg_late_phase_midnight-clocks'] = reg_late.coef_
    # result_dict['reg_all_midnight-clocks'] = reg_mid_clock.coef_
    # result_dict['reg_all_midnight-clocks-loc-phase'] = results_average.coef_
    # result_dict['reg_all_reversedphase_midnight-clocks'] = reversed_phases_reg_results.coef_
    
    return result_dict





def load_ephys_data(Data_folder):
    # import pdb; pdb.set_trace()
    mouse_a = {}
    mouse_b = {}
    mouse_c = {}
    mouse_d = {}
    mouse_e = {}
    mouse_f = {}
    mouse_g = {}
    mouse_h = {}
    
    
    mouse_recday='me11_05122021_06122021' #mouse a
    mouse_a["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
    mouse_a["anchor_lag"]= np.load(Data_folder+'Anchor_lag_'+mouse_recday+'.npy')
    mouse_a["anchor_lag_threshold"] = np.load(Data_folder+'Anchor_lag_threshold_'+mouse_recday+'.npy')
    a_no_task_configs = len(mouse_a["rewards_configs"])
    mouse_a["cells"] = np.load(Data_folder+'Phase_state_place_anchored_' + mouse_recday + '.npy')
    a_locations = list()
    a_neurons = list()
    a_timings = list()
    for session in range(0, a_no_task_configs):
        a_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        a_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        a_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))
    
    #import pdb; pdb.set_trace()
    mouse_a["locations"] = a_locations
    mouse_a["neurons"] = a_neurons
    mouse_a["timings"] = a_timings
    mouse_a["recday"] = mouse_recday
    
    mouse_a["neuron_type"] = np.zeros((len(mouse_a["anchor_lag"]), len(mouse_a["anchor_lag"][1])))
    for i, neuron in enumerate(mouse_a["anchor_lag"]):
        max_neuron = np.argmax(neuron)
        mouse_a["neuron_type"][i, max_neuron] = 1
        
        
    
    
    

    mouse_recday='me11_01122021_02122021' #mouse b 
    mouse_b["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
    mouse_b["anchor_lag"]= np.load(Data_folder+'Anchor_lag_'+mouse_recday+'.npy')
    mouse_b["anchor_lag_threshold"] = np.load(Data_folder+'Anchor_lag_threshold_'+mouse_recday+'.npy')
    b_no_task_configs = len(mouse_b["rewards_configs"])
    mouse_b["cells"] = np.load(Data_folder+'Phase_state_place_anchored_' + mouse_recday + '.npy')
    b_locations = list()
    b_neurons = list()
    b_timings = list()
    for session in range(0, b_no_task_configs):
        b_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        b_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        b_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))
    mouse_b["locations"] = b_locations
    mouse_b["neurons"] = b_neurons
    mouse_b["timings"] = b_timings
    mouse_b["recday"] = mouse_recday
    mouse_b["neuron_type"] = np.zeros((len(mouse_b["anchor_lag"]), len(mouse_b["anchor_lag"][1])))
    for i, neuron in enumerate(mouse_b["anchor_lag"]):
        max_neuron = np.argmax(neuron)
        mouse_b["neuron_type"][i, max_neuron] = 1
        

    mouse_recday='me10_09122021_10122021' #mouse c range 0,9
    mouse_c["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
    mouse_c["anchor_lag"]= np.load(Data_folder+'Anchor_lag_'+mouse_recday+'.npy')
    mouse_c["anchor_lag_threshold"] = np.load(Data_folder+'Anchor_lag_threshold_'+mouse_recday+'.npy')
    c_no_task_configs = len(mouse_c["rewards_configs"])
    mouse_c["cells"] = np.load(Data_folder+'Phase_state_place_anchored_' + mouse_recday + '.npy')
    c_locations = list()
    c_neurons = list()
    c_timings = list()
    for session in range(0, c_no_task_configs):
        c_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        c_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        c_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))
    mouse_c["locations"] = c_locations
    mouse_c["neurons"] = c_neurons
    mouse_c["timings"] = c_timings
    mouse_c["recday"] = mouse_recday
    mouse_c["neuron_type"] = np.zeros((len(mouse_c["anchor_lag"]), len(mouse_c["anchor_lag"][1])))
    for i, neuron in enumerate(mouse_c["anchor_lag"]):
        max_neuron = np.argmax(neuron)
        mouse_c["neuron_type"][i, max_neuron] = 1   

    mouse_recday='me08_10092021_11092021' #mouse d range 0,6
    mouse_d["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
    mouse_d["rewards_configs"] = mouse_d["rewards_configs"][0:-1, :].copy()
    # apparently there is one run less for this day..., so exclude that one
    # mohammady says: The ephys file for the last task on that day was lost
    mouse_d["anchor_lag"]= np.load(Data_folder+'Anchor_lag_'+mouse_recday+'.npy')
    mouse_d["anchor_lag_threshold"] = np.load(Data_folder+'Anchor_lag_threshold_'+mouse_recday+'.npy')
    d_no_task_configs = len(mouse_d["rewards_configs"])
    mouse_d["cells"] = np.load(Data_folder+'Phase_state_place_anchored_' + mouse_recday + '.npy')
    d_locations = list()
    d_neurons = list()
    d_timings = list()
    for session in range(0, d_no_task_configs):
        d_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        d_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        d_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))
    mouse_d["locations"] = d_locations
    mouse_d["neurons"] = d_neurons
    mouse_d["timings"] = d_timings
    mouse_d["recday"] = mouse_recday
    mouse_d["neuron_type"] = np.zeros((len(mouse_d["anchor_lag"]), len(mouse_d["anchor_lag"][1])))
    for i, neuron in enumerate(mouse_d["anchor_lag"]):
        max_neuron = np.argmax(neuron)
        mouse_d["neuron_type"][i, max_neuron] = 1

    mouse_recday='ah04_09122021_10122021' #mouse e range 0,8
    mouse_e["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
    mouse_e["anchor_lag"]= np.load(Data_folder+'Anchor_lag_'+mouse_recday+'.npy')
    mouse_e["anchor_lag_threshold"] = np.load(Data_folder+'Anchor_lag_threshold_'+mouse_recday+'.npy')
    e_no_task_configs = len(mouse_e["rewards_configs"])
    mouse_e["cells"] = np.load(Data_folder+'Phase_state_place_anchored_' + mouse_recday + '.npy')
    e_locations = list()
    e_neurons = list()
    e_timings = list()
    for session in range(0, e_no_task_configs):
        e_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        e_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        e_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))
    mouse_e["locations"] = e_locations
    mouse_e["neurons"] = e_neurons
    mouse_e["timings"] = e_timings 
    mouse_e["recday"] = mouse_recday
    mouse_e["neuron_type"] = np.zeros((len(mouse_e["anchor_lag"]), len(mouse_e["anchor_lag"][1])))
    for i, neuron in enumerate(mouse_e["anchor_lag"]):
        max_neuron = np.argmax(neuron)
        mouse_e["neuron_type"][i, max_neuron] = 1   
     
        
    mouse_recday='ah04_05122021_06122021' #mouse f range 0,8
    mouse_f["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
    mouse_f["anchor_lag"]= np.load(Data_folder+'Anchor_lag_'+mouse_recday+'.npy')
    mouse_f["anchor_lag_threshold"] = np.load(Data_folder+'Anchor_lag_threshold_'+mouse_recday+'.npy')
    f_no_task_configs = len(mouse_f["rewards_configs"])
    mouse_f["cells"] = np.load(Data_folder+'Phase_state_place_anchored_' + mouse_recday + '.npy')
    f_locations = list()
    f_neurons = list()
    f_timings = list()
    for session in range(0, f_no_task_configs):
        f_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        f_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        f_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))
    mouse_f["locations"] = f_locations
    mouse_f["neurons"] = f_neurons
    mouse_f["timings"] = f_timings
    mouse_f["recday"] = mouse_recday
    mouse_f["neuron_type"] = np.zeros((len(mouse_f["anchor_lag"]), len(mouse_f["anchor_lag"][1])))
    for i, neuron in enumerate(mouse_f["anchor_lag"]):
        max_neuron = np.argmax(neuron)
        mouse_f["neuron_type"][i, max_neuron] = 1

    mouse_recday='ah04_01122021_02122021' #mouse g range 0,8
    mouse_g["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
    mouse_g["anchor_lag"]= np.load(Data_folder+'Anchor_lag_'+mouse_recday+'.npy')
    mouse_g["anchor_lag_threshold"] = np.load(Data_folder+'Anchor_lag_threshold_'+mouse_recday+'.npy')
    g_no_task_configs = len(mouse_g["rewards_configs"])
    mouse_g["cells"] = np.load(Data_folder+'Phase_state_place_anchored_' + mouse_recday + '.npy')
    g_locations = list()
    g_neurons = list()
    g_timings = list()
    for session in range(0, g_no_task_configs):
        g_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        g_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        g_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))
    mouse_g["locations"] = g_locations
    mouse_g["neurons"] = g_neurons
    mouse_g["timings"] = g_timings
    mouse_g["recday"] = mouse_recday
    mouse_g["neuron_type"] = np.zeros((len(mouse_g["anchor_lag"]), len(mouse_g["anchor_lag"][1])))
    for i, neuron in enumerate(mouse_g["anchor_lag"]):
        max_neuron = np.argmax(neuron)
        mouse_g["neuron_type"][i, max_neuron] = 1
    
    mouse_recday='ah03_18082021_19082021' #mouse h range 0,8
    mouse_h["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
    mouse_h["anchor_lag"]= np.load(Data_folder+'Anchor_lag_'+mouse_recday+'.npy')
    mouse_h["anchor_lag_threshold"] = np.load(Data_folder+'Anchor_lag_threshold_'+mouse_recday+'.npy')
    h_no_task_configs = len(mouse_h["rewards_configs"])
    mouse_h["cells"] = np.load(Data_folder+'Phase_state_place_anchored_' + mouse_recday + '.npy')
    h_locations = list()
    h_neurons = list()
    h_timings = list()
    for session in range(0, h_no_task_configs):
        h_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        h_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        h_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))
    mouse_h["locations"] = h_locations
    mouse_h["neurons"] = h_neurons
    mouse_h["timings"] = h_timings
    mouse_h["recday"] = mouse_recday
    mouse_h["neuron_type"] = np.zeros((len(mouse_h["anchor_lag"]), len(mouse_h["anchor_lag"][1])))
    for i, neuron in enumerate(mouse_h["anchor_lag"]):
        max_neuron = np.argmax(neuron)
        mouse_h["neuron_type"][i, max_neuron] = 1
    # # for h, the first timings array is missing
    # # > delete the first task completely!
    # h_timings = h_timings[1::]
    # h_neurons = h_neurons[1::]
    # h_locations = h_locations[1::]
    # h_rewards_configs = h_rewards_configs[1::, :]
    return(mouse_a, mouse_b, mouse_c, mouse_d, mouse_e, mouse_f, mouse_g, mouse_h)

def clean_ephys_data(task_configs, locations_all, neurons, timings_all, mouse_recday, ignore_double_tasks = 1):
    # first clean data.
    # import pdb; pdb.set_trace()
    # load dataset
    # # first thing: check for missing data. mark those and potentially ignore.
    # # loop through the neurons.
    # missing_neurons = []
    # for task_no, recording_task in enumerate(neurons):
    #     missing_neurons.append(list(map(tuple, np.where(recording_task ==[]))))

    
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
            
                

    # if mouse_recday == 'me11_05122021_06122021': #mouse a
    # # ALL FINE WITH a!
    #     # task 5 and 9 are the same, as well as 6 and 7
    #     # data of the first 4 tasks look similar, and tasks 5,6,7,8,9 look more similar
    #     if ignore_double_tasks == 1:
    #     # task 5 and 9 are the same, as well as 6 and 7
    #     # throw out 6 and 9
    #     # 1 and 4 are nearly the same, but have a different last field... so I leave them in.
    #         ignore = [8,5]
                
    # if mouse_recday == 'me11_01122021_02122021':#mouse b
        
    #     if ignore_double_tasks == 1:
    #         ignore = [-1, 3]
    #         # and task 4 appears twice
    #     elif ignore_double_tasks == 0:
    #         ignore = [-1]
    #         # get rid of the last task because it looks somewhat whacky
        
    # if mouse_recday == 'me10_09122021_10122021':#mouse c 
    #     if ignore_double_tasks == 1:
    #         ignore = [8,3,4]
    #     elif ignore_double_tasks == 0:
    #         ignore = [8,3]
    #     # same tasks are: 1,4; and  5,6,9
    #     # 4 and 9 look whacky, so remove those
    #     # so then after removal 5 and 6 are still the same and 5 has only 6 repeats
    #     # consider also removing the penultimum one... this was before task 7, now it is 6
    #     # so far this is still inside
         
    # if mouse_recday == 'me08_10092021_11092021': #mouse d
    #     if ignore_double_tasks == 1:
    #         ignore = [3]
    #         # DOUBLE CHECK THIS!!!
    # # same tasks: 1, 4
    # # ALL FINE WITH d ONCE THE LAST BUT THE LAST EPHYS FILE WAS LOST > deleted this before

    # if mouse_recday == 'ah04_09122021_10122021': #mouse e range 0,8
    # # throw out the 4th 
    # # same tasks: (all tasks are unique, before 1 and 4 were the same but 4 is gone)
    #     ignore = [4]
    
    # if mouse_recday == 'ah04_05122021_06122021': #mouse f range 0,8
    # # throw out number 4
    # # new 4 (previous 5) and last one - 7 (previous 8) are the same
    #     if ignore_double_tasks == 1:
    #         ignore = [-1, 3]
    #     elif ignore_double_tasks == 0:
    #         ignore = [3]
     
    # if mouse_recday == 'ah04_01122021_02122021': #mouse g range 0,8
    # # same tasks: 1,4 and 5,8
    # # ALL FINE WITH g 
    #     if ignore_double_tasks == 1:
    #         ignore = [4, 0]
        
    # if mouse_recday == 'ah03_18082021_19082021': #mouse h range 0,8
    #     if ignore_double_tasks == 1:
    #         ignore = [4, 0]
    # # hmmmm here I am not sure... maybe it is alright??
    # # the fourth task looks a bit off, but I am leaving it in for now
    # # same tasks: 1,4 and 5,8
   
    
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
    # I have no idea why I wrote thi... but I think it has to be this:
    row = timings_task[no_trial_in_each_task]
    # row =  timings_task[-(no_trial_in_each_task+1),:].copy()
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
        
    return(trajectory, timings_curr_run, index_make_step, step_number, curr_neurons)



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

    for index, contrast in enumerate(data_list):
        noise = sigma * np.random.randn(len(contrast)) + mu
        data_to_plot = np.array(contrast)  # Ensure data_to_plot is an array for direct operations
        x_positions = index + 1 + noise  # Adjust index for boxplot's 1-based index
    
        ax.scatter(x_positions, data_to_plot, color=colors[index], marker='o', s=100, edgecolors='black', linewidth=1)
    
    ax.set_xticks(range(1, len(data_list) + 1))
    plt.xticks(rotation=45)
    #ax.set_xticklabels(label_string_list, fontsize=26)
    ax.set_xticklabels(label_string_list)
    ax.set_ylabel('Betas')
    plt.axhline(0, color='grey', ls='dashed', linewidth=1)
    
    plt.title(title_string)
    
    plt.rcParams.update({'font.size': 30})
    # Customize grid lines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust the layout
    plt.tight_layout()
    
    plt.show()



    # fig, ax = plt.subplots(figsize=(8,6))
    # ax.boxplot(data_list)
    # sigma = 0.08
    # mu = 0.01
    # #cmaps.Pastel1
    # cmap = plt.get_cmap('Pastel1')
    # for index, contrast in enumerate(data_list):
    #     noise = np.ones(len(data_list[index])) + sigma * np.random.randn(len(data_list[index])) + mu
    #     data_to_plot = contrast.copy()
    #     for i, elem in enumerate(data_to_plot):
    #         data_to_plot[i] = elem +noise[index]     
    #     ax.scatter(noise+index, data_to_plot, cmap=cmap, marker='o', s=100, edgecolors = 'black', linewidth = 1)
    
    # ax.set_xticks(label_tick_list)
    # plt.xticks(rotation = 45)
    # ax.set_xticklabels(label_string_list, fontsize = 18)
    # #ax.set_yticklabels(fontsize = 18)
    # plt.axhline(0, color='grey', ls='dashed', linewidth = 1)
    # plt.title(title_string)
    

    #ax.set_yticklabels([str(f) for f in field_names], fontsize = 16)
    
    
    

    
    if save_fig:
        fig.savefig(f"{save_fig}{title_string}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{save_fig}{title_string}.tiff", dpi=300, bbox_inches='tight')
    
    













    # clean data according to recording day.

    
    # if mouse_recday == 'me11_05122021_06122021': #mouse a
    # # ALL FINE WITH a!
    #     # task 5 and 9 are the same, as well as 6 and 7
    #     # data of the first 4 tasks look similar, and tasks 5,6,7,8,9 look more similar
    #     if ignore_double_tasks == 1:
    #     # task 5 and 9 are the same, as well as 6 and 7
    #     # throw out 6 and 9
    #     # 1 and 4 are nearly the same, but have a different last field... so I leave them in.
    #         ave_clocks_between = np.concatenate((ave_clocks_between[:, 0:60], ave_clocks_between[:, 72:96]), axis = 1)
    #         ave_neurons_between = np.concatenate((ave_neurons_between[:, 0:60], ave_neurons_between[:,72:96]), axis = 1)
    #         ave_midnight_between = np.concatenate((ave_midnight_between[:, 0:60], ave_midnight_between[:,72:96]), axis = 1)
    #         ave_location_between = np.concatenate((ave_location_between[:, 0:60], ave_location_between[:,72:96]), axis = 1)
    #         ave_phase_between = np.concatenate((ave_phase_between[:, 0:60], ave_phase_between[:,72:96]), axis = 1)
            


    # if mouse_recday == 'me11_01122021_02122021':#mouse b
    #     # get rid of the last task because it looks somewhat whacky
    #     ave_clocks_between = ave_clocks_between[:,0:-12].copy()
    #     ave_phase_between = ave_phase_between[:,0:-12].copy()
    #     ave_midnight_between = ave_midnight_between[:,0:-12].copy()
    #     ave_location_between = ave_location_between[:,0:-12].copy()
    #     ave_neurons_between = ave_neurons_between[:, 0:-12].copy()
    #     if ignore_double_tasks == 1:
    #         ave_clocks_between = np.concatenate((ave_clocks_between[:, 0:36], ave_clocks_between[:, 48:-12]), axis = 1)
    #         ave_neurons_between = np.concatenate((ave_neurons_between[:, 0:36], ave_neurons_between[:,48:-12]), axis = 1)
    #         ave_midnight_between = np.concatenate((ave_midnight_between[:, 0:36], ave_midnight_between[:,48:-12]), axis = 1)
    #         ave_location_between = np.concatenate((ave_location_between[:, 0:36], ave_location_between[:,48:-12]), axis = 1)
    #         ave_phase_between = np.concatenate((ave_phase_between[:, 0:36], ave_phase_between[:,48:-12]), axis = 1)

            
            
        
    # if mouse_recday == 'me10_09122021_10122021': #mouse c 
    #     # same tasks are: 1,4; and  5,6,9
    #     # 4 and 9 look whacky, so remove those
    #     # so then after removal 4 and 5 are the same 
    #     ave_clocks_between = np.concatenate((ave_clocks_between[:, 0:36], ave_clocks_between[:, 48:96]), axis = 1)
    #     ave_neurons_between = np.concatenate((ave_neurons_between[:, 0:36], ave_neurons_between[:,48:96]), axis = 1)
    #     ave_midnight_between = np.concatenate((ave_midnight_between[:, 0:36], ave_midnight_between[:,48:96]), axis = 1)
    #     ave_location_between = np.concatenate((ave_location_between[:, 0:36], ave_location_between[:,48:96]), axis = 1)
    #     ave_phase_between = np.concatenate((ave_phase_between[:, 0:36], ave_phase_between[:,48:96]), axis = 1)
    #     # consider also removing the penultimum one... this was before task 7, now it is 6
    #     # so far this is still inside
    #     if ignore_double_tasks == 1:
    #         ave_clocks_between = np.concatenate((ave_clocks_between[:, 0:36], ave_clocks_between[:, 60:96]), axis = 1)
    #         ave_neurons_between = np.concatenate((ave_neurons_between[:, 0:36], ave_neurons_between[:,60:96]), axis = 1)
    #         ave_midnight_between = np.concatenate((ave_midnight_between[:, 0:36], ave_midnight_between[:,60:96]), axis = 1)
    #         ave_location_between = np.concatenate((ave_location_between[:, 0:36], ave_location_between[:,60:96]), axis = 1)
    #         ave_phase_between = np.concatenate((ave_phase_between[:, 0:36], ave_phase_between[:,60:96]), axis = 1)
                
            
    # # if mouse_recday == 'me08_10092021_11092021': #mouse d 
    # # same tasks: 1, 4
    # # ALL FINE WITH d ONCE THE LAST BUT THE LAST EPHYS FILE WAS LOST 
    
    # if mouse_recday == 'ah04_09122021_10122021': #mouse e range 0,8
    # # throw out the 4th 
    # # same tasks: (all tasks are unique, before 1 and 4 were the same but 4 is gone)
    #     ave_clocks_between = np.concatenate((ave_clocks_between[:, 0:36], ave_clocks_between[:, 48::]), axis = 1)
    #     ave_neurons_between = np.concatenate((ave_neurons_between[:, 0:36], ave_neurons_between[:,48::]), axis = 1)
    #     ave_midnight_between = np.concatenate((ave_midnight_between[:, 0:36], ave_midnight_between[:,48::]), axis = 1)
    #     ave_location_between = np.concatenate((ave_location_between[:, 0:36], ave_location_between[:,48::]), axis = 1)
    #     ave_phase_between = np.concatenate((ave_phase_between[:, 0:36], ave_phase_between[:,48::]), axis = 1)

    # if mouse_recday == 'ah04_05122021_06122021': #mouse f range 0,8
    # # throw out number 4
    # # new 4 (previous 5) and last one - 7 (previous 8) are the same
    #     ave_clocks_between = np.concatenate((ave_clocks_between[:, 0:36], ave_clocks_between[:, 48::]), axis = 1)
    #     ave_neurons_between = np.concatenate((ave_neurons_between[:, 0:36], ave_neurons_between[:,48::]), axis = 1)
    #     ave_midnight_between = np.concatenate((ave_midnight_between[:, 0:36], ave_midnight_between[:,48::]), axis = 1)
    #     ave_location_between = np.concatenate((ave_location_between[:, 0:36], ave_location_between[:,48::]), axis = 1)
    #     ave_phase_between = np.concatenate((ave_phase_between[:, 0:36], ave_phase_between[:,48::]), axis = 1)
            
    # #if mouse_recday == 'ah04_01122021_02122021': #mouse g range 0,8
    # # same tasks: 1,4 and 5,8
    # # ALL FINE WITH g 
    # import pdb; pdb.set_trace()
    
       
    # if mouse_recday == 'ah03_18082021_19082021': #mouse h range 0,8
    #     # hmmmm here I am not sure... maybe it is alright??
    #     # the fourth task looks a bit off, but I am leaving it in for now
    #     # same tasks: 1,4 and 5,8
    #     print('yey')

