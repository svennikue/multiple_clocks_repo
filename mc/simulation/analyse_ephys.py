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




def reg_between_tasks_singleruns(task_configs, locations_all, neurons, timings_all, contrast_m, mouse_recday, continuous = True, no_bins_per_state = 0, split_by_phase = 1, number_phase_neurons = 3):
    #import pdb; pdb.set_trace()
    # mouse a
    contrast_m = np.array(contrast_m)
    coefficient = list()
    contrast_results_all = list()

    
    # prepare result variabls
    reg_early_two = []
    reg_mid_two = []
    reg_late_two = []
    reg_early_all = []
    reg_mid_all = []
    reg_late_all = []
    tval_early = []
    tval_mid = []
    tval_late = []
    tval_early_all = []
    tval_mid_all = []
    tval_late_all = []
    

    # find out which is the largest shared trial number between all task configs
    min_trialno = 60
    for task_number in timings_all:
        curr_trialno = len(task_number)
        if curr_trialno < min_trialno:
            min_trialno = curr_trialno
    
    # based on the biggest shared run number,
    # always concatenate the first,...nth run of one task with all other tasks
    coefficients_per_trial = np.zeros((min_trialno,len(contrast_m[0])))
    contrast_results = np.zeros((len(contrast_m),len(coefficients_per_trial)))
    
    if split_by_phase == 1:
        contrast_results_early = np.zeros((len(contrast_m),len(coefficients_per_trial)))
        contrast_results_mid = np.zeros((len(contrast_m),len(coefficients_per_trial)))
        contrast_results_late = np.zeros((len(contrast_m),len(coefficients_per_trial)))
    
    for no_trial_in_each_task in range(0, min_trialno):
        for task_no, task_config in enumerate(task_configs):
            trajectory, timings_curr_run, index_make_step, step_number, curr_neurons = mc.simulation.analyse_ephys.prep_ephys_per_trial(timings_all, locations_all, no_trial_in_each_task, task_no, task_config, neurons)
                    
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
            

            
            # these need to be concatenated for each run and task
            if task_no == 0:
            #if task_no == 0 and trial_no == 0:
                neurons_between = curr_neurons.copy()
                clocks_between = clocks_model.copy()
                midnight_between = midnight_model.copy()
                location_between = location_model.copy()
                phase_between = phase_model.copy()
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
                # # check if I want to split by phase.
                if split_by_phase == 1:
                    phase_separation_temp = mc.simulation.predictions.set_phase_model_ephys(trajectory, timings_curr_run, index_make_step, step_number)
                    phase_separation_temp = np.round(mc.simulation.predictions.transform_data_to_betas(phase_separation_temp, regs_phase_state_run))
                    phase_separation = np.concatenate((phase_separation, phase_separation_temp), axis = 1)

        
        if split_by_phase == 0:
            import pdb; pdb.set_trace()
            # then, for the RDMs which concatenate th nth run of each task config, create between-task RDMs
            # now create the model RDMs
            RSM_location = mc.simulation.RDMs.within_task_RDM(location_between, plotting = False, titlestring = 'Location RDM')
            RSM_clock = mc.simulation.RDMs.within_task_RDM(clocks_between, plotting = False, titlestring = 'Clock RDM')
            RSM_midnight = mc.simulation.RDMs.within_task_RDM(midnight_between, plotting = False, titlestring = 'Midnight RDM')
            RSM_phase = mc.simulation.RDMs.within_task_RDM(phase_between, plotting = False, titlestring = 'Phase RDM')
        
            # now create the data RDM
            RSM_neurons = mc.simulation.RDMs.within_task_RDM(neurons_between, plotting = False, titlestring = 'Data RDM')
        
            # Lastly, create a linear regression with RSM_loc,clock and midnight as regressors and data to be predicted
            results_reg, tvals = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons, regressor_one_matrix=RSM_clock, regressor_two_matrix= RSM_midnight, regressor_three_matrix= RSM_location, regressor_four_matrix= RSM_phase, t_val= 1)
            # print(f" The beta for the clocks model is {reg_res.coef_[0]}, for the midnight model is {reg_res.coef_[1]}, and for the location model is {reg_res.coef_[2]}")
            coefficients_per_trial[no_trial_in_each_task] = results_reg.coef_
            # print(f" Computed betas for run {trial_no} of task {task_config}")
            
            # then compute contrasts
            # I want to know: [0 0 1], [0 1 0], [1 0 0] and [-1 1 0], [0 -1 1], ....
        
            # # in case I want to have an overview of all betas for this trial config
            # x = np.linspace(0,len(coefficients_per_trial)-1,len(coefficients_per_trial))
            # plt.figure(); plt.plot(x, coefficients_per_trial[:,0], label = 'clocks'); plt.plot(x, coefficients_per_trial[:,1], label = 'midnight'); plt.plot(x, coefficients_per_trial[:,2], label = 'location'); plt.plot(x, coefficients_per_trial[:,3], label = 'phase'); plt.legend(loc="upper left"); plt.ylabel('beta'); plt.xlabel('run number'); plt.axhline(0, color='grey', ls='dashed'); plt.title(f"Recording day {mouse_recday} task {task_no}")
             
            
            for contrast_no, contrast in enumerate(contrast_m):
                contrast_results[contrast_no,no_trial_in_each_task] = np.matmul(contrast,coefficients_per_trial[no_trial_in_each_task].transpose())
                  
                
        
        elif split_by_phase == 1:
            # import pdb; pdb.set_trace()
            early_mask = np.where(phase_separation[0,:] == 1)[0]
            mid_mask = np.where(phase_separation[1,:] == 1)[0]
            late_mask = np.where(phase_separation[2,:] == 1)[0]
            
            
            # I don't think this is needed after all bc the regressions are done separetly.
            
            # # check if these are all the same lengths, and if not, drop the last datapoint
            # # CAREFUL! This is not very elegant... but necessary for the regression
            # if len(early_mask) != len(mid_mask) or len(early_mask) != len(late_mask) or len(late_mask) != len(mid_mask):
            #     min_length = min(len(early_mask), len(late_mask), len(mid_mask))
            #     early_mask = early_mask[0:min_length].copy()
            #     mid_mask = mid_mask[0:min_length].copy()
            #     late_mask = late_mask[0:min_length].copy()
                
                
            
            RSM_location_early = mc.simulation.RDMs.within_task_RDM(location_between[:, early_mask], plotting = False, titlestring = 'early Location RDM')
            RSM_location_mid = mc.simulation.RDMs.within_task_RDM(location_between[:, mid_mask], plotting = False, titlestring = 'mid Location RDM')
            RSM_location_late = mc.simulation.RDMs.within_task_RDM(location_between[:, late_mask], plotting = False, titlestring = 'late Location RDM')
            
            RSM_clocks_early = mc.simulation.RDMs.within_task_RDM(clocks_between[:,early_mask], plotting = False, titlestring = 'early Clock RDM')
            RSM_clocks_mid = mc.simulation.RDMs.within_task_RDM(clocks_between[:,mid_mask], plotting = False, titlestring = 'mid Clock RDM')
            RSM_clocks_late = mc.simulation.RDMs.within_task_RDM(clocks_between[:,late_mask], plotting = False, titlestring = 'late Clock RDM')
            
            RSM_midnight_early = mc.simulation.RDMs.within_task_RDM(midnight_between[:,early_mask], plotting = False, titlestring = 'early Midnight RDM')
            RSM_midnight_mid = mc.simulation.RDMs.within_task_RDM(midnight_between[:,mid_mask], plotting = False, titlestring = 'mid Midnight RDM')
            RSM_midnight_late = mc.simulation.RDMs.within_task_RDM(midnight_between[:,late_mask], plotting = False, titlestring = 'late Midnight RDM')
            
            RSM_phase_early = mc.simulation.RDMs.within_task_RDM(phase_between[:,early_mask], plotting = False, titlestring = 'early Phase RDM')
            RSM_phase_mid = mc.simulation.RDMs.within_task_RDM(phase_between[:,mid_mask], plotting = False, titlestring = 'mid Phase RDM')
            RSM_phase_late = mc.simulation.RDMs.within_task_RDM(phase_between[:,late_mask], plotting = False, titlestring = 'late Phase RDM')
        
        
            # now create the data RDM
            RSM_neurons_early = mc.simulation.RDMs.within_task_RDM(neurons_between[:,early_mask], plotting = False, titlestring = 'early Data RDM')
            RSM_neurons_mid = mc.simulation.RDMs.within_task_RDM(neurons_between[:,mid_mask], plotting = False, titlestring = 'mid Data RDM')
            RSM_neurons_late = mc.simulation.RDMs.within_task_RDM(neurons_between[:,late_mask], plotting = False, titlestring = 'late Data RDM')
        
            # then do the, this time 3, regressions.
            reg_early, tval_early_perrrun = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_early, regressor_one_matrix = RSM_midnight_early, regressor_two_matrix= RSM_clocks_early, t_val= 1)
            print(f"results for early trial_no {no_trial_in_each_task} are [midnight, clocks] {reg_early.coef_}")
            reg_mid, tval_mid_perrrun = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_mid, regressor_one_matrix = RSM_midnight_mid, regressor_two_matrix= RSM_clocks_mid, t_val= 1)
            print(f"results for mid trial_no {no_trial_in_each_task} are [midnight, clocks] {reg_mid.coef_}")
            reg_late, tval_late_perrrun = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_late, regressor_one_matrix = RSM_midnight_late, regressor_two_matrix= RSM_clocks_late, t_val= 1)
            print(f"results for late trial_no {no_trial_in_each_task} are [midnight, clocks] {reg_late.coef_}")
            
            # to compare [might delete later] also check with phase as regressor
            reg_early_with_phase, tval_early_with_phase_perrrun = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_early, regressor_one_matrix= RSM_clocks_early, regressor_two_matrix= RSM_midnight_early, regressor_three_matrix= RSM_location_early, regressor_four_matrix= RSM_phase_early, t_val= 1)
            print(f"results for early trial_no {no_trial_in_each_task} are [clocks, midnight, loc, phase] {reg_early_with_phase.coef_}")
            reg_mid_with_phase, tval_mid_with_phase_perrrun= mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_mid, regressor_one_matrix= RSM_clocks_mid, regressor_two_matrix= RSM_midnight_mid, regressor_three_matrix= RSM_location_mid, regressor_four_matrix= RSM_phase_mid, t_val= 1)
            print(f"results for mid trial_no {no_trial_in_each_task} are [clocks, midnight, loc, phase] {reg_mid_with_phase.coef_}")
            reg_late_with_phase, tval_late_with_phase_perrrun = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_late, regressor_one_matrix= RSM_clocks_late, regressor_two_matrix= RSM_midnight_late, regressor_three_matrix= RSM_location_late, regressor_four_matrix= RSM_phase_late, t_val= 1)
            print(f"results for late trial_no {no_trial_in_each_task} are [clocks, midnight, loc, phase] {reg_late_with_phase.coef_}")
            
            
            # then do the contrasts only for the complete model, otherwise it doesnt work
            for contrast_no, contrast in enumerate(contrast_m):
                contrast_results_early[contrast_no,no_trial_in_each_task] = np.matmul(contrast,reg_early_with_phase.coef_.transpose())
            for contrast_no, contrast in enumerate(contrast_m):
                contrast_results_mid[contrast_no,no_trial_in_each_task] = np.matmul(contrast,reg_mid_with_phase.coef_.transpose())
            for contrast_no, contrast in enumerate(contrast_m):
                contrast_results_late[contrast_no,no_trial_in_each_task] = np.matmul(contrast,reg_late_with_phase.coef_.transpose())
            reg_early_two.append(reg_early.coef_)
            reg_mid_two.append(reg_mid.coef_)
            reg_late_two.append(reg_late.coef_)
            reg_early_all.append(reg_early_with_phase.coef_)
            reg_mid_all.append(reg_mid_with_phase.coef_)
            reg_late_all.append(reg_late_with_phase.coef_)
            tval_early.append(tval_early_perrrun)
            tval_mid.append(tval_mid_perrrun)
            tval_late.append(tval_late_perrrun)
            tval_early_all.append(tval_early_with_phase_perrrun)
            tval_mid_all.append(tval_mid_with_phase_perrrun)
            tval_late_all.append(tval_late_with_phase_perrrun)
            
            
    # so right now I am not saving the individual results per loop except for the contrasts.
    result = {}
    if split_by_phase == 0:
        result["coefficients_per_trial"] = coefficients_per_trial
        result["contrast_results"] = contrast_results
        result["t-values"] = tvals
    if split_by_phase == 1:
        result["early_without_phase"] = reg_early
        result["mid_without_phase"] = reg_mid
        result["late_without_phase"] = reg_late
        result["early_with_phase"] = reg_early_all
        result["mid_with_phase"] = reg_mid_all
        result["late_with_phase"] = reg_late_all
        result["contrast_early"] = contrast_results_early
        result["contrast_mid"] = contrast_results_mid
        result["contrast_late"] = contrast_results_late
        result["tval_early_without_phase"] = tval_early
        result["tval_mid_without_phase"] = tval_mid
        result["tval_late_without_phase"] = tval_late
        result["tval_early_with_phase"] = tval_early_all
        result["tval_mid_with_phase"] = tval_mid_all
        result["tval_late_with_phase"] = tval_late_all

        # at the end of the trial, store the whole matrix in coefficient:
        #coefficient.append(coefficients_per_trial)
        print(f"done with trial_no {no_trial_in_each_task}")
        
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
def reg_across_tasks(task_configs, locations_all, neurons, timings_all, mouse_recday, plotting = False, continuous = True, no_bins_per_state = 3, number_phase_neurons = 3):
    import pdb; pdb.set_trace()
    # this is now all  based on creating an average across runs first.
    z_score_all = 1 # 2 means only z-score the data

    # find out which is the largest shared trial number between all task configs
    min_trialno = 60
    for task_number in timings_all:
        curr_trialno = len(task_number)
        if curr_trialno < min_trialno:
            min_trialno = curr_trialno

    for no_trial_in_each_task in range(0, min_trialno):
        for task_no, task_config in enumerate(task_configs):
            trajectory, timings_curr_run, index_make_step, step_number, curr_neurons = mc.simulation.analyse_ephys.prep_ephys_per_trial(timings_all, locations_all, no_trial_in_each_task, task_no, task_config, neurons)
            
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
                location_model, phase_model, state_model, midnight_model, clocks_model, phase_state_model = mc.simulation.predictions.set_continous_models_ephys(trajectory, timings_curr_run, index_make_step, step_number, no_phase_neurons= number_phase_neurons, plot = True)
            
            
            # now create the regressors per run
            regs_phase_state_run = mc.simulation.predictions.create_x_regressors_per_state(walked_path = trajectory, subpath_timings = timings_curr_run, step_no = step_number, no_regs_per_state = no_bins_per_state)
            # then use these regressors to generate a beta per neuron per run
            neurons_phase_state = mc.simulation.predictions.transform_data_to_betas(curr_neurons, regs_phase_state_run)
            clock_phase_state = mc.simulation.predictions.transform_data_to_betas(clocks_model, regs_phase_state_run)
            midnight_phase_state= mc.simulation.predictions.transform_data_to_betas(midnight_model, regs_phase_state_run)
            location_phase_state = mc.simulation.predictions.transform_data_to_betas(location_model, regs_phase_state_run)
            phase_phase_state = mc.simulation.predictions.transform_data_to_betas(phase_model, regs_phase_state_run)
            
            # these need to be concatenated for each run and task
            if task_no == 0:
            #if task_no == 0 and trial_no == 0:
                neurons_between = neurons_phase_state.copy()
                clocks_between = clock_phase_state.copy()
                midnight_between = midnight_phase_state.copy()
                location_between = location_phase_state.copy()
                phase_between = phase_phase_state.copy()
            else:
                neurons_between = np.concatenate((neurons_between, neurons_phase_state), axis = 1)
                clocks_between = np.concatenate((clocks_between, clock_phase_state), axis = 1)
                midnight_between = np.concatenate((midnight_between, midnight_phase_state), axis = 1)
                location_between = np.concatenate((location_between, location_phase_state), axis = 1)
                phase_between = np.concatenate((phase_between,phase_phase_state), axis = 1)
        
        if z_score_all == 1:
            # now z-score all matrices.
            neurons_between_z = scipy.stats.zscore(neurons_between, axis=1)
            clocks_between_z = scipy.stats.zscore(clocks_between, axis=1)
            midnight_between_z = scipy.stats.zscore(midnight_between, axis=1)
            location_between_z = scipy.stats.zscore(location_between, axis=1)
            phase_between_z = scipy.stats.zscore(phase_between, axis=1)
            # potentially not z-score the models. Now the reasoning is that I want to treat both- actual neurons 
            # and simulated neurons - the same.

            
            if no_trial_in_each_task == 0:
                # create an averaged neuron file and RDM. 
                # it's possible that the mice take different routes every time. By averaging, 
                # I should be able to account for these variations.
                sum_location_between = location_between_z.copy()
                sum_clocks_between = clocks_between_z.copy()
                sum_midnight_between = midnight_between_z.copy()
                sum_phase_between = phase_between_z.copy()
                sum_neurons_between = neurons_between_z.copy()
            if no_trial_in_each_task > 0:
                sum_location_between = sum_location_between.copy() + location_between_z.copy()
                sum_clocks_between = sum_clocks_between.copy() + clocks_between_z.copy()
                sum_midnight_between = sum_midnight_between.copy() + midnight_between_z.copy()
                sum_phase_between = sum_phase_between.copy() + phase_between_z.copy()
                sum_neurons_between = sum_neurons_between.copy() + neurons_between_z.copy()
        
        if z_score_all == 0:
            # create an averaged neuron file and RDM. 
            if no_trial_in_each_task == 0:
                sum_location_between = location_between.copy()
                sum_clocks_between = clocks_between.copy()
                sum_midnight_between = midnight_between.copy()
                sum_phase_between = phase_between.copy()
                sum_neurons_between = neurons_between.copy()
            if no_trial_in_each_task > 0:
                sum_location_between = sum_location_between.copy() + location_between.copy()
                sum_clocks_between = sum_clocks_between.copy() + clocks_between.copy()
                sum_midnight_between = sum_midnight_between.copy() + midnight_between.copy()
                sum_phase_between = sum_phase_between.copy() + phase_between.copy()
                sum_neurons_between = sum_neurons_between.copy() + neurons_between.copy()
        
        if z_score_all == 2:
            # z-score the data matrix
            neurons_between_z = scipy.stats.zscore(neurons_between, axis=1)
            
            # then create the average
            if no_trial_in_each_task == 0:
                sum_location_between = location_between.copy()
                sum_clocks_between = clocks_between.copy()
                sum_midnight_between = midnight_between.copy()
                sum_phase_between = phase_between.copy()
                sum_neurons_between = neurons_between_z.copy()
            if no_trial_in_each_task > 0:
                sum_location_between = sum_location_between.copy() + location_between.copy()
                sum_clocks_between = sum_clocks_between.copy() + clocks_between.copy()
                sum_midnight_between = sum_midnight_between.copy() + midnight_between.copy()
                sum_phase_between = sum_phase_between.copy() + phase_between.copy()
                sum_neurons_between = sum_neurons_between.copy() + neurons_between_z.copy()
        
    ave_location_between = sum_location_between/no_trial_in_each_task
    ave_clocks_between = sum_clocks_between/no_trial_in_each_task
    ave_midnight_between = sum_midnight_between/no_trial_in_each_task
    ave_phase_between = sum_phase_between/no_trial_in_each_task
    ave_neurons_between = sum_neurons_between/no_trial_in_each_task


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
    
    if plotting == True:
        # plot the averaged simulated and cleaned data
        mc.simulation.predictions.plot_without_legends(ave_location_between, titlestring= 'location average', intervalline= 12)
        mc.simulation.predictions.plot_without_legends(ave_clocks_between, titlestring= 'clock average', intervalline= 12)
        mc.simulation.predictions.plot_without_legends(ave_midnight_between, titlestring= 'midnight average', intervalline= 12)
        mc.simulation.predictions.plot_without_legends(ave_phase_between, titlestring= 'phase average', intervalline= 12)
        mc.simulation.predictions.plot_without_legends(ave_neurons_between, titlestring= 'neuron average', intervalline= 12)
    
        RSM_location_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_location_between, plotting = True, titlestring = 'Between tasks Location RSM, 12*12, averaged over runs')
        RSM_clock_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_clocks_between, plotting = True, titlestring = 'Between tasks Clocks RSM, 12*12, averaged over runs', intervalline= 12)
        RSM_midnight_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_midnight_between, plotting = True, titlestring = 'Between tasks Midnight RSM, 12*12, averaged over runs')
        RSM_phase_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_phase_between, plotting = True, titlestring = 'Between tasks Pgase RSM, 12*12, averaged over runs')
        RSM_neurons_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_neurons_between, plotting = True, titlestring = 'Between tasks Data RSM, 12*12, averaged over runs')
        
        # separately per phase
        mc.simulation.predictions.plot_without_legends(ave_clocks_between[:, 0::3], titlestring='early clocks across tasks', intervalline= 4)
        mc.simulation.predictions.plot_without_legends(ave_neurons_between[:, 0::3], titlestring='early neurons across tasks', intervalline= 4)
        mc.simulation.predictions.plot_without_legends(ave_midnight_between[:, 0::3], titlestring='early midnight across tasks', intervalline= 4)
        
        mc.simulation.predictions.plot_without_legends(ave_clocks_between[:, 1::3], titlestring='mid clocks across tasks', intervalline= 4)
        mc.simulation.predictions.plot_without_legends(ave_neurons_between[:, 1::3], titlestring='mid neurons across tasks', intervalline= 4)
        mc.simulation.predictions.plot_without_legends(ave_midnight_between[:, 1::3], titlestring='mid midnight across tasks', intervalline= 4)
        
        mc.simulation.predictions.plot_without_legends(ave_clocks_between[:, 2::3], titlestring='late clocks across tasks', intervalline= 4)
        mc.simulation.predictions.plot_without_legends(ave_neurons_between[:, 2::3], titlestring='late neurons across tasks', intervalline= 4)
        mc.simulation.predictions.plot_without_legends(ave_midnight_between[:, 2::3], titlestring='late midnight across tasks', intervalline= 4)
        
        RSM_early_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, 0::3], plotting=True, titlestring='RSM early clocks, averaged over runs', intervalline= 4)
        RSM_early_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between[:, 0::3], plotting=True, titlestring='RSM early neurons, averaged over runs', intervalline= 4)
        RSM_early_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, 0::3], plotting=True, titlestring='RSM early midnight, averaged over runs', intervalline= 4)
        
        RSM_mid_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, 1::3], plotting=True, titlestring='RSM mid clocks, averaged over runs', intervalline= 4)
        RSM_mid_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between[:, 1::3], plotting=True, titlestring='RSM mid neurons, averaged over runs', intervalline= 4)
        RSM_mid_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, 1::3], plotting=True, titlestring='RSM mid midnight, averaged over runs', intervalline= 4)
        
        RSM_late_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, 2::3], plotting=True, titlestring='RSM late clocks, averaged over runs', intervalline= 4)
        RSM_late_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between[:, 2::3], plotting=True, titlestring='RSM late neurons, averaged over runs', intervalline= 4)
        RSM_late_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, 2::3], plotting=True, titlestring='RSM late midnight, averaged over runs', intervalline= 4)
        
    
    elif plotting == False: 
        # for all phases
        RSM_location_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_location_between, plotting = False)
        RSM_clock_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_clocks_between, plotting = False)
        RSM_midnight_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_midnight_between, plotting = False)
        RSM_phase_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_phase_between, plotting = False)
        RSM_neurons_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_neurons_between, plotting = False)
        
        # for each phase separately
        RSM_early_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, 0::3], plotting = False)
        RSM_early_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between[:, 0::3], plotting = False)
        RSM_early_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, 0::3], plotting = False)
        
        RSM_mid_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, 1::3], plotting = False)
        RSM_mid_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between[:, 1::3], plotting = False)
        RSM_mid_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, 1::3], plotting = False)
        
        RSM_late_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, 2::3], plotting = False)
        RSM_late_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between[:, 2::3], plotting = False)
        RSM_late_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, 2::3], plotting = False)
        
    
    # run regressions separetly for each phase
    reg_early, scipy_early = mc.simulation.RDMs.lin_reg_RDMs(RSM_early_neuron, regressor_one_matrix = RSM_early_midnight, regressor_two_matrix= RSM_early_clocks)
    print('results for early are [midnight, clocks]', reg_early.coef_)
    reg_mid, scipy_mid = mc.simulation.RDMs.lin_reg_RDMs(RSM_mid_neuron, regressor_one_matrix = RSM_mid_midnight, regressor_two_matrix= RSM_mid_clocks)
    print('results for mid are [midnight, clocks]', reg_mid.coef_)
    reg_late, scipy_late = mc.simulation.RDMs.lin_reg_RDMs(RSM_late_neuron, regressor_one_matrix = RSM_late_midnight, regressor_two_matrix= RSM_late_clocks)
    print('results for late are [midnight, clocks]', reg_late.coef_)

    # LITTLE REGRESSION PLAYAROUND
    # this is a regression where I put all early phases of all tasks behind each other
    # then all mid phases of each tasks, etc; and then do the regression.
    from sklearn.linear_model import LinearRegression
    
    dimension = len(RSM_early_clocks)
    Yearly = list(RSM_early_neuron[np.tril_indices(dimension , -1)])
    Ymid = list(RSM_mid_neuron[np.tril_indices(dimension , -1)])
    Ylate = list(RSM_late_neuron[np.tril_indices(dimension , -1)])
    Yall = np.hstack((Yearly, Ymid, Ylate))
    
    Xearly = list(RSM_early_midnight[np.tril_indices(dimension, -1)])
    Xclock_early = list(RSM_early_clocks[np.tril_indices(dimension, -1)])
    Xearly = np.vstack((Xearly, Xclock_early))
    
    Xmid = list(RSM_mid_midnight[np.tril_indices(dimension, -1)])
    Xclock_mid = list(RSM_mid_clocks[np.tril_indices(dimension, -1)])
    Xmid = np.vstack((Xmid, Xclock_mid))
    
    Xlate = list(RSM_late_midnight[np.tril_indices(dimension, -1)])
    Xclock_late = list(RSM_late_clocks[np.tril_indices(dimension, -1)])
    Xlate = np.vstack((Xlate, Xclock_late))
    
    Xall = np.hstack((Xearly, Xmid, Xlate))
    x_all_reshaped = np.transpose(Xall)
    reversed_phases_reg_results = LinearRegression().fit(x_all_reshaped, Yall)
    print('results for putting all neurons together are [midnight, clocks]', reversed_phases_reg_results.coef_)

    # do whole regression with only midnight and clock
    reg_mid_clock, scipyblah = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_betas_ave, regressor_one_matrix = RSM_midnight_betas_ave, regressor_two_matrix= RSM_clock_betas_ave)
    print('results all normal RSMs are [midnight, clocks]', reg_mid_clock.coef_)
    
    # regression with all models
    results_average, scipy_regression_results = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_betas_ave, regressor_one_matrix=RSM_midnight_betas_ave, regressor_two_matrix= RSM_clock_betas_ave, regressor_three_matrix= RSM_location_betas_ave, regressor_four_matrix= RSM_phase_betas_ave)
    print('regression results are: [midnight, clocks,location, phase]:', results_average.coef_)

    
    # collect all values in a results table which I then output in the end.
    result_dict = {}
    result_dict['reg_early_phase_midnight-clocks'] = reg_early.coef_
    result_dict['reg_mid_phase_midnight-clocks'] = reg_mid.coef_
    result_dict['reg_late_phase_midnight-clocks'] = reg_late.coef_
    result_dict['reg_all_midnight-clocks'] = reg_mid_clock.coef_
    result_dict['reg_all_midnight-clocks-loc-phase'] = results_average.coef_
    result_dict['reg_all_reversedphase_midnight-clocks'] = reversed_phases_reg_results.coef_
    
    
    return result_dict





def load_ephys_data(Data_folder):
    
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
    a_no_task_configs = len(mouse_a["rewards_configs"])
    mouse_a["cells"] = np.load(Data_folder+'Phase_state_place_anchored_' + mouse_recday + '.npy')
    a_locations = list()
    a_neurons = list()
    a_timings = list()
    for session in range(0, a_no_task_configs):
        a_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        a_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
        a_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))
    mouse_a["locations"] = a_locations
    mouse_a["neurons"] = a_neurons
    mouse_a["timings"] = a_timings
    
    
    

    mouse_recday='me11_01122021_02122021' #mouse b 
    mouse_b["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
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


    mouse_recday='me10_09122021_10122021' #mouse c range 0,9
    mouse_c["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
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
        

    mouse_recday='me08_10092021_11092021' #mouse d range 0,6
    mouse_d["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
    mouse_d["rewards_configs"] = mouse_d["rewards_configs"][0:-1, :].copy()
    # apparently there is one run less for this day..., so exclude that one
    # mohammady says: The ephys file for the last task on that day was lost
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


    mouse_recday='ah04_09122021_10122021' #mouse e range 0,8
    mouse_e["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
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
        
     
        
    mouse_recday='ah04_05122021_06122021' #mouse f range 0,8
    mouse_f["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
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


    mouse_recday='ah04_01122021_02122021' #mouse g range 0,8
    mouse_g["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
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


    mouse_recday='ah03_18082021_19082021' #mouse h range 0,8
    mouse_h["rewards_configs"] = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
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
    # # for h, the first timings array is missing
    # # > delete the first task completely!
    # h_timings = h_timings[1::]
    # h_neurons = h_neurons[1::]
    # h_locations = h_locations[1::]
    # h_rewards_configs = h_rewards_configs[1::, :]
    return(mouse_a, mouse_b, mouse_c, mouse_d, mouse_e, mouse_f, mouse_g, mouse_h)

def clean_ephys_data(task_configs, locations_all, neurons, timings_all, mouse_recday, ignore_double_tasks = 1):
    # first clean data.
    if mouse_recday == 'me11_05122021_06122021': #mouse a
    # ALL FINE WITH a!
        # task 5 and 9 are the same, as well as 6 and 7
        # data of the first 4 tasks look similar, and tasks 5,6,7,8,9 look more similar
        if ignore_double_tasks == 1:
        # task 5 and 9 are the same, as well as 6 and 7
        # throw out 6 and 9
        # 1 and 4 are nearly the same, but have a different last field... so I leave them in.
            ignore = [8,5]
                
    if mouse_recday == 'me11_01122021_02122021':#mouse b
        
        if ignore_double_tasks == 1:
            ignore = [-1, 3]
            # and task 4 appears twice
        elif ignore_double_tasks == 0:
            ignore = [-1]
            # get rid of the last task because it looks somewhat whacky
        
    if mouse_recday == 'me10_09122021_10122021':#mouse c 
        if ignore_double_tasks == 1:
            ignore = [8,3,4]
        elif ignore_double_tasks == 0:
            ignore = [8,3]
        # same tasks are: 1,4; and  5,6,9
        # 4 and 9 look whacky, so remove those
        # so then after removal 5 and 6 are still the same and 5 has only 6 repeats
        # consider also removing the penultimum one... this was before task 7, now it is 6
        # so far this is still inside
         
    if mouse_recday == 'me08_10092021_11092021': #mouse d
        if ignore_double_tasks == 1:
            ignore = [3]
            # DOUBLE CHECK THIS!!!
    # same tasks: 1, 4
    # ALL FINE WITH d ONCE THE LAST BUT THE LAST EPHYS FILE WAS LOST > deleted this before

    if mouse_recday == 'ah04_09122021_10122021': #mouse e range 0,8
    # throw out the 4th 
    # same tasks: (all tasks are unique, before 1 and 4 were the same but 4 is gone)
        ignore = [4]
    
    if mouse_recday == 'ah04_05122021_06122021': #mouse f range 0,8
    # throw out number 4
    # new 4 (previous 5) and last one - 7 (previous 8) are the same
        if ignore_double_tasks == 1:
            ignore = [-1, 3]
        elif ignore_double_tasks == 0:
            ignore = [3]
     
    if mouse_recday == 'ah04_01122021_02122021': #mouse g range 0,8
    # same tasks: 1,4 and 5,8
    # ALL FINE WITH g 
        if ignore_double_tasks == 1:
            ignore = [4, 0]
        
    if mouse_recday == 'ah03_18082021_19082021': #mouse h range 0,8
        if ignore_double_tasks == 1:
            ignore = [4, 0]
    # hmmmm here I am not sure... maybe it is alright??
    # the fourth task looks a bit off, but I am leaving it in for now
    # same tasks: 1,4 and 5,8
   
    task_configs_clean = [elem for elem in task_configs]
    locations_all_clean = locations_all.copy()
    neurons_clean = neurons.copy()
    timings_all_clean = timings_all.copy()
    
    for ignore_task in ignore:
        task_configs_clean.pop(ignore_task)
        locations_all_clean.pop(ignore_task)
        neurons_clean.pop(ignore_task)
        timings_all_clean.pop(ignore_task)
        
    return(task_configs_clean,locations_all_clean,neurons_clean,timings_all_clean)




def prep_ephys_per_trial(timings_all, locations_all, no_trial_in_each_task, task_no, task_config, neurons):
    # first convert trial times from ms to bin number to match neuron and location arrays 
    # (1 bin = 25ms)
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
    row =  timings_task[-(no_trial_in_each_task+1),:].copy()
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
        
    return(trajectory, timings_curr_run, index_make_step, step_number, curr_neurons)



############################
### PLOTTING FUNCTIONS #####
############################

def plotting_hist_scat(data_list, label_string_list, label_tick_list, title_string):
    fig, ax = plt.subplots()
    ax.boxplot(data_list)
    for index, contrast in enumerate(data_list):
        ax.scatter(np.ones(len(contrast))+index, contrast)
    ax.set_xticks(label_tick_list)
    plt.xticks(rotation = 45)
    ax.set_xticklabels(label_string_list)
    plt.axhline(0, color='grey', ls='dashed')
    plt.title(title_string)
    
    




