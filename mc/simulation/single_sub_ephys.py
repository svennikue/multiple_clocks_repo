#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:10:30 2023

this module can be called to prepare raw ephys data and then runs the single-subject linear regression.

@author: Svenja Küchenhoff
"""

import math
import numpy as np
import mc

def reg_per_task_config(task_configs, locations_all, neurons, timings_all, contrast_m):
    # import pdb; pdb.set_trace()
    # mouse a
    contrast_m = np.array(contrast_m)
    coefficient = list()
    contrast_results_all = list()
    for task_no, task_config in enumerate(task_configs):
        # first convert trial times from ms to bin number to match neuron and location arrays 
        # (1 bin = 25ms)
        timings_task = timings_all[task_no]
        for r, row in enumerate(timings_task):
            for c, element in enumerate(row):
                timings_task[r,c] = element/25
    
        # second, change locations and rewards to 0 and ignoring bridges
        locations_task = locations_all[task_no]
        for i, field in enumerate(locations_task):
            if field > 9: 
                locations_task[i] = locations_task[i-1]
            if math.isnan(field):
                # keep the location bc of timebins
                locations_task[i] = locations_task[i-1]
                    
    
        # important: fields need to be between 0 and 8, and keep them as integers!
        locations_task = [int((field_no-1)) for field_no in locations_task]
        task_config = [int((field_no-1)) for field_no in task_config]

        # start of the second loop: loop through every single trial for this specific task config.
        coefficients_per_trial = np.zeros((len(timings_task),len(contrast_m[0])))
        for trial_no, row in enumerate(timings_task):
            
            # current data of this specific run
            trajectory = locations_task[row[0]:row[-1]]
        

            curr_neurons = neurons[task_no][:,row[0]:row[-1]]
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
                    
                    
            location_model = mc.simulation.predictions.set_location_raw_ephys(trajectory, step_time = 1, grid_size=3, plotting = False, field_no_given= 1)
            midnight_model, clocks_model = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=False)
            phase_model = mc.simulation.predictions.set_phase_model_ephys(trajectory, timings_curr_run, index_make_step, step_number)


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
        contrast_results = np.zeros((len(contrast_m),len(coefficients_per_trial)))
        for contrast_no, contrast in enumerate(contrast_m):
            contrast_results[contrast_no,:] = np.matmul(contrast,coefficients_per_trial.transpose())
              
        # at the end of the trial, store the whole matrix in coefficient:
        coefficient.append(coefficients_per_trial)
        contrast_results_all.append(contrast_results)
        print(f"done with task_config {task_no}")

        
    return coefficient, contrast_results_all


def reg_across_tasks(task_configs, locations_all, neurons, timings_all):
    import pdb; pdb.set_trace()
    coefficient = list()
    
    # find out which is the largest shared trial number between all task configs
    min_trialno = 60
    for task_number in timings_all:
        curr_trialno = len(task_number)
        if curr_trialno < min_trialno:
            min_trialno = curr_trialno
     
    # bit ugly that 4 is hard-coded, this is just the number of regressors, i.e. number of RDMs/models
    coefficients_per_trial = np.zeros((min_trialno,4))
    
    for no_trial_in_each_task in range(0, min_trialno):
        for task_no, task_config in enumerate(task_configs):
            # first convert trial times from ms to bin number to match neuron and location arrays 
            # (1 bin = 25ms)
            timings_task = timings_all[task_no]
            for r, row in enumerate(timings_task):
                for c, element in enumerate(row):
                    timings_task[r,c] = element/25
        
            # second, change locations and rewards to 0 and ignoring bridges
            locations_task = locations_all[task_no]
            for i, field in enumerate(locations_task):
                if field > 9: 
                    locations_task[i] = locations_task[i-1]
                if math.isnan(field):
                    # keep the location bc of timebins
                    locations_task[i] = locations_task[i-1]
                        
        
            # important: fields need to be between 0 and 8, and keep them as integers!
            locations_task = [int((field_no-1)) for field_no in locations_task]
            task_config = [int((field_no-1)) for field_no in task_config]
    
            # start of the second loop: loop through every single trial for this specific task config.
            # FOR NOW, ONLY DO THIS ON THE LAST 3 TRIALS PER TASK!!
            # timings_trials_I_take = timings_task[-3:]
            # new try: only do this for the last trial per task
            # timings_trials_I_take = timings_task[-1:]
            # new try: take all, but always only one at a time >
            # find out the max shared trial number across tasks!
            
            timings_trials_I_take = timings_task[-(no_trial_in_each_task+1),:]
            
            # THIS IS PROBLEMATIC NOW THAT I TAKE ONLY ONE ROW
            # change temporarily
            #for trial_no, row in enumerate(timings_trials_I_take):
                # CHANGE BACK IF I TAKE SEVERAL TRIALS PER REGRESSION!!   
            row = timings_trials_I_take.copy()
            # current data of this specific run
            trajectory = locations_task[row[0]:row[-1]]
            curr_neurons = neurons[task_no][:,row[0]:row[-1]]
            # ISSUE 21.04.23:
            # if there are ONLY 0 for one timestep, the np.corrcoef will output nan for that instance. Maybe better:
            # replace by super super low value
            # Update 09.05.23: DON'T Do that! Rather ignore those values in correlation
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
                            
            location_model = mc.simulation.predictions.set_location_raw_ephys(trajectory, step_time = 1, grid_size=3, plotting = False, field_no_given= 1)
            midnight_model, clocks_model = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=False)
            phase_model = mc.simulation.predictions.set_phase_model_ephys(trajectory, timings_curr_run, index_make_step, step_number)


            # now create the regressors per run
            regs_phase_state_run = mc.simulation.predictions.create_regressors_per_state_phase_ephys(walked_path=trajectory, subpath_timings=timings_curr_run, step_no=step_number)
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
      
            
        # try how the stat_phase RDMS look like
        RSM_location_betas = mc.simulation.RDMs.within_task_RDM(location_between, plotting = True, titlestring = 'Location phase*state dim RSM')
        RSM_clock_betas = mc.simulation.RDMs.within_task_RDM(clocks_between, plotting = True, titlestring = 'Clock phase*state dim RSM')
        RSM_midnight_betas = mc.simulation.RDMs.within_task_RDM(midnight_between, plotting = True, titlestring = 'Midnight phase*state dim RSM')
        RSM_phase_betas = mc.simulation.RDMs.within_task_RDM(phase_between, plotting = True, titlestring = 'Phase phase*state dim RSM')
        RSM_neurons_betas = mc.simulation.RDMs.within_task_RDM(neurons_between, plotting = True, titlestring = 'Data phase*state dim RDM')
        
        results_reg, scipy_regression_results = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_betas, regressor_one_matrix=RSM_clock_betas, regressor_two_matrix= RSM_midnight_betas, regressor_three_matrix= RSM_location_betas, regressor_four_matrix= RSM_phase_betas)
        coefficients_per_trial[no_trial_in_each_task] = results_reg.coef_
        # at the end of the trial, store the whole matrix in coefficient:
        coefficient.append(coefficients_per_trial) 
    
    return results_reg, scipy_regression_results, coefficient


