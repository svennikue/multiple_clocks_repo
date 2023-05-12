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


def reg_across_tasks(task_configs, locations_all, neurons, timings_all, mouse_recday):
    # import pdb; pdb.set_trace()
    coefficient = list()
    
    # find out which is the largest shared trial number between all task configs
    min_trialno = 60
    for task_number in timings_all:
        curr_trialno = len(task_number)
        if curr_trialno < min_trialno:
            min_trialno = curr_trialno
     
    # bit ugly that 4 is hard-coded, this is just the number of regressors, i.e. number of RDMs/models
    # I am checking all trials across all tasks. because they have different amounts of trials, take the smalles shared minimum
    coefficients_per_trial = np.zeros((min_trialno,4))
    
    for no_trial_in_each_task in range(0, min_trialno):
        for task_no, task_config in enumerate(task_configs):
            # Some_configs are the same!! e.g. mouse b: 0 and 3 are the same
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
            
            timings_trials_I_take = timings_task[-(no_trial_in_each_task+1),:].copy()
            
            # THIS IS PROBLEMATIC NOW THAT I TAKE ONLY ONE ROW
            # change temporarily
            #for trial_no, row in enumerate(timings_trials_I_take):
                # CHANGE BACK IF I TAKE SEVERAL TRIALS PER REGRESSION!!   
            row = timings_trials_I_take.copy()
            # current data of this specific run
            trajectory = locations_task[row[0]:row[-1]].copy()
            curr_neurons = neurons[task_no][:,row[0]:row[-1]].copy()
            # ISSUE 21.04.23:
            # if there are ONLY 0 for one timestep, the np.corrcoef will output nan for that instance. Maybe better:
            # replace by super super low value
            # Update 09.05.23: DON'T Do that! Rather ignore those values in correlation
            # for col_no, column in enumerate(curr_neurons.T):
            #     if np.all(column == 0):
            #         curr_neurons[:,col_no] = 0.00001

        
            # some pre-processing to create my models.
            # to count subpaths
            subpath_file = [locations_task[row[0]:row[1]+1], locations_task[row[1]+1:row[2]+1], locations_task[row[2]+1:row[3]+1], locations_task[row[3]+1:row[4]+1]].copy()
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
        RSM_location_betas = mc.simulation.RDMs.within_task_RDM(location_between, plotting = False, titlestring = 'Location phase*state dim RSM')
        RSM_clock_betas = mc.simulation.RDMs.within_task_RDM(clocks_between, plotting = False, titlestring = 'Clock phase*state dim RSM')
        RSM_midnight_betas = mc.simulation.RDMs.within_task_RDM(midnight_between, plotting = False, titlestring = 'Midnight phase*state dim RSM')
        RSM_phase_betas = mc.simulation.RDMs.within_task_RDM(phase_between, plotting = False, titlestring = 'Phase phase*state dim RSM')
        RSM_neurons_betas = mc.simulation.RDMs.within_task_RDM(neurons_between, plotting = False, titlestring = 'Data phase*state dim RDM')
        
        
        # UNCOMMENT LATER!
        # results_reg, scipy_regression_results = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_betas, regressor_one_matrix=RSM_clock_betas, regressor_two_matrix= RSM_midnight_betas, regressor_three_matrix= RSM_location_betas, regressor_four_matrix= RSM_phase_betas)
        # coefficients_per_trial[no_trial_in_each_task] = results_reg.coef_
        # at the end of the trial, store the whole matrix in coefficient:
        # coefficient.append(coefficients_per_trial) 
        
        # additionally, create an averaged RDM. 
        # it's possible that the mice take different routes every time. By averaging, 
        # I should be able to account for these variations.
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
        
    ave_location_between = sum_location_between/no_trial_in_each_task
    ave_clocks_between = sum_clocks_between/no_trial_in_each_task
    ave_midnight_between = sum_midnight_between/no_trial_in_each_task
    ave_phase_between = sum_phase_between/no_trial_in_each_task
    ave_neurons_between = sum_neurons_between/no_trial_in_each_task

    
    # plot those that I am interested in
    mc.simulation.predictions.plot_without_legends(ave_location_between, titlestring= 'location average', intervalline= 12)
    mc.simulation.predictions.plot_without_legends(ave_clocks_between, titlestring= 'clock average', intervalline= 12)
    mc.simulation.predictions.plot_without_legends(ave_midnight_between, titlestring= 'midnight average', intervalline= 12)
    mc.simulation.predictions.plot_without_legends(ave_phase_between, titlestring= 'phase average', intervalline= 12)
    mc.simulation.predictions.plot_without_legends(ave_neurons_between, titlestring= 'neuron average', intervalline= 12)
    
    # task_one = ave_neurons_between[:, 0:12].copy()
    # task_four = ave_neurons_between[:,36:48].copy()
    # task_two = ave_neurons_between[:, 12:24].copy()
    # corr_one_two = mc.simulation.RDMs.corr_matrices_pearson(task_one, task_two)
    
    
    RSM_location_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_location_between, plotting = True, titlestring = 'Between tasks Location RSM, 12*12, averaged over runs')
    RSM_clock_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_clocks_between, plotting = True, titlestring = 'Between tasks Clocks RSM, 12*12, averaged over runs', intervalline= 12)
    RSM_midnight_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_midnight_between, plotting = True, titlestring = 'Between tasks Midnight RSM, 12*12, averaged over runs')
    RSM_phase_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_phase_between, plotting = True, titlestring = 'Between tasks Pgase RSM, 12*12, averaged over runs')
    RSM_neurons_betas_ave = mc.simulation.RDMs.within_task_RDM(ave_neurons_between, plotting = True, titlestring = 'Between tasks Data RSM, 12*12, averaged over runs')
    
    
    
    # DICKING AROUND WITH THE DATA
    # first step: z-scoring!
    # z-score the neuron matrices
    import scipy
    ave_neurons_between_z = scipy.stats.zscore(ave_neurons_between, axis=1)
    mc.simulation.predictions.plot_without_legends(ave_neurons_between_z, titlestring='z-scored neuron average', intervalline= 12)
    RSM_neurons_betas_ave_z = mc.simulation.RDMs.within_task_RDM(ave_neurons_between_z, plotting = True, titlestring = 'Between tasks Data RSM, 12*12, averaged over runs', intervalline= 12)
    
    # get rid of the last task because it looks crappy in task b (?)
    ave_clocks_between_minusonetask = ave_clocks_between[:,0:-12].copy()
    RSM_clock_betas_ave_minusonetask = mc.simulation.RDMs.within_task_RDM(ave_clocks_between_minusonetask, plotting = True, titlestring = 'Between tasks Clocks RSM, 12*12, averaged over runs, withoutlasttask', intervalline= 12)
    ave_phase_between_minusonetask = ave_phase_between[:,0:-12].copy()
    RSM_phase_betas_ave_minusonetask = mc.simulation.RDMs.within_task_RDM(ave_phase_between_minusonetask, plotting = True, titlestring = 'Between tasks Pgase RSM, 12*12, averaged over runs withoutlasttask', intervalline= 12)
    ave_midnight_between_minusonetask = ave_midnight_between[:,0:-12].copy()
    RSM_midnight_betas_ave_minusonetask = mc.simulation.RDMs.within_task_RDM(ave_midnight_between_minusonetask, plotting = True, titlestring = 'Between tasks Midnight RSM, 12*12, averaged over runs withoutlasttask', intervalline= 12)
    ave_location_between_minusonetask = ave_location_between[:,0:-12].copy()
    RSM_location_betas_ave_minusonetask = mc.simulation.RDMs.within_task_RDM(ave_location_between_minusonetask, plotting = True, titlestring = 'Between tasks Location RSM, 12*12, averaged over runs withoutlasttask', intervalline= 12)
    # and also for the data 
    ave_neuron_data_minusonetask = ave_neurons_between_z[:, 0:-12].copy()
    RSM_neurons_betas_ave_z_minusonetask = mc.simulation.RDMs.within_task_RDM(ave_neuron_data_minusonetask, plotting = True, titlestring = 'Between tasks Data RSM, 12*12, averaged over runs', intervalline= 12)
    
    # get rid of the fourth task because it looks crappy in task e (?)
    # REMOVE AGAIN!!
    import pdb; pdb.set_trace()
    ave_clocks_between = np.concatenate((ave_clocks_between[:, 0:36], ave_clocks_between[:, 48::]), axis = 1)
    ave_neurons_between_z = np.concatenate((ave_neurons_between_z[:, 0:36], ave_neurons_between_z[:,48::]), axis = 1)
    ave_midnight_between = np.concatenate((ave_midnight_between[:, 0:36], ave_midnight_between[:,48::]), axis = 1)
    
    
    
    # # NEXT: create matrices where I plot each phase separetly: take every 3rd row
    # mc.simulation.predictions.plot_without_legends(ave_clocks_between_minusonetask[:, 0::3], titlestring='early clocks across tasks', intervalline= 4)
    # mc.simulation.predictions.plot_without_legends(ave_neuron_data_minusonetask[:, 0::3], titlestring='early neurons across tasks', intervalline= 4)
    # mc.simulation.predictions.plot_without_legends(ave_midnight_between_minusonetask[:, 0::3], titlestring='early midnight across tasks', intervalline= 4)
    
    # mc.simulation.predictions.plot_without_legends(ave_clocks_between_minusonetask[:, 1::3], titlestring='mid clocks across tasks', intervalline= 4)
    # mc.simulation.predictions.plot_without_legends(ave_neuron_data_minusonetask[:, 1::3], titlestring='mid neurons across tasks', intervalline= 4)
    # mc.simulation.predictions.plot_without_legends(ave_midnight_between_minusonetask[:, 1::3], titlestring='mid midnight across tasks', intervalline= 4)
    
    # mc.simulation.predictions.plot_without_legends(ave_clocks_between_minusonetask[:, 2::3], titlestring='late clocks across tasks', intervalline= 4)
    # mc.simulation.predictions.plot_without_legends(ave_neuron_data_minusonetask[:, 2::3], titlestring='late neurons across tasks', intervalline= 4)
    # mc.simulation.predictions.plot_without_legends(ave_midnight_between_minusonetask[:, 2::3], titlestring='late midnight across tasks', intervalline= 4)
    
    
    # RSM_early_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between_minusonetask[:, 0::3], plotting=True, titlestring='RSM early clocks, averaged over runs', intervalline= 4)
    # RSM_early_neuron = mc.simulation.RDMs.within_task_RDM(ave_neuron_data_minusonetask[:, 0::3], plotting=True, titlestring='RSM early neurons, averaged over runs', intervalline= 4)
    # RSM_early_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between_minusonetask[:, 0::3], plotting=True, titlestring='RSM early midnight, averaged over runs', intervalline= 4)
    
    # RSM_mid_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between_minusonetask[:, 1::3], plotting=True, titlestring='RSM mid clocks, averaged over runs', intervalline= 4)
    # RSM_mid_neuron = mc.simulation.RDMs.within_task_RDM(ave_neuron_data_minusonetask[:, 1::3], plotting=True, titlestring='RSM mid neurons, averaged over runs', intervalline= 4)
    # RSM_mid_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between_minusonetask[:, 1::3], plotting=True, titlestring='RSM mid midnight, averaged over runs', intervalline= 4)
    
    # RSM_late_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between_minusonetask[:, 2::3], plotting=True, titlestring='RSM late clocks, averaged over runs', intervalline= 4)
    # RSM_late_neuron = mc.simulation.RDMs.within_task_RDM(ave_neuron_data_minusonetask[:, 2::3], plotting=True, titlestring='RSM late neurons, averaged over runs', intervalline= 4)
    # RSM_late_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between_minusonetask[:, 2::3], plotting=True, titlestring='RSM late midnight, averaged over runs', intervalline= 4)
    # NEXT: create matrices where I plot each phase separetly: take every 3rd row
    mc.simulation.predictions.plot_without_legends(ave_clocks_between[:, 0::3], titlestring='early clocks across tasks', intervalline= 4)
    mc.simulation.predictions.plot_without_legends(ave_neurons_between_z[:, 0::3], titlestring='early neurons across tasks', intervalline= 4)
    mc.simulation.predictions.plot_without_legends(ave_midnight_between[:, 0::3], titlestring='early midnight across tasks', intervalline= 4)
    
    mc.simulation.predictions.plot_without_legends(ave_clocks_between[:, 1::3], titlestring='mid clocks across tasks', intervalline= 4)
    mc.simulation.predictions.plot_without_legends(ave_neurons_between_z[:, 1::3], titlestring='mid neurons across tasks', intervalline= 4)
    mc.simulation.predictions.plot_without_legends(ave_midnight_between[:, 1::3], titlestring='mid midnight across tasks', intervalline= 4)
    
    mc.simulation.predictions.plot_without_legends(ave_clocks_between[:, 2::3], titlestring='late clocks across tasks', intervalline= 4)
    mc.simulation.predictions.plot_without_legends(ave_neurons_between_z[:, 2::3], titlestring='late neurons across tasks', intervalline= 4)
    mc.simulation.predictions.plot_without_legends(ave_midnight_between[:, 2::3], titlestring='late midnight across tasks', intervalline= 4)
    
    
    RSM_early_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, 0::3], plotting=True, titlestring='RSM early clocks, averaged over runs', intervalline= 4)
    RSM_early_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between_z[:, 0::3], plotting=True, titlestring='RSM early neurons, averaged over runs', intervalline= 4)
    RSM_early_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, 0::3], plotting=True, titlestring='RSM early midnight, averaged over runs', intervalline= 4)
    
    RSM_mid_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, 1::3], plotting=True, titlestring='RSM mid clocks, averaged over runs', intervalline= 4)
    RSM_mid_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between_z[:, 1::3], plotting=True, titlestring='RSM mid neurons, averaged over runs', intervalline= 4)
    RSM_mid_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, 1::3], plotting=True, titlestring='RSM mid midnight, averaged over runs', intervalline= 4)
    
    RSM_late_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between[:, 2::3], plotting=True, titlestring='RSM late clocks, averaged over runs', intervalline= 4)
    RSM_late_neuron = mc.simulation.RDMs.within_task_RDM(ave_neurons_between_z[:, 2::3], plotting=True, titlestring='RSM late neurons, averaged over runs', intervalline= 4)
    RSM_late_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between[:, 2::3], plotting=True, titlestring='RSM late midnight, averaged over runs', intervalline= 4)
    
    # run regressions separetly for each phase
    reg_early, scipy_early = mc.simulation.RDMs.lin_reg_RDMs(RSM_early_neuron, regressor_one_matrix = RSM_early_midnight, regressor_two_matrix= RSM_early_clocks)
    print('results for early are [midnight, clocks]', reg_early.coef_)
    reg_mid, scipy_mid = mc.simulation.RDMs.lin_reg_RDMs(RSM_mid_neuron, regressor_one_matrix = RSM_mid_midnight, regressor_two_matrix= RSM_mid_clocks)
    print('results for mid are [midnight, clocks]', reg_mid.coef_)
    reg_late, scipy_late = mc.simulation.RDMs.lin_reg_RDMs(RSM_late_neuron, regressor_one_matrix = RSM_late_midnight, regressor_two_matrix= RSM_late_clocks)
    print('results for late are [midnight, clocks]', reg_late.coef_)

    # LITTLE REGRESSION PLAYAROUND DELeTE LATER
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
    regression_results = LinearRegression().fit(x_all_reshaped, Yall)
    print('results for late are [midnight, clocks]', regression_results.coef_)

    # do whole regression with only midnight and clock
    RSM_neurons = mc.simulation.RDMs.within_task_RDM(ave_neurons_between_z)
    RSM_midnight = mc.simulation.RDMs.within_task_RDM(ave_midnight_between)
    RSM_clocks = mc.simulation.RDMs.within_task_RDM(ave_clocks_between)
    
    reg_mid_clock, scipyblah = mc.simulation.RDMs.lin_reg_RDMs(ave_neurons_between_z, regressor_one_matrix = ave_midnight_between, regressor_two_matrix= ave_clocks_between)
    print('results for late are [midnight, clocks]', reg_mid_clock.coef_)
    
    
    # then test how jus the correlation with the data RDMs looks like
    # correlation
    
    corr_clocks_data = mc.simulation.RDMs.corr_matrices_pearson(RSM_neurons_betas_ave_z_minusonetask, RSM_clock_betas_ave_minusonetask)
    print('correlation between clocks and data is', corr_clocks_data)
    # .6453
    corr_phase_data = mc.simulation.RDMs.corr_matrices_pearson(RSM_neurons_betas_ave_z_minusonetask, RSM_phase_betas_ave_minusonetask)
    # .6644
    print('correlation between phase and data is', corr_phase_data)
    
    # then do a regression:
    # regression with just the two most important ones - phase and clocks
    #results_phase_clocks_minusonetask, scipy_regression_results = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_betas_ave_z_minusonetask, regressor_one_matrix=RSM_clock_betas_ave_minusonetask, regressor_two_matrix= RSM_phase_betas_ave_minusonetask)
    # print('regression results are: [clocks, phase]:', results_phase_clocks_minusonetask.coef_)

    # clocks beta = 0.16255517, phase beta = 0.25516436
    
    # regression with all models
    
    results_average_minusonetask, scipy_regression_results = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_betas_ave_z_minusonetask, regressor_one_matrix=RSM_clock_betas_ave_minusonetask, regressor_two_matrix= RSM_phase_betas_ave_minusonetask, regressor_three_matrix= RSM_location_betas_ave_minusonetask, regressor_four_matrix= RSM_midnight_betas_ave_minusonetask)
    # print('regression results are: [clocks, phase, location, midnight]:', results_average_minusonetask.coef_)
    # results_average_minusonetask, scipy_regression_results = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_betas_ave_z_minusonetask, regressor_one_matrix=RSM_clock_betas_ave_minusonetask, regressor_two_matrix= RSM_phase_betas_ave_minusonetask, regressor_three_matrix= RSM_location_betas_ave_minusonetask)
    print('regression results are: [clocks, phase, location]:', results_average_minusonetask.coef_)
    
    
    
    
    
    results_average, scipy_regression_results = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons_betas_ave_z, regressor_one_matrix=RSM_clock_betas_ave, regressor_two_matrix= RSM_midnight_betas_ave, regressor_three_matrix= RSM_location_betas_ave, regressor_four_matrix= RSM_phase_betas_ave)
    
    
    return results_reg, scipy_regression_results, coefficient, results_average


