#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:55:36 2023

@author: Svenja Küchenhoff
this script runs the entire analysis of Mohamadys data
"""

import numpy as np
import mc
import math 

# import pdb; pdb.set_trace()


# first import the WHOLE dataset (all days, all mice, all runs)
Data_folder='/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/' 

mouse_recday='me11_05122021_06122021' #mouse a
a_rewards_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
a_no_task_configs = len(a_rewards_configs)
a_all_task_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
a_locations = list()
a_neurons = list()
a_timings = list()
for session in range(0, a_no_task_configs):
    a_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    a_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    a_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))



mouse_recday='me11_01122021_02122021' #mouse b 
b_rewards_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
b_no_task_configs = len(b_rewards_configs)
b_all_task_configs = np.load(Data_folder+'Task_data_'+ mouse_recday+'.npy')
b_locations = list()
b_neurons = list()
b_timings = list()
for session in range(0, b_no_task_configs):
    b_locations.append(np.load(Data_folder+'Location_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    b_neurons.append(np.load(Data_folder+'Neuron_raw_'+mouse_recday+'_'+str(session)+'.npy'))
    b_timings.append(np.load(Data_folder+'trialtimes_'+mouse_recday+'_'+str(session)+'.npy'))


mouse_recday='me10_09122021_10122021' #mouse c range 0,9
mouse_recday='me08_10092021_11092021' #mouse d range 0,6
mouse_recday='ah04_09122021_10122021' #mouse e range 0,8 
mouse_recday='ah04_05122021_06122021' #mouse f range 0,8
mouse_recday='ah04_01122021_02122021' #mouse g range 0,8
mouse_recday='ah03_18082021_19082021' #mouse h range 0,8

# don't know if there is a smart way to do this similarly for every mouse?
# but this is what I'd do separetly per mouse:

# mouse a
for task in range(a_no_task_configs):
    task_config = a_all_task_configs[task]
    # first convert trial times from ms to bin number to match neuron and location arrays 
    # (1 bin = 25ms)
    timings = a_timings.copy()
    for r, row in enumerate(a_timings):
        for c, element in enumerate(row):
            timings[r,c] = element/25


    # second, change locations and rewards to 0 and ignoring bridges
    locations = a_locations.copy()
    for i, field in enumerate(a_locations):
        if field > 9: 
            locations[i] = locations[i-1]
        if math.isnan(field):
            # keep the location bc of timebins
            locations[i] = locations[i-1]
                

    # important: fields need to be between 0 and 8, and keep them as integers!
    locations = [int((field_no-1)) for field_no in locations]
    task_config = [int((field_no-1)) for field_no in task_config]


    # I will do this differently. it's annoying to store runs of different lengths.
    # instead, I will have my subpaths, separately for every path
    # potentiall, I will want to take a mean across runs... let's start slow. 
    row = timings[20]

    # define current data
    # > potentially turn into a loop at some point
    trajectory = locations[row[0]:row[-1]]

    # ISSUE 21.04.23:
    # if there are ONLY 0 for one timestep, the np.corrcoef will output nan for that instance. Maybe better:
    # replace by super super low value
    curr_neurons = a_neurons[:,row[0]:row[-1]]

    test_curr_neurons = curr_neurons.copy()
    for col_no, column in enumerate(test_curr_neurons.T):
        if np.all(column == 0):
            test_curr_neurons[:,col_no] = 0.00001

    # some pre-processing to create my models.
    # to count subpaths
    subpath_file = [locations[row[0]:row[1]+1], locations[row[1]+1:row[2]+1], locations[row[2]+1:row[3]+1], locations[row[3]+1:row[4]+1]]
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
            
            
    location_model = mc.simulation.predictions.set_location_raw_ephys(trajectory, step_time = 1, grid_size=3, plotting = True, field_no_given= 1)
    midnight_model, clocks_model = mc.simulation.predictions.set_clocks_raw_ephys(trajectory, timings_curr_run, index_make_step, step_number, field_no_given= 1, plotting=True)


    # now create the model RDMs
    RSM_location = mc.simulation.RDMs.within_task_RDM(location_model, plotting = True, titlestring = 'Location RDM')
    RSM_clock = mc.simulation.RDMs.within_task_RDM(clocks_model, plotting = True, titlestring = 'Clock RDM')
    RSM_midnight = mc.simulation.RDMs.within_task_RDM(midnight_model, plotting = True, titlestring = 'Midnight RDM')

    # now create the data RDM
    # I am wondering if this is correct, though - maybe should I select those neurons where I know fit my predictions?
    RSM_neurons = mc.simulation.RDMs.within_task_RDM(test_curr_neurons, plotting = True, titlestring = 'Data RDM')

    # Lastly, create a linear regression with RSM_loc,clock and midnight as regressors and data to be predicted
    reg_res, scipy_regression_results = mc.simulation.RDMs.lin_reg_RDMs(RSM_neurons, regressor_one_matrix=RSM_clock, regressor_two_matrix= RSM_midnight, regressor_three_matrix= RSM_location, t_val = 'yes')
    print(f" The beta for the clocks model is {reg_res.coef_[0]}, for the midnight model is {reg_res.coef_[1]}, and for the location model is {reg_res.coef_[2]}")
    print(scipy_regression_results.summary())

    # THIS VALUE THEN NEEDS TO BE STORED SOMEWHERE  
    