#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:25:52 2023

@author: Svenja Küchenhoff

This script includes optimisation function for different means and with different parameters.
"""


# this will be a function that optimises for low correlation between two different neural predictions.
# it will always be over time, but one can choose which ones: location, clocks, 0-angle
# one can choose if hrf convolved or not
# one can choose which parameters to optimise for: grid size, times per step, amounts of reward
# and lastly, one can determine the number of permutations to try these optimisations.

import mc
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# delete this again if I actually need the warnings..
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def optimise_task_for(prediction_one, prediction_two, hrf = True, grid_size = 3, step_time = 15, reward_no = 4, perms = 1, plot = False):
    # import pdb; pdb.set_trace()    
    # idea is to vary parameters oneself.
    # output is the optimal similarity after x permutations, with the respective parameters.
    
    if prediction_one != 'clocks' and prediction_one != 'location' and prediction_one != 'midnight':
        raise TypeError("Please enter 'location', 'midnight' or 'clocks'") 
    if prediction_two != 'clocks' and prediction_two != 'location' and prediction_two != 'midnight':
        raise TypeError("Please enter 'location', 'midnight' or 'clocks'")
    
    # start the optimisation process.
    # steps:
    # 1. create a task configuration
    # 2. create the predictions for both models
    # 2.1 check if HRF convolution wanted?
    # 3. create the RSM and similarity between the two models
    # 4. change the task configuration, do steps 2-3 and compare 
    # prep plot
    dissimilarity_values = []
    # config_nogood_corr = 0
    maximally_dissimilar = []
    best_reward_coords = []
    countgood_corr = 0
    
    for perm_no in range(perms):
        # 1. create a task configuration
        rew_coords = mc.simulation.grid.create_grid(grid_size, reward_no, plot = False)
        walk, steps_per_walk = mc.simulation.grid.walk_paths(rew_coords, grid_size, plotting = False)
        
        # 2. create the predictions for both models
        if prediction_one == 'clocks' or prediction_two == 'clocks':
            # NEW IS THIS:
            single_clock, midnight_matrix, clocks_model = mc.simulation.predictions.set_clocks_bytime(walk, steps_per_walk, step_time, grid_size)
            # THIS ONE IS OLD
            # clocksm, neuroncl, clocks_model = mc.simulation.predictions.set_clocks_bytime_one_neurone(walk, steps_per_walk, step_time, grid_size)
            
    
        if prediction_one == 'midnight' or prediction_two == 'midnight':
            # NEW:
            single_clock, midnight_model, clocks_model = mc.simulation.predictions.set_clocks_bytime(walk, steps_per_walk, step_time, grid_size)
            # OLD
            # clocksm, neuroncl, clocks_model_dummy = mc.simulation.predictions.set_clocks_bytime_one_neurone(walk, steps_per_walk, step_time, grid_size)
            # midnight_model = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_model_dummy, steps_per_walk, grid_size)

            
        if prediction_one == 'location' or prediction_two == 'location':
            locm, location_model = mc.simulation.predictions.set_location_by_time(walk, steps_per_walk, step_time, grid_size)
            # DELTE THIS AFTER DEBUGGING!!
            # ALSO CHANGE PLOTTING OPTIONS BACK TO FALSE
            #mc.simulation.predictions.plot_without_legends(location_model)
            
            
        # 2.1 check if HRF convolution wanted?
        if hrf:
            if 'location_model' in locals():
                location_model = mc.simulation.predictions.convolve_with_hrf(location_model, steps_per_walk, step_time, plotting = False)
            if 'clocks_model' in locals():
                clocks_model = mc.simulation.predictions.convolve_with_hrf(clocks_model, steps_per_walk, step_time, plotting = False)
            if 'midnight_model' in locals():
                midnight_model = mc.simulation.predictions.convolve_with_hrf(midnight_model, steps_per_walk, step_time, plotting = False)
                # clocks_model_hrf = mc.simulation.predictions.convolve_with_hrf(clocks_model, steps_per_walk, step_time, plotting = False)
                # midnight_model = mc.simulation.predictions.zero_phase_clocks_by_time(clocks_model_hrf, steps_per_walk, grid_size)
        
        # 3. create the RSM and similarity between the two models
        # make the names the same again
        if 'location_model' in locals():
            model_one = location_model[:]
            if 'clocks_model' in locals():
                model_two = clocks_model[:]
            elif 'midnight_model' in locals():
                model_two = midnight_model[:]
        # if this statement isnt true, then there is no location model, thus:
        else:
            model_one = clocks_model[:]
            model_two = midnight_model[:]
    
        # create a string for the columns
        count_columns = list(range(0,len(model_one[0])))
        col_names = count_columns.copy()
        for i in count_columns:
            col_names[i] = str(i) 
     
        RSM_one = mc.simulation.RDMs.within_task_RDM(model_one, col_names)
        RSM_two = mc.simulation.RDMs.within_task_RDM(model_two, col_names)
        
        
        correlation_sim = mc.simulation.RDMs.corr_matrices_pearson(RSM_one, RSM_two)
        similarity = correlation_sim[0,1]
        
        # i kendall
        # similarity = correlation_sim.correlation
        # store for the histogram
        dissimilarity_values.append(1 - similarity)
        
        if perm_no == 0:
            best_sim_value = similarity
            best_walk = walk[:]
            best_rewards = rew_coords[:]
            best_model_one = model_one[:]
            best_model_two = model_two[:]
        # 4. change the task configuration, do steps 2-3 and compare > loop
        else:
            if similarity < best_sim_value:
                best_sim_value = similarity
                best_walk = walk[:]
                best_rewards = rew_coords[:]
                best_model_one = model_one[:]
                best_model_two = model_two[:]
        
        # save the configurations if the similarity is really good.
        # save those configurations that have are maximally dissimlar. 
        if similarity <  0.70:   
            path = pd.DataFrame(walk)
            coef = pd.DataFrame([similarity]) 
            rewards = pd.DataFrame(rew_coords)
            if countgood_corr == 0:
                # to save as .csv file
                maximally_dissimilar = pd.concat([coef, path, rewards], axis = 1)
                # this is to have the coords in a np file in py
                temp_best_reward_coords = np.array(rew_coords)
                best_reward_coords = np.expand_dims(temp_best_reward_coords, axis = 0)
            else:
                maximally_dissimilar = pd.concat([maximally_dissimilar, coef, path, rewards], axis = 1)
                # best_reward_coords = np.stack([best_reward_coords, np.array(rew_coords)])     
                # this is to have the coords in a np file in py
                curr_coords = np.array(rew_coords)
                best_reward_coords = np.concatenate([curr_coords.reshape([1, curr_coords.shape[0], curr_coords.shape[1]]), best_reward_coords], axis=0)
                #best_reward_coords = np.stack([best_reward_coords, np.array(reward_coords)])
            countgood_corr += 1 
            
    
    if plot == True:
        # include the settings on the graps
        mc.simulation.predictions.plot_without_legends(best_model_one, 'clocks_model', hrf, grid_size, step_time, reward_no, perms)
        mc.simulation.predictions.plot_without_legends(best_model_two, 'midnight_model', hrf, grid_size, step_time, reward_no, perms)
        
        mc.simulation.grid.walk_paths(best_rewards, grid_size, plotting = True)
        # might also be nice to plot the distribution of correlation values for these settings...
        plt.figure()
        ax2 = plt.axes()
        plt.hist(dissimilarity_values)
        titletext = (f'Variance of {prediction_one} beyond {prediction_two}, hrf is {hrf}, grid is {grid_size} x {grid_size}, one step lasts {step_time} ms, over {perms} perms')
        plt.title(titletext)
        plt.ylabel = 'frequency'
        plt.xlabel = '1 - Similarity'
        
        plt.figure()
        
        best_model_one_df = pd.DataFrame(best_model_one)
        best_model_two_df = pd.DataFrame(best_model_two)
        best_RDM_one = mc.simulation.RDMs.df_based_RDM(best_model_one_df, plotting = True)
        best_RDM_two =mc.simulation.RDMs.df_based_RDM(best_model_two_df, plotting = True)
    
    # check the correlation values if you get rid of 20 timepoints in the beginning and in the end
    
    c_kendall, c_pearson = mc.simulation.RDMs.corr_matrices_no_autocorr(best_RDM_one, best_RDM_two, timepoints_to_exclude = 20, plotting = False)
    return best_sim_value, best_walk, best_rewards, best_reward_coords, maximally_dissimilar, dissimilarity_values, c_kendall, c_pearson


# FOR BETWEEN TASK OPTIMISATION!
# this now only works for the latest coding of clocks and midnights.
# (mc.simulation.predictions.set_clocks_bytime)


def optimise_several_task_configs(prediction_one, prediction_two, no_tasks, hrf = True, grid_size = 4, step_time = 15, reward_no = 4, perms = 1):
    # This is to reduce similarity both within and between tasks. 
    # You can select the number of task configurations, and all other parameters that are possible for the 
    # import pdb; pdb.set_trace()    
    # output is the optimal X optimal task configurations, with XY as similarity for the given parameters.   
    if prediction_one != 'clocks' and prediction_one != 'location' and prediction_one != 'midnight':
        raise TypeError("Please enter 'location', 'midnight' or 'clocks'") 
    if prediction_two != 'clocks' and prediction_two != 'location' and prediction_two != 'midnight':
        raise TypeError("Please enter 'location', 'midnight' or 'clocks'")
    dissimilarity_values = []
    countgood_corr = 0
    maximally_dissimilar = []
    best_reward_coords = []
    
    # first, create X random task configurations for model 1 and model 2.
    for task in range(0, no_tasks):
        # 1. create a task configuration
        rew_coords = mc.simulation.grid.create_grid(grid_size, reward_no, plot = False)
        walk, steps_per_walk = mc.simulation.grid.walk_paths(rew_coords, grid_size, plotting = False)
        
        # store the configurations.
        if task == 0:
            df_rewards = pd.DataFrame(rew_coords)
            df_walk = pd.DataFrame(walk)
            df_task_configs = pd.concat([df_rewards, df_walk], axis = 1)
        else:
            df_rewards = pd.DataFrame(rew_coords)
            df_walk = pd.DataFrame(walk)
            df_task_configs = pd.concat([df_task_configs, df_rewards, df_walk], axis = 1)
            
        
        # 2. create the predictions for both models
        if prediction_one == 'clocks' or prediction_two == 'clocks':
            single_clock, midnight_matrix, clocks_model = mc.simulation.predictions.set_clocks_bytime(walk, steps_per_walk, step_time, grid_size)

        if prediction_one == 'midnight' or prediction_two == 'midnight':
            single_clock, midnight_model, clocks_model = mc.simulation.predictions.set_clocks_bytime(walk, steps_per_walk, step_time, grid_size)
            
        if prediction_one == 'location' or prediction_two == 'location':
            locm, location_model = mc.simulation.predictions.set_location_by_time(walk, steps_per_walk, step_time, grid_size)

        # 2.1 check if HRF convolution wanted?
        if hrf:
            if 'location_model' in locals():
                location_model = mc.simulation.predictions.convolve_with_hrf(location_model, steps_per_walk, step_time, plotting = False)
            if 'clocks_model' in locals():
                clocks_model = mc.simulation.predictions.convolve_with_hrf(clocks_model, steps_per_walk, step_time, plotting = False)
            if 'midnight_model' in locals():
                midnight_model = mc.simulation.predictions.convolve_with_hrf(midnight_model, steps_per_walk, step_time, plotting = False)
    
        
        # make the names the same
        if 'location_model' in locals():
            model_one_df = pd.DataFrame(location_model)
            if 'clocks_model' in locals():
                model_two_df = pd.DataFrame(clocks_model)
            elif 'midnight_model' in locals():
                model_two_df = pd.DataFrame(midnight_model)
        # if this statement isnt true, then there is no location model, thus:
        else:
            model_one_df = pd.DataFrame(clocks_model)
            model_two_df = pd.DataFrame(midnight_model)
            
        # 2.0 prepare the column names
        # change back if this doesnt work!!!
        model_one_df = model_one_df.fillna(0)
        model_two_df = model_two_df.fillna(0)
        length_of_task = len(model_one_df.columns)
                  
        #now, concatenate these matrices
        if task < 1:
            model_one_X_tasks_df = model_one_df.copy()
            model_two_X_tasks_df = model_two_df.copy()
            length_per_task = [length_of_task]
            # store the reward coords
            temp_best_reward_coords = np.array(rew_coords)
            best_reward_coords = np.expand_dims(temp_best_reward_coords, axis = 0)
            
        if task > 0:
            model_one_X_tasks_df = pd.concat([model_one_X_tasks_df, model_one_df], axis = 1)
            model_two_X_tasks_df = pd.concat([model_two_X_tasks_df, model_two_df], axis = 1)
            length_per_task.append(length_of_task)
            # store the reward coords too
            curr_coords = np.array(rew_coords)
            best_reward_coords = np.concatenate([best_reward_coords, curr_coords.reshape([1, curr_coords.shape[0], curr_coords.shape[1]])], axis=0)
            

    
    # dimensionality check: sum(length_per_task) == len(model_one_X_tasks_df.columns)
    
    # change back if this way of naming the columns doesnt work!!        
    model_one_X_tasks_df.columns = range(model_one_X_tasks_df.columns.size)
    model_two_X_tasks_df.columns = range(model_two_X_tasks_df.columns.size)
    
    df_task_configs.columns = range(df_task_configs.columns.size)
    
    
    # 3. create RDMs and establish a similarity value
    RSM_one = mc.simulation.RDMs.df_based_RDM(model_one_X_tasks_df)
    RSM_two = mc.simulation.RDMs.df_based_RDM(model_two_X_tasks_df)
    
    # 4. identify current similarity
    correlation_sim = mc.simulation.RDMs.corr_matrices_pearson(RSM_one, RSM_two)
    similarity_between = correlation_sim[0,1]
    # corr_kendall = mc.simulation.RDMs.corr_matrices_kendall(RSM_one, RSM_two)
    
    cum_length_per_task = np.cumsum(length_per_task)
    # cum_length_per_task = np.append(0, cum_length_per_task)
    # NOW enter a loop in which I always exchange one task.
    # based on this, try to optimize the correlation coefficient (similarity_between)
    
    # After this loop, I will have 2 dataframes model1 and model2 with the same length,
    # that I will now correlate and test for their similarity. I will then continue
    # by systematically looping through the sub-models, exchanging single predictions
    # and testing if this will decrease the similarity.
    for perm in range(0, perms):        
        # create new configuration
        # 1. create a task configuration
        temp_rew_coords = mc.simulation.grid.create_grid(grid_size, reward_no, plot = False)
        temp_walk, temp_steps_per_walk_temp = mc.simulation.grid.walk_paths(temp_rew_coords, grid_size, plotting = False)
        
        # store this configuration in case I want it later
        temp_df_rewards = pd.DataFrame(temp_rew_coords)
        temp_df_walk = pd.DataFrame(temp_walk)
        temp_df_task_configs = pd.concat([temp_df_rewards, temp_df_walk], axis = 1)
        
        # create new neural predictions for this task config
        # 2. create the predictions for both models
        if prediction_one == 'clocks' or prediction_two == 'clocks':
            single_clock, temp_midnight_matrix, temp_clocks_model = mc.simulation.predictions.set_clocks_bytime(temp_walk, temp_steps_per_walk_temp, step_time, grid_size)
        if prediction_one == 'midnight' or prediction_two == 'midnight':
            single_clock, temp_midnight_model, temp_clocks_model = mc.simulation.predictions.set_clocks_bytime(temp_walk, temp_steps_per_walk_temp, step_time, grid_size)
        if prediction_one == 'location' or prediction_two == 'location':
            locm, temp_location_model = mc.simulation.predictions.set_location_by_time(temp_walk, temp_steps_per_walk_temp, step_time, grid_size)
            
            
        # 2.1 check if HRF convolution wanted?
        if hrf:
            if 'location_model' in locals():
                temp_location_model = mc.simulation.predictions.convolve_with_hrf(temp_location_model, temp_steps_per_walk_temp, step_time, plotting = False)
            if 'clocks_model' in locals():
                temp_clocks_model = mc.simulation.predictions.convolve_with_hrf(temp_clocks_model, temp_steps_per_walk_temp, step_time, plotting = False)
            if 'midnight_model' in locals():
                temp_midnight_model = mc.simulation.predictions.convolve_with_hrf(temp_midnight_model, temp_steps_per_walk_temp, step_time, plotting = False)
        
        # 3. create the RSM and similarity between the two models
        # make the names the same again
        if 'location_model' in locals():
            temp_model_one = temp_location_model[:]
            if 'clocks_model' in locals():
                temp_model_two = temp_clocks_model[:]
            elif 'midnight_model' in locals():
                temp_model_two = temp_midnight_model[:]
        # if this statement isnt true, then there is no location model, thus:
        else:
            temp_model_one = temp_clocks_model[:]
            temp_model_two = temp_midnight_model[:]
            

        # turn those into dataframe
        temp_model_one_df = pd.DataFrame(temp_model_one)
        temp_model_one_df.fillna(0)
        temp_model_two_df = pd.DataFrame(temp_model_two)
        temp_model_two_df.fillna(0)
        temp_length_of_task = len(temp_model_one_df.columns)

    
        # prepare loop here 
        temp_similarity = 1
        config_no = -1
    
        # then, replace each of the 10 tasks with the new config and text if similarity is now less (= better)
        # step out of the loop either way once looped through all columns, or when temp_similarity is lower
        while (temp_similarity > similarity_between) and (config_no < (no_tasks-1)):
            config_no+=1
            # identify which columns to cut
            temp_similarity = 1
            
            # I need to consider temp_cumsum here sometimes??
            if config_no == 0:
                cut_out_cols_min = config_no
                cut_out_cols_max = cum_length_per_task[config_no] # array that tells me how long each task is
            else:
                cut_out_cols_min = cum_length_per_task[config_no-1]
                cut_out_cols_max = cum_length_per_task[config_no]
            
            # remove the respective columns.
            # SOMETHING DOESNT WORK HERE!!!
            # is the indexing wrong???
            temp_model_one_X_tasks_df = model_one_X_tasks_df.drop(model_one_X_tasks_df.iloc[:, cut_out_cols_min: cut_out_cols_max], axis = 1)
            temp_model_two_X_tasks_df = model_two_X_tasks_df.drop(model_two_X_tasks_df.iloc[:, cut_out_cols_min: cut_out_cols_max], axis = 1)
            # dimensionality check: len(model_one_X_tasks_df.columns) - len(temp_model_one_X_tasks_df.columns) = cum_length_per_task[config_no]

            # append the new model.
            temp_model_one_X_tasks_df = pd.concat([temp_model_one_X_tasks_df, temp_model_one_df], axis = 1)
            temp_model_two_X_tasks_df = pd.concat([temp_model_two_X_tasks_df, temp_model_two_df], axis = 1)
            
            # delete the respective number, add the new column count 
            temp_length_per_task = np.delete(length_per_task, config_no)
            temp_length_per_task = np.append(temp_length_per_task, temp_length_of_task)
            # update cum_length_per_task
            temp_cum_length_per_task = np.cumsum(temp_length_per_task)
            # temp_cum_length_per_task = np.append(0, temp_cum_length_per_task)
            
            # # this is only needed in case I actually take it.
            # # and update the reward and path info
            # temp_all_task_configs = df_task_configs.copy()
            # if config_no == 0:
            #     temp_all_task_configs.iloc[:,config_no:4] = temp_df_task_configs
            # else:
            #     temp_all_task_configs.iloc[:,(config_no*4):((config_no+1)*4)] = temp_df_task_configs
  
              
            # reset the column names/ indices so I can use it again when I try the next replacement
            temp_model_one_X_tasks_df.columns = range(temp_model_one_X_tasks_df.columns.size)
            temp_model_two_X_tasks_df.columns = range(temp_model_two_X_tasks_df.columns.size)
            
            # create new RDMs for the updated task configuration
            temp_RSM_one = mc.simulation.RDMs.df_based_RDM(temp_model_one_X_tasks_df)
            temp_RSM_two = mc.simulation.RDMs.df_based_RDM(temp_model_two_X_tasks_df)
            temp_correlation_sim = mc.simulation.RDMs.corr_matrices_pearson(temp_RSM_one, temp_RSM_two)
            temp_similarity = temp_correlation_sim[0,1]
            #if temp_similarity < similarity_between, exit the loop and then take the new task combo instead.
            if temp_similarity < similarity_between:
                print(f'if we replace task at position {config_no}, similarity will go down to {temp_similarity}')
                print(f'replace {best_reward_coords[config_no]} with {temp_rew_coords}')
            # del temp_model_one_X_tasks_df
            # del temp_model_two_X_tasks_df
            # del temp_cum_length_per_task
            # del temp_length_per_task
            # del temp_correlation_sim
            # del temp_RSM_one
            # del temp_RSM_two
            # del temp_similarity
            # del temp_all_task_configs
            
        
            
        # if out of this loop, test if it was because the similarity was higher.
        # test if the new task config in this combination makes for a lower similarity
        # and if so, take this configuration from now on.
            
        if temp_similarity < similarity_between:
            similarity_between = temp_similarity.copy()
            model_one_X_tasks_df = temp_model_one_X_tasks_df.copy()
            model_two_X_tasks_df = temp_model_two_X_tasks_df.copy()
            length_per_task = temp_length_per_task.copy()
            cum_length_per_task = temp_cum_length_per_task.copy()
            
            # and update the reward and path info
            # temp_all_task_configs = df_task_configs.copy()
            # drop the respective columns
            temp_all_task_configs = df_task_configs.drop(df_task_configs.iloc[:, config_no*4:(config_no+1)*4], axis = 1)
            # and then append the new one at the end.
            temp_all_task_configs = pd.concat([temp_all_task_configs, temp_df_task_configs], axis = 1)
            df_task_configs = temp_all_task_configs.copy()
            df_task_configs.columns = range(df_task_configs.columns.size)
            
            # update rewards
            if config_no == 0:
                # delete the first one
                temp_best_reward_coords = best_reward_coords[1:]
                # append the new one
                curr_coords = np.array(temp_rew_coords)
                best_reward_coords = np.concatenate([temp_best_reward_coords, curr_coords.reshape([1, curr_coords.shape[0], curr_coords.shape[1]])], axis=0)
            if config_no == (no_tasks-1):
                # delete the last one
                temp_best_reward_coords = best_reward_coords[:-1]
                # append the new one
                curr_coords = np.array(temp_rew_coords)
                best_reward_coords = np.concatenate([temp_best_reward_coords, curr_coords.reshape([1, curr_coords.shape[0], curr_coords.shape[1]])], axis=0)
                
            elif config_no > 0: 
                # delete the current one
                temp_best_reward_coords_pt1 = best_reward_coords[0:config_no]
                temp_best_reward_coords_pt2 = best_reward_coords[(config_no+1):]
                # glue together
                temp_best_reward_coords = np.concatenate([temp_best_reward_coords_pt1,temp_best_reward_coords_pt2], axis = 0)
                # append the new one
                curr_coords = np.array(temp_rew_coords)
                best_reward_coords = np.concatenate([temp_best_reward_coords, curr_coords.reshape([1, curr_coords.shape[0], curr_coords.shape[1]])], axis=0)
                
            
        print(f'Finished perm {perm}, curr best corr {similarity_between}, temp_sim is {temp_similarity}')
    # DOUBLE CHECK IF IT WORKS!!!!
    # GO BACK TO HERE
    # IT SEEMS LIKE IT WORKED
    
    best_RSM_one = mc.simulation.RDMs.df_based_RDM(model_one_X_tasks_df)
    best_RSM_two = mc.simulation.RDMs.df_based_RDM(model_two_X_tasks_df)
    
    plt.figure(); plt.imshow(best_RSM_one); plt.title('best_clocks'); plt.figure(); plt.imshow(best_RSM_two); plt.title('best_midnight'); 
    print('done, yey!')
    return similarity_between, model_one_X_tasks_df, model_two_X_tasks_df, df_task_configs, best_reward_coords
            
            
            
# plt.figure(); plt.imshow(RSM_one); plt.figure(); plt.imshow(RSM_two)         
 

# to check if the function above works.
def testing_several_task_configs(no_tasks, perms = 1):
    # This is to reduce similarity both within and between tasks. 
    # You can select the number of task configurations, and all other parameters that are possible for the 
    # import pdb; pdb.set_trace()    
    # output is the optimal X optimal task configurations, with XY as similarity for the given parameters.   
    
    # I will write a testing function, which instead of the models, just generates random numbers in matrices
    # and then stepwise goes through the columns to improve the correlations.
    # first: generate a model-ones-matrix and a model-zero-matrix with given dimensions.
    # second: compute similarity between model one and model zero.
    # third: generate a 'new task': 2 random number columns. 
    # fourth: exchange the columns of model-one and model-zero with these random numbers.
    # five: check if the similarity improved.
    # six: iterate through the matrices.
    
    # first, create X random task configurations for model 1 and model 2.
    for task in range(0, no_tasks):
        # create a task table too
        rew_coords = [[1,1],[1,1]]
        walk = [[3,3],[3,3],[3,3]]
        
        # store the configurations.
        if task == 0:
            df_rewards = pd.DataFrame(rew_coords)
            df_walk = pd.DataFrame(walk)
            df_task_configs = pd.concat([df_rewards, df_walk], axis = 1)
        else:
            df_rewards = pd.DataFrame(rew_coords)
            df_walk = pd.DataFrame(walk)
            df_task_configs = pd.concat([df_task_configs, df_rewards, df_walk], axis = 1)
        
        # one task i 5*2 
        model_one = np.random.rand(4,3)*0.001
        model_two = np.random.rand(4,3)*1000
        
        model_one_df = pd.DataFrame(model_one)
        model_two_df = pd.DataFrame(model_two)
           
        # 2.0 prepare the column names
        # change back if this doesnt work!!!
        model_one_df.fillna(0)
        model_two_df.fillna(0)
        length_of_task = len(model_one_df.columns)
        
                  
        #now, concatenate these matrices
        if task < 1:
            model_one_X_tasks_df = model_one_df.copy()
            model_two_X_tasks_df = model_two_df.copy()
            length_per_task = [length_of_task]
        if task > 0:
            model_one_X_tasks_df = pd.concat([model_one_X_tasks_df, model_one_df], axis = 1)
            model_two_X_tasks_df = pd.concat([model_two_X_tasks_df, model_two_df], axis = 1)
            length_per_task.append(length_of_task)
    
    # dimensionality check: sum(length_per_task) == len(model_one_X_tasks_df.columns)
    
    # change back if this way of naming the columns doesnt work!!        
    model_one_X_tasks_df.columns = range(model_one_X_tasks_df.columns.size)
    model_two_X_tasks_df.columns = range(model_two_X_tasks_df.columns.size)
    
    
    # 3. create RDMs and establish a similarity value
    RSM_one = mc.simulation.RDMs.df_based_RDM(model_one_X_tasks_df)
    RSM_two = mc.simulation.RDMs.df_based_RDM(model_two_X_tasks_df)
    
    # 4. identify current similarity
    correlation_sim = mc.simulation.RDMs.corr_matrices_pearson(RSM_one, RSM_two)
    similarity_between = correlation_sim[0,1]
    corr_kendall = mc.simulation.RDMs.corr_matrices_kendall(RSM_one, RSM_two)
    
    cum_length_per_task = np.cumsum(length_per_task)
    # cum_length_per_task = np.append(0, cum_length_per_task)
    # NOW enter a loop in which I always exchange one task.
    # based on this, try to optimize the correlation coefficient (similarity_between)
    
    # After this loop, I will have 2 dataframes model1 and model2 with the same length,
    # that I will now correlate and test for their similarity. I will then continue
    # by systematically looping through the sub-models, exchanging single predictions
    # and testing if this will decrease the similarity.
    for perm in range(0, perms): 
        
        temp_rew_coords = [[0,0],[0,0]]
        temp_walk = [[0,0],[0,0],[0,0]]
        temp_df_rewards = pd.DataFrame(temp_rew_coords)
        temp_df_walk = pd.DataFrame(temp_walk)
        temp_df_task_configs = pd.concat([temp_df_rewards, temp_df_walk], axis = 1)
        
        # now create one random task without changing the size
        temp_model_one = np.random.rand(4,3)
        temp_model_two = np.random.rand(4,3)

        # turn those into dataframe
        temp_model_one_df = pd.DataFrame(temp_model_one)
        temp_model_one_df.fillna(0)
        temp_model_two_df = pd.DataFrame(temp_model_two)
        temp_model_two_df.fillna(0)
        temp_length_of_task = len(temp_model_one_df.columns)

        # prepare loop here 
        temp_similarity = 1
        config_no = 0
    
        # then, replace each of the 10 tasks with the new config and text if similarity is now less (= better)
        # step out of the loop either way once looped through all columns, or when temp_similarity is lower
        while (temp_similarity > similarity_between) and (config_no < (no_tasks)):
            # identify which columns to cut
            temp_similarity = 1
            if config_no == 0:
                cut_out_cols_min = config_no
                cut_out_cols_max = cum_length_per_task[config_no] # array that tells me how long each task is
            else:
                cut_out_cols_min = cum_length_per_task[config_no-1]
                cut_out_cols_max = cum_length_per_task[config_no]
            
            # remove the respective columns.
            temp_model_one_X_tasks_df = model_one_X_tasks_df.drop(model_one_X_tasks_df.iloc[:, cut_out_cols_min: cut_out_cols_max], axis = 1)
            temp_model_two_X_tasks_df = model_two_X_tasks_df.drop(model_two_X_tasks_df.iloc[:, cut_out_cols_min: cut_out_cols_max], axis = 1)
            # dimensionality check: len(model_one_X_tasks_df.columns) - len(temp_model_one_X_tasks_df.columns) = cum_length_per_task[config_no]

            # append the new model.
            temp_model_one_X_tasks_df = pd.concat([temp_model_one_X_tasks_df, temp_model_one_df], axis = 1)
            temp_model_two_X_tasks_df = pd.concat([temp_model_two_X_tasks_df, temp_model_two_df], axis = 1)
            
            # delete the respective number, add the new column count 
            temp_length_per_task = np.delete(length_per_task, config_no)
            temp_length_per_task = np.append(temp_length_per_task, temp_length_of_task)
            # update cum_length_per_task
            temp_cum_length_per_task = np.cumsum(length_per_task)
            temp_cum_length_per_task = np.append(0, temp_cum_length_per_task)
            
            # and update the reward and path info
            temp_all_task_configs = df_task_configs.copy()
            if config_no == 0:
                temp_all_task_configs.iloc[:,config_no:4] = temp_df_task_configs
            else:
                temp_all_task_configs.iloc[:,(config_no*4):((config_no+1)*4)] = temp_df_task_configs
            
              
            # reset the column names/ indices so I can use it again when I try the next replacement
            temp_model_one_X_tasks_df.columns = range(temp_model_one_X_tasks_df.columns.size)
            temp_model_two_X_tasks_df.columns = range(temp_model_two_X_tasks_df.columns.size)
            
            # create new RDMs for the updated task configuration
            temp_RSM_one = mc.simulation.RDMs.df_based_RDM(temp_model_one_X_tasks_df)
            temp_RSM_two = mc.simulation.RDMs.df_based_RDM(temp_model_two_X_tasks_df)
            temp_correlation_sim = mc.simulation.RDMs.corr_matrices_pearson(temp_RSM_one, temp_RSM_two)
            temp_similarity = temp_correlation_sim[0,1]
            # now if temp_similariy, we should exit the loop and then take the new task combo instead.
            # if not, we go back to the original 
            # CHECK IF ONE ALREADY EXITS THE LOOP HERE!!

            config_no+=1
            # del temp_model_one_X_tasks_df
            # del temp_model_two_X_tasks_df
            # del temp_cum_length_per_task
            # del temp_length_per_task
            # del temp_correlation_sim
            # del temp_RSM_one
            # del temp_RSM_two
            # del temp_similarity
            # del temp_all_task_configs
    
            
        # if out of this loop, test if it was because the similarity was higher.
        # test if the new task config in this combination makes for a lower similarity
        # and if so, take this configuration from now on.
            
        if temp_similarity < similarity_between:
            similarity_between = temp_similarity.copy()
            model_one_X_tasks_df = temp_model_one_X_tasks_df.copy()
            model_two_X_tasks_df = temp_model_two_X_tasks_df.copy()
            length_per_task = temp_length_per_task.copy()
            cum_length_per_task = temp_cum_length_per_task.copy()
            df_task_configs = temp_all_task_configs.copy()
            
    # DOUBLE CHECK IF IT WORKS!!!!
    # GO BACK TO HERE
    # IT SEEMS LIKE IT WORKED
    
    print('done, yey!')
    return similarity_between, model_one_X_tasks_df, model_two_X_tasks_df, df_task_configs


def show_several_taskconfigs(rew_coords, prediction_one, gridsize, timeperstep, reward_no, prediction_two = None, prediction_three = None, hrf = False):
    # import pdb; pdb.set_trace() 
    result_dict = {}
    for task, rew_conf in enumerate(rew_coords):
        walk, steps_per_walk = mc.simulation.grid.walk_paths(rew_conf, gridsize, plotting = True)
        # 2. create the predictions for  models
        if prediction_one == 'clocks' or prediction_two == 'clocks' or prediction_three == 'clocks':
            single_clock, midnight_matrix, clocks_model = mc.simulation.predictions.set_clocks_bytime(walk, steps_per_walk, timeperstep, gridsize)

        if prediction_one == 'midnight' or prediction_two == 'midnight' or prediction_three == 'midnight':
            single_clock, midnight_model, clocks_model = mc.simulation.predictions.set_clocks_bytime(walk, steps_per_walk, timeperstep, gridsize)
            
        if prediction_one == 'location' or prediction_two == 'location' or prediction_three == 'location':
            locm, location_model = mc.simulation.predictions.set_location_by_time(walk, steps_per_walk, timeperstep, gridsize)
            
        # 2.1 check if HRF convolution wanted?
        if hrf:
            if 'location_model' in locals():
                location_model = mc.simulation.predictions.convolve_with_hrf(location_model, steps_per_walk, timeperstep, plotting = False)
            if 'clocks_model' in locals():
                clocks_model = mc.simulation.predictions.convolve_with_hrf(clocks_model, steps_per_walk, timeperstep, plotting = False)
            if 'midnight_model' in locals():
                midnight_model = mc.simulation.predictions.convolve_with_hrf(midnight_model, steps_per_walk, timeperstep, plotting = False)
        
        if 'location_model' in locals():
            df_location_model = pd.DataFrame(location_model).fillna(0)
            if task < 1:
                location_X_tasks_df = df_location_model.copy()
            if task > 0:
                location_X_tasks_df = pd.concat([location_X_tasks_df, df_location_model], axis = 1)
        if 'clocks_model' in locals():
            df_clocks_model = pd.DataFrame(clocks_model).fillna(0)
            if task < 1:
                clocks_X_tasks_df = df_clocks_model.copy()
            if task > 0:
                clocks_X_tasks_df = pd.concat([clocks_X_tasks_df, df_clocks_model], axis = 1)
        if 'midnight_model' in locals():
            df_midnight_model = pd.DataFrame(midnight_model).fillna(0)
            if task < 1:
                midnight_X_tasks_df = df_midnight_model.copy()
            if task > 0:
                midnight_X_tasks_df = pd.concat([midnight_X_tasks_df, df_midnight_model], axis = 1)

    if 'location_model' in locals():
        location_X_tasks_df.columns = range(location_X_tasks_df.columns.size)
        RSM_location = mc.simulation.RDMs.df_based_RDM(location_X_tasks_df, plotting = True, titlestring = 'Location RDM')
    if 'clocks_model' in locals():
        clocks_X_tasks_df.columns = range(clocks_X_tasks_df.columns.size)
        RSM_clocks = mc.simulation.RDMs.df_based_RDM(clocks_X_tasks_df, plotting = True, titlestring = 'Clocks RDM')
    if 'midnight_model' in locals():
        midnight_X_tasks_df.columns = range(midnight_X_tasks_df.columns.size)
        RSM_midnight = mc.simulation.RDMs.df_based_RDM(midnight_X_tasks_df, plotting = True, titlestring = 'Midnight RDM')
    
    if 'location_model' and 'clocks_model' in locals():
        correlation_sim = mc.simulation.RDMs.corr_matrices_pearson(RSM_location, RSM_clocks)
        pearson_loc_clocks = correlation_sim[0,1]
        kendall_loc_clocks = mc.simulation.RDMs.corr_matrices_kendall(RSM_location, RSM_clocks)
        result_dict['pearson_loc_clocks'] = pearson_loc_clocks
        result_dict['kendall_loc_clocks'] = kendall_loc_clocks.correlation
    if 'clocks_model' and 'midnight_model' in locals():
        correlation_sim = mc.simulation.RDMs.corr_matrices_pearson(RSM_location, RSM_midnight)
        pearson_loc_midnight = correlation_sim[0,1]
        kendall_loc_midnight = mc.simulation.RDMs.corr_matrices_kendall(RSM_location, RSM_midnight)
        result_dict['pearson_loc_midnight'] = pearson_loc_midnight
        result_dict['kendall_loc_midnight'] = kendall_loc_midnight.correlation
    if 'midnight_model' and 'clocks_model' in locals():
        correlation_sim = mc.simulation.RDMs.corr_matrices_pearson(RSM_clocks, RSM_midnight)
        pearson_clocks_midnight = correlation_sim[0,1]
        kendall_clocks_midnight = mc.simulation.RDMs.corr_matrices_kendall(RSM_clocks, RSM_midnight)
        result_dict['pearson_clocks_midnight'] = pearson_clocks_midnight
        result_dict['kendall_clocks_midnight'] = kendall_clocks_midnight.correlation

    return result_dict






























