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
        rew_coords = mc.simulation.grid.create_grid(grid_size, reward_no, plot = False, old_rewards=None, step_longer_one=True)
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
        rew_coords = mc.simulation.grid.create_grid(grid_size, reward_no, plot = False, old_rewards=None, step_longer_one=True)
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
            
        # 2.0 prepare the column names - I need those to later drop the correct columns!
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
        temp_rew_coords = mc.simulation.grid.create_grid(grid_size, reward_no, plot = False, old_rewards=None, step_longer_one=True)
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
    df_task_configs.columns = range(df_task_configs.columns.size)
    
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










########## write new optimisation function with the new models. don't want to change everything.
def opt_fmri_tasks(no_tasks, grid_size, step_time, reward_no, permutations, hrf = True, no_bins_per_state = None, bin_data = False):
    # import pdb; pdb.set_trace() 
    # for i, model_name in enumerate(models):
    #     compute_model = eval()
    #     dataset = eval(f"{mouse}_reg_result_dict")
    
    # create random task configurations.
    model_dict = {}
    for task in range(0, no_tasks):
        # 1. create a task configuration
        rew_coords = mc.simulation.grid.create_grid(grid_size, reward_no, plot = False, old_rewards=None, step_longer_one=True)
        walk, steps_per_walk = mc.simulation.grid.walk_paths(rew_coords, grid_size, plotting = False)
        
        # store the configurations.
        # probably change this to dictionaries!!!
        if task == 0:
            df_rewards = pd.DataFrame(rew_coords)
            df_walk = pd.DataFrame(walk)
            df_task_configs = pd.concat([df_rewards, df_walk], axis = 1)
        else:
            df_rewards = pd.DataFrame(rew_coords)
            df_walk = pd.DataFrame(walk)
            df_task_configs = pd.concat([df_task_configs, df_rewards, df_walk], axis = 1)
        
        # 2. create model predictions for this configuration.
        loc_mod, phas_mod, stat_mod, midn_mod, clo_mod, phasestate_mod = mc.simulation.predictions.set_continous_models(walk, steps_per_walk, step_time, grid_size, no_phase_neurons=3, fire_radius = 0.25)
        
        
        
        # 2.1 HRF convolution wanted?
        if hrf == True:
            loc_mod = mc.simulation.predictions.convolve_with_hrf(loc_mod, steps_per_walk, step_time, plotting = False)
            phas_mod =  mc.simulation.predictions.convolve_with_hrf(phas_mod, steps_per_walk, step_time, plotting = False)
            midn_mod = mc.simulation.predictions.convolve_with_hrf(midn_mod, steps_per_walk, step_time, plotting = False)
            clo_mod =mc.simulation.predictions.convolve_with_hrf(clo_mod, steps_per_walk, step_time, plotting = False)

        prep_model_dict = {}
        prep_model_dict["loc_mod_df"]= loc_mod
        prep_model_dict["phas_mod_df"] = phas_mod
        prep_model_dict["midn_mod_df"] = midn_mod
        prep_model_dict["clo_mod_df"] = clo_mod
        # prep_model_dict["loc_mod_df"]= pd.DataFrame(loc_mod).fillna(0)
        # prep_model_dict["phas_mod_df"] = pd.DataFrame(phas_mod).fillna(0)
        # prep_model_dict["midn_mod_df"] = pd.DataFrame(midn_mod).fillna(0)
        # prep_model_dict["clo_mod_df"] = pd.DataFrame(clo_mod).fillna(0)
        
        
        # the current MRI sequence takes a sample every 1.256 seconds -> subsample by factor 13
        # my fMRI sequence takes a sample every 1144ms -> hopefully every 1.1 secs > subsample by factor 11
        
        # TEMPORARILY CHANGE THIS BACK!
        #for curr_model in prep_model_dict:
        #    prep_model_dict[curr_model] = mc.simulation.predictions.subsample(prep_model_dict[curr_model], subsample_factor = 11)
        
        
        #interpolation_test = mc.simulation.predictions.interpolate_neurons(prep_model_dict[curr_model], 10)
        
        
        # 2.2 binning wanted?
        if bin_data == True:
            # TEMPORARILY CHANGE THIS BACK!
            # for curr_model in prep_model_dict:
            #     prep_model_dict[curr_model] = mc.simulation.predictions.interpolate_neurons(prep_model_dict[curr_model], 10)
            
            # for now, bin by interpolating!!
            # you can change this later if you want. 
 
            timebin_regressors = mc.simulation.predictions.create_x_regressors_per_state_simulation(walk, steps_per_walk, step_time, no_regs_per_state = no_bins_per_state)
            if hrf == True:
                timebin_regressors = mc.simulation.predictions.convolve_with_hrf(timebin_regressors, steps_per_walk, step_time, plotting = False)
                #timebin_regressors = mc.simulation.predictions.subsample(timebin_regressors, subsample_factor=11)    
            for curr_model in prep_model_dict:
                prep_model_dict[curr_model] = mc.simulation.predictions.transform_data_to_betas(prep_model_dict[curr_model], timebin_regressors)
                prep_model_dict[curr_model] = pd.DataFrame(prep_model_dict[curr_model])
        
        #import pdb; pdb.set_trace()         
        # 2.0 prepare the column names - I need those to later drop the correct columns!
        for curr_model in prep_model_dict:
            prep_model_dict[curr_model] = pd.DataFrame(prep_model_dict[curr_model]).fillna(0)

        length_of_task = len(prep_model_dict["loc_mod_df"].columns)
        
        # 3. concatenate the different task simulations
        if task < 1:
            for curr_model in prep_model_dict:
                model_dict[curr_model] = prep_model_dict[curr_model].copy()
            length_all_tasks = [length_of_task]
            temp_best_reward_coords = np.array(rew_coords)
            best_reward_coords = np.expand_dims(temp_best_reward_coords, axis = 0)

        if task > 0:
            for curr_model in prep_model_dict:
                model_dict[curr_model] = pd.concat([model_dict[curr_model], prep_model_dict[curr_model]], axis = 1)
            length_all_tasks.append(length_of_task)
            # store the reward coords too
            curr_coords = np.array(rew_coords)
            best_reward_coords = np.concatenate([best_reward_coords, curr_coords.reshape([1, curr_coords.shape[0], curr_coords.shape[1]])], axis=0)
        
    # 4. name the columns so that I can drop the right ones later and 
    # 5. create RDMs and establish a similarity value  
    RSM_dict = {}
    for curr_model in model_dict:
        model_dict[curr_model].columns = range(model_dict[curr_model].columns.size)
        RSM_dict[curr_model] = mc.simulation.RDMs.df_based_RDM(model_dict[curr_model])
    
    df_task_configs.columns = range(df_task_configs.columns.size)
    
    for elem in model_dict:
        plt.figure();
        plt.imshow(model_dict[elem], aspect = 'auto')
    # for elem in RSM_dict:
    #     plt.figure();
    #     plt.imshow(RSM_dict[elem], aspect = 'auto')
    
    
    # 6 . identify current similarity - PEARSON OR KENDALL????
    similarity_between_dict = {}
    for i, curr_RSM_one in enumerate(RSM_dict):
        for j, curr_RSM_two in enumerate(RSM_dict):
            curr_corr = f"{curr_RSM_one}_with_{curr_RSM_two}"
            correlation = mc.simulation.RDMs.corr_matrices_pearson(RSM_dict[curr_RSM_one], RSM_dict[curr_RSM_two], no_tasks = None, mask_within = False, exclude_diag = True)
            similarity_between_dict[curr_corr] = correlation[0,1]
    # corr_kendall = mc.simulation.RDMs.corr_matrices_kendall(RSM_one, RSM_two)


    # NOW enter a loop in which I always exchange one task.
    # based on this, try to optimize the correlation coefficient (similarity_between)
    
    # After this loop, I will have a dictionary of dataframes with the same length,
    # that I will now correlate and test for their similarity. I will then continue
    # by systematically looping through the sub-models, exchanging single predictions
    # and testing if this will decrease the similarity.
    #for perm in range(0, perms): 
    
    # for now, just take the correlations with the clocks model.
    
    cum_length_per_task = np.cumsum(length_all_tasks)
    of_interest = ['clo_mod_df_with_loc_mod_df', 'clo_mod_df_with_midn_mod_df', 'clo_mod_df_with_phas_mod_df']
    similarity_between = 0
    # I want to make the sum of these 3 as low as possible.
    for elem in of_interest:
        similarity_between = abs(similarity_between) + abs(similarity_between_dict[elem])
    
    
    # import pdb; pdb.set_trace() 
    for perm in range(0, permutations):
        # create new configuration
        # 1. create a task configuration
        temp_rew_coords = mc.simulation.grid.create_grid(grid_size, reward_no, plot = False, old_rewards=None, step_longer_one=True)
        temp_walk, temp_steps_per_walk = mc.simulation.grid.walk_paths(temp_rew_coords, grid_size, plotting = False)
        
        # store this configuration in case I want it later
        temp_df_rewards = pd.DataFrame(temp_rew_coords)
        temp_df_walk = pd.DataFrame(temp_walk)
        temp_df_task_configs = pd.concat([temp_df_rewards, temp_df_walk], axis = 1)
        
        # create new neural predictions for this task config
        temp_loc_mod, temp_phas_mod, temp_stat_mod, temp_midn_mod, temp_clo_mod, phasestate_mod = mc.simulation.predictions.set_continous_models(temp_walk, temp_steps_per_walk, step_time, grid_size, no_phase_neurons=3, fire_radius = 0.25)
        
        
        # 2.1 check if HRF convolution wanted?
        if hrf == True:
            temp_loc_mod = mc.simulation.predictions.convolve_with_hrf(temp_loc_mod, steps_per_walk, step_time, plotting = False)
            temp_phas_mod =  mc.simulation.predictions.convolve_with_hrf(temp_phas_mod, steps_per_walk, step_time, plotting = False)
            temp_midn_mod = mc.simulation.predictions.convolve_with_hrf(temp_midn_mod, steps_per_walk, step_time, plotting = False)
            temp_clo_mod =mc.simulation.predictions.convolve_with_hrf(temp_clo_mod, steps_per_walk, step_time, plotting = False)
        
        temp_prep_model_dict = {}
        temp_prep_model_dict["loc_mod_df"]= temp_loc_mod
        temp_prep_model_dict["phas_mod_df"] = temp_phas_mod
        temp_prep_model_dict["midn_mod_df"] = temp_midn_mod
        temp_prep_model_dict["clo_mod_df"] = temp_clo_mod

        # TEMPORARILY CHANGE THIS BACK!
        # the current MRI sequence takes a sample every 1.1 seconds -> subsample by factor 11
        # for curr_model in temp_prep_model_dict:
        #     temp_prep_model_dict[curr_model] = mc.simulation.predictions.subsample(temp_prep_model_dict[curr_model], subsample_factor = 11)
         
        # 2.2 binning wanted?
        # 2.2 binning wanted?
        # if bin_data == True:
        #     for curr_model in temp_prep_model_dict:
        #         temp_prep_model_dict[curr_model] = mc.simulation.predictions.interpolate_neurons(temp_prep_model_dict[curr_model], 10)
        # # TEMPORARILY CHANGE THIS BACK!
        if bin_data == True:
            temp_timebin_regressors = mc.simulation.predictions.create_x_regressors_per_state_simulation(temp_walk, temp_steps_per_walk, step_time, no_regs_per_state = no_bins_per_state)
            if hrf == True:
                temp_timebin_regressors = mc.simulation.predictions.convolve_with_hrf(temp_timebin_regressors, steps_per_walk, step_time, plotting = False)
            # temp_timebin_regressors = mc.simulation.predictions.subsample(temp_timebin_regressors, subsample_factor=11) 
            for curr_model in prep_model_dict:
                temp_prep_model_dict[curr_model] = mc.simulation.predictions.transform_data_to_betas(temp_prep_model_dict[curr_model], temp_timebin_regressors)
                temp_prep_model_dict[curr_model] = pd.DataFrame(temp_prep_model_dict[curr_model])
        
        for curr_model in temp_prep_model_dict:
            temp_prep_model_dict[curr_model] = pd.DataFrame(temp_prep_model_dict[curr_model]).fillna(0)
        temp_length_of_task = len(temp_prep_model_dict["loc_mod_df"].columns)
        
        
        # 3.: replace every of the 10 tasks with the new one and check if the similarity sum is lower.
        # prepare loop here 
        temp_similarity = 4
        config_no = -1
        
        while (abs(temp_similarity) > abs(similarity_between)) and (config_no < (no_tasks-1)):
            config_no+=1
            # identify which columns to cut
            temp_similarity = 4
            # if bin_data == True:
            #     if config_no == 0:
            #         cut_out_cols_min = config_no
            #         cut_out_cols_max = len(steps_per_walk)*no_bins_per_state # array that tells me how long each task is
            #     else:
            #         cut_out_cols_min = config_no * (len(steps_per_walk)*no_bins_per_state)
            #         cut_out_cols_max = config_no+1 * (len(steps_per_walk)*no_bins_per_state) 
                
            # elif bin_data == False:     
            if config_no == 0:
                cut_out_cols_min = config_no
                cut_out_cols_max = cum_length_per_task[config_no] # array that tells me how long each task is
            else:
                cut_out_cols_min = cum_length_per_task[config_no-1]
                cut_out_cols_max = cum_length_per_task[config_no]
            
            # cut the old data append the new.   
            
            temp_model_dict = {}
            for curr_model in model_dict:
                #import pdb; pdb.set_trace() 
                temp_model_dict[curr_model] = model_dict[curr_model].drop(model_dict[curr_model].iloc[:, cut_out_cols_min: cut_out_cols_max], axis = 1)
                temp_model_dict[curr_model] = pd.concat([temp_model_dict[curr_model], temp_prep_model_dict[curr_model]], axis = 1)
                #print(f"the length of {curr_model} is now {len(temp_model_dict[curr_model].columns)}")
                
            # delete the respective number, add the new column count 
            temp_length_all_tasks = np.delete(length_all_tasks, config_no)
            temp_length_all_tasks = np.append(temp_length_all_tasks, temp_length_of_task)
            # update cum_length_per_task
            temp_cum_length_all_tasks = np.cumsum(temp_length_all_tasks)
            
            # create new RDMs for the updated task configuration
            temp_RSM_dict = {}
            for curr_model in temp_model_dict:
                temp_RSM_dict[curr_model] = mc.simulation.RDMs.df_based_RDM(temp_model_dict[curr_model])
                temp_model_dict[curr_model].columns = range(temp_model_dict[curr_model].columns.size)
            
            temp_similarity_between_dict = {}
            for i, curr_RSM_one in enumerate(temp_RSM_dict):
                for j, curr_RSM_two in enumerate(temp_RSM_dict):
                    curr_corr = f"{curr_RSM_one}_with_{curr_RSM_two}"
                    temp_correlation = mc.simulation.RDMs.corr_matrices_pearson(temp_RSM_dict[curr_RSM_one], temp_RSM_dict[curr_RSM_two])
                    temp_similarity_between_dict[curr_corr] = temp_correlation[0,1]
                    
            # then create the sum of interest
            temp_similarity = 0
            # I want to make the sum of these 3 as low as possible.
            for elem in of_interest:
                temp_similarity = abs(temp_similarity) + abs(temp_similarity_between_dict[elem])
            
            if abs(temp_similarity) < abs(similarity_between):
                print(f'if we replace task at position {config_no}, similarity will go down to {temp_similarity}')
                print(f'replace {best_reward_coords[config_no]} with {temp_rew_coords}')
                print(f"the length of {curr_model} is now {len(temp_model_dict[curr_model].columns)}")
            
            
        # test if the new task config in this combination makes for a lower similarity (or if all tasks were tested)
        # and if so, take this configuration from now on.
        
        if abs(temp_similarity) < abs(similarity_between):
            # import pdb; pdb.set_trace()
            final_similarity_dict = temp_similarity_between_dict.copy()
            similarity_between = temp_similarity.copy()
            model_dict = temp_model_dict.copy()
            for curr_model in model_dict:
                model_dict[curr_model].columns = range(model_dict[curr_model].columns.size)
            
            length_all_tasks = temp_length_all_tasks.copy()
            cum_length_per_task = temp_cum_length_all_tasks.copy()
            
            # and update the reward and path info
            # drop the respective columns
            # DOUBLE CHECK IF I AM ACTUALLY DROPPING THE RIGHT ONE!!! 20.07.23
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
                  
        print(f'Finished perm {perm}, curr best sum of 3 correlations is {similarity_between}, temp_sim is {temp_similarity}, length of models is {len(model_dict[curr_model].columns)}')
    
    # DOUBLE CHECK IF IT WORKS!!!!20.07.23
    
    # show the final result RDMs.
    best_RSM_dict = {}
    for curr_model in model_dict:
        best_RSM_dict[curr_model] = mc.simulation.RDMs.df_based_RDM(model_dict[curr_model])
        plt.figure();
        plt.imshow(best_RSM_dict[curr_model], aspect = 'auto')
        
    print(f"done, yey! Final values are: {final_similarity_dict['clo_mod_df_with_loc_mod_df']} for clo_with_loc, {final_similarity_dict['clo_mod_df_with_midn_mod_df']} for clo with mind, {final_similarity_dict['clo_mod_df_with_phas_mod_df']} for clo with phase")


    return final_similarity_dict, model_dict, df_task_configs, best_reward_coords





















