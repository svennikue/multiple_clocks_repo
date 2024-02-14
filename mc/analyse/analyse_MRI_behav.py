#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 13:14:44 2023
This file is to open and clean my behavioural variables.

@author: xpsy1114
"""


import pandas as pd
import numpy as np
import mc
import matplotlib.pyplot as plt
import scipy.special as sps  
import statsmodels.api as sm
from nilearn.image import load_img
import os
import nibabel as nib
import statsmodels.api as sm
import rsatoolbox.data as rsd
from rsatoolbox.rdm.calc import _build_rdms
from rsatoolbox.rdm import RDMs

def print_stuff(string_input):
    print(string_input)
    
    

def jitter(expected_step_no):
    # first randomly sample from a gamma distribution
    shape = 5.75 # this is what the mean subpath is supposed to be
    draw = np.random.standard_gamma(shape)
    
    # then make an array for each step + reward I expect to take
    step_size_maker = np.random.randint(1, expected_step_no + 4, size= expected_step_no + 1)
    
    # make the last one, the reward, twice as long as the average step
    ave_step = np.mean(step_size_maker)
    step_size_maker[-1] = ave_step*2
    
    # then multiply the fraction of all step sizes with the actual subpath length
    stepsizes = np.empty(expected_step_no + 1)
    for i in range(expected_step_no+ 1):
        stepsizes[i] = (step_size_maker[i]/ (sum(step_size_maker))) * draw
        
    # stepsizes [-1] will be reward length. if more steps than stepsizes[0:-2], randomly sample.
    
    return(stepsizes)



    # plotting how I draw the randomly jittered steps
    
    # # first randomly sample from a gamma distribution
    # # or from an exponantial
    # # then sample no of optimal steps random numbers 
    # # e.g. if 3 random numbers
    # # a/(a+b+c) * randomly sampled goal
    # # then 
    repeats = 10000
    shape, scale = 5.75, 1. # mean and width
    s = np.empty(repeats)
    for i in range(repeats):
        draw = np.random.standard_gamma(shape)
        while (draw < 3) or (draw > 15):
            draw = np.random.standard_gamma(shape)
        s[i] = draw
    
    step_no = 3
    step_size_maker = np.random.randint(1, step_no + 4, size= step_no + 1)
    ave_step = np.mean(step_size_maker)
    step_size_maker[-1] = ave_step*2
    
    # Find the index of the maximum value in the array
    # max_index = np.argmax(step_size_maker) 
    # Swap the maximum value with the last element
    # step_size_maker[max_index], step_size_maker[-1] = step_size_maker[-1], step_size_maker[max_index]

    stepsizes = np.empty(step_no + 1)
    for i in range(step_no+ 1):
        stepsizes[i] = (step_size_maker[i]/ (sum(step_size_maker))) * draw
        
    print (f'Step 1 = {stepsizes[0]} Step 2 = {stepsizes[1]} Step 3 = {stepsizes[1]}, rew = {stepsizes[-1]}, sum = {sum(stepsizes)}')
    
    plt.figure()
    count, bins, ignored = plt.hist(s, 50, density=True)
    y = bins**(shape-1) * ((np.exp(-bins/scale))/(sps.gamma(shape) * scale**shape))
    plt.plot(bins, y, linewidth=2, color='r')  
    plt.show()
    
    

# code snippet to create a regressor
def create_EV(onset, duration, magnitude, name, folder, TR_at_sec):
    if len(onset) > len(duration):
        onset = onset[:len(duration)]
        magnitude = magnitude[:len(duration)]
    elif len(duration) > len(onset):
        duration = onset[:len(onset)]
        magnitude = magnitude[:len(onset)]
    regressor_matrix = np.ones((len(magnitude),3))
    regressor_matrix[:,0] = [(time - TR_at_sec) for time in onset]
    regressor_matrix[:,1] = duration
    regressor_matrix[:,2] = magnitude
    # import pdb; pdb.set_trace()
    np.savetxt(str(folder) + 'ev_' + str(name) + '.txt', regressor_matrix, delimiter="    ", fmt='%f')
    return regressor_matrix


# to transform the locations
def transform_coord(coord, is_x = False, is_y = False):
    if is_x:
        if coord == -0.21:
            return 0
        elif coord == 0:
            return 1
        elif coord == 0.21:
            return 2
    if is_y:
        if coord == -0.29:
            return 0
        elif coord == 0:
            return 1
        elif coord == 0.29:
            return 2
    # Add more conditions if needed
    else:
        return None



# use to check if the EV making went wrong
def check_for_nan(array):
    count = 0
    while np.isnan(array).any():
        print(f"Careful! There are Nans in {array}. Pausing script")
        # import pdb; pdb.set_trace()
        # try if this is sensible: delete the rows with the nans.
        array = array[0: (len(array)-1)]
        count = count + 1
    if count > 0:   
        print(f"deteleted {count} rows to avoid nans.")
    return count, array


def make_loc_EV(dataframe, x_coord, y_coord):
    # import pdb; pdb.set_trace()

    skip_next = False
    loc_dur = []
    loc_on = []
    loc_df = dataframe[(dataframe['curr_loc_x'] == x_coord) & (dataframe['curr_loc_y'] == y_coord)]
    #loc_one_on = loc_one['t_step_press_global'].to_list()
    # import pdb; pdb.set_trace()
    # try if this one works.
    # look at to check if its really the same task. For this, create a reward type 
    # column which allows to differentiate all trials
    loc_df['config_type'] = loc_df['task_config'] + '_' + loc_df['type']
    loc_df['config_type'] = loc_df['config_type'].fillna(method='ffill')
    for index, row in loc_df.iterrows():
        if index > 0: 
            if skip_next:
                skip_next = False
                continue
            # first case: a new repeat hast just started.
            if not np.isnan(row['next_task']): 
                start = dataframe.at[index, 'start_ABCD_screen']
                duration = dataframe.at[index, 't_step_press_global'] - start
                
            # second case: it is a reward. This can never be in 'next task', so else.
            elif not np.isnan(dataframe.at[index,'rew_loc_x']):
                if index+2 < len(dataframe): # only do this if this isn't the last row
                    # so here is a difference between reward A,B,C and D.
                    start = dataframe.at[index, 't_reward_start'] 
                    if row['state'] != 'D':
                        duration = dataframe.at[index + 1, 't_step_press_global'] - start
                    elif row['state'] == 'D':
                        # in case the next reward config is a different one
                        if row['config_type'] != dataframe.at[index+1, 'task_config']:
                            duration = dataframe.at[index, 't_reward_afterwait'] - start
                        # but if its just one repeat of many, it's more precise to take:
                        else:
                            duration = dataframe.at[index+1, 'start_ABCD_screen'] - start
                            # and if it was a 'D' within repeats, then skip the next row as it will be double otherwise!
                            skip_next = True
            # third case: its neither a new repeat, nor a reward.
            elif np.isnan(dataframe.at[index,'rew_loc_x']) and np.isnan(row['next_task']):
                start = dataframe.at[index-1, 't_step_press_global']
                duration = dataframe.at[index, 't_step_press_global'] - start
                
            loc_on.append(start)
            loc_dur.append(duration)
            
    loc_mag = np.ones(len(loc_on))
        
    return(loc_on, loc_dur, loc_mag)   
            

            
# FMRI ANALYSIS

# potentially delete this one
# def my_eval(model, data):
#       "Handle one voxel, copy the code that exists already for the neural data"
#       X = sm.add_constant(model.rdm.transpose());
#       Y = data.dissimilarities.transpose();
#       est = sm.OLS(Y, X).fit()
#       # import pdb; pdb.set_trace()
#       return est.tvalues[1:], est.params[1:], est.pvalues[1:]

# write a visualisation function for data RDMs.
def visualise_data_RDM(mni_x, mni_y, mni_z, data_RDM_file, mask):
    # import pdb; pdb.set_trace()
    x, y, z = mask.shape
    index_centre = np.ravel_multi_index((mni_x, mni_y, mni_z), (x,y,z))
    index_RDM = np.where(data_RDM_file.rdm_descriptors['voxel_index']==index_centre)[0]
    RDM_I_want= data_RDM_file[index_RDM].dissimilarities
    
    # matrix_40x40 = np.zeros((40, 40))

    # # Function to fill in the lower triangular part of the matrix
    # def fill_lower_triangular(matrix, data):
    #     indices = np.tril_indices_from(matrix)
    #     matrix[indices] = data
    
    # # Call the function to fill your matrix
    # fill_lower_triangular(matrix_40x40, RDM_I_want)

    # I believe that the RDMs are 80,93 - no actually its 40,93 for the half
    # well actually, an RDM should be no_conditions * no_conditions
    # why the hell is it 0,780??? this should be from 39x39 not 40x40???
    
    
    

def save_RSA_result(result_file, data_RDM_file, file_path, file_name, mask, number_regr, ref_image_for_affine_path):
    x, y, z = mask.shape
    ref_img = load_img(ref_image_for_affine_path)
    affine_matrix = ref_img.affine
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    # import pdb; pdb.set_trace() 
    
    t_result_brain = np.zeros([x*y*z])
    t_result_brain[list(data_RDM_file.rdm_descriptors['voxel_index'])] = [vox[0][number_regr] for vox in result_file]
    t_result_brain = t_result_brain.reshape([x,y,z])
    
    t_result_brain_nifti = nib.Nifti1Image(t_result_brain, affine=affine_matrix)
    t_result_brain_file = f"{file_path}/t_val_{file_name}.nii.gz"
    nib.save(t_result_brain_nifti, t_result_brain_file)
    
    b_result_brain = np.zeros([x*y*z])
    b_result_brain[list(data_RDM_file.rdm_descriptors['voxel_index'])] = [vox[1][number_regr] for vox in result_file]
    b_result_brain = b_result_brain.reshape([x,y,z])
    
    b_result_brain_nifti = nib.Nifti1Image(b_result_brain, affine=affine_matrix)
    b_result_brain_file = f"{file_path}/beta_{file_name}.nii.gz"
    nib.save(b_result_brain_nifti, b_result_brain_file)
    
    p_result_brain = np.zeros([x*y*z])
    p_result_brain[list(data_RDM_file.rdm_descriptors['voxel_index'])] = [1 - vox[2][number_regr] for vox in result_file]
    p_result_brain = p_result_brain.reshape([x,y,z])
    
    p_result_brain_nifti = nib.Nifti1Image(p_result_brain, affine=affine_matrix)
    p_result_brain_file = f"{file_path}/p_val_{file_name}.nii.gz"
    nib.save(p_result_brain_nifti, p_result_brain_file)


def evaluate_model(model, data):
    # import pdb; pdb.set_trace()
    
    X = sm.add_constant(model.rdm.transpose());
    Y = data.dissimilarities.transpose();
    est = sm.OLS(Y, X).fit()
    # import pdb; pdb.set_trace()
    return est.tvalues[1:], est.params[1:], est.pvalues[1:]
    


def prepare_model_data(model_data, number_conditions):
    model_data = model_data.transpose()
    nCond = model_data.shape[0]/2
    nVox = model_data.shape[1]
    sessions = np.concatenate((np.zeros(int(np.shape(model_data)[0]/2)), np.ones(int(np.shape(model_data)[0]/2))))
    des = {'subj': 1}
    conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(nCond)])), (1,2)).transpose(),number_conditions*2)
    obs_des = {'conds': conds, 'sessions': sessions}
    chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
    RSA_tb_model_data_object = rsd.Dataset(measurements=model_data,
                       descriptors=des,
                       obs_descriptors=obs_des,
                       channel_descriptors=chn_des)
    return RSA_tb_model_data_object




def analyse_pathlength_beh(df):
    # identify where the next task begins by iterating through the DataFrame 
    # and collecting the indices where the column is not empty
    index_next_task = []
    for index, row in df.iterrows():
        if not pd.isna(row['start_ABCD_screen']):
            index_next_task.append(index)
    
    # compute the task length for each task
    # careful! this only works if the task was completed.
    # also this isn't super precise since it doesn't actually show where they 
    # walked but where they were able to move away from reward
    for i, index in enumerate(index_next_task):
        if i+1 < len(index_next_task):
            df.at[index, 'task_length'] = df.at[index_next_task[i+1] - 1 , 't_reward_afterwait'] - df.at[index, 'start_ABCD_screen']   
            if 'type' in df.columns:
                df.at[index, 'type'] = df.at[index+ 1, 'type']
        elif i+1 == len(index_next_task):
            df.at[index, 'task_length'] = df.at[len(df)-1, 't_reward_afterwait'] - df.at[index, 'start_ABCD_screen'] 
                    
    # not sure why I included this... seems wrong.      
    # index_next_task = index_next_task[1:]
                    
    # identify where the next reward starts by iterating through the DataFrame 
    # and collecting the indices where the column is not empty
    index_next_reward = []
    for index, row in df.iterrows():
        if not pd.isna(row['t_reward_start']):
            index_next_reward.append(index)

    # Update 06.10.23: I don't think I need this anymore, I fixed it in the exp code
    # fill the missing last reward_delay columns.
    # they should be t_reward_afterwait-t_reward_start
    # take every 4th reward index to do so.
    #for i in range(3, len(index_next_reward), 4):
    #   df.at[index_next_reward[i], 'reward_delay'] = df.at[index_next_reward[i], 't_reward_afterwait'] - df.at[index_next_reward[i], 't_reward_start'] 
    
    # fill gaps in the round_no column
    df['round_no'] = df['round_no'].fillna(method='ffill')
    # do the same for the task_config 
    df['task_config'] = df['task_config'].fillna(method='ffill')
    # and create a reward type column which allows to differentiate all trials
    df['config_type'] = df['task_config'] + '_' + df['type']
    df['config_type'] = df['config_type'].fillna(method='ffill')
                
    
    # import pdb; pdb.set_trace()
    # create a new column in which you plot how long ever subpath takes (with rew)
    j = 0
    for i, task_index in enumerate(index_next_task):
        if task_index > 1:
            while (len(index_next_reward) > j) and (index_next_reward[j] < task_index):
                df.at[index_next_reward[j], 'cum_subpath_length_without_rew'] = df.at[index_next_reward[j], 't_step_press_curr_run'] + df.at[index_next_reward[j]-1, 'length_step'] 
                df.at[index_next_reward[j], 'cum_subpath_length_with_rew'] = df.at[index_next_reward[j], 't_step_press_curr_run'] + df.at[index_next_reward[j]-1, 'length_step'] + df.at[index_next_reward[j], 'reward_delay'] 
                j += 1
            df.at[task_index-1, 'cum_subpath_length_without_rew'] = df.at[index_next_task[i-1], 'task_length'] - df.at[task_index-1, 'reward_delay']
            df.at[task_index-1, 'cum_subpath_length_with_rew'] = df.at[index_next_task[i-1], 'task_length']
            # df.at[task_index-1, 't_step_press_curr_run'] + df.at[task_index-2, 'length_step'] + df.at[task_index-1, 'reward_delay'] 
        # for the next reward count backwards
        if task_index == index_next_task[-1]:
            for i in range(4,0, -1):
                df.at[index_next_reward[-i], 'cum_subpath_length_without_rew']= df.at[index_next_reward[-i], 't_step_press_curr_run'] + df.at[index_next_reward[-i]-1, 'length_step'] 
                df.at[index_next_reward[-i], 'cum_subpath_length_with_rew']= df.at[index_next_reward[-i], 't_step_press_curr_run'] + df.at[index_next_reward[-i]-1, 'length_step'] + df.at[index_next_reward[-i], 'reward_delay']

    states = ['A', 'B', 'C', 'D']*len(index_next_task)
    
    
    
    # then, write the not- cumulative columns.
    for i, reward_index in enumerate(index_next_reward):
        if i < len(states):
            df.at[reward_index, 'state'] = states[i]
        if i > 0:
            df.at[reward_index, 'subpath_length_without_rew'] = df.at[reward_index, 'cum_subpath_length_without_rew'] - df.at[index_next_reward[i-1], 'cum_subpath_length_with_rew']
            df.at[reward_index, 'subpath_length_with_rew'] = df.at[reward_index, 'cum_subpath_length_with_rew'] - df.at[index_next_reward[i-1], 'cum_subpath_length_with_rew']

    for i in range(0, len(index_next_reward), 4):
        df.at[index_next_reward[i], 'subpath_length_without_rew'] = df.at[index_next_reward[i], 'cum_subpath_length_without_rew'] 
        df.at[index_next_reward[i], 'subpath_length_with_rew'] = df.at[index_next_reward[i], 'cum_subpath_length_with_rew']

    
    #first reduce to only including those rows that have values for rewards.
    df_clean = df.dropna(subset = ['subpath_length_with_rew'])
    
    return(df, df_clean)



# CONTINUE HERE LATER!!
def similarity_of_tasks(reward_per_task_per_taskhalf_dict):
    #import pdb; pdb.set_trace()

    all_rewards = []
    for task_half in reward_per_task_per_taskhalf_dict:
        #  make sure that the dictionary is alphabetically sorted.
        for task in sorted(reward_per_task_per_taskhalf_dict[task_half].keys()):
            all_rewards.append(reward_per_task_per_taskhalf_dict[task_half][task])
    
    
    task_similiarity = np.zeros((len(all_rewards), len(all_rewards)))
    
    for i in range(len(all_rewards)):
        for j in range(len(all_rewards)):
            if all_rewards[i] == all_rewards[j]:
                task_similiarity[i, j] = 1

    # np.corrcoef(task_similiarity[:, :10])
    # corrected_model = (task_similiarity[:, :10] + np.transpose(task_similiarity[:, :10]))/2
    # corrected_RSM_dict[model] = corrected_model[0:int(len(corrected_model)/2), int(len(corrected_model)/2):]
   
        
    # to create the right format, split this into two task halves again
    models_between_tasks = {'instruction': {key: "" for key in ['1', '2']}}
    
    models_between_tasks['instruction']['1'] = task_similiarity[:10, :10]
    models_between_tasks['instruction']['2'] = task_similiarity[10:20, :10]

    return models_between_tasks


