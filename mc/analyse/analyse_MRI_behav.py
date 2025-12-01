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
import rsatoolbox.data as rsd
from rsatoolbox.rdm.calc import _build_rdms
from rsatoolbox.rdm import RDMs
import shutil
from datetime import datetime
import rsatoolbox
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import ast
import re


def extend_for_more_evs(text_to_write, sorted_EVs, n_EVs, max_EVs_og_fsf):
    n_EVs_to_insert = n_EVs - max_EVs_og_fsf
    
    # if I have more EV's than in the OG template, I need to rewrite the file.
    # Plan is: doublicate the last EV 81. 
    # copy # EV 81 title
    # to 
    # set fmri(ortho81.81) 0
    for idx, line in enumerate(text_to_write):
        if "# EV 81 title" in line:
            start_last_EV = int(idx)
        if "set fmri(ortho81.81) 0" in line:
            end_last_EV = int(idx)
    copied_max_EV = text_to_write[start_last_EV: end_last_EV+1]
    # next, replace the incorrect indices and names with the correct ones
    idx_to_change, row_to_change = [], []
    for idx, line in enumerate(copied_max_EV):
        if str(max_EVs_og_fsf) in line:
            idx_to_change.append(idx)
            row_to_change.append(line)
    
    # import pdb; pdb.set_trace() 
    additional_EVs = []
    for additional_EV in range(max_EVs_og_fsf, max_EVs_og_fsf+n_EVs_to_insert):
        adjusted_EV = copied_max_EV.copy()
        EV_path = sorted_EVs[additional_EV]
        EV_name_ext = os.path.basename(EV_path)
        EV_name = EV_name_ext.rsplit('.',1)[0]
        # now, most bits are updated with the correct number.
        for row_idx in idx_to_change:
            # note that this will also change both numbers in fmri(ortho81.81)
            adjusted_EV[row_idx] = adjusted_EV[row_idx].replace("81", str(additional_EV+1))
            # also exchange the title
            if adjusted_EV[row_idx].startswith("set fmri(evtitle"):
                adjusted_EV[row_idx] = f'set fmri(evtitle{additional_EV+1}) "{EV_name}"\n' 
            # exchange path  
            if adjusted_EV[row_idx].startswith("set fmri(custom"):
                # print(f"my old line was: {line}")
                adjusted_EV[row_idx] = f'set fmri(custom{additional_EV+1}) "{EV_path}"\n'
     
         
        # CAREFUL! this needs to happen for all EVs, not only the new ones.
        
        # add the orthogonalisation of the additional EVs
        last_lines_appendix = []
        for add_EV_idx in range(max_EVs_og_fsf, max_EVs_og_fsf + n_EVs_to_insert+1):
            last_lines_appendix.extend([
                "\n",
                f"# Orthogonalise EV {additional_EV+1} wrt EV {add_EV_idx}\n",
                f"set fmri(ortho{additional_EV+1}.{add_EV_idx}) 0\n",
            ])
        last_lines_appendix.extend("\n")
        adjusted_EV[-3:] = last_lines_appendix
        # additional_EVs.append(adjusted_EV)
        additional_EVs.extend(adjusted_EV)

    
    insert_pos = end_last_EV +2
    # this might need to be flatten first, maybe by just doing extend its fine
    text_to_write[insert_pos:insert_pos] = additional_EVs
    # import pdb; pdb.set_trace() 
    # only the old EVs need to be adjusted now.
    for EV_idx in range(1, max_EVs_og_fsf+1):
        # print(EV_idx)
        for li, line in enumerate(text_to_write):
            if f"# Orthogonalise EV {EV_idx} wrt EV {max_EVs_og_fsf}\n" in line:
                append_ortho_EV = []
                for add_EV_idx in range(max_EVs_og_fsf, max_EVs_og_fsf + n_EVs_to_insert+1):
                    append_ortho_EV.extend([
                        f"# Orthogonalise EV {EV_idx} wrt EV {add_EV_idx}\n",
                        f"set fmri(ortho{EV_idx}.{add_EV_idx}) 0\n",
                        "\n"
                    ])
                # here I replace the text.
                # print(f"now adding lines {append_ortho_EV}")
                text_to_write[li:li+3] = append_ortho_EV
                break
            
        # # add the orthogonalisation of the additional EVs
        # last_lines_appendix = []
        # for add_EV_idx in range(max_EVs_og_fsf, max_EVs_og_fsf + n_EVs_to_insert+1):
        #     last_lines_appendix.extend([
        #         "\n",
        #         f"# Orthogonalise EV {additional_EV+1} wrt EV {add_EV_idx}\n",
        #         f"set fmri(ortho{additional_EV+1}.{add_EV_idx}) 0\n",
        #     ])
        
    # the second chunk that needs to be adjusted
    # there are 6 contrast vectors. 
    # for contrast_real vector, these go to element 82.
    # for contrast_orig vector, these go to element 81.
    
    # so, per vector, find the row in question, and add the +n_EVs_to_insert elements
    for vec in [1,2,3,4,5,6]:
        # for contrast_orig vector, these go to element 81.
        for idx, line in enumerate(text_to_write):
            if f"# Real contrast_orig vector {vec} element {max_EVs_og_fsf}" in line:
                append_contrast_orig = []
                for add_EV_idx in range(max_EVs_og_fsf, max_EVs_og_fsf + n_EVs_to_insert+1):
                    append_contrast_orig.extend([
                        f'# Real contrast_orig vector {vec} element {add_EV_idx}\n', 
                        f'set fmri(con_orig{vec}.{add_EV_idx}) 0\n', 
                        '\n'
                            ])
                text_to_write[idx:idx+3] = append_contrast_orig
                break
    
    # enter the loop again for contrast_real vectors.
    for vec in [1,2,3,4,5,6]:
        # for contrast_real vector, these go to element 82.
        for idx, line in enumerate(text_to_write):
            if f"# Real contrast_real vector {vec} element {max_EVs_og_fsf+1}" in line:
                append_contrast_real = []
                for add_EV_idx in range(max_EVs_og_fsf, max_EVs_og_fsf + n_EVs_to_insert+1):
                    append_contrast_real.extend([
                        f'# Real contrast_real vector {vec} element {add_EV_idx+1}\n', 
                        f'set fmri(con_real{vec}.{add_EV_idx+1}) 0\n', 
                        '\n'
                            ])
                text_to_write[idx:idx+3] = append_contrast_real
                break
                
    print(f"most recent stiched index at {idx}")
    # import pdb; pdb.set_trace()     
    # # do this differently.
    # for add_contr_idx in range(max_EVs_og_fsf+2, max_EVs_og_fsf+2 + n_EVs_to_insert):
    #     #print(f"now for EV {add_contr_idx}")
    #     for vec in [1,2,3,4,5,6]:
    #         #print(f"now for vector {vec}")
    #         for idx, line in enumerate(text_to_write):
    #             if f"# Real contrast_real vector {vec} element {add_contr_idx-1}" in line:
    #                 idx_to_insert = idx + 3
    #                 print(idx_to_insert)
    #                 copied_contrast_vec = text_to_write[idx: idx + 3]
    #                 # next, replace the incorrect index (82) with the correct ones
    #                 for row in range(0, len(copied_contrast_vec)):
    #                     copied_contrast_vec[row] = copied_contrast_vec[row].replace(str(add_contr_idx-1), str(add_contr_idx))
    #                 #print(f"now inserting {copied_contrast_vec}")
                    
    #                 #text_to_write[idx_to_insert:idx_to_insert] = copied_contrast_vec
    #                 # i'm not actually replacing the initial lines, so put a breakpoint here to avoid 
    #                 # unlimited loops
    #                 break


    
    
    
    
    # for add_contr_idx in range(max_EVs_og_fsf+2, max_EVs_og_fsf+2 + n_EVs_to_insert):
    #     #print(f"now for EV {add_contr_idx}")
    #     for vec in [1,2,3,4,5,6]:
    #         #print(f"now for vector {vec}")
    #         for idx, line in enumerate(text_to_write):
    #             if f"# Real contrast_orig vector {vec} element {add_contr_idx-2}" in line:
    #                 print(f"# Real contrast_orig vector {vec} element {add_contr_idx-2}")
    #                 idx_to_insert = idx + 3
    #                 print(idx_to_insert)
    #                 copied_contrast_vec = text_to_write[idx: idx + 3]
    #                 # next, replace the incorrect index (82) with the correct ones
    #                 for row in range(0, len(copied_contrast_vec)):
    #                     copied_contrast_vec[row] = copied_contrast_vec[row].replace(str(add_contr_idx-2), str(add_contr_idx-1))
    #                 print(f"now inserting {copied_contrast_vec}")
                    
    #                 # text_to_write[idx_to_insert:idx_to_insert] = copied_contrast_vec
    #                 # i'm not actually replacing the initial lines, so put a breakpoint here to avoid 
    #                 # unlimited loops
    #                 break
    #             if f"# Real contrast_real vector {vec} element {add_contr_idx-1}" in line:
    #                 idx_to_insert = idx + 3
    #                 print(idx_to_insert)
    #                 copied_contrast_vec = text_to_write[idx: idx + 3]
    #                 # next, replace the incorrect index (82) with the correct ones
    #                 for row in range(0, len(copied_contrast_vec)):
    #                     copied_contrast_vec[row] = copied_contrast_vec[row].replace(str(add_contr_idx-1), str(add_contr_idx))
    #                 #print(f"now inserting {copied_contrast_vec}")
                    
    #                 #text_to_write[idx_to_insert:idx_to_insert] = copied_contrast_vec
    #                 # i'm not actually replacing the initial lines, so put a breakpoint here to avoid 
    #                 # unlimited loops
    #                 break

                
    # import pdb; pdb.set_trace()           
    return text_to_write
    





def check_GLM_regressors(design_matrix_X):
    # to check if a GLM is ill-conditioned
    # To check that you can check the “condition number” of the design matrix - 
    # the ration between the maximum singular value (similar to eigenvalue) and the minimum singular value.. 
    # If that ration is close to 1, you’re good. If it’s very large (e.g. >1000), it means the matrix is ill-conditioned - 
    # one of your regressors is close to being a linear combination of the other two.
    
    # Assume X is your design matrix
    # Compute the Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(design_matrix_X, full_matrices=False)
    # Compute the condition number
    condition_number = np.max(S) / np.min(S)
    print(f"Condition number: {condition_number}")
    # Interpret the condition number
    if condition_number < 1000:
        print("The design matrix is well-conditioned.")
    else:
        print("The design matrix is ill-conditioned.")


def any_entry_in_row_notnan(entry):
    if isinstance(pd.notna(entry), list):
        x = True
    elif isinstance(pd.notna(entry), (str, int)):
        if pd.notna(entry):
           x = True
        else:
            x = False
    else:
        x = False
    return x

        
def determine_index_by_reg_version(reg_v, step_no):
    indx_no = np.array(range(len(step_no)))
    # but only consider some of the repeats for the only later or only early trials!
    if reg_v in ['03-e', '03-4-e']:
        # step_number = step_number[0:2].copy()
        indx_no = indx_no[0:2].copy()
    elif reg_v in ['03-l', '03-4-l']:
        # step_number = step_number[2:].copy()
        indx_no = indx_no[2:].copy()
    elif reg_v in ['03-rep1', '03-4-rep1']:
        indx_no = [indx_no[0]]
    elif reg_v in ['03-rep2', '03-4-rep2']:
        indx_no = [indx_no[1]]
    elif reg_v in ['03-rep3', '03-4-rep3']:
        indx_no = [indx_no[2]]
    elif reg_v in ['03-rep4', '03-4-rep4']:
        indx_no = [indx_no[3]]
    elif reg_v in ['03-rep5', '03-4-rep5']:
        indx_no = [indx_no[4]]
    return indx_no
    

def load_and_prep_behaviour_df(path_to_file):
    df = pd.read_csv(path_to_file)
    # the first row is empty so delete to get indices right
    df = df.iloc[1:].reset_index(drop=True)
    # fill gapss
    df['round_no'] = df['round_no'].fillna(method='ffill')
    df['task_config'] = df['task_config'].fillna(method='ffill')
    df['repeat'] = df['repeat'].fillna(method='ffill')
    # so that I cann differenatiate task config and direction
    df['config_type'] = df['task_config'] + '_' + df['type']
    # add columns whith field numbers 
    for index, row in df.iterrows():
        # current locations
        df.at[index, 'curr_loc_y_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_loc_y'], is_y=True, is_x = False)
        df.at[index, 'curr_loc_x_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_loc_x'], is_x=True, is_y = False)
        df.at[index, 'curr_rew_y_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_rew_y'], is_y=True, is_x = False)
        df.at[index, 'curr_rew_x_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_rew_x'], is_x=True, is_y = False)
        # and prepare the regressors: config type, state and reward/walking specific.
        if not pd.isna(row['state']):
            if not np.isnan(row['rew_loc_x']):
                df.at[index, 'time_bin_type'] =  df.at[index, 'config_type'] + '_' + df.at[index, 'state'] + '_reward'
            elif np.isnan(row['rew_loc_x']):
                df.at[index, 'time_bin_type'] = df.at[index, 'config_type'] + '_' + df.at[index, 'state'] + '_path'
    return df



def collect_behaviour_for_simulation(df):
    configs = df['config_type'].dropna().unique()
    behavioural_vars = ['walked_path', 'timings', 'rew_list', 'rew_timing', 'rew_index', 'subpath_after_steps', 'steps_subpath_alltasks', 'rew_index', 'subpath_after_steps']
    behaviour = {}
    for var in behavioural_vars:
        behaviour[var] = {}
        for config in configs:
            behaviour[var][config] = []

    for index, row in df.iterrows():
        task_config = row['config_type']
        
        # in case a new task has just started
        if not np.isnan(row['next_task']): 
            # first check if this is the first task of several repeats.
            if (index == 0) or (row['config_type'] != df.at[index -1, 'config_type']):
                behaviour['timings'][task_config].append(row['next_task'])
            else: # if it isnt, then take the reward start time from last rew D as start field.
                behaviour['timings'][task_config].append(df.at[index -1, 't_step_press_global'])
            behaviour['walked_path'][task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
        
        # if this is just a normal walking field
        elif not np.isnan(row['t_step_press_global']): # always except if this is reward D 
            # if its reward D, then it will be covered by the first if: if not np.isnan(row['next_task']): 
            behaviour['timings'][task_config].append(df.at[index - 1, 't_step_press_global'])  # Extract value from index-1
            behaviour['walked_path'][task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
       
        # next check if its a reward field
        if not np.isnan(row['rew_loc_x']): # if this is a reward field.
            # check if this is either at reward D(thus complete) or ignore interrupted trials
            # ignore these as they are not complete.
            if (index+2 < len(df)) or (row['state'] == 'D'):
                behaviour['rew_timing'][task_config].append(row['t_reward_start'])
                behaviour['rew_list'][task_config].append([row['curr_rew_x_coord'], row['curr_rew_y_coord']])
                behaviour['subpath_after_steps'][task_config].append(int(index-row['repeat']))  
                if row['state'] == 'D':
                    behaviour['rew_index'][task_config].append(len(behaviour['walked_path'][task_config])) #bc step has not been added yet
                    # if this is the last run of a task
                    if (index+2 < len(df)):
                        # first check if there are more tasks coming after, otherwise error
                        if (row['config_type'] != df.at[index +1, 'config_type']):
                            behaviour['walked_path'][task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                            behaviour['timings'][task_config].append(df.at[index -1, 't_reward_start'])
                    else:
                        # however also add these fields if this is the very last reward!
                        if row['repeat'] == 4:
                            behaviour['walked_path'][task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                            behaviour['timings'][task_config].append(df.at[index -1, 't_step_press_global'])
                            
                else:
                    behaviour['rew_index'][task_config].append(len(behaviour['walked_path'][task_config])-1) 
            else:
                continue
                           
    return behaviour



def get_conditions_list(RDM_dir):
    # import pdb; pdb.set_trace()
    # load the file which defines the order of the model RDMs, and hence the data RDMs
    with open(f"{RDM_dir}/sorted_keys-model_RDMs.pkl", 'rb') as file:
        sorted_keys = pickle.load(file)
    with open(f"{RDM_dir}/sorted_regs.pkl", 'rb') as file:
        reg_keys = pickle.load(file)
    list_of_conditions = {}
    list_of_conditions_flat = {}
    for split in sorted_keys:
        list_of_conditions[split] = []
        list_of_conditions_flat[split] = []
        for EV_no, task in enumerate(sorted_keys[split]):
            for regressor_sets in reg_keys:
                if regressor_sets[0].startswith(task):
                    list_of_conditions[split].append(regressor_sets)           
        list_of_conditions_flat[split] = [item for sublist in list_of_conditions[split] for item in sublist]
        
    return list_of_conditions_flat



def read_in_RDM_conds(regression_version, RDM_version, data_dir, RDM_dir, no_RDM_conditions, ref_img = None, sort_as = 'dict-two-halves'):
    # NOTE 17.10.2025
    # this is way too long and complicated.
    # the new one is in analyse > my_RSA> load_Data_EVs and is VERY simple.
    
    # sort_as can be 'dict-two-halves' (for volumetric data) or 'concat-surface' (for surface)
    # load the file which defines the order of the model RDMs, and hence the data RDMs
    
    # load the file which defines the order of the model RDMs, and hence the data RDMs
    with open(f"{RDM_dir}/sorted_keys-model_RDMs.pkl", 'rb') as file:
        sorted_keys = pickle.load(file)
    with open(f"{RDM_dir}/sorted_regs.pkl", 'rb') as file:
        reg_keys = pickle.load(file)
        
    # also store 2 dictionaries of the EVs
    if regression_version in ['03-3', '03-4']:
        regression_version = '03'
    if regression_version in ['04-4']:
        regression_version = '04'
    if regression_version in ['03-4-e']:
        regression_version = '03-e'
    if regression_version in ['03-4-l']:
        regression_version = '03-l'
    if regression_version in ['03-4-rep1']:
        regression_version = '03-rep1'
    if regression_version in ['03-4-rep2']:
        regression_version = '03-rep2'
    if regression_version in ['03-4-rep3']:
        regression_version = '03-rep3'
    if regression_version in ['03-4-rep4']:
        regression_version = '03-rep4'
    if regression_version in ['03-4-rep5']:
        regression_version = '03-rep5'
       

    pe_path_01 = f"{data_dir}/func/glm_{regression_version}_pt01.feat/stats"
    reading_in_EVs_dict_01 = {}   
    with open(f"{data_dir}/func/EVs_{regression_version}_pt01/task-to-EV.txt", 'r') as file:
        for line in file:
            index, name_ev = line.strip().split(' ', 1)
            name = name_ev.replace('ev_', '')
            reading_in_EVs_dict_01[f"{name}_EV_{int(index)+1}"] = os.path.join(pe_path_01, f"pe{int(index)+1}.nii.gz")
            
    pe_path_02 = f"{data_dir}/func/glm_{regression_version}_pt02.feat/stats"     
    reading_in_EVs_dict_02 = {}
    with open(f"{data_dir}/func/EVs_{regression_version}_pt02/task-to-EV.txt", 'r') as file:
        for line in file:
            index, name_ev = line.strip().split(' ', 1)
            name = name_ev.replace('ev_', '')
            reading_in_EVs_dict_02[f"{name}_EV_{int(index)+1}"] = os.path.join(pe_path_02, f"pe{int(index)+1}.nii.gz")
    
    
        
    print(sort_as)
    
    sorted_RDM_conds = []
    if sort_as == 'dict-two-halves':
        sorted_RDM_conds = {}
        data_RDM_file = {}
        data_RDM_file_1d = {}
        reading_in_EVs_dict = {}
        image_paths = {}
        # I want to be super careful that I create 2 *identical* splits of data.
        # A1 forwards = A2 backwards
        # A2 backwards = A1 forwards
        # etc.
        # thus, check which folder has the respective task.
        for split in sorted_keys:
            if RDM_version == '01':
                # DOUBLE CHECK IF THIS IS EVEN STILL CORRECT!!!
                # for condition 1, I am ignoring task halves. to make sure everything goes fine, use the .txt file
                # and only load the conditions in after the task-half loop.
                pe_path = f"{data_dir}/func/glm_{regression_version}_pt0{split}.feat/stats"
                with open(f"{data_dir}/func/EVs_{RDM_version}_pt0{split}/task-to-EV.txt", 'r') as file:
                    for line in file:
                        index, name = line.strip().split(' ', 1)
                        reading_in_EVs_dict[f"{name}_EV_{index}"] = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")
            else:  
                i = -1
                image_paths[split] = [None] * no_RDM_conditions # Initialize a list for each half of the dictionary
                data_RDM_file[split] = [None] * no_RDM_conditions  # Initialize a list for each half of the dictionary
                for EV_no, task in enumerate(sorted_keys[split]):
                    for regressor_sets in reg_keys:
                        if regressor_sets[0].startswith(task):
                            curr_reg_keys = regressor_sets
                    for reg_key in curr_reg_keys:
                        # print(f"now looking for {task}")
                        for EV_01 in reading_in_EVs_dict_01:
                            if EV_01.startswith(reg_key):
                                i = i + 1
                                # print(f"looking for {task} and found it in 01 {EV_01}, index {i}")
                                image_paths[split][i] = reading_in_EVs_dict_01[EV_01]  # save path to check if everything went fine later
                                data_RDM_file[split][i] = nib.load(reading_in_EVs_dict_01[EV_01]).get_fdata()
                        for EV_02 in reading_in_EVs_dict_02:
                            if EV_02.startswith(reg_key):
                                i = i + 1
                                # print(f"looking for {task} and found it in 01 {EV_02}, index {i}")
                                image_paths[split][i] = reading_in_EVs_dict_02[EV_02]
                                data_RDM_file[split][i] = nib.load(reading_in_EVs_dict_02[EV_02]).get_fdata() 
                                # Convert the list to a NumPy array
                
                print(f"This is the order now: {image_paths[split]}")
                data_RDM_file[split] = np.array(data_RDM_file[split])
                # reshape data so we have n_observations x n_voxels
                sorted_RDM_conds[split] = data_RDM_file[split].reshape([data_RDM_file[split].shape[0], -1])
                sorted_RDM_conds[split] = np.nan_to_num(sorted_RDM_conds[split]) # now this is 80timepoints x 746.496 voxels
                
                if RDM_version == f"{RDM_version}_999": # shuffle voxels randomly
                    data_RDM_file_1d[split] = sorted_RDM_conds[split].flatten()
                    np.random.shuffle(data_RDM_file_1d[split]) #shuffle all voxels randomly
                    sorted_RDM_conds[split] = data_RDM_file_1d[split].reshape(sorted_RDM_conds[split].shape) # and reshape
        
        if RDM_version in ['01']:
            data_RDM_file_2d = {}
            data_RDM_file = {}
            data_RDM_file[RDM_version] = [None] * no_RDM_conditions
            # sort across task_halves
            for i, task in enumerate(sorted(reading_in_EVs_dict.keys())):
                if task not in ['ev_press_EV_EV_index']:
                    image_paths[i] = reading_in_EVs_dict[task]
                    data_RDM_file[RDM_version][i] = nib.load(image_paths[i]).get_fdata()
            # Convert the list to a NumPy array
            data_RDM_file_np = np.array(data_RDM_file[RDM_version])
            # reshape data so we have n_observations x n_voxels
            data_RDM_file_2d = data_RDM_file_np.reshape([data_RDM_file_np.shape[0], -1])
            data_RDM_file_2d = np.nan_to_num(data_RDM_file_2d) # now this is 20timepoints x 746.496 voxels

            print(f"This is the order now: {image_paths}")  


    if sort_as == 'concat_list':
        ref_img_data = ref_img.get_fdata()
        fmri_img_list_first_half = np.empty((ref_img_data.shape[0], ref_img_data.shape[1], ref_img_data.shape[2], no_RDM_conditions*2))
        fmri_img_list_sec_half = np.empty((ref_img_data.shape[0], ref_img_data.shape[1], ref_img_data.shape[2], no_RDM_conditions*2))
        
        sorted_RDM_conds = []
        import pdb; pdb.set_trace()
        for split in sorted_keys:          
            i = -1
            image_paths[split] = [None] * no_RDM_conditions # Initialize a list for each half of the dictionary
            #data_RDM_file[split] = [None] * no_RDM_conditions  # Initialize a list for each half of the dictionary
            for EV_no, task in enumerate(sorted_keys[split]):
                for regressor_sets in reg_keys:
                    if regressor_sets[0].startswith(task):
                        curr_reg_keys = regressor_sets
                for reg_key in curr_reg_keys:
                    # print(f"now looking for {task}")
                    for EV_01 in reading_in_EVs_dict_01:
                        if EV_01.startswith(reg_key):
                            i = i + 1
                            # print(f"looking for {task} and found it in 01 {EV_01}, index {i}")
                            image_paths[split][i] = reading_in_EVs_dict_01[EV_01]  # save path to check if everything went fine later
                            fmri_img_list_first_half[:,:,:,i] = nib.load(reading_in_EVs_dict_01[EV_01]).get_fdata()
                            # if i == 0:
                            #     fmri_img = nib.load(reading_in_EVs_dict_01[EV_01])
                            # else:
                            #     next_EV = nib.load(reading_in_EVs_dict_01[EV_01])
                            #     fmri_img = nl.image.concat_imgs([fmri_img, next_EV])
                            #data_RDM_file[split][i] = nib.load(reading_in_EVs_dict_01[EV_01]).get_fdata()
                            
                    for EV_02 in reading_in_EVs_dict_02:
                        if EV_02.startswith(reg_key):
                            i = i + 1
                            # print(f"looking for {task} and found it in 01 {EV_02}, index {i}")
                            #data_RDM_file[split][i] = nib.load(reading_in_EVs_dict_02[EV_02]).get_fdata() 
                            image_paths[split][i] = reading_in_EVs_dict_02[EV_02]
                            fmri_img_list_sec_half[:,:,:,i] = nib.load(reading_in_EVs_dict_02[EV_02]).get_fdata()   

                            # fmri_img is nifti1.Nifti1Image (40,64,64,216)
                            # nifti1imiage object of nibabel.nifti1
                            # if i == 0:
                            #     fmri_img = nib.load(reading_in_EVs_dict_02[EV_02])
                            # else:
                            #     next_EV = nib.load(reading_in_EVs_dict_02[EV_02])
                            #     fmri_img = nl.image.concat_imgs([fmri_img, next_EV])

        fmri_img_pt1 = nib.Nifti1Image(fmri_img_list_first_half, affine=ref_img.affine, header=ref_img.header)               
        fmri_img_pt2 = nib.Nifti1Image(fmri_img_list_sec_half, affine=ref_img.affine, header=ref_img.header)               
        
        sorted_RDM_conds = np.stack((fmri_img_pt1,fmri_img_pt2),2)
        # I need to stack X tasklahf 1 and 2 like this; new_X = np.stack((X,X),2)
    
    return sorted_RDM_conds







def subpath_files(configs, subpath_after_steps, rew_list, rew_index, steps_subpath_alltasks):
    # import pdb; pdb.set_trace()
    for config in configs:
        rew_list[config] = [[int(value) for value in sub_list] for sub_list in rew_list[config][0:4]]
        # next step: create subpath files with rew_index and how many steps there are per subpath.
        # if task is completed
        if (len(subpath_after_steps[config])%4) == 0:
            for r in range(0, len(subpath_after_steps[config]), 4):
                subpath = subpath_after_steps[config][r:r+4]
                steps = [subpath[j] - subpath[j-1] for j in range(1,4)]
                if r == 0:
                    steps.insert(0, rew_index[config][r])
                if r > 0:
                    steps.insert(0, (subpath[0]- subpath_after_steps[config][r-1]))
                steps_subpath_alltasks[config].append(steps)
        # if task not completed
        elif (len(subpath_after_steps[config])%4) > 0:
            completed_tasks = len(subpath_after_steps[config])-(len(subpath_after_steps[config])%4)
            for r in range(0, completed_tasks, 4):
                subpath = subpath_after_steps[config][r:r+4]
                steps = [subpath[j] - subpath[j-1] for j in range(1,4)]
                if r == 0:
                    steps.insert(0, rew_index[config][r])
                if r > 0:
                    steps.insert(0, (subpath[0]- subpath_after_steps[config][r-1]))
                steps_subpath_alltasks[config].append(steps)
    
    return steps_subpath_alltasks


def extract_behaviour(file):
    # import pdb; pdb.set_trace()
    # load the two required excel sheets
    df = pd.read_csv(file)
    df_backup = df.copy()
    # the first row is empty so delete to get indices right
    df = df.iloc[1:].reset_index(drop=True)
    # fill gapss
    df['round_no'] = df['round_no'].fillna(method='ffill')
    df['task_config'] = df['task_config'].fillna(method='ffill')
    df['repeat'] = df['repeat'].fillna(method='ffill')
    # so that I cann differenatiate task config and direction
    df['config_type'] = df['task_config'] + '_' + df['type']
    
    # add a colum with nav_key presses that counted based on nav_key_task.rt and t_step_press_curr_run
    # and one with the actual keys they pressed based on nav_key_task.keys and t_step_press_curr_run
    # and one with keys presssed, but never executed.
    
    indices_with_nav_keys = df[df['nav_key_task.started'].notna()].index.to_list()
    
    for task_no, row_index in enumerate(indices_with_nav_keys):
        curr_list_of_keys = ast.literal_eval(df.at[row_index, 'nav_key_task.keys'])
        curr_key_times = ast.literal_eval(df.at[row_index, 'nav_key_task.rt'])
        count_error_keys = 0
        overall_error_counter = 0
                    
        if task_no == 0:
            # import pdb; pdb.set_trace()
            for i in range(0, indices_with_nav_keys[task_no]):
                # if the data stored a value smaller than t = 0, correct that
                if round(df.at[i, 't_step_press_curr_run'],3) < 0:
                    curr_key_times = np.insert(curr_key_times, 0, 0)
                    curr_list_of_keys = np.insert(curr_list_of_keys, 0, 0)
                    df.at[i, 't_step_press_curr_run'] = 0  
                # next, track which button was pressed. It is possible to press more buttons than
                # actually are executed (only the button that was pressed last is executed)
                # thus, check if the button press is aligned with the time the subject moved
                if round(df.at[i, 't_step_press_curr_run'],3) == round(curr_key_times[i + overall_error_counter],3):
                    count_error_keys = 0
                    df.at[i, 'curr_key'] = curr_list_of_keys[i]
                    df.at[i, 'curr_key_time'] = curr_key_times[i]
                else:
                    wrong_keys = [str(curr_list_of_keys[i + overall_error_counter])]
                    wrong_times = [str(round(curr_key_times[i + overall_error_counter],4))]
                    count_error_keys += 1
                    overall_error_counter += 1
                    
                    while round(df.at[i, 't_step_press_curr_run'],3) != round(curr_key_times[i + overall_error_counter],3) :
                        wrong_keys.append(str(curr_list_of_keys[i + overall_error_counter]))
                        wrong_times.append(str(round(curr_key_times[i + overall_error_counter], 4)))
                        count_error_keys += 1
                        overall_error_counter +=1
                    
                    # if these columns don't exist yet, there will be an error if I try to fill with
                    # several items. instead, first create with 0, then fill.
                    df.at[i, 'non-exe_key_time'] = 0
                    df.at[i, 'non-exe_key'] = 0
                    
                    # once back to a correct key, fill in the one that you missed previously
                    df.at[i, 'non-exe_key'] = wrong_keys
                    df.at[i, 'non-exe_key_time'] = wrong_times
                    
                    df.at[i, 'curr_key'] = curr_list_of_keys[i + overall_error_counter]
                    df.at[i, 'curr_key_time'] = curr_key_times[i + overall_error_counter]
                    df.at[i, 'non-exe_key_counter'] = count_error_keys
                    count_error_keys = 0        

        elif task_no > 0:               
            for i_list,i in enumerate(range(indices_with_nav_keys[task_no-1]+1, indices_with_nav_keys[task_no])): 
                # for some sad reason, there are some (rare) glitches in the behavioural tables.
                # one glitch is that the first time of t_step_press_curr_run is shorter than 0
                if round(df.at[indices_with_nav_keys[task_no-1]+1, 't_step_press_curr_run'],3) <= 0:
                    curr_key_times = np.insert(curr_key_times, 0, 0)
                    curr_list_of_keys = np.insert(curr_list_of_keys, 0, 0)
                    df.at[indices_with_nav_keys[task_no-1]+1, 't_step_press_curr_run'] = 0
                # another glitch is that the first time of t_step_press_curr_run is even later than the last recorded press of this task
                if round(df.at[indices_with_nav_keys[task_no-1]+1, 't_step_press_curr_run'],3) > round(df.at[indices_with_nav_keys[task_no]-1, 't_step_press_curr_run'],3):
                    df.at[indices_with_nav_keys[task_no-1]+1, 't_step_press_curr_run'] = curr_key_times[i_list]
                # another glitch is that there is a negative time somewhere in the middle of the task
                if round(df.at[i, 't_step_press_curr_run'],3) < 0:
                    df.at[i, 't_step_press_curr_run'] = curr_key_times[i_list]

                # then, test for what I am actually interested in:
                    # which of the key presses was the recorded one?
                if round(df.at[i, 't_step_press_curr_run'],3) == round(curr_key_times[i_list + overall_error_counter],3):
                    df.at[i, 'curr_key'] = curr_list_of_keys[i_list + overall_error_counter]
                    df.at[i, 'curr_key_time'] = curr_key_times[i_list + overall_error_counter]
                else:
                    wrong_keys = [str(curr_list_of_keys[i_list + overall_error_counter])]
                    wrong_times = [str(round(curr_key_times[i_list + overall_error_counter],4))]
                    count_error_keys += 1
                    overall_error_counter += 1
                    
                    while round(df.at[i, 't_step_press_curr_run'],3) != round(curr_key_times[i_list + overall_error_counter],3) :
                        wrong_keys.append(str(curr_list_of_keys[i_list + overall_error_counter]))
                        wrong_times.append(str(round(curr_key_times[i_list + overall_error_counter], 4)))
                        count_error_keys += 1
                        overall_error_counter +=1
                    
                    # if these columns don't exist yet, there will be an error if I try to fill with
                    # several items. instead, first create with 0, then fill.
                    df.at[i, 'non-exe_key_time'] = 0
                    df.at[i, 'non-exe_key'] = 0
                    
                    # once back to a correct key, fill in the one that you missed previously
                    df.at[i, 'non-exe_key'] = wrong_keys
                    df.at[i, 'non-exe_key_time'] = wrong_times
                    
                    df.at[i, 'curr_key'] = curr_list_of_keys[i_list + overall_error_counter]
                    df.at[i, 'curr_key_time'] = curr_key_times[i_list + overall_error_counter]
                    df.at[i, 'non-exe_key_counter'] = count_error_keys
                    count_error_keys = 0
        
        
    
    
    # add columns whith field numbers 
    for index, row in df.iterrows():
        # current locations
        df.at[index, 'curr_loc_y_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_loc_y'], is_y=True, is_x = False)
        df.at[index, 'curr_loc_x_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_loc_x'], is_x=True, is_y = False)
        df.at[index, 'curr_rew_y_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_rew_y'], is_y=True, is_x = False)
        df.at[index, 'curr_rew_x_coord'] = mc.analyse.analyse_MRI_behav.transform_coord(df.at[index,'curr_rew_x'], is_x=True, is_y = False)
        # and prepare the regressors: config type, state and reward/walking specific.
        if not pd.isna(row['state']):
            if not np.isnan(row['rew_loc_x']):
                df.at[index, 'time_bin_type'] =  df.at[index, 'config_type'] + '_' + df.at[index, 'state'] + '_reward'
            elif np.isnan(row['rew_loc_x']):
                df.at[index, 'time_bin_type'] = df.at[index, 'config_type'] + '_' + df.at[index, 'state'] + '_path'
    
     
    
    
    
    # create a dictionnary with all future regressors, to make sure the names are not messed up.
    time_bin_types = df['time_bin_type'].dropna().unique()
    regressors = {}
    for time_bin_type in time_bin_types:
        regressors[time_bin_type] = []
       
    configs = df['config_type'].dropna().unique()
    
    # initialise all dictionaries
    walked_path, timings, rew_list, rew_timing, rew_index, subpath_after_steps = {}, {}, {}, {}, {}, {}
    steps_subpath_alltasks, keys_executed, keys_not_exe, timings_not_exe = {}, {}, {}, {}
     # and all lists per dictionary
    for config in configs:
        walked_path[config], keys_executed[config], timings[config], rew_list[config] = [], [], [], []
        rew_timing[config], rew_index[config], subpath_after_steps[config], steps_subpath_alltasks[config] = [], [], [], []
        keys_not_exe[config], timings_not_exe[config] = [], []


    for index, row in df.iterrows():
        # import pdb; pdb.set_trace()   
        task_config = row['config_type']
        time_bin_type = row['time_bin_type']
        
        #iterate through the regression dictionary first
        for key in regressors.keys():
            # check if the key starts with the task_config value
            if key.startswith(task_config):
                if time_bin_type == key:
                    regressors[key].append(1)
                elif pd.notna(time_bin_type):
                    regressors[key].append(0) 
                    
        # in case a new task has just started
        if not np.isnan(row['next_task']): 
            # first check if this is the first task of several repeats.
            if (index == 0) or (row['config_type'] != df.at[index -1, 'config_type']):
                timings[task_config].append(row['next_task'])
            else: # if it isnt, then take the reward start time from last rew D as start field.
                timings[task_config].append(df.at[index -1, 't_step_press_global'])
            walked_path[task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
            keys_executed[task_config].append([row['curr_key']])
            
            # check in case a key had been pressed that wasn't executed
            if mc.analyse.analyse_MRI_behav.any_entry_in_row_notnan(row['non-exe_key']):
                keys_not_exe[task_config].append([row['non-exe_key']])
                timings_not_exe[task_config].append([row['non-exe_key_time']])
        
        # if this is just a normal walking field
        elif not np.isnan(row['t_step_press_global']): # always except if this is reward D 
            # if its reward D, then it will be covered by the first if: if not np.isnan(row['next_task']): 
            timings[task_config].append(df.at[index - 1, 't_step_press_global'])  # Extract value from index-1
            walked_path[task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
            keys_executed[task_config].append([row['curr_key']])
            if mc.analyse.analyse_MRI_behav.any_entry_in_row_notnan(row['non-exe_key']):
                keys_not_exe[task_config].append([row['non-exe_key']])
                timings_not_exe[task_config].append([row['non-exe_key_time']])
       
        
        # next check if its a reward field
        if not np.isnan(row['rew_loc_x']): # if this is a reward field.
            # check if this is either at reward D(thus complete) or ignore interrupted trials
            # ignore these as they are not complete.
            if (index+2 < len(df)) or (row['state'] == 'D'):
                rew_timing[task_config].append(row['t_reward_start'])
                rew_list[task_config].append([row['curr_rew_x_coord'], row['curr_rew_y_coord']])
                subpath_after_steps[task_config].append(int(index-row['repeat']))  
                if row['state'] == 'D':
                    rew_index[task_config].append(len(walked_path[task_config])) #bc step has not been added yet
                    # if this is the last run of a task
                    if (index+2 < len(df)):
                        # first check if there are more tasks coming after, otherwise error
                        if (row['config_type'] != df.at[index +1, 'config_type']):
                            walked_path[task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                            keys_executed[task_config].append([row['curr_key']])
                            timings[task_config].append(df.at[index -1, 't_reward_start'])
                            if mc.analyse.analyse_MRI_behav.any_entry_in_row_notnan(row['non-exe_key']):
                                keys_not_exe[task_config].append([row['non-exe_key']])
                                timings_not_exe[task_config].append([row['non-exe_key_time']])
                            
                    else:
                        # however also add these fields if this is the very last reward!
                        if row['repeat'] == 4:
                            walked_path[task_config].append([row['curr_loc_x_coord'], row['curr_loc_y_coord']])
                            keys_executed[task_config].append([row['curr_key']])
                            timings[task_config].append(df.at[index -1, 't_step_press_global'])
                            if mc.analyse.analyse_MRI_behav.any_entry_in_row_notnan(row['non-exe_key']):
                                keys_not_exe[task_config].append([row['non-exe_key']])
                                timings_not_exe[task_config].append([row['non-exe_key_time']])
                            
                else:
                    rew_index[task_config].append(len(walked_path[task_config])-1) 
            else:
                continue

    return configs, rew_list, rew_index, walked_path, steps_subpath_alltasks, subpath_after_steps, timings, regressors, keys_executed, keys_not_exe, timings_not_exe


def select_models_I_want(RDM_version):
    if RDM_version in ['01', '01-1']: # 01 doesnt work yet! 
        models_I_want = ['trial_type_similarity', 'execution_similarity', 'presentation_similarity']
    elif RDM_version in ['02', '02-A']: #modelling paths + rewards, creating all possible models 
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight', 'clocks']
    elif RDM_version in ['02-act', '02-act-1phas']:
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight', 'clocks', 'buttons', 'buttonsXphase', 'action-box', 'curr_subpath_buttons', 'one_future_subp_buttons', 'two_future_subp_buttons', 'three_future_subp_buttons']
    elif RDM_version in ['03', '03-im', '03-A', '03-l', '03-e']: # modelling only rewards, splitting clocks within the same function
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight_only-rew', 'clocks_only-rew']
    elif RDM_version in ['03-tasklag']:
        models_I_want = ['location', 'phase', 'state', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight_only-rew', 'clocks_only-rew', 'curr_rings_split_clock_sin', 'one_fut_rings_split_clock_sin', 'two_fut_rings_split_clock_sin', 'three_fut_rings_split_clock_sin', 'clocks_only-rew_sin']
    elif RDM_version in ['03-1', '03-2']:  # modelling only rewards, splitting clocks later in a different way - after the regression.
        #models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc', 'curr-and-future-rew-locs']
        # CHANGE THIS BACK IF I DONT WANT EASY MODELS ANYMORE!
        models_I_want = ['location', 'curr_rew', 'next_rew', 'second_next_rew', 'third_next_rew', 'state', 'clocks']
        #
    elif RDM_version in ['03-1-act']:
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc', 'curr-and-future-rew-locs','buttonsXphase_only-rew', 'action-box_only-rew', 'buttons', 'one_future_step2rew', 'two_future_step2rew', 'three_future_step2rew', 'curr-and-future-steps2rew']

    elif RDM_version in ['03-5', '03-5-A', '04-5', '04-5-A']:
        models_I_want = ['state', 'state_masked']
    elif RDM_version in ['03-3']:  # modelling only rewards, splitting clocks later in a different way - after the regression; ignoring reward A
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc']
    elif RDM_version in ['03-99']:  # using 03-1 - reward locations and future rew model; but EVs are scrambled.
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']
    elif RDM_version in ['03-999']:  # is debugging 2.0: using 03-1 - reward locations and future rew model; but the voxels are scrambled.
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc']
    elif RDM_version in ['04', '04-A']: # only paths. to see if the human brain represents also only those rings anchored at no-reward locations
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'midnight_no-rew', 'clocks_no-rew']
    elif RDM_version in ['05']: # only paths AND only rewards to later compare both models: based on 3.1 and 4
        models_I_want = ['location', 'phase', 'phase_state', 'state', 'task_prog', 'clocks_only-rew', 'midnight_only-rew', 'one_future_rew_loc' ,'two_future_rew_loc', 'three_future_rew_loc', 'curr-and-future-rew-locs', 'curr_rings_split_clock', 'one_fut_rings_split_clock', 'two_fut_rings_split_clock', 'three_fut_rings_split_clock', 'clocks_no-rew', 'state_masked']
    else:
        models_I_want = []
    return(models_I_want)

def determine_number_of_conditions(GLM_version, RDM_version = None):
    if GLM_version == '01':
        no_RDM_conditions = 20 # including all instruction periods
    elif GLM_version in ['02', '02-e', '02-l']:
        no_RDM_conditions = 80 # including all paths and rewards
    elif GLM_version in ['03', '04','03-99', '03-999', '03-9999', '03-l', '03-e']:
        no_RDM_conditions = 40 # only including rewards or only paths
    elif GLM_version == '03-3': #excluding reward A
        no_RDM_conditions = 30
    elif GLM_version in ['03-4', '04-4', '03-4-e', '03-4-l', '03-4-rep1', '03-4-rep2' , '03-4-rep3' , '03-4-rep4' ,'03-4-rep5' ]: # only including tasks without double reward locs: A,C,D  and only rewards
        no_RDM_conditions = 24    
    if GLM_version in ['03-4', '04-4'] and RDM_version in ['03-5-A', '02-A', '03-A', '04-A', '04-5-A']: # only TASK A,C,D, only rewards B-C-D
        no_RDM_conditions = 18
    return no_RDM_conditions



def move_files_to_subfolder(folder_path):
    # import pdb; pdb.set_trace()

    # List all files in the source folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Filter files that end with .nii.gz
    nii_gz_files = [f for f in files if f.endswith('.nii.gz')]

    if not nii_gz_files:
        print("No .nii.gz files found to move.")
    else:
        # Get today's date in the format YYYY-MM-DD
        today_date = datetime.today().strftime('%Y-%m-%d')
        subfolder_name = f"results_pre_{today_date}"
        
        # Create the sub-folder if it doesn't exist
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        # Move .nii.gz files to the target sub-folder
        for file in nii_gz_files:
            shutil.move(os.path.join(folder_path, file), subfolder_path)
            print(f"Moved {file} to {subfolder_path}/")
         
    
              


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
def create_EV(onset, duration, magnitude, name, folder, TR_at_sec, version = None, version_TR = None):
    if version not in ['03-rep1', '03-rep2', '03-rep3', '03-rep4', '03-rep5', f"01-TR{version_TR}"]:
        if len(onset) > len(duration):
            onset = onset[:len(duration)]
            magnitude = magnitude[:len(duration)]
        elif len(duration) > len(onset):
            duration = onset[:len(onset)]
            magnitude = magnitude[:len(onset)]
        regressor_matrix = np.ones((len(magnitude),3))
        regressor_matrix[:,0] = [(time - TR_at_sec) for time in onset]
    else:
        regressor_matrix = np.ones((1,3))
        regressor_matrix[:,0] = onset - TR_at_sec
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
        print(f"Careful! There are Nans in {array}.")
        # import pdb; pdb.set_trace()
        # try if this is sensible: delete the rows with the nans.
        array = array[0: (len(array)-1)]
        count = count + 1
    if count > 0:   
        print(f"deteleted {count} rows to avoid nans.")
    if array.shape[0] == 0:
        print(f"Careful! Array {array} is empty. Pausing script")
        count = -1
        import pdb; pdb.set_trace()
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

# set up RSA with multiple regressors.
def multiple_RDMs_RSA(list_of_regressor_RDMs, model_RDM_dictionary, data_RDM_file):
    
    arguments = [model_RDM_dictionary[model] for model in list_of_regressor_RDMs]
    concatenated_RDMs = rsatoolbox.rdm.concat(*arguments)
    concatenated_RDMs_model = rsatoolbox.model.ModelWeighted('concatenated_RDMs', concatenated_RDMs)
    
    # # CHANGE THIS BACK LATER
    # for d in data_RDM_file:
    #     test = mc.analyse.analyse_MRI_behav.evaluate_model(concatenated_RDMs_model, d)
    # import pdb; pdb.set_trace()
    
    result_multiple_RDMs_RSA = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(concatenated_RDMs_model, d) for d in tqdm(data_RDM_file, desc='running GLM for all searchlights in combo model'))
    
    return(result_multiple_RDMs_RSA)




def save_RSA_result_binary(result_file, data_RDM_file, file_path, file_name, mask, ref_image_for_affine_path):
    x, y, z = mask.shape
    ref_img = load_img(ref_image_for_affine_path)
    affine_matrix = ref_img.affine
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    bin_diff_result_brain = np.zeros([x*y*z])
    bin_diff_result_brain[list(data_RDM_file.rdm_descriptors['voxel_index'])] = [vox for vox in result_file]
    bin_diff_result_brain = bin_diff_result_brain.reshape([x,y,z])
    
    bin_diff_result_brain_nifti = nib.Nifti1Image(bin_diff_result_brain, affine=affine_matrix)
    bin_diff_result_brain_file = f"{file_path}/{file_name}_bin_diff.nii.gz"
    nib.save(bin_diff_result_brain_nifti, bin_diff_result_brain_file)

  
    
def save_data_RDM_as_nifti(data_RDM_file, file_path, file_name, ref_image_for_affine_path, centers_for_voxel_index, rdm_toolbox = False):
    # import pdb; pdb.set_trace() 
    ref_img = load_img(ref_image_for_affine_path)
    x, y, z = ref_img.shape
    affine_matrix = ref_img.affine
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    brain_4d = np.zeros([x,y,z,len(data_RDM_file[0])])

    for i in range(0,len(data_RDM_file[0])):
        curr_slice = np.zeros([x*y*z])
        if rdm_toolbox == False:
            curr_slice[list(centers_for_voxel_index)] = [vox[i] for vox in data_RDM_file]
        elif rdm_toolbox == True:
            curr_slice[list(data_RDM_file.rdm_descriptors['voxel_index'])] = [vox.dissimilarities[0][i] for vox in data_RDM_file]
        brain_4d[:,:,:,i] = curr_slice.reshape([x,y,z])
    
    brain_4d_nifti = nib.Nifti1Image(brain_4d, affine=affine_matrix)
    brain_4d_file = f"{file_path}/{file_name}"
    nib.save(brain_4d_nifti, brain_4d_file)
    
    np.save(f"{file_path}/data_RDM", data_RDM_file)
    
    
    
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
    t_result_brain_file = f"{file_path}/{file_name}_t_val.nii.gz"
    nib.save(t_result_brain_nifti, t_result_brain_file)
    
    b_result_brain = np.zeros([x*y*z])
    b_result_brain[list(data_RDM_file.rdm_descriptors['voxel_index'])] = [vox[1][number_regr] for vox in result_file]
    b_result_brain = b_result_brain.reshape([x,y,z])
    
    b_result_brain_nifti = nib.Nifti1Image(b_result_brain, affine=affine_matrix)
    b_result_brain_file = f"{file_path}/{file_name}_beta.nii.gz"
    nib.save(b_result_brain_nifti, b_result_brain_file)
    
    p_result_brain = np.zeros([x*y*z])
    p_result_brain[list(data_RDM_file.rdm_descriptors['voxel_index'])] = [1 - vox[2][number_regr] for vox in result_file]
    p_result_brain = p_result_brain.reshape([x,y,z])
    
    p_result_brain_nifti = nib.Nifti1Image(p_result_brain, affine=affine_matrix)
    p_result_brain_file = f"{file_path}/{file_name}_p_val.nii.gz"
    nib.save(p_result_brain_nifti, p_result_brain_file)


def evaluate_model(model, data):
    # import pdb; pdb.set_trace()

    X = sm.add_constant(model.rdm.transpose());
    # first, normalize the regressors (but not the intercept, bc std = 0 -> division by 0!)
    for i in range(1, X.shape[1]):
        X[:,i] = (X[:,i] - np.nanmean(X[:,i]))/ np.nanstd(X[:,i])
    
    # to check if a GLM is ill-conditioned
    # To check that you can check the “condition number” of the design matrix - 
    # the ration between the maximum singular value (similar to eigenvalue) and the minimum singular value.. 
    # If that ration is close to 1, you’re good. If it’s very large (e.g. >1000), it means the matrix is ill-conditioned - 
    # one of your regressors is close to being a linear combination of the other two.
    # check_GLM_regressors(X)
    # import pdb; pdb.set_trace()
    
    Y = data.dissimilarities.transpose();
    
    # to filter out potential nans in the model part
    nan_filter = np.isnan(X).any(axis=1)
    filtered_X = X[~nan_filter]
    filtered_Y = Y[~nan_filter]
    
    est = sm.OLS(filtered_Y, filtered_X).fit()
    # import pdb; pdb.set_trace()
    return est.tvalues[1:], est.params[1:], est.pvalues[1:]
    

def evaluate_binary_model(model, data, binary_val):
    model_mask_one = model.rdm <= binary_val # similar conditions are around 0
    model_mask_two = model.rdm >= binary_val # dissimilarity conditions are towards 2
    cond_one = np.nanmean(data.dissimilarities[0][model_mask_one])
    cond_two = np.nanmean(data.dissimilarities[0][model_mask_two])
    #if cond_one < cond_two:
    #    import pdb; pdb.set_trace()
    
    # where is white (dissimilar conds) bigger than black (similar conds)? 
    return cond_two-cond_one
    


def evaluate_surface_searchlights(model, data):
    import pdb; pdb.set_trace()
    
    X = sm.add_constant(model.transpose());
    Y = data.transpose();
    
    # to filter out potential nans in the model part
    nan_filter = np.isnan(X).any(axis=1)
    filtered_X = X[~nan_filter]
    filtered_Y = Y[~nan_filter]
    
    est = sm.OLS(filtered_Y, filtered_X).fit()
    # import pdb; pdb.set_trace()
    return est.tvalues[1:], est.params[1:], est.pvalues[1:]


def prepare_model_data(model_data, number_conditions, RDM_version):
    # import pdb; pdb.set_trace()
    model_data = model_data.transpose()
    if RDM_version in ['01', '01-1']:
        nCond = model_data.shape[0]
    else:
        nCond = model_data.shape[0]/2
    nVox = model_data.shape[1]
    
    sessions = np.concatenate((np.zeros(int(np.shape(model_data)[0]/2)), np.ones(int(np.shape(model_data)[0]/2))))
    des = {'subj': 1}
    if RDM_version in ['01', '01-1']:
        conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(nCond)])),(1)).transpose(),number_conditions)
    else: 
        conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(nCond)])), (1,2)).transpose(),number_conditions*2)

    obs_des = {'conds': conds, 'sessions': sessions}
    chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
    RSA_tb_model_data_object = rsd.Dataset(measurements=model_data,
                       descriptors=des,
                       obs_descriptors=obs_des,
                       channel_descriptors=chn_des)


                    # obs_des = {'events': events, 'sessions': cv_descr}
                    # ds = Dataset(data_2d[:, center_neighbors],
                    #              descriptors={'center': center},
                    #              obs_descriptors=obs_des,
                    #              channel_descriptors={'voxels': center_neighbors})


    # import pdb; pdb.set_trace()
    
    # DOUBLE CHECK WHY THE CONDITIONS HERE ARE 48 AND NOT 24!!!!
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



def similarity_of_tasks(reward_per_task_per_taskhalf_dict):
    # import pdb; pdb.set_trace() 
    
    # first, put the contents of the task-half dict into one.
    def flatten_nested_dict(nested_dict):
        flattened_dict = {}
        for key, value in nested_dict.items():
            if isinstance(value, dict):  # If the value is a dictionary, extend the flat dictionary with its items
                flattened_dict.update(value)
            else:
                flattened_dict[key] = value
        return flattened_dict
    
    rewards_experiment = flatten_nested_dict(reward_per_task_per_taskhalf_dict)
    
    all_rewards = []
    all_names = []
    #  make sure that the dictionary is alphabetically sorted.
    for task in sorted(rewards_experiment.keys()):
        all_rewards.append(rewards_experiment[task])
        all_names.append(task)
    
    
    # create 3 binary RDMs:
        # first, those that are backwards vs those that are forwards trials.
        # second, those that are executed in the same order.
        # third, those that are presented in the same order.
    
    
    # first, all those that are backwards and those that are forwards trials are equal.
    # direction_presentation = np.zeros((len(all_names), 2)) this will yield -1 and 1.
    trial_type_similarity = np.zeros((len(all_rewards), len(all_rewards)*4)) # this is -0.012658227848101285 and 1
    for i, task_name in enumerate(all_names):
        if task_name.endswith('forw'):
            trial_type_similarity[i, 0] = 1
        elif task_name.endswith('backw'):
            trial_type_similarity[i, 1] = 1
    
    # second, all those that are executed in the same order are the same.
    execution_similarity = np.zeros((len(all_rewards), len(all_rewards)*4)) # this is -0.012658227848101285 and 1
    for i in range(len(all_rewards)):
        for j in range(len(all_rewards)):
            if all_rewards[i] == all_rewards[j]:
                execution_similarity[i, j] = 1
            
            
    # third, all those that are presented in the same order are the same.
    # careful, this also changes all_rewards!! do this last.
    presented_rewards = all_rewards.copy()
    for i, task_name in enumerate(all_names):
        if task_name.endswith('backw'):
            presented_rewards[i].reverse()

    
    presentation_similarity = np.zeros((len(all_rewards), len(all_rewards)*4)) # this is -0.012658227848101285 and 1
    for i in range(len(presented_rewards)):
        for j in range(len(presented_rewards)):
            if presented_rewards[i] == presented_rewards[j]:
                presentation_similarity[i, j] = 1
    
    
    # import pdb; pdb.set_trace() 
    # np.corrcoef(presentation_similarity[:, :10])
    # corrected_model = (presentation_similarity[:, :10] + np.transpose(presentation_similarity[:, :10]))/2
    #corrected_RSM_dict[model] = corrected_model[0:int(len(corrected_model)/2), int(len(corrected_model)/2):]
   
        
    # to create the right format, split this into two task halves again
    # import pdb; pdb.set_trace() 
    models_between_tasks = {'1': {key: "" for key in ['execution_similarity', 'presentation_similarity', 'trial_type_similarity']},
                            '2': {key: "" for key in ['execution_similarity', 'presentation_similarity', 'trial_type_similarity']}}
    
    
   # x = {'execution_similarity': {key: "" for key in ['1', '2']},
   #      'presentation_similarity': {key: "" for key in ['1', '2']},
   #      'trial_type_similarity': {key: "" for key in ['1', '2']}}
    
    models_between_tasks['1']['execution_similarity'] = execution_similarity[:10].T
    models_between_tasks['2']['execution_similarity'] = execution_similarity[10:20].T
    
    models_between_tasks['1']['presentation_similarity'] = presentation_similarity[:10].T
    models_between_tasks['2']['presentation_similarity'] = presentation_similarity[10:20].T
    
    models_between_tasks['1']['trial_type_similarity'] = trial_type_similarity[:10].T
    models_between_tasks['2']['trial_type_similarity'] = trial_type_similarity[10:20].T

    # import pdb; pdb.set_trace()   
    # CONTINUE HERE!!! THE  PRESENT SIM ISNT QUITE RIGHT YET!
    return models_between_tasks

    

def plot_trajectories(data):
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, len(data) * 2))  # Adjusted figure size for better fit

    # Custom colors for each value in the 3x3 grid
    colors = {
        0: '#003366',  # Dark blue
        1: '#005577',  # Less dark blue
        2: '#007799',  # Light blue
        3: '#006666',  # Dark turquoise
        4: '#008888',  # Less dark turquoise
        5: '#00AAAA',  # Light turquoise
        6: '#005555',  # Very dark turquoise
        7: '#007777',  # Darker turquoise
        8: '#009999',  # Bright turquoise
    }

    for half_id, configs in data.items():
        for config_id, runs in configs.items():
            # Set up the figure and axis for each task-half and config
            fig, ax = plt.subplots(figsize=(10, len(runs) * 2))
            plt.title(f'{config_id}, Task Half {half_id}', fontsize=16)

            for i, (run_id, values) in enumerate(runs.items()):
                y = len(runs) - i - 1  # Calculate the y-position for the current run
                
                # Plot each number as a circle with the respective color
                for x, value in enumerate(values):
                    ax.add_patch(plt.Circle((x, y), 0.4, color=colors[value]))
                    ax.text(x, y, str(value), color='white', ha='center', va='center', fontsize=12)

            ax.set_xlim(-1, len(values))
            ax.set_ylim(-1, len(runs))
            ax.set_aspect('equal')
            ax.axis('off')  # Turn off the axis
            plt.show()




