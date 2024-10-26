#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:48:43 2024

Functions for Ripples.
Human LFPs.


@author: xpsy1114
"""

import numpy as np
from collections import defaultdict
import neo
import pandas as pd



def reference_electrodes(LFP_data, channels, rep):
    wires = defaultdict(list)
    for idx, string in enumerate(channels):
        wire_name = string[:6]  # Extract the first 6 characters as the prefix
        wires[wire_name].append(string[6:])  # Store the index

    referenced_data_dict = {}
    for wire_no, (channel_n, chan_idx) in enumerate(wires.items()):
        if len(chan_idx) < 2:
            continue  # If there are less than two entries, skip
        for idx in range(1, len(wires[channel_n])):
            array1 = LFP_data[:, wire_no*len(wires[channel_n]) + idx-1]
            array2 = LFP_data[:, wire_no*len(wires[channel_n]) + idx]
            # print(f"does index fit? index is {wire_no*len(wires[channel_n]) + idx-1} and channel is {channel_n} and {chan_idx[idx]}")
            diff = array1 - array2
            referenced_data_dict[f"{channel_n + chan_idx[idx-1]} vs {channel_n + chan_idx[idx]}"] = diff

    new_channel_names = list(referenced_data_dict.keys())

    # Step 2: Extract the values and convert to a NumPy array
    # Assuming all values are lists or arrays of equal length
    referenced_data = np.array(list(referenced_data_dict.values())).T
    return referenced_data, new_channel_names



def prep_behaviour(behaviour_all):
    # define seconds_lower[task] as a new repeat of a grid.
    # also collect grid_index (task_config) to keep track if you're still in the same grid.
    seconds_lower, seconds_upper, task_config, task_index, task_onset, new_grid_onset, found_first_D  = [], [], [], [], [], [], []
    for i in range(1, len(behaviour_all)):
        if i == 1: 
            new_grid_onset.append(behaviour_all[i-1, 10])
            seconds_lower.append(behaviour_all[i-1, 10])
            task_config.append([behaviour_all[i-1, 5], behaviour_all[i-1, 6],behaviour_all[i-1, 7],behaviour_all[i-1, 8]])
            task_index.append(int(behaviour_all[i-1,-1]))
            task_onset.append(behaviour_all[i-1, 10])
            found_first_D.append(behaviour_all[i-1, 4])
        curr_repeat = behaviour_all[i, 0]
        last_repeat = behaviour_all[i-1, 0]
        if curr_repeat != last_repeat:
            seconds_lower.append(behaviour_all[i, 10])
            seconds_upper.append(behaviour_all[i-1, 4])
            task_config.append([behaviour_all[i, 5], behaviour_all[i, 6],behaviour_all[i, 7],behaviour_all[i, 8]])
            task_index.append(int(behaviour_all[i,-1]))
            task_onset.append(behaviour_all[i, 10])
        if behaviour_all[i, 9] < behaviour_all[i-1, 9]: # 9 is repeats in current grid
            # i.e. if in a new grid
            new_grid_onset.append(behaviour_all[i, 10])
            found_first_D.append(behaviour_all[i, 4])
    seconds_upper.append(behaviour_all[i, 4])       
    return seconds_lower, seconds_upper, task_config, task_index, task_onset, new_grid_onset, found_first_D



    

def load_LFPs(LFP_dir, sub, names_blks_short):
    # instead of fully loading the files, I am only loading the reader and then 
    # looking at them in lazy-mode, only calling the shorter segments.
        reader, block_size, channel_list, orig_sampling_freq, raw_file_lazy = [], [], [], [], []
        if sub not in ['s5']: # doesn't have half 2 
            for file_half in [0,1]:
                reader.append(neo.io.BlackrockIO(filename=f"{LFP_dir}/{sub}/{names_blks_short[file_half]}", nsx_to_load=3))
                if sub in ['s11']:
                    block_size.append(reader[file_half].get_signal_size(seg_index=0, block_index=0))
                else:
                    block_size.append(reader[file_half].get_signal_size(seg_index=1, block_index=0))
                orig_sampling_freq.append(int(reader[file_half].sig_sampling_rates[3]))        
                # all of these will only be based on the second file. Should be equivalent!
                channel_names = reader[file_half].header['signal_channels']
                channel_names = [str(elem) for elem in channel_names[:]]
                channel_list = [name.split(",")[0].strip("('") for name in channel_names]
                HC_indices = []
                mPFC_indices = []
                for i, channel in enumerate(channel_list):
                    if 'Ha' in channel or 'Hb' in channel:
                        HC_indices.append(i)
                    if 'Ca' in channel:
                        mPFC_indices.append(i)    
                HC_channels = [channel_list[i] for i in HC_indices]
                mPFC_channels = [channel_list[i] for i in mPFC_indices]
                if sub in ['s11']:
                    raw_file_lazy.append(reader[file_half].read_segment(seg_index=0, lazy=True))
                else:
                    raw_file_lazy.append(reader[file_half].read_segment(seg_index=1, lazy=True))
        elif sub in ['s5']:
            for file_half in [0]:
                reader.append(neo.io.BlackrockIO(filename=f"{LFP_dir}/{sub}/{names_blks_short[file_half]}", nsx_to_load=3))
                if sub in ['s11']:
                    block_size.append(reader[file_half].get_signal_size(seg_index=0, block_index=0))
                else:
                    block_size.append(reader[file_half].get_signal_size(seg_index=1, block_index=0))
                orig_sampling_freq.append(int(reader[file_half].sig_sampling_rates[3]))        
                # all of these will only be based on the second file. Should be equivalent!
                channel_names = reader[file_half].header['signal_channels']
                channel_names = [str(elem) for elem in channel_names[:]]
                channel_list = [name.split(",")[0].strip("('") for name in channel_names]
                HC_indices = []
                mPFC_indices = []
                for i, channel in enumerate(channel_list):
                    if 'Ha' in channel or 'Hb' in channel:
                        HC_indices.append(i)
                    if 'Ca' in channel:
                        mPFC_indices.append(i)    
                HC_channels = [channel_list[i] for i in HC_indices]
                mPFC_channels = [channel_list[i] for i in mPFC_indices]
                if sub in ['s11']:
                    raw_file_lazy.append(reader[file_half].read_segment(seg_index=0, lazy=True))
                else:
                    raw_file_lazy.append(reader[file_half].read_segment(seg_index=1, lazy=True))
                    
        return raw_file_lazy, HC_channels, HC_indices, mPFC_channels, mPFC_indices, orig_sampling_freq, block_size
    


def dict_unnesting_three_levels(three_level_dict):
    plotting_dict = {}
    for session in sorted(three_level_dict.keys()):
        # import pdb; pdb.set_trace() 
        for task in sorted(three_level_dict[session].keys()):
            if task == 1:
                max_idx = len(three_level_dict[session][task])
            for boxplot_idx in sorted(three_level_dict[session][task].keys()):
                if boxplot_idx > max_idx:
                    max_idx = boxplot_idx
                if boxplot_idx not in plotting_dict:
                    plotting_dict[boxplot_idx] = []
                plotting_dict[boxplot_idx].append(three_level_dict[session][task][boxplot_idx])
    
    these_bins_exist = np.arange(1, max_idx+1, 1)
    for bin_idx in these_bins_exist:
        if bin_idx not in plotting_dict:
            plotting_dict[bin_idx] = [np.nan]
    # import pdb; pdb.set_trace() 
    return plotting_dict


def collapse_first_four_vs_rest(dict_per_state_repeat, group_labels, n_first_group, n_second_group):
    dict_two_conds = {}
    # import pdb; pdb.set_trace()
    no_groups_group_one = 0
    no_groups_group_two = 0
    for key in sorted(dict_per_state_repeat.keys()):
        if int(key) < n_first_group+1:
            if group_labels[0] not in dict_two_conds:
                dict_two_conds[group_labels[0]] = []
            dict_two_conds[group_labels[0]].append(dict_per_state_repeat[key])
            no_groups_group_one = no_groups_group_one + 1
        elif int(key) > n_first_group:
            if group_labels[1] not in dict_two_conds:
                dict_two_conds[group_labels[1]] = dict_per_state_repeat[key]
            else:
                dict_two_conds[group_labels[1]].append(dict_per_state_repeat[key]) 
            no_groups_group_two = no_groups_group_two + 1
            
    # import pdb; pdb.set_trace()
    
    for group in dict_two_conds:
        flattened_data = []
        for item in dict_two_conds[group]:
            if isinstance(item, list):
                flattened_data.extend(item)
            else:
                flattened_data.append(item)
        dict_two_conds[group] = flattened_data
    
    return dict_two_conds, no_groups_group_one, no_groups_group_two



def compute_velocity(button_info, behaviour_info):
    button_df = pd.DataFrame(button_info)
    behaviour_df = pd.DataFrame(behaviour_info)
    velocities_df = pd.DataFrame()
    no_grids = int(np.max(button_info[:,2]))
    for grid in range(1,no_grids+1):
        behaviour_curr_task = behaviour_df[behaviour_df[12]==grid]
        buttons_curr_task = button_df[(button_df[2]==grid) & (button_df[0] != 99)] 
        for repeat in range(1, len(behaviour_curr_task)+1):
            for state in range(0,4):
                #compute velocity per state transition.
                # velocity = duration/ amount of button presses
                if state == 1:
                    lower = behaviour_curr_task.iloc[repeat][10]
                else:
                    lower = behaviour_curr_task.iloc[repeat][state-1]
                upper = behaviour_curr_task.iloc[repeat][state]
                duration = upper-lower 
                # how many buttons between upper and lower?
                subset_presses = buttons_curr_task[(buttons_curr_task[1]>lower) & (buttons_curr_task[1]<upper)]
                
                # FIGURE OUT THE INDEXING HERE!!! IT HAS TO BE THE SAME AS THE BEHAVIOUR TABLE
                if state == 1 :
                    velocities_df['speed_to_A'][1] = duration/len(subset_presses)
                    velocities_df['duration_to_A'][1] = duration
                    velocities_df['count_movebuttons_A'] = len(subset_presses)
                elif state == 2:
                    velocities_df['speed_to_B'][1] = duration/len(subset_presses)
                    velocities_df['duration_to_B'][1] = duration
                    velocities_df['count_movebuttons_B'] = len(subset_presses)
                elif state == 3:
                    velocities_df['speed_to_C'][1] = duration/len(subset_presses)
                    velocities_df['duration_to_C'][1] = duration
                    velocities_df['count_movebuttons_C'] = len(subset_presses)
                elif state == 4:
                    velocities_df['speed_to_D'][1] = duration/len(subset_presses)
                    velocities_df['duration_to_D'][1] = duration
                    velocities_df['count_movebuttons_D'] = len(subset_presses)
                velocities_df['grid_no'][1] = grid
                velocities_df['repeat'][1] = repeat

        import pdb; pdb.set_trace()
    # CONTINUE HERE!!!
    
    # 
    #
    #
    
    return velocities_df




def normalise_task(ripple_info_dict, task_index):
    # import pdb; pdb.set_trace()

    normalised_ripples = []
    
    sections = [((ripple_info_dict[f"{task_index}_start_task"]), (ripple_info_dict[f"{task_index}_found_all_locs"])),
                ((ripple_info_dict[f"{task_index}_found_all_locs"]), (ripple_info_dict[f"{task_index}_first_corr_solve"])),
                ((ripple_info_dict[f"{task_index}_first_corr_solve"]), (ripple_info_dict[f"{task_index}_end_task"]))]

    all_events_task = ripple_info_dict[f"{task_index}_ripples_across_choi"]
    
    # Define normalized section ranges (e.g., each section is a third of the normalized scale)
    normalized_sections = [(0, 0.33), (0.33, 0.67), (0.67, 1.0)]

    
    for section_idx, (original_start, original_end) in enumerate(sections):
        normalized_start, normalized_end = normalized_sections[section_idx]
        
        # Find events in the current section
        section_events = [event for event in all_events_task if original_start <= event < original_end]
        
        # Normalize each event in this section
        for event in section_events:
            # Normalize to [0, 1] scale within the section
            normalized_event = ((event - original_start) / (original_end - original_start)) * (normalized_end - normalized_start) + normalized_start
            normalised_ripples.append(normalized_event)
    
    durations_per_section = [sections[0][1]-sections[0][0], sections[1][1]-sections[1][0], sections[2][1]-sections[2][0]]
    
    return normalised_ripples, durations_per_section


def sort_in_three_sections(ripple_info_dict, task_index):
    # import pdb; pdb.set_trace()
    sections = [((ripple_info_dict[f"{task_index}_start_task"]), (ripple_info_dict[f"{task_index}_found_all_locs"])),
                ((ripple_info_dict[f"{task_index}_found_all_locs"]), (ripple_info_dict[f"{task_index}_first_corr_solve"])),
                ((ripple_info_dict[f"{task_index}_first_corr_solve"]), (ripple_info_dict[f"{task_index}_end_task"]))]

    section_labels = ['find_ABCD', 'first_solve_correctly', 'all_repeats']
    all_events_task = ripple_info_dict[f"{task_index}_ripples_across_choi"]
    
    # # Define normalized section ranges (e.g., each section is a third of the normalized scale)
    # normalized_sections = [(0, 0.33), (0.33, 0.67), (0.67, 1.0)]
    ripples_in_sections = {}
    for section_idx, (original_start, original_end) in enumerate(sections):
        # Find events in the current section
        ripples_in_sections[section_labels[section_idx]] = [event for event in all_events_task if original_start <= event < original_end]
    durations_per_section = [sections[0][1]-sections[0][0], sections[1][1]-sections[1][0], sections[2][1]-sections[2][0]]

    return ripples_in_sections, durations_per_section

    
    
def sort_feedback_in_three_sections(ripple_info_dict, feedback_session_sub_dict, task_index):
    # import pdb; pdb.set_trace()
    sections = [((ripple_info_dict[f"{task_index}_start_task"]), (ripple_info_dict[f"{task_index}_found_all_locs"])),
                ((ripple_info_dict[f"{task_index}_found_all_locs"]), (ripple_info_dict[f"{task_index}_first_corr_solve"])),
                ((ripple_info_dict[f"{task_index}_first_corr_solve"]), (ripple_info_dict[f"{task_index}_end_task"]))]

    section_labels = ['find_ABCD', 'first_solve_correctly', 'all_repeats']
    all_pos_task = feedback_session_sub_dict[f"{task_index}_correct"]
    all_neg_task = feedback_session_sub_dict[f"{task_index}_error"]
    
    # # Define normalized section ranges (e.g., each section is a third of the normalized scale)
    # normalized_sections = [(0, 0.33), (0.33, 0.67), (0.67, 1.0)]
    feedback_in_sections = {}
    for section_idx, (original_start, original_end) in enumerate(sections):
        # Find events in the current section
        feedback_in_sections[f"pos_{section_labels[section_idx]}"] = [event for event in all_pos_task if original_start <= event < original_end]
        feedback_in_sections[f"neg_{section_labels[section_idx]}"] = [event for event in all_neg_task if original_start <= event < original_end]
    
    return feedback_in_sections
    

def sort_ripples_by_feedback(feedback_per_section_task, ripples_per_section_task):
    # import pdb; pdb.set_trace()
    ripples_sorted_by_feedback_and_section_all_tasks = {}
    for task in sorted(feedback_per_section_task.keys()):
        ripples_sorted_by_feedback_and_section = {}
        for section_feedback_type in sorted(feedback_per_section_task[task].keys()):
            for feedback_time in feedback_per_section_task[task][section_feedback_type]:
                if section_feedback_type.endswith('find_ABCD'):
                    if section_feedback_type not in ripples_sorted_by_feedback_and_section:
                        ripples_sorted_by_feedback_and_section[section_feedback_type] = []
                    ripples_sorted_by_feedback_and_section[section_feedback_type].append([event for event in ripples_per_section_task[task]['find_ABCD'] if feedback_time <= event < feedback_time+1])
                if section_feedback_type.endswith('first_solve_correctly'):
                    if section_feedback_type not in ripples_sorted_by_feedback_and_section:
                        ripples_sorted_by_feedback_and_section[section_feedback_type] = []
                    ripples_sorted_by_feedback_and_section[section_feedback_type].append([event for event in ripples_per_section_task[task]['first_solve_correctly'] if feedback_time <= event < feedback_time+1])
                if section_feedback_type.endswith('all_repeats'):
                    if section_feedback_type not in ripples_sorted_by_feedback_and_section:
                        ripples_sorted_by_feedback_and_section[section_feedback_type] = []
                    ripples_sorted_by_feedback_and_section[section_feedback_type].append([event for event in ripples_per_section_task[task]['all_repeats'] if feedback_time <= event < feedback_time+1])
            if section_feedback_type in ripples_sorted_by_feedback_and_section and len(ripples_sorted_by_feedback_and_section[section_feedback_type]) > 1:
                ripples_sorted_by_feedback_and_section[section_feedback_type] = [item for sublist in ripples_sorted_by_feedback_and_section[section_feedback_type] for item in sublist]
        
        ripples_sorted_by_feedback_and_section_all_tasks[task] = ripples_sorted_by_feedback_and_section.copy()
    # by task,by section, categorise as ripple pos if within 1.5 secs after pos feedback,
    # and as ripple neg if within 1.5 secs after neg feedback.
    
    return ripples_sorted_by_feedback_and_section_all_tasks
    
    







    
    