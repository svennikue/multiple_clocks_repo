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
from scipy.stats import norm
import mc
import os
import re


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


def load_spiking_data(sub):
    # Path to the folder containing your CSV files
    LFP_path = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/{sub}"
    
    # Initialize the dictionary
    spike_data = {}
    
    # Regular expression to match the desired file pattern
    pattern = r"spikeTimes_(.+)_channel_no_(\d+)\.csv"
    
    # Iterate through files in the folder
    for file_name in os.listdir(LFP_path):
        # Check if the file name matches the pattern
        match = re.match(pattern, file_name)
        if match:
            # Extract electrode_name and number from the file name
            electrode_name = match.group(1)
            channel_number = match.group(2)
            key = f"{electrode_name}-{channel_number}"
            
            # Load the CSV file as a numpy array
            file_path = os.path.join(LFP_path, file_name)
            spike_data[key] = np.loadtxt(file_path, delimiter=",")
            
    return spike_data
  
    
  
    
  

def load_LFPs(LFP_dir, sub, names_blks_short, channel_list_complete = False):
    # instead of fully loading the files, I am only loading the reader and then 
    # looking at them in lazy-mode, only calling the shorter segments.
        reader, block_size, channel_list, orig_sampling_freq, raw_file_lazy = [], [], [], [], []

        file_halve_list = [0,1]
        if sub in ['s5', 's26']:
            file_halve_list = [0]
        if sub in ['s18']:
            file_halve_list = [0,1,2]
        
        
        # if sub not in ['s5', 's26']: # doesn't have half 2 
        for file_half in file_halve_list:
            reader.append(neo.io.BlackrockIO(filename=f"{LFP_dir}/{sub}/{names_blks_short[file_half]}", nsx_to_load=3))
            if (sub in ['s11'] and file_half == 0) or (sub in ['s18'] and file_half in [1, 2]):
                block_size.append(reader[file_half].get_signal_size(seg_index=0, block_index=0))
            else:
                block_size.append(reader[file_half].get_signal_size(seg_index=1, block_index=0))
                
                
            # reader_test = neo.rawio.BlackrockRawIO(filename=f"{LFP_dir}/{sub}/{names_blks_short[file_half]}", nsx_to_load=3)
            # reader_test.parse_header()
            # delete this again
                
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
        
            if channel_list_complete == True:
                ROI_dict = {}
                ROI_list, ROI_indices = [], []
                for i, channel in enumerate(channel_list):
                    if "empty" in channel:  # Skip empty labels
                        continue
                    ROI_list.append(channel)
                    ROI_indices.append(i)
                    ROI = mc.analyse.ripple_helpers.extract_ROIs_from_channellist(channel)
                    if ROI not in ROI_dict:
                        ROI_dict[ROI] = []
                    ROI_dict[ROI].append(channel)
                    
                if sub in ['s11', 's18']:
                    raw_file_lazy.append(reader[file_half].read_segment(seg_index=0, lazy=True))
                else:
                    raw_file_lazy.append(reader[file_half].read_segment(seg_index=1, lazy=True))
                    
        # elif sub in ['s5', 's26']:
        #     for file_half in [0]:
        #         reader.append(neo.io.BlackrockIO(filename=f"{LFP_dir}/{sub}/{names_blks_short[file_half]}", nsx_to_load=3))
        #         block_size.append(reader[file_half].get_signal_size(seg_index=1, block_index=0))
        #         orig_sampling_freq.append(int(reader[file_half].sig_sampling_rates[3]))        
        #         # all of these will only be based on the second file. Should be equivalent!
        #         channel_names = reader[file_half].header['signal_channels']
        #         channel_names = [str(elem) for elem in channel_names[:]]
        #         channel_list = [name.split(",")[0].strip("('") for name in channel_names]
        #         HC_indices = []
        #         mPFC_indices = []
        #         for i, channel in enumerate(channel_list):
        #             if 'Ha' in channel or 'Hb' in channel:
        #                 HC_indices.append(i)
        #             if 'Ca' in channel:
        #                 mPFC_indices.append(i)    
        #         HC_channels = [channel_list[i] for i in HC_indices]
        #         mPFC_channels = [channel_list[i] for i in mPFC_indices]
        #         if channel_list_complete == True:
        #             ROI_dict = {}
        #             ROI_list, ROI_indices = [], []
        #             for i, channel in enumerate(channel_list):
        #                 if "empty" in channel:  # Skip empty labels
        #                     continue
        #                 ROI_list.append(channel)
        #                 ROI_indices.append(i)
        #                 ROI = mc.analyse.ripple_helpers.extract_ROIs_from_channellist(channel)
        #                 if ROI not in ROI_dict:
        #                     ROI_dict[ROI] = []
        #                 ROI_dict[ROI].append(channel)
                        
        #         raw_file_lazy.append(reader[file_half].read_segment(seg_index=1, lazy=True))
        
        if channel_list_complete == False:
            ROI_dict, ROI_list, ROI_indices = [], [], []
                   
        return raw_file_lazy, HC_channels, HC_indices, mPFC_channels, mPFC_indices, orig_sampling_freq, block_size, ROI_dict, ROI_list, ROI_indices
    


def extract_ROIs_from_channellist(channel):
    # Default values
    hemisphere = "Left" if channel.startswith("L") else "Right" if channel.startswith("R") else "Unknown Hemisphere"
    lobe = None
    region = None
    
    # Determine the lobe based on the second character
    if "F" in channel[:3]:
        lobe = "Frontal"
    elif "T" in channel[:3]:
        lobe = "Temporal"
    elif "P" in channel[:3]:
        lobe = "Parietal"
    elif "O" in channel[:3]:
        lobe = "Occipital"
    elif "C" in channel[:3]:
        lobe = "Cerebellum"
    
    # Determine the brain region based on suffix
    if "Ha" in channel:
        region = "Hippocampus"
    elif "Ca" in channel:
        region = "Cingulate"
    elif "E" in channel:
        region = "Entorhinal Cortex"
    elif "Ib" in channel:
        region = "Insula"
    elif "Cb" in channel:
        region = "Cerebellum"
    elif "Hc" in channel:
        region = "Hypothalamus"
    elif "A" in channel:
        region = "Amygdala"
    else:
        region = f"{lobe or 'Unknown Lobe'} Region"  # Fallback to lobe if region not clear

    # Combine hemisphere, lobe, and region for the key
    ROI = f"{hemisphere} {lobe or 'Unknown Lobe'} {region}"
    return ROI
    

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
        section_events = [event for event in all_events_task if original_start < event <= original_end]
        
        # Normalize each event in this section
        for event in section_events:
            # Normalize to [0, 1] scale within the section
            normalized_event = ((event - original_start) / (original_end - original_start)) * (normalized_end - normalized_start) + normalized_start
            normalised_ripples.append(normalized_event)
    
    durations_per_section = [sections[0][1]-sections[0][0], sections[1][1]-sections[1][0], sections[2][1]-sections[2][0]]
    
    return normalised_ripples, durations_per_section





def normalise_feedback(ripple_info_dict, fb_dict, task_index):
    
    
    normalised_fb_pos, normalised_fb_neg = [], []
    
    sections = [((ripple_info_dict[f"{task_index}_start_task"]), (ripple_info_dict[f"{task_index}_found_all_locs"])),
                ((ripple_info_dict[f"{task_index}_found_all_locs"]), (ripple_info_dict[f"{task_index}_first_corr_solve"])),
                ((ripple_info_dict[f"{task_index}_first_corr_solve"]), (ripple_info_dict[f"{task_index}_end_task"]))]

    # import pdb; pdb.set_trace()
    
    # CONTINUE HERE!!
    # I NEED TO SPLIT BY SECTION IN THE DCITIONARY 
    
    
    
    pos_fb = fb_dict[f"{task_index}_correct"]
    neg_fb = fb_dict[f"{task_index}_error"]
    
    # Define normalized section ranges (e.g., each section is a third of the normalized scale)
    normalized_sections = [(0, 0.33), (0.33, 0.67), (0.67, 1.0)]

    for section_idx, (original_start, original_end) in enumerate(sections):
        normalized_start, normalized_end = normalized_sections[section_idx]
        
        # Find events in the current section
        section_pos = [event for event in pos_fb if original_start < event <= original_end]
        section_neg = [event for event in neg_fb if original_start < event <= original_end]
        
        # Normalize each event in this section
        for event in section_pos:
            # Normalize to [0, 1] scale within the section
            normalized_event = ((event - original_start) / (original_end - original_start)) * (normalized_end - normalized_start) + normalized_start
            normalised_fb_pos.append(normalized_event)
        
        for event in section_neg:
            # Normalize to [0, 1] scale within the section
            normalized_event = ((event - original_start) / (original_end - original_start)) * (normalized_end - normalized_start) + normalized_start
            normalised_fb_neg.append(normalized_event)
               
    
    return normalised_fb_pos, normalised_fb_neg


def normalise_explore_by_pos_fb(start, end, explore_ripples, explore_fb, task_i, until_fb = False):
    
    
    sections = [((start, explore_fb[f"{task_i}_pos"][0])),
                ((explore_fb[f"{task_i}_pos"][0], explore_fb[f"{task_i}_pos"][1])),
                ((explore_fb[f"{task_i}_pos"][1], explore_fb[f"{task_i}_pos"][2])),
                ((explore_fb[f"{task_i}_pos"][2]), end)]
    
    normalised_ripples, normalised_fb_pos, normalised_fb_neg = [], [], []
    

    normalised_sections = [(0, 1/4), (1/4, 2/4), (2/4, 3/4), (3/4, 1)]
    for section_idx, (original_start, original_end) in enumerate(sections):
        normalised_start, normalised_end = normalised_sections[section_idx]
        
        ripples = [ripple for ripple in explore_ripples[task_i] if original_start < ripple <= original_end]
        feedbacks_pos = [fb for fb in explore_fb[f"{task_i}_pos"] if original_start < fb <= original_end]
        feedbacks_neg = [fb for fb in explore_fb[f"{task_i}_neg"] if original_start < fb <= original_end]
        
        for rip in ripples:
            normalised_rip = ((rip-original_start) / (original_end - original_start)) * (normalised_end - normalised_start) + normalised_start
            normalised_ripples.append(normalised_rip)
        for fbi in feedbacks_pos:
            normalised_fbi_pos = ((fbi - original_start) / (original_end - original_start)) * (normalised_end - normalised_start) + normalised_start
            normalised_fb_pos.append(normalised_fbi_pos)
    
        for fbi_neg in feedbacks_neg:
            normalised_fbi_neg = ((fbi_neg - original_start) / (original_end - original_start)) * (normalised_end - normalised_start) + normalised_start
            normalised_fb_neg.append(normalised_fbi_neg)
    
    
    if until_fb != False:
        import pdb; pdb.set_trace()

    
    return normalised_ripples, normalised_fb_pos, normalised_fb_neg




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
        ripples_in_sections[section_labels[section_idx]] = [event for event in all_events_task if original_start < event <= original_end]
    durations_per_section = [sections[0][1]-sections[0][0], sections[1][1]-sections[1][0], sections[2][1]-sections[2][0]]

    return ripples_in_sections, durations_per_section

def sort_HFB_event_in_three_sections(ripple_info_dict, event_dict, task_index, region_list):
    # import pdb; pdb.set_trace()
    sections = [((ripple_info_dict[f"{task_index}_start_task"]), (ripple_info_dict[f"{task_index}_found_all_locs"])),
                ((ripple_info_dict[f"{task_index}_found_all_locs"]), (ripple_info_dict[f"{task_index}_first_corr_solve"])),
                ((ripple_info_dict[f"{task_index}_first_corr_solve"]), (ripple_info_dict[f"{task_index}_end_task"]))]

    section_labels = ['find_ABCD', 'first_solve_correctly', 'all_repeats']
    events_across_ROIs = {}
    for ROI in region_list:
        all_events_task = event_dict[f"{task_index}_HFB_event_across_{ROI}"]
        
        # # Define normalized section ranges (e.g., each section is a third of the normalized scale)
        # normalized_sections = [(0, 0.33), (0.33, 0.67), (0.67, 1.0)]
        events_in_sections = {}
        for section_idx, (original_start, original_end) in enumerate(sections):
            # Find events in the current section
            events_in_sections[section_labels[section_idx]] = [event for event in all_events_task if original_start < event <= original_end]
        events_across_ROIs[ROI] = events_in_sections.copy()
        
    return events_across_ROIs

    
def neg_pos_fb_combo_in_three_sections_normalised(ripple_info_dict, feedback_session_sub_dict, task_index):
    # import pdb; pdb.set_trace()

    # define the sections I want to sort feedbacks and ripples by
    sections = [((ripple_info_dict[f"{task_index}_start_task"]), (ripple_info_dict[f"{task_index}_found_all_locs"])),
                ((ripple_info_dict[f"{task_index}_found_all_locs"]), (ripple_info_dict[f"{task_index}_first_corr_solve"])),
                ((ripple_info_dict[f"{task_index}_first_corr_solve"]), (ripple_info_dict[f"{task_index}_end_task"]))]

    # extract positive and negative feedbacks from this task
    if f"{task_index}_correct" in feedback_session_sub_dict:
        all_pos_task = feedback_session_sub_dict[f"{task_index}_correct"]
    if f"{task_index}_error" in feedback_session_sub_dict:
        all_neg_task = feedback_session_sub_dict[f"{task_index}_error"]
    
    # extract ripple events of this task
    all_events_task = ripple_info_dict[f"{task_index}_ripples_across_choi"]

    # split ripples into sections
    ripples_in_sections = {}
    section_labels = ['find_ABCD', 'first_solve_correctly', 'all_repeats']
    for section_idx, (original_start, original_end) in enumerate(sections):
        # Find events in the current section
        ripples_in_sections[section_labels[section_idx]] = [event for event in all_events_task if original_start < event <= original_end]

    

    # identify all positive-negative events combinations
    # Identify adjacent negative-positive pairs
    neg_pos_pairs = []
    used_indices = set()  # Track indices of used feedback to avoid double counting
    
    for i, neg_event in enumerate(all_neg_task):
        # Find the next positive feedback event after the current negative feedback
        for j, pos_event in enumerate(all_pos_task):
            if j in used_indices:  # Skip if the positive feedback was already paired
                continue
            if pos_event > neg_event:  # Check if the positive feedback is after the negative feedback
                # Ensure there are no other feedbacks (neg or pos) between this pair
                intermediate_events = [
                    event for event in all_neg_task[i + 1:] + all_pos_task[:j]
                    if neg_event < event < pos_event
                ]
                if not intermediate_events:
                    neg_pos_pairs.append((neg_event, pos_event))
                    used_indices.add(j)  # Mark the positive feedback as used
                    break  # Move to the next negative feedback

        
    normalized_feedback_data = {label: [] for label in section_labels}
    normalized_ripples_data = {label: [] for label in section_labels}
    section_durations = {label: [] for label in section_labels}  # Store durations for each section

    for neg_event, pos_event in neg_pos_pairs:
        # Define the time windows
        start_time = neg_event - 3
        end_time = pos_event + 3
        total_duration = end_time - start_time  # Always 6 seconds between -3 and +3 seconds

        # *** Adjusted Normalization: Normalize ripples and feedback using the correct scale ***
        def normalize_time(t):
            return (t - start_time) / total_duration * 3  # Map times to [0, 3] range

        # Normalize ripple events
        ripples_within_window = [
            normalize_time(ripple) for ripple in all_events_task
            if start_time <= ripple <= end_time
        ]

        # Add normalized feedback and ripples to the appropriate section
        for section_idx, (section_start, section_end) in enumerate(sections):
            if section_start <= neg_event <= section_end:
                section_label = section_labels[section_idx]
                # Feedback data
                normalized_feedback_data[section_label].append({
                    "neg_event": neg_event,
                    "pos_event": pos_event,
                    "normalized_times": {
                        "neg": normalize_time(neg_event),  # Should be 1.0
                        "pos": normalize_time(pos_event)   # Should be 2.0
                    }
                })
                # Ripple data
                normalized_ripples_data[section_label].append({
                    "neg_event": neg_event,
                    "pos_event": pos_event,
                    "normalized_ripples": ripples_within_window,
                    "duration": total_duration
                })
                break

    return normalized_ripples_data, normalized_feedback_data

    # for neg_event, pos_event in neg_pos_pairs:
    #     # # Define the time window: 3 seconds before the negative event to 3 seconds after the positive event
    #     # start_time = neg_event - 3
    #     # end_time = pos_event + 3
    #     # duration = end_time - start_time
        
        
    #     # Define the time windows for the three sections
    #     pre_neg_start = neg_event - 3
    #     pre_neg_end = neg_event
    #     between_start = neg_event
    #     between_end = pos_event
    #     post_pos_start = pos_event
    #     post_pos_end = pos_event + 3
        
    #     normalized_times = {
    #         "pre_neg": lambda t: ((t - pre_neg_start) / (pre_neg_end - pre_neg_start)) * -1,
    #         "between": lambda t: ((t - between_start) / (between_end - between_start)),
    #         "post_pos": lambda t: ((t - post_pos_start) / (post_pos_end - post_pos_start)) + 1
    #     }

    #     def normalize_time(t):
    #         if pre_neg_start <= t < pre_neg_end:  # Pre-negative
    #             return ((t - pre_neg_start) / (pre_neg_end - pre_neg_start)) * -1
    #         elif between_start <= t < between_end:  # Between negative and positive
    #             return ((t - between_start) / (between_end - between_start))
    #         elif post_pos_start <= t <= post_pos_end:  # Post-positive
    #             return ((t - post_pos_start) / (post_pos_end - post_pos_start)) + 1
    #         else:
    #             return None

    #     # Normalize ripples across the entire window
    #     ripples_within_window = [
    #         normalize_time(ripple) for ripple in all_events_task
    #         if pre_neg_start <= ripple <= post_pos_end
    #     ]

    #     # Remove None values (if any)
    #     ripples_within_window = [ripple for ripple in ripples_within_window if ripple is not None]

    #     # Add normalized feedback and ripples to the appropriate section
    #     for section_idx, (section_start, section_end) in enumerate(sections):
    #         if section_start <= neg_event <= section_end:
    #             section_label = section_labels[section_idx]
    #             # Feedback data
    #             normalized_feedback_data[section_label].append({
    #                 "neg_event": neg_event,
    #                 "pos_event": pos_event,
    #                 "normalized_times": {
    #                     "neg": normalize_time(neg_event),
    #                     "pos": normalize_time(pos_event)
    #                 }
    #             })
    #             # Ripple data
    #             normalized_ripples_data[section_label].append({
    #                 "neg_event": neg_event,
    #                 "pos_event": pos_event,
    #                 "normalized_ripples": ripples_within_window,
    #                 "duration": post_pos_end - pre_neg_start  # Total duration of the window
    #             })
    #             break


    # return normalized_ripples_data, normalized_feedback_data

    #     # Normalize times: 
    #     # -3 (start of the window) to 0 (negative feedback), 
    #     # 0 to 1 (negative to positive interval), 
    #     # 1 to 2 (positive feedback to end of window)
    #     duration_neg_to_pos = pos_event - neg_event
    #     normalized_times = [
    #         ((time - start_time) / (end_time - start_time)) * 2 - 1  # Map to range [-1, 1]
    #         for time in [start_time, neg_event, pos_event, end_time]
    #     ]
        
    #     # Assign normalized feedback and duration to the appropriate section
    #     for section_idx, (section_start, section_end) in enumerate(sections):
    #         if section_start <= neg_event <= section_end:
    #             section_label = section_labels[section_idx]
    #             normalized_feedback_data[section_label].append({
    #                 "neg_event": neg_event,
    #                 "pos_event": pos_event,
    #                 "normalized_times": normalized_times,
    #                 "duration": duration
    #             })
    #             section_durations[section_label].append(duration)
    #             break
        
    #     # Normalize ripple timings within the same window
    #     ripples_within_window = []
    #     for section_idx, section_ripples in ripples_in_sections.items():
    #         ripples_in_window = [
    #             ripple for ripple in section_ripples
    #             if start_time <= ripple <= end_time  # Ripples within the window
    #         ]
    #         # Normalize these ripple timings
    #         normalized_ripples = [
    #             ((ripple - start_time) / (end_time - start_time)) * 2 - 1
    #             for ripple in ripples_in_window
    #         ]
    #         ripples_within_window.extend(normalized_ripples)
       
        
    #     # Add normalized ripples to the appropriate section
    #     for section_idx, (section_start, section_end) in enumerate(sections):
    #         if section_start <= neg_event <= section_end:
    #             section_label = section_labels[section_idx]
    #             normalized_ripples_data[section_label].append({
    #                 "neg_event": neg_event,
    #                 "pos_event": pos_event,
    #                 "normalized_ripples": ripples_within_window,
    #                 "duration": duration
    #             })
    #             break


    # return normalized_ripples_data, normalized_feedback_data





    
def sort_feedback_in_three_sections(ripple_info_dict, feedback_session_sub_dict, task_index):
    # import pdb; pdb.set_trace()
    sections = [((ripple_info_dict[f"{task_index}_start_task"]), (ripple_info_dict[f"{task_index}_found_all_locs"])),
                ((ripple_info_dict[f"{task_index}_found_all_locs"]), (ripple_info_dict[f"{task_index}_first_corr_solve"])),
                ((ripple_info_dict[f"{task_index}_first_corr_solve"]), (ripple_info_dict[f"{task_index}_end_task"]))]

    section_labels = ['find_ABCD', 'first_solve_correctly', 'all_repeats']
    if f"{task_index}_correct" in feedback_session_sub_dict:
        all_pos_task = feedback_session_sub_dict[f"{task_index}_correct"]
    if f"{task_index}_error" in feedback_session_sub_dict:
        all_neg_task = feedback_session_sub_dict[f"{task_index}_error"]
    
    # # Define normalized section ranges (e.g., each section is a third of the normalized scale)
    # normalized_sections = [(0, 0.33), (0.33, 0.67), (0.67, 1.0)]
    feedback_in_sections = {}
    for section_idx, (original_start, original_end) in enumerate(sections):
        # Find events in the current section
        if f"{task_index}_correct" in feedback_session_sub_dict:
            feedback_in_sections[f"pos_{section_labels[section_idx]}"] = [event for event in all_pos_task if original_start < event <= original_end]
        if f"{task_index}_error" in feedback_session_sub_dict:
            feedback_in_sections[f"neg_{section_labels[section_idx]}"] = [event for event in all_neg_task if original_start < event <= original_end]
    # import pdb; pdb.set_trace()
    return feedback_in_sections
    

def extract_section_durations(ripple_info_dict, feedback_count, task_index, feedback_time):
    durations = {'find_ABCD': ripple_info_dict[f"{task_index}_found_all_locs"] - ripple_info_dict[f"{task_index}_start_task"],
                 'first_solve_correctly': ripple_info_dict[f"{task_index}_first_corr_solve"] - ripple_info_dict[f"{task_index}_found_all_locs"],
                 'all_repeats': ripple_info_dict[f"{task_index}_end_task"] - ripple_info_dict[f"{task_index}_first_corr_solve"]
                 }
    # then, define how much time is spent in each section for each feedback:
    time_per_feedback = {}
    for fb_sec in feedback_count:
        time_per_feedback[fb_sec] = len(feedback_count[fb_sec]) * feedback_time
    
    for section in durations:
        time_per_feedback[f"not_linked_to_fb_{section}"] = durations[section] - (time_per_feedback[f"pos_{section}"]+time_per_feedback[f"neg_{section}"])
        
    
        # sum(feedback_count*feedback_time)
        # duration - pos and neg feedback
    # import pdb; pdb.set_trace()
    
    return time_per_feedback

    


def sort_ripples_by_feedback(feedback_per_section_task, ripples_per_section_task, second_offset):
    # import pdb; pdb.set_trace()
    ripples_sorted_by_feedback_and_section_all_tasks = {}
    
    for task in sorted(feedback_per_section_task.keys()):
        ripples_sorted_by_feedback_and_section = {}
        for section_feedback_type in sorted(feedback_per_section_task[task].keys()):
            if section_feedback_type not in ripples_sorted_by_feedback_and_section:
                ripples_sorted_by_feedback_and_section[section_feedback_type] = []
            for feedback_time in feedback_per_section_task[task][section_feedback_type]:
                if section_feedback_type.endswith('find_ABCD'):
                    ripples_sorted_by_feedback_and_section[section_feedback_type].append([event for event in ripples_per_section_task[task]['find_ABCD'] if feedback_time <= event < feedback_time+second_offset])
                if section_feedback_type.endswith('first_solve_correctly'):
                    ripples_sorted_by_feedback_and_section[section_feedback_type].append([event for event in ripples_per_section_task[task]['first_solve_correctly'] if feedback_time <= event < feedback_time+second_offset])
                if section_feedback_type.endswith('all_repeats'):
                    ripples_sorted_by_feedback_and_section[section_feedback_type].append([event for event in ripples_per_section_task[task]['all_repeats'] if feedback_time <= event < feedback_time+second_offset])
            if section_feedback_type in ripples_sorted_by_feedback_and_section and len(ripples_sorted_by_feedback_and_section[section_feedback_type]) > 1:
                ripples_sorted_by_feedback_and_section[section_feedback_type] = [item for sublist in ripples_sorted_by_feedback_and_section[section_feedback_type] for item in sublist]
        
        ripples_sorted_by_feedback_and_section_all_tasks[task] = ripples_sorted_by_feedback_and_section.copy()
    # by task,by section, categorise as ripple pos if within 1.5 secs after pos feedback,
    # and as ripple neg if within 1.5 secs after neg feedback.
    
    return ripples_sorted_by_feedback_and_section_all_tasks
    

def sort_ripples_by_feedback_before_vs_after(feedback_per_section_task, ripples_per_section_task, second_offset):
    # import pdb; pdb.set_trace()
    ripples_sorted_by_feedback_and_section_all_tasks = {}
    for task in sorted(feedback_per_section_task.keys()):
        
        ripples_sorted_by_feedback_and_section = {}
        for section_feedback_type in sorted(feedback_per_section_task[task].keys()):
            ripples_sorted_by_feedback_and_section[f"{section_feedback_type}_after_event"] = []
            ripples_sorted_by_feedback_and_section[f"{section_feedback_type}_before_event"] = []
            
        for section_feedback_type in sorted(feedback_per_section_task[task].keys()):
            for feedback_time in feedback_per_section_task[task][section_feedback_type]:
                if section_feedback_type.endswith('find_ABCD'):
                    ripples_sorted_by_feedback_and_section[f"{section_feedback_type}_after_event"].append([event for event in ripples_per_section_task[task]['find_ABCD'] if feedback_time <= event < feedback_time+second_offset])
                    ripples_sorted_by_feedback_and_section[f"{section_feedback_type}_before_event"].append([event for event in ripples_per_section_task[task]['find_ABCD'] if feedback_time-second_offset <= event < feedback_time])
                if section_feedback_type.endswith('first_solve_correctly'):
                    ripples_sorted_by_feedback_and_section[f"{section_feedback_type}_after_event"].append([event for event in ripples_per_section_task[task]['first_solve_correctly'] if feedback_time <= event < feedback_time+second_offset])
                    ripples_sorted_by_feedback_and_section[f"{section_feedback_type}_before_event"].append([event for event in ripples_per_section_task[task]['first_solve_correctly'] if feedback_time-second_offset <= event < feedback_time])
                if section_feedback_type.endswith('all_repeats'):
                    ripples_sorted_by_feedback_and_section[f"{section_feedback_type}_after_event"].append([event for event in ripples_per_section_task[task]['all_repeats'] if feedback_time <= event < feedback_time+second_offset])
                    ripples_sorted_by_feedback_and_section[f"{section_feedback_type}_before_event"].append([event for event in ripples_per_section_task[task]['all_repeats'] if feedback_time-second_offset <= event < feedback_time])
   
            if f"{section_feedback_type}_after_event" in ripples_sorted_by_feedback_and_section and len(ripples_sorted_by_feedback_and_section[f"{section_feedback_type}_after_event"]) > 1:
                ripples_sorted_by_feedback_and_section[f"{section_feedback_type}_after_event"] = [item for sublist in ripples_sorted_by_feedback_and_section[f"{section_feedback_type}_after_event"] for item in sublist]
            if f"{section_feedback_type}_before_event" in ripples_sorted_by_feedback_and_section and len(ripples_sorted_by_feedback_and_section[f"{section_feedback_type}_before_event"]) > 1:    
                ripples_sorted_by_feedback_and_section[f"{section_feedback_type}_before_event"] = [item for sublist in ripples_sorted_by_feedback_and_section[f"{section_feedback_type}_before_event"] for item in sublist]
            
            ripples_sorted_by_feedback_and_section_all_tasks[task] = ripples_sorted_by_feedback_and_section.copy()
    
    return ripples_sorted_by_feedback_and_section_all_tasks






def count_event_values(ripple_dict, feedback_time, second_offset, sigma): 
    ripple_values = []
    for ripple in ripple_dict:
        if feedback_time <= ripple < feedback_time + second_offset:
            # Compute the time offset
            time_offset = ripple - feedback_time
            # Evaluate the Gaussian at this offset
            gaussian_value = norm.pdf(time_offset, loc=0, scale=sigma)
            ripple_values.append(gaussian_value)  
    return ripple_values



# NOT DONE!!
def count_gaussian_events_around_ripples(feedback_per_section_task, ripples_per_section_task):
    import pdb; pdb.set_trace()
    second_offset = 1.5 # define this as not anymore within the ripple vicinity.
    # define the standard deviation for y=1 at x=0
    sigma = 1 / np.sqrt(2 * np.pi) 
    # if a value is that far away, then the value would be 0.000851... etc anyways.
    
    # ok do this differently. 
    # I want to give each event that occurs just before a ripple a value.
    
    ripples_sorted_by_feedback_and_section_all_tasks = {}
    for task in sorted(ripples_per_section_task.keys()):
        for section in sorted(ripples_per_section_task[task].keys()):
            for ripple in ripples_per_section_task[task][section]:
                # for this ripple, count at which distance an event occurred.
                # this will be 
                for section_type in sorted(feedback_per_section_task[task].keys()):
                    for event in feedback_per_section_task[task][section_type]:
                        # if the event happened within 1.5secs before ripple
                        if ripple - second_offset <= event < ripple:
                            event_offset = event - ripple
                            gaussian_value = norm.pdf(event_offset, loc=0, scale=sigma)
                            
                        
                        
    
    for task in sorted(feedback_per_section_task.keys()):
        ripples_sorted_by_feedback_and_section = {}
        for section_feedback_type in sorted(feedback_per_section_task[task].keys()):
            for feedback_time in feedback_per_section_task[task][section_feedback_type]:
                if section_feedback_type.endswith('find_ABCD'):
                    if section_feedback_type not in ripples_sorted_by_feedback_and_section:
                        ripples_sorted_by_feedback_and_section[section_feedback_type] = []
    
                    event_values = mc.analyse.ripple_helpers.count_event_values(ripples_per_section_task[task]['find_ABCD'], feedback_time, second_offset, sigma)
                    ripples_sorted_by_feedback_and_section[section_feedback_type].append(event_values)
                
                if section_feedback_type.endswith('first_solve_correctly'):
                    if section_feedback_type not in ripples_sorted_by_feedback_and_section:
                        ripples_sorted_by_feedback_and_section[section_feedback_type] = []
                    event_values = mc.analyse.ripple_helpers.count_event_values(ripples_per_section_task[task]['first_solve_correctly'], feedback_time, second_offset, sigma)
                    ripples_sorted_by_feedback_and_section[section_feedback_type].append(event_values)

                if section_feedback_type.endswith('all_repeats'):
                    if section_feedback_type not in ripples_sorted_by_feedback_and_section:
                        ripples_sorted_by_feedback_and_section[section_feedback_type] = []
                    event_values = mc.analyse.ripple_helpers.count_event_values(ripples_per_section_task[task]['all_repeats'], feedback_time, second_offset, sigma)
                    ripples_sorted_by_feedback_and_section[section_feedback_type].append(event_values)
    
            # Flatten lists if needed (keeping only one list per feedback type)
            if section_feedback_type in ripples_sorted_by_feedback_and_section and len(ripples_sorted_by_feedback_and_section[section_feedback_type]) > 1:
                ripples_sorted_by_feedback_and_section[section_feedback_type] = [
                    item for sublist in ripples_sorted_by_feedback_and_section[section_feedback_type] for item in sublist
                ]
        ripples_sorted_by_feedback_and_section_all_tasks[task] = ripples_sorted_by_feedback_and_section.copy()
        
    return ripples_sorted_by_feedback_and_section_all_tasks


    
    