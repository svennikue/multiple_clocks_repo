#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:48:43 2024

Functions for Ripples.
Human LFPs.


@author: xpsy1114
"""

import pdb
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import neo
import seaborn as sns
from scipy.stats import gaussian_kde

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
    




