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


def plot_ripples_per_channel(data_dict, channel_list, subject):
    # import pdb; pdb.set_trace()

    # Group channels by family based on prefixes
    wires = defaultdict(list)
    for channel in channel_list:
        prefix = channel[:6]  # First 6 characters
        wires[prefix].append(channel)
    
    # Prepare data for plotting
    ripple_freq_per_channel = defaultdict(list)
    
    for family, channels in wires.items():
        for channel in channels:
            total_sum = 0
            for task, ripple_count in data_dict.items():
                if channel in ripple_count:
                    total_sum += len(ripple_count[channel])
            ripple_freq_per_channel[family].append(total_sum)
    
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Number of Ripples per Microwire')
    
    # Flatten axs for easy indexing
    axs = axs.flatten()
    # import pdb; pdb.set_trace()
    # Plot each family in its own subplot
    for i, (wire, sums) in enumerate(ripple_freq_per_channel.items()):
        axs[i].bar(range(len(sums)), sums, align='center')
        axs[i].set_title(f'Microwire: {wire} for subject {subject}')
        axs[i].set_xlabel('Channel Index')
        axs[i].set_xticks(range(len(sums)))  # Set positions
        axs[i].set_xticklabels(wires[wire], rotation=45, ha='right')  # Set labels and rotate for readability
        axs[i].set_ylabel('Ripple Amount')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    

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
    



def plot_ripple_distribution(onset_in_secs_dict, task_to_check, feedback_error_curr_task, feedback_correct_curr_task, onset_secs, found_first_D, seconds_upper, index_upper, index_lower, seconds_lower, sub):

    # y_jitter = {key: np.random.uniform(0, 0.01, len(values)) for key, values in onset_in_secs_dict[task_to_check].items()}
    # colors = plt.cm.get_cmap('tab10', len(onset_in_secs_dict[task_to_check]))
    # Define a function to extract the prefix (first 6 characters or customize as needed)
    def get_prefix(condition):
        return condition[:6]  # Adjust the slice based on how much of the key you want to use as prefix
    
    # Group by prefixes and assign y ranges to each group
    prefixes = set(get_prefix(key) for key in onset_in_secs_dict[task_to_check])
    y_ranges = {prefix: i for i, prefix in enumerate(sorted(prefixes))}
    
    # Create y-jitter based on the prefix group
    y_jitter = {}
    for condition, values in onset_in_secs_dict[task_to_check].items():
        prefix = get_prefix(condition)
        base_y = (y_ranges[prefix]+1)*0.01  # Base y-position for this prefix group
        y_jitter[condition] = base_y + np.random.uniform(0, 0.003, len(values))  # Jitter within a small range


    # # Create a KDE plot for the data
    # plt.figure();
    # for idx, (condition, values) in enumerate(onset_in_secs_dict[task_to_check].items()):
    #     plt.scatter(values, y_jitter[condition], color=colors(idx), label=condition, zorder = 1)

    # new
    
    # Define the colors: specific for Ha08, Ha09, Hb08, Hb09; grey otherwise
    color_mapping = {
        'Ha08': 'navy',  # Set the desired color for Ha08
        'HaEa08': 'navy',  # Set the desired color for Ha08
        'Ha09': 'teal',  # Set the desired color for Ha09
        'HaEa09': 'teal',  # Set the desired color for Ha09
        'Hb08': 'orchid',    # Set the desired color for Hb08
        'HbE08': 'orchid',    # Set the desired color for Hb08
        'Hb09': 'salmon',  # Set the desired color for Hb09
        'HbE09': 'salmon'  # Set the desired color for Hb09
    }
    default_color = 'grey'
    
    # Define marker types based on key starting letter
    def get_marker(condition):
        if condition.startswith('R'):
            return 'o'  # Dot for 'R'
        elif condition.startswith('L'):
            return 'x'  # Cross for 'L'
        return 'o'  # Default to dot if no match
    
    # Define color function based on condition
    def get_color(condition):
        for key in color_mapping.keys():
            if key in condition:
                return color_mapping[key]
        return default_color
    
    # Create the scatter plot
    plt.figure()
    for condition, values in onset_in_secs_dict[task_to_check].items():
        color = get_color(condition)
        marker = get_marker(condition)
        plt.scatter(values, y_jitter[condition], color=color, marker=marker, label=condition, zorder=1)

    
    # plt.scatter(onset_secs, y_jitter, color='black', label='Ripple Candidates', zorder=1)
    sns.kdeplot(onset_secs, fill=True, color='skyblue', label='Ripple Distribution')
    
    # Add a vertical line for the baseline reference
    plt.axvline(x=(found_first_D[task_to_check-1]), color='black', linestyle='--', label='Found all 4 rewards')
    plt.axvline(x=(seconds_upper[index_lower]), color='black', linestyle='--', label='First Correct')
    
    
    # Add red rods for feedback: incorrect
    sns.rugplot(feedback_error_curr_task, height=0.1, color='crimson', lw=2)  # Each data point as a 'rug'

    # Add green rods for feedback: correct
    sns.rugplot(feedback_correct_curr_task, height=0.1, color='darkgreen', lw=2)  # Each data point as a 'rug'

    
    # Add titles and labels
    plt.title(f"Ripple count (clustered by wire) for grid {task_to_check} for subj {sub} [10 correct repeats]")
    plt.xlabel('Seconds in Task')
    plt.ylabel('Ripple Frequency')
    
    plt.xlim(seconds_lower[index_lower], seconds_upper[index_upper])
    
    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
def plot_ripple(freq_to_plot, title, downsampled_data, event, min_length_ripple, filtered_cropped_vhgamma_np, power_dict, repeat, freq_bands_keys, y_label_power):
    # event[0] = onset
    # event[1] = duration
    # event[-1] = channel index
    fig, axs = plt.subplots(4)
    fig.suptitle(title)

    # Create x-values from 5500 to 9500
    # x = np.linspace(event[0]-freq_to_plot, event[0]+freq_to_plot-1, freq_to_plot*2)
    x = np.arange(-freq_to_plot, freq_to_plot) * 2
    
    # import pdb; pdb.set_trace()
    # first subplot is the raw signal:
    # axs[0].plot(x, raw_np_cropped[event[0]-freq_to_plot:event[0]+freq_to_plot, event[-1]], linewidth = 0.2)
    axs[0].plot(x, downsampled_data[event[0]-freq_to_plot:event[0]+freq_to_plot, event[-1]], linewidth = 0.2)
    axs[0].set_title('downsampled raw LFP')
    axs[0].set_xlabel('Time (ms)')
    # set a few x-ticks around the ripple in 20ms resolution. 
    axs[0].set_xticks(np.arange(0-6*min_length_ripple, 7*min_length_ripple, 3*min_length_ripple))  # Set x-ticks from 0 to 10 with a step of 0.5


    # the second subplot will be filtered for high gamma:
    axs[1].plot(x, filtered_cropped_vhgamma_np[event[-1], event[0]-freq_to_plot:event[0]+freq_to_plot], linewidth = 0.2)    
    axs[1].set_title('hgamma filtered signal')   
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_xticks(np.arange(0-6*min_length_ripple, 7*min_length_ripple, 3*min_length_ripple))  # Set x-ticks from 0 to 10 with a step of 0.5

    
    # the third subplot will be the mean power of this frequency: 
    axs[2].plot(x,power_dict[f"{repeat}_mean"]['hgamma'][event[-1], event[0]-freq_to_plot:event[0]+freq_to_plot])
    axs[2].set_title('Mean power hgamma')
    axs[2].set_xticks(np.arange(0-6*min_length_ripple, 7*min_length_ripple, 3*min_length_ripple))  # Set x-ticks from 0 to 10 with a step of 0.5

    ## the fourth subplot is the vhgamma power spectrum
    #power_to_plot_low = power_dict[f"{repeat}_stepwise"]['vhgamma'][event[-1], :, event[0]-freq_to_plot:event[0]+freq_to_plot] # Select first epoch and the specified channel
    #axs[3].imshow(power_to_plot_low, aspect='auto', origin='lower')
    
    # the fifth subplot is the overall power spectrum
    # power_to_plot_all = np.stack(power_all[freq_bands_keys[0]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq], power_all[freq_bands_keys[1]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq], power_all[freq_bands_keys[2]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq])
    power_to_plot_all = np.vstack((power_dict[f"{repeat}_stepwise"][freq_bands_keys[0]][event[-1], :, event[0]-freq_to_plot:event[0]+freq_to_plot], power_dict[f"{repeat}_stepwise"][freq_bands_keys[1]][event[-1], :, event[0]-freq_to_plot:event[0]+freq_to_plot]))
    power_to_plot_all = np.vstack((power_to_plot_all, power_dict[f"{repeat}_stepwise"][freq_bands_keys[2]][event[-1], :, event[0]-freq_to_plot:event[0]+freq_to_plot]))
    power_to_plot_all = np.vstack((power_to_plot_all, power_dict[f"{repeat}_stepwise"][freq_bands_keys[3]][event[-1], :, event[0]-freq_to_plot:event[0]+freq_to_plot]))
    axs[3].imshow(power_to_plot_all, aspect='auto', origin='lower')
    y_ticks = [5, 25, 45, 65, power_to_plot_all.shape[0] - 1]  # 5, 25, 45, 65 and the max value (y-axis max)

    # Setting the yticks and the labels
    axs[3].set_title('Power spectra of all frequency bands stacked')
    axs[3].set_yticks(y_ticks[:-1])  # Add the desired tick positions except the last one
    axs[3].set_yticks([y_ticks[-1]], minor=True)  # Add the max position as a minor tick
    axs[3].set_yticklabels(y_label_power)  # Add the desired tick labels
    plt.tight_layout()

