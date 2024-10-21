#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:12:02 2024

Functions for Ripples.
Human LFPs.


@author: Svenja Kuchenhoff
"""

import pdb
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import neo
import seaborn as sns
from scipy.stats import gaussian_kde


# import pdb; pdb.set_trace() 

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
    
    
    
def plot_ripple_distribution(onset_in_secs_dict, task_to_check, feedback_error_curr_task, feedback_correct_curr_task, ripples_across_channels, found_first_D, seconds_upper, index_upper, index_lower, seconds_lower, sub):
    
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
        base_y = (y_ranges[prefix]+1)*0.15  # Base y-position for this prefix group
        y_jitter[condition] = base_y + np.random.uniform(0, 0.08, len(values))  # Jitter within a small range
    
    # Define the colors: specific for Ha08, Ha09, Hb08, Hb09; grey otherwise
    color_mapping = {
        'Ha08': 'purple',  # Set the desired color for Ha08
        'HaEa08': 'purple',  # Set the desired color for Ha08
        'Ha09': 'maroon',  # Set the desired color for Ha09
        'HaEa09': 'maroon',  # Set the desired color for Ha09
        'Hb08': 'teal',    # Set the desired color for Hb08
        'HbE08': 'teal',    # Set the desired color for Hb08
        'Hb09': 'ligthblue',  # Set the desired color for Hb09
        'HbE09': 'lightblue'  # Set the desired color for Hb09
    }
    default_color = 'grey'
    
    # import pdb; pdb.set_trace() 
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
        plt.scatter(values, y_jitter[condition], color=color, marker=marker, label=condition, zorder=1, alpha = 0.5)
    
    # Calculate KDE using scipy's gaussian_kde
    kde = gaussian_kde(ripples_across_channels)
    x_values = np.linspace(min(ripples_across_channels), max(ripples_across_channels), 1000)
    
    # Evaluate the KDE on these x-values
    y_values = kde(x_values)
    
    # Normalize the y-values so that the maximum y-value is 1
    y_values_normalized = y_values / max(y_values)
    
    # Plot the normalized KDE
    plt.plot(x_values, y_values_normalized, color='lavender', label='Ripple Distribution')
    
    # Fill under the curve if you want to mimic the 'fill=True' from sns.kdeplot
    plt.fill_between(x_values, y_values_normalized, color='lavender', alpha=0.4)

    # sns.kdeplot(ripples_across_channels, fill=True, color='skyblue', label='Ripple Distribution')
    
    # Add a vertical line for the baseline reference
    plt.axvline(x=(found_first_D[task_to_check-1]), color='black', linestyle='--', label='Found all 4 rewards')
    plt.axvline(x=(seconds_upper[index_lower]), color='black', linestyle='--', label='First Correct')

    
    # Add red rods for feedback: incorrect
    sns.rugplot(feedback_error_curr_task, height=0.1, color='salmon', lw=2)  # Each data point as a 'rug'

    # Add green rods for feedback: correct
    sns.rugplot(feedback_correct_curr_task, height=0.1, color='yellowgreen', lw=2)  # Each data point as a 'rug'

    
    # Add titles and labels
    plt.title(f"Ripple count (clustered by wire) for grid {task_to_check} for subj {sub} [10 correct repeats]")
    plt.xlabel('Seconds in Task')
    plt.ylabel('Ripple Frequency')
    
    plt.xlim(seconds_lower[index_lower], seconds_upper[index_upper])
    plt.ylim(0, 1.2)
    
    # Add a legend
    # Get current handles and labels from the plot
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Limit the legend to show only the first 6 items
    plt.legend(handles[:8], labels[:8], loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
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
    
    
    
    
   
# import pdb; pdb.set_trace()     

def ripple_count_barchart(ripples_by_state_per_repeat, n_bars, titlestring, xstring, ystring):
    # Initialize the heights list for 40 bars
    heights = []
    # Loop from bar 1 to n-bars and get the corresponding value from the dictionary
    for i in range(1, n_bars+1):
        if i in ripples_by_state_per_repeat:
            heights.append(ripples_by_state_per_repeat[i])  # Use the value from the dictionary
        else:
            heights.append(0)  # If no key exists for this index, append 0
    
    # Generate labels for the x-axis (1 to n-bars)
    labels = [str(i) for i in range(1, n_bars+1)]
    
    plt.figure();
    # Create the bar plot
    plt.bar(labels, heights)
    
    # Add title and labels
    plt.title(titlestring)
    plt.xlabel(xstring)
    plt.ylabel(ystring)
    
    # Show the plot
    plt.show()
    
    
    
def ripple_amount_violin_scatter(plotting_dict, titlestring, xstring, ystring):
    plt.figure(figsize=(8, 6))
    
    # Extract the data for boxplots
    data_for_boxplots = [plotting_dict[key] for key in plotting_dict.keys()]
    nan_filtered_data = [list(filter(lambda x: not np.isnan(x), sublist)) for sublist in data_for_boxplots]
    for i, sublist in enumerate(nan_filtered_data):
        if sublist == []:
            nan_filtered_data[i] = 0
    # Plot boxplots
    #plt.boxplot(data_for_boxplots, widths=0.6, patch_artist=True)
    plt.violinplot(nan_filtered_data, showmeans=True)
    
    # Add scatter points for each boxplot
    for idx, (key, data_points) in enumerate(plotting_dict.items()):
        # Add some jitter to avoid overlap of scatter points
        jitter = np.random.uniform(-0.2, 0.2, size=len(data_points))  # Add small random jitter
        # Plot scatter points with jitter
        plt.scatter(np.ones(len(data_points)) * (idx + 1) + jitter, data_points, color='salmon', alpha=0.6, linewidths= 0.5)
    
    max_repeats = int(len(plotting_dict)/4)
    x_tick_labels = max_repeats*['A', 'B', 'C', 'D']
    # Set x-ticks to dictionary keys (boxplot labels)
    plt.xticks(np.arange(1, len(plotting_dict) + 1), x_tick_labels)
    
    # Label and title
    plt.xlabel(xstring)
    
    plt.ylabel(ystring)
    plt.title(titlestring)
    plt.show()        
          
        
    
    
    
    
    
    
    
