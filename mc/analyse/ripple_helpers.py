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



def plot_ripples_per_channel(data_dict, channel_list):
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
        axs[i].set_title(f'Microwire: {wire}')
        axs[i].set_xlabel('Channel Index')
        axs[i].set_xticks(range(len(sums)))  # Set positions
        axs[i].set_xticklabels(wires[wire], rotation=45, ha='right')  # Set labels and rotate for readability
        axs[i].set_ylabel('Ripple Amount')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    