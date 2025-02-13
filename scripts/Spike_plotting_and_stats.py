#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:13:20 2024

Looking at spiking data in Habiba's dataset.


"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import mc
import math 

# I want to check for place-cell coding, ideally reward-place coding
# start with mRT2bHaEa04-12 in s5
# and mRT2bHb01-33 in s11
# and mLT2aHaE06-6 in s9
# these are cells that respond to ripples


# goal: plot firing rate (hz) by space on a 3x3 grid.
# and do this analysis across all grids, and then also between grids!
# goal: is there 'remapping' in hippocampus with ripples?
# also check ripples and remapping literature
# e.g. https://www.nature.com/articles/nn.4543.pdf  or
# https://www.jneurosci.org/content/jneuro/43/12/2153.full.pdf
# "Remapping cells were recruited into sharp-wave ripples and associated replay events to a greater extent than stable
# cells, despite having similar firing rates during navigation of the maze."



# 
LFP_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
result_dir = f"{LFP_dir}/results"
ROI = 'HPC' # HPC mPFC
analysis_type = 'grid_wise' # grid_wise, exploration_trials 
preproc_type = 'channel_wise' # channel_wise 'referenced'

# sessions = ['s5', 's9', 's11']
sessions = ['s5']
# session_per_subj = {
#     'sub-1': ['s5'],
#     'sub-2': ['s9'],
#     'sub-3': ['s11'],
#     }
session_per_subj = {
    'sub-1': ['s5'],
    }

for sub in session_per_subj:
    for sesh in session_per_subj[sub]:
        # at some point do all data computations in here, and then collect them all for all sessions one subject had.
        timings_and_locs = np.genfromtxt(f"{LFP_dir}/s05/location_and_timings.csv", delimiter=',')
        # load behaviour that defines my snippets.
        behaviour_all = np.genfromtxt(f"{LFP_dir}/s05/all_trials_times.csv", delimiter=',')
        
        button_presses = np.genfromtxt(f"{LFP_dir}/s05/button_presses.csv", delimiter = ',')
        # load behavioural stuff you need for plotting all sorts of stuff
        seconds_lower, seconds_upper, task_config, task_index, task_onset, new_grid_onset, found_first_D = mc.analyse.ripple_helpers.prep_behaviour(behaviour_all)
    
        spiking_dict = mc.analyse.ripple_helpers.load_spiking_data('s05')

        with open(f"{result_dir}/{sesh}_{ROI}_{analysis_type}_{preproc_type}_ripple_by_seconds.pkl", 'rb') as file:
            ripples = pickle.load(file)
        
        with open(f"{result_dir}/{sesh}_{ROI}_{analysis_type}_feedback.pkl", 'rb') as file:
            feedback_dict = pickle.load(file)
            
        
        # import pdb; pdb.set_trace() 
            
        # PLACE TUNING ACROSS GRIDS
        # first: determine the intervals by locations.
        # second: determine the frequency (i.e. count of spikes divided by time) per interval
        # third: add the frequency to a dictionary that has locations as keys.
        # fourth: determine average firing frequency per grid.
        
        frequency_per_cell = {}
        frequency_per_cell_gridwise = {}
        
        for cell in spiking_dict:
            frequency_by_location, frequency_by_grid =  {}, {}
            for loc in range(1,10):
                frequency_by_location[loc] = []
                
            for grid in range(1, len(new_grid_onset)):
                frequency_by_grid[grid] = {loc: [] for loc in range(1, 10)}  # Grid -> Location
                
            for row, data_row in enumerate(timings_and_locs):
                curr_loc = data_row[1]
                if row == len(timings_and_locs)-1:
                    continue
                
                next_loc = timings_and_locs[row+1, 1]
                
                if row == 0:
                    start_interval = timings_and_locs[0, 0]
                else:
                    previous_loc = timings_and_locs[row-1, 1]
                    if curr_loc != previous_loc:
                        start_interval = timings_and_locs[row-1, 0]
                if curr_loc != next_loc:
                    stop_interval = timings_and_locs[row, 0]  
                    duration_interval = stop_interval - start_interval
                    # Filter spike times within the interval
                    filtered_spikes = spiking_dict[cell][(spiking_dict[cell] >= start_interval) & (spiking_dict[cell] <= stop_interval)]
                    # Count the number of spikes
                    num_spikes = len(filtered_spikes)
                    # Calculate the frequency (spikes per second)
                    frequency = num_spikes / duration_interval
                    frequency_by_location[curr_loc].append(frequency)
                    
                    # Assign to grid and location within the nested dictionary
                    for grid_no, onset_time in enumerate(new_grid_onset[:-1]):  # Exclude the last onset as there is no next grid
                        if stop_interval > onset_time and stop_interval <= new_grid_onset[grid_no + 1]:
                            frequency_by_grid[grid_no + 1][curr_loc].append(frequency)
                            break  # Ensure each interval is assigned to only one grid


            frequency_per_cell_gridwise[cell] = frequency_by_grid.copy()
            frequency_per_cell[cell] = frequency_by_location.copy()
                
               
            # Plot the heatmap across grids 
            means = [np.mean(values) for values in frequency_by_location.values()]

            # Reshape the mean values into a 3x3 grid
            heatmap_data = np.array(means).reshape(3, 3)
            
            # Create the heatmap
            plt.figure(figsize=(6, 6))
            plt.imshow(heatmap_data, cmap='viridis', aspect='equal')
            plt.colorbar(label='Firing Frequency (Hz)')
            
            # Add labels for clarity
            fields = [f"Field {i}" for i in range(1, 10)]
            for i in range(3):
                for j in range(3):
                    plt.text(j, i, f"{fields[i * 3 + j]}\n{heatmap_data[i, j]:.2f}", 
                             ha='center', va='center', color='white', fontsize=10) 
            # Customize the axis
            plt.xticks([])
            plt.yticks([])
            plt.title(f"Firing Frequency across grids for {cell}, {sesh}", fontsize=14)
            plt.show()   
                    
     
            
    # GRIDWISE PLACE TUNING 
    task_labels = ["A", "B", "C", "D"]
    for cell, grids in frequency_per_cell_gridwise.items():
        num_grids = len(grids)
        
        # Determine the global min and max across all grids and locations
        all_means = [
            np.mean(values) if values else 0
            for grid in grids.values()
            for values in grid.values()
        ]
        global_min, global_max = min(all_means), max(all_means)


        # Determine the number of rows and columns for the subplot grid
        cols = math.ceil(math.sqrt(num_grids))  # Number of columns
        rows = math.ceil(num_grids / cols)     # Number of rows
    
        # Create a subplot grid
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)
        axes = axes.flatten()  # Flatten axes to iterate easily
    
        for grid_no, (grid, locations) in enumerate(grids.items()):
            # Calculate mean firing frequency for each location
            means = [np.mean(values) if values else 0 for values in locations.values()]  # Handle empty lists
            
            # Reshape the mean values into a 3x3 grid
            heatmap_data = np.array(means).reshape(3, 3)
            
            # Plot heatmap for this grid
            ax = axes[grid_no]
            im = ax.imshow(heatmap_data, cmap='viridis', aspect='equal', vmin=global_min, vmax=global_max)
            ax.set_title(f"Grid {grid}", fontsize=10)
            
            # Add annotations
            fields = [f"Loc {i}" for i in range(1, 10)]
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{fields[i * 3 + j]}\n{heatmap_data[i, j]:.2f}",
                            ha='center', va='center', color='white', fontsize=8)

            # Find task configuration for this grid
            column_index= np.where(np.array(task_index) == grid)[0][0]
            task_fields = task_config[column_index]  # Retrieve the corresponding task configuration
            # Annotate with A, B, C, D based on task configuration
            for label_idx, field in enumerate(task_fields):  # A->field[0], B->field[1], ...
                x = (field - 1) % 3  # Column index
                y = (field - 1) // 3  # Row index
                ax.text(x, y, task_labels[label_idx], color='white', ha='center', va='center', fontsize=10, fontweight='bold')

    
        # Turn off unused axes
        for idx in range(num_grids, len(axes)):
            axes[idx].axis('off')
        
        # Add a single colorbar for all subplots
        cbar = fig.colorbar(im, ax=axes, location='right', shrink=0.8, label='Firing Frequency (Hz)')
        
        # Set the overall figure title
        fig.suptitle(f"Place-Tuning Heatmaps for Cell {cell} {sesh}", fontsize=16)
        plt.show()
    