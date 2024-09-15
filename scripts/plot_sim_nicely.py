#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:28:36 2023

plotting for the transfer report.
this is to show how I build up my simulations.

@author: xpsy1114
"""

import colormaps as cmaps

import mc
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import os

date = np.datetime64('today')
out_path = f"/Users/xpsy1114/Documents/projects/multiple_clocks/output/{date}/"
if not os.path.isdir(out_path):
    os.makedirs(out_path)


reward_coords = [[2, 0], [0, 2], [0, 0], [1, 1]]
time_per_step = 10
walk, steps_per_walk = mc.simulation.grid.walk_paths(reward_coords, 3, plotting = True)
locationm, phasem, statem, midnightm, clockm, phasestatem= mc.simulation.predictions.set_continous_models(walk, steps_per_walk, time_per_step, grid_size = 3, no_phase_neurons=3, fire_radius = 0.25, wrap_around = 1)

model_dict = {}
model_dict['Location'] = locationm
model_dict['Goal Progress'] = phasem
model_dict['State'] = statem
model_dict['Partial Schema'] = midnightm
model_dict['Schema'] = clockm
model_dict['1 Schema Module'] = clockm[120:132, :]



states = ['A', 'B', 'C', 'D']
steps_per_walk = [(step*time_per_step) for step in steps_per_walk]
cumsteps_per_walk = np.cumsum(steps_per_walk)
xticks = [2*time_per_step, 5*time_per_step, 7*time_per_step, 9*time_per_step]

cmaps.bilbao
cmap = plt.get_cmap('bilbao')
    
    

#plot all the first simulations nicely.

for any_matrix in model_dict:
    # import pdb; pdb.set_trace()
    fig, ax = plt.subplots(figsize=(5,4))
    
    plt.imshow(model_dict[any_matrix], aspect = 'auto', interpolation= 'none', cmap=cmap) 
    
    # Create a wrapped title with a maximum of 20 characters per line
    title = f"{any_matrix}"
    wrapped_title = '\n'.join(textwrap.wrap(title, width=20))
    # Set the wrapped title with larger font size
    ax.set_title(wrapped_title, fontsize=18)
    
    field_names = np.arange(0,9)

    ax.set_ylabel('simulated neurons', fontsize = 16)

    
    for interval in cumsteps_per_walk:
        ax.axvline(interval-0.5, color='black', linewidth=1)
    
    if any_matrix == 'Partial Schema':
        ax.axhline(0.5, color='grey', linewidth=1)
        ax.axhline(1.5, color='grey', linewidth=1)
        for new_module in range(0,27, 3):
            ax.axhline(new_module-0.5, color='black', linewidth=1)
        ax.set_yticks(range(1,27,3))
        ax.set_yticklabels([str(f) for f in field_names], fontsize = 16)
    if any_matrix == 'Schema':
        for new_module in range(0,324, 12):
            ax.axhline(new_module-0.5, color='grey', linewidth=1)
        for new_field in range(0,324, 36):
            ax.axhline(new_field-0.5, color='black', linewidth=1)
        ax.set_yticks(range(13,324,36))
        ax.set_yticklabels([str(f) for f in field_names], fontsize = 16)
    if any_matrix == '1 Schema Module':
        for new_module in range(0,12, 3):
            ax.axhline(new_module-0.5, color='black', linewidth=1)
            
        
    # Set x-axis and y-axis ticks and labels with 45-degree rotation
    #ticks = [int(interval/2) for interval in steps_per_walk]
    
    
    if len(model_dict[any_matrix]) < 20:
        ax.set_yticks(range(len(model_dict[any_matrix])))
        
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"State {s}" for s in states], rotation=45, ha='right', fontsize = 16)
    ax.grid(False)
    
    plt.tight_layout()
    
    fig.savefig(f"{out_path}{any_matrix}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{out_path}{any_matrix}.tiff", dpi=300, bbox_inches='tight')