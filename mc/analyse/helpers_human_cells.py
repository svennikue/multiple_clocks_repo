#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:41:06 2025

@author: Svenja Küchenhoff
helpers functions to analyse human cells


"""

import numpy as np
# import pdb; pdb.set_trace()

def load_cell_data(source_folder, subject_list):
    location_dir = {}
    cell_dir = {}
    for sub in subject_list:
        location_dir[f"sub-{sub}"] = np.genfromtxt(f"{source_folder}/s{sub}/locations_per_50ms_sub_{sub}.csv", delimiter=',')
        cell_dir[f"sub-{sub}"] = np.genfromtxt(f"{source_folder}/s{sub}/all_cells_firing_rate_sub_{sub}.csv", delimiter=',')
    return location_dir, cell_dir
        