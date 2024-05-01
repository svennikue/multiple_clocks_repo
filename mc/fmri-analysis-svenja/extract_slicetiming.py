#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:42:17 2024
this is to try and extract slice time information and saves on 
slice timing and one slice order file in the pilor/sub/func folder.
@author: xpsy1114
"""

import nibabel as nib
import json
import numpy as np

# this is to create the slice timing file as fsl requires it:
# "If a slice timings file is to be used, put one value (ie for each slice) on each line of a text file. 
# The units are in TRs, with 0 corresponding to no shift. Therefore a sensible 
# range of values will be between -0.5 and 0.5.

def normalize_slice_timings(slice_timings):
    max_time = 1.078
    midpoint = max_time / 2
    normalized = []
    
    for timing in slice_timings:
        if timing == 0:
            # Keep the 0 values as is
            normalized.append(0)
        elif timing <= midpoint:
            # Normalize values to the range [0, 0.5]
            normalized_value = (timing / midpoint) * 0.5
            normalized.append(normalized_value)
        else:
            # Normalize values to the range [-0.5, 0)
            # Subtracting from 1 to invert the scale, then shifting to [-0.5, 0)
            normalized_value = -((max_time - timing) / midpoint) * 0.5
            normalized.append(normalized_value)

    return normalized

# second, to create a slice order file as pnm requires it.

def create_slice_order(slice_timings):
    # Sort the slice timings, but also keep track of original indices for duplicates
    sorted_timings = sorted(enumerate(slice_timings, start=1), key=lambda x: x[1])
    
    # Initialize variables to store the previous value and the current index
    prev_value = None
    current_index = 0
    slice_order = [0] * len(slice_timings)
    
    for original_index, timing in sorted_timings:
        # Check if this timing is the same as the previous one
        if timing != prev_value:
            current_index += 1  # Increment the index for a new, unique timing
            prev_value = timing
        
        # Assign the current index to the original position of this timing
        slice_order[original_index - 1] = current_index

    return slice_order



task_halves = ['1', '2']
slice_timings = {}
slice_timings_transformed = {}
slice_order = {}
for sub in range(1,35):
    if sub < 10:
        subject = f"sub-0{sub}"
    else:
        subject = f"sub-{sub}"
    pilot_path = f"/home/fs0/xpsy1114/scratch/data/pilot/{subject}/func"
    for task_half in task_halves:
        json_file = f"{pilot_path}/{subject}_{task_half}_bold.json"
        with open(json_file, 'r') as f:
            data = json.load(f)
        slice_timings[f"{subject}_{task_half}"] = data.get('SliceTiming', [])
        if not slice_timings[f"{subject}_{task_half}"]:
            raise ValueError(f"No Slice Timing info available for {subject} task half {task_half}")
            
        slice_timings_transformed[f"{subject}_{task_half}"] = normalize_slice_timings(slice_timings[f"{subject}_{task_half}"])

        slice_time_path = f"{pilot_path}/{subject}_slice_timings_normalised_{task_half}.txt"
        with open(slice_time_path, 'w') as f:
            for timing in slice_timings_transformed[f"{subject}_{task_half}"]:
                f.write(f"{timing}\n")
                
        slice_order[f"{subject}_{task_half}"] = create_slice_order(slice_timings[f"{subject}_{task_half}"])

        slice_order_path = f"{pilot_path}/{subject}_slice_order_{task_half}.txt"
        with open(slice_order_path, 'w') as f:
            for timing in slice_order[f"{subject}_{task_half}"]:
                f.write(f"{timing}\n")
        
        slice_timings_raw_path = f"{pilot_path}/{subject}_slice_timings_{task_half}.txt"
        with open(slice_timings_raw_path, 'w') as f:
            for timing in slice_timings[f"{subject}_{task_half}"]:
                f.write(f"{timing} ")
                
                
        