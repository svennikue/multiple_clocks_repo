#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:27:28 2023
This script plots the physiology file that I collected for the first participant.
@author: xpsy1114
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import os
import sys


if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '26'

subjects = [f"sub-{subj_no}"]
load_old = False
visualise_RDMs = False

#subjects = ['sub-01']
task_halves = ['1', '2']


for sub in subjects:
    data_dir = f"/home/fs0/xpsy1114/scratch/data/pilot/{sub}/motion"
    output_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}/motion"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    physio_file = f"{data_dir}/{sub}_physio.txt"
    df = pd.read_csv(physio_file, sep ="\t")
    
    # Identify the first trigger: bigger than 4 in column 3 
    trigger_indexes = df.index[df.iloc[:, 3] > 4].tolist()

    # Step 2: Calculate gap sizes between TRs
    gap_sizes = np.diff(trigger_indexes)

    # Identifying the long gap = this is where the second task half starts
    long_gap_index = np.where(gap_sizes > 250)[0]

    # Assuming the first long gap clearly separates the two halves
    first_half_end_index = trigger_indexes[long_gap_index[0]]
    second_half_start_index = trigger_indexes[long_gap_index[0] + 1]

    # Step 4: Split the dataset into two halves
    first_half = df.iloc[:first_half_end_index + 1, :]
    second_half = df.iloc[second_half_start_index:, :]

    first_half.to_csv(f"{output_dir}/{sub}_physio_01.txt", sep='\t', index=False)
    second_half.to_csv(f"{output_dir}/{sub}_physio_02.txt", sep='\t', index=False)
    
    print(f"Done with subject {sub}!")