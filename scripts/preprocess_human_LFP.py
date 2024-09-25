#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:21:48 2024

@author: Svenja Kuchenhoff

This file is supposed to preprocess the LFP data such that it can be more easily
loaded and used for ripple detection in a second step.

"""


import mne
import neo
# import neo.rawio
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import mmap

# import pdb; pdb.set_trace()
names_blks_short = ['EMU-117_subj-YEU_task-ABCD_run-01_blk-01_NSP-1','EMU-118_subj-YEU_task-ABCD_run-01_blk-02_NSP-1']
LFP_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
result_dir = f"{LFP_dir}/results"

# subjects = ['s13', 's12', 's25']
sub = 's25'

# load behaviour that defines my snippets.
behaviour = np.genfromtxt(f"{LFP_dir}/{sub}/exploration_trials_times_and_ncorrect.csv", delimiter=',')
seconds_lower, seconds_upper, task_config = [], [], []
if behaviour[0, 4] == 0:
    seconds_lower.append(behaviour[0, 0])
# end of the first repeat as the lower end, and keep the upper end defined as doing it correctly for the firs time
for i in range(1, len(behaviour)):
    if behaviour[i, 5] == 1:
        # start looking for ripples once they found goal D for the first time.
        seconds_lower.append(behaviour[i, 3])
        # also store the configuration of that task.
        task_config.append([behaviour[i, -4], behaviour[i, -3],behaviour[i, -2],behaviour[i, -1]])
    # transition from correct (1, last) to incorrect (0, now) means i-1 was last correct one.
    if behaviour[i-1, 4] == 1 and behaviour[i, 4] == 0:
        # end looking for ripples once they completed the task correctly at least once.
        seconds_upper.append(behaviour[i-1, 3])  
seconds_upper.append(behaviour[-1, 3])


# instead of fully loading the files, I am only loading the reader and then 
# looking at them in lazy-mode, only calling the shorter segments.
reader, block_size, channel_list, sampling_freq = [], [], [], [], []
for file_half in [0,1]:
    reader.append(neo.io.BlackrockIO(filename=f"{LFP_dir}/{sub}/{names_blks_short[file_half]}", nsx_to_load=3))
    block_size.append(reader[file_half].get_signal_size(seg_index=1, block_index=0))
    sampling_freq.append(reader[file_half].sig_sampling_rates[3])
    
    # all of these will only be based on the second file. Should be equivalent!
    channel_names = reader[file_half].header['signal_channels']
    channel_names = [str(elem) for elem in channel_names[:]]
    channel_list = [name.split(",")[0].strip("('") for name in channel_names]
    HC_indices = []
    mPFC_indices = []
    for i, channel in enumerate(channel_list):
        if 'Ha' in channel or 'Hb' in channel:
            HC_indices.append(i)
        if 'CA' in channel:
            mPFC_indices.append(i)    
    HC_channels = [channel_list[i] for i in HC_indices]
    mPFC_channels = [channel_list[i] for i in mPFC_indices]
    
    
if sampling_freq[0] != sampling_freq[1]:
    print('Careful! the files dont have the same sampling frequency! Probably wrong filename.')
    import pdb; pdb.set_trace()
        




if os.path.exists(f"{LFP_dir}/{sub}/{sub}_raw_ns3_blck1-blck2.npy"):
    raw_np_mmap = np.load(f"{LFP_dir}/{sub}/{sub}_raw_ns3_blck1-blck2.npy", mmap_mode='r')
    channel_list = np.load(f"{LFP_dir}/{sub}/{sub}_channel_list_ns3_blck1-blck2.npy")
    HC_indices = np.load(f"{LFP_dir}/{sub}/{sub}_HC_indices_ns3_blck1-blck2.npy")
    sampling_freq = int(np.load(f"{LFP_dir}/{sub}/{sub}_frequency_ns3_blck1-blck2.npy"))
        

            raw_np = reader.get_analogsignal_chunk(channel_indexes=HC_indices, seg_index=1)
            # careful! if this is less than 60*30*2.000 datapoints, it's likely the reference file
        if block > 0:
            raw_np_two = reader.get_analogsignal_chunk(channel_indexes=HC_indices, seg_index=1)
            raw_np = np.concatenate((raw_np,raw_np_two), axis = 0)
            # empty the heavy files
            raw_np_two = []
            raw_np_one = []
    
    # format is epochs x channels x samples 
    np.save(f"{LFP_dir}/{sub}/{sub}_raw_ns3_blck1-blck2.npy", raw_np)
    np.save(f"{LFP_dir}/{sub}/{sub}_channel_list_ns3_blck1-blck2.npy", channel_list)
    np.save(f"{LFP_dir}/{sub}/{sub}_HC_indices_ns3_blck1-blck2.npy", HC_indices)
    np.save(f"{LFP_dir}/{sub}/{sub}_frequency_ns3_blck1-blck2.npy", sampling_freq)
    raw_np_mmap = np.load(f"{LFP_dir}/{sub}/{sub}_ns3_blck1-blck2.npy", mmap_mode='r')
    # empty the heavy files
    raw_np_epo = []
    raw_np = []