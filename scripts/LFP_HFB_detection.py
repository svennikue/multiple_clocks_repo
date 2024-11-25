#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:08:37 2024
This script is to extract high-frequency broadband events from contacts
NOT in hippocampus.
this is to analyse peri-ripple high-frequency broadband (HFB, 60–160 Hz) activity


"Next, we tested whether ripple-aligned cortical activity during rest is important 
for future 2D inferences on compounds. We focused on high-frequency broadband 
(HFB; 60–160 Hz, also known as high gamma; see Methods for details) cortical activity, 
which is a robust marker of local neuronal spiking (Mukamel et al., 2005; 
Parvizi & Kastner, 2018; Ray et al., 2008). "




@author: Svenja Kuchenhoff
"""

import mne
import neo
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.signal import resample
import pickle

import glob
import os
import bisect
import gc
import tracemalloc
import math
import mc

# tracemalloc.start()
gc.collect()
          
save = False
plotting_distr = False
referenced_data = False

if referenced_data == True:
    preproc_type = 'referenced'
else:
   preproc_type = 'channel_wise' 
wire_of_interest = None # 'LT1Ha' #None


# do everything that is NOT HPC
# SAFE ALL CHANNELS!!!!
# sort them by ROI
# which ones do I have???

# then, next step: check alignment to the actual 'clock neurons'


# if you want to show that there are more ripples in HPC than in mPFC
ROI = 'all' # 'mPFC' or 'all'

# if you want to collect all ripples within a single grid
index_lower = 0
index_upper = 9

# frequency of interest is high frequency broadband
HFB = [60, 160]
# 2.5 * HFB[1] -> downsampled to 400. 


# import pdb; pdb.set_trace() 
subjects = ['s25']

# subjects = ['s7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 
#             's15', 's18', 's25'] #16 doesnt have channel indices??


#subjects = ['s13', 's12', 's25']
LFP_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
result_dir = f"{LFP_dir}/results"


for sub in subjects :
    ns3_files = glob.glob(os.path.join(f"{LFP_dir}/{sub}/", '*.ns3'))
    # Initialize variables for the two files
    file_blk_01 = None
    file_blk_02 = None
    
    # Loop through files and find the ones with 'blk-01' and 'blk-02'
    for file in ns3_files:
        if 'blk-01' in file:
            file_blk_01 = os.path.splitext(os.path.basename(file))[0]  # Remove path and '.ns3'
        elif 'blk-02' in file:
            file_blk_02 = os.path.splitext(os.path.basename(file))[0]  # Remove path and '.ns3'
    
    # Create the final list in the specified order
    names_blks_short = [file_blk_01, file_blk_02]

    # load behaviour that defines my snippets.
    behaviour = np.genfromtxt(f"{LFP_dir}/{sub}/exploration_trials_times_and_ncorrect.csv", delimiter=',')
    behaviour_all = np.genfromtxt(f"{LFP_dir}/{sub}/all_trials_times.csv", delimiter=',')
    feedback = np.genfromtxt(f"{LFP_dir}/{sub}/feedback.csv", delimiter=',')
    
    # preparing behaviour for grid wise analysis
    index_lower = []
    index_upper = []
    # define seconds_lower[task] as a new repeat of a grid.
    # also collect grid_index (task_config) to keep track if you're still in the same grid.
    seconds_lower, seconds_upper, task_config, task_index, task_onset, new_grid_onset, found_first_D = mc.analyse.ripple_helpers.prep_behaviour(behaviour_all)

    # preparing the file
    raw_file_lazy, HC_channels, HC_indices, mPFC_channels, mPFC_indices, orig_sampling_freq, block_size, ROI_dict, all_ROI_list, all_ROI_indices = mc.analyse.ripple_helpers.load_LFPs(LFP_dir, sub, names_blks_short, channel_list_complete=True)
    # import pdb; pdb.set_trace() 
          
    if ROI == 'mPFC':
        channels_to_use_in_task = mPFC_channels
        channel_indices_to_use = mPFC_indices
    elif ROI == 'all':   
        channels_to_use_in_task = all_ROI_list
        channel_indices_to_use = all_ROI_indices
        
    #     # CONTINUE HERE
    #     # write something that basically says 'everything BUT the HC channels'
    #     # OR do a 'region wide' analysis like Yunzeh: he's looking at 
    #     # the alignment of HFB power (z-scored) with time to ripple peak.
    #     channels_to_use_in_task = HC_channels
    #     channel_indices_to_use = HC_indices

    
    gap_at = block_size[0]/orig_sampling_freq[0]
    skip_task_index = bisect.bisect_right(task_onset, gap_at)
    
    # then going into the loop and collecting the actual HFB power data per task.
    onset_in_secs_dict = {}
    events_dict_per_channel = {}
    
    freq_bands_keys = ['HFB']
    freq_bands = {freq_bands_keys[0]: (HFB[0], HFB[1])}
    
    # for task_to_check in range(1, 3):  
    for task_to_check in range(1, int(behaviour_all[-1,-1]+1)):  
        # first define where in behavioural table the task starts and ends
        index_lower = np.where(np.array(task_index)== task_to_check)[0][0]
        index_upper = np.where(np.array(task_index)== task_to_check)[0][-1]

        if sub not in 's5':
            if skip_task_index != len(task_index) and task_to_check in [task_index[skip_task_index]]: 
                continue
        if task_to_check in [10] and sub == 's25':
            continue
        if task_to_check in [24] and sub in ['s8', 's11', 's15']:
            continue
        if task_to_check in [14] and sub in ['s11']:
            continue
        if task_to_check in [23] and sub in ['s18']:
            continue
        
        events_dict = {}
        power_dict = {}
        channel_ripple_dict = {}
        onset_secs = []
        onset_secs_per_channel = {}
        
        # for each snippet of the dataset, now look for HFB events.
        for repeat, trial_index in enumerate(range(index_lower, index_upper+1)):
            if sub in ['s10'] and repeat == 9:
                continue
            #for task in range(7, 10):
            sec_lower = seconds_lower[trial_index]
            if sec_lower < 0:
                sec_lower = 0.1
            sec_upper = seconds_upper[trial_index]
            print(f"Now analysing sub {sub}, task {task_to_check}, repeat {repeat} between {sec_lower} and {sec_upper} secs")
            
            if sec_upper < block_size[0]/orig_sampling_freq[0]:
                block = 0
                sec_lower_neuro = sec_lower
                sec_upper_neuro = sec_upper
            else:
                block = 1
                sec_lower_neuro = sec_lower-(block_size[0]/orig_sampling_freq[0])
                sec_upper_neuro = sec_upper-(block_size[0]/orig_sampling_freq[0])
        
            # if sec_upper > block_size[0]/orig_sampling_freq[0]+block_size[1]/orig_sampling_freq[0]:
            #     print("careful, the behavioural file for {sub} seems to be longer than the LFP files! Skipping rep {repeat}, trial {trial_index}")
            #     continue
            
            reader, raw_file_lazy = [], []
            if sub not in ['s5']:
                for file_half in [0,1]:
                    # does neo.io have an 'unload' function?
                    reader.append(neo.io.BlackrockIO(filename=f"{LFP_dir}/{sub}/{names_blks_short[file_half]}", nsx_to_load=3))
                    if (sub in ['s11'] and file_half == 0) or (sub in ['s16', 's18'] and file_half == 1):
                        raw_file_lazy.append(reader[file_half].read_segment(seg_index=0, lazy=True))
                    else:
                        raw_file_lazy.append(reader[file_half].read_segment(seg_index=1, lazy=True))
            else:
                for file_half in [0]:
                    # does neo.io have an 'unload' function?
                    reader.append(neo.io.BlackrockIO(filename=f"{LFP_dir}/{sub}/{names_blks_short[file_half]}", nsx_to_load=3))
                    raw_file_lazy.append(reader[file_half].read_segment(seg_index=1, lazy=True))

            # redefine the lazy loader with every loop and see if that decreases memory load!!
            raw_analog_cropped = raw_file_lazy[block].analogsignals[0].load(time_slice = (sec_lower_neuro, sec_upper_neuro), channel_indexes = channel_indices_to_use)
            
            # Target downsampled frequency
            downsampled_sampling_rate = 2 * HFB[1]
            # Calculate the number of samples in the downsampled data
            num_samples = int(raw_analog_cropped.shape[0] * (downsampled_sampling_rate / orig_sampling_freq[0]))
            # Downsample the data and delete the big one
            downsampled_data = resample(raw_analog_cropped.magnitude, num_samples, axis=0)
            if len(downsampled_data) < 8*HFB[1]:
                print(f"Skipping task {task_to_check} repeat {repeat}. too short. only {len(downsampled_data)} samples.")
                continue
            
            del raw_analog_cropped
            
            if referenced_data == True:
                #referenced_data, new_channels = mc.analyse.ripple_helpers.reference_electrodes(downsampled_data, channels_to_use)
                downsampled_data, channels_to_use = mc.analyse.ripple_helpers.reference_electrodes(downsampled_data, channels_to_use_in_task, repeat)
            else:
                channels_to_use = channels_to_use_in_task

            #raw_analog_epo_cropped = raw_analog_cropped.T.reshape(1,raw_analog_cropped.shape[1], raw_analog_cropped.shape[0])
            downsampled_analog_epo_cropped = downsampled_data.T.reshape(1,downsampled_data.shape[1], downsampled_data.shape[0])
        
            power_mean = {}
            power_stepwise = {}
            for band, (l_freq, h_freq) in freq_bands.items():
                step = np.max([1, (h_freq - l_freq) / 20])
                freq_list = np.arange(l_freq, h_freq, step)
                # l_power = mne.time_frequency.tfr_array_morlet(raw_analog_epo_cropped, sampling_freq[block], freqs=freq_list, output="power", n_jobs=-1).squeeze()
                l_power = mne.time_frequency.tfr_array_morlet(downsampled_analog_epo_cropped, downsampled_sampling_rate, freqs=freq_list, output="power", n_jobs = 1).squeeze()
                for idx_freq in range(len(freq_list)):
                    for channel_idx in range(len(channels_to_use)):
                        l_power[channel_idx,idx_freq,:] = scipy.stats.zscore(l_power[channel_idx,idx_freq,:], axis=None)
                power_mean[band] = np.mean(l_power, axis=1)
                power_stepwise[band] = l_power
            
            power_dict[f"{repeat}_mean"] = power_mean
            power_dict[f"{repeat}_stepwise"] = power_stepwise
            
            
            
            # import pdb; pdb.set_trace() 
            # MAYBE storing this is ACTUALLY ENOUGH????
            # Maybe i prefer the continous power over the 'individual' events?
            # it's not supposed to be not-continuous anyways
            # OK YEAH THIS SHOULD BE FINE!
            # but i need to store what timings this actually covers.
            
            # x = np.linspace(0, len(power_dict['0_mean']['HFB'].T), len(power_dict['0_mean']['HFB'].T))
            # plt.figure(); plt.plot(x, power_dict['0_mean']['HFB'][1])
            
            
            # IF i want individual events, do this:
            # Collect all possible events for the current task
            
            
            
            
            length_events_in_secs = 0.02
            min_length_event = math.ceil(length_events_in_secs*downsampled_sampling_rate)
            event_id = {}
            all_clusters = np.zeros((len(channels_to_use), downsampled_analog_epo_cropped.shape[-1]))
            # all_clusters = np.zeros((len(channel_indices_to_use), raw_analog_epo_cropped.shape[-1]))
            onsets, durations, descriptions, task_config_ripple = [], [], [], []
            for new_channel_idx, channel_name in enumerate(channels_to_use):
                cluster_one_zero = np.zeros((len(power_mean.keys()), downsampled_analog_epo_cropped.shape[-1]))
                #cluster_one_zero = np.zeros((len(power_mean.keys()), raw_analog_epo_cropped.shape[-1]))
                # print(f"now looking at channel {channel_list[initial_channel_idx]}")
                for iband, band in enumerate(power_mean.keys()):
                    # set this to exceeding 4x standard deviation from power in this band 
                    threshold_hl = np.mean(power_mean[band][new_channel_idx,:]) + 4*np.std(power_mean[band][new_channel_idx,:])
                    cond = power_mean[band][new_channel_idx,:] > threshold_hl
                    # cond is a boolean array which is true in case the power is higher than the threshold
                    # each time there are more than 1 'trues' next to each other, this will be called a cluster.
                    # the first cluster will be marked by 1s, the second one by 2s, the third one by 3s,.. etc.
                    clusters, n_clusters = ndimage.label(cond)
                    for cli in range(1,n_clusters+1): #+ 1 bc it starts counting at 0 but we want start at 1
                        # import pdb; pdb.set_trace() 
                        # include the gap of at least 15 ms.
                        # check how long each cluster is. Cli is equal to the number with which the current cluster is marked
                        cl = np.where(clusters == cli)[0]
                        # length of cl is also the length of the cluster (in samples 1/freq * len = secs)
                        # according to paper, clusters need to be 15 ms or more -> 30
                        # Yunzeh: 20 ms to 200 ms, with a 30 ms interval between events
                        if len(cl) >= min_length_event:
                            # include the 15ms gap!!
                            #  check for gap before cluster
                            if cl[0] >= min_length_event:
                                gap_before = np.all(cond[cl[0] - min_length_event:cl[0]] == 0)
                            else:
                                gap_before = False
                            # check for gap after cluster
                            if len(cond) - cl[-1] - 1 >= min_length_event:
                                gap_after = np.all(cond[cl[-1] + 1:cl[-1] + 1 + min_length_event] == 0)
                            else:
                                gap_after = False
                            # Only consider clusters with sufficient gaps on both sides
                            if gap_before and gap_after:
                                cluster_one_zero[iband, cl] = 1
                            
                # import pdb; pdb.set_trace() 

                # save all clusters uniquely per channel.
                all_clusters[new_channel_idx,:] = (1+new_channel_idx)*clusters 
                for cli in range(1, n_clusters+1):
                    # again check how long each cluster is. Cli is equal to the number of how the current cluster is marked
                    cl = np.where(clusters == cli)[0]
                    onsets.append(cl[0]) #onset is in samples, not seconds.
                    # onsets.append(cl[0] + sec_lower*sampling_freq[0]) #onset is in samples, not seconds.
                    durations.append(len(cl)) #duration is also samples, not seconds.
                    descriptions.append(f"channel_{channel_name}")
                    task_config_ripple.append(task_config[trial_index])
                    if f"channel_{channel_name}" not in event_id:
                        event_id[f"channel_{channel_name}"] = new_channel_idx
                    # onset_secs.append(cl[0]/sampling_freq[block] + sec_lower)
                    onset_secs.append(cl[0]/downsampled_sampling_rate + sec_lower)
                    if channel_name not in onset_secs_per_channel:
                        onset_secs_per_channel[channel_name] = []
                    # onset_secs_per_channel[channel_list[initial_channel_idx]].append(cl[0]/sampling_freq[block] + sec_lower)
                    onset_secs_per_channel[channel_name].append(cl[0]/downsampled_sampling_rate + sec_lower)

            # I NEED A neo.io file for these raw_cropped
            ch_types = ["ecog"] * len(channels_to_use)
            info = mne.create_info(ch_names=channels_to_use, ch_types=ch_types, sfreq=downsampled_sampling_rate)
            # info = mne.create_info(ch_names=channels_to_use, ch_types=ch_types, sfreq=sampling_freq[0])
            # raw_np_cropped = raw_analog_cropped.as_array()
            # memory management
            # del raw_analog_cropped, raw_analog_epo_cropped
            raw_cropped = mne.io.RawArray(downsampled_data.T, info) 
            # raw_cropped = mne.io.RawArray(raw_np_cropped.T, info) 
            
            # import pdb; pdb.set_trace()
            # create a mne.io object
            ch_types = ["sEEG"] * len(channels_to_use)
            annot = mne.Annotations(onset=[x/downsampled_sampling_rate for x in onsets], duration=[x/downsampled_sampling_rate for x in durations], description=descriptions)
            # annot = mne.Annotations(onset=[x/sampling_freq[0] for x in onsets], duration=[x/sampling_freq[0] for x in durations], description=descriptions)
            
            raw_cropped.set_annotations(annot)
            # add events
            events, event_id = mne.events_from_annotations(raw_cropped, event_id=event_id)
            # events are times at which something happens, e.g. a ripple occurs
            events_dict[repeat] = events


        events_dict_per_channel[task_to_check] = events_dict
        onset_in_secs_dict[task_to_check] = onset_secs_per_channel
              

    if ROI == 'all':
        with open(f"{result_dir}/{sub}_all_channels_{preproc_type}_HFB_by_seconds.pkl", 'wb') as file:
            pickle.dump(onset_in_secs_dict, file)
                
        with open(f"{result_dir}/{sub}_ROI_dict.pkl", 'wb') as file:
            pickle.dump(ROI_dict, file)
                    

    with open(f"{result_dir}/{sub}_{ROI}_{preproc_type}_HFB_events_dir.pkl", 'wb') as file:
        pickle.dump(events_dict, file)
        
    with open(f"{result_dir}/{sub}_{ROI}_{preproc_type}_HFB_by_seconds.pkl", 'wb') as file:
        pickle.dump(onset_in_secs_dict, file)

    # np.save(f"{result_dir}/{sub}_beh_feedback", feedback)

        
