#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:08:37 2024
First steps looking for ripples.

Recycling stuff from Matthias 
Trying to plot filtered out events.

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
          
save = True
plotting_distr = False
plotting_ripples = False
referenced_data = False

if referenced_data == True:
    preproc_type = 'referenced'
else:
   preproc_type = 'channel_wise' 
wire_of_interest = None # 'LT1Ha' #None


# if you want to show that there are more ripples in HPC than in mPFC
ROI = 'HPC' # HPC mPFC

# if you want to collect all ripples within a single grid
analysis_type = 'grid_wise' # grid_wise, exploration_trials 

index_lower = 0
index_upper = 9

theta = [3,8]
middle = [10, 80]
gamma = [80, 180]
ultra_high_gamma = [180, 250]
# 2.5 * gamma[1] -> downsampled to 450. Yunzeh downsamples to 1000.
# -> check if this still gives you the same ripples!

# Yunzeh
# next: 'notch-filtering' for 50Hz power line interference (double check, but I don't think I have that )
# plus for its harmonics (100 Hz, 150 Hz etc) using a 3-Hz wide filter 
# Electrode contacts positioned within 3 mm of the hippocampus were selected, and a nearby white-matter contact's reference signal was subtracted to reduce common noise. 
# bandpass filtered between 70-180Hz 



# SPW-R frequency band criterion for rodents (100 to 250 Hz) is generally higher 
# than for monkeys (95 to 250 Hz) or humans (70–250 Hz, most use 80–150 Hz bandpass filters

# delta 0-4 Hz
# theta 4 - 10 Hz
# alpha 8 - 12 Hz
# beta 15 - 30 Hz
# low gamma 30 - 90 Hz
# high gamma 90 - 150 Hz
# (3, 8), freq_bands_keys[1]: (10,80), freq_bands_keys[2]: (105, 130)}
# define your frequencies

# import pdb; pdb.set_trace() 
# subjects = ['s5']

# subjects = ['s5', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 
#             's15', 's18', 's25'] #something weird in 16

# 's27', 's28'
# s27 might have had an error while recording, so the behavioural file is bad.

# s5 and s26 only have one task half.


# make s15, s16, s18 work. 
subjects = ['s15', 's16', 's18']
# subjects = ['s5'] #WHY DOES 18 NOT WORK????


# subjects = ['s10']
# subjects = ['s11', 's12', 's13', 's14', 's25']

# check what is wrong with s11 and also fix that s5 works- only a single run.
# subjects = ['s5']

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
    
    # preparing behaviour for exploration trials
    if analysis_type == 'exploration_trials':
        seconds_lower, seconds_upper, task_config  = [], [], []
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
    
    # preparing behaviour for grid wise
    elif analysis_type == 'grid_wise':
        index_lower = []
        index_upper = []
        # define seconds_lower[task] as a new repeat of a grid.
        # also collect grid_index (task_config) to keep track if you're still in the same grid.
        seconds_lower, seconds_upper, task_config, task_index, task_onset, new_grid_onset, found_first_D = mc.analyse.ripple_helpers.prep_behaviour(behaviour_all)

    
    # preparing the file
    raw_file_lazy, HC_channels, HC_indices, mPFC_channels, mPFC_indices, orig_sampling_freq, block_size, ROI_dict, ROI_list, ROI_indices = mc.analyse.ripple_helpers.load_LFPs(LFP_dir, sub, names_blks_short, channel_list_complete=False)
         
    if ROI == 'mPFC':
        channels_to_use_in_task = mPFC_channels
        channel_indices_to_use = mPFC_indices
    elif ROI == 'HPC':
        channels_to_use_in_task = HC_channels
        channel_indices_to_use = HC_indices
    
    gap_at = block_size[0]/orig_sampling_freq[0]
    skip_task_index = bisect.bisect_right(task_onset, gap_at)
    
    # then going into the loop and collecting the actual ripple data per task.
    onset_in_secs_dict = {}
    events_dict_per_channel = {}
    feedback_dict = {}
    
    freq_bands_keys = ['theta', 'middle', 'hgamma', 'ultra_high_gamma']
    freq_bands = {freq_bands_keys[0]: (theta[0], theta[1]), freq_bands_keys[1]: (middle[0],middle[1]), freq_bands_keys[2]: (gamma[0], gamma[1]), freq_bands_keys[3]: (ultra_high_gamma[0], ultra_high_gamma[1])}
    
    # for task_to_check in range(1, 3):  
    for task_to_check in range(1, int(behaviour_all[-1,-1]+1)):  
        
        # for task_to_check in range(12,17): 
        # first define where in behavioural table the task starts and ends
        index_lower = np.where(np.array(task_index)== task_to_check)[0][0]
        index_upper = np.where(np.array(task_index)== task_to_check)[0][-1]

        if sub not in ['s5', 's26']:
            if skip_task_index != len(task_index) and task_to_check in [task_index[skip_task_index]]: 
                continue
        if task_to_check in [10] and sub == 's25':
            continue
        if task_to_check in [24] and sub in ['s8', 's11', 's15', 's26']:
            continue
        if task_to_check in [14] and sub in ['s11']:
            continue
        # big_analog_chunk = raw_file_lazy[0].analogsignals[0].load(time_slice = (0.01, seconds_upper[3]), channel_indexes = HC_indices)
        # ch_types = ["ecog"] * len(HC_indices)
        # info = mne.create_info(ch_names=HC_channels, ch_types=ch_types, sfreq=sampling_freq[0])
        # big_chunk_np = big_analog_chunk.as_array()
        # big_chunk_mne = mne.io.RawArray(big_chunk_np.T, info)
        # spectrum = big_chunk_mne.compute_psd()
        # spectrum.plot()
        
        # # clear them after 
        # big_chunk_mne = []
        # big_analog_chunk = []
        # import pdb; pdb.set_trace()
        
        feedback_dict[f"{task_to_check}_correct"], feedback_dict[f"{task_to_check}_error"] = [], []
        for i in range(len(feedback)):
            if feedback[i,0] == task_to_check:
                if feedback[i,1] > 0:
                    feedback_dict[f"{task_to_check}_correct"].append(feedback[i,1])
                if feedback[i,2] > 0:
                    feedback_dict[f"{task_to_check}_error"].append(feedback[i,2])
                    
        
        
        events_dict = {}
        power_dict = {}
        channel_ripple_dict = {}
        onset_secs = []
        onset_secs_per_channel = {}
        
        # for each snippet of the dataset, now look for ripples.
        # for task in range(0, len(seconds_lower[0:4])):
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
        
            if sub not in ['s5', 's26']:
                # because of only having one block, this would create an error.
                if sec_upper > block_size[0]/orig_sampling_freq[0]+block_size[1]/orig_sampling_freq[0]:
                    print("careful, the behavioural file for {sub} seems to be longer than the LFP files! Skipping rep {repeat}, trial {trial_index}")
                    continue
            
            reader, raw_file_lazy = [], []
            if sub not in ['s5', 's26']:
                for file_half in [0,1]:
                    # does neo.io have an 'unload' function?
                    reader.append(neo.io.BlackrockIO(filename=f"{LFP_dir}/{sub}/{names_blks_short[file_half]}", nsx_to_load=3))
                    if (sub in ['s11'] and file_half == 0) or (sub in ['s18'] and file_half == 1):
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
            downsampled_sampling_rate = 2 * ultra_high_gamma[1]
            # Calculate the number of samples in the downsampled data
            num_samples = int(raw_analog_cropped.shape[0] * (downsampled_sampling_rate / orig_sampling_freq[0]))
            # Downsample the data and delete the big one
            downsampled_data = resample(raw_analog_cropped.magnitude, num_samples, axis=0)
            if len(downsampled_data) < 8*ultra_high_gamma[1]:
                print(f"Skipping task {task_to_check} repeat {repeat}. too short. only {len(downsampled_data)} samples.")
                continue
            
            del raw_analog_cropped
            
            if referenced_data == True:
                #referenced_data, new_channels = mc.analyse.ripple_helpers.reference_electrodes(downsampled_data, channels_to_use)
                downsampled_data, channels_to_use = mc.analyse.ripple_helpers.reference_electrodes(downsampled_data, channels_to_use_in_task, repeat)
            else:
                channels_to_use = channels_to_use_in_task
            
            # import pdb; pdb.set_trace()
            
            
            
            # Update the metadata if necessary, including the new sampling rate
            # import pdb; pdb.set_trace() 
            
            # IF I LEAVE THIS CHANGE THE NAMING!!!
            # GO THROUGH CODE AND CHANGE THE SAMPLING FREQUCNY EVERYWHERE!!
 
            #raw_analog_epo_cropped = raw_analog_cropped.T.reshape(1,raw_analog_cropped.shape[1], raw_analog_cropped.shape[0])
            downsampled_analog_epo_cropped = downsampled_data.T.reshape(1,downsampled_data.shape[1], downsampled_data.shape[0])
            
            # import pdb; pdb.set_trace() 
            
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
            
            # Collect all possible ripples for the current task
            # import pdb; pdb.set_trace() 
            length_ripple_in_secs = 0.02
            min_length_ripple = math.ceil(length_ripple_in_secs*downsampled_sampling_rate)
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
                    # I think this is redundant. compare against threshold if not middle, but middle also against threshold??? doesnt make much sense.
                    #cond = power_mean[band][new_channel_idx,:] > (threshold if band != "middle" else threshold)
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
                        if len(cl) >= min_length_ripple:
                            # include the 15ms gap!!
                            #  check for gap before cluster
                            if cl[0] >= min_length_ripple:
                                gap_before = np.all(cond[cl[0] - min_length_ripple:cl[0]] == 0)
                            else:
                                gap_before = False
                            # check for gap after cluster
                            if len(cond) - cl[-1] - 1 >= min_length_ripple:
                                gap_after = np.all(cond[cl[-1] + 1:cl[-1] + 1 + min_length_ripple] == 0)
                            else:
                                gap_after = False
                            # Only consider clusters with sufficient gaps on both sides
                            if gap_before and gap_after:
                                cluster_one_zero[iband, cl] = 1
                            
                            
                # Keep events with power in both low theta and vhgamm, but NOT in the middle to avoid broadband
                # this is a boolean array that is true for a cluster and false for any other sample
                # ndimage.label will returen clusters as an integer ndarray where each unique feature in input has a unique label 
                # and n_clusters as How many objects were found.
                
                # what if I ignore the middle band?
                # clusters, n_clusters = ndimage.label((cluster_one_zero[0,:] + cluster_one_zero[2,:] ) == 2)
                
                # clusters, n_clusters = ndimage.label((cluster_one_zero[0,:] + cluster_one_zero[2,:] + (1 - cluster_one_zero[1,:])) == 3)
        
                # ignore gamma
                clusters, n_clusters = ndimage.label((cluster_one_zero[2,:] + (1 - cluster_one_zero[1,:])) == 2)
        
                # now again, I save the amount of clusters; this time only if it fulfills both criteria: threshold + no middle.
                # print(f"I found {n_clusters} between {sec_lower} and {seconds_upper[repeat]} secs in channel {channel_list[initial_channel_idx]}")
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

                # channel_ripple_dict[channel_list[initial_channel_idx]] = onset_secs
        
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


            if plotting_ripples == True:
                # import pdb; pdb.set_trace()
                if wire_of_interest:
                    indices_of_interest = []
                    for i, channel in enumerate(channels_to_use):
                        if wire_of_interest in channel :
                            indices_of_interest.append(i)

                y_label_power = [f"{theta[1]} Hz", f"{middle[1]} Hz", f"{gamma[1]} Hz", f"{ultra_high_gamma[1]} Hz"]  # Custom labels
                filtered_cropped_vhgamma = raw_cropped.filter(l_freq=gamma[0], h_freq=gamma[1], picks='all', fir_design='firwin')
                filtered_cropped_vhgamma_np = filtered_cropped_vhgamma.get_data()
                for ie, event in enumerate(events_dict[repeat]):
                    # freq_to_plot = int(downsampled_sampling_rate/2)
                    # freq_to_plot = 100
                    freq_to_plot = 500
                    # freq_to_plot = int(sampling_freq[0]/2)
                    title = f"1 sec window around ripple in subj {sub}, {channels_to_use[event[-1]]} - onset {event[0]/downsampled_sampling_rate} sec; [{downsampled_sampling_rate} samples = 1 sec]"
                    # don't plot more than 5 ripples
                    if wire_of_interest:
                        if event[0] > freq_to_plot and len(downsampled_data)-event[0]>freq_to_plot and event[-1] in indices_of_interest and ie < 5:
                            mc.analyse.plotting_ripples.plot_ripple(freq_to_plot, title, downsampled_data, event, min_length_ripple, filtered_cropped_vhgamma_np, power_dict, repeat, freq_bands_keys, y_label_power)
                    else:
                        if event[0] > freq_to_plot and len(downsampled_data)-event[0]>freq_to_plot and ie < 5:
                            mc.analyse.plotting_ripples.plot_ripple(freq_to_plot, title, downsampled_data, event, min_length_ripple, filtered_cropped_vhgamma_np, power_dict, repeat, freq_bands_keys, y_label_power)
                    
                    # for publication 
                    #if event[0] > freq_to_plot and len(downsampled_data)-event[0]>freq_to_plot and ie < 5:
                     #   mc.analyse.plotting_ripples.plot_ripple(freq_to_plot, title, downsampled_data, event, min_length_ripple, filtered_cropped_vhgamma_np, power_dict, repeat, freq_bands_keys, y_label_power, for_publication=True)
                
                            
        events_dict_per_channel[task_to_check] = events_dict
        onset_in_secs_dict[task_to_check] = onset_secs_per_channel
        
        if save == True:
            raw_cropped.save(f"{LFP_dir}/{sub}/{names_blks_short[0]}-{analysis_type}_{sec_lower}-{sec_upper}-raw.fif", overwrite=True)
        if plotting_distr:
            mc.analyse.ripple_helpers.plot_ripple_distribution(onset_in_secs_dict, task_to_check, feedback_dict[f"{task_to_check}_error"], feedback_dict[f"{task_to_check}_correct"], onset_secs, found_first_D, seconds_upper, index_upper, index_lower, seconds_lower, sub)



        # memory management
        # del raw_cropped
        # del raw_cropped, raw_np_cropped
        # import pdb; pdb.set_trace()
        # Check memory allocation
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')
        
        # print("[ Top 10 memory usage ]")
        # for stat in top_stats[:10]:
        #     print(stat)

    # channels_to_use = ['RT2bHaEa01-027','RT2bHaEa02-028', 'RT2bHaEa03-029', 'RT2bHaEa04-030', 'RT2bHaEa05-031','RT2bHaEa06-032', 'RT2bHaEa07-129', 'RT2bHaEa08-130','RT2bHaEa09-131']
    # mc.analyse.plotting_ripples.plot_ripples_per_channel(onset_in_secs_dict, channels_to_use, sub)
                       

    with open(f"{result_dir}/{sub}_{ROI}_{analysis_type}_{preproc_type}_ripple_events_dir.pkl", 'wb') as file:
        pickle.dump(events_dict, file)
    with open(f"{result_dir}/{sub}_{ROI}_{analysis_type}_{preproc_type}_ripple_by_seconds.pkl", 'wb') as file:
        pickle.dump(onset_in_secs_dict, file)
    with open(f"{result_dir}/{sub}_{ROI}_{analysis_type}_feedback.pkl", 'wb') as file:
        pickle.dump(feedback_dict, file)

    np.save(f"{result_dir}/{sub}_beh_feedback", feedback)

        
