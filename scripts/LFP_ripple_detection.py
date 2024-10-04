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
import pickle
import seaborn as sns
import glob
import os
import bisect
            
            
            
save = True
plotting_distr = True
plotting_ripples = False

# if you want to show that there are more ripples in HPC than in mPFC
ROI = 'HPC' # HPC mPFC

# if you want to collect all ripples within a single grid
analysis_type = 'grid_wise' # grid_wise, exploration_trials 

index_lower = 0
index_upper = 9

theta = [3,8]
middle = [10, 80]
gamma = [80, 180]

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

subjects = ['s12']
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
        seconds_lower, seconds_upper, task_config, task_index, task_onset, new_grid_onset, found_first_D  = [], [], [], [], [], [], []
        for i in range(1, len(behaviour_all)):
            if i == 1: 
                new_grid_onset.append(behaviour_all[i-1, 10])
                seconds_lower.append(behaviour_all[i-1, 10])
                task_config.append([behaviour_all[i-1, 5], behaviour_all[i-1, 6],behaviour_all[i-1, 7],behaviour_all[i-1, 8]])
                task_index.append(behaviour_all[i-1,-1])
                task_onset.append(behaviour_all[i-1, 10])
                found_first_D.append(behaviour_all[i-1, 4])
            curr_repeat = behaviour_all[i, 0]
            last_repeat = behaviour_all[i-1, 0]
            if curr_repeat != last_repeat:
                seconds_lower.append(behaviour_all[i, 10])
                seconds_upper.append(behaviour_all[i-1, 4])
                task_config.append([behaviour_all[i, 5], behaviour_all[i, 6],behaviour_all[i, 7],behaviour_all[i, 8]])
                task_index.append(behaviour_all[i,-1])
                task_onset.append(behaviour_all[i, 10])
            if behaviour_all[i, 9] < behaviour_all[i-1, 9]: # 9 is repeats in current grid
                # i.e. if in a new grid
                new_grid_onset.append(behaviour_all[i, 10])
                found_first_D.append(behaviour_all[i, 4])
        seconds_upper.append(behaviour_all[i, 4])       
    
    # preparing the file
    # instead of fully loading the files, I am only loading the reader and then 
    # looking at them in lazy-mode, only calling the shorter segments.
    reader, block_size, channel_list, sampling_freq, raw_file_lazy = [], [], [], [], []
    for file_half in [0,1]:
        # does neo.io have an 'unload' function?
        reader.append(neo.io.BlackrockIO(filename=f"{LFP_dir}/{sub}/{names_blks_short[file_half]}", nsx_to_load=3))
        block_size.append(reader[file_half].get_signal_size(seg_index=1, block_index=0))
        sampling_freq.append(int(reader[file_half].sig_sampling_rates[3]))        
        # all of these will only be based on the second file. Should be equivalent!
        channel_names = reader[file_half].header['signal_channels']
        channel_names = [str(elem) for elem in channel_names[:]]
        channel_list = [name.split(",")[0].strip("('") for name in channel_names]
        HC_indices = []
        mPFC_indices = []
        for i, channel in enumerate(channel_list):
            if 'Ha' in channel or 'Hb' in channel:
                HC_indices.append(i)
            if 'Ca' in channel:
                mPFC_indices.append(i)    
        HC_channels = [channel_list[i] for i in HC_indices]
        mPFC_channels = [channel_list[i] for i in mPFC_indices]
        raw_file_lazy.append(reader[file_half].read_segment(seg_index=1, lazy=True))
    
    if sampling_freq[0] != sampling_freq[1]:
        print('Careful! the files dont have the same sampling frequency! Probably wrong filename.')
        import pdb; pdb.set_trace()
            
    if ROI == 'mPFC':
        channels_to_use = mPFC_channels
        channel_indices_to_use = mPFC_indices
    elif ROI == 'HPC':
        channels_to_use = HC_channels
        channel_indices_to_use = HC_indices
    
    gap_at = block_size[0]/sampling_freq[0]
    skip_task_index = bisect.bisect_right(task_onset, gap_at)
    
    # then going into the loop and collecting the actual ripple data per task.
    onset_in_secs_dict = {}
    events_dict_per_channel = {}
    
    for task_to_check in range(1, int(behaviour_all[-1,-1]+1)):  
    # for task_to_check in range(1,2):
        for i, task in enumerate(task_index):
            if task == task_to_check and index_lower == []:
                index_lower = i
            if task_index[i-1] == task_to_check and task_index[i] > task_to_check:
                index_upper = i
            if i == len(task_index)-1:
                index_upper = i
        
        import pdb; pdb.set_trace()            
        
        if task_to_check in [skip_task_index]: 
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
        
        # first look at the dataset in general.
        

        # for each snippet of the dataset, now look for ripples.
        freq_bands_keys = ['theta', 'middle', 'vhgamma']
        freq_bands = {freq_bands_keys[0]: (theta[0], theta[1]), freq_bands_keys[1]: (middle[0],middle[1]), freq_bands_keys[2]: (gamma[0], gamma[1])}
        
        feedback_correct_curr_task, feedback_error_curr_task = [], []
        for i in range(len(feedback)):
            if feedback[i,0] == task_to_check:
                if feedback[i,1] > 0:
                    feedback_correct_curr_task.append(feedback[i,1])
                if feedback[i,2] > 0:
                    feedback_error_curr_task.append(feedback[i,2])
            
        events_dict = {}
        power_dict = {}
        channel_ripple_dict = {}
        onset_secs = []
        onset_secs_per_channel = {}
        
        # for task in range(0, len(seconds_lower[0:4])):
        for task in range(index_lower, index_upper):
            #for task in range(7, 10):
            sec_lower = seconds_lower[task]
            sec_upper = seconds_upper[task]
            print(f"Now analysing task between {sec_lower} and {sec_upper} secs")
            
            if sec_upper < block_size[0]/sampling_freq[0]:
                block = 0
                sec_lower_neuro = sec_lower
                sec_upper_neuro = sec_upper
            else:
                block = 1
                sec_lower_neuro = sec_lower-block_size[0]/sampling_freq[0]
                sec_upper_neuro = sec_upper-block_size[0]/sampling_freq[0]
        
            
            # redefine the lazy loader with every loop and see if that decreases memory load!!
            
            raw_analog_cropped = raw_file_lazy[block].analogsignals[0].load(time_slice = (sec_lower_neuro, sec_upper_neuro), channel_indexes = channel_indices_to_use)
            raw_analog_epo_cropped = raw_analog_cropped.T.reshape(1,raw_analog_cropped.shape[1], raw_analog_cropped.shape[0])
            
            # import pdb; pdb.set_trace() 
            
            power_mean = {}
            power_stepwise = {}
            for band, (l_freq, h_freq) in freq_bands.items():
                step = np.max([1, (h_freq - l_freq) / 20])
                freq_list = np.arange(l_freq, h_freq, step)
                # l_power = mne.time_frequency.tfr_array_morlet(raw_analog_epo_cropped, sampling_freq[block], freqs=freq_list, output="power", n_jobs=-1).squeeze()
                l_power = mne.time_frequency.tfr_array_morlet(raw_analog_epo_cropped, sampling_freq[block], freqs=freq_list, output="power", n_jobs = 3).squeeze()
                for idx_freq in range(len(freq_list)):
                    for channel_idx in range(len(channel_indices_to_use)):
                        l_power[channel_idx,idx_freq,:] = scipy.stats.zscore(l_power[channel_idx,idx_freq,:], axis=None)
                power_mean[band] = np.mean(l_power, axis=1)
                power_stepwise[band] = l_power
            
            power_dict[f"{task}_mean"] = power_mean
            power_dict[f"{task}_stepwise"] = power_stepwise
            
            # Collect all possible ripples for the current task
            threshold_hl = 5
            event_id = {}
            all_clusters = np.zeros((len(channel_indices_to_use), raw_analog_epo_cropped.shape[-1]))
            onsets, durations, descriptions, task_config_ripple = [], [], [], []
            for new_channel_idx, initial_channel_idx in enumerate(channel_indices_to_use):
                cluster_one_zero = np.zeros((len(power_mean.keys()), raw_analog_epo_cropped.shape[-1]))
                print(f"now looking at channel {channel_list[initial_channel_idx]}")
                for iband, band in enumerate(power_mean.keys()):
                    # I think this is redundant. compare against threshold if not middle, but middle also against threshold??? doesnt make much sense.
                    #cond = power_mean[band][new_channel_idx,:] > (threshold if band != "middle" else threshold)
                    cond = power_mean[band][new_channel_idx,:] > threshold_hl
                    # cond is a boolean array which is true in case the power is higher than the threshold
                    # each time there are more than 1 'trues' next to each other, this will be called a cluster.
                    # the first cluster will be marked by 1s, the second one by 2s, the third one by 3s,.. etc.
                    clusters, n_clusters = ndimage.label(cond)
                    for cli in range(1,n_clusters+1): #+ 1 bc it starts counting at 0 but we want start at 1
                        # check how long each cluster is. Cli is equal to the number with which the current cluster is marked
                        cl = np.where(clusters == cli)[0]
                        # length of cl is also the length of the cluster (in samples 1/freq * len = secs)
                        # according to paper, clusters need to be 15 ms or more -> 30
                        if len(cl) > 30:
                            # include the 15ms gap!!
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
                print(f"I found {n_clusters} between {sec_lower} and {seconds_upper[task]} secs in channel {channel_list[initial_channel_idx]}")
                # save all clusters uniquely per channel.
                all_clusters[new_channel_idx,:] = (1+new_channel_idx)*clusters 
                for cli in range(1, n_clusters+1):
                    # again check how long each cluster is. Cli is equal to the number of how the current cluster is marked
                    cl = np.where(clusters == cli)[0]
                    onsets.append(cl[0]) #onset is in samples, not seconds.
                    # onsets.append(cl[0] + sec_lower*sampling_freq[0]) #onset is in samples, not seconds.
                    durations.append(len(cl)) #duration is also samples, not seconds.
                    descriptions.append(f"channel_idx_{initial_channel_idx}")
                    task_config_ripple.append(task_config[task])
                    if f"channel_idx_{initial_channel_idx}" not in event_id:
                        event_id[f"channel_idx_{initial_channel_idx}"] = new_channel_idx
                    onset_secs.append(cl[0]/sampling_freq[block] + sec_lower)
                    if channel_list[initial_channel_idx] not in onset_secs_per_channel:
                        onset_secs_per_channel[channel_list[initial_channel_idx]] = []
                    onset_secs_per_channel[channel_list[initial_channel_idx]].append(cl[0]/sampling_freq[block] + sec_lower)
                # channel_ripple_dict[channel_list[initial_channel_idx]] = onset_secs
        
            # I NEED A neo.io file for these raw_cropped
            ch_types = ["ecog"] * len(channel_indices_to_use)
            info = mne.create_info(ch_names=channels_to_use, ch_types=ch_types, sfreq=sampling_freq[0])
            raw_np_cropped = raw_analog_cropped.as_array()
            raw_cropped = mne.io.RawArray(raw_np_cropped.T, info) # maybe without transpose?? try!
        
            
            # import pdb; pdb.set_trace()
            # create a mne.io object
            ch_types = ["sEEG"] * len(channel_indices_to_use)
            annot = mne.Annotations(onset=[x/sampling_freq[0] for x in onsets], duration=[x/sampling_freq[0] for x in durations], description=descriptions)
            raw_cropped.set_annotations(annot)
            # add events
            events, event_id = mne.events_from_annotations(raw_cropped, event_id=event_id)
            # events are times at which something happens, e.g. a ripple occurs
            events_dict[task] = events
             # import pdb; pdb.set_trace() 
             
            events_dict_per_channel[task_to_check] = events_dict[task]
            onset_in_secs_dict[task_to_check] = onset_secs_per_channel
            
            if save == True:
                raw_cropped.save(f"{LFP_dir}/{sub}/{names_blks_short[0]}-{analysis_type}_{sec_lower}-{sec_upper}-raw.fif", overwrite=True)
            
            # NEXT STEP: PLOTTING.
            if plotting_distr == True:
                if analysis_type == 'grid_wise' and task == index_lower:
                    # y_jitter = np.random.uniform(0, 0.01, size=len(onset_secs))
                    y_jitter = {key: np.random.uniform(0, 0.01, len(values)) for key, values in onset_in_secs_dict[task_to_check].items()}
                    colors = plt.cm.get_cmap('tab10', len(onset_in_secs_dict[task_to_check]))
                    
                    # Create a KDE plot for the data
                    plt.figure();
                    for idx, (condition, values) in enumerate(onset_in_secs_dict[task_to_check].items()):
                        plt.scatter(values, y_jitter[condition], color=colors(idx), label=condition, zorder = 1)
    
    
                    # plt.scatter(onset_secs, y_jitter, color='black', label='Ripple Candidates', zorder=1)
                    sns.kdeplot(onset_secs, fill=True, color='skyblue', label='Ripple Distribution')
                    
                    # Add a vertical line for the baseline reference
                    plt.axvline(x=(found_first_D[task_to_check-1]), color='black', linestyle='--', label='Found all 4 rewards')
                    plt.axvline(x=(seconds_upper[index_lower]), color='black', linestyle='--', label='First Correct')
                    
                    
                    # Add red rods for feedback: incorrect
                    sns.rugplot(feedback_error_curr_task, height=0.1, color='red', lw=2)  # Each data point as a 'rug'
    
                    # Add green rods for feedback: correct
                    sns.rugplot(feedback_correct_curr_task, height=0.1, color='green', lw=2)  # Each data point as a 'rug'
    
                    
                    # Add titles and labels
                    plt.title(f"Ripple frequency when solving grid {task_to_check} for subj {sub} [10 correct repeats]")
                    plt.xlabel('Seconds in Task')
                    plt.ylabel('Ripple Frequency')
                    
                    plt.xlim(new_grid_onset[task_to_check-1], new_grid_onset[task_to_check])
                    
                    # Add a legend
                    plt.legend()
                    
                    # Show the plot
                    plt.show()
        
            if plotting_ripples == True:
                
                # import pdb; pdb.set_trace()
                # CONTINUE HERE!!!
                # THE FLITEREING DOESTN WORK
                filtered_cropped_vhgamma = raw_cropped.filter(l_freq=gamma[0], h_freq=gamma[1], picks='all', fir_design='firwin')
                filtered_cropped_vhgamma_np = filtered_cropped_vhgamma.get_data()
            
            
                for ie, event in enumerate(events_dict[task]):
                    freq_to_plot = int(sampling_freq[0]/2)
                    
                    if event[0] > freq_to_plot and ie < 20:
                        print(f"now starting to plot overall {len(events_dict[task])} events")
                
                        # event[0] = onset
                        # event[1] = duration
                        # event[-1] = channel index
                        fig, axs = plt.subplots(5)
                        fig.suptitle(f"Ripples in subj {sub} - channel {channels_to_use[event[-2]]} - onset {event[0]/2000} sec; [2000 samples = 1 sec]")
                    
                        # Create x-values from 5500 to 9500
                        x = np.linspace(event[0]-freq_to_plot, event[0]+freq_to_plot-1, freq_to_plot*2)
                        
                        # first subplot is the raw signal:
                        axs[0].plot(x, raw_np_cropped[event[0]-freq_to_plot:event[0]+freq_to_plot, event[-1]], linewidth = 0.2)
                        axs[0].set_title('raw LFP')
                    
                        # the second subplot will be filtered for high gamma:
                        axs[1].plot(x, filtered_cropped_vhgamma_np[event[-1], event[0]-freq_to_plot:event[0]+freq_to_plot], linewidth = 0.2)    
                        axs[1].set_title('vhgamma filtered signal')    
                        
                        # the third subplot will be the mean power of this frequency: 
                        axs[2].plot(x,power_dict[f"{task}_mean"]['vhgamma'][event[-1], event[0]-freq_to_plot:event[0]+freq_to_plot])
                        axs[2].set_title('Mean power vhgamma')
                    
                        # the fourth subplot is the vhgamma power spectrum
                        power_to_plot_low = power_dict[f"{task}_stepwise"]['vhgamma'][event[-1], :, event[0]-freq_to_plot:event[0]+freq_to_plot] # Select first epoch and the specified channel
                        axs[3].imshow(power_to_plot_low, aspect='auto', origin='lower')
                        
                        # the fifth subplot is the overall power spectrum
                        # power_to_plot_all = np.stack(power_all[freq_bands_keys[0]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq], power_all[freq_bands_keys[1]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq], power_all[freq_bands_keys[2]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq])
                        power_to_plot_all = np.vstack((power_dict[f"{task}_stepwise"][freq_bands_keys[0]][event[-1], :, event[0]-freq_to_plot:event[0]+freq_to_plot], power_dict[f"{task}_stepwise"][freq_bands_keys[1]][event[-1], :, event[0]-freq_to_plot:event[0]+freq_to_plot]))
                        power_to_plot_all = np.vstack((power_to_plot_all, power_dict[f"{task}_stepwise"][freq_bands_keys[2]][event[-1], :, event[0]-freq_to_plot:event[0]+freq_to_plot]))
                        axs[4].imshow(power_to_plot_all, aspect='auto', origin='lower')
                        
            
    # save the numpy events file!
    if save == True:
        # # first put the channel names next to the channel IDs
        # events = np.c_[events, np.nan(events.shape[0])] 
        
        # channel_list = []
        # for i, ripple in enumerate(events):
        #     channel_list.append(HC_channels[int(events[i,2])])
        #     #events[i,-1] = channel_list_non_empty[int(events[i,2])]
        # channels_to_save = np.array(channel_list)   
        # channels_to_save = channels_to_save.reshape(channels_to_save.shape[0], 1)
        # events = np.hstack((events, channels_to_save))
    
        with open(f"{result_dir}/{sub}_{ROI}_{analysis_type}_ripple_events_dir.pkl", 'wb') as file:
            pickle.dump(events_dict, file)
        with open(f"{result_dir}/{sub}_{ROI}_{analysis_type}_ripple_by_seconds.pkl", 'wb') as file:
            pickle.dump(onset_in_secs_dict, file)

        np.save(f"{result_dir}/{sub}_beh_feedback", feedback)

        
