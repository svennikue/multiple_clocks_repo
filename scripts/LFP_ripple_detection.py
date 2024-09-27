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
            
            
            
save = True
plotting = True

# if you want to show that there are more ripples in HPC than in mPFC
ROI = 'HPC' # HPC mPFC

# if you want to collect all ripples within a single grid
analysis_type = 'grid_wise' # grid_wise, exploration_trials 

task_to_check = 2

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
names_blks_short = ['EMU-117_subj-YEU_task-ABCD_run-01_blk-01_NSP-1','EMU-118_subj-YEU_task-ABCD_run-01_blk-02_NSP-1']
LFP_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
result_dir = f"{LFP_dir}/results"

# subjects = ['s13', 's12', 's25']
sub = 's25'

# load behaviour that defines my snippets.
behaviour = np.genfromtxt(f"{LFP_dir}/{sub}/exploration_trials_times_and_ncorrect.csv", delimiter=',')
behaviour_all = np.genfromtxt(f"{LFP_dir}/{sub}/all_trials_times.csv", delimiter=',')


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

elif analysis_type == 'grid_wise':
    index_lower = []
    index_upper = []
    # define seconds_lower[task] as a new repeat of a grid.
    # also collect grid_index (task_config) to keep track if you're still in the same grid.
    seconds_lower, seconds_upper, task_config, task_index  = [], [], [], []
    for i in range(1, len(behaviour_all)):
        if i == 1: 
            seconds_lower.append(behaviour_all[i-1, 1])
            task_config.append([behaviour_all[i, 5], behaviour_all[i, 6],behaviour_all[i, 7],behaviour_all[i, 8]])
            task_index.append(behaviour_all[i,-1])
        curr_repeat = behaviour_all[i, 0]
        last_repeat = behaviour_all[i-1, 0]
        if curr_repeat > last_repeat:
            seconds_lower.append(behaviour_all[i, 1])
            seconds_upper.append(behaviour_all[i-1, 4])
            task_config.append([behaviour_all[i, 5], behaviour_all[i, 6],behaviour_all[i, 7],behaviour_all[i, 8]])
            task_index.append(behaviour_all[i,-1])
        if curr_repeat < last_repeat:
            seconds_lower.append(behaviour_all[i, 1])
            seconds_upper.append(behaviour_all[i-1, 4])
            
    seconds_upper.append(behaviour_all[i, 4])

    for i, task in enumerate(task_index):
        if task == task_to_check and index_lower == []:
            index_lower = i
        if task_index[i-1] == task_to_check and task_index[i] > task_to_check:
            index_upper = i
            

# instead of fully loading the files, I am only loading the reader and then 
# looking at them in lazy-mode, only calling the shorter segments.
reader, block_size, channel_list, sampling_freq, raw_file_lazy = [], [], [], [], []
for file_half in [0,1]:
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

    
events_dict = {}
power_dict = {}
channel_ripple_dict = {}
onset_secs = []

# for task in range(0, len(seconds_lower[0:4])):
for task in range(index_lower, index_upper):
    #for task in range(7, 10):
    sec_lower = seconds_lower[task]
    sec_upper = seconds_upper[task]
    print(f"Now analysing task between {sec_lower} and {sec_upper} secs")
    
    if sec_upper < block_size[0]/sampling_freq[0]:
        block = 0
    else:
        block = 1
        sec_lower = sec_lower-block_size[0]/sampling_freq[0]
        sec_upper = sec_upper-block_size[0]/sampling_freq[0]

    raw_analog_cropped = raw_file_lazy[block].analogsignals[0].load(time_slice = (sec_lower, sec_upper), channel_indexes = channel_indices_to_use)
    raw_analog_epo_cropped = raw_analog_cropped.T.reshape(1,raw_analog_cropped.shape[1], raw_analog_cropped.shape[0])
    
    # import pdb; pdb.set_trace() 
    
    power_mean = {}
    power_stepwise = {}
    for band, (l_freq, h_freq) in freq_bands.items():
        step = np.max([1, (h_freq - l_freq) / 20])
        freq_list = np.arange(l_freq, h_freq, step)
        # l_power = mne.time_frequency.tfr_array_morlet(raw_analog_epo_cropped, sampling_freq[block], freqs=freq_list, output="power", n_jobs=-1).squeeze()
        l_power = mne.time_frequency.tfr_array_morlet(raw_analog_epo_cropped, sampling_freq[block], freqs=freq_list, output="power", n_jobs = -5).squeeze()
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
            # onsets.append(cl[0]) #onset is in samples, not seconds.
            onsets.append(cl[0] + sec_lower*sampling_freq[0]) #onset is in samples, not seconds.
            durations.append(len(cl)) #duration is also samples, not seconds.
            descriptions.append(f"channel_idx_{initial_channel_idx}")
            task_config_ripple.append(task_config[task])
            if f"channel_idx_{initial_channel_idx}" not in event_id:
                event_id[f"channel_idx_{initial_channel_idx}"] = new_channel_idx
            onset_secs.append(cl[0]/sampling_freq[block] + sec_lower)
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
    import pdb; pdb.set_trace() 
    
    # I HAVE NO IDEA WHAT IS GOUGB WRONG BUT FOR SME REASON IT DOESNT SAFE THE EVENTS ANYMORE...
    # WHY???
    
    
    # continue here!!!
    #
    #
    
    if save == True:
        raw_cropped.save(f"{LFP_dir}/{sub}/{names_blks_short[0]}-{analysis_type}_{sec_lower}-{sec_upper}-raw.fif", overwrite=True)
    
    # NEXT STEP: PLOTTING.
    if plotting == True:
        # if analysis_type == 'grid_wise':
            
        #     # Create a KDE plot for the data
        #     sns.kdeplot(onset_secs, fill=True, color='skyblue', label='Data Distribution')
            
        #     # Add a vertical line for the baseline reference
        #     plt.axvline(x=(seconds_lower[index_lower]), color='red', linestyle='--', label='Start new Grid')
        #     plt.axvline(x=(seconds_upper[index_lower]), color='red', linestyle='--', label='First Correct')
            
        #     # Add titles and labels
        #     plt.title('Ripple frequency when solving a single grid [10 correct repeats]')
        #     plt.xlabel('Seconds in Task')
        #     plt.ylabel('Ripple Frequency')
            
        #     # Add a legend
        #     plt.legend()
            
        #     # Show the plot
        #     plt.show()

        
        
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
                fig.suptitle(f"Ripple candidate - channel {channels_to_use[event[-2]]} - onset {event[0]/2000} sec; [2000 samples = 1 sec]")
            
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
        

        # np.save(f"{result_dir}/{sub}_{names_blks_short[0]}_50to100", events)
        # n_cycles = freqs / 2
        # power_morlet = mne.time_frequency.tfr_array_morlet(HC_L_raw_epo_cut, sfreq=sampling_freq, freqs=freqs, n_cycles=n_cycles, output='power')
    
        # # Plotting
        # plt.figure(figsize=(10, 6))
        # # plt.imshow(10 * np.log10(power_to_plot), aspect='auto', origin='lower',
        # #            extent=[0, HC_L_raw_epo_cut.shape[-1] / sampling_freq, freqs[0], freqs[-1]], cmap='viridis')
        # plt.imshow(power_to_plot[:, 100000:140000], aspect='auto', origin='lower',
        #            extent=[0, HC_L_raw_epo_cut.shape[-1] / sampling_freq, freqs[0], freqs[-1]], cmap='viridis', vmin = 0, vmax = 300)
        # plt.colorbar(label='Power (dB)')
        # #plt.colorbar(vmin=0, vmax=260)
        # plt.xlabel('Time (s)')
        # plt.ylabel('Frequency (Hz)')
        # plt.title(f'Power Spectrum - Channel {HC_L_channels[index_channel_for_powerplot[-1]]}')
        # # plt.yscale('log')  # Log scale for frequency axis if needed
        # plt.show()
            
    
        # raw_cropped.crop((event[0]-sampling_freq)/sampling_freq, (event[0]+sampling_freq)/sampling_freq).load_data().plot(n_channels=event[-1], scalings='auto')
        
        # raw_cropped_to_plot = raw_cropped.crop((event[0]-sampling_freq)/sampling_freq, (event[0]+sampling_freq)/sampling_freq).plot(n_channels=event[-1], scalings='auto')
               
        # axs[1].filtered_raw_vhgamma.crop(onsets[ie]-2000, onsets[ie]+2000).load_data().plot(duration=2, n_channels=event[-1], scalings='auto')
        
    # careful! if this is less than 60*30*30.000 datapoints, it's likely the reference file
    # raw_chunk = reader.get_analogsignal_chunk(block_index=0, seg_index=1)
    # float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, stream_index=0)
    # raw_sigs1 = reader.get_analogsignal_chunk(channel_indexes=[0, 2, 4], seg_index=1)  # Take 0 2 and 4
    # raw_sigs2 = reader.get_analogsignal_chunk(channel_ids=['1', '3', '5'], seg_index=1)  # Same but with their id (1 based)
    # raw_sigs3 = reader.get_analogsignal_chunk(channel_names=['mLT2aHa01-001', 'mLT2aHa02-002', 'mLT2aHa03-003'], seg_index=1)
    
    # # from MNE website
    # # https://mne.tools/dev/auto_examples/io/read_neo_format.html
    # # create fake data with NEO'S exampleIO
    # reader = neo.io.ExampleIO("fakedata.nof")
    # block = reader.read(lazy=False)[0]  # get the first block
    # segment = block.segments[0]  # get data from first (and only) segment
    # signals = segment.analogsignals[0]  # get first (multichannel) signal
    
    # data = signals.rescale("V").magnitude.T
    # sfreq = signals.sampling_rate.magnitude
    # ch_names = [f"Neo {(idx + 1):02}" for idx in range(signals.shape[1])]
    # ch_types = ["eeg"] * len(ch_names)  # if not specified, type 'misc' is assumed
    


