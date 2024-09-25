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
import neo.rawio
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import mmap

# # check this to save RAM!

# import pdb; pdb.set_trace() 

# # write a simple example file
# with open("hello.txt", "wb") as f:
#     f.write(b"Hello Python!\n")

# with open("hello.txt", "r+b") as f:
#     # memory-map the file, size 0 means whole file
#     mm = mmap.mmap(f.fileno(), 0)
#     # read content via standard file methods
#     print(mm.readline())  # prints b"Hello Python!\n"
#     # read content via slice notation
#     print(mm[:5])  # prints b"Hello"
#     # update content using slice notation;
#     # note that new content must have same size
#     mm[6:] = b" world!\n"
#     # ... and read again using standard file methods
#     mm.seek(0)
#     print(mm.readline())  # prints b"Hello  world!\n"
#     # close the map
#     mm.close()

save = False

names_blks_short = ['EMU-117_subj-YEU_task-ABCD_run-01_blk-01_NSP-1','EMU-118_subj-YEU_task-ABCD_run-01_blk-02_NSP-1']
LFP_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
result_dir = f"{LFP_dir}/results"
# ok re-write this
# include all channels
# include a flexible bit where I can shorten according to a certain time
# include a figure in which I plot the signal/filtered plus power at the 
# suspected ripple events.

# subjects = ['s13', 's12', 's25']
sub = 's25'
# ns3: contains analog data; sampled at 2000 Hz (+ digital filters)

    # delta 0-4 Hz
    # theta 4 - 10 Hz
    # alpha 8 - 12 Hz
    # beta 15 - 30 Hz
    # low gamma 30 - 90 Hz
    # high gamma 90 - 150 Hz
    # (3, 8), freq_bands_keys[1]: (10,80), freq_bands_keys[2]: (105, 130)}
# define your frequencies
theta = [3,8]
middle = [10, 80]

# SPW-R frequency band criterion for rodents (100 to 250 Hz) is generally higher 
# than for monkeys (95 to 250 Hz) or humans (70–250 Hz, most use 80–150 Hz bandpass filters
gamma = [80, 200]

# theta = 4-8 HZ
# high gamma = 70 - 150 Hz
# Nyquist = half sampling rate -> 1000 Hz for ns3


# also read the times files
# 0-3 are times of state change
# 4 is correct repeats of current context
# 5 is number of repeats independent of making mistakes
# 6-9 is current configuration of grid
behaviour = np.genfromtxt(f"{LFP_dir}/{sub}/exploration_trials_times_and_ncorrect.csv", delimiter=',')

# Initialize empty lists for t_lower and t_upper
seconds_lower, seconds_upper, task_config  = [], [], []
if behaviour[0, 4] == 0:
    seconds_lower.append(behaviour[0, 0])


# Loop through the array to identify where subjects first got the task right.
# for i in range(1, len(behaviour)):       
#     # If we have a transition from 1 to 0, i.e. a new task is starting, the last repeat of the task ends at i-1
#     if behaviour[i-1, 4] == 1 and behaviour[i, 4] == 0:
#         t_upper.append(behaviour[i-1, 3])  # Append the value from the 4th column
        
#     # If we have a transition from 0 to 1, i.e. a task has been solved correctly for the first time, the first incorrect (0) of the task is at i
#     if behaviour[i, 4] == 0 and behaviour[i-1, 4] == 1:
#         t_lower.append(behaviour[i, 0])  # Append the value from the 1st column

# maybe the better way is to use the end of the first repeat as the lower end, and keep the upper end defined as doing it correctly for the firs time
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
# import pdb; pdb.set_trace()

if os.path.exists(f"{LFP_dir}/{sub}/{sub}_raw_ns3_blck1-blck2.npy"):
    raw_np_mmap = np.load(f"{LFP_dir}/{sub}/{sub}_raw_ns3_blck1-blck2.npy", mmap_mode='r')
    channel_list = np.load(f"{LFP_dir}/{sub}/{sub}_channel_list_ns3_blck1-blck2.npy")
    HC_indices = np.load(f"{LFP_dir}/{sub}/{sub}_HC_indices_ns3_blck1-blck2.npy")
    sampling_freq = int(np.load(f"{LFP_dir}/{sub}/{sub}_frequency_ns3_blck1-blck2.npy"))
    
else:  
    for block, file_name_per_block in enumerate(names_blks_short):
        # from https://github.com/NeuralEnsemble/python-neo/blob/master/neo/rawio/blackrockrawio.py
        reader = neo.rawio.BlackrockRawIO(filename=f"{LFP_dir}/{sub}/{file_name_per_block}", nsx_to_load=3)
        reader.parse_header()
        # explore what's included in the header dict 
        for k, v in reader.header.items():
            print(k, v)
        print(reader)
        
        if block == 0:
            sampling_freq = int(reader.sig_sampling_rates[3])
            channel_names = reader.header['signal_channels']
            channel_names = [str(elem) for elem in channel_names[:]]
            channel_list = [name.split(",")[0].strip("('") for name in channel_names]
        if block > 0:
            sampling_freq_two = int(reader.sig_sampling_rates[3])
            channel_names_two = reader.header['signal_channels']
            channel_names_two = [str(elem) for elem in channel_names_two[:]]
            channel_list_two = [name.split(",")[0].strip("('") for name in channel_names_two]
            if sampling_freq_two != sampling_freq:
                print('Careful! the files dont have the same sampling frequency! Probably wrong filename.')
                import pdb; pdb.set_trace()
            if channel_list_two != channel_list:
                print('Careful! the files dont have the same channels! Probably wrong filename.')
                import pdb; pdb.set_trace()
        
        # if the files are from the same session, then continue and 
        # only select hippocampal electrodes for now.
        # these are with Ha or Hb [mPFC is CA]
        if block == 0:
            HC_indices = []
            mPFC_indices = []
            for i, channel in enumerate(channel_list):
                if 'Ha' in channel or 'Hb' in channel:
                    HC_indices.append(i)
                if 'CA' in channel:
                    mPFC_indices.append(i)    
            
            HC_channels = [channel_list[i] for i in HC_indices]
            mPFC_channels = [channel_list[i] for i in mPFC_indices]
        
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
 
raw_np_mmap_epo = raw_np_mmap.T.reshape(1,raw_np_mmap.shape[1], raw_np_mmap.shape[0])
ch_types = ["ecog"] * len(HC_indices)
HC_channels = [channel_list[i] for i in HC_indices]

info = mne.create_info(ch_names=HC_channels, ch_types=ch_types, sfreq=sampling_freq)
raw_complete = mne.io.RawArray(raw_np_mmap.T, info) # maybe without transpose?? try!

# import pdb; pdb.set_trace()   

events_dict = {}
power_dict = {}
for task, sec_lower in enumerate(seconds_lower):
    print(f" analysing task {task}")
    sample_u = int(seconds_upper[task]*sampling_freq)
    sample_l = int(sec_lower*sampling_freq)
    
    print(f"Now analysing task between {sec_lower} and {seconds_upper[task]} secs")
    
    raw_np_epo_cropped = raw_np_mmap_epo[:,:, sample_l:sample_u] #2835*2000 - 3.280 secs*2000

    # this bit comes from Mathias
    freq_bands_keys = ['theta', 'middle', 'vhgamma']
    # try 80 and 250 instead
    freq_bands = {freq_bands_keys[0]: (theta[0], theta[1]), freq_bands_keys[1]: (middle[0],middle[1]), freq_bands_keys[2]: (gamma[0], gamma[1])}
    # freq_bands = {freq_bands_keys[0]: (3, 8), freq_bands_keys[1]: (10,80), freq_bands_keys[2]: (105, 130)}
    # delta 0-4 Hz
    # theta 4 - 10 Hz
    # alpha 8 - 12 Hz
    # beta 15 - 30 Hz
    # low gamma 30 - 90 Hz
    # high gamma 90 - 150 Hz
            
    # Filter the data in the frequency bands
    power_mean = {}
    power_stepwise = {}
    for band, (l_freq, h_freq) in freq_bands.items():
        step = np.max([1, (h_freq - l_freq) / 20])
        freq_list = np.arange(l_freq, h_freq, step)
        l_power = mne.time_frequency.tfr_array_morlet(raw_np_epo_cropped, sampling_freq, freqs=freq_list, output="power", n_jobs=-1).squeeze()
        for idx_freq in range(len(freq_list)):
            for channel_idx in range(len(HC_indices)):
                l_power[channel_idx,idx_freq,:] = scipy.stats.zscore(l_power[channel_idx,idx_freq,:], axis=None)
        power_mean[band] = np.mean(l_power, axis=1)
        power_stepwise[band] = l_power
    
    power_dict[f"{task}_mean"] = power_mean
    power_dict[f"{task}_stepwise"] = power_stepwise
    
    # Collect all possible ripples
    threshold_hl = 5
    event_id = {}
    all_clusters = np.zeros((len(HC_indices), raw_np_epo_cropped.shape[-1]))
    onsets, durations, descriptions, task_config_ripple = [], [], [], []
    
    for new_channel_idx, initial_channel_idx in enumerate(HC_indices):
        cluster_one_zero = np.zeros((len(power_mean.keys()), raw_np_epo_cropped.shape[-1]))
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
                if len(cl) > 15:  #This means over 7.5ms (previous criterium was 25ms)
                    cluster_one_zero[iband, cl] = 1
        
        # Keep events with power in both low theta and vhgamm, but NOT in the middle to avoid broadband
        # this is a boolean array that is true for a cluster and false for any other sample
        # ndimage.label will returen clusters as an integer ndarray where each unique feature in input has a unique label 
        # and n_clusters as How many objects were found.
        
        # what if I ignore the middle band?
        clusters, n_clusters = ndimage.label((cluster_one_zero[0,:] + cluster_one_zero[2,:] ) == 2)
        
        # clusters, n_clusters = ndimage.label((cluster_one_zero[0,:] + cluster_one_zero[2,:] + (1 - cluster_one_zero[1,:])) == 3)
        
        
        
        # now again, I save the amount of clusters; this time only if it fulfills both criteria: threshold + no middle.
        print(f"I found {n_clusters} between {sec_lower} and {seconds_upper[task]} secs in channel {channel_list[initial_channel_idx]}")
        # save all clusters uniquely per channel.
        all_clusters[new_channel_idx,:] = (1+new_channel_idx)*clusters 
        for cli in range(1, n_clusters+1):
            # again check how long each cluster is. Cli is equal to the number of how the current cluster is marked
            cl = np.where(clusters == cli)[0]
            onsets.append(cl[0]) #onset is in samples, not seconds.
            durations.append(len(cl)) #duration is also samples, not seconds.
            descriptions.append(f"channel_idx_{initial_channel_idx}")
            task_config_ripple.append(task_config[task])
            if f"channel_idx_{initial_channel_idx}" not in event_id:
                event_id[f"channel_idx_{initial_channel_idx}"] = new_channel_idx
    
    
    # import pdb; pdb.set_trace()
    # create a mne.io object
    ch_types = ["ecog"] * len(HC_indices)
    info = mne.create_info(ch_names=HC_channels, ch_types=ch_types, sfreq=sampling_freq)
    raw_np_cropped = raw_np_epo_cropped.reshape(raw_np_epo_cropped.shape[2], raw_np_epo_cropped.shape[1])
    # add info
    raw_cropped = mne.io.RawArray(raw_np_cropped.T, info) # maybe without transpose?? try!
    # can also easily be cropped and downsampled like this:
    # raw_np.crop(2835, 3280).load_data().resample(1000)   
    # add annotation
    #annot = mne.Annotations(onset=[x/sampling_freq for x in onsets], duration=[x/sampling_freq for x in durations], description=descriptions, task_config = task_config_ripple)
    
    # MAKE SURE THE TIMINGS ARE NOT MIXED UP!!!
    annot = mne.Annotations(onset=[x/sampling_freq for x in onsets], duration=[x/sampling_freq for x in durations], description=descriptions)
    
    raw_cropped.set_annotations(annot)
    # add events
    events, event_id = mne.events_from_annotations(raw_cropped, event_id=event_id)
    # events are times at which something happens, e.g. a ripple occurs
    events_dict[task] = events
    #
    # in theory, if I would use epochs.compute_tfr, I could just do power.plot.
    # from the raw_cropped object I just created, is it possible to do this power.plot?
    # try and find out!!
    #
    
    
    if save == True:
        raw_cropped.save(f"{LFP_dir}/{sub}/{names_blks_short[0]}-{sample_l}-{sample_u}-raw.fif", overwrite=True)
    
    
    
    # import pdb; pdb.set_trace()
    
    
# write a plotting function for which you can plot 
# - the raw signal
# - the power for the high gamma for the events per channel
# - the frequency filtered event per channel
# - the time-frequency transform as power spectrum

# plt.figure()
# plt.subplot with length of events

# Apply band-pass filter between 105 Hz and 130 Hz
# filtered_raw_vhgamma = raw_cropped.filter(l_freq=gamma[0], h_freq=gamma[1], picks='all', fir_design='firwin')
# filtered_raw_vhgamma_np = filtered_raw_vhgamma.get_data()
filtered_raw_vhgamma = raw_complete.filter(l_freq=gamma[0], h_freq=gamma[1], picks='all', fir_design='firwin')
filtered_vhgamma_np = filtered_raw_vhgamma.get_data()

# I SUSPECT ALL OF THE INDEXING MIGHT BE FUCKED NOW
# chekc how i get events!!
# and how i can actually use the task config
# if events.shape[0] > 0:
#     import pdb; pdb.set_trace()
for taski, task in enumerate(events_dict):   
    sample_u = int(seconds_upper[taski]*sampling_freq)
    sample_l = int(seconds_lower[taski]*sampling_freq)
    
    print(f"Now analysing task between {seconds_lower[taski]} and {seconds_upper[taski]} secs")
    
    raw_np_epo_cropped = raw_np_epo[:,:, sample_l:sample_u] #2835*2000 - 3.280 secs*2000
    raw_np_cropped = raw_np_epo_cropped.reshape(raw_np_epo_cropped.shape[2], raw_np_epo_cropped.shape[1])

    filtered_cropped_vhgamma_np = filtered_vhgamma_np[:, sample_l:sample_u]
    
    for ie, event in enumerate(events_dict[task]):
        

        # event[0] = onset
        # event[1] = duration
        # event[-1] = channel index
        fig, axs = plt.subplots(5)
        fig.suptitle(f"Ripple candidate - channel {HC_channels[event[-2]]} - onset {event[0]/2000} sec; [2000 samples = 1 sec]")
    
        # Create x-values from 5500 to 9500
        x = np.linspace(event[0]-sampling_freq, event[0]+sampling_freq-1, sampling_freq*2)
        
        # first subplot is the raw signal:
        axs[0].plot(x, raw_np_cropped[event[0]-sampling_freq:event[0]+sampling_freq, event[-1]])
        axs[0].set_title('raw LFP')
    
        # the second subplot will be filtered for high gamma:
        axs[1].plot(x, filtered_cropped_vhgamma_np[event[-1], event[0]-sampling_freq:event[0]+sampling_freq])    
        axs[1].set_title('vhgamma filtered signal')    
        
        # the third subplot will be the mean power of this frequency: 
        axs[2].plot(x,power_dict[f"{task}_mean"]['vhgamma'][event[-1], event[0]-sampling_freq:event[0]+sampling_freq])
        axs[2].set_title('Mean power vhgamma')
    
        # the fourth subplot is the vhgamma power spectrum
        power_to_plot_low = power_dict[f"{task}_stepwise"][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq] # Select first epoch and the specified channel
        axs[3].imshow(power_to_plot_low, aspect='auto', origin='lower')
        
        # the fifth subplot is the overall power spectrum
        # power_to_plot_all = np.stack(power_all[freq_bands_keys[0]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq], power_all[freq_bands_keys[1]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq], power_all[freq_bands_keys[2]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq])
        power_to_plot_all = np.vstack((power_dict[f"{task}_stepwise"][freq_bands_keys[0]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq], power_dict[f"{task}_stepwise"][freq_bands_keys[1]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq]))
        power_to_plot_all = np.vstack((power_to_plot_all, power_dict[f"{task}_stepwise"][freq_bands_keys[2]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq]))
        axs[4].imshow(power_to_plot_all, aspect='auto', origin='lower')
        
    
    # save the numpy events file!
    if save == True:
        # first put the channel names next to the channel IDs
        events = np.c_[events, np.nan(events.shape[0])] 
        
        channel_list = []
        for i, ripple in enumerate(events):
            channel_list.append(HC_channels[int(events[i,2])])
            #events[i,-1] = channel_list_non_empty[int(events[i,2])]
        channels_to_save = np.array(channel_list)   
        channels_to_save = channels_to_save.reshape(channels_to_save.shape[0], 1)
        events = np.hstack((events, channels_to_save))
            
        np.save(f"{result_dir}/{sub}_{names_blks_short[0]}_50to100", events)
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
    


