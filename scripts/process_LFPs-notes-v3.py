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

save = False

name_short = 'EMU-117_subj-YEU_task-ABCD_run-01_blk-01_NSP-1'
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

# theta = 4-8 HZ
# high gamma = 70 - 150 Hz
# Nyquist = half sampling rate -> 1000 Hz for ns3

# from https://github.com/NeuralEnsemble/python-neo/blob/master/neo/rawio/blackrockrawio.py
reader = neo.rawio.BlackrockRawIO(filename=f"{LFP_dir}/{sub}/{name_short}", nsx_to_load=3)
reader.parse_header()
# explore what's included in the header dict 
for k, v in reader.header.items():
    print(k, v)
print(reader)

sampling_freq = int(reader.sig_sampling_rates[3])

channel_names = reader.header['signal_channels']
channel_names = [str(elem) for elem in channel_names[:]]
channel_list = [name.split(",")[0].strip("('") for name in channel_names]
# filter out all channels that are 'empty'
channels_non_empty_idx = []
channel_list_non_empty = []
for i, channel in enumerate(channel_list):
    if not channel.startswith('empty'): #actually should be mLT
        channels_non_empty_idx.append(i)
        channel_list_non_empty.append(channel)
        


raw_np = reader.get_analogsignal_chunk(channel_indexes=channels_non_empty_idx, seg_index=1)
# careful! if this is less than 60*30*2.000 datapoints, it's likely the reference file

raw_np_epo = raw_np.T.reshape(1,raw_np.shape[1], raw_np.shape[0])
# format is epochs x channels x samples 

# # ok this always crashes. try to cut some of the channels.
# # OK THIS IS STILL CRASHING
# # maybe also because of the empty channels?
# # or because of this weird parallel jobs -1 thingy? why -1? look up!
raw_np_epo = raw_np_epo[:,101:,:].copy()
channel_list_non_empty = channel_list[101:].copy()
channels_non_empty_idx = channels_non_empty_idx[101:].copy()



# import pdb; pdb.set_trace()

# I need to refine this further once I have Habibas reply.
# ideally: load a vector of times based on the mat file and then loop.
# first define the times
# e.g. start of new task to having done the task correctly 2 times.
# crop so that I save time later
# raw_np.crop(2835, 3280).load_data().resample(1000)
# time I might be interested in for now: 
    # between 1734 and 1750 secs -> 52.020 52.500 in ns5
    # between 2835 and 2863 secs -> 85.050.000 85.890.000 in ns5 ; 5.670.000 in ns3
    # between 3112 and 3.122 secs -> 93.360 93.660 in ns5
    # between 3.263 and 3.280 secs -> 97.890 98.400 in ns5; 6.560.000 in ns3

time_l = 5670000
time_u = 6560000
raw_np_epo_cropped = raw_np_epo[:,:, time_l:time_u] #2835*2000 - 3.280 secs*2000




# this bit comes from Mathias
freq_bands_keys = ['theta', 'middle', 'vhgamma']
freq_bands = {freq_bands_keys[0]: (3, 8), freq_bands_keys[1]: (10,80), freq_bands_keys[2]: (105, 130)}
# delta 0-4 Hz
# theta 4 - 10 Hz
# alpha 8 - 12 Hz
# beta 15 - 30 Hz
# low gamma 30 - 90 Hz
# high gamma 90 - 150 Hz
        
# Filter the data in the frequency bands
power = {}
power_all = {}
for band, (l_freq, h_freq) in freq_bands.items():
    step = np.max([1, (h_freq - l_freq) / 20])
    freq_list = np.arange(l_freq, h_freq, step)
    l_power = mne.time_frequency.tfr_array_morlet(raw_np_epo_cropped, sampling_freq, freqs=freq_list, output="power", n_jobs=-1).squeeze()
    for idx_freq in range(len(freq_list)):
        for ich in range(len(channels_non_empty_idx)):
            l_power[ich,idx_freq,:] = scipy.stats.zscore(l_power[ich,idx_freq,:], axis=None)
    power[band] = np.mean(l_power, axis=1)
    power_all[band] = l_power

# Collect all possible ripples
threshold = 5
event_id = {}
all_clusters = np.zeros((len(channels_non_empty_idx), raw_np_epo_cropped.shape[-1]))
onsets, durations, descriptions = [], [], []
for ich, ch in enumerate(channels_non_empty_idx):
    cleaned = np.zeros((len(power.keys()), raw_np_epo_cropped.shape[-1]))
    for iband, band in enumerate(power.keys()):
        cond = power[band][ich,:] > (threshold if band != "middle" else threshold)
        clusters, n_clusters = ndimage.label(cond)

        for cli in range(1,n_clusters+1):
            cl = np.where(clusters == cli)[0]
            if len(cl) > 12:  #This means over 25ms
                cleaned[iband, cl] = 1
    
    # Keep events with power in both low theta and vhgamm, but NOT in the middle to avoid broadband
    clusters, n_clusters = ndimage.label((cleaned[0,:] + cleaned[2,:] + (1 - cleaned[1,:])) == 3)
    all_clusters[ich,:] = (1+ich)*clusters
    for cli in range(1, n_clusters+1):
        cl = np.where(clusters == cli)[0]
        onsets.append(cl[0])
        durations.append(len(cl))
        descriptions.append(f"ripple_{ch}")
        if f"ripple_{ch}" not in event_id:
            event_id[f"ripple_{ch}"] = ich

# create a mne.io object
ch_types = ["ecog"] * len(channels_non_empty_idx)
info = mne.create_info(ch_names=channel_list_non_empty, ch_types=ch_types, sfreq=sampling_freq)
raw_np_cropped = raw_np_epo_cropped.reshape(raw_np_epo_cropped.shape[2], raw_np_epo_cropped.shape[1])
# add info
raw_cropped = mne.io.RawArray(raw_np_cropped.T, info) # maybe without transpose?? try!
# can also easily be cropped and downsampled like this:
# raw_np.crop(2835, 3280).load_data().resample(1000)   
# add annotation
annot = mne.Annotations(onset=[x/sampling_freq for x in onsets], duration=[x/sampling_freq for x in durations], description=descriptions)
raw_cropped.set_annotations(annot)
# add events
events, event_id = mne.events_from_annotations(raw_cropped, event_id=event_id)
# events are times at which something happens, e.g. a ripple occurs

#
# in theory, if I would use epochs.compute_tfr, I could just do power.plot.
# from the raw_cropped object I just created, is it possible to do this power.plot?
# try and find out!!
#


if save == True:
    raw_cropped.save(f"{LFP_dir}/{sub}/{name_short}-{time_l}-{time_u}-raw.fif", overwrite=True)



# import pdb; pdb.set_trace()


# write a plotting function for which you can plot 
# - the raw signal
# - the power for the high gamma for the events per channel
# - the frequency filtered event per channel
# - the time-frequency transform as power spectrum

# plt.figure()
# plt.subplot with length of events

# Apply band-pass filter between 105 Hz and 130 Hz
filtered_raw_vhgamma = raw_cropped.filter(l_freq=105, h_freq=130, picks='all', fir_design='firwin')
filtered_raw_vhgamma_np = filtered_raw_vhgamma.get_data()

for ie, event in enumerate(events):
    # event[0] = onset
    # event[1] = duration
    # event[-1] = channel index
    fig, axs = plt.subplots(5)
    fig.suptitle(f"Ripple candidate - channel {channel_list_non_empty[event[-1]]}")

    x = np.linspace(0, int(sampling_freq)*2-1, int(sampling_freq)*2)
    
    # first subplot is the raw signal:
    axs[0].plot(x, raw_np_cropped[event[0]-sampling_freq:event[0]+sampling_freq, event[-1]])
    axs[0].set_title('raw LFP')

    # the second subplot will be filtered for high gamma:
    axs[1].plot(x, filtered_raw_vhgamma_np[event[-1], event[0]-sampling_freq:event[0]+sampling_freq])    
    axs[1].set_title('vhgamma filtered signal')    
    
    # the third subplot will be the power of this frequency: 
    axs[2].plot(x,power['vhgamma'][event[-1], event[0]-sampling_freq:event[0]+sampling_freq])
    axs[2].set_title('Power vhgamma')

    # the fourth subplot is the vhgamma power spectrum
    power_to_plot_low = l_power[event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq] # Select first epoch and the specified channel
    axs[3].imshow(power_to_plot_low, aspect='auto', origin='lower')
    
    # the fifth subplot is the overall power spectrum
    # power_to_plot_all = np.stack(power_all[freq_bands_keys[0]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq], power_all[freq_bands_keys[1]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq], power_all[freq_bands_keys[2]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq])
    power_to_plot_all = np.vstack((power_all[freq_bands_keys[0]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq], power_all[freq_bands_keys[1]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq]))
    power_to_plot_all = np.vstack((power_to_plot_all, power_all[freq_bands_keys[2]][event[-1], :, event[0]-sampling_freq:event[0]+sampling_freq]))
    axs[4].imshow(power_to_plot_all, aspect='auto', origin='lower')
    

# save the numpy events file!
if save == True:
    # first put the channel names next to the channel IDs
    events = np.c_[events, np.nan(events.shape[0])] 
    
    channel_list = []
    for i, ripple in enumerate(events):
        channel_list.append(channel_list_non_empty[int(events[i,2])])
        #events[i,-1] = channel_list_non_empty[int(events[i,2])]
    channels_to_save = np.array(channel_list)   
    channels_to_save = channels_to_save.reshape(channels_to_save.shape[0], 1)
    events = np.hstack((events, channels_to_save))
        
    np.save(f"{result_dir}/{sub}_{name_short}_50to100", events)
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



