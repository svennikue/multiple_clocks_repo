#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:08:37 2024

@author: xpsy1114
"""

import mne
import neo
import neo.rawio
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

name_short = 'EMU-117_subj-YEU_task-ABCD_run-01_blk-01_NSP-1'
LFP_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"


# def rename_channel(name):
#     if name.startswith('mLF'):
#         return 'left_ACC_' + name.split('-')[-1]  # Replace prefix with 'left_ACC' and keep the last identifier
#     elif name.startswith('mLT'):
#         return 'left_HC_' + name.split('-')[-1]   # Replace prefix with 'left_HC' and keep the last identifier
#     elif name.startswith('mRT'):
#         return 'right_ACC_' + name.split('-')[-1] # Replace prefix with 'right_ACC' and keep the last identifier
#     elif name.startswith('mRF'):
#         return 'right_HC_' + name.split('-')[-1]  # Replace prefix with 'right_HC' and keep the last identifier
#     return name  # Return unchanged if no match


# subjects = ['s13', 's12', 's25']
subjects = ['s25']
for sub in subjects:
    # from new rawio webiste
    # https://github.com/NeuralEnsemble/python-neo/blob/master/neo/rawio/blackrockrawio.py

    # ns3: contains analog data; sampled at 2000 Hz (+ digital filters)
    # ns5: contains analog data; sampled at 30000 Hz (+ digital filters)
    # theta = 4-8 HZ
    # high gamma = 70 - 150 Hz
    # Nyquist = half sampling rate -> 1000 Hz for ns3
    
    reader = neo.rawio.BlackrockRawIO(filename=f"{LFP_dir}/{sub}/{name_short}", nsx_to_load=3)
    reader.parse_header()
    sampling_freq = reader.sig_sampling_rates[3]
    print(reader)
    
    channel_names = reader.header['signal_channels']
    channel_names = [str(elem) for elem in channel_names[:]]
    extracted_names = [name.split(",")[0].strip("('") for name in channel_names]
    
    HC_L_indices = []
    for i, channel in enumerate(extracted_names):
        if channel.startswith('LT'): #actually should be mLT
            HC_L_indices.append(i)
            
    HC_L_channels = [extracted_names[i] for i in HC_L_indices]
    
    HC_L_raw = reader.get_analogsignal_chunk(channel_indexes=HC_L_indices, seg_index=1)
    ch_types = ["ecog"] * len(HC_L_channels)
    info = mne.create_info(ch_names=HC_L_channels, ch_types=ch_types, sfreq=sampling_freq)
    raw_np = mne.io.RawArray(HC_L_raw.T, info)
    
    HC_L_raw_epo = HC_L_raw.T.reshape(1,50,9629468)
    
    # time I might be interested in for now: 
        # between 1734 and 1750 secs -> 52.020 52.500 in ns5
        # between 2835 and 2863 secs -> 85.050.000 85.890.000 in ns5 ; 5.670.000 in ns3
        # between 3112 and 3.122 secs -> 93.360 93.660 in ns5
        # between 3.263 and 3.280 secs -> 97.890 98.400 in ns5; 6.560.000 in ns3
        
    HC_L_raw_epo_cut = HC_L_raw_epo[:,:, 5670000:6560000] #2835*2000 - 3.280 secs*2000
    
    freq_bands = {'theta': (3, 8), 'middle': (10,80), 'vhgamma': (105, 130)}

    # Filter the data in the frequency bands
    power = {}
    for band, (l_freq, h_freq) in freq_bands.items():
        step = np.max([1, (h_freq - l_freq) / 20])
        freq_list = np.arange(l_freq, h_freq, step)
        l_power = mne.time_frequency.tfr_array_morlet(HC_L_raw_epo_cut, sampling_freq, freqs=freq_list, output="power", n_jobs=-1).squeeze()
        for idx_freq in range(len(freq_list)):
            for ich in range(len(HC_L_channels)):
                l_power[ich,idx_freq,:] = scipy.stats.zscore(l_power[ich,idx_freq,:], axis=None)
        power[band] = np.mean(l_power, axis=1)
    
    # Collect all possible ripples
    threshold = 5
    event_id = {}
    all_clusters = np.zeros((len(HC_L_channels), HC_L_raw_epo_cut.shape[-1]))
    onsets, durations, descriptions = [], [], []
    for ich, ch in enumerate(HC_L_channels):
        cleaned = np.zeros((len(power.keys()), HC_L_raw_epo_cut.shape[-1]))
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
                event_id[f"ripple_{ch}"] = 256 + ich
    
    # Make new annotations
    annot = mne.Annotations(onset=[x/500 for x in onsets], duration=[x/500 for x in durations], description=descriptions)
    ch_types = ["ecog"] * len(HC_L_channels)
    info = mne.create_info(ch_names=HC_L_channels, ch_types=ch_types, sfreq=sampling_freq)
    HC_L_raw_cut= HC_L_raw_epo_cut.reshape(50, 890000)
    HC_L_raw_cut_mne = mne.io.RawArray(HC_L_raw_cut, info)
    
    
    HC_L_raw_cut_mne.set_annotations(annot)
    events, event_id = mne.events_from_annotations(HC_L_raw_cut_mne, event_id=event_id)
    
    
    
    # A SpikeTrain represents the times of occurrence of action potentials (spikes).

    
    
    # simplified_names = [rename_channel(name) for name in extracted_names]
    # frequency = reader.sig_sampling_rates[5]
    # header = reader.header
    
    # careful! if this is less than 60*30*30.000 datapoints, it's likely the reference file
    # raw_chunk = reader.get_analogsignal_chunk(block_index=0, seg_index=1)
    # float_chunk = reader.rescale_signal_raw_to_float(raw_chunk, stream_index=0)
    
    # raw_sigs1 = reader.get_analogsignal_chunk(channel_indexes=[0, 2, 4], seg_index=1)  # Take 0 2 and 4
    # raw_sigs2 = reader.get_analogsignal_chunk(channel_ids=['1', '3', '5'], seg_index=1)  # Same but with their id (1 based)
    # raw_sigs3 = reader.get_analogsignal_chunk(channel_names=['mLT2aHa01-001', 'mLT2aHa02-002', 'mLT2aHa03-003'], seg_index=1)
    
    
    #plt.figure(); plt.imshow(raw_chunk, aspect = 'auto')
    #plt.xticks(ticks=np.linspace(0,raw_chunk.shape[1]-1, len(extracted_names)), labels = extracted_names, rotation=90)
    
    #only plot electrodes
    # plt.figure(); plt.imshow(raw_sigs1[0:1000000,:].T, aspect = 'auto')
    # #plt.xticks(ticks=np.linspace(0,45, 46), labels = extracted_names[0:46], rotation=45)
    
    
    # ch_names = [f"Neo {(idx + 1):02}" for idx in range(raw_sigs3.shape[1])]
    # ch_types = ["eeg"] * len(ch_names)
    # info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=frequency[5])
    # raw = mne.io.RawArray(raw_sigs3.T, info)
    # raw.plot(show_scrollbars=False)
    
    # Read a segment (e.g., first segment in the first block)
    # segment = reader.read_segment(block_index=block_index, seg_index=seg_index)
    
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
    
    # info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    # raw = mne.io.RawArray(data, info)
    # raw.plot(show_scrollbars=False)
    # # create raw_np the analogous way so that I can continue with what Mathias did!


# #
# # from Mathias
# #   
# # FIND OUT WHAT THESE ARE!!!    
# raw_np = 0
# ndimage = 0

# # from Mathias
# freq_bands = {'theta': (3, 8), 'middle': (10,80), 'vhgamma': (105, 130)}

# # Filter the data in the frequency bands
# power = {}
# for band, (l_freq, h_freq) in freq_bands.items():
#     step = np.max([1, (h_freq - l_freq) / 20])
#     freq_list = np.arange(l_freq, h_freq, step)
#     l_power = mne.time_frequency.tfr_array_morlet(raw_np, raw.info["sfreq"], freqs=freq_list, output="power", n_jobs=-1).squeeze()
#     for idx_freq in range(len(freq_list)):
#         for ich in range(len(raw.ch_names)):
#             l_power[ich,idx_freq,:] = scipy.stats.zscore(l_power[ich,idx_freq,:], axis=None)
#     power[band] = np.mean(l_power, axis=1)

# # Collect all possible ripples
# threshold = 5
# event_id = {}
# all_clusters = np.zeros((len(raw.ch_names), len(raw.times)))
# onsets, durations, descriptions = [], [], []
# for ich, ch in enumerate(raw.ch_names):
#     cleaned = np.zeros((len(power.keys()), len(raw.times)))
#     for iband, band in enumerate(power.keys()):
#         cond = power[band][ich,:] > (threshold if band != "middle" else threshold)
#         clusters, n_clusters = ndimage.label(cond)

#         for cli in range(1,n_clusters+1):
#             cl = np.where(clusters == cli)[0]
#             if len(cl) > 12:  #This means over 25ms
#                 cleaned[iband, cl] = 1
    
#     # Keep events with power in both low theta and vhgamm, but NOT in the middle to avoid broadband
#     clusters, n_clusters = ndimage.label((cleaned[0,:] + cleaned[2,:] + (1 - cleaned[1,:])) == 3)
#     all_clusters[ich,:] = (1+ich)*clusters
#     for cli in range(1, n_clusters+1):
#         cl = np.where(clusters == cli)[0]
#         onsets.append(cl[0])
#         durations.append(len(cl))
#         descriptions.append(f"ripple_{ch}")
#         if f"ripple_{ch}" not in event_id:
#             event_id[f"ripple_{ch}"] = 256 + ich

# # Make new annotations
# annot = mne.Annotations(onset=[x/500 for x in onsets], duration=[x/500 for x in durations], description=descriptions)
# raw.set_annotations(annot)
# events, event_id = mne.events_from_annotations(raw, event_id=event_id)

# # Recover existing events / annotations
# epochs = mne.read_epochs(f"../bids_data/{sub}/ieeg/{sub}-epo.fif", preload=False)
# all_events = np.concatenate([events, epochs.events])
# full_dict = {**event_id, **epochs.event_id}

# # Merge, add to raw, and save
# event_desc = {i:k for k,i in full_dict.items()}
# merged_anot = mne.annotations_from_events(all_events, raw.info["sfreq"], event_desc=event_desc)
# raw.set_annotations(merged_anot)
# raw.save(f"marked_raw/{sub}-raw.fif", overwrite=True)



# # do some plots to familiarize with the data format
# # e.g. plot timecourses

# signal = data[0].segments[0].analogsignals[0]
# plt.plot(signal.times, signal)
# plt.xlabel(f"Time ({signal.times.units.dimensionality.string})")
# plt.ylabel(f"Membrane potential ({signal.units.dimensionality.string})")


# # A SpikeTrain represents the times of occurrence of action potentials (spikes).
