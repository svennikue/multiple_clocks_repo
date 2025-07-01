#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 15:00:09 2025
This script calls raw LFP data, either in .ns2 or .ns5 format.
It then extracts sharp-wave-ripple event candidates, based on the following 
specs:
    HFB = [60, 160]
    
    
@author: Svenja Küchenhoff
"""

import fire
import sys
import numpy as np
import pandas as pd
import os
import glob
from neo.io import NeuralynxIO
import neo
from scipy.signal import resample
import mne
import scipy
import math
import scipy.ndimage as ndimage
from joblib import Parallel, delayed

print("ARGS:", sys.argv)


# some fixed parameters
theta = [3,8]
SW = [8,40]
middle = [40, 80]
gamma = [80, 150]
ultra_high_gamma = [150, 250]
# Downsample snippet to twice the highest frequency I will filter for
downsampled_sampling_rate = 2 * ultra_high_gamma[1]

freq_bands_keys = ['theta', 'SW', 'middle', 'hgamma', 'ultra_high_gamma']
freq_bands = {freq_bands_keys[0]: (theta[0], theta[1]), freq_bands_keys[1]: (SW[0],SW[1]), freq_bands_keys[2]: (middle[0],middle[1]), freq_bands_keys[3]: (gamma[0], gamma[1]), freq_bands_keys[4]: (ultra_high_gamma[0], ultra_high_gamma[1])}

Baylor_list = [5,7,8,9,10,11,12,13,14,15,16,18, 19,20,21,22,25,26,27,28,31,32,33,34,35, 36,37,38,43,44,45,46,49, 57,58,59]
Utah_list = [1,2,4,6,17,23,24,29,30,39,41,42,47,48, 52, 53, 54, 55]
UCLA_list = [3, 40, 50, 51, 56, 60]

# these are parameters I may want to adjust later on. For now fix.
length_ripple_in_secs = 0.02
# 'power increase' defined as exceeding 4x standard deviation from power in this band
higher_than_x_stds = 4


def load_behaviour(sesh):
    sub = f"{sesh:02}"
    behaviour_dict = {}
    behaviour_dict['LFP_path'] =  "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
    behaviour_dict['sesh'] = sesh
    # check if on server or local
    if not os.path.isdir(behaviour_dict['LFP_path']):
        print("running on ceph")
        behaviour_dict['LFP_path'] = "/ceph/behrens/svenja/human_ABCD_ephys"
    
    column_names = ['rep_correct', 't_A', 't_B', 't_C', 't_D', 'loc_A', 'loc_B', 'loc_C', 'loc_D', 'rep_overall', 'new_grid_onset', 'session_no', 'grid_no']
    path_to_beh = f"{behaviour_dict['LFP_path']}/derivatives/s{sub}/cells_and_beh/all_trial_times_{sub}.csv"
    # load behaviour that defines my snippets.
    df_beh = pd.read_csv(path_to_beh, header=None)
    df_beh.columns = column_names
    
    # add some rows that help me later on determine where I extracted the respective ripples
    df_beh['labels'] = df_beh.apply(
    lambda row: f"rep_corr_{int(row['rep_correct'])}_rep_all_{int(row['rep_overall'])}_session_{int(row['session_no'])}_grid_{int(row['grid_no'])}",
    axis=1)
    
    # Filter to keep only grids that were fully completed with 10 correct repeats
    df_beh = df_beh[df_beh.groupby('grid_no')['rep_correct'].transform(lambda x: (x == 9).any())]
    behaviour_dict['beh'] = df_beh
    print(f"loaded behavioural file: {path_to_beh}")
    return(behaviour_dict)


def get_channel_list(reader, ROI): 
    ## extract the channel labels I want in this analysis, e.g. only hippocampal electrodes
    # note this only works for Baylor datasets
    channel_names = reader.header['signal_channels']
    channel_names = [str(elem) for elem in channel_names[:]]
    channel_list = [name.split(",")[0].strip("('") for name in channel_names]
    ROI_indices = []
    # only
    for i, channel in enumerate(channel_list):
        if ROI == 'HPC':
            if 'H' in channel and '01' in channel:
                # H = Hippocampus
                ROI_indices.append(i)
        elif ROI == 'ACC':
            if 'C' in channel and '01' in channel:
                # C = Cingulate
                ROI_indices.append(i)   
        elif ROI == 'OFC':
            if 'OF' in channel and '01' in channel:
                # OF = orbitoFrontal Cortex
                ROI_indices.append(i) 
    ROI_channels = [channel_list[i] for i in ROI_indices]
    # I could here load the electrods*.csv to identify only those electrodes that
    # are in grey matter and HPC, and take a closest white-matter one for referencing.
    # alternative: only take the deepest channel (01)
    print(f"extracted all {ROI} channels: {ROI_channels}")
    return ROI_channels, ROI_indices
    
    
    
def pre_prepare_LFP_dataset(subj_beh):
    #import pdb; pdb.set_trace()
    ## find file path, file type and sampling rate to load. hard-coded by location ##
    # depending on where the data comes from, treat the data differently.
    lfp_dir = os.path.join(subj_beh['LFP_path'], f"s{subj_beh['sesh']:02}")
    if subj_beh['sesh'] in UCLA_list:
        data_from = 'UCLA'
        # it's stored in Neuralynx .ncs files, separately for each macro contact.
        # load the *_localizations.csv as pd.df
        nsx_to_load = 's' # files are .ncs
        # look at 1) electrode name, isMicro = FALSE, 2) grayWhite 3)aparc+aseg or 4) ashs_border
        # hippocampal electrodes have '**H-*' as electrode name, and usually also the parcellation in ashs_border
        # hippocampal contacts will always be 1,2 or 3 (= most medial)
        # take the corresponding electrode name that has is White as a reference
        # then load the respective file with the pattern f"*_{electrode}_{contact_no}.ncs"
        # hopefully using something like this:
        # reader = NeuralynxIO(dirname='path_to_data/')
        # block = reader.read_block()
        # still check the sampling rate!! chat gpt says something like
        sampling_rate_Hz = 1*1000
        # Path to the directory containing the .ncs files
        # reader = NeuralynxIO(dirname='path/to/data')
        
        # # Load one signal block (you can specify filename filters if needed)
        # block = reader.read_block(signal_group_mode='split-all')
        
        # # Access the analog signals
        # seg = block.segments[0]
        # analog_signal = seg.analogsignals[0]
        
        # # Get sampling rate
        # sampling_rate = analog_signal.sampling_rate
        # print(f"Sampling rate: {sampling_rate} Hz")
        
    if subj_beh['sesh'] in Utah_list:
        data_from = 'Utah'
        # Utah saved everyting in .ns2
        # this corresponds to 1kHz
        sampling_rate_Hz = 1*1000
        nsx_to_load = 2
        # ignore these for now. In order to know which electrode channels they are,
        # I first need to read stuff out from the .mat files online.

    if subj_beh['sesh'] in Baylor_list:
        data_from = 'Baylor'
        # Baylor saved everyting in .ns3
        # sampled at 2000 Hz (+ digital filters)
        # Baylor also just sometimes split the session into two halves.
        # this is saved in the behavioural file informatin (session 1 or 2)
        sampling_rate_Hz = 2*1000
        nsx_to_load = 3
        
    lfp_files= glob.glob(os.path.join(lfp_dir, f"*.ns{nsx_to_load}"))
    print(f"current session was recorded in {data_from}")
    return data_from, nsx_to_load, lfp_files, sampling_rate_Hz


def load_lazy_LFP_snippet(ROI, data_from, nsx_to_load, lfp_files, sampling_rate_Hz, sample_idx_start, sample_idx_end):
    # import pdb; pdb.set_trace()
    ## load the channels and data of a LFP snippet ##
    # for now, only single session ones from Baylor work.
    if data_from == 'Baylor':
        # https://neo.readthedocs.io/en/0.3.3/io.html
        reader = neo.io.BlackrockIO(filename=lfp_files[0], nsx_to_load=nsx_to_load)
        channel_list, channel_indices = get_channel_list(reader, ROI)
    if len(lfp_files) == 1:
        raw_file_lazy = reader.read_segment(seg_index=1, lazy=True)
        # just in case this was the wrong segment index.
        
        if raw_file_lazy.t_stop < 20:
            raw_file_lazy = reader.read_segment(seg_index=0, lazy=True)
        
        load_sample = 0
        num_samples = raw_file_lazy.analogsignals[load_sample].shape
        if num_samples[0] < 100:
            load_sample = 1
            
        # to decrease memory load only load segments
        raw_analog_cropped = raw_file_lazy.analogsignals[load_sample].load(time_slice = (sample_idx_start, sample_idx_end), channel_indexes = channel_indices)
        # Calculate the number of samples in the downsampled data
        num_samples = int(raw_analog_cropped.shape[0] * (downsampled_sampling_rate / sampling_rate_Hz))
        # Downsample the data and delete the big one
        downsampled_data = resample(raw_analog_cropped.magnitude, num_samples, axis=0)
        downsampled_analog_epo_cropped = downsampled_data.T.reshape(1,downsampled_data.shape[1], downsampled_data.shape[0])
        
        del raw_analog_cropped
        
        # CURRENTLY NOT IMPLEMENTED: referencing.
        # if referenced_data == True:
        #     #referenced_data, new_channels = mc.analyse.ripple_helpers.reference_electrodes(downsampled_data, channels_to_use)
        #     downsampled_data, channels_to_use = mc.analyse.ripple_helpers.reference_electrodes(downsampled_data, channels_to_use_in_task, repeat)
        # else:
        #     channels_to_use = channels_to_use_in_task 
        
        
        # probably will run into problems. build this in again some time
        # if len(downsampled_data) < 8*ultra_high_gamma[1]:
        #     print(f"Skipping task {task_to_check} repeat {repeat}. too short. only {len(downsampled_data)} samples.")
        #     continue
        
        # CAREFUL! if there is more than 1 block, then I need to do something like
        # if if len(lfp_files) > 1:
        # overall_time_block_one = len(block_0)
        # sample_idx_start = int((row['new_grid_onset']-overall_time_block_one) * sampling_rate_Hz)
    # and I believe for Neuralynx every channel is stored separately so that's going to be completely different again
    # Path to the directory containing the .ncs files
    # reader = NeuralynxIO(dirname='path/to/data')
    return downsampled_analog_epo_cropped, channel_list, channel_indices
 
       
def time_frequ_rep_morlet_one_rep(LFP, channel_indices):
    ## per frequency set, determine time frequency spectra based on morlet transforms ##
    power = {'LFP_mean_power': {}, 'LFP_stepwise_power': {}}
    for band, (l_freq, h_freq) in freq_bands.items():
        step = np.max([1, (h_freq - l_freq) / 20])
        freq_list = np.arange(l_freq, h_freq, step)
        # l_power = mne.time_frequency.tfr_array_morlet(raw_analog_epo_cropped, sampling_freq[block], freqs=freq_list, output="power", n_jobs=-1).squeeze()
        # avoid n_jobs > 1 to avoid problems of nested parallelisms
        l_power = mne.time_frequency.tfr_array_morlet(LFP, downsampled_sampling_rate, freqs=freq_list, output="power", n_jobs = 1).squeeze()
        for idx_freq in range(len(freq_list)):
            for channel_idx in range(len(channel_indices)):
                l_power[channel_idx,idx_freq,:] = scipy.stats.zscore(l_power[channel_idx,idx_freq,:], axis=None)
        power['LFP_mean_power'][band] = np.mean(l_power, axis=1)
        power['LFP_stepwise_power'][band] = l_power
    return power


def extract_ripple_from_one_rep(mean_power, channel_indices, onset_in_sec):
    ## Collect all possible ripples for the current snippet ##
    import pdb; pdb.set_trace()
    min_length_ripple = math.ceil(length_ripple_in_secs*downsampled_sampling_rate)
    all_clusters = np.zeros((len(channel_indices), mean_power['theta'].shape[1]))
    # all_clusters = np.zeros((len(channel_indices_to_use), raw_analog_epo_cropped.shape[-1]))
    onsets, durations, channels_of_curr_ripples, band_order = [], [], [], []
    for new_channel_idx, channel_name in enumerate(channel_indices):
        cluster_bin = np.zeros((len(mean_power.keys()), mean_power['theta'].shape[1]))
        for iband, band in enumerate(mean_power.keys()):
            band_order.append(band) # just to check this maintains the intended order
            # set this to exceeding e.g. 4x standard deviation from power in this band 
            threshold_hl = np.mean(mean_power[band][new_channel_idx,:]) + higher_than_x_stds * np.std(mean_power[band][new_channel_idx,:])
            cond = mean_power[band][new_channel_idx,:] > threshold_hl 
            # define everything as an event, if it's higher than threshold in theta and uh gamma, and lower in the rest.
            # cond = (mean_power[band][new_channel_idx, :] > threshold_hl) if band in ["theta", "ultra_high_gamma"] else (mean_power[band][new_channel_idx, :] < threshold_hl)
            # cond is a boolean array which is true in case the power is higher than the threshold
            # each time there are more than 1 'trues' next to each other, this will be called a cluster.
            # the first cluster will be marked by 1s, the second one by 2s, the third one by 3s,.. etc.
            clusters, n_clusters = ndimage.label(cond)
            for cluster_idx in range(1,n_clusters+1): #+ 1 bc it starts counting at 0 but we want start at 1
                # check how long each cluster is in samples 1/freq * len = secs
                # according to ripple bible paper, clusters need to be 15 ms or more
                # Yunzeh: 20 ms to 200 ms, with a 30 ms interval between events
                curr_cluster = np.where(clusters == cluster_idx)[0]
                if len(curr_cluster) >= min_length_ripple:
                    # include the gap of at least one ripple's length
                    if curr_cluster[0] >= min_length_ripple:
                        gap_before = np.all(cond[curr_cluster[0] - min_length_ripple:curr_cluster[0]] == 0)
                    else:
                        gap_before = False
                    # check for gap after cluster
                    if len(cond) - curr_cluster[-1] - 1 >= min_length_ripple:
                        gap_after = np.all(cond[curr_cluster[-1] + 1:curr_cluster[-1] + 1 + min_length_ripple] == 0)
                    else:
                        gap_after = False
                    # Only consider clusters with sufficient gaps on both sides
                    if gap_before and gap_after:
                        cluster_bin[iband, curr_cluster] = 1

        # # either keep all events that are at the same time higher than threshold for
        # # theta and ultra high gamma and not for middle
        # high_theta_and_gamma = (
        #             (cluster_bin[0, :] == 1) &
        #             (cluster_bin[2, :] == 0) &
        #             (cluster_bin[3, :] == 1) &
        #         )
        
        # clusters, n_clusters = ndimage.label(high_theta_and_vhgamma)
        
        # or just care about no broadband and ripple band (80-120Hz)
        low_middle_high_gamma = (
                    (cluster_bin[2, :] == 0) &
                    (cluster_bin[3, :] == 1)
                )
        clusters, n_clusters = ndimage.label(low_middle_high_gamma)
        # if I want to, I can here also change it to 'high frequency broadband events"
        
        # save all clusters per channel.
        all_clusters[new_channel_idx,:] = (1+new_channel_idx)*clusters 
        for cluster_idx in range(1, n_clusters+1):
            # determine how long a ripple is 
            curr_cluster_combobands = np.where(clusters == cluster_idx)[0]
            # only keep if it is still long enough!
            if len(curr_cluster_combobands) >= min_length_ripple:
                # directly convert back from samples to seconds for readability
                onsets.append(curr_cluster_combobands[0]/downsampled_sampling_rate + onset_in_sec) #onset is in samples, not seconds.
                # onsets.append(cl[0] + sec_lower*sampling_freq[0]) #onset is in samples, not seconds.
                durations.append(len(curr_cluster_combobands)/downsampled_sampling_rate) #duration is also samples, not seconds.
                channels_of_curr_ripples.append(f"channel_{channel_name}")
                    
    return onsets, durations, channels_of_curr_ripples
    
    
def run_repeat_wise_ripple_detection(row, ROI, dataset, file_type, file_list, sample_rate):
    sample_start = row['new_grid_onset']
    sample_end = row['t_D']
    label = row['labels']
    
    LFP_snippet, channels_OI, channel_idx_OI = load_lazy_LFP_snippet(
        ROI, dataset, file_type, file_list, sample_rate, sample_start, sample_end
    )
    # run a morlet transformation to get power spectra 
    power = time_frequ_rep_morlet_one_rep(LFP_snippet, channel_idx_OI)
    # then extract ripples based on increased power in the right bands
    ripple_onsets, ripple_durations, ripple_channels = extract_ripple_from_one_rep(
        power['LFP_mean_power'], channels_OI, sample_start
    )
    # Build ripple info as a list of dicts
    ripple_entries = [
        {
            'rep_corr': row['rep_correct'],
            'rep_overall': row['rep_overall'],
            'grid_no': row['grid_no'],
            'onset': onset,
            'duration': duration,
            'channels': channels
        }
        for onset, duration, channels in zip(ripple_onsets, ripple_durations, ripple_channels)
    ]

    return label, power, ripple_entries
    

def extract_ripples_from_one_session(session, ROI, save_all = False):
    # first load behaviour
    # import pdb; pdb.set_trace()
    beh_dict = load_behaviour(session)
    
    
    # STOP!
    
    # instead of this, read in the split LFPs.
    # they are stored in timepoints x channels
    # also read in the channel list!
    # then do something like
    # correct_LFP_format = LFP_snippet.T.reshape(1,LFP_snippet.shape[1], LFP_snippet.shape[0])
    
    # because for the morlet transform, it has to be in the format of 
    # (n_epochs, n_chans, n_times)
    
    
    # then collect some specifics on this session (all hospitals store LFPs differently)
    # note that for now, I only wrote a script for the Baylor LFPs.
    dataset, file_type, file_list, sample_rate = pre_prepare_LFP_dataset(beh_dict)
    # Run in parallel
    print("Now starting to detect ripples per task repeat...")
    # results = Parallel(n_jobs=4)(
    #     delayed(run_repeat_wise_ripple_detection)(row, ROI, dataset, file_type, file_list, sample_rate)
    #     for _, row in beh_dict['beh'].iterrows())

    # if not running in parallel
    power_dict, ripple_onsets, ripple_durations, ripple_channels = {}, {}, {}, {}
    for idx, row in beh_dict['beh'].iterrows():
        
        sample_start = row['new_grid_onset']
        sample_end = row['t_D']
        LFP_snippet, channels_OI, channel_idx_OI = load_lazy_LFP_snippet(ROI, dataset, file_type, file_list, sample_rate, sample_start, sample_end)
        # run a morlet transformation to get power spectra 
        power_dict[row['labels']] = time_frequ_rep_morlet_one_rep(LFP_snippet, channel_idx_OI)
        # then extract ripples based on increased power in the right bands
        ripple_onsets[row['labels']], ripple_durations[row['labels']], ripple_channels[row['labels']] = extract_ripple_from_one_rep(power_dict[row['labels']]['LFP_mean_power'], channels_OI, sample_start)
    
    import pdb; pdb.set_trace() 
    
    # Unpack the results
    power_dict = {}
    ripple_rows = []
    
    for label, power, ripple_entries in results:
        power_dict[label] = power
        ripple_rows.extend(ripple_entries)
    
    # Create DataFrame from ripple info
    ripple_df = pd.DataFrame(ripple_rows)
    if save_all == True:
        source_dir = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group"
        if not os.path.isdir(source_dir):
            print("running on ceph")
            source_dir = "/ceph/behrens/svenja/human_ABCD_ephys/derivatives/group"
        
        ripple_dir = f"{source_dir}/LFP-ripples"
        os.makedirs(ripple_dir, exist_ok=True) 
        print("Saving results in {ripple_dir}")        
        ripple_df.to_csv(f"{ripple_dir}/ripples_s{session:02}.csv", index=False)
    
    print("...Done!") 
    

# # # # if running from command line, use this one!   
# if __name__ == "__main__":
#     #print(f"starting regression for subject {sub}")
#     fire.Fire(extract_ripples_from_one_session)
#     # call this script like
#     # python wrapper_identify_ripples.py 5 --ROI='HCP'



if __name__ == "__main__":
    # For debugging, bypass Fire and call compute_one_subject directly.
    extract_ripples_from_one_session(
        session=5,
        ROI='HPC', #ROI = 'all' # 'mPFC' or 'HPC'
        save_all = False
    )

