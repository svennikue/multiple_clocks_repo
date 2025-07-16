#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 11:53:35 2025
This script will take the output of preprocess_LFP.py, which are channels x numpy arrays
per solving the ABCD loop once.
The repeats are stored in the following format:
    lfp_snippet_{sample_start:.2f}-{sample_end:.2f}sec_grid{row['grid_no']}_ABCD_{loc_string}_{downsampled_sampling_rate}_Hz
The channel names are stored in channels.npy
The behaviour is stored in /ceph/behrens/svenja/human_ABCD_ephys/derivatives/s{sesh}/cells_and_beh/all_trial_times_{sesh}.csv

first: load organising data (behaviour, channels)
second: enter a loop in which you
    - first load a single repeat snippet 
    - define all target channels (those that are hippocampal)
    - create the power spectrum
    - extract ripple-events based on the ripple specifications 
@author: xpsy1114
"""

import fire
import sys
import numpy as np
import pandas as pd
import yaml
import pickle
import re
import os
import glob
import mne
import scipy
import math
import scipy.ndimage as ndimage
from joblib import Parallel, delayed

print("ARGS:", sys.argv)


ultra_high_gamma = [150, 250]
# Downsample snippet to twice the highest frequency I will filter for
downsampled_sampling_rate = 2 * ultra_high_gamma[1]
# these are parameters I may want to adjust later on. For now fix.
length_ripple_in_secs = 0.02
# 'power increase' defined as exceeding 4x standard deviation from power in this band
higher_than_x_stds = 4


def load_behaviour(sesh):
    behaviour_dict = {}
    behaviour_dict['LFP_path'] =  "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
    behaviour_dict['sesh'] = f"{sesh:02}"
    # check if on server or local
    if not os.path.isdir(behaviour_dict['LFP_path']):
        print("running on ceph")
        behaviour_dict['LFP_path'] = "/ceph/behrens/svenja/human_ABCD_ephys"
    
    column_names = ['rep_correct', 't_A', 't_B', 't_C', 't_D', 'loc_A', 'loc_B', 'loc_C', 'loc_D', 'rep_overall', 'new_grid_onset', 'session_no', 'grid_no']
    path_to_beh = f"{behaviour_dict['LFP_path']}/derivatives/s{sesh:02}/cells_and_beh/all_trial_times_{sesh:02}.csv"
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


       
def time_frequ_rep_morlet_one_rep(LFP):
    ## per frequency set, determine time frequency spectra based on morlet transforms ##
    # import pdb; pdb.set_trace()
    # some fixed parameters
    theta = [3,8]
    SW = [8,40]
    middle = [40, 80]
    gamma = [80, 150]

    freq_bands_keys = ['theta', 'SW', 'middle', 'hgamma', 'ultra_high_gamma']
    freq_bands = {freq_bands_keys[0]: (theta[0], theta[1]), freq_bands_keys[1]: (SW[0],SW[1]), freq_bands_keys[2]: (middle[0],middle[1]), freq_bands_keys[3]: (gamma[0], gamma[1]), freq_bands_keys[4]: (ultra_high_gamma[0], ultra_high_gamma[1])}

    power = {'LFP_mean_power': {}, 'LFP_stepwise_power': {}}
    for band, (l_freq, h_freq) in freq_bands.items():
        step = np.max([1, (h_freq - l_freq) / 20])
        freq_list = np.arange(l_freq, h_freq, step)
        # l_power = mne.time_frequency.tfr_array_morlet(raw_analog_epo_cropped, sampling_freq[block], freqs=freq_list, output="power", n_jobs=-1).squeeze()
        # avoid n_jobs > 1 to avoid problems of nested parallelisms
        l_power = mne.time_frequency.tfr_array_morlet(LFP, downsampled_sampling_rate, freqs=freq_list, output="power", n_jobs = 1).squeeze()
        for idx_freq in range(len(freq_list)):
            for channel_idx in range(LFP.shape[1]):
                l_power[channel_idx,idx_freq,:] = scipy.stats.zscore(l_power[channel_idx,idx_freq,:], axis=None)
        power['LFP_mean_power'][band] = np.mean(l_power, axis=1)
        power['LFP_stepwise_power'][band] = l_power
    return power


def extract_ripple_from_one_rep(mean_power, channels, snippet_name):
    ## Collect all possible ripples for the current snippet ##
    onset_in_sec_str = re.search(r'\d+\.\d+', snippet_name).group() # \d+\.\d+ matches a decimal number
    onset_in_sec = float(onset_in_sec_str)   
    snippet_name_split = snippet_name.split("_")
    n_channels = len(channels)
    min_length_ripple = math.ceil(length_ripple_in_secs*downsampled_sampling_rate)
    all_clusters = np.zeros((n_channels, mean_power['theta'].shape[1]))
    # all_clusters = np.zeros((len(channel_indices_to_use), raw_analog_epo_cropped.shape[-1]))
    #onsets, durations, channels_of_curr_ripples, band_order = [], [], [], []
    band_order = []
    ripple_df = pd.DataFrame(columns = ["onset_in_secs", "onset_in_samples", "duration", "channel", "grid_no", "task", "snippet"])
    for new_channel_idx, row in channels.iterrows():
    
    #for new_channel_idx, channel_name in enumerate(channel_indices):
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
                # onsets.append(cl[0] + sec_lower*sampling_freq[0]) #onset is in samples, not seconds.
                new_row = {
                'onset_in_secs': curr_cluster_combobands[0]/downsampled_sampling_rate + onset_in_sec, 
                'onset_in_samples': curr_cluster_combobands[0],
                #duration is also samples, not seconds.
                'duration': len(curr_cluster_combobands)/downsampled_sampling_rate,
                'channel': row['anat_label_01']  ,
                "grid_no": snippet_name_split[1], 
                "task": snippet_name_split[3], 
                "snippet": snippet_name_split[0]
                }
                ripple_df = pd.concat([ripple_df, pd.DataFrame([new_row])], ignore_index=True)
       
    return ripple_df
    



def identify_channels_i_want(config, beh):
    # import pdb; pdb.set_trace()
    channels = np.load(f"{beh['LFP_path']}/derivatives/s{beh['sesh']}/LFP/channels.npy")
    channels_OI = pd.DataFrame(columns = ["anat_label", "ns_label", "ns_index"])
    if config['recording_site'] == 'utah':
        electrode_labels_path = f"{beh['LFP_path']}/derivatives/s{beh['sesh']}/LFP/utah_elec_labels_{beh['sesh']}.csv"
        electrode_labels = pd.read_csv(electrode_labels_path, header=None, names=["labels", "ns_index"])
        for idx, row in electrode_labels.iterrows():
            if 'HIP' in row['labels']:
                if '1' in row['labels'] or '4' in row['labels']:
                    new_row = {
                    "anat_label": row['labels'],
                    "ns_index": row['ns_index'],
                    "ns_label": channels[int(row['ns_index']) - 1]
                    }
                    channels_OI = pd.concat([channels_OI, pd.DataFrame([new_row])], ignore_index=True)

    elif config['recording_site'] == 'baylor':
        for idx, channel in enumerate(channels):
            if 'H'in channel and 'T' in channel:
                if '01-' in channel or '04-' in channel:
                    new_row = {
                    "anat_label": channel,
                    "ns_index": idx,
                    "ns_label": channel
                    }
                    channels_OI = pd.concat([channels_OI, pd.DataFrame([new_row])], ignore_index=True)
    channels_OI['ns_index'] = channels_OI['ns_index'].astype(int)
    return channels_OI



def referencing(lfp, channels):
    # Extract common prefix (before -01 or -04)
    def extract_base(label):
        return re.sub(r'(01|04)-\d+$', '', label)

    # make a new table where each row has channel 01 and 04
    channels['base'] = channels['anat_label'].apply(extract_base)
    # Split into -01 and -04
    df_01 = channels[channels['anat_label'].str.contains('01-')].copy()
    df_04 = channels[channels['anat_label'].str.contains('04-')].copy()
    
    df_01['base'] = df_01['anat_label'].apply(extract_base)
    df_04['base'] = df_04['anat_label'].apply(extract_base)
    
    # Merge on common base
    df_matched = df_01.merge(df_04[['base', 'anat_label', 'ns_index']], on='base', suffixes=('', '_ref'))
    
    # Optional cleanup
    df_matched = df_matched.rename(columns={'anat_label': 'anat_label_01', 'ns_label': 'ns_label_01', 'ns_index': 'ns_index_01',
                                            'anat_label_ref': 'anat_label_04', 'ns_index_ref': 'ns_index_04'})

    # Prepare output array
    referenced_lfp = []
    
    # then go through each matched channel to reference the electrodes (01 being the target)
    for _, row in df_matched.iterrows():
        idx_01 = int(row['ns_index_01'])
        idx_04 = int(row['ns_index_04'])
        
        # Subtract ref channel from main channel
        referenced_signal = lfp[:, idx_01] - lfp[:, idx_04]
        referenced_lfp.append(referenced_signal)
    

    # import pdb; pdb.set_trace()
    # Stack and transpose to get shape (n_times, n_channels) -> (n_channels, n_times)
    referenced_lfp = np.stack(referenced_lfp, axis=1).T  # Now shape is (n_channels, n_times)
    
    # Expand to shape (1, n_channels, n_times) for wavelets
    referenced_lfp = np.expand_dims(referenced_lfp, axis=0)

    return referenced_lfp, df_matched


def extract_ripples_from_one_session(session, save_all = False):
    # import pdb; pdb.set_trace()
    # first load behaviour
    sesh = f"{session:02}"
    beh_dict = load_behaviour(session)
    with open(f"{beh_dict['LFP_path']}/config_human_ABCD_iEEG.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    session_config = config.get(sesh)
    print(f"session is {sesh}, recording site is {session_config['recording_site']}")
    
    hippocampal_channels = identify_channels_i_want(session_config, beh_dict)
    lfp_snippets_paths = glob.glob(f"{beh_dict['LFP_path']}/derivatives/s{beh_dict['sesh']}/LFP/lfp_snippet*500_Hz.npy")
    power_dict = {}
    ripple_df = pd.DataFrame(columns = ["onset_in_secs", "onset_in_samples", "duration", "channel", "grid_no", "task", "snippet"])
    
    print("Now starting to detect ripples per task repeat...")
    for idx, lfp_snippet_path in enumerate(lfp_snippets_paths):
        if idx % 10 == 0:
            print(f"...processing LFP snippet {idx}")
        basename = lfp_snippet_path.split('lfp_snippet_')[-1]
        basename = os.path.splitext(basename)[0]
        lfp_snippet = np.load(lfp_snippet_path)
        # first, reference the central hippocampal contact with a contact
        # further away (04)
        lfp_snippet_ref, matched_channels = referencing(lfp_snippet, hippocampal_channels)
        power_dict[basename] = time_frequ_rep_morlet_one_rep(lfp_snippet_ref)
        ripple_df_curr_snippet = extract_ripple_from_one_rep(power_dict[basename]['LFP_mean_power'],matched_channels, basename)
        ripple_df = pd.concat([ripple_df, ripple_df_curr_snippet], ignore_index=True)
        
    if len(ripple_df)>0:
        print("run and extracted ripple events successfully!")
        
    # Create DataFrame from ripple info
    if save_all == True:
        source_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/s{sesh}"
        
        if not os.path.isdir(source_dir):
            print("storing results on ceph")
            source_dir = f"/ceph/behrens/svenja/human_ABCD_ephys/derivatives/s{sesh}"
        
        ripple_dir = f"{source_dir}/LFP-ripples"
        os.makedirs(ripple_dir, exist_ok=True) 
        print(f"Saving results in {ripple_dir}")        
        ripple_df.to_csv(f"{ripple_dir}/ripples_s{session:02}.csv", index=False)
        # also store the power dict
        with open(os.path.join(ripple_dir,f"ripple_power_dict_s{sesh}"), 'wb') as f:
            pickle.dump(power_dict, f)
            
    # import pdb; pdb.set_trace()
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
        save_all = True
    )

