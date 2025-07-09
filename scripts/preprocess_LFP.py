#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 14:11:14 2025

This file is to preprocess all different LFP files and store them in 
numpy arrays that are named with the respective electrode, channel,
task second, repeat and ABCD- locations to read them in the same 
way across recording sites later on.


@author: xpsy1114
"""
import fire
import os
import pandas as pd
import yaml
import glob
import neo
from scipy.signal import resample
import numpy as np
import sys
import re
import warnings


print("ARGS:", sys.argv)

# downsample everything to the same frequency:
ultra_high_gamma = [150, 250]
# Downsample snippet to twice the highest frequency I will filter for
downsampled_sampling_rate = 2 * ultra_high_gamma[1]



def get_channel_list_baylor_utah(reader): 
    ## extract the channel labels I want in this analysis, e.g. only hippocampal electrodes
    # note this only works for Baylor datasets
    channel_names = reader.header['signal_channels']
    channel_names = [str(elem) for elem in channel_names[:]]
    channel_list = [name.split(",")[0].strip("('") for name in channel_names]
    
    # I could here load the electrods*.csv to identify only those electrodes that
    # are in grey matter and HPC, and take a closest white-matter one for referencing.
    # alternative: only take the deepest channel (01)
    # print(f"extracted all {ROI} channels: {ROI_channels}")
    return channel_list




def store_LFP_snippet(file, config, start, end, save_at=False, file_name=False, block_one=False):
    if config['recording_site'] == 'baylor' or config['recording_site'] == 'utah':
        # https://neo.readthedocs.io/en/0.3.3/io.html
        reader = neo.io.BlackrockIO(filename=file, nsx_to_load=config['LFP_file_format'])
        channel_list = get_channel_list_baylor_utah(reader)
      
    if save_at:
        filepath = os.path.join(save_at, 'channels.npy')
        if not os.path.exists(filepath):
            np.save(filepath, np.array(channel_list))
            print(f"Saved channel list to {filepath}")
            
    raw_file_lazy = reader.read_segment(seg_index=0, lazy=True)
    import pdb; pdb.set_trace() 
    if raw_file_lazy.t_stop < 20:
        raw_file_lazy = reader.read_segment(seg_index=1, lazy=True)
        print("Loading second segment instead, first one was shorter than 20 samples!")
    
    len_block_one = []
    if block_one == True:
        len_block_one = raw_file_lazy.t_stop.magnitude
    
    # TAKE THE SEG_INDEX ALSO FROM INDEX!
    # ALSO THE analogsignals n...    
    # just in case this was the wrong segment index.
    # check if the segment thingy works like this! for some sessions the segment is different.
    # if (sub in ['s11'] and file_half == 0) or (sub in ['s18'] and file_half in [1, 2]):
    #     block_size.append(reader[file_half].get_signal_size(seg_index=0, block_index=0))
    # else:
    #     block_size.append(reader[file_half].get_signal_size(seg_index=1, block_index=0))

    load_sample = 0
    num_samples = raw_file_lazy.analogsignals[load_sample].shape
    if num_samples[0] < 100:
        load_sample = 1

    if start == 0:
        start = float(raw_file_lazy.t_start.magnitude)
    # to decrease memory load only load segments
    raw_analog_cropped = raw_file_lazy.analogsignals[load_sample].load(time_slice = (start, end))
    # raw_analog_cropped = raw_file_lazy.analogsignals[load_sample].load(time_slice = (sample_idx_start, sample_idx_end))
    # Calculate the number of samples in the downsampled data
    num_samples = int(raw_analog_cropped.shape[0] * (downsampled_sampling_rate / config['sampling_rate']))
    # Downsample the data and delete the big one
    downsampled_data = resample(raw_analog_cropped.magnitude, num_samples, axis=0)
    # store as timepoints x channels!
    # also store the channel list.
    if save_at:
        if file_name:
            np.save(os.path.join(save_at, f"{file_name}.npy"), downsampled_data)
    
    return len_block_one




def load_behaviour(sesh):
    behaviour_dict = {}
    behaviour_dict['LFP_path'] =  "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
    behaviour_dict['sesh'] = sesh
    # check if on server or local
    if not os.path.isdir(behaviour_dict['LFP_path']):
        print("running on ceph")
        behaviour_dict['LFP_path'] = "/ceph/behrens/svenja/human_ABCD_ephys"
    
    column_names = ['rep_correct', 't_A', 't_B', 't_C', 't_D', 'loc_A', 'loc_B', 'loc_C', 'loc_D', 'rep_overall', 'new_grid_onset', 'session_no', 'grid_no']
    path_to_beh = f"{behaviour_dict['LFP_path']}/derivatives/s{sesh}/cells_and_beh/all_trial_times_{sesh}.csv"
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

    
def sort_block_files(files, blocks):
    # Extract EMU number
    def extract_emu(file):
        match = re.search(r'EMU-(\d+)', os.path.basename(file))
        return int(match.group(1)) if match else float('inf')
    
    # Sort by EMU number
    sorted_pairs = sorted(zip(files, blocks), key=lambda x: extract_emu(x[0]))
    
    # Separate the sorted lists
    sorted_files = [f for f, _ in sorted_pairs]
    sorted_blocks = [b for _, b in sorted_pairs]
    import pdb; pdb.set_trace()
    # Check if blocks are sorted as well
    if sorted_blocks != sorted(sorted_blocks):
        warnings.warn("Block order does not match EMU number order. Proceeding with EMU-sorted files anyway.")

    return sorted_files
  


  
def preprocess_one_session(session, save_all = False):
    # first load behaviour
    # import pdb; pdb.set_trace()
    session_id = f"{session:02}"
    beh_dict = load_behaviour(session_id)
    path_to_save = f"{beh_dict['LFP_path']}/derivatives/s{session_id}/LFP"
    os.makedirs(path_to_save, exist_ok=True) 
    if save_all == False:
        path_to_save = False
    # use the behaviour to store the respective snippets of the task
    # then collect some specifics on this session (all hospitals store LFPs differently)
    # note that for now, I only wrote a script for the Baylor LFPs.
    # Load the full YAML file
    with open(f"{beh_dict['LFP_path']}/config_human_ABCD_iEEG.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    session_config = config.get(session_id)
    print(f"session is {session_id}, recording site is {session_config['recording_site']}")
    
    if session_config['recording_site'] == 'baylor' or session_config['recording_site'] == 'utah':
        lfp_files= glob.glob(os.path.join(f"{beh_dict['LFP_path']}/s{session_id}/LFP", f"*.ns{session_config['LFP_file_format']}"))
        print(f"found n = {len(lfp_files)} lfp files in the folder {beh_dict['LFP_path']}/s{session_id}/LFP")
    if len(lfp_files) == 1:
        print(f"Now starting to load and downsample LFP data to {downsampled_sampling_rate} Hz per task repeat...")
        for idx, row in beh_dict['beh'].iterrows():
            if idx % 10 == 0:
                print(f"Processing LFP snippet {idx}")
            sample_start = row['new_grid_onset']
            sample_end = row['t_D']
            locs = [int(row[f'loc_{l}']) for l in ['A', 'B', 'C', 'D']]
            loc_string = ''.join(str(l) for l in locs)
            lfp_name = f"lfp_snippet_{sample_start:.2f}-{sample_end:.2f}sec_grid{row['grid_no']}_ABCD_{loc_string}_{downsampled_sampling_rate}_Hz"
            store_LFP_snippet(lfp_files[0], session_config, sample_start, sample_end, path_to_save, lfp_name)
     
    elif len(lfp_files) > 1:
        
        # SO THIS PROBABLY JUST NEEDS TO BE SORTED 'alphabetically'.
        # this is because sometimes, there are two 'block 2', but now i can just loop
        # through the config file thingy.
        # lfp_files = sorted(lfp_files, key=lambda x: int(x.split('blk-')[1].split('_')[0]))
        
        lfp_files = sort_block_files(lfp_files, session_config['blocks'])
        import pdb; pdb.set_trace()

        for lfp_file in lfp_files:
            if 'blk-01' in lfp_file:
                print(f"Now starting to load and downsample LFP data to {downsampled_sampling_rate} Hz per task repeat, block 1...")
                # first filter for only block 1.
                curr_block_beh = beh_dict['beh'][beh_dict['beh']['session_no']==1].reset_index(drop=True)
                
                # # also just to check print this
                # block_two_beh = beh_dict['beh'][beh_dict['beh']['session_no']==2]
                # print(block_two_beh.head)
                # print('compare the time here with the length of lfp file one!')
            
                for idx, row in curr_block_beh.iterrows():
                    if idx % 10 == 0:
                        print(f"Processing LFP snippet {idx}, block 1")
                    sample_start = row['new_grid_onset']
                    sample_end = row['t_D']
                    
                    locs = [int(row[f'loc_{l}']) for l in ['A', 'B', 'C', 'D']]
                    loc_string = ''.join(str(l) for l in locs)
                    lfp_name = f"lfp_snippet_{sample_start:.2f}-{sample_end:.2f}sec_grid{row['grid_no']}_ABCD_{loc_string}_{downsampled_sampling_rate}_Hz"
                    first_block_secs = store_LFP_snippet(lfp_file, session_config, sample_start, sample_end, path_to_save, lfp_name, block_one=True)

            elif 'blk-02' in lfp_file:
                 print(f"Now starting to load and downsample LFP data to {downsampled_sampling_rate} Hz per task repeat, block 2...")
                 # first filter for only block 2.
                 curr_block_beh = beh_dict['beh'][beh_dict['beh']['session_no']==2].reset_index(drop=True)
                 # delete this once you are done.
                 # import pdb; pdb.set_trace()
                 print(f"block one was {first_block_secs} secs, next grid repeat at {curr_block_beh['new_grid_onset'].iloc[0]}")
                 print(f"difference is {curr_block_beh['new_grid_onset'].iloc[0] -  first_block_secs} secs, I reset to 0.")
                 for idx, row in curr_block_beh.iterrows():
                     if idx % 10 == 0:
                         print(f"Processing LFP snippet {idx}, block 2")
                     if idx == 0: 
                         sample_start = 0
                     else:
                         sample_start = row['new_grid_onset'] - first_block_secs
                         # I don't really know what is correct here.
                         # I think that the first grid in block 2 basically runs right from 
                         # when block 1 ends, i.e. 0 seconds will be right in the file.

                     sample_end = row['t_D'] - first_block_secs
                     locs = [int(row[f'loc_{l}']) for l in ['A', 'B', 'C', 'D']]
                     loc_string = ''.join(str(l) for l in locs)
                     lfp_name = f"lfp_snippet_{sample_start:.2f}-{sample_end:.2f}sec_grid{row['grid_no']}_ABCD_{loc_string}_{downsampled_sampling_rate}_Hz"
                     _ = store_LFP_snippet(lfp_file, session_config, sample_start, sample_end, path_to_save, lfp_name)        

    print("...Done!") 
    
    

# # if running from command line, use this one!   
# if __name__ == "__main__":
#     fire.Fire(preprocess_one_session)
#     # call this script like
#     # python preprocess_LFP.py 5 --save_all='TRUE'



if __name__ == "__main__":
    # For debugging, bypass Fire and call preprocess_one_session directly.
    preprocess_one_session(
        session=18,
        save_all = False
    )
    
    
    
    