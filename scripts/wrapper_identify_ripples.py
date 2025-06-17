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
import mc
import numpy as np
import pandas as pd
import os
import glob
from neo.io import NeuralynxIO



print("ARGS:", sys.argv)

def load_behaviour(sesh):
    sesh = f"{sesh:02}"
    behaviour_dict = {}
    behaviour_dict['LFP_path'] =  "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
    # check if on server or local
    if not os.path.isdir(behaviour_dict['LFP_path']):
        print("running on ceph")
        behaviour_dict['LFP_path'] = "/ceph/behrens/svenja/human_ABCD_ephys"
    
    column_names = ['rep_correct', 't_A', 't_B', 't_C', 't_D', 'loc_A', 'loc_B', 'loc_C', 'loc_D', 'rep_overall', 'new_grid_onset', 'session_no', 'grid_no']
    path_to_beh = f"{behaviour_dict['LFP_path']}/derivatives/s{sesh}/cells_and_beh/all_trial_times_{sesh}.csv"
    # load behaviour that defines my snippets.
    df_beh = pd.read_csv(path_to_beh, header=None)
    df_beh.columns = column_names
    behaviour_dict['beh'] = df_beh
    behaviour_dict['sesh'] = sesh
    
    # feedback = np.genfromtxt(f"{behaviour_dict['LFP_path']}/{sesh}/feedback.csv", delimiter=',')
    # also collect grid_index (task_config) to keep track if you're still in the same grid.
    #seconds_lower, seconds_upper, task_config, task_index, task_onset, new_grid_onset, found_first_D = mc.analyse.ripple_helpers.prep_behaviour(behaviour_all)
    
    return(behaviour_dict)


    
def load_all_LFPs_lazy(subj_beh):
    # depending on where the data comes from, treat the data differently.
    Baylor_list = [5,7,8,9,10,11,12,13,14,15,16,18, 19,20,21,22,25,26,27,28,31,32,33,34,35, 36,37,38,43,44,45,46,49, 57,58,59];
    Utah_list = [1,2,4,6,17,23,24,29,30,39,41,42,47,48, 52, 53, 54, 55]
    UCLA_list = [3, 40, 50, 51, 56, 60]

    
    if subj_beh['sesh'] in UCLA_list:
        # it's stored in Neuralynx .ncs files, separately for each macro contact.
        # load the *_localizations.csv as pd.df
        
        lfp_pattern = glob.glob(os.path.join(f"{subj_beh['LFP_path']}/{subj_beh['sesh']}/micros_and_macros", '*.ncs'))
        
        # look at 1) electrode name, isMicro = FALSE, 2) grayWhite 3)aparc+aseg or 4) ashs_border
        # hippocampal electrodes have '**H-*' as electrode name, and usually also the parcellation in ashs_border
        # hippocampal contacts will always be 1,2 or 3 (= most medial)
        # take the corresponding electrode name that has is White as a reference
        # then load the respective file with the pattern f"*_{electrode}_{contact_no}.ncs"
        # hopefully using something like this:
        # reader = NeuralynxIO(dirname='path_to_data/')
        # block = reader.read_block()
        # still check the sampling rate!! chat gpt says something like
        sampling_rate_kHz = 1
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
        # Utah saved everyting in .ns2
        # this corresponds to 1kHz
        sampling_rate_kHz = 1
        lfp_pattern = glob.glob(os.path.join(f"{subj_beh['LFP_path']}/{subj_beh['sesh']}/", '*.ns2'))
        
        
        
        
        
    if subj_beh['sesh'] in Baylor_list:
        # Baylor saved everyting in .ns3
        # sampled at 2000 Hz (+ digital filters)
        sampling_rate_kHz = 2
        lfp_pattern = glob.glob(os.path.join(f"{subj_beh['LFP_path']}/{subj_beh['sesh']}/", '*.ns3'))
        # Baylor also just sometimes split the session into two halves.
        # this is saved in the behavioural file informatin (session 1 or 2)
        
        



    ns_paths = []
    # first check if there are any .ns3 files

    # then test if there are any .ns2 files
    for file in ns_pattnern:
        ns_paths.append(file)
        # depending on this, collect the different basenames.
        
    
    # preparing the file
    raw_file_lazy, HC_channels, HC_indices, mPFC_channels, mPFC_indices, orig_sampling_freq, block_size, ROI_dict, all_ROI_list, all_ROI_indices = mc.analyse.ripple_helpers.load_LFPs(LFP_dir, sub, names_blks_short, channel_list_complete=True)
    # redefine the lazy loader with every loop and see if that decreases memory load!!
    
    
    raw_analog_cropped = raw_file_lazy[block].analogsignals[0].load(time_slice = (sec_lower_neuro, sec_upper_neuro), channel_indexes = channel_indices_to_use)
    
    
    
    
    # basically, test for this pattern 1), collect all paths that go like this,
    # sort them by session, and then load them all in the lazy way to not use up too much.
    
    # depending on this, downsample differently!
    
    
    # sometimes, there will be several sessions.
    # Initialize variables for the two files
    
    
    # file_blk_01 = None
    # file_blk_02 = None
    
    # # Loop through files and find the ones with 'blk-01' and 'blk-02'
    # for file in ns3_files:
    #     if 'blk-01' in file:
    #         file_blk_01 = os.path.splitext(os.path.basename(file))[0]  # Remove path and '.ns3'
    #     elif 'blk-02' in file:
    #         file_blk_02 = os.path.splitext(os.path.basename(file))[0]  # Remove path and '.ns3'
    
    # # Create the final list in the specified order
    # names_blks_short = [file_blk_01, file_blk_02]
    
    
    
    # test if this is .ns3 or .ns2
    # reader = []
    # reader.append(neo.io.BlackrockIO(filename=f"{LFP_dir}/{sub}/{names_blks_short[file_half]}", nsx_to_load=3))
    # return LFP, 
    
    
    

def extract_ripples_from_one_session(session, ROI):
    beh_dict = load_behaviour(session)
    
    # ok so what I think is happening is I need to somehow figure out how top piece the potentially split 
    # sessions together.
    for idx, row in beh_dict['beh'].iterrows():
        start_trial_at = row['new_grid_onset']
        end_trial_at = row['t_D']
        
        import pdb; pdb.set_trace()
        
        # TODO:
            # figure out how this went again.
            # goal is to only ever have the LFP data loaded from one repeat, bc I can then just go throuhg that snippet and define all SWR event candidates.
            # there is another dimension of all channels.
            # So I need two loops: 1) per repeat, 2) per channel
            # goal: have one pd df (?) where I store infos about
            # which task, which repeat, which time, and ripple event power etc, and channel.
            # loop outer: time. loop inner: channel.
                # for each channel also clean data with reference subtraction
            # load each LFP snippet iteratively, and store all candidate events.
        
        data = load_all_LFPs_lazy(beh_dict)
        
        
    
    # but then, I can just load whatever snippet I am currently looking at, instead of loading the entire thing
    # understand these split sessions better.
    # how do I store the behavioural stuff?
    # maybe all of that can be simplified.
    
    
    # reader, raw_file_lazy = [], []
    # if sub not in ['s5']:
    #     for file_half in [0,1]:
    #         # does neo.io have an 'unload' function?
    #         reader.append(neo.io.BlackrockIO(filename=f"{LFP_dir}/{sub}/{names_blks_short[file_half]}", nsx_to_load=3))
    #         if (sub in ['s11'] and file_half == 0) or (sub in ['s16', 's18'] and file_half == 1):
    #             raw_file_lazy.append(reader[file_half].read_segment(seg_index=0, lazy=True))
    #         else:
    #             raw_file_lazy.append(reader[file_half].read_segment(seg_index=1, lazy=True))
    # else:
    #     for file_half in [0]:
    #         # does neo.io have an 'unload' function?
    #         reader.append(neo.io.BlackrockIO(filename=f"{LFP_dir}/{sub}/{names_blks_short[file_half]}", nsx_to_load=3))
    #         raw_file_lazy.append(reader[file_half].read_segment(seg_index=1, lazy=True))

    # # redefine the lazy loader with every loop and see if that decreases memory load!!
    # raw_analog_cropped = raw_file_lazy[block].analogsignals[0].load(time_slice = (sec_lower_neuro, sec_upper_neuro), channel_indexes = channel_indices_to_use)
    
            

    
    
    

# # # # if running from command line, use this one!   
# if __name__ == "__main__":
#     #print(f"starting regression for subject {sub}")
#     fire.Fire(extract_ripples_from_one_session)
#     # call this script like
#     # python wrapper_identify_ripples.py 5 --ROI='HCP'



if __name__ == "__main__":
    # For debugging, bypass Fire and call compute_one_subject directly.
    extract_ripples_from_one_session(
        session=2,
        ROI='HCP' #ROI = 'all' # 'mPFC' or 'HPC'
    )

