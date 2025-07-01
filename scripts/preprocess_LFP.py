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


# write a config file that specifies how the different datasets are stored in different formats

# these are the sessions that belong to the respectively different sessions
Baylor_list = [5,7,8,9,10,11,12,13,14,15,16,18, 19,20,21,22,25,26,27,28,31,32,33,34,35, 36,37,38,43,44,45,46,49, 57,58,59]
Utah_list = [1,2,4,6,17,23,24,29,30,39,41,42,47,48, 52, 53, 54, 55]
UCLA_list = [3, 40, 50, 51, 56, 60]

# downsample everything to the same frequency:
ultra_high_gamma = [150, 250]
# Downsample snippet to twice the highest frequency I will filter for
downsampled_sampling_rate = 2 * ultra_high_gamma[1]



def get_channel_list_baylor_utah(reader, ROI): 
    ## extract the channel labels I want in this analysis, e.g. only hippocampal electrodes
    # note this only works for Baylor datasets
    channel_names = reader.header['signal_channels']
    channel_names = [str(elem) for elem in channel_names[:]]
    channel_list = [name.split(",")[0].strip("('") for name in channel_names]
    
    # I could here load the electrods*.csv to identify only those electrodes that
    # are in grey matter and HPC, and take a closest white-matter one for referencing.
    # alternative: only take the deepest channel (01)
    print(f"extracted all {ROI} channels: {ROI_channels}")
    return ROI_channels, ROI_indices


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

    
    
    

def preprocess_one_session(session, save_all = False):
    # first load behaviour
    # import pdb; pdb.set_trace()
    beh_dict = load_behaviour(session)
    # use the behaviour to store the respective snippets of the task
    # then collect some specifics on this session (all hospitals store LFPs differently)
    # note that for now, I only wrote a script for the Baylor LFPs.
    # then read in a config file instead of the pre_prepare_LFP_dataset function.
    
    
    # NEXT: figure out how to do this!
    # just load a couple of different baylor ones to start with.
    # I think there are also no_of_recordings
    # and which segment index for baylor, probably also for urah. 
    # no idea how it will work for ucla 
    sesh='03', recording_site='ucla', sampling_rate='1000', LFP_file_format='ncs'
    sesh='04', recording_site='utah', sampling_rate='1000', LFP_file_format=2
    sesh='05', recording_site='baylor', sampling_rate='2000', LFP_file_format=3
    
    
    
    
    import pdb; pdb.set_trace() 
    

    
    # dataset, file_type, file_list, sample_rate = pre_prepare_LFP_dataset(beh_dict)
    # Run in parallel
    print("Now starting to detect ripples per task repeat...")
    # results = Parallel(n_jobs=4)(
    #     delayed(run_repeat_wise_ripple_detection)(row, ROI, dataset, file_type, file_list, sample_rate)
    #     for _, row in beh_dict['beh'].iterrows())

    # if not running in parallel
    for idx, row in beh_dict['beh'].iterrows():
        sample_start = row['new_grid_onset']
        sample_end = row['t_D']
        # modify this
        store_LFP_snippet(ROI, dataset, file_type, file_list, sample_rate, sample_start, sample_end)
    import pdb; pdb.set_trace() 
    print("...Done!") 
    
        
        
        
    

# # # # if running from command line, use this one!   
# if __name__ == "__main__":
#     #print(f"starting regression for subject {sub}")
#     fire.Fire(preprocess_one_session)
#     # call this script like
#     # python preprocess_LFP.py 5 --save_all='TRUE'



if __name__ == "__main__":
    # For debugging, bypass Fire and call preprocess_one_session directly.
    preprocess_one_session(
        session=5,
        save_all = False
    )