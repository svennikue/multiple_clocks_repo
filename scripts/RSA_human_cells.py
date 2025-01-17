#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 14:25:03 2024

this script runs the musicbox RSA on human cells, treating all subjects as the one.

@author: Svenja Küchenhoff

##Location npy files:
These contain the location of the animal in each bin (should correspond exactly to the neuron bins)
locations1-9 are the 9 nodes

##Neuron_raw arrays are matrices of shape neurons X bins
each bin is the firing rate in a 50 ms timewindow
Location_raw arrays are arrays of length equal to the number of bins for the Neuron_raw matrix (may be 1 off)

"""
import numpy as np
import mc
import matplotlib.pyplot as plt

# first, load all csv files as numpys
# exclude 27 and 44 for now
subjects = [28,31,32,33,34,35,36,37,38,40,43,45,46,49,50]
data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/LFP"


locations, cells = mc.analyse.helpers_human_cells.load_cell_data(data_folder, subjects)

# what I need as well is:
    # 1. which cell is at which location?
    # 2. which grid are they solving currently (i.e. where are the rewards)?
    

# Steps: 
    # 1. Simulate neural timecourses based on behaviour
    # 2. Run regression on simulated and real neurons to put into RDMs
    # 3. Predict data RDMs with model RDM
    
# Questions:
    # - How can I make sure to combine all cells of a certain region into 
    #   "ROI RDMs" across subjects?
    #   for this, I need to somehow normalise the behaviour; i.e. how long
    #   subjects spent at each location, where they go etc. Does this work??
    #   might work for the reward locations, but not for the paths. Start with that
    #   This would mean:
    #       1. Per subject, create neural timecourses
    #       2. Run regression such that you have e.g. 4 reward-bins and equal no of repeats
    #       3. then concatenate all cells in the same ROI across all subjects 
    #       4. then compute the RSA for the ROI data RDMs
    


import pdb; pdb.set_trace()

    
# can I recycle this?   
# mc.analyse.analyse_ephys.load_ephys_data(Data_folder = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_recordings_200423/')
