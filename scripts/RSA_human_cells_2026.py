#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:27:09 2026

@author: Svenja Kuchenhoff

This script does the following:

STATE RSA.
1. for ALL datasets, create a STATE RSA.
    doing this in the SAME WAY as for the fMRI in 3 different brain regions, in a pseud-population:
        1. per ROI (enth, hipp, amyg, OFC, PCC, ACC, mixed)
        2. collapsed across the brain
        3. for temporal lobe (ent+hipp) vs. frontal lobe (ACC+OFC)
    - the goal is to, in the end, have 2 runs of 6 'unique' grids.
    - first, per session, build 6 'unique' grids and 2 runs. 
        if there are more grids, collapse them into 6, splitting up the amount of 
        data as equally as possible.
        if there are more runs of the same grid than 2, also average them/split them with (one of) the 2 runs.
        Ideally, all 6*2 grids contain averages out of equal amounts of data, but it's important to not 'spill' data between runs.
        only consider 'correct' repeats.
    - Per neuron you have, average the 360 timebins across run 1/ run2 for all grids that count as one of the 6.
    - for the same 6*2 grids, create a 'state' model RDM, where 90 = state A, 90:180 = state B, 180:270 = state C, 270:360 = state D
    - concatenate all neurons per ROI/wholebrian/lobes across the 6 configs, per run.
    - compute an across-run RSA: [cross corr], excluding the center-diagonal 
    
    

DSR RSA.
1. based on /Users/xpsy1114/Documents/projects/multiple_clocks/results/config_pivot_table.csv, check for which 
    grid configurations we have THE SAME grids across subjects (the first 7 rows of the df_pivot)
	sessions ['s27', 's28', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's40', 's43', 's44', 's45', 's46', 's49', 's50', 's51', 's53', 's55', 's56', 's57', 's58', 's59', 's60', 's61', 's62', 's63']
    	sequence
    0	3-7-9-5
    1	8-2-6-7
    2	1-9-5-8
    3	4-8-1-3
    4	6-4-2-9
    5	9-1-3-4
    6	7-3-4-2
    7	2-5-7-6
    
    

"""

