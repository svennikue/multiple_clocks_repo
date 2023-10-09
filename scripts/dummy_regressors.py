#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:00:02 2023

@author: xpsy1114
"""

import numpy as np

repeats = 5;
subpath_steps = [2, 2, 2, 2]
phases = 2;
min_jitter = 1.1;
max_jitter = 2.2;
reward_min_jitter = 1.1;
reward_max_jitter = 3.3
onsets = np.zeros((repeats, len(subpath_steps)*phases))

time = 0
for r in range(repeats):
    for subpath, curr_steps in enumerate(subpath_steps):        
        subpath_jitter = min_jitter + (max_jitter - min_jitter) * np.random.rand(curr_steps)
        phase_times = time + np.array(range(phases))*sum(subpath_jitter)/(phases-1)
        for curr_phase, curr_time in enumerate(phase_times):
            onsets[r,subpath*phases + curr_phase] = curr_time
        time = time + sum(subpath_jitter) + reward_min_jitter + (reward_max_jitter - reward_min_jitter) * np.random.rand()
            
for curr_regressor in range(onsets.shape[1]):
    regressor_matrix = np.ones((repeats,3))
    regressor_matrix[:,0] = onsets[:, curr_regressor]
    print(regressor_matrix)
    np.savetxt('ev_' + str(curr_regressor) + '.txt', regressor_matrix, delimiter="    ", fmt='%f')

            