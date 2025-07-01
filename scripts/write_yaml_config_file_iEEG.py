#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:33:09 2025
writing a config.yaml
@author: xpsy1114
"""

import yaml

# Define the session groupings
baylor_sessions = [5,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,25,26,27,28,31,32,33,34,35,36,37,38,43,44,45,46,49,57,58,59]
ucla_sessions = [3, 40, 50, 51, 56, 60]
utah_sessions = [1,2,4,6,17,23,24,29,30,39,41,42,47,48,52,53,54,55]
source_path = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"

# Create a dictionary to hold all session configurations
session_config = {}

for i in range(1, 61):
    key = f"{i:02d}"  # zero-padded session number
    if i in baylor_sessions:
        session_config[key] = {
            'recording_site': 'baylor',
            'sampling_rate': 2000,
            'LFP_file_format': 3,
            'segment': None,
            'n_rec_sessions': None
        }
    elif i in ucla_sessions:
        session_config[key] = {
            'recording_site': 'ucla',
            'sampling_rate': 1000,
            'LFP_file_format': 'ncs',
            'segment': None,
            'n_rec_sessions': None
        }
    elif i in utah_sessions:
        session_config[key] = {
            'recording_site': 'utah',
            'sampling_rate': 1000,
            'LFP_file_format': 2,
            'segment': None,
            'n_rec_sessions': None
        }
    else:
        session_config[key] = {
            'recording_site': None,
            'sampling_rate': None,
            'LFP_file_format': None,
            'segment': None,
            'n_rec_sessions': None
        }

# Save to a YAML file
output_file = f"{source_path}/config_human_ABCD_iEEG.yaml"
with open(output_file, 'w') as file:
    yaml.dump(session_config, file, default_flow_style=False)

output_file
