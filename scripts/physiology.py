#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:27:28 2023
This script plots the physiology file that I collected for the first participant.
@author: xpsy1114
"""
import pandas as pd
import matplotlib.pyplot as plt

file_path = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/biopack/without_header_Karan.txt"
file_two = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/biopack/physio_input.txt"

df = pd.read_csv(file_two, sep ="\t")

selected_df = df.iloc[160000:180000, :]
selected_df.columns = ['miliseconds', 'Respiration', 'Pulse', 'Trigger', 'empty']

# Create a figure with three subplots (one for each column)
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 6))

# Iterate through each column and plot it
for i, col in enumerate(selected_df.columns):
    axes[i].plot(selected_df.index, selected_df[col], label=col)
    axes[i].set_xlabel('time (ms)')
    axes[i].set_ylabel('Volts')
    axes[i].set_title(col)
    axes[i].legend()

# Adjust subplot layout
plt.tight_layout()

# Show the plot
plt.show()

