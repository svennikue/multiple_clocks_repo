#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn import datasets, plotting
from nilearn.image import new_img_like

import matplotlib.cm as cm
import matplotlib.colors as mcolors

import plotly.graph_objects as go
from skimage.measure import marching_cubes


# # =============================================================================
# # USER SETTINGS
# # =============================================================================

# path_to_cell_table = ("/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/neurons_with_final_roi_labels.csv")

# path_to_S_table = ("/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/subj_list-MNI-coords-Sangkyu.csv")

# my_cells = pd.read_csv(path_to_cell_table)
# S_cells = pd.read_csv(path_to_S_table)




import pandas as pd
import numpy as np

path_to_cell_table = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/neurons_with_ROI_labels.csv"
path_to_S_table = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/subj_list-MNI-coords-Sangkyu.csv"

my_cells = pd.read_csv(path_to_cell_table)
S_cells = pd.read_csv(path_to_S_table)

def clean_key(s):
    return (
        s.astype(str)
         .str.strip()
         .str.strip("'")   # remove literal single quotes around the text
         .str.strip('"')   # just in case there are double quotes too
         .str.lower()
    )

my_cells["Subject Label_merge"] = clean_key(my_cells["Subject Label"])
my_cells["electrode_label_merge"] = clean_key(my_cells["electrode label"])

S_cells["subject_ID_merge"] = clean_key(S_cells["subject_ID"])
S_cells["electrode_label_merge"] = clean_key(S_cells["electrode_label"])

merged = my_cells.merge(
    S_cells,
    left_on=["Subject Label_merge", "electrode_label_merge", "subject"],
    right_on=["subject_ID_merge", "electrode_label_merge", "subject_NO"],
    how="left",
    indicator=True,
    suffixes=("", "_S"),
)

cols = [
    "MNI_x_original","X", "MNI_y_original","Y", "MNI_z_original",
    "Z",
    "subject_ID_merge", "electrode_label_merge", 'Recording Site', 'subject'
]

subset = merged[cols]


# Check whether coordinates match
# Use np.isclose rather than == in case coordinates differ by tiny floating-point rounding
merged["MNI_x_matches_X"] = np.isclose(merged["MNI_x"], merged["X"], equal_nan=True)
merged["MNI_y_matches_Y"] = np.isclose(merged["MNI_y"], merged["Y"], equal_nan=True)
merged["MNI_z_matches_Z"] = np.isclose(merged["MNI_z"], merged["Z"], equal_nan=True)

# Mark rows where any coordinate does NOT match
merged["coordinate_mismatch"] = ~(
    merged["MNI_x_matches_X"] &
    merged["MNI_y_matches_Y"] &
    merged["MNI_z_matches_Z"]
)

# Also mark rows that had no matching row in S_cells
merged["no_match_in_S_table"] = merged["_merge"].eq("left_only")

# If you want one combined issue flag:
merged["coordinate_issue"] = merged["coordinate_mismatch"] | merged["no_match_in_S_table"]

# View problematic rows
problem_rows = merged.loc[
    merged["coordinate_issue"],
    [
        "Subject Label", "subject_ID",
        "electrode label", "electrode_label",
        "MNI_x", "X",
        "MNI_y", "Y",
        "MNI_z", "Z",
        "coordinate_mismatch",
        "no_match_in_S_table"
    ]
]

problem_rows