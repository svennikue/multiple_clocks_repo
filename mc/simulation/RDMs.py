#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:04:41 2023

@author: Svenja Küchenhoff

This script defines functions for creating RDMs and plotting them.
"""

import pandas as pd
import seaborn as sn


def within_task_RDM(activation_matrix, column_names): 
    # import pdb; pdb.set_trace()
    dataframe = pd.DataFrame(activation_matrix)
    dataframe.fillna(0)
    dataframe.columns = column_names
    corr_matrix = dataframe.corr()
    print(corr_matrix)
    sn.heatmap(corr_matrix, annot = False)
    RSM = corr_matrix.to_numpy()
    return RSM


    
