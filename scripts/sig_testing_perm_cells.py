#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:37:02 2025

@author: Svenja Küchenhoff


Run statistical tests on cells that are significant after permutations


"""


import pandas as pd
import mc
import numpy as np
from matplotlib import pyplot as plt

def pref_counts_per_roi(df_sig, df_all, roi_col='roi', state_col='pref_state', states=('A','B','C','D')):
     d_s = df_sig[[roi_col, state_col]].dropna().copy()
     d_a = df_all[[roi_col, state_col]].dropna().copy()
     # keep ROI display order as they first appear
     roi_order = d_s[roi_col].dropna().unique().tolist()
     d_s[state_col] = pd.Categorical(d_s[state_col], categories=list(states), ordered=True)
     counts_sig = (d_s.groupby([roi_col, state_col]).size()
                 .unstack(fill_value=0)
                 .reindex(index=roi_order, columns=states, fill_value=0))
     counts_all = (d_a.groupby([roi_col, state_col]).size()
                 .unstack(fill_value=0)
                 .reindex(index=roi_order, columns=states, fill_value=0))
     return counts_sig, counts_all
 
    
    
path = '/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group/state_tuning'
#file = 'pval_for_perms200_state_consistency_residualised_repeats_excl_gridwise_qc_pct_neurons.csv'
file = 'pval_for_perms200_state_consistency_late_repeats_excl_gridwise_qc_pct_neurons.csv'
pval_df = pd.read_csv(f"{path}/{file}")

sig_state_cells = pval_df[pval_df['p_perm']<0.05]
top_sig_state_cells = sig_state_cells.sort_values(by="state_cv_consistency", ascending=False).head(20)
sessions_to_load = top_sig_state_cells['session_id'].unique()


# 1) Build counts table for significant cells and all cells for proportions
counts_sig_cells, counts_all_cells = pref_counts_per_roi(sig_state_cells, df_all = pval_df, roi_col='roi', state_col='pref_state')
# 2 plot cell count significant per roi and state 
plot_pref_counts_per_roi(counts_sig_cells, stacked=False)   # set stacked=True for stacked bars
# 2a) See composition per ROI (each ROI sums to 100%) — best for comparing ROIs
plot_pref_props_per_roi(counts_sig_cells, normalize='within_roi', grouped=False)   # 100% stacked

# denominator = total recorded per ROI (sum over A–D)
denom = counts_all_cells.sum(axis=1).replace(0, np.nan)
# proportions of recorded cells that are significant (per ROI × state)
props = counts_sig_cells.div(denom, axis=0).fillna(0.0)

plot_sig_props(props, grouped=False)   # stacked
# or:
plot_sig_props(props, grouped=True)    # side-by-side
  