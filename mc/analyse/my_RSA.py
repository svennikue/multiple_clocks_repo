#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:38:27 2025

@author: Svenja Küchenhoff


All things RSA


"""
import mc
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def get_RDM_per_searchlight(fmri_data, centers, neighbors, method = 'crosscorr'):
    # import pdb; pdb.set_trace()
    if method == 'crosscorr':
        data_2d = np.concatenate((fmri_data['1'], fmri_data['2']),0)
        centers = np.array(centers)
        n_conds = fmri_data['1'].shape[0]
    
    # first step: parallelise centers/neighbors.
    n_centers = centers.shape[0]
    # For memory reasons, we chunk the data if we have more than 1000 RDMs
    # loop over chunks
    if n_centers > 1000:
        # we can't run all centers at once, that will take too much memory
        # so lets to some chunking
        chunked_center = np.split(np.arange(n_centers),
                                  np.linspace(0, n_centers,
                                              101, dtype=int)[1:-1])
        # output will INCLUDE the diagonal. so triangle number is:
        sl_rdms = np.zeros((n_centers, n_conds * (n_conds + 1) // 2))
        # if excluding the diagonal
        # sl_rdms = np.zeros((n_centers, n_conds * (n_conds - 1) // 2))
        all_centers = np.zeros((n_centers))
        #for chunks in chunked_center:
        for chunks in tqdm(chunked_center, desc='Calculating RDMs...'):            
            center_data= []
            for c in chunks:
                # grab this centers of this chunk and its and neighbors
                center_neighbors = neighbors[c]
                center_data.append(data_2d[:, center_neighbors])
            # then compute the RDM per searchlight
            if method == 'crosscorr':
                RDM_corr = mc.analyse.my_RSA.compute_crosscorr(center_data)
            sl_rdms[chunks, :] = RDM_corr
            # then store per voxel and return.
            all_centers[chunks] = centers[chunks]
            # this is the same as centers. thus you can use centres as voxel indices          
    return sl_rdms
        
        
def compute_crosscorr(data_chunk):  
    RDM = []
    for data in data_chunk:
        # centers the data around zero by subtracting the mean of each row
        data_demeaned = data - data.mean(axis=1, keepdims=True)
        # normalising data
        data_demeaned /= np.sqrt(np.einsum('ij,ij->i', data_demeaned, data_demeaned))[:, None]    
        # cosine dissimilarity
        rdm_both_halves = 1 - np.einsum('ik,jk', data_demeaned, data_demeaned)  
        
        # cutting the lower left square of the matrix
        rdm_small = rdm_both_halves[int(len(rdm_both_halves)/2):,0:int(len(rdm_both_halves)/2)]
        
        # making the matrix symmetric
        rdm = (rdm_small + np.transpose(rdm_small))/2
        
        # lastly, only store the part of the RDM I am actually interested in 
        # i.e. the lower triangle, including the diagonal.
        n = rdm.shape[1]
        RDM.append(rdm[np.triu_indices(n, k=0)]) 
    return RDM