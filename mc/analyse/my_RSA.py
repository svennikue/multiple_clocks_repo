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
import statsmodels.api as sm
import os
import nibabel as nib


def load_data_EVs(data_dir, regression_version):
    EV_dict = {}
    # names need to be 'A1_backw_A_path' etc.
    list_loaded = []
    for th in [1,2]:
        pe_path = f"{data_dir}/func/glm_{regression_version[0:2]}_pt0{th}.feat/stats"
        # order from FSL processed EVs to names is stored here:
        with open(f"{data_dir}/func/EVs_{regression_version[0:2]}_pt0{th}/task-to-EV.txt", 'r') as file:
            for line in file:
                index, name_ev = line.strip().split(' ', 1)
                name = name_ev.replace('ev_', '')
                EV_path = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")
                EV_dict[name] = np.array(nib.load(EV_path).get_fdata()).flatten()
                # reshape data so we have 1 x n_voxels
                # import pdb; pdb.set_trace()
                if name not in ['press_EV']:
                    list_loaded.append(name)
    print(f"loaded the following data EVs in dict: {list_loaded}")
    return EV_dict, list_loaded
    


def get_RDM_per_searchlight(fmri_data, centers, neighbors, method = 'crosscorr'):
    # import pdb; pdb.set_trace()
    data_2d = fmri_data
    if method == 'crosscorr':
        #data_2d = np.concatenate((fmri_data['1'], fmri_data['2']),0)
        centers = np.array(centers)
        #n_conds = fmri_data['1'].shape[0]
        n_conds = int(data_2d.shape[0]/2)
    
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
        
        
def compute_crosscorr(data_chunk, plotting = False):  
    RDM = []
    #import pdb; pdb.set_trace()
    if not isinstance(data_chunk, (list, tuple)):
        data_chunk = [data_chunk]
    
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
        # import pdb; pdb.set_trace()
        n = rdm.shape[1]
        RDM.append(rdm[np.triu_indices(n, k=0)]) 
        if plotting == True:
            plt.figure()
            plt.imshow(rdm, aspect = 'auto', cmap = 'coolwarm')
            plt.figure()
            plt.imshow(rdm_both_halves, aspect = 'auto', cmap = 'coolwarm')
            
    return RDM


def evaluate_model(model_rdm, data_rdm):
    # import pdb; pdb.set_trace()

    #X = sm.add_constant(model_rdm.transpose());
    X = sm.add_constant(model_rdm);
    # first, normalize the regressors (but not the intercept, bc std = 0 -> division by 0!)
    for i in range(1, X.shape[1]):
        X[:,i] = (X[:,i] - np.nanmean(X[:,i]))/ np.nanstd(X[:,i])
    
    # to check if a GLM is ill-conditioned
    # To check that you can check the “condition number” of the design matrix - 
    # the ration between the maximum singular value (similar to eigenvalue) and the minimum singular value.. 
    # If that ratio is close to 1, you’re good. If it’s very large (e.g. >1000), it means the matrix is ill-conditioned - 
    # one of your regressors is close to being a linear combination of the other two.
    # mc.analyse.analyse_MRI_behav.check_GLM_regressors(X)
    # import pdb; pdb.set_trace()
    
    Y = data_rdm;
    
    # to filter out potential nans in the model part
    nan_filter = np.isnan(X).any(axis=1)
    filtered_X = X[~nan_filter]
    filtered_Y = Y[~nan_filter]
    
    est = sm.OLS(filtered_Y, filtered_X).fit()
    # import pdb; pdb.set_trace()
    return est.tvalues[1:], est.params[1:], est.pvalues[1:]
