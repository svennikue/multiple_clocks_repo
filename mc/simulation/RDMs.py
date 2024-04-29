#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:04:41 2023

@author: Svenja Küchenhoff

This script defines functions for creating RDMs and plotting them.
"""

import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
import numpy as np
import mc
import scipy
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm
import colormaps as cmaps

def plot_RDMs(RDM_dict, no_tasks, save_dir = None, string_for_ticks = None):
    for RDM in RDM_dict:
        # import pdb; pdb.set_trace()
        indexline_after = int(len(RDM_dict[RDM])/no_tasks)
        fig, ax = plt.subplots(figsize=(5,4))
        cmaps.BlueYellowRed
        cmap = plt.get_cmap('BlueYellowRed')
        # Set the upper triangle to be empty
        corr_mat = RDM_dict[RDM]
        
        corr_mat[np.triu_indices(int(len(RDM_dict[RDM])), k=1)] = np.nan
        im = ax.imshow(corr_mat, cmap=cmap, interpolation = 'none', aspect = 'equal', vmin=-1, vmax=1); 
        for i in range(indexline_after-1,int(len(RDM_dict[RDM])),indexline_after):
            ax.axhline(i+0.5, color='white', linewidth=1)
            ax.axvline(i+0.5, color='white', linewidth=1)
            
        # #Add a colorbar to the right of the plot with a colormap toolbox
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # cbar = fig.colorbar(im, cax=cax)
        # cbar.set_label('Correlation', rotation=270, labelpad=15)
        
        # Set x-axis and y-axis ticks and labels
        if indexline_after == 1:
            ticks = np.arange(indexline_after-1, int(len(RDM_dict[RDM])), indexline_after)
        else:   
            ticks = np.arange(indexline_after/2, int(len(RDM_dict[RDM])), indexline_after)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if string_for_ticks == None:
            ax.set_xticklabels(['Task {}'.format(int(i // indexline_after + 1)) for i in ticks], rotation=45, ha = 'right', fontsize=16)
            ax.set_yticklabels(['Task {}'.format(int(i // indexline_after + 1)) for i in ticks], fontsize=16)
        else:
            ax.set_xticklabels(string_for_ticks, rotation=45, ha = 'right', fontsize=16)
            ax.set_yticklabels(string_for_ticks, fontsize=16)


                
        # Set axis labels and title
        ax.set_title(f"Model RDM for {RDM} model", fontsize=18)
        
        # Adjust the appearance of ticks and grid lines
        ax.grid(False)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Pearson's r", rotation=-90, va="bottom")
        
        
        # Adjust the layout to prevent cutoff of labels and colorbar
        plt.tight_layout()
        if save_dir:
            fig.savefig(f"{save_dir}{RDM}.png", dpi=300, bbox_inches='tight')
            fig.savefig(f"{save_dir}{RDM}.tiff", dpi=300, bbox_inches='tight')
        
    
    
    
    
    
def within_task_RDM(activation_matrix, ax=None, plotting = False, titlestring = None, intervalline= None):
    # import pdb; pdb.set_trace()
    activation_matrix = np.nan_to_num(activation_matrix)
    RSM = np.corrcoef(activation_matrix.T) # pairwise pearson corr of columns, excluding NA/nulls
    # replace nans with 0s afterwards
    # there will be nans in the RSM if the variance in one row is the same > division by 0
    RSM = np.nan_to_num(RSM)
    # from Nili et al., 2014: 
        # "Popular distance measures are the correlation distance (1 minus the Pearson correlation, 
        # "computed across voxels or sites of the two activity patterns), the Euclidean distance 
        # "(the square root of the sum of squared differences between the two patterns), and the Mahalanobis 
        # "distance (which is the Euclidean distance measured after linearly recoding the space so as to whiten the noise)."
    # RDM = np.ones((len(RSM), len(RSM)))
    # RDM = RDM - RSM
    if plotting == True: 
    
        
        if ax is None:
            plt.figure()
            ax = plt.axes()   
        plt.imshow(RSM, interpolation = 'none', cmap='ocean')
        plt.title(f'{titlestring}')
        if intervalline:
            intervalline = int(intervalline)
            for interval in range(0, len(activation_matrix[0]), intervalline):
                plt.axvline(interval, color='white', ls='dashed')
        # sn.heatmap(corr_matrix, annot = False)
    return RSM




# create RDM based on dataframes, doesn't matter between what
def df_based_RDM(dataframe, ax=None, plotting = False, titlestring = None):
    # import pdb; pdb.set_trace()
    dataframe = dataframe.fillna(0)
    #corr_matrix = dataframe.corr(method = 'pearson') # pairwise pearson corr of columns, excluding NA/nulls
    corr_matrix = np.corrcoef(dataframe.to_numpy().transpose())
    # from Nili et al., 2014: 
        # "Popular distance measures are the correlation distance (1 minus the Pearson correlation, 
        # "computed across voxels or sites of the two activity patterns), the Euclidean distance 
        # "(the square root of the sum of squared differences between the two patterns), and the Mahalanobis 
        # "distance (which is the Euclidean distance measured after linearly recoding the space so as to whiten the noise)."
    if plotting == True:       
        if ax is None:
            plt.figure()
            ax = plt.axes()   
        #print(corr_matrix)
        plt.imshow(corr_matrix)
        plt.title(f'{titlestring}') 
        #sn.heatmap(corr_matrix, annot = False)
    RSM = corr_matrix
    return RSM


def between_task_RDM(no_tasks, column_names, ax=None, plotting = False):
    #import pdb; pdb.set_trace()
    for i in range(0, no_tasks):
        ## Create the task and paths
        reward_coords = mc.simulation.grid.create_grid()
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords) 
        ## Setting the Clocks and Location Matrix. 
        clocks_matrix, total_steps  = mc.simulation.predictions.set_clocks(reshaped_visited_fields, all_stepnums, 3)           
        loc_matrix, total_steps = mc.simulation.predictions.set_location_matrix(reshaped_visited_fields, all_stepnums, 3, 0)
        df_rewards = pd.DataFrame(reward_coords)
        if i == 0:
            df_clocks = pd.DataFrame(clocks_matrix)
            df_locs = pd.DataFrame(loc_matrix)
            df_clocks.columns = column_names
            df_locs.columns = column_names
            df_task_configs = pd.DataFrame(reshaped_visited_fields)
            df_task_configs = pd.concat([df_rewards, df_task_configs], axis = 1)
        else:
            temp_clocks = pd.DataFrame(clocks_matrix)
            temp_locs = pd.DataFrame(loc_matrix)
            temp_clocks.columns = column_names
            temp_locs.columns = column_names
            temp_path = pd.DataFrame(reshaped_visited_fields)
            df_clocks = pd.concat([df_clocks, temp_clocks], axis = 1)               
            df_locs = pd.concat([df_locs, temp_locs], axis = 1)
            df_task_configs = pd.concat([df_task_configs, df_rewards, temp_path], axis = 1)
    df_clocks.fillna(0, inplace = True)  
    df_locs.fillna(0, inplace = True)
    corr_clocks = df_clocks.corr()
    corr_locs = df_locs.corr()
    clocks_RSM = corr_clocks.to_numpy()
    locs_RSM = corr_locs.to_numpy()
    if plotting == True:       
        if ax is None:
            plt.figure()
            fig, ax =plt.subplots(1,2)  
        sn.heatmap(corr_clocks, annot = False, ax=ax[0])
        ax[0].set_title('Clocks')
        print(corr_clocks)
        sn.heatmap(corr_locs, annot = False, ax=ax[1])
        ax[1].set_title('Location')
        print(corr_locs)  
    return clocks_RSM, locs_RSM, df_clocks, df_locs, df_task_configs
            

def find_best_tasks(loop_no, no_columns, column_names): 
#    import pdb; pdb.set_trace()
#     # this needs to be something like:
#         # 1. create 10 random tasks and the between-task corr maps.
#         # 2. compute similarity between those 2 big matrices (this needs to be exclude_diag = False!! bc thats the within task one)
#         # 3. stepwise go through each task configuration and check if replacing it with 
#         #      a new one reduces the similarity value
#         # do this a number of loops
#         # always store the current configurations/ toss the one I am replacing
    # first, create one 10 tasks x 10 tasks matrix for clocks and locations
    task_config_no = 10
    clock_RSM_matrix, loc_RSM_matrix, df_clock, df_loc, task_configs = mc.simulation.RDMs.between_task_RDM(task_config_no, column_names, plotting = False)
    # and get the similarity between those. 
    similarity_between = mc.simulation.RDMs.corr_matrices(loc_RSM_matrix, clock_RSM_matrix)
    # based on this, try to optimize the correlation coefficient (similarity_between)
    for i in range(0, loop_no):        
        # then first take the first 10 columns of df_clock and df_loc and replace it with a new config
        # create new configuration
        reward_coords = mc.simulation.grid.create_grid()
        reshaped_visited_fields, all_stepnums = mc.simulation.grid.walk_paths(reward_coords) 
        df_rewards = pd.DataFrame(reward_coords)
        df_task_configs = pd.DataFrame(reshaped_visited_fields)
        df_temp_task_configs = pd.concat([df_rewards, df_task_configs], axis = 1)
        # create new neural predictions for this task config
        clocks_matrix, total_steps  = mc.simulation.predictions.set_clocks(reshaped_visited_fields, all_stepnums, 3)           
        loc_matrix, total_steps = mc.simulation.predictions.set_location_matrix(reshaped_visited_fields, all_stepnums, 3, 0)
        # turn those into dataframe
        temp_clocks = pd.DataFrame(clocks_matrix)
        temp_locs = pd.DataFrame(loc_matrix)
        temp_clocks.columns = column_names
        temp_locs.columns = column_names
        # prepare loop here       
        temp_similarity = np.ones((2,2))
        
        count = 1 # change back to 1 once debugging is done
        # then, replace each of the 10 tasks with the new config and test if similarity is now less (= better)
        # step out of the loop either way once looped through all columns, or when temp_similarity is lower
        while (temp_similarity[0,1] > similarity_between[0,1]) and (count < task_config_no):
            # have a counter for all columns   
            temp_df_loc = df_loc.copy()
            # replace the first (count) 12 columns with the new configuration
            temp_df_loc.iloc[:, ((count-1)*no_columns):(count*no_columns)] = temp_locs
            # temp_df_loc.iloc[:, (count*no_columns):((count*no_columns)+no_columns)] = temp_locs
            temp_df_loc.fillna(0, inplace = True)
            temp_df_clock = df_clock.copy()
            temp_df_clock.iloc[:, ((count-1)*no_columns):(count*no_columns)] = temp_clocks
            temp_df_clock.fillna(0, inplace = True)
            # create new correlation matrices for the new clocks and location matrix
            temp_corr_clocks = temp_df_clock.corr()
            temp_corr_locs = temp_df_loc.corr()
            temp_clocks_RSM = temp_corr_clocks.to_numpy()
            temp_locs_RSM = temp_corr_locs.to_numpy() 
            # test the new similarity between the new RSMs
            temp_similarity = mc.simulation.RDMs.corr_matrices(temp_locs_RSM, temp_clocks_RSM)
            if temp_similarity[0,1] < similarity_between[0,1]:
                # task_configs is structured a little different, its always x,y of paths and then x,y of rewards.
                # -> 4 columns per task config              
                task_configs.iloc[:,((count-1)*4):(count*4)] = df_temp_task_configs
                # if the new RSMs correlate less, replace the current configuration and RSM with the new
                # and continue to optimize further.
                df_clock = temp_df_clock.copy()
                df_loc = temp_df_loc.copy()
                similarity_between = temp_similarity.copy()
            count += 1 
            del temp_df_loc
            del temp_df_clock
            del temp_corr_clocks
            del temp_corr_locs
            del temp_similarity
            temp_similarity = np.ones((2,2))
        
    return df_clock, df_loc, task_configs, similarity_between



# delete this one when I have all scripts adjusted!!!
def corr_matrices(matrix_one, matrix_two, exclude_diag = True):
    # import pdb; pdb.set_trace()
    dimension = len(matrix_one) 
    if exclude_diag == True:
        diag_array_one = list(matrix_one[np.tril_indices(dimension, -1)])
        diag_array_two = list(matrix_two[np.tril_indices(dimension, -1)])
    else: # this is diagonal plus upper triangle
        diag_array_one = list(matrix_one[np.triu_indices(dimension)])
        diag_array_two = list(matrix_two[np.triu_indices(dimension)])
    coef = scipy.stats.kendalltau(diag_array_one, diag_array_two) # kandalls tau, because:
    # from Nili et al., 2014:
        # "We do not in general want to assume a linear relationship between the dissimilarities.
        # "Unless we are confident that our model captures not only the neuronal representational 
        # "geometry but also its possibly nonlinear reflection in our response channels (e.g. fMRI patterns), 
        # "assuming a linear relationship between model and brain RDMs appears questionable. We therefore 
        # "prefer to assume that a model RDM predicts merely the rank order of the dissimilarities. 
        # "For this reason we recommend the use of rank-correlations for comparing RDMs.
        # " We recommend Kendall's τA
    return coef



def corr_matrices_kendall(matrix_one, matrix_two, exclude_diag = True):
    # import pdb; pdb.set_trace()
    dimension = len(matrix_one) 
    if exclude_diag == True:
        diag_array_one = list(matrix_one[np.tril_indices(dimension, -1)])
        diag_array_two = list(matrix_two[np.tril_indices(dimension, -1)])
    else: # this is diagonal plus upper triangle
        diag_array_one = list(matrix_one[np.triu_indices(dimension)])
        diag_array_two = list(matrix_two[np.triu_indices(dimension)])
    coef = scipy.stats.kendalltau(diag_array_one, diag_array_two) # kandalls tau, because:
    # from Nili et al., 2014:
        # "We do not in general want to assume a linear relationship between the dissimilarities.
        # "Unless we are confident that our model captures not only the neuronal representational 
        # "geometry but also its possibly nonlinear reflection in our response channels (e.g. fMRI patterns), 
        # "assuming a linear relationship between model and brain RDMs appears questionable. We therefore 
        # "prefer to assume that a model RDM predicts merely the rank order of the dissimilarities. 
        # "For this reason we recommend the use of rank-correlations for comparing RDMs.
        # " We recommend Kendall's τA
    return coef

def corr_matrices_pearson(matrix_one, matrix_two, no_tasks = None, mask_within = False, exclude_diag = True):
    # import pdb; pdb.set_trace()
    import numpy.ma as ma
    if mask_within == True:
        within_task_mask = np.kron(np.eye(no_tasks), np.ones((int(np.round(len(matrix_one)/no_tasks)), int(np.round(len(matrix_one)/no_tasks)))))
        
        # in case the mask doesn't match the data matrix perfectly (that's due to my estimations of 'early' 'mid' 'late')
        while len(within_task_mask) < len(matrix_one):
            within_task_mask = np.concatenate((within_task_mask, within_task_mask[:,-2:-1]), axis =1 )
            within_task_mask = np.concatenate((within_task_mask, within_task_mask[-2:-1,:]), axis =0 )
        while len(within_task_mask) > len(matrix_one):
            within_task_mask = np.delete(within_task_mask, -1, 0)
            within_task_mask = np.delete(within_task_mask, -1, 1)
            
        matrix_one[within_task_mask == 1] = np.nan
        matrix_two[within_task_mask == 1] = np.nan
        
        
    dimension = len(matrix_one) 
    if exclude_diag == True:
        diag_array_one = list(matrix_one[np.tril_indices(dimension, -1)])
        diag_array_two = list(matrix_two[np.tril_indices(dimension, -1)])
    else: # this is diagonal plus upper triangle
        diag_array_one = list(matrix_one[np.triu_indices(dimension)])
        diag_array_two = list(matrix_two[np.triu_indices(dimension)])
    

    # # CLEAN FROM nans
    # diag_array_one_clean = np.array(diag_array_one)[~np.isnan(diag_array_one)]
    # diag_array_two_clean = np.array(diag_array_two)[~np.isnan(diag_array_two)]
    # coef = np.corrcoef(diag_array_one_clean, diag_array_two_clean) 
    
    # make sure to exclude nans!
    coef = ma.corrcoef(ma.masked_invalid(diag_array_one), ma.masked_invalid(diag_array_two))
    # pearson's, because:
        # I will use a linear regression in the end, so there will be a linear
        # relationship assumed between the two model matrices. Since pearson's r
        # seems to be much higher, I should look at that one rather than the rank corr.
    return coef


def corr_matrices_no_autocorr(matrix_one, matrix_two, timepoints_to_exclude, plotting = False):
    # import pdb; pdb.set_trace()
    
    dim_mask = len(matrix_one)  
    mask= np.tril(np.full((180,180),True),-30) * ~np.tril(np.full((180,180),True),-150)
    mask = np.tril(np.full((dim_mask,dim_mask),True),-timepoints_to_exclude) * ~np.tril(np.full((dim_mask,dim_mask),True),-(dim_mask - timepoints_to_exclude))
    diag_array_one = matrix_one[mask]
    diag_array_two = matrix_two[mask]
    coef_pearson = np.corrcoef(diag_array_one, diag_array_two)
    coef_kendall = scipy.stats.kendalltau(diag_array_one, diag_array_two) # kandalls tau, because:
    # from Nili et al., 2014:
        # "We do not in general want to assume a linear relationship between the dissimilarities.
        # "Unless we are confident that our model captures not only the neuronal representational 
        # "geometry but also its possibly nonlinear reflection in our response channels (e.g. fMRI patterns), 
        # "assuming a linear relationship between model and brain RDMs appears questionable. We therefore 
        # "prefer to assume that a model RDM predicts merely the rank order of the dissimilarities. 
        # "For this reason we recommend the use of rank-correlations for comparing RDMs.
        # " We recommend Kendall's τA
    if plotting == True:
        plt.figure(); 
        plt.subplot(1,2,1); 
        plt.imshow(matrix_one*mask); 
        plt.subplot(1,2,2); 
        plt.imshow(matrix_two*mask)
    return coef_kendall, coef_pearson

def lin_reg_RDMs(data_matrix, regressor_one_matrix, regressor_two_matrix = None, regressor_three_matrix = None, regressor_four_matrix = None, t_val = None):
    # import pdb; pdb.set_trace()
    dimension = len(data_matrix) 
    diag_array_data = list(data_matrix[np.tril_indices(dimension , -1)])
    X = list(regressor_one_matrix[np.tril_indices(dimension, -1)])
    if regressor_two_matrix is not None:
        diag_array_reg_two = list(regressor_two_matrix[np.tril_indices(dimension, -1)])
        X = np.vstack((X, diag_array_reg_two))
    if regressor_three_matrix is not None:
        diag_array_reg_three = list(regressor_three_matrix[np.tril_indices(dimension, -1)])
        X = np.vstack((X, diag_array_reg_three))
    if regressor_four_matrix is not None:
        diag_array_reg_four = list(regressor_four_matrix[np.tril_indices(dimension, -1)])
        X = np.vstack((X, diag_array_reg_four))
    # now do the linear regression
    # check if there are nans
    # np.count_nonzero(np.isnan(data))
    # check correlation within the desing matrix
    design_corr = np.corrcoef(X)
    x_reshaped = np.transpose(X)
    regression_results = LinearRegression().fit(x_reshaped, diag_array_data)
    
    scipy_reg_est = None
    if t_val is not None:
    # second try
        X_3 = sm.add_constant(x_reshaped)
        est = sm.OLS(diag_array_data, X_3)
        scipy_reg_est = est.fit().tvalues
    
    return regression_results, scipy_reg_est
    
        
    
    

def GLM_RDMs(data_matrix, regressor_dict, mask_within = True, no_tasks = None, t_val = True, plotting = False):
    
    if mask_within == True:
        within_task_mask = np.kron(np.eye(no_tasks), np.ones((int(np.round(len(data_matrix)/no_tasks)), int(np.round(len(data_matrix)/no_tasks)))))
        
        # in case the mask doesn't match the data matrix perfectly (that's due to my estimations of 'early' 'mid' 'late')
        while len(within_task_mask) < len(data_matrix):
            within_task_mask = np.concatenate((within_task_mask, within_task_mask[:,-2:-1]), axis =1 )
            within_task_mask = np.concatenate((within_task_mask, within_task_mask[-2:-1,:]), axis =0 )
        while len(within_task_mask) > len(data_matrix):
            within_task_mask = np.delete(within_task_mask, -1, 0)
            within_task_mask = np.delete(within_task_mask, -1, 1)
            
        data_matrix[within_task_mask == 1] = np.nan
        for regressor_matrix in regressor_dict:
            regressor_dict[regressor_matrix][within_task_mask == 1] = np.nan
    
    if plotting == True:
        plot_data = np.tril(data_matrix, -1)
        plt.figure()
        plt.imshow(plot_data, aspect = 'auto')
        for interval in range(0, len(data_matrix), int(len(data_matrix)/(no_tasks))):
            plt.axvline(interval, color='white', ls='dashed')
    
    dimension = len(data_matrix) 
    reg_labels = []
    # import pdb; pdb.set_trace()
    for i, regressor_matrix in enumerate(regressor_dict):
        reg_labels.append(regressor_matrix)
        if i == 0:
            X = list(regressor_dict[regressor_matrix][np.tril_indices(dimension, -1)])
        else:
            diag_reg_array = list(regressor_dict[regressor_matrix][np.tril_indices(dimension, -1)])
            X = np.vstack((X, diag_reg_array))
     
    # get rid of all nans, equally for data and regressors.
    
    diag_array_data = data_matrix[np.tril_indices(dimension , -1)]
    
    
    if len(X) > 10:
        # this means that there is only one regressor
        X = np.array(X)
        X_cleaned = np.empty(len(diag_array_data[~np.isnan(diag_array_data)]))
        X_cleaned = X[~np.isnan(diag_array_data)]
    else:      
        X_cleaned = np.empty((len(X),len(diag_array_data[~np.isnan(diag_array_data)])))
        for i, row in enumerate(X):
            X_cleaned[i] = row[~np.isnan(diag_array_data)]
            
            
    diag_array_data = diag_array_data[~np.isnan(diag_array_data)]
    
    # import pdb; pdb.set_trace()
    x_cl_reshaped = np.transpose(X_cleaned)
    if len(X) > 10: 
        regression_results = LinearRegression().fit(x_cl_reshaped.reshape(-1, 1), diag_array_data)
    else:
        regression_results = LinearRegression().fit(x_cl_reshaped, diag_array_data)
    results = {}
    results['coefs'] = regression_results.coef_
    results['label_regs'] = reg_labels
    
    if t_val is not None:
    # second try
        X_3 = sm.add_constant(x_cl_reshaped)
        est = sm.OLS(diag_array_data, X_3)
        scipy_reg_est = est.fit().tvalues
        results['t_vals'] = scipy_reg_est
    
    return results
    
    
    
    
    
    
    
    
    
    
    