#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:30:37 2024

DEBUG AND PLOT

collection of functions that help to have a closer look at the data to eventually debug!


@author: xpsy1114
"""

from nilearn.image import load_img
import os
import matplotlib.pyplot as plt
import numpy as np
import mc
import nibabel as nib
import rsatoolbox
import math
import colormaps as cmaps

# Plot the input-vectors [conditions] that will eventually lead to an RDM for a certain voxel location
def plot_data_RDMconds_per_searchlight(data_RDM_2d, centers, neighbors, voxel_coord, ref_image, cond_names_dict):
    # import pdb; pdb.set_trace()
    x, y, z = ref_image.shape
    # Calculate the linear index in the flattened array
    linear_index = voxel_coord[0] * (y * z) + voxel_coord[1] * z + voxel_coord[2]
    neighbors_index = np.where(centers == linear_index)[0][0]
    
    
    voxel_conds = {}
    for half in data_RDM_2d:
        voxel_conds[half] = data_RDM_2d[half][:, neighbors[neighbors_index]]
    
    
    # Combine the data and condition names
    
    combined_data = np.vstack((voxel_conds['1'], voxel_conds['2']))
    combined_condition_names = cond_names_dict['1'] + cond_names_dict['2']
    

    # Plot the combined data matrix
    plt.figure(figsize=(12, 8))
    plt.imshow(combined_data, aspect='auto', cmap='viridis')
    
    # Add white lines to separate the conditions
    num_conditions_th1 = voxel_conds['1'].shape[0]
    num_conditions_th2 = voxel_conds['2'].shape[0]
    
    for i in range(1, num_conditions_th1 + num_conditions_th2):
        plt.axhline(i - 0.5, color='white', linewidth=1)
    
    # Set condition names as y-ticks
    plt.yticks(ticks=np.arange(len(combined_condition_names)), labels=combined_condition_names)
    
    # Add labels and title
    plt.xlabel('Voxels')
    plt.ylabel('Conditions')
    plt.title('Condition x Voxel Matrix')
    
    plt.colorbar(label='Activation')
    plt.show()
    
    # also compute RDM just doing a simple Pearson correlation
    activation_matrix = np.nan_to_num(combined_data)
    RSM = np.corrcoef(activation_matrix) # pairwise pearson corr of columns, excluding NA/nulls
    # replace nans with 0s afterwards
    # there will be nans in the RSM if the variance in one row is the same > division by 0
    RDM = np.nan_to_num(1-RSM)
    mc.plotting.deep_data_plt.RDM_plotting(RDM, titelstring = "1 - Pearson's r, complete matrix", condition_name_string = combined_condition_names)

    # also plot the averaged one, the one I am actually considering.
    RDM_avg = RDM[int(len(RDM)/2):,0:int(len(RDM)/2)]
    RDM_avg = (RDM_avg + np.transpose(RDM_avg))/2
    mc.plotting.deep_data_plt.RDM_plotting(RDM_avg, titelstring = "lower square, avg, of 1- pearson's r", condition_name_string = cond_names_dict['1'])

    



# plot the data RDM of a certain voxel location
def plot_dataRDM_by_voxel_coords(data_RDM_object, voxel_coord, ref_image, cond_names_dict):
    # import pdb; pdb.set_trace()
    x, y, z = ref_image.shape
    # Calculate the linear index in the flattened array
    linear_index = voxel_coord[0] * (y * z) + voxel_coord[1] * z + voxel_coord[2]
    RDM_index = np.where(data_RDM_object.rdm_descriptors['voxel_index']==linear_index)[0][0]
    # Extract the data point from the 1-dimensional array using the calculated index
    fig, ax, ret_vla = rsatoolbox.vis.show_rdm(data_RDM_object[RDM_index])
    print(f"The RDM index for voxel coordinates ({voxel_coord[0]}, {voxel_coord[1]}, {voxel_coord[2]}) is: {RDM_index}")
    
    # or, using matplotlib:
    dims = int(( 1 + math.sqrt(1 + 8* data_RDM_object[RDM_index].dissimilarities.shape[1])) /2)
    RDM = np.zeros((dims, dims))
    triu_indices = np.triu_indices(dims, 1)
    RDM[triu_indices] = data_RDM_object[RDM_index].dissimilarities
      
    mc.plotting.deep_data_plt.RDM_plotting(RDM, titelstring = "1 - Pearson's r", condition_name_string = cond_names_dict['1'])




# exchange a value of a certain voxel and save file
def save_changed_voxel_val(voxel_val, brain_map_to_store, RDM_reference_obj, voxel_coord, file_path, file_name, ref_image_for_affine):
    # import pdb; pdb.set_trace()
    x, y, z = ref_image_for_affine.shape
    affine_matrix = ref_image_for_affine.affine
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    brain_map = np.zeros([x*y*z])
    brain_map[list(RDM_reference_obj.rdm_descriptors['voxel_index'])] = [vox for vox in brain_map_to_store]
    brain_map = brain_map.reshape([x,y,z])
    
    brain_map[voxel_coord[0], voxel_coord[1], voxel_coord[2]] = voxel_val
    
    brain_map_nifti = nib.Nifti1Image(brain_map, affine=affine_matrix)
    brain_map_file = f"{file_path}/{file_name}_bin_diff.nii.gz"
    nib.save(brain_map_nifti, brain_map_file)




def RDM_plotting(RDM, titelstring, condition_name_string):
    fig, ax = plt.subplots(figsize=(5,4))
    cmaps.BlueYellowRed
    cmap = plt.get_cmap('BlueYellowRed')
    # import pdb; pdb.set_trace()
    im = ax.imshow(RDM, cmap=cmap, interpolation = 'none', aspect = 'equal', vmin=0, vmax=2); 
    for i in range(-1,len(RDM), 4):
        ax.axhline(i+0.5, color='white', linewidth=1)
        ax.axvline(i+0.5, color='white', linewidth=1)
    ticks = np.arange(0.5, len(RDM))    
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(condition_name_string, rotation=45, ha = 'right', fontsize=16)
    ax.set_yticklabels(condition_name_string, fontsize=16)
    # Adjust the appearance of ticks and grid lines
    ax.grid(False)
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(f"titelstring", rotation=-90, va="bottom")
    plt.tight_layout()  