#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:49:09 2024

surface based RSA

@author: xpsy1114
"""

import mc
import sys
import os
import pickle
import nibabel as nib
import numpy as np
from nilearn import datasets, surface
import nilearn as nl
from sklearn import neighbors
from joblib import Parallel, delayed
from tqdm import tqdm


fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")


# import pdb; pdb.set_trace() 
regression_version = '03-4' 
RDM_version = '03-1' 


if len (sys.argv) > 1:
    subj_no = sys.argv[1]
else:
    subj_no = '14'

subjects = [f"sub-{subj_no}"]
#subjects = ['sub-01']

load_old = False
visualise_RDMs = False


task_halves = ['1', '2']

print(f"Now running RSA for RDM version {RDM_version} based on subj GLM {regression_version} for subj {subj_no}")


models_I_want = mc.analyse.analyse_MRI_behav.models_I_want(RDM_version)


if regression_version in ['03', '04','03-99', '03-999', '03-9999', '03-l', '03-e']:
    no_RDM_conditions = 40 # only including rewards or only paths
elif regression_version in ['03-4', '04-4', '03-4-e', '03-4-l', '03-4-rep1', '03-4-rep2' , '03-4-rep3' , '03-4-rep4' ,'03-4-rep5' ]: # only including tasks without double reward locs: A,C,D  and only rewards
    no_RDM_conditions = 24

for sub in subjects:
    # load all EV files and then concatenate them such that it is a 4 dimensional fmri file.
    # call it fmri_img in the end.
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data"
        print(f"Running on Cluster, setting {data_dir} as data directory")
    if RDM_version in ['03-999']:
        RDM_dir = f"{data_dir}/derivatives/{sub}/beh/RDMs_03_glmbase_{regression_version}"
    else:
        RDM_dir = f"{data_dir}/derivatives/{sub}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"
    if not os.path.exists(RDM_dir):
        os.makedirs(RDM_dir)  
    results_dir = f"{data_dir}/derivatives/{sub}/func/RSA_{RDM_version}_glmbase_{regression_version}"   
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        os.makedirs(f"{results_dir}/results")
    # results_dir = f"{data_dir}/derivatives/{sub}/func/RSA_{RDM_version}_glmbase_{regression_version}/results"  
    fs_dir = f"{data_dir}/freesurfer/{sub}"
    results_dir = f"{fs_dir}/RSA_{RDM_version}_glmbase_{regression_version}/results"
    func_dir = f"{data_dir}/derivatives/{sub}/func"
    
    # load the file which defines the order of the model RDMs, and hence the data RDMs
    with open(f"{RDM_dir}/sorted_keys-model_RDMs.pkl", 'rb') as file:
        sorted_keys = pickle.load(file)
    with open(f"{RDM_dir}/sorted_regs.pkl", 'rb') as file:
        reg_keys = pickle.load(file)
            
    if regression_version in ['03-3', '03-4']:
        regression_version = '03'   
     
    pe_path_01 = f"{func_dir}/glm_{regression_version}_pt01.feat/stats"
    reading_in_EVs_dict_01 = {}   
    with open(f"{func_dir}/EVs_{regression_version}_pt01/task-to-EV.txt", 'r') as file:
        for line in file:
            index, name_ev = line.strip().split(' ', 1)
            name = name_ev.replace('ev_', '')
            reading_in_EVs_dict_01[f"{name}_EV_{int(index)+1}"] = os.path.join(pe_path_01, f"pe{int(index)+1}.nii.gz")
            
    pe_path_02 = f"{func_dir}/glm_{regression_version}_pt02.feat/stats"     
    reading_in_EVs_dict_02 = {}
    with open(f"{func_dir}/EVs_{regression_version}_pt02/task-to-EV.txt", 'r') as file:
        for line in file:
            index, name_ev = line.strip().split(' ', 1)
            name = name_ev.replace('ev_', '')
            reading_in_EVs_dict_02[f"{name}_EV_{int(index)+1}"] = os.path.join(pe_path_02, f"pe{int(index)+1}.nii.gz")
    
    # I need to do this slightly differently. I want to be super careful that I create 2 'identical' splits of data.
    # thus, check which folder has the respective task.
    mask = nib.load(f"{data_dir}/derivatives/{sub}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
    ref_img = nib.load(f"{func_dir}/preproc_clean_01.feat/example_func.nii.gz")
    ref_img_data = ref_img.get_fdata()
    fmri_img_list_first_half = np.empty((ref_img_data.shape[0], ref_img_data.shape[1], ref_img_data.shape[2], no_RDM_conditions*2))
    fmri_img_list_sec_half = np.empty((ref_img_data.shape[0], ref_img_data.shape[1], ref_img_data.shape[2], no_RDM_conditions*2))
    
    #data_RDM_file = {}
    reading_in_EVs_dict = {}
    image_paths = {}
    
    for split in sorted_keys:          
        i = -1
        image_paths[split] = [None] * no_RDM_conditions # Initialize a list for each half of the dictionary
        #data_RDM_file[split] = [None] * no_RDM_conditions  # Initialize a list for each half of the dictionary
        for EV_no, task in enumerate(sorted_keys[split]):
            for regressor_sets in reg_keys:
                if regressor_sets[0].startswith(task):
                    curr_reg_keys = regressor_sets
            for reg_key in curr_reg_keys:
                # print(f"now looking for {task}")
                for EV_01 in reading_in_EVs_dict_01:
                    if EV_01.startswith(reg_key):
                        i = i + 1
                        # print(f"looking for {task} and found it in 01 {EV_01}, index {i}")
                        image_paths[split][i] = reading_in_EVs_dict_01[EV_01]  # save path to check if everything went fine later
                        fmri_img_list_first_half[:,:,:,i] = nib.load(reading_in_EVs_dict_01[EV_01]).get_fdata()
                        # if i == 0:
                        #     fmri_img = nib.load(reading_in_EVs_dict_01[EV_01])
                        # else:
                        #     next_EV = nib.load(reading_in_EVs_dict_01[EV_01])
                        #     fmri_img = nl.image.concat_imgs([fmri_img, next_EV])
                        #data_RDM_file[split][i] = nib.load(reading_in_EVs_dict_01[EV_01]).get_fdata()
                        
                for EV_02 in reading_in_EVs_dict_02:
                    if EV_02.startswith(reg_key):
                        i = i + 1
                        # print(f"looking for {task} and found it in 01 {EV_02}, index {i}")
                        #data_RDM_file[split][i] = nib.load(reading_in_EVs_dict_02[EV_02]).get_fdata() 
                        image_paths[split][i] = reading_in_EVs_dict_02[EV_02]
                        fmri_img_list_sec_half[:,:,:,i] = nib.load(reading_in_EVs_dict_02[EV_02]).get_fdata()   

                        # fmri_img is nifti1.Nifti1Image (40,64,64,216)
                        # nifti1imiage object of nibabel.nifti1
                        # if i == 0:
                        #     fmri_img = nib.load(reading_in_EVs_dict_02[EV_02])
                        # else:
                        #     next_EV = nib.load(reading_in_EVs_dict_02[EV_02])
                        #     fmri_img = nl.image.concat_imgs([fmri_img, next_EV])
                        
        
        
        
        fmri_img_pt1 = nib.Nifti1Image(fmri_img_list_first_half, affine=ref_img.affine, header=ref_img.header)               
        fmri_img_pt2 = nib.Nifti1Image(fmri_img_list_sec_half, affine=ref_img.affine, header=ref_img.header)               
        
        fmri_img_stacked = np.stack((fmri_img_pt1,fmri_img_pt2),2)
        # I need to stack X tasklahf 1 and 2 like this; new_X = np.stack((X,X),2)
        # and also provide the adjacency matrix. that should be it!
        


        
    for hemi in ['lh', 'rh']:
        # # average voxels in a 4mm radius to the pial surface
        # radius = 5.0
        #pial_mesh = nib.freesurfer.io.read_geometry(f"{fs_dir}/surf/{hemi}/pial.T1")
        # /Users/xpsy1114/Documents/projects/multiple_clocks/data/freesurfer/sub-14/surf/rh.white           
        pial_mesh = nib.freesurfer.read_geometry(f"{fs_dir}/surf/{hemi}.pial.T1")
        white_mesh = nib.freesurfer.read_geometry(f"{fs_dir}/surf/{hemi}.white")
        vertices, faces = pial_mesh
        (type(vertices), vertices.shape, vertices.dtype)
        (type(faces), faces.shape, faces.dtype)
        # fmri_img is nifti1.Nifti1Image (40,64,64,216)
        # nifti1imiage object of nibabel.nifti1
        
        # averages the value of voxels of the volumentric data in a radius around the vertices.
        # each row in X corresponds to a vertex, and each column is a condition for the data
        # use nearest neighbors to avois smoothing or interpolation between voxels!
        X = surface.vol_to_surf(fmri_img_pt1, interpolation = 'nearest', surf_mesh=pial_mesh, inner_mesh=white_mesh, mask_img=mask).T
        # however now the same voxel might occur several times since the surface mesh is much finer than the voxels
        # find out how to 'disallow' the same voxel twice
        
        infl_mesh = nib.freesurfer.read_geometry(f"{fs_dir}/surf/{hemi}.inflated")
    
    
    
        ##### ALL OF THE BELOW CAN GO IN ONE FUNCTION!!!
        
        
        # these are vertices, [faces]
        # coords = weights corresponding to lengths of the edges between vertices
        coords, _ = surface.load_surf_mesh(infl_mesh)
        radius = 3.0
        nn = neighbors.NearestNeighbors(radius=radius)
        # based on coords/ faces, define searchlights
        # Compute the (weighted) graph of Neighbors for points in coords
        # Neighborhoods are restricted the points at a distance lower than radius.
        adjacency = nn.fit(coords).radius_neighbors_graph(coords).tolil()
        # stores in it's rows which voxels are in a certain radius.


        # alternatively, store between 50 and 100 values.
        # THIS ALL SHOULD PROBABLY GO INTO A FUNCTION.
        # Parameters
        min_neighbors = 50
        max_neighbors = 100
        
        # Initialize NearestNeighbors with the maximum number of neighbors
        nn = neighbors.NearestNeighbors(n_neighbors=max_neighbors)
        
        # Fit the NearestNeighbors model to the coordinates
        nn.fit(coords)
        adjacency = nn.fit(coords).radius_neighbors_graph(coords).tolil()
        
        # Ensure each row has at least `min_neighbors`
        for i in range(adjacency.shape[0]):
            if len(adjacency.rows[i]) < min_neighbors:
                # If the point has fewer than `min_neighbors`, find the nearest `min_neighbors`
                distances, indices = nn.kneighbors([coords[i]], n_neighbors=min_neighbors)
                for idx in indices[0]:
                    if idx != i:
                        adjacency[i, idx] = 1
        
        # Find and print the shortest row in the adjacency matrix
        shortest_row_length = float('inf')
        shortest_row_index = -1
        
        for i in range(adjacency.shape[0]):
            row_length = len(adjacency.rows[i])
            if row_length < shortest_row_length:
                shortest_row_length = row_length
                shortest_row_index = i
        
        print(f"The shortest row is at index {shortest_row_index} with length {shortest_row_length}")
        print(f"Shortest row neighbors: {adjacency.rows[shortest_row_index]}")

        # Compute the distances to the nearest neighbors
        distances, _ = nn.kneighbors(coords)
        
        # Get the distances to the 50th and 100th nearest neighbors for each point
        radii_50th = distances[:, min_neighbors-1]
        radii_100th = distances[:, max_neighbors-1]
        
        # Calculate the average radius for the 50th and 100th neighbors
        average_radius_50th = np.mean(radii_50th)
        average_radius_100th = np.mean(radii_100th)
        
        print(f"Average radius for {min_neighbors} neighbors: {average_radius_50th}")
        print(f"Average radius for {max_neighbors} neighbors: {average_radius_100th}")


        ##### ALL OF THE ABOVE CAN GO IN AN EXTERNAL FUNCTION!!!
        
        
        
        
        # Now, from here the question is if I can just take X and adjacency and 
        # estimate the fit between searchlight and model myself.
        # or using the functions I used before.
        
        # I need to be 100% sure about the mapping, but in theory,
        # adjacency should tell me for each value in the 135804 vertices in X,
        # which other vertices to include in the searchlight.
        # with that, I should be able to write a function that just takes these 
        # values in the condition dimensions, and computes the similariyt.
        # maybe I can even build that on top of the RSA toolbox stuff, such 
        # that I can make it fast and efficient.
        
        
        # TRY THIS ONE!
        data_RDMS = mc.analyse.data_RDM_surface_searchlights(fmri_img_stacked, adjacency, method='crosscorr', n_jobs = 2, verbose=1)
        
        
        # Step 3: load and compute the model RDMs.
        # load the data files I created.
        data_dirs = {}
        for model in models_I_want:
            if RDM_version in ['999', '9999']: # potentially delete?? this is now 03-99 nd 03-999
                RDM_dir = f"{data_dir}/beh/RDMs_09_glmbase_{regression_version}" # potentially delete??
                
            # actually, now, instead load the already readily computed upper triangle of the RDMs
            data_dirs[model]= np.load(os.path.join(RDM_dir, f"data{model}_{sub}_fmri_both_halves.npy")) 
            
            
        RDM_my_model_dir = {}
        for model in data_dirs:
            model_RDM = 4
            # next step: evaluate!
            
            RDM_my_model_dir[model] = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_surface_searchlights)(model_RDM, d) for d in tqdm(data_RDMS, desc=f"running GLM for all searchlights in {model}"))
            
            
            # save the file somehow, and plot it
            # VISUALISATION
            from nilearn import plotting

            chance = 0.5
            plotting.plot_surf_stat_map(
                infl_mesh,
                RDM_my_model_dir[model],
                view="medial",
                colorbar=True,
                threshold=0.1,
                bg_map=fsaverage[f"sulc_{hemi}"],
                title="Accuracy map, left hemisphere",
            )
            plotting.show()  
            
