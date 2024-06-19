#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 13:49:09 2024

surface based RSA

@author: xpsy1114
"""

from tqdm import tqdm
import numpy as np
import nibabel as nib
import os
import rsatoolbox.rdm as rsr
import rsatoolbox
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs
from nilearn.image import load_img
from nilearn import plotting
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import mc
import pickle
import sys
import random

# from surfer import Brain

regression_version = '03' 
RDM_version = '02' 


# import pdb; pdb.set_trace() 
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

# for sub in subjects:
#     data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data"
#     if os.path.isdir(data_dir):
#         print("Running on laptop.")
#     else:
#         data_dir = f"/home/fs0/xpsy1114/scratch/data"
#         print(f"Running on Cluster, setting {data_dir} as data directory")
#     if RDM_version in ['03-999']:
#         RDM_dir = f"{data_dir}/derivatives/{sub}/beh/RDMs_03_glmbase_{regression_version}"
#     else:
#         RDM_dir = f"{data_dir}/derivatives/{sub}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"
#     if not os.path.exists(RDM_dir):
#         os.makedirs(RDM_dir)  
#     results_dir = f"{data_dir}/derivatives/{sub}/func/RSA_{RDM_version}_glmbase_{regression_version}"   
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
#         os.makedirs(f"{results_dir}/results")
#     # results_dir = f"{data_dir}/derivatives/{sub}/func/RSA_{RDM_version}_glmbase_{regression_version}/results"  
#     fs_dir = f"{data_dir}/freesurfer/{sub}"
#     results_dir = f"{fs_dir}/RSA_{RDM_version}_glmbase_{regression_version}/results"
    
#     reading_in_EVs_dict = {task_half: "" for task_half in ['01', '02']}
#     for task_half in ['01', '02']:
#         pe_path = f"{fs_dir}/glm_{regression_version}_pt{task_half}.feat"
# #         with open(f"{data_dir}/derivatives/{sub}/func/EVs_{regression_version}_pt{task_half}/task-to-EV.txt", 'r') as file:
# #             for line in file:
# #                 index, name_ev = line.strip().split(' ', 1)
# #                 name = name_ev.replace('ev_', '')
# #                 reading_in_EVs_dict[task_half][f"{name}_EV_{int(index)+1}"] = os.path.join(pe_path, f"pe{int(index)+1}.nii.gz")
# # #                reading_in_EVs_dict_01[f"{name}_EV_{int(index)+1}"] = os.path.join(pe_path_01, f"pe{int(index)+1}.nii.gz")
                
    
    
#     # this guys says that mri_vol2surf can be performing poorly. 
    
#     # surface = nib.freesurfer.io.read_geometry(f"{pe_path}/rh.pe40.mgh")
#     surface = nib.load(f"{fs_dir}/testmri_vol2surf.mgz")
#     fMRI_vertices = surface.get_fdata() # 1 value per vertex
    
#     pial = nib.freesurfer.read_geometry(f"{fs_dir}/surf/lh.pial.T1")
#     vertices, faces = pial
#     (type(vertices), vertices.shape, vertices.dtype)
#     (type(faces), faces.shape, faces.dtype)
#     # pial, white, inflated, sphere,... all have the same set of vertices and faces in the same order
#     # per subject. 
#     # however, subjects and even within subjects but across hemisphere might have
#     # different numbers of vertices.
    
#     # if I understand correctly, faces are the coordinates of the respective vertices.
#     # so vertices would be the values; while faces tell me how each vertex has to be put together
#     # on the surface. e.g. if the first row in faces is (1,4,88), then that means 2nd, 5th and 89th vertex is 
#     # corners of the first triangle on the cortical surface.
    
    
#     # ok this doesnt work... find another way.
#     plotting.plot_surf_stat_map((vertices, faces), surface, hemi='left', view='lateral', colorbar=True)
#     plotting.show()
    
#     # surface = nib.load(f"{fs_dir}/test-vol2surf.gii")
#     # surface = nib.load(f"{pe_path}/rh.pe40.mgh")
    
    

#     # example_func from half 1, as this is where the data is corrected to.
#     # ref_img = load_img(f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz")
    
#     # load the file which defines the order of the model RDMs, and hence the data RDMs
#     # with open(f"{RDM_dir}/sorted_keys-model_RDMs.pkl", 'rb') as file:
#     #     sorted_keys = pickle.load(file)
#     # with open(f"{RDM_dir}/sorted_regs.pkl", 'rb') as file:
#     #     reg_keys = pickle.load(file)

  
#     pe_path_01 = f"{data_dir}/func/glm_{regression_version}_pt01.feat/stats"
#     reading_in_EVs_dict_01 = {}   
#     with open(f"{data_dir}/func/EVs_{regression_version}_pt01/task-to-EV.txt", 'r') as file:
#         for line in file:
#             index, name_ev = line.strip().split(' ', 1)
#             name = name_ev.replace('ev_', '')
#             reading_in_EVs_dict_01[f"{name}_EV_{int(index)+1}"] = os.path.join(pe_path_01, f"pe{int(index)+1}.nii.gz")
            
#     pe_path_02 = f"{data_dir}/func/glm_{regression_version}_pt02.feat/stats"     
#     reading_in_EVs_dict_02 = {}
#     with open(f"{data_dir}/func/EVs_{regression_version}_pt02/task-to-EV.txt", 'r') as file:
#         for line in file:
#             index, name_ev = line.strip().split(' ', 1)
#             name = name_ev.replace('ev_', '')
#             reading_in_EVs_dict_02[f"{name}_EV_{int(index)+1}"] = os.path.join(pe_path_02, f"pe{int(index)+1}.nii.gz")
    
    
    

    
    
    # Step 1: creating the searchlights
    # mask will define the searchlight positions, in pt01 space because that is 
    # where the functional files have been registered to.
    # mask = load_img(f"{data_dir}/anat/grey_matter_mask_func_01.nii.gz")
    # mask = load_img(f"{data_dir}/anat/{sub}_T1w_noCSF_brain_mask_bin_func_01.nii.gz")
    # mask = mask.get_fdata()  
    
# tutorial https://nilearn.github.io/dev/auto_examples/02_decoding/plot_haxby_searchlight_surface.html 
# LOAD THE DATASET  
import pandas as pd
from nilearn import datasets

# We fetch 2nd subject from haxby datasets (which is default)
haxby_dataset = datasets.fetch_haxby()

fmri_filename = haxby_dataset.func[0]
labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
y = labels["labels"]
run = labels["chunks"]

# RESTIRCT TO FACES AND HOUSES
from nilearn.image import index_img

condition_mask = y.isin(["face", "house"])
# fmri_img was 40x64x64 and 1578 timepoints.
# fmri_img is nifti1.Nifti1Image (40,64,64,216)

fmri_img = index_img(fmri_filename, condition_mask)
# I am masking for condition, so now it's only 216 conditions long.
# todo: prep 1 fMRI file which is all conditions concatenated.
y, run = y[condition_mask], run[condition_mask]

# SURFACE BOLD RESPONSE
from sklearn import neighbors
from nilearn import datasets, surface
# Fetch a coarse surface of the left hemisphere only for speed
fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
hemi = "left"

# Average voxels 5 mm close to the 3d pial surface
radius = 5.0
pial_mesh = fsaverage[f"pial_{hemi}"]
# X = surface.vol_to_surf(fmri_img, pial_mesh, radius=radius).T
white_mesh = fsaverage[f"white_{hemi}"]
X = surface.vol_to_surf(fmri_img, surf_mesh=pial_mesh, interpolation = 'nearest', inner_mesh=white_mesh,).T


# To define the :term:`BOLD` responses
# to be included within each searchlight "sphere"
# we define an adjacency matrix based on the inflated surface vertices such
# that nearby surfaces are concatenated within the same searchlight.

infl_mesh = fsaverage[f"infl_{hemi}"]

# these are vertices, [faces]
# coords = weights corresponding to lengths of the edges between vertices
coords, _ = surface.load_surf_mesh(infl_mesh)
radius = 7.0
nn = neighbors.NearestNeighbors(radius=radius)
# based on coords/ faces, define searchlights
# Compute the (weighted) graph of Neighbors for points in coords
# Neighborhoods are restricted the points at a distance lower than radius.
adjacency = nn.fit(coords).radius_neighbors_graph(coords).tolil()
    

# SEARCHLIHGT COMPUTATION
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from nilearn.decoding.searchlight import search_light

# Simple linear estimator preceded by a normalization step
estimator = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10.0))

# Define cross-validation scheme
cv = KFold(n_splits=3, shuffle=False)

# Cross-validated search light
# X = fMRI data [conditions x vertices], y = conditions, adjacency per vertex
# adjacency defines for each feature the neigbhoring features following a given structure of the data.
scores = search_light(X, y, estimator, adjacency, cv=cv, n_jobs=2)


# or alternatively, my new function.

# input to this shall be model RDMs, 
# my_scores = mc.analyse.SK_searchlight(X, y, estimator, adjacency, cv=cv, n_jobs=2)




# adjacency.rows[list_i] = list_rows
# # list_rows : array of arrays of int adjacency rows. For a voxel with index i in X, 
# # list_rows[i] is the list of neighboring voxels indices (in X).
# for i, row in enumerate(list_rows):
#     X[:, row]
            
    
# VISUALISATION
from nilearn import plotting

chance = 0.5
plotting.plot_surf_stat_map(
    infl_mesh,
    scores - chance,
    view="medial",
    colorbar=True,
    threshold=0.1,
    bg_map=fsaverage[f"sulc_{hemi}"],
    title="Accuracy map, left hemisphere",
)
plotting.show()   
    