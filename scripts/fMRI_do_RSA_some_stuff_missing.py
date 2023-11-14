#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:25:52 2023

create fMRI data RDMs


@author: xpsy1114
"""

import statsmodels.api as sm
from tqdm import tqdm
import numpy as np
import nibabel as nib
import os
import re
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed
import rsatoolbox.rdm as rsr
import rsatoolbox.data as rsd
import rsatoolbox
from nilearn.image import resample_to_img
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight
from rsatoolbox.util.searchlight import get_volume_searchlight
from rsatoolbox.data.dataset import Dataset
from nilearn.image import load_img
from nilearn.masking import compute_brain_mask
from nilearn import plotting
from rsatoolbox.rdm.calc import calc_rdm
from rsatoolbox.rdm import RDMs
import pickle
# from rich.progress import track
from os import path, makedirs
#import fire
from nilearn.glm.contrasts import expression_to_contrast_vector, compute_contrast
from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as plt
from nilearn.image import new_img_like
import pandas as pd
import nibabel as nib
import seaborn as sns
from nilearn import plotting
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed
from rsatoolbox.rdm import RDMs
from glob import glob
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight


import mc
import matplotlib.pyplot as plt

# import pdb; pdb.set_trace()

subjects = ['sub-01']
task_halves = ['1', '2']
RDM_version = '01'
no_RDM_conditions = 80
save_all = False
# careful this still doesnt work in row 74 for the pattern thingy

for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    RDM_dir = f"{data_dir}/beh/RDMs_{RDM_version}"
    for task_half in task_halves:
        fmri_data_dir = f"{data_dir}/func/preproc_clean_0{task_half}.feat"
        file = "example_func.nii.gz"
        ref_img = load_img(f"{fmri_data_dir}/{file}")
        x, y, z = ref_img.get_fdata().shape
        ref_img = ref_img.get_fdata()
        
        
        # Prepare searchlight positions
        mask = load_img(f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/func/mask_0{task_half}_mask.nii.gz")
        mask = mask.get_fdata()
        
        # ok for some reason this doenst work
        # CONTINUE HERE!!!
        # resample the 
        # resampled_image = resample_to_img(mask, ref_img)
        
        
        # this doesnt work with spyder.
        # plotting.view_img(mask, cmap='gray', title='Brain Mask')
        # I am a bit skeptical how this works in 3d but the searchlight thingy only 
        # works in 2d 
        centers, neighbors = get_volume_searchlight(mask, radius=3, threshold=0.5)
        
        
        pe_path = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/glm_04_pt0{task_half}.feat/stats"
        
        # Loop through files in the folder
        # define the naming conventions in this folder
        pes_I_want = re.compile(r'^pe([1-9]|[1-7][0-9]|80)\.nii\.gz$')
        # List all files in the folder
        files_in_pe_folder = os.listdir(pe_path)
        
        # loop over all images
        data_RDM_file = np.zeros((no_RDM_conditions, x, y, z))
        
        # Loop through the files and read in only those that match the pattern
        image_paths = []
        i = 0
        for filename in files_in_pe_folder:
            if pes_I_want.match(filename):
                file_path = os.path.join(pe_path, filename)
                image_paths.append(os.path.join(file_path, filename))  # Get the full file path
                data_RDM_file[i] = nib.load(file_path).get_fdata()
                i += 1
        
            
        # STEP 2: get RDM for each voxel
        # reshape data so we have n_observastions x n_voxels
        data_RDM_file_2d = data_RDM_file.reshape([data_RDM_file.shape[0], -1])
        data_RDM_file_2d = np.nan_to_num(data_RDM_file_2d) # now this is 80timepoints x 746.496 voxels
        
        # only one pattern per image
        image_value = np.arange(len(image_paths))
        
        # CONTINUE HERE!!
        # also double check glm_04!! pt 2 looks a bit weird????
        # import pdb; pdb.set_trace()
        
        # Get RDMs
        
        # this should give me a file whitch len(centers) amount of searchlights that have len(neighbours) amount of voxels each
        # centers is probably the index in which data_RDM_file_2d is divided. 
        
        # this is a collection of objects: 
        data_RDM = get_searchlight_RDMs(data_RDM_file_2d, centers, neighbors, image_value, method='correlation')
        
        # then load the data RDMs.
        # for some reason, rsatoolbox wants its objects and not my matrices.
        # location_data_RDM = np.load(os.path.join(RDM_dir, f"RSM_location_{sub}_fmri_pt{task_half}.npy"))
        # clocks_data_RDM = np.load(os.path.join(RDM_dir, f"RSM_clock_{sub}_fmri_pt{task_half}.npy"))
        # midnight_data_RDM = np.load(os.path.join(RDM_dir, f"RSM_midnight_{sub}_fmri_pt{task_half}.npy"))
        # phase_data_RDM = np.load(os.path.join(RDM_dir, f"RSM_phase_{sub}_fmri_pt{task_half}.npy"))
        state_data_RDM = np.load(os.path.join(RDM_dir, f"RSM_state_{sub}_fmri_pt{task_half}.npy"))
        
        # load the data files in case I want to use these instead.
        location_data = np.load(os.path.join(RDM_dir, f"data_location_{sub}_fmri_pt{task_half}.npy"))
        clocks_data = np.load(os.path.join(RDM_dir, f"data_clock_{sub}_fmri_pt{task_half}.npy"))
        midnight_data = np.load(os.path.join(RDM_dir, f"data_midnight_{sub}_fmri_pt{task_half}.npy"))
        phase_data = np.load(os.path.join(RDM_dir, f"data_phase_{sub}_fmri_pt{task_half}.npy"))
        state_data = np.load(os.path.join(RDM_dir, f"data_state_{sub}_fmri_pt{task_half}.npy"))
        
        # first, create the RDMs from the model files (model_data)
        location_data = location_data.transpose()
        nCond = location_data.shape[0]
        nVox = location_data.shape[1]
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        loc_data = rsd.Dataset(measurements=location_data,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)
        
        loc_RDM = rsr.calc_rdm(loc_data, method='correlation')
        fig, ax, ret_vla = rsatoolbox.vis.show_rdm(loc_RDM)
        
        # then compute the location model.
        loc_model = rsatoolbox.model.ModelFixed('loc_model_only', loc_RDM)
        eval_results_loc = evaluate_models_searchlight(data_RDM, loc_model, eval_fixed, method='spearman', n_jobs=3)
        # get the evaulation score for each voxel
        # We only have one model, but evaluations returns a list. By using float we just grab the value within that list
        eval_score_loc = [np.float(e.evaluations) for e in eval_results_loc]
        # Create an 3D array, with the size of mask, and
        x, y, z = mask.shape
        RDM_brain_loc = np.zeros([x*y*z])
        RDM_brain_loc[list(data_RDM.rdm_descriptors['voxel_index'])] = eval_score_loc
        RDM_brain_loc = RDM_brain_loc.reshape([x, y, z])
        
        
        
        # do the same with the clocks.
        # first, create the RDMs from the clocks model files (clocks_data)
        clocks_data = clocks_data.transpose()
        nCond = clocks_data.shape[0]
        nVox = clocks_data.shape[1]
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        clocks_data = rsd.Dataset(measurements=clocks_data,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)
        
        clocks_RDM = rsr.calc_rdm(clocks_data, method='correlation')
        fig, ax, ret_vla = rsatoolbox.vis.show_rdm(clocks_RDM)
        
        # then compute the clocks model.
        clocks_model = rsatoolbox.model.ModelFixed('clocks_model_only', clocks_RDM)
        # compute phase RSA
        eval_results_clocks = evaluate_models_searchlight(data_RDM, clocks_model, eval_fixed, method='spearman', n_jobs=3)
        # get the evaulation score for each voxel
        # We only have one model, but evaluations returns a list. By using float we just grab the value within that list
        eval_score_clocks = [np.float(e.evaluations) for e in eval_results_clocks]
        # Create an 3D array, with the size of mask, and
        x, y, z = mask.shape
        RDM_brain_clocks = np.zeros([x*y*z])
        RDM_brain_clocks[list(data_RDM.rdm_descriptors['voxel_index'])] = eval_score_clocks
        RDM_brain_clocks = RDM_brain_clocks.reshape([x, y, z])
        
        
        # then do the same with the midnight model.
        # first, create midnight_data RDM
        midnight_data = midnight_data.transpose()
        nCond = midnight_data.shape[0]
        nVox = midnight_data.shape[1]
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        midnight_model_RDM = rsd.Dataset(measurements=midnight_data,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)
        
        midnight_RDM = rsr.calc_rdm(midnight_model_RDM, method='correlation')
        fig, ax, ret_vla = rsatoolbox.vis.show_rdm(midnight_RDM)
        # set up model
        midnight_model = rsatoolbox.model.ModelFixed('midnight_model_only', midnight_RDM)
        # compute phase RSA
        eval_results_midnight = evaluate_models_searchlight(data_RDM, midnight_model, eval_fixed, method='spearman', n_jobs=3)
        # get the evaulation score for each voxel
        # We only have one model, but evaluations returns a list. By using float we just grab the value within that list
        eval_score_midnight = [np.float(e.evaluations) for e in eval_results_midnight]
        # Create an 3D array, with the size of mask, and
        x, y, z = mask.shape
        RDM_brain_midnight = np.zeros([x*y*z])
        RDM_brain_midnight[list(data_RDM.rdm_descriptors['voxel_index'])] = eval_score_midnight
        RDM_brain_midnight = RDM_brain_midnight.reshape([x, y, z])
        
        

        # phase_data
        # create the phase RDM object
        phase_data = phase_data.transpose()
        nCond = phase_data.shape[0]
        nVox = phase_data.shape[1]
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        phase_model_data = rsd.Dataset(measurements=phase_data,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)
        
        phase_RDM = rsr.calc_rdm(phase_model_data, method='correlation')
        fig, ax, ret_vla = rsatoolbox.vis.show_rdm(phase_RDM)

        # set up model
        phase_model = rsatoolbox.model.ModelFixed('phase_model_only', phase_RDM)
        predict_phase = phase_model.predict()
        
        
        # compute phase RSA
        eval_results_phase = evaluate_models_searchlight(data_RDM, phase_model, eval_fixed, method='spearman', n_jobs=3)
        # get the evaulation score for each voxel
        # We only have one model, but evaluations returns a list. By using float we just grab the value within that list
        eval_score_phase = [np.float(e.evaluations) for e in eval_results_phase]
        # Create an 3D array, with the size of mask, and
        x, y, z = mask.shape
        RDM_brain_phase = np.zeros([x*y*z])
        RDM_brain_phase[list(data_RDM.rdm_descriptors['voxel_index'])] = eval_score_phase
        RDM_brain_phase = RDM_brain_phase.reshape([x, y, z])
        
        
        
        # state_data
        # create the state RDM object
        state_data = state_data.transpose()
        nCond =  state_data.shape[0]
        nVox = state_data.shape[1]
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        state_model_data = rsd.Dataset(measurements=state_data,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)
        
        state_RDM = rsr.calc_rdm(state_model_data, method='correlation')
        fig, ax, ret_vla = rsatoolbox.vis.show_rdm(state_RDM)

        # set up model
        state_model = rsatoolbox.model.ModelFixed('state_model_only', state_RDM)
        # compute phase RSA
        eval_results_state = evaluate_models_searchlight(data_RDM, state_model, eval_fixed, method='spearman', n_jobs=3)
        # get the evaulation score for each voxel
        # We only have one model, but evaluations returns a list. By using float we just grab the value within that list
        eval_score_state = [np.float(e.evaluations) for e in eval_results_state]
        # Create an 3D array, with the size of mask, and
        x, y, z = mask.shape
        RDM_brain_state = np.zeros([x*y*z])
        RDM_brain_state[list(data_RDM.rdm_descriptors['voxel_index'])] = eval_score_state
        RDM_brain_state = RDM_brain_state.reshape([x, y, z])
        
        
        # create a dictionary which shows how much model RDMs overlap
        overlap_corr_model_RDMs ={}
        overlap_corr_model_RDMs['loc_with_midnight'] = np.corrcoef(loc_RDM.dissimilarities, midnight_RDM.dissimilarities)[1][0]
        overlap_corr_model_RDMs['loc_with_clocks'] = np.corrcoef(loc_RDM.dissimilarities, clocks_RDM.dissimilarities)[1][0]
        overlap_corr_model_RDMs['loc_with_state'] = np.corrcoef(loc_RDM.dissimilarities, state_RDM.dissimilarities)[1][0]
        overlap_corr_model_RDMs['loc_with_phase'] = np.corrcoef(loc_RDM.dissimilarities, phase_RDM.dissimilarities)[1][0]
        overlap_corr_model_RDMs['midnight_with_clocks'] = np.corrcoef(midnight_RDM.dissimilarities, clocks_RDM.dissimilarities)[1][0]
        overlap_corr_model_RDMs['midnight_with_state'] = np.corrcoef(midnight_RDM.dissimilarities, state_RDM.dissimilarities)[1][0]
        overlap_corr_model_RDMs['midnight_with_phase'] = np.corrcoef(midnight_RDM.dissimilarities, phase_RDM.dissimilarities)[1][0]
        overlap_corr_model_RDMs['clocks_with_state'] = np.corrcoef(clocks_RDM.dissimilarities, state_RDM.dissimilarities)[1][0]
        overlap_corr_model_RDMs['clocks_with_phase'] = np.corrcoef(clocks_RDM.dissimilarities, phase_RDM.dissimilarities)[1][0]
        overlap_corr_model_RDMs['state_with_phase'] = np.corrcoef(state_RDM.dissimilarities, phase_RDM.dissimilarities)[1][0]
        # ignore this upper triangle for now
        # loc_model = ModelFixed('Location RDM', upper_tri(location_data_RDM))
        
        

        #import pdb; pdb.set_trace() 
        
        # CAREFUL!! I am not sure if this is fitting the data pitting the models against each other...
        def my_eval(model, data):
              "Handle one voxel, copy the code that exists already for the neural data"
              X = sm.add_constant(model.rdm.transpose());
              Y = data.dissimilarities.transpose();
              est = sm.OLS(Y, X).fit()
              return est.tvalues[1:], est.params[1:]
          
            
        # combined model
        clocks_midn_states_RDM = rsatoolbox.rdm.concat(clocks_RDM, midnight_RDM, state_RDM)
        clocks_midn_states_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_RDM', clocks_midn_states_RDM)
        results_clocks_midn_states_model = Parallel(n_jobs=3)(delayed(my_eval)(clocks_midn_states_model, d) for d in tqdm(data_RDM, desc='bla'))
        
        x, y, z = mask.shape
        RDM_brain_cl_midn_st_clocks = np.zeros([x*y*z])
        RDM_brain_cl_midn_st_clocks[list(data_RDM.rdm_descriptors['voxel_index'])] = [vox[0] for vox in results_clocks_midn_states_model]
        RDM_brain_cl_midn_st_clocks = RDM_brain_cl_midn_st_clocks.reshape([x,y,z])
        
        RDM_brain_cl_midn_st_midn = np.zeros([x*y*z])
        RDM_brain_cl_midn_st_midn[list(data_RDM.rdm_descriptors['voxel_index'])] = [vox[1] for vox in results_clocks_midn_states_model]
        RDM_brain_cl_midn_st_midn = RDM_brain_cl_midn_st_midn.reshape([x,y,z])
        
        RDM_brain_cl_midn_st_state = np.zeros([x*y*z])
        RDM_brain_cl_midn_st_state[list(data_RDM.rdm_descriptors['voxel_index'])] = [vox[2] for vox in results_clocks_midn_states_model]
        RDM_brain_cl_midn_st_state = RDM_brain_cl_midn_st_state.reshape([x,y,z])
        

        
        

        
        # # use something like 
        # results_reg = mc.simulation.RDMs.GLM_RDMs(RSM_neurons, regressors, mask_within, no_tasks = len(task_configs), plotting= False)
        # # to get my own model fitting/ evaluation
        
        
        # GLM_RDMs(data_matrix, regressor_dict, mask_within = True, no_tasks = None, t_val = True, plotting = False)
        
        # ################ 
        # # this is a bit of a scratchbook rn inbetween here 
        # # Test with this:
        # results = [my_eval(models, x) for x in sl_RDM]
        
        # # When it works, use this:
        # results = Parallel(n_jobs=n_jobs)(delayed(my_eval)( models, x, ) for x in sl_RDM)
        
        
        
        
        
        # results = [my_evla(models, x) for x in sl_RDM]
        
        
        # combo_RDM = rsatoolbox.rdm.concat(clocks_RDM, state_RDM)
        # combo_model = rsatoolbox.model.ModelWeighted('combined_model', combo_RDM)
        
        # def my_eval(model, data):
        #     # print('fitting 1 sl')
        #     return model.fit(data)        
        
        # results_combo = Parallel(n_jobs=3)(delayed(my_eval)(combo_model, d) for d in data_RDM)
        
        # searchlight_fit = combo_model.fit(data_RDM[0])
        
        
        # eval_results_combo = evaluate_models_searchlight(data_RDM, combo_model, eval_fixed, method='spearman', n_jobs=3)
        # eval_results_combo = evaluate_models_searchlight(data_RDM, combo_model, rsatoolbox.model.fitter.fit_regress, method='spearman', n_jobs=3)
        
        # rsatoolbox.fitter.fit_regress 
        
        
        
        
        
        
        ############################################
        
        # Last part: SAFE THE RESULTS!!
        if save_all:
            ref_img = load_img(f"{fmri_data_dir}/{file}")
            affine_matrix = ref_img.affine
            if not os.path.exists(f"{data_dir}/func/RSA_{RDM_version}/results/"):
                os.makedirs(f"{data_dir}/func/RSA_{RDM_version}/results/")
                
                
            loc_nifti = nib.Nifti1Image(RDM_brain_loc, affine=affine_matrix)
            loc_file = f"{data_dir}/func/RSA_{RDM_version}/results/loc_model_RDM_0{task_half}.nii.gz"
            
            phase_nifti = nib.Nifti1Image(RDM_brain_phase, affine=affine_matrix)
            phase_file = f"{data_dir}/func/RSA_{RDM_version}/results/phase_model_RDM_0{task_half}.nii.gz"
            
            midnigth_nifti = nib.Nifti1Image(RDM_brain_midnight, affine=affine_matrix)
            midnight_file = f"{data_dir}/func/RSA_{RDM_version}/results/midnight_model_RDM_0{task_half}.nii.gz"
            
            clocks_nifti = nib.Nifti1Image(RDM_brain_clocks, affine=affine_matrix)
            clocks_file = f"{data_dir}/func/RSA_{RDM_version}/results/clocks_model_RDM_0{task_half}.nii.gz"
            
            state_nifti = nib.Nifti1Image(RDM_brain_state, affine=affine_matrix)
            state_file = f"{data_dir}/func/RSA_{RDM_version}/results/state_model_RDM_0{task_half}.nii.gz"
            
            # combo_state_nifti = nib.Nifti1Image(RDM_brain_cl_midn_st_state, affine=affine_matrix)
            # combo_state_file = f"{data_dir}/func/RSA_{RDM_version}/results/combo_RDM_state_0{task_half}.nii.gz"
            
            # combo_midn_nifti = nib.Nifti1Image(RDM_brain_cl_midn_st_midn, affine=affine_matrix)
            # combo_midn_file = f"{data_dir}/func/RSA_{RDM_version}/results/combo_RDM_midn_0{task_half}.nii.gz"
            
            # combo_clocks_nifti = nib.Nifti1Image(RDM_brain_cl_midn_st_clocks, affine=affine_matrix)
            # combo_clocks_file = f"{data_dir}/func/RSA_{RDM_version}/results/combo_RDM_clocks_0{task_half}.nii.gz"
            
            # Save the NIfTI image to the 
            nib.save(loc_nifti, loc_file)
            nib.save(phase_nifti, phase_file)
            nib.save(midnigth_nifti, midnight_file)
            nib.save(clocks_nifti, clocks_file)
            nib.save(state_nifti, state_file)
            
            # nib.save(combo_clocks_nifti, combo_clocks_file)
            # nib.save(combo_midn_nifti, combo_midn_file)
            # nib.save(combo_state_nifti, combo_state_file)

        # KEEP MAYBE
        # plt.figure()
        # sns.histplot(eval_score)
        # plt.title('Distributions of correlations', size=18)
        # plt.ylabel('Occurance', size=18)
        # plt.xlabel('Spearmann correlation', size=18)
        # sns.despine()
        # plt.show()
        
        
        
        
        # DELETE 
        
        # lets plot the voxels above the 99th percentile
        # threshold = np.percentile(eval_score, 99)
        # plot_img = new_img_like(ref_img, RDM_brain)
        
        # # define this
        # import matplotlib.colors
        # def RDMcolormapObject(direction=1):
        #     """
        #     Returns a matplotlib color map object for RSA and brain plotting
        #     """
        #     if direction == 0:
        #         cs = ['yellow', 'red', 'gray', 'turquoise', 'blue']
        #     elif direction == 1:
        #         cs = ['blue', 'turquoise', 'gray', 'red', 'yellow']
        #     else:
        #         raise ValueError('Direction needs to be 0 or 1')
        #     cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cs)
        #     return cmap


        # cmap = RDMcolormapObject()
        
        # coords = range(-20, 40, 5)
        # fig = plt.figure(figsize=(12, 3))
        
        # display = plotting.plot_stat_map(
        #         plot_img, colorbar=True, cut_coords=coords,threshold=threshold,
        #         display_mode='z', draw_cross=False, figure=fig,
        #         title=f'Animal model evaluation', cmap=cmap,
        #         black_bg=False, annotate=False)
        # plt.show()





        
        
        
        