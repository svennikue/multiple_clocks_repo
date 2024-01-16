#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:25:52 2023

create fMRI data RDMs


@author: xpsy1114
"""


from tqdm import tqdm
import numpy as np
import nibabel as nib
import os
import re
from rsatoolbox.inference import eval_fixed
import rsatoolbox.rdm as rsr
import rsatoolbox.data as rsd
import rsatoolbox
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight
from nilearn.image import load_img
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import mc

# import pdb; pdb.set_trace()  


subjects = ['sub-01']
task_halves = ['1', '2']
RDM_version = '05' # 04 is another try to bring the results back...'03' # 03 is teporal resolution = 1. 02 is for the report.
no_RDM_conditions = 80
load_old = True
regression_version = '06' #'04_pt01+_that_worked' 
# make all paths relative and adjust to both laptop and server!!
      
for sub in subjects:
    data_dir = f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}"
    if os.path.isdir(data_dir):
        print("Running on laptop.")
    else:
        data_dir = f"/home/fs0/xpsy1114/scratch/data/derivatives/{sub}"
        print(f"Running on Cluster, setting {data_dir} as data directory")
       
    RDM_dir = f"{data_dir}/beh/RDMs_{RDM_version}_glmbase_{regression_version}"
    if os.path.isdir(RDM_dir):
        print(f"RDM dir exists: {RDM_dir}")
    else:
        print(f"ERROR! RDM dir doesn't exist! ({RDM_dir})")
    
    # get a reference image to later project the results onto. This is usually
    # example_func from half 1, as this is where the data is corrected to.
    ref_img = load_img(f"{data_dir}/func/preproc_clean_01.feat/example_func.nii.gz")
    
    # Step 1: creating the searchlights
    # mask will define the searchlight positions.
    # IMPORTANT: on server, you still need to create this mask.
    mask = load_img(f"/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/{sub}/anat/grey_matter_func_bin.nii.gz")
    mask = mask.get_fdata()  
    # save this file to save time
    if load_old:
        centers = np.load(f"{RDM_dir}/searchlight_centers.npy", allow_pickle=True)
        neighbors = np.load(f"{RDM_dir}/searchlight_neihbors.npy", allow_pickle=True)
    else:
        # creating the searchlights
        centers, neighbors = get_volume_searchlight(mask, radius=3, threshold=0.5) # Found 175.483 searchlights
        # if I use the grey matter mask, then I find 144.905 searchlights
        # save this structure
        np.save(f"{RDM_dir}/searchlight_centers.npy", centers)
        np.save(f"{RDM_dir}/searchlight_neihbors.npy", neighbors)
            
    # Step 2: loading and computing the data RDMs
    data_RDM_file_2d = {}
    data_RDM_file = {}
    for task_half in task_halves:
        # load the relevant pre-processed task-half
        fmri_data_dir = f"{data_dir}/func/preproc_clean_0{task_half}.feat"
        pe_path = f"{data_dir}/func/glm_{regression_version}_pt0{task_half}.feat/stats"
        # define the naming conventions in this folder
        pes_I_want = re.compile(r'^pe([1-9]|[1-7][0-9]|80)\.nii\.gz$')
        # List all files in the folder
        files_in_pe_folder = os.listdir(pe_path)

        data_RDM_file[task_half] = [None] * no_RDM_conditions  # Initialize a list
        image_paths = [None] * no_RDM_conditions
        
        # Loop through the files and read in only those that are regressors I want e.g. pe01 - pe80
        for filename in files_in_pe_folder:
            match = pes_I_want.match(filename)
            if match:
                file_path = os.path.join(pe_path, filename)
                numeric_value = int(match.group(1)) - 1  # Extract the numeric value and convert to an index
                image_paths[numeric_value] = file_path  # save path to check if everything went fine later
                data_RDM_file[task_half][numeric_value] = nib.load(file_path).get_fdata()
        
        # Convert the list to a NumPy array
        data_RDM_file[task_half] = np.array(data_RDM_file[task_half])
        # reshape data so we have n_observations x n_voxels
        data_RDM_file_2d[task_half] = data_RDM_file[task_half].reshape([data_RDM_file[task_half].shape[0], -1])
        data_RDM_file_2d[task_half] = np.nan_to_num(data_RDM_file_2d[task_half]) # now this is 80timepoints x 746.496 voxels

    # define the conditions, combine both task halves
    data_nCond = data_RDM_file['1'].shape[0]
    data_conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(data_nCond)])), (1,2)).transpose(),160)  
    # now prepare the data RDM file.
    sessions = np.concatenate((np.zeros(int(data_RDM_file['1'].shape[0])), np.ones(int(data_RDM_file['2'].shape[0]))))   
    # final data RDM file; cross correlated between task-halves.
    data_RDM = get_searchlight_RDMs(data_RDM_file_2d, centers, neighbors, data_conds, method='crosscorr', cv_descr=sessions)
    mc.analyse.analyse_MRI_behav.save_as_nifti(data_RDM, file_path = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results", file_name="data_RDM", mask=mask, reference_image=ref_img)
    # save as a nifti so that I don't need to recompute always

    
    # Step 3: load and compute the model RDMs.

    # load the data files I created.
    location_data = np.load(os.path.join(RDM_dir, f"data_location_{sub}_fmri_both_halves.npy")) 
    clocks_data = np.load(os.path.join(RDM_dir, f"data_clock_{sub}_fmri_both_halves.npy"))
    midnight_data = np.load(os.path.join(RDM_dir, f"data_midnight_{sub}_fmri_both_halves.npy"))
    phase_data = np.load(os.path.join(RDM_dir, f"data_phase_{sub}_fmri_both_halves.npy"))
    state_data = np.load(os.path.join(RDM_dir, f"data_state_{sub}_fmri_both_halves.npy"))
    
    # to show how mine looked like
    # delete later!
    # my_loc_RDM = np.load(os.path.join(RDM_dir, f"RSM_location_{sub}_fmri_both_halves.npy"))
    # plt.figure(); plt.imshow(my_loc_RDM)

    # then create model RDMs.
    # 3.1
    # location RDM
    loc_data_test = mc.analyse.analyse_MRI_behav.prepare_model_data(location_data)
    import pdb; pdb.set_trace() 
    
    
    # location_data = location_data.transpose()
    # nCond = location_data.shape[0]/2
    # nVox = location_data.shape[1]
    # sessions = np.concatenate((np.zeros(int(np.shape(location_data)[0]/2)), np.ones(int(np.shape(location_data)[0]/2))))
    # des = {'subj': 1}
    # conds = np.reshape(np.tile((np.array(['cond_%02d' % x for x in np.arange(nCond)])), (1,2)).transpose(),160)
    # obs_des = {'conds': conds, 'sessions': sessions}
    # chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
    # loc_data = rsd.Dataset(measurements=location_data,
    #                    descriptors=des,
    #                    obs_descriptors=obs_des,
    #                    channel_descriptors=chn_des)
    
    loc_RDM = rsr.calc_rdm(loc_data, method='crosscorr', descriptor='conds', cv_descriptor='sessions')
    
    #loc_RDM = rsr.calc_rdm(loc_data, method='correlation')
    fig, ax, ret_vla = rsatoolbox.vis.show_rdm(loc_RDM)

    # then compute the location model.
    loc_model = rsatoolbox.model.ModelFixed('loc_model_only', loc_RDM)

    # Step 4: evaluate the model fit between model and data RDMs.
    # 4.1
    RDM_my_loc = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(loc_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in loc model'))
    mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=RDM_my_loc, data_RDM_file=data_RDM, file_path = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results", file_name= f"my_loc_{task_half}", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
    
     
    # clock RDM
    # 3.2
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
    
    # now do the same with my version as well so I can interpret the numbers.
    RDM_my_clock = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(clocks_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in clock model'))
    mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=RDM_my_clock, data_RDM_file=data_RDM, file_path = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results", file_name= f"my_clock_{task_half}", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
    
    # midnight RDM
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
    
    # now do the same with my version as well so I can interpret the numbers.
    RDM_my_midn = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(midnight_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in  midnight model'))
    mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=RDM_my_midn, data_RDM_file=data_RDM, file_path = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results", file_name= f"my_midn_{task_half}", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
    
    

    # phase_RDM
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
    
    # now do the same with my version as well so I can interpret the numbers.
    RDM_my_phase = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(phase_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in phase model'))
    mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=RDM_my_phase, data_RDM_file=data_RDM, file_path = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results", file_name= f"my_phase_{task_half}", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
    
    
    
    
    # state RDM
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
    # now do the same with my version as well so I can interpret the numbers.
    RDM_my_state = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(state_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in state model'))
    mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=RDM_my_state, data_RDM_file=data_RDM, file_path = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results", file_name= f"my_state_{task_half}", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
    
    
    
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
    
    
    # combined model 1
    # clocks, midnight and state.
    clocks_midn_states_RDM = rsatoolbox.rdm.concat(clocks_RDM, midnight_RDM, state_RDM)
    clocks_midn_states_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_RDM', clocks_midn_states_RDM)
    results_clocks_midn_states_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(clocks_midn_states_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model 1'))
    
    
    
    # combined model 2
    # clocks, midnight, state, phase and location.
    clocks_midn_states_loc_ph_RDM = rsatoolbox.rdm.concat(clocks_RDM, midnight_RDM, state_RDM, loc_RDM, phase_RDM)
    clocks_midn_states_loc_ph_model = rsatoolbox.model.ModelWeighted('clocks_midn_states_RDM', clocks_midn_states_loc_ph_RDM)
    # the first is t, the second beta. [est.tvalues[1:], est.params[1:]]
    results_clocks_midn_states_loc_ph_model = Parallel(n_jobs=3)(delayed(mc.analyse.analyse_MRI_behav.evaluate_model)(clocks_midn_states_loc_ph_model, d) for d in tqdm(data_RDM, desc='running GLM for all searchlights in combo model 2'))
    
    
    mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results", file_name= f"my_combo_clock_{task_half}", mask=mask, number_regr = 0, ref_image_for_affine_path=ref_img)
    mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results", file_name= f"my_combo_midn_{task_half}", mask=mask, number_regr = 1, ref_image_for_affine_path=ref_img)
    mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results", file_name= f"my_combo_state_{task_half}", mask=mask, number_regr = 2, ref_image_for_affine_path=ref_img)
    mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results", file_name= f"my_combo_loc_{task_half}", mask=mask, number_regr = 3, ref_image_for_affine_path=ref_img)
    mc.analyse.analyse_MRI_behav.save_RSA_result(result_file=results_clocks_midn_states_loc_ph_model, data_RDM_file=data_RDM, file_path = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results", file_name= f"my_combo_phase_{task_half}", mask=mask, number_regr = 4, ref_image_for_affine_path=ref_img)
    
 
        
    
    # ############################################
    
    # Last part: SAFE THE RESULTS!!
    # if save_all:
    #     ref_img = load_img(f"{fmri_data_dir}/{file}")
    #     affine_matrix = ref_img.affine
    #     if not os.path.exists(f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results/"):
    #         os.makedirs(f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results/")
            
            
    #     loc_nifti = nib.Nifti1Image(RDM_brain_loc, affine=affine_matrix)
    #     loc_file = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results/loc_model_RDM_0{task_half}.nii.gz"
        
    #     phase_nifti = nib.Nifti1Image(RDM_brain_phase, affine=affine_matrix)
    #     phase_file = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results/phase_model_RDM_0{task_half}.nii.gz"
        
    #     midnigth_nifti = nib.Nifti1Image(RDM_brain_midnight, affine=affine_matrix)
    #     midnight_file = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results/midnight_model_RDM_0{task_half}.nii.gz"
        
    #     clocks_nifti = nib.Nifti1Image(RDM_brain_clocks, affine=affine_matrix)
    #     clocks_file = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results/clocks_model_RDM_0{task_half}.nii.gz"
        
    #     state_nifti = nib.Nifti1Image(RDM_brain_state, affine=affine_matrix)
    #     state_file = f"{data_dir}/func/RSA_{RDM_version}_glmbase_{regression_version}/results/state_model_RDM_0{task_half}.nii.gz"
        
        

    #     # Save the NIfTI image to the 
    #     nib.save(loc_nifti, loc_file)
    #     nib.save(phase_nifti, phase_file)
    #     nib.save(midnigth_nifti, midnight_file)
    #     nib.save(clocks_nifti, clocks_file)
    #     nib.save(state_nifti, state_file)
        
        
        
        
        