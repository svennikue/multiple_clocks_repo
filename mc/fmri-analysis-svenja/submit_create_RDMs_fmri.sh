#!/bin/sh
# Run subject-level RSA analysis.
# submit like bash submit_RSA_fmri.sh
# requires results from subject_GLM_loc_press_preICA.sh

# NOTE
# first activate the virtual envirnment in the shell you are running this from.
# unfortunately doesn't work with putting the command in here..# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
conda init
conda activate spyder-env

# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  analysisDir="/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
  fslDir="/Users/xpsy1114/fsl"
fi

# 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
# exclude subject 21
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 29 30 31 32 33 34; do
for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34; do
    echo now creating RDMs for subject ${subjectTag}.
    # first delete the big file from the glm folder
    #for glm_version in 06 07 08 09; do
    #  rm ${scratchDir}/derivatives/sub-${subjectTag}/func/glm_${glm_version}_pt01.feat/filtered_func_data.nii.gz
    #  rm ${scratchDir}/derivatives/sub-${subjectTag}/func/glm_${glm_version}_pt02.feat/filtered_func_data.nii.gz
    #done
    # derivDir="${scratchDir}/derivatives/sub-${subjectTag}"
    #flirt -in ${derivDir}/anat/sub-${subjectTag}_T1w_biascorr_noCSF_brain_mask.nii.gz -ref ${derivDir}/func/preproc_clean_01.feat/example_func.nii.gz -applyxfm -init ${derivDir}/func/preproc_clean_01.feat/reg/highres2example_func.mat -out ${derivDir}/anat/sub-${subjectTag}_T1w_biascorr_noCSF_brain_mask_examplefunc_pt01.nii.gz
    #flirt -in ${derivDir}/anat/sub-${subjectTag}_T1w_biascorr_noCSF_brain_mask.nii.gz -ref ${derivDir}/func/preproc_clean_02.feat/example_func.nii.gz -applyxfm -init ${derivDir}/func/preproc_clean_02.feat/reg/highres2example_func.mat -out ${derivDir}/anat/sub-${subjectTag}_T1w_biascorr_noCSF_brain_mask_examplefunc_pt02.nii.gz
    # fsl_sub -q long.q python ${analysisDir}/multiple_clocks_repo/scripts/fMRI_do_RSA_debug_cluster.py ${subjectTag}
    fsl_sub -q veryshort.q python ${analysisDir}/multiple_clocks_repo/scripts/create_fmri_model_RDMs_between_halves.py ${subjectTag}
done