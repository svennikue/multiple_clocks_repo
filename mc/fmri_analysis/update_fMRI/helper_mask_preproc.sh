
# this script creates a mask for the functional file which will be used to bet the functional
# during registration.
# this is now also included in feat_preproc.sh and can be run after you did pre_feat_preproc.sh




#  
for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34; do
    for task_half in 01 02; do
        derivDir="/home/fs0/xpsy1114/scratch/data/derivatives/sub-${subjectTag}"
        if [ ! -f ${derivDir}/func/preproc_clean_${task_half}/reg/highres2example_func.mat ]; then
            echo now registering subject $subjectTag , task half $task_half
        else 
            echo reg matrix for $subjectTag does not exist
        fi
        
        flirt -in ${derivDir}/anat/sub-${subjectTag}_T1w_biascorr_noCSF_brain_mask.nii.gz -ref ${derivDir}/func/preproc_clean_${task_half}.feat/example_func.nii.gz -applyxfm -init ${derivDir}/func/preproc_clean_${task_half}.feat/reg/highres2example_func.mat -out ${derivDir}/anat/sub-${subjectTag}_T1w_noCSF_brain_mask_prep_func_${task_half}.nii.gz
        fslmaths ${derivDir}/anat/sub-${subjectTag}_T1w_noCSF_brain_mask_prep_func_${task_half}.nii.gz -thr 0.05 -fillh -bin ${derivDir}/anat/sub-${subjectTag}_T1w_noCSF_brain_mask_bin_func_${task_half}.nii.gz
    done
done