#!/bin/bash
# Musicbox study 2023_017, Svenja, Dec 2023

# Script to free up space.
# checks if files have been copied, and if they have been, delete the ones in the raw dir.
# 

scratchDir="/home/fs0/xpsy1114/scratch/data"



# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34; do
#     echo "now looking at folder: $scratchDir/raw/${subjectTag}_scan"
#     rawDir=${scratchDir}/raw/${subjectTag}_scan
#     anatDir=${scratchDir}/pilot/sub-$subjectTag/anat
#     fmapDir=${scratchDir}/pilot/sub-$subjectTag/fmap
#     funcDir=$scratchDir/pilot/sub-$subjectTag/func
#     # T1w
#     if test -f $anatDir/sub-${subjectTag}_T1w.nii.gz; then
#         rm $rawDir/*_MPRAGE_UP.nii
#     fi
#     # rename to T1w_biascorr if this was still wrong
#     if test -f ${scratchDir}/derivatives/sub-${subjectTag}/anat/sub-${subjectTag}_T1_biascorr_noCSF_brain.nii.gz; then
#         mv ${scratchDir}/derivatives/sub-${subjectTag}/anat/sub-${subjectTag}_T1_biascorr_noCSF_brain.nii.gz ${scratchDir}/derivatives/sub-${subjectTag}/anat/sub-${subjectTag}_T1w_biascorr_noCSF_brain.nii.gz
#     fi
#     if test -f ${scratchDir}/derivatives/sub-${subjectTag}/anat/sub-${subjectTag}_T1_biascorr_noCSF_brain_mask.nii.gz; then
#         mv ${scratchDir}/derivatives/sub-${subjectTag}/anat/sub-${subjectTag}_T1_biascorr_noCSF_brain_mask.nii.gz ${scratchDir}/derivatives/sub-${subjectTag}/anat/sub-${subjectTag}_T1w_biascorr_noCSF_brain_mask.nii.gz
#     fi
#     # functionals
#     if test -f $funcDir/sub-${subjectTag}_1_vol_1_bold.nii.gz; then
#         rm $rawDir/images_04*.nii
#     fi
#     if test -f $funcDir/sub-${subjectTag}_1_bold.nii.gz; then
#         rm $rawDir/images_05*.nii
#     fi
#     if test -f $funcDir/sub-${subjectTag}_1_vol_2_bold.nii.gz; then
#         rm $rawDir/images_07*.nii
#     fi
#     if test -f $funcDir/sub-${subjectTag}_2_bold.nii.gz; then
#         rm $rawDir/images_08*.nii
#     fi
#     if test -f $funcDir/sub-${subjectTag}_1_bold_wb.nii.gz; then
#         rm $rawDir/images_09*.nii
#     fi
#     if test -f $funcDir/sub-${subjectTag}_2_bold_wb.nii.gz; then
#         rm $rawDir/images_010*.nii
#     fi 
#     # and fieldmap
#     if test -f $fmapDir/sub-${subjectTag}_magnitude1.nii.gz; then
#         rm $rawDir/images*field*e1.nii
#     fi
#     if test -f $fmapDir/sub-${subjectTag}_magnitude2.nii.gz; then
#         rm $rawDir/images*field*e2.nii
#     fi
#     if test -f $fmapDir/sub-${subjectTag}_phasediff.nii.gz; then
#         rm $rawDir/images*field*e2_ph.json
#     fi
# done


# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34; do
#     funcDir=$scratchDir/derivatives/sub-$subjectTag/func
#     echo "now looking at folder: $funcDir"
#     # rm ${funcDir}.glm_10_pt01.feat/filtered_func_data.nii.gz
#     rm ${funcDir}/RSA_04_glmbase_06/results/data_RDM.pkl
#     rm ${funcDir}/RSA_05_glmbase_06/results/data_RDM.pkl
#     rm ${funcDir}/RSA_06_glmbase_07/results/data_RDM.pkl
#     rm ${funcDir}/RSA_07_glmbase_07/results/data_RDM.pkl
#     rm ${funcDir}/RSA_09_glmbase_07/results/data_RDM.pkl
#     rm ${funcDir}/RSA_10_glmbase_08/results/data_RDM.pkl
#     rm ${funcDir}/RSA_11_glmbase_09/results/data_RDM.pkl
#     rm ${funcDir}/RSA_999_glmbase_07/results/data_RDM.pkl

#     find "$funcDir" -type d -name 'RSA_09_glmbase_08' -exec rm -r {} +
#     find "$funcDir" -type d -name 'glm_08_pt02.feat' -exec rm -r {} +
#     find "$funcDir" -type d -name 'glm_08_pt02.feat' -exec rm -r {} +
# done


for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34; do
    funcDir=$scratchDir/derivatives/sub-$subjectTag/func
    echo "now looking at folder: $funcDir"
    for task_half in 01 02; do
        rm ${funcDir}/glm_01_pt${task_half}.feat/filtered_func_data.nii.gz
        rm ${funcDir}/glm_02_pt${task_half}.feat/filtered_func_data.nii.gz
        rm ${funcDir}/glm_03-9999_pt${task_half}.feat/filtered_func_data.nii.gz
        rm ${funcDir}/glm_03-99_pt${task_half}.feat/filtered_func_data.nii.gz
        rm ${funcDir}/glm_03-999_pt${task_half}.feat/filtered_func_data.nii.gz
    done
done
