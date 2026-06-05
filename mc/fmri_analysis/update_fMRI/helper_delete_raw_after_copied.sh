#!/bin/bash
# Musicbox study 2023_017, Svenja, Dec 2023

# Script to free up space.
# checks if files have been copied, and if they have been, delete the ones in the raw dir.
# 

scratchDir="/home/fs0/xpsy1114/scratch/data"
echo scratch dir is $scratchDir


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

# # removing residual files and filtered func data
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
#     funcDir=$scratchDir/derivatives/sub-$subjectTag/func
#     echo "now looking at folder: $funcDir"

#     for curr_folder in "$funcDir"/glm_*; do    
#         if [ -d "$curr_folder" ]; then
#             # Set path for output file
#             curr_path=$(basename "${curr_folder}")
#             echo removing $funcDir/$curr_path/stats/res4d.nii.gz
#             rm $funcDir/$curr_path/stats/res4d.nii.gz
#             echo removing $funcDir/$curr_path/filtered_func_data.nii.gz
#             rm $funcDir/$curr_path/filtered_func_data.nii.gz
#         fi
#         # if [[ -d "$curr_folder" && "$(basename "$curr_folder")" == glm*+.feat ]]; then
#         #     echo "Deleting directory: $curr_folder"
#         #     rm -rf "$curr_folder"
#         # fi
#     done
# done

for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
    funcDir="$scratchDir/derivatives/sub-$subjectTag/func"
    echo "now looking at folder: $funcDir"

    curr_folder="$funcDir/data_RDMs_state-only_excluding_rewA_tasksBD_13-11-2025_glmbase_all-rews-split_buttons"

    if [ -d "$curr_folder" ]; then
        echo "removing: $curr_folder"
        rm -rf -- "$curr_folder"
    else
        echo "not found (skipping): $curr_folder"
    fi
done

# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
#     funcDir=${scratchDir}/derivatives/sub-${subjectTag}/func
#     echo now looking at subject: ${subjectTag}
#     for half in 01 02; do
#         curr_folder=${funcDir}/glm_01-TR11_pt${half}.feat
#         if [ -f ${curr_folder}/stats/pe1.nii.gz ]; then
#             echo yes it worked - removing second run ${funcDir}/glm_01-TR11_pt${half}+.feat
#             # rm -rf ${funcDir}/glm_01-TR11_pt${half}+.feat
#         fi
#         if [ ! -f ${curr_folder}/stats/pe1.nii.gz ]; then
#             if [ -f ${funcDir}/glm_01-TR11_pt${half}+.feat/stats/pe1.nii.gz ]; then
#                 echo it is save to remove folder one
#                 echo run one didnt work - now removing $curr_folder and renaming ${funcDir}/glm_01-TR11_pt${half}+.feat
#                 #rm -rf $curr_folder
#                 # mv ${funcDir}/glm_01-TR11_pt${half}+.feat $curr_folder
#             fi
#         fi
#     done
# done


# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
#     funcDir=$scratchDir/derivatives/sub-$subjectTag/func/glm_03-rep
#     echo "now looking at folder: $funcDir"
#     for task_half in 01 02; do
#         for rep in 1 2 3 4 5; do 
#             curr_folder=${funcDir}${rep}_pt${task_half}.feat
#             if [ -d "$curr_folder" ]; then
#                 # Set path for output file
#                 curr_path=$(basename "${curr_folder}")
#                 echo removing $curr_folder
#                 rm -rf $curr_folder
#             fi
#         done
#     done
# done


# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34; do
#     behavDir=$scratchDir/derivatives/sub-$subjectTag/beh
#     echo "now looking at folder: $behavDir"
#     for curr_folder in "$behavDir"/*; do    
#         if [ -d "$curr_folder" ]; then
#             # Set path for output file
#             curr_path=$(basename "${curr_folder}")
#             rm $behavDir/$curr_path/searchlight_centers.pkl
#             rm $behavDir/$curr_path/searchlight_neighbors.pkl
#         fi
#     done
# done