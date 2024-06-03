#!/bin/sh
# computes a linear trend contrast across repeats of beta maps in standard space per subject

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
#fslDir="/opt/fmrib/fsl"
export fslDir=~/scratch/fsl
export PATH=$fslDir/share/fsl/bin/:$PATH
source $fslDir/etc/fslconf/fsl.sh

glm_version="03-4-rep"
glm_version_new="03-4-across-reps"
RSA_version="03-1"

# 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do

echo now starting computing contrasts for glm $glm_version RSA $RSA_version 
# 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 
# for subjectTag in 01 02  03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do #without 21
# next 
for subjectTag in 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
    # for every result file
    echo now subject $subjectTag
    rep1Dir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}1/results-standard-space
    rep2Dir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}2/results-standard-space
    rep3Dir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}3/results-standard-space
    rep4Dir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}4/results-standard-space
    rep5Dir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}5/results-standard-space

    newDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version_new}

    mkdir $newDir
    mkdir $newDir/results-standard-space
    mkdir $newDir/first_step
    echo created a new dir for subj $subjectTag here: $newDir

    # Loop through each .nii.gz file in the directory
    for rep1_file_path in "$rep1Dir"/*beta_std.nii.gz; do
        # Extract the filename without the extension
        file_name=$(basename "$rep1_file_path" .nii.gz)
        rep2_file_path="${rep2Dir}/${file_name}.nii.gz"     
        rep3_file_path="${rep3Dir}/${file_name}.nii.gz"
        rep4_file_path="${rep4Dir}/${file_name}.nii.gz"
        rep5_file_path="${rep5Dir}/${file_name}.nii.gz"
        

        # compute contrast
        # first, weight each repeat
        fslmaths "$rep1_file_path" -mul -2 ${newDir}/first_step/${file_name}_weighted_rep1_file.nii.gz
        fslmaths "$rep2_file_path" -mul -1 ${newDir}/first_step/${file_name}_weighted_rep2_file.nii.gz
        fslmaths "$rep3_file_path" -mul 0 ${newDir}/first_step/${file_name}_weighted_rep3_file.nii.gz
        fslmaths "$rep4_file_path" -mul 1 ${newDir}/first_step/${file_name}_weighted_rep4_file.nii.gz
        fslmaths "$rep5_file_path" -mul 2 ${newDir}/first_step/${file_name}_weighted_rep5_file.nii.gz

        # then sum: rep1 * -2 + rep2 * -1 + rep3 * 0 + rep4 * 1 + rep5 * 2
        output="${newDir}/results-standard-space/${file_name}_contrast.nii.gz"
        echo will be $output
        fslmaths ${newDir}/first_step/${file_name}_weighted_rep1_file.nii.gz -add ${newDir}/first_step/${file_name}_weighted_rep2_file.nii.gz -add ${newDir}/first_step/${file_name}_weighted_rep3_file.nii.gz -add ${newDir}/first_step/${file_name}_weighted_rep4_file.nii.gz -add ${newDir}/first_step/${file_name}_weighted_rep5_file.nii.gz $output
    done
done