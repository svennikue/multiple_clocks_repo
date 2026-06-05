#!/bin/sh
# transforms beta-results per model and subject to standard space
# to prepare group stats.
# submit like bash transform_subject_res_to_standard.sh
# requires results from submit_RSA_fmri.sh

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
#fslDir="/opt/fmrib/fsl"
export fslDir=~/scratch/fsl
export PATH=$fslDir/share/fsl/bin/:$PATH
source $fslDir/etc/fslconf/fsl.sh
module load fsl 

# data_RDMs_state-only_masked_same_locinstate_26-11-2025_glmbase_all-rews-split_buttons
# data_RDMs_state_only-masked_same_locinstate-excl_rewA_26-11-2025_glmbase_all-rews-split_buttons
# RSA_state_only-masked_same_locinstate-excl_rewA_26-11-2025_glmbase_all-rews-split_buttons_smooth5


glm_version="all_paths-stickrews-split_buttons"
RSA_version="state_Aones_and_combo_10-12-2025"

# 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do

echo now starting transforming all results of glm $glm_version RSA $RSA_version to standard space
# 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35 ; do #without 21 AND 29!

for subjectTag in 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35 ; do #without 21 AND 29!
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35 ; do #without 21 AND 29! plus from 30 the date changes to 10th of nov...
    # for every result file
    # resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}_smooth5/results
    # stdDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}_smooth5/results-standard-space
    # for every result file
    resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}/results
    stdDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}/results-standard-space
    if [ -d $stdDir ]; then
        mv $stdDir ${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}/old-results-standard-space
        mkdir $stdDir
    fi
    if [ ! -d $stdDir ]; then
        echo making new directory to save standard files: $stdDir
        mkdir $stdDir
    fi
    preprocDir=${scratchDir}/derivatives/sub-${subjectTag}/func/preproc_clean_02.feat

    echo now for subject $subjectTag glm $glm_version RSA $RSA_version
    # in case something has gone wrong before
    find "$resultDir" -type f -name 'std-std-*.nii.gz' -exec rm {} +
    find "$resultDir" -type f -name 'std-*.nii.gz' -exec rm {} +
    #find "$resultDir" -type f -name '*_std.nii.gz' -exec rm {} +

    # Loop through each .nii.gz file in the directory
    for file in "$resultDir"/*.nii.gz; do
        # only do this if the file is newer than 3 days×24 hours * 60 minutes * 60 seconds =259,200 seconds
        # 7 days: 7*24*60*60 = 604800
        if [[ ! "$(( $(date +"%s") - $(stat -c "%Y" "$file") ))" -gt "604800" ]]; then
                # Extract the filename without the extension
            file_name=$(basename "$file" .nii.gz)
            echo $file_name 
            # Define the output filename
            output="${stdDir}/${file_name}_std.nii.gz"
            
            # skip if you already transformed this
            if [ -e "${output}" ]; then
                continue
            fi
            # Transform to standard
            echo input is $file and output is $output
            # new cluster
            fsl_sub -q short flirt -in "$file" -ref ${preprocDir}/reg/standard.nii.gz -applyxfm -init ${preprocDir}/reg/example_func2standard.mat -out "$output"
            # old cluster
            # fsl_sub -q short.q flirt -in "$file" -ref ${preprocDir}/reg/standard.nii.gz -applyxfm -init ${preprocDir}/reg/example_func2standard.mat -out "$output"
        fi
    done
done