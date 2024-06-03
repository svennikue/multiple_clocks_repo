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

glm_version="03-4"
RSA_version="03-1"

# 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do

echo now starting transforming all results of glm $glm_version RSA $RSA_version to standard space
# 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34; do #without 21
for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34; do
    # for every result file
    resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}/results
    stdDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}/results-standard-space
    if [ -d $stdDir ]; then
        rm ${stdDir}/*_std.nii.gz
    fi
    if [ ! -d $stdDir ]; then
        echo making new directory to save standard files: $stdDir
        mkdir $stdDir
    fi
    preprocDir=${scratchDir}/derivatives/sub-${subjectTag}/func/preproc_clean_01.feat

    echo now for subject $subjectTag glm $glm_version RSA $RSA_version
    # in case something has gone wrong before
    find "$resultDir" -type f -name 'std-std-*.nii.gz' -exec rm {} +
    find "$resultDir" -type f -name 'std-*.nii.gz' -exec rm {} +
    find "$resultDir" -type f -name '*_std.nii.gz' -exec rm {} +

    # Loop through each .nii.gz file in the directory
    for file in "$resultDir"/*.nii.gz; do
        # Extract the filename without the extension
        file_name=$(basename "$file" .nii.gz)
        # skip if you already transformed this
        if [[ $file_name == std* ]]; then
            continue 
        fi
        # Define the output filename
        output="${stdDir}/${file_name}_std.nii.gz"
        # Transform to standard
        fsl_sub -q short.q flirt -in "$file" -ref ${preprocDir}/reg/standard.nii.gz -applyxfm -init ${preprocDir}/reg/example_func2standard.mat -out "$output"
    
    done
done



