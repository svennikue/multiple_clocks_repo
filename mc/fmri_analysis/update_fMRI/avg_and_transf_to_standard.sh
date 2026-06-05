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

glm_v_one="03-4"
glm_v_two="04-4"
RSA_version="02-A"

echo now starting averaging all results of RSA $RSA_version of glm $glm_v_one and  glm $glm_v_two

for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34; do
    resultDir_one=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_v_one}/results
    resultDir_two=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_v_two}/results

    list_of_beta_files=$(find "$resultDir_one/" -name "*beta.nii.gz" -type f)

    resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_avg_of_glm_${glm_v_one}_glm_${glm_v_two}
    if [ ! -d $resultDir ]; then
        mkdir $resultDir
        echo making new directory to save averaged: $resultDir
    fi
    for file in $list_of_beta_files ; do
        # Extract the filename
        filename=$(basename "$file")
        echo now averaging $filename for subject $subjectTag
        fslmerge -t $resultDir/merged_${filename} $resultDir_one/${filename} $resultDir_two/${filename}
        fslmaths $resultDir/merged_${filename} -Tmean $resultDir/avg_${filename}
    done

    echo now starting to transform all results of averaged RSA $RSA_version of glm $glm_v_one and glm $glm_v_two for $subjectTag

    stdDir=${resultDir}/results-standard-space
    if [ -d $stdDir ]; then
        rm ${stdDir}/*_std.nii.gz
    fi
    if [ ! -d $stdDir ]; then
        echo making new directory to save standard files: $stdDir
        mkdir $stdDir
    fi
    preprocDir=${scratchDir}/derivatives/sub-${subjectTag}/func/preproc_clean_01.feat

    # Loop through each .nii.gz file in the directory
    for file in "$resultDir"/avg*.nii.gz; do
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