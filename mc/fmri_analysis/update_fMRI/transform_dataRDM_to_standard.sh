#!/bin/sh
# transforms data RDMs per subject to standard space
# to prepare a group average map.
# submit like bash transform_dataRDM_to_standard.sh
# requires results from submit_RSA_fmri.sh


# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
#fslDir="/opt/fmrib/fsl"
export fslDir=~/scratch/fsl
export PATH=$fslDir/share/fsl/bin/:$PATH
source $fslDir/etc/fslconf/fsl.sh
module load fsl 

glm_version="03-4"

# 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do

echo now starting to transform all data RDM maps of glm $glm_version to standard space
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35 ; do #without 21 AND 29!
# later: 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35
for subjectTag in 35 ; do
    # for every result file
    dataRDMdir=${scratchDir}/derivatives/sub-${subjectTag}/func/data_RDM_glmbase_${glm_version}
    preprocDir=${scratchDir}/derivatives/sub-${subjectTag}/func/preproc_clean_02.feat

    echo now for subject $subjectTag glm $glm_version
    # in case something has gone wrong before
    find "$dataRDMdir" -type f -name 'std-std-*.nii.gz' -exec rm {} +

    # Loop through each .nii.gz file in the directory
    for file in "$dataRDMdir"/*.nii.gz; do
        # only do this if the file is newer than 3 days×24 hours * 60 minutes * 60 seconds =259,200 seconds
        # 7 days: 7*24*60*60 = 604800
        if [[ ! "$(( $(date +"%s") - $(stat -c "%Y" "$file") ))" -gt "604800" ]]; then
            # Extract the filename without the extension
            file_name=$(basename "$file" .nii.gz)
            echo $file_name 
            # Define the output filename
            output="${dataRDMdir}/${file_name}_std.nii.gz"
            
            # skip if you already transformed this
            if [ -e "${output}" ]; then
                continue
            fi
            # Transform to standard
            echo input is $file and output is $output
            # new cluster
            # fsl_sub -q short flirt -in "$file" -ref ${preprocDir}/reg/standard.nii.gz -applyxfm -init ${preprocDir}/reg/example_func2standard.mat -out "$output"
            # old cluster
            fsl_sub -q short.q flirt -in "$file" -ref ${preprocDir}/reg/standard.nii.gz -applyxfm -init ${preprocDir}/reg/example_func2standard.mat -out "$output"
        fi
    done
done