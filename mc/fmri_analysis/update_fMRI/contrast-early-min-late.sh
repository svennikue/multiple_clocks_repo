#!/bin/sh
# computes early-min late of beta maps in standard space per subject

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
#fslDir="/opt/fmrib/fsl"
export fslDir=~/scratch/fsl
export PATH=$fslDir/share/fsl/bin/:$PATH
source $fslDir/etc/fslconf/fsl.sh

glm_version="03-4"
glm_version_new="03-4-e-l"
RSA_version="02"

# 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do

echo now starting computing contrasts for glm $glm_version RSA $RSA_version 
# 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34; do #without 21
for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 ; do
    # for every result file
    echo now subject $subjectTag

    earlyDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}-e/results-standard-space
    lateDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}-l/results-standard-space
    newDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version_new}
    mkdir $newDir
    mkdir $newDir/results-standard-space
    echo made the new dir here: $newDir
    # Loop through each .nii.gz file in the directory
    for early_file_path in "$earlyDir"/*beta_std.nii.gz; do
        # Extract the filename without the extension
        file_name=$(basename "$early_file_path" .nii.gz)
        # Define the output filename
        late_file_path="${lateDir}/${file_name}.nii.gz"
        output="${newDir}/results-standard-space/${file_name}_contrast.nii.gz"
        # compute contrast
        fslmaths "$early_file_path" -sub "$late_file_path" "$output"
    done
done



