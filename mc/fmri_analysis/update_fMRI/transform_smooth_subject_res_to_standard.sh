#!/bin/sh
# transforms beta-results per model and subject to standard space
# to prepare group stats.
# submit like bash transform_smooth_subject_res_to_standard.sh
# make sure to be in some dir that you don't need (it creates MANY logs!)

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
#fslDir="/opt/fmrib/fsl"
export fslDir=~/scratch/fsl
export PATH=$fslDir/share/fsl/bin/:$PATH
source $fslDir/etc/fslconf/fsl.sh
module load fsl 

glm_version="all-paths-fixed_stickrews_split-buttons"
RSA_version="quarters_DSR_controls"

echo now starting transforming all results of glm $glm_version RSA $RSA_version to standard space


for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35 ; do #without 21 AND 29! 
# for subjectTag in 17 19 20 22 23 32 33 34 ; do
#for subjectTag in 34; do #without 21 AND 29! 
    resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_*_glmbase_${glm_version}/smoothed
    candidates=( ${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_*_glmbase_${glm_version})
    if ((${#candidates[@]})); then
        # If multiple dates exist, pick the newest by mtime
        IFS=$'\n' candidates=($(ls -1dt "${candidates[@]}"))
        RSADir="${candidates[0]}"
        # RSADir="${candidates[-1]}"
    else
        RSADir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}
    fi

    # RSADir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_DSR_hamming_path-rew-sep_combos_11-02-2026_glmbase_${glm_version}
    resultDir=$RSADir/smoothed
    stdDir=$RSADir/standard-space-smooth

    if [ -d $stdDir ]; then
        mv $stdDir ${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_*_glmbase_${glm_version}/old-results-standard-space-smooth
        mkdir $stdDir
    fi
    if [ ! -d $stdDir ]; then
        echo making new directory to save standard files: $stdDir
        mkdir $stdDir
    fi

    echo set output directory as $stdDir

    preprocDir=${scratchDir}/derivatives/sub-${subjectTag}/func/preproc_clean_02.feat
    warpfile=${preprocDir}/reg/example_func2standard_warp.nii.gz

    echo now non-linear warping of subject $subjectTag glm $glm_version RSA $RSA_version

    # Loop through each .nii.gz file in the directory
    for file in "$resultDir"/*.nii.gz; do
        # only do this if the file is newer than 3 days×24 hours * 60 minutes * 60 seconds =259,200 seconds
        # 7 days: 7*24*60*60 = 604800
        # only do this if the file is newer than 7 days (7*24*60*60 = 604800)
        if [[ ! "$(( $(date +"%s") - $(stat -c "%Y" "$file") ))" -gt "604800" ]]; then

            file_name=$(basename "$file" .nii.gz)
            output="${stdDir}/${file_name}_std.nii.gz"
            
            # skip if already transformed (do not submit)
            if [ -e "$output" ]; then
                echo "Skipping $(basename "$file"): output already exists -> $(basename "$output")"
                continue
            fi

            echo input is "$file" and output is "$output"

            if [ -f "$warpfile" ]; then
                # Nonlinear transform: example_func → standard via warp
                fsl_sub -q short applywarp \
                    -i "$file" \
                    -r "${preprocDir}/reg/standard.nii.gz" \
                    -w "$warpfile" \
                    -o "$output" \
                    --interp=spline
            else
                # Fallback: linear only (same as your original flirt call)
                echo "WARNING: no warp file for sub-${subjectTag}, falling back to FLIRT"
                fsl_sub -q short flirt \
                    -in "$file" \
                    -ref "${preprocDir}/reg/standard.nii.gz" \
                    -applyxfm \
                    -init "${preprocDir}/reg/example_func2standard.mat" \
                    -out "$output"
            fi
        fi
    done
done