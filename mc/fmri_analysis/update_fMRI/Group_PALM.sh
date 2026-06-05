#!/bin/sh
# Run PALM for group stats
# Svenja Kuchenhoff 2024
# run like bash Group_PALM.sh
# important: run module load PALM first!

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"

fslDir="/opt/fmrib/fsl"

glm_version="all-paths-fixed_stickrews_split-buttons"
RSA_version="quarters_DSR_controls"

palmno="p0_01"

module load PALM
module load fsl


# needs to be unzipped files!
groupDir=${scratchDir}/derivatives/group/contrast_RSA_${RSA_version}_glmbase_${glm_version}
# groupDir=${scratchDir}/derivatives/group/group_RSA_${RSA_version}_glmbase_${glm_version}_cropped
# groupDir=${groupDir}_cropped
# groupDir=${scratchDir}/derivatives/group/contrast_RSA_DSR_hamming_split_comps_combos_glmbase_all-paths-fixed_stickrews_split-buttons
# permDir=${scratchDir}/derivatives/group/contrast_RSA_${RSA_version}_glmbase_${glm_version}_smooth5_palm_${palmno}

# Check if the directory exists
if [ ! -d "$groupDir" ]; then
    echo "Group Directory does not exist."
    exit 1
else
    echo Folder with concatenated files for permutation testing: $groupDir
fi

gunzip $( ls ${groupDir}/*.nii.gz )

# Set parameters for permutation test
# Both mask and input files should be unzipped (.nii)
clusterThreshold=2.45
permutationNumber=1000 #should be something like 1000 or 5000 later..
# maskFile=${scratchDir}/masks/MNI152_T1_2mm_brain_mask.nii.gz


# Construct the folder for permutation testing for this analysis
# permDir=$scratchDir/derivatives/group/RSA_${RSA_version}_glmbase_${glm_version}_smooth5_palm_${palmno}
permDir=$scratchDir/derivatives/group/contrast_RSA_${RSA_version}_glmbase_${glm_version}_smooth5_palm_${palmno}
if [ ! -d "$permDir" ]; then
    mkdir ${permDir}
fi

# mask="${groupDir}/mask_all_33_subjects.nii"

# Loop through all files in the RSA group results directory
for curr_file in "$groupDir"/*; do
    # if curr_file is more new than 3 days: 3 days×24 hours * 60 minutes * 60 seconds =259,200 seconds
    if [[ ! "$(( $(date +"%s") - $(stat -c "%Y" "$curr_file") ))" -gt "259200" ]]; then
        # Check if it's a regular file
        if [ -f "$curr_file" ]; then
            # Set path for output file
            old_file_name=$(basename "${curr_file}")
            # remove extension
            file_name="${old_file_name%.*}"
            outPath=$permDir/${file_name}
            # skip if already computes (do not submit)
            if [ -e "$outPath" ]; then
                echo "Skipping $file_name: output already exists -> $outPath"
                continue
            fi

            echo saving current file in $outPath
            fsl_sub -q short palm -i ${curr_file} -T -C $clusterThreshold -Cstat mass -n $permutationNumber -o $outPath -ise -save1-p
            # fsl_sub -q short palm -i ${curr_file} -m ${mask} -T -C $clusterThreshold -Cstat mass -n $permutationNumber -o $outPath -ise -save1-p
            echo "Processed: $curr_file"
        fi
    fi
done
