#!/bin/sh
# Run PALM for group stats
# Svenja Kuchenhoff 2024
# run like bash Group_PALM_folder.sh
# important: run module load PALM first!

# >>> just drop your folder here <<<
groupDir="/home/fs0/xpsy1114/scratch/data/derivatives/group/group_state_univ_rewards_glmbase_all_paths-stickrews-split_buttons"

module load PALM
module load fsl

# needs to be unzipped files!
if [ ! -d "$groupDir" ]; then
    echo "Group Directory does not exist."
    exit 1
else
    echo Folder with concatenated files for permutation testing: $groupDir
fi

# Set parameters for permutation test
# Both mask and input files should be unzipped (.nii)
clusterThreshold=3.1
permutationNumber=1000 #should be something like 1000 or 5000 later..
# maskFile=${scratchDir}/masks/MNI152_T1_2mm_brain_mask.nii.gz

# Construct the folder for permutation testing for this analysis
permDir=${groupDir}_palm
if [ ! -d "$permDir" ]; then
    mkdir ${permDir}
fi

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
            echo saving current file in $outPath

            fsl_sub -q short palm -i ${curr_file} -T -C $clusterThreshold -Cstat mass -n $permutationNumber -o $outPath -ise -save1-p
            echo "Processed: $curr_file"
        fi
    fi
done
