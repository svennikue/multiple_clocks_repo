#!/bin/sh
# Run PALM for group stats
# Svenja Kuchenhoff 2024
# run like bash Group_PALM.sh
# important: run module load palm first!

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"

fslDir="/opt/fmrib/fsl"
glm_version="03-4"
RSA_version="03-1"
palmno="03"

module load palm


# needs to be unzipped files!
groupDir=${scratchDir}/derivatives/group/group_RSA_${RSA_version}_glmbase_${glm_version}
# Check if the directory exists
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
permDir=$scratchDir/derivatives/group/RSA_${RSA_version}_glmbase_${glm_version}_palm_${palmno}
if [ ! -d "$permDir" ]; then
    mkdir ${permDir}
fi

# Loop through all files in the RSA group results directory
for curr_file in "$groupDir"/*; do
    # Check if it's a regular file
    if [ -f "$curr_file" ]; then
        # Set path for output file
        old_file_name=$(basename "${curr_file}")
        # remove extension
        file_name="${old_file_name%.*}"
        outPath=$permDir/${file_name}
        echo saving current file in $outPath

        fsl_sub -q long.q palm -i ${curr_file} -T -C $clusterThreshold -Cstat mass -n $permutationNumber -o $outPath -ise -save1-p
        echo "Processed: $curr_file"
    fi
done
