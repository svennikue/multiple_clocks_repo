#!/usr/bin/env bash

scratchDir="/home/fs0/xpsy1114/scratch/data"
glm_version="all_paths-stickrews-split_buttons"
result_version="state_univ_rewards"

fslDir="/opt/fmrib/fsl"
export FSLDIR="$fslDir"
export PATH=$FSLDIR/bin:$PATH
source $FSLDIR/etc/fslconf/fsl.sh
module load fsl 

# group output dir
groupDir=${scratchDir}/derivatives/group/group_${result_version}_glmbase_${glm_version}
echo "this is group dir $groupDir"
mkdir -p "$groupDir"

# list of subjects (no 21, 29)
subjects="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35"

# --------- STATE A EXAMPLE (repeat for B/C/D if you like) ----------
state="A"
outFile="${groupDir}/group_2state_${state}_smoothed_univ_glmbase_${glm_version}_std.nii.gz"

states="A B C D"


for state in $states; do
    outFile="${groupDir}/group_state_${state}_smoothed_univ_glmbase_${glm_version}_std.nii.gz"
    first=1

    for subjectTag in $subjects; do
        resultDir="${scratchDir}/derivatives/sub-${subjectTag}/func/${result_version}_glmbase_${glm_version}/results-standard-space"
        [[ -d "$resultDir" ]] || { echo "skip sub-${subjectTag} (no results dir)"; continue; }

        file=$(ls "${resultDir}"/*state${state}*_std.nii.gz 2>/dev/null | head -n 1)
        [[ -f "$file" ]] || { echo "skip sub-${subjectTag} (no *state_${state}*.nii.gz)"; continue; }

        filename=$(basename "$file")
        masked="${resultDir}/masked_${filename}"

        if [[ ! -f "$masked" ]]; then
            echo "masking ${filename} for sub-${subjectTag}, state_${state}"
            fslmaths "$file" -mas "${scratchDir}/masks/brain_bin.nii.gz" "$masked"
        fi

        if [[ $first -eq 1 ]]; then
            cp "$masked" "$outFile"
            first=0
        else
            fslmerge -t "$outFile" "$outFile" "$masked"
        fi
    done

    gunzip -f "${outFile}.gz" 2>/dev/null || true
    echo "Done, group file for state_${state}: $outFile"
done

gunzip -f "${outFile}.gz" 2>/dev/null || true
echo "Done, group file for state_${state}: $outFile"
