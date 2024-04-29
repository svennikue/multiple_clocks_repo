#!/bin/sh
# Produce nuisance regressors to get rid of motion artefacts: motion outliers and CSF time series

# Command line argument 1/1: subject tag
subjects=$1
echo Subject tag for this subject: $subjectTag



# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
fi

# Show what ended up being the scratch directory
echo Scratch directory is $scratchDir

for subjectTag in "${subjects[@]}"; do
    # Make subject folder in derivatives folder: folder where any non-raw data gets stored (-p: don't if already exists)
    # Construct directory for derived data
    derivDir=$scratchDir/derivatives/sub-$subjectTag
    echo "now looking at folder: $derivDir"

    # do this twice for both task halves.
    task_halves="01 02"

    for taskHalf in "${task_halves[@]}"; do
        # Set directory containing previously run preprocessing
        preprocDir=$derivDir/func/preproc_clean_${taskHalf}.feat

        # Construct directory for nuisance regressors
        nuisanceDir=$derivDir/func/nuisance_$taskHalf
        # Remove if already exists
        rm -rf $nuisanceDir
        # And create nuisance regressor directory
        mkdir -p $nuisanceDir

        echo "now looking at this file: $preprocDir/filtered_func_data.nii.gz "
        # Generate matrix of motion outlier regressors; don't rerun motion correction
        fsl_motion_outliers -i $preprocDir/filtered_func_data.nii.gz -o $nuisanceDir/motionOutliers.txt -p $nuisanceDir/motionOutliers.png --nomoco

        # Make folder for segmentation
        mkdir $nuisanceDir/segmentation

        # Do segmentation of structural file with FAST
        fast --channels=1 --type=1 --class=3 --out=$nuisanceDir/segmentation/segm $derivDir/anat/sub-${subjectTag}_T1W_brain.nii.gz

        # Transform CSF mask in structural space to functional space
        flirt -in $nuisanceDir/segmentation/segm_pve_0.nii.gz -ref $preprocDir/example_func.nii.gz -applyxfm -init $preprocDir/reg/highres2example_func.mat -out $nuisanceDir/segmentation/CSF_func.nii.gz

        # Transform bounding box in standard space to functional space
        flirt -in $homeDir/Analysis/Masks/vent_mask.nii.gz -ref $preprocDir/example_func.nii.gz -applyxfm -init $preprocDir/reg/standard2example_func.mat -out $nuisanceDir/segmentation/CSF_bb.nii.gz

        # Do thresholding and erosion to get mask that only contains CSF
        fslmaths $nuisanceDir/segmentation/CSF_func.nii.gz -mul $nuisanceDir/segmentation/CSF_bb.nii.gz -thr 0.9 -kernel sphere 2 -ero $nuisanceDir/segmentation/CSF_mask.nii.gz

        # Extract values from functional in CSF
        fslmeants -i $preprocDir/filtered_func_data.nii.gz -o $nuisanceDir/CSFsignal.txt -m $nuisanceDir/segmentation/CSF_mask.nii.gz
    done
done

