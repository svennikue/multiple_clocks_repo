#!/bin/sh
# Produce nuisance regressors to get rid of motion artefacts: motion outliers and CSF time series

subjects="01"
# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
toolboxDir="/home/fs0/xpsy1114/scratch/analysis"
homeDir="/home/fs0/xpsy1114"

task_halves=("01" "02")
# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  toolboxDir="/Users/xpsy1114/Documents/toolboxes"
  homeDir="/Users/xpsy1114/Documents/projects/multiple_clocks"
fi

echo Scratch directory is $scratchDir
echo Home directory is $homeDir

for subjectTag in "${subjects[@]}"; do
    # Command line argument 1/1: subject tag
    echo Subject tag for this subject: $subjectTag
    
    # Construct directory for derived data
    derivDir=$scratchDir/derivatives/sub-$subjectTag
    # Set directory containing previously run preprocessing

    # do this twice, once for pt1 and once for pt2
    # this has to be 01 and 02. Adjust in the future!

    # create a loop from here to the end (marked as well)
    for task_half in "${task_halves[@]}"; do
        preprocDir=$derivDir/func/preproc_clean_${task_half}.feat
        # Construct directory for nuisance regressors
        nuisanceDir=$derivDir/func/nuisance_${task_half}
        # Remove if already exists
        rm -rf $nuisanceDir
        # And create nuisance regressor directory
        mkdir -p $nuisanceDir

        echo this is the nuisance dir $nuisanceDir

        # Generate matrix of motion outlier regressors; don't rerun motion correction
        fsl_motion_outliers -i $preprocDir/filtered_func_data.nii.gz -o $nuisanceDir/motionOutliers.txt -p $nuisanceDir/motionOutliers.png --nomoco

        # Make folder for segmentationß
        mkdir $nuisanceDir/segmentation

        # Do segmentation of structural file with FAST
        fast --channels=1 --type=1 --class=3 --out=$nuisanceDir/segmentation/segm $derivDir/anat/sub-${subjectTag}_T1W_brain.nii.gz

        # Transform CSF mask in structural space to functional space
        flirt -in $nuisanceDir/segmentation/segm_pve_0.nii.gz -ref $preprocDir/example_func.nii.gz -applyxfm -init $preprocDir/reg/highres2example_func.mat -out $nuisanceDir/segmentation/CSF_func.nii.gz

        # Transform bounding box in standard space to functional space
        flirt -in $scratchDir/masks/vent_mask.nii.gz -ref $preprocDir/example_func.nii.gz -applyxfm -init $preprocDir/reg/standard2example_func.mat -out $nuisanceDir/segmentation/CSF_bb.nii.gz

        # Do thresholding and erosion to get mask that only contains CSF
        fslmaths $nuisanceDir/segmentation/CSF_func.nii.gz -mul $nuisanceDir/segmentation/CSF_bb.nii.gz -thr 0.9 -kernel sphere 2 -ero $nuisanceDir/segmentation/CSF_mask.nii.gz

        # Extract values from functional in CSF
        fslmeants -i $preprocDir/filtered_func_data.nii.gz -o $nuisanceDir/CSFsignal.txt -m $nuisanceDir/segmentation/CSF_mask.nii.gz

        # Matlab code to combine nuisance regressors, and include motion regressor derivatives
        matlabCode="motion = load(fullfile('${preprocDir}','mc','prefiltered_func_data_mcf.par'));
        motionDerivs = [zeros(1,size(motion,2)); diff(motion)];
        motionOutliers = load(fullfile('${nuisanceDir}','motionOutliers.txt'));
        CSF = load(fullfile('${nuisanceDir}','CSFsignal.txt'));
        allNuisanceRegressors = [motion motionDerivs motionOutliers CSF];
        dlmwrite(fullfile('${nuisanceDir}','combined.txt'), allNuisanceRegressors, 'delimiter','\t');"

        # Write matlab code to temporary .m file
        echo $matlabCode > $nuisanceDir/combine.m

        # Run matlab code
        matlab -nodisplay -nosplash \< $nuisanceDir/combine.m
    done
done