##!/bin/sh
# Produce nuisance regressors to get rid of motion artefacts: motion outliers and CSF time series
# submit like fsl_sub -q short.q bash NuisanceRegs_ptTwo.sh 05
# first type module load MATLAB

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
#fslDir="/opt/fmrib/fsl"
fslDir="/home/fs0/xpsy1114/scratch/fsl"

module load MATLAB

# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  analysisDir="/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
  fslDir="/Users/xpsy1114/fsl"
fi
echo Scratch directory is $scratchDir

subjects=$1
# I changed the way of submitting this; double check if this is still correct!
#for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34; do
for subjectTag in $subjects; do
    # Command line argument 1/1: subject tag
    echo Subject tag for this subject: $subjectTag 
    # Construct directory for derived data
    derivDir=$scratchDir/derivatives/sub-$subjectTag
    # Set directory containing previously run preprocessing
    anatDir=$derivDir/anat/T1w_BiasCorr.anat

    # create a loop from here to the end (marked as well)
    for task_half in 01 02; do
      # for task_half in 02; do
        preprocDir=$derivDir/func/preproc_clean_${task_half}.feat
        # Construct directory for nuisance regressors
        nuisanceDir=$derivDir/motion/nuisance_${task_half}
        if [ -d "$nuisanceDir" ]; then
          echo this is the nuisance dir $nuisanceDir
        else
          echo Careful! First run pt1 with bash NuisanceRegs_ptONE_motion.sh
        fi

        # Make folder for segmentationß
        mkdir $nuisanceDir/segmentation

        flirt -in $anatDir/T1_fast_pve_1.nii.gz -ref $preprocDir/example_func.nii.gz -applyxfm -init $preprocDir/reg/highres2example_func.mat -out $anatDir/grey_matter_func_${task_half}.nii.gz 
        fslmaths $anatDir/grey_matter_func_${task_half}.nii.gz -thr 0.05 -fillh -bin $derivDir/anat/grey_matter_mask_func_${task_half}.nii.gz

        # Transform CSF mask in structural space to functional space
        # flirt -in $nuisanceDir/segmentation/segm_pve_0.nii.gz -ref $preprocDir/example_func.nii.gz -applyxfm -init $preprocDir/reg/highres2example_func.mat -out $nuisanceDir/segmentation/CSF_func.nii.gz
        # using different file instead.
        flirt -in $anatDir/T1_fast_pve_0.nii.gz -ref $preprocDir/example_func.nii.gz -applyxfm -init $preprocDir/reg/highres2example_func.mat -out $nuisanceDir/segmentation/CSF_func.nii

        # Transform bounding box in standard space to functional space
        flirt -in $scratchDir/masks/vent_mask.nii.gz -ref $preprocDir/example_func.nii.gz -applyxfm -init $preprocDir/reg/standard2example_func.mat -out $nuisanceDir/segmentation/CSF_bb.nii.gz

        # Do thresholding and erosion to get mask that only contains CSF
        # this one would be more careful but doesn't really include any voxels. for now, use this:
        # fslmaths CSF_func.nii.gz -mul CSF_bb.nii.gz -thr 0.8 -kernel sphere 2 -ero mul_CSF_func_bb.nii.gz
        # fslmaths $nuisanceDir/segmentation/CSF_func.nii.gz -mul $nuisanceDir/segmentation/CSF_bb.nii.gz -thr 0.9 -kernel sphere 2 -ero $nuisanceDir/segmentation/CSF_mask.nii.gz
        fslmaths $nuisanceDir/segmentation/CSF_func.nii.gz -mul $nuisanceDir/segmentation/CSF_bb.nii.gz -thr 0.8 -kernel sphere 2 -ero $nuisanceDir/segmentation/CSF_mask.nii.gz

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
