#!/bin/sh
# Produce nuisance regressors to get rid of motion artefacts: motion outliers and CSF time series

# Command line argument 1/1: subject tag
subjectTag=$1
echo Subject tag for this subject: $subjectTag

# Set scratch directory for execution on server
scratchDir=/home/fs0/jacobb/scratch
# If this is not called on the server, but on a laptop connected to the server:
if [ ! -d $scratchDir ]; then
  scratchDir=/Volumes/Scratch_jacobb
fi
# If this is not called on a laptop, but on a mac connected to the server:
if [ ! -d $scratchDir ]; then
  scratchDir=/Users/jacobb/Documents/ServerHome/scratch
fi
# Show what ended up being the scratch directory
echo Scratch directory is $scratchDir

# Set analysis directory for execution on server
homeDir=/home/fs0/jacobb
# If this is not called on the server, but on a laptop connected to the server:
if [ ! -d $homeDir ]; then
  homeDir=/Volumes/jacobb
fi
# If this is not called on a laptop, but on a mac connected to the server:
if [ ! -d $homeDir ]; then
  homeDir=/Users/jacobb/Documents/ServerHome
fi
# Show what ended up being the home directory
echo Home directory is $homeDir

# Construct directory for raw data
rawDir=$scratchDir/sub-$subjectTag
# Construct directory for derived data
derivDir=$scratchDir/derivatives/sub-$subjectTag
# Set directory containing previously run preprocessing
preprocDir=$derivDir/func/preproc_nosmooth.feat
# Construct directory for nuisance regressors
nuisanceDir=$derivDir/func/nuisance
# Remove if already exists
rm -rf $nuisanceDir
# And create nuisance regressor directory
mkdir -p $nuisanceDir

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

