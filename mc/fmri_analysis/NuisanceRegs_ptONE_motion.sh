#!/bin/sh
# Produce nuisance regressors to get rid of motion artefacts: motion outliers and CSF time series
# requires feat preproc to have run before.
# submit like bash NuisanceRegs_ptONE_motion.sh 


# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
#fslDir="/opt/fmrib/fsl"
export fslDir=~/scratch/fsl
export PATH=$fslDir/share/fsl/bin/:$PATH
source $fslDir/etc/fslconf/fsl.sh

# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  analysisDir="/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
  fslDir="/Users/xpsy1114/fsl"
fi
echo Scratch directory is $scratchDir

module load MATLAB 

# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34; do
# for subjectTag in 03 04 07 08 09; do
#for subjectTag in 03 04 07 08 09 10 11 12 14 15 16 17 20 21 23 24; do
for subjectTag in 23 33; do
    # Command line argument 1/1: subject tag
    echo Subject tag for this subject: $subjectTag
    
    # Construct directory for derived data
    derivDir=$scratchDir/derivatives/sub-$subjectTag
    # Set directory containing previously run preprocessing

    # do this twice, once for pt1 and once for pt2
    # this has to be 01 and 02. Adjust in the future!

    # create a loop from here to the end (marked as well)
    for task_half in 01 02; do
      #for task_half in 02; do
        preprocDir=$derivDir/func/preproc_clean_${task_half}.feat
        # Construct directory for nuisance regressors
        nuisanceDir=$derivDir/motion/nuisance_${task_half}
        # Remove if already exists
        # rm -rf $nuisanceDir
        # And create nuisance regressor directory
        if [ -d "$nuisanceDir" ]; then
          echo "Nuisance folder already exists!"
        else
          mkdir -p $nuisanceDir
          echo Creating nuisance dir: $nuisanceDir
        fi

        echo now create motion outliers in this nuisance dir $nuisanceDir

        # Generate matrix of motion outlier regressors; don't rerun motion correction
        fsl_sub -q long.q -T 360 -R 30 fsl_motion_outliers -i $preprocDir/filtered_func_data.nii.gz -o $nuisanceDir/motionOutliers.txt -p $nuisanceDir/motionOutliers.png --nomoco
    done
done
