#!/bin/sh
# Resample single subject COPEs to the surface

# Command line argument 1/2: subject tag
subjectTag=$1
echo Subject tag for this subject: $subjectTag

# Command line argument 2/2: version of subject level GLM
version=7
# Convert to zero padded number
version=$(printf "%02d" $version)

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

# Construct the feat folder for the current subject
featDir=$scratchDir/derivatives/sub-$subjectTag/func/glm_$version.feat

# Update freesurfer base directory for subject output data
SUBJECTS_DIR=$scratchDir/freesurfer/

# List all COPEs for this single subject analysis
allCopes=$featDir/stats/cope*.nii.gz
echo Found $(echo $allCopes | wc -w) copes for subject $subjectTag in GLM $version.

# Loop through copes and resample each to a common surface, see https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/FslGroupFeat
for currCope in $allCopes
do
  # Extract file name of current COPE
  baseName=${currCope##*/}
  # Remove extension
  baseName=${baseName%.nii.gz}
  # Output which cope is being processed
  echo "-- Processing $baseName ---"
  # Run freesurfer's resampling with mris_preproc for left hemisphere
  mris_preproc --target fsaverage --hemi lh --fwhm 5 \
    --out $featDir/stats/freesurfer/lh.$baseName.mgh \
    --iv $featDir/stats/$baseName.nii.gz $featDir/reg/freesurfer/anat2exf.register.dat
  # And do the same for the right hemisphere
  mris_preproc --target fsaverage --hemi rh --fwhm 5 \
    --out $featDir/stats/freesurfer/rh.$baseName.mgh \
    --iv $featDir/stats/$baseName.nii.gz $featDir/reg/freesurfer/anat2exf.register.dat  
  # Now run freesurfer's resampling with mris_preproc for left hemisphere for the varcopes
  mris_preproc --target fsaverage --hemi lh --fwhm 5 \
    --out $featDir/stats/freesurfer/lh.var$baseName.mgh \
    --iv $featDir/stats/var$baseName.nii.gz $featDir/reg/freesurfer/anat2exf.register.dat
  # And do the same for the right hemisphere
  mris_preproc --target fsaverage --hemi rh --fwhm 5 \
    --out $featDir/stats/freesurfer/rh.var$baseName.mgh \
    --iv $featDir/stats/var$baseName.nii.gz $featDir/reg/freesurfer/anat2exf.register.dat    
done
