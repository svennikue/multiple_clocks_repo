#!/bin/sh
# Do registration of single subject GLM to freesurfer surface

# Command line argument 1/2: subject tag
subjectTag=$1
echo Subject tag for this subject: $subjectTag

# Command line argument 2/2: version of subject level GLM
version=19
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

# Run freesurfer's registration. See https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/RegisterFeatOntoAnatomical
reg-feat2anat --feat $featDir --subject $subjectTag
