#!/bin/sh
# Run preprocessing of functional data

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
# Construct func directory for derived file
funcDir=$derivDir/func
# And create directory for derived functional files
mkdir -p $funcDir

# Get number of volumes from fslinfo and some bash tricks
#numVols=$(echo $(echo $(fslinfo $rawDir/func/sub-${subjectTag}_bold.nii.gz) | grep -o -E ' dim4 [0-9]+' | sed 's/ dim4 //'))
numVols=$(fslval $rawDir/func/sub-${subjectTag}_bold.nii.gz dim4)

# Take preprocessing template, replace subject id and number of volumes with current values and save to new file
cat $homeDir/Analysis/Templates/preproc.fsf | sed "s/s01id01/$subjectTag/g" | sed "s/1668/${numVols}/g" | sed "s:/vols/Scratch/jacobb:${scratchDir}:g" > $funcDir/sub-${subjectTag}_design_preproc.fsf

# Finally: run feat with these parameters
feat $funcDir/sub-${subjectTag}_design_preproc.fsf