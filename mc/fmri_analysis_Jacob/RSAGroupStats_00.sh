#!/bin/sh
# Run subject-level GLM on preprocessed functional data

# Command line argument 1/1: subject tag for each subject that you want to run the group analysis for
subjects=$@
echo Subjects in this group analysis: $subjects

# Set version: which GLM is this (allows to easily change some settings per GLM)?
version=0
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

# Construct the folder for function analysis for the current subject
groupDir=$scratchDir/derivatives/group

# Define RSA folder
rsaDir=$groupDir/RSA_$version
# If it somehow already exists: remove
rm -rf $rsaDir
# And create RDM folder
mkdir $rsaDir

# Create participant-specific matlab script that does group statistics
cat $homeDir/Analysis/RSAGroupStatsWrapper.m | sed "s:version:\'$version\':g" | sed "s:subjectList:\'$subjects\':g" | sed "s:outputFolder:\'$rsaDir\':g" > $rsaDir/group_${version}.m

# Add analysis folder matlab path so subject script can call matlab function
export MATLABPATH=$homeDir/Analysis

# Display status
echo Now running Matlab script $rsaDir/group_${version}.m

# Finally: execute participant-specific matlab scripts to run GLM in SPM
matlab -nodisplay -nosplash \< $rsaDir/group_${version}.m
