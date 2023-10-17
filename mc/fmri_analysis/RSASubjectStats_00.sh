#!/bin/sh
# Run stats on subject data and hypothesis RDMs

# Command line argument 1/1: subject tag
subjectTag=$1
echo Subject tag for this subject: $subjectTag

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
funcDir=$scratchDir/derivatives/sub-$subjectTag/func

# Define RSA folder
rsaDir=$funcDir/RSA_$version

# Define RDM output folder, used as input here
inputDir=$rsaDir/RDM

# Define stats output folder
statsDir=$rsaDir/stats
# If it somehow already exists: remove
rm -rf $statsDir
# And create RDM folder
mkdir $statsDir

# Unzip the brain mask because it's used for header info and SPM doesn't like .nii.gz
gunzip $funcDir/preproc_nosmooth.feat/mask.nii.gz

# Create participant-specific matlab script that produces subject stats from data and hypothesis RDMs
cat $homeDir/Analysis/RSASubjectStatsWrapper.m | sed "s:inputFolder:\'$inputDir\':g" | sed "s:subjectTag:\'${subjectTag}\':g" | sed "s:outputFolder:\'$statsDir\':g" > $rsaDir/sub-${subjectTag}_stats_${version}.m

# Add analysis folder matlab path so subject script can call matlab function
export MATLABPATH=$homeDir/Analysis

# Display status
echo Now running Matlab script $rsaDir/sub-${subjectTag}_stats_${version}.m

# Finally: execute participant-specific matlab scripts to run GLM in SPM
matlab -nodisplay -nosplash \< $rsaDir/sub-${subjectTag}_stats_${version}.m

# And re-zip the brain mask
gzip $funcDir/preproc_nosmooth.feat/mask.nii