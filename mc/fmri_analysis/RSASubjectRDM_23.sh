#!/bin/sh
# Run subject-level GLM on preprocessed functional data

# Command line argument 1/1: subject tag
subjectTag=$1
echo Subject tag for this subject: $subjectTag

# Set version: which GLM is this (allows to easily change some settings per GLM)?
version=23
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

# Define RDM output folder
rdmDir=$rsaDir/RDM
# If it somehow already exists: remove
rm -rf $rdmDir
# And create RDM folder
mkdir $rdmDir

# Define SPM output folder
inputDir=$rsaDir/SPMout

# Create participant-specific matlab script that calls the function to generate EVs and run the GLM - use wrapper trick to pass arguments to matlab function
cat $homeDir/Analysis/RSAGenerateRDMsWrapper.m | sed "s:RSARunSearchlights:RSARunSearchlightsVol:g" | sed "s:inputFolder:\'$inputDir\':g" | sed "s:subjectTag:\'${subjectTag}':g" | sed "s:outputFolder:\'$rdmDir\':g" > $rsaDir/sub-${subjectTag}_RDMs_${version}.m

# Add analysis folder matlab path so subject script can call matlab function
export MATLABPATH=$homeDir/Analysis

# Change directory to rsa dir because running the searchlights creates a TON of temp files
cd $rsaDir

# Display status
echo Now running Matlab script $rsaDir/sub-${subjectTag}_RDMs_${version}.m

# Finally: execute participant-specific matlab scripts to run GLM in SPM
matlab -nodisplay -nosplash \< $rsaDir/sub-${subjectTag}_RDMs_${version}.m