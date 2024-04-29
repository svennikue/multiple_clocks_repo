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
# If somehow the SPM folder already exists: delete it
rm -rf $rsaDir
# Create SPM folder
mkdir $rsaDir

# Define input folder
inputDir=$rsaDir/SPMin
# And output folder
outputDir=$rsaDir/SPMout
# And create both
mkdir $inputDir
mkdir $outputDir

# Create participant-specific matlab script that calls the function to generate EVs and run the GLM
echo "RSAGenerateEVsSplit('$homeDir/Analysis/Subjects/${subjectTag}.mat', '$inputDir', 1.235);
RSAGenerateGLMSplit('$inputDir/EVs.mat', '${subjectTag}', '$outputDir');" > $inputDir/sub-${subjectTag}_runGLM_${version}.m

# Add analysis folder matlab path so subject script can call matlab function
export MATLABPATH=$homeDir/Analysis

# Display status
echo Now running Matlab script $inputDir/sub-${subjectTag}_runGLM_${version}.m

# Finally: execute participant-specific matlab scripts to run GLM in SPM
matlab -nodisplay -nosplash \< $inputDir/sub-${subjectTag}_runGLM_${version}.m
