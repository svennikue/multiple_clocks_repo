#!/bin/sh
# Run subject-level GLM on preprocessed functional data

# Command line argument 1/1: subject tag
subjectTag=$1
echo Subject tag for this subject: $subjectTag

# Set version: which GLM is this (allows to easily change some settings per GLM)?
version=10
# Convert to zero padded number
version=$(printf "%02d" $version)

# Set block: which half of the data is being used for fitting this GLM
block=1

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

# If somehow the glm feat folder already exists: delete it
rm -rf $funcDir/glm_${version}.feat

# Copy the folder that was used for preprocessing so it can be used for the glm
cp -r $funcDir/preproc.feat $funcDir/glm_${version}.feat

# Remove filtered_func_data.ica since it won't be used
rm -r $funcDir/glm_${version}.feat/filtered_func_data.ica

# Make EVs directory if it doesn't exist already
mkdir -p $funcDir/EVs_${version}

# Copy in the contrasts json file to the EV directory
cp $homeDir/Analysis/Templates/contrasts_${version}.json $funcDir/EVs_${version}/contrasts.json

# Create participant-specific matlab script that calls the function to generate EVs and GLM template - use wrapper trick to pass arguments to matlab function. 
# Note that the inputFile = subjectTag, inputFolder = version, and separateBlocks = block in encodingGenerateEVs.m
cat $homeDir/Analysis/generateEVsWrapper.m | sed "s:generateEVs:encodingGenerateEVs:g" | sed "s:inputFile:\'$subjectTag\':g" | sed "s:outputFolder:\'$version\':g" | sed "s/separateBlocks/$block/g" | sed "s/doPlot/false/g" | sed "s:templateFile:\'$homeDir/Analysis/Templates/glm_subject_${version}.fsf\':g" | sed "s:inputFolder:\'$funcDir/EVs_${version}\':g" > $funcDir/sub-${subjectTag}_generateEVs_${version}.m

# Add analysis folder matlab path so subject script can call matlab function
export MATLABPATH=$homeDir/Analysis

# Execute participant-specific matlab scripts to generate EVs
matlab -nodisplay -nosplash \< $funcDir/sub-${subjectTag}_generateEVs_${version}.m

# Remove filtered_func_data as that contains the full dataset, and we will only use one half
rm $funcDir/glm_${version}.feat/filtered_func_data.nii.gz

# Rename the single block filtered_func_data so it will be used by feat
mv $funcDir/glm_${version}.feat/filtered_func_data_$block.nii.gz $funcDir/glm_${version}.feat/filtered_func_data.nii.gz

# Get number of volumes from fslinfo and some bash tricks
numVols=$(fslval $funcDir/glm_${version}.feat/filtered_func_data.nii.gz dim4)

# Display outcome: how many volumes are there?
echo Found $numVols volumes in $funcDir/glm_${version}.feat/filtered_func_data.nii.gz

# Take preprocessing template, replace subject id and number of volumes with current values, update output directory, update scratch directory, and save to new file
cat $homeDir/Analysis/Templates/glm_subject_${version}_full.fsf | sed "s/s01id01/$subjectTag/g" | sed "s/glm.feat/glm_${version}.feat/g" | sed "s:/EVs/:/EVs_${version}/:g" | sed "s/1668/${numVols}/g" | sed "s:/Volumes/Scratch_jacobb:${scratchDir}:g" | sed "s:/home/fs0/jacobb/scratch:${scratchDir}:g" > $funcDir/sub-${subjectTag}_design_glm_${version}.fsf

# Finally: run feat with these parameters
feat $funcDir/sub-${subjectTag}_design_glm_${version}.fsf