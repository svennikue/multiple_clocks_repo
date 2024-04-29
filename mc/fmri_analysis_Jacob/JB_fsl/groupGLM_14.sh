#!/bin/sh
# Run subject-level GLM on preprocessed functional data

# Command line argument 1/1: subject tag for each subject that you want to run the group analysis for
subjects=$@
echo Subjects in this group analysis: $subjects

# Set version: which GLM is this (allows to easily change some settings per GLM)?
version=14
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

# Construct the folder for group analysis for the current subject
groupDir=$scratchDir/derivatives/group

# If somehow the glm feat folder already exists: delete it
rm -rf $groupDir/group_${version}.feat

# Get first subject to extract number of copes
set -- $subjects
firstSub=$1

# List number of cope files in subject directory
copes=$(ls -1 $scratchDir/derivatives/sub-${firstSub}/func/glm_${version}.feat/stats/cope*.nii.gz | wc -l)
echo Found $copes COPEs in subject $firstSub

# Create run-specific matlab script that calls the function to generate group GLM design template - use wrapper trick to pass arguments to matlab function
cat $homeDir/Analysis/generateGLMgroupWrapper.m | sed "s:templateFile:\'$homeDir/Analysis/Templates/glm_group_${version}.fsf\':g" | sed "s:subjectDir:\'$homeDir/Analysis/Subjects\':g" | sed "s:subjectList:\'$subjects\':g" | sed "s:subjectContrasts:$copes:g" > $groupDir/group_generateDesign_${version}.m

# Add analysis folder to matlab path so subject script can call matlab function
export MATLABPATH=$homeDir/Analysis

# Execute run-specific matlab script to generate design
matlab -nodisplay -nosplash \< $groupDir/group_generateDesign_${version}.m

# Take preprocessing template, but replace scratch directory with the appropriate current scratch directory.
cat $homeDir/Analysis/Templates/glm_group_${version}_full.fsf | sed "s/group.gfeat/group_${version}.gfeat/g" | sed "s/glm.feat/glm_${version}.feat/g" | sed "s:/Volumes/Scratch_jacobb:${scratchDir}:g" | sed "s:/vols/Scratch/jacobb:${scratchDir}:g" | sed "s:/home/fs0/jacobb/scratch:${scratchDir}:g" > $groupDir/group_design_${version}.fsf

# Finally: run feat with these parameters
feat $groupDir/group_design_${version}.fsf
