#!/bin/sh
# Run freesurfer recon-all to reconstruct the brain surface from the structural image

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

# Construct directory where structural for this subject can be found
structDir=$scratchDir/sub-${subjectTag}/anat

# Add the freesurfer module. You need an additional line if you do that in a bash script, see https://sharepoint.nexus.ox.ac.uk/sites/NDCN/FMRIB/IT/User%20Guides/Modules.aspx
# These are not currently necessary because I load the module in ~/.local_profile 
#. $MODULESHOME/init/bash
#module load freesurfer/current

# Update freesurfer base directory for subject output data
SUBJECTS_DIR=$scratchDir/freesurfer/

# Check if unzipped (.nii instead of .nii.gz) exists
#if [ ! -f $structDir/sub-${subjectTag}_T1W.nii ]; then
  # If it doesn't: create .nii file by unzipping the .nii.gz, keeping the original file
#  gunzip -k $structDir/sub-${subjectTag}_T1W.nii.gz
#fi

# Run freesurfer's recon-all to reconstruct the cortical surface from a structural file
recon-all -subject $subjectTag -i $structDir/sub-${subjectTag}_T1W.nii.gz -all
