#!/bin/sh
# Prepare structural and fieldmap for preprocessing

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

# Make subject folder in derivatives folder: folder where any non-raw data gets stored (-p: don't if already exists)
mkdir -p $scratchDir/derivatives/sub-$subjectTag

# Construct anatomy directory for derived file
anatDir=$scratchDir/derivatives/sub-$subjectTag/anat
# And create directory for derived anatomy files
mkdir -p $anatDir

# Brain-extract structural file (see http://fsl.fmrib.ox.ac.uk/fslcourse/graduate/lectures/practicals/intro2/)
echo Brain-extracting structural file $scratchDir/$subjectTag/anat/sub-${subjectTag}_T1W.nii.gz, saving in $anatDir
# Do brain extraction with -R option to re-run bet multiple times and keep the best
bet $scratchDir/sub-$subjectTag/anat/sub-${subjectTag}_T1W.nii.gz $anatDir/sub-${subjectTag}_T1W_brain.nii.gz -R
# But also copy original structural file, including the head: non-linear registration looks for that file with the same filename as the beted file, minus _brain
cp $scratchDir/sub-$subjectTag/anat/sub-${subjectTag}_T1W.nii.gz $anatDir/sub-${subjectTag}_T1W.nii.gz

# Construct fieldmap directory for derived file
fmapDir=$scratchDir/derivatives/sub-$subjectTag/fmap
# And create directory for derived fieldmap files
mkdir -p $fmapDir

# Prepare fieldmap for registration (see http://fsl.fmrib.ox.ac.uk/fslcourse/graduate/lectures/practicals/registration/)
echo Preparing fieldmap files from $scratchDir/sub-$subjectTag/fmap/, saving in $fmapDir
# Select only first image of magnitude fieldmap (see https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide: two magnitude images (one for each echo time), pick the "best looking" one)
fslroi $scratchDir/sub-$subjectTag/fmap/sub-${subjectTag}_magnitude.nii.gz $fmapDir/sub-${subjectTag}_magnitude1.nii.gz 0 1
# Brain extract the magnitude fieldmap
bet $fmapDir/sub-${subjectTag}_magnitude1.nii.gz $fmapDir/sub-${subjectTag}_magnitude1_brain.nii.gz
# Create slice along z-axis for zero padding, with x and y dimensions equal to original image but z dimension of only 1
fslroi $fmapDir/sub-${subjectTag}_magnitude1_brain.nii.gz $fmapDir/sub-${subjectTag}_magnitude1_brain_slice.nii.gz 0 -1 0 -1 0 1 0 -1
# Make slice into zeros by thresholding with a really high number
fslmaths $fmapDir/sub-${subjectTag}_magnitude1_brain_slice.nii.gz -thr 9999 $fmapDir/sub-${subjectTag}_magnitude1_brain_slice_zero.nii.gz
# Add zero padding by merging zero slice and beted brain: if the beted brain touches the top edge of the image, you won't erode anything there
fslmerge -z $fmapDir/sub-${subjectTag}_magnitude1_brain_padded.nii.gz $fmapDir/sub-${subjectTag}_magnitude1_brain.nii.gz $fmapDir/sub-${subjectTag}_magnitude1_brain_slice_zero.nii.gz
# Erode image: shave off voxels near edge of brain, since phase difference is very noisy there
fslmaths $fmapDir/sub-${subjectTag}_magnitude1_brain_padded.nii.gz -ero $fmapDir/sub-${subjectTag}_magnitude1_brain_padded_ero.nii.gz
# Find out the original size along the z dimension using fslinfo and bash magic (output fslinfo, find dim3, only keep the number that comes after dim3)
#origSize=$(echo $(echo $(fslinfo $fmapDir/sub-${subjectTag}_magnitude1_brain.nii.gz) | grep -o -E ' dim3 [0-9]+' | sed 's/ dim3 //'))
origSize=$(fslval $fmapDir/sub-${subjectTag}_magnitude1_brain.nii.gz dim3)
# And remove the added zero padding slice from the eroded brain so its size matches the phasediff fieldmap
fslroi $fmapDir/sub-${subjectTag}_magnitude1_brain_padded_ero.nii.gz $fmapDir/sub-${subjectTag}_magnitude1_brain_ero.nii.gz 0 -1 0 -1 0 $origSize 0 -1
# Then prepare fieldmap from phase image, magnitude image, and difference in echo times between the two magnitude images
fsl_prepare_fieldmap SIEMENS $scratchDir/sub-$subjectTag/fmap/sub-${subjectTag}_phasediff.nii.gz $fmapDir/sub-${subjectTag}_magnitude1_brain_ero.nii.gz $fmapDir/sub-${subjectTag}_fieldmap.nii.gz 2.46