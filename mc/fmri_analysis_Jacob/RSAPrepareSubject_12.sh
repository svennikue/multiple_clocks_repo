#!/bin/sh
# Run subject-level GLM on preprocessed functional data

# Command line argument 1/1: subject tag
subjectTag=$1
echo Subject tag for this subject: $subjectTag

# Set version: which GLM is this (allows to easily change some settings per GLM)?
version=12
# Convert to zero padded number
version=$(printf "%02d" $version)

# Set whether this is for a surface (surf) or volumetric (vol) analysis
analysis=vol
# Set which preprocessed data to use as input
data=filtered_func_data_clean_smooth

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

# Set preprocessing folder for current 
preprocDic=$scratchDir/derivatives/sub-$subjectTag/func/preproc_clean.feat

# If this is a volumetric analysis: move requested data to the standard space
if [ $analysis == "vol" ]; then
		# Do warp to standard space, if it doesn't exist already
		if [ ! -f $preprocDic/${data}_standard.nii.gz ]; then 
			echo Volumetric analysis, warping ${data} to standard space
			applywarp --ref=${FSLDIR}/data/standard/MNI152_T1_2mm --in=$preprocDic/${data}.nii.gz --warp=$preprocDic/reg/example_func2standard_warp.nii.gz --out=$preprocDic/${data}_standard.nii.gz
		fi
		# Add _standard postfix to input data
		data=${data}_standard
fi

# Split the 4d nifti into separate volumes, as preferred by SPM
if [ ! -d $preprocDic/${data}_split ]; then
	echo Splitting 4d ${data} to separate volumes
	# Create split directory
	mkdir -p $preprocDic/${data}_split
	# Split 4d nifti
	fslsplit $preprocDic/${data}.nii.gz $preprocDic/${data}_split/vol
	# Unzip volumes as preferred by SPM
	gunzip $preprocDic/${data}_split/vol*.nii.gz
fi