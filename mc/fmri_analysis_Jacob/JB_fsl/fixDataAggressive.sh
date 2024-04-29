#!/bin/sh
# After classifying ICs as noise, signal, or unkown, this script regresses out the noise components and then does spatial smoothing

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
# Set directory containing previously run preprocessing
preprocDir=$derivDir/func/preproc_clean.feat

# If fix was already fun, filtered_func_data_clean.nii.gz will already exist, so rename to avoid overwriting
[ ! -f $preprocDir/filtered_func_data_clean.nii.gz ] || mv $preprocDir/filtered_func_data_clean.nii.gz $preprocDir/filtered_func_data_clean_orig.nii.gz

# Use labeled ICA components and regress out their contribution; also regress out motion regressors. Do aggressive cleanup, since some motion stuff remains
$homeDir/Analysis/fix/fix -a $preprocDir/labels.txt -m -A

# Display progress to see if any error messages are caused by fix or by later commands
echo Finished fix using labels in $preprocDir/labels.txt

# Rename the new filtered_func_data_clean.nii.gz with "aggressive" postfix
mv $preprocDir/filtered_func_data_clean.nii.gz $preprocDir/filtered_func_data_clean_aggressive.nii.gz

# And if fix was already run, rename original file back again
[ ! -f $preprocDir/filtered_func_data_clean_orig.nii.gz ] || mv $preprocDir/filtered_func_data_clean_orig.nii.gz $preprocDir/filtered_func_data_clean.nii.gz

# Get median value of data
m=$(fslstats $preprocDir/filtered_func_data_clean_aggressive.nii.gz -k $preprocDir/mask.nii.gz -p 50)

# Multiply median by 0.75 to get brigthness threshold for susan
bth=$(echo 0.75*$m | bc)

# Get sigma for gaussian, given the FWHM of 5.0mm: multiply FWHM by 1/(2*sqrt(2*ln(2))) = 0.424660900144010
s=$(echo 0.424660900144010*5.0 | bc)

# Check if these were successful
echo Running SUSAN smoothing with threshold $bth, calculated from median $m, and sigma $s

# Do smoothing for later volumetric analysis
susan $preprocDir/filtered_func_data_clean_aggressive.nii.gz $bth $s 3 1 1 $preprocDir/mean_func.nii.gz $bth $preprocDir/filtered_func_data_clean_aggressive_smooth.nii.gz

# Remove some resulting files - the usan file is particularly big
rm $preprocDir/filtered_func_data_clean_aggressive_smooth_usan_size.nii.gz $preprocDir/filtered_func_data_clean_vn.nii.gz