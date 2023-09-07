#!/bin/sh
# Resample single subject COPEs to the surface

# Command line argument 1/1: subject tag for each subject that you want to run the group analysis for
subjects=$@
echo Subjects in this group analysis: $subjects

# Version of subject level GLM
version=7
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

# Construct the folder for group analysis 
groupDir=$scratchDir/derivatives/group/fs_${version}

# If the freesurfer group analysis folder already exists: delete it
rm -rf $groupDir

# And make the freesurfer group directory
mkdir $groupDir

# Get first subject to extract number of copes
set -- $subjects
firstSub=$1

# Make directory to store glm input
mkdir $groupDir/in

# Update freesurfer base directory for subject output data
SUBJECTS_DIR=$scratchDir/freesurfer/

# Now I'll need to concatenate each of the copes for all the subjects ("stack your subjects into one file" according to https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/FslGroupFeat)
for hemi in lh rh; do
	# List number of cope files in subject directory - change for tstat
	copes=$(ls -1 $scratchDir/derivatives/sub-${firstSub}/func/glm_${version}.feat/stats/freesurfer/$hemi.cope*.mgh | wc -l)
	echo Found $copes COPEs in hemisphere $hemi of subject $firstSub	
	# Create command that concatenates all subjects for each cope
	for ((currCope=1;currCope<=copes;currCope++)); do
		# Set the beginning of the concatenation command: where to store the output
		concat="mri_concat --o $groupDir/in/$hemi.cope$currCope.mgh"
		# Then run through all subjects and add their cope to the concatenation command - change for tstat
		for currSub in $subjects; do
			concat="$concat --i $scratchDir/derivatives/sub-${currSub}/func/glm_${version}.feat/stats/freesurfer/$hemi.cope$currCope.mgh"	 
		done
		# Echo the concatenation command to see if it worked
		echo The command that will concatenate cope $currCope for all subjects is $concat
		# Finally, run the concatenation 
		eval $concat
	done
done

# Create run-specific matlab script that calls the function to generate group GLM design template - use wrapper trick to pass arguments to matlab function
cat $homeDir/Analysis/generateXgroupWrapper.m | sed "s:subjectDir:\'$homeDir/Analysis/Subjects\':g" | sed "s:subjectList:\'$subjects\':g" | sed "s:outDir:\'$groupDir\':g" > $groupDir/group_generateDesign_${version}.m

# Add analysis folder to matlab path so subject script can call matlab function
export MATLABPATH=$homeDir/Analysis

# Execute run-specific matlab script to generate design
matlab -nodisplay -nosplash \< $groupDir/group_generateDesign_${version}.m

# Run freesurfer's glmfit for group analysis, separately for each cope, see https://surfer.nmr.mgh.harvard.edu/fswiki/mri_glmfit
for hemi in lh rh; do
	# List number of cope files in subject directory
	copes=$(ls -1 $groupDir/in/$hemi.cope*.mgh | wc -l)
	echo Found $copes COPEs in input directory for GLM	
	# Run glmfit for each cope
	for ((currCope=1;currCope<=copes;currCope++)); do
		# The --label argument sets the mask. Here I use a custom label that is extended around the entorhinal cortex, made with freesurferLabel.sh		
		mri_glmfit --y $groupDir/in/$hemi.cope$currCope.mgh --X $groupDir/X.mat --C $groupDir/C.txt --glmdir $groupDir/$hemi.cope$currCope --label $homeDir/Analysis/Labels/$hemi.cortex_ext.label --surf fsaverage $hemi 
	done
done