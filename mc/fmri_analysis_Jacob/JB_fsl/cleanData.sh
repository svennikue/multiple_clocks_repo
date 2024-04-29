#!/bin/sh
# Prepare for and do cleaning of data: physiological artefacts regressors with PNM, motion artefacts with ICA-AROMA

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
preprocDir=$derivDir/func/preproc.feat
# Construct directory where physiological noise regressors will go
physioDir=$derivDir/func/physiology
# And create directory for physiology
mkdir -p $physioDir

# Set directory where original physiology file can be found (BIDS format)
physioFile=$rawDir/func/sub-${subjectTag}_physio.tsv

# Prepare physio file for use by PNM
fslFixText $physioFile $physioDir/physio_input.txt

# First part of PNM: runs popp to find peaks in cardiac and respiratory traces
pnm_stage1 -i $physioDir/physio_input.txt -o $physioDir/physio -s 50 --tr=1.235 --smoothcard=0.1 --smoothresp=0.1 --resp=2 --cardiac=4 --trigger=3 --rvt --heartrate

# Second part of PNM: generate EVs
pnm_evs -i $preprocDir/filtered_func_data.nii.gz -c $physioDir/physio_card.txt -r $physioDir/physio_resp.txt -o $physioDir/physio --tr=1.235 --oc=4 --or=4 --multc=0 --multr=0 --rvt=$physioDir/physio_rvt.txt --rvtsmooth=10 --heartrate=$physioDir/physio_hr.txt --heartratesmooth=10 --slicetiming=$homeDir/Analysis/Subjects/Physiology/sliceTiming.txt

# And make a list of all generated EVs in a text file
ls -1 `imglob -extensions $physioDir/physioev0*` > $physioDir/physio_evlist.txt

# Add python 3 module for installing seaborn. You need an additional line if you do that in a bash script, see https://sharepoint.nexus.ox.ac.uk/sites/NDCN/FMRIB/IT/User%20Guides/Modules.aspx
# These are not currently necessary because I load the module in ~/.local_profile 
#. $MODULESHOME/init/bash
#module load fmrib-python3

# I need additional modules to be able to run AROMA. Install using pip (won't install if already existing)
pip install --user pandas
pip install --user future
pip install --user seaborn

# Use ICA-AROMA to clean up motion artefacts; ideally, this would be done before temporal filtering, check if it works
python $homeDir/Analysis/ICA-AROMA/ICA_AROMA.py \
	-in $preprocDir/filtered_func_data.nii.gz \
	-out $derivDir/func/AROMA \
	-mc $preprocDir/mc/prefiltered_func_data_mcf.par \
	-affmat $preprocDir/reg/example_func2highres.mat \
	-warp $preprocDir/reg/highres2standard_warp.nii.gz \
	-md $preprocDir/filtered_func_data.ica \
	-overwrite
