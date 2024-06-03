#!/bin/bash
# Musicbox study 2023_017, Svenja, Sep 2023

#Script to run bias correction for T1 and set up derivatives directory.
# submit like: bash BiasCorr_T1.sh &

scratchDir="/home/fs0/xpsy1114/scratch/data"

#for sub in 01 02 03 04 05 06 07; do
#for sub in 01 02 03 04 05 06 07 08 09 10 11; do
#cd ${scratchDir}/sub-${sub}/anat
#if [ ! -d ${derivdir}/anat/T1BiasCorr ]; then # make folder if not exist
#    mkdir T1BiasCorr
#fi


for subjectTag in 34; do
    echo "now looking at folder: $scratchDir/pilot/sub-$subjectTag"
    # Make subject folder in derivatives folder: folder where any non-raw data gets stored (-p: don't if already exists)
    mkdir -p $scratchDir/derivatives
    mkdir -p $scratchDir/derivatives/sub-$subjectTag

    # Construct anatomy directory for derived file
    anatDir=$scratchDir/derivatives/sub-$subjectTag/anat
    # And create directory for derived anatomy files
    mkdir -p $anatDir
    cp $scratchDir/pilot/sub-${subjectTag}/anat/sub-${subjectTag}_T1w.nii.gz $anatDir/sub-${subjectTag}_T1w.nii.gz

    cd $anatDir
    fsl_sub -q short.q fsl_anat -i sub-${subjectTag}_T1w.nii.gz -o ${anatDir}/T1w_BiasCorr -t T1 --betfparam==0.1
    # only output needed is: bias corrected T1; output directory would be T1w_BiasCorr.anat

done
