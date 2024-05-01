#!/bin/bash
# Musicbox study 2023_017, Svenja, Sep 2023

#Script to run brain extractions using Synthstrip

basedir=/home/fs0/xpsy1114/scratch/data/pilot/;

# should install python 3 every time when using synthstrip-singularity
module load Python

#for sub in 01 02 03 04 05 06 07 08 09 10 11; do
for sub in 34; do

cd /home/fs0/xpsy1114/scratch/data/derivatives/sub-${sub}/anat;

echo "I am extracting brain for subject $sub"

#with csf

fsl_sub -q short.q /vols/Scratch/flange/bin/synthstrip-singularity -i T1w_BiasCorr.anat/T1_biascorr.nii.gz -o sub-${sub}_T1w_biascorr_brain.nii.gz -m sub-${sub}_T1w_biascorr_brain_mask.nii.gz
# the masks will later be used for the searchlights.

#without CSF

fsl_sub -q short.q /vols/Scratch/flange/bin/synthstrip-singularity -i T1w_BiasCorr.anat/T1_biascorr.nii.gz -o sub-${sub}_T1w_biascorr_noCSF_brain.nii.gz -m sub-${sub}_T1w_biascorr_noCSF_brain_mask.nii.gz --no-csf

done
