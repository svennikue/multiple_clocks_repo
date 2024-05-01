#!/bin/sh
# Run subject-level RSA analysis.
# submit like bash submit_create_subject_EVs.sh
# rfirst activate conda activate spyder-env

# NOTE
# first activate the virtual envirnment in the shell you are running this from.
# unfortunately doesn't work with putting the command in here..# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
conda init
conda activate spyder-env

# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  analysisDir="/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
  fslDir="/Users/xpsy1114/fsl"
fi

# 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34; do
    echo now creating EVs for RDMs for subject ${subjectTag}.
    fsl_sub -q short.q python ${analysisDir}/multiple_clocks_repo/scripts/create_EVs_for_RDMs.py ${subjectTag}
done