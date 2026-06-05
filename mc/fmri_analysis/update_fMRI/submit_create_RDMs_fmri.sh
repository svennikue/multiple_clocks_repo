#!/bin/sh
# Run subject-level RSA analysis.
# submit like bash submit_RSA_fmri.sh
# requires results from subject_GLM_loc_press_preICA.sh

# NOTE
# first activate the virtual envirnment in the shell you are running this from.
# unfortunately doesn't work with putting the command in here..# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"

conda init
source /home/fs0/xpsy1114/scratch/miniconda3/etc/profile.d/conda.sh
conda activate spyder-env


# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  analysisDir="/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
  fslDir="/Users/xpsy1114/fsl"
fi

# exclude subject 21
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 29 30 31 32 33 34; do
for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35; do
    echo now starting script that creates RDMs for subject ${subjectTag}.
    #fsl_sub -q short.q python ${analysisDir}/multiple_clocks_repo/scripts/create_fmri_model_RDMs_between_halves.py ${subjectTag}
    #python ${analysisDir}/multiple_clocks_repo/scripts/create_fmri_model_RDMs_between_halves.py ${subjectTag}
    python ${analysisDir}/multiple_clocks_repo/scripts/SLIM-create_fmri_model_RDMs_between_halves.py ${subjectTag}
done

# subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11']
# subjects = ['sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20', 'sub-22', 'sub-23']
# subjects = ['sub-24', 'sub-25', 'sub-26', 'sub-27', 'sub-28', 'sub-29', 'sub-30', 'sub-31', 'sub-32', 'sub-33', 'sub-34']