# Run subject-level RSA analysis.
# submit like fsl_sub -q long.q -T 360 -R 30 bash -i ../wrapper_submit_RSA_fmri.sh

# requires results from subject_GLM_loc_press_preICA.sh


scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"

conda init
source /home/fs0/xpsy1114/scratch/miniconda3/etc/profile.d/conda.sh
conda activate spyder-env

# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
# 05 06 07 08 09 10 11 12 13 14 15 18 19 20 22 24 25 31 32 33
# yes: 01 02 03 04 08 09 10 16 17 27 28 30 34 35
for subjectTag in 27 28 29 ; do
    echo now computing RSA for subject ${subjectTag}.
    python ${analysisDir}/multiple_clocks_repo/scripts/fMRI_run_RSA_without_rsatoolbox.py ${subjectTag}
done
