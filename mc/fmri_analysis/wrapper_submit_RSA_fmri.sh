# Run subject-level RSA analysis.
# submit like fsl_sub -q long wrapper_submit_RSA_fmri.sh

# requires results from subject_GLM_loc_press_preICA.sh


scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
module load Miniconda3/24.1.2-0
conda init
#conda activate spyder-env
conda activate /home/fs0/xpsy1114/scratch/miniconda3/envs/spyder-env


#for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
for subjectTag in 01 02 03; do
    echo now running RSA for subject ${subjectTag}.
    module load Miniconda3/24.1.2-0
    conda activate /home/fs0/xpsy1114/scratch/miniconda3/envs/spyder-env
    python ${analysisDir}/multiple_clocks_repo/scripts/fMRI_do_RSA_between_halves.py ${subjectTag}
done