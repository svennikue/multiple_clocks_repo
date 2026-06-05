# Run subject-level RSA analysis.
# submit like fsl_sub -q short bash ../wrapper_smooth_stat_maps_subj.sh

scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"

conda init
source /home/fs0/xpsy1114/scratch/miniconda3/etc/profile.d/conda.sh
conda activate spyder-env

for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35 ; do
# for subjectTag in 26 ; do
# 08 09 10 11 12 13 14 15 16 17 18 19 20
# yes: 01 02 03 04 08 09 10 16 17 27 28 30 34 35
#for subjectTag in 35 ; do
    echo now computing smoothing RSA results of subject ${subjectTag}.
    # don't forget to alter the config file!
    python ${analysisDir}/multiple_clocks_repo/scripts/smooth_subject_space.py ${subjectTag} smooth5_config.json
done
