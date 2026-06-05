# Run subject-level RSA analysis.
# submit like fsl_sub -q long.q -T 360 -R 30 bash -i ../wrapper_submit_RSA_fmri.sh

# requires results from subject_GLM_loc_press_preICA.sh


scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"

glm_version="03-4"
#RSA_version="03-1"
RSA_version="simple-clean_loc-fut-rews-state_23-10-2025"

#for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
# 05 06 07 08 09 10 11 12 13 14 15 18 19 20 22 24 25 31 32 33
# yes: 01 02 03 04 08 09 10 16 17 27 28 30 34 35
# for subjectTag in 27 28 29 ; do
for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
    echo now renaming RSA result folders for subject ${subjectTag}.
    mv ${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}_smooth5 ${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_state_only_glmbase_${glm_version}_smooth5
    mv ${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}_smooth5 ${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_state_only_glmbase_${glm_version}_smooth5
    # don't forget to alter the config file!
done
