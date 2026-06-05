# Run script that generates EVs per task half per subject.
# submit like fsl_sub -q long -T 360 -R 30 bash -i ../wrapper_create_EVs_clean_config.sh

# requires results from subject_GLM_loc_press_preICA.sh
# also requires having run clean_fmri_behaviour.py

# current config files:
# EV_config_fut-steps_split-buttons.json
# EV_config_fut-steps_states_split-buttons.json
# EV_config_all_rews_split-buttons.json
# EV_config_all_paths_rews_split-buttons.json

scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"

conda init
source /home/fs0/xpsy1114/scratch/miniconda3/etc/profile.d/conda.sh
conda activate spyder-env

#for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
# 05 06 07 08 09 10 11 12 13 14 15 18 19 20 22 24 25 31 32 33
# yes: 01 02 03 04 08 09 10 16 17 27 28 30 34 35
# for subjectTag in 27 28 29 ; do
for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35; do
# for subjectTag in 01 ; do
    echo now creating subject-level EV files for subject ${subjectTag}.
    # don't forget to alter the config file!
    python ${analysisDir}/multiple_clocks_repo/scripts/create_EVs_for_RDMs_from_clean_beh.py ${subjectTag} EV_config_all-paths-fixed_stickrews_split-buttons.json
done
