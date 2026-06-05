# Run subject-level RSA analysis.
# submit like fsl_sub -q short bash ../wrapper_store_data_RDMs.sh

scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"

conda init
source /home/fs0/xpsy1114/scratch/miniconda3/etc/profile.d/conda.sh
conda activate spyder-env

# fsl_sub -q short python ${analysisDir}/multiple_clocks_repo/scripts/store_group_RDM.py 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
python ${analysisDir}/multiple_clocks_repo/scripts/store_group_RDM.py rsa_config_DSR_stepwise_combos.json 01 02 03 04 05 06 07 08 09 10 11 13 14 15 16 17 18 19 22 23 24 25 26 27 28 30 31 32 33 34 35

