#!/bin/bash
# Run one python RSA job inside the conda env, as submitted by fsl_sub.
#
#   bash wrapper_python_fMRI_RSA_clean_config.sh <subjectTag> [config.json] [script.py]
#
# fsl_sub jobs start from a non-interactive shell that has never sourced the
# user's profile, so conda has to be initialised here by hand before `activate`
# works at all.
set -euo pipefail

analysisDir="/home/fs0/xpsy1114/scratch/analysis"

subjectTag="${1:-}"            # $1 must be subject tag
configfile="${2:-rsa_config_DSR_hamming_path_rew_sep_combos.json}"  # $2 the config file
# $3 the python script to run. It used to be hardcoded, while callers were
# already passing a script name as $3 -- so submitting the instruction-epoch RSA
# silently ran fMRI_run_RSA_without_rsatoolbox_clean.py instead. The default
# keeps the old callers working.
scriptname="${3:-fMRI_run_RSA_without_rsatoolbox_clean.py}"

echo "Subject: $subjectTag"
echo "Config:  $configfile"
echo "Script:  $scriptname"

# Activate conda environment (recommended way in non-interactive shells)
source /home/fs0/xpsy1114/scratch/miniconda3/etc/profile.d/conda.sh
conda activate spyder-env

python "${analysisDir}/multiple_clocks_repo/scripts/${scriptname}" \
    "${subjectTag}" "${configfile}"
