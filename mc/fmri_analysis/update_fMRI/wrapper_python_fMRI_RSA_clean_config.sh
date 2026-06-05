#!/bin/bash
set -euo pipefail

scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"

# Read positional args
subjectTag="${1:-}"            # $1 must be subject tag
configfile="${2:-rsa_config_DSR_hamming_path_rew_sep_combos.json}"  # $2 must be the config file

echo "Subject: $subjectTag"
echo "Config: $configfile"

# Activate conda environment (recommended way in non-interactive shells)
# Make sure this path is correct on your system.
source /home/fs0/xpsy1114/scratch/miniconda3/etc/profile.d/conda.sh
conda activate spyder-env

# Run python and forward args
python "${analysisDir}/multiple_clocks_repo/scripts/fMRI_run_RSA_without_rsatoolbox_clean.py" \
    "${subjectTag}" "${configfile}"
