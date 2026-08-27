#!/bin/bash
#SBATCH --job-name=swr_pre_7
#SBATCH --output=./logs/2026-08-27_12h07_swr_pre/swr_pre_7.out
#SBATCH --error=./logs/2026-08-27_12h07_swr_pre/swr_pre_7.err
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=0-04:00

source ~/.bashrc
micromamba activate spyder_env
python scripts/swr_extract_continuous.py --session=17 --analysis_name=swr_v1
