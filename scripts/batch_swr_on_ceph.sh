#!/bin/bash
# =============================================================================
# SLURM submitter for the SWR pipeline.
#
# Modelled on scripts/batch_on_ceph.sh (same condition-file idea, same
# micromamba activation, same logs/<timestamp>/ layout) but with CPU resources
# instead of that script's `-p gpu --gres=gpu:1 --mem=128G`, which this pipeline
# does not need and should not occupy.
#
# 
# Usage:
#   bash scripts/batch_swr_on_ceph.sh <job_name> <python_script> <condition_file> [mem] [cpus] [time] [partition]
#
# Examples:
#   bash scripts/batch_swr_on_ceph.sh swr_pre scripts/swr_extract_continuous.py \
#        condition_files/swr_sessions.txt 64G 4 0-04:00
#   bash scripts/batch_swr_on_ceph.sh swr_det scripts/swr_detect_session.py \
#        condition_files/swr_sessions.txt 24G 8 0-01:30
#
# One SLURM job per line of the condition file; each line is passed verbatim as
# arguments, so `--session=17 --analysis_name=swr_v1` reaches fire as-is.
#
# @author: Svenja Kuchenhoff
# =============================================================================
set -uo pipefail

JOB="${1:?usage: batch_swr_on_ceph.sh <job> <script> <conditions> [mem] [cpus] [time] [partition]}"
PYSCRIPT="${2:?missing python script}"
CONDFILE="${3:?missing condition file}"
MEM="${4:-32G}"
CPUS="${5:-4}"
TIME="${6:-0-02:00}"
PART="${7:-${SWR_PARTITION:-cpu}}"     # SWC partition name; override if different
ENVNAME="${SWR_ENV:-spyder_env}"

[ -f "$CONDFILE" ] || { echo "condition file not found: $CONDFILE"; exit 1; }
[ -f "$PYSCRIPT" ] || { echo "python script not found: $PYSCRIPT"; exit 1; }

# read conditions (one argument string per line, blank/# lines ignored)
job_array=()
while IFS= read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    job_array+=("$line")
done < "$CONDFILE"
[ "${#job_array[@]}" -gt 0 ] || { echo "no conditions in $CONDFILE"; exit 1; }

timestamp=$(date "+%Y-%m-%d_%Hh%M")
logs_path="./logs/${timestamp}_${JOB}"
mkdir -p "$logs_path"

echo "job        : $JOB"
echo "script     : $PYSCRIPT"
echo "conditions : $CONDFILE  (${#job_array[@]} tasks)"
echo "resources  : -p $PART -c $CPUS --mem=$MEM --time=$TIME"
echo "env        : $ENVNAME"
echo "logs       : $logs_path"
echo "-------------------------------------------------------------------------------"

for i in $(seq 0 $((${#job_array[@]} - 1))); do
    file_sbatch="$logs_path/${JOB}_${i}.sbatch"
    out_file="$logs_path/${JOB}_${i}.out"
    err_file="$logs_path/${JOB}_${i}.err"

cat <<EOT > "$file_sbatch"
#!/bin/bash
#SBATCH --job-name=${JOB}_${i}
#SBATCH --output=$out_file
#SBATCH --error=$err_file
#SBATCH -p $PART
#SBATCH -N 1
#SBATCH -c $CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME

source ~/.bashrc
micromamba activate $ENVNAME
python $PYSCRIPT ${job_array[i]}
EOT

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "  [dry] ${job_array[i]}"
    else
        sbatch "$file_sbatch" >/dev/null && echo "  submitted: ${job_array[i]}"
    fi
done

echo "-------------------------------------------------------------------------------"
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY RUN — nothing submitted. Re-run without DRY_RUN=1."
    echo "Inspect a generated script:  cat $logs_path/${JOB}_0.sbatch"
else
    echo "${#job_array[@]} jobs submitted. Watch with:  squeue -u \$USER"
    echo "Failures:  grep -l Error $logs_path/*.err"
fi
