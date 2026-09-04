#!/bin/bash
# Submit one job per (subject, instruction epoch) for the instruction-phase RSA.
#
# Replaces the per-TR submit loop. The GLMs are no longer named by a TR index
# ('01-TR4') but by the epoch they measure ('instr_see-A-first'), so instead of
# overriding the config's TR field this overrides regression_version and sets
# TR to null -- fMRI_run_RSA_instruction.py then treats regression_version as
# the full GLM name (see load_data_EVs_instr_TRwise).
#
# wrapper_python_fMRI_RSA_clean_config.sh does NOT need changing: it still just
# forwards subject, config and script name to python.

scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
cond_dir="${analysisDir}/multiple_clocks_repo/condition_files"
base_config="rsa_instruction_cumulative_rew.json"
scriptname="fMRI_run_RSA_instruction.py"

# must match EV_config_instruction.json -- create_EVs_instruction_period.py
# prints exactly this list when it runs.
version="instr"
glm_names="see-A-first see-B-first see-C-first see-D-first see-A-second see-B-second see-C-second see-D-second collapsed-first-instruction collapsed-second-instruction empty-screen"

module load fsl

for glm_name in $glm_names; do
    epoch_config="${base_config%.json}_${glm_name}.json"
    # snapshot of the base config, pointed at this epoch's GLM
    python -c "
import json
c = json.load(open('${cond_dir}/${base_config}'))
c['regression_version'] = '${version}_${glm_name}'
c['TR'] = None
json.dump(c, open('${cond_dir}/${epoch_config}', 'w'), indent=2)
"
    for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35; do
        echo "submitting subject ${subjectTag}, epoch ${glm_name}"
        fsl_sub -T 30 bash "${analysisDir}/wrapper_python_fMRI_RSA_clean_config.sh" \
            "${subjectTag}" "${epoch_config}" "${scriptname}"
    done
done
