# this is an attempt to submit each python script while loading the conda environment indepenently for each subject, to avoid memory issues.
# this is the submit-wrapper

scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
configfile="rsa_config_quarters_DSR_controls.json"

# for subjectTag in 01 ; do
for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35 ; do
    echo "now submitting python wrapper for subject ${subjectTag}."
    fsl_sub -q short bash "${analysisDir}/wrapper_python_fMRI_RSA_clean_config.sh" "${subjectTag}" "${configfile}"
done