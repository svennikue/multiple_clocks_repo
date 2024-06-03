# Prepare being able to project the functional files I want to the surface
# to do this, first make sure I have the correct transformation files.
# FS script 02

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"

# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  analysisDir="/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
  fslDir="/Users/xpsy1114/fsl"
fi

module load freesurfer/current

# Update freesurfer base directory for subject output data
SUBJECTS_DIR=$scratchDir/freesurfer/


echo Now entering the loop ....
# Show what ended up being the scratch dir
echo Scratch directory is $scratchDir
# 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35
for subjectTag in 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
    for task_half in 01 02; do
        echo Subject tag and folder for the current run: $subjectTag and task half $task_half
        # Construct the respective feat directory
        #  directory in which to find example_func
        featDir=$scratchDir/derivatives/sub-$subjectTag/func/preproc_clean_${task_half}.feat 
        # fsl_sub -q verylong.q recon-all -subject sub-$subjectTag-test-synthstrip -i $anatDir/sub-${subjectTag}_T1w_biascorr_noCSF_brain.nii.gz -all
        fsl_sub -q veryshort.q reg-feat2anat --feat $featDir --subject sub-$subjectTag
    done
done
