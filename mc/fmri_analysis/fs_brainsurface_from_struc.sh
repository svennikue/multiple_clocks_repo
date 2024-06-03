# Run freesurfer recon-all to reconstruct the brain surface from the structural image
# FS script 01

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

for subjectTag in 10; do
    echo Subject tag and folder for the current run: $subjectTag
    # Construct directory for derived data
    anatDir=$scratchDir/derivatives/sub-$subjectTag/anat
    # fsl_sub -q verylong.q recon-all -subject sub-$subjectTag-test-synthstrip -i $anatDir/sub-${subjectTag}_T1w_biascorr_noCSF_brain.nii.gz -all
    fsl_sub -q verylong.q recon-all -subject sub-$subjectTag -i $anatDir/T1w_BiasCorr.anat/T1_biascorr.nii.gz -all
done
