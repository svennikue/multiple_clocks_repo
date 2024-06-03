# this resamples individual EVs to surface space. 
# FS script 03

max_no_EVs=40
glm_version="03"


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
for subjectTag in 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
    for task_half in 01 02; do
        echo Subject tag and folder for the current run: $subjectTag and task half $task_half
        # Construct the respective feat directory
        #  directory in which the registration matrix can be found
        featregDir=$scratchDir/derivatives/sub-$subjectTag/func/preproc_clean_${task_half}.feat
        featstatsDir=$scratchDir/derivatives/sub-$subjectTag/func/glm_${glm_version}_pt${task_half}.feat/stats
        
        echo currently working on files from $featstatsDir

        outDir=$scratchDir/freesurfer/sub-$subjectTag/glm_${glm_version}_pt${task_half}.feat
        echo this is where I store the new files $outDir
        if [ ! -d $OutDir ]; then
            mkdir $OutDir
        fi

        # List all EVs for this single subject analysis
        # allEVs=()

        # Loop through all files matching the pattern pe*.nii.gz
        for currEV in ${featstatsDir}/pe*.nii.gz; do
            # Extract the number from the filename using parameter expansion
            filename=$(basename "$currEV")
            number=$(echo "$filename" | sed -e 's/^pe\([0-9]\+\)\.nii\.gz$/\1/')
            # Check if the number is within the desired range
            #echo checking if file is equal to this: $number
            if [[ "$number" =~ ^[0-9]+$ ]] && (( number >= 0 && number <= ${max_no_EVs} )); then
                # Extract file name of current EV
                baseName=${currEV##*/}
                # Remove extension
                baseName=${baseName%.nii.gz}
                # Output which cope is being processed
                echo "-- Processing $baseName ---"
                # Run freesurfer's resampling with mris_preproc for left hemisphere
                # anat2exf is from example func to anatomical surface
                fsl_sub -q long.q mris_preproc --target sub-$subjectTag --hemi lh \
                    --out $outDir/lh.$baseName.mgh \
                    --iv $featstatsDir/$baseName.nii.gz $featregDir/reg/freesurfer/anat2exf.register.dat
                # And do the same for the right hemisphere
                fsl_sub -q long.q mris_preproc --target sub-$subjectTag --hemi rh \
                    --out $outDir/rh.$baseName.mgh \
                    --iv $featstatsDir/$baseName.nii.gz $featregDir/reg/freesurfer/anat2exf.register.dat 
                # later:  anat2std.register.dat - init  FS  reg from anat to FSL-standard.
            fi
        done

    done
done