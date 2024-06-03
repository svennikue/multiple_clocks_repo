# adjust subjects list, then run with bash feat_preproc.sh
# based on Jacob Bakermans, adjusted by Svenja Kuchenhoff for 7T
# this script runs the preprocessing from FEAT based on an .fsf file
# for the first pilot scan 
# usees the local fsl installation with the config file by Rick Lange. See email Marieke Martens 14th of dec 2023.
# submit like bash feat_preproc.sh (and adjust subject list before below)

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
fslDir="/opt/fmrib/fsl"
#export fslDir=~/scratch/fsl
#export PATH=$fslDir/share/fsl/bin/:$PATH

# run source setup_test_feat.sh instead
source $analysisDir/setup_test_feat.sh
# put the config file marieke sent somehwere, point at it in the .fsf (seem Riks email)
# change folder of set fmri(regstandard) "/Users/xpsy1114/fsl/data/standard/MNI152_T1_2mm_brain"
# to a folder that doesn thave the same one in the skull


# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  analysisDir="/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
  fslDir="/Users/xpsy1114/fsl"
fi

echo Now entering the loop ....
echo fslDir ist $fslDir
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do
# for subjectTag in "${subjects[@]}"; do
# try all those where the fieldmap thingy worked.
# for subjectTag in 01 02 03 04 07 08 09 10 11 12 14 15 16 17 20 21 23 24; do
# for subjectTag in 03 04 07 08 09; do
#for subjectTag in 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 ; do
for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34; do
    # first make sure that the mask exists.
    # Construct directory for derived data
    derivDir=$scratchDir/derivatives/sub-$subjectTag
    # only temporarily comment!!
    for task_half in 01 02; do
        if [ ! -f ${derivDir}/func/preproc_clean_${task_half}/reg/highres2example_func.mat ]; then
            echo now registering subject $subjectTag , task half $task_half
        else 
            echo reg matrix for $subjectTag does not exist
        fi
        
        flirt -in ${derivDir}/anat/sub-${subjectTag}_T1w_biascorr_noCSF_brain_mask.nii.gz -ref ${derivDir}/func/preproc_clean_${task_half}.feat/example_func.nii.gz -applyxfm -init ${derivDir}/func/preproc_clean_${task_half}.feat/reg/highres2example_func.mat -out ${derivDir}/anat/sub-${subjectTag}_T1w_noCSF_brain_mask_prep_func_${task_half}.nii.gz
        fslmaths ${derivDir}/anat/sub-${subjectTag}_T1w_noCSF_brain_mask_prep_func_${task_half}.nii.gz -thr 0.05 -fillh -bin ${derivDir}/anat/sub-${subjectTag}_T1w_noCSF_brain_mask_bin_func_${task_half}.nii.gz
    done


    echo Subject tag and folder for the current run: $subjectTag
    # Construct func directory for derived file
    funcDir=$derivDir/func
    # do this twice, once for pt1 and once for pt2
    # Get number of volumes from fslinfo and some bash tricks
    numVols=$(fslval $funcDir/sub-${subjectTag}_1_bold.nii.gz dim4)
    echo Extracted number of volumes for first pt: $numVols

    # compute the number of voxels
    dim1=$(fslval $funcDir/sub-${subjectTag}_1_bold.nii.gz dim1)
    dim2=$(fslval $funcDir/sub-${subjectTag}_1_bold.nii.gz dim2)
    dim3=$(fslval $funcDir/sub-${subjectTag}_1_bold.nii.gz dim3)
    dim4=$(fslval $funcDir/sub-${subjectTag}_1_bold.nii.gz dim4)
    numVoxels=$((dim1*dim2*dim3*dim4))
    echo Extracted number of voxels for first pt: $numVoxels

    # really important to here take the respective .fsf file of the sequence I prepared manually! Different TE, TR, echo spacing
    cat $analysisDir/templates/preproc_compl.fsf | sed "s:/vols/Scratch/xpsy1114/data/derivatives/sub-02:${derivDir}:g" | sed "s/1056291840/${numVoxels}/g" |sed "s/1415/${numVols}/g" | sed "s/sub-02/sub-$subjectTag/g" | sed "s:/vols/Scratch/xpsy1114/data:${scratchDir}:g" > $funcDir/sub-${subjectTag}_1_design_preproc.fsf
    echo Now starting first feat preproc one
    
    # test if the preproc folders already exist. If they do, delete, as these folders require a lot of space.
    if test -d ${funcDir}/preproc_clean_01.feat; then
      rm -r ${funcDir}/preproc_clean_01.feat
    fi
    # fsl_sub -q long.q -T 360 -R 30 feat $funcDir/sub-${subjectTag}_1_design_preproc_clean.fsf
    fsl_sub -q long.q -T 360 -R 30 feat $funcDir/sub-${subjectTag}_1_design_preproc.fsf

    numVols=$(fslval $funcDir/sub-${subjectTag}_2_bold.nii.gz dim4)
    echo Extracted number of volumes for second pt: $numVols

    # compute the number of voxels
    dim1=$(fslval $funcDir/sub-${subjectTag}_2_bold.nii.gz dim1)
    dim2=$(fslval $funcDir/sub-${subjectTag}_2_bold.nii.gz dim2)
    dim3=$(fslval $funcDir/sub-${subjectTag}_2_bold.nii.gz dim3)
    dim4=$(fslval $funcDir/sub-${subjectTag}_2_bold.nii.gz dim4)
    numVoxels=$((dim1*dim2*dim3*dim4))
    echo Extracted number of voxels for second pt: $numVoxels

    # note I am changing all to the second task half, except for the reference image. this is what the file shall be registered to.
    # cat $analysisDir/templates/try_to_fixit.fsf | sed "s/sub-01_1_bold/sub-${subjectTag}_2_bold/g"| sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/preproc_clean_01:${funcDir}/preproc_clean_02:g" | sed "s/1246648320/${numVoxels}/g" |sed "s/1670/${numVols}/g" | sed "s:/Users/xpsy1114/fsl:${fslDir}:g" | sed "s/sub-01/sub-$subjectTag/g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data:${scratchDir}:g" > $funcDir/sub-${subjectTag}_2_design_preproc_clean.fsf
    cat $analysisDir/templates/preproc_compl.fsf | sed "s/sub-02_T1w_noCSF_brain_mask_bin_func_01.nii.gz/sub-${subjectTag}_T1w_noCSF_brain_mask_bin_func_02.nii.gz/g" |sed "s/sub-02_1_bold/sub-${subjectTag}_2_bold/g"| sed "s/preproc_clean_01.feat/preproc_clean_02.feat/g" | sed "s:/vols/Scratch/xpsy1114/data/derivatives/sub-02:${derivDir}:g" | sed "s/1056291840/${numVoxels}/g" |sed "s/1415/${numVols}/g" | sed "s/sub-02/sub-$subjectTag/g" | sed "s:/vols/Scratch/xpsy1114/data:${scratchDir}:g" > $funcDir/sub-${subjectTag}_2_design_preproc.fsf

    # test if the preproc folders already exist. If they do, delete, as these folders require a lot of space.
    if test -d ${funcDir}/preproc_clean_02.feat; then
      rm -r ${funcDir}/preproc_clean_02.feat
    fi
    echo Now starting second feat preproc two
    # fsl_sub -q long.q -T 360 -R 30 feat $funcDir/sub-${subjectTag}_2_design_preproc_clean.fsf
    fsl_sub -q long.q -T 360 -R 30 feat $funcDir/sub-${subjectTag}_2_design_preproc.fsf
done

