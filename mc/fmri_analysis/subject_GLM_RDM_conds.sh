#!/bin/sh
# Run subject-level GLM on preprocessed functional data, but without ICA cleaning
# submit like bash subject_GLM_RDM_conds.sh
# requires EV directory with EVs and a subject fsf file before, and the 
# filtered_func dataset, and the nuisance regs.

version="03-e"
echo this is version $version

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
#fslDir="/opt/fmrib/fsl"
export fslDir=~/scratch/fsl
fslDir="~/scratch/fsl"
export PATH=$fslDir/share/fsl/bin/:$PATH
source $fslDir/etc/fslconf/fsl.sh

# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  analysisDir="/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
  fslDir="/Users/xpsy1114/fsl"
fi

echo Now entering the loop ....
# Show what ended up being the scratch dir
echo Scratch directory is $scratchDir
# 01 02 03 04 05 06 07 08 09 10 11 
for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34; do
    # for subjectTag in 33 34; do
    #for subjectTag in 01; do
    echo Subject tag and folder for the current run: $subjectTag
    # for subjectTag in "${subjects[@]}"; do
    # Command line argument 1/1: subject tag
    echo Subject tag for this subject: $subjectTag and for GLM no $version
    
    # Construct directory for derived data
    derivDir=$scratchDir/derivatives/sub-$subjectTag

    # Construct the folder for function analysis for the current subject
    funcDir=$derivDir/func

    # do this twice, once for pt1 and once for pt2

    # create a loop from here to the end (marked as well)
    # for task_half in "${task_halves[@]}"; do
    for task_half in 1 2 ; do
        glmDir="$funcDir/glm_${version}_pt0${task_half}.feat" 
        if [ -d "$glmDir" ]; then
          rm -r $glmDir
          rm -r "funcDir/glm_${version}_pt0${task_half}+.feat"
        fi

        nuisanceFile="/motion/nuisance_0${task_half}/combined.txt"
        echo this will be the glm directory we are setting up: $glmDir

        EV_dir="$funcDir/EVs_${version}_pt0${task_half}"
        # mkdir -p $EV_dir

        if [ -d "$EV_dir" ]; then
          echo "EV Folder exists, all good!"
        else
          echo this is supposed to be the EV folder: $EV_dir 
          echo "ERROR: EV Folder does not exists!"
        fi

        # Get number of volumes from fslinfo and some bash tricks
        numVols=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim4)
        # Display outcome: how many volumes are there?
        echo Found $numVols volumes in $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz

        # compute the number of voxels
        dim1=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim1)
        dim2=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim2)
        dim3=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim3)
        dim4=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim4)
        numVoxels=$((dim1*dim2*dim3*dim4))
        echo Extracted number of voxels for first pt: $numVoxels

        # cat ${analysisDir}/templates/loc_press.fsf | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/preproc_clean_01:${funcDir}/preproc_clean_${task_half}:g" | sed "s/1246648320/${numVoxels}/g" |sed "s/1670/${numVols}/g" | sed "s:/Users/xpsy1114/fsl:${fslDir}:g" | sed "s/sub-01/sub-$subjectTag/g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/output/sub-01_fmri_pt1_EVs:${EV_dir}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/glm_02.feat:${glmDir}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/nuisance_01/combined.txt:${nuisanceFile}:g" > $funcDir/sub-${subjectTag}_design_glm_${version}_pt${task_half}.fsf
        # cat ${funcDir}/sub-${subjectTag}_draft_GLM_${task_half}_${version}.fsf | sed "s/nuisance_01/nuisance_${task_half}/g" | sed "s/sub-01_01_evlist.txt/sub-${subjectTag}_${task_half}_evlist.txt/g" |sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/preproc_clean_01:${funcDir}/preproc_clean_${task_half}:g" | sed "s/filtered_func_data_clean/filtered_func_data/g" | sed "s/1246648320/${numVoxels}/g" |sed "s/1670/${numVols}/g" | sed "s:/Users/xpsy1114/fsl:${fslDir}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-${subjectTag}/func/EVs_06_pt${task_half}_press_and_loc:${EV_dir}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/glm_02.feat:${glmDir}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/nuisance_${task_half}/combined.txt:${nuisanceFile}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01:${derivDir}:g" | sed "s/sub-01/sub-$subjectTag/g" > $funcDir/sub-${subjectTag}_design_glm_${version}_pt${task_half}.fsf
        # this one is without pnm, but with the combined nuisance regressor.
        # cat ${funcDir}/sub-${subjectTag}_draft_GLM_0${task_half}_${version}.fsf | sed "s/nuisance_01/nuisance_0${task_half}/g" |sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/preproc_clean_01:${funcDir}/preproc_clean_0${task_half}:g" | sed "s/filtered_func_data_clean/filtered_func_data/g" | sed "s/1246648320/${numVoxels}/g" |sed "s/1670/${numVols}/g" | sed "s:/Users/xpsy1114/fsl:${fslDir}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-${subjectTag}/func/EVs_06_pt0${task_half}_press_and_loc:${EV_dir}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/glm_02.feat:${glmDir}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/nuisance_0${task_half}/combined.txt:${nuisanceFile}:g" | sed | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01:${derivDir}:g" | sed "s/sub-01/sub-$subjectTag/g" > $funcDir/sub-${subjectTag}_design_glm_${version}_pt${task_half}.fsf
        cat ${funcDir}/sub-${subjectTag}_draft_GLM_0${task_half}_${version}.fsf | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01:${derivDir}:g" | sed "s:/func/preproc_clean_01:/func/preproc_clean_0${task_half}:g" | sed "s:/func/nuisance_01/combined.txt:${nuisanceFile}:g" | sed "s/filtered_func_data_clean/filtered_func_data/g" | sed "s/1246648320/${numVoxels}/g" | sed "s/1670/${numVols}/g" | sed "s:/Users/xpsy1114/fsl:${fslDir}:g" | sed "s:/func/glm_02.feat:/func/glm_${version}_pt0${task_half}.feat:g" | sed "s:/motion/sub-01_1_evlist.txt:/motion/sub-${subjectTag}_${task_half}_evlist.txt:g" | sed "s/sub-01/sub-$subjectTag/g" > $funcDir/sub-${subjectTag}_design_glm_${version}_pt${task_half}.fsf

        # cat ${funcDir}/sub-${subjectTag}_draft_GLM_0${task_half}_${version}.fsf | 
        # sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01:${derivDir}:g" |
        # sed "s:/func/preproc_clean_01:/func/preproc_clean_0${task_half}:g" |
        # sed "s:/func/nuisance_01/combined.txt:${nuisanceFile}:g" | 
        # sed "s/filtered_func_data_clean/filtered_func_data/g" | 
        # sed "s/1246648320/${numVoxels}/g" | 
        # sed "s/1670/${numVols}/g" | 
        # sed "s:/Users/xpsy1114/fsl:${fslDir}:g" | 
        # sed "s:/func/glm_02.feat:/func/glm_${version}_pt0${task_half}.feat:g" 
        # sed "s:/motion/sub-01_1_evlist.txt:/motion/sub-${subjectTag}_${task_half}_evlist.txt:g"
        # sed "s/sub-01/sub-$subjectTag/g" > $funcDir/sub-${subjectTag}_design_glm_${version}_pt${task_half}.fsf
        

        echo The .fsf file was successfully created. Now starting FEAT!
        fsl_sub -q long.q -T 360 -R 30 feat $funcDir/sub-${subjectTag}_design_glm_${version}_pt${task_half}.fsf

    done
done