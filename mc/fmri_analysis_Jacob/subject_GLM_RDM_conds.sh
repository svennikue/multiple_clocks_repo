#!/bin/sh
# Run subject-level GLM on preprocessed functional data
#
# bash subject_GLM_RDM_conds.sh 01 04

# Command line argument 1/2: subject tag
subjects=$1
# Command line argument 2/2: GLM version
version=$2
echo Subject tag for this subject: $subjectTag and for GLM no $version

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
toolboxDir="/home/fs0/xpsy1114/scratch/analysis"
homeDir="/home/fs0/xpsy1114"
analysisDir="${scratchDir}/analysis"


task_halves=("01" "02")
# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  toolboxDir="/Users/xpsy1114/Documents/toolboxes"
  homeDir="/Users/xpsy1114/Documents/projects/multiple_clocks"
  analysisDir="${homeDir}/multiple_clocks_repo/mc/fmri_analysis"
fi

# Show what ended up being the home directory
echo Home directory is $homeDir
echo Scratch directory is $scratchDir


for subjectTag in "${subjects[@]}"; do
    # Command line argument 1/1: subject tag
    echo Subject tag for this subject: $subjectTag and for GLM no $version
    
    # Construct directory for derived data
    derivDir=$scratchDir/derivatives/sub-$subjectTag

    # Construct the folder for function analysis for the current subject
    funcDir=$scratchDir/derivatives/sub-$subjectTag/func

    # do this twice, once for pt1 and once for pt2

    # create a loop from here to the end (marked as well)
    for task_half in "${task_halves[@]}"; do
        glmDir="$funcDir/glm_${version}_pt${task_half}.feat" 
        # you don't need this as FEAT will create it.
        # if [ -d "$glmDir" ]; then
        #   echo "EV Folder exists, all good!"
        # else
        #   mkdir $glmDir
        # fi

        nuisanceFile=$derivDir/func/nuisance_$task_half/combined.txt

        # I DONT THINK I NEED TO COPY IT. I THINK IT DOES IT AUTOMATICALLY
        
        # If somehow the glm feat folder already exists: delete it
        # rm -rf $glmDir

        # Copy the folder that was used for preprocessing so it can be used for the glm
        # cp -r $funcDir/preproc_clean_${task_half}.feat $glmDir

        # Remove filtered_func_data.ica since it won't be used
        # rm -r $funcDir/glm_${version}_pt${task_half}.feat/filtered_func_data.ica

        # check if the EV dir exists, if not output an error.
        EV_dir="$funcDir/EVs_${version}_pt${task_half}"

        if [ -d "$EV_dir" ]; then
          echo "EV Folder exists, all good!"
        else
          echo "ERROR: EV Folder does not exist!"
        fi

        # I want to additionally create a mask that I will then use as a searchlight mask for each task_half.
        # note 06.12.2023 
        # I can delete this for the server since I have a really good mask with synthstrip already. If anything, take and rename.
        bet $funcDir/preproc_clean_${task_half}.feat/example_func.nii.gz mask_pt0${task_half}.nii.gz -m -f 0.3

        # Get number of volumes from fslinfo and some bash tricks
        numVols=$(fslval $funcDir/preproc_clean_${task_half}.feat/filtered_func_data_clean.nii.gz dim4)
        # Display outcome: how many volumes are there?
        echo Found $numVols volumes in $funcDir/preproc_clean_${task_half}.feat/filtered_func_data_clean.nii.gz


        # compute the number of voxels
        dim1=$(fslval $funcDir/preproc_clean_${task_half}.feat/filtered_func_data_clean.nii.gz dim1)
        dim2=$(fslval $funcDir/preproc_clean_${task_half}.feat/filtered_func_data_clean.nii.gz dim2)
        dim3=$(fslval $funcDir/preproc_clean_${task_half}.feat/filtered_func_data_clean.nii.gz dim3)
        dim4=$(fslval $funcDir/preproc_clean_${task_half}.feat/filtered_func_data_clean.nii.gz dim4)
        numVoxels=$((dim1*dim2*dim3*dim4))
        echo Extracted number of voxels for first pt: $numVoxels

        # ADJUST THIS!!!!!
        # take this???
        # cat $analysisDir/templates/preproc_clean.fsf | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/preproc_clean_01:${funcDir}/preproc_clean_01:g" | sed "s/1246648320/${numVoxels}/g" |sed "s/1670/${numVols}/g" | sed "s:/Users/xpsy1114/fsl:${fslDir}:g" | sed "s/sub-01/sub-$subjectTag/g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data:${scratchDir}:g" > $funcDir/sub-${subjectTag}_1_design_preproc_clean.fsf


        # Take preprocessing template, replace subject id and number of volumes with current values, update output directory, update scratch directory, and save to new file
        # cat $homeDir/Analysis/Templates/glm_subject_${version}_pt${task_half}.fsf | sed "s/s01id01/$subjectTag/g" | sed "s/glm.feat/glm_${version}_pt${task_half}.feat/g" | sed "s:/EVs/:/EVs_${version}_pt${task_half}/:g" | sed "s/1670/${numVols}/g" | sed "s:/Volumes/Scratch_jacobb:${scratchDir}:g" | sed "s:/home/fs0/jacobb/scratch:${scratchDir}:g" > $funcDir/sub-${subjectTag}_design_glm_${version}.fsf

        # cat ${analysisDir}/templates/loc_press.fsf | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/preproc_clean_01:${funcDir}/preproc_clean_${task_half}:g" | sed "s/1246648320/${numVoxels}/g" |sed "s/1670/${numVols}/g" | sed "s:/Users/xpsy1114/fsl:${fslDir}:g" | sed "s/sub-01/sub-$subjectTag/g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/output/sub-01_fmri_pt1_EVs:${EV_dir}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/glm_02.feat:${glmDir}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/nuisance_01/combined.txt:${nuisanceFile}:g" > $funcDir/sub-${subjectTag}_design_glm_${version}_pt${task_half}.fsf
        cat ${funcDir}/sub-${subjectTag}_my_RDM_GLM_${task_half}_${version}.fsf | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/preproc_clean_01:${funcDir}/preproc_clean_${task_half}:g" | sed "s/1246648320/${numVoxels}/g" |sed "s/1670/${numVols}/g" | sed "s:/Users/xpsy1114/fsl:${fslDir}:g" | sed "s/sub-01/sub-$subjectTag/g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/output/sub-01_fmri_pt1_EVs:${EV_dir}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/glm_02.feat:${glmDir}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/nuisance_01/combined.txt:${nuisanceFile}:g" > $funcDir/sub-${subjectTag}_design_glm_${version}_pt${task_half}.fsf
        
        #cat ${analysisDir}/templates/my_RDM_GLM_v2.fsf | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/preproc_clean_01:${funcDir}/preproc_clean_${task_half}:g" | sed "s/1246648320/${numVoxels}/g" |sed "s/1670/${numVols}/g" | sed "s:/Users/xpsy1114/fsl:${fslDir}:g" | sed "s/sub-01/sub-$subjectTag/g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/output/sub-01_fmri_pt1_EVs:${EV_dir}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/glm_02.feat:${glmDir}:g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/nuisance_01/combined.txt:${nuisanceFile}:g" > $funcDir/sub-${subjectTag}_design_glm_${version}_pt${task_half}.fsf
        
        echo The .fsf file was successfully created. Now starting FEAT!

        # Finally: run feat with these parameters
        feat $funcDir/sub-${subjectTag}_design_glm_${version}_pt${task_half}.fsf

    done
done