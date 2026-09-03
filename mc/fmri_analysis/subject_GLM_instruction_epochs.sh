#!/bin/sh
# Run subject-level GLMs for the instruction-phase epochs.
# submit like bash subject_GLM_instruction_epochs.sh
# requires EV directories with EVs and a subject fsf file before (made by
# scripts/create_EVs_instruction_period.py), the filtered_func dataset, and the
# nuisance regs.
#
# This is subject_GLM_RDM_conds.sh with one extra loop: instead of a single
# $version there is one GLM per instruction epoch, named after what it measures.
# create_EVs_instruction_period.py prints exactly this list when it runs.

version="instr"
glm_names="see-A-first see-B-first see-C-first see-D-first see-A-second see-B-second see-C-second see-D-second collapsed-first-instruction collapsed-second-instruction empty-screen"
echo this is version $version
echo these are the epochs: $glm_names

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
#fslDir="/opt/fmrib/fsl"
#export fslDir=~/scratch/fsl
#fslDir="~/scratch/fsl"
#export PATH=$fslDir/share/fsl/bin/:$PATH
#source $fslDir/etc/fslconf/fsl.sh
module load fsl

# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  analysisDir="/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
  fslDir="/Users/xpsy1114/fsl"
fi

echo Now entering the loop ....
# Show what ended up being the scratch dir
echo Scratch directory is $scratchDir

for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35; do
# for subjectTag in 02; do
    echo Subject tag and folder for the current run: $subjectTag

    # Construct directory for derived data
    derivDir=$scratchDir/derivatives/sub-$subjectTag

    # Construct the folder for function analysis for the current subject
    funcDir=$derivDir/func

    # do this twice, once for pt1 and once for pt2
    for task_half in 1 2 ; do

        # Get number of volumes from fslinfo and some bash tricks.
        # Outside the epoch loop: it is the same file for all 11 GLMs.
        numVols=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim4)
        # Display outcome: how many volumes are there?
        echo Found $numVols volumes in $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz

        # compute the number of voxels
        dim1=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim1)
        dim2=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim2)
        dim3=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim3)
        dim4=$(fslval $funcDir/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz dim4)
        numVoxels=$((dim1*dim2*dim3*dim4))
        echo Extracted number of voxels for this pt: $numVoxels

        nuisanceFile="/motion/nuisance_0${task_half}/combined.txt"

        # one GLM per instruction epoch
        for glm_name in $glm_names; do
            version_full="${version}_${glm_name}"
            echo "--- sub-${subjectTag} pt${task_half}: ${version_full}"

            glmDir="$funcDir/glm_${version_full}_pt0${task_half}.feat"

            # if [ -f $glmDir/stats/pe1.nii.gz ]; then
            #   echo Now skipping as for ${subjectTag} half ${task_half} ${version_full} the glm already ran
            #   continue
            # fi

            echo this will be the glm directory we are setting up: $glmDir

            EV_dir="$funcDir/EVs_${version_full}_pt0${task_half}"
            if [ -d "$EV_dir" ]; then
              echo "EV Folder exists, all good!"
            else
              echo this is supposed to be the EV folder: $EV_dir
              echo "ERROR: EV Folder does not exists!"
              continue
            fi

            draftFsf="${funcDir}/sub-${subjectTag}_draft_GLM_0${task_half}_${version_full}.fsf"
            if [ ! -f "$draftFsf" ]; then
              echo "ERROR: draft fsf does not exist: $draftFsf"
              continue
            fi

            designFsf="$funcDir/sub-${subjectTag}_design_glm_${version_full}_pt${task_half}.fsf"

            cat ${draftFsf} | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01:${derivDir}:g" | sed "s:/func/preproc_clean_01:/func/preproc_clean_0${task_half}:g" | sed "s:/func/nuisance_01/combined.txt:${nuisanceFile}:g" | sed "s/filtered_func_data_clean/filtered_func_data/g" | sed "s/1246648320/${numVoxels}/g" | sed "s/1670/${numVols}/g" | sed "s:/Users/xpsy1114/fsl:${fslDir}:g" | sed "s:/func/glm_02.feat:/func/glm_${version_full}_pt0${task_half}.feat:g" | sed "s:/motion/sub-01_1_evlist.txt:/motion/sub-${subjectTag}_${task_half}_evlist.txt:g" | sed "s/sub-01/sub-$subjectTag/g" > $designFsf

            echo The .fsf file was successfully created. Now starting FEAT!
            fsl_sub -q short feat $designFsf
        done
    done
done
