# this is to create the registration transformation to then later do the proper registration.
# it should be quite short as it will use a single volume rather than the whole funtional file.

# first run this script, and only later the longer one!


# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
fslDir="/opt/fmrib/fsl"

# Rik's trick to make the 7T registration nice
source $analysisDir/setup_test_feat.sh
# put the config file marieke sent somehwere, point at it in the .fsf (seem Riks email)
# change folder of set fmri(regstandard) "/Users/xpsy1114/fsl/data/standard/MNI152_T1_2mm_brain"
# to a folder that doesn thave the same one in the skull


echo Now entering the loop ....
echo fslDir ist $fslDir

# 16 17 18 19 20 21 22 23 24 25 26 27

for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34; do
    echo Subject tag and folder for the current run: $subjectTag
    # Construct directory for derived data
    derivDir=$scratchDir/derivatives/sub-$subjectTag
    # Construct func directory for derived file
    funcDir=$derivDir/func
    pilotDir=$scratchDir/pilot/sub-$subjectTag/func 

    # do this twice, once for pt1 and once for pt2
    # Get number of volumes from fslinfo and some bash tricks
    numVols=$(fslval $pilotDir/sub-${subjectTag}_1_vol_1_bold.nii.gz dim4)
    echo Extracted number of volumes for first pt: $numVols

    # compute the number of voxels
    dim1=$(fslval $pilotDir/sub-${subjectTag}_1_vol_1_bold.nii.gz dim1)
    dim2=$(fslval $pilotDir/sub-${subjectTag}_1_vol_1_bold.nii.gz dim2)
    dim3=$(fslval $pilotDir/sub-${subjectTag}_1_vol_1_bold.nii.gz dim3)
    dim4=$(fslval $pilotDir/sub-${subjectTag}_1_vol_1_bold.nii.gz dim4)
    numVoxels=$((dim1*dim2*dim3*dim4))
    echo Extracted number of voxels for first pt: $numVoxels

    # really important to here take the respective .fsf file of the sequence I prepared manually! Different TE, TR, echo spacing
    cat $analysisDir/templates/pre-preproc.fsf | sed "s:/vols/Scratch/xpsy1114/data/derivatives/sub-02:${derivDir}:g" | sed "s/1056291840/${numVoxels}/g" |sed "s/1415/${numVols}/g" | sed "s/sub-02/sub-$subjectTag/g" | sed "s:/vols/Scratch/xpsy1114/data:${scratchDir}:g" > $funcDir/sub-${subjectTag}_1_design_pre_preproc.fsf
    echo Now starting first feat preproc one
    
    # test if the preproc folders already exist. If they do, delete, as these folders require a lot of space.
    if test -d ${funcDir}/preproc_clean_01.feat; then
      rm -r ${funcDir}/preproc_clean_01.feat
    fi
    # fsl_sub -q long.q -T 360 -R 30 feat $funcDir/sub-${subjectTag}_1_design_preproc_clean.fsf
    fsl_sub -q long.q -T 360 -R 30 feat $funcDir/sub-${subjectTag}_1_design_pre_preproc.fsf

    numVols=$(fslval $pilotDir/sub-${subjectTag}_1_vol_1_bold.nii.gz dim4)
    echo Extracted number of volumes for second pt: $numVols

    # compute the number of voxels
    dim1=$(fslval $pilotDir/sub-${subjectTag}_1_vol_1_bold.nii.gz dim1)
    dim2=$(fslval $pilotDir/sub-${subjectTag}_1_vol_1_bold.nii.gz dim2)
    dim3=$(fslval $pilotDir/sub-${subjectTag}_1_vol_1_bold.nii.gz dim3)
    dim4=$(fslval $pilotDir/sub-${subjectTag}_1_vol_1_bold.nii.gz dim4)
    numVoxels=$((dim1*dim2*dim3*dim4))
    echo Extracted number of voxels for second pt: $numVoxels

    # note I am changing all to the second task half, except for the reference image. this is what the file shall be registered to.
    # cat $analysisDir/templates/try_to_fixit.fsf | sed "s/sub-01_1_bold/sub-${subjectTag}_2_bold/g"| sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives/sub-01/func/preproc_clean_01:${funcDir}/preproc_clean_02:g" | sed "s/1246648320/${numVoxels}/g" |sed "s/1670/${numVols}/g" | sed "s:/Users/xpsy1114/fsl:${fslDir}:g" | sed "s/sub-01/sub-$subjectTag/g" | sed "s:/Users/xpsy1114/Documents/projects/multiple_clocks/data:${scratchDir}:g" > $funcDir/sub-${subjectTag}_2_design_preproc_clean.fsf
    cat $analysisDir/templates/pre-preproc.fsf | sed "s/preproc_clean_01.feat/preproc_clean_02.feat/g" | sed "s:/vols/Scratch/xpsy1114/data/derivatives/sub-02:${derivDir}:g" | sed "s/1056291840/${numVoxels}/g" |sed "s/1415/${numVols}/g" | sed "s/sub-02/sub-$subjectTag/g" | sed "s:/vols/Scratch/xpsy1114/data:${scratchDir}:g" > $funcDir/sub-${subjectTag}_2_design_pre_preproc.fsf

    # test if the preproc folders already exist. If they do, delete, as these folders require a lot of space.
    if test -d ${funcDir}/preproc_clean_02.feat; then
      rm -r ${funcDir}/preproc_clean_02.feat
    fi
    echo Now starting second feat preproc two
    # fsl_sub -q long.q -T 360 -R 30 feat $funcDir/sub-${subjectTag}_2_design_preproc_clean.fsf
    fsl_sub -q long.q -T 360 -R 30 feat $funcDir/sub-${subjectTag}_2_design_pre_preproc.fsf
done

