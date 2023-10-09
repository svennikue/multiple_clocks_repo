#this script runs the preprocessing from FEAT based on an .fsf file
#for the first pilot scan 

scratchDir="/home/fs0/xpsy1114/scratch/data"
subjects="01"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
echo Now entering the loop ....

for subjectTag in "${subjects[@]}"; do
    echo Subject tag and folder for the current run: $subjectTag
    # Construct directory for derived data
    derivDir=$scratchDir/derivatives/sub-$subjectTag
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
    cat $analysisDir/templates/preproc.fsf | sed "s:/vols/Scratch/xpsy1114/data/derivatives/sub-01/func:${funcDir}/preproc_01:g" | sed "s/1246648320/${numVoxels}/g" |sed "s/1670/${numVols}/g" | sed "s/sub-01/sub-$subjectTag/g" | sed "s:/vols/Scratch/xpsy1114:${scratchDir}:g" > $funcDir/sub-${subjectTag}_1_design_preproc.fsf
    echo Now starting first feat preproc
    feat $funcDir/sub-${subjectTag}_1_design_preproc.fsf

    numVols=$(fslval $funcDir/sub-${subjectTag}_2_bold.nii.gz dim4)
    echo Extracted number of volumes for second pt: $numVols

    # compute the number of voxels
    dim1=$(fslval $funcDir/sub-${subjectTag}_2_bold.nii.gz dim1)
    dim2=$(fslval $funcDir/sub-${subjectTag}_2_bold.nii.gz dim2)
    dim3=$(fslval $funcDir/sub-${subjectTag}_2_bold.nii.gz dim3)
    dim4=$(fslval $funcDir/sub-${subjectTag}_2_bold.nii.gz dim4)
    numVoxels=$((dim1*dim2*dim3*dim4))
    echo Extracted number of voxels for second pt: $numVoxels

    # really important to here take the respective .fsf file of the sequence I prepared manually! Different TE, TR, echo spacing
    cat $analysisDir/templates/preproc.fsf | sed "s/1246648320/${numVoxels}/g" |sed "s/1670/${numVols}/g" | sed "s/sub-01/sub-$subjectTag/g" | sed "s:/vols/Scratch/xpsy1114/data/derivatives/sub-01/func:${funcDir}/preproc_02:g" |  sed "s:/vols/Scratch/xpsy1114:${scratchDir}:g" > $funcDir/sub-${subjectTag}_2_design_preproc.fsf
    echo Now starting second feat preproc
    feat $funcDir/sub-${subjectTag}_2_design_preproc.fsf

done

