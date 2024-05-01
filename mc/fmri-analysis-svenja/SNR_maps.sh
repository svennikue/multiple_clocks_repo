# creating tSNR maps out of files of choice
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
echo Scratch directory is $scratchDir

# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do
# for subjectTag in 03 04 07 08 09; do
#for subjectTag in 03 04 07 08 09 10 11 12 14 15 16 17 20 21 23 24; do
#for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 ; do
for subjectTag in 02; do
    funcDir=$scratchDir/derivatives/sub-$subjectTag/func
    for task_half in 1 2 ; do
        # create a SNR map for the raw files
        # this is the mean over the standard deviation across the -t dimension (dim4)
        fslmaths $funcDir/sub-${subjectTag}_${task_half}_bold.nii.gz -Tmean $funcDir/sub-${subjectTag}_${task_half}_bold_mean.nii.gz
        fslmaths $funcDir/sub-${subjectTag}_${task_half}_bold.nii.gz -Tstd $funcDir/sub-${subjectTag}_${task_half}_bold_std.nii.gz
        fslmaths $funcDir/sub-${subjectTag}_${task_half}_bold_mean.nii.gz -div $funcDir/sub-${subjectTag}_${task_half}_bold_std.nii.gz $funcDir/sub-${subjectTag}_${task_half}_bold_SNR_map.nii.gz

        preprocDir=$scratchDir/derivatives/sub-$subjectTag/func/preproc_clean_0${task_half}.feat
        # I can at first create a SNR map for the whole brain.
        # this is the mean over the standard deviation across the -t dimension (dim4)
        fslmaths $preprocDir/filtered_func_data.nii.gz -Tmean $preprocDir/sub-${subjectTag}_bold_mean.nii.gz
        fslmaths $preprocDir/filtered_func_data.nii.gz -Tstd $preprocDir/sub-${subjectTag}_bold_std.nii.gz
        fslmaths $preprocDir/sub-${subjectTag}_bold_mean.nii.gz -div $preprocDir/sub-${subjectTag}_bold_std.nii.gz $preprocDir/sub-${subjectTag}_SNR_map.nii.gz
    done
done
