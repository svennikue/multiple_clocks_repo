# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"

glm_version="03-4"
RSA_version="03-1"


maskDir=$scratchDir/masks



for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 ; do
    # for every result file
    resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}/results-standard-space
    maskedDir=${resultDir}/masked_beta_small
    if [ -d $maskedDir ]; then
        rm -r $maskedDir
    fi
    mkdir $maskedDir

    echo now for subject $subjectTag and $maskedDir

    for file in ${resultDir}/*combo_split-clock_beta_std.nii.gz; do
        filename=$(basename "$file")
        #fslmaths $file -mas ${maskDir}/small_mpfc_with_ONE-REW_smaller_bin_thr-05.nii.gz $maskedDir/${filename}_mpfc_mask
        # fslmaths $file -mas ${maskDir}/MPFC_masked_bin_ONE-FUT-REW-rsa_03-1_glm_03-4_mul_TWO-FUT_mul_THREE-FUT-bin.nii.gz $maskedDir/${filename}_mpfc_mask
        
        fslmaths $file -mas ${maskDir}/mask_mpfc_tiny.nii.gz $maskedDir/${filename}_mpfc_mask
        fslmaths $file -mas ${maskDir}/small_ofc_with_ONE-REW_bin_thr-3.nii.gz $maskedDir/${filename}_ofc_mask
        #fslmaths $file -mas ${maskDir}/masked_bin_ONE-FUT-REW-rsa_03-1_glm_03-4.nii.gz $maskedDir/${filename}_mpfc_mask
        #fslmaths $file -mas ${maskDir}/masked_bin_OFC_ONE-FUT-REW-rsa_03-1_glm_03-4.nii.gz $maskedDir/${filename}_ofc_mask
    done
done
    

echo done!
