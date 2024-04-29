#subjects=("02")
subjects=("01" "02")
#folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv" "run1_T7_AB_rep" "run2_T7_2mm_3iPat_2MB" "run3_T7_15mm_3iPat_3MB" "run4_T7_15mm_2iPat_3MB" "run5_T7_superfast" "run6_T7_12mm_3iPat_3MB")
# don't do this for 7T yet bc I'm unsure about the fieldmaps
# folder_names=("Leip_sequ_transv")
folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv")
# folder_names=("run1_T7_AB_rep" "run2_T7_2mm_3iPat_2MB" "run3_T7_15mm_3iPat_3MB" "run4_T7_15mm_2iPat_3MB" "run5_T7_superfast" "run6_T7_12mm_3iPat_3MB")
scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'
derivDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives'
maskNames=("Left_Hippocampus" "Right_Hippocampus" "Frontal_Medial_Cortex" "entorhinal_cortex_L" "entorhinal_cortex_R")
maskDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data/fsl_eyes'



# first prepare the masks, if dont exist
#for ROI in "${maskNames[@]}"; do
#    fslmaths $maskDir/*${ROI}* -thr 10 -bin $maskDir/${ROI}_thr_mask.nii.gz
#    fslmaths $maskDir/${ROI}_thr_mask.nii.gz -subsamp2 -bin $maskDir/${ROI}_thr_mask_2mm.nii.gz
#done


# then per standard file, multiply with the correct mask
for folder in "${folder_names[@]}"; do
    for subjectTag in "${subjects[@]}"; do
        for ROI in "${maskNames[@]}"; do
            preprocDir=$scratchDir/derivatives/sub-$subjectTag/$folder/func/preproc.feat
            fslroi $preprocDir/filtered_func_data_standard.nii.gz $preprocDir/filtered_func_data_standard_100.nii.gz 0 100    
            fslmaths $preprocDir/filtered_func_data_standard_100.nii.gz -mul $maskDir/${ROI}_thr_mask_2mm.nii.gz $derivDir/sub-${subjectTag}/$folder/func/sub-${subjectTag}_${ROI}_bold.nii.gz
        done
    done
done


for folder in "${folder_names[@]}"; do
    for subjectTag in "${subjects[@]}"; do
        for ROI in "${maskNames[@]}"; do
            preprocDir=$scratchDir/derivatives/sub-$subjectTag/$folder/func/preproc.feat
            # I can at first create a SNR map for the whole brain.
            # this is the mean over the standard deviation across the -t dimension (dim4)
            fslmaths $derivDir/sub-${subjectTag}/$folder/func/sub-${subjectTag}_${ROI}_bold.nii.gz -Tmean $derivDir/sub-${subjectTag}/$folder/func/sub-${subjectTag}_${ROI}_mean_bold.nii.gz
            fslmaths $derivDir/sub-${subjectTag}/$folder/func/sub-${subjectTag}_${ROI}_bold.nii.gz -Tstd $derivDir/sub-${subjectTag}/$folder/func/sub-${subjectTag}_${ROI}_std_bold.nii.gz
            fslmaths $derivDir/sub-${subjectTag}/$folder/func/sub-${subjectTag}_${ROI}_mean_bold.nii.gz -div $derivDir/sub-${subjectTag}/$folder/func/sub-${subjectTag}_${ROI}_std_bold.nii.gz $derivDir/sub-${subjectTag}/$folder/func/sub-${subjectTag}_${ROI}_SNR_map.nii.gz
        done
    done
done


fslstats sub-01_entorhinal_cortex_L_SNR_map.nii.gz -M
