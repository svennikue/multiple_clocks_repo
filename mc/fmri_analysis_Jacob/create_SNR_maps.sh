#subjects=("02")
subjects=("01" "02")
#folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv" "run1_T7_AB_rep" "run2_T7_2mm_3iPat_2MB" "run3_T7_15mm_3iPat_3MB" "run4_T7_15mm_2iPat_3MB" "run5_T7_superfast" "run6_T7_12mm_3iPat_3MB")
# don't do this for 7T yet bc I'm unsure about the fieldmaps
folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv")
#folder_names=("run1_T7_AB_rep" "run2_T7_2mm_3iPat_2MB" "run3_T7_15mm_3iPat_3MB" "run4_T7_15mm_2iPat_3MB" "run5_T7_superfast" "run6_T7_12mm_3iPat_3MB")
scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'
derivDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives'

for folder in "${folder_names[@]}"; do
    for subjectTag in "${subjects[@]}"; do
        preprocDir=$scratchDir/derivatives/sub-$subjectTag/$folder/func/preproc.feat
        # I can at first create a SNR map for the whole brain.
        # this is the mean over the standard deviation across the -t dimension (dim4)
        fslmaths $preprocDir/filtered_func_data_standard.nii.gz -Tmean $derivDir/sub-${subjectTag}/$folder/func/sub-${subjectTag}_mean_bold.nii.gz
        fslmaths $preprocDir/filtered_func_data_standard.nii.gz -Tstd $derivDir/sub-${subjectTag}/$folder/func/sub-${subjectTag}_std_bold.nii.gz
        fslmaths $derivDir/sub-${subjectTag}/$folder/func/sub-${subjectTag}_mean_bold.nii.gz -div $derivDir/sub-${subjectTag}/$folder/func/sub-${subjectTag}_std_bold.nii.gz $derivDir/sub-${subjectTag}/$folder/func/sub-${subjectTag}_SNR_map.nii.gz
    done
done
