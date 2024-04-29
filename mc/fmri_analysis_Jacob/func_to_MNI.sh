# subjects=("02")
# subjects=("01" "02")
subjects="03"
folder_names=("run1_3Trep_7T2" "run2_25mm_fast_7T2" "run3_mb4_fast_7T2" "run4_mb3ipat3_7T2" "run5_ipat4_7T2")

#folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv" "run1_T7_AB_rep" "run2_T7_2mm_3iPat_2MB" "run3_T7_15mm_3iPat_3MB" "run4_T7_15mm_2iPat_3MB" "run5_T7_superfast" "run6_T7_12mm_3iPat_3MB")
# don't do this for 7T yet bc I'm unsure about the fieldmaps
#folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv")
# folder_names=("run1_T7_AB_rep" "run2_T7_2mm_3iPat_2MB" "run3_T7_15mm_3iPat_3MB" "run4_T7_15mm_2iPat_3MB" "run5_T7_superfast" "run6_T7_12mm_3iPat_3MB")
scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'
derivDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data/derivatives'
FSLdir='/Users/xpsy1114/fsl'

#for folder in "${folder_names[@]}"; do
#    for subjectTag in "${subjects[@]}"; do
#        preprocDir=$scratchDir/derivatives/sub-$subjectTag/$folder/func/preproc.feat
#        if (( "$folder" = "AB_sequ_plus30" )) || (( "$subjectTag" = "01" )); then
#            continue
#        elif (( "$folder" = "Leip_sequ_transv" )) || (( "$subjectTag" = "01" )); then
#            continue
#            applywarp --ref=${FSLDIR}/data/standard/MNI152_T1_2mm --in=$preprocDic/filtered_func.nii.gz --warp=$preprocDic/reg/example_func2standard_warp.nii.gz --out=$preprocDic/filtered_func_standard.nii.gz
#        else
#        fi
#    done 
#done

for folder in "${folder_names[@]}"; do
    for subjectTag in "${subjects[@]}"; do
        preprocDir=$scratchDir/derivatives/sub-$subjectTag/$folder/func/preproc.feat
        applywarp --ref=${FSLdir}/data/standard/MNI152_T1_2mm --in=$preprocDir/filtered_func_data.nii.gz --warp=$preprocDir/reg/example_func2standard_warp.nii.gz --out=$preprocDir/filtered_func_data_standard.nii.gz
    done 
done
