subjects=("01" "02")
folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv")
scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'


for folder in "${folder_names[@]}"; do
    for subjectTag in "${subjects[@]}"; do
        # this is true for the Leipzig sequence
        if [ "$subjectTag" = "01" ]; then
            # these numbers are for sub-01
            # rename the second structural which is bias corrected (check with fsl eyes for uniform grey colour)
            mv $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/images_017*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.json
            mv $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/images_017*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
            gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
            rm $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
        fi
        if [ "$subjectTag" = "02" ]; then
            # these numbers are for sub-02
            # rename the second structural which is bias corrected (check with fsl eyes for uniform grey colour)
            mv $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/images_015*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.json
            mv $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/images_015*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
            gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
            rm $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
        fi
    done
done