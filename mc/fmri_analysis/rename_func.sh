subjects=("01" "02")
folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv")
scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'



for subjectTag in "${subjects[@]}"; do
    for folder in "${folder_names[@]}"; do
        if [ "$folder" = "AB_sequ_plus30" ]; then
            if [ "$subjectTag" = "01" ]; then
                mv $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_013*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                mv $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_013*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.json
            fi
            if [ "$subjectTag" = "02" ]; then
                mv $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_011*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                mv $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_011*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.json
            fi
        fi
        if [ "$folder" = "AB_sequ_min30" ]; then
            if [ "$subjectTag" = "01" ]; then
                mv $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_05*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                mv $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_05*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.json
            fi
            if [ "$subjectTag" = "02" ]; then
                mv $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_03*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                mv $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_03*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.json
            fi
        fi
        if [ "$folder" = "AB_sequ_transv" ]; then
            if [ "$subjectTag" = "01" ]; then
                mv $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_09*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                mv $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_09*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.json
            fi
            if [ "$subjectTag" = "02" ]; then
                mv $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_07*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                mv $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_07*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.json
            fi
        fi
        if [ "$folder" = "Leip_sequ_transv" ] || [ "$folder" = "Leip_sequ_min30" ] || [ "$folder" = "Leip_sequ_plus30" ]; then
                mv $scratchDir/tech_scan/sub-$subjectTag/$folder/func/*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
                mv $scratchDir/tech_scan/sub-$subjectTag/$folder/func/*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.json
        fi
    done
done
        

        




# rename the bold images
# sub-01
    # AB_min30
    # we thought the task didnt work the first time, so we started again. take the longest file.
    #images_05*.nii
    #images_05*.json
    # AB_plus30
    #images_013*.nii
    #images_013*.json
    # AB_transv
    #images_09*.nii
    #images_09*.json
    # for the Leipzig sequences, there is only one .json and one .nii file, respectively
    #*.nii
    #*.json
# sub-02
    # AB_min30
    #images_03*.nii
    #images_03*.json
    # AB_plus30
    #images_011*.nii
    #images_011*.json
    # AB_transv
    #images_07*.nii
    #images_07*.json
    # for the Leipzig sequences, there is only one .json and one .nii file, respectively
    #*.nii
    #*.json