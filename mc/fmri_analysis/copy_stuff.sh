subjects=("01" "02")
folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv")
scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'
calpendoDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data/downloads'

for folder in "${folder_names[@]}"; do
    for subjectTag in "${subjects[@]}"; do
        cp $calpendoDir/sub-$subjectTag/$folder/func/* $scratchDir/tech_scan/sub-$subjectTag/$folder/func
        cp $calpendoDir/sub-$subjectTag/$folder/anat/* $scratchDir/tech_scan/sub-$subjectTag/$folder/anat
        cp $calpendoDir/sub-$subjectTag/$folder/fmap/* $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap
    done
done
