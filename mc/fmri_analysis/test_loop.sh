# test how bash loops work.

subjects=("01" "02")
folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv")


for folder in "${folder_names[@]}"; do
    for subjectTag in "${subjects[@]}"; do
        echo "now looking at folder: $scratchDir/fmri_pilot/sub-$subjectTag/$folder/beh"
    done
done
