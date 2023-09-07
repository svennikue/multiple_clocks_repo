# for example, I want to use this loop to copy all behavioural and motion data.
subjects=("01" "02")
folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv")
scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'

for folder in "${folder_names[@]}"; do
    for subjectTag in "${subjects[@]}"; do
        echo "now looking at folder: $scratchDir/fmri_pilot/sub-$subjectTag/$folder/beh"
        # Make subject folder in derivatives folder: folder where any non-raw data gets stored (-p: don't if already exists)
        mkdir -p $scratchDir/tech_scan/derivatives
        mkdir -p $scratchDir/tech_scan/derivatives/sub-$subjectTag
        mkdir -p $scratchDir/tech_scan/sub-$subjectTag/$folder/motion
        mkdir -p $scratchDir/tech_scan/sub-$subjectTag/$folder/beh
        cp -r -n $scratchDir/fmri_pilot/sub-$subjectTag/$folder/beh/* $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/
        mv $scratchDir/fmri_pilot/sub-$subjectTag/$folder/motion/*.acq $scratchDir/tech_scan/sub-$subjectTag/$folder/motion/sub-${subjectTag}_physio.acq
        mv $scratchDir/fmri_pilot/sub-$subjectTag/$folder/motion/*.txt $scratchDir/tech_scan/sub-$subjectTag/$folder/motion/sub-${subjectTag}_physio.txt

    done
done