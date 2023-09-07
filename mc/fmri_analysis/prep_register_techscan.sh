#subjects=("01" "02")
#folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv")

# if only for the 7T stuff, use these ones:
#subjects=("02")
#folder_names=("run1_T7_AB_rep" "run2_T7_2mm_3iPat_2MB" "run3_T7_15mm_3iPat_3MB" "run4_T7_15mm_2iPat_3MB" "run5_T7_superfast" "run6_T7_12mm_3iPat_3MB")

subjects="03"
folder_names=("run1_3Trep_7T2" "run2_25mm_fast_7T2" "run3_mb4_fast_7T2" "run4_mb3ipat3_7T2" "run5_ipat4_7T2")
scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'


# this is my loop
for folder in "${folder_names[@]}"; do
    for subjectTag in "${subjects[@]}"; do
        echo "now looking at folder: $scratchDir/fmri_pilot/sub-$subjectTag/$folder/beh"
        # Make subject folder in derivatives folder: folder where any non-raw data gets stored (-p: don't if already exists)
        mkdir -p $scratchDir/derivatives
        mkdir -p $scratchDir/derivatives/sub-$subjectTag
        mkdir -p $scratchDir/derivatives/sub-$subjectTag/$folder

        # Construct anatomy directory for derived file
        anatDir=$scratchDir/derivatives/sub-$subjectTag/$folder/anat
        # And create directory for derived anatomy files
        mkdir -p $anatDir


        # For the 7 T data, this needs to be bet $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1W.nii.gz $anatDir/sub-${subjectTag}_T1W_brain.nii.gz -R -f 0.05
        # for 7T part 2 do -B and -0.01
        # Brain-extract structural file (see http://fsl.fmrib.ox.ac.uk/fslcourse/graduate/lectures/practicals/intro2/)
        # also create a mask with -m flag
        echo Brain-extracting structural file $scratchDir/tech_scan/$subjectTag/$folder/anat/sub-${subjectTag}_T1W.nii.gz, saving in $anatDir
        # Do brain extraction with -R option to re-run bet multiple times and keep the best
        bet $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1W.nii.gz $anatDir/sub-${subjectTag}_T1W_brain.nii.gz -B -f 0.01
        # But also copy original structural file, including the head: non-linear registration looks for that file with the same filename as the beted file, minus _brain
        cp $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1W.nii.gz $anatDir/sub-${subjectTag}_T1W.nii.gz


        # Construct func directory for derived file
        funcDir=$scratchDir/derivatives/sub-$subjectTag/$folder/func
        # And create directory for derived anatomy files
        mkdir -p $funcDir
        # Copy original functional file, including the head
        cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii.gz $funcDir/sub-${subjectTag}_bold.nii.gz


        # Construct fieldmap directory for derived file
        fmapDir=$scratchDir/derivatives/sub-$subjectTag/$folder/fmap
        # And create directory for derived fieldmap files
        mkdir -p $fmapDir


        # Prepare fieldmap for registration (see http://fsl.fmrib.ox.ac.uk/fslcourse/graduate/lectures/practicals/registration/)
        echo Preparing fieldmap files from $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/, saving in $fmapDir
        # Select only first image of magnitude fieldmap (see https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide: two magnitude images (one for each echo time), pick the "best looking" one)
        fslroi $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii.gz $fmapDir/sub-${subjectTag}_magnitude2.nii.gz 0 1
        # Brain extract the magnitude fieldmap
        bet $fmapDir/sub-${subjectTag}_magnitude2.nii.gz $fmapDir/sub-${subjectTag}_magnitude2_brain.nii.gz
        # Create slice along z-axis for zero padding, with x and y dimensions equal to original image but z dimension of only 1 (reduce to one volume)
        fslroi $fmapDir/sub-${subjectTag}_magnitude2_brain.nii.gz $fmapDir/sub-${subjectTag}_magnitude2_brain_slice.nii.gz 0 -1 0 -1 0 1 0 -1
        # Make slice into zeros by thresholding with a really high number
        fslmaths $fmapDir/sub-${subjectTag}_magnitude2_brain_slice.nii.gz -thr 9999 $fmapDir/sub-${subjectTag}_magnitude2_brain_slice_zero.nii.gz
        # Add zero padding by merging zero slice and beted brain: if the beted brain touches the top edge of the image, you won't erode anything there
        fslmerge -z $fmapDir/sub-${subjectTag}_magnitude2_brain_padded.nii.gz $fmapDir/sub-${subjectTag}_magnitude2_brain.nii.gz $fmapDir/sub-${subjectTag}_magnitude2_brain_slice_zero.nii.gz
        # Erode image: shave off voxels near edge of brain, since phase difference is very noisy there
        fslmaths $fmapDir/sub-${subjectTag}_magnitude2_brain_padded.nii.gz -ero $fmapDir/sub-${subjectTag}_magnitude2_brain_padded_ero.nii.gz
        # Find out the original size along the z dimension using fslinfo and bash magic (output fslinfo, find dim3, only keep the number that comes after dim3)
        #origSize=$(echo $(echo $(fslinfo $fmapDir/sub-${subjectTag}_magnitude1_brain.nii.gz) | grep -o -E ' dim3 [0-9]+' | sed 's/ dim3 //'))
        origSize=$(fslval $fmapDir/sub-${subjectTag}_magnitude2_brain.nii.gz dim3)
        # And remove the added zero padding slice from the eroded brain so its size matches the phasediff fieldmap
        fslroi $fmapDir/sub-${subjectTag}_magnitude2_brain_padded_ero.nii.gz $fmapDir/sub-${subjectTag}_magnitude2_brain_ero.nii.gz 0 -1 0 -1 0 $origSize 0 -1
        # Then prepare fieldmap from phase image, magnitude image, and difference in echo times between the two magnitude images
        fsl_prepare_fieldmap SIEMENS $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii.gz $fmapDir/sub-${subjectTag}_magnitude2_brain_ero.nii.gz $fmapDir/sub-${subjectTag}_fieldmap.nii.gz 2.46
    done
done