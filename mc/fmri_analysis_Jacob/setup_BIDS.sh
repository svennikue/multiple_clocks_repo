# # subjects=("01" "02")
# # folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv" "run1_T7_AB_rep" "run2_T7_2mm_3iPat_2MB" "run3_T7_15mm_3iPat_3MB" "run4_T7_15mm_2iPat_3MB" "run5_T7_superfast" "run6_T7_12mm_3iPat_3MB")
# # scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'
# # path_to_scans=("/Users/xpsy1114/Documents/projects/multiple_clocks/data/downloads/Alon_1" "/Users/xpsy1114/Documents/projects/multiple_clocks/data/downloads/Alon_2" "/Users/xpsy1114/Documents/projects/multiple_clocks/data/downloads/Jacob_day1" "/Users/xpsy1114/Documents/projects/multiple_clocks/data/downloads/Jacob_day2")

# this is my loop
# for folder in "${folder_names[@]}"; do
#     for subjectTag in "${subjects[@]}"; do
#         # set up BIDS folder structure for raw data.
#         mkdir -p $scratchDir/tech_scan
#         mkdir -p $scratchDir/tech_scan/sub-$subjectTag
#         mkdir -p $scratchDir/tech_scan/sub-$subjectTag/$folder

#         # Construct structural directory for raw file
#         anatDir=$scratchDir/tech_scan/sub-$subjectTag/$folder/anat
#         # And create directory for derived anatomy files
#         mkdir -p $anatDir

#         # Construct functional directory for raw file
#         funcDir=$scratchDir/tech_scan/sub-$subjectTag/$folder/func
#         # And create directory for derived anatomy files
#         mkdir -p $funcDir

#         # Construct behavioural directory for raw file
#         behDir=$scratchDir/tech_scan/sub-$subjectTag/$folder/beh
#         # And create directory for derived anatomy files
#         mkdir -p $behDir

#         # Construct fieldmap directory for raw file
#         fmapDir=$scratchDir/tech_scan/sub-$subjectTag/$folder/fmap
#         # And create directory for derived anatomy files
#         mkdir -p $fmapDir

#         # Construct physiology directory for raw file
#         motionDir=$scratchDir/tech_scan/sub-$subjectTag/$folder/motion
#         # And create directory for derived anatomy files
#         mkdir -p $motionDir
#         for scan_session in "${path_to_scans[@]}"; do
#             #only copy the respective correct thing
#             if [ "$scan_session" = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/downloads/F3T_2013_40_857" ] && []; then
#     done
# done
# scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'
# path="/Users/xpsy1114/Documents/projects/multiple_clocks/data/downloads/Alon_1"
# subjectTag="02"
# folder_names=("AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv")
# for folder in "${folder_names[@]}"; do
#     #anat
#     cp $path/images_015*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/
#     cp $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/images_015*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
#     gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
#     rm $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
#     if [ "$folder" = "AB_sequ_min30" ]; then
#         #func
#         cp $path/images_03*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_03*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii

#         #fmap
#         cp $path/images_04*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_04*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#         cp $path/images_04*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_04*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii

#         cp $path/images_04*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_04*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#         cp $path/images_04*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_04*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii

#         cp $path/images_05*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_05*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#         cp $path/images_05*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_05*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#     fi
#     if [ "$folder" = "AB_sequ_transv" ]; then
#         #func
#         cp $path/images_07*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_07*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii

#         #fmap
#         cp $path/images_08*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_08*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#         cp $path/images_08*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_08*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii

#         cp $path/images_08*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_08*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#         cp $path/images_08*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_08*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii

#         cp $path/images_09*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_09*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#         cp $path/images_09*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_09*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#     fi
#     if [ "$folder" = "AB_sequ_plus30" ]; then
#         #func
#         cp $path/images_011*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_011*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii

#         #fmap
#         cp $path/images_012*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_012*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#         cp $path/images_012*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_012*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii

#         cp $path/images_012*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_012*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#         cp $path/images_012*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_012*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii

#         cp $path/images_013*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_013*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#         cp $path/images_013*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_013*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#     fi
# done


# path="/Users/xpsy1114/Documents/projects/multiple_clocks/data/downloads/Alon_2"
# subjectTag="02"
# folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30")
# for folder in "${folder_names[@]}"; do
#     #anat
#     cp /Users/xpsy1114/Documents/projects/multiple_clocks/data/downloads/Alon_1/images_015*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/
#     cp $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/images_015*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
#     gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
#     rm $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
#     if [ "$folder" = "Leip_sequ_min30" ]; then
#         #func
#         cp $path/images_02*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_02*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii

#         #fmap
#         cp $path/images_03*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_03*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#         cp $path/images_03*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_03*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii

#         cp $path/images_03*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_03*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#         cp $path/images_03*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_03*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii

#         cp $path/images_04*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_04*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#         cp $path/images_04*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_04*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#     fi
#     if [ "$folder" = "Leip_sequ_transv" ]; then
#         #func
#         cp $path/images_05*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_05*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii

#         #fmap
#         cp $path/images_06*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_06*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#         cp $path/images_06*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_06*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii

#         cp $path/images_06*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_06*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#         cp $path/images_06*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_06*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii

#         cp $path/images_07*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_07*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#         cp $path/images_07*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_07*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#     fi
#     if [ "$folder" = "Leip_sequ_plus30" ]; then
#         #func
#         cp $path/images_08*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_08*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii

#         #fmap
#         cp $path/images_09*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_09*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#         cp $path/images_09*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_09*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii

#         cp $path/images_09*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_09*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#         cp $path/images_09*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_09*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii

#         cp $path/images_010*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_010*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#         cp $path/images_010*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_010*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#     fi
# done


# path="/Users/xpsy1114/Documents/projects/multiple_clocks/data/downloads/Jacob_day1"
# subjectTag="01"
# folder_names=("AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv")
# for folder in "${folder_names[@]}"; do
#     #anat
#     cp $path/images_017*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/
#     cp $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/images_017*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
#     gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
#     rm $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
#     if [ "$folder" = "AB_sequ_min30" ]; then
#         #func
#         cp $path/images_05*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_05*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii

#         #fmap
#         cp $path/images_06*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_06*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#         cp $path/images_06*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_06*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii

#         cp $path/images_06*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_06*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#         cp $path/images_06*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_06*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii

#         cp $path/images_07*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_07*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#         cp $path/images_07*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_07*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#     fi
#     if [ "$folder" = "AB_sequ_transv" ]; then
#         #func
#         cp $path/images_09*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_09*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii

#         #fmap
#         cp $path/images_010*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_010*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#         cp $path/images_010*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_010*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii

#         cp $path/images_010*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_010*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#         cp $path/images_010*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_010*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii

#         cp $path/images_011*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_011*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#         cp $path/images_011*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_011*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#     fi
#     if [ "$folder" = "AB_sequ_plus30" ]; then
#         #func
#         cp $path/images_013*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_013*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii

#         #fmap
#         cp $path/images_014*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_014*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#         cp $path/images_014*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_014*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii

#         cp $path/images_014*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_014*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#         cp $path/images_014*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_014*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii

#         cp $path/images_015*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_015*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#         cp $path/images_015*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_015*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#     fi
# done

# path="/Users/xpsy1114/Documents/projects/multiple_clocks/data/downloads/Jacob_day2"
# subjectTag="01"
# folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30")
# for folder in "${folder_names[@]}"; do
#     #anat
#     cp /Users/xpsy1114/Documents/projects/multiple_clocks/data/downloads/Jacob_day1/images_017*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/
#     cp $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/images_017*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
#     gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
#     rm $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
#     if [ "$folder" = "Leip_sequ_min30" ]; then
#         #func
#         cp $path/images_02*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_02*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii

#         #fmap
#         cp $path/images_03*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_03*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#         cp $path/images_03*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_03*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii

#         cp $path/images_03*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_03*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#         cp $path/images_03*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_03*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii

#         cp $path/images_04*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_04*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#         cp $path/images_04*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_04*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#     fi
#     if [ "$folder" = "Leip_sequ_transv" ]; then
#         #func
#         cp $path/images_05*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_05*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii

#         #fmap
#         cp $path/images_06*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_06*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#         cp $path/images_06*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_06*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii

#         cp $path/images_06*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_06*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#         cp $path/images_06*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_06*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii

#         cp $path/images_07*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_07*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#         cp $path/images_07*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_07*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#     fi
#     if [ "$folder" = "Leip_sequ_plus30" ]; then
#         #func
#         cp $path/images_08*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_08*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii

#         #fmap
#         cp $path/images_09*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_09*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#         cp $path/images_09*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_09*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii

#         cp $path/images_09*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_09*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#         cp $path/images_09*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_09*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii

#         cp $path/images_010*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_010*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#         cp $path/images_010*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
#         cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_010*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#     fi
# done


# write an annoying loop which moves and renames the appropriate files.
# firstly remove the folders from Jacobs dir > he wasnt in the 7T...



# # 7T subjects
# subjects_7T=("02")
# folders_7T=("run1_T7_AB_rep" "run2_T7_2mm_3iPat_2MB" "run3_T7_15mm_3iPat_3MB" "run4_T7_15mm_2iPat_3MB" "run5_T7_superfast" "run6_T7_12mm_3iPat_3MB")

# calpendoDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data/F7T_2013_40_867'



# for folder in "${folders_7T[@]}"; do
#     for subjectTag in "${subjects_7T[@]}"; do
#         # struct is the same for all
#         cp $calpendoDir/images_03*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
#         gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii

#         cp $calpendoDir/images_03*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.json
#         if [ "$folder" = "run1_T7_AB_rep" ]; then
#             # run 1
#             # func
#             # only take the long one
#             mv $calpendoDir/images_05*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#             mv $calpendoDir/images_05*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.json

#             # behaviour
#             mv $calpendoDir/AB_7T_firstsequ*.csv $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.csv
#             mv $calpendoDir/AB_7T_firstsequ*.log $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.log
#             mv $calpendoDir/AB_7T_firstsequ*.psydat $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.psydat

#             # physiology
#             mv $calpendoDir/first*.txt $scratchDir/tech_scan/sub-$subjectTag/$folder/motion/sub-${subjectTag}_physio.txt
#             mv $calpendoDir/first*.acq $scratchDir/tech_scan/sub-$subjectTag/$folder/motion/sub-${subjectTag}_physio.acq

#             # fieldmap
#             mv $calpendoDir/images_06*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#             mv $calpendoDir/images_06*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#             mv $calpendoDir/images_07*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#             mv $calpendoDir/images_06*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             mv $calpendoDir/images_06*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             mv $calpendoDir/images_07*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         fi
#         if [ "$folder" = "run2_T7_2mm_3iPat_2MB" ]; then
#             # run 2
#             # func
#             # only take the long one
#             mv $calpendoDir/images_08*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#             mv $calpendoDir/images_08*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.json

#             # behaviour
#             mv $calpendoDir/AB_7T_run2*.csv $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.csv
#             mv $calpendoDir/AB_7T_run2*.log $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.log
#             mv $calpendoDir/AB_7T_run2*.psydat $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.psydat

#             # physiology
#             mv $calpendoDir/second*.txt $scratchDir/tech_scan/sub-$subjectTag/$folder/motion/sub-${subjectTag}_physio.txt
#             mv $calpendoDir/second*.acq $scratchDir/tech_scan/sub-$subjectTag/$folder/motion/sub-${subjectTag}_physio.acq

#             # fieldmap
#             # I believe this one is only for the 3T replication one, so move it
#             mv $calpendoDir/images_09*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#             mv $calpendoDir/images_09*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#             mv $calpendoDir/images_010*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#             mv $calpendoDir/images_09*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folderß/fmap/sub-${subjectTag}_magnitude1.nii
#             mv $calpendoDir/images_09*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             mv $calpendoDir/images_010*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         fi
#         if [ "$folder" = "run3_T7_15mm_3iPat_3MB" ]; then
#             # run 3
#             # func
#             # only take the repeated one
#             mv $calpendoDir/images_013*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#             mv $calpendoDir/images_013*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.json

#             # behaviour
#             mv $calpendoDir/AB_7T_run3*.csv $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.csv
#             mv $calpendoDir/AB_7T_run3*.log $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.log
#             mv $calpendoDir/AB_7T_run3*.psydat $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.psydat

#             # physiology
#             mv $calpendoDir/third*.txt $scratchDir/tech_scan/sub-$subjectTag/$folder/motion/sub-${subjectTag}_physio.txt
#             mv $calpendoDir/third*.acq $scratchDir/tech_scan/sub-$subjectTag/$folder/motion/sub-${subjectTag}_physio.acq

#             # fieldmap
#             # this time copy since its for more sequences
#             cp $calpendoDir/images_015*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#             cp $calpendoDir/images_015*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#             cp $calpendoDir/images_016*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#             cp $calpendoDir/images_015*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             cp $calpendoDir/images_015*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             cp $calpendoDir/images_016*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         fi
#         if [ "$folder" = "run4_T7_15mm_2iPat_3MB" ]; then
#             # run 4
#             # func
#             # only take the long one
#             mv $calpendoDir/images_014*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#             mv $calpendoDir/images_014*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.json

#             # behaviour
#             mv $calpendoDir/AB_run4*.csv $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.csv
#             mv $calpendoDir/AB_run4*.log $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.log
#             mv $calpendoDir/AB_run4*.psydat $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.psydat

#             # physiology
#             mv $calpendoDir/fourth*.txt $scratchDir/tech_scan/sub-$subjectTag/$folder/motion/sub-${subjectTag}_physio.txt
#             mv $calpendoDir/fourth*.acq $scratchDir/tech_scan/sub-$subjectTag/$folder/motion/sub-${subjectTag}_physio.acq

#             # fieldmap
#             # this time copy since its for more sequences
#             cp $calpendoDir/images_015*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#             cp $calpendoDir/images_015*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#             cp $calpendoDir/images_016*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#             cp $calpendoDir/images_015*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             cp $calpendoDir/images_015*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             cp $calpendoDir/images_016*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         fi
#         if [ "$folder" = "run5_T7_superfast" ]; then
#             # run 5
#             # func
#             # only take the long one
#             mv $calpendoDir/images_017*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#             mv $calpendoDir/images_017*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.json

#             # behaviour
#             mv $calpendoDir/AB_run5*.csv $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.csv
#             mv $calpendoDir/AB_run5*.log $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.log
#             mv $calpendoDir/AB_run5*.psydat $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.psydat

#             # physiology
#             mv $calpendoDir/fith*.txt $scratchDir/tech_scan/sub-$subjectTag/$folder/motion/sub-${subjectTag}_physio.txt
#             mv $calpendoDir/fith*.acq $scratchDir/tech_scan/sub-$subjectTag/$folder/motion/sub-${subjectTag}_physio.acq

#             # fieldmap
#             # this time copy since its for more sequences
#             cp $calpendoDir/images_015*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#             cp $calpendoDir/images_015*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#             cp $calpendoDir/images_016*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#             cp $calpendoDir/images_015*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             cp $calpendoDir/images_015*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             cp $calpendoDir/images_016*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         fi
#         if [ "$folder" = "run6_T7_12mm_3iPat_3MB" ]; then
#             # run 6
#             # func
#             # only take the long one
#             mv $calpendoDir/images_018*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
#             mv $calpendoDir/images_018*.json $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.json

#             # behaviour
#             mv $calpendoDir/AB_smallvox*.csv $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.csv
#             mv $calpendoDir/AB_smallvox*.log $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.log
#             mv $calpendoDir/AB_smallvox*.psydat $scratchDir/tech_scan/sub-$subjectTag/$folder/beh/sub-${subjectTag}_beh.psydat

#             # physiology
#             mv $calpendoDir/sixth*.txt $scratchDir/tech_scan/sub-$subjectTag/$folder/motion/sub-${subjectTag}_physio.txt
#             mv $calpendoDir/sixth*.acq $scratchDir/tech_scan/sub-$subjectTag/$folder/motion/sub-${subjectTag}_physio.acq

#             # fieldmap
#             # this time copy since its for more sequences
#             cp $calpendoDir/images_015*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#             cp $calpendoDir/images_015*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#             cp $calpendoDir/images_016*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#             cp $calpendoDir/images_015*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             cp $calpendoDir/images_015*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             cp $calpendoDir/images_016*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         fi
#     done
# done


scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'
path="/Users/xpsy1114/Documents/projects/multiple_clocks/data/downloads/F7T_2013_40_872"
subjectTag="03"
folder_names=("run1_3Trep_7T2" "run2_25mm_fast_7T2" "run3_mb4_fast_7T2" "run4_mb3ipat3_7T2" "run5_ipat4_7T2")

for folder in "${folder_names[@]}"; do
        # set up BIDS folder structure for raw data.
        mkdir -p $scratchDir/tech_scan
        mkdir -p $scratchDir/tech_scan/sub-$subjectTag
        mkdir -p $scratchDir/tech_scan/sub-$subjectTag/$folder

        # Construct structural directory for raw file
        anatDir=$scratchDir/tech_scan/sub-$subjectTag/$folder/anat
        # And create directory for derived anatomy files
        mkdir -p $anatDir

        # Construct functional directory for raw file
        funcDir=$scratchDir/tech_scan/sub-$subjectTag/$folder/func
        # And create directory for derived anatomy files
        mkdir -p $funcDir

        # Construct behavioural directory for raw file
        behDir=$scratchDir/tech_scan/sub-$subjectTag/$folder/beh
        # And create directory for derived anatomy files
        mkdir -p $behDir

        # Construct fieldmap directory for raw file
        fmapDir=$scratchDir/tech_scan/sub-$subjectTag/$folder/fmap
        # And create directory for derived anatomy files
        mkdir -p $fmapDir

        # Construct physiology directory for raw file
        motionDir=$scratchDir/tech_scan/sub-$subjectTag/$folder/motion
        # And create directory for derived anatomy files
        mkdir -p $motionDir
done


for folder in "${folder_names[@]}"; do
    #anat
    cp $path/images_011_MPRAGE_UP.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/
    cp $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/images_011_MPRAGE_UP.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
    gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
    rm $scratchDir/tech_scan/sub-$subjectTag/$folder/anat/sub-${subjectTag}_T1w.nii
    
    #fmap
    cp $path/images_032*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
    cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_032*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
    cp $path/images_032*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
    cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_032*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
    gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
    rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii

    cp $path/images_032*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
    cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_032*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
    cp $path/images_032*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
    cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_032*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
    gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
    rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii

    cp $path/images_033*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
    cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_033*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
    cp $path/images_033*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/
    cp $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/images_033*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
    gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
    rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii

    if [ "$folder" = "run1_3Trep_7T2" ]; then
        #func
        cp $path/images_013*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
        cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_013*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
        gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
        rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
    fi
    if [ "$folder" = "run2_25mm_fast_7T2" ]; then
        #func
        cp $path/images_017*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
        cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_017*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
        gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
        rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
    fi
    if [ "$folder" = "run3_mb4_fast_7T2" ]; then
        #func
        cp $path/images_021*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
        cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_021*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
        gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
        rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
    fi
    if [ "$folder" = "run4_mb3ipat3_7T2" ]; then
        #func
        cp $path/images_025*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
        cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_025*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
        gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
        rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
    fi
        if [ "$folder" = "run5_ipat4_7T2" ]; then
        #func
        cp $path/images_029*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/
        cp $scratchDir/tech_scan/sub-$subjectTag/$folder/func/images_029*.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
        gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
        rm $scratchDir/tech_scan/sub-$subjectTag/$folder/func/sub-${subjectTag}_bold.nii
    fi
done