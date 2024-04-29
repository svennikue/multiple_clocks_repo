# sub01 and ABmin30 is *e2.json, *e2.nii.gz, *e2_ph.json, *e2_ph.nii.gz
# here I need to find out what is what!
# I'll do that tomorrow :)
# _magnitude1.json _magnitude1.nii.gz _magnitude2.json _magnitude2.nii.gz _phasediff.json _phasediff.nii.gz


subjects=("01" "02")
folder_names=("Leip_sequ_transv" "Leip_sequ_min30" "Leip_sequ_plus30" "AB_sequ_plus30" "AB_sequ_min30" "AB_sequ_transv")
scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'
calpendoDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data/F7T_2013_40_867'

for folder in "${folder_names[@]}"; do
   for subjectTag in "${subjects[@]}"; do
       mv $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
       mv $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
       mv $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
       mv $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
       gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
       rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
       mv $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
       gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
       rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
       mv $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
       gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
       rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
   done
done


# for the 7T fieldmaps

# 7T subjects
#subjects_7T=("02")
#folders_7T=("run1_T7_AB_rep" "run2_T7_2mm_3iPat_2MB" "run3_T7_15mm_3iPat_3MB" "run4_T7_15mm_2iPat_3MB" "run5_T7_superfast" "run6_T7_12mm_3iPat_3MB")
#scratchDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data'
#calpendoDir='/Users/xpsy1114/Documents/projects/multiple_clocks/data/F7T_2013_40_867'

# for folder in "${folder_names[@]}"; do
#     for subjectTag in "${subjects[@]}"; do
#         if [ "$folder" = "run1_T7_AB_rep" ]; then
#             # fieldmap
#             # I believe this one is only for the 3T replication one, so move it
#             mv $calpendoDir/images_06*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#             mv $calpendoDir/images_06*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#             mv $calpendoDir/images_07*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#             mv $calpendoDir/images_06*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             mv $calpendoDir/images_06*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             mv $calpendoDir/images_07*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         fi
#         # used for run 2
#         if [ "$folder" = "run2_T7_2mm_3iPat_2MB" ]; then
#             # fieldmap
#             # I believe this one is only for the 3T replication one, so move it
#             mv $calpendoDir/images_09*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#             mv $calpendoDir/images_09*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#             mv $calpendoDir/images_010*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#             mv $calpendoDir/images_09*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             mv $calpendoDir/images_09*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             mv $calpendoDir/images_010*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         fi
#         # used for run 3 4 5 and 6 
#         if [ "$folder" = "run3_T7_15mm_3iPat_3MB" ]; then
#             # fieldmap
#             cp $calpendoDir/images_015*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#             cp $calpendoDir/images_015*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#             cp $calpendoDir/images_016*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#             cp $calpendoDir/images_015*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             cp $calpendoDir/images_015*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             cp $calpendoDir/images_016*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         fi
#         # used for run 3 4 5 and 6 
#         if [ "$folder" = "run4_T7_15mm_2iPat_3MB" ]; then
#             # fieldmap
#             cp $calpendoDir/images_015*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#             cp $calpendoDir/images_015*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#             cp $calpendoDir/images_016*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#             cp $calpendoDir/images_015*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             cp $calpendoDir/images_015*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             cp $calpendoDir/images_016*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         fi
#         # used for run 3 4 5 and 6 
#         if [ "$folder" = "run5_T7_superfast" ]; then
#             # fieldmap
#             cp $calpendoDir/images_015*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#             cp $calpendoDir/images_015*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#             cp $calpendoDir/images_016*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#             cp $calpendoDir/images_015*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             cp $calpendoDir/images_015*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             cp $calpendoDir/images_016*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         fi
#         # used for run 3 4 5 and 6 
#         if [ "$folder" = "run6_T7_12mm_3iPat_3MB" ]; then
#             # fieldmap
#             cp $calpendoDir/images_015*e1.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.json
#             cp $calpendoDir/images_015*e2.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.json
#             cp $calpendoDir/images_016*e2_ph.json $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.json
#             cp $calpendoDir/images_015*e1.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             cp $calpendoDir/images_015*e2.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             cp $calpendoDir/images_016*e2_ph.nii $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude1.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_magnitude2.nii
#             gzip $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#             rm $scratchDir/tech_scan/sub-$subjectTag/$folder/fmap/sub-${subjectTag}_phasediff.nii
#         fi
#     done
# done
