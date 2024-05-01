# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"

glm_version="03-4"
RSA_version="03-1"


groupDir=${scratchDir}/derivatives/group/group_RSA_${RSA_version}_glmbase_${glm_version}
echo this is group dir $groupDir
if [ ! -d $groupDir ]; then
    mkdir $groupDir
fi

example_resultDir=${scratchDir}/derivatives/sub-02/func/RSA_${RSA_version}_glmbase_${glm_version}/results-standard-space

if [ ! -d $example_resultDir ]; then
    example_resultDir=${scratchDir}/derivatives/sub-02/func/RSA_${RSA_version}_${glm_version}/results-standard-space
    list_of_std_beta_files=$(find "$example_resultDir" -name "avg*beta_std.nii.gz" -type f)
else
    list_of_std_beta_files=$(find "$example_resultDir" -name "*beta_std.nii.gz" -type f)
fi

echo this is example resultDir $example_resultDir
# Then, for each of these files
for file in $list_of_std_beta_files; do
    # Extract the filename
    filename=$(basename "$file")
    echo now moving and merging $filename
    # no 21 
    # also find a way to include sub 33 and 34 in glm 06 rsa 05 once they are done!!
    for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33; do
        # for every result file
        resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_glmbase_${glm_version}/results-standard-space
        if [ ! -d $resultDir ]; then
            resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_${glm_version}/results-standard-space
        fi

        echo now for subject $subjectTag and $resultDir
        if [ ! -f ${groupDir}/${filename} ]; then
            cp ${resultDir}/$filename ${groupDir}/${filename}
        else 
            fslmerge -t ${groupDir}/${filename} ${resultDir}/$filename ${groupDir}/${filename}
        fi
    done
done
    
gunzip $(ls ${groupDir}/*.nii.gz)
echo done!


#         if ${RSA_version}=="05"
#             cp ${resultDir}/beta_my_state_std.nii.gz ${groupDir}/state_4d.nii.gz
#             cp ${resultDir}/beta_my_state_combo_std.nii.gz ${groupDir}/state_combo_4d.nii.gz
#             cp ${resultDir}/beta_my_clock_std.nii.gz ${groupDir}/clock_4d.nii.gz
#             cp ${resultDir}/beta_my_clock_combo_std.nii.gz ${groupDir}/clock_combo_4d.nii.gz
#             cp ${resultDir}/beta_my_midn_std.nii.gz ${groupDir}/midn_4d.nii.gz
#             cp ${resultDir}/beta_my_midn_combo_std.nii.gz ${groupDir}/midn_combo_4d.nii.gz
#             cp ${resultDir}/beta_my_loc_std.nii.gz ${groupDir}/loc_4d.nii.gz
#             cp ${resultDir}/beta_my_loc_combo_std.nii.gz ${groupDir}/loc_combo_4d.nii.gz
#             cp ${resultDir}/beta_my_phase_std.nii.gz ${groupDir}/phase_4d.nii.gz
#             cp ${resultDir}/beta_my_phase_combo_std.nii.gz ${groupDir}/phase_combo_4d.nii.gz
#             cp ${resultDir}/beta_my_task_prog_std.nii.gz ${groupDir}/task_prog_4d.nii.gz
#         fi
#         if ${RSA_version}=="05"
#         # CONTINUE HERE!!! ADJUST SUCH THAT I ACUTALLY HAVE THE NAMES READY. BETTER: LOOP THROUGH STRING LIST....
#             cp ${resultDir}/beta_my_state_std.nii.gz ${groupDir}/state_4d.nii.gz
#             cp ${resultDir}/beta_my_state_combo_std.nii.gz ${groupDir}/state_combo_4d.nii.gz
#             cp ${resultDir}/beta_my_clock_std.nii.gz ${groupDir}/clock_4d.nii.gz
#             cp ${resultDir}/beta_my_clock_combo_std.nii.gz ${groupDir}/clock_combo_4d.nii.gz
#             cp ${resultDir}/beta_my_midn_std.nii.gz ${groupDir}/midn_4d.nii.gz
#             cp ${resultDir}/beta_my_midn_combo_std.nii.gz ${groupDir}/midn_combo_4d.nii.gz
#             cp ${resultDir}/beta_my_loc_std.nii.gz ${groupDir}/loc_4d.nii.gz
#             cp ${resultDir}/beta_my_loc_combo_std.nii.gz ${groupDir}/loc_combo_4d.nii.gz
#             cp ${resultDir}/beta_my_phase_std.nii.gz ${groupDir}/phase_4d.nii.gz
#             cp ${resultDir}/beta_my_phase_combo_std.nii.gz ${groupDir}/phase_combo_4d.nii.gz
#             cp ${resultDir}/beta_my_task_prog_std.nii.gz ${groupDir}/task_prog_4d.nii.gz
#         #cp ${resultDir}/beta_my_task_prog_combo_std.nii.gz ${groupDir}/task_prog_combo_4d.nii.gz
#     else
#         fslmerge -t ${groupDir}/state_4d.nii.gz ${resultDir}/beta_my_state_std.nii.gz ${groupDir}/state_4d.nii.gz
#         fslmerge -t ${groupDir}/state_combo_4d.nii.gz ${resultDir}/beta_my_state_combo_std.nii.gz ${groupDir}/state_combo_4d.nii.gz
#         fslmerge -t ${groupDir}/clock_4d.nii.gz ${resultDir}/beta_my_clock_std.nii.gz ${groupDir}/clock_4d.nii.gz
#         fslmerge -t ${groupDir}/clock_combo_4d.nii.gz ${resultDir}/beta_my_clock_combo_std.nii.gz ${groupDir}/clock_combo_4d.nii.gz
#         fslmerge -t ${groupDir}/midn_4d.nii.gz ${resultDir}/beta_my_midn_std.nii.gz ${groupDir}/midn_4d.nii.gz
#         fslmerge -t ${groupDir}/midn_combo_4d.nii.gz ${resultDir}/beta_my_midn_combo_std.nii.gz ${groupDir}/midn_combo_4d.nii.gz
#         fslmerge -t ${groupDir}/loc_4d.nii.gz ${resultDir}/beta_my_loc_std.nii.gz ${groupDir}/loc_4d.nii.gz
#         fslmerge -t ${groupDir}/loc_combo_4d.nii.gz ${resultDir}/beta_my_loc_combo_std.nii.gz ${groupDir}/loc_combo_4d.nii.gz
#         fslmerge -t ${groupDir}/phase_4d.nii.gz ${resultDir}/beta_my_phase_std.nii.gz ${groupDir}/phase_4d.nii.gz
#         fslmerge -t ${groupDir}/phase_combo_4d.nii.gz ${resultDir}/beta_my_phase_combo_std.nii.gz ${groupDir}/phase_combo_4d.nii.gz
#         fslmerge -t ${groupDir}/task_prog_4d.nii.gz ${resultDir}/beta_my_task_prog_std.nii.gz ${groupDir}/task_prog_4d.nii.gz
#         #fslmerge -t ${groupDir}/task_prog_combo_4d.nii.gz ${resultDir}/beta_my_task_prog_combo_std.nii.gz ${groupDir}/task_prog_combo_4d.nii.gz
#     fi
# done

# # because PALM needs .nii files, unzip all files in this folder to end with.
# gunzip $(ls ${groupDir}/*.nii.gz)
# # gunzip ${groupDir}/state_combo_4d.nii.gz
# # gunzip ${groupDir}/clock_4d.nii.gz
# # gunzip ${groupDir}/clock_combo_4d.nii.gz
# # gunzip ${groupDir}/midn_4d.nii.gz
# # gunzip ${groupDir}/midn_combo_4d.nii.gz
# # gunzip ${groupDir}/loc_combo_4d.nii.gz
# # gunzip ${groupDir}/phase_4d.nii.gz
# # gunzip ${groupDir}/phase_combo_4d.nii.gz
# # gunzip ${groupDir}/task_prog_4d.nii.gz


# echo done! 
