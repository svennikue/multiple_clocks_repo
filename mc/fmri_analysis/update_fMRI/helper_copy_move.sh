scratchDir="/home/fs0/xpsy1114/scratch/data"
# copy json
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34; do
#     funcDir=$scratchDir/pilot/sub-$subjectTag/func
#     rawDir=$scratchDir/raw/${subjectTag}_scan
#     # func 1 
#     cp $rawDir/images_05*.json $funcDir/
#     cp $funcDir/images_05*.json $funcDir/sub-${subjectTag}_1_bold.json

#     # func 2 
#     cp $rawDir/images_08*.json $funcDir/
#     cp $funcDir/images_08*.json $funcDir/sub-${subjectTag}_2_bold.json

#     physDir=$scratchDir/pilot/sub-$subjectTag/motion
#     if [ -e "$physDir/sub-${subjectTag}_physio.txt" ]; then
#         echo Subject $subjectTag has the biopack .txt. 
#     else
#         echo !!! missing biopack.txt for Subject $subjectTag go look !!!
#     fi
# done

#copy all result folders: EVs_*, glm_*, nuisance_*, RSA_* *.fst in a folder called 'old-pre-29-03-2024'
# this can probably be deleted at some point, but for now just move it.

# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34; do
# # for subjectTag in 35; do
#     funcDir=$scratchDir/derivatives/sub-$subjectTag/func
#     oldDir=$funcDir/RSA_DSR_bias-path-rew-splitfuts_combos_30-01-2026_glmbase_all-paths-fixed_stickrews_split-buttons/results
#     newDir=$funcDir/RSA_DSR_bias-path-rew-splitfuts_combos_31-01-2026_glmbase_all-paths-fixed_stickrews_split-buttons/results
#     echo now moving subject $subjectTag
#     mv $oldDir/* $newDir/
# done

for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34; do
    oldDir="$scratchDir/derivatives/sub-$subjectTag/func/RSA_DSR_bias-path-rew-splitfuts_combos_30-01-2026_glmbase_all-paths-fixed_stickrews_split-buttons"
    rm -rf "$oldDir"
    # for subDir in results smooth; do
    #     targetDir="$oldDir/$subDir"

    #     if [ -d "$targetDir" ]; then
    #         if [ -z "$(ls -A "$targetDir")" ]; then
    #             echo "✅ subject $subjectTag — removing empty $subDir"
    #             rmdir "$targetDir"
    #         else
    #             echo "⚠️ subject $subjectTag — $subDir NOT empty, contains:"
    #             ls -lh "$targetDir"
    #             echo
    #         fi
    #     else
    #         echo "ℹ️ subject $subjectTag — $subDir does not exist"
    #     fi
    # done

    # echo "---------------------------------------------"
done

# move all old RDM folders
# for subjectTag in 01 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34; do
#     funcDir=$scratchDir/derivatives/sub-$subjectTag/beh
#     echo now moving subject $subjectTag
#     mkdir $funcDir/old_pre-29-03-2024
#     mv $funcDir/RDMs* $funcDir/old_pre-29-03-2024/
# done


# cp all files Alif needs to his scratch


# oldDir="/home/fs0/xpsy1114/scratch/data/"
# newDir="/home/fs0/chx061/scratch/data"
# #  01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35 ; do
#     echo now copying subject $subjectTag
#     # mkdir ${newDir}/raw/sub-$subjectTag
#     # mkdir ${newDir}/raw/sub-$subjectTag/beh
#     # cp ${oldDir}/pilot/sub-$subjectTag/beh/* ${newDir}/raw/sub-$subjectTag/beh

#     # mkdir ${newDir}/derivatives/sub-$subjectTag
#     # mkdir ${newDir}/derivatives/sub-$subjectTag/anat
#     # cp ${oldDir}/derivatives/sub-$subjectTag/anat/* ${newDir}/derivatives/sub-$subjectTag/anat/

#     mkdir ${newDir}/derivatives/sub-$subjectTag/beh
#     # cp -r ${oldDir}/derivatives/sub-$subjectTag/func/preproc_clean_01.feat ${newDir}/derivatives/sub-$subjectTag/func/preproc_clean_01.feat
#     # cp -r ${oldDir}/derivatives/sub-$subjectTag/func/preproc_clean_02.feat ${newDir}/derivatives/sub-$subjectTag/func/preproc_clean_02.feat
#     # 
#     for version in 06 06-rep1 ; do
#         cp -r ${oldDir}/derivatives/sub-$subjectTag/beh/RDMs_05_glmbase_${version} ${newDir}/derivatives/sub-$subjectTag/beh/RDMs_05_glmbase_${version}
#         chmod 777 -R ${newDir}/derivatives/sub-$subjectTag/beh/RDMs_05_glmbase_${version}

#         # cp -r ${oldDir}/derivatives/sub-$subjectTag/func/glm_06_pt${th}.feat ${newDir}/derivatives/sub-$subjectTag/func/glm_06_pt${th}.feat
#         # chmod 777 -R ${newDir}/derivatives/sub-$subjectTag/func/glm_06_pt${th}.feat

#         # cp -r ${oldDir}/derivatives/sub-$subjectTag/func/glm_06-rep1_pt${th}.feat ${newDir}/derivatives/sub-$subjectTag/func/glm_06-rep1_pt${th}.feat
#         # chmod 777 -R  ${newDir}/derivatives/sub-$subjectTag/func/glm_06-rep1_pt${th}.feat

#         # cp -r ${oldDir}/derivatives/sub-$subjectTag/func/EVs_06-rep1_pt${th} ${newDir}/derivatives/sub-$subjectTag/func/EVs_06-rep1_pt${th}
#         # chmod 777 -R ${newDir}/derivatives/sub-$subjectTag/func/EVs_06-rep1_pt${th}

#         # cp -r ${oldDir}/derivatives/sub-$subjectTag/func/EVs_06_pt${th} ${newDir}/derivatives/sub-$subjectTag/func/EVs_06_pt${th}
#         # chmod 777 -R ${newDir}/derivatives/sub-$subjectTag/func/EVs_06_pt${th}
#     done
# done

# oldDir="/home/fs0/xpsy1114/scratch/data/"
# newDir="/home/fs0/chx061/scratch/data"
# for subjectTag in 35; do
#     echo now also copying the EV folder for subject ${subjectTag}
#     cp -r ${oldDir}/derivatives/sub-$subjectTag/func/EVs_01_pt01 ${newDir}/derivatives/sub-$subjectTag/func/EVs_01_pt01
#     cp -r ${oldDir}/derivatives/sub-$subjectTag/func/EVs_01_pt02 ${newDir}/derivatives/sub-$subjectTag/func/EVs_01_pt02
#     chmod 777 -R ${newDir}/derivatives/sub-$subjectTag/func/preproc_clean_01.feat
#     chmod 777 -R ${newDir}/derivatives/sub-$subjectTag/func/preproc_clean_02.feat
# done
# chmod 777 ${newDir}/derivatives

# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
#     echo now liberating subject $subjectTag
#     chmod 777 ${newDir}/derivatives/sub-$subjectTag
#     chmod 777 ${newDir}/raw/sub-$subjectTag/beh
#     chmod 777 ${newDir}/derivatives/sub-$subjectTag/anat
#     chmod 777 ${newDir}/derivatives/sub-$subjectTag/func
# done


