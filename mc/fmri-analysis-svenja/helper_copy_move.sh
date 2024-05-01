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
#     funcDir=$scratchDir/derivatives/sub-$subjectTag/func
#     echo now moving subject $subjectTag
#     mkdir $funcDir/old_pre-29-03-2024
#     mv $funcDir/EVs* $funcDir/old_pre-29-03-2024/
#     mv $funcDir/glm* $funcDir/old_pre-29-03-2024/
#     mv $funcDir/nuisance* $funcDir/old_pre-29-03-2024/
#     mv $funcDir/RSA* $funcDir/old_pre-29-03-2024/
#     mv $funcDir/*.fsf $funcDir/old_pre-29-03-2024/
# done


# move all old RDM folders
for subjectTag in 01 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34; do
    funcDir=$scratchDir/derivatives/sub-$subjectTag/beh
    echo now moving subject $subjectTag
    mkdir $funcDir/old_pre-29-03-2024
    mv $funcDir/RDMs* $funcDir/old_pre-29-03-2024/
done