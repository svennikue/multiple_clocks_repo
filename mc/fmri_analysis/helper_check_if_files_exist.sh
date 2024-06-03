# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
echo Scratch directory is $scratchDir

# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do
# for subjectTag in 03 04 07 08 09; do
#for subjectTag in 03 04 07 08 09 10 11 12 14 15 16 17 20 21 23 24; do
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 ; do
#     # Command line argument 1/1: subject tag
#     echo Subject tag for this subject: $subjectTag
#     for task_half in 01 02 ; do
#         preprocDir=$scratchDir/derivatives/sub-$subjectTag/func/preproc_clean_${task_half}.feat
#         if [ -e "$preprocDir/filtered_func_data.nii.gz" ]; then
#             echo Subject $subjectTag task half $task_half has filtered_func_data! 
#         else
#             echo !!! Faulty preproc for subject $subjectTag in task half $task_half !!!
#         fi
#     done
# done 

# to check behavioural files
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 ; do
#     # Command line argument 1/1: subject tag
#     echo Subject tag for this subject: $subjectTag
#     for task_half in 1 2 ; do
#         behDir=$scratchDir/pilot/sub-$subjectTag/beh
#         if [ -e "$behDir/sub-${subjectTag}_fmri_pt${task_half}.csv" ]; then
#             echo Subject $subjectTag task half $task_half has $behDir/sub-${subjectTag}_fmri_pt${task_half}.csv
#         else
#             echo !!! Missing behavioural data for subject $subjectTag in task half $task_half $behDir/sub-${subjectTag}_fmri_pt${task_half}.csv !!!
#         fi
#     done
# done 

# # to check motion nuisance regressors
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 ; do
#     # Command line argument 1/1: subject tag
#     echo Subject tag for this subject: $subjectTag
#     for task_half in 01 02 ; do
#         nuisanceDir=$scratchDir/derivatives/sub-$subjectTag/motion/nuisance_${task_half}
#         if [ -e "$nuisanceDir/motionOutliers.txt" ]; then
#             echo Subject $subjectTag task half $task_half has motion outlier txt! 
#         else
#             echo !!! Faulty nuisance step one for subject $subjectTag in task half $task_half !!!
#         fi
#     done
# done 

# to check if subject glms worked
for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 ; do
    # Command line argument 1/1: subject tag
    echo Subject tag for this subject: $subjectTag
    for task_half in 01 02 ; do
        glmDir=$scratchDir/derivatives/sub-$subjectTag/func
        if [ -e "$glmDir/glm_01_pt${task_half}.feat/stats/pe1.nii.gz" ]; then
            echo Subject $subjectTag task half $task_half has glm 03-e!
        else
            echo !!! Faulty glm 03-e step one for subject $subjectTag in task half $task_half !!!
        fi
        if [ -e "$glmDir/glm_03_pt${task_half}.feat/stats/pe1.nii.gz" ]; then
            echo Subject $subjectTag task half $task_half has glm 03-l!
        else
            echo !!! Faulty glm 03-l step one for subject $subjectTag in task half $task_half !!!
        fi

    done
done 

# to check if RSA has worked
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 ; do
#     # Command line argument 1/1: subject tag
#     echo Subject tag for this subject: $subjectTag
#     for analysis in 05_glmbase_06 09_glmbase_07 ; do
#         subDir=$scratchDir/derivatives/sub-$subjectTag
#         echo checking $subDir/beh/RDMs_${analysis}/RSM*both_halves.npy
#         if [ -e "$subDir/beh/RDMs_${analysis}/RSM*both_halves.npy" ]; then
#             echo Subject $subjectTag analysis $analysis has model RDMs!
#         else
#             echo !!! Faulty model RDMS for subject $subjectTag analysis $analysis in beh/RDM dir!!!
#         fi
#         echo checking $subDir/func/RSA_${analysis}/results/*_beta.nii.gz
#         if [ -e "$subDir/func/RSA_${analysis}/results/*_beta.nii.gz" ]; then
#             echo Subject $subjectTag has some beta result for RSA $analysis 
#         else
#             echo !!! Faulty analysis in func/RSA dir for $analysis subject $subjectTag !!!
#         fi

#     done
# done 