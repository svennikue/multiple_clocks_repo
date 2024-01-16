#
 Set scratch directory for execution on server
 scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
 serverDir="/Users/xpsy1114/Documents/projects/multiple_clocks/from_server/server-laptop"

scratchDir="/home/fs0/xpsy1114/scratch/data"
toolboxDir="/home/fs0/xpsy1114/scratch/analysis"
homeDir="/home/fs0/xpsy1114"
analysisDir="${scratchDir}/analysis"
laptopDir="${scratchDir}/from_laptop/for_server"

for subjectTag in 01 02 03 04; do
    mv ${laptopDir}/sub-${subjectTag}/anat/sub-${subjectTag}_T1w_biascorr_noCSF_brain.nii.gz ${scratchDir}/derivatives/sub-${subjectTag}/anat/sub-${subjectTag}_T1w_biascorr_noCSF_brain.nii.gz
    for task_half in 01 02; do
        mkdir ${scratchDir}/derivatives/sub-${subjectTag}/func/cluster_glm_06_pt${task_half}.feat
        mv ${laptopDir}/sub-${subjectTag}/${task_half}/stats ${scratchDir}/derivatives/sub-${subjectTag}/func/cluster_glm_06_pt${task_half}.feat/stats
        mv ${laptopDir}/sub-${subjectTag}/anat/sub-${subjectTag}_T1w_biascorr_noCSF_brain_mask_examplefunc_pt${task_half}.nii.gz ${scratchDir}/derivatives/sub-${subjectTag}/anat/sub-${subjectTag}_T1w_biascorr_noCSF_brain_mask_examplefunc_pt${task_half}.nii.gz
done