#
 Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
toolboxDir="/home/fs0/xpsy1114/scratch/analysis"
homeDir="/home/fs0/xpsy1114"
analysisDir="${scratchDir}/analysis"
laptopDir="${scratchDir}/from_laptop/for_server"

for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do

    mv ${laptopDir}/sub-${subjectTag}/motion/sub-${subjectTag}_physio.txt ${scratchDir}/pilot/sub-${subjectTag}/motion/sub-${subjectTag}_physio.txt
    mv ${laptopDir}/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1_all.csv ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1_all.csv 
    mv ${laptopDir}//sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2_all.csv ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2_all.csv 
    ${laptopDir}/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1.csv ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1.csv 
    ${laptopDir}/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2.csv ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2.csv 

    mv -r ${laptopDir}/sub-${subjectTag}/func/EVs_06_pt01_press_and_loc ${scratchDir}/derivatives/sub-${subjectTag}/func/EVs_06_pt01_press_and_loc 
    mv -r ${laptopDir}/sub-${subjectTag}/func/EVs_06_pt02_press_and_loc ${scratchDir}/derivatives/sub-${subjectTag}/func/EVs_06_pt02_press_and_loc 
    mv -r ${laptopDir}/sub-${subjectTag}/beh/RDMs_04_glmbase_06 ${scratchDir}/derivatives/sub-${subjectTag}/beh/RDMs_04_glmbase_06 

    mv ${laptopDir}/sub-${subjectTag}/func/sub-${subjectTag}_my_RDM_GLM_01_06.fsf ${scratchDir}/derivatives/sub-${subjectTag}/func/sub-${subjectTag}_my_RDM_GLM_01_06.fsf 
    mv ${laptopDir}/sub-${subjectTag}/func/sub-${subjectTag}_my_RDM_GLM_02_06.fsf ${scratchDir}/derivatives/sub-${subjectTag}/func/sub-${subjectTag}_my_RDM_GLM_02_06.fsf 
done