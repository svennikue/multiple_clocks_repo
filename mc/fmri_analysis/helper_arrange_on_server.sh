# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
toolboxDir="/home/fs0/xpsy1114/scratch/analysis"
homeDir="/home/fs0/xpsy1114"
analysisDir="${scratchDir}/analysis"
laptopDir="/home/fs0/xpsy1114/scratch/from_laptop/for_server"
 
for subjectTag in 01 02 03 04 05 06; do
    
    #cp ${laptopDir}/sub-${subjectTag}/motion/sub-${subjectTag}_physio.txt ${scratchDir}/pilot/sub-${subjectTag}/motion/sub-${subjectTag}_physio.txt
    #cp ${laptopDir}/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1_all.csv ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1_all.csv 
    #cp ${laptopDir}//sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2_all.csv ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2_all.csv 
    #cp ${laptopDir}/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1.csv ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1.csv 
    #cp ${laptopDir}/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2.csv ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2.csv 

    cp -r ${laptopDir}/sub-${subjectTag}/func/EVs_06_pt01_press_and_loc ${scratchDir}/derivatives/sub-${subjectTag}/func/EVs_06_pt01_press_and_loc 
    cp -r ${laptopDir}/sub-${subjectTag}/func/EVs_06_pt02_press_and_loc ${scratchDir}/derivatives/sub-${subjectTag}/func/EVs_06_pt02_press_and_loc 
    
    #mkdir ${scratchDir}/derivatives/sub-${subjectTag}/beh
    cp -r ${laptopDir}/sub-${subjectTag}/beh/RDMs_04_glmbase_06 ${scratchDir}/derivatives/sub-${subjectTag}/beh/RDMs_04_glmbase_06 

    cp ${laptopDir}/sub-${subjectTag}/func/sub-${subjectTag}_my_RDM_GLM_01_06.fsf ${scratchDir}/derivatives/sub-${subjectTag}/func/sub-${subjectTag}_my_RDM_GLM_01_06.fsf 
    cp ${laptopDir}/sub-${subjectTag}/func/sub-${subjectTag}_my_RDM_GLM_02_06.fsf ${scratchDir}/derivatives/sub-${subjectTag}/func/sub-${subjectTag}_my_RDM_GLM_02_06.fsf 
done