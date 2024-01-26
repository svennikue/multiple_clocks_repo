scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"

# for subjectTag in 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do
#     #mkdir ${scratchDir}/pilot/sub-${subjectTag}
#     #mkdir ${scratchDir}/pilot/sub-${subjectTag}/beh
#     #mkdir ${scratchDir}/pilot/sub-${subjectTag}/motion
#     scp -r ${scratchDir}/pilot/sub-${subjectTag}/motion/sub-${subjectTag}_physio.txt xpsy1114@jalapeno.fmrib.ox.ac.uk:/home/fs0/xpsy1114/scratch/data/pilot/sub-${subjectTag}/motion/sub-${subjectTag}_physio.txt
#     # cp ${scratchDir}/pilot/sub-${subjectTag}/motion/sub-${subjectTag}_physio.acq
#     scp -r ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1_all.csv xpsy1114@jalapeno.fmrib.ox.ac.uk:/home/fs0/xpsy1114/scratch/data/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1_all.csv
#     scp -r ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2_all.csv xpsy1114@jalapeno.fmrib.ox.ac.uk:/home/fs0/xpsy1114/scratch/data/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2_all.csv
#     scp -r ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1.csv xpsy1114@jalapeno.fmrib.ox.ac.uk:/home/fs0/xpsy1114/scratch/data/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1.csv
#     scp -r ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2.csv xpsy1114@jalapeno.fmrib.ox.ac.uk:/home/fs0/xpsy1114/scratch/data/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2.csv
# done


# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do
for subjectTag in 25 26; do
    mkdir ${scratchDir}/for_server/sub-${subjectTag}
    mkdir ${scratchDir}/for_server/sub-${subjectTag}/beh
    mkdir ${scratchDir}/for_server/sub-${subjectTag}/motion
    # mkdir ${scratchDir}/for_server/sub-${subjectTag}/func
    cp ${scratchDir}/pilot/sub-${subjectTag}/motion/sub-${subjectTag}_physio.txt ${scratchDir}/for_server/sub-${subjectTag}/motion/sub-${subjectTag}_physio.txt
    # cp ${scratchDir}/pilot/sub-${subjectTag}/motion/sub-${subjectTag}_physio.acq
    cp ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1_all.csv ${scratchDir}/for_server/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1_all.csv
    cp ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2_all.csv ${scratchDir}/for_server/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2_all.csv
    cp ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1.csv ${scratchDir}/for_server/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1.csv
    cp ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2.csv ${scratchDir}/for_server/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2.csv
    # cp -r ${scratchDir}/derivatives/sub-${subjectTag}/func/EVs_06_pt01_press_and_loc ${scratchDir}/for_server/sub-${subjectTag}/func/EVs_06_pt01_press_and_loc 
    # cp -r ${scratchDir}/derivatives/sub-${subjectTag}/func/EVs_06_pt02_press_and_loc ${scratchDir}/for_server/sub-${subjectTag}/func/EVs_06_pt02_press_and_loc 
    # cp -r ${scratchDir}/derivatives/sub-${subjectTag}/beh/RDMs_04_glmbase_06 ${scratchDir}/for_server/sub-${subjectTag}/beh/RDMs_04_glmbase_06
    # cp ${scratchDir}/derivatives/sub-${subjectTag}/func/sub-${subjectTag}_my_RDM_GLM_01_06.fsf ${scratchDir}/for_server/sub-${subjectTag}/func/sub-${subjectTag}_my_RDM_GLM_01_06.fsf
    # cp ${scratchDir}/derivatives/sub-${subjectTag}/func/sub-${subjectTag}_my_RDM_GLM_02_06.fsf ${scratchDir}/for_server/sub-${subjectTag}/func/sub-${subjectTag}_my_RDM_GLM_02_06.fsf
done