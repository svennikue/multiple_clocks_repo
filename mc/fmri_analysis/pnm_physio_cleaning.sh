# this file prepares the physiology file for processing with pnm and starts pnm level 1 and 2
# careful! run within spyder virtual env!


# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"
#fslDir="/opt/fmrib/fsl"
export fslDir=~/scratch/fsl
export PATH=$fslDir/share/fsl/bin/:$PATH
source $fslDir/etc/fslconf/fsl.sh

# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  analysisDir="/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
  fslDir="/Users/xpsy1114/fsl"
fi
echo Scratch directory is $scratchDir


# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do
# for subjectTag in 03 04 07 08 09; do
#for subjectTag in 03 04 07 08 09 10 11 12 14 15 16 17 20 21 23 24; do
# 04, 06, 30, 31 and 34 are still missing!!
# and for 05, I only have half of the physio file (only task half 2)
for subjectTag in 01 02 03 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 32 33; do
    # Command line argument 1/1: subject tag
    echo Subject tag for this subject: $subjectTag
    
    # Construct directory for derived data
    derivDir=$scratchDir/derivatives/sub-${subjectTag}
    mkdir $derivDir/motion
    rawPhysDir=${scratchDir}/pilot/sub-${subjectTag}/motion
    old_physio=${rawPhysDir}/sub-${subjectTag}_physio.txt
    new_physio=${rawPhysDir}/sub-${subjectTag}_physio_no_header.txt
    # Find the line number where the header ends (the line containing 'samples')
    header_end=$(grep -n 'samples' "$old_physio" | tail -1 | cut -d: -f1)
    # Write the data after the header to the new file
    tail -n +$((header_end + 1)) "$old_physio" > "$new_physio"
    
    # then call short python script that divides both parts into 2 
    python ${analysisDir}/multiple_clocks_repo/scripts/physiology_server.py ${subjectTag}

    for task_half in 1 2; do
        
         # this is the new file per task half from the python scrip, only continue once it exists
         halved_physio_file=${derivDir}/motion/sub-${subjectTag}_physio_0${task_half}.txt

        # Loop to check if the file exists
        while true; do
            if [ -f "$halved_physio_file" ]; then
                echo now running the next stage: pnm stage one
                # make sure formatting is fine
                fslFixText $halved_physio_file ${derivDir}/motion/sub-${subjectTag}_physio_0${task_half}.txt
                # then create the regressors. You can additionally define more peaks, but it's usually not necessary.
                pnm_stage1 -i ${derivDir}/motion/sub-${subjectTag}_physio_0${task_half}.txt -o ${derivDir}/motion/sub-${subjectTag}_0${task_half} -s 200 --tr=1.078 --smoothcard=0.1 --smoothresp=0.1 --resp=2 --cardiac=3 --trigger=4 --rvt
                break # Exit the loop once the file is found and function is called
            else
                echo "Waiting for file: $halved_physio_file"
                sleep 10 # Wait for 5 seconds before checking again
            fi
        done


    #     # pnm_stage2 is only called if there are any additional peaks selected. so don't bother about this
    #     # wait until this file is genereated bc it means pnm_stage1 is done
    pnm_results=${derivDir}/motion/sub-${subjectTag}_0${task_half}_pnm_stage2
    processed_bold=${derivDir}/func/preproc_clean_0${task_half}.feat/filtered_func_data.nii.gz
    #     #obase=${derivDir}/motion/
    #     #export obase

        # Loop to check if the file exists
        while true; do
            if [ -f "$pnm_results" ]; then
                echo now running making the evs matrix 
                # first remove all evs that might already be in there
                rm ${derivDir}/motion/sub-${subjectTag}_0${task_half}_pnmevs*ev0*.nii.gz
                #sliceDir=${scratchDir}/pilot/sub-${subjectTag}/func/sub-${subjectTag}_slice_order_${task_half}.txt
                sliceDir=${scratchDir}/pilot/sub-${subjectTag}/func/sub-${subjectTag}_slice_timings_${task_half}.txt
                pnm_evs --tr=1.078 -i ${processed_bold} -o ${derivDir}/motion/sub-${subjectTag}_0${task_half}_pnmevs -r ${derivDir}/motion/sub-${subjectTag}_0${task_half}_resp.txt -c ${derivDir}/motion/sub-${subjectTag}_0${task_half}_card.txt --or=4 --oc=4 --multr=2 --multc=2 --slicetiming="${sliceDir}" -v
                #pnm_evs --tr=1.078 -i ${processed_bold} -o ${derivDir}/motion/sub-${subjectTag}_0${task_half}_pnmevs -r ${derivDir}/motion/sub-${subjectTag}_0${task_half}_resp.txt -c ${derivDir}/motion/sub-${subjectTag}_0${task_half}_card.txt --or=4 --oc=4 --multr=2 --multc=2 -v
                
                # and create a list of the voxelwise confound EVs.
                ls -1 "${derivDir}/motion"/sub-${subjectTag}_0${task_half}_pnmevs*ev0*.nii.gz > ${derivDir}/motion/sub-${subjectTag}_${task_half}_evlist.txt
                #ls -1 `imglob -extensions ${obase}ev0*` > ${derivDir}/motion/${sub}_${task_half}_evlist.txt
                echo done with subject ${subjectTag} and task half 0${task_half} !
                break # Exit the loop once the file is found and function is called
            else
                echo "Waiting for file: $pnm_results"
                sleep 10 # Wait for 5 seconds before checking again
            fi
        done

    done
done

