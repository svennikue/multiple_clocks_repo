# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"

glm_version="all-paths-fixed_stickrews_split-buttons"
RSA_version="quarters_DSR_controls"

fslDir="/opt/fmrib/fsl"
export fslDir=~/scratch/fsl
export PATH=$fslDir/share/fsl/bin/:$PATH
source $fslDir/etc/fslconf/fsl.sh
module load fsl


groupDir=${scratchDir}/derivatives/group/group_RSA_${RSA_version}_glmbase_${glm_version}
echo this is group dir $groupDir
if [ ! -d $groupDir ]; then
    mkdir $groupDir
fi


#example_resultDir=${scratchDir}/derivatives/sub-34/func/RSA_${RSA_version}*_${glm_version}/standard-space-smooth
candidates=( ${scratchDir}/derivatives/sub-34/func/RSA_${RSA_version}_*_glmbase_${glm_version}/standard-space-smooth)
echo these are candidates $candidates
    if ((${#candidates[@]})); then
        # If multiple dates exist, pick the newest by mtime
        IFS=$'\n' candidates=($(ls -1dt "${candidates[@]}"))
        example_resultDir="${candidates[0]}"
    else
        example_resultDir=${scratchDir}/derivatives/sub-34/func/RSA_${RSA_version}_*_glmbase_${glm_version}/standard-space-smooth
    fi
# later include something like this:
# fslmaths /home/fs0/xpsy1114/scratch/data/derivatives/sub-08/func/RSA_03-1_glmbase_03-4_smooth5/standard-space/clocks_t_val_std.nii.gz -mul /home/fs0/xpsy1114/scratch/data/masks/brain_bin.nii.gz /home/fs0/xpsy1114/scratch/data/derivatives/sub-08/func/RSA_03-1_glmbase_03-4_smooth5/results-standard-space/clocks_t_val_std_masked.nii.gz


if [ ! -d $example_resultDir ]; then
    example_resultDir=${scratchDir}/derivatives/sub-34/func/RSA_${RSA_version}*_${glm_version}/standard-space
    list_of_std_beta_files=$(find $example_resultDir -name "avg*beta_std*.nii.gz" -type f)
else
    list_of_std_beta_files=$(find $example_resultDir -name "*beta_std.nii.gz" -type f)
fi

echo this is example resultDir $example_resultDir

# first, mask all of these files such that they fit the standard mask well.
for file in $list_of_std_beta_files; do
    filename=$(basename "$file")
    echo $filename
    for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35; do
    
    #for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35; do
        # resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_*_glmbase_${glm_version}_smooth5/standard-space
        # if [ ! -d $resultDir ]; then
        #     resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_${glm_version}_smooth5/standard-space
        # fi
            # Expand the dated (glmbase) pattern to real dirs (could be 0, 1 or many)
        candidates=( ${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}*_glmbase_${glm_version}/standard-space-smooth )

        if ((${#candidates[@]})); then
            # If multiple dates exist, pick the newest by mtime
            IFS=$'\n' candidates=($(ls -1dt "${candidates[@]}"))
            resultDir="${candidates[0]}"
        else
            resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_${glm_version}/standard-space-smooth
        fi

        [[ -d "$resultDir" ]] || { echo "skip $subjectTag (no results dir)"; continue; }
        [[ -f "$resultDir/$filename" ]] || { echo "skip $subjectTag (missing $filename)"; continue; }

        echo "now masking subject $subjectTag and $filename"

        masked_output=${resultDir}/masked_$filename 
        # skip if already transformed (do not submit)
        if [ -e "$masked_output" ]; then
            echo "Skipping $(basename "$file"): output already exists -> $(basename "$masked_output")"
            continue
        fi
        
        # TODO: make a GREY MATTER MASK AND REPLACE THIS FILE WITH THE GREY MATTER MASK!
        fslmaths ${resultDir}/$filename -mas $scratchDir/masks/brain_bin.nii.gz ${masked_output}
    done
done


# Then, for each of these files
for file in $list_of_std_beta_files; do
    # if file is more new than 3 days: 3 days×24 hours * 60 minutes * 60 seconds =259,200 seconds
    # 14 days: 1209600
    if [[ ! "$(( $(date +"%s") - $(stat -c "%Y" "$file") ))" -gt "1209600" ]]; then
        # Extract the filename
        filename=$(basename "$file")
        echo now moving and merging $filename
        out_merged=${groupDir}/masked_${filename}
        # skip if already transformed (do not submit)
        if [ -e "$out_merged" ]; then
            echo "Skipping $(basename "$file"): output already exists -> $(basename "$out_merged")"
            continue
        fi

        # no 21 or 29 
        for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35; do
            # # for every result file
            # resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_*_glmbase_${glm_version}_smooth5/standard-space
            # if [ ! -d $resultDir ]; then
            #     resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_${glm_version}_smooth5/standard-space
            # fi
            candidates=( ${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_*_glmbase_${glm_version}/standard-space-smooth )

            if ((${#candidates[@]})); then
                # If multiple dates exist, pick the newest by mtime
                IFS=$'\n' candidates=($(ls -1dt "${candidates[@]}"))
                resultDir="${candidates[0]}"
            else
                resultDir=${scratchDir}/derivatives/sub-${subjectTag}/func/RSA_${RSA_version}_${glm_version}/standard-space-smooth
            fi

            echo now for subject $subjectTag and $resultDir
            if [ ! -f ${groupDir}/masked_${filename} ]; then
                cp ${resultDir}/masked_$filename ${groupDir}/masked_${filename}
            else 
                fslmerge -t ${groupDir}/masked_${filename} ${resultDir}/masked_$filename ${groupDir}/masked_${filename}
            fi
        done
    fi
done
    
gunzip $( ls ${groupDir}/*.nii.gz )

echo done!
