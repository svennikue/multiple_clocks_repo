#!/bin/sh
# Run subject-level GLM on preprocessed functional data
# new try 06.12.2023 with the correct regressors.
# bash subject_GLM_RDM_new.sh 01 06

# Command line argument 1/2: subject tag
subjects=$1
# Command line argument 2/2: GLM version
version=$2
echo Subject tag for this subject: $subjectTag and for GLM no $version

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
toolboxDir="/home/fs0/xpsy1114/scratch/analysis"
homeDir="/home/fs0/xpsy1114"
analysisDir="${scratchDir}/analysis"


task_halves=("01" "02")
# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  toolboxDir="/Users/xpsy1114/Documents/toolboxes"
  homeDir="/Users/xpsy1114/Documents/projects/multiple_clocks"
  analysisDir="${homeDir}/multiple_clocks_repo/mc/fmri_analysis"
fi

# Show what ended up being the home directory
echo Home directory is $homeDir
echo Scratch directory is $scratchDir


for subjectTag in "${subjects[@]}"; do
    # Command line argument 1/1: subject tag
    echo Subject tag for this subject: $subjectTag and for GLM no $version
    
    # Construct directory for derived data
    derivDir=$scratchDir/derivatives/sub-$subjectTag

    # Construct the folder for function analysis for the current subject
    funcDir=$scratchDir/derivatives/sub-$subjectTag/func

    # do this twice, once for pt1 and once for pt2

    # create a loop from here to the end (marked as well)
    for task_half in "${task_halves[@]}"; do
        glmDir="$funcDir/glm_${version}_pt${task_half}.feat" 
        nuisanceFile=$derivDir/func/nuisance_$task_half/combined.txt
        # check if the EV dir exists, if not output an error.
        EV_dir="$funcDir/EVs_${version}_pt${task_half}"

        if [ -d "$EV_dir" ]; then
          echo "EV Folder exists, all good!"
        else
          echo "ERROR: EV Folder does not exist!"
        fi

        # Step 1: Collect file paths
        file_list=($(find "$EV_dir" -type f -name "ev_*.txt" | sort))

        # Check the number of files
        num_files=${#file_list[@]}
        echo "Found ${num_files} EVs"

        # Step 2: Order files alphabetically
        sorted_file_list=($(echo "${file_list[@]}" | tr ' ' '\n' | sort))

        # Print the list of file paths
        echo "These will be the EVs in this order:"
        for file_path in "${sorted_file_list[@]}"; do
            echo "$file_path"
        done

        # Step 3: Replace lines in the .fsf file
        original_fsf_file=${analysisDir}/templates/my_RDM_GLM_v2.fsf
        new_fsf_file=$funcDir/sub-${subjectTag}_design_glm_${version}_pt${task_half}.fsf

        cp "$original_fsf_file" "$new_fsf_file"


        for ((i=1; i<=$num_files; i++)); do
            old_line="set fmri(custom$i) \".*\""
            new_line="set fmri(custom$i) \"${sorted_file_list[i-1]}\""
            
            # Use sed to replace the line in the new .fsf file
            #sed -i "s|$old_line|$new_line|" "$new_fsf_file"
            sed "s|$old_line|$new_line|g" $new_fsf_file > $new_fsf_file
        done

        echo "New .fsf file created: $new_fsf_file"

        
    done



done