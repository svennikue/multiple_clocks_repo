# Set scratch directory for execution on server
laptopDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
dataDirServer="/ceph/behrens/svenja/human_ABCD_ephys"

for i in $(seq -w 10 50); do
  subjects+=("$i")
done

for sub in "${subjects[@]}"; do
    source_path="$laptopDir/s$sub"
    echo "source dir is $source_path"
    if [ -d "$source_path" ]; then # Check if path exists and is a regular file
        # Copy .nii files to the anatomy folder
        for nii_file in "$source_path"/*.nii; do
            if [ -f "$nii_file" ]; then # Check if file exists and is a regular file
                echo "file is "$nii_file""
                rsync -avr --info=progress2 ${nii_file} skuchenhoff@ssh.swc.ucl.ac.uk:${dataDirServer}/s${sub}/anat/
            fi
        done
        #for cell_file in "$source_path"/*.csv; do
        #        echo "file is "$cell_file""
        #    if [ -f "$cell_file" ]; then # Check if file exists and is a regular file
        #        rsync -avr --info=progress2 ${cell_file} skuchenhoff@ssh.swc.ucl.ac.uk:${dataDirServer}/s${sub}/cells/
        #    fi
        #done
        # Copy .ns2 and .ns3 files to the LFP folder
        for ns_file in "$source_path"/*.{ns2,ns3}; do
            if [ -f "$ns_file" ]; then # Check if file exists and is a regular file
                echo "file is "$ns_file""
                rsync -avr --info=progress2 ${ns_file} skuchenhoff@ssh.swc.ucl.ac.uk:${dataDirServer}/s${sub}/LFP/
            fi
        done
        echo "Files for sub ${sub} have been copied successfully."
    fi
done