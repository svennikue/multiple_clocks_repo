#!/bin/bash

SRC_DIR="/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"

for i in $(seq -w 1 60); do
    folder="$SRC_DIR/s$i"
    
    if [ ! -d "$folder" ]; then
        echo "Skipping missing folder: s$i"
        continue
    fi

    echo "Processing folder: s$i"

    # Create subdirectories if needed
    mkdir -p "$folder/LFP"
    mkdir -p "$folder/anat"
    mkdir -p "$folder/electrodes"

    # Move .ns2 and .ns3 files to LFP
    find "$folder" -maxdepth 1 -type f \( -name "*.ns2" -o -name "*.ns3" \) -exec mv {} "$folder/LFP/" \;

    # Move .nii files to anat
    find "$folder" -maxdepth 1 -type f -name "*.nii" -exec mv {} "$folder/anat/" \;

    # Move files containing 'electrod' to electrodes
    find "$folder" -maxdepth 1 -type f -iname "*electrod*" -exec mv {} "$folder/electrodes/" \;
done
