#!/bin/bash

# Set source and destination directories
SRC_DIR="/ceph/behrens/svenja/human_ABCD_ephys/all_box_data/ABCD_ephys"
DEST_DIR="/ceph/behrens/svenja/human_ABCD_ephys"

echo "Starting file move process..."

# --- HANDLE BAYLOR FOLDERS ---
echo "Processing Baylor (BCM) folders..."
find "$SRC_DIR" -type d -name "*BCM*" | while read -r folder; do
    base=$(basename "$folder")
    newbase=$(echo "$base" | grep -o 's[0-9]\+')
    
    if [ -z "$newbase" ]; then
        echo "Skipping $folder — no sXX basename found."
        continue
    fi

    # Create target directories
    mkdir -p "$DEST_DIR/$newbase/LFP"
    mkdir -p "$DEST_DIR/$newbase/LFP/anat"
    
    # Copy .ns3 files
    find "$folder" -maxdepth 1 -type f -name "*.ns3" -exec cp {} "$DEST_DIR/$newbase/LFP/" \;

    # Copy .nii files
    find "$folder" -maxdepth 1 -type f -name "*.nii" -exec cp {} "$DEST_DIR/$newbase/LFP/anat/" \;

    # Copy *electrodes* files
    find "$folder" -maxdepth 1 -type f -name "*electrodes*" | while read -r elec_file; do
        mkdir -p "$DEST_DIR/$newbase/electrodes"
        mv "$elec_file" "$DEST_DIR/$newbase/electrodes/"
    done

    echo "Finished processing $folder → $newbase"
done

# --- HANDLE UT FOLDERS ---
echo "Processing UT folders..."
find "$SRC_DIR" -type d -name "*UT*" | while read -r folder; do
    base=$(basename "$folder")
    newbase=$(echo "$base" | grep -o 's[0-9]\+')

    if [ -z "$newbase" ]; then
        echo "Skipping $folder — no sXX basename found."
        continue
    fi

    # Create target directories
    mkdir -p "$DEST_DIR/$newbase/LFP"
    mkdir -p "$DEST_DIR/$newbase/LFP/anat"

    # Copy .ns2 files
    find "$folder" -maxdepth 1 -type f -name "*.ns2" -exec cp {} "$DEST_DIR/$newbase/LFP/" \;

    # Copy entire 'registered' subfolder
    if [ -d "$folder/registered" ]; then
        mv -r "$folder/registered" "$DEST_DIR/$newbase/LFP/anat/"
    fi

    echo "Finished processing $folder → $newbase"
done

echo "All done."
