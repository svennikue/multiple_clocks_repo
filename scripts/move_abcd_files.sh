#!/bin/bash

# Set source and destination directories
SRC_DIR="/ceph/behrens/svenja/human_ABCD_ephys/all_box_data/ABCD_LFP"
DEST_DIR="/ceph/behrens/svenja/human_ABCD_ephys"

echo "Starting file move process..."

# --- HANDLE BAYLOR FOLDERS ---
echo "Processing Baylor (BCM) folders..."
find "$SRC_DIR" -type d -name "*BCM*" | while read -r folder; do
    base=$(basename "$folder")
    num=$(echo "$base" | grep -o '[0-9]\+' | head -n 1)
    newbase=$(printf "s%02d" "$num") 

    if [ -z "$newbase" ]; then
        echo "Skipping $folder — no sXX basename found."
        continue
    fi

    # Create target directories
    mkdir -p "$DEST_DIR/$newbase/LFP"
    mkdir -p "$DEST_DIR/$newbase/anat"
    
    # move .ns3 files
    find "$folder" -maxdepth 1 -type f -name "*.ns3" -exec mv {} "$DEST_DIR/$newbase/LFP/" \;

    # move .nii files
    find "$folder" -maxdepth 1 -type f -name "*.nii" -exec mv {} "$DEST_DIR/$newbase/anat/" \;

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
    num=$(echo "$base" | grep -o '[0-9]\+' | head -n 1)
    newbase=$(printf "s%02d" "$num") 

    if [ -z "$newbase" ]; then
        echo "Skipping $folder — no sXX basename found."
        continue
    fi

    # Create target directories
    mkdir -p "$DEST_DIR/$newbase/LFP"
    mkdir -p "$DEST_DIR/$newbase/LFP/anat"

    # move .ns2 files
    find "$folder" -maxdepth 1 -type f -name "*.ns2" -exec mv {} "$DEST_DIR/$newbase/LFP/" \;

    # move entire 'registered' subfolder
    if [ -d "$folder/registered" ]; then
        mv -r "$folder/registered" "$DEST_DIR/$newbase/LFP/anat/"
    fi

    echo "Finished processing $folder → $newbase"
done

echo "All done."
